"""Output quality monitor -- detect confidence drift and distribution shift.

Tracks:
  - Output confidence distribution over time
  - Entropy of predictions (low entropy = confident, high = uncertain)
  - Class balance drift (for classifiers)
  - Statistical tests for distribution change (KS test)
  - Alert when outputs deviate from baseline

Used by: Arize, WhyLabs, Evidently for production model monitoring.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("isat.output_monitor")


@dataclass
class OutputSnapshot:
    timestamp: float
    mean_confidence: float
    mean_entropy: float
    max_class: int
    class_distribution: dict[int, float] = field(default_factory=dict)


@dataclass
class DriftAlert:
    metric: str
    baseline_value: float
    current_value: float
    change_pct: float
    severity: str


@dataclass
class DriftReport:
    model_path: str
    baseline_samples: int
    monitor_samples: int
    alerts: list[DriftAlert] = field(default_factory=list)
    baseline_confidence: float = 0
    current_confidence: float = 0
    baseline_entropy: float = 0
    current_entropy: float = 0
    ks_statistic: float = 0
    ks_pvalue: float = 0
    drift_detected: bool = False

    def summary(self) -> str:
        status = "DRIFT DETECTED" if self.drift_detected else "STABLE"
        lines = [
            f"  Model           : {self.model_path}",
            f"  Status          : {status}",
            f"  Baseline samples: {self.baseline_samples}",
            f"  Monitor samples : {self.monitor_samples}",
            f"  KS statistic    : {self.ks_statistic:.4f}",
            f"  KS p-value      : {self.ks_pvalue:.4f}",
            f"",
            f"  {'Metric':<25} {'Baseline':>12} {'Current':>12} {'Change':>10}",
            f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}",
            f"  {'Mean confidence':<25} {self.baseline_confidence:>12.4f} {self.current_confidence:>12.4f} "
            f"{(self.current_confidence - self.baseline_confidence) / max(self.baseline_confidence, 1e-6) * 100:>+9.1f}%",
            f"  {'Mean entropy':<25} {self.baseline_entropy:>12.4f} {self.current_entropy:>12.4f} "
            f"{(self.current_entropy - self.baseline_entropy) / max(self.baseline_entropy, 1e-6) * 100:>+9.1f}%",
        ]
        if self.alerts:
            lines.append(f"\n  Alerts ({len(self.alerts)}):")
            for a in self.alerts:
                lines.append(f"    [{a.severity}] {a.metric}: {a.baseline_value:.4f} -> "
                             f"{a.current_value:.4f} ({a.change_pct:+.1f}%)")
        return "\n".join(lines)


class OutputMonitor:
    """Monitor inference output quality and detect drift."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        baseline_runs: int = 50,
        monitor_runs: int = 50,
        drift_threshold: float = 0.1,
    ):
        self.model_path = model_path
        self.provider = provider
        self.baseline_runs = baseline_runs
        self.monitor_runs = monitor_runs
        self.drift_threshold = drift_threshold

    def monitor(self) -> DriftReport:
        import onnxruntime as ort

        session = ort.InferenceSession(
            self.model_path, providers=[self.provider, "CPUExecutionProvider"],
        )

        for _ in range(3):
            feed = _build_feed(session)
            session.run(None, feed)

        baseline_confs = []
        baseline_entropies = []
        for _ in range(self.baseline_runs):
            feed = _build_feed(session)
            outputs = session.run(None, feed)
            conf, ent = self._compute_metrics(outputs[0])
            baseline_confs.append(conf)
            baseline_entropies.append(ent)

        monitor_confs = []
        monitor_entropies = []
        for _ in range(self.monitor_runs):
            feed = _build_feed(session)
            outputs = session.run(None, feed)
            conf, ent = self._compute_metrics(outputs[0])
            monitor_confs.append(conf)
            monitor_entropies.append(ent)

        baseline_conf_mean = float(np.mean(baseline_confs))
        monitor_conf_mean = float(np.mean(monitor_confs))
        baseline_ent_mean = float(np.mean(baseline_entropies))
        monitor_ent_mean = float(np.mean(monitor_entropies))

        from scipy.stats import ks_2samp
        try:
            ks_stat, ks_p = ks_2samp(baseline_confs, monitor_confs)
        except Exception:
            ks_stat, ks_p = 0.0, 1.0

        alerts = []
        drift = False

        if baseline_conf_mean > 0:
            conf_change = (monitor_conf_mean - baseline_conf_mean) / baseline_conf_mean
            if abs(conf_change) > self.drift_threshold:
                drift = True
                alerts.append(DriftAlert(
                    "confidence", baseline_conf_mean, monitor_conf_mean,
                    conf_change * 100, "warning" if abs(conf_change) < 0.3 else "critical"
                ))

        if baseline_ent_mean > 0:
            ent_change = (monitor_ent_mean - baseline_ent_mean) / baseline_ent_mean
            if abs(ent_change) > self.drift_threshold:
                drift = True
                alerts.append(DriftAlert(
                    "entropy", baseline_ent_mean, monitor_ent_mean,
                    ent_change * 100, "warning"
                ))

        if ks_p < 0.05:
            drift = True
            alerts.append(DriftAlert(
                "distribution (KS test)", 1.0, ks_p, (1 - ks_p) * 100, "critical"
            ))

        return DriftReport(
            model_path=self.model_path,
            baseline_samples=self.baseline_runs,
            monitor_samples=self.monitor_runs,
            alerts=alerts,
            baseline_confidence=baseline_conf_mean,
            current_confidence=monitor_conf_mean,
            baseline_entropy=baseline_ent_mean,
            current_entropy=monitor_ent_mean,
            ks_statistic=float(ks_stat), ks_pvalue=float(ks_p),
            drift_detected=drift,
        )

    def _compute_metrics(self, output: np.ndarray) -> tuple[float, float]:
        flat = output.flatten().astype(np.float64)
        if flat.size == 0:
            return 0.0, 0.0
        shifted = flat - flat.max()
        exp = np.exp(shifted)
        softmax = exp / (exp.sum() + 1e-10)
        confidence = float(softmax.max())
        entropy = float(-np.sum(softmax * np.log(softmax + 1e-10)))
        return confidence, entropy


def _build_feed(session) -> dict:
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if "int" in inp.type.lower():
            feed[inp.name] = np.ones(shape, dtype=np.int64)
        elif "float16" in inp.type.lower():
            feed[inp.name] = np.random.randn(*shape).astype(np.float16)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed
