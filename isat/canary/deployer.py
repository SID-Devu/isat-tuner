"""Canary deployment and traffic splitting between model versions.

Gradually shifts inference traffic from an old model to a new model
while monitoring error rate and latency. Automatically rolls back
if the new model exceeds error/latency thresholds.

Used by: Netflix, Google, Meta for safe model rollouts.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.canary")


@dataclass
class CanaryMetrics:
    model_tag: str
    requests: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        return self.errors / max(self.requests, 1)

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0

    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99)) if self.latencies_ms else 0


@dataclass
class CanaryResult:
    baseline_tag: str
    canary_tag: str
    baseline_metrics: CanaryMetrics
    canary_metrics: CanaryMetrics
    final_canary_pct: float
    rolled_back: bool
    rollback_reason: str = ""
    phases_completed: int = 0
    total_requests: int = 0

    def summary(self) -> str:
        lines = [
            f"  Baseline    : {self.baseline_tag}",
            f"  Canary      : {self.canary_tag}",
            f"  Final split : {100 - self.final_canary_pct:.0f}% / {self.final_canary_pct:.0f}%",
            f"  Phases done : {self.phases_completed}",
            f"  Requests    : {self.total_requests}",
            f"",
            f"  {'Metric':<25} {'Baseline':>12} {'Canary':>12} {'Delta':>12}",
            f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}",
        ]
        bm, cm = self.baseline_metrics, self.canary_metrics
        mean_delta = cm.mean_latency_ms - bm.mean_latency_ms
        lines.append(f"  {'Requests':<25} {bm.requests:>12} {cm.requests:>12}")
        lines.append(f"  {'Error rate':<25} {bm.error_rate:>11.2%} {cm.error_rate:>11.2%}")
        lines.append(f"  {'Mean latency ms':<25} {bm.mean_latency_ms:>12.2f} {cm.mean_latency_ms:>12.2f} {mean_delta:>+11.2f}")
        lines.append(f"  {'P99 latency ms':<25} {bm.p99_latency_ms:>12.2f} {cm.p99_latency_ms:>12.2f}")

        if self.rolled_back:
            lines.append(f"\n  ROLLED BACK: {self.rollback_reason}")
        else:
            lines.append(f"\n  Canary PASSED -- safe to promote")
        return "\n".join(lines)


class CanaryDeployer:
    """Run canary deployment between two models."""

    def __init__(
        self,
        baseline_path: str,
        canary_path: str,
        provider: str = "CPUExecutionProvider",
        phases: list[float] | None = None,
        requests_per_phase: int = 50,
        max_error_rate: float = 0.05,
        max_latency_increase_pct: float = 20.0,
    ):
        self.baseline_path = baseline_path
        self.canary_path = canary_path
        self.provider = provider
        self.phases = phases or [5, 10, 25, 50, 75, 100]
        self.requests_per_phase = requests_per_phase
        self.max_error_rate = max_error_rate
        self.max_latency_increase_pct = max_latency_increase_pct

    def deploy(self) -> CanaryResult:
        import onnxruntime as ort

        baseline_session = ort.InferenceSession(
            self.baseline_path, providers=ort_providers(self.provider),
        )
        canary_session = ort.InferenceSession(
            self.canary_path, providers=ort_providers(self.provider),
        )
        baseline_feed = _build_feed(baseline_session)
        canary_feed = _build_feed(canary_session)

        for _ in range(3):
            baseline_session.run(None, baseline_feed)
            canary_session.run(None, canary_feed)

        bm = CanaryMetrics(model_tag="baseline")
        cm = CanaryMetrics(model_tag="canary")
        rolled_back = False
        rollback_reason = ""
        phases_done = 0
        current_pct = 0.0

        for canary_pct in self.phases:
            current_pct = canary_pct
            for _ in range(self.requests_per_phase):
                use_canary = random.random() * 100 < canary_pct
                session = canary_session if use_canary else baseline_session
                feed = canary_feed if use_canary else baseline_feed
                metrics = cm if use_canary else bm

                try:
                    t0 = time.perf_counter()
                    session.run(None, feed)
                    lat = (time.perf_counter() - t0) * 1000
                    metrics.requests += 1
                    metrics.latencies_ms.append(lat)
                except Exception:
                    metrics.requests += 1
                    metrics.errors += 1

            phases_done += 1

            if cm.error_rate > self.max_error_rate:
                rolled_back = True
                rollback_reason = (
                    f"Canary error rate {cm.error_rate:.1%} "
                    f"exceeds threshold {self.max_error_rate:.1%}"
                )
                break

            if bm.mean_latency_ms > 0:
                increase = (cm.mean_latency_ms - bm.mean_latency_ms) / bm.mean_latency_ms * 100
                if increase > self.max_latency_increase_pct:
                    rolled_back = True
                    rollback_reason = (
                        f"Canary latency increase {increase:.1f}% "
                        f"exceeds threshold {self.max_latency_increase_pct:.0f}%"
                    )
                    break

        return CanaryResult(
            baseline_tag=self.baseline_path,
            canary_tag=self.canary_path,
            baseline_metrics=bm,
            canary_metrics=cm,
            final_canary_pct=current_pct,
            rolled_back=rolled_back,
            rollback_reason=rollback_reason,
            phases_completed=phases_done,
            total_requests=bm.requests + cm.requests,
        )


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
