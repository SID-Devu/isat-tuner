"""Performance regression detector.

Tracks inference performance across model versions and configurations.
Detects regressions (latency increases, throughput drops) against baselines
with statistical significance testing.

Usage:
    detector = RegressionDetector("isat_results.db")
    result = detector.check("my_model", new_latencies, threshold_pct=5.0)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("isat.regression")


@dataclass
class RegressionBaseline:
    model_name: str
    config_label: str
    mean_ms: float
    p95_ms: float
    p99_ms: float
    throughput_fps: float
    timestamp: str
    run_count: int = 0


@dataclass
class RegressionCheck:
    model_name: str
    baseline: Optional[RegressionBaseline]
    current_mean_ms: float
    current_p95_ms: float
    current_p99_ms: float
    current_throughput_fps: float
    mean_delta_pct: float
    p95_delta_pct: float
    p99_delta_pct: float
    throughput_delta_pct: float
    threshold_pct: float
    regressed: bool
    regression_details: list[str] = field(default_factory=list)
    t_statistic: float = 0.0
    p_value: float = 1.0
    statistically_significant: bool = False

    def summary(self) -> str:
        lines = []
        if not self.baseline:
            lines.append("  No baseline found -- this is the first run")
            lines.append(f"  Current mean: {self.current_mean_ms:.2f} ms")
            lines.append(f"  Saving as new baseline")
            return "\n".join(lines)

        lines.append(f"  Model        : {self.model_name}")
        lines.append(f"  Threshold    : {self.threshold_pct:.1f}%")
        lines.append(f"  Baseline     : {self.baseline.mean_ms:.2f} ms (mean), "
                     f"{self.baseline.p95_ms:.2f} ms (P95)")
        lines.append(f"  Current      : {self.current_mean_ms:.2f} ms (mean), "
                     f"{self.current_p95_ms:.2f} ms (P95)")
        lines.append(f"")
        lines.append(f"  {'Metric':<20} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>10}")
        lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        metrics = [
            ("Mean latency", self.baseline.mean_ms, self.current_mean_ms, self.mean_delta_pct),
            ("P95 latency", self.baseline.p95_ms, self.current_p95_ms, self.p95_delta_pct),
            ("P99 latency", self.baseline.p99_ms, self.current_p99_ms, self.p99_delta_pct),
            ("Throughput", self.baseline.throughput_fps, self.current_throughput_fps, self.throughput_delta_pct),
        ]

        for name, base, curr, delta in metrics:
            if name == "Throughput":
                status = "REGRESSION" if delta < -self.threshold_pct else "OK"
            else:
                status = "REGRESSION" if delta > self.threshold_pct else "OK"
            lines.append(f"  {name:<20} {base:>10.2f} {curr:>10.2f} {delta:>+9.1f}% {status:>10}")

        if self.statistically_significant:
            lines.append(f"\n  Statistical test: t={self.t_statistic:.3f}, p={self.p_value:.4f} (significant)")
        else:
            lines.append(f"\n  Statistical test: t={self.t_statistic:.3f}, p={self.p_value:.4f} (not significant)")

        verdict = "REGRESSION DETECTED" if self.regressed else "NO REGRESSION"
        lines.append(f"\n  Verdict: {verdict}")

        if self.regression_details:
            for d in self.regression_details:
                lines.append(f"    - {d}")

        return "\n".join(lines)


@dataclass
class RegressionHistory:
    model_name: str
    entries: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if not self.entries:
            return "  No regression history found"
        lines = [
            f"  {'#':<4} {'Timestamp':<22} {'Mean ms':>10} {'P95 ms':>10} {'Delta':>10} {'Status':>12}",
            f"  {'-'*4} {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*12}",
        ]
        for i, e in enumerate(self.entries):
            lines.append(
                f"  {i+1:<4} {e['timestamp']:<22} {e['mean_ms']:>10.2f} "
                f"{e['p95_ms']:>10.2f} {e.get('delta_pct', 0):>+9.1f}% "
                f"{e.get('status', 'baseline'):>12}"
            )
        return "\n".join(lines)


class RegressionDetector:
    """Detect performance regressions against stored baselines."""

    def __init__(self, db_path: str = "isat_results.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        db = sqlite3.connect(self.db_path)
        db.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                config_label TEXT DEFAULT '',
                mean_ms REAL,
                p95_ms REAL,
                p99_ms REAL,
                throughput_fps REAL,
                latencies_json TEXT,
                run_count INTEGER DEFAULT 0,
                timestamp TEXT DEFAULT (datetime('now')),
                is_baseline INTEGER DEFAULT 0
            )
        """)
        db.commit()
        db.close()

    def get_baseline(self, model_name: str) -> Optional[RegressionBaseline]:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM baselines WHERE model_name=? AND is_baseline=1 ORDER BY id DESC LIMIT 1",
            (model_name,),
        ).fetchone()
        db.close()
        if not row:
            return None
        return RegressionBaseline(
            model_name=row["model_name"],
            config_label=row["config_label"],
            mean_ms=row["mean_ms"],
            p95_ms=row["p95_ms"],
            p99_ms=row["p99_ms"],
            throughput_fps=row["throughput_fps"] or 0,
            timestamp=row["timestamp"],
            run_count=row["run_count"],
        )

    def save_baseline(self, model_name: str, latencies: list[float],
                      config_label: str = "", is_baseline: bool = True):
        arr = np.array(latencies)
        db = sqlite3.connect(self.db_path)
        if is_baseline:
            db.execute("UPDATE baselines SET is_baseline=0 WHERE model_name=?", (model_name,))
        db.execute(
            "INSERT INTO baselines (model_name, config_label, mean_ms, p95_ms, p99_ms, "
            "throughput_fps, latencies_json, run_count, is_baseline) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                model_name, config_label,
                float(np.mean(arr)), float(np.percentile(arr, 95)),
                float(np.percentile(arr, 99)),
                1000.0 / float(np.mean(arr)) if np.mean(arr) > 0 else 0,
                json.dumps(latencies),
                len(latencies), int(is_baseline),
            ),
        )
        db.commit()
        db.close()

    def check(
        self,
        model_name: str,
        latencies: list[float],
        threshold_pct: float = 5.0,
        confidence: float = 0.95,
        config_label: str = "",
    ) -> RegressionCheck:
        arr = np.array(latencies)
        current_mean = float(np.mean(arr))
        current_p95 = float(np.percentile(arr, 95))
        current_p99 = float(np.percentile(arr, 99))
        current_throughput = 1000.0 / current_mean if current_mean > 0 else 0

        baseline = self.get_baseline(model_name)

        if not baseline:
            self.save_baseline(model_name, latencies, config_label, is_baseline=True)
            return RegressionCheck(
                model_name=model_name, baseline=None,
                current_mean_ms=current_mean, current_p95_ms=current_p95,
                current_p99_ms=current_p99, current_throughput_fps=current_throughput,
                mean_delta_pct=0, p95_delta_pct=0, p99_delta_pct=0,
                throughput_delta_pct=0, threshold_pct=threshold_pct, regressed=False,
            )

        mean_delta = _pct_change(baseline.mean_ms, current_mean)
        p95_delta = _pct_change(baseline.p95_ms, current_p95)
        p99_delta = _pct_change(baseline.p99_ms, current_p99)
        throughput_delta = _pct_change(baseline.throughput_fps, current_throughput) if baseline.throughput_fps else 0

        t_stat, p_val, sig = _welch_t_test(baseline, latencies, confidence, self.db_path)

        regressed = False
        details = []
        if mean_delta > threshold_pct and sig:
            regressed = True
            details.append(f"Mean latency increased by {mean_delta:.1f}% (>{threshold_pct}%)")
        if p95_delta > threshold_pct and sig:
            regressed = True
            details.append(f"P95 latency increased by {p95_delta:.1f}%")
        if throughput_delta < -threshold_pct and baseline.throughput_fps > 0:
            regressed = True
            details.append(f"Throughput decreased by {abs(throughput_delta):.1f}%")

        self.save_baseline(model_name, latencies, config_label, is_baseline=False)

        return RegressionCheck(
            model_name=model_name, baseline=baseline,
            current_mean_ms=current_mean, current_p95_ms=current_p95,
            current_p99_ms=current_p99, current_throughput_fps=current_throughput,
            mean_delta_pct=mean_delta, p95_delta_pct=p95_delta,
            p99_delta_pct=p99_delta, throughput_delta_pct=throughput_delta,
            threshold_pct=threshold_pct, regressed=regressed,
            regression_details=details,
            t_statistic=t_stat, p_value=p_val, statistically_significant=sig,
        )

    def history(self, model_name: str, limit: int = 20) -> RegressionHistory:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT * FROM baselines WHERE model_name=? ORDER BY id DESC LIMIT ?",
            (model_name, limit),
        ).fetchall()
        db.close()

        entries = []
        prev_mean = None
        for row in reversed(rows):
            delta = _pct_change(prev_mean, row["mean_ms"]) if prev_mean else 0
            status = "baseline" if row["is_baseline"] else ("regression" if delta > 5 else "ok")
            entries.append({
                "timestamp": row["timestamp"],
                "mean_ms": row["mean_ms"],
                "p95_ms": row["p95_ms"],
                "delta_pct": delta,
                "status": status,
            })
            prev_mean = row["mean_ms"]

        return RegressionHistory(model_name=model_name, entries=entries)

    def set_baseline(self, model_name: str, run_id: int):
        db = sqlite3.connect(self.db_path)
        db.execute("UPDATE baselines SET is_baseline=0 WHERE model_name=?", (model_name,))
        db.execute("UPDATE baselines SET is_baseline=1 WHERE id=?", (run_id,))
        db.commit()
        db.close()


def _pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0
    return (new - old) / old * 100


def _welch_t_test(
    baseline: RegressionBaseline,
    new_latencies: list[float],
    confidence: float,
    db_path: str,
):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT latencies_json FROM baselines WHERE model_name=? AND is_baseline=1 ORDER BY id DESC LIMIT 1",
        (baseline.model_name,),
    ).fetchone()
    db.close()

    if not row or not row["latencies_json"]:
        return 0.0, 1.0, False

    try:
        old_lats = json.loads(row["latencies_json"])
    except (json.JSONDecodeError, TypeError):
        return 0.0, 1.0, False

    a = np.array(old_lats)
    b = np.array(new_latencies)

    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0, False

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, 1.0, False

    t_stat = (mean_b - mean_a) / se

    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1

    try:
        from scipy.stats import t
        p_value = float(2 * t.sf(abs(t_stat), df))
    except ImportError:
        p_value = 0.05 if abs(t_stat) > 2.0 else 0.5

    alpha = 1 - confidence
    return float(t_stat), p_value, p_value < alpha
