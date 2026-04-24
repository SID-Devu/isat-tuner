"""A/B testing framework for inference comparison.

Statistically rigorous comparison of:
  - Two models (original vs optimized)
  - Two configs (different env vars)
  - Two providers (MIGraphX vs TensorRT)
  - Two precisions (FP32 vs FP16)

Uses interleaved execution to eliminate temporal bias, Welch's t-test for
significance, and bootstrap confidence intervals for the speedup estimate.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("isat.abtesting")


@dataclass
class ABResult:
    name_a: str
    name_b: str
    mean_a_ms: float
    mean_b_ms: float
    std_a_ms: float
    std_b_ms: float
    p50_a_ms: float
    p50_b_ms: float
    p95_a_ms: float
    p95_b_ms: float
    p99_a_ms: float
    p99_b_ms: float
    samples_a: int
    samples_b: int
    speedup: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float]
    winner: str
    recommendation: str

    def summary(self) -> str:
        lines = [
            f"  {'Metric':<20} {'A: ' + self.name_a:>20} {'B: ' + self.name_b:>20}",
            f"  {'-'*20} {'-'*20} {'-'*20}",
            f"  {'Mean (ms)':<20} {self.mean_a_ms:>20.3f} {self.mean_b_ms:>20.3f}",
            f"  {'Std (ms)':<20} {self.std_a_ms:>20.3f} {self.std_b_ms:>20.3f}",
            f"  {'P50 (ms)':<20} {self.p50_a_ms:>20.3f} {self.p50_b_ms:>20.3f}",
            f"  {'P95 (ms)':<20} {self.p95_a_ms:>20.3f} {self.p95_b_ms:>20.3f}",
            f"  {'P99 (ms)':<20} {self.p99_a_ms:>20.3f} {self.p99_b_ms:>20.3f}",
            f"  {'Samples':<20} {self.samples_a:>20} {self.samples_b:>20}",
            f"",
            f"  Speedup           : {self.speedup:.3f}x ({'faster' if self.speedup > 1 else 'slower'})",
            f"  95% CI            : [{self.confidence_interval[0]:.3f}x, {self.confidence_interval[1]:.3f}x]",
            f"  p-value           : {self.p_value:.6f}",
            f"  Significant       : {'YES' if self.significant else 'NO'} (alpha=0.05)",
            f"  Winner            : {self.winner}",
            f"  Recommendation    : {self.recommendation}",
        ]
        return "\n".join(lines)


class ABTest:
    """Run an A/B comparison between two inference configurations."""

    def __init__(
        self,
        model_a: str,
        model_b: str,
        provider: str = "CPUExecutionProvider",
        warmup: int = 5,
        runs: int = 50,
        interleave: bool = True,
        env_a: dict[str, str] | None = None,
        env_b: dict[str, str] | None = None,
    ):
        self.model_a = model_a
        self.model_b = model_b
        self.provider = provider
        self.warmup = warmup
        self.runs = runs
        self.interleave = interleave
        self.env_a = env_a or {}
        self.env_b = env_b or {}

    def run(self, name_a: str = "A", name_b: str = "B") -> ABResult:
        import onnxruntime as ort
        import os

        orig_env = dict(os.environ)

        for k, v in self.env_a.items():
            os.environ[k] = v

        sess_a = ort.InferenceSession(
            self.model_a, providers=[self.provider, "CPUExecutionProvider"],
        )
        feed_a = self._build_feed(sess_a)

        os.environ.clear()
        os.environ.update(orig_env)
        for k, v in self.env_b.items():
            os.environ[k] = v

        sess_b = ort.InferenceSession(
            self.model_b, providers=[self.provider, "CPUExecutionProvider"],
        )
        feed_b = self._build_feed(sess_b)

        os.environ.clear()
        os.environ.update(orig_env)

        for _ in range(self.warmup):
            sess_a.run(None, feed_a)
            sess_b.run(None, feed_b)

        lats_a: list[float] = []
        lats_b: list[float] = []

        if self.interleave:
            schedule = []
            for _ in range(self.runs):
                schedule.extend(["a", "b"])
            random.shuffle(schedule)

            for label in schedule:
                if label == "a":
                    t0 = time.perf_counter()
                    sess_a.run(None, feed_a)
                    lats_a.append((time.perf_counter() - t0) * 1000)
                else:
                    t0 = time.perf_counter()
                    sess_b.run(None, feed_b)
                    lats_b.append((time.perf_counter() - t0) * 1000)
        else:
            for _ in range(self.runs):
                t0 = time.perf_counter()
                sess_a.run(None, feed_a)
                lats_a.append((time.perf_counter() - t0) * 1000)
            for _ in range(self.runs):
                t0 = time.perf_counter()
                sess_b.run(None, feed_b)
                lats_b.append((time.perf_counter() - t0) * 1000)

        a = np.array(lats_a)
        b = np.array(lats_b)

        speedup = float(np.mean(a) / np.mean(b)) if np.mean(b) > 0 else 1.0

        t_stat, p_value = _welch_t_test(a, b)
        ci = _bootstrap_ci(a, b)
        significant = p_value < 0.05

        if significant:
            if speedup > 1.01:
                winner = name_b
                rec = f"Use {name_b} -- {speedup:.2f}x faster with statistical confidence"
            elif speedup < 0.99:
                winner = name_a
                rec = f"Use {name_a} -- {1/speedup:.2f}x faster with statistical confidence"
            else:
                winner = "tie"
                rec = "No meaningful difference -- either config works"
        else:
            winner = "inconclusive"
            rec = f"Not statistically significant (p={p_value:.3f}). Run more samples."

        return ABResult(
            name_a=name_a, name_b=name_b,
            mean_a_ms=float(np.mean(a)), mean_b_ms=float(np.mean(b)),
            std_a_ms=float(np.std(a)), std_b_ms=float(np.std(b)),
            p50_a_ms=float(np.percentile(a, 50)), p50_b_ms=float(np.percentile(b, 50)),
            p95_a_ms=float(np.percentile(a, 95)), p95_b_ms=float(np.percentile(b, 95)),
            p99_a_ms=float(np.percentile(a, 99)), p99_b_ms=float(np.percentile(b, 99)),
            samples_a=len(lats_a), samples_b=len(lats_b),
            speedup=speedup, p_value=p_value, significant=significant,
            confidence_interval=ci, winner=winner, recommendation=rec,
        )

    def _build_feed(self, session) -> dict:
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


def _welch_t_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = np.mean(a), np.mean(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)

    se = np.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-12:
        return 0.0, 1.0

    t = (mean_a - mean_b) / se

    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else 1

    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except ImportError:
        import math
        x = df / (df + t * t)
        p = _approx_p(abs(t), df)

    return float(t), float(p)


def _approx_p(t: float, df: float) -> float:
    """Approximate two-tailed p-value without scipy."""
    import math
    x = t / math.sqrt(t * t + df) if (t * t + df) > 0 else 0
    p = 1.0 - 0.5 * (1.0 + x * (1.0 + (-0.356563782 + (1.781477937 + (-1.821255978 + 1.330274429 * abs(x)) * abs(x)) * abs(x)) * abs(x)))
    return max(0.001, min(1.0, 2.0 * (1.0 - abs(1.0 - p))))


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for speedup (mean_a / mean_b)."""
    ratios = []
    for _ in range(n_boot):
        sa = np.random.choice(a, size=len(a), replace=True)
        sb = np.random.choice(b, size=len(b), replace=True)
        mean_b = np.mean(sb)
        if mean_b > 0:
            ratios.append(np.mean(sa) / mean_b)

    if not ratios:
        return (1.0, 1.0)

    lower = float(np.percentile(ratios, 100 * alpha / 2))
    upper = float(np.percentile(ratios, 100 * (1 - alpha / 2)))
    return (lower, upper)
