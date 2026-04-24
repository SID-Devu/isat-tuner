"""Statistical significance testing for config comparisons.

Implements Welch's t-test to determine whether the performance
difference between two configurations is statistically significant,
not just measurement noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class WelchResult:
    t_statistic: float
    p_value: float
    df: float
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int
    is_significant: bool
    confidence_level: float
    speedup: float
    ci_lower: float
    ci_upper: float

    @property
    def summary(self) -> str:
        direction = "faster" if self.speedup > 1 else "slower"
        sig = "SIGNIFICANT" if self.is_significant else "NOT significant"
        return (
            f"Config B is {abs(self.speedup - 1):.1%} {direction} than A "
            f"(p={self.p_value:.4f}, {sig} at {self.confidence_level:.0%} confidence)"
        )


def compare_configs(
    latencies_a: list[float],
    latencies_b: list[float],
    confidence: float = 0.95,
) -> WelchResult:
    """Welch's t-test comparing two sets of latency measurements.

    Tests H0: mean_a == mean_b against H1: mean_a != mean_b.
    Does not assume equal variances (Welch's correction).
    """
    a = np.array(latencies_a, dtype=np.float64)
    b = np.array(latencies_b, dtype=np.float64)

    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a = float(np.var(a, ddof=1)) if n_a > 1 else 0.0
    var_b = float(np.var(b, ddof=1)) if n_b > 1 else 0.0
    std_a, std_b = math.sqrt(var_a), math.sqrt(var_b)

    se = math.sqrt(var_a / n_a + var_b / n_b) if (n_a > 0 and n_b > 0) else 1e-10

    if se < 1e-12:
        t_stat = 0.0
        df = max(n_a + n_b - 2, 1.0)
        p_value = 1.0
    else:
        t_stat = (mean_a - mean_b) / se

        num = (var_a / n_a + var_b / n_b) ** 2
        term_a = (var_a / n_a) ** 2 / max(n_a - 1, 1) if var_a > 1e-15 else 0.0
        term_b = (var_b / n_b) ** 2 / max(n_b - 1, 1) if var_b > 1e-15 else 0.0
        denom = term_a + term_b
        df = max(1.0, num / denom if denom > 1e-15 else float(n_a + n_b - 2))

        p_value = _two_tail_p(t_stat, df)
        if math.isnan(p_value) or math.isinf(p_value):
            p_value = 1.0

    alpha = 1 - confidence
    t_crit = _t_critical(df, alpha)
    diff = mean_a - mean_b
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se

    speedup = mean_a / mean_b if mean_b > 0 else 1.0

    return WelchResult(
        t_statistic=t_stat,
        p_value=p_value,
        df=df,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        n_a=n_a,
        n_b=n_b,
        is_significant=p_value < alpha,
        confidence_level=confidence,
        speedup=speedup,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )


def _two_tail_p(t: float, df: float) -> float:
    """Approximate two-tailed p-value for Student's t distribution."""
    try:
        from scipy.stats import t as t_dist
        return float(2 * t_dist.sf(abs(t), df))
    except ImportError:
        pass

    x = df / (df + t * t)
    p = _regularized_incomplete_beta(df / 2, 0.5, x)
    return min(max(p, 0.0), 1.0)


def _t_critical(df: float, alpha: float) -> float:
    """Approximate t critical value."""
    try:
        from scipy.stats import t as t_dist
        return float(t_dist.ppf(1 - alpha / 2, df))
    except ImportError:
        pass

    if df >= 120:
        return 1.96
    if df >= 30:
        return 2.042
    if df >= 10:
        return 2.228
    return 2.776


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Rough approximation of the regularized incomplete beta function."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    n_steps = 200
    total = 0.0
    dx = x / n_steps
    for i in range(n_steps):
        xi = (i + 0.5) * dx
        try:
            val = xi ** (a - 1) * (1 - xi) ** (b - 1) * dx
            total += val
        except (OverflowError, ValueError):
            pass

    try:
        beta_ab = math.gamma(a) * math.gamma(b) / math.gamma(a + b)
        return total / beta_ab if beta_ab > 0 else 0.5
    except (OverflowError, ValueError):
        return 0.5
