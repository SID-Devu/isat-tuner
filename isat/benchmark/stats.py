"""Statistical analysis for benchmark latencies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cv_pct: float
    n: int


def compute_stats(latencies_ms: list[float]) -> LatencyStats:
    """Compute percentile statistics from a list of latency measurements."""
    if not latencies_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0, 0, 0, 0)

    arr = np.array(latencies_ms, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return LatencyStats(
        mean_ms=mean,
        std_ms=std,
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        cv_pct=(std / mean * 100) if mean > 0 else 0.0,
        n=len(arr),
    )
