"""Warmup iteration analyzer.

Determines the optimal number of warmup iterations for a model:
  - Runs increasing warmup counts
  - Detects when latency stabilizes (converges)
  - Identifies JIT compilation boundaries
  - Reports the knee point
  - Avoids under-warming (noisy benchmarks) and over-warming (wasted time)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.warmup")


@dataclass
class WarmupProfile:
    model_path: str
    provider: str
    optimal_warmup: int
    convergence_iteration: int
    per_iteration_ms: list[float]
    steady_state_mean_ms: float
    steady_state_std_ms: float
    jit_boundary: int
    cold_vs_warm_ratio: float
    recommendation: str

    def summary(self) -> str:
        lines = [
            f"  Optimal warmup      : {self.optimal_warmup} iterations",
            f"  Convergence point   : iteration {self.convergence_iteration}",
            f"  JIT boundary        : iteration {self.jit_boundary}",
            f"  Steady-state        : {self.steady_state_mean_ms:.2f} +/- {self.steady_state_std_ms:.2f} ms",
            f"  Cold/warm ratio     : {self.cold_vs_warm_ratio:.1f}x",
            f"  Recommendation      : {self.recommendation}",
        ]
        return "\n".join(lines)


class WarmupAnalyzer:
    """Analyze warmup behavior of a model."""

    def __init__(
        self,
        model_path: str,
        provider: str = "MIGraphXExecutionProvider",
        max_iterations: int = 100,
        convergence_window: int = 10,
        convergence_cv_threshold: float = 0.05,
    ):
        self.model_path = model_path
        self.provider = provider
        self.max_iterations = max_iterations
        self.convergence_window = convergence_window
        self.cv_threshold = convergence_cv_threshold

    def analyze(self) -> WarmupProfile:
        import onnxruntime as ort

        session = ort.InferenceSession(
            self.model_path,
            providers=ort_providers(self.provider),
        )

        feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            elif "float16" in inp.type.lower():
                feed[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)

        latencies = []
        for i in range(self.max_iterations):
            t0 = time.perf_counter()
            session.run(None, feed)
            latencies.append((time.perf_counter() - t0) * 1000)

        lats = np.array(latencies)

        convergence = self._find_convergence(lats)
        jit_boundary = self._find_jit_boundary(lats)

        optimal = max(convergence, jit_boundary) + 2

        steady = lats[optimal:] if optimal < len(lats) else lats[-self.convergence_window:]
        steady_mean = float(np.mean(steady))
        steady_std = float(np.std(steady))

        cold_warm = lats[0] / steady_mean if steady_mean > 0 else 1.0

        if cold_warm > 10:
            rec = f"Heavy JIT detected ({cold_warm:.0f}x cold penalty). Use warmup >= {optimal}"
        elif cold_warm > 3:
            rec = f"Moderate JIT ({cold_warm:.1f}x cold penalty). Use warmup >= {optimal}"
        else:
            rec = f"Light warmup needed ({cold_warm:.1f}x cold penalty). Use warmup >= {optimal}"

        return WarmupProfile(
            model_path=self.model_path,
            provider=self.provider,
            optimal_warmup=optimal,
            convergence_iteration=convergence,
            per_iteration_ms=latencies,
            steady_state_mean_ms=steady_mean,
            steady_state_std_ms=steady_std,
            jit_boundary=jit_boundary,
            cold_vs_warm_ratio=cold_warm,
            recommendation=rec,
        )

    def _find_convergence(self, lats: np.ndarray) -> int:
        """Find the iteration where latency CV drops below threshold."""
        w = self.convergence_window
        for i in range(w, len(lats)):
            window = lats[i - w:i]
            mean = np.mean(window)
            if mean > 0 and np.std(window) / mean < self.cv_threshold:
                return max(0, i - w)
        return max(0, len(lats) - w)

    def _find_jit_boundary(self, lats: np.ndarray) -> int:
        """Find where per-iteration latency stops having sharp drops."""
        if len(lats) < 5:
            return 0

        median_tail = np.median(lats[len(lats) // 2:])
        threshold = median_tail * 1.5

        for i in range(len(lats)):
            if lats[i] <= threshold:
                return max(0, i - 1)
        return 0
