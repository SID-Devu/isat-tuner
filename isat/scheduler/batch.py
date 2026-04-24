"""Adaptive batch scheduler.

Finds the optimal batch size by profiling the latency-throughput tradeoff:
  - Runs inference at increasing batch sizes
  - Measures per-sample latency and total throughput
  - Finds the knee point where throughput gains diminish
  - Detects OOM boundaries
  - Recommends batch size for different objectives (latency, throughput, balanced)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.scheduler")


@dataclass
class BatchResult:
    batch_size: int
    total_latency_ms: float
    per_sample_ms: float
    throughput_fps: float
    success: bool
    error: str = ""


@dataclass
class BatchProfile:
    model_path: str
    results: list[BatchResult] = field(default_factory=list)
    recommended_latency: int = 1
    recommended_throughput: int = 1
    recommended_balanced: int = 1
    max_viable_batch: int = 1
    knee_point: int = 1

    def summary(self) -> str:
        lines = [
            f"  {'Batch':>8} {'Total ms':>12} {'Per-sample ms':>14} {'FPS':>10} {'Status':>8}",
            f"  {'-'*8} {'-'*12} {'-'*14} {'-'*10} {'-'*8}",
        ]
        for r in self.results:
            status = "OK" if r.success else "FAIL"
            lines.append(
                f"  {r.batch_size:>8} {r.total_latency_ms:>12.2f} "
                f"{r.per_sample_ms:>14.2f} {r.throughput_fps:>10.1f} {status:>8}"
            )
        lines.append(f"\n  Recommendations:")
        lines.append(f"    Lowest latency  : batch_size = {self.recommended_latency}")
        lines.append(f"    Max throughput   : batch_size = {self.recommended_throughput}")
        lines.append(f"    Balanced         : batch_size = {self.recommended_balanced}")
        lines.append(f"    Max viable       : batch_size = {self.max_viable_batch}")
        lines.append(f"    Knee point       : batch_size = {self.knee_point}")
        return "\n".join(lines)


class BatchScheduler:
    """Find optimal batch size for an ONNX model."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        warmup: int = 3,
        runs: int = 10,
        batch_sizes: list[int] | None = None,
    ):
        self.model_path = model_path
        self.provider = provider
        self.warmup = warmup
        self.runs = runs
        self.batch_sizes = batch_sizes or [1, 2, 4, 8, 16, 32, 64]

    def profile(self) -> BatchProfile:
        import onnxruntime as ort

        results: list[BatchResult] = []

        for bs in self.batch_sizes:
            try:
                session = ort.InferenceSession(
                    self.model_path,
                    providers=ort_providers(self.provider),
                )
                feed = self._build_feed(session, bs)

                for _ in range(self.warmup):
                    session.run(None, feed)

                latencies = []
                for _ in range(self.runs):
                    t0 = time.perf_counter()
                    session.run(None, feed)
                    latencies.append((time.perf_counter() - t0) * 1000)

                mean_total = float(np.mean(latencies))
                per_sample = mean_total / bs
                throughput = bs * 1000.0 / mean_total

                results.append(BatchResult(
                    batch_size=bs,
                    total_latency_ms=mean_total,
                    per_sample_ms=per_sample,
                    throughput_fps=throughput,
                    success=True,
                ))
            except Exception as e:
                log.warning("Batch size %d failed: %s", bs, e)
                results.append(BatchResult(
                    batch_size=bs, total_latency_ms=0,
                    per_sample_ms=0, throughput_fps=0,
                    success=False, error=str(e)[:100],
                ))
                break

        ok = [r for r in results if r.success]
        if not ok:
            return BatchProfile(model_path=self.model_path, results=results)

        rec_latency = min(ok, key=lambda r: r.per_sample_ms).batch_size
        rec_throughput = max(ok, key=lambda r: r.throughput_fps).batch_size
        max_viable = ok[-1].batch_size

        knee = self._find_knee(ok)

        best_efficiency = 0
        rec_balanced = 1
        for r in ok:
            efficiency = r.throughput_fps / r.per_sample_ms if r.per_sample_ms > 0 else 0
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                rec_balanced = r.batch_size

        return BatchProfile(
            model_path=self.model_path,
            results=results,
            recommended_latency=rec_latency,
            recommended_throughput=rec_throughput,
            recommended_balanced=rec_balanced,
            max_viable_batch=max_viable,
            knee_point=knee,
        )

    def _find_knee(self, results: list[BatchResult]) -> int:
        if len(results) < 3:
            return results[-1].batch_size if results else 1

        for i in range(1, len(results) - 1):
            prev_gain = (results[i].throughput_fps - results[i - 1].throughput_fps) / results[i - 1].throughput_fps if results[i - 1].throughput_fps > 0 else 0
            next_gain = (results[i + 1].throughput_fps - results[i].throughput_fps) / results[i].throughput_fps if results[i].throughput_fps > 0 else 0

            if next_gain < prev_gain * 0.3:
                return results[i].batch_size

        return results[-1].batch_size

    def _build_feed(self, session, batch_size: int) -> dict:
        feed = {}
        for inp in session.get_inputs():
            shape = list(inp.shape)
            for i, d in enumerate(shape):
                if not isinstance(d, int) or d <= 0:
                    shape[i] = batch_size if i == 0 else 1

            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            elif "float16" in inp.type.lower():
                feed[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        return feed
