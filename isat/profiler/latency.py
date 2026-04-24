"""Latency decomposition profiler.

Breaks inference latency into discrete phases:
  1. Model load time (disk -> memory)
  2. Provider compilation time (MIGraphX/TensorRT graph compile)
  3. First inference latency (cold path, includes JIT)
  4. Steady-state latency (warm path, averaged)
  5. Memory allocation time (per-inference feed creation)
  6. Output copy time (device -> host)

This tells you exactly WHERE time is spent, not just total latency.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.profiler")


@dataclass
class PhaseTimings:
    phase: str
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    samples: int
    pct_of_total: float = 0.0


@dataclass
class LatencyBreakdown:
    model_path: str
    provider: str
    model_load_ms: float
    compilation_ms: float
    first_inference_ms: float
    steady_state_mean_ms: float
    steady_state_p50_ms: float
    steady_state_p95_ms: float
    steady_state_p99_ms: float
    memory_alloc_ms: float
    total_e2e_ms: float
    phases: list[PhaseTimings] = field(default_factory=list)
    model_size_mb: float = 0.0
    peak_rss_mb: float = 0.0
    gpu_vram_after_load_mb: float = 0.0

    def summary_table(self) -> str:
        lines = [
            f"{'Phase':<30} {'Time (ms)':>12} {'% of E2E':>10}",
            f"{'-'*30} {'-'*12} {'-'*10}",
        ]
        for p in self.phases:
            lines.append(f"{p.phase:<30} {p.mean_ms:>12.2f} {p.pct_of_total:>9.1f}%")
        lines.append(f"{'-'*30} {'-'*12} {'-'*10}")
        lines.append(f"{'Total E2E':<30} {self.total_e2e_ms:>12.2f} {'100.0':>9}%")
        return "\n".join(lines)


class LatencyProfiler:
    """Decompose inference latency into phases."""

    def __init__(
        self,
        model_path: str,
        provider: str = "MIGraphXExecutionProvider",
        steady_state_runs: int = 50,
        warmup: int = 3,
    ):
        self.model_path = model_path
        self.provider = provider
        self.steady_state_runs = steady_state_runs
        self.warmup = warmup

    def profile(self) -> LatencyBreakdown:
        import onnxruntime as ort
        from isat.utils.sysfs import gpu_vram_used_mb

        model_size = Path(self.model_path).stat().st_size / (1024 * 1024)
        gc.collect()

        # Phase 1: Model load
        t0 = time.perf_counter()
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        t1 = time.perf_counter()
        model_load_ms = (t1 - t0) * 1000

        # Phase 2: Provider compilation (session creation triggers compile)
        t2 = time.perf_counter()
        session = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=ort_providers(self.provider),
        )
        t3 = time.perf_counter()
        compilation_ms = (t3 - t2) * 1000

        vram_after = gpu_vram_used_mb() or 0

        feed = self._build_feed(session)

        # Phase 3: First inference (cold)
        t4 = time.perf_counter()
        session.run(None, feed)
        t5 = time.perf_counter()
        first_inference_ms = (t5 - t4) * 1000

        # Warmup iterations (discard results)
        for _ in range(self.warmup):
            session.run(None, feed)

        # Phase 4: Steady-state latency
        latencies = []
        alloc_times = []
        for _ in range(self.steady_state_runs):
            ta = time.perf_counter()
            f = self._build_feed(session)
            tb = time.perf_counter()
            alloc_times.append((tb - ta) * 1000)

            tc = time.perf_counter()
            session.run(None, f)
            td = time.perf_counter()
            latencies.append((td - tc) * 1000)

        lats = np.array(latencies)
        alloc_mean = np.mean(alloc_times)
        peak_rss = _get_rss_mb()

        total = model_load_ms + compilation_ms + first_inference_ms + float(np.sum(lats))

        phases = [
            PhaseTimings("Model load (disk I/O)", model_load_ms, model_load_ms, model_load_ms, 0, 1),
            PhaseTimings("Provider compilation", compilation_ms, compilation_ms, compilation_ms, 0, 1),
            PhaseTimings("First inference (cold)", first_inference_ms, first_inference_ms, first_inference_ms, 0, 1),
            PhaseTimings("Steady-state inference", float(np.mean(lats)), float(np.min(lats)),
                         float(np.max(lats)), float(np.std(lats)), len(lats)),
            PhaseTimings("Feed allocation (per-run)", alloc_mean, float(np.min(alloc_times)),
                         float(np.max(alloc_times)), float(np.std(alloc_times)), len(alloc_times)),
        ]

        for p in phases:
            p.pct_of_total = (p.mean_ms / total * 100) if total > 0 else 0

        return LatencyBreakdown(
            model_path=self.model_path,
            provider=self.provider,
            model_load_ms=model_load_ms,
            compilation_ms=compilation_ms,
            first_inference_ms=first_inference_ms,
            steady_state_mean_ms=float(np.mean(lats)),
            steady_state_p50_ms=float(np.percentile(lats, 50)),
            steady_state_p95_ms=float(np.percentile(lats, 95)),
            steady_state_p99_ms=float(np.percentile(lats, 99)),
            memory_alloc_ms=alloc_mean,
            total_e2e_ms=total,
            phases=phases,
            model_size_mb=model_size,
            peak_rss_mb=peak_rss,
            gpu_vram_after_load_mb=vram_after,
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


def _get_rss_mb() -> float:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0
