"""Memory leak detection for inference sessions.

Runs repeated inferences and monitors memory growth to detect
leaks in the model, runtime, or provider.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isat.utils.sysfs import gpu_gtt_used_mb, gpu_vram_used_mb

log = logging.getLogger("isat.stress.leak")


@dataclass
class LeakReport:
    iterations: int
    duration_s: float
    host_rss_start_mb: float
    host_rss_end_mb: float
    host_rss_delta_mb: float
    gpu_vram_start_mb: float
    gpu_vram_end_mb: float
    gpu_vram_delta_mb: float
    gpu_gtt_start_mb: float
    gpu_gtt_end_mb: float
    gpu_gtt_delta_mb: float
    leak_detected: bool
    leak_rate_mb_per_1k: float
    samples: list[dict] = field(default_factory=list)
    verdict: str = ""


class MemoryLeakDetector:
    """Detect memory leaks during inference."""

    def __init__(
        self,
        model_path: str,
        provider: str = "MIGraphXExecutionProvider",
        iterations: int = 1000,
        sample_interval: int = 50,
        leak_threshold_mb: float = 50.0,
    ):
        self.model_path = model_path
        self.provider = provider
        self.iterations = iterations
        self.sample_interval = sample_interval
        self.leak_threshold_mb = leak_threshold_mb

    def run(self) -> LeakReport:
        """Run the leak detection test."""
        log.info("Memory leak detection: %d iterations, sampling every %d",
                 self.iterations, self.sample_interval)

        import onnxruntime as ort

        gc.collect()
        host_start = _get_rss_mb()
        vram_start = gpu_vram_used_mb() or 0
        gtt_start = gpu_gtt_used_mb() or 0

        session = ort.InferenceSession(
            self.model_path,
            providers=[self.provider, "CPUExecutionProvider"],
        )

        feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)

        samples: list[dict] = []
        start = time.time()

        for i in range(1, self.iterations + 1):
            session.run(None, feed)

            if i % self.sample_interval == 0:
                samples.append({
                    "iteration": i,
                    "host_rss_mb": _get_rss_mb(),
                    "gpu_vram_mb": gpu_vram_used_mb() or 0,
                    "gpu_gtt_mb": gpu_gtt_used_mb() or 0,
                    "elapsed_s": time.time() - start,
                })

        gc.collect()
        duration = time.time() - start
        host_end = _get_rss_mb()
        vram_end = gpu_vram_used_mb() or 0
        gtt_end = gpu_gtt_used_mb() or 0

        host_delta = host_end - host_start
        vram_delta = vram_end - vram_start
        gtt_delta = gtt_end - gtt_start
        total_delta = host_delta + vram_delta + gtt_delta

        leak_detected = total_delta > self.leak_threshold_mb
        leak_rate = total_delta / (self.iterations / 1000) if self.iterations > 0 else 0

        if leak_detected:
            verdict = f"LEAK DETECTED: {total_delta:.1f} MB growth over {self.iterations} iterations"
        elif total_delta > self.leak_threshold_mb * 0.5:
            verdict = f"WARNING: {total_delta:.1f} MB growth (borderline)"
        else:
            verdict = f"OK: {total_delta:.1f} MB growth (within threshold)"

        return LeakReport(
            iterations=self.iterations,
            duration_s=duration,
            host_rss_start_mb=host_start,
            host_rss_end_mb=host_end,
            host_rss_delta_mb=host_delta,
            gpu_vram_start_mb=vram_start,
            gpu_vram_end_mb=vram_end,
            gpu_vram_delta_mb=vram_delta,
            gpu_gtt_start_mb=gtt_start,
            gpu_gtt_end_mb=gtt_end,
            gpu_gtt_delta_mb=gtt_delta,
            leak_detected=leak_detected,
            leak_rate_mb_per_1k=leak_rate,
            samples=samples,
            verdict=verdict,
        )


def _get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return 0.0
