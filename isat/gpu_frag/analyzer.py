"""GPU memory fragmentation analysis.

Monitors VRAM/GTT allocation patterns during inference to detect
fragmentation that could cause OOM despite sufficient total memory.
Measures allocation/deallocation patterns and fragmentation index.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.gpu_frag")


@dataclass
class MemorySample:
    timestamp_s: float
    vram_used_mb: float
    gtt_used_mb: float
    rss_mb: float


@dataclass
class FragmentationReport:
    model_path: str
    samples: list[MemorySample] = field(default_factory=list)
    vram_peak_mb: float = 0
    vram_min_mb: float = 0
    vram_final_mb: float = 0
    gtt_peak_mb: float = 0
    fragmentation_index: float = 0
    allocation_pattern: str = ""
    recommendation: str = ""
    duration_s: float = 0
    inference_count: int = 0

    def summary(self) -> str:
        lines = [
            f"  Model           : {self.model_path}",
            f"  Duration        : {self.duration_s:.1f}s ({self.inference_count} inferences)",
            f"  VRAM range      : {self.vram_min_mb:.0f} - {self.vram_peak_mb:.0f} MB",
            f"  VRAM final      : {self.vram_final_mb:.0f} MB",
            f"  GTT peak        : {self.gtt_peak_mb:.0f} MB",
            f"  Frag index      : {self.fragmentation_index:.2f} (0=none, 1=severe)",
            f"  Alloc pattern   : {self.allocation_pattern}",
        ]
        if self.recommendation:
            lines.append(f"  Recommendation  : {self.recommendation}")
        return "\n".join(lines)


class FragmentationAnalyzer:
    """Analyze GPU memory fragmentation during inference."""

    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider",
                 runs: int = 200, sample_every: int = 5):
        self.model_path = model_path
        self.provider = provider
        self.runs = runs
        self.sample_every = sample_every

    def analyze(self) -> FragmentationReport:
        import onnxruntime as ort
        from isat.utils.sysfs import gpu_vram_used_mb, gpu_gtt_used_mb

        import os
        def get_rss():
            try:
                with open(f"/proc/{os.getpid()}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) / 1024
            except Exception:
                pass
            return 0.0

        session = ort.InferenceSession(
            self.model_path, providers=ort_providers(self.provider),
        )
        feed = self._build_feed(session)
        for _ in range(3):
            session.run(None, feed)

        samples: list[MemorySample] = []
        t0 = time.perf_counter()

        for i in range(self.runs):
            session.run(None, feed)
            if i % self.sample_every == 0:
                samples.append(MemorySample(
                    timestamp_s=time.perf_counter() - t0,
                    vram_used_mb=gpu_vram_used_mb() or 0,
                    gtt_used_mb=gpu_gtt_used_mb() or 0,
                    rss_mb=get_rss(),
                ))

        duration = time.perf_counter() - t0

        vram_values = [s.vram_used_mb for s in samples] if samples else [0]
        gtt_values = [s.gtt_used_mb for s in samples] if samples else [0]

        vram_peak = max(vram_values)
        vram_min = min(vram_values)
        vram_final = vram_values[-1] if vram_values else 0
        gtt_peak = max(gtt_values)

        frag_index = 0.0
        if vram_peak > 0:
            vram_range = vram_peak - vram_min
            vram_variance = float(np.std(vram_values))
            frag_index = min(1.0, vram_variance / max(vram_peak, 1) * 10)

        if frag_index < 0.1:
            pattern = "stable"
            rec = "Memory allocation is stable -- no action needed"
        elif frag_index < 0.3:
            pattern = "minor_fluctuation"
            rec = "Minor memory fluctuations -- monitor but not critical"
        elif frag_index < 0.6:
            pattern = "moderate_fragmentation"
            rec = "Moderate fragmentation -- consider using memory pools or pre-allocation"
        else:
            pattern = "severe_fragmentation"
            rec = "Severe fragmentation -- use ORT arena allocator or reduce concurrent sessions"

        return FragmentationReport(
            model_path=self.model_path, samples=samples,
            vram_peak_mb=vram_peak, vram_min_mb=vram_min,
            vram_final_mb=vram_final, gtt_peak_mb=gtt_peak,
            fragmentation_index=frag_index,
            allocation_pattern=pattern, recommendation=rec,
            duration_s=duration, inference_count=self.runs,
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
