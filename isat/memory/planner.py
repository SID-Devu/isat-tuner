"""Memory planner -- predict memory usage before running.

Prevents OOM crashes by:
  1. Estimating peak memory from model structure (params + activations)
  2. Comparing against available GPU VRAM and system RAM
  3. Recommending safe batch sizes
  4. Predicting whether XNACK (demand paging) is needed
  5. Estimating memory for each precision (FP32, FP16, INT8)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.memory")


@dataclass
class MemoryEstimate:
    params_mb: float
    activations_mb: float
    total_estimated_mb: float
    precision: str


@dataclass
class MemoryPlan:
    model_path: str
    model_params: int
    file_size_mb: float
    estimates: dict[str, MemoryEstimate] = field(default_factory=dict)
    gpu_vram_mb: float = 0.0
    gpu_gtt_mb: float = 0.0
    system_ram_mb: float = 0.0
    recommended_batch_sizes: dict[str, list[int]] = field(default_factory=dict)
    needs_xnack: bool = False
    oom_risk: str = "low"
    recommendations: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Model               : {self.model_path}",
            f"  Parameters          : {self.model_params:,}",
            f"  File size           : {self.file_size_mb:.1f} MB",
            f"",
            f"  Memory estimates by precision:",
            f"  {'Precision':<12} {'Params MB':>12} {'Activations MB':>16} {'Total MB':>12}",
            f"  {'-'*12} {'-'*12} {'-'*16} {'-'*12}",
        ]
        for prec, est in self.estimates.items():
            lines.append(f"  {prec:<12} {est.params_mb:>12.1f} {est.activations_mb:>16.1f} {est.total_estimated_mb:>12.1f}")

        if self.gpu_vram_mb > 0:
            lines.append(f"\n  Available GPU VRAM  : {self.gpu_vram_mb:.0f} MB")
        if self.system_ram_mb > 0:
            lines.append(f"  Available system RAM: {self.system_ram_mb:.0f} MB")

        lines.append(f"\n  OOM risk            : {self.oom_risk.upper()}")
        if self.needs_xnack:
            lines.append(f"  XNACK recommended   : YES (model exceeds VRAM)")

        if self.recommended_batch_sizes:
            lines.append(f"\n  Safe batch sizes:")
            for prec, sizes in self.recommended_batch_sizes.items():
                lines.append(f"    {prec}: {sizes}")

        if self.recommendations:
            lines.append(f"\n  Recommendations:")
            for r in self.recommendations:
                lines.append(f"    - {r}")

        return "\n".join(lines)


class MemoryPlanner:
    """Estimate memory usage and recommend safe configurations."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def plan(self, gpu_vram_mb: float = 0, system_ram_mb: float = 0) -> MemoryPlan:
        import onnx

        model = onnx.load(self.model_path, load_external_data=False)
        file_size = Path(self.model_path).stat().st_size / (1024 * 1024)

        total_params = 0
        for init in model.graph.initializer:
            n = 1
            for d in init.dims:
                n *= d
            total_params += n

        if gpu_vram_mb <= 0:
            gpu_vram_mb = self._detect_vram()
        if system_ram_mb <= 0:
            system_ram_mb = self._detect_system_ram()

        estimates: dict[str, MemoryEstimate] = {}
        for prec, bytes_per_param in [("FP32", 4), ("FP16", 2), ("INT8", 1)]:
            params_mb = total_params * bytes_per_param / (1024 * 1024)
            activation_multiplier = 2.0 if prec == "FP32" else 1.5 if prec == "FP16" else 1.2
            activations_mb = params_mb * activation_multiplier
            estimates[prec] = MemoryEstimate(
                params_mb=params_mb,
                activations_mb=activations_mb,
                total_estimated_mb=params_mb + activations_mb,
                precision=prec,
            )

        fp16_total = estimates["FP16"].total_estimated_mb
        needs_xnack = gpu_vram_mb > 0 and fp16_total > gpu_vram_mb * 0.9

        if gpu_vram_mb > 0:
            if fp16_total > gpu_vram_mb:
                oom_risk = "high"
            elif fp16_total > gpu_vram_mb * 0.7:
                oom_risk = "medium"
            else:
                oom_risk = "low"
        else:
            oom_risk = "unknown"

        batch_sizes: dict[str, list[int]] = {}
        for prec, est in estimates.items():
            if gpu_vram_mb > 0 and est.total_estimated_mb > 0:
                max_bs = int(gpu_vram_mb * 0.8 / est.total_estimated_mb)
                batch_sizes[prec] = [bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= max(1, max_bs)]
            else:
                batch_sizes[prec] = [1, 2, 4, 8]

        recs: list[str] = []
        if oom_risk == "high":
            recs.append("Model likely exceeds GPU VRAM -- enable XNACK=1 for demand paging")
            recs.append("Use FP16 precision to halve memory")
            recs.append("Keep batch size = 1")
        elif oom_risk == "medium":
            recs.append("Close to VRAM limit -- use FP16 and small batch sizes")
        if needs_xnack:
            recs.append("export HSA_XNACK=1  # Enable unified memory / demand paging")
        if total_params > 100_000_000:
            recs.append("Large model -- consider INT8 quantization for 4x memory savings")

        return MemoryPlan(
            model_path=self.model_path,
            model_params=total_params,
            file_size_mb=file_size,
            estimates=estimates,
            gpu_vram_mb=gpu_vram_mb,
            gpu_gtt_mb=0,
            system_ram_mb=system_ram_mb,
            recommended_batch_sizes=batch_sizes,
            needs_xnack=needs_xnack,
            oom_risk=oom_risk,
            recommendations=recs,
        )

    def _detect_vram(self) -> float:
        try:
            p = Path("/sys/class/drm/card0/device/mem_info_vram_total")
            if p.exists():
                return int(p.read_text().strip()) / (1024 * 1024)
        except (OSError, ValueError):
            pass
        return 0

    def _detect_system_ram(self) -> float:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / 1024
        except (OSError, ValueError):
            pass
        return 0
