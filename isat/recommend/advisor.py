"""Hardware recommendation engine.

Given a model's characteristics (size, ops, precision requirements),
recommends the optimal hardware target and configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.recommend")


@dataclass
class HWRecommendation:
    rank: int
    hardware: str
    provider: str
    estimated_latency_ms: float
    estimated_cost_hr: float
    memory_fit: str  # "comfortable", "tight", "needs_offload"
    precision: str
    notes: str


@dataclass
class RecommendationReport:
    model_path: str
    model_params: int
    model_size_mb: float
    model_class: str
    recommendations: list[HWRecommendation] = field(default_factory=list)
    constraints: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Model        : {self.model_path}",
            f"  Parameters   : {self.model_params:,}",
            f"  Size         : {self.model_size_mb:.0f} MB",
            f"  Class        : {self.model_class}",
        ]
        if self.constraints:
            lines.append(f"  Constraints  : {self.constraints}")
        lines.append(f"")
        lines.append(f"  {'Rank':<5} {'Hardware':<30} {'Provider':<30} {'Est. Latency':>12} {'Cost/hr':>10} {'Memory':>15} {'Precision':>10}")
        lines.append(f"  {'-'*5} {'-'*30} {'-'*30} {'-'*12} {'-'*10} {'-'*15} {'-'*10}")

        for r in self.recommendations:
            lat = r.estimated_latency_ms
            lat_str = "<0.1ms" if 0 < lat < 0.05 else f"{lat:.1f}ms"
            lines.append(
                f"  {r.rank:<5} {r.hardware:<30} {r.provider:<30} "
                f"{lat_str:>12} {f'${r.estimated_cost_hr:.2f}':>10} "
                f"{r.memory_fit:>15} {r.precision:>10}"
            )
            if r.notes:
                lines.append(f"  {'':>5} {r.notes}")

        return "\n".join(lines)


HARDWARE_DB = [
    {
        "name": "AMD Strix Halo APU (gfx1151)",
        "provider": "MIGraphXExecutionProvider",
        "vram_mb": 512,
        "gtt_mb": 28672,
        "unified": True,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 12.0,
        "tflops_fp16": 24.0,
        "cost_hr": 0.0,
        "best_for": ["edge", "apu", "unified_memory"],
    },
    {
        "name": "NVIDIA T4 (AWS g4dn)",
        "provider": "TensorrtExecutionProvider",
        "vram_mb": 16384,
        "gtt_mb": 0,
        "unified": False,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 8.1,
        "tflops_fp16": 65.0,
        "cost_hr": 0.526,
        "best_for": ["cloud", "inference", "cost_effective"],
    },
    {
        "name": "NVIDIA A10G (AWS g5)",
        "provider": "TensorrtExecutionProvider",
        "vram_mb": 24576,
        "gtt_mb": 0,
        "unified": False,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 31.2,
        "tflops_fp16": 125.0,
        "cost_hr": 1.21,
        "best_for": ["cloud", "inference", "mid_range"],
    },
    {
        "name": "NVIDIA A100 40GB (AWS p4d)",
        "provider": "CUDAExecutionProvider",
        "vram_mb": 40960,
        "gtt_mb": 0,
        "unified": False,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 19.5,
        "tflops_fp16": 312.0,
        "cost_hr": 4.10,
        "best_for": ["cloud", "training", "large_models"],
    },
    {
        "name": "NVIDIA H100 80GB (AWS p5)",
        "provider": "CUDAExecutionProvider",
        "vram_mb": 81920,
        "gtt_mb": 0,
        "unified": False,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 51.2,
        "tflops_fp16": 989.0,
        "cost_hr": 12.72,
        "best_for": ["cloud", "training", "very_large_models"],
    },
    {
        "name": "AMD MI250X (Azure ND A100 v4)",
        "provider": "MIGraphXExecutionProvider",
        "vram_mb": 131072,
        "gtt_mb": 0,
        "unified": False,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 47.9,
        "tflops_fp16": 383.0,
        "cost_hr": 6.50,
        "best_for": ["cloud", "training", "amd_datacenter"],
    },
    {
        "name": "AMD MI300X",
        "provider": "MIGraphXExecutionProvider",
        "vram_mb": 196608,
        "gtt_mb": 0,
        "unified": False,
        "fp16": True,
        "int8": True,
        "tflops_fp32": 81.7,
        "tflops_fp16": 653.0,
        "cost_hr": 10.00,
        "best_for": ["cloud", "training", "very_large_models"],
    },
    {
        "name": "Intel Xeon 4th Gen (CPU)",
        "provider": "CPUExecutionProvider",
        "vram_mb": 0,
        "gtt_mb": 0,
        "unified": False,
        "fp16": False,
        "int8": True,
        "tflops_fp32": 2.0,
        "tflops_fp16": 2.0,
        "cost_hr": 0.20,
        "best_for": ["cpu", "edge", "cost_effective"],
    },
]


class HardwareAdvisor:
    """Recommend optimal hardware for a given ONNX model."""

    def recommend(
        self,
        model_path: str,
        max_latency_ms: float = 0,
        max_cost_hr: float = 0,
        prefer_amd: bool = False,
    ) -> RecommendationReport:
        from isat.fingerprint.model import fingerprint_model

        fp = fingerprint_model(model_path)
        model_mem_fp32 = fp.estimated_memory_mb
        model_mem_fp16 = model_mem_fp32 / 2

        constraints = {}
        if max_latency_ms:
            constraints["max_latency_ms"] = max_latency_ms
        if max_cost_hr:
            constraints["max_cost_hr"] = max_cost_hr

        candidates = []
        for hw in HARDWARE_DB:
            total_mem = hw["vram_mb"] + hw.get("gtt_mb", 0)

            if total_mem > 0 and model_mem_fp16 < total_mem * 0.8:
                memory_fit = "comfortable"
            elif total_mem > 0 and model_mem_fp16 < total_mem:
                memory_fit = "tight"
            elif hw.get("unified"):
                memory_fit = "needs_offload"
            else:
                memory_fit = "needs_offload"

            tflops = hw["tflops_fp16"] if hw["fp16"] else hw["tflops_fp32"]
            gflops_per_inference = fp.param_count * 2 / 1e9
            est_latency = gflops_per_inference / tflops * 1000 if tflops > 0 else 99999

            precision = "FP16" if hw["fp16"] else ("INT8" if hw["int8"] else "FP32")

            if max_latency_ms and est_latency > max_latency_ms * 2:
                continue
            if max_cost_hr and hw["cost_hr"] > max_cost_hr:
                continue

            score = 0
            score += 100 - min(est_latency / 10, 100)
            score += 50 if memory_fit == "comfortable" else (25 if memory_fit == "tight" else 0)
            score -= hw["cost_hr"] * 5
            if prefer_amd and "AMD" in hw["name"]:
                score += 20

            candidates.append((score, hw, est_latency, memory_fit, precision))

        candidates.sort(key=lambda x: -x[0])

        recommendations = []
        for rank, (score, hw, est_lat, mem_fit, prec) in enumerate(candidates[:8], 1):
            notes = ""
            if mem_fit == "needs_offload":
                notes = "Model may need XNACK/demand paging or model parallelism"
            elif mem_fit == "tight":
                notes = "Memory is tight -- use FP16 or consider batch=1 only"

            recommendations.append(HWRecommendation(
                rank=rank,
                hardware=hw["name"],
                provider=hw["provider"],
                estimated_latency_ms=est_lat,
                estimated_cost_hr=hw["cost_hr"],
                memory_fit=mem_fit,
                precision=prec,
                notes=notes,
            ))

        return RecommendationReport(
            model_path=model_path,
            model_params=fp.param_count,
            model_size_mb=fp.estimated_memory_mb,
            model_class=fp.model_class,
            recommendations=recommendations,
            constraints=constraints,
        )
