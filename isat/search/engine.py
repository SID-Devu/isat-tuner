"""Search engine orchestrator.

Jointly explores memory x kernel x precision x graph dimensions,
using smart pruning to avoid wasting time on clearly bad combos.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint
from isat.search.graph import GraphConfig, GraphSearchDimension
from isat.search.kernel import KernelConfig, KernelSearchDimension
from isat.search.memory import MemoryConfig, MemorySearchDimension
from isat.search.precision import PrecisionConfig, PrecisionSearchDimension

log = logging.getLogger("isat.search")


@dataclass
class CandidateConfig:
    memory: MemoryConfig
    kernel: KernelConfig
    precision: PrecisionConfig
    graph: GraphConfig
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.memory.label}|{self.kernel.label}|{self.precision.label}|{self.graph.label}"

    @property
    def merged_env(self) -> dict[str, str]:
        env = {}
        env.update(self.memory.env_overrides)
        env.update(self.kernel.env_overrides)
        env.update(self.precision.env_overrides)
        env.update(self.graph.env_overrides)
        return env

    @property
    def effective_model_path(self) -> Optional[str]:
        return self.precision.model_path_override or self.graph.model_path_override


@dataclass
class TuneResult:
    config: CandidateConfig
    mean_latency_ms: float = float("inf")
    p50_latency_ms: float = float("inf")
    p95_latency_ms: float = float("inf")
    p99_latency_ms: float = float("inf")
    min_latency_ms: float = float("inf")
    max_latency_ms: float = float("inf")
    throughput_fps: float = 0.0
    std_dev_ms: float = 0.0

    peak_gpu_temp_c: float = 0.0
    peak_power_w: float = 0.0
    peak_vram_mb: float = 0.0
    peak_gtt_mb: float = 0.0

    warmup_runs: int = 0
    measured_runs: int = 0
    cooldown_s: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "label": self.config.label,
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "min_latency_ms": round(self.min_latency_ms, 3),
            "max_latency_ms": round(self.max_latency_ms, 3),
            "throughput_fps": round(self.throughput_fps, 2),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "peak_gpu_temp_c": round(self.peak_gpu_temp_c, 1),
            "peak_power_w": round(self.peak_power_w, 1),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "peak_gtt_mb": round(self.peak_gtt_mb, 1),
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            "cooldown_s": self.cooldown_s,
            "env": self.config.merged_env,
        }
        if self.error:
            d["error"] = self.error
        return d


class SearchEngine:
    """Orchestrate the auto-tuning search over all dimensions."""

    def __init__(
        self,
        hw: HardwareFingerprint,
        model: ModelFingerprint,
        *,
        warmup: int = 3,
        runs: int = 5,
        cooldown: float = 60.0,
        max_configs: int = 0,
        skip_precision: bool = False,
        skip_graph: bool = False,
    ):
        self.hw = hw
        self.model = model
        self.warmup = warmup
        self.runs = runs
        self.cooldown = cooldown
        self.max_configs = max_configs

        self.mem_search = MemorySearchDimension(hw, model)
        self.kernel_search = KernelSearchDimension(hw, model)
        self.precision_search = PrecisionSearchDimension(hw, model) if not skip_precision else None
        self.graph_search = GraphSearchDimension(hw, model) if not skip_graph else None

    def generate_candidates(self) -> list[CandidateConfig]:
        """Generate the full Cartesian product of search dimensions, with pruning."""
        mem_configs = self.mem_search.candidates()
        kernel_configs = self.kernel_search.candidates()

        precision_configs = self.precision_search.candidates() if self.precision_search else [
            PrecisionConfig(precision="fp32", label="fp32_native")
        ]
        graph_configs = self.graph_search.candidates() if self.graph_search else [
            GraphConfig(label="raw_opt99")
        ]

        candidates = []
        for mem, kern, prec, graph in itertools.product(
            mem_configs, kernel_configs, precision_configs, graph_configs
        ):
            if self._should_prune(mem, kern, prec, graph):
                continue
            candidates.append(CandidateConfig(
                memory=mem, kernel=kern, precision=prec, graph=graph,
            ))

        if self.max_configs > 0 and len(candidates) > self.max_configs:
            log.info("Pruning %d candidates to %d (max_configs)", len(candidates), self.max_configs)
            candidates = self._rank_and_trim(candidates)

        log.info("Generated %d candidate configurations", len(candidates))
        return candidates

    def _should_prune(
        self,
        mem: MemoryConfig,
        kern: KernelConfig,
        prec: PrecisionConfig,
        graph: GraphConfig,
    ) -> bool:
        """Prune obviously bad combinations."""
        if prec.precision == "int8" and kern.disable_mlir:
            return True

        if graph.onnxsim and prec.quantize_method != "none":
            return True

        return False

    def _rank_and_trim(self, candidates: list[CandidateConfig]) -> list[CandidateConfig]:
        """Keep diverse subset when over budget."""

        def _score(c: CandidateConfig) -> int:
            score = 0
            if c.memory.xnack == 1:
                score += 1
            if c.kernel.disable_mlir:
                score += 1
            if c.precision.precision != "fp32":
                score += 1
            if c.graph.onnxsim:
                score += 1
            return score

        candidates.sort(key=_score)

        kept: list[CandidateConfig] = []
        labels_seen: set[str] = set()
        for c in candidates:
            signature = f"{c.memory.label}:{c.kernel.label}"
            if signature not in labels_seen or len(kept) < self.max_configs:
                kept.append(c)
                labels_seen.add(signature)
            if len(kept) >= self.max_configs:
                break
        return kept

    def print_plan(self, candidates: list[CandidateConfig]) -> None:
        """Print a human-readable plan of what will be benchmarked."""
        print(f"\n{'='*72}")
        print(f"  ISAT TUNING PLAN")
        print(f"{'='*72}")
        print(f"  GPU       : {self.hw.gpu_name} ({self.hw.gfx_target})")
        print(f"  Model     : {self.model.name} ({self.model.size_class}, "
              f"{self.model.estimated_memory_mb:.0f} MB)")
        print(f"  Class     : {self.model.model_class}")
        print(f"  Warmup    : {self.warmup}  |  Runs: {self.runs}  |  "
              f"Cooldown: {self.cooldown}s")
        print(f"  Candidates: {len(candidates)}")
        print(f"{'='*72}")
        for i, c in enumerate(candidates, 1):
            print(f"  [{i:2d}] {c.label}")
            if c.merged_env:
                for k, v in c.merged_env.items():
                    print(f"       {k}={v}")
        print(f"{'='*72}\n")
