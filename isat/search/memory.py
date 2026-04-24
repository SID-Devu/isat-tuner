"""Memory strategy search dimension.

Explores combinations of:
  - HSA_XNACK (0 vs 1)
  - hipMallocManaged / hipMemAdvise hints
  - Memory pool configurations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint


@dataclass
class MemoryConfig:
    xnack: int = 0
    coarse_grain: bool = False
    pool_size_mb: int = 0
    env_overrides: dict[str, str] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = [f"xnack={self.xnack}"]
            if self.coarse_grain:
                parts.append("coarse_grain")
            if self.pool_size_mb:
                parts.append(f"pool={self.pool_size_mb}MB")
            self.label = "_".join(parts)


class MemorySearchDimension:
    """Generate candidate memory configurations based on hardware + model."""

    def __init__(self, hw: HardwareFingerprint, model: ModelFingerprint):
        self.hw = hw
        self.model = model

    def candidates(self) -> list[MemoryConfig]:
        configs: list[MemoryConfig] = []

        configs.append(MemoryConfig(
            xnack=0,
            env_overrides={"HSA_XNACK": "0"},
            label="xnack0_default",
        ))

        if self.hw.xnack_supported:
            configs.append(MemoryConfig(
                xnack=1,
                env_overrides={"HSA_XNACK": "1"},
                label="xnack1_default",
            ))

            configs.append(MemoryConfig(
                xnack=1,
                coarse_grain=True,
                env_overrides={
                    "HSA_XNACK": "1",
                    "MIGRAPHX_GPU_HIP_FLAGS": "--hip-flags=coarse_grain",
                },
                label="xnack1_coarse_grain",
            ))

        model_mb = self.model.estimated_memory_mb
        oversubscribes = (
            self.hw.is_apu
            or (model_mb > self.hw.vram_total_mb * 0.8 and self.hw.vram_total_mb > 0)
        )
        if oversubscribes and self.hw.xnack_supported:
            configs.append(MemoryConfig(
                xnack=1,
                coarse_grain=True,
                env_overrides={
                    "HSA_XNACK": "1",
                    "MIGRAPHX_GPU_HIP_FLAGS": "--hip-flags=coarse_grain",
                    "GPU_MAX_HW_QUEUES": "2",
                },
                label="xnack1_oversubscribe",
            ))

        return configs
