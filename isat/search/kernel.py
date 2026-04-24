"""Kernel backend search dimension.

Explores:
  - Default MIGraphX MLIR fusion
  - rocBLAS explicit GEMM
  - hipBLASlt
  - Compilation parallelism
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint


@dataclass
class KernelConfig:
    disable_mlir: bool = False
    gemm_provider: str = "default"
    compile_parallel: int = 0
    env_overrides: dict[str, str] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = []
            parts.append("mlir_off" if self.disable_mlir else "mlir_on")
            if self.gemm_provider != "default":
                parts.append(f"gemm={self.gemm_provider}")
            if self.compile_parallel:
                parts.append(f"par={self.compile_parallel}")
            self.label = "_".join(parts)


class KernelSearchDimension:
    """Generate candidate kernel configurations."""

    def __init__(self, hw: HardwareFingerprint, model: ModelFingerprint):
        self.hw = hw
        self.model = model

    def candidates(self) -> list[KernelConfig]:
        configs: list[KernelConfig] = []

        configs.append(KernelConfig(
            disable_mlir=False,
            gemm_provider="default",
            env_overrides={},
            label="mlir_default",
        ))

        configs.append(KernelConfig(
            disable_mlir=True,
            gemm_provider="rocblas",
            env_overrides={
                "MIGRAPHX_DISABLE_MLIR": "1",
                "MIGRAPHX_SET_GEMM_PROVIDER": "rocblas",
            },
            label="rocblas_explicit",
        ))

        if self.model.gemm_fraction > 0.3:
            configs.append(KernelConfig(
                disable_mlir=True,
                gemm_provider="hipblaslt",
                env_overrides={
                    "MIGRAPHX_DISABLE_MLIR": "1",
                    "MIGRAPHX_SET_GEMM_PROVIDER": "hipblaslt",
                },
                label="hipblaslt_explicit",
            ))

        import os
        ncpu = os.cpu_count() or 4
        if self.model.num_nodes > 100:
            configs.append(KernelConfig(
                disable_mlir=False,
                compile_parallel=ncpu,
                env_overrides={
                    "MIGRAPHX_GPU_COMPILE_PARALLEL": str(ncpu),
                },
                label=f"mlir_parallel_{ncpu}",
            ))

        return configs
