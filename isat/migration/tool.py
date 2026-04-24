"""Provider migration tool.

Helps migrate inference configurations between hardware/providers:
  - ROCm (MIGraphX) <-> CUDA (TensorRT)
  - GPU <-> CPU
  - Cloud <-> Edge
  - One GPU generation -> another

Maps environment variables, session options, and tuning parameters
between equivalent settings on different platforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("isat.migration")


@dataclass
class MigrationStep:
    action: str
    description: str
    before: str
    after: str
    breaking: bool = False


@dataclass
class MigrationPlan:
    source_provider: str
    target_provider: str
    steps: list[MigrationStep] = field(default_factory=list)
    env_changes: dict[str, str] = field(default_factory=dict)
    env_removes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    breaking_changes: int = 0

    def summary(self) -> str:
        lines = [
            f"  Migration: {self.source_provider} -> {self.target_provider}",
            f"  Steps: {len(self.steps)} ({self.breaking_changes} breaking)",
        ]
        if self.env_changes:
            lines.append(f"\n  Environment variable changes:")
            for k, v in self.env_changes.items():
                lines.append(f"    export {k}={v}")
        if self.env_removes:
            lines.append(f"\n  Remove these environment variables:")
            for k in self.env_removes:
                lines.append(f"    unset {k}")
        if self.warnings:
            lines.append(f"\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    {w}")
        if self.steps:
            lines.append(f"\n  Detailed steps:")
            for i, s in enumerate(self.steps, 1):
                flag = " [BREAKING]" if s.breaking else ""
                lines.append(f"    {i}. {s.description}{flag}")
                lines.append(f"       Before: {s.before}")
                lines.append(f"       After:  {s.after}")
        return "\n".join(lines)


# Mappings between equivalent provider settings
ENV_MAP: dict[str, dict[str, str]] = {
    "MIGraphXExecutionProvider->TensorrtExecutionProvider": {
        "MIGRAPHX_FP16_ENABLE": "ORT_TENSORRT_FP16_ENABLE",
        "MIGRAPHX_INT8_ENABLE": "ORT_TENSORRT_INT8_ENABLE",
        "MIGRAPHX_GPU_COMPILE_PARALLEL": "",
        "HSA_XNACK": "",
        "MIGRAPHX_DISABLE_MLIR": "",
    },
    "MIGraphXExecutionProvider->CUDAExecutionProvider": {
        "MIGRAPHX_FP16_ENABLE": "",
        "HSA_XNACK": "",
    },
    "TensorrtExecutionProvider->MIGraphXExecutionProvider": {
        "ORT_TENSORRT_FP16_ENABLE": "MIGRAPHX_FP16_ENABLE",
        "ORT_TENSORRT_INT8_ENABLE": "MIGRAPHX_INT8_ENABLE",
        "ORT_TENSORRT_ENGINE_CACHE_ENABLE": "",
    },
    "CUDAExecutionProvider->MIGraphXExecutionProvider": {},
    "MIGraphXExecutionProvider->CPUExecutionProvider": {
        "MIGRAPHX_FP16_ENABLE": "",
        "HSA_XNACK": "",
        "MIGRAPHX_DISABLE_MLIR": "",
        "MIGRAPHX_SET_GEMM_PROVIDER": "",
    },
}


class MigrationTool:
    """Generate migration plans between providers."""

    def plan(
        self,
        source_provider: str,
        target_provider: str,
        current_env: dict[str, str] | None = None,
    ) -> MigrationPlan:
        key = f"{source_provider}->{target_provider}"
        mapping = ENV_MAP.get(key, {})

        plan = MigrationPlan(
            source_provider=source_provider,
            target_provider=target_provider,
        )

        plan.steps.append(MigrationStep(
            action="change_provider",
            description=f"Change ORT execution provider",
            before=f'providers=["{source_provider}"]',
            after=f'providers=["{target_provider}"]',
        ))

        if current_env:
            for src_var, tgt_var in mapping.items():
                if src_var in current_env:
                    if tgt_var:
                        plan.env_changes[tgt_var] = current_env[src_var]
                        plan.steps.append(MigrationStep(
                            action="env_rename",
                            description=f"Map {src_var} -> {tgt_var}",
                            before=f"export {src_var}={current_env[src_var]}",
                            after=f"export {tgt_var}={current_env[src_var]}",
                        ))
                    plan.env_removes.append(src_var)

        if "MIGraphX" in source_provider and "CUDA" in target_provider:
            plan.warnings.append("MIGraphX-specific MLIR/rocBLAS settings have no CUDA equivalent")
            plan.warnings.append("XNACK (unified memory) is not available on NVIDIA GPUs")

        if "MIGraphX" in source_provider and "Tensorrt" in target_provider:
            plan.warnings.append("TensorRT uses different graph compilation strategy")
            plan.warnings.append("Engine cache format differs from MIGraphX cache")
            plan.steps.append(MigrationStep(
                action="enable_cache",
                description="Enable TensorRT engine cache",
                before="N/A (MIGraphX caches automatically)",
                after='export ORT_TENSORRT_ENGINE_CACHE_ENABLE=1',
            ))

        if "CPU" in target_provider:
            plan.warnings.append("CPU inference will be significantly slower")
            plan.warnings.append("FP16 may not be supported on all CPUs")
            plan.steps.append(MigrationStep(
                action="threading",
                description="Configure CPU threading",
                before="GPU handles parallelism",
                after="Set OMP_NUM_THREADS and intra/inter_op_num_threads",
                breaking=True,
            ))
            plan.breaking_changes += 1

        if "GPU" not in target_provider and "CPU" not in target_provider:
            plan.steps.append(MigrationStep(
                action="retune",
                description="Re-run ISAT tuning for new provider",
                before=f"Tuned for {source_provider}",
                after=f"isat tune model.onnx --provider {target_provider}",
            ))

        return plan

    @staticmethod
    def supported_migrations() -> list[str]:
        return list(ENV_MAP.keys())
