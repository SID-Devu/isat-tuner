"""Thread and execution tuning search dimension.

Explores:
  - ORT inter/intra op thread counts
  - Execution mode (sequential vs parallel)
  - Memory arena settings
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint


@dataclass
class ThreadConfig:
    inter_op_threads: int = 0
    intra_op_threads: int = 0
    execution_mode: str = "sequential"
    enable_mem_arena: bool = True
    enable_mem_pattern: bool = True
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = [f"inter{self.inter_op_threads}", f"intra{self.intra_op_threads}", self.execution_mode[:3]]
            self.label = "_".join(parts)


class ThreadSearchDimension:
    """Generate candidate thread configurations."""

    def __init__(self, hw: HardwareFingerprint, model: ModelFingerprint):
        self.hw = hw
        self.model = model
        self.ncpu = os.cpu_count() or 4

    def candidates(self) -> list[ThreadConfig]:
        configs: list[ThreadConfig] = []

        configs.append(ThreadConfig(
            inter_op_threads=0,
            intra_op_threads=0,
            execution_mode="sequential",
            label="threads_default_seq",
        ))

        configs.append(ThreadConfig(
            inter_op_threads=1,
            intra_op_threads=self.ncpu,
            execution_mode="sequential",
            label=f"threads_1x{self.ncpu}_seq",
        ))

        configs.append(ThreadConfig(
            inter_op_threads=2,
            intra_op_threads=max(1, self.ncpu // 2),
            execution_mode="parallel",
            label=f"threads_2x{max(1, self.ncpu // 2)}_par",
        ))

        configs.append(ThreadConfig(
            inter_op_threads=4,
            intra_op_threads=max(1, self.ncpu // 4),
            execution_mode="parallel",
            label=f"threads_4x{max(1, self.ncpu // 4)}_par",
        ))

        if self.ncpu >= 8:
            configs.append(ThreadConfig(
                inter_op_threads=1,
                intra_op_threads=4,
                execution_mode="sequential",
                enable_mem_pattern=True,
                label="threads_1x4_seq_lowmem",
            ))

        return configs

    def apply_to_session_options(self, config: ThreadConfig, sess_opts) -> None:
        """Apply thread config to ORT SessionOptions in-place."""
        if config.inter_op_threads > 0:
            sess_opts.inter_op_num_threads = config.inter_op_threads
        if config.intra_op_threads > 0:
            sess_opts.intra_op_num_threads = config.intra_op_threads
        if config.execution_mode == "parallel":
            import onnxruntime as ort
            sess_opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        else:
            import onnxruntime as ort
            sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_opts.enable_mem_reuse = config.enable_mem_arena
        sess_opts.enable_mem_pattern = config.enable_mem_pattern
