"""Batch size search dimension.

Explores optimal batch size for throughput vs latency tradeoff,
accounting for available GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint


@dataclass
class BatchConfig:
    batch_size: int = 1
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"batch_{self.batch_size}"


class BatchSearchDimension:
    """Generate candidate batch sizes based on model memory and GPU capacity."""

    def __init__(self, hw: HardwareFingerprint, model: ModelFingerprint):
        self.hw = hw
        self.model = model

    def candidates(self) -> list[BatchConfig]:
        configs: list[BatchConfig] = [BatchConfig(batch_size=1)]

        avail_mb = self.hw.vram_total_mb or self.hw.gtt_total_mb
        if avail_mb <= 0:
            avail_mb = self.hw.system_ram_mb * 0.5

        model_mb = max(self.model.estimated_memory_mb, 50)

        # 2x model size for activations/workspace
        usable_mb = avail_mb * 0.85
        max_batch = max(1, int(usable_mb / (model_mb * 2)))
        max_batch = min(max_batch, 256)

        batch_sizes = set()
        b = 2
        while b <= max_batch:
            batch_sizes.add(b)
            b *= 2

        for b in [4, 8, 16, 32, 64]:
            if b <= max_batch:
                batch_sizes.add(b)

        for b in sorted(batch_sizes):
            configs.append(BatchConfig(batch_size=b))

        return configs

    @staticmethod
    def estimate_max_batch(model_mb: float, gpu_mb: float, safety: float = 0.85) -> int:
        usable = gpu_mb * safety
        if model_mb <= 0:
            return 1
        return max(1, int(usable / (model_mb * 2)))
