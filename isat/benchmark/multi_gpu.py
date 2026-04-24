"""Multi-GPU distributed tuning.

Discovers available GPUs, distributes candidate configs across them,
and runs benchmarks in parallel to reduce total tuning time.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("isat.benchmark.multi_gpu")


@dataclass
class GPUDevice:
    index: int
    name: str = ""
    pci_bus: str = ""
    vram_mb: int = 0
    temp_c: float = 0.0


def discover_gpus() -> list[GPUDevice]:
    """Discover all available GPU devices."""
    devices: list[GPUDevice] = []

    try:
        r = subprocess.run(
            ["rocm-smi", "--showid", "--showproductname", "--showtemp", "--csv"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines()[1:]:
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0].strip())
                        devices.append(GPUDevice(index=idx, name=parts[1].strip() if len(parts) > 1 else ""))
                    except ValueError:
                        pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not devices:
        from pathlib import Path
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
            if (card / "gpu_busy_percent").exists():
                try:
                    idx = int(card.parent.name.replace("card", ""))
                    devices.append(GPUDevice(index=idx))
                except ValueError:
                    pass

    if not devices:
        devices.append(GPUDevice(index=0, name="default"))

    return devices


class MultiGPURunner:
    """Distribute tuning across multiple GPUs."""

    def __init__(self, gpus: list[GPUDevice] | None = None):
        self.gpus = gpus or discover_gpus()
        log.info("Multi-GPU runner: %d devices available", len(self.gpus))

    def partition_configs(self, configs: list, strategy: str = "round_robin") -> dict[int, list]:
        """Partition configs across GPUs."""
        partitions: dict[int, list] = {g.index: [] for g in self.gpus}

        if strategy == "round_robin":
            for i, config in enumerate(configs):
                gpu_idx = self.gpus[i % len(self.gpus)].index
                partitions[gpu_idx].append(config)
        elif strategy == "block":
            n = len(self.gpus)
            chunk_size = (len(configs) + n - 1) // n
            for i, gpu in enumerate(self.gpus):
                start = i * chunk_size
                end = min(start + chunk_size, len(configs))
                partitions[gpu.index] = configs[start:end]

        return partitions

    def estimate_time(
        self,
        n_configs: int,
        est_per_config_s: float = 120.0,
    ) -> dict[str, float]:
        """Estimate total tuning time for single vs multi-GPU."""
        single_time = n_configs * est_per_config_s
        multi_time = (n_configs / len(self.gpus)) * est_per_config_s

        return {
            "single_gpu_hours": single_time / 3600,
            "multi_gpu_hours": multi_time / 3600,
            "speedup": single_time / multi_time if multi_time > 0 else 1.0,
            "n_gpus": len(self.gpus),
        }

    @staticmethod
    def set_visible_device(gpu_index: int) -> dict[str, str | None]:
        """Set HIP_VISIBLE_DEVICES for a specific GPU. Returns backup."""
        backup = os.environ.get("HIP_VISIBLE_DEVICES")
        os.environ["HIP_VISIBLE_DEVICES"] = str(gpu_index)

        cuda_backup = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        return {"HIP_VISIBLE_DEVICES": backup, "CUDA_VISIBLE_DEVICES": cuda_backup}

    @staticmethod
    def restore_devices(backup: dict[str, str | None]) -> None:
        for key, val in backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
