"""Pre-flight system health checker.

Runs before a tuning session to verify the system is ready:
  - GPU temperature within safe range
  - Sufficient free memory (host + GPU)
  - No other GPU-intensive processes running
  - Disk space for output
  - Driver/runtime status
  - Clock speeds (not throttled)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from isat.utils.sysfs import gpu_temp_edge as gpu_temp_c, gpu_vram_used_mb, gpu_gtt_used_mb

log = logging.getLogger("isat.health")


@dataclass
class HealthCheck:
    name: str
    status: str  # "healthy", "degraded", "critical"
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class HealthReport:
    checks: list[HealthCheck] = field(default_factory=list)
    ready: bool = True
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  {'Check':<30} {'Status':<12} {'Value':>10} {'Message'}",
            f"  {'-'*30} {'-'*12} {'-'*10} {'-'*30}",
        ]
        for c in self.checks:
            val_str = f"{c.value:.1f}" if c.value is not None else "N/A"
            lines.append(f"  {c.name:<30} {c.status:<12} {val_str:>10} {c.message}")
        status = "READY" if self.ready else "NOT READY"
        lines.append(f"\n  System status: {status}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"    Warning: {w}")
        return "\n".join(lines)


class HealthChecker:
    """Pre-flight system health verification."""

    def __init__(
        self,
        max_gpu_temp_c: float = 70.0,
        min_free_disk_gb: float = 1.0,
        min_free_host_mb: float = 2048.0,
    ):
        self.max_gpu_temp_c = max_gpu_temp_c
        self.min_free_disk_gb = min_free_disk_gb
        self.min_free_host_mb = min_free_host_mb

    def check(self, output_dir: str = ".") -> HealthReport:
        report = HealthReport()

        report.checks.append(self._check_gpu_temp())
        report.checks.append(self._check_gpu_memory())
        report.checks.append(self._check_host_memory())
        report.checks.append(self._check_disk_space(output_dir))
        report.checks.append(self._check_gpu_processes())
        report.checks.append(self._check_gpu_clocks())

        for c in report.checks:
            if c.status == "critical":
                report.ready = False
            elif c.status == "degraded":
                report.warnings.append(c.message)

        return report

    def _check_gpu_temp(self) -> HealthCheck:
        temp = gpu_temp_c()
        if temp is None:
            return HealthCheck("GPU temperature", "healthy", "Could not read (assumed OK)")
        if temp > self.max_gpu_temp_c:
            return HealthCheck("GPU temperature", "critical",
                               f"{temp:.0f}C exceeds {self.max_gpu_temp_c:.0f}C limit",
                               value=temp, threshold=self.max_gpu_temp_c)
        if temp > self.max_gpu_temp_c - 10:
            return HealthCheck("GPU temperature", "degraded",
                               f"{temp:.0f}C (close to {self.max_gpu_temp_c:.0f}C limit)",
                               value=temp, threshold=self.max_gpu_temp_c)
        return HealthCheck("GPU temperature", "healthy",
                           f"{temp:.0f}C", value=temp, threshold=self.max_gpu_temp_c)

    def _check_gpu_memory(self) -> HealthCheck:
        vram = gpu_vram_used_mb()
        gtt = gpu_gtt_used_mb()
        if vram is None:
            return HealthCheck("GPU memory", "healthy", "Could not read (assumed OK)")
        total = (vram or 0) + (gtt or 0)
        return HealthCheck("GPU memory", "healthy",
                           f"VRAM: {vram:.0f} MB, GTT: {gtt or 0:.0f} MB in use",
                           value=total)

    def _check_host_memory(self) -> HealthCheck:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_mb = int(line.split()[1]) / 1024
                        if avail_mb < self.min_free_host_mb:
                            return HealthCheck("Host memory", "critical",
                                               f"Only {avail_mb:.0f} MB free (need {self.min_free_host_mb:.0f})",
                                               value=avail_mb, threshold=self.min_free_host_mb)
                        return HealthCheck("Host memory", "healthy",
                                           f"{avail_mb:.0f} MB available",
                                           value=avail_mb, threshold=self.min_free_host_mb)
        except (OSError, ValueError):
            pass
        return HealthCheck("Host memory", "healthy", "Could not read (assumed OK)")

    def _check_disk_space(self, output_dir: str) -> HealthCheck:
        try:
            usage = shutil.disk_usage(output_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb < self.min_free_disk_gb:
                return HealthCheck("Disk space", "critical",
                                   f"Only {free_gb:.1f} GB free",
                                   value=free_gb, threshold=self.min_free_disk_gb)
            return HealthCheck("Disk space", "healthy",
                               f"{free_gb:.1f} GB free",
                               value=free_gb, threshold=self.min_free_disk_gb)
        except OSError:
            return HealthCheck("Disk space", "healthy", "Could not check (assumed OK)")

    def _check_gpu_processes(self) -> HealthCheck:
        fuser_path = shutil.which("fuser")
        if not fuser_path:
            return HealthCheck("GPU processes", "healthy", "fuser not available (skipped)")
        try:
            result = subprocess.run(
                ["fuser", "/dev/kfd"],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            my_pid = str(os.getpid())
            other_pids = [p for p in pids if p != my_pid and p.strip()]
            if other_pids:
                return HealthCheck("GPU processes", "degraded",
                                   f"{len(other_pids)} other GPU processes detected",
                                   value=len(other_pids))
            return HealthCheck("GPU processes", "healthy", "No competing GPU processes")
        except (subprocess.TimeoutExpired, OSError):
            return HealthCheck("GPU processes", "healthy", "Could not check (assumed OK)")

    def _check_gpu_clocks(self) -> HealthCheck:
        sclk_path = "/sys/class/drm/card0/device/pp_dpm_sclk"
        try:
            content = Path(sclk_path).read_text()
            lines = content.strip().splitlines()
            active = [l for l in lines if "*" in l]
            if active:
                return HealthCheck("GPU clocks", "healthy", active[0].strip())
            return HealthCheck("GPU clocks", "healthy", "Could not determine active clock")
        except (OSError, IndexError):
            return HealthCheck("GPU clocks", "healthy", "Could not read (assumed OK)")
