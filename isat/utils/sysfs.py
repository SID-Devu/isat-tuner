"""Utilities for reading GPU / system state from sysfs and procfs."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional


def _read_sysfs(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return None


def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _gpu_card_path() -> Optional[Path]:
    """Find the sysfs path for the active AMD GPU (card0, card1, etc.)."""
    drm = Path("/sys/class/drm")
    if not drm.exists():
        return None
    for card in sorted(drm.glob("card[0-9]*")):
        if not (card / "device").exists():
            continue
        vendor = _read_sysfs(str(card / "device" / "vendor"))
        if vendor == "0x1002":  # AMD
            return card
    for card in sorted(drm.glob("card[0-9]*")):
        if (card / "device").exists():
            return card
    return None


def gpu_temp_edge() -> Optional[float]:
    """Read GPU edge temperature (Celsius) from hwmon."""
    for hwmon in sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*")):
        name = _read_sysfs(str(hwmon / "name"))
        if name and "amdgpu" in name:
            val = _read_sysfs(str(hwmon / "temp1_input"))
            if val:
                return int(val) / 1000.0
    return None


def gpu_power_watts() -> Optional[float]:
    for hwmon in sorted(Path("/sys/class/drm").glob("card*/device/hwmon/hwmon*")):
        val = _read_sysfs(str(hwmon / "power1_average"))
        if val:
            return int(val) / 1_000_000.0
    return None


def gpu_vram_used_mb() -> Optional[float]:
    card = _gpu_card_path()
    if not card:
        return None
    val = _read_sysfs(str(card / "device" / "mem_info_vram_used"))
    return int(val) / (1024 * 1024) if val else None


def gpu_gtt_used_mb() -> Optional[float]:
    card = _gpu_card_path()
    if not card:
        return None
    val = _read_sysfs(str(card / "device" / "mem_info_gtt_used"))
    return int(val) / (1024 * 1024) if val else None


def gpu_sclk_mhz() -> Optional[int]:
    """Current shader clock from sysfs."""
    card = _gpu_card_path()
    if not card:
        return None
    val = _read_sysfs(str(card / "device" / "pp_dpm_sclk"))
    if val:
        for line in val.splitlines():
            if "*" in line:
                m = re.search(r"(\d+)\s*Mhz", line, re.IGNORECASE)
                if m:
                    return int(m.group(1))
    return None


def system_ram_mb() -> int:
    info = Path("/proc/meminfo").read_text()
    for line in info.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) // 1024
    return 0


def kernel_version() -> str:
    return os.uname().release


def rocm_version() -> Optional[str]:
    val = _read_sysfs("/opt/rocm/.info/version")
    if val:
        return val
    out = _run(["rocm-smi", "--version"])
    if out:
        for line in out.splitlines():
            m = re.search(r"(\d+\.\d+\.\d+)", line)
            if m:
                return m.group(1)
    return None
