"""Hardware fingerprinting -- detect GPU capabilities, memory topology, and system config."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

from isat.utils.rocm import GPUAgent, parse_rocminfo, xnack_enabled, xnack_supported
from isat.utils.sysfs import (
    gpu_gtt_used_mb,
    gpu_sclk_mhz,
    gpu_temp_edge,
    gpu_vram_used_mb,
    kernel_version,
    rocm_version,
    system_ram_mb,
)


@dataclass
class HardwareFingerprint:
    gpu_name: str = "unknown"
    gfx_target: str = "unknown"
    cu_count: int = 0
    simd_count: int = 0
    max_clock_mhz: int = 0
    wavefront_size: int = 64
    lds_size_kb: int = 0
    vendor: str = ""
    product: str = ""

    vram_total_mb: int = 0
    gtt_total_mb: int = 0
    system_ram_mb: int = 0
    is_apu: bool = False
    unified_memory: bool = False
    xnack_supported: bool = False
    xnack_enabled: bool = False

    kernel_version: str = ""
    rocm_version: str = ""
    driver: str = "amdgpu"

    gpu_temp_c: Optional[float] = None
    gpu_sclk_mhz: Optional[int] = None

    fingerprint_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @property
    def memory_class(self) -> str:
        """Classify memory topology for search space selection."""
        if self.is_apu and self.unified_memory:
            return "apu_unified"
        if self.vram_total_mb > 16384:
            return "dgpu_large"
        return "dgpu_small"


def fingerprint_hardware() -> HardwareFingerprint:
    fp = HardwareFingerprint()

    agent = parse_rocminfo()
    if agent:
        fp.gpu_name = agent.product or agent.name or "unknown"
        fp.gfx_target = agent.gfx_target or "unknown"
        fp.cu_count = agent.cu_count
        fp.simd_count = agent.simd_count
        fp.max_clock_mhz = agent.max_clock_mhz
        fp.wavefront_size = agent.wavefront_size
        fp.lds_size_kb = agent.lds_size_kb
        fp.vendor = agent.vendor
        fp.product = agent.product

    fp.system_ram_mb = system_ram_mb()
    fp.kernel_version = kernel_version()
    fp.rocm_version = rocm_version() or "unknown"
    fp.xnack_supported = xnack_supported()
    fp.xnack_enabled = xnack_enabled()

    fp.gpu_temp_c = gpu_temp_edge()
    fp.gpu_sclk_mhz = gpu_sclk_mhz()

    _detect_memory_topology(fp)

    identity = f"{fp.gfx_target}:{fp.cu_count}:{fp.max_clock_mhz}:{fp.vram_total_mb}:{fp.gtt_total_mb}"
    fp.fingerprint_hash = hashlib.sha256(identity.encode()).hexdigest()[:16]

    return fp


def _detect_memory_topology(fp: HardwareFingerprint) -> None:
    """Detect VRAM/GTT sizes and whether this is an APU with unified memory."""
    from pathlib import Path

    vram_total = None
    gtt_total = None
    for card in sorted(Path("/sys/class/drm").glob("card*/device")):
        vt = card / "mem_info_vram_total"
        gt = card / "mem_info_gtt_total"
        if vt.exists():
            try:
                vram_total = int(vt.read_text().strip()) // (1024 * 1024)
            except (OSError, ValueError):
                pass
        if gt.exists():
            try:
                gtt_total = int(gt.read_text().strip()) // (1024 * 1024)
            except (OSError, ValueError):
                pass
        if vram_total is not None:
            break

    fp.vram_total_mb = vram_total or 0
    fp.gtt_total_mb = gtt_total or 0

    if fp.vram_total_mb > 0 and fp.vram_total_mb < 2048 and fp.gtt_total_mb > 4096:
        fp.is_apu = True
        fp.unified_memory = True
    elif fp.vram_total_mb == 0 and fp.gtt_total_mb > 0:
        fp.is_apu = True
        fp.unified_memory = True
