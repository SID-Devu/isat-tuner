"""ROCm-specific helpers (rocminfo parsing, rocm-smi wrappers)."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUAgent:
    name: str = ""
    gfx_target: str = ""
    cu_count: int = 0
    simd_count: int = 0
    max_clock_mhz: int = 0
    wavefront_size: int = 0
    lds_size_kb: int = 0
    vram_size_mb: int = 0
    gtt_size_mb: int = 0
    vendor: str = ""
    product: str = ""


def parse_rocminfo() -> Optional[GPUAgent]:
    """Parse rocminfo output and return the first GPU agent."""
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    agent = GPUAgent()
    in_gpu = False

    for line in r.stdout.splitlines():
        line = line.strip()

        if "Agent Type:" in line and "GPU" in line:
            in_gpu = True
        elif "Agent Type:" in line and "GPU" not in line:
            if in_gpu:
                break
            in_gpu = False

        if not in_gpu:
            continue

        if "Name:" in line and not agent.name:
            agent.name = line.split(":", 1)[1].strip()
        elif "Gfx Target" in line:
            m = re.search(r"gfx\w+", line)
            if m:
                agent.gfx_target = m.group(0)
        elif "Compute Unit:" in line:
            m = re.search(r"(\d+)", line.split(":", 1)[1])
            if m:
                agent.cu_count = int(m.group(1))
        elif "SIMDs per CU" in line:
            m = re.search(r"(\d+)", line.split(":", 1)[1])
            if m:
                agent.simd_count = int(m.group(1)) * agent.cu_count
        elif "Max Clock Freq" in line:
            m = re.search(r"(\d+)", line.split(":", 1)[1])
            if m:
                agent.max_clock_mhz = int(m.group(1))
        elif "Wavefront Size" in line:
            m = re.search(r"(\d+)", line.split(":", 1)[1])
            if m:
                agent.wavefront_size = int(m.group(1))
        elif "LDS" in line and "Size" in line:
            m = re.search(r"(\d+)", line.split(":", 1)[1])
            if m:
                agent.lds_size_kb = int(m.group(1))
        elif "Product Name" in line:
            agent.product = line.split(":", 1)[1].strip()
        elif "Marketing Name" in line:
            mname = line.split(":", 1)[1].strip()
            if mname and mname != "N/A" and not agent.product:
                agent.product = mname
        elif "Vendor Name" in line:
            agent.vendor = line.split(":", 1)[1].strip()

    if agent.name and not agent.gfx_target:
        m = re.search(r"gfx\w+", agent.name)
        if m:
            agent.gfx_target = m.group(0)

    return agent if (agent.gfx_target or agent.name) else None


def xnack_supported() -> bool:
    """Check if the GPU supports XNACK (demand paging)."""
    agent = parse_rocminfo()
    if not agent:
        return False
    return agent.gfx_target in {
        "gfx1151", "gfx1150", "gfx1100", "gfx1101", "gfx1102",
        "gfx90a", "gfx940", "gfx941", "gfx942",
    }


def rocm_smi_query(flag: str) -> Optional[str]:
    try:
        r = subprocess.run(["rocm-smi", flag], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
