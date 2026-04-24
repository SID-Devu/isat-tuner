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
    """Parse rocminfo output and return the first GPU agent.

    rocminfo outputs agent blocks where Name/Marketing Name appear BEFORE
    the Device Type line, so we collect each agent block first, then check
    whether it's a GPU agent.
    """
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in r.stdout.splitlines():
        stripped = line.strip()
        if re.match(r"^Agent \d+", stripped):
            if current:
                blocks.append(current)
            current = []
        elif stripped:
            current.append(stripped)
    if current:
        blocks.append(current)

    for block in blocks:
        device_type = ""
        for ln in block:
            if ln.startswith("Device Type:"):
                device_type = ln.split(":", 1)[1].strip()
                break
        if "GPU" not in device_type:
            continue

        agent = GPUAgent()
        for ln in block:
            if ln.startswith("Name:") and not agent.name:
                val = ln.split(":", 1)[1].strip()
                if "ISA" not in ln and "amdgcn" not in val:
                    agent.name = val
            elif ln.startswith("Marketing Name"):
                mname = ln.split(":", 1)[1].strip()
                if mname and mname not in ("N/A", ""):
                    agent.product = mname
            elif ln.startswith("Vendor Name"):
                agent.vendor = ln.split(":", 1)[1].strip()
            elif "Compute Unit:" in ln:
                m = re.search(r"(\d+)", ln.split(":", 1)[1])
                if m:
                    agent.cu_count = int(m.group(1))
            elif "SIMDs per CU" in ln:
                m = re.search(r"(\d+)", ln.split(":", 1)[1])
                if m:
                    agent.simd_count = int(m.group(1)) * max(agent.cu_count, 1)
            elif "Max Clock Freq" in ln:
                m = re.search(r"(\d+)", ln.split(":", 1)[1])
                if m:
                    agent.max_clock_mhz = int(m.group(1))
            elif "Wavefront Size" in ln:
                m = re.search(r"(\d+)", ln.split(":", 1)[1])
                if m:
                    agent.wavefront_size = int(m.group(1))
            elif "Product Name" in ln:
                pname = ln.split(":", 1)[1].strip()
                if pname and pname != "N/A":
                    agent.product = pname

        if not agent.gfx_target:
            m = re.search(r"gfx\w+", agent.name)
            if m:
                agent.gfx_target = m.group(0)

        if agent.gfx_target or agent.name:
            return agent

    return None


def xnack_supported() -> bool:
    """Check if the GPU supports XNACK (demand paging)."""
    agent = parse_rocminfo()
    if not agent:
        return False
    return agent.gfx_target in {
        "gfx1151", "gfx1150", "gfx1100", "gfx1101", "gfx1102",
        "gfx90a", "gfx940", "gfx941", "gfx942",
    }


def xnack_enabled() -> bool:
    """Check if XNACK is currently enabled (HSA_XNACK=1 or system default)."""
    import os
    env_val = os.environ.get("HSA_XNACK", "")
    if env_val == "1":
        return True
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=15)
        for line in r.stdout.splitlines():
            if "XNACK enabled" in line:
                return "YES" in line.upper()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def rocm_smi_query(flag: str) -> Optional[str]:
    try:
        r = subprocess.run(["rocm-smi", flag], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
