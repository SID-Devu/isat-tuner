"""Universal hardware detector — AMD, NVIDIA, Intel, Apple, Qualcomm.

Probes every available method (sysfs, lspci, vendor CLIs, platform APIs)
and returns a normalized HardwareProfile that downstream code can act on
without caring which vendor is present.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DetectedGPU:
    vendor: str = "unknown"           # amd, nvidia, intel, apple, qualcomm
    name: str = "unknown"
    gpu_type: str = "unknown"         # igpu, dgpu, apu, integrated, soc
    arch: str = ""                    # gfx1151, sm_89, xe_lpg, m2, ...
    compute_units: int = 0
    max_clock_mhz: int = 0
    vram_mb: int = 0
    shared_mem_mb: int = 0            # GTT for AMD APU, system RAM for iGPU
    driver: str = ""
    driver_version: str = ""
    pci_id: str = ""
    supports_fp16: bool = False
    supports_int8: bool = False
    supports_bf16: bool = False
    supports_fp8: bool = False
    matrix_cores: str = ""            # MFMA, WMMA, Tensor Core, XMX, AMX, ANE


@dataclass
class HardwareProfile:
    os_name: str = ""                 # linux, windows, darwin
    os_version: str = ""
    arch: str = ""                    # x86_64, aarch64
    kernel: str = ""
    cpu_name: str = ""
    cpu_cores: int = 0
    system_ram_mb: int = 0
    swap_mb: int = 0
    gpus: list[DetectedGPU] = field(default_factory=list)
    primary_gpu: Optional[DetectedGPU] = None
    rocm_version: str = ""
    cuda_version: str = ""
    openvino_version: str = ""

    @property
    def vendor(self) -> str:
        return self.primary_gpu.vendor if self.primary_gpu else "cpu_only"

    @property
    def is_apu(self) -> bool:
        return self.primary_gpu is not None and self.primary_gpu.gpu_type in ("apu", "igpu", "integrated")

    @property
    def is_dgpu(self) -> bool:
        return self.primary_gpu is not None and self.primary_gpu.gpu_type == "dgpu"


def _run(cmd: list[str], timeout: int = 10) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return None


def _read(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except (OSError, PermissionError):
        return None


# ---------------------------------------------------------------------------
# AMD detection
# ---------------------------------------------------------------------------

def _detect_amd_rocminfo() -> Optional[DetectedGPU]:
    out = _run(["rocminfo"])
    if not out:
        return None

    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in out.splitlines():
        s = line.strip()
        if re.match(r"^Agent \d+", s):
            if cur:
                blocks.append(cur)
            cur = []
        elif s:
            cur.append(s)
    if cur:
        blocks.append(cur)

    for block in blocks:
        dev_type = ""
        for ln in block:
            if ln.startswith("Device Type:"):
                dev_type = ln.split(":", 1)[1].strip()
                break
        if "GPU" not in dev_type:
            continue

        gpu = DetectedGPU(vendor="amd", driver="amdgpu")
        for ln in block:
            if ln.startswith("Name:") and gpu.name == "unknown":
                v = ln.split(":", 1)[1].strip()
                if "ISA" not in ln and "amdgcn" not in v:
                    gpu.name = v
            elif "Marketing Name" in ln:
                mn = ln.split(":", 1)[1].strip()
                if mn and mn not in ("N/A", ""):
                    gpu.name = mn
            elif "Product Name" in ln:
                pn = ln.split(":", 1)[1].strip()
                if pn and pn != "N/A":
                    gpu.name = pn
            elif "Vendor Name" in ln:
                gpu.pci_id = ln.split(":", 1)[1].strip()
            elif "Compute Unit:" in ln:
                m = re.search(r"(\d+)", ln.split(":", 1)[1])
                if m:
                    gpu.compute_units = int(m.group(1))
            elif "Max Clock Freq" in ln:
                m = re.search(r"(\d+)", ln.split(":", 1)[1])
                if m:
                    gpu.max_clock_mhz = int(m.group(1))

        m = re.search(r"gfx\w+", gpu.name)
        if m:
            gpu.arch = m.group(0)
        elif gpu.name != "unknown":
            for ln in block:
                m = re.search(r"gfx\w+", ln)
                if m:
                    gpu.arch = m.group(0)
                    break

        _classify_amd_gpu(gpu)
        return gpu
    return None


def _classify_amd_gpu(gpu: DetectedGPU) -> None:
    arch = gpu.arch.lower()

    # VRAM / GTT for APU detection
    vram_total = 0
    gtt_total = 0
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
        if vram_total:
            break

    gpu.vram_mb = vram_total
    gpu.shared_mem_mb = gtt_total

    if vram_total < 2048 and gtt_total > 4096:
        gpu.gpu_type = "apu"
    elif vram_total == 0 and gtt_total > 0:
        gpu.gpu_type = "apu"
    elif vram_total >= 2048:
        gpu.gpu_type = "dgpu"
    else:
        gpu.gpu_type = "apu" if "115" in arch or "gfx1036" in arch else "dgpu"

    # Capabilities by architecture family
    if arch.startswith("gfx9"):
        gpu.matrix_cores = "MFMA"
        gpu.supports_fp16 = True
        gpu.supports_bf16 = True
        gpu.supports_int8 = True
        if "gfx950" in arch:
            gpu.supports_fp8 = True
    elif arch.startswith("gfx11"):
        gpu.matrix_cores = "WMMA"
        gpu.supports_fp16 = True
        gpu.supports_bf16 = True
        gpu.supports_int8 = True
    elif arch.startswith("gfx12"):
        gpu.matrix_cores = "WMMA (gfx12)"
        gpu.supports_fp16 = True
        gpu.supports_bf16 = True
        gpu.supports_int8 = True
        gpu.supports_fp8 = True

    ver = _read("/opt/rocm/.info/version")
    if ver:
        gpu.driver_version = ver


# ---------------------------------------------------------------------------
# NVIDIA detection
# ---------------------------------------------------------------------------

def _detect_nvidia() -> Optional[DetectedGPU]:
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"])
    if not out:
        return None
    parts = out.splitlines()[0].split(",")
    if len(parts) < 4:
        return None

    name = parts[0].strip()
    vram = int(float(parts[1].strip()))
    driver = parts[2].strip()
    cc = parts[3].strip()

    gpu = DetectedGPU(
        vendor="nvidia",
        name=name,
        arch=f"sm_{cc.replace('.', '')}",
        vram_mb=vram,
        driver="nvidia",
        driver_version=driver,
        supports_fp16=True,
        supports_int8=True,
    )

    cc_major = int(cc.split(".")[0]) if "." in cc else 0
    if cc_major >= 8:
        gpu.supports_bf16 = True
        gpu.matrix_cores = "Tensor Core (3rd gen+)"
    elif cc_major >= 7:
        gpu.matrix_cores = "Tensor Core"
    else:
        gpu.matrix_cores = "CUDA Cores"

    if cc_major >= 9:
        gpu.supports_fp8 = True
        gpu.matrix_cores = "Tensor Core (4th gen)"

    # iGPU detection: NVIDIA doesn't have iGPUs in mainstream, but Jetson is SoC
    if "tegra" in name.lower() or "orin" in name.lower() or "jetson" in name.lower():
        gpu.gpu_type = "soc"
        gpu.shared_mem_mb = _system_ram_mb()
    else:
        gpu.gpu_type = "dgpu"

    # Get CU count
    cu_out = _run(["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"])
    return gpu


def _detect_nvidia_cuda_version() -> str:
    out = _run(["nvcc", "--version"])
    if out:
        m = re.search(r"release (\d+\.\d+)", out)
        if m:
            return m.group(1)
    out = _run(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Intel detection
# ---------------------------------------------------------------------------

def _detect_intel_gpu() -> Optional[DetectedGPU]:
    lspci = _run(["lspci", "-nn"])
    if not lspci:
        return None

    for line in lspci.splitlines():
        if "VGA" in line or "Display" in line or "3D" in line:
            low = line.lower()
            if "intel" in low:
                gpu = DetectedGPU(vendor="intel", driver="i915")
                # Extract name
                m = re.search(r'Intel.*?(?:\[([^\]]+)\]|$)', line)
                if m and m.group(1):
                    gpu.name = m.group(1)
                else:
                    gpu.name = re.sub(r'^\S+\s+', '', line.split('[')[0]).strip()

                # Determine type and capabilities
                if any(k in low for k in ["arc", "a770", "a750", "a580", "a380", "b580"]):
                    gpu.gpu_type = "dgpu"
                    gpu.arch = "Xe-HPG" if "arc" in low else "Xe2"
                    gpu.matrix_cores = "XMX (Xe Matrix eXtensions)"
                    gpu.supports_fp16 = True
                    gpu.supports_bf16 = True
                    gpu.supports_int8 = True
                elif any(k in low for k in ["meteor", "arrow", "lunar", "battlemage"]):
                    gpu.gpu_type = "igpu"
                    gpu.arch = "Xe-LPG"
                    gpu.matrix_cores = "XMX (Xe Matrix eXtensions)"
                    gpu.supports_fp16 = True
                    gpu.supports_int8 = True
                else:
                    gpu.gpu_type = "igpu"
                    gpu.arch = "Gen12/Xe"
                    gpu.supports_fp16 = True
                    gpu.supports_int8 = True

                ver = _run(["intel_gpu_top", "-L"])
                return gpu
    return None


# ---------------------------------------------------------------------------
# Apple detection
# ---------------------------------------------------------------------------

def _detect_apple() -> Optional[DetectedGPU]:
    if platform.system() != "Darwin":
        return None

    chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if not chip or "Apple" not in chip:
        return None

    gpu = DetectedGPU(
        vendor="apple",
        name=chip.strip(),
        gpu_type="soc",
        driver="Metal",
        supports_fp16=True,
        supports_bf16=True,
        supports_int8=True,
    )

    if "M1" in chip:
        gpu.arch = "M1"
        gpu.matrix_cores = "AMX (Apple Matrix eXtensions)"
    elif "M2" in chip:
        gpu.arch = "M2"
        gpu.matrix_cores = "AMX (Apple Matrix eXtensions)"
    elif "M3" in chip:
        gpu.arch = "M3"
        gpu.matrix_cores = "AMX (Apple Matrix eXtensions)"
    elif "M4" in chip:
        gpu.arch = "M4"
        gpu.matrix_cores = "AMX (Apple Matrix eXtensions)"
    elif "M5" in chip:
        gpu.arch = "M5"
        gpu.matrix_cores = "AMX (Apple Matrix eXtensions)"

    gpu.shared_mem_mb = _system_ram_mb()
    gpu.vram_mb = 0
    return gpu


# ---------------------------------------------------------------------------
# Qualcomm detection
# ---------------------------------------------------------------------------

def _detect_qualcomm() -> Optional[DetectedGPU]:
    # Check for Qualcomm via /proc/cpuinfo (Snapdragon) or lspci (Qualcomm NPU)
    cpuinfo = _read("/proc/cpuinfo")
    if cpuinfo and "qualcomm" in cpuinfo.lower():
        gpu = DetectedGPU(
            vendor="qualcomm",
            name="Qualcomm Adreno / Hexagon NPU",
            gpu_type="soc",
            arch="Hexagon",
            driver="QNN",
            matrix_cores="Hexagon HVX/HTP",
            supports_fp16=True,
            supports_int8=True,
        )
        gpu.shared_mem_mb = _system_ram_mb()
        return gpu

    lspci = _run(["lspci", "-nn"])
    if lspci and "qualcomm" in lspci.lower():
        gpu = DetectedGPU(
            vendor="qualcomm",
            name="Qualcomm Compute Platform",
            gpu_type="soc",
            arch="Snapdragon X",
            driver="QNN",
            matrix_cores="Hexagon NPU",
            supports_fp16=True,
            supports_int8=True,
        )
        gpu.shared_mem_mb = _system_ram_mb()
        return gpu

    return None


# ---------------------------------------------------------------------------
# Windows GPU detection via WMI / PowerShell
# ---------------------------------------------------------------------------

def _detect_windows_gpu() -> Optional[DetectedGPU]:
    """Detect GPU on Windows via PowerShell (Get-CimInstance Win32_VideoController)."""
    if platform.system() != "Windows":
        return None

    out = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,VideoProcessor,PNPDeviceID | "
        "Format-List"
    ], timeout=15)
    if not out:
        return None

    name = ""
    vram_bytes = 0
    driver_ver = ""
    pnp_id = ""
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Name"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("AdapterRAM"):
            try:
                vram_bytes = int(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("DriverVersion"):
            driver_ver = line.split(":", 1)[1].strip()
        elif line.startswith("PNPDeviceID"):
            pnp_id = line.split(":", 1)[1].strip()

    if not name:
        return None

    low = name.lower()
    gpu = DetectedGPU(name=name, driver_version=driver_ver, pci_id=pnp_id)
    gpu.vram_mb = vram_bytes // (1024 * 1024) if vram_bytes > 0 else 0

    if "amd" in low or "radeon" in low or "ati" in low:
        gpu.vendor = "amd"
        gpu.driver = "amdgpu-pro / WinML"
        gpu.supports_fp16 = True
        gpu.supports_int8 = True
        if any(k in low for k in ["780m", "760m", "890m", "8060", "radeon graphics"]):
            gpu.gpu_type = "apu"
            gpu.arch = "RDNA 3.5"
            gpu.matrix_cores = "WMMA"
            gpu.supports_bf16 = True
            gpu.shared_mem_mb = _system_ram_mb()
        elif any(k in low for k in ["7900", "7800", "7700", "7600", "6900", "6800", "6700"]):
            gpu.gpu_type = "dgpu"
            gpu.arch = "RDNA 3" if "7" in name else "RDNA 2"
            gpu.matrix_cores = "WMMA"
            gpu.supports_bf16 = True
        else:
            gpu.gpu_type = "dgpu"
            gpu.arch = "RDNA"
            gpu.matrix_cores = "WMMA"
    elif "nvidia" in low or "geforce" in low or "rtx" in low or "gtx" in low or "quadro" in low:
        gpu.vendor = "nvidia"
        gpu.driver = "nvidia"
        gpu.supports_fp16 = True
        gpu.supports_int8 = True
        if any(k in low for k in ["40", "rtx 50", "a100", "h100", "l40"]):
            gpu.matrix_cores = "Tensor Core (4th gen)"
            gpu.supports_bf16 = True
            gpu.supports_fp8 = True
            gpu.arch = "Ada Lovelace"
        elif any(k in low for k in ["30", "a6000", "a5000"]):
            gpu.matrix_cores = "Tensor Core (3rd gen)"
            gpu.supports_bf16 = True
            gpu.arch = "Ampere"
        elif "20" in low or "rtx" in low:
            gpu.matrix_cores = "Tensor Core"
            gpu.arch = "Turing"
        else:
            gpu.matrix_cores = "CUDA Cores"
            gpu.arch = "NVIDIA"
        gpu.gpu_type = "dgpu"
    elif "intel" in low:
        gpu.vendor = "intel"
        gpu.driver = "Intel Graphics"
        gpu.supports_fp16 = True
        gpu.supports_int8 = True
        if any(k in low for k in ["arc", "a770", "a750", "a580", "a380", "b580"]):
            gpu.gpu_type = "dgpu"
            gpu.arch = "Xe-HPG"
            gpu.matrix_cores = "XMX"
        else:
            gpu.gpu_type = "igpu"
            gpu.arch = "Xe/Gen12"
            gpu.shared_mem_mb = _system_ram_mb()
    elif "qualcomm" in low or "adreno" in low:
        gpu.vendor = "qualcomm"
        gpu.driver = "Qualcomm"
        gpu.gpu_type = "soc"
        gpu.arch = "Adreno"
        gpu.supports_fp16 = True
        gpu.supports_int8 = True
        gpu.shared_mem_mb = _system_ram_mb()
    else:
        gpu.vendor = "unknown"
        gpu.gpu_type = "dgpu"

    return gpu


# ---------------------------------------------------------------------------
# Fallback: lspci scan for any GPU
# ---------------------------------------------------------------------------

def _detect_lspci_fallback() -> Optional[DetectedGPU]:
    lspci = _run(["lspci", "-nn"])
    if not lspci:
        return None

    for line in lspci.splitlines():
        if "VGA" in line or "Display" in line or "3D" in line:
            low = line.lower()
            gpu = DetectedGPU()
            if "amd" in low or "ati" in low or "radeon" in low:
                gpu.vendor = "amd"
                gpu.driver = "amdgpu"
            elif "nvidia" in low:
                gpu.vendor = "nvidia"
                gpu.driver = "nvidia"
            elif "intel" in low:
                gpu.vendor = "intel"
                gpu.driver = "i915"
            else:
                continue
            gpu.name = re.sub(r'^\S+\s+', '', line.split('[')[0]).strip()
            gpu.gpu_type = "dgpu"
            gpu.supports_fp16 = True
            return gpu
    return None


# ---------------------------------------------------------------------------
# System-level helpers
# ---------------------------------------------------------------------------

def _system_ram_mb() -> int:
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        if out:
            try:
                return int(out.strip()) // (1024 * 1024)
            except ValueError:
                pass
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            return int(out) // (1024 * 1024)
    try:
        info = Path("/proc/meminfo").read_text()
        for line in info.splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _swap_mb() -> int:
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_PageFileUsage | Measure-Object -Property AllocatedBaseSize -Sum).Sum"])
        if out:
            try:
                return int(out.strip())
            except ValueError:
                pass
    try:
        info = Path("/proc/meminfo").read_text()
        for line in info.splitlines():
            if line.startswith("SwapTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _cpu_name() -> str:
    if platform.system() == "Windows":
        out = _run(["powershell", "-NoProfile", "-Command",
                     "(Get-CimInstance Win32_Processor).Name"])
        if out:
            return out.strip()
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        return out.strip() if out else platform.processor()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_hardware() -> HardwareProfile:
    """Detect all hardware on this system — works on any OS, any vendor."""
    profile = HardwareProfile(
        os_name=platform.system().lower(),
        os_version=platform.version(),
        arch=platform.machine(),
        kernel=os.uname().release if hasattr(os, "uname") else "",
        cpu_name=_cpu_name(),
        cpu_cores=os.cpu_count() or 0,
        system_ram_mb=_system_ram_mb(),
        swap_mb=_swap_mb(),
    )

    is_windows = platform.system() == "Windows"

    # Try vendor-specific detection in priority order
    amd = _detect_amd_rocminfo()
    if amd:
        profile.gpus.append(amd)
        profile.rocm_version = amd.driver_version

    nvidia = _detect_nvidia()
    if nvidia:
        profile.gpus.append(nvidia)
        profile.cuda_version = _detect_nvidia_cuda_version()

    intel = _detect_intel_gpu()
    if intel:
        profile.gpus.append(intel)

    apple = _detect_apple()
    if apple:
        profile.gpus.append(apple)

    qualcomm = _detect_qualcomm()
    if qualcomm:
        profile.gpus.append(qualcomm)

    # Windows: use WMI/PowerShell if no GPU found via CLI tools
    if not profile.gpus and is_windows:
        win_gpu = _detect_windows_gpu()
        if win_gpu:
            profile.gpus.append(win_gpu)

    # If nothing found via vendor tools, try lspci fallback (Linux)
    if not profile.gpus:
        fb = _detect_lspci_fallback()
        if fb:
            profile.gpus.append(fb)

    # Pick primary GPU (prefer discrete > APU > integrated > SoC)
    if profile.gpus:
        priority = {"dgpu": 0, "apu": 1, "igpu": 2, "integrated": 3, "soc": 4, "unknown": 5}
        profile.gpus.sort(key=lambda g: priority.get(g.gpu_type, 5))
        profile.primary_gpu = profile.gpus[0]

    return profile
