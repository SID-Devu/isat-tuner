"""Inference recommendation engine.

Given a HardwareProfile + optional model info, generates copy-paste-ready
setup commands and runtime configuration for ANY vendor's hardware.

Knowledge base sourced from:
  - AMD APU tuning (23 models on gfx1151, MIGraphX + FP16 + HSA_XNACK)
  - NVIDIA best practices for TensorRT / CUDA EP
  - Intel OpenVINO optimization guide
  - Apple CoreML / Metal Performance Shaders docs
  - Qualcomm QNN SDK documentation
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from isat.auto_detect.detector import DetectedGPU, HardwareProfile


@dataclass
class InferenceRecipe:
    """A complete, actionable inference configuration."""
    title: str = ""
    provider: str = ""
    provider_options: dict = field(default_factory=dict)
    session_options: dict = field(default_factory=dict)
    env_vars: dict = field(default_factory=dict)
    install_cmd: str = ""
    setup_steps: list[str] = field(default_factory=list)
    python_code: str = ""
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    estimated_memory_gb: float = 0.0


@dataclass
class TuneReport:
    """Full output of `isat tune`."""
    hardware: HardwareProfile = field(default_factory=HardwareProfile)
    model_path: str = ""
    model_size_mb: float = 0.0
    model_params: int = 0
    model_class: str = ""
    recipes: list[InferenceRecipe] = field(default_factory=list)
    system_ok: bool = True
    system_warnings: list[str] = field(default_factory=list)


def _model_size_mb(model_path: str) -> float:
    p = Path(model_path)
    if not p.exists():
        return 0.0
    total = p.stat().st_size
    seen = {p.resolve()}
    # External data can be a directory OR a single file
    for suffix in [".onnx_data", "_data"]:
        data_path = p.parent / (p.stem + suffix)
        if data_path.exists() and data_path.resolve() not in seen:
            seen.add(data_path.resolve())
            if data_path.is_dir():
                total += sum(f.stat().st_size for f in data_path.iterdir())
            else:
                total += data_path.stat().st_size
    # Also scan for ANY *.onnx_data or *.onnx.data or *.data in same dir
    for pat in ["*.onnx_data", "*.onnx.data", "*.data"]:
        for ed in p.parent.glob(pat):
            if ed.resolve() not in seen and ed.is_file():
                seen.add(ed.resolve())
                total += ed.stat().st_size
    return total / (1024 * 1024)


def _estimate_runtime_mem(model_mb: float) -> float:
    """Rough estimate: model weights + ~2x for activations/workspace."""
    return model_mb * 2.5


# ---------------------------------------------------------------------------
# AMD recipes
# ---------------------------------------------------------------------------

def _amd_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                 model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024
    is_apu = gpu.gpu_type == "apu"
    is_large = model_mb > 500

    # --- Recipe 1: MIGraphX EP (best performance) ---
    env = {
        "HSA_XNACK": "1",
    }
    if is_apu:
        env["MIGRAPHX_GPU_COMPILE_PARALLEL"] = "1"
    if is_large:
        env["MIGRAPHX_GPU_HIP_FLAGS"] = "-ftemplate-depth=2048 -Wno-error=stack-exhausted"

    provider_opts = {
        "migraphx_fp16_enable": "1",
        "device_id": "0",
    }

    setup = []
    warnings = []
    notes = []

    if is_apu:
        notes.append(f"APU detected ({gpu.name}): unified memory, {gpu.vram_mb}MB VRAM + {gpu.shared_mem_mb}MB GTT")
        notes.append("MIGraphX uses hipMallocManaged with HSA_XNACK=1 for large model support")
        if runtime_gb > 8:
            setup.append(f"Ensure swap >= {int(runtime_gb * 4)}GB for MIGraphX compilation")
            setup.append("  sudo fallocate -l 100G /swapfile2 && sudo chmod 600 /swapfile2")
            setup.append("  sudo mkswap /swapfile2 && sudo swapon /swapfile2")
        setup.append("Verify kernel boot params (for APU large model support):")
        setup.append("  GRUB: amdgpu.gttsize=28672 amdgpu.no_system_mem_limit=1")
        if model_mb > 2000:
            warnings.append("Model >2GB: MIGraphX compilation may take 10-60 minutes and use heavy swap")
            warnings.append("Use subprocess isolation (run each model in a separate process)")
    else:
        notes.append(f"dGPU detected ({gpu.name}): {gpu.vram_mb}MB dedicated VRAM")
        if model_mb > gpu.vram_mb * 0.7:
            warnings.append(f"Model ({model_mb:.0f}MB) may not fit in VRAM ({gpu.vram_mb}MB)")

    python_code = f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

providers = [
    ("MIGraphXExecutionProvider", {{
        "migraphx_fp16_enable": "1",
        "device_id": "0",
    }}),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("{model_path}", sess_opts, providers=providers)

# CRITICAL: verify MIGraphX is actually active (ORT silently falls back to CPU)
active_ep = session.get_providers()[0]
assert "MIGraphX" in active_ep, f"Expected MIGraphX but got {{active_ep}}"
print(f"Running on: {{active_ep}}")
"""

    recipes.append(InferenceRecipe(
        title="AMD MIGraphX EP (Best Performance)",
        provider="MIGraphXExecutionProvider",
        provider_options=provider_opts,
        session_options={"graph_optimization_level": "ORT_DISABLE_ALL"},
        env_vars=env,
        install_cmd="pip install onnxruntime-migraphx" if not is_apu else
                    "# Custom ORT build required for APU — see ISAT docs\npip install onnxruntime-migraphx",
        setup_steps=setup,
        python_code=python_code,
        warnings=warnings,
        notes=notes,
        estimated_memory_gb=runtime_gb,
    ))

    # --- Recipe 2: ROCm EP (fallback) ---
    recipes.append(InferenceRecipe(
        title="AMD ROCm EP (Fallback — if MIGraphX unavailable)",
        provider="ROCMExecutionProvider",
        provider_options={"device_id": "0"},
        session_options={"graph_optimization_level": "ORT_ENABLE_ALL"},
        env_vars={"HSA_XNACK": "1"} if is_apu else {},
        install_cmd="pip install onnxruntime-rocm",
        notes=["ROCm EP uses hipBLAS directly — simpler but no graph compilation/fusion"],
        estimated_memory_gb=runtime_gb,
    ))

    return recipes


# ---------------------------------------------------------------------------
# NVIDIA recipes
# ---------------------------------------------------------------------------

def _nvidia_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                    model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024

    # --- Recipe 1: TensorRT EP ---
    env = {}
    if gpu.supports_fp16:
        env["ORT_TENSORRT_FP16_ENABLE"] = "1"

    python_code = f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()

providers = [
    ("TensorrtExecutionProvider", {{
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
    }}),
    ("CUDAExecutionProvider", {{"device_id": 0}}),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("{model_path}", sess_opts, providers=providers)
print(f"Running on: {{session.get_providers()[0]}}")
"""

    recipes.append(InferenceRecipe(
        title="NVIDIA TensorRT EP (Best Performance)",
        provider="TensorrtExecutionProvider",
        provider_options={
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./trt_cache",
        },
        session_options={},
        env_vars=env,
        install_cmd="pip install onnxruntime-gpu",
        setup_steps=[
            "Install CUDA toolkit matching your driver version",
            "Install TensorRT (apt install tensorrt or pip install tensorrt)",
        ],
        python_code=python_code,
        notes=[
            f"dGPU detected ({gpu.name}): {gpu.vram_mb}MB VRAM, {gpu.arch}",
            "TensorRT compiles the model into an optimized engine (first run is slow)",
            "Engine cache avoids recompilation on subsequent runs",
        ],
        warnings=[f"Model ({model_mb:.0f}MB) vs VRAM ({gpu.vram_mb}MB)"] if model_mb > gpu.vram_mb * 0.7 else [],
        estimated_memory_gb=runtime_gb,
    ))

    # --- Recipe 2: CUDA EP ---
    recipes.append(InferenceRecipe(
        title="NVIDIA CUDA EP (Simpler Setup)",
        provider="CUDAExecutionProvider",
        provider_options={"device_id": 0},
        session_options={"graph_optimization_level": "ORT_ENABLE_ALL"},
        env_vars={},
        install_cmd="pip install onnxruntime-gpu",
        notes=["CUDA EP uses cuDNN/cuBLAS directly — no compilation step, faster cold start"],
        estimated_memory_gb=runtime_gb,
    ))

    # --- Recipe 3: Jetson / SoC ---
    if gpu.gpu_type == "soc":
        recipes.insert(0, InferenceRecipe(
            title="NVIDIA Jetson (SoC — Unified Memory)",
            provider="TensorrtExecutionProvider",
            provider_options={"trt_fp16_enable": True},
            install_cmd="# Use NVIDIA JetPack SDK — ORT is pre-built for Jetson",
            notes=[
                "Jetson uses unified memory (CPU + GPU share RAM)",
                "Use FP16 to halve memory footprint",
                "Consider INT8 quantization for best perf on Jetson",
            ],
            estimated_memory_gb=runtime_gb,
        ))

    return recipes


# ---------------------------------------------------------------------------
# Intel recipes
# ---------------------------------------------------------------------------

def _intel_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                   model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024
    is_igpu = gpu.gpu_type in ("igpu", "integrated")

    # --- Recipe 1: OpenVINO EP ---
    python_code = f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()

providers = [
    ("OpenVINOExecutionProvider", {{
        "device_type": "{'GPU' if not is_igpu else 'GPU.0'}",
        "precision": "FP16",
    }}),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("{model_path}", sess_opts, providers=providers)
print(f"Running on: {{session.get_providers()[0]}}")
"""

    notes = [f"Intel {'iGPU' if is_igpu else 'dGPU'} detected ({gpu.name})"]
    if is_igpu:
        notes.append("iGPU shares system RAM — no dedicated VRAM")
        notes.append(f"System RAM: {hw.system_ram_mb // 1024}GB available")
    else:
        notes.append(f"Intel Arc dGPU: {gpu.arch}")

    recipes.append(InferenceRecipe(
        title="Intel OpenVINO EP (Best Performance)",
        provider="OpenVINOExecutionProvider",
        provider_options={"device_type": "GPU", "precision": "FP16"},
        session_options={},
        env_vars={},
        install_cmd="pip install onnxruntime-openvino",
        setup_steps=["Install Intel GPU drivers: apt install intel-opencl-icd"],
        python_code=python_code,
        notes=notes,
        estimated_memory_gb=runtime_gb,
    ))

    # --- Recipe 2: CPU with INT8 ---
    cpu_threads = hw.cpu_cores
    recipes.append(InferenceRecipe(
        title="Intel CPU (INT8 Quantized — No GPU Required)",
        provider="CPUExecutionProvider",
        session_options={
            "intra_op_num_threads": str(cpu_threads),
            "graph_optimization_level": "ORT_ENABLE_ALL",
        },
        env_vars={"OMP_NUM_THREADS": str(cpu_threads)},
        install_cmd="pip install onnxruntime",
        notes=[
            f"CPU: {hw.cpu_name} ({cpu_threads} cores)",
            "Consider INT8 quantization: isat optimize model.onnx --int8",
            "Intel AMX (Advanced Matrix eXtensions) accelerates INT8/BF16 on 4th/5th Gen Xeon",
        ],
        estimated_memory_gb=runtime_gb,
    ))

    return recipes


# ---------------------------------------------------------------------------
# Apple recipes
# ---------------------------------------------------------------------------

def _apple_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                   model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024

    python_code = f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()

providers = [
    ("CoreMLExecutionProvider", {{}}),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("{model_path}", sess_opts, providers=providers)
print(f"Running on: {{session.get_providers()[0]}}")
"""

    recipes.append(InferenceRecipe(
        title=f"Apple {gpu.arch} CoreML EP (Best Performance)",
        provider="CoreMLExecutionProvider",
        provider_options={},
        session_options={},
        env_vars={},
        install_cmd="pip install onnxruntime",
        setup_steps=[
            "CoreML EP is included in macOS builds of onnxruntime",
            "Ensure macOS 12+ for Neural Engine acceleration",
        ],
        python_code=python_code,
        notes=[
            f"Apple Silicon detected ({gpu.name})",
            f"Unified memory: {hw.system_ram_mb // 1024}GB shared between CPU, GPU, and Neural Engine",
            "CoreML leverages GPU (Metal) + ANE (Apple Neural Engine) automatically",
            "For transformers, consider coremltools for direct CoreML model conversion",
        ],
        estimated_memory_gb=runtime_gb,
    ))

    return recipes


# ---------------------------------------------------------------------------
# Qualcomm recipes
# ---------------------------------------------------------------------------

def _qualcomm_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                      model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024

    recipes.append(InferenceRecipe(
        title="Qualcomm QNN EP (Hexagon NPU)",
        provider="QNNExecutionProvider",
        provider_options={"backend_path": "libQnnHtp.so"},
        session_options={},
        env_vars={},
        install_cmd="pip install onnxruntime-qnn",
        setup_steps=[
            "Install Qualcomm AI Engine Direct SDK (QNN SDK)",
            "Set QNN_SDK_ROOT environment variable",
            "For Snapdragon X Elite: use HTP (Hexagon Tensor Processor) backend",
        ],
        python_code=f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()

providers = [
    ("QNNExecutionProvider", {{
        "backend_path": "libQnnHtp.so",
    }}),
    "CPUExecutionProvider",
]

session = ort.InferenceSession("{model_path}", sess_opts, providers=providers)
print(f"Running on: {{session.get_providers()[0]}}")
""",
        notes=[
            f"Qualcomm SoC detected ({gpu.name})",
            "HTP (Hexagon Tensor Processor) provides best performance for quantized models",
            "Quantize model to INT8 for best NPU performance: isat optimize model.onnx --int8",
            f"Unified memory: {hw.system_ram_mb // 1024}GB",
        ],
        estimated_memory_gb=runtime_gb,
    ))

    return recipes


# ---------------------------------------------------------------------------
# Windows MIGraphX EP via WinML CompileApi (AMD GPU only)
# ---------------------------------------------------------------------------

def _winml_migraphx_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                             model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024
    ep_dir = hw.winml_ep_path

    python_code = f"""\
import os
import onnxruntime as ort

# 1. Add the WinML AMD EP DLL directory
ep_dir = r'{ep_dir}'
os.add_dll_directory(ep_dir)

# 2. Register MIGraphX provider via CompileApi
ort.register_execution_provider_library(
    'MIGraphXExecutionProvider',
    os.path.join(ep_dir, 'onnxruntime_providers_migraphx.dll')
)

# 3. Get EP devices and add to session
devices = ort.get_ep_devices()
migraphx_dev = [d for d in devices if d.ep_name == 'MIGraphXExecutionProvider']
cpu_dev = [d for d in devices if d.ep_name == 'CPUExecutionProvider']

so = ort.SessionOptions()
so.log_severity_level = 3
so.add_provider_for_devices(migraphx_dev, {{}})
so.add_provider_for_devices(cpu_dev, {{}})

# 4. Create session (NO providers= arg — CompileApi handles it)
session = ort.InferenceSession("{model_path}", so)

active_ep = session.get_providers()[0]
assert "MIGraphX" in active_ep, f"Expected MIGraphX but got {{active_ep}}"
print(f"Running on: {{active_ep}}")
"""

    recipes.append(InferenceRecipe(
        title="MIGraphX EP via WinML CompileApi (Best Performance — AMD Windows)",
        provider="MIGraphXExecutionProvider",
        provider_options={},
        session_options={"log_severity_level": "3"},
        env_vars={},
        install_cmd="pip install onnxruntime\n# WinML AMD EP AppX package must be installed (via Microsoft Store / MSIX)",
        setup_steps=[
            f"WinML AMD EP detected: {ep_dir}",
            "Uses ORT CompileApi: register_execution_provider_library + add_provider_for_devices",
            "IMPORTANT: Do NOT use providers= in InferenceSession() — use SessionOptions.add_provider_for_devices()",
            "First run compiles the MIGraphX graph (~30-60s) — subsequent runs are fast",
        ],
        python_code=python_code,
        notes=[
            f"AMD GPU detected ({gpu.name}) with WinML MIGraphX EP installed",
            "Execution path: Python → ORT CompileApi → WinML → MIGraphX EP → AMD GPU",
            "MIGraphX provides native FP16 quantization and graph fusion — fastest for large models",
            "Falls back to CPU for unsupported ops automatically",
        ],
        estimated_memory_gb=runtime_gb,
    ))

    return recipes


# ---------------------------------------------------------------------------
# Windows DirectML recipes (works for ANY GPU: AMD, NVIDIA, Intel, Qualcomm)
# ---------------------------------------------------------------------------

def _directml_recipes(hw: HardwareProfile, gpu: DetectedGPU,
                      model_path: str, model_mb: float) -> list[InferenceRecipe]:
    recipes = []
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024

    python_code = f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()
sess_opts.log_severity_level = 3

providers = [
    "DmlExecutionProvider",
    "CPUExecutionProvider",
]

session = ort.InferenceSession("{model_path}", sess_opts, providers=providers)

# CRITICAL: verify DML is active (ORT silently falls back to CPU)
active_ep = session.get_providers()[0]
assert "Dml" in active_ep, f"Expected DmlExecutionProvider but got {{active_ep}}"
print(f"Running on: {{active_ep}}")
"""

    notes = [
        f"Windows detected — using DirectML EP (WinML → DirectX 12 → {gpu.name})",
        "DirectML works with ANY GPU on Windows (AMD, NVIDIA, Intel, Qualcomm)",
        f"GPU: {gpu.name} ({gpu.gpu_type.upper()}, {gpu.vram_mb}MB VRAM)" if gpu.vram_mb else
        f"GPU: {gpu.name} ({gpu.gpu_type.upper()}, shared system memory)",
    ]
    warnings = []
    setup = []

    if model_mb > 2000:
        warnings.append(f"Large model ({model_mb:.0f}MB) — may need significant GPU/system memory")
    if gpu.gpu_type in ("apu", "igpu"):
        notes.append(f"iGPU/APU uses shared system RAM ({hw.system_ram_mb // 1024}GB available)")

    recipes.append(InferenceRecipe(
        title="DirectML EP — Windows GPU Acceleration (Best for Windows)",
        provider="DmlExecutionProvider",
        provider_options={},
        session_options={"log_severity_level": "3"},
        env_vars={},
        install_cmd="pip install onnxruntime-directml",
        setup_steps=setup,
        python_code=python_code,
        notes=notes,
        warnings=warnings,
        estimated_memory_gb=runtime_gb,
    ))

    # DML with graph opt disabled (for models that crash DML's custom registry)
    recipes.append(InferenceRecipe(
        title="DirectML EP — Graph Optimizations Disabled (Fallback)",
        provider="DmlExecutionProvider",
        provider_options={},
        session_options={
            "graph_optimization_level": "ORT_DISABLE_ALL",
            "log_severity_level": "3",
        },
        env_vars={},
        install_cmd="pip install onnxruntime-directml",
        notes=[
            "Use this if the default DML config crashes with AbiCustomRegistry errors",
            "Disables ORT graph optimizations that can conflict with DirectML's internal ops",
            "Models known to need this: CrossFormer, OpenVLA (dml_disable_graph_opt=True)",
        ],
        estimated_memory_gb=runtime_gb,
    ))

    return recipes


# ---------------------------------------------------------------------------
# CPU-only fallback
# ---------------------------------------------------------------------------

def _cpu_recipes(hw: HardwareProfile,
                 model_path: str, model_mb: float) -> list[InferenceRecipe]:
    runtime_gb = _estimate_runtime_mem(model_mb) / 1024
    threads = hw.cpu_cores

    python_code = f"""\
import onnxruntime as ort

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = {threads}
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession("{model_path}", sess_opts, providers=["CPUExecutionProvider"])
print(f"Running on: {{session.get_providers()[0]}}")
"""

    return [InferenceRecipe(
        title="CPU Only (No GPU Detected)",
        provider="CPUExecutionProvider",
        session_options={
            "intra_op_num_threads": str(threads),
            "graph_optimization_level": "ORT_ENABLE_ALL",
        },
        env_vars={"OMP_NUM_THREADS": str(threads)},
        install_cmd="pip install onnxruntime",
        python_code=python_code,
        notes=[
            f"No GPU detected — running on CPU: {hw.cpu_name} ({threads} cores)",
            f"System RAM: {hw.system_ram_mb // 1024}GB",
            "Consider INT8 quantization for 2-4x CPU speedup: isat optimize model.onnx --int8",
            "Consider onnxruntime-openvino for Intel CPUs (2-3x faster on INT8)",
        ],
        estimated_memory_gb=runtime_gb,
    )]


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def generate_recommendations(hw: HardwareProfile,
                             model_path: str = "") -> TuneReport:
    """Generate actionable inference recipes for the detected hardware."""
    report = TuneReport(hardware=hw, model_path=model_path)

    model_mb = _model_size_mb(model_path) if model_path else 0.0
    report.model_size_mb = model_mb

    if model_path:
        try:
            from isat.fingerprint.model import fingerprint_model
            mfp = fingerprint_model(model_path)
            report.model_params = mfp.param_count
            report.model_class = mfp.model_class
        except Exception:
            pass

    # System-level warnings
    if hw.system_ram_mb < 8192:
        report.system_warnings.append(f"Low system RAM: {hw.system_ram_mb}MB — may OOM on large models")
    if hw.swap_mb < 4096 and hw.is_apu:
        report.system_warnings.append(f"Low swap: {hw.swap_mb}MB — APU needs large swap for MIGraphX compilation")

    gpu = hw.primary_gpu
    if not gpu:
        report.recipes = _cpu_recipes(hw, model_path, model_mb)
        return report

    is_windows = hw.os_name == "windows"

    # On Windows, DirectML works for ANY GPU vendor (AMD, NVIDIA, Intel, Qualcomm)
    if is_windows:
        # If WinML AMD MIGraphX EP is installed, offer it as the BEST option
        if hw.winml_ep_path and gpu.vendor == "amd":
            report.recipes = _winml_migraphx_recipes(hw, gpu, model_path, model_mb)
        else:
            report.recipes = []
        report.recipes.extend(_directml_recipes(hw, gpu, model_path, model_mb))
        # Also add vendor-specific recipes as alternatives
        vendor_map = {
            "amd": _amd_recipes,
            "nvidia": _nvidia_recipes,
            "intel": _intel_recipes,
            "qualcomm": _qualcomm_recipes,
        }
        alt_gen = vendor_map.get(gpu.vendor)
        if alt_gen:
            report.recipes.extend(alt_gen(hw, gpu, model_path, model_mb))
    else:
        vendor_map = {
            "amd": _amd_recipes,
            "nvidia": _nvidia_recipes,
            "intel": _intel_recipes,
            "apple": _apple_recipes,
            "qualcomm": _qualcomm_recipes,
        }
        generator = vendor_map.get(gpu.vendor)
        if generator:
            report.recipes = generator(hw, gpu, model_path, model_mb)
        else:
            report.recipes = _cpu_recipes(hw, model_path, model_mb)

    return report


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def format_report(report: TuneReport) -> str:
    """Format a TuneReport as a human-readable terminal output."""
    lines = []
    hw = report.hardware
    gpu = hw.primary_gpu

    # Header
    lines.append("")
    lines.append("=" * 72)
    lines.append("  ISAT HARDWARE AUTO-DETECTION & INFERENCE RECOMMENDATIONS")
    lines.append("=" * 72)

    # System info
    lines.append("")
    lines.append("  SYSTEM")
    lines.append("  " + "-" * 40)
    lines.append(f"  OS           : {hw.os_name} {hw.os_version[:40]}")
    lines.append(f"  Kernel       : {hw.kernel}")
    lines.append(f"  CPU          : {hw.cpu_name}")
    lines.append(f"  CPU Cores    : {hw.cpu_cores}")
    lines.append(f"  System RAM   : {hw.system_ram_mb // 1024} GB ({hw.system_ram_mb} MB)")
    lines.append(f"  Swap         : {hw.swap_mb // 1024} GB ({hw.swap_mb} MB)")

    # GPU info
    if gpu:
        lines.append("")
        lines.append("  GPU")
        lines.append("  " + "-" * 40)
        lines.append(f"  Vendor       : {gpu.vendor.upper()}")
        lines.append(f"  Name         : {gpu.name}")
        lines.append(f"  Type         : {gpu.gpu_type.upper()}")
        lines.append(f"  Architecture : {gpu.arch}")
        if gpu.compute_units:
            lines.append(f"  Compute Units: {gpu.compute_units}")
        if gpu.max_clock_mhz:
            lines.append(f"  Max Clock    : {gpu.max_clock_mhz} MHz")
        lines.append(f"  Matrix Cores : {gpu.matrix_cores}")
        if gpu.vram_mb:
            lines.append(f"  VRAM         : {gpu.vram_mb} MB")
        if gpu.shared_mem_mb:
            lines.append(f"  Shared Mem   : {gpu.shared_mem_mb} MB (GTT/unified)")
        lines.append(f"  FP16         : {'YES' if gpu.supports_fp16 else 'NO'}")
        lines.append(f"  BF16         : {'YES' if gpu.supports_bf16 else 'NO'}")
        lines.append(f"  INT8         : {'YES' if gpu.supports_int8 else 'NO'}")
        lines.append(f"  FP8          : {'YES' if gpu.supports_fp8 else 'NO'}")
        if gpu.driver_version:
            lines.append(f"  Driver       : {gpu.driver} {gpu.driver_version}")
    else:
        lines.append("")
        lines.append("  GPU          : None detected — CPU only")

    if len(hw.gpus) > 1:
        lines.append("")
        lines.append(f"  Additional GPUs ({len(hw.gpus) - 1}):")
        for g in hw.gpus[1:]:
            lines.append(f"    - {g.vendor.upper()} {g.name} ({g.gpu_type})")

    # Model info
    if report.model_path:
        lines.append("")
        lines.append("  MODEL")
        lines.append("  " + "-" * 40)
        lines.append(f"  Path         : {report.model_path}")
        if report.model_size_mb:
            lines.append(f"  Size         : {report.model_size_mb:.1f} MB")
        if report.model_params:
            lines.append(f"  Parameters   : {report.model_params:,}")
        if report.model_class:
            lines.append(f"  Class        : {report.model_class}")

    # System warnings
    if report.system_warnings:
        lines.append("")
        lines.append("  SYSTEM WARNINGS")
        lines.append("  " + "-" * 40)
        for w in report.system_warnings:
            lines.append(f"  [!] {w}")

    # Recipes
    lines.append("")
    lines.append("=" * 72)
    lines.append("  RECOMMENDED INFERENCE CONFIGURATIONS")
    lines.append("=" * 72)

    for i, recipe in enumerate(report.recipes, 1):
        lines.append("")
        lines.append(f"  [{i}] {recipe.title}")
        lines.append("  " + "~" * 60)

        if recipe.notes:
            for n in recipe.notes:
                lines.append(f"      {n}")
            lines.append("")

        if recipe.warnings:
            for w in recipe.warnings:
                lines.append(f"      [WARNING] {w}")
            lines.append("")

        # Install
        lines.append(f"      Install:")
        for cmd_line in recipe.install_cmd.split("\n"):
            lines.append(f"        {cmd_line}")

        # Environment variables
        if recipe.env_vars:
            lines.append("")
            lines.append(f"      Environment Variables:")
            for k, v in recipe.env_vars.items():
                lines.append(f"        export {k}={v}")

        # Setup steps
        if recipe.setup_steps:
            lines.append("")
            lines.append(f"      Setup Steps:")
            for step in recipe.setup_steps:
                lines.append(f"        {step}")

        # Python code
        if recipe.python_code:
            lines.append("")
            lines.append(f"      Python Code:")
            lines.append(f"      {'─' * 50}")
            for code_line in recipe.python_code.rstrip().split("\n"):
                lines.append(f"      {code_line}")
            lines.append(f"      {'─' * 50}")

        if recipe.estimated_memory_gb > 0:
            lines.append(f"      Estimated Memory: ~{recipe.estimated_memory_gb:.1f} GB")

    # Quick command
    if report.model_path and report.recipes:
        best = report.recipes[0]
        lines.append("")
        lines.append("=" * 72)
        lines.append("  QUICK START (copy-paste)")
        lines.append("=" * 72)
        lines.append("")
        if best.env_vars:
            env_str = " ".join(f"{k}={v}" for k, v in best.env_vars.items())
            lines.append(f"  # Set environment")
            for k, v in best.env_vars.items():
                lines.append(f"  export {k}={v}")
            lines.append("")
        lines.append(f"  # Install")
        for cmd_line in best.install_cmd.split("\n"):
            if cmd_line.strip():
                lines.append(f"  {cmd_line.strip()}")
        lines.append("")
        lines.append(f"  # Run with ISAT auto-tune")
        lines.append(f"  isat tune {report.model_path} --provider {best.provider}")

    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)
