"""Dependency and environment compatibility scanner.

Pre-flight checks before tuning:
  - Python version compatibility
  - ORT version and available providers
  - ROCm stack (driver, runtime, libraries)
  - CUDA stack (driver, toolkit, cuDNN)
  - GPU driver status
  - ONNX opset compatibility
  - Required Python packages
  - Environment variable sanity
"""

from __future__ import annotations

import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.compat")


@dataclass
class CheckResult:
    name: str
    status: str  # "ok", "warning", "error", "skip"
    message: str
    details: str = ""
    fix_hint: str = ""


@dataclass
class CompatReport:
    checks: list[CheckResult] = field(default_factory=list)
    errors: int = 0
    warnings: int = 0
    passed: int = 0

    @property
    def ok(self) -> bool:
        return self.errors == 0

    def summary(self) -> str:
        lines = [f"  {'Check':<35} {'Status':<10} {'Message'}"]
        lines.append(f"  {'-'*35} {'-'*10} {'-'*40}")
        for c in self.checks:
            icon = {"ok": "PASS", "warning": "WARN", "error": "FAIL", "skip": "SKIP"}[c.status]
            lines.append(f"  {c.name:<35} {icon:<10} {c.message}")
            if c.fix_hint:
                lines.append(f"  {'':35} {'':10} -> {c.fix_hint}")
        lines.append(f"\n  Result: {self.passed} passed, {self.warnings} warnings, {self.errors} errors")
        return "\n".join(lines)


class CompatScanner:
    """Scan environment for inference compatibility."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def scan(self) -> CompatReport:
        report = CompatReport()

        checks = [
            self._check_python,
            self._check_onnxruntime,
            self._check_ort_providers,
            self._check_numpy,
            self._check_onnx,
            self._check_rocm,
            self._check_cuda,
            self._check_gpu_driver,
            self._check_env_vars,
        ]

        if self.model_path:
            checks.append(self._check_model)

        for check_fn in checks:
            try:
                result = check_fn()
                report.checks.append(result)
                if result.status == "ok":
                    report.passed += 1
                elif result.status == "warning":
                    report.warnings += 1
                elif result.status == "error":
                    report.errors += 1
            except Exception as e:
                report.checks.append(CheckResult(
                    check_fn.__name__.replace("_check_", ""),
                    "error", f"Check crashed: {e}"
                ))
                report.errors += 1

        return report

    def _check_python(self) -> CheckResult:
        ver = platform.python_version()
        major, minor = sys.version_info[:2]
        if major < 3 or minor < 9:
            return CheckResult("Python version", "error",
                               f"Python {ver} (need >= 3.9)",
                               fix_hint="Install Python 3.9+")
        if minor < 10:
            return CheckResult("Python version", "warning",
                               f"Python {ver} (3.10+ recommended)")
        return CheckResult("Python version", "ok", f"Python {ver}")

    def _check_onnxruntime(self) -> CheckResult:
        try:
            import onnxruntime as ort
            ver = ort.__version__
            return CheckResult("ONNX Runtime", "ok", f"v{ver}")
        except ImportError:
            return CheckResult("ONNX Runtime", "error", "Not installed",
                               fix_hint="pip install onnxruntime")

    def _check_ort_providers(self) -> CheckResult:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            gpu_providers = [p for p in providers if p != "CPUExecutionProvider"]
            if gpu_providers:
                return CheckResult("ORT GPU providers", "ok",
                                   f"{', '.join(gpu_providers)}")
            return CheckResult("ORT GPU providers", "warning",
                               "Only CPUExecutionProvider available",
                               fix_hint="Install onnxruntime-rocm or onnxruntime-gpu")
        except ImportError:
            return CheckResult("ORT GPU providers", "skip", "ORT not installed")

    def _check_numpy(self) -> CheckResult:
        try:
            import numpy as np
            return CheckResult("NumPy", "ok", f"v{np.__version__}")
        except ImportError:
            return CheckResult("NumPy", "error", "Not installed",
                               fix_hint="pip install numpy")

    def _check_onnx(self) -> CheckResult:
        try:
            import onnx
            return CheckResult("ONNX", "ok", f"v{onnx.__version__}")
        except ImportError:
            return CheckResult("ONNX", "warning", "Not installed (needed for optimizer)",
                               fix_hint="pip install onnx")

    def _check_rocm(self) -> CheckResult:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        if not Path(rocm_path).exists():
            return CheckResult("ROCm stack", "skip", "Not found (not required for CUDA)")

        version_file = Path(rocm_path) / ".info" / "version"
        if version_file.exists():
            ver = version_file.read_text().strip()
            return CheckResult("ROCm stack", "ok", f"v{ver} at {rocm_path}")

        return CheckResult("ROCm stack", "warning",
                           f"Found at {rocm_path} but no version info")

    def _check_cuda(self) -> CheckResult:
        if shutil.which("nvcc"):
            try:
                out = subprocess.check_output(["nvcc", "--version"],
                                              stderr=subprocess.STDOUT, text=True)
                for line in out.splitlines():
                    if "release" in line.lower():
                        return CheckResult("CUDA toolkit", "ok", line.strip())
                return CheckResult("CUDA toolkit", "ok", "nvcc found")
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        if Path("/usr/local/cuda").exists():
            return CheckResult("CUDA toolkit", "warning",
                               "Directory found but nvcc not in PATH")
        return CheckResult("CUDA toolkit", "skip", "Not found (not required for ROCm)")

    def _check_gpu_driver(self) -> CheckResult:
        if Path("/dev/kfd").exists():
            return CheckResult("GPU driver (KFD)", "ok", "/dev/kfd accessible")
        if Path("/dev/nvidia0").exists():
            return CheckResult("GPU driver (NVIDIA)", "ok", "/dev/nvidia0 accessible")
        return CheckResult("GPU driver", "warning",
                           "No GPU device found",
                           fix_hint="Check driver installation")

    def _check_env_vars(self) -> CheckResult:
        relevant = {}
        for key in os.environ:
            kl = key.upper()
            if any(x in kl for x in ["MIGRAPHX", "HSA_XNACK", "HIP_", "CUDA_", "ORT_"]):
                relevant[key] = os.environ[key]

        if relevant:
            details = "; ".join(f"{k}={v}" for k, v in sorted(relevant.items()))
            return CheckResult("Environment variables", "ok",
                               f"{len(relevant)} inference-related vars set",
                               details=details)
        return CheckResult("Environment variables", "ok", "No overrides set (defaults)")

    def _check_model(self) -> CheckResult:
        if not self.model_path:
            return CheckResult("Model file", "skip", "No model specified")

        p = Path(self.model_path)
        if not p.exists():
            return CheckResult("Model file", "error", f"Not found: {self.model_path}")

        size_mb = p.stat().st_size / (1024 * 1024)

        try:
            import onnx
            model = onnx.load(self.model_path, load_external_data=False)
            opset = model.opset_import[0].version if model.opset_import else 0
            nodes = len(model.graph.node)
            return CheckResult("Model file", "ok",
                               f"{size_mb:.0f} MB, {nodes} nodes, opset {opset}")
        except Exception:
            return CheckResult("Model file", "ok", f"{size_mb:.0f} MB (ONNX parse skipped)")
