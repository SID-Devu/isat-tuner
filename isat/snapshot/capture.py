"""Environment snapshot for reproducibility.

Captures the complete system state so results can be reproduced exactly:
  - Hardware (GPU model, memory, driver version, clocks)
  - Software (OS, kernel, Python, ORT, ROCm/CUDA versions)
  - Environment variables (inference-related)
  - Model hash (SHA256)
  - ISAT version and config

Saves to JSON and can be attached to reports or CI artifacts.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.snapshot")


class EnvSnapshot:
    """Capture and save a complete environment snapshot."""

    def capture(self, model_path: str = "") -> dict:
        """Capture current environment state."""
        snap = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "isat_version": self._isat_version(),
            "system": self._system_info(),
            "python": self._python_info(),
            "gpu": self._gpu_info(),
            "software": self._software_versions(),
            "environment": self._env_vars(),
        }

        if model_path and Path(model_path).exists():
            snap["model"] = {
                "path": model_path,
                "size_bytes": Path(model_path).stat().st_size,
                "sha256": self._file_hash(model_path),
            }

        return snap

    def save(self, snapshot: dict, output_path: str = "isat_snapshot.json") -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        log.info("Snapshot saved to %s", output_path)
        return output_path

    def diff(self, snap_a: dict, snap_b: dict) -> list[str]:
        """Compare two snapshots and return differences."""
        diffs = []
        self._diff_recursive(snap_a, snap_b, "", diffs)
        return diffs

    def _isat_version(self) -> str:
        try:
            import isat
            return isat.__version__
        except (ImportError, AttributeError):
            return "unknown"

    def _system_info(self) -> dict:
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "arch": platform.machine(),
            "hostname": platform.node(),
            "kernel": self._run_cmd("uname -r"),
        }

    def _python_info(self) -> dict:
        return {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        }

    def _gpu_info(self) -> dict:
        info: dict = {}

        rocm_path = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
        ver_file = rocm_path / ".info" / "version"
        if ver_file.exists():
            info["rocm_version"] = ver_file.read_text().strip()

        try:
            p = Path("/sys/class/drm/card0/device")
            if (p / "gpu_id").exists():
                info["gpu_id"] = (p / "gpu_id").read_text().strip()
            if (p / "mem_info_vram_total").exists():
                vram = int((p / "mem_info_vram_total").read_text().strip())
                info["vram_total_mb"] = vram // (1024 * 1024)
            if (p / "mem_info_gtt_total").exists():
                gtt = int((p / "mem_info_gtt_total").read_text().strip())
                info["gtt_total_mb"] = gtt // (1024 * 1024)
        except (OSError, ValueError):
            pass

        nvcc = self._run_cmd("nvcc --version")
        if nvcc:
            info["cuda_version"] = nvcc.split("release")[-1].split(",")[0].strip() if "release" in nvcc else nvcc

        return info

    def _software_versions(self) -> dict:
        versions: dict = {}
        for pkg in ["onnxruntime", "onnx", "numpy", "onnxsim", "scipy", "fastapi"]:
            try:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, "__version__", "installed")
            except ImportError:
                pass
        return versions

    def _env_vars(self) -> dict:
        relevant = {}
        for key in sorted(os.environ):
            kl = key.upper()
            if any(x in kl for x in [
                "MIGRAPHX", "HSA_XNACK", "HIP_", "CUDA_", "ORT_",
                "ROCM", "GPU_", "OMP_", "KMP_", "ISAT_",
            ]):
                relevant[key] = os.environ[key]
        return relevant

    def _file_hash(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def _run_cmd(self, cmd: str) -> str:
        try:
            return subprocess.check_output(cmd.split(), stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return ""

    def _diff_recursive(self, a, b, prefix: str, diffs: list[str]):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(a) | set(b):
                p = f"{prefix}.{key}" if prefix else key
                if key not in a:
                    diffs.append(f"+ {p}: {b[key]}")
                elif key not in b:
                    diffs.append(f"- {p}: {a[key]}")
                else:
                    self._diff_recursive(a[key], b[key], p, diffs)
        elif a != b:
            diffs.append(f"~ {prefix}: {a} -> {b}")
