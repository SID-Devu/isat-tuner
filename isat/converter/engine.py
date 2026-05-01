"""Format detection and conversion dispatch engine.

Detects the input model format from the path / identifier and routes
to the appropriate conversion backend in ``backends.py``.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.converter")


class ModelFormat(Enum):
    HUGGINGFACE = auto()
    PYTORCH = auto()
    TENSORFLOW = auto()
    TFLITE = auto()
    JAX = auto()
    SAFETENSORS = auto()
    ONNX = auto()
    UNKNOWN = auto()


@dataclass
class ConversionResult:
    success: bool
    onnx_path: str = ""
    source_format: ModelFormat = ModelFormat.UNKNOWN
    backend_used: str = ""
    opset: int = 0
    num_nodes: int = 0
    size_mb: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.success:
            return f"  Conversion FAILED: {self.error}"
        lines = [
            f"  Output      : {self.onnx_path}",
            f"  Format      : {self.source_format.name} -> ONNX",
            f"  Backend     : {self.backend_used}",
            f"  Opset       : {self.opset}",
            f"  Nodes       : {self.num_nodes}",
            f"  Size        : {self.size_mb:.1f} MB",
            f"  Time        : {self.elapsed_s:.1f}s",
        ]
        if self.warnings:
            lines.append(f"  Warnings    : {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def detect_format(input_path: str) -> ModelFormat:
    """Auto-detect model format from the input path or identifier."""
    p = Path(input_path)
    ext = p.suffix.lower()

    if ext == ".onnx":
        return ModelFormat.ONNX

    if ext == ".tflite":
        return ModelFormat.TFLITE

    if ext in (".pt", ".pth", ".bin"):
        return ModelFormat.PYTORCH

    if ext == ".safetensors":
        return ModelFormat.SAFETENSORS

    if ext == ".pb":
        return ModelFormat.TENSORFLOW

    if ext in (".jax", ".msgpack"):
        return ModelFormat.JAX

    if p.is_dir():
        if (p / "saved_model.pb").exists():
            return ModelFormat.TENSORFLOW
        if (p / "config.json").exists():
            if (p / "flax_model.msgpack").exists():
                return ModelFormat.JAX
            return ModelFormat.HUGGINGFACE
        if any(p.glob("*.pb")):
            return ModelFormat.TENSORFLOW
        if any(p.glob("*.pt")) or any(p.glob("*.pth")) or any(p.glob("*.bin")):
            return ModelFormat.PYTORCH

    if input_path.startswith("hf://"):
        return ModelFormat.HUGGINGFACE

    _KNOWN_EXTS = {
        ".onnx", ".pt", ".pth", ".bin", ".pb", ".tflite",
        ".safetensors", ".jax", ".msgpack", ".h5", ".keras",
        ".tar", ".gz", ".zip", ".npz", ".npy",
    }
    ext_is_real_format = ext in _KNOWN_EXTS

    looks_like_hf_id = (
        "/" in input_path
        and not p.exists()
        and not ext_is_real_format
        and not input_path.startswith("/")
        and not input_path.startswith("./")
    )
    if looks_like_hf_id:
        return ModelFormat.HUGGINGFACE

    if not p.exists() and not ext_is_real_format and not input_path.startswith("/"):
        return ModelFormat.HUGGINGFACE

    return ModelFormat.UNKNOWN


def convert(
    input_path: str,
    output_dir: str = ".",
    opset: int = 17,
    input_shape: Optional[str] = None,
    simplify: bool = False,
) -> ConversionResult:
    """Convert any supported model format to ONNX.

    Parameters
    ----------
    input_path : str
        Model path, directory, or HuggingFace model ID.
    output_dir : str
        Directory to write the output ONNX file.
    opset : int
        Target ONNX opset version.
    input_shape : str or None
        Comma-separated input shape (e.g. "1,3,224,224") for raw
        PyTorch / JAX checkpoints that lack shape metadata.
    simplify : bool
        Run onnxsim on the output to simplify the graph.
    """
    fmt = detect_format(input_path)
    log.info("Detected format: %s for input: %s", fmt.name, input_path)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if fmt == ModelFormat.ONNX:
        result = _passthrough_onnx(input_path, str(out))
    elif fmt == ModelFormat.HUGGINGFACE:
        from isat.converter.backends import convert_huggingface
        result = convert_huggingface(input_path, str(out), opset)
    elif fmt == ModelFormat.PYTORCH:
        from isat.converter.backends import convert_pytorch
        result = convert_pytorch(input_path, str(out), opset, input_shape)
    elif fmt == ModelFormat.TENSORFLOW:
        from isat.converter.backends import convert_tensorflow
        result = convert_tensorflow(input_path, str(out), opset)
    elif fmt == ModelFormat.TFLITE:
        from isat.converter.backends import convert_tflite
        result = convert_tflite(input_path, str(out), opset)
    elif fmt == ModelFormat.JAX:
        from isat.converter.backends import convert_jax
        result = convert_jax(input_path, str(out), opset, input_shape)
    elif fmt == ModelFormat.SAFETENSORS:
        from isat.converter.backends import convert_safetensors
        result = convert_safetensors(input_path, str(out), opset, input_shape)
    else:
        result = ConversionResult(
            success=False,
            source_format=fmt,
            error=(
                f"Cannot detect model format for '{input_path}'.\n"
                "  Supported: .onnx, .pt, .pth, .bin, .pb, .tflite, .safetensors,\n"
                "             .jax, .msgpack, TF SavedModel dirs, HuggingFace model IDs"
            ),
        )

    result.elapsed_s = time.time() - t0
    result.source_format = fmt

    if result.success and simplify:
        result = _simplify(result)

    if result.success:
        result = _validate(result)

    return result


def _passthrough_onnx(input_path: str, output_dir: str) -> ConversionResult:
    """If the input is already ONNX, just copy/symlink to the output dir."""
    import onnx

    src = Path(input_path)
    dst = Path(output_dir) / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)

    model = onnx.load(str(dst), load_external_data=False)

    return ConversionResult(
        success=True,
        onnx_path=str(dst),
        source_format=ModelFormat.ONNX,
        backend_used="passthrough (already ONNX)",
        opset=model.opset_import[0].version if model.opset_import else 0,
        num_nodes=len(model.graph.node),
        size_mb=dst.stat().st_size / (1024 * 1024),
    )


def _simplify(result: ConversionResult) -> ConversionResult:
    """Run onnxsim on the converted model."""
    try:
        import onnx
        import onnxsim

        model = onnx.load(result.onnx_path)
        model_sim, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_sim, result.onnx_path)
            result.num_nodes = len(model_sim.graph.node)
            result.size_mb = Path(result.onnx_path).stat().st_size / (1024 * 1024)
            log.info("Simplified: %d nodes", result.num_nodes)
        else:
            result.warnings.append("onnxsim simplification check failed; kept original")
    except ImportError:
        result.warnings.append("onnxsim not installed; skipping simplify (pip install onnxsim)")
    except Exception as e:
        result.warnings.append(f"onnxsim failed: {e}")
    return result


def _validate(result: ConversionResult) -> ConversionResult:
    """Basic ONNX validation on the output."""
    try:
        import onnx
        model = onnx.load(result.onnx_path, load_external_data=False)
        onnx.checker.check_model(model)
        result.opset = model.opset_import[0].version if model.opset_import else 0
        result.num_nodes = len(model.graph.node)
        result.size_mb = Path(result.onnx_path).stat().st_size / (1024 * 1024)
    except Exception as e:
        result.warnings.append(f"ONNX validation warning: {e}")
    return result
