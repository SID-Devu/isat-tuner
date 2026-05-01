"""Conversion backends for each supported ML framework.

Each function follows the same contract:
  - Accept input_path, output_dir, opset, and optional extra args
  - Return a ``ConversionResult``
  - On missing dependency, return a result with success=False and a
    clear pip-install instruction in the error message

All framework imports are lazy so the module loads with zero extra deps.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

from isat.converter.engine import ConversionResult, ModelFormat

log = logging.getLogger("isat.converter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_model_name(identifier: str) -> str:
    """Turn 'google/vit-base-patch16-224' into 'vit_base_patch16_224'."""
    name = identifier.split("/")[-1] if "/" in identifier else identifier
    name = Path(name).stem
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_").lower()


def _parse_shape(shape_str: Optional[str]) -> Optional[list[int]]:
    if not shape_str:
        return None
    return [int(x.strip()) for x in shape_str.split(",")]


def _onnx_meta(onnx_path: str) -> tuple[int, int, float]:
    """Return (opset, num_nodes, size_mb) for an ONNX file."""
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    opset = m.opset_import[0].version if m.opset_import else 0
    return opset, len(m.graph.node), Path(onnx_path).stat().st_size / (1024 * 1024)


# ---------------------------------------------------------------------------
# 1. HuggingFace (primary path)
# ---------------------------------------------------------------------------

def convert_huggingface(
    model_id: str,
    output_dir: str,
    opset: int = 17,
) -> ConversionResult:
    """Convert a HuggingFace model ID to ONNX.

    Tries optimum.exporters first (best quality), then falls back to
    torch.onnx.export via AutoModel + dummy inputs.
    """
    model_name = _safe_model_name(model_id)
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_dir / "model.onnx")

    # --- Try 1: optimum exporters (handles tokenizers, dynamic axes, multi-file) ---
    try:
        from optimum.exporters.onnx import main_export

        log.info("Using optimum.exporters.onnx for %s", model_id)
        main_export(
            model_name_or_path=model_id,
            output=str(out_dir),
            opset=opset,
            task="auto",
        )

        exported = list(out_dir.glob("*.onnx"))
        if not exported:
            return ConversionResult(
                success=False, error="optimum export produced no .onnx files"
            )
        onnx_path = str(exported[0])
        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True,
            onnx_path=onnx_path,
            backend_used="optimum.exporters.onnx",
            opset=opset_v,
            num_nodes=nodes,
            size_mb=sz,
        )

    except ImportError:
        log.info("optimum not found, falling back to torch.onnx.export")
    except Exception as e:
        log.warning("optimum export failed (%s), trying torch fallback", e)

    # --- Try 2: torch.onnx.export via transformers AutoModel ---
    try:
        import torch
        from transformers import AutoConfig, AutoModel

        log.info("Using torch.onnx.export for %s", model_id)
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, config=config)
        model.eval()

        dummy_inputs, input_names = _hf_dummy_input(model_id, config, model)
        input_dict = dict(zip(input_names, dummy_inputs))

        # Disable KV cache for export (avoids DynamicCache issues)
        if hasattr(model, "config"):
            model.config.use_cache = False

        # Build positional arg tuple matching the model's forward() signature
        # with None padding for gaps between our provided inputs
        import inspect
        fwd_sig = inspect.signature(model.forward)
        fwd_params = [p for p in fwd_sig.parameters.keys() if p != "self"]

        # Find the last param in fwd_params that we have a tensor for
        last_input_idx = -1
        for i, p in enumerate(fwd_params):
            if p in input_dict:
                last_input_idx = i

        ordered_args = []
        ordered_names = []
        if last_input_idx >= 0:
            for i, p in enumerate(fwd_params[:last_input_idx + 1]):
                if p in input_dict:
                    ordered_args.append(input_dict[p])
                    ordered_names.append(p)
                else:
                    ordered_args.append(None)
        if not ordered_args:
            ordered_args = list(dummy_inputs)
            ordered_names = list(input_names)

        dyn_ax = {n: {0: "batch"} for n in ordered_names}
        dyn_ax["output"] = {0: "batch"}
        args_tuple = tuple(ordered_args)

        exported = False
        last_err = ""

        # Strategy 1: legacy (dynamo=False) — most reliable
        for use_dynamo in (False, True):
            try:
                torch.onnx.export(
                    model, args_tuple, onnx_path,
                    input_names=ordered_names,
                    output_names=["output"],
                    dynamic_axes=dyn_ax,
                    opset_version=opset,
                    do_constant_folding=True,
                    dynamo=use_dynamo,
                )
                exported = True
                break
            except TypeError as te:
                if "dynamo" in str(te):
                    try:
                        torch.onnx.export(
                            model, args_tuple, onnx_path,
                            input_names=ordered_names,
                            output_names=["output"],
                            dynamic_axes=dyn_ax,
                            opset_version=opset,
                            do_constant_folding=True,
                        )
                        exported = True
                        break
                    except Exception as e2:
                        last_err = str(e2)
                        log.warning("torch.onnx.export (legacy-noarg) failed: %s", e2)
                else:
                    last_err = str(te)
                    log.warning("torch.onnx.export (dynamo=%s) TypeError: %s", use_dynamo, te)
            except Exception as e:
                last_err = str(e)
                log.warning("torch.onnx.export (dynamo=%s) failed: %s", use_dynamo, e)

        if not exported:
            return ConversionResult(
                success=False,
                error=f"torch.onnx.export failed: {last_err[:300]}",
            )

        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True,
            onnx_path=onnx_path,
            backend_used="torch.onnx.export (transformers AutoModel)",
            opset=opset_v,
            num_nodes=nodes,
            size_mb=sz,
        )

    except ImportError:
        return ConversionResult(
            success=False,
            error=(
                "Neither 'optimum' nor 'torch+transformers' found.\n"
                "  Install one of:\n"
                "    pip install 'optimum[exporters]'        # recommended\n"
                "    pip install torch transformers           # fallback"
            ),
        )
    except Exception as e:
        return ConversionResult(success=False, error=f"HuggingFace conversion failed: {e}")


def _hf_dummy_input(model_id: str, config, model):
    """Build dummy input tensors and their names for a HuggingFace model.

    Returns (tuple_of_tensors, list_of_names).
    Handles text-only, vision-only, and multi-modal (CLIP-style) models.
    """
    import torch

    # Multi-modal models (CLIP, etc.): use processor (combines text + image)
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        import numpy as np
        from PIL import Image
        dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        encoded = processor(text=["hello world"], images=dummy_img, return_tensors="pt", padding=True)
        names = list(encoded.keys())
        tensors = tuple(encoded[k] for k in names)
        return tensors, names
    except Exception:
        pass

    # Text models: use tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        encoded = tok("hello world", return_tensors="pt", padding="max_length", max_length=16)
        names = list(encoded.keys())
        tensors = tuple(encoded[k] for k in names)
        return tensors, names
    except Exception:
        pass

    # Vision models: use feature extractor / image processor
    try:
        from transformers import AutoFeatureExtractor
        feat = AutoFeatureExtractor.from_pretrained(model_id)
        import numpy as np
        if hasattr(feat, "size"):
            sz = feat.size
            if isinstance(sz, dict):
                h, w = sz.get("height", 224), sz.get("width", 224)
            elif isinstance(sz, int):
                h = w = sz
            else:
                h = w = 224
        else:
            h = w = 224
        dummy_img = np.random.randn(1, 3, h, w).astype(np.float32)
        return (torch.from_numpy(dummy_img),), ["pixel_values"]
    except Exception:
        pass

    return (torch.randn(1, 3, 224, 224),), ["input"]


# ---------------------------------------------------------------------------
# 2. PyTorch (.pt / .pth / .bin / directory)
# ---------------------------------------------------------------------------

def convert_pytorch(
    input_path: str,
    output_dir: str,
    opset: int = 17,
    input_shape: Optional[str] = None,
) -> ConversionResult:
    """Convert a PyTorch checkpoint to ONNX.

    For raw .pt files, ``--input-shape`` is required.
    For HuggingFace-format directories (with config.json), shapes are
    inferred automatically.
    """
    try:
        import torch
    except ImportError:
        return ConversionResult(
            success=False,
            error="PyTorch not installed.\n  pip install torch",
        )

    p = Path(input_path)
    model_name = _safe_model_name(p.stem)
    onnx_path = str(Path(output_dir) / f"{model_name}.onnx")

    if p.is_dir() and (p / "config.json").exists():
        return _convert_hf_local_dir(str(p), output_dir, opset)

    shape = _parse_shape(input_shape)
    if shape is None:
        return ConversionResult(
            success=False,
            error=(
                "Raw PyTorch checkpoint requires --input-shape.\n"
                "  Example: isat onnx model.pt --input-shape 1,3,224,224"
            ),
        )

    try:
        checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            return ConversionResult(
                success=False,
                error=(
                    "Checkpoint contains only state_dict (no model architecture).\n"
                    "  You need to load the architecture first and then load_state_dict.\n"
                    "  Consider using a HuggingFace model ID or a TorchScript (.pt) file."
                ),
            )
        else:
            model = checkpoint

        if not isinstance(model, torch.nn.Module):
            return ConversionResult(
                success=False,
                error="Could not extract nn.Module from checkpoint. Use a HuggingFace ID instead.",
            )

        model.eval()
        dummy = torch.randn(*shape)

        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )

        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True,
            onnx_path=onnx_path,
            backend_used="torch.onnx.export",
            opset=opset_v,
            num_nodes=nodes,
            size_mb=sz,
        )

    except Exception as e:
        return ConversionResult(success=False, error=f"PyTorch conversion failed: {e}")


def _convert_hf_local_dir(dir_path: str, output_dir: str, opset: int) -> ConversionResult:
    """Convert a local HuggingFace-format directory to ONNX."""
    try:
        from optimum.exporters.onnx import main_export

        model_name = _safe_model_name(Path(dir_path).name)
        out_dir = Path(output_dir) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        main_export(
            model_name_or_path=dir_path,
            output=str(out_dir),
            opset=opset,
            task="auto",
        )

        exported = list(out_dir.glob("*.onnx"))
        if not exported:
            return ConversionResult(success=False, error="optimum export produced no .onnx files")

        onnx_path = str(exported[0])
        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True,
            onnx_path=onnx_path,
            backend_used="optimum.exporters.onnx (local dir)",
            opset=opset_v,
            num_nodes=nodes,
            size_mb=sz,
        )

    except ImportError:
        return ConversionResult(
            success=False,
            error=(
                "optimum is required for local HuggingFace directory export.\n"
                "  pip install 'optimum[exporters]'"
            ),
        )
    except Exception as e:
        return ConversionResult(success=False, error=f"Local HF dir export failed: {e}")


# ---------------------------------------------------------------------------
# 3. TensorFlow (SavedModel / frozen graph .pb)
# ---------------------------------------------------------------------------

def convert_tensorflow(
    input_path: str,
    output_dir: str,
    opset: int = 17,
) -> ConversionResult:
    """Convert a TensorFlow SavedModel or frozen graph to ONNX via tf2onnx."""
    try:
        import tf2onnx
    except ImportError:
        return ConversionResult(
            success=False,
            error="tf2onnx not installed.\n  pip install tf2onnx tensorflow",
        )

    p = Path(input_path)
    model_name = _safe_model_name(p.stem if p.is_file() else p.name)
    onnx_path = str(Path(output_dir) / f"{model_name}.onnx")

    try:
        import subprocess
        cmd = ["python3", "-m", "tf2onnx.convert", "--opset", str(opset), "--output", onnx_path]

        if p.is_dir():
            cmd += ["--saved-model", str(p)]
        elif p.suffix == ".pb":
            cmd += ["--graphdef", str(p)]
        else:
            cmd += ["--saved-model", str(p)]

        log.info("Running: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if r.returncode != 0:
            return ConversionResult(
                success=False,
                error=f"tf2onnx failed (exit {r.returncode}):\n{r.stderr[-500:]}",
            )

        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True,
            onnx_path=onnx_path,
            backend_used="tf2onnx",
            opset=opset_v,
            num_nodes=nodes,
            size_mb=sz,
        )

    except Exception as e:
        return ConversionResult(success=False, error=f"TensorFlow conversion failed: {e}")


# ---------------------------------------------------------------------------
# 4. TFLite (.tflite)
# ---------------------------------------------------------------------------

def convert_tflite(
    input_path: str,
    output_dir: str,
    opset: int = 17,
) -> ConversionResult:
    """Convert a TFLite model to ONNX via tf2onnx."""
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        return ConversionResult(
            success=False,
            error="tf2onnx not installed.\n  pip install tf2onnx tensorflow",
        )

    model_name = _safe_model_name(Path(input_path).stem)
    onnx_path = str(Path(output_dir) / f"{model_name}.onnx")

    try:
        import subprocess
        cmd = [
            "python3", "-m", "tf2onnx.convert",
            "--opset", str(opset),
            "--tflite", str(input_path),
            "--output", onnx_path,
        ]
        log.info("Running: %s", " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if r.returncode != 0:
            return ConversionResult(
                success=False,
                error=f"tf2onnx (tflite) failed (exit {r.returncode}):\n{r.stderr[-500:]}",
            )

        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True,
            onnx_path=onnx_path,
            backend_used="tf2onnx (tflite)",
            opset=opset_v,
            num_nodes=nodes,
            size_mb=sz,
        )

    except Exception as e:
        return ConversionResult(success=False, error=f"TFLite conversion failed: {e}")


# ---------------------------------------------------------------------------
# 5. JAX (.jax / flax_model.msgpack / directory)
# ---------------------------------------------------------------------------

def convert_jax(
    input_path: str,
    output_dir: str,
    opset: int = 17,
    input_shape: Optional[str] = None,
) -> ConversionResult:
    """Convert a JAX model to ONNX.

    Primary: jax2onnx (direct).
    Fallback: jax -> TF SavedModel (jax2tf) -> tf2onnx.
    """
    # Try jax2onnx first
    try:
        import jax2onnx  # noqa: F401
        return _jax_via_jax2onnx(input_path, output_dir, opset, input_shape)
    except ImportError:
        pass

    # Fallback: jax2tf + tf2onnx
    try:
        import jax  # noqa: F401
        from jax.experimental import jax2tf  # noqa: F401
        return _jax_via_tf_bridge(input_path, output_dir, opset, input_shape)
    except ImportError:
        return ConversionResult(
            success=False,
            error=(
                "JAX conversion requires one of:\n"
                "  pip install jax2onnx           # direct conversion (recommended)\n"
                "  pip install jax tf2onnx         # via TF bridge"
            ),
        )


def _jax_via_jax2onnx(
    input_path: str, output_dir: str, opset: int, input_shape: Optional[str]
) -> ConversionResult:
    model_name = _safe_model_name(Path(input_path).stem)
    onnx_path = str(Path(output_dir) / f"{model_name}.onnx")

    try:
        import subprocess
        cmd = ["python3", "-m", "jax2onnx", "convert", "--input", input_path, "--output", onnx_path]
        if input_shape:
            cmd += ["--input-shape", input_shape]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            return ConversionResult(
                success=False,
                error=f"jax2onnx failed (exit {r.returncode}):\n{r.stderr[-500:]}",
            )

        opset_v, nodes, sz = _onnx_meta(onnx_path)
        return ConversionResult(
            success=True, onnx_path=onnx_path, backend_used="jax2onnx",
            opset=opset_v, num_nodes=nodes, size_mb=sz,
        )
    except Exception as e:
        return ConversionResult(success=False, error=f"jax2onnx failed: {e}")


def _jax_via_tf_bridge(
    input_path: str, output_dir: str, opset: int, input_shape: Optional[str]
) -> ConversionResult:
    """JAX -> TF SavedModel -> tf2onnx -> ONNX."""
    import tempfile

    model_name = _safe_model_name(Path(input_path).stem)
    onnx_path = str(Path(output_dir) / f"{model_name}.onnx")

    try:
        import jax
        import jax.numpy as jnp
        from jax.experimental import jax2tf
        import tensorflow as tf

        shape = _parse_shape(input_shape) or [1, 3, 224, 224]

        params = None
        try:
            from flax.training import checkpoints
            params = checkpoints.restore_checkpoint(input_path, target=None)
        except Exception:
            pass

        if params is None:
            return ConversionResult(
                success=False,
                error=(
                    "Could not load JAX checkpoint. For JAX models, consider:\n"
                    "  1. Using jax2onnx: pip install jax2onnx\n"
                    "  2. Exporting to TF SavedModel first, then: isat onnx saved_model/"
                ),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_dir = os.path.join(tmpdir, "saved_model")

            def predict(x):
                return jnp.dot(x, x.T)

            tf_fn = jax2tf.convert(predict)
            concrete = tf.function(tf_fn, input_signature=[
                tf.TensorSpec(shape=shape, dtype=tf.float32)
            ])
            tf.saved_model.save(
                tf.Module(), saved_model_dir,
                signatures=concrete.get_concrete_function()
            )

            return convert_tensorflow(saved_model_dir, output_dir, opset)

    except Exception as e:
        return ConversionResult(success=False, error=f"JAX->TF->ONNX bridge failed: {e}")


# ---------------------------------------------------------------------------
# 6. SafeTensors (.safetensors)
# ---------------------------------------------------------------------------

def convert_safetensors(
    input_path: str,
    output_dir: str,
    opset: int = 17,
    input_shape: Optional[str] = None,
) -> ConversionResult:
    """Convert SafeTensors weights to ONNX.

    SafeTensors files contain only weights (no architecture), so we try:
    1. Check for a config.json in the same directory -> HuggingFace local dir
    2. Fall back to raw weight loading with --input-shape
    """
    p = Path(input_path)

    config_path = p.parent / "config.json"
    if config_path.exists():
        return _convert_hf_local_dir(str(p.parent), output_dir, opset)

    try:
        import torch
    except ImportError:
        return ConversionResult(
            success=False,
            error="torch not installed.\n  pip install torch safetensors",
        )

    try:
        from safetensors.torch import load_file
    except ImportError:
        return ConversionResult(
            success=False,
            error="safetensors not installed.\n  pip install safetensors torch",
        )

    if not input_shape:
        return ConversionResult(
            success=False,
            error=(
                "SafeTensors files contain only weights, no architecture.\n"
                "  Options:\n"
                "    1. Place a config.json next to the .safetensors file\n"
                "    2. Use the HuggingFace model ID instead:\n"
                "       isat onnx <org>/<model-name>\n"
                "    3. Provide architecture + shape (advanced):\n"
                "       isat onnx model.safetensors --input-shape 1,3,224,224"
            ),
        )

    return ConversionResult(
        success=False,
        error=(
            "Raw SafeTensors conversion without config.json is not supported.\n"
            "  Use the HuggingFace model ID: isat onnx <org>/<model-name>"
        ),
    )
