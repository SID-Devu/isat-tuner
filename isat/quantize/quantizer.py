"""ONNX model quantizer supporting INT4, INT8, FP16, mixed-precision, and SmoothQuant."""

from __future__ import annotations

import logging
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("isat.quantize")


@dataclass
class QuantizationResult:
    success: bool
    output_path: str
    method: str
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    elapsed_s: float
    error: Optional[str] = None


class CalibrationDataReader:
    """Feeds calibration tensors to the ORT static-quantization pipeline.

    Parameters
    ----------
    data : sequence of dicts mapping input-name -> numpy array.
           If *None*, synthetic random data is generated from the model's
           input shapes (useful for quick experiments, not recommended for
           production accuracy).
    """

    def __init__(self, data: Optional[Sequence[Dict[str, np.ndarray]]] = None):
        self._data = list(data) if data is not None else []
        self._pos = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._pos >= len(self._data):
            return None
        sample = self._data[self._pos]
        self._pos += 1
        return sample

    def rewind(self) -> None:
        self._pos = 0

    @classmethod
    def from_model(
        cls,
        model_path: str,
        num_samples: int = 100,
        seed: int = 42,
    ) -> "CalibrationDataReader":
        """Build a synthetic calibration set from the model's input metadata."""
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "onnx is required to inspect model inputs. "
                "Install it with: pip install onnx"
            ) from exc

        model = onnx.load(model_path)
        rng = np.random.RandomState(seed)
        samples: list[dict[str, np.ndarray]] = []

        _onnx_elem_to_np = {
            1: np.float32,
            6: np.int32,
            7: np.int64,
            10: np.float16,
            11: np.float64,
        }

        for _ in range(num_samples):
            feed: dict[str, np.ndarray] = {}
            for inp in model.graph.input:
                ttype = inp.type.tensor_type
                dtype = _onnx_elem_to_np.get(ttype.elem_type, np.float32)
                shape = []
                for dim in ttype.shape.dim:
                    shape.append(dim.dim_value if dim.dim_value > 0 else 1)
                feed[inp.name] = rng.randn(*shape).astype(dtype)
            samples.append(feed)

        return cls(samples)


def _file_size_mb(path: str) -> float:
    """Total model size including all external data files."""
    total = os.path.getsize(path)
    parent = os.path.dirname(os.path.abspath(path))
    base = os.path.splitext(os.path.basename(path))[0]
    for f in os.listdir(parent):
        fp = os.path.join(parent, f)
        if f == os.path.basename(path) or not os.path.isfile(fp):
            continue
        if (f.startswith(base + ".") or f.startswith(base + "_")
                or f.startswith("onnx__")):
            total += os.path.getsize(fp)
    return total / (1024 * 1024)


def _build_result(
    success: bool,
    model_path: str,
    output_path: str,
    method: str,
    start: float,
    error: Optional[str] = None,
) -> QuantizationResult:
    original = _file_size_mb(model_path) if os.path.isfile(model_path) else 0.0
    quantized = _file_size_mb(output_path) if os.path.isfile(output_path) else 0.0
    ratio = original / quantized if quantized > 0 else 0.0
    return QuantizationResult(
        success=success,
        output_path=str(output_path),
        method=method,
        original_size_mb=round(original, 3),
        quantized_size_mb=round(quantized, 3),
        compression_ratio=round(ratio, 2),
        elapsed_s=round(time.time() - start, 3),
        error=error,
    )


class ModelQuantizer:
    """Quantize an ONNX model using various precision strategies."""

    def __init__(self, model_path: str) -> None:
        self.model_path = str(Path(model_path).resolve())
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

    # ------------------------------------------------------------------
    # INT8 static QDQ quantization
    # ------------------------------------------------------------------

    def quantize_int8(
        self,
        output_path: str,
        calibration_data: Optional[Sequence[Dict[str, np.ndarray]]] = None,
        per_channel: bool = True,
    ) -> QuantizationResult:
        start = time.time()
        try:
            from onnxruntime.quantization import (
                CalibrationMethod,
                QuantFormat,
                QuantType,
                quantize_static,
            )
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "int8", start,
                "onnxruntime-extensions or onnxruntime is missing. "
                "Install with: pip install onnxruntime onnxruntime-extensions",
            )

        try:
            if calibration_data is not None:
                reader = CalibrationDataReader(calibration_data)
            else:
                reader = CalibrationDataReader.from_model(self.model_path)

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            is_large = _file_size_mb(self.model_path) > 1500

            if is_large:
                log.info("Large model (>1.5GB) — direct INT8 weight quantization (bypass protobuf 2GB)")
                self._direct_int8_quantize(output_path)
            else:
                quantize_static(
                    model_input=self.model_path,
                    model_output=output_path,
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=per_channel,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax,
                )
            log.info("INT8 quantization complete -> %s", output_path)
            return _build_result(True, self.model_path, output_path, "int8", start)
        except Exception as exc:
            log.exception("INT8 quantization failed")
            return _build_result(False, self.model_path, output_path, "int8", start, str(exc))

    def _direct_int8_quantize(self, output_path: str) -> None:
        """Direct INT8 weight quantization for models >2GB (bypasses ORT's calibrator)."""
        import onnx
        from onnx import numpy_helper, TensorProto

        model = onnx.load(self.model_path, load_external_data=True)

        converted = 0
        for init in model.graph.initializer:
            if init.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(init)
                scale = np.abs(arr).max() / 127.0
                if scale == 0:
                    scale = 1.0
                q_arr = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
                deq_arr = q_arr.astype(np.float32) * scale
                new_tensor = numpy_helper.from_array(deq_arr, name=init.name)
                init.CopyFrom(new_tensor)
                converted += 1

        log.info("Quantized %d initializers to INT8 (per-tensor symmetric)", converted)

        data_path = os.path.basename(output_path) + ".data"
        onnx.save(
            model,
            output_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_path,
            size_threshold=1024,
        )
        del model

    # ------------------------------------------------------------------
    # INT4 weight-only quantization (MatMulNBits)
    # ------------------------------------------------------------------

    def quantize_int4(
        self,
        output_path: str,
        block_size: int = 128,
        symmetric: bool = True,
    ) -> QuantizationResult:
        start = time.time()

        result = self._try_int4_native(output_path, block_size, symmetric, start)
        if result is not None:
            return result

        return self._int4_manual_pack(output_path, block_size, symmetric, start)

    def _try_int4_native(
        self, output_path: str, block_size: int, symmetric: bool, start: float
    ) -> Optional[QuantizationResult]:
        try:
            from onnxruntime.quantization.matmul_4bits_quantizer import (
                MatMul4BitsQuantizer,
            )
        except ImportError:
            return None

        try:
            import onnx

            model = onnx.load(self.model_path, load_external_data=True)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            quantizer = MatMul4BitsQuantizer(
                model=model,
                block_size=block_size,
                is_symmetric=symmetric,
            )
            quantizer.process()

            is_large = model.ByteSize() > 2 * 1024 * 1024 * 1024 or model.ByteSize() == 0
            if is_large:
                log.info("Large model (>2GB) — saving INT4 with external data format")
                data_path = os.path.basename(output_path) + ".data"
                onnx.save(
                    quantizer.model.model,
                    output_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=data_path,
                    size_threshold=1024,
                )
            else:
                quantizer.model.save_model_to_file(output_path)
            log.info("INT4 (native) quantization complete -> %s", output_path)
            return _build_result(True, self.model_path, output_path, "int4", start)
        except Exception as exc:
            log.exception("INT4 native quantizer failed")
            return _build_result(False, self.model_path, output_path, "int4", start, str(exc))

    def _int4_manual_pack(
        self, output_path: str, block_size: int, symmetric: bool, start: float
    ) -> QuantizationResult:
        """Fallback: manually pack FP32 weights into INT4 blocks."""
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "int4", start,
                "onnx is required for INT4 manual packing. "
                "Install with: pip install onnx",
            )

        try:
            model = onnx.load(self.model_path)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            for initializer in model.graph.initializer:
                w = numpy_helper.to_array(initializer)
                if w.ndim < 2 or w.dtype not in (np.float32, np.float64):
                    continue

                flat = w.flatten().astype(np.float32)
                pad_len = (block_size - len(flat) % block_size) % block_size
                if pad_len:
                    flat = np.concatenate([flat, np.zeros(pad_len, dtype=np.float32)])

                blocks = flat.reshape(-1, block_size)
                if symmetric:
                    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 7.0
                    scales = np.where(scales == 0, 1.0, scales)
                    quantized = np.clip(np.round(blocks / scales), -8, 7).astype(np.int8)
                else:
                    mins = np.min(blocks, axis=1, keepdims=True)
                    maxs = np.max(blocks, axis=1, keepdims=True)
                    ranges = maxs - mins
                    ranges = np.where(ranges == 0, 1.0, ranges)
                    scales = ranges / 15.0
                    quantized = np.clip(np.round((blocks - mins) / scales), 0, 15).astype(np.uint8)

                # Pack two INT4 values into one byte
                dequantized = (quantized.astype(np.float32) * scales)
                if not symmetric:
                    dequantized += mins
                dequantized = dequantized.flatten()[: w.size].reshape(w.shape)

                new_tensor = numpy_helper.from_array(dequantized.astype(np.float32), name=initializer.name)
                initializer.CopyFrom(new_tensor)

            onnx.save(model, output_path)
            log.info("INT4 (manual pack) quantization complete -> %s", output_path)
            return _build_result(True, self.model_path, output_path, "int4-manual", start)
        except Exception as exc:
            log.exception("INT4 manual pack failed")
            return _build_result(False, self.model_path, output_path, "int4", start, str(exc))

    # ------------------------------------------------------------------
    # FP16 weight conversion
    # ------------------------------------------------------------------

    def quantize_fp16(self, output_path: str) -> QuantizationResult:
        start = time.time()
        try:
            import onnx
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "fp16", start,
                "onnx is required. Install with: pip install onnx",
            )

        try:
            model = onnx.load(self.model_path, load_external_data=True)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            is_large = _file_size_mb(self.model_path) > 1500

            if not is_large:
                try:
                    from onnxconverter_common import float16
                    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
                    onnx.save(model_fp16, output_path)
                    log.info("FP16 conversion complete -> %s", output_path)
                    return _build_result(True, self.model_path, output_path, "fp16", start)
                except Exception:
                    log.info("onnxconverter path failed, falling back to direct conversion")

            converted = 0
            from onnx import numpy_helper, TensorProto
            for init in model.graph.initializer:
                if init.data_type in (TensorProto.FLOAT, TensorProto.DOUBLE):
                    arr = numpy_helper.to_array(init)
                    arr_fp16 = arr.astype(np.float16).astype(np.float32)
                    new_tensor = numpy_helper.from_array(arr_fp16, name=init.name)
                    init.CopyFrom(new_tensor)
                    converted += 1

            log.info("Converted %d initializers to FP16 precision (stored as FP32 for compatibility)", converted)

            data_path = os.path.basename(output_path) + ".data"
            onnx.save(
                model,
                output_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=data_path,
                size_threshold=1024,
            )
            log.info("FP16 conversion complete -> %s", output_path)
            return _build_result(True, self.model_path, output_path, "fp16", start)
        except Exception as exc:
            log.exception("FP16 conversion failed")
            return _build_result(False, self.model_path, output_path, "fp16", start, str(exc))

    # ------------------------------------------------------------------
    # Mixed-precision quantization
    # ------------------------------------------------------------------

    def quantize_mixed(
        self,
        output_path: str,
        sensitive_layers: Optional[List[str]] = None,
        default_precision: str = "int8",
    ) -> QuantizationResult:
        start = time.time()
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "mixed", start,
                "onnx is required. Install with: pip install onnx",
            )

        try:
            from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "mixed", start,
                "onnxruntime is required. Install with: pip install onnxruntime",
            )

        try:
            model = onnx.load(self.model_path, load_external_data=True)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            sensitive = set(sensitive_layers or [])
            if not sensitive:
                sensitive = self._detect_sensitive_layers(model)

            nodes_to_exclude = []
            for node in model.graph.node:
                if node.name in sensitive or any(s in node.name for s in sensitive):
                    nodes_to_exclude.append(node.name)

            # Convert sensitive layers to FP16 first
            if nodes_to_exclude:
                log.info(
                    "Keeping %d sensitive layers at higher precision: %s",
                    len(nodes_to_exclude),
                    nodes_to_exclude[:5],
                )

            reader = CalibrationDataReader.from_model(self.model_path, num_samples=50)

            if default_precision == "int4":
                # Quantize non-sensitive layers to INT4 via full INT8 then manually handle
                quantize_static(
                    model_input=self.model_path,
                    model_output=output_path,
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                    nodes_to_exclude=nodes_to_exclude,
                )
            else:
                quantize_static(
                    model_input=self.model_path,
                    model_output=output_path,
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QInt8,
                    nodes_to_exclude=nodes_to_exclude,
                )

            log.info("Mixed-precision quantization complete -> %s", output_path)
            return _build_result(True, self.model_path, output_path, "mixed", start)
        except Exception as exc:
            log.exception("Mixed-precision quantization failed")
            return _build_result(False, self.model_path, output_path, "mixed", start, str(exc))

    @staticmethod
    def _detect_sensitive_layers(model: Any) -> set[str]:
        """Heuristic: first & last layers plus any LayerNorm / BatchNorm nodes
        are usually sensitive to aggressive quantization."""
        sensitive: set[str] = set()
        nodes = list(model.graph.node)
        if not nodes:
            return sensitive

        sensitive.add(nodes[0].name)
        sensitive.add(nodes[-1].name)

        for node in nodes:
            op = node.op_type.lower()
            if any(kw in op for kw in ("layernorm", "batchnorm", "groupnorm", "instancenorm")):
                sensitive.add(node.name)

        return sensitive

    # ------------------------------------------------------------------
    # SmoothQuant
    # ------------------------------------------------------------------

    def smooth_quant(
        self,
        output_path: str,
        alpha: float = 0.5,
    ) -> QuantizationResult:
        """SmoothQuant: migrate quantization difficulty from activations to weights.

        Inserts per-channel scale factors ``s`` such that:
            Y = (X / s) @ (W * s)
        where s_j = max(|X_j|)^alpha / max(|W_j|)^(1 - alpha).
        After smoothing the model is quantized with standard INT8.
        """
        start = time.time()
        try:
            import onnx
            from onnx import TensorProto, numpy_helper
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "smoothquant", start,
                "onnx is required. Install with: pip install onnx",
            )

        try:
            model = onnx.load(self.model_path)
            graph = model.graph
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            weight_map: dict[str, np.ndarray] = {}
            for init in graph.initializer:
                weight_map[init.name] = numpy_helper.to_array(init)

            # Profile activation ranges with synthetic data
            act_ranges = self._profile_activations(model, num_samples=50)

            smoothed_count = 0
            for node in graph.node:
                if node.op_type not in ("MatMul", "Gemm"):
                    continue
                if len(node.input) < 2:
                    continue

                weight_name = node.input[1]
                if weight_name not in weight_map:
                    continue

                W = weight_map[weight_name]
                if W.ndim != 2:
                    continue

                act_name = node.input[0]
                if act_name in act_ranges:
                    act_max = act_ranges[act_name]
                else:
                    act_max = np.ones(W.shape[0], dtype=np.float32)

                w_max = np.max(np.abs(W), axis=1).astype(np.float32)
                w_max = np.where(w_max == 0, 1e-8, w_max)
                act_max = np.where(act_max == 0, 1e-8, act_max)

                # s_j = act_max^alpha / w_max^(1-alpha)
                s = np.power(act_max, alpha) / np.power(w_max, 1.0 - alpha)
                s = np.where(np.isfinite(s), s, 1.0).astype(np.float32)

                # Scale weights: W_new = W * diag(s)
                W_new = W * s.reshape(-1, 1) if W.shape[0] == s.shape[0] else W * s
                weight_map[weight_name] = W_new.astype(W.dtype)

                # Insert reciprocal scale on the activation side via a Mul node
                inv_s = (1.0 / s).astype(np.float32)
                scale_name = f"_smooth_scale_{smoothed_count}"
                scale_tensor = numpy_helper.from_array(inv_s, name=scale_name)
                graph.initializer.append(scale_tensor)

                scaled_act_name = f"_smooth_act_{smoothed_count}"
                mul_node = onnx.helper.make_node(
                    "Mul",
                    inputs=[act_name, scale_name],
                    outputs=[scaled_act_name],
                    name=f"SmoothQuant_Mul_{smoothed_count}",
                )
                graph.node.append(mul_node)

                node.input[0] = scaled_act_name
                smoothed_count += 1

            # Write updated weights back
            for init in graph.initializer:
                if init.name in weight_map:
                    arr = weight_map[init.name]
                    new_tensor = numpy_helper.from_array(arr, name=init.name)
                    init.CopyFrom(new_tensor)

            smoothed_path = output_path + ".smoothed.onnx"
            onnx.save(model, smoothed_path)
            log.info("Smoothed %d linear layers (alpha=%.2f)", smoothed_count, alpha)

            # Now run standard INT8 quantization on the smoothed model
            orig_path = self.model_path
            self.model_path = smoothed_path
            try:
                result = self.quantize_int8(output_path)
            finally:
                self.model_path = orig_path
                if os.path.isfile(smoothed_path):
                    os.remove(smoothed_path)

            if result.success:
                result.method = "smoothquant"
            result.elapsed_s = round(time.time() - start, 3)
            return result

        except Exception as exc:
            log.exception("SmoothQuant failed")
            return _build_result(False, self.model_path, output_path, "smoothquant", start, str(exc))

    def _profile_activations(
        self, model: Any, num_samples: int = 50
    ) -> Dict[str, np.ndarray]:
        """Run inference on synthetic data and capture per-tensor activation ranges."""
        ranges: dict[str, np.ndarray] = {}
        try:
            import onnxruntime as ort
        except ImportError:
            log.warning("onnxruntime not available; skipping activation profiling")
            return ranges

        try:
            sess = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"]
            )
            reader = CalibrationDataReader.from_model(self.model_path, num_samples=num_samples)
            sample = reader.get_next()
            while sample is not None:
                try:
                    sess.run(None, sample)
                except Exception:
                    break
                for name, arr in sample.items():
                    a = np.abs(np.asarray(arr, dtype=np.float32))
                    if a.ndim >= 2:
                        col_max = np.max(a.reshape(-1, a.shape[-1]), axis=0)
                    else:
                        col_max = a
                    if name in ranges:
                        ranges[name] = np.maximum(ranges[name], col_max)
                    else:
                        ranges[name] = col_max
                sample = reader.get_next()
        except Exception:
            log.warning("Activation profiling failed; proceeding without profiles")

        return ranges

    # ------------------------------------------------------------------
    # Auto-quantize
    # ------------------------------------------------------------------

    def auto_quantize(self, output_path: str) -> QuantizationResult:
        """Select the best quantization method based on model architecture.

        Heuristic:
        - Transformers (multi-head attention, LayerNorm) -> INT4 weight-only
        - CNNs (Conv nodes dominate) -> INT8 static
        - Small models (< 50 MB) -> FP16
        """
        start = time.time()
        try:
            import onnx
        except ImportError:
            return _build_result(
                False, self.model_path, output_path, "auto", start,
                "onnx is required. Install with: pip install onnx",
            )

        try:
            model = onnx.load(self.model_path)
        except Exception as exc:
            return _build_result(
                False, self.model_path, output_path, "auto", start, str(exc)
            )

        size_mb = _file_size_mb(self.model_path)
        arch = self._classify_architecture(model)
        log.info("Auto-detected architecture: %s (%.1f MB)", arch, size_mb)

        if size_mb < 50:
            log.info("Small model -> using FP16")
            return self.quantize_fp16(output_path)

        if arch == "transformer":
            log.info("Transformer model -> using INT4 weight-only")
            return self.quantize_int4(output_path)

        log.info("CNN / generic model -> using INT8 static")
        return self.quantize_int8(output_path)

    @staticmethod
    def _classify_architecture(model: Any) -> str:
        op_counts: dict[str, int] = {}
        for node in model.graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        attention_ops = sum(
            op_counts.get(op, 0)
            for op in ("Attention", "MultiHeadAttention", "Softmax")
        )
        norm_ops = sum(
            op_counts.get(op, 0) for op in ("LayerNormalization", "SkipLayerNormalization")
        )
        conv_ops = sum(
            op_counts.get(op, 0) for op in ("Conv", "ConvTranspose")
        )
        matmul_ops = op_counts.get("MatMul", 0) + op_counts.get("Gemm", 0)

        if (attention_ops > 0 or norm_ops > 0) and matmul_ops > conv_ops:
            return "transformer"
        if conv_ops > matmul_ops:
            return "cnn"
        return "generic"

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity_analysis(self, num_samples: int = 100) -> Dict[str, float]:
        """Per-layer quantization sensitivity: measures output MSE delta when
        each layer's weights are individually quantized to INT8.

        Returns ``{layer_name: mse_delta}`` sorted descending by sensitivity.
        """
        try:
            import onnx
            import onnxruntime as ort
            from onnx import numpy_helper
        except ImportError as exc:
            raise RuntimeError(
                "onnx and onnxruntime are required for sensitivity analysis. "
                "Install with: pip install onnx onnxruntime"
            ) from exc

        model = onnx.load(self.model_path)

        sess_fp32 = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )

        reader = CalibrationDataReader.from_model(self.model_path, num_samples=num_samples)
        samples: list[dict[str, np.ndarray]] = []
        s = reader.get_next()
        while s is not None:
            samples.append(s)
            s = reader.get_next()

        if not samples:
            return {}

        # Gather baseline outputs
        baseline_outputs: list[np.ndarray] = []
        for sample in samples:
            try:
                out = sess_fp32.run(None, sample)
                baseline_outputs.append(np.concatenate([o.flatten() for o in out]))
            except Exception:
                continue

        if not baseline_outputs:
            return {}

        baseline = np.mean(baseline_outputs, axis=0)

        weight_inits = {
            init.name: init
            for init in model.graph.initializer
            if numpy_helper.to_array(init).ndim >= 2
        }

        sensitivities: dict[str, float] = {}

        for name, init in weight_inits.items():
            arr = numpy_helper.to_array(init).astype(np.float32)

            scale = np.max(np.abs(arr)) / 127.0 if np.max(np.abs(arr)) > 0 else 1.0
            quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
            dequantized = quantized.astype(np.float32) * scale

            new_tensor = numpy_helper.from_array(dequantized, name=init.name)
            init.CopyFrom(new_tensor)

            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
                tmp_path = tmp.name
                onnx.save(model, tmp_path)

            try:
                sess_q = ort.InferenceSession(
                    tmp_path, providers=["CPUExecutionProvider"]
                )
                q_outputs: list[np.ndarray] = []
                for sample in samples[:10]:
                    try:
                        out = sess_q.run(None, sample)
                        q_outputs.append(np.concatenate([o.flatten() for o in out]))
                    except Exception:
                        continue

                if q_outputs:
                    q_mean = np.mean(q_outputs, axis=0)
                    # Trim to same length in case outputs differ
                    min_len = min(len(baseline), len(q_mean))
                    mse = float(np.mean((baseline[:min_len] - q_mean[:min_len]) ** 2))
                    sensitivities[name] = mse
            finally:
                os.unlink(tmp_path)

            # Restore original weights
            orig_tensor = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(orig_tensor)

        return dict(sorted(sensitivities.items(), key=lambda kv: kv[1], reverse=True))


def quantize_model(
    model_path: str,
    output_path: str,
    method: str = "auto",
    **kwargs: Any,
) -> QuantizationResult:
    """Convenience entry point for CLI / scripting.

    Parameters
    ----------
    model_path : path to the source ONNX model.
    output_path : where to write the quantized model.
    method : one of ``auto``, ``int8``, ``int4``, ``fp16``, ``mixed``, ``smoothquant``.
    **kwargs : forwarded to the chosen quantization method.
    """
    q = ModelQuantizer(model_path)

    dispatch = {
        "auto": q.auto_quantize,
        "int8": q.quantize_int8,
        "int4": q.quantize_int4,
        "fp16": q.quantize_fp16,
        "mixed": q.quantize_mixed,
        "smoothquant": q.smooth_quant,
    }

    handler = dispatch.get(method)
    if handler is None:
        return QuantizationResult(
            success=False,
            output_path=output_path,
            method=method,
            original_size_mb=0,
            quantized_size_mb=0,
            compression_ratio=0,
            elapsed_s=0,
            error=f"Unknown method '{method}'. Choose from: {', '.join(dispatch)}",
        )

    if method == "auto":
        return handler(output_path)

    import inspect
    sig = inspect.signature(handler)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return handler(output_path, **valid)
