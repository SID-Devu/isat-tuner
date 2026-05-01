"""Per-layer precision profiler for ONNX models.

Quantizes each layer individually to measure accuracy impact, latency, and
memory delta, producing a full precision profile used by the optimizer to
find Pareto-optimal mixed-precision assignments.
"""

from __future__ import annotations

import copy
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

log = logging.getLogger("isat.amp")

PrecisionProfile = Dict[str, Dict[str, "LayerPrecisionResult"]]


@dataclass
class LayerPrecisionResult:
    layer_name: str
    precision: str
    mse: float
    cosine_sim: float
    latency_ms: float
    size_delta_mb: float


class PrecisionProfiler:
    """Profile every quantizable layer at multiple precisions.

    For each layer, creates a model variant with only that layer quantized
    to the target precision, then measures accuracy (MSE / cosine similarity
    vs full-precision golden outputs), latency, and model-size delta.
    """

    QUANTIZABLE_OPS = {"MatMul", "Gemm", "Conv", "ConvTranspose"}

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.provider = provider

        import onnx

        self._model = onnx.load(self.model_path)
        self._quantizable_layers = self._get_quantizable_layers(self._model.graph)
        self._base_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)

        log.info(
            "Loaded %s (%.1f MB, %d quantizable layers)",
            self.model_path,
            self._base_size_mb,
            len(self._quantizable_layers),
        )

    def _get_quantizable_layers(self, graph) -> list[dict[str, Any]]:
        """Find layers that carry weights and can be quantized."""
        import onnx
        from onnx import numpy_helper

        init_names = {init.name for init in graph.initializer}
        layers = []

        for node in graph.node:
            if node.op_type not in self.QUANTIZABLE_OPS:
                continue
            weight_inputs = [n for n in node.input if n in init_names]
            if not weight_inputs:
                continue
            layers.append({
                "name": node.name or f"{node.op_type}_{len(layers)}",
                "op_type": node.op_type,
                "weight_inputs": weight_inputs,
            })

        return layers

    def _generate_reference(self, num_samples: int) -> tuple[list[dict], list[np.ndarray]]:
        """Run full-precision model on random inputs, return (inputs, golden_outputs)."""
        import onnxruntime as ort

        sess = ort.InferenceSession(self.model_path, providers=[self.provider])
        input_meta = sess.get_inputs()
        output_names = [o.name for o in sess.get_outputs()]

        rng = np.random.RandomState(42)
        _type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
            "tensor(double)": np.float64,
        }

        inputs_list: list[dict] = []
        outputs_list: list[np.ndarray] = []

        for _ in range(num_samples):
            feed: dict[str, np.ndarray] = {}
            for inp in input_meta:
                dtype = _type_map.get(inp.type, np.float32)
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                feed[inp.name] = rng.randn(*shape).astype(dtype)
            inputs_list.append(feed)

            raw = sess.run(output_names, feed)
            flat = np.concatenate([o.flatten().astype(np.float32) for o in raw])
            outputs_list.append(flat)

        return inputs_list, outputs_list

    def _quantize_single_layer(self, model, layer_name: str, precision: str):
        """Return a copy of *model* with only the named layer's weights converted."""
        import onnx
        from onnx import numpy_helper

        model_copy = copy.deepcopy(model)

        target_node = None
        for node in model_copy.graph.node:
            node_name = node.name or f"{node.op_type}_anon"
            if node_name == layer_name:
                target_node = node
                break

        if target_node is None:
            raise ValueError(f"Layer {layer_name!r} not found in graph")

        init_map = {init.name: init for init in model_copy.graph.initializer}

        for inp_name in target_node.input:
            if inp_name not in init_map:
                continue
            init = init_map[inp_name]
            arr = numpy_helper.to_array(init).astype(np.float32)

            if precision == "fp16":
                converted = arr.astype(np.float16).astype(np.float32)
            elif precision == "int8":
                if arr.max() == arr.min():
                    converted = np.zeros_like(arr)
                else:
                    scale = (arr.max() - arr.min()) / 255.0
                    zp = np.round(-arr.min() / scale).astype(np.int32)
                    quantized = np.clip(np.round(arr / scale) + zp, 0, 255).astype(np.uint8)
                    converted = (quantized.astype(np.float32) - zp) * scale
            elif precision == "int4":
                flat = arr.flatten()
                block = 128
                pad = (block - len(flat) % block) % block
                if pad:
                    flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
                blocks = flat.reshape(-1, block)
                scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 7.0
                scales = np.where(scales == 0, 1.0, scales)
                q = np.clip(np.round(blocks / scales), -8, 7).astype(np.int8)
                deq = (q.astype(np.float32) * scales).flatten()[: arr.size]
                converted = deq.reshape(arr.shape)
            elif precision == "fp32":
                converted = arr
            else:
                raise ValueError(f"Unsupported precision: {precision}")

            new_tensor = numpy_helper.from_array(converted, name=init.name)
            init.CopyFrom(new_tensor)

        return model_copy

    def profile_layer(
        self,
        layer_name: str,
        precision: str,
        reference_outputs: list[np.ndarray],
        inputs_list: list[dict],
        num_samples: int = 50,
    ) -> LayerPrecisionResult:
        """Quantize one layer to *precision*, measure accuracy and latency."""
        import onnx
        import onnxruntime as ort

        if precision == "fp32":
            return LayerPrecisionResult(
                layer_name=layer_name,
                precision="fp32",
                mse=0.0,
                cosine_sim=1.0,
                latency_ms=0.0,
                size_delta_mb=0.0,
            )

        variant = self._quantize_single_layer(self._model, layer_name, precision)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_path = f.name
        try:
            onnx.save(variant, tmp_path)
            variant_size = os.path.getsize(tmp_path) / (1024 * 1024)
            size_delta = variant_size - self._base_size_mb

            sess = ort.InferenceSession(tmp_path, providers=[self.provider])
            output_names = [o.name for o in sess.get_outputs()]

            n = min(num_samples, len(inputs_list))
            mse_vals = []
            cos_vals = []
            latencies = []

            for i in range(n):
                t0 = time.perf_counter()
                raw = sess.run(output_names, inputs_list[i])
                latencies.append((time.perf_counter() - t0) * 1000)

                out_flat = np.concatenate([o.flatten().astype(np.float32) for o in raw])
                ref = reference_outputs[i]
                length = min(len(out_flat), len(ref))
                out_flat, ref_slice = out_flat[:length], ref[:length]

                mse_vals.append(float(np.mean((out_flat - ref_slice) ** 2)))

                norm_a = np.linalg.norm(ref_slice)
                norm_b = np.linalg.norm(out_flat)
                if norm_a > 0 and norm_b > 0:
                    cos_vals.append(float(np.dot(ref_slice, out_flat) / (norm_a * norm_b)))
                else:
                    cos_vals.append(1.0)

            return LayerPrecisionResult(
                layer_name=layer_name,
                precision=precision,
                mse=float(np.mean(mse_vals)) if mse_vals else 0.0,
                cosine_sim=float(np.mean(cos_vals)) if cos_vals else 1.0,
                latency_ms=float(np.mean(latencies)) if latencies else 0.0,
                size_delta_mb=round(size_delta, 4),
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def profile_all(
        self,
        precisions: Optional[list[str]] = None,
        num_samples: int = 50,
        batch_size: int = 10,
    ) -> PrecisionProfile:
        """Profile every quantizable layer at every requested precision.

        Processes layers in batches of *batch_size* to limit peak memory on
        large models.
        """
        if precisions is None:
            precisions = ["fp32", "fp16", "int8", "int4"]

        log.info(
            "Generating %d reference samples for golden outputs",
            num_samples,
        )
        inputs_list, reference_outputs = self._generate_reference(num_samples)

        profile: PrecisionProfile = {}
        total = len(self._quantizable_layers)

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch = self._quantizable_layers[batch_start:batch_end]
            log.info(
                "Profiling layers %d–%d of %d",
                batch_start + 1,
                batch_end,
                total,
            )

            for layer_info in batch:
                name = layer_info["name"]
                profile[name] = {}
                for prec in precisions:
                    try:
                        result = self.profile_layer(
                            name, prec, reference_outputs, inputs_list, num_samples
                        )
                        profile[name][prec] = result
                        log.debug(
                            "  %s @ %s: mse=%.6f cos=%.4f lat=%.2fms",
                            name,
                            prec,
                            result.mse,
                            result.cosine_sim,
                            result.latency_ms,
                        )
                    except Exception:
                        log.exception("Failed to profile %s @ %s", name, prec)

        log.info("Profiling complete: %d layers × %d precisions", total, len(precisions))
        return profile
