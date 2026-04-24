"""Model optimization pipeline.

Goes beyond benchmarking -- actually transforms and saves optimized models:
  1. Graph simplification (onnxsim)
  2. Shape freezing (pin dynamic dims)
  3. Precision conversion (FP16, INT8 QDQ)
  4. ORT graph optimization and export
  5. Operator fusion analysis
  6. Dead node elimination
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.optimizer")


@dataclass
class OptimizationResult:
    original_path: str
    optimized_path: str
    original_size_mb: float
    optimized_size_mb: float
    size_reduction_pct: float
    transforms_applied: list[str]
    original_nodes: int
    optimized_nodes: int
    node_reduction_pct: float
    elapsed_s: float
    errors: list[str] = field(default_factory=list)


class OptimizationPipeline:
    """End-to-end model optimization pipeline."""

    def __init__(self, output_dir: str = "isat_optimized"):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def optimize(
        self,
        model_path: str,
        *,
        simplify: bool = True,
        pin_shapes: Optional[dict[str, list[int]]] = None,
        convert_fp16: bool = False,
        quantize_int8: bool = False,
        ort_optimize: bool = True,
        ort_opt_level: int = 99,
    ) -> OptimizationResult:
        """Run the full optimization pipeline."""
        start = time.time()
        transforms: list[str] = []
        errors: list[str] = []

        import onnx
        original_size = Path(model_path).stat().st_size / (1024 * 1024)
        original_model = onnx.load(model_path, load_external_data=False)
        original_nodes = len(original_model.graph.node)

        current_path = model_path
        stem = Path(model_path).stem

        if simplify:
            result = self._simplify(current_path, stem)
            if result:
                current_path = result
                transforms.append("onnxsim")
            else:
                errors.append("onnxsim failed (non-fatal)")

        if pin_shapes:
            result = self._pin_shapes(current_path, stem, pin_shapes)
            if result:
                current_path = result
                transforms.append("shape_pinning")

        if convert_fp16:
            result = self._convert_fp16(current_path, stem)
            if result:
                current_path = result
                transforms.append("fp16_conversion")
            else:
                errors.append("FP16 conversion failed (non-fatal)")

        if quantize_int8:
            result = self._quantize_int8(current_path, stem)
            if result:
                current_path = result
                transforms.append("int8_qdq")
            else:
                errors.append("INT8 quantization failed (non-fatal)")

        if ort_optimize:
            result = self._ort_optimize(current_path, stem, ort_opt_level)
            if result:
                current_path = result
                transforms.append(f"ort_opt_{ort_opt_level}")

        final_path = str(Path(self.output_dir) / f"{stem}_optimized.onnx")
        if current_path != model_path:
            shutil.copy2(current_path, final_path)
        else:
            shutil.copy2(model_path, final_path)
            transforms.append("copy_only")

        optimized_size = Path(final_path).stat().st_size / (1024 * 1024)
        optimized_model = onnx.load(final_path, load_external_data=False)
        optimized_nodes = len(optimized_model.graph.node)

        elapsed = time.time() - start

        result = OptimizationResult(
            original_path=model_path,
            optimized_path=final_path,
            original_size_mb=original_size,
            optimized_size_mb=optimized_size,
            size_reduction_pct=((original_size - optimized_size) / original_size * 100) if original_size > 0 else 0,
            transforms_applied=transforms,
            original_nodes=original_nodes,
            optimized_nodes=optimized_nodes,
            node_reduction_pct=((original_nodes - optimized_nodes) / original_nodes * 100) if original_nodes > 0 else 0,
            elapsed_s=elapsed,
            errors=errors,
        )

        log.info("Optimization complete: %s -> %s (%.1f%% size reduction, %.1f%% node reduction)",
                 model_path, final_path, result.size_reduction_pct, result.node_reduction_pct)
        return result

    def _simplify(self, model_path: str, stem: str) -> Optional[str]:
        try:
            import onnxsim
            import onnx
            model = onnx.load(model_path)
            model_sim, ok = onnxsim.simplify(model)
            if ok:
                out = str(Path(self.output_dir) / f"{stem}_sim.onnx")
                onnx.save(model_sim, out)
                return out
        except Exception as e:
            log.warning("onnxsim failed: %s", e)
        return None

    def _pin_shapes(self, model_path: str, stem: str, shapes: dict[str, list[int]]) -> Optional[str]:
        try:
            import onnx
            from onnx.tools import update_model_dims
            model = onnx.load(model_path)
            for inp in model.graph.input:
                if inp.name in shapes:
                    dims = shapes[inp.name]
                    for i, d in enumerate(dims):
                        if i < len(inp.type.tensor_type.shape.dim):
                            inp.type.tensor_type.shape.dim[i].ClearField("dim_param")
                            inp.type.tensor_type.shape.dim[i].dim_value = d
            out = str(Path(self.output_dir) / f"{stem}_pinned.onnx")
            onnx.save(model, out)
            return out
        except Exception as e:
            log.warning("Shape pinning failed: %s", e)
        return None

    def _convert_fp16(self, model_path: str, stem: str) -> Optional[str]:
        try:
            import onnx
            from onnx import numpy_helper
            model = onnx.load(model_path)
            from onnx import TensorProto
            import numpy as np

            for init in model.graph.initializer:
                if init.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(init)
                    arr_fp16 = arr.astype(np.float16)
                    new_init = numpy_helper.from_array(arr_fp16, init.name)
                    init.CopyFrom(new_init)

            out = str(Path(self.output_dir) / f"{stem}_fp16.onnx")
            onnx.save(model, out)
            return out
        except Exception as e:
            log.warning("FP16 conversion failed: %s", e)
        return None

    def _quantize_int8(self, model_path: str, stem: str) -> Optional[str]:
        try:
            from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
            import numpy as np
            import onnx

            class DummyReader(CalibrationDataReader):
                def __init__(self, path):
                    m = onnx.load(path, load_external_data=False)
                    self._feeds = []
                    for inp in m.graph.input:
                        if inp.name not in {i.name for i in m.graph.initializer}:
                            shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
                            self._feeds.append((inp.name, shape))
                    self._idx = 0

                def get_next(self):
                    if self._idx >= 50:
                        return None
                    self._idx += 1
                    return {n: np.random.randn(*s).astype(np.float32) for n, s in self._feeds}

            out = str(Path(self.output_dir) / f"{stem}_int8.onnx")
            quantize_static(model_path, out, DummyReader(model_path),
                            quant_format=QuantFormat.QDQ,
                            weight_type=QuantType.QInt8)
            return out if Path(out).exists() else None
        except Exception as e:
            log.warning("INT8 quantization failed: %s", e)
        return None

    def _ort_optimize(self, model_path: str, stem: str, opt_level: int) -> Optional[str]:
        try:
            import onnxruntime as ort
            out = str(Path(self.output_dir) / f"{stem}_ort_opt.onnx")
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel(opt_level)
            opts.optimized_model_filepath = out
            ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            return out if Path(out).exists() else None
        except Exception as e:
            log.warning("ORT optimization failed: %s", e)
        return None
