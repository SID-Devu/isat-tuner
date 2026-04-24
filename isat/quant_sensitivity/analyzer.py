"""Per-layer quantization sensitivity analysis.

Measures the accuracy impact of quantizing each operator/layer individually,
identifying which layers are sensitive to lower precision and should remain
in FP32 while others can safely be quantized to INT8/FP16.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("isat.quant_sensitivity")


@dataclass
class LayerSensitivity:
    layer_name: str
    op_type: str
    output_mse_fp16: float
    output_mse_int8: float
    param_count: int
    fp16_safe: bool
    int8_safe: bool


@dataclass
class QuantSensitivityReport:
    model_path: str
    layers: list[LayerSensitivity] = field(default_factory=list)
    total_params: int = 0
    fp16_safe_pct: float = 0
    int8_safe_pct: float = 0
    recommended_mixed_precision: dict[str, str] = field(default_factory=dict)
    scan_time_s: float = 0

    def summary(self) -> str:
        lines = [
            f"  Model        : {self.model_path}",
            f"  Total params : {self.total_params:,}",
            f"  FP16 safe    : {self.fp16_safe_pct:.0f}% of layers",
            f"  INT8 safe    : {self.int8_safe_pct:.0f}% of layers",
            f"  Scan time    : {self.scan_time_s:.1f}s",
            f"",
        ]

        sensitive_layers = [l for l in self.layers if not l.fp16_safe or not l.int8_safe]
        if sensitive_layers:
            lines.append(f"  Sensitive layers ({len(sensitive_layers)}):")
            lines.append(f"  {'Layer':<40} {'Op':<20} {'MSE FP16':>10} {'MSE INT8':>10} {'Recommendation':<15}")
            lines.append(f"  {'-'*40} {'-'*20} {'-'*10} {'-'*10} {'-'*15}")
            for l in sensitive_layers[:20]:
                rec = "keep FP32" if not l.fp16_safe else ("FP16 only" if not l.int8_safe else "any")
                lines.append(
                    f"  {l.layer_name[:40]:<40} {l.op_type:<20} "
                    f"{l.output_mse_fp16:>10.6f} {l.output_mse_int8:>10.6f} {rec:<15}"
                )
        else:
            lines.append("  All layers are safe for quantization")

        if self.recommended_mixed_precision:
            lines.append(f"\n  Mixed-precision recipe ({len(self.recommended_mixed_precision)} layers):")
            counts = {}
            for prec in self.recommended_mixed_precision.values():
                counts[prec] = counts.get(prec, 0) + 1
            for prec, cnt in sorted(counts.items()):
                lines.append(f"    {prec}: {cnt} layers")

        return "\n".join(lines)


class QuantSensitivityAnalyzer:
    """Analyze per-layer quantization sensitivity."""

    def __init__(self, model_path: str, mse_threshold_fp16: float = 1e-3,
                 mse_threshold_int8: float = 1e-2):
        self.model_path = model_path
        self.mse_threshold_fp16 = mse_threshold_fp16
        self.mse_threshold_int8 = mse_threshold_int8

    def analyze(self) -> QuantSensitivityReport:
        t0 = time.perf_counter()
        import onnx

        model = onnx.load(str(self.model_path), load_external_data=False)
        graph = model.graph

        init_map = {}
        for init in graph.initializer:
            if init.data_location != 1:
                try:
                    from onnx.numpy_helper import to_array
                    init_map[init.name] = to_array(init)
                except Exception:
                    pass

        layers = []
        total_params = 0
        fp16_safe_count = 0
        int8_safe_count = 0
        mixed_precision = {}

        quant_ops = {"MatMul", "Conv", "Gemm", "ConvTranspose", "MatMulInteger"}

        for node in graph.node:
            if node.op_type not in quant_ops:
                continue

            param_count = 0
            weight_arrays = []
            for inp_name in node.input:
                if inp_name in init_map:
                    arr = init_map[inp_name]
                    param_count += arr.size
                    weight_arrays.append(arr)

            if not weight_arrays:
                continue

            total_params += param_count

            mse_fp16 = 0.0
            mse_int8 = 0.0
            for arr in weight_arrays:
                fp32 = arr.astype(np.float32)
                fp16 = fp32.astype(np.float16).astype(np.float32)
                mse_fp16 += float(np.mean((fp32 - fp16) ** 2))

                if fp32.max() != fp32.min():
                    scale = (fp32.max() - fp32.min()) / 255.0
                    zp = np.round(-fp32.min() / scale).astype(np.int32)
                    quantized = np.clip(np.round(fp32 / scale) + zp, 0, 255).astype(np.uint8)
                    dequantized = (quantized.astype(np.float32) - zp) * scale
                    mse_int8 += float(np.mean((fp32 - dequantized) ** 2))

            fp16_safe = mse_fp16 < self.mse_threshold_fp16
            int8_safe = mse_int8 < self.mse_threshold_int8

            if fp16_safe:
                fp16_safe_count += 1
            if int8_safe:
                int8_safe_count += 1

            if int8_safe:
                mixed_precision[node.name] = "INT8"
            elif fp16_safe:
                mixed_precision[node.name] = "FP16"
            else:
                mixed_precision[node.name] = "FP32"

            layers.append(LayerSensitivity(
                layer_name=node.name or f"{node.op_type}_{len(layers)}",
                op_type=node.op_type,
                output_mse_fp16=mse_fp16,
                output_mse_int8=mse_int8,
                param_count=param_count,
                fp16_safe=fp16_safe,
                int8_safe=int8_safe,
            ))

        n = len(layers) or 1
        return QuantSensitivityReport(
            model_path=self.model_path,
            layers=layers,
            total_params=total_params,
            fp16_safe_pct=fp16_safe_count / n * 100,
            int8_safe_pct=int8_safe_count / n * 100,
            recommended_mixed_precision=mixed_precision,
            scan_time_s=time.perf_counter() - t0,
        )
