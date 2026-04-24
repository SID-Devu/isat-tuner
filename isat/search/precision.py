"""Precision search dimension.

Explores:
  - FP32 baseline
  - FP16 (MIGraphX built-in conversion)
  - INT8 static quantization via ORT quantization tools
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint


@dataclass
class PrecisionConfig:
    precision: str = "fp32"
    quantize_method: str = "none"
    calibration_samples: int = 100
    model_path_override: Optional[str] = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.precision}_{self.quantize_method}"


class PrecisionSearchDimension:
    """Generate candidate precision configs."""

    def __init__(self, hw: HardwareFingerprint, model: ModelFingerprint):
        self.hw = hw
        self.model = model

    def candidates(self) -> list[PrecisionConfig]:
        configs: list[PrecisionConfig] = []

        configs.append(PrecisionConfig(
            precision="fp32",
            quantize_method="none",
            label="fp32_native",
        ))

        configs.append(PrecisionConfig(
            precision="fp16",
            quantize_method="migraphx_fp16",
            env_overrides={"MIGRAPHX_FP16_ENABLE": "1"},
            label="fp16_migraphx",
        ))

        configs.append(PrecisionConfig(
            precision="int8",
            quantize_method="ort_static_qdq",
            label="int8_qdq",
        ))

        return configs

    @staticmethod
    def prepare_int8_model(original_path: str, output_dir: str, calibration_samples: int = 100) -> Optional[str]:
        """Quantize a model to INT8 using ORT quantization tools.

        Returns the path to the quantized model, or None on failure.
        """
        try:
            from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
        except ImportError:
            return None

        class DummyCalibrationReader(CalibrationDataReader):
            def __init__(self, model_path: str, num_samples: int):
                import onnx
                model = onnx.load(model_path, load_external_data=False)
                self.feeds = []
                import numpy as np
                for inp in model.graph.input:
                    if inp.name in {i.name for i in model.graph.initializer}:
                        continue
                    shape = []
                    for d in inp.type.tensor_type.shape.dim:
                        shape.append(d.dim_value if d.dim_value > 0 else 1)
                    self.feeds.append((inp.name, shape))
                self._idx = 0
                self._max = num_samples

            def get_next(self):
                import numpy as np
                if self._idx >= self._max:
                    return None
                self._idx += 1
                return {name: np.random.randn(*shape).astype(np.float32) for name, shape in self.feeds}

        out_path = Path(output_dir) / f"{Path(original_path).stem}_int8.onnx"
        try:
            reader = DummyCalibrationReader(original_path, calibration_samples)
            quantize_static(
                original_path,
                str(out_path),
                reader,
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8,
                activation_type=QuantType.QInt8,
            )
            return str(out_path) if out_path.exists() else None
        except Exception:
            return None
