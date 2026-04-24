"""Graph transformation search dimension.

Explores:
  - Raw model (no transforms)
  - onnxsim simplification
  - Shape pinning (freeze dynamic dims)
  - ORT graph optimizations (levels 0-3)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint


@dataclass
class GraphConfig:
    onnxsim: bool = False
    pin_shapes: Optional[dict[str, list[int]]] = None
    ort_opt_level: int = 99
    model_path_override: Optional[str] = None
    env_overrides: dict[str, str] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            parts = []
            if self.onnxsim:
                parts.append("sim")
            if self.pin_shapes:
                parts.append("pinned")
            parts.append(f"opt{self.ort_opt_level}")
            self.label = "_".join(parts) or "raw_opt99"


class GraphSearchDimension:
    """Generate candidate graph transform configs."""

    def __init__(self, hw: HardwareFingerprint, model: ModelFingerprint):
        self.hw = hw
        self.model = model

    def candidates(self) -> list[GraphConfig]:
        configs: list[GraphConfig] = []

        configs.append(GraphConfig(
            ort_opt_level=99,
            label="raw_opt99",
        ))

        configs.append(GraphConfig(
            onnxsim=True,
            ort_opt_level=99,
            label="sim_opt99",
        ))

        if self.model.has_dynamic_inputs:
            pin_shapes = {}
            for name, shape in self.model.input_shapes.items():
                pinned = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
                pin_shapes[name] = pinned

            configs.append(GraphConfig(
                pin_shapes=pin_shapes,
                ort_opt_level=99,
                label="pinned_opt99",
            ))

        configs.append(GraphConfig(
            ort_opt_level=1,
            label="raw_opt1",
        ))

        return configs

    @staticmethod
    def simplify_model(model_path: str, output_dir: str) -> Optional[str]:
        """Run onnxsim on a model. Returns the output path, or None on failure."""
        out_path = Path(output_dir) / f"{Path(model_path).stem}_sim.onnx"
        try:
            import onnxsim
            import onnx
            model = onnx.load(model_path)
            model_sim, check = onnxsim.simplify(model)
            if check:
                onnx.save(model_sim, str(out_path))
                return str(out_path)
        except Exception:
            pass

        try:
            r = subprocess.run(
                ["python3", "-m", "onnxsim", model_path, str(out_path)],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0 and out_path.exists():
                return str(out_path)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None
