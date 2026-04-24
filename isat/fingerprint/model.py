"""Model fingerprinting -- analyze ONNX model structure for tuning decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from isat.utils.onnx_utils import ModelProfile, analyze_onnx


@dataclass
class ModelFingerprint:
    name: str = ""
    path: str = ""
    opset: int = 0
    param_count: int = 0
    param_bytes: int = 0
    estimated_memory_mb: float = 0.0
    num_nodes: int = 0
    num_initializers: int = 0

    input_shapes: dict[str, list] = field(default_factory=dict)
    output_shapes: dict[str, list] = field(default_factory=dict)
    has_dynamic_inputs: bool = False
    has_external_data: bool = False
    has_attention: bool = False

    op_histogram: dict[str, int] = field(default_factory=dict)
    top_ops: list[str] = field(default_factory=list)
    gemm_fraction: float = 0.0

    model_class: str = "unknown"
    fingerprint_hash: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @property
    def size_class(self) -> str:
        mb = self.estimated_memory_mb
        if mb < 100:
            return "small"
        if mb < 2048:
            return "medium"
        if mb < 10240:
            return "large"
        return "xlarge"


def fingerprint_model(model_path: str) -> ModelFingerprint:
    profile = analyze_onnx(model_path)
    fp = ModelFingerprint()

    fp.path = model_path
    fp.name = Path(model_path).stem
    fp.opset = profile.opset
    fp.param_count = profile.param_count
    fp.param_bytes = profile.param_bytes
    fp.estimated_memory_mb = profile.estimated_memory_mb
    fp.num_nodes = profile.num_nodes
    fp.num_initializers = profile.num_initializers

    fp.input_shapes = profile.input_shapes
    fp.output_shapes = profile.output_shapes
    fp.has_dynamic_inputs = len(profile.dynamic_inputs) > 0
    fp.has_external_data = profile.has_external_data
    fp.has_attention = profile.has_attention

    fp.op_histogram = profile.op_histogram
    fp.top_ops = list(profile.op_histogram.keys())[:10]
    fp.gemm_fraction = profile.gemm_fraction

    fp.model_class = _classify_model(profile)

    identity = f"{fp.name}:{fp.param_count}:{fp.num_nodes}:{fp.opset}"
    fp.fingerprint_hash = hashlib.sha256(identity.encode()).hexdigest()[:16]

    return fp


def _classify_model(profile: ModelProfile) -> str:
    """Heuristic classification based on op distribution."""
    ops = profile.op_histogram

    if profile.has_attention or ops.get("MultiHeadAttention", 0) > 0:
        if profile.param_count > 1_000_000_000:
            return "llm"
        return "transformer"

    conv_count = ops.get("Conv", 0)
    matmul_count = ops.get("MatMul", 0) + ops.get("Gemm", 0)

    if conv_count > matmul_count and conv_count > 5:
        return "cnn"

    if matmul_count > conv_count:
        return "transformer"

    return "mixed"
