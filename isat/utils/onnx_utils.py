"""ONNX model utilities -- shape analysis, op counting, memory estimation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


DTYPE_BYTES = {
    1: 4,   # FLOAT
    2: 1,   # UINT8
    3: 1,   # INT8
    5: 2,   # FLOAT16
    6: 4,   # INT32
    7: 8,   # INT64
    9: 1,   # BOOL
    10: 2,  # FLOAT16
    11: 8,  # DOUBLE
    12: 4,  # UINT32
    16: 2,  # BFLOAT16
}


@dataclass
class GemmShape:
    m: int
    n: int
    k: int
    count: int = 1

    @property
    def flops(self) -> int:
        return 2 * self.m * self.n * self.k * self.count


@dataclass
class ModelProfile:
    path: str = ""
    opset: int = 0
    num_nodes: int = 0
    num_initializers: int = 0
    param_count: int = 0
    param_bytes: int = 0
    op_histogram: dict[str, int] = field(default_factory=dict)
    input_shapes: dict[str, list] = field(default_factory=dict)
    output_shapes: dict[str, list] = field(default_factory=dict)
    dynamic_inputs: list[str] = field(default_factory=list)
    static_inputs: list[str] = field(default_factory=list)
    gemm_shapes: list[GemmShape] = field(default_factory=list)
    estimated_memory_mb: float = 0.0
    has_attention: bool = False
    has_external_data: bool = False
    gemm_fraction: float = 0.0


def analyze_onnx(model_path: str) -> ModelProfile:
    """Deep-analyze an ONNX model without loading weights into memory."""
    import onnx
    from onnx import TensorProto

    path = Path(model_path)
    profile = ModelProfile(path=str(path))

    profile.has_external_data = any(
        path.parent.glob(f"{path.stem}*.data")
    ) or any(path.parent.glob("*.onnx.data"))

    model = onnx.load(str(path), load_external_data=False)

    if model.opset_import:
        profile.opset = model.opset_import[0].version

    graph = model.graph
    profile.num_nodes = len(graph.node)

    init_names = {i.name for i in graph.initializer}
    profile.num_initializers = len(init_names)

    total_params = 0
    total_bytes = 0
    for init in graph.initializer:
        n_elem = 1
        for d in init.dims:
            n_elem *= d
        total_params += n_elem
        bpe = DTYPE_BYTES.get(init.data_type, 4)
        total_bytes += n_elem * bpe

    profile.param_count = total_params
    profile.param_bytes = total_bytes
    profile.estimated_memory_mb = total_bytes / (1024 * 1024)

    op_hist: dict[str, int] = {}
    gemm_ops = 0
    total_ops = 0
    for node in graph.node:
        op_hist[node.op_type] = op_hist.get(node.op_type, 0) + 1
        total_ops += 1
        if node.op_type in {"MatMul", "Gemm", "FusedMatMul"}:
            gemm_ops += 1
        if node.op_type in {"Attention", "MultiHeadAttention"}:
            profile.has_attention = True

    profile.op_histogram = dict(sorted(op_hist.items(), key=lambda x: -x[1]))
    profile.gemm_fraction = gemm_ops / total_ops if total_ops > 0 else 0.0

    def _dims(tensor_type):
        dims = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
        return dims

    for inp in graph.input:
        if inp.name in init_names:
            continue
        shape = _dims(inp.type.tensor_type) if inp.type.HasField("tensor_type") else []
        profile.input_shapes[inp.name] = shape
        if any(isinstance(d, str) for d in shape):
            profile.dynamic_inputs.append(inp.name)
        else:
            profile.static_inputs.append(inp.name)

    for out in graph.output:
        shape = _dims(out.type.tensor_type) if out.type.HasField("tensor_type") else []
        profile.output_shapes[out.name] = shape

    return profile
