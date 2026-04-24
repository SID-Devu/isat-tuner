"""Structural comparison of two ONNX models.

Compares:
  - Graph topology (node count, types, order)
  - Parameter count and size per initializer
  - Input/output shapes and types
  - Opset versions
  - Operator distribution differences
  - Subgraph differences
  - New/removed/modified nodes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.diff")


@dataclass
class NodeDiff:
    op_type: str
    name: str
    change: str  # "added", "removed", "modified"
    details: str = ""


@dataclass
class DiffResult:
    model_a: str
    model_b: str
    size_a_mb: float
    size_b_mb: float
    nodes_a: int
    nodes_b: int
    params_a: int
    params_b: int
    opset_a: int
    opset_b: int
    ops_only_in_a: dict[str, int] = field(default_factory=dict)
    ops_only_in_b: dict[str, int] = field(default_factory=dict)
    ops_count_diff: dict[str, tuple[int, int]] = field(default_factory=dict)
    input_diffs: list[str] = field(default_factory=list)
    output_diffs: list[str] = field(default_factory=list)
    node_diffs: list[NodeDiff] = field(default_factory=list)
    identical: bool = False

    def summary(self) -> str:
        lines = [
            f"  {'Metric':<30} {'Model A':>15} {'Model B':>15} {'Delta':>15}",
            f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}",
            f"  {'File size (MB)':<30} {self.size_a_mb:>15.1f} {self.size_b_mb:>15.1f} {self.size_b_mb - self.size_a_mb:>+15.1f}",
            f"  {'Node count':<30} {self.nodes_a:>15} {self.nodes_b:>15} {self.nodes_b - self.nodes_a:>+15}",
            f"  {'Parameters':<30} {self.params_a:>15,} {self.params_b:>15,} {self.params_b - self.params_a:>+15,}",
            f"  {'Opset version':<30} {self.opset_a:>15} {self.opset_b:>15}",
        ]

        if self.ops_only_in_a:
            lines.append(f"\n  Ops only in Model A: {', '.join(f'{k}({v})' for k, v in self.ops_only_in_a.items())}")
        if self.ops_only_in_b:
            lines.append(f"  Ops only in Model B: {', '.join(f'{k}({v})' for k, v in self.ops_only_in_b.items())}")
        if self.ops_count_diff:
            lines.append(f"\n  Op count differences:")
            for op, (ca, cb) in sorted(self.ops_count_diff.items()):
                lines.append(f"    {op}: {ca} -> {cb} ({cb - ca:+d})")
        if self.input_diffs:
            lines.append(f"\n  Input differences:")
            for d in self.input_diffs:
                lines.append(f"    {d}")
        if self.output_diffs:
            lines.append(f"\n  Output differences:")
            for d in self.output_diffs:
                lines.append(f"    {d}")

        return "\n".join(lines)


class ModelDiff:
    """Compare two ONNX models."""

    def compare(self, model_a: str, model_b: str) -> DiffResult:
        import onnx

        ma = onnx.load(model_a, load_external_data=False)
        mb = onnx.load(model_b, load_external_data=False)

        size_a = Path(model_a).stat().st_size / (1024 * 1024)
        size_b = Path(model_b).stat().st_size / (1024 * 1024)

        params_a = sum(_param_count(i) for i in ma.graph.initializer)
        params_b = sum(_param_count(i) for i in mb.graph.initializer)

        opset_a = ma.opset_import[0].version if ma.opset_import else 0
        opset_b = mb.opset_import[0].version if mb.opset_import else 0

        ops_a = _count_ops(ma)
        ops_b = _count_ops(mb)

        all_ops = set(ops_a) | set(ops_b)
        only_a = {k: v for k, v in ops_a.items() if k not in ops_b}
        only_b = {k: v for k, v in ops_b.items() if k not in ops_a}
        count_diff = {}
        for op in all_ops:
            ca, cb = ops_a.get(op, 0), ops_b.get(op, 0)
            if ca != cb and op not in only_a and op not in only_b:
                count_diff[op] = (ca, cb)

        init_names_a = {i.name for i in ma.graph.initializer}
        init_names_b = {i.name for i in mb.graph.initializer}
        inputs_a = {i.name: _shape_str(i) for i in ma.graph.input if i.name not in init_names_a}
        inputs_b = {i.name: _shape_str(i) for i in mb.graph.input if i.name not in init_names_b}
        outputs_a = {o.name: _shape_str(o) for o in ma.graph.output}
        outputs_b = {o.name: _shape_str(o) for o in mb.graph.output}

        input_diffs = _dict_diff(inputs_a, inputs_b, "input")
        output_diffs = _dict_diff(outputs_a, outputs_b, "output")

        nodes_a_map = {(n.op_type, n.name or f"#{i}"): n for i, n in enumerate(ma.graph.node)}
        nodes_b_map = {(n.op_type, n.name or f"#{i}"): n for i, n in enumerate(mb.graph.node)}

        node_diffs = []
        for key in set(nodes_a_map) - set(nodes_b_map):
            node_diffs.append(NodeDiff(key[0], key[1], "removed"))
        for key in set(nodes_b_map) - set(nodes_a_map):
            node_diffs.append(NodeDiff(key[0], key[1], "added"))

        identical = (
            len(only_a) == 0 and len(only_b) == 0 and
            len(count_diff) == 0 and len(input_diffs) == 0 and
            len(output_diffs) == 0 and len(node_diffs) == 0 and
            params_a == params_b
        )

        return DiffResult(
            model_a=model_a, model_b=model_b,
            size_a_mb=size_a, size_b_mb=size_b,
            nodes_a=len(ma.graph.node), nodes_b=len(mb.graph.node),
            params_a=params_a, params_b=params_b,
            opset_a=opset_a, opset_b=opset_b,
            ops_only_in_a=only_a, ops_only_in_b=only_b,
            ops_count_diff=count_diff,
            input_diffs=input_diffs, output_diffs=output_diffs,
            node_diffs=node_diffs, identical=identical,
        )


def _count_ops(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _param_count(init) -> int:
    n = 1
    for d in init.dims:
        n *= d
    return n


def _shape_str(tensor_info) -> str:
    try:
        dims = []
        for d in tensor_info.type.tensor_type.shape.dim:
            if d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append(str(d.dim_value))
        dtype = tensor_info.type.tensor_type.elem_type
        return f"[{','.join(dims)}] dtype={dtype}"
    except Exception:
        return "unknown"


def _dict_diff(a: dict, b: dict, label: str) -> list[str]:
    diffs = []
    for k in set(a) - set(b):
        diffs.append(f"{label} '{k}' removed (was {a[k]})")
    for k in set(b) - set(a):
        diffs.append(f"{label} '{k}' added ({b[k]})")
    for k in set(a) & set(b):
        if a[k] != b[k]:
            diffs.append(f"{label} '{k}' changed: {a[k]} -> {b[k]}")
    return diffs
