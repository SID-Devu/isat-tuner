"""Operator fusion analyzer -- show which ops are fused vs separate.

Compares raw ONNX graph against optimized graph to identify:
  - Which operator patterns were fused (e.g., Conv+BN+Relu)
  - Which ops remain unfused (potential optimization targets)
  - Estimated computation saved by fusion
  - Provider-specific fusion patterns

Explains why a model is slow by showing missed fusion opportunities.
"""

from __future__ import annotations

import logging
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("isat.fusion")

KNOWN_FUSION_PATTERNS: dict[str, tuple[list[str], str]] = {
    "ConvBnRelu": (
        ["Conv", "BatchNormalization", "Relu"],
        "Conv + BatchNorm + ReLU fused into single kernel",
    ),
    "ConvBn": (
        ["Conv", "BatchNormalization"],
        "Conv + BatchNorm fused (BN folded into Conv weights)",
    ),
    "ConvRelu": (["Conv", "Relu"], "Conv + ReLU fused into single kernel"),
    "MatMulAdd": (["MatMul", "Add"], "MatMul + Add fused into Gemm"),
    "GeluApprox": (["Mul", "Tanh", "Add", "Mul"], "Approximate GELU pattern"),
    "LayerNorm": (
        [
            "ReduceMean",
            "Sub",
            "Mul",
            "ReduceMean",
            "Add",
            "Sqrt",
            "Div",
            "Mul",
            "Add",
        ],
        "Layer normalization pattern",
    ),
    "Attention": (
        ["MatMul", "Add", "Reshape", "Transpose", "MatMul", "Div", "Softmax", "MatMul"],
        "Multi-head attention pattern",
    ),
    "BiasGelu": (["Add", "Gelu"], "Bias + GELU activation fused"),
    "SkipLayerNorm": (["Add", "ReduceMean", "Sub"], "Skip connection + LayerNorm fused"),
}


def _ort_optimization_level(opt_level: int) -> Any:
    import onnxruntime as ort

    if opt_level <= 0:
        return ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if opt_level == 1:
        return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if opt_level == 2:
        return ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.GraphOptimizationLevel.ORT_ENABLE_ALL


@dataclass
class FusionPattern:
    name: str
    ops: list[str]
    description: str
    count_original: int
    count_optimized: int
    fused: bool


@dataclass
class UnfusedOp:
    op_type: str
    count: int
    reason: str


@dataclass
class FusionReport:
    model_path: str
    original_ops: int
    optimized_ops: int
    ops_eliminated: int
    fusion_ratio: float
    patterns: list[FusionPattern] = field(default_factory=list)
    unfused: list[UnfusedOp] = field(default_factory=list)
    original_op_counts: dict[str, int] = field(default_factory=dict)
    optimized_op_counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Model          : {self.model_path}",
            f"  Original ops   : {self.original_ops}",
            f"  Optimized ops  : {self.optimized_ops}",
            f"  Ops eliminated : {self.ops_eliminated} ({self.fusion_ratio:.1%} reduction)",
            "",
        ]

        fused = [p for p in self.patterns if p.fused]
        if fused:
            lines.append(f"  Fused patterns ({len(fused)}):")
            for p in fused:
                lines.append(
                    f"    {p.name}: {p.description} ({p.count_original} -> {p.count_optimized})"
                )

        missed = [p for p in self.patterns if not p.fused and p.count_original > 0]
        if missed:
            lines.append(f"\n  Missed fusion opportunities ({len(missed)}):")
            for p in missed:
                lines.append(
                    f"    {p.name}: {p.description} (still {p.count_original} instances)"
                )

        if self.unfused:
            lines.append("\n  Remaining unfused ops:")
            lines.append(f"  {'Op Type':<25} {'Count':>8} {'Note':<40}")
            lines.append(f"  {'-' * 25} {'-' * 8} {'-' * 40}")
            for u in self.unfused[:15]:
                lines.append(f"  {u.op_type:<25} {u.count:>8} {u.reason:<40}")
        return "\n".join(lines)


class FusionAnalyzer:
    """Analyze operator fusion in ONNX models."""

    def __init__(self, model_path: str) -> None:
        import onnx

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = str(path.resolve())
        self.model = onnx.load(self.model_path, load_external_data=False)

    def analyze(self, opt_level: int = 99) -> FusionReport:
        import onnx as _onnx
        import onnxruntime as ort

        graph = self.model.graph
        original_ops = [n.op_type for n in graph.node]
        original_counts = Counter(original_ops)

        optimized_ops_list = list(original_ops)
        optimized_counts = Counter(original_counts)

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            optimized_path = tmp.name

        try:
            so = ort.SessionOptions()
            so.graph_optimization_level = _ort_optimization_level(opt_level)
            so.optimized_model_filepath = optimized_path

            try:
                ort.InferenceSession(
                    self.model_path,
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as exc:
                log.warning("ORT optimization failed: %s", exc)
            else:
                try:
                    opt_model = _onnx.load(optimized_path, load_external_data=False)
                    optimized_ops_list = [n.op_type for n in opt_model.graph.node]
                    optimized_counts = Counter(optimized_ops_list)
                except Exception as exc:
                    log.warning("Could not read optimized model: %s", exc)
                    optimized_ops_list = list(original_ops)
                    optimized_counts = Counter(original_counts)
        finally:
            p = Path(optimized_path)
            if p.exists():
                try:
                    p.unlink()
                except OSError as exc:
                    log.debug("Could not remove temp ONNX %s: %s", optimized_path, exc)

        patterns: list[FusionPattern] = []
        for name, (ops_seq, desc) in KNOWN_FUSION_PATTERNS.items():
            original_count = self._count_pattern(original_ops, ops_seq)
            optimized_count = self._count_pattern(optimized_ops_list, ops_seq)
            fused = optimized_count < original_count
            patterns.append(
                FusionPattern(
                    name=name,
                    ops=list(ops_seq),
                    description=desc,
                    count_original=original_count,
                    count_optimized=optimized_count,
                    fused=fused,
                )
            )

        unfused: list[UnfusedOp] = []
        expensive_ops = {
            "Conv",
            "MatMul",
            "Gemm",
            "LSTM",
            "GRU",
            "Attention",
            "Softmax",
            "LayerNormalization",
            "BatchNormalization",
        }
        for op, count in optimized_counts.most_common():
            if op in expensive_ops:
                reason = "Compute-heavy -- check if provider supports fused variant"
            elif op in {"Cast", "Reshape", "Transpose", "Squeeze", "Unsqueeze"}:
                reason = "Shape manipulation -- may be eliminated by provider"
            elif op in {"Gather", "Slice", "Concat"}:
                reason = "Memory-bound op -- check data layout"
            else:
                continue
            unfused.append(UnfusedOp(op_type=op, count=count, reason=reason))

        ops_eliminated = len(original_ops) - len(optimized_ops_list)
        fusion_ratio = ops_eliminated / max(len(original_ops), 1)

        return FusionReport(
            model_path=self.model_path,
            original_ops=len(original_ops),
            optimized_ops=len(optimized_ops_list),
            ops_eliminated=ops_eliminated,
            fusion_ratio=fusion_ratio,
            patterns=patterns,
            unfused=unfused,
            original_op_counts=dict(original_counts),
            optimized_op_counts=dict(optimized_counts),
        )

    @staticmethod
    def _count_pattern(ops: list[str], pattern: list[str]) -> int:
        count = 0
        plen = len(pattern)
        if plen == 0 or len(ops) < plen:
            return 0
        for i in range(len(ops) - plen + 1):
            if ops[i : i + plen] == pattern:
                count += 1
        return count
