"""Knowledge distillation helper -- generate student model configurations.

Given a teacher ONNX model, analyzes its architecture and generates:
  - Recommended student architectures (smaller variants)
  - Layer mapping between teacher and student
  - Distillation training config (temperature, alpha, loss weights)
  - Estimated speedup and accuracy trade-offs

Does NOT train the student -- generates the configuration for training frameworks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("isat.distillation")


@dataclass
class StudentConfig:
    name: str
    description: str
    estimated_params_m: float
    estimated_speedup: float
    estimated_accuracy_retention: float
    layer_mapping: dict[str, str] = field(default_factory=dict)
    architecture_changes: list[str] = field(default_factory=list)


@dataclass
class DistillationPlan:
    teacher_path: str
    teacher_params_m: float
    teacher_ops: int
    teacher_layers: int
    students: list[StudentConfig] = field(default_factory=list)
    temperature: float = 4.0
    alpha: float = 0.7
    training_tips: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Teacher model   : {self.teacher_path}",
            f"  Teacher params  : {self.teacher_params_m:.1f}M",
            f"  Teacher ops     : {self.teacher_ops}",
            f"  Teacher layers  : {self.teacher_layers}",
            "",
            "  Distillation settings:",
            f"    Temperature   : {self.temperature}",
            f"    Alpha (soft)  : {self.alpha}",
            f"    Beta (hard)   : {1 - self.alpha}",
            "",
            f"  Recommended students ({len(self.students)}):",
            f"  {'Name':<20} {'Params':>10} {'Speedup':>10} {'Accuracy':>10}",
            f"  {'-' * 20} {'-' * 10} {'-' * 10} {'-' * 10}",
        ]
        for s in self.students:
            lines.append(
                f"  {s.name:<20} {s.estimated_params_m:>9.1f}M "
                f"{s.estimated_speedup:>9.1f}x {s.estimated_accuracy_retention:>9.0f}%"
            )
            for change in s.architecture_changes[:3]:
                lines.append(f"    - {change}")
        if self.training_tips:
            lines.append("\n  Training tips:")
            for tip in self.training_tips:
                lines.append(f"    - {tip}")
        return "\n".join(lines)


def _build_layer_mapping(teacher_init_names: list[str], scale_key: str) -> dict[str, str]:
    """Heuristic teacher→student tensor name mapping for distillation tooling."""
    mapping: dict[str, str] = {}
    for name in teacher_init_names:
        mapping[name] = f"{scale_key}/{name}" if "/" in name or "." in name else f"student_{scale_key}_{name}"
    return mapping


class DistillationHelper:
    """Analyze teacher model and generate student configs."""

    def __init__(self, teacher_path: str) -> None:
        import onnx

        path = Path(teacher_path)
        if not path.is_file():
            raise FileNotFoundError(f"Teacher ONNX model not found: {teacher_path}")

        self.teacher_path = str(path.resolve())
        try:
            self.model = onnx.load(self.teacher_path, load_external_data=False)
        except Exception as exc:
            log.exception("Failed to load teacher model")
            raise RuntimeError(f"Could not parse ONNX model {self.teacher_path}") from exc

    def plan(self) -> DistillationPlan:
        from onnx import numpy_helper

        graph = self.model.graph
        total_params = 0
        init_names: list[str] = []
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            total_params += int(arr.size)
            init_names.append(init.name)

        params_m = total_params / 1e6
        num_ops = len(graph.node)

        op_counts: dict[str, int] = {}
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

        num_attention = op_counts.get("Attention", 0) + op_counts.get("MultiHeadAttention", 0)
        num_matmul = op_counts.get("MatMul", 0)
        num_conv = op_counts.get("Conv", 0)
        num_layers = max(num_attention, num_matmul // 2, num_conv, 1)

        students: list[StudentConfig] = []

        students.append(
            StudentConfig(
                name="tiny",
                description="25% of teacher -- fastest, lower accuracy",
                estimated_params_m=params_m * 0.25,
                estimated_speedup=4.0,
                estimated_accuracy_retention=85,
                layer_mapping=_build_layer_mapping(init_names, "tiny"),
                architecture_changes=[
                    f"Reduce layers from {num_layers} to {max(num_layers // 4, 1)}",
                    "Halve hidden dimensions",
                    "Reduce attention heads by 75%",
                ],
            )
        )

        students.append(
            StudentConfig(
                name="small",
                description="50% of teacher -- balanced speed/accuracy",
                estimated_params_m=params_m * 0.5,
                estimated_speedup=2.2,
                estimated_accuracy_retention=92,
                layer_mapping=_build_layer_mapping(init_names, "small"),
                architecture_changes=[
                    f"Reduce layers from {num_layers} to {max(num_layers // 2, 1)}",
                    "Reduce hidden dimensions by 30%",
                    "Reduce attention heads by 50%",
                ],
            )
        )

        students.append(
            StudentConfig(
                name="medium",
                description="75% of teacher -- minimal accuracy loss",
                estimated_params_m=params_m * 0.75,
                estimated_speedup=1.5,
                estimated_accuracy_retention=97,
                layer_mapping=_build_layer_mapping(init_names, "medium"),
                architecture_changes=[
                    f"Reduce layers from {num_layers} to {max(int(num_layers * 0.75), 1)}",
                    "Keep hidden dimensions",
                    "Reduce attention heads by 25%",
                ],
            )
        )

        tips = [
            "Start with temperature=4.0 and alpha=0.7 (70% soft labels, 30% hard labels)",
            "Train for 2-3x the normal epochs when distilling",
            "Use the same tokenizer/preprocessor as the teacher",
            "Validate on teacher's outputs first to ensure pipeline correctness",
        ]
        if num_conv > 5:
            tips.append("For CNNs: add feature map matching loss at intermediate layers")
        if num_attention > 0:
            tips.append("For transformers: match attention distributions (attention transfer)")
            tips.append("Consider TinyBERT-style 2-stage distillation (general + task-specific)")
        if params_m > 100:
            tips.append(
                "Large teacher (>100M params): consider progressive distillation "
                "(chain: teacher -> medium -> small)"
            )

        return DistillationPlan(
            teacher_path=self.teacher_path,
            teacher_params_m=params_m,
            teacher_ops=num_ops,
            teacher_layers=num_layers,
            students=students,
            temperature=4.0,
            alpha=0.7,
            training_tips=tips,
        )
