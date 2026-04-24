"""ONNX graph surgery -- programmatic model modification.

Operations:
  - Remove nodes (Identity, Dropout in inference, etc.)
  - Replace operators (e.g., BatchNorm -> fused Conv)
  - Extract subgraph (split encoder/decoder)
  - Insert profiling nodes
  - Rename inputs/outputs
  - Change opset version
  - Remove unused initializers

Production use case: prepare models for deployment without retraining.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.surgery")


@dataclass
class SurgeryOp:
    operation: str
    target: str
    detail: str = ""


@dataclass
class SurgeryResult:
    model_path: str
    output_path: str
    operations: list[SurgeryOp] = field(default_factory=list)
    nodes_before: int = 0
    nodes_after: int = 0
    initializers_before: int = 0
    initializers_after: int = 0
    size_before_mb: float = 0
    size_after_mb: float = 0

    def summary(self) -> str:
        lines = [
            f"  Input       : {self.model_path}",
            f"  Output      : {self.output_path}",
            f"  Nodes       : {self.nodes_before} -> {self.nodes_after} ({self.nodes_after - self.nodes_before:+d})",
            f"  Initializers: {self.initializers_before} -> {self.initializers_after}",
            f"  Size        : {self.size_before_mb:.1f} MB -> {self.size_after_mb:.1f} MB",
            f"",
            f"  Operations performed ({len(self.operations)}):",
        ]
        for i, op in enumerate(self.operations, 1):
            lines.append(f"    {i}. [{op.operation}] {op.target}")
            if op.detail:
                lines.append(f"       {op.detail}")
        return "\n".join(lines)


class GraphSurgeon:
    """Perform surgery on ONNX model graphs."""

    def __init__(self, model_path: str):
        import onnx
        self.model_path = model_path
        self.model = onnx.load(str(model_path), load_external_data=False)
        self._ops: list[SurgeryOp] = []
        self._nodes_before = len(self.model.graph.node)
        self._inits_before = len(self.model.graph.initializer)

    def remove_op_type(self, op_type: str) -> int:
        graph = self.model.graph
        to_remove = [n for n in graph.node if n.op_type == op_type]
        count = 0
        for node in to_remove:
            if len(node.input) >= 1 and len(node.output) >= 1:
                inp = node.input[0]
                out = node.output[0]
                for n2 in graph.node:
                    for i, n2_inp in enumerate(n2.input):
                        if n2_inp == out:
                            n2.input[i] = inp
                for o in graph.output:
                    if o.name == out:
                        o.name = inp
                graph.node.remove(node)
                count += 1
        self._ops.append(SurgeryOp("remove_op_type", op_type, f"Removed {count} nodes"))
        return count

    def remove_node_by_name(self, name: str) -> bool:
        graph = self.model.graph
        for node in graph.node:
            if node.name == name:
                graph.node.remove(node)
                self._ops.append(SurgeryOp("remove_node", name))
                return True
        return False

    def rename_input(self, old_name: str, new_name: str):
        graph = self.model.graph
        for inp in graph.input:
            if inp.name == old_name:
                inp.name = new_name
        for node in graph.node:
            for i, n in enumerate(node.input):
                if n == old_name:
                    node.input[i] = new_name
        self._ops.append(SurgeryOp("rename_input", f"{old_name} -> {new_name}"))

    def rename_output(self, old_name: str, new_name: str):
        graph = self.model.graph
        for out in graph.output:
            if out.name == old_name:
                out.name = new_name
        for node in graph.node:
            for i, n in enumerate(node.output):
                if n == old_name:
                    node.output[i] = new_name
        self._ops.append(SurgeryOp("rename_output", f"{old_name} -> {new_name}"))

    def remove_unused_initializers(self) -> int:
        graph = self.model.graph
        used_inputs = set()
        for node in graph.node:
            used_inputs.update(node.input)
        to_remove = [i for i in graph.initializer if i.name not in used_inputs]
        for init in to_remove:
            graph.initializer.remove(init)
        self._ops.append(SurgeryOp(
            "remove_unused_initializers", f"{len(to_remove)} removed"))
        return len(to_remove)

    def change_opset(self, new_opset: int):
        for imp in self.model.opset_import:
            if imp.domain == "" or imp.domain == "ai.onnx":
                old = imp.version
                imp.version = new_opset
                self._ops.append(SurgeryOp("change_opset", f"{old} -> {new_opset}"))
                break

    def extract_subgraph(self, input_names: list[str], output_names: list[str]) -> int:
        graph = self.model.graph
        needed_outputs = set(output_names)
        needed_nodes = []

        for node in reversed(list(graph.node)):
            if any(o in needed_outputs for o in node.output):
                needed_nodes.append(node)
                needed_outputs.update(node.input)

        original_count = len(graph.node)
        kept = set(id(n) for n in needed_nodes)
        to_remove = [n for n in graph.node if id(n) not in kept]
        for n in to_remove:
            graph.node.remove(n)

        removed = original_count - len(graph.node)
        self._ops.append(SurgeryOp(
            "extract_subgraph",
            f"inputs={input_names}, outputs={output_names}",
            f"Removed {removed} unreachable nodes",
        ))
        return removed

    def get_stats(self) -> dict:
        return {
            "nodes": len(self.model.graph.node),
            "initializers": len(self.model.graph.initializer),
            "inputs": [i.name for i in self.model.graph.input],
            "outputs": [o.name for o in self.model.graph.output],
            "ops": list(set(n.op_type for n in self.model.graph.node)),
        }

    def save(self, output_path: str) -> SurgeryResult:
        import onnx
        onnx.save(self.model, output_path)

        size_before = Path(self.model_path).stat().st_size / (1024 * 1024)
        size_after = Path(output_path).stat().st_size / (1024 * 1024)

        return SurgeryResult(
            model_path=self.model_path,
            output_path=output_path,
            operations=self._ops,
            nodes_before=self._nodes_before,
            nodes_after=len(self.model.graph.node),
            initializers_before=self._inits_before,
            initializers_after=len(self.model.graph.initializer),
            size_before_mb=size_before,
            size_after_mb=size_after,
        )
