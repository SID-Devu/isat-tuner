"""ONNX computation graph visualizer.

Generates:
  - DOT format (Graphviz compatible)
  - ASCII text representation
  - Operator summary charts

Highlights bottleneck ops and shows data flow.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("isat.visualizer")


class GraphVisualizer:
    """Visualize ONNX model computation graphs."""

    def __init__(self, model_path: str):
        self.model_path = model_path

    def to_dot(self, output_path: str = "", highlight_expensive: bool = True) -> str:
        """Generate Graphviz DOT representation."""
        import onnx
        model = onnx.load(self.model_path, load_external_data=False)

        expensive = {"Conv", "MatMul", "Gemm", "Attention", "MultiHeadAttention"}
        medium = {"LayerNormalization", "BatchNormalization", "Softmax"}

        lines = [
            'digraph ONNX {',
            '  rankdir=TB;',
            '  node [shape=box, style=filled, fontname="Helvetica"];',
            '',
        ]

        init_names = {i.name for i in model.graph.initializer}
        for inp in model.graph.input:
            if inp.name not in init_names:
                lines.append(f'  "{inp.name}" [shape=ellipse, fillcolor="#4ade80", label="{inp.name}"];')

        for out in model.graph.output:
            lines.append(f'  "{out.name}" [shape=ellipse, fillcolor="#38bdf8", label="{out.name}"];')

        for i, node in enumerate(model.graph.node):
            node_id = f"node_{i}"
            label = node.op_type
            if node.name:
                label = f"{node.op_type}\\n{node.name}"

            if highlight_expensive and node.op_type in expensive:
                color = "#ef4444"
            elif highlight_expensive and node.op_type in medium:
                color = "#f59e0b"
            else:
                color = "#e2e8f0"

            lines.append(f'  "{node_id}" [fillcolor="{color}", label="{label}"];')

            for inp_name in node.input:
                if inp_name and inp_name not in init_names:
                    lines.append(f'  "{inp_name}" -> "{node_id}";')

            for out_name in node.output:
                if out_name:
                    lines.append(f'  "{node_id}" -> "{out_name}";')

        lines.append('}')
        dot_str = "\n".join(lines)

        if output_path:
            Path(output_path).write_text(dot_str)
            log.info("DOT graph saved to %s", output_path)

        return dot_str

    def to_ascii(self, max_depth: int = 50) -> str:
        """Generate ASCII text representation of the graph."""
        import onnx
        model = onnx.load(self.model_path, load_external_data=False)

        init_names = {i.name for i in model.graph.initializer}

        lines = [f"ONNX Graph: {model.graph.name or 'unnamed'}"]
        lines.append(f"Nodes: {len(model.graph.node)}")
        lines.append("")

        lines.append("INPUTS:")
        for inp in model.graph.input:
            if inp.name not in init_names:
                shape = self._shape_str(inp)
                lines.append(f"  [{inp.name}] {shape}")

        lines.append("\nGRAPH:")
        for i, node in enumerate(model.graph.node):
            if i >= max_depth:
                lines.append(f"  ... ({len(model.graph.node) - max_depth} more nodes)")
                break

            inputs = [n for n in node.input if n and n not in init_names]
            outputs = list(node.output)
            inp_str = ", ".join(inputs[:3])
            out_str = ", ".join(outputs[:2])
            name = f" ({node.name})" if node.name else ""

            lines.append(f"  [{i:3d}] {node.op_type:<25}{name}")
            if inputs:
                lines.append(f"        in:  {inp_str}")
            lines.append(f"        out: {out_str}")

        lines.append("\nOUTPUTS:")
        for out in model.graph.output:
            shape = self._shape_str(out)
            lines.append(f"  [{out.name}] {shape}")

        return "\n".join(lines)

    def op_histogram(self) -> str:
        """Generate ASCII histogram of operator types."""
        import onnx
        model = onnx.load(self.model_path, load_external_data=False)

        counts: dict[str, int] = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1

        sorted_ops = sorted(counts.items(), key=lambda x: -x[1])
        total = sum(counts.values())
        max_count = max(counts.values()) if counts else 1
        bar_width = 40

        lines = [f"Operator Distribution ({total} total nodes)", ""]
        for op, count in sorted_ops:
            bar_len = int(count / max_count * bar_width)
            bar = "█" * bar_len
            pct = count / total * 100
            lines.append(f"  {op:<25} {bar} {count:>4} ({pct:.0f}%)")

        return "\n".join(lines)

    def _shape_str(self, tensor_info) -> str:
        try:
            dims = []
            for d in tensor_info.type.tensor_type.shape.dim:
                dims.append(d.dim_param if d.dim_param else str(d.dim_value))
            return f"[{', '.join(dims)}]"
        except Exception:
            return "[?]"
