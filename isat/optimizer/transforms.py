"""Individual model transformation utilities.

Provides standalone transforms that can be composed or used individually
for surgical model modifications.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.optimizer.transforms")


class ModelTransformer:
    """Utility class for individual model transformations."""

    @staticmethod
    def count_ops(model_path: str) -> dict[str, int]:
        """Count operators in an ONNX model."""
        import onnx
        model = onnx.load(model_path, load_external_data=False)
        counts: dict[str, int] = {}
        for node in model.graph.node:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    @staticmethod
    def find_bottleneck_ops(model_path: str) -> list[dict]:
        """Identify potential performance bottleneck operations."""
        import onnx
        model = onnx.load(model_path, load_external_data=False)

        expensive_ops = {
            "Conv": "high", "MatMul": "high", "Gemm": "high",
            "Attention": "high", "MultiHeadAttention": "high",
            "LayerNormalization": "medium", "BatchNormalization": "medium",
            "Softmax": "medium", "Transpose": "medium",
            "Reshape": "low", "Squeeze": "low", "Unsqueeze": "low",
        }

        bottlenecks = []
        for node in model.graph.node:
            cost = expensive_ops.get(node.op_type)
            if cost in ("high", "medium"):
                bottlenecks.append({
                    "name": node.name or node.op_type,
                    "op_type": node.op_type,
                    "cost": cost,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                })

        return bottlenecks

    @staticmethod
    def get_model_metadata(model_path: str) -> dict:
        """Extract comprehensive model metadata."""
        import onnx
        model = onnx.load(model_path, load_external_data=False)

        file_size = Path(model_path).stat().st_size

        total_params = 0
        for init in model.graph.initializer:
            n = 1
            for d in init.dims:
                n *= d
            total_params += n

        return {
            "file_size_mb": file_size / (1024 * 1024),
            "ir_version": model.ir_version,
            "opset": model.opset_import[0].version if model.opset_import else 0,
            "producer": model.producer_name,
            "domain": model.domain,
            "num_nodes": len(model.graph.node),
            "num_initializers": len(model.graph.initializer),
            "num_inputs": len([i for i in model.graph.input if i.name not in {x.name for x in model.graph.initializer}]),
            "num_outputs": len(model.graph.output),
            "total_params": total_params,
            "total_params_human": _human_readable(total_params),
        }

    @staticmethod
    def remove_unused_initializers(model_path: str, output_path: str) -> int:
        """Remove initializers not referenced by any node."""
        import onnx
        model = onnx.load(model_path)

        used_names = set()
        for node in model.graph.node:
            used_names.update(node.input)

        to_remove = []
        for init in model.graph.initializer:
            if init.name not in used_names:
                to_remove.append(init)

        for init in to_remove:
            model.graph.initializer.remove(init)

        onnx.save(model, output_path)
        log.info("Removed %d unused initializers", len(to_remove))
        return len(to_remove)

    @staticmethod
    def extract_subgraph(model_path: str, output_path: str, input_names: list[str], output_names: list[str]) -> bool:
        """Extract a subgraph from a model."""
        try:
            import onnx
            from onnx.utils import extract_model
            extract_model(model_path, output_path, input_names, output_names)
            return True
        except Exception as e:
            log.error("Subgraph extraction failed: %s", e)
            return False


def _human_readable(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
