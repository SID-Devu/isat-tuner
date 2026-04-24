"""ONNX model pruning -- remove low-magnitude weights to shrink models.

Strategies:
  - Magnitude pruning: zero out weights below threshold
  - Percentage pruning: remove bottom N% of weights per layer
  - Structured pruning: remove entire filters/channels
  - Global pruning: single threshold across all layers

Reports per-layer sparsity and estimated speedup.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("isat.pruning")

_VALID_STRATEGIES = frozenset({"magnitude", "percentage", "global", "structured"})


@dataclass
class LayerPruneInfo:
    name: str
    original_params: int
    pruned_params: int
    sparsity_pct: float
    shape: tuple[int, ...]


@dataclass
class PruneResult:
    model_path: str
    output_path: str
    strategy: str
    target_sparsity: float
    layers: list[LayerPruneInfo] = field(default_factory=list)
    total_params: int = 0
    total_pruned: int = 0
    overall_sparsity: float = 0.0
    size_before_mb: float = 0.0
    size_after_mb: float = 0.0
    estimated_speedup: float = 1.0

    def summary(self) -> str:
        lines = [
            f"  Strategy          : {self.strategy}",
            f"  Target sparsity   : {self.target_sparsity:.1%}",
            f"  Total params      : {self.total_params:,}",
            f"  Pruned params     : {self.total_pruned:,}",
            f"  Overall sparsity  : {self.overall_sparsity:.1%}",
            f"  Est. speedup*     : {self.estimated_speedup:.2f}x",
            f"  Size              : {self.size_before_mb:.1f} MB -> {self.size_after_mb:.1f} MB",
            "",
            f"  *Unstructured MAC upper bound ~ 1/(1-s); structured varies by kernel.",
            "",
            f"  {'Layer':<40} {'Params':>10} {'Pruned':>10} {'Sparsity':>10}",
            f"  {'-' * 40} {'-' * 10} {'-' * 10} {'-' * 10}",
        ]
        for layer in self.layers[:20]:
            lines.append(
                f"  {layer.name[:40]:<40} {layer.original_params:>10,} "
                f"{layer.pruned_params:>10,} {layer.sparsity_pct:>9.1f}%"
            )
        if len(self.layers) > 20:
            lines.append(f"  ... and {len(self.layers) - 20} more layers")
        return "\n".join(lines)


def _estimate_unstructured_speedup(overall_sparsity: float) -> float:
    """Heuristic peak speedup from MAC reduction for unstructured sparsity (ideal hardware)."""
    if overall_sparsity <= 0.0:
        return 1.0
    if overall_sparsity >= 0.999:
        return 1000.0
    # Ideal: fraction of MACs remaining = 1 - s; speedup ceiling ~ 1/(1-s) if memory not bound.
    denom = max(1.0 - overall_sparsity, 1e-6)
    return float(min(denom**-1, 1000.0))


def _initializer_consumer_ops(model: Any) -> dict[str, list[str]]:
    """Map initializer tensor names to op_types of nodes that consume them."""
    init_names = {init.name for init in model.graph.initializer}
    out: dict[str, list[str]] = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp in init_names:
                out.setdefault(inp, []).append(node.op_type)
    return out


def _apply_structured_mask(
    arr: np.ndarray,
    sparsity: float,
    consumer_ops: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Zero whole filters (Conv) or output features (Gemm). Returns (pruned_array, boolean_mask)."""
    mask = np.zeros(arr.shape, dtype=bool)
    s = float(np.clip(sparsity, 0.0, 0.999))

    if arr.ndim == 4 and "Conv" in consumer_ops:
        # [out_ch, in_ch, kH, kW]
        norms = np.sum(np.abs(arr), axis=(1, 2, 3))
        n_out = arr.shape[0]
        k = int(round(n_out * s))
        k = max(0, min(k, n_out - 1))
        if k <= 0:
            return arr.copy(), mask
        dead = np.argpartition(norms, k - 1)[:k]
        mask[dead, ...] = True
    elif arr.ndim == 2 and "Gemm" in consumer_ops:
        # [out_features, in_features] — prune output rows
        norms = np.sum(np.abs(arr), axis=1)
        n_out = arr.shape[0]
        k = int(round(n_out * s))
        k = max(0, min(k, n_out - 1))
        if k <= 0:
            return arr.copy(), mask
        dead = np.argpartition(norms, k - 1)[:k]
        mask[dead, :] = True
    else:
        # Fallback: unstructured percentage on this tensor
        flat = np.abs(arr).ravel()
        if flat.size == 0:
            return arr.copy(), mask
        thresh = float(np.percentile(flat, s * 100.0))
        mask = np.abs(arr) < thresh

    out = arr.copy()
    out[mask] = 0.0
    return out, mask


class ModelPruner:
    """Prune ONNX model weights."""

    def __init__(self, model_path: str) -> None:
        import onnx

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = str(path.resolve())
        self.model = onnx.load(self.model_path, load_external_data=True)

    def prune(
        self,
        strategy: str = "magnitude",
        sparsity: float = 0.5,
        output_path: str = "",
    ) -> PruneResult:
        import onnx
        from onnx import numpy_helper

        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected one of {sorted(_VALID_STRATEGIES)}"
            )
        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

        work = copy.deepcopy(self.model)
        layers: list[LayerPruneInfo] = []
        total_params = 0
        total_pruned = 0

        consumers = _initializer_consumer_ops(work)

        global_threshold = 0.0
        if strategy == "global":
            chunks: list[np.ndarray] = []
            for init in work.graph.initializer:
                arr = numpy_helper.to_array(init)
                if arr.size > 1 and np.issubdtype(arr.dtype, np.floating):
                    chunks.append(np.abs(arr).ravel())
            if chunks:
                concat = np.concatenate(chunks)
                global_threshold = float(np.percentile(concat, sparsity * 100.0))

        graph = work.graph
        for i, init in enumerate(graph.initializer):
            arr = numpy_helper.to_array(init)
            if arr.size <= 1 or not np.issubdtype(arr.dtype, np.floating):
                continue

            original_count = int(arr.size)
            total_params += original_count

            if strategy == "magnitude":
                scale = float(np.std(arr)) if arr.size > 1 else 0.0
                thresh = scale * (1.0 - sparsity) * 0.5
                mask = np.abs(arr) < thresh
            elif strategy == "percentage":
                flat = np.abs(arr).ravel()
                thresh = float(np.percentile(flat, sparsity * 100.0))
                mask = np.abs(arr) < thresh
            elif strategy == "global":
                mask = np.abs(arr) < global_threshold
            elif strategy == "structured":
                pruned_arr, mask = _apply_structured_mask(arr, sparsity, consumers.get(init.name, []))
                pruned_count = int(mask.sum())
                total_pruned += pruned_count
                new_init = numpy_helper.from_array(pruned_arr, name=init.name)
                graph.initializer[i].CopyFrom(new_init)
                layers.append(
                    LayerPruneInfo(
                        name=init.name,
                        original_params=original_count,
                        pruned_params=pruned_count,
                        sparsity_pct=pruned_count / max(original_count, 1) * 100.0,
                        shape=tuple(int(x) for x in arr.shape),
                    )
                )
                continue

            pruned_arr = arr.copy()
            pruned_arr[mask] = 0.0
            pruned_count = int(mask.sum())
            total_pruned += pruned_count

            new_init = numpy_helper.from_array(pruned_arr, name=init.name)
            graph.initializer[i].CopyFrom(new_init)

            layers.append(
                LayerPruneInfo(
                    name=init.name,
                    original_params=original_count,
                    pruned_params=pruned_count,
                    sparsity_pct=pruned_count / max(original_count, 1) * 100.0,
                    shape=tuple(int(x) for x in arr.shape),
                )
            )

        base = Path(self.model_path)
        out = output_path or str(base.with_name(f"{base.stem}_pruned{base.suffix or '.onnx'}"))
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        size_before = Path(self.model_path).stat().st_size / (1024 * 1024)
        onnx.save(work, str(out_path))
        size_after = out_path.stat().st_size / (1024 * 1024)

        overall = total_pruned / max(total_params, 1)
        speedup = _estimate_unstructured_speedup(overall)
        if strategy == "structured":
            # Structured zeros align with compute; use a slightly more optimistic bound than unstructured.
            speedup = float(min(max(1.0, 1.0 / max(1.0 - overall, 1e-6)), 1000.0))

        self.model = work

        return PruneResult(
            model_path=self.model_path,
            output_path=str(out_path.resolve()),
            strategy=strategy,
            target_sparsity=sparsity,
            layers=layers,
            total_params=total_params,
            total_pruned=total_pruned,
            overall_sparsity=overall,
            size_before_mb=size_before,
            size_after_mb=size_after,
            estimated_speedup=speedup,
        )

    def analyze_sparsity(self) -> dict[str, Any]:
        from onnx import numpy_helper

        result: dict[str, Any] = {
            "layers": [],
            "total_params": 0,
            "total_zeros": 0,
        }
        for init in self.model.graph.initializer:
            arr = numpy_helper.to_array(init)
            if arr.size <= 1:
                continue
            zeros = int(np.sum(arr == 0))
            result["layers"].append(
                {
                    "name": init.name,
                    "shape": list(arr.shape),
                    "params": int(arr.size),
                    "zeros": zeros,
                    "sparsity": zeros / max(arr.size, 1),
                }
            )
            result["total_params"] += int(arr.size)
            result["total_zeros"] += zeros
        tp = int(result["total_params"])
        tz = int(result["total_zeros"])
        result["overall_sparsity"] = tz / max(tp, 1)
        return result
