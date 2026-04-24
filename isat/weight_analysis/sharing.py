"""Weight sharing detector -- find tied/shared weights across layers.

Identifies:
  - Identical weight tensors (shared by name or value)
  - Near-identical weights (high cosine similarity)
  - Weight reuse patterns across layers
  - Memory savings from detected sharing

Critical for: memory optimization, model compression, architecture understanding.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("isat.weight_analysis")


@dataclass
class SharedWeightGroup:
    hash_key: str
    names: list[str]
    shape: tuple
    params: int
    similarity: float
    memory_saved_mb: float


@dataclass
class WeightSharingReport:
    model_path: str
    total_initializers: int
    total_params: int
    unique_weights: int
    shared_groups: list[SharedWeightGroup] = field(default_factory=list)
    total_memory_mb: float = 0
    potential_savings_mb: float = 0
    sharing_ratio: float = 0
    near_identical_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Model             : {self.model_path}",
            f"  Total initializers: {self.total_initializers}",
            f"  Total params      : {self.total_params:,}",
            f"  Unique weights    : {self.unique_weights}",
            f"  Sharing ratio     : {self.sharing_ratio:.1%}",
            f"  Total memory      : {self.total_memory_mb:.1f} MB",
            f"  Potential savings : {self.potential_savings_mb:.1f} MB",
        ]
        if self.shared_groups:
            lines.append(f"\n  Shared weight groups ({len(self.shared_groups)}):")
            for g in self.shared_groups[:10]:
                names_str = ", ".join(g.names[:3])
                if len(g.names) > 3:
                    names_str += f" +{len(g.names)-3} more"
                lines.append(f"    Shape {g.shape}: {names_str}")
                lines.append(f"      Params: {g.params:,}  Saved: {g.memory_saved_mb:.2f} MB")
        if self.near_identical_pairs:
            lines.append(f"\n  Near-identical pairs ({len(self.near_identical_pairs)}):")
            for a, b, sim in self.near_identical_pairs[:10]:
                lines.append(f"    {a} <-> {b}  (cosine sim: {sim:.4f})")
        return "\n".join(lines)


class WeightSharingDetector:
    """Detect shared and near-identical weights in ONNX models."""

    def __init__(self, model_path: str, similarity_threshold: float = 0.999):
        import onnx
        self.model_path = model_path
        self.model = onnx.load(str(model_path), load_external_data=False)
        self.similarity_threshold = similarity_threshold

    def analyze(self) -> WeightSharingReport:
        from onnx import numpy_helper

        hash_groups: dict[str, list[tuple[str, np.ndarray]]] = {}
        total_params = 0
        total_memory = 0

        for init in self.model.graph.initializer:
            arr = numpy_helper.to_array(init)
            total_params += arr.size
            total_memory += arr.nbytes

            h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append((init.name, arr))

        shared_groups = []
        potential_savings = 0
        for h, members in hash_groups.items():
            if len(members) > 1:
                arr = members[0][1]
                per_copy = arr.nbytes / (1024 * 1024)
                saved = per_copy * (len(members) - 1)
                potential_savings += saved
                shared_groups.append(SharedWeightGroup(
                    hash_key=h, names=[n for n, _ in members],
                    shape=tuple(arr.shape), params=arr.size,
                    similarity=1.0, memory_saved_mb=saved,
                ))

        near_pairs = []
        large_inits = [(init.name, numpy_helper.to_array(init))
                       for init in self.model.graph.initializer
                       if numpy_helper.to_array(init).size >= 100]

        for i in range(len(large_inits)):
            for j in range(i + 1, min(i + 50, len(large_inits))):
                name_a, arr_a = large_inits[i]
                name_b, arr_b = large_inits[j]
                if arr_a.shape != arr_b.shape:
                    continue
                flat_a = arr_a.flatten().astype(np.float64)
                flat_b = arr_b.flatten().astype(np.float64)
                norm_a = np.linalg.norm(flat_a)
                norm_b = np.linalg.norm(flat_b)
                if norm_a < 1e-8 or norm_b < 1e-8:
                    continue
                cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))
                if cosine > self.similarity_threshold and cosine < 1.0:
                    near_pairs.append((name_a, name_b, cosine))

        unique = len(hash_groups)
        total_inits = len(self.model.graph.initializer)
        sharing_ratio = 1 - unique / max(total_inits, 1)

        return WeightSharingReport(
            model_path=self.model_path,
            total_initializers=total_inits,
            total_params=total_params,
            unique_weights=unique,
            shared_groups=shared_groups,
            total_memory_mb=total_memory / (1024 * 1024),
            potential_savings_mb=potential_savings,
            sharing_ratio=sharing_ratio,
            near_identical_pairs=near_pairs,
        )
