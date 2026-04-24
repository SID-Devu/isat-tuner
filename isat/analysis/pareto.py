"""Pareto frontier analysis.

Finds the set of configurations that are not dominated in any
combination of objectives (e.g., latency vs memory vs power).
Useful for deployment decisions where multiple constraints matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from isat.search.engine import TuneResult


@dataclass
class ParetoPoint:
    result: TuneResult
    objectives: dict[str, float]
    is_dominated: bool = False


class ParetoFrontier:
    """Multi-objective Pareto analysis over tuning results."""

    OBJECTIVES = {
        "latency_ms": lambda r: r.mean_latency_ms,
        "p95_ms": lambda r: r.p95_latency_ms,
        "memory_mb": lambda r: r.peak_vram_mb + r.peak_gtt_mb,
        "power_w": lambda r: r.peak_power_w,
        "temp_c": lambda r: r.peak_gpu_temp_c,
    }

    def __init__(
        self,
        results: list[TuneResult],
        objectives: list[str] | None = None,
    ):
        self.results = [r for r in results if r.error is None]
        self.objective_names = objectives or ["latency_ms", "memory_mb"]

        obj_fns = {name: self.OBJECTIVES[name] for name in self.objective_names}
        self.points: list[ParetoPoint] = []
        for r in self.results:
            objs = {name: fn(r) for name, fn in obj_fns.items()}
            self.points.append(ParetoPoint(result=r, objectives=objs))

        self._compute_dominance()

    def _compute_dominance(self) -> None:
        """Mark non-dominated points (the Pareto frontier)."""
        n = len(self.points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(self.points[j], self.points[i]):
                    self.points[i].is_dominated = True
                    break

    @staticmethod
    def _dominates(a: ParetoPoint, b: ParetoPoint) -> bool:
        """Return True if a dominates b (a is <= b in all objectives, < in at least one)."""
        at_least_one_better = False
        for key in a.objectives:
            if a.objectives[key] > b.objectives[key]:
                return False
            if a.objectives[key] < b.objectives[key]:
                at_least_one_better = True
        return at_least_one_better

    @property
    def frontier(self) -> list[ParetoPoint]:
        return [p for p in self.points if not p.is_dominated]

    @property
    def dominated(self) -> list[ParetoPoint]:
        return [p for p in self.points if p.is_dominated]

    def summary(self) -> str:
        lines = [
            f"Pareto Frontier Analysis ({', '.join(self.objective_names)})",
            f"  Total configs  : {len(self.points)}",
            f"  Frontier size  : {len(self.frontier)}",
            f"  Dominated      : {len(self.dominated)}",
            "",
        ]
        for i, p in enumerate(self.frontier, 1):
            obj_str = ", ".join(f"{k}={v:.2f}" for k, v in p.objectives.items())
            lines.append(f"  [{i}] {p.result.config.label}")
            lines.append(f"      {obj_str}")
        return "\n".join(lines)

    def recommend(self, priority: str = "latency_ms") -> ParetoPoint | None:
        """Recommend the best frontier point for a given priority."""
        if not self.frontier:
            return None
        if priority not in self.objective_names:
            priority = self.objective_names[0]
        return min(self.frontier, key=lambda p: p.objectives.get(priority, float("inf")))
