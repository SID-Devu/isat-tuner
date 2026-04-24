"""Performance regression detection.

Compares current tuning results against a historical baseline
(from the results database) to detect regressions caused by
driver updates, kernel changes, or model modifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from isat.database.store import ResultsDB

log = logging.getLogger("isat.analysis.regression")


@dataclass
class RegressionAlert:
    config_label: str
    metric: str
    baseline_value: float
    current_value: float
    delta_pct: float
    severity: str

    @property
    def summary(self) -> str:
        direction = "regression" if self.delta_pct > 0 else "improvement"
        return (
            f"[{self.severity.upper()}] {self.config_label}: {self.metric} "
            f"{direction} {abs(self.delta_pct):.1f}% "
            f"({self.baseline_value:.2f} -> {self.current_value:.2f})"
        )


class RegressionDetector:
    """Compare current results against historical baselines."""

    THRESHOLDS = {
        "warning": 5.0,
        "critical": 15.0,
    }

    def __init__(
        self,
        db: ResultsDB,
        model_name: str,
        hw_hash: str,
    ):
        self.db = db
        self.model_name = model_name
        self.hw_hash = hw_hash

    def check(
        self,
        current_results: list[dict],
        metric: str = "mean_ms",
    ) -> list[RegressionAlert]:
        """Check for regressions against the best historical result for each config."""
        alerts: list[RegressionAlert] = []

        historical = self.db.all_runs(model_name=self.model_name)
        if not historical:
            log.info("No historical data for regression comparison")
            return alerts

        baseline_map: dict[str, float] = {}
        for row in historical:
            label = row["config_label"]
            val = row.get(metric)
            if val is None:
                continue
            if label not in baseline_map or val < baseline_map[label]:
                baseline_map[label] = val

        for result in current_results:
            label = result.get("label") or result.get("config_label", "")
            current_val = result.get(metric)
            if current_val is None or label not in baseline_map:
                continue

            baseline_val = baseline_map[label]
            if baseline_val <= 0:
                continue

            delta_pct = ((current_val - baseline_val) / baseline_val) * 100

            if abs(delta_pct) >= self.THRESHOLDS["critical"]:
                severity = "critical"
            elif abs(delta_pct) >= self.THRESHOLDS["warning"]:
                severity = "warning"
            else:
                continue

            if delta_pct > 0:
                alerts.append(RegressionAlert(
                    config_label=label,
                    metric=metric,
                    baseline_value=baseline_val,
                    current_value=current_val,
                    delta_pct=delta_pct,
                    severity=severity,
                ))

        return alerts
