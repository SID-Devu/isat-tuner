"""Alert rules engine for inference monitoring.

Define rules like:
  - "if P99 latency > 500ms for 3 consecutive checks, CRITICAL"
  - "if error rate > 1% for 1 check, WARNING"
  - "if GPU temp > 85C, CRITICAL"
  - "if throughput < 10 rps, WARNING"

Evaluates rules against live metrics and triggers notifications.
Used in production inference monitoring (like Prometheus AlertManager).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("isat.alerts")


@dataclass
class AlertRule:
    name: str
    metric: str
    operator: str  # ">", "<", ">=", "<=", "=="
    threshold: float
    severity: str = "warning"  # "info", "warning", "critical"
    consecutive: int = 1
    cooldown_s: float = 300.0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "metric": self.metric,
            "operator": self.operator, "threshold": self.threshold,
            "severity": self.severity, "consecutive": self.consecutive,
        }


@dataclass
class Alert:
    rule: AlertRule
    triggered_at: float
    value: float
    message: str
    acknowledged: bool = False


@dataclass
class AlertStatus:
    active_alerts: list[Alert] = field(default_factory=list)
    total_triggered: int = 0
    total_resolved: int = 0
    rules_count: int = 0
    checks_performed: int = 0

    def summary(self) -> str:
        lines = [
            f"  Rules loaded      : {self.rules_count}",
            f"  Checks performed  : {self.checks_performed}",
            f"  Total triggered   : {self.total_triggered}",
            f"  Total resolved    : {self.total_resolved}",
            f"  Active alerts     : {len(self.active_alerts)}",
        ]
        if self.active_alerts:
            lines.append(f"")
            lines.append(f"  {'Severity':<10} {'Rule':<30} {'Value':>10} {'Threshold':>10} {'Age':>10}")
            lines.append(f"  {'-'*10} {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
            now = time.time()
            for a in self.active_alerts:
                age = now - a.triggered_at
                age_str = f"{age:.0f}s" if age < 60 else f"{age/60:.0f}m"
                lines.append(
                    f"  {a.rule.severity:<10} {a.rule.name:<30} "
                    f"{a.value:>10.2f} {a.rule.threshold:>10.2f} {age_str:>10}"
                )
        else:
            lines.append(f"\n  All clear -- no active alerts")
        return "\n".join(lines)


BUILTIN_RULES = [
    AlertRule("high_p99_latency", "p99_ms", ">", 500, "critical", consecutive=3,
             message="P99 latency exceeds 500ms"),
    AlertRule("high_p95_latency", "p95_ms", ">", 200, "warning", consecutive=2),
    AlertRule("high_error_rate", "error_rate", ">", 0.01, "critical", consecutive=1,
             message="Error rate exceeds 1%"),
    AlertRule("low_throughput", "throughput_rps", "<", 10, "warning", consecutive=3),
    AlertRule("high_gpu_temp", "gpu_temp_c", ">", 85, "critical", consecutive=1,
             message="GPU temperature exceeds 85C"),
    AlertRule("high_memory_usage", "gpu_vram_used_pct", ">", 90, "warning", consecutive=2),
    AlertRule("high_queue_depth", "queue_depth", ">", 100, "warning", consecutive=2),
    AlertRule("cold_start_slow", "cold_start_ms", ">", 30000, "warning", consecutive=1),
]


class AlertEngine:
    """Evaluate alert rules against metrics."""

    def __init__(self, rules: list[AlertRule] | None = None):
        self.rules = rules or list(BUILTIN_RULES)
        self._consecutive_counts: dict[str, int] = {}
        self._active: dict[str, Alert] = {}
        self._last_triggered: dict[str, float] = {}
        self._total_triggered = 0
        self._total_resolved = 0
        self._checks = 0

    def check(self, metrics: dict[str, float]) -> list[Alert]:
        self._checks += 1
        new_alerts: list[Alert] = []

        for rule in self.rules:
            value = metrics.get(rule.metric)
            if value is None:
                self._consecutive_counts[rule.name] = 0
                continue

            violated = _evaluate(value, rule.operator, rule.threshold)

            if violated:
                self._consecutive_counts[rule.name] = self._consecutive_counts.get(rule.name, 0) + 1
            else:
                if rule.name in self._active:
                    self._total_resolved += 1
                    del self._active[rule.name]
                self._consecutive_counts[rule.name] = 0
                continue

            if self._consecutive_counts[rule.name] >= rule.consecutive:
                last = self._last_triggered.get(rule.name, 0)
                if time.time() - last < rule.cooldown_s and rule.name in self._active:
                    continue

                msg = rule.message or f"{rule.metric} {rule.operator} {rule.threshold} (actual: {value:.2f})"
                alert = Alert(rule=rule, triggered_at=time.time(), value=value, message=msg)
                self._active[rule.name] = alert
                self._last_triggered[rule.name] = time.time()
                self._total_triggered += 1
                new_alerts.append(alert)
                log.warning("ALERT [%s] %s: %s", rule.severity.upper(), rule.name, msg)

        return new_alerts

    def status(self) -> AlertStatus:
        return AlertStatus(
            active_alerts=list(self._active.values()),
            total_triggered=self._total_triggered,
            total_resolved=self._total_resolved,
            rules_count=len(self.rules),
            checks_performed=self._checks,
        )

    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)

    def remove_rule(self, name: str):
        self.rules = [r for r in self.rules if r.name != name]
        self._active.pop(name, None)

    def acknowledge(self, rule_name: str):
        if rule_name in self._active:
            self._active[rule_name].acknowledged = True

    def export_rules(self, path: str):
        data = [r.to_dict() for r in self.rules]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_rules(cls, path: str) -> "AlertEngine":
        with open(path) as f:
            data = json.load(f)
        rules = [AlertRule(**r) for r in data]
        return cls(rules=rules)

    @staticmethod
    def list_builtin() -> list[dict]:
        return [r.to_dict() for r in BUILTIN_RULES]


def _evaluate(value: float, operator: str, threshold: float) -> bool:
    ops = {">": lambda a, b: a > b, "<": lambda a, b: a < b,
           ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
           "==": lambda a, b: a == b}
    return ops.get(operator, lambda a, b: False)(value, threshold)
