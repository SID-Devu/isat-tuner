"""SLA (Service Level Agreement) validation for inference models.

Validates that a model meets production deployment requirements:
  - Latency targets (P50, P95, P99)
  - Throughput minimums
  - Memory limits
  - First-inference cold-start limits
  - Error rate tolerance
  - Tail latency bounds

Outputs a pass/fail verdict with detailed per-metric results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("isat.sla")


@dataclass
class SLARequirement:
    name: str
    metric: str  # "p50_ms", "p95_ms", "p99_ms", "throughput_rps", "memory_mb", "cold_start_ms"
    operator: str  # "<=", ">=", "<", ">"
    threshold: float
    unit: str = "ms"
    critical: bool = True


@dataclass
class SLACheckResult:
    requirement: SLARequirement
    actual_value: float
    passed: bool
    margin: float
    message: str


@dataclass
class SLAResult:
    model: str
    all_passed: bool
    checks: list[SLACheckResult] = field(default_factory=list)
    critical_failures: int = 0
    warnings: int = 0

    def summary(self) -> str:
        lines = [
            f"  {'Requirement':<30} {'Target':>12} {'Actual':>12} {'Margin':>10} {'Status':>8}",
            f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10} {'-'*8}",
        ]
        for c in self.checks:
            status = "PASS" if c.passed else ("FAIL" if c.requirement.critical else "WARN")
            margin_str = f"{c.margin:+.1f}%" if c.margin != 0 else "exact"
            lines.append(
                f"  {c.requirement.name:<30} "
                f"{c.requirement.operator}{c.requirement.threshold:>.1f}{c.requirement.unit:>4} "
                f"{c.actual_value:>8.2f}{c.requirement.unit:>4} "
                f"{margin_str:>10} "
                f"{status:>8}"
            )
        lines.append(f"\n  Verdict: {'ALL REQUIREMENTS MET' if self.all_passed else f'{self.critical_failures} CRITICAL FAILURES'}")
        return "\n".join(lines)


# Pre-built SLA templates for common deployment scenarios
SLA_TEMPLATES: dict[str, list[SLARequirement]] = {
    "realtime": [
        SLARequirement("P50 latency", "p50_ms", "<=", 10.0),
        SLARequirement("P95 latency", "p95_ms", "<=", 20.0),
        SLARequirement("P99 latency", "p99_ms", "<=", 50.0),
        SLARequirement("Throughput", "throughput_rps", ">=", 100.0, unit="rps"),
        SLARequirement("Cold start", "cold_start_ms", "<=", 5000.0),
    ],
    "batch": [
        SLARequirement("P95 latency", "p95_ms", "<=", 500.0),
        SLARequirement("P99 latency", "p99_ms", "<=", 1000.0),
        SLARequirement("Throughput", "throughput_rps", ">=", 10.0, unit="rps"),
    ],
    "edge": [
        SLARequirement("P50 latency", "p50_ms", "<=", 30.0),
        SLARequirement("P99 latency", "p99_ms", "<=", 100.0),
        SLARequirement("Memory limit", "memory_mb", "<=", 2048.0, unit="MB"),
        SLARequirement("Cold start", "cold_start_ms", "<=", 10000.0),
    ],
    "llm": [
        SLARequirement("P50 latency", "p50_ms", "<=", 200.0),
        SLARequirement("P95 latency", "p95_ms", "<=", 500.0),
        SLARequirement("P99 latency", "p99_ms", "<=", 1000.0),
        SLARequirement("Memory limit", "memory_mb", "<=", 65536.0, unit="MB"),
    ],
    "mobile": [
        SLARequirement("P50 latency", "p50_ms", "<=", 16.0),
        SLARequirement("P99 latency", "p99_ms", "<=", 33.0),
        SLARequirement("Memory limit", "memory_mb", "<=", 512.0, unit="MB"),
    ],
}


class SLAValidator:
    """Validate model inference against SLA requirements."""

    def __init__(self, requirements: list[SLARequirement] | None = None, template: str = ""):
        if template and template in SLA_TEMPLATES:
            self.requirements = list(SLA_TEMPLATES[template])
        elif requirements:
            self.requirements = requirements
        else:
            self.requirements = []

    def validate(self, metrics: dict[str, float], model: str = "") -> SLAResult:
        """Validate metrics against SLA requirements.

        metrics keys: p50_ms, p95_ms, p99_ms, throughput_rps, memory_mb, cold_start_ms
        """
        checks: list[SLACheckResult] = []
        critical_fails = 0
        warnings = 0

        for req in self.requirements:
            actual = metrics.get(req.metric)
            if actual is None:
                checks.append(SLACheckResult(
                    requirement=req, actual_value=0, passed=False,
                    margin=0, message=f"Metric '{req.metric}' not available",
                ))
                if req.critical:
                    critical_fails += 1
                continue

            passed = _evaluate(actual, req.operator, req.threshold)

            if req.threshold != 0:
                margin = ((req.threshold - actual) / req.threshold * 100
                          if req.operator in ("<=", "<")
                          else (actual - req.threshold) / req.threshold * 100)
            else:
                margin = 0

            msg = "OK" if passed else f"VIOLATED: {actual:.2f} {req.unit} vs {req.operator} {req.threshold} {req.unit}"

            checks.append(SLACheckResult(
                requirement=req, actual_value=actual, passed=passed,
                margin=margin, message=msg,
            ))

            if not passed:
                if req.critical:
                    critical_fails += 1
                else:
                    warnings += 1

        return SLAResult(
            model=model,
            all_passed=critical_fails == 0,
            checks=checks,
            critical_failures=critical_fails,
            warnings=warnings,
        )

    @staticmethod
    def list_templates() -> dict[str, list[dict]]:
        return {
            name: [{"name": r.name, "metric": r.metric, "op": r.operator,
                     "threshold": r.threshold, "unit": r.unit}
                    for r in reqs]
            for name, reqs in SLA_TEMPLATES.items()
        }


def _evaluate(value: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return value <= threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == ">":
        return value > threshold
    return False
