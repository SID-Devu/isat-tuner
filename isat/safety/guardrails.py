"""Regex-based safety guardrails for model inputs and outputs."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in pattern lists
# ---------------------------------------------------------------------------

PII_PATTERNS: Dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(
        r"(?<!\d)(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "ip_address": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
}

TOXIC_KEYWORDS: List[str] = [
    "kill yourself",
    "kys",
    "murder",
    "bomb threat",
    "shoot up",
    "suicide instructions",
    "self-harm",
    "child abuse",
    "terrorism",
    "genocide",
    "ethnic cleansing",
    "white supremacy",
    "racial slur",
    "hate speech",
    "death threat",
    "sexual assault",
    "rape",
    "molest",
    "human trafficking",
    "drug synthesis",
    "weapon manufacturing",
]

JAILBREAK_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bDAN\b.*(?:mode|prompt|jailbreak)"),
    re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions"),
    re.compile(r"(?i)disregard\s+(all\s+)?(prior|previous|above)\s+instructions"),
    re.compile(r"(?i)forget\s+(all\s+)?(your|prior|previous)\s+(rules|instructions|guidelines)"),
    re.compile(r"(?i)you\s+are\s+now\s+(a|an)\s+(?!assistant)"),
    re.compile(r"(?i)pretend\s+(you\s+are|to\s+be)\s+(?!helpful)"),
    re.compile(r"(?i)act\s+as\s+(a|an)\s+(?:unrestricted|unfiltered|evil)"),
    re.compile(r"(?i)roleplay\s+as\s+(a|an)\s+"),
    re.compile(r"(?i)respond\s+without\s+(any\s+)?(ethical|safety|moral)\s+(guidelines|filters|restrictions)"),
    re.compile(r"(?i)bypass\s+(your\s+)?(safety|content)\s+(filters?|restrictions?)"),
    re.compile(r"(?i)developer\s+mode\s+(enabled|activated|on)"),
    re.compile(r"(?i)sudo\s+mode"),
    re.compile(r"(?i)base64\s*[\(:]"),
    re.compile(r"(?i)encode\s+(this|the\s+following)\s+in\s+base64"),
    re.compile(r"(?i)translate\s+to\s+(hex|binary|rot13|base64)"),
    re.compile(r"(?i)(?:system|assistant)\s*:\s*you\s+(?:can|should|must|will)\s+now"),
    re.compile(r"(?i)override\s+(system|safety)\s+(prompt|message|instructions)"),
    re.compile(r"(?i)\[system\]\s*#"),
]

_HEDGING_PHRASES: List[re.Pattern] = [
    re.compile(r"(?i)\bI(?:'m| am) not (?:sure|certain|confident)\b"),
    re.compile(r"(?i)\bI (?:think|believe|suppose|guess)\b.*\bbut\b"),
    re.compile(r"(?i)\bthis (?:may|might|could) (?:not )?be (?:accurate|correct|right)\b"),
    re.compile(r"(?i)\bI cannot verify\b"),
    re.compile(r"(?i)\bI don'?t (?:actually )?(?:know|have)\b"),
]

_CONTRADICTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bbut\s+(?:actually|in fact|on the other hand)\b"),
    re.compile(r"(?i)\bwait,?\s+(?:no|actually)\b"),
    re.compile(r"(?i)\bcontrary to what I (?:just )?said\b"),
    re.compile(r"(?i)\bI (?:was|am) wrong\b"),
    re.compile(r"(?i)\bcorrection:\b"),
]

_PROMPT_INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)ignore\s+(the\s+)?(above|previous)\s+(text|prompt|instructions?)"),
    re.compile(r"(?i)new\s+instructions?\s*:"),
    re.compile(r"(?i)system\s*:\s*"),
    re.compile(r"(?i)<<\s*(?:SYS|SYSTEM|INST)\s*>>"),
    re.compile(r"(?i)\[INST\]"),
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SafetyCheckResult:
    passed: bool
    category: str
    severity: str  # "low" | "medium" | "high" | "critical"
    findings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyReport:
    overall_safe: bool
    checks: List[SafetyCheckResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# SafetyGuard
# ---------------------------------------------------------------------------

class SafetyGuard:
    """Configurable safety checker for model I/O."""

    _SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.pii_patterns: Dict[str, re.Pattern] = {
            **PII_PATTERNS,
            **{k: re.compile(v) for k, v in cfg.get("extra_pii_patterns", {}).items()},
        }
        self.toxic_keywords: List[str] = TOXIC_KEYWORDS + cfg.get("extra_toxic_keywords", [])
        self.jailbreak_patterns: List[re.Pattern] = JAILBREAK_PATTERNS + [
            re.compile(p) for p in cfg.get("extra_jailbreak_patterns", [])
        ]
        self.confidence_threshold: float = cfg.get("confidence_threshold", 0.1)
        self.entropy_limit: float = cfg.get("entropy_limit", 3.0)

    # ---- input checks -----------------------------------------------------

    def check_input_text(self, text: str) -> SafetyCheckResult:
        findings: List[str] = []
        details: Dict[str, Any] = {}
        severity = "low"

        pii_hits = self._scan_pii(text)
        if pii_hits:
            findings.extend(f"PII detected ({t}): {m}" for t, m in pii_hits)
            details["pii"] = [{"type": t, "match": m} for t, m in pii_hits]
            severity = self._escalate(severity, "high")

        toxic_hits = self._scan_toxicity(text)
        if toxic_hits:
            findings.extend(f"Toxic keyword: {k}" for k in toxic_hits)
            details["toxic_keywords"] = toxic_hits
            severity = self._escalate(severity, "critical")

        injection_hits = self._scan_prompt_injection(text)
        if injection_hits:
            findings.extend(f"Prompt injection pattern: {p}" for p in injection_hits)
            details["prompt_injection"] = injection_hits
            severity = self._escalate(severity, "high")

        passed = len(findings) == 0
        if not passed:
            logger.warning("Input safety check failed: %d finding(s)", len(findings))
        return SafetyCheckResult(
            passed=passed,
            category="input_text",
            severity=severity,
            findings=findings,
            details=details,
        )

    # ---- output checks ----------------------------------------------------

    def check_output_text(self, text: str) -> SafetyCheckResult:
        findings: List[str] = []
        details: Dict[str, Any] = {}
        severity = "low"

        pii_hits = self._scan_pii(text)
        if pii_hits:
            findings.extend(f"PII leak ({t}): {m}" for t, m in pii_hits)
            details["pii_leaks"] = [{"type": t, "match": m} for t, m in pii_hits]
            severity = self._escalate(severity, "critical")

        toxic_hits = self._scan_toxicity(text)
        if toxic_hits:
            findings.extend(f"Toxic content: {k}" for k in toxic_hits)
            details["toxic_content"] = toxic_hits
            severity = self._escalate(severity, "critical")

        hedging = [p.pattern for p in _HEDGING_PHRASES if p.search(text)]
        if hedging:
            findings.append(f"Hedging phrases detected ({len(hedging)})")
            details["hedging_patterns"] = hedging
            severity = self._escalate(severity, "medium")

        contradictions = [p.pattern for p in _CONTRADICTION_PATTERNS if p.search(text)]
        if contradictions:
            findings.append(f"Contradiction indicators detected ({len(contradictions)})")
            details["contradiction_patterns"] = contradictions
            severity = self._escalate(severity, "medium")

        passed = len(findings) == 0
        if not passed:
            logger.warning("Output safety check failed: %d finding(s)", len(findings))
        return SafetyCheckResult(
            passed=passed,
            category="output_text",
            severity=severity,
            findings=findings,
            details=details,
        )

    # ---- confidence checks ------------------------------------------------

    def check_confidence(
        self,
        outputs: List[float],
        threshold: Optional[float] = None,
    ) -> SafetyCheckResult:
        threshold = threshold if threshold is not None else self.confidence_threshold
        findings: List[str] = []
        details: Dict[str, Any] = {"threshold": threshold}

        if not outputs:
            return SafetyCheckResult(
                passed=False,
                category="confidence",
                severity="high",
                findings=["Empty output probabilities"],
                details=details,
            )

        max_prob = max(outputs)
        details["max_probability"] = max_prob

        total = sum(outputs)
        if total > 0:
            normed = [p / total for p in outputs]
            entropy = -sum(p * math.log(p + 1e-12) for p in normed)
        else:
            entropy = float("inf")
        details["entropy"] = entropy
        details["entropy_limit"] = self.entropy_limit

        if max_prob < threshold:
            findings.append(
                f"Max probability {max_prob:.4f} below threshold {threshold}"
            )
        if entropy > self.entropy_limit:
            findings.append(
                f"Entropy {entropy:.4f} exceeds limit {self.entropy_limit}"
            )

        passed = len(findings) == 0
        severity = "low" if passed else "high"
        return SafetyCheckResult(
            passed=passed,
            category="confidence",
            severity=severity,
            findings=findings,
            details=details,
        )

    # ---- output format checks ---------------------------------------------

    def check_output_format(
        self,
        text: str,
        schema: Optional[Dict[str, Any]] = None,
        regex_pattern: Optional[str] = None,
    ) -> SafetyCheckResult:
        findings: List[str] = []
        details: Dict[str, Any] = {}

        if schema is not None:
            try:
                parsed = json.loads(text)
                missing = self._validate_schema(parsed, schema)
                if missing:
                    findings.extend(missing)
                    details["schema_errors"] = missing
                else:
                    details["json_valid"] = True
            except json.JSONDecodeError as exc:
                findings.append(f"Invalid JSON: {exc}")
                details["json_error"] = str(exc)

        if regex_pattern is not None:
            try:
                pat = re.compile(regex_pattern)
                if not pat.fullmatch(text):
                    findings.append(
                        f"Text does not match regex: {regex_pattern}"
                    )
                    details["regex_matched"] = False
                else:
                    details["regex_matched"] = True
            except re.error as exc:
                findings.append(f"Invalid regex pattern: {exc}")
                details["regex_error"] = str(exc)

        passed = len(findings) == 0
        return SafetyCheckResult(
            passed=passed,
            category="output_format",
            severity="low" if passed else "medium",
            findings=findings,
            details=details,
        )

    # ---- jailbreak checks -------------------------------------------------

    def check_jailbreak(self, text: str) -> SafetyCheckResult:
        matched: List[str] = []
        for pat in self.jailbreak_patterns:
            if pat.search(text):
                matched.append(pat.pattern)

        passed = len(matched) == 0
        severity = "critical" if not passed else "low"
        findings = [f"Jailbreak pattern matched: {p}" for p in matched]
        if not passed:
            logger.warning("Jailbreak attempt detected: %d pattern(s)", len(matched))
        return SafetyCheckResult(
            passed=passed,
            category="jailbreak",
            severity=severity,
            findings=findings,
            details={"matched_patterns": matched},
        )

    # ---- aggregate ---------------------------------------------------------

    def run_all(
        self,
        input_text: Optional[str] = None,
        output_text: Optional[str] = None,
        outputs: Optional[List[float]] = None,
    ) -> SafetyReport:
        start = time.monotonic()
        checks: List[SafetyCheckResult] = []

        if input_text is not None:
            checks.append(self.check_input_text(input_text))
            checks.append(self.check_jailbreak(input_text))

        if output_text is not None:
            checks.append(self.check_output_text(output_text))

        if outputs is not None:
            checks.append(self.check_confidence(outputs))

        overall_safe = all(c.passed for c in checks)
        elapsed = (time.monotonic() - start) * 1000

        report = SafetyReport(
            overall_safe=overall_safe,
            checks=checks,
            timestamp=time.time(),
            elapsed_ms=round(elapsed, 3),
        )
        logger.info(
            "Safety report: safe=%s checks=%d elapsed=%.1fms",
            overall_safe,
            len(checks),
            elapsed,
        )
        return report

    # ---- private helpers ---------------------------------------------------

    def _scan_pii(self, text: str) -> List[tuple]:
        hits = []
        for pii_type, pattern in self.pii_patterns.items():
            for match in pattern.finditer(text):
                hits.append((pii_type, match.group()))
        return hits

    def _scan_toxicity(self, text: str) -> List[str]:
        lower = text.lower()
        return [kw for kw in self.toxic_keywords if kw in lower]

    def _scan_prompt_injection(self, text: str) -> List[str]:
        return [p.pattern for p in _PROMPT_INJECTION_PATTERNS if p.search(text)]

    def _escalate(self, current: str, proposed: str) -> str:
        if self._SEVERITY_RANK.get(proposed, 0) > self._SEVERITY_RANK.get(current, 0):
            return proposed
        return current

    @staticmethod
    def _validate_schema(data: Any, schema: Dict[str, Any]) -> List[str]:
        """Minimal schema validation: checks required keys and top-level types."""
        errors: List[str] = []
        expected_type = schema.get("type")
        if expected_type == "object":
            if not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
                return errors
            for key in schema.get("required", []):
                if key not in data:
                    errors.append(f"Missing required key: {key}")
            props = schema.get("properties", {})
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }
            for key, prop_schema in props.items():
                if key in data:
                    prop_type = prop_schema.get("type")
                    expected = type_map.get(prop_type)
                    if expected and not isinstance(data[key], expected):
                        errors.append(
                            f"Key '{key}': expected {prop_type}, "
                            f"got {type(data[key]).__name__}"
                        )
        elif expected_type == "array":
            if not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
        return errors


# ---------------------------------------------------------------------------
# Top-level convenience function (CLI-friendly)
# ---------------------------------------------------------------------------

def check_safety(
    model_path: Optional[str] = None,
    input_text: Optional[str] = None,
    output_text: Optional[str] = None,
) -> SafetyReport:
    """Run all safety checks and return a report.

    Parameters
    ----------
    model_path:
        Optional path to a model (reserved for future model-level checks).
    input_text:
        User-facing input to validate.
    output_text:
        Model-generated output to validate.
    """
    guard = SafetyGuard()
    if model_path:
        logger.info("Model path noted for future checks: %s", model_path)
    return guard.run_all(input_text=input_text, output_text=output_text)
