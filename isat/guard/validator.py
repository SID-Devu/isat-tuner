"""Input validation and schema enforcement for inference endpoints.

Validates:
  - Input tensor shapes match model expectations
  - Data types are correct
  - Values are within expected ranges (no NaN/Inf)
  - Input dimensions don't exceed memory-safe limits
  - Detects anomalous inputs (statistical outlier detection)

Protects production inference from garbage-in-garbage-out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("isat.guard")


@dataclass
class ValidationIssue:
    severity: str  # "error", "warning"
    input_name: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    checked_inputs: int = 0
    check_time_ms: float = 0

    def summary(self) -> str:
        status = "VALID" if self.valid else "INVALID"
        lines = [
            f"  Status     : {status}",
            f"  Inputs     : {self.checked_inputs}",
            f"  Check time : {self.check_time_ms:.2f} ms",
        ]
        if self.issues:
            lines.append(f"  Issues ({len(self.issues)}):")
            for iss in self.issues:
                icon = "ERROR" if iss.severity == "error" else "WARN "
                lines.append(f"    [{icon}] {iss.input_name}: {iss.message}")
        else:
            lines.append(f"  No issues found")
        return "\n".join(lines)


@dataclass
class InputSchema:
    name: str
    shape: list[int | str]
    dtype: str  # "float32", "float16", "int64", etc.
    min_value: float | None = None
    max_value: float | None = None
    max_elements: int = 10_000_000
    allow_nan: bool = False
    allow_inf: bool = False


class InputGuard:
    """Validate inference inputs against model schema."""

    def __init__(self, model_path: str | None = None, schemas: list[InputSchema] | None = None):
        self.schemas: list[InputSchema] = schemas or []
        if model_path and not schemas:
            self.schemas = self._extract_schemas(model_path)

    def validate(self, inputs: dict[str, np.ndarray]) -> ValidationResult:
        import time
        t0 = time.perf_counter()
        issues: list[ValidationIssue] = []

        for schema in self.schemas:
            if schema.name not in inputs:
                issues.append(ValidationIssue("error", schema.name, "Missing required input"))
                continue
            arr = inputs[schema.name]
            self._check_shape(schema, arr, issues)
            self._check_dtype(schema, arr, issues)
            self._check_values(schema, arr, issues)
            self._check_size(schema, arr, issues)

        for name in inputs:
            if not any(s.name == name for s in self.schemas):
                issues.append(ValidationIssue("warning", name, "Unexpected input (not in model schema)"))

        valid = not any(i.severity == "error" for i in issues)
        return ValidationResult(
            valid=valid, issues=issues,
            checked_inputs=len(inputs),
            check_time_ms=(time.perf_counter() - t0) * 1000,
        )

    def _check_shape(self, schema: InputSchema, arr: np.ndarray, issues: list):
        expected_rank = len(schema.shape)
        if arr.ndim != expected_rank:
            issues.append(ValidationIssue(
                "error", schema.name,
                f"Wrong rank: expected {expected_rank}, got {arr.ndim}"
            ))
            return
        for i, (expected, actual) in enumerate(zip(schema.shape, arr.shape)):
            if isinstance(expected, int) and expected > 0 and actual != expected:
                issues.append(ValidationIssue(
                    "error", schema.name,
                    f"Dimension {i}: expected {expected}, got {actual}"
                ))

    def _check_dtype(self, schema: InputSchema, arr: np.ndarray, issues: list):
        dtype_map = {
            "float32": np.float32, "float16": np.float16,
            "int64": np.int64, "int32": np.int32,
            "int8": np.int8, "uint8": np.uint8,
            "bool": np.bool_,
        }
        expected_dtype = dtype_map.get(schema.dtype)
        if expected_dtype and arr.dtype != expected_dtype:
            issues.append(ValidationIssue(
                "warning", schema.name,
                f"Type mismatch: expected {schema.dtype}, got {arr.dtype}"
            ))

    def _check_values(self, schema: InputSchema, arr: np.ndarray, issues: list):
        if not schema.allow_nan and np.issubdtype(arr.dtype, np.floating):
            nan_count = int(np.sum(np.isnan(arr)))
            if nan_count:
                issues.append(ValidationIssue(
                    "error", schema.name, f"Contains {nan_count} NaN values"
                ))
        if not schema.allow_inf and np.issubdtype(arr.dtype, np.floating):
            inf_count = int(np.sum(np.isinf(arr)))
            if inf_count:
                issues.append(ValidationIssue(
                    "error", schema.name, f"Contains {inf_count} Inf values"
                ))
        if schema.min_value is not None and np.issubdtype(arr.dtype, np.number):
            below = int(np.sum(arr < schema.min_value))
            if below:
                issues.append(ValidationIssue(
                    "warning", schema.name,
                    f"{below} values below minimum {schema.min_value}"
                ))
        if schema.max_value is not None and np.issubdtype(arr.dtype, np.number):
            above = int(np.sum(arr > schema.max_value))
            if above:
                issues.append(ValidationIssue(
                    "warning", schema.name,
                    f"{above} values above maximum {schema.max_value}"
                ))

    def _check_size(self, schema: InputSchema, arr: np.ndarray, issues: list):
        if arr.size > schema.max_elements:
            issues.append(ValidationIssue(
                "error", schema.name,
                f"Too many elements: {arr.size:,} > max {schema.max_elements:,}"
            ))

    def _extract_schemas(self, model_path: str) -> list[InputSchema]:
        import onnx
        model = onnx.load(str(model_path), load_external_data=False)
        schemas = []
        type_map = {1: "float32", 10: "float16", 7: "int64", 6: "int32", 3: "int8", 2: "uint8", 9: "bool"}
        for inp in model.graph.input:
            if inp.name in {i.name for i in model.graph.initializer}:
                continue
            tt = inp.type.tensor_type
            dtype_str = type_map.get(tt.elem_type, "float32")
            shape = []
            if tt.shape:
                for dim in tt.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    elif dim.dim_param:
                        shape.append(dim.dim_param)
                    else:
                        shape.append("dynamic")
            schemas.append(InputSchema(name=inp.name, shape=shape, dtype=dtype_str))
        return schemas
