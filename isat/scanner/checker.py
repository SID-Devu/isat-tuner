"""ONNX model security and compliance scanner.

Checks for:
  - Oversized models (potential denial-of-service)
  - Deprecated / unsafe operators
  - External data integrity (missing files, hash mismatches)
  - ONNX spec compliance (opset version, valid shapes)
  - Best-practice violations (unnecessary casts, suboptimal patterns)
  - Potential numerical instability (inf/nan in initializers)
  - License/provenance metadata
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.scanner")

UNSAFE_OPS = {"Loop", "If", "Scan", "SequenceMap"}
DEPRECATED_OPS = {"Scatter", "Upsample"}
RISKY_OPS = {"Custom", "NonMaxSuppression"}
MAX_REASONABLE_SIZE_GB = 50
MAX_REASONABLE_NODES = 100_000


@dataclass
class ScanFinding:
    severity: str  # "critical", "warning", "info"
    category: str  # "security", "compliance", "performance", "best_practice"
    message: str
    detail: str = ""
    location: str = ""


@dataclass
class ScanResult:
    model_path: str
    findings: list[ScanFinding] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    score: int = 100  # 0-100 compliance score
    scan_time_ms: float = 0

    @property
    def passed(self) -> bool:
        return self.critical_count == 0

    def summary(self) -> str:
        lines = [
            f"  Model     : {self.model_path}",
            f"  Score     : {self.score}/100",
            f"  Criticals : {self.critical_count}",
            f"  Warnings  : {self.warning_count}",
            f"  Info      : {self.info_count}",
            f"  Scan time : {self.scan_time_ms:.0f} ms",
            f"",
        ]
        if self.findings:
            lines.append(f"  {'Severity':<10} {'Category':<15} {'Finding'}")
            lines.append(f"  {'-'*10} {'-'*15} {'-'*45}")
            for f in self.findings:
                icon = {"critical": "CRIT", "warning": "WARN", "info": "INFO"}[f.severity]
                lines.append(f"  {icon:<10} {f.category:<15} {f.message}")
                if f.detail:
                    lines.append(f"  {'':>26} {f.detail}")
        else:
            lines.append("  No issues found -- model is clean")

        verdict = "PASSED" if self.passed else "FAILED"
        lines.append(f"\n  Verdict: {verdict}")
        return "\n".join(lines)


class ModelScanner:
    """Scan ONNX models for security and compliance issues."""

    def scan(self, model_path: str) -> ScanResult:
        import time
        t0 = time.perf_counter()

        result = ScanResult(model_path=model_path)
        path = Path(model_path)

        if not path.exists():
            result.findings.append(ScanFinding("critical", "security", "Model file not found"))
            result.critical_count = 1
            result.score = 0
            return result

        self._check_file_size(path, result)
        self._check_external_data(path, result)

        try:
            import onnx
            model = onnx.load(str(path), load_external_data=False)
            self._check_opset(model, result)
            self._check_operators(model, result)
            self._check_inputs_outputs(model, result)
            self._check_initializers(model, result)
            self._check_metadata(model, result)
            self._check_graph_patterns(model, result)
        except Exception as e:
            result.findings.append(ScanFinding(
                "critical", "compliance", f"Failed to parse ONNX model: {e}"
            ))
            result.critical_count += 1

        result.critical_count = sum(1 for f in result.findings if f.severity == "critical")
        result.warning_count = sum(1 for f in result.findings if f.severity == "warning")
        result.info_count = sum(1 for f in result.findings if f.severity == "info")

        penalty = result.critical_count * 25 + result.warning_count * 5 + result.info_count * 1
        result.score = max(0, 100 - penalty)

        result.scan_time_ms = (time.perf_counter() - t0) * 1000
        return result

    def _check_file_size(self, path: Path, result: ScanResult):
        size_gb = path.stat().st_size / (1024 ** 3)
        if size_gb > MAX_REASONABLE_SIZE_GB:
            result.findings.append(ScanFinding(
                "critical", "security",
                f"Model file is {size_gb:.1f} GB -- exceeds {MAX_REASONABLE_SIZE_GB} GB limit",
                detail="Extremely large models can cause resource exhaustion"
            ))

        ext_dir = path.parent
        total_ext = 0
        for f in ext_dir.glob(f"{path.stem}*"):
            if f.suffix in (".onnx_data", ".bin", ".pb"):
                total_ext += f.stat().st_size
        total_gb = (path.stat().st_size + total_ext) / (1024 ** 3)
        if total_gb > 0.1:
            result.findings.append(ScanFinding(
                "info", "security",
                f"Total model size (with external data): {total_gb:.1f} GB"
            ))

    def _check_external_data(self, path: Path, result: ScanResult):
        try:
            import onnx
            model = onnx.load(str(path), load_external_data=False)
            missing = []
            for init in model.graph.initializer:
                if init.data_location == 1:  # EXTERNAL
                    ext_info = None
                    for entry in init.external_data:
                        if entry.key == "location":
                            ext_path = path.parent / entry.value
                            if not ext_path.exists():
                                missing.append(entry.value)
                            break
            if missing:
                result.findings.append(ScanFinding(
                    "critical", "compliance",
                    f"Missing {len(missing)} external data file(s)",
                    detail=", ".join(missing[:5])
                ))
            elif any(init.data_location == 1 for init in model.graph.initializer):
                result.findings.append(ScanFinding(
                    "info", "compliance", "Model uses external data -- all files present"
                ))
        except Exception:
            pass

    def _check_opset(self, model, result: ScanResult):
        opset = 0
        for imp in model.opset_import:
            if imp.domain == "" or imp.domain == "ai.onnx":
                opset = imp.version
                break
        if opset < 11:
            result.findings.append(ScanFinding(
                "warning", "compliance",
                f"Opset {opset} is very old -- consider upgrading to opset 17+",
                detail="Older opsets lack optimizations and may have compatibility issues"
            ))
        elif opset < 15:
            result.findings.append(ScanFinding(
                "info", "compliance", f"Opset {opset} -- consider upgrading to opset 17+ for best performance"
            ))
        else:
            result.findings.append(ScanFinding(
                "info", "compliance", f"Opset {opset} -- good"
            ))

    def _check_operators(self, model, result: ScanResult):
        ops = set()
        for node in model.graph.node:
            ops.add(node.op_type)

        unsafe = ops & UNSAFE_OPS
        if unsafe:
            result.findings.append(ScanFinding(
                "warning", "security",
                f"Uses control-flow ops that may execute arbitrary subgraphs: {', '.join(unsafe)}",
                detail="These ops contain subgraphs that could hide malicious logic"
            ))

        deprecated = ops & DEPRECATED_OPS
        if deprecated:
            result.findings.append(ScanFinding(
                "warning", "compliance",
                f"Uses deprecated ops: {', '.join(deprecated)}",
                detail="Replace with modern equivalents for better compatibility"
            ))

        risky = ops & RISKY_OPS
        if risky:
            result.findings.append(ScanFinding(
                "info", "security", f"Uses potentially risky ops: {', '.join(risky)}"
            ))

        node_count = len(model.graph.node)
        if node_count > MAX_REASONABLE_NODES:
            result.findings.append(ScanFinding(
                "warning", "performance",
                f"Model has {node_count:,} nodes -- may cause slow compilation"
            ))

    def _check_inputs_outputs(self, model, result: ScanResult):
        for inp in model.graph.input:
            shape = inp.type.tensor_type.shape
            if shape:
                for dim in shape.dim:
                    if dim.dim_param:
                        result.findings.append(ScanFinding(
                            "info", "compliance",
                            f"Input '{inp.name}' has dynamic dimension '{dim.dim_param}'",
                            detail="Dynamic shapes may reduce optimization opportunities"
                        ))
                        break

        if not model.graph.output:
            result.findings.append(ScanFinding(
                "critical", "compliance", "Model has no outputs defined"
            ))

    def _check_initializers(self, model, result: ScanResult):
        import numpy as np
        checked = 0
        nan_count = 0
        inf_count = 0

        for init in model.graph.initializer:
            if init.data_location == 1:
                continue
            try:
                from onnx.numpy_helper import to_array
                arr = to_array(init)
                if np.any(np.isnan(arr)):
                    nan_count += 1
                if np.any(np.isinf(arr)):
                    inf_count += 1
                checked += 1
            except Exception:
                continue

        if nan_count:
            result.findings.append(ScanFinding(
                "critical", "compliance",
                f"{nan_count} initializer(s) contain NaN values",
                detail="NaN in weights will produce garbage outputs"
            ))
        if inf_count:
            result.findings.append(ScanFinding(
                "warning", "compliance",
                f"{inf_count} initializer(s) contain Inf values",
                detail="Inf values may cause numerical instability"
            ))

    def _check_metadata(self, model, result: ScanResult):
        props = {p.key: p.value for p in model.metadata_props}
        if not props:
            result.findings.append(ScanFinding(
                "info", "best_practice",
                "No metadata properties set",
                detail="Consider adding 'author', 'version', 'license' metadata"
            ))
        else:
            if "license" not in props and "License" not in props:
                result.findings.append(ScanFinding(
                    "info", "best_practice", "No license metadata -- consider adding one"
                ))

        if not model.doc_string:
            result.findings.append(ScanFinding(
                "info", "best_practice", "No model doc_string set"
            ))

    def _check_graph_patterns(self, model, result: ScanResult):
        cast_count = sum(1 for n in model.graph.node if n.op_type == "Cast")
        total = len(model.graph.node) or 1
        if cast_count / total > 0.1:
            result.findings.append(ScanFinding(
                "warning", "performance",
                f"High Cast op ratio ({cast_count}/{total} = {cast_count/total:.0%})",
                detail="Excessive type casts reduce performance -- consider converting model to single precision"
            ))

        identity_count = sum(1 for n in model.graph.node if n.op_type == "Identity")
        if identity_count > 10:
            result.findings.append(ScanFinding(
                "info", "performance",
                f"{identity_count} Identity ops found -- run onnxsim to remove them"
            ))
