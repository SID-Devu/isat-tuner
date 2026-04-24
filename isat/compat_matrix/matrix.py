"""Operator compatibility matrix across ONNX Runtime execution providers.

Shows which operators are supported by which providers, helping users
understand what will work on their hardware before running inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("isat.compat_matrix")

PROVIDER_OP_SUPPORT = {
    "CPUExecutionProvider": {
        "supported_all": True,
        "notes": "Supports all ONNX ops via reference implementation",
        "fp16": False,
        "int8": True,
        "bf16": False,
    },
    "MIGraphXExecutionProvider": {
        "supported_all": False,
        "unsupported": {"NonMaxSuppression", "StringNormalizer", "TfIdfVectorizer",
                       "SequenceMap", "BitShift"},
        "notes": "Most vision/language ops supported via MLIR or rocBLAS",
        "fp16": True,
        "int8": True,
        "bf16": False,
    },
    "CUDAExecutionProvider": {
        "supported_all": False,
        "unsupported": {"StringNormalizer", "TfIdfVectorizer"},
        "notes": "Broad support via cuDNN and cuBLAS",
        "fp16": True,
        "int8": True,
        "bf16": True,
    },
    "TensorrtExecutionProvider": {
        "supported_all": False,
        "unsupported": {"Loop", "If", "Scan", "NonMaxSuppression", "Resize",
                       "StringNormalizer", "TfIdfVectorizer", "SequenceMap"},
        "notes": "Best for static-shape models; control flow ops fall back to CUDA/CPU",
        "fp16": True,
        "int8": True,
        "bf16": True,
    },
    "OpenVINOExecutionProvider": {
        "supported_all": False,
        "unsupported": {"Loop", "Scan", "SequenceMap", "BitShift"},
        "notes": "Optimized for Intel CPUs/iGPUs/VPUs",
        "fp16": True,
        "int8": True,
        "bf16": False,
    },
    "QNNExecutionProvider": {
        "supported_all": False,
        "unsupported": {"Loop", "If", "Scan", "NonMaxSuppression", "SequenceMap",
                       "StringNormalizer", "Resize"},
        "notes": "Qualcomm NPU -- limited op set but very power efficient",
        "fp16": True,
        "int8": True,
        "bf16": False,
    },
}


@dataclass
class CompatResult:
    model_path: str
    model_ops: set[str] = field(default_factory=set)
    provider_results: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Model: {self.model_path}",
            f"  Unique operators: {len(self.model_ops)}",
            f"",
            f"  {'Provider':<35} {'Support':>10} {'Unsupported Ops':<30} {'FP16':>5} {'INT8':>5}",
            f"  {'-'*35} {'-'*10} {'-'*30} {'-'*5} {'-'*5}",
        ]
        for prov, info in self.provider_results.items():
            pct = f"{info['supported_pct']:.0f}%"
            unsup = ", ".join(sorted(info["unsupported_ops"])[:3])
            if len(info["unsupported_ops"]) > 3:
                unsup += f" (+{len(info['unsupported_ops'])-3})"
            fp16 = "yes" if info["fp16"] else "no"
            int8 = "yes" if info["int8"] else "no"
            lines.append(f"  {prov:<35} {pct:>10} {unsup:<30} {fp16:>5} {int8:>5}")

        lines.append(f"\n  Best provider: {self._best_provider()}")
        return "\n".join(lines)

    def _best_provider(self) -> str:
        best = ""
        best_pct = 0
        for prov, info in self.provider_results.items():
            if info["supported_pct"] > best_pct:
                best_pct = info["supported_pct"]
                best = prov
        return best


class CompatMatrix:
    """Check operator compatibility across providers."""

    def check(self, model_path: str) -> CompatResult:
        import onnx
        model = onnx.load(str(model_path), load_external_data=False)
        ops = {node.op_type for node in model.graph.node}

        result = CompatResult(model_path=model_path, model_ops=ops)

        for prov, info in PROVIDER_OP_SUPPORT.items():
            if info.get("supported_all"):
                unsupported = set()
            else:
                unsupported = ops & info.get("unsupported", set())

            supported_pct = (len(ops) - len(unsupported)) / len(ops) * 100 if ops else 100

            result.provider_results[prov] = {
                "supported_pct": supported_pct,
                "unsupported_ops": unsupported,
                "fp16": info.get("fp16", False),
                "int8": info.get("int8", False),
                "notes": info.get("notes", ""),
            }

        return result
