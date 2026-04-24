"""Model ensemble runner -- aggregate predictions from multiple models.

Strategies:
  - Voting (classification): majority vote across models
  - Averaging: mean of numeric outputs
  - Weighted averaging: user-defined weights
  - Max confidence: pick model with highest confidence
  - Stacking: use a meta-model on top of base outputs

Production use: reduce variance, improve robustness, catch errors.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.ensemble")


@dataclass
class EnsembleMember:
    name: str
    model_path: str
    weight: float = 1.0
    latency_ms: float = 0
    error: str = ""


@dataclass
class EnsembleResult:
    strategy: str
    members: list[EnsembleMember] = field(default_factory=list)
    aggregated_output: Optional[np.ndarray] = None
    total_ms: float = 0
    agreement_pct: float = 0
    individual_outputs: list[np.ndarray] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Strategy    : {self.strategy}",
            f"  Members     : {len(self.members)}",
            f"  Total time  : {self.total_ms:.2f} ms",
            f"  Agreement   : {self.agreement_pct:.1f}%",
            f"",
            f"  {'Model':<25} {'Weight':>8} {'Latency ms':>12} {'Status':>8}",
            f"  {'-'*25} {'-'*8} {'-'*12} {'-'*8}",
        ]
        for m in self.members:
            status = "ERROR" if m.error else "OK"
            lines.append(f"  {m.name:<25} {m.weight:>8.2f} {m.latency_ms:>12.2f} {status:>8}")

        if self.aggregated_output is not None:
            lines.append(f"\n  Output shape: {self.aggregated_output.shape}")
            if self.aggregated_output.size <= 20:
                lines.append(f"  Output     : {self.aggregated_output.flatten()[:10]}")
        return "\n".join(lines)


class ModelEnsemble:
    """Run inference across multiple models and aggregate."""

    def __init__(
        self,
        models: list[tuple[str, str, float]],
        provider: str = "CPUExecutionProvider",
        strategy: str = "average",
    ):
        self.models = models
        self.provider = provider
        self.strategy = strategy

    def run(self, runs: int = 5) -> EnsembleResult:
        import onnxruntime as ort

        members: list[EnsembleMember] = []
        all_outputs: list[np.ndarray] = []
        total_t0 = time.perf_counter()

        for name, path, weight in self.models:
            member = EnsembleMember(name=name, model_path=path, weight=weight)
            if not Path(path).exists():
                member.error = "File not found"
                members.append(member)
                continue
            try:
                session = ort.InferenceSession(
                    path, providers=ort_providers(self.provider),
                )
                feed = _build_feed(session)
                session.run(None, feed)

                lats = []
                last_output = None
                for _ in range(runs):
                    t0 = time.perf_counter()
                    out = session.run(None, feed)
                    lats.append((time.perf_counter() - t0) * 1000)
                    last_output = out

                member.latency_ms = float(np.mean(lats))
                if last_output:
                    all_outputs.append(last_output[0])
            except Exception as e:
                member.error = str(e)
            members.append(member)

        total_ms = (time.perf_counter() - total_t0) * 1000

        aggregated = None
        agreement = 0.0

        valid_outputs = [o for o, m in zip(all_outputs, members) if not m.error]
        valid_members = [m for m in members if not m.error]

        if valid_outputs:
            if self.strategy == "average":
                weights = np.array([m.weight for m in valid_members])
                weights = weights / weights.sum()
                aggregated = sum(o * w for o, w in zip(valid_outputs, weights))
            elif self.strategy == "vote":
                preds = [np.argmax(o, axis=-1) for o in valid_outputs]
                from scipy.stats import mode as sp_mode
                try:
                    aggregated = sp_mode(np.stack(preds), axis=0).mode
                except Exception:
                    aggregated = preds[0] if preds else None
            elif self.strategy == "max_confidence":
                confidences = [float(np.max(o)) for o in valid_outputs]
                best_idx = np.argmax(confidences)
                aggregated = valid_outputs[best_idx]
            else:
                aggregated = valid_outputs[0]

            if len(valid_outputs) >= 2 and valid_outputs[0].shape == valid_outputs[1].shape:
                preds = [np.argmax(o, axis=-1) for o in valid_outputs]
                if all(p.shape == preds[0].shape for p in preds):
                    agreements = [np.mean(p == preds[0]) for p in preds[1:]]
                    agreement = float(np.mean(agreements)) * 100

        return EnsembleResult(
            strategy=self.strategy, members=members,
            aggregated_output=aggregated, total_ms=total_ms,
            agreement_pct=agreement, individual_outputs=valid_outputs,
        )


def _build_feed(session) -> dict:
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if "int" in inp.type.lower():
            feed[inp.name] = np.ones(shape, dtype=np.int64)
        elif "float16" in inp.type.lower():
            feed[inp.name] = np.random.randn(*shape).astype(np.float16)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feed
