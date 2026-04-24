"""Multi-model pipeline optimizer.

Optimizes inference pipelines consisting of multiple ONNX models chained
together (e.g., tokenizer -> encoder -> decoder -> post-processor).
Finds bottlenecks and suggests optimizations for the overall pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.pipeline")


@dataclass
class StageResult:
    name: str
    model_path: str
    mean_ms: float
    p95_ms: float
    p99_ms: float
    memory_mb: float
    pct_of_pipeline: float
    is_bottleneck: bool = False


@dataclass
class PipelineProfile:
    stages: list[StageResult] = field(default_factory=list)
    total_mean_ms: float = 0
    total_p95_ms: float = 0
    bottleneck_stage: str = ""
    optimization_suggestions: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Pipeline stages: {len(self.stages)}",
            f"  Total latency  : {self.total_mean_ms:.2f} ms (P95: {self.total_p95_ms:.2f} ms)",
            f"  Bottleneck     : {self.bottleneck_stage}",
            f"",
            f"  {'Stage':<25} {'Model':<30} {'Mean ms':>10} {'P95 ms':>10} {'% Pipeline':>10} {'Bottleneck':>10}",
            f"  {'-'*25} {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}",
        ]
        for s in self.stages:
            bn = " <<<" if s.is_bottleneck else ""
            lines.append(
                f"  {s.name:<25} {Path(s.model_path).stem[:30]:<30} "
                f"{s.mean_ms:>10.2f} {s.p95_ms:>10.2f} {s.pct_of_pipeline:>9.1f}% {bn}"
            )

        if self.optimization_suggestions:
            lines.append(f"\n  Optimization suggestions:")
            for i, s in enumerate(self.optimization_suggestions, 1):
                lines.append(f"    {i}. {s}")

        return "\n".join(lines)


class PipelineOptimizer:
    """Profile and optimize multi-model inference pipelines."""

    def __init__(self, stages: list[tuple[str, str]],
                 provider: str = "CPUExecutionProvider", runs: int = 20):
        self.stages = stages
        self.provider = provider
        self.runs = runs

    def profile(self) -> PipelineProfile:
        import onnxruntime as ort

        results = []

        for name, model_path in self.stages:
            if not Path(model_path).exists():
                results.append(StageResult(
                    name=name, model_path=model_path,
                    mean_ms=0, p95_ms=0, p99_ms=0, memory_mb=0,
                    pct_of_pipeline=0, is_bottleneck=False,
                ))
                continue

            session = ort.InferenceSession(
                model_path,
                providers=ort_providers(self.provider),
            )
            feed = _build_feed(session)

            for _ in range(3):
                session.run(None, feed)

            latencies = []
            for _ in range(self.runs):
                t0 = time.perf_counter()
                session.run(None, feed)
                latencies.append((time.perf_counter() - t0) * 1000)

            arr = np.array(latencies)
            results.append(StageResult(
                name=name, model_path=model_path,
                mean_ms=float(np.mean(arr)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                memory_mb=Path(model_path).stat().st_size / (1024 * 1024),
                pct_of_pipeline=0,
            ))

        total_mean = sum(s.mean_ms for s in results)
        total_p95 = sum(s.p95_ms for s in results)

        if total_mean > 0:
            for s in results:
                s.pct_of_pipeline = s.mean_ms / total_mean * 100

        bottleneck = max(results, key=lambda s: s.mean_ms) if results else None
        if bottleneck:
            bottleneck.is_bottleneck = True

        suggestions = self._generate_suggestions(results, bottleneck)

        return PipelineProfile(
            stages=results,
            total_mean_ms=total_mean,
            total_p95_ms=total_p95,
            bottleneck_stage=bottleneck.name if bottleneck else "",
            optimization_suggestions=suggestions,
        )

    def _generate_suggestions(self, results: list[StageResult],
                              bottleneck: Optional[StageResult]) -> list[str]:
        suggestions = []

        if bottleneck and bottleneck.pct_of_pipeline > 60:
            suggestions.append(
                f"'{bottleneck.name}' dominates pipeline ({bottleneck.pct_of_pipeline:.0f}%) "
                f"-- focus optimization here first"
            )

        if bottleneck and bottleneck.mean_ms > 100:
            suggestions.append(
                f"Consider FP16/INT8 quantization for '{bottleneck.name}' "
                f"(run: isat optimize {bottleneck.model_path} --fp16)"
            )

        if len(results) >= 3:
            suggestions.append(
                "Consider fusing adjacent stages into a single model to reduce data transfer overhead"
            )

        for s in results:
            if s.mean_ms < 1.0 and s.memory_mb > 100:
                suggestions.append(
                    f"'{s.name}' is fast but large -- consider model compression"
                )

        fast_stages = [s for s in results if s.pct_of_pipeline < 5]
        if fast_stages:
            names = ", ".join(s.name for s in fast_stages)
            suggestions.append(f"Stages [{names}] are negligible (<5% each) -- already optimal")

        return suggestions


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
