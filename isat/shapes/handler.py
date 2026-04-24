"""Dynamic shape handler for models with variable-length inputs.

Most real-world models have dynamic axes (batch, sequence_length, image_size).
This module:
  1. Detects which inputs have dynamic dimensions
  2. Generates a sweep of realistic shapes to benchmark
  3. Benchmarks each shape independently
  4. Reports latency vs shape curves (e.g., latency vs seq_len)
  5. Finds the optimal shape for pinning (if pinning helps)
  6. Detects shape-dependent performance cliffs
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.shapes")


@dataclass
class ShapeResult:
    shape_desc: str
    shape: dict[str, list[int]]
    mean_ms: float
    p95_ms: float
    throughput_fps: float
    peak_memory_mb: float = 0.0


@dataclass
class ShapeProfile:
    model_path: str
    dynamic_dims: dict[str, list[str]]
    results: list[ShapeResult] = field(default_factory=list)
    optimal_shape: Optional[ShapeResult] = None
    performance_cliff: Optional[dict] = None

    def summary(self) -> str:
        lines = [
            f"  Dynamic dimensions detected:",
        ]
        for inp_name, dims in self.dynamic_dims.items():
            lines.append(f"    {inp_name}: {dims}")

        if self.results:
            lines.append(f"\n  {'Shape':<35} {'Mean ms':>10} {'P95 ms':>10} {'FPS':>10}")
            lines.append(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
            for r in self.results:
                lines.append(f"  {r.shape_desc:<35} {r.mean_ms:>10.2f} {r.p95_ms:>10.2f} {r.throughput_fps:>10.1f}")

        if self.optimal_shape:
            lines.append(f"\n  Optimal shape: {self.optimal_shape.shape_desc} ({self.optimal_shape.mean_ms:.2f} ms)")

        if self.performance_cliff:
            lines.append(f"\n  Performance cliff detected at {self.performance_cliff['shape']}:")
            lines.append(f"    Latency jumped {self.performance_cliff['jump_pct']:.0f}% "
                         f"({self.performance_cliff['before_ms']:.1f} -> {self.performance_cliff['after_ms']:.1f} ms)")

        return "\n".join(lines)


class DynamicShapeHandler:
    """Benchmark models across different input shapes."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        warmup: int = 3,
        runs: int = 10,
    ):
        self.model_path = model_path
        self.provider = provider
        self.warmup = warmup
        self.runs = runs

    def detect_dynamic_dims(self) -> dict[str, list[str]]:
        """Detect which input dimensions are dynamic."""
        import onnx
        model = onnx.load(self.model_path, load_external_data=False)

        init_names = {i.name for i in model.graph.initializer}
        dynamic: dict[str, list[str]] = {}

        for inp in model.graph.input:
            if inp.name in init_names:
                continue
            dyn_dims = []
            for i, dim in enumerate(inp.type.tensor_type.shape.dim):
                if dim.dim_param:
                    dyn_dims.append(f"dim{i}={dim.dim_param}")
                elif dim.dim_value <= 0:
                    dyn_dims.append(f"dim{i}=dynamic")
            if dyn_dims:
                dynamic[inp.name] = dyn_dims

        return dynamic

    def sweep(
        self,
        shape_overrides: dict[str, list[list[int]]] | None = None,
    ) -> ShapeProfile:
        """Benchmark across a sweep of input shapes."""
        import onnxruntime as ort

        dynamic_dims = self.detect_dynamic_dims()

        if shape_overrides:
            shape_configs = self._build_from_overrides(shape_overrides)
        else:
            shape_configs = self._auto_generate_shapes()

        results: list[ShapeResult] = []

        for desc, shape_dict in shape_configs:
            try:
                session = ort.InferenceSession(
                    self.model_path,
                    providers=ort_providers(self.provider),
                )
                feed = self._build_feed(session, shape_dict)

                for _ in range(self.warmup):
                    session.run(None, feed)

                latencies = []
                for _ in range(self.runs):
                    t0 = time.perf_counter()
                    session.run(None, feed)
                    latencies.append((time.perf_counter() - t0) * 1000)

                lats = np.array(latencies)
                results.append(ShapeResult(
                    shape_desc=desc,
                    shape=shape_dict,
                    mean_ms=float(np.mean(lats)),
                    p95_ms=float(np.percentile(lats, 95)),
                    throughput_fps=1000.0 / float(np.mean(lats)),
                ))
            except Exception as e:
                log.warning("Shape %s failed: %s", desc, e)

        optimal = min(results, key=lambda r: r.mean_ms) if results else None

        cliff = None
        for i in range(1, len(results)):
            prev, curr = results[i - 1].mean_ms, results[i].mean_ms
            if prev > 0 and (curr - prev) / prev > 0.5:
                cliff = {
                    "shape": results[i].shape_desc,
                    "before_ms": prev,
                    "after_ms": curr,
                    "jump_pct": (curr - prev) / prev * 100,
                }
                break

        return ShapeProfile(
            model_path=self.model_path,
            dynamic_dims=dynamic_dims,
            results=results,
            optimal_shape=optimal,
            performance_cliff=cliff,
        )

    def _auto_generate_shapes(self) -> list[tuple[str, dict[str, list[int]]]]:
        """Auto-generate shapes for common patterns (batch, seq_len, image_size)."""
        import onnx
        model = onnx.load(self.model_path, load_external_data=False)
        init_names = {i.name for i in model.graph.initializer}

        configs: list[tuple[str, dict[str, list[int]]]] = []
        batch_sizes = [1, 2, 4, 8]

        for bs in batch_sizes:
            shape_dict: dict[str, list[int]] = {}
            for inp in model.graph.input:
                if inp.name in init_names:
                    continue
                dims = []
                for dim in inp.type.tensor_type.shape.dim:
                    if dim.dim_param or dim.dim_value <= 0:
                        dims.append(bs if not dims else max(dim.dim_value, 1))
                    else:
                        dims.append(dim.dim_value)
                shape_dict[inp.name] = dims
            configs.append((f"batch={bs}", shape_dict))

        return configs

    def _build_from_overrides(self, overrides: dict) -> list[tuple[str, dict]]:
        configs = []
        keys = list(overrides.keys())
        first_key = keys[0]
        for shape_variant in overrides[first_key]:
            desc = f"{first_key}={'x'.join(map(str, shape_variant))}"
            configs.append((desc, {first_key: shape_variant}))
        return configs

    def _build_feed(self, session, shape_overrides: dict) -> dict:
        feed = {}
        for inp in session.get_inputs():
            if inp.name in shape_overrides:
                shape = shape_overrides[inp.name]
            else:
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]

            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            elif "float16" in inp.type.lower():
                feed[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        return feed
