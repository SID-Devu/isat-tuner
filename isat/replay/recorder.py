"""Inference replay/record -- capture and replay production requests.

Record mode: save input tensors + metadata to disk during inference.
Replay mode: load saved requests and re-run for debugging/regression testing.

Used by: Netflix, Uber for production debugging and regression suites.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.replay")


@dataclass
class RecordedRequest:
    request_id: str
    timestamp: float
    inputs: dict[str, np.ndarray]
    outputs: list[np.ndarray] | None = None
    latency_ms: float = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ReplayResult:
    recording_path: str
    total_requests: int
    replayed: int
    errors: int
    mean_latency_ms: float
    p95_latency_ms: float
    output_match_pct: float
    mismatches: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Recording      : {self.recording_path}",
            f"  Total requests : {self.total_requests}",
            f"  Replayed       : {self.replayed}",
            f"  Errors         : {self.errors}",
            f"  Mean latency   : {self.mean_latency_ms:.2f} ms",
            f"  P95 latency    : {self.p95_latency_ms:.2f} ms",
            f"  Output match   : {self.output_match_pct:.1f}%",
        ]
        if self.mismatches:
            lines.append(f"\n  Mismatches ({len(self.mismatches)}):")
            for m in self.mismatches[:10]:
                lines.append(f"    - {m}")
        return "\n".join(lines)


class InferenceRecorder:
    """Record inference requests to disk."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._count = 0
        self._manifest: list[dict] = []

    def record(self, request_id: str, inputs: dict[str, np.ndarray],
               outputs: list[np.ndarray] | None = None, latency_ms: float = 0,
               metadata: dict | None = None):
        req_dir = Path(self.output_dir) / f"req_{self._count:06d}"
        req_dir.mkdir(exist_ok=True)

        for name, arr in inputs.items():
            np.save(str(req_dir / f"input_{name}.npy"), arr)
        if outputs:
            for i, arr in enumerate(outputs):
                np.save(str(req_dir / f"output_{i}.npy"), arr)

        entry = {
            "request_id": request_id, "index": self._count,
            "timestamp": time.time(), "latency_ms": latency_ms,
            "input_names": list(inputs.keys()),
            "output_count": len(outputs) if outputs else 0,
            "metadata": metadata or {},
        }
        self._manifest.append(entry)
        self._count += 1

        with open(Path(self.output_dir) / "manifest.json", "w") as f:
            json.dump(self._manifest, f, indent=2)

    def record_from_model(self, model_path: str, provider: str = "CPUExecutionProvider",
                          num_requests: int = 10) -> int:
        import onnxruntime as ort
        session = ort.InferenceSession(
            model_path, providers=ort_providers(provider),
        )
        for i in range(num_requests):
            feed = _build_feed(session)
            t0 = time.perf_counter()
            outputs = session.run(None, feed)
            latency = (time.perf_counter() - t0) * 1000
            self.record(
                request_id=f"auto_{i:04d}", inputs=feed,
                outputs=outputs, latency_ms=latency,
                metadata={"model": model_path, "provider": provider},
            )
        return self._count


class InferenceReplayer:
    """Replay recorded requests against a model."""

    def __init__(self, recording_dir: str):
        self.recording_dir = recording_dir
        manifest_path = Path(recording_dir) / "manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def replay(self, model_path: str, provider: str = "CPUExecutionProvider",
               tolerance: float = 1e-4) -> ReplayResult:
        import onnxruntime as ort
        session = ort.InferenceSession(
            model_path, providers=ort_providers(provider),
        )

        latencies = []
        errors = 0
        matches = 0
        mismatches = []
        replayed = 0

        for entry in self.manifest:
            req_dir = Path(self.recording_dir) / f"req_{entry['index']:06d}"
            if not req_dir.exists():
                continue

            feed = {}
            for name in entry["input_names"]:
                npy_path = req_dir / f"input_{name}.npy"
                if npy_path.exists():
                    feed[name] = np.load(str(npy_path))

            try:
                t0 = time.perf_counter()
                new_outputs = session.run(None, feed)
                lat = (time.perf_counter() - t0) * 1000
                latencies.append(lat)
                replayed += 1

                if entry.get("output_count", 0) > 0:
                    match = True
                    for oi in range(entry["output_count"]):
                        ref_path = req_dir / f"output_{oi}.npy"
                        if ref_path.exists():
                            ref = np.load(str(ref_path))
                            diff = float(np.max(np.abs(
                                ref.astype(np.float64) - new_outputs[oi].astype(np.float64)
                            )))
                            if diff > tolerance:
                                match = False
                                mismatches.append(
                                    f"req {entry['request_id']} output[{oi}]: max_diff={diff:.6e}"
                                )
                    if match:
                        matches += 1
                else:
                    matches += 1
            except Exception as e:
                errors += 1
                mismatches.append(f"req {entry['request_id']}: {str(e)[:100]}")

        arr = np.array(latencies) if latencies else np.array([0])
        total = len(self.manifest)
        match_pct = matches / max(replayed, 1) * 100

        return ReplayResult(
            recording_path=self.recording_dir,
            total_requests=total, replayed=replayed, errors=errors,
            mean_latency_ms=float(np.mean(arr)),
            p95_latency_ms=float(np.percentile(arr, 95)),
            output_match_pct=match_pct,
            mismatches=mismatches,
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
