"""Continuous lightweight profiler for production inference.

Samples a configurable percentage of inference requests and records
latency, memory, GPU stats without impacting throughput. Generates
periodic summaries and detects anomalies.

Like Google's continuous profiling or Datadog's profiler.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("isat.continuous_profiler")


@dataclass
class ProfileSample:
    timestamp: float
    latency_ms: float
    rss_mb: float = 0
    gpu_temp_c: float = 0
    gpu_vram_mb: float = 0


@dataclass
class ProfileWindow:
    window_start: float
    window_end: float
    samples: list[ProfileSample] = field(default_factory=list)
    mean_latency_ms: float = 0
    p50_latency_ms: float = 0
    p95_latency_ms: float = 0
    p99_latency_ms: float = 0
    anomaly_count: int = 0

    def summary(self) -> str:
        duration = self.window_end - self.window_start
        lines = [
            f"  Window        : {duration:.1f}s",
            f"  Samples       : {len(self.samples)}",
            f"  Mean latency  : {self.mean_latency_ms:.2f} ms",
            f"  P50 latency   : {self.p50_latency_ms:.2f} ms",
            f"  P95 latency   : {self.p95_latency_ms:.2f} ms",
            f"  P99 latency   : {self.p99_latency_ms:.2f} ms",
            f"  Anomalies     : {self.anomaly_count}",
        ]
        return "\n".join(lines)


@dataclass
class ContinuousProfile:
    model_path: str
    total_samples: int
    sample_rate: float
    windows: list[ProfileWindow] = field(default_factory=list)
    total_anomalies: int = 0
    baseline_mean_ms: float = 0
    baseline_std_ms: float = 0

    def summary(self) -> str:
        lines = [
            f"  Model         : {self.model_path}",
            f"  Sample rate   : {self.sample_rate:.0%}",
            f"  Total samples : {self.total_samples}",
            f"  Total anomalies: {self.total_anomalies}",
            f"  Baseline      : {self.baseline_mean_ms:.2f} +/- {self.baseline_std_ms:.2f} ms",
            f"",
        ]
        if self.windows:
            lines.append(f"  {'Window':<8} {'Samples':>8} {'Mean ms':>10} {'P95 ms':>10} {'P99 ms':>10} {'Anomalies':>10}")
            lines.append(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            for i, w in enumerate(self.windows):
                lines.append(
                    f"  {i+1:<8} {len(w.samples):>8} "
                    f"{w.mean_latency_ms:>10.2f} {w.p95_latency_ms:>10.2f} "
                    f"{w.p99_latency_ms:>10.2f} {w.anomaly_count:>10}"
                )
        return "\n".join(lines)


class ContinuousProfiler:
    """Lightweight always-on profiler for production inference."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        sample_rate: float = 0.1,
        window_seconds: float = 10.0,
        total_seconds: float = 60.0,
        anomaly_threshold_sigma: float = 3.0,
    ):
        self.model_path = model_path
        self.provider = provider
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.total_seconds = total_seconds
        self.anomaly_threshold = anomaly_threshold_sigma

    def profile(self) -> ContinuousProfile:
        import onnxruntime as ort
        from isat.utils.sysfs import gpu_temp_edge, gpu_vram_used_mb

        session = ort.InferenceSession(
            self.model_path, providers=[self.provider, "CPUExecutionProvider"],
        )
        feed = self._build_feed(session)
        for _ in range(5):
            session.run(None, feed)

        all_samples: list[ProfileSample] = []
        windows: list[ProfileWindow] = []
        total_start = time.perf_counter()
        window_start = total_start
        window_samples: list[ProfileSample] = []

        baseline_lats: list[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            session.run(None, feed)
            baseline_lats.append((time.perf_counter() - t0) * 1000)

        baseline_mean = float(np.mean(baseline_lats))
        baseline_std = float(np.std(baseline_lats))
        anomaly_thresh = baseline_mean + self.anomaly_threshold * max(baseline_std, 0.1)

        total_anomalies = 0
        while time.perf_counter() - total_start < self.total_seconds:
            t0 = time.perf_counter()
            session.run(None, feed)
            latency = (time.perf_counter() - t0) * 1000

            if random.random() < self.sample_rate:
                temp = gpu_temp_edge() or 0
                vram = gpu_vram_used_mb() or 0
                sample = ProfileSample(
                    timestamp=time.perf_counter(),
                    latency_ms=latency,
                    gpu_temp_c=temp, gpu_vram_mb=vram,
                )
                window_samples.append(sample)
                all_samples.append(sample)

                if latency > anomaly_thresh:
                    total_anomalies += 1

            now = time.perf_counter()
            if now - window_start >= self.window_seconds:
                if window_samples:
                    lats = [s.latency_ms for s in window_samples]
                    arr = np.array(lats)
                    anomalies = int(np.sum(arr > anomaly_thresh))
                    windows.append(ProfileWindow(
                        window_start=window_start, window_end=now,
                        samples=window_samples,
                        mean_latency_ms=float(np.mean(arr)),
                        p50_latency_ms=float(np.percentile(arr, 50)),
                        p95_latency_ms=float(np.percentile(arr, 95)),
                        p99_latency_ms=float(np.percentile(arr, 99)),
                        anomaly_count=anomalies,
                    ))
                window_start = now
                window_samples = []

        return ContinuousProfile(
            model_path=self.model_path,
            total_samples=len(all_samples),
            sample_rate=self.sample_rate,
            windows=windows,
            total_anomalies=total_anomalies,
            baseline_mean_ms=baseline_mean,
            baseline_std_ms=baseline_std,
        )

    def _build_feed(self, session) -> dict:
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
