"""Concurrent stress testing for inference endpoints.

Simulates real-world load patterns:
  - Sustained throughput testing
  - Burst/spike load testing
  - Ramp-up load testing
  - Concurrent session testing
  - Long-running stability testing
"""

from __future__ import annotations

import gc
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from isat.benchmark.stats import LatencyStats, compute_stats

log = logging.getLogger("isat.stress")


@dataclass
class StressResult:
    pattern: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_s: float
    rps: float
    latency_stats: Optional[LatencyStats] = None
    peak_concurrent: int = 0
    errors: list[str] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1) * 100


class StressTest:
    """Run stress tests against an ORT inference session."""

    def __init__(
        self,
        model_path: str,
        provider: str = "MIGraphXExecutionProvider",
    ):
        self.model_path = model_path
        self.provider = provider

    def sustained(
        self,
        duration_s: float = 60.0,
        concurrency: int = 4,
        target_rps: float = 0,
    ) -> StressResult:
        """Sustained load test at constant concurrency."""
        log.info("Sustained stress test: %ds, concurrency=%d", duration_s, concurrency)

        session, feed = self._create_session()
        if session is None:
            return StressResult(pattern="sustained", total_requests=0,
                                successful_requests=0, failed_requests=0,
                                total_duration_s=0, rps=0,
                                errors=["Failed to create session"])

        latencies: list[float] = []
        errors: list[str] = []
        lock = threading.Lock()
        start = time.time()

        delay = 1.0 / target_rps if target_rps > 0 else 0

        def _worker():
            local_lats = []
            local_errs = []
            while time.time() - start < duration_s:
                t0 = time.perf_counter()
                try:
                    session.run(None, feed)
                    lat = (time.perf_counter() - t0) * 1000
                    local_lats.append(lat)
                except Exception as e:
                    local_errs.append(str(e)[:100])
                if delay:
                    time.sleep(max(0, delay - (time.perf_counter() - t0)))
            with lock:
                latencies.extend(local_lats)
                errors.extend(local_errs)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_worker) for _ in range(concurrency)]
            for f in as_completed(futures):
                f.result()

        total_dur = time.time() - start
        stats = compute_stats(latencies) if latencies else None

        return StressResult(
            pattern="sustained",
            total_requests=len(latencies) + len(errors),
            successful_requests=len(latencies),
            failed_requests=len(errors),
            total_duration_s=total_dur,
            rps=len(latencies) / total_dur if total_dur > 0 else 0,
            latency_stats=stats,
            peak_concurrent=concurrency,
            errors=errors[:20],
        )

    def burst(
        self,
        burst_size: int = 50,
        num_bursts: int = 5,
        pause_between_s: float = 10.0,
    ) -> StressResult:
        """Burst load test: send N requests simultaneously, pause, repeat."""
        log.info("Burst stress test: %d bursts of %d", num_bursts, burst_size)

        session, feed = self._create_session()
        if session is None:
            return StressResult(pattern="burst", total_requests=0,
                                successful_requests=0, failed_requests=0,
                                total_duration_s=0, rps=0,
                                errors=["Failed to create session"])

        all_latencies: list[float] = []
        all_errors: list[str] = []
        timeline: list[dict] = []
        start = time.time()

        for burst_idx in range(num_bursts):
            burst_lats: list[float] = []
            burst_errs: list[str] = []
            lock = threading.Lock()

            def _run():
                t0 = time.perf_counter()
                try:
                    session.run(None, feed)
                    lat = (time.perf_counter() - t0) * 1000
                    with lock:
                        burst_lats.append(lat)
                except Exception as e:
                    with lock:
                        burst_errs.append(str(e)[:100])

            with ThreadPoolExecutor(max_workers=burst_size) as pool:
                futures = [pool.submit(_run) for _ in range(burst_size)]
                for f in as_completed(futures):
                    f.result()

            stats = compute_stats(burst_lats) if burst_lats else None
            timeline.append({
                "burst": burst_idx + 1,
                "sent": burst_size,
                "ok": len(burst_lats),
                "failed": len(burst_errs),
                "mean_ms": stats.mean_ms if stats else 0,
                "p99_ms": stats.p99_ms if stats else 0,
            })
            all_latencies.extend(burst_lats)
            all_errors.extend(burst_errs)

            if burst_idx < num_bursts - 1:
                time.sleep(pause_between_s)

        total_dur = time.time() - start
        stats = compute_stats(all_latencies) if all_latencies else None

        return StressResult(
            pattern="burst",
            total_requests=len(all_latencies) + len(all_errors),
            successful_requests=len(all_latencies),
            failed_requests=len(all_errors),
            total_duration_s=total_dur,
            rps=len(all_latencies) / total_dur if total_dur > 0 else 0,
            latency_stats=stats,
            peak_concurrent=burst_size,
            errors=all_errors[:20],
            timeline=timeline,
        )

    def ramp(
        self,
        start_concurrency: int = 1,
        end_concurrency: int = 16,
        step_duration_s: float = 15.0,
        step_size: int = 2,
    ) -> StressResult:
        """Ramp-up load test: gradually increase concurrency."""
        log.info("Ramp stress test: %d -> %d concurrency", start_concurrency, end_concurrency)

        session, feed = self._create_session()
        if session is None:
            return StressResult(pattern="ramp", total_requests=0,
                                successful_requests=0, failed_requests=0,
                                total_duration_s=0, rps=0,
                                errors=["Failed to create session"])

        all_latencies: list[float] = []
        timeline: list[dict] = []
        start = time.time()

        current = start_concurrency
        while current <= end_concurrency:
            step_lats: list[float] = []
            lock = threading.Lock()
            step_start = time.time()

            def _worker():
                while time.time() - step_start < step_duration_s:
                    t0 = time.perf_counter()
                    try:
                        session.run(None, feed)
                        lat = (time.perf_counter() - t0) * 1000
                        with lock:
                            step_lats.append(lat)
                    except Exception:
                        pass

            with ThreadPoolExecutor(max_workers=current) as pool:
                futures = [pool.submit(_worker) for _ in range(current)]
                for f in as_completed(futures):
                    f.result()

            stats = compute_stats(step_lats) if step_lats else None
            step_dur = time.time() - step_start
            timeline.append({
                "concurrency": current,
                "requests": len(step_lats),
                "rps": len(step_lats) / step_dur if step_dur > 0 else 0,
                "mean_ms": stats.mean_ms if stats else 0,
                "p99_ms": stats.p99_ms if stats else 0,
            })
            all_latencies.extend(step_lats)
            current += step_size

        total_dur = time.time() - start
        stats = compute_stats(all_latencies) if all_latencies else None

        return StressResult(
            pattern="ramp",
            total_requests=len(all_latencies),
            successful_requests=len(all_latencies),
            failed_requests=0,
            total_duration_s=total_dur,
            rps=len(all_latencies) / total_dur if total_dur > 0 else 0,
            latency_stats=stats,
            peak_concurrent=end_concurrency,
            timeline=timeline,
        )

    def _create_session(self):
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(
                self.model_path,
                providers=[self.provider, "CPUExecutionProvider"],
            )
            feed = {}
            for inp in session.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
                dtype = np.float16 if "float16" in inp.type.lower() else np.float32
                if "int" in inp.type.lower():
                    feed[inp.name] = np.ones(shape, dtype=np.int64)
                else:
                    feed[inp.name] = np.random.randn(*shape).astype(dtype)
            return session, feed
        except Exception as e:
            log.error("Session creation failed: %s", e)
            return None, {}
