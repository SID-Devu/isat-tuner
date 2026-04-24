"""Benchmark runner -- execute ORT inference sessions with ISAT search configs."""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Optional

import numpy as np

from isat.benchmark.stats import LatencyStats, compute_stats
from isat.benchmark.thermal import ThermalGuard
from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint
from isat.search.engine import CandidateConfig, TuneResult
from isat.utils import ort_providers

log = logging.getLogger("isat.benchmark")


class BenchmarkRunner:
    """Run ORT inference benchmarks for each candidate configuration."""

    def __init__(
        self,
        hw: HardwareFingerprint,
        model_fp: ModelFingerprint,
        model_path: str,
        *,
        warmup: int = 3,
        runs: int = 5,
        cooldown: float = 60.0,
        provider: str = "MIGraphXExecutionProvider",
    ):
        self.hw = hw
        self.model_fp = model_fp
        self.model_path = model_path
        self.warmup = warmup
        self.runs = runs
        self.cooldown = cooldown
        self.provider = provider
        self.thermal = ThermalGuard(cooldown_seconds=cooldown)

    def run_single(self, config: CandidateConfig) -> TuneResult:
        """Benchmark a single candidate configuration."""
        result = TuneResult(
            config=config,
            warmup_runs=self.warmup,
            measured_runs=self.runs,
            cooldown_s=self.cooldown,
        )

        env_backup = {}
        for key, val in config.merged_env.items():
            env_backup[key] = os.environ.get(key)
            os.environ[key] = val
            log.info("  env: %s=%s", key, val)

        try:
            effective_path = config.effective_model_path or self.model_path
            session, feed = self._create_session(effective_path, config)
            if session is None:
                result.error = "Failed to create ORT session"
                return result

            self.thermal.reset_peaks()

            log.info("  warmup: %d runs", self.warmup)
            for i in range(self.warmup):
                session.run(None, feed)
                self.thermal.sample()

            log.info("  measuring: %d runs", self.runs)
            latencies: list[float] = []
            for i in range(self.runs):
                self.thermal.sample()

                start = time.perf_counter()
                session.run(None, feed)
                end = time.perf_counter()

                lat_ms = (end - start) * 1000
                latencies.append(lat_ms)
                self.thermal.sample()
                log.info("    run %d/%d: %.2f ms", i + 1, self.runs, lat_ms)

            result.latencies = latencies
            stats = compute_stats(latencies)
            result.mean_latency_ms = stats.mean_ms
            result.p50_latency_ms = stats.p50_ms
            result.p95_latency_ms = stats.p95_ms
            result.p99_latency_ms = stats.p99_ms
            result.min_latency_ms = stats.min_ms
            result.max_latency_ms = stats.max_ms
            result.std_dev_ms = stats.std_ms
            result.throughput_fps = 1000.0 / stats.mean_ms if stats.mean_ms > 0 else 0.0

            result.peak_gpu_temp_c = self.thermal.peak_temp
            result.peak_power_w = self.thermal.peak_power
            result.peak_vram_mb = self.thermal.peak_vram
            result.peak_gtt_mb = self.thermal.peak_gtt

        except Exception as e:
            result.error = str(e)
            log.error("  benchmark failed: %s", e)
        finally:
            for key, orig in env_backup.items():
                if orig is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = orig

            del session
            gc.collect()

        return result

    def run_all(self, candidates: list[CandidateConfig]) -> list[TuneResult]:
        """Benchmark all candidates with cooldown between each."""
        results: list[TuneResult] = []
        total = len(candidates)

        for idx, config in enumerate(candidates, 1):
            log.info("=" * 60)
            log.info("Config %d/%d: %s", idx, total, config.label)
            log.info("=" * 60)

            result = self.run_single(config)
            results.append(result)

            if result.error:
                log.warning("  FAILED: %s", result.error)
            else:
                log.info("  Result: mean=%.2f ms  p95=%.2f ms  throughput=%.1f fps",
                         result.mean_latency_ms, result.p95_latency_ms, result.throughput_fps)

            if idx < total:
                self.thermal.wait_cooldown()

        results.sort(key=lambda r: r.mean_latency_ms)
        return results

    def _create_session(
        self,
        model_path: str,
        config: CandidateConfig,
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Create an ORT InferenceSession with the specified provider and build inputs."""
        try:
            import onnxruntime as ort
        except ImportError:
            log.error("onnxruntime not installed")
            return None, {}

        try:
            providers = ort_providers(self.provider)
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel(config.graph.ort_opt_level)

            session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

            feed = {}
            for inp in session.get_inputs():
                shape = []
                for d in inp.shape:
                    shape.append(d if isinstance(d, int) and d > 0 else 1)
                dtype = np.float32
                if "int" in inp.type.lower():
                    dtype = np.int64
                elif "float16" in inp.type.lower():
                    dtype = np.float16
                feed[inp.name] = np.random.randn(*shape).astype(dtype) if dtype != np.int64 else np.ones(shape, dtype=dtype)

            return session, feed

        except Exception as e:
            log.error("Session creation failed: %s", e)
            return None, {}
