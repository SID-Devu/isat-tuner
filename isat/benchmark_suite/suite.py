"""Full benchmark suite for ONNX models: latency, throughput, memory, scalability."""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("isat.benchmark_suite")

_DIM_DEFAULTS: Dict[str, int] = {
    "batch_size": 1, "batch": 1,
    "num_channels": 3, "channels": 3, "channel": 3,
    "height": 224, "width": 224,
    "sequence_length": 16, "seq_len": 16, "seq_length": 16, "length": 16,
    "num_heads": 12, "head_dim": 64,
    "image_batch_size": 1, "text_batch_size": 1,
    "past_sequence_length": 0,
}

_ONNX_DTYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
    "tensor(uint8)": np.uint8,
    "tensor(double)": np.float64,
}


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


@dataclass
class LatencyReport:
    per_batch_size: Dict[int, LatencyStats] = field(default_factory=dict)


@dataclass
class ThroughputReport:
    batch_size: int = 1
    num_threads: int = 1
    duration_s: float = 0.0
    total_inferences: int = 0
    inferences_per_sec: float = 0.0
    mean_latency_ms: float = 0.0


@dataclass
class MemoryReport:
    per_batch_size: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ScalabilityReport:
    per_thread_count: Dict[int, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    model_path: str = ""
    model_size_mb: float = 0.0
    provider: str = ""
    timestamp: str = ""
    latency: Optional[LatencyReport] = None
    throughput: Optional[ThroughputReport] = None
    memory: Optional[MemoryReport] = None
    scalability: Optional[ScalabilityReport] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_dim(dim) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim
    if isinstance(dim, str):
        if dim in _DIM_DEFAULTS:
            return _DIM_DEFAULTS[dim]
        if "+" in dim:
            parts = [p.strip() for p in dim.split("+")]
            return max(sum(_DIM_DEFAULTS.get(p, 1) for p in parts), 1)
    return 1


def _build_random_inputs(session, batch_size: int) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = []
        for i, dim in enumerate(inp.shape):
            if i == 0 and (dim == "batch_size" or dim == "batch"
                           or (isinstance(dim, str) and "batch" in dim.lower())):
                shape.append(batch_size)
            elif i == 0 and (not isinstance(dim, int) or dim <= 0):
                shape.append(batch_size)
            else:
                shape.append(_resolve_dim(dim))
        np_dtype = _ONNX_DTYPE_MAP.get(inp.type, np.float32)
        if np.issubdtype(np_dtype, np.integer):
            inputs[inp.name] = np.random.randint(0, 100, size=shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            inputs[inp.name] = np.ones(shape, dtype=np.bool_)
        else:
            inputs[inp.name] = np.random.randn(*shape).astype(np_dtype)
    return inputs


def _get_rss_mb() -> float:
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        return rusage.ru_maxrss / 1024  # Linux reports KB
    except Exception:
        return 0.0


def _get_gpu_vram_mb() -> float:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 * 1024)
    except Exception:
        return 0.0


def _latency_stats(timings_ms: List[float]) -> LatencyStats:
    timings_ms.sort()
    n = len(timings_ms)
    return LatencyStats(
        mean_ms=round(statistics.mean(timings_ms), 4),
        p50_ms=round(timings_ms[n // 2], 4),
        p95_ms=round(timings_ms[int(n * 0.95)], 4),
        p99_ms=round(timings_ms[int(n * 0.99)], 4),
        min_ms=round(timings_ms[0], 4),
        max_ms=round(timings_ms[-1], 4),
    )


def _create_session(model_path: str, provider: str):
    import onnxruntime as ort

    if provider == "auto":
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    else:
        providers = [provider]

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


def _model_size_mb(model_path: str) -> float:
    p = Path(model_path)
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
    return 0.0


# ---------------------------------------------------------------------------
# BenchmarkSuite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    def __init__(
        self,
        model_path: str,
        provider: str = "auto",
        output_dir: Optional[str] = None,
    ):
        self.model_path = str(Path(model_path).resolve())
        self.provider = provider
        self.output_dir = output_dir
        self._session = _create_session(self.model_path, self.provider)
        self._active_providers = self._session.get_providers()
        self._model_size = _model_size_mb(self.model_path)
        log.info(
            "BenchmarkSuite initialized: %s (%.1f MB), providers=%s",
            self.model_path, self._model_size, self._active_providers,
        )

    # -- Latency --------------------------------------------------------

    def run_latency(
        self,
        batch_sizes: Optional[List[int]] = None,
        warmup: int = 5,
        iterations: int = 50,
    ) -> LatencyReport:
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        report = LatencyReport()
        for bs in batch_sizes:
            log.info("Latency benchmark: batch_size=%d", bs)
            try:
                inputs = _build_random_inputs(self._session, bs)
                out_names = [o.name for o in self._session.get_outputs()]

                for _ in range(warmup):
                    self._session.run(out_names, inputs)

                timings: List[float] = []
                for _ in range(iterations):
                    t0 = time.perf_counter()
                    self._session.run(out_names, inputs)
                    timings.append((time.perf_counter() - t0) * 1000)

                report.per_batch_size[bs] = _latency_stats(timings)
            except (MemoryError, RuntimeError) as exc:
                log.warning("OOM at batch_size=%d: %s", bs, exc)
                report.per_batch_size[bs] = LatencyStats(
                    mean_ms=-1, p50_ms=-1, p95_ms=-1, p99_ms=-1, min_ms=-1, max_ms=-1,
                )
                break

        return report

    # -- Throughput ------------------------------------------------------

    def run_throughput(
        self,
        duration_s: float = 30,
        batch_size: int = 1,
        num_threads: int = 1,
    ) -> ThroughputReport:
        log.info(
            "Throughput benchmark: duration=%ds, batch_size=%d, threads=%d",
            duration_s, batch_size, num_threads,
        )

        stop = Event()
        inputs = _build_random_inputs(self._session, batch_size)
        out_names = [o.name for o in self._session.get_outputs()]
        all_latencies: List[float] = []

        def _worker() -> List[float]:
            session = _create_session(self.model_path, self.provider)
            latencies: List[float] = []
            while not stop.is_set():
                t0 = time.perf_counter()
                try:
                    session.run(out_names, inputs)
                except (MemoryError, RuntimeError):
                    break
                latencies.append((time.perf_counter() - t0) * 1000)
            return latencies

        wall_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_worker) for _ in range(num_threads)]
            time.sleep(duration_s)
            stop.set()
            for fut in as_completed(futures):
                all_latencies.extend(fut.result())
        wall_elapsed = time.perf_counter() - wall_start

        total = len(all_latencies)
        return ThroughputReport(
            batch_size=batch_size,
            num_threads=num_threads,
            duration_s=round(wall_elapsed, 2),
            total_inferences=total,
            inferences_per_sec=round(total / wall_elapsed, 2) if wall_elapsed > 0 else 0,
            mean_latency_ms=round(statistics.mean(all_latencies), 4) if all_latencies else 0,
        )

    # -- Accuracy --------------------------------------------------------

    def run_accuracy(
        self,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        num_samples: int = 1000,
    ) -> Dict[str, Any]:
        if not dataset_path and not dataset_name:
            log.info("No dataset provided — skipping accuracy benchmark")
            return {"skipped": True, "reason": "No dataset_path or dataset_name provided."}

        log.info(
            "Accuracy benchmark: dataset_path=%s, dataset_name=%s, num_samples=%d",
            dataset_path, dataset_name, num_samples,
        )

        source_path = dataset_path or dataset_name
        data_path = Path(source_path)  # type: ignore[arg-type]
        if not data_path.exists():
            return {"skipped": True, "reason": f"Dataset path not found: {source_path}"}

        try:
            labels_file = data_path / "labels.npy"
            inputs_file = data_path / "inputs.npy"

            if not inputs_file.exists() or not labels_file.exists():
                return {
                    "skipped": True,
                    "reason": (
                        f"Expected inputs.npy and labels.npy in {data_path}. "
                        "Provide a directory with these files."
                    ),
                }

            all_inputs = np.load(str(inputs_file))
            all_labels = np.load(str(labels_file))
            num_samples = min(num_samples, len(all_labels))

            out_names = [o.name for o in self._session.get_outputs()]
            input_meta = self._session.get_inputs()[0]
            correct = 0

            for i in range(num_samples):
                sample = all_inputs[i : i + 1].astype(
                    _ONNX_DTYPE_MAP.get(input_meta.type, np.float32)
                )
                feed = {input_meta.name: sample}
                output = self._session.run(out_names, feed)[0]
                pred = int(np.argmax(output, axis=-1).flat[0])
                if pred == int(all_labels[i]):
                    correct += 1

            accuracy = correct / num_samples if num_samples else 0.0
            return {
                "skipped": False,
                "num_samples": num_samples,
                "correct": correct,
                "accuracy": round(accuracy, 6),
            }

        except Exception as exc:
            log.error("Accuracy benchmark failed: %s", exc)
            return {"skipped": True, "reason": f"Error: {exc}"}

    # -- Memory ----------------------------------------------------------

    def run_memory(
        self,
        batch_sizes: Optional[List[int]] = None,
    ) -> MemoryReport:
        if batch_sizes is None:
            batch_sizes = [1, 4, 16, 64]

        report = MemoryReport()
        out_names = [o.name for o in self._session.get_outputs()]

        for bs in batch_sizes:
            log.info("Memory benchmark: batch_size=%d", bs)
            try:
                inputs = _build_random_inputs(self._session, bs)
                rss_before = _get_rss_mb()
                vram_before = _get_gpu_vram_mb()

                self._session.run(out_names, inputs)

                rss_after = _get_rss_mb()
                vram_after = _get_gpu_vram_mb()

                report.per_batch_size[bs] = {
                    "peak_rss_mb": round(rss_after, 2),
                    "rss_delta_mb": round(rss_after - rss_before, 2),
                    "gpu_vram_mb": round(vram_after, 2),
                    "gpu_vram_delta_mb": round(vram_after - vram_before, 2),
                }
            except (MemoryError, RuntimeError) as exc:
                log.warning("OOM at batch_size=%d: %s", bs, exc)
                report.per_batch_size[bs] = {
                    "peak_rss_mb": -1,
                    "rss_delta_mb": -1,
                    "gpu_vram_mb": -1,
                    "gpu_vram_delta_mb": -1,
                    "oom": True,
                }
                break

        return report

    # -- Scalability -----------------------------------------------------

    def run_scalability(
        self,
        max_threads: Optional[int] = None,
    ) -> ScalabilityReport:
        if max_threads is None:
            max_threads = min(os.cpu_count() or 4, 16)

        report = ScalabilityReport()
        thread_counts = [1]
        t = 2
        while t <= max_threads:
            thread_counts.append(t)
            t *= 2
        if thread_counts[-1] != max_threads and max_threads > 1:
            thread_counts.append(max_threads)

        inputs = _build_random_inputs(self._session, batch_size=1)
        out_names = [o.name for o in self._session.get_outputs()]
        iterations = 30

        for tc in thread_counts:
            log.info("Scalability benchmark: threads=%d", tc)

            def _timed_run(_session, _inputs, _out_names):
                t0 = time.perf_counter()
                _session.run(_out_names, _inputs)
                return (time.perf_counter() - t0) * 1000

            sessions = [_create_session(self.model_path, self.provider) for _ in range(tc)]

            timings: List[float] = []
            try:
                with ThreadPoolExecutor(max_workers=tc) as pool:
                    for _ in range(iterations):
                        futures = [
                            pool.submit(_timed_run, sessions[j], inputs, out_names)
                            for j in range(tc)
                        ]
                        for fut in as_completed(futures):
                            timings.append(fut.result())
            except (MemoryError, RuntimeError) as exc:
                log.warning("Scalability OOM at threads=%d: %s", tc, exc)
                break

            report.per_thread_count[tc] = round(statistics.mean(timings), 4)

        return report

    # -- Run all ---------------------------------------------------------

    def run_all(self, output_dir: Optional[str] = None) -> BenchmarkResult:
        result = BenchmarkResult(
            model_path=self.model_path,
            model_size_mb=round(self._model_size, 2),
            provider=", ".join(self._active_providers),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        log.info("=== Running full benchmark suite ===")

        result.latency = self.run_latency()
        result.throughput = self.run_throughput()
        result.memory = self.run_memory()
        result.scalability = self.run_scalability()

        out = output_dir or self.output_dir
        if out:
            Path(out).mkdir(parents=True, exist_ok=True)
            report_path = str(Path(out) / "benchmark_report")
            self.generate_report(result, report_path)

        return result

    # -- Report generation -----------------------------------------------

    @staticmethod
    def generate_report(results: BenchmarkResult, output_path: str) -> None:
        json_path = output_path if output_path.endswith(".json") else output_path + ".json"
        txt_path = output_path.rsplit(".", 1)[0] + ".txt" if "." in Path(output_path).name else output_path + ".txt"

        serializable = _to_serializable(asdict(results))
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)
        log.info("JSON report written to %s", json_path)

        lines = [
            "=" * 72,
            "ONNX Benchmark Report",
            "=" * 72,
            f"Model:     {results.model_path}",
            f"Size:      {results.model_size_mb:.2f} MB",
            f"Provider:  {results.provider}",
            f"Timestamp: {results.timestamp}",
            "",
        ]

        if results.latency and results.latency.per_batch_size:
            lines.append("--- Latency (ms) ---")
            lines.append(f"{'Batch':>6}  {'Mean':>10}  {'P50':>10}  {'P95':>10}  {'P99':>10}  {'Min':>10}  {'Max':>10}")
            for bs, stats in sorted(results.latency.per_batch_size.items()):
                lines.append(
                    f"{bs:>6}  {stats.mean_ms:>10.2f}  {stats.p50_ms:>10.2f}  "
                    f"{stats.p95_ms:>10.2f}  {stats.p99_ms:>10.2f}  "
                    f"{stats.min_ms:>10.2f}  {stats.max_ms:>10.2f}"
                )
            lines.append("")

        if results.throughput:
            t = results.throughput
            lines.append("--- Throughput ---")
            lines.append(f"Batch size:          {t.batch_size}")
            lines.append(f"Threads:             {t.num_threads}")
            lines.append(f"Duration:            {t.duration_s:.1f}s")
            lines.append(f"Total inferences:    {t.total_inferences}")
            lines.append(f"Inferences/sec:      {t.inferences_per_sec:.2f}")
            lines.append(f"Mean latency:        {t.mean_latency_ms:.2f} ms")
            lines.append("")

        if results.memory and results.memory.per_batch_size:
            lines.append("--- Memory ---")
            for bs, mem in sorted(results.memory.per_batch_size.items()):
                oom_tag = " [OOM]" if mem.get("oom") else ""
                lines.append(
                    f"  batch={bs}: RSS={mem['peak_rss_mb']:.1f} MB "
                    f"(delta={mem['rss_delta_mb']:.1f} MB), "
                    f"VRAM={mem['gpu_vram_mb']:.1f} MB "
                    f"(delta={mem['gpu_vram_delta_mb']:.1f} MB){oom_tag}"
                )
            lines.append("")

        if results.scalability and results.scalability.per_thread_count:
            lines.append("--- Scalability (mean latency ms per thread count) ---")
            for tc, lat in sorted(results.scalability.per_thread_count.items()):
                lines.append(f"  threads={tc}: {lat:.2f} ms")
            lines.append("")

        lines.append("=" * 72)
        report_text = "\n".join(lines)

        with open(txt_path, "w") as f:
            f.write(report_text)
        log.info("Text report written to %s", txt_path)


def _to_serializable(obj: Any) -> Any:
    """Convert dataclass-derived dicts with int keys to JSON-safe form."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def run_benchmark_suite(
    model_path: str,
    provider: str = "auto",
    output_dir: Optional[str] = None,
) -> BenchmarkResult:
    """Run the complete benchmark suite — convenience function for CLI usage."""
    suite = BenchmarkSuite(model_path, provider=provider, output_dir=output_dir)
    return suite.run_all(output_dir=output_dir)
