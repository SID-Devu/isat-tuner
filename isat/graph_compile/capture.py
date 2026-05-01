"""CUDA/HIP graph capture and replay for ONNX Runtime inference acceleration."""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("isat.graph_compile")

_TORCH_AVAILABLE: bool = False
_ORT_AVAILABLE: bool = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    pass

_STATIC_OPS = frozenset({
    "MatMul", "Gemm", "Conv", "ConvTranspose",
    "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu", "HardSigmoid",
    "BatchNormalization", "InstanceNormalization", "LayerNormalization",
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
    "MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool",
    "Softmax", "LogSoftmax",
    "Dropout", "Flatten", "Squeeze", "Unsqueeze",
    "Transpose", "Clip", "Pad", "Concat",
    "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin",
    "Cast", "Identity", "Constant",
})

_DYNAMIC_OPS = frozenset({
    "NonZero", "Where", "Compress", "Loop", "If", "Scan",
    "SequenceConstruct", "SequenceAt", "SequenceLength",
    "DynamicQuantizeLinear", "Resize", "Unique",
    "TopK", "NMS",
})


def _require_ort() -> None:
    if not _ORT_AVAILABLE:
        raise RuntimeError(
            "onnxruntime is required for graph capture. "
            "Install with: pip install onnxruntime-gpu"
        )


def _ort_supports_cuda_graph() -> bool:
    if not _ORT_AVAILABLE:
        return False
    try:
        opts = ort.SessionOptions()
        providers = ort.get_available_providers()
        return "CUDAExecutionProvider" in providers and hasattr(opts, "add_session_config_entry")
    except Exception:
        return False


def _detect_gpu_provider() -> str:
    if _ORT_AVAILABLE:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider"
        if "ROCMExecutionProvider" in available:
            return "ROCMExecutionProvider"
    if _is_rocm_available():
        return "ROCMExecutionProvider"
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "CUDAExecutionProvider"
    return "CPUExecutionProvider"


def _is_rocm_available() -> bool:
    if os.environ.get("ROCM_HOME") or os.environ.get("HIP_PATH"):
        return True
    return shutil.which("rocminfo") is not None


def _percentile(data: Sequence[float], pct: float) -> float:
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraphCaptureMetrics:
    normal_mean_ms: float = 0.0
    normal_p99_ms: float = 0.0
    captured_mean_ms: float = 0.0
    captured_p99_ms: float = 0.0
    speedup_ratio: float = 0.0
    capture_time_ms: float = 0.0
    num_kernels_captured: int = 0
    memory_saved_mb: float = 0.0


@dataclass
class GraphRegionReport:
    static_regions: List[List[str]] = field(default_factory=list)
    dynamic_regions: List[List[str]] = field(default_factory=list)
    split_points: List[int] = field(default_factory=list)
    total_nodes: int = 0
    capturable_ratio: float = 0.0


# ---------------------------------------------------------------------------
# SessionOptionsBuilder
# ---------------------------------------------------------------------------

class SessionOptionsBuilder:
    """Constructs ORT SessionOptions tuned for graph capture."""

    @staticmethod
    def for_graph_capture(provider: str = "CUDAExecutionProvider") -> "ort.SessionOptions":
        _require_ort()
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = False

        if provider == "CUDAExecutionProvider":
            try:
                opts.add_session_config_entry(
                    "gpu_graph_id", "1"
                )
            except Exception:
                log.debug("Session config entry 'gpu_graph_id' not supported in this ORT build")

        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts


# ---------------------------------------------------------------------------
# GraphCapture (CUDA / NVIDIA)
# ---------------------------------------------------------------------------

class GraphCapture:
    """Capture and replay CUDA graphs for ONNX Runtime inference."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        provider: str = "CUDAExecutionProvider",
    ) -> None:
        _require_ort()
        self._model_path = str(model_path)
        self._provider = provider
        self._captured = False
        self._cuda_graph: Any = None
        self._static_inputs: Dict[str, np.ndarray] = {}
        self._static_outputs: Dict[str, np.ndarray] = {}
        self._capture_stream: Any = None

        if provider == "CPUExecutionProvider":
            log.warning(
                "Graph capture is a no-op on CPU. "
                "Use CUDAExecutionProvider or ROCMExecutionProvider for acceleration."
            )

        provider_options: List[Tuple[str, Any]] = []
        if provider == "CUDAExecutionProvider":
            cuda_opts: Dict[str, Any] = {
                "arena_extend_strategy": "kSameAsRequested",
            }
            if _ort_supports_cuda_graph():
                cuda_opts["cuda_graph_enable"] = "1"
            provider_options = [(provider, cuda_opts)]
        else:
            provider_options = [(provider, {})]

        sess_opts = SessionOptionsBuilder.for_graph_capture(provider)
        try:
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=sess_opts,
                providers=[p[0] for p in provider_options],
                provider_options=[p[1] for p in provider_options],
            )
        except Exception:
            log.warning("Failed to create session with graph capture options, falling back to defaults")
            self._session = ort.InferenceSession(
                self._model_path, providers=[provider]
            )

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def is_captured(self) -> bool:
        return self._captured

    def warmup(self, inputs: Dict[str, np.ndarray], num_warmup: int = 3) -> None:
        log.info("Running %d warmup iterations", num_warmup)
        for i in range(num_warmup):
            self._session.run(self._output_names, inputs)
            log.debug("Warmup iteration %d/%d complete", i + 1, num_warmup)

    def capture(self, inputs: Dict[str, np.ndarray]) -> None:
        if self._captured:
            log.warning("Graph already captured; recapturing")
            self._captured = False
            self._cuda_graph = None

        self.warmup(inputs)

        t0 = time.perf_counter()

        if _TORCH_AVAILABLE and torch.cuda.is_available() and self._provider == "CUDAExecutionProvider":
            self._capture_with_torch(inputs)
        else:
            self._capture_with_ort(inputs)

        capture_ms = (time.perf_counter() - t0) * 1000
        self._captured = True
        log.info("Graph captured in %.2f ms", capture_ms)

    def _capture_with_torch(self, inputs: Dict[str, np.ndarray]) -> None:
        stream = torch.cuda.Stream()
        self._capture_stream = stream

        for name, arr in inputs.items():
            self._static_inputs[name] = arr.copy()

        torch.cuda.synchronize()

        with torch.cuda.stream(stream):
            self._session.run(self._output_names, self._static_inputs)

        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._session.run(self._output_names, self._static_inputs)

        if outputs:
            for name, out in zip(self._output_names, outputs):
                self._static_outputs[name] = out.copy() if isinstance(out, np.ndarray) else np.array(out)

        self._cuda_graph = graph

    def _capture_with_ort(self, inputs: Dict[str, np.ndarray]) -> None:
        for name, arr in inputs.items():
            self._static_inputs[name] = arr.copy()

        outputs = self._session.run(self._output_names, self._static_inputs)
        if outputs:
            for name, out in zip(self._output_names, outputs):
                self._static_outputs[name] = out.copy() if isinstance(out, np.ndarray) else np.array(out)

        self._cuda_graph = "ort_native"
        log.info("Using ORT-native graph capture (torch unavailable for manual CUDA graph)")

    def replay(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._captured:
            raise RuntimeError("No graph captured. Call capture() first.")

        for name, arr in inputs.items():
            if name in self._static_inputs:
                np.copyto(self._static_inputs[name], arr)

        if isinstance(self._cuda_graph, str) and self._cuda_graph == "ort_native":
            raw_outputs = self._session.run(self._output_names, self._static_inputs)
            return dict(zip(self._output_names, raw_outputs))

        if _TORCH_AVAILABLE and self._cuda_graph is not None:
            self._cuda_graph.replay()
            torch.cuda.synchronize()

        return dict(self._static_outputs)

    def benchmark(
        self,
        inputs: Dict[str, np.ndarray],
        num_runs: int = 100,
    ) -> GraphCaptureMetrics:
        normal_times: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self._session.run(self._output_names, inputs)
            normal_times.append((time.perf_counter() - t0) * 1000)

        if not self._captured:
            capture_t0 = time.perf_counter()
            self.capture(inputs)
            capture_ms = (time.perf_counter() - capture_t0) * 1000
        else:
            capture_ms = 0.0

        captured_times: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.replay(inputs)
            captured_times.append((time.perf_counter() - t0) * 1000)

        normal_mean = statistics.mean(normal_times)
        captured_mean = statistics.mean(captured_times)

        mem_before = 0.0
        mem_after = 0.0
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated() / (1024 * 1024)

        return GraphCaptureMetrics(
            normal_mean_ms=round(normal_mean, 4),
            normal_p99_ms=round(_percentile(normal_times, 99), 4),
            captured_mean_ms=round(captured_mean, 4),
            captured_p99_ms=round(_percentile(captured_times, 99), 4),
            speedup_ratio=round(normal_mean / max(captured_mean, 1e-9), 2),
            capture_time_ms=round(capture_ms, 4),
            num_kernels_captured=len(self._static_outputs),
            memory_saved_mb=round(max(mem_before - mem_after, 0.0), 2),
        )


# ---------------------------------------------------------------------------
# HIPGraphCapture (AMD ROCm)
# ---------------------------------------------------------------------------

class HIPGraphCapture:
    """Capture and replay HIP graphs for AMD ROCm inference acceleration."""

    def __init__(
        self,
        model_path: str | os.PathLike,
        provider: str = "ROCMExecutionProvider",
    ) -> None:
        _require_ort()
        self._model_path = str(model_path)
        self._provider = provider
        self._captured = False
        self._hip_graph: Any = None
        self._hip_graph_exec: Any = None
        self._hip_stream: Any = None
        self._static_inputs: Dict[str, np.ndarray] = {}
        self._static_outputs: Dict[str, np.ndarray] = {}
        self._hiplib: Any = None

        if not _is_rocm_available():
            log.warning("ROCm not detected; HIP graph capture will not accelerate inference")

        sess_opts = SessionOptionsBuilder.for_graph_capture(provider)
        try:
            self._session = ort.InferenceSession(
                self._model_path,
                sess_options=sess_opts,
                providers=[provider],
            )
        except Exception:
            log.warning("Failed with ROCMExecutionProvider, falling back to CPU")
            self._session = ort.InferenceSession(
                self._model_path, providers=["CPUExecutionProvider"]
            )

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]

    @property
    def is_captured(self) -> bool:
        return self._captured

    def _load_hip_library(self) -> Any:
        if self._hiplib is not None:
            return self._hiplib

        search_paths = [
            os.path.join(os.environ.get("ROCM_HOME", "/opt/rocm"), "lib", "libamdhip64.so"),
            os.path.join(os.environ.get("HIP_PATH", "/opt/rocm/hip"), "lib", "libamdhip64.so"),
            "libamdhip64.so",
        ]
        for path in search_paths:
            try:
                self._hiplib = ctypes.CDLL(path)
                return self._hiplib
            except OSError:
                continue
        raise RuntimeError(
            "Cannot load libamdhip64.so. Ensure ROCm is installed and ROCM_HOME is set."
        )

    def _hip_graph_begin_capture(self) -> ctypes.c_void_p:
        hiplib = self._load_hip_library()
        stream = ctypes.c_void_p()
        rc = hiplib.hipStreamCreate(ctypes.byref(stream))
        if rc != 0:
            raise RuntimeError(f"hipStreamCreate failed with error code {rc}")
        self._hip_stream = stream

        # hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal=0)
        rc = hiplib.hipStreamBeginCapture(stream, ctypes.c_int(0))
        if rc != 0:
            raise RuntimeError(f"hipStreamBeginCapture failed with error code {rc}")

        return stream

    def _hip_graph_end_capture(self) -> Tuple[ctypes.c_void_p, ctypes.c_void_p]:
        hiplib = self._load_hip_library()
        graph = ctypes.c_void_p()
        rc = hiplib.hipStreamEndCapture(self._hip_stream, ctypes.byref(graph))
        if rc != 0:
            raise RuntimeError(f"hipStreamEndCapture failed with error code {rc}")

        graph_exec = ctypes.c_void_p()
        rc = hiplib.hipGraphInstantiate(
            ctypes.byref(graph_exec), graph, ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_size_t(0)
        )
        if rc != 0:
            raise RuntimeError(f"hipGraphInstantiate failed with error code {rc}")

        return graph, graph_exec

    def _hip_graph_launch(self) -> None:
        if self._hip_graph_exec is None:
            raise RuntimeError("No HIP graph instantiated for launch")

        hiplib = self._load_hip_library()
        rc = hiplib.hipGraphLaunch(self._hip_graph_exec, self._hip_stream)
        if rc != 0:
            raise RuntimeError(f"hipGraphLaunch failed with error code {rc}")

        rc = hiplib.hipStreamSynchronize(self._hip_stream)
        if rc != 0:
            raise RuntimeError(f"hipStreamSynchronize failed with error code {rc}")

    def warmup(self, inputs: Dict[str, np.ndarray], num_warmup: int = 3) -> None:
        log.info("Running %d HIP warmup iterations", num_warmup)
        for i in range(num_warmup):
            self._session.run(self._output_names, inputs)
            log.debug("HIP warmup iteration %d/%d complete", i + 1, num_warmup)

    def capture(self, inputs: Dict[str, np.ndarray]) -> None:
        if self._captured:
            log.warning("HIP graph already captured; recapturing")
            self._captured = False
            self._hip_graph = None
            self._hip_graph_exec = None

        self.warmup(inputs)

        for name, arr in inputs.items():
            self._static_inputs[name] = arr.copy()

        t0 = time.perf_counter()

        try:
            self._hip_graph_begin_capture()
            self._session.run(self._output_names, self._static_inputs)
            self._hip_graph, self._hip_graph_exec = self._hip_graph_end_capture()
        except (RuntimeError, OSError) as exc:
            log.warning("Native HIP graph capture failed (%s), using ORT fallback", exc)
            outputs = self._session.run(self._output_names, self._static_inputs)
            if outputs:
                for name, out in zip(self._output_names, outputs):
                    self._static_outputs[name] = out.copy() if isinstance(out, np.ndarray) else np.array(out)
            self._hip_graph_exec = None

        capture_ms = (time.perf_counter() - t0) * 1000
        self._captured = True
        log.info("HIP graph captured in %.2f ms", capture_ms)

    def replay(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not self._captured:
            raise RuntimeError("No HIP graph captured. Call capture() first.")

        for name, arr in inputs.items():
            if name in self._static_inputs:
                np.copyto(self._static_inputs[name], arr)

        if self._hip_graph_exec is not None:
            self._hip_graph_launch()
        else:
            raw_outputs = self._session.run(self._output_names, self._static_inputs)
            return dict(zip(self._output_names, raw_outputs))

        return dict(self._static_outputs)

    def benchmark(
        self,
        inputs: Dict[str, np.ndarray],
        num_runs: int = 100,
    ) -> GraphCaptureMetrics:
        normal_times: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self._session.run(self._output_names, inputs)
            normal_times.append((time.perf_counter() - t0) * 1000)

        if not self._captured:
            capture_t0 = time.perf_counter()
            self.capture(inputs)
            capture_ms = (time.perf_counter() - capture_t0) * 1000
        else:
            capture_ms = 0.0

        captured_times: List[float] = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.replay(inputs)
            captured_times.append((time.perf_counter() - t0) * 1000)

        normal_mean = statistics.mean(normal_times)
        captured_mean = statistics.mean(captured_times)

        return GraphCaptureMetrics(
            normal_mean_ms=round(normal_mean, 4),
            normal_p99_ms=round(_percentile(normal_times, 99), 4),
            captured_mean_ms=round(captured_mean, 4),
            captured_p99_ms=round(_percentile(captured_times, 99), 4),
            speedup_ratio=round(normal_mean / max(captured_mean, 1e-9), 2),
            capture_time_ms=round(capture_ms, 4),
            num_kernels_captured=len(self._static_outputs),
            memory_saved_mb=0.0,
        )


# ---------------------------------------------------------------------------
# GraphRegionAnalyzer
# ---------------------------------------------------------------------------

class GraphRegionAnalyzer:
    """Analyze which regions of an ONNX graph are safe for graph capture."""

    def __init__(self, model_path: str | os.PathLike) -> None:
        try:
            import onnx
            self._onnx = onnx
        except ImportError:
            raise RuntimeError("onnx package is required for graph analysis. Install with: pip install onnx")
        self._model_path = str(model_path)
        self._model = onnx.load(self._model_path, load_external_data=False)

    def analyze(self) -> GraphRegionReport:
        graph = self._model.graph
        nodes = list(graph.node)
        total = len(nodes)

        classification: List[bool] = [self._classify_node(n) for n in nodes]
        static_regions, dynamic_regions, split_points = self._find_capture_regions(
            nodes, classification
        )

        capturable = sum(len(r) for r in static_regions)
        return GraphRegionReport(
            static_regions=static_regions,
            dynamic_regions=dynamic_regions,
            split_points=split_points,
            total_nodes=total,
            capturable_ratio=round(capturable / max(total, 1), 4),
        )

    def _classify_node(self, node: Any) -> bool:
        """Return True if the node is capture-safe (static shapes)."""
        op = node.op_type
        if op in _DYNAMIC_OPS:
            return False
        if op in _STATIC_OPS:
            return True

        for attr in node.attribute:
            if attr.name in ("axes", "shape"):
                if any(d < 0 for d in attr.ints):
                    return False

        if op == "Reshape":
            return self._is_static_reshape(node)

        return True

    def _is_static_reshape(self, node: Any) -> bool:
        if len(node.input) < 2:
            return False
        shape_input = node.input[1]
        for init in self._model.graph.initializer:
            if init.name == shape_input:
                shape_vals = list(self._onnx.numpy_helper.to_array(init))
                return all(v >= 0 for v in shape_vals)
        return False

    def _find_capture_regions(
        self,
        nodes: List[Any],
        classification: List[bool],
    ) -> Tuple[List[List[str]], List[List[str]], List[int]]:
        static_regions: List[List[str]] = []
        dynamic_regions: List[List[str]] = []
        split_points: List[int] = []

        current_static: List[str] = []
        current_dynamic: List[str] = []
        prev_was_static: Optional[bool] = None

        for idx, (node, is_static) in enumerate(zip(nodes, classification)):
            label = f"{node.op_type}:{node.name or idx}"
            if is_static:
                if prev_was_static is False:
                    split_points.append(idx)
                    if current_dynamic:
                        dynamic_regions.append(current_dynamic)
                        current_dynamic = []
                current_static.append(label)
                prev_was_static = True
            else:
                if prev_was_static is True:
                    split_points.append(idx)
                    if current_static:
                        static_regions.append(current_static)
                        current_static = []
                current_dynamic.append(label)
                prev_was_static = False

        if current_static:
            static_regions.append(current_static)
        if current_dynamic:
            dynamic_regions.append(current_dynamic)

        return static_regions, dynamic_regions, split_points


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def graph_compile(
    model_path: str,
    provider: str = "auto",
    action: str = "capture",
    **kwargs: Any,
) -> Any:
    """CLI-oriented entry point for graph capture and analysis.

    Args:
        model_path: Path to the ONNX model.
        provider: Execution provider or ``"auto"`` for detection.
        action: One of ``"capture"``, ``"benchmark"``, ``"analyze"``.
        **kwargs: Forwarded to the underlying class.

    Returns:
        The result of the requested action (metrics, report, or capture object).
    """
    if provider == "auto":
        provider = _detect_gpu_provider()
        log.info("Auto-detected provider: %s", provider)

    if provider == "CPUExecutionProvider":
        log.warning(
            "No GPU provider available. Graph capture requires CUDA or ROCm. "
            "Falling back to CPU — no graph acceleration will be applied."
        )

    if action == "analyze":
        analyzer = GraphRegionAnalyzer(model_path)
        report = analyzer.analyze()
        log.info(
            "Analysis complete: %d nodes, %.1f%% capturable, %d split points",
            report.total_nodes,
            report.capturable_ratio * 100,
            len(report.split_points),
        )
        return report

    if provider == "ROCMExecutionProvider":
        capturer: GraphCapture | HIPGraphCapture = HIPGraphCapture(model_path, provider=provider)
    else:
        capturer = GraphCapture(model_path, provider=provider)

    dummy_inputs = kwargs.pop("inputs", None)
    if dummy_inputs is None:
        _require_ort()
        dummy_inputs = {}
        sess = ort.InferenceSession(model_path, providers=[provider])
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dtype_map = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
                         "tensor(int64)": np.int64, "tensor(int32)": np.int32,
                         "tensor(bool)": np.bool_}
            dtype = dtype_map.get(inp.type, np.float32)
            dummy_inputs[inp.name] = np.random.randn(*shape).astype(dtype)

    if action == "capture":
        capturer.capture(dummy_inputs)
        log.info("Graph capture complete for %s", model_path)
        return capturer

    if action == "benchmark":
        num_runs = kwargs.pop("num_runs", 100)
        metrics = capturer.benchmark(dummy_inputs, num_runs=num_runs)
        log.info(
            "Benchmark: normal=%.2fms (p99=%.2fms), captured=%.2fms (p99=%.2fms), speedup=%.2fx",
            metrics.normal_mean_ms, metrics.normal_p99_ms,
            metrics.captured_mean_ms, metrics.captured_p99_ms,
            metrics.speedup_ratio,
        )
        return metrics

    raise ValueError(f"Unknown action: {action!r}. Use 'capture', 'benchmark', or 'analyze'.")
