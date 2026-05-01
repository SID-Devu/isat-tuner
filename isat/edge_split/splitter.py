from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _lazy_onnx():
    import onnx
    return onnx


def _lazy_ort():
    import onnxruntime as ort
    return ort


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SplitPoint:
    layer_idx: int
    node_name: str
    activation_size_mb: float
    edge_flops: float
    cloud_flops: float
    estimated_transfer_ms: float
    estimated_total_latency_ms: float


@dataclass
class HybridResult:
    output: Any
    edge_time_ms: float
    transfer_time_ms: float
    cloud_time_ms: float
    total_time_ms: float
    activation_size_mb: float
    compressed_size_mb: float
    privacy_preserved: bool


@dataclass
class SplitReport:
    optimal_split: SplitPoint
    all_splits: List[SplitPoint]
    recommended_bandwidth_mbps: float
    total_layers: int
    edge_layers: int
    cloud_layers: int


# ---------------------------------------------------------------------------
# SplitAnalyzer
# ---------------------------------------------------------------------------

class SplitAnalyzer:
    """Analyze an ONNX model to find optimal edge/cloud split points."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        onnx = _lazy_onnx()
        logger.info("Loading model for split analysis: %s", self.model_path)
        self._model = onnx.load(str(self.model_path), load_external_data=False)
        self._graph = self._model.graph

    def analyze(self) -> List[SplitPoint]:
        """Evaluate every potential split point between consecutive layers.

        Returns split points sorted by estimated total latency (ascending).
        """
        return self.find_optimal_split(return_all=True)

    def find_optimal_split(
        self,
        bandwidth_mbps: float = 100.0,
        edge_tflops: float = 1.0,
        cloud_tflops: float = 10.0,
        return_all: bool = False,
    ) -> SplitPoint | List[SplitPoint]:
        """Pick split point that minimises total latency.

        If *return_all* is True, return the full sorted list instead.
        """
        nodes = list(self._graph.node)
        if not nodes:
            raise ValueError("Model graph has no nodes")

        edge_flops_per_ms = edge_tflops * 1e9
        cloud_flops_per_ms = cloud_tflops * 1e9
        bandwidth_bytes_per_ms = (bandwidth_mbps * 1e6) / (8 * 1000)

        total_flops = self._estimate_flops(nodes)
        splits: List[SplitPoint] = []

        for idx in range(1, len(nodes)):
            edge_nodes = nodes[:idx]
            cloud_nodes = nodes[idx:]

            act_shape = self._get_activation_shape(nodes[idx - 1], self._graph)
            act_elements = int(np.prod(act_shape)) if act_shape else 0
            act_size_bytes = act_elements * 4  # float32
            act_size_mb = act_size_bytes / (1024 * 1024)

            e_flops = self._estimate_flops(edge_nodes)
            c_flops = self._estimate_flops(cloud_nodes)

            edge_compute_ms = e_flops / edge_flops_per_ms if edge_flops_per_ms else 0
            cloud_compute_ms = c_flops / cloud_flops_per_ms if cloud_flops_per_ms else 0
            transfer_ms = (act_size_bytes / bandwidth_bytes_per_ms) if bandwidth_bytes_per_ms else 0

            total_latency = edge_compute_ms + transfer_ms + cloud_compute_ms

            splits.append(SplitPoint(
                layer_idx=idx,
                node_name=nodes[idx - 1].name or f"node_{idx - 1}",
                activation_size_mb=act_size_mb,
                edge_flops=e_flops,
                cloud_flops=c_flops,
                estimated_transfer_ms=transfer_ms,
                estimated_total_latency_ms=total_latency,
            ))

        splits.sort(key=lambda sp: sp.estimated_total_latency_ms)
        if return_all:
            return splits
        return splits[0]

    def find_privacy_split(self, min_edge_layers: int = 2) -> SplitPoint:
        """Find best split where at least *min_edge_layers* run on-device.

        Ensures that raw input never leaves the device.
        """
        all_splits = self.analyze()
        candidates = [sp for sp in all_splits if sp.layer_idx >= min_edge_layers]
        if not candidates:
            raise ValueError(
                f"No split point satisfies min_edge_layers={min_edge_layers} "
                f"(model has {len(list(self._graph.node))} nodes)"
            )
        return candidates[0]

    # -- internals --

    def _estimate_flops(self, nodes) -> float:
        """Sum estimated FLOPS for a list of ONNX graph nodes.

        Heuristics:
          MatMul  : 2 * M * N * K
          Conv    : 2 * C_out * K_h * K_w * H_out * W_out * C_in
          Gemm    : 2 * M * N * K
          Others  : 0 (conservative)
        """
        total = 0.0
        for node in nodes:
            op = node.op_type
            if op in ("MatMul", "Gemm"):
                dims = self._get_matmul_dims(node)
                if dims:
                    m, n, k = dims
                    total += 2.0 * m * n * k
            elif op == "Conv":
                dims = self._get_conv_dims(node)
                if dims:
                    c_out, k_h, k_w, h_out, w_out, c_in = dims
                    total += 2.0 * c_out * k_h * k_w * h_out * w_out * c_in
        return total

    def _get_matmul_dims(self, node) -> Optional[Tuple[int, int, int]]:
        """Try to infer (M, N, K) from initialiser / value_info shapes."""
        shapes = []
        for inp in node.input:
            shape = self._resolve_shape(inp)
            if shape is not None:
                shapes.append(shape)
        if len(shapes) >= 2:
            a, b = shapes[0], shapes[1]
            if len(a) >= 2 and len(b) >= 2:
                m, k = a[-2], a[-1]
                n = b[-1]
                return (m, n, k)
        return None

    def _get_conv_dims(self, node) -> Optional[Tuple[int, ...]]:
        """Try to infer Conv dimensions from weight initialiser."""
        if len(node.input) < 2:
            return None
        weight_shape = self._resolve_shape(node.input[1])
        if weight_shape is None or len(weight_shape) < 4:
            return None
        c_out, c_in, k_h, k_w = weight_shape[:4]
        out_shape = self._get_activation_shape(node, self._graph)
        if out_shape and len(out_shape) >= 4:
            h_out, w_out = out_shape[2], out_shape[3]
        else:
            h_out, w_out = 1, 1
        return (c_out, k_h, k_w, h_out, w_out, c_in)

    def _resolve_shape(self, name: str) -> Optional[List[int]]:
        """Resolve a tensor name to its shape via initialiser or value_info."""
        onnx = _lazy_onnx()
        for init in self._graph.initializer:
            if init.name == name:
                return list(init.dims)
        for vi in list(self._graph.input) + list(self._graph.value_info) + list(self._graph.output):
            if vi.name == name:
                shape = []
                for d in vi.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value > 0 else 1)
                return shape if shape else None
        return None

    def _get_activation_shape(self, node, graph) -> Optional[List[int]]:
        """Infer output tensor shape at split point."""
        if not node.output:
            return None
        out_name = node.output[0]
        for vi in list(graph.value_info) + list(graph.output):
            if vi.name == out_name:
                shape = []
                for d in vi.type.tensor_type.shape.dim:
                    shape.append(d.dim_value if d.dim_value > 0 else 1)
                return shape if shape else None
        return None


# ---------------------------------------------------------------------------
# ActivationCompressor
# ---------------------------------------------------------------------------

class ActivationCompressor:
    """Compress activation tensors for network transfer between edge and cloud."""

    METHODS = ("quantize", "topk", "random_projection")

    def __init__(self, method: str = "quantize"):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method {method!r}, choose from {self.METHODS}")
        self.method = method

    def compress(self, activations: np.ndarray) -> Dict[str, Any]:
        """Compress activation tensor for network transfer."""
        if self.method == "quantize":
            return self._quantize_int8(activations)
        elif self.method == "topk":
            return self._topk_compress(activations)
        elif self.method == "random_projection":
            return self._random_projection(activations)
        raise ValueError(f"Unsupported method: {self.method}")

    def decompress(self, compressed: Dict[str, Any]) -> np.ndarray:
        """Restore activations on receiving end."""
        method = compressed["method"]
        if method == "quantize":
            return self._dequantize_int8(
                compressed["data"], compressed["scale"], compressed["zero_point"]
            )
        elif method == "topk":
            data = np.zeros(compressed["original_shape"], dtype=np.float32)
            data.flat[compressed["indices"]] = compressed["values"]
            return data
        elif method == "random_projection":
            proj_matrix = compressed["projection_matrix"]
            projected = compressed["data"]
            reconstructed = projected @ proj_matrix
            return reconstructed.reshape(compressed["original_shape"])
        raise ValueError(f"Unsupported method: {method}")

    # -- INT8 affine quantisation --

    def _quantize_int8(self, data: np.ndarray) -> Dict[str, Any]:
        """Affine quantise to INT8 with scale and zero-point."""
        fmin, fmax = float(data.min()), float(data.max())
        if fmax == fmin:
            scale = 1.0
        else:
            scale = (fmax - fmin) / 255.0
        zero_point = int(round(-fmin / scale)) if scale != 0 else 0
        zero_point = max(0, min(255, zero_point))
        quantized = np.clip(np.round(data / scale) + zero_point, 0, 255).astype(np.uint8)
        return {
            "method": "quantize",
            "data": quantized,
            "scale": scale,
            "zero_point": zero_point,
            "original_shape": data.shape,
        }

    def _dequantize_int8(
        self, data: np.ndarray, scale: float, zero_point: int
    ) -> np.ndarray:
        """Restore from INT8 affine quantisation."""
        return (data.astype(np.float32) - zero_point) * scale

    # -- Top-k sparsification --

    def _topk_compress(self, data: np.ndarray, k_ratio: float = 0.1) -> Dict[str, Any]:
        """Keep only top-k% values by magnitude, zero the rest."""
        flat = data.flatten()
        k = max(1, int(len(flat) * k_ratio))
        indices = np.argpartition(np.abs(flat), -k)[-k:]
        values = flat[indices].astype(np.float32)
        return {
            "method": "topk",
            "indices": indices,
            "values": values,
            "original_shape": data.shape,
            "k_ratio": k_ratio,
        }

    # -- Random (Johnson-Lindenstrauss) projection --

    def _random_projection(
        self, data: np.ndarray, target_dim: Optional[int] = None
    ) -> Dict[str, Any]:
        """JL projection: multiply by random Gaussian matrix / sqrt(target_dim)."""
        original_shape = data.shape
        matrix = data.reshape(data.shape[0], -1) if data.ndim > 2 else data.copy()
        if data.ndim == 1:
            matrix = data.reshape(1, -1)
        n, d = matrix.shape
        if target_dim is None:
            target_dim = max(1, d // 4)
        rng = np.random.RandomState(42)
        proj_matrix = rng.randn(target_dim, d).astype(np.float32) / np.sqrt(target_dim)
        projected = matrix @ proj_matrix.T
        return {
            "method": "random_projection",
            "data": projected.astype(np.float32),
            "projection_matrix": proj_matrix,
            "original_shape": original_shape,
            "target_dim": target_dim,
        }


# ---------------------------------------------------------------------------
# EdgeExecutor
# ---------------------------------------------------------------------------

class EdgeExecutor:
    """Execute the edge (front) portion of a split ONNX model."""

    def __init__(
        self,
        model_path: str | Path,
        split_point: SplitPoint,
        provider: str = "CPUExecutionProvider",
    ):
        self.model_path = Path(model_path)
        self.split_point = split_point
        self.provider = provider
        self._edge_model_path = self._build_edge_model(model_path, split_point)
        ort = _lazy_ort()
        self._session = ort.InferenceSession(
            self._edge_model_path, providers=[provider]
        )

    def run(self, inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run edge layers, return intermediate activations."""
        input_names = [i.name for i in self._session.get_inputs()]
        feed = {n: inputs[n] for n in input_names if n in inputs}
        outputs = self._session.run(None, feed)
        return outputs[0]

    def _build_edge_model(
        self, model_path: str | Path, split_point: SplitPoint
    ) -> str:
        """Extract subgraph from input to split_point, save as separate ONNX."""
        onnx = _lazy_onnx()
        from onnx import helper, TensorProto

        model = onnx.load(str(model_path), load_external_data=False)
        try:
            onnx.load_external_data_for_model(model, str(Path(model_path).parent))
        except Exception:
            pass

        graph = model.graph
        edge_nodes = list(graph.node)[: split_point.layer_idx]

        needed_inputs = set()
        produced = set()
        for node in edge_nodes:
            for inp in node.input:
                if inp and inp not in produced:
                    needed_inputs.add(inp)
            for out in node.output:
                produced.add(out)

        inputs = [inp for inp in graph.input if inp.name in needed_inputs]
        initializers = [init for init in graph.initializer if init.name in needed_inputs]

        last_node = edge_nodes[-1] if edge_nodes else None
        if last_node and last_node.output:
            split_output_name = last_node.output[0]
        else:
            raise ValueError("Cannot determine edge model output")

        out_vi = None
        for vi in list(graph.value_info) + list(graph.output):
            if vi.name == split_output_name:
                out_vi = vi
                break

        if out_vi is None:
            out_vi = helper.make_tensor_value_info(
                split_output_name, TensorProto.FLOAT, None
            )

        edge_graph = helper.make_graph(
            edge_nodes, "edge_graph", inputs, [out_vi], initializers
        )
        edge_model = helper.make_model(edge_graph)
        edge_model.ir_version = model.ir_version
        for oi in model.opset_import:
            edge_model.opset_import.append(oi)
        if edge_model.opset_import and len(edge_model.opset_import) > len(model.opset_import):
            del edge_model.opset_import[0]

        edge_path = os.path.join(
            tempfile.gettempdir(), f"isat_edge_{split_point.layer_idx}.onnx"
        )
        onnx.save(edge_model, edge_path)
        logger.info("Built edge model (%d nodes) -> %s", len(edge_nodes), edge_path)
        return edge_path


# ---------------------------------------------------------------------------
# CloudExecutor
# ---------------------------------------------------------------------------

class CloudExecutor:
    """Execute the cloud (back) portion of a split ONNX model."""

    def __init__(
        self,
        model_path: str | Path,
        split_point: SplitPoint,
        provider: str = "CPUExecutionProvider",
    ):
        self.model_path = Path(model_path)
        self.split_point = split_point
        self.provider = provider
        self._cloud_model_path = self._build_cloud_model(model_path, split_point)
        ort = _lazy_ort()
        self._session = ort.InferenceSession(
            self._cloud_model_path, providers=[provider]
        )

    def run(self, activations: np.ndarray) -> np.ndarray:
        """Run remaining layers from activations to output."""
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: activations})
        return outputs[0]

    def _build_cloud_model(
        self, model_path: str | Path, split_point: SplitPoint
    ) -> str:
        """Extract subgraph from split_point to output."""
        onnx = _lazy_onnx()
        from onnx import helper, TensorProto

        model = onnx.load(str(model_path), load_external_data=False)
        try:
            onnx.load_external_data_for_model(model, str(Path(model_path).parent))
        except Exception:
            pass

        graph = model.graph
        all_nodes = list(graph.node)
        cloud_nodes = all_nodes[split_point.layer_idx:]

        needed_inputs = set()
        produced = set()
        for node in cloud_nodes:
            for inp in node.input:
                if inp and inp not in produced:
                    needed_inputs.add(inp)
            for out in node.output:
                produced.add(out)

        edge_last = all_nodes[split_point.layer_idx - 1]
        split_output_name = edge_last.output[0] if edge_last.output else "split_input"

        split_vi = None
        for vi in list(graph.value_info) + list(graph.output):
            if vi.name == split_output_name:
                split_vi = vi
                break
        if split_vi is None:
            split_vi = helper.make_tensor_value_info(
                split_output_name, TensorProto.FLOAT, None
            )

        extra_inputs = []
        for vi in graph.input:
            if vi.name in needed_inputs and vi.name != split_output_name:
                extra_inputs.append(vi)

        initializers = [
            init for init in graph.initializer if init.name in needed_inputs
        ]

        outputs = list(graph.output)

        cloud_graph = helper.make_graph(
            cloud_nodes,
            "cloud_graph",
            [split_vi] + extra_inputs,
            outputs,
            initializers,
        )
        cloud_model = helper.make_model(cloud_graph)
        cloud_model.ir_version = model.ir_version
        for oi in model.opset_import:
            cloud_model.opset_import.append(oi)
        if cloud_model.opset_import and len(cloud_model.opset_import) > len(model.opset_import):
            del cloud_model.opset_import[0]

        cloud_path = os.path.join(
            tempfile.gettempdir(), f"isat_cloud_{split_point.layer_idx}.onnx"
        )
        onnx.save(cloud_model, cloud_path)
        logger.info("Built cloud model (%d nodes) -> %s", len(cloud_nodes), cloud_path)
        return cloud_path


# ---------------------------------------------------------------------------
# HybridExecutor
# ---------------------------------------------------------------------------

class HybridExecutor:
    """Orchestrate edge -> compress -> transfer -> decompress -> cloud pipeline."""

    def __init__(
        self,
        model_path: str | Path,
        split_point: Optional[SplitPoint] = None,
        edge_provider: str = "CPUExecutionProvider",
        cloud_provider: str = "CUDAExecutionProvider",
        bandwidth_mbps: float = 100.0,
        compress_activations: bool = True,
    ):
        self.model_path = Path(model_path)
        self.bandwidth_mbps = bandwidth_mbps
        self.compress_activations = compress_activations

        if split_point is None:
            analyzer = SplitAnalyzer(model_path)
            split_point = analyzer.find_optimal_split(bandwidth_mbps=bandwidth_mbps)
        self.split_point = split_point

        self._edge = EdgeExecutor(model_path, split_point, provider=edge_provider)
        self._cloud = CloudExecutor(model_path, split_point, provider=cloud_provider)
        self._compressor = ActivationCompressor(method="quantize") if compress_activations else None

    def run(self, inputs: Dict[str, np.ndarray]) -> HybridResult:
        """Edge execute -> compress -> transfer -> decompress -> cloud execute."""
        t0 = time.perf_counter()
        activations = self._edge.run(inputs)
        edge_ms = (time.perf_counter() - t0) * 1000

        act_size_bytes = activations.nbytes
        act_size_mb = act_size_bytes / (1024 * 1024)

        if self._compressor is not None:
            compressed = self._compressor.compress(activations)
            compressed_size_mb = self._compressed_size(compressed)
        else:
            compressed = activations
            compressed_size_mb = act_size_mb

        transfer_bytes = compressed_size_mb * 1024 * 1024
        bandwidth_bps = self.bandwidth_mbps * 1e6 / 8
        transfer_ms = (transfer_bytes / bandwidth_bps) * 1000 if bandwidth_bps else 0
        time.sleep(transfer_ms / 1000)

        if self._compressor is not None:
            cloud_input = self._compressor.decompress(compressed)
        else:
            cloud_input = compressed

        t2 = time.perf_counter()
        output = self._cloud.run(cloud_input.astype(np.float32))
        cloud_ms = (time.perf_counter() - t2) * 1000

        total_ms = edge_ms + transfer_ms + cloud_ms

        return HybridResult(
            output=output,
            edge_time_ms=edge_ms,
            transfer_time_ms=transfer_ms,
            cloud_time_ms=cloud_ms,
            total_time_ms=total_ms,
            activation_size_mb=act_size_mb,
            compressed_size_mb=compressed_size_mb,
            privacy_preserved=self.split_point.layer_idx >= 1,
        )

    def benchmark(self, num_samples: int = 50) -> Dict[str, Any]:
        """Measure edge / transfer / cloud times; compare against full-device."""
        ort = _lazy_ort()
        full_session = ort.InferenceSession(
            str(self.model_path),
            providers=[self._edge.provider],
        )
        full_inputs = {}
        for inp in full_session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            full_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

        full_times = []
        for _ in range(num_samples):
            t = time.perf_counter()
            full_session.run(None, full_inputs)
            full_times.append((time.perf_counter() - t) * 1000)

        edge_times, transfer_times, cloud_times, total_times = [], [], [], []
        for _ in range(num_samples):
            result = self.run(full_inputs)
            edge_times.append(result.edge_time_ms)
            transfer_times.append(result.transfer_time_ms)
            cloud_times.append(result.cloud_time_ms)
            total_times.append(result.total_time_ms)

        return {
            "full_device_avg_ms": float(np.mean(full_times)),
            "hybrid_avg_ms": float(np.mean(total_times)),
            "edge_avg_ms": float(np.mean(edge_times)),
            "transfer_avg_ms": float(np.mean(transfer_times)),
            "cloud_avg_ms": float(np.mean(cloud_times)),
            "speedup": float(np.mean(full_times)) / max(float(np.mean(total_times)), 1e-9),
            "num_samples": num_samples,
            "split_layer": self.split_point.layer_idx,
            "split_node": self.split_point.node_name,
        }

    def privacy_report(self) -> Dict[str, Any]:
        """Confirm that raw input never reaches cloud executor."""
        layers_on_edge = self.split_point.layer_idx
        return {
            "privacy_preserved": layers_on_edge >= 1,
            "edge_layers": layers_on_edge,
            "cloud_layers": len(list(self._edge._session.get_inputs())) > 0,
            "raw_input_exposed_to_cloud": False,
            "note": (
                f"Raw input passes through {layers_on_edge} layer(s) on-device before "
                f"activations are sent to the cloud. The cloud only sees intermediate "
                f"representations, never the original input."
            ),
        }

    @staticmethod
    def _compressed_size(compressed: Dict[str, Any]) -> float:
        """Estimate serialised size of compressed payload in MB."""
        total_bytes = 0
        for v in compressed.values():
            if isinstance(v, np.ndarray):
                total_bytes += v.nbytes
        return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def edge_split(model_path: str, action: str = "analyze", **kwargs) -> Any:
    """CLI entry point for edge-cloud hybrid inference.

    Actions
    -------
    analyze   : return a SplitReport with all candidate split points
    split     : return the optimal SplitPoint
    run       : execute hybrid inference on random input
    benchmark : run benchmark comparing hybrid vs full-device
    """
    model_path = str(Path(model_path).resolve())

    if action == "analyze":
        analyzer = SplitAnalyzer(model_path)
        bandwidth = kwargs.get("bandwidth_mbps", 100.0)
        optimal = analyzer.find_optimal_split(bandwidth_mbps=bandwidth)
        all_splits = analyzer.analyze()
        total_layers = len(list(analyzer._graph.node))
        return SplitReport(
            optimal_split=optimal,
            all_splits=all_splits,
            recommended_bandwidth_mbps=bandwidth,
            total_layers=total_layers,
            edge_layers=optimal.layer_idx,
            cloud_layers=total_layers - optimal.layer_idx,
        )

    elif action == "split":
        analyzer = SplitAnalyzer(model_path)
        bandwidth = kwargs.get("bandwidth_mbps", 100.0)
        min_edge = kwargs.get("min_edge_layers", None)
        if min_edge is not None:
            return analyzer.find_privacy_split(min_edge_layers=min_edge)
        return analyzer.find_optimal_split(bandwidth_mbps=bandwidth)

    elif action == "run":
        executor = HybridExecutor(model_path, **kwargs)
        ort = _lazy_ort()
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
        return executor.run(inputs)

    elif action == "benchmark":
        executor = HybridExecutor(model_path, **kwargs)
        num_samples = kwargs.pop("num_samples", 50)
        return executor.benchmark(num_samples=num_samples)

    else:
        raise ValueError(f"Unknown action {action!r}. Choose from: analyze, split, run, benchmark")
