"""Tensor-parallelism engine for multi-GPU ONNX inference.

Splits large transformer weights across N GPUs using two complementary
strategies:

* **Column-parallel** — the weight matrix is sliced along the *output*
  dimension (columns).  Each GPU independently computes its slice of the
  output; the partial results are concatenated before the next layer.
  Typical candidates: QKV projections and FFN up-projections.

* **Row-parallel** — the weight matrix is sliced along the *input*
  dimension (rows).  Each GPU computes a partial matrix-multiply whose
  results must be **all-reduced** (summed) to produce the correct output.
  Typical candidates: attention output projection and FFN down-projection.

Layers that are cheap and require the full tensor (LayerNorm, RMSNorm,
embeddings) are **replicated** on every GPU.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import numpy as np

log = logging.getLogger("isat.parallel.tensor_parallel")

ALLREDUCE_OP_TYPE = "isat.AllReduceMarker"


# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class ParallelPlan:
    """Describes how a model will be sharded across GPUs."""

    num_gpus: int
    column_parallel_layers: List[str] = field(default_factory=list)
    row_parallel_layers: List[str] = field(default_factory=list)
    replicated_layers: List[str] = field(default_factory=list)
    estimated_memory_per_gpu_mb: float = 0.0
    communication_volume_mb: float = 0.0


@dataclass
class ParallelResult:
    """Outcome of a tensor-parallel split or run."""

    success: bool
    num_gpus: int
    shard_paths: List[str] = field(default_factory=list)
    plan: Optional[ParallelPlan] = None
    elapsed_s: float = 0.0
    error: Optional[str] = None


# -----------------------------------------------------------------------
# TensorParallelizer
# -----------------------------------------------------------------------

class TensorParallelizer:
    """Analyze an ONNX model and produce per-GPU shards.

    Parameters
    ----------
    model_path : str | Path
        Path to the source ``.onnx`` file.
    num_gpus : int
        Number of GPUs to split across.
    provider : str
        ONNX Runtime execution provider used for inference later.
    """

    def __init__(
        self,
        model_path: str | Path,
        num_gpus: int = 2,
        provider: str = "CUDAExecutionProvider",
    ) -> None:
        self.model_path = Path(model_path)
        self.num_gpus = num_gpus
        self.provider = provider

        if num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")

        try:
            import onnx
        except ImportError as exc:
            raise ImportError(
                "onnx is required for tensor parallelism – pip install onnx"
            ) from exc

        self._onnx = onnx
        self._model = onnx.load(str(self.model_path))
        self._graph = self._model.graph
        self._weight_map = self._build_weight_map()

        log.info(
            "Loaded %s (%d nodes, %d initializers, target %d GPUs)",
            self.model_path.name,
            len(self._graph.node),
            len(self._graph.initializer),
            self.num_gpus,
        )

    # ------------------------------------------------------------------ analyze

    def analyze(self) -> ParallelPlan:
        """Return a :class:`ParallelPlan` without modifying anything."""
        if self.num_gpus == 1:
            return ParallelPlan(
                num_gpus=1,
                replicated_layers=[n.name for n in self._graph.node],
            )

        attn_qkv, attn_out = self._identify_attention_projections(self._graph)
        ffn_up, ffn_down = self._identify_ffn_layers(self._graph)

        # Column-parallel: QKV projections + FFN up-projections
        col_parallel = list({*attn_qkv, *ffn_up})
        # Row-parallel: output projections + FFN down-projections (need all-reduce)
        row_parallel = list({*attn_out, *ffn_down})

        split_names = {*col_parallel, *row_parallel}
        replicated = [
            n.name or f"node_{i}"
            for i, n in enumerate(self._graph.node)
            if (n.name or f"node_{i}") not in split_names
        ]

        total_bytes = sum(
            self._tensor_nbytes(init)
            for init in self._graph.initializer
        )
        per_gpu_bytes = total_bytes / self.num_gpus
        comm_bytes = sum(
            self._tensor_nbytes(self._weight_map[n])
            for n in row_parallel
            if n in self._weight_map
        )

        return ParallelPlan(
            num_gpus=self.num_gpus,
            column_parallel_layers=col_parallel,
            row_parallel_layers=row_parallel,
            replicated_layers=replicated,
            estimated_memory_per_gpu_mb=per_gpu_bytes / 1e6,
            communication_volume_mb=comm_bytes / 1e6,
        )

    # ------------------------------------------------------------------ split

    def split(self, output_dir: str | Path) -> ParallelResult:
        """Produce ``num_gpus`` ONNX shard files inside *output_dir*."""
        t0 = time.monotonic()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.num_gpus == 1:
            dst = output_dir / f"shard_0.onnx"
            self._onnx.save(self._model, str(dst))
            return ParallelResult(
                success=True,
                num_gpus=1,
                shard_paths=[str(dst)],
                plan=self.analyze(),
                elapsed_s=time.monotonic() - t0,
            )

        plan = self.analyze()

        try:
            shard_paths = self._create_shards(plan, output_dir)
            return ParallelResult(
                success=True,
                num_gpus=self.num_gpus,
                shard_paths=shard_paths,
                plan=plan,
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            log.error("Tensor-parallel split failed: %s", exc)
            return ParallelResult(
                success=False,
                num_gpus=self.num_gpus,
                plan=plan,
                elapsed_s=time.monotonic() - t0,
                error=str(exc),
            )

    # ------------------------------------------------------------------ internals

    def _build_weight_map(self) -> Dict[str, object]:
        """Map node names → their first initializer (weight tensor)."""
        init_names = {init.name for init in self._graph.initializer}
        mapping: Dict[str, object] = {}
        for node in self._graph.node:
            for inp in node.input:
                if inp in init_names:
                    mapping[node.name] = self._get_initializer(inp)
                    break
        return mapping

    def _get_initializer(self, name: str):
        for init in self._graph.initializer:
            if init.name == name:
                return init
        return None

    def _identify_attention_projections(self, graph) -> tuple[list, list]:
        """Find QKV and output-projection MatMul/Gemm nodes.

        Heuristic: a MatMul whose weight has shape (H, 3H) or (H, H) inside
        a sub-graph that also contains Softmax is likely an attention
        projection.  Nodes whose output feeds directly into a Softmax
        neighborhood are QKV; nodes *after* the Softmax neighborhood are the
        output projection.
        """
        qkv_nodes: list[str] = []
        out_nodes: list[str] = []

        softmax_inputs = set()
        softmax_outputs = set()
        for node in graph.node:
            if node.op_type == "Softmax":
                softmax_inputs.update(node.input)
                softmax_outputs.update(node.output)

        for node in graph.node:
            if node.op_type not in ("MatMul", "Gemm"):
                continue

            name = node.name or ""
            weight = self._weight_map.get(name)
            if weight is None:
                continue

            dims = list(weight.dims)
            if len(dims) != 2:
                continue

            feeds_softmax = any(o in softmax_inputs for o in node.output)
            after_softmax = any(i in softmax_outputs for i in node.input)

            name_lower = name.lower()
            is_qkv = (
                feeds_softmax
                or "qkv" in name_lower
                or "query" in name_lower
                or "key" in name_lower
                or "value" in name_lower
                or "q_proj" in name_lower
                or "k_proj" in name_lower
                or "v_proj" in name_lower
            )
            is_out = (
                after_softmax
                or "out_proj" in name_lower
                or "o_proj" in name_lower
                or "dense" in name_lower
            )

            if is_qkv:
                qkv_nodes.append(name)
            elif is_out:
                out_nodes.append(name)

        log.debug("Attention QKV nodes: %s", qkv_nodes)
        log.debug("Attention output nodes: %s", out_nodes)
        return qkv_nodes, out_nodes

    def _identify_ffn_layers(self, graph) -> tuple[list, list]:
        """Find FFN up-projection and down-projection nodes.

        Heuristic: two consecutive MatMul/Gemm nodes with a nonlinearity
        (Relu, Gelu, Silu) between them form an FFN block.  The first is
        the *up*-projection (column-parallel), the second is the
        *down*-projection (row-parallel).
        """
        up_nodes: list[str] = []
        down_nodes: list[str] = []

        activation_ops = {"Relu", "Gelu", "Silu", "Sigmoid", "Tanh", "FastGelu"}
        output_to_node: dict[str, object] = {}
        for node in graph.node:
            for o in node.output:
                output_to_node[o] = node

        matmul_nodes = [
            n for n in graph.node if n.op_type in ("MatMul", "Gemm")
        ]

        for node in matmul_nodes:
            name = node.name or ""
            name_lower = name.lower()

            if any(kw in name_lower for kw in ("up_proj", "gate_proj", "fc1", "wi")):
                up_nodes.append(name)
                continue
            if any(kw in name_lower for kw in ("down_proj", "fc2", "wo")):
                down_nodes.append(name)
                continue

            for inp in node.input:
                producer = output_to_node.get(inp)
                if producer is not None and producer.op_type in activation_ops:
                    down_nodes.append(name)
                    break

        log.debug("FFN up-projection nodes: %s", up_nodes)
        log.debug("FFN down-projection nodes: %s", down_nodes)
        return up_nodes, down_nodes

    @staticmethod
    def _split_weight(
        weight: np.ndarray, num_parts: int, dim: int
    ) -> List[np.ndarray]:
        """Divide *weight* into *num_parts* equal slices along *dim*."""
        if weight.shape[dim] % num_parts != 0:
            raise ValueError(
                f"Weight dim {dim} (size {weight.shape[dim]}) not divisible "
                f"by {num_parts}"
            )
        return list(np.split(weight, num_parts, axis=dim))

    def _insert_allreduce_marker(self, graph, after_node) -> None:
        """Insert a custom marker node after *after_node*.

        The marker carries domain ``isat.AllReduceMarker`` so that
        :class:`TensorParallelRunner` knows to perform an all-reduce at this
        point during inference.
        """
        from onnx import helper as oh

        marker_out = f"{after_node.output[0]}_allreduced"
        marker = oh.make_node(
            "Identity",
            inputs=[after_node.output[0]],
            outputs=[marker_out],
            name=f"AllReduce_{after_node.name}",
            domain=ALLREDUCE_OP_TYPE,
        )

        idx = list(graph.node).index(after_node)
        graph.node.insert(idx + 1, marker)

        for successor in graph.node:
            if successor == marker:
                continue
            for j, inp in enumerate(successor.input):
                if inp == after_node.output[0]:
                    successor.input[j] = marker_out

    def _create_shards(
        self, plan: ParallelPlan, output_dir: Path
    ) -> List[str]:
        """Build per-GPU ONNX model shards based on *plan*."""
        from onnx import TensorProto, helper as oh, numpy_helper

        shard_paths: List[str] = []

        for gpu_idx in range(self.num_gpus):
            model_copy = self._onnx.load(str(self.model_path))
            graph = model_copy.graph

            node_by_name = {n.name: n for n in graph.node}
            init_by_name = {i.name: i for i in graph.initializer}

            # Column-parallel: slice weight along output dim (dim=1 for typical [in, out])
            for layer_name in plan.column_parallel_layers:
                node = node_by_name.get(layer_name)
                if node is None:
                    continue
                for inp_name in node.input:
                    init = init_by_name.get(inp_name)
                    if init is None:
                        continue
                    w = numpy_helper.to_array(init)
                    if w.ndim != 2:
                        continue
                    parts = self._split_weight(w, self.num_gpus, dim=1)
                    new_init = numpy_helper.from_array(
                        parts[gpu_idx], name=init.name
                    )
                    self._replace_initializer(graph, init.name, new_init)

            # Row-parallel: slice weight along input dim (dim=0), runner will all-reduce the output
            for layer_name in plan.row_parallel_layers:
                node = node_by_name.get(layer_name)
                if node is None:
                    continue
                for inp_name in node.input:
                    init = init_by_name.get(inp_name)
                    if init is None:
                        continue
                    w = numpy_helper.to_array(init)
                    if w.ndim != 2:
                        continue
                    parts = self._split_weight(w, self.num_gpus, dim=0)
                    new_init = numpy_helper.from_array(
                        parts[gpu_idx], name=init.name
                    )
                    self._replace_initializer(graph, init.name, new_init)

                if gpu_idx == 0 and node is not None:
                    self._insert_allreduce_marker(graph, node)

            dst = output_dir / f"shard_{gpu_idx}.onnx"
            self._onnx.save(model_copy, str(dst))
            shard_paths.append(str(dst))
            log.info("Wrote shard %d → %s", gpu_idx, dst)

        return shard_paths

    @staticmethod
    def _replace_initializer(graph, name: str, new_init) -> None:
        for idx, init in enumerate(graph.initializer):
            if init.name == name:
                graph.initializer.remove(init)
                graph.initializer.insert(idx, new_init)
                return

    @staticmethod
    def _tensor_nbytes(tensor) -> int:
        if tensor is None:
            return 0
        from onnx import numpy_helper

        try:
            arr = numpy_helper.to_array(tensor)
            return arr.nbytes
        except Exception:
            elem_size = 4
            return int(np.prod(tensor.dims)) * elem_size if tensor.dims else 0


# -----------------------------------------------------------------------
# TensorParallelRunner
# -----------------------------------------------------------------------

class TensorParallelRunner:
    """Execute tensor-parallel ONNX shards across GPUs.

    Parameters
    ----------
    shard_dir : str | Path
        Directory containing ``shard_0.onnx``, ``shard_1.onnx``, …
    comm : DeviceComm | None
        Communication object.  Created automatically when *None*.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        comm=None,
    ) -> None:
        self.shard_dir = Path(shard_dir)

        shard_files = sorted(self.shard_dir.glob("shard_*.onnx"))
        if not shard_files:
            raise FileNotFoundError(
                f"No shard_*.onnx files found in {self.shard_dir}"
            )

        self.num_gpus = len(shard_files)

        from isat.parallel.comm import DeviceComm

        if comm is not None:
            self.comm = comm
        else:
            devices = [f"cuda:{i}" for i in range(self.num_gpus)]
            self.comm = DeviceComm(devices)

        import onnxruntime as ort

        self.sessions: List[ort.InferenceSession] = []
        for idx, sf in enumerate(shard_files):
            providers = self._providers_for_device(idx)
            sess = ort.InferenceSession(str(sf), providers=providers)
            self.sessions.append(sess)
            log.info("Loaded shard %d from %s", idx, sf.name)

        self._allreduce_outputs = self._scan_allreduce_markers()

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run inference across all shards, performing all-reduce where marked.

        *inputs* is a single dict of named numpy arrays.  Every shard
        receives the same inputs (except for column-parallel weights which
        are already baked into the shard).
        """
        if self.num_gpus == 1:
            return dict(
                zip(
                    [o.name for o in self.sessions[0].get_outputs()],
                    self.sessions[0].run(None, inputs),
                )
            )

        def _run_shard(idx: int) -> List[np.ndarray]:
            return self.sessions[idx].run(None, inputs)

        with ThreadPoolExecutor(max_workers=self.num_gpus) as pool:
            futures = [pool.submit(_run_shard, i) for i in range(self.num_gpus)]
            per_gpu_outputs = [f.result() for f in futures]

        output_names = [o.name for o in self.sessions[0].get_outputs()]
        merged: Dict[str, np.ndarray] = {}

        for out_idx, name in enumerate(output_names):
            tensors = [per_gpu_outputs[g][out_idx] for g in range(self.num_gpus)]
            if name in self._allreduce_outputs:
                # Row-parallel outputs: sum partial results across GPUs
                reduced = self.comm.all_reduce(tensors, op="sum")
                merged[name] = reduced[0]
            else:
                # Column-parallel outputs: concatenate along last dim
                merged[name] = np.concatenate(tensors, axis=-1)

        return merged

    def benchmark(
        self,
        inputs: Dict[str, np.ndarray],
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """Measure parallel throughput and compare with single-GPU baseline.

        Returns dict with keys ``parallel_avg_s``, ``single_avg_s``,
        ``speedup``.
        """
        for _ in range(2):
            self.run(inputs)

        t0 = time.monotonic()
        for _ in range(num_runs):
            self.run(inputs)
        parallel_avg = (time.monotonic() - t0) / num_runs

        t0 = time.monotonic()
        for _ in range(num_runs):
            self.sessions[0].run(None, inputs)
        single_avg = (time.monotonic() - t0) / num_runs

        speedup = single_avg / parallel_avg if parallel_avg > 0 else 0.0
        log.info(
            "Benchmark: parallel=%.4fs  single=%.4fs  speedup=%.2fx",
            parallel_avg,
            single_avg,
            speedup,
        )
        return {
            "parallel_avg_s": parallel_avg,
            "single_avg_s": single_avg,
            "speedup": speedup,
        }

    # ------------------------------------------------------------------ helpers

    def _providers_for_device(self, device_idx: int) -> list:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]

        if "CUDAExecutionProvider" in available:
            return [
                ("CUDAExecutionProvider", {"device_id": str(device_idx)}),
                "CPUExecutionProvider",
            ]
        if "ROCMExecutionProvider" in available:
            return [
                ("ROCMExecutionProvider", {"device_id": str(device_idx)}),
                "CPUExecutionProvider",
            ]
        return ["CPUExecutionProvider"]

    def _scan_allreduce_markers(self) -> set[str]:
        """Return output names that should be all-reduced."""
        markers: set[str] = set()
        try:
            import onnx

            model = onnx.load(str(sorted(self.shard_dir.glob("shard_*.onnx"))[0]))
            for node in model.graph.node:
                if node.domain == ALLREDUCE_OP_TYPE:
                    markers.update(node.output)
        except Exception:
            pass
        return markers


# -----------------------------------------------------------------------
# CLI convenience
# -----------------------------------------------------------------------

def tensor_parallel(
    model_path: str | Path,
    num_gpus: int = 2,
    output_dir: str | Path | None = None,
    action: Literal["analyze", "split"] = "split",
) -> ParallelPlan | ParallelResult:
    """High-level entry point for tensor-parallel splitting.

    Parameters
    ----------
    model_path : path
        Source ONNX model.
    num_gpus : int
        Target GPU count.
    output_dir : path | None
        Where to write shards.  Defaults to ``<model_dir>/tp_shards/``.
    action : ``"analyze"`` | ``"split"``
        ``"analyze"`` returns a :class:`ParallelPlan` without writing files.
        ``"split"`` produces shard files and returns a :class:`ParallelResult`.
    """
    model_path = Path(model_path)
    if output_dir is None:
        output_dir = model_path.parent / "tp_shards"

    tp = TensorParallelizer(model_path, num_gpus=num_gpus)

    if action == "analyze":
        return tp.analyze()
    return tp.split(output_dir)
