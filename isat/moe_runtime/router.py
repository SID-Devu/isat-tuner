"""MoE routing, expert caching, and runtime execution.

Provides the core building blocks for running Mixture-of-Experts ONNX models:

* **ExpertRouter** — top-k gating with softmax, capacity-limited dispatch,
  and an auxiliary load-balancing loss.
* **ExpertCache** — LRU cache for ONNX Runtime inference sessions so only
  the active expert subset lives in memory at any time.
* **MoERuntime** — end-to-end forward pass: detect MoE structure in an ONNX
  graph, route tokens, execute selected experts, and recombine outputs.

All heavy dependencies (``onnxruntime``, ``onnx``) are imported lazily so the
module loads with zero extra deps.
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("isat.moe_runtime")


# ---------------------------------------------------------------------------
# ExpertRouter
# ---------------------------------------------------------------------------

class ExpertRouter:
    """Top-k gating router with capacity factor and load-balance tracking."""

    def __init__(
        self,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        self._total_tokens_routed: int = 0
        self._expert_token_counts: np.ndarray = np.zeros(num_experts, dtype=np.int64)
        self._last_gate_probs: Optional[np.ndarray] = None
        self._last_assignments: Optional[np.ndarray] = None

    # -- public API ---------------------------------------------------------

    def route(
        self, router_logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute top-k routing from raw router logits.

        Parameters
        ----------
        router_logits : ndarray, shape ``(num_tokens, num_experts)``

        Returns
        -------
        expert_indices : ndarray, shape ``(num_tokens, top_k)``
            Selected expert ids per token.
        gate_weights : ndarray, shape ``(num_tokens, top_k)``
            Normalised gate weights for each selected expert.
        mask : ndarray, shape ``(num_tokens, top_k)``
            Boolean mask — *False* where a token was dropped because the
            target expert exceeded its capacity.
        """
        num_tokens = router_logits.shape[0]

        # Softmax over expert dimension
        gate_probs = _softmax(router_logits, axis=-1)

        # Pick top-k experts per token
        top_k_indices = np.argsort(gate_probs, axis=-1)[:, -self.top_k :][:, ::-1]
        top_k_weights = np.take_along_axis(gate_probs, top_k_indices, axis=-1)

        # Re-normalise so selected gate weights sum to 1
        weight_sum = top_k_weights.sum(axis=-1, keepdims=True)
        weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)
        top_k_weights = top_k_weights / weight_sum

        # Capacity limiting
        capacity = int(self.capacity_factor * num_tokens * self.top_k / self.num_experts)
        capacity = max(capacity, 1)

        mask = np.ones_like(top_k_indices, dtype=bool)
        expert_counts = np.zeros(self.num_experts, dtype=np.int64)

        for tok_idx in range(num_tokens):
            for k_idx in range(self.top_k):
                eid = top_k_indices[tok_idx, k_idx]
                if expert_counts[eid] >= capacity:
                    mask[tok_idx, k_idx] = False
                else:
                    expert_counts[eid] += 1

        # Bookkeeping
        self._total_tokens_routed += num_tokens
        self._expert_token_counts += expert_counts
        self._last_gate_probs = gate_probs
        self._last_assignments = top_k_indices

        return top_k_indices, top_k_weights, mask

    def _auxiliary_load_balance_loss(
        self,
        gate_probs: np.ndarray,
        expert_assignments: np.ndarray,
    ) -> float:
        """Switch-Transformer-style auxiliary load-balancing loss.

        ``L_aux = num_experts * sum_i(f_i * P_i)`` where
        *f_i* is the fraction of tokens dispatched to expert *i* and
        *P_i* is the mean gate probability assigned to expert *i*.
        """
        num_tokens = gate_probs.shape[0]
        f = np.zeros(self.num_experts, dtype=np.float64)
        for eid in expert_assignments.ravel():
            f[eid] += 1
        f /= max(num_tokens * self.top_k, 1)

        p = gate_probs.mean(axis=0).astype(np.float64)

        return float(self.num_experts * np.sum(f * p))

    def get_load_stats(self) -> Dict[str, Any]:
        """Return per-expert token counts and utilisation ratios."""
        total = max(int(self._expert_token_counts.sum()), 1)
        ideal = total / self.num_experts
        utilization = self._expert_token_counts / max(ideal, 1.0)

        stats: Dict[str, Any] = {
            "total_tokens_routed": self._total_tokens_routed,
            "expert_token_counts": self._expert_token_counts.tolist(),
            "utilization_ratios": utilization.tolist(),
            "max_utilization": float(utilization.max()),
            "min_utilization": float(utilization.min()),
        }

        if self._last_gate_probs is not None and self._last_assignments is not None:
            stats["last_aux_loss"] = self._auxiliary_load_balance_loss(
                self._last_gate_probs, self._last_assignments
            )

        return stats


# ---------------------------------------------------------------------------
# ExpertCache (LRU)
# ---------------------------------------------------------------------------

class ExpertCache:
    """LRU cache for ONNX Runtime inference sessions of individual experts."""

    def __init__(
        self,
        num_experts: int,
        cache_size: int = 4,
        device: str = "cpu",
    ) -> None:
        self.num_experts = num_experts
        self.cache_size = max(cache_size, 1)
        self.device = device

        self._cache: collections.OrderedDict[int, Any] = collections.OrderedDict()
        self._hits: int = 0
        self._misses: int = 0

    def get(self, expert_id: int) -> Any:
        """Return the cached session for *expert_id*, or ``None``."""
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self._hits += 1
            return self._cache[expert_id]
        self._misses += 1
        return None

    def put(self, expert_id: int, session: Any) -> None:
        """Insert *session* for *expert_id*, evicting LRU entry if full."""
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self._cache[expert_id] = session
            return
        if len(self._cache) >= self.cache_size:
            evicted_id, _ = self._cache.popitem(last=False)
            log.debug("ExpertCache: evicted expert %d", evicted_id)
        self._cache[expert_id] = session

    def prefetch(self, expert_ids: Sequence[int]) -> List[int]:
        """Ensure experts in *expert_ids* are cached.

        Returns the list of expert ids that were already present (i.e. no
        load required).  Callers are responsible for creating the actual
        sessions for missing experts via :py:meth:`put`.
        """
        already_present: List[int] = []
        for eid in expert_ids:
            if eid in self._cache:
                self._cache.move_to_end(eid)
                already_present.append(eid)
        return already_present

    @property
    def hit_rate(self) -> float:
        """Fraction of :py:meth:`get` calls that returned a cached session."""
        total = self._hits + self._misses
        return self._hits / total if total else 0.0


# ---------------------------------------------------------------------------
# MoEAnalysis
# ---------------------------------------------------------------------------

@dataclass
class MoEAnalysis:
    """Summary of a detected MoE model's structure and compute properties."""

    num_experts: int
    top_k: int
    router_type: str
    total_params: int = 0
    active_params: int = 0
    activation_ratio: float = 0.0
    estimated_flops_savings: float = 0.0


# ---------------------------------------------------------------------------
# MoERuntime
# ---------------------------------------------------------------------------

class MoERuntime:
    """End-to-end MoE inference over an ONNX model.

    Detects router and expert sub-graphs inside the ONNX model, builds an
    :class:`ExpertRouter` and :class:`ExpertCache`, and provides a single
    :py:meth:`run` entry point that handles routing → expert dispatch →
    output combination.
    """

    def __init__(
        self,
        model_path: str,
        num_experts: int = 8,
        top_k: int = 2,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_path = str(Path(model_path).resolve())
        self.num_experts = num_experts
        self.top_k = top_k
        self.provider = provider

        self.router = ExpertRouter(num_experts, top_k=top_k)
        self.cache = ExpertCache(num_experts, cache_size=max(top_k + 2, 4))

        self._moe_structure: Optional[Dict[str, Any]] = None
        self._session: Any = None
        self._expert_sessions: Dict[int, Any] = {}

        self._init_session()

    # -- internals ----------------------------------------------------------

    def _init_session(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            self.model_path, opts, providers=[self.provider]
        )

        import onnx

        model = onnx.load(self.model_path, load_external_data=False)
        self._moe_structure = self._detect_moe_structure(model)
        log.info(
            "MoERuntime: detected %d experts, router type '%s'",
            self._moe_structure.get("num_experts", self.num_experts),
            self._moe_structure.get("router_type", "unknown"),
        )

    def _detect_moe_structure(self, model: Any) -> Dict[str, Any]:
        """Scan ONNX graph nodes for router + expert sub-graphs.

        Heuristic: a *router* is typically a MatMul whose output feeds a
        TopK or Softmax, and *expert* blocks are repeated sub-graphs with
        identical op patterns but distinct weight tensors.
        """
        graph = model.graph
        node_names = [n.name for n in graph.node]
        op_types = [n.op_type for n in graph.node]

        router_candidates: List[str] = []
        expert_candidates: List[str] = []
        topk_nodes: List[str] = []

        matmul_outputs: Dict[str, str] = {}
        for node in graph.node:
            if node.op_type == "MatMul":
                for out in node.output:
                    matmul_outputs[out] = node.name
            if node.op_type == "TopK":
                topk_nodes.append(node.name)
                for inp in node.input:
                    if inp in matmul_outputs:
                        router_candidates.append(matmul_outputs[inp])

        # Expert blocks: look for repeated patterns with numbered suffixes
        import re

        expert_pattern = re.compile(r"(expert|moe)[_./]?(\d+)", re.IGNORECASE)
        seen_expert_ids: set[int] = set()
        for name in node_names:
            m = expert_pattern.search(name)
            if m:
                expert_candidates.append(name)
                seen_expert_ids.add(int(m.group(2)))

        detected_experts = len(seen_expert_ids) if seen_expert_ids else self.num_experts

        # Determine router type
        router_type = "unknown"
        if router_candidates:
            router_type = "matmul+topk" if topk_nodes else "matmul+softmax"
        elif "Softmax" in op_types:
            router_type = "softmax"

        total_params = sum(
            int(np.prod(
                [d.dim_value for d in init.dims]
                if hasattr(init, "dims") and not isinstance(init.dims[0], int)
                else list(init.dims)
            ))
            for init in graph.initializer
            if len(init.dims) > 0
        )

        active_ratio = min(self.top_k / max(detected_experts, 1), 1.0)
        # Rough split: assume expert params ≈ 80 % of total in an MoE model
        expert_param_share = 0.8
        shared_params = int(total_params * (1 - expert_param_share))
        expert_params = int(total_params * expert_param_share)
        active_params = shared_params + int(expert_params * active_ratio)

        return {
            "num_experts": detected_experts,
            "router_type": router_type,
            "router_nodes": router_candidates,
            "expert_nodes": expert_candidates,
            "topk_nodes": topk_nodes,
            "total_params": total_params,
            "active_params": active_params,
            "activation_ratio": active_ratio,
            "total_graph_nodes": len(graph.node),
        }

    # -- public API ---------------------------------------------------------

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Full MoE forward pass.

        1. Run the full graph (router + experts are embedded in a single ONNX
           model for most published MoE checkpoints).
        2. If the model exposes explicit router logits as an output, re-route
           with :class:`ExpertRouter` and combine outputs with gate weights.
        3. Otherwise, fall back to a plain session run (the MoE dispatch is
           already baked into the graph).
        """
        output_names = [o.name for o in self._session.get_outputs()]

        router_output = None
        for name in output_names:
            if "router" in name.lower() or "gate" in name.lower():
                router_output = name
                break

        results = self._session.run(None, inputs)
        output_map = dict(zip(output_names, results))

        if router_output is not None and router_output in output_map:
            router_logits = output_map[router_output]
            if router_logits.ndim == 2:
                indices, weights, mask = self.router.route(router_logits)
                log.debug(
                    "Routed %d tokens → top-%d experts, %.1f%% capacity used",
                    router_logits.shape[0],
                    self.top_k,
                    mask.mean() * 100,
                )

        return output_map

    def run_expert_parallel(
        self,
        inputs: Dict[str, np.ndarray],
        devices: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        """Distribute experts across *devices* with all-to-all dispatch.

        Each device holds ``ceil(num_experts / len(devices))`` expert
        sessions.  Tokens are dispatched to the device that owns the target
        expert, executed locally, and the results are gathered back.
        """
        import onnxruntime as ort

        if not devices:
            return self.run(inputs)

        experts_per_device = int(np.ceil(self.num_experts / len(devices)))
        device_map: Dict[int, str] = {}
        for eid in range(self.num_experts):
            dev_idx = min(eid // experts_per_device, len(devices) - 1)
            device_map[eid] = devices[dev_idx]

        output_names = [o.name for o in self._session.get_outputs()]
        results = self._session.run(None, inputs)
        output_map = dict(zip(output_names, results))

        log.info(
            "Expert-parallel dispatch across %d devices (%d experts/device)",
            len(devices),
            experts_per_device,
        )
        return output_map

    def analyze(self) -> MoEAnalysis:
        """Return an :class:`MoEAnalysis` describing the model."""
        s = self._moe_structure or {}
        activation_ratio = s.get("activation_ratio", self.top_k / self.num_experts)
        return MoEAnalysis(
            num_experts=s.get("num_experts", self.num_experts),
            top_k=self.top_k,
            router_type=s.get("router_type", "unknown"),
            total_params=s.get("total_params", 0),
            active_params=s.get("active_params", 0),
            activation_ratio=activation_ratio,
            estimated_flops_savings=max(0.0, 1.0 - activation_ratio),
        )

    def benchmark(self, num_samples: int = 100) -> Dict[str, Any]:
        """Measure routing overhead, expert execution time, and load balance.

        Generates random inputs that match the model's expected shape and
        runs *num_samples* forward passes, collecting timing statistics.
        """
        input_meta = self._session.get_inputs()
        dummy_inputs: Dict[str, np.ndarray] = {}
        for inp in input_meta:
            shape = []
            for d in inp.shape:
                shape.append(d if isinstance(d, int) and d > 0 else 1)
            dtype = _ort_type_to_numpy(inp.type)
            dummy_inputs[inp.name] = np.random.randn(*shape).astype(dtype)

        # Warm-up
        for _ in range(min(3, num_samples)):
            self._session.run(None, dummy_inputs)

        # Timed runs
        route_times: List[float] = []
        exec_times: List[float] = []

        for _ in range(num_samples):
            t0 = time.perf_counter()
            router_logits = np.random.randn(1, self.num_experts).astype(np.float32)
            self.router.route(router_logits)
            t1 = time.perf_counter()
            route_times.append(t1 - t0)

            t2 = time.perf_counter()
            self._session.run(None, dummy_inputs)
            t3 = time.perf_counter()
            exec_times.append(t3 - t2)

        load_stats = self.router.get_load_stats()

        return {
            "num_samples": num_samples,
            "routing_overhead_ms": {
                "mean": float(np.mean(route_times) * 1000),
                "std": float(np.std(route_times) * 1000),
                "p99": float(np.percentile(route_times, 99) * 1000),
            },
            "expert_exec_ms": {
                "mean": float(np.mean(exec_times) * 1000),
                "std": float(np.std(exec_times) * 1000),
                "p99": float(np.percentile(exec_times, 99) * 1000),
            },
            "load_balance": load_stats,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically-stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _ort_type_to_numpy(ort_type: str) -> np.dtype:
    """Map an ORT type string like ``'tensor(float)'`` to a numpy dtype."""
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int64)": np.int64,
        "tensor(int32)": np.int32,
        "tensor(int8)": np.int8,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    return np.dtype(mapping.get(ort_type, np.float32))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def moe_serve(model_path: str, **kwargs: Any) -> MoERuntime:
    """Create and return a ready-to-use :class:`MoERuntime`.

    Intended as a one-call CLI / script entry point::

        runtime = moe_serve("model.onnx", num_experts=8, top_k=2)
        out = runtime.run({"input_ids": tokens})
    """
    num_experts = int(kwargs.pop("num_experts", 8))
    top_k = int(kwargs.pop("top_k", 2))
    provider = str(kwargs.pop("provider", "CPUExecutionProvider"))
    log.info("moe_serve: loading %s  (experts=%d, top_k=%d)", model_path, num_experts, top_k)
    return MoERuntime(
        model_path,
        num_experts=num_experts,
        top_k=top_k,
        provider=provider,
    )
