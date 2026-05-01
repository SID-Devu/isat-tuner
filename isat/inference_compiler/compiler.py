"""Inference compiler -- pattern-matching kernel fusion, memory planning, and graph-level optimization.

Provides:
  - PatternMatcher: discovers fusable operator subgraphs in an ONNX model
  - MemoryPlanner: computes tensor lifetimes and greedy bin-packing to minimise peak memory
  - InferenceCompiler: end-to-end compile pipeline (fuse + memory-optimise + benchmark)
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("isat.inference_compiler")


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass
class FusionPattern:
    name: str
    pattern_ops: List[str]
    replacement_op: str
    priority: int
    description: str


@dataclass
class MemoryPlan:
    peak_memory_bytes: int
    num_reuses: int
    savings_vs_naive_pct: float
    allocation_order: List[str] = field(default_factory=list)
    slot_assignments: Dict[str, int] = field(default_factory=dict)


@dataclass
class CompilerAnalysis:
    total_ops: int
    fusable_ops: int
    fusion_patterns_found: List[dict]
    estimated_speedup: float
    peak_memory_mb: float
    optimized_memory_mb: float
    memory_savings_pct: float


@dataclass
class CompilerResult:
    output_path: str
    original_ops: int
    optimized_ops: int
    fusions_applied: int
    original_latency_ms: float
    optimized_latency_ms: float
    speedup: float
    memory_reduction_pct: float


# ---------------------------------------------------------------------------
# PatternMatcher
# ---------------------------------------------------------------------------

class PatternMatcher:
    """Scan an ONNX graph for operator subgraphs that can be replaced by fused kernels."""

    def __init__(self) -> None:
        self._patterns: List[FusionPattern] = []
        for p in self._default_patterns():
            self.register_pattern(p)

    def register_pattern(self, pattern: FusionPattern) -> None:
        self._patterns.append(pattern)
        self._patterns.sort(key=lambda p: -p.priority)

    @staticmethod
    def _default_patterns() -> List[FusionPattern]:
        return [
            FusionPattern(
                name="gelu_fusion",
                pattern_ops=["MatMul", "Add", "Mul", "Tanh", "Add", "Mul"],
                replacement_op="FusedGELU",
                priority=10,
                description="MatMul + Add + Mul + Tanh + Add + Mul -> FusedGELU",
            ),
            FusionPattern(
                name="layer_norm_fusion",
                pattern_ops=[
                    "ReduceMean", "Sub", "Mul", "ReduceMean",
                    "Add", "Sqrt", "Div", "Mul", "Add",
                ],
                replacement_op="FusedLayerNorm",
                priority=20,
                description=(
                    "ReduceMean + Sub + Mul + ReduceMean + Add + Sqrt + Div + Mul + Add "
                    "-> FusedLayerNorm"
                ),
            ),
            FusionPattern(
                name="attention_fusion",
                pattern_ops=["MatMul", "Div", "Add", "Softmax", "MatMul"],
                replacement_op="FusedAttention",
                priority=30,
                description="MatMul + Div + Add + Softmax + MatMul -> FusedAttention",
            ),
            FusionPattern(
                name="bias_gelu",
                pattern_ops=["MatMul", "Add", "Gelu"],
                replacement_op="FusedBiasGELU",
                priority=15,
                description="MatMul + Add + GELU -> FusedBiasGELU",
            ),
            FusionPattern(
                name="skip_layernorm",
                pattern_ops=["Add", "LayerNormalization"],
                replacement_op="FusedSkipLayerNorm",
                priority=25,
                description="Add + LayerNorm -> FusedSkipLayerNorm",
            ),
            FusionPattern(
                name="qkv_fusion",
                pattern_ops=["MatMul", "MatMul", "MatMul"],
                replacement_op="FusedQKV",
                priority=35,
                description="Three parallel MatMul (Q, K, V) -> FusedQKV",
            ),
        ]

    # ---- graph scanning ---------------------------------------------------

    def find_matches(
        self,
        model: Any,
    ) -> List[Tuple[FusionPattern, List[Any], Dict[str, Any]]]:
        """Return ``[(pattern, matched_nodes, replacement_info), ...]``."""
        import onnx

        graph = model.graph
        node_list = list(graph.node)
        output_to_node: Dict[str, Any] = {}
        for node in node_list:
            for out in node.output:
                output_to_node[out] = node

        consumer_map: Dict[str, List[Any]] = defaultdict(list)
        for node in node_list:
            for inp in node.input:
                consumer_map[inp].append(node)

        used_nodes: set = set()
        matches: List[Tuple[FusionPattern, List[Any], Dict[str, Any]]] = []

        for pattern in self._patterns:
            for node in node_list:
                if id(node) in used_nodes:
                    continue
                if node.op_type != pattern.pattern_ops[0]:
                    continue

                matched = self._subgraph_match(
                    node, pattern.pattern_ops, graph,
                    consumer_map=consumer_map,
                    output_to_node=output_to_node,
                )
                if matched is None:
                    continue

                if any(id(n) in used_nodes for n in matched):
                    continue

                for n in matched:
                    used_nodes.add(id(n))

                external_inputs = []
                internal_outputs = {o for n in matched for o in n.output}
                for n in matched:
                    for inp in n.input:
                        if inp and inp not in internal_outputs:
                            external_inputs.append(inp)

                final_outputs = []
                matched_set = {id(n) for n in matched}
                for n in matched:
                    for out in n.output:
                        consumers = consumer_map.get(out, [])
                        if not consumers or any(id(c) not in matched_set for c in consumers):
                            final_outputs.append(out)

                replacement_info = {
                    "fused_op": pattern.replacement_op,
                    "inputs": external_inputs,
                    "outputs": final_outputs,
                }
                matches.append((pattern, matched, replacement_info))

        return matches

    # ---- DFS subgraph isomorphism -----------------------------------------

    @staticmethod
    def _subgraph_match(
        start_node: Any,
        pattern_ops: List[str],
        graph: Any,
        *,
        consumer_map: Dict[str, List[Any]] | None = None,
        output_to_node: Dict[str, Any] | None = None,
    ) -> Optional[List[Any]]:
        """DFS-based subgraph isomorphism check.

        Returns the list of matched nodes in topological order, or *None* on
        failure.
        """
        if not pattern_ops:
            return None

        if consumer_map is None:
            consumer_map = defaultdict(list)
            for node in graph.node:
                for inp in node.input:
                    consumer_map[inp].append(node)

        matched: List[Any] = []
        visited: set = set()

        def _dfs(node: Any, remaining: List[str]) -> bool:
            if not remaining:
                return True
            if node.op_type != remaining[0]:
                return False

            visited.add(id(node))
            matched.append(node)

            if len(remaining) == 1:
                return True

            for out in node.output:
                for consumer in consumer_map.get(out, []):
                    if id(consumer) in visited:
                        continue
                    if _dfs(consumer, remaining[1:]):
                        return True

            matched.pop()
            visited.discard(id(node))
            return False

        if _dfs(start_node, pattern_ops):
            return matched
        return None


# ---------------------------------------------------------------------------
# MemoryPlanner
# ---------------------------------------------------------------------------

class MemoryPlanner:
    """Compute tensor lifetimes and a greedy allocation plan to minimise peak memory."""

    def __init__(self) -> None:
        self._plan: Optional[MemoryPlan] = None

    def analyze(self, model: Any) -> MemoryPlan:
        """Compute memory layout for *model* and return a :class:`MemoryPlan`."""
        import onnx
        from onnx import numpy_helper

        graph = model.graph
        lifetimes = self._compute_tensor_lifetimes(graph)

        sizes: Dict[str, int] = {}
        initializer_names = {init.name for init in graph.initializer}

        for name in lifetimes:
            if name in initializer_names:
                for init in graph.initializer:
                    if init.name == name:
                        sizes[name] = int(np.prod(init.dims)) * 4
                        break
            else:
                sizes[name] = 4 * 1024

        plan = self._greedy_allocation(lifetimes, sizes)
        self._plan = plan
        return plan

    @staticmethod
    def _compute_tensor_lifetimes(graph: Any) -> Dict[str, Tuple[int, int]]:
        """Return ``{tensor_name: (first_op_idx, last_op_idx)}``."""
        lifetimes: Dict[str, List[int]] = defaultdict(list)

        for idx, node in enumerate(graph.node):
            for inp in node.input:
                if inp:
                    lifetimes[inp].append(idx)
            for out in node.output:
                if out:
                    lifetimes[out].append(idx)

        return {
            name: (min(indices), max(indices))
            for name, indices in lifetimes.items()
        }

    @staticmethod
    def _greedy_allocation(
        lifetimes: Dict[str, Tuple[int, int]],
        sizes: Dict[str, int],
    ) -> MemoryPlan:
        """Greedy bin-packing: assign tensors to slots, reusing when lifetimes don't overlap."""
        sorted_tensors = sorted(
            lifetimes.keys(),
            key=lambda t: sizes.get(t, 0),
            reverse=True,
        )

        slots: List[Tuple[int, int, int]] = []  # (end_time, size, slot_id)
        assignments: Dict[str, int] = {}
        reuses = 0
        naive_total = 0
        peak = 0

        for tensor in sorted_tensors:
            start, end = lifetimes[tensor]
            tsize = sizes.get(tensor, 0)
            naive_total += tsize

            assigned = False
            for i, (slot_end, slot_size, slot_id) in enumerate(slots):
                if slot_end < start and slot_size >= tsize:
                    slots[i] = (end, slot_size, slot_id)
                    assignments[tensor] = slot_id
                    reuses += 1
                    assigned = True
                    break

            if not assigned:
                slot_id = len(slots)
                slots.append((end, tsize, slot_id))
                assignments[tensor] = slot_id

        if slots:
            peak = sum(s[1] for s in slots)

        savings = 0.0
        if naive_total > 0:
            savings = (1.0 - peak / naive_total) * 100.0

        return MemoryPlan(
            peak_memory_bytes=peak,
            num_reuses=reuses,
            savings_vs_naive_pct=round(savings, 2),
            allocation_order=sorted_tensors,
            slot_assignments=assignments,
        )

    def optimize(self, model: Any) -> Any:
        """Reorder nodes via topological sort with memory-aware priority."""
        import onnx

        graph = model.graph
        lifetimes = self._compute_tensor_lifetimes(graph)

        output_to_node: Dict[str, int] = {}
        for idx, node in enumerate(graph.node):
            for out in node.output:
                output_to_node[out] = idx

        nodes = list(graph.node)
        n = len(nodes)
        in_degree: Dict[int, int] = {i: 0 for i in range(n)}
        successors: Dict[int, List[int]] = defaultdict(list)

        for idx, node in enumerate(nodes):
            for inp in node.input:
                pred = output_to_node.get(inp)
                if pred is not None and pred != idx:
                    in_degree[idx] += 1
                    successors[pred].append(idx)

        ready = sorted(
            [i for i, d in in_degree.items() if d == 0],
            key=lambda i: -self._mem_priority(nodes[i], lifetimes),
        )

        ordered: List[int] = []
        while ready:
            ready.sort(key=lambda i: -self._mem_priority(nodes[i], lifetimes))
            idx = ready.pop()
            ordered.append(idx)
            for succ in successors[idx]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    ready.append(succ)

        new_nodes = [nodes[i] for i in ordered]
        while len(graph.node):
            graph.node.pop()
        graph.node.extend(new_nodes)

        return model

    @staticmethod
    def _mem_priority(node: Any, lifetimes: Dict[str, Tuple[int, int]]) -> int:
        total_span = 0
        for out in node.output:
            if out in lifetimes:
                s, e = lifetimes[out]
                total_span += e - s
        return total_span

    def report(self) -> Optional[MemoryPlan]:
        return self._plan


# ---------------------------------------------------------------------------
# InferenceCompiler
# ---------------------------------------------------------------------------

class InferenceCompiler:
    """End-to-end inference compiler: fuse ops, plan memory, write optimised ONNX."""

    def __init__(self, model_path: str) -> None:
        import onnx

        self._model_path = str(model_path)
        self._model = onnx.load(self._model_path)
        self._pattern_matcher = PatternMatcher()
        self._memory_planner = MemoryPlanner()

    def analyze(self) -> CompilerAnalysis:
        graph = self._model.graph
        total_ops = len(graph.node)

        matches = self._pattern_matcher.find_matches(self._model)
        fusable_nodes: set = set()
        patterns_found: List[dict] = []
        for pattern, nodes, info in matches:
            for n in nodes:
                fusable_nodes.add(id(n))
            patterns_found.append({
                "pattern": pattern.name,
                "ops_fused": len(nodes),
                "replacement": pattern.replacement_op,
                "description": pattern.description,
            })

        fusable_ops = len(fusable_nodes)
        estimated_speedup = 1.0 + 0.15 * len(matches)

        mem_plan = self._memory_planner.analyze(self._model)
        peak_mb = mem_plan.peak_memory_bytes / (1024 * 1024)
        savings_pct = mem_plan.savings_vs_naive_pct
        optimized_mb = peak_mb * (1.0 - savings_pct / 100.0)

        return CompilerAnalysis(
            total_ops=total_ops,
            fusable_ops=fusable_ops,
            fusion_patterns_found=patterns_found,
            estimated_speedup=round(estimated_speedup, 2),
            peak_memory_mb=round(peak_mb, 2),
            optimized_memory_mb=round(optimized_mb, 2),
            memory_savings_pct=round(savings_pct, 2),
        )

    def compile(
        self,
        output_path: Optional[str] = None,
        *,
        enable_fusion: bool = True,
        enable_memory_opt: bool = True,
    ) -> CompilerResult:
        import onnx

        model = onnx.load(self._model_path)
        original_ops = len(model.graph.node)
        fusions_applied = 0

        if enable_fusion:
            matches = self._pattern_matcher.find_matches(model)
            model = self._apply_fusions(model, matches)
            fusions_applied = len(matches)

        if enable_memory_opt:
            plan = self._memory_planner.analyze(model)
            model = self._apply_memory_plan(model, plan)

        optimized_ops = len(model.graph.node)

        if output_path is None:
            stem = Path(self._model_path).stem
            output_path = str(
                Path(self._model_path).parent / f"{stem}_compiled.onnx"
            )
        onnx.save(model, output_path)

        original_latency = 0.0
        optimized_latency = 0.0
        speedup = 0.0
        try:
            original_latency = self._estimate_latency(self._model_path)
            optimized_latency = self._estimate_latency(output_path)
            speedup = (
                original_latency / optimized_latency
                if optimized_latency > 0
                else 0.0
            )
        except Exception:
            log.debug("Latency estimation skipped (onnxruntime unavailable)")

        mem_plan = self._memory_planner.report()
        mem_reduction = mem_plan.savings_vs_naive_pct if mem_plan else 0.0

        return CompilerResult(
            output_path=output_path,
            original_ops=original_ops,
            optimized_ops=optimized_ops,
            fusions_applied=fusions_applied,
            original_latency_ms=round(original_latency, 3),
            optimized_latency_ms=round(optimized_latency, 3),
            speedup=round(speedup, 2),
            memory_reduction_pct=round(mem_reduction, 2),
        )

    def benchmark(self, num_runs: int = 100) -> Dict[str, float]:
        """Compare original vs compiled model latency.

        Returns dict with *original_ms*, *compiled_ms*, *speedup*.
        """
        import onnx

        compiled_model = onnx.load(self._model_path)
        matches = self._pattern_matcher.find_matches(compiled_model)
        compiled_model = self._apply_fusions(compiled_model, matches)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            onnx.save(compiled_model, tmp_path)
            orig_ms = self._estimate_latency(self._model_path, num_runs=num_runs)
            comp_ms = self._estimate_latency(tmp_path, num_runs=num_runs)
        finally:
            os.unlink(tmp_path)

        speedup = orig_ms / comp_ms if comp_ms > 0 else 0.0
        return {
            "original_ms": round(orig_ms, 3),
            "compiled_ms": round(comp_ms, 3),
            "speedup": round(speedup, 2),
        }

    # ---- internal ---------------------------------------------------------

    @staticmethod
    def _apply_fusions(
        model: Any,
        matches: List[Tuple[FusionPattern, List[Any], Dict[str, Any]]],
    ) -> Any:
        """Replace matched subgraphs with fused operator nodes."""
        import onnx
        from onnx import helper

        graph = model.graph

        for pattern, matched_nodes, info in matches:
            matched_ids = {id(n) for n in matched_nodes}
            remaining = [n for n in graph.node if id(n) not in matched_ids]

            fused_node = helper.make_node(
                info["fused_op"],
                inputs=info["inputs"],
                outputs=info["outputs"],
                name=f"{pattern.name}_{id(matched_nodes[0])}",
            )

            insert_idx = 0
            for i, node in enumerate(remaining):
                for inp in info["inputs"]:
                    if inp in list(node.output):
                        insert_idx = max(insert_idx, i + 1)

            remaining.insert(insert_idx, fused_node)

            while len(graph.node):
                graph.node.pop()
            graph.node.extend(remaining)

        return model

    @staticmethod
    def _apply_memory_plan(model: Any, plan: MemoryPlan) -> Any:
        """Reorder graph nodes according to the memory plan allocation order."""
        graph = model.graph
        output_to_node: Dict[str, Any] = {}
        for node in graph.node:
            for out in node.output:
                output_to_node[out] = node

        ordered_nodes: List[Any] = []
        placed: set = set()

        def _place(node: Any) -> None:
            nid = id(node)
            if nid in placed:
                return
            for inp in node.input:
                dep = output_to_node.get(inp)
                if dep is not None and id(dep) not in placed:
                    _place(dep)
            placed.add(nid)
            ordered_nodes.append(node)

        for tensor_name in plan.allocation_order:
            node = output_to_node.get(tensor_name)
            if node is not None:
                _place(node)

        for node in graph.node:
            if id(node) not in placed:
                _place(node)

        while len(graph.node):
            graph.node.pop()
        graph.node.extend(ordered_nodes)

        return model

    @staticmethod
    def _estimate_latency(model_path: str, *, num_runs: int = 50) -> float:
        """Run the model through ORT and return mean latency in ms."""
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])

        feeds: Dict[str, np.ndarray] = {}
        for inp in sess.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            dtype = np.float32
            if inp.type and "float16" in inp.type:
                dtype = np.float16
            elif inp.type and "int" in inp.type:
                dtype = np.int64
            feeds[inp.name] = np.random.randn(*shape).astype(dtype)

        for _ in range(max(5, num_runs // 10)):
            sess.run(None, feeds)

        t0 = time.perf_counter()
        for _ in range(num_runs):
            sess.run(None, feeds)
        elapsed = (time.perf_counter() - t0) / num_runs * 1000.0
        return elapsed


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def compile_model(
    model_path: str,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> CompilerResult:
    """CLI entry point: load, compile, and save an optimised ONNX model."""
    compiler = InferenceCompiler(model_path)
    return compiler.compile(output_path=output_path, **kwargs)
