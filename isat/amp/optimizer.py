"""Pareto-optimal mixed-precision search over per-layer precision profiles.

Given a PrecisionProfile (layer -> precision -> LayerPrecisionResult), finds
precision assignments that minimize latency subject to accuracy constraints,
using dynamic programming, greedy, or beam-search strategies.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from isat.amp.profiler import LayerPrecisionResult, PrecisionProfile, PrecisionProfiler

log = logging.getLogger("isat.amp")


@dataclass
class PrecisionAssignment:
    layer_precisions: Dict[str, str]
    total_mse: float
    total_latency_ms: float
    total_size_mb: float
    speedup_vs_fp32: float
    compression_ratio: float


class MixedPrecisionOptimizer:
    """Find Pareto-optimal or constrained-optimal mixed-precision assignments."""

    def __init__(self, profile: PrecisionProfile, model_path: Optional[str] = None) -> None:
        self.profile = profile
        self.model_path = model_path
        self._layers = list(profile.keys())

    def pareto_frontier(
        self,
        accuracy_metric: str = "mse",
        latency_metric: str = "latency_ms",
    ) -> list[PrecisionAssignment]:
        """Compute the Pareto frontier of precision assignments via DP.

        Uses the same DP formulation as ``_dp_search`` but without an error
        budget, collecting all non-dominated (error, latency) states across
        layers and returning the full Pareto set.
        """
        layers = self._layers
        precisions = self._available_precisions()
        if not layers or not precisions:
            return []

        # Seed DP with the first layer's options
        # frontier_states: list of (accumulated_error, accumulated_latency, assignments_dict)
        frontier: list[tuple[float, float, dict[str, str]]] = []
        first = layers[0]
        for p in precisions:
            if p not in self.profile[first]:
                continue
            r = self.profile[first][p]
            err = getattr(r, accuracy_metric)
            lat = getattr(r, latency_metric)
            frontier.append((err, lat, {first: p}))

        frontier = self._prune_dominated(frontier)

        for layer in layers[1:]:
            next_frontier: list[tuple[float, float, dict[str, str]]] = []
            for acc_err, acc_lat, assigns in frontier:
                for p in precisions:
                    if p not in self.profile[layer]:
                        continue
                    r = self.profile[layer][p]
                    new_err = acc_err + getattr(r, accuracy_metric)
                    new_lat = acc_lat + getattr(r, latency_metric)
                    new_assigns = {**assigns, layer: p}
                    next_frontier.append((new_err, new_lat, new_assigns))
            frontier = self._prune_dominated(next_frontier)

        fp32_lat = self._fp32_total(latency_metric)
        fp32_size = self._fp32_total("size_delta_mb")
        results = []
        for err, lat, assigns in frontier:
            size = self._total_metric(assigns, "size_delta_mb")
            results.append(PrecisionAssignment(
                layer_precisions=assigns,
                total_mse=err if accuracy_metric == "mse" else self._total_metric(assigns, "mse"),
                total_latency_ms=lat if latency_metric == "latency_ms" else self._total_metric(assigns, "latency_ms"),
                total_size_mb=size,
                speedup_vs_fp32=fp32_lat / lat if lat > 0 else 1.0,
                compression_ratio=fp32_size / size if size != 0 else 1.0,
            ))

        results.sort(key=lambda a: a.total_mse)
        return results

    def optimize(
        self,
        max_mse: float = 0.001,
        max_latency_ms: Optional[float] = None,
        strategy: str = "dp",
    ) -> PrecisionAssignment:
        """Find the fastest precision assignment within the error budget."""
        layers = self._layers
        precisions = self._available_precisions()

        if strategy == "dp":
            assigns = self._dp_search(layers, precisions, max_mse)
        elif strategy == "greedy":
            assigns = self._greedy_search(layers, precisions, max_mse)
        elif strategy == "beam":
            assigns = self._beam_search(layers, precisions, max_mse, beam_width=20)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. Use 'dp', 'greedy', or 'beam'.")

        if max_latency_ms is not None:
            lat = self._total_metric(assigns, "latency_ms")
            if lat > max_latency_ms:
                log.warning(
                    "Best assignment latency %.2fms exceeds limit %.2fms",
                    lat,
                    max_latency_ms,
                )

        return self._build_assignment(assigns)

    def _dp_search(
        self,
        layers: list[str],
        precisions: list[str],
        max_error: float,
    ) -> dict[str, str]:
        """Dynamic programming over layers ordered by sensitivity.

        State space:
            After processing layers[0..i], each state is a tuple
            (accumulated_mse, accumulated_latency).  The assignment dict
            records which precision was chosen for each layer.

        Transitions:
            For every existing state and every candidate precision for
            layer[i+1], compute the new (mse, latency) and add it if
            mse <= max_error.

        Pruning:
            After expanding layer i, discard dominated states — a state
            (e1, l1) is dominated by (e2, l2) if e2 <= e1 AND l2 <= l1
            (i.e. another state is both more accurate and faster).
            This keeps the state set polynomial in practice despite the
            exponential combination space.

        Final selection:
            Among surviving states with mse <= max_error, pick the one
            with the lowest latency.
        """
        if not layers:
            return {}

        # states: list of (mse, latency, {layer: precision})
        states: list[tuple[float, float, dict[str, str]]] = []
        first = layers[0]
        for p in precisions:
            if p not in self.profile[first]:
                continue
            r = self.profile[first][p]
            if r.mse <= max_error:
                states.append((r.mse, r.latency_ms, {first: p}))

        states = self._prune_dominated(states)

        for layer in layers[1:]:
            next_states: list[tuple[float, float, dict[str, str]]] = []
            for acc_mse, acc_lat, assigns in states:
                for p in precisions:
                    if p not in self.profile[layer]:
                        continue
                    r = self.profile[layer][p]
                    new_mse = acc_mse + r.mse
                    if new_mse > max_error:
                        continue
                    new_lat = acc_lat + r.latency_ms
                    new_assigns = {**assigns, layer: p}
                    next_states.append((new_mse, new_lat, new_assigns))
            states = self._prune_dominated(next_states)

            if not states:
                log.warning(
                    "No feasible states after layer %s; relaxing to fp32 for remaining layers",
                    layer,
                )
                best_partial = min(states, key=lambda s: s[1]) if states else None
                if best_partial is None:
                    assigns_fallback: dict[str, str] = {l: "fp32" for l in layers}
                    return assigns_fallback
                break

        if not states:
            return {l: "fp32" for l in layers}

        best = min(states, key=lambda s: s[1])
        result = best[2]

        for l in layers:
            if l not in result:
                result[l] = "fp32"

        return result

    def _greedy_search(
        self,
        layers: list[str],
        precisions: list[str],
        max_error: float,
    ) -> dict[str, str]:
        """Sort layers by sensitivity (ascending MSE at lowest precision), then
        greedily assign the most aggressive precision that stays within budget."""
        sensitivity = []
        for layer in layers:
            worst_mse = max(
                (self.profile[layer][p].mse for p in precisions if p in self.profile[layer]),
                default=0.0,
            )
            sensitivity.append((worst_mse, layer))
        sensitivity.sort()

        aggressive_first = [p for p in precisions if p != "fp32"]
        aggressive_first.sort(key=lambda p: self._precision_rank(p))

        assigns: dict[str, str] = {l: "fp32" for l in layers}
        remaining_budget = max_error

        for _, layer in sensitivity:
            for p in aggressive_first:
                if p not in self.profile[layer]:
                    continue
                r = self.profile[layer][p]
                if r.mse <= remaining_budget:
                    assigns[layer] = p
                    remaining_budget -= r.mse
                    break

        return assigns

    def _beam_search(
        self,
        layers: list[str],
        precisions: list[str],
        max_error: float,
        beam_width: int = 20,
    ) -> dict[str, str]:
        """Beam search: keep top-k states by latency at each layer expansion,
        subject to the MSE constraint."""
        if not layers:
            return {}

        beam: list[tuple[float, float, dict[str, str]]] = []
        first = layers[0]
        for p in precisions:
            if p not in self.profile[first]:
                continue
            r = self.profile[first][p]
            if r.mse <= max_error:
                beam.append((r.mse, r.latency_ms, {first: p}))
        beam.sort(key=lambda s: s[1])
        beam = beam[:beam_width]

        for layer in layers[1:]:
            candidates: list[tuple[float, float, dict[str, str]]] = []
            for acc_mse, acc_lat, assigns in beam:
                for p in precisions:
                    if p not in self.profile[layer]:
                        continue
                    r = self.profile[layer][p]
                    new_mse = acc_mse + r.mse
                    if new_mse > max_error:
                        continue
                    candidates.append((new_mse, acc_lat + r.latency_ms, {**assigns, layer: p}))
            candidates.sort(key=lambda s: s[1])
            beam = candidates[:beam_width]

        if not beam:
            return {l: "fp32" for l in layers}

        best = min(beam, key=lambda s: s[1])
        result = best[2]
        for l in layers:
            if l not in result:
                result[l] = "fp32"
        return result

    @staticmethod
    def _prune_dominated(
        states: list[tuple[float, float, dict[str, str]]],
    ) -> list[tuple[float, float, dict[str, str]]]:
        """Remove states where another state is both more accurate AND faster.

        A state (e1, l1) is dominated if there exists (e2, l2) with
        e2 <= e1 and l2 <= l1, with at least one strict inequality.
        Sorting by error then doing a single latency sweep gives O(n log n).
        """
        if not states:
            return []

        states.sort(key=lambda s: (s[0], s[1]))
        pruned: list[tuple[float, float, dict[str, str]]] = []
        best_lat = float("inf")

        for s in states:
            if s[1] < best_lat:
                pruned.append(s)
                best_lat = s[1]

        return pruned

    @staticmethod
    def _precision_rank(precision: str) -> int:
        """Lower rank = more aggressive quantization."""
        return {"int4": 0, "int8": 1, "fp16": 2, "fp32": 3}.get(precision, 99)

    def _available_precisions(self) -> list[str]:
        precs: set[str] = set()
        for layer_precs in self.profile.values():
            precs.update(layer_precs.keys())
        return sorted(precs, key=self._precision_rank)

    def _total_metric(self, assigns: dict[str, str], metric: str) -> float:
        total = 0.0
        for layer, prec in assigns.items():
            if layer in self.profile and prec in self.profile[layer]:
                total += getattr(self.profile[layer][prec], metric, 0.0)
        return total

    def _fp32_total(self, metric: str) -> float:
        total = 0.0
        for layer in self._layers:
            if "fp32" in self.profile[layer]:
                total += getattr(self.profile[layer]["fp32"], metric, 0.0)
        return total

    def _build_assignment(self, assigns: dict[str, str]) -> PrecisionAssignment:
        total_mse = self._total_metric(assigns, "mse")
        total_lat = self._total_metric(assigns, "latency_ms")
        total_size = self._total_metric(assigns, "size_delta_mb")
        fp32_lat = self._fp32_total("latency_ms")
        fp32_size = self._fp32_total("size_delta_mb")

        return PrecisionAssignment(
            layer_precisions=assigns,
            total_mse=total_mse,
            total_latency_ms=total_lat,
            total_size_mb=total_size,
            speedup_vs_fp32=fp32_lat / total_lat if total_lat > 0 else 1.0,
            compression_ratio=fp32_size / total_size if total_size != 0 else 1.0,
        )

    def apply(self, assignment: PrecisionAssignment, output_path: str) -> str:
        """Create the actual mixed-precision ONNX model."""
        import onnx
        from onnx import numpy_helper

        model = onnx.load(self._resolve_model_path())
        init_map = {init.name: init for init in model.graph.initializer}

        for node in model.graph.node:
            node_name = node.name or f"{node.op_type}_anon"
            precision = assignment.layer_precisions.get(node_name, "fp32")
            if precision == "fp32":
                continue

            for inp_name in node.input:
                if inp_name not in init_map:
                    continue
                init = init_map[inp_name]
                arr = numpy_helper.to_array(init).astype(np.float32)

                if precision == "fp16":
                    converted = arr.astype(np.float16).astype(np.float32)
                elif precision == "int8":
                    if arr.max() == arr.min():
                        converted = np.zeros_like(arr)
                    else:
                        scale = (arr.max() - arr.min()) / 255.0
                        zp = np.round(-arr.min() / scale).astype(np.int32)
                        q = np.clip(np.round(arr / scale) + zp, 0, 255).astype(np.uint8)
                        converted = (q.astype(np.float32) - zp) * scale
                elif precision == "int4":
                    flat = arr.flatten()
                    block = 128
                    pad = (block - len(flat) % block) % block
                    if pad:
                        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
                    blocks = flat.reshape(-1, block)
                    scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 7.0
                    scales = np.where(scales == 0, 1.0, scales)
                    q = np.clip(np.round(blocks / scales), -8, 7).astype(np.int8)
                    deq = (q.astype(np.float32) * scales).flatten()[: arr.size]
                    converted = deq.reshape(arr.shape)
                else:
                    continue

                new_tensor = numpy_helper.from_array(converted, name=init.name)
                init.CopyFrom(new_tensor)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        onnx.save(model, output_path)
        log.info("Mixed-precision model saved to %s", output_path)
        return output_path

    def _resolve_model_path(self) -> str:
        if self.model_path and os.path.isfile(self.model_path):
            return self.model_path
        raise RuntimeError(
            "MixedPrecisionOptimizer.apply() requires the original model path. "
            "Pass model_path to __init__ or use amp_profile()."
        )

    def visualize(self, assignments: list[PrecisionAssignment]) -> str:
        """Generate a text summary of one or more precision assignments."""
        lines: list[str] = []
        prec_symbols = {"fp32": "████", "fp16": "▓▓▓▓", "int8": "▒▒▒▒", "int4": "░░░░"}

        for idx, a in enumerate(assignments):
            lines.append(f"Assignment #{idx + 1}")
            lines.append(
                f"  MSE: {a.total_mse:.6f}  |  Latency: {a.total_latency_ms:.2f}ms  |  "
                f"Speedup: {a.speedup_vs_fp32:.2f}x  |  Compression: {a.compression_ratio:.2f}x"
            )
            lines.append(f"  {'Layer':<40} {'Precision':<8} {'Visual'}")
            lines.append(f"  {'-' * 40} {'-' * 8} {'-' * 6}")

            for layer, prec in sorted(a.layer_precisions.items()):
                symbol = prec_symbols.get(prec, "????")
                lines.append(f"  {layer[:40]:<40} {prec:<8} {symbol}")
            lines.append("")

            counts: dict[str, int] = {}
            for prec in a.layer_precisions.values():
                counts[prec] = counts.get(prec, 0) + 1
            summary_parts = [f"{prec}: {cnt}" for prec, cnt in sorted(counts.items())]
            lines.append(f"  Distribution: {', '.join(summary_parts)}")
            lines.append("")

        return "\n".join(lines)


def amp_profile(
    model_path: str,
    action: str = "profile",
    max_mse: float = 0.001,
    output_path: Optional[str] = None,
    strategy: str = "dp",
    precisions: Optional[list[str]] = None,
    num_samples: int = 50,
    **kwargs: Any,
) -> Any:
    """Top-level entry point for CLI and programmatic use.

    Parameters
    ----------
    model_path : path to ONNX model
    action : "profile", "optimize", or "pareto"
    max_mse : maximum allowable total MSE (for optimize)
    output_path : where to save the mixed-precision model (for optimize)
    strategy : "dp", "greedy", or "beam"
    precisions : list of precisions to profile
    num_samples : number of samples for profiling
    """
    log.info("AMP %s on %s", action, model_path)
    t0 = time.perf_counter()

    profiler = PrecisionProfiler(model_path, **{k: v for k, v in kwargs.items() if k == "provider"})
    profile = profiler.profile_all(precisions=precisions, num_samples=num_samples)

    if action == "profile":
        log.info("Profiling complete in %.1fs", time.perf_counter() - t0)
        return profile

    optimizer = MixedPrecisionOptimizer(profile, model_path=model_path)

    if action == "pareto":
        frontier = optimizer.pareto_frontier()
        log.info("Found %d Pareto-optimal assignments", len(frontier))
        print(optimizer.visualize(frontier))
        return frontier

    if action == "optimize":
        assignment = optimizer.optimize(max_mse=max_mse, strategy=strategy)
        log.info(
            "Optimal assignment: MSE=%.6f, latency=%.2fms, speedup=%.2fx",
            assignment.total_mse,
            assignment.total_latency_ms,
            assignment.speedup_vs_fp32,
        )
        print(optimizer.visualize([assignment]))

        if output_path:
            optimizer.apply(assignment, output_path)

        return assignment

    raise ValueError(f"Unknown action: {action!r}. Use 'profile', 'optimize', or 'pareto'.")
