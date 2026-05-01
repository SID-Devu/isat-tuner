"""Merge and compose multiple ONNX models into a single graph."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

log = logging.getLogger("isat.merge")

SUPPORTED_AGGREGATIONS = ("concat", "mean", "max", "sum")


@dataclass
class MergeResult:
    success: bool
    output_path: str
    num_models_merged: int = 0
    total_nodes: int = 0
    total_size_mb: float = 0.0
    mode: str = ""
    elapsed_s: float = 0.0
    error: str = ""


def _elem_type_name(elem_type: int) -> str:
    return TensorProto.DataType.Name(elem_type)


def _get_value_info_shape(vi: onnx.ValueInfoProto) -> list[int | str]:
    shape = []
    for dim in vi.type.tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(dim.dim_value)
    return shape


def _get_value_info_elem_type(vi: onnx.ValueInfoProto) -> int:
    return vi.type.tensor_type.elem_type


class ModelMerger:
    """Merge, chain, or compose ONNX models."""

    def _rename_nodes(self, model: ModelProto, prefix: str) -> ModelProto:
        """Prefix all node names, tensor names, and I/O names to avoid collisions."""
        clone = ModelProto()
        clone.CopyFrom(model)

        graph = clone.graph
        name_map: dict[str, str] = {}

        for init in graph.initializer:
            new_name = f"{prefix}{init.name}"
            name_map[init.name] = new_name
            init.name = new_name

        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            old = vi.name
            new = f"{prefix}{old}"
            name_map[old] = new
            vi.name = new

        for node in graph.node:
            if node.name:
                node.name = f"{prefix}{node.name}"
            for i, inp in enumerate(node.input):
                node.input[i] = name_map.get(inp, f"{prefix}{inp}")
            for i, out in enumerate(node.output):
                node.output[i] = name_map.get(out, f"{prefix}{out}")

        return clone

    def _reconcile_opsets(self, models: list[ModelProto]) -> int:
        """Pick the highest opset version across all models."""
        versions: list[int] = []
        for m in models:
            for opset in m.opset_import:
                if opset.domain == "" or opset.domain == "ai.onnx":
                    versions.append(opset.version)
        return max(versions) if versions else 13

    def _auto_detect_connections(
        self,
        src_model: ModelProto,
        dst_model: ModelProto,
    ) -> dict[str, str]:
        """Match outputs of *src* to inputs of *dst* by compatible shape and type."""
        connections: dict[str, str] = {}
        dst_inputs = [
            vi
            for vi in dst_model.graph.input
            if vi.name not in {init.name for init in dst_model.graph.initializer}
        ]
        src_outputs = list(src_model.graph.output)

        used_dst: set[str] = set()
        for s_out in src_outputs:
            s_shape = _get_value_info_shape(s_out)
            s_type = _get_value_info_elem_type(s_out)
            for d_in in dst_inputs:
                if d_in.name in used_dst:
                    continue
                d_shape = _get_value_info_shape(d_in)
                d_type = _get_value_info_elem_type(d_in)
                if s_type == d_type and self._shapes_compatible(s_shape, d_shape):
                    connections[s_out.name] = d_in.name
                    used_dst.add(d_in.name)
                    break
        return connections

    @staticmethod
    def _shapes_compatible(
        a: list[int | str], b: list[int | str]
    ) -> bool:
        if len(a) != len(b):
            return False
        for da, db in zip(a, b):
            if isinstance(da, str) or isinstance(db, str):
                continue
            if da != 0 and db != 0 and da != db:
                return False
        return True

    def _merge_graphs(
        self,
        graphs: list[onnx.GraphProto],
        connections: dict[str, str],
        mode: str,
    ) -> onnx.GraphProto:
        """Combine multiple ONNX graphs into one, wiring them via *connections*."""
        all_nodes: list[onnx.NodeProto] = []
        all_initializers: list[TensorProto] = []
        all_value_info: list[onnx.ValueInfoProto] = []
        all_inputs: list[onnx.ValueInfoProto] = []
        all_outputs: list[onnx.ValueInfoProto] = []

        connected_targets = set(connections.values())
        connected_sources = set(connections.keys())

        init_names_seen: set[str] = set()
        for g in graphs:
            all_nodes.extend(g.node)
            for init in g.initializer:
                if init.name not in init_names_seen:
                    all_initializers.append(init)
                    init_names_seen.add(init.name)
            all_value_info.extend(g.value_info)

        for src_name, tgt_name in connections.items():
            for node in all_nodes:
                for i, inp in enumerate(node.input):
                    if inp == tgt_name:
                        node.input[i] = src_name

        for g in graphs:
            g_init_names = {init.name for init in g.initializer}
            for vi in g.input:
                if vi.name not in g_init_names and vi.name not in connected_targets:
                    all_inputs.append(vi)
            for vi in g.output:
                if mode == "chain" and vi.name in connected_sources:
                    all_value_info.append(vi)
                else:
                    all_outputs.append(vi)

        merged = helper.make_graph(
            all_nodes,
            "merged_graph",
            all_inputs,
            all_outputs,
            initializer=all_initializers,
        )
        merged.value_info.extend(all_value_info)
        return merged

    def chain(
        self,
        model_paths: Sequence[str | Path],
        output_path: str | Path,
        connections: list[dict[str, str]] | None = None,
    ) -> MergeResult:
        """Chain N models sequentially, piping outputs of model_i to inputs of model_{i+1}."""
        t0 = time.monotonic()
        output_path = str(output_path)
        paths = [str(p) for p in model_paths]

        if len(paths) < 2:
            return MergeResult(
                success=False,
                output_path=output_path,
                error="Need at least 2 models to chain.",
                mode="chain",
                elapsed_s=time.monotonic() - t0,
            )

        try:
            models = [onnx.load(p) for p in paths]
            opset = self._reconcile_opsets(models)

            renamed: list[ModelProto] = []
            for idx, m in enumerate(models):
                renamed.append(self._rename_nodes(m, f"m{idx}_"))
            models = renamed

            all_connections: dict[str, str] = {}
            for i in range(len(models) - 1):
                if connections and i < len(connections):
                    prefixed = {
                        f"m{i}_{k}": f"m{i+1}_{v}"
                        for k, v in connections[i].items()
                    }
                    all_connections.update(prefixed)
                else:
                    auto = self._auto_detect_connections(models[i], models[i + 1])
                    all_connections.update(auto)

            if not all_connections:
                return MergeResult(
                    success=False,
                    output_path=output_path,
                    error="Could not detect any connections between consecutive models.",
                    mode="chain",
                    elapsed_s=time.monotonic() - t0,
                )

            graph_list = [m.graph for m in models]
            merged_graph = self._merge_graphs(graph_list, all_connections, "chain")

            merged_model = helper.make_model(merged_graph)
            merged_model.ir_version = max(m.ir_version for m in models)
            del merged_model.opset_import[:]
            merged_model.opset_import.append(
                helper.make_opsetid("", opset)
            )

            onnx.save(merged_model, output_path)
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)

            return MergeResult(
                success=True,
                output_path=output_path,
                num_models_merged=len(paths),
                total_nodes=len(merged_graph.node),
                total_size_mb=round(size_mb, 2),
                mode="chain",
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        except Exception as exc:
            log.exception("Chain merge failed")
            return MergeResult(
                success=False,
                output_path=output_path,
                error=str(exc),
                mode="chain",
                elapsed_s=round(time.monotonic() - t0, 3),
            )

    def parallel(
        self,
        model_paths: Sequence[str | Path],
        output_path: str | Path,
        aggregation: str = "concat",
    ) -> MergeResult:
        """Run models in parallel on the same input and aggregate their outputs."""
        t0 = time.monotonic()
        output_path = str(output_path)
        paths = [str(p) for p in model_paths]

        if aggregation not in SUPPORTED_AGGREGATIONS:
            return MergeResult(
                success=False,
                output_path=output_path,
                error=f"Unsupported aggregation '{aggregation}'. Choose from {SUPPORTED_AGGREGATIONS}.",
                mode="parallel",
                elapsed_s=time.monotonic() - t0,
            )

        if len(paths) < 2:
            return MergeResult(
                success=False,
                output_path=output_path,
                error="Need at least 2 models for parallel merge.",
                mode="parallel",
                elapsed_s=time.monotonic() - t0,
            )

        try:
            models = [onnx.load(p) for p in paths]
            opset = self._reconcile_opsets(models)

            renamed = [self._rename_nodes(m, f"p{i}_") for i, m in enumerate(models)]
            models = renamed

            all_nodes: list[onnx.NodeProto] = []
            all_initializers: list[TensorProto] = []
            all_value_info: list[onnx.ValueInfoProto] = []
            init_names: set[str] = set()

            shared_inputs: list[onnx.ValueInfoProto] = []
            first_input_names: list[str] = []

            first_model_init_names = {init.name for init in models[0].graph.initializer}
            for vi in models[0].graph.input:
                if vi.name not in first_model_init_names:
                    shared_inputs.append(vi)
                    first_input_names.append(vi.name)

            for idx, m in enumerate(models):
                g = m.graph
                all_nodes.extend(g.node)
                for init in g.initializer:
                    if init.name not in init_names:
                        all_initializers.append(init)
                        init_names.add(init.name)
                all_value_info.extend(g.value_info)

                if idx > 0:
                    m_init_names = {init.name for init in g.initializer}
                    non_init_inputs = [
                        vi for vi in g.input if vi.name not in m_init_names
                    ]
                    for j, vi in enumerate(non_init_inputs):
                        if j < len(first_input_names):
                            old_name = vi.name
                            new_name = first_input_names[j]
                            for node in all_nodes:
                                for k, inp in enumerate(node.input):
                                    if inp == old_name:
                                        node.input[k] = new_name

            output_tensors: list[str] = []
            for m in models:
                for vi in m.graph.output:
                    output_tensors.append(vi.name)
                    all_value_info.append(vi)

            agg_output_name = "merged_output"
            if aggregation == "concat":
                agg_node = helper.make_node(
                    "Concat", output_tensors, [agg_output_name], axis=1
                )
            elif aggregation == "sum":
                agg_node = helper.make_node(
                    "Sum", output_tensors, [agg_output_name]
                )
            elif aggregation == "max":
                agg_node = helper.make_node(
                    "Max", output_tensors, [agg_output_name]
                )
            elif aggregation == "mean":
                sum_name = "_agg_sum"
                sum_node = helper.make_node("Sum", output_tensors, [sum_name])
                n_val = numpy_helper.from_array(
                    __import__("numpy").array(len(models), dtype="float32"),
                    name="_agg_n",
                )
                all_initializers.append(n_val)
                agg_node = helper.make_node("Div", [sum_name, "_agg_n"], [agg_output_name])
                all_nodes.append(sum_node)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

            all_nodes.append(agg_node)

            ref_out = models[0].graph.output[0]
            agg_output_vi = helper.make_tensor_value_info(
                agg_output_name,
                _get_value_info_elem_type(ref_out),
                None,
            )

            merged_graph = helper.make_graph(
                all_nodes,
                "parallel_merged",
                shared_inputs,
                [agg_output_vi],
                initializer=all_initializers,
            )
            merged_graph.value_info.extend(all_value_info)

            merged_model = helper.make_model(merged_graph)
            merged_model.ir_version = max(m.ir_version for m in models)
            del merged_model.opset_import[:]
            merged_model.opset_import.append(helper.make_opsetid("", opset))

            onnx.save(merged_model, output_path)
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)

            return MergeResult(
                success=True,
                output_path=output_path,
                num_models_merged=len(paths),
                total_nodes=len(merged_graph.node),
                total_size_mb=round(size_mb, 2),
                mode="parallel",
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        except Exception as exc:
            log.exception("Parallel merge failed")
            return MergeResult(
                success=False,
                output_path=output_path,
                error=str(exc),
                mode="parallel",
                elapsed_s=round(time.monotonic() - t0, 3),
            )

    def compose(
        self,
        model_a_path: str | Path,
        model_b_path: str | Path,
        output_path: str | Path,
        connections: dict[str, str],
    ) -> MergeResult:
        """General composition: connect specific outputs of A to inputs of B."""
        t0 = time.monotonic()
        output_path = str(output_path)
        try:
            model_a = onnx.load(str(model_a_path))
            model_b = onnx.load(str(model_b_path))
            opset = self._reconcile_opsets([model_a, model_b])

            model_a = self._rename_nodes(model_a, "a_")
            model_b = self._rename_nodes(model_b, "b_")

            prefixed_connections = {
                f"a_{k}": f"b_{v}" for k, v in connections.items()
            }

            merged_graph = self._merge_graphs(
                [model_a.graph, model_b.graph],
                prefixed_connections,
                "compose",
            )

            merged_model = helper.make_model(merged_graph)
            merged_model.ir_version = max(model_a.ir_version, model_b.ir_version)
            del merged_model.opset_import[:]
            merged_model.opset_import.append(helper.make_opsetid("", opset))

            onnx.save(merged_model, output_path)
            size_mb = Path(output_path).stat().st_size / (1024 * 1024)

            return MergeResult(
                success=True,
                output_path=output_path,
                num_models_merged=2,
                total_nodes=len(merged_graph.node),
                total_size_mb=round(size_mb, 2),
                mode="compose",
                elapsed_s=round(time.monotonic() - t0, 3),
            )
        except Exception as exc:
            log.exception("Compose merge failed")
            return MergeResult(
                success=False,
                output_path=output_path,
                error=str(exc),
                mode="compose",
                elapsed_s=round(time.monotonic() - t0, 3),
            )

    def validate(
        self,
        merged_path: str | Path,
        original_paths: Sequence[str | Path],
    ) -> bool:
        """Verify the merged model loads, passes ONNX checker, and has expected I/O count."""
        try:
            merged = onnx.load(str(merged_path))
            onnx.checker.check_model(merged)

            originals = [onnx.load(str(p)) for p in original_paths]
            expected_nodes = sum(len(m.graph.node) for m in originals)
            actual_nodes = len(merged.graph.node)

            if actual_nodes < expected_nodes:
                log.warning(
                    "Merged model has fewer nodes (%d) than sum of originals (%d); "
                    "this may indicate lost subgraphs.",
                    actual_nodes,
                    expected_nodes,
                )

            log.info(
                "Validation passed: %d nodes, %d inputs, %d outputs",
                actual_nodes,
                len(merged.graph.input),
                len(merged.graph.output),
            )
            return True
        except Exception as exc:
            log.error("Validation failed: %s", exc)
            return False


def merge_models(
    model_paths: Sequence[str | Path],
    output_path: str | Path,
    mode: str = "chain",
    **kwargs: Any,
) -> MergeResult:
    """Top-level entry point for CLI usage.

    Parameters
    ----------
    model_paths : sequence of paths
        ONNX files to merge.
    output_path : path
        Where to write the merged model.
    mode : ``"chain"`` | ``"parallel"`` | ``"compose"``
        Merge strategy.
    **kwargs
        Forwarded to the selected ``ModelMerger`` method
        (e.g. *connections*, *aggregation*).
    """
    merger = ModelMerger()

    if mode == "chain":
        return merger.chain(model_paths, output_path, **kwargs)
    if mode == "parallel":
        return merger.parallel(model_paths, output_path, **kwargs)
    if mode == "compose":
        if len(model_paths) != 2:
            return MergeResult(
                success=False,
                output_path=str(output_path),
                error="Compose mode requires exactly 2 model paths.",
                mode="compose",
            )
        return merger.compose(
            model_paths[0], model_paths[1], output_path, **kwargs
        )

    return MergeResult(
        success=False,
        output_path=str(output_path),
        error=f"Unknown mode '{mode}'. Choose from: chain, parallel, compose.",
        mode=mode,
    )
