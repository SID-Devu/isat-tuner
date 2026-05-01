from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class ShardAnalysis:
    total_params: int
    total_size_mb: float
    num_layers: int
    layer_sizes: dict[str, float]
    recommended_shards: int
    memory_per_shard_mb: float


@dataclass
class ShardResult:
    success: bool
    num_shards: int
    shard_paths: list[str] = field(default_factory=list)
    shard_sizes_mb: list[float] = field(default_factory=list)
    strategy: str = "auto"
    elapsed_s: float = 0.0
    error: Optional[str] = None


class ModelSharder:
    TARGET_SHARD_SIZE_MB = 2048

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        logger.info("Loading model from %s", self.model_path)
        self._model = onnx.load(str(self.model_path), load_external_data=False)
        self._external_data_dir = self.model_path.parent
        self._load_external_data()

    def _load_external_data(self):
        try:
            onnx.load_external_data_for_model(self._model, str(self._external_data_dir))
        except Exception:
            logger.debug("No external data to load or already embedded")

    def analyze(self) -> ShardAnalysis:
        graph = self._model.graph
        layer_sizes: dict[str, float] = {}
        total_params = 0
        total_bytes = 0

        for initializer in graph.initializer:
            dims = list(initializer.dims)
            param_count = int(np.prod(dims)) if dims else 1
            elem_size = TensorProto.DataType.DESCRIPTOR.values_by_number[
                initializer.data_type
            ].name
            byte_size = param_count * numpy_helper.to_array(initializer).itemsize
            total_params += param_count
            total_bytes += byte_size
            layer_sizes[initializer.name] = byte_size / (1024 * 1024)

        total_size_mb = total_bytes / (1024 * 1024)
        num_layers = len(graph.node)
        recommended_shards = max(1, int(np.ceil(total_size_mb / self.TARGET_SHARD_SIZE_MB)))
        memory_per_shard_mb = total_size_mb / max(recommended_shards, 1)

        return ShardAnalysis(
            total_params=total_params,
            total_size_mb=total_size_mb,
            num_layers=num_layers,
            layer_sizes=layer_sizes,
            recommended_shards=recommended_shards,
            memory_per_shard_mb=memory_per_shard_mb,
        )

    def split(
        self, num_shards: int, output_dir: str | Path, strategy: str = "balanced"
    ) -> ShardResult:
        start_time = time.time()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if num_shards < 1:
            return ShardResult(
                success=False, num_shards=0, error="num_shards must be >= 1"
            )

        if num_shards == 1:
            dest = output_dir / "shard_0.onnx"
            onnx.save(self._model, str(dest))
            size_mb = dest.stat().st_size / (1024 * 1024)
            return ShardResult(
                success=True,
                num_shards=1,
                shard_paths=[str(dest)],
                shard_sizes_mb=[size_mb],
                strategy=strategy,
                elapsed_s=time.time() - start_time,
            )

        try:
            graph = self._model.graph
            split_points = self._find_split_points(graph, num_shards, strategy)
            logger.info("Split points determined: %d cuts", len(split_points))

            shard_paths = []
            shard_sizes = []
            nodes = list(graph.node)
            boundaries = [0] + split_points + [len(nodes)]

            for i in range(num_shards):
                start_idx = boundaries[i]
                end_idx = boundaries[i + 1]
                start_nodes = nodes[start_idx:start_idx + 1] if start_idx < len(nodes) else []
                end_nodes = nodes[end_idx - 1:end_idx] if end_idx <= len(nodes) else []

                shard_model = self._extract_subgraph(
                    graph, start_idx, end_idx, shard_idx=i
                )
                shard_path = output_dir / f"shard_{i}.onnx"
                onnx.save(shard_model, str(shard_path))
                size_mb = shard_path.stat().st_size / (1024 * 1024)
                shard_paths.append(str(shard_path))
                shard_sizes.append(size_mb)
                logger.info("Shard %d: %.2f MB (%d nodes)", i, size_mb, end_idx - start_idx)

            elapsed = time.time() - start_time
            return ShardResult(
                success=True,
                num_shards=num_shards,
                shard_paths=shard_paths,
                shard_sizes_mb=shard_sizes,
                strategy=strategy,
                elapsed_s=elapsed,
            )
        except Exception as e:
            logger.exception("Failed to split model")
            return ShardResult(
                success=False,
                num_shards=0,
                strategy=strategy,
                elapsed_s=time.time() - start_time,
                error=str(e),
            )

    def _find_split_points(
        self, graph: onnx.GraphProto, num_shards: int, strategy: str
    ) -> list[int]:
        nodes = list(graph.node)
        num_nodes = len(nodes)

        if num_shards >= num_nodes:
            return list(range(1, num_nodes))

        if strategy == "layer":
            return self._split_at_layer_boundaries(nodes, num_shards)
        elif strategy == "balanced":
            return self._split_balanced(graph, nodes, num_shards)
        else:
            return self._split_auto(graph, nodes, num_shards)

    def _split_at_layer_boundaries(
        self, nodes: list, num_shards: int
    ) -> list[int]:
        chunk_size = len(nodes) // num_shards
        return [chunk_size * i for i in range(1, num_shards)]

    def _split_balanced(
        self, graph: onnx.GraphProto, nodes: list, num_shards: int
    ) -> list[int]:
        initializer_map: dict[str, int] = {}
        for init in graph.initializer:
            arr = numpy_helper.to_array(init)
            initializer_map[init.name] = arr.nbytes

        node_weights: list[int] = []
        for node in nodes:
            weight = sum(initializer_map.get(inp, 0) for inp in node.input)
            node_weights.append(max(weight, 1))

        total_weight = sum(node_weights)
        target_per_shard = total_weight / num_shards

        split_points = []
        cumulative = 0
        next_target = target_per_shard

        for i, w in enumerate(node_weights[:-1], start=1):
            cumulative += w
            if cumulative >= next_target and len(split_points) < num_shards - 1:
                split_points.append(i)
                next_target += target_per_shard

        while len(split_points) < num_shards - 1:
            remaining = len(nodes) - (split_points[-1] if split_points else 0)
            split_points.append(
                (split_points[-1] if split_points else 0) + remaining // 2
            )

        return sorted(set(split_points))[:num_shards - 1]

    def _split_auto(
        self, graph: onnx.GraphProto, nodes: list, num_shards: int
    ) -> list[int]:
        output_to_consumers: dict[str, list[int]] = {}
        for idx, node in enumerate(nodes):
            for inp in node.input:
                output_to_consumers.setdefault(inp, []).append(idx)

        node_outputs: dict[int, set[str]] = {}
        for idx, node in enumerate(nodes):
            node_outputs[idx] = set(node.output)

        cut_costs: list[tuple[int, int]] = []
        for idx in range(1, len(nodes)):
            live_tensors = set()
            for prev_idx in range(idx):
                for out in nodes[prev_idx].output:
                    consumers = output_to_consumers.get(out, [])
                    if any(c >= idx for c in consumers):
                        live_tensors.add(out)
            cut_costs.append((idx, len(live_tensors)))

        cut_costs.sort(key=lambda x: x[1])

        balanced = self._split_balanced(graph, nodes, num_shards)
        candidate_cuts = []
        for target in balanced:
            best = min(
                cut_costs,
                key=lambda x: abs(x[0] - target) + x[1] * 0.5,
            )
            candidate_cuts.append(best[0])

        return sorted(set(candidate_cuts))[:num_shards - 1]

    def _extract_subgraph(
        self,
        graph: onnx.GraphProto,
        start_idx: int,
        end_idx: int,
        shard_idx: int,
    ) -> onnx.ModelProto:
        nodes = list(graph.node)[start_idx:end_idx]
        node_inputs: set[str] = set()
        node_outputs: set[str] = set()
        for node in nodes:
            node_inputs.update(node.input)
            node_outputs.update(node.output)

        all_model_inputs = {inp.name for inp in graph.input}
        all_model_outputs = {out.name for out in graph.output}
        all_initializers = {init.name for init in graph.initializer}

        external_inputs = (node_inputs - node_outputs) - all_initializers
        external_inputs -= {""}

        subgraph_outputs = set()
        all_nodes = list(graph.node)
        downstream_inputs: set[str] = set()
        for node in all_nodes[end_idx:]:
            downstream_inputs.update(node.input)
        subgraph_outputs = node_outputs & downstream_inputs
        subgraph_outputs |= node_outputs & all_model_outputs

        if not subgraph_outputs:
            subgraph_outputs = set(nodes[-1].output) if nodes else set()

        input_value_infos = []
        existing_inputs = {vi.name: vi for vi in graph.input}
        for name in sorted(external_inputs):
            if name in existing_inputs:
                input_value_infos.append(existing_inputs[name])
            else:
                vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                input_value_infos.append(vi)

        output_value_infos = []
        existing_outputs = {vi.name: vi for vi in graph.output}
        for name in sorted(subgraph_outputs):
            if name in existing_outputs:
                output_value_infos.append(existing_outputs[name])
            else:
                vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
                output_value_infos.append(vi)

        needed_initializers = []
        for init in graph.initializer:
            if init.name in node_inputs:
                needed_initializers.append(init)

        sub_graph = helper.make_graph(
            nodes,
            f"shard_{shard_idx}",
            input_value_infos,
            output_value_infos,
            initializer=needed_initializers,
        )

        opset_imports = list(self._model.opset_import)
        shard_model = helper.make_model(sub_graph, opset_imports=opset_imports)
        shard_model.ir_version = self._model.ir_version

        return shard_model

    def validate_shards(self, shard_dir: str | Path) -> bool:
        shard_dir = Path(shard_dir)
        shard_files = sorted(shard_dir.glob("shard_*.onnx"))

        if not shard_files:
            logger.error("No shard files found in %s", shard_dir)
            return False

        logger.info("Validating %d shards", len(shard_files))
        previous_outputs: dict[str, list] = {}

        for idx, shard_path in enumerate(shard_files):
            try:
                model = onnx.load(str(shard_path))
                onnx.checker.check_model(model)
            except Exception as e:
                logger.error("Shard %d failed validation: %s", idx, e)
                return False

            current_inputs = {inp.name for inp in model.graph.input}
            current_initializers = {init.name for init in model.graph.initializer}
            required_inputs = current_inputs - current_initializers

            if idx > 0:
                for req_input in required_inputs:
                    if req_input not in previous_outputs:
                        is_original_model_input = any(
                            inp.name == req_input
                            for inp in self._model.graph.input
                        )
                        if not is_original_model_input:
                            logger.error(
                                "Shard %d requires input '%s' not produced by previous shards",
                                idx,
                                req_input,
                            )
                            return False

            previous_outputs.update(
                {out.name: [] for out in model.graph.output}
            )
            logger.info("Shard %d validated successfully", idx)

        logger.info("All shards validated successfully")
        return True


class ShardedRunner:
    def __init__(self, shard_dir: str | Path, providers: Optional[list[str]] = None):
        shard_dir = Path(shard_dir)
        self.shard_paths = sorted(shard_dir.glob("shard_*.onnx"))

        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")

        self.providers = providers or ort.get_available_providers()
        self.sessions: list[ort.InferenceSession] = []

        logger.info("Loading %d shards for pipeline-parallel inference", len(self.shard_paths))
        for path in self.shard_paths:
            sess = ort.InferenceSession(str(path), providers=self.providers)
            self.sessions.append(sess)
            logger.debug("Loaded shard: %s", path.name)

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        current_inputs = dict(inputs)

        for idx, session in enumerate(self.sessions):
            session_inputs = {}
            input_names = {inp.name for inp in session.get_inputs()}

            for name in input_names:
                if name in current_inputs:
                    session_inputs[name] = current_inputs[name]
                else:
                    raise ValueError(
                        f"Shard {idx} requires input '{name}' which is not available. "
                        f"Available: {list(current_inputs.keys())}"
                    )

            output_names = [out.name for out in session.get_outputs()]
            results = session.run(output_names, session_inputs)

            for name, value in zip(output_names, results):
                current_inputs[name] = value

            logger.debug("Shard %d executed: %d outputs", idx, len(results))

        final_session = self.sessions[-1]
        final_output_names = [out.name for out in final_session.get_outputs()]
        return {name: current_inputs[name] for name in final_output_names}


def shard_model(
    model_path: str | Path,
    num_shards: int,
    output_dir: str | Path,
    strategy: str = "auto",
) -> ShardResult:
    """Split an ONNX model into shards for multi-GPU inference."""
    sharder = ModelSharder(model_path)
    analysis = sharder.analyze()
    logger.info(
        "Model: %.2f MB, %d params, %d layers — splitting into %d shards (%s)",
        analysis.total_size_mb,
        analysis.total_params,
        analysis.num_layers,
        num_shards,
        strategy,
    )
    result = sharder.split(num_shards, output_dir, strategy=strategy)

    if result.success:
        valid = sharder.validate_shards(output_dir)
        if not valid:
            result.success = False
            result.error = "Post-split validation failed"

    return result
