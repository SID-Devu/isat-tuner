"""Architecture-level model transformations on ONNX graphs.

Supports attention head pruning, hidden-dimension shrinking, depth
shrinking (layer removal), and vocabulary pruning — all operating
directly on the ONNX protobuf without requiring the original framework.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("isat.arch_convert")

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArchAnalysis:
    num_layers: int = 0
    num_heads: int = 0
    hidden_dim: int = 0
    ffn_dim: int = 0
    vocab_size: int = 0
    total_params: int = 0
    layer_breakdown: dict = field(default_factory=dict)


@dataclass
class PruneResult:
    success: bool = False
    method: str = ""
    original_params: int = 0
    pruned_params: int = 0
    reduction_ratio: float = 0.0
    output_path: str = ""
    elapsed_s: float = 0.0
    error: str = ""


@dataclass
class ShrinkResult:
    success: bool = False
    method: str = ""
    original_dim: int = 0
    new_dim: int = 0
    original_params: int = 0
    new_params: int = 0
    reduction_ratio: float = 0.0
    output_path: str = ""
    elapsed_s: float = 0.0


@dataclass
class VocabPruneResult:
    success: bool = False
    original_vocab: int = 0
    new_vocab: int = 0
    removed_tokens: int = 0
    remap_table_path: str = ""
    output_path: str = ""


# ---------------------------------------------------------------------------
# Pattern constants for transformer block detection
# ---------------------------------------------------------------------------

_QKV_PATTERNS = [
    re.compile(r"(layer|block|h)[\._](\d+)[\._].*(q_proj|k_proj|v_proj|query|key|value|qkv)", re.I),
    re.compile(r"encoder[\._](\d+)[\._].*(self_attn|attention)[\._].*(q|k|v)", re.I),
    re.compile(r"transformer[\._]h[\._](\d+)[\._]attn[\._]c_attn", re.I),
]

_FFN_PATTERNS = [
    re.compile(r"(layer|block|h)[\._](\d+)[\._].*(fc[12]|dense|mlp|ffn|intermediate|gate_proj|up_proj|down_proj)", re.I),
    re.compile(r"transformer[\._]h[\._](\d+)[\._]mlp", re.I),
]

_LAYERNORM_PATTERNS = [
    re.compile(r"(layer|block|h)[\._](\d+)[\._].*(layernorm|layer_norm|ln|norm[12]|input_layernorm|post_attention_layernorm)", re.I),
]

_EMBED_PATTERNS = [
    re.compile(r"(embed|wte|wpe|word_embed|token_embed|position_embed)", re.I),
]

_OUTPUT_PROJ_PATTERNS = [
    re.compile(r"(lm_head|cls|classifier|output[\._]proj|vocab_proj)", re.I),
]


def _count_params(model) -> int:
    import onnx
    total = 0
    for init in model.graph.initializer:
        total += int(np.prod(init.dims)) if init.dims else 0
    return total


def _initializer_to_numpy(init) -> np.ndarray:
    import onnx
    return np.frombuffer(init.raw_data, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type]).reshape(init.dims).copy()


def _numpy_to_initializer(name: str, arr: np.ndarray, data_type: int):
    import onnx
    init = onnx.TensorProto()
    init.name = name
    init.data_type = data_type
    init.dims[:] = arr.shape
    init.raw_data = arr.tobytes()
    return init


def _find_initializer(model, name: str):
    for init in model.graph.initializer:
        if init.name == name:
            return init
    return None


# ---------------------------------------------------------------------------
# ArchitectureConverter
# ---------------------------------------------------------------------------

class ArchitectureConverter:

    def __init__(self, model_path: str):
        import onnx

        self._path = Path(model_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        log.info("Loading ONNX model from %s", model_path)
        self._model = onnx.load(str(self._path))
        self._graph = self._model.graph

        self._init_map: dict[str, int] = {
            init.name: idx for idx, init in enumerate(self._graph.initializer)
        }

        self._layer_info = self._detect_architecture()

    # --------------------------------------------------------------------- #
    # Architecture detection
    # --------------------------------------------------------------------- #

    def _detect_architecture(self) -> dict:
        info: dict = {
            "layers": {},
            "embed_names": [],
            "output_proj_names": [],
            "layernorm_names": [],
        }

        for init in self._graph.initializer:
            name = init.name

            for pat in _QKV_PATTERNS:
                m = pat.search(name)
                if m:
                    groups = m.groups()
                    layer_idx = self._extract_layer_idx(groups)
                    info["layers"].setdefault(layer_idx, {"qkv": [], "ffn": [], "norms": []})
                    info["layers"][layer_idx]["qkv"].append(name)
                    break

            for pat in _FFN_PATTERNS:
                m = pat.search(name)
                if m:
                    groups = m.groups()
                    layer_idx = self._extract_layer_idx(groups)
                    info["layers"].setdefault(layer_idx, {"qkv": [], "ffn": [], "norms": []})
                    info["layers"][layer_idx]["ffn"].append(name)
                    break

            for pat in _LAYERNORM_PATTERNS:
                m = pat.search(name)
                if m:
                    groups = m.groups()
                    layer_idx = self._extract_layer_idx(groups)
                    info["layers"].setdefault(layer_idx, {"qkv": [], "ffn": [], "norms": []})
                    info["layers"][layer_idx]["norms"].append(name)
                    info["layernorm_names"].append(name)
                    break

            for pat in _EMBED_PATTERNS:
                if pat.search(name):
                    info["embed_names"].append(name)
                    break

            for pat in _OUTPUT_PROJ_PATTERNS:
                if pat.search(name):
                    info["output_proj_names"].append(name)
                    break

        return info

    @staticmethod
    def _extract_layer_idx(groups: tuple) -> int:
        for g in groups:
            if g is not None and g.isdigit():
                return int(g)
        return 0

    # --------------------------------------------------------------------- #
    # Analyze
    # --------------------------------------------------------------------- #

    def analyze(self) -> ArchAnalysis:
        layers = self._layer_info["layers"]
        num_layers = len(layers)

        hidden_dim = 0
        num_heads = 0
        ffn_dim = 0
        vocab_size = 0

        if layers:
            first_layer = next(iter(layers.values()))
            for qkv_name in first_layer.get("qkv", []):
                init = _find_initializer(self._model, qkv_name)
                if init is None:
                    continue
                dims = list(init.dims)
                if len(dims) == 2:
                    # QKV projection: [hidden_dim, head_dim] or [hidden_dim, hidden_dim]
                    # For fused QKV: [hidden_dim, 3*hidden_dim]
                    hidden_dim = max(hidden_dim, dims[0])
                    out_dim = dims[1]
                    if out_dim == 3 * dims[0]:
                        num_heads = self._infer_num_heads(dims[0], out_dim // 3)
                    elif out_dim == dims[0]:
                        num_heads = max(num_heads, 1)
                    break

            if num_heads == 0 and hidden_dim > 0:
                for head_size in [64, 80, 96, 128]:
                    if hidden_dim % head_size == 0:
                        num_heads = hidden_dim // head_size
                        break

            for ffn_name in first_layer.get("ffn", []):
                init = _find_initializer(self._model, ffn_name)
                if init is None:
                    continue
                dims = list(init.dims)
                if len(dims) == 2:
                    ffn_dim = max(ffn_dim, max(dims))
                    break

        for emb_name in self._layer_info["embed_names"]:
            init = _find_initializer(self._model, emb_name)
            if init is not None and len(init.dims) == 2:
                vocab_size = max(vocab_size, init.dims[0])

        total_params = _count_params(self._model)

        breakdown = {}
        for idx, layer_data in sorted(layers.items()):
            layer_params = 0
            for group in ("qkv", "ffn", "norms"):
                for name in layer_data.get(group, []):
                    init = _find_initializer(self._model, name)
                    if init:
                        layer_params += int(np.prod(init.dims))
            breakdown[f"layer_{idx}"] = layer_params

        return ArchAnalysis(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            ffn_dim=ffn_dim,
            vocab_size=vocab_size,
            total_params=total_params,
            layer_breakdown=breakdown,
        )

    @staticmethod
    def _infer_num_heads(hidden_dim: int, proj_dim: int) -> int:
        for head_size in [64, 80, 96, 128]:
            if proj_dim % head_size == 0:
                return proj_dim // head_size
        return max(1, proj_dim // 64)

    # --------------------------------------------------------------------- #
    # Attention head pruning
    # --------------------------------------------------------------------- #

    def prune_heads(
        self,
        heads_to_prune: Optional[dict[int, list[int]]] = None,
        num_heads_to_keep: Optional[int] = None,
        importance_method: str = "magnitude",
        output_path: Optional[str] = None,
    ) -> PruneResult:
        t0 = time.monotonic()
        original_params = _count_params(self._model)

        try:
            analysis = self.analyze()
            if analysis.num_heads == 0:
                return PruneResult(error="Could not detect attention heads")

            if heads_to_prune is None:
                if num_heads_to_keep is None:
                    num_heads_to_keep = max(1, analysis.num_heads // 2)

                scores = self._compute_head_importance(importance_method)
                heads_to_prune = {}
                for layer_idx, layer_scores in scores.items():
                    ranked = sorted(range(len(layer_scores)), key=lambda i: layer_scores[i])
                    num_remove = analysis.num_heads - num_heads_to_keep
                    if num_remove > 0:
                        heads_to_prune[layer_idx] = ranked[:num_remove]

            log.info("Pruning heads: %s", heads_to_prune)
            for layer_idx, head_indices in heads_to_prune.items():
                self._remove_heads(self._graph, layer_idx, head_indices)

            pruned_params = _count_params(self._model)
            out = self._save(output_path, suffix="_head_pruned")

            return PruneResult(
                success=True,
                method=importance_method,
                original_params=original_params,
                pruned_params=pruned_params,
                reduction_ratio=1.0 - pruned_params / max(original_params, 1),
                output_path=str(out),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            log.exception("Head pruning failed")
            return PruneResult(
                error=str(exc),
                method=importance_method,
                original_params=original_params,
                elapsed_s=time.monotonic() - t0,
            )

    def _compute_head_importance(
        self,
        method: str,
        num_samples: int = 100,
    ) -> dict[int, list[float]]:
        """Compute per-head importance scores for every transformer layer.

        For "magnitude": the L2 norm of the QKV weight slice belonging to each head.
        For "entropy" / "taylor": falls back to magnitude (requires runtime data).
        """
        analysis = self.analyze()
        scores: dict[int, list[float]] = {}

        for layer_idx, layer_data in sorted(self._layer_info["layers"].items()):
            qkv_names = layer_data.get("qkv", [])
            if not qkv_names:
                continue

            head_scores = [0.0] * analysis.num_heads

            if method in ("magnitude", "entropy", "taylor"):
                for qkv_name in qkv_names:
                    init = _find_initializer(self._model, qkv_name)
                    if init is None or len(init.dims) != 2:
                        continue
                    W = _initializer_to_numpy(init)
                    out_dim = W.shape[1]
                    head_dim = out_dim // analysis.num_heads
                    if head_dim == 0:
                        continue

                    # W is [hidden_dim, out_dim].  Slice along axis-1 into
                    # num_heads chunks of width head_dim — each chunk is one
                    # head's projection weights.
                    for h in range(analysis.num_heads):
                        start = h * head_dim
                        end = start + head_dim
                        head_scores[h] += float(np.linalg.norm(W[:, start:end]))

            if method == "entropy":
                log.warning(
                    "Entropy importance requires calibration data; "
                    "falling back to magnitude scoring"
                )
            elif method == "taylor":
                log.warning(
                    "Taylor importance requires calibration data; "
                    "falling back to magnitude scoring"
                )

            scores[layer_idx] = head_scores

        return scores

    def _remove_heads(
        self,
        graph,
        layer_idx: int,
        head_indices: list[int],
    ) -> None:
        """Surgically remove attention heads from QKV weight matrices.

        Transformer QKV projections store all heads contiguously:

            W_qkv  shape = [hidden_dim, num_heads * head_dim]

        Each head occupies columns [h*head_dim : (h+1)*head_dim].  To prune
        head ``h`` we delete those columns from every Q/K/V weight *and* the
        matching rows from the output projection (Wo) that recombines heads
        back to hidden_dim.

        After pruning, the surviving weight shapes become:

            W_qkv  -> [hidden_dim, remaining_heads * head_dim]
            W_o    -> [remaining_heads * head_dim, hidden_dim]
        """
        import onnx

        layer_data = self._layer_info["layers"].get(layer_idx)
        if not layer_data:
            log.warning("Layer %d not found — skipping head removal", layer_idx)
            return

        analysis = self.analyze()
        num_heads = analysis.num_heads
        if num_heads == 0:
            return

        keep_heads = sorted(set(range(num_heads)) - set(head_indices))

        for qkv_name in layer_data.get("qkv", []):
            init = _find_initializer(self._model, qkv_name)
            if init is None or len(init.dims) != 2:
                continue

            W = _initializer_to_numpy(init)
            out_dim = W.shape[1]
            head_dim = out_dim // num_heads
            if head_dim == 0:
                continue

            keep_cols = []
            for h in keep_heads:
                keep_cols.extend(range(h * head_dim, (h + 1) * head_dim))

            W_new = W[:, keep_cols]
            idx = self._init_map[qkv_name]
            self._graph.initializer[idx].CopyFrom(
                _numpy_to_initializer(qkv_name, W_new, init.data_type)
            )

        self._prune_output_projection(layer_idx, keep_heads, num_heads)
        self._prune_attention_bias(layer_data, keep_heads, num_heads)

    def _prune_output_projection(
        self,
        layer_idx: int,
        keep_heads: list[int],
        num_heads: int,
    ) -> None:
        """Shrink the output projection (Wo) that merges head outputs.

        Wo has shape [num_heads * head_dim, hidden_dim].  We keep only the
        rows corresponding to surviving heads.
        """
        out_proj_pat = re.compile(
            rf"(layer|block|h|encoder)[\._]{layer_idx}[\._].*(o_proj|out_proj|dense|c_proj)",
            re.I,
        )
        for init in self._graph.initializer:
            if not out_proj_pat.search(init.name):
                continue
            if len(init.dims) != 2:
                continue
            W = _initializer_to_numpy(init)
            head_dim = W.shape[0] // num_heads
            if head_dim == 0:
                continue
            keep_rows = []
            for h in keep_heads:
                keep_rows.extend(range(h * head_dim, (h + 1) * head_dim))
            W_new = W[keep_rows, :]
            idx = self._init_map[init.name]
            self._graph.initializer[idx].CopyFrom(
                _numpy_to_initializer(init.name, W_new, init.data_type)
            )

    def _prune_attention_bias(
        self,
        layer_data: dict,
        keep_heads: list[int],
        num_heads: int,
    ) -> None:
        """Shrink bias vectors associated with QKV projections."""
        for qkv_name in layer_data.get("qkv", []):
            bias_name = qkv_name.replace(".weight", ".bias")
            init = _find_initializer(self._model, bias_name)
            if init is None or len(init.dims) != 1:
                continue
            b = _initializer_to_numpy(init)
            head_dim = b.shape[0] // num_heads
            if head_dim == 0:
                continue
            keep_idx = []
            for h in keep_heads:
                keep_idx.extend(range(h * head_dim, (h + 1) * head_dim))
            b_new = b[keep_idx]
            idx = self._init_map[bias_name]
            self._graph.initializer[idx].CopyFrom(
                _numpy_to_initializer(bias_name, b_new, init.data_type)
            )

    # --------------------------------------------------------------------- #
    # Width shrinking
    # --------------------------------------------------------------------- #

    def shrink_width(
        self,
        target_hidden_dim: Optional[int] = None,
        ratio: float = 0.5,
        importance_method: str = "magnitude",
        output_path: Optional[str] = None,
    ) -> ShrinkResult:
        t0 = time.monotonic()
        analysis = self.analyze()
        original_dim = analysis.hidden_dim
        original_params = _count_params(self._model)

        if original_dim == 0:
            return ShrinkResult(method=importance_method, error_msg="Cannot detect hidden dim")

        new_dim = target_hidden_dim if target_hidden_dim else max(1, int(original_dim * ratio))
        new_dim = min(new_dim, original_dim)

        try:
            keep_indices = self._rank_neurons("hidden", importance_method)[:new_dim]
            keep_indices = sorted(keep_indices)

            self._shrink_all_layers(keep_indices, analysis)

            new_params = _count_params(self._model)
            out = self._save(output_path, suffix="_width_shrunk")

            return ShrinkResult(
                success=True,
                method=importance_method,
                original_dim=original_dim,
                new_dim=new_dim,
                original_params=original_params,
                new_params=new_params,
                reduction_ratio=1.0 - new_params / max(original_params, 1),
                output_path=str(out),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            log.exception("Width shrinking failed")
            return ShrinkResult(
                method=importance_method,
                original_dim=original_dim,
                new_dim=new_dim,
                original_params=original_params,
                elapsed_s=time.monotonic() - t0,
            )

    def _rank_neurons(self, layer_name: str, method: str) -> list[int]:
        """Rank neurons by importance and return indices in descending order.

        "magnitude": L2 norm of outgoing weight columns for each neuron.
        "activation": falls back to magnitude (needs runtime data).
        """
        all_scores: Optional[np.ndarray] = None

        for init in self._graph.initializer:
            if len(init.dims) != 2:
                continue

            W = _initializer_to_numpy(init)

            if layer_name == "hidden":
                dim = W.shape[0]
            else:
                continue

            neuron_scores = np.linalg.norm(W, axis=1)
            if all_scores is None:
                all_scores = np.zeros(dim, dtype=np.float64)
            if neuron_scores.shape[0] == all_scores.shape[0]:
                all_scores += neuron_scores

        if method == "activation":
            log.warning(
                "Activation-based ranking requires calibration data; "
                "falling back to magnitude"
            )

        if all_scores is None:
            return []
        return list(np.argsort(-all_scores))

    def _shrink_layer(self, layer_name: str, keep_indices: list[int]) -> None:
        """Remove neuron indices from a single weight matrix."""
        init = _find_initializer(self._model, layer_name)
        if init is None or len(init.dims) != 2:
            return

        W = _initializer_to_numpy(init)
        W_new = W[keep_indices, :]
        idx = self._init_map[layer_name]
        self._graph.initializer[idx].CopyFrom(
            _numpy_to_initializer(layer_name, W_new, init.data_type)
        )

    def _shrink_all_layers(self, keep_indices: list[int], analysis: ArchAnalysis) -> None:
        """Apply width shrinking across all connected weight matrices.

        Every 2-D weight that has hidden_dim on either axis gets sliced:
        - axis-0 == hidden_dim  ->  keep rows
        - axis-1 == hidden_dim  ->  keep columns
        Biases and LayerNorm params with length == hidden_dim are also sliced.
        """
        hidden = analysis.hidden_dim
        keep = np.array(keep_indices)

        for i, init in enumerate(list(self._graph.initializer)):
            dims = list(init.dims)
            W = _initializer_to_numpy(init)
            changed = False

            if len(dims) == 2:
                if dims[0] == hidden and dims[1] == hidden:
                    W = W[np.ix_(keep, keep)]
                    changed = True
                elif dims[0] == hidden:
                    W = W[keep, :]
                    changed = True
                elif dims[1] == hidden:
                    W = W[:, keep]
                    changed = True
            elif len(dims) == 1 and dims[0] == hidden:
                W = W[keep]
                changed = True

            if changed:
                self._graph.initializer[i].CopyFrom(
                    _numpy_to_initializer(init.name, W, init.data_type)
                )

    # --------------------------------------------------------------------- #
    # Depth shrinking
    # --------------------------------------------------------------------- #

    def shrink_depth(
        self,
        num_layers_to_keep: Optional[int] = None,
        ratio: float = 0.5,
        method: str = "importance",
        output_path: Optional[str] = None,
    ) -> ShrinkResult:
        t0 = time.monotonic()
        analysis = self.analyze()
        original_params = _count_params(self._model)
        num_layers = analysis.num_layers

        if num_layers == 0:
            return ShrinkResult(method=method, original_dim=num_layers, new_dim=0)

        if num_layers_to_keep is None:
            num_layers_to_keep = max(1, int(num_layers * ratio))
        num_layers_to_keep = min(num_layers_to_keep, num_layers)

        try:
            sorted_layer_ids = sorted(self._layer_info["layers"].keys())

            if method == "importance":
                importance = self._compute_layer_importance()
                ranked = sorted(sorted_layer_ids, key=lambda i: importance.get(i, 0.0), reverse=True)
                keep_layers = set(ranked[:num_layers_to_keep])
            elif method == "uniform":
                step = max(1, num_layers // num_layers_to_keep)
                keep_layers = set(sorted_layer_ids[::step][:num_layers_to_keep])
            elif method == "first_last":
                keep_layers = {sorted_layer_ids[0], sorted_layer_ids[-1]}
                remaining = num_layers_to_keep - len(keep_layers)
                if remaining > 0:
                    middle = sorted_layer_ids[1:-1]
                    step = max(1, len(middle) // remaining)
                    keep_layers |= set(middle[::step][:remaining])
            else:
                raise ValueError(f"Unknown depth-shrink method: {method}")

            remove_layers = set(sorted_layer_ids) - keep_layers
            log.info(
                "Depth shrink: keeping %d/%d layers, removing %s",
                len(keep_layers), num_layers, sorted(remove_layers),
            )

            self._remove_layers(remove_layers)

            new_params = _count_params(self._model)
            out = self._save(output_path, suffix="_depth_shrunk")

            return ShrinkResult(
                success=True,
                method=method,
                original_dim=num_layers,
                new_dim=num_layers_to_keep,
                original_params=original_params,
                new_params=new_params,
                reduction_ratio=1.0 - new_params / max(original_params, 1),
                output_path=str(out),
                elapsed_s=time.monotonic() - t0,
            )
        except Exception as exc:
            log.exception("Depth shrinking failed")
            return ShrinkResult(
                method=method,
                original_dim=num_layers,
                new_dim=num_layers_to_keep,
                original_params=original_params,
                elapsed_s=time.monotonic() - t0,
            )

    def _compute_layer_importance(self, num_samples: int = 50) -> dict[int, float]:
        """Estimate each layer's contribution via L2 norm of all its parameters.

        A proper importance score would measure the change in output when the
        layer is removed (residual contribution).  Without calibration data we
        approximate by summing the Frobenius norms of every weight tensor in
        each layer — layers with larger norms contribute more to the residual.
        """
        scores: dict[int, float] = {}
        for layer_idx, layer_data in self._layer_info["layers"].items():
            layer_norm = 0.0
            for group in ("qkv", "ffn", "norms"):
                for name in layer_data.get(group, []):
                    init = _find_initializer(self._model, name)
                    if init is not None:
                        W = _initializer_to_numpy(init)
                        layer_norm += float(np.linalg.norm(W))
            scores[layer_idx] = layer_norm
        return scores

    def _remove_layers(self, remove_set: set[int]) -> None:
        """Delete all initializers belonging to removed layers.

        Also removes corresponding graph nodes whose inputs reference only
        deleted initializers, effectively short-circuiting the residual
        stream through the remaining layers.
        """
        names_to_remove: set[str] = set()
        for layer_idx in remove_set:
            layer_data = self._layer_info["layers"].get(layer_idx, {})
            for group in ("qkv", "ffn", "norms"):
                names_to_remove.update(layer_data.get(group, []))

        surviving = []
        for init in self._graph.initializer:
            if init.name not in names_to_remove:
                surviving.append(init)
        del self._graph.initializer[:]
        self._graph.initializer.extend(surviving)

        surviving_nodes = []
        for node in self._graph.node:
            if any(inp in names_to_remove for inp in node.input):
                self._reconnect_skip(node, self._graph)
            else:
                surviving_nodes.append(node)
        del self._graph.node[:]
        self._graph.node.extend(surviving_nodes)

        self._init_map = {
            init.name: idx for idx, init in enumerate(self._graph.initializer)
        }

    @staticmethod
    def _reconnect_skip(removed_node, graph) -> None:
        """Patch downstream nodes so they read from the removed node's input.

        In a residual transformer each block receives the residual stream and
        adds its output back.  When we remove a block we want the residual to
        flow straight through: every node that consumed one of our outputs
        should instead consume our *first* input (the residual).
        """
        if not removed_node.input or not removed_node.output:
            return
        residual_input = removed_node.input[0]
        outputs = set(removed_node.output)
        for node in graph.node:
            for i, inp in enumerate(node.input):
                if inp in outputs:
                    node.input[i] = residual_input

    # --------------------------------------------------------------------- #
    # Vocabulary pruning
    # --------------------------------------------------------------------- #

    def prune_vocab(
        self,
        keep_tokens: Optional[list[int]] = None,
        corpus_path: Optional[str] = None,
        min_frequency: int = 0,
        output_path: Optional[str] = None,
    ) -> VocabPruneResult:
        analysis = self.analyze()
        original_vocab = analysis.vocab_size

        if original_vocab == 0:
            return VocabPruneResult(error_msg="Could not detect embedding matrix")

        if keep_tokens is None and corpus_path is not None:
            keep_tokens = self._tokens_from_corpus(corpus_path, min_frequency, original_vocab)
        elif keep_tokens is None:
            return VocabPruneResult(
                success=False,
                original_vocab=original_vocab,
            )

        keep_tokens = sorted(set(t for t in keep_tokens if 0 <= t < original_vocab))
        if not keep_tokens:
            return VocabPruneResult(success=False, original_vocab=original_vocab)

        keep_arr = np.array(keep_tokens)
        remap = {old: new for new, old in enumerate(keep_tokens)}

        for emb_name in self._layer_info["embed_names"]:
            init = _find_initializer(self._model, emb_name)
            if init is None or len(init.dims) != 2:
                continue
            E = _initializer_to_numpy(init)
            if E.shape[0] != original_vocab:
                continue
            E_new = E[keep_arr, :]
            idx = self._init_map[emb_name]
            self._graph.initializer[idx].CopyFrom(
                _numpy_to_initializer(emb_name, E_new, init.data_type)
            )
            log.info("Pruned embedding %s: %s -> %s", emb_name, E.shape, E_new.shape)

        for proj_name in self._layer_info["output_proj_names"]:
            init = _find_initializer(self._model, proj_name)
            if init is None or len(init.dims) != 2:
                continue
            W = _initializer_to_numpy(init)
            if W.shape[0] == original_vocab:
                W_new = W[keep_arr, :]
                idx = self._init_map[proj_name]
                self._graph.initializer[idx].CopyFrom(
                    _numpy_to_initializer(proj_name, W_new, init.data_type)
                )
            elif W.shape[1] == original_vocab:
                W_new = W[:, keep_arr]
                idx = self._init_map[proj_name]
                self._graph.initializer[idx].CopyFrom(
                    _numpy_to_initializer(proj_name, W_new, init.data_type)
                )

            bias_name = proj_name.replace(".weight", ".bias")
            bias_init = _find_initializer(self._model, bias_name)
            if bias_init is not None and len(bias_init.dims) == 1 and bias_init.dims[0] == original_vocab:
                b = _initializer_to_numpy(bias_init)
                b_new = b[keep_arr]
                idx = self._init_map[bias_name]
                self._graph.initializer[idx].CopyFrom(
                    _numpy_to_initializer(bias_name, b_new, bias_init.data_type)
                )

        out = self._save(output_path, suffix="_vocab_pruned")

        remap_path = Path(out).parent / "token_remap.json"
        remap_path.write_text(json.dumps({str(k): v for k, v in remap.items()}, indent=2))

        return VocabPruneResult(
            success=True,
            original_vocab=original_vocab,
            new_vocab=len(keep_tokens),
            removed_tokens=original_vocab - len(keep_tokens),
            remap_table_path=str(remap_path),
            output_path=str(out),
        )

    @staticmethod
    def _tokens_from_corpus(
        corpus_path: str,
        min_frequency: int,
        vocab_size: int,
    ) -> list[int]:
        """Scan a whitespace-tokenised corpus file and return token IDs above threshold."""
        counts: Counter = Counter()
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for tok in line.strip().split():
                    try:
                        tid = int(tok)
                        if 0 <= tid < vocab_size:
                            counts[tid] += 1
                    except ValueError:
                        continue

        return [tid for tid, cnt in counts.items() if cnt >= min_frequency]

    # --------------------------------------------------------------------- #
    # Save helper
    # --------------------------------------------------------------------- #

    def _save(self, output_path: Optional[str], suffix: str) -> Path:
        import onnx

        if output_path:
            out = Path(output_path)
        else:
            out = self._path.parent / f"{self._path.stem}{suffix}.onnx"

        out.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(self._model, str(out))
        log.info("Saved transformed model to %s", out)
        return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def convert_architecture(model_path: str, action: str = "analyze", **kwargs):
    """Top-level dispatch for CLI usage.

    Actions: analyze, prune_heads, shrink_width, shrink_depth, prune_vocab
    """
    converter = ArchitectureConverter(model_path)

    dispatch = {
        "analyze": converter.analyze,
        "prune_heads": converter.prune_heads,
        "shrink_width": converter.shrink_width,
        "shrink_depth": converter.shrink_depth,
        "prune_vocab": converter.prune_vocab,
    }

    fn = dispatch.get(action)
    if fn is None:
        raise ValueError(
            f"Unknown action '{action}'. Choose from: {', '.join(dispatch)}"
        )

    return fn(**kwargs)
