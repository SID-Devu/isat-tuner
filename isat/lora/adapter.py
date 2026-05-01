"""LoRA adapter runtime — load, activate, hot-swap, fuse, and multi-LoRA routing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger("isat.lora")


@dataclass
class LoRAAdapter:
    name: str
    rank: int
    alpha: float
    target_modules: List[str]
    weights: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    scaling: float = field(init=False)

    def __post_init__(self):
        self.scaling = self.alpha / self.rank


@dataclass
class AdapterInfo:
    name: str
    rank: int
    alpha: float
    num_params: int
    target_layers: List[str]
    size_mb: float


class LoRARuntime:

    def __init__(self, base_model_path: str, provider: str = "CPUExecutionProvider"):
        import onnx
        import onnxruntime as ort

        self._base_model_path = Path(base_model_path)
        self._provider = provider
        self._onnx_model = onnx.load(str(self._base_model_path))
        self._linear_layers = self._identify_linear_layers(self._onnx_model.graph)
        self._original_weights: Dict[str, np.ndarray] = {}
        self._adapters: Dict[str, LoRAAdapter] = {}
        self._active_adapter: Optional[str] = None

        self._session = ort.InferenceSession(
            self._base_model_path.as_posix(), providers=[self._provider]
        )
        self._cache_original_weights()

    def _cache_original_weights(self):
        for init in self._onnx_model.graph.initializer:
            if init.name in self._linear_layers:
                self._original_weights[init.name] = np.copy(
                    np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                    if init.raw_data
                    else np.array(init.float_data, dtype=np.float32).reshape(init.dims)
                )

    def _identify_linear_layers(self, graph) -> set:
        candidates = set()
        for node in graph.node:
            if node.op_type in ("MatMul", "Gemm"):
                for inp in node.input:
                    candidates.add(inp)
        initializer_names = {init.name for init in graph.initializer}
        return candidates & initializer_names

    def load_adapter(self, adapter_path: str, name: Optional[str] = None) -> str:
        path = Path(adapter_path)
        adapter_name = name or path.stem

        weights: Dict[str, Dict[str, np.ndarray]] = {}
        rank = 0
        alpha = 1.0
        target_modules: List[str] = []

        if path.suffix == ".safetensors":
            weights, rank, alpha, target_modules = self._load_safetensors(path)
        elif path.suffix == ".npz":
            weights, rank, alpha, target_modules = self._load_npz(path)
        elif path.suffix == ".pt":
            weights, rank, alpha, target_modules = self._load_pt(path)
        else:
            raise ValueError(f"Unsupported adapter format: {path.suffix}")

        adapter = LoRAAdapter(
            name=adapter_name,
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
            weights=weights,
        )
        self._adapters[adapter_name] = adapter
        log.info("Loaded adapter '%s' (rank=%d, alpha=%.1f, layers=%d)",
                 adapter_name, rank, alpha, len(weights))
        return adapter_name

    def activate(self, adapter_name: str):
        if adapter_name not in self._adapters:
            raise KeyError(f"Adapter '{adapter_name}' not loaded")

        if self._active_adapter:
            self.deactivate()

        adapter = self._adapters[adapter_name]
        for layer_name, ab in adapter.weights.items():
            self._apply_delta(layer_name, ab["A"], ab["B"], adapter.scaling)

        self._rebuild_session()
        self._active_adapter = adapter_name
        log.info("Activated adapter '%s'", adapter_name)

    def deactivate(self):
        if not self._active_adapter:
            return

        for init in self._onnx_model.graph.initializer:
            if init.name in self._original_weights:
                w = self._original_weights[init.name]
                init.raw_data = w.tobytes()

        self._rebuild_session()
        self._active_adapter = None

    def hot_swap(self, adapter_name: str):
        if adapter_name not in self._adapters:
            raise KeyError(f"Adapter '{adapter_name}' not loaded")
        if self._active_adapter == adapter_name:
            return

        old_adapter = self._adapters.get(self._active_adapter) if self._active_adapter else None
        new_adapter = self._adapters[adapter_name]

        affected_layers = set()
        if old_adapter:
            affected_layers.update(old_adapter.weights.keys())
        affected_layers.update(new_adapter.weights.keys())

        for layer_name in affected_layers:
            if layer_name in self._original_weights:
                for init in self._onnx_model.graph.initializer:
                    if init.name == layer_name:
                        init.raw_data = self._original_weights[layer_name].tobytes()
                        break

        for layer_name, ab in new_adapter.weights.items():
            self._apply_delta(layer_name, ab["A"], ab["B"], new_adapter.scaling)

        self._rebuild_session()
        self._active_adapter = adapter_name
        log.info("Hot-swapped to adapter '%s'", adapter_name)

    def fuse(self, adapter_name: str, output_path: str):
        if adapter_name not in self._adapters:
            raise KeyError(f"Adapter '{adapter_name}' not loaded")

        import onnx

        fused_model = onnx.load(str(self._base_model_path))
        adapter = self._adapters[adapter_name]

        for init in fused_model.graph.initializer:
            if init.name in adapter.weights:
                ab = adapter.weights[init.name]
                w = (
                    np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                    if init.raw_data
                    else np.array(init.float_data, dtype=np.float32).reshape(init.dims)
                )
                delta = self._compute_delta(ab["A"], ab["B"], adapter.scaling, w.shape)
                fused = w + delta
                init.raw_data = fused.astype(np.float32).tobytes()

        onnx.save(fused_model, output_path)
        log.info("Fused adapter '%s' into '%s'", adapter_name, output_path)

    def list_adapters(self) -> List[AdapterInfo]:
        infos = []
        for name, adapter in self._adapters.items():
            num_params = sum(
                ab["A"].size + ab["B"].size for ab in adapter.weights.values()
            )
            size_mb = num_params * 4 / (1024 * 1024)
            infos.append(AdapterInfo(
                name=name,
                rank=adapter.rank,
                alpha=adapter.alpha,
                num_params=num_params,
                target_layers=list(adapter.weights.keys()),
                size_mb=size_mb,
            ))
        return infos

    def run(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        input_feed = {k: v for k, v in inputs.items()}
        return self._session.run(None, input_feed)

    def _apply_delta(self, layer_name: str, A: np.ndarray, B: np.ndarray, scaling: float):
        for init in self._onnx_model.graph.initializer:
            if init.name == layer_name:
                w = (
                    np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                    if init.raw_data
                    else np.array(init.float_data, dtype=np.float32).reshape(init.dims)
                )
                delta = self._compute_delta(A, B, scaling, w.shape)
                modified = w + delta
                init.raw_data = modified.astype(np.float32).tobytes()
                break

    def _compute_delta(
        self, A: np.ndarray, B: np.ndarray, scaling: float, target_shape: tuple
    ) -> np.ndarray:
        delta = (B @ A) * scaling
        if delta.shape != target_shape:
            if delta.T.shape == target_shape:
                delta = delta.T
            else:
                raise ValueError(
                    f"LoRA delta shape {delta.shape} incompatible with target {target_shape}"
                )
        return delta

    def _rebuild_session(self):
        import onnxruntime as ort

        serialized = self._onnx_model.SerializeToString()
        self._session = ort.InferenceSession(serialized, providers=[self._provider])

    def _load_safetensors(self, path: Path):
        from safetensors.numpy import load_file

        tensors = load_file(str(path))
        return self._parse_lora_tensors(tensors)

    def _load_npz(self, path: Path):
        data = np.load(str(path), allow_pickle=True)
        tensors = {k: data[k] for k in data.files}
        return self._parse_lora_tensors(tensors)

    def _load_pt(self, path: Path):
        import torch

        state_dict = torch.load(str(path), map_location="cpu", weights_only=True)
        tensors = {k: v.numpy() for k, v in state_dict.items()}
        return self._parse_lora_tensors(tensors)

    def _parse_lora_tensors(self, tensors: Dict[str, np.ndarray]):
        weights: Dict[str, Dict[str, np.ndarray]] = {}
        rank = 0
        alpha = 1.0
        target_modules: List[str] = []

        if "lora_alpha" in tensors:
            alpha = float(tensors["lora_alpha"])

        for key, value in tensors.items():
            if "lora_A" in key or ".A" in key:
                base_name = (
                    key.replace(".lora_A.weight", "")
                    .replace(".lora_A", "")
                    .replace(".A", "")
                )
                if base_name not in weights:
                    weights[base_name] = {}
                weights[base_name]["A"] = value.astype(np.float32)
                if rank == 0:
                    rank = value.shape[0] if value.ndim == 2 else value.shape[-1]
            elif "lora_B" in key or ".B" in key:
                base_name = (
                    key.replace(".lora_B.weight", "")
                    .replace(".lora_B", "")
                    .replace(".B", "")
                )
                if base_name not in weights:
                    weights[base_name] = {}
                weights[base_name]["B"] = value.astype(np.float32)

        target_modules = list(weights.keys())
        if rank == 0:
            rank = 8
        return weights, rank, alpha, target_modules


class MultiLoRARouter:

    def __init__(self, base_model_path: str, adapter_configs: List[Dict[str, Any]]):
        self._base_model_path = base_model_path
        self._runtimes: Dict[str, LoRARuntime] = {}
        self._adapter_configs = adapter_configs

        for cfg in adapter_configs:
            rt = LoRARuntime(base_model_path, provider=cfg.get("provider", "CPUExecutionProvider"))
            name = rt.load_adapter(cfg["path"], name=cfg.get("name"))
            rt.activate(name)
            self._runtimes[name] = rt

    def route(self, inputs: Dict[str, np.ndarray], adapter_name: str) -> List[np.ndarray]:
        if adapter_name not in self._runtimes:
            raise KeyError(f"Adapter '{adapter_name}' not available in router")
        return self._runtimes[adapter_name].run(inputs)

    def batch_route(
        self,
        inputs_list: List[Dict[str, np.ndarray]],
        adapter_names: List[str],
    ) -> List[List[np.ndarray]]:
        if len(inputs_list) != len(adapter_names):
            raise ValueError("inputs_list and adapter_names must have same length")

        groups: Dict[str, List[int]] = {}
        for idx, name in enumerate(adapter_names):
            groups.setdefault(name, []).append(idx)

        results: List[Optional[List[np.ndarray]]] = [None] * len(inputs_list)
        for adapter_name, indices in groups.items():
            for idx in indices:
                results[idx] = self.route(inputs_list[idx], adapter_name)

        return results
