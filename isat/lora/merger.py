"""Advanced weight merging: TIES, DARE, SLERP, Task Arithmetic, Model Soup."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from isat.lora.adapter import LoRARuntime

log = logging.getLogger("isat.lora")


@dataclass
class MergeResult:
    success: bool
    output_path: Optional[str]
    method: str
    num_models: int
    merged_params: int
    total_size_mb: float
    elapsed_s: float
    error: Optional[str] = None


class WeightMerger:

    def __init__(self, base_model_path: str):
        self._base_model_path = Path(base_model_path)
        self._base_weights = self._load_weights(str(self._base_model_path))

    def task_arithmetic(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        output_path: Optional[str] = None,
    ) -> MergeResult:
        t0 = time.time()
        try:
            lambdas = weights or [1.0 / len(model_paths)] * len(model_paths)
            merged = {k: np.copy(v) for k, v in self._base_weights.items()}

            for path, lam in zip(model_paths, lambdas):
                model_w = self._load_weights(path)
                for k in merged:
                    if k in model_w:
                        if model_w[k].shape != merged[k].shape:
                            continue
                        task_vector = model_w[k] - self._base_weights[k]
                        merged[k] += lam * task_vector

            out = output_path or self._default_output("task_arithmetic")
            self._save_merged(merged, out)
            return MergeResult(
                success=True, output_path=out, method="task_arithmetic",
                num_models=len(model_paths), merged_params=sum(v.size for v in merged.values()),
                total_size_mb=sum(v.nbytes for v in merged.values()) / (1024 * 1024),
                elapsed_s=time.time() - t0,
            )
        except Exception as e:
            log.error("task_arithmetic failed: %s", e)
            return MergeResult(
                success=False, output_path=None, method="task_arithmetic",
                num_models=len(model_paths), merged_params=0, total_size_mb=0,
                elapsed_s=time.time() - t0, error=str(e),
            )

    def ties_merge(
        self,
        model_paths: List[str],
        density: float = 0.5,
        weights: Optional[List[float]] = None,
        output_path: Optional[str] = None,
    ) -> MergeResult:
        t0 = time.time()
        try:
            lambdas = weights or [1.0] * len(model_paths)
            deltas = []
            for path, lam in zip(model_paths, lambdas):
                model_w = self._load_weights(path)
                delta = {}
                for k in self._base_weights:
                    if k in model_w and model_w[k].shape == self._base_weights[k].shape:
                        delta[k] = lam * (model_w[k] - self._base_weights[k])
                deltas.append(delta)

            merged = {k: np.copy(v) for k, v in self._base_weights.items()}
            param_keys = list(self._base_weights.keys())

            for k in param_keys:
                layer_deltas = [d[k] for d in deltas if k in d]
                if not layer_deltas:
                    continue

                stacked = np.stack(layer_deltas, axis=0)

                # Trim: zero out values with magnitude below the density-percentile threshold
                magnitudes = np.abs(stacked)
                threshold = np.percentile(magnitudes[magnitudes > 0], (1.0 - density) * 100)
                trimmed = np.where(magnitudes >= threshold, stacked, 0.0)

                # Elect signs: majority vote across models for each parameter
                sign_votes = np.sign(trimmed).sum(axis=0)
                elected_sign = np.sign(sign_votes)

                # Merge: average only values matching the elected sign, then add to base
                mask = np.sign(trimmed) == elected_sign[np.newaxis, ...]
                count = mask.sum(axis=0).clip(min=1)
                merged_delta = (trimmed * mask).sum(axis=0) / count
                merged[k] += merged_delta

            out = output_path or self._default_output("ties")
            self._save_merged(merged, out)
            return MergeResult(
                success=True, output_path=out, method="ties",
                num_models=len(model_paths), merged_params=sum(v.size for v in merged.values()),
                total_size_mb=sum(v.nbytes for v in merged.values()) / (1024 * 1024),
                elapsed_s=time.time() - t0,
            )
        except Exception as e:
            log.error("ties_merge failed: %s", e)
            return MergeResult(
                success=False, output_path=None, method="ties",
                num_models=len(model_paths), merged_params=0, total_size_mb=0,
                elapsed_s=time.time() - t0, error=str(e),
            )

    def dare_merge(
        self,
        model_paths: List[str],
        drop_rate: float = 0.9,
        weights: Optional[List[float]] = None,
        rescale: bool = True,
        output_path: Optional[str] = None,
    ) -> MergeResult:
        t0 = time.time()
        try:
            lambdas = weights or [1.0 / len(model_paths)] * len(model_paths)
            merged = {k: np.copy(v) for k, v in self._base_weights.items()}

            for path, lam in zip(model_paths, lambdas):
                model_w = self._load_weights(path)
                for k in merged:
                    if k not in model_w or model_w[k].shape != merged[k].shape:
                        continue
                    delta = model_w[k] - self._base_weights[k]
                    mask = np.random.binomial(1, 1.0 - drop_rate, size=delta.shape).astype(np.float32)
                    sparse_delta = delta * mask
                    if rescale and drop_rate < 1.0:
                        sparse_delta /= (1.0 - drop_rate)
                    merged[k] += lam * sparse_delta

            out = output_path or self._default_output("dare")
            self._save_merged(merged, out)
            return MergeResult(
                success=True, output_path=out, method="dare",
                num_models=len(model_paths), merged_params=sum(v.size for v in merged.values()),
                total_size_mb=sum(v.nbytes for v in merged.values()) / (1024 * 1024),
                elapsed_s=time.time() - t0,
            )
        except Exception as e:
            log.error("dare_merge failed: %s", e)
            return MergeResult(
                success=False, output_path=None, method="dare",
                num_models=len(model_paths), merged_params=0, total_size_mb=0,
                elapsed_s=time.time() - t0, error=str(e),
            )

    def slerp_merge(
        self,
        model_a_path: str,
        model_b_path: str,
        t: float = 0.5,
        output_path: Optional[str] = None,
    ) -> MergeResult:
        t0 = time.time()
        try:
            w_a = self._load_weights(model_a_path)
            w_b = self._load_weights(model_b_path)
            merged = {}

            for k in w_a:
                if k not in w_b or w_a[k].shape != w_b[k].shape:
                    merged[k] = w_a[k]
                    continue

                a_flat = w_a[k].flatten().astype(np.float64)
                b_flat = w_b[k].flatten().astype(np.float64)

                norm_a = np.linalg.norm(a_flat)
                norm_b = np.linalg.norm(b_flat)

                if norm_a < 1e-10 or norm_b < 1e-10:
                    merged[k] = ((1.0 - t) * w_a[k] + t * w_b[k]).astype(np.float32)
                    continue

                # SLERP: W = sin((1-t)*omega)/sin(omega) * W_a + sin(t*omega)/sin(omega) * W_b
                # where omega = arccos(clipped_cosine_similarity(W_a, W_b))
                cos_omega = np.dot(a_flat, b_flat) / (norm_a * norm_b)
                cos_omega = np.clip(cos_omega, -1.0, 1.0)
                omega = np.arccos(cos_omega)

                if omega < 1e-6:
                    merged[k] = ((1.0 - t) * w_a[k] + t * w_b[k]).astype(np.float32)
                else:
                    sin_omega = np.sin(omega)
                    coeff_a = np.sin((1.0 - t) * omega) / sin_omega
                    coeff_b = np.sin(t * omega) / sin_omega
                    result = coeff_a * a_flat + coeff_b * b_flat
                    merged[k] = result.reshape(w_a[k].shape).astype(np.float32)

            out = output_path or self._default_output("slerp")
            self._save_merged(merged, out)
            return MergeResult(
                success=True, output_path=out, method="slerp",
                num_models=2, merged_params=sum(v.size for v in merged.values()),
                total_size_mb=sum(v.nbytes for v in merged.values()) / (1024 * 1024),
                elapsed_s=time.time() - t0,
            )
        except Exception as e:
            log.error("slerp_merge failed: %s", e)
            return MergeResult(
                success=False, output_path=None, method="slerp",
                num_models=2, merged_params=0, total_size_mb=0,
                elapsed_s=time.time() - t0, error=str(e),
            )

    def model_soup(
        self,
        model_paths: List[str],
        output_path: Optional[str] = None,
    ) -> MergeResult:
        t0 = time.time()
        try:
            merged = {k: np.zeros_like(v) for k, v in self._base_weights.items()}
            valid_count = {k: 0 for k in self._base_weights}

            for path in model_paths:
                model_w = self._load_weights(path)
                for k in merged:
                    if k in model_w and model_w[k].shape == merged[k].shape:
                        merged[k] += model_w[k]
                        valid_count[k] += 1

            for k in merged:
                if valid_count[k] > 0:
                    merged[k] /= valid_count[k]
                else:
                    merged[k] = self._base_weights[k]

            out = output_path or self._default_output("soup")
            self._save_merged(merged, out)
            return MergeResult(
                success=True, output_path=out, method="model_soup",
                num_models=len(model_paths), merged_params=sum(v.size for v in merged.values()),
                total_size_mb=sum(v.nbytes for v in merged.values()) / (1024 * 1024),
                elapsed_s=time.time() - t0,
            )
        except Exception as e:
            log.error("model_soup failed: %s", e)
            return MergeResult(
                success=False, output_path=None, method="model_soup",
                num_models=len(model_paths), merged_params=0, total_size_mb=0,
                elapsed_s=time.time() - t0, error=str(e),
            )

    def _load_weights(self, model_path: str) -> Dict[str, np.ndarray]:
        import onnx
        from onnx.numpy_helper import to_array

        model = onnx.load(model_path, load_external_data=False)
        weights = {}
        for init in model.graph.initializer:
            try:
                weights[init.name] = to_array(init).astype(np.float32)
            except Exception:
                pass
        return weights

    def _save_merged(self, weights: Dict[str, np.ndarray], output_path: str):
        import onnx
        from onnx.numpy_helper import from_array

        model = onnx.load(str(self._base_model_path))
        init_map = {init.name: init for init in model.graph.initializer}

        for name, arr in weights.items():
            if name in init_map:
                new_tensor = from_array(arr, name=name)
                init_map[name].CopyFrom(new_tensor)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_path)
        log.info("Saved merged model to %s", output_path)

    def _default_output(self, method: str) -> str:
        stem = self._base_model_path.stem
        return str(self._base_model_path.parent / f"{stem}_merged_{method}.onnx")


# ---------------------------------------------------------------------------
# CLI-oriented helpers
# ---------------------------------------------------------------------------

def lora_manage(
    base_model: str,
    adapter_path: Optional[str] = None,
    action: str = "list",
    **kwargs: Any,
) -> Any:
    runtime = LoRARuntime(base_model, provider=kwargs.get("provider", "CPUExecutionProvider"))

    if action == "list":
        return runtime.list_adapters()

    if adapter_path is None:
        raise ValueError("adapter_path required for action '%s'" % action)

    name = runtime.load_adapter(adapter_path, name=kwargs.get("name"))

    if action == "activate":
        runtime.activate(name)
        return {"status": "activated", "adapter": name}
    elif action == "fuse":
        out = kwargs.get("output_path", str(Path(base_model).with_suffix(".fused.onnx")))
        runtime.fuse(name, out)
        return {"status": "fused", "output": out}
    elif action == "info":
        infos = runtime.list_adapters()
        return infos[0] if infos else None
    else:
        raise ValueError(f"Unknown action: {action}")


def merge_weights(
    base_model: str,
    model_paths: List[str],
    method: str = "ties",
    **kwargs: Any,
) -> MergeResult:
    merger = WeightMerger(base_model)

    if method == "task_arithmetic":
        return merger.task_arithmetic(model_paths, **kwargs)
    elif method == "ties":
        return merger.ties_merge(model_paths, **kwargs)
    elif method == "dare":
        return merger.dare_merge(model_paths, **kwargs)
    elif method == "slerp":
        if len(model_paths) != 2:
            return MergeResult(
                success=False, output_path=None, method="slerp",
                num_models=len(model_paths), merged_params=0, total_size_mb=0,
                elapsed_s=0, error="SLERP requires exactly 2 models",
            )
        return merger.slerp_merge(model_paths[0], model_paths[1], **kwargs)
    elif method == "soup":
        return merger.model_soup(model_paths, **kwargs)
    else:
        return MergeResult(
            success=False, output_path=None, method=method,
            num_models=len(model_paths), merged_params=0, total_size_mb=0,
            elapsed_s=0, error=f"Unknown method: {method}",
        )
