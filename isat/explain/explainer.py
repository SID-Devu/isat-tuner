from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger("isat.explain")

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    onnx = None  # type: ignore[assignment]
    numpy_helper = None  # type: ignore[assignment]


def _load_session_from_model(model, providers):
    """Load ORT session from an in-memory ONNX model, handling >2GB models."""
    try:
        return ort.InferenceSession(model.SerializeToString(), providers=providers)
    except Exception:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False, dir="/tmp")
        onnx.save(
            model, tmp.name,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=os.path.basename(tmp.name) + ".data",
            size_threshold=1024,
        )
        return ort.InferenceSession(tmp.name, providers=providers)


@dataclass
class ExplainReport:
    model_path: str
    input_shapes: Dict[str, tuple]
    feature_importance: Dict[str, np.ndarray]
    attention_layers: Dict[str, np.ndarray]
    top_sensitive_regions: Dict[str, np.ndarray]
    method_used: str
    elapsed_s: float


InputDict = Dict[str, np.ndarray]


def _require_ort() -> None:
    if ort is None:
        raise ImportError(
            "onnxruntime is required for model explanation. "
            "Install it with: pip install onnxruntime"
        )


def _require_onnx() -> None:
    if onnx is None:
        raise ImportError(
            "onnx is required for graph inspection. "
            "Install it with: pip install onnx"
        )


def _onnx_dtype_to_numpy(elem_type: int) -> np.dtype:
    _map = {
        1: np.float32,
        2: np.uint8,
        3: np.int8,
        5: np.int16,
        6: np.int32,
        7: np.int64,
        10: np.float16,
        11: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.uint64,
        9: bool,
    }
    return np.dtype(_map.get(elem_type, np.float32))


class ModelExplainer:
    """Run explainability analyses on an ONNX model."""

    def __init__(
        self,
        model_path: str | Path,
        provider: str | None = None,
    ) -> None:
        _require_ort()
        self.model_path = str(model_path)
        providers = [provider] if provider else ort.get_available_providers()
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        self._inputs = self._session.get_inputs()
        self._outputs = self._session.get_outputs()

    def _run(self, feeds: InputDict) -> List[np.ndarray]:
        out_names = [o.name for o in self._outputs]
        return self._session.run(out_names, feeds)

    def _input_specs(self) -> Dict[str, tuple]:
        return {inp.name: tuple(inp.shape) for inp in self._inputs}

    def _baseline_output(self, feeds: InputDict) -> np.ndarray:
        results = self._run(feeds)
        return results[0]

    # ------------------------------------------------------------------
    # feature_importance
    # ------------------------------------------------------------------

    def feature_importance(
        self,
        inputs: InputDict,
        method: str = "perturbation",
        num_samples: int = 100,
    ) -> Dict[str, np.ndarray]:
        if method != "perturbation":
            raise ValueError(f"Unsupported method: {method!r}. Use 'perturbation'.")

        baseline = self._baseline_output(inputs).copy()
        importance: Dict[str, np.ndarray] = {}

        for inp_name, inp_arr in inputs.items():
            flat = inp_arr.flatten()
            scores = np.zeros(flat.shape[0], dtype=np.float64)
            indices = np.arange(flat.shape[0])
            rng = np.random.default_rng(42)

            sample_indices = (
                indices
                if flat.shape[0] <= num_samples
                else rng.choice(indices, size=num_samples, replace=False)
            )

            for idx in sample_indices:
                perturbed = inp_arr.copy()
                original_val = perturbed.flat[idx]
                perturbed.flat[idx] = 0.0
                feeds_p = {**inputs, inp_name: perturbed}
                out_p = self._baseline_output(feeds_p)
                scores[idx] = float(np.sum(np.abs(baseline - out_p)))
                perturbed.flat[idx] = original_val

            if scores.max() > 0:
                scores /= scores.max()
            importance[inp_name] = scores.reshape(inp_arr.shape)

        return importance

    # ------------------------------------------------------------------
    # attention_map
    # ------------------------------------------------------------------

    def attention_map(self, inputs: InputDict) -> Dict[str, np.ndarray]:
        _require_onnx()
        model = onnx.load(self.model_path)

        attn_outputs: List[str] = []
        for node in model.graph.node:
            if node.op_type in ("Softmax", "MatMul"):
                for out in node.output:
                    lower = out.lower()
                    if any(k in lower for k in ("attn", "attention", "score")):
                        attn_outputs.append(out)

        if not attn_outputs:
            log.info("No attention layers detected in model graph.")
            return {}

        for tensor_name in attn_outputs:
            value_info = onnx.helper.make_tensor_value_info(
                tensor_name, onnx.TensorProto.FLOAT, None
            )
            model.graph.output.append(value_info)

        providers = ort.get_available_providers()
        sess = _load_session_from_model(model, providers)
        out_names = [o.name for o in sess.get_outputs()]
        results = sess.run(out_names, inputs)

        original_out_count = len(self._outputs)
        attention: Dict[str, np.ndarray] = {}
        for name, arr in zip(out_names[original_out_count:], results[original_out_count:]):
            attention[name] = arr

        return attention

    # ------------------------------------------------------------------
    # gradient_attribution
    # ------------------------------------------------------------------

    def gradient_attribution(
        self,
        inputs: InputDict,
        target_class: int | None = None,
        epsilon: float = 1e-4,
    ) -> Dict[str, np.ndarray]:
        baseline = self._baseline_output(inputs).copy()

        if target_class is not None and baseline.ndim >= 1:
            if baseline.ndim == 1:
                baseline_score = baseline[target_class]
            else:
                baseline_score = baseline[..., target_class]
        else:
            baseline_score = baseline

        attributions: Dict[str, np.ndarray] = {}

        for inp_name, inp_arr in inputs.items():
            if not np.issubdtype(inp_arr.dtype, np.floating):
                log.warning(
                    "Skipping gradient attribution for non-float input %r", inp_name
                )
                continue

            grad = np.zeros_like(inp_arr, dtype=np.float64)
            flat_inp = inp_arr.flatten()

            for i in range(flat_inp.shape[0]):
                perturbed = inp_arr.copy()
                perturbed.flat[i] = flat_inp[i] + epsilon
                feeds_p = {**inputs, inp_name: perturbed}
                out_p = self._baseline_output(feeds_p)

                if target_class is not None and out_p.ndim >= 1:
                    if out_p.ndim == 1:
                        perturbed_score = out_p[target_class]
                    else:
                        perturbed_score = out_p[..., target_class]
                else:
                    perturbed_score = out_p

                diff = np.sum(perturbed_score - baseline_score) / epsilon
                grad.flat[i] = diff

            attributions[inp_name] = grad.reshape(inp_arr.shape)

        return attributions

    # ------------------------------------------------------------------
    # layer_activation
    # ------------------------------------------------------------------

    def layer_activation(
        self,
        inputs: InputDict,
        layer_names: Sequence[str] | None = None,
    ) -> Dict[str, np.ndarray]:
        _require_onnx()
        model = onnx.load(self.model_path)

        all_internal = set()
        for node in model.graph.node:
            for out in node.output:
                all_internal.add(out)

        existing_outputs = {o.name for o in model.graph.output}

        if layer_names is not None:
            targets = [n for n in layer_names if n in all_internal]
            if not targets:
                raise ValueError(
                    f"None of the requested layers found. "
                    f"Available: {sorted(all_internal)[:20]}..."
                )
        else:
            targets = sorted(all_internal - existing_outputs)

        for tensor_name in targets:
            value_info = onnx.helper.make_tensor_value_info(
                tensor_name, onnx.TensorProto.FLOAT, None
            )
            model.graph.output.append(value_info)

        providers = ort.get_available_providers()
        sess = _load_session_from_model(model, providers)
        out_names = [o.name for o in sess.get_outputs()]
        results = sess.run(out_names, inputs)

        original_out_count = len(self._outputs)
        activations: Dict[str, np.ndarray] = {}
        for name, arr in zip(out_names[original_out_count:], results[original_out_count:]):
            activations[name] = arr

        return activations

    # ------------------------------------------------------------------
    # sensitivity_map
    # ------------------------------------------------------------------

    def sensitivity_map(
        self,
        inputs: InputDict,
        noise_std: float = 0.01,
        num_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        sensitivity: Dict[str, np.ndarray] = {}

        for inp_name, inp_arr in inputs.items():
            if not np.issubdtype(inp_arr.dtype, np.floating):
                continue

            outputs_stack: List[np.ndarray] = []
            for _ in range(num_samples):
                noise = rng.normal(0, noise_std, size=inp_arr.shape).astype(inp_arr.dtype)
                feeds_noisy = {**inputs, inp_name: inp_arr + noise}
                out = self._baseline_output(feeds_noisy)
                outputs_stack.append(out.flatten())

            stacked = np.stack(outputs_stack, axis=0)
            variance = np.var(stacked, axis=0)
            total_var = float(np.sum(variance))

            elem_sensitivity = np.zeros_like(inp_arr, dtype=np.float64)
            flat = inp_arr.flatten()

            patch_size = max(1, flat.shape[0] // min(flat.shape[0], 64))
            for start in range(0, flat.shape[0], patch_size):
                end = min(start + patch_size, flat.shape[0])
                patch_vars: List[float] = []
                for _ in range(min(num_samples, 20)):
                    noisy = inp_arr.copy()
                    noisy.flat[start:end] += rng.normal(
                        0, noise_std, size=end - start
                    ).astype(inp_arr.dtype)
                    feeds_p = {**inputs, inp_name: noisy}
                    out_p = self._baseline_output(feeds_p)
                    patch_vars.append(float(np.sum(np.var([out_p.flatten()], axis=0))))
                elem_sensitivity.flat[start:end] = np.mean(patch_vars)

            if elem_sensitivity.max() > 0:
                elem_sensitivity /= elem_sensitivity.max()
            sensitivity[inp_name] = elem_sensitivity.reshape(inp_arr.shape)

        return sensitivity

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self, inputs: InputDict) -> ExplainReport:
        t0 = time.perf_counter()
        methods: List[str] = []

        fi = self.feature_importance(inputs)
        methods.append("perturbation")

        attn = self.attention_map(inputs)
        if attn:
            methods.append("attention")

        sens = self.sensitivity_map(inputs)
        methods.append("sensitivity")

        top_regions: Dict[str, np.ndarray] = {}
        for name, arr in sens.items():
            threshold = np.percentile(arr, 90)
            top_regions[name] = (arr >= threshold).astype(np.float32)

        elapsed = time.perf_counter() - t0

        return ExplainReport(
            model_path=self.model_path,
            input_shapes=self._input_specs(),
            feature_importance=fi,
            attention_layers=attn,
            top_sensitive_regions=top_regions,
            method_used="+".join(methods),
            elapsed_s=round(elapsed, 3),
        )


# ------------------------------------------------------------------
# Top-level convenience
# ------------------------------------------------------------------

def _generate_random_inputs(session: "ort.InferenceSession") -> InputDict:
    rng = np.random.default_rng(1)
    feeds: InputDict = {}
    for inp in session.get_inputs():
        shape = []
        for d in inp.shape:
            shape.append(d if isinstance(d, int) and d > 0 else 1)
        dtype = _onnx_dtype_to_numpy(inp.type_as_dtype_object if hasattr(inp, "type_as_dtype_object") else 1)
        try:
            elem_type_str = inp.type
            if "float" in elem_type_str.lower():
                dtype = np.float32
            elif "int64" in elem_type_str.lower():
                dtype = np.int64
            elif "int32" in elem_type_str.lower():
                dtype = np.int32
            elif "double" in elem_type_str.lower():
                dtype = np.float64
        except Exception:
            dtype = np.float32

        if np.issubdtype(dtype, np.floating):
            feeds[inp.name] = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            feeds[inp.name] = rng.integers(0, 100, size=shape).astype(dtype)
        else:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def explain_model(
    model_path: str | Path,
    inputs: InputDict | None = None,
    method: str = "auto",
) -> ExplainReport:
    """Run explainability analysis on an ONNX model.

    If *inputs* is ``None``, random inputs are generated from the model's
    input specifications.
    """
    _require_ort()
    explainer = ModelExplainer(model_path)

    if inputs is None:
        log.info("No inputs provided; generating random inputs from model spec.")
        inputs = _generate_random_inputs(explainer._session)

    if method == "auto":
        return explainer.summary(inputs)

    report_fields: dict[str, Any] = {
        "model_path": str(model_path),
        "input_shapes": explainer._input_specs(),
        "feature_importance": {},
        "attention_layers": {},
        "top_sensitive_regions": {},
        "method_used": method,
        "elapsed_s": 0.0,
    }

    t0 = time.perf_counter()

    if method == "perturbation":
        report_fields["feature_importance"] = explainer.feature_importance(inputs)
    elif method == "attention":
        report_fields["attention_layers"] = explainer.attention_map(inputs)
    elif method == "sensitivity":
        sens = explainer.sensitivity_map(inputs)
        report_fields["top_sensitive_regions"] = sens
    elif method == "gradient":
        report_fields["feature_importance"] = explainer.gradient_attribution(inputs)
    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose from: auto, perturbation, attention, sensitivity, gradient."
        )

    report_fields["elapsed_s"] = round(time.perf_counter() - t0, 3)
    return ExplainReport(**report_fields)
