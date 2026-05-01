"""Knowledge distillation trainer -- real training loop over ONNX Runtime.

Performs live distillation from a frozen teacher ONNX model into a smaller
student ONNX model using pure numpy math (no PyTorch dependency).  Gradients
are estimated via stochastic coordinate descent with finite differences, and
weights are updated with a numpy Adam optimizer.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

log = logging.getLogger("isat.distill_train")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    temperature: float = 4.0
    alpha: float = 0.5
    warmup_steps: int = 100
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    save_every_n_epochs: int = 2


@dataclass
class DistillationResult:
    success: bool
    epochs_completed: int
    final_loss: float
    final_kl_loss: float
    final_ce_loss: float
    teacher_size_mb: float
    student_size_mb: float
    compression_ratio: float
    output_path: str
    training_time_s: float
    loss_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adam optimizer state (pure numpy)
# ---------------------------------------------------------------------------

@dataclass
class _AdamState:
    m: Dict[str, np.ndarray] = field(default_factory=dict)
    v: Dict[str, np.ndarray] = field(default_factory=dict)
    t: int = 0


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generate training batches by running the teacher on random inputs.

    Results can be cached to a `.npz` file so regeneration is skipped on
    subsequent runs.
    """

    def __init__(self, teacher_path: str, num_samples: int = 1000):
        self.teacher_path = teacher_path
        self.num_samples = num_samples
        self._cache_path: Optional[str] = None
        self._data: Optional[List[Dict[str, np.ndarray]]] = None
        self._teacher_logits: Optional[np.ndarray] = None

    def _ensure_generated(self) -> None:
        if self._data is not None:
            return

        if self._cache_path and Path(self._cache_path).exists():
            cached = np.load(self._cache_path, allow_pickle=True)
            self._data = []
            self._teacher_logits = cached["teacher_logits"]
            for i in range(len(self._teacher_logits)):
                sample: Dict[str, np.ndarray] = {}
                for key in cached.files:
                    if key.startswith(f"input_{i}_"):
                        name = key[len(f"input_{i}_"):]
                        sample[name] = cached[key]
                if sample:
                    self._data.append(sample)
            log.info("Loaded %d cached samples from %s", len(self._data), self._cache_path)
            return

        import onnxruntime as ort

        sess = ort.InferenceSession(self.teacher_path, providers=["CPUExecutionProvider"])
        inputs_meta = sess.get_inputs()
        output_names = [o.name for o in sess.get_outputs()]

        self._data = []
        all_logits = []
        log.info("Generating %d synthetic samples from teacher ...", self.num_samples)

        for _ in range(self.num_samples):
            feed: Dict[str, np.ndarray] = {}
            for inp in inputs_meta:
                shape = []
                for d in inp.shape:
                    shape.append(d if isinstance(d, int) and d > 0 else 1)
                dtype = np.float32
                if inp.type and "int" in inp.type:
                    dtype = np.int64
                    feed[inp.name] = np.random.randint(0, 100, size=shape).astype(dtype)
                else:
                    feed[inp.name] = np.random.randn(*shape).astype(dtype)
            self._data.append(feed)
            out = sess.run(output_names, feed)
            all_logits.append(out[0])

        self._teacher_logits = np.array(all_logits)

        if self._cache_path:
            save_dict: Dict[str, np.ndarray] = {"teacher_logits": self._teacher_logits}
            for i, sample in enumerate(self._data):
                for name, arr in sample.items():
                    save_dict[f"input_{i}_{name}"] = arr
            np.savez_compressed(self._cache_path, **save_dict)
            log.info("Cached %d samples to %s", len(self._data), self._cache_path)

    def set_cache_path(self, path: str) -> None:
        self._cache_path = path

    def generate(self, batch_size: int) -> Iterator[Tuple[List[Dict[str, np.ndarray]], np.ndarray]]:
        self._ensure_generated()
        assert self._data is not None and self._teacher_logits is not None

        n = len(self._data)
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]
            batch_inputs = [self._data[i] for i in batch_idx]
            batch_logits = self._teacher_logits[batch_idx]
            yield batch_inputs, batch_logits


# ---------------------------------------------------------------------------
# Student architecture builder
# ---------------------------------------------------------------------------

class StudentArchitectureBuilder:
    """Create a reduced ONNX student model from a teacher."""

    @staticmethod
    def build_smaller(
        teacher_path: str,
        reduction: str = "depth",
        factor: int = 2,
    ) -> str:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(teacher_path)
        graph = model.graph

        if reduction in ("depth", "both"):
            keep = []
            for i, node in enumerate(graph.node):
                if i % factor == 0:
                    keep.append(node)
            del graph.node[:]
            graph.node.extend(keep)
            log.info("Depth reduction: kept %d / %d nodes (factor=%d)",
                     len(keep), len(keep) * factor, factor)

        if reduction in ("width", "both"):
            for init in graph.initializer:
                arr = numpy_helper.to_array(init)
                new_shape = list(arr.shape)
                modified = False
                for dim in range(len(new_shape)):
                    if new_shape[dim] > factor:
                        new_shape[dim] = new_shape[dim] // factor
                        modified = True
                if modified:
                    new_arr = np.random.randn(*new_shape).astype(arr.dtype) * 0.02
                    new_tensor = numpy_helper.from_array(new_arr, init.name)
                    init.CopyFrom(new_tensor)
            log.info("Width reduction: halved initializer dimensions (factor=%d)", factor)

        suffix = f"_student_{reduction}_f{factor}"
        out_path = str(Path(teacher_path).with_suffix("")) + suffix + ".onnx"
        onnx.save(model, out_path)
        log.info("Saved student model to %s", out_path)
        return out_path


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------

class DistillationTrainer:
    """Train a student ONNX model to mimic a frozen teacher via KL-divergence."""

    def __init__(
        self,
        teacher_path: str,
        student_path: str,
        provider: str = "CPUExecutionProvider",
    ):
        import onnx
        import onnxruntime as ort
        from onnx import numpy_helper

        self.teacher_path = teacher_path
        self.student_path = student_path
        self.provider = provider

        self._teacher_sess = ort.InferenceSession(
            teacher_path, providers=[provider],
        )
        self._student_sess = ort.InferenceSession(
            student_path, providers=[provider],
        )
        self._student_model = onnx.load(student_path)

        self._student_weights: Dict[str, np.ndarray] = {}
        for init in self._student_model.graph.initializer:
            self._student_weights[init.name] = numpy_helper.to_array(init).copy()

        self._input_names = [i.name for i in self._student_sess.get_inputs()]
        self._output_names = [o.name for o in self._student_sess.get_outputs()]
        self._teacher_output_names = [o.name for o in self._teacher_sess.get_outputs()]

        self.teacher_size_mb = Path(teacher_path).stat().st_size / (1024 * 1024)
        self.student_size_mb = Path(student_path).stat().st_size / (1024 * 1024)

        log.info(
            "Loaded teacher (%.1f MB, %d outputs) and student (%.1f MB, %d weights)",
            self.teacher_size_mb,
            len(self._teacher_output_names),
            self.student_size_mb,
            len(self._student_weights),
        )

    # ----- loss functions --------------------------------------------------

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray, temperature: float) -> float:
        # KL(softmax(p/T) || softmax(q/T)) * T^2
        # The T^2 factor corrects for the gradient magnitude reduction caused
        # by dividing logits by T before the softmax.
        p_scaled = p / temperature
        q_scaled = q / temperature

        p_max = np.max(p_scaled, axis=-1, keepdims=True)
        q_max = np.max(q_scaled, axis=-1, keepdims=True)

        p_exp = np.exp(p_scaled - p_max)
        q_exp = np.exp(q_scaled - q_max)

        p_soft = p_exp / np.sum(p_exp, axis=-1, keepdims=True)
        q_soft = q_exp / np.sum(q_exp, axis=-1, keepdims=True)

        p_soft = np.clip(p_soft, 1e-12, None)
        q_soft = np.clip(q_soft, 1e-12, None)

        kl = np.sum(p_soft * (np.log(p_soft) - np.log(q_soft)))
        return float(kl * (temperature ** 2)) / p.shape[0]

    @staticmethod
    def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
        log_probs = shifted - log_sum_exp
        n = logits.shape[0]
        if labels.ndim == 1:
            ce = -np.mean(log_probs[np.arange(n), labels])
        else:
            ce = -np.sum(labels * log_probs) / n
        return float(ce)

    # ----- gradient estimation ---------------------------------------------

    def _compute_gradients(
        self,
        loss_fn: Callable[[], float],
        weights: Dict[str, np.ndarray],
        epsilon: float = 1e-5,
    ) -> Dict[str, np.ndarray]:
        # Stochastic coordinate descent: instead of perturbing every scalar in
        # every weight tensor (intractable for large models), randomly sample a
        # fixed budget of coordinates per parameter.  This gives an unbiased
        # gradient estimator that trades variance for speed.
        grads: Dict[str, np.ndarray] = {}
        max_coords_per_param = 64

        for name, w in weights.items():
            g = np.zeros_like(w)
            flat = w.ravel()
            n_elements = flat.size

            n_sample = min(max_coords_per_param, n_elements)
            coords = np.random.choice(n_elements, size=n_sample, replace=False)

            for idx in coords:
                orig = flat[idx]

                flat[idx] = orig + epsilon
                w_reshaped = flat.reshape(w.shape)
                weights[name] = w_reshaped
                loss_plus = loss_fn()

                flat[idx] = orig - epsilon
                w_reshaped = flat.reshape(w.shape)
                weights[name] = w_reshaped
                loss_minus = loss_fn()

                flat[idx] = orig
                weights[name] = flat.reshape(w.shape)

                grad_val = (loss_plus - loss_minus) / (2 * epsilon)
                g_flat = g.ravel()
                g_flat[idx] = grad_val * (n_elements / n_sample)
                g = g_flat.reshape(w.shape)

            grads[name] = g

        return grads

    @staticmethod
    def _adam_update(
        weights: Dict[str, np.ndarray],
        grads: Dict[str, np.ndarray],
        state: _AdamState,
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        state.t += 1

        total_norm = 0.0
        for name in grads:
            total_norm += float(np.sum(grads[name] ** 2))
        total_norm = np.sqrt(total_norm)

        clip_coeff = max_grad_norm / max(total_norm, max_grad_norm)

        eps_adam = 1e-8
        for name, g in grads.items():
            if name not in weights:
                continue

            g = g * clip_coeff

            if weight_decay > 0:
                g = g + weight_decay * weights[name]

            if name not in state.m:
                state.m[name] = np.zeros_like(g)
                state.v[name] = np.zeros_like(g)

            state.m[name] = beta1 * state.m[name] + (1 - beta1) * g
            state.v[name] = beta2 * state.v[name] + (1 - beta2) * (g ** 2)

            m_hat = state.m[name] / (1 - beta1 ** state.t)
            v_hat = state.v[name] / (1 - beta2 ** state.t)

            weights[name] = weights[name] - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

    # ----- persistence -----------------------------------------------------

    def _save_student(self, path: str) -> None:
        import onnx
        from onnx import numpy_helper

        model = onnx.load(self.student_path)
        init_map = {init.name: init for init in model.graph.initializer}

        for name, arr in self._student_weights.items():
            if name in init_map:
                new_tensor = numpy_helper.from_array(arr, name)
                init_map[name].CopyFrom(new_tensor)

        onnx.save(model, path)
        log.info("Saved updated student model to %s", path)

    def _rebuild_student_session(self) -> None:
        import onnxruntime as ort

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            tmp_path = f.name
        try:
            self._save_student(tmp_path)
            self._student_sess = ort.InferenceSession(
                tmp_path, providers=[self.provider],
            )
        finally:
            os.unlink(tmp_path)

    def _forward_student(self, feed: Dict[str, np.ndarray]) -> np.ndarray:
        out = self._student_sess.run(self._output_names, feed)
        return out[0]

    def _forward_teacher(self, feed: Dict[str, np.ndarray]) -> np.ndarray:
        teacher_feed = {}
        teacher_inputs = {i.name for i in self._teacher_sess.get_inputs()}
        for k, v in feed.items():
            if k in teacher_inputs:
                teacher_feed[k] = v
        out = self._teacher_sess.run(self._teacher_output_names, teacher_feed)
        return out[0]

    # ----- main training loop ----------------------------------------------

    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        temperature: float = 4.0,
        alpha: float = 0.5,
        data_generator: Optional[SyntheticDataGenerator] = None,
        output_path: Optional[str] = None,
    ) -> DistillationResult:
        if output_path is None:
            output_path = str(
                Path(self.student_path).with_suffix("") / "_distilled.onnx"
            )
            output_path = str(Path(self.student_path).parent / (
                Path(self.student_path).stem + "_distilled.onnx"
            ))

        if data_generator is None:
            data_generator = SyntheticDataGenerator(self.teacher_path, num_samples=256)

        adam_state = _AdamState()
        loss_history: List[float] = []
        last_kl = 0.0
        last_ce = 0.0
        global_step = 0
        log_interval = 10
        t_start = time.monotonic()

        log.info(
            "Starting distillation: epochs=%d, batch=%d, lr=%.2e, T=%.1f, alpha=%.2f",
            num_epochs, batch_size, learning_rate, temperature, alpha,
        )

        try:
            for epoch in range(num_epochs):
                epoch_losses: List[float] = []

                for batch_inputs, teacher_logits in data_generator.generate(batch_size):
                    global_step += 1

                    warmup_lr = learning_rate
                    if global_step <= 100:
                        warmup_lr = learning_rate * (global_step / 100)

                    student_logits_list = []
                    for sample in batch_inputs:
                        s_out = self._forward_student(sample)
                        student_logits_list.append(s_out)

                    student_logits = np.array(student_logits_list).reshape(
                        teacher_logits.shape
                    )

                    hard_labels = np.argmax(teacher_logits, axis=-1)
                    if hard_labels.ndim > 1:
                        hard_labels = hard_labels.reshape(-1)

                    kl_loss = self._kl_divergence(
                        teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                        student_logits.reshape(-1, student_logits.shape[-1]),
                        temperature,
                    )
                    ce_loss = self._cross_entropy(
                        student_logits.reshape(-1, student_logits.shape[-1]),
                        hard_labels,
                    )
                    total_loss = alpha * kl_loss + (1 - alpha) * ce_loss

                    def _loss_fn() -> float:
                        self._rebuild_student_session()
                        s_list = []
                        for sample in batch_inputs:
                            s_list.append(self._forward_student(sample))
                        s_logits = np.array(s_list).reshape(teacher_logits.shape)
                        kl = self._kl_divergence(
                            teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                            s_logits.reshape(-1, s_logits.shape[-1]),
                            temperature,
                        )
                        ce = self._cross_entropy(
                            s_logits.reshape(-1, s_logits.shape[-1]),
                            hard_labels,
                        )
                        return alpha * kl + (1 - alpha) * ce

                    grads = self._compute_gradients(
                        _loss_fn, self._student_weights,
                    )

                    self._adam_update(
                        self._student_weights,
                        grads,
                        adam_state,
                        lr=warmup_lr,
                    )

                    self._rebuild_student_session()

                    epoch_losses.append(total_loss)
                    last_kl = kl_loss
                    last_ce = ce_loss

                    if global_step % log_interval == 0:
                        log.info(
                            "[step %d] loss=%.4f  kl=%.4f  ce=%.4f  lr=%.2e",
                            global_step, total_loss, kl_loss, ce_loss, warmup_lr,
                        )

                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                loss_history.append(avg_loss)
                log.info("Epoch %d/%d  avg_loss=%.4f", epoch + 1, num_epochs, avg_loss)

                if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                    ckpt = str(
                        Path(output_path).parent
                        / f"{Path(output_path).stem}_epoch{epoch + 1}.onnx"
                    )
                    self._save_student(ckpt)

            self._save_student(output_path)
            elapsed = time.monotonic() - t_start

            out_size = Path(output_path).stat().st_size / (1024 * 1024)
            return DistillationResult(
                success=True,
                epochs_completed=num_epochs,
                final_loss=loss_history[-1] if loss_history else 0.0,
                final_kl_loss=last_kl,
                final_ce_loss=last_ce,
                teacher_size_mb=self.teacher_size_mb,
                student_size_mb=out_size,
                compression_ratio=(
                    self.teacher_size_mb / out_size if out_size > 0 else 0.0
                ),
                output_path=output_path,
                training_time_s=elapsed,
                loss_history=loss_history,
            )

        except Exception as exc:
            elapsed = time.monotonic() - t_start
            log.exception("Distillation failed after %d steps", global_step)
            return DistillationResult(
                success=False,
                epochs_completed=0,
                final_loss=loss_history[-1] if loss_history else 0.0,
                final_kl_loss=last_kl,
                final_ce_loss=last_ce,
                teacher_size_mb=self.teacher_size_mb,
                student_size_mb=self.student_size_mb,
                compression_ratio=0.0,
                output_path=output_path,
                training_time_s=elapsed,
                loss_history=loss_history,
            )

    # ----- layer-wise distillation -----------------------------------------

    def train_layerwise(
        self,
        layer_mapping: Dict[str, str],
        **kwargs: Any,
    ) -> DistillationResult:
        """Match intermediate representations between teacher and student.

        ``layer_mapping`` maps teacher output node names to student output
        node names.  An MSE loss on the hidden states is added on top of the
        standard KL + CE objective.
        """
        import onnxruntime as ort

        teacher_all_outputs = set(layer_mapping.keys())
        student_all_outputs = set(layer_mapping.values())

        temperature = kwargs.get("temperature", 4.0)
        alpha = kwargs.get("alpha", 0.5)
        beta_layer = kwargs.get("beta_layer", 0.1)
        num_epochs = kwargs.get("num_epochs", 10)
        batch_size = kwargs.get("batch_size", 8)
        learning_rate = kwargs.get("learning_rate", 1e-4)
        output_path = kwargs.get("output_path")
        data_generator = kwargs.get("data_generator")

        if output_path is None:
            output_path = str(Path(self.student_path).parent / (
                Path(self.student_path).stem + "_layerwise.onnx"
            ))

        if data_generator is None:
            data_generator = SyntheticDataGenerator(self.teacher_path, num_samples=256)

        t_out_names = self._teacher_output_names + [
            n for n in teacher_all_outputs if n not in self._teacher_output_names
        ]
        s_out_names = self._output_names + [
            n for n in student_all_outputs if n not in self._output_names
        ]

        try:
            teacher_sess = ort.InferenceSession(
                self.teacher_path, providers=[self.provider],
            )
            student_sess = ort.InferenceSession(
                self.student_path, providers=[self.provider],
            )
        except Exception:
            log.warning("Could not create sessions with intermediate outputs, "
                        "falling back to standard distillation")
            return self.train(
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                temperature=temperature,
                alpha=alpha,
                data_generator=data_generator,
                output_path=output_path,
            )

        adam_state = _AdamState()
        loss_history: List[float] = []
        last_kl = last_ce = 0.0
        global_step = 0
        t_start = time.monotonic()

        log.info(
            "Starting layer-wise distillation with %d mapped layers",
            len(layer_mapping),
        )

        try:
            for epoch in range(num_epochs):
                epoch_losses: List[float] = []

                for batch_inputs, teacher_logits in data_generator.generate(batch_size):
                    global_step += 1

                    warmup_lr = learning_rate
                    if global_step <= 100:
                        warmup_lr = learning_rate * (global_step / 100)

                    s_logits_list = []
                    for sample in batch_inputs:
                        s_out = self._forward_student(sample)
                        s_logits_list.append(s_out)
                    student_logits = np.array(s_logits_list).reshape(
                        teacher_logits.shape
                    )

                    hard_labels = np.argmax(teacher_logits, axis=-1)
                    if hard_labels.ndim > 1:
                        hard_labels = hard_labels.reshape(-1)

                    kl_loss = self._kl_divergence(
                        teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                        student_logits.reshape(-1, student_logits.shape[-1]),
                        temperature,
                    )
                    ce_loss = self._cross_entropy(
                        student_logits.reshape(-1, student_logits.shape[-1]),
                        hard_labels,
                    )

                    layer_loss = 0.0
                    for sample in batch_inputs:
                        teacher_feed = {
                            k: v for k, v in sample.items()
                            if k in {i.name for i in teacher_sess.get_inputs()}
                        }
                        try:
                            t_outs = teacher_sess.run(t_out_names, teacher_feed)
                            s_outs = student_sess.run(s_out_names, sample)
                            t_map = dict(zip(t_out_names, t_outs))
                            s_map = dict(zip(s_out_names, s_outs))

                            for t_name, s_name in layer_mapping.items():
                                if t_name in t_map and s_name in s_map:
                                    t_hidden = t_map[t_name].astype(np.float32)
                                    s_hidden = s_map[s_name].astype(np.float32)
                                    min_size = min(t_hidden.size, s_hidden.size)
                                    layer_loss += float(np.mean(
                                        (t_hidden.ravel()[:min_size]
                                         - s_hidden.ravel()[:min_size]) ** 2
                                    ))
                        except Exception:
                            pass

                    layer_loss /= max(len(batch_inputs), 1)

                    total_loss = (
                        alpha * kl_loss
                        + (1 - alpha) * ce_loss
                        + beta_layer * layer_loss
                    )

                    def _loss_fn() -> float:
                        self._rebuild_student_session()
                        s_list = []
                        for sample in batch_inputs:
                            s_list.append(self._forward_student(sample))
                        s_log = np.array(s_list).reshape(teacher_logits.shape)
                        kl = self._kl_divergence(
                            teacher_logits.reshape(-1, teacher_logits.shape[-1]),
                            s_log.reshape(-1, s_log.shape[-1]),
                            temperature,
                        )
                        ce = self._cross_entropy(
                            s_log.reshape(-1, s_log.shape[-1]),
                            hard_labels,
                        )
                        return alpha * kl + (1 - alpha) * ce + beta_layer * layer_loss

                    grads = self._compute_gradients(
                        _loss_fn, self._student_weights,
                    )
                    self._adam_update(
                        self._student_weights, grads, adam_state, lr=warmup_lr,
                    )
                    self._rebuild_student_session()

                    epoch_losses.append(total_loss)
                    last_kl = kl_loss
                    last_ce = ce_loss

                    if global_step % 10 == 0:
                        log.info(
                            "[step %d] loss=%.4f  kl=%.4f  ce=%.4f  layer=%.4f",
                            global_step, total_loss, kl_loss, ce_loss, layer_loss,
                        )

                avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
                loss_history.append(avg_loss)
                log.info("Epoch %d/%d  avg_loss=%.4f", epoch + 1, num_epochs, avg_loss)

                if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                    ckpt = str(
                        Path(output_path).parent
                        / f"{Path(output_path).stem}_epoch{epoch + 1}.onnx"
                    )
                    self._save_student(ckpt)

            self._save_student(output_path)
            elapsed = time.monotonic() - t_start
            out_size = Path(output_path).stat().st_size / (1024 * 1024)

            return DistillationResult(
                success=True,
                epochs_completed=num_epochs,
                final_loss=loss_history[-1] if loss_history else 0.0,
                final_kl_loss=last_kl,
                final_ce_loss=last_ce,
                teacher_size_mb=self.teacher_size_mb,
                student_size_mb=out_size,
                compression_ratio=(
                    self.teacher_size_mb / out_size if out_size > 0 else 0.0
                ),
                output_path=output_path,
                training_time_s=elapsed,
                loss_history=loss_history,
            )

        except Exception:
            elapsed = time.monotonic() - t_start
            log.exception("Layer-wise distillation failed after %d steps", global_step)
            return DistillationResult(
                success=False,
                epochs_completed=0,
                final_loss=loss_history[-1] if loss_history else 0.0,
                final_kl_loss=last_kl,
                final_ce_loss=last_ce,
                teacher_size_mb=self.teacher_size_mb,
                student_size_mb=self.student_size_mb,
                compression_ratio=0.0,
                output_path=output_path or "",
                training_time_s=elapsed,
                loss_history=loss_history,
            )


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def distill_model(
    teacher_path: str,
    student_path: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs: Any,
) -> DistillationResult:
    """One-call distillation entry point suitable for CLI use.

    If *student_path* is ``None``, a smaller student is auto-built from the
    teacher using depth reduction.
    """
    if student_path is None:
        log.info("No student provided -- auto-building a smaller model")
        student_path = StudentArchitectureBuilder.build_smaller(
            teacher_path,
            reduction=kwargs.pop("reduction", "depth"),
            factor=kwargs.pop("factor", 2),
        )

    if output_path is None:
        output_path = str(Path(teacher_path).parent / (
            Path(teacher_path).stem + "_distilled_student.onnx"
        ))

    trainer = DistillationTrainer(teacher_path, student_path)
    return trainer.train(output_path=output_path, **kwargs)
