"""Long-context inference: sliding window, attention sinks, RoPE scaling,
chunked prefill, and ring attention for sequences up to 128K+ tokens.

All heavy imports (onnxruntime, onnx) are lazy so the module loads with
zero extra deps beyond numpy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger("isat.long_context")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RingPlan:
    """Execution plan produced by :class:`RingAttention`."""

    seq_len: int
    num_devices: int
    block_size: int
    device_assignments: List[Tuple[int, int, int]]  # (device, start, end)
    communication_schedule: List[Tuple[int, int]]    # (src_device, dst_device)


@dataclass
class ContextAnalysis:
    """Result of :meth:`LongContextEngine.analyze`."""

    native_context: int
    rope_type: str
    head_dim: int
    num_heads: int
    estimated_max_context: int
    recommended_method: str
    memory_at_max_mb: float


# ---------------------------------------------------------------------------
# SlidingWindowAttention
# ---------------------------------------------------------------------------

class SlidingWindowAttention:
    """Local attention window that limits each token to attending at most
    *window_size* preceding tokens.

    Parameters
    ----------
    window_size : number of preceding tokens each position can attend to.
    """

    def __init__(self, window_size: int = 4096) -> None:
        self.window_size = window_size

    def create_mask(self, seq_len: int) -> np.ndarray:
        """Return a boolean attention mask of shape ``(seq_len, seq_len)``.

        ``mask[i, j]`` is ``True`` when position *j* is within the local
        window of position *i* (i.e. ``0 <= i - j < window_size``).
        """
        rows = np.arange(seq_len)[:, None]
        cols = np.arange(seq_len)[None, :]
        diff = rows - cols
        return (diff >= 0) & (diff < self.window_size)

    def apply(self, attention_scores: np.ndarray, seq_len: int) -> np.ndarray:
        """Mask out-of-window positions to ``-inf``.

        Parameters
        ----------
        attention_scores : array of shape ``(..., seq_len, seq_len)``.
        seq_len : sequence length (last two dims of *attention_scores*).
        """
        mask = self.create_mask(seq_len)
        masked = attention_scores.copy()
        masked[..., ~mask] = -np.inf
        return masked


# ---------------------------------------------------------------------------
# AttentionSink (StreamingLLM)
# ---------------------------------------------------------------------------

class AttentionSink:
    """StreamingLLM-style KV eviction: always keep the first *sink_size*
    tokens (attention sinks) and the most recent *window_size* tokens,
    evicting everything in between.

    Parameters
    ----------
    sink_size   : number of initial tokens to always retain.
    window_size : number of recent tokens to keep.
    """

    def __init__(self, sink_size: int = 4, window_size: int = 1024) -> None:
        self.sink_size = sink_size
        self.window_size = window_size

    def compress(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        current_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evict middle tokens if the sequence exceeds capacity.

        Parameters
        ----------
        keys, values : arrays with the token dimension at ``axis=-2``.
        current_len  : number of tokens currently stored.

        Returns
        -------
        compressed_keys, compressed_values, position_mapping
            *position_mapping* maps each retained token index to its
            original position.
        """
        capacity = self.sink_size + self.window_size
        if current_len <= capacity:
            positions = np.arange(current_len, dtype=np.int64)
            return keys, values, positions

        sink_k = keys[..., :self.sink_size, :]
        sink_v = values[..., :self.sink_size, :]
        window_k = keys[..., current_len - self.window_size:current_len, :]
        window_v = values[..., current_len - self.window_size:current_len, :]

        compressed_keys = np.concatenate([sink_k, window_k], axis=-2)
        compressed_values = np.concatenate([sink_v, window_v], axis=-2)
        position_mapping = self.get_position_ids(current_len)
        return compressed_keys, compressed_values, position_mapping

    def get_position_ids(self, current_len: int) -> np.ndarray:
        """Return position IDs accounting for evicted tokens.

        Sink tokens keep their original positions ``[0 .. sink_size)``.
        Window tokens are re-indexed starting at ``sink_size`` so that
        relative distances within the window are preserved.
        """
        capacity = self.sink_size + self.window_size
        if current_len <= capacity:
            return np.arange(current_len, dtype=np.int64)

        sink_ids = np.arange(self.sink_size, dtype=np.int64)
        window_ids = np.arange(
            self.sink_size,
            self.sink_size + self.window_size,
            dtype=np.int64,
        )
        return np.concatenate([sink_ids, window_ids])


# ---------------------------------------------------------------------------
# RoPEScaler
# ---------------------------------------------------------------------------

class RoPEScaler:
    """Scale Rotary Position Embeddings to extend context length.

    Supported methods
    -----------------
    linear  : divide positions by a constant scale factor.
    ntk     : NTK-aware scaling — modify the RoPE base frequency.
    yarn    : YaRN — interpolate between linear and NTK per frequency band.
    dynamic : adjust scaling factor based on actual sequence length.

    Parameters
    ----------
    method         : one of ``"linear"``, ``"ntk"``, ``"yarn"``, ``"dynamic"``.
    base_context   : model's original (trained) context length.
    target_context : desired extended context length.
    base           : RoPE base frequency (default 10 000).
    """

    _METHODS = {"linear", "ntk", "yarn", "dynamic"}

    def __init__(
        self,
        method: str = "ntk",
        base_context: int = 4096,
        target_context: int = 32768,
        base: float = 10000.0,
    ) -> None:
        if method not in self._METHODS:
            raise ValueError(
                f"Unknown RoPE scaling method {method!r}. "
                f"Choose from {sorted(self._METHODS)}."
            )
        self.method = method
        self.base_context = base_context
        self.target_context = target_context
        self.base = base
        self.scale_factor = target_context / base_context

    # -- core -----------------------------------------------------------------

    def scale(self, positions: np.ndarray, dim: int) -> np.ndarray:
        """Return scaled rotary embeddings (sin/cos interleaved).

        Parameters
        ----------
        positions : int array of shape ``(seq_len,)`` — token positions.
        dim       : head dimension (must be even).

        Returns
        -------
        Array of shape ``(seq_len, dim)`` containing ``[sin, cos]`` pairs.
        """
        half = dim // 2
        if self.method == "linear":
            freqs = self._linear_freqs(positions, half)
        elif self.method == "ntk":
            freqs = self._ntk_freqs(positions, half, dim)
        elif self.method == "yarn":
            freqs = self._yarn_freqs(positions, half, dim)
        elif self.method == "dynamic":
            freqs = self._dynamic_freqs(positions, half)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        sin_part = np.sin(freqs)
        cos_part = np.cos(freqs)
        return np.concatenate([sin_part, cos_part], axis=-1)

    # -- per-method frequency computation -------------------------------------

    def _inv_freq(self, half: int, base: float | None = None) -> np.ndarray:
        b = base if base is not None else self.base
        return 1.0 / (b ** (np.arange(0, half * 2, 2, dtype=np.float64) / (half * 2)))

    def _linear_freqs(self, positions: np.ndarray, half: int) -> np.ndarray:
        scaled_pos = positions.astype(np.float64) / self.scale_factor
        inv = self._inv_freq(half)
        return scaled_pos[:, None] * inv[None, :]

    def _ntk_freqs(
        self, positions: np.ndarray, half: int, dim: int,
    ) -> np.ndarray:
        base_new = self.base * (self.scale_factor ** (dim / (dim - 2)))
        inv = self._inv_freq(half, base=base_new)
        return positions.astype(np.float64)[:, None] * inv[None, :]

    def _yarn_freqs(
        self, positions: np.ndarray, half: int, dim: int,
    ) -> np.ndarray:
        linear_inv = self._inv_freq(half) / self.scale_factor
        base_ntk = self.base * (self.scale_factor ** (dim / (dim - 2)))
        ntk_inv = self._inv_freq(half, base=base_ntk)

        freq_idx = np.arange(half, dtype=np.float64)
        ramp = freq_idx / max(half - 1, 1)
        blended = (1.0 - ramp) * linear_inv + ramp * ntk_inv

        return positions.astype(np.float64)[:, None] * blended[None, :]

    def _dynamic_freqs(self, positions: np.ndarray, half: int) -> np.ndarray:
        max_pos = int(positions.max()) + 1 if positions.size else 1
        if max_pos <= self.base_context:
            inv = self._inv_freq(half)
        else:
            dynamic_scale = max_pos / self.base_context
            base_new = self.base * (dynamic_scale ** (half * 2 / (half * 2 - 2)))
            inv = self._inv_freq(half, base=base_new)
        return positions.astype(np.float64)[:, None] * inv[None, :]

    # -- ONNX patching --------------------------------------------------------

    def apply_to_model(self, model_path: str, output_path: str) -> None:
        """Modify RoPE parameters in an ONNX model on disk.

        Scans the graph for constant nodes whose names suggest RoPE
        inverse-frequency tensors and replaces them with scaled versions.
        """
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ImportError(
                "onnx is required for apply_to_model. "
                "Install with: pip install onnx"
            ) from exc

        model = onnx.load(model_path)
        patched = 0
        for init in model.graph.initializer:
            name_lower = init.name.lower()
            if "inv_freq" not in name_lower and "rope" not in name_lower:
                continue
            arr = numpy_helper.to_array(init)
            if arr.ndim != 1:
                continue
            half = arr.shape[0]
            dim = half * 2
            if self.method == "ntk":
                new_base = self.base * (self.scale_factor ** (dim / (dim - 2)))
            elif self.method == "yarn":
                new_base = self.base * (self.scale_factor ** (dim / (dim - 2)))
            else:
                new_base = self.base
            new_arr = self._inv_freq(half, base=new_base).astype(arr.dtype)
            new_tensor = numpy_helper.from_array(new_arr, name=init.name)
            init.CopyFrom(new_tensor)
            patched += 1
            log.info("Patched RoPE initializer %s (dim=%d)", init.name, dim)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_path)
        log.info(
            "Saved RoPE-scaled model to %s (%d initializers patched)",
            output_path, patched,
        )


# ---------------------------------------------------------------------------
# ChunkedPrefill
# ---------------------------------------------------------------------------

class ChunkedPrefill:
    """Process long prompts in fixed-size chunks to bound peak memory.

    Parameters
    ----------
    chunk_size : maximum number of tokens per forward pass.
    """

    def __init__(self, chunk_size: int = 4096) -> None:
        self.chunk_size = chunk_size

    def prefill(
        self,
        session: Any,
        input_ids: np.ndarray,
        chunk_size: int | None = None,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Run chunked prefill through an ONNX Runtime session.

        Parameters
        ----------
        session   : ``onnxruntime.InferenceSession`` (or compatible).
        input_ids : 1-D or 2-D int array of token IDs.
        chunk_size: override the instance-level chunk size.

        Returns
        -------
        (kv_caches, last_logits)
            *kv_caches* is a list of KV arrays accumulated across chunks.
            *last_logits* are logits from the final chunk.
        """
        cs = chunk_size or self.chunk_size
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        seq_len = input_ids.shape[1]

        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        kv_names = [n for n in input_names if "past" in n.lower() or "cache" in n.lower()]
        logit_outputs = [n for n in output_names if "logit" in n.lower()]
        kv_outputs = [n for n in output_names if "present" in n.lower() or "cache" in n.lower()]

        kv_cache: Dict[str, np.ndarray] = {}
        last_logits: Optional[np.ndarray] = None

        for start in range(0, seq_len, cs):
            end = min(start + cs, seq_len)
            chunk = input_ids[:, start:end]

            feeds: Dict[str, np.ndarray] = {"input_ids": chunk}
            pos = np.arange(start, end, dtype=np.int64)[None, :]
            if "position_ids" in input_names:
                feeds["position_ids"] = pos
            if "attention_mask" in input_names:
                mask_len = end if not kv_cache else end
                feeds["attention_mask"] = np.ones((1, mask_len), dtype=np.int64)
            for kn in kv_names:
                if kn in kv_cache:
                    feeds[kn] = kv_cache[kn]
                else:
                    meta = next(
                        (i for i in session.get_inputs() if i.name == kn), None,
                    )
                    if meta is not None:
                        shape = [
                            d if isinstance(d, int) else 1
                            for d in meta.shape
                        ]
                        seq_dim = next(
                            (i for i, d in enumerate(meta.shape)
                             if not isinstance(d, int)), -2,
                        )
                        shape[seq_dim] = 0
                        feeds[kn] = np.zeros(shape, dtype=np.float16)

            outputs = session.run(None, feeds)
            out_map = dict(zip(output_names, outputs))

            for kn_out in kv_outputs:
                kv_cache[kn_out.replace("present", "past")] = out_map[kn_out]

            if logit_outputs:
                last_logits = out_map[logit_outputs[0]]

        caches = list(kv_cache.values())
        if last_logits is None:
            last_logits = np.zeros((1, 1), dtype=np.float32)
        return caches, last_logits

    def estimate_memory(self, seq_len: int, model_path: str) -> float:
        """Estimate peak memory (MB) during chunked prefill.

        Uses the ONNX model metadata to infer KV cache dimensions.
        """
        try:
            import onnx
        except ImportError:
            log.warning("onnx not installed — returning rough estimate")
            return seq_len * 0.004  # ~4 KB/token heuristic

        model = onnx.load(model_path, load_external_data=False)

        num_layers = 0
        head_dim = 0
        num_heads = 0
        for inp in model.graph.input:
            name = inp.name.lower()
            if "past" not in name and "cache" not in name:
                continue
            num_layers += 1
            dims = [
                d.dim_value for d in inp.type.tensor_type.shape.dim
                if d.dim_value > 0
            ]
            if len(dims) >= 2:
                num_heads = max(num_heads, dims[-3] if len(dims) >= 3 else 1)
                head_dim = max(head_dim, dims[-1])
        num_layers //= 2  # key + value per layer

        if num_layers == 0 or head_dim == 0:
            return seq_len * 0.004

        bytes_per_elem = 2  # fp16
        chunk_tokens = min(self.chunk_size, seq_len)
        kv_per_chunk = (
            2 * num_layers * num_heads * chunk_tokens * head_dim * bytes_per_elem
        )
        total_kv = (
            2 * num_layers * num_heads * seq_len * head_dim * bytes_per_elem
        )
        peak = kv_per_chunk + total_kv
        return peak / (1024 * 1024)


# ---------------------------------------------------------------------------
# RingAttention
# ---------------------------------------------------------------------------

class RingAttention:
    """Distribute long sequences across multiple devices in a ring topology.

    Each device holds a contiguous block of the sequence and performs local
    attention.  KV blocks are passed around the ring so every query block
    eventually attends to every key block.  Online softmax is used for
    numerically stable accumulation.

    Parameters
    ----------
    num_devices : number of devices / ring participants.
    block_size  : tokens per block assigned to each device.
    """

    def __init__(self, num_devices: int = 2, block_size: int = 4096) -> None:
        self.num_devices = num_devices
        self.block_size = block_size

    def plan(self, seq_len: int) -> RingPlan:
        """Produce a :class:`RingPlan` describing the token-to-device mapping
        and the communication schedule.
        """
        assignments: List[Tuple[int, int, int]] = []
        tokens_per_device = math.ceil(seq_len / self.num_devices)
        for dev in range(self.num_devices):
            start = dev * tokens_per_device
            end = min(start + tokens_per_device, seq_len)
            if start < seq_len:
                assignments.append((dev, start, end))

        schedule: List[Tuple[int, int]] = []
        for step in range(self.num_devices - 1):
            for dev in range(self.num_devices):
                dst = (dev + 1) % self.num_devices
                schedule.append((dev, dst))

        return RingPlan(
            seq_len=seq_len,
            num_devices=self.num_devices,
            block_size=self.block_size,
            device_assignments=assignments,
            communication_schedule=schedule,
        )

    def execute(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        queries: np.ndarray,
        plan: RingPlan,
    ) -> np.ndarray:
        """Simulate ring attention across devices.

        Uses online softmax: each device accumulates a running weighted sum
        and a running log-sum-exp so that the final result is numerically
        identical (in exact arithmetic) to full attention.

        Parameters
        ----------
        keys, values, queries : arrays of shape
            ``(num_heads, seq_len, head_dim)``.
        plan : a :class:`RingPlan` returned by :meth:`plan`.

        Returns
        -------
        Output array of shape ``(num_heads, seq_len, head_dim)``.
        """
        num_heads, _, head_dim = queries.shape
        output = np.zeros_like(queries)

        local_kv: List[Tuple[np.ndarray, np.ndarray]] = []
        for dev, start, end in plan.device_assignments:
            local_kv.append((keys[:, start:end, :], values[:, start:end, :]))

        for dev_idx, (dev_id, q_start, q_end) in enumerate(plan.device_assignments):
            q_block = queries[:, q_start:q_end, :]
            q_len = q_end - q_start

            running_sum = np.zeros((num_heads, q_len, head_dim), dtype=np.float64)
            running_lse = np.full((num_heads, q_len, 1), -np.inf, dtype=np.float64)

            ring_kv = list(local_kv)
            order = [(dev_idx + step) % self.num_devices
                     for step in range(self.num_devices)]

            for src_idx in order:
                if src_idx >= len(ring_kv):
                    continue
                k_block, v_block = ring_kv[src_idx]

                scores = np.matmul(
                    q_block.astype(np.float64),
                    k_block.astype(np.float64).transpose(0, 2, 1),
                ) / math.sqrt(head_dim)

                block_max = scores.max(axis=-1, keepdims=True)
                exp_scores = np.exp(scores - block_max)
                block_lse = block_max + np.log(
                    exp_scores.sum(axis=-1, keepdims=True) + 1e-30,
                )
                block_attn = np.matmul(
                    exp_scores
                    / (exp_scores.sum(axis=-1, keepdims=True) + 1e-30),
                    v_block.astype(np.float64),
                )

                new_max = np.maximum(running_lse, block_lse)
                old_w = np.exp(running_lse - new_max)
                new_w = np.exp(block_lse - new_max)

                running_sum = old_w * running_sum + new_w * block_attn
                running_lse = new_max + np.log(old_w + new_w + 1e-30)

            safe_lse = running_lse.copy()
            safe_lse[safe_lse == -np.inf] = 0.0
            normalizer = np.exp(running_lse - safe_lse)
            normalizer[normalizer == 0] = 1.0
            output[:, q_start:q_end, :] = (
                running_sum / normalizer
            ).astype(queries.dtype)

        return output


# ---------------------------------------------------------------------------
# LongContextEngine
# ---------------------------------------------------------------------------

class LongContextEngine:
    """Unified long-context inference engine.

    Wraps RoPE scaling, sliding window attention, attention sinks, chunked
    prefill, and ring attention behind a single high-level API.

    Parameters
    ----------
    model_path : path to an ONNX model.
    provider   : ONNX Runtime execution provider.
    max_context: hard ceiling on extended context length.
    """

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        max_context: int = 131072,
    ) -> None:
        self.model_path = model_path
        self.provider = provider
        self.max_context = max_context
        self._session: Any = None

        analysis = self.analyze(model_path)
        self.native_context = analysis.native_context
        self.head_dim = analysis.head_dim
        self.num_heads = analysis.num_heads
        self.rope_type = analysis.rope_type

        self.sliding = SlidingWindowAttention()
        self.sinks = AttentionSink()
        self.chunked = ChunkedPrefill()
        self.scaler: Optional[RoPEScaler] = None

    # -- lazy session ---------------------------------------------------------

    def _get_session(self) -> Any:
        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required. "
                    "Install with: pip install onnxruntime"
                ) from exc
            self._session = ort.InferenceSession(
                self.model_path, providers=[self.provider],
            )
        return self._session

    # -- context extension ----------------------------------------------------

    def extend_context(
        self, target_length: int, method: str = "auto",
    ) -> RoPEScaler:
        """Select and configure a RoPE scaling strategy.

        When *method* is ``"auto"``:
        - target <= 2x native  → linear
        - 2x < target <= 8x   → ntk
        - target > 8x         → yarn + sliding window (window = native)
        """
        ratio = target_length / max(self.native_context, 1)

        if method == "auto":
            if ratio <= 2.0:
                method = "linear"
            elif ratio <= 8.0:
                method = "ntk"
            else:
                method = "yarn"
                self.sliding = SlidingWindowAttention(
                    window_size=self.native_context,
                )
                log.info(
                    "High extension ratio (%.1fx) — enabling sliding window "
                    "(w=%d) alongside YaRN", ratio, self.native_context,
                )

        self.scaler = RoPEScaler(
            method=method,
            base_context=self.native_context,
            target_context=target_length,
        )
        log.info(
            "Context extension: %s, %d → %d (%.1fx)",
            method, self.native_context, target_length, ratio,
        )
        return self.scaler

    # -- generation -----------------------------------------------------------

    def generate(
        self,
        input_ids: np.ndarray,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> np.ndarray:
        """Generate tokens with long-context support.

        Uses chunked prefill for the prompt phase and attention sinks for
        unbounded generation.

        Parameters
        ----------
        input_ids  : 1-D or 2-D int array.
        max_tokens : maximum new tokens to generate.
        **kwargs   : forwarded to the ONNX session.

        Returns
        -------
        Array of generated token IDs (prompt + new tokens).
        """
        session = self._get_session()
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        seq_len = input_ids.shape[1]

        if seq_len > self.native_context and self.scaler is None:
            self.extend_context(seq_len)

        kv_caches, logits = self.chunked.prefill(session, input_ids)

        generated: List[int] = []
        for _ in range(max_tokens):
            next_token = int(logits[0, -1].argmax())
            generated.append(next_token)

            feeds: Dict[str, np.ndarray] = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
            }
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            kv_input_names = [
                n for n in input_names
                if "past" in n.lower() or "cache" in n.lower()
            ]

            current_pos = seq_len + len(generated)
            if "position_ids" in input_names:
                feeds["position_ids"] = np.array(
                    [[current_pos - 1]], dtype=np.int64,
                )
            if "attention_mask" in input_names:
                feeds["attention_mask"] = np.ones(
                    (1, current_pos), dtype=np.int64,
                )

            for idx, kn in enumerate(kv_input_names):
                if idx < len(kv_caches):
                    kv = kv_caches[idx]
                    cache_len = kv.shape[-2] if kv.ndim >= 3 else kv.shape[-1]
                    capacity = self.sinks.sink_size + self.sinks.window_size
                    if cache_len > capacity:
                        kv, _, _ = self.sinks.compress(
                            kv, kv, cache_len,
                        )
                    feeds[kn] = kv

            outputs = session.run(None, feeds)
            out_map = dict(zip(output_names, outputs))

            kv_out_names = [
                n for n in output_names
                if "present" in n.lower() or "cache" in n.lower()
            ]
            kv_caches = [out_map[n] for n in kv_out_names]

            logit_names = [n for n in output_names if "logit" in n.lower()]
            if logit_names:
                logits = out_map[logit_names[0]]

        new_ids = np.array(generated, dtype=np.int64)
        return np.concatenate([input_ids.flatten(), new_ids])

    # -- analysis -------------------------------------------------------------

    def analyze(self, model_path: str | None = None) -> ContextAnalysis:
        """Inspect an ONNX model and return a :class:`ContextAnalysis`.

        Parameters
        ----------
        model_path : defaults to ``self.model_path`` if *None*.
        """
        path = model_path or self.model_path

        native_ctx = 2048
        rope_type = "unknown"
        head_dim = 64
        num_heads = 1

        try:
            import onnx
        except ImportError:
            log.warning("onnx not installed — returning defaults")
            return ContextAnalysis(
                native_context=native_ctx,
                rope_type=rope_type,
                head_dim=head_dim,
                num_heads=num_heads,
                estimated_max_context=native_ctx * 8,
                recommended_method="ntk",
                memory_at_max_mb=0.0,
            )

        model = onnx.load(path, load_external_data=False)

        for inp in model.graph.input:
            name = inp.name.lower()
            if "past" not in name and "cache" not in name:
                continue
            dims = [
                d.dim_value
                for d in inp.type.tensor_type.shape.dim
                if d.dim_value > 0
            ]
            if len(dims) >= 2:
                head_dim = dims[-1]
                if len(dims) >= 3:
                    num_heads = dims[-3]

        for init in model.graph.initializer:
            nl = init.name.lower()
            if "inv_freq" in nl or "rope" in nl:
                rope_type = "rotary"
                from onnx import numpy_helper
                arr = numpy_helper.to_array(init)
                if arr.ndim == 1:
                    head_dim = arr.shape[0] * 2
                break

        for inp in model.graph.input:
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_param and "seq" in d.dim_param.lower():
                    pass

        for prop in model.metadata_props:
            if "context" in prop.key.lower() or "max_position" in prop.key.lower():
                try:
                    native_ctx = int(prop.value)
                except ValueError:
                    pass

        est_max = min(native_ctx * 16, self.max_context)
        ratio = est_max / max(native_ctx, 1)
        if ratio <= 2:
            rec = "linear"
        elif ratio <= 8:
            rec = "ntk"
        else:
            rec = "yarn"

        bytes_per_elem = 2
        num_layers_est = sum(
            1 for inp in model.graph.input
            if "past" in inp.name.lower() or "cache" in inp.name.lower()
        ) // 2
        mem_mb = (
            2 * max(num_layers_est, 1) * num_heads * est_max
            * head_dim * bytes_per_elem
        ) / (1024 * 1024)

        return ContextAnalysis(
            native_context=native_ctx,
            rope_type=rope_type,
            head_dim=head_dim,
            num_heads=num_heads,
            estimated_max_context=est_max,
            recommended_method=rec,
            memory_at_max_mb=round(mem_mb, 2),
        )


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def long_context_generate(
    model_path: str,
    input_ids: np.ndarray,
    target_context: int = 32768,
    **kwargs: Any,
) -> np.ndarray:
    """CLI entry point for long-context generation.

    Creates a :class:`LongContextEngine`, extends context to
    *target_context*, and generates tokens.

    Parameters
    ----------
    model_path     : path to an ONNX model.
    input_ids      : 1-D or 2-D int array of prompt token IDs.
    target_context : desired context length.
    **kwargs       : forwarded to :meth:`LongContextEngine.generate`.

    Returns
    -------
    Array of all token IDs (prompt + generated).
    """
    engine = LongContextEngine(model_path)
    engine.extend_context(target_context)
    return engine.generate(input_ids, **kwargs)
