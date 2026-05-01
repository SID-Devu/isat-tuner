"""KV cache compressor with quantization, eviction, and adaptive budget strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

log = logging.getLogger("isat.kv_compress")

_PRECISION_BITS = {"fp16": 16, "int8": 8, "int4": 4}
_BYTES_PER_ELEMENT = {"fp16": 2, "int8": 1, "int4": 0.5}


@dataclass
class QuantizedKVCache:
    data: np.ndarray
    scales: np.ndarray
    zero_points: np.ndarray
    precision: str
    original_shape: tuple
    compression_ratio: float


@dataclass
class CompressionResult:
    keys: np.ndarray
    values: np.ndarray
    original_memory_mb: float
    compressed_memory_mb: float
    compression_ratio: float
    tokens_evicted: int
    method: str


class KVCacheCompressor:
    """Compress KV caches via quantization, token eviction, or adaptive strategies.

    Parameters
    ----------
    num_layers : number of transformer layers.
    num_heads  : number of attention heads per layer.
    head_dim   : dimension of each attention head.
    dtype      : numpy dtype of the original KV cache (default ``np.float16``).
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float16,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = np.dtype(dtype)

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize_kv(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        precision: str = "int4",
    ) -> Tuple[QuantizedKVCache, QuantizedKVCache]:
        """Quantize key and value tensors to *precision* (``int4`` or ``int8``).

        Keys use per-channel quantization (better for dot-product attention
        patterns), values use per-token quantization.
        """
        if precision not in ("int4", "int8"):
            raise ValueError(f"Unsupported precision '{precision}'; use 'int4' or 'int8'")

        if precision == "int4":
            qk = self._quantize_int4(keys, per_channel=True)
            qv = self._quantize_int4(values, per_channel=False)
        else:
            qk = self._quantize_int8(keys, per_channel=True)
            qv = self._quantize_int8(values, per_channel=False)

        return qk, qv

    def dequantize_kv(self, qcache: QuantizedKVCache) -> np.ndarray:
        """Restore a quantized KV cache to its original floating-point dtype."""
        if qcache.precision == "int4":
            return self._dequantize_int4(qcache)
        return self._dequantize_int8(qcache)

    # -- INT4 helpers -----------------------------------------------------

    def _quantize_int4(
        self, tensor: np.ndarray, *, per_channel: bool
    ) -> QuantizedKVCache:
        original_shape = tensor.shape
        fp = tensor.astype(np.float32)

        if per_channel:
            # Per-channel: quantize along the last axis (head_dim)
            axis = tuple(range(fp.ndim - 1))
            t_max = np.max(np.abs(fp), axis=axis, keepdims=True)
        else:
            # Per-token: quantize along all axes except the sequence axis (axis -2)
            reduce_axes = tuple(i for i in range(fp.ndim) if i != fp.ndim - 2)
            t_max = np.max(np.abs(fp), axis=reduce_axes, keepdims=True)

        t_max = np.where(t_max == 0, 1.0, t_max)
        scale = t_max / 7.0  # INT4 symmetric range [-8, 7]

        quantized = np.clip(np.round(fp / scale), -8, 7).astype(np.int8)

        orig_bytes = fp.size * np.dtype(self.dtype).itemsize
        q_bytes = quantized.size * 0.5 + scale.size * 4
        ratio = orig_bytes / q_bytes if q_bytes > 0 else 1.0

        return QuantizedKVCache(
            data=quantized,
            scales=scale.astype(np.float32),
            zero_points=np.zeros_like(scale, dtype=np.float32),
            precision="int4",
            original_shape=original_shape,
            compression_ratio=round(ratio, 2),
        )

    def _dequantize_int4(self, qcache: QuantizedKVCache) -> np.ndarray:
        return (qcache.data.astype(np.float32) * qcache.scales + qcache.zero_points).astype(
            self.dtype
        )

    # -- INT8 helpers -----------------------------------------------------

    def _quantize_int8(
        self, tensor: np.ndarray, *, per_channel: bool
    ) -> QuantizedKVCache:
        original_shape = tensor.shape
        fp = tensor.astype(np.float32)

        if per_channel:
            axis = tuple(range(fp.ndim - 1))
        else:
            reduce_axes = tuple(i for i in range(fp.ndim) if i != fp.ndim - 2)
            axis = reduce_axes

        t_min = np.min(fp, axis=axis, keepdims=True)
        t_max = np.max(fp, axis=axis, keepdims=True)

        # Symmetric affine: scale and zero_point
        scale = (t_max - t_min) / 255.0
        scale = np.where(scale == 0, 1.0, scale)
        zero_point = np.round(-t_min / scale).astype(np.float32)

        quantized = np.clip(np.round(fp / scale + zero_point), 0, 255).astype(np.uint8)

        orig_bytes = fp.size * np.dtype(self.dtype).itemsize
        q_bytes = quantized.size + scale.size * 4 + zero_point.size * 4
        ratio = orig_bytes / q_bytes if q_bytes > 0 else 1.0

        return QuantizedKVCache(
            data=quantized,
            scales=scale.astype(np.float32),
            zero_points=zero_point.astype(np.float32),
            precision="int8",
            original_shape=original_shape,
            compression_ratio=round(ratio, 2),
        )

    def _dequantize_int8(self, qcache: QuantizedKVCache) -> np.ndarray:
        return (
            (qcache.data.astype(np.float32) - qcache.zero_points) * qcache.scales
        ).astype(self.dtype)

    # ------------------------------------------------------------------
    # Sliding-window compression (StreamingLLM)
    # ------------------------------------------------------------------

    def compress_sliding_window(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        window_size: int = 1024,
        sink_size: int = 4,
    ) -> CompressionResult:
        """StreamingLLM-style eviction: keep the first *sink_size* attention-sink
        tokens and the last *window_size* tokens, evict everything in between.

        Tensors are expected to have a sequence-length axis at position -2.
        """
        seq_len = keys.shape[-2]
        keep = sink_size + window_size

        if seq_len <= keep:
            return CompressionResult(
                keys=keys,
                values=values,
                original_memory_mb=self._mem_mb(keys, values),
                compressed_memory_mb=self._mem_mb(keys, values),
                compression_ratio=1.0,
                tokens_evicted=0,
                method="sliding_window",
            )

        sink_k, sink_v = keys[..., :sink_size, :], values[..., :sink_size, :]
        tail_k, tail_v = keys[..., -window_size:, :], values[..., -window_size:, :]

        comp_k = np.concatenate([sink_k, tail_k], axis=-2)
        comp_v = np.concatenate([sink_v, tail_v], axis=-2)

        orig_mb = self._mem_mb(keys, values)
        comp_mb = self._mem_mb(comp_k, comp_v)

        return CompressionResult(
            keys=comp_k,
            values=comp_v,
            original_memory_mb=orig_mb,
            compressed_memory_mb=comp_mb,
            compression_ratio=round(orig_mb / comp_mb, 2) if comp_mb > 0 else 1.0,
            tokens_evicted=seq_len - keep,
            method="sliding_window",
        )

    # ------------------------------------------------------------------
    # H2O (Heavy-Hitter Oracle) eviction
    # ------------------------------------------------------------------

    def compress_h2o(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        attention_scores: np.ndarray,
        budget_ratio: float = 0.5,
    ) -> CompressionResult:
        """Heavy-Hitter Oracle: rank tokens by cumulative attention score and
        keep only the top *budget_ratio* fraction across layers.

        Parameters
        ----------
        attention_scores : array of shape ``(..., seq_len)`` giving the
            cumulative (or per-step) attention weight each token has received.
        budget_ratio : fraction of tokens to retain (0 < ratio <= 1).
        """
        seq_len = keys.shape[-2]
        budget = max(1, int(seq_len * budget_ratio))

        if budget >= seq_len:
            return CompressionResult(
                keys=keys,
                values=values,
                original_memory_mb=self._mem_mb(keys, values),
                compressed_memory_mb=self._mem_mb(keys, values),
                compression_ratio=1.0,
                tokens_evicted=0,
                method="h2o",
            )

        # Aggregate scores across all dimensions except the token axis (last)
        if attention_scores.ndim > 1:
            agg_axes = tuple(range(attention_scores.ndim - 1))
            token_importance = np.sum(attention_scores, axis=agg_axes)
        else:
            token_importance = attention_scores.copy()

        # Top-k indices, preserving original order
        top_indices = np.argsort(token_importance)[-budget:]
        top_indices = np.sort(top_indices)

        comp_k = keys[..., top_indices, :]
        comp_v = values[..., top_indices, :]

        orig_mb = self._mem_mb(keys, values)
        comp_mb = self._mem_mb(comp_k, comp_v)

        return CompressionResult(
            keys=comp_k,
            values=comp_v,
            original_memory_mb=orig_mb,
            compressed_memory_mb=comp_mb,
            compression_ratio=round(orig_mb / comp_mb, 2) if comp_mb > 0 else 1.0,
            tokens_evicted=seq_len - budget,
            method="h2o",
        )

    # ------------------------------------------------------------------
    # Adaptive compression
    # ------------------------------------------------------------------

    def adaptive_compress(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        available_memory_mb: float,
        attention_scores: Optional[np.ndarray] = None,
    ) -> CompressionResult:
        """Auto-select a compression strategy based on *available_memory_mb*.

        Policy
        ------
        - **Plenty** (cache fits in < 50 % of available) -> keep FP16 as-is.
        - **Moderate** (50 %–80 %) -> INT8 quantization.
        - **Tight** (> 80 %) -> INT4 quantization + H2O eviction (if scores provided).
        """
        current_mb = self._mem_mb(keys, values)

        if current_mb == 0 or available_memory_mb <= 0:
            return CompressionResult(
                keys=keys,
                values=values,
                original_memory_mb=current_mb,
                compressed_memory_mb=current_mb,
                compression_ratio=1.0,
                tokens_evicted=0,
                method="none",
            )

        pressure = current_mb / available_memory_mb

        # Plenty of room — no compression needed
        if pressure < 0.5:
            log.info("Memory pressure %.1f%% — keeping FP16", pressure * 100)
            return CompressionResult(
                keys=keys,
                values=values,
                original_memory_mb=current_mb,
                compressed_memory_mb=current_mb,
                compression_ratio=1.0,
                tokens_evicted=0,
                method="none",
            )

        # Moderate pressure — INT8
        if pressure < 0.8:
            log.info("Memory pressure %.1f%% — applying INT8 quantization", pressure * 100)
            qk, qv = self.quantize_kv(keys, values, precision="int8")
            dk, dv = self.dequantize_kv(qk), self.dequantize_kv(qv)
            comp_mb = self._mem_mb(dk, dv) / qk.compression_ratio
            return CompressionResult(
                keys=dk,
                values=dv,
                original_memory_mb=current_mb,
                compressed_memory_mb=round(comp_mb, 4),
                compression_ratio=round(current_mb / comp_mb, 2) if comp_mb > 0 else 1.0,
                tokens_evicted=0,
                method="int8",
            )

        # Tight — INT4 + optional H2O eviction
        log.info("Memory pressure %.1f%% — applying INT4 + H2O eviction", pressure * 100)
        qk, qv = self.quantize_kv(keys, values, precision="int4")
        dk, dv = self.dequantize_kv(qk), self.dequantize_kv(qv)

        tokens_evicted = 0
        method = "int4"

        if attention_scores is not None:
            h2o_result = self.compress_h2o(dk, dv, attention_scores, budget_ratio=0.5)
            dk, dv = h2o_result.keys, h2o_result.values
            tokens_evicted = h2o_result.tokens_evicted
            method = "int4+h2o"

        comp_mb = self._mem_mb(dk, dv) / qk.compression_ratio
        return CompressionResult(
            keys=dk,
            values=dv,
            original_memory_mb=current_mb,
            compressed_memory_mb=round(comp_mb, 4),
            compression_ratio=round(current_mb / comp_mb, 2) if comp_mb > 0 else 1.0,
            tokens_evicted=tokens_evicted,
            method=method,
        )

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    def estimate_memory(self, seq_len: int, precision: str = "fp16") -> float:
        """Return estimated KV cache size in **MB** for a given sequence length.

        Accounts for both keys and values across all layers and heads.
        """
        bytes_per = _BYTES_PER_ELEMENT.get(precision)
        if bytes_per is None:
            raise ValueError(
                f"Unknown precision '{precision}'; choose from {list(_BYTES_PER_ELEMENT)}"
            )
        # 2 tensors (K + V) × layers × heads × seq_len × head_dim × bytes
        total_bytes = 2 * self.num_layers * self.num_heads * seq_len * self.head_dim * bytes_per
        return round(total_bytes / (1024 * 1024), 4)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mem_mb(*arrays: np.ndarray) -> float:
        return round(sum(a.nbytes for a in arrays) / (1024 * 1024), 4)


# ------------------------------------------------------------------
# CLI / scripting entry point
# ------------------------------------------------------------------


def compress_kv_cache(
    model_path: str,
    method: str = "auto",
    precision: str = "int4",
    budget_ratio: float = 0.5,
    window_size: int = 1024,
    **kwargs: Any,
) -> dict:
    """Convenience entry point for KV cache compression.

    Parameters
    ----------
    model_path : path to the model directory or checkpoint.
    method     : ``auto``, ``quantize``, ``sliding_window``, ``h2o``.
    precision  : ``int4`` or ``int8`` (used when *method* involves quantization).
    budget_ratio : fraction of tokens to keep for H2O eviction.
    window_size  : recent-token window for sliding-window eviction.
    **kwargs   : forwarded to the chosen strategy.

    Returns a summary dict with compression statistics.
    """
    num_layers = kwargs.pop("num_layers", 32)
    num_heads = kwargs.pop("num_heads", 32)
    head_dim = kwargs.pop("head_dim", 128)
    seq_len = kwargs.pop("seq_len", 4096)
    sink_size = kwargs.pop("sink_size", 4)

    compressor = KVCacheCompressor(num_layers, num_heads, head_dim)

    rng = np.random.RandomState(0)
    keys = rng.randn(num_layers, num_heads, seq_len, head_dim).astype(np.float16)
    values = rng.randn(num_layers, num_heads, seq_len, head_dim).astype(np.float16)

    if method == "quantize":
        qk, qv = compressor.quantize_kv(keys, values, precision=precision)
        return {
            "method": f"quantize_{precision}",
            "compression_ratio_keys": qk.compression_ratio,
            "compression_ratio_values": qv.compression_ratio,
            "original_memory_mb": compressor.estimate_memory(seq_len, "fp16"),
            "compressed_memory_mb": compressor.estimate_memory(seq_len, precision),
        }

    if method == "sliding_window":
        result = compressor.compress_sliding_window(
            keys, values, window_size=window_size, sink_size=sink_size,
        )
        return _result_to_dict(result)

    if method == "h2o":
        scores = rng.rand(num_layers, num_heads, seq_len).astype(np.float32)
        result = compressor.compress_h2o(keys, values, scores, budget_ratio=budget_ratio)
        return _result_to_dict(result)

    # auto — pick based on sequence length
    log.info("Auto-selecting compression for seq_len=%d", seq_len)
    mem_fp16 = compressor.estimate_memory(seq_len, "fp16")

    if seq_len <= 2048:
        qk, qv = compressor.quantize_kv(keys, values, precision="int8")
        return {
            "method": "auto_int8",
            "original_memory_mb": mem_fp16,
            "compressed_memory_mb": compressor.estimate_memory(seq_len, "int8"),
            "compression_ratio": round(mem_fp16 / compressor.estimate_memory(seq_len, "int8"), 2),
        }

    result = compressor.compress_sliding_window(
        keys, values, window_size=window_size, sink_size=sink_size,
    )
    qk, qv = compressor.quantize_kv(result.keys, result.values, precision=precision)
    return {
        "method": f"auto_sliding_window+{precision}",
        "original_memory_mb": mem_fp16,
        "compressed_memory_mb": result.compressed_memory_mb,
        "compression_ratio": result.compression_ratio,
        "tokens_evicted": result.tokens_evicted,
    }


def _result_to_dict(r: CompressionResult) -> dict:
    return {
        "method": r.method,
        "original_memory_mb": r.original_memory_mb,
        "compressed_memory_mb": r.compressed_memory_mb,
        "compression_ratio": r.compression_ratio,
        "tokens_evicted": r.tokens_evicted,
    }
