"""Disaggregated prefill-decode controller.

Splits LLM inference into two phases on separate GPU pools:

- **Prefill workers** handle the compute-bound prompt encoding and produce a
  KV cache snapshot.
- **Decode workers** receive that KV cache and run the memory-bound
  autoregressive token generation.

A lightweight controller routes requests, manages KV transfer, and collects
latency telemetry.  All math uses NumPy; models run through ONNX Runtime
(lazy-imported so the module loads with zero extra deps).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.disaggregate")

_KV_PATTERNS = ("past_key_values", "past_key", "present", "cache")


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        out = np.zeros_like(logits, dtype=np.float64)
        out[np.argmax(logits)] = 1.0
        return out
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled -= np.max(scaled, axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or k >= probs.shape[-1]:
        return probs
    threshold = np.partition(probs, -k)[..., -k]
    mask = probs >= threshold[..., np.newaxis]
    filtered = probs * mask
    total = filtered.sum(axis=-1, keepdims=True)
    return np.where(total > 0, filtered / total, probs)


def _top_p_filter(probs: np.ndarray, p: float) -> np.ndarray:
    if p >= 1.0:
        return probs
    sorted_idx = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_idx, axis=-1)
    cum = np.cumsum(sorted_probs, axis=-1)
    cutoff = np.zeros_like(sorted_probs)
    cutoff[..., 1:] = cum[..., :-1]
    mask = cutoff < p
    sorted_probs *= mask
    total = sorted_probs.sum(axis=-1, keepdims=True)
    sorted_probs = np.where(total > 0, sorted_probs / total, sorted_probs)
    restore = np.argsort(sorted_idx, axis=-1)
    return np.take_along_axis(sorted_probs, restore, axis=-1)


def _sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> int:
    probs = _softmax(logits, temperature)
    probs = _top_k_filter(probs, top_k)
    probs = _top_p_filter(probs, top_p)
    probs = probs.ravel().astype(np.float64)
    total = probs.sum()
    if total < 1e-12:
        return int(np.argmax(logits.ravel()))
    probs /= total
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class KVTransferPacket:
    """Snapshot of KV cache produced by a prefill worker."""

    kv_cache: Dict[str, np.ndarray]
    seq_len: int
    transfer_size_mb: float = 0.0
    compression: str = "none"

    def __post_init__(self) -> None:
        if self.transfer_size_mb == 0.0 and self.kv_cache:
            total_bytes = sum(v.nbytes for v in self.kv_cache.values())
            self.transfer_size_mb = total_bytes / (1024 * 1024)


@dataclass
class DisaggregatedResult:
    """End-to-end result from a disaggregated inference request."""

    generated_ids: List[int] = field(default_factory=list)
    text: str = ""
    prefill_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    transfer_time_ms: float = 0.0
    prefill_worker_id: str = ""
    decode_worker_id: str = ""
    total_time_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"  Prefill time        : {self.prefill_time_ms:.1f} ms\n"
            f"  KV transfer time    : {self.transfer_time_ms:.1f} ms\n"
            f"  Decode time         : {self.decode_time_ms:.1f} ms\n"
            f"  Total time          : {self.total_time_ms:.1f} ms\n"
            f"  Tokens generated    : {len(self.generated_ids)}\n"
            f"  Prefill worker      : {self.prefill_worker_id}\n"
            f"  Decode worker       : {self.decode_worker_id}"
        )


# ---------------------------------------------------------------------------
# ORT session loader (lazy)
# ---------------------------------------------------------------------------

def _load_session(model_path: str, provider: str) -> Any:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=ort_providers(provider),
    )


# ---------------------------------------------------------------------------
# PrefillWorker
# ---------------------------------------------------------------------------

class PrefillWorker:
    """Runs the compute-bound prompt prefill and captures KV cache."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        worker_id: str = "prefill-0",
    ) -> None:
        self.model_path = model_path
        self.provider = provider
        self.worker_id = worker_id
        self.session = _load_session(model_path, provider)
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self.requests_served: int = 0
        self.total_prefill_ms: float = 0.0
        log.info("PrefillWorker %s ready  (%s)", worker_id, provider)

    # -----------------------------------------------------------------

    def prefill(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
    ) -> KVTransferPacket:
        """Encode the full prompt and return a KV cache packet."""
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        seq_len = input_ids.shape[1]

        feeds: Dict[str, np.ndarray] = {"input_ids": input_ids.astype(np.int64)}
        if attention_mask is not None:
            if attention_mask.ndim == 1:
                attention_mask = attention_mask[np.newaxis, :]
            feeds["attention_mask"] = attention_mask.astype(np.int64)
        elif "attention_mask" in self._input_names:
            feeds["attention_mask"] = np.ones_like(input_ids, dtype=np.int64)

        for name in self._input_names:
            if name not in feeds:
                inp_meta = next(i for i in self.session.get_inputs() if i.name == name)
                shape = []
                for d in inp_meta.shape:
                    shape.append(d if isinstance(d, int) else 0)
                feeds[name] = np.zeros(shape, dtype=np.float32)

        t0 = time.perf_counter()
        outputs = self.session.run(None, feeds)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        kv = self._extract_kv_from_outputs(outputs)
        self.requests_served += 1
        self.total_prefill_ms += elapsed_ms

        packet = KVTransferPacket(kv_cache=kv, seq_len=seq_len)
        log.info(
            "PrefillWorker %s  seq_len=%d  %.1f ms  kv=%.2f MB",
            self.worker_id, seq_len, elapsed_ms, packet.transfer_size_mb,
        )
        return packet

    # -----------------------------------------------------------------

    def _extract_kv_from_outputs(
        self, outputs: list[np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Parse ORT output names to find past_key_values tensors."""
        kv: Dict[str, np.ndarray] = {}
        for name, tensor in zip(self._output_names, outputs):
            if any(pat in name.lower() for pat in _KV_PATTERNS):
                kv[name] = tensor
        return kv


# ---------------------------------------------------------------------------
# DecodeWorker
# ---------------------------------------------------------------------------

class DecodeWorker:
    """Runs the memory-bound autoregressive decode using received KV cache."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        worker_id: str = "decode-0",
    ) -> None:
        self.model_path = model_path
        self.provider = provider
        self.worker_id = worker_id
        self.session = _load_session(model_path, provider)
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self.requests_served: int = 0
        self.total_decode_ms: float = 0.0
        log.info("DecodeWorker %s ready  (%s)", worker_id, provider)

    # -----------------------------------------------------------------

    def decode_step(
        self,
        input_id: np.ndarray,
        kv_cache: Dict[str, np.ndarray],
        position: int,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Run a single-token decode step with KV cache."""
        if input_id.ndim == 0:
            input_id = input_id.reshape(1, 1)
        elif input_id.ndim == 1:
            input_id = input_id[np.newaxis, :]

        feeds: Dict[str, np.ndarray] = {"input_ids": input_id.astype(np.int64)}

        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = np.ones(
                (1, position + 1), dtype=np.int64,
            )
        if "position_ids" in self._input_names:
            feeds["position_ids"] = np.array([[position]], dtype=np.int64)

        for name in self._input_names:
            if name in feeds:
                continue
            matched_kv = None
            for kv_name, kv_val in kv_cache.items():
                if _kv_names_match(name, kv_name):
                    matched_kv = kv_val
                    break
            if matched_kv is not None:
                feeds[name] = matched_kv
            elif name not in feeds:
                inp_meta = next(
                    i for i in self.session.get_inputs() if i.name == name
                )
                shape = []
                for d in inp_meta.shape:
                    shape.append(d if isinstance(d, int) else 0)
                feeds[name] = np.zeros(shape, dtype=np.float32)

        outputs = self.session.run(None, feeds)

        logits = outputs[0]
        updated_kv: Dict[str, np.ndarray] = {}
        for name, tensor in zip(self._output_names, outputs):
            if any(pat in name.lower() for pat in _KV_PATTERNS):
                updated_kv[name] = tensor

        return logits, updated_kv

    # -----------------------------------------------------------------

    def generate(
        self,
        kv_packet: KVTransferPacket,
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> tuple[List[int], float]:
        """Autoregressive generation loop using received KV cache.

        Returns (generated_ids, total_decode_ms).
        """
        generated: List[int] = []
        kv = dict(kv_packet.kv_cache)
        position = kv_packet.seq_len

        last_token = np.zeros((1, 1), dtype=np.int64)

        t0 = time.perf_counter()
        for step in range(max_tokens):
            logits, kv = self.decode_step(last_token, kv, position)

            next_logits = logits[0, -1] if logits.ndim == 3 else logits[0]
            token = _sample_token(next_logits, temperature, top_k, top_p)
            generated.append(token)
            last_token = np.array([[token]], dtype=np.int64)
            position += 1

        decode_ms = (time.perf_counter() - t0) * 1000
        self.requests_served += 1
        self.total_decode_ms += decode_ms

        log.info(
            "DecodeWorker %s  tokens=%d  %.1f ms  (%.1f tok/s)",
            self.worker_id,
            len(generated),
            decode_ms,
            len(generated) / max(decode_ms / 1000, 1e-9),
        )
        return generated, decode_ms


def _kv_names_match(input_name: str, output_name: str) -> bool:
    """Heuristic: ORT renames present.0.key -> past_key_values.0.key, etc."""
    in_lower = input_name.lower().replace("past_", "").replace("present_", "")
    out_lower = output_name.lower().replace("past_", "").replace("present_", "")
    return in_lower == out_lower


# ---------------------------------------------------------------------------
# DisaggregatedController
# ---------------------------------------------------------------------------

class DisaggregatedController:
    """Orchestrates disaggregated prefill and decode across GPU pools."""

    def __init__(
        self,
        model_path: str,
        prefill_providers: Optional[List[str]] = None,
        decode_providers: Optional[List[str]] = None,
    ) -> None:
        prefill_providers = prefill_providers or ["CPUExecutionProvider"]
        decode_providers = decode_providers or ["CPUExecutionProvider"]

        self.prefill_workers: List[PrefillWorker] = [
            PrefillWorker(model_path, prov, worker_id=f"prefill-{i}")
            for i, prov in enumerate(prefill_providers)
        ]
        self.decode_workers: List[DecodeWorker] = [
            DecodeWorker(model_path, prov, worker_id=f"decode-{i}")
            for i, prov in enumerate(decode_providers)
        ]
        self._prefill_idx = 0
        self._decode_idx = 0
        self._transfer_times: List[float] = []
        log.info(
            "DisaggregatedController  prefill=%d  decode=%d",
            len(self.prefill_workers), len(self.decode_workers),
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def process_request(
        self,
        prompt_ids: np.ndarray,
        max_tokens: int = 128,
        **sampling: Any,
    ) -> DisaggregatedResult:
        """Full disaggregated inference: prefill -> transfer -> decode."""
        wall_t0 = time.perf_counter()

        pw = self._select_prefill_worker()
        pf_t0 = time.perf_counter()
        packet = pw.prefill(prompt_ids)
        prefill_ms = (time.perf_counter() - pf_t0) * 1000

        tx_t0 = time.perf_counter()
        packet = self._transfer_kv(packet, compress=sampling.pop("compress", True))
        transfer_ms = (time.perf_counter() - tx_t0) * 1000
        self._transfer_times.append(transfer_ms)

        dw = self._select_decode_worker()
        generated, decode_ms = dw.generate(
            packet,
            max_tokens=max_tokens,
            temperature=sampling.get("temperature", 1.0),
            top_k=sampling.get("top_k", 0),
            top_p=sampling.get("top_p", 1.0),
        )

        total_ms = (time.perf_counter() - wall_t0) * 1000

        return DisaggregatedResult(
            generated_ids=generated,
            prefill_time_ms=prefill_ms,
            decode_time_ms=decode_ms,
            transfer_time_ms=transfer_ms,
            prefill_worker_id=pw.worker_id,
            decode_worker_id=dw.worker_id,
            total_time_ms=total_ms,
        )

    # -----------------------------------------------------------------
    # Worker selection
    # -----------------------------------------------------------------

    def _select_prefill_worker(self) -> PrefillWorker:
        """Round-robin with fallback to least-loaded."""
        workers = self.prefill_workers
        if len(workers) == 1:
            return workers[0]
        least = min(workers, key=lambda w: w.requests_served)
        rr = workers[self._prefill_idx % len(workers)]
        self._prefill_idx += 1
        if least.requests_served < rr.requests_served:
            return least
        return rr

    def _select_decode_worker(self) -> DecodeWorker:
        """Round-robin with fallback to least-loaded."""
        workers = self.decode_workers
        if len(workers) == 1:
            return workers[0]
        least = min(workers, key=lambda w: w.requests_served)
        rr = workers[self._decode_idx % len(workers)]
        self._decode_idx += 1
        if least.requests_served < rr.requests_served:
            return least
        return rr

    # -----------------------------------------------------------------
    # KV transfer
    # -----------------------------------------------------------------

    def _transfer_kv(
        self, packet: KVTransferPacket, compress: bool = True,
    ) -> KVTransferPacket:
        """Optionally compress KV cache (INT8 quantize) before transfer."""
        if not compress or not packet.kv_cache:
            return packet

        compressed: Dict[str, np.ndarray] = {}
        for name, tensor in packet.kv_cache.items():
            if tensor.dtype in (np.float32, np.float16):
                tmin = tensor.min()
                tmax = tensor.max()
                scale = (tmax - tmin) / 255.0 if (tmax - tmin) > 1e-12 else 1.0
                quantized = np.clip(
                    np.round((tensor - tmin) / scale), 0, 255,
                ).astype(np.uint8)
                dequantized = quantized.astype(np.float32) * scale + tmin
                compressed[name] = dequantized
            else:
                compressed[name] = tensor

        total_bytes = sum(v.nbytes for v in compressed.values())
        return KVTransferPacket(
            kv_cache=compressed,
            seq_len=packet.seq_len,
            transfer_size_mb=total_bytes / (1024 * 1024),
            compression="int8_dequant",
        )

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return worker utilization, transfer times, queue depths."""
        prefill_stats = [
            {
                "worker_id": w.worker_id,
                "provider": w.provider,
                "requests_served": w.requests_served,
                "total_prefill_ms": round(w.total_prefill_ms, 2),
                "avg_prefill_ms": round(
                    w.total_prefill_ms / max(w.requests_served, 1), 2,
                ),
            }
            for w in self.prefill_workers
        ]
        decode_stats = [
            {
                "worker_id": w.worker_id,
                "provider": w.provider,
                "requests_served": w.requests_served,
                "total_decode_ms": round(w.total_decode_ms, 2),
                "avg_decode_ms": round(
                    w.total_decode_ms / max(w.requests_served, 1), 2,
                ),
            }
            for w in self.decode_workers
        ]
        tx_times = self._transfer_times or [0.0]
        return {
            "prefill_workers": prefill_stats,
            "decode_workers": decode_stats,
            "transfer": {
                "count": len(self._transfer_times),
                "mean_ms": round(float(np.mean(tx_times)), 2),
                "p95_ms": round(float(np.percentile(tx_times, 95)), 2),
                "max_ms": round(float(np.max(tx_times)), 2),
            },
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def disaggregate_serve(
    model_path: str,
    prefill_providers: Optional[List[str]] = None,
    decode_providers: Optional[List[str]] = None,
    **kwargs: Any,
) -> DisaggregatedController:
    """Create and return a ``DisaggregatedController``.

    Intended as a CLI / programmatic entry point::

        ctrl = disaggregate_serve("model.onnx",
                                  prefill_providers=["CUDAExecutionProvider"],
                                  decode_providers=["CUDAExecutionProvider"])
        result = ctrl.process_request(prompt_ids, max_tokens=256)
    """
    log.info("Initializing disaggregated serving for %s", model_path)
    return DisaggregatedController(
        model_path,
        prefill_providers=prefill_providers,
        decode_providers=decode_providers,
    )
