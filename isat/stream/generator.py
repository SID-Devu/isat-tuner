"""Autoregressive streaming generator backed by ONNX Runtime."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence

import numpy as np

log = logging.getLogger("isat.stream")


@dataclass
class StreamMetrics:
    """Latency and throughput statistics from a streaming benchmark run."""

    ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    p50_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    total_time_ms: float = 0.0


def _sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> int:
    """Sample a single token from a logits vector using top-k / nucleus sampling."""
    logits = logits.astype(np.float64)

    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < threshold, -np.inf, logits)

    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probs = exp_logits / np.sum(exp_logits)

    if top_p < 1.0:
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff_mask = cumulative - sorted_probs > top_p
        sorted_probs[cutoff_mask] = 0.0
        probs = np.zeros_like(probs)
        probs[sorted_indices] = sorted_probs
        total = probs.sum()
        if total > 0:
            probs /= total

    return int(np.random.choice(len(probs), p=probs))


_KV_PATTERN = re.compile(r"past_key_values\.(\d+)\.(key|value)")


def _build_kv_cache_inputs(
    session,
    batch_size: int = 1,
    past_seq_len: int = 0,
) -> dict[str, np.ndarray]:
    """Inspect an ONNX session for KV-cache inputs and return zero-filled tensors.

    Matches input names like ``past_key_values.0.key``,
    ``past_key_values.0.value``, etc.  The shape is inferred from the model's
    input metadata (symbolic dims replaced with *batch_size* / *past_seq_len*).
    """
    cache_inputs: dict[str, np.ndarray] = {}

    for inp in session.get_inputs():
        m = _KV_PATTERN.search(inp.name)
        if m is None:
            continue

        raw_shape = inp.shape
        resolved: list[int] = []
        for i, dim in enumerate(raw_shape):
            if isinstance(dim, int):
                resolved.append(dim)
            elif i == 0:
                resolved.append(batch_size)
            elif "seq" in str(dim).lower() or "past" in str(dim).lower():
                resolved.append(past_seq_len)
            else:
                resolved.append(dim if isinstance(dim, int) else 1)

        dtype_str = inp.type
        if "float16" in dtype_str:
            np_dtype = np.float16
        elif "float" in dtype_str:
            np_dtype = np.float32
        else:
            np_dtype = np.float32

        cache_inputs[inp.name] = np.zeros(resolved, dtype=np_dtype)
        log.debug("KV cache input %s  shape=%s  dtype=%s", inp.name, resolved, np_dtype)

    return cache_inputs


class StreamingGenerator:
    """Token-by-token autoregressive generator using an ONNX Runtime session."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for StreamingGenerator.  "
                "Install it with:  pip install onnxruntime"
            ) from exc

        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, so, providers=[provider])

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.has_kv_cache = any(_KV_PATTERN.search(n) for n in self.input_names)

        log.info(
            "Loaded %s — inputs=%s  outputs=%s  kv_cache=%s",
            model_path,
            self.input_names,
            self.output_names,
            self.has_kv_cache,
        )

    def _run_prefill(
        self, prompt_ids: Sequence[int]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run the prompt through the model and return (logits, kv_state)."""
        input_ids = np.array([prompt_ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)

        feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask

        if self.has_kv_cache:
            feeds.update(_build_kv_cache_inputs(self.session, batch_size=1, past_seq_len=0))

        outputs = self.session.run(None, feeds)
        logits = outputs[0]

        kv_state: dict[str, np.ndarray] = {}
        if self.has_kv_cache:
            kv_output_names = [
                n for n in self.output_names if _KV_PATTERN.search(n.replace("present", "past_key_values"))
            ]
            for idx, name in enumerate(kv_output_names, start=1):
                cache_key = _KV_PATTERN.sub(
                    lambda m: m.group(0),
                    name.replace("present", "past_key_values"),
                )
                kv_state[cache_key] = outputs[idx]

        return logits, kv_state

    def _run_decode_step(
        self,
        token_id: int,
        kv_state: dict[str, np.ndarray],
        seq_len: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run a single decode step with KV cache reuse."""
        input_ids = np.array([[token_id]], dtype=np.int64)
        attention_mask = np.ones((1, seq_len), dtype=np.int64)

        feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
        if "attention_mask" in self.input_names:
            feeds["attention_mask"] = attention_mask
        feeds.update(kv_state)

        outputs = self.session.run(None, feeds)
        logits = outputs[0]

        new_kv: dict[str, np.ndarray] = {}
        if self.has_kv_cache:
            kv_output_names = [
                n for n in self.output_names if _KV_PATTERN.search(n.replace("present", "past_key_values"))
            ]
            for idx, name in enumerate(kv_output_names, start=1):
                cache_key = _KV_PATTERN.sub(
                    lambda m: m.group(0),
                    name.replace("present", "past_key_values"),
                )
                new_kv[cache_key] = outputs[idx]

        return logits, new_kv

    def generate(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 128,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        callback: Callable[[int, bool], None] | None = None,
    ) -> List[int]:
        """Autoregressively generate tokens, optionally streaming via *callback*.

        Args:
            prompt_ids: Encoded prompt token IDs.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (defaults to instance default).
            top_k: Top-k filter width (defaults to instance default).
            top_p: Nucleus probability mass (defaults to instance default).
            callback: Called as ``callback(token_id, is_first)`` for every
                generated token.  Useful for streaming output.

        Returns:
            List of generated token IDs (*excluding* the prompt).
        """
        temp = temperature if temperature is not None else self.temperature
        tk = top_k if top_k is not None else self.top_k
        tp = top_p if top_p is not None else self.top_p

        generated: list[int] = []

        if self.has_kv_cache:
            logits, kv_state = self._run_prefill(prompt_ids)
            next_logits = logits[0, -1, :]
            seq_len = len(prompt_ids)

            for i in range(max_new_tokens):
                token = _sample_token(next_logits, temp, tk, tp)
                generated.append(token)
                if callback:
                    callback(token, i == 0)

                seq_len += 1
                if seq_len >= self.max_length:
                    break

                next_logits_arr, kv_state = self._run_decode_step(token, kv_state, seq_len)
                next_logits = next_logits_arr[0, -1, :]
        else:
            current_ids = list(prompt_ids)
            for i in range(max_new_tokens):
                input_ids = np.array([current_ids], dtype=np.int64)
                feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
                if "attention_mask" in self.input_names:
                    feeds["attention_mask"] = np.ones_like(input_ids, dtype=np.int64)

                outputs = self.session.run(None, feeds)
                next_logits = outputs[0][0, -1, :]
                token = _sample_token(next_logits, temp, tk, tp)
                generated.append(token)
                current_ids.append(token)
                if callback:
                    callback(token, i == 0)

                if len(current_ids) >= self.max_length:
                    break

        return generated

    def generate_text(
        self,
        prompt: str,
        tokenizer_name: str,
        max_new_tokens: int = 128,
        **kwargs,
    ) -> str:
        """Encode *prompt*, generate, and decode back to text.

        Requires the ``transformers`` library for tokenizer access.
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for generate_text().  "
                "Install it with:  pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = self.generate(prompt_ids, max_new_tokens=max_new_tokens, **kwargs)
        return tokenizer.decode(generated_ids, skip_special_tokens=True)

    def benchmark(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int = 64,
        num_runs: int = 3,
    ) -> StreamMetrics:
        """Run generation *num_runs* times and return aggregated latency stats."""
        all_ttft: list[float] = []
        all_itl: list[float] = []
        all_total: list[float] = []
        all_counts: list[int] = []

        for run_idx in range(num_runs):
            itl_times: list[float] = []
            ttft: float = 0.0
            last_ts = time.perf_counter()
            start = last_ts

            def _on_token(_tok: int, is_first: bool) -> None:
                nonlocal ttft, last_ts
                now = time.perf_counter()
                if is_first:
                    ttft = (now - start) * 1000.0
                else:
                    itl_times.append((now - last_ts) * 1000.0)
                last_ts = now

            tokens = self.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=0,
                callback=_on_token,
            )
            elapsed = (time.perf_counter() - start) * 1000.0

            all_ttft.append(ttft)
            all_itl.extend(itl_times)
            all_total.append(elapsed)
            all_counts.append(len(tokens))
            log.info("Benchmark run %d/%d: %d tokens in %.1f ms", run_idx + 1, num_runs, len(tokens), elapsed)

        def _percentile(data: list[float], pct: float) -> float:
            if not data:
                return 0.0
            arr = np.array(sorted(data))
            k = (len(arr) - 1) * (pct / 100.0)
            lo = int(k)
            hi = min(lo + 1, len(arr) - 1)
            return float(arr[lo] + (arr[hi] - arr[lo]) * (k - lo))

        total_tokens = sum(all_counts)
        total_time = sum(all_total)

        return StreamMetrics(
            ttft_ms=_percentile(all_ttft, 50),
            mean_itl_ms=float(np.mean(all_itl)) if all_itl else 0.0,
            p50_itl_ms=_percentile(all_itl, 50),
            p95_itl_ms=_percentile(all_itl, 95),
            p99_itl_ms=_percentile(all_itl, 99),
            tokens_per_sec=(total_tokens / (total_time / 1000.0)) if total_time > 0 else 0.0,
            total_tokens=total_tokens,
            total_time_ms=total_time,
        )


def stream_generate(
    model_path: str,
    prompt_ids_or_text: Sequence[int] | str,
    provider: str = "CPUExecutionProvider",
    tokenizer_name: str | None = None,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    print_tokens: bool = True,
    **kwargs,
) -> list[int] | str:
    """One-call convenience for CLI / script usage.

    If *prompt_ids_or_text* is a string, *tokenizer_name* must be provided so
    the prompt can be encoded and the output decoded.
    """
    gen = StreamingGenerator(
        model_path,
        provider=provider,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        **{k: v for k, v in kwargs.items() if k in ("max_length",)},
    )

    if isinstance(prompt_ids_or_text, str):
        if tokenizer_name is None:
            raise ValueError("tokenizer_name is required when prompt_ids_or_text is a string")

        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required when passing text prompts.  "
                "Install it with:  pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        prompt_ids: list[int] = tokenizer.encode(prompt_ids_or_text)
    else:
        tokenizer = None
        prompt_ids = list(prompt_ids_or_text)

    def _print_cb(token_id: int, is_first: bool) -> None:
        if not print_tokens or tokenizer is None:
            return
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        print(piece, end="", flush=True)

    generated = gen.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        callback=_print_cb if print_tokens else None,
    )

    if print_tokens and tokenizer is not None:
        print()

    if tokenizer is not None:
        return tokenizer.decode(generated, skip_special_tokens=True)
    return generated
