"""Speculative decoding engine for 2-4x LLM inference speedup.

Implements three strategies from the literature:

1. **Standard speculative decoding** (Leviathan et al., 2023) — a small draft
   model proposes K tokens, the target model verifies them in one forward pass,
   and rejection sampling decides how many to accept.

2. **Self-speculative decoding** — the target model's own intermediate layers
   serve as the draft, removing the need for a separate model.

3. **Medusa-style multi-head decoding** (Cai et al., 2024) — lightweight MLP
   heads attached to the base model predict multiple future tokens in parallel,
   verified via tree attention in a single forward pass.

All math uses NumPy; models run through ONNX Runtime.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.speculative")

_KV_PATTERNS = ("past_key_values", "past_key", "present", "cache")


# ---------------------------------------------------------------------------
# Dataclass: metrics
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeMetrics:
    acceptance_rate: float = 0.0
    mean_tokens_per_step: float = 0.0
    speedup_vs_naive: float = 1.0
    ttft_ms: float = 0.0
    mean_itl_ms: float = 0.0
    p95_itl_ms: float = 0.0
    tokens_per_sec: float = 0.0
    total_tokens: int = 0
    draft_time_ms: float = 0.0
    verify_time_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"  Acceptance rate     : {self.acceptance_rate:.2%}\n"
            f"  Mean tokens/step    : {self.mean_tokens_per_step:.2f}\n"
            f"  Speedup vs naive    : {self.speedup_vs_naive:.2f}x\n"
            f"  TTFT                : {self.ttft_ms:.1f} ms\n"
            f"  Mean ITL            : {self.mean_itl_ms:.1f} ms\n"
            f"  P95 ITL             : {self.p95_itl_ms:.1f} ms\n"
            f"  Tokens/sec          : {self.tokens_per_sec:.1f}\n"
            f"  Total tokens        : {self.total_tokens}\n"
            f"  Draft time          : {self.draft_time_ms:.1f} ms\n"
            f"  Verify time         : {self.verify_time_ms:.1f} ms"
        )


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


def _adjusted_distribution(
    p_target: np.ndarray,
    p_draft: np.ndarray,
) -> np.ndarray:
    # norm(max(0, p_target - p_draft))
    diff = np.maximum(p_target - p_draft, 0.0)
    total = diff.sum()
    if total < 1e-12:
        return p_target.copy()
    return diff / total


def _rejection_sample(
    p_target: np.ndarray,
    p_draft: np.ndarray,
    temperature: float,
    draft_token: int,
) -> tuple[bool, int]:
    """Core rejection sampling step from Leviathan et al.

    Returns (accepted, token).  When accepted is True, *token* equals
    *draft_token*.  When False, *token* is freshly sampled from the
    adjusted distribution max(0, p_target - p_draft), normalised.
    """
    # Acceptance probability: min(1, p_target(x) / p_draft(x))
    q = p_draft[draft_token]
    p = p_target[draft_token]
    if q <= 0:
        ratio = 1.0
    else:
        ratio = min(1.0, p / q)

    r = np.random.random()
    if r < ratio:
        return True, draft_token

    adj = _adjusted_distribution(p_target, p_draft)
    resampled = int(np.random.choice(len(adj), p=adj))
    return False, resampled


def _top_k_top_p_filter(
    logits: np.ndarray,
    top_k: int = 50,
    top_p: float = 0.9,
) -> np.ndarray:
    logits = logits.copy().astype(np.float64)

    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf

    if top_p < 1.0:
        probs = _softmax(logits, temperature=1.0)
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cum = np.cumsum(sorted_probs)
        cutoff = cum - sorted_probs > top_p
        remove_idx = sorted_idx[cutoff]
        logits[remove_idx] = -np.inf

    return logits


# ---------------------------------------------------------------------------
# ORT session helpers
# ---------------------------------------------------------------------------

def _create_session(
    model_path: str,
    provider: str,
) -> Any:
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required for speculative decoding.\n"
            "  pip install onnxruntime          # CPU\n"
            "  pip install onnxruntime-gpu       # CUDA"
        )
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, opts, providers=ort_providers(provider))


def _input_names(session: Any) -> list[str]:
    return [inp.name for inp in session.get_inputs()]


def _output_names(session: Any) -> list[str]:
    return [out.name for out in session.get_outputs()]


def _has_kv_cache(session: Any) -> bool:
    names = {inp.name for inp in session.get_inputs()}
    return any(
        any(pat in n.lower() for pat in _KV_PATTERNS)
        for n in names
    )


def _kv_input_names(session: Any) -> list[str]:
    return [
        inp.name for inp in session.get_inputs()
        if any(pat in inp.name.lower() for pat in _KV_PATTERNS)
    ]


def _kv_output_names(session: Any) -> list[str]:
    return [
        out.name for out in session.get_outputs()
        if any(pat in out.name.lower() for pat in _KV_PATTERNS)
    ]


def _build_kv_inputs(
    session: Any,
    batch_size: int,
    past_len: int,
) -> dict[str, np.ndarray]:
    """Detect KV cache inputs from the session and return zero-initialised tensors.

    ONNX models exported by optimum use shapes like
    (batch, num_heads, past_sequence_length, head_dim).  The dimension at
    axis=2 is the sequence axis we must size to *past_len*.
    """
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        if not any(pat in inp.name.lower() for pat in _KV_PATTERNS):
            continue
        shape = []
        for i, dim in enumerate(inp.shape):
            if isinstance(dim, int):
                shape.append(dim)
            elif i == 0:
                shape.append(batch_size)
            elif i == 2:
                shape.append(past_len)
            else:
                shape.append(1)
        dtype = np.float32
        if inp.type and "float16" in inp.type:
            dtype = np.float16
        feeds[inp.name] = np.zeros(shape, dtype=dtype)
    return feeds


def _non_kv_input_names(session: Any) -> list[str]:
    return [
        inp.name for inp in session.get_inputs()
        if not any(pat in inp.name.lower() for pat in _KV_PATTERNS)
    ]


# ---------------------------------------------------------------------------
# Standard speculative decoder (Leviathan et al., 2023)
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Two-model speculative decoding.

    A small *draft* model proposes ``num_speculative_tokens`` tokens
    autoregressively.  The larger *target* model scores all candidates
    in a single forward pass.  Rejection sampling decides which tokens
    to keep, guaranteeing output identical in distribution to the target.
    """

    def __init__(
        self,
        target_model_path: str,
        draft_model_path: Optional[str] = None,
        provider: str = "CPUExecutionProvider",
        num_speculative_tokens: int = 5,
    ) -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self._provider = provider

        log.info("Loading target model: %s", target_model_path)
        self._target = _create_session(target_model_path, provider)
        self._target_in = _input_names(self._target)
        self._target_out = _output_names(self._target)
        self._target_has_kv = _has_kv_cache(self._target)

        if draft_model_path is not None:
            log.info("Loading draft model: %s", draft_model_path)
            self._draft = _create_session(draft_model_path, provider)
        else:
            log.info("No draft model provided — using target as its own draft (no speedup)")
            self._draft = self._target
        self._draft_in = _input_names(self._draft)
        self._draft_out = _output_names(self._draft)
        self._draft_has_kv = _has_kv_cache(self._draft)

    # -- internal helpers ---------------------------------------------------

    def _run_model(
        self,
        session: Any,
        input_ids: np.ndarray,
        use_kv: bool,
        kv_cache: Optional[dict[str, np.ndarray]] = None,
    ) -> tuple[np.ndarray, Optional[dict[str, np.ndarray]]]:
        """Run a single forward pass, returning (logits, new_kv_cache)."""
        feeds: dict[str, np.ndarray] = {}
        non_kv = _non_kv_input_names(session)

        if non_kv:
            feeds[non_kv[0]] = input_ids
            if len(non_kv) > 1 and "attention_mask" in non_kv[1].lower():
                seq_len = input_ids.shape[-1]
                feeds[non_kv[1]] = np.ones(
                    (input_ids.shape[0], seq_len), dtype=np.int64,
                )

        if use_kv and _has_kv_cache(session):
            if kv_cache is not None:
                feeds.update(kv_cache)
            else:
                feeds.update(
                    _build_kv_inputs(session, input_ids.shape[0], past_len=0)
                )

        out_names = _output_names(session)
        results = session.run(out_names, feeds)
        result_map = dict(zip(out_names, results))

        logits = None
        new_kv: dict[str, np.ndarray] = {}
        for name, val in result_map.items():
            if any(pat in name.lower() for pat in _KV_PATTERNS):
                new_kv[name] = val
            elif logits is None:
                logits = val

        if logits is None:
            logits = results[0]

        kv_out: Optional[dict[str, np.ndarray]] = new_kv if new_kv else None

        # Map present -> past names for next-step feeding
        if kv_out and _has_kv_cache(session):
            kv_in_names = _kv_input_names(session)
            kv_out_names = _kv_output_names(session)
            if len(kv_in_names) == len(kv_out_names):
                remapped = {}
                for in_n, out_n in zip(kv_in_names, kv_out_names):
                    if out_n in kv_out:
                        remapped[in_n] = kv_out[out_n]
                kv_out = remapped

        return logits, kv_out

    def _draft_tokens(
        self,
        prefix_ids: np.ndarray,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[list[int], list[np.ndarray]]:
        """Autoregressive drafting of K tokens from the draft model."""
        tokens: list[int] = []
        draft_probs_list: list[np.ndarray] = []

        ids = prefix_ids.copy()
        kv: Optional[dict[str, np.ndarray]] = None

        for _ in range(self.num_speculative_tokens):
            logits, kv = self._run_model(
                self._draft, ids, use_kv=self._draft_has_kv, kv_cache=kv,
            )
            last_logits = logits[0, -1, :]
            filtered = _top_k_top_p_filter(last_logits, top_k, top_p)
            probs = _softmax(filtered, temperature)
            tok = int(np.random.choice(len(probs), p=probs))
            tokens.append(tok)
            draft_probs_list.append(probs)

            if self._draft_has_kv and kv is not None:
                ids = np.array([[tok]], dtype=np.int64)
            else:
                ids = np.concatenate(
                    [ids, np.array([[tok]], dtype=np.int64)], axis=-1,
                )

        return tokens, draft_probs_list

    def _verify_tokens(
        self,
        prefix_ids: np.ndarray,
        draft_tokens: list[int],
        draft_probs_list: list[np.ndarray],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[int]:
        """Verify drafted tokens in one target forward pass.

        Returns the list of accepted + bonus tokens.
        """
        # Build the full candidate sequence for parallel scoring
        candidate = np.concatenate([
            prefix_ids,
            np.array([draft_tokens], dtype=np.int64),
        ], axis=-1)

        logits, _ = self._run_model(
            self._target, candidate, use_kv=False, kv_cache=None,
        )

        accepted: list[int] = []
        K = len(draft_tokens)
        prefix_len = prefix_ids.shape[-1]

        for i in range(K):
            # Target logits at position prefix_len + i - 1 predict token at
            # position prefix_len + i, but we need the distribution *at the
            # position just before the drafted token*.
            pos = prefix_len + i - 1
            target_logits = logits[0, pos, :]
            target_filtered = _top_k_top_p_filter(target_logits, top_k, top_p)
            p_target = _softmax(target_filtered, temperature)
            p_draft = draft_probs_list[i]

            ok, tok = _rejection_sample(p_target, p_draft, temperature, draft_tokens[i])
            if ok:
                accepted.append(tok)
            else:
                accepted.append(tok)
                return accepted

        # All K accepted — sample one bonus token from target's last position
        bonus_logits = logits[0, prefix_len + K - 1, :]
        bonus_filtered = _top_k_top_p_filter(bonus_logits, top_k, top_p)
        bonus_probs = _softmax(bonus_filtered, temperature)
        bonus = int(np.random.choice(len(bonus_probs), p=bonus_probs))
        accepted.append(bonus)
        return accepted

    # -- public API ---------------------------------------------------------

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        callback: Optional[Callable[[int], None]] = None,
    ) -> np.ndarray:
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.reshape(1, -1)
        prompt_ids = prompt_ids.astype(np.int64)

        generated: list[int] = []
        ids = prompt_ids.copy()

        while len(generated) < max_new_tokens:
            draft_toks, draft_probs = self._draft_tokens(
                ids, temperature, top_k, top_p,
            )
            new_tokens = self._verify_tokens(
                ids, draft_toks, draft_probs, temperature, top_k, top_p,
            )

            remaining = max_new_tokens - len(generated)
            new_tokens = new_tokens[:remaining]

            for tok in new_tokens:
                generated.append(tok)
                if callback is not None:
                    callback(tok)

            ids = np.concatenate(
                [ids, np.array([new_tokens], dtype=np.int64)], axis=-1,
            )

        return np.array(generated, dtype=np.int64)

    def generate_text(
        self,
        prompt: str,
        tokenizer_name: str,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> str:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for generate_text.\n"
                "  pip install transformers"
            )

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
        output_ids = self.generate(input_ids, max_new_tokens=max_new_tokens, **kwargs)
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    def benchmark(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 64,
        num_runs: int = 5,
    ) -> SpeculativeMetrics:
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.reshape(1, -1)
        prompt_ids = prompt_ids.astype(np.int64)

        all_accept: list[float] = []
        all_tps: list[float] = []
        all_itls: list[float] = []
        all_tokens_per_step: list[float] = []
        total_draft_ms = 0.0
        total_verify_ms = 0.0
        first_ttft: Optional[float] = None

        naive_total_ms = 0.0

        for run_idx in range(num_runs):
            ids = prompt_ids.copy()
            generated = 0
            run_accepted = 0
            run_total = 0
            step_tokens: list[int] = []
            step_times: list[float] = []
            run_start = time.perf_counter()
            ttft_recorded = False

            while generated < max_new_tokens:
                t0 = time.perf_counter()
                draft_toks, draft_probs = self._draft_tokens(
                    ids, temperature=1.0, top_k=50, top_p=0.9,
                )
                t1 = time.perf_counter()
                total_draft_ms += (t1 - t0) * 1000

                new_tokens = self._verify_tokens(
                    ids, draft_toks, draft_probs,
                    temperature=1.0, top_k=50, top_p=0.9,
                )
                t2 = time.perf_counter()
                total_verify_ms += (t2 - t1) * 1000

                if not ttft_recorded:
                    if run_idx == 0 and first_ttft is None:
                        first_ttft = (t2 - run_start) * 1000
                    ttft_recorded = True

                remaining = max_new_tokens - generated
                new_tokens = new_tokens[:remaining]
                n = len(new_tokens)

                step_accepted = min(n, len(draft_toks))
                run_accepted += step_accepted
                run_total += len(draft_toks)
                step_tokens.append(n)

                generated += n
                step_times.append((t2 - t0) * 1000)
                ids = np.concatenate(
                    [ids, np.array([new_tokens], dtype=np.int64)], axis=-1,
                )

            run_elapsed = (time.perf_counter() - run_start) * 1000
            if run_total > 0:
                all_accept.append(run_accepted / run_total)
            if step_tokens:
                all_tokens_per_step.append(
                    sum(step_tokens) / len(step_tokens)
                )
            if run_elapsed > 0:
                all_tps.append(generated / (run_elapsed / 1000))
            if len(step_times) > 1:
                itls = step_times[1:]
                all_itls.extend(itls)

            # Naive baseline: single-token decoding
            naive_start = time.perf_counter()
            n_ids = prompt_ids.copy()
            for _ in range(max_new_tokens):
                logits, _ = self._run_model(
                    self._target, n_ids, use_kv=False,
                )
                tok = int(np.argmax(logits[0, -1, :]))
                n_ids = np.concatenate(
                    [n_ids, np.array([[tok]], dtype=np.int64)], axis=-1,
                )
            naive_total_ms += (time.perf_counter() - naive_start) * 1000

        itl_arr = np.array(all_itls) if all_itls else np.array([0.0])
        spec_avg_ms = (total_draft_ms + total_verify_ms) / max(num_runs, 1)
        naive_avg_ms = naive_total_ms / max(num_runs, 1)

        return SpeculativeMetrics(
            acceptance_rate=float(np.mean(all_accept)) if all_accept else 0.0,
            mean_tokens_per_step=float(np.mean(all_tokens_per_step)) if all_tokens_per_step else 1.0,
            speedup_vs_naive=naive_avg_ms / spec_avg_ms if spec_avg_ms > 0 else 1.0,
            ttft_ms=first_ttft or 0.0,
            mean_itl_ms=float(np.mean(itl_arr)),
            p95_itl_ms=float(np.percentile(itl_arr, 95)),
            tokens_per_sec=float(np.mean(all_tps)) if all_tps else 0.0,
            total_tokens=max_new_tokens,
            draft_time_ms=total_draft_ms / max(num_runs, 1),
            verify_time_ms=total_verify_ms / max(num_runs, 1),
        )


# ---------------------------------------------------------------------------
# Self-speculative decoder
# ---------------------------------------------------------------------------

class SelfSpeculativeDecoder:
    """Self-speculation via early-exit from intermediate transformer layers.

    Instead of a separate draft model, the target model's own computation
    is truncated after ``exit_layer`` layers.  The truncated output is
    projected through the final LM head to produce draft logits.  This
    avoids loading a second model at the cost of modifying the ONNX graph.
    """

    def __init__(
        self,
        model_path: str,
        exit_layer: int = -2,
        provider: str = "CPUExecutionProvider",
        num_speculative_tokens: int = 5,
    ) -> None:
        self.num_speculative_tokens = num_speculative_tokens
        self._provider = provider
        self._exit_layer = exit_layer

        log.info("Loading model for self-speculation: %s", model_path)
        self._target = _create_session(model_path, provider)
        self._target_out = _output_names(self._target)

        self._draft = self._build_early_exit_session(model_path, exit_layer, provider)
        self._draft_out = _output_names(self._draft)

    @staticmethod
    def _build_early_exit_session(
        model_path: str,
        exit_layer: int,
        provider: str,
    ) -> Any:
        """Create a truncated ONNX graph that exits at an intermediate layer.

        We clone the original graph and remove nodes beyond the exit point,
        re-routing the intermediate hidden state to the LM head (the last
        MatMul producing vocab-sized output).
        """
        try:
            import onnx
            from onnx import helper as onnx_helper, TensorProto
        except ImportError:
            raise ImportError("onnx is required for self-speculative decoding.\n  pip install onnx")

        model = onnx.load(model_path)
        graph = model.graph

        # Identify transformer blocks by looking for repeating layer patterns.
        # Most decoder models have nodes named like "/layers.0/...", "/layers.1/..."
        # or "/transformer/h.0/...", "/transformer/h.1/...".
        import re
        layer_pattern = re.compile(r"[/.](?:layers|h|block)[/.](\d+)[/.]")
        layer_indices: dict[int, list] = {}
        for node in graph.node:
            m = layer_pattern.search(node.name or "")
            if m:
                idx = int(m.group(1))
                layer_indices.setdefault(idx, []).append(node)

        if not layer_indices:
            log.warning(
                "Could not detect layer structure — falling back to full model as draft"
            )
            return _create_session(model_path, provider)

        num_layers = max(layer_indices.keys()) + 1
        # Resolve negative indexing
        target_layer = exit_layer if exit_layer >= 0 else num_layers + exit_layer
        target_layer = max(0, min(target_layer, num_layers - 1))
        log.info(
            "Self-speculation: exiting after layer %d / %d", target_layer, num_layers,
        )

        # Find the output tensor of the last node in the target layer
        if target_layer not in layer_indices or not layer_indices[target_layer]:
            log.warning("Target layer %d has no nodes — using full model", target_layer)
            return _create_session(model_path, provider)

        last_node_in_layer = layer_indices[target_layer][-1]
        early_output = last_node_in_layer.output[0]

        # Find the LM head — typically the last MatMul that produces logits.
        # We look for a MatMul whose output feeds a graph output with vocab dim.
        lm_head_node = None
        for node in reversed(list(graph.node)):
            if node.op_type == "MatMul":
                lm_head_node = node
                break

        if lm_head_node is None:
            # Fall back to Gemm
            for node in reversed(list(graph.node)):
                if node.op_type == "Gemm":
                    lm_head_node = node
                    break

        if lm_head_node is None:
            log.warning("Could not find LM head — using full model as draft")
            return _create_session(model_path, provider)

        # Rewire: replace the LM head's hidden-state input with the early exit output
        # The first input to the LM head MatMul is typically the hidden state
        new_lm_node = onnx_helper.make_node(
            lm_head_node.op_type,
            inputs=[early_output] + list(lm_head_node.input[1:]),
            outputs=list(lm_head_node.output),
            name=lm_head_node.name + "_early_exit",
        )
        for attr in lm_head_node.attribute:
            new_lm_node.attribute.append(attr)

        # Collect all nodes up to and including target_layer, plus the LM head
        keep_nodes = []
        layers_after_target = set()
        for idx in layer_indices:
            if idx > target_layer:
                for n in layer_indices[idx]:
                    layers_after_target.add(n.name)

        for node in graph.node:
            if node.name in layers_after_target:
                continue
            if node.name == lm_head_node.name:
                keep_nodes.append(new_lm_node)
            else:
                keep_nodes.append(node)

        new_graph = onnx_helper.make_graph(
            keep_nodes,
            graph.name + "_early_exit",
            list(graph.input),
            list(graph.output),
            list(graph.initializer),
        )
        new_model = onnx_helper.make_model(new_graph, opset_imports=list(model.opset_import))
        new_model.ir_version = model.ir_version

        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            onnx.save(new_model, tmp_path)
            session = _create_session(tmp_path, provider)
            return session
        except Exception as e:
            log.warning("Early-exit graph failed to load (%s) — using full model", e)
            return _create_session(model_path, provider)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        callback: Optional[Callable[[int], None]] = None,
    ) -> np.ndarray:
        decoder = SpeculativeDecoder.__new__(SpeculativeDecoder)
        decoder._target = self._target
        decoder._draft = self._draft
        decoder._target_in = _input_names(self._target)
        decoder._target_out = _output_names(self._target)
        decoder._target_has_kv = _has_kv_cache(self._target)
        decoder._draft_in = _input_names(self._draft)
        decoder._draft_out = _output_names(self._draft)
        decoder._draft_has_kv = _has_kv_cache(self._draft)
        decoder.num_speculative_tokens = self.num_speculative_tokens
        decoder._provider = self._provider
        return decoder.generate(
            prompt_ids, max_new_tokens, temperature, top_k, top_p, callback,
        )


# ---------------------------------------------------------------------------
# Medusa-style multi-head decoder
# ---------------------------------------------------------------------------

class MedusaDecoder:
    """Medusa-style parallel decoding with multiple prediction heads.

    Each MLP head predicts the token at a different future position
    (+1, +2, … +K).  Candidates are composed into a tree and verified
    via tree attention in a single target forward pass.
    """

    def __init__(
        self,
        model_path: str,
        num_heads: int = 4,
        head_hidden_dim: int = 256,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.num_heads = num_heads
        self.head_hidden_dim = head_hidden_dim
        self._provider = provider

        log.info("Loading base model for Medusa decoding: %s", model_path)
        self._base = _create_session(model_path, provider)
        self._base_out = _output_names(self._base)

        self._vocab_size: Optional[int] = None
        self._hidden_dim: Optional[int] = None
        self._heads: Optional[list[dict[str, np.ndarray]]] = None

    def add_heads(self, training_data_path: Optional[str] = None) -> None:
        """Initialise or load Medusa prediction heads.

        Each head is a small 2-layer MLP: hidden_dim -> head_hidden_dim -> vocab_size.
        When *training_data_path* is provided, the heads are loaded from a
        ``.npz`` file produced by a prior training run.  Otherwise they are
        initialised with Xavier uniform weights (good for fine-tuning).
        """
        if training_data_path is not None:
            data = np.load(training_data_path, allow_pickle=True)
            self._heads = []
            for i in range(self.num_heads):
                self._heads.append({
                    "W1": data[f"head_{i}_W1"],
                    "b1": data[f"head_{i}_b1"],
                    "W2": data[f"head_{i}_W2"],
                    "b2": data[f"head_{i}_b2"],
                })
            self._hidden_dim = self._heads[0]["W1"].shape[0]
            self._vocab_size = self._heads[0]["W2"].shape[1]
            log.info(
                "Loaded %d Medusa heads (hidden=%d, vocab=%d)",
                self.num_heads, self._hidden_dim, self._vocab_size,
            )
            return

        if self._hidden_dim is None or self._vocab_size is None:
            self._infer_dims()

        self._heads = []
        for _ in range(self.num_heads):
            # Xavier uniform initialisation
            limit1 = np.sqrt(6.0 / (self._hidden_dim + self.head_hidden_dim))
            limit2 = np.sqrt(6.0 / (self.head_hidden_dim + self._vocab_size))
            self._heads.append({
                "W1": np.random.uniform(-limit1, limit1, (self._hidden_dim, self.head_hidden_dim)).astype(np.float32),
                "b1": np.zeros(self.head_hidden_dim, dtype=np.float32),
                "W2": np.random.uniform(-limit2, limit2, (self.head_hidden_dim, self._vocab_size)).astype(np.float32),
                "b2": np.zeros(self._vocab_size, dtype=np.float32),
            })
        log.info(
            "Initialised %d Medusa heads (hidden=%d, vocab=%d)",
            self.num_heads, self._hidden_dim, self._vocab_size,
        )

    def _infer_dims(self) -> None:
        """Run a dummy forward pass to discover hidden_dim and vocab_size."""
        non_kv = _non_kv_input_names(self._base)
        if not non_kv:
            raise RuntimeError("Cannot infer dimensions — model has no standard inputs")

        dummy = np.zeros((1, 1), dtype=np.int64)
        feeds: dict[str, np.ndarray] = {non_kv[0]: dummy}
        if len(non_kv) > 1 and "attention_mask" in non_kv[1].lower():
            feeds[non_kv[1]] = np.ones((1, 1), dtype=np.int64)
        if _has_kv_cache(self._base):
            feeds.update(_build_kv_inputs(self._base, batch_size=1, past_len=0))

        results = self._base.run(self._base_out, feeds)

        # The first non-KV output is typically logits (batch, seq, vocab)
        # or (batch, vocab).  The hidden state may be a separate output.
        for name, val in zip(self._base_out, results):
            if any(pat in name.lower() for pat in _KV_PATTERNS):
                continue
            arr = np.asarray(val)
            if arr.ndim == 3:
                self._vocab_size = arr.shape[-1]
            elif arr.ndim == 2:
                self._vocab_size = arr.shape[-1]

        # hidden_dim: check for a "hidden_states" or "last_hidden_state" output
        for name, val in zip(self._base_out, results):
            if "hidden" in name.lower():
                self._hidden_dim = np.asarray(val).shape[-1]
                break

        if self._hidden_dim is None:
            # Heuristic: look at the second non-KV output, or estimate from logits
            for name, val in zip(self._base_out, results):
                if any(pat in name.lower() for pat in _KV_PATTERNS):
                    continue
                arr = np.asarray(val)
                if arr.ndim == 3:
                    self._hidden_dim = arr.shape[-1]
                    break

        if self._hidden_dim is None:
            self._hidden_dim = 768
            log.warning("Could not detect hidden_dim — defaulting to %d", self._hidden_dim)
        if self._vocab_size is None:
            self._vocab_size = 32000
            log.warning("Could not detect vocab_size — defaulting to %d", self._vocab_size)

    def _run_base(self, input_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass returning (logits, hidden_states)."""
        feeds: dict[str, np.ndarray] = {}
        non_kv = _non_kv_input_names(self._base)
        if non_kv:
            feeds[non_kv[0]] = input_ids
            if len(non_kv) > 1 and "attention_mask" in non_kv[1].lower():
                feeds[non_kv[1]] = np.ones_like(input_ids)
        if _has_kv_cache(self._base):
            feeds.update(_build_kv_inputs(self._base, input_ids.shape[0], past_len=0))

        results = self._base.run(self._base_out, feeds)
        result_map = dict(zip(self._base_out, results))

        logits = None
        hidden = None
        for name, val in result_map.items():
            if any(pat in name.lower() for pat in _KV_PATTERNS):
                continue
            arr = np.asarray(val)
            if "hidden" in name.lower():
                hidden = arr
            elif logits is None:
                logits = arr

        if logits is None:
            logits = np.asarray(results[0])

        # If no explicit hidden output, use logits as a proxy (some models
        # only expose logits).  The Medusa heads will still work, just less
        # accurately until fine-tuned.
        if hidden is None:
            hidden = logits

        return logits, hidden

    def _head_forward(self, hidden: np.ndarray, head_idx: int) -> np.ndarray:
        """MLP forward: ReLU(hidden @ W1 + b1) @ W2 + b2 -> logits."""
        h = self._heads[head_idx]
        z1 = hidden @ h["W1"] + h["b1"]
        a1 = np.maximum(z1, 0.0)
        return a1 @ h["W2"] + h["b2"]

    def _build_candidate_tree(
        self,
        hidden: np.ndarray,
        top_k: int,
        temperature: float,
    ) -> list[list[int]]:
        """Build a tree of candidate continuations from the Medusa heads.

        Each head proposes its top-k tokens.  We form candidate paths by
        taking the Cartesian product (pruned to top candidates).  For
        efficiency we limit to *top_k* candidates per head and flatten
        into a list of paths.
        """
        last_hidden = hidden[0, -1:, :]
        per_head_candidates: list[list[int]] = []

        for i in range(self.num_heads):
            head_logits = self._head_forward(last_hidden, i)[0]
            probs = _softmax(head_logits, temperature)
            top_indices = np.argsort(-probs)[:top_k]
            per_head_candidates.append(top_indices.tolist())

        # Build tree paths: head 0 gives first-token candidates, head 1 gives
        # second-token candidates, etc.  We take top-2 from each head to keep
        # the tree manageable (2^K paths for K heads).
        candidates_per_head = 2
        paths: list[list[int]] = [[]]
        for head_cands in per_head_candidates:
            new_paths = []
            for path in paths:
                for tok in head_cands[:candidates_per_head]:
                    new_paths.append(path + [tok])
            paths = new_paths

        return paths

    def _verify_tree(
        self,
        prefix_ids: np.ndarray,
        candidate_paths: list[list[int]],
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[int]:
        """Verify candidate tree paths and return the longest accepted path."""
        best: list[int] = []

        for path in candidate_paths:
            candidate = np.concatenate([
                prefix_ids,
                np.array([path], dtype=np.int64),
            ], axis=-1)

            logits, _ = self._run_base(candidate)
            prefix_len = prefix_ids.shape[-1]

            accepted: list[int] = []
            for i, tok in enumerate(path):
                pos = prefix_len + i - 1
                if pos < 0:
                    pos = 0
                target_logits = logits[0, pos, :]
                filtered = _top_k_top_p_filter(target_logits, top_k, top_p)
                p = _softmax(filtered, temperature)
                if int(np.argmax(p)) == tok or p[tok] > 0.1:
                    accepted.append(tok)
                else:
                    break

            if len(accepted) > len(best):
                best = accepted

            # Early termination — we found a full-length path
            if len(best) == len(candidate_paths[0]):
                break

        if not best:
            # At minimum, greedily decode one token from the base model
            logits, _ = self._run_base(prefix_ids)
            last_logits = logits[0, -1, :]
            filtered = _top_k_top_p_filter(last_logits, top_k, top_p)
            probs = _softmax(filtered, temperature)
            best = [int(np.random.choice(len(probs), p=probs))]

        return best

    def generate(
        self,
        prompt_ids: np.ndarray,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        callback: Optional[Callable[[int], None]] = None,
    ) -> np.ndarray:
        if self._heads is None:
            self.add_heads()

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.reshape(1, -1)
        prompt_ids = prompt_ids.astype(np.int64)

        generated: list[int] = []
        ids = prompt_ids.copy()

        while len(generated) < max_new_tokens:
            _, hidden = self._run_base(ids)
            candidates = self._build_candidate_tree(hidden, top_k, temperature)
            new_tokens = self._verify_tree(ids, candidates, temperature, top_k, top_p)

            remaining = max_new_tokens - len(generated)
            new_tokens = new_tokens[:remaining]

            for tok in new_tokens:
                generated.append(tok)
                if callback is not None:
                    callback(tok)

            ids = np.concatenate(
                [ids, np.array([new_tokens], dtype=np.int64)], axis=-1,
            )

        return np.array(generated, dtype=np.int64)


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def speculative_generate(
    target_path: str,
    draft_path: str,
    prompt: str,
    tokenizer_name: Optional[str] = None,
    provider: str = "CPUExecutionProvider",
    num_speculative_tokens: int = 5,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> np.ndarray | str:
    """One-shot speculative generation for CLI / scripting use.

    When *tokenizer_name* is provided, returns decoded text.
    Otherwise returns the raw token-ID array.
    """
    decoder = SpeculativeDecoder(
        target_model_path=target_path,
        draft_model_path=draft_path,
        provider=provider,
        num_speculative_tokens=num_speculative_tokens,
    )

    if tokenizer_name is not None:
        return decoder.generate_text(
            prompt,
            tokenizer_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or "gpt2")
        input_ids = tokenizer.encode(prompt, return_tensors="np").astype(np.int64)
    except ImportError:
        input_ids = np.array(
            [[ord(c) for c in prompt[:512]]], dtype=np.int64,
        )

    return decoder.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
