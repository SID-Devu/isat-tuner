"""Grammar-constrained generation engine.

Provides :class:`ConstrainedGenerator` for producing tokens that conform to
JSON schemas, regex patterns, or GBNF grammars by applying FSM-based token
masks before sampling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import numpy as np

from isat.constrained.fsm import GBNFFsm, JsonSchemaFSM, RegexFSM

log = logging.getLogger("isat.constrained")


# ---------------------------------------------------------------------------
# TokenVocabulary
# ---------------------------------------------------------------------------

class TokenVocabulary:
    """Wraps a HuggingFace tokenizer to expose a token-id → string mapping."""

    def __init__(self, tokenizer_name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for TokenVocabulary.  "
                "Install with:  pip install transformers"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        vocab = self._tokenizer.get_vocab()
        self._id_to_str: dict[int, str] = {}
        max_id = max(vocab.values()) if vocab else 0

        for tok_str, tok_id in vocab.items():
            try:
                decoded = self._tokenizer.convert_tokens_to_string([tok_str])
            except Exception:
                decoded = tok_str
            self._id_to_str[tok_id] = decoded

        self._size = max_id + 1
        self._vocab_strings: list[str] = [
            self._id_to_str.get(i, "") for i in range(self._size)
        ]

    def token_to_str(self, token_id: int) -> str:
        """Decode a single token id to its string representation."""
        return self._id_to_str.get(token_id, "")

    def get_vocab_strings(self) -> list[str]:
        """Return a list of decoded strings indexed by token id."""
        return list(self._vocab_strings)

    @property
    def size(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# ConstrainedResult
# ---------------------------------------------------------------------------

@dataclass
class ConstrainedResult:
    """Outcome of a single constrained-generation call."""

    text: str = ""
    parsed_value: Any = None
    tokens_generated: int = 0
    tokens_rejected: int = 0
    fsm_overhead_ms: float = 0.0
    total_time_ms: float = 0.0
    valid: bool = False


# ---------------------------------------------------------------------------
# ConstrainedGenerator
# ---------------------------------------------------------------------------

class ConstrainedGenerator:
    """Token-by-token generator that masks logits via an FSM before sampling.

    Supports JSON Schema, regex, GBNF grammar, and Pydantic model
    constraints.  At each decoding step the FSM provides a boolean mask
    over the vocabulary; invalid token logits are set to ``-inf`` so the
    sampler can never select them.
    """

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for ConstrainedGenerator.  "
                "Install with:  pip install onnxruntime"
            ) from exc

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(model_path, so, providers=[provider])
        self._input_names = [i.name for i in self._session.get_inputs()]
        self._output_names = [o.name for o in self._session.get_outputs()]
        log.info("ConstrainedGenerator loaded %s", model_path)

    # ---- Public generation methods ----

    def generate_json(
        self,
        prompt_ids: np.ndarray,
        schema: dict,
        max_tokens: int = 512,
        temperature: float = 1.0,
        vocabulary: list[str] | None = None,
    ) -> ConstrainedResult:
        """Generate tokens constrained to valid JSON matching *schema*."""
        if vocabulary is None:
            raise ValueError("vocabulary is required")
        fsm = JsonSchemaFSM(schema, vocabulary)
        return self._loop(
            prompt_ids, fsm, fsm.initial_state, vocabulary,
            max_tokens, temperature, parse_fn=json.loads,
        )

    def generate_regex(
        self,
        prompt_ids: np.ndarray,
        pattern: str,
        max_tokens: int = 256,
        temperature: float = 1.0,
        vocabulary: list[str] | None = None,
    ) -> ConstrainedResult:
        """Generate tokens matching the *pattern* regex."""
        if vocabulary is None:
            raise ValueError("vocabulary is required")
        fsm = RegexFSM(pattern, vocabulary)
        return self._loop(
            prompt_ids, fsm, fsm.start_state, vocabulary,
            max_tokens, temperature,
        )

    def generate_grammar(
        self,
        prompt_ids: np.ndarray,
        grammar_str: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        vocabulary: list[str] | None = None,
    ) -> ConstrainedResult:
        """Generate tokens matching a GBNF grammar."""
        if vocabulary is None:
            raise ValueError("vocabulary is required")
        fsm = GBNFFsm(grammar_str, vocabulary)
        return self._loop(
            prompt_ids, fsm, fsm.initial_state, vocabulary,
            max_tokens, temperature,
        )

    def generate_pydantic(
        self,
        prompt_ids: np.ndarray,
        model_class: Type,
        max_tokens: int = 512,
        temperature: float = 1.0,
        vocabulary: list[str] | None = None,
    ) -> ConstrainedResult:
        """Generate JSON conforming to *model_class* and return a validated instance."""
        schema = model_class.model_json_schema()
        result = self.generate_json(
            prompt_ids, schema, max_tokens, temperature, vocabulary,
        )
        if result.valid and result.parsed_value is not None:
            try:
                result.parsed_value = model_class.model_validate(result.parsed_value)
            except Exception as exc:
                log.warning("Pydantic validation failed: %s", exc)
                result.valid = False
        return result

    # ---- Internal generation loop ----

    def _loop(
        self,
        prompt_ids: np.ndarray,
        fsm: Any,
        state: Any,
        vocabulary: list[str],
        max_tokens: int,
        temperature: float,
        parse_fn: Any = None,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> ConstrainedResult:
        t0 = time.perf_counter()
        fsm_ms = 0.0
        gen_ids: list[int] = []
        rejected = 0
        vsize = len(vocabulary)
        ids = prompt_ids.copy()

        for step in range(max_tokens):
            feeds = {self._input_names[0]: ids.reshape(1, -1).astype(np.int64)}
            outputs = self._session.run(self._output_names, feeds)
            logits = outputs[0][0, -1, :].astype(np.float64)

            t_fsm = time.perf_counter()
            mask = fsm.get_valid_tokens(state)
            fsm_ms += (time.perf_counter() - t_fsm) * 1000

            logits = self._apply_mask(logits, mask)

            if not np.any(np.isfinite(logits)):
                log.warning("All tokens masked at step %d – forcing EOS", step)
                break

            tid = self._sample(logits, temperature, top_k, top_p)
            rejected += int(vsize - np.sum(mask))

            tok_str = vocabulary[tid] if tid < vsize else ""
            new_state = fsm.advance(state, tok_str)
            if new_state is None:
                log.warning("FSM rejected sampled token '%s' at step %d", tok_str, step)
                break
            state = new_state
            gen_ids.append(tid)
            ids = np.append(ids, tid)

            if fsm.is_complete(state):
                break

        text = "".join(vocabulary[t] for t in gen_ids if t < vsize)

        parsed = None
        valid = False
        if parse_fn is not None:
            try:
                parsed = parse_fn(text)
                valid = True
            except Exception as exc:
                log.debug("Post-parse failed: %s", exc)
        else:
            valid = fsm.is_complete(state)
            parsed = text

        return ConstrainedResult(
            text=text,
            parsed_value=parsed,
            tokens_generated=len(gen_ids),
            tokens_rejected=rejected,
            fsm_overhead_ms=fsm_ms,
            total_time_ms=(time.perf_counter() - t0) * 1000,
            valid=valid,
        )

    # ---- Masking & sampling ----

    @staticmethod
    def _apply_mask(logits: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Set logits for invalid tokens to ``-inf``."""
        out = logits.copy()
        mask_len = min(len(out), len(valid_mask))
        out[:mask_len][~valid_mask[:mask_len]] = -np.inf
        if len(out) > mask_len:
            out[mask_len:] = -np.inf
        return out

    @staticmethod
    def _sample(
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> int:
        """Top-k / nucleus sampling over masked logits."""
        logits = logits.astype(np.float64)
        if temperature <= 0:
            return int(np.argmax(logits))
        logits /= temperature

        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            thr = np.partition(logits, -top_k)[-top_k]
            logits = np.where(logits < thr, -np.inf, logits)

        finite = np.isfinite(logits)
        if not np.any(finite):
            return int(np.argmax(logits))
        mx = np.max(logits[finite])
        exp = np.exp(logits - mx)
        exp[~finite] = 0.0
        total = exp.sum()
        if total == 0:
            return int(np.argmax(logits))
        probs = exp / total

        if top_p < 1.0:
            si = np.argsort(-probs)
            sp = probs[si]
            cum = np.cumsum(sp)
            sp[cum - sp > top_p] = 0.0
            probs = np.zeros_like(probs)
            probs[si] = sp
            total = probs.sum()
            if total > 0:
                probs /= total

        return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def constrained_generate(
    model_path: str,
    prompt: str,
    schema: dict | None = None,
    regex: str | None = None,
    grammar: str | None = None,
    tokenizer_name: str | None = None,
    **kwargs: Any,
) -> ConstrainedResult:
    """One-call constrained generation for CLI and scripts.

    Exactly one of *schema*, *regex*, or *grammar* must be provided.
    """
    if tokenizer_name is None:
        raise ValueError("tokenizer_name is required")

    vocab = TokenVocabulary(tokenizer_name)
    vocab_strings = vocab.get_vocab_strings()

    provider = kwargs.pop("provider", "CPUExecutionProvider")
    gen = ConstrainedGenerator(model_path, provider=provider)

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        prompt_ids = np.array(tokenizer.encode(prompt), dtype=np.int64)
    except ImportError as exc:
        raise ImportError(
            "transformers is required for prompt tokenization.  "
            "Install with:  pip install transformers"
        ) from exc

    gen_kw: dict[str, Any] = {"vocabulary": vocab_strings}
    for k in ("max_tokens", "temperature"):
        if k in kwargs:
            gen_kw[k] = kwargs[k]

    if schema is not None:
        return gen.generate_json(prompt_ids, schema, **gen_kw)
    if regex is not None:
        return gen.generate_regex(prompt_ids, regex, **gen_kw)
    if grammar is not None:
        return gen.generate_grammar(prompt_ids, grammar, **gen_kw)
    raise ValueError("Exactly one of schema, regex, or grammar must be provided")
