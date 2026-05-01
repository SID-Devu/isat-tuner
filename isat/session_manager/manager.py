"""Multi-turn session manager with KV cache persistence, incremental prefill, and compaction."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("isat.session_manager")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    session_id: str
    kv_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    token_history: List[int] = field(default_factory=list)
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    total_tokens: int = 0
    memory_mb: float = 0.0


@dataclass
class SessionInfo:
    session_id: str
    turn_count: int
    total_tokens: int
    memory_mb: float
    age_seconds: float


@dataclass
class SessionResult:
    text: str
    tokens_generated: int
    session_id: str
    turn_number: int
    prefill_time_ms: float
    generation_time_ms: float
    total_tokens_in_session: int
    memory_mb: float


# ---------------------------------------------------------------------------
# SessionStore — LRU-managed session storage with optional disk offload
# ---------------------------------------------------------------------------

class SessionStore:
    """Thread-local store for active sessions with LRU eviction and disk offload."""

    def __init__(
        self,
        max_sessions: int = 100,
        max_memory_mb: float = 4096,
        offload_dir: Optional[str] = None,
    ) -> None:
        self.max_sessions = max_sessions
        self.max_memory_mb = max_memory_mb
        self.offload_dir = offload_dir
        self._sessions: Dict[str, SessionState] = {}
        self._offloaded: set[str] = set()

        if offload_dir:
            Path(offload_dir).mkdir(parents=True, exist_ok=True)

    def create(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = uuid.uuid4().hex[:16]

        if session_id in self._sessions:
            raise ValueError(f"Session '{session_id}' already exists")

        if len(self._sessions) >= self.max_sessions:
            self._evict_lru()

        now = time.time()
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            created_at=now,
            last_access=now,
        )
        log.info("Created session %s", session_id)
        return session_id

    def get(self, session_id: str) -> SessionState:
        if session_id in self._offloaded:
            self.reload(session_id)

        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found")

        session = self._sessions[session_id]
        session.last_access = time.time()
        return session

    def update(
        self,
        session_id: str,
        new_kv: Dict[str, np.ndarray],
        new_tokens: List[int],
    ) -> None:
        session = self.get(session_id)

        for key, arr in new_kv.items():
            if key in session.kv_cache:
                session.kv_cache[key] = np.concatenate(
                    [session.kv_cache[key], arr], axis=-2,
                )
            else:
                session.kv_cache[key] = arr

        session.token_history.extend(new_tokens)
        session.total_tokens = len(session.token_history)
        session.turn_count += 1
        session.memory_mb = self._compute_memory(session)
        session.last_access = time.time()

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._offloaded.discard(session_id)

        if self.offload_dir:
            npz = Path(self.offload_dir) / f"{session_id}.npz"
            if npz.exists():
                npz.unlink()

        log.info("Deleted session %s", session_id)

    def _evict_lru(self) -> None:
        if not self._sessions:
            return

        lru_id = min(self._sessions, key=lambda k: self._sessions[k].last_access)

        if self.offload_dir:
            log.info("Evicting session %s to disk", lru_id)
            self.offload(lru_id)
        else:
            log.info("Evicting session %s (deleted)", lru_id)
            del self._sessions[lru_id]

    def offload(self, session_id: str) -> None:
        if not self.offload_dir:
            raise RuntimeError("No offload_dir configured")

        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")

        save_dict: Dict[str, Any] = {}
        for k, v in session.kv_cache.items():
            save_dict[k] = v

        npz_path = Path(self.offload_dir) / f"{session_id}.npz"
        np.savez(str(npz_path), **save_dict)

        meta_path = Path(self.offload_dir) / f"{session_id}.json"
        meta = {
            "session_id": session.session_id,
            "token_history": session.token_history,
            "turn_count": session.turn_count,
            "created_at": session.created_at,
            "last_access": session.last_access,
            "total_tokens": session.total_tokens,
            "memory_mb": session.memory_mb,
            "kv_keys": list(session.kv_cache.keys()),
        }
        meta_path.write_text(json.dumps(meta))

        session.kv_cache.clear()
        session.memory_mb = 0.0
        self._offloaded.add(session_id)
        log.info("Offloaded session %s to %s", session_id, npz_path)

    def reload(self, session_id: str) -> None:
        if session_id not in self._offloaded:
            return

        if not self.offload_dir:
            raise RuntimeError("No offload_dir configured")

        npz_path = Path(self.offload_dir) / f"{session_id}.npz"
        meta_path = Path(self.offload_dir) / f"{session_id}.json"

        if not npz_path.exists():
            self._offloaded.discard(session_id)
            raise FileNotFoundError(f"Offloaded data not found: {npz_path}")

        data = np.load(str(npz_path))
        meta = json.loads(meta_path.read_text())

        session = self._sessions.get(session_id)
        if session is None:
            session = SessionState(
                session_id=meta["session_id"],
                token_history=meta["token_history"],
                turn_count=meta["turn_count"],
                created_at=meta["created_at"],
                last_access=meta["last_access"],
                total_tokens=meta["total_tokens"],
            )
            self._sessions[session_id] = session

        for key in meta.get("kv_keys", []):
            if key in data:
                session.kv_cache[key] = data[key]

        session.memory_mb = self._compute_memory(session)
        self._offloaded.discard(session_id)
        log.info("Reloaded session %s from disk", session_id)

    def list_sessions(self) -> List[SessionInfo]:
        now = time.time()
        result = []
        for s in self._sessions.values():
            result.append(SessionInfo(
                session_id=s.session_id,
                turn_count=s.turn_count,
                total_tokens=s.total_tokens,
                memory_mb=s.memory_mb,
                age_seconds=round(now - s.created_at, 2),
            ))
        return result

    @staticmethod
    def _compute_memory(session: SessionState) -> float:
        total = sum(a.nbytes for a in session.kv_cache.values())
        return round(total / (1024 * 1024), 4)


# ---------------------------------------------------------------------------
# IncrementalPrefill — process only new tokens using cached KV state
# ---------------------------------------------------------------------------

class IncrementalPrefill:
    """Run incremental prefill: only process newly added tokens against the
    existing KV cache, avoiding redundant computation for prior turns."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_path = model_path
        self.provider = provider
        self._session = None

    @property
    def ort_session(self):
        if self._session is None:
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                self.model_path,
                providers=[self.provider],
            )
        return self._session

    def prefill_turn(
        self,
        session: SessionState,
        new_input_ids: List[int],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Process *new_input_ids* incrementally against the session's existing
        KV cache.  Returns ``(logits, updated_kv_cache)``."""
        feed = self._build_incremental_feed(session, new_input_ids)

        ort_sess = self.ort_session
        output_names = [o.name for o in ort_sess.get_outputs()]
        outputs = ort_sess.run(output_names, feed)

        logits = outputs[0]
        new_kv: Dict[str, np.ndarray] = {}
        for name, val in zip(output_names[1:], outputs[1:]):
            present_name = name.replace("present", "past_key_values")
            new_kv[present_name] = val

        return logits, new_kv

    def _build_incremental_feed(
        self,
        session: SessionState,
        new_ids: List[int],
    ) -> Dict[str, np.ndarray]:
        """Construct an ONNX Runtime feed dict that reuses the session's past KV
        cache and only sends the new input tokens."""
        past_len = len(session.token_history)
        new_len = len(new_ids)

        input_ids = np.array([new_ids], dtype=np.int64)

        position_ids = np.arange(
            past_len, past_len + new_len, dtype=np.int64,
        ).reshape(1, -1)

        total_len = past_len + new_len
        attention_mask = np.ones((1, total_len), dtype=np.int64)

        feed: Dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        ort_sess = self.ort_session
        input_names = {inp.name for inp in ort_sess.get_inputs()}

        if "position_ids" in input_names:
            feed["position_ids"] = position_ids

        for inp in ort_sess.get_inputs():
            if inp.name.startswith("past_key_values"):
                if inp.name in session.kv_cache:
                    feed[inp.name] = session.kv_cache[inp.name]
                else:
                    shape = []
                    for dim in inp.shape:
                        shape.append(dim if isinstance(dim, int) else 0)
                    shape[-2] = past_len
                    if past_len == 0:
                        feed[inp.name] = np.zeros(shape, dtype=np.float16)
                    else:
                        feed[inp.name] = np.zeros(shape, dtype=np.float16)

        return feed


# ---------------------------------------------------------------------------
# SessionCompactor — reduce context length when it exceeds the budget
# ---------------------------------------------------------------------------

class SessionCompactor:
    """Reduce session context when it grows beyond *max_context* tokens."""

    def __init__(self, max_context: int = 4096) -> None:
        self.max_context = max_context

    def compact(
        self,
        session: SessionState,
        strategy: str = "sliding",
    ) -> SessionState:
        """Compact *session* in-place using the chosen strategy.

        Strategies
        ----------
        sliding     : keep first 4 attention-sink tokens + last *max_context* tokens.
        summary     : keep system prompt (first turn) + most recent N turns.
        importance  : score each turn by recency + length, keep highest-scoring.
        """
        if session.total_tokens <= self.max_context:
            return session

        if strategy == "sliding":
            self._compact_sliding(session)
        elif strategy == "summary":
            self._compact_summary(session)
        elif strategy == "importance":
            self._compact_importance(session)
        else:
            raise ValueError(f"Unknown compaction strategy '{strategy}'")

        session.total_tokens = len(session.token_history)
        session.memory_mb = sum(
            a.nbytes for a in session.kv_cache.values()
        ) / (1024 * 1024)
        log.info(
            "Compacted session %s (%s): %d tokens remain",
            session.session_id, strategy, session.total_tokens,
        )
        return session

    def _compact_sliding(self, session: SessionState) -> None:
        sink_size = 4
        keep_tail = self.max_context - sink_size

        if keep_tail <= 0:
            keep_tail = self.max_context
            sink_size = 0

        tokens = session.token_history
        if len(tokens) <= self.max_context:
            return

        sink_tokens = tokens[:sink_size]
        tail_tokens = tokens[-keep_tail:]
        session.token_history = sink_tokens + tail_tokens

        for key, arr in session.kv_cache.items():
            seq_axis = -2
            seq_len = arr.shape[seq_axis]
            if seq_len <= self.max_context:
                continue
            sink = arr[..., :sink_size, :]
            tail = arr[..., -keep_tail:, :]
            session.kv_cache[key] = np.concatenate([sink, tail], axis=seq_axis)

    def _compact_summary(self, session: SessionState) -> None:
        if session.turn_count <= 1:
            return

        tokens_per_turn = len(session.token_history) // max(session.turn_count, 1)
        if tokens_per_turn == 0:
            tokens_per_turn = 1

        system_tokens = tokens_per_turn
        remaining_budget = self.max_context - system_tokens
        turns_to_keep = max(1, remaining_budget // tokens_per_turn)

        keep_start = session.token_history[:system_tokens]
        keep_end = session.token_history[-(turns_to_keep * tokens_per_turn):]
        session.token_history = keep_start + keep_end

        keep_count = len(session.token_history)
        for key, arr in session.kv_cache.items():
            seq_len = arr.shape[-2]
            if seq_len <= keep_count:
                continue
            front = arr[..., :system_tokens, :]
            back = arr[..., -(turns_to_keep * tokens_per_turn):, :]
            session.kv_cache[key] = np.concatenate([front, back], axis=-2)

    def _compact_importance(self, session: SessionState) -> None:
        if session.turn_count <= 1:
            return

        tokens_per_turn = len(session.token_history) // max(session.turn_count, 1)
        if tokens_per_turn == 0:
            tokens_per_turn = 1

        num_turns = max(session.turn_count, 1)
        scores: List[Tuple[float, int]] = []
        for i in range(num_turns):
            start = i * tokens_per_turn
            end = min(start + tokens_per_turn, len(session.token_history))
            turn_tokens = session.token_history[start:end]
            score = self._estimate_turn_importance(
                turn_tokens, position=i, total_turns=num_turns,
            )
            scores.append((score, i))

        scores.sort(reverse=True)
        budget_turns = max(1, self.max_context // tokens_per_turn)
        keep_indices = sorted(idx for _, idx in scores[:budget_turns])

        new_tokens: List[int] = []
        for idx in keep_indices:
            start = idx * tokens_per_turn
            end = min(start + tokens_per_turn, len(session.token_history))
            new_tokens.extend(session.token_history[start:end])
        session.token_history = new_tokens

        keep_count = len(new_tokens)
        for key, arr in session.kv_cache.items():
            if arr.shape[-2] <= keep_count:
                continue
            slices = []
            for idx in keep_indices:
                s = idx * tokens_per_turn
                e = min(s + tokens_per_turn, arr.shape[-2])
                slices.append(arr[..., s:e, :])
            if slices:
                session.kv_cache[key] = np.concatenate(slices, axis=-2)

    @staticmethod
    def _estimate_turn_importance(
        turn_tokens: List[int],
        position: int,
        total_turns: int,
    ) -> float:
        """Recency-weighted importance: recent + longer turns score higher."""
        recency = (position + 1) / total_turns
        length_score = min(len(turn_tokens) / 256, 1.0)
        return 0.7 * recency + 0.3 * length_score


# ---------------------------------------------------------------------------
# SessionManager — orchestrator for multi-turn conversations
# ---------------------------------------------------------------------------

class SessionManager:
    """High-level multi-turn conversation manager tying together the store,
    incremental prefill engine, and context compactor."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        max_sessions: int = 100,
        max_memory_mb: float = 4096,
        offload_dir: Optional[str] = None,
        max_context: int = 4096,
    ) -> None:
        self.store = SessionStore(
            max_sessions=max_sessions,
            max_memory_mb=max_memory_mb,
            offload_dir=offload_dir,
        )
        self.prefill = IncrementalPrefill(model_path, provider)
        self.compactor = SessionCompactor(max_context)

    def start_session(
        self,
        system_prompt_ids: Optional[List[int]] = None,
    ) -> str:
        sid = self.store.create()
        if system_prompt_ids:
            session = self.store.get(sid)
            t0 = time.perf_counter()
            _logits, new_kv = self.prefill.prefill_turn(session, system_prompt_ids)
            elapsed = (time.perf_counter() - t0) * 1000
            self.store.update(sid, new_kv, system_prompt_ids)
            log.info("System prompt prefilled in %.1f ms", elapsed)
        return sid

    def chat(
        self,
        session_id: str,
        user_input_ids: List[int],
        max_tokens: int = 256,
        **sampling: Any,
    ) -> SessionResult:
        session = self.store.get(session_id)

        if session.total_tokens + len(user_input_ids) > self.compactor.max_context:
            self.compactor.compact(session)

        t_prefill = time.perf_counter()
        logits, new_kv = self.prefill.prefill_turn(session, user_input_ids)
        prefill_ms = (time.perf_counter() - t_prefill) * 1000

        self.store.update(session_id, new_kv, user_input_ids)

        t_gen = time.perf_counter()
        generated_ids = self._generate(
            session_id, logits, max_tokens=max_tokens, **sampling,
        )
        gen_ms = (time.perf_counter() - t_gen) * 1000

        session = self.store.get(session_id)

        return SessionResult(
            text="".join(chr(min(t, 0x10FFFF)) for t in generated_ids),
            tokens_generated=len(generated_ids),
            session_id=session_id,
            turn_number=session.turn_count,
            prefill_time_ms=round(prefill_ms, 2),
            generation_time_ms=round(gen_ms, 2),
            total_tokens_in_session=session.total_tokens,
            memory_mb=session.memory_mb,
        )

    def _generate(
        self,
        session_id: str,
        logits: np.ndarray,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        **_extra: Any,
    ) -> List[int]:
        """Autoregressive token generation using cached KV state."""
        generated: List[int] = []
        current_logits = logits

        for _ in range(max_tokens):
            next_token_logits = current_logits[0, -1, :]

            if temperature > 0:
                scaled = next_token_logits / max(temperature, 1e-8)
            else:
                scaled = next_token_logits

            if top_k > 0:
                top_indices = np.argsort(scaled)[-top_k:]
                mask = np.full_like(scaled, -1e9)
                mask[top_indices] = scaled[top_indices]
                scaled = mask

            probs = _softmax(scaled)
            token_id = int(np.random.choice(len(probs), p=probs))
            generated.append(token_id)

            session = self.store.get(session_id)
            current_logits, new_kv = self.prefill.prefill_turn(
                session, [token_id],
            )
            self.store.update(session_id, new_kv, [token_id])

            if token_id == 2:  # common EOS
                break

        return generated

    def end_session(self, session_id: str) -> None:
        self.store.delete(session_id)

    def get_session_info(self, session_id: str) -> SessionInfo:
        session = self.store.get(session_id)
        return SessionInfo(
            session_id=session.session_id,
            turn_count=session.turn_count,
            total_tokens=session.total_tokens,
            memory_mb=session.memory_mb,
            age_seconds=round(time.time() - session.created_at, 2),
        )

    def list_sessions(self) -> List[SessionInfo]:
        return self.store.list_sessions()

    def save_all(self, path: str) -> None:
        """Persist every active session to *path* as a directory of npz + json files."""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        manifest: List[Dict[str, Any]] = []
        for sid, session in self.store._sessions.items():
            kv_file = save_dir / f"{sid}.npz"
            np.savez(str(kv_file), **session.kv_cache)

            manifest.append({
                "session_id": session.session_id,
                "token_history": session.token_history,
                "turn_count": session.turn_count,
                "created_at": session.created_at,
                "last_access": session.last_access,
                "total_tokens": session.total_tokens,
                "memory_mb": session.memory_mb,
                "kv_keys": list(session.kv_cache.keys()),
            })

        (save_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("Saved %d sessions to %s", len(manifest), path)

    def load_all(self, path: str) -> None:
        """Restore sessions previously saved with :meth:`save_all`."""
        load_dir = Path(path)
        manifest_path = load_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest found at {manifest_path}")

        manifest = json.loads(manifest_path.read_text())

        for entry in manifest:
            sid = entry["session_id"]
            session = SessionState(
                session_id=sid,
                token_history=entry["token_history"],
                turn_count=entry["turn_count"],
                created_at=entry["created_at"],
                last_access=entry["last_access"],
                total_tokens=entry["total_tokens"],
                memory_mb=entry["memory_mb"],
            )

            kv_file = load_dir / f"{sid}.npz"
            if kv_file.exists():
                data = np.load(str(kv_file))
                for key in entry.get("kv_keys", []):
                    if key in data:
                        session.kv_cache[key] = data[key]

            session.memory_mb = SessionStore._compute_memory(session)
            self.store._sessions[sid] = session

        log.info("Loaded %d sessions from %s", len(manifest), path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ---------------------------------------------------------------------------
# CLI / scripting entry point
# ---------------------------------------------------------------------------

def session_manage(
    action: str = "list",
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> dict:
    """CLI entry point for session management operations.

    Parameters
    ----------
    action : one of ``list``, ``create``, ``info``, ``delete``, ``compact``,
             ``offload``.
    session_id : required for ``info``, ``delete``, ``compact``, ``offload``.
    **kwargs : forwarded to the underlying operation.

    Returns a summary dict.
    """
    model_path = kwargs.pop("model_path", None)
    provider = kwargs.pop("provider", "CPUExecutionProvider")
    max_sessions = kwargs.pop("max_sessions", 100)
    max_memory_mb = kwargs.pop("max_memory_mb", 4096)
    offload_dir = kwargs.pop("offload_dir", None)
    max_context = kwargs.pop("max_context", 4096)

    store = SessionStore(
        max_sessions=max_sessions,
        max_memory_mb=max_memory_mb,
        offload_dir=offload_dir,
    )

    if action == "list":
        sessions = store.list_sessions()
        return {
            "action": "list",
            "count": len(sessions),
            "sessions": [
                {
                    "session_id": s.session_id,
                    "turns": s.turn_count,
                    "tokens": s.total_tokens,
                    "memory_mb": s.memory_mb,
                    "age_s": s.age_seconds,
                }
                for s in sessions
            ],
        }

    if action == "create":
        sid = store.create(session_id)
        return {"action": "create", "session_id": sid}

    if action == "info":
        if not session_id:
            return {"error": "session_id required for 'info'"}
        session = store.get(session_id)
        return {
            "action": "info",
            "session_id": session.session_id,
            "turns": session.turn_count,
            "tokens": session.total_tokens,
            "memory_mb": session.memory_mb,
            "age_s": round(time.time() - session.created_at, 2),
        }

    if action == "delete":
        if not session_id:
            return {"error": "session_id required for 'delete'"}
        store.delete(session_id)
        return {"action": "delete", "session_id": session_id}

    if action == "compact":
        if not session_id:
            return {"error": "session_id required for 'compact'"}
        session = store.get(session_id)
        strategy = kwargs.pop("strategy", "sliding")
        compactor = SessionCompactor(max_context)
        compactor.compact(session, strategy=strategy)
        return {
            "action": "compact",
            "session_id": session_id,
            "strategy": strategy,
            "tokens_after": session.total_tokens,
        }

    if action == "offload":
        if not session_id:
            return {"error": "session_id required for 'offload'"}
        if not offload_dir:
            return {"error": "offload_dir required for 'offload'"}
        store.offload(session_id)
        return {"action": "offload", "session_id": session_id}

    return {"error": f"Unknown action '{action}'"}
