from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .kv_pool import KVCachePool


class RequestState(Enum):
    WAITING = auto()
    PREFILL = auto()
    DECODE = auto()
    PREEMPTED = auto()
    DONE = auto()


@dataclass
class Request:
    id: str
    prompt_ids: List[int]
    generated_ids: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    max_tokens: int = 256
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    callback: Optional[Callable] = None

    prefill_pos: int = 0
    chunk_size: int = 0
    _swap_buffer: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = field(
        default=None, repr=False
    )
    _recompute_ids: Optional[List[int]] = field(default=None, repr=False)

    @property
    def active_prefill_ids(self) -> List[int]:
        return self._recompute_ids if self._recompute_ids is not None else self.prompt_ids


@dataclass
class SchedulerConfig:
    max_batch_size: int = 32
    max_seq_len: int = 4096
    max_waiting: int = 256
    preemption_mode: str = "recompute"
    chunk_size: int = 512


@dataclass
class ScheduleBatch:
    prefill_requests: List[Request]
    decode_requests: List[Request]
    preempted_requests: List[Request]
    input_ids: List[int] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    seq_lens: List[int] = field(default_factory=list)
    block_tables: List[List[int]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_requests and not self.decode_requests

    @property
    def all_requests(self) -> List[Request]:
        return self.prefill_requests + self.decode_requests

    @property
    def total_tokens(self) -> int:
        return sum(self.seq_lens)


class ContinuousBatchingScheduler:

    def __init__(self, config: SchedulerConfig, kv_pool: KVCachePool) -> None:
        self.config = config
        self.kv_pool = kv_pool
        self._waiting: List[Request] = []
        self._running: List[Request] = []
        self._preempted: List[Request] = []
        self._requests: Dict[str, Request] = {}

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    # ------------------------------------------------------------------ public API

    def add_request(self, request: Request) -> None:
        if len(self._waiting) >= self.config.max_waiting:
            raise RuntimeError("Waiting queue full")
        total_len = len(request.prompt_ids) + request.max_tokens
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {total_len} exceeds limit {self.config.max_seq_len}"
            )
        self._requests[request.id] = request
        self._waiting.append(request)
        self._waiting.sort(key=lambda r: (-r.priority, r.created_at))

    def schedule(self) -> ScheduleBatch:
        preempted_this_step: List[Request] = []

        # Phase 1 — guarantee every running decode request can grow by one token.
        # Sort: high-priority first; within same priority, prefer keeping decode
        # over prefill (prefill is cheaper to redo). pop() evicts from the tail.
        self._running.sort(
            key=lambda r: (-r.priority, r.state == RequestState.PREFILL, r.created_at)
        )
        while True:
            deficit = 0
            for req in self._running:
                if req.state != RequestState.DECODE:
                    continue
                total = len(req.prompt_ids) + len(req.generated_ids) + 1
                needed = -(-total // self.kv_pool.block_size)
                deficit += max(0, needed - len(req.block_table))
            if deficit <= self.kv_pool.num_free_blocks or not self._running:
                break
            victim = self._running.pop()
            self._preempt(victim, self.config.preemption_mode)
            self._preempted.append(victim)
            preempted_this_step.append(victim)

        for req in self._running:
            if req.state != RequestState.DECODE:
                continue
            total = len(req.prompt_ids) + len(req.generated_ids) + 1
            needed = -(-total // self.kv_pool.block_size)
            while len(req.block_table) < needed and self.kv_pool.num_free_blocks > 0:
                req.block_table.extend(self.kv_pool.allocate(1))

        # Phase 2 — resume previously preempted requests (highest priority first).
        still_preempted: List[Request] = []
        for req in self._preempted:
            if len(self._running) >= self.config.max_batch_size:
                still_preempted.append(req)
                continue
            if self._can_admit(req):
                self._resume(req)
                self._running.append(req)
            else:
                still_preempted.append(req)
        self._preempted = still_preempted

        # Phase 3 — admit new requests from the waiting queue.
        still_waiting: List[Request] = []
        for req in self._waiting:
            if len(self._running) >= self.config.max_batch_size:
                still_waiting.append(req)
                continue
            if self._can_admit(req):
                req.block_table = self.kv_pool.allocate(len(req.prompt_ids))
                req.state = RequestState.PREFILL
                req.prefill_pos = 0
                self._running.append(req)
            else:
                still_waiting.append(req)
        self._waiting = still_waiting

        # Phase 4 — compute per-request chunk sizes and build the merged batch.
        prefill_reqs: List[Request] = []
        decode_reqs: List[Request] = []
        for req in self._running:
            if req.state == RequestState.PREFILL:
                remaining = len(req.active_prefill_ids) - req.prefill_pos
                req.chunk_size = min(self.config.chunk_size, remaining)
                prefill_reqs.append(req)
            elif req.state == RequestState.DECODE:
                decode_reqs.append(req)

        batch = self._merge_prefill_decode(prefill_reqs, decode_reqs)
        batch.preempted_requests = preempted_this_step
        return batch

    def step_complete(
        self, request_id: str, new_token: int, finished: bool
    ) -> None:
        req = self._requests[request_id]

        if req.state == RequestState.PREFILL:
            req.prefill_pos += req.chunk_size
            if req.prefill_pos >= len(req.active_prefill_ids):
                req.state = RequestState.DECODE
                req._recompute_ids = None

        if new_token >= 0 and req.state == RequestState.DECODE:
            req.generated_ids.append(new_token)
            total = len(req.prompt_ids) + len(req.generated_ids)
            self.kv_pool.mark_slot_used(req.block_table, total)

        if finished:
            req.state = RequestState.DONE
            self.kv_pool.free(req.block_table)
            req.block_table = []
            if req in self._running:
                self._running.remove(req)

    # ------------------------------------------------------------------ internals

    def _can_admit(self, request: Request) -> bool:
        if request.state == RequestState.PREEMPTED:
            total = len(request.prompt_ids) + len(request.generated_ids)
            needed = -(-total // self.kv_pool.block_size)
            return self.kv_pool.num_free_blocks >= needed
        needed = -(-len(request.prompt_ids) // self.kv_pool.block_size)
        return self.kv_pool.num_free_blocks >= needed

    def _preempt(self, request: Request, mode: str) -> None:
        if mode == "swap":
            request._swap_buffer = {}
            for layer in range(self.kv_pool.num_layers):
                k, v = self.kv_pool.get_kv(request.block_table, layer)
                request._swap_buffer[layer] = (k.copy(), v.copy())
        self.kv_pool.free(request.block_table)
        request.block_table = []
        request.state = RequestState.PREEMPTED

    def _resume(self, request: Request) -> None:
        if request._swap_buffer is not None:
            total = len(request.prompt_ids) + len(request.generated_ids)
            request.block_table = self.kv_pool.allocate(total)
            for layer, (k, v) in request._swap_buffer.items():
                self.kv_pool.set_kv(request.block_table, layer, k, v, 0)
            request._swap_buffer = None
            request.state = RequestState.DECODE
        else:
            all_ids = request.prompt_ids + request.generated_ids
            request._recompute_ids = all_ids
            request.block_table = self.kv_pool.allocate(len(all_ids))
            request.state = RequestState.PREFILL
            request.prefill_pos = 0

    def _merge_prefill_decode(
        self,
        prefill_requests: List[Request],
        decode_requests: List[Request],
    ) -> ScheduleBatch:
        input_ids: List[int] = []
        positions: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []

        for req in prefill_requests:
            ids = req.active_prefill_ids
            chunk = ids[req.prefill_pos : req.prefill_pos + req.chunk_size]
            input_ids.extend(chunk)
            positions.extend(range(req.prefill_pos, req.prefill_pos + req.chunk_size))
            seq_lens.append(len(chunk))
            block_tables.append(req.block_table)

        for req in decode_requests:
            token = req.generated_ids[-1] if req.generated_ids else req.prompt_ids[-1]
            input_ids.append(token)
            positions.append(len(req.prompt_ids) + len(req.generated_ids) - 1)
            seq_lens.append(1)
            block_tables.append(req.block_table)

        return ScheduleBatch(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            preempted_requests=[],
            input_ids=input_ids,
            positions=positions,
            seq_lens=seq_lens,
            block_tables=block_tables,
        )
