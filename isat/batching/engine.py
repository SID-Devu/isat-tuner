"""Dynamic batching engine for real-time inference serving.

Collects individual inference requests and groups them into batches to
maximize GPU utilization. Dispatches batched inference at configurable
intervals or when the batch is full.

Like NVIDIA Triton's dynamic batcher or TF Serving's batching scheduler.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.batching")


@dataclass
class InferenceRequest:
    request_id: str
    inputs: dict[str, np.ndarray]
    future: Future
    enqueue_time: float = 0.0
    priority: int = 0  # lower = higher priority


@dataclass
class BatchStats:
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0
    avg_queue_wait_ms: float = 0
    avg_inference_ms: float = 0
    max_queue_wait_ms: float = 0
    batches_by_size: dict[int, int] = field(default_factory=dict)
    timeout_triggers: int = 0
    fullbatch_triggers: int = 0

    def summary(self) -> str:
        lines = [
            f"  Total requests     : {self.total_requests}",
            f"  Total batches      : {self.total_batches}",
            f"  Avg batch size     : {self.avg_batch_size:.1f}",
            f"  Avg queue wait     : {self.avg_queue_wait_ms:.2f} ms",
            f"  Max queue wait     : {self.max_queue_wait_ms:.2f} ms",
            f"  Avg inference      : {self.avg_inference_ms:.2f} ms",
            f"  Timeout triggers   : {self.timeout_triggers}",
            f"  Full-batch triggers: {self.fullbatch_triggers}",
        ]
        if self.batches_by_size:
            lines.append(f"\n  Batch size distribution:")
            for size in sorted(self.batches_by_size):
                lines.append(f"    size {size}: {self.batches_by_size[size]} batches")
        return "\n".join(lines)


class DynamicBatcher:
    """Collect requests and dispatch batched inference."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
        max_queue_size: int = 1000,
    ):
        self.model_path = model_path
        self.provider = provider
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_queue_size = max_queue_size

        self._queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_queue_size)
        self._session = None
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._stats = BatchStats()
        self._lock = threading.Lock()
        self._queue_waits: list[float] = []
        self._inference_times: list[float] = []

    def start(self):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            self.model_path,
            providers=ort_providers(self.provider),
        )
        self._running = True
        self._worker = threading.Thread(target=self._batch_loop, daemon=True)
        self._worker.start()
        log.info("Dynamic batcher started (max_batch=%d, timeout=%gms)",
                 self.max_batch_size, self.max_wait_ms)

    def stop(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=5)
        self._flush_stats()

    def submit(self, inputs: dict[str, np.ndarray], priority: int = 0) -> Future:
        future: Future = Future()
        req = InferenceRequest(
            request_id=uuid.uuid4().hex[:12],
            inputs=inputs,
            future=future,
            enqueue_time=time.perf_counter(),
            priority=priority,
        )
        self._queue.put((priority, req.enqueue_time, req))
        return future

    def get_stats(self) -> BatchStats:
        self._flush_stats()
        return self._stats

    def _batch_loop(self):
        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue
            self._dispatch_batch(batch)

    def _collect_batch(self) -> list[InferenceRequest]:
        batch: list[InferenceRequest] = []
        deadline = time.perf_counter() + self.max_wait_ms / 1000

        try:
            _, _, first = self._queue.get(timeout=self.max_wait_ms / 1000)
            batch.append(first)
        except queue.Empty:
            return []

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                with self._lock:
                    self._stats.timeout_triggers += 1
                break
            try:
                _, _, req = self._queue.get(timeout=max(remaining, 0.001))
                batch.append(req)
            except queue.Empty:
                with self._lock:
                    self._stats.timeout_triggers += 1
                break

        if len(batch) >= self.max_batch_size:
            with self._lock:
                self._stats.fullbatch_triggers += 1

        return batch

    def _dispatch_batch(self, batch: list[InferenceRequest]):
        now = time.perf_counter()

        waits = [(now - r.enqueue_time) * 1000 for r in batch]
        with self._lock:
            self._queue_waits.extend(waits)

        try:
            first_inputs = batch[0].inputs
            batched_feed = {}
            for key in first_inputs:
                arrays = [r.inputs[key] for r in batch]
                batched_feed[key] = np.concatenate(arrays, axis=0)

            t0 = time.perf_counter()
            outputs = self._session.run(None, batched_feed)
            inference_ms = (time.perf_counter() - t0) * 1000

            with self._lock:
                self._inference_times.append(inference_ms)

            for i, req in enumerate(batch):
                result = [out[i:i+1] for out in outputs]
                req.future.set_result(result)

        except Exception as e:
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

        with self._lock:
            self._stats.total_requests += len(batch)
            self._stats.total_batches += 1
            bs = len(batch)
            self._stats.batches_by_size[bs] = self._stats.batches_by_size.get(bs, 0) + 1

    def _flush_stats(self):
        with self._lock:
            if self._queue_waits:
                self._stats.avg_queue_wait_ms = float(np.mean(self._queue_waits))
                self._stats.max_queue_wait_ms = float(np.max(self._queue_waits))
            if self._inference_times:
                self._stats.avg_inference_ms = float(np.mean(self._inference_times))
            if self._stats.total_batches > 0:
                self._stats.avg_batch_size = self._stats.total_requests / self._stats.total_batches
