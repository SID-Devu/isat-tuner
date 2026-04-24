"""Inference result caching / memoization.

Caches inference outputs keyed by input tensor hash. For workloads
with repeated inputs (batch processing, A/B tests, CI pipelines),
this avoids redundant GPU computation.

Supports:
  - In-memory LRU cache
  - Disk-based persistent cache
  - TTL expiration
  - Hit/miss statistics
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("isat.inference_cache")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    cache_size: int = 0
    max_size: int = 0
    avg_hit_save_ms: float = 0
    disk_entries: int = 0
    disk_size_mb: float = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / max(self.total_requests, 1)

    def summary(self) -> str:
        lines = [
            f"  Cache requests : {self.total_requests}",
            f"  Hits           : {self.hits} ({self.hit_rate:.1%})",
            f"  Misses         : {self.misses}",
            f"  Evictions      : {self.evictions}",
            f"  In-memory      : {self.cache_size} / {self.max_size} entries",
            f"  Avg time saved : {self.avg_hit_save_ms:.2f} ms per hit",
        ]
        if self.disk_entries:
            lines.append(f"  Disk entries   : {self.disk_entries}")
            lines.append(f"  Disk size      : {self.disk_size_mb:.1f} MB")
        return "\n".join(lines)


class InferenceCache:
    """Cache inference results with LRU eviction and optional disk persistence."""

    def __init__(
        self,
        max_memory_entries: int = 1000,
        ttl_seconds: float = 3600,
        disk_cache_dir: str = "",
    ):
        self.max_memory_entries = max_memory_entries
        self.ttl_seconds = ttl_seconds
        self.disk_cache_dir = disk_cache_dir
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._stats = CacheStats(max_size=max_memory_entries)
        self._time_savings: list[float] = []

        if disk_cache_dir:
            Path(disk_cache_dir).mkdir(parents=True, exist_ok=True)

    def get(self, inputs: dict[str, np.ndarray]) -> Optional[list[np.ndarray]]:
        key = self._hash_inputs(inputs)
        self._stats.total_requests += 1

        if key in self._cache:
            ts, value = self._cache[key]
            if time.time() - ts < self.ttl_seconds:
                self._cache.move_to_end(key)
                self._stats.hits += 1
                return value
            else:
                del self._cache[key]

        if self.disk_cache_dir:
            disk_path = Path(self.disk_cache_dir) / f"{key}.pkl"
            if disk_path.exists():
                try:
                    mtime = disk_path.stat().st_mtime
                    if time.time() - mtime < self.ttl_seconds:
                        with open(disk_path, "rb") as f:
                            value = pickle.load(f)
                        self._cache[key] = (time.time(), value)
                        self._stats.hits += 1
                        return value
                except Exception:
                    pass

        self._stats.misses += 1
        return None

    def put(self, inputs: dict[str, np.ndarray], outputs: list[np.ndarray],
            inference_ms: float = 0):
        key = self._hash_inputs(inputs)
        self._cache[key] = (time.time(), outputs)
        self._cache.move_to_end(key)

        if inference_ms > 0:
            self._time_savings.append(inference_ms)

        while len(self._cache) > self.max_memory_entries:
            self._cache.popitem(last=False)
            self._stats.evictions += 1

        if self.disk_cache_dir:
            try:
                disk_path = Path(self.disk_cache_dir) / f"{key}.pkl"
                with open(disk_path, "wb") as f:
                    pickle.dump(outputs, f)
            except Exception:
                pass

        self._stats.cache_size = len(self._cache)

    def invalidate(self, inputs: dict[str, np.ndarray] | None = None):
        if inputs is None:
            self._cache.clear()
            if self.disk_cache_dir:
                for f in Path(self.disk_cache_dir).glob("*.pkl"):
                    f.unlink()
        else:
            key = self._hash_inputs(inputs)
            self._cache.pop(key, None)
            if self.disk_cache_dir:
                p = Path(self.disk_cache_dir) / f"{key}.pkl"
                if p.exists():
                    p.unlink()
        self._stats.cache_size = len(self._cache)

    def get_stats(self) -> CacheStats:
        self._stats.cache_size = len(self._cache)
        if self._time_savings:
            self._stats.avg_hit_save_ms = float(np.mean(self._time_savings))
        if self.disk_cache_dir:
            disk_dir = Path(self.disk_cache_dir)
            files = list(disk_dir.glob("*.pkl"))
            self._stats.disk_entries = len(files)
            self._stats.disk_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
        return self._stats

    def _hash_inputs(self, inputs: dict[str, np.ndarray]) -> str:
        h = hashlib.sha256()
        for name in sorted(inputs.keys()):
            h.update(name.encode())
            h.update(inputs[name].tobytes())
        return h.hexdigest()[:24]
