"""Compilation cache manager for MIGraphX and ORT.

MIGraphX compiles ONNX graphs into GPU programs on first run (expensive).
This module manages the cache to avoid recompilation:
  - Track cache files (location, size, age)
  - Invalidate stale caches (model changed, driver changed)
  - Warm the cache before benchmarking
  - Clean up old cache entries
  - Report cache hit/miss stats
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.cache")

DEFAULT_CACHE_DIRS = [
    Path.home() / ".cache" / "migraphx",
    Path.home() / ".cache" / "onnxruntime",
    Path("/tmp") / "migraphx_cache",
]


@dataclass
class CacheEntry:
    path: str
    size_mb: float
    modified: float
    age_hours: float
    model_hash: str = ""


@dataclass
class CacheStats:
    total_entries: int
    total_size_mb: float
    oldest_hours: float
    newest_hours: float
    cache_dirs: list[str]
    entries: list[CacheEntry] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Cache directories  : {len(self.cache_dirs)}",
            f"  Total entries      : {self.total_entries}",
            f"  Total size         : {self.total_size_mb:.1f} MB",
        ]
        if self.total_entries > 0:
            lines.append(f"  Oldest entry       : {self.oldest_hours:.1f} hours ago")
            lines.append(f"  Newest entry       : {self.newest_hours:.1f} hours ago")
        for d in self.cache_dirs:
            lines.append(f"    {d}")
        return "\n".join(lines)


class CacheManager:
    """Manage inference compilation caches."""

    def __init__(self, extra_dirs: list[str] | None = None):
        self.cache_dirs = [str(d) for d in DEFAULT_CACHE_DIRS if d.exists()]
        if extra_dirs:
            self.cache_dirs.extend(extra_dirs)

        migraphx_cache = os.environ.get("MIGRAPHX_CACHE_DIR")
        if migraphx_cache and migraphx_cache not in self.cache_dirs:
            self.cache_dirs.append(migraphx_cache)

    def stats(self) -> CacheStats:
        entries: list[CacheEntry] = []
        now = time.time()

        for cache_dir in self.cache_dirs:
            p = Path(cache_dir)
            if not p.exists():
                continue
            for f in p.rglob("*"):
                if f.is_file():
                    stat = f.stat()
                    entries.append(CacheEntry(
                        path=str(f),
                        size_mb=stat.st_size / (1024 * 1024),
                        modified=stat.st_mtime,
                        age_hours=(now - stat.st_mtime) / 3600,
                    ))

        total_size = sum(e.size_mb for e in entries)
        oldest = max(e.age_hours for e in entries) if entries else 0
        newest = min(e.age_hours for e in entries) if entries else 0

        return CacheStats(
            total_entries=len(entries),
            total_size_mb=total_size,
            oldest_hours=oldest,
            newest_hours=newest,
            cache_dirs=self.cache_dirs,
            entries=entries,
        )

    def clean(self, max_age_hours: float = 168, max_size_mb: float = 0, dry_run: bool = False) -> dict:
        """Clean old or oversized cache entries. Default max age = 7 days."""
        stats = self.stats()
        to_remove = []

        for entry in stats.entries:
            if entry.age_hours > max_age_hours:
                to_remove.append(entry)

        if max_size_mb > 0 and stats.total_size_mb > max_size_mb:
            sorted_entries = sorted(stats.entries, key=lambda e: e.age_hours, reverse=True)
            cumulative = stats.total_size_mb
            for entry in sorted_entries:
                if cumulative <= max_size_mb:
                    break
                if entry not in to_remove:
                    to_remove.append(entry)
                cumulative -= entry.size_mb

        removed_mb = 0
        for entry in to_remove:
            if not dry_run:
                try:
                    os.remove(entry.path)
                    removed_mb += entry.size_mb
                except OSError as e:
                    log.warning("Failed to remove %s: %s", entry.path, e)
            else:
                removed_mb += entry.size_mb

        return {
            "entries_removed": len(to_remove),
            "space_freed_mb": removed_mb,
            "dry_run": dry_run,
        }

    def warm(self, model_path: str, provider: str = "MIGraphXExecutionProvider") -> float:
        """Warm the cache by running a single inference. Returns compilation time in seconds."""
        import onnxruntime as ort
        import numpy as np

        start = time.time()
        session = ort.InferenceSession(
            model_path,
            providers=[provider, "CPUExecutionProvider"],
        )
        feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        session.run(None, feed)
        elapsed = time.time() - start
        log.info("Cache warmed for %s in %.1fs", model_path, elapsed)
        return elapsed

    @staticmethod
    def model_hash(model_path: str) -> str:
        """Compute SHA256 hash of model file for cache invalidation."""
        h = hashlib.sha256()
        with open(model_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()[:16]
