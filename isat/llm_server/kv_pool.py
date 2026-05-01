from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class KVBlock:
    """Fixed-size block (e.g. 16 tokens) holding key/value tensors for one attention layer."""

    index: int
    block_size: int
    ref_count: int = 1
    num_tokens: int = 0
    content_hash: Optional[int] = None


class KVCachePool:

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype=np.float16,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        # Layout: [layer, key_or_value, block, head, position_in_block, dim]
        self._pool = np.zeros(
            (num_layers, 2, num_blocks, num_heads, block_size, head_dim),
            dtype=dtype,
        )
        self._blocks: Dict[int, KVBlock] = {
            i: KVBlock(index=i, block_size=block_size, ref_count=0)
            for i in range(num_blocks)
        }
        self._free_set: Set[int] = set(range(num_blocks))
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ properties

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_set)

    @property
    def utilization(self) -> float:
        if self.num_blocks == 0:
            return 0.0
        return 1.0 - len(self._free_set) / self.num_blocks

    @property
    def fragmentation_ratio(self) -> float:
        allocated = [b for b in self._blocks.values() if b.ref_count > 0]
        if not allocated:
            return 0.0
        capacity = len(allocated) * self.block_size
        used = sum(b.num_tokens for b in allocated)
        return 1.0 - used / capacity if capacity else 0.0

    # ------------------------------------------------------------------ allocation

    def allocate(self, num_tokens: int) -> List[int]:
        needed = -(-num_tokens // self.block_size)
        with self._lock:
            if len(self._free_set) < needed:
                raise MemoryError(
                    f"Need {needed} blocks, only {len(self._free_set)} available"
                )
            table: List[int] = []
            for _ in range(needed):
                idx = self._free_set.pop()
                blk = self._blocks[idx]
                blk.ref_count = 1
                blk.num_tokens = 0
                blk.content_hash = None
                table.append(idx)
            return table

    def free(self, block_table: List[int]) -> None:
        with self._lock:
            for idx in block_table:
                blk = self._blocks[idx]
                blk.ref_count -= 1
                if blk.ref_count <= 0:
                    blk.ref_count = 0
                    blk.num_tokens = 0
                    blk.content_hash = None
                    self._pool[:, :, idx] = 0
                    self._free_set.add(idx)

    # ------------------------------------------------------------------ copy-on-write

    def copy_on_write(self, block_idx: int) -> int:
        with self._lock:
            src = self._blocks[block_idx]
            if src.ref_count <= 1:
                return block_idx
            if not self._free_set:
                raise MemoryError("No free blocks for copy-on-write")
            new_idx = self._free_set.pop()
            self._pool[:, :, new_idx] = self._pool[:, :, block_idx]
            dst = self._blocks[new_idx]
            dst.ref_count = 1
            dst.num_tokens = src.num_tokens
            dst.content_hash = src.content_hash
            src.ref_count -= 1
            return new_idx

    # ------------------------------------------------------------------ KV read / write

    def get_kv(
        self, block_table: List[int], layer_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        total = sum(self._blocks[b].num_tokens for b in block_table)
        keys = np.empty((total, self.num_heads, self.head_dim), dtype=self.dtype)
        vals = np.empty((total, self.num_heads, self.head_dim), dtype=self.dtype)
        off = 0
        for idx in block_table:
            n = self._blocks[idx].num_tokens
            if n > 0:
                # (num_heads, n, head_dim) -> (n, num_heads, head_dim)
                keys[off : off + n] = np.swapaxes(
                    self._pool[layer_idx, 0, idx, :, :n, :], 0, 1
                )
                vals[off : off + n] = np.swapaxes(
                    self._pool[layer_idx, 1, idx, :, :n, :], 0, 1
                )
                off += n
        return keys, vals

    def set_kv(
        self,
        block_table: List[int],
        layer_idx: int,
        keys: np.ndarray,
        values: np.ndarray,
        start_pos: int,
    ) -> None:
        num_new = keys.shape[0]
        written = 0
        bi = start_pos // self.block_size
        pos = start_pos % self.block_size

        while written < num_new and bi < len(block_table):
            idx = block_table[bi]

            with self._lock:
                if self._blocks[idx].ref_count > 1:
                    new_idx = self.copy_on_write(idx)
                    block_table[bi] = new_idx
                    idx = new_idx

            chunk = min(self.block_size - pos, num_new - written)
            # (chunk, heads, dim) -> (heads, chunk, dim)
            self._pool[layer_idx, 0, idx, :, pos : pos + chunk, :] = np.swapaxes(
                keys[written : written + chunk], 0, 1
            )
            self._pool[layer_idx, 1, idx, :, pos : pos + chunk, :] = np.swapaxes(
                values[written : written + chunk], 0, 1
            )
            self._blocks[idx].num_tokens = max(
                self._blocks[idx].num_tokens, pos + chunk
            )
            written += chunk
            bi += 1
            pos = 0

    # ------------------------------------------------------------------ prefix caching

    def prefix_match(self, block_table: List[int], prefix_blocks: List[int]) -> int:
        matched = 0
        for a, b in zip(block_table, prefix_blocks):
            if a == b:
                matched += 1
                continue
            ha = self._blocks[a].content_hash
            hb = self._blocks[b].content_hash
            if ha is not None and ha == hb:
                matched += 1
            else:
                break
        return matched

    def share_prefix(
        self,
        source_table: List[int],
        target_table: List[int],
        prefix_len: int,
    ) -> List[int]:
        num_shared = prefix_len // self.block_size
        result = list(target_table)
        with self._lock:
            for i in range(min(num_shared, len(source_table), len(result))):
                old = result[i]
                self._blocks[old].ref_count -= 1
                if self._blocks[old].ref_count <= 0:
                    self._blocks[old].ref_count = 0
                    self._blocks[old].num_tokens = 0
                    self._free_set.add(old)
                result[i] = source_table[i]
                self._blocks[source_table[i]].ref_count += 1
        return result

    # ------------------------------------------------------------------ bookkeeping

    def mark_slot_used(self, block_table: List[int], total_tokens: int) -> None:
        """Update token counts without writing KV data (for external KV management)."""
        for i, idx in enumerate(block_table):
            start = i * self.block_size
            self._blocks[idx].num_tokens = min(
                self.block_size, max(0, total_tokens - start)
            )
