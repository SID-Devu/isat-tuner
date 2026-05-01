"""Semantic prompt cache with radix-tree prefix matching, LRU eviction,
multi-tenant namespaces, and optional semantic similarity lookup."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("isat.prompt_cache")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CacheHit:
    kv_data: np.ndarray
    matched_tokens: int
    total_prompt_tokens: int
    savings_ratio: float
    cache_key: str


@dataclass
class CacheStats:
    total_entries: int
    memory_mb: float
    hit_rate: float
    miss_rate: float
    total_hits: int
    total_misses: int
    avg_prefix_match_len: float
    cost_savings_estimate: float
    per_namespace: Dict[str, dict] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"  Entries        : {self.total_entries}",
            f"  Memory         : {self.memory_mb:.1f} MB",
            f"  Hit rate       : {self.hit_rate:.2%}",
            f"  Total hits     : {self.total_hits}",
            f"  Total misses   : {self.total_misses}",
            f"  Avg prefix len : {self.avg_prefix_match_len:.1f}",
            f"  Cost savings   : ${self.cost_savings_estimate:.4f}",
        ]
        for ns, info in self.per_namespace.items():
            lines.append(f"  [{ns}] entries={info['entries']}, "
                         f"hits={info['hits']}, misses={info['misses']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Radix tree
# ---------------------------------------------------------------------------


class RadixNode:
    __slots__ = ("token_ids", "children", "kv_data", "ref_count",
                 "last_access", "hit_count")

    def __init__(self, token_ids: Optional[List[int]] = None):
        self.token_ids: List[int] = token_ids or []
        self.children: Dict[int, RadixNode] = {}
        self.kv_data: Optional[np.ndarray] = None
        self.ref_count: int = 0
        self.last_access: float = 0.0
        self.hit_count: int = 0


class RadixTree:
    """Compressed radix tree keyed by token-id sequences.

    Each edge stores a *run* of token ids.  Shared prefixes are split
    lazily so that every insertion and lookup is O(|key|).
    """

    def __init__(self) -> None:
        self.root = RadixNode()
        self._node_count = 0

    # -- public API ---------------------------------------------------------

    def insert(self, token_ids: List[int], kv_data: np.ndarray) -> None:
        """Insert *token_ids* with associated KV data, splitting existing
        nodes where they share a prefix."""
        node = self.root
        pos = 0

        while pos < len(token_ids):
            first = token_ids[pos]

            if first not in node.children:
                child = RadixNode(token_ids[pos:])
                child.kv_data = kv_data
                child.last_access = time.time()
                child.hit_count = 1
                child.ref_count = 1
                node.children[first] = child
                self._node_count += 1
                return

            child = node.children[first]
            edge = child.token_ids
            match_len = 0
            while match_len < len(edge) and pos + match_len < len(token_ids) \
                    and edge[match_len] == token_ids[pos + match_len]:
                match_len += 1

            if match_len == len(edge):
                pos += match_len
                node = child
                continue

            self._split_node(child, match_len)
            node = child  # child is now the split prefix node
            pos += match_len

        node.kv_data = kv_data
        node.last_access = time.time()
        node.hit_count += 1
        node.ref_count += 1

    def find_longest_prefix(
        self, token_ids: List[int]
    ) -> Tuple[int, Optional[np.ndarray]]:
        """Return *(matched_length, kv_data)* for the longest matching prefix."""
        node = self.root
        pos = 0
        best_len = 0
        best_kv: Optional[np.ndarray] = None

        while pos < len(token_ids):
            first = token_ids[pos]
            if first not in node.children:
                break

            child = node.children[first]
            edge = child.token_ids
            match_len = 0
            while match_len < len(edge) and pos + match_len < len(token_ids) \
                    and edge[match_len] == token_ids[pos + match_len]:
                match_len += 1

            if match_len < len(edge):
                break

            pos += match_len
            node = child

            if node.kv_data is not None:
                best_len = pos
                best_kv = node.kv_data
                node.last_access = time.time()
                node.hit_count += 1

        return best_len, best_kv

    def evict_lru(self, max_nodes: int) -> int:
        """Remove least-recently-used leaf nodes until the tree has at most
        *max_nodes* data-bearing nodes.  Returns the number removed."""
        removed = 0
        while self._node_count > max_nodes:
            leaf = self._find_lru_leaf(self.root)
            if leaf is None:
                break
            self._remove_leaf(self.root, leaf)
            removed += 1
        return removed

    # -- internal helpers ---------------------------------------------------

    def _split_node(self, node: RadixNode, split_pos: int) -> None:
        """Split *node* at *split_pos* so that ``node`` keeps the first
        *split_pos* tokens and a new child holds the remainder."""
        tail = RadixNode(node.token_ids[split_pos:])
        tail.children = node.children
        tail.kv_data = node.kv_data
        tail.ref_count = node.ref_count
        tail.last_access = node.last_access
        tail.hit_count = node.hit_count

        node.token_ids = node.token_ids[:split_pos]
        node.children = {tail.token_ids[0]: tail}
        node.kv_data = None
        node.ref_count = 0
        node.hit_count = 0
        self._node_count += 1

    def _find_lru_leaf(self, node: RadixNode) -> Optional[RadixNode]:
        """Walk the tree and return the leaf with the oldest last_access."""
        best: Optional[RadixNode] = None
        best_time = float("inf")

        for child in node.children.values():
            if not child.children and child.kv_data is not None:
                if child.last_access < best_time:
                    best = child
                    best_time = child.last_access
            else:
                candidate = self._find_lru_leaf(child)
                if candidate is not None and candidate.last_access < best_time:
                    best = candidate
                    best_time = candidate.last_access
        return best

    def _remove_leaf(self, parent: RadixNode, target: RadixNode) -> bool:
        for key, child in list(parent.children.items()):
            if child is target:
                del parent.children[key]
                self._node_count -= 1
                return True
            if self._remove_leaf(child, target):
                return True
        return False

    def _collect_all_data(
        self, node: RadixNode, prefix: List[int]
    ) -> List[Tuple[List[int], np.ndarray, float, int]]:
        """Collect (token_ids, kv_data, last_access, hit_count) for every
        data-bearing node reachable from *node*."""
        results: List[Tuple[List[int], np.ndarray, float, int]] = []
        current = prefix + node.token_ids
        if node.kv_data is not None:
            results.append((current, node.kv_data, node.last_access,
                            node.hit_count))
        for child in node.children.values():
            results.extend(self._collect_all_data(child, current))
        return results


# ---------------------------------------------------------------------------
# Semantic matcher (optional, lazy-loaded)
# ---------------------------------------------------------------------------


class SemanticMatcher:
    """Optional semantic similarity lookup using a lightweight ONNX model.

    When no model is available the matcher falls back to character-level
    hashing so the rest of the cache still works.
    """

    def __init__(
        self,
        embedding_model_path: Optional[str] = None,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self._session = None
        self._model_path = embedding_model_path

        if embedding_model_path and os.path.isfile(embedding_model_path):
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(embedding_model_path)
                log.info("Loaded embedding model from %s",
                         embedding_model_path)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load embedding model: %s", exc)

    def embed(self, text_or_ids: Any) -> np.ndarray:
        """Compute an embedding vector for *text_or_ids*.

        * If an ONNX model is loaded, the input is run through the model.
        * Otherwise, a deterministic hash-based pseudo-embedding is produced
          so that exact-string matches still work.
        """
        if isinstance(text_or_ids, (list, np.ndarray)):
            text = " ".join(str(t) for t in text_or_ids)
        else:
            text = str(text_or_ids)

        if self._session is not None:
            return self._run_model(text)

        return self._hash_embed(text)

    def is_similar(
        self, query_embedding: np.ndarray, cached_embedding: np.ndarray
    ) -> bool:
        """Return ``True`` when cosine similarity exceeds the threshold."""
        sim = self._cosine_similarity(query_embedding, cached_embedding)
        return float(sim) >= self.similarity_threshold

    def find_semantic_match(
        self,
        query: Any,
        cache_entries: List[Tuple[str, np.ndarray, Any]],
    ) -> Optional[Tuple[str, Any]]:
        """Find the closest semantically similar cached prompt.

        *cache_entries* is a list of ``(cache_key, embedding, payload)``
        tuples.  Returns ``(cache_key, payload)`` for the best match, or
        ``None`` if nothing exceeds the threshold.
        """
        q_emb = self.embed(query)
        best_key: Optional[str] = None
        best_payload: Any = None
        best_sim = -1.0

        for key, emb, payload in cache_entries:
            sim = self._cosine_similarity(q_emb, emb)
            if sim >= self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_key = key
                best_payload = payload

        if best_key is not None:
            return best_key, best_payload
        return None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel().astype(np.float64)
        b = b.ravel().astype(np.float64)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-12:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _run_model(self, text: str) -> np.ndarray:
        tokens = [ord(c) for c in text[:512]]
        arr = np.array([tokens], dtype=np.int64)
        inp_name = self._session.get_inputs()[0].name  # type: ignore[union-attr]
        out = self._session.run(None, {inp_name: arr})  # type: ignore[union-attr]
        return np.array(out[0], dtype=np.float32).ravel()

    @staticmethod
    def _hash_embed(text: str, dim: int = 128) -> np.ndarray:
        digest = hashlib.sha512(text.encode()).digest()
        rng = np.random.RandomState(
            int.from_bytes(digest[:4], "little")
        )
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


# ---------------------------------------------------------------------------
# Main cache
# ---------------------------------------------------------------------------

_COST_PER_TOKEN = 0.00001  # rough $/token for savings estimate


class PromptCache:
    """Prompt KV-cache with radix-tree exact prefix matching, optional
    semantic similarity lookup, multi-tenant namespace support, and
    LRU eviction with frequency weighting."""

    def __init__(
        self,
        max_entries: int = 1000,
        max_memory_mb: float = 1024,
        enable_semantic: bool = False,
        embedding_model_path: Optional[str] = None,
        similarity_threshold: float = 0.95,
    ) -> None:
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.enable_semantic = enable_semantic

        self._trees: Dict[str, RadixTree] = {}
        self._hits: Dict[str, int] = {}
        self._misses: Dict[str, int] = {}
        self._match_lengths: List[int] = []
        self._semantic_index: Dict[str, List[Tuple[str, np.ndarray, List[int]]]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        self._matcher: Optional[SemanticMatcher] = None
        if enable_semantic:
            self._matcher = SemanticMatcher(
                embedding_model_path=embedding_model_path,
                similarity_threshold=similarity_threshold,
            )

    # -- helpers ------------------------------------------------------------

    def _tree(self, namespace: str) -> RadixTree:
        if namespace not in self._trees:
            self._trees[namespace] = RadixTree()
            self._hits[namespace] = 0
            self._misses[namespace] = 0
        return self._trees[namespace]

    @staticmethod
    def _cache_key(token_ids: List[int]) -> str:
        h = hashlib.sha256(
            np.array(token_ids, dtype=np.int32).tobytes()
        ).hexdigest()[:16]
        return h

    def _memory_mb(self) -> float:
        total = 0.0
        for tree in self._trees.values():
            for _, kv, _, _ in tree._collect_all_data(tree.root, []):
                total += kv.nbytes
        return total / (1024 * 1024)

    # -- public API ---------------------------------------------------------

    def get(
        self, token_ids: List[int], namespace: str = "default"
    ) -> Optional[CacheHit]:
        """Look up exact prefix match via the radix tree."""
        tree = self._tree(namespace)
        matched, kv = tree.find_longest_prefix(token_ids)

        if matched == 0 or kv is None:
            self._misses[namespace] = self._misses.get(namespace, 0) + 1
            return None

        self._hits[namespace] = self._hits.get(namespace, 0) + 1
        self._match_lengths.append(matched)

        return CacheHit(
            kv_data=kv,
            matched_tokens=matched,
            total_prompt_tokens=len(token_ids),
            savings_ratio=matched / max(len(token_ids), 1),
            cache_key=self._cache_key(token_ids[:matched]),
        )

    def get_semantic(
        self, prompt_text: str, namespace: str = "default"
    ) -> Optional[CacheHit]:
        """Look up semantically similar cached prompt."""
        if self._matcher is None:
            return None

        entries = self._semantic_index.get(namespace, [])
        if not entries:
            self._misses[namespace] = self._misses.get(namespace, 0) + 1
            return None

        result = self._matcher.find_semantic_match(prompt_text, entries)
        if result is None:
            self._misses[namespace] = self._misses.get(namespace, 0) + 1
            return None

        cache_key, token_ids = result
        tree = self._tree(namespace)
        matched, kv = tree.find_longest_prefix(token_ids)

        if matched == 0 or kv is None:
            self._misses[namespace] = self._misses.get(namespace, 0) + 1
            return None

        self._hits[namespace] = self._hits.get(namespace, 0) + 1
        self._match_lengths.append(matched)

        return CacheHit(
            kv_data=kv,
            matched_tokens=matched,
            total_prompt_tokens=len(token_ids),
            savings_ratio=matched / max(len(token_ids), 1),
            cache_key=cache_key,
        )

    def put(
        self,
        token_ids: List[int],
        kv_data: np.ndarray,
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store *token_ids* → *kv_data* in the radix tree and optional
        semantic index.  Returns the cache key."""
        tree = self._tree(namespace)
        tree.insert(token_ids, kv_data)

        key = self._cache_key(token_ids)

        if metadata:
            self._metadata[key] = metadata

        if self._matcher is not None:
            text = metadata.get("prompt_text", "") if metadata else ""
            if not text:
                text = " ".join(str(t) for t in token_ids)
            emb = self._matcher.embed(text)
            ns_entries = self._semantic_index.setdefault(namespace, [])
            ns_entries.append((key, emb, token_ids))

        if self._memory_mb() > self.max_memory_mb:
            self.evict(target_memory_mb=self.max_memory_mb * 0.8)

        return key

    def evict(self, target_memory_mb: Optional[float] = None) -> int:
        """LRU eviction with frequency-weighted scoring.

        ``score = hit_count / (time_since_last_access + 1)``

        Nodes with the *lowest* score are evicted first.
        """
        target = target_memory_mb if target_memory_mb is not None \
            else self.max_memory_mb * 0.8
        removed = 0
        now = time.time()

        while self._memory_mb() > target:
            worst_node: Optional[RadixNode] = None
            worst_score = float("inf")
            worst_tree: Optional[RadixTree] = None

            for tree in self._trees.values():
                for _, kv, last, hits in tree._collect_all_data(tree.root, []):
                    score = hits / (now - last + 1)
                    if score < worst_score:
                        worst_score = score
                        leaf = tree._find_lru_leaf(tree.root)
                        if leaf is not None:
                            worst_node = leaf
                            worst_tree = tree

            if worst_node is None or worst_tree is None:
                break

            worst_tree._remove_leaf(worst_tree.root, worst_node)
            worst_tree._node_count -= 1
            removed += 1

        return removed

    def get_stats(self, namespace: Optional[str] = None) -> CacheStats:
        """Return current cache statistics."""
        namespaces = [namespace] if namespace else list(self._trees.keys())
        if not namespaces:
            namespaces = ["default"]

        total_entries = 0
        total_hits = 0
        total_misses = 0
        per_ns: Dict[str, dict] = {}

        for ns in namespaces:
            tree = self._trees.get(ns)
            entries = len(tree._collect_all_data(tree.root, [])) if tree else 0
            hits = self._hits.get(ns, 0)
            misses = self._misses.get(ns, 0)
            total_entries += entries
            total_hits += hits
            total_misses += misses
            per_ns[ns] = {"entries": entries, "hits": hits, "misses": misses}

        total_lookups = total_hits + total_misses
        avg_match = (
            sum(self._match_lengths) / len(self._match_lengths)
            if self._match_lengths else 0.0
        )

        return CacheStats(
            total_entries=total_entries,
            memory_mb=self._memory_mb(),
            hit_rate=total_hits / max(total_lookups, 1),
            miss_rate=total_misses / max(total_lookups, 1),
            total_hits=total_hits,
            total_misses=total_misses,
            avg_prefix_match_len=avg_match,
            cost_savings_estimate=sum(self._match_lengths) * _COST_PER_TOKEN,
            per_namespace=per_ns,
        )

    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear cache for *namespace*, or all namespaces if ``None``."""
        if namespace is not None:
            self._trees.pop(namespace, None)
            self._hits.pop(namespace, None)
            self._misses.pop(namespace, None)
            self._semantic_index.pop(namespace, None)
        else:
            self._trees.clear()
            self._hits.clear()
            self._misses.clear()
            self._semantic_index.clear()
            self._metadata.clear()
            self._match_lengths.clear()

    def save(self, path: str) -> None:
        """Persist cache to disk as a pickle file."""
        data: Dict[str, Any] = {}
        for ns, tree in self._trees.items():
            data[ns] = tree._collect_all_data(tree.root, [])

        payload = {
            "entries": data,
            "hits": dict(self._hits),
            "misses": dict(self._misses),
            "match_lengths": list(self._match_lengths),
            "metadata": dict(self._metadata),
            "semantic_index": {
                ns: [(k, emb.tolist(), tids)
                     for k, emb, tids in entries]
                for ns, entries in self._semantic_index.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("Cache saved to %s", path)

    def load(self, path: str) -> None:
        """Load cache from a previously saved pickle file."""
        with open(path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301

        self.clear()
        for ns, items in payload.get("entries", {}).items():
            for tids, kv, last, hits in items:
                tree = self._tree(ns)
                tree.insert(tids, kv)

        self._hits.update(payload.get("hits", {}))
        self._misses.update(payload.get("misses", {}))
        self._match_lengths.extend(payload.get("match_lengths", []))
        self._metadata.update(payload.get("metadata", {}))

        for ns, entries in payload.get("semantic_index", {}).items():
            self._semantic_index[ns] = [
                (k, np.array(emb, dtype=np.float32), tids)
                for k, emb, tids in entries
            ]
        log.info("Cache loaded from %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def prompt_cache_manage(action: str = "stats", **kwargs: Any) -> Any:
    """CLI entry point for cache management.

    Actions
    -------
    stats    — print cache statistics
    clear    — clear a namespace (or all)
    warmup   — pre-populate cache from a token file
    export   — save cache to disk
    """
    cache = kwargs.pop("_cache", None) or PromptCache()

    if action == "stats":
        ns = kwargs.get("namespace")
        stats = cache.get_stats(namespace=ns)
        print(stats.summary())
        return stats

    if action == "clear":
        ns = kwargs.get("namespace")
        cache.clear(namespace=ns)
        print(f"Cache cleared{' (namespace=' + ns + ')' if ns else ''}.")
        return None

    if action == "warmup":
        token_file = kwargs.get("token_file")
        ns = kwargs.get("namespace", "default")
        if not token_file or not os.path.isfile(token_file):
            raise FileNotFoundError(f"Token file not found: {token_file}")
        with open(token_file) as f:
            data = json.load(f)
        count = 0
        for entry in data:
            tids = entry["token_ids"]
            kv = np.zeros((len(tids), 64), dtype=np.float16)
            cache.put(tids, kv, namespace=ns)
            count += 1
        print(f"Warmed cache with {count} entries into namespace '{ns}'.")
        return count

    if action == "export":
        path = kwargs.get("path", "prompt_cache.pkl")
        cache.save(path)
        print(f"Cache exported to {path}.")
        return path

    raise ValueError(f"Unknown action: {action!r}")
