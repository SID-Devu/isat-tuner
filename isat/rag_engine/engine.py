"""Production RAG engine — chunking, embedding, HNSW vector index, hybrid search, reranking, citation tracking."""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResult:
    answer: str
    sources: List[Chunk]
    citations: List[Dict[str, Any]]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    num_chunks_retrieved: int
    num_chunks_used: int


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------

class TextChunker:
    STRATEGIES = ("fixed", "recursive", "semantic", "markdown")

    def __init__(
        self,
        strategy: str = "recursive",
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from {self.STRATEGIES}")
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []
        dispatch = {
            "fixed": self._chunk_fixed,
            "recursive": self._chunk_recursive,
            "semantic": self._chunk_recursive,
            "markdown": self._chunk_markdown,
        }
        return dispatch[self.strategy](text)

    # -- Fixed: character-based sliding window --------------------------------

    def _chunk_fixed(self, text: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(text=text[start:end], start_idx=start, end_idx=end))
            if end >= len(text):
                break
            start += self.chunk_size - self.overlap
        return chunks

    # -- Recursive: paragraphs → sentences → words ---------------------------

    def _chunk_recursive(self, text: str) -> List[Chunk]:
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, offset=0)

    def _recursive_split(self, text: str, separators: List[str], offset: int) -> List[Chunk]:
        if len(text) <= self.chunk_size:
            return [Chunk(text=text, start_idx=offset, end_idx=offset + len(text))]

        sep = separators[0] if separators else " "
        parts = text.split(sep)
        chunks: List[Chunk] = []
        current = ""
        current_start = offset

        for i, part in enumerate(parts):
            candidate = (current + sep + part) if current else part
            if len(candidate) > self.chunk_size and current:
                if len(current) > self.chunk_size and len(separators) > 1:
                    chunks.extend(self._recursive_split(current, separators[1:], current_start))
                else:
                    chunks.append(Chunk(text=current, start_idx=current_start, end_idx=current_start + len(current)))
                current_start = current_start + len(current) + len(sep)
                current = part
            else:
                current = candidate
        if current:
            if len(current) > self.chunk_size and len(separators) > 1:
                chunks.extend(self._recursive_split(current, separators[1:], current_start))
            else:
                chunks.append(Chunk(text=current, start_idx=current_start, end_idx=current_start + len(current)))

        return self._apply_overlap(chunks)

    def _apply_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks
        result: List[Chunk] = [chunks[0]]
        for prev, cur in zip(chunks, chunks[1:]):
            overlap_text = prev.text[-self.overlap:]
            merged_text = overlap_text + cur.text
            start = cur.start_idx - len(overlap_text)
            result.append(Chunk(text=merged_text, start_idx=start, end_idx=cur.end_idx, metadata=cur.metadata))
        return result

    # -- Markdown: split by headers -------------------------------------------

    def _chunk_markdown(self, text: str) -> List[Chunk]:
        header_pattern = re.compile(r"^(#{1,6}\s.+)$", re.MULTILINE)
        splits = header_pattern.split(text)

        chunks: List[Chunk] = []
        pos = 0
        current = ""
        current_start = 0

        for part in splits:
            if header_pattern.match(part):
                if current.strip():
                    chunks.append(Chunk(text=current.strip(), start_idx=current_start, end_idx=current_start + len(current.strip())))
                current = part
                current_start = text.find(part, pos)
                pos = current_start + len(part)
            else:
                current += part
                pos += len(part)

        if current.strip():
            chunks.append(Chunk(text=current.strip(), start_idx=current_start, end_idx=current_start + len(current.strip())))

        final: List[Chunk] = []
        for c in chunks:
            if len(c.text) > self.chunk_size:
                sub_chunker = TextChunker(strategy="recursive", chunk_size=self.chunk_size, overlap=self.overlap)
                final.extend(sub_chunker.chunk(c.text))
            else:
                final.append(c)
        return final


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(
        self,
        model_path: Optional[str] = None,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.model_path = model_path
        self.provider = provider
        self._onnx_session = None
        self._use_tfidf = True
        self._vocab: Dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._fitted = False

        if model_path and os.path.isfile(model_path):
            try:
                import onnxruntime as ort  # lazy
                self._onnx_session = ort.InferenceSession(model_path, providers=[provider])
                self._use_tfidf = False
            except Exception:
                self._use_tfidf = True

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self._use_tfidf and self._onnx_session is not None:
            return self._embed_onnx(list(texts))
        return self._embed_tfidf(list(texts))

    def embed_query(self, query: str) -> np.ndarray:
        vec = self.embed([query])
        return vec[0]

    # -- ONNX embedding -------------------------------------------------------

    def _embed_onnx(self, texts: List[str]) -> np.ndarray:
        from tokenizers import Tokenizer  # lazy

        session = self._onnx_session
        input_names = [inp.name for inp in session.get_inputs()]

        tokenizer_path = str(Path(self.model_path).parent / "tokenizer.json")
        if os.path.isfile(tokenizer_path):
            tokenizer = Tokenizer.from_file(tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

        all_embeddings: List[np.ndarray] = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer.encode_batch(batch)
            max_len = max(len(e.ids) for e in encoded)

            input_ids = np.zeros((len(batch), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)
            for j, enc in enumerate(encoded):
                length = len(enc.ids)
                input_ids[j, :length] = enc.ids
                attention_mask[j, :length] = 1

            feeds: Dict[str, np.ndarray] = {}
            if "input_ids" in input_names:
                feeds["input_ids"] = input_ids
            if "attention_mask" in input_names:
                feeds["attention_mask"] = attention_mask
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(input_ids)

            outputs = session.run(None, feeds)
            token_emb = outputs[0]  # (batch, seq, dim)
            mask_expanded = attention_mask[:, :, None].astype(np.float32)
            pooled = (token_emb * mask_expanded).sum(axis=1) / mask_expanded.sum(axis=1).clip(min=1e-9)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
            all_embeddings.append(pooled / norms)

        return np.vstack(all_embeddings)

    # -- TF-IDF fallback ------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _build_vocab(self, texts: List[str]) -> None:
        doc_freq: Dict[str, int] = {}
        for text in texts:
            seen = set(self._tokenize(text))
            for tok in seen:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        self._vocab = {tok: idx for idx, tok in enumerate(sorted(doc_freq.keys()))}
        n = len(texts)
        idf = np.zeros(len(self._vocab), dtype=np.float32)
        for tok, idx in self._vocab.items():
            idf[idx] = math.log((n + 1) / (doc_freq[tok] + 1)) + 1.0
        self._idf = idf
        self._fitted = True

    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            self._build_vocab(texts)

        dim = len(self._vocab)
        if dim == 0:
            return np.zeros((len(texts), 1), dtype=np.float32)

        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, count in tf.items():
                if tok in self._vocab:
                    idx = self._vocab[tok]
                    matrix[i, idx] = (1 + math.log(count)) * self._idf[idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        return matrix / norms


# ---------------------------------------------------------------------------
# HNSWIndex — pure-numpy approximate nearest neighbor search
# ---------------------------------------------------------------------------

class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200) -> None:
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # max edges at layer 0
        self.ef_construction = ef_construction
        self.mult = 1.0 / math.log(M) if M > 1 else 1.0

        self._vectors: List[np.ndarray] = []
        self._ids: List[Any] = []
        self._graphs: List[Dict[int, List[int]]] = []  # per-layer adjacency
        self._node_layers: List[int] = []  # max layer per node
        self._entry_point: int = -1
        self._max_layer: int = -1

    def __len__(self) -> int:
        return len(self._vectors)

    # -- Similarity -----------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim == 1:
            a = a[None, :]
        if b.ndim == 1:
            b = b[None, :]
        dot = a @ b.T
        norm_a = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
        return dot / (norm_a * norm_b.T)

    # -- Random level ---------------------------------------------------------

    def _random_level(self) -> int:
        r = np.random.uniform()
        level = int(-math.log(r) * self.mult) if r > 0 else 0
        return level

    # -- Greedy search at a single layer -------------------------------------

    def _search_layer(self, query: np.ndarray, entry: int, ef: int, layer: int) -> List[Tuple[float, int]]:
        if layer >= len(self._graphs) or entry < 0:
            return []

        graph = self._graphs[layer]
        visited = {entry}
        sim = float(self._cosine_similarity(query, self._vectors[entry])[0, 0])
        candidates = [(-sim, entry)]  # min-heap by negative similarity
        results = [(-sim, entry)]

        import heapq
        heapq.heapify(candidates)

        while candidates:
            neg_dist, current = heapq.heappop(candidates)
            worst_result = max(results)[0] if results else float("inf")
            if -neg_dist < -worst_result and len(results) >= ef:
                break

            for neighbor in graph.get(current, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nsim = float(self._cosine_similarity(query, self._vectors[neighbor])[0, 0])
                worst_result = max(results)[0] if results else float("inf")
                if len(results) < ef or -nsim < worst_result:
                    heapq.heappush(candidates, (-nsim, neighbor))
                    results.append((-nsim, neighbor))
                    if len(results) > ef:
                        results.sort()
                        results.pop()

        return [(- neg_s, idx) for neg_s, idx in results]

    # -- Connect new node to neighbors ---------------------------------------

    def _select_neighbors(self, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        candidates.sort(key=lambda x: -x[0])
        return [idx for _, idx in candidates[:M]]

    def _connect(self, node: int, neighbors: List[int], layer: int, M_max: int) -> None:
        graph = self._graphs[layer]
        graph.setdefault(node, [])
        for nb in neighbors:
            if nb == node:
                continue
            graph[node].append(nb)
            graph.setdefault(nb, [])
            graph[nb].append(node)
            if len(graph[nb]) > M_max:
                sims = [
                    (float(self._cosine_similarity(self._vectors[nb], self._vectors[n])[0, 0]), n)
                    for n in graph[nb]
                ]
                graph[nb] = self._select_neighbors(sims, M_max)

        if len(graph[node]) > M_max:
            sims = [
                (float(self._cosine_similarity(self._vectors[node], self._vectors[n])[0, 0]), n)
                for n in graph[node]
            ]
            graph[node] = self._select_neighbors(sims, M_max)

    # -- Add vectors ----------------------------------------------------------

    def add(self, vectors: np.ndarray, ids: Optional[List[Any]] = None) -> None:
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {vectors.shape[1]}")

        if ids is None:
            base = len(self._vectors)
            ids = list(range(base, base + len(vectors)))

        for i in range(len(vectors)):
            self._insert_one(vectors[i], ids[i])

    def _insert_one(self, vector: np.ndarray, ext_id: Any) -> None:
        node = len(self._vectors)
        self._vectors.append(vector.copy())
        self._ids.append(ext_id)

        level = self._random_level()
        self._node_layers.append(level)

        while len(self._graphs) <= level:
            self._graphs.append({})

        if self._entry_point < 0:
            self._entry_point = node
            self._max_layer = level
            for lyr in range(level + 1):
                self._graphs[lyr][node] = []
            return

        ep = self._entry_point
        for lyr in range(self._max_layer, level, -1):
            results = self._search_layer(vector, ep, ef=1, layer=lyr)
            if results:
                ep = max(results, key=lambda x: x[0])[1]

        for lyr in range(min(level, self._max_layer), -1, -1):
            M_max = self.M_max0 if lyr == 0 else self.M
            candidates = self._search_layer(vector, ep, ef=self.ef_construction, layer=lyr)
            neighbors = self._select_neighbors(candidates, M_max)
            self._connect(node, neighbors, lyr, M_max)
            if candidates:
                ep = max(candidates, key=lambda x: x[0])[1]

        if level > self._max_layer:
            for lyr in range(self._max_layer + 1, level + 1):
                self._graphs[lyr].setdefault(node, [])
            self._entry_point = node
            self._max_layer = level

    # -- Search ---------------------------------------------------------------

    def search(self, query: np.ndarray, k: int = 10, ef: int = 50) -> List[Tuple[Any, float]]:
        if not self._vectors:
            return []

        ep = self._entry_point
        for lyr in range(self._max_layer, 0, -1):
            results = self._search_layer(query, ep, ef=1, layer=lyr)
            if results:
                ep = max(results, key=lambda x: x[0])[1]

        candidates = self._search_layer(query, ep, ef=max(ef, k), layer=0)
        candidates.sort(key=lambda x: -x[0])
        top_k = candidates[:k]
        return [(self._ids[idx], sim) for sim, idx in top_k]

    # -- Persistence ----------------------------------------------------------

    def save(self, path: str) -> None:
        if not self._vectors:
            np.savez(path, empty=np.array([True]))
            return

        vectors = np.stack(self._vectors)
        ids = np.array(self._ids)
        node_layers = np.array(self._node_layers)

        graph_data = {}
        for lyr_idx, graph in enumerate(self._graphs):
            for node, neighbors in graph.items():
                graph_data[f"g_{lyr_idx}_{node}"] = np.array(neighbors, dtype=np.int64)

        np.savez(
            path,
            vectors=vectors,
            ids=ids,
            node_layers=node_layers,
            entry_point=np.array([self._entry_point]),
            max_layer=np.array([self._max_layer]),
            M=np.array([self.M]),
            ef_construction=np.array([self.ef_construction]),
            num_layers=np.array([len(self._graphs)]),
            **graph_data,
        )

    @classmethod
    def load(cls, path: str) -> "HNSWIndex":
        data = np.load(path, allow_pickle=True)
        if "empty" in data:
            return cls(dim=1)

        vectors = data["vectors"]
        dim = vectors.shape[1]
        M = int(data["M"][0])
        ef_construction = int(data["ef_construction"][0])
        idx = cls(dim=dim, M=M, ef_construction=ef_construction)

        idx._vectors = [vectors[i] for i in range(len(vectors))]
        idx._ids = list(data["ids"])
        idx._node_layers = list(data["node_layers"])
        idx._entry_point = int(data["entry_point"][0])
        idx._max_layer = int(data["max_layer"][0])

        num_layers = int(data["num_layers"][0])
        idx._graphs = [{} for _ in range(num_layers)]
        for key in data.files:
            if key.startswith("g_"):
                parts = key.split("_")
                lyr = int(parts[1])
                node = int(parts[2])
                idx._graphs[lyr][node] = list(data[key])

        return idx


# ---------------------------------------------------------------------------
# BM25Index
# ---------------------------------------------------------------------------

class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._inverted: Dict[str, List[Tuple[int, int]]] = {}  # term → [(doc_id, freq)]
        self._doc_lengths: List[int] = []
        self._avg_dl: float = 0.0
        self._n_docs: int = 0
        self._documents: List[str] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def index(self, documents: Sequence[str]) -> None:
        self._documents = list(documents)
        self._n_docs = len(documents)
        self._inverted = {}
        self._doc_lengths = []

        for doc_id, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            self._doc_lengths.append(len(tokens))
            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            for tok, freq in tf.items():
                self._inverted.setdefault(tok, []).append((doc_id, freq))

        total = sum(self._doc_lengths) if self._doc_lengths else 1
        self._avg_dl = total / max(self._n_docs, 1)

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        if self._n_docs == 0:
            return []

        query_tokens = self._tokenize(query)
        scores = np.zeros(self._n_docs, dtype=np.float64)

        for tok in query_tokens:
            postings = self._inverted.get(tok, [])
            if not postings:
                continue
            df = len(postings)
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, freq in postings:
                dl = self._doc_lengths[doc_id]
                tf_norm = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / self._avg_dl))
                scores[doc_id] += idf * tf_norm

        top_indices = np.argsort(-scores)[:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ---------------------------------------------------------------------------
# RAGEngine
# ---------------------------------------------------------------------------

class RAGEngine:
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        chunk_size: int = 512,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        self.chunker = TextChunker(strategy="recursive", chunk_size=chunk_size)
        self.embedder = Embedder(model_path=embedding_model, provider=provider)
        self._dense_index: Optional[HNSWIndex] = None
        self._sparse_index = BM25Index()
        self._chunks: List[Chunk] = []
        self._chunk_texts: List[str] = []
        self._reranker_session = None
        self._reranker_path: Optional[str] = None

    # -- Ingest ---------------------------------------------------------------

    def ingest(
        self,
        documents: Union[List[str], List[Path]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        all_chunks: List[Chunk] = []
        for i, doc in enumerate(documents):
            text = self._resolve_document(doc)
            meta = metadata[i] if metadata and i < len(metadata) else {}
            chunks = self.chunker.chunk(text)
            for c in chunks:
                c.metadata.update(meta)
                c.metadata.setdefault("doc_index", i)
            all_chunks.extend(chunks)

        self._chunks.extend(all_chunks)
        texts = [c.text for c in all_chunks]
        self._chunk_texts.extend(texts)

        vectors = self.embedder.embed(texts)
        dim = vectors.shape[1]

        if self._dense_index is None:
            self._dense_index = HNSWIndex(dim=dim)

        base_id = len(self._dense_index)
        ids = list(range(base_id, base_id + len(vectors)))
        self._dense_index.add(vectors, ids)

        self._sparse_index.index(self._chunk_texts)

        return len(all_chunks)

    @staticmethod
    def _resolve_document(doc: Union[str, Path]) -> str:
        if isinstance(doc, Path) or (isinstance(doc, str) and os.path.isfile(doc)):
            path = Path(doc)
            if path.is_file():
                return path.read_text(encoding="utf-8", errors="replace")
        return str(doc)

    # -- Retrieve -------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int = 10,
        method: str = "hybrid",
    ) -> List[Tuple[Chunk, float]]:
        if method == "dense":
            return self._retrieve_dense(query, k)
        elif method == "sparse":
            return self._retrieve_sparse(query, k)
        else:
            return self._retrieve_hybrid(query, k)

    def _retrieve_dense(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        if self._dense_index is None or len(self._dense_index) == 0:
            return []
        qvec = self.embedder.embed_query(query)
        results = self._dense_index.search(qvec, k=k)
        return [(self._chunks[idx], score) for idx, score in results if idx < len(self._chunks)]

    def _retrieve_sparse(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        results = self._sparse_index.search(query, k=k)
        return [(self._chunks[idx], score) for idx, score in results if idx < len(self._chunks)]

    def _retrieve_hybrid(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        dense_results = self._retrieve_dense(query, k=k * 2)
        sparse_results = self._retrieve_sparse(query, k=k * 2)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:k]

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_results: List[Tuple[Chunk, float]],
        sparse_results: List[Tuple[Chunk, float]],
        k: int = 60,
    ) -> List[Tuple[Chunk, float]]:
        scores: Dict[int, float] = {}
        chunk_map: Dict[int, Chunk] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            cid = id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = chunk

        for rank, (chunk, _) in enumerate(sparse_results):
            cid = id(chunk)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunk_map[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [(chunk_map[cid], score) for cid, score in ranked]

    # -- Rerank ---------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int = 5,
    ) -> List[Tuple[Chunk, float]]:
        if self._reranker_session is not None:
            return self._rerank_cross_encoder(query, candidates, top_k)
        return candidates[:top_k]

    def _rerank_cross_encoder(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int,
    ) -> List[Tuple[Chunk, float]]:
        try:
            from tokenizers import Tokenizer  # lazy

            tokenizer_path = str(Path(self._reranker_path).parent / "tokenizer.json")
            tokenizer = Tokenizer.from_file(tokenizer_path)

            pairs = [(query, c.text) for c, _ in candidates]
            encoded = [tokenizer.encode(q, p) for q, p in pairs]
            max_len = max(len(e.ids) for e in encoded)

            input_ids = np.zeros((len(pairs), max_len), dtype=np.int64)
            attention_mask = np.zeros((len(pairs), max_len), dtype=np.int64)
            for j, enc in enumerate(encoded):
                length = len(enc.ids)
                input_ids[j, :length] = enc.ids
                attention_mask[j, :length] = 1

            feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
            outputs = self._reranker_session.run(None, feeds)
            scores = outputs[0].flatten()

            scored = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
            scored.sort(key=lambda x: -x[1])
            return scored[:top_k]
        except Exception:
            return candidates[:top_k]

    # -- Generate (full RAG pipeline) -----------------------------------------

    def generate(
        self,
        query: str,
        llm_path: str,
        max_tokens: int = 512,
        k: int = 5,
        **kwargs: Any,
    ) -> RAGResult:
        t_start = time.perf_counter()

        t_ret_start = time.perf_counter()
        retrieved = self.retrieve(query, k=k * 2, method=kwargs.get("method", "hybrid"))
        reranked = self.rerank(query, retrieved, top_k=k)
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        context_chunks = self._pack_context(query, [c for c, _ in reranked], max_tokens=kwargs.get("context_tokens", 2048))
        context = "\n\n".join(c.text for c in context_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        t_gen_start = time.perf_counter()
        answer = self._run_llm(llm_path, prompt, max_tokens, **kwargs)
        generation_ms = (time.perf_counter() - t_gen_start) * 1000

        citations = self._extract_citations(answer, context_chunks)
        total_ms = (time.perf_counter() - t_start) * 1000

        return RAGResult(
            answer=answer,
            sources=context_chunks,
            citations=citations,
            retrieval_time_ms=round(retrieval_ms, 2),
            generation_time_ms=round(generation_ms, 2),
            total_time_ms=round(total_ms, 2),
            num_chunks_retrieved=len(retrieved),
            num_chunks_used=len(context_chunks),
        )

    @staticmethod
    def _run_llm(llm_path: str, prompt: str, max_tokens: int, **kwargs: Any) -> str:
        try:
            import onnxruntime as ort  # lazy
            from tokenizers import Tokenizer  # lazy

            session = ort.InferenceSession(llm_path, providers=["CPUExecutionProvider"])
            tok_path = str(Path(llm_path).parent / "tokenizer.json")
            tokenizer = Tokenizer.from_file(tok_path)
            encoded = tokenizer.encode(prompt)
            input_ids = np.array([encoded.ids], dtype=np.int64)

            generated: List[int] = list(encoded.ids)
            for _ in range(max_tokens):
                feeds = {"input_ids": np.array([generated], dtype=np.int64)}
                out = session.run(None, feeds)
                logits = out[0][0, -1, :]
                next_token = int(np.argmax(logits))
                if next_token == tokenizer.token_to_id("</s>") or next_token == tokenizer.token_to_id("<|endoftext|>"):
                    break
                generated.append(next_token)

            return tokenizer.decode(generated[len(encoded.ids):], skip_special_tokens=True)
        except Exception as exc:
            return f"[LLM unavailable: {exc}] Relevant context was retrieved — see sources."

    # -- Context packing ------------------------------------------------------

    @staticmethod
    def _pack_context(query: str, chunks: List[Chunk], max_tokens: int = 2048) -> List[Chunk]:
        chars_per_token = 4
        budget = max_tokens * chars_per_token
        packed: List[Chunk] = []
        used = len(query)
        for chunk in chunks:
            if used + len(chunk.text) > budget:
                remaining = budget - used
                if remaining > 50:
                    trimmed = Chunk(
                        text=chunk.text[:remaining],
                        start_idx=chunk.start_idx,
                        end_idx=chunk.start_idx + remaining,
                        metadata=chunk.metadata,
                    )
                    packed.append(trimmed)
                break
            packed.append(chunk)
            used += len(chunk.text)
        return packed

    # -- Citation extraction --------------------------------------------------

    @staticmethod
    def _extract_citations(generated_text: str, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        gen_lower = generated_text.lower()
        gen_words = set(re.findall(r"\w+", gen_lower))

        for i, chunk in enumerate(chunks):
            chunk_words = set(re.findall(r"\w+", chunk.text.lower()))
            if not chunk_words:
                continue
            overlap = gen_words & chunk_words
            relevance = len(overlap) / len(chunk_words)
            if relevance > 0.15:
                citations.append({
                    "chunk_index": i,
                    "start_idx": chunk.start_idx,
                    "end_idx": chunk.end_idx,
                    "relevance_score": round(relevance, 3),
                    "metadata": chunk.metadata,
                })

        citations.sort(key=lambda c: -c["relevance_score"])
        return citations

    # -- Persistence ----------------------------------------------------------

    def save_index(self, path: str) -> None:
        base = Path(path)
        base.mkdir(parents=True, exist_ok=True)

        if self._dense_index is not None:
            self._dense_index.save(str(base / "hnsw.npz"))

        import json
        meta = {
            "chunk_texts": self._chunk_texts,
            "chunks": [
                {"text": c.text, "start_idx": c.start_idx, "end_idx": c.end_idx, "metadata": c.metadata}
                for c in self._chunks
            ],
        }
        with open(base / "meta.json", "w") as f:
            json.dump(meta, f)

    def load_index(self, path: str) -> None:
        import json

        base = Path(path)
        hnsw_path = base / "hnsw.npz"
        if hnsw_path.is_file():
            self._dense_index = HNSWIndex.load(str(hnsw_path))

        meta_path = base / "meta.json"
        if meta_path.is_file():
            with open(meta_path) as f:
                meta = json.load(f)
            self._chunk_texts = meta.get("chunk_texts", [])
            self._chunks = [
                Chunk(text=c["text"], start_idx=c["start_idx"], end_idx=c["end_idx"], metadata=c.get("metadata", {}))
                for c in meta.get("chunks", [])
            ]
            self._sparse_index.index(self._chunk_texts)


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def rag_query(
    documents: List[str],
    query: str,
    llm_path: Optional[str] = None,
    **kwargs: Any,
) -> RAGResult:
    """CLI entry point: ingest documents, run a RAG query, return structured result."""
    engine = RAGEngine(
        embedding_model=kwargs.pop("embedding_model", None),
        chunk_size=kwargs.pop("chunk_size", 512),
        provider=kwargs.pop("provider", "CPUExecutionProvider"),
    )
    engine.ingest(documents)

    if llm_path:
        return engine.generate(query, llm_path=llm_path, **kwargs)

    t_start = time.perf_counter()
    retrieved = engine.retrieve(query, k=kwargs.get("k", 10), method=kwargs.get("method", "hybrid"))
    retrieval_ms = (time.perf_counter() - t_start) * 1000

    sources = [c for c, _ in retrieved]
    context_chunks = engine._pack_context(query, sources)
    citations = engine._extract_citations(" ".join(c.text for c in context_chunks), context_chunks)

    return RAGResult(
        answer="[No LLM provided — returning retrieved context]",
        sources=sources,
        citations=citations,
        retrieval_time_ms=round(retrieval_ms, 2),
        generation_time_ms=0.0,
        total_time_ms=round(retrieval_ms, 2),
        num_chunks_retrieved=len(retrieved),
        num_chunks_used=len(context_chunks),
    )
