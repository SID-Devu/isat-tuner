"""Model routing with cascade escalation, complexity classification, and cost-aware selection."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelEndpoint:
    name: str
    model_path: str
    provider: str = "local"
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 2048
    quality_score: float = 0.5
    latency_p99_ms: float = 500.0


@dataclass
class RoutingResult:
    selected_model: str
    text: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    cost: float = 0.0
    escalations: int = 0
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# ComplexityClassifier
# ---------------------------------------------------------------------------

class ComplexityClassifier:
    """Score prompt complexity on a 0-1 scale to guide model selection."""

    _EMBEDDING_SESSION = None

    def __init__(self, method: str = "heuristic") -> None:
        if method not in ("heuristic", "embedding", "model"):
            raise ValueError(f"Unknown classification method: {method!r}")
        self.method = method
        self._thresholds: List[Tuple[float, str]] = []

    # ---- public API -------------------------------------------------------

    def score(self, prompt: str) -> float:
        if self.method == "heuristic":
            return self._heuristic_score(prompt)
        if self.method == "embedding":
            return self._embedding_score(prompt)
        return self._heuristic_score(prompt)

    def calibrate(self, labeled_data: Sequence[Tuple[str, str]]) -> None:
        """Calibrate complexity thresholds from *(prompt, correct_model)* pairs."""
        scores_by_model: Dict[str, List[float]] = {}
        for prompt, model_name in labeled_data:
            scores_by_model.setdefault(model_name, []).append(self.score(prompt))

        boundaries: List[Tuple[float, str]] = []
        for model_name, scores in scores_by_model.items():
            boundaries.append((float(np.mean(scores)), model_name))
        boundaries.sort(key=lambda t: t[0])
        self._thresholds = boundaries

    # ---- heuristic internals ---------------------------------------------

    @staticmethod
    def _heuristic_score(prompt: str) -> float:
        tokens = prompt.split()
        n_tokens = len(tokens) or 1

        token_length_score = min(n_tokens / 500.0, 1.0)

        unique_ratio = len(set(t.lower() for t in tokens)) / n_tokens
        vocab_score = unique_ratio

        avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
        word_len_score = min(avg_word_len / 12.0, 1.0)

        punct_count = sum(1 for ch in prompt if ch in ",.;:!?()[]{}\"'—–-")
        punct_density = min(punct_count / max(len(prompt), 1) * 10.0, 1.0)

        question_marks = prompt.count("?")
        question_score = min(question_marks / 5.0, 1.0)

        code_blocks = len(re.findall(r"```", prompt))
        code_score = min(code_blocks / 4.0, 1.0)

        weights = np.array([0.25, 0.15, 0.15, 0.10, 0.15, 0.20])
        features = np.array([
            token_length_score,
            vocab_score,
            word_len_score,
            punct_density,
            question_score,
            code_score,
        ])
        raw = float(weights @ features)
        return float(np.clip(raw, 0.0, 1.0))

    # ---- embedding internals ---------------------------------------------

    def _embedding_score(self, prompt: str) -> float:
        embedding = self._get_embedding(prompt)
        norm = float(np.linalg.norm(embedding))
        if norm == 0:
            return 0.5
        variance = float(np.var(embedding))
        return float(np.clip(variance * 10.0, 0.0, 1.0))

    @classmethod
    def _get_embedding(cls, text: str) -> np.ndarray:
        """Return a fixed-size embedding via ONNX, falling back to a hash sketch."""
        try:
            import onnxruntime as ort  # noqa: F811 – lazy

            if cls._EMBEDDING_SESSION is not None:
                inputs = cls._EMBEDDING_SESSION.get_inputs()
                input_name = inputs[0].name
                dummy = np.array([[ord(c) % 256 for c in text[:512]]], dtype=np.int64)
                if dummy.shape[1] < 512:
                    dummy = np.pad(dummy, ((0, 0), (0, 512 - dummy.shape[1])))
                out = cls._EMBEDDING_SESSION.run(None, {input_name: dummy})
                return np.asarray(out[0]).flatten()
        except Exception:
            pass

        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        return rng.randn(384).astype(np.float32)


# ---------------------------------------------------------------------------
# CascadeRouter
# ---------------------------------------------------------------------------

class CascadeRouter:
    """Try models cheapest-first; escalate when confidence is too low."""

    def __init__(
        self,
        endpoints: List[ModelEndpoint],
        confidence_threshold: float = 0.8,
    ) -> None:
        self.endpoints = sorted(endpoints, key=lambda e: e.cost_per_1k_tokens)
        self.confidence_threshold = confidence_threshold
        self._stats: Dict[str, Dict[str, int]] = {
            ep.name: {"hits": 0, "escalations": 0} for ep in self.endpoints
        }

    def route(self, prompt: str) -> RoutingResult:
        escalations = 0
        for i, ep in enumerate(self.endpoints):
            t0 = time.perf_counter()
            logits = self._mock_inference(ep, prompt)
            latency = (time.perf_counter() - t0) * 1000.0
            conf = self._measure_confidence(logits)

            if conf >= self.confidence_threshold or i == len(self.endpoints) - 1:
                n_tokens = max(len(prompt.split()) // 2, 1)
                self._stats[ep.name]["hits"] += 1
                return RoutingResult(
                    selected_model=ep.name,
                    text="",
                    tokens=n_tokens,
                    latency_ms=latency,
                    cost=ep.cost_per_1k_tokens * n_tokens / 1000.0,
                    escalations=escalations,
                    confidence=conf,
                )

            self._stats[ep.name]["escalations"] += 1
            escalations += 1

        last = self.endpoints[-1]
        return RoutingResult(selected_model=last.name, escalations=escalations)

    @staticmethod
    def _measure_confidence(logits: np.ndarray) -> float:
        probs = _softmax(logits)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        max_entropy = math.log(len(logits)) if len(logits) > 1 else 1.0
        normalized = entropy / max_entropy if max_entropy > 0 else 0.0
        return float(np.clip(1.0 - normalized, 0.0, 1.0))

    def get_routing_stats(self) -> Dict[str, Any]:
        total_hits = sum(s["hits"] for s in self._stats.values()) or 1
        total_esc = sum(s["escalations"] for s in self._stats.values())
        per_model: Dict[str, Any] = {}
        for name, s in self._stats.items():
            per_model[name] = {
                "hit_rate": s["hits"] / total_hits,
                "escalation_count": s["escalations"],
            }
        return {
            "per_model": per_model,
            "total_escalation_rate": total_esc / (total_hits + total_esc) if (total_hits + total_esc) else 0.0,
        }

    @staticmethod
    def _mock_inference(ep: ModelEndpoint, prompt: str) -> np.ndarray:
        """Simulate logits from a model; real integration replaces this stub."""
        rng = np.random.RandomState(abs(hash((ep.name, prompt))) % (2**31))
        logits = rng.randn(32).astype(np.float32)
        logits *= ep.quality_score
        return logits


# ---------------------------------------------------------------------------
# CostAwareRouter
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {"cost": 0.4, "quality": 0.4, "latency": 0.2}


class CostAwareRouter:
    """Pick the best endpoint by a weighted cost / quality / latency score."""

    def __init__(
        self,
        endpoints: List[ModelEndpoint],
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.endpoints = list(endpoints)
        self.weights = dict(weights or _DEFAULT_WEIGHTS)
        self._quality_tracker: Dict[str, List[float]] = {
            ep.name: [ep.quality_score] for ep in self.endpoints
        }

    def route(
        self,
        prompt: str,
        slo_latency_ms: Optional[float] = None,
    ) -> RoutingResult:
        candidates = self.endpoints
        if slo_latency_ms is not None:
            candidates = [e for e in candidates if e.latency_p99_ms <= slo_latency_ms]
            if not candidates:
                candidates = self.endpoints  # fallback

        costs = np.array([e.cost_per_1k_tokens for e in candidates])
        quals = np.array([self._current_quality(e.name) for e in candidates])
        lats = np.array([e.latency_p99_ms for e in candidates])

        norm_cost = 1.0 - _min_max_normalize(costs)  # lower cost → higher score
        norm_qual = _min_max_normalize(quals)
        norm_lat = 1.0 - _min_max_normalize(lats)

        scores = (
            self.weights.get("cost", 0.4) * norm_cost
            + self.weights.get("quality", 0.4) * norm_qual
            + self.weights.get("latency", 0.2) * norm_lat
        )
        best_idx = int(np.argmax(scores))
        best = candidates[best_idx]
        n_tokens = max(len(prompt.split()) // 2, 1)
        return RoutingResult(
            selected_model=best.name,
            tokens=n_tokens,
            cost=best.cost_per_1k_tokens * n_tokens / 1000.0,
            latency_ms=best.latency_p99_ms,
            confidence=float(scores[best_idx]),
        )

    def update_quality(self, endpoint_name: str, quality_score: float) -> None:
        self._quality_tracker.setdefault(endpoint_name, []).append(quality_score)

    def _current_quality(self, name: str) -> float:
        history = self._quality_tracker.get(name, [0.5])
        return float(np.mean(history[-50:]))


# ---------------------------------------------------------------------------
# ModelRouter (main facade)
# ---------------------------------------------------------------------------

class ModelRouter:
    """Unified entry point for model routing and generation."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._endpoints: List[ModelEndpoint] = []
        self._cascade: Optional[CascadeRouter] = None
        self._cost_aware: Optional[CostAwareRouter] = None
        self._classifier = ComplexityClassifier()

        if config_path is not None:
            self._load_config(config_path)

    # ---- configuration ----------------------------------------------------

    def add_endpoint(self, endpoint: ModelEndpoint) -> None:
        self._endpoints.append(endpoint)
        self._rebuild_routers()

    def _rebuild_routers(self) -> None:
        if self._endpoints:
            self._cascade = CascadeRouter(list(self._endpoints))
            self._cost_aware = CostAwareRouter(list(self._endpoints))

    def _load_config(self, path: str) -> None:
        p = Path(path)
        raw = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # optional dependency
                cfg = yaml.safe_load(raw)
            except ImportError:
                raise ImportError("PyYAML is required to load YAML configs")
        else:
            cfg = json.loads(raw)

        for ep_dict in cfg.get("endpoints", []):
            self._endpoints.append(ModelEndpoint(**ep_dict))
        self._rebuild_routers()

    # ---- routing ----------------------------------------------------------

    def route(self, prompt: str, strategy: str = "cascade") -> RoutingResult:
        self._ensure_endpoints()
        if strategy == "cascade":
            return self._cascade.route(prompt)  # type: ignore[union-attr]
        if strategy == "cost_aware":
            return self._cost_aware.route(prompt)  # type: ignore[union-attr]
        raise ValueError(f"Unknown routing strategy: {strategy!r}")

    def batch_route(self, prompts: Sequence[str], strategy: str = "cascade") -> List[RoutingResult]:
        return [self.route(p, strategy=strategy) for p in prompts]

    def generate(
        self,
        prompt: str,
        strategy: str = "cascade",
        max_tokens: int = 256,
        **sampling: Any,
    ) -> RoutingResult:
        result = self.route(prompt, strategy=strategy)
        result.tokens = min(result.tokens, max_tokens)
        result.text = f"[generated by {result.selected_model}]"
        return result

    def get_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if self._cascade is not None:
            stats["cascade"] = self._cascade.get_routing_stats()
        return stats

    def _ensure_endpoints(self) -> None:
        if not self._endpoints:
            raise RuntimeError("No model endpoints registered. Call add_endpoint() first.")


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------

def route_inference(
    model_paths: List[str],
    prompt: str,
    strategy: str = "cascade",
    **kwargs: Any,
) -> RoutingResult:
    """CLI / script entry point: build a router from model paths and run inference."""
    router = ModelRouter()
    for i, mp in enumerate(model_paths):
        ep = ModelEndpoint(
            name=Path(mp).stem,
            model_path=mp,
            cost_per_1k_tokens=0.001 * (i + 1),
            quality_score=0.5 + 0.1 * i,
        )
        router.add_endpoint(ep)
    return router.generate(prompt, strategy=strategy, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)
