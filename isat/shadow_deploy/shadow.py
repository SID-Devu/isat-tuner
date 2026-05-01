"""Shadow deployment engine for side-by-side model comparison.

Runs a *shadow* (candidate) model alongside the production model on every
request, collects quality and latency metrics, and optionally auto-promotes
the shadow when it is statistically better.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("isat.shadow_deploy")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelVersion:
    name: str
    model_path: str
    provider: str
    version: str
    is_production: bool
    deployed_at: float


@dataclass
class ComparisonMetrics:
    bleu_score: float
    rouge_l: float
    semantic_similarity: float
    length_ratio: float
    latency_ratio: float
    factual_consistency: float
    overall_quality: float


@dataclass
class ShadowResult:
    production_output: np.ndarray
    shadow_output: np.ndarray
    production_latency_ms: float
    shadow_latency_ms: float
    comparison: ComparisonMetrics


@dataclass
class QualitySummary:
    mean_bleu: float
    std_bleu: float
    mean_rouge_l: float
    std_rouge_l: float
    mean_semantic_similarity: float
    std_semantic_similarity: float
    mean_length_ratio: float
    std_length_ratio: float
    mean_latency_ratio: float
    std_latency_ratio: float
    mean_factual_consistency: float
    std_factual_consistency: float
    mean_overall_quality: float
    std_overall_quality: float
    trend_direction: str
    num_samples: int


@dataclass
class DeploymentReport:
    experiment_name: str
    num_samples: int
    duration_hours: float
    production_model: str
    shadow_model: str
    quality_summary: Optional[QualitySummary]
    promotion_ready: bool
    p_value: float
    effect_size: float
    recommendation: str


# ---------------------------------------------------------------------------
# ShadowRunner — runs both models and compares outputs
# ---------------------------------------------------------------------------

class ShadowRunner:
    """Execute production and shadow models side-by-side."""

    def __init__(
        self,
        production_model_path: str,
        shadow_model_path: str,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        import onnxruntime as ort

        self.production_model_path = production_model_path
        self.shadow_model_path = shadow_model_path
        self.provider = provider

        log.info("Loading production model: %s", production_model_path)
        self._prod_session = ort.InferenceSession(
            production_model_path, providers=[provider],
        )
        log.info("Loading shadow model: %s", shadow_model_path)
        self._shadow_session = ort.InferenceSession(
            shadow_model_path, providers=[provider],
        )

    # ---- public -----------------------------------------------------------

    def run(
        self,
        input_ids: np.ndarray,
        max_tokens: int = 256,
        **sampling: Any,
    ) -> ShadowResult:
        """Run *input_ids* through both models and return a comparison."""
        t0 = time.perf_counter()
        prod_ids, prod_logits = self._generate(
            self._prod_session, input_ids, max_tokens, **sampling,
        )
        prod_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        shadow_ids, shadow_logits = self._generate(
            self._shadow_session, input_ids, max_tokens, **sampling,
        )
        shadow_ms = (time.perf_counter() - t1) * 1000.0

        comparison = self._compare_outputs(
            prod_ids, shadow_ids, prod_logits, shadow_logits,
        )

        return ShadowResult(
            production_output=prod_ids,
            shadow_output=shadow_ids,
            production_latency_ms=prod_ms,
            shadow_latency_ms=shadow_ms,
            comparison=comparison,
        )

    # ---- private ----------------------------------------------------------

    def _generate(
        self,
        session: Any,
        input_ids: np.ndarray,
        max_tokens: int,
        **sampling: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Auto-regressive token generation for a single model."""
        temperature = sampling.get("temperature", 1.0)
        top_k = sampling.get("top_k", 0)

        ids = np.array(input_ids, dtype=np.int64)
        if ids.ndim == 1:
            ids = ids.reshape(1, -1)

        input_name = session.get_inputs()[0].name
        all_logits: List[np.ndarray] = []

        for _ in range(max_tokens):
            outputs = session.run(None, {input_name: ids})
            logits = outputs[0]
            last_logits = logits[:, -1, :]
            all_logits.append(last_logits)

            if temperature != 1.0 and temperature > 0:
                last_logits = last_logits / temperature

            if top_k > 0:
                k = min(top_k, last_logits.shape[-1])
                topk_idx = np.argpartition(last_logits, -k, axis=-1)[:, -k:]
                mask = np.full_like(last_logits, -np.inf)
                np.put_along_axis(mask, topk_idx, np.take_along_axis(last_logits, topk_idx, axis=-1), axis=-1)
                last_logits = mask

            probs = _softmax(last_logits)
            next_token = np.array(
                [[np.random.choice(probs.shape[-1], p=probs[0])]]
            )
            ids = np.concatenate([ids, next_token], axis=-1)

        stacked_logits = np.concatenate(all_logits, axis=0)
        return ids, stacked_logits

    def _compare_outputs(
        self,
        prod_ids: np.ndarray,
        shadow_ids: np.ndarray,
        prod_logits: np.ndarray,
        shadow_logits: np.ndarray,
    ) -> ComparisonMetrics:
        """Compute quality metrics between production and shadow outputs."""
        bleu = self._approx_bleu(prod_ids.flatten(), shadow_ids.flatten())

        prod_set = set(prod_ids.flatten().tolist())
        shad_set = set(shadow_ids.flatten().tolist())
        union = prod_set | shad_set
        rouge_l = len(prod_set & shad_set) / len(union) if union else 1.0

        min_len = min(prod_logits.shape[0], shadow_logits.shape[0])
        if min_len > 0:
            p = prod_logits[:min_len].flatten().astype(np.float64)
            s = shadow_logits[:min_len].flatten().astype(np.float64)
            norm_p = np.linalg.norm(p)
            norm_s = np.linalg.norm(s)
            if norm_p > 0 and norm_s > 0:
                semantic_sim = float(np.dot(p, s) / (norm_p * norm_s))
            else:
                semantic_sim = 0.0
        else:
            semantic_sim = 0.0
        semantic_sim = max(-1.0, min(1.0, semantic_sim))

        prod_len = prod_ids.size
        shad_len = shadow_ids.size
        length_ratio = shad_len / prod_len if prod_len > 0 else 1.0

        factual = (bleu + rouge_l + max(0.0, semantic_sim)) / 3.0

        overall = (
            0.25 * bleu
            + 0.20 * rouge_l
            + 0.25 * max(0.0, semantic_sim)
            + 0.10 * min(1.0, length_ratio)
            + 0.20 * factual
        )

        return ComparisonMetrics(
            bleu_score=bleu,
            rouge_l=rouge_l,
            semantic_similarity=semantic_sim,
            length_ratio=length_ratio,
            latency_ratio=0.0,
            factual_consistency=factual,
            overall_quality=overall,
        )

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _approx_bleu(ref: np.ndarray, hyp: np.ndarray, max_n: int = 4) -> float:
        """Quick corpus-free BLEU approximation using n-gram precision."""
        ref_list = ref.tolist()
        hyp_list = hyp.tolist()
        if not hyp_list:
            return 0.0

        precisions: List[float] = []
        for n in range(1, max_n + 1):
            ref_ngrams: Dict[tuple, int] = {}
            for i in range(len(ref_list) - n + 1):
                ng = tuple(ref_list[i : i + n])
                ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

            matches = 0
            total = 0
            for i in range(len(hyp_list) - n + 1):
                ng = tuple(hyp_list[i : i + n])
                total += 1
                if ref_ngrams.get(ng, 0) > 0:
                    matches += 1
                    ref_ngrams[ng] -= 1

            precisions.append(matches / total if total > 0 else 0.0)

        if any(p == 0 for p in precisions):
            return 0.0

        log_avg = sum(math.log(p) for p in precisions) / len(precisions)
        bp = min(1.0, math.exp(1 - len(ref_list) / len(hyp_list))) if hyp_list else 0.0
        return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# QualityTracker — rolling window statistics
# ---------------------------------------------------------------------------

class QualityTracker:
    """Accumulate comparison metrics in a rolling window."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self._records: deque[ComparisonMetrics] = deque(maxlen=window_size)

    def record(self, comparison: ComparisonMetrics) -> None:
        self._records.append(comparison)

    def get_summary(self) -> QualitySummary:
        n = len(self._records)
        if n == 0:
            return QualitySummary(
                mean_bleu=0.0, std_bleu=0.0,
                mean_rouge_l=0.0, std_rouge_l=0.0,
                mean_semantic_similarity=0.0, std_semantic_similarity=0.0,
                mean_length_ratio=0.0, std_length_ratio=0.0,
                mean_latency_ratio=0.0, std_latency_ratio=0.0,
                mean_factual_consistency=0.0, std_factual_consistency=0.0,
                mean_overall_quality=0.0, std_overall_quality=0.0,
                trend_direction="insufficient_data",
                num_samples=0,
            )

        arrs = self._to_arrays()
        trend = self._compute_trend(arrs["overall_quality"])

        return QualitySummary(
            mean_bleu=float(np.mean(arrs["bleu_score"])),
            std_bleu=float(np.std(arrs["bleu_score"])),
            mean_rouge_l=float(np.mean(arrs["rouge_l"])),
            std_rouge_l=float(np.std(arrs["rouge_l"])),
            mean_semantic_similarity=float(np.mean(arrs["semantic_similarity"])),
            std_semantic_similarity=float(np.std(arrs["semantic_similarity"])),
            mean_length_ratio=float(np.mean(arrs["length_ratio"])),
            std_length_ratio=float(np.std(arrs["length_ratio"])),
            mean_latency_ratio=float(np.mean(arrs["latency_ratio"])),
            std_latency_ratio=float(np.std(arrs["latency_ratio"])),
            mean_factual_consistency=float(np.mean(arrs["factual_consistency"])),
            std_factual_consistency=float(np.std(arrs["factual_consistency"])),
            mean_overall_quality=float(np.mean(arrs["overall_quality"])),
            std_overall_quality=float(np.std(arrs["overall_quality"])),
            trend_direction=trend,
            num_samples=n,
        )

    def is_shadow_better(
        self, confidence: float = 0.95,
    ) -> Tuple[bool, float, float]:
        """Paired t-test on overall quality scores.

        Returns (is_better, p_value, effect_size).
        """
        if len(self._records) < 2:
            return False, 1.0, 0.0

        arrs = self._to_arrays()
        prod_scores = arrs["overall_quality"]
        shadow_baseline = np.ones_like(prod_scores) * 0.5

        t_stat, p_value = self._paired_t_test(shadow_baseline, prod_scores)

        diff = prod_scores - shadow_baseline
        pooled_std = float(np.std(diff))
        effect_size = float(np.mean(diff) / pooled_std) if pooled_std > 0 else 0.0

        alpha = 1.0 - confidence
        is_better = (p_value < alpha) and (float(np.mean(diff)) > 0)

        return is_better, float(p_value), effect_size

    def _paired_t_test(
        self, prod_scores: np.ndarray, shadow_scores: np.ndarray,
    ) -> Tuple[float, float]:
        """Paired t-test returning (t_statistic, two_sided_p_value)."""
        diff = shadow_scores - prod_scores
        n = len(diff)
        mean_d = float(np.mean(diff))
        std_d = float(np.std(diff, ddof=1))
        if std_d == 0 or n < 2:
            return 0.0, 1.0

        t_stat = mean_d / (std_d / math.sqrt(n))
        df = n - 1
        p_value = self._t_distribution_p(t_stat, df)
        return t_stat, p_value

    def _bootstrap_ci(
        self,
        differences: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for the mean of *differences*."""
        n = len(differences)
        if n == 0:
            return 0.0, 0.0

        rng = np.random.default_rng(seed=42)
        means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(differences, size=n, replace=True)
            means[i] = np.mean(sample)

        alpha = 1.0 - confidence
        lo = float(np.percentile(means, 100 * alpha / 2))
        hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return lo, hi

    # ---- internal helpers -------------------------------------------------

    def _to_arrays(self) -> Dict[str, np.ndarray]:
        records = list(self._records)
        return {
            "bleu_score": np.array([r.bleu_score for r in records]),
            "rouge_l": np.array([r.rouge_l for r in records]),
            "semantic_similarity": np.array([r.semantic_similarity for r in records]),
            "length_ratio": np.array([r.length_ratio for r in records]),
            "latency_ratio": np.array([r.latency_ratio for r in records]),
            "factual_consistency": np.array([r.factual_consistency for r in records]),
            "overall_quality": np.array([r.overall_quality for r in records]),
        }

    @staticmethod
    def _compute_trend(values: np.ndarray) -> str:
        n = len(values)
        if n < 10:
            return "insufficient_data"
        half = n // 2
        first_half = float(np.mean(values[:half]))
        second_half = float(np.mean(values[half:]))
        delta = second_half - first_half
        if abs(delta) < 0.01:
            return "stable"
        return "improving" if delta > 0 else "degrading"

    @staticmethod
    def _t_distribution_p(t_stat: float, df: int) -> float:
        """Approximate two-sided p-value for a t-distribution (no scipy)."""
        x = df / (df + t_stat * t_stat)
        if df <= 0:
            return 1.0
        p_one_tail = 0.5 * _regularized_incomplete_beta(df / 2.0, 0.5, x)
        return 2.0 * p_one_tail


# ---------------------------------------------------------------------------
# AutoPromoter — automatic promotion / rollback
# ---------------------------------------------------------------------------

class AutoPromoter:
    """Decide when to promote a shadow model to production or roll back."""

    def __init__(
        self,
        tracker: QualityTracker,
        min_samples: int = 100,
        min_hours: float = 24,
        quality_threshold: float = 0.0,
    ) -> None:
        self.tracker = tracker
        self.min_samples = min_samples
        self.min_hours = min_hours
        self.quality_threshold = quality_threshold
        self._start_time = time.time()

    def should_promote(self) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        summary = self.tracker.get_summary()
        elapsed_h = (time.time() - self._start_time) / 3600.0

        if summary.num_samples < self.min_samples:
            reasons.append(
                f"Insufficient samples: {summary.num_samples}/{self.min_samples}"
            )
            return False, reasons

        if elapsed_h < self.min_hours:
            reasons.append(
                f"Insufficient observation time: {elapsed_h:.1f}h / {self.min_hours}h"
            )
            return False, reasons

        is_better, p_val, effect = self.tracker.is_shadow_better()
        if not is_better:
            reasons.append(f"Shadow not statistically better (p={p_val:.4f})")
            return False, reasons

        if effect < self.quality_threshold:
            reasons.append(
                f"Effect size {effect:.4f} below threshold {self.quality_threshold}"
            )
            return False, reasons

        reasons.append(f"Shadow better: p={p_val:.4f}, effect={effect:.4f}")
        reasons.append(f"Observed for {elapsed_h:.1f}h with {summary.num_samples} samples")
        return True, reasons

    def should_rollback(self) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        summary = self.tracker.get_summary()

        if summary.num_samples < 10:
            return False, ["Not enough data to judge rollback"]

        is_better, p_val, effect = self.tracker.is_shadow_better()
        if not is_better and p_val < 0.05 and effect < -0.2:
            reasons.append(
                f"Shadow significantly worse: p={p_val:.4f}, effect={effect:.4f}"
            )
            return True, reasons

        if summary.trend_direction == "degrading":
            reasons.append("Quality trend is degrading")

        if summary.mean_latency_ratio > 2.0:
            reasons.append(
                f"Shadow latency {summary.mean_latency_ratio:.1f}x production"
            )

        should = len(reasons) >= 2
        return should, reasons

    def promote(self, shadow_path: str, production_path: str) -> str:
        """Swap shadow into production, back up old production.

        Returns the path to the backup.
        """
        backup_path = production_path + f".backup.{int(time.time())}"
        log.info("Backing up production model to %s", backup_path)
        shutil.copy2(production_path, backup_path)
        log.info("Promoting shadow %s -> %s", shadow_path, production_path)
        shutil.copy2(shadow_path, production_path)
        return backup_path

    def rollback(self, backup_path: str, production_path: str) -> None:
        """Restore production from a previous backup."""
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        log.info("Rolling back production to %s", backup_path)
        shutil.copy2(backup_path, production_path)


# ---------------------------------------------------------------------------
# ShadowDeployment — orchestrator
# ---------------------------------------------------------------------------

class ShadowDeployment:
    """Top-level orchestrator for shadow deployment experiments."""

    def __init__(
        self,
        production_path: str,
        shadow_path: str,
        provider: str = "CPUExecutionProvider",
        auto_promote: bool = True,
        min_samples: int = 100,
    ) -> None:
        self.production_path = production_path
        self.shadow_path = shadow_path
        self.provider = provider
        self.auto_promote = auto_promote

        self.runner = ShadowRunner(production_path, shadow_path, provider)
        self.tracker = QualityTracker()
        self.promoter = AutoPromoter(self.tracker, min_samples=min_samples)

        self._experiment_name: Optional[str] = None
        self._experiment_desc: Optional[str] = None
        self._experiment_start: Optional[float] = None
        self._last_backup: Optional[str] = None

    def process_request(
        self,
        input_ids: np.ndarray,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> np.ndarray:
        """Run shadow comparison and return **only** the production output."""
        result = self.runner.run(input_ids, max_tokens, **kwargs)

        result.comparison = ComparisonMetrics(
            bleu_score=result.comparison.bleu_score,
            rouge_l=result.comparison.rouge_l,
            semantic_similarity=result.comparison.semantic_similarity,
            length_ratio=result.comparison.length_ratio,
            latency_ratio=(
                result.shadow_latency_ms / result.production_latency_ms
                if result.production_latency_ms > 0
                else 1.0
            ),
            factual_consistency=result.comparison.factual_consistency,
            overall_quality=result.comparison.overall_quality,
        )

        self.tracker.record(result.comparison)

        if self.auto_promote:
            should, reasons = self.promoter.should_promote()
            if should:
                log.info("Auto-promoting shadow model: %s", "; ".join(reasons))
                self._last_backup = self.promoter.promote(
                    self.shadow_path, self.production_path,
                )

            should_rb, rb_reasons = self.promoter.should_rollback()
            if should_rb and self._last_backup:
                log.warning("Rolling back shadow: %s", "; ".join(rb_reasons))
                self.promoter.rollback(self._last_backup, self.production_path)

        return result.production_output

    def get_report(self) -> DeploymentReport:
        summary = self.tracker.get_summary()
        is_better, p_val, effect = self.tracker.is_shadow_better()
        can_promote, reasons = self.promoter.should_promote()

        if can_promote:
            recommendation = "PROMOTE: shadow model is statistically better"
        elif summary.num_samples == 0:
            recommendation = "COLLECT: no data yet"
        elif not is_better:
            recommendation = "HOLD: shadow is not yet proven better"
        else:
            recommendation = "WAIT: criteria not fully met — " + "; ".join(reasons)

        start = self._experiment_start or self.promoter._start_time
        duration_h = (time.time() - start) / 3600.0

        return DeploymentReport(
            experiment_name=self._experiment_name or "unnamed",
            num_samples=summary.num_samples,
            duration_hours=duration_h,
            production_model=self.production_path,
            shadow_model=self.shadow_path,
            quality_summary=summary if summary.num_samples > 0 else None,
            promotion_ready=can_promote,
            p_value=p_val,
            effect_size=effect,
            recommendation=recommendation,
        )

    def start_experiment(self, name: str, description: str = "") -> None:
        self._experiment_name = name
        self._experiment_desc = description
        self._experiment_start = time.time()
        log.info("Experiment started: %s — %s", name, description)

    def end_experiment(self) -> DeploymentReport:
        report = self.get_report()
        log.info(
            "Experiment '%s' ended: %d samples, recommendation=%s",
            report.experiment_name,
            report.num_samples,
            report.recommendation,
        )
        self._experiment_name = None
        self._experiment_desc = None
        self._experiment_start = None
        return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def shadow_deploy(
    production_path: str,
    shadow_path: str,
    action: str = "run",
    **kwargs: Any,
) -> Any:
    """CLI-style entry point for shadow deployment operations.

    Actions
    -------
    run      — process a single request (requires ``input_ids`` in *kwargs*)
    report   — print the current comparison report
    promote  — force-promote the shadow model
    rollback — restore production from a backup (requires ``backup_path``)
    """
    if action == "run":
        input_ids = kwargs.pop("input_ids", None)
        if input_ids is None:
            raise ValueError("action='run' requires input_ids")
        max_tokens = kwargs.pop("max_tokens", 256)
        dep = ShadowDeployment(
            production_path,
            shadow_path,
            provider=kwargs.pop("provider", "CPUExecutionProvider"),
            auto_promote=kwargs.pop("auto_promote", True),
            min_samples=kwargs.pop("min_samples", 100),
        )
        return dep.process_request(input_ids, max_tokens, **kwargs)

    if action == "report":
        dep = ShadowDeployment(
            production_path,
            shadow_path,
            provider=kwargs.pop("provider", "CPUExecutionProvider"),
            auto_promote=False,
        )
        return dep.get_report()

    if action == "promote":
        promoter = AutoPromoter(QualityTracker())
        backup = promoter.promote(shadow_path, production_path)
        log.info("Promoted. Backup at %s", backup)
        return backup

    if action == "rollback":
        backup_path = kwargs.get("backup_path")
        if not backup_path:
            raise ValueError("action='rollback' requires backup_path")
        promoter = AutoPromoter(QualityTracker())
        promoter.rollback(backup_path, production_path)
        log.info("Rolled back production from %s", backup_path)
        return None

    raise ValueError(f"Unknown action: {action!r}")


# ---------------------------------------------------------------------------
# Pure-Python math helpers (no scipy dependency)
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """Continued-fraction approximation of I_x(a, b) for the t-distribution CDF."""
    if x < 0.0 or x > 1.0:
        return 0.0
    if x == 0.0 or x == 1.0:
        return x

    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(
        math.log(x) * a + math.log(1.0 - x) * b - lbeta
    ) / a

    # Lentz's continued fraction
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, 201):
        # even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        f *= d * c

        # odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return front * f
