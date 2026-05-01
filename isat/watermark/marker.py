"""Kirchenbauer-style AI text watermarking with multi-bit embedding and robustness analysis."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WatermarkConfig:
    gamma: float = 0.25
    delta: float = 2.0
    secret_key: str = "isat-default"
    seeding: str = "selfhash"


@dataclass
class DetectionResult:
    z_score: float
    p_value: float
    is_watermarked: bool
    green_fraction: float
    total_tokens: int
    confidence: float
    windowed_scores: Optional[List[float]] = None


@dataclass
class RobustnessReport:
    truncation_survival: Dict[str, bool]
    deletion_survival: Dict[str, bool]
    min_tokens_for_detection: int
    overall_robustness_score: float


# ---------------------------------------------------------------------------
# Logits processor – applies the watermark during generation
# ---------------------------------------------------------------------------


class WatermarkLogitsProcessor:
    def __init__(self, config: WatermarkConfig, vocab_size: int) -> None:
        self.config = config
        self.vocab_size = vocab_size

    def _get_seed(self, token_ids: List[int], position: int) -> int:
        if self.config.seeding == "selfhash":
            prev_token = token_ids[position - 1] if position > 0 else 0
            seed_str = f"{self.config.secret_key}:{prev_token}"
        elif self.config.seeding == "lefthash":
            context = token_ids[:position] if position > 0 else [0]
            context_str = ",".join(str(t) for t in context[-5:])
            seed_str = f"{self.config.secret_key}:{context_str}"
        else:
            raise ValueError(f"Unknown seeding method: {self.config.seeding}")
        digest = hashlib.sha256(seed_str.encode()).hexdigest()
        return int(digest[:8], 16)

    def _get_greenlist(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed % (2**31))
        green_size = int(self.config.gamma * self.vocab_size)
        perm = rng.permutation(self.vocab_size)
        return perm[:green_size]

    def _compute_green_mask(self, seed: int, vocab_size: int) -> np.ndarray:
        rng = np.random.RandomState(seed % (2**31))
        green_size = int(self.config.gamma * vocab_size)
        perm = rng.permutation(vocab_size)
        mask = np.zeros(vocab_size, dtype=bool)
        mask[perm[:green_size]] = True
        return mask

    def apply(
        self, logits: np.ndarray, token_ids: List[int], position: int
    ) -> np.ndarray:
        seed = self._get_seed(token_ids, position)
        green_mask = self._compute_green_mask(seed, self.vocab_size)
        modified = logits.copy()
        modified[green_mask] += self.config.delta
        return modified


# ---------------------------------------------------------------------------
# Detector – statistical test for watermark presence
# ---------------------------------------------------------------------------


class WatermarkDetector:
    def __init__(self, config: WatermarkConfig, vocab_size: int) -> None:
        self.config = config
        self.vocab_size = vocab_size
        self._processor = WatermarkLogitsProcessor(config, vocab_size)

    @staticmethod
    def _compute_z_score(
        green_count: int, total_tokens: int, gamma: float
    ) -> float:
        expected = gamma * total_tokens
        std = math.sqrt(total_tokens * gamma * (1.0 - gamma))
        if std == 0.0:
            return 0.0
        return (green_count - expected) / std

    def detect(
        self, token_ids: List[int], threshold: float = 4.0
    ) -> DetectionResult:
        total = len(token_ids)
        scored = total - 1
        if scored <= 0:
            return DetectionResult(
                z_score=0.0,
                p_value=1.0,
                is_watermarked=False,
                green_fraction=0.0,
                total_tokens=total,
                confidence=0.0,
                windowed_scores=None,
            )

        green_count = 0
        for pos in range(1, total):
            seed = self._processor._get_seed(token_ids, pos)
            mask = self._processor._compute_green_mask(seed, self.vocab_size)
            if mask[token_ids[pos]]:
                green_count += 1

        z = self._compute_z_score(green_count, scored, self.config.gamma)
        p_value = 0.5 * math.erfc(z / math.sqrt(2.0)) if z > 0 else 1.0
        green_frac = green_count / scored
        confidence = min(1.0, max(0.0, 1.0 - p_value))

        return DetectionResult(
            z_score=z,
            p_value=p_value,
            is_watermarked=(z > threshold),
            green_fraction=green_frac,
            total_tokens=total,
            confidence=confidence,
            windowed_scores=None,
        )

    def detect_windowed(
        self,
        token_ids: List[int],
        window_size: int = 50,
        threshold: float = 4.0,
    ) -> DetectionResult:
        if len(token_ids) < window_size:
            return self.detect(token_ids, threshold)

        windowed_scores: List[float] = []
        for start in range(len(token_ids) - window_size + 1):
            window = token_ids[start : start + window_size]
            res = self.detect(window, threshold)
            windowed_scores.append(res.z_score)

        full = self.detect(token_ids, threshold)
        full.windowed_scores = windowed_scores
        return full


# ---------------------------------------------------------------------------
# Multi-bit watermark – embed a payload into generated text
# ---------------------------------------------------------------------------


class MultiBitWatermark:
    def __init__(
        self,
        config: WatermarkConfig,
        vocab_size: int,
        num_bits: int = 16,
    ) -> None:
        self.config = config
        self.vocab_size = vocab_size
        self.num_bits = num_bits
        self._processor = WatermarkLogitsProcessor(config, vocab_size)
        self.repetition_factor = 3

    def encode(
        self,
        logits: np.ndarray,
        token_ids: List[int],
        position: int,
        payload_bits: str,
    ) -> np.ndarray:
        if len(payload_bits) > self.num_bits:
            raise ValueError(f"Payload exceeds {self.num_bits} bits")
        padded = payload_bits.ljust(self.num_bits, "0")

        expanded = "".join(b * self.repetition_factor for b in padded)
        bit_index = position % len(expanded)
        target_bit = int(expanded[bit_index])

        seed = self._processor._get_seed(token_ids, position)
        rng = np.random.RandomState(seed % (2**31))
        perm = rng.permutation(self.vocab_size)

        half = self.vocab_size // 2
        modified = logits.copy()
        if target_bit == 1:
            modified[perm[half:]] += self.config.delta
        else:
            modified[perm[:half]] += self.config.delta
        return modified

    def decode(self, token_ids: List[int]) -> str:
        expanded_len = self.num_bits * self.repetition_factor
        votes = np.zeros((expanded_len, 2), dtype=int)

        for pos in range(1, len(token_ids)):
            bit_index = pos % expanded_len
            seed = self._processor._get_seed(token_ids, pos)
            rng = np.random.RandomState(seed % (2**31))
            perm = rng.permutation(self.vocab_size)

            half = self.vocab_size // 2
            perm_pos = np.searchsorted(np.sort(perm[:half]), token_ids[pos])
            in_first_half = (
                perm_pos < half and np.sort(perm[:half])[perm_pos] == token_ids[pos]
            )
            if in_first_half:
                votes[bit_index, 0] += 1
            else:
                votes[bit_index, 1] += 1

        expanded_bits = "".join(
            "1" if votes[i, 1] > votes[i, 0] else "0" for i in range(expanded_len)
        )

        decoded = []
        for i in range(self.num_bits):
            chunk = expanded_bits[
                i * self.repetition_factor : (i + 1) * self.repetition_factor
            ]
            ones = chunk.count("1")
            decoded.append("1" if ones > self.repetition_factor // 2 else "0")
        return "".join(decoded)


# ---------------------------------------------------------------------------
# Robustness analyzer – stress-test watermark survival
# ---------------------------------------------------------------------------


class RobustnessAnalyzer:
    def __init__(self, detector: WatermarkDetector) -> None:
        self.detector = detector

    def _test_truncation(
        self,
        token_ids: List[int],
        fractions: Tuple[float, ...] = (0.90, 0.75, 0.50),
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for frac in fractions:
            length = max(2, int(len(token_ids) * frac))
            head = token_ids[:length]
            results[f"first_{int(frac * 100)}%"] = self.detector.detect(
                head
            ).is_watermarked
            tail = token_ids[-length:]
            results[f"last_{int(frac * 100)}%"] = self.detector.detect(
                tail
            ).is_watermarked
        return results

    def _test_deletion(
        self,
        token_ids: List[int],
        rates: Tuple[float, ...] = (0.05, 0.10, 0.20),
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        rng = np.random.RandomState(42)
        for rate in rates:
            n_delete = max(1, int(len(token_ids) * rate))
            delete_idx = set(
                rng.choice(len(token_ids), size=n_delete, replace=False).tolist()
            )
            remaining = [t for i, t in enumerate(token_ids) if i not in delete_idx]
            if len(remaining) < 2:
                results[f"delete_{int(rate * 100)}%"] = False
                continue
            results[f"delete_{int(rate * 100)}%"] = self.detector.detect(
                remaining
            ).is_watermarked
        return results

    def _test_synonym_substitution(
        self, token_ids: List[int], rate: float = 0.10
    ) -> bool:
        rng = np.random.RandomState(42)
        modified = list(token_ids)
        n_subs = max(1, int(len(modified) * rate))
        positions = rng.choice(len(modified), size=n_subs, replace=False)
        for pos in positions:
            modified[pos] = (modified[pos] + 1) % self.detector.vocab_size
        return self.detector.detect(modified).is_watermarked

    def _test_char_perturbation(
        self, token_ids: List[int], rate: float = 0.05
    ) -> bool:
        rng = np.random.RandomState(42)
        modified = list(token_ids)
        n_perturb = max(1, int(len(modified) * rate))
        positions = rng.choice(len(modified), size=n_perturb, replace=False)
        for pos in positions:
            modified[pos] = rng.randint(0, self.detector.vocab_size)
        return self.detector.detect(modified).is_watermarked

    def analyze(self, token_ids: List[int]) -> RobustnessReport:
        truncation = self._test_truncation(token_ids)
        deletion = self._test_deletion(token_ids)

        min_tokens = len(token_ids)
        for length in range(10, len(token_ids), 10):
            if self.detector.detect(token_ids[:length]).is_watermarked:
                min_tokens = length
                break

        all_survived = (
            list(truncation.values())
            + list(deletion.values())
            + [
                self._test_synonym_substitution(token_ids),
                self._test_char_perturbation(token_ids),
            ]
        )
        score = sum(all_survived) / len(all_survived) if all_survived else 0.0

        return RobustnessReport(
            truncation_survival=truncation,
            deletion_survival=deletion,
            min_tokens_for_detection=min_tokens,
            overall_robustness_score=score,
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def watermark_text(
    action: str = "detect",
    text: str = None,
    token_ids: List[int] = None,
    **kwargs: Any,
) -> Any:
    cfg_keys = set(WatermarkConfig.__dataclass_fields__)
    cfg_args = {k: v for k, v in kwargs.items() if k in cfg_keys}
    config = WatermarkConfig(**cfg_args)
    vocab_size: int = kwargs.get("vocab_size", 32000)

    if action == "apply":
        return WatermarkLogitsProcessor(config, vocab_size)

    if action == "detect":
        if token_ids is None:
            raise ValueError("token_ids required for detection")
        detector = WatermarkDetector(config, vocab_size)
        threshold: float = kwargs.get("threshold", 4.0)
        return detector.detect(token_ids, threshold)

    if action == "analyze":
        if token_ids is None:
            raise ValueError("token_ids required for analysis")
        detector = WatermarkDetector(config, vocab_size)
        return RobustnessAnalyzer(detector).analyze(token_ids)

    raise ValueError(f"Unknown action: {action!r}. Use 'apply', 'detect', or 'analyze'.")
