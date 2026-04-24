"""Bayesian optimization search engine.

Instead of brute-force Cartesian product, uses a surrogate model
(Gaussian Process or Tree-Parzen Estimator) to intelligently explore
the most promising regions of the configuration space first.

Falls back to random search if scipy is not available.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

log = logging.getLogger("isat.search.bayesian")


@dataclass
class TrialPoint:
    """A single evaluated point in the configuration space."""
    params: dict[str, Any]
    value: float = float("inf")
    duration_s: float = 0.0
    error: Optional[str] = None
    trial_id: int = 0

    @property
    def is_valid(self) -> bool:
        return self.error is None and self.value < float("inf")


@dataclass
class SearchSpace:
    """Define the tunable parameter space."""
    dimensions: dict[str, list[Any]] = field(default_factory=dict)

    def add_categorical(self, name: str, choices: list[Any]) -> None:
        self.dimensions[name] = choices

    @property
    def total_combinations(self) -> int:
        if not self.dimensions:
            return 0
        n = 1
        for choices in self.dimensions.values():
            n *= len(choices)
        return n

    def random_sample(self) -> dict[str, Any]:
        return {name: random.choice(choices) for name, choices in self.dimensions.items()}

    def encode(self, params: dict[str, Any]) -> np.ndarray:
        """Encode categorical params as one-hot vector."""
        parts = []
        for name, choices in self.dimensions.items():
            val = params.get(name)
            one_hot = [1.0 if c == val else 0.0 for c in choices]
            parts.extend(one_hot)
        return np.array(parts, dtype=np.float64)

    def all_points(self) -> list[dict[str, Any]]:
        """Enumerate all points (only practical for small spaces)."""
        import itertools
        keys = list(self.dimensions.keys())
        all_vals = [self.dimensions[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*all_vals)]


class BayesianOptimizer:
    """Bayesian optimization over a categorical search space.

    Uses a simple kernel density estimator (Tree-Parzen Estimator style)
    when scipy is unavailable, or a Gaussian Process when it is.
    """

    def __init__(
        self,
        space: SearchSpace,
        *,
        n_initial: int = 5,
        max_trials: int = 50,
        explore_ratio: float = 0.25,
        seed: int = 42,
    ):
        self.space = space
        self.n_initial = min(n_initial, space.total_combinations)
        self.max_trials = min(max_trials, space.total_combinations)
        self.explore_ratio = explore_ratio
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.trials: list[TrialPoint] = []
        self._trial_counter = 0
        self._seen: set[str] = set()

        self._has_scipy = False
        try:
            from scipy.stats import norm
            from scipy.spatial.distance import cdist
            self._has_scipy = True
        except ImportError:
            pass

    @property
    def best_trial(self) -> Optional[TrialPoint]:
        valid = [t for t in self.trials if t.is_valid]
        return min(valid, key=lambda t: t.value) if valid else None

    def _param_hash(self, params: dict[str, Any]) -> str:
        key = str(sorted(params.items()))
        return hashlib.md5(key.encode()).hexdigest()

    def suggest(self) -> dict[str, Any]:
        """Suggest the next configuration to evaluate."""
        if len(self.trials) < self.n_initial:
            return self._suggest_initial()

        if self.rng.random() < self.explore_ratio:
            return self._suggest_random()

        if self._has_scipy and len(self.trials) >= 3:
            return self._suggest_gp()

        return self._suggest_tpe()

    def _suggest_initial(self) -> dict[str, Any]:
        """Latin hypercube-style initial exploration."""
        for _ in range(100):
            params = self.space.random_sample()
            h = self._param_hash(params)
            if h not in self._seen:
                self._seen.add(h)
                return params
        return self.space.random_sample()

    def _suggest_random(self) -> dict[str, Any]:
        for _ in range(100):
            params = self.space.random_sample()
            h = self._param_hash(params)
            if h not in self._seen:
                self._seen.add(h)
                return params
        return self.space.random_sample()

    def _suggest_tpe(self) -> dict[str, Any]:
        """Tree-Parzen Estimator: sample from 'good' distribution, avoid 'bad'."""
        valid = [t for t in self.trials if t.is_valid]
        if len(valid) < 2:
            return self._suggest_random()

        valid.sort(key=lambda t: t.value)
        split = max(1, len(valid) // 4)
        good = valid[:split]
        bad = valid[split:]

        best_score = -float("inf")
        best_params = self.space.random_sample()

        for _ in range(50):
            source = self.rng.choice(good)
            candidate = dict(source.params)

            for name, choices in self.space.dimensions.items():
                if self.rng.random() < 0.3:
                    candidate[name] = self.rng.choice(choices)

            h = self._param_hash(candidate)
            if h in self._seen:
                continue

            g_score = sum(1 for t in good if t.params == candidate) + 0.1
            b_score = sum(1 for t in bad if t.params == candidate) + 0.1
            score = g_score / b_score

            if score > best_score:
                best_score = score
                best_params = candidate

        self._seen.add(self._param_hash(best_params))
        return best_params

    def _suggest_gp(self) -> dict[str, Any]:
        """Gaussian Process-based suggestion using Expected Improvement."""
        from scipy.spatial.distance import cdist
        from scipy.stats import norm

        valid = [t for t in self.trials if t.is_valid]
        X = np.array([self.space.encode(t.params) for t in valid])
        y = np.array([t.value for t in valid])

        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_norm = (y - y_mean) / y_std

        f_best = np.min(y_norm)

        all_pts = self.space.all_points()
        if len(all_pts) > 500:
            sampled = [self.space.random_sample() for _ in range(200)]
            all_pts = sampled

        best_ei = -float("inf")
        best_params = self.space.random_sample()

        for candidate in all_pts:
            h = self._param_hash(candidate)
            if h in self._seen:
                continue

            x_new = self.space.encode(candidate).reshape(1, -1)
            dists = cdist(x_new, X, metric="hamming")[0]

            length_scale = 0.5
            K = np.exp(-dists**2 / (2 * length_scale**2))

            mu = np.dot(K, y_norm) / (np.sum(K) + 1e-8)
            sigma = max(0.01, 1.0 - np.max(K))

            z = (f_best - mu) / sigma
            ei = sigma * (z * norm.cdf(z) + norm.pdf(z))

            if ei > best_ei:
                best_ei = ei
                best_params = candidate

        self._seen.add(self._param_hash(best_params))
        return best_params

    def observe(self, params: dict[str, Any], value: float, error: Optional[str] = None) -> TrialPoint:
        """Record an observation."""
        self._trial_counter += 1
        trial = TrialPoint(
            params=params,
            value=value if error is None else float("inf"),
            trial_id=self._trial_counter,
            error=error,
        )
        self.trials.append(trial)
        return trial

    def should_stop(self) -> bool:
        """Early stopping: stop if no improvement in last N trials."""
        if len(self.trials) >= self.max_trials:
            return True

        if len(self.trials) >= self.space.total_combinations:
            return True

        valid = [t for t in self.trials if t.is_valid]
        if len(valid) < self.n_initial + 5:
            return False

        recent = valid[-5:]
        best_overall = min(t.value for t in valid)
        best_recent = min(t.value for t in recent)
        if best_recent > best_overall * 0.999:
            no_improve_count = sum(
                1 for t in valid[-10:]
                if t.value > best_overall * 1.01
            )
            if no_improve_count >= 8:
                log.info("Early stopping: no improvement in last 10 trials")
                return True

        return False

    def summary(self) -> str:
        best = self.best_trial
        valid = [t for t in self.trials if t.is_valid]
        lines = [
            f"Bayesian Optimization Summary:",
            f"  Total trials : {len(self.trials)}",
            f"  Valid trials : {len(valid)}",
            f"  Failed       : {len(self.trials) - len(valid)}",
            f"  Search space : {self.space.total_combinations} combinations",
            f"  Coverage     : {len(self.trials) / max(1, self.space.total_combinations):.1%}",
        ]
        if best:
            lines.append(f"  Best value   : {best.value:.3f}")
            lines.append(f"  Best params  : {best.params}")
        return "\n".join(lines)
