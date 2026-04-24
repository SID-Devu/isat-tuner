"""Tests for analysis modules: outliers, significance, Pareto."""

import numpy as np
import pytest


class TestOutlierDetection:
    def test_mad_detects_clear_outlier(self):
        from isat.analysis.outliers import detect_outliers
        values = [10.0, 10.1, 10.2, 10.0, 9.9, 50.0]
        report = detect_outliers(values, method="mad")
        assert report.n_outliers >= 1
        assert 50.0 in report.outlier_values

    def test_iqr_detects_outlier(self):
        from isat.analysis.outliers import detect_outliers
        values = [10.0, 10.1, 10.2, 10.0, 9.9, 50.0]
        report = detect_outliers(values, method="iqr")
        assert report.n_outliers >= 1

    def test_remove_outliers(self):
        from isat.analysis.outliers import remove_outliers
        values = [10.0, 10.1, 10.2, 10.0, 9.9, 100.0]
        cleaned, report = remove_outliers(values)
        assert len(cleaned) < len(values)
        assert 100.0 not in cleaned

    def test_no_outliers_in_uniform_data(self):
        from isat.analysis.outliers import detect_outliers
        values = [10.0] * 10
        report = detect_outliers(values)
        assert report.n_outliers == 0


class TestSignificance:
    def test_identical_distributions_not_significant(self):
        from isat.analysis.significance import compare_configs
        np.random.seed(42)
        a = list(np.random.normal(10.0, 0.1, 30))
        b = list(np.random.normal(10.0, 0.1, 30))
        result = compare_configs(a, b, confidence=0.95)
        assert not result.is_significant or result.p_value > 0.01

    def test_different_distributions_significant(self):
        from isat.analysis.significance import compare_configs
        a = list(np.random.normal(15.0, 0.5, 30))
        b = list(np.random.normal(10.0, 0.5, 30))
        result = compare_configs(a, b, confidence=0.95)
        assert result.is_significant
        assert result.speedup > 1.0

    def test_summary_text(self):
        from isat.analysis.significance import compare_configs
        result = compare_configs([10.0, 11.0, 10.5], [8.0, 8.5, 8.2])
        assert "faster" in result.summary or "slower" in result.summary


class TestPareto:
    def test_pareto_frontier_basic(self):
        from isat.analysis.pareto import ParetoFrontier
        from isat.search.engine import CandidateConfig, TuneResult
        from isat.search.memory import MemoryConfig
        from isat.search.kernel import KernelConfig
        from isat.search.precision import PrecisionConfig
        from isat.search.graph import GraphConfig

        def _make(label, mean, vram):
            config = CandidateConfig(
                memory=MemoryConfig(label=f"m_{label}"),
                kernel=KernelConfig(label=f"k_{label}"),
                precision=PrecisionConfig(label=f"p_{label}"),
                graph=GraphConfig(label=f"g_{label}"),
            )
            return TuneResult(
                config=config, mean_latency_ms=mean,
                peak_vram_mb=vram, peak_gtt_mb=0.0,
            )

        results = [
            _make("fast_big", 5.0, 8000),
            _make("slow_small", 20.0, 1000),
            _make("mid", 10.0, 4000),
            _make("dominated", 15.0, 6000),
        ]

        pareto = ParetoFrontier(results, objectives=["latency_ms", "memory_mb"])
        assert len(pareto.frontier) >= 2
        assert len(pareto.frontier) < len(results)

    def test_recommend(self):
        from isat.analysis.pareto import ParetoFrontier
        from isat.search.engine import CandidateConfig, TuneResult
        from isat.search.memory import MemoryConfig
        from isat.search.kernel import KernelConfig
        from isat.search.precision import PrecisionConfig
        from isat.search.graph import GraphConfig

        config = CandidateConfig(
            memory=MemoryConfig(label="m"),
            kernel=KernelConfig(label="k"),
            precision=PrecisionConfig(label="p"),
            graph=GraphConfig(label="g"),
        )
        results = [TuneResult(config=config, mean_latency_ms=10.0, peak_vram_mb=100, peak_gtt_mb=0)]
        pareto = ParetoFrontier(results)
        rec = pareto.recommend("latency_ms")
        assert rec is not None


class TestBayesian:
    def test_search_space(self):
        from isat.search.bayesian import SearchSpace
        space = SearchSpace()
        space.add_categorical("xnack", [0, 1])
        space.add_categorical("mlir", [True, False])
        assert space.total_combinations == 4

    def test_optimizer_flow(self):
        from isat.search.bayesian import BayesianOptimizer, SearchSpace
        space = SearchSpace()
        space.add_categorical("a", [1, 2, 3])
        space.add_categorical("b", ["x", "y"])

        opt = BayesianOptimizer(space, n_initial=3, max_trials=6)
        for _ in range(6):
            if opt.should_stop():
                break
            params = opt.suggest()
            value = params["a"] * (1 if params["b"] == "x" else 2)
            opt.observe(params, value)

        assert opt.best_trial is not None
        assert len(opt.trials) > 0


class TestProfiles:
    def test_all_profiles_exist(self):
        from isat.profiles.presets import PROFILES, get_profile
        assert len(PROFILES) >= 8
        for name in PROFILES:
            p = get_profile(name)
            assert p.warmup > 0
            assert p.runs > 0

    def test_unknown_profile_raises(self):
        from isat.profiles.presets import get_profile
        with pytest.raises(ValueError):
            get_profile("nonexistent_profile")


class TestProviders:
    def test_provider_candidates(self):
        from isat.search.provider import ProviderSearchDimension
        dim = ProviderSearchDimension()
        candidates = dim.candidates()
        assert len(candidates) >= 1
        labels = [c.label for c in candidates]
        assert any("CPU" in l for l in labels)


class TestBatch:
    def test_batch_candidates(self):
        from isat.fingerprint.hardware import HardwareFingerprint
        from isat.fingerprint.model import ModelFingerprint
        from isat.search.batch import BatchSearchDimension

        hw = HardwareFingerprint(vram_total_mb=8192)
        model = ModelFingerprint(estimated_memory_mb=100)
        dim = BatchSearchDimension(hw, model)
        candidates = dim.candidates()
        assert len(candidates) >= 1
        assert candidates[0].batch_size == 1
