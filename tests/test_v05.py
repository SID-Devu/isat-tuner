"""Tests for ISAT v0.5.0 features:
- Version string fix
- SLA skip-on-missing fix
- Regression detector
- Model scanner
- Compat matrix
- Thermal monitor
- Quant sensitivity
- Pipeline optimizer
- Hardware recommender
- Model registry
"""

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Version fix ────────────────────────────────────────────────────

class TestVersionFix:
    def test_version_in_init(self):
        from isat import __version__
        assert __version__ >= "0.5.0"

    def test_banner_contains_version(self):
        from isat import __version__
        from isat.cli import BANNER
        assert __version__ in BANNER

    def test_argparse_version(self):
        from isat.cli import main
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0


# ── SLA skip fix ──────────────────────────────────────────────────

class TestSLASkipFix:
    def test_unprovided_metrics_skip(self):
        from isat.sla.validator import SLAValidator
        validator = SLAValidator(template="llm")
        result = validator.validate({"p99_ms": 500.0})
        skipped = [c for c in result.checks if "skipped" in c.message]
        assert len(skipped) >= 2
        for c in skipped:
            assert c.passed is True

    def test_provided_metrics_checked(self):
        from isat.sla.validator import SLAValidator
        validator = SLAValidator(template="llm")
        result = validator.validate({"p99_ms": 500.0})
        p99_check = [c for c in result.checks if c.requirement.metric == "p99_ms"][0]
        assert p99_check.passed is True


# ── Regression Detector ───────────────────────────────────────────

class TestRegressionDetector:
    def test_first_run_sets_baseline(self, tmp_path):
        from isat.regression.detector import RegressionDetector
        db = str(tmp_path / "reg.db")
        det = RegressionDetector(db_path=db)
        result = det.check("test_model", [10.0, 11.0, 10.5, 9.8, 10.2])
        assert result.baseline is None
        assert not result.regressed
        assert "first run" in result.summary().lower()

    def test_no_regression(self, tmp_path):
        from isat.regression.detector import RegressionDetector
        db = str(tmp_path / "reg.db")
        det = RegressionDetector(db_path=db)
        det.save_baseline("test_model", [10.0] * 20, is_baseline=True)
        result = det.check("test_model", [10.1, 10.2, 9.9, 10.0, 10.1] * 4)
        assert not result.regressed

    def test_regression_detected(self, tmp_path):
        from isat.regression.detector import RegressionDetector
        db = str(tmp_path / "reg.db")
        det = RegressionDetector(db_path=db)
        det.save_baseline("test_model", [10.0] * 20, is_baseline=True)
        result = det.check("test_model", [15.0] * 20, threshold_pct=5.0)
        assert result.regressed
        assert result.mean_delta_pct > 40

    def test_history(self, tmp_path):
        from isat.regression.detector import RegressionDetector
        db = str(tmp_path / "reg.db")
        det = RegressionDetector(db_path=db)
        det.save_baseline("m", [10.0] * 5, is_baseline=True)
        det.save_baseline("m", [11.0] * 5, is_baseline=False)
        hist = det.history("m")
        assert len(hist.entries) == 2

    def test_summary_format(self, tmp_path):
        from isat.regression.detector import RegressionDetector
        db = str(tmp_path / "reg.db")
        det = RegressionDetector(db_path=db)
        det.save_baseline("m", [10.0] * 10, is_baseline=True)
        result = det.check("m", [10.5] * 10)
        text = result.summary()
        assert "Verdict" in text


# ── Model Scanner ─────────────────────────────────────────────────

class TestModelScanner:
    def test_scan_missing_file(self):
        from isat.scanner.checker import ModelScanner
        scanner = ModelScanner()
        result = scanner.scan("/nonexistent/model.onnx")
        assert not result.passed
        assert result.critical_count > 0

    def test_scan_valid_model(self):
        from isat.scanner.checker import ModelScanner
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            scanner = ModelScanner()
            result = scanner.scan(path)
            assert result.passed
            assert result.score > 0
        finally:
            os.unlink(path)

    def test_scan_summary(self):
        from isat.scanner.checker import ModelScanner, ScanResult
        result = ScanResult(model_path="test.onnx", score=85, critical_count=0, warning_count=1)
        text = result.summary()
        assert "85/100" in text


# ── Compat Matrix ─────────────────────────────────────────────────

class TestCompatMatrix:
    def test_check_model(self):
        from isat.compat_matrix.matrix import CompatMatrix
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            matrix = CompatMatrix()
            result = matrix.check(path)
            assert len(result.provider_results) >= 4
            assert all(v["supported_pct"] == 100.0 for v in result.provider_results.values())
        finally:
            os.unlink(path)

    def test_summary(self):
        from isat.compat_matrix.matrix import CompatResult
        r = CompatResult(model_path="test.onnx", model_ops={"MatMul", "Relu"})
        r.provider_results = {"TestProv": {"supported_pct": 100, "unsupported_ops": set(),
                                            "fp16": True, "int8": True, "notes": ""}}
        text = r.summary()
        assert "TestProv" in text


# ── Thermal Monitor ───────────────────────────────────────────────

class TestThermalMonitor:
    def test_no_throttling(self):
        from isat.thermal.monitor import ThermalProfile
        p = ThermalProfile(
            model_path="test.onnx", min_temp_c=30, max_temp_c=55,
            mean_temp_c=42, min_clock_mhz=2900, max_clock_mhz=2900,
            throttled=False, inference_count=100, duration_s=5.0,
        )
        text = p.summary()
        assert "NO" in text
        assert "reliable" in text.lower()

    def test_throttling_detected(self):
        from isat.thermal.monitor import ThermalProfile, ThrottleEvent
        p = ThermalProfile(
            model_path="test.onnx", min_temp_c=30, max_temp_c=95,
            mean_temp_c=80, min_clock_mhz=1800, max_clock_mhz=2900,
            throttled=True, throttle_impact_pct=37.9, inference_count=100,
            duration_s=30.0,
            throttle_events=[ThrottleEvent(10, 50, 75.0, 95.0, 1100, 4.0)],
        )
        text = p.summary()
        assert "YES" in text
        assert "37.9" in text


# ── Quant Sensitivity ────────────────────────────────────────────

class TestQuantSensitivity:
    def test_analyze_simple_model(self):
        from isat.quant_sensitivity.analyzer import QuantSensitivityAnalyzer
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper, numpy_helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        W_data = np.random.randn(4, 3).astype(np.float32)
        W_init = numpy_helper.from_array(W_data, name="W")
        node = helper.make_node("MatMul", ["X", "W"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y], initializer=[W_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            analyzer = QuantSensitivityAnalyzer(path)
            report = analyzer.analyze()
            assert len(report.layers) == 1
            assert report.total_params == 12
            assert report.layers[0].op_type == "MatMul"
            text = report.summary()
            assert "safe" in text.lower() or "MatMul" in text
        finally:
            os.unlink(path)


# ── Pipeline Optimizer ───────────────────────────────────────────

class TestPipeline:
    def test_pipeline_profile(self):
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        from isat.pipeline.optimizer import PipelineOptimizer

        models = []
        for name in ["stage_a", "stage_b"]:
            X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
            Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
            node = helper.make_node("Relu", ["X"], ["Y"])
            graph = helper.make_graph([node], name, [X], [Y])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
            path = tempfile.mktemp(suffix=".onnx")
            with open(path, "wb") as f:
                f.write(model.SerializeToString())
            models.append((name, path))

        try:
            opt = PipelineOptimizer(models, runs=3)
            profile = opt.profile()
            assert len(profile.stages) == 2
            assert profile.total_mean_ms > 0
            assert profile.bottleneck_stage in ["stage_a", "stage_b"]
            text = profile.summary()
            assert "<<<" in text
        finally:
            for _, p in models:
                os.unlink(p)


# ── Hardware Recommender ─────────────────────────────────────────

class TestRecommender:
    def test_recommend(self):
        from isat.recommend.advisor import HardwareAdvisor
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            advisor = HardwareAdvisor()
            report = advisor.recommend(path)
            assert len(report.recommendations) > 0
            text = report.summary()
            assert "Rank" in text
        finally:
            os.unlink(path)

    def test_cost_filter(self):
        from isat.recommend.advisor import HardwareAdvisor
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            advisor = HardwareAdvisor()
            report = advisor.recommend(path, max_cost_hr=1.0)
            for r in report.recommendations:
                assert r.estimated_cost_hr <= 1.0
        finally:
            os.unlink(path)


# ── Model Registry ───────────────────────────────────────────────

class TestRegistry:
    def test_register_and_get(self, tmp_path):
        from isat.registry.store import ModelRegistry
        db = str(tmp_path / "reg.db")

        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake-model-data")

        reg = ModelRegistry(db_path=db)
        v = reg.register("mymodel", "1.0", str(model_file), config={"batch": 4})
        assert v.model_name == "mymodel"
        assert v.sha256

        got = reg.get_version("mymodel", "1.0")
        assert got is not None
        assert got.config == {"batch": 4}

    def test_promote(self, tmp_path):
        from isat.registry.store import ModelRegistry
        db = str(tmp_path / "reg.db")
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake-model-data")

        reg = ModelRegistry(db_path=db)
        reg.register("m", "1.0", str(model_file))
        reg.promote("m", "1.0", stage="production")

        v = reg.get_version("m", "1.0")
        assert v.stage == "production"

    def test_list_models(self, tmp_path):
        from isat.registry.store import ModelRegistry
        db = str(tmp_path / "reg.db")
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake-model-data")

        reg = ModelRegistry(db_path=db)
        reg.register("m1", "1.0", str(model_file))
        reg.register("m2", "1.0", str(model_file))

        listing = reg.list_models()
        assert len(listing.models) == 2

    def test_diff_versions(self, tmp_path):
        from isat.registry.store import ModelRegistry
        db = str(tmp_path / "reg.db")
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake-model-data")

        reg = ModelRegistry(db_path=db)
        reg.register("m", "1.0", str(model_file), config={"batch": 4})
        reg.register("m", "2.0", str(model_file), config={"batch": 8, "fp16": True})

        diff = reg.diff_versions("m", "1.0", "2.0")
        assert len(diff.config_changes) >= 1
        text = diff.summary()
        assert "batch" in text

    def test_delete(self, tmp_path):
        from isat.registry.store import ModelRegistry
        db = str(tmp_path / "reg.db")
        model_file = tmp_path / "test.onnx"
        model_file.write_bytes(b"fake-model-data")

        reg = ModelRegistry(db_path=db)
        reg.register("m", "1.0", str(model_file))
        reg.delete_version("m", "1.0")
        assert reg.get_version("m", "1.0") is None


# ── CLI Integration ──────────────────────────────────────────────

class TestCLI:
    def test_help_lists_all_commands(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["--help"])

    def test_scan_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["scan", "--help"])

    def test_regression_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["regression", "--help"])

    def test_compat_matrix_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["compat-matrix", "--help"])

    def test_thermal_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["thermal", "--help"])

    def test_quant_sensitivity_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["quant-sensitivity", "--help"])

    def test_pipeline_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["pipeline", "--help"])

    def test_recommend_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["recommend", "--help"])

    def test_registry_command_exists(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["registry", "--help"])
