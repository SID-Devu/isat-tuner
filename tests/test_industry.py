"""Tests for v0.3.0 industry-grade modules."""

import json
import os
from pathlib import Path

import pytest


class TestCompatScanner:
    def test_scan_runs(self):
        from isat.compat.scanner import CompatScanner
        scanner = CompatScanner()
        report = scanner.scan()
        assert len(report.checks) >= 5
        assert report.passed > 0

    def test_python_check_passes(self):
        from isat.compat.scanner import CompatScanner
        scanner = CompatScanner()
        result = scanner._check_python()
        assert result.status in ("ok", "warning")

    def test_numpy_check(self):
        from isat.compat.scanner import CompatScanner
        scanner = CompatScanner()
        result = scanner._check_numpy()
        assert result.status == "ok"

    def test_report_summary(self):
        from isat.compat.scanner import CompatScanner
        scanner = CompatScanner()
        report = scanner.scan()
        summary = report.summary()
        assert "PASS" in summary or "WARN" in summary or "FAIL" in summary


class TestCostEstimator:
    def test_basic_estimate(self):
        from isat.cost.estimator import CostEstimator
        est = CostEstimator()
        result = est.estimate(latency_ms=10.0, gpu_type="a10g")
        assert result.cost_per_inference > 0
        assert result.cost_per_1m > 0
        assert result.throughput_rps == pytest.approx(100.0, rel=0.01)
        assert len(result.monthly_cost_at_qps) == 4

    def test_unknown_gpu_raises(self):
        from isat.cost.estimator import CostEstimator
        est = CostEstimator()
        with pytest.raises(ValueError, match="Unknown GPU"):
            est.estimate(10.0, gpu_type="nonexistent_gpu")

    def test_compare_optimization(self):
        from isat.cost.estimator import CostEstimator
        est = CostEstimator()
        result = est.compare_optimization(
            before_latency_ms=20.0, after_latency_ms=10.0,
            gpu_type="t4", monthly_qps=100,
        )
        assert result["speedup"] == pytest.approx(2.0)
        assert result["monthly_savings"] > 0
        assert result["savings_pct"] > 0

    def test_list_gpus(self):
        from isat.cost.estimator import CostEstimator
        est = CostEstimator()
        gpus = est.list_gpus()
        assert len(gpus) >= 8
        assert all("name" in g and "hourly" in g for g in gpus)

    def test_custom_pricing(self):
        from isat.cost.estimator import CostEstimator
        est = CostEstimator(custom_pricing={"my_gpu": {"hourly": 1.00, "provider": "Custom", "instance": "x1"}})
        result = est.estimate(10.0, gpu_type="my_gpu")
        assert result.hourly_rate == 1.0


class TestSLAValidator:
    def test_passing_sla(self):
        from isat.sla.validator import SLAValidator
        validator = SLAValidator(template="batch")
        result = validator.validate({"p95_ms": 100.0, "p99_ms": 200.0, "throughput_rps": 50.0})
        assert result.all_passed

    def test_failing_sla(self):
        from isat.sla.validator import SLAValidator
        validator = SLAValidator(template="realtime")
        result = validator.validate({"p50_ms": 500.0, "p95_ms": 1000.0, "p99_ms": 2000.0})
        assert not result.all_passed
        assert result.critical_failures > 0

    def test_list_templates(self):
        from isat.sla.validator import SLAValidator
        templates = SLAValidator.list_templates()
        assert "realtime" in templates
        assert "batch" in templates
        assert "edge" in templates
        assert "llm" in templates
        assert "mobile" in templates

    def test_summary_output(self):
        from isat.sla.validator import SLAValidator
        validator = SLAValidator(template="realtime")
        result = validator.validate({"p50_ms": 5.0, "p95_ms": 15.0, "p99_ms": 40.0, "throughput_rps": 200.0})
        summary = result.summary()
        assert "PASS" in summary


class TestMigrationTool:
    def test_migraphx_to_tensorrt(self):
        from isat.migration.tool import MigrationTool
        tool = MigrationTool()
        plan = tool.plan(
            "MIGraphXExecutionProvider", "TensorrtExecutionProvider",
            current_env={"MIGRAPHX_FP16_ENABLE": "1"},
        )
        assert len(plan.steps) >= 1
        assert "ORT_TENSORRT_FP16_ENABLE" in plan.env_changes
        assert "MIGRAPHX_FP16_ENABLE" in plan.env_removes

    def test_migraphx_to_cpu(self):
        from isat.migration.tool import MigrationTool
        tool = MigrationTool()
        plan = tool.plan("MIGraphXExecutionProvider", "CPUExecutionProvider")
        assert plan.breaking_changes > 0

    def test_supported_migrations(self):
        from isat.migration.tool import MigrationTool
        migrations = MigrationTool.supported_migrations()
        assert len(migrations) >= 4


class TestNotifications:
    def test_console_notifier(self, capsys):
        from isat.notifications.notifier import Notifier, ConsoleNotifier
        n = Notifier()
        n.add_sink(ConsoleNotifier())
        n.notify("job_complete", "test_model", "Tuning finished", {"latency": "5.2ms"})
        captured = capsys.readouterr()
        assert "job_complete" in captured.out
        assert "Tuning finished" in captured.out

    def test_event_data(self):
        from isat.notifications.notifier import Event
        e = Event(type="test", timestamp=1.0, model="m", message="msg", data={"k": "v"})
        d = e.to_dict()
        assert d["type"] == "test"
        assert d["data"]["k"] == "v"


class TestCacheManager:
    def test_stats(self):
        from isat.cache.manager import CacheManager
        mgr = CacheManager()
        stats = mgr.stats()
        assert stats.total_entries >= 0
        assert stats.total_size_mb >= 0

    def test_model_hash(self, tmp_path):
        from isat.cache.manager import CacheManager
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        h = CacheManager.model_hash(str(f))
        assert len(h) == 16
        h2 = CacheManager.model_hash(str(f))
        assert h == h2


class TestHealthChecker:
    def test_basic_health(self):
        from isat.health.checker import HealthChecker
        checker = HealthChecker()
        report = checker.check()
        assert len(report.checks) >= 4

    def test_summary_output(self):
        from isat.health.checker import HealthChecker
        checker = HealthChecker()
        report = checker.check()
        summary = report.summary()
        assert "healthy" in summary or "degraded" in summary or "critical" in summary


class TestModelDiff:
    def test_diff_identical(self, tmp_path):
        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "g", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        pa = str(tmp_path / "a.onnx")
        pb = str(tmp_path / "b.onnx")
        onnx.save(model, pa)
        onnx.save(model, pb)

        from isat.diff.model_diff import ModelDiff
        result = ModelDiff().compare(pa, pb)
        assert result.identical
        assert result.nodes_a == result.nodes_b

    def test_diff_different(self, tmp_path):
        import onnx
        from onnx import TensorProto, helper

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])

        node_a = helper.make_node("Relu", ["X"], ["Y"])
        graph_a = helper.make_graph([node_a], "g", [X], [Y])
        model_a = helper.make_model(graph_a, opset_imports=[helper.make_opsetid("", 13)])

        node_b = helper.make_node("Sigmoid", ["X"], ["Y"])
        graph_b = helper.make_graph([node_b], "g", [X], [Y])
        model_b = helper.make_model(graph_b, opset_imports=[helper.make_opsetid("", 13)])

        pa = str(tmp_path / "a.onnx")
        pb = str(tmp_path / "b.onnx")
        onnx.save(model_a, pa)
        onnx.save(model_b, pb)

        from isat.diff.model_diff import ModelDiff
        result = ModelDiff().compare(pa, pb)
        assert not result.identical


class TestCLICommands:
    def test_doctor_runs(self):
        from isat.cli import main
        code = main(["doctor"])
        assert isinstance(code, int)

    def test_sla_list_templates(self):
        from isat.cli import main
        code = main(["sla", "--list-templates"])
        assert code == 0

    def test_cost_list_gpus(self):
        from isat.cli import main
        code = main(["cost", "--latency", "10", "--list-gpus"])
        assert code == 0

    def test_zoo_list(self):
        from isat.cli import main
        code = main(["zoo"])
        assert code == 0

    def test_init_generates_config(self, tmp_path):
        from isat.cli import main
        path = str(tmp_path / "test.yaml")
        code = main(["init", "--output", path])
        assert code == 0
        assert Path(path).exists()

    def test_migrate_plan(self):
        from isat.cli import main
        code = main(["migrate", "--from", "MIGraphXExecutionProvider", "--to", "CPUExecutionProvider"])
        assert code == 0

    def test_cache_stats(self):
        from isat.cli import main
        code = main(["cache", "stats"])
        assert code == 0
