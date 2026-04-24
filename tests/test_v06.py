"""Tests for ISAT v0.6.0 -- real-time industry features."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_tiny_model():
    """Create a tiny ONNX model for testing."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tempfile.mktemp(suffix=".onnx")
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    return path


# ── Tracing ───────────────────────────────────────────────────────

class TestTracing:
    def test_trace_inference(self):
        onnx = pytest.importorskip("onnx")
        from isat.tracing.tracer import InferenceTracer
        path = _make_tiny_model()
        try:
            tracer = InferenceTracer()
            traces = tracer.trace_inference(path, runs=3)
            assert len(traces) == 3
            for t in traces:
                assert len(t.spans) >= 3
                assert t.total_ms > 0
        finally:
            os.unlink(path)

    def test_export_otlp(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        from isat.tracing.tracer import InferenceTracer
        path = _make_tiny_model()
        try:
            tracer = InferenceTracer()
            tracer.trace_inference(path, runs=2)
            out = str(tmp_path / "traces.json")
            result = tracer.export_otlp_json(out)
            assert Path(result).exists()
            data = json.loads(Path(result).read_text())
            assert "resourceSpans" in data
        finally:
            os.unlink(path)

    def test_stats(self):
        onnx = pytest.importorskip("onnx")
        from isat.tracing.tracer import InferenceTracer
        path = _make_tiny_model()
        try:
            tracer = InferenceTracer()
            tracer.trace_inference(path, runs=5)
            stats = tracer.get_stats()
            assert stats["total_traces"] == 5
            assert stats["e2e_mean_ms"] > 0
        finally:
            os.unlink(path)


# ── Canary ────────────────────────────────────────────────────────

class TestCanary:
    def test_canary_same_model(self):
        onnx = pytest.importorskip("onnx")
        from isat.canary.deployer import CanaryDeployer
        path = _make_tiny_model()
        try:
            deployer = CanaryDeployer(path, path, requests_per_phase=10)
            result = deployer.deploy()
            assert not result.rolled_back
            assert result.total_requests > 0
            text = result.summary()
            assert "PASSED" in text
        finally:
            os.unlink(path)


# ── Alerts ────────────────────────────────────────────────────────

class TestAlerts:
    def test_no_violations(self):
        from isat.alerts.engine import AlertEngine
        engine = AlertEngine()
        alerts = engine.check({"p99_ms": 100, "error_rate": 0.001, "gpu_temp_c": 50})
        assert len(alerts) == 0

    def test_violation_triggers(self):
        from isat.alerts.engine import AlertEngine, AlertRule
        engine = AlertEngine(rules=[
            AlertRule("test_high_lat", "p99_ms", ">", 100, "critical", consecutive=1),
        ])
        alerts = engine.check({"p99_ms": 500})
        assert len(alerts) == 1
        assert alerts[0].rule.name == "test_high_lat"

    def test_consecutive_requirement(self):
        from isat.alerts.engine import AlertEngine, AlertRule
        engine = AlertEngine(rules=[
            AlertRule("slow", "p99_ms", ">", 100, "warning", consecutive=3),
        ])
        engine.check({"p99_ms": 500})
        engine.check({"p99_ms": 500})
        alerts = engine.check({"p99_ms": 500})
        assert len(alerts) == 1

    def test_status_summary(self):
        from isat.alerts.engine import AlertEngine
        engine = AlertEngine()
        engine.check({"p99_ms": 100})
        status = engine.status()
        text = status.summary()
        assert "Rules loaded" in text

    def test_export_load(self, tmp_path):
        from isat.alerts.engine import AlertEngine
        engine = AlertEngine()
        path = str(tmp_path / "rules.json")
        engine.export_rules(path)
        loaded = AlertEngine.load_rules(path)
        assert len(loaded.rules) == len(engine.rules)


# ── Surgery ───────────────────────────────────────────────────────

class TestSurgery:
    def test_remove_identity(self):
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        from isat.surgery.graph import GraphSurgeon

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        n1 = helper.make_node("Identity", ["X"], ["mid"])
        n2 = helper.make_node("Relu", ["mid"], ["Y"])
        graph = helper.make_graph([n1, n2], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            surgeon = GraphSurgeon(path)
            count = surgeon.remove_op_type("Identity")
            assert count == 1
            stats = surgeon.get_stats()
            assert stats["nodes"] == 1
        finally:
            os.unlink(path)

    def test_rename_input(self):
        onnx = pytest.importorskip("onnx")
        from onnx import TensorProto, helper
        from isat.surgery.graph import GraphSurgeon

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model.SerializeToString())
            path = f.name

        try:
            surgeon = GraphSurgeon(path)
            surgeon.rename_input("X", "input_tensor")
            stats = surgeon.get_stats()
            assert "input_tensor" in stats["inputs"]
        finally:
            os.unlink(path)


# ── Inference Cache ───────────────────────────────────────────────

class TestInferenceCache:
    def test_hit_and_miss(self):
        from isat.inference_cache.cache import InferenceCache
        cache = InferenceCache(max_memory_entries=10)
        inp = {"X": np.array([[1.0, 2.0, 3.0]])}
        out = [np.array([[4.0, 5.0, 6.0]])]

        assert cache.get(inp) is None

        cache.put(inp, out, inference_ms=5.0)

        result = cache.get(inp)
        assert result is not None
        np.testing.assert_array_equal(result[0], out[0])

    def test_eviction(self):
        from isat.inference_cache.cache import InferenceCache
        cache = InferenceCache(max_memory_entries=2)
        for i in range(5):
            inp = {"X": np.array([[float(i)]])}
            cache.put(inp, [np.array([[float(i)]])])
        stats = cache.get_stats()
        assert stats.cache_size <= 2
        assert stats.evictions >= 3

    def test_invalidate(self):
        from isat.inference_cache.cache import InferenceCache
        cache = InferenceCache()
        inp = {"X": np.array([[1.0]])}
        cache.put(inp, [np.array([[2.0]])])
        cache.invalidate()
        assert cache.get(inp) is None

    def test_disk_cache(self, tmp_path):
        from isat.inference_cache.cache import InferenceCache
        cache = InferenceCache(disk_cache_dir=str(tmp_path / "disk_cache"))
        inp = {"X": np.array([[1.0, 2.0]])}
        out = [np.array([[3.0, 4.0]])]
        cache.put(inp, out)
        stats = cache.get_stats()
        assert stats.disk_entries == 1


# ── Input Guard ───────────────────────────────────────────────────

class TestInputGuard:
    def test_valid_inputs(self):
        from isat.guard.validator import InputGuard, InputSchema
        guard = InputGuard(schemas=[
            InputSchema("X", [1, 4], "float32"),
        ])
        result = guard.validate({"X": np.ones((1, 4), dtype=np.float32)})
        assert result.valid

    def test_wrong_shape(self):
        from isat.guard.validator import InputGuard, InputSchema
        guard = InputGuard(schemas=[
            InputSchema("X", [1, 4], "float32"),
        ])
        result = guard.validate({"X": np.ones((1, 8), dtype=np.float32)})
        assert not result.valid

    def test_nan_detection(self):
        from isat.guard.validator import InputGuard, InputSchema
        guard = InputGuard(schemas=[
            InputSchema("X", [1, 4], "float32"),
        ])
        arr = np.array([[1.0, float("nan"), 3.0, 4.0]], dtype=np.float32)
        result = guard.validate({"X": arr})
        assert not result.valid
        assert any("NaN" in i.message for i in result.issues)

    def test_missing_input(self):
        from isat.guard.validator import InputGuard, InputSchema
        guard = InputGuard(schemas=[
            InputSchema("X", [1, 4], "float32"),
        ])
        result = guard.validate({})
        assert not result.valid

    def test_from_model(self):
        onnx = pytest.importorskip("onnx")
        from isat.guard.validator import InputGuard
        path = _make_tiny_model()
        try:
            guard = InputGuard(model_path=path)
            assert len(guard.schemas) == 1
            assert guard.schemas[0].name == "X"
        finally:
            os.unlink(path)


# ── Ensemble ──────────────────────────────────────────────────────

class TestEnsemble:
    def test_average_ensemble(self):
        onnx = pytest.importorskip("onnx")
        from isat.ensemble.runner import ModelEnsemble
        path = _make_tiny_model()
        try:
            ensemble = ModelEnsemble(
                [("m1", path, 1.0), ("m2", path, 1.0)],
                strategy="average",
            )
            result = ensemble.run(runs=3)
            assert len(result.members) == 2
            assert result.total_ms > 0
            assert result.aggregated_output is not None
        finally:
            os.unlink(path)


# ── GPU Fragmentation ────────────────────────────────────────────

class TestGPUFrag:
    def test_report_structure(self):
        from isat.gpu_frag.analyzer import FragmentationReport
        report = FragmentationReport(
            model_path="test.onnx", vram_peak_mb=500, vram_min_mb=400,
            vram_final_mb=450, gtt_peak_mb=100, fragmentation_index=0.15,
            allocation_pattern="minor_fluctuation",
            recommendation="Monitor but not critical",
            duration_s=5.0, inference_count=100,
        )
        text = report.summary()
        assert "minor_fluctuation" in text
        assert "0.15" in text


# ── CLI Integration ──────────────────────────────────────────────

class TestCLIv06:
    def test_trace_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["trace", "--help"])

    def test_canary_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["canary", "--help"])

    def test_alerts_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["alerts", "--help"])

    def test_surgery_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["surgery", "--help"])

    def test_guard_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["guard", "--help"])

    def test_ensemble_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["ensemble", "--help"])

    def test_gpu_frag_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["gpu-frag", "--help"])

    def test_alerts_list(self):
        from isat.cli import main
        ret = main(["alerts", "list"])
        assert ret == 0

    def test_version_updated(self):
        from isat import __version__
        assert __version__ == "0.6.0"

    def test_total_commands(self):
        from isat.cli import main
        import io
        import contextlib
        f = io.StringIO()
        with pytest.raises(SystemExit):
            main(["--help"])
