"""Tests for v0.4.0 features."""

import json
from pathlib import Path
import pytest


def _make_tiny_model(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Relu", ["X"], ["Y"])
    graph = helper.make_graph([node], "g", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = str(tmp_path / "tiny.onnx")
    onnx.save(model, path)
    return path


class TestDynamicShapes:
    def test_detect_dynamic(self, tmp_path):
        import onnx
        from onnx import TensorProto, helper
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, ["batch", 3, 224, 224])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["batch", 1000])
        node = helper.make_node("Relu", ["X"], ["Y"])
        graph = helper.make_graph([node], "g", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        path = str(tmp_path / "dynamic.onnx")
        onnx.save(model, path)

        from isat.shapes.handler import DynamicShapeHandler
        handler = DynamicShapeHandler(path)
        dims = handler.detect_dynamic_dims()
        assert "X" in dims
        assert any("batch" in d for d in dims["X"])


class TestModelHub:
    def test_list_available(self):
        from isat.hub.downloader import ModelHub
        hub = ModelHub()
        models = hub.list_available()
        assert len(models) >= 5
        names = [m["name"] for m in models]
        assert "resnet50" in names
        assert "mobilenetv2" in names

    def test_unknown_model_raises(self):
        from isat.hub.downloader import ModelHub
        hub = ModelHub()
        with pytest.raises(ValueError, match="Unknown model"):
            hub.download("totally_nonexistent_model_xyz_123")


class TestMemoryPlanner:
    def test_plan_tiny_model(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.memory.planner import MemoryPlanner
        planner = MemoryPlanner(path)
        plan = planner.plan()
        assert "FP32" in plan.estimates
        assert "FP16" in plan.estimates
        assert "INT8" in plan.estimates
        assert plan.oom_risk in ("low", "medium", "high", "unknown")

    def test_summary_output(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.memory.planner import MemoryPlanner
        plan = MemoryPlanner(path).plan()
        summary = plan.summary()
        assert "FP32" in summary
        assert "Parameters" in summary


class TestGraphVisualizer:
    def test_ascii(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.visualizer.graph import GraphVisualizer
        viz = GraphVisualizer(path)
        output = viz.to_ascii()
        assert "Relu" in output
        assert "INPUTS:" in output

    def test_dot(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.visualizer.graph import GraphVisualizer
        viz = GraphVisualizer(path)
        dot = viz.to_dot()
        assert "digraph ONNX" in dot
        assert "Relu" in dot

    def test_histogram(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.visualizer.graph import GraphVisualizer
        viz = GraphVisualizer(path)
        hist = viz.op_histogram()
        assert "Relu" in hist
        assert "█" in hist


class TestEnvSnapshot:
    def test_capture(self):
        from isat.snapshot.capture import EnvSnapshot
        snap = EnvSnapshot()
        data = snap.capture()
        assert "system" in data
        assert "python" in data
        assert "software" in data
        assert "isat_version" in data

    def test_save_and_load(self, tmp_path):
        from isat.snapshot.capture import EnvSnapshot
        snap = EnvSnapshot()
        data = snap.capture()
        path = str(tmp_path / "snapshot.json")
        snap.save(data, path)
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text())
        assert loaded["python"]["version"] == data["python"]["version"]

    def test_diff(self):
        from isat.snapshot.capture import EnvSnapshot
        snap = EnvSnapshot()
        a = {"system": {"os": "Linux"}, "python": {"version": "3.12.0"}}
        b = {"system": {"os": "Linux"}, "python": {"version": "3.12.3"}}
        diffs = snap.diff(a, b)
        assert any("python.version" in d for d in diffs)

    def test_model_hash(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.snapshot.capture import EnvSnapshot
        data = EnvSnapshot().capture(model_path=path)
        assert "model" in data
        assert len(data["model"]["sha256"]) == 64


class TestBatchScheduler:
    def test_profile_tiny(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.scheduler.batch import BatchScheduler
        scheduler = BatchScheduler(path, batch_sizes=[1, 2, 4])
        profile = scheduler.profile()
        assert len(profile.results) >= 1
        assert profile.recommended_latency >= 1
        assert profile.recommended_throughput >= 1


class TestCLINewCommands:
    def test_download_list(self):
        from isat.cli import main
        code = main(["download", "mobilenetv2", "--list"])
        assert code == 0

    def test_snapshot(self, tmp_path):
        from isat.cli import main
        out = str(tmp_path / "snap.json")
        code = main(["snapshot", "--output", out])
        assert code == 0
        assert Path(out).exists()

    def test_memory_command(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.cli import main
        code = main(["memory", path])
        assert code == 0

    def test_visualize_ascii(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.cli import main
        code = main(["visualize", path, "--format", "ascii"])
        assert code == 0

    def test_visualize_histogram(self, tmp_path):
        path = _make_tiny_model(tmp_path)
        from isat.cli import main
        code = main(["visualize", path, "--format", "histogram"])
        assert code == 0
