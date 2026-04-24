"""Unit tests for fingerprinting modules."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_tiny_onnx(path: str) -> str:
    """Create a minimal ONNX model for testing."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 5])

    W_init = helper.make_tensor("W", TensorProto.FLOAT, [3, 5],
                                np.random.randn(15).astype(np.float32).tolist())
    B_init = helper.make_tensor("B", TensorProto.FLOAT, [5],
                                np.random.randn(5).astype(np.float32).tolist())

    matmul = helper.make_node("MatMul", ["X", "W"], ["XW"])
    add = helper.make_node("Add", ["XW", "B"], ["Y"])

    graph = helper.make_graph([matmul, add], "test", [X], [Y], [W_init, B_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    onnx.save(model, path)
    return path


class TestModelFingerprint:
    def test_analyze_tiny_model(self, tmp_path):
        model_path = str(tmp_path / "tiny.onnx")
        _make_tiny_onnx(model_path)

        from isat.fingerprint.model import fingerprint_model
        fp = fingerprint_model(model_path)

        assert fp.opset == 17
        assert fp.param_count == 20
        assert fp.num_nodes == 2
        assert "MatMul" in fp.op_histogram
        assert "Add" in fp.op_histogram
        assert fp.input_shapes["X"] == [1, 3]
        assert fp.model_class in {"transformer", "mixed"}
        assert fp.fingerprint_hash
        assert len(fp.fingerprint_hash) == 16

    def test_json_serializable(self, tmp_path):
        model_path = str(tmp_path / "tiny.onnx")
        _make_tiny_onnx(model_path)

        from isat.fingerprint.model import fingerprint_model
        fp = fingerprint_model(model_path)
        data = json.loads(fp.to_json())
        assert data["opset"] == 17

    def test_size_class(self, tmp_path):
        model_path = str(tmp_path / "tiny.onnx")
        _make_tiny_onnx(model_path)

        from isat.fingerprint.model import fingerprint_model
        fp = fingerprint_model(model_path)
        assert fp.size_class == "small"


class TestStats:
    def test_compute_stats_basic(self):
        from isat.benchmark.stats import compute_stats
        stats = compute_stats([10.0, 12.0, 11.0, 13.0, 10.5])
        assert 10.0 <= stats.mean_ms <= 13.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 13.0
        assert stats.n == 5
        assert stats.p50_ms > 0
        assert stats.p95_ms >= stats.p50_ms

    def test_empty_latencies(self):
        from isat.benchmark.stats import compute_stats
        stats = compute_stats([])
        assert stats.mean_ms == 0
        assert stats.n == 0


class TestSearchEngine:
    def test_generates_candidates(self, tmp_path):
        model_path = str(tmp_path / "tiny.onnx")
        _make_tiny_onnx(model_path)

        from isat.fingerprint.hardware import HardwareFingerprint
        from isat.fingerprint.model import fingerprint_model
        from isat.search.engine import SearchEngine

        hw = HardwareFingerprint(
            gpu_name="test", gfx_target="gfx1151",
            cu_count=8, xnack_supported=True,
        )
        model_fp = fingerprint_model(model_path)
        engine = SearchEngine(hw, model_fp, skip_precision=True, skip_graph=True)
        candidates = engine.generate_candidates()

        assert len(candidates) > 0
        labels = [c.label for c in candidates]
        assert any("xnack0" in l for l in labels)
        assert any("mlir" in l.lower() for l in labels)

    def test_pruning(self, tmp_path):
        model_path = str(tmp_path / "tiny.onnx")
        _make_tiny_onnx(model_path)

        from isat.fingerprint.hardware import HardwareFingerprint
        from isat.fingerprint.model import fingerprint_model
        from isat.search.engine import SearchEngine

        hw = HardwareFingerprint(gfx_target="gfx1151", xnack_supported=True)
        model_fp = fingerprint_model(model_path)
        engine = SearchEngine(hw, model_fp)
        candidates = engine.generate_candidates()
        labels = [c.label for c in candidates]
        for l in labels:
            assert not ("int8" in l and "mlir_off" in l)


class TestDatabase:
    def test_save_and_query(self, tmp_path):
        from isat.database.store import ResultsDB
        from isat.search.engine import CandidateConfig, TuneResult
        from isat.search.memory import MemoryConfig
        from isat.search.kernel import KernelConfig
        from isat.search.precision import PrecisionConfig
        from isat.search.graph import GraphConfig

        db_path = str(tmp_path / "test.db")
        db = ResultsDB(db_path)

        config = CandidateConfig(
            memory=MemoryConfig(label="test_mem"),
            kernel=KernelConfig(label="test_kern"),
            precision=PrecisionConfig(label="test_prec"),
            graph=GraphConfig(label="test_graph"),
        )
        result = TuneResult(config=config, mean_latency_ms=42.5, p50_latency_ms=41.0,
                            p95_latency_ms=45.0, p99_latency_ms=47.0)

        row_id = db.save_result(result, "hw123", "model456", "test_model")
        assert row_id > 0

        best = db.best_for_model("test_model")
        assert len(best) == 1
        assert best[0]["mean_ms"] == 42.5

        db.close()
