"""Tests for ISAT v0.7.0 -- advanced real-time industry features."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _make_tiny_model():
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])

    W = numpy_helper.from_array(np.random.randn(4, 4).astype(np.float32), name="W")
    B = numpy_helper.from_array(np.random.randn(4).astype(np.float32), name="B")

    matmul = helper.make_node("MatMul", ["X", "W"], ["XW"])
    add = helper.make_node("Add", ["XW", "B"], ["XWB"])
    relu = helper.make_node("Relu", ["XWB"], ["Y"])

    graph = helper.make_graph([matmul, add, relu], "test", [X], [Y], initializer=[W, B])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    path = tempfile.mktemp(suffix=".onnx")
    with open(path, "wb") as f:
        f.write(model.SerializeToString())
    return path


# ── Pruning ───────────────────────────────────────────────────────

class TestPruning:
    def test_prune_magnitude(self):
        from isat.pruning.pruner import ModelPruner
        path = _make_tiny_model()
        try:
            pruner = ModelPruner(path)
            out = path.replace(".onnx", "_pruned.onnx")
            result = pruner.prune(strategy="magnitude", sparsity=0.5, output_path=out)
            assert result.total_params > 0
            assert result.overall_sparsity >= 0
            assert Path(out).exists()
            text = result.summary()
            assert "magnitude" in text
        finally:
            for p in [path, path.replace(".onnx", "_pruned.onnx")]:
                if os.path.exists(p):
                    os.unlink(p)

    def test_prune_percentage(self):
        from isat.pruning.pruner import ModelPruner
        path = _make_tiny_model()
        try:
            pruner = ModelPruner(path)
            result = pruner.prune(strategy="percentage", sparsity=0.3)
            assert result.total_pruned >= 0
        finally:
            for f in [path, path.replace(".onnx", "_pruned.onnx")]:
                if os.path.exists(f):
                    os.unlink(f)

    def test_analyze_sparsity(self):
        from isat.pruning.pruner import ModelPruner
        path = _make_tiny_model()
        try:
            pruner = ModelPruner(path)
            info = pruner.analyze_sparsity()
            assert info["total_params"] > 0
            assert "layers" in info
        finally:
            os.unlink(path)


# ── Distillation ──────────────────────────────────────────────────

class TestDistillation:
    def test_plan(self):
        from isat.distillation.helper import DistillationHelper
        path = _make_tiny_model()
        try:
            helper = DistillationHelper(path)
            plan = helper.plan()
            assert plan.teacher_params_m > 0
            assert len(plan.students) >= 2
            assert plan.temperature > 0
            text = plan.summary()
            assert "tiny" in text or "small" in text
        finally:
            os.unlink(path)


# ── Fusion ────────────────────────────────────────────────────────

class TestFusion:
    def test_analyze(self):
        from isat.fusion.analyzer import FusionAnalyzer
        path = _make_tiny_model()
        try:
            analyzer = FusionAnalyzer(path)
            report = analyzer.analyze()
            assert report.original_ops > 0
            assert report.optimized_ops >= 0
            text = report.summary()
            assert "Original ops" in text
        finally:
            os.unlink(path)


# ── Attention ─────────────────────────────────────────────────────

class TestAttention:
    def test_profile_no_attention(self):
        from isat.attention.profiler import AttentionProfiler
        path = _make_tiny_model()
        try:
            profiler = AttentionProfiler(path)
            report = profiler.profile()
            assert report.total_attention_layers == 0
        finally:
            os.unlink(path)


# ── LLM Benchmarker ──────────────────────────────────────────────

class TestLLMBench:
    def test_benchmark_basic(self):
        from isat.llm_bench.benchmarker import LLMBenchmarker
        path = _make_tiny_model()
        try:
            bench = LLMBenchmarker(path, sequence_lengths=[4], decode_steps=5)
            result = bench.benchmark(runs=2)
            assert result.overall_tps > 0
            assert len(result.metrics) == 1
            text = result.summary()
            assert "Tokens/second" in text
        finally:
            os.unlink(path)


# ── Compiler Compare ──────────────────────────────────────────────

class TestCompilerCompare:
    def test_compare_cpu(self):
        from isat.compiler_compare.comparator import CompilerComparator
        path = _make_tiny_model()
        try:
            comp = CompilerComparator(path, providers=["CPUExecutionProvider"])
            report = comp.compare(runs=5)
            assert report.providers_available >= 1
            cpu = [r for r in report.results if r.provider == "CPUExecutionProvider"]
            assert len(cpu) == 1
            assert cpu[0].mean_ms > 0
        finally:
            os.unlink(path)


# ── Replay ────────────────────────────────────────────────────────

class TestReplay:
    def test_record_and_replay(self, tmp_path):
        from isat.replay.recorder import InferenceRecorder, InferenceReplayer
        path = _make_tiny_model()
        rec_dir = str(tmp_path / "recording")
        try:
            rec = InferenceRecorder(rec_dir)
            count = rec.record_from_model(path, num_requests=5)
            assert count == 5
            assert (tmp_path / "recording" / "manifest.json").exists()

            replayer = InferenceReplayer(rec_dir)
            result = replayer.replay(path)
            assert result.replayed == 5
            assert result.output_match_pct > 0
            text = result.summary()
            assert "Output match" in text
        finally:
            os.unlink(path)


# ── Output Drift ──────────────────────────────────────────────────

class TestOutputDrift:
    def test_monitor_stable(self):
        from isat.output_monitor.drift import OutputMonitor
        path = _make_tiny_model()
        try:
            monitor = OutputMonitor(path, baseline_runs=10, monitor_runs=10)
            report = monitor.monitor()
            assert report.baseline_samples == 10
            assert report.monitor_samples == 10
            text = report.summary()
            assert "Status" in text
        finally:
            os.unlink(path)


# ── Weight Sharing ────────────────────────────────────────────────

class TestWeightSharing:
    def test_detect(self):
        from isat.weight_analysis.sharing import WeightSharingDetector
        path = _make_tiny_model()
        try:
            detector = WeightSharingDetector(path)
            report = detector.analyze()
            assert report.total_initializers > 0
            assert report.total_params > 0
            text = report.summary()
            assert "Total params" in text
        finally:
            os.unlink(path)


# ── Codegen ───────────────────────────────────────────────────────

class TestCodegen:
    def test_generate(self, tmp_path):
        from isat.codegen.generator import CppCodeGenerator
        path = _make_tiny_model()
        out_dir = str(tmp_path / "cpp_out")
        try:
            gen = CppCodeGenerator(path)
            result = gen.generate(output_dir=out_dir)
            assert Path(result.output_path).exists()
            assert Path(result.cmake_path).exists()
            assert result.lines_of_code > 10
            cpp_text = Path(result.output_path).read_text()
            assert "onnxruntime" in cpp_text.lower()
            assert "session.Run" in cpp_text
        finally:
            os.unlink(path)


# ── CLI Integration ──────────────────────────────────────────────

class TestCLIv07:
    def test_prune_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["prune", "--help"])

    def test_distill_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["distill", "--help"])

    def test_fusion_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["fusion", "--help"])

    def test_attention_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["attention", "--help"])

    def test_llm_bench_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["llm-bench", "--help"])

    def test_compiler_compare_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["compiler-compare", "--help"])

    def test_replay_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["replay", "--help"])

    def test_drift_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["drift", "--help"])

    def test_weight_sharing_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["weight-sharing", "--help"])

    def test_codegen_help(self):
        from isat.cli import main
        with pytest.raises(SystemExit):
            main(["codegen", "--help"])

    def test_version_updated(self):
        from isat import __version__
        assert __version__ == "0.7.0"
