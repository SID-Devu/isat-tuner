"""Tests for the configuration system."""

import json
from pathlib import Path

import pytest


class TestConfigLoader:
    def test_default_config_generation(self, tmp_path):
        from isat.config.loader import generate_default_config
        path = str(tmp_path / "isat.yaml")
        generate_default_config(path)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "model_path" in content
        assert "warmup" in content
        assert "cooldown" in content

    def test_load_json_config(self, tmp_path):
        from isat.config.loader import load_config, TuneConfig
        config_data = {
            "model_path": "test.onnx",
            "provider": "CPUExecutionProvider",
            "benchmark": {"warmup": 5, "runs": 20, "cooldown": 120.0},
            "search": {"memory": True, "kernel": True, "precision": False},
        }
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config_data))

        config = load_config(str(path))
        assert config.model_path == "test.onnx"
        assert config.benchmark.warmup == 5
        assert config.benchmark.runs == 20
        assert config.search.precision is False

    def test_load_yaml_config(self, tmp_path):
        from isat.config.loader import load_config
        yaml_content = """
model_path: "test.onnx"
provider: "CPUExecutionProvider"
benchmark:
  warmup: 3
  runs: 10
  cooldown: 60.0
search:
  memory: true
  kernel: true
"""
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)

        config = load_config(str(path))
        assert config.model_path == "test.onnx"
        assert config.benchmark.warmup == 3
        assert config.search.memory is True

    def test_config_validation(self):
        from isat.config.loader import TuneConfig, BenchmarkConfig
        config = TuneConfig(
            model_path="/nonexistent/path.onnx",
            benchmark=BenchmarkConfig(warmup=-1, runs=0),
        )
        errors = config.validate()
        assert len(errors) >= 2

    def test_save_and_reload(self, tmp_path):
        from isat.config.loader import TuneConfig, save_config, load_config
        config = TuneConfig(model_path="test.onnx", description="test run")
        path = str(tmp_path / "saved.json")
        save_config(config, path)
        reloaded = load_config(path)
        assert reloaded.model_path == "test.onnx"
        assert reloaded.description == "test run"

    def test_to_yaml(self):
        from isat.config.loader import TuneConfig
        config = TuneConfig(model_path="model.onnx")
        yaml_str = config.to_yaml()
        assert "model_path" in yaml_str
        assert "model.onnx" in yaml_str


class TestModelZoo:
    def test_lookup_known_model(self):
        from isat.model_zoo import lookup
        entry = lookup("resnet50_v2")
        assert entry is not None
        assert entry.model_class == "cnn"

    def test_lookup_llm(self):
        from isat.model_zoo import lookup
        entry = lookup("llama-3-8b")
        assert entry is not None
        assert entry.model_class == "llm"
        assert "HSA_XNACK" in entry.recommended_env

    def test_lookup_unknown(self):
        from isat.model_zoo import lookup
        entry = lookup("totally_unknown_model_xyz")
        assert entry is None

    def test_list_supported(self):
        from isat.model_zoo import list_supported
        models = list_supported()
        assert len(models) >= 10
        patterns = [m["pattern"] for m in models]
        assert any("resnet" in p for p in patterns)
        assert any("bert" in p for p in patterns)

    def test_suggest_starting_config(self):
        from isat.model_zoo import suggest_starting_config
        env = suggest_starting_config("openvla-7b", hw_class="apu_unified")
        assert env is not None
        assert "MIGRAPHX_DISABLE_MLIR" in env


class TestPlugins:
    def test_registry_basics(self):
        from isat.plugins import PluginRegistry
        reg = PluginRegistry()
        reg.register_hook("pre_benchmark", lambda **kw: "ok")
        results = reg.fire("pre_benchmark")
        assert results == ["ok"]

    def test_register_search_dimension(self):
        from isat.plugins import PluginRegistry

        class DummyDim:
            pass

        reg = PluginRegistry()
        reg.register_search_dimension("dummy", DummyDim)
        assert "dummy" in reg.search_dimensions

    def test_unknown_event_raises(self):
        from isat.plugins import PluginRegistry
        reg = PluginRegistry()
        with pytest.raises(ValueError):
            reg.register_hook("nonexistent_event", lambda: None)


class TestRetry:
    def test_retry_succeeds_eventually(self):
        from isat.retry import retry

        call_count = 0

        @retry(max_attempts=3, base_delay=0.01)
        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        result = flaky()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausted(self):
        from isat.retry import retry

        @retry(max_attempts=2, base_delay=0.01)
        def always_fails():
            raise RuntimeError("always")

        with pytest.raises(RuntimeError):
            always_fails()

    def test_retry_context(self):
        from isat.retry import RetryContext

        attempt_count = 0
        with RetryContext(max_attempts=3, base_delay=0.01) as ctx:
            for attempt in ctx:
                attempt_count += 1
                if attempt < 3:
                    ctx.fail(ValueError("not yet"))
                else:
                    ctx.success("done")

        assert ctx.result == "done"
        assert attempt_count == 3


class TestLogging:
    def test_structured_formatter(self):
        import logging
        from isat.logging_config import StructuredFormatter

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="isat.test", level=logging.INFO, pathname="test.py",
            lineno=1, msg="test message", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "test message"
        assert data["level"] == "INFO"
