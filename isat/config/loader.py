"""YAML/TOML configuration system for repeatable, version-controlled tuning.

Supports:
  - Loading configs from YAML files
  - Merging CLI overrides with file-based configs
  - Config inheritance (base + model-specific overrides)
  - Schema validation
  - Auto-generation of default configs
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("isat.config")


@dataclass
class SearchConfig:
    memory: bool = True
    kernel: bool = True
    precision: bool = True
    graph: bool = True
    batch: bool = False
    threading: bool = False
    provider: bool = False
    bayesian: bool = False
    max_configs: int = 0

    xnack_values: list[int] = field(default_factory=lambda: [0, 1])
    gemm_providers: list[str] = field(default_factory=lambda: ["default", "rocblas"])
    precisions: list[str] = field(default_factory=lambda: ["fp32", "fp16"])
    batch_sizes: list[int] = field(default_factory=lambda: [1])
    ort_opt_levels: list[int] = field(default_factory=lambda: [99])


@dataclass
class BenchmarkConfig:
    warmup: int = 3
    runs: int = 10
    cooldown: float = 60.0
    outlier_method: str = "mad"
    outlier_threshold: float = 3.5
    remove_outliers: bool = True


@dataclass
class ThermalConfig:
    max_temp_c: float = 85.0
    target_temp_c: float = 55.0
    poll_interval: float = 2.0
    abort_on_throttle: bool = False


@dataclass
class OutputConfig:
    directory: str = "isat_output"
    formats: list[str] = field(default_factory=lambda: ["json", "html", "console", "env"])
    database: str = "isat_results.db"
    prometheus_path: str = ""
    triton_output: str = ""
    save_optimized_model: bool = False


@dataclass
class GateConfig:
    max_latency_ms: float = 0.0
    max_p95_ms: float = 0.0
    max_p99_ms: float = 0.0
    min_throughput_fps: float = 0.0
    fail_on_regression: bool = False
    regression_threshold_pct: float = 10.0


@dataclass
class TuneConfig:
    model_path: str = ""
    model_name: str = ""
    provider: str = "MIGraphXExecutionProvider"
    profile: str = ""
    search: SearchConfig = field(default_factory=SearchConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gates: GateConfig = field(default_factory=GateConfig)
    env_overrides: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self) -> str:
        return _dict_to_yaml(self.to_dict())

    def validate(self) -> list[str]:
        """Validate config and return list of errors."""
        errors: list[str] = []
        if self.model_path and not Path(self.model_path).exists():
            errors.append(f"Model not found: {self.model_path}")
        if self.benchmark.warmup < 0:
            errors.append("warmup must be >= 0")
        if self.benchmark.runs < 1:
            errors.append("runs must be >= 1")
        if self.benchmark.cooldown < 0:
            errors.append("cooldown must be >= 0")
        if self.thermal.max_temp_c < 50:
            errors.append("max_temp_c should be >= 50")
        return errors


def load_config(path: str) -> TuneConfig:
    """Load a TuneConfig from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = p.read_text()

    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text)
        except ImportError:
            data = _parse_simple_yaml(text)
    elif p.suffix == ".json":
        data = json.loads(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _parse_simple_yaml(text)

    return _dict_to_config(data or {})


def save_config(config: TuneConfig, path: str) -> str:
    """Save a TuneConfig to YAML or JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if p.suffix == ".json":
        p.write_text(json.dumps(config.to_dict(), indent=2))
    else:
        p.write_text(config.to_yaml())

    log.info("Config saved to %s", path)
    return path


def generate_default_config(output_path: str = "isat.yaml") -> str:
    """Generate a default config file with all options documented."""
    template = """# ISAT Tuning Configuration
# https://github.com/SID-Devu/isat-tuner

model_path: "model.onnx"
model_name: ""
provider: "MIGraphXExecutionProvider"
profile: ""
description: ""
tags: []

search:
  memory: true
  kernel: true
  precision: true
  graph: true
  batch: false
  threading: false
  provider: false
  bayesian: false
  max_configs: 0
  xnack_values: [0, 1]
  gemm_providers: ["default", "rocblas"]
  precisions: ["fp32", "fp16"]
  batch_sizes: [1]
  ort_opt_levels: [99]

benchmark:
  warmup: 3
  runs: 10
  cooldown: 60.0
  outlier_method: "mad"
  outlier_threshold: 3.5
  remove_outliers: true

thermal:
  max_temp_c: 85.0
  target_temp_c: 55.0
  poll_interval: 2.0
  abort_on_throttle: false

output:
  directory: "isat_output"
  formats: ["json", "html", "console", "env"]
  database: "isat_results.db"
  prometheus_path: ""
  triton_output: ""
  save_optimized_model: false

gates:
  max_latency_ms: 0
  max_p95_ms: 0
  max_p99_ms: 0
  min_throughput_fps: 0
  fail_on_regression: false
  regression_threshold_pct: 10.0

env_overrides: {}
"""
    Path(output_path).write_text(template)
    return output_path


def _dict_to_config(data: dict) -> TuneConfig:
    config = TuneConfig()
    config.model_path = data.get("model_path", "")
    config.model_name = data.get("model_name", "")
    config.provider = data.get("provider", "MIGraphXExecutionProvider")
    config.profile = data.get("profile", "")
    config.description = data.get("description", "")
    config.tags = data.get("tags", [])
    config.env_overrides = data.get("env_overrides", {})

    if "search" in data:
        s = data["search"]
        config.search = SearchConfig(**{k: v for k, v in s.items() if hasattr(config.search, k)})
    if "benchmark" in data:
        b = data["benchmark"]
        config.benchmark = BenchmarkConfig(**{k: v for k, v in b.items() if hasattr(config.benchmark, k)})
    if "thermal" in data:
        t = data["thermal"]
        config.thermal = ThermalConfig(**{k: v for k, v in t.items() if hasattr(config.thermal, k)})
    if "output" in data:
        o = data["output"]
        config.output = OutputConfig(**{k: v for k, v in o.items() if hasattr(config.output, k)})
    if "gates" in data:
        g = data["gates"]
        config.gates = GateConfig(**{k: v for k, v in g.items() if hasattr(config.gates, k)})

    return config


def _dict_to_yaml(d: dict, indent: int = 0) -> str:
    """Minimal YAML serializer (no dependency needed)."""
    lines = []
    prefix = "  " * indent
    for key, val in d.items():
        if isinstance(val, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_dict_to_yaml(val, indent + 1))
        elif isinstance(val, list):
            if not val:
                lines.append(f"{prefix}{key}: []")
            elif all(isinstance(v, (int, float, str, bool)) for v in val):
                items = ", ".join(json.dumps(v) for v in val)
                lines.append(f"{prefix}{key}: [{items}]")
            else:
                lines.append(f"{prefix}{key}:")
                for item in val:
                    if isinstance(item, dict):
                        lines.append(f"{prefix}  -")
                        lines.append(_dict_to_yaml(item, indent + 2))
                    else:
                        lines.append(f"{prefix}  - {json.dumps(item)}")
        elif isinstance(val, bool):
            lines.append(f"{prefix}{key}: {'true' if val else 'false'}")
        elif isinstance(val, str):
            lines.append(f'{prefix}{key}: "{val}"')
        elif val is None:
            lines.append(f"{prefix}{key}: null")
        else:
            lines.append(f"{prefix}{key}: {val}")
    return "\n".join(lines)


def _parse_simple_yaml(text: str) -> dict:
    """Bare-minimum YAML-like parser for when PyYAML is not installed."""
    result: dict = {}
    stack: list[tuple[dict, int]] = [(result, -1)]

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][1] >= indent:
            stack.pop()

        parent = stack[-1][0] if stack else result

        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip().strip('"').strip("'")
            val = val.strip()

            if not val:
                child: dict = {}
                parent[key] = child
                stack.append((child, indent))
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                if not inner:
                    parent[key] = []
                else:
                    items = []
                    for item in inner.split(","):
                        item = item.strip().strip('"').strip("'")
                        try:
                            items.append(int(item))
                        except ValueError:
                            try:
                                items.append(float(item))
                            except ValueError:
                                if item.lower() in ("true", "false"):
                                    items.append(item.lower() == "true")
                                else:
                                    items.append(item)
                    parent[key] = items
            else:
                val = val.strip('"').strip("'")
                if val.lower() == "true":
                    parent[key] = True
                elif val.lower() == "false":
                    parent[key] = False
                elif val.lower() == "null":
                    parent[key] = None
                else:
                    try:
                        parent[key] = int(val)
                    except ValueError:
                        try:
                            parent[key] = float(val)
                        except ValueError:
                            parent[key] = val

    return result
