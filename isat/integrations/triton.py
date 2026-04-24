"""Triton Inference Server configuration generator.

Converts ISAT tuning results into a Triton model repository config
(config.pbtxt) with optimal instance groups, batch settings, and
backend parameters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from isat.fingerprint.model import ModelFingerprint
from isat.search.engine import TuneResult

log = logging.getLogger("isat.integrations.triton")


def generate_triton_config(
    model_fp: ModelFingerprint,
    best_result: TuneResult,
    output_dir: str,
    max_batch_size: int = 8,
    instance_count: int = 1,
) -> str:
    """Generate a Triton config.pbtxt from ISAT tuning results."""
    config_lines = []

    model_name = model_fp.name.replace("-", "_").replace(".", "_")
    config_lines.append(f'name: "{model_name}"')
    config_lines.append(f'platform: "onnxruntime_onnx"')
    config_lines.append(f'max_batch_size: {max_batch_size}')

    config_lines.append("")
    for inp_name, inp_shape in model_fp.input_shapes.items():
        dims = []
        for d in inp_shape:
            dims.append(str(d) if isinstance(d, int) and d > 0 else "-1")
        config_lines.append(f"input {{")
        config_lines.append(f'  name: "{inp_name}"')
        config_lines.append(f"  data_type: TYPE_FP32")
        config_lines.append(f"  dims: [{', '.join(dims)}]")
        config_lines.append(f"}}")

    config_lines.append("")
    for out_name, out_shape in model_fp.output_shapes.items():
        dims = []
        for d in out_shape:
            dims.append(str(d) if isinstance(d, int) and d > 0 else "-1")
        config_lines.append(f"output {{")
        config_lines.append(f'  name: "{out_name}"')
        config_lines.append(f"  data_type: TYPE_FP32")
        config_lines.append(f"  dims: [{', '.join(dims)}]")
        config_lines.append(f"}}")

    config_lines.append("")
    config_lines.append("instance_group {")
    config_lines.append(f"  count: {instance_count}")
    config_lines.append(f"  kind: KIND_GPU")
    config_lines.append(f"  gpus: [0]")
    config_lines.append("}")

    config_lines.append("")
    config_lines.append("dynamic_batching {")
    config_lines.append(f"  preferred_batch_size: [1, {max_batch_size // 2}, {max_batch_size}]")
    latency_us = int(best_result.mean_latency_ms * 1000 * 0.5)
    config_lines.append(f"  max_queue_delay_microseconds: {latency_us}")
    config_lines.append("}")

    env = best_result.config.merged_env
    if env:
        config_lines.append("")
        config_lines.append("parameters {")
        for key, val in env.items():
            config_lines.append(f'  key: "{key}"')
            config_lines.append(f"  value: {{")
            config_lines.append(f'    string_value: "{val}"')
            config_lines.append(f"  }}")
        config_lines.append("}")

    config_lines.append("")
    config_lines.append("optimization {")
    config_lines.append("  graph {")
    config_lines.append(f"    level: {best_result.config.graph.ort_opt_level}")
    config_lines.append("  }")
    config_lines.append("}")

    config_text = "\n".join(config_lines) + "\n"

    repo_dir = Path(output_dir) / model_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    config_path = repo_dir / "config.pbtxt"
    config_path.write_text(config_text)

    model_dir = repo_dir / "1"
    model_dir.mkdir(exist_ok=True)

    log.info("Triton config written to %s", config_path)
    return str(config_path)


def generate_env_script(best_result: TuneResult, output_path: str) -> str:
    """Generate a shell script that sets up the environment for Triton."""
    lines = [
        "#!/usr/bin/env bash",
        "# ISAT-generated environment for Triton Inference Server",
        "",
    ]
    for key, val in best_result.config.merged_env.items():
        lines.append(f"export {key}={val}")

    lines.extend([
        "",
        f"# Best mean latency: {best_result.mean_latency_ms:.2f} ms",
        f"# Best P95 latency:  {best_result.p95_latency_ms:.2f} ms",
        f"# Throughput:        {best_result.throughput_fps:.1f} fps",
        "",
        'echo "ISAT environment configured. Start Triton with:"',
        'echo "  tritonserver --model-repository=./model_repository"',
        "",
    ])

    Path(output_path).write_text("\n".join(lines))
    Path(output_path).chmod(0o755)
    return output_path
