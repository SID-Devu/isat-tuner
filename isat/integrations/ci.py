"""CI/CD integration helpers.

Generate GitHub Actions workflows, GitLab CI configs, and
provide exit codes for automated performance gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from isat.search.engine import TuneResult


class PerformanceGate:
    """Enforce latency/throughput gates in CI pipelines.

    Exit with non-zero if the best config doesn't meet the target.
    """

    def __init__(
        self,
        max_latency_ms: Optional[float] = None,
        min_throughput_fps: Optional[float] = None,
        max_p95_ms: Optional[float] = None,
    ):
        self.max_latency_ms = max_latency_ms
        self.min_throughput_fps = min_throughput_fps
        self.max_p95_ms = max_p95_ms

    def check(self, results: list[TuneResult]) -> tuple[bool, list[str]]:
        """Check if results meet gates. Returns (passed, list of failure reasons)."""
        successful = [r for r in results if r.error is None]
        if not successful:
            return False, ["No successful configurations"]

        best = min(successful, key=lambda r: r.mean_latency_ms)
        failures: list[str] = []

        if self.max_latency_ms and best.mean_latency_ms > self.max_latency_ms:
            failures.append(
                f"Mean latency {best.mean_latency_ms:.2f} ms > gate {self.max_latency_ms} ms"
            )

        if self.min_throughput_fps and best.throughput_fps < self.min_throughput_fps:
            failures.append(
                f"Throughput {best.throughput_fps:.1f} fps < gate {self.min_throughput_fps} fps"
            )

        if self.max_p95_ms and best.p95_latency_ms > self.max_p95_ms:
            failures.append(
                f"P95 latency {best.p95_latency_ms:.2f} ms > gate {self.max_p95_ms} ms"
            )

        return len(failures) == 0, failures

    def enforce(self, results: list[TuneResult]) -> int:
        """Check and return exit code (0=pass, 1=fail)."""
        passed, failures = self.check(results)
        if passed:
            print("ISAT Performance Gate: PASSED")
            return 0
        print("ISAT Performance Gate: FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1


def generate_github_workflow(
    model_path: str = "model.onnx",
    output_path: str = ".github/workflows/isat-tune.yml",
) -> str:
    """Generate a GitHub Actions workflow for automated tuning."""
    workflow = f"""name: ISAT Auto-Tune

on:
  push:
    paths:
      - '**.onnx'
      - 'isat/**'
  workflow_dispatch:
    inputs:
      model_path:
        description: 'Path to ONNX model'
        default: '{model_path}'
      warmup:
        description: 'Warmup iterations'
        default: '3'
      runs:
        description: 'Measured iterations'
        default: '5'

jobs:
  tune:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install ISAT
        run: pip install -e .

      - name: Hardware Info
        run: isat hwinfo

      - name: Model Inspection
        run: isat inspect ${{{{ github.event.inputs.model_path || '{model_path}' }}}}

      - name: Auto-Tune
        run: |
          isat tune ${{{{ github.event.inputs.model_path || '{model_path}' }}}} \\
            --warmup ${{{{ github.event.inputs.warmup || '3' }}}} \\
            --runs ${{{{ github.event.inputs.runs || '5' }}}} \\
            --cooldown 60 \\
            --output-dir isat_output \\
            --verbose

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: isat-results
          path: |
            isat_output/
            isat_results.db
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(workflow)
    return output_path
