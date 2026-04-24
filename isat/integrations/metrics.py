"""Prometheus metrics exporter for ISAT tuning results.

Exposes tuning metrics in Prometheus text format so they can be
scraped by monitoring infrastructure for dashboards and alerting.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from isat.search.engine import TuneResult


class PrometheusExporter:
    """Export ISAT results as Prometheus metrics."""

    def __init__(self, model_name: str, hw_target: str):
        self.model_name = model_name
        self.hw_target = hw_target
        self._metrics: list[str] = []

    def add_result(self, result: TuneResult) -> None:
        labels = (
            f'model="{self.model_name}",'
            f'hw="{self.hw_target}",'
            f'config="{result.config.label}"'
        )

        ts = int(time.time() * 1000)

        if result.error is None:
            self._metrics.extend([
                f"isat_latency_mean_ms{{{labels}}} {result.mean_latency_ms:.3f} {ts}",
                f"isat_latency_p50_ms{{{labels}}} {result.p50_latency_ms:.3f} {ts}",
                f"isat_latency_p95_ms{{{labels}}} {result.p95_latency_ms:.3f} {ts}",
                f"isat_latency_p99_ms{{{labels}}} {result.p99_latency_ms:.3f} {ts}",
                f"isat_throughput_fps{{{labels}}} {result.throughput_fps:.2f} {ts}",
                f"isat_peak_temp_celsius{{{labels}}} {result.peak_gpu_temp_c:.1f} {ts}",
                f"isat_peak_power_watts{{{labels}}} {result.peak_power_w:.1f} {ts}",
                f"isat_peak_vram_mb{{{labels}}} {result.peak_vram_mb:.1f} {ts}",
                f"isat_tuning_success{{{labels}}} 1 {ts}",
            ])
        else:
            self._metrics.append(f"isat_tuning_success{{{labels}}} 0 {ts}")

    def add_batch(self, results: list[TuneResult]) -> None:
        for r in results:
            self.add_result(r)

        successful = [r for r in results if r.error is None]
        if successful:
            best = min(successful, key=lambda r: r.mean_latency_ms)
            labels = f'model="{self.model_name}",hw="{self.hw_target}"'
            ts = int(time.time() * 1000)
            self._metrics.extend([
                f'isat_best_latency_ms{{{labels}}} {best.mean_latency_ms:.3f} {ts}',
                f'isat_best_config{{{labels},config="{best.config.label}"}} 1 {ts}',
                f'isat_configs_tested{{{labels}}} {len(results)} {ts}',
                f'isat_configs_successful{{{labels}}} {len(successful)} {ts}',
            ])

    def render(self) -> str:
        """Render all metrics in Prometheus exposition format."""
        header = [
            "# HELP isat_latency_mean_ms Mean inference latency in milliseconds",
            "# TYPE isat_latency_mean_ms gauge",
            "# HELP isat_latency_p95_ms P95 inference latency in milliseconds",
            "# TYPE isat_latency_p95_ms gauge",
            "# HELP isat_throughput_fps Inference throughput in frames per second",
            "# TYPE isat_throughput_fps gauge",
            "# HELP isat_peak_temp_celsius Peak GPU temperature during benchmark",
            "# TYPE isat_peak_temp_celsius gauge",
            "# HELP isat_best_latency_ms Best latency found across all configs",
            "# TYPE isat_best_latency_ms gauge",
            "# HELP isat_configs_tested Total number of configs tested",
            "# TYPE isat_configs_tested counter",
            "# HELP isat_tuning_success Whether a config ran successfully (1=yes, 0=no)",
            "# TYPE isat_tuning_success gauge",
            "",
        ]
        return "\n".join(header + self._metrics) + "\n"

    def write(self, path: str) -> str:
        """Write metrics to a file for node_exporter textfile collector."""
        Path(path).write_text(self.render())
        return path
