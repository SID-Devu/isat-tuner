"""Report generator -- JSON, HTML, and console output for tuning results."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from isat.fingerprint.hardware import HardwareFingerprint
from isat.fingerprint.model import ModelFingerprint
from isat.search.engine import TuneResult

log = logging.getLogger("isat.report")


class ReportGenerator:
    """Generate reports from ISAT tuning results."""

    def __init__(
        self,
        hw: HardwareFingerprint,
        model: ModelFingerprint,
        results: list[TuneResult],
        output_dir: str = "isat_output",
    ):
        self.hw = hw
        self.model = model
        self.results = sorted(results, key=lambda r: r.mean_latency_ms)
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def best(self) -> Optional[TuneResult]:
        successful = [r for r in self.results if r.error is None]
        return successful[0] if successful else None

    @property
    def baseline(self) -> Optional[TuneResult]:
        for r in self.results:
            if "default" in r.config.label.lower() and r.error is None:
                return r
        successful = [r for r in self.results if r.error is None]
        return successful[-1] if successful else None

    def console_report(self) -> str:
        """Generate a console-friendly summary."""
        lines: list[str] = []
        w = 78

        lines.append("")
        lines.append("=" * w)
        lines.append("  ISAT TUNING RESULTS")
        lines.append("=" * w)
        lines.append(f"  GPU        : {self.hw.gpu_name} ({self.hw.gfx_target})")
        lines.append(f"  Model      : {self.model.name} ({self.model.size_class})")
        lines.append(f"  Configs    : {len(self.results)} tested")
        lines.append(f"  Successful : {sum(1 for r in self.results if r.error is None)}")
        lines.append("-" * w)

        best = self.best
        base = self.baseline
        if best and base and base.mean_latency_ms > 0:
            speedup = base.mean_latency_ms / best.mean_latency_ms
            lines.append(f"  BEST CONFIG : {best.config.label}")
            lines.append(f"  Latency     : {best.mean_latency_ms:.2f} ms (mean) / "
                         f"{best.p95_latency_ms:.2f} ms (p95)")
            lines.append(f"  Throughput  : {best.throughput_fps:.1f} fps")
            lines.append(f"  Speedup     : {speedup:.2f}x over baseline")
            lines.append(f"  Peak temp   : {best.peak_gpu_temp_c:.1f} C")
            lines.append(f"  Peak power  : {best.peak_power_w:.1f} W")
            lines.append("")
            lines.append("  RECOMMENDED ENVIRONMENT:")
            for k, v in best.config.merged_env.items():
                lines.append(f"    export {k}={v}")
            lines.append("-" * w)

        lines.append("")
        lines.append(f"  {'Rank':<5} {'Config':<40} {'Mean ms':<10} {'P95 ms':<10} {'FPS':<8} {'Err'}")
        lines.append(f"  {'-'*5} {'-'*40} {'-'*10} {'-'*10} {'-'*8} {'-'*3}")

        for rank, r in enumerate(self.results, 1):
            label = r.config.label[:40]
            if r.error:
                lines.append(f"  {rank:<5} {label:<40} {'---':<10} {'---':<10} {'---':<8} {'ERR'}")
            else:
                lines.append(f"  {rank:<5} {label:<40} {r.mean_latency_ms:<10.2f} "
                             f"{r.p95_latency_ms:<10.2f} {r.throughput_fps:<8.1f}")

        lines.append("=" * w)
        lines.append("")
        return "\n".join(lines)

    def json_report(self) -> str:
        """Generate JSON report and write to file."""
        data = {
            "isat_version": "0.1.0",
            "timestamp": time.time(),
            "hardware": {
                "gpu": self.hw.gpu_name,
                "gfx_target": self.hw.gfx_target,
                "cu_count": self.hw.cu_count,
                "vram_mb": self.hw.vram_total_mb,
                "gtt_mb": self.hw.gtt_total_mb,
                "is_apu": self.hw.is_apu,
                "rocm_version": self.hw.rocm_version,
                "fingerprint": self.hw.fingerprint_hash,
            },
            "model": {
                "name": self.model.name,
                "path": self.model.path,
                "params": self.model.param_count,
                "size_mb": round(self.model.estimated_memory_mb, 1),
                "class": self.model.model_class,
                "opset": self.model.opset,
                "fingerprint": self.model.fingerprint_hash,
            },
            "best_config": self.best.to_dict() if self.best else None,
            "all_results": [r.to_dict() for r in self.results],
        }

        out_path = Path(self.output_dir) / "isat_report.json"
        out_path.write_text(json.dumps(data, indent=2))
        log.info("JSON report: %s", out_path)
        return str(out_path)

    def html_report(self) -> str:
        """Generate a standalone HTML report."""
        best = self.best
        base = self.baseline
        speedup = (base.mean_latency_ms / best.mean_latency_ms) if (best and base and base.mean_latency_ms > 0) else 1.0

        env_lines = ""
        if best:
            for k, v in best.config.merged_env.items():
                env_lines += f"export {k}={v}\n"

        rows_html = ""
        for rank, r in enumerate(self.results, 1):
            is_best = r is best
            cls = ' class="best"' if is_best else ""
            if r.error:
                rows_html += (
                    f"<tr{cls}><td>{rank}</td><td>{r.config.label}</td>"
                    f"<td colspan='4' style='color:#e74c3c'>ERROR: {r.error[:60]}</td></tr>\n"
                )
            else:
                rows_html += (
                    f"<tr{cls}><td>{rank}</td><td>{r.config.label}</td>"
                    f"<td>{r.mean_latency_ms:.2f}</td><td>{r.p95_latency_ms:.2f}</td>"
                    f"<td>{r.throughput_fps:.1f}</td><td>{r.peak_gpu_temp_c:.1f}</td></tr>\n"
                )

        html = _HTML_TEMPLATE.format(
            gpu=self.hw.gpu_name,
            gfx=self.hw.gfx_target,
            model_name=self.model.name,
            model_class=self.model.model_class,
            model_size=f"{self.model.estimated_memory_mb:.0f} MB",
            n_configs=len(self.results),
            n_ok=sum(1 for r in self.results if r.error is None),
            best_label=best.config.label if best else "N/A",
            best_mean=f"{best.mean_latency_ms:.2f}" if best else "---",
            best_p95=f"{best.p95_latency_ms:.2f}" if best else "---",
            best_fps=f"{best.throughput_fps:.1f}" if best else "---",
            speedup=f"{speedup:.2f}",
            env_lines=env_lines or "# (no overrides)",
            rows=rows_html,
        )

        out_path = Path(self.output_dir) / "isat_report.html"
        out_path.write_text(html)
        log.info("HTML report: %s", out_path)
        return str(out_path)

    def generate_all(self) -> dict[str, str]:
        """Generate all report formats."""
        console = self.console_report()
        print(console)

        json_path = self.json_report()
        html_path = self.html_report()

        env_path = self._write_env_script()

        return {
            "console": console,
            "json": json_path,
            "html": html_path,
            "env_script": env_path,
        }

    def _write_env_script(self) -> str:
        """Write a shell script with the best config's env vars."""
        best = self.best
        out_path = Path(self.output_dir) / "best_config.sh"
        lines = ["#!/usr/bin/env bash", f"# ISAT best config for {self.model.name}", ""]
        if best:
            for k, v in best.config.merged_env.items():
                lines.append(f"export {k}={v}")
            lines.append("")
            lines.append(f"# Mean latency: {best.mean_latency_ms:.2f} ms")
            lines.append(f"# Throughput:   {best.throughput_fps:.1f} fps")
        else:
            lines.append("# No successful configurations found")
        lines.append("")

        out_path.write_text("\n".join(lines))
        out_path.chmod(0o755)
        return str(out_path)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ISAT Report - {model_name}</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --accent: #38bdf8;
           --green: #4ade80; --border: #334155; --best-bg: #14532d; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', system-ui, sans-serif; background: var(--bg);
          color: var(--text); line-height: 1.6; padding: 2rem; }}
  h1 {{ color: var(--accent); font-size: 1.8rem; margin-bottom: 0.5rem; }}
  h2 {{ color: var(--text); font-size: 1.2rem; margin: 1.5rem 0 0.8rem; }}
  .hero {{ background: var(--card); border-radius: 12px; padding: 2rem;
           margin-bottom: 2rem; border: 1px solid var(--border); }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem; margin: 1.5rem 0; }}
  .stat {{ background: var(--bg); border-radius: 8px; padding: 1rem;
           border: 1px solid var(--border); text-align: center; }}
  .stat .value {{ font-size: 1.8rem; font-weight: 700; color: var(--accent); }}
  .stat .label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th {{ background: var(--card); padding: 0.8rem; text-align: left;
       font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }}
  td {{ padding: 0.8rem; border-bottom: 1px solid var(--border); font-size: 0.9rem; }}
  tr:hover {{ background: var(--card); }}
  tr.best {{ background: var(--best-bg); }}
  pre {{ background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
         padding: 1rem; overflow-x: auto; font-size: 0.85rem; margin-top: 0.5rem; }}
  .footer {{ text-align: center; color: #475569; margin-top: 2rem; font-size: 0.8rem; }}
</style>
</head>
<body>
  <div class="hero">
    <h1>ISAT Tuning Report</h1>
    <p><strong>GPU:</strong> {gpu} ({gfx}) &nbsp;|&nbsp;
       <strong>Model:</strong> {model_name} ({model_class}, {model_size})</p>
    <div class="stats">
      <div class="stat"><div class="value">{best_mean}</div><div class="label">Best Mean (ms)</div></div>
      <div class="stat"><div class="value">{best_p95}</div><div class="label">Best P95 (ms)</div></div>
      <div class="stat"><div class="value">{best_fps}</div><div class="label">Best FPS</div></div>
      <div class="stat"><div class="value">{speedup}x</div><div class="label">Speedup</div></div>
      <div class="stat"><div class="value">{n_ok}/{n_configs}</div><div class="label">Configs OK</div></div>
    </div>
    <h2>Best Configuration</h2>
    <p><code>{best_label}</code></p>
    <pre>{env_lines}</pre>
  </div>

  <h2>All Results (ranked by mean latency)</h2>
  <table>
    <thead><tr><th>#</th><th>Config</th><th>Mean (ms)</th><th>P95 (ms)</th><th>FPS</th><th>Peak Temp</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="footer">Generated by ISAT v0.1.0 -- Inference Stack Auto-Tuner</div>
</body>
</html>"""
