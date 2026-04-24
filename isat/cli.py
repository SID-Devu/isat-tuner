"""ISAT command-line interface.

Usage:
    isat tune      MODEL.onnx          -- run full auto-tune search
    isat inspect   MODEL.onnx          -- fingerprint model without benchmarking
    isat hwinfo                        -- print hardware fingerprint
    isat history   [--model NAME]      -- show past tuning results
    isat export    [--model NAME]      -- re-generate reports from DB
    isat compare   MODEL.onnx          -- compare two configs with significance testing
    isat serve     [--port 8000]       -- launch REST API server
    isat triton    MODEL.onnx          -- generate Triton Inference Server config
    isat profiles                      -- list available tuning profiles
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from isat import __version__

BANNER = (
    r"""
  ___ ____    _  _____
 |_ _/ ___|  / \|_   _|
  | |\___ \ / _ \ | |
  | | ___) / ___ \| |
 |___|____/_/   \_\_|

"""
    + f"  Inference Stack Auto-Tuner v{__version__}\n"
    + "  by Sudheer Ibrahim Daniel Devu\n"
)


def _check_path_issue() -> str | None:
    """Return a help message if the 'isat' script is installed but not on PATH."""
    local_bin = Path.home() / ".local" / "bin"
    isat_script = local_bin / "isat"
    if isat_script.exists() and str(local_bin) not in os.environ.get("PATH", ""):
        return (
            f"\n  NOTE: The 'isat' command is installed at {isat_script}\n"
            f"  but {local_bin} is not on your PATH.\n\n"
            f"  Fix (permanent):\n"
            f'    echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc && source ~/.bashrc\n\n'
            f"  Or run directly:\n"
            f"    python3 -m isat --help\n"
        )
    if not shutil.which("isat") and not sys.argv[0].endswith("__main__.py"):
        return (
            "\n  TIP: If 'isat' is not found after install, run:\n"
            "    python3 -m isat --help\n"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="isat",
        description="Inference Stack Auto-Tuner -- find the fastest ORT config for your model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'isat <command> --help' for detailed options.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="version", version=f"isat {__version__}")
    sub = parser.add_subparsers(dest="command")

    # ── tune ──────────────────────────────────────────────────
    p_tune = sub.add_parser("tune", help="Auto-tune an ONNX model")
    p_tune.add_argument("model", help="Path to .onnx model")
    p_tune.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    p_tune.add_argument("--runs", type=int, default=5, help="Measured iterations (default: 5)")
    p_tune.add_argument("--cooldown", type=float, default=60.0, help="Cooldown between configs (seconds)")
    p_tune.add_argument("--max-configs", type=int, default=0, help="Limit configs to test (0=unlimited)")
    p_tune.add_argument("--provider", default="MIGraphXExecutionProvider",
                        help="ORT execution provider")
    p_tune.add_argument("--skip-precision", action="store_true", help="Skip precision search dimension")
    p_tune.add_argument("--skip-graph", action="store_true", help="Skip graph transform search dimension")
    p_tune.add_argument("--profile", default=None,
                        help="Use a tuning profile (edge/cloud/latency/throughput/power/quick/exhaustive/apu)")
    p_tune.add_argument("--bayesian", action="store_true",
                        help="Use Bayesian optimization instead of grid search")
    p_tune.add_argument("--pareto", nargs="*", default=None,
                        metavar="OBJ", help="Run Pareto analysis (objectives: latency_ms, memory_mb, power_w, temp_c)")
    p_tune.add_argument("--gate-latency", type=float, default=None,
                        help="CI gate: fail if best mean latency > N ms")
    p_tune.add_argument("--gate-throughput", type=float, default=None,
                        help="CI gate: fail if best throughput < N fps")
    p_tune.add_argument("--prometheus", default=None, metavar="PATH",
                        help="Export Prometheus metrics to file")
    p_tune.add_argument("--triton-output", default=None, metavar="DIR",
                        help="Generate Triton config in DIR")
    p_tune.add_argument("--output-dir", default="isat_output", help="Output directory for reports")
    p_tune.add_argument("--db", default="isat_results.db", help="Results database path")
    p_tune.add_argument("--dry-run", action="store_true", help="Show plan without benchmarking")

    # ── inspect ───────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Fingerprint a model without benchmarking")
    p_inspect.add_argument("model", help="Path to .onnx model")
    p_inspect.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # ── hwinfo ────────────────────────────────────────────────
    p_hwinfo = sub.add_parser("hwinfo", help="Print hardware fingerprint")
    p_hwinfo.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # ── history ───────────────────────────────────────────────
    p_hist = sub.add_parser("history", help="Show past tuning results from database")
    p_hist.add_argument("--model", default=None, help="Filter by model name")
    p_hist.add_argument("--top", type=int, default=10, help="Show top N results")
    p_hist.add_argument("--db", default="isat_results.db", help="Database path")
    p_hist.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # ── export ────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Re-generate reports from database")
    p_export.add_argument("--model", required=True, help="Model name to export")
    p_export.add_argument("--db", default="isat_results.db", help="Database path")
    p_export.add_argument("--output-dir", default="isat_output", help="Output directory")

    # ── compare ───────────────────────────────────────────────
    p_compare = sub.add_parser("compare", help="Compare two configs with significance testing")
    p_compare.add_argument("model", help="Path to .onnx model")
    p_compare.add_argument("--config-a", required=True, help="First config label")
    p_compare.add_argument("--config-b", required=True, help="Second config label")
    p_compare.add_argument("--runs", type=int, default=30, help="Runs per config (default: 30)")
    p_compare.add_argument("--confidence", type=float, default=0.95, help="Confidence level")

    # ── serve ─────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Launch ISAT REST API server")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port")
    p_serve.add_argument("--db", default="isat_results.db", help="Database path")
    p_serve.add_argument("--output-dir", default="isat_output", help="Output directory")

    # ── triton ────────────────────────────────────────────────
    p_triton = sub.add_parser("triton", help="Generate Triton Inference Server config")
    p_triton.add_argument("model", help="Path to .onnx model")
    p_triton.add_argument("--db", default="isat_results.db", help="Database path")
    p_triton.add_argument("--output-dir", default="model_repository", help="Triton repo dir")
    p_triton.add_argument("--max-batch", type=int, default=8, help="Max batch size")

    # ── profiles ──────────────────────────────────────────────
    sub.add_parser("profiles", help="List available tuning profiles")

    # ── optimize ──────────────────────────────────────────────
    p_opt = sub.add_parser("optimize", help="Optimize an ONNX model (simplify, quantize, export)")
    p_opt.add_argument("model", help="Path to .onnx model")
    p_opt.add_argument("--simplify", action="store_true", help="Run onnxsim")
    p_opt.add_argument("--fp16", action="store_true", help="Convert weights to FP16")
    p_opt.add_argument("--int8", action="store_true", help="INT8 QDQ quantization")
    p_opt.add_argument("--ort-optimize", action="store_true", help="ORT graph optimization")
    p_opt.add_argument("--all", action="store_true", dest="all_transforms", help="Apply all transforms")
    p_opt.add_argument("--output-dir", default="isat_optimized", help="Output directory")

    # ── stress ────────────────────────────────────────────────
    p_stress = sub.add_parser("stress", help="Run stress tests on a model")
    p_stress.add_argument("model", help="Path to .onnx model")
    p_stress.add_argument("--pattern", choices=["sustained", "burst", "ramp"], default="sustained")
    p_stress.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (sustained)")
    p_stress.add_argument("--concurrency", type=int, default=4, help="Concurrent threads")
    p_stress.add_argument("--provider", default="MIGraphXExecutionProvider")

    # ── leak-check ────────────────────────────────────────────
    p_leak = sub.add_parser("leak-check", help="Detect memory leaks during inference")
    p_leak.add_argument("model", help="Path to .onnx model")
    p_leak.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    p_leak.add_argument("--provider", default="MIGraphXExecutionProvider")

    # ── init ──────────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Generate a default isat.yaml config file")
    p_init.add_argument("--output", default="isat.yaml", help="Output path")

    # ── zoo ───────────────────────────────────────────────────
    p_zoo = sub.add_parser("zoo", help="List pre-tuned model configurations")
    p_zoo.add_argument("--search", default=None, help="Search for a specific model")

    # ── doctor ────────────────────────────────────────────────
    p_doc = sub.add_parser("doctor", help="Pre-flight system health and compatibility check")
    p_doc.add_argument("--model", default=None, help="Optional model to check compatibility")

    # ── profile ──────────────────────────────────────────────
    p_prof = sub.add_parser("profile", help="Decompose latency into phases (load/compile/inference)")
    p_prof.add_argument("model", help="Path to .onnx model")
    p_prof.add_argument("--runs", type=int, default=50, help="Steady-state runs")
    p_prof.add_argument("--warmup", type=int, default=3, help="Warmup iterations before profiling")
    p_prof.add_argument("--provider", default="MIGraphXExecutionProvider")

    # ── diff ─────────────────────────────────────────────────
    p_diff = sub.add_parser("diff", help="Structural diff between two ONNX models")
    p_diff.add_argument("model_a", help="First model")
    p_diff.add_argument("model_b", help="Second model")

    # ── cost ─────────────────────────────────────────────────
    p_cost = sub.add_parser("cost", help="Estimate cloud inference cost")
    p_cost.add_argument("--latency", type=float, required=True, help="Mean latency in ms")
    p_cost.add_argument("--gpu", default="a10g", help="GPU type (a10g, t4, a100_40gb, h100_80gb, etc.)")
    p_cost.add_argument("--batch-size", type=int, default=1)
    p_cost.add_argument("--list-gpus", action="store_true", help="List available GPU pricing")

    # ── sla ──────────────────────────────────────────────────
    p_sla = sub.add_parser("sla", help="Validate inference against SLA requirements")
    p_sla.add_argument("--template", choices=["realtime", "batch", "edge", "llm", "mobile"], default="realtime")
    p_sla.add_argument("--p50", type=float, default=0, help="P50 latency ms")
    p_sla.add_argument("--p95", type=float, default=0, help="P95 latency ms")
    p_sla.add_argument("--p99", type=float, default=0, help="P99 latency ms")
    p_sla.add_argument("--throughput", type=float, default=0, help="Throughput RPS")
    p_sla.add_argument("--memory", type=float, default=0, help="Memory MB")
    p_sla.add_argument("--cold-start", type=float, default=0, help="Cold start ms")
    p_sla.add_argument("--list-templates", action="store_true")

    # ── warmup ───────────────────────────────────────────────
    p_warm = sub.add_parser("warmup", help="Analyze warmup behavior and find optimal iterations")
    p_warm.add_argument("model", help="Path to .onnx model")
    p_warm.add_argument("--max-iterations", type=int, default=100)
    p_warm.add_argument("--provider", default="MIGraphXExecutionProvider")

    # ── cache ────────────────────────────────────────────────
    p_cache = sub.add_parser("cache", help="Manage compilation cache (MIGraphX/ORT)")
    p_cache.add_argument("action", choices=["stats", "clean", "warm"], help="Cache action")
    p_cache.add_argument("--model", default=None, help="Model path (for warm)")
    p_cache.add_argument("--max-age-hours", type=float, default=168, help="Max cache age (for clean)")
    p_cache.add_argument("--dry-run", action="store_true")

    # ── migrate ──────────────────────────────────────────────
    p_mig = sub.add_parser("migrate", help="Generate migration plan between providers")
    p_mig.add_argument("--from", dest="source", required=True, help="Source provider")
    p_mig.add_argument("--to", dest="target", required=True, help="Target provider")

    # ── shapes ───────────────────────────────────────────────
    p_shapes = sub.add_parser("shapes", help="Benchmark model across dynamic input shapes")
    p_shapes.add_argument("model", help="Path to .onnx model")
    p_shapes.add_argument("--provider", default="CPUExecutionProvider")
    p_shapes.add_argument("--runs", type=int, default=10)

    # ── download ─────────────────────────────────────────────
    p_dl = sub.add_parser("download", help="Download ONNX model by name or URL")
    p_dl.add_argument("model_name", help="Model name (resnet50, mobilenetv2, etc.) or URL")
    p_dl.add_argument("--output-dir", default=".", help="Output directory")
    p_dl.add_argument("--list", action="store_true", dest="list_models", help="List available models")

    # ── power ────────────────────────────────────────────────
    p_pow = sub.add_parser("power", help="Profile power efficiency (perf/watt, energy/inference)")
    p_pow.add_argument("model", help="Path to .onnx model")
    p_pow.add_argument("--provider", default="MIGraphXExecutionProvider")
    p_pow.add_argument("--runs", type=int, default=50)

    # ── memory ───────────────────────────────────────────────
    p_mem = sub.add_parser("memory", help="Estimate memory usage and predict OOM risk")
    p_mem.add_argument("model", help="Path to .onnx model")

    # ── abtest ───────────────────────────────────────────────
    p_ab = sub.add_parser("abtest", help="A/B test two models with statistical rigor")
    p_ab.add_argument("model_a", help="First model")
    p_ab.add_argument("model_b", help="Second model")
    p_ab.add_argument("--provider", default="CPUExecutionProvider")
    p_ab.add_argument("--runs", type=int, default=50)
    p_ab.add_argument("--name-a", default="A")
    p_ab.add_argument("--name-b", default="B")

    # ── visualize ────────────────────────────────────────────
    p_viz = sub.add_parser("visualize", help="Visualize ONNX graph (DOT, ASCII, histogram)")
    p_viz.add_argument("model", help="Path to .onnx model")
    p_viz.add_argument("--format", choices=["dot", "ascii", "histogram"], default="ascii")
    p_viz.add_argument("--output", default="", help="Output file (for DOT)")

    # ── snapshot ─────────────────────────────────────────────
    p_snap = sub.add_parser("snapshot", help="Capture environment state for reproducibility")
    p_snap.add_argument("--model", default="", help="Optional model to include hash")
    p_snap.add_argument("--output", default="isat_snapshot.json")

    # ── batch ────────────────────────────────────────────────
    p_batch = sub.add_parser("batch", help="Find optimal batch size (latency vs throughput)")
    p_batch.add_argument("model", help="Path to .onnx model")
    p_batch.add_argument("--provider", default="CPUExecutionProvider")
    p_batch.add_argument("--sizes", default="1,2,4,8,16,32", help="Comma-separated batch sizes")

    # ── scan ───────────────────────────────────────────────
    p_scan = sub.add_parser("scan", help="Security and compliance scan of ONNX model")
    p_scan.add_argument("model", help="Path to .onnx model")

    # ── regression ─────────────────────────────────────────
    p_reg = sub.add_parser("regression", help="Performance regression detection")
    p_reg.add_argument("model", help="Path to .onnx model")
    p_reg.add_argument("--threshold", type=float, default=5.0, help="Regression threshold %%")
    p_reg.add_argument("--runs", type=int, default=20, help="Benchmark runs")
    p_reg.add_argument("--provider", default="CPUExecutionProvider")
    p_reg.add_argument("--history", action="store_true", help="Show regression history")
    p_reg.add_argument("--set-baseline", action="store_true", help="Force save as baseline")
    p_reg.add_argument("--db", default="isat_results.db")

    # ── compat-matrix ──────────────────────────────────────
    p_cm = sub.add_parser("compat-matrix", help="Operator compatibility across providers")
    p_cm.add_argument("model", help="Path to .onnx model")

    # ── thermal ────────────────────────────────────────────
    p_therm = sub.add_parser("thermal", help="Thermal throttle detection during inference")
    p_therm.add_argument("model", help="Path to .onnx model")
    p_therm.add_argument("--provider", default="CPUExecutionProvider")
    p_therm.add_argument("--runs", type=int, default=100)

    # ── quant-sensitivity ──────────────────────────────────
    p_qs = sub.add_parser("quant-sensitivity", help="Per-layer quantization sensitivity analysis")
    p_qs.add_argument("model", help="Path to .onnx model")

    # ── pipeline ───────────────────────────────────────────
    p_pipe = sub.add_parser("pipeline", help="Profile multi-model inference pipeline")
    p_pipe.add_argument("models", nargs="+", help="model_name:path pairs (e.g. encoder:enc.onnx decoder:dec.onnx)")
    p_pipe.add_argument("--provider", default="CPUExecutionProvider")
    p_pipe.add_argument("--runs", type=int, default=20)

    # ── recommend ──────────────────────────────────────────
    p_rec = sub.add_parser("recommend", help="Hardware recommendation for a model")
    p_rec.add_argument("model", help="Path to .onnx model")
    p_rec.add_argument("--max-latency", type=float, default=0, help="Max acceptable latency ms")
    p_rec.add_argument("--max-cost", type=float, default=0, help="Max $/hr budget")
    p_rec.add_argument("--prefer-amd", action="store_true")

    # ── registry ───────────────────────────────────────────
    p_rgy = sub.add_parser("registry", help="Model version registry")
    p_rgy.add_argument("action", choices=["register", "list", "promote", "diff", "show"],
                       help="Registry action")
    p_rgy.add_argument("--model-name", default="", help="Model name")
    p_rgy.add_argument("--version", default="", help="Model version")
    p_rgy.add_argument("--model-path", default="", help="Path to .onnx model (for register)")
    p_rgy.add_argument("--stage", default="production", help="Stage (for promote)")
    p_rgy.add_argument("--version-b", default="", help="Second version (for diff)")
    p_rgy.add_argument("--db", default="isat_registry.db")

    # ── tracing ────────────────────────────────────────────
    p_trace = sub.add_parser("trace", help="Trace inference requests (OpenTelemetry-compatible)")
    p_trace.add_argument("model", help="Path to .onnx model")
    p_trace.add_argument("--provider", default="CPUExecutionProvider")
    p_trace.add_argument("--runs", type=int, default=10)
    p_trace.add_argument("--export", default="", help="Export traces to OTLP JSON file")

    # ── canary ─────────────────────────────────────────────
    p_canary = sub.add_parser("canary", help="Canary deployment between two model versions")
    p_canary.add_argument("baseline", help="Baseline model path")
    p_canary.add_argument("canary_model", help="Canary model path")
    p_canary.add_argument("--provider", default="CPUExecutionProvider")
    p_canary.add_argument("--requests-per-phase", type=int, default=50)
    p_canary.add_argument("--max-error-rate", type=float, default=0.05)
    p_canary.add_argument("--max-latency-increase", type=float, default=20.0, help="Max pct latency increase")

    # ── alerts ─────────────────────────────────────────────
    p_alert = sub.add_parser("alerts", help="Inference alert rules engine")
    p_alert.add_argument("action", choices=["list", "check", "export"], help="Alert action")
    p_alert.add_argument("--metrics-json", default="", help="JSON string of metrics for check")
    p_alert.add_argument("--output", default="isat_alert_rules.json")

    # ── surgery ────────────────────────────────────────────
    p_surg = sub.add_parser("surgery", help="ONNX graph surgery (remove/rename/extract)")
    p_surg.add_argument("model", help="Path to .onnx model")
    p_surg.add_argument("--remove-op", action="append", default=[], help="Remove all nodes of this op type")
    p_surg.add_argument("--remove-unused", action="store_true", help="Remove unused initializers")
    p_surg.add_argument("--rename-input", nargs=2, metavar=("OLD", "NEW"), action="append", default=[])
    p_surg.add_argument("--rename-output", nargs=2, metavar=("OLD", "NEW"), action="append", default=[])
    p_surg.add_argument("--change-opset", type=int, default=0)
    p_surg.add_argument("--output", default="", help="Output model path")

    # ── guard ──────────────────────────────────────────────
    p_guard = sub.add_parser("guard", help="Validate inference inputs against model schema")
    p_guard.add_argument("model", help="Path to .onnx model")
    p_guard.add_argument("--show-schema", action="store_true", help="Show input schema only")

    # ── ensemble ───────────────────────────────────────────
    p_ens = sub.add_parser("ensemble", help="Run model ensemble with aggregation")
    p_ens.add_argument("models", nargs="+", help="model_name:path[:weight] entries")
    p_ens.add_argument("--strategy", choices=["average", "vote", "max_confidence"], default="average")
    p_ens.add_argument("--provider", default="CPUExecutionProvider")
    p_ens.add_argument("--runs", type=int, default=5)

    # ── gpu-frag ───────────────────────────────────────────
    p_frag = sub.add_parser("gpu-frag", help="GPU memory fragmentation analysis")
    p_frag.add_argument("model", help="Path to .onnx model")
    p_frag.add_argument("--provider", default="CPUExecutionProvider")
    p_frag.add_argument("--runs", type=int, default=200)

    # ── prune ──────────────────────────────────────────────
    p_prune = sub.add_parser("prune", help="Prune model weights (magnitude/percentage/global)")
    p_prune.add_argument("model", help="Path to .onnx model")
    p_prune.add_argument("--strategy", choices=["magnitude", "percentage", "global"], default="magnitude")
    p_prune.add_argument("--sparsity", type=float, default=0.5, help="Target sparsity 0-1")
    p_prune.add_argument("--output", default="", help="Output model path")
    p_prune.add_argument("--analyze-only", action="store_true", help="Show current sparsity without pruning")

    # ── distill ────────────────────────────────────────────
    p_dist = sub.add_parser("distill", help="Knowledge distillation planning for teacher model")
    p_dist.add_argument("model", help="Path to teacher .onnx model")

    # ── fusion ─────────────────────────────────────────────
    p_fus = sub.add_parser("fusion", help="Analyze operator fusion (fused vs unfused ops)")
    p_fus.add_argument("model", help="Path to .onnx model")

    # ── attention ──────────────────────────────────────────
    p_attn = sub.add_parser("attention", help="Profile attention heads in transformer models")
    p_attn.add_argument("model", help="Path to .onnx model")
    p_attn.add_argument("--head-dim", type=int, default=64, help="Attention head dimension")

    # ── llm-bench ──────────────────────────────────────────
    p_llm = sub.add_parser("llm-bench", help="LLM token throughput benchmark (TPS, TTFT, ITL)")
    p_llm.add_argument("model", help="Path to .onnx model")
    p_llm.add_argument("--provider", default="CPUExecutionProvider")
    p_llm.add_argument("--seq-lengths", default="32,64,128,256", help="Comma-separated sequence lengths")
    p_llm.add_argument("--decode-steps", type=int, default=20)
    p_llm.add_argument("--runs", type=int, default=5)

    # ── compiler-compare ───────────────────────────────────
    p_cc = sub.add_parser("compiler-compare", help="Compare model across all execution providers")
    p_cc.add_argument("model", help="Path to .onnx model")
    p_cc.add_argument("--runs", type=int, default=30)

    # ── replay ─────────────────────────────────────────────
    p_rep = sub.add_parser("replay", help="Record or replay inference requests")
    p_rep.add_argument("action", choices=["record", "replay"], help="Record or replay")
    p_rep.add_argument("model", help="Path to .onnx model")
    p_rep.add_argument("--dir", default="isat_recording", help="Recording directory")
    p_rep.add_argument("--provider", default="CPUExecutionProvider")
    p_rep.add_argument("--num-requests", type=int, default=20)

    # ── drift ──────────────────────────────────────────────
    p_drift = sub.add_parser("drift", help="Monitor output quality and detect confidence drift")
    p_drift.add_argument("model", help="Path to .onnx model")
    p_drift.add_argument("--provider", default="CPUExecutionProvider")
    p_drift.add_argument("--baseline-runs", type=int, default=50)
    p_drift.add_argument("--monitor-runs", type=int, default=50)

    # ── weight-sharing ─────────────────────────────────────
    p_ws = sub.add_parser("weight-sharing", help="Detect shared/tied weights across layers")
    p_ws.add_argument("model", help="Path to .onnx model")
    p_ws.add_argument("--similarity", type=float, default=0.999, help="Cosine similarity threshold")

    # ── codegen ────────────────────────────────────────────
    p_cg = sub.add_parser("codegen", help="Generate standalone C++ inference code from ONNX model")
    p_cg.add_argument("model", help="Path to .onnx model")
    p_cg.add_argument("--output-dir", default="isat_cpp", help="Output directory for C++ files")

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        print(BANNER)
        path_msg = _check_path_issue()
        if path_msg:
            print(path_msg)
        parser.print_help()
        return 0

    try:
        handlers = {
            "tune": _cmd_tune,
            "inspect": _cmd_inspect,
            "hwinfo": _cmd_hwinfo,
            "history": _cmd_history,
            "export": _cmd_export,
            "compare": _cmd_compare,
            "serve": _cmd_serve,
            "triton": _cmd_triton,
            "profiles": _cmd_profiles,
            "optimize": _cmd_optimize,
            "stress": _cmd_stress,
            "leak-check": _cmd_leak_check,
            "init": _cmd_init,
            "zoo": _cmd_zoo,
            "doctor": _cmd_doctor,
            "profile": _cmd_profile,
            "diff": _cmd_diff,
            "cost": _cmd_cost,
            "sla": _cmd_sla,
            "warmup": _cmd_warmup,
            "cache": _cmd_cache,
            "migrate": _cmd_migrate,
            "shapes": _cmd_shapes,
            "download": _cmd_download,
            "power": _cmd_power,
            "memory": _cmd_memory,
            "abtest": _cmd_abtest,
            "visualize": _cmd_visualize,
            "snapshot": _cmd_snapshot,
            "batch": _cmd_batch,
            "scan": _cmd_scan,
            "regression": _cmd_regression,
            "compat-matrix": _cmd_compat_matrix,
            "thermal": _cmd_thermal,
            "quant-sensitivity": _cmd_quant_sensitivity,
            "pipeline": _cmd_pipeline,
            "recommend": _cmd_recommend,
            "registry": _cmd_registry,
            "trace": _cmd_trace,
            "canary": _cmd_canary,
            "alerts": _cmd_alerts,
            "surgery": _cmd_surgery,
            "guard": _cmd_guard,
            "ensemble": _cmd_ensemble,
            "gpu-frag": _cmd_gpu_frag,
            "prune": _cmd_prune,
            "distill": _cmd_distill,
            "fusion": _cmd_fusion,
            "attention": _cmd_attention,
            "llm-bench": _cmd_llm_bench,
            "compiler-compare": _cmd_compiler_compare,
            "replay": _cmd_replay,
            "drift": _cmd_drift,
            "weight-sharing": _cmd_weight_sharing,
            "codegen": _cmd_codegen,
        }
        handler = handlers.get(args.command)
        if handler:
            return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as e:
        logging.getLogger("isat").error("Fatal: %s", e, exc_info=True)
        return 1

    return 0


def _cmd_tune(args) -> int:
    from isat.benchmark.runner import BenchmarkRunner
    from isat.database.store import ResultsDB
    from isat.fingerprint.hardware import fingerprint_hardware
    from isat.fingerprint.model import fingerprint_model
    from isat.report.generator import ReportGenerator
    from isat.search.engine import SearchEngine

    model_path = args.model
    if not Path(model_path).exists():
        print(f"Error: model not found: {model_path}")
        return 1

    warmup, runs, cooldown = args.warmup, args.runs, args.cooldown
    skip_precision, skip_graph = args.skip_precision, args.skip_graph
    max_configs = args.max_configs

    if args.profile:
        from isat.profiles.presets import get_profile
        profile = get_profile(args.profile)
        warmup = profile.warmup
        runs = profile.runs
        cooldown = profile.cooldown
        skip_precision = profile.skip_precision
        skip_graph = profile.skip_graph
        if profile.max_configs:
            max_configs = profile.max_configs
        print(f"Using profile: {profile.name} -- {profile.description}")

    print(BANNER)
    print("[1/6] Fingerprinting hardware...")
    hw = fingerprint_hardware()

    print("[2/6] Analyzing model...")
    model_fp = fingerprint_model(model_path)

    print("[3/6] Generating search candidates...")
    engine = SearchEngine(
        hw, model_fp,
        warmup=warmup, runs=runs, cooldown=cooldown,
        max_configs=max_configs,
        skip_precision=skip_precision, skip_graph=skip_graph,
    )

    candidates = engine.generate_candidates()
    engine.print_plan(candidates)

    if args.dry_run:
        print("Dry run -- no benchmarks executed.")
        print(f"Would test {len(candidates)} configurations.")
        est_hours = len(candidates) * (warmup + runs) * 2 / 3600
        print(f"Estimated time: {est_hours:.1f} hours")
        return 0

    print(f"[4/6] Benchmarking {len(candidates)} configurations...")
    runner = BenchmarkRunner(
        hw, model_fp, model_path,
        warmup=warmup, runs=runs, cooldown=cooldown,
        provider=args.provider,
    )
    results = runner.run_all(candidates)

    print("[5/6] Saving results...")
    db = ResultsDB(args.db)
    db.save_batch(results, hw.fingerprint_hash, model_fp.fingerprint_hash, model_fp.name)
    db.close()

    print("[6/6] Generating reports...")
    reporter = ReportGenerator(hw, model_fp, results, output_dir=args.output_dir)
    paths = reporter.generate_all()

    if args.pareto is not None:
        from isat.analysis.pareto import ParetoFrontier
        objectives = args.pareto if args.pareto else ["latency_ms", "memory_mb"]
        pareto = ParetoFrontier(results, objectives=objectives)
        print(f"\n{pareto.summary()}")

    if args.prometheus:
        from isat.integrations.metrics import PrometheusExporter
        exporter = PrometheusExporter(model_fp.name, hw.gfx_target)
        exporter.add_batch(results)
        exporter.write(args.prometheus)
        print(f"  Prometheus metrics: {args.prometheus}")

    if args.triton_output:
        from isat.integrations.triton import generate_triton_config
        best = reporter.best
        if best:
            triton_path = generate_triton_config(model_fp, best, args.triton_output)
            print(f"  Triton config: {triton_path}")

    print(f"\nReports saved to: {args.output_dir}/")
    print(f"  JSON : {paths['json']}")
    print(f"  HTML : {paths['html']}")
    print(f"  ENV  : {paths['env_script']}")
    print(f"  DB   : {args.db}")

    if args.gate_latency or args.gate_throughput:
        from isat.integrations.ci import PerformanceGate
        gate = PerformanceGate(
            max_latency_ms=args.gate_latency,
            min_throughput_fps=args.gate_throughput,
        )
        return gate.enforce(results)

    return 0


def _cmd_inspect(args) -> int:
    from isat.fingerprint.model import fingerprint_model

    model_path = args.model
    if not Path(model_path).exists():
        print(f"Error: model not found: {model_path}")
        return 1

    fp = fingerprint_model(model_path)

    if args.as_json:
        print(fp.to_json())
        return 0

    print(f"\n{'='*60}")
    print(f"  MODEL FINGERPRINT: {fp.name}")
    print(f"{'='*60}")
    print(f"  Path           : {fp.path}")
    print(f"  Opset          : {fp.opset}")
    print(f"  Parameters     : {fp.param_count:,}")
    print(f"  Size           : {fp.estimated_memory_mb:.1f} MB")
    print(f"  Size class     : {fp.size_class}")
    print(f"  Model class    : {fp.model_class}")
    print(f"  Nodes          : {fp.num_nodes}")
    print(f"  Initializers   : {fp.num_initializers}")
    print(f"  GEMM fraction  : {fp.gemm_fraction:.1%}")
    print(f"  Attention ops  : {'yes' if fp.has_attention else 'no'}")
    print(f"  Dynamic inputs : {'yes' if fp.has_dynamic_inputs else 'no'}")
    print(f"  External data  : {'yes' if fp.has_external_data else 'no'}")
    print(f"  Fingerprint    : {fp.fingerprint_hash}")
    print(f"\n  Inputs:")
    for name, shape in fp.input_shapes.items():
        print(f"    {name}: {shape}")
    print(f"\n  Outputs:")
    for name, shape in fp.output_shapes.items():
        print(f"    {name}: {shape}")
    print(f"\n  Top operators:")
    for op, count in list(fp.op_histogram.items())[:15]:
        print(f"    {op:<30} {count}")
    print(f"{'='*60}\n")
    return 0


def _cmd_hwinfo(args) -> int:
    from isat.fingerprint.hardware import fingerprint_hardware

    hw = fingerprint_hardware()

    if args.as_json:
        print(hw.to_json())
        return 0

    print(f"\n{'='*60}")
    print(f"  HARDWARE FINGERPRINT")
    print(f"{'='*60}")
    print(f"  GPU name       : {hw.gpu_name}")
    print(f"  GFX target     : {hw.gfx_target}")
    print(f"  CUs            : {hw.cu_count}")
    print(f"  Max clock      : {hw.max_clock_mhz} MHz")
    print(f"  Wavefront      : {hw.wavefront_size}")
    print(f"  LDS/CU         : {hw.lds_size_kb} KB")
    print(f"  VRAM           : {hw.vram_total_mb} MB")
    print(f"  GTT            : {hw.gtt_total_mb} MB")
    print(f"  System RAM     : {hw.system_ram_mb} MB")
    print(f"  APU            : {'yes' if hw.is_apu else 'no'}")
    print(f"  Unified memory : {'yes' if hw.unified_memory else 'no'}")
    print(f"  XNACK support  : {'yes' if hw.xnack_supported else 'no'}")
    print(f"  Memory class   : {hw.memory_class}")
    print(f"  ROCm version   : {hw.rocm_version}")
    print(f"  Kernel         : {hw.kernel_version}")
    if hw.gpu_temp_c is not None:
        print(f"  GPU temp       : {hw.gpu_temp_c:.1f} C")
    if hw.gpu_sclk_mhz is not None:
        print(f"  GPU clock      : {hw.gpu_sclk_mhz} MHz")
    print(f"  Fingerprint    : {hw.fingerprint_hash}")
    print(f"{'='*60}\n")
    return 0


def _cmd_history(args) -> int:
    from isat.database.store import ResultsDB

    if not Path(args.db).exists():
        print(f"No database found at {args.db}")
        return 1

    db = ResultsDB(args.db)
    if args.model:
        rows = db.best_for_model(args.model, limit=args.top)
    else:
        rows = db.all_runs()[:args.top]
    db.close()

    if not rows:
        print("No results found.")
        return 0

    if args.as_json:
        print(json.dumps(rows, indent=2, default=str))
        return 0

    print(f"\n{'Rank':<5} {'Model':<20} {'Config':<35} {'Mean ms':<10} {'P95 ms':<10}")
    print(f"{'-'*5} {'-'*20} {'-'*35} {'-'*10} {'-'*10}")
    for i, row in enumerate(rows, 1):
        print(f"{i:<5} {row['model_name'][:20]:<20} {row['config_label'][:35]:<35} "
              f"{row['mean_ms']:<10.2f} {row['p95_ms']:<10.2f}")
    print()
    return 0


def _cmd_export(args) -> int:
    from isat.database.store import ResultsDB
    from isat.fingerprint.hardware import fingerprint_hardware
    from isat.fingerprint.model import ModelFingerprint
    from isat.report.generator import ReportGenerator
    from isat.search.engine import CandidateConfig, TuneResult
    from isat.search.memory import MemoryConfig
    from isat.search.kernel import KernelConfig
    from isat.search.precision import PrecisionConfig
    from isat.search.graph import GraphConfig

    if not Path(args.db).exists():
        print(f"No database found at {args.db}")
        return 1

    db = ResultsDB(args.db)
    rows = db.all_runs(model_name=args.model)
    db.close()

    if not rows:
        print(f"No results for model '{args.model}'")
        return 1

    hw = fingerprint_hardware()
    model_fp = ModelFingerprint(name=args.model)

    results = []
    for row in rows:
        config = CandidateConfig(
            memory=MemoryConfig(),
            kernel=KernelConfig(),
            precision=PrecisionConfig(),
            graph=GraphConfig(),
            label=row["config_label"],
        )
        r = TuneResult(
            config=config,
            mean_latency_ms=row["mean_ms"] or float("inf"),
            p50_latency_ms=row["p50_ms"] or float("inf"),
            p95_latency_ms=row["p95_ms"] or float("inf"),
            p99_latency_ms=row["p99_ms"] or float("inf"),
            min_latency_ms=row["min_ms"] or float("inf"),
            max_latency_ms=row["max_ms"] or float("inf"),
            throughput_fps=row["throughput"] or 0.0,
            peak_gpu_temp_c=row["peak_temp"] or 0.0,
            peak_power_w=row["peak_power"] or 0.0,
        )
        if row.get("error"):
            r.error = row["error"]
        results.append(r)

    reporter = ReportGenerator(hw, model_fp, results, output_dir=args.output_dir)
    paths = reporter.generate_all()
    print(f"\nExported to {args.output_dir}/")
    return 0


def _cmd_compare(args) -> int:
    """Run head-to-head comparison of two configs with significance testing."""
    from isat.analysis.significance import compare_configs
    from isat.benchmark.runner import BenchmarkRunner
    from isat.fingerprint.hardware import fingerprint_hardware
    from isat.fingerprint.model import fingerprint_model
    from isat.search.engine import CandidateConfig
    from isat.search.memory import MemoryConfig
    from isat.search.kernel import KernelConfig
    from isat.search.precision import PrecisionConfig
    from isat.search.graph import GraphConfig

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    hw = fingerprint_hardware()
    model_fp = fingerprint_model(args.model)

    def _make_config(label):
        return CandidateConfig(
            memory=MemoryConfig(), kernel=KernelConfig(),
            precision=PrecisionConfig(), graph=GraphConfig(),
            label=label,
        )

    runner = BenchmarkRunner(hw, model_fp, args.model, warmup=3, runs=args.runs, cooldown=30)

    print(f"Running {args.runs} iterations for config A: {args.config_a}")
    result_a = runner.run_single(_make_config(args.config_a))
    runner.thermal.wait_cooldown()

    print(f"Running {args.runs} iterations for config B: {args.config_b}")
    result_b = runner.run_single(_make_config(args.config_b))

    latencies_a = [result_a.mean_latency_ms] * args.runs
    latencies_b = [result_b.mean_latency_ms] * args.runs

    result = compare_configs(latencies_a, latencies_b, confidence=args.confidence)

    print(f"\n{'='*60}")
    print(f"  STATISTICAL COMPARISON")
    print(f"{'='*60}")
    print(f"  Config A     : {args.config_a}")
    print(f"  Config B     : {args.config_b}")
    print(f"  Runs         : {args.runs}")
    print(f"  Confidence   : {args.confidence:.0%}")
    print(f"  Mean A       : {result.mean_a:.2f} ms")
    print(f"  Mean B       : {result.mean_b:.2f} ms")
    print(f"  t-statistic  : {result.t_statistic:.4f}")
    print(f"  p-value      : {result.p_value:.6f}")
    print(f"  Significant  : {'YES' if result.is_significant else 'NO'}")
    print(f"  Speedup      : {result.speedup:.3f}x")
    print(f"  95% CI       : [{result.ci_lower:.2f}, {result.ci_upper:.2f}] ms")
    print(f"{'='*60}")
    print(f"\n  {result.summary}")
    print()
    return 0


def _cmd_serve(args) -> int:
    """Launch the ISAT REST API server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required for the server.")
        print("Install with: pip install 'isat[server]'")
        return 1

    from isat.server.app import create_app

    print(BANNER)
    print(f"Starting ISAT API server on {args.host}:{args.port}")
    app = create_app(db_path=args.db, output_dir=args.output_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_triton(args) -> int:
    """Generate Triton Inference Server config from best historical result."""
    from isat.database.store import ResultsDB
    from isat.fingerprint.model import fingerprint_model
    from isat.integrations.triton import generate_triton_config
    from isat.search.engine import CandidateConfig, TuneResult
    from isat.search.memory import MemoryConfig
    from isat.search.kernel import KernelConfig
    from isat.search.precision import PrecisionConfig
    from isat.search.graph import GraphConfig

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    model_fp = fingerprint_model(args.model)

    if not Path(args.db).exists():
        print(f"No database found at {args.db}. Run 'isat tune' first.")
        return 1

    db = ResultsDB(args.db)
    rows = db.best_for_model(model_fp.name, limit=1)
    db.close()

    if not rows:
        print(f"No results for '{model_fp.name}'. Run 'isat tune' first.")
        return 1

    row = rows[0]
    env = json.loads(row.get("env_json", "{}"))
    config = CandidateConfig(
        memory=MemoryConfig(), kernel=KernelConfig(),
        precision=PrecisionConfig(), graph=GraphConfig(),
        label=row["config_label"],
    )
    for k, v in env.items():
        config.memory.env_overrides[k] = v

    best = TuneResult(
        config=config,
        mean_latency_ms=row["mean_ms"] or 10.0,
        p95_latency_ms=row["p95_ms"] or 10.0,
    )

    path = generate_triton_config(model_fp, best, args.output_dir, max_batch_size=args.max_batch)
    print(f"Triton config generated: {path}")
    return 0


def _cmd_profiles(args) -> int:
    """List available tuning profiles."""
    from isat.profiles.presets import PROFILES

    print(f"\n{'='*70}")
    print(f"  AVAILABLE TUNING PROFILES")
    print(f"{'='*70}")
    for name, p in PROFILES.items():
        print(f"\n  {name}")
        print(f"    {p.description}")
        print(f"    warmup={p.warmup}  runs={p.runs}  cooldown={p.cooldown}s"
              f"  max_configs={p.max_configs or 'all'}")
        print(f"    priority: {p.priority_objective}")
        if p.constraints:
            print(f"    constraints: {p.constraints}")
    print(f"\n{'='*70}")
    print(f"\n  Usage: isat tune model.onnx --profile <name>\n")
    return 0


def _cmd_optimize(args) -> int:
    """Optimize an ONNX model."""
    from isat.optimizer.pipeline import OptimizationPipeline

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    pipeline = OptimizationPipeline(output_dir=args.output_dir)
    do_all = args.all_transforms

    result = pipeline.optimize(
        args.model,
        simplify=do_all or args.simplify,
        convert_fp16=do_all or args.fp16,
        quantize_int8=do_all or args.int8,
        ort_optimize=do_all or args.ort_optimize,
    )

    print(f"\n{'='*60}")
    print(f"  MODEL OPTIMIZATION RESULT")
    print(f"{'='*60}")
    print(f"  Original      : {result.original_path}")
    print(f"  Optimized     : {result.optimized_path}")
    print(f"  Original size : {result.original_size_mb:.1f} MB ({result.original_nodes} nodes)")
    print(f"  Optimized size: {result.optimized_size_mb:.1f} MB ({result.optimized_nodes} nodes)")
    print(f"  Size reduction: {result.size_reduction_pct:.1f}%")
    print(f"  Node reduction: {result.node_reduction_pct:.1f}%")
    print(f"  Transforms    : {', '.join(result.transforms_applied)}")
    print(f"  Time          : {result.elapsed_s:.1f}s")
    if result.errors:
        print(f"  Warnings      : {'; '.join(result.errors)}")
    print(f"{'='*60}\n")
    return 0


def _cmd_stress(args) -> int:
    """Run stress tests."""
    from isat.stress.runner import StressTest

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    test = StressTest(args.model, provider=args.provider)

    if args.pattern == "sustained":
        result = test.sustained(duration_s=args.duration, concurrency=args.concurrency)
    elif args.pattern == "burst":
        result = test.burst(burst_size=args.concurrency)
    elif args.pattern == "ramp":
        result = test.ramp(end_concurrency=args.concurrency)
    else:
        result = test.sustained(duration_s=args.duration, concurrency=args.concurrency)

    print(f"\n{'='*60}")
    print(f"  STRESS TEST RESULTS ({result.pattern})")
    print(f"{'='*60}")
    print(f"  Total requests    : {result.total_requests}")
    print(f"  Successful        : {result.successful_requests}")
    print(f"  Failed            : {result.failed_requests}")
    print(f"  Success rate      : {result.success_rate:.1f}%")
    print(f"  Duration          : {result.total_duration_s:.1f}s")
    print(f"  Throughput        : {result.rps:.1f} req/s")
    print(f"  Peak concurrency  : {result.peak_concurrent}")
    if result.latency_stats:
        s = result.latency_stats
        print(f"  Mean latency      : {s.mean_ms:.2f} ms")
        print(f"  P95 latency       : {s.p95_ms:.2f} ms")
        print(f"  P99 latency       : {s.p99_ms:.2f} ms")
    if result.timeline:
        print(f"\n  Timeline:")
        for t in result.timeline:
            print(f"    {t}")
    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for e in result.errors[:5]:
            print(f"    {e}")
    print(f"{'='*60}\n")
    return 0


def _cmd_leak_check(args) -> int:
    """Run memory leak detection."""
    from isat.stress.leak_detector import MemoryLeakDetector

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    detector = MemoryLeakDetector(
        args.model, provider=args.provider, iterations=args.iterations,
    )
    report = detector.run()

    print(f"\n{'='*60}")
    print(f"  MEMORY LEAK DETECTION")
    print(f"{'='*60}")
    print(f"  Iterations        : {report.iterations}")
    print(f"  Duration          : {report.duration_s:.1f}s")
    print(f"  Host RSS delta    : {report.host_rss_delta_mb:+.1f} MB")
    print(f"  GPU VRAM delta    : {report.gpu_vram_delta_mb:+.1f} MB")
    print(f"  GPU GTT delta     : {report.gpu_gtt_delta_mb:+.1f} MB")
    print(f"  Leak rate         : {report.leak_rate_mb_per_1k:.2f} MB / 1K iterations")
    print(f"  Verdict           : {report.verdict}")
    print(f"{'='*60}\n")
    return 1 if report.leak_detected else 0


def _cmd_init(args) -> int:
    """Generate default config file."""
    from isat.config.loader import generate_default_config
    path = generate_default_config(args.output)
    print(f"Generated default config: {path}")
    print(f"Edit it, then run: isat tune --config {path}")
    return 0


def _cmd_zoo(args) -> int:
    """List pre-tuned model configurations."""
    from isat.model_zoo import list_supported, lookup

    if args.search:
        entry = lookup(args.search)
        if entry:
            print(f"\n{'='*60}")
            print(f"  PRE-TUNED CONFIG: {args.search}")
            print(f"{'='*60}")
            print(f"  Pattern     : {entry.model_pattern}")
            print(f"  Class       : {entry.model_class}")
            print(f"  Description : {entry.description}")
            print(f"  Precision   : {entry.recommended_precision}")
            print(f"  Provider    : {entry.recommended_provider}")
            print(f"  Target HW   : {entry.hardware_target}")
            if entry.estimated_latency_ms:
                print(f"  Est. latency: {entry.estimated_latency_ms:.0f} ms")
            print(f"\n  Recommended environment:")
            for k, v in entry.recommended_env.items():
                print(f"    export {k}={v}")
            if entry.notes:
                print(f"\n  Notes: {entry.notes}")
            print(f"{'='*60}\n")
        else:
            print(f"No pre-tuned config found for '{args.search}'")
        return 0

    models = list_supported()
    print(f"\n{'='*70}")
    print(f"  ISAT MODEL ZOO -- Pre-tuned Configurations")
    print(f"{'='*70}")
    print(f"\n  {'Pattern':<25} {'Class':<15} {'Precision':<10} {'Description'}")
    print(f"  {'-'*25} {'-'*15} {'-'*10} {'-'*30}")
    for m in models:
        print(f"  {m['pattern']:<25} {m['class']:<15} {m['precision']:<10} {m['description']}")
    print(f"\n  Usage: isat zoo --search <model_name>")
    print(f"{'='*70}\n")
    return 0


def _cmd_doctor(args) -> int:
    """System health and compatibility check."""
    from isat.compat.scanner import CompatScanner
    from isat.health.checker import HealthChecker

    print(f"\n{'='*60}")
    print(f"  ISAT DOCTOR -- System Compatibility & Health")
    print(f"{'='*60}")

    scanner = CompatScanner(model_path=args.model)
    compat = scanner.scan()
    print(f"\n  Compatibility Checks:")
    print(compat.summary())

    checker = HealthChecker()
    health = checker.check()
    print(f"\n  Health Checks:")
    print(health.summary())

    print(f"{'='*60}\n")
    return 0 if compat.ok and health.ready else 1


def _cmd_profile(args) -> int:
    """Latency decomposition profiling."""
    from isat.profiler.latency import LatencyProfiler

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    profiler = LatencyProfiler(
        args.model, provider=args.provider, steady_state_runs=args.runs, warmup=args.warmup
    )
    breakdown = profiler.profile()

    print(f"\n{'='*60}")
    print(f"  LATENCY BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Model          : {breakdown.model_path}")
    print(f"  Size           : {breakdown.model_size_mb:.1f} MB")
    print(f"  Provider       : {breakdown.provider}")
    print(f"  Peak RSS       : {breakdown.peak_rss_mb:.0f} MB")
    print(f"  GPU VRAM (post): {breakdown.gpu_vram_after_load_mb:.0f} MB")
    print()
    print(breakdown.summary_table())
    print(f"\n  Steady state: {breakdown.steady_state_mean_ms:.2f} ms "
          f"(P50={breakdown.steady_state_p50_ms:.2f} "
          f"P95={breakdown.steady_state_p95_ms:.2f} "
          f"P99={breakdown.steady_state_p99_ms:.2f})")
    print(f"{'='*60}\n")
    return 0


def _cmd_diff(args) -> int:
    """Structural model diff."""
    from isat.diff.model_diff import ModelDiff

    for p in [args.model_a, args.model_b]:
        if not Path(p).exists():
            print(f"Error: model not found: {p}")
            return 1

    differ = ModelDiff()
    result = differ.compare(args.model_a, args.model_b)

    print(f"\n{'='*70}")
    print(f"  MODEL DIFF")
    print(f"{'='*70}")
    print(f"  Model A: {result.model_a}")
    print(f"  Model B: {result.model_b}")
    if result.identical:
        print(f"\n  Models are structurally IDENTICAL")
    else:
        print()
        print(result.summary())
    print(f"{'='*70}\n")
    return 0


def _cmd_cost(args) -> int:
    """Cloud cost estimation."""
    from isat.cost.estimator import CostEstimator

    estimator = CostEstimator()

    if args.list_gpus:
        gpus = estimator.list_gpus()
        print(f"\n{'='*70}")
        print(f"  AVAILABLE GPU PRICING")
        print(f"{'='*70}")
        print(f"\n  {'GPU':<20} {'$/hr':>8} {'Provider':<10} {'Instance'}")
        print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*25}")
        for g in gpus:
            print(f"  {g['name']:<20} ${g['hourly']:>6.2f} {g['provider']:<10} {g['instance']}")
        print(f"{'='*70}\n")
        return 0

    estimate = estimator.estimate(args.latency, gpu_type=args.gpu, batch_size=args.batch_size)
    print(f"\n{'='*60}")
    print(f"  INFERENCE COST ESTIMATE")
    print(f"{'='*60}")
    print(estimate.summary())
    print(f"{'='*60}\n")
    return 0


def _cmd_sla(args) -> int:
    """SLA validation."""
    from isat.sla.validator import SLAValidator

    if args.list_templates:
        templates = SLAValidator.list_templates()
        print(f"\n{'='*60}")
        print(f"  SLA TEMPLATES")
        print(f"{'='*60}")
        for name, reqs in templates.items():
            print(f"\n  [{name}]")
            for r in reqs:
                print(f"    {r['name']}: {r['metric']} {r['op']} {r['threshold']} {r['unit']}")
        print(f"{'='*60}\n")
        return 0

    validator = SLAValidator(template=args.template)
    metrics = {}
    if args.p50: metrics["p50_ms"] = args.p50
    if args.p95: metrics["p95_ms"] = args.p95
    if args.p99: metrics["p99_ms"] = args.p99
    if args.throughput: metrics["throughput_rps"] = args.throughput
    if args.memory: metrics["memory_mb"] = args.memory
    if args.cold_start: metrics["cold_start_ms"] = args.cold_start

    if not metrics:
        print("Error: provide at least one metric (--p50, --p95, --p99, --throughput, --memory, --cold-start)")
        return 1

    result = validator.validate(metrics)
    print(f"\n{'='*70}")
    print(f"  SLA VALIDATION (template: {args.template})")
    print(f"{'='*70}")
    print(result.summary())
    print(f"{'='*70}\n")
    return 1 if not result.all_passed else 0


def _cmd_warmup(args) -> int:
    """Warmup analysis."""
    from isat.warmup.analyzer import WarmupAnalyzer

    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    analyzer = WarmupAnalyzer(
        args.model, provider=args.provider, max_iterations=args.max_iterations,
    )
    profile = analyzer.analyze()

    print(f"\n{'='*60}")
    print(f"  WARMUP ANALYSIS")
    print(f"{'='*60}")
    print(profile.summary())
    print(f"{'='*60}\n")
    return 0


def _cmd_cache(args) -> int:
    """Cache management."""
    from isat.cache.manager import CacheManager

    mgr = CacheManager()

    if args.action == "stats":
        stats = mgr.stats()
        print(f"\n{'='*60}")
        print(f"  COMPILATION CACHE")
        print(f"{'='*60}")
        print(stats.summary())
        print(f"{'='*60}\n")

    elif args.action == "clean":
        result = mgr.clean(max_age_hours=args.max_age_hours, dry_run=args.dry_run)
        prefix = "[DRY RUN] " if result["dry_run"] else ""
        print(f"\n{prefix}Removed {result['entries_removed']} entries, freed {result['space_freed_mb']:.1f} MB")

    elif args.action == "warm":
        if not args.model:
            print("Error: --model required for cache warm")
            return 1
        elapsed = mgr.warm(args.model)
        print(f"\nCache warmed in {elapsed:.1f}s")

    return 0


def _cmd_migrate(args) -> int:
    """Provider migration planning."""
    from isat.migration.tool import MigrationTool

    tool = MigrationTool()

    import os
    current_env = {k: v for k, v in os.environ.items()
                   if any(x in k.upper() for x in ["MIGRAPHX", "HSA", "ORT_TENSOR", "CUDA"])}

    plan = tool.plan(args.source, args.target, current_env=current_env)

    print(f"\n{'='*70}")
    print(f"  MIGRATION PLAN")
    print(f"{'='*70}")
    print(plan.summary())
    print(f"{'='*70}\n")
    return 0


def _cmd_shapes(args) -> int:
    from isat.shapes.handler import DynamicShapeHandler
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    handler = DynamicShapeHandler(args.model, provider=args.provider, runs=args.runs)
    profile = handler.sweep()
    print(f"\n{'='*70}")
    print(f"  DYNAMIC SHAPE ANALYSIS")
    print(f"{'='*70}")
    print(profile.summary())
    print(f"{'='*70}\n")
    return 0


def _cmd_download(args) -> int:
    from isat.hub.downloader import ModelHub
    hub = ModelHub()
    if args.list_models:
        models = hub.list_available()
        print(f"\n{'='*60}")
        print(f"  AVAILABLE MODELS")
        print(f"{'='*60}")
        for m in models:
            print(f"  {m['name']:<20} opset {m['opset']:<4} {m['description']}")
        cached = hub.list_cached()
        if cached:
            print(f"\n  Cached locally:")
            for c in cached:
                print(f"    {c['name']} ({c['size_mb']:.1f} MB)")
        print(f"{'='*60}\n")
        return 0
    result = hub.download(args.model_name, output_dir=args.output_dir)
    status = "cached" if result.cached else "downloaded"
    print(f"\n  {result.model_name} ({result.size_mb:.1f} MB) [{status}]")
    print(f"  Path: {result.local_path}\n")
    return 0


def _cmd_power(args) -> int:
    from isat.power.profiler import PowerProfiler
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    profiler = PowerProfiler(args.model, provider=args.provider, runs=args.runs)
    profile = profiler.profile()
    print(f"\n{'='*60}")
    print(f"  POWER EFFICIENCY PROFILE")
    print(f"{'='*60}")
    print(profile.summary())
    print(f"{'='*60}\n")
    return 0


def _cmd_memory(args) -> int:
    from isat.memory.planner import MemoryPlanner
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    planner = MemoryPlanner(args.model)
    plan = planner.plan()
    print(f"\n{'='*70}")
    print(f"  MEMORY PLAN")
    print(f"{'='*70}")
    print(plan.summary())
    print(f"{'='*70}\n")
    return 0


def _cmd_abtest(args) -> int:
    from isat.abtesting.compare import ABTest
    for p in [args.model_a, args.model_b]:
        if not Path(p).exists():
            print(f"Error: model not found: {p}")
            return 1
    test = ABTest(args.model_a, args.model_b, provider=args.provider, runs=args.runs)
    result = test.run(name_a=args.name_a, name_b=args.name_b)
    print(f"\n{'='*65}")
    print(f"  A/B TEST RESULTS")
    print(f"{'='*65}")
    print(result.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_visualize(args) -> int:
    from isat.visualizer.graph import GraphVisualizer
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    viz = GraphVisualizer(args.model)
    if args.format == "dot":
        output = args.output or "graph.dot"
        viz.to_dot(output)
        print(f"DOT graph saved to {output}")
        print(f"Render with: dot -Tpng {output} -o graph.png")
    elif args.format == "histogram":
        print(viz.op_histogram())
    else:
        print(viz.to_ascii())
    return 0


def _cmd_snapshot(args) -> int:
    from isat.snapshot.capture import EnvSnapshot
    snap = EnvSnapshot()
    data = snap.capture(model_path=args.model)
    path = snap.save(data, args.output)
    print(f"\n  Snapshot saved to {path}")
    print(f"  System   : {data['system']['os']} {data['system']['os_release']}")
    print(f"  Python   : {data['python']['version']}")
    print(f"  ISAT     : {data['isat_version']}")
    print(f"  Packages : {', '.join(f'{k}={v}' for k,v in data['software'].items())}")
    if data.get("model"):
        print(f"  Model    : {data['model']['path']} (SHA256: {data['model']['sha256'][:16]}...)")
    print()
    return 0


def _cmd_batch(args) -> int:
    from isat.scheduler.batch import BatchScheduler
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    sizes = [int(x) for x in args.sizes.split(",")]
    scheduler = BatchScheduler(args.model, provider=args.provider, batch_sizes=sizes)
    profile = scheduler.profile()
    print(f"\n{'='*65}")
    print(f"  BATCH SIZE ANALYSIS")
    print(f"{'='*65}")
    print(profile.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_scan(args) -> int:
    from isat.scanner.checker import ModelScanner
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    scanner = ModelScanner()
    result = scanner.scan(args.model)
    print(f"\n{'='*70}")
    print(f"  ONNX MODEL SECURITY & COMPLIANCE SCAN")
    print(f"{'='*70}")
    print(result.summary())
    print(f"{'='*70}\n")
    return 0 if result.passed else 1


def _cmd_regression(args) -> int:
    from isat.regression.detector import RegressionDetector
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1

    detector = RegressionDetector(db_path=args.db)

    if args.history:
        from isat.fingerprint.model import fingerprint_model
        fp = fingerprint_model(args.model)
        hist = detector.history(fp.name)
        print(f"\n{'='*70}")
        print(f"  REGRESSION HISTORY: {fp.name}")
        print(f"{'='*70}")
        print(hist.summary())
        print(f"{'='*70}\n")
        return 0

    import numpy as np
    import onnxruntime as ort

    from isat.fingerprint.model import fingerprint_model
    fp = fingerprint_model(args.model)

    session = ort.InferenceSession(
        args.model,
        providers=[args.provider, "CPUExecutionProvider"],
    )
    feed = {}
    for inp in session.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        if "int" in inp.type.lower():
            feed[inp.name] = np.ones(shape, dtype=np.int64)
        elif "float16" in inp.type.lower():
            feed[inp.name] = np.random.randn(*shape).astype(np.float16)
        else:
            feed[inp.name] = np.random.randn(*shape).astype(np.float32)

    for _ in range(3):
        session.run(None, feed)

    latencies = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)

    if args.set_baseline:
        detector.save_baseline(fp.name, latencies, is_baseline=True)
        mean = float(np.mean(latencies))
        print(f"\nBaseline saved: {fp.name} = {mean:.2f} ms ({args.runs} runs)\n")
        return 0

    result = detector.check(fp.name, latencies, threshold_pct=args.threshold)
    print(f"\n{'='*70}")
    print(f"  REGRESSION CHECK: {fp.name}")
    print(f"{'='*70}")
    print(result.summary())
    print(f"{'='*70}\n")
    return 1 if result.regressed else 0


def _cmd_compat_matrix(args) -> int:
    from isat.compat_matrix.matrix import CompatMatrix
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    matrix = CompatMatrix()
    result = matrix.check(args.model)
    print(f"\n{'='*75}")
    print(f"  OPERATOR COMPATIBILITY MATRIX")
    print(f"{'='*75}")
    print(result.summary())
    print(f"{'='*75}\n")
    return 0


def _cmd_thermal(args) -> int:
    from isat.thermal.monitor import ThermalMonitor
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    monitor = ThermalMonitor(args.model, provider=args.provider, runs=args.runs)
    profile = monitor.monitor()
    print(f"\n{'='*65}")
    print(f"  THERMAL THROTTLE ANALYSIS")
    print(f"{'='*65}")
    print(profile.summary())
    print(f"{'='*65}\n")
    return 1 if profile.throttled else 0


def _cmd_quant_sensitivity(args) -> int:
    from isat.quant_sensitivity.analyzer import QuantSensitivityAnalyzer
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    analyzer = QuantSensitivityAnalyzer(args.model)
    report = analyzer.analyze()
    print(f"\n{'='*70}")
    print(f"  QUANTIZATION SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(report.summary())
    print(f"{'='*70}\n")
    return 0


def _cmd_pipeline(args) -> int:
    from isat.pipeline.optimizer import PipelineOptimizer
    stages = []
    for entry in args.models:
        if ":" in entry:
            name, path = entry.split(":", 1)
        else:
            name = Path(entry).stem
            path = entry
        stages.append((name, path))

    optimizer = PipelineOptimizer(stages, provider=args.provider, runs=args.runs)
    profile = optimizer.profile()
    print(f"\n{'='*75}")
    print(f"  PIPELINE ANALYSIS")
    print(f"{'='*75}")
    print(profile.summary())
    print(f"{'='*75}\n")
    return 0


def _cmd_recommend(args) -> int:
    from isat.recommend.advisor import HardwareAdvisor
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    advisor = HardwareAdvisor()
    report = advisor.recommend(
        args.model,
        max_latency_ms=args.max_latency,
        max_cost_hr=args.max_cost,
        prefer_amd=args.prefer_amd,
    )
    print(f"\n{'='*80}")
    print(f"  HARDWARE RECOMMENDATIONS")
    print(f"{'='*80}")
    print(report.summary())
    print(f"{'='*80}\n")
    return 0


def _cmd_registry(args) -> int:
    from isat.registry.store import ModelRegistry
    registry = ModelRegistry(db_path=args.db)

    if args.action == "register":
        if not args.model_name or not args.version or not args.model_path:
            print("Error: --model-name, --version, and --model-path required")
            return 1
        v = registry.register(args.model_name, args.version, args.model_path)
        print(f"\nRegistered: {v.model_name} v{v.version} (SHA256: {v.sha256[:16]}...)\n")
        return 0

    elif args.action == "list":
        listing = registry.list_models(model_name=args.model_name, stage=args.stage if args.stage != "production" else "")
        print(f"\n{'='*70}")
        print(f"  MODEL REGISTRY")
        print(f"{'='*70}")
        print(listing.summary())
        print(f"{'='*70}\n")
        return 0

    elif args.action == "promote":
        if not args.model_name or not args.version:
            print("Error: --model-name and --version required")
            return 1
        registry.promote(args.model_name, args.version, stage=args.stage)
        print(f"\nPromoted: {args.model_name} v{args.version} -> {args.stage}\n")
        return 0

    elif args.action == "diff":
        if not args.model_name or not args.version or not args.version_b:
            print("Error: --model-name, --version, and --version-b required")
            return 1
        diff = registry.diff_versions(args.model_name, args.version, args.version_b)
        print(f"\n{'='*60}")
        print(f"  VERSION DIFF")
        print(f"{'='*60}")
        print(diff.summary())
        print(f"{'='*60}\n")
        return 0

    elif args.action == "show":
        if not args.model_name or not args.version:
            print("Error: --model-name and --version required")
            return 1
        v = registry.get_version(args.model_name, args.version)
        if not v:
            print(f"Version not found: {args.model_name} v{args.version}")
            return 1
        print(f"\n  {v.model_name} v{v.version}")
        print(f"  Stage  : {v.stage}")
        print(f"  Path   : {v.model_path}")
        print(f"  SHA256 : {v.sha256[:32]}...")
        print(f"  Config : {json.dumps(v.config, indent=2)}")
        if v.metrics:
            print(f"  Metrics: {json.dumps(v.metrics, indent=2)}")
        print()
        return 0

    return 0


def _cmd_trace(args) -> int:
    from isat.tracing.tracer import InferenceTracer
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    tracer = InferenceTracer()
    traces = tracer.trace_inference(args.model, provider=args.provider, runs=args.runs)
    stats = tracer.get_stats()
    print(f"\n{'='*65}")
    print(f"  INFERENCE TRACE ({args.runs} requests)")
    print(f"{'='*65}")
    for t in traces[:3]:
        print(t.summary())
        print()
    if len(traces) > 3:
        print(f"  ... and {len(traces) - 3} more traces")
    print(f"\n  Aggregate: e2e mean={stats['e2e_mean_ms']:.2f}ms "
          f"p95={stats['e2e_p95_ms']:.2f}ms "
          f"inference={stats['inference_mean_ms']:.2f}ms")
    if args.export:
        path = tracer.export_otlp_json(args.export)
        print(f"  Exported to: {path}")
    print(f"{'='*65}\n")
    return 0


def _cmd_canary(args) -> int:
    from isat.canary.deployer import CanaryDeployer
    for p in [args.baseline, args.canary_model]:
        if not Path(p).exists():
            print(f"Error: model not found: {p}")
            return 1
    deployer = CanaryDeployer(
        args.baseline, args.canary_model, provider=args.provider,
        requests_per_phase=args.requests_per_phase,
        max_error_rate=args.max_error_rate,
        max_latency_increase_pct=args.max_latency_increase,
    )
    print(f"Running canary deployment...")
    result = deployer.deploy()
    print(f"\n{'='*70}")
    print(f"  CANARY DEPLOYMENT RESULT")
    print(f"{'='*70}")
    print(result.summary())
    print(f"{'='*70}\n")
    return 1 if result.rolled_back else 0


def _cmd_alerts(args) -> int:
    from isat.alerts.engine import AlertEngine
    engine = AlertEngine()
    if args.action == "list":
        rules = engine.list_builtin()
        print(f"\n{'='*70}")
        print(f"  BUILTIN ALERT RULES")
        print(f"{'='*70}")
        print(f"\n  {'Name':<30} {'Metric':<20} {'Op':>3} {'Threshold':>10} {'Severity':<10}")
        print(f"  {'-'*30} {'-'*20} {'-'*3} {'-'*10} {'-'*10}")
        for r in rules:
            print(f"  {r['name']:<30} {r['metric']:<20} {r['operator']:>3} "
                  f"{r['threshold']:>10.1f} {r['severity']:<10}")
        print(f"{'='*70}\n")
        return 0
    elif args.action == "check":
        if not args.metrics_json:
            print("Error: --metrics-json required for check")
            return 1
        metrics = json.loads(args.metrics_json)
        alerts = engine.check(metrics)
        status = engine.status()
        print(f"\n{'='*65}")
        print(f"  ALERT CHECK")
        print(f"{'='*65}")
        print(status.summary())
        print(f"{'='*65}\n")
        return 1 if alerts else 0
    elif args.action == "export":
        engine.export_rules(args.output)
        print(f"Exported {len(engine.rules)} rules to {args.output}")
        return 0
    return 0


def _cmd_surgery(args) -> int:
    from isat.surgery.graph import GraphSurgeon
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    surgeon = GraphSurgeon(args.model)
    has_ops = False
    for op in args.remove_op:
        count = surgeon.remove_op_type(op)
        print(f"  Removed {count} '{op}' nodes")
        has_ops = True
    if args.remove_unused:
        count = surgeon.remove_unused_initializers()
        print(f"  Removed {count} unused initializers")
        has_ops = True
    for old, new in args.rename_input:
        surgeon.rename_input(old, new)
        has_ops = True
    for old, new in args.rename_output:
        surgeon.rename_output(old, new)
        has_ops = True
    if args.change_opset:
        surgeon.change_opset(args.change_opset)
        has_ops = True
    if not has_ops:
        stats = surgeon.get_stats()
        print(f"\n  Graph stats: {stats['nodes']} nodes, {stats['initializers']} initializers")
        print(f"  Inputs : {stats['inputs']}")
        print(f"  Outputs: {stats['outputs']}")
        print(f"  Ops    : {', '.join(sorted(stats['ops']))}\n")
        return 0
    output = args.output or args.model.replace(".onnx", "_surgery.onnx")
    result = surgeon.save(output)
    print(f"\n{'='*65}")
    print(f"  GRAPH SURGERY RESULT")
    print(f"{'='*65}")
    print(result.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_guard(args) -> int:
    from isat.guard.validator import InputGuard
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    guard = InputGuard(model_path=args.model)
    if args.show_schema:
        print(f"\n{'='*65}")
        print(f"  INPUT SCHEMA: {Path(args.model).name}")
        print(f"{'='*65}")
        for s in guard.schemas:
            print(f"  {s.name:<30} shape={s.shape}  dtype={s.dtype}")
        print(f"{'='*65}\n")
        return 0
    import numpy as np
    feed = {}
    for s in guard.schemas:
        shape = [d if isinstance(d, int) else 1 for d in s.shape]
        dtype_map = {"float32": np.float32, "float16": np.float16,
                     "int64": np.int64, "int32": np.int32}
        dtype = dtype_map.get(s.dtype, np.float32)
        feed[s.name] = np.random.randn(*shape).astype(dtype)
    result = guard.validate(feed)
    print(f"\n{'='*65}")
    print(f"  INPUT VALIDATION")
    print(f"{'='*65}")
    print(result.summary())
    print(f"{'='*65}\n")
    return 0 if result.valid else 1


def _cmd_ensemble(args) -> int:
    from isat.ensemble.runner import ModelEnsemble
    models = []
    for entry in args.models:
        parts = entry.split(":")
        name = parts[0]
        path = parts[1] if len(parts) > 1 else parts[0]
        weight = float(parts[2]) if len(parts) > 2 else 1.0
        models.append((name, path, weight))
    ensemble = ModelEnsemble(models, provider=args.provider, strategy=args.strategy)
    result = ensemble.run(runs=args.runs)
    print(f"\n{'='*65}")
    print(f"  MODEL ENSEMBLE RESULT")
    print(f"{'='*65}")
    print(result.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_gpu_frag(args) -> int:
    from isat.gpu_frag.analyzer import FragmentationAnalyzer
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    analyzer = FragmentationAnalyzer(args.model, provider=args.provider, runs=args.runs)
    report = analyzer.analyze()
    print(f"\n{'='*65}")
    print(f"  GPU MEMORY FRAGMENTATION ANALYSIS")
    print(f"{'='*65}")
    print(report.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_prune(args) -> int:
    from isat.pruning.pruner import ModelPruner
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    pruner = ModelPruner(args.model)
    if args.analyze_only:
        info = pruner.analyze_sparsity()
        print(f"\n{'='*65}")
        print(f"  SPARSITY ANALYSIS")
        print(f"{'='*65}")
        print(f"  Total params    : {info['total_params']:,}")
        print(f"  Total zeros     : {info['total_zeros']:,}")
        print(f"  Overall sparsity: {info['overall_sparsity']:.1%}")
        for l in info["layers"][:15]:
            print(f"    {l['name'][:40]:<40} sparsity={l['sparsity']:.1%}")
        print(f"{'='*65}\n")
        return 0
    result = pruner.prune(strategy=args.strategy, sparsity=args.sparsity, output_path=args.output)
    print(f"\n{'='*65}")
    print(f"  MODEL PRUNING RESULT")
    print(f"{'='*65}")
    print(result.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_distill(args) -> int:
    from isat.distillation.helper import DistillationHelper
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    helper = DistillationHelper(args.model)
    plan = helper.plan()
    print(f"\n{'='*65}")
    print(f"  DISTILLATION PLAN")
    print(f"{'='*65}")
    print(plan.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_fusion(args) -> int:
    from isat.fusion.analyzer import FusionAnalyzer
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    analyzer = FusionAnalyzer(args.model)
    report = analyzer.analyze()
    print(f"\n{'='*65}")
    print(f"  OPERATOR FUSION ANALYSIS")
    print(f"{'='*65}")
    print(report.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_attention(args) -> int:
    from isat.attention.profiler import AttentionProfiler
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    profiler = AttentionProfiler(args.model)
    report = profiler.profile(head_dim=args.head_dim)
    print(f"\n{'='*65}")
    print(f"  ATTENTION HEAD ANALYSIS")
    print(f"{'='*65}")
    print(report.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_llm_bench(args) -> int:
    from isat.llm_bench.benchmarker import LLMBenchmarker
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    bench = LLMBenchmarker(
        args.model, provider=args.provider,
        sequence_lengths=seq_lengths, decode_steps=args.decode_steps,
    )
    result = bench.benchmark(runs=args.runs)
    print(f"\n{'='*70}")
    print(f"  LLM TOKEN THROUGHPUT BENCHMARK")
    print(f"{'='*70}")
    print(result.summary())
    print(f"{'='*70}\n")
    return 0


def _cmd_compiler_compare(args) -> int:
    from isat.compiler_compare.comparator import CompilerComparator
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    comp = CompilerComparator(args.model)
    report = comp.compare(runs=args.runs)
    print(f"\n{'='*75}")
    print(f"  COMPILER / PROVIDER COMPARISON")
    print(f"{'='*75}")
    print(report.summary())
    print(f"{'='*75}\n")
    return 0


def _cmd_replay(args) -> int:
    if args.action == "record":
        from isat.replay.recorder import InferenceRecorder
        if not Path(args.model).exists():
            print(f"Error: model not found: {args.model}")
            return 1
        rec = InferenceRecorder(args.dir)
        count = rec.record_from_model(args.model, provider=args.provider, num_requests=args.num_requests)
        print(f"Recorded {count} requests to {args.dir}/")
        return 0
    else:
        from isat.replay.recorder import InferenceReplayer
        if not Path(args.dir).exists():
            print(f"Error: recording not found: {args.dir}")
            return 1
        replayer = InferenceReplayer(args.dir)
        result = replayer.replay(args.model, provider=args.provider)
        print(f"\n{'='*65}")
        print(f"  INFERENCE REPLAY RESULT")
        print(f"{'='*65}")
        print(result.summary())
        print(f"{'='*65}\n")
        return 0


def _cmd_drift(args) -> int:
    from isat.output_monitor.drift import OutputMonitor
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    monitor = OutputMonitor(
        args.model, provider=args.provider,
        baseline_runs=args.baseline_runs, monitor_runs=args.monitor_runs,
    )
    report = monitor.monitor()
    print(f"\n{'='*65}")
    print(f"  OUTPUT DRIFT MONITOR")
    print(f"{'='*65}")
    print(report.summary())
    print(f"{'='*65}\n")
    return 1 if report.drift_detected else 0


def _cmd_weight_sharing(args) -> int:
    from isat.weight_analysis.sharing import WeightSharingDetector
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    detector = WeightSharingDetector(args.model, similarity_threshold=args.similarity)
    report = detector.analyze()
    print(f"\n{'='*65}")
    print(f"  WEIGHT SHARING ANALYSIS")
    print(f"{'='*65}")
    print(report.summary())
    print(f"{'='*65}\n")
    return 0


def _cmd_codegen(args) -> int:
    from isat.codegen.generator import CppCodeGenerator
    if not Path(args.model).exists():
        print(f"Error: model not found: {args.model}")
        return 1
    gen = CppCodeGenerator(args.model)
    result = gen.generate(output_dir=args.output_dir)
    print(f"\n{'='*65}")
    print(f"  C++ CODE GENERATION")
    print(f"{'='*65}")
    print(result.summary())
    print(f"{'='*65}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
