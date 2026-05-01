"""ISAT command-line interface.

Usage:
    isat tune                          -- detect hardware, show inference recommendations
    isat tune      MODEL.onnx          -- detect hw + recommend + run full auto-tune search
    isat tune      --detect-only       -- hardware detection only (no model needed)
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
from isat.utils import ort_providers

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


def _detect_provider() -> str:
    """Auto-detect the best available ORT execution provider."""
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return "CPUExecutionProvider"
    preference = [
        "MIGraphXExecutionProvider",
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ]
    for ep in preference:
        if ep in available:
            return ep
    return available[0] if available else "CPUExecutionProvider"


_DEFAULT_PROVIDER = _detect_provider()


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
    p_tune = sub.add_parser("tune", help="Auto-detect hardware + tune an ONNX model")
    p_tune.add_argument("model", nargs="?", default=None,
                        help="Path to .onnx model (omit for hardware detection only)")
    p_tune.add_argument("--detect-only", action="store_true",
                        help="Only detect hardware and show recommendations (no benchmarking)")
    p_tune.add_argument("--json", action="store_true", dest="tune_json",
                        help="Output hardware detection as JSON")
    p_tune.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    p_tune.add_argument("--runs", type=int, default=5, help="Measured iterations (default: 5)")
    p_tune.add_argument("--cooldown", type=float, default=60.0, help="Cooldown between configs (seconds)")
    p_tune.add_argument("--max-configs", type=int, default=0, help="Limit configs to test (0=unlimited)")
    p_tune.add_argument("--provider", default=_DEFAULT_PROVIDER,
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
    p_stress.add_argument("--provider", default=_DEFAULT_PROVIDER)

    # ── leak-check ────────────────────────────────────────────
    p_leak = sub.add_parser("leak-check", help="Detect memory leaks during inference")
    p_leak.add_argument("model", help="Path to .onnx model")
    p_leak.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    p_leak.add_argument("--provider", default=_DEFAULT_PROVIDER)

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
    p_prof.add_argument("--provider", default=_DEFAULT_PROVIDER)

    # ── diff ─────────────────────────────────────────────────
    p_diff = sub.add_parser("diff", help="Structural diff between two ONNX models")
    p_diff.add_argument("model_a", help="First model")
    p_diff.add_argument("model_b", help="Second model")

    # ── cost ─────────────────────────────────────────────────
    p_cost = sub.add_parser("cost", help="Estimate cloud inference cost")
    p_cost.add_argument("--latency", type=float, default=None, help="Mean latency in ms")
    p_cost.add_argument("--gpu", default="a10g", help="GPU type (a10g, t4, a100_40gb, h100_80gb, etc.)")
    p_cost.add_argument("--batch-size", type=int, default=1)
    p_cost.add_argument("--list-gpus", action="store_true", help="List available GPU pricing")

    # ── sla ──────────────────────────────────────────────────
    p_sla = sub.add_parser("sla", help="Validate inference against SLA requirements")
    p_sla.add_argument("--template", choices=["realtime", "batch", "edge", "llm", "mobile"], default="realtime")
    p_sla.add_argument("--p50", type=float, default=None, help="P50 latency ms")
    p_sla.add_argument("--p95", type=float, default=None, help="P95 latency ms")
    p_sla.add_argument("--p99", type=float, default=None, help="P99 latency ms")
    p_sla.add_argument("--throughput", type=float, default=None, help="Throughput RPS")
    p_sla.add_argument("--memory", type=float, default=None, help="Memory MB")
    p_sla.add_argument("--cold-start", type=float, default=None, help="Cold start ms")
    p_sla.add_argument("--list-templates", action="store_true")

    # ── warmup ───────────────────────────────────────────────
    p_warm = sub.add_parser("warmup", help="Analyze warmup behavior and find optimal iterations")
    p_warm.add_argument("model", help="Path to .onnx model")
    p_warm.add_argument("--max-iterations", type=int, default=100)
    p_warm.add_argument("--provider", default=_DEFAULT_PROVIDER)

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
    p_dl.add_argument("model_name", nargs="?", default=None,
                       help="Model name (resnet50, mobilenetv2, etc.) or URL")
    p_dl.add_argument("--output-dir", default=".", help="Output directory")
    p_dl.add_argument("--list", action="store_true", dest="list_models", help="List available models")

    # ── power ────────────────────────────────────────────────
    p_pow = sub.add_parser("power", help="Profile power efficiency (perf/watt, energy/inference)")
    p_pow.add_argument("model", help="Path to .onnx model")
    p_pow.add_argument("--provider", default=_DEFAULT_PROVIDER)
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

    # ── onnx (universal converter) ────────────────────────────
    p_onnx = sub.add_parser("onnx", help="Convert any model to ONNX (PyTorch, TF, HuggingFace, JAX, TFLite, SafeTensors)")
    p_onnx.add_argument("input", help="Model path, directory, or HuggingFace model ID")
    p_onnx.add_argument("--output", "-o", default=".", help="Output directory (default: current dir)")
    p_onnx.add_argument("--input-shape", default=None, help="Input shape for raw checkpoints (e.g. 1,3,224,224)")
    p_onnx.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    p_onnx.add_argument("--no-tune", action="store_true", help="Skip auto best-config after conversion")
    p_onnx.add_argument("--simplify", action="store_true", help="Simplify ONNX graph with onnxsim after conversion")

    # ── quantize ─────────────────────────────────────────────
    p_q = sub.add_parser("quantize", help="Advanced model quantization (INT4/INT8/FP16/SmoothQuant/mixed)")
    p_q.add_argument("model", help="Path to .onnx model")
    p_q.add_argument("--output", "-o", default=None, help="Output path (default: auto-named)")
    p_q.add_argument("--method", choices=["auto", "int8", "int4", "fp16", "mixed", "smooth"], default="auto")
    p_q.add_argument("--per-channel", action="store_true", default=True, help="Per-channel INT8 (default)")
    p_q.add_argument("--block-size", type=int, default=128, help="INT4 block size")
    p_q.add_argument("--alpha", type=float, default=0.5, help="SmoothQuant alpha")
    p_q.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis only")

    # ── stream ───────────────────────────────────────────────
    p_st = sub.add_parser("stream", help="Streaming token-by-token LLM inference with KV cache")
    p_st.add_argument("model", help="Path to .onnx LLM model")
    p_st.add_argument("--prompt", default="Hello, my name is", help="Input prompt")
    p_st.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer name/path")
    p_st.add_argument("--max-tokens", type=int, default=128, help="Max new tokens to generate")
    p_st.add_argument("--temperature", type=float, default=0.8)
    p_st.add_argument("--top-k", type=int, default=50)
    p_st.add_argument("--top-p", type=float, default=0.9)
    p_st.add_argument("--provider", default="CPUExecutionProvider")
    p_st.add_argument("--benchmark", action="store_true", help="Run generation benchmark")

    # ── shard ────────────────────────────────────────────────
    p_sh = sub.add_parser("shard", help="Split model into shards for multi-GPU / memory-constrained inference")
    p_sh.add_argument("model", help="Path to .onnx model")
    p_sh.add_argument("--num-shards", type=int, default=2, help="Number of shards")
    p_sh.add_argument("--output-dir", default=None, help="Output directory for shards")
    p_sh.add_argument("--strategy", choices=["balanced", "layer", "auto"], default="auto")
    p_sh.add_argument("--analyze", action="store_true", help="Show shard analysis only")
    p_sh.add_argument("--validate", action="store_true", help="Validate existing shards")

    # ── merge ────────────────────────────────────────────────
    p_mg = sub.add_parser("merge", help="Merge/compose multiple ONNX models into one")
    p_mg.add_argument("models", nargs="+", help="Paths to .onnx models to merge")
    p_mg.add_argument("--output", "-o", required=True, help="Output path for merged model")
    p_mg.add_argument("--mode", choices=["chain", "parallel"], default="chain")
    p_mg.add_argument("--aggregation", choices=["concat", "mean", "max", "sum"], default="concat",
                       help="Parallel aggregation mode")
    p_mg.add_argument("--validate", action="store_true", help="Validate merged model matches originals")

    # ── explain ──────────────────────────────────────────────
    p_ex = sub.add_parser("explain", help="Model explainability (feature importance, sensitivity, activations)")
    p_ex.add_argument("model", help="Path to .onnx model")
    p_ex.add_argument("--method", choices=["auto", "perturbation", "gradient", "sensitivity"], default="auto")
    p_ex.add_argument("--num-samples", type=int, default=50)
    p_ex.add_argument("--provider", default="CPUExecutionProvider")
    p_ex.add_argument("--layer", default=None, help="Specific layer to analyze")

    # ── benchmark-suite ──────────────────────────────────────
    p_bs = sub.add_parser("benchmark-suite", help="Comprehensive benchmark suite (latency, throughput, memory, scalability)")
    p_bs.add_argument("model", help="Path to .onnx model")
    p_bs.add_argument("--provider", default="auto")
    p_bs.add_argument("--output-dir", default=None, help="Directory for benchmark reports")
    p_bs.add_argument("--batch-sizes", default="1,2,4,8,16,32", help="Comma-separated batch sizes")
    p_bs.add_argument("--duration", type=int, default=30, help="Throughput test duration (seconds)")
    p_bs.add_argument("--latency-only", action="store_true")
    p_bs.add_argument("--throughput-only", action="store_true")
    p_bs.add_argument("--memory-only", action="store_true")

    # ── encrypt ──────────────────────────────────────────────
    p_en = sub.add_parser("encrypt", help="Encrypt, fingerprint, or protect ONNX model weights")
    p_en.add_argument("model", help="Path to .onnx model")
    p_en.add_argument("--output", "-o", required=True, help="Output path")
    p_en.add_argument("--method", choices=["encrypt", "decrypt", "obfuscate", "deobfuscate",
                                            "fingerprint", "verify"], default="encrypt")
    p_en.add_argument("--password", default=None, help="Encryption/decryption password")
    p_en.add_argument("--seed", type=int, default=None, help="Obfuscation seed")
    p_en.add_argument("--owner", default=None, help="Owner ID for fingerprinting")

    # ── safety ───────────────────────────────────────────────
    p_sf = sub.add_parser("safety", help="Model safety guardrails (PII, toxicity, jailbreak detection)")
    p_sf.add_argument("--input-text", default=None, help="Input text to check")
    p_sf.add_argument("--output-text", default=None, help="Output text to check")
    p_sf.add_argument("--input-file", default=None, help="File with input text")
    p_sf.add_argument("--output-file", default=None, help="File with output text")
    p_sf.add_argument("--check", choices=["all", "pii", "toxic", "jailbreak", "confidence", "format"],
                       default="all")

    # ── cloud-deploy ─────────────────────────────────────────
    p_cd = sub.add_parser("cloud-deploy", help="Generate cloud deployment artifacts (Docker, K8s, SageMaker, Azure, GCP)")
    p_cd.add_argument("model", help="Path to .onnx model")
    p_cd.add_argument("--output-dir", default=None, help="Output directory for deployment artifacts")
    p_cd.add_argument("--target", choices=["all", "docker", "kubernetes", "sagemaker", "azure", "gcp", "handler"],
                       default="all")
    p_cd.add_argument("--replicas", type=int, default=2, help="K8s replicas")
    p_cd.add_argument("--gpu", action="store_true", help="Enable GPU support in generated configs")

    # ── test ─────────────────────────────────────────────────
    p_t = sub.add_parser("test", help="Automated model testing (determinism, stability, edge cases, cross-provider)")
    p_t.add_argument("model", help="Path to .onnx model")
    p_t.add_argument("--provider", default="auto")
    p_t.add_argument("--output-dir", default=None, help="Output directory for test report")
    p_t.add_argument("--suite", choices=["all", "determinism", "stability", "edge", "input",
                                          "cross-provider", "latency", "memory", "golden"],
                      default="all")
    p_t.add_argument("--golden", default=None, help="Path to golden test file (.npz)")
    p_t.add_argument("--generate-golden", action="store_true", help="Generate golden test file")
    p_t.add_argument("--junit", action="store_true", help="Output JUnit XML for CI")

    # ── speculate ────────────────────────────────────────────
    p_sp = sub.add_parser("speculate", help="Speculative decoding: 2-4x LLM speedup via draft model + rejection sampling")
    p_sp.add_argument("target", help="Path to target .onnx LLM model")
    p_sp.add_argument("--draft", default=None, help="Path to draft .onnx model (smaller/faster)")
    p_sp.add_argument("--mode", choices=["draft", "self", "medusa"], default="draft")
    p_sp.add_argument("--num-speculative", type=int, default=5, help="Tokens to speculate per step")
    p_sp.add_argument("--prompt", default="The future of AI is", help="Input prompt")
    p_sp.add_argument("--tokenizer", default=None, help="Tokenizer name/path")
    p_sp.add_argument("--max-tokens", type=int, default=128)
    p_sp.add_argument("--temperature", type=float, default=0.8)
    p_sp.add_argument("--provider", default="CPUExecutionProvider")
    p_sp.add_argument("--benchmark", action="store_true", help="Run speculative vs naive benchmark")

    # ── serve-llm ────────────────────────────────────────────
    p_sl = sub.add_parser("serve-llm", help="Continuous batching LLM server with PagedAttention (OpenAI-compatible)")
    p_sl.add_argument("model", help="Path to .onnx LLM model")
    p_sl.add_argument("--port", type=int, default=8000)
    p_sl.add_argument("--provider", default="CPUExecutionProvider")
    p_sl.add_argument("--tokenizer", default=None, help="Tokenizer name/path")
    p_sl.add_argument("--max-batch-size", type=int, default=32)
    p_sl.add_argument("--max-seq-len", type=int, default=2048)
    p_sl.add_argument("--kv-blocks", type=int, default=256, help="Number of KV cache blocks")
    p_sl.add_argument("--block-size", type=int, default=16, help="Tokens per KV block")

    # ── constrain ────────────────────────────────────────────
    p_cn = sub.add_parser("constrain", help="Grammar-constrained generation (JSON schema / regex / GBNF)")
    p_cn.add_argument("model", help="Path to .onnx LLM model")
    p_cn.add_argument("--prompt", default="Generate a JSON object:", help="Input prompt")
    p_cn.add_argument("--schema", default=None, help="JSON schema string or path to .json file")
    p_cn.add_argument("--regex", default=None, help="Regex pattern to constrain output")
    p_cn.add_argument("--grammar", default=None, help="GBNF grammar string or path to .gbnf file")
    p_cn.add_argument("--tokenizer", default=None, help="Tokenizer name/path")
    p_cn.add_argument("--max-tokens", type=int, default=512)
    p_cn.add_argument("--temperature", type=float, default=0.7)
    p_cn.add_argument("--provider", default="CPUExecutionProvider")

    # ── lora ─────────────────────────────────────────────────
    p_lo = sub.add_parser("lora", help="LoRA adapter runtime: load, hot-swap, fuse, merge (TIES/DARE/SLERP)")
    p_lo.add_argument("model", help="Path to base .onnx model")
    p_lo.add_argument("--action", choices=["list", "activate", "fuse", "merge"], default="list")
    p_lo.add_argument("--adapter", default=None, help="Path to LoRA adapter (safetensors/npz)")
    p_lo.add_argument("--output", "-o", default=None, help="Output path for fused/merged model")
    p_lo.add_argument("--merge-method", choices=["ties", "dare", "slerp", "task-arithmetic", "soup"],
                       default="ties")
    p_lo.add_argument("--merge-models", nargs="*", default=[], help="Model paths for weight merging")
    p_lo.add_argument("--density", type=float, default=0.5, help="TIES density")
    p_lo.add_argument("--drop-rate", type=float, default=0.9, help="DARE drop rate")
    p_lo.add_argument("--slerp-t", type=float, default=0.5, help="SLERP interpolation factor")
    p_lo.add_argument("--provider", default="CPUExecutionProvider")

    # ── tensor-parallel ──────────────────────────────────────
    p_tp = sub.add_parser("tensor-parallel", help="True tensor parallelism: split weight matrices across GPUs")
    p_tp.add_argument("model", help="Path to .onnx model")
    p_tp.add_argument("--action", choices=["analyze", "split", "run"], default="analyze")
    p_tp.add_argument("--num-gpus", type=int, default=2)
    p_tp.add_argument("--output-dir", default=None, help="Output directory for shards")
    p_tp.add_argument("--provider", default="CUDAExecutionProvider")

    # ── graph-compile ────────────────────────────────────────
    p_gc = sub.add_parser("graph-compile", help="CUDA/HIP graph capture + replay (20-47%% decode speedup)")
    p_gc.add_argument("model", help="Path to .onnx model")
    p_gc.add_argument("--action", choices=["analyze", "capture", "benchmark"], default="benchmark")
    p_gc.add_argument("--provider", default="CUDAExecutionProvider")
    p_gc.add_argument("--warmup", type=int, default=3)
    p_gc.add_argument("--runs", type=int, default=100)

    # ── amp-profile ──────────────────────────────────────────
    p_ap = sub.add_parser("amp-profile", help="Automatic mixed precision: per-layer profiling + Pareto-optimal search")
    p_ap.add_argument("model", help="Path to .onnx model")
    p_ap.add_argument("--action", choices=["profile", "optimize", "apply"], default="profile")
    p_ap.add_argument("--precisions", default="fp32,fp16,int8,int4", help="Comma-separated precisions to test")
    p_ap.add_argument("--max-mse", type=float, default=0.001, help="Maximum MSE budget for optimization")
    p_ap.add_argument("--strategy", choices=["dp", "greedy", "beam"], default="greedy")
    p_ap.add_argument("--output", "-o", default=None, help="Output path for mixed-precision model")
    p_ap.add_argument("--num-samples", type=int, default=50)
    p_ap.add_argument("--provider", default="CPUExecutionProvider")

    # ── distill-train ────────────────────────────────────────
    p_dt = sub.add_parser("distill-train", help="Live knowledge distillation with training loop through ORT")
    p_dt.add_argument("teacher", help="Path to teacher .onnx model")
    p_dt.add_argument("--student", default=None, help="Path to student .onnx model (auto-created if omitted)")
    p_dt.add_argument("--output", "-o", default=None, help="Output path for trained student")
    p_dt.add_argument("--epochs", type=int, default=10)
    p_dt.add_argument("--batch-size", type=int, default=8)
    p_dt.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p_dt.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    p_dt.add_argument("--alpha", type=float, default=0.5, help="KL vs CE loss balance")
    p_dt.add_argument("--reduction", choices=["depth", "width", "both"], default="depth",
                       help="Student architecture reduction strategy")
    p_dt.add_argument("--provider", default="CPUExecutionProvider")

    # ── a2a ──────────────────────────────────────────────────
    p_a2a = sub.add_parser("a2a", help="Architecture-to-architecture conversion (head pruning, width/depth shrinking, vocab pruning)")
    p_a2a.add_argument("model", help="Path to .onnx model")
    p_a2a.add_argument("--action", choices=["analyze", "prune-heads", "shrink-width", "shrink-depth", "prune-vocab"],
                        default="analyze")
    p_a2a.add_argument("--output", "-o", default=None, help="Output path for converted model")
    p_a2a.add_argument("--ratio", type=float, default=0.5, help="Reduction ratio (0.5 = keep half)")
    p_a2a.add_argument("--method", choices=["magnitude", "entropy", "taylor", "activation", "uniform", "first_last", "importance"],
                        default="magnitude")
    p_a2a.add_argument("--keep-tokens", default=None, help="Comma-separated token IDs to keep (vocab pruning)")
    p_a2a.add_argument("--corpus", default=None, help="Corpus file for frequency-based vocab pruning")
    p_a2a.add_argument("--provider", default="CPUExecutionProvider")

    # ── monitor-live ─────────────────────────────────────────
    p_ml = sub.add_parser("monitor-live", help="Real-time inference monitor with anomaly detection and TUI dashboard")
    p_ml.add_argument("--pid", type=int, default=None, help="PID of running inference process")
    p_ml.add_argument("--model", default=None, help="Path to .onnx model (self-monitoring mode)")
    p_ml.add_argument("--port", type=int, default=None, help="Port of running ISAT server")
    p_ml.add_argument("--no-dashboard", action="store_true", help="Disable TUI dashboard (log-only)")
    p_ml.add_argument("--provider", default="CPUExecutionProvider")

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        print(BANNER)
        print(f"  Provider: {_DEFAULT_PROVIDER}")
        try:
            import onnxruntime as ort
            print(f"  ORT:      {ort.__version__}  ({', '.join(ort.get_available_providers())})")
        except ImportError:
            print("  WARNING:  onnxruntime not found!")
            print("            pip install onnxruntime              # CPU only")
            print("            pip install onnxruntime-migraphx     # AMD MIGraphX")
            print("            pip install onnxruntime-gpu          # NVIDIA CUDA")
        print()
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
            "onnx": _cmd_onnx,
            "quantize": _cmd_quantize,
            "stream": _cmd_stream,
            "shard": _cmd_shard,
            "merge": _cmd_merge,
            "explain": _cmd_explain,
            "benchmark-suite": _cmd_benchmark_suite,
            "encrypt": _cmd_encrypt,
            "safety": _cmd_safety,
            "cloud-deploy": _cmd_cloud_deploy,
            "test": _cmd_test,
            "speculate": _cmd_speculate,
            "serve-llm": _cmd_serve_llm,
            "constrain": _cmd_constrain,
            "lora": _cmd_lora,
            "tensor-parallel": _cmd_tensor_parallel,
            "graph-compile": _cmd_graph_compile,
            "amp-profile": _cmd_amp_profile,
            "distill-train": _cmd_distill_train,
            "a2a": _cmd_a2a,
            "monitor-live": _cmd_monitor_live,
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
    from isat.auto_detect.detector import detect_hardware
    from isat.auto_detect.recommender import format_report, generate_recommendations

    print(BANNER)
    print("[1] Detecting hardware (all vendors)...")
    hw_profile = detect_hardware()
    model_path = args.model or ""

    report = generate_recommendations(hw_profile, model_path)

    if getattr(args, "tune_json", False):
        import dataclasses
        import json as _json
        out = {
            "hardware": {
                "os": hw_profile.os_name,
                "cpu": hw_profile.cpu_name,
                "cpu_cores": hw_profile.cpu_cores,
                "ram_mb": hw_profile.system_ram_mb,
                "swap_mb": hw_profile.swap_mb,
                "gpus": [dataclasses.asdict(g) for g in hw_profile.gpus],
                "vendor": hw_profile.vendor,
                "is_apu": hw_profile.is_apu,
            },
            "model": model_path,
            "recipes": [
                {
                    "title": r.title,
                    "provider": r.provider,
                    "env_vars": r.env_vars,
                    "install_cmd": r.install_cmd,
                }
                for r in report.recipes
            ],
        }
        print(_json.dumps(out, indent=2))
        return 0

    print(format_report(report))

    # Generate and save inference script if model is provided
    if model_path and Path(model_path).exists():
        from isat.auto_detect.script_gen import save_script
        output_dir = getattr(args, "output_dir", "isat_output")
        script_path = save_script(hw_profile, model_path, output_dir)
        print(f"  SCRIPT GENERATED: {script_path}")
        print(f"  Run it directly:  python3 {script_path}")
        print()

    if not model_path or args.detect_only:
        if not model_path:
            print("  TIP: Provide a model to get specific tuning recommendations:")
            print("    isat tune model.onnx")
            print()
            print("  Or detect hardware only:")
            print("    isat tune --detect-only")
            print()
        return 0

    if not Path(model_path).exists():
        print(f"Error: model not found: {model_path}")
        return 1

    from isat.benchmark.runner import BenchmarkRunner
    from isat.database.store import ResultsDB
    from isat.fingerprint.hardware import fingerprint_hardware
    from isat.fingerprint.model import fingerprint_model
    from isat.report.generator import ReportGenerator
    from isat.search.engine import SearchEngine

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

    print("[2/6] Fingerprinting hardware (detailed)...")
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
    print(f"  XNACK enabled  : {'yes' if hw.xnack_enabled else 'no'}")
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

    latencies_a = result_a.latencies if result_a.latencies else [result_a.mean_latency_ms] * args.runs
    latencies_b = result_b.latencies if result_b.latencies else [result_b.mean_latency_ms] * args.runs

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

    if args.latency is None:
        print("Error: --latency is required (unless using --list-gpus)")
        return 1

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
    if args.p50 is not None: metrics["p50_ms"] = args.p50
    if args.p95 is not None: metrics["p95_ms"] = args.p95
    if args.p99 is not None: metrics["p99_ms"] = args.p99
    if args.throughput is not None: metrics["throughput_rps"] = args.throughput
    if args.memory is not None: metrics["memory_mb"] = args.memory
    if args.cold_start is not None: metrics["cold_start_ms"] = args.cold_start

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
    if not args.model_name:
        print("Error: model_name is required (unless using --list)")
        return 1
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
        providers=ort_providers(args.provider),
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


def _cmd_quantize(args) -> int:
    from isat.quantize.quantizer import ModelQuantizer, quantize_model

    print(BANNER)
    model_path = args.model
    output = args.output
    if output is None:
        stem = Path(model_path).stem
        output = str(Path(model_path).parent / f"{stem}_{args.method}.onnx")

    if args.sensitivity:
        print(f"  Running sensitivity analysis on {model_path} ...")
        q = ModelQuantizer(model_path)
        result = q.sensitivity_analysis()
        print(f"\n  {'Layer':<50} {'Error Delta':>12}")
        print(f"  {'─'*62}")
        for layer, delta in sorted(result.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {layer:<50} {delta:>12.6f}")
        return 0

    print(f"  Model      : {model_path}")
    print(f"  Method     : {args.method}")
    print(f"  Output     : {output}")
    print()

    result = quantize_model(
        model_path=model_path,
        output_path=output,
        method=args.method,
        per_channel=args.per_channel,
        block_size=args.block_size,
        alpha=args.alpha,
    )

    if not result.success:
        print(f"\n  QUANTIZATION FAILED: {result.error}")
        return 1

    print(f"  Method            : {result.method}")
    print(f"  Original size     : {result.original_size_mb:.1f} MB")
    print(f"  Quantized size    : {result.quantized_size_mb:.1f} MB")
    print(f"  Compression ratio : {result.compression_ratio:.2f}x")
    print(f"  Time              : {result.elapsed_s:.1f}s")
    print(f"\n  Output: {result.output_path}")
    return 0


def _cmd_stream(args) -> int:
    from isat.stream.generator import StreamingGenerator

    print(BANNER)
    model_path = args.model

    gen = StreamingGenerator(
        model_path=model_path,
        provider=args.provider,
        max_length=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    if args.benchmark:
        print(f"  Benchmarking streaming inference on {model_path} ...")
        try:
            tokenizer_name = args.tokenizer
            if tokenizer_name is None:
                print("  --tokenizer required for benchmark mode")
                return 1
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            ids = tok.encode(args.prompt)
            metrics = gen.benchmark(ids, max_new_tokens=args.max_tokens)
            print(f"\n  TTFT           : {metrics.ttft_ms:.1f} ms")
            print(f"  Mean ITL       : {metrics.mean_itl_ms:.1f} ms")
            print(f"  P95 ITL        : {metrics.p95_itl_ms:.1f} ms")
            print(f"  Tokens/sec     : {metrics.tokens_per_sec:.1f}")
            print(f"  Total tokens   : {metrics.total_tokens}")
            print(f"  Total time     : {metrics.total_time_ms:.0f} ms")
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            return 1
        return 0

    print(f"  Model    : {model_path}")
    print(f"  Prompt   : {args.prompt}")
    print(f"  Provider : {args.provider}")
    print()

    try:
        result = gen.generate_text(
            prompt=args.prompt,
            tokenizer_name=args.tokenizer,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print(f"\n  Generated text:\n")
        print(f"  {result}")
    except Exception as e:
        print(f"  Generation failed: {e}")
        return 1
    return 0


def _cmd_shard(args) -> int:
    from isat.shard.splitter import ModelSharder, shard_model

    print(BANNER)
    model_path = args.model

    sharder = ModelSharder(model_path)

    if args.analyze:
        print(f"  Analyzing {model_path} for sharding ...\n")
        analysis = sharder.analyze()
        print(f"  Total params       : {analysis.total_params:,}")
        print(f"  Total size         : {analysis.total_size_mb:.1f} MB")
        print(f"  Num layers         : {analysis.num_layers}")
        print(f"  Recommended shards : {analysis.recommended_shards}")
        print(f"  Memory per shard   : {analysis.memory_per_shard_mb:.1f} MB")
        print(f"\n  Top layers by size:")
        for name, size in sorted(analysis.layer_sizes.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {name:<50} {size:>8.2f} MB")
        return 0

    if args.validate:
        output_dir = args.output_dir or str(Path(model_path).parent / "shards")
        print(f"  Validating shards in {output_dir} ...")
        ok = sharder.validate_shards(output_dir)
        print(f"  Validation: {'PASSED' if ok else 'FAILED'}")
        return 0 if ok else 1

    output_dir = args.output_dir or str(Path(model_path).parent / f"{Path(model_path).stem}_shards")
    print(f"  Model      : {model_path}")
    print(f"  Shards     : {args.num_shards}")
    print(f"  Strategy   : {args.strategy}")
    print(f"  Output dir : {output_dir}")
    print()

    result = shard_model(model_path, args.num_shards, output_dir, args.strategy)
    if not result.success:
        print(f"\n  SHARDING FAILED: {result.error}")
        return 1

    print(f"  Created {result.num_shards} shards in {result.elapsed_s:.1f}s:")
    for p, s in zip(result.shard_paths, result.shard_sizes_mb):
        print(f"    {Path(p).name:<40} {s:>8.1f} MB")
    return 0


def _cmd_merge(args) -> int:
    from isat.merge.merger import merge_models

    print(BANNER)
    print(f"  Models : {', '.join(args.models)}")
    print(f"  Mode   : {args.mode}")
    print(f"  Output : {args.output}")
    print()

    kwargs = {}
    if args.mode == "parallel":
        kwargs["aggregation"] = args.aggregation

    result = merge_models(args.models, args.output, mode=args.mode, **kwargs)
    if not result.success:
        print(f"\n  MERGE FAILED: {result.error}")
        return 1

    print(f"  Models merged      : {result.num_models_merged}")
    print(f"  Total nodes        : {result.total_nodes}")
    print(f"  Total size         : {result.total_size_mb:.1f} MB")
    print(f"  Time               : {result.elapsed_s:.1f}s")
    print(f"\n  Output: {result.output_path}")

    if args.validate:
        from isat.merge.merger import ModelMerger
        m = ModelMerger()
        ok = m.validate(args.output, args.models)
        print(f"  Validation: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            return 1
    return 0


def _cmd_explain(args) -> int:
    from isat.explain.explainer import explain_model

    print(BANNER)
    print(f"  Model    : {args.model}")
    print(f"  Method   : {args.method}")
    print(f"  Provider : {args.provider}")
    print()

    report = explain_model(args.model, method=args.method)
    print(f"  Input shapes       : {report.input_shapes}")
    print(f"  Method used        : {report.method_used}")
    print(f"  Time               : {report.elapsed_s:.1f}s")

    if report.feature_importance:
        print(f"\n  Feature importance (top inputs):")
        for name, imp in report.feature_importance.items():
            import numpy as np
            mean_imp = float(np.mean(imp)) if hasattr(imp, '__len__') else imp
            print(f"    {name:<40} mean={mean_imp:.6f}")

    if report.top_sensitive_regions:
        print(f"\n  Top sensitive regions:")
        for region in report.top_sensitive_regions[:10]:
            print(f"    {region}")

    if report.attention_layers:
        print(f"\n  Attention layers found: {len(report.attention_layers)}")
    return 0


def _cmd_benchmark_suite(args) -> int:
    from isat.benchmark_suite.suite import BenchmarkSuite, run_benchmark_suite

    print(BANNER)
    model_path = args.model
    provider = args.provider if args.provider != "auto" else None
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    print(f"  Model        : {model_path}")
    print(f"  Provider     : {args.provider}")
    print(f"  Batch sizes  : {batch_sizes}")
    print()

    suite = BenchmarkSuite(model_path, provider=provider, output_dir=args.output_dir)

    if args.latency_only:
        report = suite.run_latency(batch_sizes=batch_sizes)
        print("  Latency results:")
        for bs, stats in report.items():
            print(f"    batch={bs:<4}  mean={stats['mean_ms']:.2f}ms  p95={stats['p95_ms']:.2f}ms  p99={stats['p99_ms']:.2f}ms")
        return 0

    if args.throughput_only:
        report = suite.run_throughput(duration_s=args.duration)
        print(f"  Throughput    : {report['inferences_per_sec']:.1f} inf/s")
        print(f"  Total infer   : {report['total_inferences']}")
        print(f"  Mean latency  : {report['mean_latency_ms']:.2f} ms")
        return 0

    if args.memory_only:
        report = suite.run_memory(batch_sizes=batch_sizes)
        print("  Memory results:")
        for bs, stats in report.items():
            print(f"    batch={bs:<4}  peak_rss={stats['peak_rss_mb']:.1f} MB")
        return 0

    result = suite.run_all(output_dir=args.output_dir)
    suite.generate_report(result, args.output_dir)
    print(f"\n  Full benchmark complete. Reports saved to: {args.output_dir or 'current directory'}")
    return 0


def _cmd_encrypt(args) -> int:
    from isat.encrypt.protector import ModelProtector, protect_model

    print(BANNER)
    model_path = args.model
    method = args.method

    print(f"  Model    : {model_path}")
    print(f"  Method   : {method}")
    print(f"  Output   : {args.output}")
    print()

    if method == "encrypt":
        if not args.password:
            print("  ERROR: --password required for encryption")
            return 1
        p = ModelProtector(model_path)
        result = p.encrypt(args.output, args.password)
    elif method == "decrypt":
        if not args.password:
            print("  ERROR: --password required for decryption")
            return 1
        p = ModelProtector(model_path)
        result = p.decrypt(model_path, args.output, args.password)
    elif method == "obfuscate":
        p = ModelProtector(model_path)
        result = p.obfuscate(args.output, seed=args.seed)
    elif method == "deobfuscate":
        if args.seed is None:
            print("  ERROR: --seed required for deobfuscation")
            return 1
        p = ModelProtector(model_path)
        result = p.deobfuscate(model_path, args.output, args.seed)
    elif method == "fingerprint":
        if not args.owner:
            print("  ERROR: --owner required for fingerprinting")
            return 1
        p = ModelProtector(model_path)
        result = p.fingerprint(args.output, args.owner)
    elif method == "verify":
        if not args.owner:
            print("  ERROR: --owner required for verification")
            return 1
        p = ModelProtector(model_path)
        verified = p.verify_fingerprint(model_path, args.owner)
        print(f"  Fingerprint verified: {verified}")
        return 0 if verified else 1
    else:
        print(f"  Unknown method: {method}")
        return 1

    if not result.success:
        print(f"\n  FAILED: {result.error}")
        return 1

    print(f"  Original size  : {result.original_size_mb:.1f} MB")
    print(f"  Output size    : {result.protected_size_mb:.1f} MB")
    print(f"  Time           : {result.elapsed_s:.1f}s")
    print(f"\n  Output: {result.output_path}")
    return 0


def _cmd_safety(args) -> int:
    from isat.safety.guardrails import SafetyGuard

    print(BANNER)

    guard = SafetyGuard()

    input_text = args.input_text
    output_text = args.output_text
    if args.input_file:
        input_text = Path(args.input_file).read_text()
    if args.output_file:
        output_text = Path(args.output_file).read_text()

    if not input_text and not output_text:
        print("  ERROR: Provide --input-text, --output-text, --input-file, or --output-file")
        return 1

    report = guard.run_all(input_text=input_text, output_text=output_text)

    print(f"  Overall safe   : {'YES' if report.overall_safe else 'NO'}")
    print(f"  Checks run     : {len(report.checks)}")
    print(f"  Time           : {report.elapsed_ms:.0f} ms")
    print()

    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.category:<20} severity={check.severity}")
        if check.findings:
            for f in check.findings[:5]:
                print(f"         - {f}")
    return 0


def _cmd_cloud_deploy(args) -> int:
    from isat.cloud_deploy.deployer import CloudDeployer, deploy_model

    print(BANNER)
    model_path = args.model
    output_dir = args.output_dir or str(Path(model_path).parent / f"{Path(model_path).stem}_deploy")

    print(f"  Model      : {model_path}")
    print(f"  Target     : {args.target}")
    print(f"  Output dir : {output_dir}")
    print()

    deployer = CloudDeployer(model_path)
    target = args.target

    os.makedirs(output_dir, exist_ok=True)

    generated = []
    if target in ("all", "handler"):
        p = deployer.generate_inference_handler(output_dir)
        generated.append(("Inference handler", p))
    if target in ("all", "docker"):
        p = deployer.generate_dockerfile(output_dir, gpu=args.gpu)
        generated.append(("Dockerfile", p))
    if target in ("all", "kubernetes"):
        paths = deployer.generate_kubernetes(output_dir, replicas=args.replicas, gpu=args.gpu)
        for pp in paths:
            generated.append(("K8s manifest", pp))
    if target in ("all", "sagemaker"):
        p = deployer.generate_sagemaker(output_dir)
        generated.append(("SageMaker artifacts", p))
    if target in ("all", "azure"):
        p = deployer.generate_azure_ml(output_dir)
        generated.append(("Azure ML artifacts", p))
    if target in ("all", "gcp"):
        p = deployer.generate_gcp_vertex(output_dir)
        generated.append(("GCP Vertex artifacts", p))

    print(f"  Generated {len(generated)} artifacts:")
    for label, path in generated:
        print(f"    {label:<25} {path}")

    cost = deployer.estimate_cost()
    print(f"\n  Estimated monthly costs (10K req/day):")
    for provider, est in cost.items():
        print(f"    {provider:<12} ${est:.0f}/month")
    return 0


def _cmd_test(args) -> int:
    from isat.model_test.tester import ModelTester, test_model

    print(BANNER)
    model_path = args.model

    tester = ModelTester(model_path, provider=args.provider if args.provider != "auto" else None)

    if args.generate_golden:
        if not args.golden:
            golden_path = str(Path(model_path).with_suffix(".golden.npz"))
        else:
            golden_path = args.golden
        print(f"  Generating golden test file: {golden_path}")
        import numpy as np
        inputs = tester._build_random_inputs(tester._session)
        tester.generate_golden(inputs, golden_path)
        print(f"  Done.")
        return 0

    print(f"  Model    : {model_path}")
    print(f"  Provider : {args.provider}")
    print(f"  Suite    : {args.suite}")
    print()

    suite_map = {
        "determinism": tester.test_determinism,
        "stability": tester.test_numerical_stability,
        "edge": tester.test_edge_cases,
        "input": tester.test_input_validation,
        "cross-provider": tester.test_cross_provider,
        "latency": tester.test_latency_consistency,
        "memory": tester.test_memory_safety,
    }

    if args.suite == "golden":
        if not args.golden:
            print("  ERROR: --golden PATH required for golden test")
            return 1
        result = tester.test_golden(args.golden)
        results = type("R", (), {"total_tests": 1, "passed": int(result.passed), "failed": int(not result.passed),
                                  "skipped": 0, "results": [result], "elapsed_s": result.elapsed_ms / 1000})()
    elif args.suite == "all":
        results = tester.run_all()
    elif args.suite in suite_map:
        r = suite_map[args.suite]()
        results = type("R", (), {"total_tests": 1, "passed": int(r.passed), "failed": int(not r.passed),
                                  "skipped": 0, "results": [r], "elapsed_s": r.elapsed_ms / 1000})()
    else:
        print(f"  Unknown suite: {args.suite}")
        return 1

    print(f"  {'='*60}")
    print(f"  TEST RESULTS")
    print(f"  {'='*60}")
    print(f"  Total    : {results.total_tests}")
    print(f"  Passed   : {results.passed}")
    print(f"  Failed   : {results.failed}")
    print(f"  Skipped  : {results.skipped}")
    print(f"  Time     : {results.elapsed_s:.1f}s")
    print()

    for r in results.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:<35} ({r.elapsed_ms:.0f}ms)")
        if not r.passed and r.details:
            print(f"         {r.details[:100]}")

    if args.junit:
        xml = results.to_junit_xml() if hasattr(results, 'to_junit_xml') else "<testsuites/>"
        junit_path = str(Path(args.output_dir or ".") / "test_results.xml")
        Path(junit_path).write_text(xml)
        print(f"\n  JUnit XML: {junit_path}")

    return 0 if results.failed == 0 else 1


def _cmd_speculate(args) -> int:
    from isat.speculative.engine import SpeculativeDecoder, SelfSpeculativeDecoder

    print(BANNER)
    print(f"  Target     : {args.target}")
    print(f"  Draft      : {args.draft or '(self-speculation)'}")
    print(f"  Mode       : {args.mode}")
    print(f"  Speculative: {args.num_speculative} tokens/step")
    print()

    if args.mode == "self":
        decoder = SelfSpeculativeDecoder(args.target, provider=args.provider)
    else:
        if not args.draft:
            print("  ERROR: --draft required for draft model mode")
            return 1
        decoder = SpeculativeDecoder(
            args.target, args.draft,
            provider=args.provider,
            num_speculative_tokens=args.num_speculative,
        )

    if args.benchmark:
        print("  Running speculative decoding benchmark...")
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.tokenizer or "gpt2")
            ids = tok.encode(args.prompt)
            metrics = decoder.benchmark(ids, max_new_tokens=args.max_tokens)
            print(f"\n  Acceptance rate    : {metrics.acceptance_rate:.1%}")
            print(f"  Tokens/step        : {metrics.mean_tokens_per_step:.2f}")
            print(f"  Speedup vs naive   : {metrics.speedup_vs_naive:.2f}x")
            print(f"  TTFT               : {metrics.ttft_ms:.1f} ms")
            print(f"  Mean ITL           : {metrics.mean_itl_ms:.1f} ms")
            print(f"  Tokens/sec         : {metrics.tokens_per_sec:.1f}")
            print(f"  Draft time         : {metrics.draft_time_ms:.0f} ms")
            print(f"  Verify time        : {metrics.verify_time_ms:.0f} ms")
        except Exception as e:
            print(f"  Benchmark failed: {e}")
            return 1
        return 0

    try:
        result = decoder.generate_text(
            args.prompt, tokenizer_name=args.tokenizer or "gpt2",
            max_new_tokens=args.max_tokens, temperature=args.temperature,
        )
        print(f"  Generated:\n\n  {result}")
    except Exception as e:
        print(f"  Generation failed: {e}")
        return 1
    return 0


def _cmd_serve_llm(args) -> int:
    from isat.llm_server.server import serve_llm

    print(BANNER)
    print(f"  Model      : {args.model}")
    print(f"  Provider   : {args.provider}")
    print(f"  Port       : {args.port}")
    print(f"  Tokenizer  : {args.tokenizer or '(none)'}")
    print(f"  Max batch  : {args.max_batch_size}")
    print(f"  KV blocks  : {args.kv_blocks} x {args.block_size}")
    print()

    serve_llm(
        args.model,
        port=args.port,
        provider=args.provider,
        tokenizer_name=args.tokenizer,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        num_kv_blocks=args.kv_blocks,
        block_size=args.block_size,
    )
    return 0


def _cmd_constrain(args) -> int:
    from isat.constrained.grammar import constrained_generate

    print(BANNER)
    print(f"  Model      : {args.model}")
    print(f"  Prompt     : {args.prompt}")
    print()

    schema = None
    regex = args.regex
    grammar = args.grammar

    if args.schema:
        import json as json_mod
        schema_str = args.schema
        if Path(schema_str).exists():
            schema_str = Path(schema_str).read_text()
        schema = json_mod.loads(schema_str)

    if args.grammar and Path(args.grammar).exists():
        grammar = Path(args.grammar).read_text()

    result = constrained_generate(
        args.model, args.prompt,
        schema=schema, regex=regex, grammar=grammar,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        provider=args.provider,
    )

    print(f"  Valid            : {result.valid}")
    print(f"  Tokens generated : {result.tokens_generated}")
    print(f"  Tokens rejected  : {result.tokens_rejected}")
    print(f"  FSM overhead     : {result.fsm_overhead_ms:.1f} ms")
    print(f"  Total time       : {result.total_time_ms:.0f} ms")
    print(f"\n  Output:\n  {result.text}")
    if result.parsed_value and isinstance(result.parsed_value, dict):
        import json as json_mod
        print(f"\n  Parsed JSON:\n  {json_mod.dumps(result.parsed_value, indent=2)}")
    return 0


def _cmd_lora(args) -> int:
    print(BANNER)
    action = args.action

    if action == "merge" and args.merge_models:
        from isat.lora.merger import WeightMerger
        merger = WeightMerger(args.model)
        method = args.merge_method
        output = args.output or f"{Path(args.model).stem}_merged.onnx"
        models = args.merge_models

        print(f"  Base       : {args.model}")
        print(f"  Models     : {', '.join(models)}")
        print(f"  Method     : {method}")
        print()

        method_map = {
            "ties": lambda: merger.ties_merge(models, density=args.density, output_path=output),
            "dare": lambda: merger.dare_merge(models, drop_rate=args.drop_rate, output_path=output),
            "slerp": lambda: merger.slerp_merge(models[0], models[1] if len(models) > 1 else models[0], t=args.slerp_t, output_path=output),
            "task-arithmetic": lambda: merger.task_arithmetic(models, output_path=output),
            "soup": lambda: merger.model_soup(models, output_path=output),
        }
        result = method_map[method]()
        if result.success:
            print(f"  Merged {result.num_models} models -> {result.output_path}")
            print(f"  Size: {result.total_size_mb:.1f} MB, Time: {result.elapsed_s:.1f}s")
        else:
            print(f"  MERGE FAILED: {result.error}")
            return 1
        return 0

    from isat.lora.adapter import LoRARuntime
    runtime = LoRARuntime(args.model, provider=args.provider)

    if action == "list":
        adapters = runtime.list_adapters()
        if not adapters:
            print("  No adapters loaded. Use --adapter PATH to load one.")
        for a in adapters:
            print(f"  {a.name}: rank={a.rank}, params={a.num_params:,}, {a.size_mb:.1f} MB")
        return 0

    if not args.adapter:
        print("  ERROR: --adapter required for this action")
        return 1

    runtime.load_adapter(args.adapter)
    if action == "activate":
        runtime.activate(Path(args.adapter).stem)
        print(f"  Adapter activated: {args.adapter}")
    elif action == "fuse":
        output = args.output or f"{Path(args.model).stem}_fused.onnx"
        runtime.fuse(Path(args.adapter).stem, output)
        print(f"  Fused model saved: {output}")
    return 0


def _cmd_tensor_parallel(args) -> int:
    from isat.parallel.tensor_parallel import TensorParallelizer, TensorParallelRunner, tensor_parallel

    print(BANNER)
    print(f"  Model    : {args.model}")
    print(f"  GPUs     : {args.num_gpus}")
    print(f"  Action   : {args.action}")
    print()

    tp = TensorParallelizer(args.model, num_gpus=args.num_gpus, provider=args.provider)

    if args.action == "analyze":
        plan = tp.analyze()
        print(f"  Column-parallel layers : {len(plan.column_parallel_layers)}")
        print(f"  Row-parallel layers    : {len(plan.row_parallel_layers)}")
        print(f"  Replicated layers      : {len(plan.replicated_layers)}")
        print(f"  Est. memory/GPU        : {plan.estimated_memory_per_gpu_mb:.1f} MB")
        print(f"  Communication volume   : {plan.communication_volume_mb:.1f} MB")
        return 0

    output_dir = args.output_dir or f"{Path(args.model).stem}_tp{args.num_gpus}"
    if args.action == "split":
        result = tp.split(output_dir)
        if result.success:
            print(f"  Split into {result.num_gpus} shards in {result.elapsed_s:.1f}s")
            for p in result.shard_paths:
                print(f"    {p}")
        else:
            print(f"  FAILED: {result.error}")
            return 1
    return 0


def _cmd_graph_compile(args) -> int:
    from isat.graph_compile.capture import GraphCapture, GraphRegionAnalyzer, graph_compile

    print(BANNER)
    print(f"  Model    : {args.model}")
    print(f"  Action   : {args.action}")
    print(f"  Provider : {args.provider}")
    print()

    if args.action == "analyze":
        analyzer = GraphRegionAnalyzer(args.model)
        report = analyzer.analyze()
        print(f"  Static regions     : {report.num_static_regions}")
        print(f"  Dynamic regions    : {report.num_dynamic_regions}")
        print(f"  Capturable nodes   : {report.capturable_nodes} / {report.total_nodes}")
        print(f"  Capture coverage   : {report.capture_coverage:.1%}")
        return 0

    gc = GraphCapture(args.model, provider=args.provider)
    if args.action == "benchmark":
        metrics = gc.benchmark(num_runs=args.runs, num_warmup=args.warmup)
        print(f"  Normal mean   : {metrics.normal_mean_ms:.2f} ms")
        print(f"  Normal P99    : {metrics.normal_p99_ms:.2f} ms")
        print(f"  Captured mean : {metrics.captured_mean_ms:.2f} ms")
        print(f"  Captured P99  : {metrics.captured_p99_ms:.2f} ms")
        print(f"  Speedup       : {metrics.speedup_ratio:.2f}x")
    return 0


def _cmd_amp_profile(args) -> int:
    from isat.amp.profiler import PrecisionProfiler

    print(BANNER)
    print(f"  Model      : {args.model}")
    print(f"  Action     : {args.action}")
    print()

    precisions = args.precisions.split(",")
    profiler = PrecisionProfiler(args.model, provider=args.provider)

    if args.action == "profile":
        profile = profiler.profile_all(precisions=precisions, num_samples=args.num_samples)
        print(f"  {'Layer':<40} {'Precision':<8} {'MSE':>10} {'CosSim':>8} {'Latency':>10}")
        print(f"  {'─' * 80}")
        for layer, prec_results in list(profile.items())[:20]:
            for prec, r in prec_results.items():
                print(f"  {layer[:40]:<40} {prec:<8} {r.mse:>10.6f} {r.cosine_sim:>8.4f} {r.latency_ms:>8.2f}ms")
        return 0

    if args.action == "optimize":
        from isat.amp.optimizer import MixedPrecisionOptimizer
        profile = profiler.profile_all(precisions=precisions, num_samples=args.num_samples)
        optimizer = MixedPrecisionOptimizer(profile)
        assignment = optimizer.optimize(max_mse=args.max_mse, strategy=args.strategy)
        print(f"  Total MSE          : {assignment.total_mse:.6f}")
        print(f"  Total latency      : {assignment.total_latency_ms:.1f} ms")
        print(f"  Speedup vs FP32    : {assignment.speedup_vs_fp32:.2f}x")
        print(f"  Compression ratio  : {assignment.compression_ratio:.2f}x")
        if args.output:
            optimizer.apply(assignment, args.output)
            print(f"\n  Mixed-precision model saved: {args.output}")
        return 0
    return 0


def _cmd_distill_train(args) -> int:
    from isat.distill_train.trainer import DistillationTrainer, distill_model

    print(BANNER)
    print(f"  Teacher    : {args.teacher}")
    print(f"  Student    : {args.student or '(auto-created)'}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha      : {args.alpha}")
    print()

    output = args.output or f"{Path(args.teacher).stem}_student.onnx"
    result = distill_model(
        teacher_path=args.teacher,
        student_path=args.student,
        output_path=output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        temperature=args.temperature,
        alpha=args.alpha,
    )

    if result.success:
        print(f"  Epochs completed   : {result.epochs_completed}")
        print(f"  Final loss         : {result.final_loss:.4f}")
        print(f"  Teacher size       : {result.teacher_size_mb:.1f} MB")
        print(f"  Student size       : {result.student_size_mb:.1f} MB")
        print(f"  Compression        : {result.compression_ratio:.2f}x")
        print(f"  Training time      : {result.training_time_s:.1f}s")
        print(f"\n  Student model: {result.output_path}")
    else:
        print(f"  DISTILLATION FAILED: {result.error}")
        return 1
    return 0


def _cmd_a2a(args) -> int:
    from isat.arch_convert.converter import ArchitectureConverter, convert_architecture

    print(BANNER)
    print(f"  Model    : {args.model}")
    print(f"  Action   : {args.action}")
    print()

    conv = ArchitectureConverter(args.model)

    if args.action == "analyze":
        analysis = conv.analyze()
        print(f"  Layers         : {analysis.num_layers}")
        print(f"  Attention heads: {analysis.num_heads}")
        print(f"  Hidden dim     : {analysis.hidden_dim}")
        print(f"  FFN dim        : {analysis.ffn_dim}")
        print(f"  Vocab size     : {analysis.vocab_size}")
        print(f"  Total params   : {analysis.total_params:,}")
        return 0

    output = args.output or f"{Path(args.model).stem}_{args.action}.onnx"

    if args.action == "prune-heads":
        result = conv.prune_heads(
            importance_method=args.method,
            num_heads_to_keep=max(1, int(conv.analyze().num_heads * args.ratio)),
            output_path=output,
        )
    elif args.action == "shrink-width":
        result = conv.shrink_width(ratio=args.ratio, importance_method=args.method, output_path=output)
    elif args.action == "shrink-depth":
        result = conv.shrink_depth(ratio=args.ratio, method=args.method, output_path=output)
    elif args.action == "prune-vocab":
        keep = None
        if args.keep_tokens:
            keep = [int(t) for t in args.keep_tokens.split(",")]
        result = conv.prune_vocab(keep_tokens=keep, corpus_path=args.corpus, output_path=output)
    else:
        print(f"  Unknown action: {args.action}")
        return 1

    if result.success:
        print(f"  Original params : {result.original_params:,}")
        print(f"  New params      : {getattr(result, 'pruned_params', getattr(result, 'new_params', 0)):,}")
        print(f"  Reduction       : {result.reduction_ratio:.2f}x")
        print(f"  Output          : {result.output_path}")
    else:
        print(f"  FAILED: {result.error}")
        return 1
    return 0


def _cmd_monitor_live(args) -> int:
    from isat.live_monitor.daemon import InferenceMonitor
    from isat.live_monitor.dashboard import MonitorDashboard, monitor_live

    print(BANNER)
    monitor_live(
        pid=args.pid,
        model_path=args.model,
        port=args.port,
        dashboard=not args.no_dashboard,
    )
    return 0


def _cmd_onnx(args) -> int:
    from isat.converter.engine import convert, detect_format

    print(BANNER)

    fmt = detect_format(args.input)
    print(f"  Converting : {args.input}")
    print(f"  Format     : {fmt.name} (auto-detected)")
    print(f"  Opset      : {args.opset}")
    if args.input_shape:
        print(f"  Input shape: {args.input_shape}")
    print()

    print("  [1/3] Converting to ONNX...")
    result = convert(
        input_path=args.input,
        output_dir=args.output,
        opset=args.opset,
        input_shape=args.input_shape,
        simplify=args.simplify,
    )

    if not result.success:
        print(f"\n  CONVERSION FAILED")
        print(f"  {result.error}")
        return 1

    print(f"         OK ({result.elapsed_s:.1f}s)")
    print()

    print("  [2/3] Validating ONNX model...")
    print(f"         OK (opset={result.opset}, {result.num_nodes} nodes, {result.size_mb:.1f} MB)")
    print()

    print(f"  {'='*65}")
    print(f"  CONVERSION RESULT")
    print(f"  {'='*65}")
    print(result.summary())
    print(f"  {'='*65}")
    print()

    if args.no_tune:
        return 0

    onnx_path = result.onnx_path
    if not Path(onnx_path).exists():
        return 0

    print("  [3/3] Detecting best configuration...")
    print()

    try:
        from isat.auto_detect.detector import detect_hardware
        from isat.auto_detect.recommender import format_report, generate_recommendations
        from isat.auto_detect.script_gen import save_script

        hw_profile = detect_hardware()
        report = generate_recommendations(hw_profile, onnx_path)
        print(format_report(report))

        output_dir = str(Path(onnx_path).parent)
        script_path = save_script(hw_profile, onnx_path, output_dir)
        print()
        print(f"  INFERENCE SCRIPT: {script_path}")
        print(f"  Run it directly : python3 {script_path}")
        print()
    except Exception as e:
        print(f"  Best-config detection failed: {e}")
        print(f"  You can still run: isat tune {onnx_path}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
