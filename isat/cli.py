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
import sys
import time
from pathlib import Path


BANNER = r"""
  _____ ____    _  _____
 |_   _/ ___|  / \|_   _|
   | | \___ \ / _ \ | |
   | |  ___) / ___ \| |
   |_| |____/_/   \_\_|

  Inference Stack Auto-Tuner v0.1.0
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="isat",
        description="Inference Stack Auto-Tuner -- find the fastest ORT config for your model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'isat <command> --help' for detailed options.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    parser.add_argument("--version", action="version", version="isat 0.1.0")
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

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        print(BANNER)
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


if __name__ == "__main__":
    sys.exit(main())
