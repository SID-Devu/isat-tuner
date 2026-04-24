"""ISAT command-line interface.

Usage:
    isat tune    MODEL.onnx          -- run full auto-tune search
    isat inspect MODEL.onnx          -- fingerprint model without benchmarking
    isat hwinfo                      -- print hardware fingerprint
    isat history [--model NAME]      -- show past tuning results
    isat export  [--model NAME]      -- re-generate reports from DB
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="isat",
        description="Inference Stack Auto-Tuner -- find the fastest ORT config for your model",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command")

    # ── tune ──────────────────────────────────────────────────
    p_tune = sub.add_parser("tune", help="Auto-tune an ONNX model")
    p_tune.add_argument("model", help="Path to .onnx model")
    p_tune.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    p_tune.add_argument("--runs", type=int, default=5, help="Measured iterations (default: 5)")
    p_tune.add_argument("--cooldown", type=float, default=60.0, help="Cooldown between configs in seconds (default: 60)")
    p_tune.add_argument("--max-configs", type=int, default=0, help="Limit total configs to test (0=unlimited)")
    p_tune.add_argument("--provider", default="MIGraphXExecutionProvider",
                        help="ORT execution provider (default: MIGraphXExecutionProvider)")
    p_tune.add_argument("--skip-precision", action="store_true", help="Skip precision search dimension")
    p_tune.add_argument("--skip-graph", action="store_true", help="Skip graph transform search dimension")
    p_tune.add_argument("--output-dir", default="isat_output", help="Output directory for reports")
    p_tune.add_argument("--db", default="isat_results.db", help="Results database path")
    p_tune.add_argument("--dry-run", action="store_true", help="Show plan without benchmarking")

    # ── inspect ───────────────────────────────────────────────
    p_inspect = sub.add_parser("inspect", help="Fingerprint a model without benchmarking")
    p_inspect.add_argument("model", help="Path to .onnx model")

    # ── hwinfo ────────────────────────────────────────────────
    sub.add_parser("hwinfo", help="Print hardware fingerprint")

    # ── history ───────────────────────────────────────────────
    p_hist = sub.add_parser("history", help="Show past tuning results from database")
    p_hist.add_argument("--model", default=None, help="Filter by model name")
    p_hist.add_argument("--top", type=int, default=10, help="Show top N results")
    p_hist.add_argument("--db", default="isat_results.db", help="Database path")

    # ── export ────────────────────────────────────────────────
    p_export = sub.add_parser("export", help="Re-generate reports from database")
    p_export.add_argument("--model", required=True, help="Model name to export")
    p_export.add_argument("--db", default="isat_results.db", help="Database path")
    p_export.add_argument("--output-dir", default="isat_output", help="Output directory")

    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command is None:
        parser.print_help()
        return 0

    try:
        if args.command == "tune":
            return _cmd_tune(args)
        elif args.command == "inspect":
            return _cmd_inspect(args)
        elif args.command == "hwinfo":
            return _cmd_hwinfo(args)
        elif args.command == "history":
            return _cmd_history(args)
        elif args.command == "export":
            return _cmd_export(args)
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

    print("\n[1/5] Fingerprinting hardware...")
    hw = fingerprint_hardware()

    print("[2/5] Analyzing model...")
    model_fp = fingerprint_model(model_path)

    print("[3/5] Generating search candidates...")
    engine = SearchEngine(
        hw, model_fp,
        warmup=args.warmup,
        runs=args.runs,
        cooldown=args.cooldown,
        max_configs=args.max_configs,
        skip_precision=args.skip_precision,
        skip_graph=args.skip_graph,
    )

    candidates = engine.generate_candidates()
    engine.print_plan(candidates)

    if args.dry_run:
        print("Dry run -- no benchmarks executed.")
        return 0

    print(f"[4/5] Benchmarking {len(candidates)} configurations...")
    runner = BenchmarkRunner(
        hw, model_fp, model_path,
        warmup=args.warmup,
        runs=args.runs,
        cooldown=args.cooldown,
        provider=args.provider,
    )
    results = runner.run_all(candidates)

    db = ResultsDB(args.db)
    db.save_batch(results, hw.fingerprint_hash, model_fp.fingerprint_hash, model_fp.name)
    db.close()

    print("[5/5] Generating reports...")
    reporter = ReportGenerator(hw, model_fp, results, output_dir=args.output_dir)
    paths = reporter.generate_all()

    print(f"\nReports saved to: {args.output_dir}/")
    print(f"  JSON : {paths['json']}")
    print(f"  HTML : {paths['html']}")
    print(f"  ENV  : {paths['env_script']}")
    print(f"  DB   : {args.db}")
    return 0


def _cmd_inspect(args) -> int:
    from isat.fingerprint.model import fingerprint_model

    model_path = args.model
    if not Path(model_path).exists():
        print(f"Error: model not found: {model_path}")
        return 1

    fp = fingerprint_model(model_path)
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


if __name__ == "__main__":
    sys.exit(main())
