#!/usr/bin/env python3
"""Quick programmatic tuning example using ISAT as a library."""

import sys
from pathlib import Path

from isat.fingerprint import fingerprint_hardware, fingerprint_model
from isat.search import SearchEngine
from isat.benchmark import BenchmarkRunner
from isat.report import ReportGenerator
from isat.database import ResultsDB


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.onnx>")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    hw = fingerprint_hardware()
    model_fp = fingerprint_model(model_path)

    print(f"GPU: {hw.gpu_name} ({hw.gfx_target})")
    print(f"Model: {model_fp.name} ({model_fp.size_class}, {model_fp.estimated_memory_mb:.0f} MB)")

    engine = SearchEngine(
        hw, model_fp,
        warmup=2,
        runs=3,
        cooldown=30,
        skip_precision=True,
    )

    candidates = engine.generate_candidates()
    engine.print_plan(candidates)

    runner = BenchmarkRunner(
        hw, model_fp, model_path,
        warmup=2, runs=3, cooldown=30,
    )

    results = runner.run_all(candidates)

    db = ResultsDB("isat_results.db")
    db.save_batch(results, hw.fingerprint_hash, model_fp.fingerprint_hash, model_fp.name)
    db.close()

    reporter = ReportGenerator(hw, model_fp, results, output_dir="isat_output")
    reporter.generate_all()


if __name__ == "__main__":
    main()
