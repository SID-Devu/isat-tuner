# ISAT -- Inference Stack Auto-Tuner

Automatically find the fastest ONNX Runtime inference configuration for any model on any GPU.

ISAT jointly searches across **memory strategy**, **kernel backend**, **precision**, and **graph transformations** -- then benchmarks each combination with thermal-aware cooldowns and reports the best config with a single command.

---

## Why ISAT?

Deploying an ONNX model on a GPU today requires manually experimenting with dozens of environment variables, quantization options, and execution provider settings. A single wrong choice can leave 40%+ performance on the table. ISAT automates this entire process:

1. **Fingerprint** your GPU and model to understand the hardware-model interaction
2. **Generate** a smart search space (not brute force -- prunes bad combos)
3. **Benchmark** each config with statistical rigor (warmup, cooldown, percentiles)
4. **Report** the winner with exportable env vars, JSON, and a visual HTML dashboard

---

## Quick Start

```bash
# Install
cd isat
pip install -e .

# One-command auto-tune
isat tune path/to/model.onnx --warmup 3 --runs 5 --cooldown 60

# Just inspect a model (no benchmarking)
isat inspect path/to/model.onnx

# Check your GPU
isat hwinfo

# See what ISAT would test without running benchmarks
isat tune path/to/model.onnx --dry-run

# View past results
isat history --model my_model --top 10
```

---

## What ISAT Searches

### Memory Strategy
| Config | What it does |
|--------|-------------|
| `xnack0_default` | Standard GPU memory (`HSA_XNACK=0`) |
| `xnack1_default` | Demand paging enabled (`HSA_XNACK=1`) |
| `xnack1_coarse_grain` | XNACK + coarse-grained coherence |
| `xnack1_oversubscribe` | XNACK tuned for memory oversubscription |

### Kernel Backend
| Config | What it does |
|--------|-------------|
| `mlir_default` | MIGraphX MLIR fusion (fuses GEMMs into compound kernels) |
| `rocblas_explicit` | Disable MLIR, use rocBLAS for standalone GEMM calls |
| `hipblaslt_explicit` | Disable MLIR, use hipBLASlt for GEMMs |
| `mlir_parallel_N` | MLIR with parallel compilation |

### Precision
| Config | What it does |
|--------|-------------|
| `fp32_native` | Original model precision |
| `fp16_migraphx` | MIGraphX built-in FP16 conversion |
| `int8_qdq` | ORT static INT8 quantization (QDQ format) |

### Graph Transforms
| Config | What it does |
|--------|-------------|
| `raw_opt99` | No transforms, full ORT optimization |
| `sim_opt99` | onnxsim simplification + full optimization |
| `pinned_opt99` | Dynamic shapes frozen to batch=1 |
| `raw_opt1` | Minimal ORT optimization (for debugging) |

---

## Output

After tuning, ISAT generates:

| File | Description |
|------|-------------|
| `isat_output/isat_report.html` | Visual dashboard with rankings and env vars |
| `isat_output/isat_report.json` | Machine-readable results for CI/CD integration |
| `isat_output/best_config.sh` | Shell script -- `source` it to apply the best config |
| `isat_results.db` | SQLite database of all historical results |

---

## CLI Reference

```
isat tune MODEL.onnx [OPTIONS]
    --warmup N          Warmup iterations (default: 3)
    --runs N            Measured iterations (default: 5)
    --cooldown SECS     Cooldown between configs (default: 60)
    --max-configs N     Limit configs to test (0=all)
    --provider NAME     ORT execution provider
    --skip-precision    Skip precision search dimension
    --skip-graph        Skip graph transform search dimension
    --output-dir DIR    Output directory (default: isat_output)
    --db PATH           Database path (default: isat_results.db)
    --dry-run           Show plan without benchmarking
    -v, --verbose       Debug logging

isat inspect MODEL.onnx
    Fingerprint a model: ops, shapes, params, class, memory estimate

isat hwinfo
    Print GPU and system fingerprint

isat history [--model NAME] [--top N] [--db PATH]
    Query past tuning results

isat export --model NAME [--db PATH] [--output-dir DIR]
    Re-generate reports from historical data
```

---

## Using as a Library

```python
from isat.fingerprint import fingerprint_hardware, fingerprint_model
from isat.search import SearchEngine
from isat.benchmark import BenchmarkRunner
from isat.report import ReportGenerator

hw = fingerprint_hardware()
model = fingerprint_model("model.onnx")

engine = SearchEngine(hw, model, warmup=3, runs=5, cooldown=60)
candidates = engine.generate_candidates()

runner = BenchmarkRunner(hw, model, "model.onnx", warmup=3, runs=5, cooldown=60)
results = runner.run_all(candidates)

reporter = ReportGenerator(hw, model, results)
reporter.generate_all()
```

---

## Architecture

```
isat/
├── cli.py                  # CLI entry point (tune, inspect, hwinfo, history, export)
├── fingerprint/
│   ├── hardware.py         # GPU detection, memory topology, XNACK capability
│   └── model.py            # ONNX analysis: ops, shapes, params, classification
├── search/
│   ├── memory.py           # XNACK, coarse-grain, pool configs
│   ├── kernel.py           # MLIR, rocBLAS, hipBLASlt, parallel compile
│   ├── precision.py        # FP32, FP16, INT8 quantization
│   ├── graph.py            # onnxsim, shape pinning, ORT opt levels
│   └── engine.py           # Cartesian product + pruning + plan display
├── benchmark/
│   ├── runner.py           # ORT session management + latency measurement
│   ├── stats.py            # Percentile computation (P50/P95/P99, CV)
│   └── thermal.py          # GPU temp/power monitoring + cooldown enforcement
├── report/
│   └── generator.py        # JSON, HTML, console reports + best_config.sh
├── database/
│   └── store.py            # SQLite results storage + query API
└── utils/
    ├── sysfs.py            # /sys/class/drm, /proc/meminfo readers
    ├── rocm.py             # rocminfo parsing, rocm-smi wrappers
    └── onnx_utils.py       # ONNX model deep analysis
```

---

## Requirements

- Python >= 3.9
- `onnxruntime` (CPU) or `onnxruntime-rocm` (ROCm GPU)
- `onnx`, `numpy`
- Optional: `onnxsim` (for graph simplification), `jinja2` (for advanced reports)
- ROCm 6.x (for GPU features)

---

## License

Apache 2.0
