# ISAT -- Inference Stack Auto-Tuner

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/isat-tuner.svg)](https://pypi.org/project/isat-tuner/)

**One command to find the fastest inference configuration for any ONNX model on any GPU.**

ISAT jointly searches across **6 dimensions** -- memory strategy, kernel backend, precision, graph transforms, batch size, and thread tuning -- then benchmarks each combination with thermal-aware cooldowns, statistical rigor, and Bayesian optimization.

```bash
pip install isat-tuner
isat tune model.onnx --profile cloud
```

---

## The Problem

Deploying an ONNX model today means manually tweaking dozens of settings:

| Setting | Choices | Impact |
|---------|---------|--------|
| `HSA_XNACK` | 0 or 1 | Up to 30% on APUs |
| `MIGRAPHX_DISABLE_MLIR` | 0 or 1 | 5-15% GEMM performance |
| `MIGRAPHX_SET_GEMM_PROVIDER` | default, rocblas, hipblaslt | 10-25% on GEMM-heavy models |
| Precision | FP32, FP16, INT8 | 2-4x throughput |
| Batch size | 1 to 256 | Linear throughput scaling |
| Graph optimization level | 0-99 | 5-20% latency reduction |
| Inter/intra op threads | 1 to N | CPU-side parallelism |

A single wrong choice can leave **40%+ performance on the table**. With 6 dimensions and 4+ choices each, there are **thousands of combinations**. Nobody has time to test them all manually.

**ISAT does it automatically.**

---

## Installation

```bash
# From PyPI (works globally for anyone)
pip install isat-tuner

# From GitHub (latest)
pip install git+https://github.com/SID-Devu/isat-tuner.git

# With all optional features
pip install "isat-tuner[all]"

# Platform-specific
pip install "isat-tuner[rocm]"      # ROCm GPU support
pip install "isat-tuner[cuda]"      # NVIDIA CUDA support
pip install "isat-tuner[server]"    # REST API server
pip install "isat-tuner[bayesian]"  # Bayesian optimization (scipy)

# Development
git clone https://github.com/SID-Devu/isat-tuner.git
cd isat && pip install -e ".[dev,all]"
```

---

## Quick Start

```bash
# One-command auto-tune
isat tune model.onnx --warmup 3 --runs 5 --cooldown 60

# Use a deployment profile
isat tune model.onnx --profile edge
isat tune model.onnx --profile cloud
isat tune model.onnx --profile latency

# Bayesian optimization (smarter than grid search)
isat tune model.onnx --bayesian --max-configs 20

# Pareto analysis (latency vs memory vs power)
isat tune model.onnx --pareto latency_ms memory_mb power_w

# CI/CD performance gate
isat tune model.onnx --gate-latency 50 --gate-throughput 100

# Generate Triton server config
isat tune model.onnx --triton-output model_repository/

# Export Prometheus metrics
isat tune model.onnx --prometheus /var/lib/prometheus/isat.prom

# Dry run (see the plan without benchmarking)
isat tune model.onnx --dry-run

# Inspect model without benchmarking
isat inspect model.onnx

# Check your hardware
isat hwinfo

# View past results
isat history --model my_model --top 10

# Launch REST API server
isat serve --port 8000

# List available profiles
isat profiles
```

---

## Search Dimensions

### 1. Memory Strategy
| Config | Environment | When to use |
|--------|-------------|-------------|
| `xnack0_default` | `HSA_XNACK=0` | Discrete GPUs, no demand paging |
| `xnack1_default` | `HSA_XNACK=1` | APUs, unified memory |
| `xnack1_coarse_grain` | XNACK=1 + coarse-grain | Large models on APU |
| `xnack1_oversubscribe` | XNACK=1 + queue limit | Models exceeding VRAM |

### 2. Kernel Backend
| Config | Environment | When to use |
|--------|-------------|-------------|
| `mlir_default` | (default) | General-purpose, fused kernels |
| `rocblas_explicit` | `MIGRAPHX_DISABLE_MLIR=1` | GEMM-heavy models |
| `hipblaslt_explicit` | `MIGRAPHX_SET_GEMM_PROVIDER=hipblaslt` | Latest GEMM tuning |
| `mlir_parallel_N` | `MIGRAPHX_GPU_COMPILE_PARALLEL=N` | Large models, faster compile |

### 3. Precision
| Config | Method | Typical speedup |
|--------|--------|-----------------|
| `fp32_native` | Original | Baseline |
| `fp16_migraphx` | MIGraphX built-in | 1.5-2x |
| `int8_qdq` | ORT static quantization | 2-4x |

### 4. Graph Transforms
| Config | Transform | Effect |
|--------|-----------|--------|
| `raw_opt99` | None + full ORT opt | Default |
| `sim_opt99` | onnxsim + full ORT opt | Remove dead ops |
| `pinned_opt99` | Freeze dynamic dims | Better kernel selection |
| `raw_opt1` | Minimal ORT opt | Debugging |

### 5. Batch Size
Auto-explores powers of 2 up to GPU memory limit.

### 6. Thread Tuning
Explores inter/intra thread counts and sequential vs parallel execution modes.

### 7. Execution Provider (Multi-Platform)
Auto-detects available providers: MIGraphX, CUDA, TensorRT, OpenVINO, ROCm, DirectML, CPU.

---

## Bayesian Optimization

Instead of brute-force grid search, ISAT can use Bayesian optimization to intelligently explore the most promising regions first:

```bash
isat tune model.onnx --bayesian --max-configs 20
```

- **Gaussian Process surrogate** with Expected Improvement acquisition
- **Tree-Parzen Estimator** fallback when scipy is unavailable
- **Early stopping** when no improvement is found
- Explores thousands of combinations by testing only 10-20

---

## Deployment Profiles

| Profile | Warmup | Runs | Cooldown | Priority | Use case |
|---------|--------|------|----------|----------|----------|
| `edge` | 3 | 10 | 30s | Latency | IoT, mobile, embedded |
| `cloud` | 5 | 20 | 120s | Throughput | Serving, batch processing |
| `latency` | 5 | 30 | 60s | P99 | Real-time inference |
| `throughput` | 3 | 15 | 120s | FPS | Max batch throughput |
| `power` | 3 | 10 | 60s | Perf/watt | Battery, thermal-constrained |
| `quick` | 1 | 3 | 15s | Latency | Fast exploration |
| `exhaustive` | 5 | 50 | 180s | Latency | Leave no stone unturned |
| `apu` | 3 | 10 | 60s | Latency | APU-specific optimization |

```bash
isat tune model.onnx --profile edge
```

---

## Output & Reports

| File | Description |
|------|-------------|
| `isat_report.html` | Interactive HTML dashboard |
| `isat_report.json` | Machine-readable results for automation |
| `best_config.sh` | Shell script -- `source` it to apply best env vars |
| `isat_results.db` | SQLite database of all historical results |
| `config.pbtxt` | Triton Inference Server config (with `--triton-output`) |
| `isat.prom` | Prometheus metrics (with `--prometheus`) |

---

## REST API Server

```bash
isat serve --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tune` | POST | Submit a tuning job |
| `/api/v1/jobs` | GET | List all jobs |
| `/api/v1/jobs/{id}` | GET | Get job status + results |
| `/api/v1/jobs/{id}/report` | GET | Get JSON report |
| `/api/v1/jobs/{id}/report/html` | GET | Get HTML dashboard |
| `/api/v1/inspect` | POST | Fingerprint a model |
| `/api/v1/hardware` | GET | Get hardware fingerprint |
| `/api/v1/history` | GET | Query historical results |
| `/health` | GET | Health check |

```bash
# Submit a tuning job
curl -X POST http://localhost:8000/api/v1/tune \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/models/model.onnx", "warmup": 3, "runs": 5}'

# Check job status
curl http://localhost:8000/api/v1/jobs/abc123
```

---

## Docker

```bash
# Build and run
docker-compose up -d

# Or standalone
docker build -t isat .
docker run --device /dev/kfd --device /dev/dri --group-add video \
  -v ./models:/models isat tune /models/model.onnx
```

---

## CI/CD Integration

### Performance Gates

```bash
# Fail CI if latency > 50ms or throughput < 100 fps
isat tune model.onnx --gate-latency 50 --gate-throughput 100
echo $?  # 0 = pass, 1 = fail
```

### GitHub Actions

A pre-built workflow is included at `.github/workflows/isat-tune.yml`. It runs tests on every push and auto-tunes on workflow dispatch.

### Regression Detection

ISAT automatically compares current results against historical baselines and flags regressions caused by driver updates, kernel changes, or model modifications.

---

## Statistical Analysis

### Outlier Detection
```python
from isat.analysis import detect_outliers, remove_outliers

cleaned, report = remove_outliers(latencies, method="mad", threshold=3.5)
print(f"Removed {report.n_outliers} outliers")
```

### Significance Testing
```python
from isat.analysis import compare_configs

result = compare_configs(latencies_a, latencies_b, confidence=0.95)
print(result.summary)
# "Config B is 12.3% faster than A (p=0.0023, SIGNIFICANT at 95% confidence)"
```

### Pareto Frontier
```python
from isat.analysis import ParetoFrontier

pareto = ParetoFrontier(results, objectives=["latency_ms", "memory_mb", "power_w"])
for point in pareto.frontier:
    print(f"{point.result.config.label}: {point.objectives}")
```

---

## Using as a Library

```python
from isat.fingerprint import fingerprint_hardware, fingerprint_model
from isat.search import SearchEngine
from isat.benchmark import BenchmarkRunner
from isat.report import ReportGenerator
from isat.database import ResultsDB
from isat.analysis import ParetoFrontier

hw = fingerprint_hardware()
model = fingerprint_model("model.onnx")

engine = SearchEngine(hw, model, warmup=3, runs=5, cooldown=60)
candidates = engine.generate_candidates()

runner = BenchmarkRunner(hw, model, "model.onnx", warmup=3, runs=5, cooldown=60)
results = runner.run_all(candidates)

# Pareto analysis: best tradeoff between latency and memory
pareto = ParetoFrontier(results, objectives=["latency_ms", "memory_mb"])
best = pareto.recommend(priority="latency_ms")

# Save and report
db = ResultsDB("isat_results.db")
db.save_batch(results, hw.fingerprint_hash, model.fingerprint_hash, model.name)

reporter = ReportGenerator(hw, model, results)
reporter.generate_all()
```

---

## Architecture

```
isat/
├── cli.py                     # 9 subcommands: tune, inspect, hwinfo, history,
│                               #   export, compare, serve, triton, profiles
├── fingerprint/
│   ├── hardware.py            # GPU detection, memory topology, XNACK
│   └── model.py               # ONNX analysis, op counting, classification
├── search/
│   ├── memory.py              # XNACK, coarse-grain, oversubscription
│   ├── kernel.py              # MLIR, rocBLAS, hipBLASlt, parallel compile
│   ├── precision.py           # FP32, FP16, INT8 quantization
│   ├── graph.py               # onnxsim, shape pinning, ORT opt levels
│   ├── batch.py               # Batch size auto-exploration
│   ├── threading.py           # Inter/intra threads, execution mode
│   ├── provider.py            # Multi-provider (CUDA, TensorRT, OpenVINO, etc.)
│   ├── bayesian.py            # Bayesian optimization (GP + TPE)
│   └── engine.py              # Cartesian product + pruning + orchestration
├── benchmark/
│   ├── runner.py              # ORT session lifecycle + latency measurement
│   ├── stats.py               # P50/P95/P99, mean, std, CV
│   ├── thermal.py             # Temp/power monitoring + cooldown enforcement
│   └── multi_gpu.py           # Multi-GPU discovery + workload distribution
├── analysis/
│   ├── outliers.py            # MAD + IQR outlier detection
│   ├── significance.py        # Welch's t-test for config comparison
│   ├── pareto.py              # Multi-objective Pareto frontier
│   └── regression.py          # Perf regression detection vs baselines
├── report/
│   └── generator.py           # JSON, HTML, console, best_config.sh
├── database/
│   └── store.py               # SQLite results DB + indexed queries
├── server/
│   └── app.py                 # FastAPI REST API with job management
├── integrations/
│   ├── triton.py              # Triton config.pbtxt generator
│   ├── metrics.py             # Prometheus exposition format
│   └── ci.py                  # Performance gates + GitHub Actions workflow
├── profiles/
│   └── presets.py             # 8 deployment profiles
└── utils/
    ├── sysfs.py               # /sys/class/drm, /proc readers
    ├── rocm.py                # rocminfo parsing, rocm-smi wrappers
    └── onnx_utils.py          # ONNX model deep analysis
```

---

## Requirements

- Python >= 3.9
- `onnxruntime` (CPU), `onnxruntime-rocm` (ROCm), or `onnxruntime-gpu` (CUDA)
- `onnx`, `numpy`

Optional:
- `scipy` -- Bayesian optimization
- `fastapi` + `uvicorn` -- REST API server
- `onnxsim` -- graph simplification
- `prometheus-client` -- metrics export

---

## License

Apache 2.0 -- see [LICENSE](LICENSE)
