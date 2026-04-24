# ISAT -- Inference Stack Auto-Tuner

[![PyPI version](https://img.shields.io/pypi/v/isat-tuner.svg)](https://pypi.org/project/isat-tuner/)
[![PyPI downloads](https://img.shields.io/pypi/dm/isat-tuner.svg)](https://pypi.org/project/isat-tuner/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SID-Devu/isat-tuner.svg)](https://github.com/SID-Devu/isat-tuner)
[![GitHub release](https://img.shields.io/github/v/release/SID-Devu/isat-tuner)](https://github.com/SID-Devu/isat-tuner/releases)

> **One command to find the fastest inference configuration for any ONNX model on any GPU.**

ISAT is a production-grade CLI toolkit for ONNX inference optimization. It jointly searches across memory strategy, kernel backend, precision, graph transforms, batch size, and thread tuning -- then benchmarks each combination with thermal-aware cooldowns, statistical rigor, and Bayesian optimization.

```bash
# Recommended (no venv needed, PATH handled automatically):
pipx install isat-tuner

# Or with pip:
pip install isat-tuner

isat tune model.onnx --profile cloud
```

> **Install note:** On modern Linux (Ubuntu 23.04+, Debian 12+), bare `pip install` is blocked by
> [PEP 668](https://peps.python.org/pep-0668/). Use **`pipx install isat-tuner`** instead --
> it creates an isolated environment and puts `isat` on your PATH automatically.
> If you don't have pipx: `sudo apt install pipx && pipx ensurepath`.

---

## Why ISAT?

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

## All 55 Commands

### Auto-Tuning & Search
| Command | What it does |
|---------|-------------|
| `isat tune` | Auto-tune an ONNX model across all search dimensions |
| `isat profiles` | List available tuning profiles (edge, cloud, latency, etc.) |
| `isat init` | Generate a default `isat.yaml` config file |
| `isat batch` | Find optimal batch size (latency vs throughput tradeoff) |
| `isat shapes` | Benchmark model across dynamic input shapes |

### Model Analysis & Inspection
| Command | What it does |
|---------|-------------|
| `isat inspect` | Deep fingerprint a model without benchmarking |
| `isat diff` | Structural diff between two ONNX models |
| `isat fusion` | Analyze operator fusion (fused vs unfused ops) |
| `isat attention` | Profile attention heads in transformer models |
| `isat weight-sharing` | Detect shared/tied weights across layers |
| `isat visualize` | Visualize ONNX graph (DOT, ASCII, histogram) |
| `isat scan` | Security and compliance scan of ONNX model |
| `isat compat-matrix` | Operator compatibility across providers |

### Benchmarking & Profiling
| Command | What it does |
|---------|-------------|
| `isat profile` | Decompose latency into load/compile/inference phases |
| `isat llm-bench` | LLM token throughput (TPS, TTFT, ITL with P95) |
| `isat compiler-compare` | Benchmark same model across ALL execution providers |
| `isat stress` | Sustained/burst/ramp stress testing |
| `isat leak-check` | Detect memory leaks during inference |
| `isat power` | Profile power efficiency (perf/watt, energy/inference) |
| `isat thermal` | Thermal throttle detection during inference |
| `isat gpu-frag` | GPU memory fragmentation analysis |
| `isat warmup` | Analyze warmup behavior, find optimal iterations |

### Model Optimization
| Command | What it does |
|---------|-------------|
| `isat optimize` | Optimize ONNX model (simplify, quantize, export) |
| `isat prune` | Prune model weights (magnitude/percentage/global) |
| `isat surgery` | ONNX graph surgery (remove/rename/extract nodes) |
| `isat quant-sensitivity` | Per-layer quantization sensitivity analysis |
| `isat distill` | Knowledge distillation planning for teacher models |

### Production Deployment
| Command | What it does |
|---------|-------------|
| `isat serve` | Launch REST API server (FastAPI) |
| `isat triton` | Generate Triton Inference Server config |
| `isat canary` | Canary deployment between two model versions |
| `isat ensemble` | Run model ensemble with aggregation |
| `isat guard` | Validate inference inputs against model schema |
| `isat codegen` | Generate standalone C++ inference code |

### Monitoring & Operations
| Command | What it does |
|---------|-------------|
| `isat alerts` | Inference alert rules engine (P99, error rate, GPU temp) |
| `isat trace` | OpenTelemetry-compatible request tracing |
| `isat drift` | Monitor output quality and detect confidence drift |
| `isat regression` | Performance regression detection across versions |
| `isat replay` | Record or replay inference requests |

### Planning & Cost
| Command | What it does |
|---------|-------------|
| `isat cost` | Estimate cloud inference cost |
| `isat sla` | Validate inference against SLA requirements |
| `isat recommend` | Hardware recommendation for a model |
| `isat migrate` | Generate migration plan between providers |
| `isat memory` | Estimate memory usage and predict OOM risk |

### Infrastructure & Utilities
| Command | What it does |
|---------|-------------|
| `isat hwinfo` | Print hardware fingerprint |
| `isat doctor` | Pre-flight system health and compatibility check |
| `isat history` | Show past tuning results from database |
| `isat export` | Re-generate reports from database |
| `isat compare` | Compare two configs with significance testing |
| `isat abtest` | A/B test two models with statistical rigor |
| `isat snapshot` | Capture environment state for reproducibility |
| `isat cache` | Manage compilation cache (MIGraphX/ORT) |
| `isat zoo` | List pre-tuned model configurations |
| `isat download` | Download ONNX model by name or URL |
| `isat registry` | Model version registry (register, promote, diff) |
| `isat pipeline` | Profile multi-model inference pipeline |

---

## Installation

```bash
# From PyPI
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

# Bayesian optimization (smarter than grid search)
isat tune model.onnx --bayesian --max-configs 20

# Inspect model
isat inspect model.onnx

# Check your hardware
isat hwinfo

# System health check
isat doctor

# LLM token benchmarking
isat llm-bench model.onnx --seq-lengths 32,64,128,256

# Compare across all available providers
isat compiler-compare model.onnx

# Prune a model
isat prune model.onnx --strategy magnitude --sparsity 0.5

# Analyze operator fusion
isat fusion model.onnx

# Generate C++ inference code
isat codegen model.onnx --output-dir cpp_build/

# Canary deployment (safe model rollout)
isat canary baseline.onnx candidate.onnx

# Monitor output drift
isat drift model.onnx

# Graph surgery (remove Identity/Dropout nodes)
isat surgery model.onnx --remove-op Identity --remove-op Dropout

# Launch REST API
isat serve --port 8000
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

### 5. Batch Size
Auto-explores powers of 2 up to GPU memory limit.

### 6. Thread Tuning
Explores inter/intra thread counts and sequential vs parallel execution modes.

### 7. Execution Provider (Multi-Platform)
Auto-detects available providers: MIGraphX, CUDA, TensorRT, OpenVINO, ROCm, DirectML, CPU.

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

---

## Output & Reports

| File | Description |
|------|-------------|
| `isat_report.html` | Interactive HTML dashboard |
| `isat_report.json` | Machine-readable results for automation |
| `best_config.sh` | Shell script -- `source` it to apply best env vars |
| `isat_results.db` | SQLite database of all historical results |
| `config.pbtxt` | Triton Inference Server config |
| `isat.prom` | Prometheus metrics |
| `traces_*.json` | OpenTelemetry-compatible trace export |
| `isat_inference.cpp` | Generated C++ inference code |

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

---

## Docker

```bash
docker-compose up -d

# Or standalone
docker build -t isat .
docker run --device /dev/kfd --device /dev/dri --group-add video \
  -v ./models:/models isat tune /models/model.onnx
```

---

## Using as a Library

```python
from isat.fingerprint import fingerprint_hardware, fingerprint_model
from isat.search import SearchEngine
from isat.benchmark import BenchmarkRunner
from isat.analysis import ParetoFrontier
from isat.pruning.pruner import ModelPruner
from isat.fusion.analyzer import FusionAnalyzer
from isat.guard.validator import InputGuard
from isat.inference_cache.cache import InferenceCache

# Auto-tune
hw = fingerprint_hardware()
model = fingerprint_model("model.onnx")
engine = SearchEngine(hw, model, warmup=3, runs=5, cooldown=60)
candidates = engine.generate_candidates()

# Prune a model
pruner = ModelPruner("model.onnx")
result = pruner.prune(strategy="magnitude", sparsity=0.5)

# Analyze fusion
analyzer = FusionAnalyzer("model.onnx")
report = analyzer.analyze()

# Validate inputs before inference
guard = InputGuard(model_path="model.onnx")
result = guard.validate({"input": my_tensor})

# Cache inference results
cache = InferenceCache(max_memory_entries=1000, disk_cache_dir="./cache")
```

---

## CI/CD Integration

```bash
# Fail CI if latency > 50ms or throughput < 100 fps
isat tune model.onnx --gate-latency 50 --gate-throughput 100
echo $?  # 0 = pass, 1 = fail
```

---

## Architecture

```
isat/
├── cli.py                 # 55 subcommands
├── fingerprint/           # Hardware + model fingerprinting
├── search/                # 7-dimension search engine + Bayesian optimization
├── benchmark/             # Runner, stats, thermal monitoring, multi-GPU
├── analysis/              # Outliers, significance, Pareto, regression
├── pruning/               # Magnitude/percentage/global weight pruning
├── distillation/          # Knowledge distillation planning
├── fusion/                # Operator fusion analysis
├── attention/             # Transformer attention head profiling
├── surgery/               # ONNX graph surgery (remove/rename/extract)
├── guard/                 # Input validation and schema enforcement
├── ensemble/              # Multi-model ensemble with aggregation
├── canary/                # Canary deployment with auto-rollback
├── alerts/                # Alert rules engine (P99, error rate, temp)
├── tracing/               # OpenTelemetry-compatible request tracing
├── inference_cache/       # LRU + disk inference result caching
├── replay/                # Record and replay inference requests
├── output_monitor/        # Confidence drift detection (KS test)
├── llm_bench/             # LLM token throughput (TPS, TTFT, ITL)
├── compiler_compare/      # Cross-provider benchmark comparison
├── codegen/               # ONNX to C++ code generator
├── weight_analysis/       # Weight sharing detection
├── continuous_profiler/   # Always-on production profiling
├── gpu_frag/              # GPU memory fragmentation analysis
├── batching/              # Dynamic request batching engine
├── scanner/               # ONNX security/compliance scanner
├── compat_matrix/         # Operator compatibility matrix
├── thermal/               # Thermal throttle detection
├── quant_sensitivity/     # Per-layer quantization sensitivity
├── pipeline/              # Multi-model pipeline optimizer
├── recommend/             # Hardware recommendation engine
├── registry/              # Model version registry
├── regression/            # Performance regression detector
├── optimizer/             # Graph transforms + quantization
├── profiler/              # Latency decomposition
├── cost/                  # Cloud cost estimation
├── sla/                   # SLA validation
├── memory/                # Memory planning + OOM prediction
├── power/                 # Power efficiency profiling
├── health/                # System health checks
├── cache/                 # Compilation cache management
├── migration/             # Provider migration planning
├── warmup/                # Warmup analysis
├── shapes/                # Dynamic shape benchmarking
├── hub/                   # Model download from HuggingFace/ONNX Zoo
├── scheduler/             # Adaptive batch scheduling
├── snapshot/              # Environment snapshotting
├── abtesting/             # A/B testing framework
├── visualizer/            # Graph visualization (DOT, ASCII)
├── stress/                # Stress testing + memory leak detection
├── notifications/         # Webhook, Slack, console notifications
├── server/                # FastAPI REST API
├── integrations/          # Triton, Prometheus, CI/CD
├── database/              # SQLite results database
├── report/                # JSON, HTML, console reports
├── config/                # YAML/JSON config loader
├── profiles/              # 8 deployment profiles
├── model_zoo.py           # Pre-tuned model configurations
├── plugins.py             # Plugin system with lifecycle hooks
├── retry.py               # Exponential backoff retry logic
└── utils/                 # sysfs, rocm, onnx utilities
```

---

## Requirements

- Python >= 3.9
- `onnxruntime` (CPU), `onnxruntime-rocm` (ROCm), or `onnxruntime-gpu` (CUDA)
- `onnx`, `numpy`

Optional: `scipy`, `fastapi`, `uvicorn`, `onnxsim`, `prometheus-client`, `pyyaml`, `jinja2`

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.7.x | Apr 2026 | Pruning, distillation, fusion analysis, LLM bench, compiler comparison, replay, drift monitor, codegen (55 commands) |
| v0.6.0 | Apr 2026 | Tracing, canary deploy, alerts, graph surgery, caching, input guard, ensemble, GPU frag (45 commands) |
| v0.5.0 | Apr 2026 | Regression detector, security scanner, compat matrix, thermal monitor, quant sensitivity, pipeline optimizer, HW recommender, model registry (38 commands) |
| v0.4.0 | Apr 2026 | Dynamic shapes, model hub, power profiler, memory planner, A/B testing, graph visualizer, env snapshot, batch scheduler (30 commands) |
| v0.3.0 | Apr 2026 | Latency profiler, cost estimator, SLA validator, health checker, migration tool, notifications (22 commands) |
| v0.2.0 | Apr 2026 | Config system, model optimization, stress testing, plugin system, model zoo (14 commands) |
| v0.1.0 | Apr 2026 | Initial release: auto-tuning, Bayesian search, multi-provider support (9 commands) |

---

## Citation

```bibtex
@software{isat_tuner,
  author = {Sudheer Ibrahim Daniel Devu},
  title = {ISAT: Inference Stack Auto-Tuner},
  year = {2026},
  version = {0.7.2},
  url = {https://github.com/SID-Devu/isat-tuner},
  license = {Apache-2.0}
}
```

---

## License

Apache 2.0 -- see [LICENSE](LICENSE)

Copyright 2026 Sudheer Ibrahim Daniel Devu
