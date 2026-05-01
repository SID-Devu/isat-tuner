# ISAT -- Inference Stack Auto-Tuner

[![PyPI version](https://img.shields.io/pypi/v/isat-tuner.svg)](https://pypi.org/project/isat-tuner/)
[![PyPI downloads](https://img.shields.io/pypi/dm/isat-tuner.svg)](https://pypi.org/project/isat-tuner/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/SID-Devu/isat-tuner.svg)](https://github.com/SID-Devu/isat-tuner)
[![GitHub release](https://img.shields.io/github/v/release/SID-Devu/isat-tuner)](https://github.com/SID-Devu/isat-tuner/releases)

> **66-command CLI to convert, quantize, shard, merge, encrypt, test, stream, explain, benchmark, and deploy any ONNX model -- on any GPU from any vendor.**

ISAT is a production-grade CLI toolkit for the full ONNX inference lifecycle. It converts models from any framework (PyTorch, TensorFlow, JAX, HuggingFace, TFLite, SafeTensors) to ONNX, auto-detects your hardware (AMD, NVIDIA, Intel, Apple, Qualcomm), and generates optimized inference configurations. Beyond conversion and tuning, ISAT provides advanced quantization (INT4/INT8/FP16/SmoothQuant), model sharding for multi-GPU, streaming LLM inference with KV cache, model merging and composition, explainability tools, comprehensive benchmarking, AES-256 model encryption, safety guardrails (PII/toxicity/jailbreak detection), one-command cloud deployment (Docker/K8s/SageMaker/Azure/GCP), and automated model testing with JUnit CI integration.

```bash
pip install isat-tuner

# Convert any model to ONNX + auto-detect hardware + generate inference script:
isat onnx google/vit-base-patch16-224
isat onnx facebook/opt-1.3b
isat onnx model.pt --input-shape 1,3,224,224

# Detect your hardware and get instant recommendations:
isat tune

# Detect + recommend + auto-tune a specific model:
isat tune model.onnx

# Full production tuning with cloud profile:
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

## All 66 Commands

### Model Conversion
| Command | What it does |
|---------|-------------|
| `isat onnx` | Convert any model (PyTorch, TF, JAX, HuggingFace, TFLite, SafeTensors) to ONNX + auto-tune |

### Quantization & Compression (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat quantize` | Advanced quantization: INT4 (MatMulNBits), INT8 (static QDQ), FP16, mixed-precision, SmoothQuant |

### Streaming Inference (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat stream` | Token-by-token LLM inference with KV cache, nucleus sampling (top-k/top-p), benchmark mode |

### Model Sharding & Merging (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat shard` | Split large models into N shards for multi-GPU / memory-constrained inference |
| `isat merge` | Merge/compose multiple ONNX models (chain or parallel with concat/mean/max/sum) |

### Explainability (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat explain` | Feature importance, gradient attribution, sensitivity mapping, layer activations |

### Comprehensive Benchmarking (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat benchmark-suite` | Full benchmark: latency (P50/P95/P99), throughput, memory profiling, scalability |

### Model Security & Safety (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat encrypt` | AES-256-GCM encryption, XOR obfuscation, LSB watermark fingerprinting, expiry dates |
| `isat safety` | PII detection, toxicity filtering, jailbreak pattern detection, confidence checks |

### Cloud Deployment (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat cloud-deploy` | Generate Dockerfile, K8s manifests, SageMaker handler, Azure ML, GCP Vertex, FastAPI server |

### Automated Testing (NEW in v0.10.0)
| Command | What it does |
|---------|-------------|
| `isat test` | Determinism, numerical stability, edge cases, cross-provider, memory leaks, golden tests + JUnit |

### Auto-Tuning & Search
| Command | What it does |
|---------|-------------|
| `isat tune` | Auto-detect hardware + recommend + tune (works with or without a model) |
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

## Supported Conversion Formats (`isat onnx`)

| Source Format | Extensions / IDs | Conversion Backend |
|---------------|-----------------|-------------------|
| **HuggingFace** | `org/model-name`, `hf://...` | `optimum.exporters.onnx` (primary), `torch.onnx.export` (fallback) |
| **PyTorch** | `.pt`, `.pth`, `.bin` | `torch.onnx.export` |
| **TensorFlow** | `.pb`, `SavedModel/` dirs | `tf2onnx` |
| **TFLite** | `.tflite` | `tflite2onnx` / `tf2onnx` |
| **JAX** | `.jax`, `.msgpack` | `jax2onnx` / `jax2tf` + `tf2onnx` |
| **SafeTensors** | `.safetensors` | `safetensors` + `torch.onnx.export` |
| **ONNX** | `.onnx` | Passthrough (optional `onnxsim` simplification) |

### Validated Models

| Model | Type | Params | Status |
|-------|------|--------|--------|
| `google/vit-base-patch16-224` | Vision Transformer | 86.6M | PASS |
| `openai/clip-vit-base-patch32` | Multimodal (CLIP) | 151.3M | PASS |
| `facebook/detr-resnet-50` | Object Detection | 41.6M | PASS |
| `Salesforce/blip-image-captioning-base` | Image Captioning | 196.2M | PASS |
| `distilgpt2` | LLM (small) | 81.9M | PASS |
| `facebook/opt-1.3b` | LLM (1.3B) | 1,315.7M | PASS |

---

## Installation

```bash
# From PyPI
pip install isat-tuner

# From GitHub (latest)
pip install git+https://github.com/SID-Devu/isat-tuner.git

# With all optional features
pip install "isat-tuner[all]"

# Model conversion (HuggingFace, PyTorch, TensorFlow)
pip install "isat-tuner[convert]"       # All conversion backends
pip install "isat-tuner[convert-hf]"    # HuggingFace only (optimum)
pip install "isat-tuner[convert-pt]"    # PyTorch only
pip install "isat-tuner[convert-tf]"    # TensorFlow only

# New v0.10.0 features
pip install "isat-tuner[stream]"        # Streaming LLM inference (transformers)
pip install "isat-tuner[encrypt]"       # Model encryption (cryptography)

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

### Convert Any Model to ONNX
```bash
# HuggingFace models (auto-detects architecture)
isat onnx google/vit-base-patch16-224               # Vision Transformer
isat onnx openai/clip-vit-base-patch32               # Multimodal (CLIP)
isat onnx facebook/detr-resnet-50                    # Object Detection
isat onnx Salesforce/blip-image-captioning-base      # Image Captioning
isat onnx distilgpt2                                 # LLM (small)
isat onnx facebook/opt-1.3b                          # LLM (1.3B params)

# Local models
isat onnx model.pt --input-shape 1,3,224,224         # PyTorch
isat onnx saved_model/                               # TensorFlow SavedModel
isat onnx model.tflite                               # TFLite
isat onnx weights.safetensors --input-shape 1,3,224,224  # SafeTensors

# Convert only (skip auto-tune)
isat onnx facebook/opt-1.3b --no-tune

# Simplify ONNX graph after conversion
isat onnx model.pt --simplify --input-shape 1,3,224,224
```

### Auto-Tune & Benchmark
```bash
# One-command auto-tune
isat tune model.onnx --warmup 3 --runs 5 --cooldown 60

# Use a deployment profile
isat tune model.onnx --profile edge
isat tune model.onnx --profile cloud

# Bayesian optimization (smarter than grid search)
isat tune model.onnx --bayesian --max-configs 20

# Hardware-only detection (no model needed)
isat tune
```

### Analyze & Optimize
```bash
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
```

### Deploy & Monitor
```bash
# Canary deployment (safe model rollout)
isat canary baseline.onnx candidate.onnx

# Monitor output drift
isat drift model.onnx

# Graph surgery (remove Identity/Dropout nodes)
isat surgery model.onnx --remove-op Identity --remove-op Dropout

# Launch REST API
isat serve --port 8000
```

### Quantize (NEW in v0.10.0)
```bash
# Auto-select best quantization method
isat quantize model.onnx

# INT8 static quantization (CNNs)
isat quantize model.onnx --method int8

# INT4 weight-only quantization (LLMs)
isat quantize model.onnx --method int4 --block-size 128

# SmoothQuant (transformers)
isat quantize model.onnx --method smooth --alpha 0.5

# Sensitivity analysis (find which layers to keep in FP16)
isat quantize model.onnx --sensitivity
```

### Stream LLM Inference (NEW in v0.10.0)
```bash
# Token-by-token generation
isat stream model.onnx --prompt "The future of AI is" --tokenizer gpt2

# Benchmark streaming performance
isat stream model.onnx --tokenizer gpt2 --benchmark
```

### Shard & Merge (NEW in v0.10.0)
```bash
# Analyze model for sharding
isat shard large_model.onnx --analyze

# Split into 4 shards
isat shard large_model.onnx --num-shards 4 --strategy balanced

# Merge two models into a pipeline
isat merge encoder.onnx decoder.onnx -o pipeline.onnx --mode chain

# Parallel ensemble merge
isat merge modelA.onnx modelB.onnx -o ensemble.onnx --mode parallel --aggregation mean
```

### Explain & Test (NEW in v0.10.0)
```bash
# Model explainability
isat explain model.onnx --method perturbation

# Full benchmark suite
isat benchmark-suite model.onnx --batch-sizes 1,4,16,64

# Automated testing with JUnit output
isat test model.onnx --junit

# Generate golden test file, then verify later
isat test model.onnx --generate-golden --golden golden.npz
isat test model.onnx --suite golden --golden golden.npz
```

### Encrypt & Safety (NEW in v0.10.0)
```bash
# Encrypt model weights
isat encrypt model.onnx -o model_encrypted.onnx --method encrypt --password "secret"

# Fingerprint model for IP tracking
isat encrypt model.onnx -o model_fp.onnx --method fingerprint --owner "ACME Corp"

# Safety scan on text
isat safety --input-text "my email is john@example.com"
```

### Cloud Deploy (NEW in v0.10.0)
```bash
# Generate all deployment artifacts
isat cloud-deploy model.onnx --output-dir deploy/

# Docker + K8s only
isat cloud-deploy model.onnx --target docker --output-dir deploy/
isat cloud-deploy model.onnx --target kubernetes --replicas 4 --gpu
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
from isat.converter.engine import convert, detect_format
from isat.auto_detect.detector import detect_hardware
from isat.auto_detect.recommender import generate_recommendations, format_report
from isat.auto_detect.script_gen import save_script
from isat.fingerprint import fingerprint_hardware, fingerprint_model
from isat.search import SearchEngine
from isat.pruning.pruner import ModelPruner
from isat.fusion.analyzer import FusionAnalyzer
from isat.guard.validator import InputGuard
from isat.inference_cache.cache import InferenceCache

# Convert any model to ONNX
result = convert("google/vit-base-patch16-224", output_dir="./output")
print(result.onnx_path, result.size_mb)

# Auto-detect hardware + generate recommendations
hw = detect_hardware()
report = generate_recommendations(hw, result.onnx_path)
print(format_report(report))

# Generate runnable inference script
script_path = save_script(hw, result.onnx_path, "./output")

# Auto-tune
hw_fp = fingerprint_hardware()
model_fp = fingerprint_model("model.onnx")
engine = SearchEngine(hw_fp, model_fp, warmup=3, runs=5, cooldown=60)
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

# v0.10.0 APIs
from isat.quantize.quantizer import quantize_model, ModelQuantizer
from isat.stream.generator import StreamingGenerator
from isat.shard.splitter import shard_model, ModelSharder
from isat.merge.merger import merge_models, ModelMerger
from isat.explain.explainer import explain_model, ModelExplainer
from isat.benchmark_suite.suite import run_benchmark_suite, BenchmarkSuite
from isat.encrypt.protector import protect_model, ModelProtector
from isat.safety.guardrails import SafetyGuard
from isat.cloud_deploy.deployer import deploy_model, CloudDeployer
from isat.model_test.tester import test_model, ModelTester

# Quantize (auto-selects best method)
result = quantize_model("model.onnx", "model_q.onnx", method="auto")
print(result.compression_ratio)

# Stream LLM tokens
gen = StreamingGenerator("llm.onnx", max_length=512)
tokens = gen.generate_text("Hello world", tokenizer_name="gpt2")

# Safety check
guard = SafetyGuard()
report = guard.run_all(input_text="user input here")

# Automated testing
tester = ModelTester("model.onnx")
results = tester.run_all()
print(f"{results.passed}/{results.total_tests} passed")

# Cloud deployment
deployer = CloudDeployer("model.onnx")
deployer.generate_all("deploy_output/")
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
├── cli.py                 # 66 subcommands
├── converter/             # Universal model-to-ONNX conversion engine
│   ├── engine.py          #   Format detection + dispatch
│   └── backends.py        #   HuggingFace, PyTorch, TF, JAX, TFLite, SafeTensors
├── quantize/              # Advanced quantization engine (v0.10.0)
│   └── quantizer.py       #   INT4/INT8/FP16/mixed/SmoothQuant
├── stream/                # Streaming LLM inference (v0.10.0)
│   └── generator.py       #   Token generation with KV cache
├── shard/                 # Model sharding (v0.10.0)
│   └── splitter.py        #   Graph splitting + pipeline runner
├── merge/                 # Model merging (v0.10.0)
│   └── merger.py          #   Chain/parallel composition
├── explain/               # Explainability (v0.10.0)
│   └── explainer.py       #   Feature importance, gradient, sensitivity
├── benchmark_suite/       # Comprehensive benchmarks (v0.10.0)
│   └── suite.py           #   Latency/throughput/memory/scalability
├── encrypt/               # Model protection (v0.10.0)
│   └── protector.py       #   AES-256, fingerprint, obfuscation
├── safety/                # Safety guardrails (v0.10.0)
│   └── guardrails.py      #   PII, toxicity, jailbreak detection
├── cloud_deploy/          # Cloud deployment (v0.10.0)
│   └── deployer.py        #   Docker/K8s/SageMaker/Azure/GCP
├── model_test/            # Automated testing (v0.10.0)
│   └── tester.py          #   Determinism, stability, golden tests
├── auto_detect/           # Hardware auto-detection + inference recommendations
│   ├── detector.py        #   Cross-platform GPU/CPU detection
│   ├── recommender.py     #   Vendor-specific recipe generation
│   └── script_gen.py      #   Runnable Python inference script generator
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

Optional:
- **Conversion**: `optimum[exporters]`, `torch`, `tensorflow`, `tf2onnx`, `safetensors`, `onnxsim`
- **Streaming**: `transformers` (for tokenizer in `isat stream`)
- **Encryption**: `cryptography` (for `isat encrypt` AES-256-GCM)
- **Server**: `fastapi`, `uvicorn`
- **Optimization**: `scipy`, `onnxsim`
- **Monitoring**: `prometheus-client`, `pyyaml`, `jinja2`

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v0.10.0 | May 2026 | 10 new modules: quantize, stream, shard, merge, explain, benchmark-suite, encrypt, safety, cloud-deploy, test (66 commands) |
| v0.9.1 | May 2026 | Universal model converter (`isat onnx`), 6-model E2E validation, PyTorch 2.9+ compat, multi-modal export (56 commands) |
| v0.8.x | Apr 2026 | Auto-detect hardware, generate inference scripts, Windows DirectML + MIGraphX via WinML, cross-platform GPU detection |
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
  version = {0.10.0},
  url = {https://github.com/SID-Devu/isat-tuner},
  license = {Apache-2.0}
}
```

---

## License

Apache 2.0 -- see [LICENSE](LICENSE)

Copyright 2026 Sudheer Ibrahim Daniel Devu
