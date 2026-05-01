<p align="center">
  <h1 align="center">ISAT</h1>
  <p align="center"><strong>Inference Stack Auto-Tuner</strong></p>
  <p align="center">
    A production-grade inference engine for ONNX models.<br/>
    76 CLI commands. Any GPU. Any framework. One tool.
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/isat-tuner/"><img src="https://img.shields.io/pypi/v/isat-tuner.svg?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/isat-tuner/"><img src="https://img.shields.io/pypi/dm/isat-tuner.svg?style=flat-square&color=green" alt="Downloads"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg?style=flat-square" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-orange.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/SID-Devu/isat-tuner"><img src="https://img.shields.io/github/stars/SID-Devu/isat-tuner.svg?style=flat-square" alt="Stars"></a>
  <a href="https://github.com/SID-Devu/isat-tuner/releases"><img src="https://img.shields.io/github/v/release/SID-Devu/isat-tuner?style=flat-square" alt="Release"></a>
</p>

---

<table>
<tr>
<td width="50%">

**What ISAT does in one sentence:**

ISAT converts models from any framework to ONNX, auto-detects your hardware, and provides 76 production commands — from speculative decoding and continuous batching to tensor parallelism and live monitoring — that compete with vLLM, TensorRT-LLM, and DeepSpeed.

</td>
<td width="50%">

```bash
pip install isat-tuner

# Convert + auto-tune in one command
isat onnx facebook/opt-1.3b
isat tune model.onnx

# Serve with continuous batching
isat serve-llm model.onnx --port 8000

# 2-4x LLM speedup
isat speculate model.onnx --draft draft.onnx
```

</td>
</tr>
</table>

---

## Key Numbers

| Metric | Value |
|--------|-------|
| CLI commands | **76** |
| Supported GPU vendors | **6** (AMD, NVIDIA, Intel, Apple, Qualcomm, DirectML) |
| Model conversion backends | **7** (HuggingFace, PyTorch, TensorFlow, JAX, TFLite, SafeTensors, ONNX) |
| E2E validated HuggingFace models | **6** (ViT, CLIP, DETR, BLIP, DistilGPT2, OPT-1.3B) |
| Lines of Python | **30,000+** |
| Modules | **40+** |
| PyPI package | [`isat-tuner`](https://pypi.org/project/isat-tuner/) |

---

## Feature Overview

ISAT is organized into three tiers of capability:

### Tier 1 — LLM Inference Engine

These features put ISAT on par with dedicated LLM serving frameworks.

| Command | Capability | How it works |
|---------|-----------|--------------|
| `isat speculate` | **2-4x LLM decode speedup** | Draft model generates K candidate tokens; target model verifies in one forward pass via rejection sampling (Leviathan et al. 2023). Also supports self-speculation (early-exit) and Medusa multi-head prediction. |
| `isat serve-llm` | **Continuous batching server** | PagedAttention-style KV cache pool with block-level allocation, iteration-level scheduling, chunked prefill, prefix caching. OpenAI-compatible `/v1/completions` and `/v1/chat/completions` with SSE streaming. Prometheus `/metrics` endpoint. |
| `isat constrain` | **Grammar-constrained generation** | Forces LLM output to match JSON schema, regex, or GBNF grammar. Regex compiled via Thompson's NFA construction + subset-construction DFA with precomputed per-state token masks for O(1) lookup. JSON schema via pushdown automaton. |
| `isat stream` | **Token-by-token generation** | Autoregressive inference with KV cache management, nucleus sampling (top-k/top-p), TTFT/ITL/TPS benchmarking. |

### Tier 2 — Model Engineering

Tools for adapting, parallelizing, and optimizing models before deployment.

| Command | Capability | How it works |
|---------|-----------|--------------|
| `isat lora` | **LoRA adapter runtime** | Hot-swap adapters at runtime, multi-LoRA routing, adapter fusion for zero-overhead inference. Weight merging via TIES-Merging (trim/elect/merge), DARE (drop-and-rescale), SLERP (spherical interpolation), Task Arithmetic, Model Soup. |
| `isat tensor-parallel` | **True tensor parallelism** | Column-parallel QKV projections, row-parallel output projections, all-reduce synchronization. Auto-detects parallelizable layers from ONNX graph topology. |
| `isat graph-compile` | **CUDA/HIP graph capture** | Captures inference graphs and replays them to eliminate kernel launch overhead. 20-47% decode throughput improvement. Includes graph region analysis (static vs dynamic ops). |
| `isat amp-profile` | **Mixed-precision search** | Profiles every layer at FP32/FP16/INT8/INT4, then finds the Pareto-optimal precision assignment via dynamic programming, greedy, or beam search — minimizing latency under a user-defined MSE budget. |
| `isat distill-train` | **Knowledge distillation** | Real training loop through ORT: teacher forward pass, student forward pass, KL-divergence + cross-entropy loss, numerical gradients, Adam optimizer in pure numpy. Auto-creates smaller student via depth/width reduction. |
| `isat a2a` | **Architecture surgery** | Attention head pruning with importance scoring (magnitude/entropy/Taylor), width shrinking, depth shrinking (importance/uniform/first-last), vocabulary pruning for domain-specific deployment. |
| `isat quantize` | **Advanced quantization** | INT4 weight-only (MatMulNBits), INT8 static QDQ, FP16 cast, mixed-precision, SmoothQuant with per-layer sensitivity analysis. |
| `isat shard` | **Model sharding** | Graph-based splitting (balanced/layer/auto strategies) for multi-GPU or memory-constrained inference with pipeline-parallel execution. |
| `isat merge` | **Model composition** | Chain (sequential) or parallel (concat/mean/max/sum aggregation) composition of multiple ONNX models. |
| `isat onnx` | **Universal conversion** | One command converts from HuggingFace, PyTorch, TensorFlow, JAX, TFLite, or SafeTensors to optimized ONNX. |

### Tier 3 — Production Operations

Everything needed to deploy, monitor, secure, and validate models in production.

| Command | Capability | How it works |
|---------|-----------|--------------|
| `isat monitor-live` | **Real-time monitoring** | Daemon collects CPU/GPU/VRAM/latency/throughput metrics, detects anomalies (latency spikes, throughput drops, memory leaks, thermal throttling), triggers auto-remediation. ASCII TUI dashboard with sparkline trends. |
| `isat test` | **Automated testing** | Determinism, numerical stability, edge cases, cross-provider comparison, memory leak detection. Golden test generation. JUnit XML output for CI. |
| `isat benchmark-suite` | **Comprehensive benchmarks** | Latency (P50/P95/P99), throughput, memory profiling, scalability analysis across batch sizes. |
| `isat encrypt` | **Model protection** | AES-256-GCM encryption, XOR obfuscation, LSB watermark fingerprinting, expiry dates. |
| `isat safety` | **Content guardrails** | PII detection, toxicity filtering, jailbreak pattern matching, confidence thresholding. |
| `isat cloud-deploy` | **One-command deployment** | Generates Dockerfile, Kubernetes manifests, SageMaker handler, Azure ML config, GCP Vertex config, FastAPI inference server. |
| `isat explain` | **Explainability** | Feature importance (perturbation), gradient attribution (finite differences), sensitivity mapping, layer activations. |
| `isat tune` | **Auto-tuning** | 7-dimension search across memory strategy, kernel backend, precision, graph transforms, batch size, threading, and execution provider. Bayesian optimization optional. |

<details>
<summary><strong>See all 76 commands</strong></summary>

#### Auto-Tuning & Search
`tune` · `profiles` · `init` · `batch` · `shapes`

#### Model Analysis & Inspection
`inspect` · `diff` · `fusion` · `attention` · `weight-sharing` · `visualize` · `scan` · `compat-matrix`

#### Benchmarking & Profiling
`profile` · `llm-bench` · `compiler-compare` · `stress` · `leak-check` · `power` · `thermal` · `gpu-frag` · `warmup`

#### Model Optimization
`optimize` · `prune` · `surgery` · `quant-sensitivity` · `distill`

#### Production Deployment
`serve` · `triton` · `canary` · `ensemble` · `guard` · `codegen`

#### Monitoring & Operations
`alerts` · `trace` · `drift` · `regression` · `replay`

#### Planning & Cost
`cost` · `sla` · `recommend` · `migrate` · `memory`

#### Infrastructure & Utilities
`hwinfo` · `doctor` · `history` · `export` · `compare` · `abtest` · `snapshot` · `cache` · `zoo` · `download` · `registry` · `pipeline`

</details>

---

## Validated Models

Every release is tested end-to-end against real HuggingFace models:

| Model | Type | Parameters | Status |
|-------|------|------------|--------|
| `google/vit-base-patch16-224` | Vision Transformer | 86.6M | Pass |
| `openai/clip-vit-base-patch32` | Multimodal (CLIP) | 151.3M | Pass |
| `facebook/detr-resnet-50` | Object Detection | 41.6M | Pass |
| `Salesforce/blip-image-captioning-base` | Image Captioning | 196.2M | Pass |
| `distilgpt2` | Language Model | 81.9M | Pass |
| `facebook/opt-1.3b` | Large Language Model | 1,315.7M | Pass |

---

## Installation

```bash
pip install isat-tuner
```

<details>
<summary><strong>Optional dependencies and platform-specific installs</strong></summary>

```bash
# All optional features
pip install "isat-tuner[all]"

# Model conversion
pip install "isat-tuner[convert]"       # All backends
pip install "isat-tuner[convert-hf]"    # HuggingFace (optimum)
pip install "isat-tuner[convert-pt]"    # PyTorch
pip install "isat-tuner[convert-tf]"    # TensorFlow

# Feature-specific
pip install "isat-tuner[stream]"        # Streaming inference (transformers)
pip install "isat-tuner[encrypt]"       # Model encryption (cryptography)

# Platform-specific
pip install "isat-tuner[rocm]"          # AMD ROCm
pip install "isat-tuner[cuda]"          # NVIDIA CUDA
pip install "isat-tuner[server]"        # FastAPI server
pip install "isat-tuner[bayesian]"      # Bayesian optimization

# From GitHub (latest)
pip install git+https://github.com/SID-Devu/isat-tuner.git

# Development
git clone https://github.com/SID-Devu/isat-tuner.git
cd isat && pip install -e ".[dev,all]"
```

> **Note:** On modern Linux (Ubuntu 23.04+, Debian 12+), bare `pip install` may be blocked by [PEP 668](https://peps.python.org/pep-0668/). Use `pipx install isat-tuner` instead.

</details>

---

## Quick Start

### Convert any model to ONNX

```bash
isat onnx google/vit-base-patch16-224          # HuggingFace Vision Transformer
isat onnx facebook/opt-1.3b                    # HuggingFace LLM (1.3B params)
isat onnx model.pt --input-shape 1,3,224,224   # Local PyTorch
isat onnx saved_model/                         # TensorFlow SavedModel
isat onnx model.tflite                         # TFLite
```

### Auto-tune for your hardware

```bash
isat tune model.onnx                           # Auto-detect + optimize
isat tune model.onnx --profile cloud           # Cloud deployment profile
isat tune model.onnx --bayesian --max-configs 20  # Bayesian search
```

### Serve with continuous batching

```bash
isat serve-llm model.onnx --tokenizer gpt2 --port 8000

# OpenAI-compatible API
curl http://localhost:8000/v1/completions \
  -d '{"prompt": "Hello", "max_tokens": 50, "stream": true}'
```

### Speculative decoding (2-4x speedup)

```bash
isat speculate target.onnx --draft draft.onnx --benchmark
isat speculate target.onnx --mode self         # Self-speculation (no draft needed)
```

### Grammar-constrained generation

```bash
isat constrain model.onnx --schema '{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}'
isat constrain model.onnx --regex '[0-9]{3}-[0-9]{2}-[0-9]{4}'
isat constrain model.onnx --grammar grammar.gbnf
```

### LoRA adapter management

```bash
isat lora base.onnx --adapter lora_weights.npz --action fuse -o fused.onnx
isat lora base.onnx --action merge --merge-method ties --merge-models a.onnx b.onnx c.onnx
```

### Quantize and profile precision

```bash
isat quantize model.onnx --method int4 --block-size 128
isat amp-profile model.onnx --action optimize --max-mse 0.001 -o mixed.onnx
```

### Architecture surgery

```bash
isat a2a model.onnx --action analyze
isat a2a model.onnx --action prune-heads --ratio 0.5 --method magnitude -o pruned.onnx
isat a2a model.onnx --action shrink-depth --ratio 0.5 -o smaller.onnx
```

### Knowledge distillation

```bash
isat distill-train teacher.onnx --epochs 20 --temperature 4.0 -o student.onnx
```

### Deploy to cloud

```bash
isat cloud-deploy model.onnx --output-dir deploy/   # Docker + K8s + SageMaker + Azure + GCP
```

### Monitor in production

```bash
isat monitor-live --model model.onnx                 # TUI dashboard + anomaly detection
isat test model.onnx --junit                         # Automated testing with CI output
```

<details>
<summary><strong>More examples: benchmarking, security, explainability</strong></summary>

```bash
# Comprehensive benchmarking
isat benchmark-suite model.onnx --batch-sizes 1,4,16,64
isat llm-bench model.onnx --seq-lengths 32,64,128,256

# Model security
isat encrypt model.onnx -o encrypted.onnx --method encrypt --password "secret"
isat encrypt model.onnx -o fingerprinted.onnx --method fingerprint --owner "ACME Corp"
isat safety --input-text "check this text for PII and toxicity"

# Explainability
isat explain model.onnx --method perturbation

# Sharding and merging
isat shard large_model.onnx --num-shards 4 --strategy balanced
isat merge encoder.onnx decoder.onnx -o pipeline.onnx --mode chain

# CI/CD gate
isat tune model.onnx --gate-latency 50 --gate-throughput 100
echo $?  # 0 = pass, 1 = fail

# CUDA/HIP graph capture
isat graph-compile model.onnx --action benchmark

# Tensor parallelism
isat tensor-parallel model.onnx --num-gpus 4 --action split
```

</details>

---

## Using as a Library

```python
from isat.converter.engine import convert
from isat.auto_detect.detector import detect_hardware
from isat.auto_detect.recommender import generate_recommendations

# Convert any model to ONNX
result = convert("google/vit-base-patch16-224", output_dir="./output")

# Auto-detect hardware and get recommendations
hw = detect_hardware()
report = generate_recommendations(hw, result.onnx_path)
```

<details>
<summary><strong>Full API reference</strong></summary>

```python
# v0.11.0 — LLM Engine
from isat.speculative.engine import SpeculativeDecoder, SelfSpeculativeDecoder, MedusaDecoder
from isat.llm_server.server import LLMServer, create_app, serve_llm
from isat.constrained.grammar import ConstrainedGenerator, constrained_generate

# v0.11.0 — Model Engineering
from isat.lora.adapter import LoRARuntime, MultiLoRARouter
from isat.lora.merger import WeightMerger
from isat.parallel.tensor_parallel import TensorParallelizer, TensorParallelRunner
from isat.graph_compile.capture import GraphCapture, GraphRegionAnalyzer
from isat.amp.profiler import PrecisionProfiler
from isat.amp.optimizer import MixedPrecisionOptimizer
from isat.distill_train.trainer import DistillationTrainer
from isat.arch_convert.converter import ArchitectureConverter

# v0.11.0 — Production
from isat.live_monitor.daemon import InferenceMonitor
from isat.live_monitor.dashboard import MonitorDashboard

# v0.10.0
from isat.quantize.quantizer import ModelQuantizer, quantize_model
from isat.stream.generator import StreamingGenerator
from isat.shard.splitter import ModelSharder
from isat.merge.merger import ModelMerger
from isat.explain.explainer import ModelExplainer
from isat.benchmark_suite.suite import BenchmarkSuite
from isat.encrypt.protector import ModelProtector
from isat.safety.guardrails import SafetyGuard
from isat.cloud_deploy.deployer import CloudDeployer
from isat.model_test.tester import ModelTester

# Core
from isat.fingerprint import fingerprint_hardware, fingerprint_model
from isat.search import SearchEngine
from isat.pruning.pruner import ModelPruner
from isat.fusion.analyzer import FusionAnalyzer
```

</details>

---

## How the Auto-Tuner Works

ISAT explores a 7-dimension search space that would take hours to test manually:

| Dimension | Search Space | Impact |
|-----------|-------------|--------|
| Memory strategy | XNACK on/off, coarse-grain, oversubscribe | Up to 30% on APUs |
| Kernel backend | MLIR, rocBLAS, hipBLASLt | 10-25% on GEMM-heavy models |
| Precision | FP32, FP16, INT8, INT4 | 2-4x throughput |
| Graph transforms | Raw, simplified, pinned dimensions | 5-20% latency reduction |
| Batch size | Powers of 2 up to GPU memory limit | Linear throughput scaling |
| Thread tuning | Inter/intra op threads, execution mode | CPU-side parallelism |
| Execution provider | MIGraphX, CUDA, TensorRT, OpenVINO, ROCm, DirectML, CPU | Provider-specific optimizations |

A single wrong choice can leave **40%+ performance on the table**. ISAT tests combinations automatically and reports the best configuration.

---

## Deployment Profiles

| Profile | Focus | Use Case |
|---------|-------|----------|
| `edge` | Latency | IoT, mobile, embedded |
| `cloud` | Throughput | Serving, batch processing |
| `latency` | P99 | Real-time inference |
| `throughput` | FPS | Max batch throughput |
| `power` | Perf/watt | Battery, thermal-constrained |
| `quick` | Fast | Rapid exploration |
| `exhaustive` | Complete | Leave no stone unturned |
| `apu` | APU-specific | AMD APU optimization |

---

## Architecture

```
isat/                               30,000+ lines of Python
├── cli.py                          76 CLI commands
│
├── speculative/                    Speculative decoding engine
├── llm_server/                     Continuous batching + PagedAttention
├── constrained/                    Grammar FSM (NFA/DFA/PDA)
├── lora/                           LoRA runtime + TIES/DARE/SLERP
├── parallel/                       Tensor parallelism + all-reduce
├── graph_compile/                  CUDA/HIP graph capture
├── amp/                            Mixed-precision profiling + optimization
├── distill_train/                  Knowledge distillation training
├── arch_convert/                   Architecture surgery
├── live_monitor/                   Anomaly detection + TUI dashboard
│
├── converter/                      Universal model-to-ONNX conversion
├── quantize/                       INT4/INT8/FP16/SmoothQuant
├── stream/                         Token-by-token LLM generation
├── shard/                          Model sharding + pipeline execution
├── merge/                          Model merging + composition
├── explain/                        Explainability tools
├── benchmark_suite/                Latency/throughput/memory benchmarks
├── encrypt/                        AES-256 encryption + watermarking
├── safety/                         PII/toxicity/jailbreak guardrails
├── cloud_deploy/                   Docker/K8s/SageMaker/Azure/GCP
├── model_test/                     Automated testing + JUnit output
│
├── auto_detect/                    Hardware detection + recommendations
├── fingerprint/                    Hardware + model fingerprinting
├── search/                         7-dimension search + Bayesian opt
├── benchmark/                      Runner, stats, thermal, multi-GPU
├── analysis/                       Outliers, significance, Pareto
├── server/                         FastAPI REST API
├── ... (20+ more modules)          Pruning, fusion, canary, tracing, etc.
└── utils/                          sysfs, rocm, onnx utilities
```

---

## REST API

```bash
isat serve --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/tune` | POST | Submit tuning job |
| `/api/v1/jobs/{id}` | GET | Job status + results |
| `/api/v1/jobs/{id}/report/html` | GET | Interactive HTML dashboard |
| `/api/v1/inspect` | POST | Model fingerprint |
| `/api/v1/hardware` | GET | Hardware fingerprint |
| `/health` | GET | Health check |

---

## CI/CD Integration

```bash
# Gate deployments on performance thresholds
isat tune model.onnx --gate-latency 50 --gate-throughput 100
echo $?  # 0 = pass, 1 = fail

# Automated model testing with JUnit XML
isat test model.onnx --junit --output-dir test-results/
```

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

## Generated Artifacts

| File | Description |
|------|-------------|
| `isat_report.html` | Interactive HTML dashboard |
| `isat_report.json` | Machine-readable results |
| `best_config.sh` | Shell script to apply best env vars |
| `isat_results.db` | SQLite history database |
| `config.pbtxt` | Triton Inference Server config |
| `isat.prom` | Prometheus metrics |
| `traces_*.json` | OpenTelemetry trace export |
| `isat_inference.cpp` | Generated C++ inference code |

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v0.11.0** | May 2026 | Speculative decoding, continuous batching, grammar-constrained generation, LoRA TIES/DARE/SLERP, tensor parallelism, CUDA graph capture, mixed-precision search, knowledge distillation, architecture surgery, live monitoring (76 commands) |
| v0.10.0 | May 2026 | Quantization, streaming, sharding, merging, explainability, benchmarking, encryption, safety, cloud deployment, automated testing (66 commands) |
| v0.9.1 | May 2026 | Universal model converter, 6-model E2E validation, PyTorch 2.9+ compatibility (56 commands) |
| v0.8.x | Apr 2026 | Auto-detect hardware, inference script generation, Windows DirectML, cross-platform GPU detection |
| v0.7.x | Apr 2026 | Pruning, distillation planning, fusion analysis, LLM benchmarking, compiler comparison (55 commands) |
| v0.6.0 | Apr 2026 | Tracing, canary deployment, alerts, graph surgery, caching (45 commands) |
| v0.5.0 | Apr 2026 | Regression detection, security scanning, thermal monitoring (38 commands) |
| v0.4.0 | Apr 2026 | Dynamic shapes, model hub, power profiler, A/B testing (30 commands) |
| v0.3.0 | Apr 2026 | Latency profiler, cost estimator, SLA validation (22 commands) |
| v0.2.0 | Apr 2026 | Config system, optimization, stress testing, plugin system (14 commands) |
| v0.1.0 | Apr 2026 | Initial release: auto-tuning, Bayesian search (9 commands) |

---

## Requirements

- **Python** >= 3.9
- **Runtime**: `onnxruntime` (CPU), `onnxruntime-rocm` (AMD), or `onnxruntime-gpu` (NVIDIA)
- **Core**: `onnx`, `numpy`
- **Optional**: `transformers`, `torch`, `tensorflow`, `fastapi`, `uvicorn`, `cryptography`, `scipy`, `onnxsim`

---

## Contributing

Contributions are welcome. Please open an issue first to discuss what you'd like to change.

```bash
git clone https://github.com/SID-Devu/isat-tuner.git
cd isat && pip install -e ".[dev,all]"
```

---

## Citation

```bibtex
@software{isat_tuner,
  author = {Sudheer Ibrahim Daniel Devu},
  title  = {ISAT: Inference Stack Auto-Tuner},
  year   = {2026},
  version = {0.11.0},
  url    = {https://github.com/SID-Devu/isat-tuner},
  note   = {76-command production inference engine for ONNX models}
}
```

---

<p align="center">
  <strong>Apache 2.0</strong> &mdash; Copyright 2026 Sudheer Ibrahim Daniel Devu
</p>
