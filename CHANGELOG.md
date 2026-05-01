# Changelog

## [0.11.0] - 2026-05-01

### Added -- 10 Cutting-Edge Modules (76 Total Commands)
- **`isat speculate`**: Speculative decoding engine with draft model + rejection sampling
  (Leviathan et al. 2023), self-speculation via early exit, Medusa-style multi-head prediction.
  Expected 2-4x decode speedup with zero quality loss.
- **`isat serve-llm`**: Continuous batching LLM server with PagedAttention-style KV cache pool,
  iteration-level scheduling, chunked prefill, prefix caching, SSE streaming, and
  OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints.
- **`isat constrain`**: Grammar-constrained generation via FSM-compiled token masking.
  JSON schema (pushdown automaton), regex (NFA->DFA via Thompson + subset construction),
  and GBNF grammar support. Per-token mask computation in <50us.
- **`isat lora`**: LoRA adapter runtime with hot-swap, multi-LoRA routing, and advanced weight
  merging: TIES-Merging (trim/elect/merge), DARE (drop-and-rescale), SLERP (spherical
  interpolation), Task Arithmetic, Model Soup. Adapter fusion for zero-overhead inference.
- **`isat tensor-parallel`**: True tensor parallelism — column-parallel QKV projections,
  row-parallel output projections, all-reduce synchronization. Auto-detect parallelizable
  layers from ONNX graph topology.
- **`isat graph-compile`**: CUDA/HIP graph capture and replay for 20-47% decode throughput
  improvement. Graph region analysis (static vs dynamic), warmup capture, input pointer
  swapping during replay.
- **`isat amp-profile`**: Automatic mixed precision — per-layer profiling at FP32/FP16/INT8/INT4,
  Pareto-optimal precision search via dynamic programming / greedy / beam search.
  Generate mixed-precision ONNX models.
- **`isat distill-train`**: Live knowledge distillation with real training loop through ORT.
  KL-divergence + cross-entropy loss, temperature scheduling, Adam optimizer in pure numpy.
  Auto-create smaller student via depth/width reduction.
- **`isat a2a`**: Architecture-to-architecture conversion — attention head pruning with
  importance scoring (magnitude/entropy/Taylor), width/depth shrinking, vocabulary pruning
  for domain-specific deployment.
- **`isat monitor-live`**: Real-time production inference monitor with anomaly detection
  (latency spikes, throughput drops, memory leaks, thermal throttling), auto-remediation,
  and ASCII TUI dashboard with sparkline latency trends.

### Changed
- CLI expanded from 66 to 76 commands
- Version bumped to 0.11.0

## [0.10.0] - 2026-05-01

### Added -- 10 New Modules (66 Total Commands)
- **`isat quantize`**: Advanced quantization with INT4 (MatMulNBits), INT8 (static QDQ),
  FP16, mixed-precision, and SmoothQuant. Includes sensitivity analysis and auto-method selection.
- **`isat stream`**: Streaming token-by-token LLM inference with KV cache management,
  nucleus sampling (top-k/top-p), and benchmark mode (TTFT, ITL, TPS metrics).
- **`isat shard`**: Split large models into N shards for multi-GPU / memory-constrained inference.
  Strategies: balanced, layer-boundary, auto. Includes ShardedRunner for pipeline-parallel execution.
- **`isat merge`**: Merge/compose multiple ONNX models via chaining (sequential) or
  parallel (concat/mean/max/sum aggregation). Graph-level merge with node renaming and opset reconciliation.
- **`isat explain`**: Model explainability via perturbation-based feature importance,
  gradient attribution (finite differences), sensitivity mapping, and layer activation extraction.
- **`isat benchmark-suite`**: Comprehensive benchmark suite: latency (P50/P95/P99 across batch sizes),
  sustained throughput, memory profiling (peak RSS), and scalability analysis.
- **`isat encrypt`**: Model IP protection: AES-256-GCM encryption, XOR obfuscation,
  LSB watermark fingerprinting in float32 mantissa, and expiry date embedding.
- **`isat safety`**: Regex-based safety guardrails: PII detection (email/phone/SSN/credit card),
  toxicity filtering, jailbreak pattern detection, confidence thresholds, and output format validation.
- **`isat cloud-deploy`**: One-command cloud deployment artifact generation: Dockerfile,
  Kubernetes manifests (Deployment + Service + HPA), SageMaker handler, Azure ML score.py,
  GCP Vertex AI config, and standalone FastAPI inference server.
- **`isat test`**: Automated model testing: determinism, numerical stability, edge cases,
  input validation, cross-provider comparison, latency consistency, memory leak detection,
  and golden file regression. JUnit XML output for CI integration.

### Changed
- CLI expanded from 56 to 66 commands
- Updated project metadata and keywords for new capabilities
- Version bumped to 0.10.0

## [0.9.1] - 2026-05-01

### Added
- **Universal Model Converter (`isat onnx`)**: Convert any model format (PyTorch, TensorFlow,
  JAX, HuggingFace, TFLite, SafeTensors) to ONNX with a single command — then auto-detect
  hardware and generate optimized inference scripts
- **Multi-modal export support**: CLIP, BLIP, DETR and other multi-input models export correctly
  via `inspect.signature`-based positional arg mapping with None-padding for gaps
- **Intelligent symbolic dimension defaults**: Generated inference scripts use smart defaults
  for ONNX symbolic dims (`num_channels`→3, `height`→224, `sequence_length`→16, etc.)
- **Robust HuggingFace ID detection**: Model names with dots (e.g. `facebook/opt-1.3b`,
  `meta-llama/Llama-3.1-8B`) are correctly identified as HuggingFace IDs

### Fixed
- **torch.onnx.export PyTorch 2.9+ compatibility**: Disabled dynamo by default (`dynamo=False`)
  and disabled KV cache (`use_cache=False`) for clean transformer export
- **Multi-input model export**: Models with multiple inputs (tokenizer + image) now correctly
  map to positional args via forward() signature inspection instead of failing with
  "got multiple values for argument" errors
- **Symbolic dim crash in generated scripts**: `_build_inputs()` no longer defaults all
  symbolic dimensions to 1 — uses a dimension-name-aware mapping for correct tensor shapes
- **HuggingFace ID format detection**: `detect_format()` no longer misclassifies model names
  containing dots (`.3b`, `.1B`, `.5-large`) as unknown file extensions

### Tested (6 HuggingFace models validated E2E)
| Model | Category | Params | ONNX Size | check_model | Inference | Script |
|-------|----------|--------|-----------|-------------|-----------|--------|
| google/vit-base-patch16-224 | Vision (ViT) | 86.6M | 330 MB | PASS | PASS | PASS |
| openai/clip-vit-base-patch32 | Multimodal (CLIP) | 151.3M | 578 MB | PASS | PASS | PASS |
| distilgpt2 | LLM | 81.9M | 313 MB | PASS | PASS | PASS |
| facebook/detr-resnet-50 | Object Detection | 41.6M | 159 MB | PASS | PASS | PASS |
| Salesforce/blip-image-captioning-base | Multimodal (BLIP) | 196.2M | 749 MB | PASS | PASS | PASS |
| facebook/opt-1.3b | 1.3B LLM | 1,315.7M | 5,412 MB | PASS | PASS | PASS |

## [0.9.0] - 2026-05-01

### Added
- **`isat onnx` command**: Universal model-to-ONNX converter with auto-tune integration
- **Format auto-detection**: Automatically identifies PyTorch, TensorFlow, JAX, HuggingFace,
  TFLite, and SafeTensors formats
- **Conversion backends**: optimum.exporters.onnx (primary), torch.onnx.export (fallback),
  tf2onnx, tflite2onnx, jax2onnx
- **ONNX validation and simplification**: Built-in onnx.checker + optional onnxsim
- **Auto-tune after conversion**: Automatically detects hardware and generates inference scripts

## [0.8.5] - 2026-04-27

### Changed
- Removed internal report references from public codebase for professionalism

## [0.8.4] - 2026-04-27

### Added
- **Windows MIGraphX EP via WinML CompileApi**: Native MIGraphX acceleration on Windows
  without WSL2 — uses the WinML AMD GPU EP AppX package
  (`MicrosoftCorporationII.WinML.AMD.GPU.EP`) with ORT 1.23+ CompileApi:
  `register_execution_provider_library()` + `add_provider_for_devices()`
- **WinML EP AppX Detection**: Auto-detects installed WinML AMD EP package via PowerShell
- **CompileApi Script Template**: Generates scripts using the correct CompileApi path
  (NOT `providers=` in `InferenceSession()` which silently falls back to CPU)
- **Recipe Priority**: On Windows + AMD GPU + WinML EP installed:
  MIGraphX (CompileApi) > DirectML > CPU

### Execution Paths (all supported)
```
Windows + WinML EP:  ORT CompileApi → WinML → MIGraphX EP → AMD GPU
Windows + DML:       ORT → DirectML EP → DirectX 12 → Any GPU
WSL2 + MIGraphX:     HIP → HSA → ROCDXG → /dev/dxg → GPU
Linux Baremetal:     HIP → HSA → amdgpu → GPU
```

## [0.8.3] - 2026-04-27

### Added
- **Windows DirectML Support**: `isat tune` now generates DirectML EP scripts on Windows
  for ANY GPU (AMD, NVIDIA, Intel, Qualcomm) via WinML → DirectX 12
- **Windows GPU Detection**: Auto-detects GPU on Windows via WMI/PowerShell
  (Get-CimInstance Win32_VideoController) — identifies vendor, VRAM, driver, type
- **Windows System Helpers**: RAM, swap, and CPU detection now work on Windows
  (was Linux/macOS only)
- **DirectML Script Template**: Full script with preflight checks, DML EP config,
  P50/P95 stats, copyright, zero-fallback for index-based ops
- **DirectML Recipes**: Two recipes per GPU — default DML + graph-opt-disabled fallback
  (for models needing dml_disable_graph_opt like CrossFormer, OpenVLA)
- **Execution Path Documentation**: Scripts show the full execution path:
  `WinML → ONNX Runtime → DirectML EP → DirectX 12 → GPU`

## [0.8.2] - 2026-04-27

### Fixed
- **GatherND/ScatterND crash with random inputs**: Models using index-based ops (e.g.
  CenterPoint) crashed with SIGSEGV when fed random test data. All 7 vendor templates
  now auto-detect this and retry with zero-filled inputs — matching production behavior
- Applied to all templates: AMD APU, AMD dGPU, NVIDIA, Intel, Apple, Qualcomm, CPU-only

### Tested
- CrossFormer (536 MB): 58.21 ms mean, 17.2 FPS — PASS
- CenterPoint (532 MB): 38.87 ms mean, 25.7 FPS — PASS (zero-fallback triggered)
- XTTS GPT2 (1,497 MB): 36.34 ms mean, 27.5 FPS — PASS
- EasyOCR Detector (79 MB): 21.53 ms mean, 46.4 FPS — PASS

## [0.8.1] - 2026-04-27

### Added
- **Auto-Generated Inference Scripts**: `isat tune MODEL.onnx` now generates a complete,
  runnable Python script (`isat_output/<model>_inference.py`) with:
  - Professional header (copyright, hardware fingerprint, generation timestamp)
  - Pre-flight system checks (GPU, XNACK, swap, ORT, model file)
  - Correct environment variables set BEFORE importing ORT
  - MIGraphX/TensorRT/OpenVINO/CoreML/QNN provider configuration
  - Random-input warmup + timed benchmark with P50/P95/FPS stats
  - EP verification (catches ORT silent CPU fallback)
- Scripts are vendor-specific: AMD APU, AMD dGPU, NVIDIA, Intel, Apple, Qualcomm, CPU-only
- Large models (>500MB) get template-depth HIP flags and swap sizing checks
- Tested on real hardware: MobileNetV2 on Strix Halo APU = 0.49ms / 2021 FPS

## [0.8.0] - 2026-04-27

### Added
- **Universal Hardware Auto-Detection**: `isat tune` now auto-detects your hardware
  vendor (AMD, NVIDIA, Intel, Apple, Qualcomm) and classifies it as iGPU, dGPU, APU,
  or SoC — works on any OS (Linux, macOS, Windows)
- **Inference Recommendations**: For each detected hardware, generates copy-paste-ready
  Python code, environment variables, install commands, and setup steps to run your model
- **AMD APU-specific guidance**: Tuned on 23 models across AMD Strix Halo APU —
  recommends HSA_XNACK, MIGraphX compile flags, swap sizing, kernel boot params,
  and subprocess isolation for large models
- **NVIDIA support**: TensorRT EP + CUDA EP recipes with engine caching, FP16 config
- **Intel support**: OpenVINO EP for iGPU/dGPU + CPU INT8 with AMX acceleration
- **Apple Silicon support**: CoreML EP with Neural Engine acceleration
- **Qualcomm support**: QNN EP with Hexagon HTP backend
- **Large model warnings**: Detects when a model won't fit in VRAM and recommends
  unified memory, swap sizing, and compilation flags
- `isat tune` now works **without a model** — shows hardware detection + general recommendations
- `isat tune MODEL.onnx --detect-only` — detect hardware + model-specific recommendations
  without running benchmarks
- `isat tune --json` — machine-readable JSON output of hardware and recommendations
- New module: `isat.auto_detect` (detector.py + recommender.py + script_gen.py)

## [0.7.8] - 2026-04-24

### Changed
- **PyPI description**: Removed "55-command" count and author name from package description
  and README intro — these belong only in the CLI banner, not in package metadata
- Cleaner, more professional PyPI page presentation

## [0.7.7] - 2026-04-24

### Fixed
- **download**: `isat download --list` no longer requires a positional `model_name` argument —
  works as a standalone flag to list all available models from the hub
- **tests**: Fixed flaky `test_canary_same_model` test that failed due to cold-start latency
  variance causing false rollback on identical models
- **tests**: Fixed `test_version_updated` assertion from exact match to `>=` so it doesn't
  break on every version bump

### Verified
- **Full 55-command test pass** on AMD Strix Halo APU (gfx1151) with MIGraphX EP
- **180/180 unit tests pass**

## [0.7.6] - 2026-04-02

### Fixed
- **hwinfo**: GPU name, GFX target, CU count now correctly parsed from `rocminfo` — was
  showing "unknown" on AMD APUs because the parser matched the CPU agent instead of GPU agent.
  Now uses `Device Type: GPU` as the section delimiter for reliable detection
- **hwinfo**: Added `XNACK enabled` status (reads `HSA_XNACK` env + `rocminfo` system flag)
  so you can see at a glance whether demand paging is active
- **leak-check**: Fixed false positive from MIGraphX JIT compilation memory — now runs warmup
  iterations before measuring so one-time compilation memory is excluded from the delta.
  Also uses trend analysis (first-third vs last-third) instead of raw delta for better accuracy
- **leak-check**: Fixed sampling never occurring when `iterations < sample_interval` — now
  auto-adjusts `sample_interval` to guarantee at least 5 sample points
- **sla**: Unprovided metrics now show `N/A` and `SKIP` instead of misleading `0.00ms PASS`.
  Fixed argparse defaults from `0` to `None` so zero-valued metrics can be provided explicitly
- **cost**: `--list-gpus` no longer requires `--latency` — works as standalone flag
- **compare**: Fixed `p-value=nan` and nonsensical t-stat caused by replicated mean values
  (zero variance). Now stores and uses actual per-iteration latencies. Added guards for
  near-zero variance edge cases in both `compare_configs` and `ABTest._welch_t_test`
- **Duplicate provider warning**: Eliminated `Duplicate provider CPUExecutionProvider` ORT
  warning across all 25+ modules by using `ort_providers()` helper that deduplicates the list

## [0.7.5] - 2026-04-02

### Fixed
- **CRITICAL: Removed `onnxruntime` from hard dependencies** — `pip install isat-tuner` no
  longer pulls the generic CPU-only `onnxruntime` from PyPI, which was **destroying** custom
  ORT builds (MIGraphX, ROCm, CUDA, TensorRT) already installed on the system
- **Auto-detect execution provider** — CLI now probes `ort.get_available_providers()` at
  startup and picks the best GPU EP automatically (MIGraphX > TensorRT > CUDA > ROCm > CPU)
- **Banner shows detected provider and ORT version** so you immediately see what backend ISAT
  will use
- If ORT is not installed at all, the banner prints clear install instructions for each variant

## [0.7.4] - 2026-04-02

### Added
- **`python3 -m isat` fallback**: Package now supports `python3 -m isat` so the CLI
  always works even when `~/.local/bin` is not on PATH
- **Smart PATH detection**: When running `isat` or `python3 -m isat` with no arguments,
  the CLI detects if `~/.local/bin/isat` exists but is not on PATH, and prints exact
  fix instructions — so users never get a silent "command not found" after install

## [0.7.0] - 2026-04-02

### Added
- **Model pruning** (`isat prune`): Magnitude, percentage, and global pruning strategies. Zero out
  low-magnitude weights per-layer or globally. Report per-layer sparsity, `--analyze-only` mode
- **Knowledge distillation** (`isat distill`): Analyze teacher model and generate student configs
  (tiny/small/medium) with recommended architecture changes, temperature, alpha, training tips
- **Operator fusion analyzer** (`isat fusion`): Compare raw vs ORT-optimized graph. Identify fused
  patterns (Conv+BN+Relu, MatMul+Add, LayerNorm, Attention), missed fusion opportunities, unfused
  compute-heavy ops
- **Attention profiler** (`isat attention`): Per-head weight norm, entropy, sparsity analysis for
  transformer models. Identifies prunable heads (bottom 20% importance), estimates speedup
- **LLM token benchmarker** (`isat llm-bench`): Tokens/second, time-to-first-token (TTFT),
  inter-token latency (ITL) with P95, prefill vs decode throughput across sequence lengths
- **Compiler comparison** (`isat compiler-compare`): Benchmark same model across all available
  execution providers (CPU, MIGraphX, CUDA, TRT, OpenVINO, DML, QNN, CoreML). Compare latency,
  numerical accuracy vs CPU baseline, speedup. No other open-source tool does this
- **Inference replay** (`isat replay`): Record production inference requests (inputs + outputs +
  metadata) to disk. Replay against new model versions for regression testing with output matching
- **Output drift monitor** (`isat drift`): Detect confidence distribution drift between baseline and
  current model outputs using KS test, entropy tracking, confidence monitoring
- **Weight sharing detector** (`isat weight-sharing`): Find identical and near-identical (cosine
  similarity) weight tensors across layers. Report memory savings from detected sharing
- **C++ code generator** (`isat codegen`): Generate standalone C++ inference code + CMakeLists.txt
  from ONNX model. For edge deployment where Python is unavailable
- CLI expanded to **55 subcommands** (10 new: prune, distill, fusion, attention, llm-bench,
  compiler-compare, replay, drift, weight-sharing, codegen)
- 23 new tests (180 total, all passing)

## [0.6.0] - 2026-04-02

### Added
- **Request tracing** (`isat trace`): OpenTelemetry-compatible span tracing through the full
  inference lifecycle (preprocess -> inference -> postprocess). Exports OTLP JSON for Jaeger/Zipkin/Datadog
- **Canary deployment** (`isat canary`): Phased traffic splitting between baseline and candidate models
  with automatic rollback on error rate or latency regression. Configurable phases (5%, 10%, 25%, 50%, 75%, 100%)
- **Alert rules engine** (`isat alerts`): Define rules like "if P99 > 500ms for 3 checks, CRITICAL".
  8 builtin rules covering latency, error rate, GPU temp, memory, throughput, queue depth. Export/import JSON rules
- **ONNX graph surgery** (`isat surgery`): Programmatic model modification -- remove op types (Identity, Dropout),
  rename inputs/outputs, remove unused initializers, extract subgraphs, change opset. Prepare models for deployment
  without retraining
- **Inference caching** (`isat` API): LRU in-memory + disk-persistent cache keyed by input tensor SHA256.
  TTL expiration, eviction tracking, hit/miss stats. Avoid redundant GPU computation for repeated inputs
- **Input validation / guard** (`isat guard`): Enforce input tensor shapes, dtypes, value ranges.
  Detects NaN/Inf, missing inputs, excessive sizes. Extracts schema directly from ONNX model definition
- **Model ensemble** (`isat ensemble`): Run N models, aggregate with averaging, voting, or max-confidence.
  Reports per-member latency, error status, and inter-model agreement percentage
- **GPU memory fragmentation analyzer** (`isat gpu-frag`): Monitor VRAM/GTT allocation patterns during
  inference, compute fragmentation index (0-1 scale), classify allocation patterns, recommend mitigations
- CLI expanded to **45 subcommands** (7 new: trace, canary, alerts, surgery, guard, ensemble, gpu-frag)
- 32 new unit tests (157 total, all passing)

## [0.5.0] - 2026-04-24

### Fixed
- **Version string**: BANNER and `--version` now dynamically read `__version__` instead of hardcoded `0.1.0`
- **SLA validator**: Unprovided metrics are now skipped (shown as PASS) instead of failing with `actual=0.00`
- **GPU detection**: `hwinfo` now parses Marketing Name and fallback gfx target from GPU agent name
- **Profile command**: Added `--warmup` flag (default 3) with actual warmup iterations before steady-state timing

### Added
- **Performance regression detector** (`isat regression`): Track latency across model versions against baselines
  with Welch's t-test, threshold %, history tracking, and `--set-baseline` for CI pipelines
- **ONNX security/compliance scanner** (`isat scan`): Check model file size, external data integrity,
  opset compliance, unsafe/deprecated operators, NaN/Inf in initializers, metadata, graph pattern analysis.
  Outputs a 0-100 compliance score
- **Operator compatibility matrix** (`isat compat-matrix`): Shows per-provider op support (CPU, MIGraphX,
  CUDA, TensorRT, OpenVINO, QNN) with FP16/INT8 capability flags
- **Thermal throttle detector** (`isat thermal`): Monitors GPU temp + clock during inference, detects
  throttling events, estimates performance impact percentage
- **Quantization sensitivity analyzer** (`isat quant-sensitivity`): Per-layer MSE analysis for FP16/INT8,
  identifies sensitive layers, generates mixed-precision recipe
- **Multi-model pipeline optimizer** (`isat pipeline`): Profile chains of ONNX models, find bottleneck stage,
  compute per-stage % of total latency, generate optimization suggestions
- **Hardware recommendation engine** (`isat recommend`): Given model characteristics, rank 8 GPU targets
  (AMD APU, MI250X, MI300X, NVIDIA T4/A10G/A100/H100, Intel CPU) by estimated latency, cost, and memory fit.
  Supports `--max-latency`, `--max-cost`, `--prefer-amd` filters
- **Model version registry** (`isat registry`): SQLite-backed model version tracking with register, list,
  promote (dev->staging->production), diff versions, and SHA256 verification
- CLI expanded to **38 subcommands** (8 new: scan, regression, compat-matrix, thermal, quant-sensitivity,
  pipeline, recommend, registry)

## [0.4.0] - 2026-04-02

### Added
- Dynamic shape handler (`isat shapes`): benchmark across input shapes/sequence lengths
- Model hub integration (`isat download`): download from ONNX Model Zoo / HuggingFace with caching
- Power efficiency profiler (`isat power`): measure perf/watt, energy per inference
- Memory planner (`isat memory`): estimate peak memory, predict OOM, recommend batch sizes
- A/B testing framework (`isat abtest`): statistically rigorous comparison of two models
- Graph visualizer (`isat visualize`): DOT, ASCII, and histogram representations
- Environment snapshot (`isat snapshot`): capture full system state for reproducibility
- Adaptive batch scheduler (`isat batch`): find optimal batch size across latency/throughput tradeoff
- CLI expanded to 30 subcommands

## [0.3.0] - 2026-04-02

### Added
- **Latency profiler**: Decompose inference latency into load/compile/first-inference/steady-state phases
- **Dependency scanner**: Pre-flight checks for Python, ORT, ROCm, CUDA, drivers, env vars, model opset
- **Model diff**: Structural comparison of two ONNX models (nodes, params, ops, inputs/outputs)
- **Cloud cost estimator**: Cost-per-inference, monthly projections at various QPS, optimization ROI calculator
  with pricing for 10+ GPU types (A100, H100, T4, L4, MI300X, MI250, etc.)
- **SLA validator**: 5 built-in SLA templates (realtime, batch, edge, llm, mobile) with per-metric pass/fail
- **Notification system**: Webhook (generic HTTP), Slack (incoming webhook), console notifiers for job events
- **Cache manager**: Track, clean, and warm MIGraphX/ORT compilation caches
- **Health checker**: Pre-flight GPU temp, memory, disk, process, and clock verification
- **Warmup analyzer**: Automatically find optimal warmup iterations by detecting convergence and JIT boundaries
- **Provider migration tool**: Generate step-by-step migration plans between ROCm/CUDA/TensorRT/CPU with env mapping
- CLI expanded to 22 subcommands: added doctor, profile, diff, cost, sla, warmup, cache, migrate

## [0.2.0] - 2026-04-02

### Added
- YAML/JSON config file system (`isat init`, `isat tune --config`)
- Model optimization pipeline (simplify, FP16, INT8 QDQ, ORT export)
- Concurrent stress testing (sustained, burst, ramp patterns)
- Memory leak detection
- Plugin system with lifecycle hooks
- Model zoo with 12 pre-tuned configs
- Structured JSON logging
- Retry logic with exponential backoff
- GitHub issue/PR templates, security policy, code of conduct
- Repo renamed to isat-tuner

## [0.1.0] - 2026-04-02

### Added
- Initial release
- Hardware fingerprinting (GPU detection, memory topology, XNACK capability)
- Model fingerprinting (ONNX deep analysis, op counting, classification)
- Multi-dimensional search engine:
  - Memory strategy (XNACK, coarse-grain, oversubscription)
  - Kernel backend (MLIR, rocBLAS, hipBLASlt, parallel compile)
  - Precision (FP32, FP16, INT8 QDQ)
  - Graph transforms (onnxsim, shape pinning, ORT opt levels)
  - Batch size auto-exploration
  - Thread/execution mode tuning
  - Multi-provider support (MIGraphX, CUDA, TensorRT, OpenVINO, ROCm, CPU)
- Bayesian optimization search (GP + TPE with early stopping)
- Thermal-aware benchmarking with enforced cooldowns
- Statistical analysis:
  - Outlier detection (MAD and IQR methods)
  - Significance testing (Welch's t-test)
  - Pareto frontier analysis (multi-objective)
  - Performance regression detection
- Report generation (JSON, HTML dashboard, console, env script)
- SQLite results database with query API
- REST API server (FastAPI) with job management
- Triton Inference Server config generation
- Prometheus metrics export
- CI/CD integration (GitHub Actions, performance gates)
- Multi-GPU device discovery and workload distribution
- 8 deployment profiles (edge, cloud, latency, throughput, power, quick, exhaustive, apu)
- Docker and docker-compose support
- CLI with 9 subcommands (tune, inspect, hwinfo, history, export, compare, serve, triton, profiles)
