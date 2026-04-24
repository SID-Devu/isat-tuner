# Changelog

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
