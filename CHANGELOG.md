# Changelog

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
