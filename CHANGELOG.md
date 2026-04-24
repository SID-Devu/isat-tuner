# Changelog

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
