"""Cloud deployment artifact generator for ONNX models.

Generates production-ready Dockerfiles, Kubernetes manifests,
and cloud-provider configs (AWS SageMaker, Azure ML, GCP Vertex AI)
along with a standalone FastAPI inference server.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("isat.cloud_deploy")

# ---------------------------------------------------------------------------
# DeploymentBundle
# ---------------------------------------------------------------------------

@dataclass
class DeploymentBundle:
    """Collects paths to every generated deployment artifact."""

    output_dir: str
    dockerfile: Optional[str] = None
    kubernetes_manifests: List[str] = field(default_factory=list)
    sagemaker_artifacts: Optional[str] = None
    azure_ml_artifacts: Optional[str] = None
    gcp_vertex_artifacts: Optional[str] = None
    inference_handler: Optional[str] = None
    cost_estimate: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CloudDeployer
# ---------------------------------------------------------------------------

class CloudDeployer:
    """Generate deployment artifacts for an ONNX model across providers.

    Parameters
    ----------
    model_path : str | Path
        Path to the ``.onnx`` model file.
    config : dict, optional
        Overrides for default deployment settings (image names, ports, etc.).
    """

    DEFAULT_PORT = 8080
    APP_DIR = "/app"

    def __init__(self, model_path: str | Path, config: Optional[Dict[str, Any]] = None):
        self.model_path = Path(model_path)
        if not self.model_path.suffix == ".onnx":
            log.warning("model_path does not end with .onnx – proceeding anyway")
        self.model_name = self.model_path.stem
        self.config = config or {}
        self.port = int(self.config.get("port", self.DEFAULT_PORT))

    # -- helpers -------------------------------------------------------------

    def _ensure_dir(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write(self, path: Path, content: str) -> str:
        path.write_text(textwrap.dedent(content).lstrip("\n"))
        log.info("wrote %s", path)
        return str(path)

    # -----------------------------------------------------------------------
    # Dockerfile
    # -----------------------------------------------------------------------

    def generate_dockerfile(
        self,
        output_dir: str | Path,
        base_image: str = "python:3.11-slim",
        provider: str = "cpu",
    ) -> str:
        out = self._ensure_dir(Path(output_dir))

        ort_package = {
            "cpu": "onnxruntime",
            "gpu": "onnxruntime-gpu",
            "tensorrt": "onnxruntime-gpu",
        }.get(provider, "onnxruntime")

        # GPU base images when requested
        if provider in ("gpu", "tensorrt") and "python" in base_image:
            base_image = "nvidia/cuda:12.2.0-runtime-ubuntu22.04"

        gpu_env_block = ""
        if provider in ("gpu", "tensorrt"):
            gpu_env_block = (
                "ENV NVIDIA_VISIBLE_DEVICES=all\n"
                "ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility\n"
            )

        dockerfile = f"""\
            # -----------------------------------------------------------------
            # Production ONNX inference image
            # Provider : {provider}
            # Model    : {self.model_name}
            # -----------------------------------------------------------------
            FROM {base_image}

            {gpu_env_block}
            # System deps needed by ORT / numpy
            RUN apt-get update && \\
                apt-get install -y --no-install-recommends \\
                    libgomp1 curl && \\
                rm -rf /var/lib/apt/lists/*

            WORKDIR {self.APP_DIR}

            # Python deps – pinned for reproducibility
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt

            # Copy model & inference server
            COPY {self.model_name}.onnx model.onnx
            COPY inference_server.py .

            EXPOSE {self.port}

            # Health-check: hit /health every 30 s
            HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
                CMD curl -f http://localhost:{self.port}/health || exit 1

            # Graceful shutdown via SIGTERM; uvicorn handles it natively
            STOPSIGNAL SIGTERM

            ENTRYPOINT ["python", "inference_server.py"]
        """

        # Also emit a matching requirements.txt
        reqs = f"""\
            {ort_package}
            fastapi>=0.110
            uvicorn[standard]>=0.29
            numpy>=1.24
            pydantic>=2.0
        """
        self._write(out / "requirements.txt", reqs)
        return self._write(out / "Dockerfile", dockerfile)

    # -----------------------------------------------------------------------
    # Kubernetes manifests
    # -----------------------------------------------------------------------

    def generate_kubernetes(
        self,
        output_dir: str | Path,
        replicas: int = 2,
        cpu_limit: str = "2",
        memory_limit: str = "4Gi",
        gpu: bool = False,
    ) -> List[str]:
        out = self._ensure_dir(Path(output_dir) / "k8s")
        image = self.config.get("image", f"{self.model_name}-inference:latest")
        ns = self.config.get("namespace", "ml-serving")
        paths: List[str] = []

        # --- ConfigMap -------------------------------------------------------
        configmap = f"""\
            # ConfigMap: model metadata & runtime tunables
            apiVersion: v1
            kind: ConfigMap
            metadata:
              name: {self.model_name}-config
              namespace: {ns}
              labels:
                app: {self.model_name}
            data:
              MODEL_NAME: "{self.model_name}"
              MODEL_PATH: "/app/model.onnx"
              LOG_LEVEL: "info"
              WORKERS: "4"
              PORT: "{self.port}"
        """
        paths.append(self._write(out / "configmap.yaml", configmap))

        # --- Deployment ------------------------------------------------------
        gpu_resources = ""
        if gpu:
            gpu_resources = '            nvidia.com/gpu: "1"'

        gpu_tolerations = ""
        if gpu:
            gpu_tolerations = textwrap.dedent("""\
                  tolerations:
                    - key: nvidia.com/gpu
                      operator: Exists
                      effect: NoSchedule
            """)

        deployment = f"""\
            apiVersion: apps/v1
            kind: Deployment
            metadata:
              name: {self.model_name}
              namespace: {ns}
              labels:
                app: {self.model_name}
            spec:
              replicas: {replicas}
              selector:
                matchLabels:
                  app: {self.model_name}
              strategy:
                rollingUpdate:
                  maxSurge: 1
                  maxUnavailable: 0
                type: RollingUpdate
              template:
                metadata:
                  labels:
                    app: {self.model_name}
                  annotations:
                    prometheus.io/scrape: "true"
                    prometheus.io/port: "{self.port}"
                    prometheus.io/path: "/metrics"
                spec:
                  terminationGracePeriodSeconds: 30
            {gpu_tolerations}\
                  containers:
                    - name: inference
                      image: {image}
                      ports:
                        - containerPort: {self.port}
                          name: http
                      envFrom:
                        - configMapRef:
                            name: {self.model_name}-config
                      resources:
                        requests:
                          cpu: "500m"
                          memory: "1Gi"
                        limits:
                          cpu: "{cpu_limit}"
                          memory: "{memory_limit}"
            {gpu_resources}
                      # Liveness: restart if the process is stuck
                      livenessProbe:
                        httpGet:
                          path: /health
                          port: {self.port}
                        initialDelaySeconds: 15
                        periodSeconds: 15
                        failureThreshold: 3
                      # Readiness: stop routing until the model is loaded
                      readinessProbe:
                        httpGet:
                          path: /health
                          port: {self.port}
                        initialDelaySeconds: 10
                        periodSeconds: 5
                        failureThreshold: 2
                      # Startup: generous budget for large-model loads
                      startupProbe:
                        httpGet:
                          path: /health
                          port: {self.port}
                        periodSeconds: 10
                        failureThreshold: 30
        """
        paths.append(self._write(out / "deployment.yaml", deployment))

        # --- Service ---------------------------------------------------------
        service = f"""\
            apiVersion: v1
            kind: Service
            metadata:
              name: {self.model_name}
              namespace: {ns}
              labels:
                app: {self.model_name}
            spec:
              type: ClusterIP
              selector:
                app: {self.model_name}
              ports:
                - port: 80
                  targetPort: {self.port}
                  protocol: TCP
                  name: http
        """
        paths.append(self._write(out / "service.yaml", service))

        # --- HPA (CPU + custom latency metric) -------------------------------
        hpa = f"""\
            # HPA: auto-scale on CPU utilisation and p99 inference latency.
            # The latency metric requires a Prometheus adapter; remove the
            # "pods" block if you only want CPU-based scaling.
            apiVersion: autoscaling/v2
            kind: HorizontalPodAutoscaler
            metadata:
              name: {self.model_name}
              namespace: {ns}
            spec:
              scaleTargetRef:
                apiVersion: apps/v1
                kind: Deployment
                name: {self.model_name}
              minReplicas: {max(1, replicas)}
              maxReplicas: {replicas * 5}
              behavior:
                scaleDown:
                  stabilizationWindowSeconds: 300
                  policies:
                    - type: Percent
                      value: 25
                      periodSeconds: 60
                scaleUp:
                  stabilizationWindowSeconds: 30
                  policies:
                    - type: Pods
                      value: 2
                      periodSeconds: 60
              metrics:
                - type: Resource
                  resource:
                    name: cpu
                    target:
                      type: Utilization
                      averageUtilization: 70
                - type: Pods
                  pods:
                    metric:
                      name: inference_latency_p99_ms
                    target:
                      type: AverageValue
                      averageValue: "100"
        """
        paths.append(self._write(out / "hpa.yaml", hpa))

        return paths

    # -----------------------------------------------------------------------
    # AWS SageMaker
    # -----------------------------------------------------------------------

    def generate_sagemaker(
        self,
        output_dir: str | Path,
        instance_type: str = "ml.m5.large",
    ) -> str:
        out = self._ensure_dir(Path(output_dir) / "sagemaker")

        # inference.py – SageMaker's custom-script contract
        inference_py = f"""\
            \"\"\"SageMaker inference handler for {self.model_name}.

            SageMaker invokes four hooks in order:
              model_fn  -> load the ONNX model once
              input_fn  -> deserialise the request body
              predict_fn -> run inference
              output_fn -> serialise the response

            Logging goes to CloudWatch automatically.
            \"\"\"
            import json
            import logging
            import os
            import signal
            import sys

            import numpy as np
            import onnxruntime as ort

            log = logging.getLogger(__name__)
            logging.basicConfig(
                level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )

            _session: ort.InferenceSession | None = None


            def _graceful_shutdown(signum, frame):
                log.info("received signal %s – shutting down", signum)
                sys.exit(0)


            signal.signal(signal.SIGTERM, _graceful_shutdown)


            def model_fn(model_dir: str) -> ort.InferenceSession:
                \"\"\"Load the ONNX model; called once at container start.\"\"\"
                global _session
                model_path = os.path.join(model_dir, "model.onnx")
                log.info("loading model from %s", model_path)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                opts.intra_op_num_threads = int(
                    os.environ.get("OMP_NUM_THREADS", "0")
                ) or os.cpu_count()
                _session = ort.InferenceSession(model_path, opts)
                log.info(
                    "model loaded – inputs: %s",
                    [i.name for i in _session.get_inputs()],
                )
                return _session


            def input_fn(request_body: str, content_type: str = "application/json"):
                \"\"\"Deserialise incoming payload.\"\"\"
                if content_type != "application/json":
                    raise ValueError(f"unsupported content type: {{content_type}}")
                payload = json.loads(request_body)
                return {{
                    k: np.array(v, dtype=np.float32) for k, v in payload.items()
                }}


            def predict_fn(input_data: dict, model: ort.InferenceSession):
                \"\"\"Run the ONNX model.\"\"\"
                outputs = model.run(None, input_data)
                return outputs


            def output_fn(prediction, accept: str = "application/json") -> str:
                \"\"\"Serialise model outputs.\"\"\"
                result = [
                    p.tolist() if hasattr(p, "tolist") else p for p in prediction
                ]
                return json.dumps({{"outputs": result}})
        """
        self._write(out / "inference.py", inference_py)

        reqs = """\
            onnxruntime
            numpy>=1.24
        """
        self._write(out / "requirements.txt", reqs)

        # deploy_config.json – consumed by CI or the SageMaker SDK
        deploy_cfg = {
            "model_name": self.model_name,
            "instance_type": instance_type,
            "initial_instance_count": 1,
            "framework": "onnxruntime",
            "entry_point": "inference.py",
            "source_dir": str(out),
            "model_data_description": (
                "Pack model.onnx into model.tar.gz at the repo root, then "
                "upload to S3.  SageMaker extracts it into /opt/ml/model."
            ),
        }
        self._write(
            out / "deploy_config.json",
            json.dumps(deploy_cfg, indent=2) + "\n",
        )

        return str(out)

    # -----------------------------------------------------------------------
    # Azure ML
    # -----------------------------------------------------------------------

    def generate_azure_ml(
        self,
        output_dir: str | Path,
        vm_size: str = "Standard_DS3_v2",
    ) -> str:
        out = self._ensure_dir(Path(output_dir) / "azure_ml")

        score_py = f"""\
            \"\"\"Azure ML scoring script for {self.model_name}.

            Azure ML calls `init()` once when the container starts and
            `run(raw_data)` for every inference request.
            \"\"\"
            import json
            import logging
            import os
            import signal
            import sys

            import numpy as np
            import onnxruntime as ort

            log = logging.getLogger(__name__)
            logging.basicConfig(
                level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )

            _session: ort.InferenceSession | None = None


            def _graceful_shutdown(signum, frame):
                log.info("received signal %s – shutting down", signum)
                sys.exit(0)


            signal.signal(signal.SIGTERM, _graceful_shutdown)


            def init():
                \"\"\"Load model – called once by Azure ML.\"\"\"
                global _session
                model_dir = os.getenv(
                    "AZUREML_MODEL_DIR", os.path.join(".", "model")
                )
                model_path = os.path.join(model_dir, "model.onnx")
                log.info("loading model from %s", model_path)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                _session = ort.InferenceSession(model_path, opts)
                log.info("model loaded successfully")


            def run(raw_data: str) -> str:
                \"\"\"Score a single request.\"\"\"
                try:
                    payload = json.loads(raw_data)
                    feeds = {{
                        k: np.array(v, dtype=np.float32)
                        for k, v in payload.items()
                    }}
                    outputs = _session.run(None, feeds)
                    result = [
                        o.tolist() if hasattr(o, "tolist") else o
                        for o in outputs
                    ]
                    return json.dumps({{"outputs": result}})
                except Exception as exc:
                    log.exception("inference failed")
                    return json.dumps({{"error": str(exc)}})
        """
        self._write(out / "score.py", score_py)

        env_yml = f"""\
            # Conda environment for Azure ML managed endpoint
            name: {self.model_name}-env
            channels:
              - defaults
              - conda-forge
            dependencies:
              - python=3.11
              - pip
              - pip:
                  - onnxruntime
                  - numpy>=1.24
                  - azureml-defaults
        """
        self._write(out / "environment.yml", env_yml)

        deployment_yml = f"""\
            # Azure ML managed online-endpoint deployment config
            # Apply with:  az ml online-deployment create -f deployment.yml
            $schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
            name: {self.model_name}-deployment
            endpoint_name: {self.model_name}-endpoint
            model:
              path: ./model
            code_configuration:
              code: .
              scoring_script: score.py
            environment:
              conda_file: environment.yml
              image: mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
            instance_type: {vm_size}
            instance_count: 1
            # Liveness / readiness probes
            liveness_probe:
              initial_delay: 30
              period: 15
              timeout: 5
              failure_threshold: 3
            readiness_probe:
              initial_delay: 20
              period: 10
              timeout: 5
              failure_threshold: 3
            request_settings:
              request_timeout_ms: 60000
              max_concurrent_requests_per_instance: 10
        """
        self._write(out / "deployment.yml", deployment_yml)

        return str(out)

    # -----------------------------------------------------------------------
    # GCP Vertex AI
    # -----------------------------------------------------------------------

    def generate_gcp_vertex(
        self,
        output_dir: str | Path,
        machine_type: str = "n1-standard-4",
    ) -> str:
        out = self._ensure_dir(Path(output_dir) / "gcp_vertex")

        predict_py = f"""\
            \"\"\"Vertex AI custom prediction handler for {self.model_name}.

            Vertex AI calls `PredictionHandler.predict` for each request.
            The container must serve HTTP on ${{AIP_HTTP_PORT}} (default 8080).
            \"\"\"
            import json
            import logging
            import os
            import signal
            import sys

            import numpy as np
            import onnxruntime as ort
            from fastapi import FastAPI, Request
            from fastapi.responses import JSONResponse
            import uvicorn

            log = logging.getLogger(__name__)
            logging.basicConfig(
                level=os.environ.get("LOG_LEVEL", "INFO").upper(),
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )

            app = FastAPI(title="{self.model_name} – Vertex AI")
            _session: ort.InferenceSession | None = None


            def _graceful_shutdown(signum, frame):
                log.info("received signal %s – shutting down", signum)
                sys.exit(0)


            signal.signal(signal.SIGTERM, _graceful_shutdown)


            @app.on_event("startup")
            def load_model():
                global _session
                model_path = os.getenv(
                    "AIP_STORAGE_URI", "/app/model.onnx"
                )
                # Vertex copies model artifacts into AIP_STORAGE_URI
                if os.path.isdir(model_path):
                    model_path = os.path.join(model_path, "model.onnx")
                log.info("loading model from %s", model_path)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                _session = ort.InferenceSession(model_path, opts)
                log.info("model loaded successfully")


            @app.get("/health")
            @app.get(os.getenv("AIP_HEALTH_ROUTE", "/health"))
            def health():
                return {{"status": "healthy", "model_loaded": _session is not None}}


            @app.post("/predict")
            @app.post(os.getenv("AIP_PREDICT_ROUTE", "/predict"))
            async def predict(request: Request):
                body = await request.json()
                instances = body.get("instances", [body])
                results = []
                for instance in instances:
                    feeds = {{
                        k: np.array(v, dtype=np.float32)
                        for k, v in instance.items()
                    }}
                    outputs = _session.run(None, feeds)
                    results.append(
                        [o.tolist() if hasattr(o, "tolist") else o for o in outputs]
                    )
                return JSONResponse({{"predictions": results}})


            if __name__ == "__main__":
                port = int(os.getenv("AIP_HTTP_PORT", "8080"))
                uvicorn.run(app, host="0.0.0.0", port=port)
        """
        self._write(out / "predict.py", predict_py)

        dockerfile = f"""\
            # Vertex AI custom container
            FROM python:3.11-slim

            RUN apt-get update && \\
                apt-get install -y --no-install-recommends curl && \\
                rm -rf /var/lib/apt/lists/*

            WORKDIR /app
            COPY requirements.txt .
            RUN pip install --no-cache-dir -r requirements.txt

            COPY {self.model_name}.onnx model.onnx
            COPY predict.py .

            # Vertex AI injects AIP_HTTP_PORT; default to 8080
            ENV AIP_HTTP_PORT=8080
            EXPOSE 8080

            HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
                CMD curl -f http://localhost:8080/health || exit 1

            STOPSIGNAL SIGTERM
            ENTRYPOINT ["python", "predict.py"]
        """
        self._write(out / "Dockerfile", dockerfile)

        reqs = """\
            onnxruntime
            fastapi>=0.110
            uvicorn[standard]>=0.29
            numpy>=1.24
        """
        self._write(out / "requirements.txt", reqs)

        config = {
            "model_name": self.model_name,
            "machine_type": machine_type,
            "accelerator": None,
            "container_uri": f"gcr.io/$PROJECT_ID/{self.model_name}:latest",
            "health_route": "/health",
            "predict_route": "/predict",
            "min_replica_count": 1,
            "max_replica_count": 5,
        }
        self._write(
            out / "vertex_config.json",
            json.dumps(config, indent=2) + "\n",
        )

        return str(out)

    # -----------------------------------------------------------------------
    # Standalone FastAPI inference handler
    # -----------------------------------------------------------------------

    def generate_inference_handler(self, output_dir: str | Path) -> str:
        out = self._ensure_dir(Path(output_dir))

        server_py = f"""\
            \"\"\"Standalone FastAPI inference server for {self.model_name}.

            Run:
                python inference_server.py                  # defaults
                MODEL_PATH=/models/v2.onnx python inference_server.py
                LOG_LEVEL=debug WORKERS=4 python inference_server.py

            Environment variables
            ---------------------
            MODEL_PATH   Path to the .onnx file        (default: model.onnx)
            PORT         HTTP port                       (default: {self.port})
            WORKERS      Uvicorn worker count            (default: 1)
            LOG_LEVEL    Python log level                 (default: info)
            \"\"\"
            import json
            import logging
            import os
            import signal
            import sys
            import time
            from contextlib import asynccontextmanager

            import numpy as np
            import onnxruntime as ort
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.responses import JSONResponse
            from pydantic import BaseModel
            import uvicorn

            # -- logging ----------------------------------------------------------

            LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
            logging.basicConfig(
                level=LOG_LEVEL,
                format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            )
            log = logging.getLogger("{self.model_name}")

            # -- globals ----------------------------------------------------------

            MODEL_PATH = os.environ.get("MODEL_PATH", "model.onnx")
            PORT = int(os.environ.get("PORT", "{self.port}"))
            WORKERS = int(os.environ.get("WORKERS", "1"))
            _session: ort.InferenceSession | None = None
            _model_meta: dict = {{}}


            # -- graceful shutdown ------------------------------------------------

            def _handle_signal(signum, frame):
                log.info("signal %s received – initiating shutdown", signum)
                sys.exit(0)

            signal.signal(signal.SIGTERM, _handle_signal)
            signal.signal(signal.SIGINT, _handle_signal)


            # -- lifespan --------------------------------------------------------

            @asynccontextmanager
            async def lifespan(application: FastAPI):
                global _session, _model_meta
                log.info("loading model from %s", MODEL_PATH)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                opts.intra_op_num_threads = os.cpu_count() or 1
                _session = ort.InferenceSession(MODEL_PATH, opts)
                _model_meta = {{
                    "inputs": [
                        {{"name": i.name, "shape": i.shape, "type": i.type}}
                        for i in _session.get_inputs()
                    ],
                    "outputs": [
                        {{"name": o.name, "shape": o.shape, "type": o.type}}
                        for o in _session.get_outputs()
                    ],
                }}
                log.info(
                    "model ready – %d input(s), %d output(s)",
                    len(_model_meta["inputs"]),
                    len(_model_meta["outputs"]),
                )
                yield
                log.info("shutting down")


            app = FastAPI(
                title="{self.model_name} Inference API",
                lifespan=lifespan,
            )


            # -- routes -----------------------------------------------------------

            @app.get("/health")
            def health():
                \"\"\"Liveness / readiness probe.\"\"\"
                return {{
                    "status": "healthy" if _session else "loading",
                    "model": MODEL_PATH,
                }}

            @app.get("/metadata")
            def metadata():
                \"\"\"Return input/output schema of the loaded model.\"\"\"
                if not _session:
                    raise HTTPException(503, "model not loaded yet")
                return _model_meta

            @app.post("/predict")
            async def predict(request: Request):
                \"\"\"Run inference.

                Expects JSON body mapping input-tensor names to nested lists::

                    {{"input_0": [[1.0, 2.0, 3.0]], "input_1": [[4.0]]}}

                Returns::

                    {{"outputs": [<array>, ...], "latency_ms": 12.3}}
                \"\"\"
                if not _session:
                    raise HTTPException(503, "model not loaded yet")
                try:
                    body = await request.json()
                except Exception:
                    raise HTTPException(400, "invalid JSON body")

                feeds = {{}}
                for inp in _session.get_inputs():
                    if inp.name not in body:
                        raise HTTPException(
                            422,
                            f"missing input tensor '{{inp.name}}'; "
                            f"expected keys: {{[i.name for i in _session.get_inputs()]}}",
                        )
                    feeds[inp.name] = np.array(body[inp.name], dtype=np.float32)

                t0 = time.perf_counter()
                outputs = _session.run(None, feeds)
                latency = (time.perf_counter() - t0) * 1000

                result = [
                    o.tolist() if hasattr(o, "tolist") else o for o in outputs
                ]
                return JSONResponse({{
                    "outputs": result,
                    "latency_ms": round(latency, 2),
                }})


            # -- entrypoint -------------------------------------------------------

            if __name__ == "__main__":
                uvicorn.run(
                    "inference_server:app",
                    host="0.0.0.0",
                    port=PORT,
                    workers=WORKERS,
                    log_level=LOG_LEVEL.lower(),
                )
        """
        return self._write(out / "inference_server.py", server_py)

    # -----------------------------------------------------------------------
    # Generate everything
    # -----------------------------------------------------------------------

    def generate_all(self, output_dir: str | Path) -> DeploymentBundle:
        out = Path(output_dir)
        bundle = DeploymentBundle(output_dir=str(out))

        bundle.inference_handler = self.generate_inference_handler(out)
        bundle.dockerfile = self.generate_dockerfile(out)
        bundle.kubernetes_manifests = self.generate_kubernetes(out)
        bundle.sagemaker_artifacts = self.generate_sagemaker(out)
        bundle.azure_ml_artifacts = self.generate_azure_ml(out)
        bundle.gcp_vertex_artifacts = self.generate_gcp_vertex(out)
        bundle.cost_estimate = self.estimate_cost()

        log.info("deployment bundle written to %s", out)
        return bundle

    # -----------------------------------------------------------------------
    # Cost estimation
    # -----------------------------------------------------------------------

    # Hourly on-demand rates (USD) – approximate as of 2025; override via
    # config["pricing"] if you need current numbers.
    _DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
        "aws": {
            "ml.m5.large": 0.115,
            "ml.m5.xlarge": 0.23,
            "ml.c5.xlarge": 0.204,
            "ml.g4dn.xlarge": 0.736,
            "ml.p3.2xlarge": 3.825,
        },
        "azure": {
            "Standard_DS3_v2": 0.293,
            "Standard_DS4_v2": 0.585,
            "Standard_NC6s_v3": 3.06,
        },
        "gcp": {
            "n1-standard-4": 0.190,
            "n1-standard-8": 0.380,
            "n1-highmem-4": 0.237,
            "a2-highgpu-1g": 3.67,
        },
    }

    def estimate_cost(
        self,
        provider: str = "aws",
        requests_per_day: int = 10_000,
        avg_latency_ms: float = 50,
    ) -> Dict[str, Any]:
        """Rough monthly cost estimate based on instance hours.

        The estimate is intentionally conservative: it assumes a single
        instance running 24/7 and adds a bandwidth surcharge.  Real costs
        depend on auto-scaling, spot pricing, and reserved commitments.
        """
        pricing = self.config.get("pricing", self._DEFAULT_PRICING)
        hours_per_month = 730

        estimates: Dict[str, Any] = {}
        providers = [provider] if provider != "all" else list(pricing.keys())

        for prov in providers:
            rates = pricing.get(prov, {})
            if not rates:
                estimates[prov] = {"error": f"no pricing data for {prov}"}
                continue

            prov_est: Dict[str, Any] = {}
            for instance, hourly in rates.items():
                compute = hourly * hours_per_month

                # Rough request-processing capacity of one instance
                reqs_per_sec = 1000.0 / max(avg_latency_ms, 1)
                reqs_per_month = requests_per_day * 30
                instance_seconds_needed = reqs_per_month / reqs_per_sec
                instance_hours_needed = instance_seconds_needed / 3600

                min_instances = max(1, int(instance_hours_needed / hours_per_month) + 1)

                # Bandwidth: ~1 KB per request is typical for tensor payloads
                bandwidth_gb = (reqs_per_month * 1024) / (1024 ** 3)
                bandwidth_cost = bandwidth_gb * 0.09

                total = compute * min_instances + bandwidth_cost

                prov_est[instance] = {
                    "hourly_rate_usd": hourly,
                    "min_instances": min_instances,
                    "compute_usd_month": round(compute * min_instances, 2),
                    "bandwidth_usd_month": round(bandwidth_cost, 2),
                    "total_usd_month": round(total, 2),
                }

            estimates[prov] = prov_est

        return estimates


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def deploy_model(
    model_path: str | Path,
    output_dir: str | Path,
    target: str = "all",
    config: Optional[Dict[str, Any]] = None,
) -> DeploymentBundle | str | List[str]:
    """CLI-friendly entry point for deployment artifact generation.

    Parameters
    ----------
    model_path : path to the ONNX model
    output_dir : directory to write artifacts into
    target : one of ``"all"``, ``"docker"``, ``"k8s"``, ``"sagemaker"``,
             ``"azure"``, ``"gcp"``, ``"handler"``
    config : optional overrides forwarded to :class:`CloudDeployer`
    """
    deployer = CloudDeployer(model_path, config=config)

    dispatch = {
        "all": deployer.generate_all,
        "docker": deployer.generate_dockerfile,
        "k8s": deployer.generate_kubernetes,
        "sagemaker": deployer.generate_sagemaker,
        "azure": deployer.generate_azure_ml,
        "gcp": deployer.generate_gcp_vertex,
        "handler": deployer.generate_inference_handler,
    }

    generator = dispatch.get(target)
    if generator is None:
        raise ValueError(
            f"unknown target {target!r}; choose from {list(dispatch.keys())}"
        )

    return generator(str(output_dir))
