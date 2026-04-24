"""ISAT REST API server.

Provides remote tuning, job management, and results querying over HTTP.
Run with: isat serve --port 8000

Requires: pip install isat[server]
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("isat.server")


@dataclass
class TuneJob:
    job_id: str
    model_path: str
    status: str = "pending"
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    progress: float = 0.0
    total_configs: int = 0
    completed_configs: int = 0
    best_latency_ms: float = float("inf")
    best_config_label: str = ""
    error: Optional[str] = None
    results: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()


class ISATServer:
    """In-process API server for ISAT. Wraps the tuning pipeline with job management."""

    def __init__(self, db_path: str = "isat_results.db", output_dir: str = "isat_output"):
        self.db_path = db_path
        self.output_dir = output_dir
        self.jobs: dict[str, TuneJob] = {}
        self._lock = threading.Lock()

    def submit_job(
        self,
        model_path: str,
        warmup: int = 3,
        runs: int = 5,
        cooldown: float = 60.0,
        provider: str = "MIGraphXExecutionProvider",
        max_configs: int = 0,
        skip_precision: bool = False,
        skip_graph: bool = False,
    ) -> str:
        """Submit a new tuning job. Returns job_id."""
        job_id = str(uuid.uuid4())[:8]
        job = TuneJob(job_id=job_id, model_path=model_path)

        with self._lock:
            self.jobs[job_id] = job

        t = threading.Thread(
            target=self._run_job,
            args=(job_id, model_path, warmup, runs, cooldown, provider, max_configs, skip_precision, skip_graph),
            daemon=True,
        )
        t.start()
        return job_id

    def get_job(self, job_id: str) -> Optional[TuneJob]:
        with self._lock:
            return self.jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "job_id": j.job_id,
                    "model_path": j.model_path,
                    "status": j.status,
                    "progress": j.progress,
                    "best_latency_ms": j.best_latency_ms,
                    "best_config": j.best_config_label,
                }
                for j in self.jobs.values()
            ]

    def _run_job(
        self,
        job_id: str,
        model_path: str,
        warmup: int,
        runs: int,
        cooldown: float,
        provider: str,
        max_configs: int,
        skip_precision: bool,
        skip_graph: bool,
    ) -> None:
        from isat.benchmark.runner import BenchmarkRunner
        from isat.database.store import ResultsDB
        from isat.fingerprint.hardware import fingerprint_hardware
        from isat.fingerprint.model import fingerprint_model
        from isat.report.generator import ReportGenerator
        from isat.search.engine import SearchEngine

        job = self.jobs[job_id]

        try:
            with self._lock:
                job.status = "running"
                job.started_at = time.time()

            hw = fingerprint_hardware()
            model_fp = fingerprint_model(model_path)

            engine = SearchEngine(
                hw, model_fp, warmup=warmup, runs=runs, cooldown=cooldown,
                max_configs=max_configs, skip_precision=skip_precision, skip_graph=skip_graph,
            )
            candidates = engine.generate_candidates()

            with self._lock:
                job.total_configs = len(candidates)

            runner = BenchmarkRunner(
                hw, model_fp, model_path, warmup=warmup, runs=runs,
                cooldown=cooldown, provider=provider,
            )

            results = []
            for idx, config in enumerate(candidates):
                result = runner.run_single(config)
                results.append(result)

                with self._lock:
                    job.completed_configs = idx + 1
                    job.progress = (idx + 1) / len(candidates) * 100

                    if result.error is None and result.mean_latency_ms < job.best_latency_ms:
                        job.best_latency_ms = result.mean_latency_ms
                        job.best_config_label = config.label

                if idx < len(candidates) - 1:
                    runner.thermal.wait_cooldown()

            db = ResultsDB(self.db_path)
            db.save_batch(results, hw.fingerprint_hash, model_fp.fingerprint_hash, model_fp.name)
            db.close()

            job_dir = os.path.join(self.output_dir, job_id)
            reporter = ReportGenerator(hw, model_fp, results, output_dir=job_dir)
            reporter.generate_all()

            with self._lock:
                job.status = "completed"
                job.completed_at = time.time()
                job.results = [r.to_dict() for r in results]

        except Exception as e:
            with self._lock:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = time.time()
            log.error("Job %s failed: %s", job_id, e, exc_info=True)


def create_app(db_path: str = "isat_results.db", output_dir: str = "isat_output"):
    """Create a FastAPI application for the ISAT server."""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the ISAT server. "
            "Install with: pip install isat[server]"
        )

    app = FastAPI(
        title="ISAT API",
        description="Inference Stack Auto-Tuner -- REST API for remote model tuning",
        version="0.1.0",
    )

    server = ISATServer(db_path=db_path, output_dir=output_dir)

    class TuneRequest(BaseModel):
        model_path: str
        warmup: int = 3
        runs: int = 5
        cooldown: float = 60.0
        provider: str = "MIGraphXExecutionProvider"
        max_configs: int = 0
        skip_precision: bool = False
        skip_graph: bool = False

    class InspectRequest(BaseModel):
        model_path: str

    @app.post("/api/v1/tune")
    def submit_tune(req: TuneRequest):
        if not Path(req.model_path).exists():
            raise HTTPException(404, f"Model not found: {req.model_path}")
        job_id = server.submit_job(
            req.model_path, req.warmup, req.runs, req.cooldown,
            req.provider, req.max_configs, req.skip_precision, req.skip_graph,
        )
        return {"job_id": job_id, "status": "submitted"}

    @app.get("/api/v1/jobs")
    def list_jobs():
        return server.list_jobs()

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str):
        job = server.get_job(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return {
            "job_id": job.job_id,
            "status": job.status,
            "progress": job.progress,
            "total_configs": job.total_configs,
            "completed_configs": job.completed_configs,
            "best_latency_ms": job.best_latency_ms,
            "best_config": job.best_config_label,
            "error": job.error,
            "results": job.results if job.status == "completed" else [],
        }

    @app.get("/api/v1/jobs/{job_id}/report")
    def get_report(job_id: str):
        report_path = Path(output_dir) / job_id / "isat_report.json"
        if not report_path.exists():
            raise HTTPException(404, "Report not ready")
        return JSONResponse(json.loads(report_path.read_text()))

    @app.get("/api/v1/jobs/{job_id}/report/html")
    def get_html_report(job_id: str):
        html_path = Path(output_dir) / job_id / "isat_report.html"
        if not html_path.exists():
            raise HTTPException(404, "HTML report not ready")
        return FileResponse(str(html_path), media_type="text/html")

    @app.post("/api/v1/inspect")
    def inspect_model(req: InspectRequest):
        if not Path(req.model_path).exists():
            raise HTTPException(404, f"Model not found: {req.model_path}")
        from isat.fingerprint.model import fingerprint_model
        fp = fingerprint_model(req.model_path)
        return json.loads(fp.to_json())

    @app.get("/api/v1/hardware")
    def hardware_info():
        from isat.fingerprint.hardware import fingerprint_hardware
        hw = fingerprint_hardware()
        return json.loads(hw.to_json())

    @app.get("/api/v1/history")
    def history(model: str = None, top: int = 20):
        from isat.database.store import ResultsDB
        if not Path(db_path).exists():
            return []
        db = ResultsDB(db_path)
        if model:
            rows = db.best_for_model(model, limit=top)
        else:
            rows = db.all_runs()[:top]
        db.close()
        return rows

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app
