"""SQLite-backed results store for ISAT tuning runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from isat.search.engine import TuneResult


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp    REAL    NOT NULL,
    hw_hash      TEXT    NOT NULL,
    model_hash   TEXT    NOT NULL,
    model_name   TEXT    NOT NULL,
    config_label TEXT    NOT NULL,
    env_json     TEXT    NOT NULL,
    mean_ms      REAL,
    p50_ms       REAL,
    p95_ms       REAL,
    p99_ms       REAL,
    min_ms       REAL,
    max_ms       REAL,
    std_ms       REAL,
    throughput   REAL,
    peak_temp    REAL,
    peak_power   REAL,
    peak_vram    REAL,
    peak_gtt     REAL,
    warmup       INTEGER,
    measured     INTEGER,
    cooldown     REAL,
    error        TEXT,
    extra_json   TEXT
);

CREATE INDEX IF NOT EXISTS idx_hw_model ON runs(hw_hash, model_hash);
CREATE INDEX IF NOT EXISTS idx_model    ON runs(model_name);
"""


class ResultsDB:
    """Persistent storage for tuning results."""

    def __init__(self, db_path: str = "isat_results.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.row_factory = sqlite3.Row

    def save_result(
        self,
        result: TuneResult,
        hw_hash: str,
        model_hash: str,
        model_name: str,
        extra: Optional[dict] = None,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO runs
            (timestamp, hw_hash, model_hash, model_name, config_label,
             env_json, mean_ms, p50_ms, p95_ms, p99_ms, min_ms, max_ms,
             std_ms, throughput, peak_temp, peak_power, peak_vram, peak_gtt,
             warmup, measured, cooldown, error, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), hw_hash, model_hash, model_name,
                result.config.label,
                json.dumps(result.config.merged_env),
                result.mean_latency_ms, result.p50_latency_ms,
                result.p95_latency_ms, result.p99_latency_ms,
                result.min_latency_ms, result.max_latency_ms,
                result.std_dev_ms, result.throughput_fps,
                result.peak_gpu_temp_c, result.peak_power_w,
                result.peak_vram_mb, result.peak_gtt_mb,
                result.warmup_runs, result.measured_runs,
                result.cooldown_s, result.error,
                json.dumps(extra) if extra else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore

    def save_batch(
        self,
        results: list[TuneResult],
        hw_hash: str,
        model_hash: str,
        model_name: str,
    ) -> list[int]:
        ids = []
        for r in results:
            ids.append(self.save_result(r, hw_hash, model_hash, model_name))
        return ids

    def best_for_model(self, model_name: str, limit: int = 5) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM runs
            WHERE model_name = ? AND error IS NULL
            ORDER BY mean_ms ASC
            LIMIT ?""",
            (model_name, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def best_for_hw_model(self, hw_hash: str, model_hash: str, limit: int = 5) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM runs
            WHERE hw_hash = ? AND model_hash = ? AND error IS NULL
            ORDER BY mean_ms ASC
            LIMIT ?""",
            (hw_hash, model_hash, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_runs(self, model_name: Optional[str] = None) -> list[dict]:
        if model_name:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE model_name = ? ORDER BY timestamp DESC",
                (model_name,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY timestamp DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
