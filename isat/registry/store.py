"""Model version registry with configuration tracking.

Tracks model versions, their optimal configurations, and deployment metadata.
Enables teams to share and reproduce inference configurations across environments.

Usage:
    registry = ModelRegistry("isat_registry.db")
    registry.register("resnet50", "v1.2", "model.onnx", config={...})
    registry.promote("resnet50", "v1.2", stage="production")
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.registry")


@dataclass
class ModelVersion:
    model_name: str
    version: str
    model_path: str
    sha256: str
    config: dict
    stage: str  # "development", "staging", "production", "archived"
    tags: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    registered_at: str = ""
    promoted_at: str = ""
    notes: str = ""


@dataclass
class RegistryListing:
    models: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        if not self.models:
            return "  Registry is empty"
        lines = [
            f"  {'Model':<25} {'Version':<10} {'Stage':<15} {'Latency ms':>12} {'SHA256':>18} {'Registered':<20}",
            f"  {'-'*25} {'-'*10} {'-'*15} {'-'*12} {'-'*18} {'-'*20}",
        ]
        for m in self.models:
            lat = f"{m['latency_ms']:.1f}" if m.get("latency_ms") else "N/A"
            lines.append(
                f"  {m['model_name']:<25} {m['version']:<10} {m['stage']:<15} "
                f"{lat:>12} {m['sha256'][:16]:>18} {m['registered_at']:<20}"
            )
        return "\n".join(lines)


@dataclass
class VersionDiff:
    model_name: str
    version_a: str
    version_b: str
    config_changes: list[str] = field(default_factory=list)
    metric_changes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  {self.model_name}: {self.version_a} vs {self.version_b}",
        ]
        if self.config_changes:
            lines.append(f"\n  Config changes:")
            for c in self.config_changes:
                lines.append(f"    {c}")
        if self.metric_changes:
            lines.append(f"\n  Metric changes:")
            for c in self.metric_changes:
                lines.append(f"    {c}")
        if not self.config_changes and not self.metric_changes:
            lines.append("  No differences")
        return "\n".join(lines)


class ModelRegistry:
    """SQLite-backed model version registry."""

    def __init__(self, db_path: str = "isat_registry.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        db = sqlite3.connect(self.db_path)
        db.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                version TEXT NOT NULL,
                model_path TEXT,
                sha256 TEXT,
                config_json TEXT DEFAULT '{}',
                stage TEXT DEFAULT 'development',
                tags_json TEXT DEFAULT '[]',
                metrics_json TEXT DEFAULT '{}',
                notes TEXT DEFAULT '',
                registered_at TEXT DEFAULT (datetime('now')),
                promoted_at TEXT DEFAULT '',
                UNIQUE(model_name, version)
            )
        """)
        db.commit()
        db.close()

    def register(
        self,
        model_name: str,
        version: str,
        model_path: str,
        config: dict | None = None,
        tags: list[str] | None = None,
        metrics: dict | None = None,
        notes: str = "",
    ) -> ModelVersion:
        sha = _sha256_file(model_path)

        db = sqlite3.connect(self.db_path)
        try:
            db.execute(
                "INSERT INTO versions (model_name, version, model_path, sha256, "
                "config_json, tags_json, metrics_json, notes) VALUES (?,?,?,?,?,?,?,?)",
                (
                    model_name, version, model_path, sha,
                    json.dumps(config or {}),
                    json.dumps(tags or []),
                    json.dumps(metrics or {}),
                    notes,
                ),
            )
            db.commit()
        except sqlite3.IntegrityError:
            db.execute(
                "UPDATE versions SET model_path=?, sha256=?, config_json=?, "
                "tags_json=?, metrics_json=?, notes=? WHERE model_name=? AND version=?",
                (
                    model_path, sha, json.dumps(config or {}),
                    json.dumps(tags or []), json.dumps(metrics or {}),
                    notes, model_name, version,
                ),
            )
            db.commit()
        finally:
            db.close()

        return ModelVersion(
            model_name=model_name, version=version, model_path=model_path,
            sha256=sha, config=config or {}, stage="development",
            tags=tags or [], metrics=metrics or {}, notes=notes,
        )

    def promote(self, model_name: str, version: str, stage: str = "production"):
        db = sqlite3.connect(self.db_path)
        if stage == "production":
            db.execute(
                "UPDATE versions SET stage='staging' WHERE model_name=? AND stage='production'",
                (model_name,),
            )
        db.execute(
            "UPDATE versions SET stage=?, promoted_at=datetime('now') WHERE model_name=? AND version=?",
            (stage, model_name, version),
        )
        db.commit()
        db.close()
        log.info("Promoted %s/%s to %s", model_name, version, stage)

    def list_models(self, model_name: str = "", stage: str = "") -> RegistryListing:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        query = "SELECT * FROM versions"
        params: list = []
        conditions = []
        if model_name:
            conditions.append("model_name=?")
            params.append(model_name)
        if stage:
            conditions.append("stage=?")
            params.append(stage)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY model_name, registered_at DESC"

        rows = db.execute(query, params).fetchall()
        db.close()

        models = []
        for r in rows:
            metrics = json.loads(r["metrics_json"]) if r["metrics_json"] else {}
            models.append({
                "model_name": r["model_name"],
                "version": r["version"],
                "stage": r["stage"],
                "sha256": r["sha256"] or "",
                "registered_at": r["registered_at"] or "",
                "latency_ms": metrics.get("mean_latency_ms"),
            })

        return RegistryListing(models=models)

    def get_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM versions WHERE model_name=? AND version=?",
            (model_name, version),
        ).fetchone()
        db.close()
        if not row:
            return None
        return ModelVersion(
            model_name=row["model_name"], version=row["version"],
            model_path=row["model_path"] or "", sha256=row["sha256"] or "",
            config=json.loads(row["config_json"]) if row["config_json"] else {},
            stage=row["stage"] or "development",
            tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
            metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else {},
            registered_at=row["registered_at"] or "",
            promoted_at=row["promoted_at"] or "",
            notes=row["notes"] or "",
        )

    def get_production(self, model_name: str) -> Optional[ModelVersion]:
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        row = db.execute(
            "SELECT * FROM versions WHERE model_name=? AND stage='production' ORDER BY id DESC LIMIT 1",
            (model_name,),
        ).fetchone()
        db.close()
        if not row:
            return None
        return self.get_version(row["model_name"], row["version"])

    def diff_versions(self, model_name: str, version_a: str, version_b: str) -> VersionDiff:
        va = self.get_version(model_name, version_a)
        vb = self.get_version(model_name, version_b)

        diff = VersionDiff(model_name=model_name, version_a=version_a, version_b=version_b)

        if not va or not vb:
            diff.config_changes.append("One or both versions not found")
            return diff

        all_keys = set(va.config.keys()) | set(vb.config.keys())
        for k in sorted(all_keys):
            old = va.config.get(k)
            new = vb.config.get(k)
            if old != new:
                diff.config_changes.append(f"{k}: {old} -> {new}")

        all_metrics = set(va.metrics.keys()) | set(vb.metrics.keys())
        for k in sorted(all_metrics):
            old = va.metrics.get(k)
            new = vb.metrics.get(k)
            if old != new:
                if isinstance(old, (int, float)) and isinstance(new, (int, float)):
                    pct = (new - old) / old * 100 if old else 0
                    diff.metric_changes.append(f"{k}: {old} -> {new} ({pct:+.1f}%)")
                else:
                    diff.metric_changes.append(f"{k}: {old} -> {new}")

        return diff

    def delete_version(self, model_name: str, version: str):
        db = sqlite3.connect(self.db_path)
        db.execute("DELETE FROM versions WHERE model_name=? AND version=?", (model_name, version))
        db.commit()
        db.close()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()
