"""Token metering, budget enforcement, and usage analytics."""

from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    name: str
    input_cost_per_1k: float = 0.01
    output_cost_per_1k: float = 0.03
    gpu_cost_per_hour: float = 2.0
    currency: str = "USD"


@dataclass
class UsageRecord:
    request_id: str
    customer_id: str
    project_id: Optional[str]
    model_name: str
    input_tokens: int
    output_tokens: int
    gpu_seconds: float
    cost_usd: float
    timestamp: str
    latency_ms: float
    provider: str = "local"


@dataclass
class Budget:
    customer_id: str
    daily_limit_usd: float = 100.0
    monthly_limit_usd: float = 2000.0
    max_tokens_per_request: int = 32_000
    max_requests_per_minute: int = 60
    current_daily_spend: float = 0.0
    current_monthly_spend: float = 0.0


@dataclass
class DailyReport:
    date: str
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    by_customer: Dict[str, float]
    by_model: Dict[str, float]
    peak_hour: int
    avg_latency_ms: float


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    request_id    TEXT PRIMARY KEY,
    customer_id   TEXT NOT NULL,
    project_id    TEXT,
    model_name    TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    gpu_seconds   REAL,
    cost_usd      REAL,
    timestamp     TEXT,
    latency_ms    REAL,
    provider      TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_customer ON usage(customer_id);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(timestamp);
"""


def _init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


# ---------------------------------------------------------------------------
# TokenMeter
# ---------------------------------------------------------------------------

class TokenMeter:
    """Per-request metering and cost attribution."""

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        db_path: Optional[str] = None,
    ) -> None:
        self.cost_model = cost_model or CostModel(name="default")
        self._records: List[UsageRecord] = []
        self._db: Optional[sqlite3.Connection] = None
        if db_path:
            self._db = _init_db(db_path)

    # -- public API ----------------------------------------------------------

    def record(
        self,
        request_id: str,
        customer_id: str,
        input_tokens: int,
        output_tokens: int,
        gpu_seconds: float = 0.0,
        model_name: str = "default",
        project_id: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> UsageRecord:
        cost = self._compute_cost(input_tokens, output_tokens, gpu_seconds)
        rec = UsageRecord(
            request_id=request_id,
            customer_id=customer_id,
            project_id=project_id,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            gpu_seconds=gpu_seconds,
            cost_usd=round(cost, 8),
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency_ms,
        )
        self._records.append(rec)
        if self._db:
            self._persist(rec)
        return rec

    def get_usage(
        self,
        customer_id: Optional[str] = None,
        project_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[UsageRecord]:
        results = self._all_records()
        if customer_id:
            results = [r for r in results if r.customer_id == customer_id]
        if project_id:
            results = [r for r in results if r.project_id == project_id]
        if start_date:
            results = [r for r in results if r.timestamp >= start_date]
        if end_date:
            results = [r for r in results if r.timestamp <= end_date]
        return results

    def get_cost_summary(
        self,
        customer_id: Optional[str] = None,
        period: str = "daily",
    ) -> Dict[str, float]:
        records = self.get_usage(customer_id=customer_id)
        buckets: Dict[str, float] = {}
        for rec in records:
            key = self._period_key(rec.timestamp, period)
            buckets[key] = buckets.get(key, 0.0) + rec.cost_usd
        return {k: round(v, 6) for k, v in sorted(buckets.items())}

    def export_csv(
        self,
        path: str,
        customer_id: Optional[str] = None,
    ) -> None:
        records = self.get_usage(customer_id=customer_id)
        fieldnames = list(UsageRecord.__dataclass_fields__.keys())
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))

    def export_prometheus(self) -> str:
        records = self._all_records()
        total_cost = sum(r.cost_usd for r in records)
        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_requests = len(records)

        by_model: Dict[str, float] = {}
        by_customer: Dict[str, float] = {}
        for rec in records:
            by_model[rec.model_name] = by_model.get(rec.model_name, 0.0) + rec.cost_usd
            by_customer[rec.customer_id] = by_customer.get(rec.customer_id, 0.0) + rec.cost_usd

        lines = [
            "# HELP isat_total_cost_usd Total cost in USD",
            "# TYPE isat_total_cost_usd gauge",
            f"isat_total_cost_usd {total_cost:.6f}",
            "# HELP isat_total_input_tokens Total input tokens processed",
            "# TYPE isat_total_input_tokens counter",
            f"isat_total_input_tokens {total_input}",
            "# HELP isat_total_output_tokens Total output tokens generated",
            "# TYPE isat_total_output_tokens counter",
            f"isat_total_output_tokens {total_output}",
            "# HELP isat_total_requests Total number of requests",
            "# TYPE isat_total_requests counter",
            f"isat_total_requests {total_requests}",
        ]

        if by_model:
            lines.append("# HELP isat_cost_by_model Cost broken down by model")
            lines.append("# TYPE isat_cost_by_model gauge")
            for model, cost in sorted(by_model.items()):
                lines.append(f'isat_cost_by_model{{model="{model}"}} {cost:.6f}')

        if by_customer:
            lines.append("# HELP isat_cost_by_customer Cost broken down by customer")
            lines.append("# TYPE isat_cost_by_customer gauge")
            for cid, cost in sorted(by_customer.items()):
                lines.append(f'isat_cost_by_customer{{customer="{cid}"}} {cost:.6f}')

        return "\n".join(lines) + "\n"

    # -- internals -----------------------------------------------------------

    def _compute_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        gpu_seconds: float,
    ) -> float:
        m = self.cost_model
        return (
            (input_tokens / 1000) * m.input_cost_per_1k
            + (output_tokens / 1000) * m.output_cost_per_1k
            + (gpu_seconds / 3600) * m.gpu_cost_per_hour
        )

    def _persist(self, rec: UsageRecord) -> None:
        assert self._db is not None
        d = asdict(rec)
        cols = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        self._db.execute(
            f"INSERT OR REPLACE INTO usage ({cols}) VALUES ({placeholders})",
            list(d.values()),
        )
        self._db.commit()

    def _all_records(self) -> List[UsageRecord]:
        if self._db:
            rows = self._db.execute("SELECT * FROM usage ORDER BY timestamp").fetchall()
            return [
                UsageRecord(**{k: row[k] for k in UsageRecord.__dataclass_fields__})
                for row in rows
            ]
        return list(self._records)

    @staticmethod
    def _period_key(timestamp: str, period: str) -> str:
        dt = datetime.fromisoformat(timestamp)
        if period == "daily":
            return dt.strftime("%Y-%m-%d")
        if period == "weekly":
            iso = dt.isocalendar()
            return f"{iso[0]}-W{iso[1]:02d}"
        if period == "monthly":
            return dt.strftime("%Y-%m")
        return dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# BudgetEnforcer
# ---------------------------------------------------------------------------

class BudgetEnforcer:
    """Rate-limit and spend-cap enforcement."""

    def __init__(self, meter: TokenMeter) -> None:
        self.meter = meter
        self._budgets: Dict[str, Budget] = {}
        self._request_log: Dict[str, List[float]] = {}  # customer -> list of timestamps

    def set_budget(
        self,
        customer_id: str,
        daily_limit: Optional[float] = None,
        monthly_limit: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_rpm: Optional[int] = None,
    ) -> Budget:
        b = self._budgets.get(customer_id, Budget(customer_id=customer_id))
        if daily_limit is not None:
            b.daily_limit_usd = daily_limit
        if monthly_limit is not None:
            b.monthly_limit_usd = monthly_limit
        if max_tokens is not None:
            b.max_tokens_per_request = max_tokens
        if max_rpm is not None:
            b.max_requests_per_minute = max_rpm
        self._budgets[customer_id] = b
        return b

    def check(
        self,
        customer_id: str,
        estimated_tokens: int = 0,
    ) -> Tuple[bool, str]:
        b = self._budgets.get(customer_id)
        if b is None:
            return True, "no budget configured"

        self._maybe_reset(b)

        now = datetime.utcnow().timestamp()
        log = self._request_log.get(customer_id, [])
        recent = [t for t in log if now - t < 60]
        if len(recent) >= b.max_requests_per_minute:
            return False, f"rate limit exceeded ({b.max_requests_per_minute} req/min)"

        if estimated_tokens > b.max_tokens_per_request:
            return False, f"token limit exceeded ({estimated_tokens} > {b.max_tokens_per_request})"

        if b.current_daily_spend >= b.daily_limit_usd:
            return False, f"daily budget exhausted (${b.current_daily_spend:.2f} / ${b.daily_limit_usd:.2f})"

        if b.current_monthly_spend >= b.monthly_limit_usd:
            return False, f"monthly budget exhausted (${b.current_monthly_spend:.2f} / ${b.monthly_limit_usd:.2f})"

        self._request_log.setdefault(customer_id, []).append(now)
        return True, "ok"

    def get_remaining(self, customer_id: str) -> Dict[str, float]:
        b = self._budgets.get(customer_id)
        if b is None:
            return {"daily_remaining": float("inf"), "monthly_remaining": float("inf")}
        self._maybe_reset(b)
        return {
            "daily_remaining": round(max(0.0, b.daily_limit_usd - b.current_daily_spend), 6),
            "monthly_remaining": round(max(0.0, b.monthly_limit_usd - b.current_monthly_spend), 6),
        }

    def _update_spend(self, customer_id: str, cost: float) -> None:
        b = self._budgets.get(customer_id)
        if b is None:
            return
        self._maybe_reset(b)
        b.current_daily_spend += cost
        b.current_monthly_spend += cost

    def _maybe_reset(self, b: Budget) -> None:
        now = datetime.utcnow()
        key = f"_last_reset_{b.customer_id}"
        last: Optional[datetime] = getattr(self, key, None)
        if last is None or last.date() < now.date():
            b.current_daily_spend = 0.0
            setattr(self, key, now)
        month_key = f"_last_month_reset_{b.customer_id}"
        last_m: Optional[datetime] = getattr(self, month_key, None)
        if last_m is None or last_m.month != now.month or last_m.year != now.year:
            b.current_monthly_spend = 0.0
            setattr(self, month_key, now)


# ---------------------------------------------------------------------------
# UsageAnalytics
# ---------------------------------------------------------------------------

class UsageAnalytics:
    """Analytical queries over usage data."""

    def __init__(self, meter: TokenMeter) -> None:
        self.meter = meter

    def daily_report(self, date: Optional[str] = None) -> DailyReport:
        target = date or datetime.utcnow().strftime("%Y-%m-%d")
        records = [
            r for r in self.meter.get_usage()
            if r.timestamp.startswith(target)
        ]
        by_customer: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        hour_counts: Dict[int, int] = {}
        latencies: List[float] = []

        for rec in records:
            by_customer[rec.customer_id] = by_customer.get(rec.customer_id, 0.0) + rec.cost_usd
            by_model[rec.model_name] = by_model.get(rec.model_name, 0.0) + rec.cost_usd
            hr = datetime.fromisoformat(rec.timestamp).hour
            hour_counts[hr] = hour_counts.get(hr, 0) + 1
            latencies.append(rec.latency_ms)

        peak = max(hour_counts, key=hour_counts.get) if hour_counts else 0

        return DailyReport(
            date=target,
            total_requests=len(records),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            total_cost=round(sum(r.cost_usd for r in records), 6),
            by_customer={k: round(v, 6) for k, v in by_customer.items()},
            by_model={k: round(v, 6) for k, v in by_model.items()},
            peak_hour=peak,
            avg_latency_ms=round(float(np.mean(latencies)), 2) if latencies else 0.0,
        )

    def cost_trend(self, days: int = 30) -> List[Dict[str, object]]:
        end = datetime.utcnow().date()
        start = end - timedelta(days=days - 1)
        summary = self.meter.get_cost_summary(period="daily")
        series = []
        current = start
        while current <= end:
            key = current.isoformat()
            series.append({"date": key, "cost": summary.get(key, 0.0)})
            current += timedelta(days=1)
        return series

    def anomaly_detection(
        self,
        customer_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        summary = self.meter.get_cost_summary(customer_id=customer_id, period="daily")
        if len(summary) < 3:
            return []

        costs = np.array(list(summary.values()), dtype=float)
        window = min(7, len(costs))
        anomalies: List[Dict[str, object]] = []

        for i in range(window, len(costs)):
            window_vals = costs[i - window : i]
            mean = float(np.mean(window_vals))
            std = float(np.std(window_vals))
            if std > 0 and costs[i] > mean + 2 * std:
                dates = list(summary.keys())
                anomalies.append({
                    "date": dates[i],
                    "cost": float(costs[i]),
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "z_score": round((float(costs[i]) - mean) / std, 2),
                })
        return anomalies

    def top_customers(
        self,
        k: int = 10,
        period: str = "monthly",
    ) -> List[Dict[str, object]]:
        records = self.meter.get_usage()
        totals: Dict[str, float] = {}
        for rec in records:
            bucket = self.meter._period_key(rec.timestamp, period)
            current = datetime.utcnow()
            if period == "monthly":
                current_bucket = current.strftime("%Y-%m")
            elif period == "weekly":
                iso = current.isocalendar()
                current_bucket = f"{iso[0]}-W{iso[1]:02d}"
            else:
                current_bucket = current.strftime("%Y-%m-%d")
            if bucket == current_bucket:
                totals[rec.customer_id] = totals.get(rec.customer_id, 0.0) + rec.cost_usd

        ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)[:k]
        return [{"customer_id": cid, "total_cost": round(c, 6)} for cid, c in ranked]

    def model_cost_breakdown(self) -> Dict[str, float]:
        records = self.meter.get_usage()
        by_model: Dict[str, float] = {}
        for rec in records:
            by_model[rec.model_name] = by_model.get(rec.model_name, 0.0) + rec.cost_usd
        return {k: round(v, 6) for k, v in sorted(by_model.items())}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def token_economics(action: str = "report", **kwargs) -> object:
    """CLI entry point for token economics operations.

    Actions:
        report  — generate a daily usage report
        budget  — set or query budget for a customer
        export  — export usage records to CSV
        analyze — run cost trend / anomaly analysis
    """
    db_path = kwargs.get("db_path", "token_usage.db")
    meter = TokenMeter(db_path=db_path)
    analytics = UsageAnalytics(meter)
    enforcer = BudgetEnforcer(meter)

    if action == "report":
        report = analytics.daily_report(date=kwargs.get("date"))
        return asdict(report)

    if action == "budget":
        cid = kwargs.get("customer_id", "default")
        if kwargs.get("set"):
            enforcer.set_budget(
                cid,
                daily_limit=kwargs.get("daily_limit"),
                monthly_limit=kwargs.get("monthly_limit"),
                max_tokens=kwargs.get("max_tokens"),
                max_rpm=kwargs.get("max_rpm"),
            )
            return {"status": "budget_set", "remaining": enforcer.get_remaining(cid)}
        return enforcer.get_remaining(cid)

    if action == "export":
        path = kwargs.get("path", "usage_export.csv")
        meter.export_csv(path, customer_id=kwargs.get("customer_id"))
        return {"status": "exported", "path": path}

    if action == "analyze":
        kind = kwargs.get("kind", "trend")
        if kind == "trend":
            return analytics.cost_trend(days=kwargs.get("days", 30))
        if kind == "anomaly":
            return analytics.anomaly_detection(customer_id=kwargs.get("customer_id"))
        if kind == "top":
            return analytics.top_customers(k=kwargs.get("k", 10))
        if kind == "model":
            return analytics.model_cost_breakdown()
        return {"error": f"unknown analysis kind: {kind}"}

    return {"error": f"unknown action: {action}"}
