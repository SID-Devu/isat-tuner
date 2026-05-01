"""SLO-aware request scheduler.

Routes inference requests through admission control, weighted fair queuing,
and priority-based preemption so that per-tier latency SLOs are honoured.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("isat.slo_scheduler")

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

PREFILL_MS_PER_TOKEN = 0.05
DECODE_MS_PER_TOKEN = 15.0


@dataclass
class SLOTarget:
    max_ttft_ms: float = 500.0
    max_e2e_latency_ms: float = 5000.0
    min_tps: float = 10.0
    priority: int = 1


@dataclass
class CustomerTier:
    name: str
    priority: int
    rate_limit_rpm: int
    max_concurrent: int
    slo: SLOTarget


@dataclass
class SLORequest:
    request_id: str
    prompt_ids: List[int]
    customer_id: str
    tier: CustomerTier
    slo: SLOTarget
    created_at: float
    max_tokens: int
    deadline_ms: float = field(init=False)

    def __post_init__(self) -> None:
        self.deadline_ms = self.created_at * 1000.0 + self.slo.max_e2e_latency_ms


@dataclass
class SLOReport:
    total_requests: int
    slo_hit_rate: float
    per_tier_stats: Dict[str, dict]
    violations: List[dict]
    avg_queue_depth: float
    avg_wait_ms: float


# ---------------------------------------------------------------------------
# Admission Controller
# ---------------------------------------------------------------------------


class AdmissionController:
    """Gate-keeps incoming requests based on capacity, rate limits, and SLO
    feasibility."""

    def __init__(self, max_capacity: int = 100) -> None:
        self.max_capacity = max_capacity
        self._customer_timestamps: Dict[str, List[float]] = defaultdict(list)

    def admit(
        self,
        request: SLORequest,
        current_load: Dict,
    ) -> bool:
        queue_depth: int = current_load.get("queue_depth", 0)
        running: List[SLORequest] = current_load.get("running_requests", [])
        customer_active: int = current_load.get("customer_active", {}).get(
            request.customer_id, 0
        )

        if queue_depth + len(running) >= self.max_capacity:
            log.debug("Rejected %s: at capacity", request.request_id)
            return False

        if customer_active >= request.tier.max_concurrent:
            log.debug("Rejected %s: concurrent limit", request.request_id)
            return False

        if not self._check_rate_limit(request):
            log.debug("Rejected %s: rate limit", request.request_id)
            return False

        if self._would_violate_existing_slos(request, running):
            log.debug("Rejected %s: would violate existing SLOs", request.request_id)
            return False

        self._customer_timestamps[request.customer_id].append(time.time())
        return True

    # -- internals ----------------------------------------------------------

    def _check_rate_limit(self, request: SLORequest) -> bool:
        now = time.time()
        window_start = now - 60.0
        stamps = self._customer_timestamps[request.customer_id]
        stamps[:] = [t for t in stamps if t >= window_start]
        return len(stamps) < request.tier.rate_limit_rpm

    def _estimate_completion_time(
        self, request: SLORequest, current_load: Dict
    ) -> float:
        prompt_len = len(request.prompt_ids)
        prefill_time = prompt_len * PREFILL_MS_PER_TOKEN
        decode_time = request.max_tokens * DECODE_MS_PER_TOKEN

        n_running = len(current_load.get("running_requests", []))
        contention_factor = 1.0 + 0.1 * n_running
        return (prefill_time + decode_time) * contention_factor

    def _would_violate_existing_slos(
        self,
        request: SLORequest,
        running_requests: List[SLORequest],
    ) -> bool:
        if not running_requests:
            return False

        now_ms = time.time() * 1000.0
        new_contention = 1.0 + 0.1 * (len(running_requests) + 1)

        for r in running_requests:
            elapsed_ms = now_ms - r.created_at * 1000.0
            remaining_decode = r.max_tokens * DECODE_MS_PER_TOKEN
            estimated_total = elapsed_ms + remaining_decode * new_contention
            if estimated_total > r.slo.max_e2e_latency_ms:
                return True
        return False


# ---------------------------------------------------------------------------
# Fair Scheduler
# ---------------------------------------------------------------------------


class FairScheduler:
    """Weighted fair queue across customer tiers with preemption support."""

    def __init__(self, tiers: List[CustomerTier]) -> None:
        self.tiers = {t.name: t for t in tiers}
        self._queues: Dict[str, deque[SLORequest]] = {
            t.name: deque() for t in tiers
        }
        self._virtual_times: Dict[str, float] = {t.name: 0.0 for t in tiers}
        self._running: Dict[str, SLORequest] = {}

    def enqueue(self, request: SLORequest) -> None:
        tier_name = request.tier.name
        if tier_name not in self._queues:
            self._queues[tier_name] = deque()
            self._virtual_times[tier_name] = 0.0
            self.tiers[tier_name] = request.tier
        self._queues[tier_name].append(request)

    def dequeue(self) -> Optional[SLORequest]:
        non_empty = {
            name: q for name, q in self._queues.items() if q
        }
        if not non_empty:
            return None

        weights: Dict[str, float] = {}
        for name in non_empty:
            tier = self.tiers[name]
            weights[name] = 1.0 / tier.priority

        total_weight = sum(weights.values())
        probs = np.array([weights[n] / total_weight for n in non_empty])
        tier_names = list(non_empty.keys())

        chosen = str(np.random.choice(tier_names, p=probs))
        request = self._queues[chosen].popleft()

        self._virtual_times[chosen] += 1.0 / weights[chosen]
        self._running[request.request_id] = request
        return request

    def preempt(
        self, request_to_protect: SLORequest
    ) -> Optional[SLORequest]:
        if not self._running:
            return None

        candidates = sorted(
            self._running.values(),
            key=lambda r: r.tier.priority,
            reverse=True,
        )

        for candidate in candidates:
            if candidate.tier.priority <= request_to_protect.tier.priority:
                continue
            if candidate.request_id == request_to_protect.request_id:
                continue

            now_ms = time.time() * 1000.0
            freed_contention = 1.0 + 0.1 * (len(self._running) - 1)
            remaining = request_to_protect.max_tokens * DECODE_MS_PER_TOKEN
            elapsed = now_ms - request_to_protect.created_at * 1000.0
            projected = elapsed + remaining * freed_contention

            if projected <= request_to_protect.slo.max_e2e_latency_ms:
                del self._running[candidate.request_id]
                self._queues[candidate.tier.name].appendleft(candidate)
                log.info(
                    "Preempted %s (tier=%s) to protect %s",
                    candidate.request_id,
                    candidate.tier.name,
                    request_to_protect.request_id,
                )
                return candidate

        return None

    def _virtual_time(self, tier: CustomerTier) -> float:
        return self._virtual_times.get(tier.name, 0.0)

    def mark_complete(self, request_id: str) -> None:
        self._running.pop(request_id, None)

    @property
    def queue_depths(self) -> Dict[str, int]:
        return {name: len(q) for name, q in self._queues.items()}

    @property
    def running_requests(self) -> List[SLORequest]:
        return list(self._running.values())


# ---------------------------------------------------------------------------
# SLO Scheduler (main orchestrator)
# ---------------------------------------------------------------------------

_DEFAULT_TIERS = [
    CustomerTier(
        name="premium",
        priority=1,
        rate_limit_rpm=600,
        max_concurrent=50,
        slo=SLOTarget(max_ttft_ms=200, max_e2e_latency_ms=500, min_tps=50, priority=1),
    ),
    CustomerTier(
        name="standard",
        priority=2,
        rate_limit_rpm=120,
        max_concurrent=20,
        slo=SLOTarget(max_ttft_ms=500, max_e2e_latency_ms=2000, min_tps=20, priority=2),
    ),
    CustomerTier(
        name="batch",
        priority=3,
        rate_limit_rpm=30,
        max_concurrent=5,
        slo=SLOTarget(max_ttft_ms=5000, max_e2e_latency_ms=30000, min_tps=5, priority=3),
    ),
]


class SLOScheduler:
    """Top-level scheduler combining admission control, fair queuing, and SLO
    tracking."""

    def __init__(self, tiers: Optional[List[CustomerTier]] = None) -> None:
        self._tiers = {t.name: t for t in (tiers or _DEFAULT_TIERS)}
        self._fair = FairScheduler(list(self._tiers.values()))
        self._admission = AdmissionController()
        self._stats: Dict[str, List[dict]] = defaultdict(list)
        self._customer_active: Dict[str, int] = defaultdict(int)
        self._queue_depth_samples: List[int] = []
        self._wait_samples: List[float] = []
        self._violations: List[dict] = []

    # -- tier management ----------------------------------------------------

    def add_tier(self, tier: CustomerTier) -> None:
        self._tiers[tier.name] = tier
        self._fair.tiers[tier.name] = tier
        if tier.name not in self._fair._queues:
            self._fair._queues[tier.name] = deque()
            self._fair._virtual_times[tier.name] = 0.0

    # -- request lifecycle --------------------------------------------------

    def submit(
        self,
        prompt_ids: List[int],
        customer_id: str = "default",
        max_tokens: int = 256,
        **kwargs,
    ) -> Optional[SLORequest]:
        tier = self._resolve_tier(customer_id, kwargs)
        slo = kwargs.get("slo", tier.slo)

        request = SLORequest(
            request_id=uuid.uuid4().hex[:12],
            prompt_ids=prompt_ids,
            customer_id=customer_id,
            tier=tier,
            slo=slo,
            created_at=time.time(),
            max_tokens=max_tokens,
        )

        current_load = {
            "queue_depth": sum(self._fair.queue_depths.values()),
            "running_requests": self._fair.running_requests,
            "customer_active": dict(self._customer_active),
        }

        if not self._admission.admit(request, current_load):
            log.info("Request %s rejected by admission control", request.request_id)
            return None

        self._fair.enqueue(request)
        self._customer_active[customer_id] += 1
        self._queue_depth_samples.append(
            sum(self._fair.queue_depths.values())
        )
        log.info(
            "Request %s submitted (tier=%s, customer=%s)",
            request.request_id,
            tier.name,
            customer_id,
        )
        return request

    def schedule(self) -> List[SLORequest]:
        batch: List[SLORequest] = []
        while True:
            req = self._fair.dequeue()
            if req is None:
                break

            now_ms = time.time() * 1000.0
            if now_ms > req.deadline_ms:
                self._record_violation(req, "deadline_expired_before_schedule")
                self._customer_active[req.customer_id] = max(
                    0, self._customer_active[req.customer_id] - 1
                )
                continue

            wait_ms = now_ms - req.created_at * 1000.0
            self._wait_samples.append(wait_ms)
            batch.append(req)

            if len(batch) >= 8:
                break

        if batch:
            for req in batch:
                if now_ms + self._admission._estimate_completion_time(
                    req, {"running_requests": self._fair.running_requests}
                ) > req.deadline_ms:
                    preempted = self._fair.preempt(req)
                    if preempted:
                        log.info(
                            "Preempted %s to protect %s",
                            preempted.request_id,
                            req.request_id,
                        )

        return batch

    def complete(self, request_id: str, latency_ms: float) -> None:
        self._fair.mark_complete(request_id)

        running = self._fair.running_requests
        req = None
        for s in self._stats.values():
            for entry in s:
                if entry.get("request_id") == request_id:
                    req = entry.get("_request")
                    break

        tier_name = "unknown"
        customer_id = "unknown"
        slo_hit = True

        if req is not None:
            tier_name = req.tier.name
            customer_id = req.customer_id
            slo_hit = latency_ms <= req.slo.max_e2e_latency_ms
            self._customer_active[customer_id] = max(
                0, self._customer_active[customer_id] - 1
            )
            if not slo_hit:
                self._record_violation(req, "latency_exceeded")
        else:
            for tname, tier in self._tiers.items():
                tier_name = tname
                break

        self._stats[tier_name].append(
            {
                "request_id": request_id,
                "latency_ms": latency_ms,
                "slo_hit": slo_hit,
                "_request": req,
            }
        )

    def get_slo_report(self) -> SLOReport:
        total = sum(len(v) for v in self._stats.values())
        hits = sum(
            sum(1 for e in v if e["slo_hit"]) for v in self._stats.values()
        )

        per_tier: Dict[str, dict] = {}
        for tier_name, entries in self._stats.items():
            n = len(entries)
            if n == 0:
                continue
            tier_hits = sum(1 for e in entries if e["slo_hit"])
            latencies = [e["latency_ms"] for e in entries]
            per_tier[tier_name] = {
                "total": n,
                "slo_hit_rate": tier_hits / n,
                "avg_latency_ms": float(np.mean(latencies)),
                "p99_latency_ms": float(np.percentile(latencies, 99)),
                "queue_depth": self._fair.queue_depths.get(tier_name, 0),
            }

        return SLOReport(
            total_requests=total,
            slo_hit_rate=hits / total if total > 0 else 1.0,
            per_tier_stats=per_tier,
            violations=list(self._violations),
            avg_queue_depth=(
                float(np.mean(self._queue_depth_samples))
                if self._queue_depth_samples
                else 0.0
            ),
            avg_wait_ms=(
                float(np.mean(self._wait_samples))
                if self._wait_samples
                else 0.0
            ),
        )

    def get_customer_stats(self, customer_id: str) -> dict:
        entries = []
        for tier_entries in self._stats.values():
            for e in tier_entries:
                req = e.get("_request")
                if req is not None and getattr(req, "customer_id", None) == customer_id:
                    entries.append(e)

        if not entries:
            return {
                "customer_id": customer_id,
                "total_requests": 0,
                "active": self._customer_active.get(customer_id, 0),
            }

        latencies = [e["latency_ms"] for e in entries]
        return {
            "customer_id": customer_id,
            "total_requests": len(entries),
            "slo_hit_rate": sum(1 for e in entries if e["slo_hit"]) / len(entries),
            "avg_latency_ms": float(np.mean(latencies)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "active": self._customer_active.get(customer_id, 0),
        }

    # -- internals ----------------------------------------------------------

    def _resolve_tier(self, customer_id: str, kwargs: dict) -> CustomerTier:
        tier_name = kwargs.get("tier_name")
        if tier_name and tier_name in self._tiers:
            return self._tiers[tier_name]
        return next(iter(self._tiers.values()))

    def _record_violation(self, request: SLORequest, reason: str) -> None:
        self._violations.append(
            {
                "request_id": request.request_id,
                "customer_id": request.customer_id,
                "tier": request.tier.name,
                "reason": reason,
                "timestamp": time.time(),
            }
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def slo_schedule(
    model_path: Optional[str] = None,
    tiers: Optional[List[CustomerTier]] = None,
    **kwargs,
) -> SLOScheduler:
    """Create and optionally demonstrate an SLO-aware scheduler.

    Parameters
    ----------
    model_path : str, optional
        Path to a model (reserved for future inference-loop integration).
    tiers : list[CustomerTier], optional
        Customer tiers.  Falls back to built-in defaults.
    **kwargs
        Forwarded to :class:`SLOScheduler`.

    Returns
    -------
    SLOScheduler
        A ready-to-use scheduler instance.
    """
    scheduler = SLOScheduler(tiers=tiers)

    if kwargs.get("demo", False):
        _run_demo(scheduler)

    return scheduler


def _run_demo(scheduler: SLOScheduler) -> None:
    """Submit a handful of synthetic requests and print an SLO report."""
    log.info("Running SLO scheduler demo …")
    rng = np.random.default_rng(42)

    tier_names = list(scheduler._tiers.keys())
    for i in range(20):
        tier_name = tier_names[i % len(tier_names)]
        prompt = rng.integers(100, 2000, size=rng.integers(32, 256)).tolist()
        req = scheduler.submit(
            prompt_ids=prompt,
            customer_id=f"customer_{i % 5}",
            max_tokens=int(rng.integers(64, 512)),
            tier_name=tier_name,
        )
        if req is None:
            log.info("Request %d rejected", i)

    batch = scheduler.schedule()
    log.info("Scheduled batch of %d requests", len(batch))

    for req in batch:
        latency = float(rng.exponential(scale=500))
        scheduler.complete(req.request_id, latency)

    report = scheduler.get_slo_report()
    log.info(
        "SLO report — total=%d  hit_rate=%.2f  avg_wait=%.1fms",
        report.total_requests,
        report.slo_hit_rate,
        report.avg_wait_ms,
    )
