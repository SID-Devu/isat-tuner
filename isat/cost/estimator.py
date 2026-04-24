"""Cloud inference cost estimator.

Given a latency benchmark and a target cloud GPU, estimates:
  - Cost per inference
  - Cost per 1M inferences
  - Monthly cost at a given QPS
  - Break-even vs. CPU
  - ROI of optimization (before vs. after tuning)

Pricing data is based on public cloud list prices and can be overridden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("isat.cost")

# On-demand hourly GPU prices (USD) as of early 2025
GPU_PRICING: dict[str, dict] = {
    "a100_40gb": {"hourly": 3.67, "provider": "AWS", "instance": "p4d.24xlarge/8", "gpu_per_instance": 8},
    "a100_80gb": {"hourly": 4.10, "provider": "AWS", "instance": "p4de.24xlarge/8", "gpu_per_instance": 8},
    "a10g": {"hourly": 1.21, "provider": "AWS", "instance": "g5.xlarge", "gpu_per_instance": 1},
    "t4": {"hourly": 0.53, "provider": "AWS", "instance": "g4dn.xlarge", "gpu_per_instance": 1},
    "l4": {"hourly": 0.81, "provider": "GCP", "instance": "g2-standard-4", "gpu_per_instance": 1},
    "h100_80gb": {"hourly": 8.10, "provider": "AWS", "instance": "p5.48xlarge/8", "gpu_per_instance": 8},
    "mi300x": {"hourly": 6.50, "provider": "Azure", "instance": "ND-MI300X-v5", "gpu_per_instance": 8},
    "mi250": {"hourly": 3.80, "provider": "Azure", "instance": "ND-A100-v4 equiv.", "gpu_per_instance": 1},
    "cpu_c5_4xl": {"hourly": 0.68, "provider": "AWS", "instance": "c5.4xlarge", "gpu_per_instance": 0},
    "cpu_c6i_4xl": {"hourly": 0.68, "provider": "AWS", "instance": "c6i.4xlarge", "gpu_per_instance": 0},
}


@dataclass
class CostEstimate:
    gpu_type: str
    hourly_rate: float
    latency_ms: float
    throughput_rps: float
    cost_per_inference: float
    cost_per_1m: float
    monthly_cost_at_qps: dict[int, float] = field(default_factory=dict)
    utilization_pct: float = 0.0
    breakeven_vs_cpu: str = ""

    def summary(self) -> str:
        lines = [
            f"  GPU type           : {self.gpu_type}",
            f"  Hourly rate        : ${self.hourly_rate:.2f}/hr",
            f"  Latency            : {self.latency_ms:.2f} ms",
            f"  Throughput         : {self.throughput_rps:.1f} req/s",
            f"  Cost per inference : ${self.cost_per_inference:.8f}",
            f"  Cost per 1M infer  : ${self.cost_per_1m:.2f}",
            f"  GPU utilization    : {self.utilization_pct:.1f}%",
        ]
        if self.monthly_cost_at_qps:
            lines.append(f"\n  Monthly cost projections:")
            for qps, cost in sorted(self.monthly_cost_at_qps.items()):
                lines.append(f"    {qps:>5} QPS -> ${cost:>10,.2f}/month")
        if self.breakeven_vs_cpu:
            lines.append(f"\n  {self.breakeven_vs_cpu}")
        return "\n".join(lines)


class CostEstimator:
    """Estimate inference cost on cloud GPUs."""

    def __init__(self, custom_pricing: dict[str, dict] | None = None):
        self.pricing = dict(GPU_PRICING)
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def estimate(
        self,
        latency_ms: float,
        gpu_type: str = "a10g",
        batch_size: int = 1,
        target_qps_list: list[int] | None = None,
    ) -> CostEstimate:
        if gpu_type not in self.pricing:
            available = ", ".join(sorted(self.pricing.keys()))
            raise ValueError(f"Unknown GPU: {gpu_type}. Available: {available}")

        info = self.pricing[gpu_type]
        hourly = info["hourly"]

        throughput = (1000.0 / latency_ms) * batch_size if latency_ms > 0 else 0
        cost_per_second = hourly / 3600
        cost_per_inference = cost_per_second / throughput if throughput > 0 else 0
        cost_per_1m = cost_per_inference * 1_000_000

        qps_list = target_qps_list or [1, 10, 100, 1000]
        monthly: dict[int, float] = {}
        for qps in qps_list:
            gpus_needed = max(1, qps / throughput) if throughput > 0 else 1
            monthly_hours = 730
            monthly[qps] = gpus_needed * hourly * monthly_hours

        utilization = min(100.0, (1.0 / throughput) * 100) if throughput > 0 else 0

        cpu_info = self.pricing.get("cpu_c6i_4xl", self.pricing.get("cpu_c5_4xl"))
        breakeven = ""
        if cpu_info and latency_ms > 0:
            cpu_hourly = cpu_info["hourly"]
            gpu_factor = hourly / cpu_hourly
            breakeven = (
                f"GPU is {gpu_factor:.1f}x more expensive per hour than CPU. "
                f"GPU must be >{gpu_factor:.0f}x faster to break even on cost/inference."
            )

        return CostEstimate(
            gpu_type=gpu_type,
            hourly_rate=hourly,
            latency_ms=latency_ms,
            throughput_rps=throughput,
            cost_per_inference=cost_per_inference,
            cost_per_1m=cost_per_1m,
            monthly_cost_at_qps=monthly,
            utilization_pct=utilization,
            breakeven_vs_cpu=breakeven,
        )

    def compare_optimization(
        self,
        before_latency_ms: float,
        after_latency_ms: float,
        gpu_type: str = "a10g",
        monthly_qps: int = 100,
    ) -> dict:
        before = self.estimate(before_latency_ms, gpu_type, target_qps_list=[monthly_qps])
        after = self.estimate(after_latency_ms, gpu_type, target_qps_list=[monthly_qps])

        savings = before.monthly_cost_at_qps[monthly_qps] - after.monthly_cost_at_qps[monthly_qps]
        savings_pct = (savings / before.monthly_cost_at_qps[monthly_qps] * 100
                       if before.monthly_cost_at_qps[monthly_qps] > 0 else 0)

        return {
            "before_latency_ms": before_latency_ms,
            "after_latency_ms": after_latency_ms,
            "speedup": before_latency_ms / after_latency_ms if after_latency_ms > 0 else 0,
            "before_monthly": before.monthly_cost_at_qps[monthly_qps],
            "after_monthly": after.monthly_cost_at_qps[monthly_qps],
            "monthly_savings": savings,
            "annual_savings": savings * 12,
            "savings_pct": savings_pct,
        }

    def list_gpus(self) -> list[dict]:
        return [
            {"name": k, "hourly": v["hourly"], "provider": v["provider"], "instance": v["instance"]}
            for k, v in sorted(self.pricing.items())
        ]
