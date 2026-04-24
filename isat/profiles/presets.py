"""Configuration presets for common deployment scenarios.

Presets constrain the search space to configurations that
make sense for a given deployment target, dramatically reducing
tuning time while ensuring relevant results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TuneProfile:
    name: str
    description: str
    warmup: int = 3
    runs: int = 5
    cooldown: float = 60.0
    max_configs: int = 0
    skip_precision: bool = False
    skip_graph: bool = False
    skip_batch: bool = False
    skip_threading: bool = False
    priority_objective: str = "latency_ms"
    env_overrides: dict[str, str] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)


PROFILES: dict[str, TuneProfile] = {
    "edge": TuneProfile(
        name="edge",
        description="Edge deployment -- minimize latency at batch=1, constrained memory",
        warmup=3,
        runs=10,
        cooldown=30,
        skip_batch=True,
        skip_threading=True,
        priority_objective="latency_ms",
        constraints={"max_memory_mb": 4096, "max_power_w": 25},
    ),
    "cloud": TuneProfile(
        name="cloud",
        description="Cloud deployment -- maximize throughput with batching",
        warmup=5,
        runs=20,
        cooldown=120,
        priority_objective="throughput_fps",
        constraints={"max_p99_ms": 100},
    ),
    "latency": TuneProfile(
        name="latency",
        description="Ultra-low latency -- minimize P99, sacrifice throughput if needed",
        warmup=5,
        runs=30,
        cooldown=60,
        skip_batch=True,
        priority_objective="p99_ms",
    ),
    "throughput": TuneProfile(
        name="throughput",
        description="Maximum throughput -- find optimal batch size and parallelism",
        warmup=3,
        runs=15,
        cooldown=120,
        priority_objective="throughput_fps",
    ),
    "power": TuneProfile(
        name="power",
        description="Power-efficient -- best perf/watt for battery or thermal-constrained deployments",
        warmup=3,
        runs=10,
        cooldown=60,
        priority_objective="latency_ms",
        constraints={"max_power_w": 15, "max_temp_c": 75},
    ),
    "quick": TuneProfile(
        name="quick",
        description="Quick scan -- fast exploration with minimal runs",
        warmup=1,
        runs=3,
        cooldown=15,
        max_configs=8,
        skip_precision=True,
        skip_graph=True,
        priority_objective="latency_ms",
    ),
    "exhaustive": TuneProfile(
        name="exhaustive",
        description="Exhaustive search -- test every combination, no shortcuts",
        warmup=5,
        runs=50,
        cooldown=180,
        priority_objective="latency_ms",
    ),
    "apu": TuneProfile(
        name="apu",
        description="APU-optimized -- focus on unified memory and XNACK settings",
        warmup=3,
        runs=10,
        cooldown=60,
        priority_objective="latency_ms",
        env_overrides={"HSA_XNACK": "1"},
        constraints={"max_power_w": 30},
    ),
}


def get_profile(name: str) -> TuneProfile:
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]


def list_profiles() -> list[dict[str, str]]:
    return [
        {"name": p.name, "description": p.description}
        for p in PROFILES.values()
    ]
