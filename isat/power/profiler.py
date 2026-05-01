"""Power efficiency profiler.

Measures:
  - Average GPU power draw during inference (watts)
  - Energy per inference (millijoules)
  - Performance per watt (inferences/second/watt)
  - TDP utilization percentage
  - Idle vs active power delta
  - Power efficiency comparison between configs

Critical for edge/embedded deployment where power budget is limited.
"""

from __future__ import annotations

import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.power")


@dataclass
class PowerProfile:
    model_path: str
    provider: str
    idle_power_w: float
    active_power_w: float
    peak_power_w: float
    tdp_w: float
    tdp_utilization_pct: float
    energy_per_inference_mj: float
    perf_per_watt: float
    latency_ms: float
    throughput_fps: float
    power_samples: list[float] = field(default_factory=list)
    temperature_samples: list[float] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  Idle power         : {self.idle_power_w:.1f} W",
            f"  Active power (avg) : {self.active_power_w:.1f} W",
            f"  Peak power         : {self.peak_power_w:.1f} W",
            f"  TDP                : {self.tdp_w:.0f} W ({self.tdp_utilization_pct:.0f}% utilized)",
            f"  Energy/inference   : {self.energy_per_inference_mj:.3f} mJ",
            f"  Perf/watt          : {self.perf_per_watt:.1f} infer/s/W",
            f"  Latency            : {self.latency_ms:.2f} ms",
            f"  Throughput         : {self.throughput_fps:.1f} FPS",
        ]
        return "\n".join(lines)


class PowerProfiler:
    """Profile power efficiency of inference."""

    def __init__(
        self,
        model_path: str,
        provider: str = "MIGraphXExecutionProvider",
        warmup: int = 5,
        runs: int = 50,
        power_sample_interval: float = 0.1,
    ):
        self.model_path = model_path
        self.provider = provider
        self.warmup = warmup
        self.runs = runs
        self.sample_interval = power_sample_interval

    def profile(self) -> PowerProfile:
        import onnxruntime as ort

        idle_power = self._read_power()
        idle_temp = self._read_temp()
        time.sleep(1.0)

        session = ort.InferenceSession(
            self.model_path,
            providers=ort_providers(self.provider),
        )
        feed = self._build_feed(session)

        for _ in range(self.warmup):
            session.run(None, feed)

        power_samples: list[float] = []
        temp_samples: list[float] = []
        stop_event = threading.Event()

        def _sample():
            while not stop_event.is_set():
                p = self._read_power()
                t = self._read_temp()
                if p is not None:
                    power_samples.append(p)
                if t is not None:
                    temp_samples.append(t)
                time.sleep(self.sample_interval)

        sampler = threading.Thread(target=_sample, daemon=True)
        sampler.start()

        latencies = []
        for _ in range(self.runs):
            t0 = time.perf_counter()
            session.run(None, feed)
            latencies.append((time.perf_counter() - t0) * 1000)

        stop_event.set()
        sampler.join(timeout=2)

        lats = np.array(latencies)
        mean_ms = float(np.mean(lats))
        throughput = 1000.0 / mean_ms

        active_power = np.mean(power_samples) if power_samples else idle_power or 0
        peak_power = max(power_samples) if power_samples else active_power

        tdp = self._read_tdp()
        tdp_util = (active_power / tdp * 100) if tdp > 0 else 0

        energy_mj = (active_power * mean_ms / 1000.0) * 1000.0
        perf_per_watt = throughput / active_power if active_power > 0 else 0

        return PowerProfile(
            model_path=self.model_path,
            provider=self.provider,
            idle_power_w=idle_power or 0,
            active_power_w=float(active_power),
            peak_power_w=float(peak_power),
            tdp_w=tdp,
            tdp_utilization_pct=tdp_util,
            energy_per_inference_mj=energy_mj,
            perf_per_watt=perf_per_watt,
            latency_ms=mean_ms,
            throughput_fps=throughput,
            power_samples=power_samples,
            temperature_samples=temp_samples,
        )

    @staticmethod
    def _find_hwmon() -> Optional[Path]:
        """Locate the amdgpu hwmon directory dynamically."""
        from isat.utils.sysfs import _gpu_card_path
        card = _gpu_card_path()
        if card:
            hwmon_dir = card / "device" / "hwmon"
            if hwmon_dir.exists():
                for hwmon in sorted(hwmon_dir.iterdir()):
                    if (hwmon / "name").exists():
                        name = hwmon / "name"
                        if "amdgpu" in name.read_text().strip():
                            return hwmon
                for hwmon in sorted(hwmon_dir.iterdir()):
                    return hwmon
        for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            try:
                if "amdgpu" in (hwmon / "name").read_text().strip():
                    return hwmon
            except OSError:
                continue
        return None

    def _read_power(self) -> Optional[float]:
        try:
            hwmon = self._find_hwmon()
            if hwmon:
                for fname in ("power1_average", "power1_input"):
                    f = hwmon / fname
                    if f.exists():
                        return int(f.read_text().strip()) / 1_000_000
        except (OSError, ValueError):
            pass
        return None

    def _read_temp(self) -> Optional[float]:
        try:
            hwmon = self._find_hwmon()
            if hwmon:
                temp_file = hwmon / "temp1_input"
                if temp_file.exists():
                    return int(temp_file.read_text().strip()) / 1000
        except (OSError, ValueError):
            pass
        return None

    def _read_tdp(self) -> float:
        try:
            hwmon = self._find_hwmon()
            if hwmon:
                cap_file = hwmon / "power1_cap"
                if cap_file.exists():
                    return int(cap_file.read_text().strip()) / 1_000_000
        except (OSError, ValueError):
            pass
        return 150.0

    def _build_feed(self, session) -> dict:
        feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            elif "float16" in inp.type.lower():
                feed[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        return feed
