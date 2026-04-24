"""Thermal throttle detection during inference benchmarks.

Monitors GPU temperature and clock speeds during inference to detect
thermal throttling, which silently degrades benchmark accuracy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("isat.thermal")


@dataclass
class ThermalSample:
    timestamp_s: float
    temp_c: float
    sclk_mhz: int
    inference_idx: int


@dataclass
class ThrottleEvent:
    start_idx: int
    end_idx: int
    start_temp_c: float
    peak_temp_c: float
    clock_drop_mhz: int
    duration_s: float


@dataclass
class ThermalProfile:
    model_path: str
    samples: list[ThermalSample] = field(default_factory=list)
    throttle_events: list[ThrottleEvent] = field(default_factory=list)
    min_temp_c: float = 0
    max_temp_c: float = 0
    mean_temp_c: float = 0
    min_clock_mhz: int = 0
    max_clock_mhz: int = 0
    throttled: bool = False
    throttle_impact_pct: float = 0
    inference_count: int = 0
    duration_s: float = 0

    def summary(self) -> str:
        lines = [
            f"  Model          : {self.model_path}",
            f"  Duration       : {self.duration_s:.1f}s ({self.inference_count} inferences)",
            f"  Temperature    : {self.min_temp_c:.0f}C -> {self.max_temp_c:.0f}C "
            f"(mean {self.mean_temp_c:.0f}C)",
            f"  Clock speed    : {self.min_clock_mhz} - {self.max_clock_mhz} MHz",
            f"  Throttled      : {'YES' if self.throttled else 'NO'}",
        ]
        if self.throttled:
            lines.append(f"  Throttle events: {len(self.throttle_events)}")
            lines.append(f"  Est. impact    : {self.throttle_impact_pct:.1f}% slower")
            for i, ev in enumerate(self.throttle_events[:5]):
                lines.append(
                    f"    Event {i+1}: iter {ev.start_idx}-{ev.end_idx}, "
                    f"peak {ev.peak_temp_c:.0f}C, clock drop {ev.clock_drop_mhz} MHz"
                )
        else:
            lines.append("  No thermal throttling detected -- results are reliable")

        return "\n".join(lines)


class ThermalMonitor:
    """Monitor GPU thermals during inference and detect throttling."""

    def __init__(self, model_path: str, provider: str = "CPUExecutionProvider",
                 runs: int = 100, sample_interval_ms: int = 100):
        self.model_path = model_path
        self.provider = provider
        self.runs = runs
        self.sample_interval_s = sample_interval_ms / 1000
        self._samples: list[ThermalSample] = []
        self._stop = threading.Event()
        self._inference_idx = 0

    def monitor(self) -> ThermalProfile:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(
            self.model_path,
            providers=[self.provider, "CPUExecutionProvider"],
        )
        feed = self._build_feed(session)

        for _ in range(3):
            session.run(None, feed)

        self._stop.clear()
        self._inference_idx = 0
        sampler = threading.Thread(target=self._sample_loop, daemon=True)
        sampler.start()

        t0 = time.perf_counter()
        latencies = []
        for i in range(self.runs):
            self._inference_idx = i
            ts = time.perf_counter()
            session.run(None, feed)
            latencies.append((time.perf_counter() - ts) * 1000)

        duration = time.perf_counter() - t0
        self._stop.set()
        sampler.join(timeout=2)

        if not self._samples:
            return ThermalProfile(model_path=self.model_path, duration_s=duration,
                                 inference_count=self.runs)

        temps = [s.temp_c for s in self._samples if s.temp_c > 0]
        clocks = [s.sclk_mhz for s in self._samples if s.sclk_mhz > 0]

        throttle_events = self._detect_throttling(clocks)
        throttled = len(throttle_events) > 0

        impact = 0.0
        if throttled and clocks:
            max_clk = max(clocks)
            min_clk = min(clocks)
            if max_clk > 0:
                impact = (max_clk - min_clk) / max_clk * 100

        return ThermalProfile(
            model_path=self.model_path,
            samples=self._samples,
            throttle_events=throttle_events,
            min_temp_c=min(temps) if temps else 0,
            max_temp_c=max(temps) if temps else 0,
            mean_temp_c=float(np.mean(temps)) if temps else 0,
            min_clock_mhz=min(clocks) if clocks else 0,
            max_clock_mhz=max(clocks) if clocks else 0,
            throttled=throttled,
            throttle_impact_pct=impact,
            inference_count=self.runs,
            duration_s=duration,
        )

    def _sample_loop(self):
        from isat.utils.sysfs import gpu_temp_edge, gpu_sclk_mhz
        t0 = time.perf_counter()
        while not self._stop.is_set():
            temp = gpu_temp_edge() or 0
            clk = gpu_sclk_mhz() or 0
            self._samples.append(ThermalSample(
                timestamp_s=time.perf_counter() - t0,
                temp_c=temp,
                sclk_mhz=clk,
                inference_idx=self._inference_idx,
            ))
            self._stop.wait(self.sample_interval_s)

    def _detect_throttling(self, clocks: list[int]) -> list[ThrottleEvent]:
        if len(clocks) < 5:
            return []

        max_clk = max(clocks)
        threshold = max_clk * 0.95
        events = []
        in_throttle = False
        start_idx = 0

        for i, clk in enumerate(clocks):
            if clk < threshold and not in_throttle:
                in_throttle = True
                start_idx = i
            elif clk >= threshold and in_throttle:
                in_throttle = False
                samples_range = self._samples[start_idx:i] if i <= len(self._samples) else []
                peak_temp = max((s.temp_c for s in samples_range), default=0)
                events.append(ThrottleEvent(
                    start_idx=start_idx, end_idx=i,
                    start_temp_c=self._samples[start_idx].temp_c if start_idx < len(self._samples) else 0,
                    peak_temp_c=peak_temp,
                    clock_drop_mhz=max_clk - min(clocks[start_idx:i]),
                    duration_s=(i - start_idx) * self.sample_interval_s,
                ))

        return events

    def _build_feed(self, session) -> dict:
        import numpy as np
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
