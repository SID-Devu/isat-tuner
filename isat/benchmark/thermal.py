"""Thermal guard -- monitor GPU temperature and enforce cooldown between runs."""

from __future__ import annotations

import logging
import time
from typing import Optional

from isat.utils.sysfs import gpu_power_watts, gpu_temp_edge, gpu_gtt_used_mb, gpu_vram_used_mb

log = logging.getLogger("isat.thermal")


class ThermalGuard:
    """Monitor GPU thermals and enforce cooldown policies."""

    def __init__(
        self,
        cooldown_seconds: float = 60.0,
        max_temp_c: float = 85.0,
        target_temp_c: float = 55.0,
        poll_interval: float = 2.0,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.max_temp_c = max_temp_c
        self.target_temp_c = target_temp_c
        self.poll_interval = poll_interval
        self._peak_temp: float = 0.0
        self._peak_power: float = 0.0
        self._peak_vram: float = 0.0
        self._peak_gtt: float = 0.0

    @property
    def peak_temp(self) -> float:
        return self._peak_temp

    @property
    def peak_power(self) -> float:
        return self._peak_power

    @property
    def peak_vram(self) -> float:
        return self._peak_vram

    @property
    def peak_gtt(self) -> float:
        return self._peak_gtt

    def sample(self) -> dict:
        """Take a single thermal/power/memory snapshot."""
        temp = gpu_temp_edge() or 0.0
        power = gpu_power_watts() or 0.0
        vram = gpu_vram_used_mb() or 0.0
        gtt = gpu_gtt_used_mb() or 0.0

        self._peak_temp = max(self._peak_temp, temp)
        self._peak_power = max(self._peak_power, power)
        self._peak_vram = max(self._peak_vram, vram)
        self._peak_gtt = max(self._peak_gtt, gtt)

        return {
            "temp_c": temp,
            "power_w": power,
            "vram_mb": vram,
            "gtt_mb": gtt,
            "timestamp": time.time(),
        }

    def reset_peaks(self) -> None:
        self._peak_temp = 0.0
        self._peak_power = 0.0
        self._peak_vram = 0.0
        self._peak_gtt = 0.0

    def wait_cooldown(self) -> None:
        """Wait for cooldown: fixed time + thermal stabilization."""
        if self.cooldown_seconds <= 0:
            return

        log.info("Cooldown: waiting %.0fs", self.cooldown_seconds)
        start = time.time()

        while time.time() - start < self.cooldown_seconds:
            temp = gpu_temp_edge()
            elapsed = time.time() - start
            if temp is not None:
                log.debug("  cooldown %.0fs / %.0fs  temp=%.1f C", elapsed, self.cooldown_seconds, temp)
            time.sleep(self.poll_interval)

        temp = gpu_temp_edge()
        if temp is not None and temp > self.target_temp_c:
            log.info("Post-cooldown temp %.1f C > target %.1f C, waiting for thermal settle", temp, self.target_temp_c)
            extra_start = time.time()
            while time.time() - extra_start < 120:
                temp = gpu_temp_edge()
                if temp is not None and temp <= self.target_temp_c:
                    log.info("Thermal settled at %.1f C", temp)
                    break
                time.sleep(self.poll_interval)

    def check_thermal_throttle(self) -> bool:
        """Return True if the GPU is currently above the max safe temp."""
        temp = gpu_temp_edge()
        if temp is not None and temp >= self.max_temp_c:
            log.warning("GPU temp %.1f C >= %.1f C max -- thermal throttle risk!", temp, self.max_temp_c)
            return True
        return False
