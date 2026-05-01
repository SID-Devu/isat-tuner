"""Monitoring daemon for real-time inference tracking, anomaly detection, and alerting."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    type: str
    severity: str  # "warning" or "critical"
    message: str
    timestamp: float
    metric_value: float
    threshold: float
    remediation_action: Optional[str] = None


@dataclass
class MonitorSnapshot:
    timestamp: float
    cpu_pct: float
    mem_rss_mb: float
    gpu_util_pct: float
    gpu_temp_c: float
    gpu_vram_mb: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_tps: float
    error_rate: float
    kv_utilization: float
    anomalies: List[Anomaly] = field(default_factory=list)


class MetricsCollector:
    """Thread-safe rolling metrics using a fixed-size circular buffer."""

    def __init__(self, window_size: int = 1000):
        self._window_size = window_size
        self._latencies: List[float] = []
        self._lat_idx = 0
        self._lat_full = False
        self._lat_buf = [0.0] * window_size

        self._throughputs: List[float] = []
        self._tp_idx = 0
        self._tp_full = False
        self._tp_buf = [0.0] * window_size

        self._error_count = 0
        self._total_count = 0
        self._lock = threading.Lock()

    def record_latency(self, ms: float) -> None:
        with self._lock:
            self._lat_buf[self._lat_idx] = ms
            self._lat_idx = (self._lat_idx + 1) % self._window_size
            if self._lat_idx == 0:
                self._lat_full = True
            self._total_count += 1

    def record_throughput(self, tps: float) -> None:
        with self._lock:
            self._tp_buf[self._tp_idx] = tps
            self._tp_idx = (self._tp_idx + 1) % self._window_size
            if self._tp_idx == 0:
                self._tp_full = True

    def record_error(self) -> None:
        with self._lock:
            self._error_count += 1
            self._total_count += 1

    def get_percentiles(self) -> Dict[str, float]:
        with self._lock:
            if self._lat_full:
                data = list(self._lat_buf)
            else:
                data = self._lat_buf[: self._lat_idx]
            if not data:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            data.sort()
            n = len(data)
            return {
                "p50": data[int(n * 0.50)],
                "p95": data[int(min(n * 0.95, n - 1))],
                "p99": data[int(min(n * 0.99, n - 1))],
            }

    def get_throughput_avg(self) -> float:
        with self._lock:
            if self._tp_full:
                data = self._tp_buf
            else:
                data = self._tp_buf[: self._tp_idx]
            return sum(data) / len(data) if data else 0.0

    def get_error_rate(self) -> float:
        with self._lock:
            if self._total_count == 0:
                return 0.0
            return self._error_count / self._total_count

    def get_latest_throughput(self) -> float:
        with self._lock:
            idx = (self._tp_idx - 1) % self._window_size
            if not self._tp_full and self._tp_idx == 0:
                return 0.0
            return self._tp_buf[idx]


class AlertManager:
    """Dispatches anomaly alerts to configured channels (console, file, webhook)."""

    def __init__(self, channels: Optional[Dict[str, Any]] = None):
        self._channels: Dict[str, Any] = channels or {"console": True}
        self._lock = threading.Lock()

    def fire(self, anomaly: Anomaly) -> None:
        with self._lock:
            if self._channels.get("console"):
                self._alert_console(anomaly)
            file_path = self._channels.get("file")
            if file_path:
                self._alert_file(anomaly, file_path)
            webhook_url = self._channels.get("webhook")
            if webhook_url:
                self._alert_webhook(anomaly, webhook_url)

    def _alert_console(self, anomaly: Anomaly) -> None:
        colors = {
            "critical": "\033[91m",
            "warning": "\033[93m",
        }
        reset = "\033[0m"
        color = colors.get(anomaly.severity, "")
        ts = time.strftime("%H:%M:%S", time.localtime(anomaly.timestamp))
        print(
            f"{color}[{anomaly.severity.upper()}] {ts}: "
            f"{anomaly.message}{reset}"
        )

    def _alert_file(self, anomaly: Anomaly, path: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(anomaly.timestamp))
        line = (
            f"[{ts}] [{anomaly.severity.upper()}] {anomaly.type}: "
            f"{anomaly.message} (value={anomaly.metric_value:.2f}, "
            f"threshold={anomaly.threshold:.2f})\n"
        )
        try:
            with open(path, "a") as f:
                f.write(line)
        except OSError as e:
            logger.error("Failed to write alert to %s: %s", path, e)

    def _alert_webhook(self, anomaly: Anomaly, url: str) -> None:
        payload = json.dumps({
            "type": anomaly.type,
            "severity": anomaly.severity,
            "message": anomaly.message,
            "timestamp": anomaly.timestamp,
            "metric_value": anomaly.metric_value,
            "threshold": anomaly.threshold,
            "remediation_action": anomaly.remediation_action,
        }).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status >= 400:
                    logger.warning("Webhook returned status %d", resp.status)
        except (urllib.error.URLError, OSError) as e:
            logger.error("Webhook alert failed: %s", e)


class InferenceMonitor:
    """Attaches to a running inference process or server and continuously
    collects system and inference metrics, detects anomalies, and triggers
    alerts with optional auto-remediation."""

    DEFAULT_CONFIG = {
        "poll_interval_s": 2.0,
        "latency_spike_factor": 2.0,
        "throughput_drop_factor": 0.5,
        "mem_leak_mb_per_hour": 100.0,
        "gpu_temp_critical_c": 85.0,
        "error_burst_rate": 0.05,
        "error_burst_window_s": 60.0,
        "metrics_window_size": 1000,
        "auto_remediate": True,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._collector = MetricsCollector(
            window_size=int(self._config["metrics_window_size"])
        )
        self._alert_manager = AlertManager(
            channels=self._config.get("alert_channels")
        )

        self._pid: Optional[int] = None
        self._port: Optional[int] = None
        self._model_path: Optional[str] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._snapshot_lock = threading.Lock()
        self._snapshot: Optional[MonitorSnapshot] = None
        self._metrics_history: List[MonitorSnapshot] = []
        self._anomaly_history: List[Anomaly] = []
        self._mem_rss_history: List[Tuple[float, float]] = []

    # -- public API -----------------------------------------------------------

    def start(
        self,
        pid: Optional[int] = None,
        model_path: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        if self._running:
            logger.warning("Monitor already running")
            return
        self._pid = pid
        self._model_path = model_path
        self._port = port
        self._start_time = time.time()
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(
            "Monitor started (pid=%s, model_path=%s, port=%s)", pid, model_path, port
        )

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Monitor stopped")

    def get_metrics(self) -> Optional[MonitorSnapshot]:
        with self._snapshot_lock:
            return self._snapshot

    @property
    def anomaly_history(self) -> List[Anomaly]:
        return list(self._anomaly_history)

    @property
    def uptime_seconds(self) -> float:
        if self._start_time == 0:
            return 0.0
        return time.time() - self._start_time

    # -- main loop ------------------------------------------------------------

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                sys_metrics = self._collect_system_metrics()
                inf_metrics = self._collect_inference_metrics()
                anomalies = self._detect_anomalies(self._metrics_history)

                snapshot = MonitorSnapshot(
                    timestamp=time.time(),
                    anomalies=anomalies,
                    **sys_metrics,
                    **inf_metrics,
                )

                with self._snapshot_lock:
                    self._snapshot = snapshot
                self._metrics_history.append(snapshot)

                if len(self._metrics_history) > 3600:
                    self._metrics_history = self._metrics_history[-1800:]

                for anomaly in anomalies:
                    self._anomaly_history.append(anomaly)
                    self._alert_manager.fire(anomaly)
                    if self._config["auto_remediate"]:
                        self._remediate(anomaly)

            except Exception:
                logger.exception("Error in monitor loop")
            time.sleep(self._config["poll_interval_s"])

    # -- system metrics -------------------------------------------------------

    def _collect_system_metrics(self) -> Dict[str, float]:
        cpu_pct = self._read_cpu_utilization()
        mem_rss_mb = self._read_memory_rss()
        gpu_util, gpu_temp, gpu_vram = self._read_gpu_metrics()

        self._mem_rss_history.append((time.time(), mem_rss_mb))
        if len(self._mem_rss_history) > 7200:
            self._mem_rss_history = self._mem_rss_history[-3600:]

        return {
            "cpu_pct": cpu_pct,
            "mem_rss_mb": mem_rss_mb,
            "gpu_util_pct": gpu_util,
            "gpu_temp_c": gpu_temp,
            "gpu_vram_mb": gpu_vram,
        }

    def _read_cpu_utilization(self) -> float:
        if self._pid:
            stat_path = f"/proc/{self._pid}/stat"
            try:
                with open(stat_path) as f:
                    parts = f.read().split()
                utime = int(parts[13])
                stime = int(parts[14])
                total = utime + stime
                clk_tck = os.sysconf("SC_CLK_TCK")
                uptime = time.time() - self._start_time
                if uptime > 0:
                    return min((total / clk_tck) / uptime * 100.0, 100.0)
            except (OSError, IndexError, ValueError):
                pass
        try:
            with open("/proc/loadavg") as f:
                load_1m = float(f.read().split()[0])
            ncpu = os.cpu_count() or 1
            return min(load_1m / ncpu * 100.0, 100.0)
        except (OSError, ValueError):
            return 0.0

    def _read_memory_rss(self) -> float:
        if self._pid:
            status_path = f"/proc/{self._pid}/status"
            try:
                with open(status_path) as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024.0
            except (OSError, ValueError, IndexError):
                pass
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            total = used = 0.0
            for line in lines:
                parts = line.split()
                if parts[0] == "MemTotal:":
                    total = float(parts[1]) / 1024.0
                elif parts[0] == "MemAvailable:":
                    used = total - float(parts[1]) / 1024.0
                    break
            return used
        except (OSError, ValueError, IndexError):
            return 0.0

    def _read_gpu_metrics(self) -> Tuple[float, float, float]:
        for cmd, parser in [
            (self._nvidia_smi_cmd, self._parse_nvidia_smi),
            (self._rocm_smi_cmd, self._parse_rocm_smi),
        ]:
            args = cmd()
            if not args:
                continue
            try:
                result = subprocess.run(
                    args, capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return parser(result.stdout)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _nvidia_smi_cmd() -> Optional[List[str]]:
        if shutil.which("nvidia-smi"):
            return [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ]
        return None

    @staticmethod
    def _parse_nvidia_smi(output: str) -> Tuple[float, float, float]:
        first_line = output.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        return float(parts[0]), float(parts[1]), float(parts[2])

    @staticmethod
    def _rocm_smi_cmd() -> Optional[List[str]]:
        if shutil.which("rocm-smi"):
            return ["rocm-smi", "--showuse", "--showtemp", "--showmemuse", "--csv"]
        return None

    @staticmethod
    def _parse_rocm_smi(output: str) -> Tuple[float, float, float]:
        lines = output.strip().splitlines()
        util = temp = vram = 0.0
        for line in lines:
            nums = re.findall(r"[\d.]+", line)
            lower = line.lower()
            if "gpu use" in lower and nums:
                util = float(nums[0])
            elif "temperature" in lower and nums:
                temp = float(nums[0])
            elif "vram" in lower and nums:
                vram = float(nums[0])
        return util, temp, vram

    # -- inference metrics ----------------------------------------------------

    def _collect_inference_metrics(self) -> Dict[str, float]:
        if self._port:
            return self._poll_metrics_endpoint()

        percentiles = self._collector.get_percentiles()
        return {
            "latency_p50_ms": percentiles["p50"],
            "latency_p95_ms": percentiles["p95"],
            "latency_p99_ms": percentiles["p99"],
            "throughput_tps": self._collector.get_latest_throughput(),
            "error_rate": self._collector.get_error_rate(),
            "kv_utilization": 0.0,
        }

    def _poll_metrics_endpoint(self) -> Dict[str, float]:
        url = f"http://127.0.0.1:{self._port}/metrics"
        defaults = {
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "throughput_tps": 0.0,
            "error_rate": 0.0,
            "kv_utilization": 0.0,
        }
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            for key in defaults:
                if key in data:
                    defaults[key] = float(data[key])
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError) as e:
            logger.debug("Failed to poll metrics endpoint: %s", e)
        return defaults

    # -- anomaly detection ----------------------------------------------------

    def _detect_anomalies(self, history: List[MonitorSnapshot]) -> List[Anomaly]:
        if len(history) < 5:
            return []

        anomalies: List[Anomaly] = []
        now = time.time()
        recent = history[-30:]

        avg_p99 = sum(s.latency_p99_ms for s in recent) / len(recent)
        current_p99 = recent[-1].latency_p99_ms
        factor = self._config["latency_spike_factor"]
        if avg_p99 > 0 and current_p99 > factor * avg_p99:
            anomalies.append(Anomaly(
                type="latency_spike",
                severity="warning",
                message=(
                    f"Latency spike P99={current_p99:.1f}ms "
                    f"({current_p99 / avg_p99:.1f}x avg)"
                ),
                timestamp=now,
                metric_value=current_p99,
                threshold=factor * avg_p99,
                remediation_action="investigate_slow_requests",
            ))

        avg_tps = sum(s.throughput_tps for s in recent) / len(recent)
        current_tps = recent[-1].throughput_tps
        drop_factor = self._config["throughput_drop_factor"]
        if avg_tps > 0 and current_tps < drop_factor * avg_tps:
            anomalies.append(Anomaly(
                type="throughput_drop",
                severity="warning",
                message=(
                    f"Throughput dropped to {current_tps:.1f} tok/s "
                    f"({current_tps / avg_tps:.2f}x avg)"
                ),
                timestamp=now,
                metric_value=current_tps,
                threshold=drop_factor * avg_tps,
            ))

        if len(self._mem_rss_history) >= 2:
            t0, m0 = self._mem_rss_history[0]
            t1, m1 = self._mem_rss_history[-1]
            elapsed_h = (t1 - t0) / 3600.0
            if elapsed_h > 0:
                growth_rate = (m1 - m0) / elapsed_h
                threshold = self._config["mem_leak_mb_per_hour"]
                if growth_rate > threshold:
                    anomalies.append(Anomaly(
                        type="memory_leak",
                        severity="warning",
                        message=f"Memory growth +{growth_rate:.0f}MB/hr",
                        timestamp=now,
                        metric_value=growth_rate,
                        threshold=threshold,
                        remediation_action="trigger_gc",
                    ))

        current = recent[-1]
        temp_thresh = self._config["gpu_temp_critical_c"]
        if current.gpu_temp_c > temp_thresh and current_p99 > avg_p99 * 1.2:
            anomalies.append(Anomaly(
                type="thermal_throttle",
                severity="critical",
                message=(
                    f"GPU thermal throttle: {current.gpu_temp_c:.0f}°C "
                    f"with latency increase"
                ),
                timestamp=now,
                metric_value=current.gpu_temp_c,
                threshold=temp_thresh,
                remediation_action="reduce_batch_size",
            ))

        err_thresh = self._config["error_burst_rate"]
        if current.error_rate > err_thresh:
            anomalies.append(Anomaly(
                type="error_burst",
                severity="critical",
                message=(
                    f"Error rate {current.error_rate * 100:.1f}% "
                    f"exceeds {err_thresh * 100:.1f}% threshold"
                ),
                timestamp=now,
                metric_value=current.error_rate,
                threshold=err_thresh,
                remediation_action="alert_and_restart",
            ))

        return anomalies

    # -- remediation ----------------------------------------------------------

    def _remediate(self, anomaly: Anomaly) -> None:
        action = anomaly.remediation_action
        if not action:
            return

        if action == "trigger_gc":
            logger.warning("Triggering GC due to memory leak detection")
            gc.collect()

        elif action == "reduce_batch_size":
            logger.warning(
                "Thermal throttle detected — consider reducing batch size or "
                "adding cooldown period"
            )

        elif action == "alert_and_restart":
            logger.critical(
                "Error burst detected — manual intervention may be required "
                "(pid=%s)", self._pid
            )

        else:
            logger.info("No handler for remediation action: %s", action)

    # -- external recording API (for self-monitor mode) -----------------------

    @property
    def collector(self) -> MetricsCollector:
        return self._collector
