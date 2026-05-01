"""Terminal-based live dashboard for the inference monitor using only stdlib."""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from typing import List, Optional, Tuple

from .daemon import Anomaly, InferenceMonitor, MonitorSnapshot

logger = logging.getLogger(__name__)

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"
_CLEAR_SCREEN = "\033[2J\033[H"

_BAR_CHARS = " ░▒▓█"
_FULL_BLOCK = "█"
_LIGHT_SHADE = "░"


class MonitorDashboard:
    """Renders a live ASCII dashboard driven by an InferenceMonitor."""

    def __init__(self, monitor: InferenceMonitor):
        self._monitor = monitor
        self._running = False
        self._cols = 60
        self._rows = 24
        self._latency_history: List[float] = []
        self._max_history = 120

    def run(self) -> None:
        self._running = True
        prev_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum, frame):
            self._running = False

        signal.signal(signal.SIGINT, _handle_sigint)

        try:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
            while self._running:
                self._update_terminal_size()
                self._render()
                time.sleep(1.0)
        finally:
            sys.stdout.write("\033[?25h\n")
            sys.stdout.flush()
            signal.signal(signal.SIGINT, prev_handler)

    def _update_terminal_size(self) -> None:
        try:
            size = os.get_terminal_size()
            self._cols = size.columns
            self._rows = size.lines
        except OSError:
            pass

    def _render(self) -> None:
        snapshot = self._monitor.get_metrics()
        buf: List[str] = []

        if snapshot:
            self._latency_history.append(snapshot.latency_p99_ms)
            if len(self._latency_history) > self._max_history:
                self._latency_history = self._latency_history[-self._max_history :]

        uptime = self._format_uptime(self._monitor.uptime_seconds)
        w = max(self._cols - 2, 58)

        header = f"{_BOLD}{_CYAN}ISAT Live Monitor{_RESET}"
        uptime_tag = f"{_DIM}[uptime: {uptime}]{_RESET}"
        pad = w - 17 - len(uptime) - 11
        buf.append(f" {header}{' ' * max(pad, 1)}{uptime_tag}")
        buf.append(f" {'═' * w}")

        if not snapshot:
            buf.append(f" {_DIM}Waiting for metrics…{_RESET}")
        else:
            self._render_system(buf, snapshot, w)
            buf.append(f" {'─' * w}")
            self._render_inference(buf, snapshot, w)
            buf.append(f" {'─' * w}")
            self._render_latency_trend(buf, w)
            buf.append(f" {'─' * w}")
            self._render_alerts(buf, w)

        out = _CLEAR_SCREEN + "\n".join(buf) + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()

    # -- sections -------------------------------------------------------------

    def _render_system(
        self, buf: List[str], snap: MonitorSnapshot, w: int
    ) -> None:
        cpu_bar = self._draw_bar(snap.cpu_pct, 100, width=14)
        mem_str = self._format_size(snap.mem_rss_mb)
        buf.append(
            f" {_BOLD}SYSTEM{_RESET}          "
            f"CPU: {cpu_bar} {snap.cpu_pct:4.0f}%    "
            f"Mem: {mem_str}"
        )
        gpu_bar = self._draw_bar(snap.gpu_util_pct, 100, width=14)
        vram_str = f"{snap.gpu_vram_mb / 1024:.1f}" if snap.gpu_vram_mb else "0.0"
        buf.append(
            f"                 "
            f"GPU: {gpu_bar} {snap.gpu_util_pct:4.0f}%    "
            f"VRAM: {vram_str} GB"
        )
        buf.append(
            f"                 "
            f"Temp: {snap.gpu_temp_c:.0f}°C"
        )

    def _render_inference(
        self, buf: List[str], snap: MonitorSnapshot, w: int
    ) -> None:
        buf.append(
            f" {_BOLD}INFERENCE{_RESET}       "
            f"Latency P50: {snap.latency_p50_ms:6.1f}ms  "
            f"P95: {snap.latency_p95_ms:6.1f}ms  "
            f"P99: {snap.latency_p99_ms:6.1f}ms"
        )
        buf.append(
            f"                 "
            f"Throughput: {snap.throughput_tps:,.0f} tok/s   "
            f"Errors: {snap.error_rate * 100:.2f}%"
        )
        if snap.kv_utilization > 0:
            kv_pct = snap.kv_utilization * 100
            buf.append(
                f"                 "
                f"KV Cache: {kv_pct:.0f}% utilized"
            )

    def _render_latency_trend(self, buf: List[str], w: int) -> None:
        buf.append(f" {_BOLD}LATENCY TREND{_RESET}   [last 60s]")
        chart_width = min(w - 8, 50)
        chart_height = 5
        values = self._latency_history[-60:]
        if not values:
            buf.append(f"                 {_DIM}no data yet{_RESET}")
            return
        lines = self._draw_sparkline(values, width=chart_width, height=chart_height)
        for line in lines:
            buf.append(f"  {line}")

    def _render_alerts(self, buf: List[str], w: int, max_lines: int = 5) -> None:
        buf.append(f" {_BOLD}ALERTS{_RESET}")
        anomalies = self._monitor.anomaly_history
        if not anomalies:
            buf.append(f"                 {_GREEN}No anomalies detected{_RESET}")
            return
        self._draw_alerts(buf, anomalies, max_lines)

    # -- drawing helpers ------------------------------------------------------

    @staticmethod
    def _draw_bar(value: float, max_value: float, width: int = 20) -> str:
        if max_value <= 0:
            return _LIGHT_SHADE * width
        ratio = max(0.0, min(value / max_value, 1.0))
        filled = int(ratio * width)
        empty = width - filled

        if ratio > 0.9:
            color = _RED
        elif ratio > 0.7:
            color = _YELLOW
        else:
            color = _GREEN

        return f"{color}{_FULL_BLOCK * filled}{_LIGHT_SHADE * empty}{_RESET}"

    @staticmethod
    def _draw_sparkline(
        values: List[float], width: int = 50, height: int = 5
    ) -> List[str]:
        if not values:
            return []

        n = len(values)
        if n > width:
            step = n / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = list(values)

        v_min = 0.0
        v_max = max(sampled) if sampled else 1.0
        if v_max == v_min:
            v_max = v_min + 1.0

        labels = []
        for row in range(height):
            frac = 1.0 - row / (height - 1) if height > 1 else 0.5
            labels.append(f"{v_min + frac * (v_max - v_min):5.0f}ms")

        grid = [[" "] * len(sampled) for _ in range(height)]
        for col, v in enumerate(sampled):
            row_f = (1.0 - (v - v_min) / (v_max - v_min)) * (height - 1)
            row = max(0, min(height - 1, int(round(row_f))))
            grid[row][col] = "●"
            for r in range(row + 1, height):
                if grid[r][col] == " ":
                    grid[r][col] = "│"

        lines = []
        for row in range(height):
            label = labels[row]
            row_str = "".join(grid[row])
            sep = "┤" if row < height - 1 else "┤"
            lines.append(f"{label} {sep}{row_str}")

        return lines

    @staticmethod
    def _draw_alerts(
        buf: List[str], anomalies: List[Anomaly], max_lines: int = 5
    ) -> None:
        recent = anomalies[-max_lines:]
        for a in reversed(recent):
            ts = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
            sev = a.severity.upper()
            color = _RED if a.severity == "critical" else _YELLOW
            buf.append(
                f"  {color}[!] {ts} {sev}: {a.message}{_RESET}"
            )

    # -- util -----------------------------------------------------------------

    @staticmethod
    def _format_uptime(seconds: float) -> str:
        s = int(seconds)
        h, rem = divmod(s, 3600)
        m, _ = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m:02d}m"
        return f"{m}m {s % 60:02d}s"

    @staticmethod
    def _format_size(mb: float) -> str:
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.0f} MB"


def monitor_live(
    pid: Optional[int] = None,
    model_path: Optional[str] = None,
    port: Optional[int] = None,
    dashboard: bool = True,
) -> InferenceMonitor:
    """CLI entry point: start monitoring and optionally launch the TUI dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    monitor = InferenceMonitor()
    monitor.start(pid=pid, model_path=model_path, port=port)

    if dashboard:
        dash = MonitorDashboard(monitor)
        try:
            dash.run()
        finally:
            monitor.stop()
    else:
        logger.info("Monitor running in headless mode (no dashboard)")

    return monitor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ISAT Live Inference Monitor")
    parser.add_argument("--pid", type=int, default=None, help="PID to monitor")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--port", type=int, default=None, help="Server metrics port")
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Run headless (no TUI)"
    )
    args = parser.parse_args()

    monitor_live(
        pid=args.pid,
        model_path=args.model_path,
        port=args.port,
        dashboard=not args.no_dashboard,
    )
