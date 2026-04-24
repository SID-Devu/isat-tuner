"""Structured logging configuration for ISAT.

Supports:
  - Structured JSON log format for production
  - Human-readable format for development
  - File + console output
  - Log level configuration
  - Request ID tracking
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production use."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        if hasattr(record, "model_name"):
            log_entry["model_name"] = record.model_name

        if hasattr(record, "config_label"):
            log_entry["config_label"] = record.config_label

        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        reset = self.RESET if color else ""

        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        ms = f"{record.msecs:03.0f}"

        return (
            f"{ts}.{ms} {color}{record.levelname:<7}{reset} "
            f"[{record.name}] {record.getMessage()}"
        )


def setup_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None,
    log_dir: str = "isat_logs",
) -> None:
    """Configure ISAT logging."""
    root = logging.getLogger("isat")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = HumanFormatter()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(StructuredFormatter())
        root.addHandler(file_handler)


def get_request_id() -> str:
    return str(uuid.uuid4())[:8]
