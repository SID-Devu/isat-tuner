"""Notification system for tuning job events.

Supports:
  - Webhook (generic HTTP POST)
  - Slack (incoming webhook)
  - Console (local notification)
  - Custom callbacks

Fires on: job_start, job_complete, job_failed, regression_detected, sla_violation
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Optional

log = logging.getLogger("isat.notifications")


@dataclass
class Event:
    type: str
    timestamp: float
    model: str
    message: str
    data: dict

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "model": self.model,
            "message": self.message,
            "data": self.data,
        }


class Notifier:
    """Base notifier that routes events to registered sinks."""

    def __init__(self):
        self._sinks: list[Callable[[Event], None]] = []

    def add_sink(self, sink: Callable[[Event], None]) -> "Notifier":
        self._sinks.append(sink)
        return self

    def notify(self, event_type: str, model: str, message: str, data: dict | None = None) -> None:
        event = Event(
            type=event_type,
            timestamp=time.time(),
            model=model,
            message=message,
            data=data or {},
        )
        for sink in self._sinks:
            try:
                sink(event)
            except Exception as e:
                log.error("Notification sink failed: %s", e)


class WebhookNotifier:
    """Send events to an HTTP webhook endpoint."""

    def __init__(self, url: str, headers: dict | None = None, timeout: int = 10):
        self.url = url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout

    def __call__(self, event: Event) -> None:
        payload = json.dumps(event.to_dict()).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                log.info("Webhook sent to %s: %d", self.url, resp.status)
        except Exception as e:
            log.error("Webhook failed: %s", e)


class SlackNotifier:
    """Send events to a Slack channel via incoming webhook."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url or os.environ.get("ISAT_SLACK_WEBHOOK", "")

    def __call__(self, event: Event) -> None:
        if not self.webhook_url:
            log.warning("Slack webhook URL not configured")
            return

        emoji = {"job_complete": ":white_check_mark:", "job_failed": ":x:",
                 "regression_detected": ":warning:", "sla_violation": ":rotating_light:"
                 }.get(event.type, ":information_source:")

        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": f"{emoji} ISAT: {event.type}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Model:* `{event.model}`\n*Message:* {event.message}"}},
        ]

        if event.data:
            fields = []
            for k, v in list(event.data.items())[:10]:
                fields.append({"type": "mrkdwn", "text": f"*{k}:* {v}"})
            blocks.append({"type": "section", "fields": fields})

        payload = json.dumps({"blocks": blocks}).encode()
        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                log.info("Slack notification sent: %d", resp.status)
        except Exception as e:
            log.error("Slack notification failed: %s", e)


class ConsoleNotifier:
    """Print events to console (for local development)."""

    def __call__(self, event: Event) -> None:
        icon = {"job_complete": "[OK]", "job_failed": "[FAIL]",
                "regression_detected": "[WARN]", "sla_violation": "[SLA]"
                }.get(event.type, "[INFO]")
        print(f"\n  {icon} {event.type}: {event.message}")
        if event.data:
            for k, v in event.data.items():
                print(f"    {k}: {v}")
