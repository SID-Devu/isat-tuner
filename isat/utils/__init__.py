"""Shared utilities for ISAT."""

from __future__ import annotations


def ort_providers(provider: str) -> list[str]:
    """Build a deduplicated ORT provider list. Avoids the
    'Duplicate provider CPUExecutionProvider' warning from ORT."""
    providers = [provider]
    if provider != "CPUExecutionProvider":
        providers.append("CPUExecutionProvider")
    return providers
