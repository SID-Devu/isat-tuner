"""Plugin system for extending ISAT with custom search dimensions and hooks.

Register plugins by creating a Python module with an `isat_plugin` function:

    # my_plugin.py
    def isat_plugin(registry):
        registry.register_search_dimension("custom", MySearchDimension)
        registry.register_hook("pre_benchmark", my_pre_hook)
        registry.register_hook("post_benchmark", my_post_hook)

Load plugins via CLI: isat tune model.onnx --plugin my_plugin
Or via config: plugins: ["my_plugin", "another_plugin"]
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Optional

log = logging.getLogger("isat.plugins")


class PluginRegistry:
    """Central registry for ISAT plugins."""

    def __init__(self):
        self._search_dimensions: dict[str, type] = {}
        self._hooks: dict[str, list[Callable]] = {
            "pre_fingerprint": [],
            "post_fingerprint": [],
            "pre_search": [],
            "post_search": [],
            "pre_benchmark": [],
            "post_benchmark": [],
            "pre_report": [],
            "post_report": [],
            "on_result": [],
            "on_error": [],
        }
        self._transforms: dict[str, Callable] = {}
        self._reporters: dict[str, Callable] = {}

    def register_search_dimension(self, name: str, cls: type) -> None:
        """Register a custom search dimension class."""
        self._search_dimensions[name] = cls
        log.info("Registered search dimension: %s", name)

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback for a lifecycle event."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event: {event}. Available: {list(self._hooks.keys())}")
        self._hooks[event].append(callback)
        log.info("Registered hook: %s -> %s", event, callback.__name__)

    def register_transform(self, name: str, fn: Callable) -> None:
        """Register a custom model transform function."""
        self._transforms[name] = fn
        log.info("Registered transform: %s", name)

    def register_reporter(self, name: str, fn: Callable) -> None:
        """Register a custom report generator."""
        self._reporters[name] = fn
        log.info("Registered reporter: %s", name)

    def fire(self, event: str, **kwargs) -> list[Any]:
        """Fire all callbacks for an event."""
        results = []
        for callback in self._hooks.get(event, []):
            try:
                result = callback(**kwargs)
                results.append(result)
            except Exception as e:
                log.error("Plugin hook %s failed: %s", callback.__name__, e)
        return results

    @property
    def search_dimensions(self) -> dict[str, type]:
        return dict(self._search_dimensions)

    @property
    def transforms(self) -> dict[str, Callable]:
        return dict(self._transforms)

    @property
    def reporters(self) -> dict[str, Callable]:
        return dict(self._reporters)


_registry = PluginRegistry()


def get_registry() -> PluginRegistry:
    return _registry


def load_plugin(module_name: str) -> None:
    """Load a plugin module and call its isat_plugin function."""
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, "isat_plugin"):
            mod.isat_plugin(_registry)
            log.info("Loaded plugin: %s", module_name)
        else:
            log.warning("Module %s has no isat_plugin function", module_name)
    except ImportError as e:
        log.error("Failed to load plugin %s: %s", module_name, e)


def load_plugins(module_names: list[str]) -> None:
    """Load multiple plugins."""
    for name in module_names:
        load_plugin(name)
