"""Execution provider search dimension.

Discovers available ORT execution providers on the current system
and generates candidates for each.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("isat.search.provider")

PROVIDER_PRIORITY = [
    "TensorrtExecutionProvider",
    "MIGraphXExecutionProvider",
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "OpenVINOExecutionProvider",
    "DmlExecutionProvider",
    "QNNExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

PROVIDER_OPTIONS: dict[str, list[dict[str, Any]]] = {
    "TensorrtExecutionProvider": [
        {"trt_fp16_enable": True, "trt_engine_cache_enable": True},
        {"trt_fp16_enable": False, "trt_engine_cache_enable": True},
        {"trt_fp16_enable": True, "trt_int8_enable": True, "trt_engine_cache_enable": True},
    ],
    "CUDAExecutionProvider": [
        {"cudnn_conv_algo_search": "EXHAUSTIVE"},
        {"cudnn_conv_algo_search": "DEFAULT"},
        {"arena_extend_strategy": "kSameAsRequested"},
    ],
    "MIGraphXExecutionProvider": [
        {"migraphx_fp16_enable": True},
        {"migraphx_fp16_enable": False},
        {"migraphx_int8_enable": True},
    ],
    "ROCMExecutionProvider": [
        {},
        {"tunable_op_enable": True, "tunable_op_tuning_enable": True},
    ],
    "OpenVINOExecutionProvider": [
        {"device_type": "GPU"},
        {"device_type": "CPU"},
        {"device_type": "GPU", "precision": "FP16"},
    ],
}


@dataclass
class ProviderConfig:
    provider_name: str = "CPUExecutionProvider"
    provider_options: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    def __post_init__(self):
        if not self.label:
            short = self.provider_name.replace("ExecutionProvider", "")
            opts = "_".join(f"{k}={v}" for k, v in self.provider_options.items())
            self.label = f"{short}_{opts}" if opts else short


class ProviderSearchDimension:
    """Discover and generate provider candidates."""

    def __init__(self):
        self.available = self._detect_providers()

    @staticmethod
    def _detect_providers() -> list[str]:
        try:
            import onnxruntime as ort
            return ort.get_available_providers()
        except ImportError:
            return ["CPUExecutionProvider"]

    def candidates(self) -> list[ProviderConfig]:
        configs: list[ProviderConfig] = []
        seen_labels: set[str] = set()

        for provider in PROVIDER_PRIORITY:
            if provider not in self.available:
                continue

            base = ProviderConfig(provider_name=provider, provider_options={})
            if base.label not in seen_labels:
                configs.append(base)
                seen_labels.add(base.label)

            for opts in PROVIDER_OPTIONS.get(provider, []):
                cfg = ProviderConfig(provider_name=provider, provider_options=dict(opts))
                if cfg.label not in seen_labels:
                    configs.append(cfg)
                    seen_labels.add(cfg.label)

        if not configs:
            configs.append(ProviderConfig(provider_name="CPUExecutionProvider"))

        return configs

    def get_provider_list(self, config: ProviderConfig) -> list[tuple[str, dict] | str]:
        """Return the ORT providers list for session creation."""
        if config.provider_options:
            return [(config.provider_name, config.provider_options), "CPUExecutionProvider"]
        return [config.provider_name, "CPUExecutionProvider"]
