"""Model hub integration -- download ONNX models by name.

Supports:
  - HuggingFace Hub (optimum ONNX exports)
  - ONNX Model Zoo (github.com/onnx/models)
  - Local filesystem paths
  - Direct URLs

Usage:
  isat download resnet50
  isat download microsoft/resnet-50 --source huggingface
  isat download https://example.com/model.onnx
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.hub")

ONNX_ZOO_BASE = "https://github.com/onnx/models/raw/main/validated/vision"

KNOWN_MODELS: dict[str, dict] = {
    "mobilenetv2": {
        "url": f"{ONNX_ZOO_BASE}/classification/mobilenet/model/mobilenetv2-12.onnx",
        "description": "MobileNetV2 image classification (13 MB)",
        "opset": 12,
    },
    "resnet50": {
        "url": f"{ONNX_ZOO_BASE}/classification/resnet/model/resnet50-v2-7.onnx",
        "description": "ResNet-50 v2 image classification (98 MB)",
        "opset": 7,
    },
    "squeezenet": {
        "url": f"{ONNX_ZOO_BASE}/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "description": "SqueezeNet 1.1 image classification (5 MB)",
        "opset": 7,
    },
    "shufflenet": {
        "url": f"{ONNX_ZOO_BASE}/classification/shufflenet/model/shufflenet-v2-12.onnx",
        "description": "ShuffleNet V2 image classification (9 MB)",
        "opset": 12,
    },
    "efficientnet": {
        "url": f"{ONNX_ZOO_BASE}/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "description": "EfficientNet-Lite4 image classification (49 MB)",
        "opset": 11,
    },
}


@dataclass
class DownloadResult:
    model_name: str
    local_path: str
    size_mb: float
    source: str
    cached: bool = False


class ModelHub:
    """Download and cache ONNX models from various sources."""

    def __init__(self, cache_dir: str = "~/.cache/isat/models"):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        model_name: str,
        output_dir: str = ".",
        source: str = "auto",
        force: bool = False,
    ) -> DownloadResult:
        """Download a model by name or URL."""

        if model_name.startswith("http://") or model_name.startswith("https://"):
            return self._download_url(model_name, output_dir, force)

        if source == "huggingface" or "/" in model_name:
            return self._download_huggingface(model_name, output_dir, force)

        name_lower = model_name.lower().replace("-", "").replace("_", "")
        for key, info in KNOWN_MODELS.items():
            if key.replace("-", "").replace("_", "") == name_lower:
                return self._download_url(info["url"], output_dir, force, display_name=key)

        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {', '.join(KNOWN_MODELS.keys())}. "
            f"Or provide a URL or HuggingFace model ID (e.g., microsoft/resnet-50)"
        )

    def _download_url(self, url: str, output_dir: str, force: bool, display_name: str = "") -> DownloadResult:
        filename = display_name or url.split("/")[-1].split("?")[0]
        if not filename.endswith(".onnx"):
            filename += ".onnx"

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cached_path = self.cache_dir / filename
        output_path = Path(output_dir) / filename

        if output_path.exists() and not force:
            size = output_path.stat().st_size / (1024 * 1024)
            log.info("Already exists: %s (%.1f MB)", output_path, size)
            return DownloadResult(filename, str(output_path), size, "cache", cached=True)

        if cached_path.exists() and not force:
            shutil.copy2(cached_path, output_path)
            size = output_path.stat().st_size / (1024 * 1024)
            return DownloadResult(filename, str(output_path), size, "cache", cached=True)

        log.info("Downloading %s ...", url)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        urllib.request.urlretrieve(url, str(output_path))
        shutil.copy2(output_path, cached_path)

        size = output_path.stat().st_size / (1024 * 1024)
        log.info("Downloaded: %s (%.1f MB)", output_path, size)
        return DownloadResult(filename, str(output_path), size, "url")

    def _download_huggingface(self, model_id: str, output_dir: str, force: bool) -> DownloadResult:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=model_id,
                filename="model.onnx",
                cache_dir=str(self.cache_dir),
            )
            dest = Path(output_dir) / f"{model_id.replace('/', '_')}.onnx"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            size = dest.stat().st_size / (1024 * 1024)
            return DownloadResult(model_id, str(dest), size, "huggingface")
        except ImportError:
            api_url = f"https://huggingface.co/{model_id}/resolve/main/model.onnx"
            return self._download_url(api_url, output_dir, force, display_name=model_id.replace("/", "_"))

    def list_available(self) -> list[dict]:
        return [
            {"name": k, "description": v["description"], "opset": v["opset"]}
            for k, v in KNOWN_MODELS.items()
        ]

    def list_cached(self) -> list[dict]:
        cached = []
        for f in self.cache_dir.glob("*.onnx"):
            cached.append({
                "name": f.stem,
                "path": str(f),
                "size_mb": f.stat().st_size / (1024 * 1024),
            })
        return cached
