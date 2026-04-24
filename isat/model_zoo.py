"""Pre-tuned configurations for popular models.

When a user tunes a model ISAT recognizes, it can start from a known-good
configuration rather than searching from scratch. These are crowd-sourced
best configs from the community.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PretuneEntry:
    model_pattern: str
    model_class: str
    description: str
    recommended_env: dict[str, str]
    recommended_provider: str = "MIGraphXExecutionProvider"
    recommended_precision: str = "fp16"
    estimated_latency_ms: float = 0.0
    hardware_target: str = "any"
    notes: str = ""
    source: str = "isat-community"


MODEL_ZOO: list[PretuneEntry] = [
    PretuneEntry(
        model_pattern="resnet50",
        model_class="cnn",
        description="ResNet-50 image classification",
        recommended_env={"MIGRAPHX_FP16_ENABLE": "1"},
        recommended_precision="fp16",
        estimated_latency_ms=2.5,
        notes="FP16 gives 1.8x speedup with no accuracy loss on classification",
    ),
    PretuneEntry(
        model_pattern="bert",
        model_class="transformer",
        description="BERT-base/large NLP",
        recommended_env={
            "MIGRAPHX_FP16_ENABLE": "1",
            "MIGRAPHX_GPU_COMPILE_PARALLEL": "8",
        },
        recommended_precision="fp16",
        estimated_latency_ms=5.0,
        notes="Parallel compile reduces cold start by 4x",
    ),
    PretuneEntry(
        model_pattern="yolo",
        model_class="cnn",
        description="YOLO object detection (v5/v8/v10)",
        recommended_env={"MIGRAPHX_FP16_ENABLE": "1"},
        recommended_precision="fp16",
        estimated_latency_ms=8.0,
        notes="Pin input shapes to target resolution for best perf",
    ),
    PretuneEntry(
        model_pattern="whisper",
        model_class="transformer",
        description="OpenAI Whisper speech recognition",
        recommended_env={
            "MIGRAPHX_FP16_ENABLE": "1",
            "MIGRAPHX_GPU_COMPILE_PARALLEL": "8",
        },
        recommended_precision="fp16",
        estimated_latency_ms=50.0,
        notes="Encoder benefits more from FP16 than decoder",
    ),
    PretuneEntry(
        model_pattern="stable_diffusion|sd_|sdxl",
        model_class="transformer",
        description="Stable Diffusion / SDXL",
        recommended_env={
            "MIGRAPHX_FP16_ENABLE": "1",
            "HSA_XNACK": "1",
        },
        recommended_precision="fp16",
        estimated_latency_ms=500.0,
        hardware_target="apu_unified",
        notes="XNACK=1 critical for APUs to avoid OOM; uses demand paging",
    ),
    PretuneEntry(
        model_pattern="llama|mistral|qwen",
        model_class="llm",
        description="Large Language Models (LLaMA, Mistral, Qwen)",
        recommended_env={
            "HSA_XNACK": "1",
            "MIGRAPHX_FP16_ENABLE": "1",
            "GPU_MAX_HW_QUEUES": "2",
        },
        recommended_precision="fp16",
        estimated_latency_ms=200.0,
        hardware_target="apu_unified",
        notes="Oversubscribes VRAM on most GPUs; XNACK + queue limiting essential",
    ),
    PretuneEntry(
        model_pattern="openvla",
        model_class="llm",
        description="OpenVLA vision-language-action model",
        recommended_env={
            "HSA_XNACK": "1",
            "MIGRAPHX_FP16_ENABLE": "1",
            "MIGRAPHX_DISABLE_MLIR": "1",
            "MIGRAPHX_SET_GEMM_PROVIDER": "rocblas",
        },
        recommended_precision="fp16",
        estimated_latency_ms=300.0,
        hardware_target="apu_unified",
        notes="rocBLAS outperforms MLIR fusion by ~8% on this model",
    ),
    PretuneEntry(
        model_pattern="deepseek",
        model_class="llm",
        description="DeepSeek-R1 reasoning model",
        recommended_env={
            "HSA_XNACK": "1",
            "MIGRAPHX_FP16_ENABLE": "1",
        },
        recommended_precision="fp16",
        estimated_latency_ms=250.0,
        hardware_target="apu_unified",
        notes="Large model; benefits heavily from unified memory on APUs",
    ),
    PretuneEntry(
        model_pattern="mobilenet|efficientnet|mobilesam",
        model_class="cnn",
        description="Lightweight mobile models",
        recommended_env={},
        recommended_precision="int8",
        estimated_latency_ms=1.0,
        notes="INT8 quantization gives 3-4x speedup with <1% accuracy loss",
    ),
    PretuneEntry(
        model_pattern="vit|deit|swin",
        model_class="transformer",
        description="Vision Transformers (ViT, DeiT, Swin)",
        recommended_env={"MIGRAPHX_FP16_ENABLE": "1"},
        recommended_precision="fp16",
        estimated_latency_ms=4.0,
        notes="Attention-heavy; FP16 + MLIR fusion optimal",
    ),
    PretuneEntry(
        model_pattern="wav2vec|hubert",
        model_class="transformer",
        description="Speech representation models",
        recommended_env={"MIGRAPHX_FP16_ENABLE": "1"},
        recommended_precision="fp16",
        estimated_latency_ms=15.0,
    ),
    PretuneEntry(
        model_pattern="clip",
        model_class="transformer",
        description="CLIP vision-language model",
        recommended_env={"MIGRAPHX_FP16_ENABLE": "1"},
        recommended_precision="fp16",
        estimated_latency_ms=6.0,
        notes="Vision encoder + text encoder; tune separately for best results",
    ),
]


def lookup(model_name: str) -> Optional[PretuneEntry]:
    """Find a matching pre-tuned config for a model name."""
    import re
    name_lower = model_name.lower()
    for entry in MODEL_ZOO:
        if re.search(entry.model_pattern, name_lower):
            return entry
    return None


def suggest_starting_config(model_name: str, hw_class: str = "any") -> Optional[dict[str, str]]:
    """Get the recommended starting environment for a model."""
    entry = lookup(model_name)
    if not entry:
        return None

    if entry.hardware_target != "any" and entry.hardware_target != hw_class:
        return None

    return dict(entry.recommended_env)


def list_supported() -> list[dict]:
    """List all models in the zoo."""
    return [
        {
            "pattern": e.model_pattern,
            "class": e.model_class,
            "description": e.description,
            "precision": e.recommended_precision,
            "target": e.hardware_target,
        }
        for e in MODEL_ZOO
    ]
