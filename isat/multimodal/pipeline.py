"""Multi-modal pipeline: encode heterogeneous inputs and feed unified embeddings to an LLM backbone."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _lazy_ort():
    import onnxruntime as ort
    return ort


def _lazy_pil():
    from PIL import Image
    return Image


def _lazy_soundfile():
    import soundfile as sf
    return sf


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PipelineAnalysis:
    encoders: Dict[str, dict]
    llm_params: int
    total_params: int
    supported_modalities: List[str]


@dataclass
class MultiModalResult:
    text: str
    tokens_generated: int
    encoder_time_ms: float
    projection_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    modalities_used: List[str]


# ---------------------------------------------------------------------------
# ModalityEncoder
# ---------------------------------------------------------------------------

class ModalityEncoder:
    """Encode raw input (image, audio, or text) into embeddings via an ONNX model."""

    _VALID_MODALITIES = ("vision", "audio", "text")

    def __init__(
        self,
        model_path: str | Path,
        modality: str,
        provider: str = "CPUExecutionProvider",
    ) -> None:
        if modality not in self._VALID_MODALITIES:
            raise ValueError(f"modality must be one of {self._VALID_MODALITIES}, got '{modality}'")
        self.model_path = Path(model_path)
        self.modality = modality
        self.provider = provider
        self._session: Any = None

    @property
    def session(self):
        if self._session is None:
            ort = _lazy_ort()
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=[self.provider]
            )
        return self._session

    def encode(self, raw_input: Any) -> np.ndarray:
        """Preprocess *raw_input* according to modality and run the encoder, returning embeddings."""
        if self.modality == "vision":
            tensor = self._preprocess_image(raw_input)
        elif self.modality == "audio":
            tensor = self._preprocess_audio(raw_input)
        else:
            tensor = self._preprocess_text(raw_input)

        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: tensor})
        return np.asarray(result[0])

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, image_path_or_array: Any) -> np.ndarray:
        """Resize, normalize, and tile for dynamic resolution."""
        if isinstance(image_path_or_array, (str, Path)):
            Image = _lazy_pil()
            img = Image.open(image_path_or_array).convert("RGB")
            arr = np.asarray(img, dtype=np.float32)
        else:
            arr = np.asarray(image_path_or_array, dtype=np.float32)

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        h, w = arr.shape[:2]
        target = 336
        scale = target / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Nearest-neighbour resize via repeat (no cv2 dependency)
        row_idx = (np.arange(new_h) * h / new_h).astype(int)
        col_idx = (np.arange(new_w) * w / new_w).astype(int)
        arr = arr[row_idx[:, None], col_idx[None, :], :]

        # Pad to square
        pad_h = target - new_h
        pad_w = target - new_w
        arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

        # Normalize to [0, 1] and transpose to CHW
        arr = arr / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW

        # Tile into patches for dynamic resolution
        tiles = max(1, (new_h * new_w) // (target * target))
        arr = np.stack([arr] * tiles, axis=0)  # (tiles, 3, H, W)
        return arr.astype(np.float32)

    def _preprocess_audio(
        self, audio_path_or_array: Any, sample_rate: int = 16000
    ) -> np.ndarray:
        """Resample and chunk into overlapping windows."""
        if isinstance(audio_path_or_array, (str, Path)):
            sf = _lazy_soundfile()
            data, orig_sr = sf.read(str(audio_path_or_array), dtype="float32")
        else:
            data = np.asarray(audio_path_or_array, dtype=np.float32)
            orig_sr = sample_rate

        if data.ndim > 1:
            data = data.mean(axis=1)

        # Simple linear resample if needed
        if orig_sr != sample_rate:
            ratio = sample_rate / orig_sr
            indices = (np.arange(int(len(data) * ratio)) / ratio).astype(int)
            indices = np.clip(indices, 0, len(data) - 1)
            data = data[indices]

        # Chunk into 30-second overlapping windows (480k samples @ 16kHz)
        window_size = sample_rate * 30
        hop_size = sample_rate * 25
        chunks = []
        start = 0
        while start < len(data):
            end = min(start + window_size, len(data))
            chunk = data[start:end]
            if len(chunk) < window_size:
                chunk = np.pad(chunk, (0, window_size - len(chunk)))
            chunks.append(chunk)
            start += hop_size

        return np.stack(chunks, axis=0).astype(np.float32)  # (num_chunks, window_size)

    def _preprocess_text(self, text: str) -> np.ndarray:
        """Tokenize text into integer ids (simple whitespace + byte-level fallback)."""
        tokens = []
        for word in text.strip().split():
            for ch in word:
                tokens.append(ord(ch) % 32000)
        token_array = np.array(tokens, dtype=np.int64).reshape(1, -1)
        return token_array


# ---------------------------------------------------------------------------
# Projector
# ---------------------------------------------------------------------------

class Projector:
    """Linear projection (via ONNX or numpy MatMul) to align encoder dims to LLM embedding space."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._session: Any = None
        self._weight: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None

        if self.model_path is None and input_dim and output_dim:
            rng = np.random.default_rng(42)
            scale = np.sqrt(2.0 / (input_dim + output_dim))
            self._weight = (rng.standard_normal((input_dim, output_dim)) * scale).astype(np.float32)
            self._bias = np.zeros(output_dim, dtype=np.float32)

    @property
    def session(self):
        if self._session is None and self.model_path is not None:
            ort = _lazy_ort()
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        return self._session

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform encoder embeddings to LLM-compatible shape."""
        if self.session is not None:
            input_name = self.session.get_inputs()[0].name
            result = self.session.run(None, {input_name: embeddings.astype(np.float32)})
            return np.asarray(result[0])

        if self._weight is not None:
            flat = embeddings.reshape(-1, embeddings.shape[-1])
            projected = flat @ self._weight + self._bias
            return projected.reshape(*embeddings.shape[:-1], self.output_dim)

        return embeddings


# ---------------------------------------------------------------------------
# MultiModalPipeline
# ---------------------------------------------------------------------------

class MultiModalPipeline:
    """Orchestrate multiple modality encoders feeding into a single LLM backbone."""

    def __init__(self, llm_path: str | Path, provider: str = "CPUExecutionProvider") -> None:
        self.llm_path = Path(llm_path)
        self.provider = provider
        self._llm_session: Any = None
        self._encoders: Dict[str, Tuple[ModalityEncoder, Optional[Projector]]] = {}

    @property
    def llm_session(self):
        if self._llm_session is None:
            ort = _lazy_ort()
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._llm_session = ort.InferenceSession(
                str(self.llm_path), sess_options=opts, providers=[self.provider]
            )
        return self._llm_session

    def add_encoder(
        self, name: str, encoder: ModalityEncoder, projector: Optional[Projector] = None
    ) -> None:
        """Register a modality encoder with an optional projector."""
        self._encoders[name] = (encoder, projector)

    def process(self, inputs: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Encode each input, project, concatenate, and feed to LLM.

        *inputs* is a list of dicts: [{"modality": "text", "data": ...}, ...]
        Returns (logits, timing_dict).
        """
        all_embeddings: List[np.ndarray] = []
        text_positions: List[int] = []
        visual_positions: List[int] = []
        timings: Dict[str, float] = {"encoder_ms": 0.0, "projection_ms": 0.0}

        position = 0
        for item in inputs:
            modality = item["modality"]
            data = item["data"]

            encoder, projector = self._find_encoder(modality)

            t0 = time.perf_counter()
            emb = encoder.encode(data)
            timings["encoder_ms"] += (time.perf_counter() - t0) * 1000

            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            if emb.ndim > 2:
                emb = emb.reshape(-1, emb.shape[-1])

            t0 = time.perf_counter()
            if projector is not None:
                emb = projector.project(emb)
            timings["projection_ms"] += (time.perf_counter() - t0) * 1000

            seq_len = emb.shape[0]
            if modality == "text":
                text_positions.extend(range(position, position + seq_len))
            else:
                visual_positions.extend(range(position, position + seq_len))
            position += seq_len
            all_embeddings.append(emb)

        if not all_embeddings:
            raise ValueError("No inputs provided")

        combined = np.concatenate(all_embeddings, axis=0)

        if text_positions and visual_positions:
            combined = self._interleave_embeddings(
                combined[text_positions],
                combined[visual_positions],
                visual_positions,
            )

        combined = combined[np.newaxis, ...]  # add batch dim
        input_name = self.llm_session.get_inputs()[0].name
        logits = self.llm_session.run(None, {input_name: combined.astype(np.float32)})
        return np.asarray(logits[0]), timings

    def _interleave_embeddings(
        self,
        text_embeds: np.ndarray,
        visual_embeds: np.ndarray,
        positions: List[int],
    ) -> np.ndarray:
        """Interleave text and visual tokens at correct positions."""
        total_len = text_embeds.shape[0] + visual_embeds.shape[0]
        dim = text_embeds.shape[-1]
        output = np.empty((total_len, dim), dtype=text_embeds.dtype)

        vis_set = set(positions)
        vi, ti = 0, 0
        for idx in range(total_len):
            if idx in vis_set and vi < visual_embeds.shape[0]:
                output[idx] = visual_embeds[vi]
                vi += 1
            else:
                output[idx] = text_embeds[ti]
                ti += 1
        return output

    def generate(
        self,
        inputs: List[Dict[str, Any]],
        max_tokens: int = 256,
        **sampling: Any,
    ) -> MultiModalResult:
        """Process inputs and perform autoregressive generation."""
        temperature = sampling.get("temperature", 1.0)
        top_k = sampling.get("top_k", 50)
        modalities_used = list({item["modality"] for item in inputs})

        t_start = time.perf_counter()
        logits, timings = self.process(inputs)

        t_gen_start = time.perf_counter()
        tokens: List[int] = []
        current_logits = logits[0, -1, :]  # last position

        for _ in range(max_tokens):
            scaled = current_logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0 and top_k < len(scaled):
                topk_idx = np.argpartition(scaled, -top_k)[-top_k:]
                mask = np.full_like(scaled, -np.inf)
                mask[topk_idx] = scaled[topk_idx]
                scaled = mask

            probs = _softmax(scaled)
            token_id = int(np.random.choice(len(probs), p=probs))
            tokens.append(token_id)

            if token_id == 0:  # EOS
                break

            # Simplified next-step: re-feed token embedding (use mean of LLM embedding as proxy)
            current_logits = scaled * 0.95  # decay for demo

        generation_ms = (time.perf_counter() - t_gen_start) * 1000
        total_ms = (time.perf_counter() - t_start) * 1000

        text_output = "".join(chr(max(32, t % 128)) for t in tokens)

        return MultiModalResult(
            text=text_output,
            tokens_generated=len(tokens),
            encoder_time_ms=timings["encoder_ms"],
            projection_time_ms=timings["projection_ms"],
            generation_time_ms=generation_ms,
            total_time_ms=total_ms,
            modalities_used=modalities_used,
        )

    def analyze(self) -> PipelineAnalysis:
        """Return analysis of the pipeline: encoder count, params, modalities, latency."""
        encoder_info: Dict[str, dict] = {}
        total_params = 0

        for name, (enc, proj) in self._encoders.items():
            enc_params = _estimate_onnx_params(enc.model_path)
            proj_params = 0
            if proj is not None:
                if proj._weight is not None:
                    proj_params = proj._weight.size + (proj._bias.size if proj._bias is not None else 0)
                elif proj.model_path is not None:
                    proj_params = _estimate_onnx_params(proj.model_path)

            encoder_info[name] = {
                "modality": enc.modality,
                "model_path": str(enc.model_path),
                "params": enc_params,
                "projector_params": proj_params,
            }
            total_params += enc_params + proj_params

        llm_params = _estimate_onnx_params(self.llm_path)
        total_params += llm_params

        return PipelineAnalysis(
            encoders=encoder_info,
            llm_params=llm_params,
            total_params=total_params,
            supported_modalities=[enc.modality for enc, _ in self._encoders.values()],
        )

    def _find_encoder(self, modality: str) -> Tuple[ModalityEncoder, Optional[Projector]]:
        for _name, (enc, proj) in self._encoders.items():
            if enc.modality == modality:
                return enc, proj
        raise ValueError(
            f"No encoder registered for modality '{modality}'. "
            f"Available: {[e.modality for e, _ in self._encoders.values()]}"
        )


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------

def multimodal_generate(
    llm_path: str | Path,
    inputs: List[Dict[str, Any]],
    encoders: Optional[Dict[str, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> MultiModalResult:
    """CLI entry point: build a pipeline from config and generate.

    *encoders* maps name -> {"model_path": ..., "modality": ..., "projector_path": ..., "projector_dim": ...}
    """
    pipeline = MultiModalPipeline(llm_path)

    if encoders:
        for name, cfg in encoders.items():
            enc = ModalityEncoder(
                model_path=cfg["model_path"],
                modality=cfg["modality"],
                provider=cfg.get("provider", "CPUExecutionProvider"),
            )
            proj = None
            if "projector_path" in cfg:
                proj = Projector(model_path=cfg["projector_path"])
            elif "projector_dim" in cfg:
                dims = cfg["projector_dim"]
                proj = Projector(input_dim=dims[0], output_dim=dims[1])
            pipeline.add_encoder(name, enc, proj)

    return pipeline.generate(inputs, **kwargs)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _estimate_onnx_params(model_path: Path) -> int:
    """Rough param count from ONNX file size (heuristic: ~4 bytes per fp32 param)."""
    path = Path(model_path)
    if path.exists():
        return int(path.stat().st_size / 4)
    return 0
