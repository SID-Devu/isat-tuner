from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import numpy as np

from .kv_pool import KVCachePool
from .scheduler import (
    ContinuousBatchingScheduler,
    Request,
    RequestState,
    ScheduleBatch,
    SchedulerConfig,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ tokeniser


class _ByteTokenizer:
    vocab_size = 256
    eos_token_id = 0

    @staticmethod
    def encode(text: str) -> List[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(ids: List[int]) -> str:
        return bytes(min(max(i, 0), 255) for i in ids).decode(
            "utf-8", errors="replace"
        )


def _load_tokenizer(model_dir: str):
    try:
        from tokenizers import Tokenizer as _HFTok

        path = os.path.join(model_dir, "tokenizer.json")
        if os.path.isfile(path):
            tok = _HFTok.from_file(path)

            class _Wrap:
                vocab_size = tok.get_vocab_size()
                eos_token_id = tok.token_to_id("</s>") or 2

                @staticmethod
                def encode(text: str) -> List[int]:
                    return tok.encode(text).ids

                @staticmethod
                def decode(ids: List[int]) -> str:
                    return tok.decode(ids)

            return _Wrap()
    except Exception:
        pass
    return _ByteTokenizer()


# ------------------------------------------------------------------ histogram


class _Histogram:
    _DEFAULT_BUCKETS = (
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
    )

    def __init__(self, buckets: Optional[Tuple[float, ...]] = None) -> None:
        self._buckets = tuple(sorted(buckets or self._DEFAULT_BUCKETS)) + (float("inf"),)
        self._counts = [0] * len(self._buckets)
        self._sum = 0.0
        self._total = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._total += 1
        for i, b in enumerate(self._buckets):
            if value <= b:
                self._counts[i] += 1
                break

    @property
    def mean(self) -> float:
        return self._sum / self._total if self._total else 0.0

    def prometheus(self, name: str, help_text: str) -> str:
        lines = [f"# HELP {name} {help_text}", f"# TYPE {name} histogram"]
        cum = 0
        for i, b in enumerate(self._buckets):
            cum += self._counts[i]
            le = "+Inf" if b == float("inf") else f"{b}"
            lines.append(f'{name}_bucket{{le="{le}"}} {cum}')
        lines += [f"{name}_sum {self._sum}", f"{name}_count {self._total}"]
        return "\n".join(lines)


# ------------------------------------------------------------------ metrics


@dataclass
class ServerMetrics:
    total_requests: int = 0
    active_requests: int = 0
    tokens_generated: int = 0
    mean_ttft_ms: float = 0.0
    mean_tps: float = 0.0
    kv_utilization: float = 0.0
    queue_depth: int = 0


# ------------------------------------------------------------------ API models
# Lazy-loaded to avoid hard dependency on fastapi/pydantic at import time

try:
    from pydantic import BaseModel as _BaseModel

    class CompletionRequest(_BaseModel):
        model: str = ""
        prompt: str = ""
        max_tokens: int = 256
        temperature: float = 1.0
        top_k: int = -1
        top_p: float = 1.0
        stream: bool = False

    class ChatMessage(_BaseModel):
        role: str = "user"
        content: str = ""

    class ChatCompletionRequest(_BaseModel):
        model: str = ""
        messages: List[ChatMessage] = []
        max_tokens: int = 256
        temperature: float = 1.0
        top_k: int = -1
        top_p: float = 1.0
        stream: bool = False

except ImportError:
    CompletionRequest = None  # type: ignore[assignment,misc]
    ChatMessage = None  # type: ignore[assignment,misc]
    ChatCompletionRequest = None  # type: ignore[assignment,misc]


# ------------------------------------------------------------------ sampling


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _sample_logits(
    logits: np.ndarray,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    logits = logits.astype(np.float32)
    if temperature <= 0:
        return int(np.argmax(logits))

    logits /= temperature

    if 0 < top_k < len(logits):
        keep = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full_like(logits, -np.inf)
        mask[keep] = logits[keep]
        logits = mask

    if 0 < top_p < 1.0:
        order = np.argsort(logits)[::-1]
        probs = _softmax(logits[order])
        cutoff = int(np.searchsorted(np.cumsum(probs), top_p)) + 1
        logits[order[cutoff:]] = -np.inf

    probs = _softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


# ------------------------------------------------------------------ chat fmt


def _format_chat(messages: List[ChatMessage]) -> str:
    parts = [f"<|{m.role}|>\n{m.content}" for m in messages]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


# ------------------------------------------------------------------ server


class LLMServer:

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        scheduler_config: Optional[Dict[str, Any]] = None,
        kv_pool_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        import onnxruntime as ort

        self._model_path = model_path
        self._model_name = os.path.splitext(os.path.basename(model_path))[0]
        self._session = ort.InferenceSession(model_path, providers=[provider])
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        kv_cfg = kv_pool_config or {}
        self.kv_pool = KVCachePool(
            num_blocks=kv_cfg.get("num_blocks", 1024),
            block_size=kv_cfg.get("block_size", 16),
            num_layers=kv_cfg.get("num_layers", 32),
            num_heads=kv_cfg.get("num_heads", 32),
            head_dim=kv_cfg.get("head_dim", 128),
        )

        self.scheduler = ContinuousBatchingScheduler(
            SchedulerConfig(**(scheduler_config or {})), self.kv_pool
        )

        self._tokenizer = _load_tokenizer(os.path.dirname(model_path))
        self._request_queue: Optional[asyncio.Queue[Request]] = None
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        self._total_requests = 0
        self._tokens_generated = 0
        self._start_time = time.monotonic()
        self._ttft_hist = _Histogram()
        self._itl_hist = _Histogram()
        self._req_created: Dict[str, float] = {}
        self._req_last_tok: Dict[str, float] = {}

    # ----- lifecycle -----

    async def start(self) -> None:
        self._request_queue = asyncio.Queue()
        self._running = True
        self._start_time = time.monotonic()
        self._loop_task = asyncio.create_task(self._inference_loop())
        logger.info("Inference loop started")

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._loop_task.cancel()
        self._executor.shutdown(wait=False)
        logger.info("Server stopped")

    # ----- core loop -----

    async def _inference_loop(self) -> None:
        while self._running:
            drained = 0
            while not self._request_queue.empty() and drained < 64:
                try:
                    self.scheduler.add_request(self._request_queue.get_nowait())
                    drained += 1
                except asyncio.QueueEmpty:
                    break

            batch = self.scheduler.schedule()
            if batch.is_empty:
                await asyncio.sleep(0.001)
                continue

            try:
                logits = await asyncio.get_running_loop().run_in_executor(
                    self._executor, self._run_model, batch
                )
                await self._process_outputs(batch, logits)
            except Exception:
                logger.exception("Inference step failed")
                await asyncio.sleep(0.01)

    def _run_model(self, batch: ScheduleBatch) -> np.ndarray:
        ids = np.array(batch.input_ids, dtype=np.int64).reshape(1, -1)
        feeds: Dict[str, np.ndarray] = {"input_ids": ids}
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = np.ones_like(ids)
        if "position_ids" in self._input_names:
            feeds["position_ids"] = np.array(
                batch.positions, dtype=np.int64
            ).reshape(1, -1)

        outputs = self._session.run(None, feeds)
        return outputs[0].squeeze(0)

    async def _process_outputs(
        self, batch: ScheduleBatch, logits: np.ndarray
    ) -> None:
        now = time.monotonic()
        offset = 0

        for req in batch.prefill_requests:
            last_logit = logits[offset + req.chunk_size - 1]
            offset += req.chunk_size
            last_chunk = req.prefill_pos + req.chunk_size >= len(
                req.active_prefill_ids
            )

            if last_chunk:
                tok = _sample_logits(
                    last_logit, req.temperature, req.top_k, req.top_p
                )
                finished = tok == self._tokenizer.eos_token_id
                self.scheduler.step_complete(req.id, tok, finished)
                self._tokens_generated += 1
                if req.id in self._req_created:
                    self._ttft_hist.observe(now - self._req_created[req.id])
                self._req_last_tok[req.id] = now
                if req.callback:
                    await req.callback(tok, finished)
            else:
                self.scheduler.step_complete(req.id, -1, False)

        for req in batch.decode_requests:
            tok = _sample_logits(
                logits[offset], req.temperature, req.top_k, req.top_p
            )
            offset += 1
            finished = (
                len(req.generated_ids) + 1 >= req.max_tokens
                or tok == self._tokenizer.eos_token_id
            )
            self.scheduler.step_complete(req.id, tok, finished)
            self._tokens_generated += 1

            prev = self._req_last_tok.get(req.id)
            if prev is not None:
                self._itl_hist.observe(now - prev)
            self._req_last_tok[req.id] = now

            if finished:
                self._req_created.pop(req.id, None)
                self._req_last_tok.pop(req.id, None)

            if req.callback:
                await req.callback(tok, finished)

    # ----- public helpers -----

    def get_metrics(self) -> ServerMetrics:
        elapsed = time.monotonic() - self._start_time
        return ServerMetrics(
            total_requests=self._total_requests,
            active_requests=self.scheduler.num_running,
            tokens_generated=self._tokens_generated,
            mean_ttft_ms=self._ttft_hist.mean * 1000,
            mean_tps=self._tokens_generated / elapsed if elapsed > 0 else 0.0,
            kv_utilization=self.kv_pool.utilization,
            queue_depth=self.scheduler.num_waiting,
        )

    async def submit_request(
        self,
        prompt_ids: List[int],
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
    ) -> AsyncIterator[Tuple[int, bool]]:
        req_id = str(uuid.uuid4())
        token_q: asyncio.Queue[Tuple[int, bool]] = asyncio.Queue()

        async def _cb(token: int, done: bool) -> None:
            await token_q.put((token, done))

        req = Request(
            id=req_id,
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            callback=_cb,
        )
        self._total_requests += 1
        self._req_created[req_id] = time.monotonic()
        await self._request_queue.put(req)

        while True:
            tok, done = await token_q.get()
            yield tok, done
            if done:
                break


# ------------------------------------------------------------------ FastAPI app


def _create_app(server: LLMServer):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await server.start()
        yield
        await server.stop()

    app = FastAPI(title="LLM Inference Server", lifespan=lifespan)

    # ----- /v1/completions -----

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        prompt_ids = server._tokenizer.encode(body.prompt)
        if body.stream:
            return StreamingResponse(
                _sse_completion(server, prompt_ids, body),
                media_type="text/event-stream",
            )
        tokens: List[int] = []
        async for tid, _ in server.submit_request(
            prompt_ids, body.max_tokens, body.temperature, body.top_k, body.top_p
        ):
            tokens.append(tid)
        return _completion_json(server, prompt_ids, tokens, body.model)

    # ----- /v1/chat/completions -----

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        prompt = _format_chat(body.messages)
        prompt_ids = server._tokenizer.encode(prompt)
        if body.stream:
            return StreamingResponse(
                _sse_chat(server, prompt_ids, body),
                media_type="text/event-stream",
            )
        tokens: List[int] = []
        async for tid, _ in server.submit_request(
            prompt_ids, body.max_tokens, body.temperature, body.top_k, body.top_p
        ):
            tokens.append(tid)
        text = server._tokenizer.decode(tokens)
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model or server._model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt_ids) + len(tokens),
            },
        })

    # ----- /v1/models -----

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": server._model_name,
                "object": "model",
                "owned_by": "local",
            }],
        })

    # ----- /health -----

    @app.get("/health")
    async def health():
        m = server.get_metrics()
        return JSONResponse({
            "status": "ok",
            "queue_depth": m.queue_depth,
            "active_requests": m.active_requests,
            "kv_utilization": m.kv_utilization,
        })

    # ----- /metrics (Prometheus) -----

    @app.get("/metrics")
    async def metrics():
        m = server.get_metrics()
        parts = [
            "# HELP llm_request_count_total Total requests processed",
            "# TYPE llm_request_count_total counter",
            f"llm_request_count_total {m.total_requests}",
            "",
            "# HELP llm_tokens_generated_total Total tokens generated",
            "# TYPE llm_tokens_generated_total counter",
            f"llm_tokens_generated_total {m.tokens_generated}",
            "",
            server._ttft_hist.prometheus(
                "llm_time_to_first_token_seconds", "Time to first token"
            ),
            "",
            server._itl_hist.prometheus(
                "llm_inter_token_latency_seconds", "Inter-token latency"
            ),
            "",
            "# HELP llm_queue_depth Current queue depth",
            "# TYPE llm_queue_depth gauge",
            f"llm_queue_depth {m.queue_depth}",
            "",
            "# HELP llm_kv_cache_utilization KV cache utilization",
            "# TYPE llm_kv_cache_utilization gauge",
            f"llm_kv_cache_utilization {m.kv_utilization}",
        ]
        return Response(
            "\n".join(parts) + "\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


# ------------------------------------------------------------------ SSE helpers


def _completion_json(
    server: LLMServer, prompt_ids: List[int], tokens: List[int], model: str
) -> JSONResponse:
    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model or server._model_name,
        "choices": [{
            "text": server._tokenizer.decode(tokens),
            "index": 0,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt_ids) + len(tokens),
        },
    })


async def _sse_completion(
    server: LLMServer, prompt_ids: List[int], body: CompletionRequest
):
    cid = f"cmpl-{uuid.uuid4().hex[:8]}"
    async for tid, done in server.submit_request(
        prompt_ids, body.max_tokens, body.temperature, body.top_k, body.top_p
    ):
        chunk = {
            "id": cid,
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [{
                "text": server._tokenizer.decode([tid]),
                "index": 0,
                "finish_reason": "stop" if done else None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def _sse_chat(
    server: LLMServer, prompt_ids: List[int], body: ChatCompletionRequest
):
    cid = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    async for tid, done in server.submit_request(
        prompt_ids, body.max_tokens, body.temperature, body.top_k, body.top_p
    ):
        chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "delta": (
                    {"content": server._tokenizer.decode([tid])}
                    if not done
                    else {}
                ),
                "finish_reason": "stop" if done else None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# ------------------------------------------------------------------ CLI entry-point


def serve_llm(
    model_path: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    provider: str = "CPUExecutionProvider",
    scheduler_config: Optional[Dict[str, Any]] = None,
    kv_pool_config: Optional[Dict[str, Any]] = None,
    log_level: str = "info",
    **kwargs: Any,
) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = LLMServer(
        model_path=model_path,
        provider=provider,
        scheduler_config=scheduler_config,
        kv_pool_config=kv_pool_config,
    )
    app = _create_app(server)

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level=log_level, **kwargs)
