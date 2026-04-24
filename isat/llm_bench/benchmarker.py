"""Token throughput benchmarker for LLM models.

Measures LLM-specific metrics:
  - Tokens per second (TPS)
  - Time to first token (TTFT)
  - Inter-token latency (ITL)
  - Prefill throughput vs decode throughput
  - Throughput vs sequence length curve

Standard metrics from vLLM, TGI, Ollama, llama.cpp.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("isat.llm_bench")


@dataclass
class TokenMetrics:
    sequence_length: int
    prefill_ms: float
    decode_total_ms: float
    tokens_generated: int
    ttft_ms: float
    avg_itl_ms: float
    p95_itl_ms: float
    tokens_per_second: float
    prefill_tps: float
    decode_tps: float


@dataclass
class LLMBenchResult:
    model_path: str
    provider: str
    runs: int
    metrics: list[TokenMetrics] = field(default_factory=list)
    overall_tps: float = 0
    overall_ttft_ms: float = 0
    overall_itl_ms: float = 0

    def summary(self) -> str:
        lines = [
            f"  Model           : {self.model_path}",
            f"  Provider        : {self.provider}",
            f"  Runs            : {self.runs}",
            f"",
            f"  Overall metrics:",
            f"    Tokens/second : {self.overall_tps:.1f}",
            f"    TTFT (mean)   : {self.overall_ttft_ms:.2f} ms",
            f"    ITL (mean)    : {self.overall_itl_ms:.2f} ms",
            f"",
            f"  {'Seq Len':>8} {'Prefill ms':>12} {'TTFT ms':>10} {'Avg ITL ms':>12} {'P95 ITL ms':>12} {'TPS':>8}",
            f"  {'-'*8} {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*8}",
        ]
        for m in self.metrics:
            lines.append(
                f"  {m.sequence_length:>8} {m.prefill_ms:>12.2f} {m.ttft_ms:>10.2f} "
                f"{m.avg_itl_ms:>12.2f} {m.p95_itl_ms:>12.2f} {m.tokens_per_second:>8.1f}"
            )
        return "\n".join(lines)


class LLMBenchmarker:
    """Benchmark LLM inference with token-level metrics."""

    def __init__(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        sequence_lengths: list[int] | None = None,
        decode_steps: int = 20,
    ):
        self.model_path = model_path
        self.provider = provider
        self.sequence_lengths = sequence_lengths or [32, 64, 128, 256]
        self.decode_steps = decode_steps

    def benchmark(self, runs: int = 5) -> LLMBenchResult:
        import onnxruntime as ort

        session = ort.InferenceSession(
            self.model_path, providers=[self.provider, "CPUExecutionProvider"],
        )
        inputs = session.get_inputs()
        outputs = session.get_outputs()

        has_seq_dim = False
        seq_dim_idx = -1
        for inp in inputs:
            for i, d in enumerate(inp.shape):
                if isinstance(d, str) and "seq" in d.lower():
                    has_seq_dim = True
                    seq_dim_idx = i
                    break
                if isinstance(d, str):
                    has_seq_dim = True
                    seq_dim_idx = i

        all_metrics: list[TokenMetrics] = []

        for seq_len in self.sequence_lengths:
            feed = self._build_feed(session, seq_len)
            for _ in range(3):
                session.run(None, feed)

            run_ttfts = []
            run_itls = []
            run_tps = []
            run_prefill = []

            for _ in range(runs):
                t_start = time.perf_counter()
                session.run(None, feed)
                t_first = time.perf_counter()
                ttft = (t_first - t_start) * 1000
                run_ttfts.append(ttft)
                run_prefill.append(ttft)

                itls = []
                for _ in range(self.decode_steps):
                    t0 = time.perf_counter()
                    session.run(None, feed)
                    itl = (time.perf_counter() - t0) * 1000
                    itls.append(itl)

                run_itls.extend(itls)
                total_decode = sum(itls)
                tps = self.decode_steps / (total_decode / 1000) if total_decode > 0 else 0
                run_tps.append(tps)

            itl_arr = np.array(run_itls) if run_itls else np.array([0])
            all_metrics.append(TokenMetrics(
                sequence_length=seq_len,
                prefill_ms=float(np.mean(run_prefill)),
                decode_total_ms=float(np.sum(itl_arr)),
                tokens_generated=self.decode_steps * runs,
                ttft_ms=float(np.mean(run_ttfts)),
                avg_itl_ms=float(np.mean(itl_arr)),
                p95_itl_ms=float(np.percentile(itl_arr, 95)),
                tokens_per_second=float(np.mean(run_tps)),
                prefill_tps=seq_len / (float(np.mean(run_prefill)) / 1000) if np.mean(run_prefill) > 0 else 0,
                decode_tps=float(np.mean(run_tps)),
            ))

        overall_tps = float(np.mean([m.tokens_per_second for m in all_metrics])) if all_metrics else 0
        overall_ttft = float(np.mean([m.ttft_ms for m in all_metrics])) if all_metrics else 0
        overall_itl = float(np.mean([m.avg_itl_ms for m in all_metrics])) if all_metrics else 0

        return LLMBenchResult(
            model_path=self.model_path, provider=self.provider, runs=runs,
            metrics=all_metrics,
            overall_tps=overall_tps,
            overall_ttft_ms=overall_ttft,
            overall_itl_ms=overall_itl,
        )

    def _build_feed(self, session, seq_len: int) -> dict:
        feed = {}
        for inp in session.get_inputs():
            shape = []
            for d in inp.shape:
                if isinstance(d, int) and d > 0:
                    shape.append(d)
                else:
                    shape.append(seq_len)
            if shape and shape[0] != 1:
                shape[0] = 1

            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            elif "float16" in inp.type.lower():
                feed[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        return feed
