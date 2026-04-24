"""Inference request tracing with OpenTelemetry-compatible spans.

Tracks each inference request through the full lifecycle:
  enqueue -> preprocess -> inference -> postprocess -> response

Exports traces in OTLP-compatible JSON for integration with Jaeger,
Zipkin, Datadog, or any OpenTelemetry collector.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("isat.tracing")


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    operation: str = ""
    service: str = "isat-inference"
    start_us: int = 0
    end_us: int = 0
    duration_us: int = 0
    status: str = "OK"  # "OK", "ERROR"
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "traceId": self.trace_id,
            "spanId": self.span_id,
            "parentSpanId": self.parent_span_id,
            "operationName": self.operation,
            "serviceName": self.service,
            "startTimeUnixNano": self.start_us * 1000,
            "endTimeUnixNano": self.end_us * 1000,
            "durationMs": self.duration_us / 1000,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }


@dataclass
class Trace:
    trace_id: str
    spans: list[Span] = field(default_factory=list)
    model: str = ""
    total_ms: float = 0

    def summary(self) -> str:
        lines = [
            f"  Trace ID   : {self.trace_id}",
            f"  Model      : {self.model}",
            f"  Total      : {self.total_ms:.2f} ms",
            f"  Spans      : {len(self.spans)}",
            f"",
            f"  {'Operation':<30} {'Duration ms':>12} {'Status':>8}",
            f"  {'-'*30} {'-'*12} {'-'*8}",
        ]
        for s in self.spans:
            dur = s.duration_us / 1000
            lines.append(f"  {s.operation:<30} {dur:>12.2f} {s.status:>8}")
        return "\n".join(lines)


class InferenceTracer:
    """Trace inference requests through the full lifecycle."""

    def __init__(self, service_name: str = "isat-inference", export_dir: str = ""):
        self.service_name = service_name
        self.export_dir = export_dir
        self._traces: list[Trace] = []

    def trace_inference(
        self,
        model_path: str,
        provider: str = "CPUExecutionProvider",
        runs: int = 10,
    ) -> list[Trace]:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(
            model_path, providers=[provider, "CPUExecutionProvider"],
        )
        feed = self._build_feed(session)

        for _ in range(3):
            session.run(None, feed)

        traces = []
        for i in range(runs):
            trace_id = uuid.uuid4().hex
            spans = []
            total_start = time.perf_counter_ns() // 1000

            # Span: preprocess
            s0 = time.perf_counter_ns() // 1000
            feed = self._build_feed(session)
            s1 = time.perf_counter_ns() // 1000
            spans.append(Span(
                trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                operation="preprocess/build_feed",
                service=self.service_name,
                start_us=s0, end_us=s1, duration_us=s1 - s0,
                attributes={"input_count": len(feed)},
            ))

            # Span: inference
            s2 = time.perf_counter_ns() // 1000
            outputs = session.run(None, feed)
            s3 = time.perf_counter_ns() // 1000
            spans.append(Span(
                trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                parent_span_id=spans[0].span_id,
                operation="inference/session.run",
                service=self.service_name,
                start_us=s2, end_us=s3, duration_us=s3 - s2,
                attributes={"provider": provider, "output_count": len(outputs)},
            ))

            # Span: postprocess
            s4 = time.perf_counter_ns() // 1000
            result_shapes = [o.shape for o in outputs]
            s5 = time.perf_counter_ns() // 1000
            spans.append(Span(
                trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                parent_span_id=spans[0].span_id,
                operation="postprocess/output_shapes",
                service=self.service_name,
                start_us=s4, end_us=s5, duration_us=s5 - s4,
                attributes={"shapes": str(result_shapes)},
            ))

            total_end = time.perf_counter_ns() // 1000

            root_span = Span(
                trace_id=trace_id, span_id=uuid.uuid4().hex[:16],
                operation="inference/e2e",
                service=self.service_name,
                start_us=total_start, end_us=total_end,
                duration_us=total_end - total_start,
                attributes={"model": model_path, "run_index": i},
            )
            all_spans = [root_span] + spans
            for s in spans:
                s.parent_span_id = root_span.span_id

            trace = Trace(
                trace_id=trace_id, spans=all_spans,
                model=model_path,
                total_ms=(total_end - total_start) / 1000,
            )
            traces.append(trace)

        self._traces.extend(traces)
        return traces

    def export_otlp_json(self, output_path: str = "") -> str:
        path = output_path or os.path.join(
            self.export_dir or ".", f"traces_{int(time.time())}.json"
        )
        data = {
            "resourceSpans": [{
                "resource": {"attributes": [
                    {"key": "service.name", "value": {"stringValue": self.service_name}},
                ]},
                "scopeSpans": [{
                    "scope": {"name": "isat.tracing"},
                    "spans": [s.to_dict() for t in self._traces for s in t.spans],
                }],
            }]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def get_stats(self) -> dict:
        if not self._traces:
            return {"total_traces": 0}
        import numpy as np
        totals = [t.total_ms for t in self._traces]
        inference_spans = [
            s.duration_us / 1000 for t in self._traces
            for s in t.spans if "session.run" in s.operation
        ]
        return {
            "total_traces": len(self._traces),
            "e2e_mean_ms": float(np.mean(totals)),
            "e2e_p95_ms": float(np.percentile(totals, 95)),
            "inference_mean_ms": float(np.mean(inference_spans)) if inference_spans else 0,
            "inference_p95_ms": float(np.percentile(inference_spans, 95)) if inference_spans else 0,
        }

    def _build_feed(self, session) -> dict:
        import numpy as np
        feed = {}
        for inp in session.get_inputs():
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
            if "int" in inp.type.lower():
                feed[inp.name] = np.ones(shape, dtype=np.int64)
            elif "float16" in inp.type.lower():
                feed[inp.name] = np.random.randn(*shape).astype(np.float16)
            else:
                feed[inp.name] = np.random.randn(*shape).astype(np.float32)
        return feed
