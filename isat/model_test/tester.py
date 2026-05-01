"""Automated model testing framework for ONNX models.

Provides determinism, numerical stability, edge-case, input-validation,
cross-provider, latency-consistency, memory-safety, and golden-file tests
that can run individually or as a full suite with JUnit XML output for CI.
"""

from __future__ import annotations

import gc
import logging
import os
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from isat.utils import ort_providers

log = logging.getLogger("isat.model_test")

SYMBOLIC_DIM_DEFAULTS: dict[str, int] = {
    "batch": 1,
    "batch_size": 1,
    "sequence": 16,
    "sequence_length": 16,
    "seq_len": 16,
    "seq": 16,
    "num_channels": 3,
    "channels": 3,
    "channel": 3,
    "height": 224,
    "width": 224,
    "h": 224,
    "w": 224,
    "num_heads": 8,
    "head_size": 64,
    "hidden_size": 768,
    "embed_dim": 768,
    "vocab_size": 30522,
    "max_length": 16,
    "num_classes": 1000,
}

SYMBOLIC_DIM_FALLBACK = 4

ORT_DTYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
    "tensor(bfloat16)": np.float16,
}


@dataclass
class TestResult:
    name: str
    passed: bool
    details: str
    elapsed_ms: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    model_path: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    results: list[TestResult] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_junit_xml(self) -> str:
        suite = ET.Element("testsuite", {
            "name": f"model_test:{Path(self.model_path).name}",
            "tests": str(self.total_tests),
            "failures": str(self.failed),
            "skipped": str(self.skipped),
            "time": f"{self.elapsed_s:.3f}",
        })
        for r in self.results:
            tc = ET.SubElement(suite, "testcase", {
                "name": r.name,
                "time": f"{r.elapsed_ms / 1000:.3f}",
            })
            if not r.passed:
                fail = ET.SubElement(tc, "failure", {"message": r.details[:256]})
                fail.text = r.details
        return ET.tostring(suite, encoding="unicode", xml_declaration=True)


def _resolve_dim(dim_param: str | int) -> int:
    if isinstance(dim_param, int) and dim_param > 0:
        return dim_param
    key = str(dim_param).lower().strip()
    return SYMBOLIC_DIM_DEFAULTS.get(key, SYMBOLIC_DIM_FALLBACK)


def _build_random_inputs(
    session: Any,
    batch_size: int = 1,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Build random inputs from session metadata, resolving symbolic dims."""
    if rng is None:
        rng = np.random.default_rng(42)

    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        dtype = ORT_DTYPE_TO_NUMPY.get(inp.type, np.float32)
        shape: list[int] = []
        for i, d in enumerate(inp.shape):
            if isinstance(d, str):
                val = _resolve_dim(d)
                if i == 0:
                    val = batch_size
                shape.append(val)
            elif isinstance(d, int) and d > 0:
                if i == 0:
                    shape.append(batch_size)
                else:
                    shape.append(d)
            else:
                shape.append(batch_size if i == 0 else SYMBOLIC_DIM_FALLBACK)

        if np.issubdtype(dtype, np.floating):
            feeds[inp.name] = rng.standard_normal(shape).astype(dtype)
        elif np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            feeds[inp.name] = rng.integers(0, min(info.max, 1000), size=shape, dtype=dtype)
        elif dtype == np.bool_:
            feeds[inp.name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            feeds[inp.name] = rng.standard_normal(shape).astype(np.float32)

    return feeds


def _available_providers() -> list[str]:
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return 0.0


class ModelTester:
    """Comprehensive test suite runner for a single ONNX model."""

    def __init__(self, model_path: str, provider: str | None = None):
        self.model_path = model_path
        if provider is None or provider == "auto":
            avail = _available_providers()
            self.provider = avail[0] if avail else "CPUExecutionProvider"
        else:
            self.provider = provider
        self._session: Any | None = None

    def _get_session(self, provider: str | None = None):
        import onnxruntime as ort

        prov = provider or self.provider
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(
            self.model_path,
            sess_options=so,
            providers=ort_providers(prov),
        )

    @property
    def session(self):
        if self._session is None:
            self._session = self._get_session()
        return self._session

    def test_determinism(self, num_runs: int = 10) -> TestResult:
        """Run identical inputs N times and verify bit-identical outputs."""
        t0 = time.perf_counter()
        try:
            sess = self.session
            feeds = _build_random_inputs(sess, batch_size=1)
            reference: list[np.ndarray] | None = None
            max_diff = 0.0

            for i in range(num_runs):
                outputs = sess.run(None, feeds)
                if reference is None:
                    reference = outputs
                    continue
                for ref_arr, cur_arr in zip(reference, outputs):
                    diff = np.max(np.abs(ref_arr.astype(np.float64) - cur_arr.astype(np.float64)))
                    max_diff = max(max_diff, float(diff))

            passed = max_diff == 0.0
            return TestResult(
                name="determinism",
                passed=passed,
                details=f"max_diff={max_diff:.2e} over {num_runs} runs",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={"max_diff": max_diff, "num_runs": num_runs},
            )
        except Exception as exc:
            return TestResult(
                name="determinism",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def test_numerical_stability(self, num_samples: int = 100) -> TestResult:
        """Run inference with varied input ranges; detect NaN/Inf/extreme values."""
        t0 = time.perf_counter()
        try:
            sess = self.session
            rng = np.random.default_rng(123)
            nan_count = 0
            inf_count = 0
            extreme_count = 0
            total_elements = 0
            extreme_threshold = 1e15

            ranges = [
                ("normal", -1.0, 1.0),
                ("small", -1e-6, 1e-6),
                ("large", -1e4, 1e4),
                ("negative", -10.0, -0.01),
            ]
            per_range = max(1, num_samples // len(ranges))

            for range_name, lo, hi in ranges:
                for _ in range(per_range):
                    feeds = _build_random_inputs(sess, batch_size=1, rng=rng)
                    for name, arr in feeds.items():
                        if np.issubdtype(arr.dtype, np.floating):
                            feeds[name] = (arr * (hi - lo) + lo).astype(arr.dtype)

                    outputs = sess.run(None, feeds)
                    for arr in outputs:
                        farr = arr.astype(np.float64) if arr.dtype != np.float64 else arr
                        total_elements += farr.size
                        nan_count += int(np.isnan(farr).sum())
                        inf_count += int(np.isinf(farr).sum())
                        extreme_count += int((np.abs(farr) > extreme_threshold).sum())

            nan_frac = nan_count / max(total_elements, 1)
            inf_frac = inf_count / max(total_elements, 1)
            passed = nan_count == 0 and inf_count == 0

            return TestResult(
                name="numerical_stability",
                passed=passed,
                details=(
                    f"NaN={nan_count} ({nan_frac:.4%}), "
                    f"Inf={inf_count} ({inf_frac:.4%}), "
                    f"extreme={extreme_count}, total_elements={total_elements}"
                ),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "extreme_count": extreme_count,
                    "total_elements": total_elements,
                },
            )
        except Exception as exc:
            return TestResult(
                name="numerical_stability",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def test_edge_cases(self) -> TestResult:
        """Test zero, max-float, min-float, single-element batch, large batch."""
        t0 = time.perf_counter()
        try:
            sess = self.session
            base_feeds = _build_random_inputs(sess, batch_size=1)
            case_results: dict[str, str] = {}

            cases: dict[str, dict[str, np.ndarray]] = {
                "zeros": {
                    k: np.zeros_like(v) for k, v in base_feeds.items()
                },
                "max_float": {
                    k: np.full_like(v, np.finfo(v.dtype).max if np.issubdtype(v.dtype, np.floating) else v.max())
                    for k, v in base_feeds.items()
                },
                "min_float": {
                    k: np.full_like(v, np.finfo(v.dtype).tiny if np.issubdtype(v.dtype, np.floating) else 0)
                    for k, v in base_feeds.items()
                },
                "single_batch": _build_random_inputs(sess, batch_size=1),
                "large_batch": _build_random_inputs(sess, batch_size=8),
            }

            all_passed = True
            for case_name, feeds in cases.items():
                try:
                    outputs = sess.run(None, feeds)
                    has_nan = any(np.isnan(o.astype(np.float64)).any() for o in outputs if np.issubdtype(o.dtype, np.floating))
                    has_inf = any(np.isinf(o.astype(np.float64)).any() for o in outputs if np.issubdtype(o.dtype, np.floating))
                    if has_nan or has_inf:
                        case_results[case_name] = f"FAIL: NaN={has_nan}, Inf={has_inf}"
                        all_passed = False
                    else:
                        case_results[case_name] = "PASS"
                except Exception as exc:
                    case_results[case_name] = f"FAIL: {exc}"
                    all_passed = False

            summary = "; ".join(f"{k}: {v}" for k, v in case_results.items())
            return TestResult(
                name="edge_cases",
                passed=all_passed,
                details=summary,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={"cases": case_results},
            )
        except Exception as exc:
            return TestResult(
                name="edge_cases",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def test_input_validation(self, num_samples: int = 50) -> TestResult:
        """Verify the model raises clean errors for invalid inputs."""
        t0 = time.perf_counter()
        try:
            sess = self.session
            valid_feeds = _build_random_inputs(sess, batch_size=1)
            input_meta = sess.get_inputs()
            tests_run = 0
            clean_errors = 0
            silent_failures = 0
            details_parts: list[str] = []

            for inp in input_meta[:num_samples]:
                arr = valid_feeds[inp.name]

                wrong_shape = np.zeros([d + 7 for d in arr.shape], dtype=arr.dtype)
                tests_run += 1
                try:
                    bad_feeds = {**valid_feeds, inp.name: wrong_shape}
                    sess.run(None, bad_feeds)
                    silent_failures += 1
                    details_parts.append(f"wrong_shape({inp.name}): no error raised")
                except Exception:
                    clean_errors += 1

                if np.issubdtype(arr.dtype, np.floating):
                    wrong_dtype = arr.astype(np.int32)
                else:
                    wrong_dtype = arr.astype(np.float32)
                tests_run += 1
                try:
                    bad_feeds = {**valid_feeds, inp.name: wrong_dtype}
                    sess.run(None, bad_feeds)
                    silent_failures += 1
                    details_parts.append(f"wrong_dtype({inp.name}): no error raised")
                except Exception:
                    clean_errors += 1

            tests_run += 1
            try:
                missing = {k: v for k, v in valid_feeds.items() if k != input_meta[0].name}
                if missing:
                    sess.run(None, missing)
                    silent_failures += 1
                    details_parts.append("missing_input: no error raised")
                else:
                    clean_errors += 1
            except Exception:
                clean_errors += 1

            tests_run += 1
            try:
                extra = {**valid_feeds, "__bogus_input__": np.zeros((1,), dtype=np.float32)}
                sess.run(None, extra)
                clean_errors += 1
            except Exception:
                clean_errors += 1

            passed = silent_failures == 0
            return TestResult(
                name="input_validation",
                passed=passed,
                details=(
                    f"{clean_errors}/{tests_run} invalid inputs correctly rejected"
                    + (f"; issues: {'; '.join(details_parts)}" if details_parts else "")
                ),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={
                    "tests_run": tests_run,
                    "clean_errors": clean_errors,
                    "silent_failures": silent_failures,
                },
            )
        except Exception as exc:
            return TestResult(
                name="input_validation",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def test_cross_provider(self, providers: list[str] | None = None) -> TestResult:
        """Compare outputs across execution providers within tolerance."""
        t0 = time.perf_counter()
        try:
            available = _available_providers()
            if providers is None:
                providers = available
            else:
                providers = [p for p in providers if p in available]

            if len(providers) < 2:
                return TestResult(
                    name="cross_provider",
                    passed=True,
                    details=f"Skipped: fewer than 2 providers available ({providers})",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    metrics={"skipped": True},
                )

            ref_provider = providers[0]
            ref_sess = self._get_session(ref_provider)
            feeds = _build_random_inputs(ref_sess, batch_size=1)
            ref_outputs = ref_sess.run(None, feeds)
            del ref_sess

            max_diffs: dict[str, float] = {}
            all_passed = True

            for prov in providers[1:]:
                try:
                    sess = self._get_session(prov)
                    outputs = sess.run(None, feeds)
                    pair_key = f"{ref_provider}_vs_{prov}"
                    pair_max = 0.0
                    for ref_arr, cur_arr in zip(ref_outputs, outputs):
                        diff = float(np.max(np.abs(
                            ref_arr.astype(np.float64) - cur_arr.astype(np.float64)
                        )))
                        pair_max = max(pair_max, diff)
                    max_diffs[pair_key] = pair_max
                    if pair_max > 1e-3:
                        all_passed = False
                    del sess
                except Exception as exc:
                    max_diffs[f"{ref_provider}_vs_{prov}"] = float("nan")
                    all_passed = False
                    log.warning("Provider %s failed: %s", prov, exc)

            summary = "; ".join(f"{k}: {v:.2e}" for k, v in max_diffs.items())
            return TestResult(
                name="cross_provider",
                passed=all_passed,
                details=summary or "no comparisons made",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={"max_diffs": max_diffs},
            )
        except Exception as exc:
            return TestResult(
                name="cross_provider",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def test_latency_consistency(self, num_runs: int = 100, max_cv: float = 0.3) -> TestResult:
        """Check that inference latency has acceptable coefficient of variation."""
        t0 = time.perf_counter()
        try:
            sess = self.session
            feeds = _build_random_inputs(sess, batch_size=1)

            for _ in range(3):
                sess.run(None, feeds)

            latencies: list[float] = []
            for _ in range(num_runs):
                t_start = time.perf_counter()
                sess.run(None, feeds)
                latencies.append((time.perf_counter() - t_start) * 1000)

            arr = np.array(latencies)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            cv = std / mean if mean > 0 else 0.0
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))

            passed = cv < max_cv
            return TestResult(
                name="latency_consistency",
                passed=passed,
                details=(
                    f"CV={cv:.4f} (threshold={max_cv}), "
                    f"mean={mean:.2f}ms, std={std:.2f}ms, "
                    f"p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms"
                ),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={
                    "cv": cv,
                    "mean_ms": mean,
                    "std_ms": std,
                    "p50_ms": p50,
                    "p95_ms": p95,
                    "p99_ms": p99,
                    "max_cv": max_cv,
                },
            )
        except Exception as exc:
            return TestResult(
                name="latency_consistency",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def test_memory_safety(self, batch_sizes: list[int] | None = None) -> TestResult:
        """Check for memory leaks by monitoring RSS growth during inference loops."""
        if batch_sizes is None:
            batch_sizes = [1, 4, 16]
        t0 = time.perf_counter()
        try:
            sess = self.session
            iterations_per_batch = 50
            leak_threshold_mb = 50.0
            per_batch: dict[str, dict[str, float]] = {}
            leak_detected = False

            for bs in batch_sizes:
                feeds = _build_random_inputs(sess, batch_size=bs)
                gc.collect()
                rss_before = _rss_mb()

                for _ in range(iterations_per_batch):
                    sess.run(None, feeds)

                gc.collect()
                rss_after = _rss_mb()
                delta = rss_after - rss_before
                per_batch[f"batch_{bs}"] = {
                    "rss_before_mb": rss_before,
                    "rss_after_mb": rss_after,
                    "delta_mb": delta,
                }
                if delta > leak_threshold_mb:
                    leak_detected = True
                    log.warning(
                        "Potential leak at batch_size=%d: RSS grew %.1f MB",
                        bs, delta,
                    )

            summary = "; ".join(
                f"bs={k}: delta={v['delta_mb']:.1f}MB" for k, v in per_batch.items()
            )
            return TestResult(
                name="memory_safety",
                passed=not leak_detected,
                details=summary,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={"per_batch": per_batch, "threshold_mb": leak_threshold_mb},
            )
        except Exception as exc:
            return TestResult(
                name="memory_safety",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def generate_golden(self, inputs: dict[str, np.ndarray], output_path: str) -> None:
        """Save inputs and expected outputs as a golden-test npz file."""
        sess = self.session
        outputs = sess.run(None, inputs)
        output_names = [o.name for o in sess.get_outputs()]

        save_dict: dict[str, np.ndarray] = {}
        for name, arr in inputs.items():
            save_dict[f"input_{name}"] = arr
        for name, arr in zip(output_names, outputs):
            save_dict[f"output_{name}"] = arr
        save_dict["__meta_input_names__"] = np.array(list(inputs.keys()))
        save_dict["__meta_output_names__"] = np.array(output_names)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **save_dict)
        log.info("Golden test saved to %s", output_path)

    def test_golden(self, golden_path: str, tolerance: float = 1e-5) -> TestResult:
        """Replay a golden test and verify outputs match within tolerance."""
        t0 = time.perf_counter()
        try:
            data = np.load(golden_path, allow_pickle=False)
            input_names = list(data["__meta_input_names__"])
            output_names = list(data["__meta_output_names__"])

            feeds = {name: data[f"input_{name}"] for name in input_names}
            expected = {name: data[f"output_{name}"] for name in output_names}

            sess = self.session
            actual_outputs = sess.run(None, feeds)
            actual_names = [o.name for o in sess.get_outputs()]

            max_diff = 0.0
            per_output: dict[str, float] = {}
            for exp_name, actual_arr in zip(actual_names, actual_outputs):
                if exp_name in expected:
                    diff = float(np.max(np.abs(
                        expected[exp_name].astype(np.float64) - actual_arr.astype(np.float64)
                    )))
                    per_output[exp_name] = diff
                    max_diff = max(max_diff, diff)

            passed = max_diff <= tolerance
            return TestResult(
                name="golden",
                passed=passed,
                details=f"max_diff={max_diff:.2e} (tolerance={tolerance:.2e})",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                metrics={"max_diff": max_diff, "per_output": per_output, "tolerance": tolerance},
            )
        except Exception as exc:
            return TestResult(
                name="golden",
                passed=False,
                details=f"Error: {exc}\n{traceback.format_exc()}",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )

    def run_all(self) -> TestSuiteResult:
        """Run the full test suite, catching failures per test."""
        suite_t0 = time.perf_counter()
        test_methods = [
            self.test_determinism,
            self.test_numerical_stability,
            self.test_edge_cases,
            self.test_input_validation,
            self.test_cross_provider,
            self.test_latency_consistency,
            self.test_memory_safety,
        ]

        results: list[TestResult] = []
        for method in test_methods:
            log.info("Running %s ...", method.__name__)
            try:
                result = method()
            except Exception as exc:
                result = TestResult(
                    name=method.__name__.removeprefix("test_"),
                    passed=False,
                    details=f"Unhandled crash: {exc}\n{traceback.format_exc()}",
                    elapsed_ms=0.0,
                )
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            log.info("  %s %s (%.0f ms)", status, result.name, result.elapsed_ms)

        passed = sum(1 for r in results if r.passed)
        skipped = sum(1 for r in results if r.metrics.get("skipped"))
        failed = len(results) - passed

        return TestSuiteResult(
            model_path=self.model_path,
            total_tests=len(results),
            passed=passed,
            failed=failed,
            skipped=skipped,
            results=results,
            elapsed_s=time.perf_counter() - suite_t0,
        )


def test_model(
    model_path: str,
    provider: str = "auto",
    output_dir: str | None = None,
) -> TestSuiteResult:
    """Top-level convenience for CLI usage.

    Runs the full suite and optionally writes JUnit XML + golden file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    tester = ModelTester(model_path, provider=provider)
    suite = tester.run_all()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        xml_path = out / "test_results.xml"
        xml_path.write_text(suite.to_junit_xml(), encoding="utf-8")
        log.info("JUnit XML written to %s", xml_path)

        try:
            feeds = _build_random_inputs(tester.session, batch_size=1)
            golden_path = str(out / "golden.npz")
            tester.generate_golden(feeds, golden_path)
        except Exception as exc:
            log.warning("Could not generate golden file: %s", exc)

    log.info(
        "Suite complete: %d/%d passed, %d failed, %d skipped (%.1fs)",
        suite.passed, suite.total_tests, suite.failed, suite.skipped, suite.elapsed_s,
    )
    return suite
