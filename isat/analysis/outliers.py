"""Outlier detection for benchmark latencies.

Implements Modified Z-Score (MAD-based) and IQR methods to
identify and remove spurious measurements caused by system
interference, thermal throttling, or GC pauses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OutlierReport:
    original_n: int
    cleaned_n: int
    n_outliers: int
    outlier_indices: list[int]
    outlier_values: list[float]
    method: str
    threshold: float


def detect_outliers_mad(values: list[float], threshold: float = 3.5) -> OutlierReport:
    """Modified Z-Score using Median Absolute Deviation.

    More robust than standard Z-score because it uses median instead of mean,
    making it resistant to the outliers it's trying to detect.
    """
    arr = np.array(values, dtype=np.float64)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))

    if mad == 0:
        return OutlierReport(len(values), len(values), 0, [], [], "mad", threshold)

    modified_z = 0.6745 * (arr - median) / mad
    outlier_mask = np.abs(modified_z) > threshold
    indices = list(np.where(outlier_mask)[0])

    return OutlierReport(
        original_n=len(values),
        cleaned_n=len(values) - len(indices),
        n_outliers=len(indices),
        outlier_indices=indices,
        outlier_values=[values[i] for i in indices],
        method="mad",
        threshold=threshold,
    )


def detect_outliers_iqr(values: list[float], k: float = 1.5) -> OutlierReport:
    """Interquartile Range method."""
    arr = np.array(values, dtype=np.float64)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outlier_mask = (arr < lower) | (arr > upper)
    indices = list(np.where(outlier_mask)[0])

    return OutlierReport(
        original_n=len(values),
        cleaned_n=len(values) - len(indices),
        n_outliers=len(indices),
        outlier_indices=indices,
        outlier_values=[values[i] for i in indices],
        method="iqr",
        threshold=k,
    )


def detect_outliers(
    values: list[float],
    method: str = "mad",
    threshold: float = 3.5,
) -> OutlierReport:
    """Detect outliers using the specified method."""
    if method == "mad":
        return detect_outliers_mad(values, threshold)
    elif method == "iqr":
        return detect_outliers_iqr(values, threshold)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mad' or 'iqr'.")


def remove_outliers(
    values: list[float],
    method: str = "mad",
    threshold: float = 3.5,
) -> tuple[list[float], OutlierReport]:
    """Remove outliers and return cleaned values + report."""
    report = detect_outliers(values, method, threshold)
    outlier_set = set(report.outlier_indices)
    cleaned = [v for i, v in enumerate(values) if i not in outlier_set]
    return cleaned, report
