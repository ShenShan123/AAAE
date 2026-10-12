"""A1 pool-weighted empirical upper correction (paper Section 2.4).

The positive residual and weighted quantile follow the E7 implementation.
This is an empirical rule, without a finite-sample coverage guarantee. Fit a
new correction when the candidate pool or its stratification changes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def _vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return array


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Return the original E7 weighted empirical quantile, without interpolation."""
    values = _vector(values, "values")
    weights = _vector(weights, "weights")
    if values.shape != weights.shape or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("weights must align with values and have positive total weight")
    if not 0.0 <= level <= 1.0:
        raise ValueError("quantile level must be in [0, 1]")
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    w = w / w.sum()
    cumulative = np.cumsum(w)
    index = int(np.searchsorted(cumulative, level, side="left"))
    if index >= len(v):
        return float("inf")
    return float(v[index])


@dataclass(frozen=True)
class EmpiricalUpperCorrection:
    alpha: float
    correction_pp: float
    calibration_count: int
    pool_count: int
    stratum_shares: dict[str, dict[str, float]]
    rule: str = "A1_pool_weighted_positive_residual"
    units: str = "percentage_points"

    def upper(self, prediction_pp: Any) -> np.ndarray:
        prediction = np.asarray(prediction_pp, dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise ValueError("predictions must be finite")
        return prediction + self.correction_pp

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_upper_correction(
    actual_calibration_pp: Any,
    predicted_calibration_pp: Any,
    calibration_bins: Any,
    pool_bins: Any,
    *,
    alpha: float = 0.10,
) -> EmpiricalUpperCorrection:
    """Fit A1 using independent calibration labels and pool covariates only.

    All accuracy values are percentage points, not fractions. Bins are the
    frozen Swin s+c proxy strata used by the original calibration protocol.
    Target-pool accuracy labels are not an input to this function.
    """
    actual = _vector(actual_calibration_pp, "actual calibration drops")
    predicted = _vector(predicted_calibration_pp, "predicted calibration drops")
    cal_bins = _vector(calibration_bins, "calibration bins")
    target_bins = _vector(pool_bins, "pool bins")
    if not actual.shape == predicted.shape == cal_bins.shape:
        raise ValueError("calibration drops, predictions and bins must align")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")

    positive = np.maximum(actual - predicted, 0.0)
    weight = np.ones(len(actual), dtype=float)
    for bin_index in np.unique(cal_bins):
        cal_share = float((cal_bins == bin_index).mean())
        pool_share = float((target_bins == bin_index).mean())
        weight[cal_bins == bin_index] = (pool_share + 1e-12) / (cal_share + 1e-12)
    shares = {
        f"bin{bin_index:g}": {
            "calibration_share": float((cal_bins == bin_index).mean()),
            "pool_share": float((target_bins == bin_index).mean()),
        }
        for bin_index in sorted(set(cal_bins.tolist()) | set(target_bins.tolist()))
    }
    return EmpiricalUpperCorrection(
        alpha=float(alpha),
        correction_pp=weighted_quantile(positive, weight, 1.0 - alpha),
        calibration_count=len(actual),
        pool_count=len(target_bins),
        stratum_shares=shares,
    )
