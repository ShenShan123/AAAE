"""Accuracy-drop metrics reported by AAAE, with explicit unit conversion."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def accuracy_drop_metrics(actual, predicted, *, units: str = "proportion") -> dict:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != predicted.shape or actual.size == 0:
        raise ValueError("actual and predicted must be nonempty aligned one-dimensional arrays")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("accuracy-drop values must be finite")
    if units not in {"proportion", "pp"}:
        raise ValueError("units must be proportion or pp")
    factor = 100.0 if units == "proportion" else 1.0
    error = np.abs(predicted - actual) * factor
    nonzero = actual != 0
    correlation = float(spearmanr(actual, predicted).statistic) if actual.size > 1 and np.ptp(actual) and np.ptp(predicted) else None
    return {
        "n_candidates": int(actual.size),
        "mae_pp": float(error.mean()),
        "spearman": correlation,
        "p95_abs_error_pp": float(np.quantile(error, 0.95)),
        "max_abs_error_pp": float(error.max()),
        "mape_pct": float(100 * np.mean(np.abs(predicted[nonzero]-actual[nonzero])/np.abs(actual[nonzero]))) if nonzero.any() else None,
    }
