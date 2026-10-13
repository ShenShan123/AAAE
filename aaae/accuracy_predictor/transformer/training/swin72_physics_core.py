from __future__ import annotations

from typing import Any


def joint_nonregression(
    baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]
) -> bool:
    """Require MAE/P99/Max/count non-regression and a real improvement."""

    names = ("mae", "p99_abs_error", "max_abs_error", "over_5pp_count")
    return all(
        float(candidate_metrics[name]) <= float(baseline_metrics[name])
        for name in names
    ) and any(
        float(candidate_metrics[name]) < float(baseline_metrics[name])
        for name in names
    )
