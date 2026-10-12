"""AAAE fixed-pool calibrated admission and budgeted selection.

Section 2.4 and Appendix D: admit with U <= tau; order admitted candidates by
predicted PDP and configuration hash; query up to K full-model labels; select
the feasible queried design with lowest predicted PDP. Selection does not
construct an iterative Pareto front, and later verification does not change it.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np


def admission_order(
    prediction_pp: Sequence[float],
    predicted_pdp_fj: Sequence[float],
    config_hashes: Sequence[str],
    *,
    correction_pp: float,
    tau_pp: float,
) -> np.ndarray:
    prediction = np.asarray(prediction_pp, dtype=float)
    pdp = np.asarray(predicted_pdp_fj, dtype=float)
    hashes = np.asarray(config_hashes, dtype=str)
    if prediction.ndim != 1 or pdp.shape != prediction.shape or hashes.shape != prediction.shape:
        raise ValueError("predictions, PDP and configuration hashes must be aligned vectors")
    if not np.isfinite(prediction).all() or not np.isfinite(pdp).all():
        raise ValueError("predictions and PDP must be finite")
    if np.any(pdp <= 0):
        raise ValueError("predicted PDP must be positive")
    if len(set(hashes.tolist())) != len(hashes) or np.any(hashes == ""):
        raise ValueError("configuration hashes must be nonempty and unique")
    if np.isnan(correction_pp) or correction_pp < 0 or not np.isfinite(tau_pp):
        raise ValueError("correction must be nonnegative and the budget threshold finite")
    admitted = np.flatnonzero(prediction + correction_pp <= tau_pp)
    order = sorted(range(len(admitted)), key=lambda i: (pdp[admitted[i]], hashes[admitted[i]]))
    return admitted[np.asarray(order, dtype=int)]


@dataclass(frozen=True)
class SelectionResult:
    admitted_positions: tuple[int, ...]
    queried_positions: tuple[int, ...]
    observed_drop_pp: tuple[float, ...]
    selected_position: int | None
    status: str


def select_with_budget(
    prediction_pp: Sequence[float],
    predicted_pdp_fj: Sequence[float],
    config_hashes: Sequence[str],
    evaluate_drop_pp: Callable[[int], float],
    *,
    correction_pp: float,
    tau_pp: float,
    budget: int,
) -> SelectionResult:
    """Query each chosen candidate once; return the frozen single-run choice."""
    if isinstance(budget, bool) or int(budget) != budget or budget <= 0:
        raise ValueError("the full-model query budget must be a positive integer")
    admitted = admission_order(
        prediction_pp, predicted_pdp_fj, config_hashes,
        correction_pp=correction_pp, tau_pp=tau_pp,
    )
    queried = admitted[:int(budget)]
    observations = tuple(float(evaluate_drop_pp(int(position))) for position in queried)
    if not np.isfinite(observations).all():
        raise ValueError("accuracy queries must return finite drops in percentage points")
    feasible = [int(position) for position, drop in zip(queried, observations) if drop <= tau_pp]
    return SelectionResult(
        admitted_positions=tuple(int(position) for position in admitted),
        queried_positions=tuple(int(position) for position in queried),
        observed_drop_pp=observations,
        selected_position=feasible[0] if feasible else None,
        status="selected" if feasible else ("abstention" if not len(admitted) else "failed_selection"),
    )
