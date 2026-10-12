from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd

from aaae.optimization_algorithm.common.pareto3 import nondominated_mask_3d


def build_pareto_frame(
    frame: pd.DataFrame,
    *,
    objective_cols: tuple[str, ...] = ("pdp_fj", "top1_drop", "area_um2"),
    tie_cols: Iterable[str] = ("config_hash",),
) -> pd.DataFrame:
    """Return a minimization Pareto front in O(n log n) for two or three objectives."""
    if frame.empty:
        return frame.copy()
    if len(objective_cols) not in (2, 3):
        raise ValueError("Pareto front requires two or three objective columns")
    missing = set(objective_cols).difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot build Pareto front; missing columns: {sorted(missing)}")

    order_cols = [*objective_cols, *[col for col in tie_cols if col in frame.columns]]
    work = frame.copy()
    for column in objective_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work[np.isfinite(work[list(objective_cols)]).all(axis=1)].copy()
    if work.empty:
        return work
    if "config_hash" in work.columns:
        work = work.sort_values(order_cols).drop_duplicates("config_hash", keep="first")
    work = work.sort_values(order_cols).drop_duplicates(list(objective_cols), keep="first").reset_index(drop=True)

    objectives = work[list(objective_cols)].to_numpy(dtype=np.float64)
    keep = np.zeros(len(objectives), dtype=bool)
    if len(objective_cols) == 2:
        best_second = math.inf
        for idx, second in enumerate(objectives[:, 1]):
            if second < best_second:
                keep[idx] = True
                best_second = second
    else:
        keep = nondominated_mask_3d(objectives)
    out = work.loc[keep].reset_index(drop=True).copy()
    out["rank"] = 0
    if "crowding_distance" not in out.columns:
        out["crowding_distance"] = np.nan
    return out


def assign_design_ids(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    sort_cols = [column for column in ("pdp_fj", "top1_drop", "area_um2", "config_hash") if column in frame.columns]
    out = frame.sort_values(sort_cols).reset_index(drop=True).copy()
    out["design_id"] = [f"{prefix}_{idx:06d}" for idx in range(len(out))]
    return out


def spaced_pareto_hashes(frame: pd.DataFrame, *, max_rows: int) -> list[str]:
    """Choose evenly spaced 3-D front members along each objective axis."""
    if frame.empty or "config_hash" not in frame.columns or max_rows <= 0:
        return []
    per_axis = max(1, min(128, int(max_rows) // 6))
    hashes: list[str] = []
    seen: set[str] = set()
    for column in ("pdp_fj", "top1_drop", "area_um2"):
        if column not in frame.columns:
            continue
        ordered = frame.sort_values([column, "config_hash"])["config_hash"].astype(str).to_numpy()
        positions = np.linspace(0, len(ordered) - 1, num=min(per_axis, len(ordered)), dtype=np.int64)
        for position in positions:
            key = str(ordered[position])
            if key not in seen:
                seen.add(key)
                hashes.append(key)
    return hashes


def top1drop_bin_lowers(*, drop_min: float = 0.0, drop_max: float = 0.50, step: float = 0.01) -> list[float]:
    if step <= 0:
        raise ValueError(f"drop bin step must be positive, got {step}")
    if drop_max <= drop_min:
        raise ValueError(f"drop bin max {drop_max} must be greater than min {drop_min}")
    count = int(round((float(drop_max) - float(drop_min)) / float(step)))
    if count <= 0:
        raise ValueError(f"no top1_drop bins for min={drop_min}, max={drop_max}, step={step}")
    return [round(float(drop_min) + idx * float(step), 10) for idx in range(count)]


def _drop_bin_id(low: float, high: float) -> str:
    return f"drop_{int(round(low * 100)):02d}_{int(round(high * 100)):02d}"


def build_top1drop_bin_pdp_curve(
    frame: pd.DataFrame,
    *,
    drop_min: float = 0.0,
    drop_max: float = 0.50,
    step: float = 0.01,
) -> pd.DataFrame:
    """Return the minimum-PDP evaluated design in each 1pp top1_drop bin.

    Bins are [0%,1%), [1%,2%), ..., [49%,50%]. The last bin includes its
    upper edge. The selected row is not necessarily globally non-dominated; it
    is the best evaluated PDP representative for that local top1_drop interval.
    """
    token_cols = [col for col in frame.columns if col.startswith("low_")]
    high_cols = [col for col in frame.columns if col.startswith("high_")]
    metric_cols = [
        "delay_ns",
        "power_uw",
        "pdp_fj",
        "area_um2",
        "top1_drop",
        "top5_drop",
        "exact_top1_acc",
        "approx_top1_acc",
        "exact_top5_acc",
        "approx_top5_acc",
        "top1_consistency",
        "top5_consistency",
        "logit_kl",
        "output_mse",
        "final_truth_mae",
        "final_truth_max_abs",
    ]
    columns = [
        "drop_bin_id",
        "target_top1_drop",
        "target_top1_drop_pct",
        "bin_low_top1_drop",
        "bin_high_top1_drop",
        "bin_low_top1_drop_pct",
        "bin_high_top1_drop_pct",
        "selected",
        "design_id",
        "config_hash",
        "bin_slack_low",
        "bin_slack_high",
        *[col for col in metric_cols if col in frame.columns],
        *token_cols,
        *high_cols,
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    missing = {"pdp_fj", "top1_drop"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot build top1_drop bin curve; missing columns: {sorted(missing)}")

    work = frame.copy()
    work["pdp_fj"] = pd.to_numeric(work["pdp_fj"], errors="coerce")
    work["top1_drop"] = pd.to_numeric(work["top1_drop"], errors="coerce")
    work = work[np.isfinite(work["pdp_fj"]) & np.isfinite(work["top1_drop"])].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["clipped_top1_drop"] = work["top1_drop"].clip(lower=float(drop_min), upper=float(drop_max))

    rows: list[dict[str, object]] = []
    eps = max(1e-12, float(step) * 1e-9)
    lowers = top1drop_bin_lowers(drop_min=drop_min, drop_max=drop_max, step=step)
    for low in lowers:
        high = round(float(low) + float(step), 10)
        is_last = math.isclose(high, float(drop_max), abs_tol=eps)
        if is_last:
            feasible = work[(work["clipped_top1_drop"] >= low - eps) & (work["clipped_top1_drop"] <= high + eps)].copy()
        else:
            feasible = work[(work["clipped_top1_drop"] >= low - eps) & (work["clipped_top1_drop"] < high - eps)].copy()

        base_row: dict[str, object] = {
            "drop_bin_id": _drop_bin_id(low, high),
            "target_top1_drop": float(low),
            "target_top1_drop_pct": float(low * 100.0),
            "bin_low_top1_drop": float(low),
            "bin_high_top1_drop": float(high),
            "bin_low_top1_drop_pct": float(low * 100.0),
            "bin_high_top1_drop_pct": float(high * 100.0),
            "selected": False,
        }
        if feasible.empty:
            rows.append(base_row)
            continue

        order_cols = ["pdp_fj", "clipped_top1_drop"]
        if "config_hash" in feasible.columns:
            order_cols.append("config_hash")
        best = feasible.sort_values(order_cols).iloc[0].to_dict()
        row = dict(base_row)
        selected_drop = float(best["clipped_top1_drop"])
        row.update(
            {
                "selected": True,
                "design_id": best.get("design_id"),
                "config_hash": best.get("config_hash"),
                "bin_slack_low": float(selected_drop - low),
                "bin_slack_high": float(high - selected_drop),
            }
        )
        for col in [*metric_cols, *token_cols, *high_cols]:
            if col not in best:
                continue
            value = best[col]
            if col.startswith(("low_", "high_")) and pd.notna(value):
                row[col] = int(value)
            else:
                row[col] = value
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def selected_top1drop_bin_front(curve_frame: pd.DataFrame) -> pd.DataFrame:
    if curve_frame.empty or "selected" not in curve_frame.columns:
        return curve_frame.iloc[0:0].copy()
    return curve_frame[curve_frame["selected"].astype(bool)].reset_index(drop=True).copy()
