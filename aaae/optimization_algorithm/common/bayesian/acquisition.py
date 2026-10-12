from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.utils import prune_inferior_points_multi_objective
from botorch.exceptions.warnings import InputDataWarning, OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import SumMarginalLogLikelihood

from .candidate_pool import TokenSet, tokens_to_feature_matrix


OBJECTIVE_COLS = ("pdp_fj", "top1_drop", "area_um2")


@dataclass(frozen=True)
class AcquisitionChoice:
    tokens: TokenSet
    config_hash: str | None
    acquisition_name: str
    acquisition_value: float
    pred_mu_top1_drop: float
    pred_sigma_top1_drop: float
    pred_mu_pdp_fj: float
    pred_sigma_pdp_fj: float
    pred_mu_area_um2: float
    pred_sigma_area_um2: float
    rank_in_pool: int
    pool_size: int
    weight_pdp: float
    weight_top1: float
    weight_area: float


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (arr - mean[None, :]) / std[None, :], mean, std


def _minmax_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    min_val = arr.min(axis=0)
    span = arr.max(axis=0) - min_val
    span = np.where(span < 1e-12, 1.0, span)
    return (arr - min_val[None, :]) / span[None, :], min_val, span


def _feature_matrix_from_rows(rows: pd.DataFrame) -> np.ndarray:
    tokens = [tuple(int(row[f"low_{idx}"]) for idx in range(9)) for _, row in rows.iterrows()]
    token_features = tokens_to_feature_matrix(tokens)
    cheap_cols = [
        col
        for col in (
            "delay_ns",
            "power_uw",
            "pdp_fj",
            "final_truth_mae",
            "final_truth_max_abs",
        )
        if col in rows.columns
    ]
    if not cheap_cols:
        return token_features
    cheap = rows[cheap_cols].to_numpy(dtype=np.float64)
    cheap, _mean, _std = _standardize(cheap)
    return np.concatenate([token_features, cheap], axis=1)


def _tokens_to_feature_matrix_with_reference(tokens: list[TokenSet], reference_rows: pd.DataFrame | None = None) -> np.ndarray:
    # For acquisition-time pool candidates, cheap features may not have been
    # computed yet. Keep the representation token-only for stability.
    del reference_rows
    return tokens_to_feature_matrix(tokens)


def _target_bin_choice_from_predictions(
    *,
    pool_tokens: list[TokenSet],
    selected_indices: np.ndarray,
    eval_tokens: list[TokenSet],
    acq_values: np.ndarray,
    means_std: np.ndarray,
    vars_std: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    target_low: float,
    target_high: float,
    valid_drop_min: float | None = None,
    valid_drop_max: float | None = None,
    seed: int,
) -> AcquisitionChoice:
    mean_max = means_std * y_scale[None, :] + y_mean[None, :]
    mean_min = -mean_max
    std_min = np.abs(np.sqrt(np.maximum(vars_std, 0.0)) * y_scale[None, :])

    pred_pdp = mean_min[:, 0]
    pred_drop = mean_min[:, 1]
    target_mid = (float(target_low) + float(target_high)) / 2.0
    valid = np.ones(pred_drop.shape, dtype=bool)
    if valid_drop_min is not None:
        valid &= pred_drop >= float(valid_drop_min)
    if valid_drop_max is not None:
        valid &= pred_drop <= float(valid_drop_max)
    if not np.any(valid):
        valid = np.ones(pred_drop.shape, dtype=bool)
    in_bin = valid & (pred_drop >= float(target_low)) & (pred_drop < float(target_high))
    if np.any(in_bin):
        local_indices = np.flatnonzero(in_bin)
        order = np.lexsort((np.abs(pred_drop[local_indices] - target_mid), -acq_values[local_indices]))
        best_local_idx = int(local_indices[int(order[0])])
    else:
        local_indices = np.flatnonzero(valid)
        distance = np.abs(pred_drop[local_indices] - target_mid)
        # Distance locates the requested bin; three-objective acquisition
        # resolves close candidates without privileging either PPA metric.
        local_acq = acq_values[local_indices]
        acq_span = max(float(np.nanmax(local_acq) - np.nanmin(local_acq)), 1e-12)
        score = distance - 0.01 * (local_acq - np.nanmin(local_acq)) / acq_span
        rng = np.random.default_rng(int(seed))
        score = score + rng.uniform(0.0, 1e-12, size=score.shape)
        best_local_idx = int(local_indices[int(np.nanargmin(score))])

    original_idx = int(selected_indices[best_local_idx])
    best_tokens = pool_tokens[original_idx]
    return AcquisitionChoice(
        tokens=best_tokens,
        config_hash=None,
        acquisition_name="botorch_target_top1drop_bin_discrete_pool",
        acquisition_value=float(acq_values[best_local_idx]),
        pred_mu_top1_drop=float(pred_drop[best_local_idx]),
        pred_sigma_top1_drop=float(std_min[best_local_idx, 1]),
        pred_mu_pdp_fj=float(pred_pdp[best_local_idx]),
        pred_sigma_pdp_fj=float(std_min[best_local_idx, 0]),
        pred_mu_area_um2=float(mean_min[best_local_idx, 2]),
        pred_sigma_area_um2=float(std_min[best_local_idx, 2]),
        rank_in_pool=0,
        pool_size=len(pool_tokens),
        weight_pdp=float("nan"),
        weight_top1=float("nan"),
        weight_area=float("nan"),
    )


def _random_scalarized_choice(
    *,
    observed: pd.DataFrame,
    pool_tokens: list[TokenSet],
    seed: int,
) -> AcquisitionChoice:
    rng = np.random.default_rng(int(seed))
    weight = rng.dirichlet(np.ones(len(OBJECTIVE_COLS)))
    scores = []
    observed_tokens = {
        tuple(int(row[f"low_{idx}"]) for idx in range(9))
        for _, row in observed.iterrows()
    } if not observed.empty else set()
    for tokens in pool_tokens:
        arr = np.asarray(tokens, dtype=np.float64)
        mean_token = float(arr.mean())
        roughness = float(np.mean(np.abs(np.diff(arr))))
        synthetic_pdp = 1.0 - (mean_token - 1.0) / 7.0
        synthetic_drop = (mean_token - 1.0) / 7.0 + 0.05 * roughness
        synthetic_area = 0.5 * (1.0 - (mean_token - 1.0) / 7.0) + 0.5 * roughness / 7.0
        novelty = 0.0 if tokens in observed_tokens else rng.uniform(0.0, 0.01)
        score = -float(weight[0] * synthetic_pdp + weight[1] * synthetic_drop + weight[2] * synthetic_area) + novelty
        scores.append(score)
    order = np.argsort(np.asarray(scores))[::-1]
    best_idx = int(order[0])
    return AcquisitionChoice(
        tokens=pool_tokens[best_idx],
        config_hash=None,
        acquisition_name="random_scalarized_warmup",
        acquisition_value=float(scores[best_idx]),
        pred_mu_top1_drop=float("nan"),
        pred_sigma_top1_drop=float("nan"),
        pred_mu_pdp_fj=float("nan"),
        pred_sigma_pdp_fj=float("nan"),
        pred_mu_area_um2=float("nan"),
        pred_sigma_area_um2=float("nan"),
        rank_in_pool=0,
        pool_size=len(pool_tokens),
        weight_pdp=float(weight[0]),
        weight_top1=float(weight[1]),
        weight_area=float(weight[2]),
    )


def _random_scalarized_choices(
    *,
    observed: pd.DataFrame,
    pool_tokens: list[TokenSet],
    seed: int,
    batch_size: int,
) -> list[AcquisitionChoice]:
    choices: list[AcquisitionChoice] = []
    working_pool = list(pool_tokens)
    for offset in range(max(1, int(batch_size))):
        if not working_pool:
            break
        choice = _random_scalarized_choice(
            observed=observed,
            pool_tokens=working_pool,
            seed=int(seed) + offset,
        )
        choices.append(choice)
        working_pool = [tokens for tokens in working_pool if tokens != choice.tokens]
    return choices


def _posterior_min_stats(
    *,
    means_std: np.ndarray,
    vars_std: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_max = means_std * y_scale[None, :] + y_mean[None, :]
    std_max = np.sqrt(np.maximum(vars_std, 0.0)) * y_scale[None, :]
    return -mean_max, np.abs(std_max)


def _valid_prediction_mask(
    *,
    acq_values: np.ndarray,
    pred_pdp: np.ndarray,
    pred_drop: np.ndarray,
    pred_area: np.ndarray,
    valid_top1_drop_range: tuple[float | None, float | None] | None,
) -> np.ndarray:
    finite = np.isfinite(acq_values) & np.isfinite(pred_pdp) & np.isfinite(pred_drop) & np.isfinite(pred_area)
    valid = finite.copy()
    if valid_top1_drop_range is not None:
        valid_low, valid_high = valid_top1_drop_range
        if valid_low is not None:
            valid &= pred_drop >= float(valid_low)
        if valid_high is not None:
            valid &= pred_drop <= float(valid_high)
    if np.any(valid):
        return valid
    if np.any(finite):
        return finite
    return np.ones(pred_drop.shape, dtype=bool)


def _mask_without_selected(mask: np.ndarray, selected_local_indices: set[int]) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    for idx in selected_local_indices:
        if 0 <= int(idx) < len(out):
            out[int(idx)] = False
    return out


def _select_acq_index(
    *,
    acq_values: np.ndarray,
    valid_mask: np.ndarray,
    selected_local_indices: set[int],
) -> tuple[int, int] | None:
    available = _mask_without_selected(valid_mask, selected_local_indices)
    if not np.any(available):
        available = _mask_without_selected(np.ones_like(valid_mask, dtype=bool), selected_local_indices)
    local_indices = np.flatnonzero(available)
    if len(local_indices) == 0:
        return None
    scores = np.nan_to_num(acq_values[local_indices], nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    order = np.argsort(scores)[::-1]
    rank = int(order[0])
    return int(local_indices[rank]), rank


def _select_target_bin_index(
    *,
    acq_values: np.ndarray,
    pred_drop: np.ndarray,
    valid_mask: np.ndarray,
    selected_local_indices: set[int],
    target_low: float,
    target_high: float,
    seed: int,
) -> tuple[int, int] | None:
    available = _mask_without_selected(valid_mask, selected_local_indices)
    if not np.any(available):
        available = _mask_without_selected(np.ones_like(valid_mask, dtype=bool), selected_local_indices)
    if not np.any(available):
        return None

    target_mid = (float(target_low) + float(target_high)) / 2.0
    in_bin = available & (pred_drop >= float(target_low)) & (pred_drop < float(target_high))
    if np.any(in_bin):
        local_indices = np.flatnonzero(in_bin)
        order = np.lexsort((np.abs(pred_drop[local_indices] - target_mid), -acq_values[local_indices]))
        rank = int(order[0])
        return int(local_indices[rank]), rank

    local_indices = np.flatnonzero(available)
    distance = np.abs(pred_drop[local_indices] - target_mid)
    acq = np.nan_to_num(acq_values[local_indices], nan=float(np.nanmin(acq_values[np.isfinite(acq_values)])) if np.any(np.isfinite(acq_values)) else 0.0)
    acq_span = max(float(np.nanmax(acq) - np.nanmin(acq)), 1e-12)
    # Distance to the requested drop bin is dominant; the three-objective
    # acquisition value resolves nearby candidates.
    score = (
        distance
        - 0.01 * (acq - np.nanmin(acq)) / acq_span
    )
    rng = np.random.default_rng(int(seed))
    score = score + rng.uniform(0.0, 1e-12, size=score.shape)
    order = np.argsort(score)
    rank = int(order[0])
    return int(local_indices[rank]), rank


def _prediction_choice(
    *,
    pool_tokens: list[TokenSet],
    selected_indices: np.ndarray,
    local_idx: int,
    rank: int,
    acquisition_name: str,
    acq_values: np.ndarray,
    mean_min: np.ndarray,
    std_min: np.ndarray,
) -> AcquisitionChoice:
    original_idx = int(selected_indices[int(local_idx)])
    return AcquisitionChoice(
        tokens=pool_tokens[original_idx],
        config_hash=None,
        acquisition_name=acquisition_name,
        acquisition_value=float(acq_values[int(local_idx)]),
        pred_mu_top1_drop=float(mean_min[int(local_idx), 1]),
        pred_sigma_top1_drop=float(std_min[int(local_idx), 1]),
        pred_mu_pdp_fj=float(mean_min[int(local_idx), 0]),
        pred_sigma_pdp_fj=float(std_min[int(local_idx), 0]),
        pred_mu_area_um2=float(mean_min[int(local_idx), 2]),
        pred_sigma_area_um2=float(std_min[int(local_idx), 2]),
        rank_in_pool=int(rank),
        pool_size=len(pool_tokens),
        weight_pdp=float("nan"),
        weight_top1=float("nan"),
        weight_area=float("nan"),
    )


def choose_next_candidates(
    *,
    observed: pd.DataFrame,
    pool_tokens: list[TokenSet],
    seed: int,
    batch_size: int,
    min_gp_points: int = 6,
    mc_samples: int = 64,
    max_pool_eval: int = 2048,
    gp_fit_maxiter: int = 100,
    acquisition_device: str | torch.device = "cpu",
    baseline_prune_samples: int | None = None,
    target_top1_drop_bins: list[tuple[float, float] | None] | None = None,
    valid_top1_drop_range: tuple[float | None, float | None] | None = None,
) -> list[AcquisitionChoice]:
    if not pool_tokens:
        raise ValueError("candidate pool is empty")
    batch_size = max(1, int(batch_size))
    gp_fit_maxiter = max(1, int(gp_fit_maxiter))
    if baseline_prune_samples is not None and int(baseline_prune_samples) < 0:
        raise ValueError("baseline_prune_samples must be non-negative when provided")
    device = torch.device(acquisition_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"acquisition device {device} was requested, but CUDA is unavailable")
        device_index = 0 if device.index is None else int(device.index)
        if device_index < 0 or device_index >= torch.cuda.device_count():
            raise ValueError(
                f"acquisition device {device} is unavailable; "
                f"visible CUDA device count is {torch.cuda.device_count()}"
            )
    if observed.empty or len(observed) < int(min_gp_points):
        return _random_scalarized_choices(
            observed=observed,
            pool_tokens=pool_tokens,
            seed=seed,
            batch_size=batch_size,
        )
    missing = set(OBJECTIVE_COLS).difference(observed.columns)
    if missing:
        return _random_scalarized_choices(
            observed=observed,
            pool_tokens=pool_tokens,
            seed=seed,
            batch_size=batch_size,
        )

    work = observed.copy()
    for col in OBJECTIVE_COLS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work[np.isfinite(work[list(OBJECTIVE_COLS)]).all(axis=1) & (work["area_um2"] > 0)].copy()
    if len(work) < int(min_gp_points):
        return _random_scalarized_choices(
            observed=observed,
            pool_tokens=pool_tokens,
            seed=seed,
            batch_size=batch_size,
        )

    rng = np.random.default_rng(int(seed))
    if len(pool_tokens) > int(max_pool_eval):
        selected_indices = rng.choice(len(pool_tokens), size=int(max_pool_eval), replace=False)
        eval_tokens = [pool_tokens[int(idx)] for idx in selected_indices]
    else:
        selected_indices = np.arange(len(pool_tokens))
        eval_tokens = pool_tokens

    train_X_np = tokens_to_feature_matrix(
        [tuple(int(row[f"low_{idx}"]) for idx in range(9)) for _, row in work.iterrows()]
    )
    pool_X_np = _tokens_to_feature_matrix_with_reference(eval_tokens, work)
    _train_X_np, x_min, x_span = _minmax_scale(train_X_np)
    pool_X_np = np.clip((pool_X_np - x_min[None, :]) / x_span[None, :], 0.0, 1.0)

    y_raw = work[list(OBJECTIVE_COLS)].to_numpy(dtype=np.float64)
    y_max = -y_raw
    y_std, y_mean, y_scale = _standardize(y_max)

    tkwargs = {"dtype": torch.double, "device": device}
    train_X = torch.as_tensor(_train_X_np, **tkwargs)
    train_Y = torch.as_tensor(y_std, **tkwargs)
    pool_X = torch.as_tensor(pool_X_np, **tkwargs)

    try:
        models = [SingleTaskGP(train_X, train_Y[:, idx : idx + 1]) for idx in range(train_Y.shape[1])]
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InputDataWarning)
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            fit_gpytorch_mll(
                mll,
                max_attempts=1,
                optimizer_kwargs={"options": {"maxiter": gp_fit_maxiter}},
            )

        ref_point = train_Y.min(dim=0).values - 0.1
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]), seed=int(seed))
        acq_baseline = train_X
        prune_baseline = baseline_prune_samples is None
        if baseline_prune_samples is not None and int(baseline_prune_samples) > 0:
            acq_baseline = prune_inferior_points_multi_objective(
                model=model,
                X=train_X,
                ref_point=ref_point,
                num_samples=int(baseline_prune_samples),
            )
            prune_baseline = False
        elif baseline_prune_samples is not None:
            prune_baseline = False
        acq = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=acq_baseline,
            sampler=sampler,
            prune_baseline=prune_baseline,
        )
        with torch.no_grad():
            values: list[torch.Tensor] = []
            posterior_means: list[torch.Tensor] = []
            posterior_vars: list[torch.Tensor] = []
            chunk = 256
            for start in range(0, pool_X.shape[0], chunk):
                x = pool_X[start : start + chunk]
                values.append(acq(x.unsqueeze(1)).detach().cpu())
                posterior = model.posterior(x)
                posterior_means.append(posterior.mean.detach().cpu())
                posterior_vars.append(posterior.variance.detach().cpu())
            acq_values = torch.cat(values, dim=0).numpy().reshape(-1)
            means_std = torch.cat(posterior_means, dim=0).numpy()
            vars_std = torch.cat(posterior_vars, dim=0).numpy()
    except Exception as exc:
        if device.type != "cpu":
            raise RuntimeError(f"GPU acquisition failed on {device}") from exc
        return _random_scalarized_choices(
            observed=observed,
            pool_tokens=pool_tokens,
            seed=seed,
            batch_size=batch_size,
        )

    mean_min, std_min = _posterior_min_stats(
        means_std=means_std,
        vars_std=vars_std,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    pred_pdp = mean_min[:, 0]
    pred_drop = mean_min[:, 1]
    pred_area = mean_min[:, 2]
    valid_mask = _valid_prediction_mask(
        acq_values=acq_values,
        pred_pdp=pred_pdp,
        pred_drop=pred_drop,
        pred_area=pred_area,
        valid_top1_drop_range=valid_top1_drop_range,
    )

    if target_top1_drop_bins is None:
        target_bins: list[tuple[float, float] | None] = [None] * batch_size
    else:
        target_bins = list(target_top1_drop_bins[:batch_size])
        while len(target_bins) < batch_size:
            target_bins.append(None)

    selected_local_indices: set[int] = set()
    choices: list[AcquisitionChoice] = []
    for offset, target_bin in enumerate(target_bins):
        if len(selected_local_indices) >= len(eval_tokens):
            break
        if target_bin is None:
            selected = _select_acq_index(
                acq_values=acq_values,
                valid_mask=valid_mask,
                selected_local_indices=selected_local_indices,
            )
            acquisition_name = "botorch_batch_qlognehvi_discrete_pool"
        else:
            selected = _select_target_bin_index(
                acq_values=acq_values,
                pred_drop=pred_drop,
                valid_mask=valid_mask,
                selected_local_indices=selected_local_indices,
                target_low=float(target_bin[0]),
                target_high=float(target_bin[1]),
                seed=int(seed) + offset,
            )
            acquisition_name = "botorch_batch_target_top1drop_bin_discrete_pool"
        if selected is None:
            break
        local_idx, rank = selected
        selected_local_indices.add(int(local_idx))
        choices.append(
            _prediction_choice(
                pool_tokens=pool_tokens,
                selected_indices=np.asarray(selected_indices),
                local_idx=int(local_idx),
                rank=int(rank),
                acquisition_name=acquisition_name,
                acq_values=acq_values,
                mean_min=mean_min,
                std_min=std_min,
            )
        )
    return choices


def choose_next_candidate(
    *,
    observed: pd.DataFrame,
    pool_tokens: list[TokenSet],
    seed: int,
    min_gp_points: int = 6,
    mc_samples: int = 64,
    max_pool_eval: int = 2048,
    gp_fit_maxiter: int = 100,
    acquisition_device: str | torch.device = "cpu",
    baseline_prune_samples: int | None = None,
    target_top1_drop_bin: tuple[float, float] | None = None,
    valid_top1_drop_range: tuple[float | None, float | None] | None = None,
) -> AcquisitionChoice:
    choices = choose_next_candidates(
        observed=observed,
        pool_tokens=pool_tokens,
        seed=seed,
        batch_size=1,
        min_gp_points=min_gp_points,
        mc_samples=mc_samples,
        max_pool_eval=max_pool_eval,
        gp_fit_maxiter=gp_fit_maxiter,
        acquisition_device=acquisition_device,
        baseline_prune_samples=baseline_prune_samples,
        target_top1_drop_bins=[target_top1_drop_bin],
        valid_top1_drop_range=valid_top1_drop_range,
    )
    if not choices:
        raise RuntimeError("failed to choose a candidate")
    return choices[0]
