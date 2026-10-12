from __future__ import annotations

import ast
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_DIM, FIXED_HIGH_TOKENS, HIGH_TOKEN_COUNT, LOW_TOKEN_COUNT


@lru_cache(maxsize=1)
def exact_product_table() -> np.ndarray:
    values = np.arange(256, dtype=np.int32)
    return (values[:, None] * values[None, :]).astype(np.int32)


def as_batch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[1:] != (256, 256):
        raise ValueError(f"Expected shape (256,256) or (N,256,256), got {arr.shape}")
    return arr


def truth_shape_features_from_delta(delta: np.ndarray, chunk_size: int = 128) -> np.ndarray:
    delta = as_batch(np.asarray(delta, dtype=np.float32))
    out = np.empty((delta.shape[0], 3072), dtype=np.float32)
    for start in range(0, delta.shape[0], chunk_size):
        d = delta[start : start + chunk_size]
        pieces: list[np.ndarray] = []
        for arr in (d, np.abs(d), np.maximum(d, 0.0), np.maximum(-d, 0.0)):
            pieces.append(arr.mean(axis=2))
            pieces.append(arr.mean(axis=1))
            pieces.append(arr.reshape(arr.shape[0], 16, 16, 16, 16).mean(axis=(2, 4)).reshape(arr.shape[0], -1))
        feat = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
        out[start : start + len(d)] = (np.sign(feat) * np.log1p(np.abs(feat))).astype(np.float32, copy=False)
    return out


def _exact_table_for_metrics(exact_table: np.ndarray | None) -> np.ndarray:
    """Return a validated exact-product table for the 12 scalar metrics.

    The retained U8 pipeline intentionally keeps its historical default.  A
    caller evaluating a different Q/DQ code domain can provide its matching
    256x256 reference table without changing the truth-shape feature layout.
    """
    if exact_table is None:
        return exact_product_table().astype(np.float32)
    exact = np.asarray(exact_table)
    if exact.shape != (256, 256):
        raise ValueError(f"exact_table must have shape (256,256), got {exact.shape}")
    if not np.issubdtype(exact.dtype, np.number):
        raise TypeError(f"exact_table must be numeric, got {exact.dtype}")
    exact = np.asarray(exact, dtype=np.float32)
    if not np.isfinite(exact).all():
        raise ValueError("exact_table must contain only finite values")
    if float(np.max(np.abs(exact), initial=0.0)) <= 0.0:
        raise ValueError("exact_table must contain at least one nonzero value")
    return exact


def metric12_from_delta(
    delta: np.ndarray,
    *,
    exact_table: np.ndarray | None = None,
) -> np.ndarray:
    d = as_batch(np.asarray(delta, dtype=np.float32))
    exact = _exact_table_for_metrics(exact_table)
    denom = np.where(exact == 0.0, np.nan, exact)
    rel = d / denom[None, :, :]
    abs_d = np.abs(d)
    abs_rel = np.abs(rel)
    max_exact_product = float(np.max(np.abs(exact)))
    rows: list[np.ndarray] = [
        np.nanmean(d, axis=(1, 2)),
        np.nanstd(d, axis=(1, 2)),
        np.nanmean(abs_d, axis=(1, 2)),
        np.nanmean(abs_d, axis=(1, 2)) / max_exact_product,
        np.nanstd(abs_d, axis=(1, 2)),
        np.nanmean(rel, axis=(1, 2)),
        np.nanstd(rel, axis=(1, 2)),
        np.nanmean(abs_rel, axis=(1, 2)),
        np.nanstd(abs_rel, axis=(1, 2)),
        np.sqrt(np.nanmean(d * d, axis=(1, 2))),
        np.sqrt(np.nanmean(rel * rel, axis=(1, 2))),
        np.nanmean(d != 0.0, axis=(1, 2)),
    ]
    metrics = np.stack(rows, axis=1).astype(np.float32)
    metrics = np.where(np.isfinite(metrics), metrics, 0.0).astype(np.float32)
    return (np.sign(metrics) * np.log1p(np.abs(metrics))).astype(np.float32)


def parse_tokens(value: Any, length: int) -> list[float]:
    if value is None:
        return [0.0] * length
    if isinstance(value, float) and np.isnan(value):
        return [0.0] * length
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        parsed = []
    if not isinstance(parsed, list):
        parsed = []
    values = [float(x) for x in parsed[:length]]
    return values + [0.0] * max(0, length - len(values))


def numeric_descriptor(series: pd.Series | None, n: int, default: np.ndarray | None = None) -> np.ndarray:
    if default is None:
        default = np.zeros(n, dtype=np.float32)
    if series is None:
        return default.astype(np.float32, copy=False)
    values = series.reset_index(drop=True)
    numeric = pd.to_numeric(values, errors="coerce")
    bool_map = values.astype(str).str.strip().str.lower().map(
        {
            "true": 1.0,
            "t": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "false": 0.0,
            "f": 0.0,
            "no": 0.0,
            "n": 0.0,
        }
    )
    filled = numeric.fillna(bool_map).fillna(pd.Series(default, index=values.index))
    return filled.to_numpy(dtype=np.float32)


def candidate_metadata(
    *,
    low_tokens: tuple[int, ...],
    high_tokens: tuple[int, ...] = FIXED_HIGH_TOKENS,
    approx_high_part: bool = False,
    design_id: str = "candidate",
    design_index: int = 0,
    config_hash: str | None = None,
) -> pd.DataFrame:
    if len(low_tokens) != LOW_TOKEN_COUNT:
        raise ValueError(f"low_tokens must contain {LOW_TOKEN_COUNT} values, got {len(low_tokens)}")
    if len(high_tokens) != HIGH_TOKEN_COUNT:
        raise ValueError(f"high_tokens must contain {HIGH_TOKEN_COUNT} values, got {len(high_tokens)}")
    data = {
        "design_id": [design_id],
        "design_index": [int(design_index)],
        "approx_high_part": [bool(approx_high_part)],
        "low_tokens": [list(low_tokens)],
        "high_tokens": [list(high_tokens)],
    }
    if config_hash is not None:
        data["config_hash"] = [str(config_hash)]
    return pd.DataFrame(data)


def descriptor_features(
    metadata: pd.DataFrame,
    delta: np.ndarray,
    *,
    exact_table: np.ndarray | None = None,
) -> np.ndarray:
    n = len(metadata)
    low = np.asarray([parse_tokens(value, LOW_TOKEN_COUNT) for value in metadata.get("low_tokens", pd.Series([None] * n))], dtype=np.float32)
    high = np.asarray([parse_tokens(value, HIGH_TOKEN_COUNT) for value in metadata.get("high_tokens", pd.Series([None] * n))], dtype=np.float32)
    approx = numeric_descriptor(metadata.get("approx_high_part"), n)[:, None]
    design_index = numeric_descriptor(metadata.get("design_index"), n, np.arange(n, dtype=np.float32))[:, None]
    metrics = metric12_from_delta(delta, exact_table=exact_table)
    return np.concatenate([low, high, approx, design_index, metrics], axis=1).astype(np.float32)


def build_common3103_features(
    *,
    metadata: pd.DataFrame,
    delta_bank: np.ndarray,
    exact_table: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    delta = as_batch(np.asarray(delta_bank, dtype=np.int32))
    shape = truth_shape_features_from_delta(delta)
    desc = descriptor_features(metadata, delta, exact_table=exact_table)
    out = np.concatenate([shape, desc], axis=1).astype(np.float32)
    if out.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} features, got {out.shape[1]}")
    return out, shape
