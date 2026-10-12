"""Unchanged main-method helpers extracted from the original Swin training code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted64 = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual64.shape != predicted64.shape or not np.isfinite(predicted64).all():
        raise ValueError("actual and predicted must be aligned finite vectors")
    absolute = np.abs(predicted64 - actual64)
    return {
        "rows": int(len(actual64)),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "over_5pp_count": int(np.sum(absolute > 0.05)),
        "at_or_over_5pp_count": int(np.sum(absolute >= 0.05)),
        "bias": float(np.mean(predicted64 - actual64)),
    }
