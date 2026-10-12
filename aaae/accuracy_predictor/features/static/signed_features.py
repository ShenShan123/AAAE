"""Build signed-S8S8 ``common3103`` static features for strict-50 training.

The 3,103-column layout is intentionally the same layout consumed by the
historical strict-50 training code.  Its values must nevertheless describe the
same multiplier domain as the runtime labels.  For this S8S8 dataset that means
the table axes are raw signed Q/DQ codes indexed as ``code + 128`` and each
residual is measured against the signed exact product.

The truth-table asset is memory mapped and processed in small row batches.  A
full 20k asset is several GiB, so this module never materializes it as one
array.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT
from aaae.paths import EA2_ROOT as EA2_ROOT
END_TO_END_ROOT = EA2_ROOT / "end_to_end_prediction"


from aaae.multiplier_generator.lut_domains import s8s8_exact_table_np  # noqa: E402
from aaae.accuracy_predictor.features.static.config import (  # noqa: E402
    FEATURE_DIM,
    HIGH_TOKEN_COUNT,
    LOW_TOKEN_COUNT,
)
from aaae.accuracy_predictor.features.static.feature_builder import (  # noqa: E402
    build_common3103_features,
)
from aaae.inference_runtime.transformer.s8s8_truth_lut_asset import (  # noqa: E402
    S8S8TruthLutAsset,
    load_s8s8_sign_magnitude_truth_lut_asset,
)


TABLE_SIDE = 256
STATIC_FEATURE_SCHEMA = "s8s8_signed_residual_shape3072_plus_descriptor31_v1"
DEFAULT_S8S8_ASSET_ROOT = (
    END_TO_END_ROOT / "datasets" / "multiplier" / "lowonly20k_s8s8_sign_magnitude_v1"
)


@dataclass(frozen=True)
class S8S8SignedPopulationValidation:
    """Validated target metadata needed to materialize a signed static matrix."""

    target_population: pd.DataFrame
    metadata: pd.DataFrame
    low_tokens: np.ndarray
    replacement_kind_counts: Mapping[str, int]

    @property
    def row_count(self) -> int:
        return int(len(self.target_population))


@dataclass(frozen=True)
class S8S8SignedCommon3103Build:
    """Signed static matrix plus the immutable feature-domain provenance."""

    common3103: np.ndarray
    validation: S8S8SignedPopulationValidation
    asset_manifest: Mapping[str, object]
    chunk_size: int

    @property
    def row_count(self) -> int:
        return int(self.common3103.shape[0])


def _as_population_frame(value: pd.DataFrame | Path | str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "population.csv"
    if not path.is_file():
        raise FileNotFoundError(f"S8S8 target population is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise KeyError(f"S8S8 target population is missing required columns: {missing}")


def _parse_bool(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _parse_tokens(value: object, *, count: int, name: str, minimum: int, maximum: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, np.ndarray)):
        raw = list(value)
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"{name} is not a token list: {value!r}") from exc
    if not isinstance(raw, list) or len(raw) != int(count):
        raise ValueError(f"{name} must contain exactly {count} tokens, got {value!r}")
    result: list[int] = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError(f"{name} contains a boolean token: {item!r}")
        try:
            numeric = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} contains a non-numeric token: {item!r}") from exc
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{name} contains a non-integral token: {item!r}")
        token = int(numeric)
        if token < int(minimum) or token > int(maximum):
            raise ValueError(f"{name} contains token {token} outside [{minimum}, {maximum}]")
        result.append(token)
    return tuple(result)


def _lowonly_config_hash(low_tokens: np.ndarray) -> str:
    payload = {
        "approx_high_part": False,
        "high_tokens": [0] * HIGH_TOKEN_COUNT,
        "low_tokens": [int(value) for value in np.asarray(low_tokens, dtype=np.int64).tolist()],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def validate_s8s8_signed_population(
    target_population: pd.DataFrame | Path | str = DEFAULT_S8S8_ASSET_ROOT,
) -> S8S8SignedPopulationValidation:
    """Validate low-only target metadata independently of historical U8 features."""
    target = _as_population_frame(target_population)
    _require_columns(
        target,
        {
            "row_index",
            "design_id",
            "runtime_design_id",
            "config_hash",
            "approx_high_part",
            "low_tokens",
            "high_tokens",
        },
    )
    if target.empty:
        raise ValueError("S8S8 target population must not be empty")

    row_indices = pd.to_numeric(target["row_index"], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(row_indices).all() or not np.equal(row_indices, np.floor(row_indices)).all():
        raise ValueError("S8S8 target row_index values must be finite integers")
    row_indices_i64 = row_indices.astype(np.int64)
    if len(np.unique(row_indices_i64)) != len(row_indices_i64) or np.any(np.diff(row_indices_i64) <= 0):
        raise ValueError("S8S8 target row_index values must be unique and ordered")

    for column in ("design_id", "runtime_design_id", "config_hash"):
        values = target[column].astype(str)
        if values.duplicated().any():
            duplicate = str(values[values.duplicated()].iloc[0])
            raise ValueError(f"S8S8 target {column} values must be unique; duplicate {duplicate!r}")

    low_tokens = np.asarray(
        [
            _parse_tokens(value, count=LOW_TOKEN_COUNT, name=f"low_tokens[{index}]", minimum=1, maximum=8)
            for index, value in enumerate(target["low_tokens"].tolist())
        ],
        dtype=np.int64,
    )
    high_tokens = np.asarray(
        [
            _parse_tokens(value, count=HIGH_TOKEN_COUNT, name=f"high_tokens[{index}]", minimum=0, maximum=8)
            for index, value in enumerate(target["high_tokens"].tolist())
        ],
        dtype=np.int64,
    )
    if np.any(high_tokens != 0):
        raise ValueError("S8S8 target population must use exact all-zero high tokens")
    approximate_high = np.asarray(
        [
            _parse_bool(value, name=f"approx_high_part[{index}]")
            for index, value in enumerate(target["approx_high_part"].tolist())
        ],
        dtype=bool,
    )
    if approximate_high.any():
        raise ValueError("S8S8 target population must disable approx_high_part")
    expected_hashes = np.asarray([_lowonly_config_hash(row) for row in low_tokens], dtype=str)
    actual_hashes = target["config_hash"].astype(str).to_numpy()
    if not np.array_equal(actual_hashes, expected_hashes):
        mismatch = int(np.flatnonzero(actual_hashes != expected_hashes)[0])
        raise ValueError(f"S8S8 target config_hash does not match low-only tokens at row {mismatch}")

    metadata = pd.DataFrame(
        {
            "low_tokens": [list(row) for row in low_tokens.tolist()],
            "high_tokens": [[0] * HIGH_TOKEN_COUNT for _ in range(len(target))],
            "approx_high_part": np.zeros(len(target), dtype=bool),
            "design_index": np.zeros(len(target), dtype=np.int64),
        }
    )
    if "replacement_kind" in target.columns:
        counts = {
            str(kind): int(count)
            for kind, count in target["replacement_kind"].astype(str).value_counts(sort=False).items()
        }
    else:
        counts = {}
    return S8S8SignedPopulationValidation(
        target_population=target,
        metadata=metadata,
        low_tokens=np.ascontiguousarray(low_tokens, dtype=np.int64),
        replacement_kind_counts=counts,
    )


def _deltas_for_rows(asset: S8S8TruthLutAsset, rows: pd.DataFrame) -> np.ndarray:
    deltas_by_runtime_id = asset.deltas_for_population_rows(rows)
    runtime_ids = rows["runtime_design_id"].astype(str).to_numpy()
    delta = np.stack([deltas_by_runtime_id[str(runtime_id)] for runtime_id in runtime_ids], axis=0)
    if delta.shape != (len(rows), TABLE_SIDE, TABLE_SIDE) or delta.dtype != np.int32:
        raise ValueError(f"S8S8 asset produced invalid residual batch {delta.shape} {delta.dtype}")
    return np.ascontiguousarray(delta)


def build_s8s8_signed_common3103(
    *,
    target_population: pd.DataFrame | Path | str = DEFAULT_S8S8_ASSET_ROOT,
    asset_root: Path | str = DEFAULT_S8S8_ASSET_ROOT,
    chunk_size: int = 64,
) -> S8S8SignedCommon3103Build:
    """Build common3103 from the signed S8S8 residual tables used at runtime."""
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    validation = validate_s8s8_signed_population(target_population)
    asset = load_s8s8_sign_magnitude_truth_lut_asset(Path(asset_root))
    exact_table = s8s8_exact_table_np()
    output = np.empty((validation.row_count, FEATURE_DIM), dtype=np.float32)
    for start in range(0, validation.row_count, int(chunk_size)):
        stop = min(start + int(chunk_size), validation.row_count)
        rows = validation.target_population.iloc[start:stop].reset_index(drop=True)
        delta = _deltas_for_rows(asset, rows)
        features, _shape = build_common3103_features(
            metadata=validation.metadata.iloc[start:stop].reset_index(drop=True),
            delta_bank=delta,
            exact_table=exact_table,
        )
        output[start:stop] = features
    if output.shape != (validation.row_count, FEATURE_DIM) or not np.isfinite(output).all():
        raise ValueError(f"S8S8 static feature builder returned invalid common3103 {output.shape}")
    if not np.array_equal(output[:, 3089], np.zeros(validation.row_count, dtype=np.float32)):
        raise ValueError("S8S8 static features did not set approx_high_part descriptor to zero")
    if not np.array_equal(output[:, 3090], np.zeros(validation.row_count, dtype=np.float32)):
        raise ValueError("S8S8 static features did not set design_index descriptor to zero")
    return S8S8SignedCommon3103Build(
        common3103=output,
        validation=validation,
        asset_manifest=dict(asset.manifest),
        chunk_size=int(chunk_size),
    )


__all__ = [
    "DEFAULT_S8S8_ASSET_ROOT",
    "STATIC_FEATURE_SCHEMA",
    "S8S8SignedCommon3103Build",
    "S8S8SignedPopulationValidation",
    "build_s8s8_signed_common3103",
    "validate_s8s8_signed_population",
]
