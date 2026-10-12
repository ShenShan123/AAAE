"""Validated access to the S8S8 sign-magnitude multiplier-LUT asset.

The historical ``truth_lut_cache`` module intentionally remains U8-only.  This
module is the explicit boundary for the newer asset whose table axes encode a
raw signed Q/DQ code as ``code + 128``.  It returns residuals against the raw
signed product, suitable for ``ApproxMatMulCudaS8Lut``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TABLE_SIDE = 256
S8_QMIN = -128
S8_QMAX = 127
S8S8_LUT_DOMAIN = "s8s8_sign_magnitude"
S8S8_TARGET_SEMANTICS = "unsigned_approximate_core_sign_magnitude_wrapper"
S8S8_INDEX_ENCODING = "signed_code_plus_128"
_S8_CODES = np.arange(S8_QMIN, S8_QMAX + 1, dtype=np.int32)
_S8_EXACT_TABLE = _S8_CODES[:, None] * _S8_CODES[None, :]


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_file(root: Path, manifest: dict[str, Any], field: str) -> Path:
    filename = manifest.get(field)
    if not isinstance(filename, str) or not filename:
        raise ValueError(f"S8S8 LUT asset manifest is missing {field}")
    candidate = (root / filename).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"S8S8 LUT asset {field} escapes asset root: {filename!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"S8S8 LUT asset file is missing: {candidate}")
    return candidate


def _validate_manifest(manifest: dict[str, Any]) -> None:
    expected = {
        "schema_version": 1,
        "asset_kind": "fixed_multiplier_lut",
        "target_lut_domain": S8S8_LUT_DOMAIN,
        "target_semantics": S8S8_TARGET_SEMANTICS,
        "index_encoding": S8S8_INDEX_ENCODING,
        "table_dtype": "int32",
    }
    mismatches = [
        f"{name}={manifest.get(name)!r} != {value!r}"
        for name, value in expected.items()
        if manifest.get(name) != value
    ]
    if mismatches:
        raise ValueError("S8S8 LUT asset manifest is incompatible: " + "; ".join(mismatches))
    if not bool(manifest.get("complete", False)):
        raise ValueError("S8S8 LUT asset is not complete")
    if manifest.get("signed_code_range") != [S8_QMIN, S8_QMAX]:
        raise ValueError("S8S8 LUT asset signed_code_range must be [-128, 127]")
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or row_count <= 0:
        raise ValueError("S8S8 LUT asset manifest has an invalid row_count")
    if manifest.get("table_shape") != [row_count, TABLE_SIDE, TABLE_SIDE]:
        raise ValueError("S8S8 LUT asset table_shape does not match [row_count, 256, 256]")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("S8S8 LUT asset manifest is missing file checksums")


def _validate_population(population: pd.DataFrame, runtime_ids: np.ndarray, *, row_count: int) -> None:
    required = {"row_index", "runtime_design_id"}
    missing = sorted(required.difference(population.columns))
    if missing:
        raise ValueError(f"S8S8 LUT asset population is missing {missing}")
    if len(population) != row_count:
        raise ValueError("S8S8 LUT asset population row count does not match table row count")
    row_indices = pd.to_numeric(population["row_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(row_indices, np.arange(row_count, dtype=np.int64)):
        raise ValueError("S8S8 LUT asset population row_index must be contiguous and ordered")
    population_ids = population["runtime_design_id"].astype(str).to_numpy()
    if len(set(population_ids.tolist())) != row_count:
        raise ValueError("S8S8 LUT asset population runtime_design_id values must be unique")
    if not np.array_equal(population_ids, runtime_ids.astype(str)):
        raise ValueError("S8S8 LUT asset population runtime IDs do not align with the LUT rows")


@dataclass(frozen=True)
class S8S8TruthLutAsset:
    """A verified, memory-mapped S8S8 truth-table population."""

    root: Path
    manifest: dict[str, Any]
    runtime_ids: np.ndarray
    completed_rows: np.ndarray
    population: pd.DataFrame
    tables: np.ndarray

    @property
    def lut_domain(self) -> str:
        return S8S8_LUT_DOMAIN

    def deltas_for_population_rows(
        self,
        rows: pd.DataFrame,
        *,
        row_index_column: str = "row_index",
        runtime_id_column: str = "runtime_design_id",
    ) -> dict[str, np.ndarray]:
        """Return raw-signed-code residual tables for selected population rows.

        The returned arrays use ``delta[q_a + 128, q_b + 128]`` and are always
        contiguous ``int32[256, 256]`` tensors.  The request must retain the
        asset's original row/ID pairing; this prevents a historical U8
        population from being accidentally matched to the signed LUT asset.
        """

        required = {row_index_column, runtime_id_column}
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise KeyError(f"population rows are missing {missing}")
        if rows.empty:
            return {}

        source_indices = pd.to_numeric(rows[row_index_column], errors="raise")
        if not np.all(np.equal(source_indices.to_numpy(), np.floor(source_indices.to_numpy()))):
            raise ValueError("S8S8 LUT asset request has non-integral row_index values")
        indices = source_indices.to_numpy(dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= len(self.runtime_ids)):
            raise IndexError("population row_index is outside the S8S8 LUT asset")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("S8S8 LUT asset request contains duplicate row_index values")

        requested_ids = rows[runtime_id_column].astype(str).to_numpy()
        expected_ids = self.runtime_ids[indices].astype(str)
        if not np.array_equal(requested_ids, expected_ids):
            raise ValueError("S8S8 LUT asset runtime IDs do not align with requested population rows")

        truth = np.asarray(self.tables[indices], dtype=np.int32)
        if truth.shape != (len(indices), TABLE_SIDE, TABLE_SIDE):
            raise ValueError("S8S8 LUT asset lookup produced an unexpected table shape")
        deltas = truth - _S8_EXACT_TABLE[None, :, :]
        return {
            str(runtime_id): np.ascontiguousarray(deltas[index], dtype=np.int32)
            for index, runtime_id in enumerate(expected_ids.tolist())
        }


def load_s8s8_sign_magnitude_truth_lut_asset(asset_root: Path) -> S8S8TruthLutAsset:
    """Open and validate a complete S8S8 sign-magnitude multiplier-LUT asset."""

    root = Path(asset_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"S8S8 LUT asset manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"S8S8 LUT asset manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("S8S8 LUT asset manifest must be a JSON object")
    _validate_manifest(manifest)

    table_path = _asset_file(root, manifest, "table_filename")
    runtime_ids_path = _asset_file(root, manifest, "runtime_ids_filename")
    completed_path = _asset_file(root, manifest, "completed_filename")
    population_path = _asset_file(root, manifest, "population_filename")
    expected_checksums = manifest["files"]
    for path in (table_path, runtime_ids_path, completed_path, population_path):
        expected_sha1 = expected_checksums.get(path.name)
        if not isinstance(expected_sha1, str) or not expected_sha1:
            raise ValueError(f"S8S8 LUT asset manifest is missing checksum for {path.name}")
        if sha1_file(path) != expected_sha1:
            raise ValueError(f"S8S8 LUT asset checksum mismatch for {path.name}")

    row_count = int(manifest["row_count"])
    runtime_ids = np.load(runtime_ids_path, allow_pickle=False)
    completed_rows = np.load(completed_path, allow_pickle=False)
    tables = np.load(table_path, mmap_mode="r", allow_pickle=False)
    population = pd.read_csv(population_path, low_memory=False)
    if runtime_ids.shape != (row_count,) or runtime_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("S8S8 LUT asset runtime_design_ids.npy has an invalid shape or dtype")
    if len(set(runtime_ids.astype(str).tolist())) != row_count:
        raise ValueError("S8S8 LUT asset runtime design IDs must be unique")
    if completed_rows.shape != (row_count,) or completed_rows.dtype != np.bool_ or not bool(np.all(completed_rows)):
        raise ValueError("S8S8 LUT asset completion bitmap is invalid")
    if tables.shape != (row_count, TABLE_SIDE, TABLE_SIDE) or tables.dtype != np.int32:
        raise ValueError(f"S8S8 LUT asset table has shape={tables.shape}, dtype={tables.dtype}")
    _validate_population(population, runtime_ids, row_count=row_count)
    return S8S8TruthLutAsset(
        root=root,
        manifest=manifest,
        runtime_ids=np.asarray(runtime_ids),
        completed_rows=np.asarray(completed_rows),
        population=population,
        tables=tables,
    )
