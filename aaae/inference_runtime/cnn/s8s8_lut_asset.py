from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .errors import LabelGenerationError


LUT_DOMAIN = "s8s8_sign_magnitude"
TARGET_SEMANTICS = "unsigned_approximate_core_sign_magnitude_wrapper"
INDEX_ENCODING = "signed_code_plus_128"
_SIGNED_CODES = np.arange(-128, 128, dtype=np.int32)
_EXACT_SIGNED_PRODUCT = _SIGNED_CODES[:, None] * _SIGNED_CODES[None, :]


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset_file(root: Path, manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not isinstance(value, str) or not value:
        raise LabelGenerationError(f"S8S8 LUT manifest is missing {key}")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise LabelGenerationError(f"S8S8 LUT {key} escapes asset_root") from exc
    if not path.is_file():
        raise LabelGenerationError(f"S8S8 LUT file is missing: {path}")
    return path


@dataclass(frozen=True)
class S8S8LutAsset:
    root: Path
    manifest: dict[str, Any]
    runtime_ids: np.ndarray
    tables: np.ndarray
    row_by_id: dict[str, int]

    def delta_for(self, design_id: str) -> np.ndarray:
        try:
            row = self.row_by_id[str(design_id)]
        except KeyError as exc:
            raise LabelGenerationError(f"design_id {design_id!r} is absent from the S8S8 LUT asset") from exc
        truth = np.asarray(self.tables[row], dtype=np.int32)
        delta = truth.astype(np.int64) - _EXACT_SIGNED_PRODUCT.astype(np.int64)
        int32 = np.iinfo(np.int32)
        if np.any(delta < int32.min) or np.any(delta > int32.max):
            raise LabelGenerationError(
                f"design_id {design_id!r} produces residuals outside the int32 runtime contract"
            )
        return np.ascontiguousarray(delta, dtype=np.int32)

    def require_design_ids(self, design_ids: Iterable[str]) -> tuple[str, ...]:
        requested = tuple(str(value).strip() for value in design_ids if str(value).strip())
        if not requested:
            raise LabelGenerationError("designs.design_ids must not be empty")
        if len(set(requested)) != len(requested):
            raise LabelGenerationError("designs.design_ids contains duplicates")
        missing = [value for value in requested if value not in self.row_by_id]
        if missing:
            raise LabelGenerationError(f"design IDs are absent from the S8S8 LUT asset: {missing[:5]}")
        return requested

    def max_abs_delta(self, design_ids: Iterable[str], *, chunk_rows: int = 64) -> int:
        """Return a checked residual bound for the requested frozen population."""

        requested = self.require_design_ids(design_ids)
        if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
            raise LabelGenerationError("S8S8 residual-bound chunk_rows must be positive")
        row_indices = np.asarray([self.row_by_id[value] for value in requested], dtype=np.int64)
        exact = _EXACT_SIGNED_PRODUCT.astype(np.int64, copy=False)
        maximum = 0
        for start in range(0, len(row_indices), chunk_rows):
            rows = row_indices[start : start + chunk_rows]
            truth = np.asarray(self.tables[rows], dtype=np.int64)
            delta = truth - exact[None, :, :]
            maximum = max(maximum, int(np.max(np.abs(delta))))
        int32 = np.iinfo(np.int32)
        if maximum > int32.max:
            raise LabelGenerationError(
                f"selected S8S8 LUT residual bound {maximum} exceeds the int32 runtime contract"
            )
        return maximum


def load_s8s8_lut_asset(asset_root: Path, *, verify_checksums: bool = True) -> S8S8LutAsset:
    root = Path(asset_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelGenerationError(f"unable to read S8S8 LUT manifest {manifest_path}: {exc}") from exc
    expected = {
        "schema_version": 1,
        "asset_kind": "fixed_multiplier_lut",
        "target_lut_domain": LUT_DOMAIN,
        "target_semantics": TARGET_SEMANTICS,
        "index_encoding": INDEX_ENCODING,
        "table_dtype": "int32",
        "signed_code_range": [-128, 127],
        "complete": True,
    }
    mismatches = [
        f"{key}={manifest.get(key)!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise LabelGenerationError("incompatible S8S8 LUT asset: " + ", ".join(mismatches))
    row_count = manifest.get("row_count")
    if not isinstance(row_count, int) or row_count <= 0:
        raise LabelGenerationError("S8S8 LUT row_count must be positive")
    if manifest.get("table_shape") != [row_count, 256, 256]:
        raise LabelGenerationError("S8S8 LUT table_shape must be [row_count, 256, 256]")

    runtime_ids_path = _asset_file(root, manifest, "runtime_ids_filename")
    completed_path = _asset_file(root, manifest, "completed_filename")
    population_path = _asset_file(root, manifest, "population_filename")
    table_path = _asset_file(root, manifest, "table_filename")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise LabelGenerationError("S8S8 LUT manifest is missing files checksums")
    if verify_checksums:
        for path in (runtime_ids_path, completed_path, population_path, table_path):
            expected_hash = files.get(path.name)
            if not isinstance(expected_hash, str) or _sha1(path) != expected_hash:
                raise LabelGenerationError(f"S8S8 LUT checksum mismatch: {path.name}")

    runtime_ids = np.load(runtime_ids_path, allow_pickle=False)
    completed = np.load(completed_path, allow_pickle=False)
    tables = np.load(table_path, mmap_mode="r", allow_pickle=False)
    if runtime_ids.shape != (row_count,) or runtime_ids.dtype.kind not in {"U", "S"}:
        raise LabelGenerationError("S8S8 LUT runtime_design_ids has an invalid schema")
    ids = runtime_ids.astype(str)
    if len(set(ids.tolist())) != row_count:
        raise LabelGenerationError("S8S8 LUT runtime design IDs are not unique")
    if completed.shape != (row_count,) or completed.dtype != np.bool_ or not bool(np.all(completed)):
        raise LabelGenerationError("S8S8 LUT completion bitmap is incomplete")
    if tables.shape != (row_count, 256, 256) or tables.dtype != np.int32:
        raise LabelGenerationError("S8S8 LUT truth table has an invalid shape or dtype")

    with population_path.open("r", encoding="utf-8", newline="") as handle:
        population = list(csv.DictReader(handle))
    if len(population) != row_count:
        raise LabelGenerationError("S8S8 LUT population row count is inconsistent")
    for index, row in enumerate(population):
        if int(row.get("row_index", -1)) != index or str(row.get("runtime_design_id", "")) != ids[index]:
            raise LabelGenerationError("S8S8 LUT population is not aligned with table rows")
    return S8S8LutAsset(
        root=root,
        manifest=manifest,
        runtime_ids=ids,
        tables=tables,
        row_by_id={value: index for index, value in enumerate(ids.tolist())},
    )
