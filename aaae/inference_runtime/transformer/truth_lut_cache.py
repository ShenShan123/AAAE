from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CACHE_SCHEMA_VERSION = 1
TABLE_SHAPE = (256, 256)


def cache_paths(cache_root: Path) -> dict[str, Path]:
    root = Path(cache_root).resolve()
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "progress": root / "progress.json",
        "runtime_ids": root / "runtime_design_ids.npy",
        "completed": root / "completed_rows.npy",
        "tables": root / "truth_tables_uint16.npy",
    }


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CachedTruthLuts:
    root: Path
    manifest: dict[str, Any]
    runtime_ids: np.ndarray
    tables: np.ndarray

    def deltas_for_population_rows(self, rows: pd.DataFrame) -> dict[str, np.ndarray]:
        required = {"row_index", "runtime_design_id"}
        missing = sorted(required.difference(rows.columns))
        if missing:
            raise KeyError(f"population rows are missing {missing}")
        row_indices = rows["row_index"].to_numpy(np.int64)
        if np.any(row_indices < 0) or np.any(row_indices >= len(self.runtime_ids)):
            raise IndexError("population row_index is outside the LUT cache")
        expected_ids = rows["runtime_design_id"].astype(str).to_numpy()
        actual_ids = np.asarray(self.runtime_ids[row_indices]).astype(str)
        if not np.array_equal(actual_ids, expected_ids):
            raise ValueError("LUT cache runtime IDs do not align with requested population rows")
        if len(expected_ids) != len(set(expected_ids.tolist())):
            raise ValueError("requested population rows contain duplicate runtime IDs")

        truth = np.asarray(self.tables[row_indices], dtype=np.int32)
        if truth.shape != (len(rows), *TABLE_SHAPE):
            raise ValueError(f"cached truth lookup has shape {truth.shape}, expected {(len(rows), *TABLE_SHAPE)}")
        exact = np.arange(256, dtype=np.int32)[:, None] * np.arange(256, dtype=np.int32)[None, :]
        deltas = truth - exact[None, :, :]
        return {
            str(runtime_id): np.ascontiguousarray(deltas[index], dtype=np.int32)
            for index, runtime_id in enumerate(expected_ids.tolist())
        }


def load_cached_truth_luts(
    cache_root: Path,
    *,
    expected_row_count: int,
    expected_rows_sha1: str,
) -> CachedTruthLuts:
    paths = cache_paths(cache_root)
    missing = [name for name in ("manifest", "runtime_ids", "completed", "tables") if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(
            f"truth LUT cache is incomplete under {paths['root']}; missing {missing}. "
            "Run materialize-lut-cache before evaluating runtime shards."
        )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise ValueError("truth LUT cache schema version is incompatible")
    if not bool(manifest.get("complete", False)):
        raise ValueError("truth LUT cache is not complete")
    if int(manifest.get("row_count", -1)) != int(expected_row_count):
        raise ValueError("truth LUT cache row count does not match the prepared population")
    if str(manifest.get("population_rows_sha1", "")) != str(expected_rows_sha1):
        raise ValueError("truth LUT cache belongs to a different prepared population")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict):
        raise ValueError("truth LUT cache manifest is missing file checksums")
    for filename, path in (
        ("runtime_design_ids.npy", paths["runtime_ids"]),
        ("completed_rows.npy", paths["completed"]),
        ("truth_tables_uint16.npy", paths["tables"]),
    ):
        expected_sha1 = expected_files.get(filename)
        if not isinstance(expected_sha1, str) or not expected_sha1:
            raise ValueError(f"truth LUT cache manifest is missing checksum for {filename}")
        actual_sha1 = sha1_file(path)
        if actual_sha1 != expected_sha1:
            raise ValueError(f"truth LUT cache checksum mismatch for {filename}")

    runtime_ids = np.load(paths["runtime_ids"], allow_pickle=False)
    completed = np.load(paths["completed"], allow_pickle=False)
    tables = np.load(paths["tables"], mmap_mode="r", allow_pickle=False)
    if runtime_ids.shape != (int(expected_row_count),):
        raise ValueError(f"cached runtime IDs have shape {runtime_ids.shape}")
    if completed.shape != (int(expected_row_count),) or completed.dtype != np.bool_ or not bool(np.all(completed)):
        raise ValueError("truth LUT cache completion bitmap is invalid")
    if tables.shape != (int(expected_row_count), *TABLE_SHAPE) or tables.dtype != np.uint16:
        raise ValueError(f"cached truth tables have shape={tables.shape}, dtype={tables.dtype}")
    return CachedTruthLuts(
        root=paths["root"],
        manifest=manifest,
        runtime_ids=np.asarray(runtime_ids),
        tables=tables,
    )
