#!/usr/bin/env python
"""Build the fixed low-only 20k S8S8 multiplier-LUT asset.

The historical retained 20k population contains a small high-part-approximate
subset.  This builder keeps every existing low-only configuration, replaces
each high-part approximation with the same low tokens and an exact high part,
and deterministically fills any resulting duplicate configuration from the
canonical low-only catalog.  It then materializes the requested S8S8
sign-magnitude wrapper directly from the frozen neural LUT decoder.
"""

from __future__ import annotations

from aaae.paths import OUTPUT_ROOT

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch


from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT
from aaae.paths import EA2_ROOT as EA2_ROOT
APPROX_ERROR_MODEL_ROOT = EA2_ROOT / "approx_error_model"

from aaae.multiplier_generator.lut_domains import u8_lut_to_s8s8_sign_magnitude  # noqa: E402
from aaae.multiplier_generator.truth_table import (  # noqa: E402
    NeuralLogicTruthDecoder,
    UINT16_MAX,
    exact_table_np,
)


ASSET_SCHEMA_VERSION = 1
ASSET_ID = "lowonly20k_s8s8_sign_magnitude_v1"
TABLE_SIDE = 256
TABLE_SHAPE = (TABLE_SIDE, TABLE_SIDE)
TABLE_FILENAME = "truth_tables_s8s8_sign_magnitude_int32.npy"
RUNTIME_IDS_FILENAME = "runtime_design_ids.npy"
COMPLETED_FILENAME = "completed_rows.npy"
POPULATION_FILENAME = "population.csv"
TARGET_SEMANTICS = "unsigned_approximate_core_sign_magnitude_wrapper"

DEFAULT_CATALOG = (
    PROJECT_ROOT
    / "resources"
    / "data"
    / "multipliers"
    / "D_mult"
    / "canonical"
    / "meta"
    / "multiplier_catalog.csv"
)
DEFAULT_CHECKPOINT = (
    APPROX_ERROR_MODEL_ROOT
    / "runs"
    / "truth_table"
    / "full17_neural_logic"
    / "checkpoints"
    / "best.pt"
)
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "datasets" / "multiplier" / ASSET_ID


def _sha1_bytes(payload: bytes) -> str:
    return hashlib.sha1(payload).hexdigest()


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _parse_tokens(value: object, *, expected_len: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = [int(item) for item in value]
    else:
        parsed = [int(item) for item in ast.literal_eval(str(value))]
    if len(parsed) != int(expected_len):
        raise ValueError(f"expected {expected_len} tokens, got {value!r}")
    if any(item < 0 or item > 8 for item in parsed):
        raise ValueError(f"tokens must be in [0, 8], got {parsed}")
    return tuple(parsed)


def _token_text(tokens: tuple[int, ...]) -> str:
    return json.dumps([int(item) for item in tokens], separators=(",", ":"))


def lowonly_config_hash(low_tokens: tuple[int, ...]) -> str:
    payload = {
        "approx_high_part": False,
        "high_tokens": [0] * 8,
        "low_tokens": [int(item) for item in low_tokens],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _catalog_lowonly_candidates(catalog_path: Path) -> Iterator[dict[str, Any]]:
    required = ["design_id", "design_index", "approx_high_part", "low_tokens", "high_tokens"]
    catalog = pd.read_csv(catalog_path, usecols=required, low_memory=False)
    catalog = catalog.sort_values(["design_index", "design_id"], kind="stable")
    for row in catalog.itertuples(index=False):
        if _parse_bool(row.approx_high_part):
            continue
        low_tokens = _parse_tokens(row.low_tokens, expected_len=9)
        high_tokens = _parse_tokens(row.high_tokens, expected_len=8)
        if high_tokens != (0,) * 8:
            continue
        yield {
            "catalog_design_id": str(row.design_id),
            "catalog_design_index": int(row.design_index),
            "low_tokens": low_tokens,
            "config_hash": lowonly_config_hash(low_tokens),
        }


def build_lowonly_population(
    source_population_path: Path,
    catalog_path: Path,
    *,
    asset_id: str = ASSET_ID,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a unique low-only population and its deterministic selection record."""
    source_population_path = Path(source_population_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    source = pd.read_csv(source_population_path, low_memory=False)
    required = {
        "row_index",
        "design_id",
        "runtime_design_id",
        "config_hash",
        "approx_high_part",
        "low_tokens",
        "high_tokens",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"{source_population_path} is missing {missing}")
    source = source.sort_values("row_index", kind="stable").reset_index(drop=True)
    expected_rows = np.arange(len(source), dtype=np.int64)
    if not np.array_equal(source["row_index"].to_numpy(np.int64), expected_rows):
        raise ValueError("source row_index must be contiguous and ordered")

    zero_high = (0,) * 8
    planned: list[dict[str, Any]] = []
    by_config_hash: dict[str, list[dict[str, Any]]] = {}
    high_replaced = 0
    lowonly_preserved = 0
    for row in source.itertuples(index=False):
        low_tokens = _parse_tokens(row.low_tokens, expected_len=9)
        source_high_tokens = _parse_tokens(row.high_tokens, expected_len=8)
        source_high_approx = _parse_bool(row.approx_high_part)
        if not source_high_approx and source_high_tokens != zero_high:
            raise ValueError(
                f"source row {row.row_index} claims low-only but has non-zero high tokens: {source_high_tokens}"
            )
        if source_high_approx:
            replacement_kind = "source_high_replaced_with_exact_high"
            high_replaced += 1
        else:
            replacement_kind = "original_low_only"
            lowonly_preserved += 1
        config_hash = lowonly_config_hash(low_tokens)
        item = {
            "source_row_index": int(row.row_index),
            "source_design_id": str(row.design_id),
            "source_runtime_design_id": str(row.runtime_design_id),
            "source_config_hash": str(row.config_hash),
            "source_approx_high_part": bool(source_high_approx),
            "source_low_tokens": low_tokens,
            "source_high_tokens": source_high_tokens,
            "low_tokens": low_tokens,
            "high_tokens": zero_high,
            "config_hash": config_hash,
            "replacement_kind": replacement_kind,
            "catalog_fill_design_id": "",
            "catalog_fill_design_index": "",
        }
        planned.append(item)
        by_config_hash.setdefault(config_hash, []).append(item)

    occupied = set(by_config_hash)
    catalog_candidates = _catalog_lowonly_candidates(catalog_path)
    duplicate_config_groups = 0
    catalog_fill_count = 0
    for config_hash, group in sorted(by_config_hash.items()):
        if len(group) == 1:
            continue
        duplicate_config_groups += 1
        keeper = min(
            group,
            key=lambda item: (
                0 if item["replacement_kind"] == "original_low_only" else 1,
                int(item["source_row_index"]),
            ),
        )
        for item in group:
            if item is keeper:
                continue
            candidate = next((entry for entry in catalog_candidates if entry["config_hash"] not in occupied), None)
            if candidate is None:
                raise RuntimeError("canonical catalog does not contain enough unused low-only candidates")
            occupied.add(str(candidate["config_hash"]))
            item.update(
                {
                    "low_tokens": tuple(candidate["low_tokens"]),
                    "high_tokens": zero_high,
                    "config_hash": str(candidate["config_hash"]),
                    "replacement_kind": "catalog_lowonly_fill_for_duplicate",
                    "catalog_fill_design_id": str(candidate["catalog_design_id"]),
                    "catalog_fill_design_index": int(candidate["catalog_design_index"]),
                }
            )
            catalog_fill_count += 1

    output_rows: list[dict[str, Any]] = []
    for item in planned:
        row_index = int(item["source_row_index"])
        output_rows.append(
            {
                "row_index": row_index,
                "design_id": f"{asset_id}_design_{row_index:06d}",
                "runtime_design_id": f"{asset_id}_design_{row_index:06d}",
                "config_hash": str(item["config_hash"]),
                "approx_high_part": False,
                "low_tokens": _token_text(tuple(item["low_tokens"])),
                "high_tokens": _token_text(zero_high),
                "replacement_kind": str(item["replacement_kind"]),
                "catalog_fill_design_id": str(item["catalog_fill_design_id"]),
                "catalog_fill_design_index": item["catalog_fill_design_index"],
                "source_row_index": row_index,
                "source_design_id": str(item["source_design_id"]),
                "source_runtime_design_id": str(item["source_runtime_design_id"]),
                "source_config_hash": str(item["source_config_hash"]),
                "source_approx_high_part": bool(item["source_approx_high_part"]),
                "source_low_tokens": _token_text(tuple(item["source_low_tokens"])),
                "source_high_tokens": _token_text(tuple(item["source_high_tokens"])),
            }
        )
    population = pd.DataFrame(output_rows).sort_values("row_index", kind="stable").reset_index(drop=True)
    if not np.array_equal(population["row_index"].to_numpy(np.int64), np.arange(len(population), dtype=np.int64)):
        raise RuntimeError("output population row_index is not contiguous")
    if population["runtime_design_id"].duplicated().any():
        raise RuntimeError("output population runtime_design_id is not unique")
    if population["config_hash"].duplicated().any():
        raise RuntimeError("output population config_hash is not unique")
    if population["approx_high_part"].astype(bool).any():
        raise RuntimeError("output population contains high-part approximation")
    if any(_parse_tokens(value, expected_len=8) != zero_high for value in population["high_tokens"].tolist()):
        raise RuntimeError("output population contains non-zero high tokens")

    selection = {
        "source_population_path": str(source_population_path),
        "source_population_sha1": _sha1_file(source_population_path),
        "catalog_path": str(catalog_path),
        "catalog_sha1": _sha1_file(catalog_path),
        "source_row_count": int(len(source)),
        "output_row_count": int(len(population)),
        "original_low_only_preserved": int(lowonly_preserved),
        "source_high_part_replaced": int(high_replaced),
        "duplicate_config_groups_after_zeroing_high": int(duplicate_config_groups),
        "catalog_lowonly_fill_count": int(catalog_fill_count),
        "replacement_kind_counts": {
            str(key): int(value)
            for key, value in population["replacement_kind"].value_counts().sort_index().items()
        },
    }
    return population, selection


def _population_csv_bytes(population: pd.DataFrame) -> bytes:
    return population.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _asset_paths(root: Path) -> dict[str, Path]:
    root = Path(root).resolve()
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "progress": root / "progress.json",
        "population": root / POPULATION_FILENAME,
        "runtime_ids": root / RUNTIME_IDS_FILENAME,
        "completed": root / COMPLETED_FILENAME,
        "tables": root / TABLE_FILENAME,
    }


def _load_decoder(checkpoint_path: Path, *, device: torch.device) -> tuple[NeuralLogicTruthDecoder, torch.Tensor, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_kind = str(checkpoint.get("model_kind", checkpoint.get("config", {}).get("model_kind", "pairwise")))
    if model_kind != "neural_logic_lut":
        raise ValueError(f"unsupported checkpoint model_kind={model_kind!r}")
    num_slots = int(checkpoint.get("num_slots", checkpoint.get("config", {}).get("num_slots", 9)))
    if num_slots not in {9, 17}:
        raise ValueError(f"unsupported decoder slot count {num_slots}")
    cfg = checkpoint.get("config", {})
    model = NeuralLogicTruthDecoder(
        num_slots=num_slots,
        lut_init=str(cfg.get("lut_init", "known")),
        trainable_lut=bool(cfg.get("trainable_compressor_lut", False)),
        lut_logit_scale=float(cfg.get("lut_logit_scale", 30.0)),
    ).to(device=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    exact = torch.from_numpy(exact_table_np().astype(np.float32)).to(device=device).unsqueeze(0)
    return model, exact, num_slots


def _manifest_expectations(
    *,
    asset_id: str,
    row_count: int,
    population_sha1: str,
    selection: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha1: str,
    row_limit: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "asset_id": asset_id,
        "row_count": int(row_count),
        "population_sha1": str(population_sha1),
        "source_population_sha1": str(selection["source_population_sha1"]),
        "catalog_sha1": str(selection["catalog_sha1"]),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha1": str(checkpoint_sha1),
        "target_lut_domain": "s8s8_sign_magnitude",
        "target_semantics": TARGET_SEMANTICS,
        "native_signed_rtl": False,
        "row_limit": None if row_limit is None else int(row_limit),
    }


def _validate_existing_asset(
    paths: dict[str, Path],
    manifest: dict[str, Any],
    *,
    expected: dict[str, Any],
    runtime_ids: np.ndarray,
) -> None:
    mismatches = [
        f"{key}={manifest.get(key)!r} != {value!r}"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]
    if mismatches:
        raise ValueError("existing fixed LUT asset is incompatible: " + "; ".join(mismatches))
    if not paths["population"].exists() or _sha1_file(paths["population"]) != expected["population_sha1"]:
        raise ValueError("existing fixed LUT asset population file does not match its manifest")
    if not bool(manifest.get("complete", False)):
        return
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("complete fixed LUT asset is missing file checksums")
    for filename, path in (
        (RUNTIME_IDS_FILENAME, paths["runtime_ids"]),
        (COMPLETED_FILENAME, paths["completed"]),
        (TABLE_FILENAME, paths["tables"]),
    ):
        expected_sha1 = files.get(filename)
        if not isinstance(expected_sha1, str) or not expected_sha1:
            raise ValueError(f"complete fixed LUT asset is missing checksum for {filename}")
        if not path.exists() or _sha1_file(path) != expected_sha1:
            raise ValueError(f"complete fixed LUT asset checksum mismatch for {filename}")
    loaded_ids = np.load(paths["runtime_ids"], allow_pickle=False)
    completed = np.load(paths["completed"], allow_pickle=False)
    tables = np.load(paths["tables"], mmap_mode="r", allow_pickle=False)
    row_count = int(expected["row_count"])
    if not np.array_equal(loaded_ids.astype(str), runtime_ids.astype(str)):
        raise ValueError("complete fixed LUT asset runtime IDs do not align with population")
    if completed.shape != (row_count,) or completed.dtype != np.bool_ or not bool(np.all(completed)):
        raise ValueError("complete fixed LUT asset completion bitmap is invalid")
    if tables.shape != (row_count, *TABLE_SHAPE) or tables.dtype != np.int32:
        raise ValueError("complete fixed LUT asset table layout is invalid")


def _validate_s8_zero_operand_contract(tables: np.ndarray) -> None:
    if np.any(tables[..., 128, :]) or np.any(tables[..., :, 128]):
        raise RuntimeError("S8S8 LUT violates the zero-operand contract")


def materialize_lowonly20k_s8s8_asset(
    *,
    output_root: Path,
    source_population_path: Path | None = None,
    catalog_path: Path = DEFAULT_CATALOG,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    device: str = "cuda:0",
    batch_rows: int = 64,
    flush_every_batches: int = 16,
    resume: bool = False,
    overwrite: bool = False,
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Materialize a resumable fixed S8S8 asset and return its manifest fields."""
    if int(batch_rows) <= 0 or int(flush_every_batches) <= 0:
        raise ValueError("batch_rows and flush_every_batches must be positive")
    if row_limit is not None and int(row_limit) <= 0:
        raise ValueError("row_limit must be positive when provided")

    if source_population_path is None:
        raise ValueError("source_population_path is required for S8S8 asset materialization")
    source_population_path = Path(source_population_path).resolve()
    catalog_path = Path(catalog_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"decoder checkpoint does not exist: {checkpoint_path}")
    population, selection = build_lowonly_population(source_population_path, catalog_path)
    full_row_count = int(len(population))
    if full_row_count != 20_000:
        raise ValueError(f"expected a 20k source population, got {full_row_count} rows")
    if row_limit is not None:
        population = population.iloc[: int(row_limit)].copy().reset_index(drop=True)
        population["row_index"] = np.arange(len(population), dtype=np.int64)
        asset_id = f"{ASSET_ID}_smoke_{int(row_limit)}"
    else:
        asset_id = ASSET_ID
    population_bytes = _population_csv_bytes(population)
    population_sha1 = _sha1_bytes(population_bytes)
    runtime_ids = population["runtime_design_id"].astype(str).to_numpy(dtype="<U96")
    checkpoint_sha1 = _sha1_file(checkpoint_path)
    expected = _manifest_expectations(
        asset_id=asset_id,
        row_count=len(population),
        population_sha1=population_sha1,
        selection=selection,
        checkpoint_path=checkpoint_path,
        checkpoint_sha1=checkpoint_sha1,
        row_limit=row_limit,
    )
    paths = _asset_paths(output_root)
    if bool(overwrite) and paths["root"].exists():
        shutil.rmtree(paths["root"])
    if paths["root"].exists() and not paths["manifest"].exists():
        raise FileExistsError(f"{paths['root']} exists but is not a resumable fixed LUT asset")

    if paths["manifest"].exists():
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        _validate_existing_asset(paths, manifest, expected=expected, runtime_ids=runtime_ids)
        if bool(manifest.get("complete", False)):
            return {"status": "reused", "asset_root": str(paths["root"]), **manifest}
        if not bool(resume):
            raise FileExistsError(f"{paths['root']} is incomplete; pass resume=True or overwrite=True")
    else:
        paths["root"].mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(paths["population"], population_bytes)
        np.save(paths["runtime_ids"], runtime_ids)
        np.save(paths["completed"], np.zeros((len(population),), dtype=np.bool_))
        tables = np.lib.format.open_memmap(
            paths["tables"],
            mode="w+",
            dtype=np.int32,
            shape=(len(population), *TABLE_SHAPE),
        )
        tables.flush()
        del tables
        manifest = {
            **expected,
            "asset_kind": "fixed_multiplier_lut",
            "complete": False,
            "asset_root": str(paths["root"]),
            "population_filename": POPULATION_FILENAME,
            "runtime_ids_filename": RUNTIME_IDS_FILENAME,
            "completed_filename": COMPLETED_FILENAME,
            "table_filename": TABLE_FILENAME,
            "table_dtype": "int32",
            "table_shape": [int(len(population)), *TABLE_SHAPE],
            "index_encoding": "signed_code_plus_128",
            "signed_code_range": [-128, 127],
            "source_magnitude_range": [0, 128],
            "unsigned_lut_generation": "full17_neural_logic_decoder_round_clamp_uint16",
            "selection": selection,
            "full_source_population_row_count": full_row_count,
            "completed_row_count": 0,
            "created_at_unix": time.time(),
        }
        _write_json(paths["manifest"], manifest)
        _write_json(paths["progress"], manifest)

    completed = np.load(paths["completed"], mmap_mode="r+", allow_pickle=False)
    tables = np.load(paths["tables"], mmap_mode="r+", allow_pickle=False)
    if completed.shape != (len(population),) or completed.dtype != np.bool_:
        raise ValueError("fixed LUT asset completion bitmap layout is invalid")
    if tables.shape != (len(population), *TABLE_SHAPE) or tables.dtype != np.int32:
        raise ValueError("fixed LUT asset table layout is invalid")
    stored_ids = np.load(paths["runtime_ids"], allow_pickle=False)
    if not np.array_equal(stored_ids.astype(str), runtime_ids.astype(str)):
        raise ValueError("fixed LUT asset runtime IDs do not align with population")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {device}, but CUDA is unavailable")
    model, exact_batched, decoder_slots = _load_decoder(checkpoint_path, device=requested_device)
    pending = np.flatnonzero(~np.asarray(completed, dtype=bool)).astype(np.int64)
    pending_commit: list[np.ndarray] = []
    try:
        for batch_number, start in enumerate(range(0, len(pending), int(batch_rows)), start=1):
            row_indices = pending[start : start + int(batch_rows)]
            rows = population.iloc[row_indices]
            low_rows = [_parse_tokens(value, expected_len=9) for value in rows["low_tokens"].tolist()]
            if decoder_slots == 17:
                slot_rows = [(*low, *(0,) * 8) for low in low_rows]
            else:
                slot_rows = low_rows
            slot_tokens = torch.as_tensor(np.asarray(slot_rows, dtype=np.int64), dtype=torch.long, device=requested_device)
            with torch.no_grad():
                delta = model(slot_tokens)["delta"]
                u8_truth = torch.clamp(torch.round(delta + exact_batched), 0.0, float(UINT16_MAX))
            u8_tables = u8_truth.detach().cpu().numpy().astype(np.uint16)
            if np.any(u8_tables[:, 0, :]) or np.any(u8_tables[:, :, 0]):
                raise RuntimeError("decoded U8 LUT violates the zero-operand contract")
            signed_tables = u8_lut_to_s8s8_sign_magnitude(u8_tables)
            _validate_s8_zero_operand_contract(signed_tables)
            tables[row_indices] = signed_tables
            pending_commit.append(row_indices)
            if batch_number % int(flush_every_batches) == 0 or start + len(row_indices) == len(pending):
                tables.flush()
                committed = np.concatenate(pending_commit)
                completed[committed] = True
                completed.flush()
                pending_commit.clear()
                manifest.update(
                    {
                        "completed_row_count": int(np.count_nonzero(completed)),
                        "decoder_num_slots": int(decoder_slots),
                        "last_progress_unix": time.time(),
                    }
                )
                _write_json(paths["progress"], manifest)
                _write_json(paths["manifest"], manifest)
    finally:
        del tables
        del completed
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed_final = np.load(paths["completed"], allow_pickle=False)
    if not bool(np.all(completed_final)):
        raise RuntimeError("fixed LUT asset materialization did not complete every row")
    final_tables = np.load(paths["tables"], mmap_mode="r", allow_pickle=False)
    for start in range(0, len(population), int(batch_rows)):
        _validate_s8_zero_operand_contract(final_tables[start : start + int(batch_rows)])
    manifest.update(
        {
            "complete": True,
            "completed_row_count": int(len(population)),
            "completed_at_unix": time.time(),
            "files": {
                POPULATION_FILENAME: _sha1_file(paths["population"]),
                RUNTIME_IDS_FILENAME: _sha1_file(paths["runtime_ids"]),
                COMPLETED_FILENAME: _sha1_file(paths["completed"]),
                TABLE_FILENAME: _sha1_file(paths["tables"]),
            },
        }
    )
    _write_json(paths["progress"], manifest)
    _write_json(paths["manifest"], manifest)
    _validate_existing_asset(paths, manifest, expected=expected, runtime_ids=runtime_ids)
    return {"status": "completed", "asset_root": str(paths["root"]), **manifest}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize the fixed low-only 20k S8S8 multiplier-LUT asset.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--source-population",
        type=Path,
        required=True,
        help="Pre-transformation Swin-Tiny population CSV used to construct the S8S8 asset.",
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=64)
    parser.add_argument("--flush-every-batches", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--row-limit", type=int, default=None, help="Build a partial smoke asset only; omitted builds all 20k rows.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    result = materialize_lowonly20k_s8s8_asset(
        output_root=args.output_root,
        source_population_path=args.source_population,
        catalog_path=args.catalog,
        checkpoint_path=args.checkpoint,
        device=str(args.device),
        batch_rows=int(args.batch_rows),
        flush_every_batches=int(args.flush_every_batches),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        row_limit=args.row_limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
