"""Validate and promote a finished Transformer dataset run to canonical storage."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from aaae.inference_runtime.transformer.dataset_generation.profiles import (
    DATASET_PROFILES,
    RAW_PROTOCOL_COUNT,
    TRUE_PROTOCOL_COUNT,
    TransformerDatasetProfile,
    get_dataset_profile,
)
from aaae.accuracy_predictor.transformer.proxy_training.transformer_ame_bank import compute_clean_bank


DATA_SCHEMA_VERSION = "transformer_s8s8_linear48_strict50_true1000_raw500_signed_static_v4"
LEGACY_SWIN_SCHEMA_VERSION = "swin_s8s8_strict50_true1000_raw500_signed_static_v2"
PROMOTION_SCHEMA_VERSION = "canonical_dataset_promotion_v1"
FULL_SPLIT_COUNTS = {"train": 14_000, "val": 3_000, "test": 3_000}

DATA_FILES = (
    "common3103.npy",
    "configuration_index.csv",
    "labels.csv",
    "merged_index.csv",
    "runtime_bank.npz",
    "runtime_design_id_map.csv",
    "targets.npy",
    "true1000_label_bank.npz",
)
SHARED_FILES = (
    "canonical_exact_baseline.npz",
    "canonical_exact_contract.json",
    "frozen_image_protocols.json",
    "input_contract.json",
    "strict50_bank_meta_contract.json",
    "strict50_bank_meta.npz",
    "true1000_exact_meta.npz",
    "union_inputs.npy",
    "union_labels.npy",
    "union_positions.npz",
)


@dataclass(frozen=True)
class PromotionRequirements:
    """Canonical protocol values; production callers use the full defaults."""

    row_count: int = 20_000
    split_counts: Mapping[str, int] | None = None
    common_feature_count: int = 3103
    true_sample_count: int = TRUE_PROTOCOL_COUNT
    raw_sample_count: int = RAW_PROTOCOL_COUNT
    require_complete_run_provenance: bool = True

    def normalized_split_counts(self) -> dict[str, int]:
        values = FULL_SPLIT_COUNTS if self.split_counts is None else self.split_counts
        return {str(name): int(count) for name, count in values.items()}


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_regular_files(root: Path, relative_paths: Sequence[str]) -> None:
    invalid = [
        str(root / value)
        for value in relative_paths
        if not (root / value).is_file() or (root / value).is_symlink()
    ]
    if invalid:
        raise FileNotFoundError(f"canonical source files are missing or not regular files: {invalid}")


def _csv_row_count(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _row in csv.DictReader(handle))


def _array_sha1(value: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(value).tobytes()).hexdigest()


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _require_array(
    payload: Mapping[str, np.ndarray],
    *,
    key: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    source: Path,
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"{source} is missing {key}")
    value = np.asarray(payload[key])
    if value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError(
            f"{source}:{key} has dtype/shape {value.dtype} {value.shape}; expected {np.dtype(dtype)} {shape}"
        )
    return value


def _require_string_array(
    payload: Mapping[str, np.ndarray],
    *,
    key: str,
    count: int,
    source: Path,
) -> np.ndarray:
    if key not in payload:
        raise ValueError(f"{source} is missing {key}")
    value = np.asarray(payload[key])
    if value.shape != (int(count),) or value.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{source}:{key} must be a one-dimensional Unicode or byte-string array")
    return value.astype(str, copy=False)


def _require_binary(value: np.ndarray, *, key: str, source: Path) -> None:
    if np.any(value > 1):
        raise ValueError(f"{source}:{key} contains values outside [0, 1]")


def _read_identity_csv(path: Path, *, expected_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"row_index", "design_id", "runtime_design_id"}
    if len(rows) != int(expected_rows) or any(not required.issubset(row) for row in rows):
        raise ValueError(f"{path} does not contain the canonical identity columns")
    try:
        row_index = np.asarray([int(row["row_index"]) for row in rows], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(f"{path} has a non-integer row_index") from exc
    return (
        row_index,
        np.asarray([row["design_id"] for row in rows], dtype=str),
        np.asarray([row["runtime_design_id"] for row in rows], dtype=str),
    )


def _read_float_column(
    rows: Sequence[Mapping[str, str]],
    *,
    key: str,
    source: Path,
) -> np.ndarray:
    try:
        return np.asarray([float(row[key]) for row in rows], dtype=np.float32)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{source} has no valid {key} column") from exc


def _validate_block_plan(run_config: Mapping[str, Any], *, row_count: int, blocks_verified: int) -> None:
    plan = run_config.get("plan")
    if not isinstance(plan, Mapping):
        raise ValueError("source run_config has no block plan")
    blocks = plan.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("source run_config block plan is empty")
    if int(plan.get("selected_row_count", -1)) != int(row_count):
        raise ValueError("source block plan row count does not match the canonical population")
    if int(plan.get("block_count", -1)) != len(blocks) or int(blocks_verified) != len(blocks):
        raise ValueError("source block counts do not match the verified data manifest")
    cursor = 0
    for expected_id, block in enumerate(blocks):
        if not isinstance(block, Mapping):
            raise ValueError("source block plan contains a non-object entry")
        start = int(block.get("start", -1))
        stop = int(block.get("stop", -1))
        if int(block.get("block_id", -1)) != expected_id or start != cursor or stop <= start:
            raise ValueError("source block plan is not contiguous and ordered")
        cursor = stop
    if cursor != int(row_count):
        raise ValueError("source block plan does not cover the full canonical population")


def _expected_shapes(requirements: PromotionRequirements) -> dict[str, list[int]]:
    rows = int(requirements.row_count)
    raw = int(requirements.raw_sample_count)
    true = int(requirements.true_sample_count)
    return {
        "common3103": [rows, int(requirements.common_feature_count)],
        "targets": [rows, 2],
        "runtime_delta_margin_top1": [rows, raw],
        "runtime_delta_margin_top5": [rows, raw],
        "runtime_approx_top1_correct_bits": [rows, raw],
        "runtime_approx_top5_correct_bits": [rows, raw],
        "true1000_approx_top1_correct_bits": [rows, true],
        "true1000_approx_top5_correct_bits": [rows, true],
    }


def _validate_bank_payloads(
    source: Path,
    *,
    requirements: PromotionRequirements,
    run_fingerprint: str,
    exact_contract: Mapping[str, Any],
    bank_contract: Mapping[str, Any],
) -> None:
    """Cross-check every persisted numerical bank before canonical publication."""

    row_count = int(requirements.row_count)
    true_count = int(requirements.true_sample_count)
    raw_count = int(requirements.raw_sample_count)
    data = source / "data"
    shared = source / "shared"
    expected_rows, expected_public, expected_runtime = _read_identity_csv(
        data / "runtime_design_id_map.csv", expected_rows=row_count
    )
    configuration_rows, configuration_public, configuration_runtime = _read_identity_csv(
        data / "configuration_index.csv", expected_rows=row_count
    )
    if (
        not np.array_equal(configuration_rows, expected_rows)
        or not np.array_equal(configuration_public, expected_public)
        or not np.array_equal(configuration_runtime, expected_runtime)
    ):
        raise ValueError("configuration_index.csv does not align with runtime_design_id_map.csv")

    targets = np.load(data / "targets.npy", mmap_mode="r")
    if targets.dtype != np.float32 or targets.shape != (row_count, 2):
        raise ValueError("targets.npy does not satisfy the canonical float32 [N,2] contract")
    targets_array = np.asarray(targets)
    if not np.isfinite(targets_array).all():
        raise ValueError("targets.npy contains non-finite values")

    runtime_path = data / "runtime_bank.npz"
    runtime = _load_npz(runtime_path)
    runtime_rows = _require_array(
        runtime, key="row_index", dtype=np.int64, shape=(row_count,), source=runtime_path
    )
    runtime_public = _require_string_array(runtime, key="public_design_id", count=row_count, source=runtime_path)
    runtime_design = _require_string_array(runtime, key="design_id", count=row_count, source=runtime_path)
    if (
        not np.array_equal(runtime_rows, expected_rows)
        or not np.array_equal(runtime_public, expected_public)
        or not np.array_equal(runtime_design, expected_runtime)
    ):
        raise ValueError("runtime_bank identities do not align with runtime_design_id_map.csv")
    runtime_matrices: dict[str, np.ndarray] = {}
    for key in ("delta_margin_top1", "delta_margin_top5"):
        value = _require_array(
            runtime, key=key, dtype=np.float32, shape=(row_count, raw_count), source=runtime_path
        )
        if not np.isfinite(value).all():
            raise ValueError(f"{runtime_path}:{key} contains non-finite values")
        runtime_matrices[key] = value
    for key in ("approx_top1_correct_bits", "approx_top5_correct_bits"):
        value = _require_array(
            runtime, key=key, dtype=np.uint8, shape=(row_count, raw_count), source=runtime_path
        )
        _require_binary(value, key=key, source=runtime_path)
        runtime_matrices[key] = value
    runtime_drops: dict[str, np.ndarray] = {}
    for key in (
        "top1_drop",
        "top5_drop",
        "true1000_top1_drop",
        "true1000_top5_drop",
        "raw500_top1_drop",
        "raw500_top5_drop",
    ):
        value = _require_array(runtime, key=key, dtype=np.float32, shape=(row_count,), source=runtime_path)
        if not np.isfinite(value).all():
            raise ValueError(f"{runtime_path}:{key} contains non-finite values")
        runtime_drops[key] = value
    if (
        not np.array_equal(runtime_drops["true1000_top1_drop"], targets_array[:, 0])
        or not np.array_equal(runtime_drops["true1000_top5_drop"], targets_array[:, 1])
        or not np.array_equal(runtime_drops["top1_drop"], runtime_drops["raw500_top1_drop"])
        or not np.array_equal(runtime_drops["top5_drop"], runtime_drops["raw500_top5_drop"])
    ):
        raise ValueError("runtime_bank target diagnostics do not match their canonical counterparts")

    labels_path = data / "labels.csv"
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        label_rows = list(csv.DictReader(handle))
    if len(label_rows) != row_count:
        raise ValueError("labels.csv does not contain the canonical population")
    label_ids = _read_identity_csv(labels_path, expected_rows=row_count)
    if any(not np.array_equal(actual, expected) for actual, expected in zip(label_ids, (expected_rows, expected_public, expected_runtime), strict=True)):
        raise ValueError("labels.csv identities do not align with runtime_design_id_map.csv")
    if (
        not np.allclose(_read_float_column(label_rows, key="top1_drop", source=labels_path), targets_array[:, 0], rtol=0.0, atol=1.0e-7)
        or not np.allclose(_read_float_column(label_rows, key="top5_drop", source=labels_path), targets_array[:, 1], rtol=0.0, atol=1.0e-7)
        or not np.allclose(_read_float_column(label_rows, key="raw500_top1_drop", source=labels_path), runtime_drops["raw500_top1_drop"], rtol=0.0, atol=1.0e-7)
        or not np.allclose(_read_float_column(label_rows, key="raw500_top5_drop", source=labels_path), runtime_drops["raw500_top5_drop"], rtol=0.0, atol=1.0e-7)
    ):
        raise ValueError("labels.csv targets do not align with the persisted banks")

    merged_path = data / "merged_index.csv"
    with merged_path.open("r", encoding="utf-8", newline="") as handle:
        merged_rows = list(csv.DictReader(handle))
    if len(merged_rows) != row_count:
        raise ValueError("merged_index.csv does not contain the canonical population")
    try:
        merged_indices = np.asarray([int(row["row_index"]) for row in merged_rows], dtype=np.int64)
        merged_designs = np.asarray([row["design_id"] for row in merged_rows], dtype=str)
    except (KeyError, ValueError) as exc:
        raise ValueError("merged_index.csv has no valid identity columns") from exc
    if (
        not np.array_equal(merged_indices, expected_rows)
        or not np.array_equal(merged_designs, expected_public)
        or not np.allclose(_read_float_column(merged_rows, key="top1_drop", source=merged_path), targets_array[:, 0], rtol=0.0, atol=1.0e-7)
        or not np.allclose(_read_float_column(merged_rows, key="top5_drop", source=merged_path), targets_array[:, 1], rtol=0.0, atol=1.0e-7)
    ):
        raise ValueError("merged_index.csv does not align with targets.npy")

    true_path = data / "true1000_label_bank.npz"
    true_bank = _load_npz(true_path)
    true_rows = _require_array(
        true_bank, key="row_index", dtype=np.int64, shape=(row_count,), source=true_path
    )
    true_public = _require_string_array(true_bank, key="public_design_id", count=row_count, source=true_path)
    true_design = _require_string_array(true_bank, key="design_id", count=row_count, source=true_path)
    if (
        not np.array_equal(true_rows, expected_rows)
        or not np.array_equal(true_public, expected_public)
        or not np.array_equal(true_design, expected_runtime)
    ):
        raise ValueError("true1000_label_bank identities do not align with runtime_design_id_map.csv")
    true_approx: dict[str, np.ndarray] = {}
    for key in ("approx_top1_correct_bits", "approx_top5_correct_bits"):
        value = _require_array(
            true_bank, key=key, dtype=np.uint8, shape=(row_count, true_count), source=true_path
        )
        _require_binary(value, key=key, source=true_path)
        true_approx[key] = value
    true_drops = {
        key: _require_array(true_bank, key=key, dtype=np.float32, shape=(row_count,), source=true_path)
        for key in ("top1_drop", "top5_drop")
    }
    if (
        not np.array_equal(true_drops["top1_drop"], targets_array[:, 0])
        or not np.array_equal(true_drops["top5_drop"], targets_array[:, 1])
    ):
        raise ValueError("true1000_label_bank drops do not match targets.npy")

    if not requirements.require_complete_run_provenance:
        return

    input_contract = _read_json(shared / "input_contract.json")
    expected_artifacts = {
        name: _sha1_file(shared / filename)
        for name, filename in {
            "union_inputs": "union_inputs.npy",
            "union_labels": "union_labels.npy",
            "union_positions": "union_positions.npz",
            "image_protocols": "frozen_image_protocols.json",
        }.items()
    }
    if input_contract.get("artifact_sha1") != expected_artifacts:
        raise ValueError("frozen input artifact digests do not match input_contract.json")
    if int(input_contract.get("union_count", -1)) <= 0:
        raise ValueError("input_contract.json has no positive frozen union count")
    union_count = int(input_contract["union_count"])
    union_inputs = np.load(shared / "union_inputs.npy", mmap_mode="r")
    if union_inputs.dtype != np.float32 or union_inputs.shape != (union_count, 3, 224, 224):
        raise ValueError("union_inputs.npy does not satisfy the frozen NCHW contract")
    union_labels = np.load(shared / "union_labels.npy", allow_pickle=False)
    if union_labels.dtype != np.int64 or union_labels.shape != (union_count,):
        raise ValueError("union_labels.npy does not satisfy the frozen label contract")
    positions = _load_npz(shared / "union_positions.npz")
    true_positions = _require_array(
        positions,
        key="true1000_positions",
        dtype=np.int64,
        shape=(true_count,),
        source=shared / "union_positions.npz",
    )
    raw_positions = _require_array(
        positions,
        key="raw500_positions",
        dtype=np.int64,
        shape=(raw_count,),
        source=shared / "union_positions.npz",
    )
    if (
        np.any(true_positions < 0)
        or np.any(true_positions >= union_count)
        or np.any(raw_positions < 0)
        or np.any(raw_positions >= union_count)
        or len(np.unique(true_positions)) != true_count
        or len(np.unique(raw_positions)) != raw_count
    ):
        raise ValueError("frozen union protocol positions are invalid")

    exact_path = shared / "canonical_exact_baseline.npz"
    exact = _load_npz(exact_path)
    logits = _require_array(
        exact, key="logits", dtype=np.float32, shape=(union_count, 1000), source=exact_path
    )
    exact_labels = _require_array(
        exact, key="union_labels", dtype=np.int64, shape=(union_count,), source=exact_path
    )
    exact_true_positions = _require_array(
        exact,
        key="true1000_positions",
        dtype=np.int64,
        shape=(true_count,),
        source=exact_path,
    )
    exact_raw_positions = _require_array(
        exact,
        key="raw500_positions",
        dtype=np.int64,
        shape=(raw_count,),
        source=exact_path,
    )
    if (
        not np.array_equal(exact_labels, union_labels)
        or not np.array_equal(exact_true_positions, true_positions)
        or not np.array_equal(exact_raw_positions, raw_positions)
        or exact_contract.get("run_fingerprint") != run_fingerprint
        or exact_contract.get("exact_logits_sha1") != _array_sha1(logits)
        or exact_contract.get("union_labels_sha1") != _array_sha1(exact_labels)
    ):
        raise ValueError("canonical exact baseline does not match its frozen-input contract")

    def clean_meta(
        path: Path,
        *,
        expected_labels: np.ndarray,
        expected_positions: np.ndarray,
    ) -> dict[str, np.ndarray]:
        payload = _load_npz(path)
        result = {
            "clean_margin_top1": _require_array(payload, key="clean_margin_top1", dtype=np.float32, shape=expected_labels.shape, source=path),
            "clean_margin_top5": _require_array(payload, key="clean_margin_top5", dtype=np.float32, shape=expected_labels.shape, source=path),
            "exact_top1_correct_bits": _require_array(payload, key="exact_top1_correct_bits", dtype=np.uint8, shape=expected_labels.shape, source=path),
            "exact_top5_correct_bits": _require_array(payload, key="exact_top5_correct_bits", dtype=np.uint8, shape=expected_labels.shape, source=path),
            "clean_entropy": _require_array(payload, key="clean_entropy", dtype=np.float32, shape=expected_labels.shape, source=path),
            "exact_pred_class": _require_array(payload, key="exact_pred_class", dtype=np.int64, shape=expected_labels.shape, source=path),
            "true_label": _require_array(payload, key="true_label", dtype=np.int64, shape=expected_labels.shape, source=path),
        }
        if not np.array_equal(result["true_label"], expected_labels):
            raise ValueError(f"{path} labels do not match canonical exact baseline")
        expected_clean = compute_clean_bank(logits[expected_positions], expected_labels)
        if any(not np.array_equal(result[key], expected_clean[key]) for key in result):
            raise ValueError(f"{path} clean metadata does not match canonical logits")
        return result

    true_meta_path = shared / "true1000_exact_meta.npz"
    raw_meta_path = shared / "strict50_bank_meta.npz"
    true_meta = clean_meta(
        true_meta_path,
        expected_labels=exact_labels[true_positions],
        expected_positions=true_positions,
    )
    raw_meta = clean_meta(
        raw_meta_path,
        expected_labels=exact_labels[raw_positions],
        expected_positions=raw_positions,
    )
    for key, value in raw_meta.items():
        _require_binary(value, key=key, source=raw_meta_path) if key.endswith("bits") else None
    raw_meta_payload = _load_npz(raw_meta_path)
    for key in ("bank_mean_top1", "bank_mean_top5"):
        value = _require_array(
            raw_meta_payload, key=key, dtype=np.float32, shape=(raw_count,), source=raw_meta_path
        )
        if not np.isfinite(value).all():
            raise ValueError(f"{raw_meta_path}:{key} contains non-finite values")
    basis_top1 = np.asarray(raw_meta_payload.get("bank_basis_top1"))
    basis_top5 = np.asarray(raw_meta_payload.get("bank_basis_top5"))
    if (
        basis_top1.dtype != np.float32
        or basis_top1.ndim != 2
        or basis_top1.shape[1:] != (raw_count,)
        or basis_top5.dtype != np.float32
        or basis_top5.shape != basis_top1.shape
        or not np.isfinite(basis_top1).all()
        or not np.isfinite(basis_top5).all()
    ):
        raise ValueError("strict50 bank basis arrays do not satisfy their numerical contract")
    basis_rank = int(basis_top1.shape[0])
    for key in ("coeff_mean_top1", "coeff_std_top1", "coeff_mean_top5", "coeff_std_top5"):
        value = _require_array(
            raw_meta_payload, key=key, dtype=np.float32, shape=(basis_rank,), source=raw_meta_path
        )
        if not np.isfinite(value).all():
            raise ValueError(f"{raw_meta_path}:{key} contains non-finite values")
    basis_ready = _require_array(
        raw_meta_payload, key="basis_ready", dtype=np.int64, shape=(), source=raw_meta_path
    )
    if int(basis_ready) not in (0, 1):
        raise ValueError("strict50 bank basis_ready must be 0 or 1")
    true_exact_top1 = _require_array(
        true_bank, key="exact_top1_correct_bits", dtype=np.uint8, shape=(true_count,), source=true_path
    )
    true_exact_top5 = _require_array(
        true_bank, key="exact_top5_correct_bits", dtype=np.uint8, shape=(true_count,), source=true_path
    )
    if (
        not np.array_equal(true_exact_top1, true_meta["exact_top1_correct_bits"])
        or not np.array_equal(true_exact_top5, true_meta["exact_top5_correct_bits"])
    ):
        raise ValueError("true1000_label_bank exact bits do not match true1000_exact_meta.npz")

    expected_true_top1 = np.mean(
        true_meta["exact_top1_correct_bits"].astype(np.float32)[None, :]
        - true_approx["approx_top1_correct_bits"].astype(np.float32),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    expected_true_top5 = np.mean(
        true_meta["exact_top5_correct_bits"].astype(np.float32)[None, :]
        - true_approx["approx_top5_correct_bits"].astype(np.float32),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    expected_raw_top1 = np.mean(
        raw_meta["exact_top1_correct_bits"].astype(np.float32)[None, :]
        - runtime_matrices["approx_top1_correct_bits"].astype(np.float32),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    expected_raw_top5 = np.mean(
        raw_meta["exact_top5_correct_bits"].astype(np.float32)[None, :]
        - runtime_matrices["approx_top5_correct_bits"].astype(np.float32),
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    if (
        not np.array_equal(targets_array[:, 0], expected_true_top1)
        or not np.array_equal(targets_array[:, 1], expected_true_top5)
        or not np.array_equal(runtime_drops["raw500_top1_drop"], expected_raw_top1)
        or not np.array_equal(runtime_drops["raw500_top5_drop"], expected_raw_top5)
    ):
        raise ValueError("bank correctness bits do not reproduce their published drop targets")
    if (
        bank_contract.get("run_fingerprint") != run_fingerprint
        or bank_contract.get("sample_count") != raw_count
        or bank_contract.get("bank_meta_sha1") != _sha1_file(raw_meta_path)
        or bank_contract.get("true1000_meta_sha1") != _sha1_file(true_meta_path)
        or bank_contract.get("canonical_exact_contract_sha1") != _sha1_file(shared / "canonical_exact_contract.json")
        or bank_contract.get("canonical_exact_logits_sha1") != _array_sha1(logits)
    ):
        raise ValueError("strict50 bank metadata contract is inconsistent with its numerical payloads")


def validate_source_run(
    source_run_root: Path,
    *,
    profile: TransformerDatasetProfile,
    requirements: PromotionRequirements | None = None,
) -> dict[str, Any]:
    """Validate a completed staging run without mutating it."""

    requirements = requirements or PromotionRequirements()
    source = Path(source_run_root).expanduser().resolve()
    required_relative = (
        *(f"data/{name}" for name in DATA_FILES),
        "data/manifest.json",
        *(f"shared/{name}" for name in SHARED_FILES),
        "run_config.json",
        "population/selected_rows.csv",
    )
    _require_regular_files(source, required_relative)

    manifest_path = source / "data" / "manifest.json"
    run_config_path = source / "run_config.json"
    selected_rows_path = source / "population" / "selected_rows.csv"
    manifest = _read_json(manifest_path)
    run_config = _read_json(run_config_path)
    contract = run_config.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("source run_config has no immutable contract")
    request = run_config.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("source run_config has no immutable request")
    if any(contract.get(key) != value for key, value in request.items()):
        raise ValueError("source request is inconsistent with its immutable contract")

    schema = str(manifest.get("schema_version", ""))
    if manifest.get("valid") is not True or schema != DATA_SCHEMA_VERSION:
        raise ValueError("source data manifest is not a valid supported Transformer dataset")
    if run_config.get("schema_version") != schema or contract.get("schema_version") != schema:
        raise ValueError("source run_config and data manifest schemas do not match")
    if manifest.get("model_key") != profile.model_key:
        raise ValueError("source data manifest belongs to a different model")
    if contract.get("model_key", profile.model_key) != profile.model_key:
        raise ValueError("source immutable contract belongs to a different model")
    expected_injection_mode = profile.injection_mode_contract()
    if any(
        value.get("injection_mode") != expected_injection_mode
        for value in (request, contract, manifest)
    ):
        raise ValueError("source run does not use the canonical linear48 injection scope")
    canonical_registry_sha1 = _sha1_file(profile.node_registry_path)
    if any(
        value.get("node_registry_sha1") != canonical_registry_sha1
        for value in (request, contract, manifest)
    ):
        raise ValueError("source node registry does not match the canonical model registry")
    canonical_model_sha1 = _sha1_file(profile.dynamic_model_path)
    if contract.get("model_sha1") != canonical_model_sha1 or manifest.get("model_sha1") != canonical_model_sha1:
        raise ValueError("source model does not match the canonical dynamic ONNX model")
    expected_label_protocol_hash = _sha1_json(
        {
            "protocol": "canonical_dynamic_int8_true1000",
            "model_sha1": canonical_model_sha1,
            "input_contract_sha1": contract.get("frozen_input_contract_sha1"),
            "batch_size": int(contract.get("batch_size", -1)),
            "session_profile": "throughput",
            "injection_scope_id": profile.injection_scope_id,
            "node_registry_sha1": canonical_registry_sha1,
        }
    )
    if contract.get("label_protocol_hash") != expected_label_protocol_hash:
        raise ValueError("source label protocol hash is inconsistent with the linear48 contract")
    if manifest.get("label_protocol_hash") != expected_label_protocol_hash:
        raise ValueError("source manifest has the wrong label protocol hash")

    row_count = int(requirements.row_count)
    expected_splits = requirements.normalized_split_counts()
    if int(manifest.get("row_count", -1)) != row_count:
        formatted = f"{row_count:,}"
        raise ValueError(f"canonical promotion requires exactly {formatted} rows")
    manifest_feature_count = manifest.get("common_feature_count")
    if manifest_feature_count is None:
        common_shape = manifest.get("shapes", {}).get("common3103", ())
        manifest_feature_count = common_shape[1] if len(common_shape) == 2 else None
    if int(manifest_feature_count or -1) != int(requirements.common_feature_count):
        raise ValueError("source common feature count is not canonical")
    if dict(manifest.get("split_counts", {})) != expected_splits:
        raise ValueError("source split counts do not match the canonical split")
    if dict(manifest.get("expected_full_split_counts", expected_splits)) != expected_splits:
        raise ValueError("source expected split contract does not match the canonical split")

    training = manifest.get("training_label_protocol", {})
    runtime = manifest.get("runtime_observation_protocol", {})
    if (
        not isinstance(training, Mapping)
        or training.get("kind") != "canonical_dynamic_int8_true1000"
        or int(training.get("sample_count", -1)) != int(requirements.true_sample_count)
        or training.get("label_protocol_hash") != expected_label_protocol_hash
    ):
        raise ValueError("source true1000 training-label protocol is not canonical")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("kind") != "canonical_dynamic_int8_raw500"
        or int(runtime.get("sample_count", -1)) != int(requirements.raw_sample_count)
    ):
        raise ValueError("source raw500 runtime-observation protocol is not canonical")

    actual_shapes = manifest.get("shapes")
    if not isinstance(actual_shapes, Mapping):
        raise ValueError("source data manifest has no shape contract")
    for name, expected in _expected_shapes(requirements).items():
        if list(actual_shapes.get(name, [])) != expected:
            raise ValueError(f"source shape {name} is not canonical: {actual_shapes.get(name)}")

    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("source data manifest has no file checksums")
    if set(files) != set(DATA_FILES):
        raise ValueError("source data manifest file allowlist is not canonical")
    for name in DATA_FILES:
        actual_sha1 = _sha1_file(source / "data" / name)
        if files.get(name) != actual_sha1:
            raise ValueError(f"source data checksum mismatch: {name}")
    common = np.load(source / "data" / "common3103.npy", mmap_mode="r")
    targets = np.load(source / "data" / "targets.npy", mmap_mode="r")
    if common.dtype != np.float32 or list(common.shape) != _expected_shapes(requirements)["common3103"]:
        raise ValueError("common3103.npy does not match its float32 shape contract")
    if targets.dtype != np.float32 or list(targets.shape) != _expected_shapes(requirements)["targets"]:
        raise ValueError("targets.npy does not match its float32 shape contract")
    del common, targets

    fingerprint = _sha1_json(contract)
    if run_config.get("run_fingerprint") != fingerprint or manifest.get("run_fingerprint") != fingerprint:
        raise ValueError("source run fingerprint is inconsistent")
    if contract.get("max_designs") is not None:
        raise ValueError("smoke or truncated runs cannot be promoted to canonical storage")
    if requirements.require_complete_run_provenance:
        _validate_block_plan(
            run_config,
            row_count=row_count,
            blocks_verified=int(manifest.get("blocks_verified", -1)),
        )
    if _csv_row_count(selected_rows_path) != row_count:
        raise ValueError("selected_rows.csv does not contain the canonical population")
    if _csv_row_count(source / "data" / "configuration_index.csv") != row_count:
        raise ValueError("configuration_index.csv does not contain the canonical population")
    for filename in ("configuration_index.csv", "labels.csv"):
        with (source / "data" / filename).open("r", encoding="utf-8", newline="") as handle:
            values = {str(row.get("label_protocol_hash", "")) for row in csv.DictReader(handle)}
        if values != {expected_label_protocol_hash}:
            raise ValueError(f"{filename} does not carry one canonical label protocol hash")
    if requirements.require_complete_run_provenance and (
        contract.get("selected_population_sha1") != _sha1_file(selected_rows_path)
    ):
        raise ValueError("selected population checksum does not match the immutable contract")

    input_contract_path = source / "shared" / "input_contract.json"
    exact_contract_path = source / "shared" / "canonical_exact_contract.json"
    bank_contract_path = source / "shared" / "strict50_bank_meta_contract.json"
    bank_meta_path = source / "shared" / "strict50_bank_meta.npz"
    true_meta_path = source / "shared" / "true1000_exact_meta.npz"
    if requirements.require_complete_run_provenance and (
        contract.get("frozen_input_contract_sha1") != _sha1_file(input_contract_path)
    ):
        raise ValueError("frozen input contract checksum is inconsistent")
    exact_contract = _read_json(exact_contract_path)
    bank_contract = _read_json(bank_contract_path)
    exact_contract_sha1 = _sha1_file(exact_contract_path)
    bank_meta_sha1 = _sha1_file(bank_meta_path)
    if requirements.require_complete_run_provenance and (
        exact_contract.get("run_fingerprint") != fingerprint
    ):
        raise ValueError("canonical exact baseline belongs to a different source run")
    if requirements.require_complete_run_provenance and (
        runtime.get("canonical_exact_contract_sha1") != exact_contract_sha1
    ):
        raise ValueError("canonical exact contract checksum is inconsistent")
    if runtime.get("bank_meta_sha1") != bank_meta_sha1:
        raise ValueError("strict50 bank metadata checksum is inconsistent")
    if requirements.require_complete_run_provenance and (
        bank_contract.get("run_fingerprint") != fingerprint
        or bank_contract.get("canonical_exact_contract_sha1") != exact_contract_sha1
        or bank_contract.get("bank_meta_sha1") != bank_meta_sha1
    ):
        raise ValueError("strict50 bank metadata contract is inconsistent")
    if bank_contract.get("true1000_meta_sha1") not in (None, _sha1_file(true_meta_path)):
        raise ValueError("true1000 metadata checksum is inconsistent")

    _validate_bank_payloads(
        source,
        requirements=requirements,
        run_fingerprint=fingerprint,
        exact_contract=exact_contract,
        bank_contract=bank_contract,
    )
    source_file_sha1 = {
        **{f"data/{name}": _sha1_file(source / "data" / name) for name in DATA_FILES},
        **{f"shared/{name}": _sha1_file(source / "shared" / name) for name in SHARED_FILES},
        "provenance/source_run_config.json": _sha1_file(run_config_path),
        "provenance/selected_rows.csv": _sha1_file(selected_rows_path),
    }

    return {
        "source_run_root": source,
        "manifest": manifest,
        "manifest_sha1": _sha1_file(manifest_path),
        "run_config": run_config,
        "run_config_sha1": _sha1_file(run_config_path),
        "selected_rows_sha1": _sha1_file(selected_rows_path),
        "source_file_sha1": source_file_sha1,
        "run_fingerprint": fingerprint,
        "requirements": requirements,
    }


def _canonical_readme(
    *,
    profile: TransformerDatasetProfile,
    data_schema: str,
    requirements: PromotionRequirements,
) -> str:
    return f"""# {profile.model_key} Signed-S8S8 Strict-50 Dataset

Canonical promoted copy of the signed-S8S8 strict-50 dataset run.

- Dataset ID: `{profile.dataset_id}`
- Data schema: `{data_schema}`
- Injection scope: `{profile.injection_scope_id}` ({profile.expected_injected_matmuls} MatMuls)
- Population: {requirements.row_count:,} designs
- Training labels: frozen `true1000` protocol
- Runtime observations: frozen `raw500` protocol

Use `data/` for features, labels, splits, and runtime observations. Use
`shared/strict50_bank_meta.npz` with `data/runtime_bank.npz` for strict-50
runtime calibration. Source-run provenance is recorded in `provenance/`.
"""


def _verify_staged_package(
    root: Path,
    *,
    profile: TransformerDatasetProfile,
    copied_sha1: Mapping[str, str],
) -> None:
    manifest = _read_json(root / "data" / "manifest.json")
    if manifest.get("dataset_id") != profile.dataset_id:
        raise ValueError("staged canonical manifest has the wrong dataset ID")
    if manifest.get("injection_mode") != profile.injection_mode_contract():
        raise ValueError("staged canonical manifest has the wrong injection scope")
    if manifest.get("node_registry_sha1") != _sha1_file(profile.node_registry_path):
        raise ValueError("staged canonical manifest has the wrong node registry")
    runtime = manifest.get("runtime_observation_protocol", {})
    if runtime.get("bank_meta") != "../shared/strict50_bank_meta.npz":
        raise ValueError("staged canonical bank metadata path is not portable")
    if set(manifest.get("files", {})) != set(DATA_FILES):
        raise ValueError("staged canonical data manifest file allowlist is not canonical")
    for relative, expected_sha1 in copied_sha1.items():
        if _sha1_file(root / relative) != expected_sha1:
            raise ValueError(f"staged canonical copy checksum mismatch: {relative}")
    for name, expected_sha1 in manifest["files"].items():
        if _sha1_file(root / "data" / name) != expected_sha1:
            raise ValueError(f"staged canonical data checksum mismatch: {name}")
    if _sha1_file(root / "shared" / "strict50_bank_meta.npz") != runtime.get("bank_meta_sha1"):
        raise ValueError("staged canonical bank metadata checksum mismatch")


def _promote_dataset_with_requirements(
    *,
    source_run_root: Path,
    destination_root: Path,
    profile: TransformerDatasetProfile,
    requirements: PromotionRequirements,
) -> dict[str, Any]:
    """Internal promotion hook used by focused fixture tests."""

    destination = Path(os.path.abspath(os.fspath(Path(destination_root).expanduser())))
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"canonical destination already exists: {destination}")

    inspection = validate_source_run(
        source_run_root,
        profile=profile,
        requirements=requirements,
    )
    source = Path(inspection["source_run_root"])
    source_resolved = source.resolve()
    destination_resolved = destination.resolve(strict=False)
    if (
        source_resolved == destination_resolved
        or source_resolved in destination_resolved.parents
        or destination_resolved in source_resolved.parents
    ):
        raise ValueError("canonical destination and source run must be separate directories")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.promoting.",
            dir=str(destination.parent),
        )
    )
    published = False
    try:
        (staging / "data").mkdir()
        (staging / "shared").mkdir()
        (staging / "provenance").mkdir()
        copied_sha1: dict[str, str] = {}

        def copy_verified(relative: str, source_path: Path) -> None:
            expected_sha1 = inspection["source_file_sha1"][relative]
            shutil.copy2(source_path, staging / relative)
            actual_sha1 = _sha1_file(staging / relative)
            if actual_sha1 != expected_sha1:
                raise ValueError(f"source changed while promotion was copying: {relative}")
            copied_sha1[relative] = actual_sha1

        for name in DATA_FILES:
            relative = f"data/{name}"
            copy_verified(relative, source / relative)
        for name in SHARED_FILES:
            relative = f"shared/{name}"
            copy_verified(relative, source / relative)
        provenance_sources = {
            "provenance/source_run_config.json": source / "run_config.json",
            "provenance/selected_rows.csv": source / "population" / "selected_rows.csv",
        }
        for relative, source_path in provenance_sources.items():
            copy_verified(relative, source_path)

        canonical_manifest = copy.deepcopy(inspection["manifest"])
        canonical_manifest["dataset_id"] = profile.dataset_id
        canonical_manifest["runtime_observation_protocol"]["bank_meta"] = (
            "../shared/strict50_bank_meta.npz"
        )
        _write_json(staging / "data" / "manifest.json", canonical_manifest)

        promotion_manifest = {
            "schema_version": PROMOTION_SCHEMA_VERSION,
            "dataset_id": profile.dataset_id,
            "data_schema_version": canonical_manifest["schema_version"],
            "injection_scope_id": profile.injection_scope_id,
            "injected_matmul_count": int(profile.expected_injected_matmuls),
            "node_registry_sha1": canonical_manifest["node_registry_sha1"],
            "promoted_on": datetime.now(timezone.utc).date().isoformat(),
            "source_run_root": str(source),
            "source_run_fingerprint": inspection["run_fingerprint"],
            "source_data_manifest_sha1": inspection["manifest_sha1"],
            "canonical_data_manifest_sha1": _sha1_file(staging / "data" / "manifest.json"),
            "source_run_config_sha1": inspection["run_config_sha1"],
            "source_selected_rows_sha1": inspection["selected_rows_sha1"],
            "copied_file_sha1": copied_sha1,
            "data_manifest_adjustments": {
                "dataset_id": "added canonical dataset identity",
                "runtime_observation_protocol.bank_meta": (
                    "rewritten from the source-run absolute path to "
                    "../shared/strict50_bank_meta.npz"
                ),
            },
            "copied_source_paths": [
                "data/",
                "shared/",
                "run_config.json -> provenance/source_run_config.json",
                "population/selected_rows.csv -> provenance/selected_rows.csv",
            ],
            "excluded_source_paths": ["blocks/", "logs/", "workers/"],
        }
        _write_json(staging / "provenance" / "promotion_manifest.json", promotion_manifest)
        (staging / "README.md").write_text(
            _canonical_readme(
                profile=profile,
                data_schema=str(canonical_manifest["schema_version"]),
                requirements=requirements,
            ),
            encoding="utf-8",
        )
        _verify_staged_package(staging, profile=profile, copied_sha1=copied_sha1)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"canonical destination appeared during promotion: {destination}")
        os.rename(staging, destination)
        published = True
        return {
            "status": "promoted",
            "dataset_id": profile.dataset_id,
            "source_run_root": str(source),
            "destination_root": str(destination),
            "row_count": int(requirements.row_count),
            "run_fingerprint": inspection["run_fingerprint"],
            "promotion_manifest": str(destination / "provenance" / "promotion_manifest.json"),
        }
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def promote_dataset(
    *,
    source_run_root: Path,
    destination_root: Path,
    profile: TransformerDatasetProfile,
) -> dict[str, Any]:
    """Atomically publish only the fixed 20k canonical data contract."""

    return _promote_dataset_with_requirements(
        source_run_root=source_run_root,
        destination_root=destination_root,
        profile=profile,
        requirements=PromotionRequirements(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=tuple(sorted(DATASET_PROFILES)), required=True)
    parser.add_argument("--source-run-root", type=Path, default=None)
    parser.add_argument("--destination-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = get_dataset_profile(args.model_key)
    result = promote_dataset(
        source_run_root=args.source_run_root or profile.default_output_root,
        destination_root=args.destination_root or profile.canonical_output_root,
        profile=profile,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
