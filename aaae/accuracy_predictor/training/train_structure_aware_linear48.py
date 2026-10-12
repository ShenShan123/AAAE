#!/usr/bin/env python
"""Train a target-free structure-aware strict-50 Transformer package.

The command keeps the frozen 3,463-dimensional strict-50 contract (including
its selected images, static priors, LightGBM family, and validation
calibration) and appends a 1,034-dimensional target-free physics block.  The
original package is never overwritten; the enhanced package is published as a
new run and can be consumed by the common ``transformer predict`` entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.transformer.artifacts.strict50_package import (
    load_strict50_top1_package,
    save_strict50_top1_package,
    verify_strict50_top1_package_integrity,
)
from aaae.accuracy_predictor.transformer.proxy_training.strict50_calibration import (
    _validation_calibration_offset,
    build_50shot_features_from_selected,
    fit_lightgbm_model,
    predict_lightgbm_model,
)

DATASET_ROOTS = {
    "vit_base_patch16_224": PROJECT_ROOT
    / "datasets"
    / "vit"
    / "s8s8_linear48_strict50_true1000_raw500_v1",
    "deit_base_patch16_224": PROJECT_ROOT
    / "datasets"
    / "deit"
    / "s8s8_linear48_strict50_true1000_raw500_v1",
}
BASE_PACKAGES = {
    "vit_base_patch16_224": PROJECT_ROOT
    / "data"
    / "runs"
    / "vit_base_patch16_224_int8_s8s8_linear48_strict50_true1000_raw500_20k_50shot_v1"
    / "strict50_package",
    "deit_base_patch16_224": PROJECT_ROOT
    / "data"
    / "runs"
    / "deit_base_patch16_224_int8_s8s8_linear48_strict50_true1000_raw500_20k_50shot_v1"
    / "strict50_package",
}
DEFAULT_OUTPUT_RUNS = {
    "vit_base_patch16_224": OUTPUT_ROOT / "accuracy"
    / "vit_base_patch16_224_int8_s8s8_linear48_structure_aware_20k_50shot_v1",
    "deit_base_patch16_224": OUTPUT_ROOT / "accuracy"
    / "deit_base_patch16_224_int8_s8s8_linear48_structure_aware_20k_50shot_v1",
}
DEFAULT_FEATURES = {
    "vit_base_patch16_224": DATASET_ROOTS["vit_base_patch16_224"]
    / "data"
    / "vit_linear48_causal_prefix_second_order1034.float32.npy",
    "deit_base_patch16_224": DATASET_ROOTS["deit_base_patch16_224"]
    / "data"
    / "deit_linear48_causal_prefix_second_order1034.float32.npy",
}
FEATURE_COUNT = 1034
BASE_FEATURE_COUNT = 3463
COMMON_FEATURE_COUNT = 3103
EXPECTED_ROWS = 20000
EXPECTED_SPLIT_COUNTS = (14000, 3000, 3000)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual64 = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted64 = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual64.shape != predicted64.shape or not np.isfinite(predicted64).all():
        raise ValueError("actual and predicted must be aligned finite vectors")
    absolute = np.abs(predicted64 - actual64)
    return {
        "rows": len(actual64),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(absolute * absolute))),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "over_5pp_count": int(np.sum(absolute > 0.05)),
        "at_or_over_5pp_count": int(np.sum(absolute >= 0.05)),
        "bias": float(np.mean(predicted64 - actual64)),
    }


def _metadata_asset_path(metadata_path: Path, raw_path: Any, *, key: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"supplemental metadata {key!r} is not a path")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = metadata_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"supplemental source asset is missing: {candidate}")
    return candidate


def _supplemental_sidecars(feature_path: Path) -> tuple[Path, Path]:
    names_path = feature_path.with_suffix(feature_path.suffix + ".features.json")
    metadata_candidates = (
        feature_path.with_suffix(feature_path.suffix + ".metadata.json"),
        feature_path.with_suffix(feature_path.suffix + ".json"),
    )
    metadata_paths = [path for path in metadata_candidates if path.is_file()]
    if not names_path.is_file() or len(metadata_paths) != 1:
        raise FileNotFoundError(
            "supplemental features require one names sidecar and exactly one "
            f"metadata sidecar: {feature_path}"
        )
    return names_path, metadata_paths[0]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_promoted_supplemental_identity(
    *,
    metadata: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, str]:
    expected = {
        "model_key": runtime_identity.get("model_key"),
        "model_sha1": runtime_identity.get("model_sha1"),
        "node_registry_sha1": runtime_identity.get("node_registry_sha1"),
        "injected_matmul_count": runtime_identity.get("injected_matmul_count"),
        "dataset_id": runtime_identity.get("dataset_id"),
        "injection_scope": runtime_identity.get("injection_scope_id"),
    }
    if metadata.get("promoted_for_formal_inference") is not True or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("promoted supplemental identity differs from base package")
    if (
        int(metadata.get("node_count", -1)) != 48
        or int(metadata.get("block_count", -1)) != 12
        or int(metadata.get("operations_per_block", -1)) != 4
        or metadata.get("operation_names") != ["qkv", "proj", "fc1", "fc2"]
        or metadata.get("block_group_counts") != [3, 3, 3, 3]
        or metadata.get("block_group_label") != "depth_segment"
    ):
        raise ValueError("promoted supplemental topology is not flat linear48")
    hash_keys = (
        "promotion_source_metadata_sha256",
        "operand_profile_metadata_sha256",
        "propagation_profile_metadata_sha256",
    )
    if any(not _is_sha256(metadata.get(key)) for key in hash_keys):
        raise ValueError("promoted supplemental provenance hashes are incomplete")
    return {key: str(metadata[key]) for key in hash_keys}


def _validate_supplemental_source_identity(
    *,
    metadata_path: Path,
    metadata: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> dict[str, str]:
    """Validate causal -> profile -> model/registry provenance end to end."""

    expected_model_sha1 = str(runtime_identity.get("model_sha1", ""))
    expected_registry_sha1 = str(runtime_identity.get("node_registry_sha1", ""))
    expected_scope = str(runtime_identity.get("injection_scope_id", ""))
    expected_nodes = int(runtime_identity.get("injected_matmul_count", -1))
    if (
        not expected_model_sha1
        or not expected_registry_sha1
        or not expected_scope
        or expected_nodes != 48
    ):
        raise ValueError("base package runtime identity is incomplete or not linear48")
    if (
        metadata.get("injection_scope") != expected_scope
        or int(metadata.get("node_count", -1)) != expected_nodes
        or int(metadata.get("block_count", -1)) != 12
        or int(metadata.get("operations_per_block", -1)) != 4
        or metadata.get("operation_names") != ["qkv", "proj", "fc1", "fc2"]
        or metadata.get("block_group_counts") != [3, 3, 3, 3]
        or metadata.get("block_group_label") != "depth_segment"
    ):
        raise ValueError(
            "supplemental topology differs from the flat linear48 contract"
        )

    operand_path = _metadata_asset_path(
        metadata_path, metadata.get("operand_profile"), key="operand_profile"
    )
    propagation_path = _metadata_asset_path(
        metadata_path,
        metadata.get("propagation_profile"),
        key="propagation_profile",
    )
    if metadata.get("operand_profile_sha256") != _sha256_file(operand_path):
        raise ValueError("causal metadata does not match its operand profile")
    if metadata.get("propagation_profile_sha256") != _sha256_file(propagation_path):
        raise ValueError("causal metadata does not match its propagation profile")

    operand_metadata_path = operand_path.with_suffix(operand_path.suffix + ".json")
    propagation_metadata_path = propagation_path.with_suffix(
        propagation_path.suffix + ".json"
    )
    if not operand_metadata_path.is_file() or not propagation_metadata_path.is_file():
        raise FileNotFoundError("supplemental source profile metadata is missing")
    operand_metadata = _read_object(operand_metadata_path)
    propagation_metadata = _read_object(propagation_metadata_path)
    propagation_hashes = propagation_metadata.get("output_hashes", {})
    propagation_assets = propagation_metadata.get("asset_hashes", {})
    if (
        operand_metadata.get("output_sha256") != metadata.get("operand_profile_sha256")
        or operand_metadata.get("propagation_profile_sha256")
        != metadata.get("propagation_profile_sha256")
        or propagation_hashes.get("sha256")
        != metadata.get("propagation_profile_sha256")
        or int(operand_metadata.get("node_count", -1)) != expected_nodes
        or int(propagation_metadata.get("node_count", -1)) != expected_nodes
    ):
        raise ValueError("causal, operand, and propagation profile contracts diverge")

    operand_model_path = _metadata_asset_path(
        operand_metadata_path, operand_metadata.get("model"), key="model"
    )
    operand_registry_path = _metadata_asset_path(
        operand_metadata_path,
        operand_metadata.get("node_registry"),
        key="node_registry",
    )
    propagation_model_path = _metadata_asset_path(
        propagation_metadata_path, propagation_metadata.get("model"), key="model"
    )
    propagation_registry_path = _metadata_asset_path(
        propagation_metadata_path,
        propagation_metadata.get("node_registry"),
        key="node_registry",
    )
    operand_registry_sha256 = _sha256_file(operand_registry_path)
    if (
        operand_metadata.get("model_sha1") != expected_model_sha1
        or _sha1_file(operand_model_path) != expected_model_sha1
        or _sha1_file(operand_registry_path) != expected_registry_sha1
        or operand_metadata.get("node_registry_sha256") != operand_registry_sha256
    ):
        raise ValueError("operand profile model/registry identity differs from package")

    model_hashes = propagation_assets.get("model", {})
    registry_hashes = propagation_assets.get("registry", {})
    if (
        model_hashes.get("sha1") != expected_model_sha1
        or model_hashes.get("sha1") != _sha1_file(propagation_model_path)
        or model_hashes.get("sha256") != _sha256_file(propagation_model_path)
        or registry_hashes.get("sha1") != expected_registry_sha1
        or registry_hashes.get("sha1") != _sha1_file(propagation_registry_path)
        or registry_hashes.get("sha256") != _sha256_file(propagation_registry_path)
        or propagation_model_path != operand_model_path
        or propagation_registry_path != operand_registry_path
    ):
        raise ValueError(
            "propagation profile model/registry identity differs from package"
        )
    return {
        "promotion_source_metadata_sha256": _sha256_file(metadata_path),
        "operand_profile_metadata_sha256": _sha256_file(operand_metadata_path),
        "propagation_profile_metadata_sha256": _sha256_file(propagation_metadata_path),
    }


def _clone_model(model: Any) -> Any:
    # strict50_calibration has already initialized the requested CUDA LightGBM
    # runtime by the time sklearn is imported here.
    from sklearn.base import clone

    return clone(model)


def _atomic_copy(source: Path, destination: Path, *, overwrite: bool) -> None:
    source_hash = _sha256_file(source)
    if destination.exists():
        if _sha256_file(destination) == source_hash:
            return
        if not overwrite:
            raise FileExistsError(
                f"canonical supplemental asset differs; pass --overwrite: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if _sha256_file(temporary) != source_hash:
            raise OSError(f"copied supplemental asset hash differs: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.partial-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selected_frame(package_dir: Path, selected: np.ndarray) -> pd.DataFrame:
    path = package_dir / "data" / "selected_images.csv"
    frame = pd.read_csv(path, low_memory=False)
    if "image_index_in_runtime_bank" not in frame.columns:
        raise ValueError("base package selected image table has no runtime-bank index")
    actual = frame.sort_values("rank")["image_index_in_runtime_bank"].to_numpy(np.int64)
    if not np.array_equal(actual, selected):
        raise ValueError("base package selected image table is not in package order")
    return frame


def _set_training_runtime(model: Any, *, device: str, gpu_id: int, jobs: int) -> Any:
    params = model.get_params(deep=True)
    updates: dict[str, Any] = {}
    for key in params:
        if key == "device_type" or key.endswith("__device_type"):
            updates[key] = str(device)
        elif key == "gpu_device_id" or key.endswith("__gpu_device_id"):
            updates[key] = int(gpu_id)
        elif key == "n_jobs" or key.endswith("__n_jobs"):
            updates[key] = 1 if device == "cuda" else int(jobs)
    if updates:
        model.set_params(**updates)
    return model


def _package_metrics(values: dict[str, float | int]) -> dict[str, Any]:
    return {
        "sample_count": int(values["rows"]),
        "mae": float(values["mae"]),
        "rmse": float(values["rmse"]),
        "bias_pred_minus_actual": float(values["bias"]),
        "p95_abs_error": float(values["p95_abs_error"]),
        "p99_abs_error": float(values["p99_abs_error"]),
        "max_abs_error": float(values["max_abs_error"]),
        "at_or_over_5pp_count": int(values["at_or_over_5pp_count"]),
        "over_5pp_count": int(values["over_5pp_count"]),
        "gate_pass_mean": bool(float(values["mae"]) <= 0.02),
        "gate_pass_max": bool(float(values["max_abs_error"]) <= 0.05),
        "gate_pass_both": bool(
            float(values["mae"]) <= 0.02 and float(values["max_abs_error"]) <= 0.05
        ),
    }


def _load_supplemental(
    path: Path,
    *,
    model_key: str,
    dataset_id: str,
    runtime_identity: dict[str, Any],
) -> tuple[np.memmap, list[str], dict[str, Any]]:
    names_path, metadata_path = _supplemental_sidecars(path)
    metadata = _read_object(metadata_path)
    names = json.loads(names_path.read_text(encoding="ascii"))
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != (EXPECTED_ROWS, FEATURE_COUNT)
        or values.dtype != np.float32
        or metadata.get("rows") != EXPECTED_ROWS
        or metadata.get("feature_count") != FEATURE_COUNT
        or metadata.get("dataset_id") != dataset_id
        or metadata.get("claim_scope")
        != "target_free_all_multiplier_static_physics_features"
        or metadata.get("schema_version")
        != f"{model_key.split('_')[0]}_linear48_causal_prefix_second_order_features_v2"
        or metadata.get("targets_loaded") is not False
        or metadata.get("runtime_bank_loaded") is not False
        or int(metadata.get("candidate_model_inference_count", -1)) != 0
        or int(metadata.get("candidate_swin_inference_count", -1)) != 0
        or metadata.get("output_sha256") != _sha256_file(path)
        or metadata.get("feature_names_sha256") != _sha256_file(names_path)
    ):
        raise ValueError("supplemental feature asset violates its target-free contract")
    if not np.isfinite(values).all():
        raise ValueError("supplemental feature matrix contains non-finite values")
    if not isinstance(names, list) or len(names) != FEATURE_COUNT:
        raise ValueError("supplemental feature names are invalid")
    if len({str(value) for value in names}) != FEATURE_COUNT:
        raise ValueError("supplemental feature names are not unique")
    source_identity = (
        _validate_promoted_supplemental_identity(
            metadata=metadata,
            runtime_identity=runtime_identity,
        )
        if metadata.get("promoted_for_formal_inference") is True
        else _validate_supplemental_source_identity(
            metadata_path=metadata_path,
            metadata=metadata,
            runtime_identity=runtime_identity,
        )
    )
    metadata = {**metadata, **source_identity}
    return values, [str(value) for value in names], metadata


def _load_runtime_inputs(dataset_root: Path, selected: np.ndarray) -> dict[str, Any]:
    data_root = dataset_root / "data"
    configuration = pd.read_csv(
        data_root / "configuration_index.csv",
        usecols=["row_index", "design_id", "runtime_design_id", "split"],
        dtype={"design_id": str, "runtime_design_id": str, "split": str},
        low_memory=False,
    )
    if len(configuration) != EXPECTED_ROWS or not np.array_equal(
        configuration["row_index"].to_numpy(np.int64),
        np.arange(EXPECTED_ROWS, dtype=np.int64),
    ):
        raise ValueError("configuration index is not canonical 20k row order")
    split = configuration["split"].replace({"validation": "val"}).to_numpy(str)
    split_rows = tuple(
        np.flatnonzero(split == name) for name in ("train", "val", "test")
    )
    if tuple(len(rows) for rows in split_rows) != EXPECTED_SPLIT_COUNTS:
        raise ValueError("canonical split counts changed")
    common = np.load(data_root / "common3103.npy", mmap_mode="r", allow_pickle=False)
    if (
        common.shape != (EXPECTED_ROWS, COMMON_FEATURE_COUNT)
        or common.dtype != np.float32
    ):
        raise ValueError("common3103 feature matrix is invalid")
    targets = np.load(data_root / "targets.npy", mmap_mode="r", allow_pickle=False)
    if targets.shape[0] != EXPECTED_ROWS or targets.ndim != 2 or targets.shape[1] < 2:
        raise ValueError("target matrix is invalid")
    with np.load(data_root / "runtime_bank.npz", allow_pickle=False) as runtime:
        rows = np.asarray(runtime["row_index"], dtype=np.int64)
        if not np.array_equal(rows, np.arange(EXPECTED_ROWS, dtype=np.int64)):
            raise ValueError("runtime bank is not in canonical row order")
        observed = {
            name: np.asarray(runtime[name][:, selected], dtype=np.float32)
            for name in (
                "approx_top1_correct_bits",
                "approx_top5_correct_bits",
                "delta_margin_top1",
                "delta_margin_top5",
            )
        }
        runtime_design_ids = np.asarray(runtime["design_id"])
    with np.load(
        dataset_root / "shared" / "strict50_bank_meta.npz", allow_pickle=False
    ) as bank:
        exact_top1 = np.asarray(
            bank["exact_top1_correct_bits"][selected], dtype=np.float32
        )
        exact_top5 = np.asarray(
            bank["exact_top5_correct_bits"][selected], dtype=np.float32
        )
    if not np.array_equal(
        runtime_design_ids.astype(str),
        configuration["runtime_design_id"].astype(str).to_numpy(),
    ):
        raise ValueError("runtime design IDs do not align with configuration")
    return {
        "configuration": configuration,
        "split_rows": split_rows,
        "common": common,
        "targets": targets,
        "observed": observed,
        "exact_top1": exact_top1,
        "exact_top5": exact_top5,
        "runtime_design_ids": runtime_design_ids,
        "runtime_rows": rows,
    }


def _strict_features(
    *,
    common: np.ndarray,
    selected: np.ndarray,
    exact_top1: np.ndarray,
    exact_top5: np.ndarray,
    observed: dict[str, np.ndarray],
    prior_top1: np.ndarray,
    prior_top5: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    features, names, _ = build_50shot_features_from_selected(
        base_x=np.asarray(common, dtype=np.float32),
        image_indices=selected,
        clean_top1_selected=exact_top1,
        clean_top5_selected=exact_top5,
        approx_top1_selected=observed["approx_top1_correct_bits"],
        approx_top5_selected=observed["approx_top5_correct_bits"],
        delta_margin_top1_selected=observed["delta_margin_top1"],
        delta_margin_top5_selected=observed["delta_margin_top5"],
        prior_top1=prior_top1,
        prior_top5=prior_top5,
        include_individual=True,
    )
    if features.shape[1] != BASE_FEATURE_COUNT:
        raise ValueError(f"strict-50 feature count changed: {features.shape[1]}")
    return np.asarray(features, dtype=np.float32), [str(value) for value in names]


def _write_prediction_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def train(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    model_key = str(args.model_key)
    dataset_root = args.dataset_root.expanduser().resolve()
    base_package_dir = args.base_package.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()
    package_dir = output_run / "strict50_package"
    if output_run.exists() and not args.overwrite:
        raise FileExistsError(f"output run exists; pass --overwrite: {output_run}")
    output_run.mkdir(parents=True, exist_ok=True)

    base = load_strict50_top1_package(base_package_dir)
    base_manifest = base.manifest
    dataset_id = str(base_manifest.get("dataset_id", ""))
    if base_manifest.get("model_key") != model_key:
        raise ValueError("base package model key does not match --model-key")
    if int(base.feature_count) != BASE_FEATURE_COUNT:
        raise ValueError(
            "structure-aware enhancement expects a 3,463-feature base package"
        )
    base_names = tuple(
        str(name) for name in base_manifest.get("top1_base_model_names", [])
    )
    if len(base_names) != 1 or base_names[0] not in base.top1_train_models:
        raise ValueError(
            "structure-aware enhancement currently requires one LightGBM base member"
        )
    base_model_name = base_names[0]
    runtime_identity = base_manifest.get("runtime_identity_contract")
    if not isinstance(runtime_identity, dict):
        raise TypeError("base package has no runtime identity contract")
    selected = np.asarray(base.selected_images, dtype=np.int64)
    if selected.shape != (50,):
        raise ValueError("base package must contain exactly 50 selected images")
    selected_frame = _selected_frame(base_package_dir, selected)

    source_features = args.supplemental_features.expanduser().resolve()
    supplemental, supplemental_names, supplemental_metadata = _load_supplemental(
        source_features,
        model_key=model_key,
        dataset_id=dataset_id,
        runtime_identity=runtime_identity,
    )
    # Keep stable, model-specific names in the canonical dataset even when the
    # source cache was produced in the experiment directory.
    canonical_features = (
        dataset_root
        / "data"
        / (
            "vit_linear48_causal_prefix_second_order1034.float32.npy"
            if model_key.startswith("vit_")
            else "deit_linear48_causal_prefix_second_order1034.float32.npy"
        )
    )
    canonical_names = canonical_features.with_suffix(
        canonical_features.suffix + ".features.json"
    )
    canonical_metadata = canonical_features.with_suffix(
        canonical_features.suffix + ".metadata.json"
    )
    _atomic_copy(source_features, canonical_features, overwrite=args.overwrite)
    _atomic_copy(
        source_features.with_suffix(source_features.suffix + ".features.json"),
        canonical_names,
        overwrite=args.overwrite,
    )

    inputs = _load_runtime_inputs(dataset_root, selected)
    configuration = inputs["configuration"]
    train_rows, val_rows, test_rows = inputs["split_rows"]
    common = inputs["common"]
    observed = inputs["observed"]
    targets = inputs["targets"]
    y = np.asarray(targets[:, 0], dtype=np.float64)

    common_float = np.asarray(common, dtype=np.float32)
    prior_train_top1 = np.clip(
        np.asarray(
            predict_lightgbm_model(base.static_prior_top1_train, common_float),
            dtype=np.float64,
        ),
        -0.05,
        1.0,
    )
    prior_train_top5 = np.clip(
        np.asarray(
            predict_lightgbm_model(base.static_prior_top5_train, common_float),
            dtype=np.float64,
        ),
        -0.05,
        1.0,
    )
    prior_refit_top1 = np.clip(
        np.asarray(
            predict_lightgbm_model(base.static_prior_top1_refit, common_float),
            dtype=np.float64,
        ),
        -0.05,
        1.0,
    )
    prior_refit_top5 = np.clip(
        np.asarray(
            predict_lightgbm_model(base.static_prior_top5_refit, common_float),
            dtype=np.float64,
        ),
        -0.05,
        1.0,
    )
    strict_train, strict_names = _strict_features(
        common=common_float,
        selected=selected,
        exact_top1=inputs["exact_top1"],
        exact_top5=inputs["exact_top5"],
        observed=observed,
        prior_top1=prior_train_top1,
        prior_top5=prior_train_top5,
    )
    strict_refit, refit_names = _strict_features(
        common=common_float,
        selected=selected,
        exact_top1=inputs["exact_top1"],
        exact_top5=inputs["exact_top5"],
        observed=observed,
        prior_top1=prior_refit_top1,
        prior_top5=prior_refit_top5,
    )
    if strict_names != refit_names:
        raise RuntimeError("train and refit strict-50 schemas differ")
    full_names = [*strict_names, *supplemental_names]
    if len(full_names) != BASE_FEATURE_COUNT + FEATURE_COUNT or len(
        set(full_names)
    ) != len(full_names):
        raise ValueError("joint feature schema is not exactly 4,497 unique columns")
    joint_train = np.concatenate(
        (strict_train, np.asarray(supplemental, dtype=np.float32)), axis=1
    )
    joint_refit = np.concatenate(
        (strict_refit, np.asarray(supplemental, dtype=np.float32)), axis=1
    )

    train_model = _set_training_runtime(
        _clone_model(base.top1_train_models[base_model_name]),
        device=str(args.lgbm_device),
        gpu_id=int(args.lgbm_gpu_id),
        jobs=int(args.jobs),
    )
    refit_model = _set_training_runtime(
        _clone_model(base.top1_refit_models[base_model_name]),
        device=str(args.lgbm_device),
        gpu_id=int(args.lgbm_gpu_id),
        jobs=int(args.jobs),
    )
    fit_lightgbm_model(train_model, joint_train[train_rows], y[train_rows])
    fit_lightgbm_model(
        refit_model,
        joint_refit[np.concatenate((train_rows, val_rows))],
        y[np.concatenate((train_rows, val_rows))],
    )

    raw_validation = np.asarray(
        predict_lightgbm_model(train_model, joint_train[val_rows]), dtype=np.float64
    )
    calibration = (
        base_manifest.get("validation_calibration")
        if args.calibration == "inherit"
        else None
        if args.calibration == "none"
        else str(args.calibration)
    )
    bias_offset = _validation_calibration_offset(
        None if calibration is None else str(calibration),
        actual=y[val_rows],
        prediction=raw_validation,
    )
    validation_prediction = np.clip(raw_validation + bias_offset, -0.05, 1.0)
    holdout_prediction = np.clip(
        np.asarray(
            predict_lightgbm_model(refit_model, joint_refit[test_rows]),
            dtype=np.float64,
        )
        + bias_offset,
        -0.05,
        1.0,
    )
    validation_metrics = _package_metrics(_metrics(y[val_rows], validation_prediction))
    holdout_metrics = _package_metrics(_metrics(y[test_rows], holdout_prediction))
    conformal_radius = float(
        np.quantile(np.abs(validation_prediction - y[val_rows]), 0.95)
    )

    runtime_ids = inputs["runtime_design_ids"]
    if supplemental_metadata.get("promoted_for_formal_inference") is True and (
        supplemental_metadata.get("runtime_design_ids_sha256")
        != _sha256_array(runtime_ids)
        or supplemental_metadata.get("row_indices_sha256")
        != _sha256_array(inputs["runtime_rows"])
    ):
        raise ValueError("promoted supplemental rows differ from the runtime bank")
    promoted_metadata = {
        **supplemental_metadata,
        "output": str(canonical_features),
        "output_sha256": _sha256_file(canonical_features),
        "feature_names": str(canonical_names),
        "feature_names_sha256": _sha256_file(canonical_names),
        "runtime_design_ids_sha256": _sha256_array(runtime_ids),
        "row_indices_sha256": _sha256_array(inputs["runtime_rows"]),
        "promotion_source": supplemental_metadata.get(
            "promotion_source", str(source_features)
        ),
        "promoted_for_formal_inference": True,
        "model_key": model_key,
        "model_sha1": runtime_identity.get("model_sha1"),
        "node_registry_sha1": runtime_identity.get("node_registry_sha1"),
        "injected_matmul_count": runtime_identity.get("injected_matmul_count"),
    }
    _atomic_json(canonical_metadata, promoted_metadata)
    required_keys = (
        "schema_version",
        "claim_scope",
        "dataset_id",
        "injection_scope",
        "feature_count",
        "node_count",
        "block_count",
        "operations_per_block",
        "operation_names",
        "profile_image_count",
        "operand_profile_sha256",
        "propagation_profile_sha256",
        "promotion_source_metadata_sha256",
        "operand_profile_metadata_sha256",
        "propagation_profile_metadata_sha256",
        "model_key",
        "model_sha1",
        "node_registry_sha1",
        "injected_matmul_count",
        "targets_loaded",
        "runtime_bank_loaded",
        "candidate_model_inference_count",
        "candidate_swin_inference_count",
        "block_group_counts",
        "block_group_label",
    )
    missing_metadata = [key for key in required_keys if key not in promoted_metadata]
    if missing_metadata:
        raise ValueError(
            f"supplemental metadata is missing package contract keys: {missing_metadata}"
        )
    supplemental_contract = {
        "name": f"{model_key.split('_')[0]}_linear48_causal_prefix_second_order",
        "feature_count": FEATURE_COUNT,
        "matrix_manifest_key": "supplemental_features",
        "names_manifest_key": "supplemental_feature_names",
        "metadata_manifest_key": "supplemental_features_metadata",
        "required_metadata": {key: promoted_metadata[key] for key in required_keys},
        "row_alignment": {
            "runtime_design_id_archive_key": "design_id",
            "metadata_design_ids_sha256_key": "runtime_design_ids_sha256",
            "runtime_row_index_archive_key": "row_index",
            "metadata_row_indices_sha256_key": "row_indices_sha256",
        },
        "required": True,
        "target_free": True,
    }
    package_id = (
        f"{model_key}_int8_s8s8_linear48_structure_aware_20k_50shot_v1_strict50_top1"
    )
    manifest: dict[str, Any] = {
        "schema_version": "transformer_strict50_top1_package_v1",
        "package_id": package_id,
        "package_status": "sealed_structure_aware_training_output",
        "source_package_id": base.package_id,
        "source_package_hash": base_manifest.get("package_hash"),
        "model_key": model_key,
        "dataset_id": dataset_id,
        "injection_scope": runtime_identity.get("injection_scope_id"),
        "injected_matmul_count": runtime_identity.get("injected_matmul_count"),
        "model_sha1": runtime_identity.get("model_sha1"),
        "node_registry_sha1": runtime_identity.get("node_registry_sha1"),
        "runtime_identity_contract": runtime_identity,
        "target": "top1_drop",
        "target_definition": "exact_top1_accuracy - approximate_top1_accuracy",
        "target_units": "proportion",
        "learned_model_family": "lightgbm_only",
        "lightgbm_training": {
            "device_type": str(args.lgbm_device),
            "gpu_device_id": int(args.lgbm_gpu_id)
            if args.lgbm_device == "cuda"
            else None,
            "n_jobs": 1 if args.lgbm_device == "cuda" else int(args.jobs),
            "requested_n_jobs": int(args.jobs),
            "base_model_name": base_model_name,
        },
        "common3103_feature_count": COMMON_FEATURE_COUNT,
        "runtime_base_feature_count": BASE_FEATURE_COUNT,
        "feature_count": len(full_names),
        "feature_segments": [
            {
                "name": "strict50_runtime_observations",
                "start": 0,
                "count": BASE_FEATURE_COUNT,
            },
            {
                "name": supplemental_contract["name"],
                "start": BASE_FEATURE_COUNT,
                "count": FEATURE_COUNT,
            },
        ],
        "runtime_supplemental_feature_contract": supplemental_contract,
        "inference_mode": "strict_50shot_plus_required_causal_prefix1034",
        "prediction_time_target_image_count": 50,
        "selected_selector": base_manifest.get("selected_selector"),
        "selected_top1_model": (
            f"{base_model_name}__cal_{calibration}" if calibration else base_model_name
        ),
        "top1_model_name": base_model_name,
        "top1_base_model_names": [base_model_name],
        "top1_model_target_bases": {base_model_name: "direct"},
        "top1_model_weights": {base_model_name: 1.0},
        "top1_aggregation_mode": "weighted_mean",
        "validation_calibration": calibration,
        "bias_offset": float(bias_offset),
        "clip_min": -0.05,
        "clip_max": 1.0,
        "conformal_radius_95": conformal_radius,
        "training_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "holdout_rows": len(test_rows),
        "test_splits": ["test"],
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "final_evaluation_status": "scored_once",
        "structure_aware_protocol": {
            "method_frozen_before_holdout_scoring": True,
            "base_model_hyperparameters_reused": True,
            "base_selected_images_reused": True,
            "supplemental_features_target_free": True,
            "calibration_policy": str(args.calibration),
            "deployment_models_published": False,
        },
    }
    validation_frame = pd.DataFrame(
        {
            "row_index": val_rows,
            "design_id": configuration.iloc[val_rows]["design_id"]
            .astype(str)
            .to_numpy(),
            "split": "validation",
            "actual": y[val_rows],
            "pred": validation_prediction,
            "abs_error": np.abs(validation_prediction - y[val_rows]),
        }
    )
    holdout_frame = pd.DataFrame(
        {
            "row_index": test_rows,
            "design_id": configuration.iloc[test_rows]["design_id"]
            .astype(str)
            .to_numpy(),
            "split": "test",
            "actual": y[test_rows],
            "pred": holdout_prediction,
            "abs_error": np.abs(holdout_prediction - y[test_rows]),
        }
    )
    save_strict50_top1_package(
        package_dir=package_dir,
        manifest=manifest,
        selected_images=selected_frame,
        feature_names=full_names,
        static_prior_top1_train=base.static_prior_top1_train,
        static_prior_top1_refit=base.static_prior_top1_refit,
        static_prior_top5_train=base.static_prior_top5_train,
        static_prior_top5_refit=base.static_prior_top5_refit,
        top1_train_models={base_model_name: train_model},
        top1_refit_models={base_model_name: refit_model},
        bias_offset=float(bias_offset),
        validation_predictions=validation_frame,
        holdout_predictions=holdout_frame,
        holdout_metrics=holdout_metrics,
    )
    integrity = verify_strict50_top1_package_integrity(package_dir)
    published = load_strict50_top1_package(package_dir)

    inference_dir = output_run / "inference"
    prediction_manifest_path = inference_dir / "prediction_manifest.json"
    prediction_manifest = {
        "schema_version": "transformer_strict50_prediction_manifest_v1",
        "common_features": str(dataset_root / "data" / "common3103.npy"),
        "runtime_observations": str(dataset_root / "data" / "runtime_bank.npz"),
        "bank_meta": str(dataset_root / "shared" / "strict50_bank_meta.npz"),
        "supplemental_features": str(canonical_features),
        "supplemental_feature_names": str(canonical_names),
        "supplemental_features_metadata": str(canonical_metadata),
        "runtime_identity": runtime_identity,
    }
    _atomic_json(prediction_manifest_path, prediction_manifest)
    run_config = {
        "schema_version": "structure_aware_linear48_training_run_v1",
        "model_key": model_key,
        "dataset_id": dataset_id,
        "base_package": str(base_package_dir),
        "base_package_hash": base_manifest.get("package_hash"),
        "output_run": str(output_run),
        "package_dir": str(package_dir),
        "prediction_manifest": str(prediction_manifest_path),
        "feature_count": len(full_names),
        "base_feature_count": BASE_FEATURE_COUNT,
        "supplemental_feature_count": FEATURE_COUNT,
        "selected_image_count": 50,
        "selected_selector": base_manifest.get("selected_selector"),
        "calibration_policy": str(args.calibration),
        "lgbm_device": str(args.lgbm_device),
        "lgbm_gpu_id": int(args.lgbm_gpu_id),
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "package_hash": published.manifest["package_hash"],
        "content_hash": integrity["content_hash"],
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(output_run / "run_config.json", run_config)
    _write_prediction_csv(
        output_run / "data" / "validation_predictions.csv", validation_frame
    )
    _write_prediction_csv(
        output_run / "data" / "holdout_predictions.csv", holdout_frame
    )
    _atomic_json(output_run / "data" / "holdout_metrics.json", holdout_metrics)
    report = "\n".join(
        [
            "# Structure-aware linear48 strict-50 package",
            "",
            f"Model: `{model_key}`",
            f"Features: `{len(full_names)}` (`{BASE_FEATURE_COUNT}` strict-50 + `{FEATURE_COUNT}` target-free physics)",
            f"Base package: `{base.package_id}`",
            "",
            "| split | MAE | P95 | Max | >5 pp |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| validation | {100 * validation_metrics['mae']:.4f} pp | {100 * validation_metrics['p95_abs_error']:.4f} pp | {100 * validation_metrics['max_abs_error']:.4f} pp | {validation_metrics['over_5pp_count']} |",
            f"| holdout | {100 * holdout_metrics['mae']:.4f} pp | {100 * holdout_metrics['p95_abs_error']:.4f} pp | {100 * holdout_metrics['max_abs_error']:.4f} pp | {holdout_metrics['over_5pp_count']} |",
            "",
            "The original base package is retained; this package has no deployment model and therefore defaults to refit at prediction time.",
            "",
        ]
    )
    (output_run / "REPORT.md").write_text(report, encoding="utf-8")
    return run_config


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", choices=sorted(DATASET_ROOTS), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--base-package", type=Path)
    parser.add_argument("--supplemental-features", type=Path)
    parser.add_argument("--output-run", type=Path)
    parser.add_argument("--lgbm-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lgbm-gpu-id", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument(
        "--calibration",
        choices=("none", "inherit", "bias", "gate_shrink", "center"),
        default="none",
        help="Validation-only offset policy; zero offset matches the deployed Swin joint4497 method.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = parser().parse_args()
    if args.jobs < 1 or args.lgbm_gpu_id < 0:
        raise SystemExit("--jobs and --lgbm-gpu-id must be non-negative/positive")
    if args.dataset_root is None:
        args.dataset_root = DATASET_ROOTS[args.model_key]
    if args.base_package is None:
        args.base_package = BASE_PACKAGES[args.model_key]
    if args.supplemental_features is None:
        args.supplemental_features = DEFAULT_FEATURES[args.model_key]
    if args.output_run is None:
        args.output_run = DEFAULT_OUTPUT_RUNS[args.model_key]
    result = train(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
