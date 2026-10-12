#!/usr/bin/env python
"""Historically promote the frozen Swin linear48 joint-4497 model once.

This command does not retrain or reopen the original label bank. It verifies
the frozen experiment artifacts, publishes the target-free causal-prefix block
under the dataset, and packages the already-scored joint LightGBM models.

This is a provenance-preserving migration utility, not a daily training entry
point. Its source run and feature defaults intentionally remain under
``experiment/``; current training and inference consume the promoted canonical
dataset assets instead.
"""

from __future__ import annotations

from aaae.serialization import load_joblib

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.transformer.artifacts.strict50_package import (
    load_strict50_top1_package,
    save_strict50_top1_package,
    verify_strict50_top1_package_integrity,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
SOURCE_RUN = (
    PROJECT_ROOT
    / "experiment"
    / "runs"
    / "swin_linear48_causal_prefix_transfer_20260801"
)
SOURCE_FEATURES = (
    PROJECT_ROOT
    / "experiment"
    / "cache"
    / "linear48_a100_causal_prefix_second_order.float32.npy"
)
BASE_PACKAGE = (
    PROJECT_ROOT
    / "data"
    / "runs"
    / "swin_tiny_linear48_e2e_trial_20260801"
    / "strict50_package"
)
OUTPUT_RUN = (
    OUTPUT_ROOT / "accuracy" / "swin_tiny_linear48_e2e_joint4497_20260801"
)
PROMOTED_FEATURES = OUTPUT_RUN / "features" / "causal_prefix_second_order1034.float32.npy"
MODEL_NAME = "lgbm_l2_c50_n300_joint4497"


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


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def _copy_verified(source: Path, destination: Path, *, overwrite: bool) -> None:
    source_hash = _sha256_file(source)
    if destination.exists():
        if _sha256_file(destination) == source_hash:
            return
        if not overwrite:
            raise FileExistsError(
                f"promoted feature asset differs; pass --overwrite: {destination}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.partial-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if _sha256_file(temporary) != source_hash:
            raise OSError(f"copied feature asset hash differs: {source}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _package_metrics(values: dict[str, Any]) -> dict[str, Any]:
    mae = float(values["mae"])
    maximum = float(values["max_abs_error"])
    return {
        "sample_count": int(values["rows"]),
        "mae": mae,
        "rmse": float(values["rmse"]),
        "bias_pred_minus_actual": float(values["bias"]),
        "p95_abs_error": float(values["p95_abs_error"]),
        "p99_abs_error": float(values["p99_abs_error"]),
        "max_abs_error": maximum,
        "at_or_over_5pp_count": int(values["at_or_over_5pp_count"]),
        "over_5pp_count": int(values["over_5pp_count"]),
        "gate_pass_mean": bool(mae <= 0.02),
        "gate_pass_max": bool(maximum <= 0.05),
        "gate_pass_both": bool(mae <= 0.02 and maximum <= 0.05),
    }


def _prediction_frame(
    *,
    rows: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    configuration: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    rows64 = np.asarray(rows, dtype=np.int64)
    actual64 = np.asarray(target, dtype=np.float64)
    predicted64 = np.asarray(prediction, dtype=np.float64)
    if actual64.shape != predicted64.shape or actual64.shape != rows64.shape:
        raise ValueError(f"{split} promotion predictions are not row aligned")
    return pd.DataFrame(
        {
            "row_index": rows64,
            "design_id": configuration.iloc[rows64]["design_id"].astype(str).to_numpy(),
            "split": np.full(len(rows64), split),
            "actual": actual64,
            "pred": predicted64,
            "abs_error": np.abs(predicted64 - actual64),
        }
    )


def promote(args: argparse.Namespace) -> dict[str, Any]:
    base_package_dir = args.base_package.expanduser().resolve()
    source_run = args.source_run.expanduser().resolve()
    source_features = args.source_features.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_run = args.output_run.expanduser().resolve()
    package_dir = output_run / "strict50_package"
    inference_dir = output_run / "inference"
    prediction_manifest_path = inference_dir / "prediction_manifest.json"
    promoted_features = args.promoted_features.expanduser().resolve()
    promoted_names = promoted_features.with_suffix(
        promoted_features.suffix + ".features.json"
    )
    promoted_metadata = promoted_features.with_suffix(
        promoted_features.suffix + ".metadata.json"
    )
    if package_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"joint4497 package exists; pass --overwrite: {package_dir}"
        )

    result_path = source_run / "result.json"
    models_path = source_run / "models.joblib"
    predictions_path = source_run / "predictions.npz"
    source_names = source_features.with_suffix(
        source_features.suffix + ".features.json"
    )
    source_metadata_path = source_features.with_suffix(source_features.suffix + ".json")
    for path in (
        result_path,
        models_path,
        predictions_path,
        source_features,
        source_names,
        source_metadata_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    result = _read_object(result_path)
    source_metadata = _read_object(source_metadata_path)
    if result.get("schema_version") != "swin_linear48_causal_prefix_transfer_v1":
        raise ValueError("source joint experiment schema is invalid")
    if result.get("protocol", {}).get("test_evaluation_count") != 1:
        raise ValueError("source joint experiment does not have exactly one test score")
    if result["artifacts"].get("models_sha256") != _sha256_file(models_path):
        raise ValueError("source joint model hash differs from result.json")
    if result["artifacts"].get("predictions_sha256") != _sha256_file(predictions_path):
        raise ValueError("source joint prediction hash differs from result.json")
    if source_metadata.get("output_sha256") != _sha256_file(source_features):
        raise ValueError("source causal-prefix feature hash differs from its metadata")
    if source_metadata.get("feature_names_sha256") != _sha256_file(source_names):
        raise ValueError("source causal-prefix names hash differs from its metadata")

    base = load_strict50_top1_package(base_package_dir)
    if base.manifest.get("package_hash") != result["method"].get("base_package_hash"):
        raise ValueError("source experiment and base package hashes differ")
    frozen_models = load_joblib(models_path)
    if frozen_models.get("schema_version") != "swin_linear48_causal_prefix_models_v1":
        raise ValueError("source joint model bundle schema is invalid")
    strict_names = [str(value) for value in frozen_models["strict50_feature_names"]]
    physics_names = [str(value) for value in frozen_models["physics_feature_names"]]
    if strict_names != base.feature_names:
        raise ValueError(
            "source joint model strict-50 names differ from the base package"
        )
    persisted_physics_names = json.loads(source_names.read_text(encoding="utf-8"))
    if physics_names != persisted_physics_names:
        raise ValueError(
            "source joint model physics names differ from the feature artifact"
        )
    full_names = [*strict_names, *physics_names]
    if len(strict_names) != 3463 or len(physics_names) != 1034:
        raise ValueError(
            "source joint model is not the required 3463 + 1034 feature model"
        )
    if len(set(full_names)) != 4497:
        raise ValueError("source joint model feature names are not unique")
    for key in ("physical_train_model", "physical_refit_model"):
        if int(getattr(frozen_models[key], "n_features_in_", -1)) != len(full_names):
            raise ValueError(f"{key} is not a 4,497-feature model")

    configuration = pd.read_csv(
        dataset_root / "data" / "configuration_index.csv",
        usecols=["row_index", "design_id", "runtime_design_id", "split"],
        dtype={"design_id": str, "runtime_design_id": str, "split": str},
        low_memory=False,
    )
    expected_rows = np.arange(len(configuration), dtype=np.int64)
    if len(configuration) != 20000 or not np.array_equal(
        configuration["row_index"].to_numpy(np.int64), expected_rows
    ):
        raise ValueError("dataset configuration index is not canonical 20k row order")
    runtime_path = dataset_root / "data" / "runtime_bank.npz"
    with np.load(runtime_path, allow_pickle=False) as runtime:
        runtime_design_ids = np.asarray(runtime["design_id"])
        runtime_rows = np.asarray(runtime["row_index"], dtype=np.int64)
    if not np.array_equal(runtime_rows, expected_rows) or not np.array_equal(
        runtime_design_ids.astype(str),
        configuration["runtime_design_id"].astype(str).to_numpy(),
    ):
        raise ValueError("runtime observations and configuration rows are not aligned")

    source_matrix = np.load(source_features, mmap_mode="r", allow_pickle=False)
    if source_matrix.shape != (len(configuration), len(physics_names)):
        raise ValueError("source causal-prefix matrix shape is invalid")
    del source_matrix
    _copy_verified(source_features, promoted_features, overwrite=args.overwrite)
    _copy_verified(source_names, promoted_names, overwrite=args.overwrite)
    promoted_feature_hash = _sha256_file(promoted_features)
    promoted_names_hash = _sha256_file(promoted_names)
    promoted_metadata_payload = {
        **source_metadata,
        "output": str(promoted_features),
        "output_sha256": promoted_feature_hash,
        "feature_names": str(promoted_names),
        "feature_names_sha256": promoted_names_hash,
        "runtime_design_ids_sha256": _sha256_array(runtime_design_ids),
        "row_indices_sha256": _sha256_array(runtime_rows),
        "promotion_source": str(source_features),
        "promotion_source_metadata_sha256": _sha256_file(source_metadata_path),
        "promoted_for_formal_inference": True,
    }
    _write_json_atomic(promoted_metadata, promoted_metadata_payload)

    required_metadata_keys = (
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
    )
    required_metadata = {
        key: promoted_metadata_payload[key] for key in required_metadata_keys
    }
    supplemental_contract = {
        "name": "swin_linear48_causal_prefix_second_order",
        "feature_count": len(physics_names),
        "matrix_manifest_key": "supplemental_features",
        "names_manifest_key": "supplemental_feature_names",
        "metadata_manifest_key": "supplemental_features_metadata",
        "required_metadata": required_metadata,
        "row_alignment": {
            "runtime_design_id_archive_key": "design_id",
            "metadata_design_ids_sha256_key": "runtime_design_ids_sha256",
            "runtime_row_index_archive_key": "row_index",
            "metadata_row_indices_sha256_key": "row_indices_sha256",
        },
        "required": True,
        "target_free": True,
    }

    with np.load(predictions_path, allow_pickle=False) as predictions:
        validation_frame = _prediction_frame(
            rows=predictions["validation_rows"],
            target=predictions["validation_target"],
            prediction=predictions["physics_validation"],
            configuration=configuration,
            split="validation",
        )
        holdout_frame = _prediction_frame(
            rows=predictions["test_rows"],
            target=predictions["test_target"],
            prediction=predictions["physics_test"],
            configuration=configuration,
            split="test",
        )
    validation_metrics = _package_metrics(result["validation"]["metrics"]["physics"])
    holdout_metrics = _package_metrics(result["test"]["metrics"]["physics"])
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        "schema_version": "transformer_strict50_top1_package_v1",
        "package_id": "swin_tiny_linear48_e2e_joint4497_20260801_strict50_top1",
        "package_status": "sealed_structure_aware_promotion",
        "created_at": created_at,
        "source_run_id": source_run.name,
        "source_package_id": base.package_id,
        "source_package_hash": base.manifest["package_hash"],
        "dataset_id": source_metadata["dataset_id"],
        "model_key": base.manifest["model_key"],
        "injection_scope": source_metadata["injection_scope"],
        "target": "top1_drop",
        "target_definition": "exact_top1_accuracy - approximate_top1_accuracy",
        "target_units": "proportion",
        "learned_model_family": "lightgbm_only",
        "selected_selector": base.manifest["selected_selector"],
        "selected_top1_model": MODEL_NAME,
        "top1_model_name": MODEL_NAME,
        "top1_base_model_names": [MODEL_NAME],
        "top1_model_target_bases": {MODEL_NAME: "direct"},
        "top1_model_weights": {MODEL_NAME: 1.0},
        "top1_aggregation_mode": "weighted_mean",
        "common3103_feature_count": base.common_feature_count,
        "ignored_common_feature_indices": list(base.ignored_common_feature_indices),
        "runtime_base_feature_count": len(strict_names),
        "runtime_supplemental_feature_contract": supplemental_contract,
        "feature_segments": [
            {
                "name": "strict50_runtime_observations",
                "start": 0,
                "count": len(strict_names),
            },
            {
                "name": supplemental_contract["name"],
                "start": len(strict_names),
                "count": len(physics_names),
            },
        ],
        "inference_mode": "strict_50shot_plus_required_causal_prefix1034",
        "prediction_time_target_image_count": 50,
        "training_rows": 14000,
        "validation_rows": 3000,
        "holdout_rows": 3000,
        "test_splits": ["test"],
        "conformal_radius_95": validation_metrics["p95_abs_error"],
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "gate": {"mean_abs_error": 0.02, "max_abs_error": 0.05},
        "final_evaluation_status": "scored_once",
        "promotion_protocol": {
            "source_test_evaluation_count": 1,
            "promotion_decision_after_test_metrics_observed": True,
            "test_is_unbiased_promotion_holdout": False,
            "requires_fresh_holdout_for_unbiased_final_estimate": True,
            "models_retrained_during_promotion": False,
            "source_prediction_artifact_read_during_promotion": True,
            "original_label_bank_reopened_during_promotion": False,
        },
        "source_artifacts": {
            "result": str(result_path),
            "result_sha256": _sha256_file(result_path),
            "models": str(models_path),
            "models_sha256": _sha256_file(models_path),
            "predictions": str(predictions_path),
            "predictions_sha256": _sha256_file(predictions_path),
            "causal_prefix_features_sha256": promoted_feature_hash,
            "causal_prefix_feature_names_sha256": promoted_names_hash,
        },
    }
    selected_frame = pd.read_csv(
        base_package_dir / "data" / "selected_images.csv", low_memory=False
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
        top1_train_models={MODEL_NAME: frozen_models["physical_train_model"]},
        top1_refit_models={MODEL_NAME: frozen_models["physical_refit_model"]},
        static_prior_top1_deployment=base.static_prior_top1_refit,
        static_prior_top5_deployment=base.static_prior_top5_refit,
        top1_deployment_models={MODEL_NAME: frozen_models["physical_refit_model"]},
        bias_offset=0.0,
        validation_predictions=validation_frame,
        holdout_predictions=holdout_frame,
        holdout_metrics=holdout_metrics,
    )
    integrity = verify_strict50_top1_package_integrity(package_dir)
    published = load_strict50_top1_package(package_dir)

    prediction_manifest = {
        "schema_version": "transformer_strict50_prediction_manifest_v1",
        "common_features": str(dataset_root / "data" / "common3103.npy"),
        "runtime_observations": str(runtime_path),
        "bank_meta": str(dataset_root / "shared" / "strict50_bank_meta.npz"),
        "supplemental_features": str(promoted_features),
        "supplemental_feature_names": str(promoted_names),
        "supplemental_features_metadata": str(promoted_metadata),
    }
    _write_json_atomic(prediction_manifest_path, prediction_manifest)
    return {
        "schema_version": "swin_linear48_joint4497_promotion_v1",
        "package_dir": str(package_dir),
        "package_id": published.package_id,
        "package_hash": published.manifest["package_hash"],
        "content_hash": integrity["content_hash"],
        "feature_count": published.feature_count,
        "runtime_base_feature_count": published.runtime_base_feature_count,
        "supplemental_feature_count": published.supplemental_feature_count,
        "deployment_models_available": published.has_deployment_models,
        "prediction_manifest": str(prediction_manifest_path),
        "promoted_features": str(promoted_features),
        "promoted_features_sha256": promoted_feature_hash,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "promotion_decision_after_test_metrics_observed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package", type=Path, default=BASE_PACKAGE)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--source-features", type=Path, default=SOURCE_FEATURES)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-run", type=Path, default=OUTPUT_RUN)
    parser.add_argument("--promoted-features", type=Path, default=PROMOTED_FEATURES)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = promote(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
