#!/usr/bin/env python
"""Transfer the frozen Swin-72 causal-prefix method to linear48.

The selected strict-50 images, LightGBM member, 5% physical blend, and q75
disagreement-gate recipe are unchanged.  Five-fold behavior-group OOF fixes
the numeric disagreement threshold.  Validation is evaluated before a frozen
method contract is written; test targets are indexed only after that freeze.
"""

from __future__ import annotations

from aaae.serialization import load_joblib

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, OUTPUT_ROOT

EXPERIMENT_ROOT = OUTPUT_ROOT / "accuracy" / "swin"

from aaae.accuracy_predictor.transformer.training.swin72_physics_core import joint_nonregression
from aaae.accuracy_predictor.transformer.training.swin_static_core import metrics, sha256_array, sha256_file, write_json
from aaae.accuracy_predictor.features.transformer.prepare_swin_linear48_causal_prefix_features import (
    DEFAULT_OUTPUT as DEFAULT_PHYSICS_FEATURES,
)
from aaae.accuracy_predictor.features.transformer.prepare_swin_linear48_causal_prefix_features import (
    SCHEMA_VERSION as PHYSICS_SCHEMA_VERSION,
)
from aaae.accuracy_predictor.transformer.proxy_training.strict50_calibration import (
    build_50shot_features_from_selected,
    fit_lightgbm_model,
    predict_lightgbm_model,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
DATA_ROOT = DATASET_ROOT / "data"
DEFAULT_PACKAGE = (
    PROJECT_ROOT
    / "data"
    / "runs"
    / "swin_tiny_linear48_e2e_trial_20260801"
    / "strict50_package"
)
DEFAULT_OUTPUT_DIR = (
    EXPERIMENT_ROOT / "runs" / "swin_linear48_causal_prefix_transfer_20260801"
)
DEFAULT_BEHAVIOR_HASHES = PROJECT_ROOT / "experiment" / "cache" / "behavior_sha256.npy"
SCHEMA_VERSION = "swin_linear48_causal_prefix_transfer_v1"
FROZEN_PHYSICS_WEIGHT = 0.05
DISAGREEMENT_QUANTILE = 0.75
EXPECTED_DATASET_ID = "swin_tiny_s8s8_linear48_strict50_true1000_raw500_v1"
EXPECTED_BASE_MODEL = "lgbm_l2_c50_n300"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return value


def _selected_images(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ranked = sorted(
        (
            int(row["rank"]),
            int(row["image_index_in_runtime_bank"]),
        )
        for row in rows
        if str(row["selector"]) == "greedy_drop50"
    )
    if [rank for rank, _image in ranked] != list(range(50)):
        raise ValueError("package does not contain ranked greedy_drop50 observations")
    selected = np.asarray([image for _rank, image in ranked], dtype=np.int64)
    if len(np.unique(selected)) != 50:
        raise ValueError("package selected-image contract contains duplicates")
    return selected


def _predict(model: Any, features: np.ndarray, offset: float = 0.0) -> np.ndarray:
    return np.clip(
        np.asarray(predict_lightgbm_model(model, features), dtype=np.float64)
        + float(offset),
        -0.05,
        1.0,
    )


def _clone_with_jobs(model: Any, jobs: int) -> Any:
    copied = clone(model)
    job_keys = [key for key in copied.get_params(deep=True) if key.endswith("__n_jobs")]
    if len(job_keys) != 1:
        raise ValueError(
            "LightGBM pipeline does not expose exactly one n_jobs parameter"
        )
    copied.set_params(**{job_keys[0]: int(jobs)})
    return copied


def _gated_blend(
    baseline: np.ndarray,
    physical: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    baseline64 = np.asarray(baseline, dtype=np.float64)
    physical64 = np.asarray(physical, dtype=np.float64)
    if baseline64.shape != physical64.shape or threshold <= 0.0:
        raise ValueError("gated blend inputs are invalid")
    active = np.abs(physical64 - baseline64) <= float(threshold)
    prediction = baseline64 + FROZEN_PHYSICS_WEIGHT * active * (physical64 - baseline64)
    return prediction, active


def _load_physics(
    path: Path, row_count: int
) -> tuple[np.memmap, list[str], dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    names_path = path.with_suffix(path.suffix + ".features.json")
    metadata = _read_object(metadata_path)
    names = json.loads(names_path.read_text(encoding="ascii"))
    if (
        metadata.get("schema_version") != PHYSICS_SCHEMA_VERSION
        or metadata.get("claim_scope")
        != "target_free_all_multiplier_static_physics_features"
        or metadata.get("dataset_id") != EXPECTED_DATASET_ID
        or metadata.get("targets_loaded") is not False
        or metadata.get("runtime_bank_loaded") is not False
        or int(metadata.get("candidate_swin_inference_count", -1)) != 0
        or metadata.get("output_sha256") != sha256_file(path)
        or metadata.get("feature_names_sha256") != sha256_file(names_path)
    ):
        raise ValueError("linear48 physics cache violates its target-free contract")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (row_count, int(metadata["feature_count"])):
        raise ValueError("linear48 physics cache shape differs from its sidecar")
    if not isinstance(names, list) or len(names) != values.shape[1]:
        raise ValueError("linear48 physics feature names are invalid")
    return values, [str(value) for value in names], metadata


def _build_strict50_features(
    *,
    common: np.ndarray,
    selected: np.ndarray,
    exact_top1: np.ndarray,
    exact_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    delta_margin_top1: np.ndarray,
    delta_margin_top5: np.ndarray,
    prior_top1: np.ndarray,
    prior_top5: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    features, names, _summary = build_50shot_features_from_selected(
        base_x=np.asarray(common, dtype=np.float32),
        image_indices=selected,
        clean_top1_selected=exact_top1,
        clean_top5_selected=exact_top5,
        approx_top1_selected=approx_top1,
        approx_top5_selected=approx_top5,
        delta_margin_top1_selected=delta_margin_top1,
        delta_margin_top5_selected=delta_margin_top5,
        prior_top1=prior_top1,
        prior_top5=prior_top5,
        include_individual=True,
    )
    return np.asarray(features, dtype=np.float32), [str(value) for value in names]


def _load_package_models(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _read_object(package_dir / "manifest.json")
    if (
        manifest.get("schema_version") != "transformer_strict50_top1_package_v1"
        or manifest.get("dataset_id") != EXPECTED_DATASET_ID
        or manifest.get("selected_selector") != "greedy_drop50"
        or manifest.get("selected_top1_model") != EXPECTED_BASE_MODEL
        or manifest.get("top1_base_model_names") != [EXPECTED_BASE_MODEL]
        or manifest.get("top1_aggregation_mode") != "weighted_mean"
        or manifest.get("validation_calibration") is not None
        or int(manifest.get("feature_count", -1)) != 3463
    ):
        raise ValueError(
            "baseline package is not the fixed single-member linear48 trial"
        )
    layout = manifest["model_layout"]
    models = {
        "static_top1_train": load_joblib(
            package_dir / layout["static_prior_top1_train"]
        ),
        "static_top1_refit": load_joblib(
            package_dir / layout["static_prior_top1_refit"]
        ),
        "static_top5_train": load_joblib(
            package_dir / layout["static_prior_top5_train"]
        ),
        "static_top5_refit": load_joblib(
            package_dir / layout["static_prior_top5_refit"]
        ),
        "top1_train": load_joblib(
            package_dir / layout["top1_train_models"][EXPECTED_BASE_MODEL]
        ),
        "top1_refit": load_joblib(
            package_dir / layout["top1_refit_models"][EXPECTED_BASE_MODEL]
        ),
    }
    return manifest, models


def _assert_package_prediction_match(
    path: Path,
    rows: np.ndarray,
    prediction: np.ndarray,
) -> float:
    recorded = pd.read_csv(path, usecols=["row_index", "pred"], low_memory=False)
    if not np.array_equal(recorded["row_index"].to_numpy(np.int64), rows):
        raise ValueError(f"package prediction rows changed: {path}")
    difference = float(
        np.max(
            np.abs(
                recorded["pred"].to_numpy(np.float64)
                - np.asarray(prediction, dtype=np.float64)
            )
        )
    )
    if difference > 1.0e-12:
        raise ValueError(f"reconstructed package predictions differ by {difference}")
    return difference


def _write_predictions(
    path: Path,
    **arrays: np.ndarray,
) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _format_metric_row(name: str, values: dict[str, Any]) -> str:
    return (
        f"| {name} | {100.0 * float(values['mae']):.4f} | "
        f"{100.0 * float(values['p95_abs_error']):.4f} | "
        f"{100.0 * float(values['p99_abs_error']):.4f} | "
        f"{100.0 * float(values['max_abs_error']):.4f} | "
        f"{int(values['over_5pp_count'])} |"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    frozen_path = output_dir / "FROZEN_METHOD.json"
    predictions_path = output_dir / "predictions.npz"
    models_path = output_dir / "models.joblib"
    report_path = output_dir / "REPORT.md"
    if (
        any(
            path.exists()
            for path in (
                result_path,
                frozen_path,
                predictions_path,
                models_path,
                report_path,
            )
        )
        and not args.overwrite
    ):
        raise FileExistsError("linear48 transfer experiment exists; pass --overwrite")

    package_dir = args.package_dir.expanduser().resolve()
    manifest, package_models = _load_package_models(package_dir)
    selected = _selected_images(package_dir / "data" / "selected_images.csv")
    configuration = pd.read_csv(
        DATA_ROOT / "configuration_index.csv",
        usecols=["row_index", "design_id", "runtime_design_id", "split"],
        dtype={"design_id": str, "runtime_design_id": str, "split": str},
        low_memory=False,
    )
    row_count = len(configuration)
    if row_count != 20000 or not np.array_equal(
        configuration["row_index"].to_numpy(np.int64),
        np.arange(row_count, dtype=np.int64),
    ):
        raise ValueError("linear48 configuration rows are not canonical")
    split = configuration["split"].replace({"validation": "val"}).to_numpy(str)
    train_rows = np.flatnonzero(split == "train")
    validation_rows = np.flatnonzero(split == "val")
    test_rows = np.flatnonzero(split == "test")
    if tuple(map(len, (train_rows, validation_rows, test_rows))) != (14000, 3000, 3000):
        raise ValueError("linear48 split counts changed")
    development_rows = np.concatenate((train_rows, validation_rows))
    development_train = np.arange(len(train_rows), dtype=np.int64)
    development_validation = np.arange(
        len(train_rows), len(development_rows), dtype=np.int64
    )

    common = np.load(DATA_ROOT / "common3103.npy", mmap_mode="r", allow_pickle=False)
    target_bank = np.load(DATA_ROOT / "targets.npy", mmap_mode="r", allow_pickle=False)
    train_target = np.asarray(target_bank[train_rows, 0], dtype=np.float64)
    validation_target = np.asarray(target_bank[validation_rows, 0], dtype=np.float64)
    behavior = np.load(args.behavior_hashes, mmap_mode="r", allow_pickle=False)
    if behavior.shape != (row_count,):
        raise ValueError("behavior-group cache is not aligned with linear48 rows")
    train_behavior = np.asarray(behavior[train_rows]).astype(str)
    physics, physics_names, physics_metadata = _load_physics(
        args.physics_features.expanduser().resolve(), row_count
    )

    with np.load(DATA_ROOT / "runtime_bank.npz", allow_pickle=False) as runtime:
        runtime_rows = np.asarray(runtime["row_index"], dtype=np.int64)
        if not np.array_equal(runtime_rows, np.arange(row_count, dtype=np.int64)):
            raise ValueError("linear48 runtime bank is not in canonical row order")
        development_approx_top1 = np.asarray(
            runtime["approx_top1_correct_bits"][development_rows][:, selected],
            dtype=np.float32,
        )
        development_approx_top5 = np.asarray(
            runtime["approx_top5_correct_bits"][development_rows][:, selected],
            dtype=np.float32,
        )
        development_delta_margin_top1 = np.asarray(
            runtime["delta_margin_top1"][development_rows][:, selected],
            dtype=np.float32,
        )
        development_delta_margin_top5 = np.asarray(
            runtime["delta_margin_top5"][development_rows][:, selected],
            dtype=np.float32,
        )
    with np.load(
        DATASET_ROOT / "shared" / "strict50_bank_meta.npz", allow_pickle=False
    ) as bank_meta:
        exact_top1 = np.asarray(
            bank_meta["exact_top1_correct_bits"][selected], dtype=np.float32
        )
        exact_top5 = np.asarray(
            bank_meta["exact_top5_correct_bits"][selected], dtype=np.float32
        )

    development_common = np.asarray(common[development_rows], dtype=np.float32)
    prior_train_top1 = _predict(package_models["static_top1_train"], development_common)
    prior_train_top5 = _predict(package_models["static_top5_train"], development_common)
    strict_development, feature_names = _build_strict50_features(
        common=development_common,
        selected=selected,
        exact_top1=exact_top1,
        exact_top5=exact_top5,
        approx_top1=development_approx_top1,
        approx_top5=development_approx_top5,
        delta_margin_top1=development_delta_margin_top1,
        delta_margin_top5=development_delta_margin_top5,
        prior_top1=prior_train_top1,
        prior_top5=prior_train_top5,
    )
    package_schema = _read_object(package_dir / "data" / "feature_schema.json")
    if feature_names != package_schema.get("feature_names"):
        raise ValueError("reconstructed strict-50 feature schema differs from package")
    baseline_validation = _predict(
        package_models["top1_train"], strict_development[development_validation]
    )
    validation_match = _assert_package_prediction_match(
        package_dir / "data" / "validation_predictions.csv",
        validation_rows,
        baseline_validation,
    )

    strict_train_rows = np.asarray(
        strict_development[development_train], dtype=np.float32
    )
    physics_train_rows = np.asarray(physics[train_rows], dtype=np.float32)
    joint_train_rows = np.concatenate(
        (strict_train_rows, physics_train_rows), axis=1
    ).astype(np.float32)
    splitter = GroupKFold(n_splits=int(args.folds))
    folds = list(
        splitter.split(np.zeros(len(train_rows)), train_target, train_behavior)
    )
    baseline_oof = np.full(len(train_rows), np.nan, dtype=np.float64)
    physics_oof = np.full(len(train_rows), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for fold_index, (fit_position, holdout_position) in enumerate(folds):
        fit_position = np.asarray(fit_position, dtype=np.int64)
        holdout_position = np.asarray(holdout_position, dtype=np.int64)
        baseline_model = _clone_with_jobs(package_models["top1_train"], args.jobs)
        fit_lightgbm_model(
            baseline_model,
            strict_train_rows[fit_position],
            train_target[fit_position],
        )
        baseline_oof[holdout_position] = _predict(
            baseline_model, strict_train_rows[holdout_position]
        )
        physical_model = _clone_with_jobs(package_models["top1_train"], args.jobs)
        fit_lightgbm_model(
            physical_model,
            joint_train_rows[fit_position],
            train_target[fit_position],
        )
        physics_oof[holdout_position] = _predict(
            physical_model, joint_train_rows[holdout_position]
        )
        fold_records.append(
            {
                "fold": fold_index,
                "fit_rows": len(fit_position),
                "holdout_rows": len(holdout_position),
                "holdout_row_sha256": sha256_array(train_rows[holdout_position]),
                "baseline_metrics": metrics(
                    train_target[holdout_position], baseline_oof[holdout_position]
                ),
                "physics_metrics": metrics(
                    train_target[holdout_position], physics_oof[holdout_position]
                ),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "oof",
                    "completed_fold": fold_index + 1,
                    "folds": int(args.folds),
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if not np.isfinite(baseline_oof).all() or not np.isfinite(physics_oof).all():
        raise RuntimeError("OOF predictions are incomplete")
    disagreement_threshold = float(
        np.quantile(np.abs(physics_oof - baseline_oof), DISAGREEMENT_QUANTILE)
    )
    gated_oof, oof_active = _gated_blend(
        baseline_oof, physics_oof, threshold=disagreement_threshold
    )
    oof_metrics = {
        "baseline": metrics(train_target, baseline_oof),
        "physics": metrics(train_target, physics_oof),
        "gated_blend05": metrics(train_target, gated_oof),
    }

    physical_train_model = _clone_with_jobs(package_models["top1_train"], args.jobs)
    fit_lightgbm_model(physical_train_model, joint_train_rows, train_target)
    joint_validation = np.concatenate(
        (
            np.asarray(strict_development[development_validation], dtype=np.float32),
            np.asarray(physics[validation_rows], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    physics_validation = _predict(physical_train_model, joint_validation)
    gated_validation, validation_active = _gated_blend(
        baseline_validation,
        physics_validation,
        threshold=disagreement_threshold,
    )
    validation_metrics = {
        "baseline": metrics(validation_target, baseline_validation),
        "physics": metrics(validation_target, physics_validation),
        "gated_blend05": metrics(validation_target, gated_validation),
    }
    frozen_method = {
        "schema_version": "swin_linear48_causal_prefix_frozen_method_v1",
        "frozen_before_test_target_indexing": True,
        "source_method": "swin72 agreement-gated 5% causal-prefix blend",
        "base_package": str(package_dir),
        "base_package_hash": manifest["package_hash"],
        "selected_images": selected.tolist(),
        "selected_images_sha256": sha256_array(selected),
        "images_reselected": False,
        "physics_weight": FROZEN_PHYSICS_WEIGHT,
        "disagreement_quantile": DISAGREEMENT_QUANTILE,
        "disagreement_threshold": disagreement_threshold,
        "threshold_fit_split": "five-fold train behavior-group OOF",
        "validation_metrics": validation_metrics,
        "validation_joint_nonregression": joint_nonregression(
            validation_metrics["baseline"], validation_metrics["gated_blend05"]
        ),
        "test_targets_indexed": 0,
        "test_runtime_rows_indexed": 0,
    }
    write_json(frozen_path, frozen_method)

    # Test phase begins only after the complete method contract above exists.
    test_target = np.asarray(target_bank[test_rows, 0], dtype=np.float64)
    with np.load(DATA_ROOT / "runtime_bank.npz", allow_pickle=False) as runtime:
        test_approx_top1 = np.asarray(
            runtime["approx_top1_correct_bits"][test_rows][:, selected],
            dtype=np.float32,
        )
        test_approx_top5 = np.asarray(
            runtime["approx_top5_correct_bits"][test_rows][:, selected],
            dtype=np.float32,
        )
        test_delta_margin_top1 = np.asarray(
            runtime["delta_margin_top1"][test_rows][:, selected],
            dtype=np.float32,
        )
        test_delta_margin_top5 = np.asarray(
            runtime["delta_margin_top5"][test_rows][:, selected],
            dtype=np.float32,
        )
    test_common = np.asarray(common[test_rows], dtype=np.float32)
    prior_refit_development_top1 = _predict(
        package_models["static_top1_refit"], development_common
    )
    prior_refit_development_top5 = _predict(
        package_models["static_top5_refit"], development_common
    )
    prior_refit_test_top1 = _predict(package_models["static_top1_refit"], test_common)
    prior_refit_test_top5 = _predict(package_models["static_top5_refit"], test_common)
    strict_refit_development, refit_feature_names = _build_strict50_features(
        common=development_common,
        selected=selected,
        exact_top1=exact_top1,
        exact_top5=exact_top5,
        approx_top1=development_approx_top1,
        approx_top5=development_approx_top5,
        delta_margin_top1=development_delta_margin_top1,
        delta_margin_top5=development_delta_margin_top5,
        prior_top1=prior_refit_development_top1,
        prior_top5=prior_refit_development_top5,
    )
    strict_refit_test, test_feature_names = _build_strict50_features(
        common=test_common,
        selected=selected,
        exact_top1=exact_top1,
        exact_top5=exact_top5,
        approx_top1=test_approx_top1,
        approx_top5=test_approx_top5,
        delta_margin_top1=test_delta_margin_top1,
        delta_margin_top5=test_delta_margin_top5,
        prior_top1=prior_refit_test_top1,
        prior_top5=prior_refit_test_top5,
    )
    if refit_feature_names != feature_names or test_feature_names != feature_names:
        raise RuntimeError("train and refit strict-50 schemas differ")
    baseline_test = _predict(package_models["top1_refit"], strict_refit_test)
    test_match = _assert_package_prediction_match(
        package_dir / "data" / "holdout_predictions.csv", test_rows, baseline_test
    )
    refit_target = np.concatenate((train_target, validation_target))
    joint_refit = np.concatenate(
        (
            np.asarray(strict_refit_development, dtype=np.float32),
            np.asarray(physics[development_rows], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    physical_refit_model = _clone_with_jobs(package_models["top1_refit"], args.jobs)
    fit_lightgbm_model(physical_refit_model, joint_refit, refit_target)
    joint_test = np.concatenate(
        (
            np.asarray(strict_refit_test, dtype=np.float32),
            np.asarray(physics[test_rows], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32)
    physics_test = _predict(physical_refit_model, joint_test)
    gated_test, test_active = _gated_blend(
        baseline_test, physics_test, threshold=disagreement_threshold
    )
    test_metrics = {
        "baseline": metrics(test_target, baseline_test),
        "physics": metrics(test_target, physics_test),
        "gated_blend05": metrics(test_target, gated_test),
    }
    baseline_error = np.abs(baseline_test - test_target)
    candidate_error = np.abs(gated_test - test_target)
    baseline_worst = int(np.argmax(baseline_error))
    candidate_worst = int(np.argmax(candidate_error))

    _write_predictions(
        predictions_path,
        train_rows=train_rows,
        train_target=train_target.astype(np.float32),
        baseline_oof=baseline_oof.astype(np.float32),
        physics_oof=physics_oof.astype(np.float32),
        gated_oof=gated_oof.astype(np.float32),
        oof_active=oof_active,
        validation_rows=validation_rows,
        validation_target=validation_target.astype(np.float32),
        baseline_validation=baseline_validation.astype(np.float32),
        physics_validation=physics_validation.astype(np.float32),
        gated_validation=gated_validation.astype(np.float32),
        validation_active=validation_active,
        test_rows=test_rows,
        test_target=test_target.astype(np.float32),
        baseline_test=baseline_test.astype(np.float32),
        physics_test=physics_test.astype(np.float32),
        gated_test=gated_test.astype(np.float32),
        test_active=test_active,
    )
    temporary_models = models_path.with_name(models_path.name + ".partial")
    joblib.dump(
        {
            "schema_version": "swin_linear48_causal_prefix_models_v1",
            "physical_train_model": physical_train_model,
            "physical_refit_model": physical_refit_model,
            "physics_feature_names": physics_names,
            "strict50_feature_names": feature_names,
            "frozen_method": frozen_method,
        },
        temporary_models,
    )
    os.replace(temporary_models, models_path)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "frozen_72node_method_transfer_to_linear48_test_scored_once",
        "method": frozen_method,
        "protocol": {
            "selected_images_changed": False,
            "test_evaluation_count": 1,
            "test_targets_indexed_after_method_freeze": len(test_rows),
            "test_runtime_rows_indexed_after_method_freeze": len(test_rows),
            "test_used_for_tuning": False,
            "baseline_package_quick_trial": True,
        },
        "configuration": {
            "folds": int(args.folds),
            "jobs": int(args.jobs),
            "base_feature_count": strict_development.shape[1],
            "physics_feature_count": physics.shape[1],
            "joint_feature_count": strict_development.shape[1] + physics.shape[1],
        },
        "baseline_reproduction": {
            "validation_max_abs_difference": validation_match,
            "test_max_abs_difference": test_match,
        },
        "oof": {
            "metrics": oof_metrics,
            "active_fraction": float(oof_active.mean()),
            "joint_nonregression": joint_nonregression(
                oof_metrics["baseline"], oof_metrics["gated_blend05"]
            ),
            "folds": fold_records,
        },
        "validation": {
            "metrics": validation_metrics,
            "active_fraction": float(validation_active.mean()),
            "joint_nonregression": frozen_method["validation_joint_nonregression"],
        },
        "test": {
            "metrics": test_metrics,
            "active_fraction": float(test_active.mean()),
            "joint_nonregression": joint_nonregression(
                test_metrics["baseline"], test_metrics["gated_blend05"]
            ),
            "baseline_worst": {
                "row_index": int(test_rows[baseline_worst]),
                "design_id": str(
                    configuration.iloc[test_rows[baseline_worst]]["design_id"]
                ),
                "actual": float(test_target[baseline_worst]),
                "prediction": float(baseline_test[baseline_worst]),
                "abs_error": float(baseline_error[baseline_worst]),
                "candidate_abs_error_same_row": float(candidate_error[baseline_worst]),
            },
            "candidate_worst": {
                "row_index": int(test_rows[candidate_worst]),
                "design_id": str(
                    configuration.iloc[test_rows[candidate_worst]]["design_id"]
                ),
                "actual": float(test_target[candidate_worst]),
                "prediction": float(gated_test[candidate_worst]),
                "abs_error": float(candidate_error[candidate_worst]),
            },
        },
        "artifacts": {
            "physics_features": str(args.physics_features.resolve()),
            "physics_features_sha256": physics_metadata["output_sha256"],
            "frozen_method": str(frozen_path),
            "frozen_method_sha256": sha256_file(frozen_path),
            "predictions": str(predictions_path),
            "predictions_sha256": sha256_file(predictions_path),
            "models": str(models_path),
            "models_sha256": sha256_file(models_path),
        },
        "elapsed_seconds": time.time() - started,
    }
    write_json(result_path, result)
    report = "\n".join(
        [
            "# Swin linear48 causal-prefix transfer",
            "",
            "The 72-node method was transferred without changing the 50 images,",
            "5% blend weight, or q75 disagreement-gate recipe.",
            "",
            "| Split / method | MAE (pp) | P95 (pp) | P99 (pp) | Max (pp) | >5 pp |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            _format_metric_row("OOF baseline", oof_metrics["baseline"]),
            _format_metric_row("OOF gated physics", oof_metrics["gated_blend05"]),
            _format_metric_row("Validation baseline", validation_metrics["baseline"]),
            _format_metric_row(
                "Validation gated physics", validation_metrics["gated_blend05"]
            ),
            _format_metric_row("Test baseline", test_metrics["baseline"]),
            _format_metric_row("Test gated physics", test_metrics["gated_blend05"]),
            "",
            f"OOF gate: `{result['oof']['joint_nonregression']}`",
            f"Validation gate: `{result['validation']['joint_nonregression']}`",
            f"Test gate: `{result['test']['joint_nonregression']}`",
            "",
            f"Test was scored once after `{frozen_path.name}` was written.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--physics-features", type=Path, default=DEFAULT_PHYSICS_FEATURES
    )
    result.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    result.add_argument("--behavior-hashes", type=Path, default=DEFAULT_BEHAVIOR_HASHES)
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    result.add_argument("--folds", type=int, default=5)
    result.add_argument("--jobs", type=int, default=12)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.folds < 2 or args.jobs < 1:
        raise SystemExit("--folds and --jobs must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
