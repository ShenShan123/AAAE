#!/usr/bin/env python
"""Build a checked CNN bundle with validation-selected delta-CNN members."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.cnn.artifacts import (
    BASE_BUNDLE_FILES,
    HYBRID_BUNDLE_SCHEMA_VERSION,
    _sha256,
    _write_checksums,
    _write_json,
    ensure_output_available,
    load_artifact,
)
from aaae.accuracy_predictor.cnn.correctness_conv import (
    CHECKPOINT_SCHEMA_VERSION as CORRECTNESS_CHECKPOINT_SCHEMA_VERSION,
)
from aaae.accuracy_predictor.cnn.correctness_conv import (
    export_correctness_conv_checkpoint,
)
from aaae.accuracy_predictor.cnn.delta_conv import (
    CHECKPOINT_SCHEMA_VERSION as DELTA_CHECKPOINT_SCHEMA_VERSION,
)
from aaae.accuracy_predictor.cnn.delta_conv import (
    SUPPORTED_INFERENCE_DTYPES,
    export_delta_conv_checkpoint,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("hybrid bundle manifest must be a JSON object")
    return payload


def _resolve(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    absolute = np.abs(error)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "over_5pp_count": int(np.sum(absolute > 0.05)),
    }


def _load_predictions(path: Path, *, expected_rows: int, field: str) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load {field} at {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (expected_rows,) or not np.isfinite(values).all():
        raise ValueError(
            f"{field} must contain exactly {expected_rows} finite predictions"
        )
    return values


def _validation_minimax(
    actual: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    row_count, member_count = candidates.shape
    objective = np.concatenate((np.zeros(member_count + 1), np.ones(1)))
    upper = np.vstack(
        (
            np.column_stack((candidates, np.ones(row_count), -np.ones(row_count))),
            np.column_stack((-candidates, -np.ones(row_count), -np.ones(row_count))),
        )
    )
    equality = np.zeros((1, member_count + 2), dtype=np.float64)
    equality[0, :member_count] = 1.0
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=np.concatenate((actual, -actual)),
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0.0, None)] * member_count + [(None, None), (0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"validation minimax optimization failed: {result.message}")
    weights = np.asarray(result.x[:member_count], dtype=np.float64)
    offset = float(result.x[member_count])
    maximum = float(result.x[member_count + 1])
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError("validation minimax returned weights that do not sum to one")
    return weights, offset, maximum


def _copy_base_files(base: Path, temporary: Path) -> None:
    replaced = {
        "manifest.json",
        "metrics.json",
        "provenance.json",
        "training_manifest.snapshot.json",
    }
    for relative in BASE_BUNDLE_FILES - replaced:
        source = base / relative
        destination = temporary / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    raw = _read_json(manifest_path)
    if raw.get("schema_version") != 1:
        raise ValueError("hybrid bundle build manifest schema_version must be 1")
    base_dir = manifest_path.parent
    base_bundle = _resolve(base_dir, raw.get("base_bundle"), field="base_bundle")
    output = _resolve(base_dir, raw.get("output"), field="output")
    split_path = _resolve(base_dir, raw.get("split"), field="split")
    labels_path = _resolve(base_dir, raw.get("labels"), field="labels")
    ensure_output_available(output)

    base_artifact = load_artifact(base_bundle)
    base_manifest = dict(base_artifact.manifest)
    if base_manifest.get("schema_version") != 2:
        raise ValueError("base_bundle must use checked LightGBM bundle schema v2")
    feature_schema = str(base_manifest.get("feature_schema"))

    split = pd.read_csv(split_path, dtype={"design_id": str})
    labels = pd.read_csv(labels_path, dtype={"design_id": str})
    if not {"design_id", "split"}.issubset(split.columns):
        raise ValueError("split CSV must contain design_id and split")
    aligned = split[["design_id", "split"]].merge(
        labels[["design_id", "top1_drop"]],
        on="design_id",
        validate="one_to_one",
    )
    counts = aligned["split"].value_counts().to_dict()
    if counts != {"train": 14000, "validation": 3000, "test": 3000}:
        raise ValueError(
            f"hybrid bundle requires exact 14000/3000/3000 splits; got {counts}"
        )
    validation_actual = aligned.loc[
        aligned["split"].eq("validation"), "top1_drop"
    ].to_numpy(dtype=np.float64)
    test_actual = aligned.loc[aligned["split"].eq("test"), "top1_drop"].to_numpy(
        dtype=np.float64
    )

    top1_raw = raw.get("top1_drop")
    if not isinstance(top1_raw, dict) or not isinstance(top1_raw.get("members"), list):
        raise TypeError("top1_drop.members must be a list")
    inference_dtype = top1_raw.get("inference_dtype", "float32")
    if inference_dtype not in SUPPORTED_INFERENCE_DTYPES:
        raise ValueError(
            f"top1_drop.inference_dtype is unsupported: {inference_dtype!r}"
        )
    member_specs: list[dict[str, Any]] = []
    validation_candidates: list[np.ndarray] = []
    test_candidates: list[np.ndarray] = []
    names: set[str] = set()
    for index, member_raw in enumerate(top1_raw["members"]):
        if not isinstance(member_raw, dict):
            raise TypeError(f"top1_drop.members[{index}] must be an object")
        name = member_raw.get("name")
        if (
            not isinstance(name, str)
            or not name
            or not name.replace("_", "").replace("-", "").isalnum()
            or name in names
        ):
            raise ValueError(f"top1_drop member has invalid/duplicate name: {name!r}")
        names.add(name)
        kind = member_raw.get("kind", "delta_conv")
        if kind not in {"delta_conv", "correctness_conv"}:
            raise ValueError(
                f"top1_drop member {name!r} has unsupported kind: {kind!r}"
            )
        checkpoint = _resolve(
            base_dir,
            member_raw.get("checkpoint"),
            field=f"top1_drop.members[{index}].checkpoint",
        )
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_feature_schema = (
            checkpoint_payload.get("feature_schema")
            if isinstance(checkpoint_payload, dict)
            else None
        )
        if checkpoint_feature_schema != feature_schema:
            raise ValueError(
                f"top1_drop member {name!r} feature schema does not match "
                f"the base bundle: {checkpoint_feature_schema!r} "
                f"!= {feature_schema!r}"
            )
        validation_path = _resolve(
            base_dir,
            member_raw.get("validation_predictions"),
            field=f"top1_drop.members[{index}].validation_predictions",
        )
        test_path = _resolve(
            base_dir,
            member_raw.get("test_predictions"),
            field=f"top1_drop.members[{index}].test_predictions",
        )
        validation_candidates.append(
            _load_predictions(
                validation_path,
                expected_rows=len(validation_actual),
                field=f"{name} validation predictions",
            )
        )
        test_candidates.append(
            _load_predictions(
                test_path,
                expected_rows=len(test_actual),
                field=f"{name} test predictions",
            )
        )
        member_specs.append(
            {
                "name": name,
                "kind": str(kind),
                "logit_temperature": float(member_raw.get("logit_temperature", 1.0)),
                "checkpoint": checkpoint,
                "checkpoint_sha256": _sha256(checkpoint),
                "validation_predictions": validation_path,
                "validation_predictions_sha256": _sha256(validation_path),
                "test_predictions": test_path,
                "test_predictions_sha256": _sha256(test_path),
            }
        )
    if len(member_specs) < 2:
        raise ValueError(
            "hybrid minimax selection requires at least two neural members"
        )

    validation_matrix = np.column_stack(validation_candidates)
    test_matrix = np.column_stack(test_candidates)
    weights, offset, solver_maximum = _validation_minimax(
        validation_actual,
        validation_matrix,
    )
    validation_prediction = validation_matrix @ weights + offset
    test_prediction = test_matrix @ weights + offset
    validation_metrics = _metrics(validation_actual, validation_prediction)
    test_metrics = _metrics(test_actual, test_prediction)
    if not math.isclose(
        validation_metrics["max_abs_error"],
        solver_maximum,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("validation minimax solver result does not reproduce")

    threshold = float(top1_raw.get("max_abs_error_threshold", 0.05))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("top1_drop.max_abs_error_threshold must be positive")
    allow_failed_gate = top1_raw.get("allow_failed_quality_gate", False)
    if not isinstance(allow_failed_gate, bool):
        raise TypeError("top1_drop.allow_failed_quality_gate must be boolean")
    top1_passed = bool(test_metrics["max_abs_error"] <= threshold)
    if not top1_passed and not allow_failed_gate:
        raise RuntimeError(
            "refusing to publish hybrid bundle: post-failure test top1 max error "
            f"{test_metrics['max_abs_error']:.9f} exceeds {threshold:.9f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _copy_base_files(base_bundle, temporary)
        top1_members: list[dict[str, Any]] = []
        for member, weight in zip(member_specs, weights, strict=True):
            relative = Path("models/top1_drop") / f"{member['name']}.pt"
            destination = temporary / relative
            if member["kind"] == "correctness_conv":
                export_correctness_conv_checkpoint(
                    member["checkpoint"],
                    destination,
                    inference_dtype=str(inference_dtype),
                    logit_temperature=float(member.get("logit_temperature", 1.0)),
                )
                checkpoint_schema = CORRECTNESS_CHECKPOINT_SCHEMA_VERSION
            else:
                export_delta_conv_checkpoint(
                    member["checkpoint"],
                    destination,
                    inference_dtype=str(inference_dtype),
                )
                checkpoint_schema = DELTA_CHECKPOINT_SCHEMA_VERSION
            top1_members.append(
                {
                    "kind": member["kind"],
                    "name": member["name"],
                    "path": relative.as_posix(),
                    "weight": float(weight),
                    "checkpoint_schema": checkpoint_schema,
                    "source_checkpoint_sha256": member["checkpoint_sha256"],
                    "validation_predictions_sha256": member[
                        "validation_predictions_sha256"
                    ],
                    "post_failure_test_predictions_sha256": member[
                        "test_predictions_sha256"
                    ],
                }
            )

        base_ensembles = base_manifest.get("model_ensembles")
        if not isinstance(base_ensembles, dict):
            raise TypeError("base bundle has no model_ensembles")
        top5_base = base_ensembles.get("top5_drop")
        if not isinstance(top5_base, dict) or not isinstance(
            top5_base.get("members"), list
        ):
            raise TypeError("base bundle has an invalid top5_drop ensemble")
        top5_members: list[dict[str, Any]] = []
        for member in top5_base["members"]:
            copied = dict(member)
            copied["kind"] = "lightgbm"
            relative = Path(str(copied["path"]))
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(base_bundle / relative, destination)
            top5_members.append(copied)

        model_ensembles = {
            "top1_drop": {"offset": offset, "members": top1_members},
            "top5_drop": {
                "offset": float(top5_base.get("offset", 0.0)),
                "members": top5_members,
            },
        }
        base_metrics = _read_json(base_bundle / "metrics.json")
        metrics = {
            "top1_drop": {
                **test_metrics,
                "evaluation_status": "post_failure_test_audit",
                "validation": validation_metrics,
                "post_failure_test": test_metrics,
                "offset": offset,
                "members": {
                    member["name"]: {"weight": float(weight)}
                    for member, weight in zip(member_specs, weights, strict=True)
                },
                "quality_gate": {
                    "metric": "max_abs_error",
                    "threshold": threshold,
                    "passed": top1_passed,
                },
            },
            "top5_drop": base_metrics["top5_drop"],
        }
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "training_manifest.snapshot.json", raw)

        provenance = _read_json(base_bundle / "provenance.json")
        provenance.update(
            {
                "selection_protocol": "validation_nonnegative_affine_minimax",
                "selection_labels": "train_and_validation_only",
                "test_evaluation_status": "post_failure_test_audit",
                "test_labels_unsealed_after_prior_frozen_candidate_failed": True,
                "test_labels_used_for_delta_conv_training": False,
                "test_labels_used_for_ensemble_weight_selection": False,
                "hybrid_build_manifest_sha256": _sha256(manifest_path),
                "base_bundle": str(base_bundle),
                "base_bundle_manifest_sha256": _sha256(base_bundle / "manifest.json"),
            }
        )
        _write_json(temporary / "provenance.json", provenance)

        final_manifest = dict(base_manifest)
        final_manifest.update(
            {
                "schema_version": HYBRID_BUNDLE_SCHEMA_VERSION,
                "bundle_kind": "cnn_ame_hybrid_accuracy_drop",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "learned_backend": "lightgbm+torch",
                "torch_version": str(torch.__version__),
                "claim_status": "measured_post_failure_audit",
                "synthetic_only": False,
                "performance_claim_allowed": False,
                "quality_gate_passed": False,
                "target_quality_gates": {
                    "top1_drop": {
                        "metric": "max_abs_error",
                        "threshold": threshold,
                        "passed": top1_passed,
                        "evaluation_status": "post_failure_test_audit",
                    },
                    "top5_drop": {
                        "metric": "max_abs_error",
                        "threshold": threshold,
                        "passed": bool(
                            base_metrics["top5_drop"]["max_abs_error"] <= threshold
                        ),
                        "evaluation_status": "original_frozen_test",
                    },
                },
                "model_ensembles": model_ensembles,
                "execution": {
                    "delta_conv_default_device": "cpu",
                    "delta_conv_a100_device": "cuda",
                    "delta_conv_cuda_dtype": str(inference_dtype),
                    "delta_conv_batch_size": 256,
                    "correctness_conv_default_device": "cpu",
                    "correctness_conv_a100_device": "cuda",
                    "correctness_conv_cuda_dtype": str(inference_dtype),
                    "correctness_conv_batch_size": 256,
                },
                "training": {
                    "top1_drop": {
                        "architecture": "neural_lut_ensemble_v1",
                        "member_training_rows": int(
                            top1_raw.get("member_training_rows", 14000)
                        ),
                        "member_selection_rows": 3000,
                        "ensemble_selection": "validation_nonnegative_affine_minimax",
                    },
                    "top5_drop": base_manifest.get("training"),
                },
            }
        )
        _write_json(temporary / "manifest.json", final_manifest)
        _write_checksums(temporary)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return {
        "bundle": str(output),
        "bundle_manifest_sha256": _sha256(output / "manifest.json"),
        "weights": {
            member["name"]: float(weight)
            for member, weight in zip(member_specs, weights, strict=True)
        },
        "offset": offset,
        "validation_metrics": validation_metrics,
        "post_failure_test_metrics": test_metrics,
        "top1_quality_gate_passed": top1_passed,
        "overall_quality_gate_passed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build(args.manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
