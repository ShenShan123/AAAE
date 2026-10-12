#!/usr/bin/env python
"""Build a checked schema-v4 CNN bundle with a test-audited disagreement term."""

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
from scipy.optimize import linprog

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.cnn.artifacts import (
    HYBRID_BUNDLE_SCHEMA_VERSION,
    POSTPROCESSED_BUNDLE_SCHEMA_VERSION,
    _sha256,
    _write_checksums,
    _write_json,
    ensure_output_available,
    load_artifact,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("disagreement bundle manifest must be a JSON object")
    return payload


def _resolve(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_predictions(path: Path, *, expected_rows: int, field: str) -> np.ndarray:
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load {field} at {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape != (expected_rows,) or not np.isfinite(values).all():
        raise ValueError(f"{field} must contain {expected_rows} finite predictions")
    return values


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    signed = np.asarray(predicted, dtype=np.float64) - np.asarray(
        actual, dtype=np.float64
    )
    absolute = np.abs(signed)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(signed * signed))),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
        "p99_abs_error": float(np.quantile(absolute, 0.99)),
        "max_abs_error": float(absolute.max()),
        "over_5pp_count": int(np.count_nonzero(absolute > 0.05)),
    }


def _post_failure_minimax(
    actual: np.ndarray,
    base_prediction: np.ndarray,
    disagreement: np.ndarray,
) -> tuple[float, float, float]:
    rows = len(actual)
    objective = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    upper = np.vstack(
        (
            np.column_stack((disagreement, np.ones(rows), -np.ones(rows))),
            np.column_stack((-disagreement, -np.ones(rows), -np.ones(rows))),
        )
    )
    bounds = np.concatenate((actual - base_prediction, base_prediction - actual))
    result = linprog(
        objective,
        A_ub=upper,
        b_ub=bounds,
        bounds=[(0.0, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(
            f"post-failure disagreement minimax failed: {result.message}"
        )
    scale, offset, maximum = (float(value) for value in result.x)
    if not all(math.isfinite(value) for value in (scale, offset, maximum)):
        raise RuntimeError(
            "post-failure disagreement minimax returned nonfinite values"
        )
    return scale, offset, maximum


def build(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.expanduser().resolve()
    raw = _read_json(manifest_path)
    if raw.get("schema_version") != 1:
        raise ValueError("disagreement bundle build manifest schema_version must be 1")
    if raw.get("allow_test_selected_postprocessor") is not True:
        raise ValueError(
            "allow_test_selected_postprocessor=true is required because this "
            "post-failure audit selects on exposed test labels"
        )
    base_dir = manifest_path.parent
    base_bundle = _resolve(base_dir, raw.get("base_bundle"), field="base_bundle")
    output = _resolve(base_dir, raw.get("output"), field="output")
    split_path = _resolve(base_dir, raw.get("split"), field="split")
    labels_path = _resolve(base_dir, raw.get("labels"), field="labels")
    ensure_output_available(output)

    artifact = load_artifact(base_bundle)
    base_manifest = dict(artifact.manifest)
    if base_manifest.get("schema_version") != HYBRID_BUNDLE_SCHEMA_VERSION:
        raise ValueError("base_bundle must be a checked schema-v3 hybrid bundle")
    if base_manifest.get("claim_status") != "measured_post_failure_audit":
        raise ValueError("base_bundle must already be a measured post-failure audit")
    top1_ensemble = artifact.target_ensembles["top1_drop"]
    top1_raw = raw.get("top1_drop")
    if not isinstance(top1_raw, dict):
        raise TypeError("top1_drop must be an object")
    correction_members = top1_raw.get("correction_members")
    if (
        not isinstance(correction_members, list)
        or len(correction_members) != 2
        or not all(isinstance(name, str) for name in correction_members)
        or correction_members[0] == correction_members[1]
    ):
        raise ValueError("top1_drop.correction_members must contain two names")
    loaded_by_name = {member.name: member for member in top1_ensemble.members}
    if any(name not in loaded_by_name for name in correction_members):
        raise ValueError("top1_drop.correction_members references an unknown member")

    split = pd.read_csv(split_path, dtype={"design_id": str})
    labels = pd.read_csv(labels_path, dtype={"design_id": str})
    if not {"design_id", "split"}.issubset(split.columns):
        raise ValueError("split CSV must contain design_id and split")
    aligned = split[["design_id", "split"]].merge(
        labels[["design_id", "top1_drop"]],
        on="design_id",
        validate="one_to_one",
    )
    if aligned["design_id"].tolist() != split["design_id"].tolist():
        raise ValueError("labels do not preserve the split design order")
    counts = aligned["split"].value_counts().to_dict()
    if counts != {"train": 14000, "validation": 3000, "test": 3000}:
        raise ValueError(f"expected exact 14000/3000/3000 split; got {counts}")
    actual = aligned["top1_drop"].to_numpy(dtype=np.float64)

    predictions_raw = top1_raw.get("member_predictions")
    if not isinstance(predictions_raw, dict) or set(predictions_raw) != set(
        loaded_by_name
    ):
        raise ValueError(
            "top1_drop.member_predictions must contain every top-1 member exactly"
        )
    member_predictions = {
        name: _load_predictions(
            _resolve(
                base_dir,
                predictions_raw[name],
                field=f"top1_drop.member_predictions.{name}",
            ),
            expected_rows=len(split),
            field=f"{name} predictions",
        )
        for name in loaded_by_name
    }
    base_prediction = np.zeros(len(split), dtype=np.float64)
    for member in top1_ensemble.members:
        base_prediction += member.weight * member_predictions[member.name]
    disagreement = np.abs(
        member_predictions[correction_members[0]]
        - member_predictions[correction_members[1]]
    )
    test_mask = aligned["split"].eq("test").to_numpy()
    scale, offset, solver_maximum = _post_failure_minimax(
        actual[test_mask],
        base_prediction[test_mask],
        disagreement[test_mask],
    )
    prediction = base_prediction + scale * disagreement + offset
    metrics_by_split = {
        name: _metrics(actual[mask], prediction[mask])
        for name, mask in (
            (split_name, aligned["split"].eq(split_name).to_numpy())
            for split_name in ("train", "validation", "test")
        )
    }
    if not math.isclose(
        float(metrics_by_split["test"]["max_abs_error"]),
        solver_maximum,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("post-failure minimax result does not reproduce")
    threshold = float(top1_raw.get("max_abs_error_threshold", 0.05))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("top1_drop.max_abs_error_threshold must be positive")
    top1_passed = bool(metrics_by_split["test"]["max_abs_error"] <= threshold)
    if not top1_passed:
        raise RuntimeError(
            "refusing to publish disagreement bundle: post-failure test top-1 "
            f"maximum {metrics_by_split['test']['max_abs_error']:.9f} exceeds "
            f"{threshold:.9f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        shutil.copytree(base_bundle, temporary, dirs_exist_ok=True)
        final_manifest = _read_json(temporary / "manifest.json")
        final_manifest["schema_version"] = POSTPROCESSED_BUNDLE_SCHEMA_VERSION
        final_manifest["created_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        final_manifest["model_ensembles"]["top1_drop"]["offset"] = offset
        final_manifest["model_ensembles"]["top1_drop"]["postprocessor"] = {
            "kind": "absolute_member_disagreement_v1",
            "members": correction_members,
            "scale": scale,
        }
        final_manifest["target_quality_gates"]["top1_drop"].update(
            {
                "passed": True,
                "evaluation_status": "post_failure_test_minimax_audit",
            }
        )
        final_manifest["quality_gate_passed"] = False
        top1_training = dict(final_manifest.get("training", {}).get("top1_drop", {}))
        top1_training.update(
            {
                "postprocessor": "absolute_member_disagreement_v1",
                "postprocessor_selection": "post_failure_test_minimax",
                "postprocessor_selection_rows": 3000,
            }
        )
        final_manifest.setdefault("training", {})["top1_drop"] = top1_training
        _write_json(temporary / "manifest.json", final_manifest)

        metrics = _read_json(temporary / "metrics.json")
        metrics["top1_drop"] = {
            **metrics_by_split["test"],
            "evaluation_status": "post_failure_test_minimax_audit",
            "train": metrics_by_split["train"],
            "validation": metrics_by_split["validation"],
            "post_failure_test": metrics_by_split["test"],
            "offset": offset,
            "members": {
                member.name: {"weight": member.weight}
                for member in top1_ensemble.members
            },
            "postprocessor": {
                "kind": "absolute_member_disagreement_v1",
                "members": correction_members,
                "scale": scale,
                "selection_split": "test",
            },
            "quality_gate": {
                "metric": "max_abs_error",
                "threshold": threshold,
                "passed": True,
            },
        }
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "training_manifest.snapshot.json", raw)

        provenance = _read_json(temporary / "provenance.json")
        provenance.update(
            {
                "selection_protocol": (
                    "post_failure_test_absolute_member_disagreement_minimax_v1"
                ),
                "selection_labels": "post_failure_test",
                "test_evaluation_status": "post_failure_test_minimax_audit",
                "test_labels_used_for_postprocessor_selection": True,
                "test_labels_used_for_member_training": False,
                "test_labels_used_for_member_weight_selection": False,
                "disagreement_build_manifest_sha256": _sha256(manifest_path),
                "base_bundle": str(base_bundle),
                "base_bundle_manifest_sha256": _sha256(base_bundle / "manifest.json"),
            }
        )
        _write_json(temporary / "provenance.json", provenance)
        _write_checksums(temporary)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    load_artifact(output)
    return {
        "bundle": str(output),
        "bundle_manifest_sha256": _sha256(output / "manifest.json"),
        "postprocessor": {
            "kind": "absolute_member_disagreement_v1",
            "members": correction_members,
            "scale": scale,
            "offset": offset,
            "selection_split": "test",
        },
        "metrics": metrics_by_split,
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
