from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Keep LightGBM first. Executing this file directly isolates it from any ONNX
# or other native libraries already loaded by the caller.
import lightgbm as lgb
import numpy as np


def _load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must be a JSON object")
    return payload


def _dataset(
    features: np.ndarray,
    target: np.ndarray,
    *,
    feature_names: list[str],
    params: dict[str, Any],
    reference: lgb.Dataset | None = None,
    weight: np.ndarray | None = None,
) -> lgb.Dataset:
    return lgb.Dataset(
        np.ascontiguousarray(features, dtype=np.float32),
        label=np.ascontiguousarray(target, dtype=np.float32),
        weight=None if weight is None else np.ascontiguousarray(weight, dtype=np.float32),
        feature_name=feature_names,
        reference=reference,
        free_raw_data=False,
        params=dict(params),
    )


def _sample_weights(target: np.ndarray, spec: dict[str, Any]) -> np.ndarray | None:
    kind = spec.get("kind", "none")
    if kind == "none":
        return None
    if kind != "target_density":
        raise ValueError(f"unsupported sample weighting kind: {kind!r}")
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    bins = int(spec["bins"])
    strength = float(spec["strength"])
    minimum = float(spec["min_weight"])
    maximum = float(spec["max_weight"])
    edges = np.linspace(float(values.min()), float(values.max()) + 1e-12, bins + 1)
    assignments = np.clip(
        np.searchsorted(edges, values, side="right") - 1,
        0,
        bins - 1,
    )
    counts = np.bincount(assignments, minlength=bins)
    populated = counts[counts > 0]
    reference = float(np.median(populated))
    raw = np.power(reference / counts[assignments], strength)
    clipped = np.clip(raw, minimum, maximum)
    return clipped / clipped.mean()


def _max_abs_error(predicted: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
    actual = dataset.get_label()
    return "max_abs_error", float(np.max(np.abs(predicted - actual))), False


def _load_models(request_path: Path) -> tuple[dict[str, lgb.Booster], dict[str, Any]]:
    request = _load_request(request_path)
    model_paths = request.get("models")
    if not isinstance(model_paths, dict) or not model_paths:
        raise ValueError("request.models must be a nonempty object")
    boosters: dict[str, lgb.Booster] = {}
    metadata: dict[str, Any] = {}
    for target, raw_path in sorted(model_paths.items()):
        if not isinstance(target, str) or not target or not isinstance(raw_path, str):
            raise ValueError("request.models must map nonempty names to paths")
        booster = lgb.Booster(model_file=raw_path)
        boosters[target] = booster
        metadata[target] = {
            "feature_count": int(booster.num_feature()),
            "feature_names": booster.feature_name(),
        }
    return boosters, metadata


def fit(args: argparse.Namespace) -> None:
    features = np.load(args.features, allow_pickle=False)
    target = np.load(args.target, allow_pickle=False)
    splits = np.load(args.splits, allow_pickle=False).astype(str)
    request = _load_request(args.request)
    params = request["params"]
    feature_names = request["feature_names"]
    n_estimators = int(request["n_estimators"])
    early_stopping_rounds = int(request["early_stopping_rounds"])
    selection_metric = str(request.get("selection_metric", "l1"))
    sample_weighting = request.get("sample_weighting", {"kind": "none"})
    if not isinstance(params, dict):
        raise ValueError("request.params must be an object")
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) and name for name in feature_names
    ):
        raise ValueError("request.feature_names must contain nonempty strings")
    if selection_metric not in {"l1", "max_abs_error"}:
        raise ValueError("selection_metric must be 'l1' or 'max_abs_error'")
    if not isinstance(sample_weighting, dict):
        raise ValueError("sample_weighting must be an object")

    train_mask = splits == "train"
    validation_mask = splits == "validation"
    test_mask = splits == "test"
    training_set = _dataset(
        features[train_mask],
        target[train_mask],
        feature_names=feature_names,
        params=params,
        weight=_sample_weights(target[train_mask], sample_weighting),
    )
    validation_set = _dataset(
        features[validation_mask],
        target[validation_mask],
        feature_names=feature_names,
        params=params,
        reference=training_set,
    )
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    fit_options: dict[str, Any] = {}
    if selection_metric == "max_abs_error":
        fit_options["feval"] = _max_abs_error
    selector = lgb.train(
        params,
        training_set,
        num_boost_round=n_estimators,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
        **fit_options,
    )
    selected_iterations = int(selector.best_iteration or n_estimators)

    refit_mask = train_mask | validation_mask
    refit_set = _dataset(
        features[refit_mask],
        target[refit_mask],
        feature_names=feature_names,
        params=params,
        weight=_sample_weights(target[refit_mask], sample_weighting),
    )
    booster = lgb.train(
        params,
        refit_set,
        num_boost_round=selected_iterations,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    predictions = booster.predict(
        np.ascontiguousarray(features[test_mask], dtype=np.float32),
        num_iteration=selected_iterations,
    )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(args.model))
    np.save(args.predictions, np.asarray(predictions, dtype=np.float64))
    args.result.write_text(
        json.dumps(
            {
                "selected_iterations": selected_iterations,
                "feature_names": booster.feature_name(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def inspect(args: argparse.Namespace) -> None:
    _, metadata = _load_models(args.request)
    args.result.write_text(
        json.dumps({"models": metadata}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def predict(args: argparse.Namespace) -> None:
    features = np.ascontiguousarray(
        np.load(args.features, allow_pickle=False),
        dtype=np.float32,
    )
    if features.ndim != 2:
        raise ValueError("prediction features must be a two-dimensional matrix")
    boosters, metadata = _load_models(args.request)
    predictions = {
        target: np.asarray(booster.predict(features), dtype=np.float32)
        for target, booster in boosters.items()
    }
    np.savez(args.predictions, **predictions)
    args.result.write_text(
        json.dumps({"models": metadata}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean-process CNN LightGBM worker")
    subcommands = parser.add_subparsers(dest="command", required=True)

    fit_parser = subcommands.add_parser("fit")
    fit_parser.add_argument("--features", type=Path, required=True)
    fit_parser.add_argument("--target", type=Path, required=True)
    fit_parser.add_argument("--splits", type=Path, required=True)
    fit_parser.add_argument("--request", type=Path, required=True)
    fit_parser.add_argument("--model", type=Path, required=True)
    fit_parser.add_argument("--predictions", type=Path, required=True)
    fit_parser.add_argument("--result", type=Path, required=True)

    inspect_parser = subcommands.add_parser("inspect")
    inspect_parser.add_argument("--request", type=Path, required=True)
    inspect_parser.add_argument("--result", type=Path, required=True)

    predict_parser = subcommands.add_parser("predict")
    predict_parser.add_argument("--features", type=Path, required=True)
    predict_parser.add_argument("--request", type=Path, required=True)
    predict_parser.add_argument("--predictions", type=Path, required=True)
    predict_parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "fit":
        fit(args)
    elif args.command == "inspect":
        inspect(args)
    else:
        predict(args)


if __name__ == "__main__":
    main()
