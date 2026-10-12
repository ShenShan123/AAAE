from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .errors import ExecutionUnavailableError, TrainingError
from .manifests import (
    EnsembleMemberConfig,
    ExecutionConfig,
    SampleWeightingConfig,
    TargetEnsembleConfig,
    TrainingConfig,
)


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    p95_abs_error: float
    max_abs_error: float
    over_5pp_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class FittedMember:
    name: str
    booster: Any
    selected_iterations: int
    weight: float


@dataclass(frozen=True)
class FittedTarget:
    members: tuple[FittedMember, ...]
    offset: float
    metrics: RegressionMetrics


@dataclass(frozen=True)
class _SerializedBooster:
    model_text: str

    def save_model(self, filename: str | Path) -> None:
        Path(filename).write_text(self.model_text, encoding="utf-8")


@dataclass(frozen=True)
class ModelMetadata:
    feature_count: int
    feature_names: tuple[str, ...]


def lightgbm_version() -> str:
    try:
        return distribution_version("lightgbm")
    except PackageNotFoundError as exc:  # pragma: no cover - dependency is required in production.
        raise TrainingError("CNN training requires the lightgbm package") from exc


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    actual_values = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual_values.shape != predicted_values.shape or len(actual_values) == 0:
        raise ValueError("metrics require nonempty actual and predicted arrays with equal shape")
    error = predicted_values - actual_values
    absolute = np.abs(error)
    return RegressionMetrics(
        mae=float(np.mean(absolute)),
        rmse=float(math.sqrt(float(np.mean(error * error)))),
        p95_abs_error=float(np.quantile(absolute, 0.95)),
        max_abs_error=float(np.max(absolute)),
        over_5pp_count=int(np.sum(absolute > 0.05)),
    )


def _params(
    execution: ExecutionConfig,
    training: TrainingConfig,
    *,
    target_seed_offset: int,
) -> dict[str, Any]:
    seed = int(execution.seed) + int(target_seed_offset)
    params: dict[str, Any] = {
        "objective": training.objective,
        "metric": "None" if training.selection_metric == "max_abs_error" else "l1",
        "learning_rate": float(training.learning_rate),
        "num_leaves": int(training.num_leaves),
        "min_data_in_leaf": int(training.min_child_samples),
        "feature_fraction": float(training.feature_fraction),
        "bagging_fraction": float(training.bagging_fraction),
        "bagging_freq": 1 if training.bagging_fraction < 1.0 else 0,
        "lambda_l2": float(training.reg_lambda),
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "verbosity": -1,
        "force_col_wise": True,
        "device_type": execution.device,
        "num_threads": 1 if execution.device == "cuda" else int(execution.n_jobs),
    }
    if execution.device == "cuda":
        params["gpu_device_id"] = int(execution.gpu_id)
    return params


def _run_worker(arguments: list[str]) -> None:
    worker = Path(__file__).with_name("_lightgbm_worker.py")
    completed = subprocess.run(
        [sys.executable, str(worker), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no output"
        raise RuntimeError(
            "clean-process LightGBM worker failed with exit code "
            f"{completed.returncode}: {detail}"
        )


def _stage_models(root: Path, model_texts: Mapping[str, str]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for target, model_text in sorted(model_texts.items()):
        if not isinstance(target, str) or not target or not isinstance(model_text, str):
            raise RuntimeError("serialized LightGBM models must map names to model text")
        path = root / f"{target}.txt"
        path.write_text(model_text, encoding="utf-8")
        paths[target] = str(path)
    if not paths:
        raise RuntimeError("at least one serialized LightGBM model is required")
    return paths


def _parse_model_metadata(
    result: object,
    expected_targets: set[str],
) -> dict[str, ModelMetadata]:
    if not isinstance(result, dict) or not isinstance(result.get("models"), dict):
        raise RuntimeError("LightGBM worker result has invalid model metadata")
    models = result["models"]
    if set(models) != expected_targets:
        raise RuntimeError("LightGBM worker returned the wrong model targets")
    parsed: dict[str, ModelMetadata] = {}
    for target, raw in models.items():
        if not isinstance(raw, dict):
            raise RuntimeError(f"LightGBM worker metadata for {target} must be an object")
        count = raw.get("feature_count")
        names = raw.get("feature_names")
        if not isinstance(count, int) or count <= 0:
            raise RuntimeError(f"LightGBM worker metadata for {target} has invalid feature_count")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise RuntimeError(f"LightGBM worker metadata for {target} has invalid feature_names")
        parsed[target] = ModelMetadata(count, tuple(names))
    return parsed


def inspect_serialized_models(
    model_texts: Mapping[str, str],
) -> dict[str, ModelMetadata]:
    with tempfile.TemporaryDirectory(prefix="cnn-lightgbm-inspect-") as temporary_name:
        temporary = Path(temporary_name)
        request_path = temporary / "request.json"
        result_path = temporary / "result.json"
        model_paths = _stage_models(temporary, model_texts)
        request_path.write_text(
            json.dumps({"models": model_paths}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _run_worker(
            [
                "inspect",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
            ]
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LightGBM inspect worker produced invalid output: {exc}") from exc
    return _parse_model_metadata(result, set(model_texts))


def predict_serialized_models(
    model_texts: Mapping[str, str],
    features: np.ndarray,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    matrix = np.ascontiguousarray(features, dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="cnn-lightgbm-predict-") as temporary_name:
        temporary = Path(temporary_name)
        features_path = temporary / "features.npy"
        request_path = temporary / "request.json"
        predictions_path = temporary / "predictions.npz"
        result_path = temporary / "result.json"
        model_paths = _stage_models(temporary, model_texts)
        np.save(features_path, matrix)
        request_path.write_text(
            json.dumps({"models": model_paths}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _run_worker(
            [
                "predict",
                "--features",
                str(features_path),
                "--request",
                str(request_path),
                "--predictions",
                str(predictions_path),
                "--result",
                str(result_path),
            ]
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            with np.load(predictions_path, allow_pickle=False) as archive:
                if set(archive.files) != set(model_texts):
                    raise RuntimeError("LightGBM worker returned the wrong prediction targets")
                predictions = {
                    target: np.asarray(archive[target], dtype=np.float32).reshape(-1)
                    for target in model_texts
                }
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"LightGBM predict worker produced invalid output: {exc}") from exc
    metadata = _parse_model_metadata(result, set(model_texts))
    for target, model in metadata.items():
        if model.feature_count != len(feature_names) or model.feature_names != feature_names:
            raise RuntimeError(f"{target} model feature schema changed after bundle loading")
        if len(predictions[target]) != len(matrix):
            raise RuntimeError(f"{target} model returned the wrong number of predictions")
    return predictions


def _fit_in_clean_process(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    params: dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int,
    selection_metric: str,
    sample_weighting: SampleWeightingConfig,
) -> tuple[_SerializedBooster, int, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="cnn-lightgbm-") as temporary_name:
        temporary = Path(temporary_name)
        features_path = temporary / "features.npy"
        target_path = temporary / "target.npy"
        splits_path = temporary / "splits.npy"
        request_path = temporary / "request.json"
        model_path = temporary / "model.txt"
        predictions_path = temporary / "test_predictions.npy"
        result_path = temporary / "result.json"

        np.save(features_path, np.ascontiguousarray(features, dtype=np.float32))
        np.save(target_path, np.ascontiguousarray(target, dtype=np.float64))
        np.save(splits_path, np.asarray(splits).astype(str))
        request_path.write_text(
            json.dumps(
                {
                    "params": params,
                    "feature_names": feature_names,
                    "n_estimators": int(n_estimators),
                    "early_stopping_rounds": int(early_stopping_rounds),
                    "selection_metric": selection_metric,
                    "sample_weighting": asdict(sample_weighting),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _run_worker(
            [
                "fit",
                "--features",
                str(features_path),
                "--target",
                str(target_path),
                "--splits",
                str(splits_path),
                "--request",
                str(request_path),
                "--model",
                str(model_path),
                "--predictions",
                str(predictions_path),
                "--result",
                str(result_path),
            ]
        )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            model_text = model_path.read_text(encoding="utf-8")
            predictions = np.load(predictions_path, allow_pickle=False)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"LightGBM worker produced invalid artifacts: {exc}") from exc

    if not isinstance(result, dict):
        raise RuntimeError("LightGBM worker result must be a JSON object")
    selected_iterations = result.get("selected_iterations")
    if not isinstance(selected_iterations, int) or selected_iterations <= 0:
        raise RuntimeError("LightGBM worker returned an invalid selected iteration count")
    reported_names = result.get("feature_names")
    if reported_names != list(feature_names):
        raise RuntimeError("LightGBM worker model feature names do not match the frozen schema")
    model_names_line = next(
        (line for line in model_text.splitlines() if line.startswith("feature_names=")),
        None,
    )
    if model_names_line is None or tuple(
        model_names_line.removeprefix("feature_names=").split()
    ) != feature_names:
        raise RuntimeError("serialized LightGBM model feature names do not match the frozen schema")
    predicted_values = np.asarray(predictions, dtype=np.float64).reshape(-1)
    return _SerializedBooster(model_text), selected_iterations, predicted_values


def fit_target(
    features: np.ndarray,
    target: np.ndarray,
    splits: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    execution: ExecutionConfig,
    training: TrainingConfig,
    target_seed_offset: int,
    ensemble: TargetEnsembleConfig | None = None,
) -> FittedTarget:
    matrix = np.asarray(features, dtype=np.float32)
    values = np.asarray(target, dtype=np.float64).reshape(-1)
    split_values = np.asarray(splits).astype(str)
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise TrainingError("feature matrix and frozen feature schema do not align")
    if len(matrix) != len(values) or len(values) != len(split_values):
        raise TrainingError("features, labels, and split assignments have different row counts")
    if not np.isfinite(matrix).all() or not np.isfinite(values).all():
        raise TrainingError("features and targets must contain only finite values")

    test_mask = split_values == "test"
    if ensemble is None:
        ensemble = TargetEnsembleConfig(
            offset=0.0,
            members=(
                EnsembleMemberConfig(
                    name="model",
                    weight=1.0,
                    seed_offset=0,
                    training=training,
                    sample_weighting=SampleWeightingConfig(),
                ),
            ),
        )
    fitted_members: list[FittedMember] = []
    combined = np.full(int(np.sum(test_mask)), ensemble.offset, dtype=np.float64)
    try:
        for member in ensemble.members:
            params = _params(
                execution,
                member.training,
                target_seed_offset=target_seed_offset + member.seed_offset,
            )
            booster, selected_iterations, predicted = _fit_in_clean_process(
                matrix,
                values,
                split_values,
                feature_names=feature_names,
                params=params,
                n_estimators=member.training.n_estimators,
                early_stopping_rounds=member.training.early_stopping_rounds,
                selection_metric=member.training.selection_metric,
                sample_weighting=member.sample_weighting,
            )
            if len(predicted) != len(combined):
                raise RuntimeError("LightGBM worker returned the wrong number of test predictions")
            combined += member.weight * predicted
            fitted_members.append(
                FittedMember(
                    name=member.name,
                    booster=booster,
                    selected_iterations=selected_iterations,
                    weight=member.weight,
                )
            )
    except Exception as exc:
        if execution.device == "cuda":
            raise ExecutionUnavailableError(
                "CUDA LightGBM training was requested but could not run on "
                f"gpu_id={execution.gpu_id}; no CPU fallback was attempted: {exc}"
            ) from exc
        raise TrainingError(f"LightGBM training failed: {exc}") from exc
    return FittedTarget(
        members=tuple(fitted_members),
        offset=ensemble.offset,
        metrics=regression_metrics(values[test_mask], combined),
    )
