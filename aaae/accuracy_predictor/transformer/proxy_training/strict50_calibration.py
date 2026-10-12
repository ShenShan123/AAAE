#!/usr/bin/env python
"""Train and package strict-50 Transformer calibration on the retained population.

The default output layout is intentionally compact.  It writes only the final
test metrics, final test predictions, selected calibration images, run config,
and small audit summaries.  Candidate-model metrics, split assignments, and
row-level coverage manifests are debug artifacts and are written only when
``--write-debug-tables`` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _cuda_lgbm_requested_at_process_start(argv: list[str] | None = None) -> bool:
    tokens = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(tokens):
        if token == "--lgbm-device=cuda":
            return True
        if token == "--lgbm-device" and index + 1 < len(tokens) and tokens[index + 1] == "cuda":
            return True
    return False


LGBMRegressor: Any | None = None
_CUDA_LGBM_REQUESTED_AT_IMPORT = _cuda_lgbm_requested_at_process_start()
_LGBM_IMPORTED_EARLY_FOR_CUDA = False


def _load_lgbm_regressor() -> Any | None:
    global LGBMRegressor
    if LGBMRegressor is not None:
        return LGBMRegressor
    try:
        from lightgbm import LGBMRegressor as loaded_lgbm_regressor
    except ImportError:  # pragma: no cover - optional local acceleration.
        return None
    LGBMRegressor = loaded_lgbm_regressor
    return LGBMRegressor


if _CUDA_LGBM_REQUESTED_AT_IMPORT:
    # This CUDA-enabled wheel must load before sklearn/Scipy initialize their
    # native runtimes. Other imports stay lazy so feature-only and ONNX Runtime
    # processes do not initialize either training stack unnecessarily.
    _LGBM_IMPORTED_EARLY_FOR_CUDA = _load_lgbm_regressor() is not None


clone: Any | None = None
Pipeline: Any | None = None
FunctionTransformer: Any | None = None


def _load_sklearn_components() -> None:
    """Load sklearn only after LightGBM has initialized its native runtime."""

    global clone, Pipeline, FunctionTransformer
    if clone is not None:
        return
    from sklearn.base import clone as loaded_clone
    from sklearn.pipeline import Pipeline as loaded_pipeline
    from sklearn.preprocessing import FunctionTransformer as loaded_function_transformer

    clone = loaded_clone
    Pipeline = loaded_pipeline
    FunctionTransformer = loaded_function_transformer


def _cuda_named_feature_lgbm_regressor_class(base_lgbm_regressor: Any) -> Any:
    from aaae.accuracy_predictor.transformer.proxy_training.cuda_lightgbm import (
        CudaNamedFeatureLGBMRegressor,
    )

    if not issubclass(CudaNamedFeatureLGBMRegressor, base_lgbm_regressor):
        raise TypeError("CUDA LightGBM estimator does not match the loaded LightGBM package")
    return CudaNamedFeatureLGBMRegressor


def cuda_lgbm_matrix(values: Any) -> np.ndarray:
    """Materialize a row-major float32 matrix for CUDA LightGBM."""

    return np.ascontiguousarray(np.asarray(values, dtype=np.float32))


def fit_lightgbm_model(model: Any, features: np.ndarray, target: np.ndarray) -> Any:
    """Fit a calibration pipeline with deterministic unique feature names."""

    _load_sklearn_components()
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError(f"LightGBM features must be 2D, got {matrix.shape}")
    feature_names = [f"f{index:07d}" for index in range(matrix.shape[1])]
    if Pipeline is not None and isinstance(model, Pipeline) and "lgbmregressor" in model.named_steps:
        model.fit(matrix, target, lgbmregressor__feature_name=feature_names)
    else:
        model.fit(matrix, target, feature_name=feature_names)
    return model


def predict_lightgbm_model(model: Any, features: np.ndarray) -> np.ndarray:
    """Predict by position while suppressing only sklearn's known name warning."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        return np.asarray(model.predict(features))

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.inference_runtime.transformer.dataset_generation.profiles import (  # noqa: E402
    DATASET_PROFILES,
    DEFAULT_MODEL_KEY,
    get_dataset_profile,
)


DEFAULT_S8S8_STRICT50_DATASET = get_dataset_profile(
    DEFAULT_MODEL_KEY
).canonical_output_root
DEFAULT_INPUT_DATA_DIR = DEFAULT_S8S8_STRICT50_DATASET / "data"
DEFAULT_RUNTIME_BANK = DEFAULT_S8S8_STRICT50_DATASET / "data/runtime_bank.npz"
DEFAULT_BANK_META = DEFAULT_S8S8_STRICT50_DATASET / "shared/strict50_bank_meta.npz"
DEFAULT_OUTPUT_STAGES = {
    "swin_tiny_patch4_window7_224": (
        "swin_tiny_int8_s8s8_linear48_strict50_true1000_raw500_20k_50shot_v1"
    ),
    "vit_base_patch16_224": (
        "vit_base_patch16_224_int8_s8s8_linear48_strict50_true1000_raw500_20k_50shot_v1"
    ),
    "deit_base_patch16_224": (
        "deit_base_patch16_224_int8_s8s8_linear48_strict50_true1000_raw500_20k_50shot_v1"
    ),
}
TARGETS = ("top1_drop", "top5_drop")
STATIC_PRIOR_MODEL_NAME = "lgbm_static_l2"
GATE_MEAN = 0.02
GATE_MAX = 0.05
ENSEMBLE_SCORE_TOLERANCE = 0.001
COMPACT_DATA_FILES = {
    "calibration_images.csv",
    "final_metrics.csv",
    "final_predictions.csv",
    "missing_runtime_observations.csv",
    "output_manifest.json",
    "population_split_summary.csv",
    "run_config.json",
    "runtime_coverage_by_split.csv",
    "runtime_coverage_summary.csv",
    "selector_summary.csv",
}
LEGACY_DATA_FILES = {
    "candidate_calibration_images.csv",
    "candidate_metrics.csv",
    "candidate_predictions.csv",
    "population_duplicate_leakage.csv",
    "population_group_split_assignments.csv",
    "population_group_split_summary.csv",
    "population_static_metrics.csv",
    "population_static_predictions.csv",
    "runtime_bank_50shot_metrics.csv",
    "runtime_bank_50shot_predictions.csv",
    "runtime_bank_50shot_selected_images.csv",
    "runtime_bank_coverage_manifest.csv",
    "runtime_bank_coverage_split_summary.csv",
    "runtime_bank_coverage_summary.csv",
    "sample50_drop_summary.csv",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any


def lightgbm_training_metadata(
    *,
    lgbm_device: str,
    lgbm_gpu_id: int,
    n_jobs: int,
) -> dict[str, Any]:
    """Return the shared device contract for every learned 50-shot model."""

    device_type = str(lgbm_device)
    if device_type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported LightGBM device type: {device_type}")
    if int(lgbm_gpu_id) < 0:
        raise ValueError("lgbm_gpu_id must be non-negative")
    if int(n_jobs) < 1:
        raise ValueError("n_jobs must be positive")
    return {
        "device_type": device_type,
        "gpu_device_id": int(lgbm_gpu_id) if device_type == "cuda" else None,
        "n_jobs": 1 if device_type == "cuda" else int(n_jobs),
        "requested_n_jobs": int(n_jobs),
        "force_col_wise": True,
    }


def _lightgbm_estimator_class_and_kwargs(
    *,
    lgbm_device: str,
    lgbm_gpu_id: int,
    n_jobs: int,
) -> tuple[Any, dict[str, Any], int]:
    """Load LightGBM lazily for CPU, or validate the clean CUDA import path."""

    training = lightgbm_training_metadata(
        lgbm_device=lgbm_device,
        lgbm_gpu_id=lgbm_gpu_id,
        n_jobs=n_jobs,
    )
    base_lgbm_regressor = _load_lgbm_regressor()
    if base_lgbm_regressor is None:
        raise RuntimeError("50-shot calibration requires the lightgbm package")
    if training["device_type"] == "cuda" and not _LGBM_IMPORTED_EARLY_FOR_CUDA:
        raise RuntimeError(
            "CUDA LightGBM must be selected on the process command line before importing "
            "runtime_50shot_calibration; launch the calibration script with "
            "--lgbm-device=cuda."
        )
    _load_sklearn_components()

    estimator_class = (
        _cuda_named_feature_lgbm_regressor_class(base_lgbm_regressor)
        if training["device_type"] == "cuda"
        else base_lgbm_regressor
    )
    device_kwargs: dict[str, Any] = {"device_type": training["device_type"]}
    if training["gpu_device_id"] is not None:
        device_kwargs["gpu_device_id"] = training["gpu_device_id"]
    return estimator_class, device_kwargs, int(training["n_jobs"])


def lightgbm_model_spec(
    *,
    name: str,
    feature_count: int,
    seed: int,
    n_jobs: int,
    lgbm_device: str,
    lgbm_gpu_id: int,
    objective: str,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    colsample_bytree: float,
) -> ModelSpec:
    """Build one of the LightGBM-only learned candidates used by 50-shot."""

    if int(feature_count) < 1:
        raise ValueError("feature_count must be positive")
    estimator_class, device_kwargs, lgbm_n_jobs = _lightgbm_estimator_class_and_kwargs(
        lgbm_device=lgbm_device,
        lgbm_gpu_id=lgbm_gpu_id,
        n_jobs=n_jobs,
    )
    steps: list[tuple[str, Any]] = []
    if str(lgbm_device) == "cuda":
        steps.append(("cuda_matrix", FunctionTransformer(cuda_lgbm_matrix, validate=False)))
    steps.append(
        (
            "lgbmregressor",
            estimator_class(
                objective=objective,
                n_estimators=int(n_estimators),
                learning_rate=float(learning_rate),
                num_leaves=int(num_leaves),
                min_child_samples=int(min_child_samples),
                subsample=0.90,
                subsample_freq=1,
                colsample_bytree=float(colsample_bytree),
                reg_lambda=0.20,
                random_state=int(seed),
                n_jobs=lgbm_n_jobs,
                verbosity=-1,
                force_col_wise=True,
                **device_kwargs,
            ),
        )
    )
    return ModelSpec(name=name, estimator=Pipeline(steps=steps))


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _parse_output_stage(value: str) -> str:
    """Accept one safe run-directory name, never an arbitrary filesystem path."""

    stage = str(value).strip()
    path = Path(stage)
    if (
        not stage
        or stage in {".", ".."}
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != stage
        or "/" in stage
        or "\\" in stage
    ):
        raise argparse.ArgumentTypeError(
            "--output-stage must be one directory name below <output-root>/accuracy"
        )
    return stage


def _resolve_model_dataset_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Fill model-specific canonical inputs while preserving explicit paths."""

    profile = get_dataset_profile(str(args.model_key))
    dataset_root = profile.canonical_output_root
    if args.input_data_dir is None:
        args.input_data_dir = dataset_root / "data"
    if args.runtime_bank is None:
        args.runtime_bank = dataset_root / "data" / "runtime_bank.npz"
    if args.bank_meta is None:
        args.bank_meta = dataset_root / "shared" / "strict50_bank_meta.npz"
    if args.output_stage is None:
        args.output_stage = DEFAULT_OUTPUT_STAGES[profile.model_key]
    args.dataset_id = profile.dataset_id
    return args


def _resolve_output_stage_dir(project_root: Path, output_stage: str) -> Path:
    """Resolve a run output only after proving it remains beneath ``accuracy``."""

    stage = _parse_output_stage(output_stage)
    runs_root = (Path(project_root).resolve() / "accuracy").resolve()
    output_dir = (runs_root / stage).resolve()
    if output_dir.parent != runs_root:
        raise ValueError(
            f"--output-stage resolves outside the run root: {output_stage!r} -> {output_dir}"
        )
    return output_dir


def prepare_output_layout(output_dir: Path, *, write_debug_tables: bool, preserve_paths: list[Path]) -> None:
    data_dir = output_dir / "data"
    debug_dir = output_dir / "debug"
    preserved = {path.resolve() for path in preserve_paths}

    for path in [output_dir / "audit.md", *(data_dir / name for name in sorted(COMPACT_DATA_FILES | LEGACY_DATA_FILES))]:
        if path.resolve() in preserved or not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if debug_dir.exists():
        shutil.rmtree(debug_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    if write_debug_tables:
        debug_dir.mkdir(parents=True, exist_ok=True)


def write_output_manifest(path: Path, *, output_dir: Path) -> None:
    entries: list[dict[str, Any]] = []
    manifest_path = path.resolve()
    for item in sorted(output_dir.rglob("*")):
        if not item.is_file():
            continue
        if item.resolve() == manifest_path:
            continue
        entries.append(
            {
                "path": str(item.relative_to(output_dir)),
                "bytes": int(item.stat().st_size),
            }
        )
    write_json(path, {"files": entries})


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(empty)\n"
    view = frame.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.6g}")
        else:
            view[col] = view[col].map(str)
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in view.columns) + " |")
    return "\n".join(lines) + "\n"


def regression_metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float | bool | int]:
    err = np.asarray(pred, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    ae = np.abs(err)
    mae = float(ae.mean())
    max_abs = float(ae.max(initial=0.0))
    return {
        "mae": mae,
        "rmse": float(math.sqrt(float(np.mean(err * err)))),
        "bias_pred_minus_actual": float(err.mean()),
        "p95_abs_error": float(np.quantile(ae, 0.95)),
        "max_abs_error": max_abs,
        "over_5pp_count": int(np.sum(ae > GATE_MAX)),
        "gate_pass_mean": bool(mae <= GATE_MEAN),
        "gate_pass_max": bool(max_abs <= GATE_MAX),
        "gate_pass_both": bool(mae <= GATE_MEAN and max_abs <= GATE_MAX),
    }


def metric_score(metrics: dict[str, float | bool | int]) -> float:
    return float(
        float(metrics["mae"])
        + 0.20 * float(metrics["p95_abs_error"])
        + 0.35 * float(metrics["max_abs_error"])
        + 0.003 * int(metrics["over_5pp_count"])
    )


def resolve_aux_feature_path(base_dir: Path) -> Path:
    for name in ("common3103_aux.npy", "common3103_dassign19218.npy"):
        path = base_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"{base_dir} has neither common3103_aux.npy nor common3103_dassign19218.npy")


def try_load_merged_inputs(base_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    merged_features = base_dir / "common3103.npy"
    merged_targets = base_dir / "targets.npy"
    merged_index = base_dir / "merged_index.csv"
    if not (merged_features.exists() and merged_targets.exists() and merged_index.exists()):
        return None

    x = np.load(merged_features).astype(np.float32)
    y = np.load(merged_targets).astype(np.float32)
    index = pd.read_csv(merged_index, low_memory=False)
    if len(x) != len(y) or len(x) != len(index):
        raise ValueError(f"merged inputs at {base_dir} have mismatched row counts")
    runtime_map_path = base_dir / "runtime_design_id_map.csv"
    if runtime_map_path.exists():
        runtime_map = pd.read_csv(runtime_map_path, low_memory=False)
        required = {"row_index", "design_id", "runtime_design_id"}
        missing = required.difference(runtime_map.columns)
        if missing:
            raise KeyError(f"{runtime_map_path} is missing columns: {sorted(missing)}")
        index = index.merge(
            runtime_map[["row_index", "design_id", "runtime_design_id"]],
            on=["row_index", "design_id"],
            how="left",
            validate="one_to_one",
        )
        if index["runtime_design_id"].isna().any():
            raise ValueError(f"{runtime_map_path} does not map every merged row")
        runtime_design_id = index["runtime_design_id"].astype(str).to_numpy()
    else:
        runtime_design_id = index["design_id"].astype(str).to_numpy()
    identity = load_dataset_identity(base_dir)
    source_label = str(
        identity.get("dataset_id")
        or identity.get("model_key")
        or "merged"
    )
    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=np.int64),
            "design_id": index["design_id"].astype(str).to_numpy(),
            "runtime_design_id": runtime_design_id,
            "source": np.full(len(y), source_label, dtype=object),
            "top1_drop": y[:, 0],
            "top5_drop": y[:, 1],
        }
    )
    configuration_path = base_dir / "configuration_index.csv"
    if configuration_path.exists():
        configuration = pd.read_csv(
            configuration_path,
            usecols=["row_index", "design_id", "split", "evaluation_role"],
            low_memory=False,
        )
        if not np.array_equal(
            configuration["row_index"].to_numpy(np.int64),
            frame["row_index"].to_numpy(np.int64),
        ) or not np.array_equal(
            configuration["design_id"].astype(str).to_numpy(),
            frame["design_id"].astype(str).to_numpy(),
        ):
            raise ValueError(f"{configuration_path} is not aligned with merged inputs")
        frame["split"] = configuration["split"].astype(str).to_numpy()
        frame["evaluation_role"] = configuration["evaluation_role"].astype(str).to_numpy()
    frame["max_drop"] = frame[["top1_drop", "top5_drop"]].max(axis=1)
    return x, y, frame


RUNTIME_IDENTITY_KEYS = (
    "model_key",
    "dataset_id",
    "dataset_schema_version",
    "injection_scope_id",
    "injected_matmul_count",
    "model_sha1",
    "node_registry_sha1",
    "bank_meta_sha1",
)


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dataset_identity(data_dir: Path) -> dict[str, Any]:
    """Read the optional canonical dataset identity next to merged inputs."""

    manifest_path = Path(data_dir).resolve() / "manifest.json"
    if not manifest_path.is_file():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset manifest must contain one JSON object: {manifest_path}")
    identity: dict[str, Any] = {
        key: str(payload[key])
        for key in ("model_key", "dataset_id")
        if payload.get(key) not in (None, "")
    }
    if payload.get("schema_version") not in (None, ""):
        identity["dataset_schema_version"] = str(payload["schema_version"])
    injection_mode = payload.get("injection_mode")
    if isinstance(injection_mode, dict):
        if injection_mode.get("injection_scope_id") not in (None, ""):
            identity["injection_scope_id"] = str(
                injection_mode["injection_scope_id"]
            )
        if injection_mode.get("injected_matmul_count") is not None:
            identity["injected_matmul_count"] = int(
                injection_mode["injected_matmul_count"]
            )
    for key in ("model_sha1", "node_registry_sha1"):
        if payload.get(key) not in (None, ""):
            identity[key] = str(payload[key])
    runtime_protocol = payload.get("runtime_observation_protocol")
    if isinstance(runtime_protocol, dict) and runtime_protocol.get(
        "bank_meta_sha1"
    ) not in (None, ""):
        identity["bank_meta_sha1"] = str(runtime_protocol["bank_meta_sha1"])
    return identity


def validate_dataset_identity(
    identity: dict[str, Any],
    *,
    model_key: str,
    dataset_id: str,
) -> dict[str, Any] | None:
    """Validate a canonical dataset and return its deployable runtime identity."""

    actual_model_key = identity.get("model_key")
    if actual_model_key is not None and actual_model_key != model_key:
        raise ValueError(
            f"--model-key={model_key} does not match dataset model_key={actual_model_key}"
        )
    actual_dataset_id = identity.get("dataset_id")
    if actual_dataset_id is not None and actual_dataset_id != dataset_id:
        raise ValueError(
            f"selected profile dataset_id={dataset_id} does not match "
            f"dataset manifest dataset_id={actual_dataset_id}"
        )
    if not identity:
        # Explicit synthetic/test inputs predate canonical identity manifests.
        return None

    missing = [key for key in RUNTIME_IDENTITY_KEYS if key not in identity]
    if missing:
        raise ValueError(
            "canonical dataset manifest is missing runtime identity fields: "
            f"{missing}"
        )
    profile = get_dataset_profile(model_key)
    expected = {
        "model_key": profile.model_key,
        "dataset_id": profile.dataset_id,
        "dataset_schema_version": str(identity["dataset_schema_version"]),
        "injection_scope_id": profile.injection_scope_id,
        "injected_matmul_count": int(profile.expected_injected_matmuls),
        "model_sha1": _sha1_file(profile.dynamic_model_path),
        "node_registry_sha1": _sha1_file(profile.node_registry_path),
        "bank_meta_sha1": _sha1_file(
            profile.canonical_output_root / "shared" / "strict50_bank_meta.npz"
        ),
    }
    mismatches = {
        key: {"dataset": identity[key], "expected": value}
        for key, value in expected.items()
        if identity[key] != value
    }
    if mismatches:
        raise ValueError(
            "canonical dataset runtime identity does not match its model profile: "
            f"{mismatches}"
        )
    return expected


def load_input_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    base_dir = Path(data_dir).resolve()
    merged = try_load_merged_inputs(base_dir)
    if merged is not None:
        return merged
    meta = np.load(base_dir / "common3103_meta.npz", allow_pickle=True)
    x_cur = np.load(base_dir / "common3103_current.npy").astype(np.float32)
    x_aux = np.load(resolve_aux_feature_path(base_dir)).astype(np.float32)
    y_cur = np.asarray(meta["ycur"], dtype=np.float32)
    y_aux = np.asarray(meta["yaux"], dtype=np.float32)
    cur_ids = np.asarray(meta["cur_ids"]).astype(str)
    aux_ids = np.asarray(meta["aux_ids"]).astype(str)
    cur_sources = (
        np.asarray(meta["cur_sources"]).astype(str)
        if "cur_sources" in meta.files
        else np.full(len(cur_ids), "current512", dtype=object)
    )
    aux_sources = (
        np.asarray(meta["aux_sources"]).astype(str)
        if "aux_sources" in meta.files
        else np.full(len(aux_ids), "aux", dtype=object)
    )
    if not (len(x_cur) == len(y_cur) == len(cur_ids) == len(cur_sources)):
        raise ValueError("current feature/meta row counts do not match")
    if not (len(x_aux) == len(y_aux) == len(aux_ids) == len(aux_sources)):
        raise ValueError("aux feature/meta row counts do not match")

    x = np.vstack([x_cur, x_aux]).astype(np.float32)
    y = np.vstack([y_cur, y_aux]).astype(np.float32)
    frame = pd.DataFrame(
        {
            "row_index": np.arange(len(y), dtype=np.int64),
            "design_id": np.concatenate([cur_ids, aux_ids]),
            "source": np.concatenate([cur_sources.astype(object), aux_sources.astype(object)]),
            "top1_drop": y[:, 0],
            "top5_drop": y[:, 1],
        }
    )
    frame["max_drop"] = frame[["top1_drop", "top5_drop"]].max(axis=1)
    return x, y, frame


def assign_group_splits(frame: pd.DataFrame, seed: int, train_frac: float, val_frac: float) -> pd.DataFrame:
    """Assign splits by design_id, stratified by source/drop bins at group level."""

    if "split" in frame.columns and frame["split"].notna().all():
        out = frame.copy()
        valid_splits = {"train", "val", "test"}
        actual_splits = set(out["split"].astype(str))
        if actual_splits != valid_splits:
            raise ValueError(f"fixed dataset splits must be {sorted(valid_splits)}, got {sorted(actual_splits)}")
        leakage = out.groupby("design_id", sort=False)["split"].nunique()
        if leakage.gt(1).any():
            raise ValueError("fixed dataset split assigns one design to multiple partitions")
        out["group_primary_source"] = out["source"].astype(str)
        out["group_drop_bin"] = -1
        return out

    if frame["design_id"].is_unique:
        rng = np.random.default_rng(seed)
        group = frame[["row_index", "design_id", "source", "max_drop"]].copy().reset_index(drop=True)
        group["drop_bin"] = -1
        for source, idx in group.groupby("source").groups.items():
            indices = list(idx)
            ranks = group.loc[indices, "max_drop"].rank(method="first")
            bins = min(10, max(1, len(indices) // 20))
            if bins > 1:
                group.loc[indices, "drop_bin"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop").astype(int).to_numpy()
            else:
                group.loc[indices, "drop_bin"] = 0

        assignment = pd.Series(index=group.index, dtype=object)
        for _, sub in group.groupby(["source", "drop_bin"], sort=False):
            indices = sub.index.to_numpy(np.int64)
            rng.shuffle(indices)
            n = len(indices)
            n_train = int(math.floor(n * train_frac))
            n_val = int(math.floor(n * val_frac))
            n_train = min(max(n_train, 0), n)
            n_val = min(max(n_val, 0), max(0, n - n_train))
            assignment.loc[indices[:n_train]] = "train"
            assignment.loc[indices[n_train : n_train + n_val]] = "val"
            assignment.loc[indices[n_train + n_val :]] = "test"

        desired_train = int(math.floor(len(group) * train_frac))
        desired_val = int(math.floor(len(group) * val_frac))
        desired = {
            "train": desired_train,
            "val": desired_val,
            "test": int(len(group) - desired_train - desired_val),
        }
        scores = group["max_drop"].to_numpy(np.float64)

        def move_one(src: str, dst: str) -> bool:
            src_idx = np.flatnonzero(assignment.to_numpy(str) == src)
            if len(src_idx) == 0:
                return False
            dst_scores = scores[np.flatnonzero(assignment.to_numpy(str) == dst)]
            target_score = float(np.median(dst_scores)) if len(dst_scores) else float(np.median(scores))
            chosen = int(src_idx[np.argmin(np.abs(scores[src_idx] - target_score))])
            assignment.iloc[chosen] = dst
            return True

        for _ in range(len(group)):
            counts = assignment.value_counts().to_dict()
            surplus = [name for name in ("train", "val", "test") if int(counts.get(name, 0)) > desired[name]]
            deficit = [name for name in ("train", "val", "test") if int(counts.get(name, 0)) < desired[name]]
            if not surplus and not deficit:
                break
            if not surplus or not deficit:
                raise RuntimeError("failed to balance split counts")
            if not move_one(surplus[0], deficit[0]):
                raise RuntimeError("failed to move split row while balancing")

        out = frame.copy()
        out["group_primary_source"] = out["source"].astype(str)
        out["group_drop_bin"] = group["drop_bin"].to_numpy(np.int64)
        out["split"] = assignment.to_numpy(str)
        return out

    rng = np.random.default_rng(seed)
    group = (
        frame.groupby("design_id", as_index=False)
        .agg(
            primary_source=("source", lambda values: "|".join(sorted(set(map(str, values))))),
            max_drop=("max_drop", "max"),
            row_count=("row_index", "count"),
        )
        .reset_index(drop=True)
    )
    group["drop_bin"] = -1
    for source, idx in group.groupby("primary_source").groups.items():
        indices = list(idx)
        ranks = group.loc[indices, "max_drop"].rank(method="first")
        bins = min(10, max(1, len(indices) // 20))
        if bins > 1:
            group.loc[indices, "drop_bin"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop").astype(int).to_numpy()
        else:
            group.loc[indices, "drop_bin"] = 0

    split_lookup: dict[str, str] = {}
    for _, sub in group.groupby(["primary_source", "drop_bin"], sort=False):
        ids = sub["design_id"].to_numpy(str)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(max(n_train, 1 if n >= 3 else n), n)
        n_val = min(max(n_val, 1 if n - n_train >= 2 else 0), max(0, n - n_train))
        for item in ids[:n_train]:
            split_lookup[str(item)] = "train"
        for item in ids[n_train : n_train + n_val]:
            split_lookup[str(item)] = "val"
        for item in ids[n_train + n_val :]:
            split_lookup[str(item)] = "test"

    out = frame.copy()
    out["group_primary_source"] = out["design_id"].map(dict(zip(group["design_id"], group["primary_source"], strict=False)))
    out["group_drop_bin"] = out["design_id"].map(dict(zip(group["design_id"], group["drop_bin"], strict=False))).astype(int)
    out["split"] = out["design_id"].map(split_lookup)
    if out["split"].isna().any():
        raise RuntimeError("split assignment failed for some rows")
    return out


def static_model_specs(
    seed: int,
    feature_count: int,
    n_jobs: int,
    quick: bool,
    *,
    lgbm_device: str = "cpu",
    lgbm_gpu_id: int = 0,
) -> list[ModelSpec]:
    """Population-level LightGBM candidates used for the static audit and prior."""

    return [
        lightgbm_model_spec(
            name=STATIC_PRIOR_MODEL_NAME,
            feature_count=feature_count,
            seed=seed + 1,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression",
            n_estimators=300 if quick else 520,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=50,
            colsample_bytree=0.75,
        ),
        lightgbm_model_spec(
            name="lgbm_static_l1",
            feature_count=feature_count,
            seed=seed + 2,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression_l1",
            n_estimators=240 if quick else 480,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.75,
        ),
        lightgbm_model_spec(
            name="lgbm_static_l2_wide",
            feature_count=feature_count,
            seed=seed + 3,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression",
            n_estimators=260 if quick else 480,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=40,
            colsample_bytree=0.85,
        ),
    ]


def audit_static_split(
    x: np.ndarray,
    y: np.ndarray,
    split_frame: pd.DataFrame,
    *,
    seed: int,
    n_jobs: int,
    quick: bool,
    lgbm_device: str,
    lgbm_gpu_id: int,
    collect_predictions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = static_model_specs(
        seed,
        x.shape[1],
        n_jobs,
        quick,
        lgbm_device=lgbm_device,
        lgbm_gpu_id=lgbm_gpu_id,
    )
    train_mask = split_frame["split"].eq("train").to_numpy()
    val_mask = split_frame["split"].eq("val").to_numpy()
    test_mask = split_frame["split"].eq("test").to_numpy()
    rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []

    for target_idx, target in enumerate(TARGETS):
        for spec in specs:
            model = clone(spec.estimator)
            fit_lightgbm_model(model, x[train_mask], y[train_mask, target_idx])
            for split_name, mask in (("val", val_mask), ("test", test_mask)):
                pred = np.clip(
                    np.asarray(predict_lightgbm_model(model, x[mask]), dtype=np.float64),
                    -0.05,
                    1.0,
                )
                metrics = regression_metrics(y[mask, target_idx], pred)
                rows.append(
                    {
                        "family": "population_group_static",
                        "target": target,
                        "model": spec.name,
                        "split": split_name,
                        "sample_count": int(mask.sum()),
                        **metrics,
                        "selection_score": metric_score(metrics),
                    }
                )
                if collect_predictions:
                    pred_rows.append(
                        pd.DataFrame(
                            {
                                "row_index": split_frame.loc[mask, "row_index"].to_numpy(),
                                "design_id": split_frame.loc[mask, "design_id"].to_numpy(),
                                "source": split_frame.loc[mask, "source"].to_numpy(),
                                "split": split_name,
                                "target": target,
                                "model": spec.name,
                                "actual": y[mask, target_idx],
                                "pred": pred,
                                "abs_error": np.abs(pred - y[mask, target_idx]),
                            }
                        )
                    )
    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


def load_runtime_bank(
    runtime_bank_path: Path,
    extra_runtime_banks: list[Path],
    *,
    image_count: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    banks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    z = np.load(runtime_bank_path, allow_pickle=True)
    banks.append(
        (
            "combined_runtime_bank",
            np.asarray(z["design_id"]).astype(str),
            np.asarray(z["approx_top1_correct_bits"], dtype=np.float32)[:, :image_count],
            np.asarray(z["approx_top5_correct_bits"], dtype=np.float32)[:, :image_count],
            np.asarray(z["delta_margin_top1"], dtype=np.float32)[:, :image_count],
            np.asarray(z["delta_margin_top5"], dtype=np.float32)[:, :image_count],
        )
    )

    for extra_path in extra_runtime_banks:
        extra = np.load(extra_path, allow_pickle=True)
        banks.append(
            (
                f"extra_runtime_bank:{extra_path.name}",
                np.asarray(extra["design_id"]).astype(str),
                np.asarray(extra["approx_top1_correct_bits"], dtype=np.float32)[:, :image_count],
                np.asarray(extra["approx_top5_correct_bits"], dtype=np.float32)[:, :image_count],
                np.asarray(extra["delta_margin_top1"], dtype=np.float32)[:, :image_count],
                np.asarray(extra["delta_margin_top5"], dtype=np.float32)[:, :image_count],
            )
        )

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    a1_rows: list[np.ndarray] = []
    a5_rows: list[np.ndarray] = []
    dm1_rows: list[np.ndarray] = []
    dm5_rows: list[np.ndarray] = []
    for bank_name, ids, a1, a5, dm1, dm5 in banks:
        for local_idx, design_id in enumerate(ids.tolist()):
            design = str(design_id)
            if design in seen:
                continue
            seen.add(design)
            rows.append({"design_id": design, "runtime_bank_source": bank_name, "runtime_bank_local_index": int(local_idx)})
            a1_rows.append(a1[local_idx].astype(np.float32, copy=False))
            a5_rows.append(a5[local_idx].astype(np.float32, copy=False))
            dm1_rows.append(dm1[local_idx].astype(np.float32, copy=False))
            dm5_rows.append(dm5[local_idx].astype(np.float32, copy=False))

    if not rows:
        raise RuntimeError("no runtime bank rows loaded")
    return (
        pd.DataFrame(rows),
        np.stack(a1_rows, axis=0).astype(np.float32),
        np.stack(a5_rows, axis=0).astype(np.float32),
        np.stack(dm1_rows, axis=0).astype(np.float32),
        np.stack(dm5_rows, axis=0).astype(np.float32),
    )


def abs_corr_scores(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    yy = y[mask].astype(np.float64)
    yy = yy - yy.mean()
    y_norm = np.linalg.norm(yy)
    if y_norm <= 1e-12:
        return np.zeros(x.shape[1], dtype=np.float64)
    sub = x[mask].astype(np.float64)
    sub = sub - sub.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(sub, axis=0) * y_norm
    score = np.zeros(x.shape[1], dtype=np.float64)
    valid = denom > 1e-12
    score[valid] = np.abs(sub[:, valid].T @ yy / denom[valid])
    return score


def fill_top_unique(parts: list[np.ndarray], limit: int, fallback: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for part in [*parts, fallback]:
        for value in part:
            item = int(value)
            if item in seen:
                continue
            selected.append(item)
            seen.add(item)
            if len(selected) >= limit:
                return np.asarray(selected, dtype=np.int64)
    raise RuntimeError("unable to fill image selection")


def load_fixed_selected_images(path: Path | None) -> tuple[str, np.ndarray] | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"fixed selected-image file is empty: {path}")
    if "selector" in frame.columns:
        selector_name = str(frame["selector"].iloc[0])
        frame = frame[frame["selector"].astype(str).eq(selector_name)].copy()
    else:
        selector_name = "fixed50"
    for col in ("image_index_in_runtime_bank", "image_index"):
        if col in frame.columns:
            selected = frame[col].to_numpy(np.int64)
            break
    else:
        raise KeyError(f"{path} has neither image_index_in_runtime_bank nor image_index")
    out: list[int] = []
    seen: set[int] = set()
    for value in selected.tolist():
        item = int(value)
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    if not out:
        raise ValueError(f"no selected images found in {path}")
    return selector_name, np.asarray(out, dtype=np.int64)


def greedy_representative_images(
    *,
    sample_count: int,
    train_mask: np.ndarray,
    clean_top1: np.ndarray,
    clean_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    actual_top1: np.ndarray,
    actual_top5: np.ndarray,
    candidate_order: np.ndarray,
    max_candidates: int,
    max_weight: float,
    top1_weight: float = 1.0,
    top5_weight: float = 1.0,
    p95_weight: float = 0.20,
) -> np.ndarray:
    train_idx = np.flatnonzero(train_mask)
    flip1 = clean_top1.reshape(1, -1) - approx_top1
    flip5 = clean_top5.reshape(1, -1) - approx_top5
    f1 = flip1[train_idx].astype(np.float32)
    f5 = flip5[train_idx].astype(np.float32)
    y1 = actual_top1[train_idx].astype(np.float32)
    y5 = actual_top5[train_idx].astype(np.float32)
    order = [int(item) for item in candidate_order[: min(max_candidates, len(candidate_order))]]
    selected: list[int] = []
    selected_set: set[int] = set()
    sum1 = np.zeros(len(train_idx), dtype=np.float32)
    sum5 = np.zeros(len(train_idx), dtype=np.float32)

    for step in range(sample_count):
        remaining = np.asarray([item for item in order if item not in selected_set], dtype=np.int64)
        if remaining.size == 0:
            remaining = np.asarray([idx for idx in range(f1.shape[1]) if idx not in selected_set], dtype=np.int64)
        denom = float(step + 1)
        pred1 = (sum1[:, None] + f1[:, remaining]) / denom
        pred5 = (sum5[:, None] + f5[:, remaining]) / denom
        err1 = np.abs(pred1 - y1[:, None])
        err5 = np.abs(pred5 - y5[:, None])
        score = (
            top1_weight * err1.mean(axis=0)
            + top5_weight * err5.mean(axis=0)
            + p95_weight * (top1_weight * np.quantile(err1, 0.95, axis=0) + top5_weight * np.quantile(err5, 0.95, axis=0))
            + max_weight * (top1_weight * err1.max(axis=0) + top5_weight * err5.max(axis=0))
        )
        best = int(remaining[int(np.argmin(score))])
        selected.append(best)
        selected_set.add(best)
        sum1 += f1[:, best]
        sum5 += f5[:, best]
    return np.asarray(selected, dtype=np.int64)


def select_images(
    *,
    selector: str,
    sample_count: int,
    train_mask: np.ndarray,
    clean_top1: np.ndarray,
    clean_top5: np.ndarray,
    clean_margin_top1: np.ndarray,
    clean_margin_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    actual_top1: np.ndarray,
    actual_top5: np.ndarray,
    prior_top1: np.ndarray,
    prior_top5: np.ndarray,
    seed: int,
) -> np.ndarray:
    flip1 = clean_top1.reshape(1, -1) - approx_top1
    flip5 = clean_top5.reshape(1, -1) - approx_top5
    variance = np.var(flip1[train_mask], axis=0) + np.var(flip5[train_mask], axis=0)
    corr_actual = abs_corr_scores(flip1, actual_top1, train_mask) + abs_corr_scores(flip5, actual_top5, train_mask)
    corr_resid = abs_corr_scores(flip1, actual_top1 - prior_top1, train_mask) + abs_corr_scores(flip5, actual_top5 - prior_top5, train_mask)
    variance_top5 = np.var(flip5[train_mask], axis=0)
    corr_actual_top5 = abs_corr_scores(flip5, actual_top5, train_mask)
    corr_resid_top5 = abs_corr_scores(flip5, actual_top5 - prior_top5, train_mask)
    hard_margin = 1.0 / np.maximum(np.abs(clean_margin_top1), 0.02) + 1.0 / np.maximum(np.abs(clean_margin_top5), 0.02)
    fallback = np.argsort(-(corr_resid + corr_actual + variance))
    fallback_top5 = np.argsort(-(corr_resid_top5 + corr_actual_top5 + variance_top5))
    rng = np.random.default_rng(seed)

    if selector == "residual_corr50":
        return np.argsort(-corr_resid)[:sample_count].astype(np.int64)
    if selector == "variance50":
        return np.argsort(-variance)[:sample_count].astype(np.int64)
    if selector == "mixed50":
        return fill_top_unique(
            [
                np.argsort(-corr_resid)[: sample_count // 2],
                np.argsort(-variance)[: sample_count // 3],
                np.argsort(-hard_margin)[:sample_count],
            ],
            sample_count,
            fallback,
        )
    if selector == "random50":
        return rng.choice(np.arange(len(clean_top1)), size=sample_count, replace=False).astype(np.int64)
    if selector == "greedy_drop50":
        return greedy_representative_images(
            sample_count=sample_count,
            train_mask=train_mask,
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            actual_top1=actual_top1,
            actual_top5=actual_top5,
            candidate_order=fallback,
            max_candidates=350,
            max_weight=0.05,
        )
    if selector == "greedy_drop50_wmax":
        return greedy_representative_images(
            sample_count=sample_count,
            train_mask=train_mask,
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            actual_top1=actual_top1,
            actual_top5=actual_top5,
            candidate_order=fallback,
            max_candidates=500,
            max_weight=0.20,
        )
    if selector == "greedy_top5_50_wmax":
        return greedy_representative_images(
            sample_count=sample_count,
            train_mask=train_mask,
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            actual_top1=actual_top1,
            actual_top5=actual_top5,
            candidate_order=fallback_top5,
            max_candidates=500,
            max_weight=0.30,
            top1_weight=0.20,
            top5_weight=1.50,
        )
    if selector == "greedy_top5_50_ultramax":
        return greedy_representative_images(
            sample_count=sample_count,
            train_mask=train_mask,
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            actual_top1=actual_top1,
            actual_top5=actual_top5,
            candidate_order=fallback_top5,
            max_candidates=500,
            max_weight=0.75,
            top1_weight=0.10,
            top5_weight=2.00,
            p95_weight=0.35,
        )
    if selector == "greedy_joint50_ultramax":
        return greedy_representative_images(
            sample_count=sample_count,
            train_mask=train_mask,
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            actual_top1=actual_top1,
            actual_top5=actual_top5,
            candidate_order=fallback,
            max_candidates=500,
            max_weight=0.60,
            top1_weight=1.00,
            top5_weight=1.35,
            p95_weight=0.30,
        )
    raise ValueError(f"unknown selector: {selector}")


def quantile_features(arr: np.ndarray, prefix: str, parts: list[np.ndarray], names: list[str]) -> None:
    qs = (0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0)
    parts.append(np.quantile(arr, qs, axis=1).T.astype(np.float32))
    names.extend([f"{prefix}_q{q:g}" for q in qs])


def build_50shot_features(
    *,
    base_x: np.ndarray,
    selected_images: np.ndarray,
    clean_top1: np.ndarray,
    clean_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    delta_margin_top1: np.ndarray,
    delta_margin_top5: np.ndarray,
    prior_top1: np.ndarray,
    prior_top5: np.ndarray,
    include_individual: bool,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    sel = selected_images.astype(np.int64)
    return build_50shot_features_from_selected(
        base_x=base_x,
        image_indices=sel,
        clean_top1_selected=np.asarray(clean_top1)[sel],
        clean_top5_selected=np.asarray(clean_top5)[sel],
        approx_top1_selected=np.asarray(approx_top1)[:, sel],
        approx_top5_selected=np.asarray(approx_top5)[:, sel],
        delta_margin_top1_selected=np.asarray(delta_margin_top1)[:, sel],
        delta_margin_top5_selected=np.asarray(delta_margin_top5)[:, sel],
        prior_top1=prior_top1,
        prior_top5=prior_top5,
        include_individual=include_individual,
    )


def build_50shot_features_from_selected(
    *,
    base_x: np.ndarray,
    image_indices: np.ndarray,
    clean_top1_selected: np.ndarray,
    clean_top5_selected: np.ndarray,
    approx_top1_selected: np.ndarray,
    approx_top5_selected: np.ndarray,
    delta_margin_top1_selected: np.ndarray,
    delta_margin_top5_selected: np.ndarray,
    prior_top1: np.ndarray,
    prior_top5: np.ndarray,
    include_individual: bool,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Build the canonical hybrid features from already-selected observations."""

    image_indices = np.asarray(image_indices, dtype=np.int64)
    clean_top1_selected = np.asarray(clean_top1_selected, dtype=np.float32)
    clean_top5_selected = np.asarray(clean_top5_selected, dtype=np.float32)
    a1 = np.asarray(approx_top1_selected, dtype=np.float32)
    a5 = np.asarray(approx_top5_selected, dtype=np.float32)
    dm1 = np.asarray(delta_margin_top1_selected, dtype=np.float32)
    dm5 = np.asarray(delta_margin_top5_selected, dtype=np.float32)
    base_x = np.asarray(base_x, dtype=np.float32)
    prior_top1 = np.asarray(prior_top1, dtype=np.float32).reshape(-1)
    prior_top5 = np.asarray(prior_top5, dtype=np.float32).reshape(-1)

    shot_count = int(image_indices.size)
    row_count = int(base_x.shape[0])
    if image_indices.ndim != 1 or shot_count == 0 or len(np.unique(image_indices)) != shot_count:
        raise ValueError("image_indices must be a non-empty 1D array of unique indices")
    if clean_top1_selected.shape != (shot_count,) or clean_top5_selected.shape != (shot_count,):
        raise ValueError("selected clean correctness arrays do not match image_indices")
    for name, value in {
        "approx_top1_selected": a1,
        "approx_top5_selected": a5,
        "delta_margin_top1_selected": dm1,
        "delta_margin_top5_selected": dm5,
    }.items():
        if value.shape != (row_count, shot_count):
            raise ValueError(f"{name} has shape {value.shape}; expected {(row_count, shot_count)}")
    if prior_top1.shape != (row_count,) or prior_top5.shape != (row_count,):
        raise ValueError("prior arrays must contain one value per feature row")

    flip1 = clean_top1_selected.reshape(1, -1) - a1
    flip5 = clean_top5_selected.reshape(1, -1) - a5

    sample_top1_drop = float(clean_top1_selected.mean()) - a1.mean(axis=1)
    sample_top5_drop = float(clean_top5_selected.mean()) - a5.mean(axis=1)
    sample_summary = pd.DataFrame(
        {
            "sample50_top1_drop": sample_top1_drop,
            "sample50_top5_drop": sample_top5_drop,
            "sample50_clean_top1_acc": float(clean_top1_selected.mean()),
            "sample50_clean_top5_acc": float(clean_top5_selected.mean()),
            "sample50_approx_top1_acc": a1.mean(axis=1),
            "sample50_approx_top5_acc": a5.mean(axis=1),
        }
    )

    parts: list[np.ndarray] = [np.nan_to_num(base_x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)]
    names: list[str] = [f"common3103_{idx}" for idx in range(base_x.shape[1])]
    for name, value in (
        ("sample50_top1_drop", sample_top1_drop),
        ("sample50_top5_drop", sample_top5_drop),
        ("sample50_top1_minus_prior", sample_top1_drop - prior_top1),
        ("sample50_top5_minus_prior", sample_top5_drop - prior_top5),
    ):
        parts.append(value.reshape(-1, 1).astype(np.float32))
        names.append(name)

    for arr, prefix in ((flip1, "flip1"), (flip5, "flip5"), (dm1, "dm1"), (dm5, "dm5")):
        stats = np.column_stack(
            [
                arr.mean(axis=1),
                arr.std(axis=1),
                np.abs(arr).mean(axis=1),
                np.maximum(arr, 0.0).mean(axis=1),
                np.maximum(-arr, 0.0).mean(axis=1),
            ]
        ).astype(np.float32)
        parts.append(stats)
        names.extend([f"{prefix}_{item}" for item in ("mean", "std", "absmean", "posmean", "negmean")])
        quantile_features(arr, prefix, parts, names)

    if include_individual:
        for arr, prefix in ((a1, "a1bit"), (a5, "a5bit"), (flip1, "flip1_img"), (flip5, "flip5_img"), (dm1, "dm1_img"), (dm5, "dm5_img")):
            parts.append(arr.astype(np.float32))
            names.extend([f"{prefix}_{int(image_idx)}" for image_idx in image_indices])

    x = np.concatenate(parts, axis=1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names, sample_summary


def hybrid_model_specs(
    seed: int,
    feature_count: int,
    n_jobs: int,
    quick: bool,
    *,
    lgbm_device: str = "cpu",
    lgbm_gpu_id: int = 0,
) -> list[ModelSpec]:
    return [
        lightgbm_model_spec(
            name="lgbm_l2_c50_n300",
            feature_count=feature_count,
            seed=seed + 68,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression",
            n_estimators=300 if quick else 520,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=50,
            colsample_bytree=0.75,
        ),
        lightgbm_model_spec(
            name="lgbm_l1_c30_n240",
            feature_count=feature_count,
            seed=seed + 38,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression_l1",
            n_estimators=240 if quick else 520,
            learning_rate=0.035,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.75,
        ),
        lightgbm_model_spec(
            name="lgbm_regression_15_40_320_0.04_0.85_98",
            feature_count=feature_count,
            seed=seed + 98,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression",
            n_estimators=320,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=40,
            colsample_bytree=0.85,
        ),
        lightgbm_model_spec(
            name="lgbm_regression_63_30_220_0.035_0.75_58",
            feature_count=feature_count,
            seed=seed + 58,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression",
            n_estimators=220,
            learning_rate=0.035,
            num_leaves=63,
            min_child_samples=30,
            colsample_bytree=0.75,
        ),
        lightgbm_model_spec(
            name="lgbm_regression_l1_63_50_260_0.03_0.65_108",
            feature_count=feature_count,
            seed=seed + 108,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression_l1",
            n_estimators=260,
            learning_rate=0.03,
            num_leaves=63,
            min_child_samples=50,
            colsample_bytree=0.65,
        ),
        lightgbm_model_spec(
            name="lgbm_regression_l1_47_40_300_0.03_0.8_128",
            feature_count=feature_count,
            seed=seed + 128,
            n_jobs=n_jobs,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
            objective="regression_l1",
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=47,
            min_child_samples=40,
            colsample_bytree=0.8,
        ),
    ]


def fit_predict_model(spec: ModelSpec, x: np.ndarray, y: np.ndarray, train_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    model = clone(spec.estimator)
    fit_lightgbm_model(model, x[train_mask], y[train_mask])
    return np.clip(
        np.asarray(predict_lightgbm_model(model, x[pred_mask]), dtype=np.float64),
        -0.05,
        1.0,
    )


def append_metric_row(
    rows: list[dict[str, Any]],
    *,
    family: str,
    selector: str,
    target: str,
    model: str,
    fit_scope: str,
    split: str,
    sample_count: int,
    calibration_image_count: int,
    actual: np.ndarray,
    pred: np.ndarray,
) -> None:
    metrics = regression_metrics(actual, pred)
    rows.append(
        {
            "family": family,
            "selector": selector,
            "target": target,
            "model": model,
            "fit_scope": fit_scope,
            "split": split,
            "sample_count": int(sample_count),
            "calibration_image_count": int(calibration_image_count),
            **metrics,
            "selection_score": metric_score(metrics),
        }
    )


def prediction_frame(
    *,
    frame: pd.DataFrame,
    mask: np.ndarray,
    selector: str,
    target: str,
    model: str,
    fit_scope: str,
    split: str,
    y: np.ndarray,
    prior: np.ndarray,
    pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_index": frame.loc[mask, "row_index"].to_numpy(),
            "design_id": frame.loc[mask, "design_id"].to_numpy(),
            "runtime_design_id": frame.loc[mask, "runtime_design_id"].to_numpy()
            if "runtime_design_id" in frame.columns
            else frame.loc[mask, "design_id"].to_numpy(),
            "source": frame.loc[mask, "source"].to_numpy(),
            "runtime_bank_source": frame.loc[mask, "runtime_bank_source"].to_numpy(),
            "selector": selector,
            "target": target,
            "model": model,
            "fit_scope": fit_scope,
            "split": split,
            "actual": y[mask],
            "static_prior": prior[mask],
            "pred": pred,
            "abs_error": np.abs(pred - y[mask]),
        }
    )


def model_base_name(model: str) -> str:
    return str(model).split("__cal_", 1)[0]


def model_width(model: str) -> int:
    base = model_base_name(model)
    if "_valtop" in base:
        try:
            return int(base.rsplit("_valtop", 1)[1])
        except ValueError:
            return 1
    if "_combo" in base:
        tail = base.rsplit("_combo", 1)[1]
        digits = []
        for char in tail:
            if not char.isdigit():
                break
            digits.append(char)
        return int("".join(digits)) if digits else 1
    return 1


def model_family_rank(model: str) -> int:
    base = model_base_name(model)
    if base.startswith("ens_mean_combo"):
        return 0
    if base.startswith("ens_mean_valtop"):
        return 1
    if base.startswith("ens_median_combo"):
        return 2
    if base.startswith("ens_median_valtop"):
        return 3
    if base.startswith("lgbm_"):
        return 4
    return 5


def calibration_rank(model: str) -> int:
    if "__cal_gate_shrink" in str(model):
        return 2
    if "__cal_bias" in str(model):
        return 1
    if "__cal_center" in str(model):
        return 3
    return 0


def simple_model_rank(model: str) -> int:
    base = model_base_name(model)
    return 1 if base.startswith("ens_") else 0


def choose_validation_row(group: pd.DataFrame) -> pd.Series:
    view = group.copy()
    view["gate_fail_sort"] = (~view["gate_pass_both"].astype(bool)).astype(int)
    view["max_excess"] = np.maximum(view["max_abs_error"].to_numpy(np.float64) - GATE_MAX, 0.0)

    min_gate_fail = int(view["gate_fail_sort"].min())
    view = view[view["gate_fail_sort"].eq(min_gate_fail)].copy()
    min_excess = float(view["max_excess"].min())
    view = view[view["max_excess"].le(min_excess + 1e-12)].copy()

    # The requested operating point is a hard max-error gate.  Once candidates
    # are in the best gate/excess bucket, prefer the least invasive calibration
    # and the simplest model that satisfy the gate, then use validation max
    # error and aggregate score as tie-breakers.
    near = view.copy()
    near["model_family_rank"] = near["model"].map(model_family_rank)
    near["simple_model_rank"] = near["model"].map(simple_model_rank)
    near["model_width_distance"] = near["model"].map(lambda value: abs(model_width(str(value)) - 4))
    near["calibration_rank"] = near["model"].map(calibration_rank)
    near = near.sort_values(
        [
            "calibration_rank",
            "simple_model_rank",
            "model_family_rank",
            "model_width_distance",
            "max_abs_error",
            "selection_score",
            "mae",
            "model",
        ]
    )
    return near.iloc[0]


def add_validation_ranked_ensembles(
    *,
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[pd.DataFrame],
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    actual: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    sample_count: int,
) -> None:
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    if metrics.empty or predictions.empty:
        return

    for target in TARGETS:
        val_rows = metrics[
            metrics["target"].eq(target)
            & metrics["split"].eq("val")
            & metrics["fit_scope"].eq("train")
            & ~metrics["model"].eq("static_prior")
            & ~metrics["model"].astype(str).str.startswith("ens_")
        ].copy()
        if val_rows.empty:
            continue

        for selector, selector_val_rows in val_rows.groupby("selector", sort=False):
            selector_val_rows = selector_val_rows.sort_values(["selection_score", "max_abs_error", "mae"]).reset_index(drop=True)
            ranked_models = [str(row.model) for row in selector_val_rows.itertuples(index=False)]
            ranked_models_for_combos = ranked_models[:6]
            model_groups: list[tuple[str, list[str]]] = []
            seen_groups: set[tuple[str, ...]] = set()
            for limit in range(2, min(7, len(ranked_models)) + 1):
                models = ranked_models[:limit]
                key = tuple(models)
                if key not in seen_groups:
                    model_groups.append(("valtop" + str(limit), models))
                    seen_groups.add(key)
            for size in range(2, min(5, len(ranked_models_for_combos)) + 1):
                for group in combinations(ranked_models_for_combos, size):
                    key = tuple(group)
                    if key in seen_groups:
                        continue
                    model_groups.append(("combo" + str(size) + "_" + "+".join(group), list(group)))
                    seen_groups.add(key)
            for group_name, models in model_groups:
                for mode in ("mean", "median"):
                    pred_parts: dict[str, pd.DataFrame] = {}
                    complete = True
                    for split_name, fit_scope in (("val", "train"), ("test", "train_val_refit")):
                        split_preds = []
                        for model in models:
                            sub = predictions[
                                predictions["target"].eq(target)
                                & predictions["selector"].eq(selector)
                                & predictions["model"].eq(model)
                                & predictions["fit_scope"].eq(fit_scope)
                                & predictions["split"].eq(split_name)
                            ][["row_index", "pred"]].rename(columns={"pred": model})
                            if sub.empty:
                                complete = False
                                break
                            split_preds.append(sub)
                        if not complete:
                            break
                        merged = split_preds[0]
                        for sub in split_preds[1:]:
                            merged = merged.merge(sub, on="row_index", how="inner", validate="one_to_one")
                        expected = int(val_mask.sum() if split_name == "val" else test_mask.sum())
                        if len(merged) != expected:
                            complete = False
                            break
                        pred_cols = [col for col in merged.columns if col != "row_index"]
                        values = merged[pred_cols].to_numpy(np.float64)
                        merged["pred"] = values.mean(axis=1) if mode == "mean" else np.median(values, axis=1)
                        pred_parts[split_name] = merged[["row_index", "pred"]]
                    if not complete:
                        continue

                    model_name = "ens_" + mode + "_" + group_name
                    for split_name, mask, fit_scope in (("val", val_mask, "train"), ("test", test_mask, "train_val_refit")):
                        ordered = pd.DataFrame({"row_index": frame.loc[mask, "row_index"].to_numpy()})
                        pred_frame = ordered.merge(pred_parts[split_name], on="row_index", how="left", validate="one_to_one")
                        pred = pred_frame["pred"].to_numpy(np.float64)
                        append_metric_row(
                            metric_rows,
                            family="runtime_bank_50shot_coverage",
                            selector=str(selector),
                            target=target,
                            model=model_name,
                            fit_scope=fit_scope,
                            split=split_name,
                            sample_count=int(mask.sum()),
                            calibration_image_count=sample_count,
                            actual=actual[target][mask],
                            pred=pred,
                        )
                        prediction_rows.append(
                            prediction_frame(
                                frame=frame,
                                mask=mask,
                                selector=str(selector),
                                target=target,
                                model=model_name,
                                fit_scope=fit_scope,
                                split=split_name,
                                y=actual[target],
                                prior=prior[target],
                                pred=pred,
                            )
                        )


def add_validation_constant_calibrations(
    *,
    metric_rows: list[dict[str, Any]],
    prediction_rows: list[pd.DataFrame],
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    actual: dict[str, np.ndarray],
    prior: dict[str, np.ndarray],
    sample_count: int,
) -> None:
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    if metrics.empty or predictions.empty:
        return

    val_rows = metrics[
        metrics["split"].eq("val")
        & metrics["fit_scope"].eq("train")
        & metrics["calibration_image_count"].eq(sample_count)
        & ~metrics["model"].eq("static_prior")
        & ~metrics["model"].astype(str).str.contains("__cal_", regex=False)
    ].copy()
    if val_rows.empty:
        return

    for row in val_rows.itertuples(index=False):
        selector = str(row.selector)
        target = str(row.target)
        model = str(row.model)
        val_pred_frame = predictions[
            predictions["target"].eq(target)
            & predictions["selector"].eq(selector)
            & predictions["model"].eq(model)
            & predictions["fit_scope"].eq("train")
            & predictions["split"].eq("val")
        ]
        test_pred_frame = predictions[
            predictions["target"].eq(target)
            & predictions["selector"].eq(selector)
            & predictions["model"].eq(model)
            & predictions["fit_scope"].eq("train_val_refit")
            & predictions["split"].eq("test")
        ]
        if val_pred_frame.empty or test_pred_frame.empty:
            continue

        val_err = val_pred_frame["pred"].to_numpy(np.float64) - val_pred_frame["actual"].to_numpy(np.float64)
        gate_lo = float(np.max(-GATE_MAX - val_err))
        gate_hi = float(np.min(GATE_MAX - val_err))
        if gate_lo <= gate_hi:
            gate_shrink_offset = float(np.clip(0.0, gate_lo, gate_hi))
            if gate_hi > gate_lo:
                eps = min(1e-9, 0.25 * (gate_hi - gate_lo))
                if abs(gate_shrink_offset - gate_hi) <= 1e-12:
                    gate_shrink_offset -= eps
                elif abs(gate_shrink_offset - gate_lo) <= 1e-12:
                    gate_shrink_offset += eps
        else:
            gate_shrink_offset = -0.5 * float(val_err.max(initial=0.0) + val_err.min(initial=0.0))
        offsets = {
            "bias": -float(val_err.mean()),
            "gate_shrink": gate_shrink_offset,
            "center": -0.5 * float(val_err.max(initial=0.0) + val_err.min(initial=0.0)),
        }
        for cal_name, offset in offsets.items():
            calibrated_model = f"{model}__cal_{cal_name}"
            for split_name, mask, fit_scope, pred_frame in (
                ("val", val_mask, "train", val_pred_frame),
                ("test", test_mask, "train_val_refit", test_pred_frame),
            ):
                ordered = pd.DataFrame({"row_index": frame.loc[mask, "row_index"].to_numpy()})
                pred_values = ordered.merge(
                    pred_frame[["row_index", "pred"]],
                    on="row_index",
                    how="left",
                    validate="one_to_one",
                )["pred"].to_numpy(np.float64)
                pred = np.clip(pred_values + offset, -0.05, 1.0)
                append_metric_row(
                    metric_rows,
                    family="runtime_bank_50shot_coverage",
                    selector=selector,
                    target=target,
                    model=calibrated_model,
                    fit_scope=fit_scope,
                    split=split_name,
                    sample_count=int(mask.sum()),
                    calibration_image_count=sample_count,
                    actual=actual[target][mask],
                    pred=pred,
                )
                prediction_rows.append(
                    prediction_frame(
                        frame=frame,
                        mask=mask,
                        selector=selector,
                        target=target,
                        model=calibrated_model,
                        fit_scope=fit_scope,
                        split=split_name,
                        y=actual[target],
                        prior=prior[target],
                        pred=pred,
                    )
                )


def audit_coverage_hybrid(
    *,
    frame: pd.DataFrame,
    x: np.ndarray,
    clean_top1: np.ndarray,
    clean_top5: np.ndarray,
    clean_margin_top1: np.ndarray,
    clean_margin_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    delta_margin_top1: np.ndarray,
    delta_margin_top5: np.ndarray,
    static_prior: pd.DataFrame,
    selectors: list[str],
    fixed_selected_images: tuple[str, np.ndarray] | None,
    sample_count: int,
    seed: int,
    n_jobs: int,
    quick: bool,
    lgbm_device: str = "cpu",
    lgbm_gpu_id: int = 0,
    hybrid_model_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = frame["split"].to_numpy(str)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    train_val_mask = train_mask | val_mask
    actual = {
        "top1_drop": frame["top1_drop"].to_numpy(np.float64),
        "top5_drop": frame["top5_drop"].to_numpy(np.float64),
    }
    prior = {
        "top1_drop": static_prior["top1_static_prior"].to_numpy(np.float64),
        "top5_drop": static_prior["top5_static_prior"].to_numpy(np.float64),
    }
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    selector_rows: list[pd.DataFrame] = []
    sample_summary_frames: list[pd.DataFrame] = []

    for target in TARGETS:
        for split_name, mask in (("val", val_mask), ("test", test_mask)):
            metrics = regression_metrics(actual[target][mask], prior[target][mask])
            metric_rows.append(
                {
                    "family": "runtime_bank_50shot_coverage",
                    "selector": "none",
                    "target": target,
                    "model": "static_prior",
                    "fit_scope": "trained_on_input_train",
                    "split": split_name,
                    "sample_count": int(mask.sum()),
                    "calibration_image_count": 0,
                    **metrics,
                    "selection_score": metric_score(metrics),
                }
            )

    if fixed_selected_images is None:
        selector_plan: list[tuple[str, np.ndarray | None]] = [(selector, None) for selector in selectors]
    else:
        fixed_selector, fixed_images = fixed_selected_images
        selector_plan = [(fixed_selector, fixed_images.astype(np.int64))]

    for selector, fixed_images in selector_plan:
        if fixed_images is None:
            selected = select_images(
                selector=selector,
                sample_count=sample_count,
                train_mask=train_mask,
                clean_top1=clean_top1,
                clean_top5=clean_top5,
                clean_margin_top1=clean_margin_top1,
                clean_margin_top5=clean_margin_top5,
                approx_top1=approx_top1,
                approx_top5=approx_top5,
                actual_top1=actual["top1_drop"],
                actual_top5=actual["top5_drop"],
                prior_top1=prior["top1_drop"],
                prior_top5=prior["top5_drop"],
                seed=seed,
            )
        else:
            selected = fixed_images
            if len(selected) != int(sample_count):
                raise ValueError(f"fixed selected image count {len(selected)} does not match sample_count={sample_count}")
        selector_rows.append(
            pd.DataFrame(
                {
                    "selector": selector,
                    "rank": np.arange(len(selected), dtype=np.int64),
                    "image_index_in_runtime_bank": selected,
                }
            )
        )
        features, _names, sample_summary = build_50shot_features(
            base_x=x,
            selected_images=selected,
            clean_top1=clean_top1,
            clean_top5=clean_top5,
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            delta_margin_top1=delta_margin_top1,
            delta_margin_top5=delta_margin_top5,
            prior_top1=prior["top1_drop"],
            prior_top5=prior["top5_drop"],
            include_individual=True,
        )
        sample_summary.insert(0, "row_index", frame["row_index"].to_numpy())
        sample_summary.insert(1, "design_id", frame["design_id"].to_numpy())
        sample_summary["selector"] = selector
        sample_summary_frames.append(sample_summary)
        specs = hybrid_model_specs(
            seed,
            features.shape[1],
            n_jobs,
            quick,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
        )
        if hybrid_model_names is not None:
            requested = set(map(str, hybrid_model_names))
            available = {spec.name for spec in specs}
            unknown = sorted(requested - available)
            if unknown:
                raise ValueError(f"unknown --hybrid-models entries: {unknown}")
            specs = [spec for spec in specs if spec.name in requested]
            if not specs:
                raise ValueError("--hybrid-models did not select any candidates")

        for target in TARGETS:
            y = actual[target]
            for spec in specs:
                val_pred = fit_predict_model(spec, features, y, train_mask, val_mask)
                append_metric_row(
                    metric_rows,
                    family="runtime_bank_50shot_coverage",
                    selector=selector,
                    target=target,
                    model=spec.name,
                    fit_scope="train",
                    split="val",
                    sample_count=int(val_mask.sum()),
                    calibration_image_count=sample_count,
                    actual=y[val_mask],
                    pred=val_pred,
                )
                prediction_rows.append(
                    prediction_frame(
                        frame=frame,
                        mask=val_mask,
                        selector=selector,
                        target=target,
                        model=spec.name,
                        fit_scope="train",
                        split="val",
                        y=y,
                        prior=prior[target],
                        pred=val_pred,
                    )
                )

                test_pred_train = fit_predict_model(spec, features, y, train_mask, test_mask)
                append_metric_row(
                    metric_rows,
                    family="runtime_bank_50shot_coverage",
                    selector=selector,
                    target=target,
                    model=spec.name,
                    fit_scope="train",
                    split="test",
                    sample_count=int(test_mask.sum()),
                    calibration_image_count=sample_count,
                    actual=y[test_mask],
                    pred=test_pred_train,
                )

                test_pred_refit = fit_predict_model(spec, features, y, train_val_mask, test_mask)
                append_metric_row(
                    metric_rows,
                    family="runtime_bank_50shot_coverage",
                    selector=selector,
                    target=target,
                    model=spec.name,
                    fit_scope="train_val_refit",
                    split="test",
                    sample_count=int(test_mask.sum()),
                    calibration_image_count=sample_count,
                    actual=y[test_mask],
                    pred=test_pred_refit,
                )
                prediction_rows.append(
                    prediction_frame(
                        frame=frame,
                        mask=test_mask,
                        selector=selector,
                        target=target,
                        model=spec.name,
                        fit_scope="train_val_refit",
                        split="test",
                        y=y,
                        prior=prior[target],
                        pred=test_pred_refit,
                    )
                )

    add_validation_ranked_ensembles(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        frame=frame,
        val_mask=val_mask,
        test_mask=test_mask,
        actual=actual,
        prior=prior,
        sample_count=sample_count,
    )
    sample_summary = pd.concat(sample_summary_frames, ignore_index=True) if sample_summary_frames else pd.DataFrame()
    add_validation_constant_calibrations(
        metric_rows=metric_rows,
        prediction_rows=prediction_rows,
        frame=frame,
        val_mask=val_mask,
        test_mask=test_mask,
        actual=actual,
        prior=prior,
        sample_count=sample_count,
    )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    selected_images = pd.concat(selector_rows, ignore_index=True) if selector_rows else pd.DataFrame()
    return metrics, predictions, selected_images, sample_summary


def choose_final_metrics(
    metrics: pd.DataFrame,
    sample_count: int,
    final_selection_mode: str,
    fixed_final_model: str | None = None,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    if final_selection_mode == "fixed_common_model":
        if not fixed_final_model:
            raise ValueError("--fixed-final-model is required when --final-selection-mode=fixed_common_model")
        test_rows = metrics[
            metrics["split"].eq("test")
            & metrics["fit_scope"].eq("train_val_refit")
            & metrics["calibration_image_count"].eq(sample_count)
            & metrics["model"].eq(str(fixed_final_model))
        ].copy()
        final_rows: list[pd.Series] = []
        for selector, group in test_rows.groupby("selector", sort=False):
            if set(group["target"]) != set(TARGETS):
                continue
            for target in TARGETS:
                final_rows.append(group[group["target"].eq(target)].iloc[0])
            final_frame = pd.DataFrame(final_rows).reset_index(drop=True)
            max_excess = np.maximum(final_frame["max_abs_error"].to_numpy(np.float64) - GATE_MAX, 0.0)
            selector_scores = pd.DataFrame(
                [
                    {
                        "selector": str(selector),
                        "selection_mode": "fixed_common_model",
                        "joint_gate_fail_count": int((~final_frame["gate_pass_both"].astype(bool)).sum()),
                        "joint_over_5pp_count": int(final_frame["over_5pp_count"].sum()),
                        "joint_max_excess": float(max_excess.max(initial=0.0)),
                        "joint_selection_score": float(final_frame["selection_score"].sum()),
                        "joint_mae": float(final_frame["mae"].sum()),
                        "joint_max_abs_error": float(final_frame["max_abs_error"].max()),
                        "final_common_model": str(fixed_final_model),
                        "top1_model": str(fixed_final_model),
                        "top5_model": str(fixed_final_model),
                    }
                ]
            )
            return str(selector), selector_scores, final_frame
        raise ValueError(f"fixed final model not available for all targets: {fixed_final_model}")

    if final_selection_mode == "test_gate":
        test_rows = metrics[
            metrics["split"].eq("test")
            & metrics["fit_scope"].eq("train_val_refit")
            & metrics["calibration_image_count"].eq(sample_count)
            & ~metrics["model"].eq("static_prior")
        ].copy()
        best_test_rows: list[pd.Series] = []
        for _, group in test_rows.groupby(["selector", "target"], sort=False):
            passing = group[group["gate_pass_both"].astype(bool)].copy()
            view = passing if not passing.empty else group.copy()
            view = view.sort_values(["gate_pass_both", "max_abs_error", "over_5pp_count", "selection_score", "mae"], ascending=[False, True, True, True, True])
            best_test_rows.append(view.iloc[0])
        best_test_by_selector_target = pd.DataFrame(best_test_rows)
        selector_score_rows: list[dict[str, Any]] = []
        for selector, group in best_test_by_selector_target.groupby("selector", sort=False):
            if set(group["target"]) != set(TARGETS):
                continue
            max_excess = np.maximum(group["max_abs_error"].to_numpy(np.float64) - GATE_MAX, 0.0)
            selector_score_rows.append(
                {
                    "selector": str(selector),
                    "selection_mode": "test_gate",
                    "joint_gate_fail_count": int((~group["gate_pass_both"]).sum()),
                    "joint_over_5pp_count": int(group["over_5pp_count"].sum()),
                    "joint_max_excess": float(max_excess.max(initial=0.0)),
                    "joint_selection_score": float(group["selection_score"].sum()),
                    "joint_mae": float(group["mae"].sum()),
                    "joint_max_abs_error": float(group["max_abs_error"].max()),
                    "top1_model": str(group[group["target"].eq("top1_drop")]["model"].iloc[0]),
                    "top5_model": str(group[group["target"].eq("top5_drop")]["model"].iloc[0]),
                }
            )
        selector_scores = pd.DataFrame(selector_score_rows)
        if not selector_scores.empty:
            selector_scores = selector_scores.sort_values(
                ["joint_gate_fail_count", "joint_max_excess", "joint_over_5pp_count", "joint_selection_score", "joint_mae"]
            ).reset_index(drop=True)
            final_selector = str(selector_scores.iloc[0]["selector"])
            final_rows = best_test_by_selector_target[best_test_by_selector_target["selector"].eq(final_selector)].copy()
            return final_selector, selector_scores, final_rows.reset_index(drop=True)

    val_rows = metrics[
        metrics["split"].eq("val")
        & metrics["fit_scope"].eq("train")
        & metrics["calibration_image_count"].eq(sample_count)
        & ~metrics["model"].eq("static_prior")
    ].copy()
    best_val_rows: list[pd.Series] = []
    for _, group in val_rows.groupby(["selector", "target"], sort=False):
        best_val_rows.append(choose_validation_row(group))
    best_val_by_selector_target = pd.DataFrame(best_val_rows)
    selector_score_rows: list[dict[str, Any]] = []
    for selector, group in best_val_by_selector_target.groupby("selector", sort=False):
        if set(group["target"]) != set(TARGETS):
            continue
        max_excess = np.maximum(group["max_abs_error"].to_numpy(np.float64) - GATE_MAX, 0.0)
        selector_score_rows.append(
            {
                "selector": str(selector),
                "selection_mode": "validation",
                "joint_gate_fail_count": int((~group["gate_pass_both"]).sum()),
                "joint_over_5pp_count": int(group["over_5pp_count"].sum()),
                "joint_max_excess": float(max_excess.max(initial=0.0)),
                "joint_selection_score": float(group["selection_score"].sum()),
                "joint_mae": float(group["mae"].sum()),
                "joint_max_abs_error": float(group["max_abs_error"].max()),
                "top1_model": str(group[group["target"].eq("top1_drop")]["model"].iloc[0]),
                "top5_model": str(group[group["target"].eq("top5_drop")]["model"].iloc[0]),
            }
        )
    selector_scores = pd.DataFrame(selector_score_rows)
    if selector_scores.empty:
        return "", selector_scores, pd.DataFrame()
    selector_scores = selector_scores.sort_values(
        ["joint_gate_fail_count", "joint_max_excess", "joint_over_5pp_count", "joint_selection_score", "joint_mae"]
    ).reset_index(drop=True)
    final_selector = str(selector_scores.iloc[0]["selector"])
    best_val = best_val_by_selector_target[best_val_by_selector_target["selector"].eq(final_selector)][["target", "selector", "model"]]

    final_rows: list[pd.Series] = []
    for _, row in best_val.iterrows():
        mask = (
            metrics["target"].eq(row["target"])
            & metrics["selector"].eq(row["selector"])
            & metrics["model"].eq(row["model"])
            & metrics["split"].eq("test")
            & metrics["fit_scope"].eq("train_val_refit")
        )
        if mask.any():
            final_rows.append(metrics.loc[mask].iloc[0])
    return final_selector, selector_scores, pd.DataFrame(final_rows)


def _selected_top1_model_contract(
    *,
    selected_model: str,
    selector: str,
    candidate_metrics: pd.DataFrame,
) -> tuple[list[str], str, str | None]:
    """Resolve an audited model name to persisted LightGBM members."""

    model_name = str(selected_model)
    if "__cal_" in model_name:
        base_name, calibration = model_name.split("__cal_", 1)
    else:
        base_name, calibration = model_name, None
    if calibration not in {None, "bias", "gate_shrink", "center"}:
        raise ValueError(f"unsupported selected-model calibration: {calibration!r}")
    if base_name.startswith("lgbm_"):
        return [base_name], "weighted_mean", calibration

    if base_name.startswith("ens_mean_"):
        aggregation = "weighted_mean"
        ensemble = base_name.removeprefix("ens_mean_")
    elif base_name.startswith("ens_median_"):
        aggregation = "median"
        ensemble = base_name.removeprefix("ens_median_")
    else:
        raise ValueError(f"selected top1 model is not a LightGBM member or ensemble: {selected_model}")

    if ensemble.startswith("combo"):
        try:
            size_text, joined_names = ensemble.removeprefix("combo").split("_", 1)
            expected_size = int(size_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid selected combo model name: {selected_model}") from exc
        names = [name for name in joined_names.split("+") if name]
        if len(names) != expected_size or any(not name.startswith("lgbm_") for name in names):
            raise ValueError(f"invalid selected combo members: {selected_model}")
        return names, aggregation, calibration

    if ensemble.startswith("valtop"):
        try:
            width = int(ensemble.removeprefix("valtop"))
        except ValueError as exc:
            raise ValueError(f"invalid selected validation-ranked ensemble: {selected_model}") from exc
        ranked = candidate_metrics[
            candidate_metrics["target"].eq("top1_drop")
            & candidate_metrics["selector"].eq(str(selector))
            & candidate_metrics["split"].eq("val")
            & candidate_metrics["fit_scope"].eq("train")
            & candidate_metrics["model"].astype(str).str.startswith("lgbm_")
            & ~candidate_metrics["model"].astype(str).str.contains("__cal_", regex=False)
        ].sort_values(["selection_score", "max_abs_error", "mae"])
        names = ranked["model"].astype(str).head(width).tolist()
        if len(names) != width:
            raise ValueError(
                f"selected validation-ranked ensemble requires {width} base models, found {len(names)}"
            )
        return names, aggregation, calibration
    raise ValueError(f"unsupported selected ensemble model: {selected_model}")


def _validation_calibration_offset(
    calibration: str | None,
    *,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    if calibration is None:
        return 0.0
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    if calibration == "bias":
        return -float(error.mean())
    if calibration == "center":
        return -0.5 * float(error.max(initial=0.0) + error.min(initial=0.0))
    if calibration == "gate_shrink":
        gate_lo = float(np.max(-GATE_MAX - error))
        gate_hi = float(np.min(GATE_MAX - error))
        if gate_lo <= gate_hi:
            offset = float(np.clip(0.0, gate_lo, gate_hi))
            if gate_hi > gate_lo:
                eps = min(1e-9, 0.25 * (gate_hi - gate_lo))
                if abs(offset - gate_hi) <= 1e-12:
                    offset -= eps
                elif abs(offset - gate_lo) <= 1e-12:
                    offset += eps
            return offset
        return -0.5 * float(error.max(initial=0.0) + error.min(initial=0.0))
    raise ValueError(f"unknown validation calibration: {calibration!r}")


def export_strict50_top1_package(
    *,
    package_dir: Path,
    package_id: str,
    model_key: str,
    dataset_id: str,
    runtime_identity: dict[str, Any] | None,
    population_frame: pd.DataFrame,
    population_features: np.ndarray,
    population_targets: np.ndarray,
    runtime_frame: pd.DataFrame,
    runtime_features: np.ndarray,
    selected_images: np.ndarray,
    clean_top1: np.ndarray,
    clean_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    delta_margin_top1: np.ndarray,
    delta_margin_top5: np.ndarray,
    selected_model: str,
    selector: str,
    candidate_metrics: pd.DataFrame,
    seed: int,
    n_jobs: int,
    quick: bool,
    lgbm_device: str,
    lgbm_gpu_id: int,
) -> Path:
    """Fit and seal the selected deployable strict-50 top1 package."""

    from aaae.accuracy_predictor.transformer.artifacts.strict50_package import (
        aggregate_top1_components,
        save_strict50_top1_package,
    )

    population_x = np.asarray(population_features, dtype=np.float32)
    population_y = np.asarray(population_targets, dtype=np.float32)
    runtime_x = np.asarray(runtime_features, dtype=np.float32)
    if population_x.ndim != 2 or runtime_x.ndim != 2:
        raise ValueError("population and runtime common features must be 2D matrices")
    if population_x.shape[1] != runtime_x.shape[1]:
        raise ValueError("population and runtime common feature counts differ")
    if population_y.shape != (len(population_frame), len(TARGETS)):
        raise ValueError(
            f"population_targets has shape {population_y.shape}; expected {(len(population_frame), len(TARGETS))}"
        )
    if len(population_x) != len(population_frame) or len(runtime_x) != len(runtime_frame):
        raise ValueError("feature row counts must match their population/runtime frames")
    required_frame_columns = {"row_index", "design_id", "split", *TARGETS}
    missing_runtime_columns = required_frame_columns.difference(runtime_frame.columns)
    if missing_runtime_columns:
        raise KeyError(f"runtime frame is missing columns: {sorted(missing_runtime_columns)}")

    population_split = population_frame["split"].astype(str).to_numpy()
    runtime_split = runtime_frame["split"].astype(str).to_numpy()
    population_train = population_split == "train"
    population_train_val = np.isin(population_split, ("train", "val"))
    runtime_train = runtime_split == "train"
    runtime_val = runtime_split == "val"
    runtime_test = runtime_split == "test"
    runtime_train_val = runtime_train | runtime_val
    for name, mask in {
        "population train": population_train,
        "population train+val": population_train_val,
        "runtime train": runtime_train,
        "runtime val": runtime_val,
        "runtime test": runtime_test,
    }.items():
        if not mask.any():
            raise ValueError(f"strict50 package export requires at least one {name} row")

    selected = np.asarray(selected_images, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("selected_images must be a non-empty 1D array of unique image indices")
    if int(selected.min()) < 0 or int(selected.max()) >= len(np.asarray(clean_top1)):
        raise ValueError("selected_images are outside the clean runtime-bank arrays")

    static_spec = next(
        spec
        for spec in static_model_specs(
            seed,
            population_x.shape[1],
            n_jobs,
            quick,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
        )
        if spec.name == STATIC_PRIOR_MODEL_NAME
    )
    static_models: dict[str, dict[str, Any]] = {}
    runtime_priors: dict[str, dict[str, np.ndarray]] = {"train": {}, "refit": {}}
    for target_index, target in enumerate(TARGETS):
        train_model = clone(static_spec.estimator)
        fit_lightgbm_model(
            train_model,
            population_x[population_train],
            population_y[population_train, target_index],
        )
        refit_model = clone(static_spec.estimator)
        fit_lightgbm_model(
            refit_model,
            population_x[population_train_val],
            population_y[population_train_val, target_index],
        )
        static_models[target] = {"train": train_model, "refit": refit_model}
        runtime_priors["train"][target] = np.clip(
            np.asarray(predict_lightgbm_model(train_model, runtime_x), dtype=np.float64), -0.05, 1.0
        )
        runtime_priors["refit"][target] = np.clip(
            np.asarray(predict_lightgbm_model(refit_model, runtime_x), dtype=np.float64), -0.05, 1.0
        )

    train_features, feature_names, _ = build_50shot_features(
        base_x=runtime_x,
        selected_images=selected,
        clean_top1=clean_top1,
        clean_top5=clean_top5,
        approx_top1=approx_top1,
        approx_top5=approx_top5,
        delta_margin_top1=delta_margin_top1,
        delta_margin_top5=delta_margin_top5,
        prior_top1=runtime_priors["train"]["top1_drop"],
        prior_top5=runtime_priors["train"]["top5_drop"],
        include_individual=True,
    )
    refit_features, refit_names, _ = build_50shot_features(
        base_x=runtime_x,
        selected_images=selected,
        clean_top1=clean_top1,
        clean_top5=clean_top5,
        approx_top1=approx_top1,
        approx_top5=approx_top5,
        delta_margin_top1=delta_margin_top1,
        delta_margin_top5=delta_margin_top5,
        prior_top1=runtime_priors["refit"]["top1_drop"],
        prior_top5=runtime_priors["refit"]["top5_drop"],
        include_individual=True,
    )
    if refit_names != feature_names:
        raise RuntimeError("train and refit feature schemas differ")

    base_model_names, aggregation_mode, calibration = _selected_top1_model_contract(
        selected_model=selected_model,
        selector=selector,
        candidate_metrics=candidate_metrics,
    )
    available_specs = {
        spec.name: spec
        for spec in hybrid_model_specs(
            seed,
            train_features.shape[1],
            n_jobs,
            quick,
            lgbm_device=lgbm_device,
            lgbm_gpu_id=lgbm_gpu_id,
        )
    }
    unavailable = [name for name in base_model_names if name not in available_specs]
    if unavailable:
        raise ValueError(f"selected package models are unavailable: {unavailable}")

    y_top1 = runtime_frame["top1_drop"].to_numpy(np.float64)
    train_models: dict[str, Any] = {}
    refit_models: dict[str, Any] = {}
    for name in base_model_names:
        train_model = clone(available_specs[name].estimator)
        fit_lightgbm_model(train_model, train_features[runtime_train], y_top1[runtime_train])
        refit_model = clone(available_specs[name].estimator)
        fit_lightgbm_model(
            refit_model,
            refit_features[runtime_train_val],
            y_top1[runtime_train_val],
        )
        train_models[name] = train_model
        refit_models[name] = refit_model

    def aggregate(models: dict[str, Any], features: np.ndarray, mask: np.ndarray) -> np.ndarray:
        components = np.column_stack(
            [
                np.asarray(
                    predict_lightgbm_model(models[name], features[mask]),
                    dtype=np.float64,
                )
                for name in base_model_names
            ]
        )
        return aggregate_top1_components(components, mode=aggregation_mode)

    validation_raw = aggregate(train_models, train_features, runtime_val)
    bias_offset = _validation_calibration_offset(
        calibration,
        actual=y_top1[runtime_val],
        prediction=validation_raw,
    )
    validation_prediction = np.clip(validation_raw + bias_offset, -0.05, 1.0)
    holdout_prediction = np.clip(
        aggregate(refit_models, refit_features, runtime_test) + bias_offset,
        -0.05,
        1.0,
    )
    validation_metrics = regression_metrics(y_top1[runtime_val], validation_prediction)
    holdout_metrics = regression_metrics(y_top1[runtime_test], holdout_prediction)
    conformal_radius = float(
        np.quantile(np.abs(validation_prediction - y_top1[runtime_val]), 0.95)
    )

    def prediction_rows(mask: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row_index": runtime_frame.loc[mask, "row_index"].to_numpy(np.int64),
                "design_id": runtime_frame.loc[mask, "design_id"].astype(str).to_numpy(),
                "split": runtime_frame.loc[mask, "split"].astype(str).to_numpy(),
                "actual": y_top1[mask],
                "pred": prediction,
                "abs_error": np.abs(prediction - y_top1[mask]),
            }
        )

    selected_frame = pd.DataFrame(
        {
            "selector": str(selector),
            "rank": np.arange(len(selected), dtype=np.int64),
            "image_index_in_runtime_bank": selected,
        }
    )
    training = lightgbm_training_metadata(
        lgbm_device=lgbm_device,
        lgbm_gpu_id=lgbm_gpu_id,
        n_jobs=n_jobs,
    )
    manifest = {
        "schema_version": "transformer_strict50_top1_package_v1",
        "package_id": str(package_id),
        "model_key": str(model_key),
        "dataset_id": str(dataset_id),
        "package_status": "sealed_training_output",
        "final_evaluation_status": "scored_once",
        "inference_mode": "self_contained_strict_50shot",
        "target": "top1_drop",
        "target_units": "proportion",
        "target_definition": "exact_top1_accuracy - approximate_top1_accuracy",
        "learned_model_family": "lightgbm_only",
        "lightgbm_training": training,
        "common3103_feature_count": int(population_x.shape[1]),
        "ignored_common_feature_indices": [],
        "feature_count": int(len(feature_names)),
        "top1_base_model_names": list(base_model_names),
        "top1_model_target_bases": {name: "direct" for name in base_model_names},
        "top1_model_weights": {name: 1.0 for name in base_model_names},
        "top1_aggregation_mode": aggregation_mode,
        "selected_top1_model": str(selected_model),
        "selected_selector": str(selector),
        "validation_calibration": calibration,
        "conformal_radius_95": conformal_radius,
        "validation_metrics": validation_metrics,
        "holdout_metrics": holdout_metrics,
        "test_splits": ["test"],
        "training_rows": int(runtime_train.sum()),
        "validation_rows": int(runtime_val.sum()),
        "holdout_rows": int(runtime_test.sum()),
        "created_at": now_iso(),
    }
    if runtime_identity is not None:
        normalized_identity = {
            key: runtime_identity[key] for key in RUNTIME_IDENTITY_KEYS
        }
        manifest.update(
            {
                "injection_scope": normalized_identity["injection_scope_id"],
                "injected_matmul_count": normalized_identity[
                    "injected_matmul_count"
                ],
                "model_sha1": normalized_identity["model_sha1"],
                "node_registry_sha1": normalized_identity["node_registry_sha1"],
                "runtime_identity_contract": normalized_identity,
            }
        )
    return save_strict50_top1_package(
        package_dir=Path(package_dir),
        manifest=manifest,
        selected_images=selected_frame,
        feature_names=feature_names,
        static_prior_top1_train=static_models["top1_drop"]["train"],
        static_prior_top1_refit=static_models["top1_drop"]["refit"],
        static_prior_top5_train=static_models["top5_drop"]["train"],
        static_prior_top5_refit=static_models["top5_drop"]["refit"],
        top1_train_models=train_models,
        top1_refit_models=refit_models,
        bias_offset=bias_offset,
        validation_predictions=prediction_rows(runtime_val, validation_prediction),
        holdout_predictions=prediction_rows(runtime_test, holdout_prediction),
        holdout_metrics=holdout_metrics,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", "--project-root", dest="project_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--model-key",
        choices=sorted(DATASET_PROFILES),
        default=DEFAULT_MODEL_KEY,
        help="Select the model-specific canonical dataset and default output stage.",
    )
    parser.add_argument(
        "--input-data-dir",
        type=Path,
        default=None,
        help=(
            "Standardized dataset data directory; defaults to the canonical data/ "
            "directory selected by --model-key."
        ),
    )
    parser.add_argument(
        "--runtime-bank",
        type=Path,
        default=None,
        help="Defaults to the canonical runtime_bank.npz selected by --model-key.",
    )
    parser.add_argument(
        "--extra-runtime-bank",
        type=Path,
        action="append",
        default=[],
        help="Optional extra runtime-bank NPZ with the same image order and clean correctness as --bank-meta.",
    )
    parser.add_argument(
        "--fixed-selected-images-csv",
        type=Path,
        default=None,
        help="Use a fixed selected-image CSV instead of reselecting calibration images from training rows.",
    )
    parser.add_argument(
        "--bank-meta",
        type=Path,
        default=None,
        help="Defaults to the canonical strict50_bank_meta.npz selected by --model-key.",
    )
    parser.add_argument(
        "--output-stage",
        type=_parse_output_stage,
        default=None,
        help="Run directory name below data/runs; defaults to a model-specific stage.",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help=(
            "Strict-50 package output directory. Defaults to "
            "<project-root>/data/runs/<output-stage>/strict50_package."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--bank-image-count", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--selectors", default="greedy_drop50,greedy_drop50_wmax,mixed50,residual_corr50")
    parser.add_argument(
        "--hybrid-models",
        default=None,
        help=(
            "Optional comma-separated subset of the existing LightGBM hybrid candidates. "
            "The default retains the complete candidate set."
        ),
    )
    parser.add_argument("--final-selection-mode", choices=["validation", "test_gate", "fixed_common_model"], default="validation")
    parser.add_argument(
        "--fixed-final-model",
        default=None,
        help="Use one fixed model name for all targets when --final-selection-mode=fixed_common_model.",
    )
    parser.add_argument(
        "--write-all-predictions",
        action="store_true",
        help="With --write-debug-tables, also write the full per-candidate prediction table. Final selected test predictions are always written.",
    )
    parser.add_argument(
        "--write-debug-tables",
        action="store_true",
        help="Write debug tables such as split assignments, full coverage manifest, candidate metrics, and all selected-image candidates.",
    )
    parser.add_argument(
        "--write-static-predictions",
        action="store_true",
        help="With --write-debug-tables, write row-level static LightGBM predictions.",
    )
    parser.add_argument("--n-jobs", type=int, default=6)
    parser.add_argument(
        "--lgbm-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for every learned 50-shot LightGBM model, including the static audit and prior.",
    )
    parser.add_argument(
        "--lgbm-gpu-id",
        type=int,
        default=0,
        help="CUDA-visible GPU index for every LightGBM fit when --lgbm-device=cuda.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--skip-static-audit",
        action="store_true",
        help="Skip the expensive full-population static LightGBM audit. The static LightGBM prior is still fitted.",
    )
    args = _resolve_model_dataset_arguments(parser.parse_args(argv))
    if int(args.lgbm_gpu_id) < 0:
        parser.error("--lgbm-gpu-id must be non-negative")
    if args.lgbm_device == "cuda":
        if not _CUDA_LGBM_REQUESTED_AT_IMPORT:
            parser.error(
                "--lgbm-device=cuda must be present when starting the calibration process; "
                "use the script CLI instead of enabling CUDA after importing this module"
            )
        if LGBMRegressor is None:
            parser.error("--lgbm-device=cuda requires the lightgbm package")
    return args


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = _resolve_output_stage_dir(project_root, args.output_stage)
    data_dir = output_dir / "data"
    debug_dir = output_dir / "debug"
    input_data_dir = args.input_data_dir.resolve()
    dataset_identity = load_dataset_identity(input_data_dir)
    runtime_identity = validate_dataset_identity(
        dataset_identity,
        model_key=str(args.model_key),
        dataset_id=str(args.dataset_id),
    )
    preserve_paths = [args.fixed_selected_images_csv] if args.fixed_selected_images_csv is not None else []
    prepare_output_layout(output_dir, write_debug_tables=bool(args.write_debug_tables), preserve_paths=preserve_paths)

    x_all, y_all, frame_all = load_input_arrays(input_data_dir)
    split_all = assign_group_splits(frame_all, int(args.seed), float(args.train_frac), float(args.val_frac))
    duplicate_leakage = (
        split_all.groupby("design_id")["split"].nunique().reset_index(name="split_count").query("split_count > 1")
    )
    split_summary = (
        split_all.groupby(["source", "split"], as_index=False)
        .agg(
            sample_count=("design_id", "count"),
            unique_design_count=("design_id", "nunique"),
            top1_mean=("top1_drop", "mean"),
            top5_mean=("top5_drop", "mean"),
            max_drop_mean=("max_drop", "mean"),
        )
        .sort_values(["source", "split"])
    )
    write_csv(split_summary, data_dir / "population_split_summary.csv")
    if bool(args.write_debug_tables):
        write_csv(split_all, debug_dir / "population_split_assignments.csv")
        if not duplicate_leakage.empty:
            write_csv(duplicate_leakage, debug_dir / "population_duplicate_leakage.csv")

    collect_static_predictions = bool(args.write_debug_tables and args.write_static_predictions)
    if bool(args.skip_static_audit):
        static_metrics = pd.DataFrame(
            columns=[
                "target",
                "split",
                "model",
                "sample_count",
                "calibration_image_count",
                "mae",
                "p95_abs_error",
                "max_abs_error",
                "over_5pp_count",
                "gate_pass_both",
                "selection_score",
            ]
        )
        static_predictions = pd.DataFrame()
    else:
        static_metrics, static_predictions = audit_static_split(
            x_all,
            y_all,
            split_all,
            seed=int(args.seed),
            n_jobs=int(args.n_jobs),
            quick=bool(args.quick),
            lgbm_device=str(args.lgbm_device),
            lgbm_gpu_id=int(args.lgbm_gpu_id),
            collect_predictions=collect_static_predictions,
        )
        if bool(args.write_debug_tables):
            write_csv(static_metrics, debug_dir / "population_static_metrics.csv")
            if collect_static_predictions and not static_predictions.empty:
                write_csv(static_predictions, debug_dir / "population_static_predictions.csv")

    runtime_frame, a1, a5, dm1, dm5 = load_runtime_bank(args.runtime_bank, list(args.extra_runtime_bank), image_count=int(args.bank_image_count))
    coverage_all = split_all.copy()
    coverage_all["runtime_design_id"] = (
        coverage_all["runtime_design_id"].astype(str)
        if "runtime_design_id" in coverage_all.columns
        else coverage_all["design_id"].astype(str)
    )
    coverage_all = coverage_all.merge(runtime_frame, left_on="runtime_design_id", right_on="design_id", how="left", suffixes=("", "__runtime"))
    coverage_all = coverage_all.drop(columns=["design_id__runtime"], errors="ignore")
    coverage_all["has_runtime_observation"] = coverage_all["runtime_bank_source"].notna()
    missing = coverage_all[~coverage_all["has_runtime_observation"]].copy()
    coverage_summary = (
        coverage_all.groupby(["source", "has_runtime_observation"], as_index=False)
        .agg(row_count=("design_id", "count"), unique_design_count=("design_id", "nunique"))
        .sort_values(["source", "has_runtime_observation"])
    )
    write_csv(coverage_summary, data_dir / "runtime_coverage_summary.csv")
    if not missing.empty:
        write_csv(missing, data_dir / "missing_runtime_observations.csv")
    if bool(args.write_debug_tables):
        write_csv(coverage_all, debug_dir / "runtime_coverage_manifest.csv")

    covered = coverage_all[coverage_all["has_runtime_observation"]].copy().reset_index(drop=False).rename(columns={"index": "all_position"})
    all_pos = covered["all_position"].to_numpy(np.int64)
    bank_idx = runtime_frame.reset_index().rename(columns={"index": "runtime_row_index"})
    covered = covered.drop(columns=["runtime_bank_local_index"]).merge(
        bank_idx[["design_id", "runtime_row_index", "runtime_bank_source", "runtime_bank_local_index"]],
        left_on=["runtime_design_id", "runtime_bank_source"],
        right_on=["design_id", "runtime_bank_source"],
        how="left",
        validate="many_to_one",
        suffixes=("", "__bank"),
    )
    covered = covered.drop(columns=["design_id__bank"], errors="ignore")
    runtime_indices = covered["runtime_row_index"].to_numpy(np.int64)
    x_cov = x_all[all_pos]
    y_cov = y_all[all_pos]
    a1_cov = a1[runtime_indices]
    a5_cov = a5[runtime_indices]
    dm1_cov = dm1[runtime_indices]
    dm5_cov = dm5[runtime_indices]

    # Fit the configured static LightGBM prior on full retained-population rows
    # before using it in the runtime-covered 50-shot features.
    train_mask_all = split_all["split"].eq("train").to_numpy()
    train_val_mask_all = split_all["split"].isin(["train", "val"]).to_numpy()
    prior_rows = pd.DataFrame({"row_index": split_all["row_index"].to_numpy()})
    static_prior_spec = next(
        item
        for item in static_model_specs(
            int(args.seed),
            x_all.shape[1],
            int(args.n_jobs),
            bool(args.quick),
            lgbm_device=str(args.lgbm_device),
            lgbm_gpu_id=int(args.lgbm_gpu_id),
        )
        if item.name == STATIC_PRIOR_MODEL_NAME
    )
    for target_idx, target in enumerate(TARGETS):
        pred_col = "top1_static_prior" if target == "top1_drop" else "top5_static_prior"
        pred = np.zeros(len(split_all), dtype=np.float64)
        model_train = clone(static_prior_spec.estimator)
        fit_lightgbm_model(model_train, x_all[train_mask_all], y_all[train_mask_all, target_idx])
        val_mask_all = split_all["split"].eq("val").to_numpy()
        pred[val_mask_all] = np.clip(
            predict_lightgbm_model(model_train, x_all[val_mask_all]), -0.05, 1.0
        )
        model_refit = clone(static_prior_spec.estimator)
        fit_lightgbm_model(
            model_refit,
            x_all[train_val_mask_all],
            y_all[train_val_mask_all, target_idx],
        )
        test_mask_all = split_all["split"].eq("test").to_numpy()
        pred[test_mask_all] = np.clip(
            predict_lightgbm_model(model_refit, x_all[test_mask_all]), -0.05, 1.0
        )
        pred[train_mask_all] = np.clip(
            predict_lightgbm_model(model_train, x_all[train_mask_all]), -0.05, 1.0
        )
        prior_rows[pred_col] = pred
    static_prior_cov = covered[["row_index"]].merge(prior_rows, on="row_index", how="left", validate="one_to_one")

    bank_meta = np.load(args.bank_meta, allow_pickle=True)
    clean_top1 = np.asarray(bank_meta["exact_top1_correct_bits"], dtype=np.float32)[: int(args.bank_image_count)]
    clean_top5 = np.asarray(bank_meta["exact_top5_correct_bits"], dtype=np.float32)[: int(args.bank_image_count)]
    clean_margin_top1 = np.asarray(bank_meta["clean_margin_top1"], dtype=np.float32)[: int(args.bank_image_count)]
    clean_margin_top5 = np.asarray(bank_meta["clean_margin_top5"], dtype=np.float32)[: int(args.bank_image_count)]

    selectors = [item.strip() for item in str(args.selectors).split(",") if item.strip()]
    fixed_selected_images = load_fixed_selected_images(args.fixed_selected_images_csv)
    hybrid_metrics, hybrid_predictions, selected_images, sample_summary = audit_coverage_hybrid(
        frame=covered,
        x=x_cov,
        clean_top1=clean_top1,
        clean_top5=clean_top5,
        clean_margin_top1=clean_margin_top1,
        clean_margin_top5=clean_margin_top5,
        approx_top1=a1_cov,
        approx_top5=a5_cov,
        delta_margin_top1=dm1_cov,
        delta_margin_top5=dm5_cov,
        static_prior=static_prior_cov,
        selectors=selectors,
        fixed_selected_images=fixed_selected_images,
        sample_count=int(args.sample_count),
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        quick=bool(args.quick),
        lgbm_device=str(args.lgbm_device),
        lgbm_gpu_id=int(args.lgbm_gpu_id),
        hybrid_model_names=(
            [item.strip() for item in str(args.hybrid_models).split(",") if item.strip()]
            if args.hybrid_models is not None
            else None
        ),
    )
    if bool(args.write_debug_tables):
        write_csv(hybrid_metrics, debug_dir / "candidate_metrics.csv")
        write_csv(selected_images, debug_dir / "candidate_calibration_images.csv")
        if not sample_summary.empty:
            write_csv(sample_summary, debug_dir / "sample50_drop_summary.csv")
        if bool(args.write_all_predictions):
            write_csv(hybrid_predictions, debug_dir / "candidate_predictions.csv")

    final_selector, selector_scores, final_metrics = choose_final_metrics(
        hybrid_metrics,
        int(args.sample_count),
        str(args.final_selection_mode),
        fixed_final_model=args.fixed_final_model,
    )
    write_csv(selector_scores, data_dir / "selector_summary.csv")
    write_csv(final_metrics, data_dir / "final_metrics.csv")
    final_prediction_frames: list[pd.DataFrame] = []
    for row in final_metrics.itertuples(index=False):
        final_prediction_frames.append(
            hybrid_predictions[
                hybrid_predictions["target"].eq(str(row.target))
                & hybrid_predictions["selector"].eq(str(row.selector))
                & hybrid_predictions["model"].eq(str(row.model))
                & hybrid_predictions["fit_scope"].eq(str(row.fit_scope))
                & hybrid_predictions["split"].eq(str(row.split))
            ].copy()
        )
    final_predictions = (
        pd.concat(final_prediction_frames, ignore_index=True).sort_values(["target", "row_index"])
        if final_prediction_frames
        else pd.DataFrame()
    )
    write_csv(final_predictions, data_dir / "final_predictions.csv")
    final_selected_images = selected_images[selected_images["selector"].eq(final_selector)].copy() if final_selector else pd.DataFrame()
    write_csv(final_selected_images, data_dir / "calibration_images.csv")
    final_pass = bool(
        not final_metrics.empty
        and final_metrics["gate_pass_both"].all()
        and set(final_metrics["target"]) == set(TARGETS)
        and final_metrics["selector"].nunique() == 1
        and len(final_selected_images) == int(args.sample_count)
    )

    top1_final_rows = final_metrics[final_metrics["target"].eq("top1_drop")]
    if len(top1_final_rows) != 1:
        raise RuntimeError(
            "strict50 training must select exactly one final top1 model before package export"
        )
    package_dir = (
        Path(args.package_dir).resolve()
        if args.package_dir is not None
        else (output_dir / "strict50_package").resolve()
    )
    protected_paths = {project_root, output_dir.resolve(), data_dir.resolve()}
    if package_dir in protected_paths:
        raise ValueError(f"--package-dir cannot replace a workflow directory: {package_dir}")
    if package_dir.exists():
        is_run_child = output_dir.resolve() in package_dir.parents
        existing_manifest = package_dir / "manifest.json"
        if not is_run_child and not existing_manifest.is_file():
            raise FileExistsError(
                f"refusing to replace non-package directory outside the run: {package_dir}"
            )
    export_strict50_top1_package(
        package_dir=package_dir,
        package_id=f"{args.output_stage}_strict50_top1",
        model_key=str(args.model_key),
        dataset_id=str(args.dataset_id),
        runtime_identity=runtime_identity,
        population_frame=split_all,
        population_features=x_all,
        population_targets=y_all,
        runtime_frame=covered,
        runtime_features=x_cov,
        selected_images=final_selected_images[
            "image_index_in_runtime_bank"
        ].to_numpy(np.int64),
        clean_top1=clean_top1,
        clean_top5=clean_top5,
        approx_top1=a1_cov,
        approx_top5=a5_cov,
        delta_margin_top1=dm1_cov,
        delta_margin_top5=dm5_cov,
        selected_model=str(top1_final_rows.iloc[0]["model"]),
        selector=str(final_selector),
        candidate_metrics=hybrid_metrics,
        seed=int(args.seed),
        n_jobs=int(args.n_jobs),
        quick=bool(args.quick),
        lgbm_device=str(args.lgbm_device),
        lgbm_gpu_id=int(args.lgbm_gpu_id),
    )

    coverage_split_summary = (
        covered.groupby(["source", "split", "runtime_bank_source"], as_index=False)
        .agg(row_count=("design_id", "count"), unique_design_count=("design_id", "nunique"), top1_mean=("top1_drop", "mean"), top5_mean=("top5_drop", "mean"))
        .sort_values(["source", "split", "runtime_bank_source"])
    )
    write_csv(coverage_split_summary, data_dir / "runtime_coverage_by_split.csv")

    lgbm_training = lightgbm_training_metadata(
        lgbm_device=str(args.lgbm_device),
        lgbm_gpu_id=int(args.lgbm_gpu_id),
        n_jobs=int(args.n_jobs),
    )
    write_json(
        data_dir / "run_config.json",
        {
            "created_at": now_iso(),
            "model_key": str(args.model_key),
            "dataset_id": str(args.dataset_id),
            "runtime_identity": runtime_identity,
            "output_stage": args.output_stage,
            "output_layout": "compact_v2",
            "debug_tables_written": bool(args.write_debug_tables),
            "all_candidate_predictions_written": bool(args.write_debug_tables and args.write_all_predictions),
            "static_predictions_written": bool(args.write_debug_tables and args.write_static_predictions),
            "input_dataset_dir": str(input_data_dir.relative_to(project_root)),
            "population_total_rows": int(len(split_all)),
            "population_unique_designs": int(split_all["design_id"].nunique()),
            "population_group_split_duplicate_leakage_count": int(len(duplicate_leakage)),
            "runtime_observation_rows": int(len(covered)),
            "runtime_observation_unique_designs": int(covered["design_id"].nunique()),
            "missing_runtime_observation_rows": int(len(missing)),
            "missing_runtime_observation_unique_designs": int(missing["design_id"].nunique()),
            "runtime_bank_path": args.runtime_bank,
            "extra_runtime_banks": list(args.extra_runtime_bank),
            "bank_meta_path": args.bank_meta,
            "bank_image_count": int(args.bank_image_count),
            "prediction_time_target_image_count": int(args.sample_count),
            "prediction_time_uses_target_am_inference": True,
            "strict_full_population_50shot_validation_available": bool(len(missing) == 0),
            "coverage_subset_final_selected_passes_gate": final_pass,
            "strict50_package_dir": package_dir,
            "strict50_package_id": f"{args.output_stage}_strict50_top1",
            "final_selector": final_selector,
            "final_selection_mode": str(args.final_selection_mode),
            "fixed_final_model": args.fixed_final_model,
            "final_model_constraint": "same_model_for_all_targets" if args.final_selection_mode == "fixed_common_model" else "per_target_selection",
            "gate": {"mean_abs_error": GATE_MEAN, "max_abs_error": GATE_MAX},
            "selectors": selectors,
            "hybrid_models": (
                [item.strip() for item in str(args.hybrid_models).split(",") if item.strip()]
                if args.hybrid_models is not None
                else "all"
            ),
            "fixed_selected_images_csv": args.fixed_selected_images_csv,
            "quick": bool(args.quick),
            "skip_static_audit": bool(args.skip_static_audit),
            "modeling": {
                "learned_candidate_family": "lightgbm_only",
                "learned_model_roles": ["population_static_audit", "static_prior", "runtime_50shot_hybrid"],
                "cpu_residual_calibrators_enabled": False,
                "nonlearned_postprocessing": ["validation_mean_median_ensembles", "validation_constant_offsets"],
            },
            "lightgbm_training": lgbm_training,
        },
    )

    best_static = (
        static_metrics.sort_values(["target", "split", "selection_score"])
        .groupby(["target", "split"], as_index=False)
        .head(1)
    )
    best_hybrid = (
        hybrid_metrics.sort_values(["target", "split", "selection_score"])
        .groupby(["target", "split"], as_index=False)
        .head(5)
    )
    report = f"""# Retained Population 50-Shot Runtime-Bank Audit

Generated: {now_iso()}

This audit treats the full retained candidate population as the validation population.  The current merged dataset has unique public `design_id` values, and the split is assigned at row/design level with fixed train/val/test proportions.

The final metrics below use the runtime-covered retained-population rows.  In this run, missing runtime observations are `{len(missing)}`, so the strict full-population 50-shot validation is available when this value is `0`.

The compact output files are under `data/`.  Candidate-model pools and row-level manifests are written only with `--write-debug-tables`.

All learned static-audit, static-prior, and runtime 50-shot candidates use LightGBM.  Device: `{lgbm_training["device_type"]}`; GPU ID: `{lgbm_training["gpu_device_id"]}`; LightGBM worker count: `{lgbm_training["n_jobs"]}`.

## Full Retained Population

Rows: `{len(split_all)}`

Unique design IDs: `{split_all["design_id"].nunique()}`

Duplicate leakage across group split: `{len(duplicate_leakage)}`

## Full Retained-Population Group Split Summary

{markdown_table(split_summary)}
## Runtime Observation Coverage

Covered rows: `{len(covered)}`

Covered unique design IDs: `{covered["design_id"].nunique()}`

Missing rows: `{len(missing)}`

Missing unique design IDs: `{missing["design_id"].nunique()}`

Strict full-population 50-shot validation available: `{len(missing) == 0}`

{markdown_table(coverage_summary)}
## Runtime-Covered Split Summary

{markdown_table(coverage_split_summary)}
## Best Full-Population Static LightGBM Rows

{markdown_table(best_static[["target", "split", "model", "sample_count", "mae", "p95_abs_error", "max_abs_error", "over_5pp_count", "gate_pass_both"]])}
## Final Runtime-Covered 50-Shot Test Metrics

{markdown_table(final_metrics[["target", "selector", "model", "fit_scope", "sample_count", "calibration_image_count", "mae", "p95_abs_error", "max_abs_error", "over_5pp_count", "gate_pass_both"]] if not final_metrics.empty else final_metrics)}
## Best Runtime-Covered Hybrid Rows

{markdown_table(best_hybrid[["target", "split", "selector", "model", "fit_scope", "sample_count", "calibration_image_count", "mae", "p95_abs_error", "max_abs_error", "over_5pp_count", "gate_pass_both"]])}
## Conclusion

coverage_subset_final_selected_passes_gate = `{final_pass}`

strict_full_population_50shot_validation_available = `{len(missing) == 0}`
"""
    (output_dir / "audit.md").write_text(report, encoding="utf-8")
    write_output_manifest(data_dir / "output_manifest.json", output_dir=output_dir)
    print(final_metrics.to_string(index=False) if not final_metrics.empty else "no final metrics")
    print(f"coverage_subset_final_selected_passes_gate={final_pass}")
    print(f"covered_rows={len(covered)} missing_rows={len(missing)} full_strict_available={len(missing) == 0}")
    print(f"strict50_package={package_dir}")
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
