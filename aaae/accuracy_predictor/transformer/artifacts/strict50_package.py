"""Persist, verify, load, and evaluate Transformer strict-50 packages."""

from __future__ import annotations

from aaae.serialization import load_joblib

import ctypes
import errno
import hashlib
import json
import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd


PACKAGE_DATA_DIRNAME = "data"
PACKAGE_MODELS_DIRNAME = "models"
DEFAULT_COMMON_FEATURE_COUNT = 3103
DESIGN_INDEX_FEATURE_INDEX = 3090
FINAL_EVALUATION_JOURNAL_FILENAME = "final_evaluation_journal.json"


def _canonical_package_path(value: Path) -> Path:
    """Resolve a package target without following a final symlink."""

    requested = Path(value).expanduser()
    if requested.name in {"", ".", ".."}:
        raise ValueError(f"package_dir must name a directory, got {requested!s}")
    return requested.parent.resolve() / requested.name


@contextmanager
def _package_publication_lock(package_dir: Path) -> Iterator[None]:
    """Serialize writers while allowing readers to keep using the old package."""

    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - supported production hosts are POSIX.
        raise RuntimeError("strict50 package publication requires POSIX file locking") from exc

    package_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = package_dir.parent / f".{package_dir.name}.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FileExistsError(f"another process is publishing {package_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    """Make staged package contents durable before publishing its directory."""

    root = Path(root)
    directories: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file():
            _fsync_file(path)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted([*directories, root], key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _try_exchange_directories(staging_dir: Path, package_dir: Path) -> bool:
    """Atomically swap two same-filesystem directories when renameat2 is available."""

    if os.name != "posix":  # pragma: no cover - package training targets Linux CUDA hosts.
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:  # pragma: no cover - older POSIX runtimes.
        return False
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_exchange = 0x2
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging_dir),
        at_fdcwd,
        os.fsencode(package_dir),
        rename_exchange,
    )
    if result == 0:
        return True
    error_number = ctypes.get_errno()
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        return False
    raise OSError(error_number, os.strerror(error_number), str(package_dir))


def _publish_staged_package(staging_dir: Path, package_dir: Path) -> None:
    """Publish a verified staged package without exposing partially written contents."""

    staging_dir = Path(staging_dir)
    package_dir = Path(package_dir)
    parent = package_dir.parent
    if package_dir.exists() or package_dir.is_symlink():
        if package_dir.is_symlink() or not package_dir.is_dir():
            raise FileExistsError(f"refusing to replace non-directory package target: {package_dir}")
        if not (package_dir / "manifest.json").is_file():
            raise FileExistsError(f"refusing to replace unsealed package directory: {package_dir}")
        if _try_exchange_directories(staging_dir, package_dir):
            _fsync_directory(parent)
            shutil.rmtree(staging_dir, ignore_errors=True)
            return

        backup_dir = Path(tempfile.mkdtemp(prefix=f".{package_dir.name}.previous-", dir=parent))
        backup_dir.rmdir()
        os.replace(package_dir, backup_dir)
        try:
            os.replace(staging_dir, package_dir)
        except BaseException:
            os.replace(backup_dir, package_dir)
            _fsync_directory(parent)
            raise
        _fsync_directory(parent)
        shutil.rmtree(backup_dir, ignore_errors=True)
        return

    os.replace(staging_dir, package_dir)
    _fsync_directory(parent)


def normalize_common_features(
    features: np.ndarray,
    *,
    ignored_indices: tuple[int, ...] | list[int] = (),
) -> np.ndarray:
    """Return common features with source identifiers removed by package contract."""

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"common features must be a 2D matrix, got shape {matrix.shape}")
    normalized = matrix.copy()
    for index in ignored_indices:
        feature_index = int(index)
        if not 0 <= feature_index < normalized.shape[1]:
            raise ValueError(
                f"ignored common feature index {feature_index} is outside {normalized.shape[1]} columns"
            )
        normalized[:, feature_index] = 0.0
    return normalized


def aggregate_top1_components(
    component_matrix: np.ndarray,
    *,
    mode: str = "weighted_mean",
    weights: np.ndarray | None = None,
    sample50: np.ndarray | None = None,
    sample50_blend_weight: float = 0.0,
    component_clip_radius: float | None = None,
) -> np.ndarray:
    """Apply the package's deterministic robust aggregation rule."""

    components = np.asarray(component_matrix, dtype=np.float64)
    if components.ndim != 2 or components.shape[1] == 0:
        raise ValueError("component_matrix must have shape (rows, members) with at least one member")
    if not np.isfinite(components).all():
        raise ValueError("component predictions must be finite")

    if weights is None:
        weight_array = np.ones(components.shape[1], dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.shape != (components.shape[1],):
        raise ValueError("component weights must have one value per member")
    if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0) or float(weight_array.sum()) <= 0.0:
        raise ValueError("component weights must be finite, non-negative, and have a positive sum")

    normalized_weights = weight_array / weight_array.sum()
    if mode == "weighted_mean":
        center = np.average(components, axis=1, weights=normalized_weights)
    elif mode == "winsorized_mean":
        if component_clip_radius is None or float(component_clip_radius) <= 0.0:
            raise ValueError("winsorized_mean requires a positive component_clip_radius")
        median = np.median(components, axis=1)
        radius = float(component_clip_radius)
        clipped = np.clip(components, median[:, None] - radius, median[:, None] + radius)
        center = np.average(clipped, axis=1, weights=normalized_weights)
    elif mode == "median":
        center = np.median(components, axis=1)
    elif mode == "median_with_sample50":
        if sample50 is None:
            raise ValueError("median_with_sample50 aggregation requires sample50")
        sample = np.asarray(sample50, dtype=np.float64)
        if sample.shape != (components.shape[0],) or not np.isfinite(sample).all():
            raise ValueError("sample50 must be a finite vector with one value per row")
        center = np.median(np.column_stack([components, sample]), axis=1)
    else:
        raise ValueError(f"unknown top1 aggregation mode: {mode!r}")

    blend = float(sample50_blend_weight)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("sample50_blend_weight must be in [0, 1]")
    if blend > 0.0:
        if sample50 is None:
            raise ValueError("sample50 blending requires sample50")
        sample = np.asarray(sample50, dtype=np.float64)
        if sample.shape != (components.shape[0],) or not np.isfinite(sample).all():
            raise ValueError("sample50 must be a finite vector with one value per row")
        center = (1.0 - blend) * center + blend * sample
    return np.asarray(center, dtype=np.float64)


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_text(payload: Any) -> str:
    return json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, _json_text(payload))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _sha1_json(payload: Any) -> str:
    return _sha1_bytes(json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_asset_paths(package_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    """Return every persisted asset that defines a strict-50 package's behavior."""

    layout = manifest.get("model_layout", {})
    if not isinstance(layout, dict):
        raise ValueError("package manifest model_layout must be an object")
    relative_paths = {
        f"{PACKAGE_DATA_DIRNAME}/selected_images.csv",
        f"{PACKAGE_DATA_DIRNAME}/feature_schema.json",
        f"{PACKAGE_DATA_DIRNAME}/validation_predictions.csv",
    }
    for key in (
        "static_prior_top1_train",
        "static_prior_top1_refit",
        "static_prior_top5_train",
        "static_prior_top5_refit",
        "static_prior_top1_deployment",
        "static_prior_top5_deployment",
        "calibration",
    ):
        value = layout.get(key)
        if value is not None:
            relative_paths.add(str(value))
    for key in ("top1_train_models", "top1_refit_models", "top1_deployment_models"):
        values = layout.get(key, {})
        if not isinstance(values, dict):
            raise ValueError(f"package model_layout {key} must be an object")
        relative_paths.update(str(value) for value in values.values())
    if str(manifest.get("final_evaluation_status", "")) == "scored_once":
        relative_paths.update(
            {
                f"{PACKAGE_DATA_DIRNAME}/holdout_predictions.csv",
                f"{PACKAGE_DATA_DIRNAME}/holdout_metrics.json",
            }
        )

    root = package_dir.resolve()
    paths = []
    for relative in sorted(relative_paths):
        candidate = (package_dir / relative).resolve()
        if candidate.parent != root and root not in candidate.parents:
            raise ValueError(f"package asset escapes package directory: {relative}")
        if not candidate.is_file():
            raise FileNotFoundError(f"package integrity asset is missing: {candidate}")
        paths.append(candidate)
    return paths


def _package_content_integrity(package_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(package_dir).resolve()
    assets = {
        str(path.relative_to(root)): _sha1_file(path)
        for path in _package_asset_paths(root, manifest)
    }
    return {
        "algorithm": "sha1",
        "asset_hashes": assets,
        "content_hash": _sha1_json(assets),
    }


def _package_manifest_hash(manifest: dict[str, Any]) -> str:
    payload = {key: value for key, value in manifest.items() if key != "package_hash"}
    return _sha1_json(payload)


def verify_strict50_top1_package_integrity(package_dir: Path) -> dict[str, Any]:
    """Validate the sealed file inventory before deserializing package models.

    Historical packages predate the inventory and remain readable.  New sealed
    packages always carry ``content_integrity`` and therefore fail closed when
    any persisted model or strict-50 schema asset changes.
    """

    package_dir = Path(package_dir).resolve()
    manifest = _read_json(package_dir / "manifest.json")
    integrity = manifest.get("content_integrity")
    if integrity is None:
        return {"verified": False, "reason": "legacy_package_without_content_inventory"}
    if not isinstance(integrity, dict) or integrity.get("algorithm") != "sha1":
        raise ValueError("package integrity metadata is invalid")
    expected_assets = integrity.get("asset_hashes")
    if not isinstance(expected_assets, dict) or not expected_assets:
        raise ValueError("package integrity metadata has no asset hashes")
    actual = _package_content_integrity(package_dir, manifest)
    if actual["asset_hashes"] != {str(key): str(value) for key, value in expected_assets.items()}:
        raise ValueError("package integrity asset hash mismatch")
    if str(integrity.get("content_hash", "")) != str(actual["content_hash"]):
        raise ValueError("package integrity content hash mismatch")
    if str(manifest.get("package_hash", "")) != _package_manifest_hash(manifest):
        raise ValueError("package integrity manifest hash mismatch")
    return {
        "verified": True,
        "content_hash": str(actual["content_hash"]),
        "asset_count": int(len(actual["asset_hashes"])),
    }


def _save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path, compress=3)


def _load_joblib(path: Path) -> Any:
    return load_joblib(path)


def _predict_positionally(model: Any, features: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but .* was fitted with feature names",
            category=UserWarning,
        )
        return np.asarray(model.predict(features))


def _load_selected_images(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"selected image file is empty: {path}")
    for col in ("image_index_in_runtime_bank", "image_index"):
        if col in frame.columns:
            values = frame[col].to_numpy(np.int64)
            break
    else:
        raise KeyError(f"{path} has neither image_index_in_runtime_bank nor image_index")
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values.tolist():
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return np.asarray(ordered, dtype=np.int64)


def _feature_matrix(features: np.ndarray, *, expected_count: int, name: str) -> np.ndarray:
    matrix = np.asarray(features)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D feature matrix, got shape {matrix.shape}")
    if int(matrix.shape[1]) != int(expected_count):
        raise ValueError(f"{name} must have {expected_count} features, got {matrix.shape[1]}")
    return matrix


def _validate_model_feature_count(model: Any, *, expected_count: int, name: str) -> None:
    actual = getattr(model, "n_features_in_", None)
    if actual is not None and int(actual) != int(expected_count):
        raise ValueError(f"{name} expects {actual} features; package contract requires {expected_count}")


def _format_metric_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.12g}"
    return str(value)


def render_strict50_top1_readme(
    *,
    manifest: dict[str, Any],
    holdout_metrics: dict[str, Any],
    validation_metrics: dict[str, Any] | None = None,
) -> str:
    validation_metrics = validation_metrics or manifest.get("validation_metrics")
    future_data_contract = manifest.get("future_data_contract", {})
    label_columns = future_data_contract.get("new_labels_allowed_for_training", [])
    forbidden_labels = future_data_contract.get("forbidden_training_labels", [])
    readme_lines = [
        "# Strict 50-Shot Top1 Predictor Package",
        "",
        f"Package ID: `{manifest.get('package_id', 'unknown')}`",
        f"Created At: `{manifest.get('created_at', 'unknown')}`",
        f"Source Run: `{manifest.get('source_run_id', 'unknown')}`",
        f"Selected Images: `{manifest.get('selected_image_count', 'unknown')}`",
        f"Feature Count: `{manifest.get('feature_count', 'unknown')}`",
        f"Feature Schema Hash: `{manifest.get('feature_schema_hash', 'unknown')}`",
        f"Selected Images Hash: `{manifest.get('selected_images_hash', 'unknown')}`",
        f"Model Name: `{manifest.get('top1_model_name', 'unknown')}`",
    ]
    if manifest.get("input_run_id"):
        readme_lines.insert(5, f"Input Run: `{manifest.get('input_run_id')}`")
    if manifest.get("source_package_id"):
        readme_lines.append(f"Source Package ID: `{manifest.get('source_package_id')}`")
    if manifest.get("bank_root"):
        readme_lines.append(f"Bank Root: `{manifest.get('bank_root')}`")
    if manifest.get("bank_manifest"):
        bank_manifest = manifest.get("bank_manifest", {})
        if isinstance(bank_manifest, dict):
            readme_lines.append(f"Observation Bank: `{bank_manifest.get('bank_id', 'unknown')}`")
            readme_lines.append(f"Observation Bank Rows: `{bank_manifest.get('row_count', 'unknown')}`")
            readme_lines.append(f"Observation Bank Batches: `{bank_manifest.get('batch_count', 'unknown')}`")

    supplemental_contract = manifest.get("runtime_supplemental_feature_contract")
    if isinstance(supplemental_contract, dict):
        readme_lines.extend(
            [
                "",
                "## Runtime Feature Contract",
                "",
                f"Strict-50 feature count: `{manifest.get('runtime_base_feature_count', 'unknown')}`",
                f"Supplemental feature count: `{supplemental_contract.get('feature_count', 'unknown')}`",
                f"Supplemental block: `{supplemental_contract.get('name', 'unknown')}`",
                "",
                "Runtime-observation manifests must provide the supplemental matrix, feature-name sidecar, and metadata declared by this package. Missing, non-finite, misordered, or row-misaligned supplemental inputs are rejected.",
            ]
        )

    readme_lines.extend([""])
    if holdout_metrics:
        readme_lines.extend(
            [
                "## Holdout Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
    else:
        readme_lines.extend(
            [
                "## Final Evaluation",
                "",
                "Status: `pending`",
            ]
        )
    metric_keys = (
        "sample_count",
        "mae",
        "p95_abs_error",
        "max_abs_error",
        "over_3pp_count",
        "over_3pp_rate",
        "over_5pp_count",
        "rmse",
        "bias_pred_minus_actual",
        "gate_pass_max_3pp",
        "gate_pass_mean_2pp",
        "gate_pass_both",
        "gate_pass_max",
        "gate_pass_mean",
    )
    if holdout_metrics:
        for key in metric_keys:
            if key in holdout_metrics:
                readme_lines.append(f"| {key} | {_format_metric_value(holdout_metrics[key])} |")

    if validation_metrics:
        readme_lines.extend(
            [
                "",
                "## Validation Metrics",
                "",
                "| metric | value |",
                "| --- | ---: |",
            ]
        )
        for key in metric_keys:
            if key in validation_metrics:
                readme_lines.append(f"| {key} | {_format_metric_value(validation_metrics[key])} |")

    if future_data_contract:
        readme_lines.extend(
            [
                "",
                "## Future Data Contract",
                "",
                "This package is versioned for continual learning. New designs may be appended to the observation bank, but the following labels must not be used as training truth:",
            ]
        )
        if forbidden_labels:
            readme_lines.append("")
            readme_lines.append("Forbidden training labels:")
            for label in forbidden_labels:
                readme_lines.append(f"- `{label}`")
        if label_columns:
            readme_lines.append("")
            readme_lines.append("Allowed future high-fidelity labels:")
            for label in label_columns:
                readme_lines.append(f"- `{label}`")
    readme_lines.extend(
        [
            "",
            "## Package Contract",
            "",
            "The exported predictor must be reloaded with the same feature schema and selected 50 calibration images.  New data should be appended as a new observation batch and used for the next versioned retrain, not mixed silently into this package.",
        ]
    )
    return "\n".join(readme_lines).rstrip() + "\n"


@dataclass
class Strict50ShotTop1Predictor:
    package_dir: Path
    manifest: dict[str, Any]
    feature_names: list[str]
    selected_images: np.ndarray
    static_prior_top1_train: Any
    static_prior_top1_refit: Any
    static_prior_top5_train: Any
    static_prior_top5_refit: Any
    top1_train_models: dict[str, Any]
    top1_refit_models: dict[str, Any]
    bias_offset: float
    static_prior_top1_deployment: Any | None = None
    static_prior_top5_deployment: Any | None = None
    top1_deployment_models: dict[str, Any] | None = None
    clip_min: float = -0.05
    clip_max: float = 1.0

    @property
    def package_id(self) -> str:
        return str(self.manifest.get("package_id", self.package_dir.name))

    @property
    def top1_base_model_names(self) -> tuple[str, ...]:
        names = self.manifest.get("top1_base_model_names", [])
        return tuple(str(name) for name in names)

    @property
    def top1_model_target_bases(self) -> dict[str, str]:
        payload = self.manifest.get("top1_model_target_bases", {})
        return {str(name): str(value) for name, value in dict(payload).items()}

    @property
    def top1_model_weights(self) -> dict[str, float]:
        payload = self.manifest.get("top1_model_weights", {})
        return {str(name): float(value) for name, value in dict(payload).items()}

    @property
    def top1_aggregation_mode(self) -> str:
        return str(self.manifest.get("top1_aggregation_mode", "weighted_mean"))

    @property
    def top1_sample50_blend_weight(self) -> float:
        return float(self.manifest.get("top1_sample50_blend_weight", 0.0))

    @property
    def top1_component_clip_radius(self) -> float | None:
        value = self.manifest.get("top1_component_clip_radius")
        return None if value is None else float(value)

    @property
    def ignored_common_feature_indices(self) -> tuple[int, ...]:
        values = self.manifest.get("ignored_common_feature_indices", [])
        return tuple(int(value) for value in values)

    @property
    def feature_count(self) -> int:
        return int(len(self.feature_names))

    @property
    def runtime_base_feature_count(self) -> int:
        """Number of leading features constructed from strict-50 observations."""

        return int(self.manifest.get("runtime_base_feature_count", self.feature_count))

    @property
    def supplemental_feature_count(self) -> int:
        return int(self.feature_count - self.runtime_base_feature_count)

    @property
    def supplemental_feature_names(self) -> tuple[str, ...]:
        return tuple(self.feature_names[self.runtime_base_feature_count :])

    @property
    def common_feature_count(self) -> int:
        return int(self.manifest.get("common3103_feature_count", DEFAULT_COMMON_FEATURE_COUNT))

    @property
    def has_deployment_models(self) -> bool:
        return bool(
            self.static_prior_top1_deployment is not None
            and self.static_prior_top5_deployment is not None
            and self.top1_deployment_models
        )

    def resolve_model_scope(
        self,
        scope: str | None = None,
        *,
        use_refit: bool = True,
    ) -> str:
        """Resolve a requested model scope with legacy-package fallback."""

        requested = ("refit" if use_refit else "train") if scope is None else str(scope)
        if requested not in {"train", "refit", "deployment"}:
            raise ValueError(f"unknown strict50 model scope: {requested!r}")
        if requested == "deployment" and not self.has_deployment_models:
            return "refit"
        return requested

    def _top1_models_for_scope(self, scope: str) -> dict[str, Any]:
        if scope == "train":
            return self.top1_train_models
        if scope == "refit":
            return self.top1_refit_models
        if scope == "deployment":
            return dict(self.top1_deployment_models or {})
        raise ValueError(f"unknown strict50 model scope: {scope!r}")

    def _static_prior_for_scope(self, *, target: str, scope: str) -> Any:
        if target not in {"top1", "top5"}:
            raise ValueError(f"unknown static-prior target: {target!r}")
        if scope == "train":
            return self.static_prior_top1_train if target == "top1" else self.static_prior_top5_train
        if scope == "refit":
            return self.static_prior_top1_refit if target == "top1" else self.static_prior_top5_refit
        if scope == "deployment":
            return self.static_prior_top1_deployment if target == "top1" else self.static_prior_top5_deployment
        raise ValueError(f"unknown strict50 model scope: {scope!r}")

    def _feature_column(self, features: np.ndarray, name: str) -> np.ndarray:
        try:
            index = self.feature_names.index(name)
        except ValueError as exc:
            raise ValueError(f"package feature schema has no {name!r} column") from exc
        return np.asarray(features[:, index], dtype=np.float64)

    def _prediction_base(self, features: np.ndarray, target_base: str) -> np.ndarray:
        if target_base == "direct":
            return np.zeros(features.shape[0], dtype=np.float64)
        sample50 = self._feature_column(features, "sample50_top1_drop")
        if target_base == "sample50":
            return sample50
        if target_base == "prior":
            sample_minus_prior = self._feature_column(features, "sample50_top1_minus_prior")
            return sample50 - sample_minus_prior
        raise ValueError(f"unknown package top1 target base: {target_base!r}")

    def predict_top1_with_uncertainty(
        self,
        features: np.ndarray,
        *,
        use_refit: bool = True,
        scope: str | None = None,
    ) -> dict[str, Any]:
        features = _feature_matrix(features, expected_count=self.feature_count, name="strict50 features")
        requested_scope = ("refit" if use_refit else "train") if scope is None else str(scope)
        resolved_scope = self.resolve_model_scope(scope, use_refit=use_refit)
        models = self._top1_models_for_scope(resolved_scope)
        if not models:
            raise ValueError("top1 models are not loaded")
        names = self.top1_base_model_names
        if not names:
            names = tuple(sorted(models))
        target_bases = self.top1_model_target_bases
        configured_weights = self.top1_model_weights
        pred_parts = []
        weights = []
        for name in names:
            if name not in models:
                raise KeyError(f"missing top1 base model: {name}")
            correction = np.asarray(_predict_positionally(models[name], features), dtype=np.float64)
            base = self._prediction_base(features, target_bases.get(name, "direct"))
            pred_parts.append(base + correction)
            weights.append(float(configured_weights.get(name, 1.0)))
        if not pred_parts:
            raise ValueError("no top1 base predictions available")
        weight_array = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0) or float(weight_array.sum()) <= 0.0:
            raise ValueError("top1 model weights must be finite, non-negative, and have a positive sum")
        component_matrix = np.column_stack(pred_parts)
        normalized_weights = weight_array / weight_array.sum()
        sample50 = self._feature_column(features, "sample50_top1_drop")
        raw_prediction = aggregate_top1_components(
            component_matrix,
            mode=self.top1_aggregation_mode,
            weights=normalized_weights,
            sample50=sample50,
            sample50_blend_weight=self.top1_sample50_blend_weight,
            component_clip_radius=self.top1_component_clip_radius,
        )
        disagreement = np.sqrt(
            np.average(
                (component_matrix - np.average(component_matrix, axis=1, weights=normalized_weights)[:, None]) ** 2,
                axis=1,
                weights=normalized_weights,
            )
        )
        prediction = np.clip(raw_prediction + float(self.bias_offset), self.clip_min, self.clip_max)
        conformal_radius = float(self.manifest.get("conformal_radius_95", float("nan")))
        return {
            "prediction": prediction,
            "ensemble_disagreement": disagreement,
            "conformal_radius_95": conformal_radius,
            "conformal_lower_95": prediction - conformal_radius,
            "conformal_upper_95": prediction + conformal_radius,
            "component_names": names,
            "component_predictions": component_matrix,
            "requested_model_scope": requested_scope,
            "model_scope": resolved_scope,
            "deployment_fallback": bool(requested_scope == "deployment" and resolved_scope != "deployment"),
        }

    def predict_top1_from_features(
        self,
        features: np.ndarray,
        *,
        use_refit: bool = True,
        scope: str | None = None,
    ) -> np.ndarray:
        return np.asarray(
            self.predict_top1_with_uncertainty(
                features,
                use_refit=use_refit,
                scope=scope,
            )["prediction"],
            dtype=np.float64,
        )

    def predict_static_prior_top1(
        self,
        common3103_features: np.ndarray,
        *,
        use_refit: bool = True,
        scope: str | None = None,
    ) -> np.ndarray:
        common3103_features = _feature_matrix(
            common3103_features,
            expected_count=self.common_feature_count,
            name="common features",
        )
        common3103_features = self.prepare_common_features(common3103_features)
        resolved_scope = self.resolve_model_scope(scope, use_refit=use_refit)
        model = self._static_prior_for_scope(target="top1", scope=resolved_scope)
        pred = np.asarray(_predict_positionally(model, common3103_features), dtype=np.float64)
        return np.clip(pred, self.clip_min, self.clip_max)

    def predict_static_prior_top5(
        self,
        common3103_features: np.ndarray,
        *,
        use_refit: bool = True,
        scope: str | None = None,
    ) -> np.ndarray:
        common3103_features = _feature_matrix(
            common3103_features,
            expected_count=self.common_feature_count,
            name="common features",
        )
        common3103_features = self.prepare_common_features(common3103_features)
        resolved_scope = self.resolve_model_scope(scope, use_refit=use_refit)
        model = self._static_prior_for_scope(target="top5", scope=resolved_scope)
        pred = np.asarray(_predict_positionally(model, common3103_features), dtype=np.float64)
        return np.clip(pred, self.clip_min, self.clip_max)

    def prepare_common_features(self, common3103_features: np.ndarray) -> np.ndarray:
        common3103_features = _feature_matrix(
            common3103_features,
            expected_count=self.common_feature_count,
            name="common features",
        )
        return normalize_common_features(
            common3103_features,
            ignored_indices=self.ignored_common_feature_indices,
        )

    def predict_internal_priors(
        self,
        common3103_features: np.ndarray,
        *,
        use_refit: bool = True,
        scope: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the internal priors required to construct strict 50-shot features."""

        return (
            self.predict_static_prior_top1(
                common3103_features,
                use_refit=use_refit,
                scope=scope,
            ),
            self.predict_static_prior_top5(
                common3103_features,
                use_refit=use_refit,
                scope=scope,
            ),
        )

    def build_features_from_runtime_observations(
        self,
        *,
        common3103_features: np.ndarray,
        image_indices: np.ndarray,
        clean_top1_correct_bits: np.ndarray,
        clean_top5_correct_bits: np.ndarray,
        approx_top1_correct_bits: np.ndarray,
        approx_top5_correct_bits: np.ndarray,
        delta_margin_top1: np.ndarray,
        delta_margin_top5: np.ndarray,
        supplemental_features: np.ndarray | None = None,
        supplemental_feature_names: list[str] | tuple[str, ...] | None = None,
        scope: str = "deployment",
    ) -> np.ndarray:
        """Construct package-ordered features from explicit runtime observations."""

        common = self.prepare_common_features(common3103_features)
        observed_indices = np.asarray(image_indices, dtype=np.int64)
        if observed_indices.ndim != 1 or observed_indices.size == 0:
            raise ValueError("runtime observation image_indices must be a non-empty 1D array")
        if len(np.unique(observed_indices)) != len(observed_indices):
            raise ValueError("runtime observation image_indices must be unique")
        index_to_column = {int(value): index for index, value in enumerate(observed_indices.tolist())}
        missing = [int(value) for value in self.selected_images if int(value) not in index_to_column]
        if missing:
            raise ValueError(
                "runtime observations do not contain every package-selected image; "
                f"missing {missing[:10]}"
            )
        selected_columns = np.asarray(
            [index_to_column[int(value)] for value in self.selected_images],
            dtype=np.int64,
        )
        row_count = int(common.shape[0])
        image_count = int(observed_indices.size)

        def correctness_matrix(value: np.ndarray, name: str) -> np.ndarray:
            matrix = np.asarray(value, dtype=np.float32)
            if matrix.shape != (row_count, image_count):
                raise ValueError(
                    f"{name} has shape {matrix.shape}; expected {(row_count, image_count)}"
                )
            if not np.isfinite(matrix).all() or not np.isin(matrix, (0.0, 1.0)).all():
                raise ValueError(f"{name} must contain only finite binary correctness values")
            return matrix

        def margin_matrix(value: np.ndarray, name: str) -> np.ndarray:
            matrix = np.asarray(value, dtype=np.float32)
            if matrix.shape != (row_count, image_count):
                raise ValueError(
                    f"{name} has shape {matrix.shape}; expected {(row_count, image_count)}"
                )
            if not np.isfinite(matrix).all():
                raise ValueError(f"{name} must contain only finite values")
            return matrix

        def clean_vector(value: np.ndarray, name: str) -> np.ndarray:
            vector = np.asarray(value, dtype=np.float32)
            if (
                vector.ndim == 1
                and vector.shape != (image_count,)
                and int(observed_indices.min()) >= 0
                and int(observed_indices.max()) < len(vector)
            ):
                vector = vector[observed_indices]
            if vector.shape != (image_count,):
                raise ValueError(f"{name} has shape {vector.shape}; expected {(image_count,)}")
            if not np.isfinite(vector).all() or not np.isin(vector, (0.0, 1.0)).all():
                raise ValueError(f"{name} must contain only finite binary correctness values")
            return vector

        clean_top1 = clean_vector(clean_top1_correct_bits, "clean_top1_correct_bits")
        clean_top5 = clean_vector(clean_top5_correct_bits, "clean_top5_correct_bits")
        approx_top1 = correctness_matrix(approx_top1_correct_bits, "approx_top1_correct_bits")
        approx_top5 = correctness_matrix(approx_top5_correct_bits, "approx_top5_correct_bits")
        margin_top1 = margin_matrix(delta_margin_top1, "delta_margin_top1")
        margin_top5 = margin_matrix(delta_margin_top5, "delta_margin_top5")
        prior_top1, prior_top5 = self.predict_internal_priors(common, scope=scope)

        # Import lazily so loading the package contract does not pull the
        # calibration trainer (and sklearn) into unrelated inference imports.
        from aaae.accuracy_predictor.transformer.proxy_training.strict50_calibration import (
            build_50shot_features_from_selected,
        )

        features, names, _ = build_50shot_features_from_selected(
            base_x=common,
            image_indices=self.selected_images,
            clean_top1_selected=clean_top1[selected_columns],
            clean_top5_selected=clean_top5[selected_columns],
            approx_top1_selected=approx_top1[:, selected_columns],
            approx_top5_selected=approx_top5[:, selected_columns],
            delta_margin_top1_selected=margin_top1[:, selected_columns],
            delta_margin_top5_selected=margin_top5[:, selected_columns],
            prior_top1=prior_top1,
            prior_top5=prior_top5,
            include_individual=True,
        )
        expected_base_names = self.feature_names[: self.runtime_base_feature_count]
        if names != expected_base_names:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(names, expected_base_names, strict=False)
                    )
                    if actual != expected
                ),
                min(len(names), len(expected_base_names)),
            )
            raise ValueError(
                "runtime observations produced a feature schema that differs from the package "
                f"base segment at column {mismatch}; produced={len(names)} "
                f"package_base={len(expected_base_names)}"
            )

        supplemental_count = self.supplemental_feature_count
        if supplemental_count == 0:
            if supplemental_features is not None or supplemental_feature_names is not None:
                raise ValueError("this package does not accept supplemental runtime features")
            return features
        if supplemental_features is None or supplemental_feature_names is None:
            raise ValueError(
                f"this package requires {supplemental_count} supplemental runtime features"
            )
        supplemental = _feature_matrix(
            supplemental_features,
            expected_count=supplemental_count,
            name="supplemental runtime features",
        )
        if supplemental.shape[0] != row_count:
            raise ValueError(
                "supplemental runtime features must have the same rows as strict-50 "
                f"observations; got {supplemental.shape[0]} and {row_count}"
            )
        if not np.issubdtype(supplemental.dtype, np.number) or not np.isfinite(
            supplemental
        ).all():
            raise ValueError("supplemental runtime features must contain only finite numbers")
        actual_supplemental_names = [str(name) for name in supplemental_feature_names]
        expected_supplemental_names = list(self.supplemental_feature_names)
        if actual_supplemental_names != expected_supplemental_names:
            mismatch = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(
                            actual_supplemental_names,
                            expected_supplemental_names,
                            strict=False,
                        )
                    )
                    if actual != expected
                ),
                min(
                    len(actual_supplemental_names),
                    len(expected_supplemental_names),
                ),
            )
            raise ValueError(
                "supplemental runtime feature names differ from the package schema "
                f"at column {mismatch}; supplied={len(actual_supplemental_names)} "
                f"package={len(expected_supplemental_names)}"
            )
        combined = np.concatenate(
            (np.asarray(features, dtype=np.float32), np.asarray(supplemental, dtype=np.float32)),
            axis=1,
        )
        return _feature_matrix(
            combined,
            expected_count=self.feature_count,
            name="assembled package features",
        )

    def to_manifest(self) -> dict[str, Any]:
        return dict(self.manifest)


def _write_strict50_top1_package(
    *,
    package_dir: Path,
    manifest: dict[str, Any],
    selected_images: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    static_prior_top1_train: Any,
    static_prior_top1_refit: Any,
    static_prior_top5_train: Any,
    static_prior_top5_refit: Any,
    top1_train_models: dict[str, Any],
    top1_refit_models: dict[str, Any],
    bias_offset: float,
    validation_predictions: pd.DataFrame,
    holdout_predictions: pd.DataFrame | None = None,
    holdout_metrics: dict[str, Any] | None = None,
    static_prior_top1_deployment: Any | None = None,
    static_prior_top5_deployment: Any | None = None,
    top1_deployment_models: dict[str, Any] | None = None,
    readme_text: str | None = None,
    published_package_dir: Path | None = None,
) -> Path:
    package_dir = Path(package_dir)
    data_dir = package_dir / PACKAGE_DATA_DIRNAME
    models_dir = package_dir / PACKAGE_MODELS_DIRNAME
    package_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(selected_images, pd.DataFrame):
        selected_frame = selected_images.copy()
    else:
        selected_frame = pd.DataFrame({"image_index_in_runtime_bank": np.asarray(selected_images, dtype=np.int64)})
    selected_frame.to_csv(data_dir / "selected_images.csv", index=False)

    schema_payload = {
        "feature_count": int(len(feature_names)),
        "feature_names": list(feature_names),
        "feature_schema_hash": _sha1_json(feature_names),
    }
    _write_json(data_dir / "feature_schema.json", schema_payload)

    deployment_assets = (
        static_prior_top1_deployment is not None,
        static_prior_top5_deployment is not None,
        bool(top1_deployment_models),
    )
    if any(deployment_assets) and not all(deployment_assets):
        raise ValueError(
            "deployment package assets must include both static priors and at least one top1 model"
        )
    has_deployment_models = all(deployment_assets)

    model_layout = {
        "static_prior_top1_train": "models/static_prior_top1_train.joblib",
        "static_prior_top1_refit": "models/static_prior_top1_refit.joblib",
        "static_prior_top5_train": "models/static_prior_top5_train.joblib",
        "static_prior_top5_refit": "models/static_prior_top5_refit.joblib",
        "top1_train_models": {name: f"models/top1_train_{name}.joblib" for name in top1_train_models},
        "top1_refit_models": {name: f"models/top1_refit_{name}.joblib" for name in top1_refit_models},
        "calibration": "models/top1_calibration.json",
    }
    if has_deployment_models:
        model_layout.update(
            {
                "static_prior_top1_deployment": "models/static_prior_top1_deployment.joblib",
                "static_prior_top5_deployment": "models/static_prior_top5_deployment.joblib",
                "top1_deployment_models": {
                    name: f"models/top1_deployment_{name}.joblib"
                    for name in dict(top1_deployment_models or {})
                },
            }
        )

    manifest_out = {
        **_json_safe(manifest),
        "package_dir": str(published_package_dir or package_dir),
        "selected_images_hash": _sha1_bytes(selected_frame.to_csv(index=False).encode("utf-8")),
        "selected_image_count": int(len(selected_frame)),
        "feature_count": int(len(feature_names)),
        "feature_schema_hash": schema_payload["feature_schema_hash"],
        "bias_offset": float(bias_offset),
        "clip_min": -0.05,
        "clip_max": 1.0,
        "available_model_scopes": [
            "train",
            "refit",
            *(["deployment"] if has_deployment_models else []),
        ],
        "runtime_preferred_model_scope": "deployment" if has_deployment_models else "refit",
        "model_layout": model_layout,
    }

    _save_joblib(static_prior_top1_train, models_dir / "static_prior_top1_train.joblib")
    _save_joblib(static_prior_top1_refit, models_dir / "static_prior_top1_refit.joblib")
    _save_joblib(static_prior_top5_train, models_dir / "static_prior_top5_train.joblib")
    _save_joblib(static_prior_top5_refit, models_dir / "static_prior_top5_refit.joblib")
    for name, model in top1_train_models.items():
        _save_joblib(model, models_dir / f"top1_train_{name}.joblib")
    for name, model in top1_refit_models.items():
        _save_joblib(model, models_dir / f"top1_refit_{name}.joblib")
    if has_deployment_models:
        _save_joblib(
            static_prior_top1_deployment,
            models_dir / "static_prior_top1_deployment.joblib",
        )
        _save_joblib(
            static_prior_top5_deployment,
            models_dir / "static_prior_top5_deployment.joblib",
        )
        for name, model in dict(top1_deployment_models or {}).items():
            _save_joblib(model, models_dir / f"top1_deployment_{name}.joblib")

    _write_json(
        models_dir / "top1_calibration.json",
        {
            "bias_offset": float(bias_offset),
            "clip_min": -0.05,
            "clip_max": 1.0,
            "calibration_type": "bias",
        },
    )

    validation_predictions.to_csv(data_dir / "validation_predictions.csv", index=False)
    if holdout_predictions is not None:
        if holdout_metrics is None:
            raise ValueError("holdout_metrics are required when holdout_predictions are supplied")
        holdout_predictions.to_csv(data_dir / "holdout_predictions.csv", index=False)
        _write_json(data_dir / "holdout_metrics.json", holdout_metrics)
    elif holdout_metrics:
        raise ValueError("holdout_metrics require holdout_predictions")

    manifest_out["content_integrity"] = _package_content_integrity(package_dir, manifest_out)
    manifest_out["package_hash"] = _package_manifest_hash(manifest_out)
    _write_json(package_dir / "manifest.json", manifest_out)

    readme_path = package_dir / "README.md"
    readme_payload = readme_text or render_strict50_top1_readme(
        manifest=manifest_out,
        holdout_metrics=holdout_metrics or {},
        validation_metrics=manifest_out.get("validation_metrics"),
    )
    readme_path.write_text(readme_payload, encoding="utf-8")

    return package_dir


def save_strict50_top1_package(
    *,
    package_dir: Path,
    manifest: dict[str, Any],
    selected_images: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    static_prior_top1_train: Any,
    static_prior_top1_refit: Any,
    static_prior_top5_train: Any,
    static_prior_top5_refit: Any,
    top1_train_models: dict[str, Any],
    top1_refit_models: dict[str, Any],
    bias_offset: float,
    validation_predictions: pd.DataFrame,
    holdout_predictions: pd.DataFrame | None = None,
    holdout_metrics: dict[str, Any] | None = None,
    static_prior_top1_deployment: Any | None = None,
    static_prior_top5_deployment: Any | None = None,
    top1_deployment_models: dict[str, Any] | None = None,
    readme_text: str | None = None,
) -> Path:
    """Write and atomically publish a sealed strict-50 package.

    The final path is never used as a workspace.  A concurrent reader therefore
    observes either the previously sealed package or the fully verified new one.
    """

    target_dir = _canonical_package_path(package_dir)
    with _package_publication_lock(target_dir):
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{target_dir.name}.staging-", dir=target_dir.parent)
        )
        try:
            _write_strict50_top1_package(
                package_dir=staging_dir,
                manifest=manifest,
                selected_images=selected_images,
                feature_names=feature_names,
                static_prior_top1_train=static_prior_top1_train,
                static_prior_top1_refit=static_prior_top1_refit,
                static_prior_top5_train=static_prior_top5_train,
                static_prior_top5_refit=static_prior_top5_refit,
                top1_train_models=top1_train_models,
                top1_refit_models=top1_refit_models,
                bias_offset=bias_offset,
                validation_predictions=validation_predictions,
                holdout_predictions=holdout_predictions,
                holdout_metrics=holdout_metrics,
                static_prior_top1_deployment=static_prior_top1_deployment,
                static_prior_top5_deployment=static_prior_top5_deployment,
                top1_deployment_models=top1_deployment_models,
                readme_text=readme_text,
                published_package_dir=target_dir,
            )
            verify_strict50_top1_package_integrity(staging_dir)
            _fsync_tree(staging_dir)
            _publish_staged_package(staging_dir, target_dir)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise
    return target_dir


def _validate_final_evaluation_predictions(predictions: pd.DataFrame) -> None:
    required = {"split", "actual", "pred", "abs_error"}
    missing = required.difference(predictions.columns)
    if missing:
        raise KeyError(f"final evaluation predictions are missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("final evaluation predictions must not be empty")
    numeric = predictions.loc[:, ["actual", "pred", "abs_error"]].apply(pd.to_numeric, errors="raise")
    values = numeric.to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("final evaluation actual, prediction, and absolute-error values must be finite")
    expected_abs_error = np.abs(
        numeric["pred"].to_numpy(np.float64) - numeric["actual"].to_numpy(np.float64)
    )
    if not np.allclose(
        numeric["abs_error"].to_numpy(np.float64),
        expected_abs_error,
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("final evaluation abs_error must equal abs(pred - actual)")


def _validate_final_evaluation_metrics(metrics: dict[str, Any]) -> None:
    if not isinstance(metrics, dict):
        raise TypeError("final evaluation metrics must be an object")

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
            raise ValueError("final evaluation metrics must contain only finite numeric values")

    visit(_json_safe(metrics))


def _final_evaluation_journal_path(package_dir: Path) -> Path:
    return Path(package_dir) / PACKAGE_DATA_DIRNAME / FINAL_EVALUATION_JOURNAL_FILENAME


def _prepared_final_evaluation_journal(
    *,
    predictions: pd.DataFrame,
    holdout_metrics: dict[str, Any],
    evaluation_metadata: dict[str, Any],
    package_pre_score_hash: str,
    package_pre_score_content_hash: str,
) -> dict[str, Any]:
    _validate_final_evaluation_metrics(holdout_metrics)
    prediction_csv = predictions.to_csv(index=False, lineterminator="\n")
    metrics_json = _json_text(holdout_metrics)
    return {
        "journal_version": 2,
        "phase": "prepared_without_additional_label_access",
        "package_pre_score_hash": str(package_pre_score_hash),
        "package_pre_score_content_hash": str(package_pre_score_content_hash),
        "prediction_csv": prediction_csv,
        "prediction_csv_sha1": _sha1_bytes(prediction_csv.encode("utf-8")),
        "metrics_json": metrics_json,
        "metrics_json_sha1": _sha1_bytes(metrics_json.encode("utf-8")),
        "holdout_metrics": _json_safe(holdout_metrics),
        "evaluation_metadata": _json_safe(evaluation_metadata),
    }


def _read_prepared_final_evaluation_journal(journal_path: Path) -> dict[str, Any]:
    journal = _read_json(journal_path)
    required = {
        "journal_version",
        "phase",
        "prediction_csv",
        "prediction_csv_sha1",
        "metrics_json",
        "metrics_json_sha1",
        "holdout_metrics",
        "evaluation_metadata",
        "package_pre_score_hash",
        "package_pre_score_content_hash",
    }
    missing = required.difference(journal)
    if missing or int(journal.get("journal_version", -1)) != 2:
        raise ValueError(f"final evaluation journal is invalid: missing={sorted(missing)}")
    if str(journal.get("phase", "")) != "prepared_without_additional_label_access":
        raise ValueError("final evaluation journal is not recoverable")
    prediction_csv = str(journal["prediction_csv"])
    metrics_json = str(journal["metrics_json"])
    if _sha1_bytes(prediction_csv.encode("utf-8")) != str(journal["prediction_csv_sha1"]):
        raise ValueError("final evaluation journal prediction payload hash differs")
    if _sha1_bytes(metrics_json.encode("utf-8")) != str(journal["metrics_json_sha1"]):
        raise ValueError("final evaluation journal metrics payload hash differs")
    if not isinstance(journal["holdout_metrics"], dict) or not isinstance(journal["evaluation_metadata"], dict):
        raise ValueError("final evaluation journal metadata is invalid")
    try:
        parsed_metrics = json.loads(metrics_json)
    except json.JSONDecodeError as exc:
        raise ValueError("final evaluation journal metrics payload is not valid JSON") from exc
    if not isinstance(parsed_metrics, dict):
        raise ValueError("final evaluation journal metrics payload must be an object")
    _validate_final_evaluation_metrics(parsed_metrics)
    if _json_text(parsed_metrics) != metrics_json or _json_text(journal["holdout_metrics"]) != metrics_json:
        raise ValueError("final evaluation journal metrics copies differ")
    if not str(journal["package_pre_score_hash"]) or not str(journal["package_pre_score_content_hash"]):
        raise ValueError("final evaluation journal is missing package pre-score identity")
    return journal


def recover_strict50_top1_package_evaluation(package_dir: Path) -> dict[str, Any]:
    """Complete a journaled final evaluation without reopening any label source."""

    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    data_dir = package_dir / PACKAGE_DATA_DIRNAME
    journal_path = _final_evaluation_journal_path(package_dir)
    if not journal_path.exists():
        raise FileNotFoundError(f"{package_dir} has no recoverable final evaluation journal")
    manifest = _read_json(manifest_path)
    status = str(manifest.get("final_evaluation_status", ""))
    if status == "scored_once":
        verify_strict50_top1_package_integrity(package_dir)
        _atomic_write_text(
            package_dir / "README.md",
            render_strict50_top1_readme(
                manifest=manifest,
                holdout_metrics=dict(manifest.get("holdout_metrics", {})),
                validation_metrics=manifest.get("validation_metrics"),
            ),
        )
        journal_path.unlink(missing_ok=True)
        _fsync_directory(data_dir)
        return manifest
    if not status.startswith("pending"):
        raise ValueError(f"package is not pending final evaluation: {status or 'missing status'}")
    integrity = verify_strict50_top1_package_integrity(package_dir)
    journal = _read_prepared_final_evaluation_journal(journal_path)
    if str(journal["package_pre_score_hash"]) != str(manifest.get("package_hash", "")):
        raise ValueError("final evaluation journal package pre-score hash differs")
    if str(journal["package_pre_score_content_hash"]) != str(integrity.get("content_hash", "")):
        raise ValueError("final evaluation journal package pre-score content hash differs")
    holdout_path = data_dir / "holdout_predictions.csv"
    metrics_path = data_dir / "holdout_metrics.json"
    _atomic_write_text(holdout_path, str(journal["prediction_csv"]))
    _atomic_write_text(metrics_path, str(journal["metrics_json"]))
    predictions = pd.read_csv(holdout_path)
    _validate_final_evaluation_predictions(predictions)

    manifest["holdout_metrics"] = dict(journal["holdout_metrics"])
    manifest["test_splits"] = sorted(predictions["split"].astype(str).unique().tolist())
    manifest["final_evaluation_status"] = "scored_once"
    manifest["package_status"] = "sealed_external_evaluation_scored_once"
    manifest["final_evaluation"] = dict(journal["evaluation_metadata"])
    manifest["content_integrity"] = _package_content_integrity(package_dir, manifest)
    manifest["package_hash"] = _package_manifest_hash(manifest)
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_text(
        package_dir / "README.md",
        render_strict50_top1_readme(
            manifest=manifest,
            holdout_metrics=dict(journal["holdout_metrics"]),
            validation_metrics=manifest.get("validation_metrics"),
        ),
    )
    journal_path.unlink(missing_ok=True)
    _fsync_directory(data_dir)
    return manifest


def finalize_strict50_top1_package_evaluation(
    *,
    package_dir: Path,
    holdout_predictions: pd.DataFrame,
    holdout_metrics: dict[str, Any],
    evaluation_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Journal then atomically complete one labelled final package evaluation.

    The journal contains enough already-derived output to recover a process or
    power interruption without ever reopening the final label source.
    """

    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    data_dir = package_dir / PACKAGE_DATA_DIRNAME
    journal_path = _final_evaluation_journal_path(package_dir)
    holdout_path = data_dir / "holdout_predictions.csv"
    metrics_path = data_dir / "holdout_metrics.json"
    if journal_path.exists():
        return recover_strict50_top1_package_evaluation(package_dir)
    if holdout_path.exists() or metrics_path.exists():
        raise FileExistsError(f"{package_dir} already has a final evaluation")
    manifest = _read_json(manifest_path)
    integrity = verify_strict50_top1_package_integrity(package_dir)
    status = str(manifest.get("final_evaluation_status", ""))
    if not status.startswith("pending"):
        raise ValueError(f"package is not pending final evaluation: {status or 'missing status'}")
    predictions = pd.DataFrame(holdout_predictions).copy()
    _validate_final_evaluation_predictions(predictions)
    _atomic_write_json(
        journal_path,
        _prepared_final_evaluation_journal(
            predictions=predictions,
            holdout_metrics=holdout_metrics,
            evaluation_metadata=evaluation_metadata,
            package_pre_score_hash=str(manifest.get("package_hash", "")),
            package_pre_score_content_hash=str(integrity.get("content_hash", "")),
        ),
    )
    return recover_strict50_top1_package_evaluation(package_dir)


def load_strict50_top1_package(package_dir: Path) -> Strict50ShotTop1Predictor:
    package_dir = Path(package_dir)
    verify_strict50_top1_package_integrity(package_dir)
    manifest = _read_json(package_dir / "manifest.json")
    data_dir = package_dir / PACKAGE_DATA_DIRNAME
    models_dir = package_dir / PACKAGE_MODELS_DIRNAME
    schema = _read_json(data_dir / "feature_schema.json")
    feature_names = [str(item) for item in schema["feature_names"]]
    feature_schema_hash = _sha1_json(feature_names)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("package feature names must be non-empty and unique")
    if int(schema.get("feature_count", -1)) != len(feature_names):
        raise ValueError("package feature schema count differs from its feature names")
    if str(schema.get("feature_schema_hash", "")) != feature_schema_hash:
        raise ValueError("package feature schema hash differs from its feature names")
    if int(manifest.get("feature_count", len(feature_names))) != len(feature_names):
        raise ValueError("package manifest feature count differs from its feature schema")
    if str(manifest.get("feature_schema_hash", feature_schema_hash)) != feature_schema_hash:
        raise ValueError("package manifest feature hash differs from its feature schema")

    runtime_base_feature_count = int(
        manifest.get("runtime_base_feature_count", len(feature_names))
    )
    if not 1 <= runtime_base_feature_count <= len(feature_names):
        raise ValueError("package runtime_base_feature_count is outside its feature schema")
    supplemental_count = len(feature_names) - runtime_base_feature_count
    supplemental_contract = manifest.get("runtime_supplemental_feature_contract")
    if supplemental_count:
        if not isinstance(supplemental_contract, dict):
            raise ValueError(
                "package with supplemental features requires a runtime supplemental contract"
            )
        if int(supplemental_contract.get("feature_count", -1)) != supplemental_count:
            raise ValueError(
                "package supplemental feature contract count differs from its feature schema"
            )
        for key in (
            "name",
            "matrix_manifest_key",
            "names_manifest_key",
            "metadata_manifest_key",
        ):
            if not isinstance(supplemental_contract.get(key), str) or not str(
                supplemental_contract[key]
            ).strip():
                raise ValueError(
                    f"package supplemental feature contract requires non-empty {key!r}"
                )
    elif supplemental_contract is not None:
        raise ValueError(
            "package without supplemental features must not declare a supplemental contract"
        )
    selected_images = _load_selected_images(data_dir / "selected_images.csv")
    calibration = _read_json(models_dir / "top1_calibration.json")
    bias_offset = float(calibration["bias_offset"])

    static_prior_top1_train = _load_joblib(models_dir / "static_prior_top1_train.joblib")
    static_prior_top1_refit = _load_joblib(models_dir / "static_prior_top1_refit.joblib")
    static_prior_top5_train = _load_joblib(models_dir / "static_prior_top5_train.joblib")
    static_prior_top5_refit = _load_joblib(models_dir / "static_prior_top5_refit.joblib")

    top1_train_models: dict[str, Any] = {}
    top1_refit_models: dict[str, Any] = {}
    top1_deployment_models: dict[str, Any] = {}
    model_layout = manifest.get("model_layout", {})
    train_model_names = model_layout.get("top1_train_models", {})
    refit_model_names = model_layout.get("top1_refit_models", {})
    deployment_model_names = model_layout.get("top1_deployment_models", {})
    for name, relpath in train_model_names.items():
        top1_train_models[str(name)] = _load_joblib(package_dir / str(relpath))
    for name, relpath in refit_model_names.items():
        top1_refit_models[str(name)] = _load_joblib(package_dir / str(relpath))
    for name, relpath in deployment_model_names.items():
        top1_deployment_models[str(name)] = _load_joblib(package_dir / str(relpath))

    static_prior_top1_deployment = None
    static_prior_top5_deployment = None
    deployment_top1_path = model_layout.get("static_prior_top1_deployment")
    deployment_top5_path = model_layout.get("static_prior_top5_deployment")
    deployment_layout_flags = (
        deployment_top1_path is not None,
        deployment_top5_path is not None,
        bool(deployment_model_names),
    )
    if any(deployment_layout_flags) and not all(deployment_layout_flags):
        raise ValueError("package manifest contains incomplete deployment model layout")
    if all(deployment_layout_flags):
        static_prior_top1_deployment = _load_joblib(package_dir / str(deployment_top1_path))
        static_prior_top5_deployment = _load_joblib(package_dir / str(deployment_top5_path))

    common_feature_count = int(manifest.get("common3103_feature_count", DEFAULT_COMMON_FEATURE_COUNT))
    for name, model in {
        "static_prior_top1_train": static_prior_top1_train,
        "static_prior_top1_refit": static_prior_top1_refit,
        "static_prior_top5_train": static_prior_top5_train,
        "static_prior_top5_refit": static_prior_top5_refit,
        **(
            {
                "static_prior_top1_deployment": static_prior_top1_deployment,
                "static_prior_top5_deployment": static_prior_top5_deployment,
            }
            if static_prior_top1_deployment is not None
            else {}
        ),
    }.items():
        _validate_model_feature_count(model, expected_count=common_feature_count, name=name)
    for scope, models in (
        ("train", top1_train_models),
        ("refit", top1_refit_models),
        ("deployment", top1_deployment_models),
    ):
        for name, model in models.items():
            _validate_model_feature_count(model, expected_count=len(feature_names), name=f"top1_{scope}_{name}")

    return Strict50ShotTop1Predictor(
        package_dir=package_dir,
        manifest=manifest,
        feature_names=feature_names,
        selected_images=selected_images,
        static_prior_top1_train=static_prior_top1_train,
        static_prior_top1_refit=static_prior_top1_refit,
        static_prior_top5_train=static_prior_top5_train,
        static_prior_top5_refit=static_prior_top5_refit,
        top1_train_models=top1_train_models,
        top1_refit_models=top1_refit_models,
        bias_offset=bias_offset,
        static_prior_top1_deployment=static_prior_top1_deployment,
        static_prior_top5_deployment=static_prior_top5_deployment,
        top1_deployment_models=top1_deployment_models or None,
        clip_min=float(calibration.get("clip_min", -0.05)),
        clip_max=float(calibration.get("clip_max", 1.0)),
    )
