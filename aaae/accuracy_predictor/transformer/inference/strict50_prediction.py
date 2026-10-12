"""Strict-50 Transformer top-1 prediction from features or runtime observations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any

PREDICTION_MANIFEST_SCHEMA = "transformer_strict50_prediction_manifest_v1"
MODEL_SCOPES = frozenset({"train", "refit", "deployment"})


def _manifest_path(manifest_path: Path, payload: dict[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"prediction manifest requires a non-empty {key!r} path")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _validate_runtime_identity(*, predictor: Any, payload: dict[str, Any]) -> None:
    contract = predictor.manifest.get("runtime_identity_contract")
    if contract is None:
        return
    if not isinstance(contract, dict) or not contract:
        raise ValueError("package runtime identity contract must be a non-empty object")
    supplied = payload.get("runtime_identity")
    if not isinstance(supplied, dict):
        raise ValueError(
            "prediction manifest requires a runtime_identity object for this package"
        )
    mismatches = {
        str(key): {"supplied": supplied.get(str(key)), "expected": expected}
        for key, expected in contract.items()
        if supplied.get(str(key)) != expected
    }
    if mismatches:
        raise ValueError(
            "prediction runtime identity differs from the package contract: "
            f"{mismatches}"
        )


def _load_npy_matrix(path: Path, *, name: str) -> Any:
    import numpy as np

    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.ndarray):
        loaded.close()
        raise TypeError(f"{name} must point to one .npy array, not an NPZ archive")
    matrix = np.asarray(loaded)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional, got {matrix.shape}")
    return matrix


def _load_feature_names(path: Path, *, name: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("feature_names")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{name} must contain a non-empty JSON feature-name list")
    names = [str(value) for value in payload]
    if any(not value for value in names) or len(set(names)) != len(names):
        raise ValueError(f"{name} must contain unique non-empty feature names")
    return names


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_supplemental_features(
    *,
    predictor: Any,
    manifest_path: Path,
    payload: dict[str, Any],
    observations: Any,
) -> tuple[Any | None, list[str] | None]:
    contract = predictor.manifest.get("runtime_supplemental_feature_contract")
    if predictor.supplemental_feature_count == 0:
        return None, None
    if not isinstance(contract, dict):
        raise ValueError("joint package has no runtime supplemental feature contract")

    matrix_key = str(contract["matrix_manifest_key"])
    names_key = str(contract["names_manifest_key"])
    metadata_key = str(contract["metadata_manifest_key"])
    matrix_path = _manifest_path(manifest_path, payload, matrix_key)
    names_path = _manifest_path(manifest_path, payload, names_key)
    metadata_path = _manifest_path(manifest_path, payload, metadata_key)
    matrix = _load_npy_matrix(matrix_path, name=f"prediction manifest {matrix_key}")
    names = _load_feature_names(names_path, name=f"prediction manifest {names_key}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(
            f"prediction manifest {metadata_key} must contain a JSON object"
        )

    if int(metadata.get("rows", -1)) != int(matrix.shape[0]):
        raise ValueError(
            "supplemental feature metadata row count differs from its matrix"
        )
    if int(metadata.get("feature_count", -1)) != int(matrix.shape[1]):
        raise ValueError("supplemental feature metadata width differs from its matrix")
    if int(matrix.shape[1]) != predictor.supplemental_feature_count:
        raise ValueError(
            "supplemental feature matrix width differs from the package contract; "
            f"got {matrix.shape[1]}, expected {predictor.supplemental_feature_count}"
        )
    if str(metadata.get("output_sha256", "")) != _sha256_file(matrix_path):
        raise ValueError(
            "supplemental feature matrix SHA-256 differs from its metadata"
        )
    if str(metadata.get("feature_names_sha256", "")) != _sha256_file(names_path):
        raise ValueError("supplemental feature-name SHA-256 differs from its metadata")

    required_metadata = contract.get("required_metadata", {})
    if not isinstance(required_metadata, dict):
        raise ValueError("package supplemental required_metadata must be an object")
    for key, expected in required_metadata.items():
        if metadata.get(str(key)) != expected:
            raise ValueError(
                f"supplemental feature metadata {key!r} differs from the package contract"
            )

    row_alignment = contract.get("row_alignment", {})
    if not isinstance(row_alignment, dict):
        raise ValueError("package supplemental row_alignment must be an object")
    for archive_key, metadata_hash_key in (
        (
            row_alignment.get("runtime_design_id_archive_key"),
            row_alignment.get("metadata_design_ids_sha256_key"),
        ),
        (
            row_alignment.get("runtime_row_index_archive_key"),
            row_alignment.get("metadata_row_indices_sha256_key"),
        ),
    ):
        if not isinstance(archive_key, str) or not archive_key:
            raise ValueError(
                "package supplemental row alignment archive key is invalid"
            )
        if not isinstance(metadata_hash_key, str) or not metadata_hash_key:
            raise ValueError(
                "package supplemental row alignment metadata key is invalid"
            )
        if archive_key not in observations.files:
            raise KeyError(
                f"runtime observation archive is missing row identity {archive_key!r}"
            )
        actual_hash = _sha256_array(observations[archive_key])
        if str(metadata.get(metadata_hash_key, "")) != actual_hash:
            raise ValueError(
                f"supplemental feature rows do not align with runtime {archive_key!r}"
            )
    return matrix, names


def _observation_value(archive: Any, *names: str) -> Any:
    for name in names:
        if name in archive.files:
            return archive[name]
    raise KeyError(f"runtime observation archive is missing one of {list(names)}")


def build_features_from_manifest(
    *,
    predictor: Any,
    manifest_path: Path,
    scope: str,
) -> Any:
    """Build package-ordered features from a versioned runtime manifest."""
    import numpy as np

    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("prediction manifest root must be a JSON object")
    if payload.get("schema_version") != PREDICTION_MANIFEST_SCHEMA:
        raise ValueError(
            "unsupported prediction manifest schema: "
            f"{payload.get('schema_version')!r}; expected {PREDICTION_MANIFEST_SCHEMA!r}"
        )
    _validate_runtime_identity(predictor=predictor, payload=payload)
    common_path = _manifest_path(manifest_path, payload, "common_features")
    observations_path = _manifest_path(manifest_path, payload, "runtime_observations")
    common = _load_npy_matrix(common_path, name="prediction manifest common_features")

    with ExitStack() as stack:
        observations = stack.enter_context(
            np.load(observations_path, allow_pickle=False)
        )
        clean_source = observations
        if payload.get("bank_meta") is not None:
            bank_meta_path = _manifest_path(manifest_path, payload, "bank_meta")
            identity_contract = predictor.manifest.get("runtime_identity_contract")
            if isinstance(identity_contract, dict) and identity_contract.get(
                "bank_meta_sha1"
            ) not in (None, ""):
                actual_bank_meta_sha1 = _sha1_file(bank_meta_path)
                if actual_bank_meta_sha1 != str(identity_contract["bank_meta_sha1"]):
                    raise ValueError(
                        "prediction bank_meta SHA-1 differs from the package runtime "
                        "identity contract"
                    )
            clean_source = stack.enter_context(
                np.load(bank_meta_path, allow_pickle=False)
            )
        elif isinstance(
            predictor.manifest.get("runtime_identity_contract"), dict
        ):
            raise ValueError(
                "prediction manifest requires bank_meta for this identity-bound package"
            )
        image_indices = (
            np.asarray(observations["image_indices"], dtype=np.int64)
            if "image_indices" in observations.files
            else np.arange(
                np.asarray(
                    _observation_value(observations, "approx_top1_correct_bits")
                ).shape[1],
                dtype=np.int64,
            )
        )
        supplemental_features, supplemental_feature_names = _load_supplemental_features(
            predictor=predictor,
            manifest_path=manifest_path,
            payload=payload,
            observations=observations,
        )
        return predictor.build_features_from_runtime_observations(
            common3103_features=common,
            image_indices=image_indices,
            clean_top1_correct_bits=_observation_value(
                clean_source,
                "clean_top1_correct_bits",
                "exact_top1_correct_bits",
            ),
            clean_top5_correct_bits=_observation_value(
                clean_source,
                "clean_top5_correct_bits",
                "exact_top5_correct_bits",
            ),
            approx_top1_correct_bits=_observation_value(
                observations,
                "approx_top1_correct_bits",
            ),
            approx_top5_correct_bits=_observation_value(
                observations,
                "approx_top5_correct_bits",
            ),
            delta_margin_top1=_observation_value(observations, "delta_margin_top1"),
            delta_margin_top5=_observation_value(observations, "delta_margin_top5"),
            supplemental_features=supplemental_features,
            supplemental_feature_names=supplemental_feature_names,
            scope=scope,
        )


def _write_predictions_atomic(frame: Any, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_csv.name}.tmp-",
        suffix=".csv",
        dir=output_csv.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output_csv)
    finally:
        if temporary.exists():
            temporary.unlink()


def predict_top1(
    *,
    package_dir: Path,
    output_csv: Path,
    features_path: Path | None = None,
    manifest_path: Path | None = None,
    scope: str = "deployment",
) -> dict[str, Any]:
    """Load a strict-50 package, predict top-1 drop, and atomically publish CSV."""
    import numpy as np
    import pandas as pd

    from aaae.accuracy_predictor.transformer.artifacts.strict50_package import load_strict50_top1_package

    if (features_path is None) == (manifest_path is None):
        raise ValueError("exactly one of features_path or manifest_path is required")
    if scope not in MODEL_SCOPES:
        raise ValueError(f"scope must be one of {sorted(MODEL_SCOPES)}; got {scope!r}")

    package_dir = Path(package_dir)
    output_csv = Path(output_csv)
    predictor = load_strict50_top1_package(package_dir)
    if features_path is not None:
        features_path = Path(features_path)
        features = _load_npy_matrix(features_path, name="--features")
        input_mode = "package_ready_features"
        input_source = str(features_path.resolve())
    else:
        assert manifest_path is not None
        manifest_path = Path(manifest_path)
        features = build_features_from_manifest(
            predictor=predictor,
            manifest_path=manifest_path,
            scope=scope,
        )
        input_mode = "runtime_observation_manifest"
        input_source = str(manifest_path.resolve())

    result = predictor.predict_top1_with_uncertainty(features, scope=scope)
    component_names = tuple(str(name) for name in result["component_names"])
    component_predictions = np.asarray(
        result["component_predictions"], dtype=np.float64
    )
    frame = pd.DataFrame(
        {
            "row_index": np.arange(features.shape[0], dtype=np.int64),
            "top1_drop_prediction": np.asarray(result["prediction"], dtype=np.float64),
            "ensemble_disagreement": np.asarray(
                result["ensemble_disagreement"],
                dtype=np.float64,
            ),
            "conformal_lower_95": np.asarray(
                result["conformal_lower_95"], dtype=np.float64
            ),
            "conformal_upper_95": np.asarray(
                result["conformal_upper_95"], dtype=np.float64
            ),
        }
    )
    for index, name in enumerate(component_names):
        frame[f"component_{name}"] = component_predictions[:, index]
    _write_predictions_atomic(frame, output_csv)

    return {
        "package_dir": str(package_dir.resolve()),
        "package_id": predictor.package_id,
        "input_mode": input_mode,
        "input_source": input_source,
        "features": str(features_path.resolve()) if features_path is not None else None,
        "manifest": str(manifest_path.resolve()) if manifest_path is not None else None,
        "feature_count": int(features.shape[1]),
        "runtime_base_feature_count": int(predictor.runtime_base_feature_count),
        "supplemental_feature_count": int(predictor.supplemental_feature_count),
        "row_count": int(features.shape[0]),
        "requested_model_scope": str(result["requested_model_scope"]),
        "model_scope": str(result["model_scope"]),
        "deployment_fallback": bool(result["deployment_fallback"]),
        "output_csv": str(output_csv.resolve()),
    }
