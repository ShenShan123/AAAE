from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .behavior import BEHAVIOR_HASH_COLUMN, behavior_sha256_rows
from .contracts import ApplicationProfile, DataContractError, MultiplierDesignBatch
from .errors import ManifestError
from .features import DEFAULT_FEATURE_SCHEMA, SUPPORTED_FEATURE_SCHEMAS


SCHEMA_VERSION = 1
TARGETS = ("top1_drop", "top5_drop")
TARGET_UNITS = "proportion"
SPLITS = ("train", "validation", "test")
SPLIT_GROUPINGS = frozenset({"behavior", "configuration"})
CONFIG_HASH_COLUMN = "config_hash"
SIGNED_PRODUCTION_DOMAINS = frozenset({"s8_numeric", "s8_twos_complement"})
CANONICAL_DATASET_SCHEMA_VERSION = "cnn_canonical_dataset_v1"
CANONICAL_INTEGRITY_SCHEMA_VERSION = 1
CANONICAL_FILE_LAYOUT = {
    "profile_metadata": "profile_metadata.json",
    "profile_arrays": "profile_arrays.npz",
    "design_ids": "design_ids.csv",
    "delta": "delta.npy",
    "labels": "labels.csv",
}
CANONICAL_CHECKSUM_FILES = frozenset(
    {
        *CANONICAL_FILE_LAYOUT.values(),
        "dataset_manifest.json",
        "train_manifest.json",
    }
)
CANONICAL_IMMUTABLE_FILES = frozenset(CANONICAL_CHECKSUM_FILES - {"train_manifest.json"})


@dataclass(frozen=True)
class ExecutionConfig:
    device: str = "cpu"
    gpu_id: int = 0
    n_jobs: int = 1
    seed: int = 20260721
    chunk_designs: int = 64


@dataclass(frozen=True)
class TrainingConfig:
    objective: str = "regression_l1"
    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 10
    early_stopping_rounds: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    reg_lambda: float = 0.2
    selection_metric: str = "l1"


@dataclass(frozen=True)
class SampleWeightingConfig:
    kind: str = "none"
    bins: int = 40
    strength: float = 1.0
    min_weight: float = 0.5
    max_weight: float = 6.0


@dataclass(frozen=True)
class EnsembleMemberConfig:
    name: str
    weight: float
    seed_offset: int
    training: TrainingConfig
    sample_weighting: SampleWeightingConfig


@dataclass(frozen=True)
class TargetEnsembleConfig:
    offset: float
    members: tuple[EnsembleMemberConfig, ...]


@dataclass(frozen=True)
class QualityGate:
    mae_max: float = 0.02
    max_abs_error_max: float = 0.05


@dataclass(frozen=True)
class TrainingInputs:
    manifest_path: Path
    manifest: Mapping[str, Any]
    profile: ApplicationProfile
    profile_metadata: Mapping[str, Any]
    designs: MultiplierDesignBatch
    labels: Mapping[str, np.ndarray]
    splits: np.ndarray
    split_counts: Mapping[str, int]
    execution: ExecutionConfig
    training: TrainingConfig
    feature_schema: str
    target_ensembles: Mapping[str, TargetEnsembleConfig]
    quality_gate: QualityGate
    output: Path
    provenance: Mapping[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return payload


def _object(payload: Mapping[str, Any], key: str, *, source: Path) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"{source}: {key!r} must be an object")
    return value


def _relative_path(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{field} must be a nonempty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _required_array(archive: Any, key: str, *, path: Path) -> np.ndarray:
    if key not in archive:
        raise ManifestError(f"{path} is missing array {key!r}")
    return np.asarray(archive[key])


def _signed_production_domain(value: object, *, field: str) -> str:
    if value not in SIGNED_PRODUCTION_DOMAINS:
        raise DataContractError(
            f"{field} must use a signed S8S8 production domain; got {value!r}. "
            "u8_numeric is isolated to generic AME research features and cannot be "
            "used by training or prediction manifests"
        )
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file_reference(value: object, *, field: str, expected: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"canonical {field} must be {expected!r}")
    candidate = Path(value)
    if candidate.is_absolute() or candidate.as_posix() != expected or ".." in candidate.parts:
        raise DataContractError(f"canonical {field} must be the local file {expected!r}")


def _canonical_sha256(value: object, *, field: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DataContractError(f"canonical checksum for {field} is not a SHA-256 digest")
    return digest


def _verify_canonical_training_integrity(payload: Mapping[str, Any], *, base: Path) -> None:
    """Verify immutable canonical data before a generated train manifest is used.

    Direct training manifests remain supported for historical workflows.  A
    canonical manifest is deliberately stricter: only model-training knobs and
    output paths may change after materialization; profile, design, label, and
    dataset-contract files must match the sealed checksum inventory.
    """

    integrity = payload.get("canonical_integrity")
    if integrity is None:
        return
    if not isinstance(integrity, Mapping):
        raise ManifestError("canonical_integrity must be an object")
    if integrity.get("schema_version") != CANONICAL_INTEGRITY_SCHEMA_VERSION:
        raise ManifestError(
            "canonical_integrity.schema_version must be "
            f"{CANONICAL_INTEGRITY_SCHEMA_VERSION}"
        )
    _canonical_file_reference(
        integrity.get("dataset_manifest"),
        field="dataset_manifest",
        expected="dataset_manifest.json",
    )
    _canonical_file_reference(
        integrity.get("checksums"),
        field="checksums",
        expected="checksums.json",
    )

    profile_spec = _object(payload, "profile", source=base / "train_manifest.json")
    designs_spec = _object(payload, "designs", source=base / "train_manifest.json")
    _canonical_file_reference(
        profile_spec.get("metadata"),
        field="profile.metadata",
        expected=CANONICAL_FILE_LAYOUT["profile_metadata"],
    )
    _canonical_file_reference(
        profile_spec.get("arrays"),
        field="profile.arrays",
        expected=CANONICAL_FILE_LAYOUT["profile_arrays"],
    )
    _canonical_file_reference(
        designs_spec.get("ids"),
        field="designs.ids",
        expected=CANONICAL_FILE_LAYOUT["design_ids"],
    )
    _canonical_file_reference(
        designs_spec.get("delta"),
        field="designs.delta",
        expected=CANONICAL_FILE_LAYOUT["delta"],
    )
    _canonical_file_reference(
        payload.get("labels"),
        field="labels",
        expected=CANONICAL_FILE_LAYOUT["labels"],
    )

    checksum_path = base / "checksums.json"
    checksums = read_json(checksum_path)
    if set(checksums) != CANONICAL_CHECKSUM_FILES:
        raise DataContractError(
            "canonical checksums inventory does not match the schema-v1 file set"
        )
    for name, expected_digest in checksums.items():
        _canonical_sha256(expected_digest, field=str(name))
    for name in CANONICAL_IMMUTABLE_FILES:
        path = base / name
        if not path.is_file():
            raise DataContractError(f"canonical immutable file is missing: {name}")
        actual = _sha256_file(path)
        expected = _canonical_sha256(checksums[name], field=name)
        if actual != expected:
            raise DataContractError(f"canonical checksum mismatch for {name}")

    dataset_manifest = read_json(base / "dataset_manifest.json")
    if dataset_manifest.get("schema_version") != CANONICAL_DATASET_SCHEMA_VERSION:
        raise DataContractError("canonical dataset_manifest has an unsupported schema_version")
    if dataset_manifest.get("target_units") != TARGET_UNITS:
        raise DataContractError("canonical dataset_manifest target_units must be 'proportion'")
    if dataset_manifest.get("files") != CANONICAL_FILE_LAYOUT:
        raise DataContractError("canonical dataset_manifest file layout differs from schema-v1")
    if dataset_manifest.get("domain") != designs_spec.get("domain"):
        raise DataContractError("canonical dataset_manifest domain differs from training designs")
    if designs_spec.get("error_convention", "approx_minus_exact") != "approx_minus_exact":
        raise DataContractError("canonical training designs use an unsupported error convention")
    provenance = _object(payload, "provenance", source=base / "train_manifest.json")
    dataset_synthetic = dataset_manifest.get("synthetic_smoke_only")
    dataset_claim_allowed = dataset_manifest.get("performance_claim_allowed")
    if not isinstance(dataset_synthetic, bool) or not isinstance(dataset_claim_allowed, bool):
        raise DataContractError("canonical dataset_manifest claim flags must be booleans")
    if dataset_synthetic is dataset_claim_allowed:
        raise DataContractError("canonical dataset_manifest claim flags are inconsistent")
    expected_kind = "synthetic" if dataset_synthetic else "measured"
    if provenance.get("kind") != expected_kind:
        raise DataContractError("canonical provenance kind differs from dataset_manifest")
    if provenance.get("synthetic_smoke_only") is not dataset_synthetic:
        raise DataContractError("canonical provenance synthetic flag differs from dataset_manifest")
    if provenance.get("performance_claim_allowed") is not dataset_claim_allowed:
        raise DataContractError("canonical provenance claim flag differs from dataset_manifest")


def load_profile(spec: Mapping[str, Any], *, base: Path) -> ApplicationProfile:
    metadata_path = _relative_path(base, spec.get("metadata"), field="profile.metadata")
    arrays_path = _relative_path(base, spec.get("arrays"), field="profile.arrays")
    metadata = read_json(metadata_path)
    try:
        with np.load(arrays_path, allow_pickle=False) as archive:
            layer_ids = tuple(str(item) for item in _required_array(archive, "layer_ids", path=arrays_path).tolist())
            activation = _required_array(archive, "activation_histograms", path=arrays_path)
            weights = _required_array(archive, "weight_histograms", path=arrays_path)
            mac_counts = _required_array(archive, "mac_counts", path=arrays_path)
            gains = _required_array(archive, "propagation_gains", path=arrays_path)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read profile arrays {arrays_path}: {exc}") from exc

    required_metadata = ("network_id", "dataset_id", "domain", "profile_evidence")
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ManifestError(f"{metadata_path} is missing fields: {missing}")
    return ApplicationProfile(
        network_id=str(metadata["network_id"]),
        dataset_id=str(metadata["dataset_id"]),
        domain=_signed_production_domain(metadata["domain"], field="profile domain"),
        layer_ids=layer_ids,
        activation_histograms=activation,
        weight_histograms=weights,
        mac_counts=mac_counts,
        propagation_gains=gains,
        profile_evidence=str(metadata["profile_evidence"]),
    )


def _read_ids(path: Path) -> tuple[str, ...]:
    try:
        frame = pd.read_csv(path, dtype={"design_id": str})
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read design IDs {path}: {exc}") from exc
    if "design_id" not in frame.columns:
        raise ManifestError(f"{path} is missing required column 'design_id'")
    return tuple(frame["design_id"].fillna("").astype(str).tolist())


def _validate_design_behavior_bindings(
    path: Path,
    design_ids: tuple[str, ...],
    behavior_hashes: tuple[str, ...],
) -> None:
    try:
        frame = pd.read_csv(path, dtype={"design_id": str, BEHAVIOR_HASH_COLUMN: str})
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read design bindings {path}: {exc}") from exc
    if BEHAVIOR_HASH_COLUMN not in frame.columns:
        return
    if (
        len(frame) != len(design_ids)
        or frame["design_id"].isna().any()
        or frame["design_id"].duplicated().any()
    ):
        raise DataContractError(f"{path} behavior bindings do not align with design IDs")
    indexed = frame.set_index("design_id", drop=False)
    if set(indexed.index.astype(str)) != set(design_ids):
        raise DataContractError(f"{path} behavior bindings do not align with design IDs")
    for design_id, expected in zip(design_ids, behavior_hashes, strict=True):
        supplied = str(indexed.loc[design_id, BEHAVIOR_HASH_COLUMN]).strip().lower()
        if supplied != expected:
            raise DataContractError(
                f"{path} behavior_sha256 mismatch for design {design_id!r}: "
                f"got {supplied}, expected {expected}"
            )


def load_designs(spec: Mapping[str, Any], *, base: Path) -> MultiplierDesignBatch:
    if spec.get("error_convention", "approx_minus_exact") != "approx_minus_exact":
        raise DataContractError("design error_convention must be 'approx_minus_exact'")
    delta_path = _relative_path(base, spec.get("delta"), field="designs.delta")
    ids_path = _relative_path(base, spec.get("ids"), field="designs.ids")
    domain = _signed_production_domain(spec.get("domain"), field="designs.domain")
    try:
        delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read design delta table {delta_path}: {exc}") from exc
    return MultiplierDesignBatch(
        design_ids=_read_ids(ids_path),
        delta=delta,
        domain=domain,
    )


def _stable_splits(behavior_hashes: tuple[str, ...], seed: int) -> np.ndarray:
    assigned: list[str] = []
    for behavior_hash in behavior_hashes:
        digest = hashlib.sha256(f"{seed}:{behavior_hash}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") / float(1 << 64)
        assigned.append("train" if value < 0.70 else "validation" if value < 0.85 else "test")
    return np.asarray(assigned, dtype="U10")


def _align_labels(
    path: Path,
    design_ids: tuple[str, ...],
    *,
    behavior_hashes: tuple[str, ...],
    split_path: Path | None,
    split_grouping: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    try:
        frame = pd.read_csv(path, dtype={"design_id": str})
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read labels {path}: {exc}") from exc
    required = {"design_id", *TARGETS}
    missing = required.difference(frame.columns)
    if missing:
        raise ManifestError(f"{path} is missing columns: {sorted(missing)}")
    if frame["design_id"].isna().any() or frame["design_id"].duplicated().any():
        raise DataContractError(f"{path} contains empty or duplicate design_id values")

    design_set = set(design_ids)
    label_set = set(frame["design_id"].astype(str))
    if label_set != design_set or len(frame) != len(design_ids):
        missing_ids = sorted(design_set - label_set)[:5]
        extra_ids = sorted(label_set - design_set)[:5]
        raise DataContractError(
            f"labels must align one-to-one with designs; missing={missing_ids}, extra={extra_ids}"
        )
    indexed = frame.set_index("design_id", drop=False).loc[list(design_ids)]

    if BEHAVIOR_HASH_COLUMN in indexed.columns:
        for design_id, supplied, expected in zip(
            design_ids,
            indexed[BEHAVIOR_HASH_COLUMN].astype(str),
            behavior_hashes,
            strict=True,
        ):
            if supplied.strip().lower() != expected:
                raise DataContractError(
                    f"{path} behavior_sha256 mismatch for design {design_id!r}: "
                    f"got {supplied!r}, expected {expected}"
                )

    labels: dict[str, np.ndarray] = {}
    for target in TARGETS:
        try:
            values = indexed[target].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataContractError(f"label column {target!r} must be numeric") from exc
        if not np.isfinite(values).all():
            raise DataContractError(f"label column {target!r} must contain only finite values")
        if np.any((values < -1.0) | (values > 1.0)):
            raise DataContractError(
                f"label column {target!r} must be in [-1, 1] proportion units"
            )
        labels[target] = values

    if split_grouping == "configuration" and split_path is None:
        raise ManifestError(
            "split_grouping 'configuration' requires an explicit split CSV"
        )

    if split_path is not None:
        try:
            split_frame = pd.read_csv(
                split_path,
                dtype={"design_id": str, CONFIG_HASH_COLUMN: str},
            )
        except (OSError, ValueError) as exc:
            raise ManifestError(f"cannot read split assignments {split_path}: {exc}") from exc
        required_split_columns = {"design_id", "split"}
        if split_grouping == "configuration":
            required_split_columns.add(CONFIG_HASH_COLUMN)
        missing_split_columns = required_split_columns.difference(split_frame.columns)
        if missing_split_columns:
            raise ManifestError(
                f"{split_path} is missing split columns: {sorted(missing_split_columns)}"
            )
        if (
            split_frame["design_id"].isna().any()
            or split_frame["design_id"].duplicated().any()
            or len(split_frame) != len(design_ids)
            or set(split_frame["design_id"].astype(str)) != design_set
        ):
            raise DataContractError("split assignments must align one-to-one with design IDs")
        if split_grouping == "configuration":
            config_hashes = split_frame[CONFIG_HASH_COLUMN]
            if config_hashes.isna().any() or config_hashes.astype(str).str.strip().eq("").any():
                raise DataContractError("configuration split contains an empty config_hash")
            normalized_hashes = config_hashes.astype(str).str.strip()
            if normalized_hashes.duplicated().any():
                raise DataContractError(
                    "configuration split requires one unique config_hash per design"
                )
        splits = split_frame.set_index("design_id").loc[list(design_ids), "split"].astype(str).to_numpy()
    elif "split" in indexed.columns:
        splits = indexed["split"].astype(str).to_numpy()
    else:
        splits = _stable_splits(behavior_hashes, seed)

    unknown = sorted(set(splits) - set(SPLITS))
    if unknown:
        raise DataContractError(f"unsupported split names: {unknown}")
    if split_grouping == "behavior":
        group_splits: dict[str, str] = {}
        for behavior_hash, split in zip(behavior_hashes, splits, strict=True):
            previous = group_splits.setdefault(behavior_hash, str(split))
            if previous != split:
                raise DataContractError(
                    "one behavior group is assigned to multiple splits: "
                    f"{behavior_hash} -> {sorted({previous, str(split)})}"
                )
    counts = {name: int(np.sum(splits == name)) for name in SPLITS}
    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        raise DataContractError(f"all train/validation/test splits must be nonempty; empty={empty}")
    return labels, np.asarray(splits, dtype="U10"), counts


def _execution(payload: Mapping[str, Any], *, source: Path) -> ExecutionConfig:
    section = payload.get("execution", {})
    if not isinstance(section, dict):
        raise ManifestError(f"{source}: execution must be an object")
    config = ExecutionConfig(
        device=str(section.get("device", "cpu")),
        gpu_id=int(section.get("gpu_id", 0)),
        n_jobs=int(section.get("n_jobs", 1)),
        seed=int(section.get("seed", 20260721)),
        chunk_designs=int(section.get("chunk_designs", 64)),
    )
    if config.device not in {"cpu", "cuda"}:
        raise ManifestError("execution.device must be 'cpu' or 'cuda'")
    if config.gpu_id < 0 or config.n_jobs < 1 or config.chunk_designs < 1:
        raise ManifestError("execution gpu_id, n_jobs, and chunk_designs must be nonnegative/positive")
    return config


def _training(payload: Mapping[str, Any], *, source: Path) -> TrainingConfig:
    section = payload.get("training", {})
    if not isinstance(section, dict):
        raise ManifestError(f"{source}: training must be an object")
    config = TrainingConfig(
        objective=str(section.get("objective", "regression_l1")),
        n_estimators=int(section.get("n_estimators", 600)),
        learning_rate=float(section.get("learning_rate", 0.03)),
        num_leaves=int(section.get("num_leaves", 31)),
        min_child_samples=int(section.get("min_child_samples", 10)),
        early_stopping_rounds=int(section.get("early_stopping_rounds", 50)),
        feature_fraction=float(section.get("feature_fraction", 0.9)),
        bagging_fraction=float(section.get("bagging_fraction", 0.9)),
        reg_lambda=float(section.get("reg_lambda", 0.2)),
        selection_metric=str(section.get("selection_metric", "l1")),
    )
    if config.objective not in {"regression", "regression_l1"}:
        raise ManifestError("training.objective must be 'regression' or 'regression_l1'")
    if config.n_estimators < 1 or config.num_leaves < 2 or config.min_child_samples < 1:
        raise ManifestError("training estimator counts and leaf sizes must be positive")
    if config.learning_rate <= 0.0 or config.early_stopping_rounds < 0:
        raise ManifestError("training learning_rate must be positive and early_stopping_rounds nonnegative")
    if not 0.0 < config.feature_fraction <= 1.0 or not 0.0 < config.bagging_fraction <= 1.0:
        raise ManifestError("training feature_fraction and bagging_fraction must be in (0, 1]")
    if config.selection_metric not in {"l1", "max_abs_error"}:
        raise ManifestError("training.selection_metric must be 'l1' or 'max_abs_error'")
    if not math.isfinite(config.reg_lambda) or config.reg_lambda < 0.0:
        raise ManifestError("training.reg_lambda must be finite and nonnegative")
    return config


def _member_training(
    global_training: TrainingConfig,
    raw: object,
    *,
    source: Path,
) -> TrainingConfig:
    if raw is None:
        return global_training
    if not isinstance(raw, dict):
        raise ManifestError(f"{source}: ensemble member training must be an object")
    merged = {
        "objective": global_training.objective,
        "n_estimators": global_training.n_estimators,
        "learning_rate": global_training.learning_rate,
        "num_leaves": global_training.num_leaves,
        "min_child_samples": global_training.min_child_samples,
        "early_stopping_rounds": global_training.early_stopping_rounds,
        "feature_fraction": global_training.feature_fraction,
        "bagging_fraction": global_training.bagging_fraction,
        "reg_lambda": global_training.reg_lambda,
        "selection_metric": global_training.selection_metric,
        **raw,
    }
    return _training({"training": merged}, source=source)


def _sample_weighting(raw: object, *, source: Path) -> SampleWeightingConfig:
    if raw is None:
        return SampleWeightingConfig()
    if not isinstance(raw, dict):
        raise ManifestError(f"{source}: sample_weighting must be an object")
    config = SampleWeightingConfig(
        kind=str(raw.get("kind", "none")),
        bins=int(raw.get("bins", 40)),
        strength=float(raw.get("strength", 1.0)),
        min_weight=float(raw.get("min_weight", 0.5)),
        max_weight=float(raw.get("max_weight", 6.0)),
    )
    if config.kind not in {"none", "target_density"}:
        raise ManifestError("sample_weighting.kind must be 'none' or 'target_density'")
    numeric = (config.strength, config.min_weight, config.max_weight)
    if config.bins < 2 or not all(math.isfinite(value) for value in numeric):
        raise ManifestError("sample_weighting parameters must be finite with bins >= 2")
    if config.strength < 0.0 or config.min_weight <= 0.0 or config.max_weight < config.min_weight:
        raise ManifestError("sample_weighting bounds/strength are invalid")
    return config


def _target_ensembles(
    payload: Mapping[str, Any],
    *,
    global_training: TrainingConfig,
    source: Path,
) -> dict[str, TargetEnsembleConfig]:
    raw_ensembles = payload.get("target_ensembles", {})
    if not isinstance(raw_ensembles, dict):
        raise ManifestError(f"{source}: target_ensembles must be an object")
    unknown_targets = sorted(set(raw_ensembles) - set(TARGETS))
    if unknown_targets:
        raise ManifestError(f"unsupported target_ensembles keys: {unknown_targets}")
    parsed: dict[str, TargetEnsembleConfig] = {}
    for target in TARGETS:
        raw_target = raw_ensembles.get(target)
        if raw_target is None:
            parsed[target] = TargetEnsembleConfig(
                offset=0.0,
                members=(
                    EnsembleMemberConfig(
                        name=target,
                        weight=1.0,
                        seed_offset=0,
                        training=global_training,
                        sample_weighting=SampleWeightingConfig(),
                    ),
                ),
            )
            continue
        if not isinstance(raw_target, dict):
            raise ManifestError(f"target_ensembles.{target} must be an object")
        offset = float(raw_target.get("offset", 0.0))
        members_value = raw_target.get("members")
        if not math.isfinite(offset):
            raise ManifestError(f"target_ensembles.{target}.offset must be finite")
        if not isinstance(members_value, list) or not members_value:
            raise ManifestError(f"target_ensembles.{target}.members must be a nonempty list")
        members: list[EnsembleMemberConfig] = []
        names: set[str] = set()
        for index, raw_member in enumerate(members_value):
            if not isinstance(raw_member, dict):
                raise ManifestError(f"target_ensembles.{target}.members[{index}] must be an object")
            name = str(raw_member.get("name", ""))
            if not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,63}", name):
                raise ManifestError(
                    f"target_ensembles.{target}.members[{index}].name has an invalid format"
                )
            if name in names:
                raise ManifestError(f"target_ensembles.{target} contains duplicate member {name!r}")
            names.add(name)
            weight = float(raw_member.get("weight", 0.0))
            seed_offset = int(raw_member.get("seed_offset", 0))
            if not math.isfinite(weight) or weight < 0.0:
                raise ManifestError(f"target_ensembles.{target}.{name}.weight must be nonnegative")
            if seed_offset < 0:
                raise ManifestError(f"target_ensembles.{target}.{name}.seed_offset must be nonnegative")
            members.append(
                EnsembleMemberConfig(
                    name=name,
                    weight=weight,
                    seed_offset=seed_offset,
                    training=_member_training(
                        global_training,
                        raw_member.get("training"),
                        source=source,
                    ),
                    sample_weighting=_sample_weighting(
                        raw_member.get("sample_weighting"),
                        source=source,
                    ),
                )
            )
        weight_sum = sum(member.weight for member in members)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ManifestError(
                f"target_ensembles.{target} member weights must sum to 1; got {weight_sum}"
            )
        parsed[target] = TargetEnsembleConfig(offset=offset, members=tuple(members))
    return parsed


def load_training_inputs(path: str | Path) -> TrainingInputs:
    manifest_path = Path(path).expanduser().resolve()
    payload = read_json(manifest_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"{manifest_path}: schema_version must be {SCHEMA_VERSION}")
    if payload.get("target_units") != TARGET_UNITS:
        raise ManifestError(f"{manifest_path}: target_units must be {TARGET_UNITS!r}")
    base = manifest_path.parent
    _verify_canonical_training_integrity(payload, base=base)
    execution = _execution(payload, source=manifest_path)
    split_grouping = str(payload.get("split_grouping", "behavior"))
    if split_grouping not in SPLIT_GROUPINGS:
        raise ManifestError(
            f"unsupported split_grouping {split_grouping!r}; "
            f"expected one of {sorted(SPLIT_GROUPINGS)}"
        )
    feature_schema = str(payload.get("feature_schema", DEFAULT_FEATURE_SCHEMA))
    if feature_schema not in SUPPORTED_FEATURE_SCHEMAS:
        raise ManifestError(
            f"unsupported feature_schema {feature_schema!r}; "
            f"expected one of {sorted(SUPPORTED_FEATURE_SCHEMAS)}"
        )
    training = _training(payload, source=manifest_path)
    profile_spec = _object(payload, "profile", source=manifest_path)
    profile_metadata_path = _relative_path(
        base,
        profile_spec.get("metadata"),
        field="profile.metadata",
    )
    profile_metadata = read_json(profile_metadata_path)
    profile = load_profile(profile_spec, base=base)
    designs = load_designs(_object(payload, "designs", source=manifest_path), base=base)
    design_hashes = behavior_sha256_rows(designs.delta)
    designs_spec = _object(payload, "designs", source=manifest_path)
    _validate_design_behavior_bindings(
        _relative_path(base, designs_spec.get("ids"), field="designs.ids"),
        designs.design_ids,
        design_hashes,
    )
    labels_path = _relative_path(base, payload.get("labels"), field="labels")
    split_value = payload.get("split")
    split_path = None if split_value is None else _relative_path(base, split_value, field="split")
    labels, splits, counts = _align_labels(
        labels_path,
        designs.design_ids,
        behavior_hashes=design_hashes,
        split_path=split_path,
        split_grouping=split_grouping,
        seed=execution.seed,
    )
    provenance = _object(payload, "provenance", source=manifest_path)
    kind = provenance.get("kind")
    if kind not in {"measured", "synthetic"}:
        raise ManifestError("provenance.kind must be 'measured' or 'synthetic'")
    output = _relative_path(base, payload.get("output"), field="output")
    gate_section = payload.get("quality_gate", {})
    if not isinstance(gate_section, dict):
        raise ManifestError("quality_gate must be an object")
    gate = QualityGate(
        mae_max=float(gate_section.get("mae_max", 0.02)),
        max_abs_error_max=float(gate_section.get("max_abs_error_max", 0.05)),
    )
    if gate.mae_max < 0.0 or gate.max_abs_error_max < 0.0:
        raise ManifestError("quality gate thresholds must be nonnegative")
    return TrainingInputs(
        manifest_path=manifest_path,
        manifest=payload,
        profile=profile,
        profile_metadata=profile_metadata,
        designs=designs,
        labels=labels,
        splits=splits,
        split_counts=counts,
        execution=execution,
        training=training,
        feature_schema=feature_schema,
        target_ensembles=_target_ensembles(
            payload,
            global_training=training,
            source=manifest_path,
        ),
        quality_gate=gate,
        output=output,
        provenance=dict(provenance),
    )


def load_prediction_designs(path: str | Path) -> tuple[MultiplierDesignBatch, Path | None]:
    manifest_path = Path(path).expanduser().resolve()
    payload = read_json(manifest_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ManifestError(f"{manifest_path}: schema_version must be {SCHEMA_VERSION}")
    if payload.get("target_units") != TARGET_UNITS:
        raise ManifestError(f"{manifest_path}: target_units must be {TARGET_UNITS!r}")
    designs = load_designs(_object(payload, "designs", source=manifest_path), base=manifest_path.parent)
    designs_spec = _object(payload, "designs", source=manifest_path)
    _validate_design_behavior_bindings(
        _relative_path(manifest_path.parent, designs_spec.get("ids"), field="designs.ids"),
        designs.design_ids,
        behavior_sha256_rows(designs.delta),
    )
    output_value = payload.get("output")
    output = None if output_value is None else _relative_path(manifest_path.parent, output_value, field="output")
    return designs, output
