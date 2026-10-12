from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import ApplicationProfile, MultiplierDesignBatch
from .errors import ArtifactError
from .features import SUPPORTED_FEATURE_SCHEMAS, build_features
from .modeling import FittedTarget, inspect_serialized_models, lightgbm_version

BUNDLE_SCHEMA_VERSION = 2
HYBRID_BUNDLE_SCHEMA_VERSION = 3
POSTPROCESSED_BUNDLE_SCHEMA_VERSION = 4
COMPOSITE_BUNDLE_SCHEMA_VERSION = 5
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = frozenset(
    {
        1,
        BUNDLE_SCHEMA_VERSION,
        HYBRID_BUNDLE_SCHEMA_VERSION,
        POSTPROCESSED_BUNDLE_SCHEMA_VERSION,
        COMPOSITE_BUNDLE_SCHEMA_VERSION,
    }
)
TARGETS = ("top1_drop", "top5_drop")
SIGNED_PRODUCTION_DOMAINS = frozenset({"s8_numeric", "s8_twos_complement"})
BASE_BUNDLE_FILES = frozenset(
    {
        "application_matrix.npz",
        "application_profile.npz",
        "feature_schema.json",
        "manifest.json",
        "metrics.json",
        "profile_metadata.json",
        "provenance.json",
        "split_summary.json",
        "training_manifest.snapshot.json",
    }
)
V1_EXPECTED_BUNDLE_FILES = frozenset(
    {*BASE_BUNDLE_FILES, "models/top1_drop.txt", "models/top5_drop.txt"}
)


@dataclass(frozen=True)
class LoadedEnsembleMember:
    model_key: str
    name: str
    weight: float
    kind: str
    path: str
    checkpoint_member: str | None = None


@dataclass(frozen=True)
class LoadedTargetPostprocessor:
    kind: str
    member_keys: tuple[str, str]
    scale: float


@dataclass(frozen=True)
class LoadedTargetEnsemble:
    offset: float
    members: tuple[LoadedEnsembleMember, ...]
    postprocessor: LoadedTargetPostprocessor | None = None


@dataclass(frozen=True)
class LoadedArtifact:
    path: Path
    manifest: Mapping[str, Any]
    profile: ApplicationProfile
    feature_names: tuple[str, ...]
    model_texts: Mapping[str, str]
    delta_conv_paths: Mapping[str, tuple[Path, str | None]]
    correctness_conv_paths: Mapping[str, tuple[Path, str | None]]
    target_ensembles: Mapping[str, LoadedTargetEnsemble]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"bundle is missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid bundle JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError(f"bundle JSON must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _application_matrix(profile: ApplicationProfile) -> tuple[np.ndarray, np.ndarray]:
    activation = np.asarray(profile.activation_histograms, dtype=np.float64)
    weights = np.asarray(profile.weight_histograms, dtype=np.float64)
    activation = activation / activation.sum(axis=1, keepdims=True)
    weights = weights / weights.sum(axis=1, keepdims=True)
    coefficients = np.asarray(profile.mac_counts, dtype=np.float64) * np.asarray(
        profile.propagation_gains, dtype=np.float64
    )
    matrix = np.einsum("l,li,lj->ij", coefficients, activation, weights, optimize=True)
    return matrix, np.abs(matrix)


def _write_profile(
    root: Path,
    profile: ApplicationProfile,
    profile_metadata: Mapping[str, Any],
) -> None:
    metadata = dict(profile_metadata)
    metadata.update(
        {
            "schema_version": 1,
            "network_id": profile.network_id,
            "dataset_id": profile.dataset_id,
            "domain": profile.domain,
            "profile_evidence": profile.profile_evidence,
        }
    )
    _write_json(
        root / "profile_metadata.json",
        metadata,
    )
    np.savez_compressed(
        root / "application_profile.npz",
        layer_ids=np.asarray(profile.layer_ids, dtype="U"),
        activation_histograms=np.asarray(profile.activation_histograms),
        weight_histograms=np.asarray(profile.weight_histograms),
        mac_counts=np.asarray(profile.mac_counts),
        propagation_gains=np.asarray(profile.propagation_gains),
    )
    application_matrix, magnitude_matrix = _application_matrix(profile)
    np.savez_compressed(
        root / "application_matrix.npz",
        application_matrix=application_matrix,
        magnitude_matrix=magnitude_matrix,
    )


def _write_checksums(root: Path) -> None:
    entries = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    _write_json(root / "checksums.json", {"algorithm": "sha256", "files": entries})


def ensure_output_available(path: Path) -> None:
    if path.exists():
        raise ArtifactError(f"refusing to overwrite existing bundle path: {path}")


def publish_bundle(
    *,
    output: Path,
    profile: ApplicationProfile,
    profile_metadata: Mapping[str, Any],
    profile_digest: str,
    feature_names: tuple[str, ...],
    feature_schema: str,
    fitted: Mapping[str, FittedTarget],
    split_counts: Mapping[str, int],
    claim_status: str,
    performance_claim_allowed: bool,
    quality_gate_passed: bool | None,
    training_manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    execution: Mapping[str, Any],
    training: Mapping[str, Any],
) -> Path:
    output = output.expanduser().resolve()
    ensure_output_available(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _write_profile(temporary, profile, profile_metadata)
        _write_json(
            temporary / "feature_schema.json",
            {
                "schema_name": feature_schema,
                "feature_names": feature_names,
                "feature_count": len(feature_names),
                "signed_transform": "sign_x_log1p_abs_x",
                "nonnegative_transform": "log1p",
            },
        )
        model_dir = temporary / "models"
        model_dir.mkdir(parents=True)
        model_ensembles: dict[str, Any] = {}
        for target in TARGETS:
            target_fit = fitted[target]
            member_rows: list[dict[str, Any]] = []
            for member in target_fit.members:
                if len(target_fit.members) == 1:
                    relative = Path("models") / f"{target}.txt"
                else:
                    relative = Path("models") / target / f"{member.name}.txt"
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                member.booster.save_model(str(destination))
                member_rows.append(
                    {
                        "name": member.name,
                        "path": relative.as_posix(),
                        "weight": member.weight,
                        "selected_iterations": member.selected_iterations,
                    }
                )
            model_ensembles[target] = {
                "offset": target_fit.offset,
                "members": member_rows,
            }
        metrics = {
            target: {
                **fitted[target].metrics.to_dict(),
                "offset": fitted[target].offset,
                "members": {
                    member.name: {
                        "selected_iterations": member.selected_iterations,
                        "weight": member.weight,
                    }
                    for member in fitted[target].members
                },
            }
            for target in TARGETS
        }
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "split_summary.json", {"counts": split_counts})
        _write_json(temporary / "training_manifest.snapshot.json", training_manifest)
        _write_json(temporary / "provenance.json", provenance)
        _write_json(
            temporary / "manifest.json",
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "bundle_kind": "cnn_ame_lightgbm_accuracy_drop",
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "network_id": profile.network_id,
                "dataset_id": profile.dataset_id,
                "domain": profile.domain,
                "error_convention": "approx_minus_exact",
                "target_units": "proportion",
                "targets": TARGETS,
                "profile_digest": profile_digest,
                "feature_schema": feature_schema,
                "feature_count": len(feature_names),
                "learned_backend": "lightgbm",
                "lightgbm_version": lightgbm_version(),
                "numpy_version": np.__version__,
                "claim_status": claim_status,
                "synthetic_only": claim_status == "synthetic_smoke_only",
                "performance_claim_allowed": performance_claim_allowed,
                "quality_gate_passed": quality_gate_passed,
                "execution": execution,
                "training": training,
                "model_ensembles": model_ensembles,
            },
        )
        _write_checksums(temporary)
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def _verify_checksums(root: Path) -> None:
    payload = _load_json(root / "checksums.json")
    if payload.get("algorithm") != "sha256" or not isinstance(
        payload.get("files"), dict
    ):
        raise ArtifactError("bundle checksum manifest is malformed")
    expected_files = payload["files"]
    for relative, expected in expected_files.items():
        path = root / str(relative)
        if not path.is_file():
            raise ArtifactError(f"bundle checksum references missing file: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise ArtifactError(
                f"bundle checksum mismatch for {relative}: expected {expected}, got {actual}"
            )
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    tracked_files = set(expected_files)
    if actual_files != tracked_files:
        untracked = sorted(actual_files - tracked_files)
        missing_from_disk = sorted(tracked_files - actual_files)
        raise ArtifactError(
            "bundle checksum inventory disagrees with files on disk; "
            f"untracked={untracked}, missing={missing_from_disk}"
        )


def _load_target_ensembles(
    manifest: Mapping[str, Any],
    *,
    schema_version: int,
) -> tuple[dict[str, LoadedTargetEnsemble], frozenset[str]]:
    if schema_version == 1:
        ensembles = {
            target: LoadedTargetEnsemble(
                offset=0.0,
                members=(
                    LoadedEnsembleMember(
                        target,
                        target,
                        1.0,
                        "lightgbm",
                        f"models/{target}.txt",
                    ),
                ),
            )
            for target in TARGETS
        }
        return ensembles, V1_EXPECTED_BUNDLE_FILES
    raw = manifest.get("model_ensembles")
    if not isinstance(raw, dict) or set(raw) != set(TARGETS):
        raise ArtifactError(f"bundle model_ensembles must contain exactly {TARGETS!r}")
    ensembles: dict[str, LoadedTargetEnsemble] = {}
    model_references: dict[str, set[tuple[str, str | None]]] = {}
    for target in TARGETS:
        target_raw = raw[target]
        if not isinstance(target_raw, dict):
            raise ArtifactError(f"model_ensembles.{target} must be an object")
        try:
            offset = float(target_raw.get("offset", 0.0))
        except (TypeError, ValueError) as exc:
            raise ArtifactError(
                f"model_ensembles.{target}.offset must be numeric"
            ) from exc
        members_raw = target_raw.get("members")
        if (
            not math.isfinite(offset)
            or not isinstance(members_raw, list)
            or not members_raw
        ):
            raise ArtifactError(f"model_ensembles.{target} has invalid offset/members")
        members: list[LoadedEnsembleMember] = []
        names: set[str] = set()
        for index, member_raw in enumerate(members_raw):
            if not isinstance(member_raw, dict):
                raise ArtifactError(
                    f"model_ensembles.{target}.members[{index}] must be an object"
                )
            name = member_raw.get("name")
            path_value = member_raw.get("path")
            kind = member_raw.get("kind", "lightgbm" if schema_version == 2 else None)
            checkpoint_member = member_raw.get("checkpoint_member")
            try:
                weight = float(member_raw.get("weight"))
            except (TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"model_ensembles.{target} has a nonnumeric weight"
                ) from exc
            if not isinstance(name, str) or not name or name in names:
                raise ArtifactError(
                    f"model_ensembles.{target} has invalid/duplicate member names"
                )
            if not isinstance(path_value, str):
                raise ArtifactError(
                    f"model_ensembles.{target}.{name}.path must be a string"
                )
            if kind not in {"lightgbm", "delta_conv", "correctness_conv"}:
                raise ArtifactError(
                    f"model_ensembles.{target}.{name}.kind is unsupported"
                )
            if checkpoint_member is not None and (
                schema_version < COMPOSITE_BUNDLE_SCHEMA_VERSION
                or kind not in {"delta_conv", "correctness_conv"}
                or not isinstance(checkpoint_member, str)
                or not checkpoint_member
            ):
                raise ArtifactError(
                    f"model_ensembles.{target}.{name}.checkpoint_member requires "
                    "a schema-v5 neural composite"
                )
            path = Path(path_value)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[:1] != ("models",)
            ):
                raise ArtifactError(f"model_ensembles.{target}.{name}.path is unsafe")
            relative = path.as_posix()
            references = model_references.setdefault(relative, set())
            reference = (str(kind), checkpoint_member)
            if references:
                shared_composite = (
                    schema_version >= COMPOSITE_BUNDLE_SCHEMA_VERSION
                    and kind in {"delta_conv", "correctness_conv"}
                    and checkpoint_member is not None
                    and all(
                        prior_kind == kind and prior_member is not None
                        for prior_kind, prior_member in references
                    )
                )
                if not shared_composite:
                    raise ArtifactError(
                        "two model ensemble members reference the same file"
                    )
                if reference in references:
                    raise ArtifactError(
                        "two model ensemble members select the same composite member"
                    )
            if not math.isfinite(weight) or weight < 0.0:
                raise ArtifactError(
                    f"model_ensembles.{target}.{name}.weight is invalid"
                )
            names.add(name)
            references.add(reference)
            model_key = f"{target}::{name}"
            members.append(
                LoadedEnsembleMember(
                    model_key,
                    name,
                    weight,
                    str(kind),
                    relative,
                    checkpoint_member,
                )
            )
        weight_sum = sum(member.weight for member in members)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ArtifactError(f"model_ensembles.{target} weights do not sum to 1")
        postprocessor_raw = target_raw.get("postprocessor")
        postprocessor: LoadedTargetPostprocessor | None = None
        if postprocessor_raw is not None:
            if schema_version < POSTPROCESSED_BUNDLE_SCHEMA_VERSION:
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor requires bundle schema v4"
                )
            if not isinstance(postprocessor_raw, dict):
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor must be an object"
                )
            kind = postprocessor_raw.get("kind")
            member_names = postprocessor_raw.get("members")
            try:
                scale = float(postprocessor_raw.get("scale"))
            except (TypeError, ValueError) as exc:
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor.scale must be numeric"
                ) from exc
            if kind != "absolute_member_disagreement_v1":
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor.kind is unsupported"
                )
            if (
                not isinstance(member_names, list)
                or len(member_names) != 2
                or not all(isinstance(name, str) for name in member_names)
                or member_names[0] == member_names[1]
            ):
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor.members must contain "
                    "two distinct member names"
                )
            member_keys = {member.name: member.model_key for member in members}
            if any(name not in member_keys for name in member_names):
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor references an unknown member"
                )
            if not math.isfinite(scale) or scale < 0.0:
                raise ArtifactError(
                    f"model_ensembles.{target}.postprocessor.scale is invalid"
                )
            postprocessor = LoadedTargetPostprocessor(
                kind=str(kind),
                member_keys=(
                    member_keys[member_names[0]],
                    member_keys[member_names[1]],
                ),
                scale=scale,
            )
        ensembles[target] = LoadedTargetEnsemble(
            offset=offset,
            members=tuple(members),
            postprocessor=postprocessor,
        )
    return ensembles, frozenset({*BASE_BUNDLE_FILES, *model_references})


def _load_profile(root: Path) -> ApplicationProfile:
    metadata = _load_json(root / "profile_metadata.json")
    try:
        with np.load(root / "application_profile.npz", allow_pickle=False) as archive:
            profile = ApplicationProfile(
                network_id=str(metadata["network_id"]),
                dataset_id=str(metadata["dataset_id"]),
                domain=str(metadata["domain"]),
                layer_ids=tuple(str(item) for item in archive["layer_ids"].tolist()),
                activation_histograms=np.asarray(archive["activation_histograms"]),
                weight_histograms=np.asarray(archive["weight_histograms"]),
                mac_counts=np.asarray(archive["mac_counts"]),
                propagation_gains=np.asarray(archive["propagation_gains"]),
                profile_evidence=str(metadata["profile_evidence"]),
            )
    except (KeyError, OSError, ValueError) as exc:
        raise ArtifactError(
            f"cannot load application profile from {root}: {exc}"
        ) from exc
    return profile


def _validate_bundle_metadata(
    manifest: Mapping[str, Any],
    profile: ApplicationProfile,
    provenance: Mapping[str, Any],
    *,
    schema_version: int,
) -> None:
    if profile.domain not in SIGNED_PRODUCTION_DOMAINS:
        raise ArtifactError(
            "bundle domain must use a signed S8S8 production domain; "
            f"got {profile.domain!r}"
        )
    for field, profile_value in (
        ("network_id", profile.network_id),
        ("dataset_id", profile.dataset_id),
        ("domain", profile.domain),
    ):
        if manifest.get(field) != profile_value:
            raise ArtifactError(
                f"bundle manifest {field} does not match bundled profile: "
                f"{manifest.get(field)!r} != {profile_value!r}"
            )
    if manifest.get("error_convention") != "approx_minus_exact":
        raise ArtifactError("bundle error_convention must be 'approx_minus_exact'")
    if manifest.get("target_units") != "proportion":
        raise ArtifactError("bundle target_units must be 'proportion'")
    if tuple(manifest.get("targets", ())) != TARGETS:
        raise ArtifactError(f"bundle targets must be ordered as {TARGETS!r}")
    expected_backend = "lightgbm+torch" if schema_version >= 3 else "lightgbm"
    if manifest.get("learned_backend") != expected_backend:
        raise ArtifactError(f"bundle learned_backend must be {expected_backend!r}")

    provenance_kind = provenance.get("kind")
    if provenance_kind not in {"measured", "synthetic"}:
        raise ArtifactError("bundle provenance.kind must be 'measured' or 'synthetic'")
    measured = profile.profile_evidence == "measured" and provenance_kind == "measured"
    claim_status = manifest.get("claim_status")
    allowed_measured_statuses = {"measured_holdout"}
    if schema_version >= 3:
        allowed_measured_statuses.add("measured_post_failure_audit")
    if measured and claim_status not in allowed_measured_statuses:
        raise ArtifactError(
            "bundle claim_status disagrees with measured profile/provenance evidence"
        )
    expected_status = claim_status if measured else "synthetic_smoke_only"
    expected_synthetic = not measured
    expected_claim_allowed = measured and claim_status == "measured_holdout"
    if claim_status != expected_status:
        raise ArtifactError(
            "bundle claim_status disagrees with profile evidence and provenance: "
            f"expected {expected_status!r}"
        )
    if manifest.get("synthetic_only") is not expected_synthetic:
        raise ArtifactError("bundle synthetic_only disagrees with claim_status")
    if manifest.get("performance_claim_allowed") is not expected_claim_allowed:
        raise ArtifactError(
            "bundle performance_claim_allowed disagrees with claim_status"
        )
    if (
        "synthetic_smoke_only" in provenance
        and provenance.get("synthetic_smoke_only") is not expected_synthetic
    ):
        raise ArtifactError(
            "bundle provenance.synthetic_smoke_only disagrees with claim_status"
        )


def load_artifact(path: str | Path) -> LoadedArtifact:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ArtifactError(f"bundle directory does not exist: {root}")
    _verify_checksums(root)
    manifest = _load_json(root / "manifest.json")
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
        raise ArtifactError(
            f"unsupported bundle schema_version {manifest.get('schema_version')!r}; "
            f"expected one of {sorted(SUPPORTED_BUNDLE_SCHEMA_VERSIONS)}"
        )
    expected_kind = (
        "cnn_ame_hybrid_accuracy_drop"
        if schema_version >= HYBRID_BUNDLE_SCHEMA_VERSION
        else "cnn_ame_lightgbm_accuracy_drop"
    )
    if manifest.get("bundle_kind") != expected_kind:
        raise ArtifactError(f"bundle_kind must be {expected_kind!r}")
    feature_schema = manifest.get("feature_schema")
    if feature_schema not in SUPPORTED_FEATURE_SCHEMAS:
        raise ArtifactError(
            f"unsupported feature schema: {manifest.get('feature_schema')!r}"
        )

    target_ensembles, expected_files = _load_target_ensembles(
        manifest,
        schema_version=int(schema_version),
    )
    actual_files = {
        str(file.relative_to(root))
        for file in root.rglob("*")
        if file.is_file() and file.name != "checksums.json"
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        raise ArtifactError(
            "bundle artifact inventory is not the exact declared schema inventory; "
            f"missing={missing}, unexpected={unexpected}"
        )

    profile = _load_profile(root)
    provenance = _load_json(root / "provenance.json")
    _validate_bundle_metadata(
        manifest,
        profile,
        provenance,
        schema_version=int(schema_version),
    )
    schema = _load_json(root / "feature_schema.json")
    if schema.get("schema_name") != feature_schema:
        raise ArtifactError(
            "feature_schema.json schema_name disagrees with the bundle manifest"
        )
    names_value = schema.get("feature_names")
    if not isinstance(names_value, list) or not all(
        isinstance(item, str) for item in names_value
    ):
        raise ArtifactError("feature_schema.json has invalid feature_names")
    feature_names = tuple(names_value)
    if len(feature_names) != int(manifest.get("feature_count", -1)):
        raise ArtifactError(
            "bundle manifest and feature schema disagree on feature count"
        )
    zero_design = MultiplierDesignBatch(
        design_ids=("__profile_digest_check__",),
        delta=np.zeros((1, 256, 256), dtype=np.int8),
        domain=profile.domain,
    )
    generated = build_features(profile, zero_design, feature_schema=str(feature_schema))
    if generated.profile_digest != manifest.get("profile_digest"):
        raise ArtifactError("application profile digest does not match bundle manifest")
    if generated.feature_names != feature_names:
        raise ArtifactError("application profile produces a different feature schema")

    try:
        model_texts: dict[str, str] = {}
        delta_conv_paths: dict[str, tuple[Path, str | None]] = {}
        correctness_conv_paths: dict[str, tuple[Path, str | None]] = {}
        delta_conv_groups: dict[Path, list[LoadedEnsembleMember]] = {}
        correctness_conv_groups: dict[Path, list[LoadedEnsembleMember]] = {}
        for target, ensemble in target_ensembles.items():
            for member in ensemble.members:
                model_path = root / member.path
                if member.kind == "lightgbm":
                    model_texts[member.model_key] = model_path.read_text(
                        encoding="utf-8"
                    )
                elif member.kind == "delta_conv":
                    delta_conv_paths[member.model_key] = (
                        model_path,
                        member.checkpoint_member,
                    )
                    delta_conv_groups.setdefault(model_path, []).append(member)
                else:
                    correctness_conv_paths[member.model_key] = (
                        model_path,
                        member.checkpoint_member,
                    )
                    correctness_conv_groups.setdefault(model_path, []).append(member)
        if delta_conv_groups:
            from .delta_conv import (
                inspect_delta_conv_checkpoint,
                inspect_delta_conv_composite_checkpoint,
            )

            for model_path, members in delta_conv_groups.items():
                checkpoint_members = tuple(
                    member.checkpoint_member for member in members
                )
                if any(name is not None for name in checkpoint_members):
                    if not all(isinstance(name, str) for name in checkpoint_members):
                        raise RuntimeError(
                            "composite delta-CNN references must all select members"
                        )
                    inspect_delta_conv_composite_checkpoint(
                        model_path,
                        expected_members=tuple(
                            str(name) for name in checkpoint_members
                        ),
                        expected_feature_count=len(feature_names),
                        expected_feature_schema=str(feature_schema),
                    )
                else:
                    inspect_delta_conv_checkpoint(
                        model_path,
                        expected_feature_count=len(feature_names),
                        expected_feature_schema=str(feature_schema),
                    )
        if correctness_conv_groups:
            from .correctness_conv import (
                inspect_correctness_conv_checkpoint,
                inspect_correctness_conv_composite_checkpoint,
            )

            for model_path, members in correctness_conv_groups.items():
                checkpoint_members = tuple(
                    member.checkpoint_member for member in members
                )
                if any(name is not None for name in checkpoint_members):
                    if not all(isinstance(name, str) for name in checkpoint_members):
                        raise RuntimeError(
                            "composite correctness-CNN references must all select members"
                        )
                    inspect_correctness_conv_composite_checkpoint(
                        model_path,
                        expected_members=tuple(
                            str(name) for name in checkpoint_members
                        ),
                        expected_feature_count=len(feature_names),
                        expected_feature_schema=str(feature_schema),
                    )
                else:
                    inspect_correctness_conv_checkpoint(
                        model_path,
                        expected_feature_count=len(feature_names),
                        expected_feature_schema=str(feature_schema),
                    )
        metadata = inspect_serialized_models(model_texts) if model_texts else {}
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ArtifactError(f"cannot load bundle models from {root}: {exc}") from exc
    for target, model in metadata.items():
        if model.feature_count != len(feature_names):
            raise ArtifactError(
                f"{target} model feature count does not match feature schema"
            )
        if model.feature_names != feature_names:
            raise ArtifactError(
                f"{target} model feature names do not match feature schema order"
            )
    return LoadedArtifact(
        path=root,
        manifest=manifest,
        profile=profile,
        feature_names=feature_names,
        model_texts=MappingProxyType(model_texts),
        delta_conv_paths=MappingProxyType(delta_conv_paths),
        correctness_conv_paths=MappingProxyType(correctness_conv_paths),
        target_ensembles=MappingProxyType(target_ensembles),
    )
