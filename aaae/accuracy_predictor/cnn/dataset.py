from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .behavior import BEHAVIOR_HASH_COLUMN, BEHAVIOR_HASH_SCHEMA, behavior_sha256_rows
from .contracts import ApplicationProfile, DataContractError


BUILD_SCHEMA_VERSION = "cnn_dataset_build_manifest_v1"
DATASET_SCHEMA_VERSION = "cnn_canonical_dataset_v1"
TRAIN_SCHEMA_VERSION = 1
ERROR_CONVENTION = "approx_minus_exact"
TARGET_UNITS = "proportion"
SIGNED_PRODUCTION_DOMAINS = {"s8_numeric", "s8_twos_complement"}
ASSET_SHA256_COLUMN = "asset_sha256"


@dataclass(frozen=True)
class DatasetBuildOutcome:
    output_dir: Path
    design_count: int
    dataset_manifest_path: Path
    train_manifest_path: Path
    checksums_path: Path
    synthetic_smoke_only: bool
    performance_claim_allowed: bool


@dataclass(frozen=True)
class _DeltaSource:
    shape: tuple[int, int, int]
    output_dtype: np.dtype[Any]
    kind: str
    direct: np.ndarray | None = None
    approximate: np.ndarray | None = None
    exact: np.ndarray | None = None

    def write(self, path: Path, *, chunk_designs: int) -> None:
        output = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=self.output_dtype,
            shape=self.shape,
        )
        try:
            for start in range(0, self.shape[0], chunk_designs):
                stop = min(start + chunk_designs, self.shape[0])
                if self.direct is not None:
                    chunk = self._canonical_chunk(self.direct[start:stop], context="delta")
                else:
                    assert self.approximate is not None and self.exact is not None
                    chunk = self._derived_chunk(
                        self.approximate[start:stop],
                        self.exact[start:stop],
                    )
                output[start:stop] = chunk
            output.flush()
        finally:
            del output

    def _canonical_chunk(self, values: np.ndarray, *, context: str) -> np.ndarray:
        if np.issubdtype(self.output_dtype, np.integer):
            chunk = _integer_chunk(values, context=context)
            limits = np.iinfo(self.output_dtype)
            if np.any(chunk < limits.min) or np.any(chunk > limits.max):
                raise DataContractError(f"{context} values do not fit in canonical int32 delta")
            return chunk.astype(self.output_dtype, copy=False)
        chunk = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(chunk)):
            raise DataContractError(f"{context} must contain finite numeric values")
        limit = np.finfo(self.output_dtype).max
        if np.any(np.abs(chunk) > limit):
            raise DataContractError(f"{context} values do not fit in canonical float32 delta")
        return chunk.astype(self.output_dtype, copy=False)

    def _derived_chunk(self, approximate: np.ndarray, exact: np.ndarray) -> np.ndarray:
        if np.issubdtype(self.output_dtype, np.integer):
            approximate_chunk = _integer_chunk(approximate, context="approximate_products")
            exact_chunk = _integer_chunk(exact, context="exact_products")
            difference = approximate_chunk - exact_chunk
            return self._canonical_chunk(difference, context="derived delta")
        approximate_chunk = np.asarray(approximate, dtype=np.float64)
        exact_chunk = np.asarray(exact, dtype=np.float64)
        difference = approximate_chunk - exact_chunk
        return self._canonical_chunk(difference, context="derived delta")


def _integer_chunk(values: np.ndarray, *, context: str) -> np.ndarray:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.unsignedinteger):
        if array.size and np.max(array) > np.iinfo(np.int64).max:
            raise DataContractError(f"{context} values cannot be represented safely")
    return np.asarray(array, dtype=np.int64)


def _require_mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise DataContractError(f"{context} must be a JSON object")
    return value


def _resolve_relative(base: Path, value: object, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DataContractError(f"{context} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute():
        raise DataContractError(f"{context} must be relative to the build manifest")
    return (base / path).resolve()


def _load_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"cannot read {context} at {path}: {exc}") from exc
    return _require_mapping(value, context)


def _validated_mac_counts(values: object) -> np.ndarray:
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise DataContractError("mac_counts must be a numeric array") from exc
    if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(
        raw.dtype, np.complexfloating
    ):
        raise DataContractError("mac_counts must be a real numeric array")
    if not np.isfinite(raw).all():
        raise DataContractError("mac_counts must contain only finite values")
    if np.issubdtype(raw.dtype, np.floating) and np.any(raw != np.floor(raw)):
        raise DataContractError("mac_counts must contain integers")
    if np.any(raw <= 0):
        raise DataContractError("mac_counts must be positive")

    if np.issubdtype(raw.dtype, np.unsignedinteger):
        outside_int64 = np.any(raw > np.iinfo(np.int64).max)
    elif np.issubdtype(raw.dtype, np.floating):
        outside_int64 = np.any(raw.astype(np.longdouble) >= np.longdouble(1 << 63))
    else:
        outside_int64 = np.any(raw > np.iinfo(np.int64).max)
    if outside_int64:
        raise DataContractError("mac_counts cannot be represented as int64")
    try:
        converted = raw.astype(np.int64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise DataContractError("mac_counts cannot be represented as int64") from exc
    if np.any(converted.astype(np.longdouble) != raw.astype(np.longdouble)):
        raise DataContractError("mac_counts cannot be represented as int64")
    return converted


def _require_signed_production_domain(value: object, *, field: str) -> str:
    if value not in SIGNED_PRODUCTION_DOMAINS:
        raise DataContractError(
            f"{field} must use a signed S8S8 production domain; got {value!r}. "
            "u8_numeric is isolated to generic AME research features and cannot be "
            "materialized as a production dataset"
        )
    return str(value)


def _load_profile(metadata_path: Path, arrays_path: Path) -> ApplicationProfile:
    metadata = _load_json(metadata_path, "profile metadata")
    required_metadata = ("network_id", "dataset_id", "domain", "profile_evidence")
    missing_metadata = [key for key in required_metadata if key not in metadata]
    if missing_metadata:
        raise DataContractError(f"profile metadata is missing: {', '.join(missing_metadata)}")
    domain = _require_signed_production_domain(metadata["domain"], field="profile domain")
    if metadata["profile_evidence"] not in {"measured", "synthetic"}:
        raise DataContractError("profile_evidence must be 'measured' or 'synthetic'")

    try:
        with np.load(arrays_path, allow_pickle=False) as payload:
            required_arrays = (
                "layer_ids",
                "activation_histograms",
                "weight_histograms",
                "mac_counts",
                "propagation_gains",
            )
            missing_arrays = [key for key in required_arrays if key not in payload]
            if missing_arrays:
                raise DataContractError(f"profile arrays are missing: {', '.join(missing_arrays)}")
            layer_ids = tuple(str(value) for value in payload["layer_ids"].tolist())
            activation = np.asarray(payload["activation_histograms"], dtype=np.float64)
            weights = np.asarray(payload["weight_histograms"], dtype=np.float64)
            mac_counts = _validated_mac_counts(payload["mac_counts"])
            gains = np.asarray(payload["propagation_gains"], dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise DataContractError(f"cannot read profile arrays at {arrays_path}: {exc}") from exc

    layer_count = len(layer_ids)
    if layer_count == 0:
        raise DataContractError("profile must contain at least one layer")
    if any(not layer_id for layer_id in layer_ids) or len(set(layer_ids)) != layer_count:
        raise DataContractError("profile layer_ids must be non-empty and unique")
    if activation.shape != (layer_count, 256) or weights.shape != (layer_count, 256):
        raise DataContractError("profile histograms must have shape [layer_count, 256]")
    if mac_counts.shape != (layer_count,) or gains.shape != (layer_count,):
        raise DataContractError("mac_counts and propagation_gains must have shape [layer_count]")
    if not np.all(np.isfinite(activation)) or not np.all(np.isfinite(weights)):
        raise DataContractError("profile histograms must be finite")
    if np.any(activation < 0) or np.any(weights < 0):
        raise DataContractError("profile histograms must be non-negative")
    activation_sums = activation.sum(axis=1, keepdims=True)
    weight_sums = weights.sum(axis=1, keepdims=True)
    if np.any(activation_sums <= 0) or np.any(weight_sums <= 0):
        raise DataContractError("every profile histogram must have positive mass")
    if np.any(mac_counts <= 0):
        raise DataContractError("mac_counts must be positive")
    if not np.all(np.isfinite(gains)) or np.any(gains < 0):
        raise DataContractError("propagation_gains must be finite and non-negative")

    return ApplicationProfile(
        network_id=str(metadata["network_id"]),
        dataset_id=str(metadata["dataset_id"]),
        domain=domain,
        layer_ids=layer_ids,
        activation_histograms=activation / activation_sums,
        weight_histograms=weights / weight_sums,
        mac_counts=mac_counts,
        propagation_gains=gains,
        profile_evidence=metadata["profile_evidence"],
    )


def _read_csv(path: Path, context: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise DataContractError(f"{context} must have a header")
            return list(reader.fieldnames), list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataContractError(f"cannot read {context} at {path}: {exc}") from exc


def _normalized_sha256(value: object, *, field: str, design_id: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DataContractError(
            f"{field} must be a lowercase/uppercase SHA-256 digest for design {design_id!r}"
        )
    return text


def _load_design_index(
    path: Path,
) -> tuple[list[str], list[dict[str, object]], tuple[str, ...]]:
    fieldnames, rows = _read_csv(path, "design IDs")
    if "design_id" not in fieldnames:
        raise DataContractError("design IDs CSV must contain a design_id column")
    design_ids = tuple(row["design_id"].strip() for row in rows)
    if not design_ids or any(not design_id for design_id in design_ids):
        raise DataContractError("design_id values must be non-empty")
    if len(set(design_ids)) != len(design_ids):
        raise DataContractError("design_id values must be unique")
    canonical_rows: list[dict[str, object]] = [dict(row) for row in rows]
    for row, design_id in zip(canonical_rows, design_ids, strict=True):
        row["design_id"] = design_id
        for field in (BEHAVIOR_HASH_COLUMN, ASSET_SHA256_COLUMN):
            if field in fieldnames:
                row[field] = _normalized_sha256(
                    row.get(field),
                    field=field,
                    design_id=design_id,
                )
    return fieldnames, canonical_rows, design_ids


def _load_npy(path: Path, context: str) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise DataContractError(f"cannot read {context} at {path}: {exc}") from exc
    if not isinstance(array, np.ndarray):
        close = getattr(array, "close", None)
        if close is not None:
            close()
        raise DataContractError(f"{context} must be a .npy array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise DataContractError(f"{context} must contain real numeric values")
    return array


def _expected_exact_products(domain: str) -> np.ndarray:
    if domain == "s8_numeric":
        operands = np.arange(-128, 128, dtype=np.int32)
    elif domain == "s8_twos_complement":
        codes = np.arange(256, dtype=np.int32)
        operands = np.where(codes < 128, codes, codes - 256)
    else:
        raise DataContractError(f"unsupported arithmetic domain: {domain!r}")
    return np.multiply.outer(operands, operands)


def _validate_exact_products(exact: np.ndarray, *, domain: str) -> None:
    expected = _expected_exact_products(domain)
    for start in range(0, exact.shape[0], 64):
        stop = min(start + 64, exact.shape[0])
        matches = np.equal(exact[start:stop], expected[None, :, :])
        if np.all(matches):
            continue
        offset, row, column = np.argwhere(~matches)[0]
        design_index = start + int(offset)
        actual = np.asarray(exact[design_index, row, column]).item()
        wanted = int(expected[row, column])
        raise DataContractError(
            "exact_products does not match the exact multiplication table for "
            f"{domain} index order at design {design_index}, [{row}, {column}]: "
            f"got {actual!r}, expected {wanted}"
        )


def _load_delta_source(
    designs_manifest: Mapping[str, Any],
    base: Path,
    *,
    domain: str,
) -> _DeltaSource:
    has_delta = "delta" in designs_manifest
    has_approximate = "approximate_products" in designs_manifest
    has_exact = "exact_products" in designs_manifest
    if has_delta == (has_approximate or has_exact):
        raise DataContractError(
            "designs must specify either delta or both approximate_products and exact_products"
        )
    if has_delta:
        delta_path = _resolve_relative(base, designs_manifest["delta"], "designs.delta")
        delta = _load_npy(delta_path, "delta")
        arrays = (delta,)
    else:
        if not (has_approximate and has_exact):
            raise DataContractError(
                "designs must provide approximate_products and exact_products together"
            )
        approximate_path = _resolve_relative(
            base, designs_manifest["approximate_products"], "designs.approximate_products"
        )
        exact_path = _resolve_relative(base, designs_manifest["exact_products"], "designs.exact_products")
        approximate = _load_npy(approximate_path, "approximate_products")
        exact = _load_npy(exact_path, "exact_products")
        if approximate.shape != exact.shape:
            raise DataContractError("approximate_products and exact_products must have identical shapes")
        arrays = (approximate, exact)

    shape = arrays[0].shape
    if len(shape) != 3 or shape[1:] != (256, 256):
        raise DataContractError("delta must have shape [design_count, 256, 256]")
    if not has_delta:
        _validate_exact_products(exact, domain=domain)
    all_integer = all(np.issubdtype(array.dtype, np.integer) for array in arrays)
    output_dtype = np.dtype(np.int32 if all_integer else np.float32)
    if has_delta:
        return _DeltaSource(
            shape=shape,
            output_dtype=output_dtype,
            kind="delta",
            direct=delta,
        )
    return _DeltaSource(
        shape=shape,
        output_dtype=output_dtype,
        kind="derived_products",
        approximate=approximate,
        exact=exact,
    )


def _load_labels(path: Path, design_ids: tuple[str, ...]) -> tuple[list[str], list[dict[str, object]]]:
    fieldnames, rows = _read_csv(path, "labels")
    required = ("design_id", "top1_drop", "top5_drop")
    missing = [name for name in required if name not in fieldnames]
    if missing:
        raise DataContractError(f"labels CSV is missing: {', '.join(missing)}")

    by_id: dict[str, dict[str, object]] = {}
    for source_row in rows:
        design_id = source_row["design_id"].strip()
        if not design_id or design_id in by_id:
            raise DataContractError("label design_id values must be non-empty and unique")
        row: dict[str, object] = dict(source_row)
        row["design_id"] = design_id
        for field in (BEHAVIOR_HASH_COLUMN, ASSET_SHA256_COLUMN):
            if field in fieldnames:
                row[field] = _normalized_sha256(
                    source_row.get(field),
                    field=field,
                    design_id=design_id,
                )
        for target in ("top1_drop", "top5_drop"):
            try:
                value = float(source_row[target])
            except (TypeError, ValueError) as exc:
                raise DataContractError(f"{target} must be numeric for design {design_id!r}") from exc
            if not np.isfinite(value):
                raise DataContractError(f"{target} must be finite for design {design_id!r}")
            if not -1.0 <= value <= 1.0:
                raise DataContractError(
                    f"{target} must be in [-1, 1] proportion units for design {design_id!r}"
                )
            row[target] = value
        by_id[design_id] = row

    expected = set(design_ids)
    actual = set(by_id)
    if actual != expected:
        missing_ids = sorted(expected - actual)
        extra_ids = sorted(actual - expected)
        raise DataContractError(
            f"labels must align one-to-one with designs; missing={missing_ids}, extra={extra_ids}"
        )
    ordered_fields = ["design_id", "top1_drop", "top5_drop"] + [
        name for name in fieldnames if name not in required
    ]
    return ordered_fields, [by_id[design_id] for design_id in design_ids]


def _bind_rows_to_behavior(
    *,
    design_rows: list[dict[str, object]],
    labels: list[dict[str, object]],
    behavior_hashes: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str | None]:
    bound_designs = [dict(row) for row in design_rows]
    bound_labels = [dict(row) for row in labels]
    asset_digests: set[str] = set()
    for design_row, label_row, expected in zip(
        bound_designs,
        bound_labels,
        behavior_hashes,
        strict=True,
    ):
        design_id = str(design_row["design_id"])
        if str(label_row["design_id"]) != design_id:
            raise DataContractError("labels and design rows changed order during canonical binding")
        for source_name, row in (("design index", design_row), ("labels", label_row)):
            supplied = row.get(BEHAVIOR_HASH_COLUMN)
            if supplied is not None and supplied != expected:
                raise DataContractError(
                    f"{source_name} behavior_sha256 mismatch for design {design_id!r}: "
                    f"got {supplied}, expected {expected}"
                )
            row[BEHAVIOR_HASH_COLUMN] = expected
            asset_digest = row.get(ASSET_SHA256_COLUMN)
            if asset_digest is not None:
                asset_digests.add(str(asset_digest))
    if len(asset_digests) > 1:
        raise DataContractError(
            "canonical source mixes multiple asset_sha256 digests; behavior-bound rows "
            "must come from one frozen multiplier asset"
        )
    asset_digest = next(iter(asset_digests), None)
    if asset_digest is not None:
        for row in (*bound_designs, *bound_labels):
            supplied = row.get(ASSET_SHA256_COLUMN)
            if supplied is not None and supplied != asset_digest:
                raise DataContractError("asset_sha256 binding changed during canonical materialization")
            row[ASSET_SHA256_COLUMN] = asset_digest
    return bound_designs, bound_labels, asset_digest


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize(
    directory: Path,
    *,
    profile: ApplicationProfile,
    profile_metadata: Mapping[str, Any],
    design_ids: tuple[str, ...],
    design_fields: list[str],
    design_rows: list[dict[str, object]],
    delta_source: _DeltaSource,
    chunk_designs: int,
    label_fields: list[str],
    labels: list[dict[str, object]],
    source: str,
    build_manifest_sha256: str,
) -> None:
    delta_path = directory / "delta.npy"
    delta_source.write(delta_path, chunk_designs=chunk_designs)
    canonical_delta = np.load(delta_path, mmap_mode="r", allow_pickle=False)
    behavior_hashes = behavior_sha256_rows(canonical_delta)
    output_design_rows, output_labels, asset_sha256 = _bind_rows_to_behavior(
        design_rows=design_rows,
        labels=labels,
        behavior_hashes=behavior_hashes,
    )
    output_design_fields = list(design_fields)
    if BEHAVIOR_HASH_COLUMN not in output_design_fields:
        output_design_fields.append(BEHAVIOR_HASH_COLUMN)
    if asset_sha256 is not None and ASSET_SHA256_COLUMN not in output_design_fields:
        output_design_fields.append(ASSET_SHA256_COLUMN)
    _write_csv(
        directory / "design_ids.csv",
        output_design_fields,
        output_design_rows,
    )
    output_label_fields = list(label_fields)
    if BEHAVIOR_HASH_COLUMN not in output_label_fields:
        output_label_fields.append(BEHAVIOR_HASH_COLUMN)
    if asset_sha256 is not None and ASSET_SHA256_COLUMN not in output_label_fields:
        output_label_fields.append(ASSET_SHA256_COLUMN)
    _write_csv(directory / "labels.csv", output_label_fields, output_labels)

    canonical_profile_metadata = dict(profile_metadata)
    canonical_profile_metadata["error_convention"] = ERROR_CONVENTION
    canonical_profile_metadata["histograms_normalized"] = True
    _write_json(directory / "profile_metadata.json", canonical_profile_metadata)
    np.savez_compressed(
        directory / "profile_arrays.npz",
        layer_ids=np.asarray(profile.layer_ids),
        activation_histograms=np.asarray(profile.activation_histograms, dtype=np.float64),
        weight_histograms=np.asarray(profile.weight_histograms, dtype=np.float64),
        mac_counts=np.asarray(profile.mac_counts, dtype=np.int64),
        propagation_gains=np.asarray(profile.propagation_gains, dtype=np.float64),
    )

    synthetic_smoke_only = source == "synthetic" or profile.profile_evidence == "synthetic"
    dataset_manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source": source,
        "source_build_manifest_sha256": build_manifest_sha256,
        "network_id": profile.network_id,
        "dataset_id": profile.dataset_id,
        "domain": profile.domain,
        "error_convention": ERROR_CONVENTION,
        "design_count": len(design_ids),
        "delta_dtype": delta_source.output_dtype.name,
        "delta_source": delta_source.kind,
        "chunk_designs": chunk_designs,
        "targets": ["top1_drop", "top5_drop"],
        "target_units": TARGET_UNITS,
        "behavior_group": {
            "column": BEHAVIOR_HASH_COLUMN,
            "hash": "sha256",
            "schema": BEHAVIOR_HASH_SCHEMA,
        },
        "asset_sha256": asset_sha256,
        "synthetic_smoke_only": synthetic_smoke_only,
        "performance_claim_allowed": not synthetic_smoke_only,
        "files": {
            "profile_metadata": "profile_metadata.json",
            "profile_arrays": "profile_arrays.npz",
            "design_ids": "design_ids.csv",
            "delta": "delta.npy",
            "labels": "labels.csv",
        },
    }
    _write_json(directory / "dataset_manifest.json", dataset_manifest)

    train_manifest = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "target_units": TARGET_UNITS,
        "profile": {
            "metadata": "profile_metadata.json",
            "arrays": "profile_arrays.npz",
        },
        "designs": {
            "ids": "design_ids.csv",
            "delta": "delta.npy",
            "domain": profile.domain,
            "error_convention": ERROR_CONVENTION,
        },
        "labels": "labels.csv",
        "canonical_integrity": {
            "schema_version": 1,
            "dataset_manifest": "dataset_manifest.json",
            "checksums": "checksums.json",
        },
        "provenance": {
            "kind": "synthetic" if synthetic_smoke_only else "measured",
            "synthetic_smoke_only": synthetic_smoke_only,
            "performance_claim_allowed": not synthetic_smoke_only,
        },
        "execution": {},
        "training": {},
        "output": "model_bundle",
    }
    _write_json(directory / "train_manifest.json", train_manifest)

    checksum_names = (
        "delta.npy",
        "design_ids.csv",
        "labels.csv",
        "profile_metadata.json",
        "profile_arrays.npz",
        "dataset_manifest.json",
        "train_manifest.json",
    )
    _write_json(directory / "checksums.json", {name: _sha256(directory / name) for name in checksum_names})


def build_dataset_from_manifest(path: str | Path) -> DatasetBuildOutcome:
    manifest_path = Path(path).resolve()
    manifest = _load_json(manifest_path, "dataset build manifest")
    if manifest.get("schema_version") != BUILD_SCHEMA_VERSION:
        raise DataContractError(f"schema_version must be {BUILD_SCHEMA_VERSION!r}")
    if manifest.get("target_units") != TARGET_UNITS:
        raise DataContractError(f"target_units must be {TARGET_UNITS!r}")
    base = manifest_path.parent
    source = manifest.get("source")
    if source not in {"measured", "synthetic"}:
        raise DataContractError("source must be 'measured' or 'synthetic'")
    if manifest.get("error_convention", ERROR_CONVENTION) != ERROR_CONVENTION:
        raise DataContractError(f"error_convention must be {ERROR_CONVENTION!r}")
    chunk_designs = manifest.get("chunk_designs", 64)
    if isinstance(chunk_designs, bool) or not isinstance(chunk_designs, int) or chunk_designs < 1:
        raise DataContractError("chunk_designs must be a positive integer")

    profile_manifest = _require_mapping(manifest.get("profile"), "profile")
    designs_manifest = _require_mapping(manifest.get("designs"), "designs")
    profile_metadata_path = _resolve_relative(base, profile_manifest.get("metadata"), "profile.metadata")
    profile_arrays_path = _resolve_relative(base, profile_manifest.get("arrays"), "profile.arrays")
    design_ids_path = _resolve_relative(base, designs_manifest.get("design_ids"), "designs.design_ids")
    labels_path = _resolve_relative(base, manifest.get("labels"), "labels")
    output_dir = _resolve_relative(base, manifest.get("output_dir"), "output_dir")
    if output_dir.exists():
        raise DataContractError(f"output directory already exists: {output_dir}")

    profile_metadata = _load_json(profile_metadata_path, "profile metadata")
    profile = _load_profile(profile_metadata_path, profile_arrays_path)
    design_domain = designs_manifest.get("domain")
    design_domain = _require_signed_production_domain(
        design_domain,
        field="designs.domain",
    )
    if design_domain != profile.domain:
        raise DataContractError(
            f"design domain {design_domain!r} does not match profile domain {profile.domain!r}"
        )
    design_fields, design_rows, design_ids = _load_design_index(design_ids_path)
    delta_source = _load_delta_source(designs_manifest, base, domain=design_domain)
    if delta_source.shape[0] != len(design_ids):
        raise DataContractError(
            f"delta has {delta_source.shape[0]} designs but design_ids.csv has {len(design_ids)} rows"
        )
    label_fields, labels = _load_labels(labels_path, design_ids)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        _materialize(
            temporary,
            profile=profile,
            profile_metadata=profile_metadata,
            design_ids=design_ids,
            design_fields=design_fields,
            design_rows=design_rows,
            delta_source=delta_source,
            chunk_designs=chunk_designs,
            label_fields=label_fields,
            labels=labels,
            source=source,
            build_manifest_sha256=_sha256(manifest_path),
        )
        if output_dir.exists():
            raise DataContractError(f"output directory already exists: {output_dir}")
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    synthetic_smoke_only = source == "synthetic" or profile.profile_evidence == "synthetic"
    return DatasetBuildOutcome(
        output_dir=output_dir,
        design_count=len(design_ids),
        dataset_manifest_path=output_dir / "dataset_manifest.json",
        train_manifest_path=output_dir / "train_manifest.json",
        checksums_path=output_dir / "checksums.json",
        synthetic_smoke_only=synthetic_smoke_only,
        performance_claim_allowed=not synthetic_smoke_only,
    )
