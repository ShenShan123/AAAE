#!/usr/bin/env python
"""Build a resumable signed-S8S8 Transformer dataset for strict-50 training.

The dataset intentionally keeps two independent image protocols:

* frozen seed-1 / 1,000-image logits provide high-fidelity training labels;
* frozen seed-7 / 500-image logits provide the runtime observations used by
  the historical strict-50 feature pipeline.

This module is separate from :mod:`runtime.run_swin_s8s8_bulk`, whose JSONL
output is deliberately limited to aggregate true-1000 metrics.
"""

from __future__ import annotations

from aaae.paths import OUTPUT_ROOT

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


from aaae.accuracy_predictor.transformer.proxy_training.transformer_ame_bank import compute_clean_bank, compute_runtime_bank_from_logits
from aaae.paths import TRANSFORMER_CACHE_ROOT, TRANSFORMER_DATA_ROOT, TRANSFORMER_ROOT
from aaae.inference_runtime.transformer.dataset_generation.frozen_inputs import DEFAULT_IMAGE_ROOT, ensure_frozen_input_cache
from aaae.inference_runtime.transformer.dataset_generation.profiles import (
    DATASET_PROFILES,
    DEFAULT_MODEL_KEY,
    RAW_PROTOCOL_COUNT,
    RAW_PROTOCOL_SEED,
    TRUE_PROTOCOL_COUNT,
    TRUE_PROTOCOL_SEED,
    get_dataset_profile,
)


PROJECT_ROOT = TRANSFORMER_ROOT
from aaae.paths import EA2_ROOT as EA2_ROOT
END_TO_END_ROOT = EA2_ROOT / "end_to_end_prediction"
AI_MODELS_ROOT = EA2_ROOT / "ai_models"

SCHEMA_VERSION = "transformer_s8s8_linear48_strict50_true1000_raw500_signed_static_v4"
LEGACY_SWIN_SCHEMA_VERSION = "swin_s8s8_strict50_true1000_raw500_signed_static_v2"
LEGACY_TRANSFORMER_SCHEMA_VERSION = "transformer_s8s8_strict50_true1000_raw500_signed_static_v3"
DEFAULT_OUTPUT_ROOT = (
    OUTPUT_ROOT / "accuracy"
    / "swin_tiny_int8_s8s8_linear48_strict50_true1000_raw500_20k_v1"
)
DEFAULT_SOURCE_DATA_DIR = (
    END_TO_END_ROOT
    / "data"
    / "runs"
    / "swin_tiny_int8_s8s8_strict50_true1000_raw500_20k_v2"
    / "data"
)
DEFAULT_S8_ASSET_ROOT = (
    END_TO_END_ROOT / "datasets" / "multiplier" / "lowonly20k_s8s8_sign_magnitude_v1"
)
DEFAULT_NODE_REGISTRY = (
    END_TO_END_ROOT
    / "resources"
    / "data"
    / "swin_tiny_s8s8_linear48"
    / "node_registry"
    / "node_registry.json"
)
DEFAULT_DYNAMIC_MODEL_PATH = (
    AI_MODELS_ROOT
    / "onnx_models"
    / "classification"
    / "swin_tiny_patch4_window7_224_imagenet1k_int8_dynamic.onnx"
)
DEFAULT_TRUE1000_SUBSET = TRANSFORMER_DATA_ROOT / "frozen_protocols" / "calib_subset_1000.json"
DEFAULT_RAW500_SUBSET = TRANSFORMER_DATA_ROOT / "frozen_protocols" / "calib_subset_500.json"
DEFAULT_CALIBRATION_CACHE_DIR = TRANSFORMER_CACHE_ROOT / "calibration"
DEFAULT_TRUE1000_CACHE = DEFAULT_CALIBRATION_CACHE_DIR / "swin_tiny_patch4_window7_224_n1000_seed1.npy"
DEFAULT_RAW500_CACHE = DEFAULT_CALIBRATION_CACHE_DIR / "swin_tiny_patch4_window7_224_n500_seed7.npy"
DEFAULT_BATCH_SIZE = 16
DEFAULT_BLOCK_SIZE = 16
DEFAULT_S8_RESIDUAL_ABS_BOUND = 856
DEFAULT_BANK_BASIS_RANK = 32
EXPECTED_FULL_SPLIT_COUNTS = {"train": 14_000, "val": 3_000, "test": 3_000}
RUN_MARKER_SCHEMA_VERSION = "transformer_s8s8_dataset_run_marker_v1"
RUN_MARKER_GENERATOR = "aaae.inference_runtime.transformer.build_swin_s8s8_strict50_dataset"
RUN_MARKER_FILENAME = ".transformer_s8s8_dataset_run.json"
CANONICAL_BASELINE_ARTIFACT_KEYS = (
    "canonical_exact",
    "canonical_exact_contract",
    "strict50_bank_meta",
    "strict50_bank_meta_contract",
    "true1000_meta",
)
WORKER_TERMINATION_TIMEOUT_SECONDS = 5.0


def _bulk_module():
    """Import GPU/ONNX-heavy bulk helpers only on a runtime code path."""

    from aaae.inference_runtime.transformer import run_swin_s8s8_bulk

    return run_swin_s8s8_bulk


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha1_json(value: Any) -> str:
    return hashlib.sha1(_json_canonical(value).encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")
    try:
        np.save(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.{os.getpid()}{path.suffix}")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class FrozenImageSubset:
    """One immutable image protocol and its already-preprocessed tensors."""

    role: str
    sample_seed: int
    image_names: tuple[str, ...]
    indices: np.ndarray
    labels: np.ndarray
    inputs: np.ndarray


@dataclass(frozen=True)
class FrozenImageUnion:
    """Deduplicated input tensor bank plus positions in each frozen protocol."""

    image_names: tuple[str, ...]
    indices: np.ndarray
    labels: np.ndarray
    inputs: np.ndarray
    true1000_positions: np.ndarray
    raw500_positions: np.ndarray


def _validate_frozen_subset(subset: FrozenImageSubset) -> None:
    count = len(subset.image_names)
    if not str(subset.role):
        raise ValueError("frozen subset role must be non-empty")
    if count <= 0:
        raise ValueError(f"{subset.role}: frozen subset is empty")
    if len(set(subset.image_names)) != count:
        raise ValueError(f"{subset.role}: image_names must be unique")
    indices = np.asarray(subset.indices, dtype=np.int64)
    labels = np.asarray(subset.labels, dtype=np.int64)
    inputs = np.asarray(subset.inputs, dtype=np.float32)
    if indices.shape != (count,) or labels.shape != (count,):
        raise ValueError(f"{subset.role}: image names, indices, and labels must have the same length")
    if inputs.ndim != 4 or inputs.shape[0] != count:
        raise ValueError(f"{subset.role}: inputs must be [N,C,H,W] aligned with image_names, got {inputs.shape}")
    if np.any(indices < 0):
        raise ValueError(f"{subset.role}: indices must be non-negative")
    if np.any(labels < 0):
        raise ValueError(f"{subset.role}: labels must be non-negative")


def build_frozen_union(*, true1000: FrozenImageSubset, raw500: FrozenImageSubset) -> FrozenImageUnion:
    """Deduplicate two frozen subsets while retaining each original order.

    The true-1000 ordering is placed first to make label generation easy.
    Images that also occur in raw500 are referenced through ``raw500_positions``
    rather than copied or re-preprocessed.
    """

    _validate_frozen_subset(true1000)
    _validate_frozen_subset(raw500)
    if str(true1000.role) != "true1000" or str(raw500.role) != "raw500":
        raise ValueError("build_frozen_union requires roles 'true1000' and 'raw500'")

    names: list[str] = []
    indices: list[int] = []
    labels: list[int] = []
    tensors: list[np.ndarray] = []
    positions: dict[str, int] = {}

    def append_subset(subset: FrozenImageSubset) -> np.ndarray:
        subset_positions = np.empty((len(subset.image_names),), dtype=np.int64)
        subset_indices = np.asarray(subset.indices, dtype=np.int64)
        subset_labels = np.asarray(subset.labels, dtype=np.int64)
        subset_inputs = np.asarray(subset.inputs, dtype=np.float32)
        for local_index, image_name in enumerate(subset.image_names):
            existing = positions.get(str(image_name))
            if existing is None:
                existing = len(names)
                positions[str(image_name)] = existing
                names.append(str(image_name))
                indices.append(int(subset_indices[local_index]))
                labels.append(int(subset_labels[local_index]))
                tensors.append(np.asarray(subset_inputs[local_index], dtype=np.float32))
            else:
                if int(indices[existing]) != int(subset_indices[local_index]):
                    raise ValueError(f"{image_name}: frozen subset indices disagree")
                if int(labels[existing]) != int(subset_labels[local_index]):
                    raise ValueError(f"{image_name}: frozen subset labels disagree")
                if not np.array_equal(tensors[existing], subset_inputs[local_index]):
                    raise ValueError(f"{image_name}: frozen input caches disagree")
            subset_positions[local_index] = existing
        return subset_positions

    true_positions = append_subset(true1000)
    raw_positions = append_subset(raw500)
    return FrozenImageUnion(
        image_names=tuple(names),
        indices=np.asarray(indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        inputs=np.ascontiguousarray(np.stack(tensors, axis=0), dtype=np.float32),
        true1000_positions=true_positions,
        raw500_positions=raw_positions,
    )


def runtime_response_from_logits(
    approximate_logits: np.ndarray,
    clean_bank: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    """Return the historical strict-50 response for one approximate run.

    ``clean_bank`` is the canonical exact reference created by
    :func:`proxy_training.transformer_ame_bank.compute_clean_bank`.  Keeping
    the public wrapper here gives the dataset builder a stable seam while
    preserving the exact old feature semantics.
    """

    required = (
        "clean_margin_top1",
        "clean_margin_top5",
        "exact_pred_class",
        "true_label",
        "exact_top1_correct_bits",
        "exact_top5_correct_bits",
    )
    missing = [name for name in required if name not in clean_bank]
    if missing:
        raise KeyError(f"clean_bank is missing required arrays: {missing}")
    return compute_runtime_bank_from_logits(
        np.asarray(approximate_logits, dtype=np.float32),
        clean_margin_top1=np.asarray(clean_bank["clean_margin_top1"], dtype=np.float32),
        clean_margin_top5=np.asarray(clean_bank["clean_margin_top5"], dtype=np.float32),
        exact_pred_class=np.asarray(clean_bank["exact_pred_class"], dtype=np.int64),
        true_label=np.asarray(clean_bank["true_label"], dtype=np.int64),
        exact_top1_correct_bits=np.asarray(clean_bank["exact_top1_correct_bits"], dtype=np.uint8),
        exact_top5_correct_bits=np.asarray(clean_bank["exact_top5_correct_bits"], dtype=np.uint8),
    )


def validate_full_population_splits(frame) -> dict[str, int]:
    """Require the retained 20k train/validation/test membership exactly."""

    if "split" not in frame:
        raise KeyError("configuration has no split column")
    actual = {
        str(name): int(count)
        for name, count in frame["split"].astype(str).value_counts().items()
    }
    if actual != EXPECTED_FULL_SPLIT_COUNTS:
        raise ValueError(
            f"full 20k split contract changed: {actual}; expected {EXPECTED_FULL_SPLIT_COUNTS}"
        )
    return actual


def validate_zero_residual_identity(
    identity_logits: np.ndarray,
    canonical_logits: np.ndarray,
) -> dict[str, Any]:
    """Gate approximate execution on an exact semantic-zero residual pass."""

    identity = np.asarray(identity_logits, dtype=np.float32)
    canonical = np.asarray(canonical_logits, dtype=np.float32)
    if identity.shape != canonical.shape or not np.isfinite(identity).all():
        raise ValueError(
            f"zero-residual identity pass returned {identity.shape}; expected finite {canonical.shape}"
        )
    difference = np.abs(identity - canonical)
    metrics = {
        "max_abs_logit_difference_vs_canonical": float(np.max(difference, initial=0.0)),
        "mean_abs_logit_difference_vs_canonical": float(
            np.mean(difference, dtype=np.float64)
        ),
        "top1_changed_count_vs_canonical": int(
            np.count_nonzero(np.argmax(identity, axis=1) != np.argmax(canonical, axis=1))
        ),
    }
    if not np.array_equal(identity, canonical):
        raise RuntimeError(f"zero-residual identity gate failed: {metrics}")
    return metrics


def _selection_hash(image_names: Sequence[str], labels: np.ndarray) -> str:
    return _sha1_json(
        {
            "image_names": [str(name) for name in image_names],
            "labels": np.asarray(labels, dtype=np.int64).tolist(),
        }
    )


def _validate_image_names_against_indices(*, image_names: Sequence[str], indices: np.ndarray, role: str) -> None:
    for image_name, index in zip(image_names, np.asarray(indices, dtype=np.int64), strict=True):
        stem = Path(str(image_name)).stem
        try:
            parsed = int(stem.rsplit("_", 1)[1]) - 1
        except (IndexError, ValueError) as exc:
            raise ValueError(f"{role}: cannot derive ImageNet index from {image_name!r}") from exc
        if parsed != int(index):
            raise ValueError(f"{role}: {image_name} maps to {parsed}, not frozen index {int(index)}")


def _load_frozen_subset(
    *,
    role: str,
    subset_path: Path,
    cache_path: Path,
    expected_sample_count: int,
    expected_sample_seed: int,
    labels: np.ndarray,
    model_key: str = DEFAULT_MODEL_KEY,
    preprocess_key: str = DEFAULT_MODEL_KEY,
) -> tuple[FrozenImageSubset, dict[str, Any], dict[str, Any]]:
    subset_path = Path(subset_path).resolve()
    cache_path = Path(cache_path).resolve()
    cache_meta_path = cache_path.with_suffix(".json")
    if not subset_path.is_file():
        raise FileNotFoundError(subset_path)
    if not cache_path.is_file() or not cache_meta_path.is_file():
        raise FileNotFoundError(f"{role}: frozen cache requires {cache_path} and {cache_meta_path}")
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    image_names = tuple(str(value) for value in payload.get("image_names", ()))
    indices = np.asarray(payload.get("indices", ()), dtype=np.int64)
    actual_labels = np.asarray(labels, dtype=np.int64)
    if int(payload.get("sample_count", -1)) != int(expected_sample_count):
        raise ValueError(f"{role}: subset sample_count does not match {expected_sample_count}")
    if int(payload.get("sample_seed", -1)) != int(expected_sample_seed):
        raise ValueError(f"{role}: subset sample_seed does not match {expected_sample_seed}")
    if len(image_names) != int(expected_sample_count) or indices.shape != (int(expected_sample_count),):
        raise ValueError(f"{role}: frozen image_names/indices do not match sample_count")
    if actual_labels.shape != (int(expected_sample_count),):
        raise ValueError(f"{role}: labels do not match sample_count")
    _validate_image_names_against_indices(image_names=image_names, indices=indices, role=role)
    if "labels" in payload:
        frozen_labels = np.asarray(payload["labels"], dtype=np.int64)
        if not np.array_equal(frozen_labels, actual_labels):
            raise ValueError(f"{role}: frozen labels differ from the ImageNet ground-truth mapping")

    cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    selection_sha1 = _selection_hash(image_names, actual_labels)
    expected_meta = {
        "model_key": str(model_key),
        "preprocess_key": str(preprocess_key),
        "sample_count": int(expected_sample_count),
        "sample_seed": int(expected_sample_seed),
        "selection_hash": selection_sha1,
        "dtype": "float32",
        "cached_shape": [int(expected_sample_count), 3, 224, 224],
    }
    mismatches = [
        f"{key}={cache_meta.get(key)!r} != {value!r}"
        for key, value in expected_meta.items()
        if cache_meta.get(key) != value
    ]
    if mismatches:
        raise ValueError(f"{role}: frozen cache metadata is incompatible: {'; '.join(mismatches)}")
    inputs = np.load(cache_path, mmap_mode="r")
    if inputs.dtype != np.float32 or tuple(inputs.shape) != tuple(expected_meta["cached_shape"]):
        raise ValueError(f"{role}: frozen input cache has {inputs.dtype} {inputs.shape}, expected float32 {expected_meta['cached_shape']}")
    if cache_meta.get("cache_sha1") != _sha1_file(cache_path):
        raise ValueError(f"{role}: frozen input cache content digest does not match its metadata")
    subset = FrozenImageSubset(
        role=role,
        sample_seed=int(expected_sample_seed),
        image_names=image_names,
        indices=indices,
        labels=actual_labels,
        inputs=inputs,
    )
    _validate_frozen_subset(subset)
    return subset, payload, cache_meta


def load_frozen_image_protocols(
    *,
    true1000_subset_path: Path = DEFAULT_TRUE1000_SUBSET,
    raw500_subset_path: Path = DEFAULT_RAW500_SUBSET,
    true1000_cache_path: Path = DEFAULT_TRUE1000_CACHE,
    raw500_cache_path: Path = DEFAULT_RAW500_CACHE,
    model_key: str = DEFAULT_MODEL_KEY,
    preprocess_key: str = DEFAULT_MODEL_KEY,
    true_sample_count: int = TRUE_PROTOCOL_COUNT,
    raw_sample_count: int = RAW_PROTOCOL_COUNT,
    true_sample_seed: int = TRUE_PROTOCOL_SEED,
    raw_sample_seed: int = RAW_PROTOCOL_SEED,
) -> tuple[FrozenImageSubset, FrozenImageSubset, dict[str, Any]]:
    """Load and cross-check the immutable true1000 and raw500 protocols."""

    true_payload = json.loads(Path(true1000_subset_path).read_text(encoding="utf-8"))
    raw_payload = json.loads(Path(raw500_subset_path).read_text(encoding="utf-8"))
    all_indices = np.concatenate(
        [
            np.asarray(true_payload.get("indices", ()), dtype=np.int64),
            np.asarray(raw_payload.get("indices", ()), dtype=np.int64),
        ]
    )
    if all_indices.size == 0 or np.any(all_indices < 0):
        raise ValueError("frozen image subsets contain no valid ImageNet indices")
    bulk = _bulk_module()
    labels_all = np.asarray(bulk.imagenet_gt_indices(int(np.max(all_indices)) + 1), dtype=np.int64)
    if labels_all.shape[0] <= int(np.max(all_indices)):
        raise ValueError("ImageNet ground-truth table does not cover frozen subset indices")
    true_labels = labels_all[np.asarray(true_payload["indices"], dtype=np.int64)]
    raw_labels = labels_all[np.asarray(raw_payload["indices"], dtype=np.int64)]
    true1000, _true_payload, true_cache_meta = _load_frozen_subset(
        role="true1000",
        subset_path=Path(true1000_subset_path),
        cache_path=Path(true1000_cache_path),
        expected_sample_count=int(true_sample_count),
        expected_sample_seed=int(true_sample_seed),
        labels=true_labels,
        model_key=str(model_key),
        preprocess_key=str(preprocess_key),
    )
    raw500, _raw_payload, raw_cache_meta = _load_frozen_subset(
        role="raw500",
        subset_path=Path(raw500_subset_path),
        cache_path=Path(raw500_cache_path),
        expected_sample_count=int(raw_sample_count),
        expected_sample_seed=int(raw_sample_seed),
        labels=raw_labels,
        model_key=str(model_key),
        preprocess_key=str(preprocess_key),
    )
    if true_cache_meta.get("preprocess_source_sha1") != raw_cache_meta.get("preprocess_source_sha1"):
        raise ValueError("true1000 and raw500 caches use different preprocessing source revisions")
    provenance = {
        "true1000": {
            "subset_path": str(Path(true1000_subset_path).resolve()),
            "subset_sha1": _sha1_file(Path(true1000_subset_path)),
            "cache_path": str(Path(true1000_cache_path).resolve()),
            "cache_sha1": _sha1_file(Path(true1000_cache_path)),
            "cache_meta_path": str(Path(true1000_cache_path).with_suffix(".json").resolve()),
            "cache_meta_sha1": _sha1_file(Path(true1000_cache_path).with_suffix(".json")),
            "sample_count": int(len(true1000.image_names)),
            "sample_seed": int(true1000.sample_seed),
            "selection_sha1": _selection_hash(true1000.image_names, true1000.labels),
        },
        "raw500": {
            "subset_path": str(Path(raw500_subset_path).resolve()),
            "subset_sha1": _sha1_file(Path(raw500_subset_path)),
            "cache_path": str(Path(raw500_cache_path).resolve()),
            "cache_sha1": _sha1_file(Path(raw500_cache_path)),
            "cache_meta_path": str(Path(raw500_cache_path).with_suffix(".json").resolve()),
            "cache_meta_sha1": _sha1_file(Path(raw500_cache_path).with_suffix(".json")),
            "sample_count": int(len(raw500.image_names)),
            "sample_seed": int(raw500.sample_seed),
            "selection_sha1": _selection_hash(raw500.image_names, raw500.labels),
        },
        "preprocess_source_sha1": str(true_cache_meta["preprocess_source_sha1"]),
    }
    return true1000, raw500, provenance


def _paths(output_root: Path) -> dict[str, Path]:
    root = Path(os.path.abspath(os.fspath(Path(output_root).expanduser())))
    shared = root / "shared"
    data = root / "data"
    return {
        "root": root,
        "run_marker": root / RUN_MARKER_FILENAME,
        "run_config": root / "run_config.json",
        "population": root / "population",
        "selected_rows": root / "population" / "selected_rows.csv",
        "shared": shared,
        "union_inputs": shared / "union_inputs.npy",
        "union_labels": shared / "union_labels.npy",
        "union_positions": shared / "union_positions.npz",
        "image_protocols": shared / "frozen_image_protocols.json",
        "input_contract": shared / "input_contract.json",
        "canonical_exact": shared / "canonical_exact_baseline.npz",
        "canonical_exact_contract": shared / "canonical_exact_contract.json",
        "strict50_bank_meta": shared / "strict50_bank_meta.npz",
        "strict50_bank_meta_contract": shared / "strict50_bank_meta_contract.json",
        "true1000_meta": shared / "true1000_exact_meta.npz",
        "blocks": root / "blocks",
        "logs": root / "logs",
        "workers": root / "workers",
        "data": data,
        "common3103": data / "common3103.npy",
        "configuration_index": data / "configuration_index.csv",
        "runtime_map": data / "runtime_design_id_map.csv",
        "merged_index": data / "merged_index.csv",
        "targets": data / "targets.npy",
        "labels": data / "labels.csv",
        "runtime_bank": data / "runtime_bank.npz",
        "true1000_label_bank": data / "true1000_label_bank.npz",
        "manifest": data / "manifest.json",
    }


def _write_run_marker(paths: Mapping[str, Path]) -> None:
    root = Path(paths["root"]).resolve(strict=False)
    _atomic_write_json(
        Path(paths["run_marker"]),
        {
            "schema_version": RUN_MARKER_SCHEMA_VERSION,
            "generator": RUN_MARKER_GENERATOR,
            "dataset_schema_version": SCHEMA_VERSION,
            "output_root": str(root),
            "created_at_unix": time.time(),
        },
    )


def _has_valid_run_marker(root: Path) -> bool:
    marker_path = Path(root) / RUN_MARKER_FILENAME
    if not marker_path.is_file() or marker_path.is_symlink():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("schema_version") == RUN_MARKER_SCHEMA_VERSION
        and marker.get("generator") == RUN_MARKER_GENERATOR
        and marker.get("dataset_schema_version") == SCHEMA_VERSION
        and marker.get("output_root") == str(Path(root).resolve(strict=False))
    )


def _has_valid_legacy_run_config(root: Path) -> bool:
    config_path = Path(root) / "run_config.json"
    if not config_path.is_file() or config_path.is_symlink():
        return False
    try:
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        contract = run_config["contract"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    schemas = {
        SCHEMA_VERSION,
        LEGACY_SWIN_SCHEMA_VERSION,
        LEGACY_TRANSFORMER_SCHEMA_VERSION,
    }
    return (
        isinstance(contract, dict)
        and run_config.get("schema_version") in schemas
        and contract.get("schema_version") in schemas
        and run_config.get("run_fingerprint") == _sha1_json(contract)
    )


def _safe_remove_output_root(output_root: Path) -> None:
    """Remove only a positively identified dataset run, never a broad path."""

    root = Path(os.path.abspath(os.fspath(Path(output_root).expanduser())))
    resolved = root.resolve(strict=False)
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        END_TO_END_ROOT.resolve(),
        (END_TO_END_ROOT / "data").resolve(),
        (END_TO_END_ROOT / "data" / "runs").resolve(),
    }
    if resolved in protected or any(resolved == parent for parent in END_TO_END_ROOT.resolve().parents):
        raise ValueError(f"refusing to remove protected output root: {root}")
    if root.is_symlink():
        raise ValueError(f"refusing to remove symlink output root: {root}")
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError(f"dataset output root is not a directory: {root}")
    if not (_has_valid_run_marker(root) or _has_valid_legacy_run_config(root)):
        raise ValueError(f"not a recognized Transformer dataset run: {root}")
    shutil.rmtree(root)


def _validate_output_root_is_separate(output_root: Path, input_paths: Sequence[Path]) -> None:
    root = Path(output_root)
    if root.is_symlink():
        raise ValueError(f"output root must not be a symlink: {root}")
    resolved_root = root.resolve(strict=False)
    for input_path in input_paths:
        resolved_input = Path(input_path).resolve(strict=False)
        if (
            resolved_input == resolved_root
            or resolved_root in resolved_input.parents
            or resolved_input in resolved_root.parents
        ):
            raise ValueError(
                f"output root {resolved_root} must not overlap input path {resolved_input}"
            )


def _subset_json(subset: FrozenImageSubset) -> dict[str, Any]:
    return {
        "role": str(subset.role),
        "sample_seed": int(subset.sample_seed),
        "sample_count": int(len(subset.image_names)),
        "image_names": list(subset.image_names),
        "indices": np.asarray(subset.indices, dtype=np.int64).tolist(),
        "labels": np.asarray(subset.labels, dtype=np.int64).tolist(),
        "selection_sha1": _selection_hash(subset.image_names, subset.labels),
    }


def _frozen_union_artifact_sha1(paths: Mapping[str, Path]) -> dict[str, str]:
    """Return file-level integrity anchors for the persisted frozen union."""

    return {
        name: _sha1_file(Path(paths[name]))
        for name in ("union_inputs", "union_labels", "union_positions", "image_protocols")
    }


def _materialize_frozen_union(
    *,
    paths: Mapping[str, Path],
    true1000: FrozenImageSubset,
    raw500: FrozenImageSubset,
    provenance: Mapping[str, Any],
) -> FrozenImageUnion:
    """Persist the deduplicated union without materializing an extra 850 MiB RAM copy."""

    _validate_frozen_subset(true1000)
    _validate_frozen_subset(raw500)
    if tuple(true1000.inputs.shape[1:]) != tuple(raw500.inputs.shape[1:]):
        raise ValueError("frozen input caches have different C/H/W dimensions")

    true_names = tuple(true1000.image_names)
    position_by_name = {name: index for index, name in enumerate(true_names)}
    union_names = list(true_names)
    union_indices = [int(value) for value in np.asarray(true1000.indices, dtype=np.int64)]
    union_labels = [int(value) for value in np.asarray(true1000.labels, dtype=np.int64)]
    raw_positions = np.empty((len(raw500.image_names),), dtype=np.int64)
    for raw_index, image_name in enumerate(raw500.image_names):
        existing = position_by_name.get(image_name)
        if existing is None:
            existing = len(union_names)
            position_by_name[image_name] = existing
            union_names.append(image_name)
            union_indices.append(int(raw500.indices[raw_index]))
            union_labels.append(int(raw500.labels[raw_index]))
        else:
            if int(true1000.indices[existing]) != int(raw500.indices[raw_index]):
                raise ValueError(f"{image_name}: true1000/raw500 frozen indices disagree")
            if int(true1000.labels[existing]) != int(raw500.labels[raw_index]):
                raise ValueError(f"{image_name}: true1000/raw500 labels disagree")
            if not np.array_equal(true1000.inputs[existing], raw500.inputs[raw_index]):
                raise ValueError(f"{image_name}: true1000/raw500 input tensors disagree")
        raw_positions[raw_index] = existing
    true_positions = np.arange(len(true1000.image_names), dtype=np.int64)
    union_count = len(union_names)
    union_labels_array = np.asarray(union_labels, dtype=np.int64)
    union_indices_array = np.asarray(union_indices, dtype=np.int64)
    expected_contract = {
        "schema_version": SCHEMA_VERSION,
        "input_dtype": "float32",
        "input_shape": [union_count, *[int(value) for value in true1000.inputs.shape[1:]]],
        "union_count": int(union_count),
        "true1000_positions_sha1": hashlib.sha1(true_positions.tobytes()).hexdigest(),
        "raw500_positions_sha1": hashlib.sha1(raw_positions.tobytes()).hexdigest(),
        "union_selection_sha1": _sha1_json(
            {
                "image_names": union_names,
                "indices": union_indices_array.tolist(),
                "labels": union_labels_array.tolist(),
            }
        ),
        "frozen_provenance": dict(provenance),
    }
    existing_files = (
        "union_inputs",
        "union_labels",
        "union_positions",
        "image_protocols",
        "input_contract",
    )
    if all(Path(paths[name]).is_file() for name in existing_files):
        contract = json.loads(Path(paths["input_contract"]).read_text(encoding="utf-8"))
        cached_inputs = np.load(paths["union_inputs"], mmap_mode="r")
        cached_labels = np.load(paths["union_labels"], allow_pickle=False)
        cached_positions = np.load(paths["union_positions"], allow_pickle=False)
        artifact_sha1 = _frozen_union_artifact_sha1(paths)
        if (
            contract == {**expected_contract, "artifact_sha1": artifact_sha1}
            and cached_inputs.dtype == np.float32
            and tuple(cached_inputs.shape) == tuple(expected_contract["input_shape"])
            and np.array_equal(cached_labels, union_labels_array)
            and np.array_equal(cached_positions["true1000_positions"], true_positions)
            and np.array_equal(cached_positions["raw500_positions"], raw_positions)
        ):
            return FrozenImageUnion(
                image_names=tuple(union_names),
                indices=union_indices_array,
                labels=union_labels_array,
                inputs=cached_inputs,
                true1000_positions=true_positions,
                raw500_positions=raw_positions,
            )
        raise RuntimeError("existing frozen union is incomplete or incompatible; use a new output root")
    if any(Path(paths[name]).exists() for name in existing_files):
        raise RuntimeError("existing frozen union is incomplete; use a new output root")

    Path(paths["shared"]).mkdir(parents=True, exist_ok=True)
    temporary = Path(paths["union_inputs"]).with_name(
        f".{Path(paths['union_inputs']).stem}.tmp.{os.getpid()}.npy"
    )
    try:
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=tuple(expected_contract["input_shape"]),
        )
        try:
            output[: len(true1000.image_names)] = true1000.inputs
            for raw_index, target_index in enumerate(raw_positions):
                if int(target_index) >= len(true1000.image_names):
                    output[int(target_index)] = raw500.inputs[raw_index]
            output.flush()
        finally:
            del output
        os.replace(temporary, paths["union_inputs"])
    finally:
        if temporary.exists():
            temporary.unlink()
    _atomic_save_npy(paths["union_labels"], union_labels_array)
    _atomic_save_npz(
        paths["union_positions"],
        true1000_positions=true_positions,
        raw500_positions=raw_positions,
    )
    _atomic_write_json(
        paths["image_protocols"],
        {
            "schema_version": SCHEMA_VERSION,
            "true1000": _subset_json(true1000),
            "raw500": _subset_json(raw500),
            "union": {
                "sample_count": int(union_count),
                "image_names": union_names,
                "indices": union_indices_array.tolist(),
                "labels": union_labels_array.tolist(),
                "selection_sha1": expected_contract["union_selection_sha1"],
                "overlap_count": int(len(true1000.image_names) + len(raw500.image_names) - union_count),
            },
        },
    )
    _atomic_write_json(
        paths["input_contract"],
        {**expected_contract, "artifact_sha1": _frozen_union_artifact_sha1(paths)},
    )
    return FrozenImageUnion(
        image_names=tuple(union_names),
        indices=union_indices_array,
        labels=union_labels_array,
        inputs=np.load(paths["union_inputs"], mmap_mode="r"),
        true1000_positions=true_positions,
        raw500_positions=raw_positions,
    )


def _require_binary_matrix(
    payload: Mapping[str, Any],
    *,
    key: str,
    row_count: int,
    sample_count: int,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"runtime payload is missing {key}")
    values = np.asarray(payload[key], dtype=np.uint8)
    if values.shape != (int(row_count), int(sample_count)):
        raise ValueError(f"{key} has {values.shape}, expected {(int(row_count), int(sample_count))}")
    if np.any(values > 1):
        raise ValueError(f"{key} contains values outside [0, 1]")
    return values


def _require_float_vector(payload: Mapping[str, Any], *, key: str, row_count: int) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"runtime payload is missing {key}")
    values = np.asarray(payload[key], dtype=np.float32)
    if values.shape != (int(row_count),):
        raise ValueError(f"{key} has {values.shape}, expected {(int(row_count),)}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains non-finite values")
    return values


def _require_float_matrix(
    payload: Mapping[str, Any],
    *,
    key: str,
    row_count: int,
    sample_count: int,
) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"runtime payload is missing {key}")
    values = np.asarray(payload[key], dtype=np.float32)
    if values.shape != (int(row_count), int(sample_count)):
        raise ValueError(f"{key} has {values.shape}, expected {(int(row_count), int(sample_count))}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{key} contains non-finite values")
    return values


def _recompute_drop(exact_bits: np.ndarray, approximate_bits: np.ndarray) -> np.ndarray:
    exact = np.asarray(exact_bits, dtype=np.uint8)
    approximate = np.asarray(approximate_bits, dtype=np.uint8)
    if exact.ndim != 1 or approximate.ndim != 2 or approximate.shape[1] != exact.shape[0]:
        raise ValueError("correctness bits do not align for drop recomputation")
    return np.asarray(
        np.mean(
            exact.astype(np.float32)[None, :] - approximate.astype(np.float32),
            axis=1,
            dtype=np.float64,
        ),
        dtype=np.float32,
    )


def validate_runtime_payload(
    payload: Mapping[str, Any],
    *,
    expected_row_index: np.ndarray,
    expected_public_design_id: np.ndarray,
    expected_runtime_design_id: np.ndarray,
    true1000_clean: Mapping[str, np.ndarray],
    raw500_clean: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Validate one or merged block and return its true1000 training targets.

    ``raw500_*`` values remain in the payload for strict-50 feature extraction.
    The returned two-column array is deliberately recomputed only from the
    true1000 correctness bits, so a 500-image observation cannot become a
    training label through a naming accident.
    """

    expected_rows = np.asarray(expected_row_index, dtype=np.int64)
    expected_public = np.asarray(expected_public_design_id).astype(str)
    expected_runtime = np.asarray(expected_runtime_design_id).astype(str)
    row_count = int(expected_rows.shape[0])
    if expected_public.shape != (row_count,) or expected_runtime.shape != (row_count,):
        raise ValueError("expected row and design-ID arrays must share one dimension")
    for key in ("row_index", "public_design_id", "runtime_design_id"):
        if key not in payload:
            raise KeyError(f"runtime payload is missing {key}")
    if not np.array_equal(np.asarray(payload["row_index"], dtype=np.int64), expected_rows):
        raise ValueError("runtime payload row_index does not match the frozen block rows")
    if not np.array_equal(np.asarray(payload["public_design_id"]).astype(str), expected_public):
        raise ValueError("runtime payload public_design_id does not match the frozen block rows")
    if not np.array_equal(np.asarray(payload["runtime_design_id"]).astype(str), expected_runtime):
        raise ValueError("runtime payload runtime_design_id does not match the frozen block rows")

    required_clean = ("exact_top1_correct_bits", "exact_top5_correct_bits")
    missing_true = [name for name in required_clean if name not in true1000_clean]
    missing_raw = [name for name in required_clean if name not in raw500_clean]
    if missing_true or missing_raw:
        raise KeyError(f"clean correctness metadata missing true1000={missing_true}, raw500={missing_raw}")
    true_top1_clean = np.asarray(true1000_clean["exact_top1_correct_bits"], dtype=np.uint8).reshape(-1)
    true_top5_clean = np.asarray(true1000_clean["exact_top5_correct_bits"], dtype=np.uint8).reshape(-1)
    raw_top1_clean = np.asarray(raw500_clean["exact_top1_correct_bits"], dtype=np.uint8).reshape(-1)
    raw_top5_clean = np.asarray(raw500_clean["exact_top5_correct_bits"], dtype=np.uint8).reshape(-1)
    if true_top1_clean.shape != true_top5_clean.shape or raw_top1_clean.shape != raw_top5_clean.shape:
        raise ValueError("top1/top5 clean correctness arrays do not align")
    if np.any(true_top1_clean > 1) or np.any(true_top5_clean > 1):
        raise ValueError("true1000 clean correctness arrays contain values outside [0, 1]")
    if np.any(raw_top1_clean > 1) or np.any(raw_top5_clean > 1):
        raise ValueError("raw500 clean correctness arrays contain values outside [0, 1]")

    true_top1 = _require_binary_matrix(
        payload,
        key="true1000_approx_top1_correct_bits",
        row_count=row_count,
        sample_count=len(true_top1_clean),
    )
    true_top5 = _require_binary_matrix(
        payload,
        key="true1000_approx_top5_correct_bits",
        row_count=row_count,
        sample_count=len(true_top5_clean),
    )
    raw_top1 = _require_binary_matrix(
        payload,
        key="approx_top1_correct_bits",
        row_count=row_count,
        sample_count=len(raw_top1_clean),
    )
    raw_top5 = _require_binary_matrix(
        payload,
        key="approx_top5_correct_bits",
        row_count=row_count,
        sample_count=len(raw_top5_clean),
    )
    _require_float_matrix(
        payload,
        key="delta_margin_top1",
        row_count=row_count,
        sample_count=len(raw_top1_clean),
    )
    _require_float_matrix(
        payload,
        key="delta_margin_top5",
        row_count=row_count,
        sample_count=len(raw_top5_clean),
    )

    true_top1_drop = _recompute_drop(true_top1_clean, true_top1)
    true_top5_drop = _recompute_drop(true_top5_clean, true_top5)
    raw_top1_drop = _recompute_drop(raw_top1_clean, raw_top1)
    raw_top5_drop = _recompute_drop(raw_top5_clean, raw_top5)
    declared = {
        "true1000_top1_drop": true_top1_drop,
        "true1000_top5_drop": true_top5_drop,
        "raw500_top1_drop": raw_top1_drop,
        "raw500_top5_drop": raw_top5_drop,
    }
    for key, recomputed in declared.items():
        stored = _require_float_vector(payload, key=key, row_count=row_count)
        if not np.allclose(stored, recomputed, atol=1.0e-7, rtol=0.0):
            raise ValueError(f"{key} does not match correctness-bit recomputation")
    return np.column_stack([true_top1_drop, true_top5_drop]).astype(np.float32, copy=False)


@dataclass(frozen=True)
class DatasetBlock:
    block_id: int
    start: int
    stop: int
    physical_gpu_id: int


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not ids:
        raise ValueError("--gpu-ids must contain at least one CUDA GPU ID")
    if any(item < 0 for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("--gpu-ids must contain unique non-negative CUDA GPU IDs")
    return ids


def _build_blocks(*, row_count: int, gpu_ids: Sequence[int], block_size: int) -> tuple[DatasetBlock, ...]:
    if int(row_count) <= 0:
        raise ValueError("selected population must contain at least one row")
    if int(block_size) <= 0:
        raise ValueError("--block-size must be positive")
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    return tuple(
        DatasetBlock(
            block_id=start // int(block_size),
            start=start,
            stop=min(start + int(block_size), int(row_count)),
            physical_gpu_id=int(gpu_ids[(start // int(block_size)) % len(gpu_ids)]),
        )
        for start in range(0, int(row_count), int(block_size))
    )


def _source_file_contract(*, source_data_dir: Path, s8_asset_root: Path) -> dict[str, Any]:
    source = Path(source_data_dir).resolve()
    asset = Path(s8_asset_root).resolve()
    required_source = {
        "historical_configuration": source / "configuration_index.csv",
    }
    for path in [*required_source.values(), asset / "manifest.json", asset / "population.csv"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads((asset / "manifest.json").read_text(encoding="utf-8"))
    table_filename = str(manifest.get("table_filename", ""))
    population_filename = str(manifest.get("population_filename", ""))
    if not table_filename or not (asset / table_filename).is_file():
        raise FileNotFoundError(asset / table_filename)
    if not population_filename or not (asset / population_filename).is_file():
        raise FileNotFoundError(asset / population_filename)
    files = manifest.get("files")
    if not isinstance(files, dict) or table_filename not in files or population_filename not in files:
        raise ValueError("S8 asset manifest does not provide verified table/population file hashes")
    return {
        "source_data_dir": str(source),
        "s8_asset_root": str(asset),
        "historical_configuration_sha1": _sha1_file(required_source["historical_configuration"]),
        "s8_asset_manifest_sha1": _sha1_file(asset / "manifest.json"),
        "s8_population_filename": population_filename,
        "s8_population_sha1": str(files[population_filename]),
        "s8_table_filename": table_filename,
        "s8_table_sha1": str(files[table_filename]),
        "s8_lut_domain": str(manifest.get("target_lut_domain", "")),
        "s8_table_shape": list(manifest.get("table_shape", ())),
        "s8_asset_row_count": int(manifest.get("row_count", -1)),
    }


def _load_selected_s8_population(*, s8_asset_root: Path, max_designs: int | None):
    """Read the small S8 metadata and select the immutable leading smoke/full slice."""

    import pandas as pd

    asset = Path(s8_asset_root).resolve()
    manifest = json.loads((asset / "manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("target_lut_domain")) != "s8s8_sign_magnitude":
        raise ValueError("strict50 dataset builder requires the S8S8 sign-magnitude asset")
    population = pd.read_csv(asset / str(manifest["population_filename"]), low_memory=False)
    expected_count = int(manifest["row_count"])
    required = {
        "row_index",
        "design_id",
        "runtime_design_id",
        "config_hash",
        "approx_high_part",
        "low_tokens",
        "high_tokens",
        "replacement_kind",
        "source_row_index",
    }
    missing = sorted(required.difference(population.columns))
    if missing:
        raise KeyError(f"S8 population is missing {missing}")
    if len(population) != expected_count:
        raise ValueError("S8 population row count does not match asset manifest")
    row_index = population["row_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(row_index, np.arange(len(population), dtype=np.int64)):
        raise ValueError("S8 population row_index must be contiguous and ordered")
    for column in ("design_id", "runtime_design_id", "config_hash"):
        if population[column].astype(str).duplicated().any():
            raise ValueError(f"S8 population {column} values must be unique")
    if max_designs is not None:
        if int(max_designs) <= 0:
            raise ValueError("--max-designs must be positive")
        population = population.iloc[: int(max_designs)].copy()
    if population.empty:
        raise ValueError("selected S8 population is empty")
    selected = population.reset_index(drop=True)
    if not np.array_equal(selected["row_index"].to_numpy(dtype=np.int64), np.arange(len(selected), dtype=np.int64)):
        raise ValueError("strict50 output requires a leading contiguous S8 population prefix")
    return selected, manifest


def _enrich_configuration(*, selected_population, source_data_dir: Path, label_protocol_hash: str):
    """Retain historical split membership while replacing identity and label provenance."""

    import pandas as pd

    source = pd.read_csv(Path(source_data_dir).resolve() / "configuration_index.csv", low_memory=False)
    required = {"row_index", "split", "evaluation_role", "legacy_split"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"historical configuration is missing {missing}")
    source = source.set_index("row_index", verify_integrity=True)
    out = selected_population.copy()
    source_rows = out["source_row_index"].to_numpy(dtype=np.int64)
    if np.any(source_rows < 0) or np.any(source_rows >= len(source)):
        raise ValueError("target source_row_index falls outside the historical configuration")
    historical = source.loc[source_rows]
    out["split"] = historical["split"].astype(str).to_numpy()
    out["evaluation_role"] = historical["evaluation_role"].astype(str).to_numpy()
    out["legacy_split"] = historical["legacy_split"].astype(str).to_numpy()
    out["source_split"] = out["split"].astype(str)
    out["source_evaluation_role"] = out["evaluation_role"].astype(str)
    out["label_source"] = "s8s8_dynamic_canonical_true1000"
    out["label_fidelity"] = "image1000"
    out["label_sample_count"] = np.int64(1000)
    out["label_protocol_hash"] = str(label_protocol_hash)
    out["source_dataset_id"] = "s8s8_lowonly20k_sign_magnitude_v1"
    return out


def _build_static_features(
    *,
    selected_population,
    s8_asset_root: Path,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize signed-S8S8 3,103 static features for selected rows."""

    from aaae.accuracy_predictor.features.static import signed_features
    from aaae.accuracy_predictor.features.static.signed_features import (
        STATIC_FEATURE_SCHEMA,
        build_s8s8_signed_common3103,
    )

    builder = build_s8s8_signed_common3103(
        target_population=selected_population,
        asset_root=Path(s8_asset_root).resolve(),
        chunk_size=int(chunk_size),
    )
    common = np.asarray(builder.common3103, dtype=np.float32)
    if common.shape != (len(selected_population), 3103) or not np.isfinite(common).all():
        raise ValueError(f"static feature builder returned invalid common3103 {common.shape}")
    if not np.array_equal(common[:, 3089], np.zeros(len(common), dtype=np.float32)):
        raise ValueError("S8S8 static features did not set approx_high_part descriptor to zero")
    if not np.array_equal(common[:, 3090], np.zeros(len(common), dtype=np.float32)):
        raise ValueError("S8S8 static features did not set design_index descriptor to zero")
    asset_manifest = builder.asset_manifest
    metadata = {
        "schema": STATIC_FEATURE_SCHEMA,
        "runtime_domain": "s8s8_sign_magnitude_int8_qdq",
        "static_lut_domain": str(asset_manifest["target_lut_domain"]),
        "static_lut_index_encoding": str(asset_manifest["index_encoding"]),
        "static_exact_reference": "signed_qdq_code_product_for_codes_minus128_to_127",
        "static_metric_nmed_denominator": 16384,
        "static_feature_chunk_size": int(builder.chunk_size),
        "static_table_filename": str(asset_manifest["table_filename"]),
        "static_table_sha1": str(asset_manifest["files"][str(asset_manifest["table_filename"])]),
        "static_feature_builder_sha1": _sha1_file(
            Path(signed_features.__file__)
        ),
        "replacement_kind_counts": {
            str(key): int(value) for key, value in builder.validation.replacement_kind_counts.items()
        },
    }
    return common, metadata


def _prepare_request(
    args: argparse.Namespace,
    *,
    source_contract: Mapping[str, Any],
    profile,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_key": str(args.model_key),
        "preprocess_key": str(args.preprocess_key),
        "model_path": str(Path(args.model_path).resolve()),
        "model_sha1": _sha1_file(Path(args.model_path)),
        "node_registry_path": str(Path(args.node_registry).resolve()),
        "node_registry_sha1": _sha1_file(Path(args.node_registry)),
        "injection_mode": profile.injection_mode_contract(),
        "gpu_ids": list(_parse_gpu_ids(args.gpu_ids)),
        "baseline_physical_gpu_id": int(args.baseline_gpu_id),
        "batch_size": int(args.batch_size),
        "block_size": int(args.block_size),
        "input_residency": str(args.input_residency),
        "runner_factory": str(args.runner_factory),
        "s8_residual_abs_bound": int(args.s8_residual_abs_bound),
        "max_designs": None if args.max_designs is None else int(args.max_designs),
        "true1000_subset_path": str(Path(args.true1000_subset).resolve()),
        "raw500_subset_path": str(Path(args.raw500_subset).resolve()),
        "true1000_cache_path": str(Path(args.true1000_cache).resolve()),
        "raw500_cache_path": str(Path(args.raw500_cache).resolve()),
        "source_contract": dict(source_contract),
    }


def _can_resume_legacy_swin(
    existing: Mapping[str, Any],
    current_request: Mapping[str, Any],
    *,
    model_key: str,
) -> bool:
    """Legacy 72-node runs are never resumable as linear48 runs."""

    del existing, current_request, model_key
    return False


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the selected S8 population, static features, and image union."""

    if bool(args.resume) and bool(args.overwrite):
        raise ValueError("--resume and --overwrite are mutually exclusive")
    profile = _resolve_profile_arguments(args)
    profile.validate_local_assets()
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if args.baseline_gpu_id is None:
        args.baseline_gpu_id = int(gpu_ids[0])
    if int(args.baseline_gpu_id) not in gpu_ids:
        raise ValueError("--baseline-gpu-id must be included in --gpu-ids")
    bulk = _bulk_module()
    if int(args.batch_size) <= 1:
        raise ValueError("strict50 dynamic dataset generation requires --batch-size > 1")
    if int(args.s8_residual_abs_bound) != DEFAULT_S8_RESIDUAL_ABS_BOUND:
        raise ValueError(
            f"this S8S8 asset is validated for --s8-residual-abs-bound={DEFAULT_S8_RESIDUAL_ABS_BOUND}"
        )
    bulk._validate_model_batch_contract(
        model_path=Path(args.model_path).resolve(),
        batch_size=int(args.batch_size),
        execution_profile="throughput",
    )
    bulk._load_node_names(
        Path(args.node_registry).resolve(),
        Path(args.model_path).resolve(),
        model_key=profile.model_key,
    )
    source_contract = _source_file_contract(
        source_data_dir=Path(args.source_data_dir),
        s8_asset_root=Path(args.s8_asset_root),
    )
    request = _prepare_request(args, source_contract=source_contract, profile=profile)
    paths = _paths(Path(args.output_root))
    selected_population, asset_manifest = _load_selected_s8_population(
        s8_asset_root=Path(args.s8_asset_root),
        max_designs=args.max_designs,
    )
    gpu_ids = tuple(request["gpu_ids"])
    blocks = _build_blocks(
        row_count=len(selected_population),
        gpu_ids=gpu_ids,
        block_size=int(args.block_size),
    )
    if bool(args.dry_run):
        return {
            "status": "dry_run",
            "output_root": str(paths["root"]),
            "request": request,
            "selected_row_count": int(len(selected_population)),
            "block_count": int(len(blocks)),
            "blocks_by_gpu": {
                str(gpu_id): int(sum(block.stop - block.start for block in blocks if block.physical_gpu_id == gpu_id))
                for gpu_id in gpu_ids
            },
        }

    _validate_output_root_is_separate(
        paths["root"],
        (
            Path(args.source_data_dir),
            Path(args.s8_asset_root),
            Path(args.model_path),
            Path(args.node_registry),
            Path(args.image_root),
            Path(args.true1000_subset),
            Path(args.raw500_subset),
            Path(args.true1000_cache),
            Path(args.raw500_cache),
        ),
    )
    if paths["root"].exists() and bool(args.overwrite):
        _safe_remove_output_root(paths["root"])
    if paths["run_config"].is_file():
        if not bool(args.resume):
            raise FileExistsError(f"{paths['root']} already exists; pass --resume or --overwrite")
        _existing_paths, existing, existing_contract = _load_run_config(paths["root"])
        if existing.get("schema_version") != SCHEMA_VERSION or existing.get("request") != request:
            raise ValueError("existing output root does not match the requested immutable dataset contract")
        if any(existing_contract.get(key) != value for key, value in request.items()):
            raise ValueError("existing request is inconsistent with its immutable dataset contract")
        return existing
    if paths["root"].exists():
        raise FileExistsError(f"{paths['root']} exists but is not a resumable strict50 dataset run")

    paths["root"].mkdir(parents=True, exist_ok=False)
    try:
        _write_run_marker(paths)
        ensure_frozen_input_cache(
            profile=profile,
            role="true1000",
            sample_count=TRUE_PROTOCOL_COUNT,
            sample_seed=TRUE_PROTOCOL_SEED,
            subset_path=Path(args.true1000_subset),
            cache_path=Path(args.true1000_cache),
            image_root=Path(args.image_root),
            rebuild=bool(args.rebuild_input_caches),
        )
        ensure_frozen_input_cache(
            profile=profile,
            role="raw500",
            sample_count=RAW_PROTOCOL_COUNT,
            sample_seed=RAW_PROTOCOL_SEED,
            subset_path=Path(args.raw500_subset),
            cache_path=Path(args.raw500_cache),
            image_root=Path(args.image_root),
            rebuild=bool(args.rebuild_input_caches),
        )
        true1000, raw500, frozen_provenance = load_frozen_image_protocols(
            true1000_subset_path=Path(args.true1000_subset),
            raw500_subset_path=Path(args.raw500_subset),
            true1000_cache_path=Path(args.true1000_cache),
            raw500_cache_path=Path(args.raw500_cache),
            model_key=profile.model_key,
            preprocess_key=profile.preprocess_key,
        )
        frozen_union = _materialize_frozen_union(
            paths=paths,
            true1000=true1000,
            raw500=raw500,
            provenance=frozen_provenance,
        )
        input_contract_sha1 = _sha1_file(paths["input_contract"])
        common, static_metadata = _build_static_features(
            selected_population=selected_population,
            s8_asset_root=Path(args.s8_asset_root),
            chunk_size=int(args.static_feature_batch_size),
        )
        label_protocol_hash = _sha1_json(
            {
                "protocol": "canonical_dynamic_int8_true1000",
                "model_sha1": request["model_sha1"],
                "input_contract_sha1": input_contract_sha1,
                "batch_size": int(args.batch_size),
                "session_profile": "throughput",
                "injection_scope_id": profile.injection_scope_id,
                "node_registry_sha1": request["node_registry_sha1"],
            }
        )
        configuration = _enrich_configuration(
            selected_population=selected_population,
            source_data_dir=Path(args.source_data_dir),
            label_protocol_hash=label_protocol_hash,
        )
        split_counts = (
            validate_full_population_splits(configuration)
            if args.max_designs is None
            else {
                str(name): int(count)
                for name, count in configuration["split"].astype(str).value_counts().items()
            }
        )
        paths["population"].mkdir(parents=True, exist_ok=True)
        paths["data"].mkdir(parents=True, exist_ok=True)
        selected_population.to_csv(paths["selected_rows"], index=False)
        _atomic_save_npy(paths["common3103"], common)
        configuration.to_csv(paths["configuration_index"], index=False)
        configuration[["row_index", "design_id", "runtime_design_id"]].to_csv(paths["runtime_map"], index=False)
        contract = {
            **request,
            "asset_manifest": {
                "schema_version": str(asset_manifest.get("schema_version", "")),
                "asset_id": str(asset_manifest.get("asset_id", "")),
                "target_semantics": str(asset_manifest.get("target_semantics", "")),
            },
            "frozen_input_contract_sha1": input_contract_sha1,
            "frozen_union_count": int(len(frozen_union.image_names)),
            "true1000_count": int(len(true1000.image_names)),
            "raw500_count": int(len(raw500.image_names)),
            "selected_population_sha1": _sha1_file(paths["selected_rows"]),
            "common3103_sha1": _sha1_file(paths["common3103"]),
            "configuration_index_sha1": _sha1_file(paths["configuration_index"]),
            "runtime_map_sha1": _sha1_file(paths["runtime_map"]),
            "split_counts": split_counts,
            "expected_full_split_counts": dict(EXPECTED_FULL_SPLIT_COUNTS),
            "static_features": static_metadata,
            "label_protocol_hash": label_protocol_hash,
            "canonical_reference_policy": {
                "kind": "single_global_dynamic_b16_clean_reference",
                "baseline_physical_gpu_id": int(args.baseline_gpu_id),
                "cross_device_canonical_reference": True,
                "allowed_dynamic_batch_numeric_variation": True,
                "reason": "strict50 bank_meta has one global margin reference; approximate workers may run on other GPUs",
            },
        }
        run_fingerprint = _sha1_json(contract)
        run_config = {
            "schema_version": SCHEMA_VERSION,
            "created_at_unix": time.time(),
            "request": request,
            "contract": contract,
            "run_fingerprint": run_fingerprint,
            "plan": {
                "selected_row_count": int(len(selected_population)),
                "block_size": int(args.block_size),
                "block_count": int(len(blocks)),
                "gpu_ids": list(gpu_ids),
                "blocks": [
                    {
                        "block_id": int(block.block_id),
                        "start": int(block.start),
                        "stop": int(block.stop),
                        "physical_gpu_id": int(block.physical_gpu_id),
                    }
                    for block in blocks
                ],
            },
        }
        _atomic_write_json(paths["run_config"], run_config)
        return run_config
    except Exception:
        # A partial output root must never look resumable or trainable.
        try:
            _safe_remove_output_root(paths["root"])
        except (OSError, ValueError):
            pass
        raise


def _load_run_config(output_root: Path) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    paths = _paths(output_root)
    if not paths["run_config"].is_file():
        raise FileNotFoundError(paths["run_config"])
    run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
    contract = run_config.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("run_config has no immutable contract")
    fingerprint = _sha1_json(contract)
    if run_config.get("run_fingerprint") != fingerprint:
        raise ValueError("run_config fingerprint does not match its immutable contract")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("run_config belongs to an incompatible dataset schema")
    return paths, run_config, contract


def _load_frozen_run_inputs(
    *,
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = ("union_inputs", "union_labels", "union_positions", "input_contract")
    missing = [str(paths[name]) for name in required if not Path(paths[name]).is_file()]
    if missing:
        raise FileNotFoundError(f"prepared frozen image inputs are missing: {missing}")
    if _sha1_file(paths["input_contract"]) != str(contract["frozen_input_contract_sha1"]):
        raise ValueError("frozen image input contract changed after run preparation")
    input_contract = json.loads(Path(paths["input_contract"]).read_text(encoding="utf-8"))
    if input_contract.get("artifact_sha1") != _frozen_union_artifact_sha1(paths):
        raise ValueError("frozen union artifact digests do not match the immutable input contract")
    inputs = np.load(paths["union_inputs"], mmap_mode="r")
    labels = np.load(paths["union_labels"], allow_pickle=False).astype(np.int64, copy=False)
    positions = np.load(paths["union_positions"], allow_pickle=False)
    true_positions = np.asarray(positions["true1000_positions"], dtype=np.int64)
    raw_positions = np.asarray(positions["raw500_positions"], dtype=np.int64)
    if inputs.dtype != np.float32 or inputs.ndim != 4 or inputs.shape[0] != len(labels):
        raise ValueError("frozen union input array does not satisfy its shape/dtype contract")
    if inputs.shape[0] != int(contract["frozen_union_count"]):
        raise ValueError("frozen union count does not match run contract")
    if true_positions.shape != (int(contract["true1000_count"]),):
        raise ValueError("true1000 positions do not match run contract")
    if raw_positions.shape != (int(contract["raw500_count"]),):
        raise ValueError("raw500 positions do not match run contract")
    if np.any(true_positions < 0) or np.any(true_positions >= len(labels)):
        raise ValueError("true1000 positions fall outside the frozen union")
    if np.any(raw_positions < 0) or np.any(raw_positions >= len(labels)):
        raise ValueError("raw500 positions fall outside the frozen union")
    return inputs, labels, true_positions, raw_positions


def _write_bank_meta(path: Path, clean: Mapping[str, np.ndarray], *, basis_rank: int) -> None:
    sample_count = int(np.asarray(clean["true_label"]).shape[0])
    rank = int(basis_rank)
    if rank < 0:
        raise ValueError("basis rank must be non-negative")
    _atomic_save_npz(
        path,
        clean_margin_top1=np.asarray(clean["clean_margin_top1"], dtype=np.float32),
        clean_margin_top5=np.asarray(clean["clean_margin_top5"], dtype=np.float32),
        exact_top1_correct_bits=np.asarray(clean["exact_top1_correct_bits"], dtype=np.uint8),
        exact_top5_correct_bits=np.asarray(clean["exact_top5_correct_bits"], dtype=np.uint8),
        clean_entropy=np.asarray(clean["clean_entropy"], dtype=np.float32),
        exact_pred_class=np.asarray(clean["exact_pred_class"], dtype=np.int64),
        true_label=np.asarray(clean["true_label"], dtype=np.int64),
        bank_mean_top1=np.zeros((sample_count,), dtype=np.float32),
        bank_mean_top5=np.zeros((sample_count,), dtype=np.float32),
        bank_basis_top1=np.zeros((rank, sample_count), dtype=np.float32),
        bank_basis_top5=np.zeros((rank, sample_count), dtype=np.float32),
        coeff_mean_top1=np.zeros((rank,), dtype=np.float32),
        coeff_std_top1=np.ones((rank,), dtype=np.float32),
        coeff_mean_top5=np.zeros((rank,), dtype=np.float32),
        coeff_std_top5=np.ones((rank,), dtype=np.float32),
        basis_ready=np.asarray(0, dtype=np.int64),
    )


def _load_clean_meta(path: Path) -> dict[str, np.ndarray]:
    required = (
        "clean_margin_top1",
        "clean_margin_top5",
        "exact_top1_correct_bits",
        "exact_top5_correct_bits",
        "clean_entropy",
        "exact_pred_class",
        "true_label",
    )
    with np.load(path, allow_pickle=False) as payload:
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise KeyError(f"{path} is missing clean-bank fields {missing}")
        return {
            "clean_margin_top1": np.asarray(payload["clean_margin_top1"], dtype=np.float32),
            "clean_margin_top5": np.asarray(payload["clean_margin_top5"], dtype=np.float32),
            "exact_top1_correct_bits": np.asarray(payload["exact_top1_correct_bits"], dtype=np.uint8),
            "exact_top5_correct_bits": np.asarray(payload["exact_top5_correct_bits"], dtype=np.uint8),
            "clean_entropy": np.asarray(payload["clean_entropy"], dtype=np.float32),
            "exact_pred_class": np.asarray(payload["exact_pred_class"], dtype=np.int64),
            "true_label": np.asarray(payload["true_label"], dtype=np.int64),
        }


def _clean_meta_matches_logits(
    meta: Mapping[str, np.ndarray],
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    positions: np.ndarray,
) -> bool:
    expected = compute_clean_bank(logits[positions], labels[positions])
    return all(
        name in meta and np.array_equal(np.asarray(meta[name]), np.asarray(expected[name]))
        for name in (
            "clean_margin_top1",
            "clean_margin_top5",
            "exact_top1_correct_bits",
            "exact_top5_correct_bits",
            "clean_entropy",
            "exact_pred_class",
            "true_label",
        )
    )


def _baseline_contract(
    *,
    contract: Mapping[str, Any],
    physical_gpu_id: int,
    runtime_device_id: int,
) -> dict[str, Any]:
    return {
        "schema_version": str(contract["schema_version"]),
        "run_fingerprint": _sha1_json(contract),
        "model_path": str(contract["model_path"]),
        "model_sha1": str(contract["model_sha1"]),
        "frozen_input_contract_sha1": str(contract["frozen_input_contract_sha1"]),
        "sample_count": int(contract["frozen_union_count"]),
        "true1000_count": int(contract["true1000_count"]),
        "raw500_count": int(contract["raw500_count"]),
        "batch_size": int(contract["batch_size"]),
        "cuda_session_profile": "throughput",
        "physical_gpu_id": int(physical_gpu_id),
        "runtime_device_id": int(runtime_device_id),
        "reference_policy": "one canonical global clean baseline for all strict50 workers",
    }


def build_canonical_exact_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Create the single B=16 clean reference used by every worker and raw500 bank."""

    paths, _run_config, contract = _load_run_config(Path(args.output_root))
    physical_gpu_id = int(args.physical_gpu_id)
    runtime_device_id = int(args.runtime_device_id)
    if physical_gpu_id != int(contract["baseline_physical_gpu_id"]):
        raise ValueError("baseline physical GPU does not match the immutable run contract")
    if runtime_device_id < 0:
        raise ValueError("runtime device ID must be non-negative")
    expected_contract = _baseline_contract(
        contract=contract,
        physical_gpu_id=physical_gpu_id,
        runtime_device_id=runtime_device_id,
    )
    existing_files = CANONICAL_BASELINE_ARTIFACT_KEYS
    if all(Path(paths[name]).is_file() for name in existing_files):
        actual_contract = json.loads(paths["canonical_exact_contract"].read_text(encoding="utf-8"))
        bank_meta_contract = json.loads(paths["strict50_bank_meta_contract"].read_text(encoding="utf-8"))
        with np.load(paths["canonical_exact"], allow_pickle=False) as payload:
            logits = np.asarray(payload["logits"], dtype=np.float32)
            labels = np.asarray(payload["union_labels"], dtype=np.int64)
            cached_true_positions = np.asarray(payload["true1000_positions"], dtype=np.int64)
            cached_raw_positions = np.asarray(payload["raw500_positions"], dtype=np.int64)
        cached_true_clean = _load_clean_meta(paths["true1000_meta"])
        _inputs, expected_labels, expected_true_positions, expected_raw_positions = _load_frozen_run_inputs(
            paths=paths,
            contract=contract,
        )
        if (
            all(actual_contract.get(key) == value for key, value in expected_contract.items())
            and logits.shape == (int(contract["frozen_union_count"]), 1000)
            and labels.shape == (int(contract["frozen_union_count"]),)
            and actual_contract.get("exact_logits_sha1") == hashlib.sha1(logits.tobytes()).hexdigest()
            and actual_contract.get("union_labels_sha1") == hashlib.sha1(labels.tobytes()).hexdigest()
            and np.array_equal(labels, expected_labels)
            and np.array_equal(cached_true_positions, expected_true_positions)
            and np.array_equal(cached_raw_positions, expected_raw_positions)
            and bank_meta_contract.get("run_fingerprint") == _sha1_json(contract)
            and bank_meta_contract.get("bank_meta_sha1") == _sha1_file(paths["strict50_bank_meta"])
            and (
                bank_meta_contract.get("true1000_meta_sha1") == _sha1_file(paths["true1000_meta"])
                or (
                    "true1000_meta_sha1" not in bank_meta_contract
                    and _clean_meta_matches_logits(
                        cached_true_clean,
                        logits=logits,
                        labels=labels,
                        positions=cached_true_positions,
                    )
                )
            )
            and bank_meta_contract.get("canonical_exact_contract_sha1") == _sha1_file(paths["canonical_exact_contract"])
        ):
            return {"status": "reused", **actual_contract}
        raise RuntimeError("existing canonical exact baseline is incompatible with the prepared run")
    if any(Path(paths[name]).exists() for name in existing_files):
        _clear_incomplete_baseline(paths)

    inputs, labels, true_positions, raw_positions = _load_frozen_run_inputs(paths=paths, contract=contract)
    from aaae.inference_runtime.transformer.ort_session import create_cuda_session

    session, provider = create_cuda_session(
        Path(contract["model_path"]),
        device_id=runtime_device_id,
        fallback_parallelism=1,
        session_profile="throughput",
    )
    try:
        input_name = str(session.get_inputs()[0].name)
        logits = np.empty((len(inputs), 1000), dtype=np.float32)
        started = time.perf_counter()
        for start in range(0, len(inputs), int(contract["batch_size"])):
            stop = min(start + int(contract["batch_size"]), len(inputs))
            output = np.asarray(
                session.run(None, {input_name: np.asarray(inputs[start:stop], dtype=np.float32)})[0],
                dtype=np.float32,
            )
            if output.shape != (stop - start, 1000):
                raise ValueError(f"canonical exact model returned {output.shape}, expected {(stop - start, 1000)}")
            logits[start:stop] = output
        elapsed_seconds = float(time.perf_counter() - started)
    finally:
        session.close()
    true_clean = compute_clean_bank(logits[true_positions], labels[true_positions])
    raw_clean = compute_clean_bank(logits[raw_positions], labels[raw_positions])
    if len(true_clean["true_label"]) != int(contract["true1000_count"]):
        raise ValueError("canonical true1000 clean bank has incorrect length")
    if len(raw_clean["true_label"]) != int(contract["raw500_count"]):
        raise ValueError("canonical raw500 clean bank has incorrect length")
    _atomic_save_npz(
        paths["canonical_exact"],
        logits=logits,
        union_labels=labels,
        true1000_positions=true_positions,
        raw500_positions=raw_positions,
    )
    _write_bank_meta(paths["strict50_bank_meta"], raw_clean, basis_rank=DEFAULT_BANK_BASIS_RANK)
    _atomic_save_npz(
        paths["true1000_meta"],
        clean_margin_top1=np.asarray(true_clean["clean_margin_top1"], dtype=np.float32),
        clean_margin_top5=np.asarray(true_clean["clean_margin_top5"], dtype=np.float32),
        exact_top1_correct_bits=np.asarray(true_clean["exact_top1_correct_bits"], dtype=np.uint8),
        exact_top5_correct_bits=np.asarray(true_clean["exact_top5_correct_bits"], dtype=np.uint8),
        clean_entropy=np.asarray(true_clean["clean_entropy"], dtype=np.float32),
        exact_pred_class=np.asarray(true_clean["exact_pred_class"], dtype=np.int64),
        true_label=np.asarray(true_clean["true_label"], dtype=np.int64),
    )
    exact_contract = {
        **expected_contract,
        "provider": str(provider),
        "elapsed_seconds": elapsed_seconds,
        "exact_logits_sha1": hashlib.sha1(logits.tobytes()).hexdigest(),
        "union_labels_sha1": hashlib.sha1(labels.tobytes()).hexdigest(),
        "exact_top1_accuracy_true1000": float(
            np.mean(np.asarray(true_clean["exact_top1_correct_bits"], dtype=np.float32), dtype=np.float64)
        ),
        "exact_top5_accuracy_true1000": float(
            np.mean(np.asarray(true_clean["exact_top5_correct_bits"], dtype=np.float32), dtype=np.float64)
        ),
        "exact_top1_accuracy_raw500": float(
            np.mean(np.asarray(raw_clean["exact_top1_correct_bits"], dtype=np.float32), dtype=np.float64)
        ),
        "exact_top5_accuracy_raw500": float(
            np.mean(np.asarray(raw_clean["exact_top5_correct_bits"], dtype=np.float32), dtype=np.float64)
        ),
    }
    _atomic_write_json(paths["canonical_exact_contract"], exact_contract)
    _atomic_write_json(
        paths["strict50_bank_meta_contract"],
        {
            "schema_version": str(contract["schema_version"]),
            "run_fingerprint": _sha1_json(contract),
            "sample_count": int(contract["raw500_count"]),
            "bank_meta_sha1": _sha1_file(paths["strict50_bank_meta"]),
            "true1000_meta_sha1": _sha1_file(paths["true1000_meta"]),
            "canonical_exact_contract_sha1": _sha1_file(paths["canonical_exact_contract"]),
            "canonical_exact_logits_sha1": exact_contract["exact_logits_sha1"],
            "basis_ready": False,
            "basis_rank": DEFAULT_BANK_BASIS_RANK,
        },
    )
    return {"status": "completed", **exact_contract}


def _blocks_from_run_config(run_config: Mapping[str, Any]) -> tuple[DatasetBlock, ...]:
    plan = run_config.get("plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("blocks"), list):
        raise ValueError("run_config has no valid block plan")
    blocks = tuple(
        DatasetBlock(
            block_id=int(value["block_id"]),
            start=int(value["start"]),
            stop=int(value["stop"]),
            physical_gpu_id=int(value["physical_gpu_id"]),
        )
        for value in plan["blocks"]
    )
    if not blocks:
        raise ValueError("run_config block plan is empty")
    expected_ids = tuple(range(len(blocks)))
    if tuple(block.block_id for block in blocks) != expected_ids:
        raise ValueError("run_config block IDs must be contiguous and ordered")
    return blocks


def _block_paths(paths: Mapping[str, Path], block_id: int) -> tuple[Path, Path]:
    stem = f"block_{int(block_id):05d}"
    return Path(paths["blocks"]) / f"{stem}.runtime.npz", Path(paths["blocks"]) / f"{stem}.manifest.json"


def _clear_incomplete_block(paths: Mapping[str, Path], *, block_id: int) -> None:
    artifacts = _block_paths(paths, block_id)
    present = tuple(path.exists() or path.is_symlink() for path in artifacts)
    if all(present):
        raise RuntimeError(f"refusing to clear block {block_id} because both artifacts exist")
    invalid = [
        str(path)
        for path in artifacts
        if (path.exists() or path.is_symlink()) and (path.is_symlink() or not path.is_file())
    ]
    if invalid:
        raise RuntimeError(f"block {block_id} artifact paths are not regular files: {invalid}")
    for path in artifacts:
        path.unlink(missing_ok=True)


def _clear_incomplete_baseline(paths: Mapping[str, Path]) -> None:
    artifacts = tuple(Path(paths[name]) for name in CANONICAL_BASELINE_ARTIFACT_KEYS)
    present = tuple(path.exists() or path.is_symlink() for path in artifacts)
    invalid = [
        str(path)
        for path in artifacts
        if (path.exists() or path.is_symlink()) and (path.is_symlink() or not path.is_file())
    ]
    if invalid:
        raise RuntimeError(f"canonical baseline artifact paths are not regular files: {invalid}")
    if all(present):
        raise RuntimeError("refusing to clear a complete canonical baseline")
    for path in artifacts:
        path.unlink(missing_ok=True)


def _row_identity_sha1(rows) -> str:
    identities = [
        {
            "row_index": int(row.row_index),
            "design_id": str(row.design_id),
            "runtime_design_id": str(row.runtime_design_id),
        }
        for row in rows.itertuples(index=False)
    ]
    return _sha1_json(identities)


def _load_canonical_clean_banks(
    *,
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, str]:
    required = (
        "canonical_exact",
        "canonical_exact_contract",
        "strict50_bank_meta",
        "true1000_meta",
    )
    missing = [str(paths[name]) for name in required if not Path(paths[name]).is_file()]
    if missing:
        raise FileNotFoundError(f"canonical exact baseline is missing: {missing}")
    exact_contract = json.loads(paths["canonical_exact_contract"].read_text(encoding="utf-8"))
    bank_meta_contract = (
        json.loads(paths["strict50_bank_meta_contract"].read_text(encoding="utf-8"))
        if paths["strict50_bank_meta_contract"].is_file()
        else {}
    )
    if exact_contract.get("run_fingerprint") != _sha1_json(contract):
        raise ValueError("canonical exact baseline belongs to a different immutable run")
    if bank_meta_contract and (
        bank_meta_contract.get("run_fingerprint") != _sha1_json(contract)
        or bank_meta_contract.get("bank_meta_sha1") != _sha1_file(paths["strict50_bank_meta"])
        or (
            bank_meta_contract.get("true1000_meta_sha1") is not None
            and bank_meta_contract.get("true1000_meta_sha1") != _sha1_file(paths["true1000_meta"])
        )
        or bank_meta_contract.get("canonical_exact_contract_sha1")
        != _sha1_file(paths["canonical_exact_contract"])
    ):
        raise ValueError("canonical clean metadata hashes do not match their contract")
    with np.load(paths["canonical_exact"], allow_pickle=False) as payload:
        logits = np.asarray(payload["logits"], dtype=np.float32)
        labels = np.asarray(payload["union_labels"], dtype=np.int64)
        true_positions = np.asarray(payload["true1000_positions"], dtype=np.int64)
        raw_positions = np.asarray(payload["raw500_positions"], dtype=np.int64)
    if logits.shape != (int(contract["frozen_union_count"]), 1000):
        raise ValueError("canonical exact logits have an incompatible shape")
    if exact_contract.get("exact_logits_sha1") != hashlib.sha1(logits.tobytes()).hexdigest():
        raise ValueError("canonical exact logits hash does not match its contract")
    _inputs, expected_labels, expected_true_positions, expected_raw_positions = _load_frozen_run_inputs(
        paths=paths,
        contract=contract,
    )
    if (
        not np.array_equal(labels, expected_labels)
        or not np.array_equal(true_positions, expected_true_positions)
        or not np.array_equal(raw_positions, expected_raw_positions)
    ):
        raise ValueError("canonical exact labels or subset positions do not match the frozen union")
    true_clean = _load_clean_meta(paths["true1000_meta"])
    raw_clean = _load_clean_meta(paths["strict50_bank_meta"])
    if bank_meta_contract.get("true1000_meta_sha1") is None and not _clean_meta_matches_logits(
        true_clean,
        logits=logits,
        labels=labels,
        positions=true_positions,
    ):
        raise ValueError("legacy true1000 clean metadata does not match canonical logits")
    if len(true_clean["true_label"]) != int(contract["true1000_count"]):
        raise ValueError("true1000 canonical clean metadata has an incompatible length")
    if len(raw_clean["true_label"]) != int(contract["raw500_count"]):
        raise ValueError("raw500 canonical clean metadata has an incompatible length")
    if not np.array_equal(true_clean["true_label"], labels[true_positions]):
        raise ValueError("true1000 clean metadata labels do not match the canonical exact baseline")
    if not np.array_equal(raw_clean["true_label"], labels[raw_positions]):
        raise ValueError("raw500 clean metadata labels do not match the canonical exact baseline")
    return (
        true_clean,
        raw_clean,
        _sha1_file(paths["canonical_exact_contract"]),
        _sha1_file(paths["strict50_bank_meta"]),
    )


def _is_complete_block(
    *,
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
    block: DatasetBlock,
    rows,
    true1000_clean: Mapping[str, np.ndarray],
    raw500_clean: Mapping[str, np.ndarray],
    canonical_exact_contract_sha1: str,
    bank_meta_sha1: str,
) -> bool:
    data_path, manifest_path = _block_paths(paths, block.block_id)
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != str(contract["schema_version"])
            or manifest.get("run_fingerprint") != _sha1_json(contract)
            or int(manifest.get("block_id", -1)) != int(block.block_id)
            or int(manifest.get("physical_gpu_id", -1)) != int(block.physical_gpu_id)
            or int(manifest.get("row_count", -1)) != int(len(rows))
            or manifest.get("row_identity_sha1") != _row_identity_sha1(rows)
            or manifest.get("canonical_exact_contract_sha1") != canonical_exact_contract_sha1
            or manifest.get("strict50_bank_meta_sha1") != bank_meta_sha1
            or manifest.get("data_sha1") != _sha1_file(data_path)
        ):
            return False
        with np.load(data_path, allow_pickle=False) as payload:
            validate_runtime_payload(
                payload,
                expected_row_index=rows["row_index"].to_numpy(dtype=np.int64),
                expected_public_design_id=rows["design_id"].astype(str).to_numpy(),
                expected_runtime_design_id=rows["runtime_design_id"].astype(str).to_numpy(),
                true1000_clean=true1000_clean,
                raw500_clean=raw500_clean,
            )
        return True
    except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _write_runtime_block(
    *,
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
    block: DatasetBlock,
    rows,
    payload: Mapping[str, np.ndarray],
    elapsed_seconds: float,
    runner_input_residency: str,
    identity: Mapping[str, Any],
    canonical_exact_contract_sha1: str,
    bank_meta_sha1: str,
) -> dict[str, Any]:
    data_path, manifest_path = _block_paths(paths, block.block_id)
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"block {block.block_id} already exists")
    _atomic_save_npz(data_path, **{name: np.asarray(value) for name, value in payload.items()})
    manifest = {
        "schema_version": str(contract["schema_version"]),
        "run_fingerprint": _sha1_json(contract),
        "block_id": int(block.block_id),
        "physical_gpu_id": int(block.physical_gpu_id),
        "runtime_device_id": 0,
        "row_count": int(len(rows)),
        "row_index_start": int(rows["row_index"].iloc[0]),
        "row_index_stop": int(rows["row_index"].iloc[-1]),
        "row_identity_sha1": _row_identity_sha1(rows),
        "canonical_exact_contract_sha1": canonical_exact_contract_sha1,
        "strict50_bank_meta_sha1": bank_meta_sha1,
        "data_file": data_path.name,
        "data_sha1": _sha1_file(data_path),
        "payload": {
            name: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
            for name, value in payload.items()
        },
        "elapsed_seconds": float(elapsed_seconds),
        "runner_input_residency": str(runner_input_residency),
        "identity": dict(identity),
        "completed_at_unix": time.time(),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _worker_progress_path(paths: Mapping[str, Path], physical_gpu_id: int) -> Path:
    return Path(paths["workers"]) / f"gpu_{int(physical_gpu_id):02d}.progress.json"


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate all blocks assigned to one masked physical GPU child process."""

    import pandas as pd

    paths, run_config, contract = _load_run_config(Path(args.output_root))
    physical_gpu_id = int(args.physical_gpu_id)
    runtime_device_id = int(args.runtime_device_id)
    if runtime_device_id < 0:
        raise ValueError("runtime device ID must be non-negative")
    if physical_gpu_id not in tuple(int(value) for value in contract["gpu_ids"]):
        raise ValueError("worker GPU is not part of the immutable run contract")
    rows = pd.read_csv(paths["selected_rows"], low_memory=False)
    if len(rows) != int(run_config["plan"]["selected_row_count"]):
        raise ValueError("selected population row count changed after preparation")
    if _sha1_file(paths["selected_rows"]) != str(contract["selected_population_sha1"]):
        raise ValueError("selected population changed after preparation")
    blocks = _blocks_from_run_config(run_config)
    true_clean, raw_clean, exact_contract_sha1, bank_meta_sha1 = _load_canonical_clean_banks(
        paths=paths,
        contract=contract,
    )
    pending = []
    for block in blocks:
        if block.physical_gpu_id != physical_gpu_id:
            continue
        block_rows = rows.iloc[block.start : block.stop].copy().reset_index(drop=True)
        data_path, manifest_path = _block_paths(paths, block.block_id)
        artifacts_present = (data_path.exists(), manifest_path.exists())
        if artifacts_present == (True, True):
            if not _is_complete_block(
                paths=paths,
                contract=contract,
                block=block,
                rows=block_rows,
                true1000_clean=true_clean,
                raw500_clean=raw_clean,
                canonical_exact_contract_sha1=exact_contract_sha1,
                bank_meta_sha1=bank_meta_sha1,
            ):
                raise RuntimeError(
                    f"block {block.block_id} has both checkpoint artifacts but failed validation; "
                    "preserving both files for diagnosis"
                )
            continue
        if artifacts_present != (False, False):
            _clear_incomplete_block(paths, block_id=block.block_id)
        pending.append((block, block_rows))
    if not pending:
        return {"status": "reused", "physical_gpu_id": physical_gpu_id, "block_count": 0}

    bulk = _bulk_module()
    asset = bulk.load_s8s8_sign_magnitude_truth_lut_asset(Path(contract["source_contract"]["s8_asset_root"]))
    if str(asset.manifest["files"][str(asset.manifest["table_filename"])]) != str(contract["source_contract"]["s8_table_sha1"]):
        raise ValueError("S8 LUT table hash changed after preparation")
    if _sha1_file(Path(contract["source_contract"]["s8_asset_root"]) / "manifest.json") != str(
        contract["source_contract"]["s8_asset_manifest_sha1"]
    ):
        raise ValueError("S8 asset manifest changed after preparation")
    inputs, _labels, true_positions, raw_positions = _load_frozen_run_inputs(paths=paths, contract=contract)
    node_names = bulk._load_node_names(
        Path(contract["node_registry_path"]),
        Path(contract["model_path"]),
        model_key=str(contract.get("model_key", DEFAULT_MODEL_KEY)),
    )
    runner = bulk._build_runner(
        factory_spec=str(contract["runner_factory"]),
        model_path=Path(contract["model_path"]),
        node_names=node_names,
        inputs=inputs,
        gpu_id=runtime_device_id,
        batch_size=int(contract["batch_size"]),
        s8_residual_abs_bound=int(contract["s8_residual_abs_bound"]),
        session_profile="throughput",
        input_residency=str(contract["input_residency"]),
        static_shape_reference_path=get_dataset_profile(
            str(contract.get("model_key", DEFAULT_MODEL_KEY))
        ).static_model_path,
    )
    identity: dict[str, Any] = {}
    completed_rows = 0
    worker_started = time.monotonic()
    try:
        with np.load(paths["canonical_exact"], allow_pickle=False) as baseline:
            canonical_logits = np.asarray(baseline["logits"], dtype=np.float32)
        zero_residual = np.zeros((256, 256), dtype=np.int32)
        runner.update_lut(zero_residual)
        identity_logits = np.asarray(runner.run_all(), dtype=np.float32)
        identity = validate_zero_residual_identity(identity_logits, canonical_logits)
        for completed_blocks, (block, block_rows) in enumerate(pending, start=1):
            deltas = asset.deltas_for_population_rows(block_rows)
            count = len(block_rows)
            true_top1_bits = np.empty((count, len(true_clean["true_label"])), dtype=np.uint8)
            true_top5_bits = np.empty((count, len(true_clean["true_label"])), dtype=np.uint8)
            raw_top1_bits = np.empty((count, len(raw_clean["true_label"])), dtype=np.uint8)
            raw_top5_bits = np.empty((count, len(raw_clean["true_label"])), dtype=np.uint8)
            raw_margin_top1 = np.empty((count, len(raw_clean["true_label"])), dtype=np.float32)
            raw_margin_top5 = np.empty((count, len(raw_clean["true_label"])), dtype=np.float32)
            true_top1_drop = np.empty((count,), dtype=np.float32)
            true_top5_drop = np.empty((count,), dtype=np.float32)
            raw_top1_drop = np.empty((count,), dtype=np.float32)
            raw_top5_drop = np.empty((count,), dtype=np.float32)
            elapsed_per_design = np.empty((count,), dtype=np.float32)
            block_started = time.monotonic()
            for local_index, row in enumerate(block_rows.itertuples(index=False)):
                runtime_design_id = str(row.runtime_design_id)
                residual = deltas[runtime_design_id]
                bulk._validate_residual_bound(
                    residual=residual,
                    runtime_design_id=runtime_design_id,
                    bound=int(contract["s8_residual_abs_bound"]),
                )
                runner.update_lut(residual)
                approximate_logits = np.asarray(runner.run_all(), dtype=np.float32)
                if approximate_logits.shape != canonical_logits.shape:
                    raise ValueError(
                        f"{runtime_design_id}: approximate logits {approximate_logits.shape} do not match union {canonical_logits.shape}"
                    )
                true_response = runtime_response_from_logits(approximate_logits[true_positions], true_clean)
                raw_response = runtime_response_from_logits(approximate_logits[raw_positions], raw_clean)
                true_top1_bits[local_index] = np.asarray(true_response["approx_top1_correct_bits"], dtype=np.uint8)
                true_top5_bits[local_index] = np.asarray(true_response["approx_top5_correct_bits"], dtype=np.uint8)
                raw_top1_bits[local_index] = np.asarray(raw_response["approx_top1_correct_bits"], dtype=np.uint8)
                raw_top5_bits[local_index] = np.asarray(raw_response["approx_top5_correct_bits"], dtype=np.uint8)
                raw_margin_top1[local_index] = np.asarray(raw_response["delta_margin_top1"], dtype=np.float32)
                raw_margin_top5[local_index] = np.asarray(raw_response["delta_margin_top5"], dtype=np.float32)
                true_top1_drop[local_index] = float(true_response["top1_drop"])
                true_top5_drop[local_index] = float(true_response["top5_drop"])
                raw_top1_drop[local_index] = float(raw_response["top1_drop"])
                raw_top5_drop[local_index] = float(raw_response["top5_drop"])
                elapsed_per_design[local_index] = float(getattr(runner, "last_run_elapsed_seconds", np.nan))
            payload = {
                "row_index": block_rows["row_index"].to_numpy(dtype=np.int64),
                "public_design_id": block_rows["design_id"].astype(str).to_numpy(dtype="<U128"),
                "runtime_design_id": block_rows["runtime_design_id"].astype(str).to_numpy(dtype="<U128"),
                "true1000_approx_top1_correct_bits": true_top1_bits,
                "true1000_approx_top5_correct_bits": true_top5_bits,
                "approx_top1_correct_bits": raw_top1_bits,
                "approx_top5_correct_bits": raw_top5_bits,
                "delta_margin_top1": raw_margin_top1,
                "delta_margin_top5": raw_margin_top5,
                "true1000_top1_drop": true_top1_drop,
                "true1000_top5_drop": true_top5_drop,
                "raw500_top1_drop": raw_top1_drop,
                "raw500_top5_drop": raw_top5_drop,
                "approximate_elapsed_seconds": elapsed_per_design,
            }
            validate_runtime_payload(
                payload,
                expected_row_index=block_rows["row_index"].to_numpy(dtype=np.int64),
                expected_public_design_id=block_rows["design_id"].astype(str).to_numpy(),
                expected_runtime_design_id=block_rows["runtime_design_id"].astype(str).to_numpy(),
                true1000_clean=true_clean,
                raw500_clean=raw_clean,
            )
            _write_runtime_block(
                paths=paths,
                contract=contract,
                block=block,
                rows=block_rows,
                payload=payload,
                elapsed_seconds=float(time.monotonic() - block_started),
                runner_input_residency=str(getattr(runner, "input_residency", "unknown")),
                identity=identity,
                canonical_exact_contract_sha1=exact_contract_sha1,
                bank_meta_sha1=bank_meta_sha1,
            )
            completed_rows += count
            elapsed = max(time.monotonic() - worker_started, 1.0e-9)
            remaining = sum(len(item_rows) for _item, item_rows in pending[completed_blocks:])
            rate = completed_rows / elapsed
            _atomic_write_json(
                _worker_progress_path(paths, physical_gpu_id),
                {
                    "schema_version": str(contract["schema_version"]),
                    "run_fingerprint": _sha1_json(contract),
                    "physical_gpu_id": physical_gpu_id,
                    "completed_blocks_this_invocation": completed_blocks,
                    "completed_rows_this_invocation": completed_rows,
                    "elapsed_seconds_this_invocation": elapsed,
                    "designs_per_second": rate,
                    "remaining_rows_this_invocation": remaining,
                    "eta_seconds_this_invocation": remaining / rate if rate > 0 else None,
                    "identity": identity,
                },
            )
    finally:
        runner.close()
    return {
        "status": "completed",
        "physical_gpu_id": physical_gpu_id,
        "block_count": len(pending),
        "row_count": completed_rows,
        "elapsed_seconds": float(time.monotonic() - worker_started),
        "identity": identity,
    }


def _quantization_distance(values: np.ndarray, sample_count: int) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) * float(sample_count)
    return np.abs(scaled - np.rint(scaled))


def merge(args: argparse.Namespace) -> dict[str, Any]:
    """Verify every block and write the training-compatible data directory."""

    import pandas as pd

    paths, run_config, contract = _load_run_config(Path(args.output_root))
    rows = pd.read_csv(paths["selected_rows"], low_memory=False)
    blocks = _blocks_from_run_config(run_config)
    true_clean, raw_clean, exact_contract_sha1, bank_meta_sha1 = _load_canonical_clean_banks(
        paths=paths,
        contract=contract,
    )
    payload_parts: list[dict[str, np.ndarray]] = []
    missing_blocks: list[int] = []
    for block in blocks:
        block_rows = rows.iloc[block.start : block.stop].copy().reset_index(drop=True)
        if not _is_complete_block(
            paths=paths,
            contract=contract,
            block=block,
            rows=block_rows,
            true1000_clean=true_clean,
            raw500_clean=raw_clean,
            canonical_exact_contract_sha1=exact_contract_sha1,
            bank_meta_sha1=bank_meta_sha1,
        ):
            missing_blocks.append(block.block_id)
            continue
        data_path, _manifest_path = _block_paths(paths, block.block_id)
        with np.load(data_path, allow_pickle=False) as payload:
            payload_parts.append({name: np.asarray(payload[name]) for name in payload.files})
    if missing_blocks:
        raise RuntimeError(f"cannot merge incomplete/invalid blocks: {missing_blocks[:20]}")
    if not payload_parts:
        raise RuntimeError("no validated runtime blocks are available to merge")
    required_fields = (
        "row_index",
        "public_design_id",
        "runtime_design_id",
        "true1000_approx_top1_correct_bits",
        "true1000_approx_top5_correct_bits",
        "approx_top1_correct_bits",
        "approx_top5_correct_bits",
        "delta_margin_top1",
        "delta_margin_top5",
        "true1000_top1_drop",
        "true1000_top5_drop",
        "raw500_top1_drop",
        "raw500_top5_drop",
    )
    missing_fields = [
        name for name in required_fields if any(name not in payload_part for payload_part in payload_parts)
    ]
    if missing_fields:
        raise KeyError(f"runtime block payloads are missing {missing_fields}")
    merged = {
        name: np.concatenate([payload_part[name] for payload_part in payload_parts], axis=0)
        for name in required_fields
    }
    order = np.argsort(np.asarray(merged["row_index"], dtype=np.int64), kind="stable")
    merged = {name: np.asarray(value)[order] for name, value in merged.items()}
    expected_rows = rows["row_index"].to_numpy(dtype=np.int64)
    expected_public = rows["design_id"].astype(str).to_numpy()
    expected_runtime = rows["runtime_design_id"].astype(str).to_numpy()
    targets = validate_runtime_payload(
        merged,
        expected_row_index=expected_rows,
        expected_public_design_id=expected_public,
        expected_runtime_design_id=expected_runtime,
        true1000_clean=true_clean,
        raw500_clean=raw_clean,
    )
    if np.any(_quantization_distance(targets, int(contract["true1000_count"])) > 5.0e-5):
        raise ValueError("true1000 targets violate 1/1000 quantization")
    raw_drops = np.column_stack([merged["raw500_top1_drop"], merged["raw500_top5_drop"]]).astype(np.float32)
    if np.any(_quantization_distance(raw_drops, int(contract["raw500_count"])) > 5.0e-5):
        raise ValueError("raw500 diagnostics violate 1/500 quantization")

    configuration = pd.read_csv(paths["configuration_index"], low_memory=False)
    if not np.array_equal(configuration["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("prepared configuration_index no longer aligns with selected population")
    if not np.array_equal(configuration["design_id"].astype(str).to_numpy(), expected_public):
        raise ValueError("prepared configuration_index public IDs no longer align")
    if not np.array_equal(configuration["runtime_design_id"].astype(str).to_numpy(), expected_runtime):
        raise ValueError("prepared configuration_index runtime IDs no longer align")
    split_counts = {
        str(name): int(count)
        for name, count in configuration["split"].astype(str).value_counts().items()
    }
    if len(configuration) == sum(EXPECTED_FULL_SPLIT_COUNTS.values()):
        validate_full_population_splits(configuration)
    prepared_split_counts = contract.get("split_counts")
    if isinstance(prepared_split_counts, Mapping) and dict(prepared_split_counts) != split_counts:
        raise ValueError("configuration split counts changed after dataset preparation")

    _atomic_save_npy(paths["targets"], targets)
    merged_index = pd.DataFrame(
        {
            "row_index": expected_rows,
            "design_id": expected_public,
            "top1_drop": targets[:, 0],
            "top5_drop": targets[:, 1],
        }
    )
    merged_index.to_csv(paths["merged_index"], index=False)
    labels = configuration.copy()
    labels["top1_drop"] = targets[:, 0]
    labels["top5_drop"] = targets[:, 1]
    labels["raw500_top1_drop"] = raw_drops[:, 0]
    labels["raw500_top5_drop"] = raw_drops[:, 1]
    labels.to_csv(paths["labels"], index=False)
    _atomic_save_npz(
        paths["runtime_bank"],
        design_id=expected_runtime.astype("<U128"),
        public_design_id=expected_public.astype("<U128"),
        row_index=expected_rows,
        delta_margin_top1=np.asarray(merged["delta_margin_top1"], dtype=np.float32),
        delta_margin_top5=np.asarray(merged["delta_margin_top5"], dtype=np.float32),
        approx_top1_correct_bits=np.asarray(merged["approx_top1_correct_bits"], dtype=np.uint8),
        approx_top5_correct_bits=np.asarray(merged["approx_top5_correct_bits"], dtype=np.uint8),
        # Keep historical runtime-bank key names for consumers.  These are
        # raw500 diagnostics, not the true1000 training labels in targets.npy.
        top1_drop=raw_drops[:, 0],
        top5_drop=raw_drops[:, 1],
        true1000_top1_drop=targets[:, 0],
        true1000_top5_drop=targets[:, 1],
        raw500_top1_drop=raw_drops[:, 0],
        raw500_top5_drop=raw_drops[:, 1],
    )
    _atomic_save_npz(
        paths["true1000_label_bank"],
        design_id=expected_runtime.astype("<U128"),
        public_design_id=expected_public.astype("<U128"),
        row_index=expected_rows,
        exact_top1_correct_bits=np.asarray(true_clean["exact_top1_correct_bits"], dtype=np.uint8),
        exact_top5_correct_bits=np.asarray(true_clean["exact_top5_correct_bits"], dtype=np.uint8),
        approx_top1_correct_bits=np.asarray(merged["true1000_approx_top1_correct_bits"], dtype=np.uint8),
        approx_top5_correct_bits=np.asarray(merged["true1000_approx_top5_correct_bits"], dtype=np.uint8),
        top1_drop=targets[:, 0],
        top5_drop=targets[:, 1],
    )
    manifest = {
        "schema_version": str(contract["schema_version"]),
        "model_key": str(contract.get("model_key", DEFAULT_MODEL_KEY)),
        "model_sha1": str(contract["model_sha1"]),
        "node_registry_sha1": str(contract["node_registry_sha1"]),
        "label_protocol_hash": str(contract["label_protocol_hash"]),
        "injection_mode": dict(contract["injection_mode"]),
        "run_fingerprint": _sha1_json(contract),
        "row_count": int(len(rows)),
        "common_feature_count": 3103,
        "split_counts": split_counts,
        "expected_full_split_counts": dict(EXPECTED_FULL_SPLIT_COUNTS),
        "training_label_protocol": {
            "kind": "canonical_dynamic_int8_true1000",
            "sample_count": int(contract["true1000_count"]),
            "label_protocol_hash": str(contract["label_protocol_hash"]),
            "target_files": ["targets.npy", "labels.csv", "true1000_label_bank.npz"],
            "forbidden_as_training_labels": ["raw500_top1_drop", "raw500_top5_drop", "sample50_only"],
        },
        "runtime_observation_protocol": {
            "kind": "canonical_dynamic_int8_raw500",
            "sample_count": int(contract["raw500_count"]),
            "runtime_bank": "runtime_bank.npz",
            "bank_meta": str(paths["strict50_bank_meta"]),
            "bank_meta_sha1": bank_meta_sha1,
            "canonical_exact_contract_sha1": exact_contract_sha1,
        },
        "static_feature_domain": str(contract["static_features"]["schema"]),
        "runtime_label_domain": str(contract["static_features"]["runtime_domain"]),
        "static_features": dict(contract["static_features"]),
        "canonical_reference_policy": dict(contract["canonical_reference_policy"]),
        "source_contract": dict(contract["source_contract"]),
        "files": {
            "common3103.npy": _sha1_file(paths["common3103"]),
            "configuration_index.csv": _sha1_file(paths["configuration_index"]),
            "runtime_design_id_map.csv": _sha1_file(paths["runtime_map"]),
            "merged_index.csv": _sha1_file(paths["merged_index"]),
            "targets.npy": _sha1_file(paths["targets"]),
            "labels.csv": _sha1_file(paths["labels"]),
            "runtime_bank.npz": _sha1_file(paths["runtime_bank"]),
            "true1000_label_bank.npz": _sha1_file(paths["true1000_label_bank"]),
        },
        "shapes": {
            "common3103": [int(len(rows)), 3103],
            "targets": list(targets.shape),
            "runtime_delta_margin_top1": list(np.asarray(merged["delta_margin_top1"]).shape),
            "runtime_delta_margin_top5": list(np.asarray(merged["delta_margin_top5"]).shape),
            "runtime_approx_top1_correct_bits": list(np.asarray(merged["approx_top1_correct_bits"]).shape),
            "runtime_approx_top5_correct_bits": list(np.asarray(merged["approx_top5_correct_bits"]).shape),
            "true1000_approx_top1_correct_bits": list(
                np.asarray(merged["true1000_approx_top1_correct_bits"]).shape
            ),
            "true1000_approx_top5_correct_bits": list(
                np.asarray(merged["true1000_approx_top5_correct_bits"]).shape
            ),
        },
        "blocks_verified": int(len(blocks)),
        "valid": True,
    }
    _atomic_write_json(paths["manifest"], manifest)
    return {"status": "completed", "output_root": str(paths["root"]), **manifest}


def _run_child(
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(environment),
        )
        return_code = process.wait()
    except BaseException:
        if process is not None:
            _cleanup_worker_processes([(process, handle)])
        raise
    finally:
        if not handle.closed:
            handle.close()
    if return_code != 0:
        raise RuntimeError(f"child worker failed with exit code {return_code}; see {log_path}")


def _cleanup_worker_processes(
    workers: Sequence[tuple[subprocess.Popen[bytes], Any]],
) -> None:
    """Best-effort terminate and reap children without masking the caller's error."""

    for process, _handle in workers:
        try:
            process.terminate()
        except BaseException:
            pass

    try:
        for process, _handle in workers:
            try:
                process.wait(timeout=WORKER_TERMINATION_TIMEOUT_SECONDS)
                continue
            except BaseException:
                pass
            try:
                process.kill()
            except BaseException:
                pass
            try:
                process.wait(timeout=WORKER_TERMINATION_TIMEOUT_SECONDS)
            except BaseException:
                pass
    finally:
        for _process, handle in workers:
            try:
                handle.close()
            except BaseException:
                pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare, build a canonical baseline, dispatch GPU workers, then merge."""

    prepared = prepare(args)
    if prepared.get("status") == "dry_run":
        return prepared
    paths, run_config, contract = _load_run_config(Path(args.output_root))
    bulk = _bulk_module()
    baseline_gpu_id = int(contract["baseline_physical_gpu_id"])
    baseline_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "baseline",
        "--output-root",
        str(paths["root"]),
        "--physical-gpu-id",
        str(baseline_gpu_id),
        "--runtime-device-id",
        "0",
    ]
    _run_child(
        command=baseline_command,
        environment=bulk._isolated_worker_environment(physical_gpu_id=baseline_gpu_id),
        log_path=paths["logs"] / f"baseline_gpu_{baseline_gpu_id:02d}.log",
    )

    failures: list[str] = []
    processes: list[tuple[int, subprocess.Popen[bytes], Any, Path]] = []
    Path(paths["logs"]).mkdir(parents=True, exist_ok=True)
    try:
        for physical_gpu_id in (int(value) for value in contract["gpu_ids"]):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "worker",
                "--output-root",
                str(paths["root"]),
                "--physical-gpu-id",
                str(physical_gpu_id),
                "--runtime-device-id",
                "0",
            ]
            log_path = paths["logs"] / f"worker_gpu_{physical_gpu_id:02d}.log"
            handle = log_path.open("wb")
            try:
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=bulk._isolated_worker_environment(physical_gpu_id=physical_gpu_id),
                )
            except BaseException:
                handle.close()
                raise
            processes.append((physical_gpu_id, process, handle, log_path))

        for physical_gpu_id, process, handle, log_path in processes:
            try:
                return_code = process.wait()
            finally:
                handle.close()
            if return_code != 0:
                failures.append(f"GPU {physical_gpu_id} exited with {return_code}; see {log_path}")
        if failures:
            raise RuntimeError("; ".join(failures))
    except BaseException:
        _cleanup_worker_processes(
            [(process, handle) for _gpu_id, process, handle, _log_path in processes]
        )
        raise
    result = merge(argparse.Namespace(output_root=paths["root"]))
    result["run_fingerprint"] = run_config["run_fingerprint"]
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _resolve_profile_arguments(args: argparse.Namespace):
    profile = get_dataset_profile(getattr(args, "model_key", DEFAULT_MODEL_KEY))
    defaults = {
        "output_root": profile.default_output_root,
        "source_data_dir": profile.default_source_data_dir,
        "model_path": profile.dynamic_model_path,
        "node_registry": profile.node_registry_path,
        "true1000_subset": profile.default_true1000_subset,
        "raw500_subset": profile.default_raw500_subset,
        "true1000_cache": profile.default_true1000_cache,
        "raw500_cache": profile.default_raw500_cache,
        "image_root": DEFAULT_IMAGE_ROOT,
    }
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    args.preprocess_key = profile.preprocess_key
    return profile


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-key",
        choices=tuple(sorted(DATASET_PROFILES)),
        default=DEFAULT_MODEL_KEY,
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source-data-dir", type=Path, default=None)
    parser.add_argument("--s8-asset-root", type=Path, default=DEFAULT_S8_ASSET_ROOT)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--node-registry", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--true1000-subset", type=Path, default=None)
    parser.add_argument("--raw500-subset", type=Path, default=None)
    parser.add_argument("--true1000-cache", type=Path, default=None)
    parser.add_argument("--raw500-cache", type=Path, default=None)
    parser.add_argument(
        "--rebuild-input-caches",
        action="store_true",
        help="replace incompatible model-specific preprocessing caches",
    )
    parser.add_argument("--gpu-ids", default="0,1,2", help="comma-separated physical CUDA GPU IDs")
    parser.add_argument("--baseline-gpu-id", type=int, default=None)
    # Kept only so old launch wrappers fail neither parsing nor resume checks.
    # Signed common3103 construction is CPU streaming and does not use this ID.
    parser.add_argument("--static-feature-gpu-id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--static-feature-batch-size",
        type=int,
        default=64,
        help="number of signed S8S8 LUT rows processed per CPU feature batch",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--max-designs", type=int, default=None)
    parser.add_argument("--runner-factory", default="persistent_iobinding")
    parser.add_argument("--input-residency", choices=("auto", "device", "staged"), default="staged")
    parser.add_argument("--s8-residual-abs-bound", type=int, default=DEFAULT_S8_RESIDUAL_ABS_BOUND)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="freeze image protocols, population, and static common3103")
    _add_prepare_arguments(prepare_parser)
    prepare_mode = prepare_parser.add_mutually_exclusive_group()
    prepare_mode.add_argument("--resume", action="store_true")
    prepare_mode.add_argument("--overwrite", action="store_true")
    prepare_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser("run", help="prepare, evaluate all assigned GPUs, and merge the dataset")
    _add_prepare_arguments(run_parser)
    run_mode = run_parser.add_mutually_exclusive_group()
    run_mode.add_argument("--resume", action="store_true")
    run_mode.add_argument("--overwrite", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    baseline_parser = subparsers.add_parser("baseline", help=argparse.SUPPRESS)
    baseline_parser.add_argument("--output-root", type=Path, required=True)
    baseline_parser.add_argument("--physical-gpu-id", type=int, required=True)
    baseline_parser.add_argument("--runtime-device-id", type=int, default=0)

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--output-root", type=Path, required=True)
    worker_parser.add_argument("--physical-gpu-id", type=int, required=True)
    worker_parser.add_argument("--runtime-device-id", type=int, default=0)

    merge_parser = subparsers.add_parser("merge", help="verify finished blocks and write data/")
    merge_parser.add_argument("--output-root", type=Path, required=True)

    promote_parser = subparsers.add_parser(
        "promote",
        help="validate a finished 20k run and atomically publish its canonical dataset",
    )
    promote_parser.add_argument(
        "--model-key",
        choices=tuple(sorted(DATASET_PROFILES)),
        required=True,
    )
    promote_parser.add_argument("--source-run-root", type=Path, default=None)
    promote_parser.add_argument("--destination-root", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "run":
        result = run(args)
    elif args.command == "baseline":
        result = build_canonical_exact_baseline(args)
    elif args.command == "worker":
        result = run_worker(args)
    elif args.command == "merge":
        result = merge(args)
    elif args.command == "promote":
        from aaae.inference_runtime.transformer.dataset_generation.promotion import promote_dataset

        profile = get_dataset_profile(args.model_key)
        result = promote_dataset(
            source_run_root=args.source_run_root or profile.default_output_root,
            destination_root=args.destination_root or profile.canonical_output_root,
            profile=profile,
        )
    else:  # pragma: no cover - argparse owns the command domain.
        raise AssertionError(args.command)
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
