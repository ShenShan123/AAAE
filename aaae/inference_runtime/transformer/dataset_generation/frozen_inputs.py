"""Deterministic ImageNet protocol selection and model-specific input caches."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import onnx

from aaae.inference_runtime.transformer.dataset_generation.profiles import TransformerDatasetProfile
from aaae.paths import AI_MODELS_ROOT, AI_MODELS_SRC



from aaae.inference_runtime.preprocess import prepare_classification_input  # noqa: E402
from aaae.inference_runtime.imagenet import imagenet_gt_indices  # noqa: E402


FROZEN_PROTOCOL_SCHEMA = "transformer_frozen_imagenet_protocol_v1"
DEFAULT_IMAGE_ROOT = AI_MODELS_ROOT / "dataset" / "imagenet-1k"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def select_protocol_indices(*, total: int, sample_count: int, sample_seed: int) -> np.ndarray:
    """Match the retained Swin protocol's seeded, sorted sampling policy."""

    if int(total) <= 0:
        raise ValueError("ImageNet protocol requires a positive population size")
    if int(sample_count) <= 0 or int(sample_count) > int(total):
        raise ValueError(f"sample_count={sample_count} is outside 1..{total}")
    rng = np.random.default_rng(int(sample_seed))
    selected = sorted(
        int(value)
        for value in rng.choice(int(total), size=int(sample_count), replace=False).tolist()
    )
    return np.asarray(selected, dtype=np.int64)


def _model_input_metadata(model_path: Path) -> tuple[str, tuple[int | str | None, ...]]:
    model = onnx.load(str(model_path), load_external_data=False)
    if not model.graph.input:
        raise ValueError(f"{model_path} has no graph input")
    value = model.graph.input[0]
    shape: list[int | str | None] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        elif dim.dim_param:
            shape.append(str(dim.dim_param))
        else:
            shape.append(None)
    return str(value.name), tuple(shape)


def _selection_hash(image_names: list[str], labels: np.ndarray) -> str:
    return hashlib.sha1(
        _canonical_json(
            {
                "image_names": image_names,
                "labels": np.asarray(labels, dtype=np.int64).tolist(),
            }
        ).encode("utf-8")
    ).hexdigest()


def _cache_metadata_matches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    """Accept a relocated model path only when every content anchor matches."""

    actual_values = dict(actual)
    expected_values = dict(expected)
    actual_model_path = actual_values.pop("model_path", None)
    expected_model_path = expected_values.pop("model_path", None)
    return (
        isinstance(actual_model_path, str)
        and bool(actual_model_path)
        and isinstance(expected_model_path, str)
        and bool(expected_model_path)
        and actual_values == expected_values
    )


def _load_or_create_subset(
    *,
    subset_path: Path,
    image_paths: list[Path],
    labels_all: np.ndarray,
    role: str,
    sample_count: int,
    sample_seed: int,
) -> tuple[dict[str, Any], list[Path], np.ndarray]:
    expected_indices = select_protocol_indices(
        total=len(image_paths),
        sample_count=int(sample_count),
        sample_seed=int(sample_seed),
    )
    expected_names = [image_paths[int(index)].name for index in expected_indices]
    expected_labels = np.asarray(labels_all[expected_indices], dtype=np.int64)
    if subset_path.is_file():
        payload = json.loads(subset_path.read_text(encoding="utf-8"))
        actual_indices = np.asarray(payload.get("indices", ()), dtype=np.int64)
        actual_names = [str(value) for value in payload.get("image_names", ())]
        actual_labels = np.asarray(payload.get("labels", expected_labels), dtype=np.int64)
        if int(payload.get("sample_count", -1)) != int(sample_count):
            raise ValueError(f"{subset_path}: sample_count does not match {sample_count}")
        if int(payload.get("sample_seed", -1)) != int(sample_seed):
            raise ValueError(f"{subset_path}: sample_seed does not match {sample_seed}")
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError(f"{subset_path}: indices do not match the deterministic seed protocol")
        if actual_names != expected_names or not np.array_equal(actual_labels, expected_labels):
            raise ValueError(f"{subset_path}: image names or labels do not match ImageNet")
        return payload, [image_paths[int(index)] for index in actual_indices], actual_labels

    payload = {
        "schema_version": FROZEN_PROTOCOL_SCHEMA,
        "model_key": "model_independent",
        "role": str(role),
        "sample_count": int(sample_count),
        "sample_seed": int(sample_seed),
        "selection_policy": "sorted numpy.default_rng(seed).choice(50000, replace=False)",
        "image_names": expected_names,
        "indices": expected_indices.tolist(),
        "labels": expected_labels.tolist(),
    }
    _atomic_write_json(subset_path, payload)
    return payload, [image_paths[int(index)] for index in expected_indices], expected_labels


def ensure_frozen_input_cache(
    *,
    profile: TransformerDatasetProfile,
    role: str,
    sample_count: int,
    sample_seed: int,
    subset_path: Path,
    cache_path: Path,
    image_root: Path = DEFAULT_IMAGE_ROOT,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Create or verify one model-specific cache for a frozen image protocol."""

    image_root = Path(image_root).resolve()
    subset_path = Path(subset_path).resolve()
    cache_path = Path(cache_path).resolve()
    cache_meta_path = cache_path.with_suffix(".json")
    image_paths = sorted(image_root.glob("ILSVRC2012_val_*.JPEG"))
    if not image_paths:
        raise FileNotFoundError(f"no ImageNet validation images found under {image_root}")
    labels_all = np.asarray(imagenet_gt_indices(len(image_paths)), dtype=np.int64)
    if labels_all.shape != (len(image_paths),):
        raise ValueError("ImageNet ground-truth mapping does not align with validation images")
    subset, selected_paths, labels = _load_or_create_subset(
        subset_path=subset_path,
        image_paths=image_paths,
        labels_all=labels_all,
        role=str(role),
        sample_count=int(sample_count),
        sample_seed=int(sample_seed),
    )
    image_names = [path.name for path in selected_paths]

    input_name, input_shape = _model_input_metadata(profile.dynamic_model_path)
    if len(input_shape) != 4 or tuple(input_shape[1:]) != (3, 224, 224):
        raise ValueError(
            f"{profile.dynamic_model_path} has unsupported input shape {input_shape}; expected [B,3,224,224]"
        )
    preprocess_source = Path(prepare_classification_input.__code__.co_filename).resolve()
    expected_meta = {
        "schema_version": FROZEN_PROTOCOL_SCHEMA,
        "model_key": profile.model_key,
        "model_path": str(profile.dynamic_model_path.resolve()),
        "model_sha1": _sha1_file(profile.dynamic_model_path),
        "model_input_name": input_name,
        "model_input_shape": list(input_shape),
        "preprocess_key": profile.preprocess_key,
        "preprocess_source_sha1": _sha1_file(preprocess_source),
        "sample_count": int(sample_count),
        "sample_seed": int(sample_seed),
        "selection_hash": _selection_hash(image_names, labels),
        "subset_sha1": _sha1_file(subset_path),
        "dtype": "float32",
        "cached_shape": [int(sample_count), 3, 224, 224],
    }
    if cache_path.is_file() and cache_meta_path.is_file() and not bool(rebuild):
        actual_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
        cache = np.load(cache_path, mmap_mode="r")
        cache_sha1 = _sha1_file(cache_path)
        metadata_without_hash = dict(actual_meta)
        recorded_cache_sha1 = metadata_without_hash.pop("cache_sha1", None)
        metadata_matches = _cache_metadata_matches(metadata_without_hash, expected_meta)
        valid = (
            metadata_matches
            and cache.dtype == np.float32
            and tuple(cache.shape) == tuple(expected_meta["cached_shape"])
            and (recorded_cache_sha1 is None or recorded_cache_sha1 == cache_sha1)
        )
        del cache
        if not valid:
            raise RuntimeError(
                f"{role}: existing cache is incompatible; pass --rebuild-input-caches to replace it"
            )
        # Older cache metadata predates the content digest.  Upgrade only
        # after its deterministic metadata and array contract have passed.
        if recorded_cache_sha1 is None or metadata_without_hash != expected_meta:
            _atomic_write_json(cache_meta_path, {**expected_meta, "cache_sha1": cache_sha1})
        return {
            "status": "reused",
            "role": str(role),
            "subset_path": str(subset_path),
            "cache_path": str(cache_path),
            "sample_count": int(sample_count),
            "cache_sha1": cache_sha1,
        }
    if (cache_path.exists() or cache_meta_path.exists()) and not bool(rebuild):
        raise RuntimeError(
            f"{role}: input cache is incomplete; pass --rebuild-input-caches to replace it"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.stem}.tmp.{os.getpid()}{cache_path.suffix}")
    try:
        cache = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=tuple(expected_meta["cached_shape"]),
        )
        try:
            for position, image_path in enumerate(selected_paths):
                tensor = prepare_classification_input(profile.preprocess_key, image_path, nhwc=False)
                if tensor.shape != (1, 3, 224, 224):
                    raise ValueError(f"{image_path}: preprocessing returned {tensor.shape}")
                cache[position] = tensor[0]
                if (position + 1) % 100 == 0 or position + 1 == len(selected_paths):
                    print(
                        f"[{profile.model_key}:{role}] preprocessed {position + 1}/{len(selected_paths)}",
                        flush=True,
                    )
            cache.flush()
        finally:
            del cache
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    cache_sha1 = _sha1_file(cache_path)
    _atomic_write_json(cache_meta_path, {**expected_meta, "cache_sha1": cache_sha1})
    return {
        "status": "created",
        "role": str(role),
        "subset_path": str(subset_path),
        "cache_path": str(cache_path),
        "sample_count": int(sample_count),
        "cache_sha1": cache_sha1,
        "selection_model_key": str(subset.get("model_key", "model_independent")),
    }
