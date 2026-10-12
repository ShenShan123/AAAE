from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from aaae.accuracy_predictor.cnn.behavior import behavior_sha256_rows
from aaae.accuracy_predictor.cnn.dataset import build_dataset_from_manifest
from aaae.inference_runtime.cnn.errors import LabelGenerationError
from aaae.inference_runtime.cnn.s8s8_conv_graph import validate_s8s8_lut_conv_model
from aaae.inference_runtime.cnn.s8s8_lut_asset import S8S8LutAsset, load_s8s8_lut_asset


SCHEMA_VERSION = "cnn_real_labels_v1"
LUT_DOMAIN = "s8s8_sign_magnitude"


class LabelRuntime(Protocol):
    def predict_exact(self, inputs: np.ndarray) -> np.ndarray: ...

    def predict_approx(self, inputs: np.ndarray, lut_delta: np.ndarray) -> np.ndarray: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class LabelGenerationPlan:
    manifest_path: Path
    manifest: dict[str, Any]
    model_path: Path
    model_sha256: str
    model_input_name: str
    dataset_path: Path
    dataset_sha256: str
    preprocessing_id: str
    node_names: tuple[str, ...]
    asset_root: Path
    design_ids: tuple[str, ...]
    gpu_id: int
    gpu_ids: tuple[int, ...]
    batch_size: int
    block_designs: int
    output_dir: Path
    canonical_profile_metadata: Path | None = None
    canonical_profile_arrays: Path | None = None
    canonical_output_subdir: str = "canonical"
    canonical_chunk_designs: int = 64
    input_residency: str = "auto"
    preallocate_outputs: bool = True
    cudnn_conv_algo_search: str = "DEFAULT"
    residual_abs_bound: int | None = None


@dataclass(frozen=True)
class LabelGenerationOutcome:
    output_dir: Path
    design_count: int
    sample_count: int


RuntimeFactory = Callable[[LabelGenerationPlan], LabelRuntime]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise LabelGenerationError(f"manifest.{key} must be an object")
    return value


def _resolve_file(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LabelGenerationError(f"{field} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise LabelGenerationError(f"{field} is missing: {path}")
    return path


def _resolve_dir(base: Path, value: object, *, field: str, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LabelGenerationError(f"{field} must be a nonempty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if must_exist and not path.is_dir():
        raise LabelGenerationError(f"{field} is missing: {path}")
    return path


def _relative_output_subdir(value: object) -> str:
    if value is None:
        return "canonical"
    if not isinstance(value, str) or not value.strip():
        raise LabelGenerationError("canonical.output_subdir must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or not path.parts or path == Path(".") or ".." in path.parts:
        raise LabelGenerationError("canonical.output_subdir must stay inside the label output")
    return path.as_posix()


def _checked_sha256(path: Path, value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise LabelGenerationError(f"{field} must contain a SHA-256 checksum")
    actual = _sha256(path)
    if actual != value.lower():
        raise LabelGenerationError(f"{field} checksum does not match {path}")
    return actual


def load_label_generation_plan(
    manifest_path: Path,
    *,
    allow_existing_output: bool = False,
) -> LabelGenerationPlan:
    path = Path(manifest_path).expanduser().resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelGenerationError(f"unable to read label-generation manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise LabelGenerationError("label-generation manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise LabelGenerationError(f"schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("task") != "classification":
        raise LabelGenerationError("CNN real labels currently support task='classification' only")
    base = path.parent

    model = _required_dict(manifest, "model")
    if model.get("arithmetic_domain") != LUT_DOMAIN:
        raise LabelGenerationError(
            f"model.arithmetic_domain must be {LUT_DOMAIN!r}; unsigned/proxy domains are not accepted"
        )
    model_path = _resolve_file(base, model.get("path"), field="model.path")
    model_hash = _checked_sha256(model_path, model.get("sha256"), field="model.sha256")
    input_name = model.get("input_name")
    if not isinstance(input_name, str) or not input_name:
        raise LabelGenerationError("model.input_name must be nonempty")

    dataset = _required_dict(manifest, "dataset")
    if dataset.get("kind") != "preprocessed_npz_v1":
        raise LabelGenerationError("dataset.kind must be 'preprocessed_npz_v1'")
    dataset_path = _resolve_file(base, dataset.get("path"), field="dataset.path")
    dataset_hash = _checked_sha256(dataset_path, dataset.get("sha256"), field="dataset.sha256")
    preprocessing_id = dataset.get("preprocessing_id")
    if not isinstance(preprocessing_id, str) or not preprocessing_id:
        raise LabelGenerationError("dataset.preprocessing_id must be nonempty")

    injection = _required_dict(manifest, "injection")
    if injection.get("lut_domain") != LUT_DOMAIN or injection.get("semantic_zero_mask") is not True:
        raise LabelGenerationError(
            "injection requires lut_domain='s8s8_sign_magnitude' and semantic_zero_mask=true"
        )
    raw_node_names = injection.get("node_names")
    if not isinstance(raw_node_names, list):
        raise LabelGenerationError("injection.node_names must be a list")
    node_names = tuple(str(value).strip() for value in raw_node_names if str(value).strip())
    if not node_names or len(set(node_names)) != len(node_names):
        raise LabelGenerationError("injection.node_names must be nonempty and unique")

    designs = _required_dict(manifest, "designs")
    asset_root = _resolve_dir(base, designs.get("asset_root"), field="designs.asset_root", must_exist=True)
    raw_design_ids = designs.get("design_ids")
    if not isinstance(raw_design_ids, list):
        raise LabelGenerationError("designs.design_ids must be a list")
    design_ids = tuple(str(value).strip() for value in raw_design_ids if str(value).strip())
    if not design_ids or len(set(design_ids)) != len(design_ids):
        raise LabelGenerationError("designs.design_ids must be nonempty and unique")

    execution = _required_dict(manifest, "execution")
    if execution.get("device") != "cuda":
        raise LabelGenerationError(
            "execution.device must be 'cuda'; the approximate Conv has no CPU implementation"
        )
    raw_gpu_ids = execution.get("gpu_ids")
    if raw_gpu_ids is None:
        raw_gpu_ids = [execution.get("gpu_id")]
    if not isinstance(raw_gpu_ids, list) or not raw_gpu_ids:
        raise LabelGenerationError("execution.gpu_ids must be a nonempty list")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw_gpu_ids):
        raise LabelGenerationError("execution.gpu_ids must contain nonnegative integers")
    gpu_ids = tuple(int(value) for value in raw_gpu_ids)
    if len(set(gpu_ids)) != len(gpu_ids):
        raise LabelGenerationError("execution.gpu_ids must be unique")
    gpu_id = gpu_ids[0]
    batch_size = execution.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise LabelGenerationError("execution.batch_size must be a positive integer")
    block_designs = execution.get("block_designs", 16)
    if isinstance(block_designs, bool) or not isinstance(block_designs, int) or block_designs < 1:
        raise LabelGenerationError("execution.block_designs must be a positive integer")
    input_residency = str(execution.get("input_residency", "auto")).strip().lower()
    if input_residency not in {"auto", "device", "staged"}:
        raise LabelGenerationError(
            "execution.input_residency must be 'auto', 'device', or 'staged'"
        )
    preallocate_outputs = execution.get("preallocate_outputs", True)
    if not isinstance(preallocate_outputs, bool):
        raise LabelGenerationError("execution.preallocate_outputs must be boolean")
    cudnn_conv_algo_search = str(
        execution.get("cudnn_conv_algo_search", "DEFAULT")
    ).strip().upper()
    if cudnn_conv_algo_search not in {"DEFAULT", "HEURISTIC", "EXHAUSTIVE"}:
        raise LabelGenerationError(
            "execution.cudnn_conv_algo_search must be DEFAULT, HEURISTIC, or EXHAUSTIVE"
        )
    configured_bound = execution.get("residual_abs_bound")
    if configured_bound is not None and (
        isinstance(configured_bound, bool)
        or not isinstance(configured_bound, int)
        or configured_bound < 0
    ):
        raise LabelGenerationError("execution.residual_abs_bound must be a nonnegative integer")

    output_dir = _resolve_dir(base, manifest.get("output_dir"), field="output_dir", must_exist=False)
    if output_dir.exists() and not allow_existing_output:
        raise LabelGenerationError(f"output_dir already exists: {output_dir}")

    canonical_profile_metadata: Path | None = None
    canonical_profile_arrays: Path | None = None
    canonical_output_subdir = "canonical"
    canonical_chunk_designs = 64
    canonical = manifest.get("canonical")
    if canonical is not None:
        if not isinstance(canonical, dict):
            raise LabelGenerationError("manifest.canonical must be an object")
        profile = canonical.get("profile")
        if not isinstance(profile, dict):
            raise LabelGenerationError(
                "manifest.canonical.profile must provide metadata and arrays"
            )
        canonical_profile_metadata = _resolve_file(
            base,
            profile.get("metadata"),
            field="canonical.profile.metadata",
        )
        canonical_profile_arrays = _resolve_file(
            base,
            profile.get("arrays"),
            field="canonical.profile.arrays",
        )
        if canonical.get("domain") != "s8_numeric":
            raise LabelGenerationError(
                "canonical.domain must be 's8_numeric': signed_code_plus_128 uses numeric -128..127 order"
            )
        canonical_output_subdir = _relative_output_subdir(canonical.get("output_subdir"))
        raw_chunk = canonical.get("chunk_designs", 64)
        if isinstance(raw_chunk, bool) or not isinstance(raw_chunk, int) or raw_chunk < 1:
            raise LabelGenerationError("canonical.chunk_designs must be a positive integer")
        canonical_chunk_designs = int(raw_chunk)
    validate_s8s8_lut_conv_model(model_path, node_names=node_names)
    return LabelGenerationPlan(
        manifest_path=path,
        manifest=manifest,
        model_path=model_path,
        model_sha256=model_hash,
        model_input_name=input_name,
        dataset_path=dataset_path,
        dataset_sha256=dataset_hash,
        preprocessing_id=preprocessing_id,
        node_names=node_names,
        asset_root=asset_root,
        design_ids=design_ids,
        gpu_id=gpu_id,
        gpu_ids=gpu_ids,
        batch_size=batch_size,
        block_designs=block_designs,
        output_dir=output_dir,
        canonical_profile_metadata=canonical_profile_metadata,
        canonical_profile_arrays=canonical_profile_arrays,
        canonical_output_subdir=canonical_output_subdir,
        canonical_chunk_designs=canonical_chunk_designs,
        input_residency=input_residency,
        preallocate_outputs=preallocate_outputs,
        cudnn_conv_algo_search=cudnn_conv_algo_search,
        residual_abs_bound=configured_bound,
    )


def _load_samples(plan: LabelGenerationPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(plan.dataset_path, allow_pickle=False) as asset:
            inputs = np.asarray(asset["inputs"], dtype=np.float32)
            raw_labels = np.asarray(asset["labels"])
            sample_ids = np.asarray(asset["sample_ids"]).astype(str)
    except (OSError, KeyError, ValueError) as exc:
        raise LabelGenerationError(f"invalid preprocessed sample asset {plan.dataset_path}: {exc}") from exc
    if raw_labels.dtype.kind not in {"i", "u"} or raw_labels.dtype == np.bool_:
        raise LabelGenerationError("sample labels must use an integer dtype")
    if raw_labels.dtype.kind == "u" and np.any(raw_labels > np.iinfo(np.int64).max):
        raise LabelGenerationError("sample labels exceed the int64 classification contract")
    labels = np.asarray(raw_labels, dtype=np.int64)
    if inputs.ndim < 2 or labels.shape != (inputs.shape[0],) or sample_ids.shape != (inputs.shape[0],):
        raise LabelGenerationError("sample inputs, labels, and sample_ids do not share one leading dimension")
    if inputs.shape[0] == 0 or not np.isfinite(inputs).all():
        raise LabelGenerationError("sample inputs must be nonempty and finite")
    if np.any(labels < 0) or len(set(sample_ids.tolist())) != len(sample_ids):
        raise LabelGenerationError("sample labels must be nonnegative and sample_ids must be unique")
    return inputs, labels, sample_ids


def _correctness(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[0] != labels.shape[0] or values.shape[1] < 2:
        raise LabelGenerationError(
            f"classification runtime returned logits with shape {values.shape}; expected [samples, classes]"
        )
    if not np.isfinite(values).all() or np.any(labels >= values.shape[1]):
        raise LabelGenerationError("classification logits are nonfinite or labels exceed the class count")
    sample_rows = np.arange(values.shape[0], dtype=np.int64)
    label_scores = values[sample_rows, labels]
    class_indices = np.arange(values.shape[1], dtype=np.int64)[None, :]
    rank = np.count_nonzero(values > label_scores[:, None], axis=1)
    rank += np.count_nonzero(
        (values == label_scores[:, None]) & (class_indices < labels[:, None]),
        axis=1,
    )
    top1 = rank == 0
    top5 = rank < min(5, values.shape[1])
    return np.asarray(top1, dtype=np.bool_), np.asarray(top5, dtype=np.bool_)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(root: Path) -> None:
    checksums = {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _default_runtime_factory(plan: LabelGenerationPlan) -> LabelRuntime:
    from aaae.inference_runtime.cnn.s8s8_conv_cuda import S8S8ConvCudaRuntime

    return S8S8ConvCudaRuntime.from_plan(plan)


def _materialize_canonical_dataset(plan: LabelGenerationPlan, staging: Path) -> Path | None:
    if plan.canonical_profile_metadata is None or plan.canonical_profile_arrays is None:
        return None

    profile_metadata_name = "canonical_profile_metadata.source.json"
    profile_arrays_name = "canonical_profile_arrays.source.npz"
    shutil.copy2(plan.canonical_profile_metadata, staging / profile_metadata_name)
    shutil.copy2(plan.canonical_profile_arrays, staging / profile_arrays_name)
    build_manifest_path = staging / "canonical_build_manifest.json"
    build_manifest = {
        "schema_version": "cnn_dataset_build_manifest_v1",
        "source": "measured",
        "target_units": "proportion",
        "error_convention": "approx_minus_exact",
        "chunk_designs": plan.canonical_chunk_designs,
        "profile": {
            "metadata": profile_metadata_name,
            "arrays": profile_arrays_name,
        },
        "designs": {
            "design_ids": "design_ids.csv",
            "delta": "delta.npy",
            "domain": "s8_numeric",
        },
        "labels": "labels.csv",
        "output_dir": plan.canonical_output_subdir,
    }
    build_manifest_path.write_text(
        json.dumps(build_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outcome = build_dataset_from_manifest(build_manifest_path)
    return outcome.train_manifest_path


def _run_generation(
    plan: LabelGenerationPlan,
    *,
    asset: S8S8LutAsset,
    runtimes: tuple[tuple[int, LabelRuntime], ...],
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    staging: Path,
) -> LabelGenerationOutcome:
    inputs, labels, sample_ids = samples
    if not runtimes:
        raise LabelGenerationError("CNN label generation requires at least one CUDA runtime")

    def exact_for_runtime(entry: tuple[int, LabelRuntime]) -> tuple[int, np.ndarray, np.ndarray]:
        gpu_id, runtime = entry
        logits = runtime.predict_exact(inputs)
        top1, top5 = _correctness(logits, labels)
        return gpu_id, top1, top5

    if len(runtimes) == 1:
        exact_results = [exact_for_runtime(runtimes[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=len(runtimes),
            thread_name_prefix="cnn-exact-gpu",
        ) as executor:
            exact_results = list(executor.map(exact_for_runtime, runtimes))
    exact_top1 = exact_results[0][1]
    exact_top5 = exact_results[0][2]
    for gpu_id, candidate_top1, candidate_top5 in exact_results[1:]:
        if not np.array_equal(candidate_top1, exact_top1) or not np.array_equal(
            candidate_top5,
            exact_top5,
        ):
            raise LabelGenerationError(
                "multi-GPU exact top-k baselines disagree; refusing to mix labels from "
                f"gpu_id={gpu_id} into one canonical dataset"
            )
    design_count = len(plan.design_ids)
    sample_count = len(labels)

    np.save(staging / "design_ids.npy", np.asarray(plan.design_ids))
    np.save(staging / "sample_ids.npy", sample_ids)
    np.save(staging / "class_labels.npy", labels)
    np.save(staging / "exact_top1_correct.npy", exact_top1)
    np.save(staging / "exact_top5_correct.npy", exact_top5)
    approx_top1 = np.lib.format.open_memmap(
        staging / "approx_top1_correct.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(design_count, sample_count),
    )
    approx_top5 = np.lib.format.open_memmap(
        staging / "approx_top5_correct.npy",
        mode="w+",
        dtype=np.bool_,
        shape=(design_count, sample_count),
    )
    canonical_delta = np.lib.format.open_memmap(
        staging / "delta.npy",
        mode="w+",
        dtype=np.int32,
        shape=(design_count, 256, 256),
    )
    rows: list[dict[str, object] | None] = [None] * design_count
    design_rows: list[dict[str, object] | None] = [None] * design_count
    exact_top1_accuracy = float(np.mean(exact_top1))
    exact_top5_accuracy = float(np.mean(exact_top5))
    asset_manifest_path = asset.root / "manifest.json"
    asset_sha256 = _sha256(asset_manifest_path)
    effective_block_designs = min(
        plan.block_designs,
        max(1, math.ceil(design_count / len(runtimes))),
    )
    blocks = [
        tuple(range(start, min(start + effective_block_designs, design_count)))
        for start in range(0, design_count, effective_block_designs)
    ]
    assignments: list[list[int]] = [[] for _ in runtimes]
    for block_index, block in enumerate(blocks):
        assignments[block_index % len(runtimes)].extend(block)
    worker_timings: dict[int, dict[str, float | int]] = {}

    def evaluate_assignment(
        entry: tuple[int, LabelRuntime],
        design_indices: list[int],
    ) -> None:
        gpu_id, runtime = entry
        started = time.perf_counter()
        for design_index in design_indices:
            design_id = plan.design_ids[design_index]
            delta = asset.delta_for(design_id)
            behavior_sha256 = behavior_sha256_rows(delta[None, :, :])[0]
            canonical_delta[design_index] = delta
            logits = runtime.predict_approx(inputs, delta)
            top1, top5 = _correctness(logits, labels)
            approx_top1[design_index] = top1
            approx_top5[design_index] = top5
            top1_accuracy = float(np.mean(top1))
            top5_accuracy = float(np.mean(top5))
            rows[design_index] = {
                "design_id": design_id,
                "behavior_sha256": behavior_sha256,
                "asset_sha256": asset_sha256,
                "sample_count": sample_count,
                "exact_top1_accuracy": format(exact_top1_accuracy, ".17g"),
                "exact_top5_accuracy": format(exact_top5_accuracy, ".17g"),
                "approx_top1_accuracy": format(top1_accuracy, ".17g"),
                "approx_top5_accuracy": format(top5_accuracy, ".17g"),
                "top1_drop": format(exact_top1_accuracy - top1_accuracy, ".17g"),
                "top5_drop": format(exact_top5_accuracy - top5_accuracy, ".17g"),
            }
            design_rows[design_index] = {
                "design_id": design_id,
                "behavior_sha256": behavior_sha256,
                "asset_sha256": asset_sha256,
            }
        elapsed = time.perf_counter() - started
        worker_timings[gpu_id] = {
            "design_count": len(design_indices),
            "elapsed_seconds": elapsed,
            "designs_per_second": len(design_indices) / elapsed if elapsed > 0 else 0.0,
        }

    if len(runtimes) == 1:
        evaluate_assignment(runtimes[0], assignments[0])
    else:
        with ThreadPoolExecutor(
            max_workers=len(runtimes),
            thread_name_prefix="cnn-approx-gpu",
        ) as executor:
            futures = [
                executor.submit(evaluate_assignment, entry, indices)
                for entry, indices in zip(runtimes, assignments, strict=True)
                if indices
            ]
            for future in futures:
                future.result()
    if any(row is None for row in rows) or any(row is None for row in design_rows):
        raise LabelGenerationError("multi-GPU CNN evaluation did not cover every requested design")
    approx_top1.flush()
    approx_top5.flush()
    canonical_delta.flush()
    # Workers have joined; release the maps while keeping closure names bound.
    approx_top1 = approx_top5 = canonical_delta = None
    completed_rows = [row for row in rows if row is not None]
    completed_design_rows = [row for row in design_rows if row is not None]
    _write_csv(staging / "labels.csv", completed_rows)
    _write_csv(staging / "design_ids.csv", completed_design_rows)
    shutil.copy2(plan.manifest_path, staging / "label_manifest.snapshot.json")
    shutil.copy2(asset_manifest_path, staging / "lut_asset_manifest.snapshot.json")
    runtime_metadata_by_gpu: dict[str, dict[str, object]] = {}
    for gpu_id, runtime in runtimes:
        runtime_metadata = getattr(runtime, "execution_metadata", None)
        if runtime_metadata is not None and not isinstance(runtime_metadata, dict):
            raise LabelGenerationError("label runtime execution_metadata must be a dictionary")
        runtime_metadata_by_gpu[str(gpu_id)] = runtime_metadata or {
            "requested_device": "cuda",
            "runtime_factory": "injected_test_boundary",
        }
    run_manifest = {
        "schema_version": 1,
        "source_manifest_schema": SCHEMA_VERSION,
        "task": "classification",
        "source": "measured_real_cuda",
        "target_convention": "exact_accuracy_minus_approximate_accuracy",
        "target_units": "proportion",
        "runtime_domain": "s8s8_sign_magnitude_int8_qdq",
        "runtime_operand_dtype": "int8",
        "uint8_runtime_data_allowed": False,
        "lut_domain": LUT_DOMAIN,
        "lut_index_encoding": "signed_code_plus_128",
        "canonical_behavior_hash_schema": "cnn_canonical_delta_behavior_v1",
        "lut_asset_manifest_sha256": asset_sha256,
        "lut_asset_target_semantics": asset.manifest.get("target_semantics"),
        "lut_asset_row_count": asset.manifest.get("row_count"),
        "semantic_zero_mask": True,
        "semantic_zero_definition": "raw_int8_code_equals_affine_zero_point",
        "model_path": str(plan.model_path),
        "model_sha256": plan.model_sha256,
        "dataset_path": str(plan.dataset_path),
        "dataset_sha256": plan.dataset_sha256,
        "preprocessing_id": plan.preprocessing_id,
        "selected_conv_nodes": list(plan.node_names),
        "design_count": design_count,
        "sample_count": sample_count,
        "gpu_id": plan.gpu_id,
        "gpu_ids": list(plan.gpu_ids),
        "block_designs": plan.block_designs,
        "effective_block_designs": effective_block_designs,
        "residual_abs_bound": plan.residual_abs_bound,
        "execution": runtime_metadata_by_gpu[str(plan.gpu_id)],
        "execution_by_gpu": runtime_metadata_by_gpu,
        "worker_timings": {str(key): value for key, value in sorted(worker_timings.items())},
        "performance_claim_allowed": True,
        "per_image_alignment": {
            "design_axis": "design_ids.npy",
            "sample_axis": "sample_ids.npy",
            "top1_correct": "approx_top1_correct.npy",
            "top5_correct": "approx_top5_correct.npy",
        },
        "canonical_dataset": {
            "enabled": plan.canonical_profile_metadata is not None,
            "domain": "s8_numeric",
            "build_manifest": (
                "canonical_build_manifest.json"
                if plan.canonical_profile_metadata is not None
                else None
            ),
            "train_manifest": (
                f"{plan.canonical_output_subdir}/train_manifest.json"
                if plan.canonical_profile_metadata is not None
                else None
            ),
        },
    }
    (staging / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _materialize_canonical_dataset(plan, staging)
    _write_checksums(staging)
    return LabelGenerationOutcome(
        output_dir=plan.output_dir,
        design_count=design_count,
        sample_count=sample_count,
    )


def generate_labels_from_manifest(
    manifest_path: Path,
    *,
    runtime_factory: RuntimeFactory | None = None,
) -> LabelGenerationOutcome:
    """Generate measured CNN accuracy-drop labels from one frozen manifest.

    The optional factory is the GPU system boundary used by focused tests. A
    normal call always constructs the signed-LUT CUDA Conv runtime.
    """

    plan = load_label_generation_plan(manifest_path)
    samples = _load_samples(plan)
    asset = load_s8s8_lut_asset(plan.asset_root, verify_checksums=True)
    design_ids = asset.require_design_ids(plan.design_ids)
    if design_ids != plan.design_ids:
        raise LabelGenerationError("design order changed during LUT validation")
    actual_residual_abs_bound = asset.max_abs_delta(plan.design_ids)
    if (
        plan.residual_abs_bound is not None
        and actual_residual_abs_bound > plan.residual_abs_bound
    ):
        raise LabelGenerationError(
            "selected S8S8 LUT residual abs max "
            f"{actual_residual_abs_bound} exceeds execution.residual_abs_bound="
            f"{plan.residual_abs_bound}"
        )
    plan = replace(plan, residual_abs_bound=actual_residual_abs_bound)

    staging = plan.output_dir.with_name(
        f".{plan.output_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    runtimes: list[tuple[int, LabelRuntime]] = []
    try:
        factory = runtime_factory or _default_runtime_factory
        for gpu_id in plan.gpu_ids:
            runtime_plan = replace(plan, gpu_id=gpu_id)
            runtimes.append((gpu_id, factory(runtime_plan)))
        outcome = _run_generation(
            plan,
            asset=asset,
            runtimes=tuple(runtimes),
            samples=samples,
            staging=staging,
        )
        staging.replace(plan.output_dir)
        return outcome
    except LabelGenerationError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise LabelGenerationError(f"CNN real-label generation failed: {type(exc).__name__}: {exc}") from exc
    finally:
        for _, runtime in runtimes:
            runtime.close()


__all__ = [
    "LabelGenerationError",
    "LabelGenerationOutcome",
    "LabelGenerationPlan",
    "generate_labels_from_manifest",
    "load_label_generation_plan",
]
