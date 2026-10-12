from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from aaae.accuracy_predictor.cnn.behavior import behavior_sha256_rows
from aaae.inference_runtime.cnn.errors import LabelGenerationError
from aaae.inference_runtime.cnn.s8s8_lut_asset import S8S8LutAsset, load_s8s8_lut_asset

from .real_labels import (
    LUT_DOMAIN,
    SCHEMA_VERSION,
    LabelGenerationOutcome,
    LabelGenerationPlan,
    LabelRuntime,
    RuntimeFactory,
    _correctness,
    _default_runtime_factory,
    _load_samples,
    _sha256,
    _write_checksums,
    _write_csv,
    load_label_generation_plan,
)


SHARD_CONTRACT_SCHEMA = "cnn_real_label_shards_v1"
SHARED_SCHEMA = "cnn_real_label_shared_exact_v1"
SHARD_SCHEMA = "cnn_real_label_design_shard_v1"
MERGE_SCHEMA = "cnn_real_label_shard_merge_v1"

_SHARED_FILES = frozenset(
    {
        "sample_ids.npy",
        "class_labels.npy",
        "exact_top1_correct.npy",
        "exact_top5_correct.npy",
        "shared_manifest.json",
        "checksums.json",
    }
)
_SHARD_FILES = frozenset(
    {
        "design_ids.npy",
        "approx_top1_correct.npy",
        "approx_top5_correct.npy",
        "delta.npy",
        "labels.csv",
        "design_ids.csv",
        "shard_manifest.json",
        "checksums.json",
    }
)
_FINAL_FILES = frozenset(
    {
        "design_ids.npy",
        "sample_ids.npy",
        "class_labels.npy",
        "exact_top1_correct.npy",
        "exact_top5_correct.npy",
        "approx_top1_correct.npy",
        "approx_top5_correct.npy",
        "delta.npy",
        "labels.csv",
        "design_ids.csv",
        "label_manifest.snapshot.json",
        "lut_asset_manifest.snapshot.json",
        "shard_merge_manifest.json",
        "run_manifest.json",
        "checksums.json",
    }
)


@dataclass(frozen=True)
class DesignShard:
    shard_index: int
    gpu_id: int
    start: int
    stop: int
    directory: Path

    @property
    def design_count(self) -> int:
        return self.stop - self.start


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _design_ids_sha256(design_ids: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for design_id in design_ids:
        digest.update(design_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelGenerationError(f"unable to read {context} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabelGenerationError(f"{context} must be a JSON object: {path}")
    return payload


def _read_csv(path: Path, *, context: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise LabelGenerationError(f"unable to read {context} {path}: {exc}") from exc


def _discard(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


@contextmanager
def _exclusive_checkpoint_lock(checkpoint_dir: Path) -> Iterator[None]:
    lock_path = checkpoint_dir.with_name(f".{checkpoint_dir.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LabelGenerationError(
                f"another sharded CNN generator holds the checkpoint lock: {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            lock_path.unlink(missing_ok=True)


def _verify_checksums(root: Path, *, expected_files: frozenset[str]) -> None:
    if not root.is_dir():
        raise LabelGenerationError(f"checkpoint directory is missing: {root}")
    checksums = _read_json(root / "checksums.json", context="checkpoint checksums")
    actual_files = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != set(expected_files):
        raise LabelGenerationError(
            f"checkpoint file inventory mismatch for {root}: {sorted(actual_files)}"
        )
    if set(checksums) != set(expected_files) - {"checksums.json"}:
        raise LabelGenerationError(f"checkpoint checksum inventory mismatch for {root}")
    for relative_path, expected in checksums.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise LabelGenerationError(f"invalid SHA-256 entry for {root / relative_path}")
        actual = _sha256(root / relative_path)
        if actual != expected:
            raise LabelGenerationError(
                f"checkpoint checksum mismatch for {root / relative_path}: {actual} != {expected}"
            )


def _build_shards(plan: LabelGenerationPlan, checkpoint_dir: Path) -> tuple[DesignShard, ...]:
    design_count = len(plan.design_ids)
    gpu_count = len(plan.gpu_ids)
    quotient, remainder = divmod(design_count, gpu_count)
    shards: list[DesignShard] = []
    range_start = 0
    shard_index = 0
    for gpu_position, gpu_id in enumerate(plan.gpu_ids):
        range_count = quotient + int(gpu_position < remainder)
        range_stop = range_start + range_count
        gpu_root = checkpoint_dir / f"gpu_{gpu_id:02d}"
        for start in range(range_start, range_stop, plan.block_designs):
            stop = min(start + plan.block_designs, range_stop)
            shards.append(
                DesignShard(
                    shard_index=shard_index,
                    gpu_id=gpu_id,
                    start=start,
                    stop=stop,
                    directory=gpu_root / f"rows_{start:06d}_{stop:06d}",
                )
            )
            shard_index += 1
        range_start = range_stop
    if range_start != design_count:
        raise LabelGenerationError("contiguous GPU shard partition did not cover every design")
    return tuple(shards)


def _checkpoint_contract(
    plan: LabelGenerationPlan,
    *,
    asset: S8S8LutAsset,
    sample_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": SHARD_CONTRACT_SCHEMA,
        "source_manifest": str(plan.manifest_path),
        "source_manifest_sha256": _sha256(plan.manifest_path),
        "model_sha256": plan.model_sha256,
        "dataset_sha256": plan.dataset_sha256,
        "preprocessing_id": plan.preprocessing_id,
        "node_names": list(plan.node_names),
        "lut_asset_manifest_sha256": _sha256(asset.root / "manifest.json"),
        "lut_asset_row_count": int(asset.manifest["row_count"]),
        "lut_domain": LUT_DOMAIN,
        "design_count": len(plan.design_ids),
        "design_ids_sha256": _design_ids_sha256(plan.design_ids),
        "sample_count": sample_count,
        "gpu_ids": list(plan.gpu_ids),
        "gpu_partition": "contiguous_balanced_v1",
        "batch_size": plan.batch_size,
        "shard_designs": plan.block_designs,
        "input_residency": plan.input_residency,
        "preallocate_outputs": plan.preallocate_outputs,
        "cudnn_conv_algo_search": plan.cudnn_conv_algo_search,
        "residual_abs_bound": plan.residual_abs_bound,
        "output_dir": str(plan.output_dir),
    }


def _initialize_checkpoint(
    checkpoint_dir: Path,
    *,
    contract: dict[str, Any],
    resume: bool,
) -> str:
    contract_path = checkpoint_dir / "checkpoint_contract.json"
    if checkpoint_dir.exists():
        if not checkpoint_dir.is_dir():
            raise LabelGenerationError(f"checkpoint path is not a directory: {checkpoint_dir}")
        if not contract_path.is_file():
            entries = list(checkpoint_dir.iterdir())
            if entries:
                raise LabelGenerationError(
                    f"checkpoint directory has no contract and is not empty: {checkpoint_dir}"
                )
        else:
            if not resume:
                raise LabelGenerationError(
                    f"checkpoint directory already exists; pass --resume to reuse it: {checkpoint_dir}"
                )
            existing = _read_json(contract_path, context="checkpoint contract")
            if existing != contract:
                raise LabelGenerationError(
                    "checkpoint contract differs from the requested CNN label run"
                )
    else:
        checkpoint_dir.mkdir(parents=True)
    if not contract_path.exists():
        _atomic_write_json(contract_path, contract)
    return _json_sha256(contract)


def _cleanup_staging(plan: LabelGenerationPlan, checkpoint_dir: Path) -> None:
    candidates = list(checkpoint_dir.glob(".shared.tmp.*"))
    candidates.extend(checkpoint_dir.glob("gpu_*/.rows_*.tmp.*"))
    candidates.extend(plan.output_dir.parent.glob(f".{plan.output_dir.name}.tmp.*"))
    for path in candidates:
        _discard(path)


def _run_exact_baselines(
    runtimes: tuple[tuple[int, LabelRuntime], ...],
    inputs: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def evaluate(entry: tuple[int, LabelRuntime]) -> tuple[int, np.ndarray, np.ndarray]:
        gpu_id, runtime = entry
        logits = runtime.predict_exact(inputs)
        top1, top5 = _correctness(logits, labels)
        return gpu_id, top1, top5

    if len(runtimes) == 1:
        results = [evaluate(runtimes[0])]
    else:
        with ThreadPoolExecutor(
            max_workers=len(runtimes),
            thread_name_prefix="cnn-shard-exact-gpu",
        ) as executor:
            results = list(executor.map(evaluate, runtimes))
    exact_top1 = results[0][1]
    exact_top5 = results[0][2]
    for gpu_id, candidate_top1, candidate_top5 in results[1:]:
        if not np.array_equal(candidate_top1, exact_top1) or not np.array_equal(
            candidate_top5, exact_top5
        ):
            raise LabelGenerationError(
                "multi-GPU exact top-k baselines disagree; refusing to create resumable "
                f"CNN shards from gpu_id={gpu_id}"
            )
    return exact_top1, exact_top5


def _write_shared_exact(
    checkpoint_dir: Path,
    *,
    contract_sha256: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    exact_top1: np.ndarray,
    exact_top5: np.ndarray,
) -> Path:
    destination = checkpoint_dir / "shared"
    if destination.exists():
        raise LabelGenerationError(f"shared exact checkpoint already exists: {destination}")
    staging = checkpoint_dir / f".shared.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir(parents=False)
    try:
        np.save(staging / "sample_ids.npy", sample_ids)
        np.save(staging / "class_labels.npy", labels)
        np.save(staging / "exact_top1_correct.npy", exact_top1)
        np.save(staging / "exact_top5_correct.npy", exact_top5)
        manifest = {
            "schema_version": SHARED_SCHEMA,
            "checkpoint_contract_sha256": contract_sha256,
            "sample_count": len(labels),
            "exact_top1_accuracy": float(np.mean(exact_top1)),
            "exact_top5_accuracy": float(np.mean(exact_top5)),
        }
        _atomic_write_json(staging / "shared_manifest.json", manifest)
        _write_checksums(staging)
        staging.replace(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_shared_exact(
    checkpoint_dir: Path,
    *,
    contract_sha256: str,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    exact_top1: np.ndarray | None = None,
    exact_top5: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    root = checkpoint_dir / "shared"
    _verify_checksums(root, expected_files=_SHARED_FILES)
    manifest = _read_json(root / "shared_manifest.json", context="shared exact manifest")
    if manifest.get("schema_version") != SHARED_SCHEMA or manifest.get(
        "checkpoint_contract_sha256"
    ) != contract_sha256:
        raise LabelGenerationError("shared exact checkpoint contract mismatch")
    stored_ids = np.load(root / "sample_ids.npy", allow_pickle=False).astype(str)
    stored_labels = np.load(root / "class_labels.npy", allow_pickle=False)
    stored_top1 = np.load(root / "exact_top1_correct.npy", allow_pickle=False)
    stored_top5 = np.load(root / "exact_top5_correct.npy", allow_pickle=False)
    if not np.array_equal(stored_ids, sample_ids) or not np.array_equal(stored_labels, labels):
        raise LabelGenerationError("shared exact checkpoint sample axis changed")
    if stored_top1.shape != labels.shape or stored_top1.dtype != np.bool_:
        raise LabelGenerationError("shared exact top-1 array has an invalid schema")
    if stored_top5.shape != labels.shape or stored_top5.dtype != np.bool_:
        raise LabelGenerationError("shared exact top-5 array has an invalid schema")
    if exact_top1 is not None and not np.array_equal(stored_top1, exact_top1):
        raise LabelGenerationError("resumed exact top-1 baseline differs from the checkpoint")
    if exact_top5 is not None and not np.array_equal(stored_top5, exact_top5):
        raise LabelGenerationError("resumed exact top-5 baseline differs from the checkpoint")
    return np.asarray(stored_top1), np.asarray(stored_top5)


def _validate_shard(
    spec: DesignShard,
    *,
    plan: LabelGenerationPlan,
    contract_sha256: str,
    sample_count: int,
) -> dict[str, Any] | None:
    if not spec.directory.exists():
        return None
    _verify_checksums(spec.directory, expected_files=_SHARD_FILES)
    manifest = _read_json(spec.directory / "shard_manifest.json", context="design shard manifest")
    expected_fields = {
        "schema_version": SHARD_SCHEMA,
        "checkpoint_contract_sha256": contract_sha256,
        "shard_index": spec.shard_index,
        "gpu_id": spec.gpu_id,
        "start": spec.start,
        "stop": spec.stop,
        "design_count": spec.design_count,
        "sample_count": sample_count,
    }
    mismatches = [
        f"{key}={manifest.get(key)!r}"
        for key, expected in expected_fields.items()
        if manifest.get(key) != expected
    ]
    if mismatches:
        raise LabelGenerationError(
            f"design shard contract mismatch for {spec.directory}: {', '.join(mismatches)}"
        )
    expected_ids = np.asarray(plan.design_ids[spec.start : spec.stop])
    actual_ids = np.load(spec.directory / "design_ids.npy", allow_pickle=False).astype(str)
    if not np.array_equal(actual_ids, expected_ids):
        raise LabelGenerationError(f"design IDs changed in shard {spec.directory}")
    arrays = {
        "approx_top1_correct.npy": (np.bool_, (spec.design_count, sample_count)),
        "approx_top5_correct.npy": (np.bool_, (spec.design_count, sample_count)),
        "delta.npy": (np.int32, (spec.design_count, 256, 256)),
    }
    for filename, (dtype, shape) in arrays.items():
        value = np.load(spec.directory / filename, mmap_mode="r", allow_pickle=False)
        if value.dtype != dtype or value.shape != shape:
            raise LabelGenerationError(
                f"invalid {filename} schema in shard {spec.directory}: {value.dtype} {value.shape}"
            )
    label_rows = _read_csv(spec.directory / "labels.csv", context="shard labels")
    design_rows = _read_csv(spec.directory / "design_ids.csv", context="shard design IDs")
    expected_list = expected_ids.astype(str).tolist()
    if [row.get("design_id") for row in label_rows] != expected_list:
        raise LabelGenerationError(f"label rows are misaligned in shard {spec.directory}")
    if [row.get("design_id") for row in design_rows] != expected_list:
        raise LabelGenerationError(f"design rows are misaligned in shard {spec.directory}")
    return manifest


def _write_progress(
    checkpoint_dir: Path,
    *,
    gpu_id: int,
    completed_shards: int,
    total_shards: int,
    completed_designs: int,
    total_designs: int,
    last_spec: DesignShard | None,
) -> None:
    payload = {
        "schema_version": SHARD_CONTRACT_SCHEMA,
        "gpu_id": gpu_id,
        "completed_shards": completed_shards,
        "total_shards": total_shards,
        "completed_designs": completed_designs,
        "total_designs": total_designs,
        "last_completed_range": (
            None if last_spec is None else [last_spec.start, last_spec.stop]
        ),
        "updated_at_unix": time.time(),
    }
    _atomic_write_json(checkpoint_dir / f"worker_gpu_{gpu_id:02d}.progress.json", payload)


def _write_design_shard(
    spec: DesignShard,
    *,
    plan: LabelGenerationPlan,
    asset: S8S8LutAsset,
    runtime: LabelRuntime,
    inputs: np.ndarray,
    labels: np.ndarray,
    exact_top1: np.ndarray,
    exact_top5: np.ndarray,
    contract_sha256: str,
    asset_sha256: str,
) -> dict[str, Any]:
    spec.directory.parent.mkdir(parents=True, exist_ok=True)
    staging = spec.directory.with_name(
        f".{spec.directory.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=False)
    sample_count = len(labels)
    exact_top1_accuracy = float(np.mean(exact_top1))
    exact_top5_accuracy = float(np.mean(exact_top5))
    approx_top1 = np.empty((spec.design_count, sample_count), dtype=np.bool_)
    approx_top5 = np.empty((spec.design_count, sample_count), dtype=np.bool_)
    delta_rows = np.empty((spec.design_count, 256, 256), dtype=np.int32)
    label_rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for local_index, design_index in enumerate(range(spec.start, spec.stop)):
            design_id = plan.design_ids[design_index]
            delta = asset.delta_for(design_id)
            behavior_sha256 = behavior_sha256_rows(delta[None, :, :])[0]
            logits = runtime.predict_approx(inputs, delta)
            top1, top5 = _correctness(logits, labels)
            top1_accuracy = float(np.mean(top1))
            top5_accuracy = float(np.mean(top5))
            delta_rows[local_index] = delta
            approx_top1[local_index] = top1
            approx_top5[local_index] = top5
            label_rows.append(
                {
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
            )
            design_rows.append(
                {
                    "design_id": design_id,
                    "behavior_sha256": behavior_sha256,
                    "asset_sha256": asset_sha256,
                }
            )
        elapsed = time.perf_counter() - started
        np.save(staging / "design_ids.npy", np.asarray(plan.design_ids[spec.start : spec.stop]))
        np.save(staging / "approx_top1_correct.npy", approx_top1)
        np.save(staging / "approx_top5_correct.npy", approx_top5)
        np.save(staging / "delta.npy", delta_rows)
        _write_csv(staging / "labels.csv", label_rows)
        _write_csv(staging / "design_ids.csv", design_rows)
        runtime_metadata = getattr(runtime, "execution_metadata", None)
        if runtime_metadata is not None and not isinstance(runtime_metadata, dict):
            raise LabelGenerationError("label runtime execution_metadata must be a dictionary")
        manifest = {
            "schema_version": SHARD_SCHEMA,
            "checkpoint_contract_sha256": contract_sha256,
            "shard_index": spec.shard_index,
            "gpu_id": spec.gpu_id,
            "start": spec.start,
            "stop": spec.stop,
            "design_count": spec.design_count,
            "sample_count": sample_count,
            "elapsed_seconds": elapsed,
            "designs_per_second": spec.design_count / elapsed if elapsed > 0 else 0.0,
            "runtime_metadata": runtime_metadata
            or {
                "requested_device": "cuda",
                "runtime_factory": "injected_test_boundary",
                "logical_gpu_id": spec.gpu_id,
            },
            "completed_at_unix": time.time(),
        }
        _atomic_write_json(staging / "shard_manifest.json", manifest)
        _write_checksums(staging)
        staging.replace(spec.directory)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _semantic_shard_rows(
    spec: DesignShard,
    *,
    plan: LabelGenerationPlan,
    exact_top1: np.ndarray,
    exact_top5: np.ndarray,
    asset_sha256: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    label_rows = _read_csv(spec.directory / "labels.csv", context="shard labels")
    design_rows = _read_csv(spec.directory / "design_ids.csv", context="shard design IDs")
    approx_top1 = np.load(
        spec.directory / "approx_top1_correct.npy", mmap_mode="r", allow_pickle=False
    )
    approx_top5 = np.load(
        spec.directory / "approx_top5_correct.npy", mmap_mode="r", allow_pickle=False
    )
    delta = np.load(spec.directory / "delta.npy", mmap_mode="r", allow_pickle=False)
    behavior_hashes = behavior_sha256_rows(np.asarray(delta))
    exact_top1_accuracy = float(np.mean(exact_top1))
    exact_top5_accuracy = float(np.mean(exact_top5))

    def require_float(row: dict[str, str], key: str, expected: float) -> None:
        try:
            actual = float(row[key])
        except (KeyError, ValueError) as exc:
            raise LabelGenerationError(
                f"invalid {key} in shard {spec.directory}"
            ) from exc
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise LabelGenerationError(
                f"semantic {key} mismatch in shard {spec.directory}: {actual} != {expected}"
            )

    for local_index, design_index in enumerate(range(spec.start, spec.stop)):
        design_id = plan.design_ids[design_index]
        label_row = label_rows[local_index]
        design_row = design_rows[local_index]
        behavior_hash = str(behavior_hashes[local_index])
        if label_row.get("design_id") != design_id or design_row.get("design_id") != design_id:
            raise LabelGenerationError(f"design order changed in shard {spec.directory}")
        for row in (label_row, design_row):
            if row.get("behavior_sha256") != behavior_hash:
                raise LabelGenerationError(f"behavior hash mismatch in shard {spec.directory}")
            if row.get("asset_sha256") != asset_sha256:
                raise LabelGenerationError(f"LUT asset binding mismatch in shard {spec.directory}")
        if int(label_row.get("sample_count", -1)) != exact_top1.shape[0]:
            raise LabelGenerationError(f"sample count mismatch in shard {spec.directory}")
        top1_accuracy = float(np.mean(approx_top1[local_index]))
        top5_accuracy = float(np.mean(approx_top5[local_index]))
        require_float(label_row, "exact_top1_accuracy", exact_top1_accuracy)
        require_float(label_row, "exact_top5_accuracy", exact_top5_accuracy)
        require_float(label_row, "approx_top1_accuracy", top1_accuracy)
        require_float(label_row, "approx_top5_accuracy", top5_accuracy)
        require_float(label_row, "top1_drop", exact_top1_accuracy - top1_accuracy)
        require_float(label_row, "top5_drop", exact_top5_accuracy - top5_accuracy)
    return label_rows, design_rows


def _merge_shards(
    plan: LabelGenerationPlan,
    *,
    checkpoint_dir: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    specs: tuple[DesignShard, ...],
    asset: S8S8LutAsset,
    samples: tuple[np.ndarray, np.ndarray, np.ndarray],
    remove_checkpoints_after_merge: bool,
    resume_requested: bool,
) -> LabelGenerationOutcome:
    _inputs, labels, sample_ids = samples
    exact_top1, exact_top5 = _validate_shared_exact(
        checkpoint_dir,
        contract_sha256=contract_sha256,
        sample_ids=sample_ids,
        labels=labels,
    )
    staging = plan.output_dir.with_name(
        f".{plan.output_dir.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True, exist_ok=False)
    design_count = len(plan.design_ids)
    sample_count = len(labels)
    asset_manifest_path = asset.root / "manifest.json"
    asset_sha256 = _sha256(asset_manifest_path)
    try:
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
        delta_output = np.lib.format.open_memmap(
            staging / "delta.npy",
            mode="w+",
            dtype=np.int32,
            shape=(design_count, 256, 256),
        )
        all_label_rows: list[dict[str, str]] = []
        all_design_rows: list[dict[str, str]] = []
        shard_records: list[dict[str, object]] = []
        runtime_metadata_by_gpu: dict[str, dict[str, object]] = {}
        timing_by_gpu: dict[int, dict[str, float | int]] = {
            gpu_id: {"design_count": 0, "elapsed_seconds": 0.0}
            for gpu_id in plan.gpu_ids
        }
        expected_start = 0
        for spec in specs:
            if spec.start != expected_start:
                raise LabelGenerationError("design shards are not contiguous during merge")
            manifest = _validate_shard(
                spec,
                plan=plan,
                contract_sha256=contract_sha256,
                sample_count=sample_count,
            )
            if manifest is None:
                raise LabelGenerationError(f"required design shard is missing: {spec.directory}")
            label_rows, design_rows = _semantic_shard_rows(
                spec,
                plan=plan,
                exact_top1=exact_top1,
                exact_top5=exact_top5,
                asset_sha256=asset_sha256,
            )
            approx_top1[spec.start : spec.stop] = np.load(
                spec.directory / "approx_top1_correct.npy", mmap_mode="r", allow_pickle=False
            )
            approx_top5[spec.start : spec.stop] = np.load(
                spec.directory / "approx_top5_correct.npy", mmap_mode="r", allow_pickle=False
            )
            delta_output[spec.start : spec.stop] = np.load(
                spec.directory / "delta.npy", mmap_mode="r", allow_pickle=False
            )
            all_label_rows.extend(label_rows)
            all_design_rows.extend(design_rows)
            elapsed = float(manifest["elapsed_seconds"])
            timing_by_gpu[spec.gpu_id]["design_count"] = int(
                timing_by_gpu[spec.gpu_id]["design_count"]
            ) + spec.design_count
            timing_by_gpu[spec.gpu_id]["elapsed_seconds"] = float(
                timing_by_gpu[spec.gpu_id]["elapsed_seconds"]
            ) + elapsed
            runtime_metadata = manifest.get("runtime_metadata")
            if not isinstance(runtime_metadata, dict):
                raise LabelGenerationError(f"runtime metadata is invalid in {spec.directory}")
            runtime_metadata_by_gpu[str(spec.gpu_id)] = runtime_metadata
            shard_records.append(
                {
                    "shard_index": spec.shard_index,
                    "gpu_id": spec.gpu_id,
                    "start": spec.start,
                    "stop": spec.stop,
                    "relative_path": str(spec.directory.relative_to(checkpoint_dir)),
                    "shard_manifest_sha256": _sha256(spec.directory / "shard_manifest.json"),
                    "checksums_sha256": _sha256(spec.directory / "checksums.json"),
                }
            )
            expected_start = spec.stop
        if expected_start != design_count:
            raise LabelGenerationError("merged design shards do not cover the requested population")
        approx_top1.flush()
        approx_top5.flush()
        delta_output.flush()
        del approx_top1, approx_top5, delta_output
        _write_csv(staging / "labels.csv", all_label_rows)
        _write_csv(staging / "design_ids.csv", all_design_rows)
        shutil.copy2(plan.manifest_path, staging / "label_manifest.snapshot.json")
        shutil.copy2(asset_manifest_path, staging / "lut_asset_manifest.snapshot.json")
        merge_manifest = {
            "schema_version": MERGE_SCHEMA,
            "checkpoint_contract": contract,
            "checkpoint_contract_sha256": contract_sha256,
            "partition": "contiguous_balanced_v1",
            "shard_count": len(specs),
            "design_count": design_count,
            "sample_count": sample_count,
            "shards": shard_records,
            "semantic_validation": {
                "checksums_verified": True,
                "contiguous_nonoverlapping_ranges_verified": True,
                "design_order_verified": True,
                "sample_axis_verified": True,
                "exact_baseline_verified": True,
                "behavior_hashes_recomputed": True,
                "accuracy_drops_recomputed": True,
            },
        }
        _atomic_write_json(staging / "shard_merge_manifest.json", merge_manifest)
        worker_timings: dict[str, dict[str, float | int]] = {}
        for gpu_id, timing in timing_by_gpu.items():
            count = int(timing["design_count"])
            elapsed = float(timing["elapsed_seconds"])
            worker_timings[str(gpu_id)] = {
                "design_count": count,
                "elapsed_seconds": elapsed,
                "designs_per_second": count / elapsed if elapsed > 0 else 0.0,
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
            "effective_block_designs": plan.block_designs,
            "residual_abs_bound": plan.residual_abs_bound,
            "execution": runtime_metadata_by_gpu[str(plan.gpu_id)],
            "execution_by_gpu": runtime_metadata_by_gpu,
            "worker_timings": worker_timings,
            "performance_claim_allowed": True,
            "per_image_alignment": {
                "design_axis": "design_ids.npy",
                "sample_axis": "sample_ids.npy",
                "top1_correct": "approx_top1_correct.npy",
                "top5_correct": "approx_top5_correct.npy",
            },
            "canonical_dataset": {
                "enabled": False,
                "domain": "s8_numeric",
                "build_manifest": None,
                "train_manifest": None,
            },
            "sharded_generation": {
                "schema_version": SHARD_CONTRACT_SCHEMA,
                "checkpoint_contract_sha256": contract_sha256,
                "partition": "contiguous_balanced_v1",
                "shard_count": len(specs),
                "shard_designs": plan.block_designs,
                "resume_requested": resume_requested,
                "checkpoints_removed_after_merge": remove_checkpoints_after_merge,
                "merge_manifest": "shard_merge_manifest.json",
            },
        }
        _atomic_write_json(staging / "run_manifest.json", run_manifest)
        _write_checksums(staging)
        _verify_checksums(staging, expected_files=_FINAL_FILES)
        staging.replace(plan.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return LabelGenerationOutcome(
        output_dir=plan.output_dir,
        design_count=design_count,
        sample_count=sample_count,
    )


def _validate_final_output(
    plan: LabelGenerationPlan,
    *,
    contract_sha256: str,
    sample_count: int,
) -> LabelGenerationOutcome:
    _verify_checksums(plan.output_dir, expected_files=_FINAL_FILES)
    run_manifest = _read_json(plan.output_dir / "run_manifest.json", context="merged run manifest")
    sharded = run_manifest.get("sharded_generation")
    if not isinstance(sharded, dict) or sharded.get(
        "checkpoint_contract_sha256"
    ) != contract_sha256:
        raise LabelGenerationError("existing merged output does not match the checkpoint contract")
    if run_manifest.get("design_count") != len(plan.design_ids) or run_manifest.get(
        "sample_count"
    ) != sample_count:
        raise LabelGenerationError("existing merged output has incompatible dimensions")
    design_ids = np.load(plan.output_dir / "design_ids.npy", allow_pickle=False).astype(str)
    if design_ids.tolist() != list(plan.design_ids):
        raise LabelGenerationError("existing merged output design order changed")
    expected_arrays = {
        "sample_ids.npy": (sample_count,),
        "class_labels.npy": (sample_count,),
        "exact_top1_correct.npy": (sample_count,),
        "exact_top5_correct.npy": (sample_count,),
        "approx_top1_correct.npy": (len(plan.design_ids), sample_count),
        "approx_top5_correct.npy": (len(plan.design_ids), sample_count),
        "delta.npy": (len(plan.design_ids), 256, 256),
    }
    for filename, shape in expected_arrays.items():
        array = np.load(plan.output_dir / filename, mmap_mode="r", allow_pickle=False)
        if array.shape != shape:
            raise LabelGenerationError(f"existing merged {filename} has shape {array.shape}")
    return LabelGenerationOutcome(
        output_dir=plan.output_dir,
        design_count=len(plan.design_ids),
        sample_count=sample_count,
    )


def generate_sharded_labels_from_manifest(
    manifest_path: Path,
    *,
    checkpoint_dir: Path,
    resume: bool = False,
    remove_checkpoints_after_merge: bool = False,
    runtime_factory: RuntimeFactory | None = None,
) -> LabelGenerationOutcome:
    """Generate real CNN labels as resumable per-GPU contiguous design shards."""

    plan = load_label_generation_plan(manifest_path, allow_existing_output=True)
    if plan.canonical_profile_metadata is not None:
        raise LabelGenerationError(
            "sharded real-label generation does not materialize LightGBM/canonical features"
        )
    checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
    if (
        checkpoint_root == plan.output_dir
        or checkpoint_root in plan.output_dir.parents
        or plan.output_dir in checkpoint_root.parents
    ):
        raise LabelGenerationError("checkpoint_dir and output_dir must be separate sibling trees")
    try:
        with _exclusive_checkpoint_lock(checkpoint_root):
            samples = _load_samples(plan)
            inputs, labels, sample_ids = samples
            asset = load_s8s8_lut_asset(plan.asset_root, verify_checksums=True)
            design_ids = asset.require_design_ids(plan.design_ids)
            if design_ids != plan.design_ids:
                raise LabelGenerationError("design order changed during LUT validation")
            actual_bound = asset.max_abs_delta(plan.design_ids)
            if plan.residual_abs_bound is not None and actual_bound > plan.residual_abs_bound:
                raise LabelGenerationError(
                    f"selected residual bound {actual_bound} exceeds configured "
                    f"execution.residual_abs_bound={plan.residual_abs_bound}"
                )
            plan = replace(plan, residual_abs_bound=actual_bound)
            contract = _checkpoint_contract(plan, asset=asset, sample_count=len(labels))
            contract_sha256 = _json_sha256(contract)
            if plan.output_dir.exists():
                outcome = _validate_final_output(
                    plan,
                    contract_sha256=contract_sha256,
                    sample_count=len(labels),
                )
                if remove_checkpoints_after_merge and checkpoint_root.exists():
                    _initialize_checkpoint(
                        checkpoint_root,
                        contract=contract,
                        resume=True,
                    )
                    _discard(checkpoint_root)
                print(f"[cnn-shards] reused validated merged output {plan.output_dir}", flush=True)
                return outcome

            contract_sha256 = _initialize_checkpoint(
                checkpoint_root,
                contract=contract,
                resume=resume,
            )
            _cleanup_staging(plan, checkpoint_root)
            specs = _build_shards(plan, checkpoint_root)
            completed: dict[int, dict[str, Any]] = {}
            for spec in specs:
                manifest = _validate_shard(
                    spec,
                    plan=plan,
                    contract_sha256=contract_sha256,
                    sample_count=len(labels),
                )
                if manifest is not None:
                    completed[spec.shard_index] = manifest
            shared_exists = (checkpoint_root / "shared").exists()
            if completed and not shared_exists:
                raise LabelGenerationError("design shards exist without the shared exact checkpoint")
            if shared_exists:
                _validate_shared_exact(
                    checkpoint_root,
                    contract_sha256=contract_sha256,
                    sample_ids=sample_ids,
                    labels=labels,
                )
            pending = tuple(spec for spec in specs if spec.shard_index not in completed)
            print(
                f"[cnn-shards] validated={len(completed)} pending={len(pending)} "
                f"total={len(specs)} designs={len(plan.design_ids)}",
                flush=True,
            )
            if pending:
                runtimes: list[tuple[int, LabelRuntime]] = []
                try:
                    factory = runtime_factory or _default_runtime_factory
                    for gpu_id in plan.gpu_ids:
                        runtimes.append((gpu_id, factory(replace(plan, gpu_id=gpu_id))))
                    exact_top1, exact_top5 = _run_exact_baselines(
                        tuple(runtimes), inputs, labels
                    )
                    if shared_exists:
                        _validate_shared_exact(
                            checkpoint_root,
                            contract_sha256=contract_sha256,
                            sample_ids=sample_ids,
                            labels=labels,
                            exact_top1=exact_top1,
                            exact_top5=exact_top5,
                        )
                    else:
                        _write_shared_exact(
                            checkpoint_root,
                            contract_sha256=contract_sha256,
                            sample_ids=sample_ids,
                            labels=labels,
                            exact_top1=exact_top1,
                            exact_top5=exact_top5,
                        )
                    asset_sha256 = _sha256(asset.root / "manifest.json")
                    pending_by_gpu = {
                        gpu_id: [spec for spec in pending if spec.gpu_id == gpu_id]
                        for gpu_id in plan.gpu_ids
                    }
                    all_by_gpu = {
                        gpu_id: [spec for spec in specs if spec.gpu_id == gpu_id]
                        for gpu_id in plan.gpu_ids
                    }

                    def evaluate_gpu(entry: tuple[int, LabelRuntime]) -> None:
                        gpu_id, runtime = entry
                        gpu_specs = pending_by_gpu[gpu_id]
                        already_completed = len(all_by_gpu[gpu_id]) - len(gpu_specs)
                        completed_designs = sum(
                            spec.design_count
                            for spec in all_by_gpu[gpu_id]
                            if spec.shard_index in completed
                        )
                        for local_index, spec in enumerate(gpu_specs, start=1):
                            _write_design_shard(
                                spec,
                                plan=plan,
                                asset=asset,
                                runtime=runtime,
                                inputs=inputs,
                                labels=labels,
                                exact_top1=exact_top1,
                                exact_top5=exact_top5,
                                contract_sha256=contract_sha256,
                                asset_sha256=asset_sha256,
                            )
                            completed_designs += spec.design_count
                            _write_progress(
                                checkpoint_root,
                                gpu_id=gpu_id,
                                completed_shards=already_completed + local_index,
                                total_shards=len(all_by_gpu[gpu_id]),
                                completed_designs=completed_designs,
                                total_designs=sum(s.design_count for s in all_by_gpu[gpu_id]),
                                last_spec=spec,
                            )
                            print(
                                f"[cnn-shards] gpu={gpu_id} completed rows "
                                f"[{spec.start},{spec.stop}) "
                                f"shards={already_completed + local_index}/{len(all_by_gpu[gpu_id])}",
                                flush=True,
                            )

                    if len(runtimes) == 1:
                        evaluate_gpu(runtimes[0])
                    else:
                        with ThreadPoolExecutor(
                            max_workers=len(runtimes),
                            thread_name_prefix="cnn-shard-approx-gpu",
                        ) as executor:
                            futures = [executor.submit(evaluate_gpu, entry) for entry in runtimes]
                            for future in futures:
                                future.result()
                finally:
                    for _, runtime in runtimes:
                        runtime.close()

            for spec in specs:
                if _validate_shard(
                    spec,
                    plan=plan,
                    contract_sha256=contract_sha256,
                    sample_count=len(labels),
                ) is None:
                    raise LabelGenerationError(f"design shard was not completed: {spec.directory}")
            print("[cnn-shards] all shards verified; starting semantic merge", flush=True)
            outcome = _merge_shards(
                plan,
                checkpoint_dir=checkpoint_root,
                contract=contract,
                contract_sha256=contract_sha256,
                specs=specs,
                asset=asset,
                samples=samples,
                remove_checkpoints_after_merge=remove_checkpoints_after_merge,
                resume_requested=resume,
            )
            _validate_final_output(
                plan,
                contract_sha256=contract_sha256,
                sample_count=len(labels),
            )
            if remove_checkpoints_after_merge:
                _discard(checkpoint_root)
            print(f"[cnn-shards] merged output published at {plan.output_dir}", flush=True)
            return outcome
    except LabelGenerationError:
        raise
    except Exception as exc:
        raise LabelGenerationError(
            f"sharded CNN real-label generation failed: {type(exc).__name__}: {exc}"
        ) from exc


__all__ = [
    "DesignShard",
    "generate_sharded_labels_from_manifest",
]
