#!/usr/bin/env python
"""Resumable multi-GPU bulk evaluation for a Transformer S8S8 LUT population.

This is deliberately separate from ``run_swin_s8s8_uniform_injection.py``.
That script is a forensic, one-design runner: it creates a fresh CUDA session,
validates the complete 4.88 GiB LUT asset, recomputes the exact baseline, and
writes per-image artifacts every time.  Those are useful properties for a
small investigation and the wrong execution shape for 20,000 designs.

The default throughput protocol keeps one dynamic-B=16 CUDA EP session and one
memory-mapped LUT asset alive per GPU worker.  It computes the exact baseline
with the *same* dynamic model, ordered batches, and CUDA session profile as
the approximate path.  This makes the reported drop internally coherent even
though dynamic S8S8 B>1 is not bit-identical to strict B=1.

The default adapter is a persistent I/O-binding runner.  An alternative can be
selected with ``--runner-factory`` without changing this orchestration module,
provided it implements the small protocol below:

    runner.update_lut(residual_i32_256x256)
    logits = runner.run_all()  # float32 [sample_count, 1000]
    runner.close()

``--execution-profile strict_b1`` retains the fixed-B=1 model as an optional
audit protocol.  It is not the default 20k throughput path.
"""

from __future__ import annotations

from aaae.paths import OUTPUT_ROOT

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import onnx
import pandas as pd


from aaae.inference_runtime.transformer.assignment_runtime import build_uniform_assignment_runtime_pool
from aaae.inference_runtime.transformer.ort_session import create_cuda_session
from aaae.inference_runtime.transformer.s8s8_truth_lut_asset import (
    S8S8_LUT_DOMAIN,
    S8S8TruthLutAsset,
    load_s8s8_sign_magnitude_truth_lut_asset,
)
from aaae.inference_runtime.transformer.dataset_generation.profiles import DATASET_PROFILES, DEFAULT_MODEL_KEY, get_dataset_profile
from aaae.inference_runtime.transformer.dataset_generation.matmul_scope import select_weight_stationary_block_linears


from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT
from aaae.paths import EA2_ROOT as EA2_ROOT
END_TO_END_ROOT = EA2_ROOT / "end_to_end_prediction"
AI_MODELS_ROOT = EA2_ROOT / "ai_models"
AI_MODELS_SRC = AI_MODELS_ROOT / "src"


from aaae.inference_runtime.preprocess import prepare_classification_input  # noqa: E402
from aaae.inference_runtime.imagenet import imagenet_gt_indices  # noqa: E402


BULK_SCHEMA_VERSION = "transformer_s8s8_linear48_bulk_dynamic_b16_v7"
DEFAULT_SOURCE_DATA_DIR = (
    END_TO_END_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_strict50_true1000_raw500_v2"
    / "data"
)
DEFAULT_S8_ASSET_ROOT = (
    END_TO_END_ROOT / "datasets" / "multiplier" / "lowonly20k_s8s8_sign_magnitude_v1"
)
DEFAULT_NODE_REGISTRY = (
    END_TO_END_ROOT / "resources" / "data" / "swin_tiny_s8s8" / "node_registry" / "node_registry.json"
)
DEFAULT_THROUGHPUT_MODEL_PATH = (
    AI_MODELS_ROOT
    / "onnx_models"
    / "classification"
    / "swin_tiny_patch4_window7_224_imagenet1k_int8_dynamic.onnx"
)
DEFAULT_STRICT_B1_MODEL_PATH = (
    AI_MODELS_ROOT / "onnx_models" / "classification" / "swin_tiny_patch4_window7_224_imagenet1k_int8.onnx"
)
DEFAULT_IMAGE_ROOT = AI_MODELS_ROOT / "dataset" / "imagenet-1k"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "accuracy" / "swin_tiny_int8_s8s8_dynamic_b16_bulk_20k_20260717"
DEFAULT_BLOCK_SIZE = 16
DEFAULT_GPU_IDS = (0, 1, 2)
DEFAULT_RUNNER_FACTORY = "persistent_iobinding"
DEFAULT_BATCH_SIZE = 16
# The immutable 20k v1 S8S8 asset was scanned in full before enabling the
# narrowed accumulator. Its residual range is [-856, +856]. With Swin's
# maximum K=3072, the largest possible residual sum is 2,629,632, safely
# below INT32_MAX. Every LUT is checked against this contract before upload.
DEFAULT_S8_RESIDUAL_ABS_BOUND = 856

# Measured on the accepted fixed-B=1 72-node row-644 run.  It is a conservative
# fallback reference for --dry-run until this dynamic-batch protocol has a pilot
# measurement; workers persist their actual dynamic throughput in block files.
MEASURED_B1_APPROX_SECONDS_PER_DESIGN = 28.747155760996975
MEASURED_B1_EXACT_SECONDS_PER_DESIGN = 15.282151986961253
# Row 644, 1,000 ImageNet images, dynamic B=16, SM89 custom op, staged
# I/O-binding inputs, and a shared GPU 2 on 2026-07-17. Device-resident input
# caching was within measurement noise of staging because the 72 residual
# kernels dominate. Use this for planning only; workers write their observed
# per-design timings to block records.
MEASURED_DYNAMIC_B16_APPROX_SECONDS_PER_DESIGN = 5.40941605810076


class PersistentBulkRunner(Protocol):
    """Minimal adapter boundary for the bulk scheduler."""

    def update_lut(self, residual: np.ndarray) -> None:
        """Install one contiguous ``int32[256,256]`` S8S8 residual LUT."""

    def run_all(self) -> np.ndarray:
        """Return approximate logits for all frozen inputs in their fixed batches."""

    def close(self) -> None:
        """Release device-side resources."""


@dataclass(frozen=True)
class BlockSpec:
    block_id: int
    start: int
    stop: int
    gpu_id: int


@dataclass(frozen=True)
class WorkPlan:
    gpu_ids: tuple[int, ...]
    block_size: int
    blocks: tuple[BlockSpec, ...]
    selected_rows: pd.DataFrame


@dataclass(frozen=True)
class ExactReference:
    """Same-batch clean predictions cached once per worker for metric scoring."""

    top1: np.ndarray
    top5: np.ndarray
    top1_correct: np.ndarray
    top5_correct: np.ndarray
    top1_correct_count: int
    top5_correct_count: int


class CompatPoolRunner:
    """Persistent-session fallback until an I/O-binding factory is selected.

    It keeps the rewritten model/session alive and never reopens the LUT asset,
    but still feeds the LUT through the current public runtime API for every
    prepared image batch.  This adapter is correct, simple, and intentionally
    not a claim that it is the final high-throughput path.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        node_names: Sequence[str],
        inputs: np.ndarray,
        gpu_id: int,
        lut_domain: str,
        batch_size: int,
        s8_residual_abs_bound: int,
        session_profile: str,
        input_residency: str,
        static_shape_reference_path: Path | None = None,
    ) -> None:
        self._inputs = np.asarray(inputs, dtype=np.float32)
        if self._inputs.ndim != 4 or self._inputs.shape[0] <= 0:
            raise ValueError(f"input cache must be [N,C,H,W], got {self._inputs.shape}")
        self._pool = build_uniform_assignment_runtime_pool(
            Path(model_path),
            selected_node_names=[str(name) for name in node_names],
            shared_lut_input_name="uniform_s8s8_residual",
            device_id=int(gpu_id),
            runtime_count=1,
            batch_fallback_parallelism=1,
            lut_domain=str(lut_domain),
            s8_residual_abs_bound=int(s8_residual_abs_bound),
            session_profile=str(session_profile),
            static_shape_reference_path=static_shape_reference_path,
        )
        self._batch_size = int(batch_size)
        if self._batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.input_residency = "host"
        self._residual: np.ndarray | None = None

    def update_lut(self, residual: np.ndarray) -> None:
        value = np.asarray(residual, dtype=np.int32)
        if value.shape != (256, 256):
            raise ValueError(f"S8S8 residual must be int32[256,256], got {value.shape}")
        self._residual = np.ascontiguousarray(value)

    def run_all(self) -> np.ndarray:
        if self._residual is None:
            raise RuntimeError("update_lut() must be called before run_all()")
        started = time.perf_counter()
        logits = np.empty((self._inputs.shape[0], 1000), dtype=np.float32)
        for index in range(0, self._inputs.shape[0], self._batch_size):
            stop = min(index + self._batch_size, self._inputs.shape[0])
            output = np.asarray(
                self._pool.run_uniform(
                    np.asarray(self._inputs[index:stop], dtype=np.float32),
                    approx_delta=self._residual,
                )[0],
                dtype=np.float32,
            )
            if output.shape != (stop - index, 1000):
                raise ValueError(f"compat runtime returned {output.shape}, expected {(stop - index, 1000)}")
            logits[index:stop] = output
        self.last_run_elapsed_seconds = float(time.perf_counter() - started)
        return logits

    def close(self) -> None:
        self._pool.close()


class PersistentIoBindingRunner:
    """Default persistent transport implementation for dynamic throughput.

    ``PersistentLutCudaIoBindingRunner`` owns one stable device LUT buffer and
    uploads the frozen image batches once.  The surrounding pool remains
    responsible for compiling/owning the rewritten S8S8 graph and session.
    """

    def __init__(
        self,
        *,
        model_path: Path,
        node_names: Sequence[str],
        inputs: np.ndarray,
        gpu_id: int,
        lut_domain: str,
        batch_size: int,
        s8_residual_abs_bound: int,
        session_profile: str,
        input_residency: str,
        static_shape_reference_path: Path | None = None,
    ) -> None:
        from aaae.inference_runtime.transformer.lut_cuda_iobinding import PersistentLutCudaIoBindingRunner

        values = np.asarray(inputs, dtype=np.float32)
        if values.ndim != 4 or values.shape[0] <= 0:
            raise ValueError(f"input cache must be [N,C,H,W], got {values.shape}")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        requested_residency = str(input_residency).strip().lower()
        if requested_residency not in {"auto", "device", "staged"}:
            raise ValueError("input_residency must be 'auto', 'device', or 'staged'")
        batches = [
            np.ascontiguousarray(values[start : min(start + int(batch_size), values.shape[0])], dtype=np.float32)
            for start in range(0, values.shape[0], int(batch_size))
        ]

        def create_binding(residency: str):
            pool = build_uniform_assignment_runtime_pool(
                Path(model_path),
                selected_node_names=[str(name) for name in node_names],
                shared_lut_input_name="uniform_s8s8_residual",
                device_id=int(gpu_id),
                runtime_count=1,
                batch_fallback_parallelism=1,
                lut_domain=str(lut_domain),
                s8_residual_abs_bound=int(s8_residual_abs_bound),
                session_profile=str(session_profile),
                static_shape_reference_path=static_shape_reference_path,
            )
            binding = PersistentLutCudaIoBindingRunner(
                pool.runtimes[0],
                lut_input_name=pool.shared_lut_input_name,
                device_id=int(gpu_id),
                preallocate_outputs=True,
                session_profile=str(session_profile),
                input_residency=residency,
            )
            try:
                binding.prepare_batches(batches)
            except Exception:
                binding.close()
                pool.close()
                raise
            return pool, binding

        if requested_residency == "auto":
            try:
                self._pool, self._binding = create_binding("device")
            except RuntimeError as exc:
                if not self._is_cuda_allocation_error(exc):
                    raise
                # Drop the failed device-resident session before allocating
                # the small staged alternative on a memory-constrained GPU.
                gc.collect()
                self._pool, self._binding = create_binding("staged")
        else:
            self._pool, self._binding = create_binding(requested_residency)
        self.input_residency = self._binding.metadata.input_residency
        self._sample_count = int(values.shape[0])
        self.last_run_elapsed_seconds = 0.0

    @staticmethod
    def _is_cuda_allocation_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        markers = (
            "alloc_failed",
            "out of memory",
            "cudaerror memory allocation",
            "cuda_error_out_of_memory",
            "cublas_status_alloc_failed",
        )
        return any(marker in message for marker in markers)

    def update_lut(self, residual: np.ndarray) -> None:
        self._binding.update_lut(np.asarray(residual, dtype=np.int32))

    def run_all(self) -> np.ndarray:
        result = self._binding.run_all(return_logits=True)
        if result.logits is None or result.logits.shape != (self._sample_count, 1000):
            shape = None if result.logits is None else result.logits.shape
            raise ValueError(f"I/O-binding runner returned logits {shape}, expected {(self._sample_count, 1000)}")
        self.last_run_elapsed_seconds = float(result.elapsed_seconds)
        return np.asarray(result.logits, dtype=np.float32)

    def close(self) -> None:
        try:
            self._binding.close()
        finally:
            self._pool.close()


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def _topk(logits: np.ndarray, k: int = 5) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < k:
        raise ValueError(f"expected logits [N,C] with C >= {k}, got {values.shape}")
    partial = np.argpartition(values, kth=values.shape[1] - k, axis=1)[:, -k:]
    partial_scores = np.take_along_axis(values, partial, axis=1)
    order = np.argsort(partial_scores, axis=1)[:, ::-1]
    return np.take_along_axis(partial, order, axis=1).astype(np.int64, copy=False)


def _parse_gpu_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not ids:
        raise ValueError("--gpu-ids must contain at least one CUDA device ID")
    if any(item < 0 for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("--gpu-ids must be unique non-negative CUDA device IDs")
    return ids


def _isolated_worker_environment(
    *,
    physical_gpu_id: int,
    base_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Expose exactly one assigned CUDA ordinal to a bulk worker.

    The CLI GPU IDs are CUDA ordinals, matching the historical direct
    ``device_id`` behavior. Masking the child process to its assigned ordinal
    makes that physical GPU appear as local ``cuda:0``. It prevents a
    long-running worker from creating or selecting a context on another card
    while retaining the physical ID in block and baseline metadata.
    """

    if int(physical_gpu_id) < 0:
        raise ValueError("physical_gpu_id must be non-negative")
    environment = dict(os.environ if base_environment is None else base_environment)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu_id))
    return environment


def _load_population_metadata(asset_root: Path) -> tuple[dict[str, Any], pd.DataFrame, str]:
    """Read the small metadata files without scanning the 4.88 GiB table.

    Each persistent worker uses the fully validating asset loader once before
    touching a LUT.  The dispatcher only needs this inexpensive view to plan
    blocks and make ``--dry-run`` genuinely dry.
    """

    root = Path(asset_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_lut_domain") != S8S8_LUT_DOMAIN:
        raise ValueError("bulk runner requires the S8S8 sign-magnitude LUT asset")
    row_count = int(manifest.get("row_count", -1))
    if row_count <= 0:
        raise ValueError("asset manifest has no positive row_count")
    population_name = manifest.get("population_filename")
    if not isinstance(population_name, str) or not population_name:
        raise ValueError("asset manifest has no population_filename")
    population_path = root / population_name
    population = pd.read_csv(population_path, low_memory=False)
    required = {"row_index", "runtime_design_id"}
    missing = sorted(required.difference(population.columns))
    if missing:
        raise ValueError(f"asset population is missing {missing}")
    if len(population) != row_count:
        raise ValueError("asset population row count does not match manifest")
    row_indices = pd.to_numeric(population["row_index"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(row_indices, np.arange(row_count, dtype=np.int64)):
        raise ValueError("asset population row_index must be contiguous and ordered")
    if population["runtime_design_id"].astype(str).duplicated().any():
        raise ValueError("asset population runtime_design_id values must be unique")
    return manifest, population, _sha1_file(manifest_path)


def _load_node_names(
    registry_path: Path,
    model_path: Path,
    *,
    model_key: str = DEFAULT_MODEL_KEY,
) -> list[str]:
    profile = get_dataset_profile(model_key)
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if payload.get("injection_scope_id") != profile.injection_scope_id:
        raise ValueError(
            f"{registry_path} injection scope {payload.get('injection_scope_id')!r} does not match "
            f"{profile.injection_scope_id!r}"
        )
    if tuple(payload.get("included_roles", ())) != profile.included_roles:
        raise ValueError(f"{registry_path} does not declare the canonical included roles")
    if int(payload.get("total_matmul_count", -1)) != profile.expected_total_matmuls:
        raise ValueError(f"{registry_path} does not declare the canonical total MatMul count")
    if set(payload.get("excluded_matmuls", ())) != set(profile.expected_excluded_matmuls):
        raise ValueError(f"{registry_path} does not declare the canonical excluded MatMuls")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"{registry_path} has no node list")
    names = [str(node["node_id"]) for node in nodes]
    if len(names) != profile.expected_injected_matmuls or len(set(names)) != len(names):
        raise ValueError(
            f"{profile.model_key} registry must contain "
            f"{profile.expected_injected_matmuls} unique nodes, got {len(names)}"
        )
    registry_model_key = str(payload.get("model_key", ""))
    accepted_registry_keys = {profile.model_key, f"{profile.model_key}_int8"}
    if registry_model_key and registry_model_key not in accepted_registry_keys:
        raise ValueError(
            f"registry model_key {registry_model_key!r} does not match {profile.model_key!r}"
        )
    model = onnx.load(str(model_path), load_external_data=False)
    partition = select_weight_stationary_block_linears(model, expected_block_count=12)
    matmuls = {str(ref.node.name) for ref in partition.all_matmuls}
    if len(partition.all_matmuls) != profile.expected_total_matmuls:
        raise ValueError(
            f"{profile.model_key} model must contain {profile.expected_total_matmuls} unique MatMuls, "
            f"got {len(partition.all_matmuls)}"
        )
    missing = sorted(set(names).difference(matmuls))
    if missing:
        raise ValueError(f"{profile.model_key} registry does not match model: {missing}")
    canonical_names = list(partition.selected_names)
    if names != canonical_names:
        raise ValueError(
            f"{profile.model_key} registry order/content does not match the semantic linear scope"
        )
    excluded = matmuls.difference(names)
    expected_excluded = set(profile.expected_excluded_matmuls)
    if excluded != expected_excluded:
        raise ValueError(
            f"{profile.model_key} model/registry MatMul partition changed: {sorted(excluded)}"
        )
    return names


def _build_work_plan(
    rows: pd.DataFrame,
    *,
    gpu_ids: Sequence[int],
    block_size: int,
    max_designs: int | None,
) -> WorkPlan:
    if int(block_size) <= 0:
        raise ValueError("--block-size must be positive")
    ids = tuple(int(item) for item in gpu_ids)
    if not ids:
        raise ValueError("at least one GPU is required")
    if "row_index" not in rows or "runtime_design_id" not in rows:
        raise ValueError("work rows require row_index and runtime_design_id")
    selected = rows.sort_values("row_index", kind="stable").reset_index(drop=True).copy()
    if max_designs is not None:
        if int(max_designs) <= 0:
            raise ValueError("--max-designs must be positive when supplied")
        selected = selected.iloc[: int(max_designs)].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError("work plan selected no designs")
    if selected["runtime_design_id"].astype(str).duplicated().any():
        raise ValueError("work plan contains duplicate runtime_design_id values")
    blocks: list[BlockSpec] = []
    for start in range(0, len(selected), int(block_size)):
        block_id = start // int(block_size)
        blocks.append(
            BlockSpec(
                block_id=block_id,
                start=start,
                stop=min(start + int(block_size), len(selected)),
                gpu_id=ids[block_id % len(ids)],
            )
        )
    return WorkPlan(gpu_ids=ids, block_size=int(block_size), blocks=tuple(blocks), selected_rows=selected)


def _plan_summary(
    plan: WorkPlan,
    *,
    execution_profile: str,
    batch_size: int,
) -> dict[str, Any]:
    designs_by_gpu = {
        str(gpu_id): int(sum(block.stop - block.start for block in plan.blocks if block.gpu_id == gpu_id))
        for gpu_id in plan.gpu_ids
    }
    if str(execution_profile) == "throughput" and int(batch_size) == DEFAULT_BATCH_SIZE:
        seconds_per_design = MEASURED_DYNAMIC_B16_APPROX_SECONDS_PER_DESIGN
        estimate_source = (
            "row 644, 1,000 images, dynamic B=16, staged I/O binding, SM89 custom op; "
            "measured under shared-GPU load on 2026-07-17"
        )
    else:
        seconds_per_design = MEASURED_B1_APPROX_SECONDS_PER_DESIGN
        estimate_source = (
            "strict B=1 row-644 approximate reference; use a pilot for the selected dynamic batch/profile"
        )
    seconds = float(len(plan.selected_rows)) * seconds_per_design
    return {
        "design_count": int(len(plan.selected_rows)),
        "block_size": int(plan.block_size),
        "block_count": int(len(plan.blocks)),
        "gpu_ids": list(plan.gpu_ids),
        "designs_by_gpu": designs_by_gpu,
        "execution_profile": str(execution_profile),
        "batch_size": int(batch_size),
        "estimated_approx_seconds_per_design": seconds_per_design,
        "estimated_approx_seconds_total_single_gpu": seconds,
        "estimated_approx_hours_wall_clock_at_ideal_balance": seconds / len(plan.gpu_ids) / 3600.0,
        "estimate_source": estimate_source,
        "pilot_still_recommended": str(execution_profile) != "throughput" or int(batch_size) != DEFAULT_BATCH_SIZE,
    }


def _block_paths(output_root: Path, block_id: int) -> tuple[Path, Path]:
    blocks_dir = Path(output_root) / "blocks"
    stem = f"block_{int(block_id):05d}"
    return blocks_dir / f"{stem}.jsonl", blocks_dir / f"{stem}.manifest.json"


def _row_identity(row: pd.Series) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_index": int(row["row_index"]),
        "runtime_design_id": str(row["runtime_design_id"]),
    }
    for field in ("source_row_index", "source_runtime_design_id", "replacement_kind", "source_config_hash"):
        if field in row.index and pd.notna(row[field]):
            value = row[field]
            result[field] = int(value) if field == "source_row_index" else str(value)
    return result


def _expected_block_identities(plan: WorkPlan, block: BlockSpec) -> list[dict[str, Any]]:
    return [_row_identity(row) for _, row in plan.selected_rows.iloc[block.start : block.stop].iterrows()]


def _load_block_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path} has a blank JSONL record at line {line_number}")
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} has a non-object JSONL record at line {line_number}")
            records.append(payload)
    return records


def _is_complete_block(
    *,
    output_root: Path,
    run_fingerprint: str,
    plan: WorkPlan,
    block: BlockSpec,
) -> bool:
    data_path, manifest_path = _block_paths(output_root, block.block_id)
    if not data_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != BULK_SCHEMA_VERSION:
            return False
        if manifest.get("run_fingerprint") != run_fingerprint:
            return False
        if int(manifest.get("block_id", -1)) != block.block_id:
            return False
        if int(manifest.get("gpu_id", -1)) != block.gpu_id:
            return False
        if manifest.get("data_sha1") != _sha1_file(data_path):
            return False
        records = _load_block_records(data_path)
        expected = _expected_block_identities(plan, block)
        actual = [
            {key: record[key] for key in ("row_index", "runtime_design_id")}
            for record in records
        ]
        expected_minimal = [{key: item[key] for key in ("row_index", "runtime_design_id")} for item in expected]
        return actual == expected_minimal
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False


def _resolve_execution(args: argparse.Namespace) -> tuple[Path, int, str, str]:
    profile = get_dataset_profile(getattr(args, "model_key", DEFAULT_MODEL_KEY))
    execution_profile = str(args.execution_profile).strip().lower()
    if execution_profile not in {"throughput", "strict_b1"}:
        raise ValueError("--execution-profile must be 'throughput' or 'strict_b1'")
    if int(args.s8_residual_abs_bound) < 0:
        raise ValueError("--s8-residual-abs-bound must be non-negative")
    if execution_profile == "strict_b1":
        if int(args.batch_size) != 1:
            raise ValueError("strict_b1 requires --batch-size=1")
        model_path = (
            profile.static_model_path if args.model_path is None else Path(args.model_path)
        )
        return Path(model_path).resolve(), 1, execution_profile, "strict"
    if int(args.batch_size) <= 1:
        raise ValueError("throughput requires --batch-size > 1; use --execution-profile strict_b1 for B=1")
    model_path = (
        profile.dynamic_model_path if args.model_path is None else Path(args.model_path)
    )
    return Path(model_path).resolve(), int(args.batch_size), execution_profile, "throughput"


def _validate_model_batch_contract(*, model_path: Path, batch_size: int, execution_profile: str) -> None:
    model = onnx.load(str(model_path), load_external_data=False)
    if not model.graph.input or not model.graph.output:
        raise ValueError(f"{model_path} has no public input/output")
    input_dims = model.graph.input[0].type.tensor_type.shape.dim
    output_dims = model.graph.output[0].type.tensor_type.shape.dim
    if len(input_dims) != 4 or len(output_dims) != 2:
        raise ValueError(f"{model_path} does not expose the expected rank-4 input/rank-2 logits output")
    input_batch_is_fixed_one = input_dims[0].HasField("dim_value") and int(input_dims[0].dim_value) == 1
    output_batch_is_fixed_one = output_dims[0].HasField("dim_value") and int(output_dims[0].dim_value) == 1
    input_batch_is_symbolic = bool(input_dims[0].dim_param)
    output_batch_is_symbolic = bool(output_dims[0].dim_param)
    if execution_profile == "throughput":
        if not input_batch_is_symbolic or not output_batch_is_symbolic:
            raise ValueError("throughput requires a symbolic dynamic-batch model, not a fixed-batch model")
        if int(batch_size) <= 1:
            raise ValueError("throughput requires batch_size > 1")
    elif execution_profile == "strict_b1":
        if int(batch_size) != 1:
            raise ValueError("strict_b1 requires batch_size=1")
        if not input_batch_is_fixed_one or not output_batch_is_fixed_one:
            raise ValueError("strict_b1 requires a fixed-B=1 model")
    else:  # pragma: no cover - _resolve_execution owns this validation.
        raise AssertionError(execution_profile)


def _create_run_contract(
    args: argparse.Namespace,
    plan: WorkPlan,
    asset_manifest: dict[str, Any],
    asset_manifest_sha1: str,
    *,
    model_path: Path,
    batch_size: int,
    execution_profile: str,
    session_profile: str,
) -> dict[str, Any]:
    model_path = Path(model_path).resolve()
    node_registry = Path(args.node_registry).resolve()
    source_data_dir = Path(args.source_data_dir).resolve()
    asset_root = Path(args.s8_asset_root).resolve()
    contract = {
        "schema_version": BULK_SCHEMA_VERSION,
        "model_key": str(args.model_key),
        "model_path": str(model_path),
        "model_sha1": _sha1_file(model_path),
        "node_registry_path": str(node_registry),
        "node_registry_sha1": _sha1_file(node_registry),
        "source_data_dir": str(source_data_dir),
        "s8_asset_root": str(asset_root),
        "s8_asset_manifest_sha1": asset_manifest_sha1,
        "s8_table_filename": str(asset_manifest["table_filename"]),
        "s8_table_sha1": str(asset_manifest["files"][str(asset_manifest["table_filename"])]),
        "s8_lut_domain": S8S8_LUT_DOMAIN,
        "s8_residual_abs_bound": int(args.s8_residual_abs_bound),
        "injection_mode": get_dataset_profile(args.model_key).injection_mode_contract(),
        "sample_protocol": {
            "sample_count": int(args.sample_count),
            "selection": "first N ImageNet validation images in lexicographic ILSVRC2012_val_*.JPEG order",
            "batch_size": int(batch_size),
            "execution_profile": str(execution_profile),
            "model_input_policy": (
                "dynamic ONNX exact baseline and approximate path use the same ordered batches, including the final short batch"
                if execution_profile == "throughput"
                else "fixed-B=1 strict audit path"
            ),
        },
        "cuda_session_profile": str(session_profile),
        "worker_device_policy": (
            "dispatcher masks each child with CUDA_VISIBLE_DEVICES=<assigned CUDA ordinal>; "
            "the child uses local CUDA device 0 while records retain the assigned physical GPU ID"
        ),
        "input_residency": str(args.input_residency),
        "image_root": str(Path(args.image_root).resolve()),
        "gpu_ids": list(plan.gpu_ids),
        "block_size": int(plan.block_size),
        "selected_design_count": int(len(plan.selected_rows)),
        "selected_row_indices_sha1": hashlib.sha1(
            np.asarray(plan.selected_rows["row_index"], dtype=np.int64).tobytes()
        ).hexdigest(),
        "selected_runtime_ids_sha1": hashlib.sha1(
            "\n".join(plan.selected_rows["runtime_design_id"].astype(str).tolist()).encode("utf-8")
        ).hexdigest(),
        "runner_factory": str(args.runner_factory),
    }
    return contract


def _run_fingerprint(contract: dict[str, Any]) -> str:
    return hashlib.sha1(_json_canonical(contract).encode("utf-8")).hexdigest()


def _selection_path(output_root: Path) -> Path:
    return Path(output_root) / "population" / "selected_rows.csv"


def _shared_paths(output_root: Path) -> dict[str, Path]:
    shared = Path(output_root) / "shared"
    return {
        "root": shared,
        "inputs": shared / "inputs.npy",
        "labels": shared / "labels.npy",
        "selection": shared / "image_selection.json",
        "input_contract": shared / "input_contract.json",
    }


def _exact_baseline_paths(output_root: Path, physical_gpu_id: int) -> dict[str, Path]:
    """Return per-GPU same-batch exact-baseline paths.

    Each worker scores approximate logits against clean logits produced with
    the same physical GPU, session profile, and batch partition. This avoids
    treating a cross-device numerical difference as an approximation error.
    """

    shared = _shared_paths(output_root)
    stem = f"exact_baseline_gpu_{int(physical_gpu_id):02d}"
    return {
        "root": shared["root"],
        "exact": shared["root"] / f"{stem}.npz",
        "exact_contract": shared["root"] / f"{stem}.json",
    }


def _prepare_inputs(
    *,
    output_root: Path,
    model_path: Path,
    image_root: Path,
    sample_count: int,
    preprocess_key: str = "swin_tiny_patch4_window7_224",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    paths = _shared_paths(output_root)
    expected_contract = {
        "model_path": str(Path(model_path).resolve()),
        "model_sha1": _sha1_file(model_path),
        "image_root": str(Path(image_root).resolve()),
        "sample_count": int(sample_count),
        "preprocess_key": str(preprocess_key),
        "selection": "first N ImageNet validation images in lexicographic ILSVRC2012_val_*.JPEG order",
        "dtype": "float32",
        "shape_tail": [3, 224, 224],
    }
    if all(paths[name].is_file() for name in ("inputs", "labels", "selection", "input_contract")):
        try:
            actual_contract = json.loads(paths["input_contract"].read_text(encoding="utf-8"))
            inputs = np.load(paths["inputs"], mmap_mode="r")
            labels = np.load(paths["labels"], allow_pickle=False)
            selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
            if (
                actual_contract == expected_contract
                and inputs.dtype == np.float32
                and tuple(inputs.shape) == (int(sample_count), 3, 224, 224)
                and labels.shape == (int(sample_count),)
                and labels.dtype == np.int64
                and len(selection.get("image_names", [])) == int(sample_count)
            ):
                return inputs, np.asarray(labels, dtype=np.int64), selection
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if paths["root"].exists() and any(paths[name].exists() for name in ("inputs", "labels", "selection", "input_contract")):
        raise RuntimeError("shared input cache is incomplete or incompatible; use a new output root")

    images = sorted(Path(image_root).glob("ILSVRC2012_val_*.JPEG"))
    if len(images) < int(sample_count):
        raise ValueError(f"sample_count={sample_count} exceeds ImageNet images={len(images)}")
    selected_images = images[: int(sample_count)]
    labels = np.asarray(imagenet_gt_indices(int(sample_count)), dtype=np.int64)
    if labels.shape != (int(sample_count),):
        raise ValueError("ImageNet labels do not align with selected images")
    paths["root"].mkdir(parents=True, exist_ok=True)
    temporary = paths["inputs"].with_name(f".{paths['inputs'].stem}.tmp.{os.getpid()}.npy")
    try:
        cache = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(int(sample_count), 3, 224, 224),
        )
        try:
            for index, image_path in enumerate(selected_images):
                tensor = prepare_classification_input(str(preprocess_key), image_path, nhwc=False)
                if tensor.shape != (1, 3, 224, 224):
                    raise ValueError(f"unexpected input for {image_path.name}: {tensor.shape}")
                cache[index] = tensor[0]
            cache.flush()
        finally:
            del cache
        os.replace(temporary, paths["inputs"])
    finally:
        if temporary.exists():
            temporary.unlink()
    selection = {
        "image_names": [path.name for path in selected_images],
        "labels": labels.tolist(),
        "selection_sha1": hashlib.sha1(
            _json_canonical({"image_names": [path.name for path in selected_images], "labels": labels.tolist()}).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_save_npy(paths["labels"], labels)
    _atomic_write_json(paths["selection"], selection)
    _atomic_write_json(paths["input_contract"], expected_contract)
    return np.load(paths["inputs"], mmap_mode="r"), labels, selection


def _prepare_exact_baseline(
    *,
    output_root: Path,
    model_path: Path,
    inputs: np.ndarray,
    labels: np.ndarray,
    physical_gpu_id: int,
    runtime_device_id: int,
    selection: dict[str, Any],
    batch_size: int,
    session_profile: str,
    execution_profile: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    paths = _exact_baseline_paths(output_root, physical_gpu_id)
    expected_contract = {
        "model_path": str(Path(model_path).resolve()),
        "model_sha1": _sha1_file(model_path),
        "sample_count": int(inputs.shape[0]),
        "selection_sha1": str(selection["selection_sha1"]),
        "batch_size": int(batch_size),
        "cuda_session_profile": str(session_profile),
        "execution_profile": str(execution_profile),
        # A bulk worker is launched with only its assigned physical GPU visible,
        # so CUDA device 0 is intentionally local to that child process. Keep
        # both IDs in the cache contract: the former owns the result, while the
        # latter is the CUDA EP ordinal used to produce it.
        "physical_gpu_id": int(physical_gpu_id),
        "runtime_device_id": int(runtime_device_id),
        "labels_sha1": hashlib.sha1(np.asarray(labels, dtype=np.int64).tobytes()).hexdigest(),
    }
    if paths["exact"].is_file() and paths["exact_contract"].is_file():
        try:
            contract = json.loads(paths["exact_contract"].read_text(encoding="utf-8"))
            with np.load(paths["exact"], allow_pickle=False) as payload:
                logits = np.asarray(payload["logits"], dtype=np.float32)
                cached_labels = np.asarray(payload["labels"], dtype=np.int64)
            if (
                all(contract.get(key) == value for key, value in expected_contract.items())
                and logits.shape == (inputs.shape[0], 1000)
                and np.array_equal(cached_labels, labels)
            ):
                return logits, contract
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass
    if paths["root"].exists() and (paths["exact"].exists() or paths["exact_contract"].exists()):
        raise RuntimeError("shared exact baseline is incomplete or incompatible; use a new output root")

    session, provider = create_cuda_session(
        Path(model_path),
        device_id=int(runtime_device_id),
        fallback_parallelism=1,
        session_profile=str(session_profile),
    )
    try:
        input_name = str(session.get_inputs()[0].name)
        outputs = np.empty((inputs.shape[0], 1000), dtype=np.float32)
        started = time.perf_counter()
        for index in range(0, inputs.shape[0], int(batch_size)):
            stop = min(index + int(batch_size), inputs.shape[0])
            value = np.asarray(
                session.run(None, {input_name: np.asarray(inputs[index:stop], dtype=np.float32)})[0],
                dtype=np.float32,
            )
            if value.shape != (stop - index, 1000):
                raise ValueError(f"exact model returned {value.shape}, expected {(stop - index, 1000)}")
            outputs[index:stop] = value
        elapsed = time.perf_counter() - started
    finally:
        session.close()
    contract = {
        **expected_contract,
        "provider": str(provider),
        "elapsed_seconds": float(elapsed),
        "exact_logits_sha1": hashlib.sha1(outputs.tobytes()).hexdigest(),
    }
    _atomic_save_npz(paths["exact"], logits=outputs, labels=np.asarray(labels, dtype=np.int64))
    _atomic_write_json(paths["exact_contract"], contract)
    return outputs, contract


def _load_runner_factory(factory_spec: str):
    if str(factory_spec) == DEFAULT_RUNNER_FACTORY:
        return PersistentIoBindingRunner
    if str(factory_spec) == "compat_pool":
        return CompatPoolRunner
    module_name, separator, attribute = str(factory_spec).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "--runner-factory must be 'persistent_iobinding', 'compat_pool', or 'module.path:callable'"
        )
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"runner factory {factory_spec!r} is not callable")
    return factory


def _build_runner(
    *,
    factory_spec: str,
    model_path: Path,
    node_names: Sequence[str],
    inputs: np.ndarray,
    gpu_id: int,
    batch_size: int,
    s8_residual_abs_bound: int,
    session_profile: str,
    input_residency: str,
    static_shape_reference_path: Path | None = None,
) -> PersistentBulkRunner:
    factory = _load_runner_factory(factory_spec)
    factory_kwargs = {
        "model_path": Path(model_path),
        "node_names": list(node_names),
        "inputs": inputs,
        "gpu_id": int(gpu_id),
        "lut_domain": S8S8_LUT_DOMAIN,
        "batch_size": int(batch_size),
        "s8_residual_abs_bound": int(s8_residual_abs_bound),
        "session_profile": str(session_profile),
        "input_residency": str(input_residency),
    }
    if static_shape_reference_path is not None:
        factory_kwargs["static_shape_reference_path"] = Path(static_shape_reference_path)
    runner = factory(**factory_kwargs)
    for method in ("update_lut", "run_all", "close"):
        if not callable(getattr(runner, method, None)):
            raise TypeError(f"runner factory {factory_spec!r} returned no callable {method}()")
    return runner


def _make_exact_reference(*, exact: np.ndarray, labels: np.ndarray) -> ExactReference:
    exact = np.asarray(exact, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if exact.shape != (len(labels), 1000):
        raise ValueError(f"exact logits must be [N,1000], got {exact.shape}")
    top5 = _topk(exact, 5)
    top1 = top5[:, 0]
    top1_correct = top1 == labels
    top5_correct = np.any(top5 == labels[:, None], axis=1)
    return ExactReference(
        top1=top1,
        top5=top5,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        top1_correct_count=int(np.count_nonzero(top1_correct)),
        top5_correct_count=int(np.count_nonzero(top5_correct)),
    )


def _score_logits(
    *,
    approximate: np.ndarray,
    exact: np.ndarray,
    labels: np.ndarray,
    exact_reference: ExactReference,
) -> dict[str, Any]:
    approximate = np.asarray(approximate, dtype=np.float32)
    exact = np.asarray(exact, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if approximate.shape != exact.shape or approximate.shape != (len(labels), 1000):
        raise ValueError(f"approximate logits {approximate.shape} do not align with exact {exact.shape}")
    approximate_top5 = _topk(approximate, 5)
    approximate_top1 = approximate_top5[:, 0]
    approximate_top1_correct = approximate_top1 == labels
    approximate_top5_correct = np.any(approximate_top5 == labels[:, None], axis=1)
    sample_count = int(len(labels))
    exact_top1_count = int(exact_reference.top1_correct_count)
    exact_top5_count = int(exact_reference.top5_correct_count)
    approximate_top1_count = int(np.count_nonzero(approximate_top1_correct))
    approximate_top5_count = int(np.count_nonzero(approximate_top5_correct))
    abs_diff = np.abs(approximate - exact)
    return {
        "sample_count": sample_count,
        "exact_top1_accuracy": exact_top1_count / sample_count,
        "exact_top5_accuracy": exact_top5_count / sample_count,
        "approximate_top1_accuracy": approximate_top1_count / sample_count,
        "approximate_top5_accuracy": approximate_top5_count / sample_count,
        "top1_drop": (exact_top1_count - approximate_top1_count) / sample_count,
        "top5_drop": (exact_top5_count - approximate_top5_count) / sample_count,
        "top1_prediction_changed_count": int(np.count_nonzero(exact_reference.top1 != approximate_top1)),
        "top5_set_changed_count": int(
            np.count_nonzero(
                np.any(np.sort(exact_reference.top5, axis=1) != np.sort(approximate_top5, axis=1), axis=1)
            )
        ),
        "max_abs_logit_difference": float(np.max(abs_diff)),
        "mean_abs_logit_difference": float(np.mean(abs_diff, dtype=np.float64)),
    }


def _validate_residual_bound(*, residual: np.ndarray, runtime_design_id: str, bound: int) -> int:
    """Prove the LUT meets the bound embedded in every rewritten custom node."""

    value = np.asarray(residual, dtype=np.int32)
    if value.shape != (256, 256):
        raise ValueError(f"{runtime_design_id}: residual must be int32[256,256], got {value.shape}")
    actual = int(np.max(np.abs(value.astype(np.int64, copy=False))))
    if actual > int(bound):
        raise ValueError(
            f"{runtime_design_id}: residual abs max {actual} exceeds graph-bound s8_residual_abs_bound={bound}"
        )
    return actual


def _write_block(
    *,
    output_root: Path,
    run_fingerprint: str,
    block: BlockSpec,
    rows: pd.DataFrame,
    records: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if len(rows) != len(records):
        raise ValueError("block rows and records do not align")
    expected = [_row_identity(row) for _, row in rows.iterrows()]
    actual = [{key: record[key] for key in ("row_index", "runtime_design_id")} for record in records]
    expected_minimal = [{key: item[key] for key in ("row_index", "runtime_design_id")} for item in expected]
    if actual != expected_minimal:
        raise ValueError("block records do not preserve the frozen row/ID order")
    data_path, manifest_path = _block_paths(output_root, block.block_id)
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"block {block.block_id} already exists; resume must validate it first")
    text = "".join(_json_canonical(record) + "\n" for record in records)
    _atomic_write_text(data_path, text)
    manifest = {
        "schema_version": BULK_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "block_id": int(block.block_id),
        "gpu_id": int(block.gpu_id),
        "row_count": int(len(rows)),
        "row_index_start": int(rows["row_index"].iloc[0]),
        "row_index_stop": int(rows["row_index"].iloc[-1]),
        "runtime_design_ids_sha1": hashlib.sha1(
            "\n".join(rows["runtime_design_id"].astype(str).tolist()).encode("utf-8")
        ).hexdigest(),
        "data_file": data_path.name,
        "data_sha1": _sha1_file(data_path),
        "elapsed_seconds": float(elapsed_seconds),
        "completed_at_unix": time.time(),
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _worker_progress_path(output_root: Path, gpu_id: int) -> Path:
    return Path(output_root) / "workers" / f"gpu_{int(gpu_id):02d}.progress.json"


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    run_config_path = output_root / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(run_config_path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    contract = run_config.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("run_config has no immutable contract")
    run_fingerprint = _run_fingerprint(contract)
    if run_config.get("run_fingerprint") != run_fingerprint:
        raise ValueError("run_config fingerprint does not match its contract")
    gpu_id = int(args.gpu_id)
    runtime_device_id = gpu_id if args.runtime_device_id is None else int(args.runtime_device_id)
    if runtime_device_id < 0:
        raise ValueError("--runtime-device-id must be non-negative")
    gpu_ids = tuple(int(item) for item in contract["gpu_ids"])
    if gpu_id not in gpu_ids:
        raise ValueError(f"GPU {gpu_id} is not assigned by this run contract")
    selected_rows = pd.read_csv(_selection_path(output_root), low_memory=False)
    plan = _build_work_plan(
        selected_rows,
        gpu_ids=gpu_ids,
        block_size=int(contract["block_size"]),
        max_designs=None,
    )
    pending_blocks = [
        block
        for block in plan.blocks
        if block.gpu_id == gpu_id
        and not _is_complete_block(
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            plan=plan,
            block=block,
        )
    ]
    if not pending_blocks:
        return {"status": "reused", "gpu_id": gpu_id, "block_count": 0}

    # This is deliberately once per long-lived worker.  It performs the full
    # checksum validation before returning a memory-mapped 20k LUT table.
    asset = load_s8s8_sign_magnitude_truth_lut_asset(Path(contract["s8_asset_root"]))
    if _sha1_file(Path(contract["s8_asset_root"]) / "manifest.json") != contract["s8_asset_manifest_sha1"]:
        raise ValueError("S8 asset manifest changed after bulk run preparation")
    if str(asset.manifest["files"][str(asset.manifest["table_filename"])]) != contract["s8_table_sha1"]:
        raise ValueError("S8 LUT table fingerprint changed after bulk run preparation")

    inputs = np.load(_shared_paths(output_root)["inputs"], mmap_mode="r")
    labels = np.load(_shared_paths(output_root)["labels"], allow_pickle=False).astype(np.int64, copy=False)
    selection = json.loads(_shared_paths(output_root)["selection"].read_text(encoding="utf-8"))
    if inputs.shape != (len(labels), 3, 224, 224):
        raise ValueError("shared inputs do not satisfy the expected shape contract")
    exact_logits, exact_contract = _prepare_exact_baseline(
        output_root=output_root,
        model_path=Path(contract["model_path"]),
        inputs=inputs,
        labels=labels,
        physical_gpu_id=gpu_id,
        runtime_device_id=runtime_device_id,
        selection=selection,
        batch_size=int(contract["sample_protocol"]["batch_size"]),
        session_profile=str(contract["cuda_session_profile"]),
        execution_profile=str(contract["sample_protocol"]["execution_profile"]),
    )
    if exact_logits.shape != (len(labels), 1000):
        raise ValueError("per-GPU exact baseline does not satisfy the expected shape contract")
    exact_reference = _make_exact_reference(exact=exact_logits, labels=labels)
    node_names = _load_node_names(
        Path(contract["node_registry_path"]),
        Path(contract["model_path"]),
        model_key=str(contract.get("model_key", DEFAULT_MODEL_KEY)),
    )
    runner = _build_runner(
        factory_spec=str(contract["runner_factory"]),
        model_path=Path(contract["model_path"]),
        node_names=node_names,
        inputs=inputs,
        gpu_id=runtime_device_id,
        batch_size=int(contract["sample_protocol"]["batch_size"]),
        s8_residual_abs_bound=int(contract["s8_residual_abs_bound"]),
        session_profile=str(contract["cuda_session_profile"]),
        input_residency=str(contract["input_residency"]),
        static_shape_reference_path=get_dataset_profile(
            str(contract.get("model_key", DEFAULT_MODEL_KEY))
        ).static_model_path,
    )
    completed_designs = 0
    worker_started = time.monotonic()
    try:
        for block_position, block in enumerate(pending_blocks, start=1):
            rows = plan.selected_rows.iloc[block.start : block.stop].copy().reset_index(drop=True)
            deltas = asset.deltas_for_population_rows(rows)
            records: list[dict[str, Any]] = []
            block_started = time.monotonic()
            for _, row in rows.iterrows():
                runtime_design_id = str(row["runtime_design_id"])
                residual_abs_max = _validate_residual_bound(
                    residual=deltas[runtime_design_id],
                    runtime_design_id=runtime_design_id,
                    bound=int(contract["s8_residual_abs_bound"]),
                )
                runner.update_lut(deltas[runtime_design_id])
                approximate_logits = runner.run_all()
                metrics = _score_logits(
                    approximate=approximate_logits,
                    exact=exact_logits,
                    labels=labels,
                    exact_reference=exact_reference,
                )
                records.append(
                    {
                        **_row_identity(row),
                        **metrics,
                        "exact_baseline_device_id": gpu_id,
                        "exact_baseline_logits_sha1": str(exact_contract["exact_logits_sha1"]),
                        "input_residency": str(getattr(runner, "input_residency", "unknown")),
                        "s8_residual_abs_max": residual_abs_max,
                        "approximate_elapsed_seconds": float(getattr(runner, "last_run_elapsed_seconds", float("nan"))),
                    }
                )
            elapsed = time.monotonic() - block_started
            _write_block(
                output_root=output_root,
                run_fingerprint=run_fingerprint,
                block=block,
                rows=rows,
                records=records,
                elapsed_seconds=elapsed,
            )
            completed_designs += len(rows)
            total_elapsed = max(time.monotonic() - worker_started, 1e-9)
            remaining_designs = sum(item.stop - item.start for item in pending_blocks[block_position:])
            rate = completed_designs / total_elapsed
            _atomic_write_json(
                _worker_progress_path(output_root, gpu_id),
                {
                    "schema_version": BULK_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "gpu_id": gpu_id,
                    "completed_blocks_this_invocation": block_position,
                    "pending_blocks_at_start": len(pending_blocks),
                    "completed_designs_this_invocation": completed_designs,
                    "elapsed_seconds_this_invocation": total_elapsed,
                    "designs_per_second": rate,
                    "remaining_designs_this_invocation": remaining_designs,
                    "eta_seconds_this_invocation": remaining_designs / rate if rate > 0 else None,
                    "last_completed_block": block.block_id,
                },
            )
    finally:
        runner.close()
    return {
        "status": "completed",
        "gpu_id": gpu_id,
        "block_count": len(pending_blocks),
        "design_count": completed_designs,
        "elapsed_seconds": time.monotonic() - worker_started,
    }


def _merge(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    run_config = json.loads((output_root / "run_config.json").read_text(encoding="utf-8"))
    contract = run_config["contract"]
    run_fingerprint = _run_fingerprint(contract)
    if run_config.get("run_fingerprint") != run_fingerprint:
        raise ValueError("run_config fingerprint does not match its contract")
    rows = pd.read_csv(_selection_path(output_root), low_memory=False)
    plan = _build_work_plan(
        rows,
        gpu_ids=tuple(int(item) for item in contract["gpu_ids"]),
        block_size=int(contract["block_size"]),
        max_designs=None,
    )
    all_records: list[dict[str, Any]] = []
    missing_blocks: list[int] = []
    for block in plan.blocks:
        if not _is_complete_block(
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            plan=plan,
            block=block,
        ):
            missing_blocks.append(block.block_id)
            continue
        data_path, _manifest_path = _block_paths(output_root, block.block_id)
        all_records.extend(_load_block_records(data_path))
    if missing_blocks:
        raise RuntimeError(f"cannot merge; incomplete or invalid blocks: {missing_blocks[:20]}")
    result = pd.DataFrame(all_records)
    required = {
        "row_index",
        "runtime_design_id",
        "top1_drop",
        "top5_drop",
        "sample_count",
        "exact_top1_accuracy",
        "exact_top5_accuracy",
        "exact_baseline_device_id",
        "exact_baseline_logits_sha1",
    }
    missing = sorted(required.difference(result.columns))
    if missing:
        raise ValueError(f"bulk records are missing {missing}")
    result["row_index"] = pd.to_numeric(result["row_index"], errors="raise").astype(np.int64)
    result = result.sort_values("row_index", kind="stable").reset_index(drop=True)
    expected_rows = plan.selected_rows["row_index"].to_numpy(dtype=np.int64)
    expected_ids = plan.selected_rows["runtime_design_id"].astype(str).to_numpy()
    if not np.array_equal(result["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("merged rows do not exactly cover the frozen selected population")
    if not np.array_equal(result["runtime_design_id"].astype(str).to_numpy(), expected_ids):
        raise ValueError("merged runtime IDs do not exactly cover the frozen selected population")
    if result["runtime_design_id"].astype(str).duplicated().any():
        raise ValueError("merged runtime IDs are duplicated")
    sample_count = int(contract["sample_protocol"]["sample_count"])
    for field in ("top1_drop", "top5_drop"):
        values = pd.to_numeric(result[field], errors="raise").to_numpy(dtype=np.float64)
        if np.any(np.abs(values * sample_count - np.rint(values * sample_count)) > 5e-6):
            raise ValueError(f"{field} violates the true-{sample_count} metric quantization contract")
    baseline_columns = [
        "exact_baseline_device_id",
        "exact_baseline_logits_sha1",
        "exact_top1_accuracy",
        "exact_top5_accuracy",
    ]
    exact_baselines = (
        result[baseline_columns]
        .drop_duplicates()
        .sort_values(["exact_baseline_device_id", "exact_baseline_logits_sha1"], kind="stable")
        .to_dict(orient="records")
    )

    merged = output_root / "merged"
    merged.mkdir(parents=True, exist_ok=True)
    labels = result.copy()
    labels_path = merged / "labels.csv"
    temporary_csv = labels_path.with_name(f".{labels_path.name}.tmp.{os.getpid()}")
    try:
        labels.to_csv(temporary_csv, index=False)
        os.replace(temporary_csv, labels_path)
    finally:
        if temporary_csv.exists():
            temporary_csv.unlink()
    targets = labels[["top1_drop", "top5_drop"]].to_numpy(dtype=np.float32)
    targets_path = merged / "targets.npy"
    _atomic_save_npy(targets_path, targets)
    manifest = {
        "schema_version": BULK_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "row_count": int(len(labels)),
        "sample_count": sample_count,
        "exact_baselines": exact_baselines,
        "labels_sha1": _sha1_file(labels_path),
        "targets_sha1": _sha1_file(targets_path),
        "blocks_verified": int(len(plan.blocks)),
        "valid": True,
    }
    _atomic_write_json(merged / "manifest.json", manifest)
    return {"status": "completed", "output_root": str(output_root), **manifest}


def _run(args: argparse.Namespace) -> dict[str, Any]:
    profile = _resolve_profile_arguments(args)
    model_path, batch_size, execution_profile, session_profile = _resolve_execution(args)
    _validate_model_batch_contract(
        model_path=model_path,
        batch_size=batch_size,
        execution_profile=execution_profile,
    )
    output_root = Path(args.output_root).resolve()
    asset_manifest, population, asset_manifest_sha1 = _load_population_metadata(Path(args.s8_asset_root))
    plan = _build_work_plan(
        population,
        gpu_ids=_parse_gpu_ids(args.gpu_ids),
        block_size=int(args.block_size),
        max_designs=args.max_designs,
    )
    _load_node_names(Path(args.node_registry), model_path, model_key=profile.model_key)
    contract = _create_run_contract(
        args,
        plan,
        asset_manifest,
        asset_manifest_sha1,
        model_path=model_path,
        batch_size=batch_size,
        execution_profile=execution_profile,
        session_profile=session_profile,
    )
    run_fingerprint = _run_fingerprint(contract)
    summary = _plan_summary(plan, execution_profile=execution_profile, batch_size=batch_size)
    if bool(args.dry_run):
        return {
            "status": "dry_run",
            "output_root": str(output_root),
            "run_fingerprint": run_fingerprint,
            "contract": contract,
            "plan": summary,
        }

    if output_root.exists():
        if not bool(args.resume):
            raise FileExistsError(f"{output_root} already exists; use --resume or choose a new output root")
        run_config_path = output_root / "run_config.json"
        if not run_config_path.is_file():
            raise FileNotFoundError(f"cannot resume without {run_config_path}")
        prior = json.loads(run_config_path.read_text(encoding="utf-8"))
        if prior.get("contract") != contract or prior.get("run_fingerprint") != run_fingerprint:
            raise ValueError("resume arguments do not match the existing immutable bulk run contract")
        selected_rows = pd.read_csv(_selection_path(output_root), low_memory=False)
        if not selected_rows.equals(plan.selected_rows):
            raise ValueError("frozen selected_rows.csv does not match the requested asset population")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _selection_path(output_root).parent.mkdir(parents=True, exist_ok=True)
        plan.selected_rows.to_csv(_selection_path(output_root), index=False)
        _atomic_write_json(
            output_root / "run_config.json",
            {
                "schema_version": BULK_SCHEMA_VERSION,
                "created_at_unix": time.time(),
                "run_fingerprint": run_fingerprint,
                "contract": contract,
                "plan": summary,
            },
        )

    inputs, labels, selection = _prepare_inputs(
        output_root=output_root,
        model_path=model_path,
        image_root=Path(args.image_root),
        sample_count=int(args.sample_count),
        preprocess_key=profile.preprocess_key,
    )
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    for gpu_id in plan.gpu_ids:
        log_path = logs_dir / f"gpu_{gpu_id:02d}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--output-root",
            str(output_root),
            "--gpu-id",
            str(gpu_id),
            "--runtime-device-id",
            "0",
        ]
        handle = log_path.open("wb")
        worker_environment = _isolated_worker_environment(physical_gpu_id=gpu_id)
        processes.append(
            (
                gpu_id,
                subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=worker_environment,
                ),
                handle,
            )
        )
    failures: list[str] = []
    for gpu_id, process, handle in processes:
        return_code = process.wait()
        handle.close()
        if return_code != 0:
            failures.append(f"GPU {gpu_id} worker exited with {return_code}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return _merge(argparse.Namespace(output_root=output_root))


def _resolve_profile_arguments(args: argparse.Namespace):
    profile = get_dataset_profile(getattr(args, "model_key", DEFAULT_MODEL_KEY))
    defaults = {
        "output_root": profile.default_output_root,
        "source_data_dir": profile.default_source_data_dir,
        "node_registry": profile.node_registry_path,
        "image_root": DEFAULT_IMAGE_ROOT,
    }
    for name, value in defaults.items():
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, value)
    return profile


def _add_common_arguments(parser: argparse.ArgumentParser, *, default_output: Path | None) -> None:
    parser.add_argument("--output-root", type=Path, default=default_output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="prepare, dispatch persistent GPU workers, then merge")
    _add_common_arguments(run_parser, default_output=None)
    run_parser.add_argument(
        "--model-key",
        choices=tuple(sorted(DATASET_PROFILES)),
        default=DEFAULT_MODEL_KEY,
    )
    run_parser.add_argument("--source-data-dir", type=Path, default=None)
    run_parser.add_argument("--s8-asset-root", type=Path, default=DEFAULT_S8_ASSET_ROOT)
    run_parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="override the profile-selected model; throughput requires symbolic batch and strict_b1 requires fixed B=1",
    )
    run_parser.add_argument("--node-registry", type=Path, default=None)
    run_parser.add_argument("--image-root", type=Path, default=None)
    run_parser.add_argument("--sample-count", type=int, default=1000)
    run_parser.add_argument("--gpu-ids", default=",".join(str(item) for item in DEFAULT_GPU_IDS))
    run_parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    run_parser.add_argument("--max-designs", type=int, default=None)
    run_parser.add_argument("--runner-factory", default=DEFAULT_RUNNER_FACTORY)
    run_parser.add_argument("--execution-profile", choices=("throughput", "strict_b1"), default="throughput")
    run_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    run_parser.add_argument("--s8-residual-abs-bound", type=int, default=DEFAULT_S8_RESIDUAL_ABS_BOUND)
    run_parser.add_argument(
        "--input-residency",
        choices=("auto", "device", "staged"),
        default="auto",
        help="keep all input batches on GPU when possible; auto falls back to staged inputs only after CUDA allocation failure",
    )
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")

    worker_parser = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    _add_common_arguments(worker_parser, default_output=DEFAULT_OUTPUT_ROOT)
    worker_parser.add_argument("--gpu-id", type=int, required=True)
    worker_parser.add_argument(
        "--runtime-device-id",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )

    merge_parser = subparsers.add_parser("merge", help="verify completed blocks and write merged labels")
    _add_common_arguments(merge_parser, default_output=DEFAULT_OUTPUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        result = _run(args)
    elif args.command == "worker":
        result = _worker(args)
    elif args.command == "merge":
        result = _merge(args)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
