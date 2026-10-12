#!/usr/bin/env python
"""Build a reusable clean-application propagation profile for dynamic-B16 Swin.

The profile is deliberately multiplier-independent.  It uses a fixed subset of
clean frozen inputs and, for every registered MatMul, measures the symmetric
logit response to full constant ``+1`` and ``-1`` signed residual tables.  It
never loads a candidate LUT, a per-design runtime bank, or design targets.

The opt-in ``--full-logit-response`` mode retains the complete per-node,
per-image, per-class central response instead of only its fixed-runner-up
summary.  It also records the frozen true1000 application labels so a later
static-only stage can reconstruct class switches and clean-wrong recoveries.
Those labels describe the shared application profile; they are not multiplier
design targets and do not add any candidate inference.

``--dry-run`` validates the complete frozen-asset contract and computes the
clean class references, but it does not import the LUT runtime, compile a CUDA
custom op, create an ORT session, or write an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer.common import FORMAL_CACHE_ROOT

DATASET_ROOT = (
    PROJECT_ROOT / "datasets" / "swin_tiny" / "s8s8_strict50_true1000_raw500_v2"
)
SHARED_ROOT = DATASET_ROOT / "shared"
DEFAULT_MODEL = (
    EA2_ROOT
    / "ai_models"
    / "onnx_models"
    / "classification"
    / "swin_tiny_patch4_window7_224_imagenet1k_int8_dynamic.onnx"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "resources"
    / "data"
    / "swin_tiny_s8s8"
    / "node_registry"
    / "node_registry.json"
)
DEFAULT_INPUTS = SHARED_ROOT / "union_inputs.npy"
DEFAULT_EXACT_BASELINE = SHARED_ROOT / "canonical_exact_baseline.npz"
DEFAULT_EXACT_META = SHARED_ROOT / "true1000_exact_meta.npz"
DEFAULT_EXACT_CONTRACT = SHARED_ROOT / "canonical_exact_contract.json"
DEFAULT_INPUT_CONTRACT = SHARED_ROOT / "input_contract.json"
DEFAULT_OUTPUT = FORMAL_CACHE_ROOT / "dynamic_b16_propagation_profile.npz"

SCHEMA_VERSION = "swin_dynamic_b16_propagation_profile_v1"
FULL_LOGIT_SCHEMA_VERSION = "swin_dynamic_b16_propagation_full_logits_v2"
LUT_DOMAIN = "s8s8_sign_magnitude"
EXPECTED_NODE_COUNT = 72
EXPECTED_MATMUL_COUNT = 75
EXPECTED_CLASS_COUNT = 1000
EXPECTED_UNION_COUNT = 1486
EXPECTED_TRUE1000_COUNT = 1000
EXPECTED_INPUT_SHAPE = (EXPECTED_UNION_COUNT, 3, 224, 224)
EXPECTED_EXCLUDED_MATMULS = (
    "/layers.0/downsample/reduction/MatMul",
    "/layers.1/downsample/reduction/MatMul",
    "/layers.2/downsample/reduction/MatMul",
)

# These hashes make this profile an explicit contract for the current frozen
# dynamic-B16 application.  A new model or frozen input bank requires a new
# audited profile rather than silently reusing this one.
EXPECTED_ASSET_HASHES: dict[str, dict[str, str]] = {
    "model": {
        "sha1": "dd78030b0b5b36c5d85de5a24b177573998b7661",
        "sha256": "b56f6e5c3c243e835a8bbad3b460847fa1351e76ecd40e9d68502b04327bb4e3",
    },
    "registry": {
        "sha1": "f78106e49723fc32f38a47350d807f97c67c2939",
        "sha256": "9f725d50f5ff2420c9b2beca32934f9cc19c24ba063e8ffdef7a4f3025271851",
    },
    "inputs": {
        "sha1": "e31bfccc5ec7100d982c611897be05b731b5dfce",
        "sha256": "009861250f94853f2cd13c3043ab43646feb57da249fe768442ece010622c343",
    },
    "exact_baseline": {
        "sha1": "9ef965ca2182caa99fc74a97586e80f6d83ebd1e",
        "sha256": "820f9ddc07264879bb4b34e1bedab4c92a6d88dd54a7ce00ec4467d4618169e9",
    },
    "exact_meta": {
        "sha1": "9736d9f13e0eac6de5a3d76ca2df7f824893a499",
        "sha256": "2d95ef0359e22e556449c99de3cd7091e4163b0cc46db4a92836ca71c9d155d6",
    },
    "exact_contract": {
        "sha1": "6b831cc074a2a9d4930de5a248fa5c235ddeb983",
        "sha256": "19e53b17c9a6c13095e5d62309d922edd5aaf8b7622fd4186b8b9ba08259f5fe",
    },
    "input_contract": {
        "sha1": "7331d179e141f8ad8916ad35d34fe6e59f24999c",
        "sha256": "a97587052367146d96587f67af8c436757cae13c41efe4992433fa632faf3708",
    },
}
EXPECTED_EXACT_LOGITS_SHA1 = "c4ddfc4c39f038cef7ca2482b29c6d174ba2f350"
EXPECTED_TRUE1000_POSITIONS_SHA1 = "ff48114ac81ed8350e3c81dd07789b095d8504f7"


@dataclass(frozen=True)
class PropagationPlan:
    model_path: Path
    registry_path: Path
    inputs_path: Path
    exact_baseline_path: Path
    exact_meta_path: Path
    exact_contract_path: Path
    input_contract_path: Path
    output_path: Path
    node_ids: tuple[str, ...]
    lut_input_names: tuple[str, ...]
    selected_true1000_indices: np.ndarray
    selected_union_positions: np.ndarray
    clean_pred_class: np.ndarray
    clean_runner_up_class: np.ndarray
    clean_margin: np.ndarray
    metadata: dict[str, Any]
    selected_clean_logits: np.ndarray | None = None
    selected_true_label: np.ndarray | None = None
    selected_exact_top1_correct: np.ndarray | None = None
    retain_full_logit_response: bool = False


def _file_hashes(path: Path) -> dict[str, str]:
    digests = {"sha1": hashlib.sha1(), "sha256": hashlib.sha256()}
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            for digest in digests.values():
                digest.update(chunk)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def _array_hash(values: np.ndarray, algorithm: str = "sha256") -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.new(algorithm)
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON contract must be an object: {path}")
    return value


def _temporary_sibling(path: Path, purpose: str) -> Path:
    token = uuid.uuid4().hex
    return path.with_name(f".{path.name}.{purpose}.{os.getpid()}.{token}")


def _write_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_npz_exact(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    # Passing a path to np.savez_compressed silently appends ".npz" when the
    # path has another suffix.  A binary handle preserves the exact CLI path.
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path, "partial")
    try:
        _write_json_exact(temporary, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(path, "partial")
    try:
        _write_npz_exact(temporary, arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _profile_metadata_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.json")


def _publish_profile_pair(
    output_path: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Publish NPZ first and its JSON completion marker last.

    POSIX cannot atomically replace two paths as one transaction.  Both files
    are therefore staged completely before publication, while the sidecar is
    treated as the completion marker.  On a caught publication failure, a new
    incomplete pair is removed or an overwritten old pair is restored.
    """
    output_path = Path(output_path)
    metadata_path = _profile_metadata_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if (output_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"profile output exists; pass --overwrite: {output_path}")

    staged_output = _temporary_sibling(output_path, "partial")
    staged_metadata = _temporary_sibling(metadata_path, "partial")
    backup_output = _temporary_sibling(output_path, "backup")
    backup_metadata = _temporary_sibling(metadata_path, "backup")
    backed_output = False
    backed_metadata = False
    installed_output = False
    installed_metadata = False
    complete_metadata: dict[str, Any] | None = None
    try:
        _write_npz_exact(staged_output, arrays)
        output_hashes = _file_hashes(staged_output)
        complete_metadata = {
            **dict(metadata),
            "artifact_complete": True,
            "publication_contract": "sidecar_json_is_completion_marker_v1",
            "output_hashes": output_hashes,
        }
        _write_json_exact(staged_metadata, complete_metadata)

        # Recheck after staging so a concurrent writer is not silently replaced
        # when overwrite was not authorized.
        if (output_path.exists() or metadata_path.exists()) and not overwrite:
            raise FileExistsError(
                f"profile output appeared during staging: {output_path}"
            )

        try:
            # Remove the old completion marker before changing its NPZ.  A
            # reader can therefore never see an old sidecar naming a new file.
            if metadata_path.exists():
                os.replace(metadata_path, backup_metadata)
                backed_metadata = True
            if output_path.exists():
                os.replace(output_path, backup_output)
                backed_output = True
            os.replace(staged_output, output_path)
            installed_output = True
            os.replace(staged_metadata, metadata_path)
            installed_metadata = True
        except BaseException as publish_error:
            rollback_errors: list[OSError] = []
            for installed, path in (
                (installed_metadata, metadata_path),
                (installed_output, output_path),
            ):
                if installed and path.exists():
                    try:
                        path.unlink()
                    except (
                        OSError
                    ) as error:  # pragma: no cover - catastrophic filesystem failure
                        rollback_errors.append(error)
            # Restore the data first and its completion marker last.
            for backed, backup, destination in (
                (backed_output, backup_output, output_path),
                (backed_metadata, backup_metadata, metadata_path),
            ):
                if backed and backup.exists():
                    try:
                        os.replace(backup, destination)
                    except (
                        OSError
                    ) as error:  # pragma: no cover - catastrophic filesystem failure
                        rollback_errors.append(error)
            if rollback_errors:
                raise RuntimeError(
                    "profile publication failed and the previous pair could not be fully restored; "
                    f"backups remain beside {output_path}"
                ) from publish_error
            raise
        else:
            for backup in (backup_output, backup_metadata):
                if backup.exists():
                    try:
                        backup.unlink()
                    except OSError:
                        # The committed pair is valid; a stale hidden backup is
                        # preferable to rolling back a successful publication.
                        pass
    finally:
        for staged in (staged_output, staged_metadata):
            if staged.exists():
                staged.unlink()
    if complete_metadata is None:  # pragma: no cover - defensive invariant
        raise AssertionError("profile metadata was not staged")
    return complete_metadata


def clean_fixed_margin_reference(
    logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic clean top-1, runner-up, and fixed-class margin."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"clean logits must have shape [N,C>=2], got {values.shape}")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("clean logits must be finite floating-point values")
    # np.argmax resolves ties to the lowest class index.  Replacing that one
    # entry with -inf gives the same deterministic rule for the runner-up.
    predicted = np.argmax(values, axis=1).astype(np.int64, copy=False)
    masked = np.asarray(values, dtype=np.float64).copy()
    masked[np.arange(len(masked)), predicted] = -np.inf
    runner_up = np.argmax(masked, axis=1).astype(np.int64, copy=False)
    rows = np.arange(len(values))
    margin = (
        np.asarray(values[rows, predicted], dtype=np.float64)
        - np.asarray(values[rows, runner_up], dtype=np.float64)
    ).astype(np.float32)
    if np.any(margin < 0.0):
        raise AssertionError("clean predicted-vs-runner-up margin cannot be negative")
    return predicted, runner_up, margin


def central_logit_response(
    plus_logits: np.ndarray,
    minus_logits: np.ndarray,
    *,
    probe_magnitude: int = 1,
) -> np.ndarray:
    plus = np.asarray(plus_logits)
    minus = np.asarray(minus_logits)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError(
            f"plus/minus logits must be aligned rank-two arrays: {plus.shape}, {minus.shape}"
        )
    if probe_magnitude <= 0:
        raise ValueError("probe_magnitude must be positive")
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise ValueError("plus/minus logits must be finite")
    return np.asarray(
        (plus.astype(np.float64) - minus.astype(np.float64))
        / (2.0 * float(probe_magnitude)),
        dtype=np.float32,
    )


def fixed_margin_response(
    derivative_logits: np.ndarray,
    clean_pred_class: np.ndarray,
    clean_runner_up_class: np.ndarray,
) -> np.ndarray:
    derivative = np.asarray(derivative_logits)
    predicted = np.asarray(clean_pred_class, dtype=np.int64)
    runner_up = np.asarray(clean_runner_up_class, dtype=np.int64)
    if (
        derivative.ndim != 2
        or predicted.shape != (len(derivative),)
        or runner_up.shape != predicted.shape
    ):
        raise ValueError("derivative logits and fixed clean classes are not aligned")
    if np.any(predicted < 0) or np.any(predicted >= derivative.shape[1]):
        raise ValueError("clean predicted class is outside the logit range")
    if np.any(runner_up < 0) or np.any(runner_up >= derivative.shape[1]):
        raise ValueError("clean runner-up class is outside the logit range")
    rows = np.arange(len(derivative))
    return np.asarray(
        derivative[rows, predicted] - derivative[rows, runner_up],
        dtype=np.float32,
    )


def linearized_logits_from_node_coefficients(
    clean_logits: np.ndarray,
    dlogits_d_uniform_residual: np.ndarray,
    node_image_coefficients: np.ndarray,
) -> np.ndarray:
    """Reconstruct logits from static per-node, per-image residual coefficients.

    The coefficient unit is the magnitude of the uniform signed-int8 residual
    table used by this profile.  Coefficients may describe one candidate as
    ``[image,node]`` or a candidate batch as ``[candidate,image,node]``.  This
    matches the frozen per-image operand-profile layout.
    """
    clean = np.asarray(clean_logits)
    derivative = np.asarray(dlogits_d_uniform_residual)
    coefficients = np.asarray(node_image_coefficients)
    if clean.ndim != 2:
        raise ValueError(
            f"clean logits must have shape [image,class], got {clean.shape}"
        )
    if derivative.ndim != 3 or derivative.shape[1:] != clean.shape:
        raise ValueError(
            "full logit response must have shape [node,image,class] aligned with clean logits"
        )
    expected_coefficient_shape = (derivative.shape[1], derivative.shape[0])
    if (
        coefficients.ndim not in (2, 3)
        or coefficients.shape[-2:] != expected_coefficient_shape
    ):
        raise ValueError(
            "image-node coefficients must have shape [image,node] or "
            "[candidate,image,node]"
        )
    if not all(
        np.issubdtype(values.dtype, np.floating) and np.isfinite(values).all()
        for values in (clean, derivative, coefficients)
    ):
        raise ValueError(
            "linearized reconstruction inputs must be finite floating-point arrays"
        )
    if coefficients.ndim == 2:
        shift = np.einsum(
            "in,nic->ic",
            coefficients.astype(np.float64, copy=False),
            derivative.astype(np.float64, copy=False),
            optimize=True,
        )
        return np.asarray(clean.astype(np.float64) + shift, dtype=np.float32)
    shift = np.einsum(
        "din,nic->dic",
        coefficients.astype(np.float64, copy=False),
        derivative.astype(np.float64, copy=False),
        optimize=True,
    )
    return np.asarray(clean.astype(np.float64)[None, :, :] + shift, dtype=np.float32)


def top1_correct_bits_from_logits(
    logits: np.ndarray,
    true_label: np.ndarray,
) -> np.ndarray:
    """Return exact class-competition correctness for one or many candidates."""
    values = np.asarray(logits)
    labels = np.asarray(true_label)
    if values.ndim not in (2, 3):
        raise ValueError(
            "logits must have shape [image,class] or [candidate,image,class]"
        )
    if labels.shape != (values.shape[-2],) or labels.dtype != np.int64:
        raise ValueError("true labels must be int64[image] aligned with logits")
    if np.any(labels < 0) or np.any(labels >= values.shape[-1]):
        raise ValueError("true labels are outside the logit class range")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("logits must be finite floating-point values")
    predicted = np.argmax(values, axis=-1)
    expected = labels if values.ndim == 2 else labels[None, :]
    return np.asarray(predicted == expected, dtype=np.uint8)


def constant_probe_table(value: int) -> np.ndarray:
    if isinstance(value, bool) or int(value) != value:
        raise TypeError("constant probe value must be an integer")
    return np.full((256, 256), int(value), dtype=np.int32)


def select_fixed_images(
    true1000_positions: np.ndarray,
    *,
    image_count: int,
    selection_seed: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    positions = np.asarray(true1000_positions)
    if positions.shape != (EXPECTED_TRUE1000_COUNT,) or positions.dtype != np.int64:
        raise ValueError("true1000 positions must be int64[1000]")
    if len(np.unique(positions)) != EXPECTED_TRUE1000_COUNT:
        raise ValueError("true1000 positions must be unique")
    if not 1 <= int(image_count) <= EXPECTED_TRUE1000_COUNT:
        raise ValueError("image_count must be in [1, 1000]")
    ranked: list[tuple[bytes, int, int]] = []
    for local_index, union_position in enumerate(positions.tolist()):
        key = hashlib.sha256(
            f"{int(selection_seed)}\0{int(union_position)}".encode("ascii")
        ).digest()
        ranked.append((key, int(union_position), int(local_index)))
    ranked.sort()
    chosen = ranked[: int(image_count)]
    local = np.asarray([item[2] for item in chosen], dtype=np.int64)
    union = np.asarray([item[1] for item in chosen], dtype=np.int64)
    selection_hash = _canonical_json_hash(
        {
            "algorithm": "lowest_sha256_seed_nul_union_position_v1",
            "seed": int(selection_seed),
            "selected_true1000_indices": local.tolist(),
            "selected_union_positions": union.tolist(),
        }
    )
    return local, union, selection_hash


def _validate_asset_hashes(paths: Mapping[str, Path]) -> dict[str, dict[str, str]]:
    actual: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen asset is missing: {path}")
        hashes = _file_hashes(path)
        expected = EXPECTED_ASSET_HASHES[name]
        if hashes != expected:
            raise ValueError(
                f"{name} hash mismatch for the frozen dynamic-B16 contract: "
                f"expected {expected}, got {hashes}"
            )
        actual[name] = hashes
    return actual


def _validate_model_and_registry(
    model_path: Path,
    registry_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    registry = _read_json(registry_path)
    legacy_contract = (
        registry.get("schema_version") == "swin_tiny_s8s8_node_registry/v1"
        and registry.get("lut_domain") == LUT_DOMAIN
    )
    linear_contract = (
        registry.get("injection_scope_id")
        == "sram_cim_weight_stationary_linear_only_v1"
    )
    if not (legacy_contract or linear_contract):
        raise ValueError("unexpected Swin node-registry schema")
    if registry.get("node_count") != EXPECTED_NODE_COUNT:
        raise ValueError("node registry has an unexpected injected-node count")
    node_items = registry.get("nodes")
    if not isinstance(node_items, list):
        raise TypeError("node registry has no nodes list")
    node_ids = tuple(str(item["node_id"]) for item in node_items)
    if (
        len(node_ids) != EXPECTED_NODE_COUNT
        or len(set(node_ids)) != EXPECTED_NODE_COUNT
    ):
        raise ValueError("node registry does not contain the expected unique node IDs")

    model = onnx.load(str(model_path), load_external_data=False)
    graph_inputs = list(model.graph.input)
    graph_outputs = list(model.graph.output)
    if len(graph_inputs) != 1 or len(graph_outputs) != 1:
        raise ValueError(
            "dynamic-B16 Swin must have one predictor input and one output"
        )
    input_dims = graph_inputs[0].type.tensor_type.shape.dim
    output_dims = graph_outputs[0].type.tensor_type.shape.dim
    if len(input_dims) != 4 or input_dims[0].dim_param != "batch":
        raise ValueError(
            "Swin predictor input does not expose the required dynamic 'batch' dimension"
        )
    if tuple(int(dim.dim_value) for dim in input_dims[1:]) != (3, 224, 224):
        raise ValueError("Swin predictor input must have shape [batch,3,224,224]")
    if (
        len(output_dims) != 2
        or output_dims[0].dim_param != "batch"
        or int(output_dims[1].dim_value) != 1000
    ):
        raise ValueError("Swin output must have shape [batch,1000]")

    matmul_ids = tuple(
        str(node.name) for node in model.graph.node if node.op_type == "MatMul"
    )
    if (
        len(matmul_ids) != EXPECTED_MATMUL_COUNT
        or len(set(matmul_ids)) != EXPECTED_MATMUL_COUNT
    ):
        raise ValueError("dynamic-B16 Swin must contain 75 uniquely named MatMuls")
    missing = sorted(set(node_ids).difference(matmul_ids))
    excluded = tuple(name for name in matmul_ids if name not in set(node_ids))
    if missing or excluded != EXPECTED_EXCLUDED_MATMULS:
        raise ValueError(
            f"injected/excluded MatMul partition mismatch: missing={missing}, excluded={excluded}"
        )
    injected_in_graph_order = tuple(
        name for name in matmul_ids if name in set(node_ids)
    )
    if node_ids != injected_in_graph_order:
        raise ValueError(
            "registry node order no longer matches dynamic-B16 graph order"
        )
    lut_input_names = tuple(
        f"propagation_probe_node_{index:02d}" for index in range(len(node_ids))
    )
    model_contract = {
        "predictor_input_name": str(graph_inputs[0].name),
        "predictor_output_name": str(graph_outputs[0].name),
        "predictor_input_shape": ["batch", 3, 224, 224],
        "predictor_output_shape": ["batch", 1000],
        "matmul_count": len(matmul_ids),
        "injected_matmul_count": len(node_ids),
        "excluded_matmuls": list(excluded),
        "node_order_sha256": hashlib.sha256(
            "\n".join(node_ids).encode("utf-8")
        ).hexdigest(),
    }
    return node_ids, lut_input_names, model_contract


def _validate_clean_assets(
    inputs_path: Path,
    exact_baseline_path: Path,
    exact_contract_path: Path,
    input_contract_path: Path,
    *,
    image_count: int,
    selection_seed: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    exact_contract = _read_json(exact_contract_path)
    input_contract = _read_json(input_contract_path)
    if exact_contract.get("model_sha1") != EXPECTED_ASSET_HASHES["model"]["sha1"]:
        raise ValueError(
            "canonical exact contract is not bound to the current dynamic-B16 model"
        )
    if exact_contract.get("exact_logits_sha1") != EXPECTED_EXACT_LOGITS_SHA1:
        raise ValueError("canonical exact logits contract changed")
    if (
        exact_contract.get("frozen_input_contract_sha1")
        != EXPECTED_ASSET_HASHES["input_contract"]["sha1"]
    ):
        raise ValueError(
            "canonical exact contract is not bound to the frozen input contract"
        )
    if (
        exact_contract.get("batch_size") != 16
        or exact_contract.get("cuda_session_profile") != "throughput"
    ):
        raise ValueError(
            "canonical exact baseline was not generated with dynamic-B16 throughput contract"
        )
    if exact_contract.get("sample_count") != EXPECTED_UNION_COUNT:
        raise ValueError("canonical exact contract has an unexpected sample count")
    if (
        input_contract.get("input_shape") != list(EXPECTED_INPUT_SHAPE)
        or input_contract.get("input_dtype") != "float32"
    ):
        raise ValueError(
            "frozen input contract has an unexpected tensor shape or dtype"
        )
    if input_contract.get("union_count") != EXPECTED_UNION_COUNT:
        raise ValueError("frozen input contract has an unexpected union count")
    if (
        input_contract.get("true1000_positions_sha1")
        != EXPECTED_TRUE1000_POSITIONS_SHA1
    ):
        raise ValueError("frozen true1000 position contract changed")

    inputs = np.load(inputs_path, mmap_mode="r", allow_pickle=False)
    if inputs.shape != EXPECTED_INPUT_SHAPE or inputs.dtype != np.float32:
        raise ValueError(
            f"frozen union inputs are invalid: {inputs.shape} {inputs.dtype}"
        )
    del inputs

    # Intentionally index only these two arrays.  union_labels and the raw500
    # position vector are not read by this clean, label-free profile.
    with np.load(exact_baseline_path, allow_pickle=False) as exact:
        logits = np.asarray(exact["logits"])
        true1000_positions = np.asarray(exact["true1000_positions"])
    if (
        logits.shape != (EXPECTED_UNION_COUNT, EXPECTED_CLASS_COUNT)
        or logits.dtype != np.float32
    ):
        raise ValueError(
            f"canonical exact logits are invalid: {logits.shape} {logits.dtype}"
        )
    if not np.isfinite(logits).all():
        raise ValueError("canonical exact logits contain non-finite values")
    if _array_hash(logits, "sha1") != EXPECTED_EXACT_LOGITS_SHA1:
        raise ValueError(
            "canonical exact logits bytes do not match their frozen contract"
        )
    if (
        true1000_positions.shape != (EXPECTED_TRUE1000_COUNT,)
        or true1000_positions.dtype != np.int64
    ):
        raise ValueError("canonical true1000 positions are not int64[1000]")
    if _array_hash(true1000_positions, "sha1") != EXPECTED_TRUE1000_POSITIONS_SHA1:
        raise ValueError(
            "canonical true1000 positions do not match their frozen contract"
        )
    if len(np.unique(true1000_positions)) != EXPECTED_TRUE1000_COUNT:
        raise ValueError("canonical true1000 positions are not unique")
    if np.any(true1000_positions < 0) or np.any(
        true1000_positions >= EXPECTED_UNION_COUNT
    ):
        raise ValueError(
            "canonical true1000 positions are outside the union input bank"
        )

    selected_local, selected_union, selection_sha256 = select_fixed_images(
        true1000_positions,
        image_count=image_count,
        selection_seed=selection_seed,
    )
    selected_logits = np.ascontiguousarray(logits[selected_union], dtype=np.float32)
    predicted, runner_up, margin = clean_fixed_margin_reference(selected_logits)
    clean_contract = {
        "selection_algorithm": "lowest_sha256_seed_nul_union_position_v1",
        "selection_seed": int(selection_seed),
        "selection_sha256": selection_sha256,
        "selected_true1000_indices": selected_local.tolist(),
        "selected_union_positions": selected_union.tolist(),
        "selected_clean_logits_sha256": _array_hash(selected_logits),
        "clean_pred_class_sha256": _array_hash(predicted),
        "clean_runner_up_class_sha256": _array_hash(runner_up),
        "clean_margin_sha256": _array_hash(margin),
        "clean_margin_min": float(np.min(margin)),
        "clean_margin_mean": float(np.mean(margin, dtype=np.float64)),
        "clean_margin_max": float(np.max(margin)),
    }
    return (
        selected_local,
        selected_union,
        selected_logits,
        predicted,
        runner_up,
        margin,
        clean_contract,
    )


def _load_selected_application_labels(
    exact_meta_path: Path,
    selected_true1000_indices: np.ndarray,
    selected_clean_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and independently verify frozen labels for the selected inputs."""
    selected = np.asarray(selected_true1000_indices)
    clean_logits = np.asarray(selected_clean_logits)
    if selected.ndim != 1 or selected.dtype != np.int64:
        raise ValueError("selected true1000 indices must be int64[image]")
    if clean_logits.shape != (len(selected), EXPECTED_CLASS_COUNT):
        raise ValueError(
            "selected clean logits are not aligned with selected true1000 indices"
        )
    with np.load(exact_meta_path, allow_pickle=False) as exact_meta:
        true_label_bank = np.asarray(exact_meta["true_label"])
        exact_correct_bank = np.asarray(exact_meta["exact_top1_correct_bits"])
    if (
        true_label_bank.shape != (EXPECTED_TRUE1000_COUNT,)
        or true_label_bank.dtype != np.int64
    ):
        raise ValueError("true1000 true_label must be int64[1000]")
    if np.any(true_label_bank < 0) or np.any(true_label_bank >= EXPECTED_CLASS_COUNT):
        raise ValueError("true1000 true_label contains a class outside [0,1000)")
    if (
        exact_correct_bank.shape != (EXPECTED_TRUE1000_COUNT,)
        or exact_correct_bank.dtype != np.uint8
    ):
        raise ValueError("true1000 exact_top1_correct_bits must be uint8[1000]")
    if not np.all((exact_correct_bank == 0) | (exact_correct_bank == 1)):
        raise ValueError("true1000 exact_top1_correct_bits must be binary")
    if np.any(selected < 0) or np.any(selected >= EXPECTED_TRUE1000_COUNT):
        raise ValueError("selected true1000 indices are outside [0,1000)")

    labels = np.ascontiguousarray(true_label_bank[selected], dtype=np.int64)
    recorded_correct = np.ascontiguousarray(
        exact_correct_bank[selected], dtype=np.uint8
    )
    derived_correct = top1_correct_bits_from_logits(clean_logits, labels)
    if not np.array_equal(recorded_correct, derived_correct):
        mismatch = np.flatnonzero(recorded_correct != derived_correct)
        raise ValueError(
            "frozen exact correctness disagrees with canonical clean logits and labels at "
            f"{len(mismatch)} selected rows"
        )
    contract = {
        "application_profile_label_arrays_read": [
            "true_label",
            "exact_top1_correct_bits",
        ],
        "selected_true_label_sha256": _array_hash(labels),
        "selected_exact_top1_correct_sha256": _array_hash(recorded_correct),
        "selected_exact_top1_correct_count": int(recorded_correct.sum()),
        "selected_clean_wrong_count": int(
            len(recorded_correct) - recorded_correct.sum()
        ),
        "exact_correctness_verification": (
            "argmax(selected canonical clean logits) == selected true_label"
        ),
    }
    return labels, recorded_correct, contract


def build_plan(args: argparse.Namespace) -> PropagationPlan:
    retain_full_logit_response = bool(getattr(args, "full_logit_response", False))
    paths = {
        "model": Path(args.model).expanduser().resolve(),
        "registry": Path(args.node_registry).expanduser().resolve(),
        "inputs": Path(args.inputs).expanduser().resolve(),
        "exact_baseline": Path(args.exact_baseline).expanduser().resolve(),
        "exact_contract": Path(args.exact_contract).expanduser().resolve(),
        "input_contract": Path(args.input_contract).expanduser().resolve(),
    }
    exact_meta_path = (
        Path(getattr(args, "exact_meta", DEFAULT_EXACT_META)).expanduser().resolve()
    )
    hash_paths = dict(paths)
    if retain_full_logit_response:
        hash_paths["exact_meta"] = exact_meta_path
    output_path = Path(args.output).expanduser().resolve()
    metadata_path = _profile_metadata_path(output_path)
    if (
        not args.dry_run
        and (output_path.exists() or metadata_path.exists())
        and not args.overwrite
    ):
        raise FileExistsError(f"profile output exists; pass --overwrite: {output_path}")
    asset_hashes = _validate_asset_hashes(hash_paths)
    node_ids, lut_input_names, model_contract = _validate_model_and_registry(
        paths["model"], paths["registry"]
    )
    (
        selected_local,
        selected_union,
        selected_logits,
        predicted,
        runner_up,
        margin,
        clean_contract,
    ) = _validate_clean_assets(
        paths["inputs"],
        paths["exact_baseline"],
        paths["exact_contract"],
        paths["input_contract"],
        image_count=int(args.image_count),
        selection_seed=int(args.selection_seed),
    )
    selected_labels: np.ndarray | None = None
    selected_exact_correct: np.ndarray | None = None
    label_contract: dict[str, Any] = {}
    if retain_full_logit_response:
        selected_labels, selected_exact_correct, label_contract = (
            _load_selected_application_labels(
                exact_meta_path,
                selected_local,
                selected_logits,
            )
        )
    batch_count = (int(args.image_count) + int(args.batch_size) - 1) // int(
        args.batch_size
    )
    metadata: dict[str, Any] = {
        "schema_version": (
            FULL_LOGIT_SCHEMA_VERSION if retain_full_logit_response else SCHEMA_VERSION
        ),
        "profile_scope": "one_time_clean_application_propagation_profile",
        "deployment_reuse": "frozen application profile shared by every candidate multiplier",
        "model": str(paths["model"]),
        "node_registry": str(paths["registry"]),
        "inputs": str(paths["inputs"]),
        "exact_baseline": str(paths["exact_baseline"]),
        "exact_meta": str(exact_meta_path) if retain_full_logit_response else None,
        "exact_contract": str(paths["exact_contract"]),
        "input_contract": str(paths["input_contract"]),
        "asset_hashes": asset_hashes,
        **model_contract,
        **clean_contract,
        **label_contract,
        "node_count": len(node_ids),
        "image_count": int(args.image_count),
        "batch_size": int(args.batch_size),
        "probe_method": "symmetric_constant_residual_central_difference",
        "probe_magnitude": int(args.probe_magnitude),
        "probe_table_shape": [256, 256],
        "probe_table_dtype": "int32",
        "semantic_zero_policy": (
            "full constant raw-code table; ApproxMatMulCudaS8Lut masks qA==zpA or qB==zpB per node"
        ),
        "margin_reference": "fixed_clean_predicted_class_vs_fixed_clean_runner_up",
        "clean_label_usage": (
            "frozen_true1000_application_profile_for_full_class_competition"
            if retain_full_logit_response
            else "none"
        ),
        "application_profile_labels_loaded": retain_full_logit_response,
        "design_accuracy_drop_targets_loaded": False,
        "exact_baseline_arrays_read": ["logits", "true1000_positions"],
        "canonical_targets_loaded": False,
        "candidate_runtime_bank_loaded": False,
        "candidate_multiplier_inference_count": 0,
        "candidate_multiplier_luts_loaded": False,
        "per_design_runtime_observations_used": False,
        "retain_full_logit_response": retain_full_logit_response,
        "full_logit_response_shape": (
            [len(node_ids), int(args.image_count), EXPECTED_CLASS_COUNT]
            if retain_full_logit_response
            else None
        ),
        "full_logit_response_dtype": "float32" if retain_full_logit_response else None,
        "full_logit_response_uncompressed_bytes": (
            len(node_ids) * int(args.image_count) * EXPECTED_CLASS_COUNT * 4
            if retain_full_logit_response
            else 0
        ),
        "planned_probe_conditions": len(node_ids) * 2,
        "planned_clean_application_runtime_calls": len(node_ids) * 2 * batch_count,
        "lut_domain": LUT_DOMAIN,
        "s8_residual_abs_bound": 1,
        "session_profile": "throughput",
        "runtime_device_id": int(args.device_id),
        "output": str(output_path),
    }
    return PropagationPlan(
        model_path=paths["model"],
        registry_path=paths["registry"],
        inputs_path=paths["inputs"],
        exact_baseline_path=paths["exact_baseline"],
        exact_meta_path=exact_meta_path,
        exact_contract_path=paths["exact_contract"],
        input_contract_path=paths["input_contract"],
        output_path=output_path,
        node_ids=node_ids,
        lut_input_names=lut_input_names,
        selected_true1000_indices=selected_local,
        selected_union_positions=selected_union,
        clean_pred_class=predicted,
        clean_runner_up_class=runner_up,
        clean_margin=margin,
        metadata=metadata,
        selected_clean_logits=(selected_logits if retain_full_logit_response else None),
        selected_true_label=selected_labels,
        selected_exact_top1_correct=selected_exact_correct,
        retain_full_logit_response=retain_full_logit_response,
    )


def _artifact_arrays(
    plan: PropagationPlan,
    *,
    margin_by_node: np.ndarray,
    logit_rms: np.ndarray,
    logit_mean_abs: np.ndarray,
    plus_hashes: Sequence[str],
    minus_hashes: Sequence[str],
    dlogits_d_uniform_residual: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    margins = np.asarray(margin_by_node, dtype=np.float32)
    arrays = {
        "node_ids": np.asarray(plan.node_ids, dtype=np.str_),
        "lut_input_names": np.asarray(plan.lut_input_names, dtype=np.str_),
        "selected_true1000_indices": np.asarray(
            plan.selected_true1000_indices, dtype=np.int64
        ),
        "selected_union_positions": np.asarray(
            plan.selected_union_positions, dtype=np.int64
        ),
        "clean_pred_class": np.asarray(plan.clean_pred_class, dtype=np.int64),
        "clean_runner_up_class": np.asarray(plan.clean_runner_up_class, dtype=np.int64),
        "clean_pred_vs_runner_up_margin": np.asarray(
            plan.clean_margin, dtype=np.float32
        ),
        "fixed_margin_response": margins,
        "logit_rms_response": np.asarray(logit_rms, dtype=np.float32),
        "logit_mean_abs_response": np.asarray(logit_mean_abs, dtype=np.float32),
        "margin_response_mean": np.mean(margins, axis=1, dtype=np.float64).astype(
            np.float32
        ),
        "margin_response_rms": np.sqrt(
            np.mean(np.square(margins, dtype=np.float64), axis=1)
        ).astype(np.float32),
        "margin_response_mean_abs": np.mean(
            np.abs(margins).astype(np.float64), axis=1
        ).astype(np.float32),
        "plus_logits_sha256": np.asarray(plus_hashes, dtype="<U64"),
        "minus_logits_sha256": np.asarray(minus_hashes, dtype="<U64"),
        "candidate_multiplier_inference_count": np.asarray(0, dtype=np.int64),
    }
    if dlogits_d_uniform_residual is not None:
        if not plan.retain_full_logit_response:
            raise ValueError(
                "full logit response was supplied for a legacy propagation plan"
            )
        if (
            plan.selected_clean_logits is None
            or plan.selected_true_label is None
            or plan.selected_exact_top1_correct is None
        ):
            raise ValueError(
                "full logit propagation plan is missing clean logits or labels"
            )
        full_response = np.asarray(dlogits_d_uniform_residual, dtype=np.float32)
        arrays.update(
            {
                "clean_logits": np.asarray(
                    plan.selected_clean_logits, dtype=np.float32
                ),
                "true_label": np.asarray(plan.selected_true_label, dtype=np.int64),
                "exact_top1_correct": np.asarray(
                    plan.selected_exact_top1_correct, dtype=np.uint8
                ),
                "dlogits_d_uniform_residual": full_response,
            }
        )
    elif plan.retain_full_logit_response:
        raise ValueError(
            "full logit propagation plan requires the complete response tensor"
        )
    validate_artifact_arrays(
        arrays,
        node_count=len(plan.node_ids),
        image_count=len(plan.clean_margin),
        full_logit_response=plan.retain_full_logit_response,
    )
    return arrays


def validate_artifact_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    node_count: int,
    image_count: int,
    full_logit_response: bool | None = None,
) -> None:
    expected_shapes = {
        "node_ids": (node_count,),
        "lut_input_names": (node_count,),
        "selected_true1000_indices": (image_count,),
        "selected_union_positions": (image_count,),
        "clean_pred_class": (image_count,),
        "clean_runner_up_class": (image_count,),
        "clean_pred_vs_runner_up_margin": (image_count,),
        "fixed_margin_response": (node_count, image_count),
        "logit_rms_response": (node_count,),
        "logit_mean_abs_response": (node_count,),
        "margin_response_mean": (node_count,),
        "margin_response_rms": (node_count,),
        "margin_response_mean_abs": (node_count,),
        "plus_logits_sha256": (node_count,),
        "minus_logits_sha256": (node_count,),
        "candidate_multiplier_inference_count": (),
    }
    full_shapes = {
        "clean_logits": (image_count, EXPECTED_CLASS_COUNT),
        "true_label": (image_count,),
        "exact_top1_correct": (image_count,),
        "dlogits_d_uniform_residual": (
            node_count,
            image_count,
            EXPECTED_CLASS_COUNT,
        ),
    }
    full_keys_present = set(full_shapes).intersection(arrays)
    if full_logit_response is None:
        full_logit_response = bool(full_keys_present)
    if full_logit_response:
        expected_shapes.update(full_shapes)
    elif full_keys_present:
        raise ValueError(
            "legacy propagation artifact unexpectedly contains full-logit arrays"
        )
    missing = sorted(set(expected_shapes).difference(arrays))
    if missing:
        raise ValueError(f"profile artifact is missing arrays: {missing}")
    for name, shape in expected_shapes.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(
                f"profile array {name} has shape {np.asarray(arrays[name]).shape}, expected {shape}"
            )
    numeric_names = (
        "clean_pred_vs_runner_up_margin",
        "fixed_margin_response",
        "logit_rms_response",
        "logit_mean_abs_response",
        "margin_response_mean",
        "margin_response_rms",
        "margin_response_mean_abs",
    )
    if full_logit_response:
        numeric_names += ("clean_logits", "dlogits_d_uniform_residual")
    for name in numeric_names:
        if not np.isfinite(np.asarray(arrays[name])).all():
            raise ValueError(f"profile array {name} contains non-finite values")
    if np.any(np.asarray(arrays["clean_pred_vs_runner_up_margin"]) < 0.0):
        raise ValueError("clean fixed margins must be non-negative")
    if np.any(np.asarray(arrays["logit_rms_response"]) < 0.0):
        raise ValueError("logit RMS responses must be non-negative")
    if np.any(np.asarray(arrays["logit_mean_abs_response"]) < 0.0):
        raise ValueError("logit mean-absolute responses must be non-negative")
    if int(np.asarray(arrays["candidate_multiplier_inference_count"]).item()) != 0:
        raise ValueError("candidate multiplier inference count must remain zero")
    for name in ("plus_logits_sha256", "minus_logits_sha256"):
        if any(len(str(value)) != 64 for value in np.asarray(arrays[name]).tolist()):
            raise ValueError(f"{name} must contain SHA256 strings")
    if full_logit_response:
        labels = np.asarray(arrays["true_label"])
        exact_correct = np.asarray(arrays["exact_top1_correct"])
        if (
            labels.dtype != np.int64
            or np.any(labels < 0)
            or np.any(labels >= EXPECTED_CLASS_COUNT)
        ):
            raise ValueError("true_label must be int64 with values in [0,1000)")
        if exact_correct.dtype != np.uint8 or not np.all(
            (exact_correct == 0) | (exact_correct == 1)
        ):
            raise ValueError("exact_top1_correct must be binary uint8")
        derived_correct = top1_correct_bits_from_logits(arrays["clean_logits"], labels)
        if not np.array_equal(derived_correct, exact_correct):
            raise ValueError(
                "exact_top1_correct disagrees with clean logits and true_label"
            )
        clean_predicted, clean_runner_up, clean_margin = clean_fixed_margin_reference(
            arrays["clean_logits"]
        )
        for name, derived in (
            ("clean_pred_class", clean_predicted),
            ("clean_runner_up_class", clean_runner_up),
            ("clean_pred_vs_runner_up_margin", clean_margin),
        ):
            if not np.array_equal(np.asarray(arrays[name]), derived):
                raise ValueError(f"{name} disagrees with clean_logits")
        derivative = np.asarray(arrays["dlogits_d_uniform_residual"])
        image_rows = np.arange(image_count)
        derived_margin = np.asarray(
            derivative[:, image_rows, np.asarray(arrays["clean_pred_class"])]
            - derivative[:, image_rows, np.asarray(arrays["clean_runner_up_class"])],
            dtype=np.float32,
        )
        if not np.array_equal(
            derived_margin, np.asarray(arrays["fixed_margin_response"])
        ):
            raise ValueError(
                "fixed margin response is not an exact projection of full dlogits"
            )
        derivative64 = derivative.astype(np.float64)
        derived_rms = np.sqrt(
            np.mean(np.square(derivative64), axis=(1, 2), dtype=np.float64)
        ).astype(np.float32)
        derived_mean_abs = np.mean(
            np.abs(derivative64), axis=(1, 2), dtype=np.float64
        ).astype(np.float32)
        if not np.array_equal(derived_rms, np.asarray(arrays["logit_rms_response"])):
            raise ValueError(
                "logit_rms_response is not an exact summary of full dlogits"
            )
        if not np.array_equal(
            derived_mean_abs, np.asarray(arrays["logit_mean_abs_response"])
        ):
            raise ValueError(
                "logit_mean_abs_response is not an exact summary of full dlogits"
            )


def _run_gpu_profile(
    plan: PropagationPlan,
    args: argparse.Namespace,
    *,
    runtime_builder: Any | None = None,
) -> dict[str, Any]:
    if runtime_builder is None:
        # Keep this import below the dry-run return.  Importing the runtime
        # exposes the CUDA build path, which dry-run is required not to touch.
        from aaae.inference_runtime.transformer.assignment_runtime import build_assignment_runtime

        runtime_builder = build_assignment_runtime
    mapping = dict(zip(plan.node_ids, plan.lut_input_names, strict=True))
    runtime = runtime_builder(
        plan.model_path,
        node_to_lut_input_name=mapping,
        device_id=int(args.device_id),
        batch_fallback_parallelism=1,
        lut_domain=LUT_DOMAIN,
        s8_residual_abs_bound=1,
        session_profile="throughput",
    )
    try:
        if runtime.provider != "CUDAExecutionProvider":
            raise RuntimeError(
                f"propagation profile requires CUDAExecutionProvider, got {runtime.provider}"
            )
        if runtime.lut_input_name_by_node != mapping:
            raise RuntimeError(
                "compiled LUT runtime changed the audited node/input mapping"
            )
        runtime_model_path = str(runtime.model_path)
        runtime_model_sha256 = _file_hashes(Path(runtime.model_path))["sha256"]
        provider = str(runtime.provider)

        image_count = len(plan.selected_union_positions)
        node_count = len(plan.node_ids)
        inputs = np.load(plan.inputs_path, mmap_mode="r", allow_pickle=False)
        positive = constant_probe_table(int(args.probe_magnitude))
        negative = constant_probe_table(-int(args.probe_magnitude))
        zero = constant_probe_table(0)
        zero_feeds = {name: zero for name in plan.lut_input_names}
        margin_by_node = np.empty((node_count, image_count), dtype=np.float32)
        full_response = (
            np.empty(
                (node_count, image_count, EXPECTED_CLASS_COUNT),
                dtype=np.float32,
            )
            if plan.retain_full_logit_response
            else None
        )
        logit_rms = np.empty(node_count, dtype=np.float32)
        logit_mean_abs = np.empty(node_count, dtype=np.float32)
        plus_hashes: list[str] = []
        minus_hashes: list[str] = []
        started = time.perf_counter()

        def run_condition(target_input: str, table: np.ndarray) -> np.ndarray:
            result = np.empty((image_count, EXPECTED_CLASS_COUNT), dtype=np.float32)
            feeds = dict(zero_feeds)
            feeds[target_input] = table
            for start in range(0, image_count, int(args.batch_size)):
                stop = min(start + int(args.batch_size), image_count)
                batch = np.ascontiguousarray(
                    inputs[plan.selected_union_positions[start:stop]],
                    dtype=np.float32,
                )
                outputs = runtime.run_prebuilt(batch, lut_feeds_by_input=feeds)
                if len(outputs) != 1:
                    raise RuntimeError(
                        f"rewritten Swin returned {len(outputs)} outputs, expected one"
                    )
                logits = np.asarray(outputs[0], dtype=np.float32)
                if (
                    logits.shape != (stop - start, EXPECTED_CLASS_COUNT)
                    or not np.isfinite(logits).all()
                ):
                    raise RuntimeError(
                        f"rewritten Swin returned invalid logits {logits.shape} "
                        f"for rows {start}:{stop}"
                    )
                result[start:stop] = logits
            return result

        for node_index, lut_input_name in enumerate(plan.lut_input_names):
            plus = run_condition(lut_input_name, positive)
            minus = run_condition(lut_input_name, negative)
            derivative = central_logit_response(
                plus,
                minus,
                probe_magnitude=int(args.probe_magnitude),
            )
            margin_by_node[node_index] = fixed_margin_response(
                derivative,
                plan.clean_pred_class,
                plan.clean_runner_up_class,
            )
            if full_response is not None:
                full_response[node_index] = derivative
            derivative64 = derivative.astype(np.float64)
            logit_rms[node_index] = np.float32(
                np.sqrt(np.mean(np.square(derivative64), dtype=np.float64))
            )
            logit_mean_abs[node_index] = np.float32(
                np.mean(np.abs(derivative64), dtype=np.float64)
            )
            plus_hashes.append(_array_hash(plus))
            minus_hashes.append(_array_hash(minus))
        elapsed = float(time.perf_counter() - started)
    finally:
        runtime.close()

    arrays = _artifact_arrays(
        plan,
        margin_by_node=margin_by_node,
        logit_rms=logit_rms,
        logit_mean_abs=logit_mean_abs,
        plus_hashes=plus_hashes,
        minus_hashes=minus_hashes,
        dlogits_d_uniform_residual=full_response,
    )
    full_logit_metadata: dict[str, Any] = {}
    if plan.retain_full_logit_response:
        full_logit_metadata = {
            "full_logit_array_hashes": {
                name: _array_hash(arrays[name])
                for name in (
                    "clean_logits",
                    "true_label",
                    "exact_top1_correct",
                    "dlogits_d_uniform_residual",
                )
            },
            "node_image_class_order_sha256": _canonical_json_hash(
                {
                    "node_ids": list(plan.node_ids),
                    "selected_true1000_indices": (
                        plan.selected_true1000_indices.tolist()
                    ),
                    "selected_union_positions": plan.selected_union_positions.tolist(),
                    "class_order": "imagenet1k_class_index_0_through_999",
                }
            ),
            "linearized_reconstruction_contract": (
                "clean_logits[i,c] + sum_n coefficient[d,i,n] * "
                "dlogits_d_uniform_residual[n,i,c]"
            ),
            "coefficient_unit": "uniform_signed_int8_residual_table_magnitude",
            "class_competition": "argmax_over_all_1000_reconstructed_logits",
            "clean_wrong_recovery_supported": True,
        }
    metadata = {
        **plan.metadata,
        **full_logit_metadata,
        "dry_run": False,
        "provider": provider,
        "compiled_runtime_model": runtime_model_path,
        "compiled_runtime_model_sha256": runtime_model_sha256,
        "completed_probe_conditions": node_count * 2,
        "completed_clean_application_runtime_calls": int(
            node_count
            * 2
            * ((image_count + int(args.batch_size) - 1) // int(args.batch_size))
        ),
        "candidate_multiplier_inference_count": 0,
        "elapsed_seconds": elapsed,
    }
    return _publish_profile_pair(
        plan.output_path,
        arrays,
        metadata,
        overwrite=bool(args.overwrite),
    )


def dry_run_summary(plan: PropagationPlan) -> dict[str, Any]:
    return {
        **plan.metadata,
        "dry_run": True,
        "runtime_imported": False,
        "cuda_custom_op_compiled": False,
        "ort_session_created": False,
        "artifact_written": False,
        "candidate_multiplier_inference_count": 0,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    if args.dry_run:
        return dry_run_summary(plan)
    return _run_gpu_profile(plan, args)


def _self_test() -> dict[str, Any]:
    logits = np.asarray([[3.0, 3.0, 1.0], [-2.0, 4.0, 4.0]], dtype=np.float32)
    predicted, runner, margin = clean_fixed_margin_reference(logits)
    np.testing.assert_array_equal(predicted, np.asarray([0, 1]))
    np.testing.assert_array_equal(runner, np.asarray([1, 2]))
    np.testing.assert_array_equal(margin, np.zeros(2, dtype=np.float32))
    plus = np.asarray([[5.0, 1.0, -1.0], [2.0, 8.0, 4.0]], dtype=np.float32)
    minus = np.asarray([[1.0, 3.0, -1.0], [0.0, 2.0, 6.0]], dtype=np.float32)
    derivative = central_logit_response(plus, minus)
    np.testing.assert_allclose(
        derivative,
        np.asarray([[2.0, -1.0, 0.0], [1.0, 3.0, -1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        fixed_margin_response(derivative, predicted, runner),
        np.asarray([3.0, 4.0], dtype=np.float32),
    )
    for value in (-1, 0, 1):
        table = constant_probe_table(value)
        assert table.shape == (256, 256) and table.dtype == np.int32
        assert np.all(table == value)
    positions = np.arange(EXPECTED_TRUE1000_COUNT, dtype=np.int64)
    first = select_fixed_images(positions, image_count=64, selection_seed=20260727)
    second = select_fixed_images(positions, image_count=64, selection_seed=20260727)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    return {
        "self_test": "passed",
        "tests": [
            "clean_fixed_margin_ties",
            "central_logit_and_fixed_margin_response",
            "full_constant_int32_probe_tables",
            "deterministic_image_selection",
        ],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--node-registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    result.add_argument("--exact-baseline", type=Path, default=DEFAULT_EXACT_BASELINE)
    result.add_argument("--exact-meta", type=Path, default=DEFAULT_EXACT_META)
    result.add_argument("--exact-contract", type=Path, default=DEFAULT_EXACT_CONTRACT)
    result.add_argument("--input-contract", type=Path, default=DEFAULT_INPUT_CONTRACT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--image-count", type=int, default=64)
    result.add_argument("--selection-seed", type=int, default=20260727)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--probe-magnitude", type=int, default=1)
    result.add_argument(
        "--full-logit-response",
        action="store_true",
        help=(
            "retain [node,image,1000] central responses and frozen application labels "
            "for static full-class reconstruction"
        ),
    )
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--self-test", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
        return 0
    if not 1 <= int(args.image_count) <= EXPECTED_TRUE1000_COUNT:
        raise SystemExit("--image-count must be in [1, 1000]")
    if int(args.batch_size) != 16:
        raise SystemExit(
            "--batch-size must remain 16 for the frozen dynamic-B16 contract"
        )
    if int(args.device_id) < 0:
        raise SystemExit("--device-id must be non-negative")
    if int(args.probe_magnitude) != 1:
        raise SystemExit(
            "--probe-magnitude must remain 1 for the audited +1/-1 profile"
        )
    print(json.dumps(execute(args), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
