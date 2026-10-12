#!/usr/bin/env python
"""Build a frozen per-image/node full-byte Swin operand profile.

The profile is aligned to the images in the audited 256-image propagation
profile.  For each selected clean image and each registered MatMul, it stores
the complete signed-S8S8 joint operand distribution weighted by the native
Q/DQ product scale.  Each 256x256 surface is normalized to sum to one, so its
contraction with a candidate residual LUT is the multiplier-specific amplitude
relative to the propagation profile's constant +1 probe.

No candidate LUT, per-design target, runtime bank, or candidate inference is
loaded by this one-time clean-application profiling step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer.common import (
    FORMAL_CACHE_ROOT,
    sha256_array,
    sha256_file,
)
from aaae.accuracy_predictor.features.transformer.prepare_dynamic_operand_fullbyte_profile import (
    BYTE_CODES,
    _pair_histogram_fullbyte,
    _static_b_histograms,
)
from aaae.accuracy_predictor.features.transformer.prepare_dynamic_operand_fullbyte_profile import (
    run_synthetic_checks as run_fullbyte_synthetic_checks,
)
from aaae.accuracy_predictor.features.transformer.prepare_dynamic_operand_profile import (
    DEFAULT_INPUTS,
    DEFAULT_INSTRUMENTED_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REGISTRY,
    QuantizedInput,
    _dynamic_subset,
    _instrumentation_contract,
    _load_nodes,
    _prepare_instrumented_model,
    _sha1_file,
)
from aaae.inference_runtime.transformer.ort_session import create_cuda_session

DEFAULT_PROPAGATION_PROFILE = (
    FORMAL_CACHE_ROOT / "dynamic_b16_propagation_profile256.npz"
)
DEFAULT_OUTPUT = (
    FORMAL_CACHE_ROOT / "dynamic_b16_per_image_operand_scale256.float32.npy"
)

SCHEMA_VERSION = "swin_dynamic_b16_per_image_operand_scale_v1"
PROPAGATION_SCHEMA = "swin_dynamic_b16_propagation_profile_v1"
PROBE_BUDGET_PROPAGATION_SCHEMA = "probe_budget_application_propagation_profile_v1"
REFERENCE_IMAGE_COUNT = 256
MAX_IMAGE_COUNT = 1000
EXPECTED_NODE_COUNT = 72
LUT_CELLS = BYTE_CODES * BYTE_CODES


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must be an object: {path}")
    return value


def _ordered_strings_sha256(values: list[str]) -> str:
    encoded = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _index_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".index.npz")


def _metadata_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex}")


def _resource_estimate(
    image_count: int,
    node_count: int = EXPECTED_NODE_COUNT,
) -> dict[str, Any]:
    cells = image_count * node_count * LUT_CELLS
    raw_bytes = cells * np.dtype(np.float32).itemsize
    return {
        "shape": [image_count, node_count, LUT_CELLS],
        "cells": cells,
        "raw_bytes": raw_bytes,
        "raw_gib": raw_bytes / (1024.0**3),
        "observed_two_image_dense_support_fraction": [
            0.5476322174072266,
            0.5526777903238932,
        ],
        "storage_choice": (
            "dense float32; observed CSR values+uint16 indices save only about "
            "17 percent before row-pointer/format overhead"
        ),
        "profile_runtime_projection_seconds": [120, 240],
    }


def _validate_propagation_profile(
    path: Path,
    *,
    expected_node_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("propagation profile or sidecar is missing")
    metadata = _read_json(metadata_path)
    schema = str(metadata.get("schema_version", ""))
    if schema not in {PROPAGATION_SCHEMA, PROBE_BUDGET_PROPAGATION_SCHEMA}:
        raise ValueError("unsupported propagation profile schema")
    hashes = metadata.get("output_hashes")
    if not isinstance(hashes, dict) or hashes.get("sha256") != sha256_file(path):
        raise ValueError("propagation profile checksum does not match its sidecar")
    forbidden_true = (
        "candidate_multiplier_luts_loaded",
        "candidate_runtime_bank_loaded",
        "canonical_targets_loaded",
        "per_design_runtime_observations_used",
    )
    if any(bool(metadata.get(name, True)) for name in forbidden_true):
        raise ValueError("propagation profile violates the static-only protocol")
    if int(metadata.get("candidate_multiplier_inference_count", -1)) != 0:
        raise ValueError("propagation profile used candidate multiplier inference")
    expected_label_usage = (
        "profile_grouping_only"
        if schema == PROBE_BUDGET_PROPAGATION_SCHEMA
        else "none"
    )
    if metadata.get("clean_label_usage") != expected_label_usage:
        raise ValueError("propagation profile has unexpected clean-label usage")

    image_index_key = str(
        metadata.get("image_index_key", "selected_true1000_indices")
    )
    expected_index_space = {
        "selected_true1000_indices": ("true1000", 1000),
        "selected_raw500_indices": ("raw500", 500),
    }
    if image_index_key not in expected_index_space:
        raise ValueError("propagation profile has an unsupported image index key")
    image_index_space, image_index_count = expected_index_space[image_index_key]
    if str(metadata.get("image_index_space", image_index_space)) != image_index_space:
        raise ValueError("propagation image index key and index space disagree")

    with np.load(path, allow_pickle=False) as profile:
        node_ids = np.asarray(profile["node_ids"]).astype(str).tolist()
        selected_local = np.asarray(profile[image_index_key], dtype=np.int64)
        selected_union = np.asarray(profile["selected_union_positions"], dtype=np.int64)
        candidate_count = int(
            np.asarray(profile["candidate_multiplier_inference_count"]).item()
        )
    if node_ids != expected_node_ids:
        raise ValueError("propagation and operand registry node orders differ")
    if selected_local.shape != selected_union.shape or selected_local.ndim != 1:
        raise ValueError("propagation selected-image arrays are not aligned")
    if len(selected_local) != int(metadata.get("image_count", -1)):
        raise ValueError("propagation image count differs from its selected arrays")
    if len(np.unique(selected_local)) != len(selected_local):
        raise ValueError("propagation selected image indices are not unique")
    if len(np.unique(selected_union)) != len(selected_union):
        raise ValueError("propagation selected union positions are not unique")
    if np.any(selected_local < 0) or np.any(selected_local >= image_index_count):
        raise ValueError(
            f"propagation {image_index_space} index is outside [0,{image_index_count})"
        )
    if np.any(selected_union < 0) or np.any(selected_union >= 1486):
        raise ValueError("propagation union position is outside [0,1486)")
    if metadata.get(image_index_key) != selected_local.tolist():
        raise ValueError("propagation local image order differs from its sidecar")
    if metadata.get("selected_union_positions") != selected_union.tolist():
        raise ValueError("propagation union image order differs from its sidecar")
    if candidate_count != 0:
        raise ValueError("embedded propagation candidate inference count is nonzero")
    return selected_local, selected_union, metadata


def _normalized_scale_surface(
    count: np.ndarray,
    scale_mass: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    count_values = np.asarray(count)
    scale_values = np.asarray(scale_mass, dtype=np.float64)
    if (
        count_values.shape != (BYTE_CODES, BYTE_CODES)
        or count_values.dtype != np.uint64
    ):
        raise ValueError("full-byte MAC-count surface has an invalid shape or dtype")
    if scale_values.shape != count_values.shape:
        raise ValueError("full-byte scale surface is not aligned to MAC counts")
    semantic_macs = int(count_values.sum(dtype=np.uint64))
    total_scale = float(scale_values.sum(dtype=np.float64))
    if semantic_macs <= 0 or not np.isfinite(total_scale) or total_scale <= 0.0:
        raise ValueError("per-image/node profile has no positive semantic scale mass")
    normalized = np.asarray(scale_values / total_scale, dtype=np.float32)
    if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise ValueError("normalized scale surface contains invalid values")
    constant_response = float(normalized.sum(dtype=np.float64))
    reconstruction_error = abs(constant_response - 1.0)
    if reconstruction_error > 2.0e-6:
        raise RuntimeError(
            "float32 normalized surface does not reconstruct constant +1: "
            f"{constant_response}"
        )
    return normalized.reshape(-1), semantic_macs, total_scale, reconstruction_error


def run_synthetic_checks() -> dict[str, Any]:
    inherited = run_fullbyte_synthetic_checks()
    count = np.zeros((BYTE_CODES, BYTE_CODES), dtype=np.uint64)
    scale = np.zeros_like(count, dtype=np.float64)
    observations = [
        (-128, 127, 0.125, -7),
        (-7, 3, 0.25, 11),
        (-7, 3, 0.50, 11),
        (8, 9, 1.25, -2),
        (127, -128, 0.375, 5),
    ]
    delta = np.zeros((BYTE_CODES, BYTE_CODES), dtype=np.float64)
    for qa, qb, product_scale, residual in observations:
        index = (qa + 128, qb + 128)
        count[index] += 1
        scale[index] += product_scale
        delta[index] = residual
    normalized, semantic_macs, total_scale, constant_error = _normalized_scale_surface(
        count, scale
    )
    observed = float(normalized.astype(np.float64) @ delta.reshape(-1))
    expected = float(
        sum(product_scale * residual for _, _, product_scale, residual in observations)
        / sum(product_scale for _, _, product_scale, _ in observations)
    )
    contraction_error = abs(observed - expected)
    if contraction_error > 2.0e-6:
        raise AssertionError(
            f"normalized candidate contraction differs from enumeration: {contraction_error}"
        )
    return {
        "inherited_fullbyte_checks": inherited,
        "normalized_constant_plus_one_error": constant_error,
        "normalized_candidate_contraction_error": contraction_error,
        "semantic_mac_count": semantic_macs,
        "total_qdq_scale_mass": total_scale,
    }


def _operand_for_image(
    operand: QuantizedInput,
    image_row: int,
    *,
    dynamic: dict[str, np.ndarray],
    batch_size: int,
    subset_cache: dict[str, np.ndarray],
) -> np.ndarray:
    if operand.static_value is not None:
        return operand.static_value
    if operand.name not in subset_cache:
        subset_cache[operand.name] = _dynamic_subset(
            dynamic[operand.name],
            np.asarray([image_row], dtype=np.int64),
            batch_size=batch_size,
            tensor_name=operand.name,
        )
    return subset_cache[operand.name]


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_path = args.output.expanduser().resolve()
    index_path = _index_path(output_path)
    metadata_path = _metadata_path(output_path)
    publication_paths = (output_path, index_path, metadata_path)
    if any(path.exists() for path in publication_paths) and not args.overwrite:
        raise FileExistsError("per-image operand profile exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = args.model.expanduser().resolve()
    registry_path = args.node_registry.expanduser().resolve()
    model, nodes, dynamic_names = _load_nodes(model_path, registry_path)
    node_count = len(nodes)
    if node_count < 1 or len(dynamic_names) < node_count:
        raise ValueError(
            "operand profile requires at least one dynamic q-code per node"
        )
    node_ids = [node.node_id for node in nodes]
    selected_local, selected_union, propagation_metadata = (
        _validate_propagation_profile(
            args.propagation_profile.expanduser().resolve(),
            expected_node_ids=node_ids,
        )
    )
    image_index_key = str(
        propagation_metadata.get("image_index_key", "selected_true1000_indices")
    )
    image_index_space = str(
        propagation_metadata.get("image_index_space", "true1000")
    )
    if args.image_limit:
        selected_local = selected_local[: int(args.image_limit)]
        selected_union = selected_union[: int(args.image_limit)]
    image_count = len(selected_local)
    if image_count < 1 or image_count > MAX_IMAGE_COUNT:
        raise ValueError("profile image count must be in [1,1000]")

    inputs_path = args.inputs.expanduser().resolve()
    inputs = np.load(inputs_path, mmap_mode="r", allow_pickle=False)
    if inputs.shape != (1486, 3, 224, 224) or inputs.dtype != np.float32:
        raise ValueError(
            f"frozen union input bank is invalid: {inputs.shape} {inputs.dtype}"
        )

    instrumentation = _instrumentation_contract(
        model_path, registry_path, dynamic_names
    )
    instrumented_path = _prepare_instrumented_model(
        model,
        output_path=args.instrumented_model.expanduser(),
        contract=instrumentation,
        overwrite=args.overwrite_instrumented,
    )
    static_b_cache = {
        node.node_id: _static_b_histograms(node.input_b)
        for node in nodes
        if node.input_b.is_static
    }

    temporary_output = _temporary_sibling(output_path)
    temporary_index = _temporary_sibling(index_path)
    temporary_metadata = _temporary_sibling(metadata_path)
    destination: np.memmap | None = None
    session = None
    started = time.time()
    try:
        destination = np.lib.format.open_memmap(
            temporary_output,
            mode="w+",
            dtype=np.float32,
            shape=(image_count, node_count, LUT_CELLS),
        )
        semantic_mac_count = np.empty((image_count, node_count), dtype=np.uint64)
        total_qdq_scale_mass = np.empty((image_count, node_count), dtype=np.float64)
        normalization_error = np.empty((image_count, node_count), dtype=np.float64)
        observed_shapes: dict[str, list[int]] = {}

        session, provider = create_cuda_session(
            instrumented_path,
            device_id=int(args.device_id),
            fallback_parallelism=1,
            session_profile="throughput",
        )
        if provider != "CUDAExecutionProvider":
            raise RuntimeError(
                f"per-image operand profile requires CUDA, got {provider}"
            )
        input_name = str(session.get_inputs()[0].name)
        available_outputs = {str(item.name) for item in session.get_outputs()}
        missing = sorted(set(dynamic_names).difference(available_outputs))
        if missing:
            raise ValueError(
                f"instrumented session is missing q-code outputs: {missing[:3]}"
            )

        for start in range(0, image_count, int(args.batch_size)):
            stop = min(start + int(args.batch_size), image_count)
            batch = np.ascontiguousarray(
                inputs[selected_union[start:stop]], dtype=np.float32
            )
            values = session.run(dynamic_names, {input_name: batch})
            dynamic = {
                name: np.asarray(value)
                for name, value in zip(dynamic_names, values, strict=True)
            }
            for name, value in dynamic.items():
                shape = list(map(int, value.shape))
                previous = observed_shapes.setdefault(name, shape)
                if previous[1:] != shape[1:]:
                    raise RuntimeError(f"q-code non-batch shape changed for {name}")

            for local_image in range(stop - start):
                profile_image = start + local_image
                subset_cache: dict[str, np.ndarray] = {}
                for node_index, node in enumerate(nodes):
                    count, scale_mass, _scale_square_mass = _pair_histogram_fullbyte(
                        node,
                        _operand_for_image(
                            node.input_a,
                            local_image,
                            dynamic=dynamic,
                            batch_size=stop - start,
                            subset_cache=subset_cache,
                        ),
                        _operand_for_image(
                            node.input_b,
                            local_image,
                            dynamic=dynamic,
                            batch_size=stop - start,
                            subset_cache=subset_cache,
                        ),
                        static_b_cache=static_b_cache,
                        row_block_size=int(args.row_block_size),
                    )
                    normalized, macs, scale_total, reconstruction_error = (
                        _normalized_scale_surface(count, scale_mass)
                    )
                    destination[profile_image, node_index] = normalized
                    semantic_mac_count[profile_image, node_index] = macs
                    total_qdq_scale_mass[profile_image, node_index] = scale_total
                    normalization_error[profile_image, node_index] = (
                        reconstruction_error
                    )
            destination.flush()
            print(
                json.dumps(
                    {
                        "profiled_images": stop,
                        "total_images": image_count,
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        del destination
        destination = None
        with temporary_index.open("xb") as handle:
            index_arrays = {
                "schema_version": np.asarray(SCHEMA_VERSION),
                "node_ids": np.asarray(node_ids),
                image_index_key: np.asarray(selected_local, dtype=np.int64),
                "selected_union_positions": np.asarray(selected_union, dtype=np.int64),
                "semantic_mac_count": semantic_mac_count,
                "total_qdq_scale_mass": total_qdq_scale_mass,
                "float32_constant_probe_reconstruction_error": normalization_error,
            }
            np.savez_compressed(handle, **index_arrays)
            handle.flush()
            os.fsync(handle.fileno())

        output_sha256 = sha256_file(temporary_output)
        index_sha256 = sha256_file(temporary_index)
        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_complete": True,
            "publication_contract": "json_sidecar_is_completion_marker_v1",
            "output": str(output_path),
            "output_sha256": output_sha256,
            "index": str(index_path),
            "index_sha256": index_sha256,
            "shape": [image_count, node_count, LUT_CELLS],
            "dtype": "float32",
            "profile_value": "normalized_qdq_scale_weight",
            "normalization": (
                "H[i,n,qa,qb] = qdq_scale_mass[i,n,qa,qb] / "
                "sum_qa_qb qdq_scale_mass[i,n,qa,qb]"
            ),
            "constant_plus_one_amplitude": 1.0,
            "constant_probe_reconstruction_max_abs_error": float(
                normalization_error.max(initial=0.0)
            ),
            "image_count": image_count,
            "formal_256_image_profile": image_count == REFERENCE_IMAGE_COUNT,
            "node_count": node_count,
            "node_order_sha256": _ordered_strings_sha256(node_ids),
            "image_index_key": image_index_key,
            "image_index_space": image_index_space,
            image_index_key: selected_local.tolist(),
            f"{image_index_key}_sha256": sha256_array(selected_local),
            "selected_union_positions": selected_union.tolist(),
            "selected_union_positions_sha256": sha256_array(selected_union),
            "semantic_mac_count_sha256": sha256_array(semantic_mac_count),
            "total_qdq_scale_mass_sha256": sha256_array(total_qdq_scale_mass),
            "model": str(model_path),
            "model_sha1": _sha1_file(model_path),
            "node_registry": str(registry_path),
            "node_registry_sha256": sha256_file(registry_path),
            "instrumented_model": str(instrumented_path.resolve()),
            "instrumentation_contract": instrumentation,
            "dynamic_qcode_output_count": len(dynamic_names),
            "dynamic_qcode_observed_shapes": observed_shapes,
            "inputs": str(inputs_path),
            "inputs_sha256": sha256_file(inputs_path),
            "propagation_profile": str(args.propagation_profile.expanduser().resolve()),
            "propagation_profile_sha256": sha256_file(
                args.propagation_profile.expanduser().resolve()
            ),
            "propagation_selection_sha256": propagation_metadata["selection_sha256"],
            "semantic_zero_mask": (
                "exclude qA == zpA or qB == zpB before scale-weight histogramming"
            ),
            "semantic_zero_mask_applied_before_fullbyte_joint": True,
            "exact_fullbyte_joint_before_float32_storage": True,
            "exact_static_per_output_channel_scale_mass": True,
            "dynamic_broadcast_prefix_preserved_through_joint_contraction": True,
            "lut_domain": "signed_int8_raw_code_plus_128",
            "candidate_multiplier_inference_count": 0,
            "candidate_multiplier_luts_loaded": False,
            "candidate_runtime_bank_loaded": False,
            "per_design_runtime_observations_used": False,
            "per_design_targets_loaded": False,
            "validation_runtime_or_correctness_loaded": False,
            "test_targets_loaded": False,
            "clean_application_runtime_calls": (
                (image_count + int(args.batch_size) - 1) // int(args.batch_size)
            ),
            "deployment_reuse": "frozen application profile shared by every multiplier",
            "provider": provider,
            "device_id": int(args.device_id),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "batch_size": int(args.batch_size),
            "row_block_size": int(args.row_block_size),
            "resource_estimate": _resource_estimate(image_count, node_count),
            "elapsed_seconds": time.time() - started,
        }
        with temporary_metadata.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_output, output_path)
        os.replace(temporary_index, index_path)
        os.replace(temporary_metadata, metadata_path)
        return result
    finally:
        if session is not None:
            session.close()
        if destination is not None:
            del destination
        for temporary in (temporary_output, temporary_index, temporary_metadata):
            if temporary.exists():
                temporary.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--node-registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    result.add_argument(
        "--instrumented-model", type=Path, default=DEFAULT_INSTRUMENTED_MODEL
    )
    result.add_argument(
        "--propagation-profile", type=Path, default=DEFAULT_PROPAGATION_PROFILE
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--row-block-size", type=int, default=8192)
    result.add_argument("--image-limit", type=int, default=0)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--overwrite-instrumented", action="store_true")
    result.add_argument("--self-test-only", action="store_true")
    result.add_argument("--estimate-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.batch_size < 1 or args.row_block_size < 1:
        raise SystemExit("--batch-size and --row-block-size must be positive")
    if args.image_limit < 0 or args.image_limit > MAX_IMAGE_COUNT:
        raise SystemExit("--image-limit must be zero or in [1,1000]")
    if args.self_test_only:
        print(json.dumps(run_synthetic_checks(), indent=2, sort_keys=True))
        return 0
    if args.estimate_only:
        count = int(args.image_limit) if args.image_limit else REFERENCE_IMAGE_COUNT
        registry = _read_json(args.node_registry.expanduser().resolve())
        node_count = int(registry.get("node_count", EXPECTED_NODE_COUNT))
        print(
            json.dumps(_resource_estimate(count, node_count), indent=2, sort_keys=True)
        )
        return 0
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
