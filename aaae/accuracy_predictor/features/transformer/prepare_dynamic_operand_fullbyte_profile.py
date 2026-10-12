#!/usr/bin/env python
"""Build an exact full-byte dynamic-B16 Swin operand profile.

This is a one-time clean-application profile.  It observes only native int8
Q/DQ operands from the exact dynamic Swin graph.  For every profiled MatMul it
stores the complete ``(qA + 128, qB + 128)`` joint distribution after applying
the runtime semantic-zero rule.  It never loads or executes a candidate
multiplier.

The profile includes both the MAC count and first/second Q/DQ-scale mass.  The
second scale moment is needed to contract a candidate LUT into a physically
meaningful RMS accumulator-error response when static B uses per-output-channel
scales.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer.common import (
    FORMAL_CACHE_ROOT,
    sha256_array,
    sha256_file,
    write_json,
)
from aaae.accuracy_predictor.features.transformer.prepare_dynamic_operand_profile import (
    DEFAULT_EXACT_BASELINE,
    DEFAULT_EXACT_META,
    DEFAULT_INPUTS,
    DEFAULT_INSTRUMENTED_MODEL,
    DEFAULT_MODEL,
    DEFAULT_REGISTRY,
    NodeOperands,
    QuantizedInput,
    _dynamic_subset,
    _instrumentation_contract,
    _load_nodes,
    _prepare_instrumented_model,
    _sha1_file,
    _top1_groups,
)
from aaae.inference_runtime.transformer.ort_session import create_cuda_session

DEFAULT_OUTPUT = FORMAL_CACHE_ROOT / "dynamic_b16_operand_fullbyte_profile_top1.npz"
PROFILE_SCHEMA = "swin_dynamic_b16_operand_fullbyte_profile_v1"
HISTOGRAM_SHAPE = (10, 72, 256, 256)
BYTE_CODES = 256


def _ordered_strings_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _profile_resource_estimate(row_block_size: int) -> dict[str, Any]:
    cells = int(np.prod(HISTOGRAM_SHAPE, dtype=np.int64))
    array_bytes = cells * np.dtype(np.uint64).itemsize
    block_bytes = 2 * row_block_size * BYTE_CODES * np.dtype(np.float64).itemsize
    return {
        "histogram_shape": list(HISTOGRAM_SHAPE),
        "histogram_cells": cells,
        "bytes_per_uint64_or_float64_histogram": array_bytes,
        "mib_per_uint64_or_float64_histogram": array_bytes / (1024.0**2),
        "raw_bytes_for_count_scale_and_scale_square": 3 * array_bytes,
        "raw_gib_for_count_scale_and_scale_square": 3 * array_bytes / (1024.0**3),
        "dense_row_count_working_set_mib": block_bytes / (1024.0**2),
        "compressed_output_projection_gib": [0.35, 1.05],
        "runtime_projection": (
            "The existing coarse true1000 profile took 1129 s. Full-byte CPU "
            "contingency contraction is projected at roughly 25-70 minutes on "
            "this host, plus 5-20 minutes for NPZ compression; measure an "
            "image-limited run before scheduling the full run."
        ),
    }


def _byte_ids(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int8:
        raise ValueError(f"q-code values must be int8, got {array.dtype}")
    return np.asarray(array, dtype=np.int16) + 128


def _category_counts_by_row(
    value: np.ndarray,
    zero_point: int,
) -> np.ndarray:
    """Count full-byte categories independently for every row."""

    array = np.asarray(value)
    if array.dtype != np.int8 or array.ndim != 2:
        raise ValueError(f"row category input must be rank-two int8, got {array.shape}")
    row_count, width = array.shape
    ids = _byte_ids(array)
    valid = array != int(zero_point)
    offsets = (np.arange(row_count, dtype=np.int64) * BYTE_CODES)[:, None]
    linear = offsets + ids
    counts = np.bincount(linear[valid], minlength=row_count * BYTE_CODES).reshape(
        row_count, BYTE_CODES
    )
    if counts.dtype != np.int64 or int(counts.sum()) != int(valid.sum()):
        raise RuntimeError("full-byte row category counts do not conserve values")
    if width == 0:
        raise ValueError("q-code category input has an empty reduction dimension")
    return np.asarray(counts, dtype=np.uint64)


def _category_counts_by_column(
    value: np.ndarray,
    zero_point: int,
    *,
    sample_block_size: int = 65536,
) -> np.ndarray:
    """Count full-byte categories per K column without a large transpose."""

    array = np.asarray(value)
    if array.dtype != np.int8 or array.ndim != 2:
        raise ValueError(
            f"column category input must be rank-two int8, got {array.shape}"
        )
    sample_count, column_count = array.shape
    if sample_count < 1 or column_count < 1:
        raise ValueError("q-code category input has an empty dimension")
    result = np.zeros((column_count, BYTE_CODES), dtype=np.uint64)
    offsets = np.arange(column_count, dtype=np.int64) * BYTE_CODES
    for start in range(0, sample_count, sample_block_size):
        block = array[start : start + sample_block_size]
        ids = _byte_ids(block)
        valid = block != int(zero_point)
        linear = ids + offsets[None, :]
        counts = np.bincount(
            linear[valid], minlength=column_count * BYTE_CODES
        ).reshape(column_count, BYTE_CODES)
        result += np.asarray(counts, dtype=np.uint64)
    if int(result.sum()) != int(np.count_nonzero(array != int(zero_point))):
        raise RuntimeError("full-byte column category counts do not conserve values")
    return result


def _static_b_histograms(
    operand: QuantizedInput,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return count, scale mass, and squared-scale mass for static B per K."""

    if operand.static_value is None:
        raise ValueError("static B histograms requested for a dynamic operand")
    value = np.asarray(operand.static_value)
    if value.dtype != np.int8 or value.ndim != 2:
        raise ValueError(f"static MatMul B must be rank-two int8, got {value.shape}")
    k_dim, output_columns = value.shape
    ids = _byte_ids(value)
    valid = value != int(operand.zero_point)
    offsets = (np.arange(k_dim, dtype=np.int64) * BYTE_CODES)[:, None]
    linear = offsets + ids
    count = np.bincount(linear[valid], minlength=k_dim * BYTE_CODES).reshape(
        k_dim, BYTE_CODES
    )
    if operand.scale.size == 1:
        scales = np.full(output_columns, float(operand.scale[0]), dtype=np.float64)
    elif operand.scale.shape == (output_columns,):
        scales = np.asarray(operand.scale, dtype=np.float64)
    else:
        raise ValueError(
            f"static B scale shape {operand.scale.shape} does not match N={output_columns}"
        )
    scale_values = np.broadcast_to(scales[None, :], value.shape)
    scale_mass = np.bincount(
        linear[valid],
        weights=scale_values[valid],
        minlength=k_dim * BYTE_CODES,
    ).reshape(k_dim, BYTE_CODES)
    scale_square_mass = np.bincount(
        linear[valid],
        weights=(scale_values * scale_values)[valid],
        minlength=k_dim * BYTE_CODES,
    ).reshape(k_dim, BYTE_CODES)
    count_u64 = np.asarray(count, dtype=np.uint64)
    if int(count_u64.sum()) != int(valid.sum()):
        raise RuntimeError("static B full-byte histograms do not conserve values")
    return (
        count_u64,
        np.asarray(scale_mass, dtype=np.float64),
        np.asarray(scale_square_mass, dtype=np.float64),
    )


def _exact_integer_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Use float64 BLAS while retaining exact integer results below 2**53."""

    product = np.asarray(left, dtype=np.float64).T @ np.asarray(right, dtype=np.float64)
    rounded = np.rint(product)
    if not np.array_equal(product, rounded):
        difference = float(np.max(np.abs(product - rounded)))
        raise RuntimeError(
            f"integer contingency GEMM lost exactness: max diff {difference}"
        )
    if float(rounded.max(initial=0.0)) >= float(2**53):
        raise OverflowError("full-byte contingency count reached float64 integer limit")
    return rounded


def _dynamic_joint_histogram(
    a_rows: np.ndarray,
    b_rows: np.ndarray,
    *,
    zero_point_a: int,
    zero_point_b: int,
    row_block_size: int,
) -> np.ndarray:
    if a_rows.ndim != 2 or b_rows.ndim != 2 or len(a_rows) != len(b_rows):
        raise ValueError(
            f"dynamic category rows are misaligned: A={a_rows.shape}, B={b_rows.shape}"
        )
    joint = np.zeros((BYTE_CODES, BYTE_CODES), dtype=np.float64)
    expected_total = 0
    for start in range(0, len(a_rows), row_block_size):
        stop = min(start + row_block_size, len(a_rows))
        a_count = _category_counts_by_row(a_rows[start:stop], zero_point_a)
        b_count = _category_counts_by_row(b_rows[start:stop], zero_point_b)
        joint += _exact_integer_matmul(a_count, b_count)
        valid_a = a_count.sum(axis=1, dtype=np.uint64)
        valid_b = b_count.sum(axis=1, dtype=np.uint64)
        expected_total += int(np.sum(valid_a * valid_b, dtype=np.uint64))
    rounded = np.rint(joint)
    if not np.array_equal(joint, rounded):
        raise RuntimeError("dynamic full-byte joint histogram is not integral")
    result = np.asarray(rounded, dtype=np.uint64)
    if int(result.sum(dtype=np.uint64)) != expected_total:
        raise RuntimeError("dynamic full-byte joint histogram does not conserve MACs")
    return result


def _pair_histogram_fullbyte(
    node: NodeOperands,
    input_a: np.ndarray,
    input_b: np.ndarray,
    *,
    static_b_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    row_block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an exact semantic-MAC full-byte joint for one MatMul/group."""

    a = np.asarray(input_a)
    b = np.asarray(input_b)
    if a.dtype != np.int8 or b.dtype != np.int8:
        raise ValueError(f"MatMul operands at {node.node_id} must both be int8")
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"MatMul operand shape mismatch at {node.node_id}: A={a.shape}, B={b.shape}"
        )
    a_scale = float(node.input_a.scale[0])
    if node.input_b.is_static:
        k_dim = a.shape[-1]
        a_flat = a.reshape(-1, k_dim)
        a_count = _category_counts_by_column(a_flat, node.input_a.zero_point)
        b_count, b_scale, b_scale_square = static_b_cache[node.node_id]
        if b_count.shape[0] != k_dim:
            raise ValueError(f"static B K dimension changed at {node.node_id}")
        count_float = _exact_integer_matmul(a_count, b_count)
        count = np.asarray(count_float, dtype=np.uint64)
        scale_mass = a_scale * (np.asarray(a_count, dtype=np.float64).T @ b_scale)
        scale_square_mass = (a_scale * a_scale) * (
            np.asarray(a_count, dtype=np.float64).T @ b_scale_square
        )
    else:
        prefix = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a_view = np.broadcast_to(a, (*prefix, *a.shape[-2:])).reshape(
            -1, a.shape[-2], a.shape[-1]
        )
        b_view = np.broadcast_to(b, (*prefix, *b.shape[-2:])).reshape(
            -1, b.shape[-2], b.shape[-1]
        )
        # Keep each broadcast-prefix/K row aligned.  Aggregating across this
        # dimension before the outer product would mix unrelated windows/heads.
        a_rows = np.swapaxes(a_view, -1, -2).reshape(-1, a.shape[-2])
        b_rows = b_view.reshape(-1, b.shape[-1])
        count = _dynamic_joint_histogram(
            a_rows,
            b_rows,
            zero_point_a=node.input_a.zero_point,
            zero_point_b=node.input_b.zero_point,
            row_block_size=row_block_size,
        )
        product_scale = a_scale * float(node.input_b.scale[0])
        count_float = np.asarray(count, dtype=np.float64)
        scale_mass = count_float * product_scale
        scale_square_mass = count_float * (product_scale * product_scale)
    if count.shape != (BYTE_CODES, BYTE_CODES):
        raise RuntimeError("full-byte pair histogram has an invalid shape")
    if not np.isfinite(scale_mass).all() or not np.isfinite(scale_square_mass).all():
        raise RuntimeError("full-byte scale histogram contains non-finite values")
    if np.any(scale_mass < 0.0) or np.any(scale_square_mass < 0.0):
        raise RuntimeError("Q/DQ scales must produce non-negative histogram mass")
    if not np.array_equal(count == 0, scale_mass == 0.0):
        raise RuntimeError("count and Q/DQ-scale histogram support differ")
    return count, scale_mass, scale_square_mass


def _derive_all_group(
    pair_count: np.ndarray,
    scale_mass: np.ndarray,
    scale_square_mass: np.ndarray,
) -> None:
    """Derive group 0 from the exact non-overlapping groups 1..N in place."""

    if pair_count.ndim != 4 or pair_count.shape[0] < 2:
        raise ValueError("grouped pair-count profile must have at least two groups")
    if (
        scale_mass.shape != pair_count.shape
        or scale_square_mass.shape != pair_count.shape
    ):
        raise ValueError("grouped full-byte profile arrays are not shape aligned")
    pair_count[0] = pair_count[1:].sum(axis=0, dtype=np.uint64)
    scale_mass[0] = scale_mass[1:].sum(axis=0, dtype=np.float64)
    scale_square_mass[0] = scale_square_mass[1:].sum(axis=0, dtype=np.float64)


def _operand_value_for_group(
    operand: QuantizedInput,
    group: int,
    *,
    dynamic: dict[str, np.ndarray],
    group_rows: dict[int, np.ndarray],
    batch_size: int,
    subset_cache: dict[tuple[str, int], np.ndarray],
) -> np.ndarray:
    if operand.static_value is not None:
        return operand.static_value
    key = (operand.name, group)
    if key not in subset_cache:
        subset_cache[key] = _dynamic_subset(
            dynamic[operand.name],
            group_rows[group],
            batch_size=batch_size,
            tensor_name=operand.name,
        )
    return subset_cache[key]


def _brute_force_pair_histogram(
    a: np.ndarray,
    b: np.ndarray,
    *,
    zero_point_a: int,
    zero_point_b: int,
    scale_a: float,
    scale_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_view = np.broadcast_to(a, (*prefix, *a.shape[-2:]))
    b_view = np.broadcast_to(b, (*prefix, *b.shape[-2:]))
    scales_b = np.asarray(scale_b, dtype=np.float64).reshape(-1)
    if scales_b.size == 1:
        scales_b = np.full(b.shape[-1], float(scales_b[0]), dtype=np.float64)
    if scales_b.shape != (b.shape[-1],):
        raise ValueError("brute-force B scale shape is invalid")
    count = np.zeros((BYTE_CODES, BYTE_CODES), dtype=np.uint64)
    scale_mass = np.zeros_like(count, dtype=np.float64)
    scale_square_mass = np.zeros_like(count, dtype=np.float64)
    for prefix_index in np.ndindex(prefix):
        for m_index in range(a.shape[-2]):
            for k_index in range(a.shape[-1]):
                qa = int(a_view[prefix_index + (m_index, k_index)])
                if qa == zero_point_a:
                    continue
                for n_index in range(b.shape[-1]):
                    qb = int(b_view[prefix_index + (k_index, n_index)])
                    if qb == zero_point_b:
                        continue
                    index = (qa + 128, qb + 128)
                    product_scale = float(scale_a) * float(scales_b[n_index])
                    count[index] += 1
                    scale_mass[index] += product_scale
                    scale_square_mass[index] += product_scale * product_scale
    return count, scale_mass, scale_square_mass


def run_synthetic_checks() -> dict[str, Any]:
    """Brute-force exactness checks for dynamic and static broadcast paths."""

    rng = np.random.default_rng(20260727)
    a_dynamic = rng.integers(-7, 8, size=(2, 1, 3, 4), dtype=np.int8)
    b_dynamic = rng.integers(-7, 8, size=(1, 3, 4, 5), dtype=np.int8)
    a_dynamic[..., 0, 0] = -3
    b_dynamic[..., 0, 0] = 2
    dynamic_node = NodeOperands(
        node_id="synthetic_dynamic",
        input_a=QuantizedInput(
            name="a",
            static_value=None,
            scale=np.asarray([0.125]),
            zero_point=-3,
            axis=None,
        ),
        input_b=QuantizedInput(
            name="b",
            static_value=None,
            scale=np.asarray([0.375]),
            zero_point=2,
            axis=None,
        ),
    )
    observed_dynamic = _pair_histogram_fullbyte(
        dynamic_node,
        a_dynamic,
        b_dynamic,
        static_b_cache={},
        row_block_size=3,
    )
    expected_dynamic = _brute_force_pair_histogram(
        a_dynamic,
        b_dynamic,
        zero_point_a=-3,
        zero_point_b=2,
        scale_a=0.125,
        scale_b=np.asarray([0.375]),
    )

    a_static = rng.integers(-8, 9, size=(2, 3, 4), dtype=np.int8)
    b_static = rng.integers(-8, 9, size=(4, 5), dtype=np.int8)
    a_static[..., 1] = 1
    b_static[2, 3] = -2
    b_scales = np.asarray([0.2, 0.3, 0.5, 0.7, 1.1], dtype=np.float64)
    static_b = QuantizedInput(
        name="b_static",
        static_value=b_static,
        scale=b_scales,
        zero_point=-2,
        axis=1,
    )
    static_node = NodeOperands(
        node_id="synthetic_static",
        input_a=QuantizedInput(
            name="a",
            static_value=None,
            scale=np.asarray([0.25]),
            zero_point=1,
            axis=None,
        ),
        input_b=static_b,
    )
    observed_static = _pair_histogram_fullbyte(
        static_node,
        a_static,
        b_static,
        static_b_cache={"synthetic_static": _static_b_histograms(static_b)},
        row_block_size=2,
    )
    expected_static = _brute_force_pair_histogram(
        a_static,
        b_static,
        zero_point_a=1,
        zero_point_b=-2,
        scale_a=0.25,
        scale_b=b_scales,
    )

    scalar_static_b = QuantizedInput(
        name="b_static_scalar",
        static_value=b_static,
        scale=np.asarray([0.625], dtype=np.float64),
        zero_point=-2,
        axis=None,
    )
    scalar_static_node = NodeOperands(
        node_id="synthetic_static_scalar",
        input_a=static_node.input_a,
        input_b=scalar_static_b,
    )
    observed_static_scalar = _pair_histogram_fullbyte(
        scalar_static_node,
        a_static,
        b_static,
        static_b_cache={
            "synthetic_static_scalar": _static_b_histograms(scalar_static_b)
        },
        row_block_size=2,
    )
    expected_static_scalar = _brute_force_pair_histogram(
        a_static,
        b_static,
        zero_point_a=1,
        zero_point_b=-2,
        scale_a=0.25,
        scale_b=np.asarray([0.625]),
    )

    results: dict[str, Any] = {}
    for label, observed, expected in (
        ("dynamic_dynamic_broadcast", observed_dynamic, expected_dynamic),
        ("dynamic_static_per_channel", observed_static, expected_static),
        ("dynamic_static_scalar", observed_static_scalar, expected_static_scalar),
    ):
        count_ok = np.array_equal(observed[0], expected[0])
        scale_diff = float(np.max(np.abs(observed[1] - expected[1])))
        square_diff = float(np.max(np.abs(observed[2] - expected[2])))
        if not count_ok or scale_diff > 1e-12 or square_diff > 1e-12:
            raise AssertionError(
                f"{label} failed: count={count_ok}, scale={scale_diff}, square={square_diff}"
            )
        results[label] = {
            "count_exact": count_ok,
            "scale_weight_max_abs_difference": scale_diff,
            "scale_square_weight_max_abs_difference": square_diff,
            "semantic_mac_count": int(observed[0].sum()),
        }
    partition_count = np.zeros((3, 1, BYTE_CODES, BYTE_CODES), dtype=np.uint64)
    partition_scale = np.zeros_like(partition_count, dtype=np.float64)
    partition_square = np.zeros_like(partition_count, dtype=np.float64)
    partition_count[1, 0] = observed_dynamic[0]
    partition_scale[1, 0] = observed_dynamic[1]
    partition_square[1, 0] = observed_dynamic[2]
    partition_count[2, 0] = observed_static[0]
    partition_scale[2, 0] = observed_static[1]
    partition_square[2, 0] = observed_static[2]
    _derive_all_group(partition_count, partition_scale, partition_square)
    expected_partition_count = observed_dynamic[0] + observed_static[0]
    expected_partition_scale = observed_dynamic[1] + observed_static[1]
    expected_partition_square = observed_dynamic[2] + observed_static[2]
    if not np.array_equal(partition_count[0, 0], expected_partition_count):
        raise AssertionError("derived all-group count is not an exact partition sum")
    if not np.array_equal(partition_scale[0, 0], expected_partition_scale):
        raise AssertionError("derived all-group scale mass is not a partition sum")
    if not np.array_equal(partition_square[0, 0], expected_partition_square):
        raise AssertionError(
            "derived all-group scale-square mass is not a partition sum"
        )
    results["derived_all_group_partition"] = {
        "count_exact": True,
        "scale_weight_exact": True,
        "scale_square_weight_exact": True,
    }
    return results


def _validate_profile_arrays(
    pair_count: np.ndarray,
    scale_mass: np.ndarray,
    scale_square_mass: np.ndarray,
    image_counts: np.ndarray,
) -> None:
    if pair_count.shape != HISTOGRAM_SHAPE or pair_count.dtype != np.uint64:
        raise RuntimeError("full-byte count profile has an invalid shape or dtype")
    for name, values in (
        ("scale", scale_mass),
        ("scale-square", scale_square_mass),
    ):
        if values.shape != HISTOGRAM_SHAPE or values.dtype != np.float64:
            raise RuntimeError(
                f"full-byte {name} profile has an invalid shape or dtype"
            )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise RuntimeError(f"full-byte {name} profile has invalid values")
    totals = pair_count.sum(axis=(2, 3), dtype=np.uint64)
    populated = image_counts > 0
    if np.any(totals[populated] == 0):
        raise RuntimeError("a populated group/node has zero semantic MAC mass")
    for node_index in range(HISTOGRAM_SHAPE[1]):
        expected_count = pair_count[1:, node_index].sum(axis=0, dtype=np.uint64)
        if not np.array_equal(pair_count[0, node_index], expected_count):
            raise RuntimeError(f"group count partition failed at node {node_index}")
        for name, values in (
            ("scale", scale_mass),
            ("scale-square", scale_square_mass),
        ):
            expected = values[1:, node_index].sum(axis=0, dtype=np.float64)
            if not np.allclose(values[0, node_index], expected, rtol=2e-13, atol=1e-15):
                raise RuntimeError(
                    f"group {name} partition failed at node {node_index}"
                )
        support = pair_count[:, node_index] != 0
        if not np.array_equal(support, scale_mass[:, node_index] != 0.0):
            raise RuntimeError(f"count/scale support mismatch at node {node_index}")
        if not np.array_equal(support, scale_square_mass[:, node_index] != 0.0):
            raise RuntimeError(
                f"count/scale-square support mismatch at node {node_index}"
            )


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    if (output.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("full-byte operand profile exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model.expanduser().resolve()
    registry_path = args.node_registry.expanduser().resolve()
    model, nodes, dynamic_names = _load_nodes(model_path, registry_path)
    if len(nodes) != 72 or len(dynamic_names) != 96:
        raise ValueError(
            f"expected 72 nodes and 96 dynamic q-code tensors, got {len(nodes)}, {len(dynamic_names)}"
        )
    node_ids = [node.node_id for node in nodes]
    instrumentation = _instrumentation_contract(
        model_path, registry_path, dynamic_names
    )
    instrumented = _prepare_instrumented_model(
        model,
        output_path=args.instrumented_model.expanduser(),
        contract=instrumentation,
        overwrite=args.overwrite_instrumented,
    )

    inputs_path = args.inputs.expanduser().resolve()
    inputs = np.load(inputs_path, mmap_mode="r", allow_pickle=False)
    if inputs.shape != (1486, 3, 224, 224) or inputs.dtype != np.float32:
        raise ValueError(
            f"frozen union input cache is invalid: {inputs.shape} {inputs.dtype}"
        )
    exact_baseline_path = args.exact_baseline.expanduser().resolve()
    with np.load(exact_baseline_path, allow_pickle=False) as exact:
        positions = np.asarray(exact["true1000_positions"], dtype=np.int64)
    if positions.shape != (1000,) or len(np.unique(positions)) != 1000:
        raise ValueError("true1000 union positions are invalid")
    exact_meta_path = args.exact_meta.expanduser().resolve()
    group_assignment, group_names, group_metadata = _top1_groups(exact_meta_path)
    image_count = min(int(args.image_limit), 1000) if args.image_limit else 1000
    positions = positions[:image_count]
    group_assignment = group_assignment[:image_count]
    image_counts = np.zeros(10, dtype=np.int64)
    image_counts[0] = image_count
    for group in range(1, 10):
        image_counts[group] = int(np.sum(group_assignment == group))

    pair_count = np.zeros(HISTOGRAM_SHAPE, dtype=np.uint64)
    scale_mass = np.zeros(HISTOGRAM_SHAPE, dtype=np.float64)
    scale_square_mass = np.zeros(HISTOGRAM_SHAPE, dtype=np.float64)
    static_b_cache = {
        node.node_id: _static_b_histograms(node.input_b)
        for node in nodes
        if node.input_b.is_static
    }

    session, provider = create_cuda_session(
        instrumented,
        device_id=int(args.device_id),
        fallback_parallelism=1,
        session_profile="throughput",
    )
    input_name = str(session.get_inputs()[0].name)
    available_outputs = {str(item.name) for item in session.get_outputs()}
    missing_outputs = sorted(set(dynamic_names).difference(available_outputs))
    if missing_outputs:
        session.close()
        raise ValueError(
            f"instrumented session is missing outputs: {missing_outputs[:3]}"
        )

    started = time.time()
    observed_shapes: dict[str, list[int]] = {}
    try:
        for start in range(0, image_count, int(args.batch_size)):
            stop = min(start + int(args.batch_size), image_count)
            batch = np.ascontiguousarray(
                inputs[positions[start:stop]], dtype=np.float32
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
                    raise RuntimeError(
                        f"dynamic tensor non-batch shape changed: {name}"
                    )
            batch_groups = group_assignment[start:stop]
            group_rows: dict[int, np.ndarray] = {}
            for group in np.unique(batch_groups):
                group_rows[int(group)] = np.flatnonzero(batch_groups == group)
            subset_cache: dict[tuple[str, int], np.ndarray] = {}

            for group, local_rows in group_rows.items():
                if len(local_rows) == 0:
                    continue
                for node_index, node in enumerate(nodes):
                    count, scaled, scaled_square = _pair_histogram_fullbyte(
                        node,
                        _operand_value_for_group(
                            node.input_a,
                            group,
                            dynamic=dynamic,
                            group_rows=group_rows,
                            batch_size=stop - start,
                            subset_cache=subset_cache,
                        ),
                        _operand_value_for_group(
                            node.input_b,
                            group,
                            dynamic=dynamic,
                            group_rows=group_rows,
                            batch_size=stop - start,
                            subset_cache=subset_cache,
                        ),
                        static_b_cache=static_b_cache,
                        row_block_size=int(args.row_block_size),
                    )
                    previous = pair_count[group, node_index]
                    updated = previous + count
                    if np.any(updated < previous):
                        raise OverflowError("uint64 pair-count accumulator overflowed")
                    pair_count[group, node_index] = updated
                    scale_mass[group, node_index] += scaled
                    scale_square_mass[group, node_index] += scaled_square
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
    finally:
        session.close()

    _derive_all_group(pair_count, scale_mass, scale_square_mass)
    _validate_profile_arrays(pair_count, scale_mass, scale_square_mass, image_counts)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    save = np.savez if args.uncompressed else np.savez_compressed
    with temporary.open("wb") as handle:
        save(
            handle,
            schema_version=np.asarray(PROFILE_SCHEMA),
            group_names=np.asarray(group_names),
            group_image_counts=image_counts,
            node_ids=np.asarray(node_ids),
            q_input_a_names=np.asarray([node.input_a.name for node in nodes]),
            q_input_b_names=np.asarray([node.input_b.name for node in nodes]),
            input_a_zero_points=np.asarray(
                [node.input_a.zero_point for node in nodes], dtype=np.int16
            ),
            input_b_zero_points=np.asarray(
                [node.input_b.zero_point for node in nodes], dtype=np.int16
            ),
            input_a_scales=np.asarray(
                [float(node.input_a.scale[0]) for node in nodes], dtype=np.float32
            ),
            input_b_scale_min=np.asarray(
                [float(node.input_b.scale.min()) for node in nodes], dtype=np.float32
            ),
            input_b_scale_max=np.asarray(
                [float(node.input_b.scale.max()) for node in nodes], dtype=np.float32
            ),
            input_b_is_static=np.asarray(
                [node.input_b.is_static for node in nodes], dtype=np.bool_
            ),
            pair_count=pair_count,
            qdq_scale_weight=scale_mass,
            qdq_scale_square_weight=scale_square_mass,
        )
    os.replace(temporary, output)
    node_order_sha256 = _ordered_strings_sha256(node_ids)
    group_order_sha256 = _ordered_strings_sha256(group_names)
    result: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "compressed": not bool(args.uncompressed),
        "histogram_shape": list(HISTOGRAM_SHAPE),
        "histogram_dtypes": {
            "pair_count": "uint64",
            "qdq_scale_weight": "float64",
            "qdq_scale_square_weight": "float64",
        },
        "model": str(model_path),
        "model_sha1": _sha1_file(model_path),
        "instrumented_model": str(instrumented),
        "instrumentation_contract": instrumentation,
        "provider": provider,
        "batch_size": int(args.batch_size),
        "row_block_size": int(args.row_block_size),
        "image_count": image_count,
        "node_count": len(nodes),
        "dynamic_qcode_output_count": len(dynamic_names),
        "dynamic_qcode_observed_shapes": observed_shapes,
        "profile_target": "top1_clean_margin_loss_bins_plus_exact_wrong_recovery",
        "group_names": group_names,
        "group_order_sha256": group_order_sha256,
        "node_order_sha256": node_order_sha256,
        "group_image_counts": image_counts.tolist(),
        "group_assignment_sha256": sha256_array(group_assignment),
        "group_metadata": group_metadata,
        "semantic_zero_mask": "exclude qA == zpA or qB == zpB before histogramming",
        "semantic_zero_mask_applied_before_fullbyte_joint": True,
        "exact_fullbyte_joint": True,
        "exact_static_per_output_channel_scale_moments": True,
        "dynamic_broadcast_prefix_preserved_through_joint_contraction": True,
        "all_group_derived_as_exact_partition_sum": True,
        "lut_domain": "signed_int8_raw_code_plus_128",
        "candidate_multiplier_inference_count": 0,
        "candidate_multiplier_luts_loaded": False,
        "per_design_runtime_observations_used": False,
        "test_targets_or_runtime_loaded": False,
        "deployment_reuse": "frozen application profile shared by every new multiplier",
        "node_registry": str(registry_path),
        "node_registry_sha256": sha256_file(registry_path),
        "inputs": str(inputs_path),
        "inputs_sha256": sha256_file(inputs_path),
        "exact_baseline": str(exact_baseline_path),
        "exact_baseline_sha256": sha256_file(exact_baseline_path),
        "true1000_positions_sha256": sha256_array(positions),
        "exact_meta": str(exact_meta_path),
        "exact_meta_sha256": sha256_file(exact_meta_path),
        "resource_estimate": _profile_resource_estimate(int(args.row_block_size)),
        "elapsed_seconds": time.time() - started,
    }
    write_json(metadata_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--node-registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    result.add_argument("--exact-baseline", type=Path, default=DEFAULT_EXACT_BASELINE)
    result.add_argument("--exact-meta", type=Path, default=DEFAULT_EXACT_META)
    result.add_argument(
        "--instrumented-model", type=Path, default=DEFAULT_INSTRUMENTED_MODEL
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--row-block-size", type=int, default=8192)
    result.add_argument("--image-limit", type=int, default=0)
    result.add_argument("--uncompressed", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--overwrite-instrumented", action="store_true")
    result.add_argument("--self-test-only", action="store_true")
    result.add_argument("--estimate-only", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.row_block_size < 1:
        raise SystemExit("--row-block-size must be positive")
    if args.self_test_only:
        print(json.dumps(run_synthetic_checks(), indent=2, sort_keys=True))
        return 0
    if args.estimate_only:
        print(
            json.dumps(
                _profile_resource_estimate(int(args.row_block_size)),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2 for the dynamic-B16 profile")
    if args.image_limit < 0 or args.image_limit > 1000:
        raise SystemExit("--image-limit must be zero or in [1, 1000]")
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
