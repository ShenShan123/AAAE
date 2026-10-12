#!/usr/bin/env python
"""Build target-free causal-prefix and second-order Transformer features.

For every multiplier, the frozen operand/LUT contraction is combined with the
frozen margin response at the injected MatMul nodes. Features summarize block
and architecture-specific group trajectories, upstream-by-current second-order
interactions, cancellation, and soft decision-boundary crossings. No candidate
design target or runtime observation is loaded.
"""

from __future__ import annotations

import argparse
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
from aaae.accuracy_predictor.features.transformer.physics_core import (
    CAUSAL_PREFIX_FEATURE_SCHEMA_VERSION,
    DEFAULT_COEFFICIENTS,
    DEFAULT_PROPAGATION,
    NODE_COUNT,
    PROFILE_IMAGE_COUNT,
    parse_node_structure,
    profile_group_weights,
)

DEFAULT_OUTPUT = (
    FORMAL_CACHE_ROOT / "dynamic_b16_causal_prefix_second_order.float32.npy"
)
SCHEMA_VERSION = CAUSAL_PREFIX_FEATURE_SCHEMA_VERSION
BLOCK_COUNT = 12
OPS_PER_BLOCK = 6
STAGE_BLOCK_COUNTS = (2, 2, 6, 2)
SOFT_TEMPERATURES = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60)
QUANTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


def compact_profile_groups(
    profile_groups: np.ndarray,
    profile_group_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Merge eight correct-margin bins into low/mid/high populations."""

    weights = np.asarray(profile_groups, dtype=np.float32)
    expected_names = [
        "all",
        *[f"clean_correct_margin_bin{index}" for index in range(8)],
        "clean_wrong",
    ]
    if (
        weights.shape != (10, PROFILE_IMAGE_COUNT)
        or profile_group_names != expected_names
    ):
        raise ValueError("profile groups do not match the registered ten-group schema")
    compact = np.vstack(
        (
            weights[0],
            weights[1:3].mean(axis=0),
            weights[3:5].mean(axis=0),
            weights[5:9].mean(axis=0),
            weights[9],
        )
    ).astype(np.float32)
    if not np.allclose(compact.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6):
        raise RuntimeError("compact profile groups are not normalized")
    return compact, ["all", "correct_low", "correct_mid", "correct_high", "wrong"]


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _append_group_moments(
    parts: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    *,
    prefix: str,
    channel_names: list[str],
    group_weights: np.ndarray,
    group_names: list[str],
    moments: tuple[str, ...],
) -> None:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[1] != PROFILE_IMAGE_COUNT:
        raise ValueError(f"{prefix} must have shape [row, 256, channel]")
    if array.shape[2] != len(channel_names):
        raise ValueError(f"{prefix} channel names are not aligned")
    for moment in moments:
        if moment == "signed_mean":
            summary = _signed_log1p(np.einsum("bik,gi->bkg", array, group_weights))
        elif moment == "abs_mean":
            summary = np.log1p(np.einsum("bik,gi->bkg", np.abs(array), group_weights))
        elif moment == "rms":
            summary = np.log1p(
                np.sqrt(
                    np.maximum(
                        np.einsum("bik,gi->bkg", np.square(array), group_weights),
                        0.0,
                    )
                )
            )
        elif moment == "mean":
            summary = np.einsum("bik,gi->bkg", array, group_weights)
        else:
            raise ValueError(f"unsupported feature moment: {moment}")
        parts.append(summary.reshape(len(array), -1).astype(np.float32))
        names.extend(
            f"{prefix}_{channel}_{group}_{moment}"
            for channel in channel_names
            for group in group_names
        )


def build_causal_prefix_features(
    coefficients: np.ndarray,
    *,
    fixed_margin_response: np.ndarray,
    profile_margin: np.ndarray,
    group_weights: np.ndarray,
    group_names: list[str],
    operation_names: tuple[str, ...] | None = None,
    block_group_counts: tuple[int, ...] = STAGE_BLOCK_COUNTS,
    block_group_label: str = "stage",
) -> tuple[np.ndarray, list[str]]:
    """Return compact physics features for an in-memory coefficient batch."""

    operation_labels = (
        operation_names
        if operation_names is not None
        else ("qkv", "qk", "av", "proj", "fc1", "fc2")
    )
    operations_per_block = len(operation_labels)
    node_count = BLOCK_COUNT * operations_per_block
    if operations_per_block < 1 or len(set(operation_labels)) != operations_per_block:
        raise ValueError("operation names must be non-empty and unique")
    group_counts = tuple(int(value) for value in block_group_counts)
    if (
        not group_counts
        or any(value < 1 for value in group_counts)
        or sum(group_counts) != BLOCK_COUNT
    ):
        raise ValueError(
            f"block group counts must be positive and sum to {BLOCK_COUNT}"
        )
    group_label = str(block_group_label).strip()
    if not group_label or not group_label.replace("_", "").isalnum():
        raise ValueError("block group label must be a non-empty identifier")
    coefficient = np.asarray(coefficients, dtype=np.float32)
    response = np.asarray(fixed_margin_response, dtype=np.float32)
    margin = np.asarray(profile_margin, dtype=np.float32).reshape(-1)
    weights = np.asarray(group_weights, dtype=np.float32)
    if coefficient.ndim != 3 or coefficient.shape[1:] != (
        PROFILE_IMAGE_COUNT,
        node_count,
    ):
        raise ValueError("coefficient batch has an unexpected shape")
    if response.shape != (node_count, PROFILE_IMAGE_COUNT):
        raise ValueError("fixed margin response has an unexpected shape")
    if (
        margin.shape != (PROFILE_IMAGE_COUNT,)
        or weights.shape[1] != PROFILE_IMAGE_COUNT
    ):
        raise ValueError("profile margin/groups are not aligned")
    if len(group_names) != len(weights):
        raise ValueError("profile group names are not aligned")

    batch = len(coefficient)
    amplitude = coefficient.reshape(
        batch, PROFILE_IMAGE_COUNT, BLOCK_COUNT, operations_per_block
    )
    contribution = amplitude * response.T.reshape(
        1, PROFILE_IMAGE_COUNT, BLOCK_COUNT, operations_per_block
    )
    block_amplitude = amplitude.sum(axis=3)
    block_contribution = contribution.sum(axis=3)
    block_prefix = block_contribution.cumsum(axis=2)
    prefix_before = block_prefix - block_contribution
    second_order = block_contribution * prefix_before

    stage_starts = np.cumsum((0, *group_counts[:-1]))
    stage_ends = np.cumsum(group_counts)
    stage_amplitude = np.stack(
        [
            block_amplitude[:, :, start:end].sum(axis=2)
            for start, end in zip(stage_starts, stage_ends, strict=True)
        ],
        axis=2,
    )
    stage_contribution = np.stack(
        [
            block_contribution[:, :, start:end].sum(axis=2)
            for start, end in zip(stage_starts, stage_ends, strict=True)
        ],
        axis=2,
    )
    operation_amplitude = amplitude.sum(axis=2)
    operation_contribution = contribution.sum(axis=2)

    parts: list[np.ndarray] = []
    names: list[str] = []
    block_names = [f"block{index:02d}" for index in range(BLOCK_COUNT)]
    stage_names = [f"{group_label}{index}" for index in range(len(group_counts))]
    operation_name_list = list(operation_labels)
    _append_group_moments(
        parts,
        names,
        block_amplitude,
        prefix="amplitude_block",
        channel_names=block_names,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean"),
    )
    _append_group_moments(
        parts,
        names,
        stage_amplitude,
        prefix="amplitude_stage",
        channel_names=stage_names,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean"),
    )
    _append_group_moments(
        parts,
        names,
        operation_amplitude,
        prefix="amplitude_operation",
        channel_names=operation_name_list,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean"),
    )
    _append_group_moments(
        parts,
        names,
        block_contribution,
        prefix="propagation_block",
        channel_names=block_names,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean", "rms"),
    )
    _append_group_moments(
        parts,
        names,
        block_prefix,
        prefix="causal_prefix",
        channel_names=block_names,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean", "rms"),
    )
    _append_group_moments(
        parts,
        names,
        second_order,
        prefix="upstream_x_current",
        channel_names=block_names,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean"),
    )
    _append_group_moments(
        parts,
        names,
        stage_contribution,
        prefix="propagation_stage",
        channel_names=stage_names,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean", "rms"),
    )
    _append_group_moments(
        parts,
        names,
        operation_contribution,
        prefix="propagation_operation",
        channel_names=operation_name_list,
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean", "rms"),
    )

    node_contribution = contribution.reshape(batch, PROFILE_IMAGE_COUNT, node_count)
    final_shift = block_prefix[:, :, -1]
    absolute_path = np.abs(node_contribution).sum(axis=2)
    positive_path = np.maximum(node_contribution, 0.0).sum(axis=2)
    negative_path = np.minimum(node_contribution, 0.0).sum(axis=2)
    node_energy = np.square(node_contribution).sum(axis=2)
    cancellation = 1.0 - np.abs(final_shift) / (absolute_path + 1.0e-5)
    path_values = np.stack(
        (absolute_path, positive_path, negative_path, node_energy, cancellation),
        axis=2,
    )
    _append_group_moments(
        parts,
        names,
        path_values,
        prefix="path",
        channel_names=["absolute", "positive", "negative", "energy", "cancellation"],
        group_weights=weights,
        group_names=group_names,
        moments=("signed_mean", "abs_mean"),
    )

    stage_prefix = block_prefix[:, :, np.asarray(stage_ends) - 1]
    baseline_margin = margin[None, :, None]
    for temperature in SOFT_TEMPERATURES:
        before = 1.0 / (
            1.0 + np.exp(np.clip(baseline_margin / temperature, -40.0, 40.0))
        )
        after = 1.0 / (
            1.0
            + np.exp(
                np.clip((baseline_margin + stage_prefix) / temperature, -40.0, 40.0)
            )
        )
        _append_group_moments(
            parts,
            names,
            after - before,
            prefix=f"soft_cross_t{temperature:.2f}".replace(".", "p"),
            channel_names=stage_names,
            group_weights=weights,
            group_names=group_names,
            moments=("mean",),
        )
    hard_cross = (baseline_margin + stage_prefix < 0.0).astype(np.float32)
    _append_group_moments(
        parts,
        names,
        hard_cross,
        prefix="hard_cross",
        channel_names=stage_names,
        group_weights=weights,
        group_names=group_names,
        moments=("mean",),
    )

    normalized_stage_shift = stage_prefix / (baseline_margin + 0.05)
    quantile_values = np.quantile(
        normalized_stage_shift,
        QUANTILES,
        axis=1,
    ).transpose(1, 2, 0)
    parts.append(quantile_values.reshape(batch, -1).astype(np.float32))
    names.extend(
        f"normalized_stage_shift_{stage}_q{quantile:.2f}".replace(".", "p")
        for stage in stage_names
        for quantile in QUANTILES
    )

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if features.shape[1] != len(names) or len(set(names)) != len(names):
        raise RuntimeError("causal-prefix feature names are not unique and aligned")
    if not np.isfinite(features).all():
        raise RuntimeError("causal-prefix features contain non-finite values")
    return features, names


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = args.output.expanduser().resolve()
    names_path = output.with_suffix(output.suffix + ".features.json")
    metadata_path = output.with_suffix(output.suffix + ".json")
    paths = (output, names_path, metadata_path)
    if any(path.exists() for path in paths) and not args.overwrite:
        raise FileExistsError("causal-prefix cache exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    coefficients = np.load(args.coefficients, mmap_mode="r", allow_pickle=False)
    if coefficients.ndim != 3 or coefficients.shape[1:] != (
        PROFILE_IMAGE_COUNT,
        NODE_COUNT,
    ):
        raise ValueError("coefficient cache has an unexpected shape")
    with np.load(args.propagation, allow_pickle=False) as propagation:
        node_ids = np.asarray(propagation["node_ids"]).astype(str)
        fixed_response = np.asarray(
            propagation["fixed_margin_response"], dtype=np.float32
        )
        profile_margin = np.asarray(
            propagation["clean_pred_vs_runner_up_margin"], dtype=np.float32
        )
        profile_exact = np.asarray(propagation["exact_top1_correct"], dtype=np.uint8)
    node_stage, node_operation = parse_node_structure(node_ids)
    if not (
        np.array_equal(node_operation, np.tile(np.arange(OPS_PER_BLOCK), BLOCK_COUNT))
        and [int(np.sum(node_stage == stage)) // OPS_PER_BLOCK for stage in range(4)]
        == list(STAGE_BLOCK_COUNTS)
    ):
        raise ValueError("node registry is not the expected 12-block execution order")
    full_groups, full_names = profile_group_weights(profile_margin, profile_exact)
    groups, group_names = compact_profile_groups(full_groups, full_names)

    probe, feature_names = build_causal_prefix_features(
        np.asarray(coefficients[:1], dtype=np.float32),
        fixed_margin_response=fixed_response,
        profile_margin=profile_margin,
        group_weights=groups,
        group_names=group_names,
    )
    temporary = output.with_name(output.name + ".partial")
    cache = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(coefficients), probe.shape[1]),
    )
    for start in range(0, len(coefficients), args.batch_size):
        end = min(start + args.batch_size, len(coefficients))
        batch_features, names = build_causal_prefix_features(
            np.asarray(coefficients[start:end], dtype=np.float32),
            fixed_margin_response=fixed_response,
            profile_margin=profile_margin,
            group_weights=groups,
            group_names=group_names,
        )
        if names != feature_names:
            raise RuntimeError("causal-prefix schema changed between batches")
        cache[start:end] = batch_features
    cache.flush()
    del cache
    os.replace(temporary, output)
    names_path.write_text(
        json.dumps(feature_names, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": "target_free_all_multiplier_static_physics_features",
        "rows": len(coefficients),
        "feature_count": len(feature_names),
        "dtype": "float32",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "feature_names": str(names_path),
        "feature_names_sha256": sha256_file(names_path),
        "coefficients": str(args.coefficients.resolve()),
        "coefficients_sha256": sha256_file(args.coefficients),
        "propagation": str(args.propagation.resolve()),
        "propagation_sha256": sha256_file(args.propagation),
        "node_ids_sha256": sha256_array(node_ids),
        "node_count": NODE_COUNT,
        "block_count": BLOCK_COUNT,
        "stage_block_counts": list(STAGE_BLOCK_COUNTS),
        "profile_image_count": PROFILE_IMAGE_COUNT,
        "profile_groups": group_names,
        "soft_temperatures": list(SOFT_TEMPERATURES),
        "quantiles": list(QUANTILES),
        "targets_loaded": False,
        "runtime_bank_loaded": False,
        "candidate_swin_inference_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    write_json(metadata_path, metadata)
    return metadata


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    result.add_argument("--propagation", type=Path, default=DEFAULT_PROPAGATION)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    print(json.dumps(prepare(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
