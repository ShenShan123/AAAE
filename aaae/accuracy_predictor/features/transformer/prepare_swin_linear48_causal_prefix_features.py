#!/usr/bin/env python
"""Build target-free 48-node causal-prefix features from the A100 profile.

The full-byte operand profile is contracted directly with every candidate LUT.
Only the resulting ``[design, image, node]`` amplitudes are materialized per
batch, avoiding a multi-gigabyte intermediate coefficient cache.
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
import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer.build_swin_per_image_operand_profile import (
    LUT_CELLS,
)
from aaae.accuracy_predictor.features.transformer.build_swin_per_image_operand_profile import (
    SCHEMA_VERSION as OPERAND_PROFILE_SCHEMA,
)
from aaae.accuracy_predictor.features.transformer.common import (
    FORMAL_CACHE_ROOT,
    sha256_array,
    sha256_file,
    write_json,
)
from aaae.accuracy_predictor.features.transformer.physics_core import (
    profile_group_weights,
)
from aaae.accuracy_predictor.features.transformer.prepare_causal_prefix_features import (
    build_causal_prefix_features,
    compact_profile_groups,
)
from aaae.accuracy_predictor.features.static.signed_features import (
    load_s8s8_sign_magnitude_truth_lut_asset,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
DEFAULT_OPERAND_PROFILE = (
    FORMAL_CACHE_ROOT / "linear48_a100_operand_profile256.float32.npy"
)
DEFAULT_PROPAGATION_PROFILE = (
    FORMAL_CACHE_ROOT / "linear48_a100_propagation_profile256.npz"
)
DEFAULT_OUTPUT = (
    FORMAL_CACHE_ROOT / "linear48_a100_causal_prefix_second_order.float32.npy"
)
DEFAULT_ASSET_ROOT = (
    PROJECT_ROOT / "datasets" / "multiplier" / "lowonly20k_s8s8_sign_magnitude_v1"
)
SCHEMA_VERSION = "swin_linear48_causal_prefix_second_order_features_v1"
DEFAULT_DATASET_ID = "swin_tiny_s8s8_linear48_strict50_true1000_raw500_v1"
PROPAGATION_SCHEMA = "swin_dynamic_b16_propagation_profile_v1"
PROBE_BUDGET_PROPAGATION_SCHEMA = "probe_budget_application_propagation_profile_v1"
IMAGE_COUNT = 256
NODE_COUNT = 48
BLOCK_COUNT = 12
OPERATION_NAMES = ("qkv", "proj", "fc1", "fc2")
EXPECTED_ROWS = 20000


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return value


def _load_profile_contracts(
    operand_path: Path,
    propagation_path: Path,
    exact_meta_path: Path,
) -> tuple[
    np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]
]:
    operand_metadata_path = operand_path.with_suffix(operand_path.suffix + ".json")
    operand_index_path = operand_path.with_suffix(operand_path.suffix + ".index.npz")
    propagation_metadata_path = propagation_path.with_suffix(
        propagation_path.suffix + ".json"
    )
    required_paths = [
        operand_path,
        operand_metadata_path,
        operand_index_path,
        propagation_path,
        propagation_metadata_path,
    ]
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"required linear48 profile asset is missing: {path}"
            )

    operand_metadata = _read_object(operand_metadata_path)
    propagation_metadata = _read_object(propagation_metadata_path)
    expected_shape = (IMAGE_COUNT, NODE_COUNT, LUT_CELLS)
    if (
        operand_metadata.get("schema_version") != OPERAND_PROFILE_SCHEMA
        or operand_metadata.get("artifact_complete") is not True
        or operand_metadata.get("shape") != list(expected_shape)
        or operand_metadata.get("node_count") != NODE_COUNT
        or operand_metadata.get("output_sha256") != sha256_file(operand_path)
        or operand_metadata.get("index_sha256") != sha256_file(operand_index_path)
    ):
        raise ValueError("linear48 operand profile contract is invalid")
    propagation_schema = str(propagation_metadata.get("schema_version", ""))
    if (
        propagation_schema
        not in {PROPAGATION_SCHEMA, PROBE_BUDGET_PROPAGATION_SCHEMA}
        or propagation_metadata.get("node_count") != NODE_COUNT
        or propagation_metadata.get("image_count") != IMAGE_COUNT
        or propagation_metadata.get("output_hashes", {}).get("sha256")
        != sha256_file(propagation_path)
    ):
        raise ValueError("linear48 propagation profile contract is invalid")
    if (
        operand_metadata.get("propagation_profile_sha256")
        != propagation_metadata["output_hashes"]["sha256"]
    ):
        raise ValueError("operand and propagation profiles are not bound together")

    image_index_key = str(
        propagation_metadata.get("image_index_key", "selected_true1000_indices")
    )
    image_index_space = str(
        propagation_metadata.get("image_index_space", "true1000")
    )
    if image_index_key not in {
        "selected_true1000_indices",
        "selected_raw500_indices",
    }:
        raise ValueError("linear48 profile has an unsupported image index key")
    if operand_metadata.get("image_index_key", image_index_key) != image_index_key:
        raise ValueError("operand and propagation image index keys differ")
    if operand_metadata.get("image_index_space", image_index_space) != image_index_space:
        raise ValueError("operand and propagation image index spaces differ")

    operand = np.load(operand_path, mmap_mode="c", allow_pickle=False)
    if operand.shape != expected_shape or operand.dtype != np.float32:
        raise ValueError("linear48 operand profile array has an invalid shape or dtype")
    with np.load(operand_index_path, allow_pickle=False) as index:
        node_ids = np.asarray(index["node_ids"]).astype(str).tolist()
        selected_local = np.asarray(index[image_index_key], dtype=np.int64)
        selected_union = np.asarray(index["selected_union_positions"], dtype=np.int64)
    with np.load(propagation_path, allow_pickle=False) as propagation:
        propagation_nodes = np.asarray(propagation["node_ids"]).astype(str).tolist()
        propagation_local = np.asarray(propagation[image_index_key], dtype=np.int64)
        propagation_union = np.asarray(
            propagation["selected_union_positions"], dtype=np.int64
        )
        response = np.asarray(propagation["fixed_margin_response"], dtype=np.float32)
        clean_margin = np.asarray(
            propagation["clean_pred_vs_runner_up_margin"], dtype=np.float32
        )
        embedded_correct = (
            np.asarray(propagation["exact_top1_correct"], dtype=np.uint8)
            if "exact_top1_correct" in propagation.files
            else None
        )
    if (
        len(node_ids) != NODE_COUNT
        or len(set(node_ids)) != NODE_COUNT
        or propagation_nodes != node_ids
        or not np.array_equal(selected_local, propagation_local)
        or not np.array_equal(selected_union, propagation_union)
        or selected_local.shape != (IMAGE_COUNT,)
        or response.shape != (NODE_COUNT, IMAGE_COUNT)
        or clean_margin.shape != (IMAGE_COUNT,)
    ):
        raise ValueError("linear48 operand and propagation arrays are not aligned")

    expected_suffixes = tuple(
        suffix
        for _block in range(BLOCK_COUNT)
        for suffix in (
            "/attn/qkv/MatMul",
            "/attn/proj/MatMul",
            "/mlp/fc1/MatMul",
            "/mlp/fc2/MatMul",
        )
    )
    if not all(
        node.endswith(suffix)
        for node, suffix in zip(node_ids, expected_suffixes, strict=True)
    ):
        raise ValueError("linear48 nodes are not in 12-block x 4-operation order")

    if image_index_key == "selected_true1000_indices":
        if not exact_meta_path.is_file():
            raise FileNotFoundError(
                f"required true1000 exact metadata is missing: {exact_meta_path}"
            )
        with np.load(exact_meta_path, allow_pickle=False) as exact_meta:
            correct_bank = np.asarray(
                exact_meta["exact_top1_correct_bits"], dtype=np.uint8
            )
        if correct_bank.shape != (1000,) or not np.all(
            (correct_bank == 0) | (correct_bank == 1)
        ):
            raise ValueError("canonical exact correctness bank is invalid")
        exact_correct = np.asarray(correct_bank[selected_local], dtype=np.uint8)
        correctness_source = str(exact_meta_path)
    else:
        if embedded_correct is None or embedded_correct.shape != (IMAGE_COUNT,):
            raise ValueError(
                "raw500 propagation profile must embed exact_top1_correct[256]"
            )
        if not np.all((embedded_correct == 0) | (embedded_correct == 1)):
            raise ValueError("embedded profile correctness must be binary")
        exact_correct = embedded_correct
        correctness_source = "propagation_profile:exact_top1_correct"
    return (
        operand,
        selected_local,
        selected_union,
        response,
        clean_margin,
        node_ids,
        {
            "operand_metadata": operand_metadata,
            "propagation_metadata": propagation_metadata,
            "exact_correct": exact_correct,
            "image_index_key": image_index_key,
            "image_index_space": image_index_space,
            "correctness_source": correctness_source,
        },
    )


def _torch_device(specification: str) -> Any:
    import torch

    if specification == "cpu":
        return torch.device("cpu")
    if not specification.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("--device must identify an available CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch.device(specification)


def build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_path = args.output.expanduser().resolve()
    names_path = output_path.with_suffix(output_path.suffix + ".features.json")
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if (
        any(path.exists() for path in (output_path, names_path, metadata_path))
        and not args.overwrite
    ):
        raise FileExistsError("linear48 causal-prefix cache exists; pass --overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    block_group_counts = tuple(
        int(value.strip())
        for value in str(args.block_group_counts).split(",")
        if value.strip()
    )
    if (
        not block_group_counts
        or any(value < 1 for value in block_group_counts)
        or sum(block_group_counts) != BLOCK_COUNT
    ):
        raise ValueError(
            f"--block-group-counts must be positive and sum to {BLOCK_COUNT}"
        )
    block_group_label = str(args.block_group_label).strip()

    operand_path = args.operand_profile.expanduser().resolve()
    propagation_path = args.propagation_profile.expanduser().resolve()
    exact_meta_path = args.exact_meta.expanduser().resolve()
    (
        operand,
        selected_local,
        selected_union,
        response,
        clean_margin,
        node_ids,
        contracts,
    ) = _load_profile_contracts(operand_path, propagation_path, exact_meta_path)
    full_groups, full_group_names = profile_group_weights(
        clean_margin, contracts["exact_correct"]
    )
    groups, group_names = compact_profile_groups(full_groups, full_group_names)

    probe, feature_names = build_causal_prefix_features(
        np.zeros((1, IMAGE_COUNT, NODE_COUNT), dtype=np.float32),
        fixed_margin_response=response,
        profile_margin=clean_margin,
        group_weights=groups,
        group_names=group_names,
        operation_names=OPERATION_NAMES,
        block_group_counts=block_group_counts,
        block_group_label=block_group_label,
    )
    configuration = pd.read_csv(
        args.configuration_index.expanduser().resolve(),
        usecols=["row_index", "runtime_design_id"],
        dtype={"runtime_design_id": str},
        low_memory=False,
    )
    if not np.array_equal(
        configuration["row_index"].to_numpy(np.int64),
        np.arange(EXPECTED_ROWS, dtype=np.int64),
    ):
        raise ValueError("linear48 configuration index is not in canonical row order")
    asset = load_s8s8_sign_magnitude_truth_lut_asset(
        args.asset_root.expanduser().resolve()
    )
    if not np.array_equal(
        configuration["runtime_design_id"].astype(str).to_numpy(),
        asset.runtime_ids.astype(str),
    ):
        raise ValueError("linear48 rows do not align with the multiplier LUT asset")

    import torch

    device = _torch_device(str(args.device))
    kernel = torch.as_tensor(np.asarray(operand), dtype=torch.float32).to(device=device)
    kernel = kernel.reshape(IMAGE_COUNT * NODE_COUNT, LUT_CELLS)
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    destination = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(EXPECTED_ROWS, probe.shape[1]),
    )
    try:
        for start in range(0, EXPECTED_ROWS, int(args.chunk_size)):
            stop = min(start + int(args.chunk_size), EXPECTED_ROWS)
            truth = np.asarray(asset.tables[start:stop], dtype=np.int32)
            delta = np.asarray(truth - exact[None, :, :], dtype=np.float32).reshape(
                -1, LUT_CELLS
            )
            with torch.inference_mode():
                amplitude = torch.as_tensor(delta, device=device) @ kernel.T
                amplitude_host = np.asarray(
                    amplitude.reshape(-1, IMAGE_COUNT, NODE_COUNT).cpu().numpy(),
                    dtype=np.float32,
                )
            features, names = build_causal_prefix_features(
                amplitude_host,
                fixed_margin_response=response,
                profile_margin=clean_margin,
                group_weights=groups,
                group_names=group_names,
                operation_names=OPERATION_NAMES,
                block_group_counts=block_group_counts,
                block_group_label=block_group_label,
            )
            if names != feature_names:
                raise RuntimeError(
                    "linear48 causal-prefix schema changed between batches"
                )
            destination[start:stop] = features
            destination.flush()
            print(
                json.dumps(
                    {
                        "completed_rows": stop,
                        "total_rows": EXPECTED_ROWS,
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        del destination
        del kernel
    os.replace(temporary, output_path)
    names_path.write_text(
        json.dumps(feature_names, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    metadata: dict[str, Any] = {
        "schema_version": str(args.schema_version),
        "claim_scope": "target_free_all_multiplier_static_physics_features",
        "dataset_id": str(args.dataset_id),
        "injection_scope": "sram_cim_weight_stationary_linear_only_v1",
        "rows": EXPECTED_ROWS,
        "feature_count": len(feature_names),
        "dtype": "float32",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "feature_names": str(names_path),
        "feature_names_sha256": sha256_file(names_path),
        "operand_profile": str(operand_path),
        "operand_profile_sha256": sha256_file(operand_path),
        "propagation_profile": str(propagation_path),
        "propagation_profile_sha256": sha256_file(propagation_path),
        "exact_meta": (
            str(exact_meta_path)
            if contracts["image_index_key"] == "selected_true1000_indices"
            else None
        ),
        "exact_meta_sha256": (
            sha256_file(exact_meta_path)
            if contracts["image_index_key"] == "selected_true1000_indices"
            else None
        ),
        "profile_correctness_source": contracts["correctness_source"],
        "node_ids_sha256": sha256_array(np.asarray(node_ids)),
        "image_index_key": contracts["image_index_key"],
        "image_index_space": contracts["image_index_space"],
        contracts["image_index_key"]: selected_local.tolist(),
        f"{contracts['image_index_key']}_sha256": sha256_array(selected_local),
        "selected_union_positions": selected_union.tolist(),
        "selected_union_positions_sha256": sha256_array(selected_union),
        "node_count": NODE_COUNT,
        "block_count": BLOCK_COUNT,
        "operations_per_block": len(OPERATION_NAMES),
        "operation_names": list(OPERATION_NAMES),
        "block_group_counts": list(block_group_counts),
        "block_group_label": block_group_label,
        "profile_image_count": IMAGE_COUNT,
        "profile_groups": group_names,
        "contraction_device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32_enabled": False,
        "targets_loaded": False,
        "runtime_bank_loaded": False,
        "candidate_model_inference_count": 0,
        "candidate_swin_inference_count": 0,
        "elapsed_seconds": time.time() - started,
    }
    write_json(metadata_path, metadata)
    return metadata


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--operand-profile", type=Path, default=DEFAULT_OPERAND_PROFILE)
    result.add_argument(
        "--propagation-profile", type=Path, default=DEFAULT_PROPAGATION_PROFILE
    )
    result.add_argument(
        "--exact-meta",
        type=Path,
        default=DATASET_ROOT / "shared" / "true1000_exact_meta.npz",
    )
    result.add_argument(
        "--configuration-index",
        type=Path,
        default=DATASET_ROOT / "data" / "configuration_index.csv",
    )
    result.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    result.add_argument("--schema-version", default=SCHEMA_VERSION)
    result.add_argument("--block-group-counts", default="2,2,6,2")
    result.add_argument("--block-group-label", default="stage")
    result.add_argument("--device", default="cuda:2")
    result.add_argument("--chunk-size", type=int, default=64)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
