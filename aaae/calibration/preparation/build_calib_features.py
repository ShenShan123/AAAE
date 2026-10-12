"""E7: build s (common3103) and Transformer c features for the calibration pool.

``s`` reuses ``build_s8s8_signed_common3103`` directly.  ``c`` replicates the
frozen causal-prefix builder (same operand/propagation profiles, same block
groups, same schema) for an arbitrary row count instead of the hardcoded 20000.

Usage:
    python \
        -m aaae.calibration.preparation.build_calib_features \
        --asset-root end_to_end_prediction/datasets/multiplier/lowonly_calibpool4500_s8s8_sign_magnitude_v1 \
        --output-root experiment/e07_conformal_screening/artifacts/features_pool \
        --models swin_tiny --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

BLOCK_GROUP_PARAMS = {
    "swin_tiny": ((2, 2, 6, 2), "stage"),
    "vit_base": ((3, 3, 3, 3), "depth_segment"),
    "deit_base": ((3, 3, 3, 3), "depth_segment"),
}
TRANSIENT_DESCRIPTOR_ZERO_COLUMNS = (3089, 3090)


def build_s_features(population_path: Path, asset_root: Path) -> tuple[np.ndarray, list[str]]:
    from aaae.accuracy_predictor.features.static.signed_features import build_s8s8_signed_common3103

    built = build_s8s8_signed_common3103(
        target_population=population_path, asset_root=asset_root, chunk_size=64
    )
    values = np.asarray(built.common3103, dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("s features contain non-finite values")
    for column in TRANSIENT_DESCRIPTOR_ZERO_COLUMNS:
        if not np.array_equal(values[:, column], np.zeros(len(values), dtype=np.float32)):
            raise RuntimeError(f"s descriptor column {column} is not zero")
    columns = [f"common3103_col{index:04d}" for index in range(values.shape[1])]
    return values, columns


CACHE_PROFILE_PREFIX = {"swin_tiny": "", "vit_base": "vit_", "deit_base": "deit_"}
REFERENCE_DATASETS = {
    "swin_tiny": E2E_ROOT / "datasets/swin_tiny/s8s8_linear48_strict50_true1000_raw500_v1",
    "vit_base": E2E_ROOT / "datasets/vit/s8s8_linear48_strict50_true1000_raw500_v1",
    "deit_base": E2E_ROOT / "datasets/deit/s8s8_linear48_strict50_true1000_raw500_v1",
}


def build_c_features(model: str, population_path: Path, asset_root: Path, device: str, chunk_size: int, profile_mode: str = "raw500"):
    from aaae.accuracy_predictor.features.transformer.physics_core import profile_group_weights
    from aaae.accuracy_predictor.features.transformer.prepare_causal_prefix_features import (
        build_causal_prefix_features,
        compact_profile_groups,
    )
    from aaae.accuracy_predictor.features.transformer.prepare_swin_linear48_causal_prefix_features import (
        _load_profile_contracts,
    )
    from aaae.inference_runtime.transformer.s8s8_truth_lut_asset import load_s8s8_sign_magnitude_truth_lut_asset

    if profile_mode == "cache":
        prefix = CACHE_PROFILE_PREFIX[model]
        profile_root = E2E_ROOT / "experiment/cache"
        operand_path = profile_root / f"{prefix}linear48_a100_operand_profile256.float32.npy"
        propagation_path = profile_root / f"{prefix}linear48_a100_propagation_profile256.npz"
        exact_meta_path = REFERENCE_DATASETS[model] / "shared" / "true1000_exact_meta.npz"
    else:
        profile_root = EA2_ROOT / "experiment/probe_budget/artifacts/profile256_v1" / model
        operand_path = profile_root / "operand_profile256.float32.npy"
        propagation_path = profile_root / "propagation_profile256.npz"
        exact_meta_path = Path("/nonexistent/not_used_for_raw500")
    operand, _selected_local, _selected_union, response, clean_margin, _node_ids, contracts = (
        _load_profile_contracts(operand_path, propagation_path, exact_meta_path)
    )
    full_groups, full_names = profile_group_weights(clean_margin, contracts["exact_correct"])
    groups, group_names = compact_profile_groups(full_groups, full_names)
    group_counts, group_label = BLOCK_GROUP_PARAMS[model]

    population = pd.read_csv(population_path, low_memory=False)
    population["row_index"] = population["row_index"].astype(np.int64)
    asset = load_s8s8_sign_magnitude_truth_lut_asset(asset_root)
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]

    torch_device = torch.device(device)
    kernel = torch.as_tensor(np.asarray(operand), dtype=torch.float32)
    kernel = kernel.reshape(256 * 48, 65536).to(torch_device)

    outputs: list[np.ndarray] = []
    names_reference: list[str] | None = None
    row_count = len(population)
    started = time.time()
    try:
        for start in range(0, row_count, chunk_size):
            stop = min(start + chunk_size, row_count)
            truth = np.asarray(asset.tables[start:stop], dtype=np.int32)
            delta = (truth - exact[None, :, :]).reshape(-1, 65536).astype(np.float32)
            with torch.inference_mode():
                amplitude = torch.as_tensor(delta, device=torch_device) @ kernel.T
                amplitude = amplitude.reshape(-1, 256, 48)
                amplitude_host = np.asarray(amplitude.cpu().numpy(), dtype=np.float32)
            features, names = build_causal_prefix_features(
                amplitude_host,
                fixed_margin_response=response,
                profile_margin=clean_margin,
                group_weights=groups,
                group_names=group_names,
                operation_names=("qkv", "proj", "fc1", "fc2"),
                block_group_counts=group_counts,
                block_group_label=group_label,
            )
            if names_reference is None:
                names_reference = list(names)
            elif list(names) != names_reference:
                raise RuntimeError("causal-prefix schema changed between chunks")
            outputs.append(features.astype(np.float32))
            print(f"[{model}] c rows={stop}/{row_count} elapsed={time.time() - started:.1f}s", flush=True)
    finally:
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    values = np.concatenate(outputs, axis=0)
    if not np.isfinite(values).all():
        raise RuntimeError("c features contain non-finite values")
    return values, list(names_reference or [])


def write_block(output_root: Path, model: str, name: str, values: np.ndarray, columns: list[str], provenance: dict) -> None:
    directory = output_root / model / name
    directory.mkdir(parents=True, exist_ok=True)
    values_path = directory / "values.npy"
    np.save(values_path, np.ascontiguousarray(values, dtype=np.float32))
    (directory / "columns.txt").write_text("\n".join(columns) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=["swin_tiny"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--features", nargs="*", default=["s", "c"])
    parser.add_argument("--profile-mode", choices=("raw500", "cache"), default="raw500")
    args = parser.parse_args()

    population_path = Path(args.asset_root) / "population.csv"
    manifest = {"schema_version": "e07_calib_features_v1", "asset_root": str(args.asset_root), "models": {}}
    for model in args.models:
        entry: dict = {}
        if "s" in args.features:
            values, columns = build_s_features(population_path, Path(args.asset_root))
            write_block(args.output_root, model, "s_multi", values, columns, {})
            entry["s_multi"] = {"shape": list(values.shape), "columns_path": "columns.txt"}
            print(f"[{model}] s shape={values.shape}", flush=True)
        if "c" in args.features and model in BLOCK_GROUP_PARAMS:
            values, columns = build_c_features(
                model, population_path, Path(args.asset_root), args.device, args.chunk_size, args.profile_mode
            )
            write_block(args.output_root, model, "c_true", values, columns, {})
            entry["c_true"] = {"shape": list(values.shape)}
            print(f"[{model}] c shape={values.shape}", flush=True)
        manifest["models"][model] = entry
    (args.output_root / "features_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
