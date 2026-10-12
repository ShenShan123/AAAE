"""E7: true1000 labels for the calibration pool (Transformer, multi-GPU shardable).

Runs the frozen injected model over the 1000-image D_eval for each calibration
candidate and writes E2/E3-compatible label artifacts into a calibration dataset
root.

Worker:
    ... --model swin_tiny --shard-index 0 --shard-count 2 --device cuda:0
Merge:
    ... --model swin_tiny --merge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

REFERENCE_DATASETS = {
    "swin_tiny": E2E_ROOT / "datasets/swin_tiny/s8s8_linear48_strict50_true1000_raw500_v1",
    "vit_base": E2E_ROOT / "datasets/vit/s8s8_linear48_strict50_true1000_raw500_v1",
    "deit_base": E2E_ROOT / "datasets/deit/s8s8_linear48_strict50_true1000_raw500_v1",
}
MODEL_KEYS = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "vit_base": "vit_base_patch16_224",
    "deit_base": "deit_base_patch16_224",
}


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_worker(args: argparse.Namespace) -> int:
    import torch

    from aaae.inference_runtime.transformer.dataset_generation.profiles import get_dataset_profile
    from aaae.accuracy_predictor.transformer.proxy_training.transformer_ame_bank import compute_runtime_bank_from_logits
    from aaae.inference_runtime.transformer import run_swin_s8s8_bulk as bulk
    from aaae.inference_runtime.transformer.s8s8_truth_lut_asset import load_s8s8_sign_magnitude_truth_lut_asset

    model = args.model
    reference = REFERENCE_DATASETS[model]
    shared = reference / "shared"
    asset = load_s8s8_sign_magnitude_truth_lut_asset(args.asset_root)
    meta = np.load(shared / "true1000_exact_meta.npz", allow_pickle=False)
    union_positions = np.load(shared / "union_positions.npz", allow_pickle=False)
    positions = np.asarray(union_positions["true1000_positions"], dtype=np.int64)
    if positions.shape != (1000,):
        raise RuntimeError(f"{model}: true1000 positions shape {positions.shape}")
    union_inputs = np.asarray(np.load(shared / "union_inputs.npy", mmap_mode="r"), dtype=np.float32)
    inputs = np.ascontiguousarray(union_inputs[positions])

    run_config = json.loads((reference / "data" / "manifest.json").read_text(encoding="utf-8"))
    profile = get_dataset_profile(MODEL_KEYS[model])
    node_names = bulk._load_node_names(
        profile.node_registry_path, profile.dynamic_model_path, model_key=MODEL_KEYS[model]
    )
    rows_total = len(asset.population)
    assigned = [index for index in range(rows_total) if index % args.shard_count == args.shard_index]
    print(f"[{model}] shard {args.shard_index}/{args.shard_count} candidates={len(assigned)}", flush=True)

    device_id = int(args.device)
    if torch.cuda.is_available():
        torch.cuda.init()
    runner = bulk._build_runner(
        factory_spec=bulk.DEFAULT_RUNNER_FACTORY,
        model_path=profile.dynamic_model_path,
        node_names=node_names,
        inputs=inputs,
        gpu_id=device_id,
        batch_size=int(args.batch_size),
        s8_residual_abs_bound=int(bulk.DEFAULT_S8_RESIDUAL_ABS_BOUND),
        session_profile="throughput",
        input_residency="staged",
        static_shape_reference_path=profile.static_model_path,
    )
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]
    top1 = np.empty(len(assigned), dtype=np.float32)
    top5 = np.empty(len(assigned), dtype=np.float32)
    bits1 = np.empty((len(assigned), 1000), dtype=np.uint8)
    bits5 = np.empty((len(assigned), 1000), dtype=np.uint8)
    started = time.time()
    try:
        for offset, row_index in enumerate(assigned):
            residual = np.asarray(asset.tables[row_index], dtype=np.int32) - exact
            runner.update_lut(residual)
            approx_logits = np.asarray(runner.run_all(), dtype=np.float32)
            response = compute_runtime_bank_from_logits(
                approx_logits,
                clean_margin_top1=np.asarray(meta["clean_margin_top1"], dtype=np.float32),
                clean_margin_top5=np.asarray(meta["clean_margin_top5"], dtype=np.float32),
                exact_pred_class=np.asarray(meta["exact_pred_class"], dtype=np.int64),
                true_label=np.asarray(meta["true_label"], dtype=np.int64),
                exact_top1_correct_bits=np.asarray(meta["exact_top1_correct_bits"], dtype=np.uint8),
                exact_top5_correct_bits=np.asarray(meta["exact_top5_correct_bits"], dtype=np.uint8),
            )
            top1[offset] = float(response["top1_drop"])
            top5[offset] = float(response["top5_drop"])
            bits1[offset] = np.asarray(response["approx_top1_correct_bits"], dtype=np.uint8)
            bits5[offset] = np.asarray(response["approx_top5_correct_bits"], dtype=np.uint8)
            if (offset + 1) % 50 == 0 or offset + 1 == len(assigned):
                print(
                    f"[{model}] {offset + 1}/{len(assigned)} elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    finally:
        runner.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    sample_count = 1000
    if np.any(np.abs(top1 * sample_count - np.round(top1 * sample_count)) > 1e-3):
        raise RuntimeError(f"{model}: top1_drop is not on the 1/{sample_count} grid")
    shard_dir = args.output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"true1000_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.npz"
    np.savez_compressed(
        shard_path,
        row_index=np.asarray(assigned, dtype=np.int64),
        top1_drop=top1,
        top5_drop=top5,
        approx_top1_correct_bits=bits1,
        approx_top5_correct_bits=bits5,
    )
    write_json(
        shard_dir / f"true1000_shard_{args.shard_index:03d}_of_{args.shard_count:03d}.json",
        {
            "model": model,
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "candidates": int(len(assigned)),
            "device_id": device_id,
            "batch_size": int(args.batch_size),
            "asset_root": str(args.asset_root),
            "asset_manifest_sha1": sha1_file(Path(args.asset_root) / "manifest.json"),
            "elapsed_seconds": float(time.time() - started),
            "shard_file": shard_path.name,
            "shard_sha256": sha256_file(shard_path),
        },
    )
    print(f"[{model}] wrote {shard_path}", flush=True)
    return 0


def run_merge(args: argparse.Namespace) -> int:
    model = args.model
    asset_root = Path(args.asset_root)
    output_root = Path(args.output_root)
    shard_dir = output_root / "shards"
    shards = sorted(shard_dir.glob("true1000_shard_*_of_*.npz"))
    if not shards:
        raise SystemExit(f"no shards under {shard_dir}")
    row_index = np.concatenate([np.load(path, allow_pickle=False)["row_index"] for path in shards])
    order = np.argsort(row_index, kind="stable")
    merged = {}
    for key in ("top1_drop", "top5_drop", "approx_top1_correct_bits", "approx_top5_correct_bits"):
        merged[key] = np.concatenate([np.load(path, allow_pickle=False)[key] for path in shards], axis=0)[order]
    row_index = row_index[order]
    expected = np.arange(3000, dtype=np.int64)
    if not np.array_equal(row_index, expected):
        raise SystemExit("merged shards do not cover the full calibration pool")

    import pandas as pd

    population = pd.read_csv(asset_root / "population.csv", low_memory=False)
    design_ids = population["design_id"].astype(str).to_numpy()

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    true1000_path = data_dir / "true1000_label_bank.npz"
    np.savez_compressed(
        true1000_path,
        design_id=design_ids.astype("<U128"),
        public_design_id=design_ids.astype("<U128"),
        row_index=row_index,
        exact_top1_correct_bits=np.asarray(
            np.load(REFERENCE_DATASETS[model] / "shared" / "true1000_exact_meta.npz", allow_pickle=False)[
                "exact_top1_correct_bits"
            ],
            dtype=np.uint8,
        ),
        exact_top5_correct_bits=np.asarray(
            np.load(REFERENCE_DATASETS[model] / "shared" / "true1000_exact_meta.npz", allow_pickle=False)[
                "exact_top5_correct_bits"
            ],
            dtype=np.uint8,
        ),
        approx_top1_correct_bits=merged["approx_top1_correct_bits"],
        approx_top5_correct_bits=merged["approx_top5_correct_bits"],
        top1_drop=merged["top1_drop"],
        top5_drop=merged["top5_drop"],
    )
    targets = np.column_stack([merged["top1_drop"], merged["top5_drop"]]).astype(np.float32)
    np.save(data_dir / "targets.npy", targets)
    labels = pd.DataFrame(
        {
            "row_index": row_index,
            "design_id": design_ids,
            "runtime_design_id": population["runtime_design_id"].astype(str),
            "split": "calibration",
            "top1_drop": merged["top1_drop"],
            "top5_drop": merged["top5_drop"],
        }
    )
    labels.to_csv(data_dir / "labels.csv", index=False)

    def sha1_of(path: Path) -> str:
        return hashlib.sha1(path.read_bytes()).hexdigest()

    checksums = {
        name: sha1_of(data_dir / name)
        for name in ("true1000_label_bank.npz", "targets.npy", "labels.csv")
    }
    write_json(
        data_dir / "run_manifest.json",
        {
            "schema_version": "e07_calibration_true1000_v1",
            "model": model,
            "rows": int(len(row_index)),
            "sample_count": 1000,
            "asset_root": str(asset_root),
            "asset_manifest_sha1": sha1_of(asset_root / "manifest.json"),
            "reference_dataset": str(REFERENCE_DATASETS[model]),
            "shards": [path.name for path in shards],
            "checksums": checksums,
            "test_targets_loaded": False,
            "targets_loaded_from_d_eval": True,
        },
    )
    write_json(data_dir / "checksums.json", checksums)
    print(f"[{model}] merged {len(shards)} shards rows={len(row_index)} -> {data_dir}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(REFERENCE_DATASETS))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        return run_merge(args)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
