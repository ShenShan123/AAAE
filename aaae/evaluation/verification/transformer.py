"""Literal 1,000-image verification of explicitly selected Transformer designs.

Derived from E10 verify_selected_true1000.py. Selection and extra diagnostic
sampling are performed by the caller, so this module has no baseline experiment
or fixed E10 result-directory dependency.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

MODEL_KEYS = {
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "vit_base": "vit_base_patch16_224",
    "deit_base": "deit_base_patch16_224",
}


def verify_candidates(
    model: str,
    candidates: pd.DataFrame,
    *,
    dataset_root: Path,
    truth_tables: Path,
    gpu_id: int = 0,
    batch_size: int = 16,
) -> pd.DataFrame:
    from aaae.inference_runtime.transformer.dataset_generation.profiles import get_dataset_profile
    from aaae.accuracy_predictor.transformer.proxy_training.transformer_ame_bank import compute_runtime_bank_from_logits
    from aaae.inference_runtime.transformer import run_swin_s8s8_bulk as bulk

    if model not in MODEL_KEYS:
        raise ValueError(f"unsupported Transformer model: {model}")
    if candidates["row_index"].duplicated().any():
        raise ValueError("verification row indexes must be unique")
    shared = Path(dataset_root) / "shared"
    with np.load(shared / "true1000_exact_meta.npz", allow_pickle=False) as archive:
        meta = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(shared / "union_positions.npz", allow_pickle=False) as archive:
        positions = np.asarray(archive["true1000_positions"], dtype=np.int64)
    if positions.shape != (1000,) or len(np.unique(positions)) != 1000:
        raise ValueError("verification must use the frozen 1,000-image evaluation set")
    union_inputs = np.load(shared / "union_inputs.npy", mmap_mode="r", allow_pickle=False)
    inputs = np.ascontiguousarray(union_inputs[positions], dtype=np.float32)
    profile = get_dataset_profile(MODEL_KEYS[model])
    node_names = bulk._load_node_names(profile.node_registry_path, profile.dynamic_model_path, model_key=MODEL_KEYS[model])
    runner = bulk._build_runner(
        factory_spec=bulk.DEFAULT_RUNNER_FACTORY,
        model_path=profile.dynamic_model_path,
        node_names=node_names,
        inputs=inputs,
        gpu_id=int(gpu_id),
        batch_size=int(batch_size),
        s8_residual_abs_bound=int(bulk.DEFAULT_S8_RESIDUAL_ABS_BOUND),
        session_profile="throughput",
        input_residency="staged",
        static_shape_reference_path=profile.static_model_path,
    )
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]
    tables = np.load(truth_tables, mmap_mode="r", allow_pickle=False)
    records = []
    try:
        for row in candidates.to_dict(orient="records"):
            index = int(row["row_index"])
            if index < 0 or index >= len(tables):
                raise ValueError(f"invalid multiplier table row_index: {index}")
            started = time.monotonic()
            runner.update_lut(np.asarray(tables[index], dtype=np.int32) - exact)
            response = compute_runtime_bank_from_logits(
                np.asarray(runner.run_all(), dtype=np.float32),
                clean_margin_top1=np.asarray(meta["clean_margin_top1"], dtype=np.float32),
                clean_margin_top5=np.asarray(meta["clean_margin_top5"], dtype=np.float32),
                exact_pred_class=np.asarray(meta["exact_pred_class"], dtype=np.int64),
                true_label=np.asarray(meta["true_label"], dtype=np.int64),
                exact_top1_correct_bits=np.asarray(meta["exact_top1_correct_bits"], dtype=np.uint8),
                exact_top5_correct_bits=np.asarray(meta["exact_top5_correct_bits"], dtype=np.uint8),
            )
            records.append({**row, "model": model,
                            "verified_top1_drop_pp": float(response["top1_drop"]) * 100.0,
                            "verified_top5_drop_pp": float(response["top5_drop"]) * 100.0,
                            "wall_seconds": time.monotonic() - started})
    finally:
        runner.close()
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_KEYS), required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--truth-tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = verify_candidates(args.model, pd.read_csv(args.candidates), dataset_root=args.dataset_root,
                               truth_tables=args.truth_tables, gpu_id=args.gpu_id, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
