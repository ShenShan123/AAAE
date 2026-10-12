"""E7: sealed (E3) Ours predictions for the calibration pool.

Two modes:

- ``--validate-e3`` reproduces the E3 sealed Test predictions from existing
  dataset assets and compares them with ``predictions/<model>__ours.csv`` to pin
  the exact API/scope used by E3.
- ``--predict`` builds calibration-pool predictions: common3103 + causal-prefix
  supplemental features + runtime observations on the package's own selected 50
  images (produced by the runner).

Outputs predictions in p.p. (proportion x 100).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

PACKAGES = {
    "swin_tiny": E2E_ROOT
    / "data/runs/swin_tiny_linear48_e2e_joint4497_20260801/strict50_package",
    "vit_base": E2E_ROOT
    / "data/runs/vit_base_patch16_224_int8_s8s8_linear48_structure_aware_20k_50shot_v1/strict50_package",
    "deit_base": E2E_ROOT
    / "data/runs/deit_base_patch16_224_int8_s8s8_linear48_structure_aware_20k_50shot_v1/strict50_package",
}
DATASETS = {
    "swin_tiny": E2E_ROOT / "datasets/swin_tiny/s8s8_linear48_strict50_true1000_raw500_v1",
    "vit_base": E2E_ROOT / "datasets/vit/s8s8_linear48_strict50_true1000_raw500_v1",
    "deit_base": E2E_ROOT / "datasets/deit/s8s8_linear48_strict50_true1000_raw500_v1",
}
C_FEATURES = {
    model: EA2_ROOT / "experiment/probe_budget/artifacts/profile256_v1" / model
    / "causal_prefix_second_order1034.float32.npy"
    for model in PACKAGES
}
C_FEATURES_CACHE = {
    "swin_tiny": E2E_ROOT / "experiment/cache/linear48_a100_causal_prefix_second_order.float32.npy",
    "vit_base": E2E_ROOT / "experiment/cache/vit_linear48_a100_causal_prefix_second_order.float32.npy",
    "deit_base": E2E_ROOT / "experiment/cache/deit_linear48_a100_causal_prefix_second_order.float32.npy",
}
E3_PREDICTIONS = EA2_ROOT / "experiment/e03_main_accuracy_prediction/results/predictions"


def _selected_images(package: Path) -> np.ndarray:
    frame = pd.read_csv(package / "data" / "selected_images.csv")
    return frame["image_index_in_runtime_bank"].to_numpy(np.int64)


def _c_path(model: str, mode: str) -> Path:
    return C_FEATURES_CACHE[model] if mode == "cache" else C_FEATURES[model]


def _c_names(model: str, mode: str = "cache") -> list[str]:
    path = Path(str(_c_path(model, mode)) + ".features.json")
    return [str(name) for name in json.loads(path.read_text(encoding="utf-8"))]


def _load_predictor(package: Path):
    from aaae.accuracy_predictor.transformer.artifacts.strict50_package import load_strict50_top1_package

    return load_strict50_top1_package(package)


def _features_from_observations(
    predictor,
    *,
    common: np.ndarray,
    c_features: np.ndarray,
    c_names: list[str],
    image_indices: np.ndarray,
    clean_top1: np.ndarray,
    clean_top5: np.ndarray,
    approx_top1: np.ndarray,
    approx_top5: np.ndarray,
    dm1: np.ndarray,
    dm5: np.ndarray,
):
    return predictor.build_features_from_runtime_observations(
        common3103_features=common,
        image_indices=image_indices,
        clean_top1_correct_bits=clean_top1,
        clean_top5_correct_bits=clean_top5,
        approx_top1_correct_bits=approx_top1,
        approx_top5_correct_bits=approx_top5,
        delta_margin_top1=dm1,
        delta_margin_top5=dm5,
        supplemental_features=c_features,
        supplemental_feature_names=c_names,
    )


def validate_e3(args: argparse.Namespace) -> int:
    summary = {}
    for model, package in PACKAGES.items():
        predictor = _load_predictor(package)
        dataset = DATASETS[model]
        bank = np.load(dataset / "data" / "runtime_bank.npz", allow_pickle=False)
        meta = np.load(dataset / "shared" / "strict50_bank_meta.npz", allow_pickle=False)
        selected = _selected_images(package)
        common = np.asarray(
            np.load(dataset / "data" / "common3103.npy", mmap_mode="r"), dtype=np.float32
        )
        c_features = np.asarray(np.load(_c_path(model, args.c_mode), mmap_mode="r"), dtype=np.float32)
        features = _features_from_observations(
            predictor,
            common=common,
            c_features=c_features,
            c_names=_c_names(model),
            image_indices=selected,
            clean_top1=np.asarray(meta["exact_top1_correct_bits"], dtype=np.float32)[selected],
            clean_top5=np.asarray(meta["exact_top5_correct_bits"], dtype=np.float32)[selected],
            approx_top1=np.asarray(bank["approx_top1_correct_bits"], dtype=np.float32)[:, selected],
            approx_top5=np.asarray(bank["approx_top5_correct_bits"], dtype=np.float32)[:, selected],
            dm1=np.asarray(bank["delta_margin_top1"], dtype=np.float32)[:, selected],
            dm5=np.asarray(bank["delta_margin_top5"], dtype=np.float32)[:, selected],
        )
        reference = pd.read_csv(package / "data" / "holdout_predictions.csv", low_memory=False)
        if "row_index" in reference.columns:
            rows = reference["row_index"].to_numpy(np.int64)
        else:
            rows = np.arange(len(reference), dtype=np.int64)
        expected = reference["pred"].to_numpy(np.float64)
        if float(np.abs(expected).max()) <= 1.5:
            expected = expected * 100.0
        results = {}
        for use_refit, scope in ((True, None), (False, None), (True, "deployment"), (True, "refit")):
            key = f"use_refit={use_refit},scope={scope}"
            prediction = predictor.predict_top1_from_features(
                features[rows], use_refit=use_refit, scope=scope
            ) * 100.0
            diff = float(np.abs(prediction - expected).max())
            results[key] = diff
        best = min(results, key=results.get)
        summary[model] = {"max_abs_diff_by_call": results, "best": best}
        print(f"[{model}] best={best} diffs={results}", flush=True)
    (Path(args.output_dir) / "e3_reproduction_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def predict(args: argparse.Namespace) -> int:
    import torch

    from aaae.inference_runtime.transformer.dataset_generation.profiles import get_dataset_profile
    from aaae.accuracy_predictor.transformer.proxy_training.transformer_ame_bank import compute_runtime_bank_from_logits
    from aaae.inference_runtime.transformer import run_swin_s8s8_bulk as bulk
    from aaae.inference_runtime.transformer.s8s8_truth_lut_asset import load_s8s8_sign_magnitude_truth_lut_asset

    asset = load_s8s8_sign_magnitude_truth_lut_asset(args.asset_root)
    features_root = Path(args.features_root)
    device = int(args.device)
    if torch.cuda.is_available():
        torch.cuda.init()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]

    for model, package in PACKAGES.items():
        if args.models and model not in args.models:
            continue
        predictor = _load_predictor(package)
        dataset = DATASETS[model]
        shared = dataset / "shared"
        meta = np.load(shared / "strict50_bank_meta.npz", allow_pickle=False)
        union_positions = np.load(shared / "union_positions.npz", allow_pickle=False)
        raw_positions = np.asarray(union_positions["raw500_positions"], dtype=np.int64)
        selected = _selected_images(package)
        image_positions = raw_positions[selected]
        union_inputs = np.asarray(np.load(shared / "union_inputs.npy", mmap_mode="r"), dtype=np.float32)
        inputs = np.ascontiguousarray(union_inputs[image_positions])

        profile = get_dataset_profile(
            {"swin_tiny": "swin_tiny_patch4_window7_224", "vit_base": "vit_base_patch16_224",
             "deit_base": "deit_base_patch16_224"}[model]
        )
        node_names = bulk._load_node_names(
            profile.node_registry_path, profile.dynamic_model_path,
            model_key={"swin_tiny": "swin_tiny_patch4_window7_224", "vit_base": "vit_base_patch16_224",
                       "deit_base": "deit_base_patch16_224"}[model],
        )
        runner = bulk._build_runner(
            factory_spec=bulk.DEFAULT_RUNNER_FACTORY,
            model_path=profile.dynamic_model_path,
            node_names=node_names,
            inputs=inputs,
            gpu_id=device,
            batch_size=int(args.batch_size),
            s8_residual_abs_bound=int(bulk.DEFAULT_S8_RESIDUAL_ABS_BOUND),
            session_profile="throughput",
            input_residency="staged",
            static_shape_reference_path=profile.static_model_path,
        )
        rows = len(asset.population)
        approx_top1 = np.empty((rows, len(selected)), dtype=np.float32)
        approx_top5 = np.empty((rows, len(selected)), dtype=np.float32)
        dm1 = np.empty((rows, len(selected)), dtype=np.float32)
        dm5 = np.empty((rows, len(selected)), dtype=np.float32)
        started = time.time()
        try:
            for index in range(rows):
                residual = np.asarray(asset.tables[index], dtype=np.int32) - exact
                runner.update_lut(residual)
                approx_logits = np.asarray(runner.run_all(), dtype=np.float32)
                response = compute_runtime_bank_from_logits(
                    approx_logits,
                    clean_margin_top1=np.asarray(meta["clean_margin_top1"], dtype=np.float32)[selected],
                    clean_margin_top5=np.asarray(meta["clean_margin_top5"], dtype=np.float32)[selected],
                    exact_pred_class=np.asarray(meta["exact_pred_class"], dtype=np.int64)[selected],
                    true_label=np.asarray(meta["true_label"], dtype=np.int64)[selected],
                    exact_top1_correct_bits=np.asarray(meta["exact_top1_correct_bits"], dtype=np.uint8)[selected],
                    exact_top5_correct_bits=np.asarray(meta["exact_top5_correct_bits"], dtype=np.uint8)[selected],
                )
                approx_top1[index] = np.asarray(response["approx_top1_correct_bits"], dtype=np.float32)
                approx_top5[index] = np.asarray(response["approx_top5_correct_bits"], dtype=np.float32)
                dm1[index] = np.asarray(response["delta_margin_top1"], dtype=np.float32)
                dm5[index] = np.asarray(response["delta_margin_top5"], dtype=np.float32)
                if (index + 1) % 200 == 0 or index + 1 == rows:
                    print(f"[{model}] obs {index + 1}/{rows} elapsed={time.time() - started:.1f}s", flush=True)
        finally:
            runner.close()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        s_root = Path(args.s_root) if args.s_root else features_root
        common = np.asarray(np.load(s_root / model / "s_multi" / "values.npy", mmap_mode="r"), dtype=np.float32)
        c_features = np.asarray(
            np.load(features_root / model / "c_true" / "values.npy", mmap_mode="r"), dtype=np.float32
        )
        features = _features_from_observations(
            predictor,
            common=common,
            c_features=c_features,
            c_names=_c_names(model, args.c_mode),
            image_indices=selected,
            clean_top1=np.asarray(meta["exact_top1_correct_bits"], dtype=np.float32)[selected],
            clean_top5=np.asarray(meta["exact_top5_correct_bits"], dtype=np.float32)[selected],
            approx_top1=approx_top1,
            approx_top5=approx_top5,
            dm1=dm1,
            dm5=dm5,
        )
        prediction = predictor.predict_top1_from_features(features) * 100.0
        output = output_root / f"{model}__ours_sealed.csv"
        pd.DataFrame(
            {
                "row_index": np.arange(rows, dtype=np.int64),
                "design_id": asset.population["design_id"].astype(str),
                "pred": prediction,
            }
        ).to_csv(output, index=False, float_format="%.8g")
        print(f"[{model}] wrote {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-e3", action="store_true")
    parser.add_argument("--c-mode", choices=("cache", "profile256"), default="cache")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--features-root", type=Path)
    parser.add_argument("--s-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_e3:
        return validate_e3(args)
    if args.predict:
        if args.asset_root is None or args.features_root is None:
            raise SystemExit("--predict requires --asset-root and --features-root")
        return predict(args)
    raise SystemExit("choose --validate-e3 or --predict")


if __name__ == "__main__":
    raise SystemExit(main())
