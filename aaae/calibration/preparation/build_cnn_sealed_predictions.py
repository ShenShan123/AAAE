"""E7: sealed (E3) CNN Ours predictions for the calibration pool.

Uses ``CnnDropPredictor.predict(designs)`` from the frozen bundles; features are
built internally from the bundle profile.  Predictions are returned in p.p.
(proportion x 100).

Validation mode reproduces the bundle's stored test predictions on the existing
20k CNN dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

BUNDLES = {
    "resnet50": E2E_ROOT
    / "data/runs/resnet50_imagenet1k_s8s8_true1000_20k_ame_config14k3k3k_v3/model_bundle_top1_delta_conv_v7",
    "mobilenetv3": E2E_ROOT
    / "data/runs/mobilenetv3_imagenet1k_s8s8_true1000_20k_ame_config14k3k3k_v1/model_bundle_top1_correctness_conv_v3",
}
DATASETS = {
    "resnet50": E2E_ROOT / "datasets/resnet50/s8s8_true1000_20k_v1",
    "mobilenetv3": E2E_ROOT / "datasets/mobilenetv3/s8s8_true1000_20k_v1",
}


def _predictor(model: str, device: str):
    from aaae.accuracy_predictor.cnn.workflow import load_bundle

    return load_bundle(BUNDLES[model], torch_device=device, torch_batch_size=256)


def validate_e3(args: argparse.Namespace) -> int:
    summary = {}
    for model, dataset in DATASETS.items():
        predictor = _predictor(model, args.device)
        delta = np.load(dataset / "data" / "delta.npy", mmap_mode="r")
        design_ids = np.load(dataset / "data" / "design_ids.npy", allow_pickle=False).astype(str)
        from aaae.accuracy_predictor.cnn.contracts import MultiplierDesignBatch

        table = predictor.predict(
            MultiplierDesignBatch(design_ids=tuple(design_ids.tolist()), delta=np.asarray(delta), domain="s8_numeric")
        )
        prediction = np.asarray(table.top1_drop, dtype=np.float64).reshape(-1)
        reference = pd.read_csv(
            BUNDLES[model].parents[0] / "data" / "final_predictions.csv",
            low_memory=False,
        )
        reference = reference[reference["split"].eq("test") & reference["target"].eq("top1_drop")]
        rows = reference["row_index"].to_numpy(np.int64)
        expected = reference["pred"].to_numpy(np.float64)
        if float(np.abs(expected).max()) <= 1.5:
            expected = expected * 100.0
        diff = float(np.abs(prediction[rows] * 100.0 - expected).max())
        summary[model] = {"max_abs_diff_pp": diff, "rows": int(len(rows))}
        print(f"[{model}] max diff {diff} rows={len(rows)}", flush=True)
    (Path(args.output_dir) / "cnn_e3_reproduction_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def predict(args: argparse.Namespace) -> int:
    from aaae.accuracy_predictor.cnn.contracts import MultiplierDesignBatch

    asset_root = Path(args.asset_root)
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]
    tables = np.load(asset_root / "truth_tables_s8s8_sign_magnitude_int32.npy", mmap_mode="r")
    delta = np.asarray(tables, dtype=np.int32) - exact[None, :, :]
    population = pd.read_csv(asset_root / "population.csv", low_memory=False)
    design_ids = population["runtime_design_id"].astype(str).tolist()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for model in BUNDLES:
        if args.models and model not in args.models:
            continue
        predictor = _predictor(model, args.device)
        table = predictor.predict(
            MultiplierDesignBatch(design_ids=tuple(design_ids), delta=delta, domain="s8_numeric")
        )
        prediction = np.asarray(table.top1_drop, dtype=np.float64).reshape(-1) * 100.0
        output = output_root / f"{model}__ours_sealed.csv"
        pd.DataFrame(
            {
                "row_index": np.arange(len(prediction), dtype=np.int64),
                "design_id": design_ids,
                "pred": prediction,
            }
        ).to_csv(output, index=False, float_format="%.8g")
        print(f"[{model}] wrote {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-e3", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.validate_e3:
        return validate_e3(args)
    if args.predict:
        if args.asset_root is None:
            raise SystemExit("--predict requires --asset-root")
        return predict(args)
    raise SystemExit("choose --validate-e3 or --predict")


if __name__ == "__main__":
    raise SystemExit(main())
