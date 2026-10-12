"""E7: build the CNN calibration label manifest (3000 design ids, existing D_eval)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

REFERENCE_DATASETS = {
    "resnet50": E2E_ROOT / "datasets/resnet50/s8s8_true1000_20k_v1",
    "mobilenetv3": E2E_ROOT / "datasets/mobilenetv3/s8s8_true1000_20k_v1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(REFERENCE_DATASETS))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-designs", type=int, default=16)
    parser.add_argument("--input-residency", default="auto")
    args = parser.parse_args()

    import pandas as pd

    reference = REFERENCE_DATASETS[args.model]
    snapshot = json.loads((reference / "data/label_manifest.snapshot.json").read_text(encoding="utf-8"))
    preprocessed = reference / "shared" / f"{args.model}_true1000_seed1_preprocessed.npz"
    if not preprocessed.is_file():
        raise SystemExit(f"missing preprocessed asset {preprocessed}")

    population = pd.read_csv(Path(args.asset_root) / "population.csv", low_memory=False)
    design_ids = population["runtime_design_id"].astype(str).tolist()
    if len(design_ids) != 3000 or len(set(design_ids)) != 3000:
        raise SystemExit(f"expected 3000 unique design ids, got {len(design_ids)}")

    manifest = {
        "schema_version": snapshot["schema_version"],
        "task": snapshot["task"],
        "model": snapshot["model"],
        "dataset": {
            "kind": "preprocessed_npz_v1",
            "path": str(preprocessed.resolve()),
            "sha256": sha256_file(preprocessed),
            "preprocessing_id": snapshot["dataset"]["preprocessing_id"],
            "image_root": snapshot["dataset"].get("image_root"),
            "sample_count": 1000,
        },
        "injection": snapshot["injection"],
        "designs": {
            "asset_root": str(Path(args.asset_root).resolve()),
            "design_ids": design_ids,
        },
        "execution": {
            "device": "cuda",
            "gpu_ids": [int(v) for v in args.gpu_ids],
            "batch_size": int(args.batch_size),
            "block_designs": int(args.block_designs),
            "input_residency": str(args.input_residency),
            "preallocate_outputs": True,
            "cudnn_conv_algo_search": "DEFAULT",
        },
        "output_dir": str((Path(args.run_root) / "data").resolve()),
        "labels": None,
        "provenance": {
            "kind": "e07_calibration_manifest",
            "reference_dataset": str(reference),
            "asset_root": str(Path(args.asset_root).resolve()),
            "asset_manifest_sha1": hashlib.sha1((Path(args.asset_root) / "manifest.json").read_bytes()).hexdigest(),
        },
    }
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    target = run_root / "resolved_label_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {target} design_ids={len(design_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
