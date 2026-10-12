"""E7: finalize calibration dataset packages (shared/, provenance/, contracts).

Given a run directory containing ``data/`` outputs, this mirrors the existing
dataset package layout under the destination root and writes dataset_contract,
package_checksums and README.

Usage:
    python -m aaae.calibration.preparation.finalize_calib_datasets \
        --kind transformer --model swin_tiny \
        --run-root <...>/runs/calib_true1000/swin_tiny \
        --dataset-root end_to_end_prediction/datasets/swin_tiny/s8s8_linear48_calib3000_true1000_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

REFERENCE = {
    "swin_tiny": E2E_ROOT / "datasets/swin_tiny/s8s8_linear48_strict50_true1000_raw500_v1",
    "vit_base": E2E_ROOT / "datasets/vit/s8s8_linear48_strict50_true1000_raw500_v1",
    "deit_base": E2E_ROOT / "datasets/deit/s8s8_linear48_strict50_true1000_raw500_v1",
    "resnet50": E2E_ROOT / "datasets/resnet50/s8s8_true1000_20k_v1",
    "mobilenetv3": E2E_ROOT / "datasets/mobilenetv3/s8s8_true1000_20k_v1",
}
SHARED_FILES = {
    "transformer": (
        "true1000_exact_meta.npz",
        "union_positions.npz",
        "frozen_image_protocols.json",
        "canonical_exact_baseline.npz",
        "strict50_bank_meta.npz",
        "input_contract.json",
    ),
    "cnn": ("input_contract.json",),
}
CNN_ASSETS = {
    "resnet50": "resnet50_true1000_seed1_preprocessed.npz",
    "mobilenetv3": "mobilenetv3_true1000_seed1_preprocessed.npz",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("transformer", "cnn"), required=True)
    parser.add_argument("--model", required=True, choices=sorted(REFERENCE))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_data = Path(args.run_root) / "data"
    if not run_data.is_dir():
        raise SystemExit(f"run data dir missing: {run_data}")
    dataset_root = Path(args.dataset_root)
    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{dataset_root} exists; pass --overwrite")
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)

    shutil.copytree(run_data, dataset_root / "data")
    shared_root = dataset_root / "shared"
    shared_root.mkdir(parents=True)
    reference = REFERENCE[args.model]
    for name in SHARED_FILES[args.kind]:
        source = reference / "shared" / name
        if source.is_file():
            shutil.copy2(source, shared_root / name)
    if args.kind == "cnn":
        asset_name = CNN_ASSETS[args.model]
        source = reference / "shared" / asset_name
        if source.is_file():
            shutil.copy2(source, shared_root / asset_name)
    shutil.copytree(reference.parent / "provenance", dataset_root / "provenance", dirs_exist_ok=False) if (
        reference.parent / "provenance"
    ).is_dir() and args.kind == "cnn" else None

    provenance_dir = dataset_root / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        provenance_dir / "promotion_manifest.json",
        {
            "schema_version": "e07_calib_promotion_v1",
            "kind": args.kind,
            "model": args.model,
            "run_root": str(Path(args.run_root).resolve()),
            "asset_root": str(Path(args.asset_root).resolve()),
            "asset_manifest_sha1": sha1_file(Path(args.asset_root) / "manifest.json"),
            "reference_dataset": str(reference),
        },
    )

    files = sorted(path for path in dataset_root.rglob("*") if path.is_file())
    package_checksums = {
        str(path.relative_to(dataset_root)): sha256_file(path)
        for path in files
        if path.name != "package_checksums.json"
    }
    write_json(dataset_root / "package_checksums.json", package_checksums)
    write_json(
        dataset_root / "dataset_contract.json",
        {
            "schema_version": "e07_calibration_dataset_contract_v1",
            "dataset_id": f"{args.model}_s8s8_calib3000_true1000_v1",
            "kind": args.kind,
            "model": args.model,
            "row_count": 3000,
            "sample_count": 1000,
            "asset_root": str(Path(args.asset_root).resolve()),
            "asset_manifest_sha1": sha1_file(Path(args.asset_root) / "manifest.json"),
            "labels": "data/labels.csv",
            "purpose": "E7 conformal calibration pool (independent of the 20k train/val/test split)",
            "test_targets_loaded": False,
        },
    )
    (dataset_root / "README.md").write_text(
        "\n".join(
            [
                f"# {args.model} calibration dataset (E7)",
                "",
                "3000 generated low-only candidates used ONLY for conformal calibration.",
                "Behavior-deduplicated against the 20k protocol pool; labels measured on the",
                "same frozen true1000 D_eval images as the main datasets.",
                "",
                f"- asset: `{Path(args.asset_root).resolve()}`",
                f"- run: `{Path(args.run_root).resolve()}`",
                f"- reference shared assets: `{reference}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"finalized {dataset_root} files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
