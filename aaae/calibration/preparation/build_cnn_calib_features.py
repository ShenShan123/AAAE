"""E7: CNN c features for the calibration pool (reuses the frozen ApplicationProfile)."""

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

CNN_C_INCLUDE_PREFIXES = ("paper_ame_", "weighted_", "basis_", "stage_role_")
CNN_C_EXCLUDE_PREFIXES = ("global_",)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("resnet50", "mobilenetv3"))
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chunk-designs", type=int, default=64)
    args = parser.parse_args()

    from aaae.accuracy_predictor.cnn.contracts import ApplicationProfile, MultiplierDesignBatch
    from aaae.accuracy_predictor.cnn.features import build_features

    profile_root = EA2_ROOT / "experiment/probe_budget/artifacts/cnn_profile256_v1" / args.model
    arrays = np.load(profile_root / "profile_arrays.npz", allow_pickle=False)
    metadata = json.loads((profile_root / "profile_metadata.json").read_text(encoding="utf-8"))
    static_manifest = json.loads(
        (EA2_ROOT / "experiment/probe_budget/artifacts/cnn_static_v1" / args.model / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    references = json.loads(
        (EA2_ROOT / "experiment/probe_budget/artifacts/cnn_static_v1" / args.model / "feature_names.json").read_text(
            encoding="utf-8"
        )
    )
    reference_names = references["feature_names"] if isinstance(references, dict) else references
    include = [
        index
        for index, name in enumerate(reference_names)
        if name.startswith(CNN_C_INCLUDE_PREFIXES) and not name.startswith(CNN_C_EXCLUDE_PREFIXES)
    ]

    profile = ApplicationProfile(
        network_id=str(metadata.get("network_id", args.model)),
        dataset_id=str(metadata.get("dataset_id", "")),
        domain=str(metadata.get("domain", "s8_numeric")),
        layer_ids=tuple(str(v) for v in np.asarray(arrays["layer_ids"]).astype(str).tolist()),
        activation_histograms=np.asarray(arrays["activation_histograms"], dtype=np.float64),
        weight_histograms=np.asarray(arrays["weight_histograms"], dtype=np.float64),
        mac_counts=np.asarray(arrays["mac_counts"], dtype=np.int64),
        propagation_gains=np.asarray(arrays["propagation_gains"], dtype=np.float64),
        profile_evidence="measured",
    )

    population = pd.read_csv(Path(args.asset_root) / "population.csv", low_memory=False)
    design_ids = population["runtime_design_id"].astype(str).tolist()
    codes = np.arange(-128, 128, dtype=np.int32)
    exact = codes[:, None] * codes[None, :]
    tables = np.load(Path(args.asset_root) / "truth_tables_s8s8_sign_magnitude_int32.npy", mmap_mode="r")
    delta = np.asarray(tables, dtype=np.int32) - exact[None, :, :]
    designs = MultiplierDesignBatch(design_ids=tuple(design_ids), delta=delta, domain="s8_numeric")
    table = build_features(
        profile, designs, chunk_designs=int(args.chunk_designs), feature_schema=static_manifest["feature_schema"]
    )
    values = np.asarray(table.values, dtype=np.float32)[:, include]
    names = [table.feature_names[index] for index in include]
    if values.shape[0] != 3000:
        raise SystemExit(f"unexpected feature rows {values.shape}")
    output = Path(args.output_root) / args.model / "c_true"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "values.npy", values)
    (output / "columns.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "e07_cnn_calib_features_v1",
                "model": args.model,
                "shape": list(values.shape),
                "feature_schema": static_manifest["feature_schema"],
                "profile_root": str(profile_root),
                "asset_root": str(Path(args.asset_root).resolve()),
                "asset_manifest_sha1": __import__("hashlib").sha1(
                    (Path(args.asset_root) / "manifest.json").read_bytes()
                ).hexdigest(),
                "targets_loaded": False,
                "true1000_correctness_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{args.model}: c shape={values.shape} names_match={names == [reference_names[i] for i in include]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
