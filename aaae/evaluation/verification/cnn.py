"""Selected-design CNN verification through the frozen 1,000-image label flow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify_candidates(
    candidates: pd.DataFrame,
    *,
    label_manifest: Path,
    work_dir: Path,
    gpu_id: int = 0,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Use explicit external model/dataset assets; create fresh verification outputs."""
    from aaae.inference_runtime.cnn.dataset_builders.real_labels import generate_labels_from_manifest

    if candidates["design_id"].duplicated().any():
        raise ValueError("verification design IDs must be unique")
    snapshot = json.loads(Path(label_manifest).read_text(encoding="utf-8"))
    work_dir = Path(work_dir)
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(f"verification work directory must be empty: {work_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    design_ids = candidates["design_id"].astype(str).tolist()
    snapshot["designs"]["design_ids"] = design_ids
    labels_dir = work_dir / "labels"
    snapshot["output_dir"] = str(labels_dir.resolve())
    snapshot["execution"]["gpu_ids"] = [int(gpu_id)]
    snapshot["execution"]["batch_size"] = int(batch_size)
    manifest_path = work_dir / "verification_manifest.json"
    manifest_path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    generate_labels_from_manifest(manifest_path)
    approx = np.load(labels_dir / "approx_top1_correct.npy", allow_pickle=False).astype(np.int16)
    exact = np.load(labels_dir / "exact_top1_correct.npy", allow_pickle=False).astype(np.int16)
    observed_ids = pd.read_csv(labels_dir / "design_ids.csv").iloc[:, 0].astype(str).tolist()
    if approx.shape != (len(design_ids), 1000) or exact.shape != (1000,):
        raise ValueError("verification output must contain exactly 1,000 paired image outcomes")
    result = candidates.set_index("design_id").loc[observed_ids].reset_index()
    result["verified_top1_drop_pp"] = (exact[None, :] - approx).mean(axis=1) * 100.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--label-manifest", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = verify_candidates(pd.read_csv(args.candidates), label_manifest=args.label_manifest,
                               work_dir=args.work_dir, gpu_id=args.gpu_id, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
