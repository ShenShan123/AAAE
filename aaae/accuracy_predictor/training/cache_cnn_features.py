#!/usr/bin/env python
"""Atomically cache profile-bound CNN features by configured data split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.cnn.features import build_features
from aaae.accuracy_predictor.cnn.manifests import load_training_inputs

SPLIT_NAMES = ("train", "validation", "test")
CACHE_SCHEMA = "cnn_profile_bound_feature_cache_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cache_features(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    manifest_path = manifest_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"feature cache output already exists: {output_dir}")

    inputs = load_training_inputs(manifest_path)
    table = build_features(
        inputs.profile,
        inputs.designs,
        chunk_designs=inputs.execution.chunk_designs,
        feature_schema=inputs.feature_schema,
    )
    if table.design_ids != inputs.designs.design_ids:
        raise ValueError("computed feature rows do not match training design order")
    split_values = np.asarray(inputs.splits, dtype=str)
    counts = {name: int(np.count_nonzero(split_values == name)) for name in SPLIT_NAMES}
    if counts != dict(inputs.split_counts):
        raise ValueError(
            f"computed feature split counts differ from the manifest: {counts} "
            f"!= {dict(inputs.split_counts)}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        for name in SPLIT_NAMES:
            values = np.ascontiguousarray(table.values[split_values == name])
            np.save(staging / f"{name}.npy", values, allow_pickle=False)
        _write_json(
            staging / "feature_schema.json",
            {
                "schema_name": table.schema_name,
                "feature_count": len(table.feature_names),
                "feature_names": list(table.feature_names),
                "profile_digest": table.profile_digest,
            },
        )
        manifest = {
            "schema_version": CACHE_SCHEMA,
            "source_training_manifest": str(manifest_path),
            "source_training_manifest_sha256": _sha256(manifest_path),
            "network_id": inputs.profile.network_id,
            "dataset_id": inputs.profile.dataset_id,
            "feature_schema": table.schema_name,
            "feature_count": len(table.feature_names),
            "profile_digest": table.profile_digest,
            "split_counts": counts,
            "dtype": str(table.values.dtype),
        }
        _write_json(staging / "cache_manifest.json", manifest)
        checksums = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _write_json(
            staging / "checksums.json",
            {"algorithm": "sha256", "files": checksums},
        )
        os.replace(staging, output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **manifest,
        "output_dir": str(output_dir),
        "checksums": str(output_dir / "checksums.json"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            cache_features(args.train_manifest, args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
