"""Fit the paper's A1 correction from explicitly supplied external CSV files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .empirical import fit_upper_correction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True, help="CSV with y, pred and bin columns, in percentage points")
    parser.add_argument("--pool", type=Path, required=True, help="Target-pool CSV with the same frozen bin encoding")
    parser.add_argument("--model", required=True)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = pd.read_csv(args.calibration)
    pool = pd.read_csv(args.pool)
    correction = fit_upper_correction(
        calibration["y"], calibration["pred"], calibration["bin"], pool["bin"], alpha=args.alpha,
    )
    payload = correction.as_dict()
    payload.update({
        "model": args.model,
        "calibration_source": str(args.calibration.resolve()),
        "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
        "pool_source": str(args.pool.resolve()),
        "pool_sha256": hashlib.sha256(args.pool.read_bytes()).hexdigest(),
        "coverage_guarantee": "none; pool-weighted empirical correction",
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"A1 {args.model}: q={correction.correction_pp:.8g} p.p. -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
