"""Frozen Swin s+c proxy strata for the independent A1 calibration protocol."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BIN_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf)


def proxy_bins(proxy_drop_pp: np.ndarray) -> np.ndarray:
    return np.digitize(np.asarray(proxy_drop_pp), BIN_EDGES[1:-1], right=False)


def score_pool(
    row_indexes: np.ndarray,
    *,
    checkpoint: Path,
    common3103: Path,
    conditioned1034: Path,
) -> pd.DataFrame:
    """Apply the frozen stratification checkpoint; do not train an ablation model."""
    import lightgbm as lgb

    indexes = np.asarray(row_indexes, dtype=np.int64)
    common = np.asarray(np.load(common3103, mmap_mode="r", allow_pickle=False), dtype=np.float32)[indexes]
    conditioned = np.asarray(np.load(conditioned1034, mmap_mode="r", allow_pickle=False), dtype=np.float32)[indexes]
    booster = lgb.Booster(model_file=str(checkpoint))
    proxy = np.clip(booster.predict(np.concatenate([common, conditioned], axis=1)), -5.0, 100.0)
    return pd.DataFrame({"row_index": indexes, "proxy_pp": proxy, "bin": proxy_bins(proxy)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="Candidate CSV with row_index")
    parser.add_argument("--checkpoint", type=Path, required=True, help="External frozen A04 stratification checkpoint")
    parser.add_argument("--common3103", type=Path, required=True)
    parser.add_argument("--conditioned1034", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = pd.read_csv(args.pool)
    result = score_pool(pool["row_index"].to_numpy(np.int64), checkpoint=args.checkpoint,
                        common3103=args.common3103, conditioned1034=args.conditioned1034)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
