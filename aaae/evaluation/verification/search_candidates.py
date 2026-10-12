"""Same-image Swin verification of candidates produced by three-objective search."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def verify_candidates(
    candidates: pd.DataFrame,
    *,
    gpu_id: int = 0,
    batch_size: int = 16,
    runtime_options: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Re-run supplied low_0..low_8 designs; do not alter the optimizer's choice."""
    from aaae.optimization_algorithm.nsga2.evaluators import CandidateConfig, DMultTruthTableEvaluator
    from .true1000_runtime import True1000RuntimeEvaluator

    configs = []
    for row in candidates.to_dict(orient="records"):
        config = CandidateConfig(tuple(int(row[f"low_{index}"]) for index in range(9)))
        if "config_hash" in row and str(row["config_hash"]) != config.config_hash():
            raise ValueError("candidate tokens do not match the recorded configuration hash")
        configs.append(config)
    if len({config.config_hash() for config in configs}) != len(configs):
        raise ValueError("verification candidates must have unique configuration hashes")
    options = dict(runtime_options or {})
    options.setdefault("gpu_id", gpu_id)
    options.setdefault("batch_size", batch_size)
    decoder = DMultTruthTableEvaluator(device=f"cuda:{gpu_id}")
    runtime = True1000RuntimeEvaluator(**options)
    rows = []
    try:
        for config in configs:
            truth = decoder.predict(config)
            metrics = runtime.evaluate_one(config_hash=config.config_hash(), runtime_delta_i32=truth["delta"])
            rows.append({"config_hash": config.config_hash(),
                         "verified_top1_drop_pp": float(metrics["top1_drop"]) * 100.0,
                         "verified_top5_drop_pp": float(metrics["top5_drop"]) * 100.0,
                         **{f"low_{i}": token for i, token in enumerate(config.low_tokens)}})
    finally:
        runtime.close()
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    result = verify_candidates(pd.read_csv(args.candidates), gpu_id=args.gpu_id, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
