"""Create an AAAE query plan or replay fixed-pool selection from external labels."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .core import admission_order, select_with_budget


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="CSV: config_hash, yhat_ours (p.p.), pdp_fj")
    parser.add_argument("--correction", type=Path, required=True, help="A1 correction JSON from fit_calibration")
    parser.add_argument("--tau", type=float, required=True, help="Top-1 drop threshold, percentage points")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--labels", type=Path, help="Optional explicit replay CSV: config_hash, y (p.p.)")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")
    pool = pd.read_csv(args.pool)
    correction = json.loads(args.correction.read_text(encoding="utf-8"))
    if correction.get("units") != "percentage_points" or correction.get("rule") != "A1_pool_weighted_positive_residual":
        raise ValueError("expected an A1 correction in percentage points")
    q = float(correction["correction_pp"])
    predictions = pool["yhat_ours"].to_numpy(float)
    pdp = pool["pdp_fj"].to_numpy(float)
    hashes = pool["config_hash"].astype(str).tolist()
    admitted = admission_order(predictions, pdp, hashes, correction_pp=q, tau_pp=args.tau)
    query = pool.iloc[admitted[:args.budget]].copy()
    query["upper_drop_pp"] = query["yhat_ours"] + q
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query.to_csv(args.output_dir / "query_plan.csv", index=False)
    payload = {"status": "planned" if len(admitted) else "abstention", "admitted_count": len(admitted),
               "query_count": len(query), "tau_pp": args.tau, "budget": args.budget, "correction": correction}
    if args.labels is not None:
        labels = pd.read_csv(args.labels)
        if labels["config_hash"].duplicated().any():
            raise ValueError("replay labels have duplicate configuration hashes")
        bank = labels.set_index("config_hash")["y"]
        result = select_with_budget(predictions, pdp, hashes, lambda position: bank.loc[hashes[position]],
                                    correction_pp=q, tau_pp=args.tau, budget=args.budget)
        payload.update(asdict(result))
        payload["label_source"] = str(args.labels.resolve())
        payload["label_queries_replayed"] = True
        query["observed_drop_pp"] = result.observed_drop_pp
        query.to_csv(args.output_dir / "queried_candidates.csv", index=False)
        selected = pool.iloc[[result.selected_position]] if result.selected_position is not None else pool.iloc[:0]
        selected.to_csv(args.output_dir / "selected_candidate.csv", index=False)
    (args.output_dir / "selection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"AAAE {payload['status']}: admitted={len(admitted)}, queries={len(query)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
