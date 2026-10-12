"""E7: spectrum-scan calibration candidates and select 3000 with Swin-drop coverage.

The unused canonical catalog rows are concentrated at high error, so random
sampling cannot cover the low accuracy-drop intervals that E7 screens
(tau = 0.5/1/2/5 p.p.).  This script samples candidates from controlled token
ranges, decodes their LUTs, behavior-dedups against the 20k protocol pool,
builds s + Swin c features, predicts the frozen Swin A04 proxy, and selects
3000 candidates with a per-bin minimum quota over the predicted drop.

Outputs (under --output-root):
- scan_summary.json
- selection.csv
- selected_s.npy / selected_c.npy / selected_proxy.npy

Usage:
    python \
        -m aaae.calibration.preparation.scan_and_select_calib \
        --output-root experiment/e07_conformal_screening/artifacts/calib3000 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT
E7_ROOT = OUTPUT_ROOT / "calibration"
from aaae.paths import EA2_ROOT as EA2_ROOT
E2E_ROOT = EA2_ROOT / "end_to_end_prediction"

from aaae.multiplier_generator.materialize_asset import (  # noqa: E402
    _load_decoder,
    lowonly_config_hash,
)
from aaae.accuracy_predictor.cnn.behavior import behavior_sha256_rows  # noqa: E402
from aaae.multiplier_generator.lut_domains import (  # noqa: E402
    s8s8_exact_table_np,
    u8_lut_to_s8s8_sign_magnitude,
)
from aaae.multiplier_generator.truth_table import UINT16_MAX  # noqa: E402
from aaae.accuracy_predictor.features.static.feature_builder import (  # noqa: E402
    build_common3103_features,
)

DEFAULT_PROTOCOL_BEHAVIOR = EA2_ROOT / "experiment/probe_budget/artifacts/protocol_v1/behavior_sha256.npy"
DEFAULT_EXISTING_POPULATION = E2E_ROOT / "datasets/multiplier/lowonly20k_s8s8_sign_magnitude_v1/population.csv"
DEFAULT_CHECKPOINT = EA2_ROOT / "approx_error_model/runs/truth_table/full17_neural_logic/checkpoints/best.pt"
DEFAULT_PROXY_CHECKPOINT = (
    EA2_ROOT
    / "experiment/e04_feature_ablation/runs/p4_frozen/swin_tiny/A04/seed_20260921/checkpoint.txt"
)
DEFAULT_PROXY_MANIFEST = (
    EA2_ROOT
    / "experiment/e04_feature_ablation/runs/p3_validation/swin_tiny/A04/run_manifest.json"
)
PROFILE_ROOT = EA2_ROOT / "experiment/probe_budget/artifacts/profile256_v1/swin_tiny"
BLOCK_GROUP_COUNTS = (2, 2, 6, 2)
BLOCK_GROUP_LABEL = "stage"

DEFAULT_GROUPS = (
    ("g1_1", 1, 1, 1500),
    ("g1_2", 1, 2, 4500),
    ("g1_3", 1, 3, 6000),
    ("g2_4", 2, 4, 6000),
    ("g3_5", 3, 5, 6000),
    ("g4_6", 4, 6, 6000),
    ("g5_7", 5, 7, 6000),
    ("g6_8", 6, 8, 6000),
    ("uniform", 1, 8, 8000),
)
EXTRA_LOW_TAIL_GROUPS = (
    ("lowtail_eight1_one234", 4000),
    ("lowtail_seven1_two234", 6000),
    ("lowtail_mix1_p80_234", 8000),
)


def low_tail_sampler(name: str):
    def eight_one(rng: np.random.Generator, rows: int) -> np.ndarray:
        tokens = np.ones((rows, 9), dtype=np.int64)
        positions = rng.integers(0, 9, size=rows)
        values = rng.integers(2, 5, size=rows)
        tokens[np.arange(rows), positions] = values
        return tokens

    def seven_one(rng: np.random.Generator, rows: int) -> np.ndarray:
        tokens = np.ones((rows, 9), dtype=np.int64)
        for offset in range(rows):
            positions = rng.choice(9, size=2, replace=False)
            tokens[offset, positions] = rng.integers(2, 5, size=2)
        return tokens

    def mix_one(rng: np.random.Generator, rows: int) -> np.ndarray:
        tokens = rng.integers(2, 5, size=(rows, 9))
        mask = rng.random((rows, 9)) < 0.8
        tokens[mask] = 1
        return tokens

    return {
        "lowtail_eight1_one234": eight_one,
        "lowtail_seven1_two234": seven_one,
        "lowtail_mix1_p80_234": mix_one,
    }[name]
BIN_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf)
BIN_MIN_QUOTA = {0: 300, 1: 400, 2: 500, 3: 500, 4: 300, 5: 0}
TOTAL_SELECTION = 3000
SELECTION_SEED = 20260929


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_swin_profiles():
    from aaae.accuracy_predictor.features.transformer.physics_core import profile_group_weights
    from aaae.accuracy_predictor.features.transformer.prepare_causal_prefix_features import (
        compact_profile_groups,
    )
    from aaae.accuracy_predictor.features.transformer.prepare_swin_linear48_causal_prefix_features import (
        _load_profile_contracts,
    )

    operand_path = PROFILE_ROOT / "operand_profile256.float32.npy"
    propagation_path = PROFILE_ROOT / "propagation_profile256.npz"
    operand, _local, _union, response, clean_margin, _nodes, contracts = _load_profile_contracts(
        operand_path, propagation_path, Path("/nonexistent/not_used_for_raw500")
    )
    full_groups, full_names = profile_group_weights(clean_margin, contracts["exact_correct"])
    groups, group_names = compact_profile_groups(full_groups, full_names)
    return {
        "operand": operand,
        "response": response,
        "clean_margin": clean_margin,
        "groups": groups,
        "group_names": group_names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol-behavior", type=Path, default=DEFAULT_PROTOCOL_BEHAVIOR)
    parser.add_argument("--existing-population", type=Path, default=DEFAULT_EXISTING_POPULATION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--proxy-checkpoint", type=Path, default=DEFAULT_PROXY_CHECKPOINT)
    parser.add_argument("--proxy-manifest", type=Path, default=DEFAULT_PROXY_MANIFEST)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=256)
    parser.add_argument("--budget-rows", type=int, default=0, help="0 keeps the default group sizes")
    parser.add_argument("--extra-low-tail", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb

    protocol_hashes = set(np.load(args.protocol_behavior, allow_pickle=False).astype(str))
    existing = pd.read_csv(args.existing_population, low_memory=False)
    existing_configs = set(existing["config_hash"].astype(str))
    proxy_manifest = json.loads(Path(args.proxy_manifest).read_text(encoding="utf-8"))
    source_order = [source["block"] for source in proxy_manifest["sources"]]
    if source_order != ["s_multi", "c_true"]:
        raise SystemExit(f"unexpected A04 source order {source_order}")
    booster = lgb.Booster(model_file=str(args.proxy_checkpoint))

    rng = np.random.default_rng(SELECTION_SEED)
    device = torch.device(args.device)
    decoder, exact_batched, decoder_slots = _load_decoder(Path(args.checkpoint), device=device)
    exact = s8s8_exact_table_np()

    profiles = load_swin_profiles()
    from aaae.accuracy_predictor.features.transformer.prepare_causal_prefix_features import (
        build_causal_prefix_features,
    )

    reference_c_names_path = PROFILE_ROOT / "causal_prefix_second_order1034.float32.npy.features.json"
    reference_c_names = json.loads(reference_c_names_path.read_text(encoding="utf-8"))
    reference_c_names = [str(name) for name in reference_c_names]

    torch_device = torch.device(args.device)
    kernel = torch.as_tensor(np.asarray(profiles["operand"]), dtype=torch.float32)
    kernel = kernel.reshape(256 * 48, 65536).to(torch_device)

    seen: set[str] = set()
    records: list[dict] = []
    s_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    proxy_parts: list[np.ndarray] = []
    dedup_protocol = 0
    dedup_internal = 0
    started = time.time()
    try:
        group_specs: list[tuple[str, object, int]] = [
            (
                name,
                (lambda rng, rows, lo=token_lo, hi=token_hi: rng.integers(lo, hi + 1, size=(rows, 9))),
                group_rows,
            )
            for group_name, token_lo, token_hi, group_rows in DEFAULT_GROUPS
            for name in [group_name]
        ]
        if args.extra_low_tail:
            group_specs.extend(
                (name, low_tail_sampler(name), rows) for name, rows in EXTRA_LOW_TAIL_GROUPS
            )
        for group_name, sampler, group_rows in group_specs:
            remaining = int(args.budget_rows or group_rows)
            scanned = 0
            while remaining > 0:
                batch_rows = min(int(args.batch_rows), remaining)
                remaining -= batch_rows
                tokens = sampler(rng, batch_rows)
                if decoder_slots == 17:
                    slot_rows = np.concatenate([tokens, np.zeros((batch_rows, 8), dtype=np.int64)], axis=1)
                else:
                    slot_rows = tokens
                with torch.no_grad():
                    delta = decoder(torch.as_tensor(slot_rows, dtype=torch.long, device=device))["delta"]
                    u8_truth = torch.clamp(torch.round(delta + exact_batched), 0.0, float(UINT16_MAX))
                u8_tables = u8_truth.detach().cpu().numpy().astype(np.uint16)
                if np.any(u8_tables[:, 0, :]) or np.any(u8_tables[:, :, 0]):
                    raise RuntimeError("decoded U8 LUT violates the zero-operand contract")
                signed = u8_lut_to_s8s8_sign_magnitude(u8_tables).astype(np.int32)
                residual = signed - exact[None, :, :]
                hashes = behavior_sha256_rows(residual)

                keep_indices: list[int] = []
                keep_hashes: list[str] = []
                for offset, behavior in enumerate(hashes):
                    scanned += 1
                    if behavior in protocol_hashes:
                        dedup_protocol += 1
                        continue
                    if behavior in seen:
                        dedup_internal += 1
                        continue
                    seen.add(behavior)
                    keep_indices.append(offset)
                    keep_hashes.append(behavior)
                if not keep_indices:
                    continue
                kept_tokens = tokens[keep_indices]
                kept_delta = residual[keep_indices]
                metadata = pd.DataFrame(
                    {
                        "low_tokens": [list(row) for row in kept_tokens.tolist()],
                        "high_tokens": [[0] * 8 for _ in keep_indices],
                    }
                )
                s_batch, _shape = build_common3103_features(
                    metadata=metadata, delta_bank=kept_delta, exact_table=exact
                )
                with torch.inference_mode():
                    amplitude = torch.as_tensor(
                        kept_delta.reshape(len(keep_indices), -1).astype(np.float32), device=torch_device
                    ) @ kernel.T
                    amplitude = amplitude.reshape(-1, 256, 48)
                    amplitude_host = np.asarray(amplitude.cpu().numpy(), dtype=np.float32)
                c_batch, names = build_causal_prefix_features(
                    amplitude_host,
                    fixed_margin_response=profiles["response"],
                    profile_margin=profiles["clean_margin"],
                    group_weights=profiles["groups"],
                    group_names=profiles["group_names"],
                    operation_names=("qkv", "proj", "fc1", "fc2"),
                    block_group_counts=BLOCK_GROUP_COUNTS,
                    block_group_label=BLOCK_GROUP_LABEL,
                )
                if reference_c_names is not None and list(names) != reference_c_names:
                    raise RuntimeError("c feature names differ from the frozen swin c_true schema")
                proxy_raw = booster.predict(np.concatenate([s_batch, c_batch], axis=1))
                proxy = np.clip(proxy_raw, -5.0, 100.0)

                for local, offset in enumerate(keep_indices):
                    token_tuple = tuple(int(v) for v in tokens[offset].tolist())
                    records.append(
                        {
                            "group": group_name,
                            "low_tokens": list(token_tuple),
                            "config_hash": lowonly_config_hash(token_tuple),
                            "behavior_sha256": keep_hashes[local],
                            "proxy_raw": float(proxy_raw[local]),
                            "proxy_clipped": float(proxy[local]),
                        }
                    )
                s_parts.append(s_batch.astype(np.float32))
                c_parts.append(c_batch.astype(np.float32))
                proxy_parts.append(proxy.astype(np.float32))
            print(
                f"[scan] {group_name} accepted_total={len(records)} scanned={scanned} "
                f"dup_protocol={dedup_protocol} dup_internal={dedup_internal} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scan = pd.DataFrame.from_records(records)
    if scan.empty:
        raise SystemExit("scan produced no candidates")
    s_matrix = np.concatenate(s_parts, axis=0)
    c_matrix = np.concatenate(c_parts, axis=0)
    proxy_all = np.concatenate(proxy_parts, axis=0)
    if len(scan) != len(proxy_all):
        raise RuntimeError("scan record/feature length mismatch")

    # candidate config_hash collisions with the existing pool are impossible here
    if scan["config_hash"].isin(existing_configs).any():
        raise RuntimeError("scan contains config hashes that exist in the 20k pool")

    bins = np.digitize(proxy_all, BIN_EDGES[1:-1], right=False)
    scan["bin"] = bins
    available = {int(b): int((bins == b).sum()) for b in range(len(BIN_EDGES) - 1)}
    quota: dict[int, int] = {}
    for bin_index, minimum in BIN_MIN_QUOTA.items():
        quota[bin_index] = min(int(minimum), available.get(bin_index, 0))
    assigned = sum(quota.values())
    if assigned < TOTAL_SELECTION:
        remaining_bins = [b for b in quota if available.get(b, 0) > quota[b]]
        weights = np.asarray([available[b] - quota[b] for b in remaining_bins], dtype=np.float64)
        weights = weights / weights.sum()
        extra = TOTAL_SELECTION - assigned
        for bin_index, weight in zip(remaining_bins, weights):
            quota[bin_index] += int(round(weight * extra))
    selected_indices: list[int] = []
    select_rng = np.random.default_rng(SELECTION_SEED + 1)
    for bin_index in sorted(quota):
        pool = np.flatnonzero(bins == bin_index)
        take = min(quota[bin_index], len(pool))
        if take:
            chosen = select_rng.choice(pool, size=take, replace=False)
            selected_indices.extend(int(v) for v in chosen)
    selected_indices = sorted(selected_indices)
    if len(selected_indices) > TOTAL_SELECTION:
        drop = select_rng.choice(np.asarray(selected_indices), size=len(selected_indices) - TOTAL_SELECTION, replace=False)
        keep = np.setdiff1d(np.asarray(selected_indices), drop, assume_unique=True)
        selected_indices = sorted(int(v) for v in keep)
    if len(selected_indices) < TOTAL_SELECTION:
        shortfall = TOTAL_SELECTION - len(selected_indices)
        remaining_pool = np.setdiff1d(np.arange(len(scan)), np.asarray(selected_indices), assume_unique=False)
        extra = select_rng.choice(remaining_pool, size=shortfall, replace=False)
        selected_indices.extend(int(v) for v in extra)
        selected_indices = sorted(selected_indices)

    selection = scan.iloc[selected_indices].reset_index(drop=True)
    selection.insert(0, "selection_index", np.arange(len(selection), dtype=np.int64))
    selection.to_csv(output_root / "selection.csv", index=False)
    np.save(output_root / "selected_s.npy", s_matrix[selected_indices])
    np.save(output_root / "selected_c.npy", c_matrix[selected_indices])
    np.save(output_root / "selected_proxy.npy", proxy_all[selected_indices])

    realized = {
        f"[{BIN_EDGES[b]},{BIN_EDGES[b + 1]})": int((selection["bin"] == b).sum())
        for b in range(len(BIN_EDGES) - 1)
    }
    summary = {
        "schema_version": "e07_calib_scan_selection_v1",
        "scan_rows": int(len(scan)),
        "unique_behaviors_scanned": int(len(seen)),
        "dedup_protocol": int(dedup_protocol),
        "dedup_internal": int(dedup_internal),
        "proxy_checkpoint": str(args.proxy_checkpoint),
        "proxy_manifest": str(args.proxy_manifest),
        "protocol_behavior": str(args.protocol_behavior),
        "bin_edges": [float(v) if np.isfinite(v) else "inf" for v in BIN_EDGES],
        "available_by_bin": available,
        "quota_by_bin": quota,
        "selected_by_bin": realized,
        "selection_rows": int(len(selection)),
        "selection_seed": SELECTION_SEED,
        "selection_config_hash_sha256": hashlib.sha256(
            ",".join(selection["config_hash"].astype(str)).encode("ascii")
        ).hexdigest(),
        "proxy_stats_selected": {
            "min": float(proxy_all[selected_indices].min()),
            "p25": float(np.percentile(proxy_all[selected_indices], 25)),
            "median": float(np.median(proxy_all[selected_indices])),
            "p75": float(np.percentile(proxy_all[selected_indices], 75)),
            "p90": float(np.percentile(proxy_all[selected_indices], 90)),
            "max": float(proxy_all[selected_indices].max()),
        },
        "s_feature_shape": list(s_matrix.shape),
        "c_feature_shape": list(c_matrix.shape),
        "c_feature_names": list(names),
    }
    write_json(output_root / "scan_summary.json", summary)
    print(json.dumps(realized, indent=1), flush=True)
    print(f"wrote {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
