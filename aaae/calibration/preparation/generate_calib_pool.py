"""E7: generate the calibration candidate pool (new structures, behavior-deduped).

Samples unused low-only canonical-catalog candidates deterministically, decodes
their S8S8 sign-magnitude LUTs with the frozen neural decoder, drops candidates
whose behavior already exists in the 20k protocol pool (or repeats inside the
pool), and writes a standard fixed-multiplier asset that the existing loaders can
consume.

Usage:
    python \
        -m aaae.calibration.preparation.generate_calib_pool \
        --pool-size 4500 --seed 20260929 --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    _catalog_lowonly_candidates,
    _load_decoder,
    _token_text,
    lowonly_config_hash,
)
from aaae.accuracy_predictor.cnn.behavior import behavior_sha256_rows  # noqa: E402
from aaae.multiplier_generator.lut_domains import (  # noqa: E402
    s8s8_exact_table_np,
    u8_lut_to_s8s8_sign_magnitude,
)
from aaae.multiplier_generator.truth_table import UINT16_MAX  # noqa: E402
from aaae.inference_runtime.transformer.s8s8_truth_lut_asset import (  # noqa: E402
    S8S8_INDEX_ENCODING,
    S8S8_LUT_DOMAIN,
    S8S8_TARGET_SEMANTICS,
    S8_QMAX,
    S8_QMIN,
)

DEFAULT_CATALOG = E2E_ROOT / "resources/data/multipliers/D_mult/canonical/meta/multiplier_catalog.csv"
DEFAULT_EXISTING_POPULATION = (
    E2E_ROOT / "datasets/multiplier/lowonly20k_s8s8_sign_magnitude_v1/population.csv"
)
DEFAULT_PROTOCOL_BEHAVIOR = (
    EA2_ROOT / "experiment/probe_budget/artifacts/protocol_v1/behavior_sha256.npy"
)
DEFAULT_CHECKPOINT = (
    EA2_ROOT / "approx_error_model/runs/truth_table/full17_neural_logic/checkpoints/best.pt"
)
DEFAULT_OUTPUT_ROOT = (
    E7_ROOT / "datasets/multiplier/lowonly_calibpool4500_s8s8_sign_magnitude_v1"
)

TABLE_FILENAME = "truth_tables_s8s8_sign_magnitude_int32.npy"
TABLE_SHAPE = (256, 256)


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sample_order(candidates: list[dict], seed: int) -> list[dict]:
    def rank(candidate: dict) -> str:
        return hashlib.sha256(f"{seed}|{candidate['catalog_design_id']}".encode("ascii")).hexdigest()

    return sorted(candidates, key=rank)


def assemble_population(accepted: list[dict], *, asset_id: str, behavior_hashes: list[str]) -> pd.DataFrame:
    records = []
    for index, (candidate, behavior) in enumerate(zip(accepted, behavior_hashes, strict=True)):
        records.append(
            {
                "row_index": index,
                "design_id": f"{asset_id}_design_{index:06d}",
                "runtime_design_id": f"{asset_id}_design_{index:06d}",
                "config_hash": candidate["config_hash"],
                "approx_high_part": False,
                "low_tokens": _token_text(candidate["low_tokens"]),
                "high_tokens": _token_text((0,) * 8),
                "replacement_kind": "generated_calib_pool",
                "catalog_design_id": candidate["catalog_design_id"],
                "catalog_design_index": int(candidate["catalog_design_index"]),
                "behavior_sha256": behavior,
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--existing-population", type=Path, default=DEFAULT_EXISTING_POPULATION)
    parser.add_argument("--protocol-behavior", type=Path, default=DEFAULT_PROTOCOL_BEHAVIOR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--asset-id", default="lowonly_calibpool4500_s8s8_sign_magnitude_v1")
    parser.add_argument("--pool-size", type=int, default=4500)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dedup-report", type=Path, default=None,
        help="defaults to <output-root>/dedup_report.json",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    catalog_path = Path(args.catalog)
    existing = pd.read_csv(args.existing_population, low_memory=False)
    existing_configs = set(existing["config_hash"].astype(str))
    protocol_hashes = set(np.load(args.protocol_behavior, allow_pickle=False).astype(str))

    candidates = [
        candidate
        for candidate in _catalog_lowonly_candidates(catalog_path)
        if candidate["config_hash"] not in existing_configs
    ]
    candidates = sample_order(candidates, args.seed)
    print(
        f"catalog candidates={len(candidates)} excluded_existing={len(existing_configs)} "
        f"protocol_hashes={len(protocol_hashes)}",
        flush=True,
    )

    device = torch.device(args.device)
    model, exact_batched, decoder_slots = _load_decoder(Path(args.checkpoint), device=device)
    exact = s8s8_exact_table_np()

    accepted: list[dict] = []
    tables_parts: list[np.ndarray] = []
    accepted_hashes: list[str] = []
    seen: set[str] = set()
    cursor = 0
    duplicates_protocol = 0
    duplicates_internal = 0
    started = time.time()
    try:
        while len(accepted) < args.pool_size and cursor < len(candidates):
            batch = candidates[cursor : cursor + args.batch_rows]
            cursor += args.batch_rows
            if decoder_slots == 17:
                slot_rows = [(*candidate["low_tokens"], *(0,) * 8) for candidate in batch]
            else:
                slot_rows = [candidate["low_tokens"] for candidate in batch]
            slot_tokens = torch.as_tensor(np.asarray(slot_rows, dtype=np.int64), dtype=torch.long, device=device)
            with torch.no_grad():
                delta = model(slot_tokens)["delta"]
                u8_truth = torch.clamp(torch.round(delta + exact_batched), 0.0, float(UINT16_MAX))
            u8_tables = u8_truth.detach().cpu().numpy().astype(np.uint16)
            if np.any(u8_tables[:, 0, :]) or np.any(u8_tables[:, :, 0]):
                raise RuntimeError("decoded U8 LUT violates the zero-operand contract")
            signed = u8_lut_to_s8s8_sign_magnitude(u8_tables).astype(np.int32)
            residual = signed - exact[None, :, :]
            hashes = behavior_sha256_rows(residual)
            for offset, behavior in enumerate(hashes):
                if behavior in protocol_hashes:
                    duplicates_protocol += 1
                    continue
                if behavior in seen:
                    duplicates_internal += 1
                    continue
                seen.add(behavior)
                accepted.append(batch[offset])
                tables_parts.append(signed[offset])
                accepted_hashes.append(behavior)
                if len(accepted) >= args.pool_size:
                    break
            print(
                f"accepted={len(accepted)}/{args.pool_size} scanned={cursor} "
                f"dup_protocol={duplicates_protocol} dup_internal={duplicates_internal} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(accepted) < args.pool_size:
        raise RuntimeError(f"only {len(accepted)} unique behaviors found; increase candidates or pool")

    tables = np.stack(tables_parts, axis=0).astype(np.int32)
    if tables.shape != (len(accepted), *TABLE_SHAPE):
        raise RuntimeError(f"assembled tables have shape {tables.shape}")
    population = assemble_population(accepted, asset_id=args.asset_id, behavior_hashes=accepted_hashes)

    population_path = output_root / "population.csv"
    table_path = output_root / TABLE_FILENAME
    runtime_ids_path = output_root / "runtime_design_ids.npy"
    completed_path = output_root / "completed_rows.npy"
    behavior_path = output_root / "behavior_sha256.npy"
    population.to_csv(population_path, index=False)
    np.save(table_path, tables)
    np.save(runtime_ids_path, population["runtime_design_id"].astype(str).to_numpy(dtype="<U128"))
    np.save(completed_path, np.ones(len(accepted), dtype=np.bool_))
    np.save(behavior_path, np.asarray(accepted_hashes, dtype="S64"))

    now = time.time()
    manifest = {
        "schema_version": 1,
        "asset_id": args.asset_id,
        "asset_kind": "fixed_multiplier_lut",
        "target_lut_domain": S8S8_LUT_DOMAIN,
        "target_semantics": S8S8_TARGET_SEMANTICS,
        "index_encoding": S8S8_INDEX_ENCODING,
        "table_dtype": "int32",
        "signed_code_range": [S8_QMIN, S8_QMAX],
        "row_count": int(len(accepted)),
        "table_shape": [int(len(accepted)), *TABLE_SHAPE],
        "complete": True,
        "native_signed_rtl": False,
        "population_filename": population_path.name,
        "table_filename": table_path.name,
        "runtime_ids_filename": runtime_ids_path.name,
        "completed_filename": completed_path.name,
        "row_limit": None,
        "full_source_population_row_count": int(len(accepted)),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha1": sha1_file(Path(args.checkpoint)),
        "decoder_num_slots": int(decoder_slots),
        "created_at_unix": now,
        "completed_at_unix": time.time(),
        "files": {
            population_path.name: sha1_file(population_path),
            table_path.name: sha1_file(table_path),
            runtime_ids_path.name: sha1_file(runtime_ids_path),
            completed_path.name: sha1_file(completed_path),
            behavior_path.name: sha1_file(behavior_path),
        },
        "selection": {
            "kind": "unused_lowonly_catalog_sample_behavior_deduped",
            "catalog_path": str(catalog_path.resolve()),
            "catalog_sha1": sha1_file(catalog_path),
            "seed": int(args.seed),
            "protocol_behavior_path": str(Path(args.protocol_behavior).resolve()),
            "protocol_behavior_sha256": sha256_file(Path(args.protocol_behavior)),
            "existing_population_path": str(Path(args.existing_population).resolve()),
            "existing_population_sha256": sha256_file(Path(args.existing_population)),
            "duplicates_protocol": int(duplicates_protocol),
            "duplicates_internal": int(duplicates_internal),
            "unique_behaviors_in_pool": int(len(seen)),
            "scanned_candidates": int(cursor),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "progress.json", manifest)

    dedup_report = {
        "schema_version": "e07_calib_pool_dedup_v1",
        "asset_id": args.asset_id,
        "pool_rows": int(len(accepted)),
        "unique_behaviors": int(len(seen)),
        "duplicates_vs_protocol_v1": int(duplicates_protocol),
        "duplicates_internal": int(duplicates_internal),
        "protocol_behavior_path": str(Path(args.protocol_behavior).resolve()),
        "protocol_behavior_unique_count": int(len(protocol_hashes)),
        "existing_population_config_count": int(len(existing_configs)),
        "scanned_candidates": int(cursor),
        "seed": int(args.seed),
        "behavior_sha256_file": str(behavior_path),
        "behavior_sha256_file_sha1": sha1_file(behavior_path),
    }
    report_path = Path(args.dedup_report) if args.dedup_report else output_root / "dedup_report.json"
    write_json(report_path, dedup_report)
    print(f"wrote asset {output_root} rows={len(accepted)} report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
