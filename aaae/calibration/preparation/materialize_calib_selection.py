"""E7: materialize the selected 3000 calibration candidates as a fixed-LUT asset.

Reads the frozen selection produced by ``scan_and_select_calib.py``, re-decodes
the selected LUTs (deterministic decoder), verifies behavior hashes against the
selection record, and writes a standard fixed-multiplier asset plus the frozen
feature matrices used by the calibration step.

Usage:
    python \
        -m aaae.calibration.preparation.materialize_calib_selection \
        --selection-dir experiment/e07_conformal_screening/artifacts/calib3000 \
        --output-root end_to_end_prediction/datasets/multiplier/lowonly_calib3000_s8s8_sign_magnitude_v1 \
        --device cuda:0
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
    _load_decoder,
    _token_text,
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

DEFAULT_CHECKPOINT = EA2_ROOT / "approx_error_model/runs/truth_table/full17_neural_logic/checkpoints/best.pt"
TABLE_FILENAME = "truth_tables_s8s8_sign_magnitude_int32.npy"
ASSET_ID = "lowonly_calib3000_s8s8_sign_magnitude_v1"


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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--asset-id", default=ASSET_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selection_path = Path(args.selection_dir) / "selection.csv"
    summary_path = Path(args.selection_dir) / "scan_summary.json"
    selection = pd.read_csv(selection_path, low_memory=False)
    scan_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(selection) != 3000:
        raise SystemExit(f"selection has {len(selection)} rows, expected 3000")

    output_root = Path(args.output_root)
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    tokens = np.stack(
        [np.asarray(json.loads(value), dtype=np.int64) for value in selection["low_tokens"]],
        axis=0,
    )
    expected_behaviors = selection["behavior_sha256"].astype(str).tolist()

    device = torch.device(args.device)
    decoder, exact_batched, decoder_slots = _load_decoder(Path(args.checkpoint), device=device)
    exact = s8s8_exact_table_np()
    tables = np.empty((len(tokens), 256, 256), dtype=np.int32)
    behaviors: list[str] = []
    started = time.time()
    try:
        for start in range(0, len(tokens), int(args.batch_rows)):
            stop = min(start + int(args.batch_rows), len(tokens))
            batch_tokens = tokens[start:stop]
            if decoder_slots == 17:
                slot_rows = np.concatenate([batch_tokens, np.zeros((len(batch_tokens), 8), dtype=np.int64)], axis=1)
            else:
                slot_rows = batch_tokens
            with torch.no_grad():
                delta = decoder(torch.as_tensor(slot_rows, dtype=torch.long, device=device))["delta"]
                u8_truth = torch.clamp(torch.round(delta + exact_batched), 0.0, float(UINT16_MAX))
            u8_tables = u8_truth.detach().cpu().numpy().astype(np.uint16)
            if np.any(u8_tables[:, 0, :]) or np.any(u8_tables[:, :, 0]):
                raise RuntimeError("decoded U8 LUT violates the zero-operand contract")
            signed = u8_lut_to_s8s8_sign_magnitude(u8_tables).astype(np.int32)
            residual = signed - exact[None, :, :]
            hashes = behavior_sha256_rows(residual)
            tables[start:stop] = signed
            behaviors.extend(hashes)
            print(f"decoded {stop}/{len(tokens)} elapsed={time.time() - started:.1f}s", flush=True)
    finally:
        del decoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mismatches = [index for index, (actual, expected) in enumerate(zip(behaviors, expected_behaviors)) if actual != expected]
    if mismatches:
        raise RuntimeError(f"{len(mismatches)} behavior hashes differ from the selection record")

    population = pd.DataFrame(
        {
            "row_index": np.arange(len(tokens), dtype=np.int64),
            "design_id": [f"{args.asset_id}_design_{index:06d}" for index in range(len(tokens))],
            "runtime_design_id": [f"{args.asset_id}_design_{index:06d}" for index in range(len(tokens))],
            "config_hash": selection["config_hash"].astype(str),
            "approx_high_part": False,
            "low_tokens": [_token_text(tuple(int(v) for v in row)) for row in tokens.tolist()],
            "high_tokens": [_token_text((0,) * 8)] * len(tokens),
            "replacement_kind": "generated_calib_selection",
            "behavior_sha256": behaviors,
            "sampling_group": selection["group"].astype(str),
            "proxy_clipped": selection["proxy_clipped"].astype(float),
            "coverage_bin": selection["bin"].astype(int),
        }
    )

    population_path = output_root / "population.csv"
    table_path = output_root / TABLE_FILENAME
    runtime_ids_path = output_root / "runtime_design_ids.npy"
    completed_path = output_root / "completed_rows.npy"
    behavior_path = output_root / "behavior_sha256.npy"
    population.to_csv(population_path, index=False)
    np.save(table_path, tables)
    np.save(runtime_ids_path, population["runtime_design_id"].astype(str).to_numpy(dtype="<U128"))
    np.save(completed_path, np.ones(len(tokens), dtype=np.bool_))
    np.save(behavior_path, np.asarray(behaviors, dtype="S64"))

    manifest = {
        "schema_version": 1,
        "asset_id": args.asset_id,
        "asset_kind": "fixed_multiplier_lut",
        "target_lut_domain": S8S8_LUT_DOMAIN,
        "target_semantics": S8S8_TARGET_SEMANTICS,
        "index_encoding": S8S8_INDEX_ENCODING,
        "table_dtype": "int32",
        "signed_code_range": [S8_QMIN, S8_QMAX],
        "row_count": int(len(tokens)),
        "table_shape": [int(len(tokens)), 256, 256],
        "complete": True,
        "native_signed_rtl": False,
        "population_filename": population_path.name,
        "table_filename": table_path.name,
        "runtime_ids_filename": runtime_ids_path.name,
        "completed_filename": completed_path.name,
        "row_limit": None,
        "full_source_population_row_count": int(len(tokens)),
        "checkpoint_path": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha1": sha1_file(Path(args.checkpoint)),
        "decoder_num_slots": int(decoder_slots),
        "created_at_unix": time.time(),
        "completed_at_unix": time.time(),
        "files": {
            population_path.name: sha1_file(population_path),
            table_path.name: sha1_file(table_path),
            runtime_ids_path.name: sha1_file(runtime_ids_path),
            completed_path.name: sha1_file(completed_path),
            behavior_path.name: sha1_file(behavior_path),
        },
        "selection": {
            "kind": "spectrum_scan_swin_proxy_quota_selection",
            "selection_csv": str(selection_path.resolve()),
            "selection_csv_sha256": sha256_file(selection_path),
            "scan_summary": str(summary_path.resolve()),
            "scan_summary_sha256": sha256_file(summary_path),
            "selection_rows": int(len(tokens)),
            "selected_by_bin": scan_summary.get("selected_by_bin"),
            "proxy_checkpoint": scan_summary.get("proxy_checkpoint"),
            "protocol_behavior": scan_summary.get("protocol_behavior"),
            "selection_seed": scan_summary.get("selection_seed"),
        },
    }
    write_json(output_root / "manifest.json", manifest)
    write_json(output_root / "progress.json", manifest)

    features_dir = E7_ROOT / "artifacts/calib3000/features"
    features_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(args.selection_dir) / "selected_s.npy", features_dir / "s_multi.npy")
    shutil.copy2(Path(args.selection_dir) / "selected_c.npy", features_dir / "c_true_swin.npy")
    shutil.copy2(Path(args.selection_dir) / "selected_proxy.npy", features_dir / "proxy_swin_A04.npy")
    write_json(
        features_dir / "features_manifest.json",
        {
            "schema_version": "e07_calib3000_features_v1",
            "s_multi": {"path": "s_multi.npy", "sha256": sha256_file(features_dir / "s_multi.npy"), "shape": [3000, 3103]},
            "c_true_swin": {"path": "c_true_swin.npy", "sha256": sha256_file(features_dir / "c_true_swin.npy"), "shape": [3000, 1034]},
            "proxy_swin_A04": {"path": "proxy_swin_A04.npy", "sha256": sha256_file(features_dir / "proxy_swin_A04.npy")},
            "c_feature_names": scan_summary.get("c_feature_names"),
        },
    )
    print(f"wrote asset {output_root} rows={len(tokens)} behaviors_verified={len(behaviors)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
