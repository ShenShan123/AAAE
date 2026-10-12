from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from aaae.paths import OUTPUT_ROOT


from aaae.multiplier_generator.catalog.build_fixed_topology_views import write_shared_topology  # noqa: E402
from aaae.multiplier_generator.catalog.pack_truth_tables import TruthShardWriter  # noqa: E402
from aaae.multiplier_generator.catalog.schema import (  # noqa: E402
    APPROX_WRAPPER_NODE_KIND_ID,
    DATASET_NAME,
    EXACT_42_FAMILY_ID,
    EXACT_42_NODE_KIND_ID,
    FULL24_COUNT,
    HIGH_COMPRESSOR_COUNT,
    HIGH_DISABLED_SENTINEL,
    LOW_COMPRESSOR_COUNT,
    M_MAX_U8_MUL,
    N_BITS,
    SPLIT_TO_CODE,
    SplitConfig,
    TOPOLOGY_KEY,
    TRUTH_TABLE_SHAPE,
    VAR17_COUNT,
)


from aaae.paths import EA2_ROOT, LEGACY_WORKFLOW_ROOT

PROJECT_ROOT = LEGACY_WORKFLOW_ROOT
APPROX_LIB_DIR = EA2_ROOT / "approx_lib"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "multiplier" / DATASET_NAME
DEFAULT_TEMP_DIR = OUTPUT_ROOT / "multiplier" / "temp" / "d_mult_build"
DEFAULT_LOG_DIR = OUTPUT_ROOT / "multiplier" / "logs"
COMPRESSOR_BASE = 8
COMPRESSOR_CHOICES = tuple(range(1, COMPRESSOR_BASE + 1))
TOTAL_SPEC_DIGITS = LOW_COMPRESSOR_COUNT + HIGH_COMPRESSOR_COUNT
LOW_ONLY_SPEC_DIGITS = LOW_COMPRESSOR_COUNT

EXACT_TABLE = np.arange(256, dtype=np.int32)[:, None] * np.arange(256, dtype=np.int32)[None, :]
EXACT_FLOAT = EXACT_TABLE.astype(np.float64)
NONZERO_MASK = EXACT_TABLE != 0
ZERO_OPERAND_MASK = (np.arange(256)[:, None] == 0) | (np.arange(256)[None, :] == 0)


TB_TEMPLATE = r"""`timescale 1ns/1ps
module tb;
  localparam int N = {n_bits};

  logic [N-1:0] a, b;
  wire  [2*N-1:0] p;

  Appro4_2_{n_bits}bit dut(.N1(a), .N2(b), .P_APPROX(p));

  string out_path;
  integer f;

  initial begin
    if (!$value$plusargs("OUT=%s", out_path)) out_path = "truth.bin";
    f = $fopen(out_path, "wb");
    if (f == 0) begin
      $display("ERROR: failed to open OUT=%s", out_path);
      $finish;
    end

    for (int i = 0; i < (1<<N); i++) begin
      for (int j = 0; j < (1<<N); j++) begin
        a = i;
        b = j;
        #1;
        $fwrite(f, "%c%c", p[7:0], p[15:8]);
      end
    end

    $fclose(f);
    $finish;
  end
endmodule
"""


@dataclass(frozen=True)
class DesignSpec:
    index: int
    design_id: str
    split: str
    approx_high_part: bool
    compressor_types_low: tuple[int, ...]
    compressor_types_high: tuple[int, ...]
    spec_hash: str


@dataclass(frozen=True)
class WorkerConfig:
    temp_root: str
    tb_path: str
    iverilog: str
    vvp: str


def _ensure_approx_lib_import() -> None:
    approx_lib_path = str(APPROX_LIB_DIR.resolve())
    if approx_lib_path not in sys.path:
        pass  # Local package imports are resolved by the entry point.


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _spec_hash(low: tuple[int, ...], high: tuple[int, ...], *, approx_high_part: bool) -> str:
    payload = {
        "kind": "approx4_2",
        "n_bits": N_BITS,
        "approx_high_part": bool(approx_high_part),
        "compressor_types_low": list(low),
        "compressor_types_high": list(high),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _decode_sample(sample_value: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    digits = [0] * TOTAL_SPEC_DIGITS
    value = int(sample_value)
    for pos in range(TOTAL_SPEC_DIGITS - 1, -1, -1):
        value, rem = divmod(value, COMPRESSOR_BASE)
        digits[pos] = int(rem) + 1
    low = tuple(digits[:LOW_COMPRESSOR_COUNT])
    high = tuple(digits[LOW_COMPRESSOR_COUNT:])
    return low, high


def _decode_low_only_sample(sample_value: int) -> tuple[int, ...]:
    digits = [0] * LOW_ONLY_SPEC_DIGITS
    value = int(sample_value)
    for pos in range(LOW_ONLY_SPEC_DIGITS - 1, -1, -1):
        value, rem = divmod(value, COMPRESSOR_BASE)
        digits[pos] = int(rem) + 1
    return tuple(digits)


def _build_specs(count: int, split_cfg: SplitConfig, seed: int) -> list[DesignSpec]:
    if count % 2 != 0:
        raise ValueError("Count must be even so high-enabled and low-only designs can stay balanced")
    if split_cfg.train % 2 != 0 or split_cfg.test % 2 != 0 or split_cfg.val % 2 != 0:
        raise ValueError("Train/test/val counts must all be even for 1:1 high/low balance")
    sample_space_high = COMPRESSOR_BASE ** TOTAL_SPEC_DIGITS
    sample_space_low = COMPRESSOR_BASE ** LOW_ONLY_SPEC_DIGITS
    high_total = count // 2
    low_total = count // 2
    if high_total > sample_space_high:
        raise ValueError(f"Requested {high_total} unique high-enabled specs, but only {sample_space_high} combinations exist")
    if low_total > sample_space_low:
        raise ValueError(f"Requested {low_total} unique low-only specs, but only {sample_space_low} combinations exist")

    rng = random.Random(seed)
    sampled_high = rng.sample(range(sample_space_high), high_total)
    sampled_low = rng.sample(range(sample_space_low), low_total)
    split_sequence = (
        [("train", True), ("train", False)] * (split_cfg.train // 2)
        + [("test", True), ("test", False)] * (split_cfg.test // 2)
        + [("val", True), ("val", False)] * (split_cfg.val // 2)
    )
    if len(split_sequence) != count:
        raise RuntimeError("Split sequencing bug: balanced split sequence does not match count")

    specs: list[DesignSpec] = []
    high_cursor = 0
    low_cursor = 0
    for index, (split, approx_high_part) in enumerate(split_sequence):
        if approx_high_part:
            low, high = _decode_sample(sampled_high[high_cursor])
            high_cursor += 1
        else:
            low = _decode_low_only_sample(sampled_low[low_cursor])
            high = tuple([HIGH_DISABLED_SENTINEL] * HIGH_COMPRESSOR_COUNT)
            low_cursor += 1
        specs.append(
            DesignSpec(
                index=index,
                design_id=f"Appro4_2_8bit_{index:06d}",
                split=split,
                approx_high_part=approx_high_part,
                compressor_types_low=low,
                compressor_types_high=high,
                spec_hash=_spec_hash(low, high, approx_high_part=approx_high_part),
            )
        )
    if high_cursor != high_total or low_cursor != low_total:
        raise RuntimeError("Spec generation bug: did not consume all sampled high/low specs")
    return specs


def _write_binary_testbench(temp_root: Path) -> Path:
    tb_path = temp_root / "tb_Appro4_2_8bit_binary.sv"
    _write_text(tb_path, TB_TEMPLATE.format(n_bits=N_BITS) + "\n")
    return tb_path


def _compute_metrics(table: np.ndarray, changed_rows: int) -> dict[str, Any]:
    errors = table.astype(np.int32) - EXACT_TABLE
    err_f = errors.astype(np.float64)
    ed = np.abs(errors).astype(np.float64)

    re = np.zeros(TRUTH_TABLE_SHAPE, dtype=np.float64)
    red = np.zeros(TRUTH_TABLE_SHAPE, dtype=np.float64)
    re[NONZERO_MASK] = err_f[NONZERO_MASK] / EXACT_FLOAT[NONZERO_MASK]
    red[NONZERO_MASK] = ed[NONZERO_MASK] / EXACT_FLOAT[NONZERO_MASK]

    total = int(table.size)
    muE = float(err_f.mean())
    muED = float(ed.mean())
    muRE = float(re.mean())
    muRED = float(red.mean())

    sigmaE = float((err_f * err_f).mean() - (muE * muE))
    sigmaED = float((ed * ed).mean() - (muED * muED))
    sigmaRE = float((re * re).mean() - (muRE * muRE))
    sigmaRED = float((red * red).mean() - (muRED * muRED))

    rmsE = float(math.sqrt(float((err_f * err_f).mean())))
    rmsRE = float(math.sqrt(float((re * re).mean())))
    er = float(np.count_nonzero(errors) / total)

    return {
        "vectors": total,
        "muE": muE,
        "sigmaE": sigmaE,
        "muED": muED,
        "NMED": float(muED / M_MAX_U8_MUL),
        "sigmaED": sigmaED,
        "muRE": muRE,
        "sigmaRE": sigmaRE,
        "muRED": muRED,
        "sigmaRED": sigmaRED,
        "rmsE": rmsE,
        "rmsRE": rmsRE,
        "ER": er,
        "changed_rows": int(changed_rows),
        "changed_frac": float(changed_rows / total),
    }


def _simulate_one(spec: DesignSpec, worker_cfg: WorkerConfig) -> dict[str, Any]:
    _ensure_approx_lib_import()
    from aaae.multiplier_generator.nbit_approx import MultiplierGeneratorArbitrary  # type: ignore

    temp_root = Path(worker_cfg.temp_root)
    temp_dir = temp_root / f"worker_{os.getpid()}" / spec.design_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    rtl_path = temp_dir / f"{spec.design_id}.v"
    build_path = temp_dir / f"{spec.design_id}.out"
    truth_path = temp_dir / f"{spec.design_id}.bin"

    try:
        gen = MultiplierGeneratorArbitrary(
            n_bits=N_BITS,
            compressor_types=list(spec.compressor_types_low),
            high_compressor_types=(list(spec.compressor_types_high) if spec.approx_high_part else []),
            approx_high_part=bool(spec.approx_high_part),
        )
        rtl = gen.generate_verilog_code()
        rtl_path.write_text(rtl, encoding="utf-8")

        compile_proc = subprocess.run(
            [
                worker_cfg.iverilog,
                "-g2012",
                "-o",
                str(build_path),
                str(rtl_path),
                str(Path(worker_cfg.tb_path)),
            ],
            capture_output=True,
            text=True,
        )
        if compile_proc.returncode != 0:
            raise RuntimeError(
                f"iverilog failed for {spec.design_id}: "
                f"{compile_proc.stderr.strip() or compile_proc.stdout.strip()}"
            )

        run_proc = subprocess.run(
            [worker_cfg.vvp, str(build_path), f"+OUT={truth_path}"],
            capture_output=True,
            text=True,
        )
        if run_proc.returncode != 0:
            raise RuntimeError(
                f"vvp failed for {spec.design_id}: "
                f"{run_proc.stderr.strip() or run_proc.stdout.strip()}"
            )

        raw = np.fromfile(truth_path, dtype="<u2")
        if raw.size != TRUTH_TABLE_SHAPE[0] * TRUTH_TABLE_SHAPE[1]:
            raise RuntimeError(
                f"Unexpected truth size for {spec.design_id}: "
                f"expected {TRUTH_TABLE_SHAPE[0] * TRUTH_TABLE_SHAPE[1]} entries, got {raw.size}"
            )
        table = raw.reshape(TRUTH_TABLE_SHAPE)
        changed_rows = int(np.count_nonzero(table[ZERO_OPERAND_MASK]))
        if changed_rows:
            table = table.copy()
            table[ZERO_OPERAND_MASK] = 0
        metrics = _compute_metrics(table, changed_rows)
        return {
            "index": spec.index,
            "design_id": spec.design_id,
            "split": spec.split,
            "approx_high_part": bool(spec.approx_high_part),
            "spec_hash": spec.spec_hash,
            "compressor_types_low": np.asarray(spec.compressor_types_low, dtype=np.uint8),
            "compressor_types_high": np.asarray(spec.compressor_types_high, dtype=np.uint8),
            "truth_table": table,
            "metrics": metrics,
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _simulate_batch(batch: list[DesignSpec], worker_cfg: WorkerConfig) -> list[dict[str, Any]]:
    return [_simulate_one(spec, worker_cfg) for spec in batch]


def _batched(items: list[DesignSpec], batch_size: int) -> Iterable[list[DesignSpec]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _prepare_output_dirs(output_dir: Path, *, overwrite: bool) -> dict[str, Path]:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "shared": output_dir / "shared",
        "meta": output_dir / "meta",
        "arrays": output_dir / "arrays",
        "splits": output_dir / "splits",
        "results": output_dir / "results",
        "truth_shards": output_dir / "truth_shards",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _clean_temp_dir(temp_dir: Path) -> None:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)


def _write_split_files(split_dir: Path, design_ids: list[str], split_codes: np.ndarray) -> None:
    for split_name, split_code in SPLIT_TO_CODE.items():
        indices = np.flatnonzero(split_codes == split_code).astype(np.int32)
        np.save(split_dir / f"{split_name}_indices.npy", indices)
        ids = [design_ids[int(idx)] for idx in indices]
        (split_dir / f"{split_name}_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "id",
        "kind",
        "n_bits",
        "top",
        "vectors",
        "muE",
        "sigmaE",
        "muED",
        "NMED",
        "sigmaED",
        "muRE",
        "sigmaRE",
        "muRED",
        "sigmaRED",
        "rmsE",
        "rmsRE",
        "ER",
        "changed_rows",
        "changed_frac",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _resolve_binary(name: str) -> str:
    resolved = shutil.which(name)
    return resolved or name


def build_dataset(
    *,
    output_dir: Path,
    temp_dir: Path,
    count: int,
    split_cfg: SplitConfig,
    seed: int,
    shard_size: int,
    workers: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    if count != split_cfg.total:
        raise ValueError(f"Count {count} does not match split total {split_cfg.total}")

    dirs = _prepare_output_dirs(output_dir, overwrite=overwrite)
    _clean_temp_dir(temp_dir)

    shared = write_shared_topology(dirs["shared"])
    specs = _build_specs(count=count, split_cfg=split_cfg, seed=seed)
    design_ids = [spec.design_id for spec in specs]

    split_codes = np.empty((count,), dtype=np.uint8)
    approx_high_part_flags = np.empty((count,), dtype=np.uint8)
    compressor_low = np.empty((count, LOW_COMPRESSOR_COUNT), dtype=np.uint8)
    compressor_high = np.empty((count, HIGH_COMPRESSOR_COUNT), dtype=np.uint8)
    compressor_var17 = np.empty((count, VAR17_COUNT), dtype=np.uint8)
    node_kind_full24 = np.repeat(shared["node_kind_template"][None, :], count, axis=0)
    compressor_family_full24 = np.repeat(shared["default_family_template"][None, :], count, axis=0)

    low_var_indices = np.asarray(shared["low_var_indices"], dtype=np.int64)
    high_var_indices = np.asarray(shared["high_var_indices"], dtype=np.int64)
    truth_writer = TruthShardWriter(dirs["truth_shards"], shard_size=shard_size)

    metrics_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    tb_path = _write_binary_testbench(temp_dir)
    worker_cfg = WorkerConfig(
        temp_root=str(temp_dir),
        tb_path=str(tb_path),
        iverilog=_resolve_binary("iverilog"),
        vvp=_resolve_binary("vvp"),
    )

    start = time.time()
    processed = 0
    batches = list(_batched(specs, batch_size))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for batch_result in executor.map(_simulate_batch, batches, [worker_cfg] * len(batches), chunksize=1):
            for result in batch_result:
                idx = int(result["index"])
                split = str(result["split"])
                split_codes[idx] = SPLIT_TO_CODE[split]
                approx_high_part = bool(result["approx_high_part"])
                approx_high_part_flags[idx] = int(approx_high_part)
                low = np.asarray(result["compressor_types_low"], dtype=np.uint8)
                high = np.asarray(result["compressor_types_high"], dtype=np.uint8)
                var17 = np.concatenate([low, high], dtype=np.uint8)

                compressor_low[idx] = low
                compressor_high[idx] = high
                compressor_var17[idx] = var17
                compressor_family_full24[idx, low_var_indices] = low
                if approx_high_part:
                    compressor_family_full24[idx, high_var_indices] = high
                    node_kind_full24[idx, high_var_indices] = APPROX_WRAPPER_NODE_KIND_ID
                else:
                    compressor_family_full24[idx, high_var_indices] = EXACT_42_FAMILY_ID
                    node_kind_full24[idx, high_var_indices] = EXACT_42_NODE_KIND_ID

                table = np.asarray(result["truth_table"], dtype=np.uint16)
                location = truth_writer.add(table)
                metrics = dict(result["metrics"])
                metrics_row = {
                    "id": result["design_id"],
                    "kind": "approx4_2",
                    "n_bits": N_BITS,
                    "top": "Appro4_2_8bit",
                    **metrics,
                }
                metrics_rows.append(metrics_row)
                manifest_rows.append(
                    {
                        "id": result["design_id"],
                        "kind": "approx4_2",
                        "n_bits": N_BITS,
                        "top": "Appro4_2_8bit",
                        "approx_high_part": approx_high_part,
                        "required": {"low": LOW_COMPRESSOR_COUNT, "high": HIGH_COMPRESSOR_COUNT},
                        "compressor_types_low": low.tolist(),
                        "compressor_types_high": high.tolist(),
                        "token_view": {
                            "full_token_count": FULL24_COUNT,
                            "variable_token_count": VAR17_COUNT,
                        },
                        "spec_hash": str(result["spec_hash"]),
                        "topology_key": TOPOLOGY_KEY,
                        "split": split,
                        "split_code": int(split_codes[idx]),
                        "truth_table_bin": location.path,
                        "truth_table_shard": int(location.shard_id),
                        "truth_table_row": int(location.row_in_shard),
                        "metrics_row_index": idx,
                    }
                )
                processed += 1
                if processed % max(batch_size * 4, 128) == 0 or processed == count:
                    elapsed = max(time.time() - start, 1e-9)
                    rate = processed / elapsed
                    print(
                        f"[D_mult] processed={processed}/{count} "
                        f"rate={rate:.2f} design/s elapsed={elapsed/60.0:.1f} min",
                        flush=True,
                    )

    truth_writer.finish()
    _write_metrics_csv(dirs["results"] / "metrics_zero_operand_fixed.csv", metrics_rows)
    _write_manifest(dirs["meta"] / "designs.jsonl", manifest_rows)

    np.save(dirs["arrays"] / "compressor_type_vector_low.npy", compressor_low)
    np.save(dirs["arrays"] / "compressor_type_vector_high.npy", compressor_high)
    np.save(dirs["arrays"] / "compressor_type_vector_var17.npy", compressor_var17)
    np.save(dirs["arrays"] / "node_kind_vector_full24.npy", node_kind_full24)
    np.save(dirs["arrays"] / "compressor_family_vector_full24.npy", compressor_family_full24)
    np.save(dirs["arrays"] / "split_codes.npy", split_codes)
    np.save(dirs["arrays"] / "approx_high_part_flags.npy", approx_high_part_flags)

    _write_split_files(dirs["splits"], design_ids, split_codes)

    build_config = {
        "dataset_name": DATASET_NAME,
        "topology_key": TOPOLOGY_KEY,
        "count": count,
        "split": {
            "train": split_cfg.train,
            "test": split_cfg.test,
            "val": split_cfg.val,
        },
        "approx_high_part_balance": {
            "train": {"enabled": split_cfg.train // 2, "disabled": split_cfg.train // 2},
            "test": {"enabled": split_cfg.test // 2, "disabled": split_cfg.test // 2},
            "val": {"enabled": split_cfg.val // 2, "disabled": split_cfg.val // 2},
            "total": {"enabled": count // 2, "disabled": count // 2},
        },
        "seed": seed,
        "unique_sampling": {
            "base": COMPRESSOR_BASE,
            "digits_high_enabled": TOTAL_SPEC_DIGITS,
            "digits_low_only": LOW_ONLY_SPEC_DIGITS,
            "choices": list(COMPRESSOR_CHOICES),
        },
        "truth_storage": {
            "dtype": "uint16",
            "shape": list(TRUTH_TABLE_SHAPE),
            "shard_size": shard_size,
        },
        "build_runtime": {
            "workers": workers,
            "batch_size": batch_size,
            "iverilog": worker_cfg.iverilog,
            "vvp": worker_cfg.vvp,
        },
        "paths": {
            "shared": "shared",
            "meta": "meta/designs.jsonl",
            "metrics": "results/metrics_zero_operand_fixed.csv",
            "truth_shards": "truth_shards",
        },
    }
    _write_json(dirs["meta"] / "build_config.json", build_config)

    final_elapsed = time.time() - start
    _write_json(
        dirs["meta"] / "summary.json",
        {
            "count": count,
            "approx_high_part_enabled": int(np.count_nonzero(approx_high_part_flags)),
            "approx_high_part_disabled": int(count - np.count_nonzero(approx_high_part_flags)),
            "elapsed_seconds": final_elapsed,
            "elapsed_minutes": final_elapsed / 60.0,
            "output_dir": str(output_dir),
        },
    )
    shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical D_mult dataset in the current project.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--count", type=int, default=80_000)
    parser.add_argument("--train-count", type=int, default=72_000)
    parser.add_argument("--test-count", type=int, default=4_000)
    parser.add_argument("--val-count", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    split_cfg = SplitConfig(
        train=int(args.train_count),
        test=int(args.test_count),
        val=int(args.val_count),
    )
    build_dataset(
        output_dir=args.output_dir.resolve(),
        temp_dir=args.temp_dir.resolve(),
        count=int(args.count),
        split_cfg=split_cfg,
        seed=int(args.seed),
        shard_size=int(args.shard_size),
        workers=max(1, int(args.workers)),
        batch_size=max(1, int(args.batch_size)),
        overwrite=bool(args.overwrite),
    )
    print(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
