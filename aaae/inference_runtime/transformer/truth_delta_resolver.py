from __future__ import annotations

import ast
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from aaae.multiplier_generator.catalog.build_multiplier_dataset import (
    DesignSpec,
    WorkerConfig,
    _simulate_batch,
    _spec_hash,
    _write_binary_testbench,
)

from .design_records import MultiplierRecord, load_all_multiplier_records, load_truth_table


def _parse_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _parse_token_list(value: object, *, expected_len: int) -> tuple[int, ...]:
    if isinstance(value, (list, tuple, np.ndarray)):
        tokens = [int(item) for item in value]
    else:
        text = str(value).strip()
        if " " in text and not text.startswith("["):
            tokens = [int(item) for item in text.split()]
        else:
            tokens = [int(item) for item in ast.literal_eval(text)]
    if len(tokens) != int(expected_len):
        raise ValueError(f"expected {expected_len} tokens, got {len(tokens)} from {value!r}")
    return tuple(tokens)


def _delta_from_truth_table(truth_table: np.ndarray) -> np.ndarray:
    exact = np.arange(256, dtype=np.int32)[:, None] * np.arange(256, dtype=np.int32)[None, :]
    truth = np.asarray(truth_table, dtype=np.int32)
    if truth.shape != (256, 256):
        raise ValueError(f"truth table has shape {truth.shape}, expected (256, 256)")
    return np.ascontiguousarray(truth - exact, dtype=np.int32)


def _canonical_truth_deltas_by_id(records: list[MultiplierRecord]) -> dict[str, np.ndarray]:
    return {
        str(record.design_id): _delta_from_truth_table(load_truth_table(record))
        for record in records
    }


def resolve_truth_deltas_for_population_rows(
    rows: pd.DataFrame,
    *,
    temp_dir: Path,
    simulate_workers: int,
    simulate_batch_size: int,
    runtime_id_column: str = "runtime_design_id",
) -> dict[str, np.ndarray]:
    """Resolve uniform-LUT deltas for canonical and generated multiplier rows.

    Canonical DMult IDs are read from their frozen truth-table shards. Generated
    ``GenBin`` IDs are not present in that catalog, so their immutable token
    configuration is re-simulated with the same DMult Verilog flow used when
    the retained Swin population was built. The caller owns ``temp_dir``.
    """

    if rows.empty:
        return {}
    required = {runtime_id_column}
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"population rows are missing {missing}")

    population = rows.copy().reset_index(drop=True)
    runtime_ids = population[runtime_id_column].astype(str).tolist()
    if len(runtime_ids) != len(set(runtime_ids)):
        raise ValueError("runtime design IDs must be unique within a truth-resolution request")

    canonical_by_id = {record.design_id: record for record in load_all_multiplier_records()}
    canonical_records = [canonical_by_id[design_id] for design_id in runtime_ids if design_id in canonical_by_id]
    result = _canonical_truth_deltas_by_id(canonical_records)

    generated_mask = np.asarray([design_id not in canonical_by_id for design_id in runtime_ids], dtype=bool)
    if generated_mask.any():
        generated = population.loc[generated_mask].copy().reset_index(drop=True)
        required_generated = {"low_tokens", "high_tokens", "approx_high_part"}
        missing_generated = sorted(required_generated.difference(generated.columns))
        if missing_generated:
            raise KeyError(
                "generated runtime IDs require frozen multiplier tokens: "
                + ", ".join(missing_generated)
            )
        if int(simulate_workers) <= 0 or int(simulate_batch_size) <= 0:
            raise ValueError("truth simulation workers and batch size must both be positive")

        root = Path(temp_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        worker_cfg = WorkerConfig(
            temp_root=str(root),
            tb_path=str(_write_binary_testbench(root)),
            iverilog=str(shutil.which("iverilog") or "iverilog"),
            vvp=str(shutil.which("vvp") or "vvp"),
        )
        specs: list[DesignSpec] = []
        for index, row in enumerate(generated.itertuples(index=False)):
            runtime_id = str(getattr(row, runtime_id_column))
            low_tokens = _parse_token_list(getattr(row, "low_tokens"), expected_len=9)
            high_tokens = _parse_token_list(getattr(row, "high_tokens"), expected_len=8)
            approx_high_part = _parse_bool(getattr(row, "approx_high_part"))
            specs.append(
                DesignSpec(
                    index=index,
                    design_id=runtime_id,
                    split=str(getattr(row, "split", "generated")),
                    approx_high_part=approx_high_part,
                    compressor_types_low=low_tokens,
                    compressor_types_high=high_tokens,
                    spec_hash=_spec_hash(
                        low_tokens,
                        high_tokens,
                        approx_high_part=approx_high_part,
                    ),
                )
            )

        batches = [
            specs[start : start + int(simulate_batch_size)]
            for start in range(0, len(specs), int(simulate_batch_size))
        ]
        worker_count = min(max(1, int(simulate_workers)), len(batches))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for batch_result in executor.map(_simulate_batch, batches, [worker_cfg] * len(batches), chunksize=1):
                for payload in batch_result:
                    runtime_id = str(payload["design_id"])
                    if runtime_id in result:
                        raise ValueError(f"duplicate truth table for runtime ID {runtime_id}")
                    result[runtime_id] = _delta_from_truth_table(np.asarray(payload["truth_table"]))

    missing_ids = sorted(set(runtime_ids).difference(result))
    if missing_ids:
        raise KeyError(f"truth tables missing for runtime IDs: {missing_ids[:5]}")
    return result
