from __future__ import annotations

import csv
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT
from aaae.paths import EA2_ROOT as EA2_ROOT
D_MULT_DIR = (
    EA2_ROOT
    / "end_to_end_prediction"
    / "resources"
    / "data"
    / "multipliers"
    / "D_mult"
    / "canonical"
)


@dataclass(frozen=True)
class MultiplierRecord:
    design_id: str
    index: int
    split: str
    split_code: int
    spec_hash: str
    truth_shard: int
    truth_row: int
    truth_path: Path
    metrics_row: dict[str, str]
    manifest_row: dict[str, Any]


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _read_metrics_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {str(row["id"]): {str(key): str(value) for key, value in row.items()} for row in reader}


@functools.lru_cache(maxsize=1)
def load_all_multiplier_records() -> tuple[MultiplierRecord, ...]:
    manifest_rows = _read_manifest(D_MULT_DIR / "meta" / "designs.jsonl")
    metrics_by_id = _read_metrics_by_id(D_MULT_DIR / "results" / "metrics_zero_operand_fixed.csv")
    records: list[MultiplierRecord] = []
    for index, row in enumerate(manifest_rows):
        design_id = str(row["id"])
        truth_path = D_MULT_DIR / str(row["truth_table_bin"])
        metrics_row = metrics_by_id[design_id]
        records.append(
            MultiplierRecord(
                design_id=design_id,
                index=index,
                split=str(row["split"]),
                split_code=int(row["split_code"]),
                spec_hash=str(row["spec_hash"]),
                truth_shard=int(row["truth_table_shard"]),
                truth_row=int(row["truth_table_row"]),
                truth_path=truth_path,
                metrics_row=metrics_row,
                manifest_row=row,
            )
        )
    return tuple(records)


def load_records_by_ids(design_ids: list[str]) -> list[MultiplierRecord]:
    by_id = {record.design_id: record for record in load_all_multiplier_records()}
    return [by_id[design_id] for design_id in design_ids]


@functools.lru_cache(maxsize=256)
def _load_truth_shard(path_str: str) -> np.ndarray:
    return np.load(Path(path_str), mmap_mode="r")


def load_truth_table(record: MultiplierRecord) -> np.ndarray:
    shard = _load_truth_shard(str(record.truth_path.resolve()))
    return np.asarray(shard[int(record.truth_row)], dtype=np.uint16).copy()
