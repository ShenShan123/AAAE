from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def append_or_write_csv(row: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if not path.exists():
        atomic_write_csv(frame, path)
        return

    with path.open("r", encoding="utf-8", newline="") as handle:
        columns = next(csv.reader(handle))
    # Earlier phases can have fewer columns than later BO trials. Rewrite once
    # when the schema grows, then append subsequent rows in the existing order.
    if any(column not in columns for column in frame.columns):
        existing = pd.read_csv(path, low_memory=False)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        atomic_write_csv(combined, path)
        return

    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        has_final_newline = handle.read(1) == b"\n"
    if not has_final_newline:
        existing = pd.read_csv(path, low_memory=False)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        atomic_write_csv(combined, path)
        return

    record = frame.reindex(columns=columns).to_csv(index=False, header=False).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND)
    try:
        original_size = os.fstat(descriptor).st_size
        try:
            remaining = memoryview(record)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("failed to append BO trial CSV row")
                remaining = remaining[written:]
            os.fsync(descriptor)
        except BaseException:
            # A caught write/fsync failure must leave a parsable checkpoint.
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
            raise
    finally:
        os.close(descriptor)
