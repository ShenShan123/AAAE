from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aaae.multiplier_generator.catalog.schema import TRUTH_TABLE_SHAPE


@dataclass(frozen=True)
class TruthShardLocation:
    shard_id: int
    row_in_shard: int
    path: str


class TruthShardWriter:
    def __init__(self, output_dir: Path, *, shard_size: int = 512) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self._buffer = np.empty((self.shard_size, *TRUTH_TABLE_SHAPE), dtype=np.uint16)
        self._count = 0
        self._shard_id = 0

    def add(self, table: np.ndarray) -> TruthShardLocation:
        if table.shape != TRUTH_TABLE_SHAPE:
            raise ValueError(f"Expected truth table shape {TRUTH_TABLE_SHAPE}, got {table.shape}")
        if table.dtype != np.uint16:
            raise ValueError(f"Expected uint16 truth table, got {table.dtype}")
        row = self._count
        self._buffer[row] = table
        location = TruthShardLocation(
            shard_id=self._shard_id,
            row_in_shard=row,
            path=f"truth_shards/shard_{self._shard_id:05d}.npy",
        )
        self._count += 1
        if self._count == self.shard_size:
            self.flush()
        return location

    def flush(self) -> None:
        if self._count == 0:
            return
        shard_path = self.output_dir / f"shard_{self._shard_id:05d}.npy"
        np.save(shard_path, self._buffer[: self._count].copy())
        self._count = 0
        self._shard_id += 1

    def finish(self) -> None:
        self.flush()
