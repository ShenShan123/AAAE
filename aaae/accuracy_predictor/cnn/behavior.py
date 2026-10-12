from __future__ import annotations

import hashlib
import json

import numpy as np

from .contracts import DataContractError


BEHAVIOR_HASH_COLUMN = "behavior_sha256"
BEHAVIOR_HASH_SCHEMA = "cnn_canonical_delta_behavior_v1"


def behavior_sha256_rows(delta: np.ndarray) -> tuple[str, ...]:
    """Return content hashes that identify numerically identical multiplier behavior."""
    values = np.asarray(delta)
    if values.ndim != 3 or values.shape[1:] != (256, 256):
        raise DataContractError("delta must have shape [design_count, 256, 256]")
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise DataContractError("delta must contain real numeric values")

    integer = np.issubdtype(values.dtype, np.integer)
    if integer and np.issubdtype(values.dtype, np.unsignedinteger):
        if values.size and np.max(values) > np.iinfo(np.int64).max:
            raise DataContractError("delta values cannot be represented safely")

    header = json.dumps(
        {
            "canonical_dtype": "int64" if integer else "float64",
            "row_shape": [256, 256],
            "schema": BEHAVIOR_HASH_SCHEMA,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    hashes: list[str] = []
    for row in values:
        if integer:
            canonical = np.ascontiguousarray(row, dtype=np.dtype("<i8"))
        else:
            try:
                canonical = np.ascontiguousarray(row, dtype=np.dtype("<f8"))
            except (OverflowError, TypeError, ValueError) as exc:
                raise DataContractError("delta cannot be represented as float64") from exc
            if not np.isfinite(canonical).all():
                raise DataContractError("delta must contain finite numeric values")
            # Positive and negative zero describe the same multiplier behavior.
            canonical[canonical == 0.0] = 0.0
        digest = hashlib.sha256()
        digest.update(header)
        digest.update(canonical.tobytes(order="C"))
        hashes.append(digest.hexdigest())
    return tuple(hashes)
