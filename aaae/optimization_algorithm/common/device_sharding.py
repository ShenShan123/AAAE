"""Deterministic, order-preserving device sharding for parallel evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar


T = TypeVar("T")


def validate_device_weights(weights: Sequence[int] | None, device_count: int) -> tuple[int, ...]:
    """Return one positive integer weight per device, defaulting to equal weights."""
    if device_count <= 0:
        raise ValueError("at least one device is required")
    if weights is None:
        return (1,) * device_count
    if isinstance(weights, str) or len(weights) != device_count:
        raise ValueError(f"expected {device_count} device weights, one per device")
    if any(isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0 for weight in weights):
        raise ValueError("device weights must be positive integers")
    return tuple(weights)


def parse_device_weights(raw: str | None, device_count: int) -> tuple[int, ...]:
    """Parse CLI weights in the same order as ``--devices``."""
    if raw is None:
        return validate_device_weights(None, device_count)
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != device_count or any(not part.isdecimal() for part in parts):
        raise ValueError(f"--eval-device-weights requires {device_count} positive comma-separated integers")
    return validate_device_weights(tuple(int(part) for part in parts), device_count)


def split_weighted(items: Sequence[T], weights: Sequence[int]) -> list[list[T]]:
    """Smooth weighted round robin; equal weights exactly match ordinary round robin.

    Each device's shard preserves the input order. Callers restore the full result
    order using configuration hashes after workers complete.
    """
    normalized = validate_device_weights(weights, len(weights))
    shards: list[list[T]] = [[] for _ in normalized]
    current = [0] * len(normalized)
    total = sum(normalized)
    for item in items:
        for index, weight in enumerate(normalized):
            current[index] += weight
        selected = max(range(len(normalized)), key=current.__getitem__)
        current[selected] -= total
        shards[selected].append(item)
    return shards
