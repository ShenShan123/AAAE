from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOW_TOKEN_COUNT, LOW_TOKEN_MAX, LOW_TOKEN_MIN, TOKEN_COLUMNS


TokenSet = tuple[int, ...]


def normalize_tokens(tokens: Sequence[int]) -> TokenSet:
    out = tuple(int(token) for token in tokens)
    if len(out) != LOW_TOKEN_COUNT:
        raise ValueError(f"expected {LOW_TOKEN_COUNT} tokens, got {len(out)}")
    if any(token < LOW_TOKEN_MIN or token > LOW_TOKEN_MAX for token in out):
        raise ValueError(f"tokens must be in [{LOW_TOKEN_MIN}, {LOW_TOKEN_MAX}], got {out}")
    return out


def unique_token_sets(items: Iterable[Sequence[int]]) -> list[TokenSet]:
    seen: set[TokenSet] = set()
    out: list[TokenSet] = []
    for item in items:
        try:
            tokens = normalize_tokens(item)
        except Exception:
            continue
        if tokens in seen:
            continue
        seen.add(tokens)
        out.append(tokens)
    return out


def all_constant_token_sets() -> list[TokenSet]:
    return [(token,) * LOW_TOKEN_COUNT for token in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1)]


def structured_warm_start_token_sets() -> list[TokenSet]:
    patterns: list[TokenSet] = []
    patterns.extend(all_constant_token_sets())
    for token in (3, 4, 8):
        for count in range(LOW_TOKEN_COUNT + 1):
            patterns.append((token,) * count + (1,) * (LOW_TOKEN_COUNT - count))
            patterns.append((1,) * (LOW_TOKEN_COUNT - count) + (token,) * count)
    for token in (2, 3, 4, 8):
        for pos in range(LOW_TOKEN_COUNT):
            arr = [1] * LOW_TOKEN_COUNT
            arr[pos] = token
            patterns.append(tuple(arr))
    for token in (3, 4, 8):
        for pos1 in range(LOW_TOKEN_COUNT):
            for pos2 in range(pos1 + 1, LOW_TOKEN_COUNT):
                arr = [1] * LOW_TOKEN_COUNT
                arr[pos1] = token
                arr[pos2] = token
                patterns.append(tuple(arr))
    return unique_token_sets(patterns)


def token_sets_from_frame(frame: pd.DataFrame, *, sort_by: Sequence[str] = ("pdp_fj", "top1_drop")) -> list[TokenSet]:
    if frame.empty or any(column not in frame.columns for column in TOKEN_COLUMNS):
        return []
    work = frame.copy()
    order_cols = [column for column in sort_by if column in work.columns]
    if order_cols:
        work = work.sort_values(order_cols)
    token_frame = work[TOKEN_COLUMNS].apply(pd.to_numeric, errors="coerce")
    token_frame = token_frame.dropna()
    if token_frame.empty:
        return []
    return unique_token_sets(token_frame.astype(int).itertuples(index=False, name=None))


def token_sets_from_cache_csv(path: Path, *, limit: int | None = None) -> list[TokenSet]:
    path = Path(path)
    if not path.exists():
        return []
    frame = pd.read_csv(path, low_memory=False)
    tokens = token_sets_from_frame(frame)
    if limit is None:
        return tokens
    return tokens[: max(0, int(limit))]


def hamming_neighbors(tokens: Sequence[int], *, radius: int = 1, limit: int | None = None, seed: int = 0) -> list[TokenSet]:
    base = normalize_tokens(tokens)
    rng = np.random.default_rng(int(seed))
    neighbors: list[TokenSet] = []
    if radius <= 0:
        return neighbors
    for pos in range(LOW_TOKEN_COUNT):
        for value in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1):
            if value == base[pos]:
                continue
            arr = list(base)
            arr[pos] = value
            neighbors.append(tuple(arr))
    if radius >= 2:
        for pos1 in range(LOW_TOKEN_COUNT):
            for pos2 in range(pos1 + 1, LOW_TOKEN_COUNT):
                for value1 in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1):
                    for value2 in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1):
                        if value1 == base[pos1] and value2 == base[pos2]:
                            continue
                        arr = list(base)
                        arr[pos1] = value1
                        arr[pos2] = value2
                        neighbors.append(tuple(arr))
    out = unique_token_sets(neighbors)
    rng.shuffle(out)
    if limit is not None:
        out = out[: max(0, int(limit))]
    return out


def random_token_sets(count: int, *, seed: int) -> list[TokenSet]:
    rng = np.random.default_rng(int(seed))
    values = rng.integers(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1, size=(max(0, int(count)), LOW_TOKEN_COUNT))
    return unique_token_sets(values.tolist())


def generate_candidate_pool(
    *,
    pool_size: int,
    seed: int,
    observed: Iterable[Sequence[int]] = (),
    cache_paths: Iterable[Path] = (),
    include_warm_start: bool = True,
    neighbor_radius: int = 1,
    neighbor_limit_per_observed: int = 32,
    cache_limit_per_file: int = 256,
) -> list[TokenSet]:
    ordered: list[TokenSet] = []
    if include_warm_start:
        ordered.extend(structured_warm_start_token_sets())
    for path in cache_paths:
        ordered.extend(token_sets_from_cache_csv(Path(path), limit=cache_limit_per_file))

    observed_tokens = unique_token_sets(observed)
    ordered.extend(observed_tokens)
    for idx, tokens in enumerate(observed_tokens):
        ordered.extend(
            hamming_neighbors(
                tokens,
                radius=neighbor_radius,
                limit=neighbor_limit_per_observed,
                seed=int(seed) + idx,
            )
        )

    out = unique_token_sets(ordered)
    seen = set(out)
    attempts = 0
    while len(out) < int(pool_size) and attempts < 32:
        missing = int(pool_size) - len(out)
        for tokens in random_token_sets(max(missing * 2, 128), seed=int(seed) + attempts * 7919):
            if tokens not in seen:
                seen.add(tokens)
                out.append(tokens)
        attempts += 1
    return out[: max(0, int(pool_size))]


def tokens_to_feature_matrix(tokens: Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray([normalize_tokens(item) for item in tokens], dtype=np.float64)
    ordinal = (arr - LOW_TOKEN_MIN) / float(LOW_TOKEN_MAX - LOW_TOKEN_MIN)
    pieces = [ordinal]
    one_hot = np.zeros((arr.shape[0], LOW_TOKEN_COUNT * (LOW_TOKEN_MAX - LOW_TOKEN_MIN + 1)), dtype=np.float64)
    for row_idx in range(arr.shape[0]):
        for pos in range(LOW_TOKEN_COUNT):
            token_idx = int(arr[row_idx, pos] - LOW_TOKEN_MIN)
            one_hot[row_idx, pos * (LOW_TOKEN_MAX - LOW_TOKEN_MIN + 1) + token_idx] = 1.0
    pieces.append(one_hot)
    return np.concatenate(pieces, axis=1)


def token_columns_from_frame(frame: pd.DataFrame) -> list[TokenSet]:
    return token_sets_from_frame(frame, sort_by=())
