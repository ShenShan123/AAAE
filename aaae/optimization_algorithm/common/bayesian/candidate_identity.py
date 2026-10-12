"""Shared candidate identity helpers from the canonical BO implementation."""
from __future__ import annotations
import hashlib
import json
from typing import Any
from aaae.optimization_algorithm.nsga2.config import FIXED_HIGH_TOKENS, LOW_TOKEN_COUNT, LOW_TOKEN_MIN, LOW_TOKEN_MAX
from aaae.optimization_algorithm.nsga2.evaluators import CandidateConfig

def build_candidate(low_tokens: tuple[int, ...] | list[int]) -> CandidateConfig:
    return CandidateConfig(
        low_tokens=tuple(int(token) for token in low_tokens),
        high_tokens=FIXED_HIGH_TOKENS,
        approx_high_part=False,
    )


def _fixed_config_hash_template() -> tuple[bytes, bytes]:
    config_dict = build_candidate((LOW_TOKEN_MIN,) * LOW_TOKEN_COUNT).config_dict()
    config_dict["low_tokens"] = ["__LOW_TOKENS__"]
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    prefix, suffix = payload.split(b'"__LOW_TOKENS__"')
    return prefix, suffix


_HASH_PREFIX, _HASH_SUFFIX = _fixed_config_hash_template()
_TOKEN_BYTES = {token: str(token).encode("ascii") for token in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1)}


def config_hash_for_low_tokens(low_tokens: tuple[int, ...] | list[int]) -> str:
    """Hash a fixed-high BO candidate without constructing 50k configs per pool."""
    tokens = tuple(int(token) for token in low_tokens)
    if len(tokens) != LOW_TOKEN_COUNT:
        return build_candidate(tokens).config_hash()
    try:
        token_bytes = b",".join(_TOKEN_BYTES[token] for token in tokens)
    except KeyError:
        return build_candidate(tokens).config_hash()
    payload = _HASH_PREFIX + token_bytes + _HASH_SUFFIX
    return hashlib.sha1(payload).hexdigest()[:16]


def candidate_token_row(config: CandidateConfig) -> dict[str, Any]:
    row: dict[str, Any] = {
        "config_hash": config.config_hash(),
        "approx_high_part": bool(config.approx_high_part),
        "design_index": int(config.design_index()),
    }
    for idx, token in enumerate(config.low_tokens):
        row[f"low_{idx}"] = int(token)
    for idx, token in enumerate(config.high_tokens):
        row[f"high_{idx}"] = int(token)
    return row


