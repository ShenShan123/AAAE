"""Structure contracts shared by formal Transformer physics features."""

from __future__ import annotations

import re

import numpy as np

from aaae.accuracy_predictor.features.transformer.common import FORMAL_CACHE_ROOT

NODE_COUNT = 72
PROFILE_IMAGE_COUNT = 256
CAUSAL_PREFIX_FEATURE_SCHEMA_VERSION = "swin72_causal_prefix_second_order_features_v1"
DEFAULT_COEFFICIENTS = (
    FORMAL_CACHE_ROOT / "dynamic_b16_per_image_coefficients256.float32.npy"
)
DEFAULT_PROPAGATION = FORMAL_CACHE_ROOT / "dynamic_b16_propagation_full_logits256.npz"


def profile_group_weights(
    clean_margin: np.ndarray,
    exact_correct: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Return normalized all/correct-margin-bin/wrong profile groups."""

    margin = np.asarray(clean_margin, dtype=np.float32).reshape(-1)
    correct = np.asarray(exact_correct, dtype=bool).reshape(-1)
    if margin.shape != (PROFILE_IMAGE_COUNT,) or correct.shape != margin.shape:
        raise ValueError("profile grouping expects 256 aligned images")
    correct_rows = np.flatnonzero(correct)
    wrong_rows = np.flatnonzero(~correct)
    if len(correct_rows) < 8 or len(wrong_rows) == 0:
        raise ValueError("profile grouping requires correct and wrong examples")
    ordered = correct_rows[np.argsort(margin[correct_rows], kind="stable")]
    groups = [np.arange(PROFILE_IMAGE_COUNT, dtype=np.int64)]
    names = ["all"]
    for index, rows in enumerate(np.array_split(ordered, 8)):
        groups.append(np.asarray(rows, dtype=np.int64))
        names.append(f"clean_correct_margin_bin{index}")
    groups.append(wrong_rows)
    names.append("clean_wrong")
    weights = np.zeros((len(groups), PROFILE_IMAGE_COUNT), dtype=np.float32)
    for index, rows in enumerate(groups):
        if len(rows) == 0:
            raise ValueError(f"profile group is empty: {names[index]}")
        weights[index, rows] = 1.0 / float(len(rows))
    return weights, names


def parse_node_structure(node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map the 72 registered Swin nodes to stage and operation identifiers."""

    values = np.asarray(node_ids).astype(str)
    if values.shape != (NODE_COUNT,) or len(set(values.tolist())) != NODE_COUNT:
        raise ValueError("node registry must contain 72 unique ordered nodes")
    operation_suffixes = (
        "/attn/qkv/MatMul",
        "/attn/MatMul",
        "/attn/MatMul_1",
        "/attn/proj/MatMul",
        "/mlp/fc1/MatMul",
        "/mlp/fc2/MatMul",
    )
    pattern = re.compile(r"^/layers\.(\d+)/blocks\.(\d+)/")
    stages: list[int] = []
    operations: list[int] = []
    for node_id in values.tolist():
        match = pattern.match(node_id)
        if match is None:
            raise ValueError(f"cannot parse Swin node position: {node_id}")
        stage = int(match.group(1))
        if stage not in range(4):
            raise ValueError(f"Swin stage is outside [0, 3]: {node_id}")
        matches = [
            index
            for index, suffix in enumerate(operation_suffixes)
            if node_id.endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(f"cannot parse Swin operation type: {node_id}")
        stages.append(stage)
        operations.append(matches[0])
    return np.asarray(stages, dtype=np.int64), np.asarray(operations, dtype=np.int64)
