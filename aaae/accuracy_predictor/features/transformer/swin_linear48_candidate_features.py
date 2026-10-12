"""Immediate Swin-linear48 causal-prefix features for candidate DMult LUTs.

This module is the small-batch inference API for the 1,034 structure features
materialized by :mod:`prepare_swin_linear48_causal_prefix_features`.  It reuses
the frozen linear48 operand/propagation profile caches and the existing
``build_causal_prefix_features`` summary implementation; it does not run Swin,
load design targets, or use per-design runtime observations.

Input domain
------------
Public LUT inputs must be current S8S8 sign-magnitude truth tables:
``lut[q_a + 128, q_b + 128]`` is the signed approximate product for raw signed
int8 codes ``q_a`` and ``q_b``.  Public residual inputs must use the same axes
and equal ``lut - (q_a * q_b)``.  Both APIs fail closed on non-integer,
mis-shaped, non-sign-magnitude, or out-of-int32-domain inputs.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer.build_swin_per_image_operand_profile import (  # noqa: E501
    LUT_CELLS,
)
from aaae.accuracy_predictor.features.transformer.build_swin_per_image_operand_profile import (  # noqa: E501
    SCHEMA_VERSION as OPERAND_PROFILE_SCHEMA,
)
from aaae.accuracy_predictor.features.transformer.common import (
    FORMAL_CACHE_ROOT,
    sha256_file,
)
from aaae.accuracy_predictor.features.transformer.physics_core import (
    profile_group_weights,
)
from aaae.accuracy_predictor.features.transformer.prepare_causal_prefix_features import (
    build_causal_prefix_features,
    compact_profile_groups,
)

DATASET_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
PROMOTED_FEATURE_CACHE = (
    DATASET_ROOT / "data" / "causal_prefix_second_order1034.float32.npy"
)
EXPERIMENT_CACHE_ROOT = PROJECT_ROOT / "experiment" / "cache"
DEFAULT_OPERAND_PROFILE_CANDIDATES = (
    FORMAL_CACHE_ROOT / "linear48_a100_operand_profile256.float32.npy",
    EXPERIMENT_CACHE_ROOT / "linear48_a100_operand_profile256.float32.npy",
)
DEFAULT_PROPAGATION_PROFILE_CANDIDATES = (
    FORMAL_CACHE_ROOT / "linear48_a100_propagation_profile256.npz",
    EXPERIMENT_CACHE_ROOT / "linear48_a100_propagation_profile256.npz",
)
DEFAULT_EXACT_META = DATASET_ROOT / "shared" / "true1000_exact_meta.npz"
DEFAULT_FEATURE_NAMES = PROMOTED_FEATURE_CACHE.with_suffix(
    PROMOTED_FEATURE_CACHE.suffix + ".features.json"
)

SCHEMA_VERSION = "swin_linear48_causal_prefix_second_order_features_v1"
PROPAGATION_SCHEMA = "swin_dynamic_b16_propagation_profile_v1"
LUT_DOMAIN = "s8s8_sign_magnitude"
RESIDUAL_DOMAIN = "s8s8_sign_magnitude_residual_from_signed_exact"
INDEX_ENCODING = "signed_code_plus_128"
TABLE_SIDE = 256
IMAGE_COUNT = 256
NODE_COUNT = 48
BLOCK_COUNT = 12
OPERATION_NAMES = ("qkv", "proj", "fc1", "fc2")
DEFAULT_BLOCK_GROUP_COUNTS = (2, 2, 6, 2)
DEFAULT_BLOCK_GROUP_LABEL = "stage"
FEATURE_COUNT = 1034
MAX_ABS_SIGN_MAGNITUDE_LUT_VALUE = 255 * 255
MAX_ABS_RESIDUAL_VALUE = MAX_ABS_SIGN_MAGNITUDE_LUT_VALUE + 128 * 128


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return value


def _first_existing(candidates: tuple[Path, ...], *, name: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"no {name} cache exists at any registered path: {joined}")


def _metadata_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _index_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".index.npz")


def _promoted_metadata_path() -> Path:
    return PROMOTED_FEATURE_CACHE.with_suffix(
        PROMOTED_FEATURE_CACHE.suffix + ".metadata.json"
    )


def _path_from_promoted_metadata(field: str) -> Path | None:
    metadata_path = _promoted_metadata_path()
    if not metadata_path.is_file():
        return None
    value = _read_object(metadata_path).get(field)
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    return None


def _default_operand_profile() -> Path:
    promoted = _path_from_promoted_metadata("operand_profile")
    if promoted is not None:
        return promoted
    return _first_existing(
        DEFAULT_OPERAND_PROFILE_CANDIDATES,
        name="linear48 operand profile",
    )


def _default_propagation_profile() -> Path:
    promoted = _path_from_promoted_metadata("propagation_profile")
    if promoted is not None:
        return promoted
    return _first_existing(
        DEFAULT_PROPAGATION_PROFILE_CANDIDATES,
        name="linear48 propagation profile",
    )


def s8s8_exact_product_table() -> np.ndarray:
    """Return ``q_a * q_b`` indexed by ``signed_code + 128``."""

    codes = np.arange(-128, 128, dtype=np.int32)
    return codes[:, None] * codes[None, :]


def _as_batched_integer_tables(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3 or array.shape[1:] != (TABLE_SIDE, TABLE_SIDE):
        raise ValueError(f"{name} must have shape (256, 256) or (N, 256, 256)")
    if array.shape[0] < 1:
        raise ValueError(f"{name} batch must contain at least one table")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise TypeError(f"{name} must use a numeric dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype, got {array.dtype}")
    as_i64 = np.asarray(array, dtype=np.int64)
    if (
        int(as_i64.min()) < int(np.iinfo(np.int32).min)
        or int(as_i64.max()) > int(np.iinfo(np.int32).max)
    ):
        raise ValueError(f"{name} values must fit in int32")
    return np.ascontiguousarray(as_i64, dtype=np.int32)


def _validate_abs_bound(values: np.ndarray, *, name: str, maximum: int) -> None:
    if int(np.max(np.abs(values.astype(np.int64, copy=False)))) > int(maximum):
        raise ValueError(f"{name} absolute values must be <= {int(maximum)}")


def _representative_indices_for_magnitude() -> np.ndarray:
    # magnitude 0 is signed code 0 (index 128), 1..127 are positive codes,
    # and magnitude 128 is represented only by code -128 (index 0).
    return np.asarray([128, *range(129, 256), 0], dtype=np.int64)


def _validate_s8s8_sign_magnitude_luts(luts: np.ndarray, *, name: str) -> None:
    codes = np.arange(-128, 128, dtype=np.int32)
    signs = np.sign(codes).astype(np.int32)
    sign_product = signs[:, None] * signs[None, :]
    magnitudes = np.abs(codes).astype(np.int64)
    representatives = _representative_indices_for_magnitude()
    abs_values = np.abs(luts.astype(np.int64, copy=False))
    expected_abs = abs_values[
        :,
        representatives[magnitudes[:, None]],
        representatives[magnitudes[None, :]],
    ]
    if not np.array_equal(abs_values, expected_abs):
        raise ValueError(
            f"{name} is not a {LUT_DOMAIN} table: magnitudes are not sign-symmetric"
        )
    signed_values = luts.astype(np.int64, copy=False) * sign_product[None, :, :]
    if np.any(signed_values < 0):
        raise ValueError(
            f"{name} is not a {LUT_DOMAIN} table: signs do not follow operand signs"
        )
    if np.any(luts[:, 128, :] != 0) or np.any(luts[:, :, 128] != 0):
        raise ValueError(
            f"{name} is not a {LUT_DOMAIN} table: signed zero axes must be zero"
        )


def s8s8_sign_magnitude_residuals_from_luts(lut_tables: np.ndarray) -> np.ndarray:
    """Validate S8S8 sign-magnitude LUTs and return int32 signed residuals.

    ``lut_tables`` may be one table with shape ``(256, 256)`` or a batch with
    shape ``(N, 256, 256)``.  Axes are raw signed int8 codes encoded as
    ``code + 128``.  The returned residual is ``lut - exact_signed_product``.
    """

    luts = _as_batched_integer_tables(lut_tables, name="lut_tables")
    _validate_abs_bound(
        luts,
        name="lut_tables",
        maximum=MAX_ABS_SIGN_MAGNITUDE_LUT_VALUE,
    )
    _validate_s8s8_sign_magnitude_luts(luts, name="lut_tables")
    residual = luts.astype(np.int64) - s8s8_exact_product_table()[
        None,
        :,
        :,
    ].astype(np.int64)
    if (
        int(residual.min()) < int(np.iinfo(np.int32).min)
        or int(residual.max()) > int(np.iinfo(np.int32).max)
    ):
        raise ValueError("lut residuals must fit in int32")
    return np.ascontiguousarray(residual, dtype=np.int32)


def validate_s8s8_sign_magnitude_residuals(residual_tables: np.ndarray) -> np.ndarray:
    """Validate int32-domain residuals and return a contiguous batch.

    Residual axes use ``signed_code + 128`` and values must reconstruct a valid
    S8S8 sign-magnitude LUT after adding the exact signed product.
    """

    residual = _as_batched_integer_tables(residual_tables, name="residual_tables")
    _validate_abs_bound(
        residual,
        name="residual_tables",
        maximum=MAX_ABS_RESIDUAL_VALUE,
    )
    luts = residual.astype(np.int64) + s8s8_exact_product_table()[
        None,
        :,
        :,
    ].astype(np.int64)
    if (
        int(luts.min()) < int(np.iinfo(np.int32).min)
        or int(luts.max()) > int(np.iinfo(np.int32).max)
    ):
        raise ValueError("residual_tables reconstruct LUT values outside int32")
    luts_i32 = np.ascontiguousarray(luts, dtype=np.int32)
    _validate_s8s8_sign_magnitude_luts(luts_i32, name="residual_tables + exact")
    return residual


def _normalize_device(device: str | None) -> str:
    specification = "cpu" if device is None else str(device).strip().lower()
    if specification == "cpu":
        return specification
    if not specification.startswith("cuda"):
        raise ValueError("device must be 'cpu' or an available CUDA device")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is not available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return specification


def _validate_node_order(node_ids: list[str]) -> None:
    expected_suffixes = tuple(
        suffix
        for _block in range(BLOCK_COUNT)
        for suffix in (
            "/attn/qkv/MatMul",
            "/attn/proj/MatMul",
            "/mlp/fc1/MatMul",
            "/mlp/fc2/MatMul",
        )
    )
    if len(node_ids) != NODE_COUNT or len(set(node_ids)) != NODE_COUNT:
        raise ValueError("linear48 node registry must contain 48 unique ordered nodes")
    if not all(
        node.endswith(suffix)
        for node, suffix in zip(node_ids, expected_suffixes, strict=True)
    ):
        raise ValueError("linear48 nodes are not in 12-block x 4-operation order")


def _load_linear48_profile_contracts(
    *,
    operand_profile: Path,
    propagation_profile: Path,
    exact_meta: Path,
    verify_profile_hashes: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    operand_path = Path(operand_profile).expanduser().resolve()
    propagation_path = Path(propagation_profile).expanduser().resolve()
    exact_meta_path = Path(exact_meta).expanduser().resolve()
    operand_metadata_path = _metadata_sidecar(operand_path)
    operand_index_path = _index_sidecar(operand_path)
    propagation_metadata_path = _metadata_sidecar(propagation_path)
    for path in (
        operand_path,
        operand_metadata_path,
        operand_index_path,
        propagation_path,
        propagation_metadata_path,
        exact_meta_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"required linear48 profile asset is missing: {path}"
            )

    operand_metadata = _read_object(operand_metadata_path)
    propagation_metadata = _read_object(propagation_metadata_path)
    expected_shape = (IMAGE_COUNT, NODE_COUNT, LUT_CELLS)
    if (
        operand_metadata.get("schema_version") != OPERAND_PROFILE_SCHEMA
        or operand_metadata.get("artifact_complete") is not True
        or operand_metadata.get("shape") != list(expected_shape)
        or operand_metadata.get("node_count") != NODE_COUNT
        or operand_metadata.get("index_sha256") is None
    ):
        raise ValueError("linear48 operand profile contract is invalid")
    if (
        propagation_metadata.get("schema_version") != PROPAGATION_SCHEMA
        or propagation_metadata.get("node_count") != NODE_COUNT
        or propagation_metadata.get("image_count") != IMAGE_COUNT
        or propagation_metadata.get("output_hashes", {}).get("sha256") is None
    ):
        raise ValueError("linear48 propagation profile contract is invalid")
    if (
        operand_metadata.get("propagation_profile_sha256")
        != propagation_metadata["output_hashes"]["sha256"]
    ):
        raise ValueError("operand and propagation profiles are not bound together")
    if verify_profile_hashes:
        if operand_metadata.get("output_sha256") != sha256_file(operand_path):
            raise ValueError("linear48 operand profile checksum mismatch")
        if operand_metadata.get("index_sha256") != sha256_file(operand_index_path):
            raise ValueError("linear48 operand index checksum mismatch")
        if (
            propagation_metadata["output_hashes"]["sha256"]
            != sha256_file(propagation_path)
        ):
            raise ValueError("linear48 propagation profile checksum mismatch")

    # Copy-on-write keeps the 3 GiB profile file-backed while exposing a
    # writable NumPy view that Torch can safely wrap without warning.
    operand = np.load(operand_path, mmap_mode="c", allow_pickle=False)
    if operand.shape != expected_shape or operand.dtype != np.float32:
        raise ValueError("linear48 operand profile array has an invalid shape or dtype")
    with np.load(operand_index_path, allow_pickle=False) as index:
        node_ids = np.asarray(index["node_ids"]).astype(str).tolist()
        selected_local = np.asarray(index["selected_true1000_indices"], dtype=np.int64)
        selected_union = np.asarray(index["selected_union_positions"], dtype=np.int64)
    with np.load(propagation_path, allow_pickle=False) as propagation:
        propagation_nodes = np.asarray(propagation["node_ids"]).astype(str).tolist()
        propagation_local = np.asarray(
            propagation["selected_true1000_indices"], dtype=np.int64
        )
        propagation_union = np.asarray(
            propagation["selected_union_positions"], dtype=np.int64
        )
        response = np.asarray(propagation["fixed_margin_response"], dtype=np.float32)
        clean_margin = np.asarray(
            propagation["clean_pred_vs_runner_up_margin"], dtype=np.float32
        )
    if (
        propagation_nodes != node_ids
        or not np.array_equal(selected_local, propagation_local)
        or not np.array_equal(selected_union, propagation_union)
        or selected_local.shape != (IMAGE_COUNT,)
        or response.shape != (NODE_COUNT, IMAGE_COUNT)
        or clean_margin.shape != (IMAGE_COUNT,)
    ):
        raise ValueError("linear48 operand and propagation arrays are not aligned")
    _validate_node_order(node_ids)

    with np.load(exact_meta_path, allow_pickle=False) as exact_meta_npz:
        correct_bank = np.asarray(
            exact_meta_npz["exact_top1_correct_bits"],
            dtype=np.uint8,
        )
    if correct_bank.shape != (1000,) or not np.all(
        (correct_bank == 0) | (correct_bank == 1)
    ):
        raise ValueError("canonical exact correctness bank is invalid")
    exact_correct = np.asarray(correct_bank[selected_local], dtype=np.uint8)
    return operand, selected_local, selected_union, response, clean_margin, exact_correct


@dataclass(frozen=True)
class SwinLinear48CandidateFeatureEvaluator:
    """Reusable evaluator for small batches of candidate DMult residuals/LUTs."""

    operand_profile: np.ndarray
    fixed_margin_response: np.ndarray
    clean_margin: np.ndarray
    group_weights: np.ndarray
    group_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    device: str
    selected_true1000_indices: np.ndarray
    selected_union_positions: np.ndarray
    operand_profile_path: Path
    propagation_profile_path: Path
    exact_meta_path: Path
    block_group_counts: tuple[int, ...] = DEFAULT_BLOCK_GROUP_COUNTS
    block_group_label: str = DEFAULT_BLOCK_GROUP_LABEL
    _device_kernel: Any | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    @classmethod
    def from_default_cache(
        cls,
        *,
        device: str = "cpu",
        verify_profile_hashes: bool = False,
    ) -> "SwinLinear48CandidateFeatureEvaluator":
        """Open the registered Swin linear48 A100 profile cache.

        The default paths are resolved from the promoted dataset metadata when
        available, with a fallback to the historical ``experiment/cache`` and
        ``FORMAL_CACHE_ROOT`` locations.  Set ``verify_profile_hashes=True`` for
        an audit pass that re-hashes the multi-GiB operand profile.
        """

        return cls.from_profile_cache(
            operand_profile=_default_operand_profile(),
            propagation_profile=_default_propagation_profile(),
            exact_meta=DEFAULT_EXACT_META,
            feature_names_path=DEFAULT_FEATURE_NAMES,
            device=device,
            verify_profile_hashes=verify_profile_hashes,
        )

    @classmethod
    def from_profile_cache(
        cls,
        *,
        operand_profile: Path | str,
        propagation_profile: Path | str,
        exact_meta: Path | str = DEFAULT_EXACT_META,
        feature_names_path: Path | str | None = DEFAULT_FEATURE_NAMES,
        block_group_counts: tuple[int, ...] = DEFAULT_BLOCK_GROUP_COUNTS,
        block_group_label: str = DEFAULT_BLOCK_GROUP_LABEL,
        device: str = "cpu",
        verify_profile_hashes: bool = False,
    ) -> "SwinLinear48CandidateFeatureEvaluator":
        """Open validated profile assets and precompute the feature schema."""

        normalized_device = _normalize_device(device)
        (
            operand,
            selected_local,
            selected_union,
            response,
            clean_margin,
            exact_correct,
        ) = _load_linear48_profile_contracts(
            operand_profile=Path(operand_profile),
            propagation_profile=Path(propagation_profile),
            exact_meta=Path(exact_meta),
            verify_profile_hashes=verify_profile_hashes,
        )
        full_groups, full_group_names = profile_group_weights(
            clean_margin,
            exact_correct,
        )
        groups, group_names = compact_profile_groups(full_groups, full_group_names)
        probe, feature_names = build_causal_prefix_features(
            np.zeros((1, IMAGE_COUNT, NODE_COUNT), dtype=np.float32),
            fixed_margin_response=response,
            profile_margin=clean_margin,
            group_weights=groups,
            group_names=group_names,
            operation_names=OPERATION_NAMES,
            block_group_counts=block_group_counts,
            block_group_label=block_group_label,
        )
        if probe.shape != (1, FEATURE_COUNT) or len(feature_names) != FEATURE_COUNT:
            raise ValueError("linear48 causal-prefix feature schema is not 1034-wide")
        names_path = (
            Path(feature_names_path).expanduser().resolve()
            if feature_names_path is not None
            else None
        )
        if names_path is not None and names_path.is_file():
            cached_names = json.loads(names_path.read_text(encoding="ascii"))
            if cached_names != feature_names:
                raise ValueError(
                    "generated feature names do not match the registered cache"
                )
        return cls(
            operand_profile=operand,
            fixed_margin_response=response,
            clean_margin=clean_margin,
            group_weights=groups,
            group_names=tuple(group_names),
            feature_names=tuple(feature_names),
            device=normalized_device,
            selected_true1000_indices=selected_local,
            selected_union_positions=selected_union,
            operand_profile_path=Path(operand_profile).expanduser().resolve(),
            propagation_profile_path=Path(propagation_profile).expanduser().resolve(),
            exact_meta_path=Path(exact_meta).expanduser().resolve(),
            block_group_counts=tuple(int(value) for value in block_group_counts),
            block_group_label=str(block_group_label),
        )

    def features_from_lut(self, lut_tables: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Return ``float32[N, 1034]`` features from S8S8 sign-magnitude LUTs."""

        return self.features_from_signed_residuals(
            s8s8_sign_magnitude_residuals_from_luts(lut_tables)
        )

    def features_from_signed_residuals(
        self,
        residual_tables: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Return ``float32[N, 1034]`` features from S8S8 residual tables."""

        residual = validate_s8s8_sign_magnitude_residuals(residual_tables)
        amplitude = self._contract_residuals(residual)
        features, names = build_causal_prefix_features(
            np.asarray(amplitude, dtype=np.float32),
            fixed_margin_response=self.fixed_margin_response,
            profile_margin=self.clean_margin,
            group_weights=self.group_weights,
            group_names=list(self.group_names),
            operation_names=OPERATION_NAMES,
            block_group_counts=self.block_group_counts,
            block_group_label=self.block_group_label,
        )
        if (
            features.shape != (len(residual), FEATURE_COUNT)
            or names != list(self.feature_names)
            or features.dtype != np.float32
            or not np.isfinite(features).all()
        ):
            raise RuntimeError("linear48 causal-prefix feature generation failed")
        return features, list(names)

    def features_from_residual(
        self,
        residual_tables: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Alias for :meth:`features_from_signed_residuals`."""

        return self.features_from_signed_residuals(residual_tables)

    def _contract_residuals(self, residual: np.ndarray) -> np.ndarray:
        flat_residual = residual.astype(np.float32, copy=False).reshape(
            len(residual),
            LUT_CELLS,
        )
        if self.device == "cpu":
            flat_kernel = self.operand_profile.reshape(
                IMAGE_COUNT * NODE_COUNT,
                LUT_CELLS,
            )
            return (
                flat_residual @ flat_kernel.T
            ).reshape(len(residual), IMAGE_COUNT, NODE_COUNT).astype(np.float32)

        import torch

        torch_device = torch.device(self.device)
        kernel = self._device_kernel
        if kernel is None:
            kernel = torch.as_tensor(
                np.asarray(self.operand_profile),
                dtype=torch.float32,
                device=torch_device,
            ).reshape(IMAGE_COUNT * NODE_COUNT, LUT_CELLS)
            object.__setattr__(self, "_device_kernel", kernel)
        with torch.inference_mode():
            amplitude = torch.as_tensor(flat_residual, device=torch_device) @ kernel.T
            return np.asarray(
                amplitude.reshape(len(residual), IMAGE_COUNT, NODE_COUNT)
                .cpu()
                .numpy(),
                dtype=np.float32,
            )

    def close(self) -> None:
        """Release the lazily cached CUDA profile tensor, if one was created."""

        object.__setattr__(self, "_device_kernel", None)


def build_swin_linear48_causal_prefix_features_from_lut(
    lut_tables: np.ndarray,
    *,
    evaluator: SwinLinear48CandidateFeatureEvaluator | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """One-shot convenience API for S8S8 sign-magnitude LUT tables."""

    active = (
        evaluator
        if evaluator is not None
        else SwinLinear48CandidateFeatureEvaluator.from_default_cache(device=device)
    )
    return active.features_from_lut(lut_tables)


def build_swin_linear48_candidate_features_from_signed_residuals(
    residual_tables: np.ndarray,
    *,
    evaluator: SwinLinear48CandidateFeatureEvaluator | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """One-shot candidate-evaluator API for S8S8 signed residual tables."""

    active = (
        evaluator
        if evaluator is not None
        else SwinLinear48CandidateFeatureEvaluator.from_default_cache(device=device)
    )
    return active.features_from_signed_residuals(residual_tables)


def build_swin_linear48_causal_prefix_features_from_residual(
    residual_tables: np.ndarray,
    *,
    evaluator: SwinLinear48CandidateFeatureEvaluator | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    """Compatibility alias for signed residual candidate features."""

    return build_swin_linear48_candidate_features_from_signed_residuals(
        residual_tables,
        evaluator=evaluator,
        device=device,
    )


SwinLinear48CausalPrefixFeatureBuilder = SwinLinear48CandidateFeatureEvaluator


__all__ = [
    "FEATURE_COUNT",
    "INDEX_ENCODING",
    "LUT_DOMAIN",
    "MAX_ABS_RESIDUAL_VALUE",
    "MAX_ABS_SIGN_MAGNITUDE_LUT_VALUE",
    "RESIDUAL_DOMAIN",
    "SwinLinear48CandidateFeatureEvaluator",
    "SwinLinear48CausalPrefixFeatureBuilder",
    "build_swin_linear48_candidate_features_from_signed_residuals",
    "build_swin_linear48_causal_prefix_features_from_lut",
    "build_swin_linear48_causal_prefix_features_from_residual",
    "s8s8_exact_product_table",
    "s8s8_sign_magnitude_residuals_from_luts",
    "validate_s8s8_sign_magnitude_residuals",
]
