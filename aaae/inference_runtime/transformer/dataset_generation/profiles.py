"""Immutable model-specific inputs for the shared signed-S8S8 pipeline."""

from __future__ import annotations

from aaae.paths import OUTPUT_ROOT

from dataclasses import dataclass
from pathlib import Path

from aaae.paths import (
    AI_MODELS_ROOT,
    END_TO_END_ROOT,
    TRANSFORMER_CACHE_ROOT,
    TRANSFORMER_DATA_ROOT,
)
from aaae.inference_runtime.transformer.dataset_generation.matmul_scope import INCLUDED_ROLES, INJECTION_SCOPE_ID


DEFAULT_MODEL_KEY = "swin_tiny_patch4_window7_224"
TRUE_PROTOCOL_COUNT = 1000
TRUE_PROTOCOL_SEED = 1
RAW_PROTOCOL_COUNT = 500
RAW_PROTOCOL_SEED = 7

_CLASSIFICATION_MODELS = AI_MODELS_ROOT / "onnx_models" / "classification"
_HISTORICAL_SWIN_S8_DATASET_ROOT = (
    END_TO_END_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_strict50_true1000_raw500_v2"
)
_SWIN_S8_DATA = _HISTORICAL_SWIN_S8_DATASET_ROOT / "data"
_SWIN_S8_DATASET_ROOT = (
    END_TO_END_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
_VIT_S8_DATASET_ROOT = (
    END_TO_END_ROOT
    / "datasets"
    / "vit"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
_DEIT_S8_DATASET_ROOT = (
    END_TO_END_ROOT
    / "datasets"
    / "deit"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
_PROTOCOL_ROOT = TRANSFORMER_DATA_ROOT / "frozen_protocols"
_CACHE_ROOT = TRANSFORMER_CACHE_ROOT / "calibration"


@dataclass(frozen=True)
class TransformerDatasetProfile:
    """Backbone-specific values consumed by the common dataset protocol."""

    model_key: str
    dataset_key: str
    dataset_id: str
    static_model_path: Path
    dynamic_model_path: Path
    preprocess_key: str
    node_registry_path: Path
    expected_total_matmuls: int
    expected_injected_matmuls: int
    expected_excluded_matmuls: tuple[str, ...]
    injection_scope_id: str
    included_roles: tuple[str, ...]
    default_output_root: Path
    canonical_output_root: Path
    default_source_data_dir: Path = _SWIN_S8_DATA

    @property
    def default_true1000_subset(self) -> Path:
        return _PROTOCOL_ROOT / "true1000_seed1.json"

    @property
    def default_raw500_subset(self) -> Path:
        return _PROTOCOL_ROOT / "raw500_seed7.json"

    @property
    def default_true1000_cache(self) -> Path:
        return _CACHE_ROOT / f"{self.model_key}_n{TRUE_PROTOCOL_COUNT}_seed{TRUE_PROTOCOL_SEED}.npy"

    @property
    def default_raw500_cache(self) -> Path:
        return _CACHE_ROOT / f"{self.model_key}_n{RAW_PROTOCOL_COUNT}_seed{RAW_PROTOCOL_SEED}.npy"

    def validate_local_assets(self) -> None:
        missing = [
            str(path)
            for path in (self.static_model_path, self.dynamic_model_path, self.node_registry_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{self.model_key}: required dataset assets are missing: {missing}")

    def injection_mode_contract(self) -> dict[str, object]:
        return {
            "kind": "uniform_s8s8_sign_magnitude_residual",
            "injection_scope_id": self.injection_scope_id,
            "total_matmul_count": int(self.expected_total_matmuls),
            "injected_matmul_count": int(self.expected_injected_matmuls),
            "included_roles": list(self.included_roles),
            "excluded_matmuls": list(self.expected_excluded_matmuls),
            "semantic_zero_masking_required": True,
        }


_VIT_ATTENTION_EXCLUDED = tuple(
    f"/blocks/blocks.{block_id}/attn/MatMul{suffix}"
    for block_id in range(12)
    for suffix in ("", "_1")
)
_SWIN_BLOCKS = tuple(
    (stage_id, block_id)
    for stage_id, block_count in enumerate((2, 2, 6, 2))
    for block_id in range(block_count)
)
_SWIN_REDUCTION_EXCLUDED = (
    "/layers.0/downsample/reduction/MatMul",
    "/layers.1/downsample/reduction/MatMul",
    "/layers.2/downsample/reduction/MatMul",
)


def _swin_excluded_matmuls() -> tuple[str, ...]:
    names: list[str] = []
    for stage_id, block_count in enumerate((2, 2, 6, 2)):
        for block_id in range(block_count):
            names.extend(
                (
                    f"/layers.{stage_id}/blocks.{block_id}/attn/MatMul",
                    f"/layers.{stage_id}/blocks.{block_id}/attn/MatMul_1",
                )
            )
        if stage_id < len(_SWIN_REDUCTION_EXCLUDED):
            names.append(_SWIN_REDUCTION_EXCLUDED[stage_id])
    return tuple(names)


_SWIN_EXCLUDED = _swin_excluded_matmuls()


DATASET_PROFILES: dict[str, TransformerDatasetProfile] = {
    "swin_tiny_patch4_window7_224": TransformerDatasetProfile(
        model_key="swin_tiny_patch4_window7_224",
        dataset_key="swin_tiny",
        dataset_id="swin_tiny_s8s8_linear48_strict50_true1000_raw500_v1",
        static_model_path=_CLASSIFICATION_MODELS / "swin_tiny_patch4_window7_224_imagenet1k_int8.onnx",
        dynamic_model_path=(
            _CLASSIFICATION_MODELS / "swin_tiny_patch4_window7_224_imagenet1k_int8_dynamic.onnx"
        ),
        preprocess_key="swin_tiny_patch4_window7_224",
        node_registry_path=(
            END_TO_END_ROOT
            / "resources"
            / "data"
            / "swin_tiny_s8s8_linear48"
            / "node_registry"
            / "node_registry.json"
        ),
        expected_total_matmuls=75,
        expected_injected_matmuls=48,
        expected_excluded_matmuls=_SWIN_EXCLUDED,
        injection_scope_id=INJECTION_SCOPE_ID,
        included_roles=INCLUDED_ROLES,
        default_output_root=(
            OUTPUT_ROOT / "transformer" / "runs"
            / "swin_tiny_int8_s8s8_linear48_strict50_true1000_raw500_20k_v1"
        ),
        canonical_output_root=OUTPUT_ROOT / "datasets" / "swin_tiny" / _SWIN_S8_DATASET_ROOT.name,
    ),
    "vit_base_patch16_224": TransformerDatasetProfile(
        model_key="vit_base_patch16_224",
        dataset_key="vit",
        dataset_id="vit_base_patch16_224_s8s8_linear48_strict50_true1000_raw500_v1",
        static_model_path=_CLASSIFICATION_MODELS / "vit_base_patch16_224_imagenet1k_int8.onnx",
        dynamic_model_path=(
            _CLASSIFICATION_MODELS / "vit_base_patch16_224_imagenet1k_int8_dynamic.onnx"
        ),
        preprocess_key="vit_base_patch16_224",
        node_registry_path=(
            END_TO_END_ROOT
            / "resources"
            / "data"
            / "vit_base_patch16_224_s8s8_linear48"
            / "node_registry"
            / "node_registry.json"
        ),
        expected_total_matmuls=72,
        expected_injected_matmuls=48,
        expected_excluded_matmuls=_VIT_ATTENTION_EXCLUDED,
        injection_scope_id=INJECTION_SCOPE_ID,
        included_roles=INCLUDED_ROLES,
        default_output_root=(
            OUTPUT_ROOT / "transformer" / "runs"
            / "vit_base_patch16_224_int8_s8s8_linear48_strict50_true1000_raw500_20k_v1"
        ),
        canonical_output_root=OUTPUT_ROOT / "datasets" / "vit" / _VIT_S8_DATASET_ROOT.name,
    ),
    "deit_base_patch16_224": TransformerDatasetProfile(
        model_key="deit_base_patch16_224",
        dataset_key="deit",
        dataset_id="deit_base_patch16_224_s8s8_linear48_strict50_true1000_raw500_v1",
        static_model_path=(
            _CLASSIFICATION_MODELS
            / "deit_base_patch16_224_imagenet1k_int8_symmetric.onnx"
        ),
        dynamic_model_path=(
            _CLASSIFICATION_MODELS
            / "deit_base_patch16_224_imagenet1k_int8_symmetric_dynamic.onnx"
        ),
        preprocess_key="deit_base_patch16_224",
        node_registry_path=(
            END_TO_END_ROOT
            / "resources"
            / "data"
            / "deit_base_patch16_224_s8s8_linear48"
            / "node_registry"
            / "node_registry.json"
        ),
        expected_total_matmuls=72,
        expected_injected_matmuls=48,
        expected_excluded_matmuls=_VIT_ATTENTION_EXCLUDED,
        injection_scope_id=INJECTION_SCOPE_ID,
        included_roles=INCLUDED_ROLES,
        default_output_root=(
            OUTPUT_ROOT / "transformer" / "runs"
            / "deit_base_patch16_224_int8_s8s8_linear48_strict50_true1000_raw500_20k_v1"
        ),
        canonical_output_root=OUTPUT_ROOT / "datasets" / "deit" / _DEIT_S8_DATASET_ROOT.name,
    ),
}


def get_dataset_profile(model_key: str) -> TransformerDatasetProfile:
    try:
        return DATASET_PROFILES[str(model_key)]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_PROFILES))
        raise KeyError(f"Unknown Transformer dataset model key {model_key!r}. Choices: {choices}") from exc
