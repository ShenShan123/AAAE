from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aaae.paths import AI_MODELS_ROOT, AI_MODELS_SRC, TRANSFORMER_ROOT

# Keep the historical public constants, but source their values from the
# migrated package's stable location contract instead of this file's parent.
PROJECT_ROOT = TRANSFORMER_ROOT
ONNX_ROOT = AI_MODELS_ROOT / "onnx_models"
DATA_ROOT = AI_MODELS_ROOT / "dataset"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    task: str
    dataset: str
    model_path: Path
    model_name_for_preprocess: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "vit_base_patch16_224": ModelSpec(
        key="vit_base_patch16_224",
        task="classification",
        dataset="imagenet1k",
        model_path=ONNX_ROOT / "classification" / "vit_base_patch16_224_imagenet1k_int8.onnx",
        model_name_for_preprocess="vit_base_patch16_224",
    ),
    "swin_tiny_patch4_window7_224": ModelSpec(
        key="swin_tiny_patch4_window7_224",
        task="classification",
        dataset="imagenet1k",
        model_path=ONNX_ROOT / "classification" / "swin_tiny_patch4_window7_224_imagenet1k_int8.onnx",
        model_name_for_preprocess="swin_tiny_patch4_window7_224",
    ),
    "deit_base_patch16_224": ModelSpec(
        key="deit_base_patch16_224",
        task="classification",
        dataset="imagenet1k",
        model_path=(
            ONNX_ROOT
            / "classification"
            / "deit_base_patch16_224_imagenet1k_int8_symmetric.onnx"
        ),
        model_name_for_preprocess="deit_base_patch16_224",
    ),
}


def get_model_spec(model_key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[str(model_key)]
    except KeyError as exc:
        raise KeyError(f"Unknown model key '{model_key}'. Choices: {', '.join(sorted(MODEL_REGISTRY))}") from exc
