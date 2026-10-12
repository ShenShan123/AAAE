"""Locations of local code and optional external experiment assets.

Source code is always imported from this project. External roots are used only
for datasets, checkpoints, ONNX models and tool resources, which are not bundled.
No directory is created when this module is imported.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_config_path = Path(os.environ.get("AAAE_PATH_CONFIG", PROJECT_ROOT / "configs/paths.json"))
_config = json.loads(_config_path.read_text()) if _config_path.is_file() else {}


def _location(key: str, environment: str, default: str) -> Path:
    value = os.environ.get(environment) or _config.get(key) or default
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


EXTERNAL_ROOT = _location("external_root", "AAAE_EXTERNAL_ROOT", "../EA_2")
OUTPUT_ROOT = _location("output_root", "AAAE_OUTPUT_ROOT", "output")
PPA_ROOT = _location("ppa_root", "AAAE_PPA_ROOT", "../ppa")
EA2_ROOT = EXTERNAL_ROOT
END_TO_END_ROOT = EXTERNAL_ROOT / "end_to_end_prediction"
APPROX_ERROR_MODEL_ROOT = EXTERNAL_ROOT / "approx_error_model"
AI_MODELS_ROOT = _location("ai_models_root", "AAAE_AI_MODELS_ROOT", str(EXTERNAL_ROOT / "ai_models"))
AI_MODELS_SRC = PROJECT_ROOT / "aaae/inference_runtime"
TRANSFORMER_ROOT = PROJECT_ROOT / "aaae/inference_runtime/transformer"
TRANSFORMER_DATA_ROOT = OUTPUT_ROOT / "transformer"
TRANSFORMER_CACHE_ROOT = TRANSFORMER_DATA_ROOT / "cache"
LEGACY_WORKFLOW_ROOT = END_TO_END_ROOT
GNN_ASSET_ROOT = _location(
    "gnn_asset_root", "AAAE_GNN_ASSET_ROOT",
    str(EXTERNAL_ROOT / "ppa_prediction/multiplier_gnn/8-bitGNN"),
)
NATIVE_ROOT = PROJECT_ROOT / "aaae/inference_runtime/native"
ORT_INCLUDE_ROOT = _location(
    "ort_include_root", "AAAE_ORT_INCLUDE_ROOT",
    str(EXTERNAL_ROOT / "ai_models/cuda_ops/cnn_cuda_ops/mux_to_approx/ort_runtime_root/include"),
)
CNN_ORT_INCLUDE_ROOT = _location(
    "cnn_ort_include_root", "AAAE_CNN_ORT_INCLUDE_ROOT",
    str(END_TO_END_ROOT / "CNN/runtime/native/onnxruntime/include"),
)


def asset_path(relative: str | Path) -> Path:
    return EXTERNAL_ROOT / relative


def output_path(relative: str | Path) -> Path:
    return OUTPUT_ROOT / relative
