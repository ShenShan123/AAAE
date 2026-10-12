from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
from aaae.paths import EXTERNAL_ROOT, END_TO_END_ROOT, AI_MODELS_ROOT, APPROX_ERROR_MODEL_ROOT, OUTPUT_ROOT

from aaae.paths import EA2_ROOT as EA2_ROOT
PPA_GNN_DIR = EA2_ROOT / "ppa_prediction" / "multiplier_gnn" / "8-bitGNN"
# Runtime source and canonical assets live inside end_to_end_prediction.
RUNTIME_PROJECT_ROOT = END_TO_END_ROOT

DMULT_CHECKPOINT = APPROX_ERROR_MODEL_ROOT / "runs" / "truth_table" / "full17_neural_logic" / "checkpoints" / "best.pt"
SWIN_LINEAR48_DATASET_ROOT = (
    END_TO_END_ROOT
    / "datasets"
    / "swin_tiny"
    / "s8s8_linear48_strict50_true1000_raw500_v1"
)
STANDARDIZED_DATA_DIR = SWIN_LINEAR48_DATASET_ROOT / "data"
MERGED_DATA_DIR = STANDARDIZED_DATA_DIR
STRICT50_TOP1_PACKAGE_DIR = (
    END_TO_END_ROOT
    / "data"
    / "runs"
    / "swin_tiny_linear48_e2e_joint4497_20260801"
    / "strict50_package"
)
STRICT50_BANK_META = SWIN_LINEAR48_DATASET_ROOT / "shared" / "strict50_bank_meta.npz"
STRICT50_BANK_META_CONTRACT = (
    SWIN_LINEAR48_DATASET_ROOT / "shared" / "strict50_bank_meta_contract.json"
)
STRICT50_DATASET_MANIFEST = SWIN_LINEAR48_DATASET_ROOT / "data" / "manifest.json"
STRICT50_RUNTIME_BANK = SWIN_LINEAR48_DATASET_ROOT / "data" / "runtime_bank.npz"
STRICT50_SUPPLEMENTAL_FEATURES = (
    SWIN_LINEAR48_DATASET_ROOT
    / "data"
    / "causal_prefix_second_order1034.float32.npy"
)
STRICT50_SUPPLEMENTAL_FEATURE_NAMES = STRICT50_SUPPLEMENTAL_FEATURES.with_suffix(
    STRICT50_SUPPLEMENTAL_FEATURES.suffix + ".features.json"
)
STRICT50_SUPPLEMENTAL_FEATURE_METADATA = STRICT50_SUPPLEMENTAL_FEATURES.with_suffix(
    STRICT50_SUPPLEMENTAL_FEATURES.suffix + ".metadata.json"
)
STRICT50_NODE_REGISTRY = (
    END_TO_END_ROOT
    / "resources"
    / "data"
    / "swin_tiny_s8s8_linear48"
    / "node_registry"
    / "node_registry.json"
)
STRICT50_DYNAMIC_MODEL = (
    AI_MODELS_ROOT
    / "onnx_models"
    / "classification"
    / "swin_tiny_patch4_window7_224_imagenet1k_int8_dynamic.onnx"
)
STRICT50_STATIC_MODEL = (
    AI_MODELS_ROOT
    / "onnx_models"
    / "classification"
    / "swin_tiny_patch4_window7_224_imagenet1k_int8.onnx"
)
STRICT50_UNION_INPUTS = SWIN_LINEAR48_DATASET_ROOT / "shared" / "union_inputs.npy"
STRICT50_UNION_LABELS = SWIN_LINEAR48_DATASET_ROOT / "shared" / "union_labels.npy"
STRICT50_UNION_POSITIONS = SWIN_LINEAR48_DATASET_ROOT / "shared" / "union_positions.npz"
STRICT50_CANONICAL_EXACT_BASELINE = (
    SWIN_LINEAR48_DATASET_ROOT / "shared" / "canonical_exact_baseline.npz"
)
STRICT50_TRUE1000_EXACT_META = (
    SWIN_LINEAR48_DATASET_ROOT / "shared" / "true1000_exact_meta.npz"
)
STRICT50_STRUCTURE_OPERAND_PROFILE = (
    END_TO_END_ROOT / "experiment" / "cache" / "linear48_a100_operand_profile256.float32.npy"
)
STRICT50_STRUCTURE_PROPAGATION_PROFILE = (
    END_TO_END_ROOT / "experiment" / "cache" / "linear48_a100_propagation_profile256.npz"
)
STRICT50_MODEL_KEY = "swin_tiny_patch4_window7_224"
STRICT50_DATASET_ID = "swin_tiny_s8s8_linear48_strict50_true1000_raw500_v1"
STRICT50_DATASET_SCHEMA_VERSION = (
    "transformer_s8s8_linear48_strict50_true1000_raw500_signed_static_v4"
)
STRICT50_INJECTION_SCOPE = "sram_cim_weight_stationary_linear_only_v1"
STRICT50_LUT_DOMAIN = "s8s8_sign_magnitude"
STRICT50_SHARED_LUT_INPUT_NAME = "uniform_s8s8_residual"
STRICT50_RUNTIME_DELTA_SOURCE = (
    "neural_logic_u8_truth_u16_to_s8s8_sign_magnitude_residual_v1"
)
STRICT50_S8_RESIDUAL_ABS_BOUND = 856
STRICT50_EXPECTED_NODE_COUNT = 48
STRICT50_EXPECTED_MODEL_SHA1 = "dd78030b0b5b36c5d85de5a24b177573998b7661"
STRICT50_EXPECTED_STATIC_MODEL_SHA1 = "a26dee6906f60a358bc5c320889c413114ecbe89"
STRICT50_EXPECTED_NODE_REGISTRY_SHA1 = "a515f4180d1ca3492546e7eed600659e4c9cd3d0"
STRICT50_EXPECTED_BANK_META_SHA1 = "3c95eaa3d353ceac7c18fd1f98ed0b15e77bbd7f"
STRICT50_EXPECTED_PACKAGE_ID = (
    "swin_tiny_linear48_e2e_joint4497_20260801_strict50_top1"
)
STRICT50_EXPECTED_PACKAGE_HASH = "f97356b790fdee00a1ba7a7a237a1088e2cc2d7c"
STRICT50_EXPECTED_SELECTED_IMAGES_HASH = "5554a7a942ff953e7bc829c0ad5b122422002cf1"
STRICT50_EXPECTED_FEATURE_SCHEMA_HASH = "e8db06261b330099cb2c6f09680e8ee7380686fb"

DEFAULT_RUN_NAME = "nsga2_amcfg_pdp_top1drop_v1"
DEFAULT_STRICT50_RUN_NAME = "nsga2_amcfg_pdp_top1drop_swin_linear48_joint4497_v1"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "optimization" / "nsga2"
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / DEFAULT_RUN_NAME
DEFAULT_STRICT50_OUTPUT_DIR = DEFAULT_OUTPUT_ROOT / DEFAULT_STRICT50_RUN_NAME

LOW_TOKEN_MIN = 1
LOW_TOKEN_MAX = 8
LOW_TOKEN_COUNT = 9
HIGH_TOKEN_COUNT = 8
FIXED_HIGH_TOKENS = (0,) * HIGH_TOKEN_COUNT

LOW_TOKEN_TO_PPA_STATE = {token: token - 1 for token in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1)}
PPA_STATE_TO_LOW_TOKEN = {state: token for token, state in LOW_TOKEN_TO_PPA_STATE.items()}

TARGET_COLUMNS = ("top1_drop", "top5_drop")
FEATURE_DIM = 3103
STRICT50_RUNTIME_BASE_FEATURE_DIM = 3463
STRICT50_SUPPLEMENTAL_FEATURE_DIM = 1034
STRICT50_FEATURE_DIM = STRICT50_RUNTIME_BASE_FEATURE_DIM + STRICT50_SUPPLEMENTAL_FEATURE_DIM

PPA_MIN_MAX_8BIT = {
    "MRED": (0.00086214, 0.15909895),
    "NMED": (0.00009676, 0.00313291),
    "Area": (435.97400000, 480.92800000),
    "Delay": (1.30000000, 1.34000000),
    "Power": (146.00000000, 184.00000000),
}
