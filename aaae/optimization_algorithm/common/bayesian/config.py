from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parent
# This package lives below ``optimization_algorithms``; keep all shared
# datasets and model assets rooted at the EA_2 project directory.
from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT

from aaae.paths import EA2_ROOT as EA2_ROOT


from aaae.optimization_algorithm.nsga2.config import (  # noqa: E402
    APPROX_ERROR_MODEL_ROOT,
    AI_MODELS_ROOT,
    FIXED_HIGH_TOKENS,
    LOW_TOKEN_COUNT,
    LOW_TOKEN_MAX,
    LOW_TOKEN_MIN,
    RUNTIME_PROJECT_ROOT,
    STRICT50_CANONICAL_EXACT_BASELINE,
    STRICT50_DYNAMIC_MODEL,
    STRICT50_EXPECTED_MODEL_SHA1,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_EXPECTED_STATIC_MODEL_SHA1,
    STRICT50_LUT_DOMAIN,
    STRICT50_NODE_REGISTRY,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_S8_RESIDUAL_ABS_BOUND,
    STRICT50_SHARED_LUT_INPUT_NAME,
    STRICT50_STATIC_MODEL,
    STRICT50_UNION_INPUTS,
    STRICT50_UNION_LABELS,
    STRICT50_UNION_POSITIONS,
)


DEFAULT_RUN_NAME = "bo_amcfg_pdp_top1drop_area_true1000_v1"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "optimization" / "bayesian"
TRUE1000_TOP1_METHOD = "true1000_uniform_runtime"

DEFAULT_MODEL_KEY = "swin_tiny_patch4_window7_224"
DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_SAMPLE_SEED = 1
DEFAULT_BATCH_SIZE = 16
DEFAULT_GPU_ID = 0
DEFAULT_SHARED_LUT_INPUT_NAME = STRICT50_SHARED_LUT_INPUT_NAME
DEFAULT_SESSIONS_PER_GPU = 1
# The promoted linear48 dataset and measured production throughput use this profile.
DEFAULT_SESSION_PROFILE = "throughput"

# The current entry reads the canonical true1000 inputs and exact logits from
# the promoted linear48 dataset. Historical 72-node baseline caches are not
# valid warm-start assets for this runtime.
DEFAULT_ARCHIVE_BASELINE_DIR: Path | None = None

DEFAULT_STRICT50_RUN_DIR = (
    EA2_ROOT
    / "optimization_algorithms"
    / "nsga2_amcfg_pdp_top1drop"
    / "data"
    / "runs"
    / "nsga2_amcfg_pdp_top1drop_swin_linear48_joint4497_v1"
)

TOKEN_COLUMNS = [f"low_{idx}" for idx in range(LOW_TOKEN_COUNT)]
HIGH_TOKEN_COLUMNS = [f"high_{idx}" for idx in range(len(FIXED_HIGH_TOKENS))]
