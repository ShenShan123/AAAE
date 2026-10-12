from __future__ import annotations

import sys
from pathlib import Path
import json


PACKAGE_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PACKAGE_ROOT.parent
# This package lives below ``optimization_algorithms``; keep all shared
# datasets and model assets rooted at the EA_2 project directory.
from aaae.paths import EXTERNAL_ROOT, OUTPUT_ROOT

from aaae.paths import EA2_ROOT as EA2_ROOT


from aaae.optimization_algorithm.nsga2.config import (  # noqa: E402
    AI_MODELS_ROOT,
    FIXED_HIGH_TOKENS,
    LOW_TOKEN_COUNT,
    LOW_TOKEN_MAX,
    LOW_TOKEN_MIN,
    RUNTIME_PROJECT_ROOT,
    STRICT50_BANK_META,
    STRICT50_DATASET_ID,
    STRICT50_DATASET_MANIFEST,
    STRICT50_DATASET_SCHEMA_VERSION,
    STRICT50_EXPECTED_BANK_META_SHA1,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_FEATURE_DIM,
    STRICT50_INJECTION_SCOPE,
    STRICT50_NODE_REGISTRY,
    STRICT50_RUNTIME_BASE_FEATURE_DIM,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_SUPPLEMENTAL_FEATURE_DIM,
    STRICT50_TOP1_PACKAGE_DIR,
)


DEFAULT_RUN_NAME = "bo_amcfg_pdp_top1drop_area_swin_linear48_joint4497_v1"
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT / "optimization" / "bayesian"
STRICT50_TOP1_METHOD = "strict50_v3"

DEFAULT_MODEL_KEY = "swin_tiny_patch4_window7_224"
DEFAULT_SAMPLE_COUNT = 500
DEFAULT_SAMPLE_SEED = 7
DEFAULT_BATCH_SIZE = 32
DEFAULT_GPU_ID = 0
DEFAULT_SESSIONS_PER_GPU = 1
DEFAULT_STRICT50_SELECTED_IMAGE_COUNT = 50
DEFAULT_STRICT50_MODEL_PACKAGE_ID = (
    "swin_tiny_linear48_e2e_joint4497_20260801_strict50_top1"
)


def _load_strict50_manifest() -> dict:
    path = STRICT50_TOP1_PACKAGE_DIR / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


STRICT50_PACKAGE_MANIFEST = _load_strict50_manifest()
STRICT50_SELECTED_IMAGES_HASH = str(STRICT50_PACKAGE_MANIFEST.get("selected_images_hash", ""))
STRICT50_FEATURE_SCHEMA_HASH = str(STRICT50_PACKAGE_MANIFEST.get("feature_schema_hash", ""))
STRICT50_PACKAGE_HASH = str(STRICT50_PACKAGE_MANIFEST.get("package_hash", ""))

# Historical 72-node/U8 caches are deliberately not warm-start inputs for the
# signed linear48 package.  A new cache can still be supplied explicitly.
DEFAULT_SEED_CACHE: Path | None = None
DEFAULT_STRICT50_HISTORY_RUN_DIRS: list[Path] = []

TOKEN_COLUMNS = [f"low_{idx}" for idx in range(LOW_TOKEN_COUNT)]
HIGH_TOKEN_COLUMNS = [f"high_{idx}" for idx in range(len(FIXED_HIGH_TOKENS))]
