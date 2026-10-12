from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aaae.optimization_algorithm.common.bayesian.io_utils import atomic_write_csv, now_iso
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_ID,
    DEFAULT_MODEL_KEY,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SESSIONS_PER_GPU,
    DEFAULT_STRICT50_MODEL_PACKAGE_ID,
    DEFAULT_STRICT50_SELECTED_IMAGE_COUNT,
    FIXED_HIGH_TOKENS,
    HIGH_TOKEN_COLUMNS,
    LOW_TOKEN_COUNT,
    STRICT50_BANK_META,
    STRICT50_DATASET_ID,
    STRICT50_DATASET_SCHEMA_VERSION,
    STRICT50_EXPECTED_BANK_META_SHA1,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_FEATURE_DIM,
    STRICT50_FEATURE_SCHEMA_HASH,
    STRICT50_INJECTION_SCOPE,
    STRICT50_NODE_REGISTRY,
    STRICT50_PACKAGE_HASH,
    STRICT50_RUNTIME_BASE_FEATURE_DIM,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_SELECTED_IMAGES_HASH,
    STRICT50_SUPPLEMENTAL_FEATURE_DIM,
    STRICT50_TOP1_METHOD,
    STRICT50_TOP1_PACKAGE_DIR,
    TOKEN_COLUMNS,
)


from aaae.optimization_algorithm.nsga2.evaluators import (  # noqa: E402
    CandidateConfig,
    OptimizationEvaluator,
)
from aaae.optimization_algorithm.nsga2.strict50_runtime import (  # noqa: E402
    Strict50RuntimeEvaluator,
)


def sha1_file(path: Path) -> str | None:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except Exception:
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _safe_int(value: Any, default: int) -> int:
    if _missing(value):
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    if _missing(value):
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def selected_node_count(node_registry_path: Path = STRICT50_NODE_REGISTRY) -> int:
    try:
        payload = json.loads(Path(node_registry_path).read_text(encoding="utf-8"))
        return int(len(payload.get("nodes", [])))
    except Exception:
        return 0


def build_candidate(low_tokens: tuple[int, ...] | list[int]) -> CandidateConfig:
    return CandidateConfig(
        low_tokens=tuple(int(token) for token in low_tokens),
        high_tokens=FIXED_HIGH_TOKENS,
        approx_high_part=False,
    )


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


class Strict50EvaluationCache:
    def __init__(
        self,
        *,
        cache_csv: Path | None,
        model_key: str = DEFAULT_MODEL_KEY,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gpu_id: int = DEFAULT_GPU_ID,
        sessions_per_gpu: int = DEFAULT_SESSIONS_PER_GPU,
        node_registry_path: Path = STRICT50_NODE_REGISTRY,
        expected_package_id: str = DEFAULT_STRICT50_MODEL_PACKAGE_ID,
        selected_image_count: int = DEFAULT_STRICT50_SELECTED_IMAGE_COUNT,
    ) -> None:
        self.cache_csv = Path(cache_csv) if cache_csv is not None else None
        self.model_key = str(model_key)
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.batch_size = int(batch_size)
        self.gpu_id = int(gpu_id)
        self.sessions_per_gpu = int(sessions_per_gpu)
        self.node_registry_path = Path(node_registry_path).resolve()
        self.node_registry_hash = sha1_file(self.node_registry_path)
        self.selected_node_count = selected_node_count(self.node_registry_path)
        self.expected_package_id = str(expected_package_id)
        self.selected_image_count = int(selected_image_count)
        if self.model_key != DEFAULT_MODEL_KEY:
            raise ValueError(f"strict50 cache requires model_key={DEFAULT_MODEL_KEY!r}")
        if self.sample_count != DEFAULT_SAMPLE_COUNT or self.sample_seed != DEFAULT_SAMPLE_SEED:
            raise ValueError("strict50 cache requires canonical sample_count=500 and sample_seed=7")
        if self.expected_package_id != DEFAULT_STRICT50_MODEL_PACKAGE_ID:
            raise ValueError("strict50 cache package ID differs from the canonical joint4497 package")
        if self.selected_image_count != DEFAULT_STRICT50_SELECTED_IMAGE_COUNT:
            raise ValueError("strict50 cache requires exactly 50 selected images")
        if self.node_registry_hash != STRICT50_EXPECTED_NODE_REGISTRY_SHA1:
            raise ValueError("strict50 cache node registry hash is not canonical linear48")
        if self.selected_node_count != STRICT50_EXPECTED_NODE_COUNT:
            raise ValueError("strict50 cache node registry does not contain 48 nodes")
        self.rows: dict[str, dict[str, Any]] = {}
        self.load_stats: dict[str, int] = {"loaded": 0, "skipped": 0}
        self.import_stats: list[dict[str, Any]] = []
        self._load()

    def _normalize_row(
        self,
        row: dict[str, Any],
        *,
        cache_origin: str,
        require_strict_metadata: bool,
    ) -> dict[str, Any] | None:
        try:
            low_tokens = tuple(int(row[f"low_{idx}"]) for idx in range(LOW_TOKEN_COUNT))
            config = build_candidate(low_tokens)
        except Exception:
            return None

        expected_hash = config.config_hash()
        row_hash = str(row.get("config_hash", expected_hash))
        if row_hash and row_hash != expected_hash:
            return None

        method = str(row.get("top1_eval_method", ""))
        if method != STRICT50_TOP1_METHOD:
            return None

        if not require_strict_metadata:
            return None
        required_fields = (
            "model_package_id",
            "package_hash",
            "selected_images_hash",
            "feature_schema_hash",
            "strict50_feature_count",
            "strict50_selected_image_count",
            "runtime_selected_image_count",
            "runtime_base_feature_count",
            "supplemental_feature_count",
            "model_key",
            "sample_count",
            "sample_seed",
            "selected_node_count",
            "node_registry_hash",
            "runtime_delta_source",
            "dataset_id",
            "dataset_schema_version",
            "injection_scope",
            "bank_meta_sha1",
        )
        if any(_missing(row.get(field)) for field in required_fields):
            return None
        expected_strings = {
            "model_package_id": self.expected_package_id,
            "package_hash": STRICT50_PACKAGE_HASH,
            "selected_images_hash": STRICT50_SELECTED_IMAGES_HASH,
            "feature_schema_hash": STRICT50_FEATURE_SCHEMA_HASH,
            "model_key": self.model_key,
            "node_registry_hash": STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
            "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
            "dataset_id": STRICT50_DATASET_ID,
            "dataset_schema_version": STRICT50_DATASET_SCHEMA_VERSION,
            "injection_scope": STRICT50_INJECTION_SCOPE,
            "bank_meta_sha1": STRICT50_EXPECTED_BANK_META_SHA1,
        }
        for field, expected in expected_strings.items():
            if not expected or str(row.get(field)) != expected:
                return None
        expected_ints = {
            "strict50_feature_count": STRICT50_FEATURE_DIM,
            "strict50_selected_image_count": self.selected_image_count,
            "runtime_selected_image_count": self.selected_image_count,
            "runtime_base_feature_count": STRICT50_RUNTIME_BASE_FEATURE_DIM,
            "supplemental_feature_count": STRICT50_SUPPLEMENTAL_FEATURE_DIM,
            "sample_count": self.sample_count,
            "sample_seed": self.sample_seed,
            "selected_node_count": self.selected_node_count,
        }
        for field, expected in expected_ints.items():
            if _safe_int(row.get(field), -1) != expected:
                return None

        pdp_fj = _safe_float(row.get("pdp_fj"))
        top1_drop = _safe_float(row.get("top1_drop"))
        area_um2 = _safe_float(row.get("area_um2"))
        if not (math.isfinite(pdp_fj) and math.isfinite(top1_drop) and math.isfinite(area_um2) and area_um2 > 0):
            return None

        out = dict(row)
        out.update(candidate_token_row(config))
        out["pdp_fj"] = pdp_fj
        out["top1_drop"] = top1_drop
        out["area_um2"] = area_um2
        out["top5_drop"] = _safe_float(out.get("top5_drop"))
        out["delay_ns"] = _safe_float(out.get("delay_ns"))
        out["power_uw"] = _safe_float(out.get("power_uw"))
        out["final_truth_mae"] = _safe_float(out.get("final_truth_mae"))
        out["final_truth_max_abs"] = _safe_float(out.get("final_truth_max_abs"))
        out["sample50_top1_drop"] = _safe_float(out.get("sample50_top1_drop"))
        out["sample50_top5_drop"] = _safe_float(out.get("sample50_top5_drop"))
        out["approx_high_part"] = _as_bool(out.get("approx_high_part", False))
        for idx, token in enumerate(FIXED_HIGH_TOKENS):
            out[f"high_{idx}"] = int(out.get(f"high_{idx}", token))

        out["top1_eval_method"] = STRICT50_TOP1_METHOD
        out["model_key"] = self.model_key
        out["model_package_id"] = self.expected_package_id
        out["package_hash"] = STRICT50_PACKAGE_HASH
        out["selected_images_hash"] = STRICT50_SELECTED_IMAGES_HASH
        out["feature_schema_hash"] = STRICT50_FEATURE_SCHEMA_HASH
        out["strict50_feature_count"] = STRICT50_FEATURE_DIM
        out["strict50_selected_image_count"] = self.selected_image_count
        out["runtime_selected_image_count"] = self.selected_image_count
        out["runtime_base_feature_count"] = STRICT50_RUNTIME_BASE_FEATURE_DIM
        out["supplemental_feature_count"] = STRICT50_SUPPLEMENTAL_FEATURE_DIM
        out["dataset_id"] = STRICT50_DATASET_ID
        out["dataset_schema_version"] = STRICT50_DATASET_SCHEMA_VERSION
        out["injection_scope"] = STRICT50_INJECTION_SCOPE
        out["bank_meta_sha1"] = STRICT50_EXPECTED_BANK_META_SHA1
        out["sample_count"] = self.sample_count
        out["sample_seed"] = self.sample_seed
        out["batch_size"] = self.batch_size
        out["gpu_id"] = self.gpu_id
        out["sessions_per_gpu"] = self.sessions_per_gpu
        out["selected_node_count"] = self.selected_node_count
        out["node_registry_hash"] = self.node_registry_hash or ""
        out["runtime_delta_source"] = STRICT50_RUNTIME_DELTA_SOURCE
        out["cache_origin"] = cache_origin
        if _missing(out.get("created_at")):
            out["created_at"] = now_iso()
        return out

    def _load(self) -> None:
        if self.cache_csv is None or not self.cache_csv.exists():
            return
        frame = pd.read_csv(self.cache_csv, low_memory=False)
        if frame.empty or "config_hash" not in frame.columns:
            return
        area = pd.to_numeric(frame["area_um2"], errors="coerce") if "area_um2" in frame.columns else None
        if area is None or not (np.isfinite(area) & (area > 0)).all():
            raise ValueError(
                f"existing BO cache lacks finite area_um2: {self.cache_csv}. "
                "Use a new run directory or a separate area-backfilled cache."
            )
        for row in frame.to_dict(orient="records"):
            normalized = self._normalize_row(
                row,
                cache_origin=str(row.get("cache_origin", "live_cache")),
                require_strict_metadata=True,
            )
            if normalized is None:
                self.load_stats["skipped"] += 1
                continue
            self.rows[str(normalized["config_hash"])] = normalized
            self.load_stats["loaded"] += 1

    def import_csv(
        self,
        path: Path,
        *,
        cache_origin: str | None = None,
        overwrite: bool = False,
        limit: int | None = None,
        require_strict_metadata: bool = True,
        save: bool = True,
    ) -> dict[str, Any]:
        path = Path(path)
        stats: dict[str, Any] = {
            "path": str(path),
            "exists": path.exists(),
            "rows_seen": 0,
            "rows_imported": 0,
            "rows_skipped": 0,
            "rows_duplicate": 0,
        }
        if not path.exists():
            self.import_stats.append(stats)
            return stats

        frame = pd.read_csv(path, low_memory=False)
        if limit is not None:
            frame = frame.head(max(0, int(limit)))
        origin = cache_origin or f"seed_cache:{path}"
        stats["rows_seen"] = int(len(frame))
        for row in frame.to_dict(orient="records"):
            normalized = self._normalize_row(
                row,
                cache_origin=origin,
                require_strict_metadata=require_strict_metadata,
            )
            if normalized is None:
                stats["rows_skipped"] += 1
                continue
            key = str(normalized["config_hash"])
            if key in self.rows and not overwrite:
                stats["rows_duplicate"] += 1
                continue
            self.rows[key] = normalized
            stats["rows_imported"] += 1
        if save:
            self.save()
        self.import_stats.append(stats)
        return stats

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(str(key))

    def put(self, key: str, row: dict[str, Any], *, save: bool = True) -> None:
        normalized = self._normalize_row(
            row,
            cache_origin=str(row.get("cache_origin", "live_new_eval")),
            require_strict_metadata=True,
        )
        if normalized is None:
            raise ValueError(f"cannot cache incompatible strict50 row for {key}")
        self.rows[str(key)] = normalized
        if save:
            self.save()

    def frame(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        frame = pd.DataFrame(list(self.rows.values()))
        sort_cols = [col for col in ("pdp_fj", "top1_drop", "area_um2", "config_hash") if col in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols)
        ordered = [
            "design_id",
            "config_hash",
            "approx_high_part",
            "design_index",
            *TOKEN_COLUMNS,
            *HIGH_TOKEN_COLUMNS,
            "delay_ns",
            "power_uw",
            "pdp_fj",
            "area_um2",
            "top1_drop",
            "top5_drop",
            "final_truth_mae",
            "final_truth_max_abs",
            "runtime_delta_source",
            "top1_eval_method",
            "model_key",
            "model_package_id",
            "package_hash",
            "selected_images_hash",
            "feature_schema_hash",
            "strict50_feature_count",
            "strict50_selected_image_count",
            "runtime_selected_image_count",
            "runtime_base_feature_count",
            "supplemental_feature_count",
            "dataset_id",
            "dataset_schema_version",
            "injection_scope",
            "bank_meta_sha1",
            "sample50_top1_drop",
            "sample50_top5_drop",
            "sample_count",
            "sample_seed",
            "batch_size",
            "gpu_id",
            "sessions_per_gpu",
            "selected_node_count",
            "node_registry_hash",
            "cache_origin",
            "created_at",
        ]
        cols = [col for col in ordered if col in frame.columns]
        extra_cols = [col for col in frame.columns if col not in cols]
        return frame[cols + extra_cols].reset_index(drop=True)

    def save(self) -> None:
        if self.cache_csv is None:
            return
        atomic_write_csv(self.frame(), self.cache_csv)


class Strict50BoEvaluator:
    def __init__(
        self,
        *,
        cache_csv: Path | None,
        model_key: str = DEFAULT_MODEL_KEY,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gpu_id: int = DEFAULT_GPU_ID,
        sessions_per_gpu: int = DEFAULT_SESSIONS_PER_GPU,
        device: str | None = None,
        initialize_models: bool = True,
    ) -> None:
        self.model_key = str(model_key)
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.batch_size = int(batch_size)
        self.gpu_id = int(gpu_id)
        self.sessions_per_gpu = int(sessions_per_gpu)
        self.device = device
        self.cache = Strict50EvaluationCache(
            cache_csv=cache_csv,
            model_key=self.model_key,
            sample_count=self.sample_count,
            sample_seed=self.sample_seed,
            batch_size=self.batch_size,
            gpu_id=self.gpu_id,
            sessions_per_gpu=self.sessions_per_gpu,
        )
        self.strict50_runtime: Strict50RuntimeEvaluator | None = None
        self.engine: OptimizationEvaluator | None = None
        self.models_initialized = bool(initialize_models)
        if self.models_initialized:
            self.strict50_runtime = Strict50RuntimeEvaluator(
                package_dir=STRICT50_TOP1_PACKAGE_DIR,
                bank_meta_path=STRICT50_BANK_META,
                node_registry_path=STRICT50_NODE_REGISTRY,
                model_key=self.model_key,
                sample_count=self.sample_count,
                sample_seed=self.sample_seed,
                batch_size=self.batch_size,
                gpu_id=self.gpu_id,
                sessions_per_gpu=self.sessions_per_gpu,
            )
            self.engine = OptimizationEvaluator(
                strict50_evaluator=self.strict50_runtime,
                top1_eval_method=STRICT50_TOP1_METHOD,
                cache_csv=None,
                device=self.device,
                strict50_gpu_id=self.gpu_id,
                initialize_models=True,
            )

    @property
    def can_evaluate_new(self) -> bool:
        return self.engine is not None

    def import_seed_cache(
        self,
        path: Path,
        *,
        limit: int | None = None,
        overwrite: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        return self.cache.import_csv(
            path,
            cache_origin=f"seed_cache:{Path(path)}",
            overwrite=overwrite,
            limit=limit,
            require_strict_metadata=True,
            save=save,
        )

    def add_result_row(self, row: dict[str, Any], *, save: bool = True) -> dict[str, Any]:
        key = str(row.get("config_hash", ""))
        if not key:
            low_tokens = tuple(int(row[f"low_{idx}"]) for idx in range(LOW_TOKEN_COUNT))
            key = build_candidate(low_tokens).config_hash()
        normalized = dict(row)
        normalized.setdefault("cache_origin", "parallel_worker_eval")
        normalized.setdefault("created_at", now_iso())
        cached = self.cache._normalize_row(
            normalized,
            cache_origin=str(normalized.get("cache_origin", "parallel_worker_eval")),
            require_strict_metadata=True,
        )
        if cached is None:
            raise ValueError(f"cannot add incompatible strict50 worker row for {key}")
        self.cache.rows[str(cached["config_hash"])] = cached
        if save:
            self.cache.save()
        return cached

    def add_result_rows(self, rows: list[dict[str, Any]], *, save: bool = True) -> int:
        count = 0
        for row in rows:
            self.add_result_row(row, save=False)
            count += 1
        if save and count:
            self.cache.save()
        return count

    def evaluate(self, config: CandidateConfig, *, save: bool = True) -> dict[str, Any]:
        key = config.config_hash()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.engine is None:
            raise RuntimeError("Strict50BoEvaluator was initialized without models and cannot evaluate new candidates")
        result = self.engine.evaluate(config)
        row = result.as_row()
        row.update(
            {
                "model_key": self.model_key,
                "sample_count": self.sample_count,
                "sample_seed": self.sample_seed,
                "batch_size": self.batch_size,
                "gpu_id": self.gpu_id,
                "sessions_per_gpu": self.sessions_per_gpu,
                "runtime_selected_image_count": DEFAULT_STRICT50_SELECTED_IMAGE_COUNT,
                "selected_node_count": self.cache.selected_node_count,
                "node_registry_hash": self.cache.node_registry_hash or "",
                "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
                "dataset_schema_version": STRICT50_DATASET_SCHEMA_VERSION,
                "cache_origin": "live_new_eval",
                "created_at": now_iso(),
            }
        )
        self.cache.put(key, row, save=save)
        return self.cache.get(key) or row

    def evaluate_tokens(self, tokens: tuple[int, ...] | list[int], *, save: bool = True) -> dict[str, Any]:
        return self.evaluate(build_candidate(tokens), save=save)

    def cache_frame(self) -> pd.DataFrame:
        return self.cache.frame()

    def save_cache_csv(self) -> None:
        self.cache.save()

    def close(self) -> None:
        runtime = self.strict50_runtime
        self.engine = None
        self.strict50_runtime = None
        if runtime is not None:
            runtime.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
