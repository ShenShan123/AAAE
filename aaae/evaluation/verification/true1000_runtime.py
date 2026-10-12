from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from aaae.optimization_algorithm.common.bayesian.config import (
    APPROX_ERROR_MODEL_ROOT,
    AI_MODELS_ROOT,
    DEFAULT_ARCHIVE_BASELINE_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_ID,
    DEFAULT_MODEL_KEY,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SESSION_PROFILE,
    DEFAULT_SESSIONS_PER_GPU,
    DEFAULT_SHARED_LUT_INPUT_NAME,
    RUNTIME_PROJECT_ROOT,
    STRICT50_CANONICAL_EXACT_BASELINE,
    STRICT50_DYNAMIC_MODEL,
    STRICT50_EXPECTED_MODEL_SHA1,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_EXPECTED_STATIC_MODEL_SHA1,
    STRICT50_LUT_DOMAIN,
    STRICT50_NODE_REGISTRY,
    STRICT50_S8_RESIDUAL_ABS_BOUND,
    STRICT50_STATIC_MODEL,
    STRICT50_UNION_INPUTS,
    STRICT50_UNION_LABELS,
    STRICT50_UNION_POSITIONS,
)


def _ensure_path(path: Path) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        pass  # Local package imports are resolved by the entry point.


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


os.environ.setdefault("APPROX_AI_MODELS_ROOT", str(AI_MODELS_ROOT))
_ensure_path(RUNTIME_PROJECT_ROOT)
_ensure_path(APPROX_ERROR_MODEL_ROOT)

from aaae.multiplier_generator.lut_domains import (  # type: ignore  # noqa: E402
    s8s8_exact_table_np,
    u8_lut_to_s8s8_sign_magnitude,
)
from aaae.accuracy_predictor.features.transformer.swin_linear48_candidate_features import (  # noqa: E402
    validate_s8s8_sign_magnitude_residuals,
)


def runtime_delta_from_truth_u16(truth_u16: np.ndarray) -> np.ndarray:
    """Convert the neural decoder's U8 truth table to canonical S8S8 residuals."""

    truth = np.asarray(truth_u16)
    if truth.shape != (256, 256):
        raise ValueError(f"truth_u16 must have shape (256, 256), got {truth.shape}")
    if not np.issubdtype(truth.dtype, np.integer):
        raise TypeError(f"truth_u16 must use an integer dtype, got {truth.dtype}")
    if int(truth.min()) < 0 or int(truth.max()) > int(np.iinfo(np.uint16).max):
        raise ValueError("truth_u16 values must fit in uint16")
    signed_lut = np.asarray(
        u8_lut_to_s8s8_sign_magnitude(truth.astype(np.uint16, copy=False)),
        dtype=np.int32,
    )
    residual = np.ascontiguousarray(
        signed_lut - s8s8_exact_table_np().astype(np.int32, copy=False),
        dtype=np.int32,
    )
    return validate_s8s8_sign_magnitude_residuals(residual)[0]


@dataclass(frozen=True)
class EvalContext:
    spec: Any
    inputs: np.ndarray
    labels: np.ndarray
    exact_logits: np.ndarray
    exact_probs: np.ndarray
    exact_top1: np.ndarray
    exact_top5: np.ndarray
    exact_top1_acc: float
    exact_top5_acc: float
    sample_count: int
    baseline_logits_hash: str


class True1000RuntimeEvaluator:
    """Run exact true1000 evaluation with the current linear48 S8S8 model."""

    def __init__(
        self,
        *,
        model_key: str = DEFAULT_MODEL_KEY,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gpu_id: int = DEFAULT_GPU_ID,
        sessions_per_gpu: int = DEFAULT_SESSIONS_PER_GPU,
        session_profile: str = DEFAULT_SESSION_PROFILE,
        shared_lut_input_name: str = DEFAULT_SHARED_LUT_INPUT_NAME,
        node_registry_path: Path = STRICT50_NODE_REGISTRY,
        dynamic_model_path: Path = STRICT50_DYNAMIC_MODEL,
        static_model_path: Path = STRICT50_STATIC_MODEL,
        union_inputs_path: Path = STRICT50_UNION_INPUTS,
        union_labels_path: Path = STRICT50_UNION_LABELS,
        union_positions_path: Path = STRICT50_UNION_POSITIONS,
        canonical_exact_baseline_path: Path = STRICT50_CANONICAL_EXACT_BASELINE,
        baseline_cache_dir: Path | None = None,
        archive_baseline_dir: Path | None = DEFAULT_ARCHIVE_BASELINE_DIR,
        batch_fallback_parallelism: int = 1,
        build_runtime: bool = True,
    ) -> None:
        self.model_key = str(model_key)
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.batch_size = max(1, int(batch_size))
        self.gpu_id = int(gpu_id)
        self.sessions_per_gpu = max(1, int(sessions_per_gpu))
        self.session_profile = str(session_profile).strip().lower()
        if self.session_profile not in {"strict", "throughput"}:
            raise ValueError(
                "session_profile must be one of {'strict', 'throughput'}, "
                f"got {session_profile!r}"
            )
        self.shared_lut_input_name = str(shared_lut_input_name)
        self.node_registry_path = Path(node_registry_path).resolve()
        self.dynamic_model_path = Path(dynamic_model_path).resolve()
        self.static_model_path = Path(static_model_path).resolve()
        self.union_inputs_path = Path(union_inputs_path).resolve()
        self.union_labels_path = Path(union_labels_path).resolve()
        self.union_positions_path = Path(union_positions_path).resolve()
        self.canonical_exact_baseline_path = Path(canonical_exact_baseline_path).resolve()
        # Retained in the signature for CLI compatibility. The current entry
        # always uses the sealed canonical exact baseline instead.
        self.baseline_cache_dir = (
            Path(baseline_cache_dir).resolve() if baseline_cache_dir is not None else None
        )
        self.archive_baseline_dir = (
            Path(archive_baseline_dir).resolve() if archive_baseline_dir is not None else None
        )
        self.batch_fallback_parallelism = max(1, int(batch_fallback_parallelism))
        self.runtime_pool: Any | None = None

        if self.model_key != DEFAULT_MODEL_KEY:
            raise ValueError(
                f"canonical true1000 runtime requires model_key={DEFAULT_MODEL_KEY!r}"
            )
        if self.sample_count != 1000 or self.sample_seed != 1:
            raise ValueError("canonical true1000 runtime requires sample_count=1000 and sample_seed=1")
        if self.shared_lut_input_name != DEFAULT_SHARED_LUT_INPUT_NAME:
            raise ValueError(
                f"canonical true1000 runtime requires LUT input {DEFAULT_SHARED_LUT_INPUT_NAME!r}"
            )

        required = (
            self.node_registry_path,
            self.dynamic_model_path,
            self.static_model_path,
            self.union_inputs_path,
            self.union_labels_path,
            self.union_positions_path,
            self.canonical_exact_baseline_path,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"canonical true1000 assets are missing: {missing}")

        self.node_registry_hash = _sha1_file(self.node_registry_path)
        self.dynamic_model_sha1 = _sha1_file(self.dynamic_model_path)
        self.static_model_sha1 = _sha1_file(self.static_model_path)
        if self.node_registry_hash != STRICT50_EXPECTED_NODE_REGISTRY_SHA1:
            raise ValueError("true1000 node registry is not the canonical linear48 registry")
        if self.dynamic_model_sha1 != STRICT50_EXPECTED_MODEL_SHA1:
            raise ValueError("true1000 dynamic model differs from the canonical model")
        if self.static_model_sha1 != STRICT50_EXPECTED_STATIC_MODEL_SHA1:
            raise ValueError("true1000 static shape reference differs from the canonical model")

        node_registry = json.loads(self.node_registry_path.read_text(encoding="utf-8"))
        self.selected_node_names = [str(row["node_id"]) for row in node_registry["nodes"]]
        self.selected_node_count = len(self.selected_node_names)
        if (
            self.selected_node_count != STRICT50_EXPECTED_NODE_COUNT
            or len(set(self.selected_node_names)) != self.selected_node_count
        ):
            raise ValueError("true1000 node registry must contain 48 unique nodes")

        self.eval_ctx = self._prepare_eval_context()
        if build_runtime:
            self._build_runtime_pool()

    def _runtime_imports(self) -> dict[str, Any]:
        _ensure_path(RUNTIME_PROJECT_ROOT)
        from aaae.inference_runtime.transformer.metrics import (
            _metric_accumulator_finalize,
            _metric_accumulator_init,
            _metric_accumulator_update,
            _softmax,
            _topk_indices,
        )
        from aaae.inference_runtime.transformer.assignment_runtime import build_uniform_assignment_runtime_pool

        return {
            "metric_accumulator_finalize": _metric_accumulator_finalize,
            "metric_accumulator_init": _metric_accumulator_init,
            "metric_accumulator_update": _metric_accumulator_update,
            "softmax": _softmax,
            "topk_indices": _topk_indices,
            "build_uniform_assignment_runtime_pool": build_uniform_assignment_runtime_pool,
        }

    def _prepare_eval_context(self) -> EvalContext:
        imports = self._runtime_imports()
        union_inputs = np.load(self.union_inputs_path, mmap_mode="r", allow_pickle=False)
        union_labels = np.load(self.union_labels_path, mmap_mode="r", allow_pickle=False)
        with np.load(self.union_positions_path, allow_pickle=False) as positions:
            true1000_positions = np.asarray(positions["true1000_positions"], dtype=np.int64)
            raw500_positions = np.asarray(positions["raw500_positions"], dtype=np.int64)
        with np.load(self.canonical_exact_baseline_path, allow_pickle=False) as baseline:
            all_exact_logits = np.asarray(baseline["logits"], dtype=np.float32)
            baseline_labels = np.asarray(baseline["union_labels"], dtype=np.int64)
            baseline_true_positions = np.asarray(
                baseline["true1000_positions"], dtype=np.int64
            )
            baseline_raw_positions = np.asarray(
                baseline["raw500_positions"], dtype=np.int64
            )

        if (
            union_inputs.shape != (1486, 3, 224, 224)
            or union_inputs.dtype != np.float32
            or union_labels.shape != (1486,)
            or true1000_positions.shape != (1000,)
            or raw500_positions.shape != (500,)
            or all_exact_logits.shape != (1486, 1000)
            or not np.array_equal(baseline_labels, union_labels)
            or not np.array_equal(baseline_true_positions, true1000_positions)
            or not np.array_equal(baseline_raw_positions, raw500_positions)
            or len(np.unique(true1000_positions)) != 1000
        ):
            raise ValueError("canonical true1000 union/baseline assets are not aligned")

        inputs = np.ascontiguousarray(union_inputs[true1000_positions], dtype=np.float32)
        labels = np.asarray(union_labels[true1000_positions], dtype=np.int64)
        exact_logits = np.asarray(all_exact_logits[true1000_positions], dtype=np.float32)
        exact_probs = imports["softmax"](np.asarray(exact_logits, dtype=np.float64))
        exact_top1 = np.argmax(exact_logits, axis=1)
        exact_top5 = imports["topk_indices"](exact_logits, 5)
        exact_top1_acc = float(np.count_nonzero(exact_top1 == labels) / labels.shape[0])
        exact_top5_acc = float(
            np.count_nonzero(np.any(exact_top5 == labels[:, None], axis=1))
            / labels.shape[0]
        )
        return EvalContext(
            spec=SimpleNamespace(model_path=self.dynamic_model_path),
            inputs=inputs,
            labels=labels,
            exact_logits=exact_logits,
            exact_probs=np.asarray(exact_probs, dtype=np.float64),
            exact_top1=np.asarray(exact_top1, dtype=np.int64),
            exact_top5=np.asarray(exact_top5, dtype=np.int64),
            exact_top1_acc=exact_top1_acc,
            exact_top5_acc=exact_top5_acc,
            sample_count=int(inputs.shape[0]),
            baseline_logits_hash=_sha1_file(self.canonical_exact_baseline_path),
        )

    def _build_runtime_pool(self) -> None:
        imports = self._runtime_imports()
        self.runtime_pool = imports["build_uniform_assignment_runtime_pool"](
            self.dynamic_model_path,
            selected_node_names=self.selected_node_names,
            shared_lut_input_name=self.shared_lut_input_name,
            device_id=self.gpu_id,
            runtime_count=self.sessions_per_gpu,
            batch_fallback_parallelism=self.batch_fallback_parallelism,
            lut_domain=STRICT50_LUT_DOMAIN,
            s8_residual_abs_bound=STRICT50_S8_RESIDUAL_ABS_BOUND,
            session_profile=self.session_profile,
            static_shape_reference_path=self.static_model_path,
        )

    @staticmethod
    def _validated_runtime_delta(value: np.ndarray) -> np.ndarray:
        residual = validate_s8s8_sign_magnitude_residuals(value)
        if residual.shape != (1, 256, 256):
            raise ValueError(f"one candidate residual is required, got {residual.shape}")
        if int(np.max(np.abs(residual.astype(np.int64)))) > STRICT50_S8_RESIDUAL_ABS_BOUND:
            raise ValueError(
                f"runtime residual exceeds canonical bound {STRICT50_S8_RESIDUAL_ABS_BOUND}"
            )
        return np.ascontiguousarray(residual[0], dtype=np.int32)

    def evaluate_one(
        self,
        *,
        config_hash: str,
        runtime_delta_i32: np.ndarray,
    ) -> dict[str, float | int | str]:
        key = str(config_hash)
        return self.evaluate_many({key: runtime_delta_i32})[key]

    def evaluate_many(
        self,
        deltas_by_hash: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float | int | str]]:
        if self.runtime_pool is None:
            raise RuntimeError("runtime pool is not initialized")
        if not deltas_by_hash:
            return {}
        deltas: dict[str, np.ndarray] = {}
        for config_hash, value in deltas_by_hash.items():
            key = str(config_hash)
            if key in deltas:
                raise ValueError(f"duplicate true1000 config key {key!r}")
            deltas[key] = self._validated_runtime_delta(value)

        imports = self._runtime_imports()
        accumulators = {
            config_hash: imports["metric_accumulator_init"]()
            for config_hash in deltas
        }
        sample_count = int(self.eval_ctx.sample_count)
        start_time = time.time()
        for start in range(0, sample_count, self.batch_size):
            end = min(sample_count, start + self.batch_size)
            batch = self.eval_ctx.inputs[start:end]
            logits_by_key = self.runtime_pool.run_uniform_many(
                batch,
                approx_deltas_by_key=deltas,
            )
            if set(logits_by_key) != set(deltas):
                raise RuntimeError("uniform true1000 runtime returned a different key set")
            for config_hash in deltas:
                approx_logits = np.asarray(logits_by_key[config_hash], dtype=np.float32)
                imports["metric_accumulator_update"](
                    accumulators[config_hash],
                    approx_logits=approx_logits,
                    batch_exact=self.eval_ctx.exact_logits[start:end],
                    batch_exact_probs=self.eval_ctx.exact_probs[start:end],
                    batch_exact_top1=self.eval_ctx.exact_top1[start:end],
                    batch_labels=self.eval_ctx.labels[start:end],
                )
        elapsed_seconds = float(time.time() - start_time)

        results: dict[str, dict[str, float | int | str]] = {}
        for config_hash, accumulator in accumulators.items():
            metrics = imports["metric_accumulator_finalize"](
                accumulator,
                sample_count=sample_count,
                exact_top1_acc=float(self.eval_ctx.exact_top1_acc),
                exact_top5_acc=float(self.eval_ctx.exact_top5_acc),
            )
            metrics.update(
                {
                    "elapsed_seconds": elapsed_seconds,
                    "multi_config_batch_size": len(deltas),
                    "model_key": self.model_key,
                    "sample_count": self.sample_count,
                    "sample_seed": self.sample_seed,
                    "batch_size": self.batch_size,
                    "gpu_id": self.gpu_id,
                    "sessions_per_gpu": self.sessions_per_gpu,
                    "session_profile": self.session_profile,
                    "selected_node_count": self.selected_node_count,
                    "node_registry_hash": self.node_registry_hash,
                    "model_sha1": self.dynamic_model_sha1,
                    "baseline_logits_hash": self.eval_ctx.baseline_logits_hash,
                    "runtime_delta_source": (
                        "neural_logic_u8_truth_u16_to_s8s8_sign_magnitude_residual_v1"
                    ),
                    "runtime_lut_domain": STRICT50_LUT_DOMAIN,
                }
            )
            results[config_hash] = metrics
        return results

    def evaluate_zero_delta(self) -> dict[str, float | int | str]:
        return self.evaluate_one(
            config_hash="zero_delta_exact",
            runtime_delta_i32=np.zeros((256, 256), dtype=np.int32),
        )

    def close(self) -> None:
        pool = getattr(self, "runtime_pool", None)
        if pool is not None:
            pool.close()
        self.runtime_pool = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
