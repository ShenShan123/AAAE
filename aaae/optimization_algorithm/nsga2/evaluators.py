from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import (
    APPROX_ERROR_MODEL_ROOT,
    DMULT_CHECKPOINT,
    FIXED_HIGH_TOKENS,
    LOW_TOKEN_COUNT,
    LOW_TOKEN_MAX,
    LOW_TOKEN_MIN,
    LOW_TOKEN_TO_PPA_STATE,
    PPA_GNN_DIR,
    PPA_MIN_MAX_8BIT,
    STRICT50_DATASET_ID,
    STRICT50_EXPECTED_BANK_META_SHA1,
    STRICT50_EXPECTED_FEATURE_SCHEMA_HASH,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_EXPECTED_PACKAGE_HASH,
    STRICT50_EXPECTED_PACKAGE_ID,
    STRICT50_EXPECTED_SELECTED_IMAGES_HASH,
    STRICT50_FEATURE_DIM,
    STRICT50_INJECTION_SCOPE,
    STRICT50_MODEL_KEY,
    STRICT50_RUNTIME_BASE_FEATURE_DIM,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_SUPPLEMENTAL_FEATURE_DIM,
)
from aaae.accuracy_predictor.features.static.feature_builder import candidate_metadata
from .strict50_runtime import Strict50RuntimeEvaluator


TRUE1000_TOP1_METHOD = "true1000_uniform_runtime"
DEFAULT_TRUE1000_MODEL_KEY = "swin_tiny_patch4_window7_224"
DEFAULT_TRUE1000_SAMPLE_COUNT = 1000
DEFAULT_TRUE1000_SAMPLE_SEED = 1
DEFAULT_TRUE1000_BATCH_SIZE = 16
DEFAULT_TRUE1000_SESSIONS_PER_GPU = 1
DEFAULT_TRUE1000_SESSION_PROFILE = "throughput"
DEFAULT_TRUE1000_CONFIG_BATCH_SIZE = 16


def _ensure_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        pass  # Local package imports are resolved by the entry point.


_ensure_path(PPA_GNN_DIR)
_ensure_path(APPROX_ERROR_MODEL_ROOT)

from aaae.multiplier_gnn.my_io import state_to_graph  # type: ignore  # noqa: E402
from aaae.multiplier_gnn.gnn_predictor import predict_metrics  # type: ignore  # noqa: E402
from aaae.multiplier_generator.truth_table import (  # type: ignore  # noqa: E402
    NeuralLogicTruthDecoder,
    UINT16_MAX,
    exact_table_np,
)
from aaae.multiplier_generator.lut_domains import (  # type: ignore  # noqa: E402
    s8s8_exact_table_np,
    s8s8_metric12_from_lut,
    u8_lut_to_s8s8_sign_magnitude,
)


def cuda_gpu_id_from_device(device: str | torch.device | None) -> int:
    if device is None:
        return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    parsed = torch.device(device)
    if parsed.type != "cuda":
        return 0
    if parsed.index is not None:
        return int(parsed.index)
    return int(torch.cuda.current_device()) if torch.cuda.is_available() else 0


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def now_token_string(tokens: tuple[int, ...]) -> str:
    return ",".join(str(int(x)) for x in tokens)


def denormalize(value: Any, key: str, min_max_dict: dict[str, tuple[float, float]]) -> float:
    min_val, max_val = min_max_dict[key]
    if hasattr(value, "item"):
        value = value.item()
    return float(value) * (max_val - min_val) + min_val


def ensure_low_tokens(low_tokens: tuple[int, ...]) -> tuple[int, ...]:
    if len(low_tokens) != LOW_TOKEN_COUNT:
        raise ValueError(f"Expected {LOW_TOKEN_COUNT} low tokens, got {len(low_tokens)}")
    if any(token < LOW_TOKEN_MIN or token > LOW_TOKEN_MAX for token in low_tokens):
        raise ValueError(f"Low tokens must be in [{LOW_TOKEN_MIN}, {LOW_TOKEN_MAX}], got {low_tokens}")
    return tuple(int(token) for token in low_tokens)


@dataclass(frozen=True)
class CandidateConfig:
    low_tokens: tuple[int, ...]
    high_tokens: tuple[int, ...] = FIXED_HIGH_TOKENS
    approx_high_part: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "low_tokens", ensure_low_tokens(self.low_tokens))
        if len(self.high_tokens) != len(FIXED_HIGH_TOKENS):
            raise ValueError(f"Expected {len(FIXED_HIGH_TOKENS)} high tokens, got {len(self.high_tokens)}")
        high_tokens = tuple(int(token) for token in self.high_tokens)
        if bool(self.approx_high_part):
            if any(token < LOW_TOKEN_MIN or token > LOW_TOKEN_MAX for token in high_tokens):
                raise ValueError(
                    f"Approximated high tokens must be in [{LOW_TOKEN_MIN}, {LOW_TOKEN_MAX}], "
                    f"got {high_tokens}"
                )
        elif high_tokens != FIXED_HIGH_TOKENS:
            raise ValueError(
                "Non-approximated high-part configs must use the fixed all-zero high tokens"
            )
        object.__setattr__(self, "high_tokens", high_tokens)

    def config_dict(self) -> dict[str, Any]:
        return {
            "low_tokens": list(self.low_tokens),
            "high_tokens": list(self.high_tokens),
            "approx_high_part": bool(self.approx_high_part),
        }

    def config_hash(self) -> str:
        payload = json.dumps(self.config_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]

    def ppa_state(self) -> list[int]:
        return [LOW_TOKEN_TO_PPA_STATE[token] for token in self.low_tokens]

    def slot_tokens(self) -> list[int]:
        return [*self.low_tokens, *self.high_tokens]

    def design_index(self) -> int:
        return int(int(self.config_hash(), 16) % 20_000)

    def metadata_frame(self) -> pd.DataFrame:
        return candidate_metadata(
            low_tokens=self.low_tokens,
            high_tokens=self.high_tokens,
            approx_high_part=self.approx_high_part,
            design_id=f"candidate_{self.config_hash()}",
            design_index=self.design_index(),
            config_hash=self.config_hash(),
        )


@dataclass
class EvaluationResult:
    config_hash: str
    low_tokens: tuple[int, ...]
    high_tokens: tuple[int, ...]
    approx_high_part: bool
    design_index: int
    delay_ns: float
    power_uw: float
    pdp_fj: float
    area_um2: float
    top1_drop: float
    top5_drop: float
    final_truth_mae: float
    final_truth_max_abs: float
    top1_eval_method: str
    model_package_id: str
    selected_images_hash: str
    feature_schema_hash: str
    strict50_feature_count: int
    strict50_selected_image_count: int
    sample50_top1_drop: float
    sample50_top5_drop: float
    package_hash: str = ""
    runtime_base_feature_count: int = 0
    supplemental_feature_count: int = 0
    dataset_id: str = ""
    injection_scope: str = ""
    bank_meta_sha1: str = ""
    ensemble_disagreement: float = float("nan")
    conformal_radius_95: float = float("nan")
    conformal_lower_95: float = float("nan")
    conformal_upper_95: float = float("nan")
    exact_top1_acc: float = float("nan")
    approx_top1_acc: float = float("nan")
    exact_top5_acc: float = float("nan")
    approx_top5_acc: float = float("nan")
    top1_consistency: float = float("nan")
    top5_consistency: float = float("nan")
    logit_kl: float = float("nan")
    logit_cosine: float = float("nan")
    output_mse: float = float("nan")
    output_cosine: float = float("nan")
    runtime_delta_source: str = ""
    model_key: str = ""
    sample_count: int = 0
    sample_seed: int = 0
    batch_size: int = 0
    gpu_id: int = 0
    sessions_per_gpu: int = 0
    session_profile: str = ""
    selected_node_count: int = 0
    node_registry_hash: str = ""
    baseline_logits_hash: str = ""
    elapsed_seconds: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        row = {
            "config_hash": self.config_hash,
            "approx_high_part": bool(self.approx_high_part),
            "design_index": int(self.design_index),
            "delay_ns": float(self.delay_ns),
            "power_uw": float(self.power_uw),
            "pdp_fj": float(self.pdp_fj),
            "area_um2": float(self.area_um2),
            "top1_drop": float(self.top1_drop),
            "top5_drop": float(self.top5_drop),
            "final_truth_mae": float(self.final_truth_mae),
            "final_truth_max_abs": float(self.final_truth_max_abs),
            "top1_eval_method": str(self.top1_eval_method),
            "model_package_id": str(self.model_package_id),
            "selected_images_hash": str(self.selected_images_hash),
            "feature_schema_hash": str(self.feature_schema_hash),
            "strict50_feature_count": int(self.strict50_feature_count),
            "strict50_selected_image_count": int(self.strict50_selected_image_count),
            "sample50_top1_drop": float(self.sample50_top1_drop),
            "sample50_top5_drop": float(self.sample50_top5_drop),
            "package_hash": str(self.package_hash),
            "runtime_base_feature_count": int(self.runtime_base_feature_count),
            "supplemental_feature_count": int(self.supplemental_feature_count),
            "dataset_id": str(self.dataset_id),
            "injection_scope": str(self.injection_scope),
            "bank_meta_sha1": str(self.bank_meta_sha1),
            "ensemble_disagreement": float(self.ensemble_disagreement),
            "conformal_radius_95": float(self.conformal_radius_95),
            "conformal_lower_95": float(self.conformal_lower_95),
            "conformal_upper_95": float(self.conformal_upper_95),
            "exact_top1_acc": float(self.exact_top1_acc),
            "approx_top1_acc": float(self.approx_top1_acc),
            "exact_top5_acc": float(self.exact_top5_acc),
            "approx_top5_acc": float(self.approx_top5_acc),
            "top1_consistency": float(self.top1_consistency),
            "top5_consistency": float(self.top5_consistency),
            "logit_kl": float(self.logit_kl),
            "logit_cosine": float(self.logit_cosine),
            "output_mse": float(self.output_mse),
            "output_cosine": float(self.output_cosine),
            "runtime_delta_source": str(self.runtime_delta_source),
            "model_key": str(self.model_key),
            "sample_count": int(self.sample_count),
            "sample_seed": int(self.sample_seed),
            "batch_size": int(self.batch_size),
            "gpu_id": int(self.gpu_id),
            "sessions_per_gpu": int(self.sessions_per_gpu),
            "session_profile": str(self.session_profile),
            "selected_node_count": int(self.selected_node_count),
            "node_registry_hash": str(self.node_registry_hash),
            "baseline_logits_hash": str(self.baseline_logits_hash),
            "elapsed_seconds": float(self.elapsed_seconds),
        }
        for idx, token in enumerate(self.low_tokens):
            row[f"low_{idx}"] = int(token)
        for idx, token in enumerate(self.high_tokens):
            row[f"high_{idx}"] = int(token)
        return row


class DMultTruthTableEvaluator:
    def __init__(self, checkpoint_path: Path = DMULT_CHECKPOINT, *, device: str | torch.device | None = None) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.num_slots = int(checkpoint.get("num_slots", checkpoint.get("config", {}).get("num_slots", 17)))
        model_kind = str(checkpoint.get("model_kind", checkpoint.get("config", {}).get("model_kind", "pairwise")))
        if model_kind != "neural_logic_lut":
            raise ValueError(f"Unsupported D_mult checkpoint kind: {model_kind!r}")
        cfg = checkpoint.get("config", {})
        self.model = NeuralLogicTruthDecoder(
            num_slots=self.num_slots,
            lut_init=str(cfg.get("lut_init", "known")),
            trainable_lut=bool(cfg.get("trainable_compressor_lut", False)),
            lut_logit_scale=float(cfg.get("lut_logit_scale", 30.0)),
        ).to(device=self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=True)
        self.model.eval()
        exact = torch.from_numpy(exact_table_np().astype(np.float32)).to(device=self.device)
        self.exact = exact
        self.exact_batched = exact.unsqueeze(0)
        self.s8s8_exact = s8s8_exact_table_np().astype(np.int32, copy=False)

    def predict_deltas(self, configs: list[CandidateConfig]) -> np.ndarray:
        if not configs:
            return np.zeros((0, 256, 256), dtype=np.int32)
        slot_tokens = torch.tensor(
            [config.slot_tokens() for config in configs],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            decoder_delta = self.model(slot_tokens)["delta"]
            truth_u16 = torch.clamp(
                torch.round(decoder_delta + self.exact_batched),
                min=0.0,
                max=float(UINT16_MAX),
            )
        unsigned_truth = truth_u16.detach().cpu().numpy().astype(np.uint16, copy=False)
        signed_truth = u8_lut_to_s8s8_sign_magnitude(unsigned_truth).astype(
            np.int32,
            copy=False,
        )
        return np.ascontiguousarray(
            signed_truth - self.s8s8_exact[None, :, :],
            dtype=np.int32,
        )

    def predict_max_abs_residuals(self, configs: list[CandidateConfig]) -> np.ndarray:
        """Return exact S8S8 residual maxima without materializing signed LUTs.

        The sign-magnitude wrapper changes only the sign of each nonzero
        magnitude pair. Therefore its absolute residual is the unsigned LUT
        error over magnitudes 1..128; zero operands always have zero residual.
        """
        if not configs:
            return np.zeros(0, dtype=np.int32)
        maxima: list[np.ndarray] = []
        for start in range(0, len(configs), 128):
            batch = configs[start : start + 128]
            slot_tokens = torch.tensor(
                [config.slot_tokens() for config in batch],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                decoder_delta = self.model(slot_tokens)["delta"]
                truth_u16 = torch.clamp(
                    torch.round(decoder_delta + self.exact_batched),
                    min=0.0,
                    max=float(UINT16_MAX),
                )
                unsigned_residual = (
                    truth_u16[:, 1:129, 1:129]
                    - self.exact_batched[:, 1:129, 1:129]
                )
                maximum = torch.amax(torch.abs(unsigned_residual), dim=(1, 2))
            maxima.append(maximum.to(dtype=torch.int32).cpu().numpy())
        if self.device.type == "cuda":
            # A separate worker runs the expensive runtime on this same GPU.
            # Return temporary decoder buffers to it after each preflight.
            torch.cuda.empty_cache()
        return np.concatenate(maxima)

    def predict(self, config: CandidateConfig) -> dict[str, Any]:
        slot_tokens = torch.tensor([config.slot_tokens()], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model(slot_tokens)
            decoder_delta = out["delta"]
            pred_truth = torch.clamp(
                torch.round(decoder_delta + self.exact_batched),
                min=0.0,
                max=float(UINT16_MAX),
            )
        truth_u16 = pred_truth.detach().cpu().numpy().astype(np.uint16, copy=False)[0]
        truth_s8s8 = np.asarray(
            u8_lut_to_s8s8_sign_magnitude(truth_u16),
            dtype=np.int32,
        )
        runtime_delta = np.ascontiguousarray(
            truth_s8s8 - self.s8s8_exact,
            dtype=np.int32,
        )
        metric12_np = np.asarray(
            s8s8_metric12_from_lut(truth_s8s8),
            dtype=np.float32,
        )
        abs_truth_error = np.abs(runtime_delta.astype(np.float32, copy=False))
        return {
            "truth_u16": truth_u16,
            "truth_s8s8_i32": truth_s8s8,
            "delta": runtime_delta,
            "metric12": metric12_np,
            "truth_mae": float(abs_truth_error.mean()),
            "truth_max_abs": float(abs_truth_error.max(initial=0.0)),
        }


class PdpEvaluator:
    def __init__(self, *, device: str | torch.device | None = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, config: CandidateConfig) -> dict[str, float]:
        graph = state_to_graph(config.ppa_state(), self.device)
        reward = predict_metrics(graph)
        delay_ns = denormalize(reward[2], "Delay", PPA_MIN_MAX_8BIT)
        area_um2 = denormalize(reward[3], "Area", PPA_MIN_MAX_8BIT)
        power_uw = denormalize(reward[4], "Power", PPA_MIN_MAX_8BIT)
        if not np.isfinite(area_um2) or area_um2 <= 0:
            raise ValueError(f"GNN predicted invalid area_um2: {area_um2}")
        return {
            "delay_ns": float(delay_ns),
            "power_uw": float(power_uw),
            "pdp_fj": float(delay_ns * power_uw),
            "area_um2": float(area_um2),
        }


class OptimizationEvaluator:
    def __init__(
        self,
        *,
        truth_table_evaluator: DMultTruthTableEvaluator | None = None,
        pdp_evaluator: PdpEvaluator | None = None,
        strict50_evaluator: Strict50RuntimeEvaluator | None = None,
        true1000_evaluator: Any | None = None,
        top1_eval_method: str = "strict50_v3",
        cache_csv: Path | None = None,
        device: str | torch.device | None = None,
        strict50_gpu_id: int | None = None,
        true1000_model_key: str = DEFAULT_TRUE1000_MODEL_KEY,
        true1000_sample_count: int = DEFAULT_TRUE1000_SAMPLE_COUNT,
        true1000_sample_seed: int = DEFAULT_TRUE1000_SAMPLE_SEED,
        true1000_batch_size: int = DEFAULT_TRUE1000_BATCH_SIZE,
        true1000_sessions_per_gpu: int = DEFAULT_TRUE1000_SESSIONS_PER_GPU,
        true1000_session_profile: str = DEFAULT_TRUE1000_SESSION_PROFILE,
        true1000_baseline_cache_dir: Path | None = None,
        true1000_archive_baseline_dir: Path | None = None,
        true1000_batch_fallback_parallelism: int = 1,
        initialize_models: bool = True,
        ) -> None:
        self.top1_eval_method = str(top1_eval_method)
        self.strict50_gpu_id = cuda_gpu_id_from_device(device) if strict50_gpu_id is None else int(strict50_gpu_id)
        self.truth_table_evaluator = truth_table_evaluator
        self.pdp_evaluator = pdp_evaluator
        self.strict50_evaluator = strict50_evaluator
        self.true1000_evaluator = true1000_evaluator
        self.true1000_model_key = str(true1000_model_key)
        self.true1000_sample_count = int(true1000_sample_count)
        self.true1000_sample_seed = int(true1000_sample_seed)
        self.true1000_batch_size = int(true1000_batch_size)
        self.true1000_sessions_per_gpu = int(true1000_sessions_per_gpu)
        self.true1000_session_profile = str(true1000_session_profile).strip().lower()
        if self.true1000_session_profile not in {"strict", "throughput"}:
            raise ValueError(
                "true1000_session_profile must be one of {'strict', 'throughput'}, "
                f"got {true1000_session_profile!r}"
            )
        self.true1000_baseline_cache_dir = Path(true1000_baseline_cache_dir) if true1000_baseline_cache_dir is not None else None
        self.true1000_archive_baseline_dir = Path(true1000_archive_baseline_dir) if true1000_archive_baseline_dir is not None else None
        self.true1000_batch_fallback_parallelism = int(true1000_batch_fallback_parallelism)
        self.models_initialized = bool(initialize_models)
        if self.top1_eval_method not in {"strict50_v3", TRUE1000_TOP1_METHOD}:
            raise ValueError(f"unsupported top1_eval_method: {self.top1_eval_method}")
        if self.models_initialized:
            self.truth_table_evaluator = self.truth_table_evaluator or DMultTruthTableEvaluator(device=device)
            self.pdp_evaluator = self.pdp_evaluator or PdpEvaluator(device=device)
            if self.top1_eval_method == "strict50_v3":
                self.strict50_evaluator = self.strict50_evaluator or Strict50RuntimeEvaluator(gpu_id=self.strict50_gpu_id)
            elif self.top1_eval_method == TRUE1000_TOP1_METHOD:
                if self.true1000_evaluator is None:
                    from aaae.evaluation.verification.true1000_runtime import (
                        True1000RuntimeEvaluator,
                    )

                    self.true1000_evaluator = True1000RuntimeEvaluator(
                        model_key=self.true1000_model_key,
                        sample_count=self.true1000_sample_count,
                        sample_seed=self.true1000_sample_seed,
                        batch_size=self.true1000_batch_size,
                        gpu_id=self.strict50_gpu_id,
                        sessions_per_gpu=self.true1000_sessions_per_gpu,
                        session_profile=self.true1000_session_profile,
                        baseline_cache_dir=self.true1000_baseline_cache_dir,
                        archive_baseline_dir=self.true1000_archive_baseline_dir,
                        batch_fallback_parallelism=self.true1000_batch_fallback_parallelism,
                    )
        self.cache: dict[str, EvaluationResult] = {}
        self.cache_csv = Path(cache_csv) if cache_csv is not None else None
        self.cache_load_report: dict[str, Any] = {
            "source_rows": 0,
            "loaded_rows": 0,
            "rejected_rows": 0,
            "rejection_counts": {},
        }
        self._load_cache_csv()

    @staticmethod
    def result_from_row(row: pd.Series | dict[str, Any]) -> EvaluationResult:
        low_tokens = tuple(int(row[f"low_{idx}"]) for idx in range(LOW_TOKEN_COUNT))
        high_tokens = tuple(int(row[f"high_{idx}"]) for idx in range(len(FIXED_HIGH_TOKENS)))
        return EvaluationResult(
            config_hash=str(row["config_hash"]),
            low_tokens=low_tokens,
            high_tokens=high_tokens,
            approx_high_part=_as_bool(row.get("approx_high_part", False)),
            design_index=int(row.get("design_index", 0)),
            delay_ns=float(row["delay_ns"]),
            power_uw=float(row["power_uw"]),
            pdp_fj=float(row["pdp_fj"]),
            area_um2=float(row.get("area_um2", float("nan"))),
            top1_drop=float(row["top1_drop"]),
            top5_drop=float(row.get("top5_drop", float("nan"))),
            final_truth_mae=float(row.get("final_truth_mae", float("nan"))),
            final_truth_max_abs=float(row.get("final_truth_max_abs", float("nan"))),
            top1_eval_method=str(row.get("top1_eval_method", "")),
            model_package_id=str(row.get("model_package_id", "")),
            selected_images_hash=str(row.get("selected_images_hash", "")),
            feature_schema_hash=str(row.get("feature_schema_hash", "")),
            strict50_feature_count=int(row.get("strict50_feature_count", 0)),
            strict50_selected_image_count=int(row.get("strict50_selected_image_count", 0)),
            sample50_top1_drop=float(row.get("sample50_top1_drop", float("nan"))),
            sample50_top5_drop=float(row.get("sample50_top5_drop", float("nan"))),
            package_hash=str(row.get("package_hash", "")),
            runtime_base_feature_count=int(row.get("runtime_base_feature_count", 0)),
            supplemental_feature_count=int(row.get("supplemental_feature_count", 0)),
            dataset_id=str(row.get("dataset_id", "")),
            injection_scope=str(row.get("injection_scope", "")),
            bank_meta_sha1=str(row.get("bank_meta_sha1", "")),
            ensemble_disagreement=float(row.get("ensemble_disagreement", float("nan"))),
            conformal_radius_95=float(row.get("conformal_radius_95", float("nan"))),
            conformal_lower_95=float(row.get("conformal_lower_95", float("nan"))),
            conformal_upper_95=float(row.get("conformal_upper_95", float("nan"))),
            exact_top1_acc=float(row.get("exact_top1_acc", float("nan"))),
            approx_top1_acc=float(row.get("approx_top1_acc", float("nan"))),
            exact_top5_acc=float(row.get("exact_top5_acc", float("nan"))),
            approx_top5_acc=float(row.get("approx_top5_acc", float("nan"))),
            top1_consistency=float(row.get("top1_consistency", float("nan"))),
            top5_consistency=float(row.get("top5_consistency", float("nan"))),
            logit_kl=float(row.get("logit_kl", float("nan"))),
            logit_cosine=float(row.get("logit_cosine", float("nan"))),
            output_mse=float(row.get("output_mse", float("nan"))),
            output_cosine=float(row.get("output_cosine", float("nan"))),
            runtime_delta_source=str(row.get("runtime_delta_source", "")),
            model_key=str(row.get("model_key", "")),
            sample_count=int(row.get("sample_count", 0)),
            sample_seed=int(row.get("sample_seed", 0)),
            batch_size=int(row.get("batch_size", 0)),
            gpu_id=int(row.get("gpu_id", 0)),
            sessions_per_gpu=int(row.get("sessions_per_gpu", 0)),
            session_profile=str(row.get("session_profile", "")),
            selected_node_count=int(row.get("selected_node_count", 0)),
            node_registry_hash=str(row.get("node_registry_hash", "")),
            baseline_logits_hash=str(row.get("baseline_logits_hash", "")),
            elapsed_seconds=float(row.get("elapsed_seconds", float("nan"))),
        )

    def _cache_incompatibility_reason(self, result: EvaluationResult) -> str | None:
        if not np.isfinite(result.area_um2) or result.area_um2 <= 0:
            return "area_um2_missing_or_invalid"
        if result.top1_eval_method != self.top1_eval_method:
            return "top1_eval_method_mismatch"
        if self.top1_eval_method == "strict50_v3":
            expected = {
                "model_package_id": STRICT50_EXPECTED_PACKAGE_ID,
                "package_hash": STRICT50_EXPECTED_PACKAGE_HASH,
                "selected_images_hash": STRICT50_EXPECTED_SELECTED_IMAGES_HASH,
                "feature_schema_hash": STRICT50_EXPECTED_FEATURE_SCHEMA_HASH,
                "strict50_feature_count": STRICT50_FEATURE_DIM,
                "strict50_selected_image_count": 50,
                "runtime_base_feature_count": STRICT50_RUNTIME_BASE_FEATURE_DIM,
                "supplemental_feature_count": STRICT50_SUPPLEMENTAL_FEATURE_DIM,
                "model_key": STRICT50_MODEL_KEY,
                "sample_count": 500,
                "sample_seed": 7,
                "selected_node_count": STRICT50_EXPECTED_NODE_COUNT,
                "node_registry_hash": STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
                "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
                "dataset_id": STRICT50_DATASET_ID,
                "injection_scope": STRICT50_INJECTION_SCOPE,
                "bank_meta_sha1": STRICT50_EXPECTED_BANK_META_SHA1,
            }
            for field, expected_value in expected.items():
                if getattr(result, field) != expected_value:
                    return f"{field}_mismatch"
        if self.top1_eval_method == TRUE1000_TOP1_METHOD:
            if result.model_key != self.true1000_model_key:
                return "true1000_model_key_mismatch"
            if int(result.sample_count) != self.true1000_sample_count:
                return "true1000_sample_count_mismatch"
            if int(result.sample_seed) != self.true1000_sample_seed:
                return "true1000_sample_seed_mismatch"
            if int(result.selected_node_count) != STRICT50_EXPECTED_NODE_COUNT:
                return "true1000_selected_node_count_mismatch"
            if result.node_registry_hash != STRICT50_EXPECTED_NODE_REGISTRY_SHA1:
                return "true1000_node_registry_hash_mismatch"
            if result.runtime_delta_source != STRICT50_RUNTIME_DELTA_SOURCE:
                return "true1000_runtime_delta_source_mismatch"
            if result.session_profile != self.true1000_session_profile:
                return "true1000_session_profile_mismatch"
        return None

    def _load_cache_csv(self) -> None:
        if self.cache_csv is None or not self.cache_csv.exists():
            return
        frame = pd.read_csv(self.cache_csv, low_memory=False)
        if frame.empty or "config_hash" not in frame.columns:
            return
        self.cache_load_report["source_rows"] = int(len(frame))
        if "area_um2" not in frame.columns:
            self.cache_load_report = {
                "source_rows": int(len(frame)),
                "loaded_rows": 0,
                "rejected_rows": int(len(frame)),
                "rejection_counts": {"area_um2_missing_or_invalid": int(len(frame))},
            }
            return
        rejection_counts: dict[str, int] = {}
        for row in frame.to_dict(orient="records"):
            try:
                result = self.result_from_row(row)
                reason = self._cache_incompatibility_reason(result)
                if reason is not None:
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                    continue
                self.cache[result.config_hash] = result
            except Exception:
                rejection_counts["invalid_cache_row"] = rejection_counts.get("invalid_cache_row", 0) + 1
        self.cache_load_report = {
            "source_rows": int(len(frame)),
            "loaded_rows": int(len(self.cache)),
            "rejected_rows": int(sum(rejection_counts.values())),
            "rejection_counts": rejection_counts,
        }

    def _save_cache_csv(self) -> None:
        if self.cache_csv is None:
            return
        frame = self.cache_frame()
        self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.cache_csv, index=False)

    def save_cache_csv(self) -> None:
        self._save_cache_csv()

    def add_result(self, result: EvaluationResult, *, save: bool = True) -> None:
        reason = self._cache_incompatibility_reason(result)
        if reason is not None:
            raise ValueError(f"cannot add incompatible optimization result: {reason}")
        self.cache[str(result.config_hash)] = result
        if save:
            self._save_cache_csv()

    def add_result_row(self, row: pd.Series | dict[str, Any], *, save: bool = True) -> None:
        self.add_result(self.result_from_row(row), save=save)

    def evaluate(self, config: CandidateConfig, *, save: bool = True) -> EvaluationResult:
        key = config.config_hash()
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.top1_eval_method == TRUE1000_TOP1_METHOD:
            return self.evaluate_configs([config])[0]
        if not self.models_initialized:
            raise RuntimeError("OptimizationEvaluator was initialized without models and cannot evaluate new candidates")
        if self.truth_table_evaluator is None or self.pdp_evaluator is None:
            raise RuntimeError("OptimizationEvaluator model components are not initialized")

        truth = self.truth_table_evaluator.predict(config)
        pdp = self.pdp_evaluator.predict(config)
        metadata = config.metadata_frame()
        extra_metrics: dict[str, Any] = {}
        if self.top1_eval_method == "strict50_v3":
            if self.strict50_evaluator is None:
                raise RuntimeError("strict50 evaluator is not initialized")
            strict50 = self.strict50_evaluator.evaluate(
                config_hash=key,
                metadata=metadata,
                delta=truth["delta"],
            )
            top1_drop = float(strict50.top1_drop)
            top5_drop = float(strict50.top5_sample_drop)
            model_package_id = strict50.model_package_id
            selected_images_hash = strict50.selected_images_hash
            feature_schema_hash = strict50.feature_schema_hash
            strict50_feature_count = int(strict50.feature_count)
            strict50_selected_image_count = int(strict50.selected_image_count)
            sample50_top1_drop = float(strict50.sample50_top1_drop)
            sample50_top5_drop = float(strict50.sample50_top5_drop)
            extra_metrics = {
                "package_hash": strict50.package_hash,
                "runtime_base_feature_count": int(strict50.runtime_base_feature_count),
                "supplemental_feature_count": int(strict50.supplemental_feature_count),
                "dataset_id": strict50.dataset_id,
                "injection_scope": strict50.injection_scope,
                "bank_meta_sha1": strict50.bank_meta_sha1,
                "ensemble_disagreement": float(strict50.ensemble_disagreement),
                "conformal_radius_95": float(strict50.conformal_radius_95),
                "conformal_lower_95": float(strict50.conformal_lower_95),
                "conformal_upper_95": float(strict50.conformal_upper_95),
                "runtime_delta_source": strict50.runtime_delta_source,
                "model_key": strict50.model_key,
                "sample_count": int(strict50.sample_count),
                "sample_seed": int(strict50.sample_seed),
                "batch_size": int(self.strict50_evaluator.batch_size),
                "gpu_id": int(self.strict50_evaluator.gpu_id),
                "sessions_per_gpu": int(self.strict50_evaluator.sessions_per_gpu),
                "selected_node_count": int(strict50.selected_node_count),
                "node_registry_hash": strict50.node_registry_hash,
            }
        elif self.top1_eval_method == TRUE1000_TOP1_METHOD:
            if self.true1000_evaluator is None:
                raise RuntimeError("true1000 evaluator is not initialized")
            runtime_metrics = self.true1000_evaluator.evaluate_one(
                config_hash=key,
                runtime_delta_i32=truth["delta"],
            )
            top1_drop = float(runtime_metrics["top1_drop"])
            top5_drop = float(runtime_metrics.get("top5_drop", float("nan")))
            model_package_id = str(runtime_metrics.get("model_key", self.true1000_model_key))
            selected_images_hash = ""
            feature_schema_hash = ""
            strict50_feature_count = 0
            strict50_selected_image_count = 0
            sample50_top1_drop = float("nan")
            sample50_top5_drop = float("nan")
            extra_metrics = {
                "exact_top1_acc": float(runtime_metrics.get("exact_top1_acc", float("nan"))),
                "approx_top1_acc": float(runtime_metrics.get("approx_top1_acc", float("nan"))),
                "exact_top5_acc": float(runtime_metrics.get("exact_top5_acc", float("nan"))),
                "approx_top5_acc": float(runtime_metrics.get("approx_top5_acc", float("nan"))),
                "top1_consistency": float(runtime_metrics.get("top1_consistency", float("nan"))),
                "top5_consistency": float(runtime_metrics.get("top5_consistency", float("nan"))),
                "logit_kl": float(runtime_metrics.get("logit_kl", float("nan"))),
                "logit_cosine": float(runtime_metrics.get("logit_cosine", float("nan"))),
                "output_mse": float(runtime_metrics.get("output_mse", float("nan"))),
                "output_cosine": float(runtime_metrics.get("output_cosine", float("nan"))),
                "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
                "model_key": str(runtime_metrics.get("model_key", self.true1000_model_key)),
                "sample_count": int(runtime_metrics.get("sample_count", self.true1000_sample_count)),
                "sample_seed": int(runtime_metrics.get("sample_seed", self.true1000_sample_seed)),
                "batch_size": int(runtime_metrics.get("batch_size", self.true1000_batch_size)),
                "gpu_id": int(runtime_metrics.get("gpu_id", self.strict50_gpu_id)),
                "sessions_per_gpu": int(runtime_metrics.get("sessions_per_gpu", self.true1000_sessions_per_gpu)),
                "session_profile": str(
                    runtime_metrics.get("session_profile", self.true1000_session_profile)
                ),
                "selected_node_count": int(runtime_metrics.get("selected_node_count", 0)),
                "node_registry_hash": str(runtime_metrics.get("node_registry_hash", "")),
                "baseline_logits_hash": str(runtime_metrics.get("baseline_logits_hash", "")),
                "elapsed_seconds": float(runtime_metrics.get("elapsed_seconds", float("nan"))),
            }
        else:
            raise ValueError(f"unsupported top1_eval_method: {self.top1_eval_method}")
        result = EvaluationResult(
            config_hash=key,
            low_tokens=config.low_tokens,
            high_tokens=config.high_tokens,
            approx_high_part=config.approx_high_part,
            design_index=config.design_index(),
            delay_ns=float(pdp["delay_ns"]),
            power_uw=float(pdp["power_uw"]),
            pdp_fj=float(pdp["pdp_fj"]),
            area_um2=float(pdp["area_um2"]),
            top1_drop=top1_drop,
            top5_drop=top5_drop,
            final_truth_mae=float(truth["truth_mae"]),
            final_truth_max_abs=float(truth["truth_max_abs"]),
            top1_eval_method=self.top1_eval_method,
            model_package_id=model_package_id,
            selected_images_hash=selected_images_hash,
            feature_schema_hash=feature_schema_hash,
            strict50_feature_count=strict50_feature_count,
            strict50_selected_image_count=strict50_selected_image_count,
            sample50_top1_drop=sample50_top1_drop,
            sample50_top5_drop=sample50_top5_drop,
            **extra_metrics,
        )
        self.cache[key] = result
        if save:
            self._save_cache_csv()
        return result

    def evaluate_configs(
        self,
        configs: list[CandidateConfig],
        *,
        config_batch_size: int = DEFAULT_TRUE1000_CONFIG_BATCH_SIZE,
    ) -> list[EvaluationResult]:
        if not configs:
            return []
        config_hashes = [config.config_hash() for config in configs]
        if len(set(config_hashes)) != len(config_hashes):
            raise ValueError("evaluate_configs requires unique config hashes")
        if int(config_batch_size) <= 0:
            raise ValueError("config_batch_size must be positive")
        if self.top1_eval_method != TRUE1000_TOP1_METHOD:
            resolved: list[EvaluationResult] = []
            cached_before = len(self.cache)
            try:
                for start in range(0, len(configs), int(config_batch_size)):
                    chunk = configs[start : start + int(config_batch_size)]
                    resolved.extend(self.evaluate(config, save=False) for config in chunk)
            finally:
                if len(self.cache) > cached_before:
                    self._save_cache_csv()
            return resolved
        if not self.models_initialized:
            raise RuntimeError("OptimizationEvaluator was initialized without models and cannot evaluate new candidates")
        if self.truth_table_evaluator is None or self.pdp_evaluator is None:
            raise RuntimeError("OptimizationEvaluator model components are not initialized")
        if self.true1000_evaluator is None:
            raise RuntimeError("true1000 evaluator is not initialized")

        resolved: dict[str, EvaluationResult] = {
            config_hash: self.cache[config_hash]
            for config_hash in config_hashes
            if config_hash in self.cache
        }
        pending = [config for config in configs if config.config_hash() not in resolved]
        for start in range(0, len(pending), int(config_batch_size)):
            chunk = pending[start : start + int(config_batch_size)]
            truth_rows = [self.truth_table_evaluator.predict(config) for config in chunk]
            pdp_rows = [self.pdp_evaluator.predict(config) for config in chunk]
            runtime_metrics = self.true1000_evaluator.evaluate_many(
                {
                    config.config_hash(): truth["delta"]
                    for config, truth in zip(chunk, truth_rows, strict=True)
                }
            )
            expected = {config.config_hash() for config in chunk}
            if set(runtime_metrics) != expected:
                raise RuntimeError(
                    "true1000 runtime returned a different config set: "
                    f"missing={sorted(expected.difference(runtime_metrics))[:10]} "
                    f"unexpected={sorted(set(runtime_metrics).difference(expected))[:10]}"
                )
            for config, truth, pdp in zip(chunk, truth_rows, pdp_rows, strict=True):
                key = config.config_hash()
                metrics = runtime_metrics[key]
                result = EvaluationResult(
                    config_hash=key,
                    low_tokens=config.low_tokens,
                    high_tokens=config.high_tokens,
                    approx_high_part=config.approx_high_part,
                    design_index=config.design_index(),
                    delay_ns=float(pdp["delay_ns"]),
                    power_uw=float(pdp["power_uw"]),
                    pdp_fj=float(pdp["pdp_fj"]),
                    area_um2=float(pdp["area_um2"]),
                    top1_drop=float(metrics["top1_drop"]),
                    top5_drop=float(metrics.get("top5_drop", float("nan"))),
                    final_truth_mae=float(truth["truth_mae"]),
                    final_truth_max_abs=float(truth["truth_max_abs"]),
                    top1_eval_method=self.top1_eval_method,
                    model_package_id=str(metrics.get("model_key", self.true1000_model_key)),
                    selected_images_hash="",
                    feature_schema_hash="",
                    strict50_feature_count=0,
                    strict50_selected_image_count=0,
                    sample50_top1_drop=float("nan"),
                    sample50_top5_drop=float("nan"),
                    exact_top1_acc=float(metrics.get("exact_top1_acc", float("nan"))),
                    approx_top1_acc=float(metrics.get("approx_top1_acc", float("nan"))),
                    exact_top5_acc=float(metrics.get("exact_top5_acc", float("nan"))),
                    approx_top5_acc=float(metrics.get("approx_top5_acc", float("nan"))),
                    top1_consistency=float(metrics.get("top1_consistency", float("nan"))),
                    top5_consistency=float(metrics.get("top5_consistency", float("nan"))),
                    logit_kl=float(metrics.get("logit_kl", float("nan"))),
                    logit_cosine=float(metrics.get("logit_cosine", float("nan"))),
                    output_mse=float(metrics.get("output_mse", float("nan"))),
                    output_cosine=float(metrics.get("output_cosine", float("nan"))),
                    runtime_delta_source=STRICT50_RUNTIME_DELTA_SOURCE,
                    model_key=str(metrics.get("model_key", self.true1000_model_key)),
                    sample_count=int(metrics.get("sample_count", self.true1000_sample_count)),
                    sample_seed=int(metrics.get("sample_seed", self.true1000_sample_seed)),
                    batch_size=int(metrics.get("batch_size", self.true1000_batch_size)),
                    gpu_id=int(metrics.get("gpu_id", self.strict50_gpu_id)),
                    sessions_per_gpu=int(
                        metrics.get("sessions_per_gpu", self.true1000_sessions_per_gpu)
                    ),
                    session_profile=str(
                        metrics.get("session_profile", self.true1000_session_profile)
                    ),
                    selected_node_count=int(metrics.get("selected_node_count", 0)),
                    node_registry_hash=str(metrics.get("node_registry_hash", "")),
                    baseline_logits_hash=str(metrics.get("baseline_logits_hash", "")),
                    elapsed_seconds=float(metrics.get("elapsed_seconds", float("nan"))),
                )
                self.cache[key] = result
                resolved[key] = result
        if pending:
            self._save_cache_csv()
        return [resolved[config_hash] for config_hash in config_hashes]

    def cache_frame(self) -> pd.DataFrame:
        if not self.cache:
            return pd.DataFrame()
        rows = [item.as_row() for item in self.cache.values()]
        frame = pd.DataFrame(rows)
        token_cols = [f"low_{idx}" for idx in range(LOW_TOKEN_COUNT)] + [f"high_{idx}" for idx in range(len(FIXED_HIGH_TOKENS))]
        ordered = [
            "config_hash",
            "approx_high_part",
            "design_index",
            *token_cols,
            "delay_ns",
            "power_uw",
            "pdp_fj",
            "area_um2",
            "top1_drop",
            "top5_drop",
            "final_truth_mae",
            "final_truth_max_abs",
            "top1_eval_method",
            "model_package_id",
            "package_hash",
            "selected_images_hash",
            "feature_schema_hash",
            "strict50_feature_count",
            "strict50_selected_image_count",
            "runtime_base_feature_count",
            "supplemental_feature_count",
            "dataset_id",
            "injection_scope",
            "bank_meta_sha1",
            "sample50_top1_drop",
            "sample50_top5_drop",
            "ensemble_disagreement",
            "conformal_radius_95",
            "conformal_lower_95",
            "conformal_upper_95",
            "exact_top1_acc",
            "approx_top1_acc",
            "exact_top5_acc",
            "approx_top5_acc",
            "top1_consistency",
            "top5_consistency",
            "logit_kl",
            "logit_cosine",
            "output_mse",
            "output_cosine",
            "runtime_delta_source",
            "model_key",
            "sample_count",
            "sample_seed",
            "batch_size",
            "gpu_id",
            "sessions_per_gpu",
            "session_profile",
            "selected_node_count",
            "node_registry_hash",
            "baseline_logits_hash",
            "elapsed_seconds",
        ]
        ordered = [column for column in ordered if column in frame.columns]
        return frame[ordered].sort_values(["pdp_fj", "top1_drop", "area_um2", "config_hash"]).reset_index(drop=True)

    def close(self) -> None:
        if self.strict50_evaluator is not None:
            self.strict50_evaluator.close()
        if self.true1000_evaluator is not None and hasattr(self.true1000_evaluator, "close"):
            self.true1000_evaluator.close()
