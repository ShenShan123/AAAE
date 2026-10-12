from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import (
    AI_MODELS_ROOT,
    END_TO_END_ROOT,
    RUNTIME_PROJECT_ROOT,
    STRICT50_BANK_META,
    STRICT50_BANK_META_CONTRACT,
    STRICT50_CANONICAL_EXACT_BASELINE,
    STRICT50_DATASET_ID,
    STRICT50_DATASET_MANIFEST,
    STRICT50_DATASET_SCHEMA_VERSION,
    STRICT50_DYNAMIC_MODEL,
    STRICT50_EXPECTED_BANK_META_SHA1,
    STRICT50_EXPECTED_FEATURE_SCHEMA_HASH,
    STRICT50_EXPECTED_MODEL_SHA1,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_EXPECTED_PACKAGE_HASH,
    STRICT50_EXPECTED_PACKAGE_ID,
    STRICT50_EXPECTED_SELECTED_IMAGES_HASH,
    STRICT50_EXPECTED_STATIC_MODEL_SHA1,
    STRICT50_FEATURE_DIM,
    STRICT50_INJECTION_SCOPE,
    STRICT50_LUT_DOMAIN,
    STRICT50_MODEL_KEY,
    STRICT50_NODE_REGISTRY,
    STRICT50_RUNTIME_BASE_FEATURE_DIM,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_S8_RESIDUAL_ABS_BOUND,
    STRICT50_SHARED_LUT_INPUT_NAME,
    STRICT50_STATIC_MODEL,
    STRICT50_STRUCTURE_OPERAND_PROFILE,
    STRICT50_STRUCTURE_PROPAGATION_PROFILE,
    STRICT50_SUPPLEMENTAL_FEATURE_DIM,
    STRICT50_SUPPLEMENTAL_FEATURE_NAMES,
    STRICT50_TOP1_PACKAGE_DIR,
    STRICT50_TRUE1000_EXACT_META,
    STRICT50_UNION_INPUTS,
    STRICT50_UNION_LABELS,
    STRICT50_UNION_POSITIONS,
)
from aaae.accuracy_predictor.features.static.feature_builder import build_common3103_features


def _ensure_path(path: Path) -> None:
    resolved = str(Path(path).resolve())
    if resolved not in sys.path:
        pass  # Local package imports are resolved by the entry point.


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON document must contain an object: {path}")
    return payload


os.environ.setdefault("APPROX_AI_MODELS_ROOT", str(AI_MODELS_ROOT))
_ensure_path(RUNTIME_PROJECT_ROOT)
_ensure_path(END_TO_END_ROOT)

from aaae.accuracy_predictor.features.transformer.swin_linear48_candidate_features import (  # noqa: E402
    SwinLinear48CandidateFeatureEvaluator,
    s8s8_exact_product_table,
    validate_s8s8_sign_magnitude_residuals,
)
from aaae.accuracy_predictor.transformer.artifacts.strict50_package import load_strict50_top1_package  # noqa: E402
from aaae.accuracy_predictor.transformer.proxy_training.transformer_ame_bank import (  # noqa: E402
    compute_runtime_bank_from_logits,
    load_transformer_ame_bank_meta,
)
from aaae.inference_runtime.transformer.assignment_runtime import build_uniform_assignment_runtime_pool  # noqa: E402


@dataclass(frozen=True)
class Strict50Evaluation:
    top1_drop: float
    top5_sample_drop: float
    sample50_top1_drop: float
    sample50_top5_drop: float
    feature_count: int
    runtime_base_feature_count: int
    supplemental_feature_count: int
    selected_image_count: int
    model_package_id: str
    package_hash: str
    selected_images_hash: str
    feature_schema_hash: str
    dataset_id: str
    injection_scope: str
    model_key: str
    sample_count: int
    sample_seed: int
    selected_node_count: int
    node_registry_hash: str
    bank_meta_sha1: str
    runtime_delta_source: str
    ensemble_disagreement: float
    conformal_radius_95: float
    conformal_lower_95: float
    conformal_upper_95: float


@dataclass(frozen=True)
class Strict50RawResponse:
    approx_top1_correct_bits: np.ndarray
    approx_top5_correct_bits: np.ndarray
    delta_margin_top1: np.ndarray
    delta_margin_top5: np.ndarray
    sample50_top1_drop: float
    sample50_top5_drop: float


class Strict50RuntimeEvaluator:
    """Evaluate optimizer candidates with the canonical linear48 joint4497 path."""

    def __init__(
        self,
        *,
        package_dir: Path = STRICT50_TOP1_PACKAGE_DIR,
        bank_meta_path: Path = STRICT50_BANK_META,
        bank_meta_contract_path: Path = STRICT50_BANK_META_CONTRACT,
        dataset_manifest_path: Path = STRICT50_DATASET_MANIFEST,
        node_registry_path: Path = STRICT50_NODE_REGISTRY,
        dynamic_model_path: Path = STRICT50_DYNAMIC_MODEL,
        static_model_path: Path = STRICT50_STATIC_MODEL,
        union_inputs_path: Path = STRICT50_UNION_INPUTS,
        union_labels_path: Path = STRICT50_UNION_LABELS,
        union_positions_path: Path = STRICT50_UNION_POSITIONS,
        canonical_exact_baseline_path: Path = STRICT50_CANONICAL_EXACT_BASELINE,
        true1000_exact_meta_path: Path = STRICT50_TRUE1000_EXACT_META,
        operand_profile_path: Path = STRICT50_STRUCTURE_OPERAND_PROFILE,
        propagation_profile_path: Path = STRICT50_STRUCTURE_PROPAGATION_PROFILE,
        supplemental_feature_names_path: Path = STRICT50_SUPPLEMENTAL_FEATURE_NAMES,
        model_key: str = STRICT50_MODEL_KEY,
        sample_count: int = 500,
        sample_seed: int = 7,
        batch_size: int = 32,
        gpu_id: int = 0,
        sessions_per_gpu: int = 1,
        shared_lut_input_name: str = STRICT50_SHARED_LUT_INPUT_NAME,
        lut_domain: str = STRICT50_LUT_DOMAIN,
        s8_residual_abs_bound: int = STRICT50_S8_RESIDUAL_ABS_BOUND,
        structure_feature_device: str | None = None,
        verify_structure_profile_hashes: bool = False,
        build_runtime: bool = True,
    ) -> None:
        self.package_dir = Path(package_dir).resolve()
        self.bank_meta_path = Path(bank_meta_path).resolve()
        self.bank_meta_contract_path = Path(bank_meta_contract_path).resolve()
        self.dataset_manifest_path = Path(dataset_manifest_path).resolve()
        self.node_registry_path = Path(node_registry_path).resolve()
        self.dynamic_model_path = Path(dynamic_model_path).resolve()
        self.static_model_path = Path(static_model_path).resolve()
        self.union_inputs_path = Path(union_inputs_path).resolve()
        self.union_labels_path = Path(union_labels_path).resolve()
        self.union_positions_path = Path(union_positions_path).resolve()
        self.canonical_exact_baseline_path = Path(canonical_exact_baseline_path).resolve()
        self.true1000_exact_meta_path = Path(true1000_exact_meta_path).resolve()
        self.operand_profile_path = Path(operand_profile_path).resolve()
        self.propagation_profile_path = Path(propagation_profile_path).resolve()
        self.supplemental_feature_names_path = Path(supplemental_feature_names_path).resolve()
        self.model_key = str(model_key)
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.batch_size = max(1, int(batch_size))
        self.gpu_id = int(gpu_id)
        self.sessions_per_gpu = max(1, int(sessions_per_gpu))
        self.shared_lut_input_name = str(shared_lut_input_name)
        self.lut_domain = str(lut_domain)
        self.s8_residual_abs_bound = int(s8_residual_abs_bound)
        self.runtime_pool: Any | None = None

        if self.model_key != STRICT50_MODEL_KEY:
            raise ValueError(
                f"joint4497 package requires model_key={STRICT50_MODEL_KEY!r}, got {self.model_key!r}"
            )
        if self.sample_count != 500 or self.sample_seed != 7:
            raise ValueError("canonical strict50 runtime requires sample_count=500 and sample_seed=7")
        if self.shared_lut_input_name != STRICT50_SHARED_LUT_INPUT_NAME:
            raise ValueError(
                f"canonical runtime requires shared LUT input {STRICT50_SHARED_LUT_INPUT_NAME!r}"
            )
        if self.lut_domain != STRICT50_LUT_DOMAIN:
            raise ValueError(f"canonical runtime requires lut_domain={STRICT50_LUT_DOMAIN!r}")
        if self.s8_residual_abs_bound != STRICT50_S8_RESIDUAL_ABS_BOUND:
            raise ValueError(
                f"canonical runtime requires residual bound {STRICT50_S8_RESIDUAL_ABS_BOUND}"
            )

        required_files = (
            self.package_dir / "manifest.json",
            self.bank_meta_path,
            self.bank_meta_contract_path,
            self.dataset_manifest_path,
            self.node_registry_path,
            self.dynamic_model_path,
            self.static_model_path,
            self.union_inputs_path,
            self.union_labels_path,
            self.union_positions_path,
            self.canonical_exact_baseline_path,
            self.true1000_exact_meta_path,
            self.operand_profile_path,
            self.propagation_profile_path,
            self.supplemental_feature_names_path,
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"canonical strict50 assets are missing: {missing}")

        self.predictor = load_strict50_top1_package(self.package_dir)
        self.package_manifest = dict(self.predictor.manifest)
        self.dataset_manifest = _read_json(self.dataset_manifest_path)
        self.bank_meta_contract = _read_json(self.bank_meta_contract_path)
        self.node_registry = _read_json(self.node_registry_path)
        self.node_registry_hash = _sha1_file(self.node_registry_path)
        self.bank_meta_sha1 = _sha1_file(self.bank_meta_path)
        self.dynamic_model_sha1 = _sha1_file(self.dynamic_model_path)
        self.static_model_sha1 = _sha1_file(self.static_model_path)
        self._validate_identity_contract()

        self.selected_images = np.asarray(self.predictor.selected_images, dtype=np.int64)
        if self.selected_images.shape != (50,) or len(np.unique(self.selected_images)) != 50:
            raise ValueError("strict50 package must contain 50 unique selected image indices")
        if np.any(self.selected_images < 0) or np.any(self.selected_images >= self.sample_count):
            raise ValueError("strict50 selected image indices are outside the raw500 bank")

        self.bank_meta = load_transformer_ame_bank_meta(self.bank_meta_path)
        if int(self.bank_meta.sample_count) != self.sample_count:
            raise ValueError(
                f"bank meta sample_count={self.bank_meta.sample_count} does not match {self.sample_count}"
            )
        self.runtime_inputs = self._load_canonical_runtime_inputs()

        requested_structure_device = structure_feature_device
        if requested_structure_device is None:
            # Allow the launcher to keep the large profile contraction on CPU
            # when another process has left too little device memory for the
            # roughly 3 GiB operand-profile tensor. Numerical fidelity is
            # unchanged; only this auxiliary feature contraction moves.
            requested_structure_device = os.environ.get("EA2_STRUCTURE_FEATURE_DEVICE")
        if requested_structure_device is None:
            requested_structure_device = f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu"
        self.structure_feature_device = str(requested_structure_device)
        self.structure_feature_evaluator = (
            SwinLinear48CandidateFeatureEvaluator.from_profile_cache(
                operand_profile=self.operand_profile_path,
                propagation_profile=self.propagation_profile_path,
                exact_meta=self.true1000_exact_meta_path,
                feature_names_path=self.supplemental_feature_names_path,
                device=self.structure_feature_device,
                verify_profile_hashes=bool(verify_structure_profile_hashes),
            )
        )
        if list(self.structure_feature_evaluator.feature_names) != list(
            self.predictor.supplemental_feature_names
        ):
            raise ValueError("candidate structure feature names differ from the joint4497 package")

        selected_nodes = self.node_registry.get("nodes", [])
        self.selected_node_names = [str(row["node_id"]) for row in selected_nodes]
        self.selected_node_count = len(self.selected_node_names)
        if (
            self.selected_node_count != STRICT50_EXPECTED_NODE_COUNT
            or len(set(self.selected_node_names)) != self.selected_node_count
        ):
            raise ValueError("canonical node registry must contain 48 unique selected nodes")

        if build_runtime:
            if not torch.cuda.is_available():
                raise RuntimeError("canonical strict50 runtime requires CUDA")
            self.device_name = str(torch.cuda.get_device_name(self.gpu_id))
            self.runtime_pool = build_uniform_assignment_runtime_pool(
                self.dynamic_model_path,
                selected_node_names=self.selected_node_names,
                shared_lut_input_name=self.shared_lut_input_name,
                device_id=self.gpu_id,
                runtime_count=self.sessions_per_gpu,
                lut_domain=self.lut_domain,
                s8_residual_abs_bound=self.s8_residual_abs_bound,
                static_shape_reference_path=self.static_model_path,
            )
        else:
            self.device_name = "not_initialized"

    def _validate_identity_contract(self) -> None:
        package_expected = {
            "package_id": STRICT50_EXPECTED_PACKAGE_ID,
            "package_hash": STRICT50_EXPECTED_PACKAGE_HASH,
            "model_key": STRICT50_MODEL_KEY,
            "dataset_id": STRICT50_DATASET_ID,
            "injection_scope": STRICT50_INJECTION_SCOPE,
            "feature_count": STRICT50_FEATURE_DIM,
            "runtime_base_feature_count": STRICT50_RUNTIME_BASE_FEATURE_DIM,
            "selected_images_hash": STRICT50_EXPECTED_SELECTED_IMAGES_HASH,
            "feature_schema_hash": STRICT50_EXPECTED_FEATURE_SCHEMA_HASH,
        }
        for key, expected in package_expected.items():
            if self.package_manifest.get(key) != expected:
                raise ValueError(
                    f"strict50 package {key} mismatch: {self.package_manifest.get(key)!r} != {expected!r}"
                )
        if self.predictor.supplemental_feature_count != STRICT50_SUPPLEMENTAL_FEATURE_DIM:
            raise ValueError("strict50 package supplemental feature count is not 1034")

        dataset_expected = {
            "dataset_id": STRICT50_DATASET_ID,
            "schema_version": STRICT50_DATASET_SCHEMA_VERSION,
            "model_key": STRICT50_MODEL_KEY,
            "model_sha1": STRICT50_EXPECTED_MODEL_SHA1,
            "node_registry_sha1": STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
        }
        for key, expected in dataset_expected.items():
            if self.dataset_manifest.get(key) != expected:
                raise ValueError(
                    f"strict50 dataset {key} mismatch: {self.dataset_manifest.get(key)!r} != {expected!r}"
                )
        injection = self.dataset_manifest.get("injection_mode", {})
        if (
            injection.get("injection_scope_id") != STRICT50_INJECTION_SCOPE
            or int(injection.get("injected_matmul_count", -1)) != STRICT50_EXPECTED_NODE_COUNT
            or injection.get("kind") != "uniform_s8s8_sign_magnitude_residual"
        ):
            raise ValueError("strict50 dataset injection contract is not canonical linear48 S8S8")

        if self.dynamic_model_sha1 != STRICT50_EXPECTED_MODEL_SHA1:
            raise ValueError("dynamic ONNX hash differs from the canonical dataset")
        if self.static_model_sha1 != STRICT50_EXPECTED_STATIC_MODEL_SHA1:
            raise ValueError("static ONNX hash differs from the canonical shape reference")
        if self.node_registry_hash != STRICT50_EXPECTED_NODE_REGISTRY_SHA1:
            raise ValueError("node registry hash differs from the canonical linear48 registry")
        if self.bank_meta_sha1 != STRICT50_EXPECTED_BANK_META_SHA1:
            raise ValueError("strict50 bank meta hash differs from the canonical bank")
        if self.bank_meta_contract.get("bank_meta_sha1") != self.bank_meta_sha1:
            raise ValueError("strict50 bank meta contract does not bind the selected bank")
        if (
            self.bank_meta_contract.get("schema_version") != STRICT50_DATASET_SCHEMA_VERSION
            or int(self.bank_meta_contract.get("sample_count", -1)) != 500
        ):
            raise ValueError("strict50 bank meta contract is stale")
        if (
            self.node_registry.get("model_key") != STRICT50_MODEL_KEY
            or self.node_registry.get("injection_scope_id") != STRICT50_INJECTION_SCOPE
            or int(self.node_registry.get("node_count", -1)) != STRICT50_EXPECTED_NODE_COUNT
        ):
            raise ValueError("node registry metadata differs from the canonical linear48 contract")

    def _load_canonical_runtime_inputs(self) -> np.ndarray:
        union_inputs = np.load(self.union_inputs_path, mmap_mode="r", allow_pickle=False)
        union_labels = np.load(self.union_labels_path, mmap_mode="r", allow_pickle=False)
        with np.load(self.union_positions_path, allow_pickle=False) as positions:
            true1000_positions = np.asarray(positions["true1000_positions"], dtype=np.int64)
            raw500_positions = np.asarray(positions["raw500_positions"], dtype=np.int64)
        if (
            union_inputs.shape != (1486, 3, 224, 224)
            or union_inputs.dtype != np.float32
            or union_labels.shape != (1486,)
            or true1000_positions.shape != (1000,)
            or raw500_positions.shape != (500,)
        ):
            raise ValueError("canonical union input/position assets have unexpected shapes")
        if (
            len(np.unique(true1000_positions)) != 1000
            or len(np.unique(raw500_positions)) != 500
            or np.any(true1000_positions < 0)
            or np.any(raw500_positions < 0)
            or np.any(true1000_positions >= len(union_inputs))
            or np.any(raw500_positions >= len(union_inputs))
        ):
            raise ValueError("canonical union positions are invalid")

        with np.load(self.canonical_exact_baseline_path, allow_pickle=False) as exact:
            exact_logits = np.asarray(exact["logits"], dtype=np.float32)
            exact_labels = np.asarray(exact["union_labels"], dtype=np.int64)
            exact_true_positions = np.asarray(exact["true1000_positions"], dtype=np.int64)
            exact_raw_positions = np.asarray(exact["raw500_positions"], dtype=np.int64)
        if (
            exact_logits.shape != (len(union_inputs), 1000)
            or not np.array_equal(exact_labels, union_labels)
            or not np.array_equal(exact_true_positions, true1000_positions)
            or not np.array_equal(exact_raw_positions, raw500_positions)
        ):
            raise ValueError("canonical exact baseline is not aligned with union inputs")

        raw_labels = np.asarray(union_labels[raw500_positions], dtype=np.int64)
        raw_logits = np.asarray(exact_logits[raw500_positions], dtype=np.float32)
        exact_pred = np.argmax(raw_logits, axis=1).astype(np.int64, copy=False)
        top5 = np.argpartition(raw_logits, kth=raw_logits.shape[1] - 5, axis=1)[:, -5:]
        exact_top5_bits = np.any(top5 == raw_labels[:, None], axis=1).astype(np.uint8)
        if (
            not np.array_equal(raw_labels, self.bank_meta.true_label)
            or not np.array_equal(exact_pred, self.bank_meta.exact_pred_class)
            or not np.array_equal(
                (exact_pred == raw_labels).astype(np.uint8),
                self.bank_meta.exact_top1_correct_bits,
            )
            or not np.array_equal(exact_top5_bits, self.bank_meta.exact_top5_correct_bits)
        ):
            raise RuntimeError("canonical raw500 exact baseline does not match strict50 bank meta")

        selected_union_positions = raw500_positions[self.selected_images]
        return np.ascontiguousarray(
            union_inputs[selected_union_positions],
            dtype=np.float32,
        )

    @property
    def package_id(self) -> str:
        return self.predictor.package_id

    @property
    def package_hash(self) -> str:
        return str(self.package_manifest.get("package_hash", ""))

    @property
    def selected_images_hash(self) -> str:
        return str(self.package_manifest.get("selected_images_hash", ""))

    @property
    def feature_schema_hash(self) -> str:
        return str(self.package_manifest.get("feature_schema_hash", ""))

    def close(self) -> None:
        pool = getattr(self, "runtime_pool", None)
        if pool is not None:
            pool.close()
        self.runtime_pool = None
        if getattr(self, "structure_feature_evaluator", None) is not None:
            self.structure_feature_evaluator.close()
            self.structure_feature_evaluator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _validated_delta(self, delta: np.ndarray) -> np.ndarray:
        validated = validate_s8s8_sign_magnitude_residuals(delta)
        if validated.shape != (1, 256, 256):
            raise ValueError(f"one candidate delta is required, got {validated.shape}")
        if int(np.max(np.abs(validated.astype(np.int64)))) > self.s8_residual_abs_bound:
            raise ValueError(
                f"candidate residual exceeds canonical absolute bound {self.s8_residual_abs_bound}"
            )
        return np.ascontiguousarray(validated[0], dtype=np.int32)

    def runtime_responses_many(
        self,
        *,
        deltas_by_config_hash: dict[str, np.ndarray],
    ) -> dict[str, Strict50RawResponse]:
        if not deltas_by_config_hash:
            return {}
        if self.runtime_pool is None:
            raise RuntimeError("strict50 runtime pool is not initialized")
        normalized_deltas: dict[str, np.ndarray] = {}
        for config_hash, delta in deltas_by_config_hash.items():
            key = str(config_hash)
            if key in normalized_deltas:
                raise ValueError(f"duplicate strict50 config hash {key!r}")
            normalized_deltas[key] = self._validated_delta(delta)

        keys = list(normalized_deltas)
        logits_by_key = {
            key: np.zeros((len(self.runtime_inputs), 1000), dtype=np.float32)
            for key in keys
        }
        for start in range(0, len(self.runtime_inputs), self.batch_size):
            end = min(len(self.runtime_inputs), start + self.batch_size)
            batch_logits = self.runtime_pool.run_uniform_many(
                self.runtime_inputs[start:end],
                approx_deltas_by_key=normalized_deltas,
            )
            if set(batch_logits) != set(keys):
                raise RuntimeError("uniform strict50 runtime returned a different candidate key set")
            for key in keys:
                logits_by_key[key][start:end] = np.asarray(batch_logits[key], dtype=np.float32)

        selected = self.selected_images
        responses: dict[str, Strict50RawResponse] = {}
        for key in keys:
            bank = compute_runtime_bank_from_logits(
                logits_by_key[key],
                clean_margin_top1=self.bank_meta.clean_margin_top1[selected],
                clean_margin_top5=self.bank_meta.clean_margin_top5[selected],
                exact_pred_class=self.bank_meta.exact_pred_class[selected],
                true_label=self.bank_meta.true_label[selected],
                exact_top1_correct_bits=self.bank_meta.exact_top1_correct_bits[selected],
                exact_top5_correct_bits=self.bank_meta.exact_top5_correct_bits[selected],
            )
            responses[key] = Strict50RawResponse(
                approx_top1_correct_bits=np.asarray(
                    bank["approx_top1_correct_bits"], dtype=np.uint8
                ),
                approx_top5_correct_bits=np.asarray(
                    bank["approx_top5_correct_bits"], dtype=np.uint8
                ),
                delta_margin_top1=np.asarray(bank["delta_margin_top1"], dtype=np.float32),
                delta_margin_top5=np.asarray(bank["delta_margin_top5"], dtype=np.float32),
                sample50_top1_drop=float(bank["top1_drop"]),
                sample50_top5_drop=float(bank["top5_drop"]),
            )
        return responses

    def _runtime_response(self, *, config_hash: str, delta: np.ndarray) -> Strict50RawResponse:
        return self.runtime_responses_many(
            deltas_by_config_hash={str(config_hash): np.asarray(delta)},
        )[str(config_hash)]

    def evaluate(
        self,
        *,
        config_hash: str,
        metadata: Any,
        delta: np.ndarray,
    ) -> Strict50Evaluation:
        runtime_delta = self._validated_delta(delta)
        candidate_metadata = metadata.copy()
        candidate_metadata["design_index"] = 0
        common3103, _shape = build_common3103_features(
            metadata=candidate_metadata,
            delta_bank=runtime_delta[None, :, :],
            exact_table=s8s8_exact_product_table(),
        )
        supplemental, supplemental_names = (
            self.structure_feature_evaluator.features_from_signed_residuals(
                runtime_delta
            )
        )
        response = self._runtime_response(
            config_hash=config_hash,
            delta=runtime_delta,
        )
        selected = self.selected_images
        features = self.predictor.build_features_from_runtime_observations(
            common3103_features=common3103,
            image_indices=selected,
            clean_top1_correct_bits=self.bank_meta.exact_top1_correct_bits[selected],
            clean_top5_correct_bits=self.bank_meta.exact_top5_correct_bits[selected],
            approx_top1_correct_bits=response.approx_top1_correct_bits[None, :],
            approx_top5_correct_bits=response.approx_top5_correct_bits[None, :],
            delta_margin_top1=response.delta_margin_top1[None, :],
            delta_margin_top5=response.delta_margin_top5[None, :],
            supplemental_features=supplemental,
            supplemental_feature_names=supplemental_names,
            scope="deployment",
        )
        if features.shape != (1, STRICT50_FEATURE_DIM):
            raise ValueError(
                f"expected one {STRICT50_FEATURE_DIM}-wide package feature row, got {features.shape}"
            )
        prediction = self.predictor.predict_top1_with_uncertainty(
            features,
            scope="deployment",
        )
        pred_top1 = np.asarray(prediction["prediction"], dtype=np.float64)
        disagreement = np.asarray(prediction["ensemble_disagreement"], dtype=np.float64)
        lower = np.asarray(prediction["conformal_lower_95"], dtype=np.float64)
        upper = np.asarray(prediction["conformal_upper_95"], dtype=np.float64)
        return Strict50Evaluation(
            top1_drop=float(pred_top1[0]),
            top5_sample_drop=float(response.sample50_top5_drop),
            sample50_top1_drop=float(response.sample50_top1_drop),
            sample50_top5_drop=float(response.sample50_top5_drop),
            feature_count=int(features.shape[1]),
            runtime_base_feature_count=int(self.predictor.runtime_base_feature_count),
            supplemental_feature_count=int(self.predictor.supplemental_feature_count),
            selected_image_count=int(len(selected)),
            model_package_id=self.package_id,
            package_hash=self.package_hash,
            selected_images_hash=self.selected_images_hash,
            feature_schema_hash=self.feature_schema_hash,
            dataset_id=STRICT50_DATASET_ID,
            injection_scope=STRICT50_INJECTION_SCOPE,
            model_key=self.model_key,
            sample_count=self.sample_count,
            sample_seed=self.sample_seed,
            selected_node_count=self.selected_node_count,
            node_registry_hash=self.node_registry_hash,
            bank_meta_sha1=self.bank_meta_sha1,
            runtime_delta_source=STRICT50_RUNTIME_DELTA_SOURCE,
            ensemble_disagreement=float(disagreement[0]),
            conformal_radius_95=float(prediction["conformal_radius_95"]),
            conformal_lower_95=float(lower[0]),
            conformal_upper_95=float(upper[0]),
        )
