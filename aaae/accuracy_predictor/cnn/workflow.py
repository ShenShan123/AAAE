from __future__ import annotations

import copy
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .artifacts import (
    LoadedArtifact,
    ensure_output_available,
    load_artifact,
    publish_bundle,
)
from .contracts import MultiplierDesignBatch
from .errors import ArtifactError, UnqualifiedModelError
from .features import build_features
from .manifests import TARGETS, load_prediction_designs, load_training_inputs
from .modeling import (
    FittedTarget,
    RegressionMetrics,
    fit_target,
    predict_serialized_models,
)


@dataclass(frozen=True)
class PredictionTable:
    design_ids: tuple[str, ...]
    top1_drop: np.ndarray
    top5_drop: np.ndarray
    profile_digest: str
    synthetic_only: bool

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "design_id": self.design_ids,
                "predicted_top1_drop": self.top1_drop,
                "predicted_top5_drop": self.top5_drop,
            }
        )


@dataclass(frozen=True)
class TrainOutcome:
    bundle_path: Path
    metrics: Mapping[str, RegressionMetrics]
    split_counts: Mapping[str, int]
    claim_status: str
    performance_claim_allowed: bool
    quality_gate_passed: bool | None
    profile_digest: str


class CnnDropPredictor:
    """Profile-bound, read-only predictor loaded from a checked bundle."""

    def __init__(
        self,
        artifact: LoadedArtifact,
        *,
        torch_device: str = "cpu",
        torch_batch_size: int = 256,
    ) -> None:
        self._artifact = artifact
        self._delta_conv: Any | None = None
        self._correctness_conv: Any | None = None
        if artifact.delta_conv_paths:
            from .delta_conv import DeltaConvInference

            try:
                self._delta_conv = DeltaConvInference(
                    artifact.delta_conv_paths,
                    feature_count=len(artifact.feature_names),
                    feature_schema=str(artifact.manifest["feature_schema"]),
                    device=torch_device,
                    batch_size=torch_batch_size,
                )
            except (RuntimeError, ValueError) as exc:
                raise ArtifactError(
                    f"cannot initialize delta-CNN inference: {exc}"
                ) from exc
        if artifact.correctness_conv_paths:
            from .correctness_conv import CorrectnessConvInference

            try:
                self._correctness_conv = CorrectnessConvInference(
                    artifact.correctness_conv_paths,
                    feature_count=len(artifact.feature_names),
                    feature_schema=str(artifact.manifest["feature_schema"]),
                    device=torch_device,
                    batch_size=torch_batch_size,
                )
            except (RuntimeError, ValueError) as exc:
                raise ArtifactError(
                    f"cannot initialize correctness-CNN inference: {exc}"
                ) from exc

    @property
    def bundle_path(self) -> Path:
        return self._artifact.path

    @property
    def manifest(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._artifact.manifest)

    def predict(self, designs: MultiplierDesignBatch) -> PredictionTable:
        table = build_features(
            self._artifact.profile,
            designs,
            feature_schema=str(self._artifact.manifest["feature_schema"]),
        )
        if table.profile_digest != self._artifact.manifest["profile_digest"]:
            raise ArtifactError(
                "prediction profile digest does not match the loaded bundle"
            )
        if table.feature_names != self._artifact.feature_names:
            raise ArtifactError(
                "prediction feature schema does not match the loaded bundle"
            )
        matrix = np.ascontiguousarray(table.values, dtype=np.float32)
        member_predictions: dict[str, np.ndarray] = {}
        if self._artifact.model_texts:
            try:
                member_predictions.update(
                    predict_serialized_models(
                        self._artifact.model_texts,
                        matrix,
                        feature_names=self._artifact.feature_names,
                    )
                )
            except RuntimeError as exc:
                raise ArtifactError(
                    f"clean-process LightGBM prediction failed: {exc}"
                ) from exc
        if self._delta_conv is not None:
            try:
                member_predictions.update(
                    self._delta_conv.predict(designs.delta, matrix)
                )
            except (RuntimeError, ValueError) as exc:
                raise ArtifactError(f"delta-CNN prediction failed: {exc}") from exc
        if self._correctness_conv is not None:
            try:
                member_predictions.update(
                    self._correctness_conv.predict(designs.delta, matrix)
                )
            except (RuntimeError, ValueError) as exc:
                raise ArtifactError(
                    f"correctness-CNN prediction failed: {exc}"
                ) from exc
        predicted: dict[str, np.ndarray] = {}
        for target, ensemble in self._artifact.target_ensembles.items():
            values = np.full(
                len(table.design_ids),
                ensemble.offset,
                dtype=np.float64,
            )
            for member in ensemble.members:
                values += member.weight * member_predictions[member.model_key]
            postprocessor = ensemble.postprocessor
            if postprocessor is not None:
                first_key, second_key = postprocessor.member_keys
                values += postprocessor.scale * np.abs(
                    member_predictions[first_key] - member_predictions[second_key]
                )
            predicted[target] = np.asarray(values, dtype=np.float32)
        return PredictionTable(
            design_ids=table.design_ids,
            top1_drop=predicted["top1_drop"],
            top5_drop=predicted["top5_drop"],
            profile_digest=table.profile_digest,
            synthetic_only=bool(self._artifact.manifest.get("synthetic_only", False)),
        )


def _claim_status(inputs: Any) -> tuple[str, bool]:
    measured = (
        inputs.profile.profile_evidence == "measured"
        and inputs.provenance.get("kind") == "measured"
    )
    return ("measured_holdout", True) if measured else ("synthetic_smoke_only", False)


def _quality_gate(
    fitted: Mapping[str, FittedTarget],
    *,
    performance_claim_allowed: bool,
    mae_max: float,
    max_abs_error_max: float,
) -> bool | None:
    if not performance_claim_allowed:
        return None
    return all(
        model.metrics.mae <= mae_max
        and model.metrics.max_abs_error <= max_abs_error_max
        for model in fitted.values()
    )


def train_from_manifest(path: str | Path) -> TrainOutcome:
    inputs = load_training_inputs(path)
    ensure_output_available(inputs.output)

    feature_table = build_features(
        inputs.profile,
        inputs.designs,
        chunk_designs=inputs.execution.chunk_designs,
        feature_schema=inputs.feature_schema,
    )
    fitted = {
        target: fit_target(
            feature_table.values,
            inputs.labels[target],
            inputs.splits,
            feature_names=feature_table.feature_names,
            execution=inputs.execution,
            training=inputs.training,
            target_seed_offset=index * 1000,
            ensemble=inputs.target_ensembles[target],
        )
        for index, target in enumerate(TARGETS)
    }
    claim_status, performance_claim_allowed = _claim_status(inputs)
    gate_passed = _quality_gate(
        fitted,
        performance_claim_allowed=performance_claim_allowed,
        mae_max=inputs.quality_gate.mae_max,
        max_abs_error_max=inputs.quality_gate.max_abs_error_max,
    )
    bundle_path = publish_bundle(
        output=inputs.output,
        profile=inputs.profile,
        profile_metadata=inputs.profile_metadata,
        profile_digest=feature_table.profile_digest,
        feature_names=feature_table.feature_names,
        feature_schema=feature_table.schema_name,
        fitted=fitted,
        split_counts=inputs.split_counts,
        claim_status=claim_status,
        performance_claim_allowed=performance_claim_allowed,
        quality_gate_passed=gate_passed,
        training_manifest=inputs.manifest,
        provenance=inputs.provenance,
        execution=asdict(inputs.execution),
        training=asdict(inputs.training),
    )
    return TrainOutcome(
        bundle_path=bundle_path,
        metrics=MappingProxyType(
            {target: fitted[target].metrics for target in TARGETS}
        ),
        split_counts=MappingProxyType(dict(inputs.split_counts)),
        claim_status=claim_status,
        performance_claim_allowed=performance_claim_allowed,
        quality_gate_passed=gate_passed,
        profile_digest=feature_table.profile_digest,
    )


def load_bundle(
    path: str | Path,
    *,
    allow_synthetic: bool = False,
    torch_device: str = "cpu",
    torch_batch_size: int = 256,
) -> CnnDropPredictor:
    artifact = load_artifact(path)
    if artifact.manifest.get("synthetic_only") and not allow_synthetic:
        raise UnqualifiedModelError(
            "bundle is marked synthetic_smoke_only; pass allow_synthetic=True only for "
            "software validation, never for a CNN performance claim"
        )
    return CnnDropPredictor(
        artifact,
        torch_device=torch_device,
        torch_batch_size=torch_batch_size,
    )


def _write_predictions_atomic(table: PredictionTable, output: Path) -> None:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", suffix=".csv", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_frame().to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def predict_from_manifest(
    bundle: str | Path,
    manifest: str | Path,
    *,
    allow_synthetic: bool = False,
    torch_device: str = "cpu",
    torch_batch_size: int = 256,
) -> PredictionTable:
    designs, output = load_prediction_designs(manifest)
    table = load_bundle(
        bundle,
        allow_synthetic=allow_synthetic,
        torch_device=torch_device,
        torch_batch_size=torch_batch_size,
    ).predict(designs)
    if output is not None:
        _write_predictions_atomic(table, output)
    return table
