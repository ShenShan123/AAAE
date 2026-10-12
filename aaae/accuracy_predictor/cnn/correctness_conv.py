from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch
from torch import nn

from .delta_conv import (
    DEFAULT_MODEL_ARCHITECTURE,
    INPUT_LOG_SCALE,
    MAGNITUDE_ORDER,
    SUPPORTED_INFERENCE_DTYPES,
    SUPPORTED_MODEL_ARCHITECTURES,
    create_delta_conv_regressor,
)

CHECKPOINT_SCHEMA_VERSION = "cnn_correctness_conv_bundle_member_v1"
COMPOSITE_CHECKPOINT_SCHEMA_VERSION = "cnn_correctness_conv_composite_v1"
DEFAULT_CORRECTNESS_HEAD = "linear_v1"
DEEP_CORRECTNESS_HEAD = "deep_v2"
SUPPORTED_CORRECTNESS_HEADS = frozenset(
    {DEFAULT_CORRECTNESS_HEAD, DEEP_CORRECTNESS_HEAD}
)


class CorrectnessConvRegressor(nn.Module):
    def __init__(
        self,
        feature_count: int,
        *,
        architecture: str,
        correctness_head: str = DEFAULT_CORRECTNESS_HEAD,
    ) -> None:
        super().__init__()
        self.encoder = create_delta_conv_regressor(
            feature_count,
            architecture=architecture,
        )
        if correctness_head == DEFAULT_CORRECTNESS_HEAD:
            self.correctness_head = nn.Sequential(
                nn.Linear(self.encoder.ENCODED_FEATURES, 512),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(512, 1000),
            )
        elif correctness_head == DEEP_CORRECTNESS_HEAD:
            self.correctness_head = nn.Sequential(
                nn.Linear(self.encoder.ENCODED_FEATURES, 1024),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(1024, 512),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(512, 1000),
            )
        else:
            raise ValueError(
                f"unsupported correctness head architecture: {correctness_head!r}"
            )

    def logits(self, image: torch.Tensor, engineered: torch.Tensor) -> torch.Tensor:
        return self.correctness_head(self.encoder.encode(image, engineered))

    def trainable_parameters(self) -> list[nn.Parameter]:
        modules = (
            self.encoder.stem,
            self.encoder.blocks,
            self.encoder.engineered,
            self.correctness_head,
        )
        return [parameter for module in modules for parameter in module.parameters()]


@dataclass(frozen=True)
class CorrectnessConvMetadata:
    feature_count: int
    feature_schema: str
    inference_dtype: str
    logit_temperature: float


@dataclass(frozen=True)
class _LoadedMember:
    model: CorrectnessConvRegressor
    feature_mean: np.ndarray
    feature_std: np.ndarray
    inference_dtype: str
    exact_accuracy: float
    logit_temperature: float


def _load_payload(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise RuntimeError(
            f"cannot safely load correctness-CNN checkpoint {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"correctness-CNN checkpoint must be a mapping: {path}")
    return payload


def _validated_member_payload(
    payload: Mapping[str, object],
    *,
    source: str,
    expected_feature_count: int,
    expected_feature_schema: str,
    device: torch.device,
) -> _LoadedMember:
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported correctness-CNN checkpoint schema in {source}")
    if payload.get("feature_schema") != expected_feature_schema:
        raise RuntimeError(
            f"correctness-CNN feature schema does not match bundle: {source}"
        )
    if payload.get("feature_count") != expected_feature_count:
        raise RuntimeError(
            f"correctness-CNN feature count does not match bundle: {source}"
        )
    inference_dtype = payload.get("inference_dtype")
    if inference_dtype not in SUPPORTED_INFERENCE_DTYPES:
        raise RuntimeError(f"correctness-CNN inference dtype is unsupported: {source}")
    architecture = payload.get("model_architecture", DEFAULT_MODEL_ARCHITECTURE)
    if architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        raise RuntimeError(
            f"correctness-CNN model architecture is unsupported: {source}"
        )
    correctness_head = payload.get("correctness_head", DEFAULT_CORRECTNESS_HEAD)
    if correctness_head not in SUPPORTED_CORRECTNESS_HEADS:
        raise RuntimeError(
            f"correctness-CNN head architecture is unsupported: {source}"
        )

    feature_mean_tensor = payload.get("feature_mean")
    feature_std_tensor = payload.get("feature_std")
    magnitude_order_tensor = payload.get("magnitude_order")
    state = payload.get("model_state")
    if not isinstance(feature_mean_tensor, torch.Tensor) or not isinstance(
        feature_std_tensor, torch.Tensor
    ):
        raise TypeError(f"correctness-CNN normalization tensors are missing: {source}")
    if not isinstance(magnitude_order_tensor, torch.Tensor) or not isinstance(
        state, dict
    ):
        raise TypeError(f"correctness-CNN input/model tensors are missing: {source}")
    feature_mean = (
        feature_mean_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    feature_std = (
        feature_std_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if feature_mean.shape != (expected_feature_count,) or feature_std.shape != (
        expected_feature_count,
    ):
        raise RuntimeError(f"correctness-CNN normalization shape is invalid: {source}")
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all():
        raise RuntimeError(f"correctness-CNN normalization must be finite: {source}")
    if np.any(feature_std <= 0.0):
        raise RuntimeError(
            "correctness-CNN normalization standard deviations must be positive: "
            f"{source}"
        )
    order = magnitude_order_tensor.detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.array_equal(order, MAGNITUDE_ORDER):
        raise RuntimeError(
            f"correctness-CNN magnitude order does not match the runtime: {source}"
        )
    try:
        input_log_scale = float(payload.get("input_log_scale"))
        exact_accuracy = float(payload.get("exact_accuracy"))
        logit_temperature = float(payload.get("logit_temperature"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"correctness-CNN scalar metadata is invalid: {source}"
        ) from exc
    if input_log_scale != INPUT_LOG_SCALE:
        raise RuntimeError(
            f"correctness-CNN input log scale does not match the runtime: {source}"
        )
    if not 0.0 <= exact_accuracy <= 1.0:
        raise RuntimeError(f"correctness-CNN exact accuracy is invalid: {source}")
    if not math.isfinite(logit_temperature) or logit_temperature <= 0.0:
        raise RuntimeError(f"correctness-CNN logit temperature is invalid: {source}")

    model = CorrectnessConvRegressor(
        expected_feature_count,
        architecture=str(architecture),
        correctness_head=str(correctness_head),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"correctness-CNN model state is incompatible: {source}: {exc}"
        ) from exc
    model = model.to(device=device, memory_format=torch.channels_last)
    model.eval()
    return _LoadedMember(
        model,
        feature_mean,
        feature_std,
        str(inference_dtype),
        exact_accuracy,
        logit_temperature,
    )


def _composite_members(
    payload: Mapping[str, object],
    path: Path,
    *,
    expected_feature_count: int,
    expected_feature_schema: str,
) -> Mapping[str, object]:
    if payload.get("schema_version") != COMPOSITE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported correctness-CNN composite checkpoint schema in {path}"
        )
    if payload.get("feature_schema") != expected_feature_schema:
        raise RuntimeError(
            f"correctness-CNN composite feature schema does not match bundle: {path}"
        )
    if payload.get("feature_count") != expected_feature_count:
        raise RuntimeError(
            f"correctness-CNN composite feature count does not match bundle: {path}"
        )
    members = payload.get("members")
    member_order = payload.get("member_order")
    if (
        not isinstance(members, dict)
        or not members
        or not all(isinstance(name, str) and name for name in members)
        or not isinstance(member_order, list)
        or member_order != list(members)
    ):
        raise RuntimeError(f"correctness-CNN composite member index is invalid: {path}")
    return members


def _validated_member(
    path: Path,
    *,
    expected_feature_count: int,
    expected_feature_schema: str,
    device: torch.device,
    checkpoint_member: str | None = None,
    composite_cache: dict[Path, Mapping[str, object]] | None = None,
) -> _LoadedMember:
    resolved = path.expanduser().resolve()
    payload = (
        composite_cache.get(resolved)
        if checkpoint_member is not None and composite_cache is not None
        else None
    )
    if payload is None:
        payload = _load_payload(resolved)
        if checkpoint_member is not None and composite_cache is not None:
            composite_cache[resolved] = payload
    source = str(resolved)
    if checkpoint_member is not None:
        members = _composite_members(
            payload,
            resolved,
            expected_feature_count=expected_feature_count,
            expected_feature_schema=expected_feature_schema,
        )
        if checkpoint_member not in members:
            raise RuntimeError(
                "correctness-CNN composite checkpoint does not contain member "
                f"{checkpoint_member!r}: {resolved}"
            )
        selected = members[checkpoint_member]
        if not isinstance(selected, dict):
            raise TypeError(
                "correctness-CNN composite member must be a mapping: "
                f"{resolved}#{checkpoint_member}"
            )
        payload = selected
        source = f"{resolved}#{checkpoint_member}"
    elif payload.get("schema_version") == COMPOSITE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"correctness-CNN composite checkpoint requires a member selector: {resolved}"
        )
    return _validated_member_payload(
        payload,
        source=source,
        expected_feature_count=expected_feature_count,
        expected_feature_schema=expected_feature_schema,
        device=device,
    )


def inspect_correctness_conv_checkpoint(
    path: str | Path,
    *,
    expected_feature_count: int,
    expected_feature_schema: str,
    checkpoint_member: str | None = None,
) -> CorrectnessConvMetadata:
    member = _validated_member(
        Path(path),
        expected_feature_count=expected_feature_count,
        expected_feature_schema=expected_feature_schema,
        device=torch.device("cpu"),
        checkpoint_member=checkpoint_member,
    )
    return CorrectnessConvMetadata(
        feature_count=len(member.feature_mean),
        feature_schema=expected_feature_schema,
        inference_dtype=member.inference_dtype,
        logit_temperature=member.logit_temperature,
    )


def inspect_correctness_conv_composite_checkpoint(
    path: str | Path,
    *,
    expected_members: tuple[str, ...],
    expected_feature_count: int,
    expected_feature_schema: str,
) -> Mapping[str, CorrectnessConvMetadata]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = _load_payload(checkpoint_path)
    members = _composite_members(
        payload,
        checkpoint_path,
        expected_feature_count=expected_feature_count,
        expected_feature_schema=expected_feature_schema,
    )
    if tuple(members) != expected_members:
        raise RuntimeError(
            "correctness-CNN composite members do not exactly match the bundle: "
            f"expected {expected_members!r}, got {tuple(members)!r}"
        )
    cache = {checkpoint_path: payload}
    metadata = {}
    for name in expected_members:
        member = _validated_member(
            checkpoint_path,
            expected_feature_count=expected_feature_count,
            expected_feature_schema=expected_feature_schema,
            device=torch.device("cpu"),
            checkpoint_member=name,
            composite_cache=cache,
        )
        metadata[name] = CorrectnessConvMetadata(
            feature_count=len(member.feature_mean),
            feature_schema=expected_feature_schema,
            inference_dtype=member.inference_dtype,
            logit_temperature=member.logit_temperature,
        )
    return MappingProxyType(metadata)


def export_correctness_conv_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    inference_dtype: str = "bfloat16",
    logit_temperature: float = 1.0,
) -> Path:
    if inference_dtype not in SUPPORTED_INFERENCE_DTYPES:
        raise ValueError(f"unsupported inference dtype: {inference_dtype}")
    if not math.isfinite(logit_temperature) or logit_temperature <= 0.0:
        raise ValueError("logit_temperature must be finite and positive")
    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"source checkpoint must contain a mapping: {source_path}")
    state = payload.get("model_state")
    feature_mean = np.asarray(payload.get("feature_mean"), dtype=np.float32)
    feature_std = np.asarray(payload.get("feature_std"), dtype=np.float32)
    architecture = payload.get("model_architecture", DEFAULT_MODEL_ARCHITECTURE)
    correctness_head = payload.get("correctness_head", DEFAULT_CORRECTNESS_HEAD)
    try:
        exact_accuracy = float(payload.get("exact_accuracy"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source checkpoint exact accuracy is invalid: {source_path}"
        ) from exc
    if architecture not in SUPPORTED_MODEL_ARCHITECTURES:
        raise ValueError(
            f"source checkpoint has unsupported model architecture: {architecture!r}"
        )
    if correctness_head not in SUPPORTED_CORRECTNESS_HEADS:
        raise ValueError(
            f"source checkpoint has unsupported correctness head: {correctness_head!r}"
        )
    if (
        not isinstance(state, dict)
        or feature_mean.ndim != 1
        or feature_std.shape != feature_mean.shape
        or not 0.0 <= exact_accuracy <= 1.0
    ):
        raise ValueError(
            f"source checkpoint has an invalid model/normalization payload: {source_path}"
        )
    safe_state: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise TypeError(
                f"source checkpoint model_state must contain named tensors: {source_path}"
            )
        safe_state[name] = value.detach().cpu().contiguous()
    safe_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema": str(payload.get("feature_schema")),
        "feature_count": len(feature_mean),
        "feature_mean": torch.from_numpy(np.ascontiguousarray(feature_mean)),
        "feature_std": torch.from_numpy(np.ascontiguousarray(feature_std)),
        "magnitude_order": torch.from_numpy(MAGNITUDE_ORDER.copy()),
        "input_log_scale": INPUT_LOG_SCALE,
        "inference_dtype": inference_dtype,
        "model_architecture": str(architecture),
        "correctness_head": str(correctness_head),
        "exact_accuracy": exact_accuracy,
        "logit_temperature": float(logit_temperature),
        "model_state": safe_state,
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(safe_payload, destination_path)
    inspect_correctness_conv_checkpoint(
        destination_path,
        expected_feature_count=len(feature_mean),
        expected_feature_schema=str(payload.get("feature_schema")),
    )
    return destination_path


def export_correctness_conv_composite_checkpoint(
    checkpoints: Mapping[str, str | Path],
    destination: str | Path,
    *,
    expected_feature_count: int,
    expected_feature_schema: str,
) -> Path:
    if len(checkpoints) < 2:
        raise ValueError("a composite correctness-CNN checkpoint requires two members")
    member_payloads: dict[str, Mapping[str, object]] = {}
    for name, source in checkpoints.items():
        if not isinstance(name, str) or not name:
            raise ValueError("composite correctness-CNN member names must be nonempty")
        source_path = Path(source).expanduser().resolve()
        payload = _load_payload(source_path)
        _validated_member_payload(
            payload,
            source=str(source_path),
            expected_feature_count=expected_feature_count,
            expected_feature_schema=expected_feature_schema,
            device=torch.device("cpu"),
        )
        member_payloads[name] = payload
    destination_path = Path(destination).expanduser().resolve()
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": COMPOSITE_CHECKPOINT_SCHEMA_VERSION,
            "feature_schema": expected_feature_schema,
            "feature_count": expected_feature_count,
            "member_order": list(member_payloads),
            "members": member_payloads,
        },
        destination_path,
    )
    inspect_correctness_conv_composite_checkpoint(
        destination_path,
        expected_members=tuple(member_payloads),
        expected_feature_count=expected_feature_count,
        expected_feature_schema=expected_feature_schema,
    )
    return destination_path


class CorrectnessConvInference:
    def __init__(
        self,
        checkpoints: Mapping[str, Path | tuple[Path, str | None]],
        *,
        feature_count: int,
        feature_schema: str,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> None:
        if not checkpoints:
            raise ValueError("at least one correctness-CNN checkpoint is required")
        if batch_size < 1:
            raise ValueError("correctness-CNN batch_size must be positive")
        try:
            torch_device = torch.device(device)
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                f"invalid correctness-CNN device {device!r}: {exc}"
            ) from exc
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for correctness-CNN inference but is unavailable"
            )
        if torch_device.type not in {"cpu", "cuda"}:
            raise RuntimeError(
                "correctness-CNN inference supports only CPU and CUDA devices"
            )
        self._device = torch_device
        self._batch_size = batch_size
        composite_cache: dict[Path, Mapping[str, object]] = {}
        members: dict[str, _LoadedMember] = {}
        for key, reference in checkpoints.items():
            if isinstance(reference, tuple):
                path, checkpoint_member = reference
            else:
                path, checkpoint_member = reference, None
            if checkpoint_member is not None and (
                not isinstance(checkpoint_member, str) or not checkpoint_member
            ):
                raise ValueError(
                    "correctness-CNN checkpoint member selector must be nonempty"
                )
            members[key] = _validated_member(
                Path(path),
                expected_feature_count=feature_count,
                expected_feature_schema=feature_schema,
                device=torch_device,
                checkpoint_member=checkpoint_member,
                composite_cache=composite_cache,
            )
        self._members = MappingProxyType(members)

    @torch.inference_mode()
    def predict(
        self,
        delta: np.ndarray,
        engineered_features: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        raw_delta = np.asarray(delta)
        engineered = np.asarray(engineered_features, dtype=np.float32)
        if raw_delta.ndim != 3 or raw_delta.shape[1:] != (256, 256):
            raise ValueError(
                "correctness-CNN input must have shape [N, 256, 256]; "
                f"got {raw_delta.shape}"
            )
        if engineered.ndim != 2 or len(engineered) != len(raw_delta):
            raise ValueError(
                "correctness-CNN engineered feature rows do not match delta rows"
            )
        outputs = {
            key: np.empty(len(raw_delta), dtype=np.float64) for key in self._members
        }
        for start in range(0, len(raw_delta), self._batch_size):
            stop = min(start + self._batch_size, len(raw_delta))
            quadrant = np.asarray(
                raw_delta[start:stop, MAGNITUDE_ORDER[:, None], MAGNITUDE_ORDER],
                dtype=np.float32,
            )
            magnitude = np.log1p(np.abs(quadrant)) / INPUT_LOG_SCALE
            image = np.stack((np.sign(quadrant) * magnitude, magnitude), axis=1).astype(
                np.float32,
                copy=False,
            )
            image_tensor = torch.from_numpy(image).to(
                device=self._device,
                non_blocking=self._device.type == "cuda",
            )
            image_tensor = image_tensor.contiguous(memory_format=torch.channels_last)
            for key, member in self._members.items():
                normalized = np.clip(
                    (engineered[start:stop] - member.feature_mean) / member.feature_std,
                    -12.0,
                    12.0,
                )
                engineered_tensor = torch.from_numpy(
                    np.ascontiguousarray(normalized, dtype=np.float32)
                ).to(
                    device=self._device,
                    non_blocking=self._device.type == "cuda",
                )
                use_bfloat16 = (
                    self._device.type == "cuda" and member.inference_dtype == "bfloat16"
                )
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=torch.bfloat16,
                    enabled=use_bfloat16,
                ):
                    logits = member.model.logits(image_tensor, engineered_tensor)
                    predicted = member.exact_accuracy - torch.sigmoid(
                        logits.float() / member.logit_temperature
                    ).mean(dim=1)
                outputs[key][start:stop] = predicted.cpu().numpy()
        return MappingProxyType(outputs)
