from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from aaae.inference_runtime.cnn.dataset_builders.real_labels import (
    LabelGenerationPlan,
    _load_samples,
    load_label_generation_plan,
)
from aaae.inference_runtime.cnn.errors import LabelGenerationError
from aaae.inference_runtime.cnn.s8s8_conv_cuda import (
    S8S8ConvCudaRuntime,
    _create_session,
    _import_onnxruntime,
)
from aaae.inference_runtime.cnn.s8s8_conv_graph import _selected_conv_nodes

PROFILE_BUILD_SCHEMA = "cnn_qdq_application_profile_build_v1"
PROFILE_METADATA_SCHEMA = "cnn_application_profile_metadata_v1"
PROFILE_AUDIT_SCHEMA = "cnn_qdq_application_profile_audit_v1"
PROPAGATION_METHOD = "per_layer_central_constant_residual_logit_rms_v1"
DOMAIN = "s8_numeric"
ZERO_CODE_INDEX = 128


@dataclass(frozen=True)
class QdqProfileBuildPlan:
    manifest_path: Path
    manifest: dict[str, Any]
    source_label_manifest: Path
    source_label_manifest_sha256: str
    label_plan: LabelGenerationPlan
    network_id: str
    dataset_id: str
    histogram_sample_count: int
    propagation_sample_count: int
    sample_seed: int
    probe_magnitude: int
    output_dir: Path


@dataclass(frozen=True)
class ProfileBuildOutcome:
    output_dir: Path
    profile_metadata: Path
    profile_arrays: Path
    layer_count: int
    histogram_sample_count: int
    propagation_sample_count: int
    profile_sha256: str


@dataclass(frozen=True)
class _LayerSpec:
    node_name: str
    activation_raw_name: str
    weight_raw_name: str
    weights: np.ndarray
    strides: tuple[int, int]
    dilations: tuple[int, int]
    pads: tuple[int, int, int, int]
    group: int


@dataclass(frozen=True)
class _ConvGeometry:
    activation_reuse: np.ndarray
    weight_spatial_uses: np.ndarray
    mac_count: int


@dataclass(frozen=True)
class _HistogramResult:
    layer_ids: tuple[str, ...]
    activation_histograms: np.ndarray
    weight_histograms: np.ndarray
    mac_counts: np.ndarray
    activation_shapes: tuple[tuple[int, int, int], ...]
    placement_event_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LabelGenerationError(f"unable to read {field} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LabelGenerationError(f"{field} must contain a JSON object: {path}")
    return payload


def _required_object(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise LabelGenerationError(f"manifest.{key} must be an object")
    return value


def _required_text(payload: Mapping[str, Any], key: str, *, field: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise LabelGenerationError(f"{field}.{key} must be a nonempty string")
    return value.strip()


def _resolve_file(base: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LabelGenerationError(f"{field} must be a nonempty path")
    candidate = Path(value).expanduser()
    path = (candidate if candidate.is_absolute() else base / candidate).resolve()
    if not path.is_file():
        raise LabelGenerationError(f"{field} is missing: {path}")
    return path


def _resolve_output(base: Path, value: object) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LabelGenerationError("manifest.output_dir must be a nonempty path")
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LabelGenerationError(f"{field} must be a positive integer")
    return int(value)


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabelGenerationError(f"{field} must be a nonnegative integer")
    return int(value)


def load_qdq_profile_build_plan(
    path: str | Path,
    *,
    allow_existing_output: bool = False,
) -> QdqProfileBuildPlan:
    manifest_path = Path(path).expanduser().resolve()
    manifest = _read_json(manifest_path, field="profile build manifest")
    if manifest.get("schema_version") != PROFILE_BUILD_SCHEMA:
        raise LabelGenerationError(
            f"manifest.schema_version must be {PROFILE_BUILD_SCHEMA!r}"
        )
    base = manifest_path.parent

    source = _required_object(manifest, "source")
    source_label_manifest = _resolve_file(
        base,
        source.get("real_label_manifest"),
        field="source.real_label_manifest",
    )
    expected_source_hash = _required_text(
        source,
        "real_label_manifest_sha256",
        field="source",
    ).lower()
    actual_source_hash = _sha256(source_label_manifest)
    if actual_source_hash != expected_source_hash:
        raise LabelGenerationError(
            "source.real_label_manifest_sha256 mismatch: "
            f"expected {expected_source_hash}, got {actual_source_hash}"
        )
    label_plan = load_label_generation_plan(
        source_label_manifest,
        allow_existing_output=True,
    )

    input_asset = _required_object(manifest, "input_asset")
    input_path = _resolve_file(base, input_asset.get("path"), field="input_asset.path")
    expected_input_hash = _required_text(
        input_asset,
        "sha256",
        field="input_asset",
    ).lower()
    actual_input_hash = _sha256(input_path)
    if actual_input_hash != expected_input_hash:
        raise LabelGenerationError(
            f"input_asset.sha256 mismatch: expected {expected_input_hash}, got {actual_input_hash}"
        )
    if actual_input_hash != label_plan.dataset_sha256:
        raise LabelGenerationError(
            "profile input asset differs from the frozen real-label input asset"
        )

    profile = _required_object(manifest, "profile")
    network_id = _required_text(profile, "network_id", field="profile")
    dataset_id = _required_text(profile, "dataset_id", field="profile")
    if profile.get("domain") != DOMAIN:
        raise LabelGenerationError(
            f"profile.domain must be {DOMAIN!r} for signed_code_plus_128 LUTs"
        )
    histogram_sample_count = _positive_int(
        profile.get("histogram_sample_count"),
        field="profile.histogram_sample_count",
    )
    propagation_sample_count = _positive_int(
        profile.get("propagation_sample_count"),
        field="profile.propagation_sample_count",
    )
    if propagation_sample_count > histogram_sample_count:
        raise LabelGenerationError(
            "profile.propagation_sample_count must not exceed histogram_sample_count"
        )
    sample_seed = _nonnegative_int(
        profile.get("sample_seed"), field="profile.sample_seed"
    )

    propagation = _required_object(manifest, "propagation")
    if propagation.get("method") != PROPAGATION_METHOD:
        raise LabelGenerationError(f"propagation.method must be {PROPAGATION_METHOD!r}")
    probe_magnitude = _positive_int(
        propagation.get("probe_magnitude"),
        field="propagation.probe_magnitude",
    )

    execution = _required_object(manifest, "execution")
    if execution.get("device") != "cuda":
        raise LabelGenerationError("execution.device must be 'cuda'")
    gpu_id = _nonnegative_int(execution.get("gpu_id"), field="execution.gpu_id")
    batch_size = _positive_int(
        execution.get("batch_size"), field="execution.batch_size"
    )
    input_residency = str(execution.get("input_residency", "staged")).strip().lower()
    if input_residency not in {"auto", "device", "staged"}:
        raise LabelGenerationError(
            "execution.input_residency must be 'auto', 'device', or 'staged'"
        )
    preallocate_outputs = execution.get("preallocate_outputs", True)
    if not isinstance(preallocate_outputs, bool):
        raise LabelGenerationError("execution.preallocate_outputs must be boolean")
    cudnn_conv_algo_search = (
        str(execution.get("cudnn_conv_algo_search", "DEFAULT")).strip().upper()
    )
    if cudnn_conv_algo_search not in {"DEFAULT", "HEURISTIC", "EXHAUSTIVE"}:
        raise LabelGenerationError(
            "execution.cudnn_conv_algo_search must be DEFAULT, HEURISTIC, or EXHAUSTIVE"
        )

    output_dir = _resolve_output(base, manifest.get("output_dir"))
    if output_dir.exists() and not allow_existing_output:
        raise LabelGenerationError(f"output_dir already exists: {output_dir}")

    resolved_label_plan = replace(
        label_plan,
        dataset_path=input_path,
        dataset_sha256=actual_input_hash,
        gpu_id=gpu_id,
        gpu_ids=(gpu_id,),
        batch_size=batch_size,
        input_residency=input_residency,
        preallocate_outputs=preallocate_outputs,
        cudnn_conv_algo_search=cudnn_conv_algo_search,
    )
    return QdqProfileBuildPlan(
        manifest_path=manifest_path,
        manifest=manifest,
        source_label_manifest=source_label_manifest,
        source_label_manifest_sha256=actual_source_hash,
        label_plan=resolved_label_plan,
        network_id=network_id,
        dataset_id=dataset_id,
        histogram_sample_count=histogram_sample_count,
        propagation_sample_count=propagation_sample_count,
        sample_seed=sample_seed,
        probe_magnitude=probe_magnitude,
        output_dir=output_dir,
    )


def _selected_sample_indices(
    sample_ids: Sequence[str],
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    if count > len(sample_ids):
        raise LabelGenerationError(
            f"requested {count} profile samples but the frozen asset has {len(sample_ids)}"
        )
    ranked = sorted(
        range(len(sample_ids)),
        key=lambda index: (
            hashlib.sha256(f"{seed}\0{sample_ids[index]}".encode()).digest(),
            index,
        ),
    )
    return np.asarray(ranked[:count], dtype=np.int64)


def _node_attributes(node: onnx.NodeProto) -> dict[str, Any]:
    return {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def _pair(value: object, *, default: tuple[int, int], field: str) -> tuple[int, int]:
    if value is None:
        return default
    values = tuple(int(item) for item in value)
    if len(values) != 2 or any(item < 1 for item in values):
        raise LabelGenerationError(f"{field} must contain two positive integers")
    return values


def _pads(value: object) -> tuple[int, int, int, int]:
    if value is None:
        return (0, 0, 0, 0)
    values = tuple(int(item) for item in value)
    if len(values) != 4 or any(item < 0 for item in values):
        raise LabelGenerationError("Conv pads must contain four nonnegative integers")
    return values


def _load_layer_specs(
    plan: QdqProfileBuildPlan,
) -> tuple[onnx.ModelProto, tuple[_LayerSpec, ...]]:
    try:
        model = onnx.load(str(plan.label_plan.model_path))
    except Exception as exc:
        raise LabelGenerationError(
            f"unable to load profile ONNX model {plan.label_plan.model_path}: {exc}"
        ) from exc
    selected, infos = _selected_conv_nodes(model, plan.label_plan.node_names)
    initializers = {
        str(item.name): np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    specs: list[_LayerSpec] = []
    for node in selected:
        activation = infos[str(node.input[0])]
        weight = infos[str(node.input[1])]
        if np.any(activation.zero_point != 0) or np.any(weight.zero_point != 0):
            raise LabelGenerationError(
                f"profile layer {node.name!r} is not symmetric INT8; zero points must all be zero"
            )
        raw_weight = initializers[weight.raw_code_name]
        attributes = _node_attributes(node)
        specs.append(
            _LayerSpec(
                node_name=str(node.name),
                activation_raw_name=activation.raw_code_name,
                weight_raw_name=weight.raw_code_name,
                weights=np.asarray(raw_weight),
                strides=_pair(
                    attributes.get("strides"),
                    default=(1, 1),
                    field=f"{node.name}.strides",
                ),
                dilations=_pair(
                    attributes.get("dilations"),
                    default=(1, 1),
                    field=f"{node.name}.dilations",
                ),
                pads=_pads(attributes.get("pads")),
                group=int(attributes.get("group", 1)),
            )
        )
    return model, tuple(specs)


def _axis_usage(
    input_size: int,
    kernel_size: int,
    *,
    stride: int,
    dilation: int,
    pad_before: int,
    pad_after: int,
) -> tuple[np.ndarray, np.ndarray]:
    effective_kernel = dilation * (kernel_size - 1) + 1
    output_size = (input_size + pad_before + pad_after - effective_kernel) // stride + 1
    if output_size < 1:
        raise LabelGenerationError("Conv geometry produces an empty output")
    input_reuse = np.zeros(input_size, dtype=np.int64)
    kernel_uses = np.zeros(kernel_size, dtype=np.int64)
    for output_index in range(output_size):
        origin = output_index * stride - pad_before
        for kernel_index in range(kernel_size):
            input_index = origin + kernel_index * dilation
            if 0 <= input_index < input_size:
                input_reuse[input_index] += 1
                kernel_uses[kernel_index] += 1
    if int(input_reuse.sum()) != int(kernel_uses.sum()):
        raise RuntimeError("Conv axis-usage accounting is inconsistent")
    return input_reuse, kernel_uses


def _conv_geometry(spec: _LayerSpec, activation_shape: Sequence[int]) -> _ConvGeometry:
    if len(activation_shape) != 3:
        raise LabelGenerationError(
            f"activation for {spec.node_name!r} must be CHW; got {tuple(activation_shape)}"
        )
    channels, height, width = (int(value) for value in activation_shape)
    if spec.weights.dtype != np.int8 or spec.weights.ndim != 4:
        raise LabelGenerationError(
            f"weights for {spec.node_name!r} must be rank-4 INT8"
        )
    output_channels, channels_per_group, kernel_h, kernel_w = spec.weights.shape
    if spec.group < 1 or channels != channels_per_group * spec.group:
        raise LabelGenerationError(
            f"activation channels do not match grouped weights for {spec.node_name!r}"
        )
    reuse_h, use_h = _axis_usage(
        height,
        kernel_h,
        stride=spec.strides[0],
        dilation=spec.dilations[0],
        pad_before=spec.pads[0],
        pad_after=spec.pads[2],
    )
    reuse_w, use_w = _axis_usage(
        width,
        kernel_w,
        stride=spec.strides[1],
        dilation=spec.dilations[1],
        pad_before=spec.pads[1],
        pad_after=spec.pads[3],
    )
    activation_reuse = np.multiply.outer(reuse_h, reuse_w)
    weight_spatial_uses = np.multiply.outer(use_h, use_w)
    if int(activation_reuse.sum()) != int(weight_spatial_uses.sum()):
        raise RuntimeError("Conv spatial-usage accounting is inconsistent")
    mac_count = (
        int(output_channels) * int(channels_per_group) * int(weight_spatial_uses.sum())
    )
    if mac_count < 1 or mac_count > np.iinfo(np.int64).max:
        raise LabelGenerationError(
            f"invalid MAC count for {spec.node_name!r}: {mac_count}"
        )
    return _ConvGeometry(
        activation_reuse=activation_reuse,
        weight_spatial_uses=weight_spatial_uses,
        mac_count=mac_count,
    )


def _activation_histogram(values: np.ndarray, reuse: np.ndarray) -> np.ndarray:
    if values.dtype != np.int8 or values.ndim != 4 or values.shape[2:] != reuse.shape:
        raise LabelGenerationError(
            f"invalid quantized activation batch {values.dtype} {values.shape} for reuse {reuse.shape}"
        )
    indices = values.astype(np.int16, copy=False) + 128
    weights = np.broadcast_to(reuse, values.shape)
    return np.bincount(
        indices.reshape(-1),
        weights=weights.reshape(-1),
        minlength=256,
    ).astype(np.float64, copy=False)


def _weight_histogram(spec: _LayerSpec, spatial_uses: np.ndarray) -> np.ndarray:
    histogram = np.zeros(256, dtype=np.float64)
    for kernel_h in range(spec.weights.shape[2]):
        for kernel_w in range(spec.weights.shape[3]):
            count = int(spatial_uses[kernel_h, kernel_w])
            if count == 0:
                continue
            values = (
                spec.weights[:, :, kernel_h, kernel_w].astype(np.int16, copy=False)
                + 128
            )
            histogram += np.bincount(values.reshape(-1), minlength=256) * count
    return histogram


def _profile_event_node_names(event: Mapping[str, Any]) -> set[str]:
    args = event.get("args")
    raw_values: list[object] = [event.get("name")]
    if isinstance(args, Mapping):
        raw_values.insert(0, args.get("node_name"))
        raw_values.insert(1, args.get("nodeName"))
    names: set[str] = set()
    suffixes = ("_kernel_time", "_fence_before", "_fence_after")
    for raw_value in raw_values:
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        candidate = raw_value.strip()
        names.add(candidate)
        if candidate.startswith("fused "):
            names.add(candidate[6:])
        for suffix in suffixes:
            if candidate.endswith(suffix):
                names.add(candidate[: -len(suffix)])
    return names


def _verify_cuda_placement(session: Any, node_names: Sequence[str]) -> int:
    try:
        profile_path = Path(session.end_profiling())
        events = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LabelGenerationError(
            f"unable to verify profile Conv CUDA placement: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(events, list):
        raise LabelGenerationError("profile placement trace is not a JSON event list")
    matches: dict[str, list[Mapping[str, Any]]] = {name: [] for name in node_names}
    for event in events:
        if not isinstance(event, Mapping):
            continue
        args = event.get("args")
        if not isinstance(args, Mapping) or not args.get("provider"):
            continue
        names = _profile_event_node_names(event)
        for node_name, node_events in matches.items():
            if node_name in names:
                node_events.append(event)
    missing = [name for name, node_events in matches.items() if not node_events]
    providers = {
        str(event["args"].get("provider", ""))
        for node_events in matches.values()
        for event in node_events
    }
    if missing or providers != {"CUDAExecutionProvider"}:
        raise LabelGenerationError(
            "profile Conv placement verification failed: "
            f"missing_nodes={missing}, providers={sorted(providers)}"
        )
    return sum(len(node_events) for node_events in matches.values())


def _collect_histograms(
    plan: QdqProfileBuildPlan,
    inputs: np.ndarray,
    specs: tuple[_LayerSpec, ...],
    model: onnx.ModelProto,
    *,
    progress: Callable[[str], None],
) -> _HistogramResult:
    raw_names = tuple(dict.fromkeys(spec.activation_raw_name for spec in specs))
    existing_outputs = {str(value.name) for value in model.graph.output}
    for raw_name in raw_names:
        if raw_name not in existing_outputs:
            model.graph.output.append(
                helper.make_tensor_value_info(raw_name, TensorProto.INT8, None)
            )
    if len(model.graph.output) != 1 + len(raw_names):
        raise LabelGenerationError(
            "profile graph must expose one classifier output plus the selected raw activations"
        )
    classifier_output = str(model.graph.output[0].name)
    requested_outputs = [*raw_names, classifier_output]

    activation_histograms = np.zeros((len(specs), 256), dtype=np.float64)
    weight_histograms = np.zeros((len(specs), 256), dtype=np.float64)
    mac_counts = np.zeros(len(specs), dtype=np.int64)
    geometries: list[_ConvGeometry] = []
    activation_shapes: list[tuple[int, int, int]] = []
    placement_event_count = 0

    ort = _import_onnxruntime()
    with tempfile.TemporaryDirectory(prefix="cnn-qdq-profile-") as temporary:
        temporary_root = Path(temporary)
        augmented_model = temporary_root / "profile_outputs.onnx"
        onnx.save(model, str(augmented_model))
        session = _create_session(
            ort,
            augmented_model,
            gpu_id=plan.label_plan.gpu_id,
            custom_op_library=None,
            profile_file_prefix=temporary_root / "profile-placement",
            cudnn_conv_algo_search=plan.label_plan.cudnn_conv_algo_search,
        )
        output_index = {name: index for index, name in enumerate(raw_names)}
        for start in range(0, inputs.shape[0], plan.label_plan.batch_size):
            stop = min(start + plan.label_plan.batch_size, inputs.shape[0])
            batch = np.ascontiguousarray(inputs[start:stop], dtype=np.float32)
            try:
                results = session.run(
                    requested_outputs,
                    {plan.label_plan.model_input_name: batch},
                )
            except Exception as exc:
                raise LabelGenerationError(
                    f"quantized activation profiling failed for samples [{start}:{stop}]: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if start == 0:
                placement_event_count = _verify_cuda_placement(
                    session,
                    [spec.node_name for spec in specs],
                )
                for spec in specs:
                    values = np.asarray(results[output_index[spec.activation_raw_name]])
                    if values.dtype != np.int8 or values.ndim != 4:
                        raise LabelGenerationError(
                            f"raw activation {spec.activation_raw_name!r} has invalid "
                            f"dtype/shape {values.dtype} {values.shape}"
                        )
                    activation_shape = tuple(int(value) for value in values.shape[1:])
                    geometry = _conv_geometry(spec, activation_shape)
                    geometries.append(geometry)
                    activation_shapes.append(activation_shape)
                    mac_counts[len(geometries) - 1] = geometry.mac_count
                    weight_histograms[len(geometries) - 1] = _weight_histogram(
                        spec,
                        geometry.weight_spatial_uses,
                    )
            for layer_index, (spec, geometry) in enumerate(
                zip(specs, geometries, strict=True)
            ):
                values = np.asarray(results[output_index[spec.activation_raw_name]])
                activation_histograms[layer_index] += _activation_histogram(
                    values,
                    geometry.activation_reuse,
                )
            progress(f"activation histograms: {stop}/{inputs.shape[0]} samples")

    if np.any(activation_histograms.sum(axis=1) <= 0):
        raise LabelGenerationError("one or more activation histograms are empty")
    if np.any(weight_histograms.sum(axis=1) <= 0):
        raise LabelGenerationError("one or more weight histograms are empty")
    return _HistogramResult(
        layer_ids=tuple(spec.node_name for spec in specs),
        activation_histograms=activation_histograms,
        weight_histograms=weight_histograms,
        mac_counts=mac_counts,
        activation_shapes=tuple(activation_shapes),
        placement_event_count=placement_event_count,
    )


def _state_contract(
    plan: QdqProfileBuildPlan,
    histogram_indices: np.ndarray,
    propagation_indices: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, Any]:
    semantic = {
        "schema_version": PROFILE_BUILD_SCHEMA,
        "source_label_manifest_sha256": plan.source_label_manifest_sha256,
        "model_sha256": plan.label_plan.model_sha256,
        "input_sha256": plan.label_plan.dataset_sha256,
        "network_id": plan.network_id,
        "dataset_id": plan.dataset_id,
        "domain": DOMAIN,
        "node_names": list(plan.label_plan.node_names),
        "histogram_indices": histogram_indices.tolist(),
        "histogram_sample_ids": sample_ids[histogram_indices].astype(str).tolist(),
        "propagation_indices": propagation_indices.tolist(),
        "propagation_sample_ids": sample_ids[propagation_indices].astype(str).tolist(),
        "sample_seed": plan.sample_seed,
        "propagation_method": PROPAGATION_METHOD,
        "probe_magnitude": plan.probe_magnitude,
        "gpu_id": plan.label_plan.gpu_id,
        "batch_size": plan.label_plan.batch_size,
        "cudnn_conv_algo_search": plan.label_plan.cudnn_conv_algo_search,
    }
    serialized = json.dumps(
        semantic, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return {
        **semantic,
        "build_fingerprint": _sha256_bytes(serialized.encode("utf-8")),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".json",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npz_atomic(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_histogram_checkpoint(
    path: Path,
    *,
    expected_layer_ids: tuple[str, ...],
) -> _HistogramResult:
    try:
        with np.load(path, allow_pickle=False) as archive:
            layer_ids = tuple(str(value) for value in archive["layer_ids"].tolist())
            activation = np.asarray(archive["activation_histograms"], dtype=np.float64)
            weights = np.asarray(archive["weight_histograms"], dtype=np.float64)
            mac_counts = np.asarray(archive["mac_counts"], dtype=np.int64)
            activation_shapes_array = np.asarray(
                archive["activation_shapes"], dtype=np.int64
            )
            placement_event_count = int(archive["placement_event_count"].item())
    except (OSError, KeyError, ValueError) as exc:
        raise LabelGenerationError(
            f"invalid histogram checkpoint {path}: {exc}"
        ) from exc
    layer_count = len(expected_layer_ids)
    if layer_ids != expected_layer_ids:
        raise LabelGenerationError(
            "histogram checkpoint layer order differs from the build plan"
        )
    if activation.shape != (layer_count, 256) or weights.shape != (layer_count, 256):
        raise LabelGenerationError("histogram checkpoint arrays have invalid shapes")
    if mac_counts.shape != (layer_count,) or activation_shapes_array.shape != (
        layer_count,
        3,
    ):
        raise LabelGenerationError(
            "histogram checkpoint geometry arrays have invalid shapes"
        )
    return _HistogramResult(
        layer_ids=layer_ids,
        activation_histograms=activation,
        weight_histograms=weights,
        mac_counts=mac_counts,
        activation_shapes=tuple(
            tuple(int(item) for item in row) for row in activation_shapes_array
        ),
        placement_event_count=placement_event_count,
    )


def _probe_lut(magnitude: int) -> np.ndarray:
    probe = np.full((256, 256), magnitude, dtype=np.int32)
    probe[ZERO_CODE_INDEX, :] = 0
    probe[:, ZERO_CODE_INDEX] = 0
    return probe


def _measure_layer_gain(
    plan: QdqProfileBuildPlan,
    node_name: str,
    inputs: np.ndarray,
    *,
    mac_count: int,
    activation_histogram: np.ndarray,
    weight_histogram: np.ndarray,
) -> dict[str, Any]:
    layer_plan = replace(plan.label_plan, node_names=(node_name,))
    positive = _probe_lut(plan.probe_magnitude)
    negative = np.ascontiguousarray(-positive)
    with S8S8ConvCudaRuntime.from_plan(layer_plan) as runtime:
        plus_logits = runtime.predict_approx(inputs, positive)
        minus_logits = runtime.predict_approx(inputs, negative)
        execution = runtime.execution_metadata
    derivative = (plus_logits.astype(np.float64) - minus_logits.astype(np.float64)) / (
        2.0 * plan.probe_magnitude
    )
    if not np.isfinite(derivative).all():
        raise LabelGenerationError(
            f"propagation probe returned non-finite logits for {node_name!r}"
        )
    response_rms = float(np.sqrt(np.mean(np.square(derivative))))
    response_mean_abs = float(np.mean(np.abs(derivative)))
    activation_mass = float(activation_histogram.sum())
    weight_mass = float(weight_histogram.sum())
    nonzero_mass = (
        1.0 - float(activation_histogram[ZERO_CODE_INDEX]) / activation_mass
    ) * (1.0 - float(weight_histogram[ZERO_CODE_INDEX]) / weight_mass)
    if not np.isfinite(nonzero_mass) or nonzero_mass < 0:
        raise LabelGenerationError(
            f"profile layer {node_name!r} has invalid nonzero operand mass"
        )
    inactive_under_profile_inputs = nonzero_mass == 0.0
    if inactive_under_profile_inputs:
        if not np.array_equal(plus_logits, minus_logits):
            raise LabelGenerationError(
                f"profile layer {node_name!r} has zero nonzero operand mass "
                "but a nonzero propagation response"
            )
        gain = 0.0
    else:
        gain = response_rms / (float(mac_count) * nonzero_mass)
    if not np.isfinite(gain) or gain < 0:
        raise LabelGenerationError(
            f"profile layer {node_name!r} produced an invalid gain"
        )
    return {
        "node_name": node_name,
        "propagation_gain": gain,
        "central_logit_rms_per_unit_residual": response_rms,
        "central_logit_mean_abs_per_unit_residual": response_mean_abs,
        "independent_nonzero_operand_mass": nonzero_mass,
        "inactive_under_profile_inputs": inactive_under_profile_inputs,
        "mac_count": int(mac_count),
        "probe_magnitude": plan.probe_magnitude,
        "sample_count": int(inputs.shape[0]),
        "plus_logits_sha256": _sha256_bytes(
            np.ascontiguousarray(plus_logits).tobytes()
        ),
        "minus_logits_sha256": _sha256_bytes(
            np.ascontiguousarray(minus_logits).tobytes()
        ),
        "cuda_placement_verified": bool(execution["approx_conv_placement_verified"]),
        "cuda_placement_event_count": int(execution["approx_conv_profile_event_count"]),
        "custom_op_build": execution["custom_op_build"],
    }


def _validate_gain_checkpoint(
    payload: Mapping[str, Any],
    *,
    node_name: str,
    mac_count: int,
    sample_count: int,
    probe_magnitude: int,
) -> dict[str, Any]:
    expected = {
        "node_name": node_name,
        "mac_count": int(mac_count),
        "sample_count": int(sample_count),
        "probe_magnitude": int(probe_magnitude),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise LabelGenerationError(
                f"propagation checkpoint for {node_name!r} has mismatched {field}"
            )
    gain = payload.get("propagation_gain")
    if not isinstance(gain, (int, float)) or not np.isfinite(gain) or gain < 0:
        raise LabelGenerationError(
            f"propagation checkpoint for {node_name!r} has an invalid gain"
        )
    if payload.get("cuda_placement_verified") is not True:
        raise LabelGenerationError(
            f"propagation checkpoint for {node_name!r} lacks CUDA placement proof"
        )
    return dict(payload)


def _profile_digest(
    *,
    network_id: str,
    dataset_id: str,
    layer_ids: tuple[str, ...],
    activation_histograms: np.ndarray,
    weight_histograms: np.ndarray,
    mac_counts: np.ndarray,
    propagation_gains: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "dataset_id": dataset_id,
        "domain": DOMAIN,
        "layer_ids": list(layer_ids),
        "network_id": network_id,
        "profile_evidence": "measured",
        "schema": "cnn_application_profile_digest_v1",
    }
    digest.update(
        json.dumps(
            metadata, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    )
    normalized_activation = activation_histograms / activation_histograms.sum(
        axis=1, keepdims=True
    )
    normalized_weights = weight_histograms / weight_histograms.sum(
        axis=1, keepdims=True
    )
    arrays = (
        ("activation_histograms", normalized_activation, "<f8"),
        ("weight_histograms", normalized_weights, "<f8"),
        ("mac_counts", mac_counts, "<i8"),
        ("propagation_gains", propagation_gains, "<f8"),
    )
    for name, values, dtype in arrays:
        array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
        digest.update(name.encode("ascii"))
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_published_profile(
    plan: QdqProfileBuildPlan,
    state_contract: Mapping[str, Any],
) -> ProfileBuildOutcome:
    output = plan.output_dir
    required = {
        "build_manifest.snapshot.json",
        "profile_arrays.npz",
        "profile_audit.json",
        "profile_metadata.json",
    }
    checksum_path = output / "checksums.json"
    checksums = _read_json(checksum_path, field="published profile checksums")
    files = checksums.get("files")
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.name != "checksums.json"
    }
    if (
        checksums.get("algorithm") != "sha256"
        or not isinstance(files, Mapping)
        or set(files) != required
        or actual_files != required
    ):
        raise LabelGenerationError("published profile checksum inventory changed")
    for relative, expected in files.items():
        if not isinstance(expected, str) or _sha256(output / relative) != expected:
            raise LabelGenerationError(
                f"published profile checksum mismatch: {output / relative}"
            )

    snapshot = _read_json(
        output / "build_manifest.snapshot.json",
        field="published profile build snapshot",
    )
    if snapshot.get("requested") != plan.manifest or snapshot.get("resolved") != dict(
        state_contract
    ):
        raise LabelGenerationError(
            "published profile contract differs from the requested build"
        )

    metadata = _read_json(
        output / "profile_metadata.json", field="published profile metadata"
    )
    audit = _read_json(output / "profile_audit.json", field="published profile audit")
    model_contract = metadata.get("model")
    input_contract = metadata.get("input_protocol")
    source_contract = metadata.get("source_real_label_manifest")
    if (
        metadata.get("schema_version") != PROFILE_METADATA_SCHEMA
        or metadata.get("network_id") != plan.network_id
        or metadata.get("dataset_id") != plan.dataset_id
        or metadata.get("domain") != DOMAIN
        or metadata.get("profile_evidence") != "measured"
        or not isinstance(model_contract, Mapping)
        or model_contract.get("sha256") != plan.label_plan.model_sha256
        or not isinstance(input_contract, Mapping)
        or input_contract.get("sha256") != plan.label_plan.dataset_sha256
        or not isinstance(source_contract, Mapping)
        or source_contract.get("sha256") != plan.source_label_manifest_sha256
        or audit.get("schema_version") != PROFILE_AUDIT_SCHEMA
        or audit.get("build_fingerprint") != state_contract["build_fingerprint"]
        or audit.get("histogram_sample_ids")
        != state_contract["histogram_sample_ids"]
        or audit.get("propagation_sample_ids")
        != state_contract["propagation_sample_ids"]
    ):
        raise LabelGenerationError("published profile provenance contract changed")

    try:
        with np.load(output / "profile_arrays.npz", allow_pickle=False) as archive:
            if set(archive.files) != {
                "layer_ids",
                "activation_histograms",
                "weight_histograms",
                "mac_counts",
                "propagation_gains",
            }:
                raise LabelGenerationError("published profile array inventory changed")
            layer_ids = tuple(str(value) for value in archive["layer_ids"].tolist())
            activation = np.asarray(archive["activation_histograms"])
            weights = np.asarray(archive["weight_histograms"])
            mac_counts = np.asarray(archive["mac_counts"])
            gains = np.asarray(archive["propagation_gains"])
    except (OSError, KeyError, ValueError) as exc:
        raise LabelGenerationError(f"invalid published profile arrays: {exc}") from exc

    layer_count = len(layer_ids)
    audit_layers = audit.get("layers")
    if (
        layer_count < 1
        or int(metadata.get("layer_count", -1)) != layer_count
        or activation.dtype != np.float64
        or weights.dtype != np.float64
        or mac_counts.dtype != np.int64
        or gains.dtype != np.float64
        or activation.shape != (layer_count, 256)
        or weights.shape != (layer_count, 256)
        or mac_counts.shape != (layer_count,)
        or gains.shape != (layer_count,)
        or not isinstance(audit_layers, list)
        or not all(isinstance(row, Mapping) for row in audit_layers)
        or tuple(str(row.get("node_name")) for row in audit_layers) != layer_ids
        or not all(np.isfinite(values).all() for values in (activation, weights, gains))
        or np.any(activation.sum(axis=1) <= 0)
        or np.any(weights.sum(axis=1) <= 0)
        or np.any(mac_counts <= 0)
        or np.any(gains < 0)
    ):
        raise LabelGenerationError("published profile arrays have an invalid contract")
    digest = _profile_digest(
        network_id=plan.network_id,
        dataset_id=plan.dataset_id,
        layer_ids=layer_ids,
        activation_histograms=activation,
        weight_histograms=weights,
        mac_counts=mac_counts,
        propagation_gains=gains,
    )
    if metadata.get("profile_digest") != digest or audit.get("profile_digest") != digest:
        raise LabelGenerationError("published profile digest mismatch")
    return ProfileBuildOutcome(
        output_dir=output,
        profile_metadata=output / "profile_metadata.json",
        profile_arrays=output / "profile_arrays.npz",
        layer_count=layer_count,
        histogram_sample_count=plan.histogram_sample_count,
        propagation_sample_count=plan.propagation_sample_count,
        profile_sha256=digest,
    )


def _publish_profile(
    plan: QdqProfileBuildPlan,
    state_contract: Mapping[str, Any],
    histograms: _HistogramResult,
    gain_rows: Sequence[Mapping[str, Any]],
) -> ProfileBuildOutcome:
    output = plan.output_dir
    if output.exists():
        raise LabelGenerationError(f"output_dir already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.publishing-", dir=output.parent)
    )
    try:
        propagation_gains = np.asarray(
            [float(row["propagation_gain"]) for row in gain_rows],
            dtype=np.float64,
        )
        if not np.any(propagation_gains > 0):
            raise LabelGenerationError("all measured propagation gains are zero")
        profile_digest = _profile_digest(
            network_id=plan.network_id,
            dataset_id=plan.dataset_id,
            layer_ids=histograms.layer_ids,
            activation_histograms=histograms.activation_histograms,
            weight_histograms=histograms.weight_histograms,
            mac_counts=histograms.mac_counts,
            propagation_gains=propagation_gains,
        )
        metadata = {
            "schema_version": PROFILE_METADATA_SCHEMA,
            "network_id": plan.network_id,
            "dataset_id": plan.dataset_id,
            "domain": DOMAIN,
            "profile_evidence": "measured",
            "error_convention": "approx_minus_exact",
            "layer_count": len(histograms.layer_ids),
            "profile_digest": profile_digest,
            "model": {
                "path": str(plan.label_plan.model_path),
                "sha256": plan.label_plan.model_sha256,
                "input_name": plan.label_plan.model_input_name,
            },
            "input_protocol": {
                "path": str(plan.label_plan.dataset_path),
                "sha256": plan.label_plan.dataset_sha256,
                "preprocessing_id": plan.label_plan.preprocessing_id,
                "histogram_sample_count": plan.histogram_sample_count,
                "propagation_sample_count": plan.propagation_sample_count,
                "sample_seed": plan.sample_seed,
                "sample_selection": "lowest_sha256(seed\\0sample_id)_v1",
            },
            "histogram_provenance": {
                "kind": "measured_quantized_operands",
                "activation_source": "raw INT8 QuantizeLinear outputs on frozen inputs",
                "weight_source": "frozen INT8 ONNX initializers",
                "padding_policy": "exclude semantic-zero padded multiplications",
                "spatial_policy": "weight operands by exact Conv reuse counts",
                "selected_conv_cuda_placement_verified": True,
                "selected_conv_profile_event_count": histograms.placement_event_count,
            },
            "propagation_gain_provenance": {
                "kind": "measured",
                "method": PROPAGATION_METHOD,
                "probe_magnitude": plan.probe_magnitude,
                "normalization": "central_logit_rms/(valid_mac_count*independent_nonzero_operand_mass)",
                "selected_custom_op_cuda_placement_verified": True,
            },
            "source_real_label_manifest": {
                "path": str(plan.source_label_manifest),
                "sha256": plan.source_label_manifest_sha256,
            },
        }
        _write_json_atomic(staging / "profile_metadata.json", metadata)
        _write_npz_atomic(
            staging / "profile_arrays.npz",
            layer_ids=np.asarray(histograms.layer_ids, dtype="U"),
            activation_histograms=np.asarray(
                histograms.activation_histograms, dtype=np.float64
            ),
            weight_histograms=np.asarray(
                histograms.weight_histograms, dtype=np.float64
            ),
            mac_counts=np.asarray(histograms.mac_counts, dtype=np.int64),
            propagation_gains=propagation_gains,
        )
        audit = {
            "schema_version": PROFILE_AUDIT_SCHEMA,
            "profile_digest": profile_digest,
            "build_fingerprint": state_contract["build_fingerprint"],
            "histogram_sample_ids": state_contract["histogram_sample_ids"],
            "propagation_sample_ids": state_contract["propagation_sample_ids"],
            "layers": [
                {
                    **dict(row),
                    "activation_shape": list(histograms.activation_shapes[index]),
                    "activation_zero_mass": float(
                        histograms.activation_histograms[index, ZERO_CODE_INDEX]
                        / histograms.activation_histograms[index].sum()
                    ),
                    "weight_zero_mass": float(
                        histograms.weight_histograms[index, ZERO_CODE_INDEX]
                        / histograms.weight_histograms[index].sum()
                    ),
                }
                for index, row in enumerate(gain_rows)
            ],
        }
        _write_json_atomic(staging / "profile_audit.json", audit)
        _write_json_atomic(
            staging / "build_manifest.snapshot.json",
            {
                "requested": plan.manifest,
                "resolved": dict(state_contract),
            },
        )
        files = {
            path.relative_to(staging).as_posix(): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        _write_json_atomic(
            staging / "checksums.json",
            {"algorithm": "sha256", "files": files},
        )
        os.replace(staging, output)
        return ProfileBuildOutcome(
            output_dir=output,
            profile_metadata=output / "profile_metadata.json",
            profile_arrays=output / "profile_arrays.npz",
            layer_count=len(histograms.layer_ids),
            histogram_sample_count=plan.histogram_sample_count,
            propagation_sample_count=plan.propagation_sample_count,
            profile_sha256=profile_digest,
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def build_qdq_application_profile(
    path: str | Path,
    *,
    resume: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ProfileBuildOutcome:
    report = progress if progress is not None else lambda _message: None
    plan = load_qdq_profile_build_plan(path, allow_existing_output=resume)
    inputs, _labels, sample_ids = _load_samples(plan.label_plan)
    histogram_indices = _selected_sample_indices(
        sample_ids,
        count=plan.histogram_sample_count,
        seed=plan.sample_seed,
    )
    propagation_indices = histogram_indices[: plan.propagation_sample_count]
    contract = _state_contract(
        plan,
        histogram_indices,
        propagation_indices,
        sample_ids,
    )

    state_dir = plan.output_dir.with_name(f".{plan.output_dir.name}.profile-state")
    if plan.output_dir.exists():
        if state_dir.exists():
            raise LabelGenerationError(
                "published profile and checkpoint state both exist; manual audit required"
            )
        outcome = _validate_published_profile(plan, contract)
        report(f"reused verified published profile {plan.output_dir}")
        return outcome
    state_manifest = state_dir / "state_contract.json"
    if state_dir.exists():
        if not resume:
            raise LabelGenerationError(
                f"profile checkpoint state already exists; pass --resume: {state_dir}"
            )
        current = _read_json(state_manifest, field="profile checkpoint contract")
        if current != contract:
            raise LabelGenerationError(
                "profile checkpoint contract differs from the requested build"
            )
        report(f"resuming profile state {state_dir}")
    else:
        state_dir.mkdir(parents=True, exist_ok=False)
        _write_json_atomic(state_manifest, contract)

    model, specs = _load_layer_specs(plan)
    histogram_checkpoint = state_dir / "histograms.npz"
    if histogram_checkpoint.is_file():
        histograms = _load_histogram_checkpoint(
            histogram_checkpoint,
            expected_layer_ids=tuple(spec.node_name for spec in specs),
        )
        report("reused verified activation/weight histogram checkpoint")
    else:
        histogram_inputs = np.ascontiguousarray(
            inputs[histogram_indices], dtype=np.float32
        )
        histograms = _collect_histograms(
            plan,
            histogram_inputs,
            specs,
            model,
            progress=report,
        )
        _write_npz_atomic(
            histogram_checkpoint,
            layer_ids=np.asarray(histograms.layer_ids, dtype="U"),
            activation_histograms=histograms.activation_histograms,
            weight_histograms=histograms.weight_histograms,
            mac_counts=histograms.mac_counts,
            activation_shapes=np.asarray(histograms.activation_shapes, dtype=np.int64),
            placement_event_count=np.asarray(
                histograms.placement_event_count, dtype=np.int64
            ),
        )
        report("published activation/weight histogram checkpoint")

    propagation_inputs = np.ascontiguousarray(
        inputs[propagation_indices], dtype=np.float32
    )
    gain_dir = state_dir / "propagation"
    gain_dir.mkdir(exist_ok=True)
    gain_rows: list[dict[str, Any]] = []
    for index, node_name in enumerate(histograms.layer_ids):
        checkpoint = gain_dir / f"{index:03d}.json"
        if checkpoint.is_file():
            row = _validate_gain_checkpoint(
                _read_json(checkpoint, field="propagation checkpoint"),
                node_name=node_name,
                mac_count=int(histograms.mac_counts[index]),
                sample_count=plan.propagation_sample_count,
                probe_magnitude=plan.probe_magnitude,
            )
            report(
                f"propagation {index + 1}/{len(histograms.layer_ids)} reused: {node_name}"
            )
        else:
            row = _measure_layer_gain(
                plan,
                node_name,
                propagation_inputs,
                mac_count=int(histograms.mac_counts[index]),
                activation_histogram=histograms.activation_histograms[index],
                weight_histogram=histograms.weight_histograms[index],
            )
            _write_json_atomic(checkpoint, row)
            report(
                f"propagation {index + 1}/{len(histograms.layer_ids)} measured: {node_name}"
            )
        gain_rows.append(row)

    outcome = _publish_profile(plan, contract, histograms, gain_rows)
    shutil.rmtree(state_dir)
    return outcome
