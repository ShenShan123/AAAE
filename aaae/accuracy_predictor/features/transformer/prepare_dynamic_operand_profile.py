#!/usr/bin/env python
"""Profile dynamic-B16 Swin operand pairs without running candidate designs.

The generated bank is application-only state. It observes quantized operands
from the clean dynamic Swin model, applies the exact affine semantic-zero mask,
and groups images by the frozen clean top-1 margin protocol. No approximate
multiplier LUT or per-design runtime observation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SCRIPT_ROOT = Path(__file__).resolve().parent
from aaae.paths import END_TO_END_ROOT as PROJECT_ROOT, EA2_ROOT, OUTPUT_ROOT

from aaae.accuracy_predictor.features.transformer.common import (
    FORMAL_CACHE_ROOT,
    SWIN_72_DATASET_ROOT,
    sha256_array,
    sha256_file,
    write_json,
)
from aaae.inference_runtime.transformer.ort_session import create_cuda_session

DEFAULT_MODEL = (
    EA2_ROOT
    / "ai_models"
    / "onnx_models"
    / "classification"
    / "swin_tiny_patch4_window7_224_imagenet1k_int8_dynamic.onnx"
)
DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "resources"
    / "data"
    / "swin_tiny_s8s8"
    / "node_registry"
    / "node_registry.json"
)
DEFAULT_INPUTS = SWIN_72_DATASET_ROOT / "shared" / "union_inputs.npy"
DEFAULT_EXACT_BASELINE = (
    SWIN_72_DATASET_ROOT / "shared" / "canonical_exact_baseline.npz"
)
DEFAULT_EXACT_META = SWIN_72_DATASET_ROOT / "shared" / "true1000_exact_meta.npz"
DEFAULT_INSTRUMENTED_MODEL = FORMAL_CACHE_ROOT / "swin_dynamic_b16_qcodes.onnx"
DEFAULT_OUTPUT = FORMAL_CACHE_ROOT / "dynamic_b16_operand_profile_top1.npz"


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attribute_int(node: onnx.NodeProto, name: str) -> int | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return int(helper.get_attribute_value(attribute))
    return None


@dataclass(frozen=True)
class QuantizedInput:
    name: str
    static_value: np.ndarray | None
    scale: np.ndarray
    zero_point: int
    axis: int | None

    @property
    def is_static(self) -> bool:
        return self.static_value is not None


@dataclass(frozen=True)
class NodeOperands:
    node_id: str
    input_a: QuantizedInput
    input_b: QuantizedInput


def _uniform_zero_point(value: np.ndarray, *, node_id: str, slot: str) -> int:
    flattened = np.asarray(value).reshape(-1)
    if flattened.size < 1 or not np.all(flattened == flattened[0]):
        raise ValueError(
            f"{node_id} input {slot} uses a non-uniform zero point; "
            "the current S8 CUDA semantic-zero ABI requires a uniform value"
        )
    return int(flattened[0])


def _resolve_quantized_input(
    tensor_name: str,
    *,
    node_id: str,
    slot: str,
    producer: dict[str, onnx.NodeProto],
    initializers: dict[str, np.ndarray],
) -> QuantizedInput:
    dq = producer.get(str(tensor_name))
    if dq is None or dq.op_type != "DequantizeLinear" or len(dq.input) < 3:
        raise ValueError(f"{node_id} input {slot} is not produced by DequantizeLinear")
    quantized_name, scale_name, zero_name = map(str, dq.input[:3])
    if scale_name not in initializers or zero_name not in initializers:
        raise ValueError(f"{node_id} input {slot} has dynamic quantization parameters")
    scale = np.asarray(initializers[scale_name], dtype=np.float64).reshape(-1)
    zero_raw = np.asarray(initializers[zero_name])
    if zero_raw.dtype != np.int8:
        raise ValueError(
            f"{node_id} input {slot} has dtype {zero_raw.dtype}; expected signed int8"
        )
    if scale.size < 1 or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError(f"{node_id} input {slot} has invalid quantization scales")
    static = initializers.get(quantized_name)
    if static is not None:
        static = np.asarray(static)
        if static.dtype != np.int8:
            raise ValueError(
                f"{node_id} input {slot} initializer has dtype {static.dtype}; expected int8"
            )
    return QuantizedInput(
        name=quantized_name,
        static_value=static,
        scale=scale,
        zero_point=_uniform_zero_point(zero_raw, node_id=node_id, slot=slot),
        axis=_attribute_int(dq, "axis"),
    )


def _load_nodes(
    model_path: Path,
    registry_path: Path,
) -> tuple[onnx.ModelProto, list[NodeOperands], list[str]]:
    model = onnx.load(str(model_path), load_external_data=True)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    node_ids = [str(item["node_id"]) for item in registry.get("nodes", [])]
    registry_node_count = int(registry.get("node_count", -1))
    known_registry_contract = (
        registry.get("schema_version") == "swin_tiny_s8s8_node_registry/v1"
        or registry.get("injection_scope_id")
        == "sram_cim_weight_stationary_linear_only_v1"
    )
    if (
        not known_registry_contract
        or registry_node_count < 1
        or len(node_ids) != registry_node_count
        or len(set(node_ids)) != registry_node_count
    ):
        raise ValueError("the signed Swin registry has an invalid node contract")
    node_map = {str(node.name): node for node in model.graph.node if node.name}
    producer = {
        str(output): node
        for node in model.graph.node
        for output in node.output
        if output
    }
    initializers = {
        str(value.name): np.asarray(numpy_helper.to_array(value))
        for value in model.graph.initializer
    }
    nodes: list[NodeOperands] = []
    dynamic_names: set[str] = set()
    for node_id in node_ids:
        node = node_map.get(node_id)
        if node is None or node.op_type != "MatMul" or len(node.input) < 2:
            raise ValueError(f"registered MatMul node is missing: {node_id}")
        input_a = _resolve_quantized_input(
            str(node.input[0]),
            node_id=node_id,
            slot="A",
            producer=producer,
            initializers=initializers,
        )
        input_b = _resolve_quantized_input(
            str(node.input[1]),
            node_id=node_id,
            slot="B",
            producer=producer,
            initializers=initializers,
        )
        if input_a.is_static:
            raise ValueError(f"unexpected static A operand at {node_id}")
        if input_a.scale.size != 1:
            raise ValueError(f"per-axis A scale is unsupported at {node_id}")
        if not input_a.is_static:
            dynamic_names.add(input_a.name)
        if not input_b.is_static:
            if input_b.scale.size != 1:
                raise ValueError(
                    f"dynamic per-axis B scale is unsupported at {node_id}"
                )
            dynamic_names.add(input_b.name)
        elif input_b.scale.size > 1:
            if input_b.axis not in {1, -1}:
                raise ValueError(
                    f"static per-axis B scale at {node_id} must use output-column axis"
                )
            if (
                input_b.static_value is None
                or input_b.scale.size != input_b.static_value.shape[-1]
            ):
                raise ValueError(f"static B scale length is invalid at {node_id}")
        nodes.append(NodeOperands(node_id=node_id, input_a=input_a, input_b=input_b))
    return model, nodes, sorted(dynamic_names)


def _instrumentation_contract(
    model_path: Path,
    registry_path: Path,
    dynamic_names: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "swin_dynamic_qcode_instrumentation_v1",
        "source_model": str(model_path.resolve()),
        "source_model_sha1": _sha1_file(model_path),
        "registry": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
        "dynamic_output_names": dynamic_names,
    }


def _prepare_instrumented_model(
    model: onnx.ModelProto,
    *,
    output_path: Path,
    contract: dict[str, Any],
    overwrite: bool,
) -> Path:
    output_path = output_path.resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() and metadata_path.exists() and not overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing == contract:
            return output_path
        raise ValueError(
            "instrumented model contract changed; pass --overwrite-instrumented"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_outputs = {str(item.name) for item in model.graph.output}
    for name in contract["dynamic_output_names"]:
        if name not in existing_outputs:
            model.graph.output.append(
                helper.make_tensor_value_info(str(name), TensorProto.INT8, None)
            )
            existing_outputs.add(str(name))
    with tempfile.NamedTemporaryFile(
        suffix=".onnx", dir=str(output_path.parent), delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        onnx.save(model, str(temporary))
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_json(metadata_path, contract)
    return output_path


def _top1_groups(exact_meta_path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    with np.load(exact_meta_path, allow_pickle=False) as exact:
        margin = np.asarray(exact["clean_margin_top1"], dtype=np.float32)
        correct = np.asarray(exact["exact_top1_correct_bits"], dtype=np.uint8)
    if margin.shape != (1000,) or correct.shape != (1000,):
        raise ValueError("true1000 exact metadata has an invalid shape")
    if not np.all((correct == 0) | (correct == 1)):
        raise ValueError("exact top-1 correctness metadata is not binary")
    correct_rows = np.flatnonzero(correct == 1)
    wrong_rows = np.flatnonzero(correct == 0)
    ordered = correct_rows[np.argsort(margin[correct_rows], kind="stable")]
    loss_groups = [
        np.asarray(group, dtype=np.int64) for group in np.array_split(ordered, 8)
    ]
    names = [
        "all",
        *[f"loss_margin_bin_{index}" for index in range(8)],
        "recovery_wrong",
    ]
    assignment = np.empty(1000, dtype=np.int64)
    for index, rows in enumerate(loss_groups, start=1):
        assignment[rows] = index
    assignment[wrong_rows] = 9
    return (
        assignment,
        names,
        {
            "loss_group_sizes": [len(group) for group in loss_groups],
            "recovery_group_size": len(wrong_rows),
            "exact_correct_count": len(correct_rows),
            "clean_margin_sha256": sha256_array(margin),
            "exact_correctness_sha256": sha256_array(correct),
        },
    )


def _dynamic_subset(
    value: np.ndarray,
    local_rows: np.ndarray,
    *,
    batch_size: int,
    tensor_name: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int8 or array.ndim < 2:
        raise ValueError(
            f"dynamic q-code tensor {tensor_name} has {array.shape} {array.dtype}"
        )
    if array.shape[0] % batch_size != 0:
        raise ValueError(
            f"dynamic q-code tensor {tensor_name} first dimension {array.shape[0]} "
            f"is not divisible by batch size {batch_size}"
        )
    factor = array.shape[0] // batch_size
    reshaped = array.reshape(batch_size, factor, *array.shape[1:])
    return np.asarray(reshaped[local_rows])


def _code_bins(value: np.ndarray, zero_point: int, *, high: bool) -> np.ndarray:
    raw = np.asarray(value, dtype=np.int16)
    index = raw + 128
    result = (index >> 4) if high else (index & 15)
    return np.where(raw == int(zero_point), -1, result).astype(np.int16, copy=False)


def _counts_last_reduction(ids: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.count_nonzero(ids == code, axis=-1) for code in range(16)],
        axis=-1,
    ).astype(np.float64, copy=False)


def _static_b_counts(
    operand: QuantizedInput,
    *,
    high: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if operand.static_value is None:
        raise ValueError("static B counts requested for a dynamic operand")
    value = np.asarray(operand.static_value)
    if value.ndim != 2:
        raise ValueError(f"static MatMul B must be rank two, got {value.shape}")
    ids = _code_bins(value, operand.zero_point, high=high)
    count = _counts_last_reduction(ids)
    if operand.scale.size == 1:
        weighted = count * float(operand.scale[0])
    else:
        scale = operand.scale.reshape(1, -1)
        weighted = np.stack(
            [np.sum((ids == code) * scale, axis=-1) for code in range(16)],
            axis=-1,
        ).astype(np.float64, copy=False)
    return count, weighted


def _pair_histogram(
    node: NodeOperands,
    input_a: np.ndarray,
    input_b: np.ndarray,
    *,
    high: bool,
    static_b_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(input_a)
    b = np.asarray(input_b)
    if a.ndim < 2 or b.ndim < 2 or a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"MatMul operand shape mismatch at {node.node_id}: A={a.shape}, B={b.shape}"
        )
    a_ids = _code_bins(a, node.input_a.zero_point, high=high)
    a_scale = float(node.input_a.scale[0])
    if node.input_b.is_static:
        # Static B is shared across every image and broadcast prefix. Aggregate
        # A over all prefix and M rows, then contract through K.
        k_dim = a.shape[-1]
        a_flat = a_ids.reshape(-1, a.shape[-2], k_dim)
        a_count = np.stack(
            [np.count_nonzero(a_flat == code, axis=(0, 1)) for code in range(16)],
            axis=-1,
        ).astype(np.float64, copy=False)
        b_count, b_weight = static_b_cache[(node.node_id, high)]
        count_hist = a_count.T @ b_count
        weighted_hist = a_scale * (a_count.T @ b_weight)
        return count_hist, weighted_hist

    prefix = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    a_view = np.broadcast_to(a_ids, (*prefix, *a.shape[-2:])).reshape(
        -1, a.shape[-2], a.shape[-1]
    )
    b_ids = _code_bins(b, node.input_b.zero_point, high=high)
    b_view = np.broadcast_to(b_ids, (*prefix, *b.shape[-2:])).reshape(
        -1, b.shape[-2], b.shape[-1]
    )
    # [prefix, K, code]. Counts preserve matched attention windows/heads.
    a_count = _counts_last_reduction(np.swapaxes(a_view, -1, -2))
    b_count = _counts_last_reduction(b_view)
    count_hist = np.einsum("pkc,pkd->cd", a_count, b_count, optimize=True)
    weighted_hist = count_hist * a_scale * float(node.input_b.scale[0])
    return count_hist, weighted_hist


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    metadata_path = output.with_suffix(output.suffix + ".json")
    if (output.exists() or metadata_path.exists()) and not args.overwrite:
        raise FileExistsError("dynamic operand profile exists; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.model.expanduser().resolve()
    registry_path = args.node_registry.expanduser().resolve()
    model, nodes, dynamic_names = _load_nodes(model_path, registry_path)
    if len(dynamic_names) != 96:
        raise ValueError(
            f"expected 96 unique dynamic q-code tensors, got {len(dynamic_names)}"
        )
    instrumentation = _instrumentation_contract(
        model_path, registry_path, dynamic_names
    )
    instrumented = _prepare_instrumented_model(
        model,
        output_path=args.instrumented_model.expanduser(),
        contract=instrumentation,
        overwrite=args.overwrite_instrumented,
    )

    inputs = np.load(
        args.inputs.expanduser().resolve(), mmap_mode="r", allow_pickle=False
    )
    if inputs.shape != (1486, 3, 224, 224) or inputs.dtype != np.float32:
        raise ValueError(
            f"frozen union input cache is invalid: {inputs.shape} {inputs.dtype}"
        )
    with np.load(
        args.exact_baseline.expanduser().resolve(), allow_pickle=False
    ) as exact:
        positions = np.asarray(exact["true1000_positions"], dtype=np.int64)
    if positions.shape != (1000,) or len(np.unique(positions)) != 1000:
        raise ValueError("true1000 union positions are invalid")
    group_assignment, group_names, group_metadata = _top1_groups(
        args.exact_meta.expanduser().resolve()
    )
    image_count = min(int(args.image_limit), 1000) if args.image_limit else 1000
    positions = positions[:image_count]
    group_assignment = group_assignment[:image_count]

    group_count = len(group_names)
    node_count = len(nodes)
    low_count = np.zeros((group_count, node_count, 16, 16), dtype=np.float64)
    high_count = np.zeros_like(low_count)
    low_weight = np.zeros_like(low_count)
    high_weight = np.zeros_like(low_count)
    image_counts = np.zeros(group_count, dtype=np.int64)
    image_counts[0] = image_count
    for group in range(1, group_count):
        image_counts[group] = int(np.sum(group_assignment == group))

    static_b_cache: dict[tuple[str, bool], tuple[np.ndarray, np.ndarray]] = {}
    for node in nodes:
        if node.input_b.is_static:
            static_b_cache[(node.node_id, False)] = _static_b_counts(
                node.input_b, high=False
            )
            static_b_cache[(node.node_id, True)] = _static_b_counts(
                node.input_b, high=True
            )

    session, provider = create_cuda_session(
        instrumented,
        device_id=int(args.device_id),
        fallback_parallelism=1,
        session_profile="throughput",
    )
    input_name = str(session.get_inputs()[0].name)
    available_outputs = {str(item.name) for item in session.get_outputs()}
    missing_outputs = sorted(set(dynamic_names).difference(available_outputs))
    if missing_outputs:
        session.close()
        raise ValueError(
            f"instrumented session is missing outputs: {missing_outputs[:3]}"
        )

    started = time.time()
    try:
        for start in range(0, image_count, int(args.batch_size)):
            stop = min(start + int(args.batch_size), image_count)
            batch = np.ascontiguousarray(
                inputs[positions[start:stop]], dtype=np.float32
            )
            values = session.run(dynamic_names, {input_name: batch})
            dynamic = {
                name: np.asarray(value)
                for name, value in zip(dynamic_names, values, strict=True)
            }
            batch_groups = group_assignment[start:stop]
            group_rows: dict[int, np.ndarray] = {
                0: np.arange(stop - start, dtype=np.int64)
            }
            for group in np.unique(batch_groups):
                group_rows[int(group)] = np.flatnonzero(batch_groups == group)
            subset_cache: dict[tuple[str, int], np.ndarray] = {}

            def operand_value(
                operand: QuantizedInput,
                group: int,
                *,
                current_subset_cache: dict[tuple[str, int], np.ndarray] = subset_cache,
                current_dynamic: dict[str, np.ndarray] = dynamic,
                current_group_rows: dict[int, np.ndarray] = group_rows,
                current_batch_size: int = stop - start,
            ) -> np.ndarray:
                if operand.static_value is not None:
                    return operand.static_value
                key = (operand.name, group)
                if key not in current_subset_cache:
                    current_subset_cache[key] = _dynamic_subset(
                        current_dynamic[operand.name],
                        current_group_rows[group],
                        batch_size=current_batch_size,
                        tensor_name=operand.name,
                    )
                return current_subset_cache[key]

            for group, local_rows in group_rows.items():
                if len(local_rows) == 0:
                    continue
                for node_index, node in enumerate(nodes):
                    a_value = operand_value(node.input_a, group)
                    b_value = operand_value(node.input_b, group)
                    low, low_scaled = _pair_histogram(
                        node,
                        a_value,
                        b_value,
                        high=False,
                        static_b_cache=static_b_cache,
                    )
                    high, high_scaled = _pair_histogram(
                        node,
                        a_value,
                        b_value,
                        high=True,
                        static_b_cache=static_b_cache,
                    )
                    low_count[group, node_index] += low
                    high_count[group, node_index] += high
                    low_weight[group, node_index] += low_scaled
                    high_weight[group, node_index] += high_scaled
            print(
                json.dumps(
                    {
                        "profiled_images": stop,
                        "total_images": image_count,
                        "elapsed_seconds": time.time() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        session.close()

    if not np.allclose(
        low_count.sum(axis=(2, 3)), high_count.sum(axis=(2, 3)), rtol=0.0, atol=0.5
    ):
        raise RuntimeError(
            "low/high q-code histograms do not conserve semantic MAC counts"
        )
    if np.any(~np.isfinite(low_weight)) or np.any(~np.isfinite(high_weight)):
        raise RuntimeError("scale-weighted operand profile contains non-finite values")
    rounded_low = np.rint(low_count).astype(np.uint64)
    rounded_high = np.rint(high_count).astype(np.uint64)
    temporary = output.with_name(output.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            group_names=np.asarray(group_names),
            group_image_counts=image_counts,
            node_ids=np.asarray([node.node_id for node in nodes]),
            q_input_a_names=np.asarray([node.input_a.name for node in nodes]),
            q_input_b_names=np.asarray([node.input_b.name for node in nodes]),
            input_a_zero_points=np.asarray(
                [node.input_a.zero_point for node in nodes], dtype=np.int16
            ),
            input_b_zero_points=np.asarray(
                [node.input_b.zero_point for node in nodes], dtype=np.int16
            ),
            input_a_scales=np.asarray(
                [float(node.input_a.scale[0]) for node in nodes], dtype=np.float32
            ),
            input_b_scale_min=np.asarray(
                [float(node.input_b.scale.min()) for node in nodes], dtype=np.float32
            ),
            input_b_scale_max=np.asarray(
                [float(node.input_b.scale.max()) for node in nodes], dtype=np.float32
            ),
            input_b_is_static=np.asarray(
                [node.input_b.is_static for node in nodes], dtype=np.bool_
            ),
            low4_pair_count=rounded_low,
            high4_pair_count=rounded_high,
            low4_scale_weight=low_weight,
            high4_scale_weight=high_weight,
        )
    os.replace(temporary, output)
    result: dict[str, Any] = {
        "schema_version": "swin_dynamic_b16_operand_profile_v1",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "model": str(model_path),
        "model_sha1": _sha1_file(model_path),
        "instrumented_model": str(instrumented),
        "instrumentation_contract": instrumentation,
        "provider": provider,
        "batch_size": int(args.batch_size),
        "image_count": image_count,
        "node_count": node_count,
        "dynamic_qcode_output_count": len(dynamic_names),
        "profile_target": "top1_clean_margin_loss_bins_plus_exact_wrong_recovery",
        "group_names": group_names,
        "group_image_counts": image_counts.tolist(),
        "group_metadata": group_metadata,
        "semantic_zero_mask": "exclude qA == zpA or qB == zpB before histogramming",
        "lut_domain": "signed_int8_raw_code_plus_128",
        "candidate_multiplier_inference_count": 0,
        "candidate_multiplier_luts_loaded": False,
        "per_design_runtime_observations_used": False,
        "deployment_reuse": "frozen application profile shared by every new multiplier",
        "inputs": str(args.inputs.expanduser().resolve()),
        "inputs_sha256": sha256_file(args.inputs.expanduser().resolve()),
        "true1000_positions_sha256": sha256_array(positions),
        "exact_meta": str(args.exact_meta.expanduser().resolve()),
        "exact_meta_sha256": sha256_file(args.exact_meta.expanduser().resolve()),
        "elapsed_seconds": time.time() - started,
    }
    write_json(metadata_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    result.add_argument("--node-registry", type=Path, default=DEFAULT_REGISTRY)
    result.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    result.add_argument("--exact-baseline", type=Path, default=DEFAULT_EXACT_BASELINE)
    result.add_argument("--exact-meta", type=Path, default=DEFAULT_EXACT_META)
    result.add_argument(
        "--instrumented-model", type=Path, default=DEFAULT_INSTRUMENTED_MODEL
    )
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--device-id", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--image-limit", type=int, default=0)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--overwrite-instrumented", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.batch_size < 2:
        raise SystemExit("--batch-size must be at least 2 for the dynamic-B16 profile")
    if args.image_limit < 0 or args.image_limit > 1000:
        raise SystemExit("--image-limit must be zero or in [1, 1000]")
    print(json.dumps(build(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
