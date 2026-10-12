from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .errors import LabelGenerationError


LUT_DOMAIN = "s8s8_sign_magnitude"
LUT_INDEX_ENCODING = "signed_code_plus_128"
DEFAULT_LUT_INPUT_NAME = "cnn_s8s8_residual_lut"


@dataclass(frozen=True)
class ConvGraphExport:
    output_path: Path
    replaced_node_names: tuple[str, ...]
    lut_input_name: str


@dataclass(frozen=True)
class _DequantInfo:
    raw_code_name: str
    scale: np.ndarray
    zero_point: np.ndarray
    axis: int | None


def _dequant_info(model: onnx.ModelProto) -> dict[str, _DequantInfo]:
    initializers = {
        str(item.name): np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    infos: dict[str, _DequantInfo] = {}
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 3 or not node.output:
            continue
        scale = initializers.get(str(node.input[1]))
        zero_point = initializers.get(str(node.input[2]))
        if scale is None or zero_point is None:
            continue
        axis = None
        for attribute in node.attribute:
            if attribute.name == "axis":
                axis = int(helper.get_attribute_value(attribute))
                break
        infos[str(node.output[0])] = _DequantInfo(
            raw_code_name=str(node.input[0]),
            scale=np.asarray(scale),
            zero_point=np.asarray(zero_point),
            axis=axis,
        )
    return infos


def _assert_int8_only_qdq(model: onnx.ModelProto) -> None:
    initializers = {
        str(item.name): np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    consumers: dict[str, list[tuple[str, int]]] = {}
    for consumer in model.graph.node:
        for input_index, input_name in enumerate(consumer.input):
            consumers.setdefault(str(input_name), []).append((consumer.op_type, input_index))
    incompatible: list[str] = []
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        if len(node.input) < 3 or not node.input[2]:
            incompatible.append(f"{node.name or node.output[0]}(implicit UINT8 zero point)")
            continue
        zero_point = initializers.get(str(node.input[2]))
        if zero_point is None:
            incompatible.append(f"{node.name or node.output[0]}(dynamic zero point)")
        elif zero_point.dtype == np.int8:
            continue
        elif zero_point.dtype == np.int32 and node.op_type == "DequantizeLinear":
            raw_value = initializers.get(str(node.input[0]))
            output_consumers = consumers.get(str(node.output[0]), [])
            is_bias = (
                raw_value is not None
                and raw_value.dtype == np.int32
                and bool(output_consumers)
                and all(op_type in {"Conv", "Gemm"} and index == 2 for op_type, index in output_consumers)
            )
            if not is_bias:
                incompatible.append(f"{node.name or node.output[0]}(non-bias int32)")
        else:
            incompatible.append(f"{node.name or node.output[0]}({zero_point.dtype})")
    if incompatible:
        raise LabelGenerationError(
            "CNN real-label generation requires a frozen INT8 Q/DQ-only graph; "
            f"incompatible quantizers were found: {incompatible[:5]}"
        )


def _selected_conv_nodes(
    model: onnx.ModelProto,
    node_names: Iterable[str],
) -> tuple[list[onnx.NodeProto], dict[str, _DequantInfo]]:
    requested = tuple(str(name).strip() for name in node_names if str(name).strip())
    if not requested:
        raise LabelGenerationError("injection.node_names must select at least one Conv")
    if len(set(requested)) != len(requested):
        raise LabelGenerationError("injection.node_names contains duplicate names")

    infos = _dequant_info(model)
    _assert_int8_only_qdq(model)
    initializers = {
        str(item.name): np.asarray(numpy_helper.to_array(item))
        for item in model.graph.initializer
    }
    by_name = {str(node.name): node for node in model.graph.node if node.name}
    missing = [name for name in requested if name not in by_name]
    if missing:
        raise LabelGenerationError(f"selected ONNX nodes do not exist: {missing}")

    selected: list[onnx.NodeProto] = []
    for name in requested:
        node = by_name[name]
        if node.op_type != "Conv" or len(node.input) < 2 or len(node.output) != 1:
            raise LabelGenerationError(f"selected node {name!r} is not a supported single-output Conv")
        x_info = infos.get(str(node.input[0]))
        w_info = infos.get(str(node.input[1]))
        if x_info is None or w_info is None:
            raise LabelGenerationError(
                f"selected Conv {name!r} does not have paired initializer-backed INT8 Q/DQ operands"
            )
        if np.dtype(x_info.zero_point.dtype) != np.dtype(np.int8) or np.dtype(w_info.zero_point.dtype) != np.dtype(np.int8):
            raise LabelGenerationError(f"selected Conv {name!r} requires paired INT8 Q/DQ operands")
        if x_info.scale.size != 1 or x_info.zero_point.size != 1:
            raise LabelGenerationError(f"selected Conv {name!r} requires per-tensor activation quantization")
        if not np.isfinite(x_info.scale).all() or np.any(x_info.scale <= 0):
            raise LabelGenerationError(f"selected Conv {name!r} has an invalid activation scale")
        if not np.isfinite(w_info.scale).all() or np.any(w_info.scale <= 0):
            raise LabelGenerationError(f"selected Conv {name!r} has an invalid weight scale")
        raw_weight = initializers.get(w_info.raw_code_name)
        if raw_weight is None or raw_weight.dtype != np.int8 or raw_weight.ndim != 4:
            raise LabelGenerationError(
                f"selected Conv {name!r} requires a frozen rank-4 INT8 weight initializer"
            )
        output_channels = int(raw_weight.shape[0])
        if w_info.scale.size not in {1, output_channels} or w_info.zero_point.size not in {1, output_channels}:
            raise LabelGenerationError(
                f"selected Conv {name!r} weight quantization arrays must be scalar or match "
                f"its {output_channels} output channels"
            )
        if (w_info.scale.size > 1 or w_info.zero_point.size > 1) and w_info.axis != 0:
            raise LabelGenerationError(f"selected Conv {name!r} requires weight quantization axis 0")

        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        auto_pad = attributes.get("auto_pad", b"NOTSET")
        if isinstance(auto_pad, str):
            auto_pad = auto_pad.encode("ascii", errors="strict")
        if auto_pad not in {b"", b"NOTSET"}:
            raise LabelGenerationError(
                f"selected Conv {name!r} uses unsupported auto_pad={auto_pad!r}; use explicit pads"
            )
        for attribute_name, expected_length in (("strides", 2), ("dilations", 2), ("pads", 4)):
            value = attributes.get(attribute_name)
            if value is not None and len(value) != expected_length:
                raise LabelGenerationError(
                    f"selected Conv {name!r} must be a two-dimensional NCHW convolution"
                )
        kernel_shape = attributes.get("kernel_shape")
        if kernel_shape is not None and tuple(int(value) for value in kernel_shape) != tuple(raw_weight.shape[2:]):
            raise LabelGenerationError(
                f"selected Conv {name!r} kernel_shape does not match its frozen INT8 weights"
            )
        group = int(attributes.get("group", 1))
        if group <= 0 or output_channels % group != 0:
            raise LabelGenerationError(f"selected Conv {name!r} has an invalid group attribute")
        selected.append(node)
    return selected, infos


def validate_s8s8_lut_conv_model(model_path: Path, *, node_names: Iterable[str]) -> tuple[str, ...]:
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise LabelGenerationError(f"ONNX model is missing: {path}")
    try:
        model = onnx.load(str(path))
    except Exception as exc:
        raise LabelGenerationError(f"unable to read ONNX model {path}: {exc}") from exc
    selected, _ = _selected_conv_nodes(model, node_names)
    return tuple(str(node.name) for node in selected)


def _fresh_name(preferred: str, occupied: set[str]) -> str:
    candidate = preferred
    suffix = 1
    while candidate in occupied:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    occupied.add(candidate)
    return candidate


def export_s8s8_lut_conv_model(
    model_path: Path,
    output_path: Path,
    *,
    node_names: Iterable[str],
    lut_input_name: str = DEFAULT_LUT_INPUT_NAME,
    residual_abs_bound: int | None = None,
) -> ConvGraphExport:
    source = Path(model_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    try:
        model = onnx.load(str(source))
    except Exception as exc:
        raise LabelGenerationError(f"unable to read ONNX model {source}: {exc}") from exc
    selected, infos = _selected_conv_nodes(model, node_names)
    selected_names = {str(node.name) for node in selected}

    occupied = {
        str(value)
        for node in model.graph.node
        for value in (*node.input, *node.output)
        if value
    }
    occupied.update(str(item.name) for item in model.graph.input)
    occupied.update(str(item.name) for item in model.graph.output)
    occupied.update(str(item.name) for item in model.graph.initializer)
    resolved_lut_name = _fresh_name(str(lut_input_name), occupied)

    new_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if str(node.name) not in selected_names:
            new_nodes.append(node)
            continue

        x_info = infos[str(node.input[0])]
        w_info = infos[str(node.input[1])]
        exact_node = onnx.NodeProto()
        exact_node.CopyFrom(node)
        exact_node.name = f"{node.name}__exact_base"
        exact_output = _fresh_name(f"{node.output[0]}__exact_base", occupied)
        exact_node.output[0] = exact_output
        new_nodes.append(exact_node)

        conv_attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
            if attribute.name in {"dilations", "group", "kernel_shape", "pads", "strides"}
        }
        attributes: dict[str, object] = {
            **conv_attributes,
            "scale_x": float(x_info.scale.reshape(-1)[0]),
            "zp_x": int(x_info.zero_point.reshape(-1)[0]),
            "scale_w": float(w_info.scale.reshape(-1)[0]),
            "zp_w": int(w_info.zero_point.reshape(-1)[0]),
            "lut_domain": LUT_DOMAIN,
            "lut_index_encoding": LUT_INDEX_ENCODING,
            "semantic_zero_mask": 1,
        }
        if residual_abs_bound is not None:
            if isinstance(residual_abs_bound, bool) or int(residual_abs_bound) < 0:
                raise LabelGenerationError("residual_abs_bound must be a nonnegative integer")
            attributes["residual_abs_bound"] = int(residual_abs_bound)
        if w_info.scale.size > 1:
            attributes["scale_w_arr"] = [float(value) for value in w_info.scale.reshape(-1).tolist()]
        if w_info.zero_point.size > 1:
            attributes["zp_w_arr"] = [int(value) for value in w_info.zero_point.reshape(-1).tolist()]
        new_nodes.append(
            helper.make_node(
                "ApproxConvCudaS8Lut",
                inputs=[x_info.raw_code_name, w_info.raw_code_name, exact_output, resolved_lut_name],
                outputs=list(node.output),
                name=f"{node.name}__s8s8_lut",
                domain="approxdomain",
                **attributes,
            )
        )

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.input.append(
        helper.make_tensor_value_info(resolved_lut_name, TensorProto.INT32, [256, 256])
    )
    if not any(item.domain == "approxdomain" for item in model.opset_import):
        model.opset_import.append(helper.make_operatorsetid("approxdomain", 1))

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    try:
        onnx.save(model, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return ConvGraphExport(
        output_path=destination,
        replaced_node_names=tuple(str(node.name) for node in selected),
        lut_input_name=resolved_lut_name,
    )
