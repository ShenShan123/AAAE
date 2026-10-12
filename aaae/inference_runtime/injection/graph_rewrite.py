"""Core implementation extracted from ai_models/src/approx_injection/graph_rewrite.py."""

from __future__ import annotations


from dataclasses import dataclass

from pathlib import Path

from typing import Any, Iterable, Iterator, Mapping

import numpy as np

import onnx

from onnx import helper, numpy_helper

@dataclass(frozen=True)
class _QuantParams:
    scale: float
    zp: int
    qmin: int
    qmax: int
    offset: int


@dataclass(frozen=True)
class _DequantInfo:
    params: _QuantParams
    scale_arr: np.ndarray
    zp_arr: np.ndarray
    axis: int | None
    quantized_input_name: str
    quantized_input_dtype: np.dtype


@dataclass(frozen=True)
class _DequantScope:
    bound_names: frozenset[str]
    info_by_output: Mapping[str, _DequantInfo]


def _default_quant_params() -> _QuantParams:
    return _QuantParams(scale=1.0, zp=0, qmin=0, qmax=255, offset=0)


def _quant_info_from_zp(scale_arr: np.ndarray, zp_arr: np.ndarray) -> _QuantParams:
    scale_val = float(scale_arr.reshape(-1)[0])
    zp_val = int(zp_arr.reshape(-1)[0])
    dtype = zp_arr.dtype
    if np.issubdtype(dtype, np.signedinteger):
        info = np.iinfo(dtype)
        qmin, qmax = int(info.min), int(info.max)
        offset = -qmin
    elif np.issubdtype(dtype, np.unsignedinteger):
        info = np.iinfo(dtype)
        qmin, qmax = int(info.min), int(info.max)
        offset = 0
    else:
        qmin, qmax, offset = 0, 255, 0
    return _QuantParams(scale=scale_val, zp=zp_val, qmin=qmin, qmax=qmax, offset=offset)


def _collect_graph_dequant_info(
    graph: onnx.GraphProto,
    *,
    initializer_map: Mapping[str, np.ndarray],
) -> dict[str, _DequantInfo]:
    out: dict[str, _DequantInfo] = {}
    for node in graph.node:
        if node.op_type != "DequantizeLinear" or len(node.input) < 3 or not node.output:
            continue
        scale_name, zp_name = node.input[1], node.input[2]
        if scale_name not in initializer_map or zp_name not in initializer_map:
            continue
        scale_arr = np.asarray(initializer_map[scale_name])
        zp_arr = np.asarray(initializer_map[zp_name])
        if scale_arr.size == 0 or zp_arr.size == 0:
            continue
        axis = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(helper.get_attribute_value(attr))
                break
        out[str(node.output[0])] = _DequantInfo(
            params=_quant_info_from_zp(scale_arr, zp_arr),
            scale_arr=scale_arr.reshape(-1),
            zp_arr=zp_arr.reshape(-1),
            axis=axis,
            quantized_input_name=str(node.input[0]),
            quantized_input_dtype=np.dtype(zp_arr.dtype),
        )
    return out


def _child_graphs(node: onnx.NodeProto) -> Iterator[onnx.GraphProto]:
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            yield attribute.g
        elif attribute.type == onnx.AttributeProto.GRAPHS:
            yield from attribute.graphs


def _iter_graphs(graph: onnx.GraphProto) -> Iterator[onnx.GraphProto]:
    yield graph
    for node in graph.node:
        for child in _child_graphs(node):
            yield from _iter_graphs(child)


def _occupied_value_names(graph: onnx.GraphProto) -> set[str]:
    occupied: set[str] = set()
    for current in _iter_graphs(graph):
        occupied.update(str(value.name) for value in current.input if value.name)
        occupied.update(str(value.name) for value in current.output if value.name)
        occupied.update(str(value.name) for value in current.initializer if value.name)
        for node in current.node:
            occupied.update(str(value) for value in (*node.input, *node.output) if value)
    return occupied


def _lookup_dequant_info(
    tensor_name: str,
    scopes: tuple[_DequantScope, ...],
) -> _DequantInfo | None:
    name = str(tensor_name)
    for scope in scopes:
        if name in scope.bound_names:
            return scope.info_by_output.get(name)
    return None


def _should_skip_by_fragments(name: str, fragments: tuple[str, ...] | None) -> bool:
    if not fragments:
        return False
    return any(fragment in name for fragment in fragments)


def _fresh_value_name(base_name: str, occupied_names: set[str]) -> str:
    candidate = str(base_name)
    suffix = 1
    while candidate in occupied_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    occupied_names.add(candidate)
    return candidate


def _attach_per_axis_quant_attrs(
    attr_to_add: dict[str, Any],
    prefix: str,
    info: _DequantInfo | None,
    *,
    node_name: str,
) -> None:
    if info is None:
        return
    scale_arr = info.scale_arr.reshape(-1)
    zp_arr = info.zp_arr.reshape(-1)
    if scale_arr.size <= 1 and zp_arr.size <= 1:
        return
    if info.axis is None:
        raise RuntimeError(
            f"MatMul node '{node_name}' uses per-axis quantization on input {prefix.upper()} but DequantizeLinear has no axis"
        )
    attr_to_add[f"axis_{prefix}"] = int(info.axis)
    if scale_arr.size > 1:
        attr_to_add[f"scale_{prefix}_arr"] = [float(value) for value in scale_arr.tolist()]
    # A per-channel RHS scale still needs its axis, but a uniform RHS zero
    # point is semantically a scalar.  Keeping it scalar avoids an otherwise
    # redundant device read for every output column in the tiled S8 path.
    uniform_rhs_zp = prefix == "b" and zp_arr.size > 1 and bool(np.all(zp_arr == zp_arr[0]))
    if zp_arr.size > 1 and not uniform_rhs_zp:
        attr_to_add[f"zp_{prefix}_arr"] = [int(value) for value in zp_arr.astype(np.int64).tolist()]


def _shared_quantized_code_domain(
    a_info: _DequantInfo | None,
    b_info: _DequantInfo | None,
) -> str | None:
    """Return the direct code-tensor domain supported by the CUDA LUT ops."""

    if a_info is None or b_info is None:
        return None
    a_dtype = np.dtype(a_info.quantized_input_dtype)
    b_dtype = np.dtype(b_info.quantized_input_dtype)
    if a_dtype == np.dtype(np.uint8) and b_dtype == np.dtype(np.uint8):
        return "u8"
    if a_dtype == np.dtype(np.int8) and b_dtype == np.dtype(np.int8):
        return "s8"
    return None


def export_matmul_lut_cuda_model(
    *,
    model_path: Path,
    output_path: Path,
    lut_delta_path: Path | None = None,
    lut_input_name_by_node: dict[str, str] | None = None,
    include_node_names: Iterable[str] | None,
    skip_name_fragments: Iterable[str] | None,
    s8_residual_abs_bound: int | None = None,
) -> dict[str, Any]:
    """Export selected MatMuls with a LUT residual custom-op replacement.

    ``s8_residual_abs_bound`` is an opt-in verified bound for dynamic S8S8
    residual LUT entries. It is emitted only on S8 nodes, where the CUDA
    custom op can safely choose a narrower accumulator when its runtime K
    dimension also fits the bound.
    """

    if s8_residual_abs_bound is not None:
        if isinstance(s8_residual_abs_bound, bool) or not isinstance(s8_residual_abs_bound, int):
            raise TypeError("s8_residual_abs_bound must be an integer or None")
        if int(s8_residual_abs_bound) < 0:
            raise ValueError("s8_residual_abs_bound must be non-negative")
    model = onnx.load(str(model_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requested_names = None if include_node_names is None else [str(name) for name in include_node_names]
    if requested_names is not None and len(requested_names) != len(set(requested_names)):
        raise ValueError("include_node_names contains duplicates")
    include_names = None if requested_names is None else set(requested_names)
    if include_names == set():
        raise ValueError("include_node_names must not be empty")
    name_filters = tuple(skip_name_fragments) if skip_name_fragments else None
    use_dynamic_input = lut_input_name_by_node is not None
    lut_input_name_by_node = (
        {str(key): str(value) for key, value in lut_input_name_by_node.items()}
        if lut_input_name_by_node is not None
        else {}
    )
    if use_dynamic_input and lut_delta_path is not None:
        raise ValueError("export_matmul_lut_cuda_model expects either lut_delta_path or lut_input_name_by_node, not both")
    if use_dynamic_input:
        if not lut_input_name_by_node:
            raise ValueError("lut_input_name_by_node must not be empty")
        mapping_names = set(lut_input_name_by_node)
        if include_names is None:
            include_names = mapping_names
        elif mapping_names != include_names:
            raise ValueError("lut_input_name_by_node keys must exactly match include_node_names")
    if not use_dynamic_input:
        if lut_delta_path is None:
            raise ValueError("export_matmul_lut_cuda_model requires lut_delta_path when lut_input_name_by_node is not provided")
        lut_delta_path = Path(lut_delta_path).resolve()
    dynamic_input_names: list[str] = []
    occupied_value_names = _occupied_value_names(model.graph)
    replaced_stats = {"MatMul": 0}
    replaced_node_names: list[str] = []
    quantized_code_nodes = 0
    uint8_code_nodes = 0
    int8_code_nodes = 0

    def rewrite_graph(
        graph: onnx.GraphProto,
        *,
        inherited_initializers: Mapping[str, np.ndarray],
        inherited_dequant_scopes: tuple[_DequantScope, ...],
    ) -> None:
        nonlocal quantized_code_nodes, uint8_code_nodes, int8_code_nodes

        initializer_map = dict(inherited_initializers)
        initializer_map.update(
            {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
        )
        local_dequant_info = _collect_graph_dequant_info(
            graph,
            initializer_map=initializer_map,
        )
        bound_names = {
            str(value.name)
            for value in (*graph.input, *graph.initializer)
            if value.name
        }
        bound_names.update(
            str(output_name)
            for graph_node in graph.node
            for output_name in graph_node.output
            if output_name
        )
        dequant_scopes = (
            _DequantScope(
                bound_names=frozenset(bound_names),
                info_by_output=local_dequant_info,
            ),
            *inherited_dequant_scopes,
        )

        new_nodes: list[onnx.NodeProto] = []
        for node in graph.node:
            for child_graph in _child_graphs(node):
                rewrite_graph(
                    child_graph,
                    inherited_initializers=initializer_map,
                    inherited_dequant_scopes=dequant_scopes,
                )

            node_name = str(node.name or "")
            if node.op_type != "MatMul" or len(node.input) < 2 or not node.output:
                new_nodes.append(node)
                continue
            if include_names is not None and node_name not in include_names:
                new_nodes.append(node)
                continue
            if _should_skip_by_fragments(node_name, name_filters):
                new_nodes.append(node)
                continue

            in_a = str(node.input[0])
            in_b = str(node.input[1])
            a_info = _lookup_dequant_info(in_a, dequant_scopes)
            b_info = _lookup_dequant_info(in_b, dequant_scopes)
            qp_a = a_info.params if a_info is not None else _default_quant_params()
            qp_b = b_info.params if b_info is not None else _default_quant_params()

            attr_to_add: dict[str, Any] = {
                "scale_a": float(qp_a.scale),
                "zp_a": int(qp_a.zp),
                "qmin_a": int(qp_a.qmin),
                "qmax_a": int(qp_a.qmax),
                "scale_b": float(qp_b.scale),
                "zp_b": int(qp_b.zp),
                "qmin_b": int(qp_b.qmin),
                "qmax_b": int(qp_b.qmax),
                "multiplier": "exact",
                "multiplier_kind": "exact",
                # A zero LUT must remain an exact identity across a chain of
                # rewritten MatMuls, including MatMuls inside control flow.
                "base_input": 1,
            }
            _attach_per_axis_quant_attrs(attr_to_add, "a", a_info, node_name=node_name or "(unnamed)")
            _attach_per_axis_quant_attrs(attr_to_add, "b", b_info, node_name=node_name or "(unnamed)")
            base_output_name = _fresh_value_name(
                f"{str(node.output[0])}__approx_exact_base",
                occupied_value_names,
            )
            base_node = onnx.NodeProto()
            base_node.CopyFrom(node)
            base_node.output[0] = base_output_name
            new_nodes.append(base_node)

            # Input slot 2 is the INT32 LUT and slot 3 is the exact float
            # baseline. Slots 4/5 carry native Q/DQ codes when available.
            approx_inputs = [in_a, in_b]
            if use_dynamic_input:
                lut_input_name = lut_input_name_by_node[node_name]
                dynamic_input_names.append(str(lut_input_name))
                approx_inputs.append(str(lut_input_name))
            else:
                attr_to_add["lut_delta_path"] = str(lut_delta_path)
                approx_inputs.append("")
            approx_inputs.append(base_output_name)
            code_domain = _shared_quantized_code_domain(a_info, b_info)
            use_quantized_codes = code_domain is not None
            approx_inputs.append(
                a_info.quantized_input_name if use_quantized_codes and a_info is not None else ""
            )
            approx_inputs.append(
                b_info.quantized_input_name if use_quantized_codes and b_info is not None else ""
            )
            op_type = "ApproxMatMulCudaS8Lut" if code_domain == "s8" else "ApproxMatMulCuda"
            if code_domain == "s8":
                attr_to_add["lut_domain"] = "s8s8_sign_magnitude"
                attr_to_add["semantic_zero_mask"] = 1
                if s8_residual_abs_bound is not None:
                    attr_to_add["s8_residual_abs_bound"] = int(s8_residual_abs_bound)

            approx_node = helper.make_node(
                op_type,
                inputs=approx_inputs,
                outputs=list(node.output),
                name=f"{node_name}_lut_cuda" if node_name else "",
                domain="approxdomain",
                **attr_to_add,
            )
            new_nodes.append(approx_node)
            replaced_stats["MatMul"] += 1
            replaced_node_names.append(node_name)
            quantized_code_nodes += int(use_quantized_codes)
            uint8_code_nodes += int(code_domain == "u8")
            int8_code_nodes += int(code_domain == "s8")

        del graph.node[:]
        graph.node.extend(new_nodes)

    rewrite_graph(
        model.graph,
        inherited_initializers={},
        inherited_dequant_scopes=(),
    )

    if replaced_stats["MatMul"] == 0:
        raise RuntimeError("No matching MatMul nodes found for LUT CUDA replacement")
    if include_names is not None:
        replaced_names = set(replaced_node_names)
        missing = sorted(include_names.difference(replaced_names))
        unexpected = sorted(replaced_names.difference(include_names))
        duplicates = sorted(
            name for name in replaced_names if replaced_node_names.count(name) != 1
        )
        if missing or unexpected or duplicates:
            raise RuntimeError(
                "LUT CUDA replacement did not match the requested unique node set: "
                f"missing={missing}, unexpected={unexpected}, duplicates={duplicates}"
            )
    if use_dynamic_input:
        existing_inputs = {str(value_info.name) for value_info in model.graph.input}
        for input_name in sorted(set(dynamic_input_names)):
            if input_name in existing_inputs:
                continue
            model.graph.input.append(
                helper.make_tensor_value_info(
                    input_name,
                    onnx.TensorProto.INT32,
                    [256, 256],
                )
            )
    if not any(op.domain == "approxdomain" for op in model.opset_import):
        model.opset_import.append(helper.make_operatorsetid("approxdomain", 1))
    onnx.save(model, str(output_path))
    return {
        **replaced_stats,
        "quantized_code_nodes": quantized_code_nodes,
        "uint8_code_nodes": uint8_code_nodes,
        "int8_code_nodes": int8_code_nodes,
        "replaced_node_names": list(replaced_node_names),
        "dynamic_lut_inputs": sorted(set(dynamic_input_names)),
        "lut_input_name_by_node": dict(sorted(lut_input_name_by_node.items())),
    }


