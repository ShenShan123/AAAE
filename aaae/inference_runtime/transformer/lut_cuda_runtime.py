from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import onnx
import onnxruntime as ort
import torch

from aaae.paths import AI_MODELS_ROOT, AI_MODELS_SRC, TRANSFORMER_CACHE_ROOT, TRANSFORMER_ROOT
from aaae.inference_runtime.transformer.dataset_generation.matmul_scope import iter_graph_nodes


PROJECT_ROOT = TRANSFORMER_ROOT

from aaae.inference_runtime.injection.graph_rewrite import export_matmul_lut_cuda_model  # noqa: E402
from aaae.inference_runtime.injection.runtime_build import (  # noqa: E402
    build_lut_cuda_library,
    cuda_architecture_cache_tag,
    resolve_cuda_architectures,
)

from .ort_session import (
    _cuda_provider_config,
    _session_options,
    create_cuda_session,
    ensure_gpu_runtime,
)
from .s8s8_truth_lut_asset import S8S8_LUT_DOMAIN


# This cache is generated at runtime and deliberately does not live under the
# source package or the retired external transformer workspace.
RUNTIME_CACHE_ROOT = TRANSFORMER_CACHE_ROOT / "lut_cuda"
LIB_BUILD_ROOT = RUNTIME_CACHE_ROOT / "lib"
MODEL_CACHE_DIR = RUNTIME_CACHE_ROOT / "models"
LUT_CUDA_REWRITE_ABI = "v10_recursive_control_flow_preoptimized_loop_s8_affine_semantic_zero_mask"
LOOP_PREOPTIMIZATION_ABI = "v1_static_cuda_optimize_then_restore_parity_loop"
U8_LUT_DOMAIN = "u8"
_PARITY_LOOP_NAME = "__ea2_static_b1_batch_loop"
_PARITY_GATHER_NAME = "__ea2_loop_gather_image"
_PARITY_UNSQUEEZE_NAME = "__ea2_loop_restore_b1"
_PARITY_CONDITION_NAME = "__ea2_loop_continue"


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _default_lut_input_name(node_name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(node_name)).strip("_")
    if not sanitized:
        sanitized = "unnamed"
    return f"approx_lut__{sanitized}"


@lru_cache(maxsize=1)
def _lut_custom_op_library() -> Path:
    cuda_architectures = resolve_cuda_architectures()
    architecture_root = LIB_BUILD_ROOT / cuda_architecture_cache_tag(cuda_architectures)
    architecture_root.mkdir(parents=True, exist_ok=True)
    return Path(
        build_lut_cuda_library(
            build_root=architecture_root,
            cuda_architectures=cuda_architectures,
        )
    ).resolve()


def _dynamic_model_cache_key(
    model_path: Path,
    *,
    node_to_lut_input_name: Mapping[str, str],
    lut_domain: str | None,
    s8_residual_abs_bound: int | None,
) -> str:
    payload = {
        "mode": LUT_CUDA_REWRITE_ABI,
        "model_path": str(Path(model_path).resolve()),
        "model_sha": _sha1_file(Path(model_path)),
        "node_to_lut_input_name": dict(sorted((str(k), str(v)) for k, v in node_to_lut_input_name.items())),
        "lut_domain": lut_domain,
        "s8_residual_abs_bound": s8_residual_abs_bound,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _exported_lut_domain(model: onnx.ModelProto) -> str:
    domains = {
        S8S8_LUT_DOMAIN if node.op_type == "ApproxMatMulCudaS8Lut" else U8_LUT_DOMAIN
        for ref in iter_graph_nodes(model)
        for node in (ref.node,)
        if node.domain == "approxdomain" and node.op_type in {"ApproxMatMulCuda", "ApproxMatMulCudaS8Lut"}
    }
    if not domains:
        raise RuntimeError("LUT CUDA graph contains no rewritten ApproxMatMul CUDA nodes")
    if len(domains) != 1:
        raise ValueError(
            "One LUT CUDA runtime cannot combine U8 and S8S8 LUT domains; split the selected MatMul nodes by domain"
        )
    return next(iter(domains))


def _validate_requested_lut_domain(*, actual_lut_domain: str, requested_lut_domain: str | None) -> str:
    if actual_lut_domain == S8S8_LUT_DOMAIN and requested_lut_domain is None:
        raise ValueError(
            "INT8 Q/DQ LUT injection requires lut_domain='s8s8_sign_magnitude' and a signed raw-code residual table "
            "with affine semantic-zero masking. Do not feed the historical U8 truth_tables_uint16 cache to this runtime."
        )
    if requested_lut_domain is not None and requested_lut_domain != actual_lut_domain:
        raise ValueError(
            f"requested lut_domain={requested_lut_domain!r} does not match rewritten model domain {actual_lut_domain!r}"
        )
    return actual_lut_domain


@dataclass(frozen=True)
class _ParityLoopWrapper:
    loop_node: onnx.NodeProto
    body: onnx.GraphProto
    gather_node: onnx.NodeProto
    unsqueeze_node: onnx.NodeProto
    condition_node: onnx.NodeProto
    predictor_input_name: str
    predictor_output_name: str
    body_input_name: str
    body_output_name: str


def _parity_loop_wrapper(model: onnx.ModelProto) -> _ParityLoopWrapper | None:
    matching_loops = [node for node in model.graph.node if node.name == _PARITY_LOOP_NAME]
    if not matching_loops:
        return None
    if len(matching_loops) != 1:
        raise ValueError(f"expected one {_PARITY_LOOP_NAME!r} node, found {len(matching_loops)}")
    loop_node = matching_loops[0]
    if loop_node.op_type != "Loop":
        raise ValueError(f"{_PARITY_LOOP_NAME!r} must be an ONNX Loop")
    body_attributes = [
        attribute
        for attribute in loop_node.attribute
        if attribute.type == onnx.AttributeProto.GRAPH and attribute.name == "body"
    ]
    if len(body_attributes) != 1:
        raise ValueError(f"{_PARITY_LOOP_NAME!r} must have exactly one body graph")
    body = body_attributes[0].g
    if len(model.graph.input) < 1 or len(model.graph.output) != 1:
        raise ValueError("parity Loop wrapper requires at least one graph input and exactly one graph output")
    if len(body.input) != 2 or len(body.output) != 2 or len(body.node) < 4:
        raise ValueError("parity Loop body has an unexpected input/output or node contract")
    gather_node = body.node[0]
    unsqueeze_node = body.node[1]
    condition_node = body.node[-1]
    expected_boundary = (
        (gather_node, _PARITY_GATHER_NAME, "Gather"),
        (unsqueeze_node, _PARITY_UNSQUEEZE_NAME, "Unsqueeze"),
        (condition_node, _PARITY_CONDITION_NAME, "Identity"),
    )
    for node, expected_name, expected_op_type in expected_boundary:
        if node.name != expected_name or node.op_type != expected_op_type:
            raise ValueError(
                f"parity Loop boundary expected {expected_name}/{expected_op_type}, "
                f"got {node.name}/{node.op_type}"
            )
    predictor_input_name = str(model.graph.input[0].name)
    predictor_output_name = str(model.graph.output[0].name)
    if not gather_node.input or str(gather_node.input[0]) != predictor_input_name:
        raise ValueError("parity Loop Gather does not consume the predictor input")
    if not unsqueeze_node.output or not body.output[1].name:
        raise ValueError("parity Loop body is missing its static input or scan output")
    return _ParityLoopWrapper(
        loop_node=loop_node,
        body=body,
        gather_node=gather_node,
        unsqueeze_node=unsqueeze_node,
        condition_node=condition_node,
        predictor_input_name=predictor_input_name,
        predictor_output_name=predictor_output_name,
        body_input_name=str(unsqueeze_node.output[0]),
        body_output_name=str(body.output[1].name),
    )


def _replace_node_value_names(
    nodes: list[onnx.NodeProto],
    replacements: Mapping[str, str],
) -> list[onnx.NodeProto]:
    rewritten = copy.deepcopy(nodes)
    for node in rewritten:
        for names in (node.input, node.output):
            for index, value_name in enumerate(names):
                replacement = replacements.get(str(value_name))
                if replacement is not None:
                    names[index] = replacement
    return rewritten


def _replace_value_info_names(
    values: list[onnx.ValueInfoProto],
    replacements: Mapping[str, str],
) -> list[onnx.ValueInfoProto]:
    rewritten = copy.deepcopy(values)
    for value in rewritten:
        replacement = replacements.get(str(value.name))
        if replacement is not None:
            value.name = replacement
    return rewritten


def _set_static_batch_one(value: onnx.ValueInfoProto) -> onnx.ValueInfoProto:
    rewritten = copy.deepcopy(value)
    shape = rewritten.type.tensor_type.shape
    if not shape.dim:
        raise ValueError(f"{rewritten.name!r} has no leading tensor dimension")
    leading = shape.dim[0]
    leading.ClearField("dim_param")
    leading.dim_value = 1
    return rewritten


def _extract_static_parity_loop_model(
    model: onnx.ModelProto,
    *,
    shape_reference: onnx.ModelProto | None = None,
) -> onnx.ModelProto:
    wrapper = _parity_loop_wrapper(model)
    if wrapper is None:
        raise ValueError("model is not an EA2 static-B1 parity Loop wrapper")
    replacements = {
        wrapper.body_input_name: wrapper.predictor_input_name,
        wrapper.body_output_name: wrapper.predictor_output_name,
    }
    static_nodes = _replace_node_value_names(list(wrapper.body.node[2:-1]), replacements)
    static_graph = onnx.GraphProto()
    static_graph.name = f"{model.graph.name}_extracted_static_b1"
    static_graph.node.extend(static_nodes)
    static_graph.input.extend(
        [
            _set_static_batch_one(model.graph.input[0]),
            *copy.deepcopy(list(model.graph.input[1:])),
        ]
    )
    static_graph.output.extend([_set_static_batch_one(model.graph.output[0])])
    static_graph.initializer.extend(copy.deepcopy(list(model.graph.initializer)))
    static_graph.sparse_initializer.extend(copy.deepcopy(list(model.graph.sparse_initializer)))
    if shape_reference is not None:
        if len(shape_reference.graph.input) != 1 or len(shape_reference.graph.output) != 1:
            raise ValueError("static shape reference must have exactly one graph input and output")
        if (
            str(shape_reference.graph.input[0].name) != wrapper.predictor_input_name
            or str(shape_reference.graph.output[0].name) != wrapper.predictor_output_name
        ):
            raise ValueError("static shape reference input/output names do not match the parity Loop model")
        reference_input_shape = shape_reference.graph.input[0].type.tensor_type.shape
        reference_output_shape = shape_reference.graph.output[0].type.tensor_type.shape
        if (
            not reference_input_shape.dim
            or int(reference_input_shape.dim[0].dim_value) != 1
            or not reference_output_shape.dim
            or int(reference_output_shape.dim[0].dim_value) != 1
        ):
            raise ValueError("static shape reference must have fixed leading batch one")
        static_graph.value_info.extend(copy.deepcopy(list(shape_reference.graph.value_info)))
        static_graph.quantization_annotation.extend(
            copy.deepcopy(list(shape_reference.graph.quantization_annotation))
        )
    extracted = copy.deepcopy(model)
    extracted.graph.CopyFrom(static_graph)
    onnx.checker.check_model(extracted)
    return extracted


def _strip_serialized_cuda_memcpy_nodes(model: onnx.ModelProto) -> int:
    unsupported = [
        node
        for node in model.graph.node
        if not node.domain and node.op_type.startswith("Memcpy") and node.op_type != "MemcpyFromHost"
    ]
    if unsupported:
        raise RuntimeError(
            "CUDA optimized graph contains unsupported serialized transfer nodes: "
            f"{[(node.name, node.op_type) for node in unsupported]}"
        )
    memcpy_nodes = [
        node
        for node in model.graph.node
        if not node.domain and node.op_type == "MemcpyFromHost"
    ]
    aliases: dict[str, str] = {}
    for node in memcpy_nodes:
        if len(node.input) != 1 or len(node.output) != 1:
            raise RuntimeError(f"serialized {node.op_type} node {node.name!r} is not unary")
        aliases[str(node.output[0])] = str(node.input[0])
    if not aliases:
        return 0
    graph_outputs = {str(value.name) for value in model.graph.output}
    if graph_outputs.intersection(aliases):
        raise RuntimeError("serialized MemcpyFromHost unexpectedly feeds a graph output directly")

    def resolve(value_name: str) -> str:
        seen: set[str] = set()
        while value_name in aliases:
            if value_name in seen:
                raise RuntimeError("serialized CUDA memcpy aliases contain a cycle")
            seen.add(value_name)
            value_name = aliases[value_name]
        return value_name

    retained: list[onnx.NodeProto] = []
    memcpy_ids = {id(node) for node in memcpy_nodes}
    for node in model.graph.node:
        if id(node) in memcpy_ids:
            continue
        rewritten = copy.deepcopy(node)
        for index, input_name in enumerate(rewritten.input):
            rewritten.input[index] = resolve(str(input_name))
        retained.append(rewritten)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    retained_value_info = [
        copy.deepcopy(value)
        for value in model.graph.value_info
        if str(value.name) not in aliases
    ]
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)
    return len(memcpy_nodes)


def _restore_optimized_parity_loop_model(
    optimized_static: onnx.ModelProto,
    original_dynamic: onnx.ModelProto,
) -> onnx.ModelProto:
    wrapper = _parity_loop_wrapper(original_dynamic)
    if wrapper is None:
        raise ValueError("original model is not an EA2 static-B1 parity Loop wrapper")
    if [str(value.name) for value in optimized_static.graph.input] != [
        str(value.name) for value in original_dynamic.graph.input
    ]:
        raise ValueError("optimized static graph inputs changed while preparing the parity Loop")
    if len(optimized_static.graph.output) != 1 or (
        str(optimized_static.graph.output[0].name) != wrapper.predictor_output_name
    ):
        raise ValueError("optimized static graph output changed while preparing the parity Loop")

    replacements = {
        wrapper.predictor_input_name: wrapper.body_input_name,
        wrapper.predictor_output_name: wrapper.body_output_name,
    }
    optimized_body_nodes = _replace_node_value_names(
        list(optimized_static.graph.node),
        replacements,
    )
    restored_body = copy.deepcopy(wrapper.body)
    del restored_body.node[:]
    restored_body.node.extend(
        [
            copy.deepcopy(wrapper.gather_node),
            copy.deepcopy(wrapper.unsqueeze_node),
            *optimized_body_nodes,
            copy.deepcopy(wrapper.condition_node),
        ]
    )
    del restored_body.initializer[:]
    del restored_body.sparse_initializer[:]
    del restored_body.value_info[:]
    restored_body.value_info.extend(
        _replace_value_info_names(list(optimized_static.graph.value_info), replacements)
    )

    restored_outer_nodes = copy.deepcopy(list(original_dynamic.graph.node))
    restored_loop_count = 0
    for node in restored_outer_nodes:
        if node.name != _PARITY_LOOP_NAME:
            continue
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH and attribute.name == "body":
                attribute.g.CopyFrom(restored_body)
                restored_loop_count += 1
    if restored_loop_count != 1:
        raise RuntimeError(f"restored {restored_loop_count} parity Loop bodies; expected one")

    original_initializer_map = {
        str(initializer.name): initializer
        for initializer in original_dynamic.graph.initializer
    }
    wrapper_nodes = [
        *original_dynamic.graph.node,
        wrapper.gather_node,
        wrapper.unsqueeze_node,
        wrapper.condition_node,
    ]
    wrapper_initializer_names = {
        str(input_name)
        for node in wrapper_nodes
        for input_name in node.input
        if str(input_name) in original_initializer_map
    }
    restored_initializers = copy.deepcopy(list(optimized_static.graph.initializer))
    restored_names = {str(initializer.name) for initializer in restored_initializers}
    for name in sorted(wrapper_initializer_names):
        if name not in restored_names:
            restored_initializers.append(copy.deepcopy(original_initializer_map[name]))

    restored_graph = copy.deepcopy(original_dynamic.graph)
    del restored_graph.node[:]
    restored_graph.node.extend(restored_outer_nodes)
    del restored_graph.initializer[:]
    restored_graph.initializer.extend(restored_initializers)
    del restored_graph.sparse_initializer[:]
    restored_graph.sparse_initializer.extend(copy.deepcopy(list(optimized_static.graph.sparse_initializer)))
    restored = copy.deepcopy(optimized_static)
    restored.graph.CopyFrom(restored_graph)
    onnx.checker.check_model(restored)
    return restored


def _write_cuda_optimized_static_model(
    source_path: Path,
    output_path: Path,
    *,
    custom_ops: Path,
    device_id: int,
) -> None:
    ensure_gpu_runtime()
    options = _session_options(session_profile="throughput")
    options.optimized_model_filepath = str(Path(output_path).resolve())
    options.register_custom_ops_library(str(Path(custom_ops).resolve()))
    session = ort.InferenceSession(
        str(Path(source_path).resolve()),
        sess_options=options,
        providers=_cuda_provider_config(device_id=int(device_id), session_profile="throughput"),
    )
    provider = session.get_providers()[0]
    del session
    if provider != "CUDAExecutionProvider":
        raise RuntimeError(f"offline Loop optimization did not bind to CUDAExecutionProvider; got {provider}")
    if not Path(output_path).is_file():
        raise RuntimeError("ONNX Runtime did not write the requested optimized static model")


def _loop_preoptimized_cache_key(
    rewritten_model_path: Path,
    *,
    custom_ops: Path,
    device_id: int,
    static_shape_reference_path: Path | None,
) -> str:
    capability = torch.cuda.get_device_capability(int(device_id))
    payload = {
        "mode": LOOP_PREOPTIMIZATION_ABI,
        "rewritten_model_sha1": _sha1_file(rewritten_model_path),
        "custom_ops_sha1": _sha1_file(custom_ops),
        "static_shape_reference_sha1": (
            None
            if static_shape_reference_path is None
            else _sha1_file(static_shape_reference_path)
        ),
        "onnxruntime_version": str(ort.__version__),
        "cuda_architectures": resolve_cuda_architectures(),
        "optimizer_device_capability": [int(capability[0]), int(capability[1])],
        "session_profile": "throughput",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _preoptimize_parity_loop_model(
    rewritten_model_path: Path,
    *,
    custom_ops: Path,
    device_id: int,
    session_profile: str,
    static_shape_reference_path: Path | None = None,
) -> Path:
    if str(session_profile).strip().lower() != "throughput":
        return Path(rewritten_model_path)
    original_dynamic = onnx.load(str(rewritten_model_path))
    if _parity_loop_wrapper(original_dynamic) is None:
        return Path(rewritten_model_path)
    if static_shape_reference_path is None:
        raise ValueError(
            "throughput execution of an EA2 static-B1 parity Loop requires "
            "static_shape_reference_path so CUDA graph optimization preserves the clean numeric path"
        )

    key = _loop_preoptimized_cache_key(
        Path(rewritten_model_path),
        custom_ops=Path(custom_ops),
        device_id=int(device_id),
        static_shape_reference_path=(
            None
            if static_shape_reference_path is None
            else Path(static_shape_reference_path)
        ),
    )
    output_path = MODEL_CACHE_DIR / f"{key}.onnx"
    if output_path.exists():
        return output_path
    lock_path = MODEL_CACHE_DIR / f"{key}.lock"
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            acquired = False
        if acquired:
            temporary_paths: list[Path] = []
            try:
                if not output_path.exists():
                    for suffix in (".static.onnx", ".optimized.onnx", ".final.onnx"):
                        with tempfile.NamedTemporaryFile(
                            dir=str(MODEL_CACHE_DIR),
                            prefix=f"{key}.",
                            suffix=suffix,
                            delete=False,
                        ) as handle:
                            temporary_paths.append(Path(handle.name))
                    static_path, optimized_path, final_path = temporary_paths
                    shape_reference = (
                        None
                        if static_shape_reference_path is None
                        else onnx.load(str(static_shape_reference_path))
                    )
                    static_model = _extract_static_parity_loop_model(
                        original_dynamic,
                        shape_reference=shape_reference,
                    )
                    onnx.save(static_model, str(static_path))
                    optimized_path.unlink()
                    _write_cuda_optimized_static_model(
                        static_path,
                        optimized_path,
                        custom_ops=Path(custom_ops),
                        device_id=int(device_id),
                    )
                    optimized_static = onnx.load(str(optimized_path))
                    _strip_serialized_cuda_memcpy_nodes(optimized_static)
                    restored = _restore_optimized_parity_loop_model(
                        optimized_static,
                        original_dynamic,
                    )
                    onnx.save(restored, str(final_path))
                    os.replace(final_path, output_path)
            finally:
                for temporary_path in temporary_paths:
                    try:
                        temporary_path.unlink()
                    except FileNotFoundError:
                        pass
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            break
        if output_path.exists():
            break
        time.sleep(0.1)
    return output_path


def _export_dynamic_lut_model(
    model_path: Path,
    *,
    node_to_lut_input_name: Mapping[str, str],
    lut_domain: str | None,
    s8_residual_abs_bound: int | None = None,
) -> tuple[Path, dict[str, str], str]:
    if s8_residual_abs_bound is not None:
        if isinstance(s8_residual_abs_bound, bool) or not isinstance(s8_residual_abs_bound, int):
            raise TypeError("s8_residual_abs_bound must be an integer or None")
        if int(s8_residual_abs_bound) < 0:
            raise ValueError("s8_residual_abs_bound must be non-negative")
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _dynamic_model_cache_key(
        model_path,
        node_to_lut_input_name=node_to_lut_input_name,
        lut_domain=lut_domain,
        s8_residual_abs_bound=s8_residual_abs_bound,
    )
    output_path = MODEL_CACHE_DIR / f"{key}.onnx"
    if not output_path.exists():
        lock_path = MODEL_CACHE_DIR / f"{key}.lock"
        while True:
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                acquired = True
            except FileExistsError:
                acquired = False
            if acquired:
                try:
                    if not output_path.exists():
                        with tempfile.NamedTemporaryFile(
                            dir=str(MODEL_CACHE_DIR),
                            prefix=f"{key}.",
                            suffix=".tmp.onnx",
                            delete=False,
                        ) as handle:
                            temp_path = Path(handle.name)
                        try:
                            export_matmul_lut_cuda_model(
                                model_path=Path(model_path),
                                output_path=temp_path,
                                lut_input_name_by_node=dict(node_to_lut_input_name),
                                include_node_names=list(node_to_lut_input_name.keys()),
                                skip_name_fragments=None,
                                s8_residual_abs_bound=s8_residual_abs_bound,
                            )
                            os.replace(temp_path, output_path)
                        finally:
                            if temp_path.exists():
                                temp_path.unlink()
                finally:
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                break
            if output_path.exists():
                break
            time.sleep(0.1)
    exported_model = onnx.load(str(output_path))
    actual_lut_domain = _validate_requested_lut_domain(
        actual_lut_domain=_exported_lut_domain(exported_model),
        requested_lut_domain=lut_domain,
    )
    custom_node_names = [
        str(ref.node.name)
        for ref in iter_graph_nodes(exported_model)
        if ref.node.domain == "approxdomain"
        and ref.node.op_type in {"ApproxMatMulCuda", "ApproxMatMulCudaS8Lut"}
    ]
    expected_custom_names = {f"{node_name}_lut_cuda" for node_name in node_to_lut_input_name}
    if set(custom_node_names) != expected_custom_names or len(custom_node_names) != len(expected_custom_names):
        raise RuntimeError(
            "cached/exported LUT CUDA graph does not match the requested node set: "
            f"expected={sorted(expected_custom_names)}, actual={sorted(custom_node_names)}"
        )
    return output_path, dict(node_to_lut_input_name), actual_lut_domain


def _normalize_lut_delta(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.int32)
    if array.shape == (256, 256):
        return np.ascontiguousarray(array)
    if array.size == 256 * 256:
        return np.ascontiguousarray(array.reshape(256, 256))
    raise ValueError(f"LUT delta must have shape (256, 256) or 65536 elements, got {array.shape}")


@dataclass
class CompiledLutCudaRuntime:
    model_path: Path
    predictor_input_name: str
    predictor_input_shape: tuple[object, ...]
    provider: str
    session: object
    lut_input_name_by_node: dict[str, str]
    lut_domain: str
    session_profile: str

    def run(
        self,
        batch_input: np.ndarray,
        *,
        approx_deltas_by_node: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        approx_deltas_by_input: Mapping[str, np.ndarray | torch.Tensor] | None = None,
    ) -> list[np.ndarray]:
        if (approx_deltas_by_node is None) == (approx_deltas_by_input is None):
            raise ValueError("run expects exactly one of approx_deltas_by_node or approx_deltas_by_input")
        lut_feeds: dict[str, np.ndarray] = {}
        if approx_deltas_by_node is not None:
            for node_name, value in approx_deltas_by_node.items():
                input_name = self.lut_input_name_by_node.get(str(node_name))
                if input_name is None:
                    raise KeyError(f"Unknown LUT node '{node_name}'")
                lut_feeds[input_name] = _normalize_lut_delta(value)
        else:
            for input_name, value in approx_deltas_by_input.items():
                lut_feeds[str(input_name)] = _normalize_lut_delta(value)
        return self.run_prebuilt(batch_input, lut_feeds_by_input=lut_feeds)

    def run_prebuilt(
        self,
        batch_input: np.ndarray,
        *,
        lut_feeds_by_input: Mapping[str, np.ndarray],
    ) -> list[np.ndarray]:
        feed = {
            self.predictor_input_name: np.asarray(batch_input, dtype=np.float32),
            **{str(input_name): np.asarray(value, dtype=np.int32) for input_name, value in lut_feeds_by_input.items()},
        }
        return self.session.run(None, feed)

    def close(self) -> None:
        self.session = None


@dataclass
class CompiledSingleMatMulLutCudaRuntimePool:
    runtimes: list[CompiledLutCudaRuntime]
    lut_input_name: str

    def __post_init__(self) -> None:
        if not self.runtimes:
            raise ValueError("CompiledSingleMatMulLutCudaRuntimePool requires at least one runtime")
        self._executor = ThreadPoolExecutor(max_workers=len(self.runtimes), thread_name_prefix="lut_cuda_rt")

    def capture_probe_batch(self, batch_input: np.ndarray):
        return None

    def run(
        self,
        batch_input: np.ndarray,
        *,
        approx_delta: np.ndarray | torch.Tensor,
        cached_fetches=None,
    ) -> list[np.ndarray]:
        del cached_fetches
        return self.runtimes[0].run(
            batch_input,
            approx_deltas_by_input={self.lut_input_name: approx_delta},
        )

    def run_many(
        self,
        batch_input: np.ndarray,
        *,
        approx_deltas_by_key: Mapping[str, np.ndarray | torch.Tensor],
        cached_fetches=None,
    ) -> dict[str, np.ndarray]:
        del cached_fetches
        keys = list(approx_deltas_by_key.keys())
        if not keys:
            return {}
        if len(self.runtimes) == 1:
            runtime = self.runtimes[0]
            return {
                str(key): np.asarray(
                    runtime.run(
                        batch_input,
                        approx_deltas_by_input={self.lut_input_name: approx_deltas_by_key[key]},
                    )[0],
                    dtype=np.float32,
                )
                for key in keys
            }

        worker_count = min(len(self.runtimes), len(keys))
        groups: list[list[str]] = []
        for worker_idx in range(worker_count):
            group = [keys[idx] for idx in range(worker_idx, len(keys), worker_count)]
            if group:
                groups.append(group)

        def worker(runtime: CompiledLutCudaRuntime, key_group: list[str]) -> dict[str, np.ndarray]:
            local: dict[str, np.ndarray] = {}
            for key in key_group:
                local[str(key)] = np.asarray(
                    runtime.run(
                        batch_input,
                        approx_deltas_by_input={self.lut_input_name: approx_deltas_by_key[key]},
                    )[0],
                    dtype=np.float32,
                )
            return local

        futures = []
        for runtime, key_group in zip(self.runtimes, groups, strict=False):
            futures.append(self._executor.submit(worker, runtime, key_group))
        merged: dict[str, np.ndarray] = {}
        for future in futures:
            merged.update(future.result())
        return merged

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        for runtime in self.runtimes:
            runtime.close()


def compile_lut_cuda_runtime(
    model_path: Path,
    *,
    node_to_lut_input_name: Mapping[str, str],
    device_id: int = 0,
    batch_fallback_parallelism: int = 1,
    lut_domain: str | None = None,
    s8_residual_abs_bound: int | None = None,
    session_profile: str = "strict",
    static_shape_reference_path: Path | None = None,
) -> CompiledLutCudaRuntime:
    resolved_mapping = {str(node): str(name) for node, name in node_to_lut_input_name.items()}
    exported_model_path, resolved_mapping, resolved_lut_domain = _export_dynamic_lut_model(
        Path(model_path),
        node_to_lut_input_name=resolved_mapping,
        lut_domain=lut_domain,
        s8_residual_abs_bound=s8_residual_abs_bound,
    )
    custom_ops = _lut_custom_op_library()
    runtime_model_path = _preoptimize_parity_loop_model(
        exported_model_path,
        custom_ops=custom_ops,
        device_id=int(device_id),
        session_profile=str(session_profile),
        static_shape_reference_path=static_shape_reference_path,
    )
    exported_model = onnx.load(str(runtime_model_path))
    resolved_lut_domain = _validate_requested_lut_domain(
        actual_lut_domain=_exported_lut_domain(exported_model),
        requested_lut_domain=lut_domain,
    )
    actual_custom_names = [
        str(ref.node.name)
        for ref in iter_graph_nodes(exported_model)
        if ref.node.domain == "approxdomain"
        and ref.node.op_type in {"ApproxMatMulCuda", "ApproxMatMulCudaS8Lut"}
    ]
    expected_custom_names = {f"{node_name}_lut_cuda" for node_name in resolved_mapping}
    if set(actual_custom_names) != expected_custom_names or len(actual_custom_names) != len(expected_custom_names):
        raise RuntimeError(
            "runtime LUT CUDA graph does not match the requested node set after Loop optimization: "
            f"expected={sorted(expected_custom_names)}, actual={sorted(actual_custom_names)}"
        )
    predictor_input_name = str(exported_model.graph.input[0].name)
    session, provider = create_cuda_session(
        runtime_model_path,
        batch_input_names=(predictor_input_name,),
        device_id=int(device_id),
        custom_ops=custom_ops,
        fallback_parallelism=int(batch_fallback_parallelism),
        session_profile=str(session_profile),
    )
    input_meta = next(meta for meta in session.get_inputs() if str(meta.name) == predictor_input_name)
    return CompiledLutCudaRuntime(
        model_path=runtime_model_path,
        predictor_input_name=predictor_input_name,
        predictor_input_shape=tuple(input_meta.shape),
        provider=provider,
        session=session,
        lut_input_name_by_node=resolved_mapping,
        lut_domain=resolved_lut_domain,
        session_profile=str(session_profile),
    )


def compile_single_matmul_lut_cuda_runtime(
    model_path: Path,
    *,
    node_name: str,
    device_id: int = 0,
    batch_fallback_parallelism: int = 1,
    lut_domain: str | None = None,
    s8_residual_abs_bound: int | None = None,
    session_profile: str = "strict",
    static_shape_reference_path: Path | None = None,
) -> CompiledLutCudaRuntime:
    return compile_lut_cuda_runtime(
        model_path,
        node_to_lut_input_name={str(node_name): _default_lut_input_name(str(node_name))},
        device_id=int(device_id),
        batch_fallback_parallelism=int(batch_fallback_parallelism),
        lut_domain=lut_domain,
        s8_residual_abs_bound=s8_residual_abs_bound,
        session_profile=str(session_profile),
        static_shape_reference_path=static_shape_reference_path,
    )
