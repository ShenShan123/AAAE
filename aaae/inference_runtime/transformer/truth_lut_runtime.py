from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

from .ort_session import create_cuda_session_from_bytes


@dataclass(frozen=True)
class TensorSpec:
    tensor_name: str
    fetch_name: str | None
    const_q: np.ndarray | None
    scale: np.ndarray | None
    zp: np.ndarray | None
    axis: int | None
    qmin: int
    qmax: int


@dataclass(frozen=True)
class ApproxOpSpec:
    name: str
    input_names: tuple[str, ...]
    output_name: str
    override_input_name: str
    a: TensorSpec
    b: TensorSpec


@dataclass
class PreparedMatMulState:
    exact: torch.Tensor
    a_flat: torch.Tensor
    b_flat: torch.Tensor
    scale: torch.Tensor
    output_shape: tuple[int, ...]


@dataclass
class PreparedSample:
    batch_input: np.ndarray
    fetch_map: dict[str, np.ndarray]
    state: PreparedMatMulState


@dataclass
class CompiledMatMulRuntime:
    node_name: str
    predictor_input_name: str
    predictor_input_shape: tuple[object, ...]
    provider: str
    spec: ApproxOpSpec
    probe_fetch_names: list[str]
    suffix_frontier_input_names: tuple[str, ...]
    pass_predictor_input_to_injected: bool
    override_input_name: str
    probe_session: object
    injected_session: object
    device: torch.device

    def prepare_batch(
        self,
        batch_input: np.ndarray,
        *,
        cached_fetches: list[dict[str, np.ndarray]] | None = None,
    ) -> list[PreparedSample]:
        batch_size = int(batch_input.shape[0])
        if cached_fetches is not None and len(cached_fetches) != batch_size:
            raise ValueError(
                f"cached_fetches length {len(cached_fetches)} does not match batch size {batch_size}"
            )
        if cached_fetches is None:
            cached_fetches = self.capture_probe_batch(batch_input)
        prepared: list[PreparedSample] = []
        for idx in range(batch_size):
            fetch_map = dict(cached_fetches[idx])
            prepared.append(
                PreparedSample(
                    batch_input=batch_input[idx : idx + 1],
                    fetch_map=fetch_map,
                    state=_prepare_matmul_state(self.spec, fetch_map, device=self.device),
                )
            )
        return prepared

    def run_prepared(
        self,
        prepared_samples: list[PreparedSample],
        *,
        approx_delta: torch.Tensor,
    ) -> list[np.ndarray]:
        per_sample: list[list[np.ndarray]] = []
        for prepared in prepared_samples:
            output = _compute_matmul_from_state(
                prepared.state,
                approx_delta=approx_delta,
            )
            injected_feeds: dict[str, np.ndarray] = {
                name: np.asarray(prepared.fetch_map[name])
                for name in self.suffix_frontier_input_names
            }
            if self.pass_predictor_input_to_injected:
                injected_feeds[self.predictor_input_name] = prepared.batch_input
            injected_feeds[self.override_input_name] = output.detach().cpu().numpy()
            per_sample.append(self.injected_session.run(None, injected_feeds))

        merged: list[np.ndarray] = []
        for slot_values in zip(*per_sample, strict=True):
            arrays = [np.asarray(value) for value in slot_values]
            if arrays[0].ndim == 0:
                merged.append(np.asarray(arrays))
            else:
                merged.append(np.concatenate(arrays, axis=0))
        return merged

    def _run_single(
        self,
        batch_input: np.ndarray,
        *,
        approx_delta: torch.Tensor,
        fetch_map: dict[str, np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        if batch_input.shape[0] <= 0:
            raise ValueError("_run_single expects a non-empty batch")
        cached_fetches = (None if fetch_map is None else [fetch_map])
        prepared = self.prepare_batch(batch_input, cached_fetches=cached_fetches)
        return self.run_prepared(prepared, approx_delta=approx_delta)

    def capture_probe_batch(self, batch_input: np.ndarray) -> list[dict[str, np.ndarray]]:
        cached: list[dict[str, np.ndarray]] = []
        for idx in range(int(batch_input.shape[0])):
            single_input = batch_input[idx : idx + 1]
            if not self.probe_fetch_names:
                cached.append({})
                continue
            fetch_values = self.probe_session.run(self.probe_fetch_names, {self.predictor_input_name: single_input})
            cached.append({name: value for name, value in zip(self.probe_fetch_names, fetch_values, strict=True)})
        return cached

    def run(
        self,
        batch_input: np.ndarray,
        *,
        approx_delta: torch.Tensor,
        cached_fetches: list[dict[str, np.ndarray]] | None = None,
    ) -> list[np.ndarray]:
        prepared = self.prepare_batch(batch_input, cached_fetches=cached_fetches)
        return self.run_prepared(prepared, approx_delta=approx_delta)

    def close(self) -> None:
        self.probe_session = None
        self.injected_session = None


@dataclass
class CompiledMatMulRuntimePool:
    runtimes: list[CompiledMatMulRuntime]

    def __post_init__(self) -> None:
        if not self.runtimes:
            raise ValueError("CompiledMatMulRuntimePool requires at least one runtime")
        self._executor = ThreadPoolExecutor(max_workers=len(self.runtimes), thread_name_prefix="d_node_rt")

    def _groups(self, sample_count: int) -> list[list[int]]:
        worker_count = min(len(self.runtimes), int(sample_count))
        groups: list[list[int]] = []
        for worker_idx in range(worker_count):
            indices = list(range(worker_idx, int(sample_count), worker_count))
            if indices:
                groups.append(indices)
        return groups

    def capture_probe_batch(self, batch_input: np.ndarray) -> list[dict[str, np.ndarray]]:
        if batch_input.shape[0] <= 1 or len(self.runtimes) == 1:
            return self.runtimes[0].capture_probe_batch(batch_input)
        results: list[dict[str, np.ndarray] | None] = [None] * int(batch_input.shape[0])

        def worker(runtime: CompiledMatMulRuntime, indices: list[int]) -> list[tuple[int, dict[str, np.ndarray]]]:
            local: list[tuple[int, dict[str, np.ndarray]]] = []
            for idx in indices:
                local.append((idx, runtime.capture_probe_batch(batch_input[idx : idx + 1])[0]))
            return local

        futures = []
        groups = self._groups(int(batch_input.shape[0]))
        for runtime, indices in zip(self.runtimes, groups, strict=False):
            futures.append(self._executor.submit(worker, runtime, indices))
        for future in futures:
            for idx, fetch_map in future.result():
                results[int(idx)] = fetch_map
        return [value for value in results if value is not None]

    def run_many(
        self,
        batch_input: np.ndarray,
        *,
        approx_deltas_by_key: dict[str, torch.Tensor],
        cached_fetches: list[dict[str, np.ndarray]] | None = None,
    ) -> dict[str, np.ndarray]:
        keys = list(approx_deltas_by_key.keys())
        if not keys:
            return {}
        if len(self.runtimes) == 1:
            runtime = self.runtimes[0]
            prepared = runtime.prepare_batch(batch_input, cached_fetches=cached_fetches)
            out: dict[str, np.ndarray] = {}
            for key in keys:
                out[str(key)] = np.asarray(
                    runtime.run_prepared(prepared, approx_delta=approx_deltas_by_key[key])[0],
                    dtype=np.float32,
                )
            return out

        worker_count = min(len(self.runtimes), len(keys))
        groups: list[list[str]] = []
        for worker_idx in range(worker_count):
            group = [keys[idx] for idx in range(worker_idx, len(keys), worker_count)]
            if group:
                groups.append(group)

        def worker(runtime: CompiledMatMulRuntime, key_group: list[str]) -> dict[str, np.ndarray]:
            prepared = runtime.prepare_batch(batch_input, cached_fetches=cached_fetches)
            local: dict[str, np.ndarray] = {}
            for key in key_group:
                local[str(key)] = np.asarray(
                    runtime.run_prepared(prepared, approx_delta=approx_deltas_by_key[key])[0],
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

    def run(
        self,
        batch_input: np.ndarray,
        *,
        approx_delta: torch.Tensor,
        cached_fetches: list[dict[str, np.ndarray]] | None = None,
    ) -> list[np.ndarray]:
        if cached_fetches is not None and len(cached_fetches) != int(batch_input.shape[0]):
            raise ValueError(
                f"cached_fetches length {len(cached_fetches)} does not match batch size {int(batch_input.shape[0])}"
            )
        if batch_input.shape[0] <= 1 or len(self.runtimes) == 1:
            return self.runtimes[0].run(batch_input, approx_delta=approx_delta, cached_fetches=cached_fetches)

        sample_count = int(batch_input.shape[0])
        outputs_per_sample: list[list[np.ndarray] | None] = [None] * sample_count

        def worker(runtime: CompiledMatMulRuntime, indices: list[int]) -> list[tuple[int, list[np.ndarray]]]:
            local: list[tuple[int, list[np.ndarray]]] = []
            for idx in indices:
                local.append(
                    (
                        idx,
                        runtime.run(
                            batch_input[idx : idx + 1],
                            approx_delta=approx_delta,
                            cached_fetches=(None if cached_fetches is None else [cached_fetches[idx]]),
                        ),
                    )
                )
            return local

        futures = []
        groups = self._groups(sample_count)
        for runtime, indices in zip(self.runtimes, groups, strict=False):
            futures.append(self._executor.submit(worker, runtime, indices))
        for future in futures:
            for idx, outputs in future.result():
                outputs_per_sample[int(idx)] = outputs

        merged: list[np.ndarray] = []
        realized_outputs = [value for value in outputs_per_sample if value is not None]
        for slot_values in zip(*realized_outputs, strict=True):
            arrays = [np.asarray(value) for value in slot_values]
            if arrays[0].ndim == 0:
                merged.append(np.asarray(arrays))
            else:
                merged.append(np.concatenate(arrays, axis=0))
        return merged

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        for runtime in self.runtimes:
            runtime.close()


def _qrange_from_dtype(dtype: np.dtype | None) -> tuple[int, int]:
    if dtype is None:
        return 0, 255
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.signedinteger):
        info = np.iinfo(dt)
        return int(info.min), int(info.max)
    if np.issubdtype(dt, np.unsignedinteger):
        info = np.iinfo(dt)
        return int(info.min), int(info.max)
    return 0, 255


def _resolve_dequant_info(
    tensor_name: str,
    *,
    initializer_map: dict[str, np.ndarray],
    producer_map: dict[str, onnx.NodeProto],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, int | None]:
    node = producer_map.get(str(tensor_name))
    if node is None or node.op_type != "DequantizeLinear" or len(node.input) < 3:
        return None, None, None, None
    q_name = str(node.input[0])
    scale_name = str(node.input[1])
    zp_name = str(node.input[2])
    scale = initializer_map.get(scale_name)
    zp = initializer_map.get(zp_name)
    if scale is None or zp is None:
        return None, None, None, None
    q = initializer_map.get(q_name)
    axis = None
    for attr in node.attribute:
        if attr.name == "axis":
            axis = int(helper.get_attribute_value(attr))
            break
    return q, np.asarray(scale), np.asarray(zp), axis


def _make_dynamic_tensor_spec(
    tensor_name: str,
    *,
    initializer_map: dict[str, np.ndarray],
    producer_map: dict[str, onnx.NodeProto],
) -> TensorSpec:
    q, scale, zp, axis = _resolve_dequant_info(
        tensor_name,
        initializer_map=initializer_map,
        producer_map=producer_map,
    )
    if scale is not None and zp is not None:
        dtype = q.dtype if q is not None else zp.dtype
        qmin, qmax = _qrange_from_dtype(dtype)
        return TensorSpec(
            tensor_name=str(tensor_name),
            fetch_name=str(tensor_name) if q is None else None,
            const_q=np.asarray(q).copy() if q is not None else None,
            scale=np.asarray(scale).copy(),
            zp=np.asarray(zp).copy(),
            axis=axis,
            qmin=qmin,
            qmax=qmax,
        )
    if str(tensor_name) in initializer_map:
        arr = np.asarray(initializer_map[str(tensor_name)])
        if np.issubdtype(arr.dtype, np.integer):
            qmin, qmax = _qrange_from_dtype(arr.dtype)
            return TensorSpec(
                tensor_name=str(tensor_name),
                fetch_name=None,
                const_q=arr.copy(),
                scale=np.asarray([1.0], dtype=np.float64),
                zp=np.asarray([0], dtype=np.int64),
                axis=None,
                qmin=qmin,
                qmax=qmax,
            )
    return TensorSpec(
        tensor_name=str(tensor_name),
        fetch_name=str(tensor_name),
        const_q=None,
        scale=None,
        zp=None,
        axis=None,
        qmin=-127,
        qmax=127,
    )


def _build_override_model(model: onnx.ModelProto, *, output_name: str) -> tuple[onnx.ModelProto, str]:
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(model)
    override_name = f"approx_override::{output_name}"
    override_out = f"{output_name}::approx_override"
    index = 0
    while index < len(model_copy.graph.node):
        node = model_copy.graph.node[index]
        if node.output and str(node.output[0]) == str(output_name):
            add_node = helper.make_node(
                "Identity",
                inputs=[override_name],
                outputs=[override_out],
                name=f"ApproxOverride_{str(node.name or output_name).replace('/', '_')}",
            )
            model_copy.graph.node.insert(index + 1, add_node)
            model_copy.graph.input.append(helper.make_tensor_value_info(override_name, TensorProto.FLOAT, None))
            for later_idx in range(index + 2, len(model_copy.graph.node)):
                later = model_copy.graph.node[later_idx]
                for inp_idx, inp in enumerate(later.input):
                    if str(inp) == str(output_name):
                        later.input[inp_idx] = override_out
            for out in model_copy.graph.output:
                if str(out.name) == str(output_name):
                    out.name = override_out
            break
        index += 1
    return model_copy, override_name


def _build_probe_model(model: onnx.ModelProto, *, extra_outputs: Sequence[str]) -> bytes:
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(model)
    del model_copy.graph.output[:]
    seen: set[str] = set()
    for name in extra_outputs:
        if str(name) in seen:
            continue
        seen.add(str(name))
        model_copy.graph.output.append(helper.make_tensor_value_info(str(name), TensorProto.FLOAT, None))
    return model_copy.SerializeToString()


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    out: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        for output in node.output:
            out[str(output)] = node
    return out


def collect_single_matmul_spec(model_path: Path, *, node_name: str) -> ApproxOpSpec:
    model = onnx.load(str(model_path))
    initializer_map = {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}
    producer_map = _producer_map(model)
    for node in model.graph.node:
        current_name = str(node.name or "")
        if current_name != str(node_name):
            continue
        if node.op_type != "MatMul" or len(node.input) < 2 or not node.output:
            raise RuntimeError(f"Node '{node_name}' is not a supported MatMul node")
        return ApproxOpSpec(
            name=current_name,
            input_names=tuple(str(inp) for inp in node.input),
            output_name=str(node.output[0]),
            override_input_name=f"approx_override::{str(node.output[0])}",
            a=_make_dynamic_tensor_spec(str(node.input[0]), initializer_map=initializer_map, producer_map=producer_map),
            b=_make_dynamic_tensor_spec(str(node.input[1]), initializer_map=initializer_map, producer_map=producer_map),
        )
    raise KeyError(f"Node '{node_name}' not found in {model_path}")


def _resolve_tensor_runtime(
    spec: TensorSpec,
    fetch_map: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if spec.const_q is not None:
        q = torch.as_tensor(spec.const_q, device=device, dtype=torch.int64)
        scale = torch.as_tensor(spec.scale if spec.scale is not None else np.asarray([1.0]), device=device, dtype=torch.float64)
        zp = torch.as_tensor(spec.zp if spec.zp is not None else np.asarray([0]), device=device, dtype=torch.int64)
        return q, scale, zp, int(spec.qmin)

    value = fetch_map.get(str(spec.fetch_name or spec.tensor_name))
    if value is None:
        raise KeyError(f"Missing runtime tensor '{spec.fetch_name or spec.tensor_name}'")
    x = torch.as_tensor(value, device=device, dtype=torch.float64)
    if spec.scale is not None and spec.zp is not None:
        scale = torch.as_tensor(spec.scale, device=device, dtype=torch.float64)
        zp = torch.as_tensor(spec.zp, device=device, dtype=torch.float64)
        if scale.numel() == 1:
            q = torch.round(x / scale.reshape(()) + zp.reshape(()))
        else:
            axis = int(spec.axis if spec.axis is not None else 0)
            if axis < 0:
                axis += x.ndim
            shape = [1] * x.ndim
            shape[axis] = int(scale.numel())
            q = torch.round(x / scale.reshape(shape) + zp.reshape(shape))
        q = torch.clamp(q, int(spec.qmin), int(spec.qmax)).to(torch.int64)
        return q, scale, torch.as_tensor(spec.zp, device=device, dtype=torch.int64), int(spec.qmin)

    absmax = torch.max(torch.abs(x))
    scale_value = float(absmax.item() / 127.0) if float(absmax.item()) > 0.0 else 1.0
    q = torch.round(x / scale_value).to(torch.int64)
    q = torch.clamp(q, int(spec.qmin), int(spec.qmax))
    return (
        q,
        torch.as_tensor([scale_value], device=device, dtype=torch.float64),
        torch.as_tensor([0], device=device, dtype=torch.int64),
        int(spec.qmin),
    )


def _broadcast_zp(zp: torch.Tensor, *, axis: int | None, target_shape: Sequence[int], device: torch.device) -> torch.Tensor:
    zp = torch.as_tensor(zp, device=device, dtype=torch.int64)
    if zp.numel() == 1:
        return zp.reshape(())
    axis_idx = int(axis if axis is not None else 0)
    if axis_idx < 0:
        axis_idx += len(target_shape)
    shape = [1] * len(target_shape)
    shape[axis_idx] = int(zp.numel())
    return zp.reshape(shape)


def _normalize_matmul_operand(
    q: torch.Tensor,
    *,
    zp: torch.Tensor,
    axis: int | None,
    qmin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis_work = int(axis if axis is not None else (q.ndim - 1))
    if axis_work < 0:
        axis_work += q.ndim
    zp_b = _broadcast_zp(zp, axis=axis_work, target_shape=tuple(q.shape), device=q.device)
    unsigned = (q - int(qmin)).to(torch.int32)
    centered = (q - zp_b).to(torch.float64)
    return unsigned, centered


def _lookup_accumulate_matmul(a_unsigned: torch.Tensor, b_unsigned: torch.Tensor, *, approx_delta: torch.Tensor) -> torch.Tensor:
    batch = int(a_unsigned.shape[0])
    m_size = int(a_unsigned.shape[1])
    n_size = int(b_unsigned.shape[2])
    out = torch.zeros((batch, m_size, n_size), device=a_unsigned.device, dtype=torch.int32)
    chunk_m = 32
    chunk_n = 32
    b_t = b_unsigned.transpose(1, 2).contiguous()
    flat_delta = approx_delta.reshape(-1)
    for m0 in range(0, m_size, chunk_m):
        a_chunk = a_unsigned[:, m0 : m0 + chunk_m, :]
        for n0 in range(0, n_size, chunk_n):
            bt_chunk = b_t[:, n0 : n0 + chunk_n, :]
            lookup_index = (
                a_chunk[:, :, None, :].to(torch.int32) * 256
                + bt_chunk[:, None, :, :].to(torch.int32)
            ).to(torch.int64)
            delta = flat_delta[lookup_index]
            out[:, m0 : m0 + delta.shape[1], n0 : n0 + delta.shape[2]] = delta.sum(dim=-1, dtype=torch.int32)
    return out


def _prepare_matmul_state(
    spec: ApproxOpSpec,
    fetch_map: dict[str, np.ndarray],
    *,
    device: torch.device,
) -> PreparedMatMulState:
    a_q, a_scale, a_zp, qmin_a = _resolve_tensor_runtime(spec.a, fetch_map, device=device)
    b_q, b_scale, b_zp, qmin_b = _resolve_tensor_runtime(spec.b, fetch_map, device=device)
    a_unsigned, a_centered = _normalize_matmul_operand(a_q, zp=a_zp, axis=spec.a.axis, qmin=qmin_a)
    b_unsigned, b_centered = _normalize_matmul_operand(b_q, zp=b_zp, axis=spec.b.axis, qmin=qmin_b)
    exact = torch.matmul(a_centered, b_centered)
    batch_shape = torch.broadcast_shapes(tuple(a_unsigned.shape[:-2]), tuple(b_unsigned.shape[:-2]))
    a_u = a_unsigned.expand(*batch_shape, *a_unsigned.shape[-2:])
    b_u = b_unsigned.expand(*batch_shape, *b_unsigned.shape[-2:])
    scale = a_scale.reshape(-1)[0] * b_scale.reshape(-1)[0]
    return PreparedMatMulState(
        exact=exact,
        a_flat=a_u.reshape(-1, a_u.shape[-2], a_u.shape[-1]),
        b_flat=b_u.reshape(-1, b_u.shape[-2], b_u.shape[-1]),
        scale=scale,
        output_shape=tuple(int(dim) for dim in exact.shape),
    )


def _compute_matmul_from_state(
    state: PreparedMatMulState,
    *,
    approx_delta: torch.Tensor,
) -> torch.Tensor:
    delta = _lookup_accumulate_matmul(
        state.a_flat,
        state.b_flat,
        approx_delta=approx_delta,
    ).reshape(*state.output_shape).to(state.exact.dtype)
    y = (state.exact + delta) * state.scale
    return y.to(torch.float32)


def _compute_matmul_like(spec: ApproxOpSpec, fetch_map: dict[str, np.ndarray], *, approx_delta: torch.Tensor, device: torch.device) -> torch.Tensor:
    state = _prepare_matmul_state(spec, fetch_map, device=device)
    return _compute_matmul_from_state(state, approx_delta=approx_delta)


def compile_single_matmul_runtime(
    model_path: Path,
    *,
    node_name: str,
    device_id: int = 0,
) -> CompiledMatMulRuntime:
    model = onnx.load(str(model_path))
    spec = collect_single_matmul_spec(model_path, node_name=node_name)
    injected_model, override_name = _build_override_model(model, output_name=spec.output_name)

    step_model = onnx.ModelProto()
    step_model.CopyFrom(model)
    step_nodes = {str(node.name or ""): node for node in step_model.graph.node}
    step_node = step_nodes.get(spec.name)
    if step_node is None:
        raise RuntimeError(f"Selected node '{node_name}' disappeared while compiling runtime")
    inputs_now = tuple(str(inp) for inp in step_node.input)
    step_spec = replace(
        spec,
        input_names=inputs_now,
        a=replace(spec.a, fetch_name=(inputs_now[0] if spec.a.fetch_name is not None else None)),
        b=replace(spec.b, fetch_name=(inputs_now[1] if spec.b.fetch_name is not None and len(inputs_now) > 1 else None)),
    )
    matmul_fetch_names = [name for name in (step_spec.a.fetch_name, step_spec.b.fetch_name) if name]
    probe_fetch_names = list(matmul_fetch_names)
    frontier_input_names: tuple[str, ...] = tuple()
    probe_bytes = _build_probe_model(step_model, extra_outputs=probe_fetch_names)

    probe_session, _ = create_cuda_session_from_bytes(
        probe_bytes,
        batch_input_names=(str(model.graph.input[0].name),),
        device_id=int(device_id),
    )
    injected_session, provider = create_cuda_session_from_bytes(
        injected_model.SerializeToString(),
        batch_input_names=(str(model.graph.input[0].name),),
        device_id=int(device_id),
    )
    input_meta = probe_session.get_inputs()[0]

    return CompiledMatMulRuntime(
        node_name=str(node_name),
        predictor_input_name=str(input_meta.name),
        predictor_input_shape=tuple(input_meta.shape),
        provider=provider,
        spec=step_spec,
        probe_fetch_names=probe_fetch_names,
        suffix_frontier_input_names=tuple(),
        pass_predictor_input_to_injected=True,
        override_input_name=override_name,
        probe_session=probe_session,
        injected_session=injected_session,
        device=torch.device(f"cuda:{int(device_id)}"),
    )
