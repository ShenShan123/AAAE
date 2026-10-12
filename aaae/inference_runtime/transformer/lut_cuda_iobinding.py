"""Persistent CUDA I/O binding for one shared S8S8 LUT graph input.

This module owns only transport and buffer lifecycle.  It does not interpret
LUT values, change the graph, or implement approximate-matmul semantics.
Callers prepare image batches in their required order, update the current LUT
once per design, then choose either fully CUDA-resident or size-keyed staged
input buffers for every run.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import onnxruntime as ort


LUT_SHAPE = (256, 256)


class _RuntimeLike(Protocol):
    session: Any
    predictor_input_name: str
    provider: str


OutputShapeResolver = Callable[[int], Mapping[str, Sequence[int]]]
OutputHandler = Callable[["IoBindingBatchOutput"], None]


@dataclass(frozen=True)
class IoBindingBatchOutput:
    """CPU outputs for one caller-ordered input batch."""

    batch_index: int
    sample_start: int
    sample_end: int
    outputs: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class IoBindingRunResult:
    """Summary of one design execution.

    ``logits`` is populated only when ``run_all(return_logits=True)`` was
    requested.  Bulk evaluation should normally use ``output_handler`` and
    leave it unset to avoid retaining every design's full logits.
    """

    sample_count: int
    batch_count: int
    elapsed_seconds: float
    output_names: tuple[str, ...]
    logits: np.ndarray | None


@dataclass(frozen=True)
class PersistentIoBindingMetadata:
    """Observable buffer state for diagnostics and focused tests."""

    device_id: int
    provider: str
    session_profile: str
    predictor_input_name: str
    lut_input_name: str
    output_names: tuple[str, ...]
    preallocate_outputs: bool
    input_residency: str
    batch_sizes: tuple[int, ...]
    input_device_pointers: tuple[int, ...]
    input_device_pointers_by_batch_size: tuple[tuple[int, tuple[int, ...]], ...]
    lut_device_pointer: int | None
    output_device_pointers: tuple[tuple[int, tuple[int, ...]], ...]
    lut_update_count: int
    closed: bool


_ORT_OUTPUT_TO_NUMPY: dict[str, np.dtype[Any]] = {
    "tensor(bool)": np.dtype(np.bool_),
    "tensor(float)": np.dtype(np.float32),
    "tensor(double)": np.dtype(np.float64),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(int16)": np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
}


class PersistentLutCudaIoBindingRunner:
    """Reuse CUDA inputs and a stable shared-LUT device buffer.

    The wrapped runtime must expose the public attributes used by
    ``CompiledLutCudaRuntime``: ``session``, ``predictor_input_name``, and
    ``provider``.  It remains owned by the caller; :meth:`close` releases only
    I/O bindings and OrtValues created by this runner.

    ``session_profile`` is metadata supplied by the caller, for example
    ``"strict"`` or ``"performance_b32"``.  The runner never changes session
    options, provider options, or numerical injection semantics.

    ``input_residency="device"`` retains one CUDA input OrtValue per prepared
    batch. ``input_residency="staged"`` retains caller-normalized host batches
    and only one CUDA input OrtValue per distinct batch size. The latter trades
    host-to-device transfer per run for materially lower GPU input residency.
    """

    def __init__(
        self,
        runtime: _RuntimeLike,
        *,
        lut_input_name: str,
        device_id: int,
        preallocate_outputs: bool = True,
        output_shape_resolver: OutputShapeResolver | None = None,
        session_profile: str = "strict",
        input_residency: str = "device",
        ort_module: Any | None = None,
    ) -> None:
        if not str(lut_input_name):
            raise ValueError("lut_input_name must be non-empty")
        if int(device_id) < 0:
            raise ValueError("device_id must be non-negative")
        if not str(session_profile):
            raise ValueError("session_profile must be non-empty")
        normalized_input_residency = str(input_residency).strip().lower()
        if normalized_input_residency not in {"device", "staged"}:
            raise ValueError("input_residency must be 'device' or 'staged'")

        self._runtime = runtime
        self._session = runtime.session
        self._predictor_input_name = str(runtime.predictor_input_name)
        self._lut_input_name = str(lut_input_name)
        self._provider = str(runtime.provider)
        self._device_id = int(device_id)
        self._preallocate_outputs = bool(preallocate_outputs)
        self._output_shape_resolver = output_shape_resolver
        self._session_profile = str(session_profile)
        self._input_residency = normalized_input_residency
        self._ort = ort if ort_module is None else ort_module

        self._output_metas = tuple(self._session.get_outputs())
        if not self._output_metas:
            raise ValueError("runtime session has no outputs")
        self._output_names = tuple(str(meta.name) for meta in self._output_metas)
        if len(set(self._output_names)) != len(self._output_names):
            raise ValueError("runtime session has duplicate output names")

        self._lut_device = self._ort.OrtValue.ortvalue_from_shape_and_type(
            LUT_SHAPE,
            np.int32,
            device_type="cuda",
            device_id=self._device_id,
        )
        self._lut_is_initialized = False
        self._lut_update_count = 0
        self._device_batches: tuple[Any, ...] = ()
        self._host_batches: tuple[np.ndarray, ...] = ()
        self._batch_sizes: tuple[int, ...] = ()
        self._bindings_by_batch: tuple[Any, ...] = ()
        self._staged_input_devices_by_batch_size: dict[int, Any] = {}
        self._staged_bindings_by_batch_size: dict[int, Any] = {}
        self._output_buffers_by_batch_size: dict[int, tuple[Any, ...]] = {}
        self._closed = False

    @property
    def metadata(self) -> PersistentIoBindingMetadata:
        if self._input_residency == "device":
            input_pointers = tuple(int(value.data_ptr()) for value in self._device_batches)
            grouped_input_pointers: dict[int, list[int]] = {}
            for batch_size, pointer in zip(self._batch_sizes, input_pointers, strict=True):
                grouped_input_pointers.setdefault(batch_size, []).append(pointer)
        else:
            grouped_input_pointers = {
                batch_size: [int(value.data_ptr())]
                for batch_size, value in self._staged_input_devices_by_batch_size.items()
            }
            input_pointers = tuple(
                pointers[0]
                for _, pointers in sorted(grouped_input_pointers.items())
            )
        output_pointers = tuple(
            (batch_size, tuple(int(value.data_ptr()) for value in buffers))
            for batch_size, buffers in sorted(self._output_buffers_by_batch_size.items())
        )
        return PersistentIoBindingMetadata(
            device_id=self._device_id,
            provider=self._provider,
            session_profile=self._session_profile,
            predictor_input_name=self._predictor_input_name,
            lut_input_name=self._lut_input_name,
            output_names=self._output_names,
            preallocate_outputs=self._preallocate_outputs,
            input_residency=self._input_residency,
            batch_sizes=self._batch_sizes,
            input_device_pointers=input_pointers,
            input_device_pointers_by_batch_size=tuple(
                (batch_size, tuple(pointers))
                for batch_size, pointers in sorted(grouped_input_pointers.items())
            ),
            lut_device_pointer=(None if self._lut_device is None else int(self._lut_device.data_ptr())),
            output_device_pointers=output_pointers,
            lut_update_count=self._lut_update_count,
            closed=self._closed,
        )

    def prepare_batches(self, batch_inputs: Sequence[np.ndarray]) -> PersistentIoBindingMetadata:
        """Prepare caller-ordered, non-empty batches according to input residency.

        The sequence order is preserved exactly.  Every batch must share the
        same non-batch shape and use a positive leading batch dimension.  A
        later call intentionally replaces the old input/binding cache. In
        ``staged`` mode callers must leave prepared host arrays unchanged until
        the next :meth:`prepare_batches` or :meth:`close` call.
        """

        self._ensure_open()
        normalized = self._normalize_batches(batch_inputs)
        self._release_prepared_bindings()
        self._device_batches = ()
        self._host_batches = ()
        self._batch_sizes = ()
        self._batch_sizes = tuple(int(batch.shape[0]) for batch in normalized)
        if self._input_residency == "device":
            self._device_batches = tuple(
                self._ort.OrtValue.ortvalue_from_numpy(batch, device_type="cuda", device_id=self._device_id)
                for batch in normalized
            )
            self._bindings_by_batch = tuple(
                self._create_bound_batch(device_batch, batch_size)
                for device_batch, batch_size in zip(self._device_batches, self._batch_sizes, strict=True)
            )
        else:
            self._host_batches = normalized
            trailing_shape = tuple(int(dim) for dim in normalized[0].shape[1:])
            for batch_size in sorted(set(self._batch_sizes)):
                staged_input = self._ort.OrtValue.ortvalue_from_shape_and_type(
                    (batch_size, *trailing_shape),
                    np.float32,
                    device_type="cuda",
                    device_id=self._device_id,
                )
                self._staged_input_devices_by_batch_size[batch_size] = staged_input
                self._staged_bindings_by_batch_size[batch_size] = self._create_bound_batch(
                    staged_input,
                    batch_size,
                )
        return self.metadata

    def update_lut(self, lut: np.ndarray) -> PersistentIoBindingMetadata:
        """Copy one shared INT32 residual LUT into the stable CUDA buffer."""

        self._ensure_open()
        array = np.ascontiguousarray(np.asarray(lut, dtype=np.int32))
        if array.shape != LUT_SHAPE:
            raise ValueError(f"LUT must have shape {LUT_SHAPE}, got {array.shape}")
        self._lut_device.update_inplace(array)
        self._lut_is_initialized = True
        self._lut_update_count += 1
        return self.metadata

    def run_all(
        self,
        batch_inputs: Sequence[np.ndarray] | None = None,
        *,
        output_handler: OutputHandler | None = None,
        return_logits: bool = False,
    ) -> IoBindingRunResult:
        """Run all caller-ordered batches using the current device LUT.

        Passing ``batch_inputs`` replaces the retained device-batch cache before
        execution.  Pass ``None`` to reuse batches prepared by
        :meth:`prepare_batches`.  Full logits are accumulated only when
        ``return_logits=True``; ``output_handler`` receives each CPU batch in
        order without retaining all batches in this class.
        """

        self._ensure_open()
        if batch_inputs is not None:
            self.prepare_batches(batch_inputs)
        if not self._batch_sizes:
            raise RuntimeError("no input batches are prepared")
        if not self._lut_is_initialized:
            raise RuntimeError("LUT must be initialized with update_lut before run_all")
        if return_logits and len(self._output_names) != 1:
            raise ValueError("return_logits=True requires a single-output runtime; use output_handler for multiple outputs")

        collected_logits: list[np.ndarray] | None = [] if return_logits else None
        sample_start = 0
        start_time = perf_counter()
        for batch_index, batch_size in enumerate(self._batch_sizes):
            if self._input_residency == "device":
                binding = self._bindings_by_batch[batch_index]
            else:
                staged_input = self._staged_input_devices_by_batch_size[batch_size]
                staged_input.update_inplace(self._host_batches[batch_index])
                binding = self._staged_bindings_by_batch_size[batch_size]
                # Like the LUT binding below, ORT snapshots device inputs when
                # they are bound. Refresh the binding after staging new bytes.
                binding.bind_ortvalue_input(self._predictor_input_name, staged_input)
            # ORT snapshots a bound input internally. Rebind this stable
            # OrtValue after update_inplace() so every binding observes the
            # current design's LUT, while the immutable image input remains
            # permanently bound from prepare_batches().
            binding.bind_ortvalue_input(self._lut_input_name, self._lut_device)
            binding.synchronize_inputs()
            self._session.run_with_iobinding(binding)
            binding.synchronize_outputs()

            if output_handler is None and collected_logits is None:
                sample_start += batch_size
                continue

            outputs = tuple(np.asarray(value) for value in binding.copy_outputs_to_cpu())
            self._validate_output_batch(outputs, batch_size)
            sample_end = sample_start + batch_size
            payload = IoBindingBatchOutput(
                batch_index=batch_index,
                sample_start=sample_start,
                sample_end=sample_end,
                outputs=outputs,
            )
            if output_handler is not None:
                output_handler(payload)
            if collected_logits is not None:
                collected_logits.append(outputs[0])
            sample_start = sample_end

        elapsed_seconds = perf_counter() - start_time
        logits = None if collected_logits is None else np.concatenate(collected_logits, axis=0)
        return IoBindingRunResult(
            sample_count=sum(self._batch_sizes),
            batch_count=len(self._batch_sizes),
            elapsed_seconds=elapsed_seconds,
            output_names=self._output_names,
            logits=logits,
        )

    def close(self) -> None:
        """Release runner-owned bindings and device buffers without closing its session."""

        if self._closed:
            return
        self._release_prepared_bindings()
        self._device_batches = ()
        self._host_batches = ()
        self._batch_sizes = ()
        self._lut_device = None
        self._closed = True

    def _create_bound_batch(self, input_device: Any, batch_size: int) -> Any:
        """Create a permanent binding for one prepared device batch.

        A device-resident image or staged input keeps a stable address for the
        lifetime of its cache. ORT snapshots dynamic inputs, so run_all()
        refreshes staged input bindings after each update and LUT bindings
        after each LUT update. Output storage may still be shared by all
        sequential batches of the same shape.
        """

        binding = self._session.io_binding()
        binding.bind_ortvalue_input(self._predictor_input_name, input_device)
        if self._preallocate_outputs:
            buffers = self._output_buffers_by_batch_size.get(batch_size)
            if buffers is None:
                shapes = self._resolve_output_shapes(batch_size)
                built: list[Any] = []
                for meta in self._output_metas:
                    output_name = str(meta.name)
                    shape = tuple(int(dim) for dim in shapes[output_name])
                    dtype = self._output_dtype(meta)
                    built.append(
                        self._ort.OrtValue.ortvalue_from_shape_and_type(
                            shape,
                            dtype,
                            device_type="cuda",
                            device_id=self._device_id,
                        )
                    )
                buffers = tuple(built)
                self._output_buffers_by_batch_size[batch_size] = buffers
            for output_name, value in zip(self._output_names, buffers, strict=True):
                binding.bind_ortvalue_output(output_name, value)
        else:
            for output_name in self._output_names:
                binding.bind_output(output_name, "cuda", self._device_id)
        return binding

    def _release_prepared_bindings(self) -> None:
        bindings = (*self._bindings_by_batch, *self._staged_bindings_by_batch_size.values())
        for binding in bindings:
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()
        self._bindings_by_batch = ()
        self._staged_input_devices_by_batch_size.clear()
        self._staged_bindings_by_batch_size.clear()
        self._output_buffers_by_batch_size.clear()

    def _resolve_output_shapes(self, batch_size: int) -> Mapping[str, Sequence[int]]:
        if self._output_shape_resolver is not None:
            resolved = self._output_shape_resolver(batch_size)
            names = set(str(name) for name in resolved)
            if names != set(self._output_names):
                raise ValueError(
                    "output_shape_resolver must return exactly the runtime output names; "
                    f"expected={self._output_names}, got={tuple(sorted(names))}"
                )
            return {str(name): tuple(int(dim) for dim in shape) for name, shape in resolved.items()}

        resolved: dict[str, tuple[int, ...]] = {}
        for meta in self._output_metas:
            raw_shape = tuple(getattr(meta, "shape", ()))
            if not raw_shape:
                raise ValueError(
                    f"cannot preallocate scalar output '{meta.name}'; pass preallocate_outputs=False or output_shape_resolver"
                )
            shape = [batch_size]
            for dim in raw_shape[1:]:
                if not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
                    raise ValueError(
                        f"cannot infer dynamic non-batch output shape for '{meta.name}'; "
                        "pass preallocate_outputs=False or output_shape_resolver"
                    )
                shape.append(int(dim))
            resolved[str(meta.name)] = tuple(shape)
        return resolved

    @staticmethod
    def _output_dtype(meta: Any) -> np.dtype[Any]:
        ort_type = str(getattr(meta, "type", ""))
        dtype = _ORT_OUTPUT_TO_NUMPY.get(ort_type)
        if dtype is None:
            raise ValueError(f"unsupported preallocated output type {ort_type!r} for '{meta.name}'")
        return dtype

    @staticmethod
    def _normalize_batches(batch_inputs: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        if len(batch_inputs) == 0:
            raise ValueError("batch_inputs must contain at least one batch")
        normalized: list[np.ndarray] = []
        trailing_shape: tuple[int, ...] | None = None
        for batch_index, batch in enumerate(batch_inputs):
            array = np.ascontiguousarray(np.asarray(batch, dtype=np.float32))
            if array.ndim < 1 or int(array.shape[0]) <= 0:
                raise ValueError(f"batch {batch_index} must have a positive leading batch dimension")
            current_trailing_shape = tuple(int(dim) for dim in array.shape[1:])
            if trailing_shape is None:
                trailing_shape = current_trailing_shape
            elif current_trailing_shape != trailing_shape:
                raise ValueError(
                    f"batch {batch_index} has trailing shape {current_trailing_shape}, expected {trailing_shape}"
                )
            normalized.append(array)
        return tuple(normalized)

    def _validate_output_batch(self, outputs: tuple[np.ndarray, ...], batch_size: int) -> None:
        if len(outputs) != len(self._output_names):
            raise RuntimeError(f"I/O binding returned {len(outputs)} outputs, expected {len(self._output_names)}")
        for output_name, output in zip(self._output_names, outputs, strict=True):
            if output.ndim < 1 or int(output.shape[0]) != batch_size:
                raise RuntimeError(
                    f"I/O binding output '{output_name}' has shape {output.shape}, expected leading batch {batch_size}"
                )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("PersistentLutCudaIoBindingRunner is closed")
