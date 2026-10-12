from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .errors import LabelGenerationError


_OUTPUT_DTYPES = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
}


class PersistentCnnIoBinding:
    """Reuse CUDA inputs, outputs, bindings, and one stable LUT allocation.

    The frozen evaluation inputs are either retained on the GPU (``device``)
    or copied through one reusable device buffer per batch shape (``staged``).
    ``auto`` first tries device residency and falls back only for a recognized
    CUDA allocation failure.
    """

    def __init__(
        self,
        session: Any,
        *,
        ort_module: Any,
        input_name: str,
        lut_input_name: str,
        output_name: str,
        device_id: int,
        input_residency: str,
        preallocate_outputs: bool,
        lut_shape: tuple[int, int] = (256, 256),
    ) -> None:
        requested = str(input_residency).strip().lower()
        if requested not in {"auto", "device", "staged"}:
            raise LabelGenerationError(
                "execution.input_residency must be 'auto', 'device', or 'staged'"
            )
        self._session = session
        self._ort = ort_module
        self._input_name = str(input_name)
        self._lut_input_name = str(lut_input_name)
        self._output_name = str(output_name)
        self._device_id = int(device_id)
        self._requested_residency = requested
        self._input_residency = "device" if requested == "auto" else requested
        self._preallocate_outputs = bool(preallocate_outputs)
        if (
            len(lut_shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in lut_shape)
        ):
            raise LabelGenerationError("persistent CNN LUT shape must contain two positive integers")
        self._lut_shape = tuple(int(value) for value in lut_shape)
        self._lut_device = self._ort.OrtValue.ortvalue_from_shape_and_type(
            self._lut_shape,
            np.int32,
            device_type="cuda",
            device_id=self._device_id,
        )
        self._device_batches: tuple[Any, ...] = ()
        self._host_batches: tuple[np.ndarray, ...] = ()
        self._batch_sizes: tuple[int, ...] = ()
        self._bindings: tuple[Any, ...] = ()
        self._staged_inputs: dict[int, Any] = {}
        self._staged_bindings: dict[int, Any] = {}
        self._output_buffers: dict[int, Any] = {}
        self._prepared_input: np.ndarray | None = None
        self._lut_update_count = 0
        self._run_count = 0
        self._closed = False

    @staticmethod
    def _is_allocation_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "alloc_failed",
                "out of memory",
                "cudaerror memory allocation",
                "cuda_error_out_of_memory",
                "cublas_status_alloc_failed",
            )
        )

    @property
    def metadata(self) -> dict[str, object]:
        if self._input_residency == "device":
            input_bytes = int(
                sum(np.prod(value.shape) * np.dtype(np.float32).itemsize for value in self._host_batches)
            )
            input_allocations = len(self._device_batches)
        else:
            input_bytes = int(
                sum(value.array.nbytes for value in self._staged_inputs.values())
                if self._staged_inputs
                and all(hasattr(value, "array") for value in self._staged_inputs.values())
                else 0
            )
            input_allocations = len(self._staged_inputs)
        return {
            "requested_input_residency": self._requested_residency,
            "input_residency": self._input_residency,
            "batch_sizes": list(self._batch_sizes),
            "prepared_batch_count": len(self._batch_sizes),
            "input_device_allocation_count": input_allocations,
            "estimated_input_device_bytes": input_bytes,
            "preallocated_outputs": self._preallocate_outputs,
            "output_device_allocation_count": len(self._output_buffers),
            "stable_lut_device_buffer": self._lut_device is not None,
            "lut_shape": list(self._lut_shape),
            "lut_dtype": "int32",
            "lut_update_count": self._lut_update_count,
            "run_count": self._run_count,
        }

    def prepare(self, inputs: np.ndarray, *, batch_size: int) -> None:
        self._ensure_open()
        if self._prepared_input is inputs and self._batch_sizes:
            return
        if self._prepared_input is not None:
            raise LabelGenerationError(
                "persistent CNN runtime cannot replace the frozen evaluation input array"
            )
        batches = tuple(
            np.ascontiguousarray(inputs[start : start + batch_size], dtype=np.float32)
            for start in range(0, inputs.shape[0], batch_size)
        )
        if not batches:
            raise LabelGenerationError("persistent CNN runtime requires at least one input batch")
        self._host_batches = batches
        self._batch_sizes = tuple(int(batch.shape[0]) for batch in batches)
        try:
            self._prepare_for_residency(self._input_residency)
        except Exception as exc:
            self._release_bindings()
            if self._requested_residency != "auto" or not self._is_allocation_error(exc):
                raise LabelGenerationError(
                    f"unable to prepare persistent CUDA inputs: {type(exc).__name__}: {exc}"
                ) from exc
            self._input_residency = "staged"
            try:
                self._prepare_for_residency("staged")
            except Exception as staged_exc:
                self._release_bindings()
                raise LabelGenerationError(
                    "device-resident CNN inputs exhausted CUDA memory and staged fallback failed: "
                    f"{type(staged_exc).__name__}: {staged_exc}"
                ) from staged_exc
        self._prepared_input = inputs

    def _prepare_for_residency(self, residency: str) -> None:
        if residency == "device":
            self._device_batches = tuple(
                self._ort.OrtValue.ortvalue_from_numpy(
                    batch,
                    device_type="cuda",
                    device_id=self._device_id,
                )
                for batch in self._host_batches
            )
            self._bindings = tuple(
                self._create_binding(device_batch, batch_size)
                for device_batch, batch_size in zip(
                    self._device_batches,
                    self._batch_sizes,
                    strict=True,
                )
            )
            return

        trailing_shape = tuple(int(value) for value in self._host_batches[0].shape[1:])
        for batch_size in sorted(set(self._batch_sizes)):
            device_input = self._ort.OrtValue.ortvalue_from_shape_and_type(
                (batch_size, *trailing_shape),
                np.float32,
                device_type="cuda",
                device_id=self._device_id,
            )
            self._staged_inputs[batch_size] = device_input
            self._staged_bindings[batch_size] = self._create_binding(device_input, batch_size)

    def _output_shape(self, batch_size: int) -> tuple[int, ...] | None:
        metadata = next(
            (item for item in self._session.get_outputs() if str(item.name) == self._output_name),
            None,
        )
        if metadata is None:
            raise LabelGenerationError(f"CNN output metadata is missing {self._output_name!r}")
        raw_shape = tuple(metadata.shape)
        if not raw_shape:
            return None
        shape = [int(batch_size)]
        for dimension in raw_shape[1:]:
            if not isinstance(dimension, (int, np.integer)) or int(dimension) <= 0:
                return None
            shape.append(int(dimension))
        return tuple(shape)

    def _create_binding(self, device_input: Any, batch_size: int) -> Any:
        binding = self._session.io_binding()
        binding.bind_ortvalue_input(self._input_name, device_input)
        output_shape = self._output_shape(batch_size) if self._preallocate_outputs else None
        output_meta = next(
            item for item in self._session.get_outputs() if str(item.name) == self._output_name
        )
        output_dtype = _OUTPUT_DTYPES.get(str(output_meta.type))
        if output_shape is not None and output_dtype is not None:
            output = self._output_buffers.get(batch_size)
            if output is None:
                output = self._ort.OrtValue.ortvalue_from_shape_and_type(
                    output_shape,
                    output_dtype,
                    device_type="cuda",
                    device_id=self._device_id,
                )
                self._output_buffers[batch_size] = output
            binding.bind_ortvalue_output(self._output_name, output)
        else:
            binding.bind_output(self._output_name, "cuda", self._device_id)
        return binding

    def update_lut(self, lut_delta: np.ndarray) -> None:
        self._ensure_open()
        if (
            not isinstance(lut_delta, np.ndarray)
            or lut_delta.dtype != np.int32
            or lut_delta.shape != self._lut_shape
            or not lut_delta.flags.c_contiguous
        ):
            raise LabelGenerationError(
                "persistent CNN LUT update must be a C-contiguous int32 array with shape "
                f"{self._lut_shape}; got {getattr(lut_delta, 'dtype', None)} "
                f"{getattr(lut_delta, 'shape', None)}"
            )
        self._lut_device.update_inplace(lut_delta)
        self._lut_update_count += 1

    def run(
        self,
        *,
        after_first_batch: Callable[[], None] | None = None,
    ) -> np.ndarray:
        self._ensure_open()
        if not self._batch_sizes or self._lut_update_count <= 0:
            raise LabelGenerationError("persistent CNN CUDA inputs and LUT must be prepared before run")
        outputs: list[np.ndarray] = []
        callback = after_first_batch
        for batch_index, batch_size in enumerate(self._batch_sizes):
            if self._input_residency == "device":
                binding = self._bindings[batch_index]
            else:
                device_input = self._staged_inputs[batch_size]
                device_input.update_inplace(self._host_batches[batch_index])
                binding = self._staged_bindings[batch_size]
                binding.bind_ortvalue_input(self._input_name, device_input)
            binding.bind_ortvalue_input(self._lut_input_name, self._lut_device)
            binding.synchronize_inputs()
            self._session.run_with_iobinding(binding)
            binding.synchronize_outputs()
            if callback is not None:
                callback()
                callback = None
            copied = binding.copy_outputs_to_cpu()
            if len(copied) != 1:
                raise LabelGenerationError(
                    f"CNN CUDA runtime returned {len(copied)} outputs; expected one"
                )
            value = np.asarray(copied[0])
            if value.dtype != np.float32 or value.shape[0] != batch_size:
                raise LabelGenerationError(
                    f"CNN CUDA runtime returned invalid output dtype/shape {value.dtype} {value.shape}"
                )
            outputs.append(value)
        self._run_count += 1
        return np.concatenate(outputs, axis=0)

    def close(self) -> None:
        if self._closed:
            return
        self._release_bindings()
        self._host_batches = ()
        self._batch_sizes = ()
        self._prepared_input = None
        self._lut_device = None
        self._closed = True

    def _release_bindings(self) -> None:
        bindings = (*self._bindings, *self._staged_bindings.values())
        for binding in bindings:
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()
        self._device_batches = ()
        self._bindings = ()
        self._staged_inputs.clear()
        self._staged_bindings.clear()
        self._output_buffers.clear()

    def _ensure_open(self) -> None:
        if self._closed:
            raise LabelGenerationError("persistent CNN CUDA binding is closed")


__all__ = ["PersistentCnnIoBinding"]
