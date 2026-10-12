from __future__ import annotations

import ctypes
import glob
import os
import site
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("ORT_CUDA_CONV_ALLOW_TF32", "0")
os.environ.setdefault("ORT_CUDA_MATMUL_ALLOW_TF32", "0")

import onnxruntime as ort


CUDA_LIB_PATTERNS = [
    "nvidia/cuda_runtime/lib/libcudart.so*",
    "nvidia/cublas/lib/libcublas.so*",
    "nvidia/cublas/lib/libcublasLt.so*",
    "nvidia/cudnn/lib/libcudnn.so*",
    "nvidia/cufft/lib/libcufft.so*",
    "nvidia/curand/lib/libcurand.so*",
    "nvidia/cusolver/lib/libcusolver.so*",
    "nvidia/nvjitlink/lib/libnvJitLink.so*",
    "nvidia/cusparse/lib/libcusparse.so*",
]
CUDA_RUNTIME_READY = False

STRICT_SESSION_PROFILE = "strict"
THROUGHPUT_SESSION_PROFILE = "throughput"
_SUPPORTED_SESSION_PROFILES = frozenset(
    {
        STRICT_SESSION_PROFILE,
        THROUGHPUT_SESSION_PROFILE,
    }
)


class CudaSessionWrapper:
    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        batch_input_names: tuple[str, ...] | None = None,
        parallel_fallback_sessions: tuple[ort.InferenceSession, ...] = (),
    ) -> None:
        self._session = session
        self._batch_input_names = (None if batch_input_names is None else tuple(str(name) for name in batch_input_names))
        self._parallel_fallback_sessions = (self._session, *parallel_fallback_sessions)
        self._fallback_executor = (
            ThreadPoolExecutor(
                max_workers=len(self._parallel_fallback_sessions),
                thread_name_prefix="ort_cuda_fallback",
            )
            if len(self._parallel_fallback_sessions) > 1
            else None
        )

    def __getattr__(self, name: str):
        return getattr(self._session, name)

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def get_providers(self):
        return self._session.get_providers()

    def run(self, output_names, input_feed, run_options=None):
        slice_names = self._slice_input_names()
        batch_size = self._batch_size_from_feed(input_feed, slice_names=slice_names)
        if batch_size <= 1:
            return self._session.run(output_names, input_feed, run_options)
        if not self._needs_batch_fallback(input_feed, slice_names=slice_names):
            try:
                return self._session.run(output_names, input_feed, run_options)
            except Exception as exc:
                if not self._is_batch_dimension_error(exc):
                    raise

        return self._run_fallback_sliced(
            output_names,
            input_feed,
            run_options=run_options,
            slice_names=slice_names,
            batch_size=batch_size,
        )

    def _slice_input_names(self) -> tuple[str, ...]:
        if self._batch_input_names is None:
            return tuple(
                str(meta.name)
                for meta in self._session.get_inputs()
                if not str(meta.name).startswith("approx_override::")
            )
        return self._batch_input_names

    def _batch_size_from_feed(self, input_feed, *, slice_names: tuple[str, ...] | None = None) -> int:
        batch_size = 1
        names = self._slice_input_names() if slice_names is None else slice_names
        for name in names:
            value = input_feed.get(str(name))
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                batch_size = max(batch_size, int(value.shape[0]))
        return batch_size

    def _needs_batch_fallback(self, input_feed, *, slice_names: tuple[str, ...]) -> bool:
        input_meta_by_name = {str(meta.name): meta for meta in self._session.get_inputs()}
        for name in slice_names:
            value = input_feed.get(str(name))
            if not isinstance(value, np.ndarray) or value.ndim < 1:
                continue
            if int(value.shape[0]) <= 1:
                continue
            meta = input_meta_by_name.get(str(name))
            if meta is None:
                continue
            shape = tuple(meta.shape)
            if not shape:
                continue
            first_dim = shape[0]
            if isinstance(first_dim, int) and int(first_dim) == 1:
                return True
        return False

    @staticmethod
    def _is_batch_dimension_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "InvalidArgument" in exc.__class__.__name__
            or "Got invalid dimensions for input" in message
            or "Expected: 1" in message
        )

    def _slice_feed_for_batch_idx(
        self,
        input_feed,
        *,
        slice_names: tuple[str, ...],
        batch_size: int,
        batch_idx: int,
    ):
        sliced_feed = {}
        for name, value in input_feed.items():
            if (
                str(name) in slice_names
                and isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == batch_size
            ):
                sliced_feed[name] = value[batch_idx : batch_idx + 1]
            else:
                sliced_feed[name] = value
        return sliced_feed

    def _run_fallback_sliced(
        self,
        output_names,
        input_feed,
        *,
        run_options,
        slice_names: tuple[str, ...],
        batch_size: int,
    ):
        if len(self._parallel_fallback_sessions) == 1 or self._fallback_executor is None:
            per_output: list[list[np.ndarray]] | None = None
            for batch_idx in range(batch_size):
                sliced_feed = self._slice_feed_for_batch_idx(
                    input_feed,
                    slice_names=slice_names,
                    batch_size=batch_size,
                    batch_idx=batch_idx,
                )
                current = self._session.run(output_names, sliced_feed, run_options)
                if per_output is None:
                    per_output = [[] for _ in current]
                for slot, value in enumerate(current):
                    per_output[slot].append(np.asarray(value))
            return self._merge_sliced_outputs(per_output, output_names, input_feed, run_options)

        worker_count = min(len(self._parallel_fallback_sessions), batch_size)
        batch_groups = [
            list(range(worker_idx, batch_size, worker_count))
            for worker_idx in range(worker_count)
        ]

        def worker(session: ort.InferenceSession, indices: list[int]):
            local: list[tuple[int, list[np.ndarray]]] = []
            for batch_idx in indices:
                sliced_feed = self._slice_feed_for_batch_idx(
                    input_feed,
                    slice_names=slice_names,
                    batch_size=batch_size,
                    batch_idx=batch_idx,
                )
                current = session.run(output_names, sliced_feed, run_options)
                local.append((batch_idx, [np.asarray(value) for value in current]))
            return local

        futures = [
            self._fallback_executor.submit(worker, session, indices)
            for session, indices in zip(self._parallel_fallback_sessions, batch_groups, strict=False)
            if indices
        ]
        ordered_outputs: list[list[np.ndarray]] | None = None
        output_slots: list[list[np.ndarray]] | None = None
        if futures:
            ordered = sorted(
                (item for future in futures for item in future.result()),
                key=lambda entry: entry[0],
            )
            for _, current in ordered:
                if output_slots is None:
                    output_slots = [[] for _ in current]
                for slot, value in enumerate(current):
                    output_slots[slot].append(np.asarray(value))
            ordered_outputs = output_slots
        return self._merge_sliced_outputs(ordered_outputs, output_names, input_feed, run_options)

    def _merge_sliced_outputs(self, per_output, output_names, input_feed, run_options):
        if per_output is None:
            return self._session.run(output_names, input_feed, run_options)
        merged: list[np.ndarray] = []
        for slot_values in per_output:
            first = slot_values[0]
            if first.ndim == 0:
                merged.append(np.asarray(slot_values))
            else:
                merged.append(np.concatenate(slot_values, axis=0))
        return merged

    def close(self) -> None:
        if self._fallback_executor is not None:
            self._fallback_executor.shutdown(wait=True, cancel_futures=False)


def ensure_gpu_runtime() -> None:
    global CUDA_RUNTIME_READY
    if CUDA_RUNTIME_READY:
        return
    for base in site.getsitepackages():
        for pattern in CUDA_LIB_PATTERNS:
            for lib_path in sorted(glob.glob(os.path.join(base, pattern))):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
    ort.set_default_logger_severity(4)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is unavailable in the current pytorch environment")
    CUDA_RUNTIME_READY = True


def _normalize_session_profile(session_profile: str) -> str:
    normalized = str(session_profile).strip().lower()
    if normalized not in _SUPPORTED_SESSION_PROFILES:
        choices = ", ".join(sorted(_SUPPORTED_SESSION_PROFILES))
        raise ValueError(f"Unsupported CUDA session_profile={session_profile!r}; expected one of: {choices}")
    return normalized


def _session_options(*, session_profile: str = STRICT_SESSION_PROFILE) -> ort.SessionOptions:
    profile = _normalize_session_profile(session_profile)
    options = ort.SessionOptions()
    options.log_severity_level = 4
    if profile == STRICT_SESSION_PROFILE:
        # Keep the historical strict configuration unchanged for reproducible checks.
        try:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        except Exception:
            pass
        try:
            options.use_deterministic_compute = True
        except Exception:
            pass
        try:
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        except Exception:
            pass
    else:
        # This profile is deliberately TF32-free. It is for repeated batched
        # throughput work where ORT graph rewrites are acceptable, not a new
        # numerical contract.
        try:
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        except Exception:
            pass
        try:
            options.use_deterministic_compute = False
            options.intra_op_num_threads = 0
            options.inter_op_num_threads = 0
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = True
            options.enable_mem_reuse = True
        except Exception:
            pass
    return options


def _cuda_provider_config(*, device_id: int, session_profile: str) -> list[tuple[str, dict[str, str]]]:
    _normalize_session_profile(session_profile)
    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": str(int(device_id)),
                "do_copy_in_default_stream": "1",
                "cudnn_conv_algo_search": "DEFAULT",
                "use_tf32": "0",
            },
        )
    ]


def create_cuda_session(
    model_path: Path,
    *,
    batch_input_names: tuple[str, ...] | None = None,
    device_id: int = 0,
    custom_ops: Path | None = None,
    fallback_parallelism: int = 1,
    session_profile: str = STRICT_SESSION_PROFILE,
) -> tuple[ort.InferenceSession, str]:
    ensure_gpu_runtime()
    profile = _normalize_session_profile(session_profile)
    provider_config = _cuda_provider_config(device_id=int(device_id), session_profile=profile)

    def build_session() -> ort.InferenceSession:
        options = _session_options(session_profile=profile)
        if custom_ops is not None:
            options.register_custom_ops_library(str(Path(custom_ops).resolve()))
        return ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=provider_config,
        )

    pool_size = max(1, int(fallback_parallelism))
    sessions = [build_session() for _ in range(pool_size)]
    provider = sessions[0].get_providers()[0]
    if provider != "CUDAExecutionProvider":
        raise RuntimeError(f"{model_path.name} did not bind to CUDAExecutionProvider; got {provider}")
    return CudaSessionWrapper(
        sessions[0],
        batch_input_names=batch_input_names,
        parallel_fallback_sessions=tuple(sessions[1:]),
    ), provider


def create_cuda_session_from_bytes(
    model_bytes: bytes,
    *,
    batch_input_names: tuple[str, ...] | None = None,
    device_id: int = 0,
    custom_ops: Path | None = None,
    fallback_parallelism: int = 1,
    session_profile: str = STRICT_SESSION_PROFILE,
) -> tuple[ort.InferenceSession, str]:
    ensure_gpu_runtime()
    profile = _normalize_session_profile(session_profile)
    provider_config = _cuda_provider_config(device_id=int(device_id), session_profile=profile)

    def build_session() -> ort.InferenceSession:
        options = _session_options(session_profile=profile)
        if custom_ops is not None:
            options.register_custom_ops_library(str(Path(custom_ops).resolve()))
        return ort.InferenceSession(
            model_bytes,
            sess_options=options,
            providers=provider_config,
        )

    pool_size = max(1, int(fallback_parallelism))
    sessions = [build_session() for _ in range(pool_size)]
    provider = sessions[0].get_providers()[0]
    if provider != "CUDAExecutionProvider":
        raise RuntimeError(f"Model bytes did not bind to CUDAExecutionProvider; got {provider}")
    return CudaSessionWrapper(
        sessions[0],
        batch_input_names=batch_input_names,
        parallel_fallback_sessions=tuple(sessions[1:]),
    ), provider
