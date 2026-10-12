from __future__ import annotations

import ctypes
import fcntl
import gc
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aaae.paths import CNN_ORT_INCLUDE_ROOT, NATIVE_ROOT, OUTPUT_ROOT
from .errors import LabelGenerationError
from .persistent_iobinding import PersistentCnnIoBinding
from .s8s8_conv_graph import export_s8s8_lut_conv_model

if TYPE_CHECKING:
    from aaae.inference_runtime.cnn.dataset_builders.real_labels import LabelGenerationPlan


# These must be set before ONNX Runtime is imported. They make the exact base
# and injected sessions use the same conservative floating-point policy.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("ORT_CUDA_CONV_ALLOW_TF32", "0")
os.environ.setdefault("ORT_CUDA_MATMUL_ALLOW_TF32", "0")

_NATIVE_SOURCE = NATIVE_ROOT / "cnn" / "s8s8_conv_cuda.cu"
_EXPECTED_HEADERS = (
    "onnxruntime_c_api.h",
    "onnxruntime_cxx_api.h",
    "onnxruntime_cxx_inline.h",
    "onnxruntime_ep_c_api.h",
    "onnxruntime_float16.h",
)
_NATIVE_DEPENDENCY_HANDLES: list[ctypes.CDLL] = []


def validate_residual_lut(
    lut_delta: np.ndarray,
    *,
    abs_bound: int | None = None,
) -> np.ndarray:
    """Validate the canonical dynamic signed residual LUT without coercion."""

    if not isinstance(lut_delta, np.ndarray) or lut_delta.dtype != np.int32:
        dtype = getattr(lut_delta, "dtype", None)
        raise LabelGenerationError(
            f"signed residual LUT must have dtype int32 without conversion; got {dtype}"
        )
    if lut_delta.shape != (256, 256):
        raise LabelGenerationError(
            f"signed residual LUT must have shape (256, 256); got {lut_delta.shape}"
        )
    if not lut_delta.flags.c_contiguous:
        raise LabelGenerationError("signed residual LUT must be C-contiguous")
    if abs_bound is not None:
        if isinstance(abs_bound, bool) or not isinstance(abs_bound, int) or abs_bound < 0:
            raise LabelGenerationError("signed residual LUT abs_bound must be a nonnegative integer")
        actual = int(np.max(np.abs(lut_delta.astype(np.int64, copy=False))))
        if actual > abs_bound:
            raise LabelGenerationError(
                f"signed residual LUT abs max {actual} exceeds the proven bound {abs_bound}"
            )
    return lut_delta


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - depends on deployment environment
        raise LabelGenerationError(f"ONNX Runtime import failed: {type(exc).__name__}: {exc}") from exc
    try:
        ort.preload_dlls(cuda=True, cudnn=True, directory="")
    except Exception as exc:
        raise LabelGenerationError(
            f"ONNX Runtime CUDA dependency preload failed: {type(exc).__name__}: {exc}"
        ) from exc
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise LabelGenerationError(
            "CUDAExecutionProvider is unavailable; CNN real labels never fall back to CPU"
        )
    return ort


def _cuda_capability(gpu_id: int) -> str:
    try:
        import torch

        count = int(torch.cuda.device_count())
        if gpu_id >= count:
            raise LabelGenerationError(
                f"execution.gpu_id={gpu_id} is unavailable; CUDA exposes {count} logical devices"
            )
        torch.cuda.set_device(gpu_id)
        major, minor = torch.cuda.get_device_capability(gpu_id)
    except LabelGenerationError:
        raise
    except Exception as exc:
        raise LabelGenerationError(
            f"unable to inspect CUDA device {gpu_id}: {type(exc).__name__}: {exc}"
        ) from exc
    if major <= 0 or minor < 0:
        raise LabelGenerationError(f"CUDA device {gpu_id} returned invalid capability {major}.{minor}")
    return f"{major}{minor}"


def _ort_sdk(ort: Any) -> tuple[Path, Path]:
    candidates: list[Path] = []
    configured = os.environ.get("CNN_ORT_INCLUDE_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(CNN_ORT_INCLUDE_ROOT)

    include_dir = next(
        (
            candidate.resolve()
            for candidate in candidates
            if all((candidate / name).is_file() for name in _EXPECTED_HEADERS)
        ),
        None,
    )
    if include_dir is None:
        raise LabelGenerationError(
            "ONNX Runtime development headers are unavailable; set CNN_ORT_INCLUDE_DIR "
            f"to an ORT {ort.__version__} include directory"
        )

    try:
        import onnxruntime.capi as capi
    except Exception as exc:  # pragma: no cover - installation-specific
        raise LabelGenerationError(f"unable to locate onnxruntime.capi: {exc}") from exc
    capi_dir = Path(capi.__file__).resolve().parent
    libraries = sorted(capi_dir.glob("libonnxruntime.so.*"), reverse=True)
    if not libraries:
        raise LabelGenerationError(f"ONNX Runtime shared library is missing under {capi_dir}")
    return include_dir, libraries[0].resolve()


def _build_custom_op(ort: Any, *, cuda_arch: str) -> Path:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise LabelGenerationError("nvcc is unavailable; the signed S8S8 CUDA custom op cannot be built")
    if not _NATIVE_SOURCE.is_file():
        raise LabelGenerationError(f"signed S8S8 CUDA source is missing: {_NATIVE_SOURCE}")
    include_dir, ort_library = _ort_sdk(ort)
    source = _NATIVE_SOURCE.read_bytes()
    header_payload = b"\0".join(
        name.encode("ascii") + b"\0" + (include_dir / name).read_bytes()
        for name in _EXPECTED_HEADERS
    )
    header_sha256 = _sha256_bytes(header_payload)
    key_payload = b"\0".join(
        (
            source,
            header_sha256.encode("ascii"),
            str(ort.__version__).encode("ascii"),
            cuda_arch.encode("ascii"),
            str(include_dir).encode("utf-8"),
            str(ort_library).encode("utf-8"),
        )
    )
    build_key = _sha256_bytes(key_payload)[:20]
    configured_cache = os.environ.get("CNN_S8S8_CUDA_CACHE")
    cache_root = (
        Path(configured_cache).expanduser()
        if configured_cache
        else Path.home() / ".cache" / "ea2_cnn_s8s8_cuda"
    )
    build_dir = cache_root.resolve() / build_key
    library = build_dir / "libcnn_s8s8_conv_cuda.so"
    manifest_path = build_dir / "build_manifest.json"
    build_dir.mkdir(parents=True, exist_ok=True)
    lock_path = build_dir / ".build.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        expected_manifest = {
            "schema_version": 1,
            "source_sha256": _sha256_bytes(source),
            "onnxruntime_headers_sha256": header_sha256,
            "onnxruntime_version": str(ort.__version__),
            "cuda_architecture": cuda_arch,
            "semantic_domain": "s8s8_sign_magnitude",
            "index_encoding": "signed_code_plus_128",
            "dynamic_lut_dtype": "int32",
        }
        if library.is_file() and manifest_path.is_file():
            try:
                current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                current_contract = {
                    key: current_manifest.get(key)
                    for key in expected_manifest
                }
                if (
                    current_contract == expected_manifest
                    and current_manifest.get("library_sha256") == _sha256_file(library)
                ):
                    return library
            except (OSError, json.JSONDecodeError):
                pass

        temporary = build_dir / f".{library.name}.tmp.{os.getpid()}"
        temporary.unlink(missing_ok=True)
        command = [
            nvcc,
            "-std=c++17",
            "-O3",
            "--shared",
            "-Xcompiler=-fPIC",
            f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}",
            f"-I{include_dir}",
            str(_NATIVE_SOURCE),
            f"-L{ort_library.parent}",
            f"-l:{ort_library.name}",
            "-lcudart",
            f"-Xlinker=-rpath,{ort_library.parent}",
            "-o",
            str(temporary),
        ]
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0 or not temporary.is_file():
            temporary.unlink(missing_ok=True)
            details = result.stdout[-8000:].strip()
            raise LabelGenerationError(
                "failed to build the signed S8S8 CUDA custom op "
                f"(nvcc exit {result.returncode}):\n{details}"
            )
        temporary.replace(library)
        manifest_path.write_text(
            json.dumps(
                {**expected_manifest, "library_sha256": _sha256_file(library)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return library


def _create_session(
    ort: Any,
    model_path: Path,
    *,
    gpu_id: int,
    custom_op_library: Path | None,
    profile_file_prefix: Path | None = None,
    cudnn_conv_algo_search: str = "DEFAULT",
) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_mem_pattern = True
    options.enable_mem_reuse = True
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    if profile_file_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_file_prefix)
    try:
        options.use_deterministic_compute = True
    except Exception:
        pass
    if custom_op_library is not None:
        try:
            _, ort_library = _ort_sdk(ort)
            _NATIVE_DEPENDENCY_HANDLES.append(
                ctypes.CDLL(str(ort_library), mode=ctypes.RTLD_GLOBAL)
            )
            options.register_custom_ops_library(str(custom_op_library))
        except Exception as exc:
            raise LabelGenerationError(
                f"unable to load the signed S8S8 CUDA custom op {custom_op_library}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    provider_options = {
        "device_id": str(gpu_id),
        "do_copy_in_default_stream": "1",
        "cudnn_conv_algo_search": str(cudnn_conv_algo_search),
        "use_tf32": "0",
    }
    try:
        session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"],
        )
    except Exception as exc:
        kind = "injected" if custom_op_library is not None else "exact"
        raise LabelGenerationError(
            f"unable to create {kind} CUDA-primary ONNX session on GPU {gpu_id}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not session.get_providers() or session.get_providers()[0] != "CUDAExecutionProvider":
        raise LabelGenerationError(
            f"ONNX model did not bind CUDAExecutionProvider first on GPU {gpu_id}; refusing CPU-only execution"
        )
    return session


class S8S8ConvCudaRuntime:
    """CUDA-primary executor for exact and dynamic signed-residual CNN graphs.

    ORT may place immutable initializer DequantizeLinear work on its CPU EP.
    The approximate Conv custom op itself has only a CUDA implementation.
    """

    def __init__(
        self,
        *,
        exact_session: Any,
        approx_session: Any,
        input_name: str,
        output_name: str,
        lut_input_name: str,
        batch_size: int,
        temporary_directory: tempfile.TemporaryDirectory[str],
        ort: Any,
        gpu_id: int,
        expected_custom_node_count: int,
        selected_conv_node_names: tuple[str, ...],
        custom_op_build: dict[str, object],
        input_residency: str = "auto",
        preallocate_outputs: bool = True,
        residual_abs_bound: int | None = None,
    ) -> None:
        self._exact_session = exact_session
        self._approx_session = approx_session
        self._input_name = input_name
        self._output_name = output_name
        self._lut_input_name = lut_input_name
        self._batch_size = batch_size
        self._temporary_directory = temporary_directory
        self._ort = ort
        self._gpu_id = gpu_id
        self._expected_custom_node_count = expected_custom_node_count
        self._selected_conv_node_names = tuple(str(name) for name in selected_conv_node_names)
        self._custom_op_build = dict(custom_op_build)
        self._input_residency = str(input_residency)
        self._preallocate_outputs = bool(preallocate_outputs)
        self._residual_abs_bound = residual_abs_bound
        self._approx_binding = PersistentCnnIoBinding(
            approx_session,
            ort_module=ort,
            input_name=input_name,
            lut_input_name=lut_input_name,
            output_name=output_name,
            device_id=gpu_id,
            input_residency=input_residency,
            preallocate_outputs=preallocate_outputs,
        )
        self._validated_inputs: np.ndarray | None = None
        self._placement_verified = False
        self._placement_event_count = 0
        self._exact_conv_placement_verified = False
        self._exact_conv_profile_event_count = 0
        self._closed = False

    @property
    def execution_metadata(self) -> dict[str, object]:
        exact_providers = list(self._exact_session.get_providers())
        approx_providers = list(self._approx_session.get_providers())
        return {
            "requested_device": "cuda",
            "logical_gpu_id": self._gpu_id,
            "provider_order": approx_providers,
            "exact_provider_order": exact_providers,
            "cuda_provider_required_first": True,
            "hybrid_partition": "CPUExecutionProvider" in approx_providers,
            "cpu_auxiliary_boundary": (
                "ORT may execute immutable initializer DequantizeLinear and transfer nodes on CPU; "
                "selected ApproxConvCudaS8Lut nodes have CUDAExecutionProvider kernels only"
            ),
            "approx_conv_provider": "CUDAExecutionProvider",
            "approx_conv_placement_verified": self._placement_verified,
            "approx_conv_profile_event_count": self._placement_event_count,
            "selected_conv_node_names": list(self._selected_conv_node_names),
            "exact_conv_placement_verified": self._exact_conv_placement_verified,
            "exact_conv_profile_event_count": self._exact_conv_profile_event_count,
            "cpu_only_fallback_allowed": False,
            "custom_op_build": dict(self._custom_op_build),
            "residual_abs_bound": getattr(self, "_residual_abs_bound", None),
            "persistent_iobinding": (
                self._approx_binding.metadata
                if getattr(self, "_approx_binding", None) is not None
                else None
            ),
        }

    @classmethod
    def from_plan(cls, plan: LabelGenerationPlan) -> S8S8ConvCudaRuntime:
        cuda_arch = _cuda_capability(plan.gpu_id)
        ort = _import_onnxruntime()
        custom_op_library = _build_custom_op(ort, cuda_arch=cuda_arch)
        try:
            custom_op_build = json.loads(
                custom_op_library.with_name("build_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise LabelGenerationError(f"unable to read signed S8S8 custom-op build manifest: {exc}") from exc
        custom_op_build["onnxruntime_version"] = str(ort.__version__)
        temporary_directory = tempfile.TemporaryDirectory(prefix="cnn-s8s8-runtime-")
        try:
            injected_path = Path(temporary_directory.name) / "injected.onnx"
            export = export_s8s8_lut_conv_model(
                plan.model_path,
                injected_path,
                node_names=plan.node_names,
                residual_abs_bound=getattr(plan, "residual_abs_bound", None),
            )
            exact_session = _create_session(
                ort,
                plan.model_path,
                gpu_id=plan.gpu_id,
                custom_op_library=None,
                profile_file_prefix=Path(temporary_directory.name) / "exact-placement",
                cudnn_conv_algo_search=getattr(plan, "cudnn_conv_algo_search", "DEFAULT"),
            )
            approx_session = _create_session(
                ort,
                export.output_path,
                gpu_id=plan.gpu_id,
                custom_op_library=custom_op_library,
                profile_file_prefix=Path(temporary_directory.name) / "approx-placement",
                cudnn_conv_algo_search=getattr(plan, "cudnn_conv_algo_search", "DEFAULT"),
            )
            output_name = cls._validate_sessions(
                exact_session,
                approx_session,
                input_name=plan.model_input_name,
                lut_input_name=export.lut_input_name,
            )
            return cls(
                exact_session=exact_session,
                approx_session=approx_session,
                input_name=plan.model_input_name,
                output_name=output_name,
                lut_input_name=export.lut_input_name,
                batch_size=plan.batch_size,
                temporary_directory=temporary_directory,
                ort=ort,
                gpu_id=plan.gpu_id,
                expected_custom_node_count=len(export.replaced_node_names),
                selected_conv_node_names=export.replaced_node_names,
                custom_op_build=custom_op_build,
                input_residency=getattr(plan, "input_residency", "auto"),
                preallocate_outputs=getattr(plan, "preallocate_outputs", True),
                residual_abs_bound=getattr(plan, "residual_abs_bound", None),
            )
        except Exception:
            temporary_directory.cleanup()
            raise

    @staticmethod
    def _validate_sessions(
        exact_session: Any,
        approx_session: Any,
        *,
        input_name: str,
        lut_input_name: str,
    ) -> str:
        exact_inputs = exact_session.get_inputs()
        approx_inputs = approx_session.get_inputs()
        if len(exact_inputs) != 1 or exact_inputs[0].name != input_name:
            names = [value.name for value in exact_inputs]
            raise LabelGenerationError(
                f"CNN real-label runtime requires one model input named {input_name!r}; got {names}"
            )
        if exact_inputs[0].type != "tensor(float)":
            raise LabelGenerationError(
                f"model input {input_name!r} must be float32; got {exact_inputs[0].type}"
            )
        approx_input = next((value for value in approx_inputs if value.name == input_name), None)
        if approx_input is None or approx_input.type != exact_inputs[0].type:
            raise LabelGenerationError("exact and injected CNN input types do not match")
        approx_names = [value.name for value in approx_inputs]
        if set(approx_names) != {input_name, lut_input_name} or len(approx_names) != 2:
            raise LabelGenerationError(
                "injected CNN model inputs do not equal the original input plus one residual LUT: "
                f"{approx_names}"
            )
        lut_meta = next(value for value in approx_inputs if value.name == lut_input_name)
        if lut_meta.type != "tensor(int32)" or tuple(lut_meta.shape) != (256, 256):
            raise LabelGenerationError(
                f"injected residual LUT input has invalid metadata: {lut_meta.type} {lut_meta.shape}"
            )
        exact_outputs = exact_session.get_outputs()
        approx_outputs = approx_session.get_outputs()
        if len(exact_outputs) != 1 or len(approx_outputs) != 1:
            raise LabelGenerationError("CNN real-label runtime currently requires one classification output")
        if (
            exact_outputs[0].name != approx_outputs[0].name
            or exact_outputs[0].type != "tensor(float)"
            or approx_outputs[0].type != "tensor(float)"
        ):
            raise LabelGenerationError("exact and injected CNN outputs do not share one float32 contract")
        return str(exact_outputs[0].name)

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        if self._closed:
            raise LabelGenerationError("signed S8S8 CUDA runtime is closed")
        if not isinstance(inputs, np.ndarray) or inputs.dtype != np.float32:
            raise LabelGenerationError("CNN runtime inputs must be a float32 numpy array")
        if inputs.ndim < 2 or inputs.shape[0] <= 0:
            raise LabelGenerationError("CNN runtime inputs must have a nonempty batch dimension")
        if self._validated_inputs is inputs:
            return inputs
        if not np.isfinite(inputs).all():
            raise LabelGenerationError("CNN runtime inputs must have a nonempty finite batch dimension")
        values = np.ascontiguousarray(inputs)
        self._validated_inputs = values
        return values

    def _run(self, session: Any, inputs: np.ndarray, lut_delta: np.ndarray | None) -> np.ndarray:
        if lut_delta is not None:
            self._approx_binding.prepare(inputs, batch_size=self._batch_size)
            self._approx_binding.update_lut(lut_delta)
            try:
                return self._approx_binding.run(
                    after_first_batch=(
                        None
                        if self._placement_verified
                        else lambda: self._verify_custom_op_placement(session)
                    )
                )
            except LabelGenerationError:
                raise
            except Exception as exc:
                raise LabelGenerationError(
                    f"approximate CUDA CNN inference failed: {type(exc).__name__}: {exc}"
                ) from exc

        batches: list[np.ndarray] = []
        for start in range(0, inputs.shape[0], self._batch_size):
            batch = inputs[start : start + self._batch_size]
            try:
                result = session.run(
                    [self._output_name],
                    {self._input_name: batch},
                )[0]
                self._verify_exact_conv_placement(session)
            except Exception as exc:
                raise LabelGenerationError(
                    f"exact CUDA CNN inference failed: {type(exc).__name__}: {exc}"
                ) from exc
            values = np.asarray(result)
            if values.dtype != np.float32 or values.shape[0] != batch.shape[0]:
                raise LabelGenerationError(
                    f"CUDA CNN runtime returned invalid output dtype/shape {values.dtype} {values.shape}"
                )
            batches.append(values)
        return np.concatenate(batches, axis=0)

    @staticmethod
    def _profile_event_node_names(event: dict[str, object]) -> set[str]:
        """Return the original node names represented by one ORT profile event.

        ORT releases have used both ``args.node_name`` and the event ``name``
        for this value.  Kernel events also append a profiling suffix, so
        normalize only the documented event suffixes instead of doing a loose
        prefix match that could confuse similarly named Conv nodes.
        """

        args = event.get("args")
        raw_values: list[object] = [event.get("name")]
        if isinstance(args, dict):
            raw_values.insert(0, args.get("node_name"))
            raw_values.insert(1, args.get("nodeName"))
        names: set[str] = set()
        suffixes = ("_kernel_time", "_fence_before", "_fence_after")
        for raw_value in raw_values:
            if not isinstance(raw_value, str) or not raw_value:
                continue
            candidate = raw_value.strip()
            if not candidate:
                continue
            names.add(candidate)
            if candidate.startswith("fused "):
                names.add(candidate[6:])
            for suffix in suffixes:
                if candidate.endswith(suffix):
                    names.add(candidate[: -len(suffix)])
        return names

    def _verify_exact_conv_placement(self, session: Any) -> None:
        """Verify every selected original Conv ran on CUDA in the exact graph."""

        if self._exact_conv_placement_verified:
            return
        try:
            profile_path = Path(session.end_profiling())
            events = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LabelGenerationError(
                "unable to verify exact Conv CUDA placement: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(events, list):
            raise LabelGenerationError("exact Conv placement profile is not a JSON event list")

        matches: dict[str, list[dict[str, object]]] = {
            name: [] for name in self._selected_conv_node_names
        }
        for event in events:
            if not isinstance(event, dict) or event.get("cat") not in {None, "Node"}:
                continue
            args = event.get("args")
            if not isinstance(args, dict) or not args.get("provider"):
                continue
            event_names = self._profile_event_node_names(event)
            for selected_name in matches:
                if selected_name in event_names:
                    matches[selected_name].append(event)

        missing = [name for name, selected_events in matches.items() if not selected_events]
        matching_events = [event for selected_events in matches.values() for event in selected_events]
        providers = {
            str(event["args"].get("provider", ""))
            for event in matching_events
            if isinstance(event.get("args"), dict)
        }
        if missing or providers != {"CUDAExecutionProvider"}:
            cpu_or_other = sorted(providers)
            raise LabelGenerationError(
                "exact Conv placement verification failed: "
                f"missing_nodes={missing}, events={len(matching_events)}, providers={cpu_or_other}"
            )
        self._exact_conv_profile_event_count = len(matching_events)
        self._exact_conv_placement_verified = True

    def _verify_custom_op_placement(self, session: Any) -> None:
        if self._placement_verified:
            return
        try:
            profile_path = Path(session.end_profiling())
            events = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LabelGenerationError(
                f"unable to verify ApproxConvCudaS8Lut CUDA placement: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(events, list):
            raise LabelGenerationError("ONNX Runtime placement profile is not a JSON event list")
        matching = [
            event
            for event in events
            if isinstance(event, dict)
            and isinstance(event.get("args"), dict)
            and event["args"].get("op_name") == "ApproxConvCudaS8Lut"
        ]
        providers = {
            str(event["args"].get("provider", ""))
            for event in matching
            if event["args"].get("provider")
        }
        if (
            len(matching) < self._expected_custom_node_count
            or providers != {"CUDAExecutionProvider"}
        ):
            raise LabelGenerationError(
                "ApproxConvCudaS8Lut placement verification failed: "
                f"events={len(matching)}, expected_at_least={self._expected_custom_node_count}, "
                f"providers={sorted(providers)}"
            )
        self._placement_event_count = len(matching)
        self._placement_verified = True

    def predict_exact(self, inputs: np.ndarray) -> np.ndarray:
        values = self._validate_inputs(inputs)
        return self._run(self._exact_session, values, None)

    def predict_approx(self, inputs: np.ndarray, lut_delta: np.ndarray) -> np.ndarray:
        values = self._validate_inputs(inputs)
        delta = validate_residual_lut(lut_delta, abs_bound=self._residual_abs_bound)
        return self._run(self._approx_session, values, delta)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._approx_binding is not None:
            self._approx_binding.close()
        self._approx_binding = None
        self._approx_session = None
        self._exact_session = None
        self._validated_inputs = None
        gc.collect()
        self._temporary_directory.cleanup()

    def __enter__(self) -> S8S8ConvCudaRuntime:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["S8S8ConvCudaRuntime", "validate_residual_lut"]
