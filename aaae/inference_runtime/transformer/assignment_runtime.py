from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .lut_cuda_runtime import CompiledLutCudaRuntime, _default_lut_input_name, compile_lut_cuda_runtime


def build_assignment_runtime(
    model_path: Path,
    *,
    node_to_lut_input_name: Mapping[str, str] | None = None,
    selected_node_names: Sequence[str] | None = None,
    shared_lut_input_name: str | None = None,
    device_id: int = 0,
    batch_fallback_parallelism: int = 1,
    lut_domain: str | None = None,
    s8_residual_abs_bound: int | None = None,
    session_profile: str = "strict",
    static_shape_reference_path: Path | None = None,
) -> CompiledLutCudaRuntime:
    if node_to_lut_input_name is None:
        if not selected_node_names:
            raise ValueError("build_assignment_runtime requires selected_node_names when node_to_lut_input_name is omitted")
        if shared_lut_input_name is not None:
            resolved_mapping = {str(node_name): str(shared_lut_input_name) for node_name in selected_node_names}
        else:
            resolved_mapping = {
                str(node_name): _default_lut_input_name(str(node_name))
                for node_name in selected_node_names
            }
    else:
        resolved_mapping = {str(node_name): str(input_name) for node_name, input_name in node_to_lut_input_name.items()}
    return compile_lut_cuda_runtime(
        Path(model_path),
        node_to_lut_input_name=resolved_mapping,
        device_id=int(device_id),
        batch_fallback_parallelism=int(batch_fallback_parallelism),
        lut_domain=lut_domain,
        s8_residual_abs_bound=s8_residual_abs_bound,
        session_profile=str(session_profile),
        static_shape_reference_path=static_shape_reference_path,
    )


@dataclass
class CompiledAssignmentRuntimePool:
    runtimes: list[CompiledLutCudaRuntime]
    shared_lut_input_name: str

    def __post_init__(self) -> None:
        if not self.runtimes:
            raise ValueError("CompiledAssignmentRuntimePool requires at least one runtime")
        self._executor = ThreadPoolExecutor(max_workers=len(self.runtimes), thread_name_prefix="d_assign_rt")

    def run_uniform(
        self,
        batch_input,
        *,
        approx_delta,
    ):
        return self.runtimes[0].run(
            batch_input,
            approx_deltas_by_input={self.shared_lut_input_name: approx_delta},
        )

    def run_uniform_many(
        self,
        batch_input,
        *,
        approx_deltas_by_key: Mapping[str, object],
    ) -> dict[str, object]:
        keys = list(approx_deltas_by_key.keys())
        if not keys:
            return {}
        if len(self.runtimes) == 1:
            runtime = self.runtimes[0]
            return {
                str(key): runtime.run(
                    batch_input,
                    approx_deltas_by_input={self.shared_lut_input_name: approx_deltas_by_key[key]},
                )[0]
                for key in keys
            }

        worker_count = min(len(self.runtimes), len(keys))
        groups: list[list[str]] = []
        for worker_idx in range(worker_count):
            group = [keys[idx] for idx in range(worker_idx, len(keys), worker_count)]
            if group:
                groups.append(group)

        def worker(runtime: CompiledLutCudaRuntime, key_group: list[str]) -> dict[str, object]:
            local: dict[str, object] = {}
            for key in key_group:
                local[str(key)] = runtime.run(
                    batch_input,
                    approx_deltas_by_input={self.shared_lut_input_name: approx_deltas_by_key[key]},
                )[0]
            return local

        futures = []
        for runtime, key_group in zip(self.runtimes, groups, strict=False):
            futures.append(self._executor.submit(worker, runtime, key_group))
        merged: dict[str, object] = {}
        for future in futures:
            merged.update(future.result())
        return merged

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        for runtime in self.runtimes:
            runtime.close()


@dataclass
class CompiledMappedAssignmentRuntimePool:
    runtimes: list[CompiledLutCudaRuntime]

    def __post_init__(self) -> None:
        if not self.runtimes:
            raise ValueError("CompiledMappedAssignmentRuntimePool requires at least one runtime")
        self._executor = ThreadPoolExecutor(max_workers=len(self.runtimes), thread_name_prefix="d_assign_map_rt")

    def run_many(
        self,
        batch_input,
        *,
        lut_feeds_by_key: Mapping[object, Mapping[str, np.ndarray]],
    ) -> dict[object, object]:
        keys = list(lut_feeds_by_key.keys())
        if not keys:
            return {}
        if len(self.runtimes) == 1:
            runtime = self.runtimes[0]
            return {
                key: runtime.run_prebuilt(
                    batch_input,
                    lut_feeds_by_input=lut_feeds_by_key[key],
                )[0]
                for key in keys
            }

        worker_count = min(len(self.runtimes), len(keys))
        groups: list[list[object]] = []
        for worker_idx in range(worker_count):
            group = [keys[idx] for idx in range(worker_idx, len(keys), worker_count)]
            if group:
                groups.append(group)

        def worker(runtime: CompiledLutCudaRuntime, key_group: list[object]) -> dict[object, object]:
            local: dict[object, object] = {}
            for key in key_group:
                local[key] = runtime.run_prebuilt(
                    batch_input,
                    lut_feeds_by_input=lut_feeds_by_key[key],
                )[0]
            return local

        futures = []
        for runtime, key_group in zip(self.runtimes, groups, strict=False):
            futures.append(self._executor.submit(worker, runtime, key_group))
        merged: dict[object, object] = {}
        for future in futures:
            merged.update(future.result())
        return merged

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        for runtime in self.runtimes:
            runtime.close()


def build_uniform_assignment_runtime_pool(
    model_path: Path,
    *,
    selected_node_names: Sequence[str],
    shared_lut_input_name: str,
    device_id: int = 0,
    runtime_count: int = 1,
    batch_fallback_parallelism: int = 1,
    lut_domain: str | None = None,
    s8_residual_abs_bound: int | None = None,
    session_profile: str = "strict",
    static_shape_reference_path: Path | None = None,
) -> CompiledAssignmentRuntimePool:
    runtimes = [
        build_assignment_runtime(
            model_path,
            selected_node_names=selected_node_names,
            shared_lut_input_name=shared_lut_input_name,
            device_id=int(device_id),
            batch_fallback_parallelism=int(batch_fallback_parallelism),
            lut_domain=lut_domain,
            s8_residual_abs_bound=s8_residual_abs_bound,
            session_profile=str(session_profile),
            static_shape_reference_path=static_shape_reference_path,
        )
        for _ in range(max(1, int(runtime_count)))
    ]
    return CompiledAssignmentRuntimePool(
        runtimes=runtimes,
        shared_lut_input_name=str(shared_lut_input_name),
    )


def build_assignment_runtime_pool(
    model_path: Path,
    *,
    node_to_lut_input_name: Mapping[str, str],
    device_id: int = 0,
    runtime_count: int = 1,
    batch_fallback_parallelism: int = 1,
    lut_domain: str | None = None,
    s8_residual_abs_bound: int | None = None,
    session_profile: str = "strict",
    static_shape_reference_path: Path | None = None,
) -> CompiledMappedAssignmentRuntimePool:
    runtimes: list[CompiledLutCudaRuntime] = []
    target = max(1, int(runtime_count))
    for _ in range(target):
        try:
            runtimes.append(
                build_assignment_runtime(
                    model_path,
                    node_to_lut_input_name=node_to_lut_input_name,
                    device_id=int(device_id),
                    batch_fallback_parallelism=int(batch_fallback_parallelism),
                    lut_domain=lut_domain,
                    s8_residual_abs_bound=s8_residual_abs_bound,
                    session_profile=str(session_profile),
                    static_shape_reference_path=static_shape_reference_path,
                )
            )
        except Exception:
            if not runtimes:
                raise
            break
    return CompiledMappedAssignmentRuntimePool(runtimes=runtimes)
