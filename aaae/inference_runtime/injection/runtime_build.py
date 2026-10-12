"""Core implementation extracted from ai_models/src/approx_injection/runtime_build.py."""

from __future__ import annotations


import fcntl

import hashlib

import json

import os

import re

import shutil

import site

import subprocess

import sys

from contextlib import contextmanager

from shutil import SameFileError

from pathlib import Path

from typing import Any, Iterable

from aaae.paths import EA2_ROOT, NATIVE_ROOT, ORT_INCLUDE_ROOT, OUTPUT_ROOT


CNN_CUDA_OPS_DIR = NATIVE_ROOT / "cnn"


TRANSFORMER_CUDA_OPS_DIR = NATIVE_ROOT / "transformer"


CUSTOM_OP_DIR = CNN_CUDA_OPS_DIR / "mux_to_approx"


DEFAULT_ORT_ROOT = OUTPUT_ROOT / "build" / "onnxruntime"


HEADER_CANDIDATES = [ORT_INCLUDE_ROOT]


DEFAULT_SM80_S8_TILE_MODE = "AUTO"


def _sha1_file(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _normalize_cuda_architectures(raw: str) -> str:
    values = [item.strip() for item in re.split(r"[;,]", raw) if item.strip()]
    if not values or any(re.fullmatch(r"[0-9]{2,3}", item) is None for item in values):
        raise ValueError(
            "EA2_CUDA_ARCHITECTURES must be a comma- or semicolon-separated list "
            f"of numeric CUDA architectures, got {raw!r}"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"EA2_CUDA_ARCHITECTURES contains duplicate architectures: {raw!r}")
    return ";".join(values)


def _detect_visible_cuda_architectures() -> str:
    """Detect architectures through CUDA visibility as seen by this process."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - the project runtime ships torch.
        raise RuntimeError(
            "automatic CUDA architecture detection requires PyTorch; set "
            "EA2_CUDA_ARCHITECTURES explicitly when building outside the pytorch environment"
        ) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        raise RuntimeError(
            "no CUDA device is visible for automatic architecture detection; set "
            "EA2_CUDA_ARCHITECTURES explicitly for a cross-compile"
        )
    values = sorted(
        {
            f"{int(major)}{int(minor)}"
            for major, minor in (
                torch.cuda.get_device_capability(index)
                for index in range(torch.cuda.device_count())
            )
        },
        key=int,
    )
    return _normalize_cuda_architectures(";".join(values))


def resolve_cuda_architectures() -> str:
    """Return the explicit override or architectures of CUDA-visible GPUs."""

    raw = os.environ.get("EA2_CUDA_ARCHITECTURES")
    if raw is None:
        return _detect_visible_cuda_architectures()
    return _normalize_cuda_architectures(raw)


def cuda_architecture_cache_tag(cuda_architectures: str | None = None) -> str:
    """Return a filesystem-safe tag for one normalized architecture set."""

    normalized = (
        resolve_cuda_architectures()
        if cuda_architectures is None
        else _normalize_cuda_architectures(cuda_architectures)
    )
    return "sm_" + "_".join(normalized.split(";"))


def _normalize_sm80_s8_tile_mode(value: str) -> str:
    mode = str(value).strip().upper()
    if mode not in {"AUTO", "ON", "OFF"}:
        raise ValueError("sm80_s8_tile_mode must be AUTO, ON, or OFF")
    return mode


def _symlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.is_symlink() or dst.exists():
            dst.unlink()
    except Exception:
        pass
    try:
        dst.symlink_to(src)
    except Exception:
        try:
            shutil.copy2(src, dst)
        except SameFileError:
            return


def _resolve_bundled_ort_include_dir() -> Path | None:
    for candidate in HEADER_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _prepare_onnxruntime_root(ort_root: Path) -> Path:
    ort_root = Path(ort_root).resolve()
    include_dir = ort_root / "include"
    lib_dir = ort_root / "lib"
    include_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)

    bundled_include = _resolve_bundled_ort_include_dir()
    if bundled_include is not None and not any(include_dir.iterdir()):
        for item in bundled_include.iterdir():
            dst = include_dir / item.name
            if dst.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
    if not (include_dir / "onnxruntime_cxx_api.h").exists():
        raise RuntimeError(
            "Unable to populate ONNX Runtime headers for building ort_approx. "
            f"Checked candidates: {[str(path) for path in HEADER_CANDIDATES]}"
        )

    capi_dir = None
    try:
        import onnxruntime as ort  # type: ignore
        import onnxruntime.capi as capi  # type: ignore

        _ = ort.__version__
        capi_dir = Path(capi.__file__).resolve().parent
    except Exception:
        capi_dir = None

    if capi_dir is None:
        search_roots = [Path(base) for base in site.getsitepackages()]
        for base in search_roots:
            candidate = base / "onnxruntime" / "capi"
            if candidate.is_dir():
                capi_dir = candidate
                break
    if capi_dir is None:
        raise RuntimeError("Unable to locate onnxruntime.capi for building ort_approx")

    main_lib = None
    for pattern in ("libonnxruntime.so*", "libonnxruntime.dylib*", "onnxruntime.dll"):
        libs = sorted(capi_dir.glob(pattern), reverse=True)
        if libs:
            main_lib = libs[0]
            break
    if main_lib is None:
        raise RuntimeError(f"Unable to locate ONNX Runtime shared library under {capi_dir}")

    _symlink_or_copy(main_lib, lib_dir / main_lib.name)
    if sys.platform.startswith("linux") and main_lib.name != "libonnxruntime.so":
        _symlink_or_copy(main_lib, lib_dir / "libonnxruntime.so")
        _symlink_or_copy(main_lib, lib_dir / "libonnxruntime.so.1")
    for provider_lib in capi_dir.glob("libonnxruntime_providers_*.so*"):
        _symlink_or_copy(provider_lib, lib_dir / provider_lib.name)
    shared = capi_dir / "libonnxruntime_providers_shared.so"
    if shared.exists():
        _symlink_or_copy(shared, lib_dir / shared.name)
    return ort_root


def _find_shared_library(directory: Path) -> Path:
    candidates = sorted(directory.rglob("libort_approx.so"))
    if candidates:
        return candidates[0]
    candidates = sorted(directory.rglob("ort_approx.dll"))
    if candidates:
        return candidates[0]
    candidates = sorted(directory.rglob("libort_approx.dylib"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Unable to find built custom-op library under {directory}")


def _resolve_real_cuda_custom_op_dir(op_types: Iterable[str] | None) -> Path:
    wanted = {str(op).strip() for op in (op_types or ["Conv"]) if str(op).strip()}
    if not wanted or wanted <= {"Conv"}:
        return CUSTOM_OP_DIR
    if wanted <= {"MatMul"}:
        return TRANSFORMER_CUDA_OPS_DIR
    if wanted <= {"Gemm"} or "Gemm" in wanted:
        raise RuntimeError("real_cuda currently supports Conv or MatMul replacement, but not Gemm")
    if "Conv" in wanted and "MatMul" in wanted:
        raise RuntimeError("real_cuda does not support building a mixed Conv+MatMul custom-op library yet")
    raise RuntimeError(f"Unsupported real_cuda op types: {sorted(wanted)}")


def _custom_op_source_path(custom_op_dir: Path) -> Path:
    candidates = sorted(custom_op_dir.glob("*.cu"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Unable to locate CUDA source under {custom_op_dir}")


def _run_command(command: list[str], *, workdir: Path) -> None:
    subprocess.run(command, cwd=str(workdir), check=True)


def _maybe_reuse_build(build_root: Path, *, manifest: dict[str, Any]) -> Path | None:
    manifest_path = build_root / "build_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if current != manifest:
        return None
    try:
        return _find_shared_library(build_root / "lib")
    except FileNotFoundError:
        return None


def _write_manifest(build_root: Path, manifest: dict[str, Any]) -> None:
    (build_root / "build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _reset_relocated_cmake_cache(cmake_build_dir: Path, *, custom_op_dir: Path) -> None:
    """Discard only a derived CMake cache created under a different checkout path."""

    cache_path = cmake_build_dir / "CMakeCache.txt"
    if not cache_path.exists():
        return
    cache = cache_path.read_text(encoding="utf-8", errors="replace")
    expected_source = f"CMAKE_HOME_DIRECTORY:INTERNAL={Path(custom_op_dir).resolve()}"
    expected_build = f"CMAKE_CACHEFILE_DIR:INTERNAL={Path(cmake_build_dir).resolve()}"
    if expected_source in cache and expected_build in cache:
        return
    shutil.rmtree(cmake_build_dir)


@contextmanager
def _build_lock(build_root: Path):
    build_root.mkdir(parents=True, exist_ok=True)
    lock_path = build_root / ".build.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _configure_and_build(
    *,
    build_root: Path,
    custom_op_dir: Path,
    onnxruntime_root: Path,
    generated_include_dir: Path | None,
    cuda_architectures: str,
    sm80_s8_tile_mode: str | None,
    manifest: dict[str, Any],
) -> Path:
    with _build_lock(build_root):
        build_root.mkdir(parents=True, exist_ok=True)
        lib_dir = build_root / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        reused = _maybe_reuse_build(build_root, manifest=manifest)
        if reused is not None:
            return reused

        cmake_build_dir = build_root / "cmake_build"
        _reset_relocated_cmake_cache(cmake_build_dir, custom_op_dir=custom_op_dir)
        cmake_build_dir.mkdir(parents=True, exist_ok=True)
        configure_cmd = [
            "cmake",
            "-S",
            str(custom_op_dir),
            "-B",
            str(cmake_build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={lib_dir}",
            f"-DONNXRUNTIME_ROOT={Path(onnxruntime_root).resolve()}",
        ]
        if generated_include_dir is not None:
            configure_cmd.append(f"-DGENERATED_INCLUDE_DIR={Path(generated_include_dir).resolve()}")
        if sm80_s8_tile_mode is not None:
            configure_cmd.append(f"-DEA2_SM80_S8_TILE_MODE={sm80_s8_tile_mode}")
        _run_command(configure_cmd, workdir=build_root)
        build_jobs = max(1, int(os.environ.get("EA2_CMAKE_BUILD_JOBS", "1")))
        _run_command(
            ["cmake", "--build", str(cmake_build_dir), "--target", "ort_approx", f"-j{build_jobs}"],
            workdir=build_root,
        )
        library_path = _find_shared_library(lib_dir)
        _write_manifest(build_root, manifest=manifest)
        return library_path


def build_proxy_op_library(
    *,
    build_root: Path,
    onnxruntime_root: Path | None = None,
    op_types: Iterable[str] | None = None,
    cuda_architectures: str | None = None,
    sm80_s8_tile_mode: str = DEFAULT_SM80_S8_TILE_MODE,
) -> Path:
    ort_root = _prepare_onnxruntime_root(DEFAULT_ORT_ROOT if onnxruntime_root is None else Path(onnxruntime_root))
    custom_op_dir = _resolve_real_cuda_custom_op_dir(op_types)
    custom_op_source = _custom_op_source_path(custom_op_dir)
    cuda_architectures = (
        resolve_cuda_architectures()
        if cuda_architectures is None
        else _normalize_cuda_architectures(cuda_architectures)
    )
    tile_mode = (
        _normalize_sm80_s8_tile_mode(sm80_s8_tile_mode)
        if custom_op_dir == TRANSFORMER_CUDA_OPS_DIR
        else None
    )
    manifest = {
        "mode": "proxy_op",
        "ort_root": str(ort_root),
        "custom_op_dir": str(custom_op_dir),
        "custom_op_sha": _sha1_file(custom_op_source),
        "cmake_sha": _sha1_file(custom_op_dir / "CMakeLists.txt"),
        "cuda_architectures": cuda_architectures,
        "sm80_s8_tile_mode": tile_mode,
    }
    return _configure_and_build(
        build_root=Path(build_root).resolve(),
        custom_op_dir=custom_op_dir,
        onnxruntime_root=ort_root,
        generated_include_dir=None,
        cuda_architectures=cuda_architectures,
        sm80_s8_tile_mode=tile_mode,
        manifest=manifest,
    )


def build_lut_cuda_library(
    *,
    build_root: Path,
    onnxruntime_root: Path | None = None,
    op_types: Iterable[str] | None = None,
    cuda_architectures: str | None = None,
    sm80_s8_tile_mode: str = DEFAULT_SM80_S8_TILE_MODE,
) -> Path:
    return build_proxy_op_library(
        build_root=build_root,
        onnxruntime_root=onnxruntime_root,
        op_types=(["MatMul"] if op_types is None else op_types),
        cuda_architectures=cuda_architectures,
        sm80_s8_tile_mode=sm80_s8_tile_mode,
    )


