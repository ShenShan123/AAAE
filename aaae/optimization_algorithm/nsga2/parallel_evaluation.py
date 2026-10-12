from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from aaae.optimization_algorithm.common.device_sharding import split_weighted, validate_device_weights

from .config import FIXED_HIGH_TOKENS, LOW_TOKEN_COUNT
from .evaluators import (
    DEFAULT_TRUE1000_BATCH_SIZE,
    DEFAULT_TRUE1000_MODEL_KEY,
    DEFAULT_TRUE1000_SAMPLE_COUNT,
    DEFAULT_TRUE1000_SAMPLE_SEED,
    DEFAULT_TRUE1000_SESSION_PROFILE,
    DEFAULT_TRUE1000_SESSIONS_PER_GPU,
    CandidateConfig,
    EvaluationResult,
    OptimizationEvaluator,
)


def parse_devices(device: str | None, devices: str | None) -> list[str]:
    raw = str(devices or device or "cuda:0")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        values = ["cuda:0"]
    normalized: list[str] = []
    for value in values:
        if value.isdigit():
            normalized.append(f"cuda:{value}")
        else:
            normalized.append(value)
    return normalized


def gpu_id_from_device(device: str) -> int:
    value = str(device)
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    if value == "cuda":
        return 0
    return 0


def result_from_row(row: dict[str, Any]) -> EvaluationResult:
    return OptimizationEvaluator.result_from_row(row)


def split_evenly(items: list[tuple[int, ...]], parts: int) -> list[list[tuple[int, ...]]]:
    return split_weighted(items, (1,) * max(1, int(parts)))


def config_payload(config: CandidateConfig) -> dict[str, object]:
    return {
        "low_tokens": [int(value) for value in config.low_tokens],
        "high_tokens": [int(value) for value in config.high_tokens],
        "approx_high_part": bool(config.approx_high_part),
        "config_hash": config.config_hash(),
    }


@dataclass
class WorkerProcess:
    device: str
    process: subprocess.Popen[str]


class ParallelEvaluationPool:
    def __init__(
        self,
        *,
        devices: Iterable[str],
        device_weights: Iterable[int] | None = None,
        python_executable: str | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        top1_eval_method: str = "strict50_v3",
        true1000_model_key: str = DEFAULT_TRUE1000_MODEL_KEY,
        true1000_sample_count: int = DEFAULT_TRUE1000_SAMPLE_COUNT,
        true1000_sample_seed: int = DEFAULT_TRUE1000_SAMPLE_SEED,
        true1000_batch_size: int = DEFAULT_TRUE1000_BATCH_SIZE,
        true1000_sessions_per_gpu: int = DEFAULT_TRUE1000_SESSIONS_PER_GPU,
        true1000_session_profile: str = DEFAULT_TRUE1000_SESSION_PROFILE,
        true1000_baseline_cache_dir: Path | None = None,
        true1000_archive_baseline_dir: Path | None = None,
        true1000_batch_fallback_parallelism: int = 1,
    ) -> None:
        self.devices = [str(device) for device in devices]
        if not self.devices:
            raise ValueError("at least one device is required")
        self.device_weights = validate_device_weights(
            tuple(device_weights) if device_weights is not None else None, len(self.devices)
        )
        self.python_executable = python_executable or sys.executable
        self.cwd = Path(cwd).resolve() if cwd is not None else Path.cwd()
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        self.top1_eval_method = str(top1_eval_method)
        self.true1000_model_key = str(true1000_model_key)
        self.true1000_sample_count = int(true1000_sample_count)
        self.true1000_sample_seed = int(true1000_sample_seed)
        self.true1000_batch_size = int(true1000_batch_size)
        self.true1000_sessions_per_gpu = int(true1000_sessions_per_gpu)
        self.true1000_session_profile = str(true1000_session_profile).strip().lower()
        if self.true1000_session_profile not in {"strict", "throughput"}:
            raise ValueError(
                "true1000_session_profile must be one of {'strict', 'throughput'}, "
                f"got {true1000_session_profile!r}"
            )
        self.true1000_baseline_cache_dir = Path(true1000_baseline_cache_dir) if true1000_baseline_cache_dir is not None else None
        self.true1000_archive_baseline_dir = Path(true1000_archive_baseline_dir) if true1000_archive_baseline_dir is not None else None
        self.true1000_batch_fallback_parallelism = int(true1000_batch_fallback_parallelism)
        self.workers: list[WorkerProcess] = []
        self._closed = False
        self._start_workers()

    def _worker_env(self, device: str) -> dict[str, str]:
        env = dict(self.env)
        if str(device).startswith("cuda"):
            gpu_id = gpu_id_from_device(device)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return env

    def _start_workers(self) -> None:
        for device in self.devices:
            worker_device = "cuda:0" if str(device).startswith("cuda") else str(device)
            gpu_id = 0 if str(device).startswith("cuda") else gpu_id_from_device(device)
            command = [
                self.python_executable,
                "-m",
                "aaae.optimization_algorithm.nsga2.parallel_worker",
                "--device",
                worker_device,
                "--strict50-gpu-id",
                str(gpu_id),
                "--top1-eval-method",
                self.top1_eval_method,
                "--true1000-model-key",
                self.true1000_model_key,
                "--true1000-sample-count",
                str(self.true1000_sample_count),
                "--true1000-sample-seed",
                str(self.true1000_sample_seed),
                "--true1000-batch-size",
                str(self.true1000_batch_size),
                "--true1000-sessions-per-gpu",
                str(self.true1000_sessions_per_gpu),
                "--true1000-session-profile",
                self.true1000_session_profile,
                "--true1000-batch-fallback-parallelism",
                str(self.true1000_batch_fallback_parallelism),
            ]
            if self.true1000_baseline_cache_dir is not None:
                command.extend(["--true1000-baseline-cache-dir", str(self.true1000_baseline_cache_dir)])
            if self.true1000_archive_baseline_dir is not None:
                command.extend(["--true1000-archive-baseline-dir", str(self.true1000_archive_baseline_dir)])
            process = subprocess.Popen(
                command,
                cwd=str(self.cwd),
                env=self._worker_env(device),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            if process.stdin is None or process.stdout is None:
                raise RuntimeError(f"failed to start worker for {device}")
            ready = process.stdout.readline()
            if not ready:
                process.kill()
                raise RuntimeError(f"worker for {device} exited before ready")
            payload = json.loads(ready)
            if payload.get("event") != "ready":
                process.kill()
                raise RuntimeError(f"worker for {device} did not report ready: {payload}")
            self.workers.append(WorkerProcess(device=device, process=process))

    def evaluate(
        self,
        tokens: list[tuple[int, ...]],
        *,
        log_every: int = 0,
    ) -> list[EvaluationResult]:
        configs = [
            CandidateConfig(
                low_tokens=item,
                high_tokens=FIXED_HIGH_TOKENS,
                approx_high_part=False,
            )
            for item in tokens
        ]
        return self.evaluate_configs(configs, log_every=log_every)

    def evaluate_configs(
        self,
        configs: list[CandidateConfig],
        *,
        log_every: int = 0,
    ) -> list[EvaluationResult]:
        if self._closed:
            raise RuntimeError("parallel evaluation pool is closed")
        if not configs:
            return []
        shards = split_weighted(configs, self.device_weights)
        active: list[tuple[WorkerProcess, str]] = []
        for index, (worker, shard) in enumerate(zip(self.workers, shards, strict=True)):
            if not shard:
                continue
            task_id = f"task_{index:03d}"
            payload = {
                "cmd": "eval",
                "task_id": task_id,
                "configs": [config_payload(config) for config in shard],
                "log_every": int(log_every),
            }
            assert worker.process.stdin is not None
            worker.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            worker.process.stdin.flush()
            active.append((worker, task_id))

        results: list[EvaluationResult] = []
        for worker, task_id in active:
            assert worker.process.stdout is not None
            while True:
                line = worker.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"worker {worker.device} exited while waiting for {task_id}")
                payload = json.loads(line)
                event = str(payload.get("event", ""))
                if event == "result" and str(payload.get("task_id", "")) == task_id:
                    rows = payload.get("rows", [])
                    results.extend(result_from_row(row) for row in rows)
                    break
                if event == "error":
                    raise RuntimeError(
                        f"worker {worker.device} failed task {payload.get('task_id')}: "
                        f"{payload.get('error')}\n{payload.get('traceback', '')}"
                    )
        by_hash = {result.config_hash: result for result in results}
        expected_hashes = [config.config_hash() for config in configs]
        if len(by_hash) != len(results):
            raise RuntimeError("parallel workers returned duplicate config hashes")
        missing = [config_hash for config_hash in expected_hashes if config_hash not in by_hash]
        unexpected = sorted(set(by_hash).difference(expected_hashes))
        if missing or unexpected:
            raise RuntimeError(
                "parallel worker result set differs from requested configs: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return [by_hash[config_hash] for config_hash in expected_hashes]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self.workers:
            process = worker.process
            try:
                if process.stdin is not None and process.poll() is None:
                    process.stdin.write(json.dumps({"cmd": "shutdown", "task_id": "shutdown"}) + "\n")
                    process.stdin.flush()
            except Exception:
                pass
        for worker in self.workers:
            process = worker.process
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    def __enter__(self) -> "ParallelEvaluationPool":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def merge_results_into_evaluator(evaluator: OptimizationEvaluator, results: Iterable[EvaluationResult]) -> int:
    count = 0
    for result in results:
        evaluator.add_result(result, save=False)
        count += 1
    if count:
        evaluator.save_cache_csv()
    return count
