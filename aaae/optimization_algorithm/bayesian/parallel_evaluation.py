from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from aaae.optimization_algorithm.common.device_sharding import split_weighted, validate_device_weights

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_KEY,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SESSIONS_PER_GPU,
    EA2_ROOT,
)
from .strict50_bo_evaluator import Strict50BoEvaluator, build_candidate


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


def split_evenly(items: list[tuple[int, ...]], parts: int) -> list[list[tuple[int, ...]]]:
    return split_weighted(items, (1,) * max(1, int(parts)))


@dataclass
class WorkerProcess:
    device: str
    process: subprocess.Popen[str]


class ParallelStrict50EvaluationPool:
    def __init__(
        self,
        *,
        devices: Iterable[str],
        device_weights: Iterable[int] | None = None,
        python_executable: str | None = None,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        model_key: str = DEFAULT_MODEL_KEY,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        sample_seed: int = DEFAULT_SAMPLE_SEED,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sessions_per_gpu: int = DEFAULT_SESSIONS_PER_GPU,
    ) -> None:
        self.devices = [str(device) for device in devices]
        if not self.devices:
            raise ValueError("at least one device is required")
        self.device_weights = validate_device_weights(
            tuple(device_weights) if device_weights is not None else None, len(self.devices)
        )
        self.python_executable = python_executable or sys.executable
        self.cwd = Path(cwd).resolve() if cwd is not None else EA2_ROOT
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        self.model_key = str(model_key)
        self.sample_count = int(sample_count)
        self.sample_seed = int(sample_seed)
        self.batch_size = int(batch_size)
        self.sessions_per_gpu = int(sessions_per_gpu)
        self.workers: list[WorkerProcess] = []
        self._closed = False
        self._start_workers()

    def _worker_env(self, device: str) -> dict[str, str]:
        env = dict(self.env)
        if str(device).startswith("cuda"):
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id_from_device(device))
        return env

    def _start_workers(self) -> None:
        for device in self.devices:
            worker_device = "cuda:0" if str(device).startswith("cuda") else str(device)
            gpu_id = 0 if str(device).startswith("cuda") else gpu_id_from_device(device)
            process = subprocess.Popen(
                [
                    self.python_executable,
                    "-m",
                    "aaae.optimization_algorithm.bayesian.parallel_worker",
                    "--model-key",
                    self.model_key,
                    "--sample-count",
                    str(self.sample_count),
                    "--sample-seed",
                    str(self.sample_seed),
                    "--batch-size",
                    str(self.batch_size),
                    "--gpu-id",
                    str(gpu_id),
                    "--sessions-per-gpu",
                    str(self.sessions_per_gpu),
                    "--device",
                    worker_device,
                ],
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

    def evaluate(self, tokens: list[tuple[int, ...]], *, log_every: int = 0) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("parallel evaluation pool is closed")
        if not tokens:
            return []
        expected_hashes = [build_candidate(item).config_hash() for item in tokens]
        if len(set(expected_hashes)) != len(expected_hashes):
            raise ValueError("parallel strict50 evaluation requires unique candidate hashes")
        shards = split_weighted(tokens, self.device_weights)
        active: list[tuple[WorkerProcess, str]] = []
        for index, (worker, shard) in enumerate(zip(self.workers, shards, strict=True)):
            if not shard:
                continue
            task_id = f"task_{index:03d}"
            payload = {
                "cmd": "eval",
                "task_id": task_id,
                "tokens": [[int(value) for value in item] for item in shard],
                "log_every": int(log_every),
            }
            assert worker.process.stdin is not None
            worker.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            worker.process.stdin.flush()
            active.append((worker, task_id))

        rows: list[dict[str, Any]] = []
        for worker, task_id in active:
            assert worker.process.stdout is not None
            while True:
                line = worker.process.stdout.readline()
                if not line:
                    raise RuntimeError(f"worker {worker.device} exited while waiting for {task_id}")
                payload = json.loads(line)
                event = str(payload.get("event", ""))
                if event == "result" and str(payload.get("task_id", "")) == task_id:
                    for row in payload.get("rows", []):
                        row["worker_device"] = str(worker.device)
                        rows.append(row)
                    break
                if event == "error":
                    raise RuntimeError(
                        f"worker {worker.device} failed task {payload.get('task_id')}: "
                        f"{payload.get('error')}\n{payload.get('traceback', '')}"
                    )

        by_hash = {str(row.get("config_hash", "")): row for row in rows}
        if len(by_hash) != len(rows):
            raise RuntimeError("parallel strict50 workers returned duplicate config hashes")
        missing = [key for key in expected_hashes if key not in by_hash]
        unexpected = sorted(set(by_hash).difference(expected_hashes))
        if missing or unexpected:
            raise RuntimeError(
                "parallel strict50 worker result set differs from requested candidates: "
                f"missing={missing[:10]} unexpected={unexpected[:10]}"
            )
        return [by_hash[key] for key in expected_hashes]

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

    def __enter__(self) -> "ParallelStrict50EvaluationPool":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def merge_rows_into_evaluator(evaluator: Strict50BoEvaluator, rows: Iterable[dict[str, Any]]) -> int:
    rows_list = list(rows)
    return evaluator.add_result_rows(rows_list, save=True)
