from __future__ import annotations

import argparse
import json
import sys
import traceback
from typing import Any

from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_KEY,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SESSIONS_PER_GPU,
)
from .strict50_bo_evaluator import Strict50BoEvaluator, build_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-lived strict50 BO evaluation worker.")
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--sessions-per-gpu", type=int, default=DEFAULT_SESSIONS_PER_GPU)
    parser.add_argument("--device", default="cuda:0")
    return parser


def write_message(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    evaluator: Strict50BoEvaluator | None = None
    try:
        evaluator = Strict50BoEvaluator(
            cache_csv=None,
            model_key=str(args.model_key),
            sample_count=int(args.sample_count),
            sample_seed=int(args.sample_seed),
            batch_size=int(args.batch_size),
            gpu_id=int(args.gpu_id),
            sessions_per_gpu=int(args.sessions_per_gpu),
            device=str(args.device),
            initialize_models=True,
        )
        write_message(
            {
                "event": "ready",
                "device": str(args.device),
                "gpu_id": int(args.gpu_id),
                "sample_count": int(args.sample_count),
                "sample_seed": int(args.sample_seed),
            }
        )
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                command = str(request.get("cmd", "eval"))
                task_id = str(request.get("task_id", ""))
                if command == "shutdown":
                    write_message({"event": "shutdown", "task_id": task_id})
                    break
                if command != "eval":
                    raise ValueError(f"unsupported worker command: {command}")

                rows: list[dict[str, Any]] = []
                tokens_list = request.get("tokens", [])
                log_every = int(request.get("log_every", 0))
                assert evaluator is not None
                for index, tokens in enumerate(tokens_list, start=1):
                    result = evaluator.evaluate(build_candidate(tuple(int(value) for value in tokens)))
                    row = dict(result)
                    row["cache_origin"] = "parallel_worker_eval"
                    row["worker_device"] = str(args.device)
                    rows.append(row)
                    if log_every > 0 and index % log_every == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "worker_progress",
                                    "task_id": task_id,
                                    "evaluated_in_task": index,
                                },
                                ensure_ascii=False,
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                write_message({"event": "result", "task_id": task_id, "evaluated": len(rows), "rows": rows})
            except Exception as exc:
                write_message(
                    {
                        "event": "error",
                        "task_id": str(request.get("task_id", "")),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        if evaluator is not None:
            evaluator.close()


if __name__ == "__main__":
    main()
