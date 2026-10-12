from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .config import FIXED_HIGH_TOKENS
from .evaluators import (
    DEFAULT_TRUE1000_BATCH_SIZE,
    DEFAULT_TRUE1000_MODEL_KEY,
    DEFAULT_TRUE1000_SAMPLE_COUNT,
    DEFAULT_TRUE1000_SAMPLE_SEED,
    DEFAULT_TRUE1000_SESSION_PROFILE,
    DEFAULT_TRUE1000_SESSIONS_PER_GPU,
    TRUE1000_TOP1_METHOD,
    CandidateConfig,
    OptimizationEvaluator,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Long-lived AMCFG top1_drop evaluation worker.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict50-gpu-id", type=int, default=0)
    parser.add_argument("--top1-eval-method", default="strict50_v3")
    parser.add_argument("--true1000-model-key", default=DEFAULT_TRUE1000_MODEL_KEY)
    parser.add_argument("--true1000-sample-count", type=int, default=DEFAULT_TRUE1000_SAMPLE_COUNT)
    parser.add_argument("--true1000-sample-seed", type=int, default=DEFAULT_TRUE1000_SAMPLE_SEED)
    parser.add_argument("--true1000-batch-size", type=int, default=DEFAULT_TRUE1000_BATCH_SIZE)
    parser.add_argument("--true1000-sessions-per-gpu", type=int, default=DEFAULT_TRUE1000_SESSIONS_PER_GPU)
    parser.add_argument(
        "--true1000-session-profile",
        choices=("strict", "throughput"),
        default=DEFAULT_TRUE1000_SESSION_PROFILE,
    )
    parser.add_argument("--true1000-baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--true1000-archive-baseline-dir", type=Path, default=None)
    parser.add_argument("--true1000-batch-fallback-parallelism", type=int, default=1)
    return parser


def write_message(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def config_from_payload(payload: dict[str, Any]) -> CandidateConfig:
    config = CandidateConfig(
        low_tokens=tuple(int(value) for value in payload["low_tokens"]),
        high_tokens=tuple(int(value) for value in payload.get("high_tokens", FIXED_HIGH_TOKENS)),
        approx_high_part=bool(payload.get("approx_high_part", False)),
    )
    expected_hash = str(payload.get("config_hash", "")).strip()
    if expected_hash and config.config_hash() != expected_hash:
        raise ValueError(
            f"worker config payload hash {expected_hash} does not match {config.config_hash()}"
        )
    return config


def main() -> None:
    args = build_parser().parse_args()
    evaluator: OptimizationEvaluator | None = None
    try:
        evaluator = OptimizationEvaluator(
            top1_eval_method=str(args.top1_eval_method),
            cache_csv=None,
            device=args.device,
            strict50_gpu_id=int(args.strict50_gpu_id),
            true1000_model_key=str(args.true1000_model_key),
            true1000_sample_count=int(args.true1000_sample_count),
            true1000_sample_seed=int(args.true1000_sample_seed),
            true1000_batch_size=int(args.true1000_batch_size),
            true1000_sessions_per_gpu=int(args.true1000_sessions_per_gpu),
            true1000_session_profile=str(args.true1000_session_profile),
            true1000_baseline_cache_dir=args.true1000_baseline_cache_dir,
            true1000_archive_baseline_dir=args.true1000_archive_baseline_dir,
            true1000_batch_fallback_parallelism=int(args.true1000_batch_fallback_parallelism),
        )
        write_message(
            {
                "event": "ready",
                "device": str(args.device),
                "strict50_gpu_id": int(args.strict50_gpu_id),
                "top1_eval_method": str(args.top1_eval_method),
                "true1000_sample_count": int(args.true1000_sample_count)
                if str(args.top1_eval_method) == TRUE1000_TOP1_METHOD
                else None,
                "true1000_session_profile": str(args.true1000_session_profile)
                if str(args.top1_eval_method) == TRUE1000_TOP1_METHOD
                else None,
            }
        )
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                command = str(request.get("cmd", "eval"))
                task_id = str(request.get("task_id", ""))
                if command == "shutdown":
                    write_message({"event": "shutdown", "task_id": task_id})
                    break
                if command != "eval":
                    raise ValueError(f"unsupported worker command: {command}")

                if "configs" in request:
                    configs = [config_from_payload(payload) for payload in request.get("configs", [])]
                else:
                    configs = [
                        CandidateConfig(
                            low_tokens=tuple(int(value) for value in tokens),
                            high_tokens=FIXED_HIGH_TOKENS,
                            approx_high_part=False,
                        )
                        for tokens in request.get("tokens", [])
                    ]
                results = evaluator.evaluate_configs(configs)
                rows = [result.as_row() for result in results]
                write_message({"event": "result", "task_id": task_id, "evaluated": len(rows), "rows": rows})
            except Exception as exc:
                write_message(
                    {
                        "event": "error",
                        "task_id": str(locals().get("request", {}).get("task_id", "")),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        if evaluator is not None:
            evaluator.close()


if __name__ == "__main__":
    main()
