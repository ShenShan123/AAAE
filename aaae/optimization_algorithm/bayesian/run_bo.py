from __future__ import annotations

from aaae.paths import OUTPUT_ROOT

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aaae.optimization_algorithm.common.device_sharding import parse_device_weights

from aaae.optimization_algorithm.common.bayesian.acquisition import choose_next_candidates
from aaae.optimization_algorithm.common.bayesian.candidate_pool import generate_candidate_pool, token_columns_from_frame
from aaae.optimization_algorithm.common.bayesian.candidate_identity import config_hash_for_low_tokens
from aaae.optimization_algorithm.common.bayesian.io_utils import append_or_write_csv, atomic_write_csv, atomic_write_json, json_safe, now_iso
from aaae.optimization_algorithm.common.bayesian.pareto import (
    assign_design_ids,
    build_pareto_frame,
    build_top1drop_bin_pdp_curve,
    selected_top1drop_bin_front,
    spaced_pareto_hashes,
    top1drop_bin_lowers,
)
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_KEY,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_NAME,
    DEFAULT_SAMPLE_COUNT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SEED_CACHE,
    DEFAULT_SESSIONS_PER_GPU,
    DEFAULT_STRICT50_HISTORY_RUN_DIRS,
    DEFAULT_STRICT50_MODEL_PACKAGE_ID,
    DEFAULT_STRICT50_SELECTED_IMAGE_COUNT,
    EA2_ROOT,
    STRICT50_BANK_META,
    STRICT50_DATASET_ID,
    STRICT50_DATASET_MANIFEST,
    STRICT50_DATASET_SCHEMA_VERSION,
    STRICT50_EXPECTED_BANK_META_SHA1,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_FEATURE_DIM,
    STRICT50_FEATURE_SCHEMA_HASH,
    STRICT50_INJECTION_SCOPE,
    STRICT50_NODE_REGISTRY,
    STRICT50_PACKAGE_HASH,
    STRICT50_RUNTIME_BASE_FEATURE_DIM,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_SELECTED_IMAGES_HASH,
    STRICT50_SUPPLEMENTAL_FEATURE_DIM,
    STRICT50_TOP1_METHOD,
    STRICT50_TOP1_PACKAGE_DIR,
    TOKEN_COLUMNS,
)
from .parallel_evaluation import ParallelStrict50EvaluationPool, gpu_id_from_device, merge_rows_into_evaluator, parse_devices
from .strict50_bo_evaluator import Strict50BoEvaluator, build_candidate, sha1_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Three-objective Bayesian optimization of PDP, strict50 top1_drop, and GNN multiplier area."
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-key", default=DEFAULT_MODEL_KEY)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--sessions-per-gpu", type=int, default=DEFAULT_SESSIONS_PER_GPU)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for parallel strict50 evaluation, e.g. cuda:0,cuda:1,cuda:2. Defaults to --device.",
    )
    parser.add_argument(
        "--eval-device-weights",
        default=None,
        help="Positive integer evaluation shares in --devices order, e.g. 2,1 for a faster first GPU.",
    )
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--initial-true-evals", type=int, default=100)
    parser.add_argument("--bo-iterations", type=int, default=400)
    parser.add_argument(
        "--output-every", type=int, default=100,
        help="Refresh derived CSVs, Pareto fronts, and plots every N batches; the live evaluation cache is saved each batch.",
    )
    parser.add_argument(
        "--parallel-eval-batch", type=int, default=None,
        help="BO candidates per iteration; defaults to the number of devices when --devices has multiple entries, otherwise 1.",
    )
    parser.add_argument(
        "--max-new-evals",
        type=int,
        default=None,
        help="Hard cap on new unique evaluations after startup; use 30000 for the fair 30k protocol.",
    )
    parser.add_argument("--batch-log-every", type=int, default=0)
    parser.add_argument("--candidate-pool-size", type=int, default=100_000)
    parser.add_argument("--max-pool-acq-eval", type=int, default=4096)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument(
        "--gp-fit-maxiter",
        type=int,
        default=100,
        help="Maximum optimizer iterations for each exact GP fit.",
    )
    parser.add_argument(
        "--acquisition-device",
        default="cpu",
        help="Torch device for exact-GP fitting and acquisition scoring, e.g. cuda:4.",
    )
    parser.add_argument(
        "--baseline-prune-samples",
        type=int,
        default=32,
        help="Samples for baseline pruning (default 32 for faster three-objective BO); 0 disables pruning.",
    )
    parser.add_argument("--seed-cache", type=Path, default=DEFAULT_SEED_CACHE)
    parser.add_argument("--seed-cache-mode", choices=("none", "warm_start", "import_as_observed"), default="none")
    parser.add_argument("--seed-cache-limit", type=int, default=None)
    parser.add_argument("--max-observed-for-acq", type=int, default=512)
    parser.add_argument("--max-observed-for-pool", type=int, default=2000)
    parser.add_argument("--no-model-init", action="store_true", help="Do not initialize strict50 runtime; only cache import/output steps can run.")
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include built-in structured candidate seeds. Disabled by default for from-zero runs.",
    )
    parser.add_argument(
        "--allow-existing-cache",
        action="store_true",
        help="Explicitly reuse this run directory's evaluation_cache_live.csv instead of requiring a fresh run name.",
    )
    parser.add_argument("--plot-all", action="store_true")
    parser.add_argument("--stop-after-initial", action="store_true")
    parser.add_argument("--disable-drop-bin-targeting", action="store_true")
    parser.add_argument("--drop-bin-min", type=float, default=0.0)
    parser.add_argument("--drop-bin-max", type=float, default=0.50)
    parser.add_argument("--drop-bin-step", type=float, default=0.01)
    parser.add_argument(
        "--disable-valid-drop-filter",
        action="store_true",
        help="Disable acquisition-time filtering to the valid top1_drop range.",
    )
    parser.add_argument(
        "--valid-drop-margin",
        type=float,
        default=0.02,
        help="Prediction margin around [drop-bin-min, drop-bin-max] used for acquisition filtering.",
    )
    parser.add_argument(
        "--disable-valid-biased-context",
        action="store_true",
        help="Disable valid/near-valid biased observed context and candidate-pool seeding.",
    )
    parser.add_argument(
        "--context-valid-drop-max",
        type=float,
        default=0.55,
        help="Maximum observed top1_drop used as near-valid BO context before falling back to global rows.",
    )
    parser.add_argument(
        "--context-global-fraction",
        type=float,
        default=0.08,
        help="Fraction of observed context reserved for global exploratory rows when valid-biased context is enabled.",
    )
    parser.add_argument(
        "--cache-path-pool-seeds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Seed each candidate pool from raw cache CSV files in addition to selected observed context.",
    )
    return parser


def prepare_output_dir(output_dir: Path, *, allow_existing_cache: bool) -> None:
    protected_names = (
        "evaluation_cache_live.csv",
        "evaluation_cache.csv",
        "bo_trials.csv",
        "run_manifest.json",
        "summary.json",
        "pareto_front.csv",
    )
    existing = [name for name in protected_names if (output_dir / name).exists()]
    if existing and not allow_existing_cache:
        raise FileExistsError(
            f"output directory already contains a BO run: {output_dir} ({', '.join(existing)}). "
            "Choose a fresh --run-name or explicitly resume an area3 run."
        )
    if existing:
        live_cache = output_dir / "evaluation_cache_live.csv"
        if not live_cache.exists() or "area_um2" not in pd.read_csv(live_cache, nrows=0).columns:
            raise ValueError(f"cannot reuse a BO run without an area_um2 live cache: {output_dir}")
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if "area_um2" not in manifest.get("objectives", []):
                raise ValueError(f"cannot reuse a two-objective BO run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def resolve_parallel_eval_batch(args: argparse.Namespace, devices: list[str]) -> int:
    if args.parallel_eval_batch is not None:
        if int(args.parallel_eval_batch) <= 0:
            raise ValueError("--parallel-eval-batch must be positive")
        return int(args.parallel_eval_batch)
    return len(devices) if bool(args.devices) and len(devices) > 1 else 1


def strict50_cache_paths(seed_cache: Path | None = None) -> list[Path]:
    paths: list[Path] = []
    if seed_cache is not None:
        paths.append(Path(seed_cache))
    for run_dir in DEFAULT_STRICT50_HISTORY_RUN_DIRS:
        paths.extend(
            [
                run_dir / "pareto_front.csv",
                run_dir / "curve_points.csv",
                run_dir / "evaluation_cache.csv",
            ]
        )
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(Path(path).resolve())
        if key in seen or not Path(path).exists():
            continue
        seen.add(key)
        out.append(Path(path))
    return out


def warm_start_cache_paths(args: argparse.Namespace) -> list[Path]:
    if str(args.seed_cache_mode) == "none":
        return []
    return strict50_cache_paths(args.seed_cache)


def config_hash_for_tokens(tokens: tuple[int, ...]) -> str:
    return config_hash_for_low_tokens(tokens)


def filter_unevaluated(pool: list[tuple[int, ...]], evaluated_hashes: set[str]) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    seen: set[str] = set()
    for tokens in pool:
        key = config_hash_for_tokens(tokens)
        if key in evaluated_hashes or key in seen:
            continue
        seen.add(key)
        out.append(tokens)
    return out


def summarize(
    frame: pd.DataFrame,
    pareto: pd.DataFrame,
    valid_frame: pd.DataFrame,
    valid_pareto: pd.DataFrame,
    bin_curve: pd.DataFrame,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    seed_import_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    devices = parse_devices(args.device, args.devices)
    device_weights = parse_device_weights(args.eval_device_weights, len(devices))
    selected_bins = bin_curve[bin_curve["selected"].astype(bool)].copy() if not bin_curve.empty and "selected" in bin_curve else pd.DataFrame()
    missing_bins = bin_curve[~bin_curve["selected"].astype(bool)].copy() if not bin_curve.empty and "selected" in bin_curve else pd.DataFrame()
    imported_count = 0
    new_count = 0
    if not frame.empty and "cache_origin" in frame.columns:
        origins = frame["cache_origin"].astype(str)
        imported_count = int(origins.str.startswith("seed_cache:").sum())
        new_count = int(origins.isin(["live_new_eval", "parallel_worker_eval"]).sum())
    return {
        "created_at": now_iso(),
        "run_name": str(args.run_name),
        "output_dir": str(output_dir),
        "model_key": str(args.model_key),
        "sample_count": int(args.sample_count),
        "sample_seed": int(args.sample_seed),
        "batch_size": int(args.batch_size),
        "gpu_id": gpu_id_from_device(devices[0]) if args.gpu_id is None else int(args.gpu_id),
        "sessions_per_gpu": int(args.sessions_per_gpu),
        "device": devices[0],
        "devices": devices,
        "eval_device_weights": list(device_weights),
        "parallel_eval_batch": int(args.parallel_eval_batch),
        "parallel_worker_count": len(devices) if args.devices and len(devices) > 1 else 0,
        "acquisition_device": str(args.acquisition_device),
        "gp_fit_maxiter": int(args.gp_fit_maxiter),
        "baseline_prune_samples": args.baseline_prune_samples,
        "seed": int(args.seed),
        "initial_true_evals": int(args.initial_true_evals),
        "bo_iterations": int(args.bo_iterations),
        "output_every": int(args.output_every),
        "max_new_evals": args.max_new_evals,
        "evaluation_budget_reached": (
            None if args.max_new_evals is None else bool(new_count >= int(args.max_new_evals))
        ),
        "warm_start": bool(args.warm_start),
        "candidate_pool_size": int(args.candidate_pool_size),
        "max_pool_acq_eval": int(args.max_pool_acq_eval),
        "mc_samples": int(args.mc_samples),
        "max_observed_for_acq": int(args.max_observed_for_acq),
        "max_observed_for_pool": int(args.max_observed_for_pool),
        "evaluation_count": int(len(frame)),
        "valid_evaluation_count": int(len(valid_frame)),
        "seed_cache_imported_count": imported_count,
        "new_evaluation_count": new_count,
        "pareto_count": int(len(pareto)),
        "valid_pareto_count": int(len(valid_pareto)),
        "best_pdp": None if pareto.empty else float(pareto["pdp_fj"].min()),
        "best_top1_drop": None if pareto.empty else float(pareto["top1_drop"].min()),
        "best_area_um2": None if pareto.empty else float(pareto["area_um2"].min()),
        "best_valid_pdp": None if valid_pareto.empty else float(valid_pareto["pdp_fj"].min()),
        "best_valid_top1_drop": None if valid_pareto.empty else float(valid_pareto["top1_drop"].min()),
        "best_valid_area_um2": None if valid_pareto.empty else float(valid_pareto["area_um2"].min()),
        "top1_eval_method": STRICT50_TOP1_METHOD,
        "model_package_id": DEFAULT_STRICT50_MODEL_PACKAGE_ID,
        "package_hash": STRICT50_PACKAGE_HASH,
        "selected_images_hash": STRICT50_SELECTED_IMAGES_HASH,
        "feature_schema_hash": STRICT50_FEATURE_SCHEMA_HASH,
        "strict50_feature_count": STRICT50_FEATURE_DIM,
        "strict50_runtime_base_feature_count": STRICT50_RUNTIME_BASE_FEATURE_DIM,
        "strict50_supplemental_feature_count": STRICT50_SUPPLEMENTAL_FEATURE_DIM,
        "strict50_selected_image_count": DEFAULT_STRICT50_SELECTED_IMAGE_COUNT,
        "dataset_id": STRICT50_DATASET_ID,
        "dataset_schema_version": STRICT50_DATASET_SCHEMA_VERSION,
        "injection_scope": STRICT50_INJECTION_SCOPE,
        "node_registry_hash": STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
        "selected_node_count": STRICT50_EXPECTED_NODE_COUNT,
        "bank_meta_sha1": STRICT50_EXPECTED_BANK_META_SHA1,
        "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
        "seed_cache": None if args.seed_cache is None else str(args.seed_cache),
        "seed_cache_mode": str(args.seed_cache_mode),
        "seed_cache_limit": args.seed_cache_limit,
        "seed_import_stats": seed_import_stats,
        "drop_bin_targeting_enabled": not bool(args.disable_drop_bin_targeting),
        "drop_bin_min": float(args.drop_bin_min),
        "drop_bin_max": float(args.drop_bin_max),
        "drop_bin_step": float(args.drop_bin_step),
        "valid_drop_filter_enabled": not bool(args.disable_valid_drop_filter),
        "valid_drop_filter_margin": float(args.valid_drop_margin),
        "valid_drop_filter_min": None if bool(args.disable_valid_drop_filter) else float(args.drop_bin_min) - float(args.valid_drop_margin),
        "valid_drop_filter_max": None if bool(args.disable_valid_drop_filter) else float(args.drop_bin_max) + float(args.valid_drop_margin),
        "valid_biased_context_enabled": not bool(args.disable_valid_biased_context),
        "context_valid_drop_max": float(args.context_valid_drop_max),
        "context_global_fraction": float(args.context_global_fraction),
        "cache_path_pool_seeds": bool(args.cache_path_pool_seeds),
        "drop_bin_target_count": int(len(bin_curve)),
        "drop_bin_selected_count": int(len(selected_bins)),
        "drop_bin_missing_count": int(len(missing_bins)),
        "drop_bin_missing_ids": missing_bins["drop_bin_id"].astype(str).tolist() if not missing_bins.empty else [],
        "drop_bin_unique_design_count": int(selected_bins["config_hash"].nunique()) if not selected_bins.empty and "config_hash" in selected_bins else 0,
    }


def valid_drop_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if frame.empty or "top1_drop" not in frame.columns:
        return frame.iloc[0:0].copy()
    work = frame.copy()
    work["top1_drop"] = pd.to_numeric(work["top1_drop"], errors="coerce")
    return work[
        np.isfinite(work["top1_drop"])
        & (work["top1_drop"] >= float(args.drop_bin_min) - 1e-12)
        & (work["top1_drop"] <= float(args.drop_bin_max) + 1e-12)
    ].copy()


def build_valid_top1drop_bin_curve(valid_frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    curve = build_top1drop_bin_pdp_curve(
        valid_frame,
        drop_min=float(args.drop_bin_min),
        drop_max=float(args.drop_bin_max),
        step=float(args.drop_bin_step),
    )
    if not curve.empty or not valid_frame.empty:
        return curve

    rows: list[dict[str, Any]] = []
    for low in top1drop_bin_lowers(
        drop_min=float(args.drop_bin_min),
        drop_max=float(args.drop_bin_max),
        step=float(args.drop_bin_step),
    ):
        high = round(float(low) + float(args.drop_bin_step), 10)
        rows.append(
            {
                "drop_bin_id": f"drop_{int(round(low * 100)):02d}_{int(round(high * 100)):02d}",
                "target_top1_drop": float(low),
                "target_top1_drop_pct": float(low * 100.0),
                "bin_low_top1_drop": float(low),
                "bin_high_top1_drop": float(high),
                "bin_low_top1_drop_pct": float(low * 100.0),
                "bin_high_top1_drop_pct": float(high * 100.0),
                "selected": False,
            }
        )
    return pd.DataFrame(rows, columns=curve.columns)


def write_outputs(
    *,
    evaluator: Strict50BoEvaluator,
    output_dir: Path,
    args: argparse.Namespace,
    seed_import_stats: dict[str, Any] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = evaluator.cache_frame()
    if not frame.empty:
        frame = assign_design_ids(frame, prefix="bo_eval")
    pareto = build_pareto_frame(frame) if not frame.empty else pd.DataFrame()
    if not pareto.empty:
        pareto = assign_design_ids(pareto, prefix="bo_amcfg")
    valid_frame = valid_drop_frame(frame, args)
    if not valid_frame.empty:
        valid_frame = assign_design_ids(valid_frame, prefix="bo_eval_valid")
    valid_pareto = build_pareto_frame(valid_frame) if not valid_frame.empty else pd.DataFrame()
    if not valid_pareto.empty:
        valid_pareto = assign_design_ids(valid_pareto, prefix="bo_amcfg_valid")
    bin_curve = build_valid_top1drop_bin_curve(valid_frame, args)
    binned_front = selected_top1drop_bin_front(bin_curve)

    atomic_write_csv(frame, output_dir / "evaluation_cache.csv")
    atomic_write_csv(frame, output_dir / "evaluation_cache_live.csv")
    atomic_write_csv(valid_frame, output_dir / "evaluation_cache_valid_0_50.csv")
    atomic_write_csv(pareto, output_dir / "pareto_front.csv")
    atomic_write_csv(valid_pareto, output_dir / "pareto_front_valid_0_50.csv")
    atomic_write_csv(pareto.head(min(20, len(pareto))), output_dir / "best_solutions.csv")
    atomic_write_csv(valid_pareto.head(min(20, len(valid_pareto))), output_dir / "best_solutions_valid_0_50.csv")
    atomic_write_csv(bin_curve, output_dir / "top1drop_1pp_min_pdp_curve.csv")
    atomic_write_csv(binned_front, output_dir / "top1drop_binned_front.csv")
    atomic_write_json(
        output_dir / "summary.json",
        summarize(
            frame,
            pareto,
            valid_frame,
            valid_pareto,
            bin_curve,
            args=args,
            output_dir=output_dir,
            seed_import_stats=seed_import_stats,
        ),
    )
    return frame, pareto


def plot_pareto(cache_frame: pd.DataFrame, pareto_frame: pd.DataFrame, *, output_path: Path, plot_all: bool) -> None:
    plt.figure(figsize=(10, 8))
    has_labels = False
    if plot_all and not cache_frame.empty:
        plt.scatter(cache_frame["pdp_fj"], cache_frame["top1_drop"], s=16, alpha=0.22, color="#666666", label="evaluated")
        has_labels = True
    if not pareto_frame.empty:
        points = plt.scatter(
            pareto_frame["pdp_fj"],
            pareto_frame["top1_drop"],
            s=38,
            alpha=0.9,
            c=pareto_frame["area_um2"],
            cmap="viridis",
            edgecolor="black",
            linewidth=0.5,
            label="3-objective Pareto front",
        )
        plt.colorbar(points, label="GNN multiplier area (µm²)")
        has_labels = True
    plt.xlabel("PDP (fJ)")
    plt.ylabel("top1_drop")
    plt.title("PDP / accuracy projection of the 3-objective Pareto front")
    plt.grid(True, alpha=0.25)
    if has_labels:
        plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_top1drop_bin_curve(curve_frame: pd.DataFrame, *, output_path: Path) -> None:
    selected = curve_frame[curve_frame["selected"].astype(bool)].copy() if not curve_frame.empty and "selected" in curve_frame else pd.DataFrame()
    missing = curve_frame[~curve_frame["selected"].astype(bool)].copy() if not curve_frame.empty and "selected" in curve_frame else pd.DataFrame()
    plt.figure(figsize=(11, 7))
    has_labels = False
    if not selected.empty:
        plt.scatter(
            selected["pdp_fj"],
            selected["target_top1_drop_pct"],
            s=34,
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.35,
            label="1pp bin min PDP",
        )
        has_labels = True
    if not missing.empty:
        x_ref = float(selected["pdp_fj"].min()) if not selected.empty else 0.0
        plt.scatter(
            np.full(len(missing), x_ref),
            missing["target_top1_drop_pct"],
            s=24,
            marker="x",
            color="#c0392b",
            label="missing bin",
        )
        has_labels = True
    plt.xlabel("PDP (fJ)")
    plt.ylabel("top1_drop bin lower bound (%)")
    plt.grid(True, alpha=0.25)
    if has_labels:
        plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_current_bin_curve(observed: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = assign_design_ids(observed, prefix="bo_eval") if not observed.empty else observed.copy()
    return build_valid_top1drop_bin_curve(valid_drop_frame(frame, args), args)


def choose_missing_drop_bin(observed: pd.DataFrame, *, args: argparse.Namespace, iteration: int) -> tuple[float, float, str] | None:
    if bool(args.disable_drop_bin_targeting) or observed.empty:
        return None
    curve = build_current_bin_curve(observed, args)
    if curve.empty or "selected" not in curve.columns:
        return None
    missing = curve[~curve["selected"].astype(bool)].copy()
    if missing.empty:
        return None
    missing = missing.sort_values(["bin_low_top1_drop", "drop_bin_id"]).reset_index(drop=True)
    row = missing.iloc[int(iteration) % len(missing)]
    return float(row["bin_low_top1_drop"]), float(row["bin_high_top1_drop"]), str(row["drop_bin_id"])


def choose_target_drop_bin(
    observed: pd.DataFrame,
    *,
    args: argparse.Namespace,
    iteration: int,
    curve: pd.DataFrame | None = None,
) -> tuple[float, float, str] | None:
    if bool(args.disable_drop_bin_targeting) or observed.empty:
        return None
    if curve is None:
        curve = build_current_bin_curve(observed, args)
    if curve.empty or "selected" not in curve.columns:
        return None
    missing = curve[~curve["selected"].astype(bool)].copy()
    if not missing.empty:
        missing = missing.sort_values(["bin_low_top1_drop", "drop_bin_id"]).reset_index(drop=True)
        row = missing.iloc[int(iteration) % len(missing)]
        return float(row["bin_low_top1_drop"]), float(row["bin_high_top1_drop"]), str(row["drop_bin_id"])
    curve = curve.sort_values(["bin_low_top1_drop", "drop_bin_id"]).reset_index(drop=True)
    row = curve.iloc[int(iteration) % len(curve)]
    return float(row["bin_low_top1_drop"]), float(row["bin_high_top1_drop"]), str(row["drop_bin_id"])


def choose_target_drop_bins(
    observed: pd.DataFrame,
    *,
    args: argparse.Namespace,
    start_iteration: int,
    batch_size: int,
) -> list[tuple[float, float, str] | None]:
    """Use one snapshot of the current drop-bin curve for this BO batch."""
    curve = (
        build_current_bin_curve(
            observed[[name for name in ("config_hash", "pdp_fj", "top1_drop", "area_um2") if name in observed]],
            args,
        )
        if not bool(args.disable_drop_bin_targeting) and not observed.empty
        else None
    )
    return [
        choose_target_drop_bin(observed, args=args, iteration=start_iteration + index, curve=curve)
        for index in range(batch_size)
    ]


def _rows_by_hash(frame: pd.DataFrame, hashes: list[str]) -> pd.DataFrame:
    if frame.empty or "config_hash" not in frame.columns or not hashes:
        return frame.iloc[0:0].copy()
    order = {key: idx for idx, key in enumerate(hashes)}
    work = frame[frame["config_hash"].astype(str).isin(order)].copy()
    if work.empty:
        return work
    work["_order"] = work["config_hash"].astype(str).map(order)
    return work.sort_values("_order").drop(columns=["_order"])


def near_valid_drop_frame(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if frame.empty or "top1_drop" not in frame.columns:
        return frame.iloc[0:0].copy()
    work = frame.copy()
    work["top1_drop"] = pd.to_numeric(work["top1_drop"], errors="coerce")
    max_drop = max(float(args.drop_bin_max), float(args.context_valid_drop_max))
    return work[
        np.isfinite(work["top1_drop"])
        & (work["top1_drop"] >= float(args.drop_bin_min) - 1e-12)
        & (work["top1_drop"] <= max_drop + 1e-12)
    ].copy()


def select_observed_context(frame: pd.DataFrame, *, max_rows: int, args: argparse.Namespace, seed: int) -> pd.DataFrame:
    if frame.empty or int(max_rows) <= 0 or len(frame) <= int(max_rows):
        return frame
    # Context selection needs only objective values and hashes; carry the full
    # cached rows only after their hashes have been chosen.
    context_columns = [name for name in ("config_hash", "pdp_fj", "top1_drop", "area_um2") if name in frame]
    work = frame[context_columns].copy()
    hashes: list[str] = []
    seen: set[str] = set()

    def add_hashes(values: pd.Series | list[Any]) -> None:
        for value in values:
            key = str(value)
            if key and key not in seen:
                seen.add(key)
                hashes.append(key)

    valid_work = valid_drop_frame(work, args)
    near_valid_work = near_valid_drop_frame(work, args)
    context_global_fraction = min(max(float(args.context_global_fraction), 0.0), 1.0)
    global_budget = int(round(int(max_rows) * context_global_fraction)) if not bool(args.disable_valid_biased_context) else int(max_rows)
    focused_budget = max(0, int(max_rows) - global_budget)

    if not bool(args.disable_valid_biased_context):
        for column in ("pdp_fj", "top1_drop", "area_um2"):
            if column in valid_work.columns and not valid_work.empty:
                add_hashes(valid_work.nsmallest(64, column)["config_hash"].astype(str).tolist())
        try:
            valid_pareto = build_pareto_frame(valid_work)
            if "config_hash" in valid_pareto:
                add_hashes(spaced_pareto_hashes(valid_pareto, max_rows=focused_budget))
                add_hashes(valid_pareto["config_hash"].astype(str).tolist())
        except Exception:
            pass

        try:
            curve = build_valid_top1drop_bin_curve(assign_design_ids(valid_work, prefix="bo_eval_valid"), args)
            selected = selected_top1drop_bin_front(curve)
            if "config_hash" in selected:
                add_hashes(selected["config_hash"].dropna().astype(str).tolist())
        except Exception:
            pass

        for column in ("pdp_fj", "top1_drop", "area_um2"):
            if column in valid_work.columns and not valid_work.empty:
                add_hashes(valid_work.sort_values([column, "config_hash"])["config_hash"].head(512).astype(str).tolist())
            if column in near_valid_work.columns and not near_valid_work.empty:
                add_hashes(near_valid_work.sort_values([column, "config_hash"])["config_hash"].head(512).astype(str).tolist())
        if len(hashes) < focused_budget and not near_valid_work.empty:
            excluded = set(hashes)
            rest = near_valid_work[~near_valid_work["config_hash"].astype(str).isin(excluded)]
            if not rest.empty:
                sampled = rest.sample(n=min(focused_budget - len(hashes), len(rest)), random_state=int(seed))
                add_hashes(sampled["config_hash"].astype(str).tolist())

    try:
        pareto = build_pareto_frame(work if bool(args.disable_valid_biased_context) else near_valid_work)
        if "config_hash" in pareto:
            add_hashes(spaced_pareto_hashes(pareto, max_rows=int(max_rows)))
            add_hashes(pareto["config_hash"].astype(str).tolist())
    except Exception:
        pass

    try:
        curve = build_current_bin_curve(work if bool(args.disable_valid_biased_context) else valid_work, args)
        selected = selected_top1drop_bin_front(curve)
        if "config_hash" in selected:
            add_hashes(selected["config_hash"].dropna().astype(str).tolist())
    except Exception:
        pass

    if bool(args.disable_valid_biased_context):
        for column in ("pdp_fj", "top1_drop", "area_um2"):
            if column in work.columns:
                add_hashes(work.sort_values([column, "config_hash"])["config_hash"].head(256).astype(str).tolist())
    else:
        selected_hashes = hashes[:focused_budget]
        hashes = selected_hashes
        seen = set(hashes)
        if global_budget > 0:
            global_hashes: list[str] = []
            global_seen: set[str] = set()

            def add_global(values: pd.Series | list[Any]) -> None:
                for value in values:
                    key = str(value)
                    if key and key not in seen and key not in global_seen:
                        global_seen.add(key)
                        global_hashes.append(key)

            try:
                global_pareto = build_pareto_frame(work)
                if "config_hash" in global_pareto:
                    add_global(spaced_pareto_hashes(global_pareto, max_rows=global_budget))
                    add_global(global_pareto["config_hash"].astype(str).tolist())
            except Exception:
                pass
            for column in ("pdp_fj", "top1_drop", "area_um2"):
                if column in work.columns:
                    add_global(work.sort_values([column, "config_hash"])["config_hash"].head(128).astype(str).tolist())
            remaining_global = global_budget - len(global_hashes)
            if remaining_global > 0:
                excluded = set(hashes) | set(global_hashes)
                rest = work[~work["config_hash"].astype(str).isin(excluded)]
                if not rest.empty:
                    sampled = rest.sample(n=min(remaining_global, len(rest)), random_state=int(seed) + 17)
                    add_global(sampled["config_hash"].astype(str).tolist())
            hashes.extend(global_hashes[:global_budget])

    selected_hashes = hashes[: int(max_rows)]
    remaining_slots = int(max_rows) - len(selected_hashes)
    if remaining_slots > 0:
        excluded = set(selected_hashes)
        rest = work[~work["config_hash"].astype(str).isin(excluded)]
        if not rest.empty:
            sampled = rest.sample(n=min(remaining_slots, len(rest)), random_state=int(seed))
            add_hashes(sampled["config_hash"].astype(str).tolist())
    return _rows_by_hash(frame, hashes[: int(max_rows)])


def write_manifest(output_dir: Path, args: argparse.Namespace, *, seed_import_stats: dict[str, Any] | None) -> None:
    devices = parse_devices(args.device, args.devices)
    device_weights = parse_device_weights(args.eval_device_weights, len(devices))
    manifest = {
        "created_at": now_iso(),
        "command_args": vars(args),
        "objectives": ["pdp_fj", "top1_drop", "area_um2"],
        "optimizer": "discrete_pool_botorch_qlognehvi_with_top1drop_1pp_bin_targeting",
        "pdp_eval_method": "ppa_gnn_8bit",
        "area_eval_method": "ppa_gnn_8bit_multiplier_area_um2",
        "top1_eval_method": STRICT50_TOP1_METHOD,
        "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
        "model_key": str(args.model_key),
        "dataset_id": STRICT50_DATASET_ID,
        "dataset_schema_version": STRICT50_DATASET_SCHEMA_VERSION,
        "dataset_manifest": str(Path(STRICT50_DATASET_MANIFEST).resolve()),
        "injection_scope": STRICT50_INJECTION_SCOPE,
        "model_package_id": DEFAULT_STRICT50_MODEL_PACKAGE_ID,
        "package_hash": STRICT50_PACKAGE_HASH,
        "selected_images_hash": STRICT50_SELECTED_IMAGES_HASH,
        "feature_schema_hash": STRICT50_FEATURE_SCHEMA_HASH,
        "strict50_top1_package_dir": str(Path(STRICT50_TOP1_PACKAGE_DIR).resolve()),
        "strict50_bank_meta": str(Path(STRICT50_BANK_META).resolve()),
        "strict50_feature_count": STRICT50_FEATURE_DIM,
        "strict50_runtime_base_feature_count": STRICT50_RUNTIME_BASE_FEATURE_DIM,
        "strict50_supplemental_feature_count": STRICT50_SUPPLEMENTAL_FEATURE_DIM,
        "strict50_selected_image_count": DEFAULT_STRICT50_SELECTED_IMAGE_COUNT,
        "bank_meta_sha1": STRICT50_EXPECTED_BANK_META_SHA1,
        "sample_count": int(args.sample_count),
        "sample_seed": int(args.sample_seed),
        "gpu_id": gpu_id_from_device(devices[0]) if args.gpu_id is None else int(args.gpu_id),
        "device": devices[0],
        "devices": devices,
        "eval_device_weights": list(device_weights),
        "parallel_eval_batch": int(args.parallel_eval_batch),
        "parallel_worker_count": len(devices) if args.devices and len(devices) > 1 else 0,
        "acquisition_device": str(args.acquisition_device),
        "gp_fit_maxiter": int(args.gp_fit_maxiter),
        "baseline_prune_samples": args.baseline_prune_samples,
        "candidate_pool_size": int(args.candidate_pool_size),
        "max_pool_acq_eval": int(args.max_pool_acq_eval),
        "mc_samples": int(args.mc_samples),
        "output_every": int(args.output_every),
        "max_new_evals": args.max_new_evals,
        "warm_start": bool(args.warm_start),
        "allow_existing_cache": bool(args.allow_existing_cache),
        "node_registry_path": str(Path(STRICT50_NODE_REGISTRY).resolve()),
        "node_registry_hash": sha1_file(Path(STRICT50_NODE_REGISTRY)),
        "selected_node_count": STRICT50_EXPECTED_NODE_COUNT,
        "seed_cache_path": None if args.seed_cache is None else str(args.seed_cache),
        "seed_cache_mode": str(args.seed_cache_mode),
        "seed_cache_limit": args.seed_cache_limit,
        "seed_import_stats": seed_import_stats,
        "max_observed_for_acq": int(args.max_observed_for_acq),
        "max_observed_for_pool": int(args.max_observed_for_pool),
        "drop_bin_targeting_enabled": not bool(args.disable_drop_bin_targeting),
        "drop_bin_min": float(args.drop_bin_min),
        "drop_bin_max": float(args.drop_bin_max),
        "drop_bin_step": float(args.drop_bin_step),
        "valid_drop_filter_enabled": not bool(args.disable_valid_drop_filter),
        "valid_drop_filter_margin": float(args.valid_drop_margin),
        "valid_drop_filter_min": None if bool(args.disable_valid_drop_filter) else float(args.drop_bin_min) - float(args.valid_drop_margin),
        "valid_drop_filter_max": None if bool(args.disable_valid_drop_filter) else float(args.drop_bin_max) + float(args.valid_drop_margin),
        "valid_biased_context_enabled": not bool(args.disable_valid_biased_context),
        "context_valid_drop_max": float(args.context_valid_drop_max),
        "context_global_fraction": float(args.context_global_fraction),
        "cache_path_pool_seeds": bool(args.cache_path_pool_seeds),
        "warm_start_paths": [str(path) for path in warm_start_cache_paths(args)],
        "token_columns": TOKEN_COLUMNS,
    }
    atomic_write_json(output_dir / "run_manifest.json", json_safe(manifest))


def evaluate_initial_designs(
    *,
    evaluator: Strict50BoEvaluator,
    eval_pool: ParallelStrict50EvaluationPool | None,
    output_dir: Path,
    args: argparse.Namespace,
    seed_import_stats: dict[str, Any] | None,
    start_evaluation_count: int,
) -> None:
    observed = evaluator.cache_frame()
    evaluated = set(observed["config_hash"].astype(str)) if "config_hash" in observed else set()
    cache_paths = [] if not bool(args.warm_start) or not bool(args.cache_path_pool_seeds) else warm_start_cache_paths(args)
    pool_context = select_observed_context(
        observed,
        max_rows=int(args.max_observed_for_pool),
        args=args,
        seed=int(args.seed),
    )
    pool = generate_candidate_pool(
        pool_size=max(int(args.initial_true_evals) * 4, 256),
        seed=int(args.seed),
        observed=token_columns_from_frame(pool_context) if not pool_context.empty else (),
        cache_paths=cache_paths,
        include_warm_start=bool(args.warm_start),
    )
    pool = filter_unevaluated(pool, evaluated)

    remaining_budget = None
    if args.max_new_evals is not None:
        used = max(0, len(evaluator.cache.rows) - int(start_evaluation_count))
        remaining_budget = max(0, int(args.max_new_evals) - used)
    initial_to_add = int(args.initial_true_evals)
    if remaining_budget is not None:
        initial_to_add = min(initial_to_add, remaining_budget)
    target = len(evaluator.cache.rows) + initial_to_add
    if len(observed) < target and not evaluator.can_evaluate_new and eval_pool is None:
        raise RuntimeError("--no-model-init was used but initial_true_evals requires new strict50 evaluations")
    needed = max(0, target - len(evaluator.cache_frame()))
    selected = pool[:needed]
    eval_batch = max(1, int(args.parallel_eval_batch))
    for start in range(0, len(selected), eval_batch):
        batch = selected[start : start + eval_batch]
        results = evaluate_token_batch(evaluator, eval_pool, batch, log_every=int(args.batch_log_every))
        if not results:
            break
        for offset, result in enumerate(results):
            append_or_write_csv(
                {
                    "iteration": -1,
                    "phase": "initial",
                    "batch_index": offset,
                    "config_hash": result["config_hash"],
                    "candidate_source": "initial_pool",
                    "acquisition_name": "initial_design",
                    "acquisition_value": np.nan,
                    "target_drop_bin_id": None,
                    "target_drop_bin_low": np.nan,
                    "target_drop_bin_high": np.nan,
                    "pred_mu_top1_drop": np.nan,
                    "pred_sigma_top1_drop": np.nan,
                    "pred_mu_pdp_fj": np.nan,
                    "pred_sigma_pdp_fj": np.nan,
                    "pred_mu_area_um2": np.nan,
                    "pred_sigma_area_um2": np.nan,
                    "rank_in_pool": start + offset,
                    "pool_size": len(pool),
                    "weight_pdp": np.nan,
                    "weight_top1": np.nan,
                    "weight_area": np.nan,
                    "parallel_batch_size": len(batch),
                    "true_eval_status": "ok",
                    "created_at": now_iso(),
                },
                output_dir / "bo_trials.csv",
            )
        batch_number = start // eval_batch + 1
        if batch_number % int(args.output_every) == 0:
            write_outputs(evaluator=evaluator, output_dir=output_dir, args=args, seed_import_stats=seed_import_stats)


def evaluate_token_batch(
    evaluator: Strict50BoEvaluator,
    eval_pool: ParallelStrict50EvaluationPool | None,
    tokens_list: list[tuple[int, ...]],
    *,
    log_every: int = 0,
) -> list[dict[str, Any]]:
    if not tokens_list:
        return []
    if eval_pool is None:
        initial_rows = len(evaluator.cache.rows)
        try:
            return [evaluator.evaluate_tokens(tokens, save=False) for tokens in tokens_list]
        finally:
            if len(evaluator.cache.rows) > initial_rows:
                evaluator.save_cache_csv()
    rows = eval_pool.evaluate(tokens_list, log_every=log_every)
    merge_rows_into_evaluator(evaluator, rows)
    out: list[dict[str, Any]] = []
    for tokens, row in zip(tokens_list, rows, strict=False):
        key = build_candidate(tokens).config_hash()
        out.append(evaluator.cache.get(key) or row)
    return out


def choose_candidate_batch(
    *,
    observed: pd.DataFrame,
    acq_context: pd.DataFrame,
    pool: list[tuple[int, ...]],
    args: argparse.Namespace,
    iteration: int,
) -> list[tuple[int, Any, tuple[float, float, str] | None]]:
    batch_size = max(1, int(args.parallel_eval_batch))
    target_bins = choose_target_drop_bins(
        observed,
        args=args,
        start_iteration=iteration * batch_size,
        batch_size=batch_size,
    )
    target_top1_drop_bins = [
        None if target_bin is None else (target_bin[0], target_bin[1])
        for target_bin in target_bins
    ]
    valid_top1_drop_range = None
    if not bool(args.disable_valid_drop_filter):
        margin = max(0.0, float(args.valid_drop_margin))
        valid_top1_drop_range = (
            float(args.drop_bin_min) - margin,
            float(args.drop_bin_max) + margin,
        )
    raw_choices = choose_next_candidates(
        observed=acq_context,
        pool_tokens=list(pool),
        seed=int(args.seed) + iteration * 10_000,
        batch_size=batch_size,
        mc_samples=int(args.mc_samples),
        max_pool_eval=int(args.max_pool_acq_eval),
        gp_fit_maxiter=int(args.gp_fit_maxiter),
        acquisition_device=str(args.acquisition_device),
        baseline_prune_samples=args.baseline_prune_samples,
        target_top1_drop_bins=target_top1_drop_bins,
        valid_top1_drop_range=valid_top1_drop_range,
    )

    choices: list[tuple[int, Any, tuple[float, float, str] | None]] = []
    selected_hashes: set[str] = set()
    for batch_idx, choice in enumerate(raw_choices):
        key = build_candidate(choice.tokens).config_hash()
        if key in selected_hashes:
            continue
        selected_hashes.add(key)
        target_bin = target_bins[batch_idx] if batch_idx < len(target_bins) else None
        choices.append((batch_idx, choice, target_bin))
    return choices


def run_bo(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.output_every) <= 0:
        raise ValueError("--output-every must be positive")
    if args.max_new_evals is not None and int(args.max_new_evals) <= 0:
        raise ValueError("--max-new-evals must be positive when provided")
    devices = parse_devices(args.device, args.devices)
    use_parallel = bool(args.devices) and len(devices) > 1
    device_weights = parse_device_weights(args.eval_device_weights, len(devices))
    if not use_parallel and args.eval_device_weights is not None:
        raise ValueError("--eval-device-weights requires multi-device --devices")
    selected_gpu_id = gpu_id_from_device(devices[0])
    if args.gpu_id is not None and int(args.gpu_id) != selected_gpu_id:
        raise ValueError("--gpu-id must match the first evaluation device in --devices/--device")
    args.gpu_id = selected_gpu_id
    seed_everything(int(args.seed))
    output_dir = (Path(args.output_root) / str(args.run_name)).resolve()
    prepare_output_dir(output_dir, allow_existing_cache=bool(args.allow_existing_cache))
    cache_csv = output_dir / "evaluation_cache_live.csv"
    args.parallel_eval_batch = resolve_parallel_eval_batch(args, devices)

    evaluator = Strict50BoEvaluator(
        cache_csv=cache_csv,
        model_key=str(args.model_key),
        sample_count=int(args.sample_count),
        sample_seed=int(args.sample_seed),
        batch_size=int(args.batch_size),
        gpu_id=int(args.gpu_id),
        sessions_per_gpu=int(args.sessions_per_gpu),
        device=args.device or devices[0],
        initialize_models=(not bool(args.no_model_init)) and (not use_parallel),
    )
    eval_pool: ParallelStrict50EvaluationPool | None = None
    seed_import_stats: dict[str, Any] | None = None
    try:
        if use_parallel:
            eval_pool = ParallelStrict50EvaluationPool(
                devices=devices,
                device_weights=device_weights,
                python_executable=sys.executable,
                cwd=EA2_ROOT,
                model_key=str(args.model_key),
                sample_count=int(args.sample_count),
                sample_seed=int(args.sample_seed),
                batch_size=int(args.batch_size),
                sessions_per_gpu=int(args.sessions_per_gpu),
            )
        if str(args.seed_cache_mode) == "import_as_observed" and args.seed_cache is not None:
            seed_import_stats = evaluator.import_seed_cache(
                args.seed_cache,
                limit=args.seed_cache_limit,
                overwrite=False,
                save=False,
            )
            if seed_import_stats["rows_seen"] and seed_import_stats["rows_imported"] == 0 and seed_import_stats["rows_skipped"]:
                raise ValueError(f"seed cache has no compatible area_um2 rows: {args.seed_cache}")
            if seed_import_stats["rows_imported"]:
                evaluator.save_cache_csv()

        start_evaluation_count = int(len(evaluator.cache.rows))
        write_manifest(output_dir, args, seed_import_stats=seed_import_stats)
        evaluate_initial_designs(
            evaluator=evaluator,
            eval_pool=eval_pool,
            output_dir=output_dir,
            args=args,
            seed_import_stats=seed_import_stats,
            start_evaluation_count=start_evaluation_count,
        )
        if not bool(args.stop_after_initial):
            if int(args.bo_iterations) > 0 and not evaluator.can_evaluate_new and eval_pool is None:
                raise RuntimeError("--no-model-init was used but bo_iterations requires new strict50 evaluations")
            for iteration in range(int(args.bo_iterations)):
                used_evaluations = max(0, len(evaluator.cache.rows) - start_evaluation_count)
                if args.max_new_evals is not None and used_evaluations >= int(args.max_new_evals):
                    break
                observed = evaluator.cache_frame()
                evaluated = set(observed["config_hash"].astype(str)) if "config_hash" in observed else set()
                pool_context = select_observed_context(
                    observed,
                    max_rows=int(args.max_observed_for_pool),
                    args=args,
                    seed=int(args.seed) + 101 * iteration,
                )
                acq_context = select_observed_context(
                    observed,
                    max_rows=int(args.max_observed_for_acq),
                    args=args,
                    seed=int(args.seed) + 211 * iteration,
                )
                cache_paths = [] if not bool(args.warm_start) or not bool(args.cache_path_pool_seeds) else warm_start_cache_paths(args)
                pool = generate_candidate_pool(
                    pool_size=int(args.candidate_pool_size),
                    seed=int(args.seed) + 10_000 + iteration,
                    observed=token_columns_from_frame(pool_context) if not pool_context.empty else (),
                    cache_paths=cache_paths,
                    include_warm_start=bool(args.warm_start),
                    neighbor_radius=1,
                    neighbor_limit_per_observed=48,
                )
                pool = filter_unevaluated(pool, evaluated)
                if not pool:
                    break
                chosen = choose_candidate_batch(
                    observed=acq_context,
                    acq_context=acq_context,
                    pool=pool,
                    args=args,
                    iteration=iteration,
                )
                if not chosen:
                    break
                if args.max_new_evals is not None:
                    remaining = int(args.max_new_evals) - used_evaluations
                    chosen = chosen[: max(0, remaining)]
                if not chosen:
                    break
                tokens_to_eval = [choice.tokens for _batch_idx, choice, _target_bin in chosen]
                results = evaluate_token_batch(evaluator, eval_pool, tokens_to_eval, log_every=int(args.batch_log_every))
                for result, (batch_idx, choice, target_bin) in zip(results, chosen, strict=True):
                    append_or_write_csv(
                        {
                            "iteration": iteration,
                            "phase": "bo",
                            "batch_index": int(batch_idx),
                            "config_hash": result["config_hash"],
                            "candidate_source": "bo_candidate_pool",
                            "acquisition_name": choice.acquisition_name,
                            "acquisition_value": choice.acquisition_value,
                            "target_drop_bin_id": None if target_bin is None else target_bin[2],
                            "target_drop_bin_low": np.nan if target_bin is None else target_bin[0],
                            "target_drop_bin_high": np.nan if target_bin is None else target_bin[1],
                            "pred_mu_top1_drop": choice.pred_mu_top1_drop,
                            "pred_sigma_top1_drop": choice.pred_sigma_top1_drop,
                            "pred_mu_pdp_fj": choice.pred_mu_pdp_fj,
                            "pred_sigma_pdp_fj": choice.pred_sigma_pdp_fj,
                            "pred_mu_area_um2": choice.pred_mu_area_um2,
                            "pred_sigma_area_um2": choice.pred_sigma_area_um2,
                            "rank_in_pool": choice.rank_in_pool,
                            "pool_size": choice.pool_size,
                            "weight_pdp": choice.weight_pdp,
                            "weight_top1": choice.weight_top1,
                            "weight_area": choice.weight_area,
                            "parallel_batch_size": len(chosen),
                            "true_eval_status": "ok",
                            "observed_for_acq": int(len(acq_context)),
                            "observed_for_pool": int(len(pool_context)),
                            "created_at": now_iso(),
                        },
                        output_dir / "bo_trials.csv",
                    )
                checkpoint = (iteration + 1) % int(args.output_every) == 0
                if checkpoint:
                    frame, pareto = write_outputs(
                        evaluator=evaluator,
                        output_dir=output_dir,
                        args=args,
                        seed_import_stats=seed_import_stats,
                    )
                if checkpoint and bool(args.plot_all):
                    plot_pareto(frame, pareto, output_path=output_dir / "pareto_front.png", plot_all=True)
                    valid_frame = valid_drop_frame(frame, args)
                    valid_pareto = build_pareto_frame(valid_frame) if not valid_frame.empty else pd.DataFrame()
                    plot_pareto(
                        valid_frame,
                        valid_pareto,
                        output_path=output_dir / "pareto_front_valid_0_50.png",
                        plot_all=True,
                    )
                    plot_top1drop_bin_curve(
                        build_valid_top1drop_bin_curve(valid_frame, args),
                        output_path=output_dir / "top1drop_1pp_min_pdp_curve.png",
                    )

        frame, pareto = write_outputs(evaluator=evaluator, output_dir=output_dir, args=args, seed_import_stats=seed_import_stats)
        plot_pareto(frame, pareto, output_path=output_dir / "pareto_front.png", plot_all=bool(args.plot_all))
        valid_frame = valid_drop_frame(frame, args)
        valid_pareto = build_pareto_frame(valid_frame) if not valid_frame.empty else pd.DataFrame()
        plot_pareto(
            valid_frame,
            valid_pareto,
            output_path=output_dir / "pareto_front_valid_0_50.png",
            plot_all=bool(args.plot_all),
        )
        bin_curve = build_valid_top1drop_bin_curve(valid_frame, args)
        plot_top1drop_bin_curve(bin_curve, output_path=output_dir / "top1drop_1pp_min_pdp_curve.png")
        summary = summarize(
            frame,
            pareto,
            valid_frame,
            valid_pareto,
            bin_curve,
            args=args,
            output_dir=output_dir,
            seed_import_stats=seed_import_stats,
        )
        print(json.dumps(json_safe(summary), ensure_ascii=False, sort_keys=True))
        return summary
    finally:
        if eval_pool is not None:
            eval_pool.close()
        evaluator.close()


def main() -> None:
    run_bo(build_parser().parse_args())


if __name__ == "__main__":
    main()
