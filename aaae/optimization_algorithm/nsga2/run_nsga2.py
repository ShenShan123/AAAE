from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import (
    DEFAULT_STRICT50_OUTPUT_DIR,
    DEFAULT_STRICT50_RUN_NAME,
    DEFAULT_RUN_NAME,
    FIXED_HIGH_TOKENS,
    LOW_TOKEN_COUNT,
    LOW_TOKEN_MAX,
    LOW_TOKEN_MIN,
    STRICT50_BANK_META,
    STRICT50_DATASET_ID,
    STRICT50_DATASET_MANIFEST,
    STRICT50_DATASET_SCHEMA_VERSION,
    STRICT50_EXPECTED_BANK_META_SHA1,
    STRICT50_EXPECTED_FEATURE_SCHEMA_HASH,
    STRICT50_EXPECTED_NODE_COUNT,
    STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
    STRICT50_EXPECTED_PACKAGE_HASH,
    STRICT50_EXPECTED_PACKAGE_ID,
    STRICT50_EXPECTED_SELECTED_IMAGES_HASH,
    STRICT50_FEATURE_DIM,
    STRICT50_INJECTION_SCOPE,
    STRICT50_NODE_REGISTRY,
    STRICT50_RUNTIME_BASE_FEATURE_DIM,
    STRICT50_RUNTIME_DELTA_SOURCE,
    STRICT50_S8_RESIDUAL_ABS_BOUND,
    STRICT50_SUPPLEMENTAL_FEATURE_DIM,
    STRICT50_TOP1_PACKAGE_DIR,
)
from .evaluators import (
    DEFAULT_TRUE1000_BATCH_SIZE,
    DEFAULT_TRUE1000_MODEL_KEY,
    DEFAULT_TRUE1000_SAMPLE_COUNT,
    DEFAULT_TRUE1000_SAMPLE_SEED,
    DEFAULT_TRUE1000_SESSION_PROFILE,
    DEFAULT_TRUE1000_SESSIONS_PER_GPU,
    TRUE1000_TOP1_METHOD,
    CandidateConfig,
    DMultTruthTableEvaluator,
    EvaluationResult,
    OptimizationEvaluator,
)
from .parallel_evaluation import ParallelEvaluationPool, merge_results_into_evaluator, parse_devices
from aaae.optimization_algorithm.common.device_sharding import parse_device_weights
from aaae.optimization_algorithm.common.pareto3 import nondominated_mask_3d


from aaae.optimization_algorithm.common.nsga2_core.population import Population
from aaae.optimization_algorithm.common.nsga2_core.problem import Problem
from aaae.optimization_algorithm.common.nsga2_core.utils import NSGA2Utils


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def require_area_cache_csv(path: Path) -> None:
    """Reject old two-objective cache files before any run can write output."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if "area_um2" not in header:
        raise FileExistsError(f"cannot reuse a cache without area_um2: {path}")
    area = pd.to_numeric(pd.read_csv(path, usecols=["area_um2"])["area_um2"], errors="coerce").to_numpy(dtype=float)
    if np.any(~np.isfinite(area) | (area <= 0)):
        raise FileExistsError(f"cannot reuse a cache with invalid area_um2 values: {path}")


def validate_seed_cache(
    path: Path,
    *,
    top1_eval_method: str,
    true1000_model_key: str = DEFAULT_TRUE1000_MODEL_KEY,
    true1000_sample_count: int = DEFAULT_TRUE1000_SAMPLE_COUNT,
    true1000_sample_seed: int = DEFAULT_TRUE1000_SAMPLE_SEED,
    true1000_session_profile: str = DEFAULT_TRUE1000_SESSION_PROFILE,
) -> tuple[list[EvaluationResult], dict[str, Any]]:
    """Load an area cache read-only and require every row to match this run."""
    source = path.resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"seed cache must be a CSV file: {source}")
    validator = OptimizationEvaluator(
        top1_eval_method=top1_eval_method,
        cache_csv=source,
        device="cpu",
        true1000_model_key=true1000_model_key,
        true1000_sample_count=true1000_sample_count,
        true1000_sample_seed=true1000_sample_seed,
        true1000_session_profile=true1000_session_profile,
        initialize_models=False,
    )
    try:
        report = dict(validator.cache_load_report)
        source_rows = int(report["source_rows"])
        loaded_rows = int(report["loaded_rows"])
        if source_rows <= 0 or int(report["rejected_rows"]) or loaded_rows != source_rows:
            raise ValueError(f"seed cache contains empty, duplicate, or protocol-incompatible rows: {source}; {report}")
        results = list(validator.cache.values())
        for result in results:
            if result.approx_high_part or result.high_tokens != FIXED_HIGH_TOKENS:
                raise ValueError(f"seed cache contains a non-canonical AMCFG configuration: {result.config_hash}")
            try:
                config = CandidateConfig(low_tokens=result.low_tokens, high_tokens=result.high_tokens, approx_high_part=False)
            except ValueError as exc:
                raise ValueError(f"seed cache contains invalid AMCFG tokens: {result.config_hash}") from exc
            if result.config_hash != config.config_hash():
                raise ValueError(f"seed cache has a mismatched config_hash: {result.config_hash}")
            if not np.isfinite([result.pdp_fj, result.top1_drop, result.delay_ns, result.power_uw]).all():
                raise ValueError(f"seed cache has non-finite objective or PPA values: {result.config_hash}")
            if result.pdp_fj <= 0 or result.delay_ns <= 0 or result.power_uw <= 0:
                raise ValueError(f"seed cache has non-positive PDP, delay, or power: {result.config_hash}")
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata = {
            "path": str(source),
            "sha256": digest.hexdigest(),
            "source_rows": source_rows,
            "imported_rows": loaded_rows,
            "mode": "evaluation_cache_only",
            "initial_population_seeded": False,
        }
        return results, metadata
    finally:
        validator.close()


def install_seed_results(
    evaluator: OptimizationEvaluator,
    results: list[EvaluationResult],
    metadata: dict[str, Any],
) -> None:
    """Create a new run-local cache from already validated seed rows."""
    if evaluator.cache_csv is None or evaluator.cache:
        raise FileExistsError("seed results require a new, empty run-local cache")
    for result in results:
        evaluator.add_result(result, save=False)
    evaluator.save_cache_csv()
    evaluator.cache_load_report = {
        "source_rows": int(metadata["source_rows"]),
        "loaded_rows": int(metadata["imported_rows"]),
        "rejected_rows": 0,
        "rejection_counts": {},
    }


def require_fresh_seed_destination(output_dir: Path, *, allow_existing_cache: bool) -> None:
    if allow_existing_cache:
        raise ValueError("--seed-cache requires a fresh output directory and cannot be combined with --allow-existing-cache")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"--seed-cache requires an empty output directory: {output_dir}")


def prepare_output_dir(output_dir: Path, *, allow_existing_cache: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    live_cache = output_dir / "evaluation_cache_live.csv"
    manifest_path = output_dir / "run_manifest.json"
    final_artifacts = ("evaluation_cache.csv", "pareto_front.csv", "summary.json")
    has_final_artifacts = any((output_dir / name).exists() for name in final_artifacts)
    if (live_cache.exists() or manifest_path.exists() or has_final_artifacts) and not allow_existing_cache:
        raise FileExistsError(
            "output directory already has run artifacts; choose a fresh --output-dir "
            "for a from-zero experiment or pass --allow-existing-cache explicitly"
        )
    if allow_existing_cache and has_final_artifacts and not manifest_path.exists():
        raise FileExistsError(f"cannot reuse completed run artifacts without a three-objective manifest: {output_dir}")
    if allow_existing_cache and (has_final_artifacts or manifest_path.exists()) and not live_cache.exists():
        raise FileExistsError(f"cannot reuse run artifacts without a live three-objective cache: {output_dir}")
    if allow_existing_cache and live_cache.exists():
        require_area_cache_csv(live_cache)
    if allow_existing_cache and manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("objectives") != ["pdp_fj", "top1_drop", "area_um2"]:
            raise FileExistsError(f"cannot reuse a non-three-objective run directory: {output_dir}")


def build_candidate_from_state(state: list[int] | tuple[int, ...]) -> CandidateConfig:
    return CandidateConfig(low_tokens=tuple(int(token) for token in state), high_tokens=FIXED_HIGH_TOKENS, approx_high_part=False)


def make_objective_functions(evaluator: OptimizationEvaluator):
    def objective_pdp(state: list[int]) -> float:
        return float(evaluator.evaluate(build_candidate_from_state(state)).pdp_fj)

    def objective_top1drop(state: list[int]) -> float:
        return float(evaluator.evaluate(build_candidate_from_state(state)).top1_drop)

    def objective_area(state: list[int]) -> float:
        return float(evaluator.evaluate(build_candidate_from_state(state)).area_um2)

    return objective_pdp, objective_top1drop, objective_area


def normalize_top1_eval_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"strict50", "strict50_v3", "50shot", "50_shot"}:
        return "strict50_v3"
    if normalized in {"true1000", "true_1000", "1000shot", "1000_shot", TRUE1000_TOP1_METHOD}:
        return TRUE1000_TOP1_METHOD
    raise ValueError(f"unsupported top1 eval method: {value}")


def resolve_preflight_device(requested: str, evaluation_device: str) -> str:
    if str(requested) != "auto":
        return str(requested)
    if str(evaluation_device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(evaluation_device)


class CanonicalRuntimeCandidateFilter:
    """Keep NSGA-II proposals inside the residual range accepted by the runtime."""

    def __init__(
        self,
        *,
        predict_deltas: Callable[[list[CandidateConfig]], np.ndarray] | None = None,
        predict_max_abs_residuals: Callable[[list[CandidateConfig]], np.ndarray] | None = None,
        residual_abs_bound: int = STRICT50_S8_RESIDUAL_ABS_BOUND,
        device: str = "cpu",
    ) -> None:
        if predict_deltas is None and predict_max_abs_residuals is None:
            raise ValueError("a runtime candidate residual predictor is required")
        self.predict_deltas = predict_deltas
        self.predict_max_abs_residuals = predict_max_abs_residuals
        self.residual_abs_bound = int(residual_abs_bound)
        self.device = str(device)
        self._accepted_by_hash: dict[str, bool] = {}
        self._max_abs_by_hash: dict[str, int] = {}
        self.screened_unique_count = 0
        self.rejected_unique_count = 0

    def filter_individuals(self, individuals: list[Any]) -> list[Any]:
        pending_by_hash: dict[str, CandidateConfig] = {}
        for individual in individuals:
            config = build_candidate_from_state(individual.features)
            key = config.config_hash()
            if key not in self._accepted_by_hash:
                pending_by_hash.setdefault(key, config)

        if pending_by_hash:
            configs = list(pending_by_hash.values())
            if self.predict_max_abs_residuals is None:
                assert self.predict_deltas is not None
                deltas = np.asarray(self.predict_deltas(configs), dtype=np.int32)
                expected_shape = (len(configs), 256, 256)
                if deltas.shape != expected_shape:
                    raise ValueError(
                        "runtime candidate preflight returned unexpected delta shape: "
                        f"{deltas.shape}, expected {expected_shape}"
                    )
                max_abs = np.max(np.abs(deltas.astype(np.int64, copy=False)), axis=(1, 2))
            else:
                max_abs = np.asarray(self.predict_max_abs_residuals(configs), dtype=np.int32)
                if max_abs.shape != (len(configs),):
                    raise ValueError(
                        "runtime candidate preflight returned unexpected maximum residual shape: "
                        f"{max_abs.shape}, expected {(len(configs),)}"
                    )
            for key, value in zip(pending_by_hash, max_abs, strict=True):
                value_int = int(value)
                self._max_abs_by_hash[key] = value_int
                self._accepted_by_hash[key] = value_int <= self.residual_abs_bound
                self.screened_unique_count += 1
                if not self._accepted_by_hash[key]:
                    self.rejected_unique_count += 1

        return [
            individual
            for individual in individuals
            if self._accepted_by_hash[build_candidate_from_state(individual.features).config_hash()]
        ]

    def summary(self) -> dict[str, object]:
        rejected = [
            {"config_hash": key, "max_abs": value}
            for key, value in self._max_abs_by_hash.items()
            if not self._accepted_by_hash[key]
        ]
        return {
            "enabled": True,
            "device": self.device,
            "residual_abs_bound": int(self.residual_abs_bound),
            "screened_unique_count": int(self.screened_unique_count),
            "rejected_unique_count": int(self.rejected_unique_count),
            "rejected_examples": rejected[:10],
        }


class BatchedCandidateEvaluator:
    """Evaluate one NSGA-II generation through a single run-local cache."""

    def __init__(
        self,
        *,
        evaluator: OptimizationEvaluator,
        pool: ParallelEvaluationPool | None,
        start_evaluation_count: int,
        max_new_evals: int | None,
        log_every: int,
    ) -> None:
        self.evaluator = evaluator
        self.pool = pool
        self.start_evaluation_count = int(start_evaluation_count)
        self.max_new_evals = None if max_new_evals is None else int(max_new_evals)
        self.log_every = int(log_every)

    @property
    def new_evaluation_count(self) -> int:
        return max(0, int(len(self.evaluator.cache)) - self.start_evaluation_count)

    def remaining_new_evaluations(self) -> int | None:
        if self.max_new_evals is None:
            return None
        return max(0, int(self.max_new_evals) - self.new_evaluation_count)

    def evaluate_states(self, states: list[list[int] | tuple[int, ...]]) -> list[tuple[float, float, float]]:
        configs = [build_candidate_from_state(tuple(int(value) for value in state)) for state in states]
        pending: list[CandidateConfig] = []
        queued_hashes: set[str] = set()
        for config in configs:
            key = config.config_hash()
            if key in self.evaluator.cache or key in queued_hashes:
                continue
            queued_hashes.add(key)
            pending.append(config)

        if pending:
            remaining = self.remaining_new_evaluations()
            if remaining is not None and len(pending) > remaining:
                raise RuntimeError(
                    "NSGA-II evaluation budget would be exceeded: "
                    f"pending={len(pending)} remaining={remaining} max_new_evals={self.max_new_evals}"
                )
            if self.pool is None:
                self.evaluator.evaluate_configs(pending)
            else:
                results = self.pool.evaluate_configs(pending, log_every=self.log_every)
                merge_results_into_evaluator(self.evaluator, results)

        objectives: list[tuple[float, float, float]] = []
        for config in configs:
            result = self.evaluator.cache.get(config.config_hash())
            if result is None:
                raise RuntimeError(f"missing evaluation result for {config.config_hash()}")
            objectives.append((float(result.pdp_fj), float(result.top1_drop), float(result.area_um2)))
        return objectives


class BatchedEvolution:
    """Vanilla OpenACM NSGA-II selection with generation-level batch evaluation."""

    def __init__(
        self,
        *,
        problem: Problem,
        batch_evaluator: BatchedCandidateEvaluator,
        num_of_generations: int,
        num_of_individuals: int,
        num_of_tour_particips: int,
        tournament_prob: float,
        crossover_param: float,
        mutation_param: float,
        runtime_candidate_filter: CanonicalRuntimeCandidateFilter | None = None,
    ) -> None:
        self.problem = problem
        self.utils = NSGA2Utils(
            problem,
            num_of_individuals,
            num_of_tour_particips,
            tournament_prob,
            crossover_param,
            mutation_param,
        )
        self.batch_evaluator = batch_evaluator
        self.population: Population | None = None
        self.num_of_generations = int(num_of_generations)
        self.num_of_individuals = int(num_of_individuals)
        self.generation_history: list[dict[str, Any]] = []
        self.runtime_candidate_filter = runtime_candidate_filter

    def _filter_runtime_valid(self, individuals: list[Any]) -> list[Any]:
        if self.runtime_candidate_filter is None:
            return individuals
        return self.runtime_candidate_filter.filter_individuals(individuals)

    def _create_runtime_valid_initial_population(self) -> Population:
        population = Population()
        attempts = 0
        while len(population) < self.num_of_individuals:
            remaining = self.num_of_individuals - len(population)
            proposals = [self.problem.generate_individual() for _ in range(max(remaining, 16))]
            population.extend(self._filter_runtime_valid(proposals)[:remaining])
            attempts += 1
            if attempts > 1_000:
                raise RuntimeError("unable to create a runtime-valid NSGA-II initial population")
        return population

    def _create_runtime_valid_children(self, *, count: int) -> list[Any]:
        if self.population is None:
            raise RuntimeError("NSGA-II population is not initialized")
        children: list[Any] = []
        attempts = 0
        while len(children) < int(count):
            remaining = int(count) - len(children)
            proposals = self.utils.create_children(
                self.population,
                count=max(remaining, 16),
                evaluate=False,
            )
            children.extend(self._filter_runtime_valid(proposals)[:remaining])
            attempts += 1
            if attempts > 1_000:
                raise RuntimeError("unable to create runtime-valid NSGA-II offspring")
        return children

    def _rank_and_crowd(self, population: Population) -> None:
        self.utils.fast_nondominated_sort(population)
        for front in population.fronts:
            self.utils.calculate_crowding_distance(front)

    def _evaluate_individuals(self, individuals: list[Any]) -> None:
        states = [list(individual.features) for individual in individuals]
        objectives = self.batch_evaluator.evaluate_states(states)
        if len(objectives) != len(individuals):
            raise RuntimeError(
                f"batch evaluator returned {len(objectives)} objectives for {len(individuals)} individuals"
            )
        for individual, values in zip(individuals, objectives, strict=True):
            individual.objectives = [float(value) for value in values]

    def _select_next_population(self, children: list[Any]) -> None:
        assert self.population is not None
        self.population.extend(children)
        self.utils.fast_nondominated_sort(self.population)
        new_population = Population()
        front_num = 0
        while (
            front_num < len(self.population.fronts)
            and len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals
        ):
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            new_population.extend(self.population.fronts[front_num])
            front_num += 1
        if len(new_population) < self.num_of_individuals and front_num < len(self.population.fronts):
            front = self.population.fronts[front_num]
            self.utils.calculate_crowding_distance(front)
            front.sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(front[: self.num_of_individuals - len(new_population)])
        if len(new_population) != self.num_of_individuals:
            raise RuntimeError(
                f"NSGA-II selection produced {len(new_population)} individuals, expected {self.num_of_individuals}"
            )
        self.population = new_population
        self._rank_and_crowd(self.population)

    def _generation_record(self, generation: int, evaluated_new: int) -> dict[str, Any]:
        assert self.population is not None
        pareto_front = self.population.fronts[0] if self.population.fronts else []
        if pareto_front:
            best_pdp = min(individual.objectives[0] for individual in pareto_front)
            best_top1 = min(individual.objectives[1] for individual in pareto_front)
            best_area = min(individual.objectives[2] for individual in pareto_front)
            mean_pdp = sum(individual.objectives[0] for individual in pareto_front) / len(pareto_front)
            mean_top1 = sum(individual.objectives[1] for individual in pareto_front) / len(pareto_front)
            mean_area = sum(individual.objectives[2] for individual in pareto_front) / len(pareto_front)
        else:
            best_pdp = None
            best_top1 = None
            best_area = None
            mean_pdp = None
            mean_top1 = None
            mean_area = None
        return {
            "generation": int(generation),
            "population_size": len(self.population),
            "pareto_count": len(pareto_front),
            "best_pdp": best_pdp,
            "best_top1drop": best_top1,
            "best_area_um2": best_area,
            "mean_pdp": mean_pdp,
            "mean_top1drop": mean_top1,
            "mean_area_um2": mean_area,
            "evaluated_new": int(evaluated_new),
            "total_new_evaluations": int(self.batch_evaluator.new_evaluation_count),
            "total_cached_evaluations": int(len(self.batch_evaluator.evaluator.cache)),
        }

    def evolve(self) -> list[Any]:
        print("\n" + "=" * 70)
        print("Starting NSGA-II optimization with generation-level batch evaluation...")
        print(
            f"Population size: {self.num_of_individuals}, "
            f"Generations: {self.num_of_generations}"
        )
        print("=" * 70)

        self.population = self._create_runtime_valid_initial_population()
        before_initial = self.batch_evaluator.new_evaluation_count
        self._evaluate_individuals(self.population.population)
        initial_new = self.batch_evaluator.new_evaluation_count - before_initial
        self._rank_and_crowd(self.population)
        print(f"Initial population evaluated: {initial_new} new unique configurations")

        for generation in range(1, self.num_of_generations + 1):
            remaining = self.batch_evaluator.remaining_new_evaluations()
            if remaining is not None and remaining <= 0:
                break
            child_count = self.num_of_individuals if remaining is None else min(self.num_of_individuals, remaining)
            children = self._create_runtime_valid_children(count=child_count)
            before_children = self.batch_evaluator.new_evaluation_count
            self._evaluate_individuals(children)
            evaluated_new = self.batch_evaluator.new_evaluation_count - before_children
            self._select_next_population(children)
            record = self._generation_record(generation, evaluated_new)
            self.generation_history.append(record)
            if generation == 1 or generation % 10 == 0 or remaining is not None and self.batch_evaluator.remaining_new_evaluations() == 0:
                print(
                    "Generation {}/{}: Pareto={} new={} total_new={}".format(
                        generation,
                        self.num_of_generations,
                        record["pareto_count"],
                        record["evaluated_new"],
                        record["total_new_evaluations"],
                    )
                )

        assert self.population is not None
        return self.population.fronts[0] if self.population.fronts else []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_STRICT50_OUTPUT_DIR.parent / f"{DEFAULT_STRICT50_RUN_NAME}_area3obj",
    )
    parser.add_argument(
        "--seed-cache",
        type=Path,
        default=None,
        help="Import a compatible area-backfilled evaluation CSV into a fresh three-objective run directory.",
    )
    parser.add_argument(
        "--top1-eval-method",
        default="strict50_v3",
        help=(
            "top1_drop evaluator: strict50_v3/50shot or true1000/1000shot. "
            "strict50_v3 is the canonical Swin linear48 joint4497 package."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for one shared evaluation pool, e.g. cuda:0,cuda:2,cuda:3.",
    )
    parser.add_argument(
        "--eval-device-weights",
        default=None,
        help="Positive integer evaluation shares in --devices order, e.g. 2,1 for a faster first GPU. Default is equal shares.",
    )
    parser.add_argument(
        "--preflight-device",
        default="auto",
        help="Device for fast runtime-candidate screening; auto uses the first evaluation GPU, or CPU when CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--tour-participants", type=int, default=2)
    parser.add_argument("--tournament-prob", type=float, default=0.9)
    parser.add_argument("--crossover-param", type=float, default=2.0)
    parser.add_argument("--mutation-param", type=float, default=5.0)
    parser.add_argument(
        "--max-new-evals",
        type=int,
        default=None,
        help="Hard cap on new unique configurations evaluated after this run starts.",
    )
    parser.add_argument("--batch-log-every", type=int, default=0)
    parser.add_argument("--plot-all", action="store_true", help="Plot all unique evaluations in addition to the Pareto front.")
    parser.add_argument(
        "--warm-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Evaluate the built-in structured seed patterns before NSGA-II. Disabled by default for from-zero runs.",
    )
    parser.add_argument(
        "--allow-existing-cache",
        action="store_true",
        help="Explicitly reuse this output directory's evaluation_cache_live.csv instead of requiring a fresh run directory.",
    )
    parser.add_argument("--true1000-model-key", default=DEFAULT_TRUE1000_MODEL_KEY)
    parser.add_argument("--true1000-sample-count", type=int, default=DEFAULT_TRUE1000_SAMPLE_COUNT)
    parser.add_argument("--true1000-sample-seed", type=int, default=DEFAULT_TRUE1000_SAMPLE_SEED)
    parser.add_argument("--true1000-batch-size", type=int, default=DEFAULT_TRUE1000_BATCH_SIZE)
    parser.add_argument("--true1000-sessions-per-gpu", type=int, default=DEFAULT_TRUE1000_SESSIONS_PER_GPU)
    parser.add_argument(
        "--true1000-session-profile",
        choices=("strict", "throughput"),
        default=DEFAULT_TRUE1000_SESSION_PROFILE,
        help="ORT/CUDA runtime profile for true1000 approximate injection.",
    )
    parser.add_argument("--true1000-baseline-cache-dir", type=Path, default=None)
    parser.add_argument("--true1000-archive-baseline-dir", type=Path, default=None)
    parser.add_argument("--true1000-batch-fallback-parallelism", type=int, default=1)
    parser.add_argument("--baseline-top1-acc", type=float, default=0.80)
    parser.add_argument("--curve-acc-high", type=float, default=0.79, help="Highest target top1 accuracy for the PDP curve.")
    parser.add_argument("--curve-acc-low", type=float, default=0.30, help="Lowest target top1 accuracy for the PDP curve.")
    parser.add_argument("--curve-acc-step", type=float, default=0.01, help="Top1 accuracy step for the PDP curve.")
    return parser


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def warm_start_token_sets() -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []
    for token in range(LOW_TOKEN_MIN, LOW_TOKEN_MAX + 1):
        patterns.append((token,) * LOW_TOKEN_COUNT)
    for token in (3, 4, 8):
        for count in range(LOW_TOKEN_COUNT + 1):
            patterns.append((token,) * count + (1,) * (LOW_TOKEN_COUNT - count))
            patterns.append((1,) * (LOW_TOKEN_COUNT - count) + (token,) * count)
    for token in (2, 3, 4, 8):
        for pos in range(LOW_TOKEN_COUNT):
            arr = [1] * LOW_TOKEN_COUNT
            arr[pos] = token
            patterns.append(tuple(arr))
    for token in (3, 4, 8):
        for pos1 in range(LOW_TOKEN_COUNT):
            for pos2 in range(pos1 + 1, LOW_TOKEN_COUNT):
                arr = [1] * LOW_TOKEN_COUNT
                arr[pos1] = token
                arr[pos2] = token
                patterns.append(tuple(arr))
    seen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    for tokens in patterns:
        if tokens in seen:
            continue
        seen.add(tokens)
        ordered.append(tokens)
    return ordered


def evaluate_warm_start(
    evaluator: OptimizationEvaluator,
    *,
    batch_evaluator: BatchedCandidateEvaluator | None = None,
) -> int:
    tokens_list = warm_start_token_sets()
    if batch_evaluator is None:
        for tokens in tokens_list:
            evaluator.evaluate(CandidateConfig(low_tokens=tokens, high_tokens=FIXED_HIGH_TOKENS, approx_high_part=False))
    else:
        batch_evaluator.evaluate_states([list(tokens) for tokens in tokens_list])
    return len(tokens_list)


def individual_to_row(individual, *, design_id: str) -> dict[str, Any]:
    row = {
        "design_id": design_id,
        "pdp_fj": float(individual.objectives[0]),
        "top1_drop": float(individual.objectives[1]),
        "area_um2": float(individual.objectives[2]),
        "rank": int(individual.rank) if individual.rank is not None else None,
        "crowding_distance": float(individual.crowding_distance) if individual.crowding_distance is not None else None,
    }
    for idx, token in enumerate(individual.features):
        row[f"low_{idx}"] = int(token)
    for idx, token in enumerate(FIXED_HIGH_TOKENS):
        row[f"high_{idx}"] = int(token)
    return row


def assign_design_ids(frame: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.sort_values(["pdp_fj", "top1_drop", "area_um2", "config_hash"]).reset_index(drop=True)
    frame = frame.copy()
    frame["design_id"] = [f"{prefix}_{idx:06d}" for idx in range(len(frame))]
    return frame


def build_global_pareto_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the exact three-objective minimization front in O(n log n) time.

    Sorting PDP first makes each prior row no worse in PDP. A Fenwick prefix
    minimum over top1_drop then finds the smallest area among prior rows that
    are also no worse in top1_drop. Equal objective vectors remain on the
    front because neither dominates the other; config hashes are unique.
    """
    if frame.empty:
        return frame.copy()
    required = {"pdp_fj", "top1_drop", "area_um2", "config_hash"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Cannot build Pareto front; missing columns: {sorted(missing)}")

    work = frame.drop_duplicates(subset=["config_hash"], keep="first").copy()
    objectives = ("pdp_fj", "top1_drop", "area_um2")
    work[list(objectives)] = work[list(objectives)].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(work[list(objectives)].to_numpy(dtype=float)).all():
        raise ValueError("Cannot build Pareto front; PDP, top1_drop and area_um2 must be finite")
    work = work.sort_values([*objectives, "config_hash"]).reset_index(drop=True)

    selected = nondominated_mask_3d(work[list(objectives)].to_numpy(dtype=float))
    pareto = work.loc[selected].reset_index(drop=True).copy()
    pareto["rank"] = 0
    pareto["crowding_distance"] = np.nan
    return pareto


def plot_pareto_front(
    *,
    cache_frame: pd.DataFrame,
    pareto_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    plot_all: bool,
) -> None:
    plt.figure(figsize=(10, 8))
    if plot_all and not cache_frame.empty:
        plt.scatter(cache_frame["pdp_fj"], cache_frame["top1_drop"], alpha=0.22, s=18, color="#8a8a8a", label="unique evaluations")
    if not pareto_frame.empty:
        points = plt.scatter(
            pareto_frame["pdp_fj"], pareto_frame["top1_drop"],
            c=pareto_frame["area_um2"], cmap="viridis", alpha=0.9, s=38,
            edgecolor="black", linewidth=0.5, label="3-objective Pareto front",
        )
        plt.colorbar(points, label="GNN-predicted 8-bit multiplier area (µm²)")
    plt.xlabel("PDP (fJ)")
    plt.ylabel("top1drop")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def target_accuracy_values(*, high: float, low: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"curve step must be positive, got {step}")
    if high < low:
        raise ValueError(f"curve high accuracy {high} is lower than low accuracy {low}")
    values: list[float] = []
    current = float(high)
    epsilon = step / 100.0
    while current >= float(low) - epsilon:
        values.append(round(current, 10))
        current -= float(step)
    return values


def build_accuracy_pdp_curve(
    frame: pd.DataFrame,
    *,
    baseline_top1_acc: float,
    target_acc_values: list[float],
) -> pd.DataFrame:
    columns = [
        "curve_id",
        "target_top1_acc",
        "target_top1_acc_pct",
        "target_top1_drop",
        "selected",
        "design_id",
        "config_hash",
        "pdp_fj",
        "area_um2",
        "pred_top1_drop",
        "clipped_top1_drop",
        "pred_top1_acc",
        "clipped_top1_acc",
        "accuracy_slack",
        "delay_ns",
        "power_uw",
        "top5_drop",
        "final_truth_mae",
        "final_truth_max_abs",
        *[f"low_{idx}" for idx in range(LOW_TOKEN_COUNT)],
        *[f"high_{idx}" for idx in range(len(FIXED_HIGH_TOKENS))],
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    work = frame.copy()
    work["pred_top1_drop"] = work["top1_drop"].astype(float)
    work["clipped_top1_drop"] = work["pred_top1_drop"].clip(lower=0.0)
    work["pred_top1_acc"] = float(baseline_top1_acc) - work["pred_top1_drop"]
    work["clipped_top1_acc"] = (float(baseline_top1_acc) - work["clipped_top1_drop"]).clip(lower=0.0, upper=float(baseline_top1_acc))

    rows: list[dict[str, Any]] = []
    for idx, target_acc in enumerate(target_acc_values):
        target_drop = float(baseline_top1_acc) - float(target_acc)
        feasible = work[work["clipped_top1_drop"] <= target_drop + 1e-12].copy()
        if feasible.empty:
            rows.append(
                {
                    "curve_id": f"acc_{int(round(target_acc * 100)):03d}",
                    "target_top1_acc": float(target_acc),
                    "target_top1_acc_pct": float(target_acc * 100.0),
                    "target_top1_drop": float(target_drop),
                    "selected": False,
                }
            )
            continue

        feasible = feasible.sort_values(["pdp_fj", "clipped_top1_drop", "area_um2", "config_hash"]).reset_index(drop=True)
        best = feasible.iloc[0].to_dict()
        row = {
            "curve_id": f"acc_{int(round(target_acc * 100)):03d}",
            "target_top1_acc": float(target_acc),
            "target_top1_acc_pct": float(target_acc * 100.0),
            "target_top1_drop": float(target_drop),
            "selected": True,
            "design_id": best.get("design_id"),
            "config_hash": best.get("config_hash"),
            "pdp_fj": float(best["pdp_fj"]),
            "area_um2": float(best["area_um2"]),
            "pred_top1_drop": float(best["pred_top1_drop"]),
            "clipped_top1_drop": float(best["clipped_top1_drop"]),
            "pred_top1_acc": float(best["pred_top1_acc"]),
            "clipped_top1_acc": float(best["clipped_top1_acc"]),
            "accuracy_slack": float(best["clipped_top1_acc"] - target_acc),
            "delay_ns": float(best["delay_ns"]),
            "power_uw": float(best["power_uw"]),
            "top5_drop": float(best["top5_drop"]),
            "final_truth_mae": float(best["final_truth_mae"]),
            "final_truth_max_abs": float(best["final_truth_max_abs"]),
        }
        for token_idx in range(LOW_TOKEN_COUNT):
            row[f"low_{token_idx}"] = int(best[f"low_{token_idx}"])
        for token_idx in range(len(FIXED_HIGH_TOKENS)):
            row[f"high_{token_idx}"] = int(best[f"high_{token_idx}"])
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def build_accuracy_bin_pdp_curve(
    frame: pd.DataFrame,
    *,
    baseline_top1_acc: float,
    target_acc_values: list[float],
    step: float,
) -> pd.DataFrame:
    columns = [
        "curve_id",
        "target_top1_acc",
        "target_top1_acc_pct",
        "bin_low_top1_acc",
        "bin_high_top1_acc",
        "selected",
        "design_id",
        "config_hash",
        "pdp_fj",
        "area_um2",
        "pred_top1_drop",
        "clipped_top1_drop",
        "pred_top1_acc",
        "clipped_top1_acc",
        "bin_slack_low",
        "bin_slack_high",
        "delay_ns",
        "power_uw",
        "top5_drop",
        "final_truth_mae",
        "final_truth_max_abs",
        *[f"low_{idx}" for idx in range(LOW_TOKEN_COUNT)],
        *[f"high_{idx}" for idx in range(len(FIXED_HIGH_TOKENS))],
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    work = frame.copy()
    work["pred_top1_drop"] = work["top1_drop"].astype(float)
    work["clipped_top1_drop"] = work["pred_top1_drop"].clip(lower=0.0)
    work["pred_top1_acc"] = float(baseline_top1_acc) - work["pred_top1_drop"]
    work["clipped_top1_acc"] = (float(baseline_top1_acc) - work["clipped_top1_drop"]).clip(lower=0.0, upper=float(baseline_top1_acc))

    rows: list[dict[str, Any]] = []
    for target_acc in target_acc_values:
        bin_low = float(target_acc)
        bin_high = min(float(target_acc) + float(step), float(baseline_top1_acc))
        feasible = work[(work["clipped_top1_acc"] >= bin_low - 1e-12) & (work["clipped_top1_acc"] < bin_high - 1e-12)].copy()
        if feasible.empty and math.isclose(bin_high, float(baseline_top1_acc)):
            feasible = work[(work["clipped_top1_acc"] >= bin_low - 1e-12) & (work["clipped_top1_acc"] <= bin_high + 1e-12)].copy()
        if feasible.empty:
            rows.append(
                {
                    "curve_id": f"acc_{int(round(target_acc * 100)):03d}",
                    "target_top1_acc": float(target_acc),
                    "target_top1_acc_pct": float(target_acc * 100.0),
                    "bin_low_top1_acc": float(bin_low),
                    "bin_high_top1_acc": float(bin_high),
                    "selected": False,
                }
            )
            continue

        feasible = feasible.sort_values(["pdp_fj", "clipped_top1_drop", "area_um2", "config_hash"]).reset_index(drop=True)
        best = feasible.iloc[0].to_dict()
        row = {
            "curve_id": f"acc_{int(round(target_acc * 100)):03d}",
            "target_top1_acc": float(target_acc),
            "target_top1_acc_pct": float(target_acc * 100.0),
            "bin_low_top1_acc": float(bin_low),
            "bin_high_top1_acc": float(bin_high),
            "selected": True,
            "design_id": best.get("design_id"),
            "config_hash": best.get("config_hash"),
            "pdp_fj": float(best["pdp_fj"]),
            "area_um2": float(best["area_um2"]),
            "pred_top1_drop": float(best["pred_top1_drop"]),
            "clipped_top1_drop": float(best["clipped_top1_drop"]),
            "pred_top1_acc": float(best["pred_top1_acc"]),
            "clipped_top1_acc": float(best["clipped_top1_acc"]),
            "bin_slack_low": float(best["clipped_top1_acc"] - bin_low),
            "bin_slack_high": float(bin_high - best["clipped_top1_acc"]),
            "delay_ns": float(best["delay_ns"]),
            "power_uw": float(best["power_uw"]),
            "top5_drop": float(best["top5_drop"]),
            "final_truth_mae": float(best["final_truth_mae"]),
            "final_truth_max_abs": float(best["final_truth_max_abs"]),
        }
        for token_idx in range(LOW_TOKEN_COUNT):
            row[f"low_{token_idx}"] = int(best[f"low_{token_idx}"])
        for token_idx in range(len(FIXED_HIGH_TOKENS)):
            row[f"high_{token_idx}"] = int(best[f"high_{token_idx}"])
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def plot_accuracy_pdp_curve(curve_frame: pd.DataFrame, *, output_path: Path, title: str) -> None:
    selected = curve_frame[curve_frame["selected"].eq(True)].copy() if "selected" in curve_frame else pd.DataFrame()
    plt.figure(figsize=(10, 7))
    if not selected.empty:
        y_col = "clipped_top1_acc" if "clipped_top1_acc" in selected.columns else "target_top1_acc"
        plt.plot(selected["pdp_fj"], selected[y_col] * 100.0, marker="o", linewidth=1.6, markersize=4.5, color="#1f77b4")
    plt.xlabel("PDP (fJ)")
    plt.ylabel("target top1 accuracy (%)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    top1_eval_method = normalize_top1_eval_method(str(args.top1_eval_method))
    devices = parse_devices(args.device, args.devices)
    device_weights = parse_device_weights(args.eval_device_weights, len(devices))
    use_parallel = bool(args.devices) and len(devices) > 1
    if not use_parallel and args.eval_device_weights is not None:
        raise ValueError("--eval-device-weights requires multi-device --devices")
    if args.max_new_evals is not None and int(args.max_new_evals) <= 0:
        raise ValueError("--max-new-evals must be positive when provided")
    seed_everything(int(args.seed))
    output_dir = args.output_dir.resolve()
    seed_results: list[EvaluationResult] = []
    seed_cache_metadata: dict[str, Any] | None = None
    if args.seed_cache is not None:
        require_fresh_seed_destination(output_dir, allow_existing_cache=bool(args.allow_existing_cache))
        seed_results, seed_cache_metadata = validate_seed_cache(
            Path(args.seed_cache),
            top1_eval_method=top1_eval_method,
            true1000_model_key=str(args.true1000_model_key),
            true1000_sample_count=int(args.true1000_sample_count),
            true1000_sample_seed=int(args.true1000_sample_seed),
            true1000_session_profile=str(args.true1000_session_profile),
        )
    prepare_output_dir(output_dir, allow_existing_cache=bool(args.allow_existing_cache))
    if top1_eval_method == "strict50_v3":
        run_name = f"{DEFAULT_STRICT50_RUN_NAME}_area3obj"
    elif top1_eval_method == TRUE1000_TOP1_METHOD:
        run_name = "nsga2_amcfg_pdp_top1drop_true1000_uniform_runtime_area3obj_v1"
    else:
        run_name = f"{DEFAULT_RUN_NAME}_area3obj"
    evaluator = OptimizationEvaluator(
        top1_eval_method=top1_eval_method,
        cache_csv=output_dir / "evaluation_cache_live.csv",
        device=devices[0],
        true1000_model_key=str(args.true1000_model_key),
        true1000_sample_count=int(args.true1000_sample_count),
        true1000_sample_seed=int(args.true1000_sample_seed),
        true1000_batch_size=int(args.true1000_batch_size),
        true1000_sessions_per_gpu=int(args.true1000_sessions_per_gpu),
        true1000_session_profile=str(args.true1000_session_profile),
        true1000_baseline_cache_dir=args.true1000_baseline_cache_dir,
        true1000_archive_baseline_dir=args.true1000_archive_baseline_dir,
        true1000_batch_fallback_parallelism=int(args.true1000_batch_fallback_parallelism),
        initialize_models=not use_parallel,
    )
    if seed_cache_metadata is not None:
        install_seed_results(evaluator, seed_results, seed_cache_metadata)
    runtime_candidate_filter: CanonicalRuntimeCandidateFilter | None = None
    if top1_eval_method in {"strict50_v3", TRUE1000_TOP1_METHOD}:
        # The immutable runtime asset accepts only residuals inside this range.
        # Preflight and runtime evaluation are sequential within a generation.
        preflight_device = resolve_preflight_device(str(args.preflight_device), devices[0])
        preflight_truth_evaluator = DMultTruthTableEvaluator(device=preflight_device)
        runtime_candidate_filter = CanonicalRuntimeCandidateFilter(
            predict_max_abs_residuals=preflight_truth_evaluator.predict_max_abs_residuals,
            device=preflight_device,
        )
    start_evaluation_count = int(len(evaluator.cache))
    eval_pool: ParallelEvaluationPool | None = None
    try:
        if use_parallel:
            eval_pool = ParallelEvaluationPool(
                devices=devices,
                device_weights=device_weights,
                python_executable=sys.executable,
                cwd=Path(__file__).resolve().parents[2],
                top1_eval_method=top1_eval_method,
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
        batch_evaluator = BatchedCandidateEvaluator(
            evaluator=evaluator,
            pool=eval_pool,
            start_evaluation_count=start_evaluation_count,
            max_new_evals=args.max_new_evals,
            log_every=int(args.batch_log_every),
        )
        warm_start_count = (
            evaluate_warm_start(evaluator, batch_evaluator=batch_evaluator) if bool(args.warm_start) else 0
        )
        objective_pdp, objective_top1drop, objective_area = make_objective_functions(evaluator)

        variables_type = ["int"] * LOW_TOKEN_COUNT
        variables_range = [(LOW_TOKEN_MIN, LOW_TOKEN_MAX)]
        prob = Problem(
            num_of_variables=LOW_TOKEN_COUNT,
            objectives=[objective_pdp, objective_top1drop, objective_area],
            variables_range=variables_range,
            variables_type=variables_type,
            expand=False,
            same_range=True,
        )

        evolution = BatchedEvolution(
            problem=prob,
            batch_evaluator=batch_evaluator,
            num_of_generations=int(args.generations),
            num_of_individuals=int(args.population),
            num_of_tour_particips=int(args.tour_participants),
            tournament_prob=float(args.tournament_prob),
            crossover_param=float(args.crossover_param),
            mutation_param=float(args.mutation_param),
            runtime_candidate_filter=runtime_candidate_filter,
        )
        pareto_front = evolution.evolve()
    finally:
        if eval_pool is not None:
            eval_pool.close()
        evaluator.close()

    cache_frame = evaluator.cache_frame()

    all_eval_frame = cache_frame.copy()
    if not all_eval_frame.empty:
        all_eval_frame = assign_design_ids(all_eval_frame, prefix="amcfg_eval")

    pareto_frame = build_global_pareto_frame(all_eval_frame)
    if not pareto_frame.empty:
        pareto_frame = assign_design_ids(pareto_frame, prefix="amcfg")
        preferred = [
            "design_id",
            "config_hash",
            "approx_high_part",
            "design_index",
            *[f"low_{idx}" for idx in range(LOW_TOKEN_COUNT)],
            *[f"high_{idx}" for idx in range(len(FIXED_HIGH_TOKENS))],
            "delay_ns",
            "power_uw",
            "pdp_fj",
            "top1_drop",
            "area_um2",
            "top5_drop",
            "final_truth_mae",
            "final_truth_max_abs",
            "rank",
            "crowding_distance",
            "top1_eval_method",
            "model_package_id",
            "selected_images_hash",
            "feature_schema_hash",
            "strict50_feature_count",
            "strict50_selected_image_count",
            "sample50_top1_drop",
            "sample50_top5_drop",
        ]
        pareto_frame = pareto_frame[[column for column in preferred if column in pareto_frame.columns]]

    best_frame = pareto_frame.head(min(10, len(pareto_frame))) if not pareto_frame.empty else pd.DataFrame()
    curve_targets = target_accuracy_values(high=float(args.curve_acc_high), low=float(args.curve_acc_low), step=float(args.curve_acc_step))
    curve_threshold_frame = build_accuracy_pdp_curve(
        all_eval_frame,
        baseline_top1_acc=float(args.baseline_top1_acc),
        target_acc_values=curve_targets,
    )
    curve_frame = build_accuracy_bin_pdp_curve(
        all_eval_frame,
        baseline_top1_acc=float(args.baseline_top1_acc),
        target_acc_values=curve_targets,
        step=float(args.curve_acc_step),
    )
    write_csv(pareto_frame, output_dir / "pareto_front.csv")
    write_csv(all_eval_frame, output_dir / "evaluation_cache.csv")
    write_csv(best_frame, output_dir / "best_solutions.csv")
    write_csv(curve_frame, output_dir / "curve_points.csv")
    write_csv(curve_threshold_frame, output_dir / "curve_threshold_points.csv")
    write_csv(pd.DataFrame(evolution.generation_history), output_dir / "generation_log.csv")

    selected_curve = curve_frame[curve_frame["selected"].eq(True)].copy() if not curve_frame.empty else pd.DataFrame()
    missing_curve_count = int((~curve_frame["selected"].astype(bool)).sum()) if not curve_frame.empty else len(curve_targets)
    selected_threshold_curve = curve_threshold_frame[curve_threshold_frame["selected"].eq(True)].copy() if not curve_threshold_frame.empty else pd.DataFrame()
    missing_threshold_curve_count = int((~curve_threshold_frame["selected"].astype(bool)).sum()) if not curve_threshold_frame.empty else len(curve_targets)
    manifest = {
        "created_at": now_iso(),
        "run_name": run_name,
        "output_dir": str(output_dir),
        "objectives": ["pdp_fj", "top1_drop", "area_um2"],
        "area_metric": "GNN-predicted 8-bit multiplier area (um2)",
        "top1_eval_method": top1_eval_method,
        "device": devices[0],
        "devices": devices,
        "eval_device_weights": list(device_weights),
        "parallel_evaluation": bool(use_parallel),
        "parallel_worker_count": int(len(devices)) if use_parallel else 0,
        "runtime_candidate_preflight": (
            runtime_candidate_filter.summary()
            if runtime_candidate_filter is not None
            else {"enabled": False}
        ),
        "token_range": [LOW_TOKEN_MIN, LOW_TOKEN_MAX],
        "low_token_count": LOW_TOKEN_COUNT,
        "fixed_high_tokens": list(FIXED_HIGH_TOKENS),
        "population": int(args.population),
        "generations": int(args.generations),
        "warm_start": bool(args.warm_start),
        "warm_start_count": int(warm_start_count),
        "evaluation_count": int(len(all_eval_frame)),
        "start_evaluation_count": int(start_evaluation_count),
        "new_evaluation_count": int(batch_evaluator.new_evaluation_count),
        "max_new_evals": args.max_new_evals,
        "evaluation_budget_reached": (
            None
            if args.max_new_evals is None
            else bool(batch_evaluator.new_evaluation_count >= int(args.max_new_evals))
        ),
        "cache_load_report": evaluator.cache_load_report,
        "seed_cache": seed_cache_metadata,
        "pareto_count": int(len(pareto_frame)),
        "pareto_selection_policy": "global_non_dominated_from_all_unique_evaluations",
        "baseline_top1_acc": float(args.baseline_top1_acc),
        "curve_target_acc_high": float(args.curve_acc_high),
        "curve_target_acc_low": float(args.curve_acc_low),
        "curve_target_acc_step": float(args.curve_acc_step),
        "curve_target_count": int(len(curve_targets)),
        "curve_selected_count": int(len(selected_curve)),
        "curve_missing_count": int(missing_curve_count),
        "curve_unique_design_count": int(selected_curve["config_hash"].nunique()) if not selected_curve.empty else 0,
        "curve_selection_policy": "min_pdp_in_1pp_accuracy_bin",
        "threshold_curve_selected_count": int(len(selected_threshold_curve)),
        "threshold_curve_missing_count": int(missing_threshold_curve_count),
        "threshold_curve_unique_design_count": int(selected_threshold_curve["config_hash"].nunique()) if not selected_threshold_curve.empty else 0,
        "threshold_curve_selection_policy": "min_pdp_under_accuracy_threshold",
        "plot_all": bool(args.plot_all),
        "true1000_model_key": str(args.true1000_model_key) if top1_eval_method == TRUE1000_TOP1_METHOD else None,
        "true1000_sample_count": int(args.true1000_sample_count) if top1_eval_method == TRUE1000_TOP1_METHOD else None,
        "true1000_sample_seed": int(args.true1000_sample_seed) if top1_eval_method == TRUE1000_TOP1_METHOD else None,
        "true1000_batch_size": int(args.true1000_batch_size) if top1_eval_method == TRUE1000_TOP1_METHOD else None,
        "true1000_sessions_per_gpu": int(args.true1000_sessions_per_gpu) if top1_eval_method == TRUE1000_TOP1_METHOD else None,
        "true1000_session_profile": str(args.true1000_session_profile) if top1_eval_method == TRUE1000_TOP1_METHOD else None,
    }
    if top1_eval_method == "strict50_v3":
        manifest.update(
            {
                "top1_model_package_dir": str(STRICT50_TOP1_PACKAGE_DIR.resolve()),
                "top1_model_package_id": STRICT50_EXPECTED_PACKAGE_ID,
                "top1_package_hash": STRICT50_EXPECTED_PACKAGE_HASH,
                "top1_selected_image_count": 50,
                "top1_selected_images_hash": STRICT50_EXPECTED_SELECTED_IMAGES_HASH,
                "top1_feature_schema_hash": STRICT50_EXPECTED_FEATURE_SCHEMA_HASH,
                "top1_feature_count": STRICT50_FEATURE_DIM,
                "top1_runtime_base_feature_count": STRICT50_RUNTIME_BASE_FEATURE_DIM,
                "top1_supplemental_feature_count": STRICT50_SUPPLEMENTAL_FEATURE_DIM,
                "dataset_id": STRICT50_DATASET_ID,
                "dataset_schema_version": STRICT50_DATASET_SCHEMA_VERSION,
                "dataset_manifest": str(STRICT50_DATASET_MANIFEST.resolve()),
                "injection_scope": STRICT50_INJECTION_SCOPE,
                "runtime_delta_source": STRICT50_RUNTIME_DELTA_SOURCE,
                "bank_meta": str(STRICT50_BANK_META.resolve()),
                "bank_meta_sha1": STRICT50_EXPECTED_BANK_META_SHA1,
                "node_registry": str(STRICT50_NODE_REGISTRY.resolve()),
                "node_registry_hash": STRICT50_EXPECTED_NODE_REGISTRY_SHA1,
                "selected_node_count": STRICT50_EXPECTED_NODE_COUNT,
            }
        )
    write_json(output_dir / "run_manifest.json", manifest)

    plot_pareto_front(
        cache_frame=cache_frame,
        pareto_frame=pareto_frame,
        output_path=output_dir / "pareto_front.png",
        title=f"NSGA-II Pareto Front - {run_name}",
        plot_all=bool(args.plot_all),
    )
    plot_accuracy_pdp_curve(
        curve_frame,
        output_path=output_dir / "accuracy_pdp_curve.png",
        title=f"1pp Top1 Accuracy Bins vs Min PDP - {run_name}",
    )
    plot_accuracy_pdp_curve(
        curve_threshold_frame,
        output_path=output_dir / "accuracy_threshold_pdp_curve.png",
        title=f"Top1 Accuracy Threshold vs Min PDP - {run_name}",
    )

    summary = {
        "pareto_count": int(len(pareto_frame)),
        "evaluation_count": int(len(all_eval_frame)),
        "start_evaluation_count": int(start_evaluation_count),
        "new_evaluation_count": int(batch_evaluator.new_evaluation_count),
        "max_new_evals": args.max_new_evals,
        "evaluation_budget_reached": (
            None
            if args.max_new_evals is None
            else bool(batch_evaluator.new_evaluation_count >= int(args.max_new_evals))
        ),
        "devices": devices,
        "eval_device_weights": list(device_weights),
        "parallel_evaluation": bool(use_parallel),
        "parallel_worker_count": int(len(devices)) if use_parallel else 0,
        "seed_cache": seed_cache_metadata,
        "runtime_candidate_preflight": (
            runtime_candidate_filter.summary()
            if runtime_candidate_filter is not None
            else {"enabled": False}
        ),
        "best_pdp": float(pareto_frame["pdp_fj"].min()) if not pareto_frame.empty else None,
        "best_top1drop": float(pareto_frame["top1_drop"].min()) if not pareto_frame.empty else None,
        "best_area_um2": float(pareto_frame["area_um2"].min()) if not pareto_frame.empty else None,
        "best_solution_count": int(len(best_frame)),
        "pareto_selection_policy": "global_non_dominated_from_all_unique_evaluations",
        "baseline_top1_acc": float(args.baseline_top1_acc),
        "curve_target_count": int(len(curve_targets)),
        "curve_selected_count": int(len(selected_curve)),
        "curve_missing_count": int(missing_curve_count),
        "curve_unique_design_count": int(selected_curve["config_hash"].nunique()) if not selected_curve.empty else 0,
        "curve_min_target_top1_acc": float(selected_curve["target_top1_acc"].min()) if not selected_curve.empty else None,
        "curve_max_target_top1_acc": float(selected_curve["target_top1_acc"].max()) if not selected_curve.empty else None,
        "curve_selection_policy": "min_pdp_in_1pp_accuracy_bin",
        "threshold_curve_selected_count": int(len(selected_threshold_curve)),
        "threshold_curve_missing_count": int(missing_threshold_curve_count),
        "threshold_curve_unique_design_count": int(selected_threshold_curve["config_hash"].nunique()) if not selected_threshold_curve.empty else 0,
        "top1_eval_method": top1_eval_method,
        "true1000_session_profile": (
            str(args.true1000_session_profile)
            if top1_eval_method == TRUE1000_TOP1_METHOD
            else None
        ),
    }
    if top1_eval_method == "strict50_v3":
        summary.update(
            {
                "model_package_id": STRICT50_EXPECTED_PACKAGE_ID,
                "package_hash": STRICT50_EXPECTED_PACKAGE_HASH,
                "dataset_id": STRICT50_DATASET_ID,
                "feature_count": STRICT50_FEATURE_DIM,
                "selected_node_count": STRICT50_EXPECTED_NODE_COUNT,
            }
        )
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
