#!/usr/bin/env python3
"""Run the legacy non-macro PPA flow for configurations listed in a CSV table."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from aaae.paths import OUTPUT_ROOT, PPA_ROOT as EXTERNAL_PPA_ROOT


ROOT = Path(__file__).resolve().parent
FLOW_ROOT = ROOT / "flow"
PPA_ROOT = FLOW_ROOT / "PPA"
RESOURCE_ROOT = EXTERNAL_PPA_ROOT / "data_no_macro" / "PPA"
SOURCE_CSV_PATH = OUTPUT_ROOT / "optimization" / "candidates.csv"
CSV_PATH = OUTPUT_ROOT / "hardware" / "verified_candidates.csv"
RUN_ROOT = OUTPUT_ROOT / "hardware" / "eda"
RTL_ROOT = RUN_ROOT / "generated_rtl"
WORK_ROOT = RUN_ROOT / "work"
LOG_ROOT = RUN_ROOT / "logs"
STATE_PATH = RUN_ROOT / "status.json"
RESULTS_PATH = RUN_ROOT / "measured_results.csv"
LOCK_PATH = RUN_ROOT / "run.lock"
STOP_EVENT = threading.Event()

COMPRESSORS = ["AP1", "kong2", "antonio", "momeni", "ha", "akbar1", "akbar2", "sabetz"]
STAGE1_SITES = ["AP1 AP1_pp_5", "AP1 AP1_pp_6", "AP1 AP1_pp_7_1", "AP1 AP1_pp_7_2"]
STAGE2_SITES = ["AP1 AP1_pp_3", "AP1 AP1_pp_4", "AP1 AP1_pp_5", "AP1 AP1_pp_6", "AP1 AP1_pp_7"]
STAGE1_SUFFIXES = [" AP1_pp_5", " AP1_pp_6", " AP1_pp_7_1", " AP1_pp_7_2"]
STAGE2_SUFFIXES = [" AP1_pp_3", " AP1_pp_4", " AP1_pp_5", " AP1_pp_6", " AP1_pp_7"]

MEASURED_FIELDS = [
    "nonmacro_measured_power_uw",
    "nonmacro_measured_delay_ns",
    "nonmacro_measured_area_um2",
    "nonmacro_measured_pdp_fj",
]

WORK_FILES = [
    "filelist.f",
    "Testbench.v",
    "ppa.py",
    "synth.ys",
    "timing.tcl",
    "power.tcl",
    "Makefile",
    "pthread_yield_fix.o",
]

ERROR_PATTERNS = (
    re.compile(r"^ERROR:", re.MULTILINE),
    re.compile(r"^Error:", re.MULTILINE),
    re.compile(r"^Fatal:", re.MULTILINE),
)


class FlowError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_status(**values: object) -> None:
    current: dict[str, object] = {}
    if STATE_PATH.exists():
        try:
            current = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            current = {}
    current.update(values)
    current["updated_at"] = utc_now()
    atomic_write_text(STATE_PATH, json.dumps(current, indent=2, sort_keys=True) + "\n")


def initialize_table() -> None:
    if CSV_PATH.exists():
        return
    if not SOURCE_CSV_PATH.is_file():
        raise FlowError(f"source CSV is missing: {SOURCE_CSV_PATH}")

    with SOURCE_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FlowError(f"source CSV has no header: {SOURCE_CSV_PATH}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    for field in MEASURED_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
        for row in rows:
            row[field] = ""

    fd, tmp_name = tempfile.mkstemp(prefix=f".{CSV_PATH.name}.", dir=CSV_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, CSV_PATH)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_table() -> tuple[list[str], list[dict[str, str]]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise FlowError(f"CSV has no header: {CSV_PATH}")
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    required = {
        "config_hash",
        "compressor_config",
        *(f"low_{index}" for index in range(9)),
        *MEASURED_FIELDS,
    }
    missing = sorted(required.difference(fieldnames))
    if missing:
        raise FlowError(f"CSV is missing required columns: {', '.join(missing)}")
    if not rows:
        raise FlowError("CSV contains no configurations")
    return fieldnames, rows


def vector_for_row(row: dict[str, str]) -> tuple[int, ...]:
    try:
        one_based = tuple(int(row[f"low_{index}"]) for index in range(9))
    except (KeyError, ValueError) as exc:
        raise FlowError(f"invalid low_0..low_8 values for {row.get('config_hash', '<unknown>')}") from exc
    if any(value < 1 or value > 8 for value in one_based):
        raise FlowError(f"compressor indices must be in 1..8, got {one_based}")

    expected_names = [COMPRESSORS[value - 1] for value in one_based]
    actual_names = [value.strip() for value in row["compressor_config"].split(",")]
    if actual_names != expected_names:
        raise FlowError(
            f"compressor mapping mismatch for {row['config_hash']}: "
            f"indices imply {expected_names}, CSV contains {actual_names}"
        )
    return tuple(value - 1 for value in one_based)


def vector_name(vector: tuple[int, ...]) -> str:
    return "_".join(str(value) for value in vector)


def validate_table(rows: list[dict[str, str]]) -> dict[tuple[int, ...], list[int]]:
    vector_rows: dict[tuple[int, ...], list[int]] = {}
    hash_vectors: dict[str, tuple[int, ...]] = {}
    for row_index, row in enumerate(rows):
        vector = vector_for_row(row)
        vector_rows.setdefault(vector, []).append(row_index)
        config_hash = row["config_hash"]
        previous = hash_vectors.setdefault(config_hash, vector)
        if previous != vector:
            raise FlowError(f"config_hash {config_hash} maps to multiple vectors")

        present = [bool(row[field].strip()) for field in MEASURED_FIELDS]
        if any(present) and not all(present):
            raise FlowError(f"row {row_index + 2} has a partial measured PPA result")

    for vector, indices in vector_rows.items():
        existing = {
            tuple(rows[index][field] for field in MEASURED_FIELDS)
            for index in indices
            if all(rows[index][field].strip() for field in MEASURED_FIELDS)
        }
        if len(existing) > 1:
            raise FlowError(f"duplicate vector {vector_name(vector)} has inconsistent measured results")
    return vector_rows


def write_table(fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{CSV_PATH.name}.", dir=CSV_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, CSV_PATH)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def ensure_backup() -> None:
    backup = RUN_ROOT / f"{CSV_PATH.name}.original"
    if not backup.exists():
        shutil.copy2(CSV_PATH, backup)


def generate_rtl(vector: tuple[int, ...]) -> Path:
    output_dir = RTL_ROOT / vector_name(vector)
    stage1_path = output_dir / "STAGE1.v"
    stage2_path = output_dir / "STAGE2.v"
    if stage1_path.exists() and stage2_path.exists():
        return output_dir

    stage1 = (FLOW_ROOT / "STAGE1.v").read_text(encoding="utf-8")
    stage2 = (FLOW_ROOT / "STAGE2.v").read_text(encoding="utf-8")
    for index, compressor_index in enumerate(vector):
        compressor = COMPRESSORS[compressor_index]
        if index < 4:
            old = STAGE1_SITES[index]
            new = compressor + STAGE1_SUFFIXES[index]
            if stage1.count(old) != 1:
                raise FlowError(f"expected exactly one STAGE1 site {old!r}")
            stage1 = stage1.replace(old, new)
        else:
            site_index = index - 4
            old = STAGE2_SITES[site_index]
            new = compressor + STAGE2_SUFFIXES[site_index]
            if stage2.count(old) != 1:
                raise FlowError(f"expected exactly one STAGE2 site {old!r}")
            stage2 = stage2.replace(old, new)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(stage1_path, stage1)
    atomic_write_text(stage2_path, stage2)
    return output_dir


def prepare_work_dir(rtl_dir: Path, work_root: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    for filename in WORK_FILES:
        source = PPA_ROOT / filename
        if not source.is_file():
            source = RESOURCE_ROOT / filename
        if not source.is_file():
            raise FlowError(f"required PPA flow file is missing: {source}")
        shutil.copy2(source, work_root / filename)

    replacements = {
        "@AAAE_FLOW_ROOT@": str(FLOW_ROOT),
        "@AAAE_GENERATED_RTL@": str(rtl_dir),
        "@AAAE_PPA_RESOURCES@": str(RESOURCE_ROOT),
    }
    for filename in ("synth.ys", "timing.tcl", "power.tcl", "filelist.f", "Makefile"):
        path = work_root / filename
        text = path.read_text(encoding="utf-8")
        for token, value in replacements.items():
            text = text.replace(token, value)
        atomic_write_text(path, text)

    for stale in ("ppa.log", "area.txt", "timing.txt", "power.txt", "synth.v", "timing.sdf", "top.vcd"):
        try:
            (work_root / stale).unlink()
        except FileNotFoundError:
            pass


def run_command(
    name: str,
    command: list[str],
    stdout_path: Path,
    work_root: Path,
    combined: bool = False,
) -> None:
    if STOP_EVENT.is_set():
        raise FlowError(f"{name} cancelled because another parallel job failed")
    stderr_target: int | subprocess.Popen[str]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle:
        stderr_target = subprocess.STDOUT if combined else subprocess.PIPE
        completed = subprocess.run(
            command,
            cwd=work_root,
            env={**os.environ, "PATH": f"/opt/OpenSTA/OpenSTA:{os.environ.get('PATH', '')}"},
            stdout=stdout_handle,
            stderr=stderr_target,
            text=True,
            check=False,
        )
    stderr_text = completed.stderr or ""
    if not combined and stderr_text:
        atomic_write_text(stdout_path.with_suffix(stdout_path.suffix + ".stderr"), stderr_text)
    output_text = stdout_path.read_text(encoding="utf-8", errors="replace") + "\n" + stderr_text
    if completed.returncode != 0:
        raise FlowError(f"{name} failed with exit code {completed.returncode}; see {stdout_path}")
    for pattern in ERROR_PATTERNS:
        match = pattern.search(output_text)
        if match:
            line = output_text[match.start():].splitlines()[0]
            raise FlowError(f"{name} reported an error despite exit code 0: {line}; see {stdout_path}")


def run_flow(vector: tuple[int, ...], config_hash: str) -> tuple[float, float, float, float]:
    name = vector_name(vector)
    case_log_dir = LOG_ROOT / f"{name}_{config_hash}"
    work_root = WORK_ROOT / f"{name}_{config_hash}"
    if case_log_dir.exists():
        shutil.rmtree(case_log_dir)
    if work_root.exists():
        shutil.rmtree(work_root)
    case_log_dir.mkdir(parents=True)

    rtl_dir = generate_rtl(vector)
    prepare_work_dir(rtl_dir, work_root)
    try:
        run_command("yosys", ["yosys", "synth.ys"], work_root / "area.txt", work_root)
        run_command(
            "OpenSTA timing",
            ["opensta", "-exit", "timing.tcl"],
            work_root / "timing.txt",
            work_root,
        )
        run_command(
            "VCS compile",
            ["make", "com"],
            case_log_dir / "make_com.log",
            work_root,
            combined=True,
        )
        run_command(
            "VCS simulation",
            ["make", "sim"],
            case_log_dir / "make_sim.log",
            work_root,
            combined=True,
        )
        run_command(
            "OpenSTA power",
            ["opensta", "-exit", "power.tcl"],
            work_root / "power.txt",
            work_root,
        )
        run_command(
            "PPA extraction",
            [sys.executable, "ppa.py"],
            case_log_dir / "ppa_extract.log",
            work_root,
            combined=True,
        )
    except BaseException:
        STOP_EVENT.set()
        raise

    ppa_lines = (work_root / "ppa.log").read_text(encoding="utf-8").splitlines()
    if len(ppa_lines) != 1:
        raise FlowError(f"expected exactly one PPA result, found {len(ppa_lines)}")
    fields = ppa_lines[0].split()
    if len(fields) != 3:
        raise FlowError(f"invalid PPA result: {ppa_lines[0]!r}")
    delay_ns, area_um2, power_w = (float(value) for value in fields)
    if not (0.0 < delay_ns < 100.0 and 0.0 < area_um2 < 1_000_000.0 and 0.0 < power_w < 1.0):
        raise FlowError(
            f"implausible PPA values: delay={delay_ns}, area={area_um2}, power={power_w}"
        )
    power_uw = power_w * 1_000_000.0
    pdp_fj = power_uw * delay_ns

    for filename in ("area.txt", "timing.txt", "power.txt", "ppa.log", "synth.ys", "filelist.f"):
        source = work_root / filename
        if source.exists():
            shutil.copy2(source, case_log_dir / filename)
    return power_uw, delay_ns, area_um2, pdp_fj


def format_measurement(value: float) -> str:
    return f"{value:.12g}"


def append_result(config_hash: str, vector: tuple[int, ...], values: tuple[float, float, float, float]) -> None:
    new_file = not RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if new_file:
            writer.writerow(["completed_at", "config_hash", "vector", *MEASURED_FIELDS])
        writer.writerow([utc_now(), config_hash, vector_name(vector), *(format_measurement(value) for value in values)])
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    global SOURCE_CSV_PATH, CSV_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV_PATH, help="candidate table to verify")
    parser.add_argument("--output-csv", type=Path, default=CSV_PATH, help="verified candidate table")
    parser.add_argument("--max-new", type=int, default=None, help="stop after this many newly measured unique vectors")
    parser.add_argument("--jobs", type=int, default=1, help="number of configurations to run in parallel")
    parser.add_argument("--reset", action="store_true", help="discard prior non-macro results and rerun all configurations")
    parser.add_argument("--validate-only", action="store_true", help="validate inputs without running EDA tools")
    args = parser.parse_args()
    SOURCE_CSV_PATH = args.source_csv.resolve()
    CSV_PATH = args.output_csv.resolve()
    if args.max_new is not None and args.max_new < 1:
        parser.error("--max-new must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    if args.reset:
        shutil.rmtree(RUN_ROOT, ignore_errors=True)
        try:
            CSV_PATH.unlink()
        except FileNotFoundError:
            pass

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    initialize_table()
    STOP_EVENT.clear()
    lock_handle = LOCK_PATH.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise FlowError(f"another table PPA runner holds {LOCK_PATH}") from exc
    lock_handle.write(f"pid={os.getpid()} started={utc_now()}\n")
    lock_handle.flush()

    fieldnames, rows = load_table()
    vector_rows = validate_table(rows)
    ensure_backup()
    total_unique = len(vector_rows)
    completed_unique = sum(
        all(rows[indices[0]][field].strip() for field in MEASURED_FIELDS)
        for indices in vector_rows.values()
    )
    write_status(
        state="validated" if args.validate_only else "running",
        pid=os.getpid(),
        csv=str(CSV_PATH),
        total_rows=len(rows),
        total_unique=total_unique,
        completed_unique=completed_unique,
        failed_vector=None,
        error=None,
        reason=None,
    )
    print(f"[INFO] rows={len(rows)} unique={total_unique} completed_unique={completed_unique}", flush=True)
    if args.validate_only:
        return 0

    new_count = 0
    pending: list[tuple[tuple[int, ...], list[int], str]] = []
    for vector, indices in vector_rows.items():
        completed_values = None
        for index in indices:
            if all(rows[index][field].strip() for field in MEASURED_FIELDS):
                completed_values = tuple(rows[index][field] for field in MEASURED_FIELDS)
                break
        if completed_values is not None:
            changed = False
            for index in indices:
                if not all(rows[index][field].strip() for field in MEASURED_FIELDS):
                    rows[index].update(dict(zip(MEASURED_FIELDS, completed_values)))
                    changed = True
            if changed:
                write_table(fieldnames, rows)
            continue
        pending.append((vector, indices, rows[indices[0]]["config_hash"]))

    if args.max_new is not None:
        pending = pending[: args.max_new]

    for vector, _, _ in pending:
        generate_rtl(vector)

    task_iter = iter(pending)
    futures: dict[object, tuple[tuple[int, ...], list[int], str]] = {}
    failure: BaseException | None = None
    with ThreadPoolExecutor(max_workers=args.jobs, thread_name_prefix="ppa") as executor:
        for _ in range(min(args.jobs, len(pending))):
            task = next(task_iter, None)
            if task is None:
                break
            vector, _, config_hash = task
            print(f"[START] vector={vector_name(vector)} hash={config_hash}", flush=True)
            futures[executor.submit(run_flow, vector, config_hash)] = task

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                vector, indices, config_hash = futures.pop(future)
                name = vector_name(vector)
                try:
                    values = future.result()
                except BaseException as exc:
                    failure = exc
                    STOP_EVENT.set()
                    write_status(state="failed", failed_vector=name, error=str(exc))
                    break

                formatted = tuple(format_measurement(value) for value in values)
                for index in indices:
                    rows[index].update(dict(zip(MEASURED_FIELDS, formatted)))
                write_table(fieldnames, rows)
                append_result(config_hash, vector, values)
                new_count += 1
                write_status(
                    completed_unique=completed_unique + new_count,
                    last_completed_vector=name,
                    last_completed_hash=config_hash,
                    active_jobs=len(futures),
                    jobs=args.jobs,
                )
                print(
                    f"[DONE] completed={completed_unique + new_count}/{total_unique} vector={name} "
                    f"power_uw={values[0]:.6f} delay_ns={values[1]:.6f} "
                    f"area_um2={values[2]:.6f} pdp_fj={values[3]:.6f}",
                    flush=True,
                )

                task = next(task_iter, None)
                if task is not None and not STOP_EVENT.is_set():
                    next_vector, _, next_hash = task
                    print(f"[START] vector={vector_name(next_vector)} hash={next_hash}", flush=True)
                    futures[executor.submit(run_flow, next_vector, next_hash)] = task

            if failure is not None:
                for future in futures:
                    future.cancel()
                break

    if failure is not None:
        raise failure

    if args.max_new is not None and completed_unique + new_count < total_unique:
        write_status(state="paused", reason=f"--max-new={args.max_new} reached")
        return 0

    write_status(
        state="complete",
        completed_unique=total_unique,
        current_vector=None,
        current_hash=None,
        current_row=None,
        error=None,
    )
    print(f"[COMPLETE] measured all {total_unique} unique configurations", flush=True)
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(143))
    try:
        raise SystemExit(main())
    except FlowError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
