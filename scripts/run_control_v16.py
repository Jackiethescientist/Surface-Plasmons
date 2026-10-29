"""Repeat the exact validated Ex/Ez sphere controls in one or two processes.

From the project root, in the existing plasmon environment:
    python scripts/run_control_v16.py --workers 2 --dry-run
    python scripts/run_control_v16.py --workers 2

This launcher preserves the v15 execution restriction. It does not authorise
new particle pairs, chains, metals, wavelengths, or field-map measurements.
"""
from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import os
import signal
import subprocess
import sys
import time

from surface_plasmons.project import (
    estimate_runtime, load_project_config, make_simulation_config,
    normalise_project_config,
)
from surface_plasmons.results import config_hash, safe_simulation_name
from surface_plasmons.validated_scope import enforce_validated_project
from surface_plasmons.workflow import _reference_signature


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


@dataclass
class ControlJob:
    polarisation: str
    config: dict[str, Any]
    target: Path
    reference_key: str
    base_bytes: float
    reuse: bool = False
    resume: bool = False


def plan_jobs(root: Path, name: str, results_root: Path, resume: bool = False) -> list[ControlJob]:
    """Preflight both cases before starting either calculation or writing files."""
    jobs = []
    for pol in ("Ex", "Ez"):
        template = root / "configs" / f"validated_sphere_{pol.lower()}_v15.json"
        if not template.is_file():
            raise ValueError(f"Missing {template}. Install the included v13/v15 support files first.")
        config = normalise_project_config(
            load_project_config(template), name=f"{safe_simulation_name(name)}_{pol.lower()}"
        )
        config["numerics"]["record_database"] = False
        enforce_validated_project(config, field_map_selection="none")
        if config["source"]["polarisation"] != pol:
            raise ValueError(f"{template.name} must use {pol}.")
        target = results_root / safe_simulation_name(config["simulation_name"])
        raw = make_simulation_config(
            config, experiment_directory=target, stage_name="frequency_scan/particle"
        )
        reference = make_simulation_config(
            config, experiment_directory=target, stage_name="reference", reference=True
        )
        job = ControlJob(pol, config, target, config_hash(_reference_signature(reference)),
                         float(estimate_runtime(raw)["estimated_memory_gb"]) * 1e9)
        if target.exists():
            manifest = target / "manifest.json"
            if not manifest.is_file():
                raise ValueError(f"Existing folder has no manifest: {target}. Use a new --name.")
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved.get("configuration_hash") != config_hash(config):
                raise ValueError(f"Existing result uses different settings: {target}. Use a new --name.")
            job.reuse = saved.get("status") == "completed"
            job.resume = not job.reuse and resume
            if not job.reuse and not resume:
                raise ValueError(f"Interrupted result exists: {target}. Add --resume or use a new --name.")
        jobs.append(job)
    if len({job.reference_key for job in jobs}) != len(jobs):
        raise ValueError("The controls share a reference-cache key; parallel launch is refused.")
    return jobs


def available_memory_bytes() -> int | None:
    """Read WSL/Linux available RAM, respecting a smaller container allowance."""
    try:
        rows = Path("/proc/meminfo").read_text().splitlines()
        available = next(int(row.split()[1]) * 1024 for row in rows if row.startswith("MemAvailable:"))
    except (OSError, ValueError, StopIteration):
        return None
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if limit != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available = min(available, max(0, int(limit) - used))
    except (OSError, ValueError):
        pass
    return available


def memory_budget_bytes(jobs: list[ControlJob], workers: int) -> int:
    """Planning allowance, not measured peak RSS or a guarantee of fit."""
    active = sorted((job.base_bytes for job in jobs if not job.reuse), reverse=True)[:workers]
    if not active:
        return 0
    # The existing estimate covers base fields. Allow Python/geometry overhead
    # and leave another 0.75 GiB available beyond the concurrently active jobs.
    return int(sum(2.5 * value + 0.25 * GIB for value in active) + 0.75 * GIB)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def stop_children(active: list[tuple[ControlJob, subprocess.Popen, Any]]) -> None:
    for _, process, _ in active:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 8
    while any(process.poll() is None for _, process, _ in active) and time.monotonic() < deadline:
        time.sleep(0.1)
    for _, process, _ in active:
        if process.poll() is None:
            process.terminate()
    for _, process, handle in active:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        handle.close()


def execute_jobs(jobs: list[ControlJob], *, root: Path, batch: Path, workers: int) -> int:
    """Use fresh Python processes: Meep state is never shared between jobs."""
    pending = [job for job in jobs if not job.reuse]
    active: list[tuple[ControlJob, subprocess.Popen, Any]] = []
    failed = False
    try:
        while pending or active:
            while pending and len(active) < workers and not failed:
                job = pending.pop(0)
                config_path = batch / f"{job.polarisation.lower()}_config.json"
                config_path.write_text(json.dumps(job.config, indent=2) + "\n", encoding="utf-8")
                log_path = batch / f"{job.polarisation.lower()}.log"
                command = [sys.executable, "-u", str(root / "scripts" / "run_single.py"),
                           "--config", str(config_path), "--name", job.config["simulation_name"],
                           "--results-root", str(job.target.parent), "--field-maps", "none"]
                if job.resume:
                    command.append("--resume")
                handle = log_path.open("a" if job.resume else "w", encoding="utf-8")
                try:
                    process = subprocess.Popen(command, cwd=root, env=child_environment(),
                                               stdout=handle, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    handle.close()
                    raise
                active.append((job, process, handle))
                print(f"Started {job.polarisation} (PID {process.pid}); log: {log_path}", flush=True)
            for entry in active[:]:
                job, process, handle = entry
                code = process.poll()
                if code is not None:
                    handle.close()
                    active.remove(entry)
                    failed |= code != 0
                    print(f"{job.polarisation}: {'finished' if code == 0 else f'FAILED (exit {code})'}", flush=True)
            if failed:
                pending.clear()
            if active:
                time.sleep(0.2)
    except BaseException:
        stop_children(active)
        raise
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--name", default="control_sphere_v16", help="Prefix; _ex and _ez are appended.")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "data" / "results")
    parser.add_argument("--resume", action="store_true", help="Resume matching interrupted folders; complete stages are reused.")
    parser.add_argument("--dry-run", action="store_true", help="Preflight and estimate only; write nothing and do not import Meep.")
    args = parser.parse_args(argv)
    # This is independent-job parallelism. Wrapping the launcher in mpirun
    # would duplicate its orchestration and output writes.
    for key in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MV2_COMM_WORLD_SIZE"):
        try:
            size = int(os.environ.get(key, "1"))
        except ValueError:
            size = 1
        if size > 1:
            raise ValueError("Run this launcher with python directly, without mpirun.")
    result_root = args.results_root.expanduser().resolve()
    jobs = plan_jobs(PROJECT_ROOT, args.name, result_root, args.resume)
    for job in jobs:
        action = "reuse completed solver result" if job.reuse else "resume" if job.resume else "new run"
        print(f"{job.polarisation}: {action}; {job.target}")
    budget = memory_budget_bytes(jobs, args.workers)
    available = available_memory_bytes()
    print(f"Workers: {args.workers}; estimated concurrent RAM allowance: {budget/GIB:.2f} GiB (includes headroom).")
    print(f"Available RAM: {available/GIB:.2f} GiB" if available is not None else "Available RAM could not be read.")
    print("Each worker uses this Python environment with one numerical thread.")
    if args.dry_run:
        if available is not None and budget > available:
            print("Insufficient available RAM for the allowance; use --workers 1 or free memory before running.")
        print("Dry run complete: no calculations or file writes.")
        return 0
    if not budget:
        print("Both solver results already exist. No Meep runs were started; retain and inspect their validation reports.")
        return 0
    if available is not None and budget > available:
        raise ValueError("Insufficient available RAM for the estimated allowance. Use --workers 1 or free memory.")
    batch = PROJECT_ROOT / "data" / "control_batches" / safe_simulation_name(args.name)
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch / "launcher.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Launcher lock exists: {lock}. Check for an active launcher before removing a stale lock.") from exc
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(str(os.getpid()) + "\n")
        # Recheck under the batch lock before any child can create results.
        jobs = plan_jobs(PROJECT_ROOT, args.name, result_root, args.resume)
        return execute_jobs(jobs, root=PROJECT_ROOT, batch=batch, workers=args.workers)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Controls interrupted; worker results were preserved. Repeat with --resume.")
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Control launch failed: {exc}") from exc
