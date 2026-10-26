"""Exercise real spawn processes and disk I/O without requiring PyMEEP."""

from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time
import uuid

import numpy as np
import pytest

from surface_plasmons.parallel import validate_workers, worker_pool
from surface_plasmons.project import normalise_project_config
from surface_plasmons.results import ExperimentWorkspace
from surface_plasmons.sweep import parameter_sweep
from surface_plasmons import workflow


_barrier = None


def _fake_solver(config):
    if _barrier is not None:
        _barrier.wait(timeout=15)
    root = Path(os.environ["TEST_SOLVER_ROOT"])
    # Append is one short write; the reference-cache test verifies only one
    # solver ever enters this function for the same shared reference key.
    with (root / "calls.jsonl").open("a") as handle:
        handle.write(json.dumps({"pid": os.getpid(), "run_id": config.get("run_id")}) + "\n")
    time.sleep(.1)
    output = config.get("output", {})
    run_id = config.get("run_id") or uuid.uuid4().hex
    experiment = output.get("experiment_directory")
    directory = (Path(experiment) / "stages" / output["stage_name"]
                 if experiment else root / run_id)
    arrays = directory / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    for component in ("Ex", "Ey", "Ez"):
        np.savez_compressed(arrays / f"field_{component}_spectrum.npz",
                            values=np.ones((1, 4, 1, 1), dtype=complex), frequencies=[1.5])
    np.savez(arrays / "coordinates.npz", x=np.arange(4), y=[0.], z=[0.])
    return SimpleNamespace(run_id=run_id, status="completed", output_directory=str(directory),
                           elapsed_seconds=.1, metrics={"worker_pid": float(os.getpid())})


def _fake_initializer(root, barrier=None):
    global _barrier
    _barrier = barrier
    os.environ["TEST_SOLVER_ROOT"] = str(root)
    workflow.run_simulation = _fake_solver
    workflow.find_project_root = lambda: Path(root)
    # __init__ exports a compatibility function named sweep; import explicitly.
    import importlib
    importlib.import_module("surface_plasmons.sweep").run_simulation = _fake_solver


def _probe(_):
    from surface_plasmons import simulation
    return (os.getpid(), get_context().get_start_method(),
            os.environ["OMP_NUM_THREADS"], simulation._execute.__name__)


def _crash(_):
    os._exit(9)


def test_broadband_reference_and_particle_overlap_and_parent_keeps_manifest(tmp_path, monkeypatch):
    barrier = get_context("spawn").Barrier(2)
    monkeypatch.setenv("TEST_SOLVER_ROOT", str(tmp_path))
    monkeypatch.setattr(workflow, "run_simulation", _fake_solver)

    def pool(workers):
        return worker_pool(workers, initializer=_fake_initializer, initargs=(tmp_path, barrier))

    monkeypatch.setattr(workflow, "worker_pool", pool)
    config = normalise_project_config({"simulation_name": "parallel"})
    workspace = ExperimentWorkspace.create("parallel", config, results_root=tmp_path / "results")
    result, actual, reference, _ = workflow.run_broadband_stage(
        config, workspace, stage_prefix="test_scan", workers=2)
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 2
    assert len({row["pid"] for row in calls}) == 2
    assert os.getpid() not in {row["pid"] for row in calls}
    assert result.status == "completed"
    assert actual.fields["Ex"].shape == reference.fields["Ex"].shape == (1, 4, 1, 1)
    assert workspace.manifest()["stages"]["test_scan"]["status"] == "completed"


def test_shared_reference_is_calculated_once_and_disabled_cache_is_isolated(tmp_path):
    config = normalise_project_config({"simulation_name": "shared"})
    payload = (config, {"wavelength_um": None, "cache_enabled": True})
    with worker_pool(2, initializer=_fake_initializer, initargs=(tmp_path,)) as pool:
        futures = [pool.submit(workflow._reference_worker, payload) for _ in range(2)]
        directories = [job.result(timeout=20) for job in futures]
        assert directories[0] == directories[1]
        assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 1
        uncached = (config, {"wavelength_um": None, "cache_enabled": False})
        futures = [pool.submit(workflow._reference_worker, uncached) for _ in range(2)]
        fresh = [job.result(timeout=20) for job in futures]
    assert len(set(directories + fresh)) == 3
    assert len((tmp_path / "calls.jsonl").read_text().splitlines()) == 3
    for directory in directories + fresh:
        assert (Path(directory).parents[1] / "complete.json").is_file()


def test_two_worker_sweep_writes_all_rows_in_parent(tmp_path):
    result = parameter_sweep(
        {"geometry_name": "nanoparticle_sphere", "record_database": False},
        {"geometry_parameters.radius": [.02, .04, .06, .08]},
        workers=2, resume=False, output_root=str(tmp_path / "sweeps"),
        worker_initializer=_fake_initializer, worker_initargs=(tmp_path,),
    )
    assert result.status == "completed"
    assert [run.index for run in result.runs] == [0, 1, 2, 3]
    assert len({run.metrics["worker_pid"] for run in result.runs}) == 2
    saved = json.loads(Path(result.manifest_path).read_text())
    assert saved["completed_runs"] == 4
    assert len(Path(result.table_path).read_text().splitlines()) == 5


def test_sweep_cache_initializer_reaches_spawned_worker_and_environment_restores(monkeypatch):
    from scripts import run_sweep
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    with worker_pool(2, initializer=run_sweep.initialise_sweep_worker, initargs=(False,)) as pool:
        pid, method, threads, execute = pool.submit(_probe, None).result(timeout=20)
    assert pid != os.getpid()
    assert method == "spawn"
    assert threads == "1"
    assert execute == "cached_execute"
    assert os.environ["OMP_NUM_THREADS"] == "7"


def test_abrupt_worker_death_propagates_instead_of_hanging():
    with pytest.raises(BrokenProcessPool):
        with worker_pool(2) as pool:
            pool.submit(_crash, None).result(timeout=20)


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5, "2"])
def test_invalid_worker_counts_fail_before_starting_solvers(workers):
    with pytest.raises(ValueError, match="positive integer"):
        validate_workers(workers)
