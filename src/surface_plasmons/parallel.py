"""Spawned solver processes and inter-process reference-cache locks.

Keep MEEP out of threads and never fork a process that may have initialised
MPI. Only the parent writes experiment/sweep manifests.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path
import os
import signal


def validate_workers(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    return workers


@contextmanager
def reference_lock(path: Path):
    """Hold a kernel lock through cache inspection, creation and publication.

    Linux/WSL releases flock automatically if a solver is interrupted or dies;
    an empty .lock file left on disk is harmless.
    """
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _initialise_worker(initializer, initargs):
    # The parent handles Ctrl+C and shuts down the whole pool together.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if initializer is not None:
        initializer(*initargs)


@contextmanager
def worker_pool(workers: int, *, initializer=None, initargs=()):
    """Create isolated solvers with one numerical-library thread per worker.

    Set the environment before spawn: NumPy/OpenMP may initialise while the
    child's main module is being imported, before its initializer runs.
    """
    validate_workers(workers)
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    executor = None
    try:
        for name in names:
            os.environ[name] = "1"
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("spawn"),
            initializer=_initialise_worker, initargs=(initializer, initargs),
        )
        yield executor
    except BaseException:
        if executor is not None:
            terminate = getattr(executor, "terminate_workers", None)
            if terminate is not None:  # Python 3.14+
                terminate()
            else:
                # Supported Python 3.10-3.13 lack a public terminate method.
                # Stop running native solvers too, rather than waiting hours
                # for shutdown(wait=True) after Ctrl+C or a failed sibling.
                for process in tuple((getattr(executor, "_processes", None) or {}).values()):
                    if process.is_alive():
                        process.terminate()
                executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
