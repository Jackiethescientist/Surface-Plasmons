"""Order 5: shift sphere index 2 by +4,+8,+16 nm in a five-sphere 80 nm chain.

Run one displacement batch at a time. Each batch runs Ex and Ez, with
one worker by default. Existing perfect-chain results supply the 0% baseline.

    python scripts/run_displacements_v21.py --shift-nm 4 --name chain5_dx4 --dry-run
    python scripts/run_displacements_v21.py --shift-nm 4 --name chain5_dx4

Only the valid z mirror is retained. Keeping an x mirror would change the
asymmetric geometry. Numerical convergence checks are separate from theory.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1]/'src'
if not (_LOCAL_SOURCE/'surface_plasmons/__init__.py').is_file():
    raise SystemExit('Place this file in scripts/ inside the existing project.')
sys.path.insert(0, str(_LOCAL_SOURCE))

try:
    from .run_chains_v19 import make_chain_config
    from .run_control_v16 import (ControlJob, GIB, available_memory_bytes, execute_jobs, memory_budget_bytes)
except ImportError:
    from run_chains_v19 import make_chain_config
    from run_control_v16 import (ControlJob, GIB, available_memory_bytes, execute_jobs, memory_budget_bytes)

from surface_plasmons import project as _project_source
from surface_plasmons.project import estimate_runtime, make_simulation_config, normalise_project_config
from surface_plasmons.results import config_hash, safe_simulation_name
from surface_plasmons.simulation import SimulationConfig
from surface_plasmons.validated_scope import enforce_validated_project
from surface_plasmons.workflow import _reference_signature

if (not getattr(_project_source, 'RESEARCH_PARTIAL_MIRRORS_SUPPORTED', False)
        or 'mirror_axes' not in SimulationConfig.__dataclass_fields__):
    raise SystemExit('Install all code files from the order-5 update, including the updated src/surface_plasmons/ files.')

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_displacement_config(root, shift_nm, polarisation, name):
    if shift_nm not in (4, 8, 16):
        raise ValueError('Choose a positive central displacement of 4, 8 or 16 nm.')
    config = make_chain_config(root, 5, 80, polarisation, name)
    config['geometry']['defects'] = [{'particle_index': 2, 'shift_um': [shift_nm/1000, 0., 0.]}]
    config['numerics'].update(mirror_symmetry=True, mirror_axes=['z'])
    config = normalise_project_config(config)
    enforce_validated_project(config, field_map_selection='none')
    return config


def plan_jobs(root, shift_nm, names, results_root, resume=False):
    if safe_simulation_name(names[0]) == safe_simulation_name(names[1]):
        raise ValueError('Ex and Ez require different result-folder names.')
    jobs = []
    for pol, name in zip(('Ex', 'Ez'), names):
        config = make_displacement_config(root, shift_nm, pol, name)
        target = results_root/safe_simulation_name(name)
        raw = make_simulation_config(config, experiment_directory=target, stage_name='frequency_scan/particle')
        reference = make_simulation_config(config, experiment_directory=target, stage_name='reference', reference=True)
        job = ControlJob(pol, config, target, config_hash(_reference_signature(reference)),
                         float(estimate_runtime(raw)['estimated_memory_gb'])*1e9)
        if target.exists():
            manifest = target/'manifest.json'
            if not manifest.is_file():
                raise ValueError(f'Existing folder has no manifest: {target}. Choose a new name.')
            saved = json.loads(manifest.read_text(encoding='utf-8'))
            if saved.get('configuration_hash') != config_hash(config):
                raise ValueError(f'Existing result uses different settings: {target}. Choose a new name.')
            job.reuse = saved.get('status') == 'completed'
            job.resume = not job.reuse and resume
            if not job.reuse and not resume:
                raise ValueError(f'Interrupted result exists: {target}. Add --resume or choose a new name.')
        jobs.append(job)
    if len({j.reference_key for j in jobs}) != len(jobs):
        raise ValueError('Parallel workers require separate reference-cache identities.')
    return jobs


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--shift-nm', type=int, choices=(4, 8, 16), required=True)
    parser.add_argument('--name', help='Batch and result prefix; _ex/_ez are appended.')
    parser.add_argument('--ex-name')
    parser.add_argument('--ez-name')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key, '1')) > 1:
            raise ValueError('Run this launcher directly with python, without mpirun.')
    prefix = safe_simulation_name(args.name or f'chain5_dx{args.shift_nm}_v21')
    names = (args.ex_name or prefix+'_ex', args.ez_name or prefix+'_ez')
    results_root = (args.results_root or PROJECT_ROOT/'data/results').expanduser().resolve()
    jobs = plan_jobs(PROJECT_ROOT, args.shift_nm, names, results_root, args.resume)
    print(f'Central sphere index 2: +{args.shift_nm} nm along x ({100*args.shift_nm/80:g}% of nominal spacing).')
    print(f'Adjacent centre spacings: {80+args.shift_nm} nm left; {80-args.shift_nm} nm right.')
    print('Mirrors: z only. The perfect-chain results remain the 0% baseline.')
    for job in jobs:
        action = 'reuse completed solver result' if job.reuse else 'resume' if job.resume else 'new run'
        print(f'{job.polarisation}: {action}; {job.target}', flush=True)
    budget = memory_budget_bytes(jobs, args.workers)
    available = available_memory_bytes()
    print(f'Workers: {args.workers}; RAM allowance: {budget/GIB:.2f} GiB (includes headroom).')
    print(f'Available RAM: {available/GIB:.2f} GiB' if available is not None else 'Available RAM could not be read.')
    if args.dry_run:
        if available is not None and budget > available:
            print('Insufficient available RAM for the allowance. Close other workloads before running.')
        print('Dry run complete: no calculations or file writes.')
        return 0
    if not budget:
        print('Both solver results already exist. Inspect the saved validation reports.')
        return 0
    if available is not None and budget > available:
        raise ValueError('Insufficient available RAM for the allowance. Use one worker and free memory before running.')
    batch = PROJECT_ROOT/'data/control_batches'/prefix
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch/'launcher.lock'
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f'Launcher lock exists: {lock}. Check for an active launcher before removing a stale lock.') from exc
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(str(os.getpid())+'\n')
        jobs = plan_jobs(PROJECT_ROOT, args.shift_nm, names, results_root, args.resume)
        return execute_jobs(jobs, root=PROJECT_ROOT, batch=batch, workers=args.workers, heartbeat_seconds=30)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit('Displacements interrupted; results were preserved. Repeat with --resume.')
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Displacement launch failed: {exc}') from exc
