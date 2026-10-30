"""Run step 2: one named Ex/Ez sphere pair at 80, 60, or 100 nm spacing.

    python scripts/run_pairs_v18.py --spacing-nm 80 --name pair80 --workers 2 --dry-run
    python scripts/run_pairs_v18.py --spacing-nm 80 --name pair80 --workers 2

These are explicitly unvalidated research geometries. Default single-sphere
control restrictions, their names, and their saved configuration hashes are
unchanged. The launcher never starts numerical convergence runs automatically.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import sys

# Script execution otherwise searches scripts/ and the installed package, which
# may be an editable install pointing at another checkout of this project.
_LOCAL_SOURCE = Path(__file__).resolve().parents[1] / 'src'
if not (_LOCAL_SOURCE / 'surface_plasmons' / '__init__.py').is_file():
    raise SystemExit('Project source is missing. Merge all contents of code/ into your existing '
                     'project root so scripts/ and src/surface_plasmons/ are alongside each other.')
sys.path.insert(0, str(_LOCAL_SOURCE))

try:
    from .run_control_v16 import (ControlJob, GIB, available_memory_bytes, execute_jobs,
                                  memory_budget_bytes)
except ImportError:
    from run_control_v16 import (ControlJob, GIB, available_memory_bytes, execute_jobs,
                                 memory_budget_bytes)

from surface_plasmons.project import (estimate_runtime, load_project_config,
    make_simulation_config, normalise_project_config)
from surface_plasmons.results import config_hash, safe_simulation_name
from surface_plasmons.validated_scope import enforce_validated_project
from surface_plasmons.workflow import _reference_signature
from surface_plasmons import project as _project_source

if not getattr(_project_source, 'RESEARCH_DIELECTRIC_LINE_SUPPORTED', False):
    raise SystemExit('Pair launch failed: the source in ' + str(Path(_project_source.__file__).resolve())
                     + ' does not include the v18 research extension. Merge every file from '
                     'the updated code/src/ folder into this project, including project.py, '
                     'simulation.py, validated_scope.py and research_scope.py.')


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_pair_config(root: Path, spacing_nm: int, polarisation: str, name: str):
    if spacing_nm not in (60, 80, 100):
        raise ValueError('Choose centre spacing 60, 80 or 100 nm.')
    if polarisation not in ('Ex', 'Ez'):
        raise ValueError('Choose Ex or Ez.')
    config = load_project_config(root / 'configs' / f'validated_sphere_{polarisation.lower()}_v15.json')
    config['simulation_name'] = safe_simulation_name(name)
    config['geometry'].update(particle_count=2, spacing_um=spacing_nm / 1000.)
    # All spacings use the same physical x span, [-120,120] nm. Only the
    # common exterior subset may be compared between geometries.
    config['measurement'].update(validation_region='all_exterior',
        probe_margin_um=.12 - .02 - spacing_nm / 2000.)
    config['numerics'].update(execution_scope='research_dielectric_line',
        progress_interval=10., record_database=False)
    config = normalise_project_config(config)
    enforce_validated_project(config, field_map_selection='none')
    return config


def plan_pair_jobs(root: Path, spacing_nm: int, names: tuple[str, str], results_root: Path,
                   resume: bool = False):
    if safe_simulation_name(names[0]) == safe_simulation_name(names[1]):
        raise ValueError('Ex and Ez require different result-folder names.')
    jobs = []
    for pol, name in zip(('Ex', 'Ez'), names):
        config = make_pair_config(root, spacing_nm, pol, name)
        target = results_root / safe_simulation_name(name)
        raw = make_simulation_config(config, experiment_directory=target,
                                     stage_name='frequency_scan/particle')
        reference = make_simulation_config(config, experiment_directory=target,
                                           stage_name='reference', reference=True)
        job = ControlJob(pol, config, target, config_hash(_reference_signature(reference)),
                         float(estimate_runtime(raw)['estimated_memory_gb']) * 1e9)
        if target.exists():
            manifest = target / 'manifest.json'
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
    if len({job.reference_key for job in jobs}) != len(jobs):
        raise ValueError('Parallel workers must have separate reference-cache identities.')
    return jobs


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--spacing-nm', type=int, choices=(80, 60, 100), default=80)
    parser.add_argument('--name', help='Batch name and default result prefix; _ex/_ez are appended.')
    parser.add_argument('--ex-name', help='Optional exact Ex result-folder name.')
    parser.add_argument('--ez-name', help='Optional exact Ez result-folder name.')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print('Project source: ' + str(Path(_project_source.__file__).resolve()), flush=True)
    for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key, '1')) > 1:
            raise ValueError('Run the launcher directly with python, without mpirun.')
    prefix = safe_simulation_name(args.name or f'pair_{args.spacing_nm}nm_v18')
    names = (args.ex_name or prefix + '_ex', args.ez_name or prefix + '_ez')
    results_root = (args.results_root or PROJECT_ROOT / 'data/results').expanduser().resolve()
    jobs = plan_pair_jobs(PROJECT_ROOT, args.spacing_nm, names, results_root, args.resume)
    for job in jobs:
        action = 'reuse completed solver result' if job.reuse else 'resume' if job.resume else 'new run'
        print(f'{job.polarisation}: {action}; {job.target}', flush=True)
    print(f'Research pair: {args.spacing_nm} nm centre spacing; {args.spacing_nm - 40} nm surface gap.')
    print('Theory agreement, model applicability and numerical convergence are separate statuses.')
    budget = memory_budget_bytes(jobs, args.workers)
    available = available_memory_bytes()
    print(f'Workers: {args.workers}; concurrent RAM allowance: {budget/GIB:.2f} GiB (includes headroom).')
    print(f'Available RAM: {available/GIB:.2f} GiB' if available is not None else 'Available RAM could not be read.')
    if args.dry_run:
        if available is not None and budget > available:
            print('Use --workers 1 or free memory before a real run.')
        print('Dry run complete: no calculations or file writes.')
        return 0
    if not budget:
        print('Both solver results already exist. Inspect the saved validation reports.')
        return 0
    if available is not None and budget > available:
        raise ValueError('Insufficient available RAM for the allowance. Use --workers 1 or free memory.')
    batch = PROJECT_ROOT / 'data/control_batches' / prefix
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch / 'launcher.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f'Launcher lock exists: {lock}. Check for an active launcher before removing a stale lock.') from exc
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(str(os.getpid()) + '\n')
        jobs = plan_pair_jobs(PROJECT_ROOT, args.spacing_nm, names, results_root, args.resume)
        return execute_jobs(jobs, root=PROJECT_ROOT, batch=batch, workers=args.workers, heartbeat_seconds=30)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit('Pairs interrupted; worker results were preserved. Repeat with --resume.')
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Pair launch failed: {exc}') from exc
