"""Order 6: missing middle, missing right end and a regular four-sphere control.

    python scripts/run_vacancies_v22.py --case middle --name chain5_missing_middle --dry-run
    python scripts/run_vacancies_v22.py --case middle --name chain5_missing_middle

Each batch runs Ex and Ez with one worker by default. Reuse the completed
perfect five-sphere 80 nm results. The nominal five-sphere envelope and line
monitor are kept for every new case; padding changes still grow the domain.
The end vacancy and regular four-sphere chain are equivalent after translation
in a homogeneous host, so their comparison is a numerical consistency check.
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
from surface_plasmons.particle_line import line_extents
from surface_plasmons.project import (estimate_runtime, geometry_parameters, make_simulation_config,
                                      normalise_project_config)
from surface_plasmons.results import config_hash, safe_simulation_name
from surface_plasmons.validated_scope import enforce_validated_project
from surface_plasmons.workflow import _reference_signature

if not getattr(_project_source, 'RESEARCH_VACANCIES_SUPPORTED', False):
    raise SystemExit('Install the updated src/surface_plasmons/project.py included in the order-6 download.')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NAMES = {'middle':'chain5_missing_middle', 'end':'chain5_missing_end', 'four':'chain4_regular'}


def make_vacancy_config(root, case, polarisation, name):
    if case not in DEFAULT_NAMES:
        raise ValueError('Choose case middle, end or four.')
    config = make_chain_config(root, 5, 80, polarisation, name)
    envelope = line_extents({**geometry_parameters(config, include_particles=True), 'center':(0.,0.,0.)})
    perfect = make_simulation_config(config, experiment_directory='data/results/vacancy_preview', stage_name='frequency_scan/particle')
    config['numerics']['minimum_geometry_extent_um'] = list(envelope)
    config['measurement'].update(probe_line_x_center_um=perfect['output']['field_region_center_um'][0],
                                  probe_line_span_um=perfect['output']['field_region_size_um'][0])
    if case == 'four':
        config['geometry'].update(particle_count=4, defects=[])
        config['numerics'].pop('mirror_axes', None)
    else:
        config['geometry']['defects'] = [{'particle_index':2 if case=='middle' else 4, 'missing':True}]
        if case == 'end':
            config['numerics']['mirror_axes'] = ['z']
        else:
            config['numerics'].pop('mirror_axes', None)
    config = normalise_project_config(config)
    enforce_validated_project(config, field_map_selection='none')
    return config


def plan_jobs(root, case, names, results_root, resume=False):
    if safe_simulation_name(names[0]) == safe_simulation_name(names[1]):
        raise ValueError('Ex and Ez require different result-folder names.')
    jobs = []
    for pol, name in zip(('Ex','Ez'), names):
        config = make_vacancy_config(root, case, pol, name)
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
    parser.add_argument('--case', choices=tuple(DEFAULT_NAMES), required=True)
    parser.add_argument('--name')
    parser.add_argument('--ex-name')
    parser.add_argument('--ez-name')
    parser.add_argument('--workers', type=int, choices=(1,2), default=1)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for key in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','PMIX_SIZE','MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key,'1')) > 1:
            raise ValueError('Run directly with python, without mpirun.')
    prefix = safe_simulation_name(args.name or DEFAULT_NAMES[args.case])
    names = (args.ex_name or prefix+'_ex', args.ez_name or prefix+'_ez')
    results_root = (args.results_root or PROJECT_ROOT/'data/results').expanduser().resolve()
    jobs = plan_jobs(PROJECT_ROOT, args.case, names, results_root, args.resume)
    print(f'Case: {args.case}; four active spheres; nominal spacing 80 nm.')
    print('Cell and line monitor match the perfect five-sphere baseline.')
    print('Mirrors: '+('z only' if args.case=='end' else 'x and z'))
    for job in jobs:
        action = 'reuse completed solver result' if job.reuse else 'resume' if job.resume else 'new run'
        print(f'{job.polarisation}: {action}; {job.target}', flush=True)
    budget = memory_budget_bytes(jobs, args.workers)
    available = available_memory_bytes()
    print(f'Workers: {args.workers}; RAM allowance: {budget/GIB:.2f} GiB (includes headroom).')
    print(f'Available RAM: {available/GIB:.2f} GiB' if available is not None else 'Available RAM could not be read.')
    if args.dry_run:
        if available is not None and budget > available:
            print('Insufficient available RAM for the allowance. Use one worker or free memory.')
        print('Dry run complete: no calculations or file writes.')
        return 0
    if not budget:
        print('Both solver results already exist. Inspect their validation reports.')
        return 0
    if available is not None and budget > available:
        raise ValueError('Insufficient available RAM for the allowance. Use one worker and free memory.')
    batch = PROJECT_ROOT/'data/control_batches'/prefix
    batch.mkdir(parents=True, exist_ok=True)
    lock = batch/'launcher.lock'
    try:
        fd = os.open(lock, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f'Launcher lock exists: {lock}. Check for an active launcher before removing a stale lock.') from exc
    try:
        with os.fdopen(fd,'w') as handle:
            handle.write(str(os.getpid())+'\n')
        jobs = plan_jobs(PROJECT_ROOT, args.case, names, results_root, args.resume)
        return execute_jobs(jobs, root=PROJECT_ROOT, batch=batch, workers=args.workers, heartbeat_seconds=30)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit('Vacancy runs interrupted; results were preserved. Repeat with --resume.')
    except (OSError,ValueError,RuntimeError) as exc:
        raise SystemExit(f'Vacancy launch failed: {exc}') from exc
