"""Step 3: perfect three- or five-sphere chains, with separate Ex/Ez workers.

Requires the corrected v18 update. From the project root:
    python scripts/run_chains_v19.py --particles 3 --name chain3_80 --workers 2 --dry-run
    python scripts/run_chains_v19.py --particles 3 --name chain3_80 --workers 2
    python scripts/run_chains_v19.py --particles 5 --name chain5_80 --workers 2 --dry-run

The default centre spacing is 80 nm. Existing N=1 and N=2 simulations are
reused for comparison, not rerun by this launcher. Numerical convergence
checks remain separate from the automatic electrostatic theory comparison.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1] / 'src'
if not (_LOCAL_SOURCE / 'surface_plasmons' / '__init__.py').is_file():
    raise SystemExit('Place this file in scripts/ inside the existing project containing src/surface_plasmons/.')
sys.path.insert(0, str(_LOCAL_SOURCE))

try:
    from .run_pairs_v18 import make_pair_config
    from .run_control_v16 import (ControlJob, GIB, available_memory_bytes,
                                  execute_jobs, memory_budget_bytes)
except ImportError:
    from run_pairs_v18 import make_pair_config
    from run_control_v16 import (ControlJob, GIB, available_memory_bytes,
                                 execute_jobs, memory_budget_bytes)

from surface_plasmons.project import estimate_runtime, make_simulation_config, normalise_project_config
from surface_plasmons.results import config_hash, safe_simulation_name
from surface_plasmons.validated_scope import enforce_validated_project
from surface_plasmons.workflow import _reference_signature


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_chain_config(root, count, spacing_nm, polarisation, name):
    if count not in (3, 5):
        raise ValueError('Choose --particles 3 or 5 for this stage.')
    config = make_pair_config(root, spacing_nm, polarisation, name)
    config['geometry']['particle_count'] = count
    # Include every sphere and 40 nm beyond each outer surface. The monitor
    # grows with the chain; comparisons use particle-relative exterior probes.
    config['measurement']['probe_margin_um'] = .04
    config = normalise_project_config(config)
    enforce_validated_project(config, field_map_selection='none')
    return config


def plan_chain_jobs(root, count, spacing_nm, names, results_root, resume=False):
    if safe_simulation_name(names[0]) == safe_simulation_name(names[1]):
        raise ValueError('Ex and Ez require different result-folder names.')
    jobs = []
    for pol, name in zip(('Ex', 'Ez'), names):
        config = make_chain_config(root, count, spacing_nm, pol, name)
        target = results_root / safe_simulation_name(name)
        raw = make_simulation_config(config, experiment_directory=target, stage_name='frequency_scan/particle')
        reference = make_simulation_config(config, experiment_directory=target, stage_name='reference', reference=True)
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
    if len({j.reference_key for j in jobs}) != len(jobs):
        raise ValueError('Parallel workers must have separate reference-cache identities.')
    return jobs


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--particles', type=int, choices=(3, 5), required=True)
    parser.add_argument('--spacing-nm', type=int, choices=(80, 60, 100), default=80)
    parser.add_argument('--name', help='Batch and result prefix; _ex/_ez are appended.')
    parser.add_argument('--ex-name', help='Optional individual Ex result-folder name.')
    parser.add_argument('--ez-name', help='Optional individual Ez result-folder name.')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--results-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    for key in ('OMPI_COMM_WORLD_SIZE', 'PMI_SIZE', 'PMIX_SIZE', 'MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(key, '1')) > 1:
            raise ValueError('Run this launcher directly with python, without mpirun.')
    prefix = safe_simulation_name(args.name or f'chain{args.particles}_{args.spacing_nm}nm_v19')
    names = (args.ex_name or prefix + '_ex', args.ez_name or prefix + '_ez')
    results_root = (args.results_root or PROJECT_ROOT / 'data/results').expanduser().resolve()
    jobs = plan_chain_jobs(PROJECT_ROOT, args.particles, args.spacing_nm, names, results_root, args.resume)
    print('Project source: ' + str(_LOCAL_SOURCE), flush=True)
    for job in jobs:
        action = 'reuse completed solver result' if job.reuse else 'resume' if job.resume else 'new run'
        print(f'{job.polarisation}: {action}; {job.target}', flush=True)
    print(f'Perfect chain: N={args.particles}; {args.spacing_nm} nm centre spacing; {args.spacing_nm-40} nm surface gap.')
    print('No single-sphere or pair simulations will be started.')
    print('Theory agreement, model applicability and numerical convergence remain separate.')
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
        print('Both solver results already exist. Inspect their validation reports.')
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
        jobs = plan_chain_jobs(PROJECT_ROOT, args.particles, args.spacing_nm, names, results_root, args.resume)
        return execute_jobs(jobs, root=PROJECT_ROOT, batch=batch, workers=args.workers, heartbeat_seconds=30)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit('Chains interrupted; worker results were preserved. Repeat with --resume.')
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'Chain launch failed: {exc}') from exc
