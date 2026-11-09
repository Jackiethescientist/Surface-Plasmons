"""Order 4: compare saved perfect five-sphere chains at 60,80,100 nm spacing.

Requires the v19 chain launcher and the updated comparison helper in this
download. Existing simulation folders are read only; no Meep runs are started.

    python scripts/compare_chain_spacings_v20.py --spacing60 chain5_60_ex --spacing80 chain5_80_ex --spacing100 chain5_100_ex --name chain5_spacing_ex

Fields are sampled at identical offsets from the leftmost sphere centre,
matching the coordinate convention used for the particle-count comparison.
Default offsets are -45,-40,-35,-30 nm, with a two-cell exterior buffer.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import inspect
import json
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1] / 'src'
if not (_LOCAL_SOURCE / 'surface_plasmons' / '__init__.py').is_file():
    raise SystemExit('Place this file in scripts/ inside your existing project.')
sys.path.insert(0, str(_LOCAL_SOURCE))

try:
    from . import compare_particle_counts_v19 as shared
except ImportError:
    import compare_particle_counts_v19 as shared

if 'expected_spacing_nm' not in inspect.signature(shared.read_run).parameters:
    raise SystemExit('Replace compare_particle_counts_v19.py with the updated helper included in the order-4 download.')

import numpy as np
from surface_plasmons.results import safe_simulation_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACINGS_NM = (60, 80, 100)


def compare_spacings(runs, offsets_nm):
    if (len(runs) != 3 or any(r['count'] != 5 for r in runs)
            or not np.allclose([r['spacing_nm'] for r in runs], SPACINGS_NM, rtol=0, atol=1e-8)):
        raise ValueError('Supply perfect N=5 chains in order: 60, 80 and 100 nm centre spacing.')
    return shared.compare_runs(runs, offsets_nm, expected_counts=(5, 5, 5))


def save_comparison(runs, records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():
        raise ValueError(f'Comparison folder already exists: {output}. Choose a new --name.')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'chain_spacing_fields.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    pol = runs[0]['config']['source']['polarisation']
    offsets = sorted({r['offset_from_leftmost_centre_nm'] for r in records})
    probe = min(offsets, key=lambda x: abs(x+40))
    grid = runs[0]['grid']
    wavelengths = sorted({float(grid[0]), float(grid[np.argmin(abs(grid-1100))]), float(grid[-1])})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for spacing in SPACINGS_NM:
        values = sorted([r for r in records if np.isclose(r['centre_spacing_nm'], spacing)
                         and r['offset_from_leftmost_centre_nm']==probe], key=lambda r:r['wavelength_nm'])
        for axis, part in zip(axes, ('real', 'imag')):
            axis.plot([r['wavelength_nm'] for r in values], [r[f'S_{pol}_{part}'] for r in values], label=f'd={spacing} nm')
            axis.set(xlabel='Vacuum wavelength (nm)', ylabel=f'{part.capitalize()}(scattered {pol} / Eref)')
            axis.grid(alpha=.2)
            axis.legend()
    fig.suptitle(f'N=5; exterior probe {probe:g} nm from leftmost sphere centre')
    fig.tight_layout()
    fig.savefig(output / 'complex_spectra_by_spacing.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    for wavelength in wavelengths:
        values = sorted([r for r in records if r['offset_from_leftmost_centre_nm']==probe
                         and r['wavelength_nm']==wavelength], key=lambda r:r['centre_spacing_nm'])
        axis.plot([r['centre_spacing_nm'] for r in values], [r[f'S_{pol}_magnitude'] for r in values], 'o-', label=f'{wavelength:.1f} nm')
    axis.set(xlabel='Centre-to-centre spacing (nm)', ylabel=f'|scattered {pol} / Eref|',
             xticks=list(SPACINGS_NM), title=f'N=5; exterior end probe {probe:g} nm')
    axis.grid(alpha=.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'field_magnitude_vs_spacing.png', dpi=180)
    plt.close(fig)
    summary = dict(particle_count=5, spacings_nm=list(SPACINGS_NM),
                   surface_gaps_nm=[d-40 for d in SPACINGS_NM], polarisation=pol,
                   coordinate_frame='xi = x - leftmost sphere centre x',
                   offsets_nm=offsets, primary_plot_probe_nm=probe,
                   measurement='complex scattered E / common complex Eref; field amplitude is not squared',
                   interpolation='real and imaginary parts interpolated separately inside eligible exterior samples',
                   no_new_simulations_started=True,
                   numerical_acceptance_note='Review per-case convergence and compare physical differences with numerical changes.',
                   cases=[dict(centre_spacing_nm=r['spacing_nm'], result_folder=str(r['directory']),
                               leftmost_sphere_x_nm=r['first_x']*1000, **r['statuses']) for r in runs])
    (output / 'comparison_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    for spacing in SPACINGS_NM:
        parser.add_argument(f'--spacing{spacing}', required=True, help='Saved simulation name or directory.')
    parser.add_argument('--name', required=True, help='New comparison-folder name.')
    parser.add_argument('--results-root', type=Path, default=PROJECT_ROOT / 'data/results')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'data/comparisons')
    parser.add_argument('--offsets-nm', default='-45,-40,-35,-30', help='Use --offsets-nm=-45,-40,...')
    args = parser.parse_args(argv)
    runs = []
    for spacing in SPACINGS_NM:
        value = getattr(args, f'spacing{spacing}')
        path = Path(value).expanduser()
        directory = path.resolve() if path.is_dir() else args.results_root.expanduser().resolve()/safe_simulation_name(value)
        runs.append(shared.read_run(directory, 5, expected_spacing_nm=spacing))
    records = compare_spacings(runs, [float(v) for v in args.offsets_nm.split(',') if v.strip()])
    output = args.output_root.expanduser().resolve()/safe_simulation_name(args.name)
    save_comparison(runs, records, output)
    print(f'Saved spacing comparison: {output}')
    print('Reused the saved 80 nm chain. All source result folders were read only.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f'Chain-spacing comparison failed: {exc}') from exc
