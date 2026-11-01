"""Compare the perfect chain with +4,+8,+16 nm central-sphere displacements.

    python scripts/compare_central_displacements_v21.py --perfect chain5_80_ex --shift4 chain5_dx4_ex --shift8 chain5_dx8_ex --shift16 chain5_dx16_ex --name central_shift_ex

No solver is started. Original result folders are read only. All fields are
compared at fixed laboratory x coordinates, relative to the original central
sphere at x=0, y=20 nm, z=0. Shared exterior profiles exclude every sphere's
two-cell surface buffer in every configuration. Complex interpolation never
crosses an excluded interval and never extrapolates.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import inspect
import json
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1]/'src'
if not (_LOCAL_SOURCE/'surface_plasmons/__init__.py').is_file():
    raise SystemExit('Place this file in scripts/ inside the existing project.')
sys.path.insert(0, str(_LOCAL_SOURCE))

try:
    from . import compare_particle_counts_v19 as shared
except ImportError:
    import compare_particle_counts_v19 as shared

if 'expected_central_shift_nm' not in inspect.signature(shared.read_run).parameters:
    raise SystemExit('Install the updated compare_particle_counts_v19.py included in the order-5 download.')

import numpy as np
from surface_plasmons.results import safe_simulation_name

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHIFTS_NM = (0, 4, 8, 16)
COMPONENTS = ('Ex', 'Ey', 'Ez')


def segment_is_exterior(left, right, runs, resolution, extra_cells=0):
    """Distance from each sphere centre to the full interpolation segment."""
    for run in runs:
        centres = run['centres'][:, 0]
        distances = abs(centres-np.clip(centres, left, right))
        if np.any(distances < run['radius']+(2+extra_cells)/resolution-1e-12):
            return False
    return True


def sample_at(run, wavelength, x, runs, resolution):
    if not segment_is_exterior(x, x, runs, resolution):
        raise ValueError(f'Probe x={x*1000:g} nm lies inside a sphere or surface buffer in a compared case.')
    keep = np.isclose(run['wavelengths'], wavelength, rtol=1e-10, atol=1e-6)
    xs, values = run['coordinates'][keep, 0], run['values'][keep]
    order = np.argsort(xs)
    xs, values = xs[order], values[order]
    if len(xs) < 2 or np.any(np.diff(xs) <= 0):
        raise ValueError(f"{run['directory']}: distinct line samples are required.")
    exact = np.flatnonzero(np.isclose(xs, x, rtol=0, atol=1e-12))
    if len(exact):
        return values[exact[0]]
    index = int(np.searchsorted(xs, x))
    if index == 0 or index == len(xs):
        raise ValueError(f"{run['directory']}: x={1000*x:g} nm is outside the saved line; no extrapolation is performed.")
    left, right = xs[index-1], xs[index]
    if not segment_is_exterior(left, right, runs, resolution):
        raise ValueError(f'x={1000*x:g} nm would require interpolation across an excluded surface interval.')
    weight = (x-left)/(right-left)
    return values[index-1]*(1-weight)+values[index]*weight


def _record(run, wavelength, x, field, baseline):
    row = dict(central_shift_nm=run['central_shift_nm'],
               shift_percent_of_nominal_spacing=100*run['central_shift_nm']/80,
               wavelength_nm=float(wavelength), x_nm=float(x*1000), y_nm=20., z_nm=0.,
               source_result_folder=str(run['directory']))
    for j, component in enumerate(COMPONENTS):
        row.update({f'S_{component}_real': float(field[j].real),
                    f'S_{component}_imag': float(field[j].imag),
                    f'S_{component}_magnitude': float(abs(field[j])),
                    f'delta_S_{component}_real': float((field[j]-baseline[j]).real),
                    f'delta_S_{component}_imag': float((field[j]-baseline[j]).imag)})
    norm = float(np.linalg.norm(baseline))
    row['relative_complex_field_change_from_perfect'] = float(np.linalg.norm(field-baseline)/norm) if norm > 1e-20 else None
    return row


def compare_displacements(runs, probes_nm=(-200, -40, 48, 200)):
    if (len(runs) != 4 or [r['central_shift_nm'] for r in runs] != list(SHIFTS_NM)
            or any(r['count'] != 5 or not np.isclose(r['spacing_nm'], 80) for r in runs)):
        raise ValueError('Supply the perfect N=5 chain followed by +4,+8,+16 nm central displacements at d=80 nm.')
    grid = runs[0]['grid']
    pol = runs[0]['config']['source']['polarisation']
    if not len(grid):
        raise ValueError('No saved wavelength samples.')
    for run in runs:
        if run['config']['source']['polarisation'] != pol:
            raise ValueError('Compare one polarisation at a time.')
        if run['grid'].shape != grid.shape or not np.allclose(run['grid'], grid, rtol=1e-10, atol=1e-6):
            raise ValueError('Saved wavelength grids differ.')
    probes = np.unique(np.asarray(probes_nm, dtype=float))/1000
    if not len(probes) or not np.all(np.isfinite(probes)):
        raise ValueError('Provide finite physical probe coordinates.')
    resolution = min(r['resolution'] for r in runs)
    # Extra space for interpolation neighbours and monitor-grid rounding.
    lower = max(float(r['coordinates'][:, 0].min()) for r in runs)+2/resolution
    upper = min(float(r['coordinates'][:, 0].max()) for r in runs)-2/resolution
    xs = np.unique(runs[0]['coordinates'][:, 0])
    xs = [float(x) for x in xs if lower <= x <= upper
          and segment_is_exterior(float(x), float(x), runs, resolution, extra_cells=1)]
    if len(xs) < 4:
        raise ValueError('Too few common exterior points for position profiles.')
    probe_rows, profile_rows = [], []
    for wavelength in grid:
        for points, records in ((probes, probe_rows), (xs, profile_rows)):
            for x in points:
                fields = [sample_at(run, wavelength, float(x), runs, resolution) for run in runs]
                for run, field in zip(runs, fields):
                    records.append(_record(run, wavelength, float(x), field, fields[0]))
    return probe_rows, profile_rows


def _csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison(runs, probe_rows, profile_rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():
        raise ValueError(f'Comparison folder already exists: {output}. Choose a new --name.')
    output.mkdir(parents=True, exist_ok=False)
    _csv(output/'displacement_probe_fields.csv', probe_rows)
    _csv(output/'displacement_profile_fields.csv', profile_rows)
    pol = runs[0]['config']['source']['polarisation']
    probes = sorted({r['x_nm'] for r in probe_rows})
    local = [min(probes, key=lambda x:abs(x-target)) for target in (-40, 48)]
    grid = runs[0]['grid']
    mid = float(grid[np.argmin(abs(grid-1100))])
    wavelengths = sorted({float(grid[0]), mid, float(grid[-1])})
    fig, axes = plt.subplots(len(local), 2, figsize=(11, 7), squeeze=False)
    for i, x in enumerate(local):
        for shift in SHIFTS_NM:
            rows = sorted([r for r in probe_rows if r['x_nm']==x and r['central_shift_nm']==shift], key=lambda r:r['wavelength_nm'])
            for j, part in enumerate(('real','imag')):
                axes[i,j].plot([r['wavelength_nm'] for r in rows], [r[f'S_{pol}_{part}'] for r in rows], label=f'+{shift} nm')
                axes[i,j].set(xlabel='Vacuum wavelength (nm)', ylabel=f'{part.capitalize()}(scattered {pol} / Eref)', title=f'Fixed exterior probe x={x:g} nm')
                axes[i,j].grid(alpha=.2)
                axes[i,j].legend()
    fig.tight_layout()
    fig.savefig(output/'complex_spectra_near_defect.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(local), figsize=(11, 4), squeeze=False)
    for i, x in enumerate(local):
        for wave in wavelengths:
            rows = sorted([r for r in probe_rows if r['x_nm']==x and r['wavelength_nm']==wave], key=lambda r:r['central_shift_nm'])
            values = [100*r['relative_complex_field_change_from_perfect'] if r['relative_complex_field_change_from_perfect'] is not None else np.nan for r in rows]
            axes[0,i].plot([r['shift_percent_of_nominal_spacing'] for r in rows], values, 'o-', label=f'{wave:.1f} nm')
        axes[0,i].set(xlabel='Central displacement (% of 80 nm)', ylabel='Complex-field change from perfect (%)', title=f'Fixed exterior probe x={x:g} nm', xticks=[0,5,10,20])
        axes[0,i].grid(alpha=.2)
        axes[0,i].legend()
    fig.tight_layout()
    fig.savefig(output/'change_vs_displacement.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for shift in SHIFTS_NM:
        rows = sorted([r for r in profile_rows if r['central_shift_nm']==shift and r['wavelength_nm']==mid], key=lambda r:r['x_nm'])
        for axis, part in zip(axes, ('real','imag')):
            xvalues, yvalues = [], []
            previous = None
            for row in rows:
                if previous is not None and row['x_nm']-previous > 1500/min(r['resolution'] for r in runs):
                    xvalues.append(np.nan); yvalues.append(np.nan)
                xvalues.append(row['x_nm']); yvalues.append(row[f'S_{pol}_{part}']); previous=row['x_nm']
            axis.plot(xvalues, yvalues, '.-', label=f'+{shift} nm', markersize=3)
            axis.set(xlabel='Fixed laboratory x (nm)', ylabel=f'{part.capitalize()}(scattered {pol} / Eref)')
            axis.grid(alpha=.2)
            axis.legend()
    fig.suptitle(f'Common exterior position profiles at {mid:.1f} nm; gaps exclude particle surfaces')
    fig.tight_layout()
    fig.savefig(output/'complex_exterior_profiles.png', dpi=180)
    plt.close(fig)
    summary = dict(nominal_spacing_nm=80, particle_count=5, shifted_particle_index=2,
                   central_shifts_nm=list(SHIFTS_NM), shift_percentages=[0,5,10,20],
                   polarisation=pol, fixed_probe_x_nm=probes,
                   coordinate_frame='fixed laboratory x, relative to the original central sphere centre x=0; y=20 nm, z=0',
                   measurement='complex scattered E/Eref; amplitudes are not squared',
                   physical_change_definition='norm(S_shifted-S_perfect)/norm(S_perfect); not a theory-agreement error',
                   no_new_simulations_started=True,
                   numerical_acceptance_note='Each new defect geometry requires numerical convergence; the 4 nm shift is only 1.2 grid cells at resolution 300/um.',
                   cases=[dict(central_shift_nm=r['central_shift_nm'], result_folder=str(r['directory']),
                               adjacent_centre_spacings_nm=[80+r['central_shift_nm'],80-r['central_shift_nm']],
                               mirror_axes=r['config']['numerics'].get('mirror_axes', ['x','z'] if r['config']['numerics'].get('mirror_symmetry') else []),
                               **r['statuses']) for r in runs])
    (output/'comparison_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--perfect', required=True, help='Saved perfect five-sphere 80 nm case.')
    for shift in (4, 8, 16):
        parser.add_argument(f'--shift{shift}', required=True, help='Saved case name or directory.')
    parser.add_argument('--name', required=True)
    parser.add_argument('--results-root', type=Path, default=PROJECT_ROOT/'data/results')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT/'data/comparisons')
    parser.add_argument('--probes-nm', default='-200,-40,48,200', help='Use --probes-nm=-200,-40,48,200')
    args = parser.parse_args(argv)
    runs = []
    for shift, name in zip(SHIFTS_NM, (args.perfect,args.shift4,args.shift8,args.shift16)):
        path = Path(name).expanduser()
        directory = path.resolve() if path.is_dir() else args.results_root.expanduser().resolve()/safe_simulation_name(name)
        runs.append(shared.read_run(directory, 5, expected_spacing_nm=80, expected_central_shift_nm=shift))
    probes, profiles = compare_displacements(runs, [float(x) for x in args.probes_nm.split(',') if x.strip()])
    output = args.output_root.expanduser().resolve()/safe_simulation_name(args.name)
    save_comparison(runs, probes, profiles, output)
    print(f'Saved displacement comparison: {output}')
    print('Reused the perfect chain; all source result folders were read only.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f'Displacement comparison failed: {exc}') from exc
