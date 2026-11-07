"""Compare saved N=1,2,3,5 fields at offsets from the leftmost sphere centre.

No Meep runs are started and source result folders are never modified.
    python scripts/compare_particle_counts_v19.py --single control_sphere_v16_ex --pair pair80_ex --three chain3_80_ex --five chain5_80_ex --name particle_counts_ex

The old N=1 and N=2 centre-line monitors have no common eligible exterior
points at equal absolute x. This analysis instead fixes xi=x-x_leftmost,
which compares the same observation location relative to the chain's end.
Illumination propagates along -y, so this x translation changes no incident
phase. Default xi offsets are -45,-40,-35,-30 nm, all outside the spheres.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1] / 'src'
if not (_LOCAL_SOURCE / 'surface_plasmons' / '__init__.py').is_file():
    raise SystemExit('Place this file in scripts/ inside your existing project.')
sys.path.insert(0, str(_LOCAL_SOURCE))

import numpy as np
from surface_plasmons.particle_line import particle_line_positions
from surface_plasmons.project import PRECISION_MODES, geometry_parameters, normalise_project_config
from surface_plasmons.results import safe_simulation_name
from surface_plasmons.validation import _saved_convergence_reports

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ('Ex', 'Ey', 'Ez')


def read_run(directory, expected_count, *, expected_spacing_nm=80, expected_central_shift_nm=None,
             expected_vacancy_index=None):
    if expected_central_shift_nm is not None and expected_vacancy_index is not None:
        raise ValueError('Request either a displacement or a vacancy check.')
    config = normalise_project_config(json.loads((directory / 'config_resolved.json').read_text(encoding='utf-8')))
    g, source, measurement = config['geometry'], config['source'], config['measurement']
    if (g['particle_count'] != expected_count or g['particle_shape'] != 'sphere'
            or (g.get('defects') and expected_central_shift_nm is None and expected_vacancy_index is None)
            or g['particle_material'] != 'glass_constant'
            or g['background_material'] != 'vacuum' or g['substrate_material'] != 'vacuum'
            or not np.isclose(g['particle_radius_um'], .02)
            or (expected_count > 1 and not np.isclose(g['spacing_um'], expected_spacing_nm / 1000))):
        raise ValueError(f'{directory}: expected a perfect N={expected_count}, 20 nm radius glass/vacuum chain at {expected_spacing_nm:g} nm spacing.')
    if (source['propagation_axis'] != 'y' or source['propagation_direction'] != -1
            or source['polarisation'] not in ('Ex', 'Ez')
            or not np.allclose(source['wavelength_range_nm'], [1000, 1200])
            or measurement['normalization_mode'] != 'global_complex_reference'):
        raise ValueError(f'{directory}: incompatible source, wavelength band or field normalisation.')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed':
        raise ValueError(f'{directory}: solver results are not marked completed.')
    path = directory / 'frequency_scan/field_strength_vs_position_and_frequency.csv'
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'{path}: no saved field data.')
    positions = particle_line_positions({**geometry_parameters(config, include_particles=True), 'center': (0., 0., 0.)})
    centres = np.array([(p.actual_x_um, p.actual_y_um, p.actual_z_um) for p in positions])
    if expected_vacancy_index is not None:
        if expected_count != 5 or expected_spacing_nm != 80 or expected_vacancy_index not in (2, 4):
            raise ValueError('The vacancy comparison requires nominal N=5, d=80 nm and missing index 2 or 4.')
        expected = np.delete((np.arange(5)-2)*.08, expected_vacancy_index)
        if (len(centres) != 4 or not np.allclose(centres[:, 0], expected, rtol=0, atol=1e-12)
                or not np.allclose(centres[:, 1:], [.02, 0], rtol=0, atol=1e-12)):
            raise ValueError(f'{directory}: occupied positions do not match missing index {expected_vacancy_index}.')
    if expected_central_shift_nm is not None:
        if (expected_count != 5 or expected_spacing_nm != 80
                or expected_central_shift_nm not in (0, 4, 8, 16)):
            raise ValueError('The central-defect comparison requires N=5, d=80 nm and shifts 0,4,8,16 nm.')
        expected = (np.arange(5)-2)*.08
        expected[2] += expected_central_shift_nm/1000
        if (len(centres) != 5 or not np.allclose(centres[:, 0], expected, rtol=0, atol=1e-12)
                or not np.allclose(centres[:, 1:], [.02, 0], rtol=0, atol=1e-12)):
            raise ValueError(f'{directory}: the occupied sphere positions do not match a +{expected_central_shift_nm:g} nm shift of index 2.')
    coordinates = np.array([[float(r[f'{c}_um']) for c in 'xyz'] for r in rows])
    if not np.allclose(coordinates[:, 1:], centres[0, 1:], rtol=0, atol=1e-8):
        raise ValueError(f'{directory}: expected the same y/z plane as the sphere centres.')
    wavelengths = np.array([float(r['wavelength_nm']) for r in rows])
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError(f'{directory}: nonfinite sample coordinates or wavelengths.')
    values = []
    for row in rows:
        vector = []
        for component in COMPONENTS:
            if f'normalised_scattered_{component}_real' in row:
                v = complex(float(row[f'normalised_scattered_{component}_real']), float(row[f'normalised_scattered_{component}_imag']))
            else:
                v = (complex(float(row[f'normalised_{component}_real']), float(row[f'normalised_{component}_imag']))
                     - complex(float(row[f'normalised_reference_{component}_real']), float(row[f'normalised_reference_{component}_imag'])))
            vector.append(v)
        values.append(vector)
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{directory}: nonfinite complex scattered fields.')
    profile = {**PRECISION_MODES[config['precision']], **config['numerics']}
    convergence = _saved_convergence_reports(directory, config, directory / 'not_a_validation_folder')
    automatic_path = directory / 'automatic_theory_validation.json'
    automatic = json.loads(automatic_path.read_text(encoding='utf-8')) if automatic_path.is_file() else {}
    theory = automatic.get('validation_reports', {}).get('theory', {})
    return dict(directory=directory, config=config, count=expected_count, coordinates=coordinates,
                spacing_nm=float(g['spacing_um']*1000) if expected_count > 1 else None,
                central_shift_nm=float(expected_central_shift_nm) if expected_central_shift_nm is not None else None,
                active_count=len(centres), vacancy_index=expected_vacancy_index,
                wavelengths=wavelengths, grid=np.unique(wavelengths), values=values, centres=centres,
                first_x=float(np.min(centres[:, 0])), radius=float(g['particle_radius_um']),
                resolution=float(profile['broadband_resolution']),
                statuses=dict(theory_agreement=theory.get('agreement_status', 'not_available'),
                    model_applicability=theory.get('applicability_status', 'not_available'),
                    numerical_convergence={kind: convergence.get(kind, {}).get('convergence_status', 'not_established_in_this_folder')
                                           for kind in ('resolution', 'domain', 'time')}))


def compare_runs(runs, offsets_nm, *, expected_counts=(1, 2, 3, 5)):
    if [r['count'] for r in runs] != list(expected_counts):
        raise ValueError(f'Supply particle counts in order {list(expected_counts)}.')
    offsets = np.asarray(offsets_nm, dtype=float) / 1000
    if not len(offsets) or not np.all(np.isfinite(offsets)) or np.any(offsets >= 0):
        raise ValueError('Offsets must be finite, negative distances to the left of the first sphere.')
    offsets = np.unique(offsets)
    grid = runs[0]['grid']
    if not len(grid):
        raise ValueError('No frequency samples found.')
    pol = runs[0]['config']['source']['polarisation']
    coarse_resolution = min(r['resolution'] for r in runs)
    records = []
    for run in runs:
        if run['config']['source']['polarisation'] != pol:
            raise ValueError('Compare one polarisation at a time; Ex and Ez cannot be mixed.')
        if run['grid'].shape != grid.shape or not np.allclose(run['grid'], grid, rtol=1e-10, atol=1e-6):
            raise ValueError('The saved wavelength grids differ; these files cannot be overlaid directly.')
        distances = np.linalg.norm(run['coordinates'][:, None, :] - run['centres'][None, :, :], axis=2)
        safe = np.all(distances >= run['radius'] + 2 / coarse_resolution - 1e-12, axis=1)
        safe &= run['coordinates'][:, 0] < run['first_x']
        if run['config']['measurement'].get('validation_region') == 'electrostatic_near_zone':
            maximum_k = 2 * np.pi / (min(run['config']['source']['wavelength_range_nm']) / 1000)
            safe &= np.all(maximum_k * distances <= .3 + 1e-12, axis=1)
        for wavelength in grid:
            selected = safe & np.isclose(run['wavelengths'], wavelength, rtol=1e-10, atol=1e-6)
            x = run['coordinates'][selected, 0] - run['first_x']
            order = np.argsort(x)
            x, values = x[order], run['values'][selected][order]
            if len(x) < 2 or np.any(np.diff(x) <= 0):
                raise ValueError(f"{run['directory']}: need distinct eligible left-exterior samples.")
            if offsets.min() < x.min() - 1e-12 or offsets.max() > x.max() + 1e-12:
                raise ValueError(f"N={run['count']}: requested offsets are outside the saved eligible exterior samples "
                                 f"[{1000*x.min():.3f}, {1000*x.max():.3f}] nm. No extrapolation is performed.")
            sampled = np.stack([np.interp(offsets, x, values[:, j].real)
                                + 1j * np.interp(offsets, x, values[:, j].imag) for j in range(3)], axis=-1)
            for offset, field in zip(offsets, sampled):
                row = dict(particle_count=run['count'], polarisation=pol, wavelength_nm=float(wavelength),
                           centre_spacing_nm=run['spacing_nm'],
                           offset_from_leftmost_centre_nm=float(offset*1000),
                           original_x_nm=float((run['first_x']+offset)*1000),
                           y_nm=float(run['centres'][0, 1]*1000), z_nm=float(run['centres'][0, 2]*1000),
                           source_result_folder=str(run['directory']))
                for j, component in enumerate(COMPONENTS):
                    row.update({f'S_{component}_real': float(field[j].real),
                                f'S_{component}_imag': float(field[j].imag),
                                f'S_{component}_magnitude': float(abs(field[j]))})
                row['scattered_vector_magnitude'] = float(np.linalg.norm(field))
                records.append(row)
    return records


def save_comparison(runs, records, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if output.exists():
        raise ValueError(f'Comparison folder already exists: {output}. Choose a new --name.')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'particle_count_fields.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    pol = runs[0]['config']['source']['polarisation']
    offsets = sorted({r['offset_from_leftmost_centre_nm'] for r in records})
    probe = min(offsets, key=lambda x: abs(x + 40))
    wavelengths = runs[0]['grid']
    selected_waves = sorted({float(wavelengths[0]), float(wavelengths[np.argmin(abs(wavelengths-1100))]), float(wavelengths[-1])})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n in (1, 2, 3, 5):
        values = sorted([r for r in records if r['particle_count']==n and r['offset_from_leftmost_centre_nm']==probe], key=lambda r:r['wavelength_nm'])
        for axis, part in zip(axes, ('real', 'imag')):
            axis.plot([r['wavelength_nm'] for r in values], [r[f'S_{pol}_{part}'] for r in values], label=f'N={n}')
            axis.set(xlabel='Vacuum wavelength (nm)', ylabel=f'{part.capitalize()}(scattered {pol} / Eref)')
            axis.grid(alpha=.2)
            axis.legend()
    fig.suptitle(f'Exterior probe: {probe:g} nm from leftmost sphere centre')
    fig.tight_layout()
    fig.savefig(output / 'complex_spectra_by_particle_count.png', dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6, 4))
    for wavelength in selected_waves:
        values = sorted([r for r in records if r['offset_from_leftmost_centre_nm']==probe and r['wavelength_nm']==wavelength], key=lambda r:r['particle_count'])
        axis.plot([r['particle_count'] for r in values], [r[f'S_{pol}_magnitude'] for r in values], 'o-', label=f'{wavelength:.1f} nm')
    axis.set(xlabel='Particle count N', ylabel=f'|scattered {pol} / Eref|', xticks=[1, 2, 3, 5], title=f'Exterior end probe: {probe:g} nm')
    axis.grid(alpha=.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'field_magnitude_vs_particle_count.png', dpi=180)
    plt.close(fig)
    summary = dict(coordinate_frame='xi = x - leftmost sphere centre x; not equal absolute laboratory x',
                   measurement='complex scattered E / common complex Eref; no intensity squaring',
                   interpolation='real and imaginary parts interpolated separately inside eligible left-exterior samples',
                   offsets_nm=offsets, primary_plot_probe_nm=probe, polarisation=pol,
                   no_new_simulations_started=True,
                   numerical_acceptance_note='A count comparison does not establish convergence; review each run separately.',
                   cases=[dict(particle_count=r['count'], result_folder=str(r['directory']),
                               leftmost_sphere_x_nm=r['first_x']*1000, **r['statuses']) for r in runs])
    (output / 'comparison_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    for flag in ('single', 'pair', 'three', 'five'):
        parser.add_argument('--'+flag, required=True, help='Saved simulation name or directory.')
    parser.add_argument('--name', required=True, help='New comparison-folder name.')
    parser.add_argument('--results-root', type=Path, default=PROJECT_ROOT / 'data/results')
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'data/comparisons')
    parser.add_argument('--offsets-nm', default='-45,-40,-35,-30', help='Negative offsets; use --offsets-nm=-45,-40,...')
    args = parser.parse_args(argv)
    runs = []
    for n, name in zip((1, 2, 3, 5), (args.single, args.pair, args.three, args.five)):
        path = Path(name).expanduser()
        directory = path.resolve() if path.is_dir() else args.results_root.expanduser().resolve() / safe_simulation_name(name)
        runs.append(read_run(directory, n))
    records = compare_runs(runs, [float(v) for v in args.offsets_nm.split(',') if v.strip()])
    output = args.output_root.expanduser().resolve() / safe_simulation_name(args.name)
    save_comparison(runs, records, output)
    print(f'Saved comparison: {output}')
    print('Compared equal offsets from the leftmost sphere centre. Existing results were read only.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise SystemExit(f'Particle-count comparison failed: {exc}') from exc
