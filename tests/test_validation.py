import csv
import json
from pathlib import Path

import numpy as np
import pytest

from surface_plasmons.notes_theory import green, polarizability, solve
from surface_plasmons.project import normalise_project_config, make_simulation_config
from surface_plasmons.validation import ValidationError, _theory_check, run_validation
from surface_plasmons.workflow import StageFields


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def synthetic(tmp_path, *, multiplier=1, polarisation='Ex', substrate=1):
    config = normalise_project_config({
        'simulation_name': 'single',
        'geometry': {'particle_count': 1, 'particle_radius_um': .005, 'substrate_material': 'air'},
        'source': {'polarisation': polarisation},
        'measurement': {'probe_margin_um': .006, 'probe_line_y_um': .005},
        'numerics': {'broadband_resolution': 1600},
    })
    directory = tmp_path / 'simulation'
    directory.mkdir()
    (directory/'config_resolved.json').write_text(json.dumps(config))
    fields, eps = [], []
    for wavelength, ep_real in zip((480, 500, 520, 540, 560), (-1, -1.5, -2, -2.5, -3)):
        ep = ep_real + .2j
        # Independently evaluate the two scalar formulas printed in the notes.
        alpha = 4*np.pi*.005**3*(ep-1)/(ep+2)
        for x in (-.011, -.009, -.007, .007, .009, .011):
            field = multiplier * alpha * (2 if polarisation == 'Ex' else -1) / (4*np.pi*abs(x)**3)
            row = {'wavelength_nm': wavelength, 'x_um': x, 'y_um': .005, 'z_um': 0}
            for component in ('Ex', 'Ey', 'Ez'):
                reference = 1 if component == polarisation else 0
                value = reference + (field if component == polarisation else 0)
                row.update({f'normalised_{component}_real': float(np.real(value)),
                            f'normalised_{component}_imag': float(np.imag(value)),
                            f'normalised_reference_{component}_real': reference,
                            f'normalised_reference_{component}_imag': 0})
            fields.append(row)
        for role, value in (('particle', ep), ('background', 1), ('substrate', substrate)):
            eps.append({'material_role': role, 'wavelength_nm': wavelength,
                        'epsilon_real': float(np.real(value)), 'epsilon_imag': float(np.imag(value))})
    _csv(directory/'frequency_scan/field_strength_vs_position_and_frequency.csv', fields)
    _csv(directory/'frequency_scan/dielectric_function_vs_frequency.csv', eps)
    return directory, config, fields


def test_scalar_fields_include_correct_transverse_complex_sign():
    q = np.array([1+2j, 0, 0])
    assert green([.02, 0, 0]) @ q == pytest.approx(2*q/(4*np.pi*.02**3))
    assert green([0, .02, 0]) @ q == pytest.approx(-q/(4*np.pi*.02**3))
    # Host-medium prefactors cancel correctly between alpha and G.
    assert polarizability(.005, -4+.1j, 2) == pytest.approx(4*np.pi*2*.005**3*(-6+.1j)/(.1j))


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('axis,coefficient', [(0, 2), (2, -1)])
def test_pair_symmetric_and_antisymmetric_closed_forms(sign, axis, coefficient):
    alpha, distance = 1e-6+2e-7j, .02
    incident = np.zeros((2, 3), dtype=complex)
    incident[:, axis] = [1, sign]
    actual, residual = solve([[-distance/2, 0, 0], [distance/2, 0, 0]], alpha, incident)
    coupling = coefficient/(4*np.pi*distance**3)
    expected = alpha/(1-sign*alpha*coupling)
    assert actual[:, axis] == pytest.approx([expected, sign*expected])
    assert residual < 1e-12


@pytest.mark.parametrize('pol', ['Ex', 'Ez'])
def test_exact_notes_data_pass_without_rescaling(tmp_path, pol):
    directory, config, _ = synthetic(tmp_path, polarisation=pol)
    report = _theory_check(config, directory, directory/'validation')
    assert report['agreement_status'] == 'passed'
    assert report['applicability_status'] == 'passed'
    assert report['complex_scattered_field_relative_l2_error'] < 1e-12
    assert report['independent_curve_rescaling'] is False
    assert report['maximum_dipole_fit_relative_residual'] < 1e-12
    assert report['meep_peak']['wavelength_nm'] == 520
    assert not (directory/'validation/model_hierarchy_spectrum.csv').exists()


@pytest.mark.parametrize('multiplier', [2, 1j, -1])
def test_amplitude_and_phase_mismatch_cannot_be_hidden(tmp_path, multiplier):
    directory, config, _ = synthetic(tmp_path, multiplier=multiplier)
    report = _theory_check(config, directory, directory/'validation')
    assert report['agreement_status'] == 'failed'
    assert report['complex_scattered_field_relative_l2_error'] > .5


def test_host_interface_is_skipped_instead_of_adding_unrequested_model(tmp_path):
    directory, config, _ = synthetic(tmp_path, substrate=2)
    report = _theory_check(config, directory, directory/'validation')
    assert report['status'] == 'skipped'
    assert 'No substrate correction' in report['reason']


def test_identical_validation_reruns_are_idempotent_and_changed_settings_refused(tmp_path):
    directory, _, _ = synthetic(tmp_path)
    first = run_validation(directory)
    second = run_validation(directory)
    assert first == second
    assert len(list((directory/'validation').iterdir())) == 1
    with pytest.raises(ValidationError, match='different settings'):
        run_validation(directory, tolerance=.1)
    with pytest.raises(ValidationError, match='hierarchy'):
        run_validation(directory, checks=['hierarchy'])


def test_band_edge_peak_never_becomes_a_pass(tmp_path):
    directory, config, fields = synthetic(tmp_path)
    rows = [r for r in fields if r['wavelength_nm'] <= 520]
    _csv(directory/'frequency_scan/field_strength_vs_position_and_frequency.csv', rows)
    report = _theory_check(config, directory, directory/'validation')
    assert report['agreement_status'] == 'inconclusive_peak_not_resolved'
    assert report['overall_status'] == 'not_validated'


def test_resolution_time_and_resume_use_actual_new_configurations(tmp_path, monkeypatch):
    directory, config, raw = synthetic(tmp_path)
    calls = []
    worker_counts = []

    def fake_run(candidate, workspace, **kwargs):
        calls.append(candidate)
        worker_counts.append(kwargs["workers"])
        wavelengths = np.asarray([480.,500.,520.,540.,560.])
        data = {c:[] for c in ('Ex','Ey','Ez')}
        for wave in wavelengths:
            rows = [r for r in raw if r['wavelength_nm'] == wave]
            for c in data:
                data[c].append([complex(r[f'normalised_{c}_real'], r[f'normalised_{c}_imag']) for r in rows])
        fields = {c:np.asarray(v)[:,:,None,None] for c,v in data.items()}
        coordinates = {'x':np.asarray([-.011,-.009,-.007,.007,.009,.011]), 'y':np.asarray([.005]), 'z':np.asarray([0])}
        actual = StageFields(1000/wavelengths, fields, coordinates, workspace.directory)
        reference = StageFields(1000/wavelengths, {c:np.ones_like(v) if c=='Ex' else np.zeros_like(v) for c,v in fields.items()}, coordinates, workspace.directory)
        return None, actual, reference, {}

    monkeypatch.setattr('surface_plasmons.workflow.run_broadband_stage', fake_run)
    output = run_validation(directory, validation_name='numerics', checks=['resolution','time'],
                            resolutions=[1200,1600], time_factors=[1,2], workers=2)
    report = json.loads((output/'validation_summary.json').read_text())
    assert report['resolution']['convergence_status'] == 'passed'
    assert report['time']['convergence_status'] == 'passed'
    # Resolution 1600 and time factor 1 reuse the saved baseline.
    assert len(calls) == 2
    assert worker_counts == [2, 2]
    assert calls[-1]['numerics']['maximum_run_time'] == 200
    assert calls[-1]['numerics']['minimum_run_time'] == 40
    assert calls[-1]['numerics']['dft_decay_tolerance'] == pytest.approx(5e-6)
    run_validation(directory, validation_name='numerics', checks=['resolution','time'],
                   resolutions=[1200,1600], time_factors=[1,2])
    assert len(calls) == 2


def test_project_numerics_and_centreline_monitor_reach_simulation_config(tmp_path):
    config = normalise_project_config({'geometry': {'particle_count':1},
        'source': {'span_pml':True}, 'measurement':{'probe_line_y_um':.04},
        'numerics':{'broadband_resolution':400, 'nfreq':21, 'maximum_run_time':250}})
    result = make_simulation_config(config, experiment_directory=tmp_path, stage_name='test')
    assert result['resolution'] == 400
    assert result['output']['field_nfreq'] == 21
    assert result['maximum_run_time'] == 250
    assert result['output']['field_region_center_um'][1] == .04
    assert result['source']['span_pml'] is True


def test_pair_report_exports_both_modes_but_only_driven_mode_is_tested(tmp_path):
    directory, config, _ = synthetic(tmp_path)
    config['geometry']['particle_count'] = 2
    config['geometry']['spacing_um'] = .02
    fields = []
    for wavelength, ep_real in zip((480, 500, 520, 540, 560), (-1, -1.5, -2, -2.5, -3)):
        ep = ep_real+.2j
        alpha = 4*np.pi*.005**3*(ep-1)/(ep+2)
        coupling = 2/(4*np.pi*.02**3)
        q = alpha/(1-alpha*coupling)
        for x in (-.025, -.022, 0, .022, .025):
            value = sum(2*q/(4*np.pi*abs(x-center)**3) for center in (-.01, .01))
            row = {'wavelength_nm': wavelength, 'x_um':x, 'y_um':.005, 'z_um':0}
            for c in ('Ex','Ey','Ez'):
                total = 1+value if c=='Ex' else 0
                row.update({f'normalised_{c}_real':float(np.real(total)), f'normalised_{c}_imag':float(np.imag(total)),
                            f'normalised_reference_{c}_real':int(c=='Ex'), f'normalised_reference_{c}_imag':0})
            fields.append(row)
    _csv(directory/'frequency_scan/field_strength_vs_position_and_frequency.csv', fields)
    report = _theory_check(config, directory, directory/'validation')
    assert report['complex_scattered_field_relative_l2_error'] < 1e-12
    peaks = json.loads((directory/'validation/notes_pair_mode_peaks.json').read_text())
    assert len(peaks) == 4
    tested = [r for r in peaks if r['meep_mode_tested']]
    assert [(r['orientation'], r['mode']) for r in tested] == [('longitudinal', 'symmetric')]


def test_sequence_cases_share_phase_and_have_correct_surface_gap():
    from scripts.run_notes_sequence import cases
    from surface_plasmons.project import load_project_config
    base = load_project_config(Path(__file__).resolve().parents[1]/'configs/notes_single_sphere.json')
    entries = cases(base, 'test')
    assert len(entries) == 8
    for kind, orientation, gap, config in entries:
        assert config['source']['polarisation'] == ('Ex' if orientation == 'longitudinal' else 'Ez')
        assert config['measurement']['probe_line_y_um'] == config['geometry']['particle_radius_um']
        if kind == 'pair':
            assert config['geometry']['spacing_um']-2*config['geometry']['particle_radius_um'] == pytest.approx(gap/1000)


def test_all_sequence_orders_single_convergence_before_pair_runs(tmp_path, monkeypatch):
    from scripts import run_notes_sequence as sequence
    history = []

    def ensure(config, root, cache_enabled):
        target = root / config['simulation_name']
        target.mkdir()
        (target/'config_resolved.json').write_text('{}')
        return target

    def validate(directory, *, validation_name, **kwargs):
        history.append(('pair' if '_pair_' in directory.name else 'single', validation_name))
        return directory

    monkeypatch.setattr(sequence, 'ensure_simulation', ensure)
    monkeypatch.setattr(sequence, 'run_validation', validate)
    monkeypatch.setattr(sequence, 'write_summary', lambda *args: None)
    monkeypatch.setattr('sys.argv', ['run_notes_sequence.py', '--stage', 'all', '--results-root', str(tmp_path)])
    sequence.main()
    assert history[:4] == [('single','notes_comparison')]*2 + [('single','notes_convergence')]*2
    assert history[4:10] == [('pair','notes_comparison')]*6
    assert history[10:] == [('pair','notes_convergence')]*6

