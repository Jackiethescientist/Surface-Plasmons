"""Nonresonant acceptance must never weaken the dispersive-gold benchmark."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from surface_plasmons.project import ProjectConfigError, load_project_config, normalise_project_config, make_simulation_config
from surface_plasmons.simulation import benchmark_mirror_planes, validate_config
from surface_plasmons.workflow import StageFields
from surface_plasmons.validation import (ValidationError, _theory_check, _convergence_metrics,
    _convergence_passed, _baseline_analysis_signature, run_validation)
from test_validation import synthetic, _csv


def dielectric_data(tmp_path, polarisation='Ex', multiplier=1):
    directory, config, fields = synthetic(tmp_path, polarisation=polarisation)
    config['geometry'].update(particle_material='glass_constant', background_material='vacuum', substrate_material='vacuum')
    config['measurement']['validation_benchmark'] = 'nonresonant_dielectric'
    config = normalise_project_config(config)
    eps = []
    alpha = 4*np.pi*.005**3*(2.25-1)/(2.25+2)
    for row in fields:
        field = multiplier*alpha*(2 if polarisation=='Ex' else -1)/(4*np.pi*abs(row['x_um'])**3)
        for c in ('Ex','Ey','Ez'):
            value = (1+field) if c==polarisation else 0j
            row[f'normalised_{c}_real'],row[f'normalised_{c}_imag'] = float(np.real(value)),float(np.imag(value))
    for wave in sorted({r['wavelength_nm'] for r in fields}):
        for role, value in [('particle',2.25),('background',1),('substrate',1)]:
            eps.append(dict(material_role=role,wavelength_nm=wave,epsilon_real=value,epsilon_imag=0))
    _csv(directory/'frequency_scan/field_strength_vs_position_and_frequency.csv',fields)
    _csv(directory/'frequency_scan/dielectric_function_vs_frequency.csv',eps)
    (directory/'config_resolved.json').write_text(json.dumps(config))
    return directory,config,fields,eps


@pytest.mark.parametrize('pol', ['Ex','Ez'])
def test_flat_dielectric_response_passes_fields_but_requires_convergence(tmp_path,pol):
    directory,config,_,_ = dielectric_data(tmp_path,pol)
    report = _theory_check(config,directory,directory/'check')
    assert report['agreement_status']=='passed'
    assert report['agreement_checks']['peak_wavelength']['status']=='not_applicable'
    assert report['agreement_checks']['worst_frequency_complex_field']['value']<1e-12
    assert report['resonance_check_required'] is False
    output = run_validation(directory)
    summary = json.loads((output/'validation_summary.json').read_text())
    assert summary['overall']['status']=='not_validated'
    assert summary['overall']['scope']=='nonresonant_dielectric_pipeline'
    assert 'gold' in summary['overall']['scope_limitation']


@pytest.mark.parametrize('factor',[1.1,1j,-1])
def test_nonresonant_policy_does_not_hide_amplitude_or_phase_errors(tmp_path,factor):
    directory,config,_,_=dielectric_data(tmp_path,multiplier=factor)
    report=_theory_check(config,directory,directory/'check')
    assert report['agreement_status']=='failed'
    assert 'complex_scattered_field' in report['failure_reasons']


def test_one_bad_frequency_fails_even_when_global_average_passes(tmp_path):
    directory,config,fields,_=dielectric_data(tmp_path)
    for r in fields:
        if r['wavelength_nm']==480:
            r['normalised_Ex_real']=1+1.1*(r['normalised_Ex_real']-1)
    _csv(directory/'frequency_scan/field_strength_vs_position_and_frequency.csv',fields)
    report=_theory_check(config,directory,directory/'check')
    assert report['complex_scattered_field_relative_l2_error']<.05
    assert report['agreement_status']=='failed'
    assert 'worst_frequency_complex_field' in report['failure_reasons']


def test_gold_cannot_disable_resonance_checks(tmp_path):
    _,config,_=synthetic(tmp_path)
    config['measurement']['validation_benchmark']='nonresonant_dielectric'
    with pytest.raises(ProjectConfigError,match='cannot disable gold'):
        normalise_project_config(config)


def test_saved_dispersive_data_cannot_masquerade_as_dielectric(tmp_path):
    directory,config,_,eps=dielectric_data(tmp_path)
    eps[0]['epsilon_imag']=.1
    _csv(directory/'frequency_scan/dielectric_function_vs_frequency.csv',eps)
    with pytest.raises(ValidationError,match='constant positive'):
        _theory_check(config,directory,directory/'check')


def test_nonresonant_convergence_accepts_flat_spectrum_and_rejects_local_error():
    waves=np.linspace(1000,1200,21)
    fields=np.ones((21,6,3),complex)
    reference=(400,waves,fields)
    assert not _convergence_passed(_convergence_metrics((300,waves,fields),reference))
    metrics=_convergence_metrics((300,waves,fields*1.01),reference,resonance_required=False)
    assert _convergence_passed(metrics)
    assert metrics['peak_check_status']=='not_applicable'
    bad=fields.copy();bad[0]*=1.1
    metrics=_convergence_metrics((300,waves,bad),reference,resonance_required=False)
    assert metrics['complex_field_relative_l2_change']<.05
    assert not _convergence_passed(metrics)


@pytest.mark.parametrize('pol,phases',[('Ex',(('x',-1),('z',1))),('Ez',(('x',1),('z',-1)))])
def test_new_configs_have_correct_mirrors_and_unchanged_analysis_policy(tmp_path,pol,phases):
    config=load_project_config(Path(__file__).parents[1]/f'configs/dielectric_sphere_{pol.lower()}_v13.json')
    sim=make_simulation_config(config,experiment_directory=tmp_path,stage_name='test')
    assert benchmark_mirror_planes(validate_config(sim))==phases
    assert sim['resolution']==300 and sim['record_database'] is False
    changed=deepcopy(config);changed['measurement'].pop('validation_benchmark')
    assert _baseline_analysis_signature(config)!=_baseline_analysis_signature(changed)


def test_nonresonant_saved_checks_reused_only_with_matching_policy(tmp_path):
    directory,config,_,_=dielectric_data(tmp_path)
    old=directory/'validation/prior';old.mkdir(parents=True)
    (old/'validation_config.json').write_text(json.dumps({'config':config}))
    report={'status':'completed','convergence_status':'passed','resonance_check_required':False,
            'criteria':{'complex_field_change':.05,'worst_frequency_complex_field_change':.05,'maximum_spectral_amplitude_change':.05}}
    (old/'validation_summary.json').write_text(json.dumps({k:report for k in ['resolution','domain','time']}))
    out=run_validation(directory,validation_name='reanalysis')
    summary=json.loads((out/'validation_summary.json').read_text())
    assert summary['overall']['status']=='validated'
    assert summary['overall']['scope']=='nonresonant_dielectric_pipeline'
    report.pop('resonance_check_required')
    (old/'validation_summary.json').write_text(json.dumps({'resolution':report}))
    # Remove the compatible completed reanalysis, so only the invalid prior remains.
    (out/'validation_summary.json').unlink()
    out2=run_validation(directory,validation_name='policy_guard')
    assert json.loads((out2/'validation_summary.json').read_text())['overall']['status']=='not_validated'


def test_all_nonresonant_checks_reuse_baseline_and_resume_new_cases(tmp_path,monkeypatch):
    directory,config,raw,_=dielectric_data(tmp_path)
    calls=[]
    def fake_run(candidate,workspace,**kwargs):
        calls.append(deepcopy(candidate))
        waves=np.asarray(sorted({r['wavelength_nm'] for r in raw}))
        coordinates={'x':np.array(sorted({r['x_um'] for r in raw})),
                     'y':np.array([.005]),'z':np.array([0.])}
        fields={}
        for c in ['Ex','Ey','Ez']:
            fields[c]=np.array([[complex(r[f'normalised_{c}_real'],r[f'normalised_{c}_imag'])
                               for r in raw if r['wavelength_nm']==wave] for wave in waves])[:,:,None,None]
        reference={c:np.ones_like(v) if c=='Ex' else np.zeros_like(v) for c,v in fields.items()}
        return None,StageFields(1000/waves,fields,coordinates,workspace.directory),StageFields(1000/waves,reference,coordinates,workspace.directory),{}
    monkeypatch.setattr('surface_plasmons.workflow.run_broadband_stage',fake_run)
    kwargs=dict(validation_name='complete',checks=['all'],resolutions=[1600,1800],
                paddings_um=[.3,.4],time_factors=[1,2])
    out=run_validation(directory,**kwargs)
    report=json.loads((out/'validation_summary.json').read_text())
    assert len(calls)==3
    assert report['overall']['status']=='validated'
    for kind in ['resolution','domain','time']:
        assert report[kind]['convergence_status']=='passed'
        assert report[kind]['finest_comparison']['peak_check_status']=='not_applicable'
        assert report[kind]['finest_comparison']['source']=='saved_baseline'
    run_validation(directory,**kwargs)
    assert len(calls)==3
