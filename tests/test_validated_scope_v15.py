"""Strict default execution boundary; saved-data parsing remains unrestricted."""
from pathlib import Path
from copy import deepcopy
import pytest

from surface_plasmons.project import load_project_config, normalise_project_config, make_simulation_config
from surface_plasmons.simulation import validate_config, create_simulation, run_simulation
from surface_plasmons.validated_scope import (ValidatedScopeError, enforce_validated_project,
                                              enforce_validated_simulation)


def config(pol='ex'):
    return load_project_config(Path(__file__).parents[1]/f'configs/validated_sphere_{pol}_v15.json')


@pytest.mark.parametrize('pol',['ex','ez'])
@pytest.mark.parametrize('resolution,padding,min_time,max_time,tolerance',[
    (300,.12,20.,100.,1e-5),(400,.12,20.,100.,1e-5),
    (300,.22,20.,100.,1e-5),(300,.12,122.,610.,1e-5/6.1)])
def test_only_tested_numerical_combinations_are_accepted(tmp_path,pol,resolution,padding,min_time,max_time,tolerance):
    c=config(pol);c['simulation_name']='custom safe name'
    c['numerics'].update(broadband_resolution=resolution,padding_um=padding,
        minimum_run_time=min_time,maximum_run_time=max_time,dft_decay_tolerance=tolerance)
    enforce_validated_project(c)
    for reference in [False,True]:
        raw=make_simulation_config(c,experiment_directory=tmp_path,stage_name='test',reference=reference)
        enforce_validated_simulation(validate_config(raw))


@pytest.mark.parametrize('section,key,value',[
    ('geometry','particle_radius_um',.025),('geometry','particle_material','au'),
    ('geometry','substrate_material','glass_constant'),('geometry','background_material','air'),
    ('geometry','particle_shape','cube'),('geometry','particle_count',2),
    ('source','polarisation','Ey'),('source','propagation_direction',1),
    ('source','wavelength_range_nm',[1050,1200]),('source','span_pml',False),
    ('measurement','probe_line_y_um',.025),('measurement','probe_margin_um',.05),
    ('measurement','normalization_mode','pointwise_amplitude_reference'),
    ('measurement','validation_region','all_exterior'),('measurement','field_map_frequencies','auto'),
    ('numerics','broadband_resolution',350),('numerics','nfreq',41),
    ('numerics','mirror_symmetry',False),('numerics','courant',.4),
    ('numerics','pml_thickness_um',.3),('numerics','minimum_run_time',40),
])
def test_project_changes_outside_scope_are_rejected(section,key,value):
    c=config();c[section][key]=value
    with pytest.raises((ValidatedScopeError,ValueError)):
        enforce_validated_project(c)


def test_no_arbitrary_combination_of_individually_tested_values():
    c=config();c['numerics'].update(broadband_resolution=400,padding_um=.22)
    with pytest.raises(ValidatedScopeError,match='tested tuple'):
        enforce_validated_project(c)


@pytest.mark.parametrize('method',[create_simulation,run_simulation])
def test_low_level_entrypoints_cannot_bypass_guard(tmp_path,monkeypatch,method):
    import surface_plasmons.simulation as simulation
    c=config();raw=make_simulation_config(c,experiment_directory=tmp_path/'uncreated',stage_name='test')
    raw['geometry_parameters']['radius']=.03
    def forbidden():
        pytest.fail('Meep must not be imported for a rejected configuration')
    monkeypatch.setattr(simulation,'_import_meep',forbidden)
    with pytest.raises(ValidatedScopeError,match='radius'):
        method(raw)
    assert not (tmp_path/'uncreated').exists()


def test_cli_field_map_override_rejected_before_output_creation(tmp_path,monkeypatch):
    from surface_plasmons.workflow import run_project
    with pytest.raises(ValidatedScopeError,match='field maps'):
        run_project(config(),results_root=tmp_path/'results',field_map_selection='auto')
    assert not (tmp_path/'results').exists()


def test_historical_gold_config_can_still_be_loaded_and_analysed():
    old=load_project_config(Path(__file__).parents[1]/'configs/notes_quasistatic_single_sphere_v5_longitudinal.json')
    assert normalise_project_config(old)['geometry']['particle_material']=='au'
    from surface_plasmons.validation import _baseline_analysis_signature
    assert _baseline_analysis_signature(old)
    with pytest.raises(ValidatedScopeError):
        enforce_validated_project(old)


def test_sweep_workers_reject_untested_candidates_before_meep(tmp_path,monkeypatch):
    from surface_plasmons.sweep import parameter_sweep
    import surface_plasmons.simulation as simulation
    raw=make_simulation_config(config(),experiment_directory=tmp_path/'runs',stage_name='test')
    def forbidden():
        pytest.fail('A rejected sweep candidate must not import Meep')
    monkeypatch.setattr(simulation,'_import_meep',forbidden)
    result=parameter_sweep(raw,{'geometry_parameters.radius':[.025,.03]},
        workers=1,continue_on_error=True,resume=False,output_root=str(tmp_path/'sweep'))
    assert len(result.runs)==2
    assert all(r.status=='failed' and r.error_type=='ValidatedScopeError' for r in result.runs)
