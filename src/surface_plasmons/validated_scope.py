"""Execution guard for the exact dielectric benchmarks tested through v14.

This is a reproducibility restriction, not a security boundary or a claim that
every output is validated. Historical configurations remain readable. New solver
construction and execution always enforce this scope, including reference runs.
"""
from __future__ import annotations

import math


class ValidatedScopeError(ValueError):
    """A requested new calculation exceeds the tested benchmark scope."""


def _same(actual, expected):
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)):
        return (not isinstance(actual, bool) and isinstance(actual, (int, float))
                and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12))
    if isinstance(expected, (list, tuple)):
        return (isinstance(actual, (list, tuple)) and len(actual) == len(expected)
                and all(_same(a, b) for a, b in zip(actual, expected)))
    return actual == expected


def _require(actual, expected, name, errors):
    if not _same(actual, expected):
        errors.append(f"{name}: expected {expected!r}, received {actual!r}")


def _reject(errors):
    if errors:
        raise ValidatedScopeError(
            "Validated-only mode rejected this new simulation:\n- " + "\n- ".join(errors)
            + "\nUse configs/validated_sphere_ex_v15.json or validated_sphere_ez_v15.json. "
              "Only run names, Ex/Ez, and the four tested numerical cases are editable. "
              "Saved results remain readable; no new validation claim is inherited automatically.")


def enforce_validated_project(config, *, field_map_selection=None):
    """Check project-only settings before creating any experiment directory."""
    scope = config['numerics'].get('execution_scope', 'validated_control')
    if scope == 'research_dielectric_line':
        from .research_scope import enforce_research_project
        return enforce_research_project(config, field_map_selection=field_map_selection)
    if scope != 'validated_control':
        raise ValidatedScopeError('Unknown numerics.execution_scope: ' + repr(scope))
    from .project import make_simulation_config
    from .simulation import validate_config
    errors = []
    geometry = config['geometry']
    for name, expected in {'particle_shape':'sphere','particle_count':1,
            'particle_material':'glass_constant','background_material':'vacuum',
            'substrate_material':'vacuum','particle_radius_um':.02,'cube_size_um':.08,
            'spacing_um':.08,'substrate_thickness_um':.02,'surface_y_um':0.,
            'particle_surface_gap_um':0.,'defects':[]}.items():
        _require(geometry.get(name),expected,'geometry.'+name,errors)
    measurement = config['measurement']
    for name, expected in {'probe_margin_um':.04,'probe_line_y_um':.02,
            'probe_offset_from_particle_surface_um':0.,'normalization_mode':'global_complex_reference',
            'field_map_frequencies':'none','validation_region':'electrostatic_near_zone',
            'validation_benchmark':'nonresonant_dielectric'}.items():
        _require(measurement.get(name),expected,'measurement.'+name,errors)
    if field_map_selection is not None and field_map_selection != 'none':
        errors.append('field maps were not part of the validated output; select none')
    _require(config['precision'],'fast','precision',errors)
    _reject(errors)
    simulation = make_simulation_config(config,experiment_directory='data/results/scope_preview',
                                        stage_name='frequency_scan/particle')
    enforce_validated_simulation(validate_config(simulation))


def enforce_validated_simulation(cfg):
    """Gate low-level execution as well as project runs and sweep candidates."""
    if 'execution:research_dielectric_line' in cfg.tags:
        from .research_scope import enforce_research_simulation
        return enforce_research_simulation(cfg)
    errors = []
    _require(cfg.mirror_axes, None, 'mirror_axes', errors)
    for name, expected in {'geometry_name':'particle_line_on_substrate','dimensions':3,
            'pml_thickness_um':.2,'courant':.45,'k_point':(0.,0.,0.),
            'force_complex_fields':False,'mirror_symmetry':True,'stop_method':'dft_decay'}.items():
        _require(getattr(cfg,name),expected,name,errors)
    if cfg.pml_axes is not None:
        errors.append('pml_axes: keep the default PML on all axes')
    expected_geometry = {'particle':'glass_constant','background':'vacuum','substrate':'vacuum',
            'particle_shape':'sphere','particle_count':1,'radius':.02,'cube_size':.08,
            'spacing':.08,'substrate_thickness':.02,'surface_y':0.,'particle_surface_gap':0.,
            'defects':[],'allow_out_of_range':False,'allow_unstable':False}
    for name, expected in expected_geometry.items():
        _require(cfg.geometry_parameters.get(name),expected,'geometry_parameters.'+name,errors)
    _require(cfg.geometry_parameters.get('center',(0.,0.,0.)),(0.,0.,0.),'geometry_parameters.center',errors)
    permitted_geometry = set(expected_geometry) | {'center','include_particles'}
    if set(cfg.geometry_parameters)-permitted_geometry:
        errors.append('geometry_parameters contains untested extra settings')
    if not isinstance(cfg.geometry_parameters.get('include_particles',True),bool):
        errors.append('include_particles must be boolean (false is reserved for the empty reference)')
    for name, expected in {'kind':'gaussian','wavelength_range_um':(1.,1.2),'axis':'y',
            'direction':-1,'amplitude':1.,'center_um':None,'size_um':None,
            'offset_from_pml_um':.05,'cutoff':5.,'is_integrated':True,'span_pml':True}.items():
        _require(getattr(cfg.source,name),expected,'source.'+name,errors)
    if cfg.source.component not in {'Ex','Ez'}:
        errors.append('source.component: only Ex or Ez was tested')
    # These are discrete measured cases, not arbitrary combinations or a range.
    cases = [(300,.12,20.,100.,1e-5),(400,.12,20.,100.,1e-5),
             (300,.22,20.,100.,1e-5),(300,.12,122.,610.,1e-5/6.1)]
    actual = (cfg.resolution,cfg.padding_um,cfg.minimum_run_time,
              cfg.maximum_run_time,cfg.dft_decay_tolerance)
    if not any(_same(actual,case) for case in cases):
        errors.append('numerics: use one tested tuple (resolution, padding, min_time, max_time, DFT_tolerance): '+repr(cases))
    size = (.04+2*(cfg.padding_um+.2),.06+2*(cfg.padding_um+.2),.04+2*(cfg.padding_um+.2))
    _require(cfg.cell_size_um,size,'cell_size_um',errors)
    _require(cfg.flux.enabled,False,'flux.enabled',errors)
    for name, expected in {'save_fields':True,'save_images':False,'save_epsilon':False,'save_spectra':False,
            'field_region_center_um':(0.,.02,0.),'field_region_size_um':(.12,0.,0.),
            'field_nfreq':21,'field_wavelength_range_um':(1.,1.2)}.items():
        _require(getattr(cfg.output,name),expected,'output.'+name,errors)
    if sorted(cfg.output.field_components) != ['Ex','Ey','Ez']:
        errors.append('output.field_components must be Ex, Ey, Ez')
    _reject(errors)
