"""Explicit research extension; these configurations inherit no validation pass.

The v15/v16 default boundary remains in validated_scope. This opt-in covers
the proposed dielectric pair/finite-line study using the same materials and
illumination as the single-sphere controls. It is not a new physical model.
"""
from __future__ import annotations

import numpy as np

from .particle_line import particle_line_positions


SCOPE = 'research_dielectric_line'
TAG = 'execution:' + SCOPE


def _require(value, expected, name):
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        matches = (not isinstance(value, bool) and isinstance(value, (int, float))
                   and np.isfinite(value) and np.isclose(value, expected, rtol=1e-10, atol=1e-12))
    else:
        matches = value == expected
    if not matches:
        raise ValueError(f'Research dielectric scope requires {name}={expected!r}; got {value!r}.')


def _geometry(parameters):
    expected = {'particle': 'glass_constant', 'background': 'vacuum', 'substrate': 'vacuum',
                'particle_shape': 'sphere', 'radius': .02, 'surface_y': 0.,
                'particle_surface_gap': 0., 'substrate_thickness': .02,
                'allow_out_of_range': False, 'allow_unstable': False}
    for key, value in expected.items():
        _require(parameters.get(key), value, 'geometry.' + key)
    count = parameters.get('particle_count')
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
        raise ValueError('Research dielectric scope requires 1 to 5 nominal spheres.')
    spacing = parameters.get('spacing')
    if spacing is None or not any(np.isclose(spacing, d, rtol=0, atol=1e-12) for d in (.06, .08, .10)):
        raise ValueError('Research dielectric centre spacing must be 60, 80 or 100 nm.')
    if not np.allclose(parameters.get('center', (0., 0., 0.)), (0., 0., 0.), rtol=0, atol=1e-12):
        raise ValueError('Keep the research line centred at x=z=0.')
    # The shared geometry helper also checks displacement format and overlap.
    positions = particle_line_positions(parameters)
    for particle in positions:
        if (abs(particle.shift_y_um) > 1e-12 or abs(particle.shift_z_um) > 1e-12
                or abs(particle.shift_x_um) > .2 * spacing + 1e-12):
            raise ValueError('Research shifts must be along x and no greater than 20% of spacing.')


def symmetric_research_positions(parameters):
    """Check actual occupied centres, including every vacancy or displacement."""
    points = np.asarray([(p.actual_x_um, p.actual_y_um, p.actual_z_um)
                         for p in particle_line_positions(parameters)])
    for axis in (0, 2):
        reflected = points.copy()
        reflected[:, axis] *= -1
        if any(not np.any(np.all(np.isclose(points, p, rtol=0, atol=1e-12), axis=1)) for p in reflected):
            return False
    return True


def enforce_research_simulation(cfg):
    _require(cfg.geometry_name, 'particle_line_on_substrate', 'geometry_name')
    _require(cfg.dimensions, 3, 'dimensions')
    _geometry(cfg.geometry_parameters)
    _require(cfg.source.kind, 'gaussian', 'source.kind')
    _require(cfg.source.axis, 'y', 'source.axis')
    _require(cfg.source.direction, -1, 'source.direction')
    _require(cfg.source.span_pml, True, 'source.span_pml')
    if cfg.source.component not in {'Ex', 'Ez'}:
        raise ValueError('Research polarisation must be Ex or Ez.')
    if not np.allclose(cfg.source.wavelength_range_um, (1., 1.2), rtol=0, atol=1e-12):
        raise ValueError('Research wavelengths must remain 1000–1200 nm.')
    if cfg.source.center_um is not None or cfg.source.size_um is not None:
        raise ValueError('Use the default centred research plane source.')
    if any(abs(v) > 1e-12 for v in cfg.k_point):
        raise ValueError('Research k_point must be zero.')
    _require(cfg.stop_method, 'dft_decay', 'stop_method')
    _require(cfg.flux.enabled, False, 'flux.enabled')
    _require(cfg.output.field_nfreq, 21, 'field_nfreq')
    if sorted(cfg.output.field_components) != ['Ex', 'Ey', 'Ez']:
        raise ValueError('Save all three complex electric field components.')
    size = cfg.output.field_region_size_um
    if size is None or size[0] <= 0 or size[1] != 0 or size[2] != 0:
        raise ValueError('Research measurements currently support an x-line monitor only.')


def enforce_research_project(config, *, field_map_selection=None):
    from .project import make_simulation_config
    from .simulation import validate_config
    measurement = config['measurement']
    _require(measurement['normalization_mode'], 'global_complex_reference', 'normalization_mode')
    _require(measurement.get('validation_benchmark'), 'nonresonant_dielectric', 'validation_benchmark')
    _require(measurement.get('validation_region'), 'all_exterior', 'validation_region')
    _require(measurement['field_map_frequencies'], 'none', 'field_map_frequencies')
    if field_map_selection not in (None, 'none'):
        raise ValueError('This research extension supports --field-maps none.')
    raw = make_simulation_config(config, experiment_directory='data/results/research_preview',
                                 stage_name='frequency_scan/particle')
    enforce_research_simulation(validate_config(raw))
