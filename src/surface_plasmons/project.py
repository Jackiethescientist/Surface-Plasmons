"""Project-specific configuration for finite particle-line simulations."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping
import json
import math

from .particle_line import (
    line_extents,
    missing_particle_indices,
    particle_diameter,
    particle_line_positions,
)
from .results import safe_simulation_name


class ProjectConfigError(ValueError):
    """Raised before MEEP starts when a project configuration is invalid."""


PRECISION_MODES: dict[str, dict[str, float | int]] = {
    "fast": {
        "broadband_resolution": 60,
        "field_resolution": 80,
        "nfreq": 81,
        "dft_decay_tolerance": 1e-5,
        "minimum_run_time": 20.0,
        "maximum_run_time": 100.0,
        "narrow_relative_bandwidth": 0.05,
    },
    "study": {
        "broadband_resolution": 80,
        "field_resolution": 120,
        "nfreq": 151,
        "dft_decay_tolerance": 1e-6,
        "minimum_run_time": 20.0,
        "maximum_run_time": 200.0,
        "narrow_relative_bandwidth": 0.04,
    },
    "final": {
        "broadband_resolution": 120,
        "field_resolution": 160,
        "nfreq": 301,
        "dft_decay_tolerance": 1e-8,
        "minimum_run_time": 20.0,
        "maximum_run_time": 400.0,
        "narrow_relative_bandwidth": 0.02,
    },
}


DEFAULT_PROJECT_CONFIG: dict[str, Any] = {
    "simulation_name": "perfect_line_n9_spacing150nm",
    "precision": "fast",
    "geometry": {
        "particle_shape": "sphere",
        "particle_count": 9,
        "particle_material": "au",
        "background_material": "air",
        "substrate_material": "sio2",
        "particle_radius_um": 0.040,
        "cube_size_um": 0.080,
        "spacing_um": 0.150,
        "substrate_thickness_um": 0.300,
        "surface_y_um": 0.0,
        "particle_surface_gap_um": 0.0,
        "defects": [],
    },
    "source": {
        "wavelength_range_nm": [450.0, 800.0],
        "propagation_axis": "y",
        "propagation_direction": -1,
        "polarisation": "Ex",
    },
    "measurement": {
        "field_components": ["Ex", "Ey", "Ez"],
        "probe_margin_um": 0.050,
        "probe_offset_from_particle_surface_um": 0.020,
        "normalization_mode": "global_complex_reference",
        "field_map_margin_um": 0.100,
        "field_map_height_um": 0.300,
        "field_map_depth_um": 0.300,
        "field_map_frequencies": "auto",
        "maximum_auto_field_maps": 1,
    },
    "numerics": {
        "padding_um": 0.300,
        "pml_thickness_um": 0.200,
        "courant": 0.45,
        "record_database": True,
        "allow_out_of_range_materials": False,
        "allow_unstable_materials": False,
    },
}


def _merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_project_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        supplied = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectConfigError(f"Could not read configuration {source}: {exc}") from exc
    if not isinstance(supplied, Mapping):
        raise ProjectConfigError("The project configuration must be a JSON object.")
    return normalise_project_config(supplied)


def normalise_project_config(
    config: Mapping[str, Any],
    *,
    name: str | None = None,
    precision: str | None = None,
) -> dict[str, Any]:
    unknown = set(config) - set(DEFAULT_PROJECT_CONFIG)
    if unknown:
        raise ProjectConfigError(f"Unknown top-level configuration field(s): {sorted(unknown)}.")
    result = _merge(DEFAULT_PROJECT_CONFIG, config)
    if name is not None:
        result["simulation_name"] = name
    if precision is not None:
        result["precision"] = precision
    if result["precision"] == "sweep":
        result["precision"] = "study"
    if result["precision"] not in PRECISION_MODES:
        raise ProjectConfigError(
            f"precision must be one of {sorted(PRECISION_MODES)} (legacy 'sweep' means 'study')."
        )
    result["simulation_name"] = str(result["simulation_name"]).strip()
    safe_simulation_name(result["simulation_name"])

    geometry = result["geometry"]
    parameters = geometry_parameters(result, include_particles=True)
    positions = particle_line_positions(parameters)
    missing_count = len(
        missing_particle_indices(
            geometry.get("defects", []),
            int(geometry["particle_count"]),
        )
    )
    if len(positions) != int(geometry["particle_count"]) - missing_count:
        raise ProjectConfigError("Particle-position generation failed.")

    source = result["source"]
    wavelengths = source["wavelength_range_nm"]
    if (
        not isinstance(wavelengths, list)
        or len(wavelengths) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in wavelengths)
        or min(wavelengths) <= 0
        or wavelengths[0] >= wavelengths[1]
    ):
        raise ProjectConfigError("source.wavelength_range_nm must be [minimum, maximum].")
    axis = str(source["propagation_axis"]).lower()
    if axis not in {"x", "y", "z"}:
        raise ProjectConfigError("source.propagation_axis must be x, y, or z.")
    component = str(source["polarisation"])
    if component.lower() not in {"ex", "ey", "ez"}:
        raise ProjectConfigError("source.polarisation must be Ex, Ey, or Ez.")
    if component[-1].lower() == axis:
        raise ProjectConfigError(
            "The electric-field polarisation must be transverse to the propagation axis."
        )
    direction = source["propagation_direction"]
    if direction not in {-1, 1}:
        raise ProjectConfigError("source.propagation_direction must be -1 or +1.")

    components = result["measurement"]["field_components"]
    if sorted(str(value).lower() for value in components) != ["ex", "ey", "ez"]:
        raise ProjectConfigError("measurement.field_components must contain Ex, Ey, and Ez.")
    measurement = result["measurement"]
    if float(measurement["probe_offset_from_particle_surface_um"]) < 0:
        raise ProjectConfigError(
            "measurement.probe_offset_from_particle_surface_um cannot be negative."
        )
    if measurement["normalization_mode"] not in {
        "global_complex_reference", "pointwise_amplitude_reference"
    }:
        raise ProjectConfigError(
            "measurement.normalization_mode must be global_complex_reference or "
            "pointwise_amplitude_reference."
        )
    return result


def geometry_parameters(config: Mapping[str, Any], *, include_particles: bool) -> dict[str, Any]:
    geometry = config["geometry"]
    numerics = config["numerics"]
    return {
        "particle": geometry["particle_material"],
        "background": geometry["background_material"],
        "substrate": geometry["substrate_material"],
        "particle_shape": geometry["particle_shape"],
        "particle_count": int(geometry["particle_count"]),
        "radius": float(geometry["particle_radius_um"]),
        "cube_size": float(geometry["cube_size_um"]),
        "spacing": float(geometry["spacing_um"]),
        "substrate_thickness": float(geometry["substrate_thickness_um"]),
        "surface_y": float(geometry["surface_y_um"]),
        "particle_surface_gap": float(geometry["particle_surface_gap_um"]),
        "defects": deepcopy(geometry.get("defects", [])),
        "include_particles": bool(include_particles),
        "allow_out_of_range": bool(numerics["allow_out_of_range_materials"]),
        "allow_unstable": bool(numerics["allow_unstable_materials"]),
    }


def cell_size_um(config: Mapping[str, Any]) -> tuple[float, float, float]:
    parameters = geometry_parameters(config, include_particles=True)
    extents = line_extents({**parameters, "center": (0.0, 0.0, 0.0)})
    numerics = config["numerics"]
    margin = 2.0 * (float(numerics["padding_um"]) + float(numerics["pml_thickness_um"]))
    return tuple(float(value + margin) for value in extents)


def _monitor_geometry(config: Mapping[str, Any], *, full_volume: bool) -> tuple[list[float], list[float]]:
    parameters = {
        **geometry_parameters(config, include_particles=True),
        "center": (0.0, 0.0, 0.0),
    }
    positions = particle_line_positions(parameters)
    measurement = config["measurement"]
    diameter = particle_diameter(parameters)
    min_x = min(item.actual_x_um for item in positions) - diameter / 2.0
    max_x = max(item.actual_x_um for item in positions) + diameter / 2.0
    if not full_volume:
        margin = float(measurement["probe_margin_um"])
        half_height = diameter / 2.0
        probe_y = max(item.actual_y_um + half_height for item in positions) + float(
            measurement["probe_offset_from_particle_surface_um"]
        )
        centre = [0.5 * (min_x + max_x), probe_y, 0.0]
        size = [max_x - min_x + 2.0 * margin, 0.0, 0.0]
        return centre, size
    margin = float(measurement["field_map_margin_um"])
    height = float(measurement["field_map_height_um"])
    depth = float(measurement["field_map_depth_um"])
    surface_y = float(config["geometry"]["surface_y_um"])
    centre = [0.5 * (min_x + max_x), surface_y + 0.25 * height, 0.0]
    size = [max_x - min_x + 2.0 * margin, height, depth]
    return centre, size


def _narrow_band(wavelength_um: float, relative_width: float) -> list[float]:
    centre_frequency = 1.0 / float(wavelength_um)
    half = 0.5 * relative_width * centre_frequency
    return [1.0 / (centre_frequency + half), 1.0 / (centre_frequency - half)]


def make_simulation_config(
    project_config: Mapping[str, Any],
    *,
    experiment_directory: str | Path,
    stage_name: str,
    reference: bool = False,
    wavelength_um: float | None = None,
    resolution: int | None = None,
    padding_um: float | None = None,
) -> dict[str, Any]:
    config = normalise_project_config(project_config)
    profile = PRECISION_MODES[config["precision"]]
    is_broadband = wavelength_um is None
    selected_resolution = int(
        resolution
        if resolution is not None
        else profile["broadband_resolution" if is_broadband else "field_resolution"]
    )
    if is_broadband:
        wavelength_band = [float(value) / 1000.0 for value in config["source"]["wavelength_range_nm"]]
        field_nfreq = int(profile["nfreq"])
    else:
        wavelength_band = _narrow_band(
            float(wavelength_um), float(profile["narrow_relative_bandwidth"])
        )
        field_nfreq = 1
    monitor_center, monitor_size = _monitor_geometry(config, full_volume=not is_broadband)
    geometry = geometry_parameters(config, include_particles=not reference)
    name = config["simulation_name"]
    slug = safe_simulation_name(name)
    source = config["source"]
    numerics = config["numerics"]
    selected_padding = float(padding_um if padding_um is not None else numerics["padding_um"])
    selected_cell = list(cell_size_um(config))
    if padding_um is not None:
        difference = 2.0 * (selected_padding - float(numerics["padding_um"]))
        selected_cell = [value + difference for value in selected_cell]
        if any(value <= 0 for value in selected_cell):
            raise ProjectConfigError("The padding override makes the simulation cell non-positive.")
    return {
        "geometry_name": "particle_line_on_substrate",
        "geometry_parameters": geometry,
        "dimensions": 3,
        "resolution": selected_resolution,
        "cell_size_um": selected_cell,
        "padding_um": selected_padding,
        "pml_thickness_um": float(numerics["pml_thickness_um"]),
        "courant": float(numerics["courant"]),
        "progress_interval": 1.0e9,
        "stop_method": "dft_decay",
        "dft_decay_tolerance": float(profile["dft_decay_tolerance"]),
        "minimum_run_time": float(profile["minimum_run_time"]),
        "maximum_run_time": float(profile["maximum_run_time"]),
        "source": {
            "kind": "gaussian",
            "axis": source["propagation_axis"],
            "direction": int(source["propagation_direction"]),
            "component": source["polarisation"],
            "wavelength_range_um": wavelength_band,
        },
        "flux": {"enabled": False, "nfreq": 1, "normalization": "none"},
        "output": {
            "save_config": True,
            "save_summary": True,
            "save_spectra": False,
            "save_epsilon": False,
            "save_fields": True,
            "save_images": False,
            "field_components": list(config["measurement"]["field_components"]),
            "field_region_center_um": monitor_center,
            "field_region_size_um": monitor_size,
            "field_nfreq": field_nfreq,
            "field_wavelength_range_um": wavelength_band,
            "experiment_directory": str(Path(experiment_directory).resolve()),
            "stage_name": stage_name,
        },
        "run_name": f"{name} — {stage_name}",
        "run_id": f"{slug}__{safe_simulation_name(stage_name)}",
        "record_database": bool(numerics["record_database"]),
        "tags": ["particle-line", "3d", "substrate", f"precision:{config['precision']}", stage_name],
    }


def estimate_runtime(config: Mapping[str, Any]) -> dict[str, float | int | str]:
    simulation = dict(config)
    cell = [float(value) for value in simulation["cell_size_um"]]
    resolution = float(simulation["resolution"])
    grid = math.prod(max(1, math.ceil(value * resolution)) for value in cell)
    courant = float(simulation.get("courant", 0.45))
    maximum_time = float(simulation.get("maximum_run_time", 100.0))
    steps = maximum_time * resolution / courant
    # A conservative single-process value for dispersive 3-D laptop runs.
    cell_updates_per_second = 5_000_000.0
    maximum_seconds = grid * steps / cell_updates_per_second
    minimum_seconds = 0.20 * maximum_seconds
    memory_gb = grid * 220.0 / 1_000_000_000.0
    return {
        "grid_cells": int(grid),
        "estimated_min_seconds": minimum_seconds,
        "estimated_max_seconds": maximum_seconds,
        "estimated_memory_gb": memory_gb,
        "basis": "single-process conservative estimate; the first completed stage is the best calibration",
    }


__all__ = [
    "DEFAULT_PROJECT_CONFIG",
    "PRECISION_MODES",
    "ProjectConfigError",
    "cell_size_um",
    "estimate_runtime",
    "geometry_parameters",
    "load_project_config",
    "make_simulation_config",
    "normalise_project_config",
]
