"""
The runner consumes the registries in :mod:`surface_plasmons.materials` and
:mod:`surface_plasmons.geometries`, constructs a MEEP ``Simulation``, adds a
source plus spectral/field monitors, runs to a controlled stopping condition,
saves output files, and records the run in ``data/simulations.db``.

All length-like values use micrometres.  Therefore a 40 nm radius is written as
``0.040`` and MEEP frequency is ``1 / wavelength_um``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from importlib import import_module
from numbers import Integral, Real
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
import csv
import json
import math
import traceback

import numpy as np

from .database import SimulationDatabase, find_project_root, new_run_id
from .geometries import GeometryBuild, build_geometry, get_geometry_spec
from .particle_line import line_extents
from .validated_scope import enforce_validated_simulation


class SimulationConfigError(ValueError):
    """Raised before MEEP starts when a simulation configuration is invalid."""


@dataclass(slots=True)
class SourceConfig:
    """Broadband plane-wave or point-dipole source settings."""

    kind: str = "gaussian"              # gaussian, continuous, or dipole
    wavelength_range_um: tuple[float, float] = (0.45, 0.80)
    component: str = "Ez"
    axis: str | None = None              # inferred from geometry when None
    direction: int | None = None         # +1 or -1; inferred when None
    amplitude: float = 1.0
    center_um: tuple[float, float, float] | None = None
    size_um: tuple[float, float, float] | None = None
    offset_from_pml_um: float = 0.05
    cutoff: float = 5.0
    is_integrated: bool = True
    span_pml: bool = False

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "SourceConfig":
        data = dict(value or {})
        aliases = {
            "wavelengths_um": "wavelength_range_um",
            "polarization": "component",
            "propagation_axis": "axis",
            "propagation_direction": "direction",
        }
        _rename_keys(data, aliases)
        _reject_unknown(cls, data, "source")
        if "wavelength_range_um" in data:
            data["wavelength_range_um"] = _normalise_band(data["wavelength_range_um"])
        for key in ("center_um", "size_um"):
            if data.get(key) is not None:
                data[key] = _vector3(data[key], key)
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        self.wavelength_range_um = _normalise_band(self.wavelength_range_um)
        if not isinstance(self.span_pml, bool):
            raise SimulationConfigError("source.span_pml must be a boolean.")
        if self.span_pml and not self.is_integrated:
            raise SimulationConfigError("A plane source spanning PML requires is_integrated=True.")
        if self.kind not in {"gaussian", "continuous", "dipole"}:
            raise SimulationConfigError("source.kind must be gaussian, continuous, or dipole.")
        if self.axis is not None and self.axis.lower() not in {"x", "y", "z"}:
            raise SimulationConfigError("source.axis must be x, y, z, or None.")
        if self.axis is not None:
            self.axis = self.axis.lower()
        if self.direction not in {None, -1, 1}:
            raise SimulationConfigError("source.direction must be -1, +1, or None.")
        if not isinstance(self.component, str) or not self.component.strip():
            raise SimulationConfigError("source.component must name a MEEP field component.")
        _positive("source.amplitude", abs(self.amplitude))
        _positive("source.offset_from_pml_um", self.offset_from_pml_um, allow_zero=True)
        _positive("source.cutoff", self.cutoff)


@dataclass(slots=True)
class FluxConfig:
    """Input/output spectral monitor settings."""

    enabled: bool = True
    nfreq: int = 101
    input_position_um: float | None = None
    output_position_um: float | None = None
    input_name: str = "input_flux"
    output_name: str = "output_flux"
    normalization: str = "none"         # none or empty_cell

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "FluxConfig":
        data = dict(value or {})
        _rename_keys(data, {"normalisation": "normalization"})
        _reject_unknown(cls, data, "flux")
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise SimulationConfigError("flux.enabled must be true or false.")
        if isinstance(self.nfreq, bool) or not isinstance(self.nfreq, Integral) or self.nfreq < 1:
            raise SimulationConfigError("flux.nfreq must be a positive integer.")
        if self.normalization not in {"none", "empty_cell"}:
            raise SimulationConfigError("flux.normalization must be 'none' or 'empty_cell'.")
        if not self.input_name.strip() or not self.output_name.strip():
            raise SimulationConfigError("Flux monitor names must not be empty.")
        if self.input_name == self.output_name:
            raise SimulationConfigError("Input and output flux monitor names must differ.")
        for name in ("input_position_um", "output_position_um"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
                raise SimulationConfigError(f"flux.{name} must be a number or None.")


@dataclass(slots=True)
class OutputConfig:
    """Controls saved arrays, spectra, figures, and run summaries."""

    results_root: str | None = None
    save_config: bool = True
    save_summary: bool = True
    save_spectra: bool = True
    save_epsilon: bool = True
    save_fields: bool = True
    save_images: bool = True
    field_components: tuple[str, ...] = ("Ez",)
    field_region_center_um: tuple[float, float, float] | None = None
    field_region_size_um: tuple[float, float, float] | None = None
    field_nfreq: int = 1
    field_wavelength_range_um: tuple[float, float] | None = None
    experiment_directory: str | None = None
    stage_name: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "OutputConfig":
        data = dict(value or {})
        _reject_unknown(cls, data, "output")
        if "field_components" in data:
            components = data["field_components"]
            if isinstance(components, str):
                components = (components,)
            data["field_components"] = tuple(str(item) for item in components)
        for key in ("field_region_center_um", "field_region_size_um"):
            if data.get(key) is not None:
                data[key] = _vector3(data[key], key)
        if data.get("field_wavelength_range_um") is not None:
            data["field_wavelength_range_um"] = _normalise_band(
                data["field_wavelength_range_um"]
            )
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        for name in (
            "save_config", "save_summary", "save_spectra", "save_epsilon",
            "save_fields", "save_images",
        ):
            if not isinstance(getattr(self, name), bool):
                raise SimulationConfigError(f"output.{name} must be true or false.")
        if not self.field_components:
            raise SimulationConfigError("At least one output field component is required.")
        if any(not item.strip() for item in self.field_components):
            raise SimulationConfigError("Field component names must not be empty.")
        if self.field_region_size_um is not None:
            if any(value < 0 for value in self.field_region_size_um):
                raise SimulationConfigError("Field-region sizes cannot be negative.")
        if isinstance(self.field_nfreq, bool) or not isinstance(self.field_nfreq, Integral) or self.field_nfreq < 1:
            raise SimulationConfigError("output.field_nfreq must be a positive integer.")
        if self.field_wavelength_range_um is not None:
            self.field_wavelength_range_um = _normalise_band(self.field_wavelength_range_um)
        if self.stage_name is not None:
            if not self.stage_name.strip() or any(part in {"", ".", ".."} for part in Path(self.stage_name).parts):
                raise SimulationConfigError("output.stage_name must be a safe relative path.")
            if Path(self.stage_name).is_absolute():
                raise SimulationConfigError("output.stage_name must be relative.")


@dataclass(slots=True)
class SimulationConfig:
    """Complete serialisable simulation configuration."""

    geometry_name: str = "nanoparticle_sphere"
    geometry_parameters: dict[str, Any] = field(default_factory=dict)
    dimensions: int = 2
    resolution: float = 80.0
    cell_size_um: tuple[float, float, float] | None = None
    padding_um: float = 0.30
    transverse_span_um: float = 1.50
    pml_thickness_um: float = 0.20
    pml_axes: tuple[str, ...] | None = None
    courant: float = 0.45
    k_point: tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_complex_fields: bool = False
    mirror_symmetry: bool = False
    mirror_axes: tuple[str, ...] | None = None
    progress_interval: float = 4.0
    stop_method: str = "dft_decay"       # dft_decay, field_decay, fixed_time
    dft_decay_tolerance: float = 1e-8
    minimum_run_time: float = 20.0
    maximum_run_time: float = 400.0
    field_decay_interval: float = 20.0
    field_decay_by: float = 1e-8
    fixed_run_time: float = 200.0
    source: SourceConfig = field(default_factory=SourceConfig)
    flux: FluxConfig = field(default_factory=FluxConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run_name: str | None = None
    run_id: str | None = None
    notes: str | None = None
    tags: tuple[str, ...] = ()
    database_path: str | None = None
    record_database: bool = True
    code_version: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationConfig":
        if not isinstance(value, Mapping):
            raise SimulationConfigError("Simulation configuration must be a mapping.")
        data = dict(value)

        # Accept both compact and explicit geometry forms.
        geometry_value = data.pop("geometry", None)
        if isinstance(geometry_value, Mapping):
            data.setdefault(
                "geometry_name",
                geometry_value.get("name", geometry_value.get("type", geometry_value.get("geometry"))),
            )
            data.setdefault(
                "geometry_parameters",
                geometry_value.get("parameters", geometry_value.get("params", {})),
            )
        elif geometry_value is not None:
            data.setdefault("geometry_name", geometry_value)

        _rename_keys(
            data,
            {
                "parameters": "geometry_parameters",
                "cell_size": "cell_size_um",
                "pml_thickness": "pml_thickness_um",
                "Courant": "courant",
            },
        )
        source_data = dict(data.pop("source", {}) or {})
        for alias in ("wavelength_range_um", "wavelengths_um"):
            if alias in data:
                source_data.setdefault("wavelength_range_um", data.pop(alias))
        flux_data = data.pop("flux", {})
        output_data = data.pop("output", {})
        _reject_unknown(cls, data, "simulation")

        if data.get("cell_size_um") is not None:
            data["cell_size_um"] = _cell_vector(data["cell_size_um"], int(data.get("dimensions", 2)))
        if data.get("pml_axes") is not None:
            axes = data["pml_axes"]
            if isinstance(axes, str):
                axes = (axes,)
            data["pml_axes"] = tuple(str(axis).lower() for axis in axes)
        if "k_point" in data:
            data["k_point"] = _vector3(data["k_point"], "k_point")
        if "tags" in data:
            data["tags"] = tuple(str(tag) for tag in data["tags"])
        if data.get('mirror_axes') is not None:
            axes = data['mirror_axes']
            if not isinstance(axes, (list, tuple)):
                raise SimulationConfigError('mirror_axes must be a list containing x and/or z.')
            data['mirror_axes'] = tuple(str(axis).lower() for axis in axes)
        if "geometry_parameters" in data:
            data["geometry_parameters"] = dict(data["geometry_parameters"] or {})

        result = cls(
            **data,
            source=SourceConfig.from_dict(source_data),
            flux=FluxConfig.from_dict(flux_data),
            output=OutputConfig.from_dict(output_data),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.geometry_name, str) or not self.geometry_name.strip():
            raise SimulationConfigError("geometry_name must be a non-empty string.")
        get_geometry_spec(self.geometry_name)
        if self.dimensions not in {2, 3}:
            raise SimulationConfigError("dimensions must be 2 or 3.")
        _positive("resolution", self.resolution)
        _positive("padding_um", self.padding_um, allow_zero=True)
        _positive("transverse_span_um", self.transverse_span_um)
        _positive("pml_thickness_um", self.pml_thickness_um, allow_zero=True)
        if not isinstance(self.courant, Real) or isinstance(self.courant, bool) or not 0 < self.courant <= 0.5:
            raise SimulationConfigError("courant must be greater than 0 and at most 0.5.")
        _positive("progress_interval", self.progress_interval, allow_zero=True)
        if self.stop_method not in {"dft_decay", "field_decay", "fixed_time"}:
            raise SimulationConfigError("stop_method must be dft_decay, field_decay, or fixed_time.")
        for name in (
            "dft_decay_tolerance", "minimum_run_time", "maximum_run_time",
            "field_decay_interval", "field_decay_by", "fixed_run_time",
        ):
            _positive(name, getattr(self, name), allow_zero=name == "minimum_run_time")
        if self.maximum_run_time < self.minimum_run_time:
            raise SimulationConfigError("maximum_run_time cannot be smaller than minimum_run_time.")
        if self.cell_size_um is not None:
            self.cell_size_um = _cell_vector(self.cell_size_um, self.dimensions)
            active = self.cell_size_um[:2] if self.dimensions == 2 else self.cell_size_um
            if any(value <= 0 for value in active):
                raise SimulationConfigError("Active cell dimensions must be positive.")
        if self.pml_axes is not None:
            allowed = {"x", "y"} if self.dimensions == 2 else {"x", "y", "z"}
            invalid = set(self.pml_axes) - allowed
            if invalid:
                raise SimulationConfigError(f"Invalid PML axes: {sorted(invalid)}.")
        self.k_point = _vector3(self.k_point, "k_point")
        self.source.validate()
        benchmark_mirror_planes(self)
        self.flux.validate()
        self.output.validate()
        if self.source.axis == "z" and self.dimensions == 2:
            raise SimulationConfigError("A 2-D xy simulation cannot propagate along z.")
        if self.source.kind != "dipole" and self.source.axis is not None:
            component = self.source.component.strip().lower()
            if component in {"ex", "ey", "ez"} and component[-1] == self.source.axis:
                raise SimulationConfigError(
                    "A plane wave cannot have its electric-field component parallel "
                    "to its propagation axis. Choose a transverse Ex, Ey, or Ez component."
                )
        if not isinstance(self.record_database, bool):
            raise SimulationConfigError("record_database must be true or false.")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Preserve the serialized identity of every earlier default case.
        if value['mirror_axes'] is None:
            value.pop('mirror_axes')
        return _jsonable(value)


@dataclass(slots=True)
class SimulationContext:
    config: SimulationConfig
    run_id: str
    simulation: Any
    geometry: GeometryBuild
    cell_size_um: tuple[float, float, float]
    source_axis: str
    source_direction: int
    source_center_um: tuple[float, float, float]
    flux_monitors: dict[str, Any]
    flux_positions_um: dict[str, tuple[float, float, float]]
    dft_fields: Any | None
    dft_components: dict[str, Any]
    field_frequency: float
    field_frequencies: tuple[float, ...]
    decay_point_um: tuple[float, float, float]


@dataclass(slots=True)
class SimulationResult:
    run_id: str
    status: str
    geometry_name: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    meep_time: float
    timesteps: int | None
    frequencies: list[float]
    wavelengths_um: list[float]
    spectra: dict[str, list[float]]
    metrics: dict[str, float]
    artifacts: dict[str, str]
    output_directory: str | None
    warnings: list[str]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def validate_config(config: SimulationConfig | Mapping[str, Any]) -> SimulationConfig:
    if isinstance(config, SimulationConfig):
        config.validate()
        return config
    return SimulationConfig.from_dict(config)


def benchmark_mirror_planes(cfg: SimulationConfig) -> tuple[tuple[str, int], ...]:
    """Return source-compatible x/z mirrors for the supported sphere benchmark.

    This deliberately accepts only a centred Au sphere in air or a constant
    glass sphere in vacuum/air, illuminated
    along y with Ex or Ez. More general geometries need their own symmetry
    checks; Meep does not verify the user's choice of symmetry.
    """
    if not isinstance(cfg.mirror_symmetry, bool):
        raise SimulationConfigError("mirror_symmetry must be true or false.")
    if cfg.mirror_axes is not None:
        if (not isinstance(cfg.mirror_axes, (list, tuple)) or not cfg.mirror_axes
                or len(set(cfg.mirror_axes)) != len(cfg.mirror_axes)
                or any(axis not in {'x', 'z'} for axis in cfg.mirror_axes)):
            raise SimulationConfigError('mirror_axes must contain distinct x and/or z axes.')
        if not cfg.mirror_symmetry:
            raise SimulationConfigError('Explicit mirror_axes requires mirror_symmetry=true.')
        if 'execution:research_dielectric_line' not in cfg.tags:
            raise SimulationConfigError('Explicit mirror_axes is supported only for research dielectric lines.')
    if not cfg.mirror_symmetry:
        return ()
    if 'execution:research_dielectric_line' in cfg.tags:
        from .research_scope import enforce_research_simulation, symmetric_research_positions
        enforce_research_simulation(cfg)
        axes = cfg.mirror_axes if cfg.mirror_axes is not None else ('x', 'z')
        if not symmetric_research_positions(cfg.geometry_parameters, axes=tuple({'x':0, 'z':2}[axis] for axis in axes)):
            raise SimulationConfigError(
                'mirror_symmetry requires the complete research geometry to be symmetric '
                'under every requested reflection. For an x displacement, request mirror_axes=["z"].'
            )
        component_axis = cfg.source.component[-1].lower()
        return tuple((axis, -1 if component_axis == axis else 1) for axis in axes)
    parameters = cfg.geometry_parameters
    center = parameters.get("center", (0., 0., 0.))
    source_center = cfg.source.center_um or (0., 0., 0.)
    valid = (
        cfg.dimensions == 3
        and cfg.geometry_name == "particle_line_on_substrate"
        and parameters.get("particle_shape", "sphere") == "sphere"
        and parameters.get("particle_count") == 1
        and ((parameters.get("particle") == "au"
              and parameters.get("background") == parameters.get("substrate") == "air")
             or (parameters.get("particle") == "glass_constant"
                 and parameters.get("background") == parameters.get("substrate")
                 and parameters.get("background") in {"vacuum", "air"}))
        and not parameters.get("defects")
        and len(center) == 3
        and all(abs(float(center[i])) <= 1e-12 for i in (0, 2))
        and all(abs(float(source_center[i])) <= 1e-12 for i in (0, 2))
        and cfg.source.kind in {"gaussian", "continuous"}
        and cfg.source.axis == "y"
        and cfg.source.component.lower() in {"ex", "ez"}
        and not any(abs(value) > 1e-12 for value in cfg.k_point)
    )
    if not valid:
        raise SimulationConfigError(
            "mirror_symmetry supports one centred Au sphere in air or constant glass "
            "sphere in vacuum/air, without defects, illuminated along y with Ex or Ez "
            "and zero k_point."
        )
    component_axis = cfg.source.component[-1].lower()
    return tuple((axis, -1 if component_axis == axis else 1) for axis in ("x", "z"))


def create_simulation(
    config: SimulationConfig | Mapping[str, Any],
    *,
    run_id: str | None = None,
    empty_geometry: bool = False,
) -> SimulationContext:
    """Construct MEEP objects and monitors without advancing the fields."""

    cfg = validate_config(config)
    enforce_validated_simulation(cfg)
    mp = _import_meep()
    identity = run_id or cfg.run_id or new_run_id()
    geometry_parameters = dict(cfg.geometry_parameters)
    geometry_parameters["dimensions"] = cfg.dimensions
    geometry_parameters["wavelength_um"] = cfg.source.wavelength_range_um
    built = build_geometry(cfg.geometry_name, geometry_parameters)

    cell_size = _resolve_cell_size(cfg, built)
    source_axis, source_direction = _resolve_source_direction(cfg, built)
    if cfg.source.kind != "dipole":
        component_name = cfg.source.component.strip().lower()
        if component_name in {"ex", "ey", "ez"} and component_name[-1] == source_axis:
            raise SimulationConfigError(
                "A plane wave cannot have its electric-field component parallel "
                f"to propagation along {source_axis}."
            )
    periodic_axes = set(built.periodic_axes)
    pml_axes = _resolve_pml_axes(cfg, periodic_axes)
    boundaries = _build_pml(mp, cfg.pml_thickness_um, pml_axes)
    source, source_center = _build_source(
        mp, cfg, cell_size, source_axis, source_direction, pml_axes
    )

    k_vector = mp.Vector3(*cfg.k_point)
    use_bloch = bool(periodic_axes or any(abs(value) > 0 for value in cfg.k_point))
    simulation = mp.Simulation(
        cell_size=mp.Vector3(*cell_size),
        resolution=cfg.resolution,
        geometry=[] if empty_geometry else built.geometry,
        sources=[source],
        dimensions=cfg.dimensions,
        boundary_layers=boundaries,
        default_material=built.default_material,
        k_point=k_vector if use_bloch else False,
        ensure_periodicity=True,
        force_complex_fields=cfg.force_complex_fields,
        symmetries=[mp.Mirror(_direction(mp, axis), phase=phase)
                    for axis, phase in benchmark_mirror_planes(cfg)],
        Courant=cfg.courant,
        filename_prefix=identity,
        progress_interval=cfg.progress_interval,
    )

    fcen, df, nfreq = _frequency_parameters(cfg.source.wavelength_range_um, cfg.flux.nfreq)
    flux_monitors: dict[str, Any] = {}
    flux_positions: dict[str, tuple[float, float, float]] = {}
    device_extent = _estimate_geometry_extent(cfg, built)
    if cfg.flux.enabled:
        regions, flux_positions = _build_flux_regions(
            mp, cfg, cell_size, device_extent, source_axis, source_direction, pml_axes
        )
        for name, region in regions.items():
            flux_monitors[name] = simulation.add_flux(fcen, df, nfreq, region)

    dft_fields = None
    dft_components: dict[str, Any] = {}
    field_frequency = fcen
    field_frequencies: tuple[float, ...] = (field_frequency,)
    if cfg.output.save_fields or cfg.output.save_images:
        dft_components = {
            name: _component(mp, name) for name in cfg.output.field_components
        }
        region_center, region_size = _field_region(cfg, cell_size, pml_axes)
        field_band = cfg.output.field_wavelength_range_um or cfg.source.wavelength_range_um
        field_fcen, field_df, field_nfreq = _frequency_parameters(
            field_band, cfg.output.field_nfreq
        )
        field_frequency = field_fcen
        if field_nfreq == 1:
            field_frequencies = (field_fcen,)
        else:
            field_frequencies = tuple(
                float(value)
                for value in np.linspace(
                    field_fcen - field_df / 2.0,
                    field_fcen + field_df / 2.0,
                    field_nfreq,
                )
            )
        dft_fields = simulation.add_dft_fields(
            list(dft_components.values()),
            field_fcen,
            field_df,
            field_nfreq,
            center=mp.Vector3(*region_center),
            size=mp.Vector3(*region_size),
        )

    if cfg.flux.enabled:
        decay_point = flux_positions[cfg.flux.output_name]
    else:
        decay_point = tuple(float(value) for value in built.parameters.get("center", (0, 0, 0)))

    return SimulationContext(
        config=cfg,
        run_id=identity,
        simulation=simulation,
        geometry=built,
        cell_size_um=cell_size,
        source_axis=source_axis,
        source_direction=source_direction,
        source_center_um=source_center,
        flux_monitors=flux_monitors,
        flux_positions_um=flux_positions,
        dft_fields=dft_fields,
        dft_components=dft_components,
        field_frequency=field_frequency,
        field_frequencies=field_frequencies,
        decay_point_um=decay_point,
    )


build_simulation = create_simulation


def run_simulation(
    config: SimulationConfig | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> SimulationResult:
    """Run one simulation, save outputs, and record it in SQLite.

    ``config`` may be a :class:`SimulationConfig` or a dictionary.  Keyword
    overrides are accepted only when ``config`` is a dictionary or omitted.
    """

    if config is None:
        raw: Mapping[str, Any] = overrides
    elif isinstance(config, SimulationConfig):
        if overrides:
            raise SimulationConfigError(
                "Keyword overrides cannot be combined with a SimulationConfig object."
            )
        raw = config
    else:
        raw = {**config, **overrides}
    cfg = validate_config(raw)
    enforce_validated_simulation(cfg)
    mp = _import_meep()
    identity = cfg.run_id or new_run_id()
    output_paths = _output_paths(cfg, identity)
    master = bool(getattr(mp, "am_master", lambda: True)())
    database = (
        SimulationDatabase(cfg.database_path) if cfg.record_database and master else None
    )
    start_clock = perf_counter()
    started_at = _utc_now()
    context: SimulationContext | None = None
    warnings: list[str] = []
    artifacts: dict[str, str] = {}

    if master:
        for path in output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        if cfg.output.save_config:
            config_path = output_paths["run"] / "config.json"
            _write_json(config_path, cfg.to_dict())
            artifacts["config"] = str(config_path)

    if database is not None:
        database.create_run(
            cfg.to_dict(),
            run_id=identity,
            name=cfg.run_name,
            output_directory=output_paths["run"],
            notes=cfg.notes,
            tags=cfg.tags,
            code_version=cfg.code_version,
            meep_version=str(getattr(mp, "__version__", "unknown")),
        )
        database.mark_running(identity, output_directory=output_paths["run"])

    try:
        normalization_data = None
        incident_flux: list[float] | None = None
        if cfg.flux.enabled and cfg.flux.normalization == "empty_cell":
            if cfg.source.kind == "dipole":
                raise SimulationConfigError(
                    "empty_cell flux normalization is defined for a plane-wave source, not a dipole."
                )
            normalization = create_simulation(cfg, run_id=f"{identity}_normalization", empty_geometry=True)
            _execute(normalization)
            input_monitor = normalization.flux_monitors[cfg.flux.input_name]
            incident_flux = [float(value) for value in mp.get_fluxes(input_monitor)]
            normalization_data = normalization.simulation.get_flux_data(input_monitor)
            normalization.simulation.reset_meep()
            if database is not None:
                database.add_event(
                    identity,
                    "normalization_completed",
                    message="Empty-cell incident flux calculated",
                )

        context = create_simulation(cfg, run_id=identity)
        if normalization_data is not None:
            context.simulation.load_minus_flux_data(
                context.flux_monitors[cfg.flux.input_name], normalization_data
            )
        _execute(context)

        frequencies, spectra = _collect_flux(mp, context)
        wavelengths = [1.0 / value for value in frequencies]
        if incident_flux is not None:
            normalized = _normalise_spectra(
                incident_flux,
                spectra[cfg.flux.input_name],
                spectra[cfg.flux.output_name],
            )
            spectra.update(normalized)

        elapsed = perf_counter() - start_clock
        meep_time = float(context.simulation.meep_time())
        timestep_function = getattr(context.simulation, "timestep", None)
        timesteps = int(timestep_function()) if callable(timestep_function) else None
        metrics: dict[str, float] = {
            "elapsed_seconds": elapsed,
            "meep_time": meep_time,
        }
        if timesteps is not None:
            metrics["timesteps"] = float(timesteps)
        for name, values in spectra.items():
            if values:
                finite = [value for value in values if math.isfinite(value)]
                if finite:
                    metrics[f"{name}_minimum"] = min(finite)
                    metrics[f"{name}_maximum"] = max(finite)

        if master:
            saved, output_metrics, output_warnings = _save_outputs(
                context,
                frequencies,
                wavelengths,
                spectra,
                output_paths,
            )
            artifacts.update(saved)
            metrics.update(output_metrics)
            warnings.extend(output_warnings)

        finished_at = _utc_now()
        result = SimulationResult(
            run_id=identity,
            status="completed",
            geometry_name=context.geometry.name,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_seconds=elapsed,
            meep_time=meep_time,
            timesteps=timesteps,
            frequencies=frequencies,
            wavelengths_um=wavelengths,
            spectra=spectra,
            metrics=metrics,
            artifacts=artifacts,
            output_directory=str(output_paths["run"]) if master else None,
            warnings=warnings,
            config=cfg.to_dict(),
        )

        if master and cfg.output.save_summary:
            summary_path = output_paths["run"] / "summary.json"
            artifacts["summary"] = str(summary_path)
            result.artifacts = artifacts
            _write_json(summary_path, result.to_dict())

        if database is not None:
            database.record_metrics(
                identity,
                metrics,
                units={
                    "elapsed_seconds": "s",
                    "meep_time": "simulation_time",
                    "timesteps": "timesteps",
                },
            )
            for monitor_name, values in spectra.items():
                database.record_spectrum(
                    identity,
                    monitor_name,
                    frequencies,
                    values,
                    unit="flux" if "flux" in monitor_name else "dimensionless",
                )
            for kind, path in artifacts.items():
                database.record_artifact(
                    identity,
                    kind,
                    path,
                    mime_type=_mime_type(Path(path)),
                )
            database.mark_completed(
                identity,
                elapsed_seconds=elapsed,
                output_directory=output_paths["run"],
            )
        return result
    except BaseException as exc:
        elapsed = perf_counter() - start_clock
        if database is not None:
            database.mark_failed(
                identity,
                exc,
                elapsed_seconds=elapsed,
                traceback_text=traceback.format_exc(),
                output_directory=output_paths["run"],
            )
        raise
    finally:
        if context is not None:
            reset = getattr(context.simulation, "reset_meep", None)
            if callable(reset):
                reset()


simulate = run_simulation
run = run_simulation


def _execute(context: SimulationContext) -> None:
    mp = _import_meep()
    cfg = context.config
    simulation = context.simulation
    if cfg.source.kind == "continuous" or cfg.stop_method == "fixed_time":
        simulation.run(until=cfg.fixed_run_time)
        return
    if cfg.stop_method == "dft_decay" and (
        context.flux_monitors or context.dft_fields is not None
    ):
        condition = mp.stop_when_dft_decayed(
            tol=cfg.dft_decay_tolerance,
            minimum_run_time=cfg.minimum_run_time,
            maximum_run_time=cfg.maximum_run_time,
        )
        simulation.run(until_after_sources=condition)
        return
    component = _component(mp, cfg.source.component)
    point = mp.Vector3(*context.decay_point_um)
    condition = mp.stop_when_fields_decayed(
        cfg.field_decay_interval,
        component,
        point,
        cfg.field_decay_by,
    )
    simulation.run(until_after_sources=condition)


def _build_source(
    mp: Any,
    cfg: SimulationConfig,
    cell_size: tuple[float, float, float],
    axis: str,
    direction: int,
    pml_axes: set[str],
) -> tuple[Any, tuple[float, float, float]]:
    fcen, df, _ = _frequency_parameters(cfg.source.wavelength_range_um, 1)
    source_width = df if df > 0 else 0.1 * fcen
    if cfg.source.kind == "continuous":
        source_time = mp.ContinuousSource(frequency=fcen, width=source_width)
    else:
        source_time = mp.GaussianSource(
            frequency=fcen,
            fwidth=source_width,
            cutoff=cfg.source.cutoff,
            is_integrated=cfg.source.is_integrated,
        )

    if cfg.source.center_um is not None:
        center = cfg.source.center_um
    else:
        center_values = [0.0, 0.0, 0.0]
        axis_index = _AXIS_INDEX[axis]
        pml = cfg.pml_thickness_um if axis in pml_axes else 0.0
        half = cell_size[axis_index] / 2
        coordinate = -direction * (half - pml - cfg.source.offset_from_pml_um)
        if abs(coordinate) <= 0:
            raise SimulationConfigError("Cell is too small to place the source outside the PML.")
        center_values[axis_index] = coordinate
        center = tuple(center_values)

    if cfg.source.size_um is not None:
        size = cfg.source.size_um
    elif cfg.source.kind == "dipole":
        size = (0.0, 0.0, 0.0)
    else:
        size_values = list(cell_size)
        size_values[_AXIS_INDEX[axis]] = 0.0
        for candidate in ("x", "y", "z"):
            index = _AXIS_INDEX[candidate]
            if candidate in pml_axes and size_values[index] > 0 and not cfg.source.span_pml:
                size_values[index] = max(0.0, size_values[index] - 2 * cfg.pml_thickness_um)
        if cfg.dimensions == 2:
            size_values[2] = 0.0
        size = tuple(size_values)

    source = mp.Source(
        src=source_time,
        component=_component(mp, cfg.source.component),
        center=mp.Vector3(*center),
        size=mp.Vector3(*size),
        amplitude=cfg.source.amplitude,
    )
    return source, tuple(float(value) for value in center)


def _build_flux_regions(
    mp: Any,
    cfg: SimulationConfig,
    cell_size: tuple[float, float, float],
    device_extent: tuple[float, float, float],
    axis: str,
    direction: int,
    pml_axes: set[str],
) -> tuple[dict[str, Any], dict[str, tuple[float, float, float]]]:
    axis_index = _AXIS_INDEX[axis]
    half_cell = cell_size[axis_index] / 2
    pml = cfg.pml_thickness_um if axis in pml_axes else 0.0
    safe_limit = half_cell - pml - 0.02
    device_half = device_extent[axis_index] / 2
    automatic_offset = min(max(device_half + 0.05, half_cell * 0.20), safe_limit * 0.75)
    if automatic_offset <= device_half:
        raise SimulationConfigError(
            "Cell is too small to place flux monitors outside the geometry. "
            "Increase cell_size_um or padding_um."
        )

    input_coordinate = (
        float(cfg.flux.input_position_um)
        if cfg.flux.input_position_um is not None
        else -direction * automatic_offset
    )
    output_coordinate = (
        float(cfg.flux.output_position_um)
        if cfg.flux.output_position_um is not None
        else direction * automatic_offset
    )
    for label, coordinate in (("input", input_coordinate), ("output", output_coordinate)):
        if abs(coordinate) >= safe_limit:
            raise SimulationConfigError(
                f"{label} flux monitor lies in or beyond the PML; move it inward."
            )

    size = list(cell_size)
    size[axis_index] = 0.0
    for candidate in pml_axes:
        index = _AXIS_INDEX[candidate]
        if index != axis_index and size[index] > 0:
            size[index] = max(0.0, size[index] - 2 * cfg.pml_thickness_um)
    if cfg.dimensions == 2:
        size[2] = 0.0

    axis_direction = _direction(mp, axis)
    positions: dict[str, tuple[float, float, float]] = {}
    regions: dict[str, Any] = {}
    for name, coordinate in (
        (cfg.flux.input_name, input_coordinate),
        (cfg.flux.output_name, output_coordinate),
    ):
        center = [0.0, 0.0, 0.0]
        center[axis_index] = coordinate
        positions[name] = tuple(center)
        regions[name] = mp.FluxRegion(
            center=mp.Vector3(*center),
            size=mp.Vector3(*size),
            direction=axis_direction,
        )
    return regions, positions


def _field_region(
    cfg: SimulationConfig,
    cell_size: tuple[float, float, float],
    pml_axes: set[str],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    center = cfg.output.field_region_center_um or (0.0, 0.0, 0.0)
    if cfg.output.field_region_size_um is not None:
        return center, cfg.output.field_region_size_um
    size = list(cell_size)
    for axis in pml_axes:
        index = _AXIS_INDEX[axis]
        size[index] = max(0.0, size[index] - 2 * cfg.pml_thickness_um)
    if cfg.dimensions == 2:
        size[2] = 0.0
    return center, tuple(size)


def _collect_flux(mp: Any, context: SimulationContext) -> tuple[list[float], dict[str, list[float]]]:
    if not context.flux_monitors:
        return [context.field_frequency], {}
    first = next(iter(context.flux_monitors.values()))
    frequencies = [float(value) for value in mp.get_flux_freqs(first)]
    spectra = {
        name: [float(value) for value in mp.get_fluxes(monitor)]
        for name, monitor in context.flux_monitors.items()
    }
    return frequencies, spectra


def _normalise_spectra(
    incident: Sequence[float], reflected_scattered: Sequence[float], transmitted: Sequence[float]
) -> dict[str, list[float]]:
    reflectance: list[float] = []
    transmittance: list[float] = []
    absorption: list[float] = []
    for inc, refl, trans in zip(incident, reflected_scattered, transmitted):
        if abs(inc) <= 1e-30:
            reflectance.append(float("nan"))
            transmittance.append(float("nan"))
            absorption.append(float("nan"))
            continue
        r_value = -float(refl) / float(inc)
        t_value = float(trans) / float(inc)
        reflectance.append(r_value)
        transmittance.append(t_value)
        absorption.append(1.0 - r_value - t_value)
    return {
        "reflectance": reflectance,
        "transmittance": transmittance,
        "absorption": absorption,
    }


def _save_outputs(
    context: SimulationContext,
    frequencies: Sequence[float],
    wavelengths: Sequence[float],
    spectra: Mapping[str, Sequence[float]],
    paths: Mapping[str, Path],
) -> tuple[dict[str, str], dict[str, float], list[str]]:
    cfg = context.config
    simulation = context.simulation
    artifacts: dict[str, str] = {}
    metrics: dict[str, float] = {}
    warnings: list[str] = []

    if cfg.output.save_spectra and spectra:
        spectrum_path = paths["run"] / "spectra.csv"
        with spectrum_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["monitor", "sample_index", "frequency", "wavelength_um", "value"]
            )
            for monitor_name, values in spectra.items():
                for index, (frequency, wavelength, value) in enumerate(
                    zip(frequencies, wavelengths, values)
                ):
                    writer.writerow([monitor_name, index, frequency, wavelength, value])
        artifacts["spectra"] = str(spectrum_path)

    epsilon_array: np.ndarray | None = None
    if cfg.output.save_epsilon or cfg.output.save_images:
        epsilon_array = np.asarray(simulation.get_epsilon(frequency=context.field_frequency))
        if cfg.output.save_epsilon:
            epsilon_path = paths["arrays"] / "epsilon.npy"
            np.save(epsilon_path, epsilon_array)
            artifacts["epsilon_array"] = str(epsilon_path)
        finite_epsilon = np.abs(epsilon_array[np.isfinite(epsilon_array)])
        if finite_epsilon.size:
            metrics["epsilon_abs_max"] = float(np.max(finite_epsilon))

    field_arrays: dict[str, np.ndarray] = {}
    if context.dft_fields is not None:
        frequency_path = paths["arrays"] / "field_frequencies.npy"
        np.save(frequency_path, np.asarray(context.field_frequencies, dtype=float))
        artifacts["field_frequencies"] = str(frequency_path)
        for name, component in context.dft_components.items():
            arrays = [
                np.asarray(simulation.get_dft_array(context.dft_fields, component, index))
                for index in range(len(context.field_frequencies))
            ]
            array = arrays[len(arrays) // 2]
            field_arrays[name] = array
            stacked = np.stack(arrays, axis=0)
            finite_field = np.abs(stacked[np.isfinite(stacked)])
            if finite_field.size:
                metrics[f"{name}_abs_max"] = float(np.max(finite_field))
            if cfg.output.save_fields:
                if len(arrays) == 1:
                    field_path = paths["arrays"] / f"field_{name}.npy"
                    np.save(field_path, array)
                    artifacts[f"field_{name}_array"] = str(field_path)
                else:
                    field_path = paths["arrays"] / f"field_{name}_spectrum.npz"
                    np.savez_compressed(
                        field_path,
                        values=stacked,
                        frequencies=np.asarray(context.field_frequencies, dtype=float),
                        wavelengths_um=1.0 / np.asarray(context.field_frequencies, dtype=float),
                    )
                    artifacts[f"field_{name}_spectrum"] = str(field_path)

        if cfg.output.save_fields:
            try:
                x, y, z, weights = simulation.get_array_metadata(dft_cell=context.dft_fields)
                coordinate_path = paths["arrays"] / "coordinates.npz"
                np.savez(
                    coordinate_path,
                    x=np.asarray(x),
                    y=np.asarray(y),
                    z=np.asarray(z),
                    weights=np.asarray(weights),
                )
                artifacts["array_coordinates"] = str(coordinate_path)
            except Exception as exc:
                warnings.append(f"Could not save array coordinates: {exc}")

    if cfg.output.save_images:
        try:
            image_artifacts = _save_images(epsilon_array, field_arrays, paths["images"])
            artifacts.update(image_artifacts)
        except Exception as exc:
            warnings.append(f"Could not create PNG previews: {exc}")

    return artifacts, metrics, warnings


def _save_images(
    epsilon_array: np.ndarray | None,
    field_arrays: Mapping[str, np.ndarray],
    image_directory: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    artifacts: dict[str, str] = {}
    if epsilon_array is not None:
        data = np.real(_two_dimensional_slice(epsilon_array))
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = axis.imshow(data.T, origin="lower", cmap="viridis", aspect="equal")
        axis.set_title("Relative permittivity at monitor frequency")
        axis.set_xlabel("x grid index")
        axis.set_ylabel("y grid index")
        figure.colorbar(image, ax=axis, label="Re(epsilon)")
        path = image_directory / "epsilon.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        artifacts["epsilon_image"] = str(path)

    for name, array in field_arrays.items():
        data = np.abs(_two_dimensional_slice(array))
        figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
        image = axis.imshow(data.T, origin="lower", cmap="magma", aspect="equal")
        axis.set_title(f"|{name}| at monitor frequency")
        axis.set_xlabel("x grid index")
        axis.set_ylabel("y grid index")
        figure.colorbar(image, ax=axis, label=f"|{name}|")
        path = image_directory / f"field_{name}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        artifacts[f"field_{name}_image"] = str(path)
    return artifacts


def _two_dimensional_slice(array: np.ndarray) -> np.ndarray:
    value = np.squeeze(np.asarray(array))
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value.reshape(value.shape[0], 1)
    while value.ndim > 2:
        value = np.take(value, value.shape[-1] // 2, axis=-1)
    return value


def _resolve_cell_size(
    cfg: SimulationConfig, built: GeometryBuild
) -> tuple[float, float, float]:
    if cfg.cell_size_um is not None:
        return cfg.cell_size_um
    extent = _estimate_geometry_extent(cfg, built)
    periodic = set(built.periodic_axes)
    hints = built.hints or {}
    suggested = hints.get("suggested_cell_size_um", (None, None, None))
    active_axes = ("x", "y") if cfg.dimensions == 2 else ("x", "y", "z")
    result = [0.0, 0.0, 0.0]
    for axis in active_axes:
        index = _AXIS_INDEX[axis]
        if axis in periodic and index < len(suggested) and suggested[index] is not None:
            result[index] = float(suggested[index])
        else:
            result[index] = (
                extent[index]
                + 2 * cfg.padding_um
                + 2 * cfg.pml_thickness_um
            )
    if cfg.dimensions == 2:
        result[2] = 0.0
    return tuple(result)


def _estimate_geometry_extent(
    cfg: SimulationConfig, built: GeometryBuild
) -> tuple[float, float, float]:
    p = built.parameters
    span = cfg.transverse_span_um
    name = built.name
    if name == "nanoparticle_sphere":
        return (2 * p["radius"], 2 * p["radius"], 2 * p["radius"])
    if name == "nanoparticle_disk":
        return (2 * p["radius"], 2 * p["radius"], p["height"])
    if name == "nanocube":
        return (p["size_x"], p["size_y"], p["size_z"])
    if name == "ellipsoid":
        return (p["size_x"], p["size_y"], p["size_z"])
    if name == "nanorod":
        return (p["length"], p["diameter"], p["diameter"])
    if name == "sphere_dimer":
        return (4 * p["radius"] + p["gap"], 2 * p["radius"], 2 * p["radius"])
    if name == "disk_dimer":
        return (4 * p["radius"] + p["gap"], 2 * p["radius"], p["height"])
    if name in {"core_shell_sphere", "core_shell_cylinder"}:
        height = 2 * p["outer_radius"] if name.endswith("sphere") else p["height"]
        return (2 * p["outer_radius"], 2 * p["outer_radius"], height)
    if name == "nanoring":
        return (2 * p["outer_radius"], 2 * p["outer_radius"], p["height"])
    if name == "bowtie_antenna":
        return (
            2 * p["triangle_length"] + p["gap"],
            p["triangle_width"],
            p["height"],
        )
    if name == "planar_interface":
        return (span, p["metal_depth"], span)
    if name == "metal_film":
        return (span, p["thickness"], span)
    if name == "kretschmann_stack":
        return (span, p["metal_thickness"] + p["substrate_depth"], span)
    if name == "multilayer_stack":
        return (span, sum(float(layer["thickness"]) for layer in p["layers"]), span)
    if name in {"nanohole_film", "nanoslit_film", "nanohole_array"}:
        return (span, p["film_thickness"], span)
    if name == "ridge_grating":
        return (
            p["period"] * p["periods"],
            p["base_thickness"] + p["ridge_height"],
            span,
        )
    if name == "groove_grating":
        return (p["period"] * p["periods"], p["metal_depth"], span)
    if name == "particle_array":
        if p["shape"] == "cube":
            return (p["size"], p["size"], p["height"])
        return (2 * p["radius"], 2 * p["radius"], p["height"])
    if name == "particle_line_on_substrate":
        return line_extents(p)
    if name == "mim_waveguide":
        return (span, p["gap"] + 2 * p["metal_thickness"], span)
    if name == "nanoparticle_on_mirror":
        return (span, p["mirror_depth"] + p["gap"] + 2 * p["radius"], span)
    if name == "nanocube_on_mirror":
        return (span, p["mirror_depth"] + p["gap"] + p["size"], span)
    if name == "tip_substrate":
        return (span, p["substrate_depth"] + p["gap"] + p["tip_length"], span)
    return (span, span, span)


def _resolve_source_direction(
    cfg: SimulationConfig, built: GeometryBuild
) -> tuple[str, int]:
    if cfg.source.axis is not None:
        axis = cfg.source.axis
    elif built.name in {
        "planar_interface", "metal_film", "kretschmann_stack", "multilayer_stack",
        "nanohole_film", "nanoslit_film", "ridge_grating", "groove_grating",
        "particle_array", "nanohole_array", "nanoparticle_on_mirror",
        "nanocube_on_mirror", "tip_substrate",
    }:
        axis = "y"
    else:
        axis = "x"
    if axis == "z" and cfg.dimensions == 2:
        raise SimulationConfigError("2-D simulations cannot propagate along z.")
    if cfg.source.direction is not None:
        direction = cfg.source.direction
    else:
        direction = -1 if axis == "y" and built.name != "mim_waveguide" else 1
    return axis, direction


def _resolve_pml_axes(cfg: SimulationConfig, periodic_axes: set[str]) -> set[str]:
    active = {"x", "y"} if cfg.dimensions == 2 else {"x", "y", "z"}
    if cfg.pml_axes is None:
        result = active - periodic_axes
    else:
        result = set(cfg.pml_axes)
        conflict = result & periodic_axes
        if conflict:
            raise SimulationConfigError(
                f"Axes cannot be both periodic and PML: {sorted(conflict)}."
            )
    if cfg.pml_thickness_um == 0:
        return set()
    return result


def _build_pml(mp: Any, thickness: float, axes: set[str]) -> list[Any]:
    if thickness <= 0:
        return []
    return [mp.PML(thickness, direction=_direction(mp, axis)) for axis in sorted(axes)]


def _frequency_parameters(
    wavelength_range_um: tuple[float, float], nfreq: int
) -> tuple[float, float, int]:
    wavelength_min, wavelength_max = _normalise_band(wavelength_range_um)
    frequency_min = 1.0 / wavelength_max
    frequency_max = 1.0 / wavelength_min
    fcen = 0.5 * (frequency_min + frequency_max)
    df = frequency_max - frequency_min
    return fcen, df, 1 if df == 0 else int(nfreq)


def _component(mp: Any, name: str) -> Any:
    candidates = {
        "ex": "Ex", "ey": "Ey", "ez": "Ez",
        "hx": "Hx", "hy": "Hy", "hz": "Hz",
        "dx": "Dx", "dy": "Dy", "dz": "Dz",
        "bx": "Bx", "by": "By", "bz": "Bz",
    }
    attribute = candidates.get(name.lower())
    if attribute is None or not hasattr(mp, attribute):
        raise SimulationConfigError(f"Unsupported MEEP field component {name!r}.")
    return getattr(mp, attribute)


def _direction(mp: Any, axis: str) -> Any:
    attribute = axis.upper()
    if not hasattr(mp, attribute):
        raise SimulationConfigError(f"MEEP does not expose direction {attribute}.")
    return getattr(mp, attribute)


def _import_meep() -> Any:
    try:
        return import_module("meep")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "MEEP is not importable. Open the project through WSL and select the "
            "Miniforge 'plasmon' Python interpreter."
        ) from exc


def _output_paths(cfg: SimulationConfig, run_id: str) -> dict[str, Path]:
    if cfg.output.experiment_directory:
        root = Path(cfg.output.experiment_directory).expanduser().resolve()
        stage = cfg.output.stage_name or run_id
        run = root / "stages" / stage
        return {
            "root": root,
            "run": run,
            "arrays": run / "arrays",
            "images": run / "figures",
        }
    root = (
        Path(cfg.output.results_root).expanduser().resolve()
        if cfg.output.results_root
        else find_project_root() / "data" / "results"
    )
    return {
        "root": root,
        "run": root / "runs" / run_id,
        "arrays": root / "arrays" / run_id,
        "images": root / "images" / run_id,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _mime_type(path: Path) -> str | None:
    return {
        ".json": "application/json",
        ".csv": "text/csv",
        ".npy": "application/octet-stream",
        ".npz": "application/octet-stream",
        ".png": "image/png",
    }.get(path.suffix.lower())


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    return value


def _normalise_band(value: Any) -> tuple[float, float]:
    if isinstance(value, Real) and not isinstance(value, bool):
        values = (float(value), float(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(float(item) for item in value)
        if len(values) == 1:
            values = (values[0], values[0])
        elif len(values) != 2:
            raise SimulationConfigError("Wavelength range must contain one or two values.")
    else:
        raise SimulationConfigError("Wavelength range must be a number or two-value sequence.")
    if any(not math.isfinite(item) or item <= 0 for item in values):
        raise SimulationConfigError("Wavelengths must be finite positive numbers.")
    return min(values), max(values)


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise SimulationConfigError(f"{name} must contain exactly three numbers.")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise SimulationConfigError(f"{name} must contain finite numbers.")
    return result


def _cell_vector(value: Any, dimensions: int) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SimulationConfigError("cell_size_um must be a sequence.")
    if len(value) == 2 and dimensions == 2:
        value = (*value, 0.0)
    result = _vector3(value, "cell_size_um")
    return (result[0], result[1], 0.0) if dimensions == 2 else result


def _positive(name: str, value: Any, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise SimulationConfigError(f"{name} must be a finite number.")
    if (allow_zero and value < 0) or (not allow_zero and value <= 0):
        comparison = "non-negative" if allow_zero else "greater than zero"
        raise SimulationConfigError(f"{name} must be {comparison}.")


def _reject_unknown(cls: type, data: Mapping[str, Any], label: str) -> None:
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SimulationConfigError(
            f"Unknown {label} setting(s): {', '.join(unknown)}. "
            f"Allowed: {', '.join(sorted(allowed))}."
        )


def _rename_keys(data: dict[str, Any], aliases: Mapping[str, str]) -> None:
    for old, new in aliases.items():
        if old in data:
            if new in data:
                raise SimulationConfigError(f"Do not supply both {old!r} and {new!r}.")
            data[new] = data.pop(old)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


__all__ = [
    "FluxConfig",
    "OutputConfig",
    "SimulationConfig",
    "SimulationConfigError",
    "SimulationContext",
    "SimulationResult",
    "SourceConfig",
    "build_simulation",
    "create_simulation",
    "run",
    "run_simulation",
    "simulate",
    "validate_config",
]

