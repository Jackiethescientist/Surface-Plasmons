"""
Edit ``SIMULATION_CONFIG`` below, then run from the project root:

    python scripts/run_single_v5.py --precision fast
    python scripts/run_single_v5.py --precision sweep
    python scripts/run_single_v5.py --precision final

The research CSV contains only the independent and dependent variables chosen
for this project.  Internal MEEP files remain available for reproducibility,
but numerical run-time/debug metrics are not copied into the research CSV.
Before each solver stage, V5 prints an estimated wall-time range. Estimates
automatically calibrate to the computer after a stage finishes.
"""

from __future__ import annotations

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from surface_plasmons.analysis import analyze_spectrum
import surface_plasmons.simulation as simulation_core
from surface_plasmons.simulation import SimulationResult, create_simulation, run_simulation


# ---------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH SIMULATION
# ---------------------------------------------------------------------------

DEFAULT_SIMULATION_NAME = "gold sphere 40 nm 3D two stage"

# Select a mode with --precision fast, --precision sweep, or --precision final.
# The command-line option changes these values automatically; it never changes
# the scientific geometry, material, wavelength range, or polarisation.
PRECISION_MODES: dict[str, dict[str, float | int]] = {
    "fast": {
        "broadband_resolution": 60,
        "resonance_resolution": 80,
        "nfreq": 101,
        "dft_decay_tolerance": 1e-5,
        "minimum_run_time": 20.0,
        "maximum_run_time": 100.0,
        "resonance_relative_bandwidth": 0.05,
    },
    "sweep": {
        "broadband_resolution": 80,
        "resonance_resolution": 120,
        "nfreq": 151,
        "dft_decay_tolerance": 1e-6,
        "minimum_run_time": 20.0,
        "maximum_run_time": 200.0,
        "resonance_relative_bandwidth": 0.05,
    },
    "final": {
        "broadband_resolution": 120,
        "resonance_resolution": 160,
        "nfreq": 301,
        "dft_decay_tolerance": 1e-8,
        "minimum_run_time": 20.0,
        "maximum_run_time": 400.0,
        "resonance_relative_bandwidth": 0.02,
    },
}
DEFAULT_PRECISION_MODE = "fast"
BROADBAND_RESOLUTION = int(
    PRECISION_MODES[DEFAULT_PRECISION_MODE]["broadband_resolution"]
)
RESONANCE_RESOLUTION = int(
    PRECISION_MODES[DEFAULT_PRECISION_MODE]["resonance_resolution"]
)
RESONANCE_RELATIVE_BANDWIDTH = float(
    PRECISION_MODES[DEFAULT_PRECISION_MODE]["resonance_relative_bandwidth"]
)
RESONANCE_FIELD_REGION_SIZE_UM = [0.30, 0.30, 0.30]

# Compatible references are stored here and reused on later runs. A cache key
# includes all settings that affect the empty-cell calculation. Particle
# material is intentionally absent because there is no particle in that run.
REFERENCE_CACHE_ENABLED = True
REFERENCE_CACHE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "reference_cache_v5"
)
# Used only before this process has measured the local computer. Later stages
# automatically replace it with an effective rate measured from completed runs.
DEFAULT_CELL_UPDATES_PER_SECOND = 25_000_000.0

# All lengths are in micrometres: 0.040 um = 40 nm.
SIMULATION_CONFIG: dict[str, Any] = {
    "geometry_name": "nanoparticle_sphere",
    "geometry_parameters": {
        "particle": "au",       # Particle material
        "background": "air",   # Surrounding dielectric/background
        "radius": 0.040,         # Particle size / geometry dimension
    },
    # A real 3-D MEEP domain. Runtime and memory use are much larger than in 2-D.
    "dimensions": 3,
    "resolution": BROADBAND_RESOLUTION,
    "dft_decay_tolerance": PRECISION_MODES[DEFAULT_PRECISION_MODE]["dft_decay_tolerance"],
    "minimum_run_time": PRECISION_MODES[DEFAULT_PRECISION_MODE]["minimum_run_time"],
    "maximum_run_time": PRECISION_MODES[DEFAULT_PRECISION_MODE]["maximum_run_time"],
    "source": {
        "axis": "z",                          # Source propagation axis
        "wavelength_range_um": [0.45, 0.80],  # Incident wavelength range
        "component": "Ez",                    # Polarisation / field component
    },
    "flux": {
        "nfreq": 301,
        "normalization": "empty_cell",
    },
    "output": {
        "save_config": True,
        "save_summary": True,
        "save_spectra": True,
        "save_epsilon": False,
        # The broadband stage finds the resonance; it does not store large 3-D fields.
        "save_fields": False,
        "save_images": True,
        # Save the full electric-field vector so |E|^2 uses Ex, Ey and Ez.
        "field_components": ["Ex", "Ey", "Ez"],
    },
    "notes": "",
    "tags": ["single", "research", "3d", "two-stage"],
}

# The response used to locate resonances and calculate FWHM/Q.
# "auto" chooses extinction, then scattering, then absorption, depending on
# which physically named spectrum is available.
RESONANCE_RESPONSE = "auto"
PEAK_PROMINENCE: float | None = None
MAXIMUM_MODES = 10


# ---------------------------------------------------------------------------
# RESEARCH RESULT DEFINITIONS -- shared with run_sweep.py
# ---------------------------------------------------------------------------

RESEARCH_SPECTRA = ("absorption", "scattering", "extinction")

RESEARCH_CSV_COLUMNS = [
    # Organisational identifiers (not scientific result variables).
    "simulation_name",
    "run_id",
    # Independent variables from the research-aim discussion.
    "particle_material",
    "background_material",
    "geometry",
    "particle_size_um",
    "aspect_ratio",
    "other_geometry_dimensions",
    "incident_wavelength_um",
    "polarisation_field_component",
    # Dependent variables from the research-aim discussion.
    "resonance_wavelength_um",
    "peak_spectral_response",
    "peak_spectral_response_type",
    "electric_field_enhancement",
    "field_spatial_distribution",
    "resonance_linewidth_fwhm_um",
    "quality_factor",
    "absorption",
    "peak_absorption",
    "scattering",
    "peak_scattering",
    "extinction",
    "peak_extinction",
    "number_of_modes",
    "mode_locations_um",
]


def safe_name(name: str) -> str:
    """Convert a display name into a safe, readable filename component."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("The simulation name must contain at least one letter or number.")
    return cleaned[:70]


def make_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_name(name)}__{timestamp}"


def _get(result: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _finite_peak(values: Sequence[Any] | None) -> float | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else None


def _same_length(*arrays: Sequence[Any]) -> bool:
    lengths = {len(array) for array in arrays}
    return len(lengths) == 1


def research_spectra(result: Mapping[str, Any] | Any) -> tuple[dict[str, list[float]], list[str]]:
    """Return only absorption/scattering/extinction, deriving one if justified."""

    raw = _get(result, "spectra", {}) or {}
    spectra: dict[str, list[float]] = {}
    warnings: list[str] = []
    for name in RESEARCH_SPECTRA:
        if name in raw:
            spectra[name] = [float(value) for value in raw[name]]

    # For compatible cross-section/fraction spectra: extinction = absorption + scattering.
    if "extinction" not in spectra and all(name in spectra for name in ("absorption", "scattering")):
        if _same_length(spectra["absorption"], spectra["scattering"]):
            spectra["extinction"] = (
                np.asarray(spectra["absorption"]) + np.asarray(spectra["scattering"])
            ).tolist()
            warnings.append("Extinction was derived as absorption + scattering.")
    if "scattering" not in spectra and all(name in spectra for name in ("extinction", "absorption")):
        if _same_length(spectra["extinction"], spectra["absorption"]):
            spectra["scattering"] = (
                np.asarray(spectra["extinction"]) - np.asarray(spectra["absorption"])
            ).tolist()
            warnings.append("Scattering was derived as extinction - absorption.")

    missing = [name for name in RESEARCH_SPECTRA if name not in spectra]
    if missing:
        warnings.append(
            "Unavailable research spectra were left blank: " + ", ".join(missing) + "."
        )
    return spectra, warnings


def _response_name(spectra: Mapping[str, Sequence[float]]) -> str | None:
    if RESONANCE_RESPONSE != "auto":
        return RESONANCE_RESPONSE if RESONANCE_RESPONSE in spectra else None
    return next((name for name in ("extinction", "scattering", "absorption") if name in spectra), None)


def _particle_size_item(parameters: Mapping[str, Any]) -> tuple[str | None, float | None]:
    for key in ("radius", "diameter", "size", "size_x", "outer_radius"):
        value = parameters.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    return None, None


def _dimension_summary(parameters: Mapping[str, Any]) -> str:
    ignored = {
        "particle", "background", "metal", "dielectric", "core", "shell",
        "dimensions", "center", "wavelength_um", "allow_out_of_range", "allow_unstable",
    }
    particle_size_key, _ = _particle_size_item(parameters)
    dimensions = {
        str(key): value
        for key, value in parameters.items()
        if key not in ignored
        and key != particle_size_key
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    return json.dumps(dimensions, ensure_ascii=False, sort_keys=True)


def _particle_size(parameters: Mapping[str, Any]) -> float | None:
    return _particle_size_item(parameters)[1]


def _aspect_ratio(parameters: Mapping[str, Any]) -> float | None:
    pairs = (
        ("length", "diameter"),
        ("size_x", "size_y"),
        ("triangle_length", "triangle_width"),
        ("tip_length", "tip_radius"),
    )
    for numerator, denominator in pairs:
        first = parameters.get(numerator)
        second = parameters.get(denominator)
        if isinstance(first, (int, float)) and isinstance(second, (int, float)) and second:
            ratio = float(first) / float(second)
            if numerator == "tip_length" and denominator == "tip_radius":
                ratio /= 2.0
            return ratio
    return None


def independent_variables(result: Mapping[str, Any] | Any) -> dict[str, Any]:
    config = _get(result, "config", {}) or {}
    parameters = config.get("geometry_parameters", {}) or {}
    source = config.get("source", {}) or {}
    particle = next(
        (parameters[key] for key in ("particle", "metal", "tip", "core") if key in parameters),
        "",
    )
    background = next(
        (parameters[key] for key in ("background", "dielectric", "spacer") if key in parameters),
        "",
    )
    return {
        "particle_material": particle,
        "background_material": background,
        "geometry": config.get("geometry_name", _get(result, "geometry_name", "")),
        "particle_size_um": _particle_size(parameters),
        "aspect_ratio": _aspect_ratio(parameters),
        "other_geometry_dimensions": _dimension_summary(parameters),
        "polarisation_field_component": source.get("component", ""),
    }


def _central_plane(array: np.ndarray) -> np.ndarray:
    plane = np.asarray(array)
    while plane.ndim > 2:
        plane = np.take(plane, plane.shape[-1] // 2, axis=-1)
    if plane.ndim == 1:
        plane = plane[np.newaxis, :]
    return plane


def field_research_outputs(result: Mapping[str, Any] | Any) -> tuple[float | None, str, list[str]]:
    """Calculate total electric |E|^2 enhancement and save its spatial map."""

    artifacts = _get(result, "artifacts", {}) or {}
    config = _get(result, "config", {}) or {}
    output_directory = _get(result, "output_directory")
    run_id = str(_get(result, "run_id", "run"))
    warnings: list[str] = []
    arrays: list[np.ndarray] = []
    used_components: list[str] = []

    for component in ("Ex", "Ey", "Ez"):
        path_value = artifacts.get(f"field_{component}_array")
        if path_value and Path(path_value).is_file():
            arrays.append(np.load(path_value))
            used_components.append(component)

    if not arrays:
        return None, "", ["No saved electric-field array was available."]
    if len({array.shape for array in arrays}) != 1:
        return None, "", ["Electric-field component arrays had different shapes."]

    intensity = sum(np.abs(array) ** 2 for array in arrays)
    source = config.get("source", {}) or {}
    incident_amplitude = float(source.get("amplitude", 1.0))
    enhancement = intensity / (incident_amplitude**2)
    finite = enhancement[np.isfinite(enhancement)]
    maximum = float(np.max(finite)) if finite.size else None
    if not output_directory:
        return maximum, "", warnings

    destination = Path(output_directory) / f"{run_id}_field_spatial_distribution.png"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots(figsize=(7.0, 5.5))
        image = axis.imshow(
            _central_plane(enhancement).T,
            origin="lower",
            aspect="equal",
            cmap="inferno",
        )
        axis.set_xlabel("x grid index")
        axis.set_ylabel("y grid index")
        axis.set_title("Electric-field enhancement |E|² / |E₀|²")
        figure.colorbar(image, ax=axis, label="Field enhancement")
        figure.tight_layout()
        figure.savefig(destination, dpi=220, bbox_inches="tight")
        plt.close(figure)
    except Exception as exc:
        warnings.append(f"Could not create field spatial-distribution plot: {exc}")
        return maximum, "", warnings

    warnings.append("Field enhancement used components: " + ", ".join(used_components) + ".")
    warnings.append(
        "Field enhancement was normalised to the configured source amplitude. "
        "An empty-cell field reference is preferable for publication-grade enhancement."
    )
    return maximum, str(destination), warnings


def analyze_research_result(
    result: Mapping[str, Any] | Any,
    *,
    create_field_plot: bool = True,
) -> dict[str, Any]:
    """Extract only the selected independent/dependent research variables."""

    wavelengths = [float(value) for value in (_get(result, "wavelengths_um", []) or [])]
    spectra, warnings = research_spectra(result)
    response_name = _response_name(spectra)
    resonance = None
    modes: list[float] = []
    if response_name and wavelengths and len(spectra[response_name]) == len(wavelengths):
        try:
            report = analyze_spectrum(
                wavelengths,
                spectra[response_name],
                monitor_name=response_name,
                prominence=PEAK_PROMINENCE,
                max_peaks=MAXIMUM_MODES,
            )
            resonance = report.primary_peak
            modes = [float(peak.wavelength_um) for peak in report.peaks]
        except Exception as exc:
            warnings.append(f"Could not analyse resonances: {exc}")
    else:
        warnings.append("No valid absorption/scattering/extinction response was available for resonance analysis.")

    if create_field_plot:
        field_enhancement, field_plot, field_warnings = field_research_outputs(result)
        warnings.extend(field_warnings)
    else:
        field_enhancement, field_plot = None, ""

    summary = {
        **independent_variables(result),
        "resonance_wavelength_um": resonance.wavelength_um if resonance else None,
        "peak_spectral_response": resonance.value if resonance else None,
        "peak_spectral_response_type": response_name or "",
        "electric_field_enhancement": field_enhancement,
        "field_spatial_distribution": field_plot,
        "resonance_linewidth_fwhm_um": resonance.width_um if resonance else None,
        "quality_factor": resonance.quality_factor if resonance else None,
        "peak_absorption": _finite_peak(spectra.get("absorption")),
        "peak_scattering": _finite_peak(spectra.get("scattering")),
        "peak_extinction": _finite_peak(spectra.get("extinction")),
        "number_of_modes": len(modes),
        "mode_locations_um": json.dumps(modes),
    }
    return {
        "simulation_name": (_get(result, "config", {}) or {}).get("run_name", ""),
        "run_id": str(_get(result, "run_id", "")),
        "wavelengths_um": wavelengths,
        "spectra": spectra,
        "summary": summary,
        "warnings": warnings,
    }


def research_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Create long-format rows: one row per incident wavelength."""

    wavelengths = list(analysis.get("wavelengths_um", []))
    spectra = analysis.get("spectra", {})
    sample_count = max(
        [len(wavelengths)] + [len(values) for values in spectra.values()],
        default=0,
    )
    common = {
        "simulation_name": analysis.get("simulation_name", ""),
        "run_id": analysis.get("run_id", ""),
        **analysis.get("summary", {}),
    }
    rows: list[dict[str, Any]] = []
    for index in range(max(sample_count, 1)):
        row = {
            **common,
            "incident_wavelength_um": wavelengths[index] if index < len(wavelengths) else "",
        }
        for name in RESEARCH_SPECTRA:
            values = spectra.get(name, [])
            row[name] = values[index] if index < len(values) else ""
        rows.append({column: row.get(column, "") for column in RESEARCH_CSV_COLUMNS})
    return rows


def write_research_csv(rows: Sequence[Mapping[str, Any]], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESEARCH_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_research_spectra(analysis: Mapping[str, Any], destination: str | Path) -> Path | None:
    spectra = analysis.get("spectra", {})
    wavelengths = np.asarray(analysis.get("wavelengths_um", []), dtype=float)
    available = [name for name in RESEARCH_SPECTRA if name in spectra]
    if wavelengths.size == 0 or not available:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(wavelengths)
    figure, axis = plt.subplots(figsize=(8.0, 5.2))
    for name in available:
        values = np.asarray(spectra[name], dtype=float)
        if values.size == wavelengths.size:
            axis.plot(wavelengths[order] * 1000, values[order], label=name.title(), linewidth=1.8)
    modes = json.loads(analysis.get("summary", {}).get("mode_locations_um", "[]"))
    for index, wavelength in enumerate(modes):
        axis.axvline(
            float(wavelength) * 1000,
            color="#555555",
            linestyle="--",
            alpha=0.55,
            label="Detected mode" if index == 0 else None,
        )
    axis.set_xlabel("Incident wavelength (nm)")
    axis.set_ylabel("Spectral response")
    axis.set_title("Absorption, scattering and extinction")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return path


def get_precision_profile(name: str) -> dict[str, float | int]:
    """Return one validated precision profile without exposing shared state."""

    if name not in PRECISION_MODES:
        raise ValueError(
            f"Unknown precision mode {name!r}; choose from {sorted(PRECISION_MODES)}."
        )
    return deepcopy(PRECISION_MODES[name])


def apply_broadband_precision(
    config: Mapping[str, Any],
    profile: Mapping[str, float | int],
) -> dict[str, Any]:
    """Apply numerical controls while leaving the scientific variables unchanged."""

    updated = deepcopy(dict(config))
    updated["resolution"] = int(profile["broadband_resolution"])
    updated["dft_decay_tolerance"] = float(profile["dft_decay_tolerance"])
    updated["minimum_run_time"] = float(profile["minimum_run_time"])
    updated["maximum_run_time"] = float(profile["maximum_run_time"])
    # Suppress MEEP's time-step messages; V5 prints estimated and actual times.
    updated["progress_interval"] = 1.0e9
    flux = dict(updated.get("flux", {}) or {})
    flux["nfreq"] = int(profile["nfreq"])
    updated["flux"] = flux
    return updated


def _default_background_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Identify the material that remains when geometry is removed."""

    geometry_name = str(config.get("geometry_name", ""))
    parameters = dict(config.get("geometry_parameters", {}) or {})
    key = {
        "planar_interface": "dielectric",
        "kretschmann_stack": "analyte",
        "mim_waveguide": "dielectric",
    }.get(geometry_name, "background")
    return {"parameter": key, "material": parameters.get(key, "air")}


def _reference_payload(context: Any, kind: str) -> dict[str, Any]:
    """Describe every resolved setting that can affect an empty-cell reference."""

    import meep as mp

    config = context.config.to_dict()
    payload: dict[str, Any] = {
        "cache_schema": 1,
        "kind": kind,
        "meep_version": str(getattr(mp, "__version__", "unknown")),
        "processors": int(getattr(mp, "count_processors", lambda: 1)()),
        "dimensions": config.get("dimensions"),
        "resolution": config.get("resolution"),
        "cell_size_um": list(context.cell_size_um),
        "geometry_name": config.get("geometry_name"),
        "background": _default_background_signature(config),
        "pml_thickness_um": config.get("pml_thickness_um"),
        "pml_axes": config.get("pml_axes"),
        "courant": config.get("courant"),
        "k_point": config.get("k_point"),
        "force_complex_fields": config.get("force_complex_fields"),
        "source": config.get("source"),
        "source_axis": context.source_axis,
        "source_direction": context.source_direction,
        "source_center_um": list(context.source_center_um),
        "stopping": {
            "method": config.get("stop_method"),
            "dft_decay_tolerance": config.get("dft_decay_tolerance"),
            "minimum_run_time": config.get("minimum_run_time"),
            "maximum_run_time": config.get("maximum_run_time"),
        },
    }
    if kind == "broadband_flux":
        payload["flux"] = config.get("flux")
        payload["flux_positions_um"] = {
            name: list(position)
            for name, position in sorted(context.flux_positions_um.items())
        }
    elif kind == "resonance_fields":
        output = dict(config.get("output", {}) or {})
        payload["field_frequency"] = float(context.field_frequency)
        payload["field_components"] = sorted(context.dft_components)
        payload["field_region_center_um"] = output.get("field_region_center_um")
        payload["field_region_size_um"] = output.get("field_region_size_um")
    else:
        raise ValueError(f"Unsupported reference-cache kind: {kind}")
    return payload


def _cache_key(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


@contextmanager
def _cache_lock(cache_path: Path) -> Iterator[None]:
    """Prevent two serial sweep processes from calculating the same reference."""

    import fcntl

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_flux_data(path: Path, flux_data: Any) -> None:
    fields = tuple(getattr(flux_data, "_fields", ()))
    if not fields:
        raise TypeError("MEEP FluxData did not expose named NumPy-array fields.")
    arrays = {f"field_{name}": np.asarray(getattr(flux_data, name)) for name in fields}
    _atomic_save_npz(path, arrays)


def _load_flux_data(path: Path, template: Any) -> Any:
    fields = tuple(getattr(template, "_fields", ()))
    if not fields:
        raise TypeError("MEEP FluxData template did not expose named fields.")
    with np.load(path, allow_pickle=False) as saved:
        values = [np.asarray(saved[f"field_{name}"]) for name in fields]
    return type(template)(*values)


def _source_end_time(simulation: Any) -> float:
    """Best available estimate of when all Gaussian sources have switched off."""

    endings: list[float] = []
    for source in getattr(simulation, "sources", ()):
        source_time = getattr(source, "src", None)
        last_time = getattr(source_time, "last_time", None)
        if callable(last_time):
            try:
                endings.append(float(last_time()))
            except (TypeError, ValueError):
                pass
    return max(endings, default=0.0)


def _grid_cell_count(context: Any) -> int:
    resolution = float(context.config.resolution)
    counts = [max(1, int(math.ceil(abs(float(size)) * resolution))) for size in context.cell_size_um]
    if int(context.config.dimensions) == 2:
        counts[2] = 1
    return math.prod(counts)


def _time_bounds(context: Any) -> tuple[float, float]:
    config = context.config
    if config.source.kind == "continuous" or config.stop_method == "fixed_time":
        value = float(config.fixed_run_time)
        return value, value
    source_end = _source_end_time(context.simulation)
    if config.stop_method == "dft_decay":
        return (
            source_end + float(config.minimum_run_time),
            source_end + float(config.maximum_run_time),
        )
    # A field-decay stop has no strict maximum. Use the configured DFT maximum
    # as a conservative planning estimate without changing the actual stop rule.
    return (
        source_end + float(config.field_decay_interval),
        source_end + float(config.maximum_run_time),
    )


def _work_units(context: Any, meep_time: float) -> float:
    config = context.config
    approximate_steps = max(1.0, float(meep_time) * float(config.resolution) / float(config.courant))
    return float(_grid_cell_count(context)) * approximate_steps


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.1f} min"
    hours = minutes / 60
    if hours < 36:
        return f"{hours:.1f} h"
    return f"{hours / 24:.1f} days"


class _RuntimeEstimator:
    """Estimate from grid work, then calibrate against this computer."""

    def __init__(self) -> None:
        self.cell_updates_per_second = DEFAULT_CELL_UPDATES_PER_SECOND
        self.samples = 0

    def announce(self, context: Any, label: str) -> None:
        minimum_time, maximum_time = _time_bounds(context)
        minimum_seconds = _work_units(context, minimum_time) / self.cell_updates_per_second
        maximum_seconds = _work_units(context, maximum_time) / self.cell_updates_per_second
        # Before calibration, allow a wider hardware-dependent range. Afterwards
        # the remaining spread is mainly the unknown decay stopping time.
        if self.samples == 0:
            minimum_seconds *= 0.4
            maximum_seconds *= 2.5
            basis = "configuration estimate; not yet calibrated to this computer"
        else:
            minimum_seconds *= 0.75
            maximum_seconds *= 1.5
            basis = f"calibrated from {self.samples} completed stage(s)"
        grid = _grid_cell_count(context)
        resolution = int(context.config.resolution)
        print(
            f"{label} estimated time: {_format_duration(minimum_seconds)} to "
            f"{_format_duration(maximum_seconds)}"
        )
        print(
            f"  estimate basis: {grid:,} grid cells, resolution {resolution}, "
            f"{basis}"
        )

    def observe(self, context: Any, elapsed_seconds: float) -> None:
        actual_time = float(context.simulation.meep_time())
        if elapsed_seconds <= 0 or actual_time <= 0:
            return
        observed = _work_units(context, actual_time) / elapsed_seconds
        if not math.isfinite(observed) or observed <= 0:
            return
        if self.samples == 0:
            self.cell_updates_per_second = observed
        else:
            # Give recent stages some weight without making estimates jump wildly.
            self.cell_updates_per_second = (
                0.65 * self.cell_updates_per_second + 0.35 * observed
            )
        self.samples += 1


_RUNTIME_ESTIMATOR = _RuntimeEstimator()


def _run_with_estimate(context: Any, label: str, execute: Any) -> None:
    _RUNTIME_ESTIMATOR.announce(context, label)
    started = perf_counter()
    execute(context)
    elapsed = perf_counter() - started
    _RUNTIME_ESTIMATOR.observe(context, elapsed)
    print(f"{label} actual time: {_format_duration(elapsed)}")


def _context_stage_label(context: Any) -> str:
    identity = str(context.run_id)
    if identity.endswith("_normalization"):
        return "Broadband empty reference"
    if identity.endswith("__empty_reference"):
        return "Resonance empty reference"
    if "__resonance_field" in identity:
        return "Resonance particle"
    return "Broadband particle"


@contextmanager
def simulation_runtime_features(cache_enabled: bool = True) -> Iterator[None]:
    """Install time estimation and serial empty-reference reuse."""

    original_execute = simulation_core._execute

    def cached_execute(context: Any) -> None:
        import meep as mp

        config = context.config
        is_normalisation = (
            str(context.run_id).endswith("_normalization")
            and config.flux.enabled
            and config.flux.normalization == "empty_cell"
        )
        # FluxData is tied to the processor count and chunk layout. V5 caching
        # deliberately remains serial; MPI should run the original calculation.
        processor_count = int(getattr(mp, "count_processors", lambda: 1)())
        if not is_normalisation or not cache_enabled or processor_count != 1:
            _run_with_estimate(context, _context_stage_label(context), original_execute)
            return

        monitor = context.flux_monitors[config.flux.input_name]
        payload = _reference_payload(context, "broadband_flux")
        key = _cache_key(payload)
        cache_path = REFERENCE_CACHE_DIRECTORY / "broadband_flux" / f"{key}.npz"
        with _cache_lock(cache_path):
            if cache_path.is_file():
                try:
                    context.simulation.init_sim()
                    template = context.simulation.get_flux_data(monitor)
                    cached = _load_flux_data(cache_path, template)
                    context.simulation.load_flux_data(monitor, cached)
                    print(f"Reference cache hit: broadband flux {key}")
                    return
                except Exception as exc:
                    print(f"Reference cache warning ({key}): {exc}; recalculating.")

            print(f"Reference cache miss: broadband flux {key}; calculating once.")
            _run_with_estimate(context, "Broadband empty reference", original_execute)
            _save_flux_data(cache_path, context.simulation.get_flux_data(monitor))
            _atomic_save_json(cache_path.with_suffix(".json"), payload)

    simulation_core._execute = cached_execute
    try:
        yield
    finally:
        simulation_core._execute = original_execute


def _resonance_band(wavelength_um: float, relative_bandwidth: float) -> list[float]:
    """Return a narrow wavelength band whose centre frequency is the resonance."""

    if wavelength_um <= 0:
        raise ValueError("Resonance wavelength must be positive.")
    if not 0 < relative_bandwidth < 1:
        raise ValueError("Resonance relative bandwidth must be between 0 and 1.")
    centre_frequency = 1.0 / wavelength_um
    half_width = 0.5 * relative_bandwidth * centre_frequency
    return [
        1.0 / (centre_frequency + half_width),
        1.0 / (centre_frequency - half_width),
    ]


def prepare_resonance_config(
    broadband_result: Mapping[str, Any] | SimulationResult,
    resonance_wavelength_um: float,
    *,
    precision: Mapping[str, float | int],
    run_name: str | None = None,
) -> dict[str, Any]:
    """Create the high-resolution narrowband configuration from the broadband run."""

    config = deepcopy(_get(broadband_result, "config", {}) or {})
    if not config:
        raise ValueError("The broadband result does not contain its simulation config.")
    broadband_id = str(_get(broadband_result, "run_id", "broadband"))
    source = dict(config.get("source", {}) or {})
    source.update(
        {
            "kind": "gaussian",
            "wavelength_range_um": _resonance_band(
                resonance_wavelength_um,
                float(precision["resonance_relative_bandwidth"]),
            ),
        }
    )
    config["source"] = source
    flux = dict(config.get("flux", {}) or {})
    flux.update({"enabled": False, "nfreq": 1, "normalization": "none"})
    config["flux"] = flux
    parameters = config.get("geometry_parameters", {}) or {}
    centre = parameters.get("center", [0.0, 0.0, 0.0])
    output = dict(config.get("output", {}) or {})
    output.update(
        {
            "save_config": True,
            "save_summary": True,
            "save_spectra": False,
            "save_epsilon": False,
            "save_fields": True,
            "save_images": False,
            "field_components": ["Ex", "Ey", "Ez"],
            "field_region_center_um": list(centre),
            "field_region_size_um": list(RESONANCE_FIELD_REGION_SIZE_UM),
        }
    )
    config["output"] = output
    config["dimensions"] = 3
    config["resolution"] = int(precision["resonance_resolution"])
    config["dft_decay_tolerance"] = float(precision["dft_decay_tolerance"])
    config["minimum_run_time"] = float(precision["minimum_run_time"])
    config["maximum_run_time"] = float(precision["maximum_run_time"])
    config["run_id"] = f"{broadband_id}__resonance_field"
    config["run_name"] = run_name or f"{config.get('run_name') or broadband_id} resonance field"
    config["tags"] = list(
        dict.fromkeys([*config.get("tags", []), "stage:resonance-field", "three-plane-field"])
    )
    return config


def _run_empty_reference_fields(
    config: Mapping[str, Any],
    *,
    cache_enabled: bool,
) -> dict[str, np.ndarray]:
    """Load or run the narrowband empty-cell field reference."""

    import meep as mp

    reference = create_simulation(
        config,
        run_id=f"{config.get('run_id', 'resonance')}__empty_reference",
        empty_geometry=True,
    )
    if reference.dft_fields is None:
        raise RuntimeError("The empty reference did not create a DFT field monitor.")
    processor_count = int(getattr(mp, "count_processors", lambda: 1)())
    use_cache = cache_enabled and processor_count == 1
    payload = _reference_payload(reference, "resonance_fields")
    key = _cache_key(payload)
    cache_path = REFERENCE_CACHE_DIRECTORY / "resonance_fields" / f"{key}.npz"

    def load_cached() -> dict[str, np.ndarray]:
        with np.load(cache_path, allow_pickle=False) as saved:
            fields = {
                component: np.asarray(saved[component])
                for component in ("Ex", "Ey", "Ez")
            }
        shapes = {array.shape for array in fields.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 3:
            raise ValueError("Cached resonance fields have invalid array shapes.")
        return fields

    def calculate() -> dict[str, np.ndarray]:
        simulation_core._execute(reference)
        return {
            name: np.asarray(
                reference.simulation.get_dft_array(reference.dft_fields, component, 0)
            )
            for name, component in reference.dft_components.items()
        }

    try:
        if not use_cache:
            return calculate()
        with _cache_lock(cache_path):
            if cache_path.is_file():
                try:
                    fields = load_cached()
                    print(f"Reference cache hit: resonance fields {key}")
                    return fields
                except Exception as exc:
                    print(f"Reference cache warning ({key}): {exc}; recalculating.")
            print(f"Reference cache miss: resonance fields {key}; calculating once.")
            fields = calculate()
            _atomic_save_npz(cache_path, fields)
            _atomic_save_json(cache_path.with_suffix(".json"), payload)
            return fields
    finally:
        reference.simulation.reset_meep()


def _load_saved_fields(result: Mapping[str, Any] | SimulationResult) -> dict[str, np.ndarray]:
    artifacts = _get(result, "artifacts", {}) or {}
    fields: dict[str, np.ndarray] = {}
    for component in ("Ex", "Ey", "Ez"):
        path_value = artifacts.get(f"field_{component}_array")
        if not path_value or not Path(path_value).is_file():
            raise FileNotFoundError(f"Missing saved {component} field array.")
        fields[component] = np.asarray(np.load(path_value))
    shapes = {array.shape for array in fields.values()}
    if len(shapes) != 1:
        raise ValueError("Saved Ex, Ey and Ez arrays do not have matching shapes.")
    if len(next(iter(shapes))) != 3:
        raise ValueError("The resonance fields are not three-dimensional arrays.")
    return fields


def _coordinate_axes(
    result: Mapping[str, Any] | SimulationResult,
    shape: Sequence[int],
) -> dict[str, np.ndarray]:
    artifacts = _get(result, "artifacts", {}) or {}
    path_value = artifacts.get("array_coordinates")
    if path_value and Path(path_value).is_file():
        with np.load(path_value) as coordinates:
            axes = {
                "x": np.asarray(coordinates["x"], dtype=float).reshape(-1),
                "y": np.asarray(coordinates["y"], dtype=float).reshape(-1),
                "z": np.asarray(coordinates["z"], dtype=float).reshape(-1),
            }
        if tuple(len(axes[name]) for name in ("x", "y", "z")) == tuple(shape):
            return axes
    return {
        "x": np.arange(shape[0], dtype=float),
        "y": np.arange(shape[1], dtype=float),
        "z": np.arange(shape[2], dtype=float),
    }


def _plane_data(array: np.ndarray, plane: str) -> tuple[np.ndarray, int]:
    if plane == "xy":
        index = array.shape[2] // 2
        return array[:, :, index], index
    if plane == "xz":
        index = array.shape[1] // 2
        return array[:, index, :], index
    if plane == "yz":
        index = array.shape[0] // 2
        return array[index, :, :], index
    raise ValueError(f"Unknown field plane: {plane}")


def create_resonance_field_maps(
    result: Mapping[str, Any] | SimulationResult,
    reference_fields: Mapping[str, np.ndarray],
    resonance_wavelength_um: float,
    *,
    relative_bandwidth: float,
) -> dict[str, Any]:
    """Save Ex/Ey/Ez contributions and total field enhancement on three planes."""

    actual = _load_saved_fields(result)
    if set(reference_fields) != {"Ex", "Ey", "Ez"}:
        raise ValueError("The empty reference must contain Ex, Ey and Ez.")
    if any(reference_fields[name].shape != actual[name].shape for name in actual):
        raise ValueError("Reference and particle field arrays have different shapes.")

    incident_intensity = sum(np.abs(reference_fields[name]) ** 2 for name in actual)
    incident_max = float(np.nanmax(incident_intensity))
    if not math.isfinite(incident_max) or incident_max <= 0:
        raise ValueError("The empty-cell incident field intensity is zero or invalid.")
    safe_incident = np.where(incident_intensity > incident_max * 1e-12, incident_intensity, np.nan)
    contributions = {
        name: np.abs(array) ** 2 / safe_incident
        for name, array in actual.items()
    }
    total_enhancement = sum(contributions.values())
    finite = total_enhancement[np.isfinite(total_enhancement)]
    maximum = float(np.max(finite)) if finite.size else None

    output_directory = Path(str(_get(result, "output_directory", ".")))
    analysis_directory = output_directory / "resonance_field_maps"
    analysis_directory.mkdir(parents=True, exist_ok=True)
    run_id = str(_get(result, "run_id", "resonance"))
    array_path = analysis_directory / f"{run_id}_total_field_enhancement.npy"
    np.save(array_path, total_enhancement)
    coordinates = _coordinate_axes(result, total_enhancement.shape)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane_axes = {
        "xy": ("x", "y", "z"),
        "xz": ("x", "z", "y"),
        "yz": ("y", "z", "x"),
    }
    plane_images: dict[str, str] = {}
    for plane, (horizontal, vertical, fixed) in plane_axes.items():
        total_plane, fixed_index = _plane_data(total_enhancement, plane)
        component_planes = {
            name: _plane_data(values, plane)[0]
            for name, values in contributions.items()
        }
        finite_plane = total_plane[np.isfinite(total_plane)]
        colour_max = float(np.max(finite_plane)) if finite_plane.size else 1.0
        if colour_max <= 0:
            colour_max = 1.0
        h_values = coordinates[horizontal]
        v_values = coordinates[vertical]
        extent = [
            float(h_values[0]), float(h_values[-1]),
            float(v_values[0]), float(v_values[-1]),
        ]
        fixed_value = float(coordinates[fixed][fixed_index])
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True)
        panels = [
            ("Ex", component_planes["Ex"]),
            ("Ey", component_planes["Ey"]),
            ("Ez", component_planes["Ez"]),
            ("Total |E|²", total_plane),
        ]
        image = None
        for axis, (label, values) in zip(axes.ravel(), panels):
            image = axis.imshow(
                values.T,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="inferno",
                vmin=0.0,
                vmax=colour_max,
            )
            axis.set_title(label)
            axis.set_xlabel(f"{horizontal} (µm)")
            axis.set_ylabel(f"{vertical} (µm)")
        if image is not None:
            figure.colorbar(
                image,
                ax=axes.ravel().tolist(),
                label="Contribution to |E|² / |E₀|²",
                shrink=0.86,
            )
        figure.suptitle(
            f"Resonance field at {resonance_wavelength_um * 1000:.2f} nm — "
            f"{plane} plane ({fixed} = {fixed_value:.4g} µm)"
        )
        image_path = analysis_directory / f"{run_id}_field_{plane}.png"
        figure.savefig(image_path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        plane_images[plane] = str(image_path)

    manifest_path = analysis_directory / f"{run_id}_field_manifest.json"
    result_config = dict(_get(result, "config", {}) or {})
    manifest = {
        "run_id": run_id,
        "resonance_wavelength_um": resonance_wavelength_um,
        "resolution": result_config.get("resolution"),
        "relative_bandwidth": relative_bandwidth,
        "field_components": ["Ex", "Ey", "Ez"],
        "planes": plane_images,
        "total_field_enhancement_array": str(array_path),
        "maximum_field_enhancement": maximum,
        "normalization": "pointwise empty-cell total electric-field intensity",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def run_resonance_field_stage(
    broadband_result: Mapping[str, Any] | SimulationResult,
    broadband_analysis: dict[str, Any],
    *,
    precision: Mapping[str, float | int],
    cache_enabled: bool = REFERENCE_CACHE_ENABLED,
    run_name: str | None = None,
) -> tuple[SimulationResult, dict[str, Any]]:
    """Run and analyse the high-resolution narrowband resonance-field stage."""

    resonance_value = broadband_analysis.get("summary", {}).get("resonance_wavelength_um")
    if resonance_value in (None, ""):
        raise ValueError("No broadband resonance was detected; narrowband stage was not run.")
    resonance_wavelength_um = float(resonance_value)
    config = prepare_resonance_config(
        broadband_result,
        resonance_wavelength_um,
        precision=precision,
        run_name=run_name,
    )
    print("  Resonance reference: checking compatible empty-cell cache")
    reference_fields = _run_empty_reference_fields(
        config,
        cache_enabled=cache_enabled,
    )
    print("  Resonance particle: running high-resolution field calculation")
    result = run_simulation(config)
    field_report = create_resonance_field_maps(
        result,
        reference_fields,
        resonance_wavelength_um,
        relative_bandwidth=float(precision["resonance_relative_bandwidth"]),
    )
    broadband_analysis["summary"]["electric_field_enhancement"] = field_report[
        "maximum_field_enhancement"
    ]
    broadband_analysis["summary"]["field_spatial_distribution"] = json.dumps(
        field_report["planes"], sort_keys=True
    )
    return result, field_report


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_SIMULATION_NAME)
    parser.add_argument(
        "--precision",
        choices=tuple(PRECISION_MODES),
        default=DEFAULT_PRECISION_MODE,
        help="fast = debug, sweep = ordinary study, final = convergence-quality output",
    )
    parser.add_argument(
        "--no-reference-cache",
        action="store_true",
        help="Recalculate empty-cell references instead of loading compatible cached data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_name = args.name.strip()
    if not display_name:
        raise SystemExit("Simulation name cannot be empty.")

    precision = get_precision_profile(args.precision)
    cache_enabled = REFERENCE_CACHE_ENABLED and not args.no_reference_cache
    config = apply_broadband_precision(SIMULATION_CONFIG, precision)
    run_id = make_run_id(display_name)
    config["run_name"] = display_name
    config["run_id"] = run_id
    config["tags"] = list(
        dict.fromkeys(
            [
                *config.get("tags", []),
                f"name:{safe_name(display_name)}",
                f"precision:{args.precision}",
            ]
        )
    )

    print(f"Starting broadband stage: {display_name}")
    print(f"Broadband run ID / results name: {run_id}")
    print(f"Precision mode: {args.precision}")
    print(f"Broadband resolution: {int(precision['broadband_resolution'])} pixels/µm")
    print(f"Broadband frequencies: {int(precision['nfreq'])}")
    print(f"Reference cache: {'enabled' if cache_enabled else 'disabled'}")
    with simulation_runtime_features(cache_enabled):
        result: SimulationResult = run_simulation(config)
        analysis = analyze_research_result(result, create_field_plot=False)

        resonance_result: SimulationResult | None = None
        field_report: dict[str, Any] | None = None
        resonance_wavelength = analysis["summary"].get("resonance_wavelength_um")
        if resonance_wavelength is not None:
            print(
                f"Starting resonance-field stage at {float(resonance_wavelength) * 1000:.3f} nm "
                f"with resolution {int(precision['resonance_resolution'])} pixels/µm"
            )
            resonance_result, field_report = run_resonance_field_stage(
                result,
                analysis,
                precision=precision,
                cache_enabled=cache_enabled,
                run_name=f"{display_name} resonance field",
            )
        else:
            analysis["warnings"].append(
                "No broadband resonance was detected, so the resonance-field stage was skipped."
            )

    output_directory = Path(result.output_directory or ".")
    csv_path = write_research_csv(
        research_rows(analysis),
        output_directory / f"{run_id}_research_results.csv",
    )
    spectrum_plot = plot_research_spectra(
        analysis,
        output_directory / f"{run_id}_research_spectra.png",
    )

    print(f"Broadband status: {result.status}")
    if resonance_result is not None:
        print(f"Resonance-field status: {resonance_result.status}")
    print(f"Research CSV: {csv_path}")
    if spectrum_plot:
        print(f"Research spectra plot: {spectrum_plot}")
    if field_report:
        print(f"Resonance xy field map: {field_report['planes']['xy']}")
        print(f"Resonance xz field map: {field_report['planes']['xz']}")
        print(f"Resonance yz field map: {field_report['planes']['yz']}")
        print(f"Resonance field manifest: {field_report['manifest']}")
    for warning in analysis["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
