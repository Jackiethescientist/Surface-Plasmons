"""
Run a single simulation for research purposes, producing a CSV of results and
automatic visualisations.  The simulation is defined by a configuration dictionary
"""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

from surface_plasmons.analysis import analyze_spectrum
from surface_plasmons.simulation import SimulationResult, run_simulation


# ---------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH SIMULATION
# ---------------------------------------------------------------------------

DEFAULT_SIMULATION_NAME = "gold sphere 40 nm 3D"

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
    "resolution": 80,
    "source": {
        "wavelength_range_um": [0.45, 0.80],  # Incident wavelength range
        "component": "Ez",                    # Polarisation / field component
    },
    "flux": {
        "nfreq": 101,
        "normalization": "empty_cell",
    },
    "output": {
        "save_config": True,
        "save_summary": True,
        "save_spectra": True,
        "save_epsilon": False,
        "save_fields": True,
        "save_images": False,
        # Save the full electric-field vector so |E|^2 uses Ex, Ey and Ez.
        "field_components": ["Ex", "Ey", "Ez"],
    },
    "notes": "",
    "tags": ["single", "research", "3d"],
}

# The response used to locate resonances and calculate FWHM/Q.
# "auto" chooses extinction, then scattering, then absorption, depending on
# which physically named spectrum is available.
RESONANCE_RESPONSE = "auto"
PEAK_PROMINENCE: float | None = None
MAXIMUM_MODES = 10


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


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_SIMULATION_NAME)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_name = args.name.strip()
    if not display_name:
        raise SystemExit("Simulation name cannot be empty.")

    config = deepcopy(SIMULATION_CONFIG)
    run_id = make_run_id(display_name)
    config["run_name"] = display_name
    config["run_id"] = run_id
    config["tags"] = list(dict.fromkeys([*config.get("tags", []), f"name:{safe_name(display_name)}"]))

    print(f"Starting simulation: {display_name}")
    print(f"Run ID / results name: {run_id}")
    result: SimulationResult = run_simulation(config)
    analysis = analyze_research_result(result)

    output_directory = Path(result.output_directory or ".")
    csv_path = write_research_csv(
        research_rows(analysis),
        output_directory / f"{run_id}_research_results.csv",
    )
    spectrum_plot = plot_research_spectra(
        analysis,
        output_directory / f"{run_id}_research_spectra.png",
    )

    print(f"Status: {result.status}")
    print(f"Research CSV: {csv_path}")
    if spectrum_plot:
        print(f"Research spectra plot: {spectrum_plot}")
    field_plot = analysis["summary"].get("field_spatial_distribution")
    if field_plot:
        print(f"Field spatial distribution: {field_plot}")
    for warning in analysis["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
