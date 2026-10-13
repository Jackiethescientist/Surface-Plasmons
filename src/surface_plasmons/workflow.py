"""Broadband discovery and selected-frequency field maps for particle lines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any, Mapping, Sequence
import csv
import json
import math
import shutil

import numpy as np

from .database import canonical_json, find_project_root
from .materials import dielectric_function
from .particle_line import particle_line_positions
from .project import (
    estimate_runtime,
    geometry_parameters,
    make_simulation_config,
    normalise_project_config,
)
from .results import ExperimentWorkspace, config_hash, safe_simulation_name
from .simulation import SimulationResult, run_simulation


class WorkflowError(RuntimeError):
    """Raised when compatible simulation stages cannot be combined safely."""


@dataclass(slots=True)
class StageFields:
    frequencies: np.ndarray
    fields: dict[str, np.ndarray]
    coordinates: dict[str, np.ndarray]
    output_directory: Path

    @property
    def wavelengths_um(self) -> np.ndarray:
        return 1.0 / self.frequencies

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(next(iter(self.fields.values())).shape[1:])


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    return path


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def print_stage_estimate(label: str, simulation_config: Mapping[str, Any]) -> None:
    estimate = estimate_runtime(simulation_config)
    print(
        f"{label}: approximately {_format_duration(float(estimate['estimated_min_seconds']))} "
        f"to {_format_duration(float(estimate['estimated_max_seconds']))}; "
        f"{int(estimate['grid_cells']):,} grid cells; "
        f"about {float(estimate['estimated_memory_gb']):.2f} GB base field memory."
    )


def _reference_signature(simulation_config: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(canonical_json(simulation_config))
    value.pop("run_id", None)
    value.pop("run_name", None)
    value.pop("tags", None)
    value["record_database"] = False
    output = dict(value.get("output", {}))
    output.pop("experiment_directory", None)
    output.pop("stage_name", None)
    value["output"] = output
    parameters = dict(value.get("geometry_parameters", {}))
    if parameters.get("include_particles") is False:
        # These settings do not change a substrate-only calculation. Cell size
        # and monitor geometry remain in the signature through their resolved
        # simulation/output values, so arrays are reused only when compatible.
        for key in (
            "particle",
            "particle_shape",
            "particle_count",
            "radius",
            "cube_size",
            "spacing",
            "particle_surface_gap",
            "defects",
        ):
            parameters.pop(key, None)
        value["geometry_parameters"] = parameters
    return value


def _reference_directory(simulation_config: Mapping[str, Any]) -> Path:
    signature = _reference_signature(simulation_config)
    digest = config_hash(signature)[:24]
    return find_project_root() / "data" / "reference_cache_v7" / digest


def _field_paths(stage_directory: Path) -> dict[str, Path]:
    arrays = stage_directory / "arrays"
    result: dict[str, Path] = {}
    for component in ("Ex", "Ey", "Ez"):
        spectrum = arrays / f"field_{component}_spectrum.npz"
        single = arrays / f"field_{component}.npy"
        result[component] = spectrum if spectrum.is_file() else single
    return result


def load_stage_fields(stage_directory: str | Path) -> StageFields:
    directory = Path(stage_directory).resolve()
    paths = _field_paths(directory)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise WorkflowError(f"Missing field arrays in {directory}: {', '.join(missing)}.")
    fields: dict[str, np.ndarray] = {}
    frequencies: np.ndarray | None = None
    for name, path in paths.items():
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                values = np.asarray(saved["values"])
                current_frequencies = np.asarray(saved["frequencies"], dtype=float)
        else:
            values = np.asarray(np.load(path, allow_pickle=False))[np.newaxis, ...]
            frequency_path = directory / "arrays" / "field_frequencies.npy"
            current_frequencies = np.asarray(np.load(frequency_path, allow_pickle=False), dtype=float)
        if frequencies is None:
            frequencies = current_frequencies
        elif not np.allclose(frequencies, current_frequencies):
            raise WorkflowError("Ex, Ey, and Ez use different frequency samples.")
        fields[name] = values
    shapes = {array.shape for array in fields.values()}
    if len(shapes) != 1:
        raise WorkflowError("Ex, Ey, and Ez field arrays have different shapes.")
    coordinate_path = directory / "arrays" / "coordinates.npz"
    if not coordinate_path.is_file():
        raise WorkflowError(f"Missing coordinate metadata: {coordinate_path}")
    with np.load(coordinate_path, allow_pickle=False) as saved:
        coordinates = {
            axis: np.asarray(saved[axis], dtype=float).reshape(-1)
            for axis in ("x", "y", "z")
        }
    return StageFields(
        frequencies=np.asarray(frequencies, dtype=float),
        fields=fields,
        coordinates=coordinates,
        output_directory=directory,
    )


def _run_reference(
    project_config: Mapping[str, Any],
    *,
    wavelength_um: float | None,
    cache_enabled: bool,
    resolution: int | None = None,
    padding_um: float | None = None,
) -> StageFields:
    temporary_config = make_simulation_config(
        project_config,
        experiment_directory=find_project_root() / "data" / "reference_cache_v7" / "pending",
        stage_name="reference",
        reference=True,
        wavelength_um=wavelength_um,
        resolution=resolution,
        padding_um=padding_um,
    )
    cache_directory = _reference_directory(temporary_config)
    stage_directory = cache_directory / "stages" / "reference"
    complete_marker = cache_directory / "complete.json"
    if cache_enabled and complete_marker.is_file():
        try:
            fields = load_stage_fields(stage_directory)
            print(f"Reference cache hit: {cache_directory.name}")
            return fields
        except Exception as exc:
            print(f"Reference cache warning: {exc}; recalculating this entry.")
    cache_directory.mkdir(parents=True, exist_ok=True)
    reference_config = make_simulation_config(
        project_config,
        experiment_directory=cache_directory,
        stage_name="reference",
        reference=True,
        wavelength_um=wavelength_um,
        resolution=resolution,
        padding_um=padding_um,
    )
    reference_config["record_database"] = False
    print_stage_estimate("Substrate-only reference", reference_config)
    result = run_simulation(reference_config)
    fields = load_stage_fields(result.output_directory or stage_directory)
    _write_json(
        complete_marker,
        {
            "status": "completed",
            "reference_kind": "substrate_only",
            "signature": _reference_signature(reference_config),
        },
    )
    return fields


def _coordinate_grid(fields: StageFields) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spatial_shape = fields.spatial_shape
    axes = [fields.coordinates[name] for name in ("x", "y", "z")]
    full_shape = tuple(len(axis) for axis in axes)
    point_count = int(np.prod(spatial_shape, dtype=int))
    if int(np.prod(full_shape, dtype=int)) == point_count:
        x, y, z = np.meshgrid(*axes, indexing="ij")
        return x.reshape(-1), y.reshape(-1), z.reshape(-1)
    if len(axes[0]) == point_count:
        return (
            axes[0],
            np.full(point_count, axes[1][0] if axes[1].size else 0.0),
            np.full(point_count, axes[2][0] if axes[2].size else 0.0),
        )
    raise WorkflowError(
        f"Coordinate metadata {full_shape} does not match field shape {spatial_shape}."
    )


def _normalised_fields(
    actual: StageFields,
    reference: StageFields,
    *,
    reference_component: str = "Ex",
    mode: str = "global_complex_reference",
) -> dict[str, np.ndarray]:
    """Put MEEP fields on a dimensionless, explicitly defined scale.

    Broadband Equation-2 comparisons use one complex particle-free reference
    amplitude per frequency. This cancels the pulse spectrum without dividing
    by small pointwise numerical values. Full-volume maps use a pointwise
    reference amplitude. Every returned enhancement is an E/E_reference
    amplitude ratio, never its square.
    """

    if actual.fields.keys() != reference.fields.keys():
        raise WorkflowError("Particle and reference stages contain different components.")
    if not np.allclose(actual.frequencies, reference.frequencies):
        raise WorkflowError("Particle and reference stages use different frequencies.")
    if any(actual.fields[name].shape != reference.fields[name].shape for name in actual.fields):
        raise WorkflowError("Particle and reference field arrays have different shapes.")
    if reference_component not in reference.fields:
        raise WorkflowError(f"Reference component {reference_component!r} is unavailable.")
    actual_intensity = sum(np.abs(value) ** 2 for value in actual.fields.values())
    reference_intensity = sum(np.abs(value) ** 2 for value in reference.fields.values())
    maximum_reference = float(np.nanmax(reference_intensity))
    if not math.isfinite(maximum_reference) or maximum_reference <= 0:
        raise WorkflowError("The particle-free reference field is zero or invalid.")

    result: dict[str, np.ndarray] = {}
    if mode == "global_complex_reference":
        reference_flat = reference.fields[reference_component].reshape(
            len(reference.frequencies), -1
        )
        scales: list[complex] = []
        for values in reference_flat:
            finite = values[np.isfinite(values.real) & np.isfinite(values.imag)]
            if not finite.size:
                raise WorkflowError("No finite complex reference samples were found.")
            scale = complex(np.median(finite.real), np.median(finite.imag))
            if abs(scale) <= max(float(np.max(np.abs(finite))) * 1e-12, 1e-30):
                raise WorkflowError("The complex reference amplitude is zero or unstable.")
            scales.append(scale)
        reshape = (len(scales),) + (1,) * (actual_intensity.ndim - 1)
        complex_reference = np.asarray(scales, dtype=complex).reshape(reshape)
        for name, value in actual.fields.items():
            normalised = value / complex_reference
            result[f"normalised_{name}"] = normalised
            result[f"normalised_reference_{name}"] = (
                reference.fields[name] / complex_reference
            )
            result[f"normalised_scattered_{name}"] = (
                normalised - result[f"normalised_reference_{name}"]
            )
            result[f"{name}_amplitude_ratio"] = np.abs(normalised)
        amplitude_enhancement = np.sqrt(sum(
            result[f"{name}_amplitude_ratio"] ** 2 for name in actual.fields
        ))
        result["reference_complex_scale"] = np.broadcast_to(
            complex_reference, actual_intensity.shape
        )
    elif mode == "pointwise_amplitude_reference":
        safe_reference_intensity = np.where(
            reference_intensity > maximum_reference * 1e-12,
            reference_intensity,
            np.nan,
        )
        safe_reference_amplitude = np.sqrt(safe_reference_intensity)
        reference_field = reference.fields[reference_component]
        reference_component_magnitude = np.abs(reference_field)
        reference_phase = np.where(
            reference_component_magnitude**2 > maximum_reference * 1e-12,
            reference_field / reference_component_magnitude,
            np.nan + 1j * np.nan,
        )
        safe_reference_field = reference_phase * safe_reference_amplitude
        for name, value in actual.fields.items():
            result[f"normalised_{name}"] = value / safe_reference_field
            result[f"normalised_reference_{name}"] = (
                reference.fields[name] / safe_reference_field
            )
            result[f"normalised_scattered_{name}"] = (
                result[f"normalised_{name}"]
                - result[f"normalised_reference_{name}"]
            )
            result[f"{name}_amplitude_ratio"] = (
                np.abs(value) / safe_reference_amplitude
            )
        amplitude_enhancement = np.sqrt(actual_intensity) / safe_reference_amplitude
        result["reference_complex_scale"] = safe_reference_field
    else:
        raise WorkflowError(f"Unknown normalization mode {mode!r}.")

    result["actual_intensity"] = actual_intensity
    result["reference_intensity"] = reference_intensity
    result["amplitude_enhancement"] = amplitude_enhancement
    result["scattered_amplitude_enhancement"] = np.sqrt(
        sum(
            np.abs(result[f"normalised_scattered_{name}"]) ** 2
            for name in actual.fields
        )
    )
    result["enhancement"] = amplitude_enhancement
    return result


def _save_geometry(project_config: Mapping[str, Any], workspace: ExperimentWorkspace) -> None:
    parameters = {
        **geometry_parameters(project_config, include_particles=True),
        "center": (0.0, 0.0, 0.0),
    }
    positions = particle_line_positions(parameters)
    columns = list(positions[0].to_dict())
    table_path = _write_csv(
        workspace.directory / "geometry" / "particle_positions.csv",
        columns,
        [item.to_dict() for item in positions],
    )
    workspace.record_artifact("particle_positions", table_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    geometry = project_config["geometry"]
    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    surface_y = float(geometry["surface_y_um"])
    substrate_depth = float(geometry["substrate_thickness_um"])
    x_values = [item.actual_x_um for item in positions]
    span = max(x_values) - min(x_values) if len(x_values) > 1 else 0.2
    diameter = 2.0 * float(geometry["particle_radius_um"]) if geometry["particle_shape"] == "sphere" else float(geometry["cube_size_um"])
    left = min(x_values) - diameter - 0.08 * max(span, diameter)
    width = max(x_values) - min(x_values) + 2.0 * diameter + 0.16 * max(span, diameter)
    axis.add_patch(Rectangle((left, surface_y - substrate_depth), width, substrate_depth, color="#87a8c7", alpha=0.75))
    for item in positions:
        if geometry["particle_shape"] == "sphere":
            patch = Circle((item.actual_x_um, item.actual_y_um), diameter / 2.0, color="#d7a928")
        else:
            patch = Rectangle((item.actual_x_um - diameter / 2.0, item.actual_y_um - diameter / 2.0), diameter, diameter, color="#d7a928")
        axis.add_patch(patch)
        axis.text(item.actual_x_um, item.actual_y_um + diameter * 0.75, str(item.particle_index), ha="center", fontsize=8)
    axis.axhline(surface_y, color="#315c7d", linewidth=1.2)
    axis.set_xlim(left, left + width)
    axis.set_ylim(surface_y - substrate_depth - 0.03, surface_y + diameter * 2.2)
    axis.set_aspect("equal")
    axis.set_xlabel("x (µm) — particle-line direction")
    axis.set_ylabel("y (µm) — surface normal")
    axis.set_title("Particle-line geometry (x–y view)")
    figure.tight_layout()
    plot_path = workspace.directory / "geometry" / "geometry_xy.png"
    figure.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    workspace.record_artifact("geometry_xy", plot_path)


def _save_dielectric_table(
    project_config: Mapping[str, Any],
    frequencies: np.ndarray,
    workspace: ExperimentWorkspace,
) -> Path:
    geometry = project_config["geometry"]
    materials = {
        "particle": geometry["particle_material"],
        "substrate": geometry["substrate_material"],
        "background": geometry["background_material"],
    }
    rows: list[dict[str, Any]] = []
    for role, material in materials.items():
        epsilon = dielectric_function(
            material,
            frequencies,
            allow_out_of_range=bool(project_config["numerics"]["allow_out_of_range_materials"]),
            allow_unstable=bool(project_config["numerics"]["allow_unstable_materials"]),
        )
        for frequency, value in zip(frequencies, epsilon):
            rows.append(
                {
                    "material_role": role,
                    "material": material,
                    "frequency_1_per_um": float(frequency),
                    "wavelength_nm": float(1000.0 / frequency),
                    "epsilon_real": float(np.real(value)),
                    "epsilon_imag": float(np.imag(value)),
                }
            )
    path = _write_csv(
        workspace.directory / "frequency_scan" / "dielectric_function_vs_frequency.csv",
        ("material_role", "material", "frequency_1_per_um", "wavelength_nm", "epsilon_real", "epsilon_imag"),
        rows,
    )
    workspace.record_artifact("dielectric_function_vs_frequency", path)
    return path


def _save_broadband_results(
    project_config: Mapping[str, Any],
    actual: StageFields,
    reference: StageFields,
    workspace: ExperimentWorkspace,
) -> dict[str, Any]:
    normalization_mode = str(project_config["measurement"]["normalization_mode"])
    reference_component = str(project_config["source"]["polarisation"])
    normalised = _normalised_fields(
        actual,
        reference,
        reference_component=reference_component,
        mode=normalization_mode,
    )
    x, y, z = _coordinate_grid(actual)
    rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    amplitude_matrix = normalised["amplitude_enhancement"].reshape(
        len(actual.frequencies), -1
    )
    actual_flat = {name: value.reshape(len(actual.frequencies), -1) for name, value in actual.fields.items()}
    normalised_flat = {
        name: normalised[f"normalised_{name}"].reshape(len(actual.frequencies), -1)
        for name in actual.fields
    }
    normalised_reference_flat = {
        name: normalised[f"normalised_reference_{name}"].reshape(
            len(actual.frequencies), -1
        )
        for name in actual.fields
    }
    normalised_scattered_flat = {
        name: normalised[f"normalised_scattered_{name}"].reshape(
            len(actual.frequencies), -1
        )
        for name in actual.fields
    }
    scattered_amplitude_matrix = normalised[
        "scattered_amplitude_enhancement"
    ].reshape(len(actual.frequencies), -1)
    reference_scale_flat = normalised["reference_complex_scale"].reshape(
        len(actual.frequencies), -1
    )
    component_amplitudes = {
        name: normalised[f"{name}_amplitude_ratio"].reshape(len(actual.frequencies), -1)
        for name in actual.fields
    }
    for frequency_index, frequency in enumerate(actual.frequencies):
        amplitude = amplitude_matrix[frequency_index]
        finite = np.where(np.isfinite(amplitude), amplitude, np.nan)
        maximum_index = int(np.nanargmax(finite))
        response_rows.append(
            {
                "frequency_1_per_um": float(frequency),
                "wavelength_nm": float(1000.0 / frequency),
                "maximum_field_amplitude_enhancement": float(np.nanmax(amplitude)),
                "mean_field_amplitude_enhancement": float(np.nanmean(amplitude)),
                "maximum_scattered_field_amplitude_ratio": float(
                    np.nanmax(scattered_amplitude_matrix[frequency_index])
                ),
                "mean_scattered_field_amplitude_ratio": float(
                    np.nanmean(scattered_amplitude_matrix[frequency_index])
                ),
                "maximum_field_enhancement": float(np.nanmax(amplitude)),
                "maximum_x_um": float(x[maximum_index]),
                "maximum_y_um": float(y[maximum_index]),
                "maximum_z_um": float(z[maximum_index]),
            }
        )
        for point_index in range(x.size):
            ex = actual_flat["Ex"][frequency_index, point_index]
            ey = actual_flat["Ey"][frequency_index, point_index]
            ez = actual_flat["Ez"][frequency_index, point_index]
            nex = normalised_flat["Ex"][frequency_index, point_index]
            ney = normalised_flat["Ey"][frequency_index, point_index]
            nez = normalised_flat["Ez"][frequency_index, point_index]
            reference_scale = reference_scale_flat[frequency_index, point_index]
            normalised_reference_values = {
                name: normalised_reference_flat[name][frequency_index, point_index]
                for name in ("Ex", "Ey", "Ez")
            }
            normalised_scattered_values = {
                name: normalised_scattered_flat[name][frequency_index, point_index]
                for name in ("Ex", "Ey", "Ez")
            }
            rows.append(
                {
                    "frequency_1_per_um": float(frequency),
                    "wavelength_nm": float(1000.0 / frequency),
                    "x_um": float(x[point_index]),
                    "y_um": float(y[point_index]),
                    "z_um": float(z[point_index]),
                    "abs_Ex": float(abs(ex)),
                    "abs_Ey": float(abs(ey)),
                    "abs_Ez": float(abs(ez)),
                    "Ex_real": float(np.real(ex)),
                    "Ex_imag": float(np.imag(ex)),
                    "Ey_real": float(np.real(ey)),
                    "Ey_imag": float(np.imag(ey)),
                    "Ez_real": float(np.real(ez)),
                    "Ez_imag": float(np.imag(ez)),
                    "E_magnitude": float(np.sqrt(abs(ex) ** 2 + abs(ey) ** 2 + abs(ez) ** 2)),
                    "normalised_Ex_real": float(np.real(nex)),
                    "normalised_Ex_imag": float(np.imag(nex)),
                    "normalised_Ey_real": float(np.real(ney)),
                    "normalised_Ey_imag": float(np.imag(ney)),
                    "normalised_Ez_real": float(np.real(nez)),
                    "normalised_Ez_imag": float(np.imag(nez)),
                    **{
                        f"normalised_reference_{name}_{part}": float(
                            getattr(normalised_reference_values[name], part)
                        )
                        for name in ("Ex", "Ey", "Ez")
                        for part in ("real", "imag")
                    },
                    **{
                        f"normalised_scattered_{name}_{part}": float(
                            getattr(normalised_scattered_values[name], part)
                        )
                        for name in ("Ex", "Ey", "Ez")
                        for part in ("real", "imag")
                    },
                    "reference_scale_real": float(np.real(reference_scale)),
                    "reference_scale_imag": float(np.imag(reference_scale)),
                    "Ex_amplitude_ratio": float(component_amplitudes["Ex"][frequency_index, point_index]),
                    "Ey_amplitude_ratio": float(component_amplitudes["Ey"][frequency_index, point_index]),
                    "Ez_amplitude_ratio": float(component_amplitudes["Ez"][frequency_index, point_index]),
                    "field_amplitude_enhancement": float(amplitude[point_index]),
                    "scattered_field_amplitude_ratio": float(
                        scattered_amplitude_matrix[frequency_index, point_index]
                    ),
                    "total_field_enhancement": float(amplitude[point_index]),
                    "normalization_mode": normalization_mode,
                }
            )
    field_path = _write_csv(
        workspace.directory / "frequency_scan" / "field_strength_vs_position_and_frequency.csv",
        (
            "frequency_1_per_um", "wavelength_nm", "x_um", "y_um", "z_um",
            "abs_Ex", "abs_Ey", "abs_Ez",
            "Ex_real", "Ex_imag", "Ey_real", "Ey_imag", "Ez_real", "Ez_imag",
            "E_magnitude",
            "normalised_Ex_real", "normalised_Ex_imag",
            "normalised_Ey_real", "normalised_Ey_imag",
            "normalised_Ez_real", "normalised_Ez_imag",
            "normalised_reference_Ex_real", "normalised_reference_Ex_imag",
            "normalised_reference_Ey_real", "normalised_reference_Ey_imag",
            "normalised_reference_Ez_real", "normalised_reference_Ez_imag",
            "normalised_scattered_Ex_real", "normalised_scattered_Ex_imag",
            "normalised_scattered_Ey_real", "normalised_scattered_Ey_imag",
            "normalised_scattered_Ez_real", "normalised_scattered_Ez_imag",
            "reference_scale_real", "reference_scale_imag",
            "Ex_amplitude_ratio", "Ey_amplitude_ratio", "Ez_amplitude_ratio",
            "field_amplitude_enhancement", "scattered_field_amplitude_ratio",
            "total_field_enhancement",
            "normalization_mode",
        ),
        rows,
    )
    response_path = _write_csv(
        workspace.directory / "frequency_scan" / "field_response_summary.csv",
        (
            "frequency_1_per_um", "wavelength_nm",
            "maximum_field_amplitude_enhancement", "mean_field_amplitude_enhancement",
            "maximum_scattered_field_amplitude_ratio", "mean_scattered_field_amplitude_ratio",
            "maximum_field_enhancement", "maximum_x_um", "maximum_y_um", "maximum_z_um",
        ),
        response_rows,
    )
    workspace.record_artifact("field_strength_vs_position_and_frequency", field_path)
    workspace.record_artifact("field_response_summary", response_path)
    _save_broadband_plots(project_config, actual, amplitude_matrix, x, workspace)
    _save_dielectric_table(project_config, actual.frequencies, workspace)
    return {
        "response_rows": response_rows,
        "amplitude_enhancement": amplitude_matrix,
        "enhancement": amplitude_matrix,
        "x": x,
        "maximum_field_amplitude_enhancement": float(np.nanmax(amplitude_matrix)),
        "maximum_field_enhancement": float(np.nanmax(amplitude_matrix)),
    }


def _save_broadband_plots(
    project_config: Mapping[str, Any],
    fields: StageFields,
    amplitude_enhancement: np.ndarray,
    x: np.ndarray,
    workspace: ExperimentWorkspace,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelengths = 1000.0 / fields.frequencies
    wavelength_order = np.argsort(wavelengths)
    x_order = np.argsort(x)
    # Equation 2 is an electric-field equation, so every research plot uses the
    # field-amplitude ratio E/E_reference rather than its square.
    amplitude_matrix = amplitude_enhancement[wavelength_order][:, x_order]
    figure, axis = plt.subplots(figsize=(9.0, 5.7))
    image = axis.imshow(
        amplitude_matrix,
        origin="lower",
        aspect="auto",
        extent=[float(x[x_order][0]), float(x[x_order][-1]), float(wavelengths[wavelength_order][0]), float(wavelengths[wavelength_order][-1])],
        cmap="viridis",
    )
    axis.set_xlabel("Position along particle line, x (µm)")
    axis.set_ylabel("Wavelength (nm)")
    axis.set_title("Electric-field amplitude enhancement versus position and wavelength")
    figure.colorbar(
        image,
        ax=axis,
        label=r"Field-amplitude ratio, $|E|/|E_{\mathrm{ref}}|$",
    )
    figure.tight_layout()
    heatmap_path = workspace.directory / "frequency_scan" / "field_position_frequency_heatmap.png"
    figure.savefig(heatmap_path, dpi=230, bbox_inches="tight")
    plt.close(figure)
    workspace.record_artifact("field_position_frequency_heatmap", heatmap_path)

    parameters = {
        **geometry_parameters(project_config, include_particles=True),
        "center": (0.0, 0.0, 0.0),
    }
    positions = particle_line_positions(parameters)
    figure, axis = plt.subplots(figsize=(8.8, 5.5))
    for item in positions:
        index = int(np.argmin(np.abs(x - item.actual_x_um)))
        axis.plot(
            wavelengths[wavelength_order],
            amplitude_enhancement[wavelength_order, index],
            linewidth=1.3,
            label=f"Above particle {item.particle_index}",
        )
    axis.set_xlabel("Wavelength (nm)")
    axis.set_ylabel(r"Field-amplitude ratio, $|E|/|E_{\mathrm{ref}}|$")
    axis.set_title("External field response sampled above each particle")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, fontsize=8, frameon=False)
    figure.tight_layout()
    spectra_path = workspace.directory / "frequency_scan" / "particle_field_spectra.png"
    figure.savefig(spectra_path, dpi=230, bbox_inches="tight")
    plt.close(figure)
    workspace.record_artifact("particle_field_spectra", spectra_path)


def _select_wavelengths(
    response_rows: Sequence[Mapping[str, Any]],
    selection: str | Sequence[float],
    maximum_auto: int,
) -> list[float]:
    if isinstance(selection, str):
        mode = selection.strip().lower()
        if mode == "none":
            return []
        if mode == "auto":
            wavelengths = np.asarray([float(row["wavelength_nm"]) for row in response_rows])
            response = np.asarray(
                [
                    float(
                        row["maximum_field_amplitude_enhancement"]
                        if "maximum_field_amplitude_enhancement" in row
                        else row["maximum_field_enhancement"]
                    )
                    for row in response_rows
                ]
            )
            try:
                from scipy.signal import find_peaks
                indices, properties = find_peaks(response, prominence=max(1e-12, 0.05 * np.nanmax(response)))
                if indices.size:
                    ranked = indices[np.argsort(response[indices])[::-1]]
                else:
                    ranked = np.asarray([int(np.nanargmax(response))])
            except Exception:
                ranked = np.asarray([int(np.nanargmax(response))])
            return [float(wavelengths[index]) for index in ranked[: max(1, int(maximum_auto))]]
        values = [token.strip() for token in selection.split(",") if token.strip()]
        try:
            return [float(value) for value in values]
        except ValueError as exc:
            raise WorkflowError("Field-map wavelengths must be 'none', 'auto', or comma-separated nm values.") from exc
    return [float(value) for value in selection]


def _plane(array: np.ndarray, name: str) -> tuple[np.ndarray, int]:
    if array.ndim != 3:
        raise WorkflowError(f"Full field maps require 3-D arrays; received shape {array.shape}.")
    if name == "xy":
        index = array.shape[2] // 2
        return array[:, :, index], index
    if name == "xz":
        index = array.shape[1] // 2
        return array[:, index, :], index
    if name == "yz":
        index = array.shape[0] // 2
        return array[index, :, :], index
    raise WorkflowError(f"Unknown plane {name!r}.")


def _save_field_map(
    wavelength_nm: float,
    actual: StageFields,
    reference: StageFields,
    workspace: ExperimentWorkspace,
    project_config: Mapping[str, Any],
) -> dict[str, Any]:
    normalised = _normalised_fields(
        actual,
        reference,
        reference_component=str(project_config["source"]["polarisation"]),
        mode="pointwise_amplitude_reference",
    )
    folder = workspace.directory / "field_maps" / f"{wavelength_nm:.3f}nm"
    # ``exist_ok`` supports an explicit --resume after interruption during
    # post-processing. Completed experiments cannot be resumed or overwritten.
    folder.mkdir(parents=True, exist_ok=True)
    for component, values in actual.fields.items():
        np.save(folder / f"field_{component}.npy", values[0])
    amplitude_enhancement = normalised["amplitude_enhancement"][0]
    np.save(folder / "field_total_enhancement.npy", amplitude_enhancement)
    np.save(folder / "field_amplitude_enhancement.npy", amplitude_enhancement)
    np.savez(folder / "coordinates.npz", **actual.coordinates)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plane_axes = {
        "xy": ("x", "y", "z"),
        "xz": ("x", "z", "y"),
        "yz": ("y", "z", "x"),
    }
    images: dict[str, str] = {}
    for plane_name, (horizontal, vertical, fixed) in plane_axes.items():
        total_plane, fixed_index = _plane(amplitude_enhancement, plane_name)
        component_planes = {
            name: _plane(normalised[f"{name}_amplitude_ratio"][0], plane_name)[0]
            for name in ("Ex", "Ey", "Ez")
        }
        h = actual.coordinates[horizontal]
        v = actual.coordinates[vertical]
        extent = [float(h[0]), float(h[-1]), float(v[0]), float(v[-1])]
        colour_max = float(np.nanmax(total_plane))
        figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
        image = None
        for axis, (label, data) in zip(
            axes.ravel(),
            [("|Ex|", component_planes["Ex"]), ("|Ey|", component_planes["Ey"]), ("|Ez|", component_planes["Ez"]), ("Total |E|", total_plane)],
        ):
            image = axis.imshow(data.T, origin="lower", extent=extent, aspect="equal", cmap="viridis", vmin=0.0, vmax=max(colour_max, 1e-30))
            axis.set_title(label)
            axis.set_xlabel(f"{horizontal} (µm)")
            axis.set_ylabel(f"{vertical} (µm)")
        figure.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=r"Field-amplitude ratio, $|E_{\mathrm{component}}|/|E_{\mathrm{ref}}|$",
            shrink=0.85,
        )
        fixed_value = float(actual.coordinates[fixed][fixed_index])
        figure.suptitle(f"{wavelength_nm:.3f} nm — {plane_name} plane ({fixed}={fixed_value:.4g} µm)")
        path = folder / f"field_components_{plane_name}.png"
        figure.savefig(path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        images[plane_name] = str(path.relative_to(workspace.directory))

        amplitude_plane, _ = _plane(amplitude_enhancement, plane_name)
        figure, axis = plt.subplots(figsize=(8.2, 6.3), constrained_layout=True)
        amplitude_image = axis.imshow(
            amplitude_plane.T,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="viridis",
            vmin=0.0,
            vmax=max(float(np.nanmax(amplitude_plane)), 1e-30),
        )
        axis.set_title(f"Electric-field amplitude enhancement — {plane_name} plane")
        axis.set_xlabel(f"{horizontal} position (µm)")
        axis.set_ylabel(f"{vertical} position (µm)")
        figure.colorbar(
            amplitude_image,
            ax=axis,
            label=r"Field-amplitude ratio, $|E|/|E_{\mathrm{ref}}|$",
        )
        amplitude_path = folder / f"field_amplitude_{plane_name}.png"
        figure.savefig(amplitude_path, dpi=240, bbox_inches="tight")
        plt.close(figure)
        images[f"{plane_name}_amplitude"] = str(
            amplitude_path.relative_to(workspace.directory)
        )
    maximum_index = np.unravel_index(
        int(np.nanargmax(amplitude_enhancement)), amplitude_enhancement.shape
    )
    maximum_location = {
        axis: float(actual.coordinates[axis][maximum_index[index]])
        for index, axis in enumerate(("x", "y", "z"))
    }
    manifest = {
        "wavelength_nm": wavelength_nm,
        "maximum_field_amplitude_enhancement": float(np.nanmax(amplitude_enhancement)),
        "maximum_field_enhancement": float(np.nanmax(amplitude_enhancement)),
        "maximum_location_um": maximum_location,
        "normalization": "pointwise particle-free electric-field amplitude",
        "scale_definition": "|E| / |E_reference|",
        "components": ["Ex", "Ey", "Ez"],
        "images": images,
    }
    _write_json(folder / "field_map_manifest.json", manifest)
    workspace.record_artifact(f"field_map_{wavelength_nm:.3f}nm", folder / "field_map_manifest.json")
    return manifest


def run_broadband_stage(
    project_config: Mapping[str, Any],
    workspace: ExperimentWorkspace,
    *,
    cache_enabled: bool = True,
    stage_prefix: str = "frequency_scan",
    resolution: int | None = None,
    padding_um: float | None = None,
) -> tuple[SimulationResult | None, StageFields, StageFields, dict[str, Any]]:
    stage_name = f"{stage_prefix}/particle"
    simulation_config = make_simulation_config(
        project_config,
        experiment_directory=workspace.directory,
        stage_name=stage_name,
        reference=False,
        resolution=resolution,
        padding_um=padding_um,
    )
    reference = _run_reference(
        project_config,
        wavelength_um=None,
        cache_enabled=cache_enabled,
        resolution=resolution,
        padding_um=padding_um,
    )
    previous_stage = workspace.manifest().get("stages", {}).get(stage_prefix, {})
    previous_directory = workspace.directory / "stages" / stage_name
    if previous_stage.get("status") == "completed":
        print(f"Resume: reusing completed stage {stage_prefix}")
        actual = load_stage_fields(previous_directory)
        analysis = _save_broadband_results(project_config, actual, reference, workspace) if stage_prefix == "frequency_scan" else {}
        return None, actual, reference, analysis
    if previous_stage:
        simulation_config["run_id"] = f"{simulation_config['run_id']}__retry_{time_ns()}"
    print_stage_estimate("Broadband particle-line stage", simulation_config)
    workspace.record_stage(stage_prefix, "running", resolution=simulation_config["resolution"])
    result = run_simulation(simulation_config)
    actual = load_stage_fields(result.output_directory or workspace.directory / "stages" / stage_name)
    analysis = _save_broadband_results(project_config, actual, reference, workspace) if stage_prefix == "frequency_scan" else {}
    workspace.record_stage(stage_prefix, "completed", output_directory=result.output_directory, elapsed_seconds=result.elapsed_seconds)
    return result, actual, reference, analysis


def run_project(
    project_config: Mapping[str, Any],
    *,
    results_root: str | Path | None = None,
    resume: bool = False,
    cache_enabled: bool = True,
    field_map_selection: str | Sequence[float] | None = None,
) -> ExperimentWorkspace:
    config = normalise_project_config(project_config)
    workspace = ExperimentWorkspace.create(
        config["simulation_name"], config, results_root=results_root, resume=resume
    )
    _write_json(workspace.directory / "config_input.json", dict(project_config))
    resolved_path = _write_json(workspace.directory / "config_resolved.json", config)
    workspace.record_artifact("config_resolved", resolved_path)
    log_path = workspace.directory / "run.log"
    log_path.touch(exist_ok=True)
    workspace.record_artifact("run_log", log_path)
    start = perf_counter()
    try:
        _save_geometry(config, workspace)
        _, _, _, broadband = run_broadband_stage(
            config, workspace, cache_enabled=cache_enabled
        )
        selection = field_map_selection
        if selection is None:
            selection = config["measurement"]["field_map_frequencies"]
        wavelengths_nm = _select_wavelengths(
            broadband["response_rows"],
            selection,
            int(config["measurement"]["maximum_auto_field_maps"]),
        )
        selected_path = _write_csv(
            workspace.directory / "frequency_scan" / "selected_frequencies.csv",
            ("selection_index", "wavelength_nm"),
            [
                {"selection_index": index + 1, "wavelength_nm": value}
                for index, value in enumerate(wavelengths_nm)
            ],
        )
        workspace.record_artifact("selected_frequencies", selected_path)

        map_reports: list[dict[str, Any]] = []
        for index, wavelength_nm in enumerate(wavelengths_nm, start=1):
            wavelength_um = wavelength_nm / 1000.0
            stage = f"field_maps/{wavelength_nm:.3f}nm/particle"
            simulation_config = make_simulation_config(
                config,
                experiment_directory=workspace.directory,
                stage_name=stage,
                reference=False,
                wavelength_um=wavelength_um,
            )
            previous_stage = workspace.manifest().get("stages", {}).get(stage, {})
            existing_report = workspace.directory / "field_maps" / f"{wavelength_nm:.3f}nm" / "field_map_manifest.json"
            if previous_stage.get("status") == "completed" and existing_report.is_file():
                print(f"Resume: reusing completed field map at {wavelength_nm:.3f} nm")
                map_reports.append(json.loads(existing_report.read_text(encoding="utf-8")))
                continue
            if previous_stage:
                simulation_config["run_id"] = f"{simulation_config['run_id']}__retry_{time_ns()}"
            print_stage_estimate(f"Narrowband field map {index}/{len(wavelengths_nm)}", simulation_config)
            workspace.record_stage(stage, "running", wavelength_nm=wavelength_nm)
            reference = _run_reference(config, wavelength_um=wavelength_um, cache_enabled=cache_enabled)
            result = run_simulation(simulation_config)
            actual = load_stage_fields(result.output_directory or workspace.directory / "stages" / stage)
            report = _save_field_map(wavelength_nm, actual, reference, workspace, config)
            map_reports.append(report)
            workspace.record_stage(stage, "completed", elapsed_seconds=result.elapsed_seconds)

        active_positions = particle_line_positions(
            {
                **geometry_parameters(config, include_particles=True),
                "center": (0.0, 0.0, 0.0),
            }
        )
        summary_rows = [
            {
                "simulation_name": config["simulation_name"],
                "precision": config["precision"],
                "particle_count": config["geometry"]["particle_count"],
                "active_particle_count": len(active_positions),
                "particle_shape": config["geometry"]["particle_shape"],
                "particle_material": config["geometry"]["particle_material"],
                "substrate_material": config["geometry"]["substrate_material"],
                "background_material": config["geometry"]["background_material"],
                "spacing_um": config["geometry"]["spacing_um"],
                "defect_count": len(config["geometry"].get("defects", [])),
                "missing_particle_count": (
                    int(config["geometry"]["particle_count"]) - len(active_positions)
                ),
                "polarisation": config["source"]["polarisation"],
                "maximum_broadband_field_amplitude_enhancement": broadband[
                    "maximum_field_amplitude_enhancement"
                ],
                # Backwards-compatible alias; this value is now amplitude enhancement.
                "maximum_broadband_field_enhancement": broadband["maximum_field_enhancement"],
                "field_map_wavelengths_nm": json.dumps(wavelengths_nm),
            }
        ]
        summary_path = _write_csv(
            workspace.directory / "summary" / "research_results.csv",
            tuple(summary_rows[0]),
            summary_rows,
        )
        workspace.record_artifact("research_results", summary_path)
        _write_json(workspace.directory / "summary" / "field_map_reports.json", map_reports)
        workspace.update_manifest(elapsed_seconds=perf_counter() - start)
        workspace.set_status("completed", message="All requested stages completed")
        return workspace
    except KeyboardInterrupt:
        workspace.update_manifest(elapsed_seconds=perf_counter() - start)
        workspace.set_status("interrupted", message="Stopped by user; compatible stages remain on disk")
        raise
    except BaseException as exc:
        workspace.update_manifest(elapsed_seconds=perf_counter() - start)
        workspace.set_status("failed", message="Simulation failed", error=f"{type(exc).__name__}: {exc}")
        raise


__all__ = [
    "StageFields",
    "WorkflowError",
    "load_stage_fields",
    "print_stage_estimate",
    "run_broadband_stage",
    "run_project",
]
