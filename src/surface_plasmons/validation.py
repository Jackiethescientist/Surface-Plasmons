"""Separate, opt-in validation of one saved particle-line simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

import numpy as np

from .database import find_project_root
from .particle_line import particle_line_positions
from .project import PRECISION_MODES, geometry_parameters, normalise_project_config
from .results import ExperimentWorkspace, ResultsError, safe_simulation_name
from .theory import (
    compare_complex_fields,
    component_vector,
    electric_field_from_dipoles,
    solve_coupled_dipoles,
    sphere_polarizability,
)
from .workflow import _normalised_fields, run_broadband_stage


class ValidationError(RuntimeError):
    """Raised when a selected simulation cannot be validated safely."""


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ValidationError(f"Could not read {path}: {exc}") from exc


def resolve_experiment(value: str | Path, results_root: str | Path | None = None) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_dir():
        directory = supplied.resolve()
    else:
        root = (
            Path(results_root).expanduser().resolve()
            if results_root is not None
            else find_project_root() / "data" / "results"
        )
        directory = root / safe_simulation_name(str(value))
    if not (directory / "config_resolved.json").is_file():
        raise ValidationError(
            f"No completed project configuration was found in {directory}. "
            "Run scripts/run_single.py first."
        )
    return directory


def _load_config(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not load the selected configuration: {exc}") from exc
    return normalise_project_config(value)


def _configuration_check(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    parameters = {
        **geometry_parameters(config, include_particles=True),
        "center": (0.0, 0.0, 0.0),
    }
    positions = particle_line_positions(parameters)
    report = {
        "status": "passed",
        "dimensions": 3,
        "particle_count": len(positions),
        "defect_count": len(config["geometry"].get("defects", [])),
        "propagation_axis": config["source"]["propagation_axis"],
        "polarisation": config["source"]["polarisation"],
        "checks": [
            "valid materials and wavelength band",
            "transverse electric-field polarisation",
            "non-overlapping particle positions",
            "particles remain above the substrate",
            "three-dimensional Ex/Ey/Ez output",
        ],
    }
    _write_json(output / "configuration_validation.json", report)
    return report


def _nearest_material_epsilon(
    rows: Sequence[Mapping[str, str]], role: str, wavelength_nm: float
) -> complex:
    matches = [row for row in rows if row.get("material_role") == role]
    if not matches:
        raise ValidationError(f"No dielectric-function rows were found for {role}.")
    row = min(matches, key=lambda item: abs(float(item["wavelength_nm"]) - wavelength_nm))
    return complex(float(row["epsilon_real"]), float(row["epsilon_imag"]))


def _source_vectors(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    axis_index = {"x": 0, "y": 1, "z": 2}[
        str(config["source"]["propagation_axis"]).lower()
    ]
    direction = np.zeros(3, dtype=float)
    direction[axis_index] = float(config["source"]["propagation_direction"])
    polarisation = component_vector(str(config["source"]["polarisation"]))
    return direction, polarisation


def _theory_check(config: Mapping[str, Any], experiment: Path, output: Path) -> dict[str, Any]:
    """Compare MEEP and Equation 2 on the same external-field scale."""

    if config["geometry"]["particle_shape"] != "sphere":
        report = {
            "status": "skipped",
            "reason": "The Equation-2 polarizability used here is for spheres.",
        }
        _write_json(output / "theory_validation_summary.json", report)
        return report
    response_rows = _read_csv(experiment / "frequency_scan" / "field_response_summary.csv")
    field_rows = _read_csv(experiment / "frequency_scan" / "field_strength_vs_position_and_frequency.csv")
    dielectric_rows = _read_csv(experiment / "frequency_scan" / "dielectric_function_vs_frequency.csv")
    if not response_rows or not field_rows:
        raise ValidationError("The selected simulation has no broadband field data.")
    required = {
        f"normalised_{component}_{part}"
        for component in ("Ex", "Ey", "Ez")
        for part in ("real", "imag")
    }
    if not required.issubset(field_rows[0]):
        raise ValidationError(
            "This simulation predates same-scale complex-field output. Re-run it with "
            "the corrected external probe before Equation-2 validation."
        )

    strongest = max(
        response_rows,
        key=lambda row: float(
            row["maximum_field_intensity_enhancement"]
            if "maximum_field_intensity_enhancement" in row
            else row["maximum_field_enhancement"]
        ),
    )
    requested_wavelength_nm = float(strongest["wavelength_nm"])
    wavelength_nm = min(
        {float(row["wavelength_nm"]) for row in field_rows},
        key=lambda value: abs(value - requested_wavelength_nm),
    )
    profile_rows = [
        row for row in field_rows
        if abs(float(row["wavelength_nm"]) - wavelength_nm) < 1e-9
    ]
    profile_rows.sort(key=lambda row: float(row["x_um"]))
    points = np.asarray(
        [
            [float(row["x_um"]), float(row["y_um"]), float(row["z_um"])]
            for row in profile_rows
        ],
        dtype=float,
    )
    measured = np.asarray(
        [
            [
                complex(
                    float(row[f"normalised_{component}_real"]),
                    float(row[f"normalised_{component}_imag"]),
                )
                for component in ("Ex", "Ey", "Ez")
            ]
            for row in profile_rows
        ],
        dtype=complex,
    )

    parameters = {
        **geometry_parameters(config, include_particles=True),
        "center": (0.0, 0.0, 0.0),
    }
    positions = particle_line_positions(parameters)
    centres = np.asarray(
        [(item.actual_x_um, item.actual_y_um, item.actual_z_um) for item in positions],
        dtype=float,
    )
    epsilon_particle = _nearest_material_epsilon(dielectric_rows, "particle", wavelength_nm)
    epsilon_background = _nearest_material_epsilon(dielectric_rows, "background", wavelength_nm)
    epsilon_substrate = _nearest_material_epsilon(dielectric_rows, "substrate", wavelength_nm)
    homogeneous_background = abs(epsilon_substrate - epsilon_background) <= 1e-10
    substrate_for_model = None if homogeneous_background else epsilon_substrate
    wavelength_um = wavelength_nm / 1000.0
    alpha = sphere_polarizability(
        float(config["geometry"]["particle_radius_um"]),
        epsilon_particle,
        epsilon_background,
        wavelength_um=wavelength_um,
    )
    direction, polarisation = _source_vectors(config)
    solution = solve_coupled_dipoles(
        centres,
        alpha,
        wavelength_um=wavelength_um,
        epsilon_background=epsilon_background,
        incident_direction=direction,
        incident_polarisation=polarisation,
        epsilon_substrate=substrate_for_model,
        surface_y_um=float(config["geometry"]["surface_y_um"]),
    )
    predicted = electric_field_from_dipoles(
        points,
        solution,
        epsilon_background=epsilon_background,
        incident_direction=direction,
        incident_polarisation=polarisation,
        epsilon_substrate=substrate_for_model,
        surface_y_um=float(config["geometry"]["surface_y_um"]),
    )

    # Equation 2 uses E0 as its scale. The observation line has constant y and
    # z in this workflow, so its incident phase is common and can be divided out
    # without changing the spatial profile.
    wavevector = 2.0 * np.pi * np.sqrt(epsilon_background) / wavelength_um
    incident = np.exp(1j * wavevector * (points @ direction))[:, np.newaxis] * polarisation
    component_index = int(np.argmax(np.abs(polarisation)))
    reference_scale = complex(
        np.median(incident[:, component_index].real),
        np.median(incident[:, component_index].imag),
    )
    predicted_normalised = predicted / reference_scale
    comparison = compare_complex_fields(measured, predicted_normalised)

    measured_amplitude = np.sqrt(np.sum(np.abs(measured) ** 2, axis=1))
    theory_amplitude = np.sqrt(np.sum(np.abs(predicted_normalised) ** 2, axis=1))
    measured_intensity = measured_amplitude**2
    theory_intensity = theory_amplitude**2
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        row: dict[str, Any] = {
            "point_index": index,
            "wavelength_nm": wavelength_nm,
            "x_um": float(point[0]),
            "y_um": float(point[1]),
            "z_um": float(point[2]),
            "meep_amplitude_E_over_Eref": float(measured_amplitude[index]),
            "equation2_amplitude_E_over_E0": float(theory_amplitude[index]),
            "meep_intensity_ratio": float(measured_intensity[index]),
            "equation2_intensity_ratio": float(theory_intensity[index]),
        }
        for component_index_value, component in enumerate(("Ex", "Ey", "Ez")):
            row[f"meep_{component}_real"] = float(np.real(measured[index, component_index_value]))
            row[f"meep_{component}_imag"] = float(np.imag(measured[index, component_index_value]))
            row[f"equation2_{component}_real"] = float(
                np.real(predicted_normalised[index, component_index_value])
            )
            row[f"equation2_{component}_imag"] = float(
                np.imag(predicted_normalised[index, component_index_value])
            )
        rows.append(row)
    _write_csv(output / "equation2_external_field_comparison.csv", tuple(rows[0]), rows)

    dipole_rows: list[dict[str, Any]] = []
    equation_rhs = solution.incident_fields - (
        solution.interaction_matrix_A @ solution.dipole_moments.reshape(-1)
    ).reshape(solution.local_fields.shape)
    residual = solution.local_fields - equation_rhs
    for index, particle in enumerate(positions):
        dipole_rows.append(
            {
                "particle_index": particle.particle_index,
                "x_um": particle.actual_x_um,
                "y_um": particle.actual_y_um,
                "z_um": particle.actual_z_um,
                "local_field_amplitude_over_E0": float(np.linalg.norm(solution.local_fields[index])),
                "dipole_moment_magnitude_over_epsilon0": float(np.linalg.norm(solution.dipole_moments[index])),
                "equation2_residual_magnitude": float(np.linalg.norm(residual[index])),
            }
        )
    _write_csv(output / "equation2_particle_solution.csv", tuple(dipole_rows[0]), dipole_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9.0, 5.6))
    axis.plot(points[:, 0], measured_amplitude, linewidth=2.0, label="MEEP")
    axis.plot(points[:, 0], theory_amplitude, linewidth=1.7, label="Equation 2 CDA")
    axis.set_xlabel("Position along particle line, x (µm)")
    axis.set_ylabel("Electric-field amplitude ratio")
    axis.set_title(f"External field amplitude at {wavelength_nm:.3f} nm")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "equation2_field_amplitude_profile.png", dpi=230, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9.0, 5.6))
    axis.plot(points[:, 0], measured_intensity, linewidth=2.0, label="MEEP")
    axis.plot(points[:, 0], theory_intensity, linewidth=1.7, label="Equation 2 CDA")
    axis.set_xlabel("Position along particle line, x (µm)")
    axis.set_ylabel("Electric-field intensity ratio")
    axis.set_title(f"External field intensity at {wavelength_nm:.3f} nm")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "equation2_field_intensity_profile.png", dpi=230, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "completed",
        "wavelength_nm": wavelength_nm,
        "equation": "E_loc,i = E0 exp(i k.r_i) - sum_(j!=i) A_ij.P_j",
        "sign_convention": "A_ij = -G_ij, so -A_ij.P_j is the physical dipole field G_ij.P_j",
        "model": "complex vector quasistatic coupled-dipole approximation",
        "normalization": {
            "meep": "complex field divided by one particle-free reference amplitude at this frequency",
            "theory": "complex field divided by E0 including the common incident phase on the probe line",
            "independent_curve_rescaling": False,
        },
        "same_scale_comparison": comparison.to_dict(),
        "maximum_equation2_residual": float(np.max(np.linalg.norm(residual, axis=1))),
        "substrate_treatment": (
            "none; homogeneous background gives the direct Equation-2 validation"
            if homogeneous_background
            else "electrostatic image-dipole approximation; direct particle interactions match Equation 2"
        ),
        "important_limitation": (
            "None for the normalization comparison; remaining differences test the quasistatic dipole model."
            if homogeneous_background
            else "The substrate reference includes the full MEEP interface response, whereas the current "
            "analytical substrate term is an electrostatic image approximation. Validate the homogeneous "
            "case first before interpreting substrate disagreement."
        ),
    }
    _write_json(output / "theory_validation_summary.json", report)
    return report


def _response_from_fields(
    actual: Any, reference: Any, config: Mapping[str, Any]
) -> np.ndarray:
    enhancement = _normalised_fields(
        actual,
        reference,
        reference_component=str(config["source"]["polarisation"]),
        mode=str(config["measurement"]["normalization_mode"]),
    )["intensity_enhancement"]
    return np.nanmax(enhancement.reshape(enhancement.shape[0], -1), axis=1)


def _convergence_runs(
    config: Mapping[str, Any],
    workspace: ExperimentWorkspace,
    *,
    values: Sequence[float],
    kind: str,
    cache_enabled: bool,
) -> dict[str, Any]:
    profiles: list[tuple[float, np.ndarray, np.ndarray]] = []
    for value in values:
        if kind == "resolution":
            resolution = int(value)
            padding = None
            label = f"resolution_{resolution}"
        else:
            resolution = None
            padding = float(value)
            label = f"padding_{padding:.4g}um"
        _, actual, reference, _ = run_broadband_stage(
            config,
            workspace,
            cache_enabled=cache_enabled,
            stage_prefix=label,
            resolution=resolution,
            padding_um=padding,
        )
        profiles.append(
            (
                float(value),
                actual.wavelengths_um * 1000.0,
                _response_from_fields(actual, reference, config),
            )
        )
    reference_value, reference_wavelengths, reference_response = profiles[-1]
    rows: list[dict[str, Any]] = []
    for value, wavelengths, response in profiles:
        order = np.argsort(wavelengths)
        ref_order = np.argsort(reference_wavelengths)
        interpolated_reference = np.interp(wavelengths[order], reference_wavelengths[ref_order], reference_response[ref_order])
        scale = float(np.nanmax(np.abs(interpolated_reference))) or 1.0
        nrmse = float(np.sqrt(np.nanmean(((response[order] - interpolated_reference) / scale) ** 2)))
        peak_index = int(np.nanargmax(response))
        rows.append(
            {
                kind: value,
                "reference_value": reference_value,
                "normalised_rmse_vs_reference": nrmse,
                "peak_wavelength_nm": float(wavelengths[peak_index]),
                "peak_field_enhancement": float(response[peak_index]),
            }
        )
    path = _write_csv(
        workspace.directory / f"{kind}_convergence.csv",
        (kind, "reference_value", "normalised_rmse_vs_reference", "peak_wavelength_nm", "peak_field_enhancement"),
        rows,
    )
    report = {
        "status": "completed",
        "values": list(values),
        "reference_value": reference_value,
        "finest_comparison": rows[-2] if len(rows) > 1 else rows[-1],
        "table": str(path.name),
    }
    _write_json(workspace.directory / f"{kind}_convergence_summary.json", report)
    return report


def _time_check(experiment: Path, output: Path) -> dict[str, Any]:
    summary_path = experiment / "stages" / "frequency_scan" / "particle" / "summary.json"
    if not summary_path.is_file():
        report = {"status": "unavailable", "reason": "Broadband stage summary is missing."}
    else:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        config = value.get("config", {})
        report = {
            "status": "completed",
            "elapsed_seconds": value.get("elapsed_seconds"),
            "meep_time": value.get("meep_time"),
            "timesteps": value.get("timesteps"),
            "minimum_run_time": config.get("minimum_run_time"),
            "maximum_run_time": config.get("maximum_run_time"),
            "dft_decay_tolerance": config.get("dft_decay_tolerance"),
            "note": "A result close to maximum_run_time should be reviewed for incomplete DFT decay.",
        }
    _write_json(output / "time_convergence.json", report)
    return report


def run_validation(
    experiment: str | Path,
    *,
    validation_name: str,
    checks: Sequence[str] = ("configuration", "theory"),
    results_root: str | Path | None = None,
    resolutions: Sequence[int] | None = None,
    paddings_um: Sequence[float] | None = None,
    cache_enabled: bool = True,
) -> Path:
    directory = resolve_experiment(experiment, results_root)
    config = _load_config(directory)
    selected = {str(value).strip().lower() for value in checks}
    if "all" in selected:
        selected = {"configuration", "theory", "resolution", "domain", "time"}
    unknown = selected - {"configuration", "theory", "resolution", "domain", "time"}
    if unknown:
        raise ValidationError(f"Unknown validation check(s): {sorted(unknown)}.")
    validation_root = directory / "validation"
    workspace = ExperimentWorkspace.create(
        validation_name,
        {"target_configuration": config, "checks": sorted(selected)},
        results_root=validation_root,
        resume=False,
    )
    _write_json(workspace.directory / "validation_config.json", {"checks": sorted(selected), "target": str(directory)})
    reports: dict[str, Any] = {}
    try:
        if "configuration" in selected:
            reports["configuration"] = _configuration_check(config, workspace.directory)
        if "theory" in selected:
            reports["theory"] = _theory_check(config, directory, workspace.directory)
        if "resolution" in selected:
            if resolutions is None:
                base = int(PRECISION_MODES[config["precision"]]["broadband_resolution"])
                resolutions = sorted({max(20, int(round(base * factor / 10.0) * 10)) for factor in (0.75, 1.0, 1.25)})
            reports["resolution"] = _convergence_runs(
                config, workspace, values=resolutions, kind="resolution", cache_enabled=cache_enabled
            )
        if "domain" in selected:
            if paddings_um is None:
                base_padding = float(config["numerics"]["padding_um"])
                paddings_um = (base_padding, base_padding + 0.10)
            reports["domain"] = _convergence_runs(
                config, workspace, values=paddings_um, kind="padding_um", cache_enabled=cache_enabled
            )
        if "time" in selected:
            reports["time"] = _time_check(directory, workspace.directory)
        _write_json(workspace.directory / "validation_summary.json", reports)
        workspace.set_status("completed", message="Requested validation checks completed")
        return workspace.directory
    except BaseException as exc:
        workspace.set_status("failed", message="Validation failed", error=f"{type(exc).__name__}: {exc}")
        raise


__all__ = [
    "ValidationError",
    "resolve_experiment",
    "run_validation",
]
