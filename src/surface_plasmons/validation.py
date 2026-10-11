"""Separate, opt-in validation of one saved particle-line simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

import numpy as np

from .database import find_project_root
from .particle_line import missing_particle_indices, particle_line_positions
from .project import PRECISION_MODES, geometry_parameters, normalise_project_config
from .results import ExperimentWorkspace, ResultsError, safe_simulation_name
from .theory import (
    compare_complex_fields,
    component_vector,
    electric_field_from_dipoles,
    mie_sphere_cross_sections,
    solve_coupled_dipoles,
    solve_one_dimensional_chain,
    sphere_polarizability,
    one_dimensional_dipole_sum,
    uniform_chain_dipole_moment,
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
    """Compare MEEP with the paper's Section-2 CDA on one field scale.

    This validation deliberately uses the paper's simplest internally named
    combination: Equation-1 quasistatic sphere polarizability plus the
    retarded Equation-3 interaction inside Equation 2. Radiation-reaction and
    other Section-2.2 finite-size corrections are not silently mixed in.
    """

    benchmark_requirements = {
        "particle_shape": "sphere",
        "optical_environment": (
            "homogeneous: geometry.background_material must equal "
            "geometry.substrate_material"
        ),
        "substrate_interface": "absent",
        "interaction_model": "paper_retarded",
        "polarizability_model": "quasistatic",
    }
    if config["geometry"]["particle_shape"] != "sphere":
        report = {
            "status": "skipped",
            "benchmark": "formal_equation2_homogeneous",
            "benchmark_status": "failed_precondition",
            "benchmark_requirements": benchmark_requirements,
            "reason": "The Equation-2 polarizability used here is for spheres.",
        }
        _write_json(output / "theory_validation_summary.json", report)
        return report
    background_name = str(config["geometry"]["background_material"]).strip().lower()
    substrate_name = str(config["geometry"]["substrate_material"]).strip().lower()
    if background_name != substrate_name:
        report = {
            "status": "skipped",
            "benchmark": "formal_equation2_homogeneous",
            "benchmark_status": "failed_precondition",
            "benchmark_requirements": benchmark_requirements,
            "reason": (
                "The formal Equation-2 benchmark requires an optically "
                "homogeneous environment with no substrate interface. Set "
                "geometry.substrate_material equal to geometry.background_material "
                "and run the separately named homogeneous benchmark first."
            ),
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
            row["maximum_field_amplitude_enhancement"]
            if "maximum_field_amplitude_enhancement" in row
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
    if not homogeneous_background:
        report = {
            "status": "skipped",
            "benchmark": "formal_equation2_homogeneous",
            "benchmark_status": "failed_precondition",
            "benchmark_requirements": benchmark_requirements,
            "reason": (
                "The configured background and substrate names match, but their "
                "stored dielectric functions differ at the selected wavelength. "
                "The formal no-interface benchmark cannot proceed."
            ),
            "wavelength_nm": wavelength_nm,
            "epsilon_background": str(epsilon_background),
            "epsilon_substrate": str(epsilon_substrate),
        }
        _write_json(output / "theory_validation_summary.json", report)
        return report
    wavelength_um = wavelength_nm / 1000.0
    alpha = sphere_polarizability(
        float(config["geometry"]["particle_radius_um"]),
        epsilon_particle,
        epsilon_background,
        wavelength_um=wavelength_um,
        model="quasistatic",
    )
    direction, polarisation = _source_vectors(config)
    solution = solve_coupled_dipoles(
        centres,
        alpha,
        wavelength_um=wavelength_um,
        epsilon_background=epsilon_background,
        incident_direction=direction,
        incident_polarisation=polarisation,
        interaction_model="paper_retarded",
    )
    predicted = electric_field_from_dipoles(
        points,
        solution,
        epsilon_background=epsilon_background,
        incident_direction=direction,
        incident_polarisation=polarisation,
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
    axis.plot(
        points[:, 0],
        theory_amplitude,
        linewidth=1.7,
        label="Equations 2-3 CDA",
    )
    axis.set_xlabel("Position along particle line, x (µm)")
    axis.set_ylabel("Electric-field amplitude ratio")
    axis.set_title(f"External field amplitude at {wavelength_nm:.3f} nm")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "equation2_field_amplitude_profile.png", dpi=230, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "completed",
        "benchmark": "formal_equation2_homogeneous",
        "benchmark_status": "passed_preconditions",
        "benchmark_requirements": benchmark_requirements,
        "wavelength_nm": wavelength_nm,
        "equation": "E_loc,i = E0 exp(i k.r_i) - sum_(j!=i) A_ij.P_j",
        "sign_convention": "A_ij = -G_ij, so -A_ij.P_j is the physical dipole field G_ij.P_j",
        "model": "paper Section-2 complex vector coupled-dipole approximation",
        "paper_reference": {
            "article": "Kravets et al., Chemical Reviews 2018, DOI 10.1021/acs.chemrev.8b00243",
            "local_field": "Equation 2",
            "interaction_tensor": "Equation 3 (retarded)",
            "sphere_polarizability": "Equation 1, generalized to a homogeneous host",
        },
        "interaction_model": "paper_retarded",
        "polarizability_model": "quasistatic",
        "finite_size_corrections": "disabled for the strict Section-2.1 baseline",
        "time_convention": (
            "suppressed exp(-i omega t), inferred from the paper's outgoing exp(+i k r) factor"
        ),
        "units": (
            "lengths in um; dipole moment stored as p_SI/epsilon_0; alpha in um^3; "
            "host-medium 1/(4*pi*epsilon_background) factor included in A"
        ),
        "normalization": {
            "meep": "complex field divided by one particle-free reference amplitude at this frequency",
            "theory": "complex field divided by E0 including the common incident phase on the probe line",
            "independent_curve_rescaling": False,
        },
        "same_scale_comparison": comparison.to_dict(),
        "maximum_equation2_residual": float(np.max(np.linalg.norm(residual, axis=1))),
        "substrate_treatment": (
            "none; background and nominal substrate use the same material, so no optical interface exists"
        ),
        "important_limitation": (
            "The particle polarizability is the quasistatic Equation-1 value; radiation damping, "
            "dynamic depolarization and higher multipoles are intentionally excluded."
        ),
    }
    _write_json(output / "theory_validation_summary.json", report)
    return report


def _same_scale_spectrum_metrics(
    reference: Sequence[float],
    prediction: Sequence[float],
) -> dict[str, float | int | None]:
    measured = np.asarray(reference, dtype=float)
    modelled = np.asarray(prediction, dtype=float)
    scale = float(np.sqrt(np.mean(measured**2))) or 1.0
    nrmse = float(np.sqrt(np.mean((modelled - measured) ** 2)) / scale)
    correlation = (
        float(np.corrcoef(measured, modelled)[0, 1])
        if measured.size > 1 and np.std(measured) > 0 and np.std(modelled) > 0
        else None
    )
    return {
        "same_scale_nrmse": nrmse,
        "correlation": correlation,
        "meep_peak_index": int(np.argmax(measured)),
        "model_peak_index": int(np.argmax(modelled)),
    }


def _coordinate_edges(values: np.ndarray, minimum_half_width: float) -> np.ndarray:
    if values.size == 1:
        return np.asarray(
            [values[0] - minimum_half_width, values[0] + minimum_half_width],
            dtype=float,
        )
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(
        ([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])])
    )


def _hierarchy_check(config: Mapping[str, Any], experiment: Path, output: Path) -> dict[str, Any]:
    """Compare 1-D CDA, vector CDA, Mie corrections, and 3-D MEEP.

    Cross-sections and near-field amplitudes are deliberately drawn in
    separate panels because they are different observables. Exact Mie theory
    is used only for the single-sphere response; the array model labelled
    ``Mie-dipole CDA`` uses Mie's electric ``a1`` coefficient as the particle
    polarizability inside Equation 2.
    """

    requirements = {
        "particle_shape": "sphere",
        "environment": "homogeneous and lossless background",
        "geometry": "active particle centres collinear along x",
        "saved_data": "same-scale complex Ex/Ey/Ez broadband fields",
    }
    if config["geometry"]["particle_shape"] != "sphere":
        report = {
            "status": "skipped",
            "reason": "The Mie and Equation-1 hierarchy implemented here requires spheres.",
            "requirements": requirements,
        }
        _write_json(output / "model_hierarchy_summary.json", report)
        return report
    background_name = str(config["geometry"]["background_material"]).strip().lower()
    substrate_name = str(config["geometry"]["substrate_material"]).strip().lower()
    if background_name != substrate_name:
        report = {
            "status": "skipped",
            "reason": (
                "The staged CDA-Mie benchmark requires a homogeneous optical environment. "
                "Run the separately named homogeneous perfect or defect configuration first."
            ),
            "requirements": requirements,
        }
        _write_json(output / "model_hierarchy_summary.json", report)
        return report

    field_rows = _read_csv(
        experiment / "frequency_scan" / "field_strength_vs_position_and_frequency.csv"
    )
    dielectric_rows = _read_csv(
        experiment / "frequency_scan" / "dielectric_function_vs_frequency.csv"
    )
    if not field_rows:
        raise ValidationError("The selected simulation has no broadband field data.")
    required = {
        f"normalised_{component}_{part}"
        for component in ("Ex", "Ey", "Ez")
        for part in ("real", "imag")
    }
    if not required.issubset(field_rows[0]):
        raise ValidationError(
            "This simulation predates same-scale complex-field output. Re-run it before "
            "using the model hierarchy validation."
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
    if np.ptp(centres[:, 1]) > 1e-12 or np.ptp(centres[:, 2]) > 1e-12:
        report = {
            "status": "skipped",
            "reason": "The direct 1-D formulas require all active centres to remain on one x-directed line.",
            "requirements": requirements,
        }
        _write_json(output / "model_hierarchy_summary.json", report)
        return report

    direction, polarisation = _source_vectors(config)
    component_index = int(np.argmax(np.abs(polarisation)))
    orientation = "longitudinal" if component_index == 0 else "transverse"
    wavelengths_nm = sorted({float(row["wavelength_nm"]) for row in field_rows})
    radius_um = float(config["geometry"]["particle_radius_um"])
    defects = list(config["geometry"].get("defects", []))
    missing_indices = missing_particle_indices(
        defects,
        int(config["geometry"]["particle_count"]),
    )
    shifted_indices = sorted(
        int(item["particle_index"])
        for item in defects
        if not bool(item.get("missing", False))
    )
    is_perfect = not defects

    spectrum_rows: list[dict[str, Any]] = []
    particle_rows: list[dict[str, Any]] = []
    local_scalar_by_wavelength: list[np.ndarray] = []
    scalar_vector_relative_errors: list[float] = []
    for wavelength_nm in wavelengths_nm:
        wavelength_um = wavelength_nm / 1000.0
        profile_rows = [
            row
            for row in field_rows
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
        epsilon_particle = _nearest_material_epsilon(
            dielectric_rows, "particle", wavelength_nm
        )
        epsilon_background = _nearest_material_epsilon(
            dielectric_rows, "background", wavelength_nm
        )
        epsilon_substrate = _nearest_material_epsilon(
            dielectric_rows, "substrate", wavelength_nm
        )
        if abs(epsilon_substrate - epsilon_background) > 1e-10:
            report = {
                "status": "skipped",
                "reason": (
                    "The stored substrate and background dielectric functions differ, so the "
                    "homogeneous CDA-Mie hierarchy is not applicable."
                ),
                "requirements": requirements,
                "wavelength_nm": wavelength_nm,
            }
            _write_json(output / "model_hierarchy_summary.json", report)
            return report
        if abs(epsilon_background.imag) > 1e-12 or epsilon_background.real <= 0:
            report = {
                "status": "skipped",
                "reason": "Exact Mie cross-sections require a positive, lossless background.",
                "requirements": requirements,
                "wavelength_nm": wavelength_nm,
            }
            _write_json(output / "model_hierarchy_summary.json", report)
            return report

        alpha_quasistatic = sphere_polarizability(
            radius_um,
            epsilon_particle,
            epsilon_background,
            wavelength_um=wavelength_um,
            model="quasistatic",
        )
        alpha_mie = sphere_polarizability(
            radius_um,
            epsilon_particle,
            epsilon_background,
            wavelength_um=wavelength_um,
            model="mie_electric_dipole",
        )
        wavevector = 2.0 * np.pi * np.sqrt(epsilon_background) / wavelength_um
        common_incident_phase = np.exp(
            1j
            * wavevector
            * (
                centres[0, 1] * float(direction[1])
                + centres[0, 2] * float(direction[2])
            )
        )
        scalar = solve_one_dimensional_chain(
            centres[:, 0],
            alpha_quasistatic,
            wavelength_um=wavelength_um,
            epsilon_background=epsilon_background,
            orientation=orientation,
            incident_direction_cosine=float(direction[0]),
            incident_amplitude=common_incident_phase,
            interaction_model="paper_retarded",
        )
        vector_quasistatic = solve_coupled_dipoles(
            centres,
            alpha_quasistatic,
            wavelength_um=wavelength_um,
            epsilon_background=epsilon_background,
            incident_direction=direction,
            incident_polarisation=polarisation,
            interaction_model="quasistatic",
        )
        vector_retarded = solve_coupled_dipoles(
            centres,
            alpha_quasistatic,
            wavelength_um=wavelength_um,
            epsilon_background=epsilon_background,
            incident_direction=direction,
            incident_polarisation=polarisation,
            interaction_model="paper_retarded",
        )
        vector_mie = solve_coupled_dipoles(
            centres,
            alpha_mie,
            wavelength_um=wavelength_um,
            epsilon_background=epsilon_background,
            incident_direction=direction,
            incident_polarisation=polarisation,
            interaction_model="paper_retarded",
        )
        field_quasistatic = electric_field_from_dipoles(
            points,
            vector_quasistatic,
            epsilon_background=epsilon_background,
            incident_direction=direction,
            incident_polarisation=polarisation,
        )
        field_retarded = electric_field_from_dipoles(
            points,
            vector_retarded,
            epsilon_background=epsilon_background,
            incident_direction=direction,
            incident_polarisation=polarisation,
        )
        field_mie = electric_field_from_dipoles(
            points,
            vector_mie,
            epsilon_background=epsilon_background,
            incident_direction=direction,
            incident_polarisation=polarisation,
        )
        vector_local_component = vector_retarded.local_fields[:, component_index]
        consistency_scale = float(np.linalg.norm(vector_local_component)) or 1.0
        scalar_vector_error = float(
            np.linalg.norm(scalar.local_fields - vector_local_component)
            / consistency_scale
        )
        scalar_vector_relative_errors.append(scalar_vector_error)
        local_scalar_by_wavelength.append(np.abs(scalar.local_fields))
        for particle, local_field, dipole in zip(
            positions, scalar.local_fields, scalar.dipole_moments
        ):
            particle_rows.append(
                {
                    "wavelength_nm": wavelength_nm,
                    "particle_index": particle.particle_index,
                    "actual_x_um": particle.actual_x_um,
                    "defect_kind": (
                        "shifted" if particle.particle_index in shifted_indices else "none"
                    ),
                    "local_field_real_over_E0": float(local_field.real),
                    "local_field_imag_over_E0": float(local_field.imag),
                    "local_field_amplitude_over_E0": float(abs(local_field)),
                    "dipole_real_over_epsilon0_E0_um3": float(dipole.real),
                    "dipole_imag_over_epsilon0_E0_um3": float(dipole.imag),
                    "dipole_amplitude_over_epsilon0_E0_um3": float(abs(dipole)),
                }
            )

        mie = mie_sphere_cross_sections(
            radius_um,
            wavelength_um,
            epsilon_particle,
            epsilon_background,
        )
        equation4_magnitude: float | str = ""
        if is_perfect and len(positions) >= 3:
            neighbours = max(1, (len(positions) - 1) // 2)
            dipole_sum = one_dimensional_dipole_sum(
                float(config["geometry"]["spacing_um"]),
                wavelength_um=wavelength_um,
                neighbour_count=neighbours,
                epsilon_background=epsilon_background,
                orientation=orientation,
                interaction_model="paper_retarded",
            )
            equation4_magnitude = float(
                abs(uniform_chain_dipole_moment(alpha_quasistatic, dipole_sum))
            )
        spectrum_rows.append(
            {
                "wavelength_nm": wavelength_nm,
                "meep_max_external_E_over_Eref": float(
                    np.max(np.linalg.norm(measured, axis=1))
                ),
                "cda_quasistatic_max_external_E_over_E0": float(
                    np.max(np.linalg.norm(field_quasistatic, axis=1))
                ),
                "cda_equations2_3_max_external_E_over_E0": float(
                    np.max(np.linalg.norm(field_retarded, axis=1))
                ),
                "mie_dipole_cda_max_external_E_over_E0": float(
                    np.max(np.linalg.norm(field_mie, axis=1))
                ),
                "scalar_1d_max_local_E_over_E0": float(
                    np.max(np.abs(scalar.local_fields))
                ),
                "vector_cda_max_local_E_over_E0": float(
                    np.max(np.abs(vector_local_component))
                ),
                "scalar_vs_vector_relative_error": scalar_vector_error,
                "equation4_uniform_dipole_magnitude_over_epsilon0_E0_um3": (
                    equation4_magnitude
                ),
                "mie_exact_extinction_cross_section_um2": mie.extinction_cross_section_um2,
                "mie_exact_scattering_cross_section_um2": mie.scattering_cross_section_um2,
                "mie_exact_absorption_cross_section_um2": mie.absorption_cross_section_um2,
                "mie_maximum_multipole_order": mie.maximum_order,
            }
        )

    _write_csv(
        output / "model_hierarchy_spectrum.csv",
        tuple(spectrum_rows[0]),
        spectrum_rows,
    )
    _write_csv(
        output / "one_dimensional_particle_response.csv",
        tuple(particle_rows[0]),
        particle_rows,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    meep_external = np.asarray(
        [row["meep_max_external_E_over_Eref"] for row in spectrum_rows], dtype=float
    )
    quasistatic_external = np.asarray(
        [row["cda_quasistatic_max_external_E_over_E0"] for row in spectrum_rows], dtype=float
    )
    retarded_external = np.asarray(
        [row["cda_equations2_3_max_external_E_over_E0"] for row in spectrum_rows], dtype=float
    )
    mie_dipole_external = np.asarray(
        [row["mie_dipole_cda_max_external_E_over_E0"] for row in spectrum_rows], dtype=float
    )
    scalar_local = np.asarray(
        [row["scalar_1d_max_local_E_over_E0"] for row in spectrum_rows], dtype=float
    )
    vector_local = np.asarray(
        [row["vector_cda_max_local_E_over_E0"] for row in spectrum_rows], dtype=float
    )
    figure, axes = plt.subplots(3, 1, figsize=(9.5, 13.0), sharex=False)
    axes[0].plot(wavelengths, meep_external, label="3-D MEEP FDTD", linewidth=2.0)
    axes[0].plot(wavelengths, quasistatic_external, label="Quasistatic CDA", linewidth=1.4)
    axes[0].plot(wavelengths, retarded_external, label="Equations 2-3 CDA", linewidth=1.5)
    axes[0].plot(wavelengths, mie_dipole_external, label="Mie-dipole CDA", linewidth=1.5)
    axes[0].set_xlabel("Vacuum wavelength (nm)")
    axes[0].set_ylabel(
        "Maximum external field-amplitude ratio\n"
        r"MEEP: $|E|/|E_{ref}|$; theory: $|E|/|E_0|$"
    )
    axes[0].set_title("Same-observable near-field model hierarchy")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].plot(wavelengths, scalar_local, label="Direct scalar 1-D CDA", linewidth=1.8)
    axes[1].plot(wavelengths, vector_local, "--", label="3-D vector CDA component", linewidth=1.5)
    axes[1].set_xlabel("Vacuum wavelength (nm)")
    axes[1].set_ylabel(r"Maximum local field-amplitude ratio, $|E_{loc}|/|E_0|$")
    axes[1].set_title("Direct 1-D formula check against the vector Equation-2 solver")
    axes[1].grid(alpha=0.22)
    axes[1].legend(frameon=False)

    axes[2].plot(
        wavelengths,
        [row["mie_exact_extinction_cross_section_um2"] for row in spectrum_rows],
        label="Extinction",
    )
    axes[2].plot(
        wavelengths,
        [row["mie_exact_scattering_cross_section_um2"] for row in spectrum_rows],
        label="Scattering",
    )
    axes[2].plot(
        wavelengths,
        [row["mie_exact_absorption_cross_section_um2"] for row in spectrum_rows],
        label="Absorption",
    )
    axes[2].set_xlabel("Vacuum wavelength (nm)")
    axes[2].set_ylabel(r"Single-sphere cross-section ($\mu m^2$)")
    axes[2].set_title("Exact single-sphere Mie theory; not the array near-field observable")
    axes[2].grid(alpha=0.22)
    axes[2].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "model_hierarchy_comparison.png", dpi=230, bbox_inches="tight")
    plt.close(figure)

    nominal_count = int(config["geometry"]["particle_count"])
    heatmap = np.full((len(wavelengths), nominal_count), np.nan, dtype=float)
    for wavelength_index, amplitudes in enumerate(local_scalar_by_wavelength):
        for particle, amplitude in zip(positions, amplitudes):
            heatmap[wavelength_index, particle.particle_index] = float(amplitude)
    figure, axis = plt.subplots(figsize=(9.2, 5.8))
    colour_map = plt.get_cmap("magma").copy()
    colour_map.set_bad("lightgray")
    image = axis.pcolormesh(
        np.arange(nominal_count + 1, dtype=float) - 0.5,
        _coordinate_edges(wavelengths, 0.5),
        heatmap,
        shading="flat",
        cmap=colour_map,
    )
    axis.set_xticks(np.arange(nominal_count))
    axis.set_xlabel("Nominal particle index along the x-directed line")
    axis.set_ylabel("Vacuum wavelength (nm)")
    axis.set_title("Direct 1-D CDA local-field response; gray columns are missing particles")
    figure.colorbar(
        image,
        ax=axis,
        label=r"Local field-amplitude ratio, $|E_{loc}|/|E_0|$",
    )
    figure.tight_layout()
    figure.savefig(output / "one_dimensional_defect_response.png", dpi=230, bbox_inches="tight")
    plt.close(figure)

    report = {
        "status": "completed",
        "requirements": requirements,
        "orientation": orientation,
        "active_particle_count": len(positions),
        "nominal_particle_count": int(config["geometry"]["particle_count"]),
        "shifted_particle_indices": shifted_indices,
        "missing_particle_indices": list(missing_indices),
        "maximum_scalar_vs_vector_relative_error": max(scalar_vector_relative_errors),
        "model_order": [
            "direct scalar 1-D CDA",
            "3-D vector quasistatic CDA",
            "3-D vector retarded Equations 2-3 CDA",
            "Mie electric-dipole polarizability inside retarded CDA",
            "exact multipole Mie theory for one sphere",
            "3-D MEEP FDTD for the complete finite structure",
        ],
        "same_observable_external_field_comparisons": {
            "quasistatic_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_external, quasistatic_external
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_external, retarded_external
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_external, mie_dipole_external
            ),
        },
        "equation4_dipole_sum": (
            "reported for the undefected equal-spacing chain"
            if is_perfect and len(positions) >= 3
            else "not applied because Equation 4 assumes an identical periodic chain"
        ),
        "plot_scale_note": (
            "Every axis and colorbar is labelled. Near-field amplitude ratios and Mie "
            "cross-sections are separated because they are different physical observables."
        ),
        "limitations": [
            "Exact Mie multipoles are a single-sphere benchmark, not a multiple-sphere solution.",
            "Mie-dipole CDA retains only the electric a1 response for each sphere.",
            "MEEP is the highest-level full-wave array calculation in this project.",
        ],
    }
    _write_json(output / "model_hierarchy_summary.json", report)
    return report


def _response_from_fields(
    actual: Any, reference: Any, config: Mapping[str, Any]
) -> np.ndarray:
    enhancement = _normalised_fields(
        actual,
        reference,
        reference_component=str(config["source"]["polarisation"]),
        mode=str(config["measurement"]["normalization_mode"]),
    )["amplitude_enhancement"]
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
    checks: Sequence[str] = ("configuration", "theory", "hierarchy"),
    results_root: str | Path | None = None,
    resolutions: Sequence[int] | None = None,
    paddings_um: Sequence[float] | None = None,
    cache_enabled: bool = True,
) -> Path:
    directory = resolve_experiment(experiment, results_root)
    config = _load_config(directory)
    selected = {str(value).strip().lower() for value in checks}
    if "all" in selected:
        selected = {"configuration", "theory", "hierarchy", "resolution", "domain", "time"}
    unknown = selected - {"configuration", "theory", "hierarchy", "resolution", "domain", "time"}
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
        if "hierarchy" in selected:
            reports["hierarchy"] = _hierarchy_check(config, directory, workspace.directory)
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
