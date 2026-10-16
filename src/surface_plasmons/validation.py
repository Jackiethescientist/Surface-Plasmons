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


def _write_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")
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
    resolution = int(PRECISION_MODES[str(config["precision"])]["broadband_resolution"])
    if str(config["geometry"]["particle_shape"]) == "sphere":
        particle_width_um = 2.0 * float(config["geometry"]["particle_radius_um"])
    else:
        particle_width_um = float(config["geometry"]["cube_size_um"])
    probe_offset_um = float(
        config["measurement"]["probe_offset_from_particle_surface_um"]
    )
    cells_across_particle = particle_width_um * resolution
    cells_across_probe_offset = probe_offset_um * resolution
    sampling_warnings: list[str] = []
    if cells_across_particle < 12.0:
        sampling_warnings.append(
            "Fewer than 12 broadband grid cells span the particle width; "
            "resolution convergence is required before quantitative use."
        )
    if cells_across_probe_offset < 3.0:
        sampling_warnings.append(
            "Fewer than 3 broadband grid cells span the probe-to-surface offset; "
            "the near-surface field is strongly grid-sensitive."
        )
    report = {
        "status": "passed",
        "dimensions": 3,
        "particle_count": len(positions),
        "defect_count": len(config["geometry"].get("defects", [])),
        "propagation_axis": config["source"]["propagation_axis"],
        "polarisation": config["source"]["polarisation"],
        "numerical_sampling": {
            "precision_mode": config["precision"],
            "broadband_resolution_pixels_per_um": resolution,
            "particle_width_um": particle_width_um,
            "cells_across_particle": cells_across_particle,
            "probe_offset_from_surface_um": probe_offset_um,
            "cells_across_probe_offset": cells_across_probe_offset,
            "sampling_status": (
                "warning_requires_convergence"
                if sampling_warnings
                else "candidate_still_requires_convergence_test"
            ),
            "warnings": sampling_warnings,
        },
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


def _field_agreement_status(comparison: Any) -> str:
    """Classify agreement without confusing successful execution with a pass."""

    correlation = comparison.amplitude_correlation
    if comparison.amplitude_nrmse <= 0.10 and (
        correlation is None or correlation >= 0.90
    ):
        return "passed_quantitative"
    if comparison.amplitude_nrmse <= 0.20 and (
        correlation is None or correlation >= 0.70
    ):
        return "passed_qualitative_only"
    return "failed"


def _percent(value: Any) -> str:
    """Format a stored fractional error as a percentage for the reader."""

    if value is None or value == "":
        return "Not available"
    return f"{100.0 * float(value):.2f}%"


def _correlation(value: Any) -> str:
    if value is None or value == "":
        return "Not available"
    return f"{float(value):.3f}"


def _agreement_label(status: Any) -> str:
    return {
        "passed_quantitative": "PASS — quantitative",
        "passed_qualitative_only": "PARTIAL — qualitative only",
        "passed": "PASS",
        "failed": "FAIL",
        "skipped": "NOT TESTED",
        "unavailable": "NOT AVAILABLE",
    }.get(str(status), str(status).replace("_", " ").upper())


def _agreement_meaning(status: Any) -> str:
    return {
        "passed_quantitative": (
            "The model and MEEP agree within the quantitative tolerances."
        ),
        "passed_qualitative_only": (
            "The overall trend agrees, but the result is not accurate enough for "
            "quantitative claims."
        ),
        "passed": "This check meets its stated tolerance.",
        "failed": (
            "The model does not reproduce this MEEP observable within the stated "
            "tolerances. This is model disagreement, not automatically a MEEP error."
        ),
        "skipped": "The assumptions required for this theory check were not satisfied.",
        "unavailable": "The saved simulation does not contain the data needed for this check.",
    }.get(str(status), "Review the detailed validation report for this status.")


def _comparison_summary_markdown(
    config: Mapping[str, Any], reports: Mapping[str, Any]
) -> str:
    """Build a short human-readable validity report from computed metrics."""

    theory = reports.get("theory", {})
    hierarchy = reports.get("hierarchy", {})
    configuration = reports.get("configuration", {})
    resolution = reports.get("resolution", {})
    domain = reports.get("domain", {})

    equation2_status = theory.get("agreement_status")
    scattered_status = theory.get("scattered_field_agreement_status")
    equation2_statuses = [
        status
        for status in (equation2_status, scattered_status)
        if status is not None
    ]
    numerical_statuses = [
        report.get("convergence_status")
        for report in (resolution, domain)
        if report
    ]
    both_convergence_checks_run = bool(resolution) and bool(domain)
    both_converged = (
        both_convergence_checks_run
        and all(status == "passed" for status in numerical_statuses)
    )
    convergence_failed = any(status == "failed" for status in numerical_statuses)
    if theory.get("status") == "skipped":
        final_result = "NOT VALIDATED AGAINST EQUATION 2"
        final_meaning = str(theory.get("reason", "Equation 2 was not applicable."))
    elif "failed" in equation2_statuses:
        final_result = "NOT VALIDATED AGAINST EQUATION 2"
        final_meaning = (
            "At least one Section-2 coupled-dipole field comparison does not match "
            "the MEEP field within the quantitative or qualitative tolerances."
        )
    elif convergence_failed:
        final_result = "NOT NUMERICALLY CONVERGED"
        final_meaning = (
            "At least one resolution or domain convergence check failed, so the MEEP "
            "values are not yet stable enough for a final theory comparison."
        )
    elif (
        equation2_statuses
        and all(status == "passed_quantitative" for status in equation2_statuses)
        and both_converged
    ):
        final_result = "VALIDATED FOR THE REPORTED EQUATION-2 OBSERVABLE"
        final_meaning = (
            "The MEEP field agrees quantitatively with Equation 2 and both resolution "
            "and domain convergence checks passed."
        )
    elif equation2_statuses and all(
        status in {"passed_quantitative", "passed_qualitative_only"}
        for status in equation2_statuses
    ):
        final_result = "PROVISIONAL — NUMERICAL CONVERGENCE STILL REQUIRED"
        final_meaning = (
            "The theory comparison is acceptable at its reported level, but both "
            "resolution and domain convergence must pass before calling it validated."
        )
    else:
        final_result = "INCOMPLETE"
        final_meaning = "No completed Equation-2 comparison is available for this simulation."

    lines = [
        "# Comparison summary",
        "",
        f"**Simulation:** `{config['simulation_name']}`  ",
        f"**Final result:** **{final_result}**  ",
        f"**Meaning:** {final_meaning}",
        "",
        "Errors below are normalized root-mean-square errors (NRMSE) unless stated "
        "otherwise. Zero percent is exact agreement; smaller is better. Correlation "
        "describes curve shape, where +1 is identical, 0 is unrelated, and -1 is "
        "opposite. The field quantities are amplitudes (E/E_ref or E/E0), not squared "
        "intensities.",
        "",
        "## Direct Equation 2 field check",
        "",
        "| Theory check | Variables compared | Error | Correlation | Result | Meaning |",
        "|---|---|---:|---:|---|---|",
    ]

    if theory.get("status") == "completed":
        wavelength = float(theory.get("wavelength_nm", 0.0))
        for label, key, status_key, variables in (
            (
                "Equation 2 total field",
                "same_scale_total_field_comparison",
                "agreement_status",
                "MEEP total field amplitude E/E_ref vs CDA total field amplitude E/E0",
            ),
            (
                "Equation 2 scattered field",
                "same_scale_scattered_field_comparison",
                "scattered_field_agreement_status",
                "MEEP scattered amplitude (E-E_ref)/E_ref vs CDA dipole-scattered amplitude E_scat/E0",
            ),
        ):
            metrics = theory.get(key, {})
            status = theory.get(status_key)
            lines.append(
                f"| {label} at {wavelength:.3f} nm | {variables} | "
                f"{_percent(metrics.get('amplitude_nrmse'))} | "
                f"{_correlation(metrics.get('amplitude_correlation'))} | "
                f"{_agreement_label(status)} | {_agreement_meaning(status)} |"
            )
        residual = theory.get("relative_equation2_residual")
        residual_status = (
            "passed"
            if residual is not None and float(residual) <= 1e-10
            else "failed"
        )
        lines.append(
            "| Equation 2 solver identity | Solved local field vs the right-hand side "
            "of Equation 2 | "
            f"{_percent(residual)} | Not applicable | "
            f"{_agreement_label(residual_status)} | Checks the algebra and implementation "
            "only; it does not show agreement with MEEP. |"
        )
    else:
        status = theory.get("status", "unavailable")
        reason = str(theory.get("reason", "Equation-2 validation was not run."))
        lines.append(
            f"| Equation 2 | No comparable variables | Not available | Not available | "
            f"{_agreement_label(status)} | {reason} |"
        )

    lines.extend(
        [
            "",
            "## Model hierarchy across the wavelength scan",
            "",
            "| Theory | Observable | Variables compared | NRMSE | Peak wavelength error | Correlation | Result | Meaning |",
            "|---|---|---|---:|---:|---:|---|---|",
        ]
    )
    comparison_groups = (
        (
            "Maximum external field",
            "same_observable_external_field_comparisons",
            "maximum MEEP amplitude over the external probe line vs maximum theory amplitude",
        ),
        (
            "Fixed central probe",
            "fixed_central_probe_comparisons",
            "MEEP amplitude at the fixed central probe vs theory amplitude at the same probe",
        ),
        (
            "RMS external field",
            "rms_probe_line_comparisons",
            "MEEP RMS amplitude over the probe line vs theory RMS amplitude over the same line",
        ),
    )
    theory_names = {
        "quasistatic_cda_vs_meep": "Equation 1 quasistatic CDA",
        "retarded_equations2_3_cda_vs_meep": "Equations 2–3 retarded CDA",
        "mie_dipole_cda_vs_meep": "Mie electric-dipole CDA",
    }
    hierarchy_rows = 0
    for observable, group_key, variables in comparison_groups:
        for key, metrics in hierarchy.get(group_key, {}).items():
            hierarchy_rows += 1
            status = metrics.get("agreement_status")
            boundary_note = (
                " Peak lies at the scan boundary, so the true theory peak may be outside the scan."
                if metrics.get("model_peak_at_scan_boundary")
                else ""
            )
            lines.append(
                f"| {theory_names.get(key, key)} | {observable} | {variables} | "
                f"{_percent(metrics.get('same_scale_nrmse'))} | "
                f"{_percent(metrics.get('relative_peak_wavelength_error'))} | "
                f"{_correlation(metrics.get('correlation'))} | "
                f"{_agreement_label(status)} | {_agreement_meaning(status)}{boundary_note} |"
            )
    if hierarchy_rows == 0:
        status = hierarchy.get("status", "unavailable")
        reason = str(hierarchy.get("reason", "The model hierarchy was not run."))
        lines.append(
            f"| CDA and Mie-dipole hierarchy | Not available | No directly comparable "
            f"variables | Not available | Not available | Not available | "
            f"{_agreement_label(status)} | {reason} |"
        )

    if hierarchy.get("status") == "completed":
        one_d_error = hierarchy.get("maximum_scalar_vs_vector_relative_error")
        one_d_status = (
            "passed"
            if one_d_error is not None and float(one_d_error) <= 1e-10
            else "failed"
        )
        lines.extend(
            [
                "",
                "## Formula consistency and Mie scope",
                "",
                "| Check | Variables compared | Error | Result | Meaning |",
                "|---|---|---:|---|---|",
                "| Direct 1-D formula | Scalar 1-D local field E_loc/E0 vs the matching "
                "component of the 3-D vector CDA local field E_loc/E0 | "
                f"{_percent(one_d_error)} | {_agreement_label(one_d_status)} | "
                "This checks that the direct 1-D formula is an exact reduction of the vector CDA solver. |",
                "| Exact multipole Mie theory | Single-sphere extinction/scattering/absorption "
                "cross-sections vs array near-field amplitude | Not directly comparable | "
                "NOT A VALID DIRECT CHECK | These are different observables; no honest error "
                "percentage can be calculated from them. |",
                f"| Equations 4–5 uniform-chain result | Uniform-chain dipole moment vs finite "
                f"simulation near field | Not directly compared | CALCULATED ONLY | "
                f"{hierarchy.get('equation4_dipole_sum', 'No result was reported.')} |",
            ]
        )

    sampling = configuration.get("numerical_sampling", {})
    lines.extend(
        [
            "",
            "## Numerical validity",
            "",
            "| Check | Values compared | Error/change | Result | Meaning |",
            "|---|---|---:|---|---|",
        ]
    )
    if sampling:
        warnings = sampling.get("warnings", [])
        sampling_result = "WARNING" if warnings else "CANDIDATE"
        sampling_meaning = (
            " ".join(str(item) for item in warnings)
            if warnings
            else "Grid sampling is plausible, but a convergence test is still required."
        )
        lines.append(
            f"| Grid sampling | {float(sampling.get('cells_across_particle', 0.0)):.2f} "
            "cells across the particle; "
            f"{float(sampling.get('cells_across_probe_offset', 0.0)):.2f} cells across "
            f"the probe offset | Not an error metric | {sampling_result} | {sampling_meaning} |"
        )
    else:
        lines.append(
            "| Grid sampling | Not available | Not available | NOT RUN | Configuration check was not run. |"
        )
    for title, report in (("Resolution convergence", resolution), ("Domain convergence", domain)):
        if report:
            status = report.get("convergence_status")
            finest = report.get("finest_comparison", {})
            changes = (
                f"spectrum NRMSE {_percent(finest.get('normalised_rmse_vs_reference'))}; "
                f"peak wavelength {_percent(report.get('relative_peak_wavelength_change'))}; "
                f"peak amplitude {_percent(report.get('relative_peak_amplitude_change'))}"
            )
            lines.append(
                f"| {title} | Two finest tested settings | {changes} | "
                f"{_agreement_label(status)} | {_agreement_meaning(status)} |"
            )
        else:
            lines.append(
                f"| {title} | No runs compared | Not available | NOT RUN | Required before "
                "the MEEP result can be called numerically converged. |"
            )

    lines.extend(
        [
            "",
            "## Decision rules",
            "",
            "- Field-profile quantitative agreement: amplitude NRMSE ≤ 10% and correlation ≥ 0.90.",
            "- Field-profile qualitative agreement: amplitude NRMSE ≤ 20% and correlation ≥ 0.70.",
            "- Spectrum quantitative agreement also requires peak-wavelength error ≤ 2% and no boundary peak.",
            "- Spectrum qualitative agreement also requires peak-wavelength error ≤ 5% and no boundary peak.",
            "- Numerical convergence requires spectrum NRMSE ≤ 5%, peak-wavelength change ≤ 2%, and peak-amplitude change ≤ 5%.",
            "",
            "A failed simplified theory does not by itself prove the MEEP simulation is wrong. "
            "It means that approximation does not reproduce the selected MEEP observable. "
            "MEEP validity still requires resolution and domain convergence.",
        ]
    )
    return "\n".join(lines)


def _accumulated_validation_reports(
    validation_root: Path, current_reports: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep the latest available result for every check across validation runs."""

    accumulated: dict[str, Any] = {}
    summaries = sorted(
        validation_root.glob("*/validation_summary.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    for path in summaries:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            accumulated.update(value)
    accumulated.update(current_reports)
    return accumulated


def _normalised_reference_from_rows(
    rows: Sequence[Mapping[str, str]],
    analytical_fallback: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Read the saved particle-free field, falling back for pre-v0.5 results."""

    required = {
        f"normalised_reference_{component}_{part}"
        for component in ("Ex", "Ey", "Ez")
        for part in ("real", "imag")
    }
    if rows and required.issubset(rows[0]):
        return (
            np.asarray(
                [
                    [
                        complex(
                            float(row[f"normalised_reference_{component}_real"]),
                            float(row[f"normalised_reference_{component}_imag"]),
                        )
                        for component in ("Ex", "Ey", "Ez")
                    ]
                    for row in rows
                ],
                dtype=complex,
            ),
            "stored point-matched particle-free MEEP field",
        )
    return np.asarray(analytical_fallback, dtype=complex), (
        "analytical incident field fallback; rerun the simulation to store the "
        "point-matched particle-free vector"
    )


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
    predicted_incident_normalised = incident / reference_scale
    measured_reference, reference_vector_source = _normalised_reference_from_rows(
        profile_rows,
        predicted_incident_normalised,
    )
    measured_scattered = measured - measured_reference
    predicted_scattered = predicted_normalised - predicted_incident_normalised
    scattered_comparison = compare_complex_fields(
        measured_scattered,
        predicted_scattered,
    )

    measured_amplitude = np.sqrt(np.sum(np.abs(measured) ** 2, axis=1))
    theory_amplitude = np.sqrt(np.sum(np.abs(predicted_normalised) ** 2, axis=1))
    measured_scattered_amplitude = np.linalg.norm(measured_scattered, axis=1)
    theory_scattered_amplitude = np.linalg.norm(predicted_scattered, axis=1)
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
            "meep_scattered_amplitude_Escat_over_Eref": float(
                measured_scattered_amplitude[index]
            ),
            "equation2_scattered_amplitude_Escat_over_E0": float(
                theory_scattered_amplitude[index]
            ),
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
            row[f"meep_reference_{component}_real"] = float(
                np.real(measured_reference[index, component_index_value])
            )
            row[f"meep_reference_{component}_imag"] = float(
                np.imag(measured_reference[index, component_index_value])
            )
            row[f"meep_scattered_{component}_real"] = float(
                np.real(measured_scattered[index, component_index_value])
            )
            row[f"meep_scattered_{component}_imag"] = float(
                np.imag(measured_scattered[index, component_index_value])
            )
            row[f"equation2_scattered_{component}_real"] = float(
                np.real(predicted_scattered[index, component_index_value])
            )
            row[f"equation2_scattered_{component}_imag"] = float(
                np.imag(predicted_scattered[index, component_index_value])
            )
        rows.append(row)
    _write_csv(output / "equation2_external_field_comparison.csv", tuple(rows[0]), rows)

    dipole_rows: list[dict[str, Any]] = []
    equation_rhs = solution.incident_fields - (
        solution.interaction_matrix_A @ solution.dipole_moments.reshape(-1)
    ).reshape(solution.local_fields.shape)
    residual = solution.local_fields - equation_rhs
    local_field_scale = float(
        np.max(np.linalg.norm(solution.local_fields, axis=1))
    ) or 1.0
    maximum_equation2_residual = float(
        np.max(np.linalg.norm(residual, axis=1))
    )
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

    figure, axis = plt.subplots(figsize=(9.0, 5.6))
    axis.plot(
        points[:, 0],
        measured_scattered_amplitude,
        linewidth=2.0,
        label="MEEP: particle field minus reference field",
    )
    axis.plot(
        points[:, 0],
        theory_scattered_amplitude,
        linewidth=1.7,
        label="Equations 2-3 CDA: dipole-scattered field",
    )
    axis.set_xlabel("Position along particle line, x (µm)")
    axis.set_ylabel(r"Scattered electric-field amplitude ratio, $|E_{scat}|/|E_{ref}|$")
    axis.set_title(f"Scattered field amplitude at {wavelength_nm:.3f} nm")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(
        output / "equation2_scattered_field_amplitude_profile.png",
        dpi=230,
        bbox_inches="tight",
    )
    plt.close(figure)

    report = {
        "status": "completed",
        "execution_status": "completed",
        "applicability_status": "passed",
        "agreement_status": _field_agreement_status(comparison),
        "numerical_convergence_status": "not_run_by_theory_check",
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
        "same_scale_total_field_comparison": comparison.to_dict(),
        "same_scale_scattered_field_comparison": scattered_comparison.to_dict(),
        "scattered_field_agreement_status": _field_agreement_status(
            scattered_comparison
        ),
        "reference_vector_source": reference_vector_source,
        "maximum_equation2_residual": maximum_equation2_residual,
        "relative_equation2_residual": maximum_equation2_residual / local_field_scale,
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
    wavelengths_nm: Sequence[float] | None = None,
) -> dict[str, float | int | bool | str | None]:
    measured = np.asarray(reference, dtype=float)
    modelled = np.asarray(prediction, dtype=float)
    scale = float(np.sqrt(np.mean(measured**2))) or 1.0
    nrmse = float(np.sqrt(np.mean((modelled - measured) ** 2)) / scale)
    correlation = (
        float(np.corrcoef(measured, modelled)[0, 1])
        if measured.size > 1 and np.std(measured) > 0 and np.std(modelled) > 0
        else None
    )
    meep_peak_index = int(np.argmax(measured))
    model_peak_index = int(np.argmax(modelled))
    result: dict[str, float | int | bool | str | None] = {
        "same_scale_nrmse": nrmse,
        "correlation": correlation,
        "meep_peak_index": meep_peak_index,
        "model_peak_index": model_peak_index,
    }
    peak_error: float | None = None
    model_peak_at_boundary = model_peak_index in {0, measured.size - 1}
    if wavelengths_nm is not None:
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if wavelengths.size != measured.size:
            raise ValidationError("Spectrum wavelengths and responses have different lengths.")
        meep_peak_wavelength = float(wavelengths[meep_peak_index])
        model_peak_wavelength = float(wavelengths[model_peak_index])
        peak_error = abs(model_peak_wavelength - meep_peak_wavelength) / meep_peak_wavelength
        result.update(
            {
                "meep_peak_wavelength_nm": meep_peak_wavelength,
                "model_peak_wavelength_nm": model_peak_wavelength,
                "relative_peak_wavelength_error": peak_error,
                "model_peak_at_scan_boundary": model_peak_at_boundary,
            }
        )
    if nrmse <= 0.10 and (correlation is None or correlation >= 0.90) and (
        peak_error is None or peak_error <= 0.02
    ) and not model_peak_at_boundary:
        agreement = "passed_quantitative"
    elif nrmse <= 0.20 and (correlation is None or correlation >= 0.70) and (
        peak_error is None or peak_error <= 0.05
    ) and not model_peak_at_boundary:
        agreement = "passed_qualitative_only"
    else:
        agreement = "failed"
    result["agreement_status"] = agreement
    return result


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
    } | {
        f"normalised_reference_{component}_{part}"
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
        measured_reference = np.asarray(
            [
                [
                    complex(
                        float(row[f"normalised_reference_{component}_real"]),
                        float(row[f"normalised_reference_{component}_imag"]),
                    )
                    for component in ("Ex", "Ey", "Ez")
                ]
                for row in profile_rows
            ],
            dtype=complex,
        )
        measured_scattered = measured - measured_reference
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
        alpha_radiation_reaction = sphere_polarizability(
            radius_um,
            epsilon_particle,
            epsilon_background,
            wavelength_um=wavelength_um,
            model="radiation_reaction",
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
        incident_at_points = (
            np.exp(1j * wavevector * (points @ direction))[:, np.newaxis]
            * polarisation
        )
        quasistatic_scattered = field_quasistatic - incident_at_points
        retarded_scattered = field_retarded - incident_at_points
        mie_dipole_scattered = field_mie - incident_at_points
        centre_particle_x = min(
            (particle.actual_x_um for particle in positions),
            key=abs,
        )
        centre_probe_index = int(
            np.argmin(np.abs(points[:, 0] - centre_particle_x))
        )
        measured_amplitudes = np.linalg.norm(measured, axis=1)
        quasistatic_amplitudes = np.linalg.norm(field_quasistatic, axis=1)
        retarded_amplitudes = np.linalg.norm(field_retarded, axis=1)
        mie_dipole_amplitudes = np.linalg.norm(field_mie, axis=1)
        measured_scattered_amplitudes = np.linalg.norm(
            measured_scattered, axis=1
        )
        quasistatic_scattered_amplitudes = np.linalg.norm(
            quasistatic_scattered, axis=1
        )
        retarded_scattered_amplitudes = np.linalg.norm(
            retarded_scattered, axis=1
        )
        mie_dipole_scattered_amplitudes = np.linalg.norm(
            mie_dipole_scattered, axis=1
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
        dipole_sum: complex | None = None
        equation4_dipole: complex | None = None
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
            equation4_dipole = uniform_chain_dipole_moment(
                alpha_quasistatic, dipole_sum
            )
        spectrum_rows.append(
            {
                "wavelength_nm": wavelength_nm,
                "wavelength_um": wavelength_um,
                "particle_radius_um": radius_um,
                "spacing_um": float(config["geometry"]["spacing_um"]),
                "nominal_particle_count": int(config["geometry"]["particle_count"]),
                "active_particle_count": len(positions),
                "orientation": orientation,
                "epsilon_particle_real": float(epsilon_particle.real),
                "epsilon_particle_imag": float(epsilon_particle.imag),
                "epsilon_background_real": float(epsilon_background.real),
                "epsilon_background_imag": float(epsilon_background.imag),
                "wavevector_real_per_um": float(wavevector.real),
                "wavevector_imag_per_um": float(wavevector.imag),
                "alpha_equation1_real_um3": float(alpha_quasistatic.real),
                "alpha_equation1_imag_um3": float(alpha_quasistatic.imag),
                "alpha_equation1_magnitude_um3": float(abs(alpha_quasistatic)),
                "alpha_radiation_reaction_real_um3": float(
                    alpha_radiation_reaction.real
                ),
                "alpha_radiation_reaction_imag_um3": float(
                    alpha_radiation_reaction.imag
                ),
                "alpha_radiation_reaction_magnitude_um3": float(
                    abs(alpha_radiation_reaction)
                ),
                "alpha_mie_a1_real_um3": float(alpha_mie.real),
                "alpha_mie_a1_imag_um3": float(alpha_mie.imag),
                "alpha_mie_a1_magnitude_um3": float(abs(alpha_mie)),
                "equation5_dipole_sum_real_per_um3": (
                    "" if dipole_sum is None else float(dipole_sum.real)
                ),
                "equation5_dipole_sum_imag_per_um3": (
                    "" if dipole_sum is None else float(dipole_sum.imag)
                ),
                "equation4_uniform_dipole_real_over_epsilon0_E0_um3": (
                    "" if equation4_dipole is None else float(equation4_dipole.real)
                ),
                "equation4_uniform_dipole_imag_over_epsilon0_E0_um3": (
                    "" if equation4_dipole is None else float(equation4_dipole.imag)
                ),
                "central_particle_x_um": centre_particle_x,
                "central_probe_x_um": float(points[centre_probe_index, 0]),
                "meep_central_probe_E_over_Eref": float(
                    measured_amplitudes[centre_probe_index]
                ),
                "cda_quasistatic_central_probe_E_over_E0": float(
                    quasistatic_amplitudes[centre_probe_index]
                ),
                "cda_equations2_3_central_probe_E_over_E0": float(
                    retarded_amplitudes[centre_probe_index]
                ),
                "mie_dipole_cda_central_probe_E_over_E0": float(
                    mie_dipole_amplitudes[centre_probe_index]
                ),
                "meep_rms_external_E_over_Eref": float(
                    np.sqrt(np.mean(measured_amplitudes**2))
                ),
                "cda_quasistatic_rms_external_E_over_E0": float(
                    np.sqrt(np.mean(quasistatic_amplitudes**2))
                ),
                "cda_equations2_3_rms_external_E_over_E0": float(
                    np.sqrt(np.mean(retarded_amplitudes**2))
                ),
                "mie_dipole_cda_rms_external_E_over_E0": float(
                    np.sqrt(np.mean(mie_dipole_amplitudes**2))
                ),
                "meep_max_external_E_over_Eref": float(np.max(measured_amplitudes)),
                "cda_quasistatic_max_external_E_over_E0": float(
                    np.max(quasistatic_amplitudes)
                ),
                "cda_equations2_3_max_external_E_over_E0": float(
                    np.max(retarded_amplitudes)
                ),
                "mie_dipole_cda_max_external_E_over_E0": float(
                    np.max(mie_dipole_amplitudes)
                ),
                "meep_central_probe_Escat_over_Eref": float(
                    measured_scattered_amplitudes[centre_probe_index]
                ),
                "cda_quasistatic_central_probe_Escat_over_E0": float(
                    quasistatic_scattered_amplitudes[centre_probe_index]
                ),
                "cda_equations2_3_central_probe_Escat_over_E0": float(
                    retarded_scattered_amplitudes[centre_probe_index]
                ),
                "mie_dipole_cda_central_probe_Escat_over_E0": float(
                    mie_dipole_scattered_amplitudes[centre_probe_index]
                ),
                "meep_rms_external_Escat_over_Eref": float(
                    np.sqrt(np.mean(measured_scattered_amplitudes**2))
                ),
                "cda_quasistatic_rms_external_Escat_over_E0": float(
                    np.sqrt(np.mean(quasistatic_scattered_amplitudes**2))
                ),
                "cda_equations2_3_rms_external_Escat_over_E0": float(
                    np.sqrt(np.mean(retarded_scattered_amplitudes**2))
                ),
                "mie_dipole_cda_rms_external_Escat_over_E0": float(
                    np.sqrt(np.mean(mie_dipole_scattered_amplitudes**2))
                ),
                "meep_max_external_Escat_over_Eref": float(
                    np.max(measured_scattered_amplitudes)
                ),
                "cda_quasistatic_max_external_Escat_over_E0": float(
                    np.max(quasistatic_scattered_amplitudes)
                ),
                "cda_equations2_3_max_external_Escat_over_E0": float(
                    np.max(retarded_scattered_amplitudes)
                ),
                "mie_dipole_cda_max_external_Escat_over_E0": float(
                    np.max(mie_dipole_scattered_amplitudes)
                ),
                "scalar_1d_max_local_E_over_E0": float(
                    np.max(np.abs(scalar.local_fields))
                ),
                "vector_cda_max_local_E_over_E0": float(
                    np.max(np.abs(vector_local_component))
                ),
                "scalar_vs_vector_relative_error": scalar_vector_error,
                "equation4_uniform_dipole_magnitude_over_epsilon0_E0_um3": (
                    "" if equation4_dipole is None else float(abs(equation4_dipole))
                ),
                "mie_size_parameter": mie.size_parameter,
                "mie_exact_extinction_cross_section_um2": mie.extinction_cross_section_um2,
                "mie_exact_scattering_cross_section_um2": mie.scattering_cross_section_um2,
                "mie_exact_absorption_cross_section_um2": mie.absorption_cross_section_um2,
                "mie_exact_extinction_efficiency": mie.extinction_efficiency,
                "mie_exact_scattering_efficiency": mie.scattering_efficiency,
                "mie_exact_absorption_efficiency": mie.absorption_efficiency,
                "mie_a1_real": float(mie.electric_coefficients[0].real),
                "mie_a1_imag": float(mie.electric_coefficients[0].imag),
                "mie_b1_real": float(mie.magnetic_coefficients[0].real),
                "mie_b1_imag": float(mie.magnetic_coefficients[0].imag),
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
    meep_central = np.asarray(
        [row["meep_central_probe_E_over_Eref"] for row in spectrum_rows], dtype=float
    )
    quasistatic_central = np.asarray(
        [row["cda_quasistatic_central_probe_E_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    retarded_central = np.asarray(
        [row["cda_equations2_3_central_probe_E_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    mie_dipole_central = np.asarray(
        [row["mie_dipole_cda_central_probe_E_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    meep_rms = np.asarray(
        [row["meep_rms_external_E_over_Eref"] for row in spectrum_rows], dtype=float
    )
    quasistatic_rms = np.asarray(
        [row["cda_quasistatic_rms_external_E_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    retarded_rms = np.asarray(
        [row["cda_equations2_3_rms_external_E_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    mie_dipole_rms = np.asarray(
        [row["mie_dipole_cda_rms_external_E_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    meep_scattered_external = np.asarray(
        [row["meep_max_external_Escat_over_Eref"] for row in spectrum_rows],
        dtype=float,
    )
    quasistatic_scattered_external = np.asarray(
        [row["cda_quasistatic_max_external_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    retarded_scattered_external = np.asarray(
        [row["cda_equations2_3_max_external_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    mie_dipole_scattered_external = np.asarray(
        [row["mie_dipole_cda_max_external_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    meep_scattered_central = np.asarray(
        [row["meep_central_probe_Escat_over_Eref"] for row in spectrum_rows],
        dtype=float,
    )
    quasistatic_scattered_central = np.asarray(
        [row["cda_quasistatic_central_probe_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    retarded_scattered_central = np.asarray(
        [row["cda_equations2_3_central_probe_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    mie_dipole_scattered_central = np.asarray(
        [row["mie_dipole_cda_central_probe_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    meep_scattered_rms = np.asarray(
        [row["meep_rms_external_Escat_over_Eref"] for row in spectrum_rows],
        dtype=float,
    )
    quasistatic_scattered_rms = np.asarray(
        [row["cda_quasistatic_rms_external_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    retarded_scattered_rms = np.asarray(
        [row["cda_equations2_3_rms_external_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
    )
    mie_dipole_scattered_rms = np.asarray(
        [row["mie_dipole_cda_rms_external_Escat_over_E0"] for row in spectrum_rows],
        dtype=float,
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

    figure, axes = plt.subplots(2, 1, figsize=(9.5, 9.0), sharex=True)
    for axis, measured_values, model_values, title, ylabel in (
        (
            axes[0],
            meep_central,
            (quasistatic_central, retarded_central, mie_dipole_central),
            "Fixed probe nearest the central particle",
            r"Field-amplitude ratio, $|E|/|E_{ref}|$ or $|E|/|E_0|$",
        ),
        (
            axes[1],
            meep_rms,
            (quasistatic_rms, retarded_rms, mie_dipole_rms),
            "RMS over the complete external probe line",
            r"RMS field-amplitude ratio",
        ),
    ):
        axis.plot(wavelengths, measured_values, linewidth=2.0, label="3-D MEEP FDTD")
        axis.plot(wavelengths, model_values[0], label="Quasistatic CDA")
        axis.plot(wavelengths, model_values[1], label="Equations 2-3 CDA")
        axis.plot(wavelengths, model_values[2], label="Mie-dipole CDA")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    axes[1].set_xlabel("Vacuum wavelength (nm)")
    figure.tight_layout()
    figure.savefig(
        output / "fixed_probe_and_rms_spectrum_comparison.png",
        dpi=230,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(9.5, 12.0), sharex=True)
    for axis, measured_values, model_values, title, ylabel in (
        (
            axes[0],
            meep_scattered_external,
            (
                quasistatic_scattered_external,
                retarded_scattered_external,
                mie_dipole_scattered_external,
            ),
            "Maximum scattered field over the external probe line",
            r"Maximum $|E_{scat}|/|E_{ref}|$ or $|E_{scat}|/|E_0|$",
        ),
        (
            axes[1],
            meep_scattered_central,
            (
                quasistatic_scattered_central,
                retarded_scattered_central,
                mie_dipole_scattered_central,
            ),
            "Scattered field at the fixed central probe",
            r"Central $|E_{scat}|/|E_{ref}|$ or $|E_{scat}|/|E_0|$",
        ),
        (
            axes[2],
            meep_scattered_rms,
            (
                quasistatic_scattered_rms,
                retarded_scattered_rms,
                mie_dipole_scattered_rms,
            ),
            "RMS scattered field over the external probe line",
            r"RMS $|E_{scat}|/|E_{ref}|$ or $|E_{scat}|/|E_0|$",
        ),
    ):
        axis.plot(wavelengths, measured_values, linewidth=2.0, label="3-D MEEP FDTD")
        axis.plot(wavelengths, model_values[0], label="Quasistatic CDA")
        axis.plot(wavelengths, model_values[1], label="Equations 2-3 CDA")
        axis.plot(wavelengths, model_values[2], label="Mie-dipole CDA")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(frameon=False)
    axes[2].set_xlabel("Vacuum wavelength (nm)")
    figure.tight_layout()
    figure.savefig(
        output / "scattered_field_model_hierarchy.png",
        dpi=230,
        bbox_inches="tight",
    )
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
        "execution_status": "completed",
        "applicability_status": "passed",
        "agreement_status": _same_scale_spectrum_metrics(
            meep_scattered_central, retarded_scattered_central, wavelengths
        )["agreement_status"],
        "numerical_convergence_status": "not_run_by_hierarchy_check",
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
                meep_external, quasistatic_external, wavelengths
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_external, retarded_external, wavelengths
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_external, mie_dipole_external, wavelengths
            ),
        },
        "fixed_central_probe_comparisons": {
            "quasistatic_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_central, quasistatic_central, wavelengths
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_central, retarded_central, wavelengths
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_central, mie_dipole_central, wavelengths
            ),
        },
        "rms_probe_line_comparisons": {
            "quasistatic_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_rms, quasistatic_rms, wavelengths
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_rms, retarded_rms, wavelengths
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_rms, mie_dipole_rms, wavelengths
            ),
        },
        "maximum_scattered_field_comparisons": {
            "quasistatic_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_external,
                quasistatic_scattered_external,
                wavelengths,
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_external,
                retarded_scattered_external,
                wavelengths,
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_external,
                mie_dipole_scattered_external,
                wavelengths,
            ),
        },
        "fixed_central_probe_scattered_field_comparisons": {
            "quasistatic_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_central,
                quasistatic_scattered_central,
                wavelengths,
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_central,
                retarded_scattered_central,
                wavelengths,
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_central,
                mie_dipole_scattered_central,
                wavelengths,
            ),
        },
        "rms_scattered_probe_line_comparisons": {
            "quasistatic_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_rms,
                quasistatic_scattered_rms,
                wavelengths,
            ),
            "retarded_equations2_3_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_rms,
                retarded_scattered_rms,
                wavelengths,
            ),
            "mie_dipole_cda_vs_meep": _same_scale_spectrum_metrics(
                meep_scattered_rms,
                mie_dipole_scattered_rms,
                wavelengths,
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
    ordered_values = sorted({float(value) for value in values})
    if not ordered_values:
        raise ValidationError(f"At least one {kind} convergence value is required.")
    for value in ordered_values:
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
    finest_comparison = rows[-2] if len(rows) > 1 else rows[-1]
    reference_row = rows[-1]
    peak_scale = abs(float(reference_row["peak_wavelength_nm"])) or 1.0
    amplitude_scale = abs(float(reference_row["peak_field_enhancement"])) or 1.0
    relative_peak_shift = abs(
        float(finest_comparison["peak_wavelength_nm"])
        - float(reference_row["peak_wavelength_nm"])
    ) / peak_scale
    relative_peak_amplitude_change = abs(
        float(finest_comparison["peak_field_enhancement"])
        - float(reference_row["peak_field_enhancement"])
    ) / amplitude_scale
    convergence_status = (
        "passed"
        if float(finest_comparison["normalised_rmse_vs_reference"]) <= 0.05
        and relative_peak_shift <= 0.02
        and relative_peak_amplitude_change <= 0.05
        else "failed"
    )
    report = {
        "status": "completed",
        "convergence_status": convergence_status,
        "criteria": {
            "maximum_normalised_rmse": 0.05,
            "maximum_relative_peak_wavelength_change": 0.02,
            "maximum_relative_peak_amplitude_change": 0.05,
        },
        "values": ordered_values,
        "reference_value": reference_value,
        "finest_comparison": finest_comparison,
        "relative_peak_wavelength_change": relative_peak_shift,
        "relative_peak_amplitude_change": relative_peak_amplitude_change,
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
        accumulated_reports = _accumulated_validation_reports(validation_root, reports)
        comparison_summary = _comparison_summary_markdown(config, accumulated_reports)
        _write_text(workspace.directory / "comparison_summary.md", comparison_summary)
        _write_text(validation_root / "comparison_summary.md", comparison_summary)
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
