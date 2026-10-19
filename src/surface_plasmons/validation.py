"""Validation against the supplied electrostatic notes, with no model hierarchy."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import json

import numpy as np

from .database import find_project_root
from .notes_theory import FORMULAS, field_matrix, pair_couplings, polarizability, solve
from .particle_line import particle_line_positions
from .project import PRECISION_MODES, geometry_parameters, normalise_project_config
from .results import ExperimentWorkspace, config_hash, safe_simulation_name


class ValidationError(RuntimeError):
    """The selected data cannot support the requested validation."""


def _write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise ValidationError(f"Could not read {path}: {exc}") from exc
    if not rows:
        raise ValidationError(f"No data rows in {path}.")
    return rows


def resolve_experiment(value: str | Path, results_root=None) -> Path:
    supplied = Path(value).expanduser()
    root = Path(results_root) if results_root is not None else find_project_root() / "data/results"
    directory = supplied.resolve() if supplied.is_dir() else root.resolve() / safe_simulation_name(str(value))
    if not (directory / "config_resolved.json").is_file():
        raise ValidationError(f"No saved project configuration in {directory}. Run scripts/run_single.py first.")
    return directory


def _positions(config):
    particles = particle_line_positions({**geometry_parameters(config, include_particles=True), "center": (0, 0, 0)})
    return np.asarray([(p.actual_x_um, p.actual_y_um, p.actual_z_um) for p in particles])


def _profile(config):
    return {**PRECISION_MODES[config["precision"]], **config["numerics"]}


def _configuration_check(config, output):
    positions = _positions(config)
    radius = float(config["geometry"]["particle_radius_um"])
    resolution = int(_profile(config)["broadband_resolution"])
    report = {"status": "passed", "particle_count": len(positions),
              "resolution_pixels_per_um": resolution, "cells_across_diameter": 2*radius*resolution,
              "precision_label": config["precision"],
              "resolution_from_explicit_override": "broadband_resolution" in config["numerics"],
              "note": "Configuration validity does not establish numerical convergence."}
    _write_json(output / "configuration_validation.json", report)
    return report


def _epsilon(rows, role, wavelength):
    matches = [r for r in rows if r["material_role"] == role]
    exact = [r for r in matches if np.isclose(float(r["wavelength_nm"]), wavelength, rtol=1e-8, atol=1e-7)]
    if len(exact) != 1:
        raise ValidationError(f"Need one matching {role} permittivity at {wavelength:g} nm.")
    value = complex(float(exact[0]["epsilon_real"]), float(exact[0]["epsilon_imag"]))
    if not np.isfinite(value):
        raise ValidationError("Nonfinite dielectric function.")
    return value


def _vectors(rows, prefix):
    try:
        result = np.asarray([[complex(float(r[f"{prefix}{c}_real"]), float(r[f"{prefix}{c}_imag"]))
                              for c in ("Ex", "Ey", "Ez")] for r in rows])
    except (KeyError, ValueError) as exc:
        raise ValidationError("Complex total and particle-free reference fields are required. Rerun the simulation.") from exc
    if not np.all(np.isfinite(result)):
        raise ValidationError("Nonfinite complex fields in saved data.")
    return result


def _nrmse(actual, expected):
    scale = np.linalg.norm(expected)
    return None if scale <= 1e-20 else float(np.linalg.norm(actual - expected) / scale)


def _peak(wavelengths, values):
    wavelengths, values = np.asarray(wavelengths), np.asarray(values)
    index = int(np.argmax(values))
    if len(values) < 5 or index in (0, len(values)-1) or np.max(values) - np.min(values) <= 1e-12 * max(np.max(values), 1e-30):
        return {"status": "unresolved_or_band_edge", "wavelength_nm": float(wavelengths[index])}
    spacing = max(wavelengths[index] - wavelengths[index-1], wavelengths[index+1] - wavelengths[index])
    return {"status": "resolved_sampled_peak", "wavelength_nm": float(wavelengths[index]),
            "sampling_uncertainty_nm": float(spacing / 2)}


def _validation_region(config):
    region = config["measurement"].get("validation_region", "all_exterior")
    if region not in {"all_exterior", "electrostatic_near_zone"}:
        raise ValidationError("measurement.validation_region must be all_exterior or electrostatic_near_zone.")
    return region


def _maximum_host_k(dielectric, wavelengths):
    return max(float(abs(2*np.pi*np.sqrt(_epsilon(dielectric, "background", wave))/(wave/1000))) for wave in wavelengths)


def _applicability_details(spectra, reference_error, tolerance):
    values = {"particle_size_ka": max(r["ka"] for r in spectra),
              "probe_distance_kr": max(r["maximum_probe_to_particle_kr"] for r in spectra),
              "particle_separation_kd": max(r["maximum_particle_separation_kr"] for r in spectra),
              "reference_plane_wave_relative_error": reference_error}
    checks = {name: {"value": value, "limit": tolerance if name.startswith("reference_") else .3,
                     "status": "passed" if value <= (tolerance if name.startswith("reference_") else .3) else "failed"}
              for name, value in values.items()}
    return checks, [name for name, check in checks.items() if check["status"] == "failed"]


def _theory_check(config, experiment, output, *, tolerance=0.05, peak_tolerance=0.02):
    geometry = config["geometry"]
    if geometry["particle_shape"] != "sphere":
        report = {"status": "skipped", "reason": "The notes specify sphere polarizability; this geometry is not spherical."}
        _write_json(output / "notes_validation.json", report)
        return report
    if config["measurement"]["normalization_mode"] != "global_complex_reference":
        report = {"status": "skipped", "reason": "Use global_complex_reference to preserve common spatial amplitude and phase."}
        _write_json(output / "notes_validation.json", report)
        return report
    scan = experiment / "frequency_scan"
    rows = _read_csv(scan / "field_strength_vs_position_and_frequency.csv")
    dielectric = _read_csv(scan / "dielectric_function_vs_frequency.csv")
    wavelengths = sorted({float(r["wavelength_nm"]) for r in rows})
    # Check optical, rather than material-name, equality over the entire band.
    for wavelength in wavelengths:
        em = _epsilon(dielectric, "background", wavelength)
        es = _epsilon(dielectric, "substrate", wavelength)
        if not np.isclose(em, es, rtol=1e-8, atol=1e-10) or em.real <= 0 or abs(em.imag) > 1e-10:
            report = {"status": "skipped", "reason": "The notes require a homogeneous, lossless positive dielectric host. No substrate correction is added."}
            _write_json(output / "notes_validation.json", report)
            return report
    region = _validation_region(config)
    maximum_k = _maximum_host_k(dielectric, wavelengths)
    positions = _positions(config)
    radius = float(geometry["particle_radius_um"])
    resolution = int(_profile(config)["broadband_resolution"])
    # Reject interior and interface-interpolation points before any field comparison.
    minimum_distance = radius + 2 / resolution
    pol = np.eye(3)[{"ex": 0, "ey": 1, "ez": 2}[config["source"]["polarisation"].lower()]]
    axis = {"x": 0, "y": 1, "z": 2}[config["source"]["propagation_axis"]]
    direction = np.eye(3)[axis] * config["source"]["propagation_direction"]
    spectra, dipoles, fields, modes = [], [], [], []
    all_actual, all_theory = [], []
    max_residual, reference_errors, near_parameters = 0.0, [], []
    fit_errors = []
    for wavelength in wavelengths:
        selected = [r for r in rows if np.isclose(float(r["wavelength_nm"]), wavelength, rtol=1e-10, atol=1e-9)]
        points = np.asarray([[float(r[f"{c}_um"]) for c in "xyz"] for r in selected])
        distances = np.linalg.norm(points[:, None, :] - positions[None, :, :], axis=2)
        keep = np.all(distances >= minimum_distance, axis=1)
        if region == "electrostatic_near_zone":
            keep &= np.all(maximum_k * distances <= .3 + 1e-12, axis=1)
        if np.count_nonzero(keep) < max(4, len(positions) + 1):
            if region == "electrostatic_near_zone":
                report = {"status": "skipped", "agreement_status": "not_evaluated",
                          "applicability_status": "insufficient_near_field_samples", "validation_region": region,
                          "reason": "Too few exterior points satisfy both the two-cell surface buffer and kr <= 0.3 over the entire band.",
                          "maximum_near_zone_distance_um": .3/maximum_k,
                          "minimum_surface_safe_distance_um": minimum_distance,
                          "note": "Use a smaller physical benchmark when the safe distance exceeds the near-zone radius. A higher resolution alone cannot fix an oversized particle."}
                _write_json(output / "notes_validation.json", report)
                return report
            raise ValidationError("Too few exterior field samples after the two-grid-cell surface buffer. Increase probe_margin_um or resolution.")
        total = _vectors(selected, "normalised_")[keep]
        reference = _vectors(selected, "normalised_reference_")[keep]
        points = points[keep]
        em = _epsilon(dielectric, "background", wavelength)
        ep = _epsilon(dielectric, "particle", wavelength)
        alpha = polarizability(radius, ep, em)
        k = 2 * np.pi * np.sqrt(em) / (wavelength / 1000)
        # Transport measured reference to one origin; this is illumination only.
        # Particle-particle and dipole-probe propagation remain strictly static.
        phase_at_points = np.exp(1j * k * (points @ direction))
        amplitudes = (reference @ pol) / phase_at_points
        amplitude = complex(np.median(amplitudes.real), np.median(amplitudes.imag))
        if abs(amplitude) <= 1e-20:
            raise ValidationError("Driven reference component is zero.")
        expected_reference = amplitude * phase_at_points[:, None] * pol
        reference_errors.append(_nrmse(reference, expected_reference))
        incident = amplitude * np.exp(1j * k * (positions @ direction))[:, None] * pol
        predicted, residual = solve(positions, alpha, incident, em)
        max_residual = max(max_residual, residual)
        matrix = field_matrix(points, positions, em)
        predicted_field = (matrix @ predicted.reshape(-1)).reshape(-1, 3)
        measured = total - reference
        # Inverse fit is a diagnostic, never used to rescale the predicted field.
        fitted, _, rank, singular = np.linalg.lstsq(matrix, measured.reshape(-1), rcond=None)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        fit_ok = rank == matrix.shape[1] and condition is not None and condition < 1e8
        fitted = fitted.reshape(-1, 3)
        fit_residual = _nrmse((matrix @ fitted.reshape(-1)).reshape(-1, 3), measured)
        fit_errors.append(fit_residual)
        field_error = _nrmse(measured, predicted_field)
        projected = predicted @ pol
        fitted_projected = fitted @ pol
        for j in range(len(positions)):
            q, f = projected[j], fitted_projected[j]
            dipoles.append({"wavelength_nm": wavelength, "particle_index": j,
                           "theory_q_parallel_real_um3": float(q.real), "theory_q_parallel_imag_um3": float(q.imag),
                           "fitted_q_parallel_real_um3": float(f.real) if fit_ok else "",
                           "fitted_q_parallel_imag_um3": float(f.imag) if fit_ok else "",
                           "fit_identifiable": bool(fit_ok), "fit_condition_number": condition,
                           "fit_relative_residual": fit_residual,
                           "relative_phase_to_first_rad": float(np.angle(f / fitted_projected[0])) if fit_ok and abs(fitted_projected[0]) > 1e-20 else "",
                           "amplitude_ratio_to_first": float(abs(f / fitted_projected[0])) if fit_ok and abs(fitted_projected[0]) > 1e-20 else ""})
        for i, point in enumerate(points):
            row = {"wavelength_nm": wavelength, **{f"{c}_um": float(point[j]) for j, c in enumerate("xyz")},
                   "meep_scattered_amplitude": float(np.linalg.norm(measured[i])),
                   "theory_scattered_amplitude": float(np.linalg.norm(predicted_field[i]))}
            for j, c in enumerate(("Ex", "Ey", "Ez")):
                for label, array in (("meep", measured), ("theory", predicted_field)):
                    row[f"{label}_{c}_real"] = float(array[i, j].real)
                    row[f"{label}_{c}_imag"] = float(array[i, j].imag)
            fields.append(row)
        size_parameter = float(abs(k) * radius)
        probe_kr = float(abs(k) * np.max(np.linalg.norm(points[:,None,:] - positions[None,:,:], axis=2)))
        separation_kr = float(abs(k) * np.max(np.linalg.norm(positions[:,None,:] - positions[None,:,:], axis=2)))
        near_parameters.append(max(size_parameter, probe_kr, separation_kr))
        spectra.append({"wavelength_nm": wavelength, "epsilon_particle_real": float(ep.real),
                        "epsilon_particle_imag": float(ep.imag), "epsilon_host_real": float(em.real),
                        "radius_um": radius, "alpha0_real_um3": float(alpha.real), "alpha0_imag_um3": float(alpha.imag),
                        "alpha0_magnitude_um3": float(abs(alpha)),
                        "theory_rms_dipole_um3": float(np.sqrt(np.mean(np.abs(projected)**2))),
                        "fitted_rms_dipole_um3": float(np.sqrt(np.mean(np.abs(fitted_projected)**2))) if fit_ok else "",
                        "meep_rms_scattered_field": float(np.sqrt(np.mean(np.abs(measured)**2))),
                        "theory_rms_scattered_field": float(np.sqrt(np.mean(np.abs(predicted_field)**2))),
                        "complex_field_relative_l2_error": field_error,
                        "dipole_relative_l2_error": _nrmse(fitted, predicted) if fit_ok else "",
                        "ka": size_parameter, "maximum_probe_to_particle_kr": probe_kr,
                        "maximum_particle_separation_kr": separation_kr,
                        "equation_relative_residual": residual})
        all_actual.append(measured.reshape(-1))
        all_theory.append(predicted_field.reshape(-1))
        if len(positions) == 2:
            distance = float(np.linalg.norm(positions[1] - positions[0]))
            for orientation, coupling in pair_couplings(distance, em).items():
                for mode, sign in (("symmetric", 1), ("antisymmetric", -1)):
                    denominator = 1 - sign * alpha * coupling
                    susceptibility = abs(alpha / denominator)
                    modes.append({"wavelength_nm": wavelength, "orientation": orientation, "mode": mode,
                                  "surface_gap_um": distance - 2 * radius, "centre_distance_um": distance,
                                  "C_real_per_um3": float(coupling.real),
                                  "real_alpha_inverse_minus_eigenvalue": float((1/alpha - sign*coupling).real) if alpha != 0 else "",
                                  "modal_response_um3": float(susceptibility),
                                  "meep_mode_tested": bool(mode == "symmetric" and np.ptp(positions @ direction) < 1e-10
                                                           and (orientation == "longitudinal") == (abs(pol @ ((positions[1]-positions[0])/distance)) > 0.999))})
    _write_csv(output / "notes_spectrum.csv", spectra)
    _write_csv(output / "notes_dipoles.csv", dipoles)
    _write_csv(output / "notes_fields.csv", fields)
    _write_csv(output / "notes_pair_modes.csv", modes)
    mode_summary = []
    for orientation in ("longitudinal", "transverse"):
        for mode in ("symmetric", "antisymmetric"):
            selected_modes = [r for r in modes if r["orientation"] == orientation and r["mode"] == mode]
            if not selected_modes:
                continue
            mode_peak = _peak(wavelengths, [r["modal_response_um3"] for r in selected_modes])
            crossings = []
            for left, right in zip(selected_modes[:-1], selected_modes[1:]):
                f0, f1 = left["real_alpha_inverse_minus_eigenvalue"], right["real_alpha_inverse_minus_eigenvalue"]
                if isinstance(f0, (int, float)) and isinstance(f1, (int, float)):
                    if f0 == 0:
                        crossings.append(left["wavelength_nm"])
                    elif f0*f1 < 0:
                        crossings.append(left["wavelength_nm"] + (right["wavelength_nm"]-left["wavelength_nm"])*(-f0)/(f1-f0))
            mode_summary.append({"orientation": orientation, "mode": mode, "sampled_susceptibility_peak": mode_peak,
                                 "real_inverse_alpha_crossings_nm": crossings,
                                 "meep_mode_tested": selected_modes[0]["meep_mode_tested"],
                                 "note": "With absorption, the real-part crossing is a resonance indicator, not an exact zero of the complex denominator."})
    if mode_summary:
        _write_json(output / "notes_pair_mode_peaks.json", mode_summary)
    measured_peak = _peak(wavelengths, [r["meep_rms_scattered_field"] for r in spectra])
    predicted_peak = _peak(wavelengths, [r["theory_rms_scattered_field"] for r in spectra])
    peak_resolved = all(p["status"] == "resolved_sampled_peak" for p in (measured_peak, predicted_peak))
    peak_sampling_sufficient = peak_resolved and all(p["sampling_uncertainty_nm"] / p["wavelength_nm"] <= peak_tolerance for p in (measured_peak, predicted_peak))
    peak_error = abs(measured_peak["wavelength_nm"] - predicted_peak["wavelength_nm"]) / predicted_peak["wavelength_nm"] if peak_resolved else None
    actual_vector = np.concatenate(all_actual).reshape(-1, 3)
    theory_vector = np.concatenate(all_theory).reshape(-1, 3)
    error = _nrmse(actual_vector, theory_vector)
    amplitude_error = _nrmse(np.linalg.norm(actual_vector, axis=1), np.linalg.norm(theory_vector, axis=1))
    agreement = "passed" if error is not None and error <= tolerance and peak_error is not None and peak_error <= peak_tolerance and peak_sampling_sufficient else "failed"
    if not peak_sampling_sufficient and error is not None and error <= tolerance:
        agreement = "inconclusive_peak_not_resolved"
    reference_error = max(e for e in reference_errors if e is not None)
    applicability_checks, applicability_reasons = _applicability_details(spectra, reference_error, tolerance)
    applicability = "passed" if not applicability_reasons else "outside_applicability_checks"
    agreement_checks = {
        "complex_scattered_field": {"value": error, "limit": tolerance,
                                   "status": "passed" if error is not None and error <= tolerance else "failed"},
        "peak_wavelength": {"value": peak_error, "limit": peak_tolerance,
                            "status": "passed" if peak_error is not None and peak_error <= peak_tolerance else "failed_or_unresolved"},
        "peak_sampling": {"status": "passed" if peak_sampling_sufficient else "unresolved"},
    }
    failure_reasons = [name for name, check in agreement_checks.items() if check["status"] != "passed"]
    warnings = []
    if 2*radius*resolution < 12:
        warnings.append("The diameter has fewer than 12 grid cells. This is a coarse initial mesh; convergence is required and may need finer settings.")
    if applicability_reasons:
        warnings.append("An electrostatic model disagreement outside its applicability screen does not by itself show an FDTD solver error.")
    report = {"status": "completed", "model": "notes_electrostatic_only", "formulas": FORMULAS,
              "agreement_status": agreement, "applicability_status": applicability,
              "agreement_checks": agreement_checks, "failure_reasons": failure_reasons,
              "applicability_checks": applicability_checks, "applicability_failure_reasons": applicability_reasons,
              "validation_region": region, "cells_across_particle_diameter": 2*radius*resolution,
              "benchmark_recovery": {
                  "maximum_near_zone_radius_um": .3/maximum_k,
                  "minimum_surface_safe_probe_radius_um": minimum_distance,
                  "sphere_fits_near_zone": bool(radius < .3/maximum_k),
                  "surface_safe_near_zone_exists_at_saved_resolution": bool(minimum_distance < .3/maximum_k),
                  "reanalysis_changes_original_radius_or_resolution": False,
                  "action": "Run a new smaller-sphere configuration for a strict benchmark; retain these data as a model-disagreement case."
                            if radius >= .3/maximum_k else "Check numerical convergence and whether the saved probes cover the near zone.",
              },
              "warnings": warnings,
              "overall_status": "provisional_agreement_requires_convergence" if agreement == "passed" and applicability == "passed" else "not_validated",
              "complex_scattered_field_relative_l2_error": error,
              "amplitude_only_diagnostic_relative_l2_error": amplitude_error,
              "maximum_equation_relative_residual": max_residual,
              "maximum_reference_plane_wave_relative_error": reference_error,
              "maximum_ka_or_kr": max(near_parameters), "electrostatic_screening_limit": 0.3,
              "meep_peak": measured_peak, "theory_peak": predicted_peak, "relative_peak_wavelength_error": peak_error,
              "peak_sampling_sufficient": bool(peak_sampling_sufficient),
              "criteria": {"field_relative_l2_tolerance": tolerance, "peak_wavelength_relative_tolerance": peak_tolerance},
              "independent_curve_rescaling": False, "surface_exclusion_um": minimum_distance - radius,
              "dipole_extraction": "Least-squares inversion of exterior scattered fields, conditional on the electrostatic model; not a volume polarization integral or independent validation.",
              "maximum_dipole_fit_relative_residual": max((e for e in fit_errors if e is not None), default=None),
              "mode_scope": "Both pair eigenmodes are predictions. Uniform identical-pair illumination tests the symmetric mode only; antisymmetric response is not claimed as FDTD validated.",
              "unit_definition": "q=p_SI/(epsilon_0*E_ref*1e-18); alpha0=alpha_SI/(epsilon_0*1e-18), in um^3. E_ref is one complex scale per frequency.",
              "numerical_convergence_status": "not_established_by_theory_check"}
    _write_json(output / "notes_validation.json", report)
    _plots(output, spectra, fields, measured_peak["wavelength_nm"])
    return report


def _plots(output, spectra, fields, peak_wavelength):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    wave = [r["wavelength_nm"] for r in spectra]
    axes[0].plot(wave, [r["meep_rms_scattered_field"] for r in spectra], label="MEEP")
    axes[0].plot(wave, [r["theory_rms_scattered_field"] for r in spectra], label="Notes electrostatic", linestyle="--")
    axes[0].set(xlabel="Vacuum wavelength (nm)", ylabel="RMS scattered E / Eref")
    profile = sorted([r for r in fields if r["wavelength_nm"] == peak_wavelength], key=lambda r:r["x_um"])
    axes[1].plot([r["x_um"] for r in profile], [r["meep_scattered_amplitude"] for r in profile], ".", label="MEEP")
    axes[1].plot([r["x_um"] for r in profile], [r["theory_scattered_amplitude"] for r in profile], label="Notes electrostatic")
    axes[1].set(xlabel="x (um)", ylabel="Scattered |E| / |Eref|", title=f"Exterior profile at {peak_wavelength:.1f} nm")
    for axis in axes:
        axis.legend(); axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "notes_comparison.png", dpi=180)
    plt.close(figure)


def _sample_response(actual, reference, config, points):
    from .workflow import _normalised_fields, _coordinate_grid
    normalised = _normalised_fields(actual, reference, reference_component=config["source"]["polarisation"], mode="global_complex_reference")
    x, y, z = _coordinate_grid(actual)
    if np.ptp(y) > 1e-10 or np.ptp(z) > 1e-10:
        raise ValidationError("Convergence requires the broadband x-line monitor.")
    order = np.argsort(x)
    if min(points) < min(x)-1e-10 or max(points) > max(x)+1e-10:
        raise ValidationError("Convergence monitors do not cover common physical sample points.")
    components = []
    for component in ("Ex", "Ey", "Ez"):
        field = normalised[f"normalised_scattered_{component}"].reshape(len(actual.frequencies), -1)
        sampled = np.asarray([np.interp(points, x[order], row[order].real) + 1j*np.interp(points, x[order], row[order].imag) for row in field])
        components.append(sampled)
    return np.stack(components, axis=-1)


def _convergence_metrics(profile, reference):
    value, wave, field = profile
    _, ref_wave, ref_field = reference
    if wave.shape != ref_wave.shape or not np.allclose(wave, ref_wave):
        raise ValidationError("Convergence cases have different frequency grids.")
    response = np.sqrt(np.mean(np.abs(field)**2, axis=(1, 2)))
    ref_response = np.sqrt(np.mean(np.abs(ref_field)**2, axis=(1, 2)))
    peak, refpeak = _peak(wave, response), _peak(wave, ref_response)
    resolved = peak["status"] == refpeak["status"] == "resolved_sampled_peak"
    sampling_ok = resolved and all(p["sampling_uncertainty_nm"] / p["wavelength_nm"] <= .02 for p in (peak, refpeak))
    return {"value": value, "complex_field_relative_l2_change": _nrmse(field, ref_field),
            "relative_peak_wavelength_change": abs(peak["wavelength_nm"]-refpeak["wavelength_nm"])/refpeak["wavelength_nm"],
            "peak_resolved": bool(sampling_ok),
            "relative_peak_amplitude_change": float(abs(np.max(response)-np.max(ref_response))/max(np.max(ref_response), 1e-30))}


def _convergence_passed(metrics):
    error = metrics["complex_field_relative_l2_change"]
    return (error is not None and error <= .05
            and metrics["relative_peak_wavelength_change"] <= .02
            and metrics["relative_peak_amplitude_change"] <= .05 and metrics["peak_resolved"])


def _saved_baseline_samples(raw, points):
    """Reuse saved baseline fields; no duplicate MEEP calculation is needed."""
    wavelengths = np.asarray(sorted({float(row["wavelength_nm"]) for row in raw}))
    responses = []
    for wavelength in wavelengths:
        selected = sorted([row for row in raw if float(row["wavelength_nm"]) == wavelength], key=lambda row:float(row["x_um"]))
        x = np.asarray([float(row["x_um"]) for row in selected])
        scattered = _vectors(selected, "normalised_") - _vectors(selected, "normalised_reference_")
        responses.append(np.stack([np.interp(points, x, scattered[:,j].real)
                                   + 1j*np.interp(points, x, scattered[:,j].imag) for j in range(3)], axis=-1))
    return wavelengths, np.asarray(responses)


def _cached_samples(saved, marker, signature):
    """Reuse earlier samples when the requested safe points are a subset."""
    if not saved.is_file() or not marker.is_file():
        return None
    previous = json.loads(marker.read_text(encoding="utf-8"))
    if previous.get("config_hash") != signature["config_hash"]:
        return None
    old_points = np.asarray(previous.get("sample_x_um", []), dtype=float)
    points = np.asarray(signature["sample_x_um"], dtype=float)
    if not old_points.size:
        return None
    indices = np.argmin(np.abs(old_points[:, None] - points[None, :]), axis=0)
    if not np.allclose(old_points[indices], points, rtol=0, atol=1e-12):
        return None
    with np.load(saved, allow_pickle=False) as archive:
        wavelengths = np.array(archive["wavelengths"])
        fields = np.array(archive["fields"])
    if fields.ndim != 3 or fields.shape[1] != len(old_points):
        raise ValidationError(f"Invalid cached convergence samples: {saved}")
    return wavelengths, fields[:, indices, :]


def _convergence(config, experiment, output, kind, values, cache_enabled, *, adaptive_resolution=True):
    from .workflow import run_broadband_stage
    if config["measurement"]["normalization_mode"] != "global_complex_reference":
        raise ValidationError("Numerical field comparison requires global_complex_reference normalization.")
    values = sorted(set(float(v) for v in values))
    if len(values) < 2 or any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValidationError(f"{kind} needs at least two distinct positive values.")
    if kind == "resolution" and any(v != int(v) for v in values):
        raise ValidationError("Resolution values must be integers.")
    if kind == "time" and any(v < 1 for v in values):
        raise ValidationError("Time factors must be at least 1.")
    raw = _read_csv(experiment / "frequency_scan/field_strength_vs_position_and_frequency.csv")
    first_wave = float(raw[0]["wavelength_nm"])
    original_points = np.unique([[float(r[f"{c}_um"]) for c in "xyz"] for r in raw if float(r["wavelength_nm"]) == first_wave], axis=0)
    if np.ptp(original_points[:,1]) > 1e-10 or np.ptp(original_points[:,2]) > 1e-10:
        raise ValidationError("Convergence requires an x-line monitor.")
    base_res = float(_profile(config)["broadband_resolution"])
    coarse_res = min(values) if kind == "resolution" else base_res
    distance = np.linalg.norm(original_points[:,None,:] - _positions(config)[None,:,:], axis=2)
    keep = np.all(distance >= float(config["geometry"]["particle_radius_um"]) + 2/coarse_res, axis=1)
    # Monitor endpoints can move when MEEP rounds different grids to cells.
    # Keep the same interior coordinates for every run; never extrapolate.
    edge_buffer_um = 2.0 / coarse_res
    x_min = float(np.min(original_points[:, 0])) + edge_buffer_um
    x_max = float(np.max(original_points[:, 0])) - edge_buffer_um
    keep &= (original_points[:, 0] >= x_min - 1e-12)
    keep &= (original_points[:, 0] <= x_max + 1e-12)
    if _validation_region(config) == "electrostatic_near_zone":
        dielectric = _read_csv(experiment / "frequency_scan/dielectric_function_vs_frequency.csv")
        maximum_k = _maximum_host_k(dielectric, sorted({float(r["wavelength_nm"]) for r in raw}))
        keep &= np.all(maximum_k * distance <= .3 + 1e-12, axis=1)
    points = original_points[keep, 0]
    if len(points) < 4:
        raise ValidationError("Too few common exterior points for convergence; extend the monitor.")
    profiles, sources = [], []
    for value in values:
        baseline_value = base_res if kind == "resolution" else (float(config["numerics"]["padding_um"]) if kind == "domain" else 1.)
        if value == baseline_value:
            wavelengths, response = _saved_baseline_samples(raw, points)
            sources.append("saved_baseline")
        else:
            candidate = deepcopy(config)
            if kind == "resolution": candidate["numerics"]["broadband_resolution"] = int(value)
            elif kind == "domain": candidate["numerics"]["padding_um"] = value
            else:
                profile = _profile(config)
                candidate["numerics"].update(minimum_run_time=float(profile["minimum_run_time"])*value,
                                              maximum_run_time=float(profile["maximum_run_time"])*value,
                                              dft_decay_tolerance=float(profile["dft_decay_tolerance"])/value)
            run_identity = config_hash({"config": candidate, "validation_directory": str(output), "kind": kind, "value": value})[:10]
            candidate["simulation_name"] = f"{safe_simulation_name(config['simulation_name'])[:40]}_{kind}_{value:g}_{run_identity}"
            digest = config_hash(candidate)
            runroot = output / "numerical_runs" / kind
            target = runroot / safe_simulation_name(candidate["simulation_name"])
            saved = target / "comparison_samples.npz"
            marker = target / "comparison_signature.json"
            signature = {"config_hash": digest, "sample_x_um": points.tolist()}
            cached = _cached_samples(saved, marker, signature)
            if cached is not None:
                wavelengths, response = cached
                sources.append("saved_convergence_samples")
            else:
                workspace = ExperimentWorkspace.create(candidate["simulation_name"], candidate, results_root=runroot, resume=target.exists())
                _, actual, reference, _ = run_broadband_stage(candidate, workspace, cache_enabled=cache_enabled)
                response = _sample_response(actual, reference, candidate, points)
                wavelengths = actual.wavelengths_um * 1000
                np.savez_compressed(saved, wavelengths=wavelengths, fields=response)
                _write_json(marker, signature)
                workspace.set_status("completed", message="Convergence samples saved")
                sources.append("new_simulation")
        order = np.argsort(wavelengths)
        profiles.append((value, wavelengths[order], response[order]))
        if kind == "resolution" and adaptive_resolution and len(profiles) >= 2:
            if _convergence_passed(_convergence_metrics(profiles[-2], profiles[-1])):
                break
    records = [_convergence_metrics(profile, profiles[-1]) for profile in profiles]
    for record, source in zip(records, sources):
        record["source"] = source
    last = records[-2]
    passed = _convergence_passed(last)
    report = {"status": "completed", "convergence_status": "passed" if passed else "failed_or_unresolved",
              "kind": kind, "planned_values": values, "values": [p[0] for p in profiles],
              "stopped_early": len(profiles) < len(values), "finest_comparison": last,
              "recommended_resolution": int(profiles[-1][0]) if kind == "resolution" and passed else None,
              "criteria": {"complex_field_change": 0.05, "peak_wavelength_change": 0.02, "peak_amplitude_change": 0.05},
              "sample_count": len(points), "common_physical_sample_points": True,
              "monitor_edge_buffer_um": edge_buffer_um, "sample_x_um": points.tolist()}
    _write_csv(output / f"{kind}_convergence.csv", records)
    _write_json(output / f"{kind}_convergence_summary.json", report)
    return report


def _baseline_analysis_signature(config):
    """Compare effective broadband settings, including their analysis region."""
    from .project import make_simulation_config
    resolved = normalise_project_config(config)
    simulation = make_simulation_config(resolved, experiment_directory=Path.cwd()/"signature_only", stage_name="frequency_scan/particle")
    for key in ("run_id", "run_name", "tags", "record_database"):
        simulation.pop(key, None)
    for key in ("experiment_directory", "stage_name"):
        simulation["output"].pop(key, None)
    return config_hash({"simulation": simulation,
                        "normalization_mode": resolved["measurement"]["normalization_mode"],
                        "validation_region": _validation_region(resolved)})


def _saved_convergence_reports(directory, config, exclude):
    """Read compatible completed checks; missing checks are not called unrun."""
    expected = _baseline_analysis_signature(config)
    reports = {}
    root = directory/"validation"
    markers = sorted(root.glob("*/validation_config.json"), key=lambda path:path.stat().st_mtime, reverse=True)
    for marker in markers:
        if marker.parent.resolve() == exclude.resolve():
            continue
        try:
            identity = json.loads(marker.read_text(encoding="utf-8"))
            baseline = identity.get("config")
            if not isinstance(baseline, dict) or _baseline_analysis_signature(baseline) != expected:
                continue
            summary = json.loads((marker.parent/"validation_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        for kind in ("resolution", "domain", "time"):
            report = summary.get(kind, {})
            if kind in reports or report.get("status") != "completed":
                continue
            if report.get("convergence_status") not in {"passed", "failed", "failed_or_unresolved"}:
                continue
            criteria = report.get("criteria", {})
            required = {"complex_field_change": .05, "peak_wavelength_change": .02, "peak_amplitude_change": .05}
            if any(not isinstance(criteria.get(key), (float, int)) or not 0 < criteria[key] <= limit for key, limit in required.items()):
                continue
            reports[kind] = {**report, "loaded_from_saved_results": True,
                             "source_validation_folder": marker.parent.name}
    return reports


def run_validation(experiment: str | Path, *, validation_name="notes_validation_v1",
                   checks: Sequence[str] = ("configuration", "theory"), results_root=None,
                   resolutions=None, paddings_um=None, time_factors=None, cache_enabled=True,
                   tolerance=0.05, peak_tolerance=0.02, adaptive_resolution=True) -> Path:
    directory = resolve_experiment(experiment, results_root)
    config = normalise_project_config(json.loads((directory / "config_resolved.json").read_text(encoding="utf-8")))
    selected = {str(v).strip().lower() for v in checks}
    allowed = {"configuration", "theory", "resolution", "domain", "time"}
    if "all" in selected: selected = (selected - {"all"}) | allowed
    if not selected or selected - allowed:
        raise ValidationError(f"Choose checks from {sorted(allowed)}. The old hierarchy comparison has been removed.")
    if not (0 < tolerance < 1 and 0 < peak_tolerance < 1):
        raise ValidationError("Tolerances must be fractions between 0 and 1.")
    profile = _profile(config)
    resolutions = list(resolutions) if resolutions is not None else [int(profile["broadband_resolution"])+v for v in (0, 40, 80)]
    paddings_um = list(paddings_um) if paddings_um is not None else [config["numerics"]["padding_um"]+v for v in (0, 0.1)]
    time_factors = list(time_factors) if time_factors is not None else [1, 2]
    identity = {"schema": "electrostatic_notes_v2", "config": config, "checks": sorted(selected),
                "resolutions": resolutions if "resolution" in selected else None,
                "paddings": paddings_um if "domain" in selected else None,
                "time_factors": time_factors if "time" in selected else None,
                "tolerance": tolerance, "peak_tolerance": peak_tolerance,
                "adaptive_resolution": bool(adaptive_resolution) if "resolution" in selected else None}
    output = directory / "validation" / safe_simulation_name(validation_name)
    marker = output / "validation_config.json"
    if output.exists() and (not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != identity):
        raise ValidationError("This validation name belongs to different settings. Use a new --name; identical reruns reuse the same folder.")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(marker, identity)
    reports = _saved_convergence_reports(directory, config, output)
    if "configuration" in selected: reports["configuration"] = _configuration_check(config, output)
    if "theory" in selected: reports["theory"] = _theory_check(config, directory, output, tolerance=tolerance, peak_tolerance=peak_tolerance)
    for kind, values in (("resolution", resolutions), ("domain", paddings_um), ("time", time_factors)):
        if kind in selected: reports[kind] = _convergence(config, directory, output, kind, values, cache_enabled, adaptive_resolution=adaptive_resolution)
    reports["numerical_check_availability"] = {
        kind: {"status": "computed_in_this_analysis" if kind in selected else
                         ("loaded_compatible_saved_check" if kind in reports else "not_requested_no_compatible_saved_report_found"),
               "source_validation_folder": reports.get(kind, {}).get("source_validation_folder")}
        for kind in ("resolution", "domain", "time")}
    theory = reports.get("theory", {})
    numerical_pass = all(reports.get(k, {}).get("convergence_status") == "passed" for k in ("resolution", "domain", "time"))
    model_pass = theory.get("agreement_status") == "passed" and theory.get("applicability_status") == "passed"
    reports["overall"] = {"status": "validated" if model_pass and numerical_pass else "not_validated",
                          "model_and_applicability_passed": model_pass, "all_numerical_checks_passed": numerical_pass}
    _write_json(output / "validation_summary.json", reports)
    text = ["# Electrostatic notes validation", "", f"Overall: {reports['overall']['status']}", f"Model agreement: {theory.get('agreement_status', theory.get('status', 'not requested'))}",
            f"Applicability: {theory.get('applicability_status', theory.get('reason', 'not evaluated'))}",
            f"Complex scattered-field relative L2 error: {theory.get('complex_scattered_field_relative_l2_error', 'unavailable')}",
            f"Amplitude-only diagnostic relative L2 error: {theory.get('amplitude_only_diagnostic_relative_l2_error', 'unavailable')}",
            "", "Individual model checks:"]
    for key, check in theory.get("agreement_checks", {}).items():
        text.append(f"- {key}: {check['status']} (value={check.get('value', 'n/a')}, limit={check.get('limit', 'n/a')})")
    text.append("\nApplicability checks:")
    for key, check in theory.get("applicability_checks", {}).items():
        text.append(f"- {key}: {check['status']} (value={check['value']:.6g}, limit={check['limit']:.6g})")
    text.append("\nConvergence:")
    for kind in ("resolution", "domain", "time"):
        numerical = reports.get(kind, {})
        status = numerical.get("convergence_status", "not requested in this reanalysis; no compatible saved report found")
        source = numerical.get("source_validation_folder")
        text.append(f"- {kind}: {status}" + (f" (saved report: {source})" if source else ""))
    recovery = theory.get("benchmark_recovery", {})
    if recovery:
        text += ["", "Benchmark recovery:",
                 f"- Maximum near-zone radius: {1000*recovery['maximum_near_zone_radius_um']:.3f} nm",
                 f"- Minimum surface-safe probe radius at saved resolution: {1000*recovery['minimum_surface_safe_probe_radius_um']:.3f} nm",
                 f"- {recovery['action']}",
                 "- Reanalysis retains the saved particle radius, resolution and fields."]
    text += ["", "A theory match alone is provisional. Approve quantitative results only after all three numerical checks pass.",
             "The inverse dipole fit is conditional on the electrostatic model. Antisymmetric pair modes are theory predictions unless separately excited."]
    (output / "comparison_summary.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    return output
