import csv

import pytest

from surface_plasmons.project import normalise_project_config
from surface_plasmons.validation import (
    _comparison_summary_markdown,
    _hierarchy_check,
    _theory_check,
)


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_equation2_validation_keeps_common_scale_and_labels_axes(tmp_path) -> None:
    experiment = tmp_path / "experiment"
    scan = experiment / "frequency_scan"
    output = tmp_path / "validation"
    _write_csv(
        scan / "field_response_summary.csv",
        ["wavelength_nm", "maximum_field_amplitude_enhancement", "maximum_field_enhancement"],
        [{"wavelength_nm": 600.0, "maximum_field_amplitude_enhancement": 1.0, "maximum_field_enhancement": 1.0}],
    )
    columns = [
        "wavelength_nm", "x_um", "y_um", "z_um",
        "normalised_Ex_real", "normalised_Ex_imag",
        "normalised_Ey_real", "normalised_Ey_imag",
        "normalised_Ez_real", "normalised_Ez_imag",
    ]
    rows = []
    for x in (-0.12, -0.06, 0.0, 0.06, 0.12):
        rows.append(
            {
                "wavelength_nm": 600.0,
                "x_um": x,
                "y_um": 0.10,
                "z_um": 0.0,
                "normalised_Ex_real": 1.0,
                "normalised_Ex_imag": 0.0,
                "normalised_Ey_real": 0.0,
                "normalised_Ey_imag": 0.0,
                "normalised_Ez_real": 0.0,
                "normalised_Ez_imag": 0.0,
            }
        )
    _write_csv(scan / "field_strength_vs_position_and_frequency.csv", columns, rows)
    dielectric_rows = [
        {"material_role": "particle", "wavelength_nm": 600.0, "epsilon_real": -10.0, "epsilon_imag": 1.0},
        {"material_role": "background", "wavelength_nm": 600.0, "epsilon_real": 1.0, "epsilon_imag": 0.0},
        {"material_role": "substrate", "wavelength_nm": 600.0, "epsilon_real": 1.0, "epsilon_imag": 0.0},
    ]
    _write_csv(
        scan / "dielectric_function_vs_frequency.csv",
        ["material_role", "wavelength_nm", "epsilon_real", "epsilon_imag"],
        dielectric_rows,
    )
    config = normalise_project_config(
        {
            "simulation_name": "equation2-test",
            "geometry": {"particle_count": 1, "substrate_material": "air"},
        }
    )

    report = _theory_check(config, experiment, output)

    assert report["status"] == "completed"
    assert report["benchmark"] == "formal_equation2_homogeneous"
    assert report["benchmark_status"] == "passed_preconditions"
    assert report["execution_status"] == "completed"
    assert report["applicability_status"] == "passed"
    assert report["agreement_status"] in {
        "passed_quantitative",
        "passed_qualitative_only",
        "failed",
    }
    assert report["same_scale_total_field_comparison"]["spatial_sample_count"] == 5
    assert "same_scale_scattered_field_comparison" in report
    assert report["interaction_model"] == "paper_retarded"
    assert report["polarizability_model"] == "quasistatic"
    assert report["finite_size_corrections"] == (
        "disabled for the strict Section-2.1 baseline"
    )
    assert report["normalization"]["independent_curve_rescaling"] is False
    assert report["maximum_equation2_residual"] < 1e-12
    assert report["relative_equation2_residual"] < 1e-12
    assert (output / "equation2_field_amplitude_profile.png").is_file()
    assert (output / "equation2_scattered_field_amplitude_profile.png").is_file()
    assert not (output / "equation2_field_intensity_profile.png").exists()

    hierarchy = _hierarchy_check(config, experiment, output)

    assert hierarchy["status"] == "completed"
    assert hierarchy["model_order"][0] == "direct scalar 1-D CDA"
    assert hierarchy["model_order"][-1] == (
        "3-D MEEP FDTD for the complete finite structure"
    )
    assert hierarchy["maximum_scalar_vs_vector_relative_error"] < 1e-12
    assert (output / "model_hierarchy_comparison.png").is_file()
    assert (output / "one_dimensional_defect_response.png").is_file()
    assert (output / "fixed_probe_and_rms_spectrum_comparison.png").is_file()
    with (output / "model_hierarchy_spectrum.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        hierarchy_row = next(csv.DictReader(handle))
    assert float(hierarchy_row["particle_radius_um"]) == pytest.approx(
        config["geometry"]["particle_radius_um"]
    )
    assert float(hierarchy_row["epsilon_particle_real"]) == pytest.approx(-10.0)
    assert "alpha_equation1_real_um3" in hierarchy_row
    assert "alpha_radiation_reaction_real_um3" in hierarchy_row
    assert "alpha_mie_a1_real_um3" in hierarchy_row
    assert "mie_a1_real" in hierarchy_row
    assert "meep_central_probe_E_over_Eref" in hierarchy_row
    assert "meep_rms_external_E_over_Eref" in hierarchy_row
    assert hierarchy["numerical_convergence_status"] == "not_run_by_hierarchy_check"
    assert "fixed_central_probe_comparisons" in hierarchy


def test_formal_equation2_benchmark_rejects_substrate_interface(tmp_path) -> None:
    config = normalise_project_config(
        {
            "simulation_name": "equation2-substrate-not-formal",
            "geometry": {
                "particle_count": 1,
                "background_material": "air",
                "substrate_material": "sio2",
            },
        }
    )
    output = tmp_path / "validation"

    report = _theory_check(config, tmp_path / "experiment", output)

    assert report["status"] == "skipped"
    assert report["benchmark"] == "formal_equation2_homogeneous"
    assert report["benchmark_status"] == "failed_precondition"
    assert "no substrate interface" in report["reason"]
    assert not (output / "equation2_field_amplitude_profile.png").exists()


def test_comparison_summary_is_plain_language_and_uses_percent_errors() -> None:
    config = normalise_project_config({"simulation_name": "summary-test"})
    spectrum_metrics = {
        "same_scale_nrmse": 0.125,
        "correlation": 0.81,
        "relative_peak_wavelength_error": 0.03,
        "model_peak_at_scan_boundary": False,
        "agreement_status": "passed_qualitative_only",
    }
    reports = {
        "configuration": {
            "status": "passed",
            "numerical_sampling": {
                "cells_across_particle": 16.0,
                "cells_across_probe_offset": 4.0,
                "warnings": [],
            },
        },
        "theory": {
            "status": "completed",
            "wavelength_nm": 600.0,
            "agreement_status": "passed_quantitative",
            "scattered_field_agreement_status": "failed",
            "same_scale_total_field_comparison": {
                "amplitude_nrmse": 0.075,
                "amplitude_correlation": 0.95,
            },
            "same_scale_scattered_field_comparison": {
                "amplitude_nrmse": 0.25,
                "amplitude_correlation": 0.20,
            },
            "relative_equation2_residual": 2e-16,
        },
        "hierarchy": {
            "status": "completed",
            "maximum_scalar_vs_vector_relative_error": 1e-15,
            "same_observable_external_field_comparisons": {
                "quasistatic_cda_vs_meep": spectrum_metrics,
            },
            "fixed_central_probe_comparisons": {},
            "rms_probe_line_comparisons": {},
            "equation4_dipole_sum": "reported for the undefected equal-spacing chain",
        },
    }

    summary = _comparison_summary_markdown(config, reports)

    assert "# Comparison summary" in summary
    assert "`summary-test`" in summary
    assert "7.50%" in summary
    assert "25.00%" in summary
    assert "12.50%" in summary
    assert "3.00%" in summary
    assert "E/E_ref" in summary
    assert "not squared intensities" in summary
    assert "Exact multipole Mie theory" in summary
    assert "Not directly comparable" in summary
    assert "Resolution convergence" in summary
