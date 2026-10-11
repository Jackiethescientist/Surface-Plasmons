import csv

from surface_plasmons.project import normalise_project_config
from surface_plasmons.validation import _theory_check


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
    assert report["interaction_model"] == "paper_retarded"
    assert report["polarizability_model"] == "quasistatic"
    assert report["finite_size_corrections"] == (
        "disabled for the strict Section-2.1 baseline"
    )
    assert report["normalization"]["independent_curve_rescaling"] is False
    assert report["maximum_equation2_residual"] < 1e-12
    assert (output / "equation2_field_amplitude_profile.png").is_file()
    assert not (output / "equation2_field_intensity_profile.png").exists()


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
