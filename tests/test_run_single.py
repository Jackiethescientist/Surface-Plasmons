import csv
import json
from types import SimpleNamespace

from scripts.run_single import (
    _available_validation_name,
    _write_automatic_validation_file,
)
from surface_plasmons.project import normalise_project_config


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_automatic_validation_file_is_stored_in_simulation_folder(tmp_path) -> None:
    simulation = tmp_path / "one_simulation"
    validation = simulation / "validation" / "automatic_theory_validation_v1"
    validation.mkdir(parents=True)
    (validation / "validation_summary.json").write_text(
        json.dumps({"configuration": {"status": "passed"}}),
        encoding="utf-8",
    )
    _write_csv(
        validation / "notes_spectrum.csv",
        [
            {
                "wavelength_nm": "600.0",
                "particle_radius_um": "0.04",
                "alpha_equation1_real_um3": "0.001",
                "meep_rms_scattered_field": "2.5",
            }
        ],
    )
    config = normalise_project_config({"simulation_name": "one simulation"})
    particle = SimpleNamespace(to_dict=lambda: {"particle_index": 0})

    destination = _write_automatic_validation_file(
        simulation, validation, config, [particle]
    )

    assert destination == simulation / "automatic_theory_validation.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["simulation_variables"]["simulation_name"] == "one simulation"
    assert payload["derived_geometry"]["active_particle_count"] == 1
    assert payload["overall_validation_status"] == (
        "not_validated"
    )
    assert payload["validation_scope"]["manual_checks_not_run"] == [
        "resolution",
        "domain",
        "time",
    ]
    assert payload["formula_substitutions_and_model_results_by_wavelength"][0][
        "meep_rms_scattered_field"
    ] == 2.5
    assert "coupled_dipoles" in payload["theory_formulas"]


def test_automatic_validation_name_never_overwrites(tmp_path) -> None:
    simulation = tmp_path / "simulation"
    (simulation / "validation" / "automatic_theory_validation_v1").mkdir(
        parents=True
    )

    assert _available_validation_name(
        simulation, "automatic_theory_validation_v1"
    ) == "automatic_theory_validation_v1_v2"

