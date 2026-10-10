from pathlib import Path

import pytest

from surface_plasmons.particle_line import ParticleLineError, particle_line_positions
from surface_plasmons.project import make_simulation_config, normalise_project_config
from surface_plasmons.results import ExperimentWorkspace, ResultsError


def _parameters(**updates):
    value = {
        "particle_shape": "sphere",
        "particle_count": 5,
        "radius": 0.04,
        "cube_size": 0.08,
        "spacing": 0.15,
        "surface_y": 0.0,
        "particle_surface_gap": 0.0,
        "substrate_thickness": 0.3,
        "center": (0.0, 0.0, 0.0),
        "defects": [],
    }
    value.update(updates)
    return value


def test_perfect_line_is_centred_and_equally_spaced() -> None:
    positions = particle_line_positions(_parameters())

    assert [item.actual_x_um for item in positions] == pytest.approx(
        [-0.30, -0.15, 0.0, 0.15, 0.30]
    )
    assert all(item.actual_y_um == pytest.approx(0.04) for item in positions)


def test_one_defect_changes_only_selected_particle() -> None:
    positions = particle_line_positions(
        _parameters(defects=[{"particle_index": 2, "shift_um": [0.02, 0.0, 0.0]}])
    )

    assert positions[2].actual_x_um == pytest.approx(0.02)
    assert positions[1].actual_x_um == pytest.approx(-0.15)
    assert positions[3].actual_x_um == pytest.approx(0.15)


def test_overlapping_defect_is_rejected() -> None:
    with pytest.raises(ParticleLineError, match="overlap"):
        particle_line_positions(
            _parameters(defects=[{"particle_index": 2, "shift_um": [0.09, 0.0, 0.0]}])
        )


def test_project_rejects_longitudinal_plane_wave() -> None:
    with pytest.raises(ValueError, match="transverse"):
        normalise_project_config(
            {"source": {"propagation_axis": "y", "polarisation": "Ey"}}
        )


def test_broadband_config_uses_vector_fields_and_one_experiment_folder(tmp_path: Path) -> None:
    project = normalise_project_config({})
    simulation = make_simulation_config(
        project,
        experiment_directory=tmp_path / "experiment",
        stage_name="frequency_scan/particle",
    )

    assert simulation["dimensions"] == 3
    assert simulation["source"]["axis"] == "y"
    assert simulation["source"]["component"] == "Ex"
    assert simulation["output"]["field_components"] == ["Ex", "Ey", "Ez"]
    assert simulation["output"]["field_nfreq"] > 1
    # A 40 nm-radius sphere resting at y=0 has its top at 0.080 µm.
    # The broadband probe must be outside the metal.
    assert simulation["output"]["field_region_center_um"][1] > 0.080
    assert simulation["output"]["experiment_directory"] == str((tmp_path / "experiment").resolve())


def test_workspace_never_silently_overwrites(tmp_path: Path) -> None:
    config = normalise_project_config({})
    workspace = ExperimentWorkspace.create("named simulation", config, results_root=tmp_path)
    assert workspace.directory.name == "named_simulation"

    with pytest.raises(ResultsError, match="already exists"):
        ExperimentWorkspace.create("named simulation", config, results_root=tmp_path)

    workspace.set_status("interrupted")
    resumed = ExperimentWorkspace.create("named simulation", config, results_root=tmp_path, resume=True)
    assert resumed.directory == workspace.directory
