"""Symmetry parity, compatibility, and saved-baseline reuse without Meep."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from surface_plasmons.project import estimate_runtime, load_project_config, make_simulation_config
from surface_plasmons.simulation import SimulationConfigError, create_simulation, validate_config
from surface_plasmons.validation import run_validation
from surface_plasmons.workflow import StageFields, _reference_signature


def _config(tmp_path, *, polarisation="Ex", mirrors=True, reference=False):
    project = load_project_config(Path(__file__).parents[1] / "configs/notes_quasistatic_single_sphere_v5_longitudinal.json")
    project["source"]["polarisation"] = polarisation
    if mirrors:
        project["numerics"]["mirror_symmetry"] = True
    return make_simulation_config(project, experiment_directory=tmp_path, stage_name="test", reference=reference)


@pytest.mark.parametrize("polarisation,phases", [("Ex", [-1, 1]), ("Ez", [1, -1])])
def test_meep_receives_correct_mirrors_without_changing_full_geometry(tmp_path, monkeypatch, polarisation, phases):
    import surface_plasmons.simulation as runner
    project = load_project_config(Path(__file__).parents[1] / 'configs' / f'validated_sphere_{polarisation.lower()}_v15.json')
    config = make_simulation_config(project, experiment_directory=tmp_path, stage_name='test')
    arguments = {}

    class FakeSimulation:
        def __init__(self, **kwargs):
            arguments.update(kwargs)

        def add_dft_fields(self, *args, **kwargs):
            return "monitor"

    fake = SimpleNamespace(X="x", Y="y", Z="z", Ex="Ex", Ey="Ey", Ez="Ez",
                           Vector3=lambda *v: tuple(v), Mirror=lambda axis, phase: (axis, phase),
                           Simulation=FakeSimulation)
    monkeypatch.setattr(runner, "_import_meep", lambda: fake)
    monkeypatch.setattr(runner, "build_geometry", lambda *args: SimpleNamespace(
        name="particle_line_on_substrate", geometry=["original sphere"], default_material="air",
        parameters=config["geometry_parameters"], periodic_axes=()))
    monkeypatch.setattr(runner, "_build_source", lambda *args: ("original source", (0., .1, 0.)))
    monkeypatch.setattr(runner, "_build_pml", lambda *args: ["original PML"])
    create_simulation(config)
    assert arguments["symmetries"] == [("x", phases[0]), ("z", phases[1])]
    assert arguments["cell_size"] == tuple(config["cell_size_um"])
    assert arguments["geometry"] == ["original sphere"]
    assert arguments["resolution"] == 300


@pytest.mark.parametrize("change", [
    {"geometry_parameters": {"particle_count": 2}},
    {"geometry_parameters": {"defects": [{"particle_index": 0, "shift_um": [.01, 0., 0.]}]}},
    {"geometry_parameters": {"center": [.01, 0., 0.]}},
    {"geometry_parameters": {"substrate": "sio2"}},
    {"source": {"axis": "x", "component": "Ez"}},
    {"source": {"center_um": [0., .1, .01]}},
    {"k_point": [0., 0., .1]},
])
def test_unsupported_symmetry_is_rejected_before_meep_starts(tmp_path, change):
    config = _config(tmp_path)
    for key, value in change.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    with pytest.raises(SimulationConfigError, match="mirror_symmetry"):
        validate_config(config)


def test_mirrored_reference_has_distinct_cache_identity_and_quarter_estimate(tmp_path):
    original = _config(tmp_path, mirrors=False, reference=True)
    mirrored = _config(tmp_path, reference=True)
    assert "mirror_symmetry" not in original
    assert _reference_signature(original) != _reference_signature(mirrored)
    assert validate_config(mirrored).mirror_symmetry is True
    coarse = estimate_runtime(original)
    fine = deepcopy(mirrored)
    fine["resolution"] = 500
    estimate = estimate_runtime(fine)
    assert coarse["grid_cells"] == 8_320_000
    assert estimate["full_cell_grid_cells"] == 16_250_000
    assert estimate["grid_cells"] == 4_062_500


def test_mirrored_convergence_reuses_saved_baseline_and_completed_new_case(tmp_path, monkeypatch):
    from test_validation import synthetic
    directory, _, raw = synthetic(tmp_path)
    original_config = (directory / "config_resolved.json").read_bytes()
    calls = []

    def fake_run(candidate, workspace, **kwargs):
        calls.append(deepcopy(candidate))
        config = make_simulation_config(candidate, experiment_directory=workspace.directory, stage_name="test")
        validate_config(config)
        assert config["mirror_symmetry"] is True
        waves = np.array([480., 500., 520., 540., 560.])
        fields = {c: [] for c in ("Ex", "Ey", "Ez")}
        for wave in waves:
            selected = [r for r in raw if r["wavelength_nm"] == wave]
            for c in fields:
                fields[c].append([complex(r[f"normalised_{c}_real"], r[f"normalised_{c}_imag"]) for r in selected])
        fields = {c: np.asarray(v)[:, :, None, None] for c, v in fields.items()}
        coords = {"x": np.array([-.011, -.009, -.007, .007, .009, .011]), "y": np.array([.005]), "z": np.array([0.])}
        actual = StageFields(1000/waves, fields, coords, workspace.directory)
        reference = StageFields(1000/waves, {c: np.ones_like(v) if c == "Ex" else np.zeros_like(v)
                                           for c, v in fields.items()}, coords, workspace.directory)
        return None, actual, reference, {}

    monkeypatch.setattr("surface_plasmons.workflow.run_broadband_stage", fake_run)
    kwargs = dict(validation_name="mirrored", checks=["resolution"], resolutions=[1600, 1640], mirror_symmetry=True)
    output = run_validation(directory, **kwargs)
    report = json.loads((output / "resolution_convergence_summary.json").read_text())
    assert len(calls) == 1
    assert calls[0]["numerics"]["broadband_resolution"] == 1640
    assert report["finest_comparison"]["source"] == "saved_baseline"
    assert report["mirror_symmetry_for_new_runs"] is True
    assert report["baseline_reused_without_rerunning"] is True
    assert report["convergence_status"] == "passed"
    run_validation(directory, **kwargs)
    assert len(calls) == 1
    assert (directory / "config_resolved.json").read_bytes() == original_config
