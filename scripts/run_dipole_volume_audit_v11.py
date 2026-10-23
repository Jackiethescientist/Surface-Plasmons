"""Measure the sphere's dipole directly, without fitting exterior fields.

Place this new file in scripts/. Requires the existing v7 mirror update.
Example (from the project root, in the plasmon environment):

    python scripts/run_dipole_volume_audit_v11.py --simulation notes_quasistatic_v5_single_longitudinal --resolution 600 --name dipole_volume_600_v11

One new particle simulation records a small E/D volume as well as the usual
line monitor. A compatible particle-free reference is reused when available.
Outputs are under the original experiment's validation/<name>/ directory.
The same command reuses completed audit arrays. Partial time stepping is not
checkpointed. The audit writes CSV/JSON/NPZ files and does not write SQLite.

In Meep units, q = integral(D - epsilon_host*E) dV / E_reference, in um^3.
The integration box includes the whole sphere and a three-cell air buffer.
There is no sphere mask or fitted amplitude correction in this integral.
This is an additional numerical diagnostic, not proof of continuum accuracy.
"""
from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import csv
import json

import numpy as np

from surface_plasmons.materials import dielectric_function
from surface_plasmons.notes_theory import field_matrix
from surface_plasmons.project import make_simulation_config, normalise_project_config
from surface_plasmons.results import ExperimentWorkspace, _write_json, safe_simulation_name
from surface_plasmons.simulation import create_simulation, validate_config, _execute
from surface_plasmons.validation import resolve_experiment, _theory_check
from surface_plasmons.workflow import (
    StageFields, _normalised_fields, _run_reference, _save_broadband_results,
    load_stage_fields, print_stage_estimate,
)


def volume_moment(electric, displacement, epsilon_host, weights):
    """Integrate contrast polarization; inputs share Meep's centred grid."""
    e, d, w = np.asarray(electric), np.asarray(displacement), np.asarray(weights)
    if e.shape != d.shape or e.shape != w.shape + (3,):
        raise ValueError("E/D components and volume weights must share a grid.")
    if not all(np.all(np.isfinite(a)) for a in (e, d, w)) or not np.isfinite(epsilon_host):
        raise ValueError("Nonfinite volume data.")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid volume cubature weights.")
    return np.sum((d - epsilon_host * e) * w[..., None], axis=tuple(range(w.ndim)))


def _relative(actual, predicted):
    denominator = np.linalg.norm(predicted)
    if denominator <= 1e-25:
        raise ValueError("A diagnostic reference norm is zero.")
    return float(np.linalg.norm(actual - predicted) / denominator)


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _array(simulation, monitor, component, index, shape):
    value = np.asarray(simulation.get_dft_array(monitor, component, index))
    # Meep may return scalar zero for a component that vanishes by symmetry.
    return np.array(np.broadcast_to(value, shape), copy=True)


def _record_particle(config, workspace):
    import meep as mp
    if getattr(mp, "count_processors", lambda: 1)() != 1:
        raise ValueError("Run this diagnostic with serial Python, without mpirun.")
    simulation_config = make_simulation_config(config, experiment_directory=workspace.directory,
                                                stage_name="frequency_scan/particle")
    simulation_config["record_database"] = False
    print_stage_estimate("Particle plus dipole-volume monitor", simulation_config)
    context = create_simulation(simulation_config)
    simulation = context.simulation
    try:
        radius = float(config["geometry"]["particle_radius_um"])
        center = np.array([0., config["measurement"]["probe_line_y_um"], 0.])
        half = radius + 3 / float(simulation_config["resolution"])
        interior = np.array(context.cell_size_um) / 2 - simulation_config["pml_thickness_um"]
        if np.any(np.abs(center) + half >= interior):
            raise ValueError("The moment box reaches the absorbing layer.")
        if abs(context.source_center_um[1] - center[1]) <= half:
            raise ValueError("The plane source intersects the moment box.")
        frequencies = np.asarray(context.field_frequencies)
        monitor = simulation.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez, mp.Dx, mp.Dy, mp.Dz],
            float((frequencies[0] + frequencies[-1]) / 2),
            float(frequencies[-1] - frequencies[0]), len(frequencies),
            center=mp.Vector3(*center), size=mp.Vector3(*(2 * half * np.ones(3))),
        )
        workspace.record_stage("frequency_scan", "running")
        started = perf_counter()
        _execute(context)
        x, y, z, weights = simulation.get_array_metadata(dft_cell=monitor)
        weights = np.asarray(weights)
        eps_host = dielectric_function(config["geometry"]["background_material"], frequencies)
        moments = []
        for index, epsilon in enumerate(eps_host):
            e = np.stack([_array(simulation, monitor, c, index, weights.shape)
                          for c in (mp.Ex, mp.Ey, mp.Ez)], axis=-1)
            d = np.stack([_array(simulation, monitor, c, index, weights.shape)
                          for c in (mp.Dx, mp.Dy, mp.Dz)], axis=-1)
            moments.append(volume_moment(e, d, epsilon, weights))
        line_x, line_y, line_z, line_weights = simulation.get_array_metadata(dft_cell=context.dft_fields)
        shape = np.asarray(line_weights).shape
        stage = workspace.directory / "stages/frequency_scan/particle"
        arrays = stage / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        _write_json(stage / "config.json", context.config.to_dict())
        for name, component in context.dft_components.items():
            values = np.stack([_array(simulation, context.dft_fields, component, i, shape)
                               for i in range(len(frequencies))])
            np.savez_compressed(arrays / f"field_{name}_spectrum.npz", values=values, frequencies=frequencies)
        np.savez(arrays / "coordinates.npz", x=line_x, y=line_y, z=line_z, weights=line_weights)
        np.savez_compressed(workspace.directory / "dipole_integral.npz", frequencies=frequencies,
                            raw_moments=np.asarray(moments), volume_um3=float(weights.sum()),
                            volume_shape=np.asarray(weights.shape), box_center_um=center,
                            box_size_um=np.full(3, 2 * half), host_epsilon=np.asarray(eps_host))
        workspace.record_stage("frequency_scan", "completed", elapsed_seconds=perf_counter() - started)
    finally:
        simulation.reset_meep()


def _analyse(config, workspace, reference):
    actual = load_stage_fields(workspace.directory / "stages/frequency_scan/particle")
    with np.load(workspace.directory / "dipole_integral.npz", allow_pickle=False) as archive:
        frequencies = archive["frequencies"]
        raw = archive["raw_moments"]
        box_volume = float(archive["volume_um3"])
    if not np.allclose(actual.frequencies, frequencies):
        raise ValueError("Moment and line monitors have different frequency grids.")
    normalised = _normalised_fields(actual, reference,
        reference_component=config["source"]["polarisation"], mode="global_complex_reference")
    scales = normalised["reference_complex_scale"].reshape(len(frequencies), -1)[:, 0]
    moments = raw / scales[:, None]
    _save_broadband_results(config, actual, reference, workspace)
    theory_dir = workspace.directory / "notes_comparison"
    theory = _theory_check(config, workspace.directory, theory_dir)
    if theory.get("status") != "completed":
        raise ValueError("The notes check could not select enough eligible exterior probes.")
    dipoles = _read(theory_dir / "notes_dipoles.csv")
    fields = _read(theory_dir / "notes_fields.csv")
    spectrum = _read(theory_dir / "notes_spectrum.csv")
    axis = {"ex": 0, "ez": 2}[config["source"]["polarisation"].lower()]
    center = np.array([[0., config["measurement"]["probe_line_y_um"], 0.]])
    rows, theory_p, fitted_p, direct_p, measured_fields, direct_fields = [], [], [], [], [], []
    for i, frequency in enumerate(frequencies):
        wave = 1000 / frequency
        row = min(dipoles, key=lambda r: abs(float(r["wavelength_nm"]) - wave))
        if abs(float(row["wavelength_nm"]) - wave) > 1e-6:
            raise ValueError("Moment frequency is absent from the theory report.")
        predicted = complex(float(row["theory_q_parallel_real_um3"]), float(row["theory_q_parallel_imag_um3"]))
        fitted = complex(float(row["fitted_q_parallel_real_um3"]), float(row["fitted_q_parallel_imag_um3"]))
        selected = [r for r in fields if abs(float(r["wavelength_nm"]) - wave) < 1e-6]
        points = np.array([[float(r[f"{c}_um"]) for c in "xyz"] for r in selected])
        em = complex(float(min(spectrum, key=lambda r: abs(float(r["wavelength_nm"]) - wave))["epsilon_host_real"]))
        direct_field = (field_matrix(points, center, em) @ moments[i]).reshape(-1, 3)
        measured_field = np.array([[complex(float(r[f"meep_{c}_real"]), float(r[f"meep_{c}_imag"]))
                                   for c in ("Ex", "Ey", "Ez")] for r in selected])
        measured_fields.append(measured_field.reshape(-1))
        direct_fields.append(direct_field.reshape(-1))
        theory_p.append(predicted); fitted_p.append(fitted); direct_p.append(moments[i, axis])
        rows.append({"wavelength_nm": wave, "theory_q_real_um3": predicted.real,
                     "theory_q_imag_um3": predicted.imag, "fitted_q_real_um3": fitted.real,
                     "fitted_q_imag_um3": fitted.imag,
                     **{f"volume_q_{c}_{part}_um3": float(getattr(moments[i, j], part))
                        for j, c in enumerate("xyz") for part in ("real", "imag")}})
    with (workspace.directory / "dipole_volume_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = {
        "status": "completed", "resolution": config["numerics"]["broadband_resolution"],
        "method": "q_volume = integral(D - epsilon_host*E) dV / E_reference",
        "unit": "um^3; equivalent to p_SI/(epsilon_0*E_reference*1e-18)",
        "integration_box_volume_um3": box_volume,
        "volume_dipole_vs_notes_relative_l2": _relative(np.array(direct_p), np.array(theory_p)),
        "fitted_dipole_vs_volume_relative_l2": _relative(np.array(fitted_p), np.array(direct_p)),
        "exterior_field_vs_static_volume_dipole_relative_l2": _relative(np.concatenate(measured_fields), np.concatenate(direct_fields)),
        "exterior_field_vs_notes_relative_l2": theory["complex_scattered_field_relative_l2_error"],
        "interpretation": "The volume dipole is independently measured from D/E. The exterior fitted dipole remains conditional on the static kernel. This audit does not change tolerances or prove continuum accuracy.",
        "overall_validation_status": "diagnostic_only",
    }
    _write_json(workspace.directory / "dipole_volume_summary.json", report)
    workspace.set_status("completed", message="Independent dipole audit saved")
    return report


def run_audit(simulation, *, name="dipole_volume_600_v11", resolution=600, results_root=None):
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError("resolution must be a positive integer.")
    baseline = resolve_experiment(simulation, results_root)
    config = normalise_project_config(json.loads((baseline / "config_resolved.json").read_text()))
    config = deepcopy(config)
    config["simulation_name"] = safe_simulation_name(name)
    config["numerics"].update(broadband_resolution=resolution, mirror_symmetry=True, record_database=False)
    config["measurement"]["field_map_frequencies"] = "none"
    expected_y = float(config["geometry"]["surface_y_um"]) + float(config["geometry"]["particle_surface_gap_um"]) + float(config["geometry"]["particle_radius_um"])
    if abs(float(config["measurement"].get("probe_line_y_um", float("inf"))) - expected_y) > 1e-12:
        raise ValueError("This audit requires the existing centreline monitor through the sphere centre.")
    root = baseline / "validation"
    output = root / safe_simulation_name(name)
    preview = make_simulation_config(config, experiment_directory=output, stage_name="frequency_scan/particle")
    validate_config(preview)  # existing v7 guard enforces the supported Au/air sphere and source
    identity = {"schema": "independent_volume_dipole_v11", "config": config, "buffer_cells": 3}
    marker = output / "dipole_audit_config.json"
    if output.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != identity:
            raise ValueError("The audit name belongs to different settings; choose another --name.")
        summary = output / "dipole_volume_summary.json"
        if summary.is_file() and json.loads(summary.read_text()).get("status") == "completed":
            print(f"Reusing completed audit: {summary}")
            return json.loads(summary.read_text())
    workspace = ExperimentWorkspace.create(name, config, results_root=root, resume=output.exists())
    _write_json(marker, identity)
    _write_json(output / "config_resolved.json", config)
    reference = _run_reference(config, wavelength_um=None, cache_enabled=True)
    if workspace.manifest().get("stages", {}).get("frequency_scan", {}).get("status") != "completed":
        _record_particle(config, workspace)
    return _analyse(config, workspace, reference)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--name", default="dipole_volume_600_v11")
    parser.add_argument("--resolution", type=int, default=600)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    try:
        report = run_audit(args.simulation, name=args.name, resolution=args.resolution, results_root=args.results_root)
    except KeyboardInterrupt:
        raise SystemExit("Audit interrupted. Rerun the same command to reuse the reference and any completed audit arrays.")
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise SystemExit(f"Dipole audit failed: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
