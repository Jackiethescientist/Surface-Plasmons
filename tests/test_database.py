from surface_plasmons.database import SimulationDatabase


def test_database_records_and_exports_a_run(tmp_path) -> None:
    database = SimulationDatabase(tmp_path / "simulations.db")
    config = {
        "geometry_name": "nanoparticle_sphere",
        "geometry_parameters": {"particle": "au", "radius": 0.04},
        "dimensions": 2,
        "resolution": 80,
        "source": {"wavelength_range_um": [0.5, 0.8]},
    }

    run_id = database.create_run(config, name="database-test")
    database.mark_running(run_id, output_directory=tmp_path / "run")
    database.record_metric(run_id, "absorption_maximum", 0.72)
    database.record_spectrum(
        run_id,
        "absorption",
        frequencies=[1.25, 1.5, 2.0],
        values=[0.2, 0.72, 0.1],
    )
    database.mark_completed(run_id, elapsed_seconds=1.25)

    run = database.get_run(run_id)
    assert run["status"] == "completed"
    assert run["geometry_name"] == "nanoparticle_sphere"

    metrics = database.get_metrics(run_id)
    assert metrics[0]["name"] == "absorption_maximum"
    assert metrics[0]["value"] == 0.72

    spectra = database.get_spectra(run_id)
    assert len(spectra["absorption"]) == 3

    exported = database.export_run(run_id)
    assert exported["run"]["run_id"] == run_id
    assert "absorption" in exported["spectra"]
