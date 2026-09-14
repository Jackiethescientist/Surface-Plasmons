"""Small end-to-end example. Run this file from the project root."""

from surface_plasmons.analysis import analyze_run, export_analysis
from surface_plasmons.sweep import parameter_sweep
from surface_plasmons.visualization import (
    create_run_summary_figure,
    plot_sweep_1d,
)


def main() -> None:
    sweep = parameter_sweep(
        base_config={
            "geometry_name": "nanoparticle_sphere",
            "geometry_parameters": {
                "particle": "au",
                "background": "air",
            },
            "dimensions": 2,
            "resolution": 80,
            "source": {
                "wavelength_range_um": [0.45, 0.80],
                "component": "Ez",
            },
            "flux": {
                "nfreq": 101,
                "normalization": "empty_cell",
            },
        },
        parameters={
            "geometry_parameters.radius": [0.03, 0.04, 0.05],
        },
        name="gold_sphere_radius",
        workers=1,
    )
    print(f"Sweep: {sweep.status}")
    print(f"Manifest: {sweep.manifest_path}")

    completed = [run for run in sweep.runs if run.run_id]
    if not completed:
        print("No completed run was available to analyse.")
        return

    run_id = completed[0].run_id
    report = analyze_run(run_id)
    export_analysis(report, f"data/results/analysis/{run_id}.json")
    create_run_summary_figure(
        run_id,
        save_path=f"data/results/images/{run_id}_summary.png",
    )
    plot_sweep_1d(
        sweep,
        parameter="geometry_parameters.radius",
        metric="absorption_maximum",
        parameter_label="Sphere radius (µm)",
        metric_label="Maximum absorption",
        save_path="data/results/images/radius_vs_absorption.png",
    )


if __name__ == "__main__":
    main()
