"""Run a resumable radius sweep, analyse one run, and save figures."""

from surface_plasmons import (
    analyze_run,
    create_run_summary_figure,
    export_analysis,
    parameter_sweep,
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
            "geometry_parameters.radius": [0.030, 0.040, 0.050],
        },
        name="gold_sphere_radius",
        sweep_id="gold_sphere_radius",
        workers=1,
        resume=True,
    )
    print(f"Sweep: {sweep.status}")
    print(f"Manifest: {sweep.manifest_path}")

    completed = [
        run
        for run in sweep.runs
        if run.status in {"completed", "skipped"} and run.run_id
    ]
    if not completed:
        print("No completed run was available to analyse.")
        return

    run_id = completed[0].run_id
    analysis_path = export_analysis(
        analyze_run(run_id),
        f"data/results/analysis/{run_id}.json",
    )
    summary_path = f"data/results/images/{run_id}_summary.png"
    sweep_plot_path = "data/results/images/radius_vs_absorption.png"

    create_run_summary_figure(run_id, save_path=summary_path)
    plot_sweep_1d(
        sweep,
        parameter="geometry_parameters.radius",
        metric="absorption_maximum",
        parameter_label="Sphere radius (µm)",
        metric_label="Maximum absorption",
        save_path=sweep_plot_path,
    )

    print(f"Analysis: {analysis_path}")
    print(f"Run figure: {summary_path}")
    print(f"Sweep figure: {sweep_plot_path}")


if __name__ == "__main__":
    main()
