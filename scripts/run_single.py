"""Run one named finite particle-line simulation.

Examples, from the project root:

    python scripts/run_single.py --config configs/perfect_line.json --precision fast --field-maps none
    python scripts/run_single.py --config configs/one_shifted_particle.json --name defect_dx20nm --field-maps auto
    python scripts/run_single.py --config configs/perfect_line.json --precision final --field-maps 620,670

Each command creates exactly one top-level folder in ``data/results``.  The
broadband stage always measures Ex/Ey/Ez versus line position and frequency.
The optional narrowband stage stores full 3-D fields only at selected
wavelengths.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from surface_plasmons.project import (
    ProjectConfigError,
    estimate_runtime,
    load_project_config,
    make_simulation_config,
    normalise_project_config,
)
from surface_plasmons.results import ResultsError
from surface_plasmons.workflow import WorkflowError, run_project


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "perfect_line.json"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--name", help="Override simulation_name from the JSON configuration.")
    parser.add_argument(
        "--precision",
        choices=("fast", "study", "final", "sweep"),
        help="sweep is accepted as a legacy alias for study.",
    )
    parser.add_argument(
        "--field-maps",
        help="none, auto, or comma-separated wavelengths in nm; default comes from the JSON file.",
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-reference-cache", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and estimate the broadband stage without creating result files or starting MEEP.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        loaded = load_project_config(args.config)
        config = normalise_project_config(loaded, name=args.name, precision=args.precision)
        preview = make_simulation_config(
            config,
            experiment_directory=Path("data/results") / "preview_only",
            stage_name="frequency_scan/particle",
        )
        estimate = estimate_runtime(preview)
    except (OSError, ProjectConfigError, ResultsError, ValueError) as exc:
        raise SystemExit(f"Invalid simulation configuration: {exc}") from exc

    print(f"Simulation: {config['simulation_name']}")
    print(f"Configuration: {args.config.resolve()}")
    print(f"Precision: {config['precision']}")
    print(f"Particles: {config['geometry']['particle_count']} {config['geometry']['particle_shape']}(s)")
    print(
        "Broadband estimate: "
        f"{_duration(float(estimate['estimated_min_seconds']))} to "
        f"{_duration(float(estimate['estimated_max_seconds']))}; "
        f"{int(estimate['grid_cells']):,} grid cells."
    )
    maps = args.field_maps if args.field_maps is not None else config["measurement"]["field_map_frequencies"]
    print(f"Narrowband field maps: {maps}")
    print(f"Reference reuse: {'disabled' if args.no_reference_cache else 'enabled'}")
    if args.dry_run:
        print("Dry run complete; MEEP was not started and no result folder was created.")
        return

    try:
        workspace = run_project(
            config,
            results_root=args.results_root,
            resume=args.resume,
            cache_enabled=not args.no_reference_cache,
            field_map_selection=args.field_maps,
        )
    except KeyboardInterrupt:
        raise SystemExit("Simulation interrupted. Partial results and status were preserved.")
    except (ProjectConfigError, ResultsError, WorkflowError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Simulation failed: {exc}") from exc
    print(f"Completed simulation folder: {workspace.directory}")


if __name__ == "__main__":
    main()
