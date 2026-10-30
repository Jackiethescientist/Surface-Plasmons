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
import csv
import json
import sys

_LOCAL_SOURCE = Path(__file__).resolve().parents[1] / 'src'
if not (_LOCAL_SOURCE / 'surface_plasmons' / '__init__.py').is_file():
    raise SystemExit('Project source is missing. Merge code/ into your existing project root; '
                     'scripts/ and src/surface_plasmons/ must be alongside each other.')
sys.path.insert(0, str(_LOCAL_SOURCE))

from surface_plasmons.project import (
    ProjectConfigError,
    estimate_runtime,
    load_project_config,
    make_simulation_config,
    normalise_project_config,
)
from surface_plasmons.particle_line import particle_line_positions
from surface_plasmons.results import ResultsError, safe_simulation_name
from surface_plasmons.validation import ValidationError, run_validation
from surface_plasmons.workflow import WorkflowError, run_project


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "validated_sphere_ex_v15.json"

from surface_plasmons.notes_theory import FORMULAS as THEORY_FORMULAS
from surface_plasmons.validated_scope import enforce_validated_project


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _available_validation_name(simulation_directory: Path, requested: str) -> str:
    """Choose a validation name without overwriting an earlier attempt."""

    base = safe_simulation_name(requested)
    validation_root = simulation_directory / "validation"
    candidate = base
    revision = 2
    while (validation_root / candidate).exists():
        candidate = f"{base}_v{revision}"
        revision += 1
    return candidate


def _csv_value(value: str) -> Any:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    if number.is_integer() and all(token not in stripped.lower() for token in (".", "e")):
        return int(number)
    return number


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: _csv_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _write_automatic_validation_file(
    simulation_directory: Path,
    validation_directory: Path,
    config: dict[str, Any],
    active_particles: Any,
) -> Path:
    """Create one self-contained theory-validation record per simulation."""

    summary_path = validation_directory / "validation_summary.json"
    reports = json.loads(summary_path.read_text(encoding="utf-8"))
    generated_files = sorted(
        str(path.relative_to(simulation_directory))
        for path in validation_directory.rglob("*")
        if path.is_file()
    )
    theory = reports.get("theory", {})
    payload = {
        "schema_version": 2,
        "simulation_name": config["simulation_name"],
        "simulation_variables": config,
        "overall_validation_status": reports.get("overall", {}).get("status", "not_validated"),
        "theory_agreement_status": theory.get("agreement_status", "not_evaluated"),
        "model_applicability_status": theory.get("applicability_status", "not_evaluated"),
        "numerical_convergence_status": {
            kind: reports.get(kind, {}).get("convergence_status", "not_established")
            for kind in ("resolution", "domain", "time")
        },
        "validation_scope": {
            "automatic_checks": ["configuration", "theory"],
            "uses_existing_meep_results": True,
            "starts_additional_meep_runs": False,
            "manual_checks_not_run": ["resolution", "domain", "time"],
        },
        "derived_geometry": {
            "active_particle_count": len(active_particles),
            "particle_positions_um": [particle.to_dict() for particle in active_particles],
        },
        "theory_formulas": THEORY_FORMULAS,
        "formula_substitutions_and_model_results_by_wavelength": _read_csv_records(validation_directory / "notes_spectrum.csv"),
        "dipole_diagnostics": _read_csv_records(validation_directory / "notes_dipoles.csv"),
        "pair_mode_predictions": _read_csv_records(validation_directory / "notes_pair_modes.csv"),
        "validation_reports": reports,
        "generated_validation_files": generated_files,
    }
    destination = simulation_directory / "automatic_theory_validation.json"
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


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
        "--skip-validation",
        action="store_true",
        help="Skip automatic read-only validation against the electrostatic notes.",
    )
    parser.add_argument(
        "--validation-name",
        default="automatic_theory_validation_v1",
        help="Name for this run's detailed validation subfolder.",
    )
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
        enforce_validated_project(config, field_map_selection=args.field_maps)
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
    active_particles = particle_line_positions(
        {**preview["geometry_parameters"], "center": (0.0, 0.0, 0.0)}
    )
    print(
        "Particles: "
        f"{len(active_particles)} active of {config['geometry']['particle_count']} nominal "
        f"{config['geometry']['particle_shape']}(s)"
    )
    print(
        "Broadband estimate: "
        f"{_duration(float(estimate['estimated_min_seconds']))} to "
        f"{_duration(float(estimate['estimated_max_seconds']))}; "
        f"{int(estimate['grid_cells']):,} grid cells."
    )
    maps = args.field_maps if args.field_maps is not None else config["measurement"]["field_map_frequencies"]
    print(f"Narrowband field maps: {maps}")
    print(f"Reference reuse: {'disabled' if args.no_reference_cache else 'enabled'}")
    print(f"Automatic theory validation: {'disabled' if args.skip_validation else 'enabled'}")
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
    if not args.skip_validation:
        validation_name = args.validation_name
        try:
            validation_directory = run_validation(
                workspace.directory,
                validation_name=validation_name,
                checks=("configuration", "theory"),
            )
            automatic_file = _write_automatic_validation_file(
                workspace.directory,
                validation_directory,
                config,
                active_particles,
            )
        except (OSError, ValidationError, ResultsError, RuntimeError, ValueError) as exc:
            raise SystemExit(
                "Simulation completed, but automatic validation failed: "
                f"{exc}. The simulation results were preserved in {workspace.directory}."
            ) from exc
        print(f"Detailed validation folder: {validation_directory}")
        print(f"Automatic validation file: {automatic_file}")
    print(f"Completed simulation folder: {workspace.directory}")


if __name__ == "__main__":
    import sys

    DRY_RUN = False  # Change to False for a real simulation

    if DRY_RUN and "--dry-run" not in sys.argv:
        sys.argv.append("--dry-run")

    main()

