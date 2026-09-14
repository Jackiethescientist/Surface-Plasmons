"""Run a named parameter sweep and combine its tabular results into one CSV.

Edit the clearly labelled section below, then run from the project root:

    python scripts/run_sweep.py

Or give this particular sweep a name from the terminal:

    python scripts/run_sweep.py --name "gold sphere radius study"

The combined CSV has one row per run and wavelength.  It contains the varied
parameters, scalar metrics, full spectra, configuration, status and errors.
Large field/epsilon arrays remain in NumPy files because a spreadsheet is not a
practical representation of those multidimensional arrays.
"""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from surface_plasmons.sweep import SweepResult, parameter_sweep


# ---------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH SWEEP
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_NAME = "gold sphere radius study"

# Every length is in micrometres: 0.040 um = 40 nm.
BASE_CONFIG: dict[str, Any] = {
    "geometry_name": "nanoparticle_sphere",
    "geometry_parameters": {
        "particle": "au",
        "background": "air",
        "radius": 0.040,
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
    "output": {
        "save_config": True,
        "save_summary": True,   # required for the combined CSV
        "save_spectra": True,
        "save_epsilon": True,
        "save_fields": True,
        "save_images": True,
        "field_components": ["Ez"],
    },
    "notes": "",
    "tags": ["sweep"],
}

# Use dot paths to identify values inside BASE_CONFIG.
# "grid" mode tests every possible combination of the listed values.
# "zip" mode pairs the first values, then the second values, and so on.
SWEEP_PARAMETERS: dict[str, Sequence[Any]] = {
    "geometry_parameters.radius": [0.030, 0.040, 0.050],
}
SWEEP_MODE = "grid"
REPETITIONS = 1
DEFAULT_WORKERS = 1  # Keep at 1 until individual MEEP runs work reliably.


# ---------------------------------------------------------------------------
# RUNNER CODE -- normally you do not need to edit below this line
# ---------------------------------------------------------------------------

EXCEL_MAX_DATA_ROWS = 1_048_575  # Header uses the remaining Excel worksheet row.


def safe_name(name: str) -> str:
    """Convert a display name into a safe, readable filename component."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("The sweep name must contain at least one letter or number.")
    return cleaned[:70]


def make_sweep_id(name: str) -> str:
    """Return ``readable_name__UTC_timestamp`` for collision-safe storage."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_name(name)}__{timestamp}"


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _read_summary(output_directory: str | None) -> dict[str, Any]:
    if not output_directory:
        return {}
    path = Path(output_directory) / "summary.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: could not read {path}: {exc}")
        return {}


def _discover_columns(result: SweepResult) -> tuple[list[str], list[str]]:
    metric_names = {name for run in result.runs for name in run.metrics}
    monitor_names: set[str] = set()
    for run in result.runs:
        summary = _read_summary(run.output_directory)
        metric_names.update(map(str, summary.get("metrics", {})))
        monitor_names.update(map(str, summary.get("spectra", {})))
    return sorted(metric_names), sorted(monitor_names)


def _at(values: Sequence[Any], index: int) -> Any:
    return values[index] if index < len(values) else ""


def export_combined_csv(result: SweepResult) -> tuple[Path, int]:
    """Write one Excel-ready file containing every sweep run and spectrum."""

    destination = Path(result.output_directory) / f"{result.sweep_id}_all_results.csv"
    parameter_names = [axis.path for axis in result.axes]
    metric_names, monitor_names = _discover_columns(result)
    fixed_columns = [
        "sweep_name",
        "sweep_id",
        "run_number",
        "sweep_index",
        "repetition",
        "status",
        "simulation_name",
        "run_id",
        "config_hash",
        "elapsed_seconds",
        "output_directory",
        "error_type",
        "error_message",
        "sample_index",
        "frequency",
        "wavelength_um",
        "config_json",
        "warnings_json",
    ]
    fieldnames = [
        *fixed_columns,
        *(f"parameter.{name}" for name in parameter_names),
        *(f"metric.{name}" for name in metric_names),
        *(f"spectrum.{name}" for name in monitor_names),
    ]

    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    # utf-8-sig lets Excel recognise UTF-8 automatically.
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for run_number, run in enumerate(result.runs, start=1):
            summary = _read_summary(run.output_directory)
            config = summary.get("config", {})
            spectra = summary.get("spectra", {})
            frequencies = summary.get("frequencies", [])
            wavelengths = summary.get("wavelengths_um", [])
            metrics = {**run.metrics, **summary.get("metrics", {})}
            simulation_name = config.get(
                "run_name",
                f"{result.name}-{run.index + 1:04d}-r{run.repetition + 1}",
            )
            sample_count = max(
                [len(frequencies), len(wavelengths)]
                + [len(values) for values in spectra.values()],
                default=0,
            )
            common = {
                "sweep_name": result.name,
                "sweep_id": result.sweep_id,
                "run_number": run_number,
                "sweep_index": run.index,
                "repetition": run.repetition + 1,
                "status": run.status,
                "simulation_name": simulation_name,
                "run_id": run.run_id or "",
                "config_hash": run.config_hash,
                "elapsed_seconds": run.elapsed_seconds,
                "output_directory": run.output_directory or "",
                "error_type": run.error_type or "",
                "error_message": run.error_message or "",
                "config_json": _csv_value(config),
                "warnings_json": _csv_value(summary.get("warnings", [])),
                **{
                    f"parameter.{name}": _csv_value(run.parameters.get(name))
                    for name in parameter_names
                },
                **{
                    f"metric.{name}": _csv_value(metrics.get(name))
                    for name in metric_names
                },
            }

            # Failed runs, or runs without flux monitors, still receive one row.
            for sample_index in range(max(sample_count, 1)):
                row = {
                    **common,
                    "sample_index": sample_index if sample_count else "",
                    "frequency": _at(frequencies, sample_index),
                    "wavelength_um": _at(wavelengths, sample_index),
                }
                for monitor_name in monitor_names:
                    values = spectra.get(monitor_name, [])
                    row[f"spectrum.{monitor_name}"] = _at(values, sample_index)
                writer.writerow(row)
                row_count += 1

    return destination, row_count


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=DEFAULT_SWEEP_NAME,
        help="Readable sweep name (default: value set at the top of the file).",
    )
    parser.add_argument(
        "--sweep-id",
        help=(
            "Reuse an exact existing sweep ID to continue it. If omitted, a new "
            "timestamped ID is created so previous results are preserved."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of isolated MEEP worker processes.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Run every point even if an identical completed run is in the database.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first failed sweep point.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_name = args.name.strip()
    if not display_name:
        raise SystemExit("Sweep name cannot be empty.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    sweep_id = args.sweep_id or make_sweep_id(display_name)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sweep_id):
        raise SystemExit(
            "--sweep-id may contain only letters, numbers, full stops, underscores and hyphens."
        )

    base_config = deepcopy(BASE_CONFIG)
    base_config["tags"] = list(
        dict.fromkeys([*base_config.get("tags", []), f"sweep:{safe_name(display_name)}"])
    )

    print(f"Starting sweep: {display_name}")
    print(f"Sweep ID / results name: {sweep_id}")
    result = parameter_sweep(
        base_config=base_config,
        parameters=deepcopy(SWEEP_PARAMETERS),
        mode=SWEEP_MODE,
        name=display_name,
        sweep_id=sweep_id,
        workers=args.workers,
        repetitions=REPETITIONS,
        continue_on_error=not args.stop_on_error,
        resume=not args.no_resume,
    )

    csv_path, row_count = export_combined_csv(result)
    print(f"Status: {result.status}")
    print(f"Completed/skipped runs: {result.completed_runs}")
    print(f"Failed runs: {result.failed_runs}")
    print(f"Results folder: {result.output_directory}")
    print(f"Excel-ready combined CSV: {csv_path}")
    print(f"CSV data rows: {row_count}")
    if row_count > EXCEL_MAX_DATA_ROWS:
        print(
            "Warning: this CSV is larger than one Excel worksheet. The data is "
            "still valid, but use Excel's Data Model, Power Query, Python, or another tool."
        )

    if result.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    # This guard is required when workers > 1.
    main()
