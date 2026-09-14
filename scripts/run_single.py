"""Run one named surface-plasmon simulation.

Edit ``DEFAULT_SIMULATION_NAME`` and ``SIMULATION_CONFIG`` below, then run
this file from the project root:

    python scripts/run_single.py

You can also supply a one-off name without editing the file:

    python scripts/run_single.py --name "gold sphere 40 nm test"

The readable name is stored in the database.  A filesystem-safe version plus
a UTC timestamp becomes the run ID and results-folder name, so old results are
never silently overwritten.
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

from surface_plasmons.simulation import SimulationResult, run_simulation


# ---------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH SIMULATION
# ---------------------------------------------------------------------------

DEFAULT_SIMULATION_NAME = "gold sphere 40 nm"

# Every length is in micrometres: 0.040 um = 40 nm.
SIMULATION_CONFIG: dict[str, Any] = {
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
        "save_summary": True,
        "save_spectra": True,
        "save_epsilon": True,
        "save_fields": True,
        "save_images": True,
        "field_components": ["Ez"],
    },
    "notes": "",
    "tags": ["single"],
}


# ---------------------------------------------------------------------------
# RUNNER CODE -- normally you do not need to edit below this line
# ---------------------------------------------------------------------------

def safe_name(name: str) -> str:
    """Convert a display name into a safe, readable filename component."""

    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("The simulation name must contain at least one letter or number.")
    return cleaned[:70]


def make_run_id(name: str) -> str:
    """Return ``readable_name__UTC_timestamp`` for collision-safe storage."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_name(name)}__{timestamp}"


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _flatten(prefix: str, value: Any) -> dict[str, Any]:
    """Flatten nested settings into Excel-friendly columns."""

    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, item))
        return flattened
    return {prefix: _csv_value(value)}


def export_single_csv(result: SimulationResult) -> Path:
    """Save all scalar settings, metrics and spectral samples in one CSV."""

    if not result.output_directory:
        raise RuntimeError("The master MEEP process did not return an output directory.")

    destination = Path(result.output_directory) / f"{result.run_id}_all_results.csv"
    config_columns = {
        f"config.{key}": value
        for key, value in _flatten("", result.config).items()
    }
    metric_columns = {
        f"metric.{key}": value for key, value in sorted(result.metrics.items())
    }
    monitor_names = sorted(result.spectra)
    sample_count = max(
        [len(result.frequencies), len(result.wavelengths_um)]
        + [len(result.spectra[name]) for name in monitor_names]
    )

    fixed_columns = [
        "simulation_name",
        "run_id",
        "status",
        "geometry_name",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "meep_time",
        "timesteps",
        "output_directory",
        "sample_index",
        "frequency",
        "wavelength_um",
    ]
    fieldnames = [
        *fixed_columns,
        *sorted(config_columns),
        *sorted(metric_columns),
        *(f"spectrum.{name}" for name in monitor_names),
    ]
    common = {
        "simulation_name": result.config.get("run_name", ""),
        "run_id": result.run_id,
        "status": result.status,
        "geometry_name": result.geometry_name,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "elapsed_seconds": result.elapsed_seconds,
        "meep_time": result.meep_time,
        "timesteps": result.timesteps,
        "output_directory": result.output_directory,
        **config_columns,
        **metric_columns,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig lets Excel recognise UTF-8 automatically.
    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(max(sample_count, 1)):
            row = {
                **common,
                "sample_index": index if sample_count else "",
                "frequency": _at(result.frequencies, index),
                "wavelength_um": _at(result.wavelengths_um, index),
            }
            for name in monitor_names:
                row[f"spectrum.{name}"] = _at(result.spectra[name], index)
            writer.writerow(row)
    return destination


def _at(values: Sequence[Any], index: int) -> Any:
    return values[index] if index < len(values) else ""


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default=DEFAULT_SIMULATION_NAME,
        help="Readable simulation name (default: value set at the top of the file).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    display_name = args.name.strip()
    if not display_name:
        raise SystemExit("Simulation name cannot be empty.")

    config = deepcopy(SIMULATION_CONFIG)
    run_id = make_run_id(display_name)
    config["run_name"] = display_name
    config["run_id"] = run_id
    config["tags"] = list(dict.fromkeys([*config.get("tags", []), f"name:{safe_name(display_name)}"]))

    print(f"Starting simulation: {display_name}")
    print(f"Run ID / results name: {run_id}")
    result = run_simulation(config)
    csv_path = export_single_csv(result)

    print(f"Status: {result.status}")
    print(f"Results folder: {result.output_directory}")
    print(f"Excel-ready CSV: {csv_path}")


if __name__ == "__main__":
    main()
