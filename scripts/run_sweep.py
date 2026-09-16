"""Run a parameter sweep for research purposes, producing a single CSV of results and
automatic visualisations.  The sweep is defined by a base configuration plus a set of
independent variables.  
One active variable produces line/scatter plots.  Two variables produce 2-D
heatmaps.  Three or more variables produce pairwise heatmaps, averaged over
the other active variables.  All combinations are simulated (a Cartesian grid).
"""

from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
import math
from pathlib import Path
import re
from statistics import fmean
from typing import Any, Mapping, Sequence

import numpy as np

from surface_plasmons.sweep import SweepResult, parameter_sweep

# Both files belong in the same scripts/ folder.
from run_single import (
    RESEARCH_CSV_COLUMNS,
    analyze_research_result,
    research_rows,
    safe_name,
)


# ---------------------------------------------------------------------------
# EDIT THIS SECTION FOR EACH SWEEP
# ---------------------------------------------------------------------------

DEFAULT_SWEEP_NAME = "gold sphere radius sweep"

# Compact geometry form makes material/background axes easy to combine.
# All lengths are in micrometres: 0.040 um = 40 nm.
BASE_CONFIG: dict[str, Any] = {
    "geometry": {
        "name": "nanoparticle_sphere",
        "parameters": {
            "particle": "au",
            "background": "air",
            "radius": 0.020,
        },
    },
    # A real 3-D MEEP domain. Keep workers at 1 until memory use is known.
    "dimensions": 3,
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
        "save_summary": True,   # Required for post-sweep research analysis.
        "save_spectra": True,
        "save_epsilon": False,
        "save_fields": True,
        "save_images": True,
        # Save all electric components for total |E|^2 field enhancement.
        "field_components": ["Ex", "Ey", "Ez"],
    },
    "notes": "",
    "tags": ["sweep", "research", "3d"],
}


SWEEP_VARIABLES: dict[str, dict[str, Any]] = {
    "particle_material": {
        "path": "geometry.parameters.particle",
        "values": ["au", "ag", "al"],
        "label": "Particle material",
    },
    "background_material": {
        "path": "geometry.parameters.background",
        "values": ["air", "water", "sio2"],
        "label": "Background material",
    },
    "geometry": {
        "path": "geometry",
        "values": ["sphere", "rod", "cube", "disk"],
        "config_values": [
            {
                "name": "nanoparticle_sphere",
                "parameters": {"particle": "au", "background": "air", "radius": 0.040},
            },
            {
                "name": "nanorod",
                "parameters": {
                    "particle": "au", "background": "air",
                    "length": 0.120, "diameter": 0.040,
                },
            },
            {
                "name": "nanocube",
                "parameters": {
                    "particle": "au", "background": "air",
                    "size_x": 0.080, "size_y": 0.080, "size_z": 0.080,
                },
            },
            {
                "name": "nanoparticle_disk",
                "parameters": {
                    "particle": "au", "background": "air",
                    "radius": 0.040, "height": 0.030,
                },
            },
        ],
        "label": "Geometry / shape",
    },
    "particle_size": {
        # Change this path for a different base geometry, e.g. diameter for a nanorod.
        "path": "geometry.parameters.radius",
        "values": [0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100],
        "label": "Particle size (µm)",
    },
    "aspect_ratio": {
        # For a nanorod: length = aspect ratio × fixed diameter.
        "path": "geometry.parameters.length",
        "values": [2.0, 3.0, 4.0, 5.0],
        "transform": "nanorod_aspect_ratio",
        "label": "Aspect ratio",
    },
    "length": {
        "path": "geometry.parameters.length",
        "values": [0.080, 0.120, 0.160],
        "label": "Length (µm)",
    },
    "width": {
        # This example uses nanorod diameter as its width.
        "path": "geometry.parameters.diameter",
        "values": [0.030, 0.040, 0.050],
        "label": "Width / diameter (µm)",
    },
    "thickness": {
        # Suitable for nanoparticle_disk. Change the path for other geometries.
        "path": "geometry.parameters.height",
        "values": [0.020, 0.030, 0.040],
        "label": "Thickness (µm)",
    },
    "curvature": {
        # Example for tip_substrate; curvature is controlled by apex radius.
        "path": "geometry.parameters.apex_radius",
        "values": [0.002, 0.005, 0.010],
        "label": "Curvature radius (µm)",
    },
    "incident_wavelength": {
        # Each pair is a complete simulated wavelength band.
        "path": "source.wavelength_range_um",
        "values": [[0.40, 0.70], [0.45, 0.80], [0.50, 0.90]],
        "label": "Incident wavelength band (µm)",
    },
    "polarisation": {
        "path": "source.component",
        "values": ["Ez", "Ey"],
        "label": "Polarisation / field component",
    },
}

# ONE name = ordinary 1-D sweep.
# TWO OR MORE names = every combination (Cartesian grid).
ACTIVE_SWEEP_PARAMETERS = ["particle_size"]

REPETITIONS = 1
DEFAULT_WORKERS = 1  # MEEP is memory intensive; start with one worker.

# Every available numeric research result is plotted when data exists.
PLOT_RESULTS = {
    "resonance_wavelength_um": "Resonance wavelength (µm)",
    "peak_spectral_response": "Peak spectral response",
    "electric_field_enhancement": "Electric-field enhancement",
    "resonance_linewidth_fwhm_um": "Resonance linewidth / FWHM (µm)",
    "quality_factor": "Quality factor Q",
    "peak_absorption": "Peak absorption",
    "peak_scattering": "Peak scattering",
    "peak_extinction": "Peak extinction",
    "number_of_modes": "Number of modes",
}

# RUNNER CODE BELOW --------------------------------------------------------------

EXCEL_MAX_DATA_ROWS = 1_048_575


def make_sweep_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{safe_name(name)}__{timestamp}"


def _nested_value(config: Mapping[str, Any], path: str) -> Any:
    current: Any = config
    for token in path.split("."):
        if not isinstance(current, Mapping) or token not in current:
            raise KeyError(path)
        current = current[token]
    return current


def _value_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _display_value(value: Any) -> str:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{value[0]}–{value[1]}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def build_sweep_axes() -> tuple[dict[str, Sequence[Any]], dict[str, dict[str, Any]]]:
    """Build package sweep paths plus friendly research-axis metadata."""

    if not ACTIVE_SWEEP_PARAMETERS:
        raise ValueError("ACTIVE_SWEEP_PARAMETERS must contain at least one variable name.")
    unknown = [name for name in ACTIVE_SWEEP_PARAMETERS if name not in SWEEP_VARIABLES]
    if unknown:
        raise ValueError(f"Unknown active sweep variable(s): {unknown}")

    active = list(dict.fromkeys(ACTIVE_SWEEP_PARAMETERS))
    # A full geometry preset must be installed before material/background overrides.
    if "geometry" in active:
        active.remove("geometry")
        active.insert(0, "geometry")
    geometry_specific = {"particle_size", "aspect_ratio", "length", "width", "thickness", "curvature"}
    if "geometry" in active and geometry_specific.intersection(active):
        raise ValueError(
            "A geometry grid cannot be combined directly with geometry-specific dimensions, "
            "because sphere/rod/cube/disk use different parameter names. Run one geometry at "
            "a time for size/aspect-ratio studies, or combine geometry with material, "
            "background, wavelength, or polarisation."
        )

    parameters: dict[str, Sequence[Any]] = {}
    runtime: dict[str, dict[str, Any]] = {}
    used_paths: set[str] = set()
    for name in active:
        spec = deepcopy(SWEEP_VARIABLES[name])
        path = str(spec["path"])
        display_values = list(spec["values"])
        config_values = list(spec.get("config_values", display_values))
        if len(display_values) != len(config_values) or not display_values:
            raise ValueError(f"Sweep variable {name!r} has invalid values/config_values.")

        if spec.get("transform") == "nanorod_aspect_ratio":
            geometry_name = _nested_value(BASE_CONFIG, "geometry.name")
            if geometry_name != "nanorod":
                raise ValueError(
                    "The aspect_ratio sweep requires BASE_CONFIG geometry.name = 'nanorod'."
                )
            diameter = float(_nested_value(BASE_CONFIG, "geometry.parameters.diameter"))
            config_values = [float(ratio) * diameter for ratio in display_values]

        if path in used_paths:
            raise ValueError(
                f"Two active variables target {path!r}. Choose only one of them."
            )
        used_paths.add(path)
        parameters[path] = config_values
        runtime[name] = {
            "path": path,
            "label": spec.get("label", name),
            "display_values": display_values,
            "config_values": config_values,
            "lookup": {
                _value_key(config_value): display_value
                for config_value, display_value in zip(config_values, display_values)
            },
        }
    return parameters, runtime


def _friendly_parameters(run_parameters: Mapping[str, Any], runtime: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, axis in runtime.items():
        actual = run_parameters.get(str(axis["path"]))
        values[name] = axis["lookup"].get(_value_key(actual), actual)
    return values


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


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def export_research_results(
    result: SweepResult,
    runtime: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, int, list[dict[str, Any]], list[str]]:
    """Create one long-format CSV containing only the selected research variables."""

    destination = Path(result.output_directory) / f"{result.sweep_id}_research_results.csv"
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    row_count = 0
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESEARCH_CSV_COLUMNS)
        writer.writeheader()
        for run in result.runs:
            if run.status not in {"completed", "skipped"}:
                warnings.append(
                    f"Sweep point {run.index + 1} failed and was omitted from the research CSV."
                )
                continue
            summary = _read_summary(run.output_directory)
            if not summary:
                warnings.append(
                    f"Sweep point {run.index + 1} had no readable summary.json and was omitted."
                )
                continue

            analysis = analyze_research_result(summary, create_field_plot=True)
            rows = research_rows(analysis)
            writer.writerows(rows)
            row_count += len(rows)
            record = {
                **_friendly_parameters(run.parameters, runtime),
                **analysis["summary"],
                "simulation_name": analysis["simulation_name"],
                "run_id": analysis["run_id"],
            }
            records.append(record)
            warnings.extend(
                f"{analysis['simulation_name'] or analysis['run_id']}: {message}"
                for message in analysis["warnings"]
            )
    return destination, row_count, records, warnings


def _axis_positions(values: Sequence[Any]) -> tuple[np.ndarray, list[str], bool]:
    numbers = [_number(value) for value in values]
    numeric = all(value is not None for value in numbers)
    if numeric:
        return np.asarray(numbers, dtype=float), [_display_value(value) for value in values], True
    return np.arange(len(values), dtype=float), [_display_value(value) for value in values], False


def _group_metric(records: Sequence[Mapping[str, Any]], axis: str, values: Sequence[Any], metric: str) -> tuple[np.ndarray, np.ndarray]:
    means = np.full(len(values), np.nan)
    spread = np.full(len(values), np.nan)
    for index, expected in enumerate(values):
        samples = [
            number
            for record in records
            if _value_key(record.get(axis)) == _value_key(expected)
            for number in [_number(record.get(metric))]
            if number is not None
        ]
        if samples:
            means[index] = fmean(samples)
            spread[index] = float(np.std(samples))
    return means, spread


def _plot_1d(
    records: Sequence[Mapping[str, Any]],
    axis_name: str,
    axis: Mapping[str, Any],
    metric: str,
    metric_label: str,
    output_directory: Path,
) -> Path | None:
    values = list(axis["display_values"])
    means, spread = _group_metric(records, axis_name, values, metric)
    finite = np.isfinite(means)
    if not np.any(finite):
        return None

    import matplotlib.pyplot as plt

    x, tick_labels, numeric = _axis_positions(values)
    figure, plot_axis = plt.subplots(figsize=(7.2, 5.0))
    if numeric:
        plot_axis.plot(x[finite], means[finite], marker="o", linewidth=1.8)
    else:
        plot_axis.scatter(x[finite], means[finite], s=42)
    spread_finite = finite & np.isfinite(spread) & (spread > 0)
    if np.any(spread_finite):
        plot_axis.errorbar(
            x[spread_finite], means[spread_finite], yerr=spread[spread_finite],
            fmt="none", capsize=3, color="#555555",
        )
    if not numeric:
        plot_axis.set_xticks(x, tick_labels, rotation=25, ha="right")
    plot_axis.set_xlabel(str(axis["label"]))
    plot_axis.set_ylabel(metric_label)
    plot_axis.set_title(f"{metric_label} vs {axis['label']}")
    plot_axis.grid(alpha=0.22)
    figure.tight_layout()
    destination = output_directory / f"{safe_name(metric)}_vs_{safe_name(axis_name)}.png"
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return destination


def _plot_heatmap(
    records: Sequence[Mapping[str, Any]],
    x_name: str,
    x_axis: Mapping[str, Any],
    y_name: str,
    y_axis: Mapping[str, Any],
    metric: str,
    metric_label: str,
    output_directory: Path,
    averaged: bool,
) -> Path | None:
    x_values = list(x_axis["display_values"])
    y_values = list(y_axis["display_values"])
    matrix = np.full((len(y_values), len(x_values)), np.nan)
    for row, y_value in enumerate(y_values):
        for column, x_value in enumerate(x_values):
            samples = [
                number
                for record in records
                if _value_key(record.get(x_name)) == _value_key(x_value)
                and _value_key(record.get(y_name)) == _value_key(y_value)
                for number in [_number(record.get(metric))]
                if number is not None
            ]
            if samples:
                matrix[row, column] = fmean(samples)
    if not np.any(np.isfinite(matrix)):
        return None

    import matplotlib.pyplot as plt

    figure, plot_axis = plt.subplots(figsize=(8.0, 6.0))
    image = plot_axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    plot_axis.set_xticks(range(len(x_values)), [_display_value(value) for value in x_values], rotation=25, ha="right")
    plot_axis.set_yticks(range(len(y_values)), [_display_value(value) for value in y_values])
    plot_axis.set_xlabel(str(x_axis["label"]))
    plot_axis.set_ylabel(str(y_axis["label"]))
    suffix = " (mean over other grid variables)" if averaged else ""
    plot_axis.set_title(metric_label + suffix)
    figure.colorbar(image, ax=plot_axis, label=metric_label)
    figure.tight_layout()
    destination = output_directory / (
        f"{safe_name(metric)}_grid_{safe_name(x_name)}_by_{safe_name(y_name)}.png"
    )
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return destination


def _plot_mode_locations(
    records: Sequence[Mapping[str, Any]],
    axis_name: str,
    axis: Mapping[str, Any],
    output_directory: Path,
) -> Path | None:
    values = list(axis["display_values"])
    x, labels, numeric = _axis_positions(values)
    points_x: list[float] = []
    points_y: list[float] = []
    for record in records:
        try:
            modes = json.loads(str(record.get("mode_locations_um", "[]")))
        except json.JSONDecodeError:
            continue
        match = next(
            (index for index, value in enumerate(values) if _value_key(value) == _value_key(record.get(axis_name))),
            None,
        )
        if match is None:
            continue
        for wavelength in modes:
            number = _number(wavelength)
            if number is not None:
                points_x.append(float(x[match]))
                points_y.append(number * 1000.0)
    if not points_y:
        return None

    import matplotlib.pyplot as plt

    figure, plot_axis = plt.subplots(figsize=(7.2, 5.0))
    plot_axis.scatter(points_x, points_y, alpha=0.75)
    if not numeric:
        plot_axis.set_xticks(x, labels, rotation=25, ha="right")
    plot_axis.set_xlabel(str(axis["label"]))
    plot_axis.set_ylabel("Mode location (nm)")
    plot_axis.set_title(f"Mode locations vs {axis['label']}")
    plot_axis.grid(alpha=0.22)
    figure.tight_layout()
    destination = output_directory / f"mode_locations_vs_{safe_name(axis_name)}.png"
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return destination


def create_sweep_visualisations(
    records: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Mapping[str, Any]],
    output_directory: str | Path,
) -> list[Path]:
    """Create 1-D plots or multi-variable grid heatmaps automatically."""

    if not records:
        return []
    import matplotlib

    matplotlib.use("Agg")
    plot_directory = Path(output_directory) / "research_plots"
    plot_directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    names = list(runtime)

    if len(names) == 1:
        name = names[0]
        for metric, label in PLOT_RESULTS.items():
            path = _plot_1d(records, name, runtime[name], metric, label, plot_directory)
            if path:
                paths.append(path)
    else:
        for x_name, y_name in combinations(names, 2):
            for metric, label in PLOT_RESULTS.items():
                path = _plot_heatmap(
                    records,
                    x_name,
                    runtime[x_name],
                    y_name,
                    runtime[y_name],
                    metric,
                    label,
                    plot_directory,
                    averaged=len(names) > 2,
                )
                if path:
                    paths.append(path)

    # Mode-location plots remain readable for both 1-D and multi-variable grids.
    for name in names:
        path = _plot_mode_locations(records, name, runtime[name], plot_directory)
        if path:
            paths.append(path)
    return paths


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=DEFAULT_SWEEP_NAME)
    parser.add_argument("--sweep-id")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
        raise SystemExit("--sweep-id contains unsupported characters.")

    try:
        parameters, runtime = build_sweep_axes()
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid sweep setup: {exc}") from exc

    base_config = deepcopy(BASE_CONFIG)
    base_config["tags"] = list(
        dict.fromkeys([*base_config.get("tags", []), f"sweep:{safe_name(display_name)}"])
    )
    grid_points = math.prod(len(axis["config_values"]) for axis in runtime.values())
    total_runs = grid_points * REPETITIONS
    print(f"Starting sweep: {display_name}")
    print(f"Active variables: {', '.join(runtime)}")
    print(f"Grid points: {grid_points}; total simulations: {total_runs}")
    print(f"Sweep ID / results name: {sweep_id}")

    result = parameter_sweep(
        base_config=base_config,
        parameters=parameters,
        mode="grid",
        name=display_name,
        sweep_id=sweep_id,
        workers=args.workers,
        repetitions=REPETITIONS,
        continue_on_error=not args.stop_on_error,
        resume=not args.no_resume,
    )

    csv_path, row_count, records, warnings = export_research_results(result, runtime)
    plot_paths = create_sweep_visualisations(records, runtime, result.output_directory)

    print(f"Status: {result.status}")
    print(f"Completed/skipped runs: {result.completed_runs}")
    print(f"Failed runs: {result.failed_runs}")
    print(f"Research CSV: {csv_path}")
    print(f"Research CSV rows: {row_count}")
    print(f"Sweep visualisations: {len(plot_paths)} in {Path(result.output_directory) / 'research_plots'}")
    if row_count > EXCEL_MAX_DATA_ROWS:
        print("Warning: the CSV exceeds the row limit of one Excel worksheet.")
    for warning in dict.fromkeys(warnings):
        print(f"Warning: {warning}")

    if result.status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    # Required when workers > 1.
    main()
