"""
Parameter-sweep engine for :mod:`surface_plasmons.simulation`.
Sweeps can vary nested configuration keys such as
``geometry_parameters.radius`` or ``source.wavelength_range_um``.  Serial
execution is the safe default for memory-intensive MEEP jobs; independent
process execution is available when explicitly requested.
"""

from __future__ import annotations

from concurrent.futures import as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
import csv
import hashlib
import json
import math
import re
import traceback
import uuid

from .database import SimulationDatabase, canonical_json, find_project_root
from .simulation import run_simulation, validate_config
from .parallel import worker_pool


class SweepError(ValueError):
    """Raised for an invalid sweep configuration or parameter path."""


@dataclass(frozen=True, slots=True)
class SweepAxis:
    """One nested simulation setting and the values it should take."""

    path: str
    values: tuple[Any, ...]
    label: str | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise SweepError("Sweep-axis path must be a non-empty string.")
        if not self.values:
            raise SweepError(f"Sweep axis {self.path!r} has no values.")
        _parse_path(self.path)

    @classmethod
    def from_value(cls, value: "SweepAxis | Mapping[str, Any]") -> "SweepAxis":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise SweepError("Each sweep axis must be a SweepAxis or mapping.")
        unknown = set(value) - {"path", "values", "label", "unit"}
        if unknown:
            raise SweepError(f"Unknown sweep-axis field(s): {sorted(unknown)}.")
        values = value.get("values")
        if isinstance(values, str) or not isinstance(values, Iterable):
            raise SweepError("Sweep-axis values must be a non-string iterable.")
        return cls(
            path=str(value.get("path", "")),
            values=tuple(values),
            label=str(value["label"]) if value.get("label") is not None else None,
            unit=str(value["unit"]) if value.get("unit") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class SweepConfig:
    """Complete configuration for a grid or paired parameter sweep."""

    base_config: dict[str, Any]
    axes: tuple[SweepAxis, ...]
    mode: str = "grid"                 # grid or zip
    name: str = "parameter_sweep"
    sweep_id: str | None = None
    workers: int = 1
    repetitions: int = 1
    continue_on_error: bool = True
    resume: bool = True
    database_path: str | None = None
    output_root: str | None = None
    simulation_results_root: str | None = None
    save_manifest: bool = True
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SweepConfig":
        if not isinstance(value, Mapping):
            raise SweepError("Sweep configuration must be a mapping.")
        data = dict(value)
        parameters = data.pop("parameters", None)
        raw_axes = data.pop("axes", None)
        if raw_axes is not None and parameters is not None:
            raise SweepError("Supply either axes or parameters, not both.")
        if parameters is not None:
            if not isinstance(parameters, Mapping):
                raise SweepError("parameters must map nested paths to value sequences.")
            raw_axes = [
                {"path": path, "values": values}
                for path, values in parameters.items()
            ]
        if raw_axes is None:
            raise SweepError("Sweep configuration requires axes or parameters.")
        if "base_config" not in data:
            raise SweepError("Sweep configuration requires base_config.")
        if isinstance(raw_axes, Mapping):
            raw_axes = [
                {"path": path, "values": values}
                for path, values in raw_axes.items()
            ]
        data["axes"] = tuple(SweepAxis.from_value(axis) for axis in raw_axes)
        if "base_config" in data:
            data["base_config"] = deepcopy(dict(data["base_config"]))
        if "tags" in data:
            data["tags"] = tuple(str(tag) for tag in data["tags"])
        allowed = set(cls.__dataclass_fields__)
        unknown = set(data) - allowed
        if unknown:
            raise SweepError(f"Unknown sweep setting(s): {sorted(unknown)}.")
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if not isinstance(self.base_config, Mapping):
            raise SweepError("base_config must be a simulation configuration mapping.")
        self.axes = tuple(SweepAxis.from_value(axis) for axis in self.axes)
        if not self.axes:
            raise SweepError("At least one sweep axis is required.")
        paths = [axis.path for axis in self.axes]
        if len(paths) != len(set(paths)):
            raise SweepError("Sweep-axis paths must be unique.")
        if self.mode not in {"grid", "zip"}:
            raise SweepError("mode must be 'grid' or 'zip'.")
        if self.mode == "zip" and len({len(axis.values) for axis in self.axes}) != 1:
            raise SweepError("All axes in zip mode must have the same number of values.")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise SweepError("workers must be a positive integer.")
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions < 1
        ):
            raise SweepError("repetitions must be a positive integer.")
        for name in ("continue_on_error", "resume", "save_manifest"):
            if not isinstance(getattr(self, name), bool):
                raise SweepError(f"{name} must be true or false.")
        if not self.name.strip():
            raise SweepError("name must not be empty.")
        if self.sweep_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", self.sweep_id):
            raise SweepError("sweep_id may contain only letters, digits, '.', '_', and '-'.")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class SweepRun:
    """Outcome of one point in a sweep."""

    index: int
    repetition: int
    status: str
    parameters: dict[str, Any]
    run_id: str | None
    config_hash: str
    elapsed_seconds: float
    output_directory: str | None
    metrics: dict[str, float]
    error_type: str | None = None
    error_message: str | None = None
    traceback_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(slots=True)
class SweepResult:
    """Complete sweep result plus paths to its continuously written manifests."""

    sweep_id: str
    name: str
    status: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    mode: str
    axes: tuple[SweepAxis, ...]
    runs: list[SweepRun]
    output_directory: str
    manifest_path: str | None = None
    table_path: str | None = None

    @property
    def completed_runs(self) -> int:
        return sum(run.status in {"completed", "skipped"} for run in self.runs)

    @property
    def failed_runs(self) -> int:
        return sum(run.status == "failed" for run in self.runs)

    def to_dict(self) -> dict[str, Any]:
        result = _jsonable(asdict(self))
        result["completed_runs"] = self.completed_runs
        result["failed_runs"] = self.failed_runs
        return result


def linear_axis(
    path: str,
    start: float,
    stop: float,
    points: int,
    *,
    label: str | None = None,
    unit: str | None = None,
) -> SweepAxis:
    """Create an evenly spaced numerical sweep axis without requiring NumPy."""

    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise SweepError("points must be a positive integer.")
    if points == 1:
        values = (float(start),)
    else:
        values = tuple(
            float(start) + (float(stop) - float(start)) * index / (points - 1)
            for index in range(points)
        )
    return SweepAxis(path, values, label=label, unit=unit)


def logarithmic_axis(
    path: str,
    start: float,
    stop: float,
    points: int,
    *,
    base: float = 10.0,
    label: str | None = None,
    unit: str | None = None,
) -> SweepAxis:
    """Create a logarithmically spaced positive numerical sweep axis."""

    if start <= 0 or stop <= 0 or base <= 0 or base == 1:
        raise SweepError("Logarithmic-axis bounds must be positive and base cannot be 1.")
    if isinstance(points, bool) or not isinstance(points, int) or points < 1:
        raise SweepError("points must be a positive integer.")
    log_start = math.log(float(start), base)
    log_stop = math.log(float(stop), base)
    linear = linear_axis(path, log_start, log_stop, points, label=label, unit=unit)
    return SweepAxis(
        path,
        tuple(base**value for value in linear.values),
        label=label,
        unit=unit,
    )


def expand_sweep(config: SweepConfig | Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], int]]:
    """Expand a sweep into ``(simulation_config, parameters, repetition)`` rows."""

    sweep = _coerce_sweep_config(config)
    if sweep.mode == "grid":
        combinations = product(*(axis.values for axis in sweep.axes))
    else:
        combinations = zip(*(axis.values for axis in sweep.axes))
    expanded = []
    for combination in combinations:
        parameter_values = {
            axis.path: deepcopy(value)
            for axis, value in zip(sweep.axes, combination)
        }
        for repetition in range(sweep.repetitions):
            candidate = deepcopy(dict(sweep.base_config))
            candidate.pop("run_id", None)
            for path, value in parameter_values.items():
                set_config_value(candidate, path, deepcopy(value))
            expanded.append((candidate, deepcopy(parameter_values), repetition))
    return expanded


def set_config_value(config: dict[str, Any], path: str, value: Any) -> None:
    """Set a dot/bracket path, creating missing dictionaries when safe.

    Examples: ``geometry_parameters.radius``, ``source.component``, and
    ``geometry_parameters.layers[0].thickness``.
    """

    tokens = _parse_path(path)
    current: Any = config
    for index, token in enumerate(tokens[:-1]):
        next_token = tokens[index + 1]
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                raise SweepError(f"List index {token} is unavailable in path {path!r}.")
            current = current[token]
        else:
            if not isinstance(current, dict):
                raise SweepError(f"Cannot enter {token!r} in path {path!r}.")
            if token not in current:
                if isinstance(next_token, int):
                    raise SweepError(
                        f"Cannot create missing list {token!r} in path {path!r}."
                    )
                current[token] = {}
            current = current[token]
    final = tokens[-1]
    if isinstance(final, int):
        if not isinstance(current, list) or final >= len(current):
            raise SweepError(f"List index {final} is unavailable in path {path!r}.")
        current[final] = value
    else:
        if not isinstance(current, dict):
            raise SweepError(f"Cannot set {final!r} in path {path!r}.")
        current[final] = value


def run_sweep(config: SweepConfig | Mapping[str, Any], *,
              worker_initializer=None, worker_initargs=()) -> SweepResult:
    """Execute a parameter sweep and continuously save JSON/CSV manifests."""

    sweep = _coerce_sweep_config(config)
    identity = sweep.sweep_id or _new_sweep_id(sweep.name)
    output_directory = _sweep_output_directory(sweep, identity)
    output_directory.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    start_clock = perf_counter()
    manifest_path = output_directory / "sweep.json" if sweep.save_manifest else None
    table_path = output_directory / "runs.csv" if sweep.save_manifest else None
    records_enabled = bool(sweep.base_config.get("record_database", True))
    effective_database_path = sweep.database_path or sweep.base_config.get("database_path")
    database = (
        SimulationDatabase(effective_database_path)
        if sweep.resume and records_enabled
        else None
    )

    prepared: list[tuple[int, int, dict[str, Any], dict[str, Any], str]] = []
    completed: dict[int, SweepRun] = {}
    for index, (candidate, parameters, repetition) in enumerate(expand_sweep(sweep)):
        _decorate_candidate(candidate, sweep, identity, index, repetition)
        normalized = validate_config(candidate).to_dict()
        digest = hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()
        previous = database.find_by_config(normalized, status="completed") if database else []
        if previous:
            row = previous[0]
            previous_metrics = {
                str(metric["name"]): float(metric["value"])
                for metric in database.get_metrics(str(row["run_id"]))
            }
            completed[index] = SweepRun(
                index=index,
                repetition=repetition,
                status="skipped",
                parameters=parameters,
                run_id=str(row["run_id"]),
                config_hash=digest,
                elapsed_seconds=0.0,
                output_directory=row.get("output_directory"),
                metrics=previous_metrics,
            )
        else:
            prepared.append((index, repetition, normalized, parameters, digest))

    partial = SweepResult(
        sweep_id=identity,
        name=sweep.name,
        status="running",
        started_at=started_at,
        finished_at=started_at,
        elapsed_seconds=0.0,
        mode=sweep.mode,
        axes=sweep.axes,
        runs=[completed[index] for index in sorted(completed)],
        output_directory=str(output_directory),
        manifest_path=str(manifest_path) if manifest_path else None,
        table_path=str(table_path) if table_path else None,
    )
    _save_progress(partial, manifest_path, table_path)

    if sweep.workers == 1:
        for item in prepared:
            record = _execute_variant(*item)
            completed[record.index] = record
            partial.runs = [completed[index] for index in sorted(completed)]
            partial.elapsed_seconds = perf_counter() - start_clock
            _save_progress(partial, manifest_path, table_path)
            if record.status == "failed" and not sweep.continue_on_error:
                break
    else:
        # MEEP simulations are isolated in spawned processes; do not use threads.
        with worker_pool(
            sweep.workers, initializer=worker_initializer, initargs=worker_initargs,
        ) as executor:
            futures = {
                executor.submit(_execute_variant, *item): item
                for item in prepared
            }
            stop_submitting = False
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    index, repetition, _, parameters, digest = item
                    record = SweepRun(
                        index=index,
                        repetition=repetition,
                        status="failed",
                        parameters=parameters,
                        run_id=None,
                        config_hash=digest,
                        elapsed_seconds=0.0,
                        output_directory=None,
                        metrics={},
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        traceback_text=traceback.format_exc(),
                    )
                completed[record.index] = record
                partial.runs = [completed[index] for index in sorted(completed)]
                partial.elapsed_seconds = perf_counter() - start_clock
                _save_progress(partial, manifest_path, table_path)
                if record.status == "failed" and not sweep.continue_on_error:
                    stop_submitting = True
                    for pending in futures:
                        pending.cancel()
                    break
            if stop_submitting:
                pass

    runs = [completed[index] for index in sorted(completed)]
    failed = sum(run.status == "failed" for run in runs)
    expected = len(prepared) + sum(run.status == "skipped" for run in runs)
    if failed == 0 and len(runs) == expected:
        status = "completed"
    elif runs and failed < len(runs):
        status = "partial"
    else:
        status = "failed"
    result = SweepResult(
        sweep_id=identity,
        name=sweep.name,
        status=status,
        started_at=started_at,
        finished_at=_utc_now(),
        elapsed_seconds=perf_counter() - start_clock,
        mode=sweep.mode,
        axes=sweep.axes,
        runs=runs,
        output_directory=str(output_directory),
        manifest_path=str(manifest_path) if manifest_path else None,
        table_path=str(table_path) if table_path else None,
    )
    _save_progress(result, manifest_path, table_path)
    return result


def parameter_sweep(
    base_config: Mapping[str, Any],
    parameters: Mapping[str, Sequence[Any]],
    *,
    worker_initializer=None,
    worker_initargs=(),
    **options: Any,
) -> SweepResult:
    """Compact wrapper around :func:`run_sweep`."""

    return run_sweep(
        SweepConfig.from_dict(
            {"base_config": dict(base_config), "parameters": parameters, **options}
        ),
        worker_initializer=worker_initializer, worker_initargs=worker_initargs,
    )


def load_sweep(path: str | Path) -> SweepResult:
    """Load a saved ``sweep.json`` manifest."""

    source = Path(path).expanduser().resolve()
    if source.is_dir():
        source = source / "sweep.json"
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SweepError(f"Could not load sweep manifest {source}: {exc}") from exc
    axes = tuple(SweepAxis.from_value(value) for value in payload.get("axes", []))
    runs = [
        SweepRun(
            index=int(value["index"]),
            repetition=int(value.get("repetition", 0)),
            status=str(value["status"]),
            parameters=dict(value.get("parameters", {})),
            run_id=value.get("run_id"),
            config_hash=str(value.get("config_hash", "")),
            elapsed_seconds=float(value.get("elapsed_seconds", 0.0)),
            output_directory=value.get("output_directory"),
            metrics={str(key): float(item) for key, item in value.get("metrics", {}).items()},
            error_type=value.get("error_type"),
            error_message=value.get("error_message"),
            traceback_text=value.get("traceback_text"),
        )
        for value in payload.get("runs", [])
    ]
    return SweepResult(
        sweep_id=str(payload["sweep_id"]),
        name=str(payload.get("name", "parameter_sweep")),
        status=str(payload.get("status", "unknown")),
        started_at=str(payload.get("started_at", "")),
        finished_at=str(payload.get("finished_at", "")),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        mode=str(payload.get("mode", "grid")),
        axes=axes,
        runs=runs,
        output_directory=str(payload.get("output_directory", source.parent)),
        manifest_path=str(payload.get("manifest_path", source)),
        table_path=payload.get("table_path"),
    )


def _execute_variant(
    index: int,
    repetition: int,
    config: dict[str, Any],
    parameters: dict[str, Any],
    digest: str,
) -> SweepRun:
    started = perf_counter()
    try:
        result = run_simulation(config)
        return SweepRun(
            index=index,
            repetition=repetition,
            status="completed",
            parameters=parameters,
            run_id=result.run_id,
            config_hash=digest,
            elapsed_seconds=perf_counter() - started,
            output_directory=result.output_directory,
            metrics={key: float(value) for key, value in result.metrics.items()},
        )
    except Exception as exc:
        return SweepRun(
            index=index,
            repetition=repetition,
            status="failed",
            parameters=parameters,
            run_id=None,
            config_hash=digest,
            elapsed_seconds=perf_counter() - started,
            output_directory=None,
            metrics={},
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
        )


def _decorate_candidate(
    candidate: dict[str, Any],
    sweep: SweepConfig,
    sweep_id: str,
    index: int,
    repetition: int,
) -> None:
    candidate["run_name"] = (
        candidate.get("run_name")
        or f"{sweep.name}-{index + 1:04d}-r{repetition + 1}"
    )
    tags = [str(tag) for tag in candidate.get("tags", [])]
    tags.extend(sweep.tags)
    tags.append(f"sweep:{sweep_id}")
    candidate["tags"] = list(dict.fromkeys(tags))
    if sweep.database_path is not None:
        candidate["database_path"] = sweep.database_path
    if sweep.simulation_results_root is not None:
        output = dict(candidate.get("output", {}))
        output["results_root"] = sweep.simulation_results_root
        candidate["output"] = output


def _save_progress(
    result: SweepResult,
    manifest_path: Path | None,
    table_path: Path | None,
) -> None:
    if manifest_path is None or table_path is None:
        return
    _atomic_json(manifest_path, result.to_dict())
    parameter_names = [axis.path for axis in result.axes]
    metric_names = sorted({key for run in result.runs for key in run.metrics})
    columns = [
        "index", "repetition", "status", "run_id", "config_hash",
        "elapsed_seconds", "output_directory", "error_type", "error_message",
        *parameter_names,
        *metric_names,
    ]
    temporary = table_path.with_suffix(table_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in result.runs:
            row = {
                "index": run.index,
                "repetition": run.repetition,
                "status": run.status,
                "run_id": run.run_id,
                "config_hash": run.config_hash,
                "elapsed_seconds": run.elapsed_seconds,
                "output_directory": run.output_directory,
                "error_type": run.error_type,
                "error_message": run.error_message,
                **{name: _csv_value(run.parameters.get(name)) for name in parameter_names},
                **run.metrics,
            }
            writer.writerow(row)
    temporary.replace(table_path)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _coerce_sweep_config(value: SweepConfig | Mapping[str, Any]) -> SweepConfig:
    if isinstance(value, SweepConfig):
        value.validate()
        return value
    return SweepConfig.from_dict(value)


def _parse_path(path: str) -> tuple[str | int, ...]:
    if not isinstance(path, str) or not path.strip():
        raise SweepError("Parameter path must not be empty.")
    tokens: list[str | int] = []
    position = 0
    pattern = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
    while position < len(path):
        if path[position] == ".":
            position += 1
            continue
        match = pattern.match(path, position)
        if match is None:
            raise SweepError(f"Invalid parameter path {path!r} near position {position}.")
        tokens.append(match.group(1) if match.group(1) is not None else int(match.group(2)))
        position = match.end()
    if not tokens or isinstance(tokens[0], int):
        raise SweepError("Parameter paths must begin with a configuration key.")
    return tuple(tokens)


def _sweep_output_directory(sweep: SweepConfig, sweep_id: str) -> Path:
    root = (
        Path(sweep.output_root).expanduser().resolve()
        if sweep.output_root
        else find_project_root() / "data" / "results" / "sweeps"
    )
    return root / sweep_id


def _new_sweep_id(name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.-") or "sweep"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{safe_name}_{timestamp}_{uuid.uuid4().hex[:8]}"


def _csv_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return canonical_json(_jsonable(value))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


__all__ = [
    "SweepAxis",
    "SweepConfig",
    "SweepError",
    "SweepResult",
    "SweepRun",
    "expand_sweep",
    "linear_axis",
    "load_sweep",
    "logarithmic_axis",
    "parameter_sweep",
    "run_sweep",
    "set_config_value",
]


# Compatibility names used by compact scripts.
run_parameter_sweep = parameter_sweep
sweep = parameter_sweep
__all__.extend(["run_parameter_sweep", "sweep"])

