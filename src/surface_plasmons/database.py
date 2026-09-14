"""SQLite persistence layer for surface-plasmon simulations.
This module stores *individual runs*: their exact configuration, lifecycle, spectra,
metrics, artifacts, and diagnostic events.

``SimulationDatabase`` creates ``data/simulations.db`` automatically and is safe to call repeatedly.  
Connections are short-lived, foreign keys are enabled, and write-ahead logging is used so result readers 
do not block a running sweep.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
import json
import os
import platform
import sqlite3
import uuid


SCHEMA_VERSION = 1
VALID_STATUSES = frozenset({"queued", "running", "completed", "failed", "cancelled"})


class DatabaseError(RuntimeError):
    """Base error for invalid database operations."""


class RunNotFoundError(DatabaseError):
    """Raised when a requested run identifier does not exist."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the project containing ``pyproject.toml`` and ``src``."""

    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).expanduser().resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve()])

    seen: set[Path] = set()
    for candidate in candidates:
        current = candidate if candidate.is_dir() else candidate.parent
        for parent in (current, *current.parents):
            if parent in seen:
                continue
            seen.add(parent)
            if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
                return parent

    # For the normal src-layout this is Surface_Plasmons/src/surface_plasmons/database.py.
    return Path(__file__).resolve().parents[2]


def default_database_path() -> Path:
    override = os.environ.get("SURFACE_PLASMONS_DB")
    if override:
        return Path(override).expanduser().resolve()
    return find_project_root() / "data" / "simulations.db"


def new_run_id() -> str:
    return uuid.uuid4().hex


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def configuration_hash(config: Mapping[str, Any]) -> str:
    return sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _decode_json(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS simulations (
    run_id TEXT PRIMARY KEY,
    name TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled')
    ),
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    elapsed_seconds REAL,
    geometry_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions IN (1, 2, 3)),
    wavelength_min_um REAL,
    wavelength_max_um REAL,
    resolution REAL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    output_directory TEXT,
    notes TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    parent_run_id TEXT,
    code_version TEXT,
    meep_version TEXT,
    python_version TEXT,
    host_name TEXT,
    error_type TEXT,
    error_message TEXT,
    traceback_text TEXT,
    FOREIGN KEY (parent_run_id) REFERENCES simulations(run_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_simulations_status
    ON simulations(status);
CREATE INDEX IF NOT EXISTS idx_simulations_created_at
    ON simulations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_simulations_geometry
    ON simulations(geometry_name);
CREATE INDEX IF NOT EXISTS idx_simulations_config_hash
    ON simulations(config_hash);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    wavelength_um REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES simulations(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_name
    ON metrics(run_id, name);

CREATE TABLE IF NOT EXISTS spectra (
    spectrum_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    monitor_name TEXT NOT NULL,
    sample_index INTEGER NOT NULL,
    frequency REAL NOT NULL,
    wavelength_um REAL NOT NULL,
    value_real REAL NOT NULL,
    value_imag REAL NOT NULL DEFAULT 0.0,
    unit TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES simulations(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, monitor_name, sample_index)
);

CREATE INDEX IF NOT EXISTS idx_spectra_run_monitor
    ON spectra(run_id, monitor_name, sample_index);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    mime_type TEXT,
    size_bytes INTEGER,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES simulations(run_id) ON DELETE CASCADE,
    UNIQUE (run_id, path)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind
    ON artifacts(run_id, kind);

CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    event TEXT NOT NULL,
    message TEXT,
    data_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES simulations(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_run_time
    ON events(run_id, event_id);
"""


class SimulationDatabase:
    """High-level API for the project's SQLite run database."""

    def __init__(self, path: str | Path | None = None, *, timeout: float = 30.0):
        self.path = Path(path).expanduser().resolve() if path else default_database_path()
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise DatabaseError("timeout must be greater than zero.")
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current_version > SCHEMA_VERSION:
                raise DatabaseError(
                    f"Database schema {current_version} is newer than supported "
                    f"schema {SCHEMA_VERSION}. Update surface_plasmons."
                )
            connection.executescript(_SCHEMA)
            connection.execute(
                "INSERT OR IGNORE INTO schema_info(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, utc_now()),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        return self.path

    def create_run(
        self,
        config: Mapping[str, Any],
        *,
        run_id: str | None = None,
        name: str | None = None,
        status: str = "queued",
        output_directory: str | Path | None = None,
        notes: str | None = None,
        tags: Sequence[str] = (),
        parent_run_id: str | None = None,
        code_version: str | None = None,
        meep_version: str | None = None,
    ) -> str:
        if status not in VALID_STATUSES:
            raise DatabaseError(f"Invalid status {status!r}.")
        if not isinstance(config, Mapping):
            raise DatabaseError("config must be a mapping.")

        identity = run_id or new_run_id()
        geometry_name = str(
            config.get("geometry_name", config.get("geometry", config.get("type", "unknown")))
        )
        geometry_parameters = config.get("geometry_parameters", config.get("parameters", {}))
        dimensions = int(
            config.get(
                "dimensions",
                geometry_parameters.get("dimensions", 2)
                if isinstance(geometry_parameters, Mapping)
                else 2,
            )
        )
        wavelengths = _extract_wavelength_range(config)
        resolution = config.get("resolution")
        created_at = utc_now()
        started_at = created_at if status == "running" else None
        finished_at = created_at if status in {"completed", "failed", "cancelled"} else None
        config_text = canonical_json(config)

        try:
            with self.transaction(immediate=True) as connection:
                connection.execute(
                    """
                    INSERT INTO simulations (
                        run_id, name, status, created_at, started_at, finished_at,
                        geometry_name, dimensions, wavelength_min_um,
                        wavelength_max_um, resolution, config_json, config_hash,
                        output_directory, notes, tags_json, parent_run_id,
                        code_version, meep_version, python_version, host_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity,
                        name,
                        status,
                        created_at,
                        started_at,
                        finished_at,
                        geometry_name,
                        dimensions,
                        wavelengths[0],
                        wavelengths[1],
                        float(resolution) if resolution is not None else None,
                        config_text,
                        sha256(config_text.encode("utf-8")).hexdigest(),
                        str(Path(output_directory).resolve()) if output_directory else None,
                        notes,
                        canonical_json(list(tags)),
                        parent_run_id,
                        code_version,
                        meep_version,
                        platform.python_version(),
                        platform.node(),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO events(run_id, timestamp, level, event, message, data_json)
                    VALUES (?, ?, 'INFO', 'created', 'Simulation run created', '{}')
                    """,
                    (identity, created_at),
                )
        except sqlite3.IntegrityError as exc:
            raise DatabaseError(f"Could not create run {identity!r}: {exc}") from exc
        return identity

    def update_status(
        self,
        run_id: str,
        status: str,
        *,
        elapsed_seconds: float | None = None,
        output_directory: str | Path | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        traceback_text: str | None = None,
    ) -> None:
        if status not in VALID_STATUSES:
            raise DatabaseError(f"Invalid status {status!r}.")
        now = utc_now()
        fields: dict[str, Any] = {"status": status}
        if status == "running":
            fields["started_at"] = now
        if status in {"completed", "failed", "cancelled"}:
            fields["finished_at"] = now
        if elapsed_seconds is not None:
            fields["elapsed_seconds"] = float(elapsed_seconds)
        if output_directory is not None:
            fields["output_directory"] = str(Path(output_directory).resolve())
        if error_type is not None:
            fields["error_type"] = error_type
        if error_message is not None:
            fields["error_message"] = error_message
        if traceback_text is not None:
            fields["traceback_text"] = traceback_text

        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [*fields.values(), run_id]
        with self.transaction(immediate=True) as connection:
            cursor = connection.execute(
                f"UPDATE simulations SET {assignments} WHERE run_id = ?", values
            )
            if cursor.rowcount != 1:
                raise RunNotFoundError(run_id)
            connection.execute(
                """
                INSERT INTO events(run_id, timestamp, level, event, message, data_json)
                VALUES (?, ?, ?, 'status_changed', ?, ?)
                """,
                (
                    run_id,
                    now,
                    "ERROR" if status == "failed" else "INFO",
                    f"Status changed to {status}",
                    canonical_json({"status": status}),
                ),
            )

    def mark_running(self, run_id: str, *, output_directory: str | Path | None = None) -> None:
        self.update_status(run_id, "running", output_directory=output_directory)

    def mark_completed(
        self,
        run_id: str,
        *,
        elapsed_seconds: float | None = None,
        output_directory: str | Path | None = None,
    ) -> None:
        self.update_status(
            run_id,
            "completed",
            elapsed_seconds=elapsed_seconds,
            output_directory=output_directory,
        )

    def mark_failed(
        self,
        run_id: str,
        error: BaseException | str,
        *,
        elapsed_seconds: float | None = None,
        traceback_text: str | None = None,
        output_directory: str | Path | None = None,
    ) -> None:
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
        self.update_status(
            run_id,
            "failed",
            elapsed_seconds=elapsed_seconds,
            output_directory=output_directory,
            error_type=error_type,
            error_message=str(error),
            traceback_text=traceback_text,
        )

    def cancel_run(self, run_id: str, message: str | None = None) -> None:
        self.update_status(run_id, "cancelled", error_message=message)

    def add_event(
        self,
        run_id: str,
        event: str,
        *,
        message: str | None = None,
        level: str = "INFO",
        data: Mapping[str, Any] | None = None,
    ) -> int:
        if not event.strip():
            raise DatabaseError("event must not be empty.")
        with self.transaction(immediate=True) as connection:
            self._require_run(connection, run_id)
            cursor = connection.execute(
                """
                INSERT INTO events(run_id, timestamp, level, event, message, data_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    utc_now(),
                    level.upper(),
                    event,
                    message,
                    canonical_json(data or {}),
                ),
            )
            return int(cursor.lastrowid)

    def record_metric(
        self,
        run_id: str,
        name: str,
        value: RealNumber,
        *,
        unit: str | None = None,
        wavelength_um: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if not name.strip():
            raise DatabaseError("Metric name must not be empty.")
        with self.transaction(immediate=True) as connection:
            self._require_run(connection, run_id)
            cursor = connection.execute(
                """
                INSERT INTO metrics(
                    run_id, name, value, unit, wavelength_um, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    name,
                    float(value),
                    unit,
                    float(wavelength_um) if wavelength_um is not None else None,
                    canonical_json(metadata or {}),
                    utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_metrics(
        self,
        run_id: str,
        metrics: Mapping[str, RealNumber],
        *,
        units: Mapping[str, str] | None = None,
    ) -> None:
        units = units or {}
        with self.transaction(immediate=True) as connection:
            self._require_run(connection, run_id)
            now = utc_now()
            connection.executemany(
                """
                INSERT INTO metrics(
                    run_id, name, value, unit, wavelength_um, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, NULL, '{}', ?)
                """,
                [
                    (run_id, name, float(value), units.get(name), now)
                    for name, value in metrics.items()
                ],
            )

    def record_spectrum(
        self,
        run_id: str,
        monitor_name: str,
        frequencies: Sequence[float],
        values: Sequence[complex | float],
        *,
        unit: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        replace: bool = True,
    ) -> None:
        if len(frequencies) != len(values):
            raise DatabaseError("frequencies and values must have the same length.")
        if not monitor_name.strip():
            raise DatabaseError("monitor_name must not be empty.")
        rows = []
        for index, (frequency, value) in enumerate(zip(frequencies, values)):
            frequency_value = float(frequency)
            if frequency_value <= 0:
                raise DatabaseError("Spectrum frequencies must be positive.")
            complex_value = complex(value)
            rows.append(
                (
                    run_id,
                    monitor_name,
                    index,
                    frequency_value,
                    1.0 / frequency_value,
                    complex_value.real,
                    complex_value.imag,
                    unit,
                    canonical_json(metadata or {}),
                )
            )
        with self.transaction(immediate=True) as connection:
            self._require_run(connection, run_id)
            if replace:
                connection.execute(
                    "DELETE FROM spectra WHERE run_id = ? AND monitor_name = ?",
                    (run_id, monitor_name),
                )
            connection.executemany(
                """
                INSERT INTO spectra(
                    run_id, monitor_name, sample_index, frequency, wavelength_um,
                    value_real, value_imag, unit, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def record_artifact(
        self,
        run_id: str,
        kind: str,
        path: str | Path,
        *,
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        calculate_hash: bool = True,
    ) -> int:
        artifact_path = Path(path).expanduser().resolve()
        exists = artifact_path.is_file()
        size_bytes = artifact_path.stat().st_size if exists else None
        digest = _file_sha256(artifact_path) if exists and calculate_hash else None
        with self.transaction(immediate=True) as connection:
            self._require_run(connection, run_id)
            connection.execute(
                """
                INSERT INTO artifacts(
                    run_id, kind, path, mime_type, size_bytes, sha256,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, path) DO UPDATE SET
                    kind = excluded.kind,
                    mime_type = excluded.mime_type,
                    size_bytes = excluded.size_bytes,
                    sha256 = excluded.sha256,
                    metadata_json = excluded.metadata_json
                """,
                (
                    run_id,
                    kind,
                    str(artifact_path),
                    mime_type,
                    size_bytes,
                    digest,
                    canonical_json(metadata or {}),
                    utc_now(),
                ),
            )
            row = connection.execute(
                "SELECT artifact_id FROM artifacts WHERE run_id = ? AND path = ?",
                (run_id, str(artifact_path)),
            ).fetchone()
            return int(row[0])

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM simulations WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return _decode_run_row(row)

    def find_by_config(
        self,
        config: Mapping[str, Any],
        *,
        status: str | None = "completed",
    ) -> list[dict[str, Any]]:
        digest = configuration_hash(config)
        sql = "SELECT * FROM simulations WHERE config_hash = ?"
        values: list[Any] = [digest]
        if status is not None:
            sql += " AND status = ?"
            values.append(status)
        sql += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(sql, values).fetchall()
        return [_decode_run_row(row) for row in rows]

    def list_runs(
        self,
        *,
        status: str | None = None,
        geometry_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 10_000:
            raise DatabaseError("limit must be between 1 and 10000.")
        if offset < 0:
            raise DatabaseError("offset cannot be negative.")
        clauses: list[str] = []
        values: list[Any] = []
        if status is not None:
            if status not in VALID_STATUSES:
                raise DatabaseError(f"Invalid status {status!r}.")
            clauses.append("status = ?")
            values.append(status)
        if geometry_name is not None:
            clauses.append("geometry_name = ?")
            values.append(geometry_name)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        values.extend([limit, offset])
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM simulations{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                values,
            ).fetchall()
        return [_decode_run_row(row) for row in rows]

    def get_metrics(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._require_run(connection, run_id)
            rows = connection.execute(
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY metric_id", (run_id,)
            ).fetchall()
        return [_decode_metadata_row(row) for row in rows]

    def get_spectra(
        self, run_id: str, monitor_name: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        sql = "SELECT * FROM spectra WHERE run_id = ?"
        values: list[Any] = [run_id]
        if monitor_name is not None:
            sql += " AND monitor_name = ?"
            values.append(monitor_name)
        sql += " ORDER BY monitor_name, sample_index"
        with self._connect() as connection:
            self._require_run(connection, run_id)
            rows = connection.execute(sql, values).fetchall()
        result: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            item = _decode_metadata_row(row)
            item["value"] = complex(item.pop("value_real"), item.pop("value_imag"))
            result.setdefault(item["monitor_name"], []).append(item)
        return result

    def get_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._require_run(connection, run_id)
            rows = connection.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY artifact_id", (run_id,)
            ).fetchall()
        return [_decode_metadata_row(row) for row in rows]

    def get_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            self._require_run(connection, run_id)
            rows = connection.execute(
                "SELECT * FROM events WHERE run_id = ? ORDER BY event_id", (run_id,)
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["data"] = _decode_json(item.pop("data_json"), {})
            result.append(item)
        return result

    def status_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM simulations GROUP BY status"
            ).fetchall()
        counts = {status: 0 for status in sorted(VALID_STATUSES)}
        counts.update({row["status"]: int(row["count"]) for row in rows})
        return counts

    def export_run(self, run_id: str) -> dict[str, Any]:
        return {
            "run": self.get_run(run_id),
            "metrics": self.get_metrics(run_id),
            "spectra": self.get_spectra(run_id),
            "artifacts": self.get_artifacts(run_id),
            "events": self.get_events(run_id),
        }

    @staticmethod
    def _require_run(connection: sqlite3.Connection, run_id: str) -> None:
        row = connection.execute(
            "SELECT 1 FROM simulations WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)


RealNumber = int | float


def _extract_wavelength_range(config: Mapping[str, Any]) -> tuple[float | None, float | None]:
    direct = config.get("wavelength_range_um", config.get("wavelengths_um"))
    source = config.get("source", {})
    if direct is None and isinstance(source, Mapping):
        direct = source.get("wavelength_range_um", source.get("wavelengths_um"))
    if direct is None:
        return None, None
    if isinstance(direct, Real) and not isinstance(direct, bool):
        value = float(direct)
        return value, value
    try:
        values = [float(value) for value in direct]
    except (TypeError, ValueError):
        return None, None
    if not values:
        return None, None
    return min(values), max(values)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_run_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["config"] = _decode_json(result.pop("config_json"), {})
    result["tags"] = _decode_json(result.pop("tags_json"), [])
    return result


def _decode_metadata_row(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result["metadata"] = _decode_json(result.pop("metadata_json"), {})
    return result


# Compact function API retained for scripts that do not want to instantiate the class.
def init_database(path: str | Path | None = None) -> Path:
    return SimulationDatabase(path).path


init_db = init_database


def get_simulation(run_id: str, path: str | Path | None = None) -> dict[str, Any]:
    return SimulationDatabase(path).get_run(run_id)


def list_simulations(
    path: str | Path | None = None, **filters: Any
) -> list[dict[str, Any]]:
    return SimulationDatabase(path).list_runs(**filters)


def save_simulation(
    config: Mapping[str, Any],
    *,
    path: str | Path | None = None,
    run_id: str | None = None,
    status: str = "completed",
    metrics: Mapping[str, RealNumber] | None = None,
    name: str | None = None,
) -> str:
    database = SimulationDatabase(path)
    identity = database.create_run(config, run_id=run_id, name=name, status=status)
    if metrics:
        database.record_metrics(identity, metrics)
    return identity


__all__ = [
    "DatabaseError",
    "RunNotFoundError",
    "SCHEMA_VERSION",
    "SimulationDatabase",
    "VALID_STATUSES",
    "canonical_json",
    "configuration_hash",
    "default_database_path",
    "find_project_root",
    "get_simulation",
    "init_database",
    "init_db",
    "list_simulations",
    "new_run_id",
    "save_simulation",
    "utc_now",
]
