"""Safe one-folder experiment storage for the particle-line project."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import json
import os
import re
import time

from .database import canonical_json, find_project_root


class ResultsError(RuntimeError):
    """Raised when result creation could overwrite or mix experiments."""


def safe_simulation_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    if not cleaned:
        raise ResultsError("The simulation name must contain a letter or number.")
    return cleaned[:100]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def config_hash(config: Mapping[str, Any]) -> str:
    return sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Write JSON atomically where possible, with a OneDrive-safe fallback.

    Windows/WSL folders synchronised by OneDrive can briefly refuse the final
    ``os.replace`` even though both the destination directory and temporary
    file are writable.  Retry that atomic operation first.  If the sync client
    keeps the destination locked, fall back to an in-place write so a completed
    simulation is not incorrectly reported as failed solely while updating its
    manifest.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    replace_error: PermissionError | None = None
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            replace_error = exc
            time.sleep(min(0.05 * (2**attempt), 0.5))

    try:
        path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        raise ResultsError(
            f"Could not update {path}. The atomic rename was repeatedly denied "
            "and the direct-write fallback also failed. Pause OneDrive syncing "
            "for this folder or move the project outside OneDrive."
        ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            # The requested JSON has already been written successfully.  A
            # sync-client lock on the stale .tmp file is not fatal.
            pass

    if replace_error is None:  # pragma: no cover - defensive guard
        raise ResultsError(f"Could not atomically update {path}.")


@dataclass(slots=True)
class ExperimentWorkspace:
    name: str
    slug: str
    directory: Path
    manifest_path: Path
    status_path: Path
    configuration_hash: str

    @classmethod
    def create(
        cls,
        name: str,
        config: Mapping[str, Any],
        *,
        results_root: str | Path | None = None,
        resume: bool = False,
    ) -> "ExperimentWorkspace":
        slug = safe_simulation_name(name)
        root = (
            Path(results_root).expanduser().resolve()
            if results_root is not None
            else find_project_root() / "data" / "results"
        )
        directory = root / slug
        manifest_path = directory / "manifest.json"
        status_path = directory / "status.json"
        digest = config_hash(config)

        if directory.exists():
            if not resume:
                raise ResultsError(
                    f"Result folder already exists: {directory}. Choose a new --name "
                    "(for example, add _v2) or pass --resume for the identical configuration."
                )
            if not manifest_path.is_file():
                raise ResultsError("The existing folder has no manifest and cannot be resumed safely.")
            try:
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ResultsError(f"Could not read the existing manifest: {exc}") from exc
            if previous.get("configuration_hash") != digest:
                raise ResultsError(
                    "The existing simulation name belongs to a different configuration. "
                    "Choose a new name instead of mixing results."
                )
            if previous.get("status") == "completed":
                raise ResultsError(
                    "This simulation is already complete. Choose a new name to preserve it."
                )
        else:
            directory.mkdir(parents=True, exist_ok=False)

        for relative in (
            "geometry",
            "frequency_scan",
            "field_maps",
            "summary",
            "stages",
            "validation",
        ):
            (directory / relative).mkdir(parents=True, exist_ok=True)

        workspace = cls(name, slug, directory, manifest_path, status_path, digest)
        if not manifest_path.exists():
            _write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "simulation_name": name,
                    "simulation_slug": slug,
                    "configuration_hash": digest,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                    "status": "created",
                    "stages": {},
                    "artifacts": {},
                    "warnings": [],
                },
            )
        workspace.set_status("running", message="Simulation started or resumed")
        return workspace

    def manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def update_manifest(self, **updates: Any) -> None:
        value = self.manifest()
        value.update(updates)
        value["updated_at"] = utc_now()
        _write_json(self.manifest_path, value)

    def set_status(self, status: str, *, message: str = "", error: str | None = None) -> None:
        payload = {
            "simulation_name": self.name,
            "status": status,
            "updated_at": utc_now(),
            "message": message,
            "error": error,
        }
        _write_json(self.status_path, payload)
        self.update_manifest(status=status)

    def record_stage(self, stage: str, status: str, **metadata: Any) -> None:
        value = self.manifest()
        stages = dict(value.get("stages", {}))
        stages[stage] = {"status": status, "updated_at": utc_now(), **metadata}
        value["stages"] = stages
        value["updated_at"] = utc_now()
        _write_json(self.manifest_path, value)

    def record_artifact(self, label: str, path: str | Path) -> None:
        destination = Path(path).resolve()
        try:
            relative = destination.relative_to(self.directory.resolve())
        except ValueError as exc:
            raise ResultsError("Experiment artifacts must remain inside its result folder.") from exc
        value = self.manifest()
        artifacts = dict(value.get("artifacts", {}))
        artifacts[label] = str(relative)
        value["artifacts"] = artifacts
        value["updated_at"] = utc_now()
        _write_json(self.manifest_path, value)

    def add_warning(self, message: str) -> None:
        value = self.manifest()
        warnings = list(value.get("warnings", []))
        warnings.append(str(message))
        value["warnings"] = warnings
        value["updated_at"] = utc_now()
        _write_json(self.manifest_path, value)


__all__ = [
    "ExperimentWorkspace",
    "ResultsError",
    "config_hash",
    "safe_simulation_name",
    "utc_now",
]
