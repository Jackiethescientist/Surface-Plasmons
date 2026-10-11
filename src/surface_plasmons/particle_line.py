"""Pure-Python helpers for finite one-dimensional particle lines.

The functions in this module deliberately do not import MEEP.  They are used
by the geometry builder, the project runners, validation, plotting, and tests
so every part of the project uses exactly the same particle coordinates.
All distances are in micrometres.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence
import math


class ParticleLineError(ValueError):
    """Raised when a particle-line definition is ambiguous or unphysical."""


@dataclass(frozen=True, slots=True)
class ParticlePosition:
    particle_index: int
    ideal_x_um: float
    ideal_y_um: float
    ideal_z_um: float
    actual_x_um: float
    actual_y_um: float
    actual_z_um: float
    shift_x_um: float
    shift_y_um: float
    shift_z_um: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_defects(value: Any, particle_count: int | None = None) -> tuple[dict[str, Any], ...]:
    """Validate displaced or missing-particle defects.

    A displacement uses ``{"particle_index": i, "shift_um": [dx, dy, dz]}``.
    A vacancy uses ``{"particle_index": i, "missing": true}``.  The two
    forms are deliberately mutually exclusive so a configuration cannot hide
    an unused displacement on a particle that is not present.
    """

    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ParticleLineError("defects must be a list of mappings.")
    normalised: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ParticleLineError(f"Defect {row} must be a mapping.")
        unknown = set(item) - {"particle_index", "shift_um", "missing"}
        if unknown:
            raise ParticleLineError(f"Defect {row} has unknown field(s): {sorted(unknown)}.")
        index = item.get("particle_index")
        has_shift = "shift_um" in item
        missing = item.get("missing", False)
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise ParticleLineError(f"Defect {row} particle_index must be an integer.")
        index = int(index)
        if index < 0 or (particle_count is not None and index >= particle_count):
            limit = f" from 0 to {particle_count - 1}" if particle_count is not None else ""
            raise ParticleLineError(f"Defect {row} particle_index must be{limit}.")
        if index in seen:
            raise ParticleLineError(f"Particle {index} appears in more than one defect entry.")
        if not isinstance(missing, bool):
            raise ParticleLineError(f"Defect {row} missing must be true or false.")
        if missing and has_shift:
            raise ParticleLineError(
                f"Defect {row} cannot contain both missing=true and shift_um."
            )
        if missing:
            seen.add(index)
            normalised.append({"particle_index": index, "missing": True})
            continue
        if not has_shift:
            raise ParticleLineError(
                f"Defect {row} requires shift_um or missing=true."
            )
        shift = item["shift_um"]
        if (
            not isinstance(shift, Sequence)
            or isinstance(shift, (str, bytes))
            or len(shift) != 3
            or any(isinstance(component, bool) or not isinstance(component, Real) for component in shift)
        ):
            raise ParticleLineError(f"Defect {row} shift_um must contain three numbers.")
        seen.add(index)
        normalised.append(
            {
                "particle_index": index,
                "shift_um": tuple(float(component) for component in shift),
                "missing": False,
            }
        )
    return tuple(normalised)


def particle_diameter(parameters: Mapping[str, Any]) -> float:
    shape = str(parameters.get("particle_shape", "sphere")).lower()
    if shape == "sphere":
        return 2.0 * float(parameters["radius"])
    if shape == "cube":
        return float(parameters["cube_size"])
    raise ParticleLineError("particle_shape must be 'sphere' or 'cube'.")


def particle_line_positions(parameters: Mapping[str, Any]) -> tuple[ParticlePosition, ...]:
    """Return ideal and actual centres for a centred finite line along x."""

    count = parameters.get("particle_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 1:
        raise ParticleLineError("particle_count must be a positive integer.")
    count = int(count)
    spacing = parameters.get("spacing")
    if isinstance(spacing, bool) or not isinstance(spacing, Real) or spacing <= 0:
        raise ParticleLineError("spacing must be positive.")
    spacing = float(spacing)
    diameter = particle_diameter(parameters)
    if spacing < diameter:
        raise ParticleLineError(
            f"spacing ({spacing:g} um) is smaller than the particle width ({diameter:g} um)."
        )

    center = parameters.get("center", (0.0, 0.0, 0.0))
    if (
        not isinstance(center, Sequence)
        or isinstance(center, (str, bytes))
        or len(center) != 3
    ):
        raise ParticleLineError("center must contain three numbers.")
    center_x, _, center_z = (float(value) for value in center)
    surface_y = float(parameters.get("surface_y", 0.0))
    gap = float(parameters.get("particle_surface_gap", 0.0))
    half_height = diameter / 2.0
    particle_y = surface_y + gap + half_height
    first_x = center_x - 0.5 * (count - 1) * spacing
    defects = validate_defects(parameters.get("defects", ()), count)
    missing = {
        int(item["particle_index"])
        for item in defects
        if bool(item.get("missing", False))
    }
    if len(missing) == count:
        raise ParticleLineError("At least one particle must remain in the line.")
    shifts = {
        int(item["particle_index"]): tuple(item["shift_um"])
        for item in defects
        if not bool(item.get("missing", False))
    }

    result: list[ParticlePosition] = []
    for index in range(count):
        if index in missing:
            continue
        ideal = (first_x + index * spacing, particle_y, center_z)
        shift = shifts.get(index, (0.0, 0.0, 0.0))
        actual = tuple(ideal[axis] + shift[axis] for axis in range(3))
        if actual[1] - half_height < surface_y - 1e-12:
            raise ParticleLineError(
                f"Particle {index} enters the substrate after its shift; increase shift_y or the gap."
            )
        result.append(
            ParticlePosition(
                particle_index=index,
                ideal_x_um=ideal[0],
                ideal_y_um=ideal[1],
                ideal_z_um=ideal[2],
                actual_x_um=actual[0],
                actual_y_um=actual[1],
                actual_z_um=actual[2],
                shift_x_um=shift[0],
                shift_y_um=shift[1],
                shift_z_um=shift[2],
            )
        )

    for left in range(len(result)):
        for right in range(left + 1, len(result)):
            a = result[left]
            b = result[right]
            distance = math.dist(
                (a.actual_x_um, a.actual_y_um, a.actual_z_um),
                (b.actual_x_um, b.actual_y_um, b.actual_z_um),
            )
            if distance < diameter - 1e-12:
                raise ParticleLineError(
                    f"Particles {a.particle_index} and {b.particle_index} overlap "
                    f"(centre separation {distance:g} um; required at least {diameter:g} um)."
                )
    return tuple(result)


def missing_particle_indices(
    defects: Any,
    particle_count: int | None = None,
) -> tuple[int, ...]:
    """Return sorted nominal indices removed by vacancy defects."""

    return tuple(
        sorted(
            int(item["particle_index"])
            for item in validate_defects(defects, particle_count)
            if bool(item.get("missing", False))
        )
    )


def line_extents(parameters: Mapping[str, Any]) -> tuple[float, float, float]:
    """Return an axis-aligned structure extent including the substrate depth."""

    positions = particle_line_positions(parameters)
    diameter = particle_diameter(parameters)
    half = diameter / 2.0
    x_values = [item.actual_x_um for item in positions]
    z_values = [item.actual_z_um for item in positions]
    surface_y = float(parameters.get("surface_y", 0.0))
    substrate_bottom = surface_y - float(parameters["substrate_thickness"])
    particle_top = max(item.actual_y_um + half for item in positions)
    return (
        max(x_values) - min(x_values) + diameter,
        particle_top - substrate_bottom,
        max(z_values) - min(z_values) + diameter,
    )


__all__ = [
    "ParticleLineError",
    "ParticlePosition",
    "line_extents",
    "missing_particle_indices",
    "particle_diameter",
    "particle_line_positions",
    "validate_defects",
]
