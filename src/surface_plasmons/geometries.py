"""
Coordinate convention
---------------------
* x: horizontal direction in the simulation image
* y: vertical direction and surface normal for films/stacks
* z: invariant direction in 2-D, or the second in-plane direction in 3-D

Every numerical length is in micrometres.  Later MEEP geometry objects override
earlier objects where they overlap; the builders use this rule for cores, holes,
slots, rings, and grooves.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Sequence
import difflib
import re

from .materials import MaterialError, build_material, get_material_spec
from .particle_line import (
    ParticleLineError,
    particle_line_positions,
    validate_defects,
)


class GeometryError(ValueError):
    """Raised when a geometry name or parameter set is invalid."""


_REQUIRED = object()


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    name: str
    kind: str
    description: str
    default: Any = _REQUIRED
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()

    @property
    def required(self) -> bool:
        return self.default is _REQUIRED

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "required": self.required,
            "default": None if self.required else self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "choices": list(self.choices),
        }


@dataclass(frozen=True, slots=True)
class GeometrySpec:
    key: str
    display_name: str
    category: str
    description: str
    dimensions: tuple[int, ...]
    parameters: tuple[ParameterSpec, ...]
    aliases: tuple[str, ...] = ()
    periodic_axes: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        schema = (*COMMON_PARAMETERS, *self.parameters)
        return {
            "key": self.key,
            "display_name": self.display_name,
            "category": self.category,
            "description": self.description,
            "dimensions": list(self.dimensions),
            "parameters": {item.name: item.to_dict() for item in schema},
            "aliases": list(self.aliases),
            "periodic_axes": list(self.periodic_axes),
            "notes": self.notes,
        }


@dataclass(slots=True)
class GeometryBuild:
    """Objects and background medium ready to pass to ``mp.Simulation``."""

    name: str
    objects: list[Any]
    default_material: Any
    parameters: dict[str, Any]
    periodic_axes: tuple[str, ...] = ()
    hints: dict[str, Any] | None = None

    @property
    def geometry(self) -> list[Any]:
        """Alias matching MEEP's ``geometry=...`` argument."""

        return self.objects


def P(
    name: str,
    kind: str,
    description: str,
    default: Any = _REQUIRED,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    choices: Sequence[Any] = (),
) -> ParameterSpec:
    return ParameterSpec(
        name=name,
        kind=kind,
        description=description,
        default=default,
        minimum=minimum,
        maximum=maximum,
        choices=tuple(choices),
    )


COMMON_PARAMETERS = (
    P("dimensions", "choice", "MEEP simulation dimensionality.", 2, choices=(2, 3)),
    P("center", "vector3", "Geometry centre (x, y, z) in um.", (0.0, 0.0, 0.0)),
    P(
        "wavelength_um",
        "number_or_sequence_or_none",
        "Single wavelength or simulated wavelength band used for material checks.",
        None,
    ),
    P(
        "allow_out_of_range",
        "bool",
        "Permit use outside a material model's fitted wavelength range.",
        False,
    ),
    P(
        "allow_unstable",
        "bool",
        "Permit a MEEP material model explicitly marked unstable.",
        False,
    ),
)


MAT_METAL = P("metal", "material", "Metal material key.", "au")
MAT_BACKGROUND = P("background", "material", "Default surrounding medium.", "air")
MAT_DIELECTRIC = P("dielectric", "material", "Dielectric material key.", "air")


GEOMETRIES: dict[str, GeometrySpec] = {
    "planar_interface": GeometrySpec(
        "planar_interface", "Planar metal-dielectric interface", "film",
        "One flat metal region below a dielectric half-space.", (2, 3),
        (
            MAT_METAL,
            P("dielectric", "material", "Upper half-space material.", "air"),
            P("metal_depth", "positive", "Metal depth represented inside the cell.", 0.5),
        ),
        aliases=("interface", "single_interface", "spp_interface"),
        notes="Choose metal_depth large enough to reach the lower PML.",
    ),
    "metal_film": GeometrySpec(
        "metal_film", "Metal film", "film",
        "Finite metal film surrounded by one background medium.", (2, 3),
        (MAT_METAL, MAT_BACKGROUND, P("thickness", "positive", "Film thickness.", 0.05)),
        aliases=("film", "thin_film", "imi_film"),
    ),
    "kretschmann_stack": GeometrySpec(
        "kretschmann_stack", "Kretschmann coupling stack", "film",
        "Substrate, thin metal film, and analyte/superstrate stack.", (2, 3),
        (
            MAT_METAL,
            P("substrate", "material", "Prism/substrate material.", "bk7"),
            P("analyte", "material", "Medium above the metal.", "air"),
            P("metal_thickness", "positive", "Metal-film thickness.", 0.05),
            P("substrate_depth", "positive", "Represented substrate depth.", 0.8),
        ),
        aliases=("kretschmann", "spr_stack"),
        notes="Set incidence angle with k_point/source settings in simulation.py.",
    ),
    "multilayer_stack": GeometrySpec(
        "multilayer_stack", "Arbitrary multilayer stack", "film",
        "Any bottom-to-top sequence of material layers.", (2, 3),
        (
            MAT_BACKGROUND,
            P(
                "layers", "layers",
                "List of {'material': key, 'thickness': um, optional 'label': text} records.",
            ),
        ),
        aliases=("multilayer", "layer_stack"),
    ),
    "nanoparticle_sphere": GeometrySpec(
        "nanoparticle_sphere", "Spherical nanoparticle", "particle",
        "Single metal or dielectric sphere; a circle in a 2-D cross-section.", (2, 3),
        (
            P("particle", "material", "Particle material.", "au"),
            MAT_BACKGROUND,
            P("radius", "positive", "Sphere radius.", 0.04),
        ),
        aliases=("sphere", "nanoparticle", "nanosphere"),
    ),
    "nanoparticle_disk": GeometrySpec(
        "nanoparticle_disk", "Nanodisk", "particle",
        "Circular disk/cylinder with its axis along z.", (2, 3),
        (
            P("particle", "material", "Disk material.", "au"),
            MAT_BACKGROUND,
            P("radius", "positive", "Disk radius.", 0.04),
            P("height", "positive", "3-D disk height.", 0.03),
        ),
        aliases=("disk", "cylinder", "nanocylinder"),
    ),
    "nanocube": GeometrySpec(
        "nanocube", "Nanocube or nanorectangle", "particle",
        "Rectangular particle with independent x, y, and z dimensions.", (2, 3),
        (
            P("particle", "material", "Particle material.", "au"),
            MAT_BACKGROUND,
            P("size_x", "positive", "Width along x.", 0.08),
            P("size_y", "positive", "Height along y.", 0.08),
            P("size_z", "positive", "Depth along z in 3-D.", 0.08),
        ),
        aliases=("cube", "nanoblock", "rectangle"),
    ),
    "ellipsoid": GeometrySpec(
        "ellipsoid", "Ellipsoidal nanoparticle", "particle",
        "Ellipsoid with three independently controlled axis lengths.", (2, 3),
        (
            P("particle", "material", "Particle material.", "au"),
            MAT_BACKGROUND,
            P("size_x", "positive", "Full x-axis length.", 0.10),
            P("size_y", "positive", "Full y-axis length.", 0.05),
            P("size_z", "positive", "Full z-axis length.", 0.05),
        ),
        aliases=("nanoellipsoid", "spheroid"),
    ),
    "nanorod": GeometrySpec(
        "nanorod", "Rounded nanorod", "particle",
        "Rod oriented along x; rounded in both 2-D and 3-D.", (2, 3),
        (
            P("particle", "material", "Rod material.", "au"),
            MAT_BACKGROUND,
            P("length", "positive", "Tip-to-tip rod length.", 0.12),
            P("diameter", "positive", "Rod diameter.", 0.04),
        ),
        aliases=("rod", "nano_rod"),
    ),
    "sphere_dimer": GeometrySpec(
        "sphere_dimer", "Sphere dimer", "coupled_particles",
        "Two equal spheres separated by a surface-to-surface gap along x.", (2, 3),
        (
            P("particle", "material", "Particle material.", "au"),
            MAT_BACKGROUND,
            P("radius", "positive", "Radius of each sphere.", 0.04),
            P("gap", "nonnegative", "Surface-to-surface gap.", 0.01),
        ),
        aliases=("dimer", "nanosphere_dimer"),
    ),
    "disk_dimer": GeometrySpec(
        "disk_dimer", "Nanodisk dimer", "coupled_particles",
        "Two disks separated by a surface-to-surface gap along x.", (2, 3),
        (
            P("particle", "material", "Disk material.", "au"),
            MAT_BACKGROUND,
            P("radius", "positive", "Radius of each disk.", 0.04),
            P("height", "positive", "Disk height in 3-D.", 0.03),
            P("gap", "nonnegative", "Surface-to-surface gap.", 0.01),
        ),
        aliases=("nanodisk_dimer",),
    ),
    "core_shell_sphere": GeometrySpec(
        "core_shell_sphere", "Core-shell sphere", "core_shell",
        "Concentric core and shell spheres.", (2, 3),
        (
            P("core", "material", "Core material.", "sio2"),
            P("shell", "material", "Shell material.", "au"),
            MAT_BACKGROUND,
            P("core_radius", "positive", "Core radius.", 0.03),
            P("outer_radius", "positive", "Total outer radius.", 0.04),
        ),
        aliases=("nanoshell", "core_shell", "core_shell_particle"),
    ),
    "core_shell_cylinder": GeometrySpec(
        "core_shell_cylinder", "Core-shell cylinder", "core_shell",
        "Concentric cylindrical core and shell.", (2, 3),
        (
            P("core", "material", "Core material.", "sio2"),
            P("shell", "material", "Shell material.", "au"),
            MAT_BACKGROUND,
            P("core_radius", "positive", "Core radius.", 0.03),
            P("outer_radius", "positive", "Total outer radius.", 0.04),
            P("height", "positive", "Cylinder height in 3-D.", 0.03),
        ),
        aliases=("core_shell_disk",),
    ),
    "nanoring": GeometrySpec(
        "nanoring", "Nanoring", "core_shell",
        "Metal ring made by overriding the centre of a larger disk.", (2, 3),
        (
            P("ring", "material", "Ring material.", "au"),
            MAT_BACKGROUND,
            P("inner_radius", "positive", "Inner radius.", 0.025),
            P("outer_radius", "positive", "Outer radius.", 0.04),
            P("height", "positive", "Ring height in 3-D.", 0.03),
        ),
        aliases=("ring", "nano_ring"),
    ),
    "bowtie_antenna": GeometrySpec(
        "bowtie_antenna", "Bowtie nanoantenna", "antenna",
        "Two triangular prisms facing across a nanoscale gap.", (2, 3),
        (
            P("particle", "material", "Triangle material.", "au"),
            MAT_BACKGROUND,
            P("triangle_length", "positive", "Tip-to-base length.", 0.10),
            P("triangle_width", "positive", "Full base width.", 0.08),
            P("gap", "nonnegative", "Tip-to-tip gap.", 0.01),
            P("height", "positive", "Prism height along z.", 0.03),
        ),
        aliases=("bowtie", "bow_tie"),
    ),
    "nanohole_film": GeometrySpec(
        "nanohole_film", "Nanohole in a metal film", "aperture",
        "Metal film with a circular dielectric aperture.", (2, 3),
        (
            MAT_METAL, MAT_BACKGROUND,
            P("hole_material", "material", "Material filling the hole.", "air"),
            P("film_thickness", "positive", "Film thickness.", 0.05),
            P("hole_radius", "positive", "Hole radius.", 0.04),
        ),
        aliases=("nanohole", "hole_film", "circular_aperture"),
    ),
    "nanoslit_film": GeometrySpec(
        "nanoslit_film", "Nanoslit in a metal film", "aperture",
        "Metal film with a rectangular dielectric slit.", (2, 3),
        (
            MAT_METAL, MAT_BACKGROUND,
            P("slit_material", "material", "Material filling the slit.", "air"),
            P("film_thickness", "positive", "Film thickness.", 0.05),
            P("slit_width", "positive", "Slit width along x.", 0.03),
            P("slit_depth", "positive", "Slit depth along z in 3-D.", 0.20),
        ),
        aliases=("nanoslit", "slit_film", "rectangular_aperture"),
    ),
    "ridge_grating": GeometrySpec(
        "ridge_grating", "Metal ridge grating", "periodic",
        "Finite preview of a 1-D periodic metal grating and optional base film.", (2, 3),
        (
            MAT_METAL, MAT_BACKGROUND,
            P("period", "positive", "Grating period along x.", 0.5),
            P("fill_factor", "positive", "Ridge width divided by period.", 0.5, maximum=1.0),
            P("ridge_height", "positive", "Ridge height.", 0.05),
            P("base_thickness", "nonnegative", "Continuous metal base thickness.", 0.05),
            P("periods", "integer", "Number of ridges in the finite preview.", 1, minimum=1),
        ),
        aliases=("grating", "metal_grating"), periodic_axes=("x",),
        notes="Use periods=1 with an x-periodic simulation cell for an infinite grating.",
    ),
    "groove_grating": GeometrySpec(
        "groove_grating", "Grooved metal grating", "periodic",
        "Metal film with repeated rectangular dielectric grooves.", (2, 3),
        (
            MAT_METAL, MAT_BACKGROUND,
            P("groove_material", "material", "Material filling grooves.", "air"),
            P("period", "positive", "Groove period along x.", 0.5),
            P("fill_factor", "positive", "Groove width divided by period.", 0.5, maximum=1.0),
            P("groove_depth", "positive", "Depth cut into metal.", 0.05),
            P("metal_depth", "positive", "Total represented metal depth.", 0.25),
            P("periods", "integer", "Number of grooves in the finite preview.", 1, minimum=1),
        ),
        aliases=("grooves",), periodic_axes=("x",),
    ),
    "particle_array": GeometrySpec(
        "particle_array", "Periodic nanoparticle array unit cell", "periodic",
        "One particle for use with periodic boundary conditions.", (2, 3),
        (
            P("particle", "material", "Particle material.", "au"),
            MAT_BACKGROUND,
            P("shape", "choice", "Unit-cell particle shape.", "sphere", choices=("sphere", "disk", "cube")),
            P("radius", "positive", "Sphere/disk radius.", 0.04),
            P("size", "positive", "Cube side length.", 0.08),
            P("height", "positive", "Disk or cube depth in 3-D.", 0.03),
            P("period_x", "positive", "Suggested x cell size.", 0.2),
            P("period_z", "positive", "Suggested z cell size in 3-D.", 0.2),
        ),
        aliases=("nanoparticle_array", "array"), periodic_axes=("x", "z"),
    ),
    "particle_line_on_substrate": GeometrySpec(
        "particle_line_on_substrate",
        "Finite particle line on a substrate",
        "finite_array",
        "A finite 1-D line of spheres or cubes with displaced or missing-particle defects.",
        (3,),
        (
            P("particle", "material", "Particle material.", "au"),
            MAT_BACKGROUND,
            P("substrate", "material", "Substrate material.", "sio2"),
            P("particle_shape", "choice", "Particle shape.", "sphere", choices=("sphere", "cube")),
            P("particle_count", "integer", "Number of particles in the line.", 9, minimum=1),
            P("radius", "positive", "Sphere radius.", 0.040),
            P("cube_size", "positive", "Cube side length.", 0.080),
            P("spacing", "positive", "Ideal centre-to-centre spacing.", 0.150),
            P("substrate_thickness", "positive", "Represented substrate depth.", 0.300),
            P("surface_y", "number", "y coordinate of the substrate surface.", 0.0),
            P("particle_surface_gap", "nonnegative", "Gap between particles and substrate.", 0.0),
            P("defects", "defects", "Particle-indexed displacement or vacancy records.", ()),
            P("include_particles", "bool", "False produces the substrate-only reference.", True),
        ),
        aliases=("particle_line", "finite_particle_line", "defect_line"),
        notes="The line is centred along x; y is the surface normal and z is transverse.",
    ),
    "nanohole_array": GeometrySpec(
        "nanohole_array", "Periodic nanohole array unit cell", "periodic",
        "One circular hole in a metal film for periodic repetition.", (2, 3),
        (
            MAT_METAL, MAT_BACKGROUND,
            P("hole_material", "material", "Material filling the hole.", "air"),
            P("film_thickness", "positive", "Film thickness.", 0.05),
            P("hole_radius", "positive", "Hole radius.", 0.04),
            P("period_x", "positive", "Suggested x cell size.", 0.2),
            P("period_z", "positive", "Suggested z cell size in 3-D.", 0.2),
        ),
        aliases=("hole_array", "eot_array"), periodic_axes=("x", "z"),
    ),
    "mim_waveguide": GeometrySpec(
        "mim_waveguide", "Metal-insulator-metal waveguide", "waveguide",
        "Dielectric slot between two metal claddings.", (2, 3),
        (
            MAT_METAL,
            P("dielectric", "material", "Slot material.", "air"),
            P("gap", "positive", "Insulating slot width along y.", 0.05),
            P("metal_thickness", "positive", "Thickness of each metal cladding.", 0.25),
        ),
        aliases=("mim", "slot_waveguide"),
    ),
    "nanoparticle_on_mirror": GeometrySpec(
        "nanoparticle_on_mirror", "Nanoparticle-on-mirror cavity", "hybrid",
        "Sphere above a metal mirror separated by a dielectric spacer.", (2, 3),
        (
            P("particle", "material", "Particle material.", "au"),
            P("mirror", "material", "Mirror material.", "au"),
            P("spacer", "material", "Gap/spacer material.", "sio2"),
            MAT_BACKGROUND,
            P("radius", "positive", "Particle radius.", 0.04),
            P("gap", "positive", "Spacer thickness.", 0.002),
            P("mirror_depth", "positive", "Represented mirror depth.", 0.20),
        ),
        aliases=("npom", "sphere_on_mirror"),
    ),
    "nanocube_on_mirror": GeometrySpec(
        "nanocube_on_mirror", "Nanocube-on-mirror cavity", "hybrid",
        "Metal cube above a metal mirror separated by a dielectric spacer.", (2, 3),
        (
            P("particle", "material", "Cube material.", "ag"),
            P("mirror", "material", "Mirror material.", "au"),
            P("spacer", "material", "Gap/spacer material.", "sio2"),
            MAT_BACKGROUND,
            P("size", "positive", "Cube side length.", 0.075),
            P("gap", "positive", "Spacer thickness.", 0.003),
            P("mirror_depth", "positive", "Represented mirror depth.", 0.20),
        ),
        aliases=("ncom", "cube_on_mirror"),
    ),
    "tip_substrate": GeometrySpec(
        "tip_substrate", "Metal tip above substrate", "hybrid",
        "Conical probe tip separated from a flat substrate by a gap.", (2, 3),
        (
            P("tip", "material", "Tip material.", "au"),
            P("substrate", "material", "Substrate material.", "au"),
            MAT_BACKGROUND,
            P("tip_radius", "positive", "Cone base radius.", 0.05),
            P("tip_length", "positive", "Cone length.", 0.15),
            P("apex_radius", "nonnegative", "Truncated apex radius; zero gives a sharp cone.", 0.005),
            P("gap", "positive", "Apex-to-substrate gap.", 0.01),
            P("substrate_depth", "positive", "Represented substrate depth.", 0.20),
        ),
        aliases=("probe_tip", "ters_tip"),
    ),
}


def _normalise_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise GeometryError("Geometry name must be a non-empty string.")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _alias_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for key, spec in GEOMETRIES.items():
        for candidate in (key, spec.display_name, *spec.aliases):
            result[_normalise_name(candidate)] = key
    return result


_ALIASES = _alias_map()


def get_geometry_spec(name: str) -> GeometrySpec:
    normalised = _normalise_name(name)
    key = _ALIASES.get(normalised)
    if key is not None:
        return GEOMETRIES[key]
    matches = difflib.get_close_matches(name.lower(), sorted(GEOMETRIES), n=3, cutoff=0.4)
    suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
    raise GeometryError(f"Unknown geometry {name!r}.{suggestion}")


def list_geometries(
    *, category: str | None = None, dimensions: int | None = None
) -> list[GeometrySpec]:
    results = []
    for spec in GEOMETRIES.values():
        if category is not None and spec.category != category:
            continue
        if dimensions is not None and dimensions not in spec.dimensions:
            continue
        results.append(spec)
    return sorted(results, key=lambda item: item.key)


def geometry_catalog() -> dict[str, dict[str, Any]]:
    """Return a JSON-serialisable catalogue suitable for configuration forms."""

    return {spec.key: spec.to_dict() for spec in list_geometries()}


def validate_geometry_parameters(
    name: str,
    parameters: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Apply defaults, type checks, range checks, and cross-field checks."""

    spec = get_geometry_spec(name)
    supplied = dict(parameters or {})
    supplied.update(overrides)
    schema = {item.name: item for item in (*COMMON_PARAMETERS, *spec.parameters)}

    unknown = sorted(set(supplied) - set(schema))
    if unknown:
        raise GeometryError(
            f"Unexpected parameter(s) for {spec.key}: {', '.join(unknown)}. "
            f"Allowed: {', '.join(schema)}"
        )

    resolved: dict[str, Any] = {}
    for parameter in schema.values():
        if parameter.name in supplied:
            value = supplied[parameter.name]
        elif parameter.required:
            raise GeometryError(
                f"Missing required parameter {parameter.name!r} for {spec.key}."
            )
        else:
            value = parameter.default
        _validate_value(parameter, value)
        resolved[parameter.name] = value

    dimensions = resolved["dimensions"]
    if dimensions not in spec.dimensions:
        raise GeometryError(
            f"{spec.key!r} supports dimensions {spec.dimensions}, not {dimensions}."
        )
    _cross_validate(spec.key, resolved)
    return resolved


def build_geometry(
    name: str,
    parameters: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> GeometryBuild:
    """Validate a configuration and construct its MEEP geometry objects."""

    spec = get_geometry_spec(name)
    resolved = validate_geometry_parameters(spec.key, parameters, **overrides)
    try:
        mp = import_module("meep")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "MEEP is not importable. Select the WSL/Miniforge interpreter that "
            "contains pymeep before building geometry."
        ) from exc
    builder = _BUILDERS[spec.key]
    objects, default_material, hints = builder(mp, resolved)
    return GeometryBuild(
        name=spec.key,
        objects=objects,
        default_material=default_material,
        parameters=resolved,
        periodic_axes=tuple(hints.get("periodic_axes", spec.periodic_axes)),
        hints=hints,
    )


def build_geometry_from_config(config: Mapping[str, Any]) -> GeometryBuild:
    """Build from ``{'geometry': name, 'parameters': {...}}`` configuration."""

    if not isinstance(config, Mapping):
        raise GeometryError("Geometry configuration must be a mapping.")
    name = config.get("geometry", config.get("type", config.get("name")))
    if name is None:
        raise GeometryError("Configuration requires 'geometry', 'type', or 'name'.")
    parameters = config.get("parameters", config.get("params", {}))
    if not isinstance(parameters, Mapping):
        raise GeometryError("Configuration 'parameters' must be a mapping.")
    extras = set(config) - {"geometry", "type", "name", "parameters", "params"}
    if extras:
        parameters = {**parameters, **{key: config[key] for key in extras}}
    return build_geometry(str(name), parameters)


def get_geometry(
    name: str,
    parameters: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> list[Any]:
    """Convenience function returning only the MEEP object list."""

    return build_geometry(name, parameters, **overrides).objects


def _validate_value(spec: ParameterSpec, value: Any) -> None:
    kind = spec.kind
    if kind in {"positive", "nonnegative", "number"}:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise GeometryError(f"{spec.name} must be a number; received {value!r}.")
        number = float(value)
        if kind == "positive" and number <= 0:
            raise GeometryError(f"{spec.name} must be greater than zero.")
        if kind == "nonnegative" and number < 0:
            raise GeometryError(f"{spec.name} cannot be negative.")
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise GeometryError(f"{spec.name} must be an integer; received {value!r}.")
        number = float(value)
    elif kind == "bool":
        if not isinstance(value, bool):
            raise GeometryError(f"{spec.name} must be true or false.")
        return
    elif kind == "choice":
        if value not in spec.choices:
            raise GeometryError(f"{spec.name} must be one of {spec.choices}; received {value!r}.")
        return
    elif kind == "vector3":
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 3
            or any(isinstance(item, bool) or not isinstance(item, Real) for item in value)
        ):
            raise GeometryError(f"{spec.name} must contain three numbers.")
        return
    elif kind == "number_or_sequence_or_none":
        if value is None:
            return
        values = (value,) if isinstance(value, Real) and not isinstance(value, bool) else value
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or not values
            or any(isinstance(item, bool) or not isinstance(item, Real) or item <= 0 for item in values)
        ):
            raise GeometryError(f"{spec.name} must be positive wavelength value(s).")
        return
    elif kind == "material":
        try:
            get_material_spec(value)
        except (MaterialError, TypeError) as exc:
            raise GeometryError(f"Invalid material for {spec.name}: {value!r}.") from exc
        return
    elif kind == "layers":
        _validate_layers(value)
        return
    elif kind == "defects":
        try:
            validate_defects(value)
        except ParticleLineError as exc:
            raise GeometryError(str(exc)) from exc
        return
    else:
        raise GeometryError(f"Internal error: unsupported parameter kind {kind!r}.")

    if spec.minimum is not None and number < spec.minimum:
        raise GeometryError(f"{spec.name} must be at least {spec.minimum}.")
    if spec.maximum is not None and number > spec.maximum:
        raise GeometryError(f"{spec.name} must be at most {spec.maximum}.")


def _validate_layers(value: Any) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise GeometryError("layers must be a non-empty list.")
    for index, layer in enumerate(value):
        if not isinstance(layer, Mapping):
            raise GeometryError(f"Layer {index} must be a mapping.")
        unknown = set(layer) - {"material", "thickness", "label"}
        if unknown:
            raise GeometryError(f"Layer {index} has unknown field(s): {sorted(unknown)}.")
        if "material" not in layer or "thickness" not in layer:
            raise GeometryError(f"Layer {index} requires material and thickness.")
        try:
            get_material_spec(layer["material"])
        except (MaterialError, TypeError) as exc:
            raise GeometryError(f"Layer {index} has an invalid material.") from exc
        thickness = layer["thickness"]
        if isinstance(thickness, bool) or not isinstance(thickness, Real) or thickness <= 0:
            raise GeometryError(f"Layer {index} thickness must be positive.")


def _cross_validate(name: str, p: Mapping[str, Any]) -> None:
    if name in {"core_shell_sphere", "core_shell_cylinder"}:
        if p["core_radius"] >= p["outer_radius"]:
            raise GeometryError("core_radius must be smaller than outer_radius.")
    if name == "nanoring" and p["inner_radius"] >= p["outer_radius"]:
        raise GeometryError("inner_radius must be smaller than outer_radius.")
    if name == "nanorod" and p["length"] < p["diameter"]:
        raise GeometryError("nanorod length must be at least its diameter.")
    if name == "groove_grating" and p["groove_depth"] > p["metal_depth"]:
        raise GeometryError("groove_depth cannot exceed metal_depth.")
    if name == "tip_substrate" and p["apex_radius"] > p["tip_radius"]:
        raise GeometryError("apex_radius cannot exceed tip_radius.")
    if name == "particle_line_on_substrate":
        try:
            particle_line_positions(p)
        except ParticleLineError as exc:
            raise GeometryError(str(exc)) from exc


def _medium(p: Mapping[str, Any], field: str) -> Any:
    return build_material(
        p[field],
        wavelength_um=p["wavelength_um"],
        allow_out_of_range=p["allow_out_of_range"],
        allow_unstable=p["allow_unstable"],
    )


def _centre(mp: Any, p: Mapping[str, Any]) -> Any:
    return mp.Vector3(*map(float, p["center"]))


def _cylinder_height(mp: Any, p: Mapping[str, Any], value: float) -> float:
    return mp.inf if p["dimensions"] == 2 else float(value)


Builder = Callable[[Any, Mapping[str, Any]], tuple[list[Any], Any, dict[str, Any]]]


def _planar_interface(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    metal = _medium(p, "metal")
    default = _medium(p, "dielectric")
    depth = p["metal_depth"]
    obj = mp.Block(
        center=mp.Vector3(c.x, c.y - depth / 2, c.z),
        size=mp.Vector3(mp.inf, depth, mp.inf), material=metal,
    )
    return [obj], default, {"interface_y_um": c.y}


def _metal_film(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    obj = mp.Block(
        center=c, size=mp.Vector3(mp.inf, p["thickness"], mp.inf),
        material=_medium(p, "metal"),
    )
    return [obj], _medium(p, "background"), {"film_normal": "y"}


def _kretschmann(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    film_t = p["metal_thickness"]
    substrate_d = p["substrate_depth"]
    substrate = mp.Block(
        center=mp.Vector3(c.x, c.y - film_t / 2 - substrate_d / 2, c.z),
        size=mp.Vector3(mp.inf, substrate_d, mp.inf),
        material=_medium(p, "substrate"),
    )
    film = mp.Block(
        center=mp.Vector3(c.x, c.y, c.z),
        size=mp.Vector3(mp.inf, film_t, mp.inf), material=_medium(p, "metal"),
    )
    return [substrate, film], _medium(p, "analyte"), {
        "film_normal": "y", "set_incidence_in_simulation": True,
    }


def _multilayer(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    total = sum(float(layer["thickness"]) for layer in p["layers"])
    cursor = c.y - total / 2
    objects = []
    interfaces = [cursor]
    for index, layer in enumerate(p["layers"]):
        thickness = float(layer["thickness"])
        centre_y = cursor + thickness / 2
        objects.append(
            mp.Block(
                center=mp.Vector3(c.x, centre_y, c.z),
                size=mp.Vector3(mp.inf, thickness, mp.inf),
                material=build_material(
                    layer["material"], wavelength_um=p["wavelength_um"],
                    allow_out_of_range=p["allow_out_of_range"],
                    allow_unstable=p["allow_unstable"],
                ),
                label=str(layer.get("label", f"layer_{index}")),
            )
        )
        cursor += thickness
        interfaces.append(cursor)
    return objects, _medium(p, "background"), {"interfaces_y_um": interfaces}


def _sphere(mp: Any, p: Mapping[str, Any]):
    obj = mp.Sphere(center=_centre(mp, p), radius=p["radius"], material=_medium(p, "particle"))
    return [obj], _medium(p, "background"), {"dipole_axis": "x"}


def _disk(mp: Any, p: Mapping[str, Any]):
    obj = mp.Cylinder(
        center=_centre(mp, p), radius=p["radius"],
        height=_cylinder_height(mp, p, p["height"]),
        axis=mp.Vector3(0, 0, 1), material=_medium(p, "particle"),
    )
    return [obj], _medium(p, "background"), {"axis": "z"}


def _cube(mp: Any, p: Mapping[str, Any]):
    depth = mp.inf if p["dimensions"] == 2 else p["size_z"]
    obj = mp.Block(
        center=_centre(mp, p), size=mp.Vector3(p["size_x"], p["size_y"], depth),
        material=_medium(p, "particle"),
    )
    return [obj], _medium(p, "background"), {}


def _ellipsoid(mp: Any, p: Mapping[str, Any]):
    obj = mp.Ellipsoid(
        center=_centre(mp, p), size=mp.Vector3(p["size_x"], p["size_y"], p["size_z"]),
        material=_medium(p, "particle"),
    )
    return [obj], _medium(p, "background"), {"long_axis": "largest size component"}


def _nanorod(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    material = _medium(p, "particle")
    radius = p["diameter"] / 2
    straight = p["length"] - p["diameter"]
    if p["dimensions"] == 3:
        obj = mp.Cylinder(
            center=c, radius=radius, height=p["length"],
            axis=mp.Vector3(1, 0, 0), material=material,
        )
        objects = [obj]
    else:
        objects = []
        if straight > 0:
            objects.append(
                mp.Block(
                    center=c, size=mp.Vector3(straight, p["diameter"], mp.inf),
                    material=material,
                )
            )
        offset = straight / 2
        objects.extend(
            [
                mp.Cylinder(
                    center=mp.Vector3(c.x - offset, c.y, c.z), radius=radius,
                    height=mp.inf, axis=mp.Vector3(0, 0, 1), material=material,
                ),
                mp.Cylinder(
                    center=mp.Vector3(c.x + offset, c.y, c.z), radius=radius,
                    height=mp.inf, axis=mp.Vector3(0, 0, 1), material=material,
                ),
            ]
        )
    return objects, _medium(p, "background"), {"long_axis": "x"}


def _sphere_dimer(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    offset = p["radius"] + p["gap"] / 2
    material = _medium(p, "particle")
    objects = [
        mp.Sphere(center=mp.Vector3(c.x - offset, c.y, c.z), radius=p["radius"], material=material),
        mp.Sphere(center=mp.Vector3(c.x + offset, c.y, c.z), radius=p["radius"], material=material),
    ]
    return objects, _medium(p, "background"), {"hotspot": (c.x, c.y, c.z)}


def _disk_dimer(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    offset = p["radius"] + p["gap"] / 2
    material = _medium(p, "particle")
    height = _cylinder_height(mp, p, p["height"])
    objects = [
        mp.Cylinder(center=mp.Vector3(c.x - offset, c.y, c.z), radius=p["radius"], height=height, material=material),
        mp.Cylinder(center=mp.Vector3(c.x + offset, c.y, c.z), radius=p["radius"], height=height, material=material),
    ]
    return objects, _medium(p, "background"), {"hotspot": (c.x, c.y, c.z)}


def _core_shell_sphere(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    objects = [
        mp.Sphere(center=c, radius=p["outer_radius"], material=_medium(p, "shell")),
        mp.Sphere(center=c, radius=p["core_radius"], material=_medium(p, "core")),
    ]
    return objects, _medium(p, "background"), {"shell_thickness_um": p["outer_radius"] - p["core_radius"]}


def _core_shell_cylinder(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    height = _cylinder_height(mp, p, p["height"])
    objects = [
        mp.Cylinder(center=c, radius=p["outer_radius"], height=height, material=_medium(p, "shell")),
        mp.Cylinder(center=c, radius=p["core_radius"], height=height, material=_medium(p, "core")),
    ]
    return objects, _medium(p, "background"), {"shell_thickness_um": p["outer_radius"] - p["core_radius"]}


def _nanoring(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    height = _cylinder_height(mp, p, p["height"])
    background = _medium(p, "background")
    objects = [
        mp.Cylinder(center=c, radius=p["outer_radius"], height=height, material=_medium(p, "ring")),
        mp.Cylinder(center=c, radius=p["inner_radius"], height=height, material=background),
    ]
    return objects, background, {"ring_width_um": p["outer_radius"] - p["inner_radius"]}


def _bowtie(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    half_gap = p["gap"] / 2
    length = p["triangle_length"]
    half_width = p["triangle_width"] / 2
    z0 = c.z - p["height"] / 2
    left = [
        mp.Vector3(c.x - half_gap, c.y, z0),
        mp.Vector3(c.x - half_gap - length, c.y + half_width, z0),
        mp.Vector3(c.x - half_gap - length, c.y - half_width, z0),
    ]
    right = [
        mp.Vector3(c.x + half_gap, c.y, z0),
        mp.Vector3(c.x + half_gap + length, c.y - half_width, z0),
        mp.Vector3(c.x + half_gap + length, c.y + half_width, z0),
    ]
    material = _medium(p, "particle")
    height = mp.inf if p["dimensions"] == 2 else p["height"]
    objects = [
        mp.Prism(vertices=left, height=height, axis=mp.Vector3(0, 0, 1), material=material),
        mp.Prism(vertices=right, height=height, axis=mp.Vector3(0, 0, 1), material=material),
    ]
    return objects, _medium(p, "background"), {"hotspot": (c.x, c.y, c.z)}


def _nanohole(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    film = mp.Block(
        center=c, size=mp.Vector3(mp.inf, p["film_thickness"], mp.inf),
        material=_medium(p, "metal"),
    )
    if p["dimensions"] == 3:
        hole = mp.Cylinder(
            center=c, radius=p["hole_radius"], height=p["film_thickness"] * 1.02,
            axis=mp.Vector3(0, 1, 0), material=_medium(p, "hole_material"),
        )
    else:
        hole = mp.Cylinder(
            center=c, radius=p["hole_radius"], height=mp.inf,
            axis=mp.Vector3(0, 0, 1), material=_medium(p, "hole_material"),
        )
    return [film, hole], _medium(p, "background"), {"aperture_axis": "y" if p["dimensions"] == 3 else "z"}


def _nanoslit(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    film = mp.Block(
        center=c, size=mp.Vector3(mp.inf, p["film_thickness"], mp.inf),
        material=_medium(p, "metal"),
    )
    depth = mp.inf if p["dimensions"] == 2 else p["slit_depth"]
    slit = mp.Block(
        center=c,
        size=mp.Vector3(p["slit_width"], p["film_thickness"] * 1.02, depth),
        material=_medium(p, "slit_material"),
    )
    return [film, slit], _medium(p, "background"), {"slit_axis": "z"}


def _ridge_grating(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    material = _medium(p, "metal")
    objects = []
    if p["base_thickness"] > 0:
        objects.append(
            mp.Block(
                center=mp.Vector3(c.x, c.y - p["base_thickness"] / 2, c.z),
                size=mp.Vector3(mp.inf, p["base_thickness"], mp.inf), material=material,
            )
        )
    ridge_width = p["period"] * p["fill_factor"]
    start = -(p["periods"] - 1) * p["period"] / 2
    for index in range(p["periods"]):
        x = c.x + start + index * p["period"]
        objects.append(
            mp.Block(
                center=mp.Vector3(x, c.y + p["ridge_height"] / 2, c.z),
                size=mp.Vector3(ridge_width, p["ridge_height"], mp.inf), material=material,
            )
        )
    return objects, _medium(p, "background"), {
        "suggested_cell_size_um": (p["period"], None, None),
        "ridge_width_um": ridge_width,
    }


def _groove_grating(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    metal_depth = p["metal_depth"]
    objects = [
        mp.Block(
            center=mp.Vector3(c.x, c.y - metal_depth / 2, c.z),
            size=mp.Vector3(mp.inf, metal_depth, mp.inf), material=_medium(p, "metal"),
        )
    ]
    groove_width = p["period"] * p["fill_factor"]
    start = -(p["periods"] - 1) * p["period"] / 2
    fill = _medium(p, "groove_material")
    for index in range(p["periods"]):
        x = c.x + start + index * p["period"]
        objects.append(
            mp.Block(
                center=mp.Vector3(x, c.y - p["groove_depth"] / 2, c.z),
                size=mp.Vector3(groove_width, p["groove_depth"] * 1.001, mp.inf),
                material=fill,
            )
        )
    return objects, _medium(p, "background"), {
        "suggested_cell_size_um": (p["period"], None, None),
        "groove_width_um": groove_width,
    }


def _particle_array(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    material = _medium(p, "particle")
    if p["shape"] == "sphere":
        obj = mp.Sphere(center=c, radius=p["radius"], material=material)
    elif p["shape"] == "disk":
        obj = mp.Cylinder(
            center=c, radius=p["radius"], height=_cylinder_height(mp, p, p["height"]),
            axis=mp.Vector3(0, 0, 1), material=material,
        )
    else:
        depth = mp.inf if p["dimensions"] == 2 else p["height"]
        obj = mp.Block(center=c, size=mp.Vector3(p["size"], p["size"], depth), material=material)
    axes = ("x",) if p["dimensions"] == 2 else ("x", "z")
    return [obj], _medium(p, "background"), {
        "periodic_axes": axes,
        "suggested_cell_size_um": (p["period_x"], None, p["period_z"]),
    }


def _particle_line_on_substrate(mp: Any, p: Mapping[str, Any]):
    positions = particle_line_positions(p)
    substrate = mp.Block(
        center=mp.Vector3(
            float(p["center"][0]),
            float(p["surface_y"]) - float(p["substrate_thickness"]) / 2.0,
            float(p["center"][2]),
        ),
        size=mp.Vector3(mp.inf, float(p["substrate_thickness"]), mp.inf),
        material=_medium(p, "substrate"),
    )
    objects = [substrate]
    if p["include_particles"]:
        particle_material = _medium(p, "particle")
        for item in positions:
            centre = mp.Vector3(item.actual_x_um, item.actual_y_um, item.actual_z_um)
            if p["particle_shape"] == "sphere":
                objects.append(
                    mp.Sphere(center=centre, radius=float(p["radius"]), material=particle_material)
                )
            else:
                size = float(p["cube_size"])
                objects.append(
                    mp.Block(center=centre, size=mp.Vector3(size, size, size), material=particle_material)
                )
    return objects, _medium(p, "background"), {
        "line_axis": "x",
        "surface_normal": "y",
        "surface_y_um": float(p["surface_y"]),
        "particle_positions": [item.to_dict() for item in positions],
        "reference_kind": "substrate_only" if not p["include_particles"] else "particle_line",
    }


def _nanohole_array(mp: Any, p: Mapping[str, Any]):
    objects, default, hints = _nanohole(mp, p)
    axes = ("x",) if p["dimensions"] == 2 else ("x", "z")
    hints.update(
        periodic_axes=axes,
        suggested_cell_size_um=(p["period_x"], None, p["period_z"]),
    )
    return objects, default, hints


def _mim(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    offset = (p["gap"] + p["metal_thickness"]) / 2
    metal = _medium(p, "metal")
    objects = [
        mp.Block(
            center=mp.Vector3(c.x, c.y - offset, c.z),
            size=mp.Vector3(mp.inf, p["metal_thickness"], mp.inf), material=metal,
        ),
        mp.Block(
            center=mp.Vector3(c.x, c.y + offset, c.z),
            size=mp.Vector3(mp.inf, p["metal_thickness"], mp.inf), material=metal,
        ),
    ]
    return objects, _medium(p, "dielectric"), {"propagation_axis": "x", "slot_width_um": p["gap"]}


def _npom(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    mirror = mp.Block(
        center=mp.Vector3(c.x, c.y - p["mirror_depth"] / 2, c.z),
        size=mp.Vector3(mp.inf, p["mirror_depth"], mp.inf), material=_medium(p, "mirror"),
    )
    spacer = mp.Block(
        center=mp.Vector3(c.x, c.y + p["gap"] / 2, c.z),
        size=mp.Vector3(mp.inf, p["gap"], mp.inf), material=_medium(p, "spacer"),
    )
    particle = mp.Sphere(
        center=mp.Vector3(c.x, c.y + p["gap"] + p["radius"], c.z),
        radius=p["radius"], material=_medium(p, "particle"),
    )
    return [mirror, spacer, particle], _medium(p, "background"), {
        "gap_monitor_y_um": c.y + p["gap"] / 2,
    }


def _ncom(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    mirror = mp.Block(
        center=mp.Vector3(c.x, c.y - p["mirror_depth"] / 2, c.z),
        size=mp.Vector3(mp.inf, p["mirror_depth"], mp.inf), material=_medium(p, "mirror"),
    )
    spacer = mp.Block(
        center=mp.Vector3(c.x, c.y + p["gap"] / 2, c.z),
        size=mp.Vector3(mp.inf, p["gap"], mp.inf), material=_medium(p, "spacer"),
    )
    depth = mp.inf if p["dimensions"] == 2 else p["size"]
    cube = mp.Block(
        center=mp.Vector3(c.x, c.y + p["gap"] + p["size"] / 2, c.z),
        size=mp.Vector3(p["size"], p["size"], depth), material=_medium(p, "particle"),
    )
    return [mirror, spacer, cube], _medium(p, "background"), {
        "gap_monitor_y_um": c.y + p["gap"] / 2,
    }


def _tip_substrate(mp: Any, p: Mapping[str, Any]):
    c = _centre(mp, p)
    substrate = mp.Block(
        center=mp.Vector3(c.x, c.y - p["substrate_depth"] / 2, c.z),
        size=mp.Vector3(mp.inf, p["substrate_depth"], mp.inf),
        material=_medium(p, "substrate"),
    )
    # ``radius2`` is at the end pointed to by ``axis``.  A -y axis therefore
    # places the small apex above the substrate and the wide base above it.
    tip_center_y = c.y + p["gap"] + p["tip_length"] / 2
    tip = mp.Cone(
        center=mp.Vector3(c.x, tip_center_y, c.z),
        radius=p["tip_radius"], radius2=p["apex_radius"],
        height=p["tip_length"], axis=mp.Vector3(0, -1, 0), material=_medium(p, "tip"),
    )
    return [substrate, tip], _medium(p, "background"), {
        "gap_monitor_y_um": c.y + p["gap"] / 2,
        "verify_tip_orientation": True,
    }


_BUILDERS: dict[str, Builder] = {
    "planar_interface": _planar_interface,
    "metal_film": _metal_film,
    "kretschmann_stack": _kretschmann,
    "multilayer_stack": _multilayer,
    "nanoparticle_sphere": _sphere,
    "nanoparticle_disk": _disk,
    "nanocube": _cube,
    "ellipsoid": _ellipsoid,
    "nanorod": _nanorod,
    "sphere_dimer": _sphere_dimer,
    "disk_dimer": _disk_dimer,
    "core_shell_sphere": _core_shell_sphere,
    "core_shell_cylinder": _core_shell_cylinder,
    "nanoring": _nanoring,
    "bowtie_antenna": _bowtie,
    "nanohole_film": _nanohole,
    "nanoslit_film": _nanoslit,
    "ridge_grating": _ridge_grating,
    "groove_grating": _groove_grating,
    "particle_array": _particle_array,
    "particle_line_on_substrate": _particle_line_on_substrate,
    "nanohole_array": _nanohole_array,
    "mim_waveguide": _mim,
    "nanoparticle_on_mirror": _npom,
    "nanocube_on_mirror": _ncom,
    "tip_substrate": _tip_substrate,
}


if set(_BUILDERS) != set(GEOMETRIES):
    missing = set(GEOMETRIES) - set(_BUILDERS)
    extra = set(_BUILDERS) - set(GEOMETRIES)
    raise RuntimeError(f"Geometry registry/builder mismatch; missing={missing}, extra={extra}")


__all__ = [
    "COMMON_PARAMETERS",
    "GEOMETRIES",
    "GeometryBuild",
    "GeometryError",
    "GeometrySpec",
    "ParameterSpec",
    "build_geometry",
    "build_geometry_from_config",
    "geometry_catalog",
    "get_geometry",
    "get_geometry_spec",
    "list_geometries",
    "validate_geometry_parameters",
]
