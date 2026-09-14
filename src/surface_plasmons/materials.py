"""
All lengths and wavelengths are expressed in micrometres (um).  The registry
stores metadata and aliases without importing MEEP, so menus and configuration
validation still work on a machine where MEEP is not active.  ``build_material``
performs the MEEP import only when a simulation is actually being assembled.

Dispersive entries deliberately reuse ``meep.materials``.  This keeps the
Lorentz/Drude coefficients in one authoritative place and avoids replacing a
wavelength-dependent metal with a single complex permittivity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from importlib import import_module
from numbers import Real
from typing import Any, Iterable, Sequence
import difflib
import re


MEEP_MATERIALS_URL = "https://meep.readthedocs.io/en/latest/Materials/"
MEEP_MATERIALS_SOURCE_URL = (
    "https://github.com/NanoComp/meep/blob/master/python/materials.py"
)


class MaterialError(ValueError):
    """Base exception for invalid material names or settings."""


class MaterialUnavailableError(RuntimeError):
    """Raised when a registered MEEP material is absent from the installation."""


class MaterialCategory(str, Enum):
    AMBIENT = "ambient"
    METAL = "metal"
    DIELECTRIC = "dielectric"
    SEMICONDUCTOR = "semiconductor"
    POLYMER = "polymer"
    ANISOTROPIC = "anisotropic"


class MaterialModel(str, Enum):
    CONSTANT_INDEX = "constant_index"
    CONSTANT_EPSILON = "constant_epsilon"
    LORENTZ = "lorentz"
    DRUDE_LORENTZ = "drude_lorentz"
    SELLMEIER_LORENTZ = "sellmeier_lorentz"
    ANISOTROPIC_LORENTZ = "anisotropic_lorentz"


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    """One material definition and the metadata needed to use it safely."""

    key: str
    display_name: str
    formula: str
    category: MaterialCategory
    model: MaterialModel
    aliases: tuple[str, ...] = ()
    meep_name: str | None = None
    refractive_index: float | None = None
    epsilon: float | None = None
    valid_wavelength_um: tuple[float, float] | None = None
    stable: bool = True
    reference: str = ""
    notes: str = ""

    @property
    def is_dispersive(self) -> bool:
        return self.meep_name is not None

    def supports_wavelength(self, wavelength_um: float) -> bool:
        _require_positive_number("wavelength_um", wavelength_um)
        if self.valid_wavelength_um is None:
            return True
        low, high = self.valid_wavelength_um
        return low <= float(wavelength_um) <= high

    def supports_band(self, wavelengths_um: Sequence[float]) -> bool:
        values = _normalise_wavelengths(wavelengths_um)
        return all(self.supports_wavelength(value) for value in values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["category"] = self.category.value
        result["model"] = self.model.value
        result["is_dispersive"] = self.is_dispersive
        return result


def _constant_index(
    key: str,
    display_name: str,
    formula: str,
    index: float,
    category: MaterialCategory,
    *,
    aliases: tuple[str, ...] = (),
    notes: str = "",
) -> MaterialSpec:
    return MaterialSpec(
        key=key,
        display_name=display_name,
        formula=formula,
        category=category,
        model=MaterialModel.CONSTANT_INDEX,
        aliases=aliases,
        refractive_index=index,
        reference="Constant-index approximation; replace for precision spectroscopy.",
        notes=notes,
    )


def _meep(
    key: str,
    display_name: str,
    formula: str,
    category: MaterialCategory,
    model: MaterialModel,
    meep_name: str,
    wavelength_range: tuple[float, float],
    reference: str,
    *,
    aliases: tuple[str, ...] = (),
    stable: bool = True,
    notes: str = "",
) -> MaterialSpec:
    return MaterialSpec(
        key=key,
        display_name=display_name,
        formula=formula,
        category=category,
        model=model,
        aliases=aliases,
        meep_name=meep_name,
        valid_wavelength_um=wavelength_range,
        stable=stable,
        reference=reference,
        notes=notes,
    )


RAKIC_1998 = "Rakic et al., Applied Optics 37, 5271-5283 (1998)."
JC_1972 = "Johnson and Christy, Physical Review B 6, 4370-4379 (1972)."
HORIBA = "Horiba optical-model technical-note fit supplied by MEEP."
MEEP_LIBRARY = "Published optical fit supplied by the MEEP materials library."


# Canonical keys are lowercase snake_case.  Aliases are resolved case- and
# punctuation-insensitively by ``get_material_spec``.
MATERIALS: dict[str, MaterialSpec] = {
    # Ambient and useful simple approximations
    "vacuum": _constant_index(
        "vacuum", "Vacuum", "vacuum", 1.0, MaterialCategory.AMBIENT,
        aliases=("empty_space",),
    ),
    "air": _constant_index(
        "air", "Air", "air", 1.000277, MaterialCategory.AMBIENT,
        aliases=("atmosphere",),
        notes="Approximate dry-air index near visible wavelengths.",
    ),
    "water": _constant_index(
        "water", "Water", "H2O", 1.333, MaterialCategory.AMBIENT,
        aliases=("h2o", "aqueous"),
        notes="Approximate room-temperature visible index.",
    ),
    "glass_constant": _constant_index(
        "glass_constant", "Generic glass", "glass", 1.50,
        MaterialCategory.DIELECTRIC, aliases=("generic_glass",),
    ),
    "immersion_oil": _constant_index(
        "immersion_oil", "Immersion oil", "oil", 1.515,
        MaterialCategory.AMBIENT, aliases=("oil",),
    ),
    "glycerol": _constant_index(
        "glycerol", "Glycerol", "C3H8O3", 1.473,
        MaterialCategory.AMBIENT,
    ),

    # Semiconductors, glasses, dielectrics, and polymers from meep.materials
    "csi": _meep(
        "csi", "Crystalline silicon", "Si", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.LORENTZ, "cSi", (0.4, 1.0), MEEP_LIBRARY,
        aliases=("crystalline_silicon", "c_si"),
    ),
    "asi": _meep(
        "asi", "Amorphous silicon", "a-Si", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.LORENTZ, "aSi", (0.21, 0.83), HORIBA,
        aliases=("amorphous_silicon", "a_si"),
    ),
    "asi_h": _meep(
        "asi_h", "Hydrogenated amorphous silicon", "a-Si:H",
        MaterialCategory.SEMICONDUCTOR, MaterialModel.LORENTZ, "aSi_H",
        (0.21, 0.83), HORIBA, aliases=("hydrogenated_amorphous_silicon",),
    ),
    "ito": _meep(
        "ito", "Indium tin oxide", "ITO", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.LORENTZ, "ITO", (0.21, 0.83), HORIBA,
        aliases=("indium_tin_oxide",),
    ),
    "al2o3": _meep(
        "al2o3", "Alumina", "Al2O3", MaterialCategory.DIELECTRIC,
        MaterialModel.LORENTZ, "Al2O3", (0.21, 2.07), HORIBA,
        aliases=("alumina",),
    ),
    "aln": _meep(
        "aln", "Aluminium nitride", "AlN", MaterialCategory.DIELECTRIC,
        MaterialModel.LORENTZ, "AlN", (0.26, 1.65), HORIBA,
        aliases=("aluminium_nitride", "aluminum_nitride"),
    ),
    "alas": _meep(
        "alas", "Aluminium arsenide", "AlAs", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.SELLMEIER_LORENTZ, "AlAs", (0.56, 2.2), MEEP_LIBRARY,
        aliases=("aluminium_arsenide", "aluminum_arsenide"),
    ),
    "bk7": _meep(
        "bk7", "BK7 borosilicate glass", "BK7", MaterialCategory.DIELECTRIC,
        MaterialModel.SELLMEIER_LORENTZ, "BK7", (0.3, 2.5),
        "SCHOTT Zemax catalogue fit supplied by MEEP.",
        aliases=("borosilicate", "bk7_glass"),
    ),
    "fused_quartz": _meep(
        "fused_quartz", "Fused quartz", "SiO2", MaterialCategory.DIELECTRIC,
        MaterialModel.SELLMEIER_LORENTZ, "fused_quartz", (0.21, 6.7),
        "Malitson, Journal of the Optical Society of America 55, 1205-1209 (1965).",
        aliases=("silica", "fused_silica", "quartz", "glass"),
    ),
    "gaas": _meep(
        "gaas", "Gallium arsenide", "GaAs", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.SELLMEIER_LORENTZ, "GaAs", (0.97, 17.0), MEEP_LIBRARY,
        aliases=("gallium_arsenide",),
    ),
    "si3n4_visnir": _meep(
        "si3n4_visnir", "Silicon nitride (VIS-NIR)", "Si3N4",
        MaterialCategory.DIELECTRIC, MaterialModel.LORENTZ, "Si3N4_VISNIR",
        (0.207, 1.24), MEEP_LIBRARY,
    ),
    "si3n4_nir": _meep(
        "si3n4_nir", "Silicon nitride (broad NIR fit)", "Si3N4",
        MaterialCategory.DIELECTRIC, MaterialModel.LORENTZ, "Si3N4_NIR",
        (0.310, 5.504), MEEP_LIBRARY,
    ),
    "sin": _meep(
        "sin", "Non-stoichiometric silicon nitride", "SiN",
        MaterialCategory.DIELECTRIC, MaterialModel.LORENTZ, "SiN",
        (0.21, 2.07), HORIBA, aliases=("silicon_nitride",),
    ),
    "si3n4": _meep(
        "si3n4", "Stoichiometric silicon nitride", "Si3N4",
        MaterialCategory.DIELECTRIC, MaterialModel.LORENTZ, "Si3N4",
        (0.23, 0.83), HORIBA,
    ),
    "sio2": _meep(
        "sio2", "Silicon dioxide thin-film fit", "SiO2",
        MaterialCategory.DIELECTRIC, MaterialModel.LORENTZ, "SiO2",
        (0.25, 1.77), HORIBA, aliases=("silicon_dioxide",),
    ),
    "inp": _meep(
        "inp", "Indium phosphide", "InP", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.SELLMEIER_LORENTZ, "InP", (0.95, 10.0), MEEP_LIBRARY,
        aliases=("indium_phosphide",),
    ),
    "ge": _meep(
        "ge", "Germanium", "Ge", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.SELLMEIER_LORENTZ, "Ge", (2.5, 12.0), MEEP_LIBRARY,
        aliases=("germanium",),
    ),
    "si_ir": _meep(
        "si_ir", "Silicon (infrared fit)", "Si", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.SELLMEIER_LORENTZ, "Si", (1.36, 11.0), MEEP_LIBRARY,
        aliases=("silicon_ir", "silicon"),
    ),
    "pmma": _meep(
        "pmma", "Poly(methyl methacrylate)", "PMMA", MaterialCategory.POLYMER,
        MaterialModel.SELLMEIER_LORENTZ, "PMMA", (0.437, 1.052), MEEP_LIBRARY,
        aliases=("acrylic",),
    ),
    "pc": _meep(
        "pc", "Polycarbonate", "PC", MaterialCategory.POLYMER,
        MaterialModel.SELLMEIER_LORENTZ, "PC", (0.437, 1.052), MEEP_LIBRARY,
        aliases=("polycarbonate",),
    ),
    "ps": _meep(
        "ps", "Polystyrene", "PS", MaterialCategory.POLYMER,
        MaterialModel.SELLMEIER_LORENTZ, "PS", (0.437, 1.052), MEEP_LIBRARY,
        aliases=("polystyrene",),
    ),
    "cellulose": _meep(
        "cellulose", "Cellulose", "CLS", MaterialCategory.POLYMER,
        MaterialModel.SELLMEIER_LORENTZ, "CLS", (0.437, 1.052), MEEP_LIBRARY,
        aliases=("cls",),
    ),
    "tio2": _meep(
        "tio2", "Titanium dioxide", "TiO2", MaterialCategory.DIELECTRIC,
        MaterialModel.LORENTZ, "TiO2", (0.4, 0.7), MEEP_LIBRARY,
        aliases=("titania", "titanium_dioxide"),
    ),
    "ta2o5": _meep(
        "ta2o5", "Tantalum pentoxide", "Ta2O5", MaterialCategory.DIELECTRIC,
        MaterialModel.LORENTZ, "Ta2O5", (0.4, 0.8), MEEP_LIBRARY,
        aliases=("tantalum_pentoxide",),
    ),
    "hfo2": _meep(
        "hfo2", "Hafnium dioxide", "HfO2", MaterialCategory.DIELECTRIC,
        MaterialModel.LORENTZ, "HfO2", (0.2, 2.0), MEEP_LIBRARY,
        aliases=("hafnia", "hafnium_dioxide"),
    ),

    # Stable broad-band Lorentz-Drude metals.  These are the normal defaults.
    "ag": _meep(
        "ag", "Silver", "Ag", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Ag", (0.24797, 12.398), RAKIC_1998,
        aliases=("silver",),
    ),
    "au": _meep(
        "au", "Gold", "Au", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Au", (0.24797, 6.1992), RAKIC_1998,
        aliases=("gold",),
    ),
    "cu": _meep(
        "cu", "Copper", "Cu", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Cu", (0.20664, 12.398), RAKIC_1998,
        aliases=("copper",),
    ),
    "al": _meep(
        "al", "Aluminium", "Al", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Al", (0.20664, 12.398), RAKIC_1998,
        aliases=("aluminium", "aluminum"),
    ),
    "be": _meep(
        "be", "Beryllium", "Be", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Be", (0.20664, 12.398), RAKIC_1998,
        aliases=("beryllium",),
    ),
    "cr": _meep(
        "cr", "Chromium", "Cr", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Cr", (0.20664, 12.398), RAKIC_1998,
        aliases=("chromium",),
    ),
    "ni": _meep(
        "ni", "Nickel", "Ni", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Ni", (0.20664, 12.398), RAKIC_1998,
        aliases=("nickel",),
    ),
    "pd": _meep(
        "pd", "Palladium", "Pd", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Pd", (0.20664, 12.398), RAKIC_1998,
        aliases=("palladium",),
    ),
    "pt": _meep(
        "pt", "Platinum", "Pt", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Pt", (0.20664, 12.398), RAKIC_1998,
        aliases=("platinum",),
    ),
    "ti": _meep(
        "ti", "Titanium", "Ti", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Ti", (0.20664, 12.398), RAKIC_1998,
        aliases=("titanium",),
    ),
    "w": _meep(
        "w", "Tungsten", "W", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "W", (0.20664, 12.398), RAKIC_1998,
        aliases=("tungsten", "wolfram"),
    ),

    # Alternative visible-band and Drude-only fits.
    "au_jc_visible": _meep(
        "au_jc_visible", "Gold (Johnson-Christy visible fit)", "Au",
        MaterialCategory.METAL, MaterialModel.DRUDE_LORENTZ, "Au_JC_visible",
        (0.4, 0.8), JC_1972, aliases=("gold_jc", "gold_visible_jc"),
    ),
    "au_visible": _meep(
        "au_visible", "Gold (Palik visible fit)", "Au", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Au_visible", (0.4, 0.8),
        "Palik optical constants; fit supplied by MEEP.",
    ),
    "ag_visible": _meep(
        "ag_visible", "Silver (Palik visible fit)", "Ag", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Ag_visible", (0.4, 0.8),
        "Palik optical constants; fit supplied by MEEP.", stable=False,
        notes="MEEP marks this fit unstable; field divergence may occur.",
    ),
    "al_visible": _meep(
        "al_visible", "Aluminium (Palik visible fit)", "Al",
        MaterialCategory.METAL, MaterialModel.DRUDE_LORENTZ, "Al_visible",
        (0.4, 0.8), "Palik optical constants; fit supplied by MEEP.",
        stable=False, notes="MEEP marks this fit unstable; field divergence may occur.",
    ),
    "cr_visible": _meep(
        "cr_visible", "Chromium (Palik visible fit)", "Cr",
        MaterialCategory.METAL, MaterialModel.DRUDE_LORENTZ, "Cr_visible",
        (0.4, 0.8), "Palik optical constants; fit supplied by MEEP.",
    ),
    "ti_visible": _meep(
        "ti_visible", "Titanium (Palik visible fit)", "Ti",
        MaterialCategory.METAL, MaterialModel.DRUDE_LORENTZ, "Ti_visible",
        (0.4, 0.8), "Palik optical constants; fit supplied by MEEP.",
    ),
    "al_drude": _meep(
        "al_drude", "Aluminium (Drude-only fit)", "Al", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Al_drude", (0.19, 0.83), HORIBA,
    ),
    "co": _meep(
        "co", "Cobalt", "Co", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Co", (0.26, 1.65), HORIBA,
        aliases=("cobalt",),
    ),
    "mo": _meep(
        "mo", "Molybdenum", "Mo", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Mo", (0.25, 0.83), HORIBA,
        aliases=("molybdenum",), stable=False,
        notes="MEEP marks this fit unstable; field divergence may occur.",
    ),
    "nicr": _meep(
        "nicr", "Nickel-chromium alloy", "NiCr", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "NiCr", (0.25, 0.83), HORIBA,
        aliases=("nickel_chromium", "nichrome"),
    ),
    "nife": _meep(
        "nife", "Nickel-iron alloy", "NiFe", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "NiFe", (0.25, 0.83), HORIBA,
        aliases=("nickel_iron",),
    ),
    "ti_drude": _meep(
        "ti_drude", "Titanium (Drude-only fit)", "Ti", MaterialCategory.METAL,
        MaterialModel.DRUDE_LORENTZ, "Ti_drude", (0.21, 1.24), HORIBA,
    ),

    # Anisotropic and specialist optical materials available in MEEP.
    "bbo": _meep(
        "bbo", "Beta barium borate", "BaB2O4", MaterialCategory.ANISOTROPIC,
        MaterialModel.ANISOTROPIC_LORENTZ, "BaB2O4", (0.188, 5.2), MEEP_LIBRARY,
        aliases=("bab2o4", "beta_barium_borate"),
    ),
    "linbo3": _meep(
        "linbo3", "Lithium niobate", "LiNbO3", MaterialCategory.ANISOTROPIC,
        MaterialModel.ANISOTROPIC_LORENTZ, "LiNbO3", (0.4, 5.0), MEEP_LIBRARY,
        aliases=("lithium_niobate",),
    ),
    "cawo4": _meep(
        "cawo4", "Calcium tungstate", "CaWO4", MaterialCategory.ANISOTROPIC,
        MaterialModel.ANISOTROPIC_LORENTZ, "CaWO4", (0.45, 4.0), MEEP_LIBRARY,
    ),
    "caco3": _meep(
        "caco3", "Calcium carbonate", "CaCO3", MaterialCategory.ANISOTROPIC,
        MaterialModel.ANISOTROPIC_LORENTZ, "CaCO3", (0.204, 2.172), MEEP_LIBRARY,
        aliases=("calcite",),
    ),
    "sio2_aniso": _meep(
        "sio2_aniso", "Anisotropic silicon dioxide", "SiO2",
        MaterialCategory.ANISOTROPIC, MaterialModel.ANISOTROPIC_LORENTZ,
        "SiO2_aniso", (0.198, 2.0531), MEEP_LIBRARY,
    ),
    "gan": _meep(
        "gan", "Wurtzite gallium nitride", "GaN", MaterialCategory.ANISOTROPIC,
        MaterialModel.ANISOTROPIC_LORENTZ, "GaN", (0.35, 10.0), MEEP_LIBRARY,
        aliases=("gallium_nitride",),
    ),
    "aln_aniso": _meep(
        "aln_aniso", "Anisotropic aluminium nitride", "AlN",
        MaterialCategory.ANISOTROPIC, MaterialModel.ANISOTROPIC_LORENTZ,
        "AlN_aniso", (0.22, 5.0), MEEP_LIBRARY,
    ),
    "al2o3_aniso": _meep(
        "al2o3_aniso", "Sapphire", "Al2O3", MaterialCategory.ANISOTROPIC,
        MaterialModel.ANISOTROPIC_LORENTZ, "Al2O3_aniso", (0.2, 5.0),
        MEEP_LIBRARY, aliases=("sapphire",),
    ),
    "y2o3": _meep(
        "y2o3", "Yttrium oxide", "Y2O3", MaterialCategory.DIELECTRIC,
        MaterialModel.SELLMEIER_LORENTZ, "Y2O3", (0.25, 9.6), MEEP_LIBRARY,
    ),
    "yag": _meep(
        "yag", "Yttrium aluminium garnet", "Y3Al5O12",
        MaterialCategory.DIELECTRIC, MaterialModel.SELLMEIER_LORENTZ, "YAG",
        (0.4, 5.0), MEEP_LIBRARY,
    ),
    "cdte": _meep(
        "cdte", "Cadmium telluride", "CdTe", MaterialCategory.SEMICONDUCTOR,
        MaterialModel.SELLMEIER_LORENTZ, "CdTe", (0.86, 2.5), MEEP_LIBRARY,
        aliases=("cadmium_telluride",),
    ),
}


def _normalise_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise MaterialError("Material name must be a non-empty string.")
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _build_alias_map() -> dict[str, str]:
    # Canonical keys get first priority.  This matters when a formula names
    # several fits: ``sio2`` must resolve to the explicit ``sio2`` entry rather
    # than whichever SiO2-based material happened to be declared first.
    aliases: dict[str, str] = {
        _normalise_name(key): key for key in MATERIALS
    }
    for key, spec in MATERIALS.items():
        for candidate in (spec.display_name, spec.formula, *spec.aliases):
            normalised = _normalise_name(candidate)
            previous = aliases.get(normalised)
            if previous is not None and previous != key:
                # Formula collisions (for example several SiO2 fits) should not
                # silently change an already explicit mapping.
                continue
            aliases[normalised] = key
    return aliases


_ALIASES = _build_alias_map()


def get_material_spec(name: str) -> MaterialSpec:
    """Return a material record using a canonical key or friendly alias."""

    normalised = _normalise_name(name)
    key = _ALIASES.get(normalised)
    if key is not None:
        return MATERIALS[key]

    choices = sorted(MATERIALS)
    matches = difflib.get_close_matches(name.lower(), choices, n=3, cutoff=0.45)
    suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
    raise MaterialError(f"Unknown material {name!r}.{suggestion}")


def list_materials(
    *,
    category: MaterialCategory | str | None = None,
    model: MaterialModel | str | None = None,
    stable_only: bool = False,
) -> list[MaterialSpec]:
    """Return sorted material records, optionally filtered by metadata."""

    category_value = MaterialCategory(category) if category is not None else None
    model_value = MaterialModel(model) if model is not None else None
    results = []
    for spec in MATERIALS.values():
        if category_value is not None and spec.category is not category_value:
            continue
        if model_value is not None and spec.model is not model_value:
            continue
        if stable_only and not spec.stable:
            continue
        results.append(spec)
    return sorted(results, key=lambda item: item.key)


def material_catalog(*, stable_only: bool = False) -> dict[str, dict[str, Any]]:
    """Return a JSON-serialisable material catalogue for a UI or API."""

    return {spec.key: spec.to_dict() for spec in list_materials(stable_only=stable_only)}


def list_backend_material_names() -> list[str]:
    """List every name exposed by the installed MEEP materials library.

    On recent MEEP versions this includes entries added after this project's
    static metadata was released, which keeps the database forward-compatible.
    """

    try:
        backend = import_module("meep.materials")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaterialUnavailableError(
            "MEEP is not importable. Run this inside the WSL/Miniforge "
            "environment where pymeep is installed."
        ) from exc

    library = getattr(backend, "materials_library", None)
    if isinstance(library, dict):
        return sorted(str(name) for name in library)
    return sorted({spec.meep_name for spec in MATERIALS.values() if spec.meep_name})


def build_material(
    name: str,
    *,
    wavelength_um: float | Sequence[float] | None = None,
    allow_out_of_range: bool = False,
    allow_unstable: bool = False,
) -> Any:
    """Build and return a ``meep.Medium`` for a registered material.

    ``wavelength_um`` may be a single wavelength or a sequence representing a
    simulation band.  Dispersive fits are rejected outside their published
    range unless ``allow_out_of_range=True`` is explicitly supplied.
    """

    spec = get_material_spec(name)
    if not spec.stable and not allow_unstable:
        raise MaterialError(
            f"{spec.key!r} is marked unstable by MEEP. Choose a stable broad-band "
            "model or pass allow_unstable=True after checking convergence."
        )

    if wavelength_um is not None and not allow_out_of_range:
        wavelengths = _normalise_wavelengths(wavelength_um)
        outside = [value for value in wavelengths if not spec.supports_wavelength(value)]
        if outside:
            low, high = spec.valid_wavelength_um or (None, None)
            raise MaterialError(
                f"{spec.key!r} is valid from {low:g} to {high:g} um, but received "
                f"{min(outside):g} to {max(outside):g} um."
            )

    try:
        mp = import_module("meep")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaterialUnavailableError(
            "MEEP is not importable. Select the WSL/Miniforge Python interpreter "
            "that contains pymeep."
        ) from exc

    if spec.meep_name is None:
        if spec.refractive_index is not None:
            return mp.Medium(index=spec.refractive_index)
        if spec.epsilon is not None:
            return mp.Medium(epsilon=spec.epsilon)
        raise MaterialUnavailableError(f"{spec.key!r} has no material constructor.")

    try:
        backend = import_module("meep.materials")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaterialUnavailableError(
            "The installed pymeep does not provide meep.materials."
        ) from exc

    medium = getattr(backend, spec.meep_name, None)
    if medium is None:
        getter = getattr(backend, "get_material", None)
        if callable(getter):
            try:
                medium = getter(spec.meep_name)
            except (KeyError, ValueError):
                medium = None
    if medium is None:
        raise MaterialUnavailableError(
            f"Material {spec.meep_name!r} is not available in this MEEP version. "
            "Update pymeep or choose another registered material."
        )
    return medium


def make_constant_material(
    *,
    refractive_index: float | None = None,
    epsilon: float | None = None,
) -> Any:
    """Create an unregistered nondispersive medium for a controlled experiment."""

    if (refractive_index is None) == (epsilon is None):
        raise MaterialError("Supply exactly one of refractive_index or epsilon.")
    value = refractive_index if refractive_index is not None else epsilon
    _require_positive_number("material value", value)
    try:
        mp = import_module("meep")
    except (ImportError, ModuleNotFoundError) as exc:
        raise MaterialUnavailableError("MEEP is not importable.") from exc
    if refractive_index is not None:
        return mp.Medium(index=float(refractive_index))
    return mp.Medium(epsilon=float(epsilon))


def validate_material_band(
    names: Iterable[str], wavelengths_um: float | Sequence[float]
) -> dict[str, list[float]]:
    """Return out-of-range wavelengths for each selected material."""

    wavelengths = _normalise_wavelengths(wavelengths_um)
    problems: dict[str, list[float]] = {}
    for name in names:
        spec = get_material_spec(name)
        outside = [value for value in wavelengths if not spec.supports_wavelength(value)]
        if outside:
            problems[spec.key] = outside
    return problems


def _normalise_wavelengths(
    wavelengths_um: float | Sequence[float],
) -> tuple[float, ...]:
    if isinstance(wavelengths_um, Real) and not isinstance(wavelengths_um, bool):
        values = (float(wavelengths_um),)
    elif isinstance(wavelengths_um, Sequence) and not isinstance(wavelengths_um, str):
        values = tuple(float(value) for value in wavelengths_um)
    else:
        raise MaterialError("Wavelength must be a number or a sequence of numbers.")
    if not values:
        raise MaterialError("At least one wavelength is required.")
    for value in values:
        _require_positive_number("wavelength_um", value)
    return values


def _require_positive_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or float(value) <= 0:
        raise MaterialError(f"{name} must be a positive number; received {value!r}.")


__all__ = [
    "MATERIALS",
    "MEEP_MATERIALS_SOURCE_URL",
    "MEEP_MATERIALS_URL",
    "MaterialCategory",
    "MaterialError",
    "MaterialModel",
    "MaterialSpec",
    "MaterialUnavailableError",
    "build_material",
    "get_material_spec",
    "list_backend_material_names",
    "list_materials",
    "make_constant_material",
    "material_catalog",
    "validate_material_band",
]
