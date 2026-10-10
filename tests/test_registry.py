import surface_plasmons as sp
from surface_plasmons.geometries import (
    get_geometry_spec,
    list_geometries,
    validate_geometry_parameters,
)
from surface_plasmons.materials import get_material_spec, list_materials


def test_release_version_and_public_exports() -> None:
    assert sp.__version__ == "0.3.1"
    assert callable(sp.run_simulation)
    assert callable(sp.run_sweep)
    assert callable(sp.analyze_spectrum)
    assert callable(sp.plot_spectrum)


def test_material_registry_is_available_without_importing_meep() -> None:
    materials = list_materials()

    assert len(materials) >= 65
    assert get_material_spec(materials[0].key) == materials[0]


def test_geometry_registry_and_parameter_validation() -> None:
    geometries = list_geometries()

    assert len(geometries) >= 26
    assert get_geometry_spec("nanoparticle_sphere").key == "nanoparticle_sphere"

    parameters = validate_geometry_parameters(
        "nanoparticle_sphere",
        {
            "particle": "au",
            "background": "air",
            "radius": 0.04,
        },
    )
    assert parameters["radius"] == 0.04
