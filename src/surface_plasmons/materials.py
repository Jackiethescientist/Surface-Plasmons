from meep.materials import (
    Au,
    Ag,
    Al,
    Cu,
    SiO2,
    Si3N4,
    cSi,
)

MATERIALS = {
    "gold": Au,
    "silver": Ag,
    "aluminium": Al,
    "copper": Cu,
    "silica": SiO2,
    "silicon_nitride": Si3N4,
    "silicon": cSi,
}


def get_material(name):
    try:
        return MATERIALS[name]
    except KeyError:
        raise ValueError(
            f"Unknown material '{name}'. "
            f"Available materials: {list(MATERIALS)}"
        )