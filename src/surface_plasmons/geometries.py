import math
import meep as mp


def create_geometry(shape, material, size=0.1):
    """
    size is specified in micrometres.

    Example:
        0.1 = 100 nm
        0.05 = 50 nm
    """

    if shape == "circle":
        return [
            mp.Cylinder(
                radius=size / 2,
                material=material
            )
        ]

    elif shape == "square":
        return [
            mp.Block(
                size=mp.Vector3(size, size, mp.inf),
                material=material
            )
        ]

    elif shape == "rectangle":
        return [
            mp.Block(
                size=mp.Vector3(size * 1.5, size / 1.5, mp.inf),
                material=material
            )
        ]

    elif shape == "ellipse":
        return [
            mp.Ellipsoid(
                size=mp.Vector3(size * 1.5, size / 1.5, mp.inf),
                material=material
            )
        ]

    elif shape == "triangle":
        r = size / math.sqrt(3)

        vertices = [
            mp.Vector3(
                r * math.cos(a),
                r * math.sin(a)
            )
            for a in [
                math.pi / 2,
                math.pi / 2 + 2 * math.pi / 3,
                math.pi / 2 + 4 * math.pi / 3
            ]
        ]

        return [
            mp.Prism(
                vertices=vertices,
                height=mp.inf,
                material=material
            )
        ]

    elif shape == "hexagon":
        r = size / 2

        vertices = [
            mp.Vector3(
                r * math.cos(i * math.pi / 3),
                r * math.sin(i * math.pi / 3)
            )
            for i in range(6)
        ]

        return [
            mp.Prism(
                vertices=vertices,
                height=mp.inf,
                material=material
            )
        ]

    else:
        raise ValueError(f"Unknown shape: {shape}")