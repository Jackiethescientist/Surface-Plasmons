import meep as mp
import matplotlib.pyplot as plt

from .materials import get_material
from .geometries import create_geometry


def run_simulation(
    material_name,
    shape,
    size,
    wavelength,
    resolution=100,
    save_path=None
):

    material = get_material(material_name)

    geometry = create_geometry(
        shape=shape,
        material=material,
        size=size
    )

    padding = 0.5
    pml = 0.3

    cell_width = size + 2 * padding + 2 * pml

    cell = mp.Vector3(
        cell_width,
        cell_width,
        0
    )

    frequency = 1 / wavelength

    sources = [
        mp.Source(
            src=mp.ContinuousSource(
                frequency=frequency
            ),

            # In-plane electric field / Hz mode
            # is useful for plasmonic behaviour
            component=mp.Hz,

            center=mp.Vector3(
                -cell_width / 2 + pml + 0.1,
                0
            ),

            size=mp.Vector3(
                0,
                cell_width - 2 * pml
            )
        )
    ]

    sim = mp.Simulation(
        cell_size=cell,
        boundary_layers=[
            mp.PML(pml)
        ],
        geometry=geometry,
        sources=sources,
        resolution=resolution
    )

    sim.run(until=100)

    fig, ax = plt.subplots(figsize=(7, 7))

    sim.plot2D(
        ax=ax,
        fields=mp.Hz
    )

    ax.set_title(
        f"{material_name} | {shape} | "
        f"{size * 1000:.0f} nm | "
        f"{wavelength * 1000:.0f} nm"
    )

    if save_path:
        plt.savefig(
            save_path,
            dpi=250,
            bbox_inches="tight"
        )

    return sim, fig