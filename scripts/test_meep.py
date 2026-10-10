"""Minimal headless MEEP installation check."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import meep as mp


def main() -> None:
    print(f"MEEP version: {mp.__version__}")

    simulation = mp.Simulation(
        cell_size=mp.Vector3(2, 2, 0),
        geometry=[
            mp.Cylinder(
                radius=0.25,
                material=mp.Medium(epsilon=4),
            )
        ],
        resolution=50,
    )

    output = Path("data/results/images/meep_test.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    axis = simulation.plot2D()
    axis.set_xlabel("x position (µm)")
    axis.set_ylabel("y position (µm)")
    axis.set_title("MEEP installation test geometry")
    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
