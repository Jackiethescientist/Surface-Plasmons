"""Run one normalized gold-sphere simulation from the project root."""

from surface_plasmons import __version__, run_simulation


def main() -> None:
    result = run_simulation(
        {
            "geometry_name": "nanoparticle_sphere",
            "geometry_parameters": {
                "particle": "au",
                "background": "air",
                "radius": 0.040,
            },
            "dimensions": 2,
            "resolution": 80,
            "source": {
                "wavelength_range_um": [0.45, 0.80],
                "component": "Ez",
            },
            "flux": {
                "nfreq": 101,
                "normalization": "empty_cell",
            },
            "run_name": "gold_sphere",
            "code_version": __version__,
        }
    )

    print(f"Run ID: {result.run_id}")
    print(f"Status: {result.status}")
    print(f"Output: {result.output_directory}")


if __name__ == "__main__":
    main()
