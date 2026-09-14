from surface_plasmons.simulation import run_simulation

result = run_simulation(
    {
        "geometry_name": "nanoparticle_sphere",
        "geometry_parameters": {
            "particle": "au",
            "background": "air",
            "radius": 0.04,
        },
        "dimensions": 2,
        "resolution": 80,
    }
)

print(result.status)
print(result.output_directory)