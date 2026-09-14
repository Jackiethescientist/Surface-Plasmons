# Surface Plasmon Model

Surface Plasmon Model is a Python toolkit for building, running, recording, analysing, sweeping, and visualising surface-plasmon simulations with MEEP.

Version 0.2.0 expands the project from basic analytical helpers into a complete simulation workflow:

- 65 material definitions with aliases, metadata, and wavelength validation
- 25 reusable plasmonic geometries
- MEEP simulation configuration and execution
- SQLite-backed run metadata, spectra, metrics, artifacts, and events
- resumable serial or process-based parameter sweeps
- spectral, field, energy-balance, and analytical SPP calculations
- publication-ready spectrum, field, sweep, heatmap, and summary plots

All lengths and wavelengths passed to the Python API use micrometres. For example, 40 nm is written as 0.040.

## Installation

### Full simulation environment

PyMEEP is distributed through conda-forge, so this is the recommended setup for simulations. On Windows, run these commands inside WSL2.

~~~bash
git clone https://github.com/Jackiethescientist/Surface__Plasmons.git
cd Surface__Plasmons

conda env create -f environment.yml
conda activate plasmon
python -m pip install -e ".[dev]"
~~~

Verify the package and MEEP:

~~~bash
python -c "import surface_plasmons as sp; print(sp.__version__)"
python scripts/test_meep.py
~~~

The MEEP check writes data/results/images/meep_test.png.

### Analysis-only installation

The numerical analysis, database, sweep construction, and plotting modules do not import MEEP at package-import time:

~~~bash
python -m pip install surface-plasmon-model
~~~

A conda environment containing PyMEEP is still required when run_simulation or run_sweep actually executes a simulation.

## Quick start

Run a two-dimensional gold-sphere simulation:

~~~python
from surface_plasmons import run_simulation

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
    }
)

print(result.run_id)
print(result.status)
print(result.output_directory)
~~~

The same example is available as:

~~~bash
python scripts/run_single.py
~~~

Run modules through the installed package or through the scripts directory. Do not execute files inside src/surface_plasmons directly; those files use package-relative imports.

## Materials and geometries

Inspect the available registries without starting MEEP:

~~~python
from surface_plasmons import list_geometries, list_materials

for material in list_materials():
    print(material.key, material.category, material.valid_wavelength_um)

for geometry in list_geometries():
    print(geometry.key, geometry.category, geometry.dimensions)
~~~

Use material_catalog() and geometry_catalog() when a JSON-ready mapping is more convenient.

## Parameter sweeps

A stable sweep_id allows an interrupted sweep to resume and lets completed configurations be reused from the database:

~~~python
from surface_plasmons import parameter_sweep

sweep = parameter_sweep(
    base_config={
        "geometry_name": "nanoparticle_sphere",
        "geometry_parameters": {
            "particle": "au",
            "background": "air",
        },
        "dimensions": 2,
        "resolution": 80,
        "source": {"wavelength_range_um": [0.45, 0.80]},
        "flux": {"nfreq": 101, "normalization": "empty_cell"},
    },
    parameters={
        "geometry_parameters.radius": [0.030, 0.040, 0.050],
    },
    name="gold_sphere_radius",
    sweep_id="gold_sphere_radius",
    workers=1,
    resume=True,
)

print(sweep.status)
print(sweep.manifest_path)
~~~

Use workers=1 first. MEEP simulations can use substantial memory; increase process workers only after measuring one run.

The complete sweep, analysis, and plotting workflow is:

~~~bash
python scripts/run_sweep.py
~~~

## Analysis and visualisation

Analyse synthetic or saved spectra:

~~~python
import numpy as np
from surface_plasmons import analyze_spectrum, plot_spectrum

wavelengths = np.linspace(0.45, 0.80, 401)
response = np.exp(-0.5 * ((wavelengths - 0.620) / 0.025) ** 2)

report = analyze_spectrum(
    wavelengths,
    response,
    monitor_name="extinction",
    prominence=0.05,
)

print(report.primary_peak)
plot_spectrum(
    wavelengths,
    response,
    label="Extinction",
    mark_peak=True,
    save_path="data/results/images/example_spectrum.png",
)
~~~

For saved simulations, use analyze_run(run_id), export_analysis(...), plot_run_spectra(...), or create_run_summary_figure(...). SweepResult objects and saved sweep.json manifests can be passed to plot_sweep_1d and plot_sweep_heatmap.

## Database

SimulationDatabase creates data/simulations.db automatically unless a different path is supplied:

~~~python
from surface_plasmons import SimulationDatabase

database = SimulationDatabase()
completed = database.list_runs(status="completed", limit=10)

for run in completed:
    print(run["run_id"], run["geometry_name"], run["output_directory"])
~~~

Set the SURFACE_PLASMONS_DB environment variable to override the default database path.

## Output layout

Generated data is intentionally ignored by Git:

~~~text
data/
├── simulations.db
└── results/
    ├── analysis/
    ├── images/
    ├── runs/
    └── sweeps/
~~~

Each simulation can save its resolved configuration, summary, spectra, epsilon array, field arrays, and figures. Sweep directories contain a continuously updated sweep.json manifest and runs.csv table.

## Package map

| Module | Purpose |
| --- | --- |
| materials | Material registry, aliases, model construction, wavelength checks |
| geometries | Geometry registry, validation, and MEEP object construction |
| simulation | Typed configuration, validation, execution, normalization, and output |
| database | SQLite run, metric, spectrum, artifact, and event persistence |
| sweep | Grid/zip expansion, resumable execution, and JSON/CSV manifests |
| analysis | Peak/FWHM/Q analysis, fields, energy balance, and SPP formulas |
| visualization | Spectra, fields, sweeps, heatmaps, dispersion, and run summaries |

## Development and release

Install development dependencies and run the test suite:

~~~bash
python -m pip install -e ".[dev]"
pytest
python -m build
python -m twine check dist/*
~~~

The standard tests do not require MEEP. scripts/test_meep.py is the separate environment smoke test.

Releases use semantic versioning. After a release pull request is merged, publish a GitHub release whose tag exactly matches the package version, such as v0.2.0. The release workflow verifies the tag, tests the project, builds both distributions, checks them with Twine, and publishes through PyPI trusted publishing.

See CHANGELOG.md for release notes.

## License

MIT License. See LICENSE.
