# Surface Plasmons — Finite Particle-Line Project

This version is focused on one research question:

> How do positional defects in a finite line of plasmonic particles change the
> frequency-dependent electric-field distribution, and when does an
> equal-spacing coupled-particle model stop describing the system accurately?

The simulations are genuinely three-dimensional. A finite line lies along
`x`, the substrate surface normal is `y`, and `z` is the transverse in-plane
direction. A normal-incidence source normally propagates along `-y`; `Ex`
polarisation is parallel to the line and `Ez` is perpendicular to the line.

All lengths supplied to the Python package use micrometres. JSON example files
use nanometres only where the field name explicitly ends in `_nm`.

## What is included

- A finite line of spheres or cubes on a dielectric substrate.
- Explicit one- or two-particle position defects.
- Broadband complex `Ex`, `Ey`, `Ez`, field-amplitude enhancement and field-intensity enhancement versus position and frequency.
- Complex dielectric function versus frequency for particle, substrate and background.
- Optional narrowband full-volume maps on the `xy`, `xz` and `yz` planes.
- Substrate-only reference calculations with compatible-result caching.
- One named result folder per physical simulation.
- A separate validation runner for a user-selected simulation.
- Numerical resolution/domain checks and a same-scale Equation-2 coupled-dipole comparison.
- `fast`, `study` and `final` precision modes (`sweep` remains a legacy alias for `study`).
- Runtime and memory estimates before each expensive stage.

## Installation

PyMEEP is supplied through conda-forge. On Windows, use these commands inside WSL2:

```bash
conda env create -f environment.yml
conda activate plasmon
python -m pip install -e ".[dev]"
python scripts/test_meep.py
```

The ordinary unit tests do not start MEEP:

```bash
pytest
```

## First safe test

Validate the configuration and print a runtime estimate without starting MEEP:

```bash
python scripts/run_single.py \
  --config configs/perfect_line.json \
  --precision fast \
  --field-maps none \
  --dry-run
```

Then run the broadband simulation without the expensive 3-D narrowband map:

```bash
python scripts/run_single.py \
  --config configs/perfect_line.json \
  --precision fast \
  --field-maps none
```

Once that completes correctly, request one automatically selected field map:

```bash
python scripts/run_single.py \
  --config configs/perfect_line.json \
  --name perfect_line_n9_spacing150nm_with_map \
  --precision fast \
  --field-maps auto
```

Use a new name because completed results are never silently overwritten.

## Example configurations

| File | Purpose |
| --- | --- |
| `configs/equation2_perfect_line_homogeneous.json` | Controlled homogeneous-background benchmark for Equation 2 |
| `configs/perfect_line.json` | Equal-spacing nine-sphere validation case |
| `configs/one_shifted_particle.json` | Central sphere shifted 20 nm along the line |
| `configs/two_shifted_particles.json` | Two symmetrically shifted spheres |
| `configs/cube_line.json` | Equal-spacing line of cubes |

The important defect format is:

```json
"defects": [
  {
    "particle_index": 4,
    "shift_um": [0.020, 0.0, 0.0]
  }
]
```

Particle indices start at zero. `shift_um` gives `[x, y, z]` displacement.
The configuration is rejected before MEEP starts if particles overlap or enter
the substrate.

## Broadband and narrowband stages

The broadband stage is the main research measurement. It stores an external
line monitor above the particle surfaces at many frequencies. The default
offset is 20 nm above the highest particle surface, so the samples are fields
in the surrounding medium rather than fields inside gold. This makes the
position-frequency maps both physically comparable to a point-dipole model and
manageable in memory and storage.

The narrowband stage is optional. It reruns only selected wavelengths and stores
full three-dimensional vector fields. Available choices are:

```bash
--field-maps none
--field-maps auto
--field-maps 620,670
```

Saving full 3-D fields at every broadband frequency is intentionally avoided.

## Result folders

A simulation named `one_defect_n9_particle4_dx20nm` produces one top-level folder:

```text
data/results/one_defect_n9_particle4_dx20nm/
├── config_input.json
├── config_resolved.json
├── manifest.json
├── status.json
├── run.log
├── geometry/
│   ├── particle_positions.csv
│   └── geometry_xy.png
├── frequency_scan/
│   ├── dielectric_function_vs_frequency.csv
│   ├── field_strength_vs_position_and_frequency.csv
│   ├── field_response_summary.csv
│   ├── selected_frequencies.csv
│   ├── field_position_frequency_heatmap.png (amplitude, |E|/|E_reference|)
│   ├── field_intensity_position_frequency_heatmap.png (squared intensity ratio)
│   └── particle_field_spectra.png
├── field_maps/
│   └── <wavelength>nm/
│       ├── field_Ex.npy
│       ├── field_Ey.npy
│       ├── field_Ez.npy
│       ├── field_amplitude_enhancement.npy
│       ├── field_intensity_enhancement.npy
│       ├── field_total_enhancement.npy (legacy intensity alias)
│       ├── coordinates.npz
│       ├── field_components_xy.png
│       ├── field_components_xz.png
│       ├── field_components_yz.png
│       ├── field_amplitude_xy.png
│       ├── field_amplitude_xz.png
│       ├── field_amplitude_yz.png
│       └── field_map_manifest.json
├── summary/
│   ├── research_results.csv
│   └── field_map_reports.json
├── stages/
│   └── labelled raw MEEP stages
└── validation/
    └── separately named validation attempts
```

CSV files use UTF-8 with a byte-order mark so they import cleanly into Excel.
The shared `data/reference_cache_v7` and `data/simulations.db` are internal
indexes/caches, not research-result folders.

## Electric-field scales

The broadband complex vector is divided by one complex reference amplitude for
each frequency. That reference is the robust spatial value of the driven
component (`Ex` by default) from the particle-free run. A single scale preserves
the spatial profile and avoids divisions by isolated near-zero reference pixels.

The total field amplitude and intensity are calculated using all three complex
components:

```text
F_E(r,f) = sqrt(|Ex/Eref|² + |Ey/Eref|² + |Ez/Eref|²)
F_I(r,f) = F_E(r,f)²
```

Every plot title and colour bar states which scale it displays. The legacy
`total_field_enhancement` output is retained for compatibility and means
intensity enhancement, `F_I`. Full-volume maps retain pointwise particle-free
intensity normalization because they compare matching spatial grids. The
reference keeps the substrate and removes only the particles; compatible
references are reused across configurations.

## Validation

Validation never runs automatically for every result. Select one simulation:

```bash
python scripts/run_validation.py \
  --simulation perfect_line_n9_spacing150nm \
  --name quick_validation_v1
```

The default `configuration,theory` checks read existing data and are fast. Runs
created before version 0.3.1 must be rerun because their CSV files do not contain
the normalized complex vector field required for a same-scale comparison.
Resolution and domain convergence start additional MEEP calculations:

```bash
python scripts/run_validation.py \
  --simulation perfect_line_n9_spacing150nm \
  --name final_validation_v1 \
  --checks configuration,theory,resolution,domain,time
```

Custom numerical values can be supplied:

```bash
--resolutions 60,80,100
--paddings 0.30,0.40
```

Validation results remain inside the selected simulation's `validation` folder.
Use a new validation name for every attempt.

## Theory scope

`surface_plasmons.theory` implements the paper's complex vector equation

```text
E_loc,i = E0 exp(i k.r_i) - sum_(j != i) A_ij.P_j,
P_i = alpha_i E_loc,i.
```

For the quasistatic Green tensor `G`, the code defines `A_ij = -G_ij`; therefore
the paper's `-A_ij.P_j` is the physical dipole field `+G_ij.P_j`. The solver
includes the incident phase and all Cartesian dipole components. Validation
evaluates the solved dipoles at the exact external MEEP sample points, compares
the complex fields on one common scale, and reports the Equation-2 residual.

Start with the optically homogeneous
`configs/equation2_perfect_line_homogeneous.json` case. For configurations with
a substrate, validation adds a simple electrostatic image-dipole correction and
labels it as an approximation rather than claiming that it is the exact
homogeneous-medium Equation 2.

This baseline is deliberately limited: it does not contain the full retarded
substrate dyadic Green tensor or higher multipoles. Those limitations are part
of the research question rather than being hidden.

## Generic sweeps

The old generic sweep functionality is preserved for compatibility in
`scripts/run_sweep.py` and `surface_plasmons.sweep`, but it is not part of the
recommended defect-line workflow. Frequency sampling inside one broadband run
is still essential and is not the same as a parameter sweep.

## Interruption and naming

- A new run refuses to use an existing result name.
- Add a suffix such as `_v2` for a new configuration.
- `--resume` is accepted only when the configuration hash matches an existing
  interrupted folder.
- `Ctrl+C` records an `interrupted` status instead of pretending the run completed.
- Internal reference-cache files may be recalculated; research result folders
  are never silently replaced.

## Package map

| Module | Purpose |
| --- | --- |
| `particle_line` | Validated ideal and defect particle positions |
| `geometries` | MEEP geometry registry including the finite line and substrate |
| `project` | JSON schema, precision modes, monitor regions and estimates |
| `simulation` | MEEP construction, execution and multi-frequency vector fields |
| `workflow` | Broadband/reference/narrowband orchestration and labelled outputs |
| `results` | Safe one-folder storage, manifests and status |
| `theory` | Coupled-dipole baseline and profile comparison |
| `validation` | Selected-simulation theory and convergence checks |
| `materials` | Material registry and complex dielectric-function evaluation |
| `database` | Internal run-stage metadata and artifacts |

## License

MIT. See `LICENSE`.
