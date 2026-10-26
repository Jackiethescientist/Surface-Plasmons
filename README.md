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
- Broadband complex `Ex`, `Ey`, `Ez`, and field-amplitude enhancement versus position and frequency.
- Complex dielectric function versus frequency for particle, substrate and background.
- Optional narrowband full-volume maps on the `xy`, `xz` and `yz` planes.
- Substrate-only reference calculations with compatible-result caching.
- One named result folder per physical simulation.
- A separate validation runner for a user-selected simulation.
- Numerical resolution/domain/time checks and formula-only electrostatic comparisons.
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
│   └── particle_field_spectra.png
├── field_maps/
│   └── <wavelength>nm/
│       ├── field_Ex.npy
│       ├── field_Ey.npy
│       ├── field_Ez.npy
│       ├── field_amplitude_enhancement.npy
│       ├── field_total_enhancement.npy (amplitude alias)
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

The total field-amplitude ratio is calculated using all three complex
components:

```text
F_E(r,f) = sqrt(|Ex/Eref|² + |Ey/Eref|² + |Ez/Eref|²)
```

All component maps, total-field maps, spectra, summaries, peak selection and
validation use this unsquared amplitude ratio. The legacy
`total_field_enhancement` name is retained for compatibility but now also means
`F_E`. Full-volume maps use the matching particle-free field amplitude at each
point. The reference keeps the substrate and removes only the particles;
compatible references are reused across configurations.

## Validation

Validation now compares only the updated electrostatic notes: sphere
polarizability, signed axial/transverse dipole fields, and static coupled
particles. Follow [VALIDATION_SEQUENCE.md](VALIDATION_SEQUENCE.md) for the full
sequence, equations, tolerances and output definitions.

```bash
python scripts/run_notes_sequence.py --dry-run
python scripts/run_notes_sequence.py --stage single
python scripts/run_notes_sequence.py --stage convergence --convergence-scope single
python scripts/run_notes_sequence.py --stage pairs
python scripts/run_notes_sequence.py --stage convergence --convergence-scope pairs
python scripts/run_notes_sequence.py --stage summary
```

Identical reruns reuse results. Model comparison uses complex scattered fields
without independent rescaling. Only homogeneous sphere configurations qualify
for this theory check. The numerical resolution, domain and time checks each
perform additional calculations. Both pair eigenmodes are predicted; uniform
illumination tests only the symmetric mode.

`run_single.py` automatically compares the saved fields to these formulas unless
`--skip-validation` is supplied. That automatic comparison does not run the
extra numerical convergence cases. The previous `hierarchy` check is removed.
General theory APIs remain available for compatibility.

## Generic sweeps

The old generic sweep functionality is preserved for compatibility in
`scripts/run_sweep.py` and `surface_plasmons.sweep`, but it is not part of the
recommended defect-line workflow. Frequency sampling inside one broadband run
is still essential and is not the same as a parameter sweep.

## Two solver workers (WSL/Linux)

Add `--workers 2` to your usual command. The default remains one worker:

```bash
python scripts/run_single.py --config configs/perfect_line.json --name perfect_line_two_workers --workers 2
python scripts/run_validation.py --simulation perfect_line_two_workers --name convergence --checks resolution,domain,time --workers 2
python scripts/run_notes_sequence.py --stage single --workers 2
python scripts/run_sweep.py --workers 2
```

For the particle-line workflow (`run_single`, `run_validation`, and
`run_notes_sequence`), the two processes calculate the independent particle and
substrate-only reference fields concurrently. The parent combines the completed
fields and writes the experiment manifest. Narrowband maps follow the broadband
scan, and adaptive resolution checks retain their order and early stopping.
This does not change any physical parameters, tolerances, or validation scope.
Changing worker count does not invalidate saved configurations or validation
samples. A warm reference cache may leave only the particle calculation to run,
so two workers do not guarantee a speedup.

For the older generic sweep, workers run different broadband parameter points.
Each child installs its own runtime estimator and reference-cache hooks.
Selected resonance-field stages still run sequentially after the sweep.

All workers use `spawn`, with numerical-library thread counts limited to one
per child. Compatible references are protected by Linux/WSL file locks; disabling
reference reuse creates separate reference directories. An interrupted or failed
pool stops its other workers, and completed results remain available for resume.
Do not launch these worker commands under `mpirun`; this is independent-process
parallelism rather than MPI domain decomposition.

Both simultaneous solvers need to fit in WSL RAM. The printed base-field memory
estimate excludes some solver/DFT and analysis allocations; it is not a measured
peak or a memory guarantee. With 16 GB physical RAM and about 7.4 GiB accessible
to WSL, measure a demanding run before using two workers. Use `--workers 1` for
cases whose combined memory would exceed the available budget.

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
| `theory` | General coupled-dipole and scattering APIs |
| `notes_theory` | Only the electrostatic equations in the validation notes |
| `validation` | Selected-simulation theory and convergence checks |
| `materials` | Material registry and complex dielectric-function evaluation |
| `database` | Internal run-stage metadata and artifacts |

## License

MIT. See `LICENSE`.


