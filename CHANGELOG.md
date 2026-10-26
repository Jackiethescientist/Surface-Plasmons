# Changelog

All notable changes to this project are documented here. The project follows Semantic Versioning.

## Unreleased

- Add opt-in `--workers 2` support to particle-line runs, validation, notes
  sequences and generic sweeps; retain one worker by default.
- Run independent particle/reference solves in spawned processes while keeping
  manifest writes and adaptive convergence decisions in the parent.
- Lock shared particle-line references, isolate uncached reference outputs,
  and install timing/cache hooks inside generic sweep workers.
- Limit numerical-library threads in child processes and stop sibling workers
  when a solve fails or the parent is interrupted.

## 0.3.1 - 2026-10-09

### Added

- A homogeneous-background nine-sphere configuration for the first controlled Equation-2 comparison.
- Same-scale complex-field, amplitude, and intensity comparison outputs for coupled-dipole validation.
- An amplitude-enhancement main heatmap and explicit intensity diagnostic heatmap.

### Changed

- The coupled-dipole baseline now solves the full complex vector form
  `E_loc,i = E0 exp(i k.r_i) - sum_(j != i) A_ij.P_j`, including incident phase.
- Broadband measurements are sampled on an external line above the particles and use one complex reference scale per frequency.
- Every generated plot labels its coordinate axes and states whether its colour/value scale is field amplitude or intensity.
- The canonical `field_position_frequency_heatmap.png` now displays `|E|/|E_reference|`; the squared ratio is saved separately as `field_intensity_position_frequency_heatmap.png`.

### Fixed

- Removed independent maximum normalization from Equation-2 validation, which previously hid absolute scale disagreement.
- Made the sign convention explicit: `A_ij = -G_ij` for the quasistatic Green tensor used by the implementation.
- Prevented near-zero pointwise reference samples from creating artificial broadband enhancement spikes.

## 0.3.0 - 2026-09-26

### Added

- Finite 3-D sphere/cube lines on dielectric substrates with indexed positional defects.
- Broadband Ex/Ey/Ez line monitors and position-frequency enhancement tables.
- Optional selected-frequency full-volume field maps on xy, xz, and yz planes.
- Substrate-only reference caching, one-folder experiment storage, and runtime estimates.
- A separate selected-simulation validation runner with theory, resolution, domain, and time checks.
- Coupled-dipole equal-spacing and defect-aware comparison helpers.
- Perfect-line, one-defect, two-defect, and cube-line JSON examples.

### Changed

- The main `run_single.py` is now the focused particle-line project runner.
- The earlier general runner is retained as `run_general_single.py` for the optional legacy sweep script.
- Version increased to 0.3.0.

## 0.2.0 - 2026-09-14

### Added

- A registry of 65 metals, dielectrics, semiconductors, liquids, and custom material models.
- A registry of 25 planar, particle, dimer, core-shell, grating, array, cavity, and tip geometries.
- Typed simulation configuration, validation, MEEP construction, flux normalization, stopping rules, and saved artifacts.
- SQLite persistence for runs, metrics, spectra, artifacts, events, status transitions, and configuration lookup.
- Resumable grid and zip parameter sweeps with serial or process-based execution.
- Spectrum, field-enhancement, energy-balance, and analytical SPP analysis.
- Spectrum, field, epsilon, sweep, heatmap, SPP-dispersion, and run-summary visualizations.
- End-to-end single-run and sweep scripts.
- Pull-request CI across supported Python versions.

### Changed

- The public surface_plasmons package now exports the primary simulation, database, sweep, analysis, and visualization APIs.
- Runtime dependency metadata now reflects the numerical and plotting stack.
- The conda environment now includes PyMEEP plus development, notebook, build, and release tools.
- Documentation now covers installation, units, simulations, resumable sweeps, analysis, outputs, testing, and releases.

### Fixed

- Spectrum integration works across NumPy releases where np.trapz is unavailable.
- Tests no longer import the obsolete surface_plasmon_model package name.
- The previously empty MIT license file now contains the license text.

## 0.1.2 - 2026-09-13

- Initial packaged analytical surface-plasmon model.

