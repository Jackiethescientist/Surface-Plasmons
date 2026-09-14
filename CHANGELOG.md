# Changelog

All notable changes to this project are documented here. The project follows Semantic Versioning.

## Unreleased

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
