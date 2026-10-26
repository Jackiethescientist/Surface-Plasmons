"""Compare a saved simulation only with the supplied electrostatic formulas.

Identical reruns reuse the same validation folder and completed numerical cases.
"""
from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
from surface_plasmons.validation import ValidationError, run_validation
from surface_plasmons.results import ResultsError


def _numbers(value, cast=float):
    return None if value is None else [cast(v.strip()) for v in value.split(",") if v.strip()]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", required=True)
    parser.add_argument("--name", default="notes_validation_v1")
    parser.add_argument("--checks", default="configuration,theory", help="configuration,theory,resolution,domain,time or all")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--resolutions")
    parser.add_argument("--paddings")
    parser.add_argument("--time-factors", help="Comma-separated duration factors; default 1,2")
    parser.add_argument("--field-tolerance", type=float, default=0.05)
    parser.add_argument("--peak-tolerance", type=float, default=0.02)
    parser.add_argument("--no-reference-cache", action="store_true")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1,
                        help="Concurrent reference/particle solves within each convergence case.")
    parser.add_argument("--full-resolution-ladder", action="store_true", help="Run every requested resolution instead of stopping at the first converged pair.")
    parser.add_argument("--mirror-symmetry", action="store_true",
                        help="Use x/z mirrors for supported single Au/air or constant-glass/vacuum benchmarks; reuse the saved baseline.")
    args = parser.parse_args()
    try:
        output = run_validation(args.simulation, validation_name=args.name,
            checks=args.checks.split(","), results_root=args.results_root,
            resolutions=_numbers(args.resolutions, int), paddings_um=_numbers(args.paddings),
            time_factors=_numbers(args.time_factors), tolerance=args.field_tolerance,
            peak_tolerance=args.peak_tolerance, cache_enabled=not args.no_reference_cache,
            adaptive_resolution=not args.full_resolution_ladder,
            mirror_symmetry=args.mirror_symmetry, workers=args.workers)
    except KeyboardInterrupt:
        raise SystemExit("Validation interrupted. Completed numerical samples were preserved.")
    except (ValidationError, ResultsError, RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc
    print(f"Validation results: {output}")


if __name__ == "__main__":
    main()

