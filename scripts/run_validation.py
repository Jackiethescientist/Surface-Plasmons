"""Validate one explicitly selected saved particle-line simulation.

Quick validation reads existing data only and evaluates the staged model hierarchy:

    python scripts/run_validation.py --simulation perfect_line_n9_spacing150nm

Numerical convergence runs are optional because they can take many hours:

    python scripts/run_validation.py --simulation perfect_line_n9_spacing150nm \
        --name final_validation --checks configuration,theory,hierarchy,resolution,domain,time
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from surface_plasmons.results import ResultsError
from surface_plasmons.validation import ValidationError, run_validation


def _integers(value: str | None) -> list[int] | None:
    return None if not value else [int(token.strip()) for token in value.split(",") if token.strip()]


def _floats(value: str | None) -> list[float] | None:
    return None if not value else [float(token.strip()) for token in value.split(",") if token.strip()]


def parse_args() -> Any:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", required=True, help="Simulation folder name or full folder path.")
    parser.add_argument("--name", default="validation_v1", help="Unique validation result name.")
    parser.add_argument(
        "--checks",
        default="configuration,theory,hierarchy",
        help="configuration,theory,hierarchy,resolution,domain,time or all.",
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--resolutions", help="Optional comma-separated resolution values.")
    parser.add_argument("--paddings", help="Optional comma-separated padding values in µm.")
    parser.add_argument("--no-reference-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checks = [value.strip() for value in args.checks.split(",") if value.strip()]
    try:
        destination = run_validation(
            args.simulation,
            validation_name=args.name,
            checks=checks,
            results_root=args.results_root,
            resolutions=_integers(args.resolutions),
            paddings_um=_floats(args.paddings),
            cache_enabled=not args.no_reference_cache,
        )
    except KeyboardInterrupt:
        raise SystemExit("Validation interrupted. Completed validation stages were preserved.")
    except (ValidationError, ResultsError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Validation failed: {exc}") from exc
    print(f"Validation results: {destination}")


if __name__ == "__main__":
    main()
