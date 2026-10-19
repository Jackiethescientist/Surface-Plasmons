"""Run the supplied-notes validation sequence; identical reruns reuse results."""
from __future__ import annotations
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
import json
import csv

import numpy as np

from surface_plasmons.database import find_project_root
from surface_plasmons.project import PRECISION_MODES, load_project_config, normalise_project_config
from surface_plasmons.results import config_hash, safe_simulation_name
from surface_plasmons.validation import run_validation, ValidationError
from surface_plasmons.workflow import run_project


def cases(base, prefix):
    base = deepcopy(base)
    profile = {**PRECISION_MODES[base["precision"]], **base["numerics"]}
    for key in ("broadband_resolution", "field_resolution", "nfreq"):
        base["numerics"][key] = int(profile[key])
    result = []
    for orientation, pol in (("longitudinal", "Ex"), ("transverse", "Ez")):
        config = deepcopy(base)
        config["simulation_name"] = f"{prefix}_single_{orientation}"
        config["source"]["polarisation"] = pol
        result.append(("single", orientation, None, normalise_project_config(config)))
    for gap_nm in (40.0, 60.0, 80.0):
        for orientation, pol in (("longitudinal", "Ex"), ("transverse", "Ez")):
            config = deepcopy(base)
            config["simulation_name"] = f"{prefix}_pair_gap{gap_nm:g}nm_{orientation}"
            config["geometry"].update(particle_count=2, spacing_um=2*base["geometry"]["particle_radius_um"]+gap_nm/1000)
            config["source"]["polarisation"] = pol
            config["numerics"].update(broadband_resolution=max(120, int(base["numerics"]["broadband_resolution"])),
                                      field_resolution=max(120, int(base["numerics"]["field_resolution"])),
                                      nfreq=max(81, int(base["numerics"]["nfreq"])))
            result.append(("pair", orientation, gap_nm, normalise_project_config(config)))
    return result


def ensure_simulation(config, root, cache_enabled):
    target = root / safe_simulation_name(config["simulation_name"])
    if target.exists():
        manifest = target / "manifest.json"
        if not manifest.is_file():
            raise ValidationError(f"Existing result has no manifest: {target}")
        previous = json.loads(manifest.read_text())
        if previous.get("configuration_hash") != config_hash(config):
            raise ValidationError(f"Existing sequence result uses different settings: {target}. Choose a new --name.")
        if previous.get("status") == "completed":
            print(f"Reusing completed simulation: {target.name}")
            return target
    run_project(config, results_root=root, resume=target.exists(), cache_enabled=cache_enabled, field_map_selection="none")
    return target


def write_summary(root, entries, prefix, comparison_name="notes_comparison", summary_name="sequence_summary"):
    baseline = {}
    reports, lines = [], ["# Electrostatic validation sequence", ""]
    for kind, orientation, gap, config in entries:
        directory = root / safe_simulation_name(config["simulation_name"])
        path = directory / "validation" / comparison_name / "notes_validation.json"
        report = json.loads(path.read_text()) if path.is_file() else {}
        numeric = directory / "validation/notes_convergence/validation_summary.json"
        convergence = json.loads(numeric.read_text()) if numeric.is_file() else {}
        theory_peak = report.get("theory_peak", {}).get("wavelength_nm")
        meep_peak = report.get("meep_peak", {}).get("wavelength_nm")
        resolved = report.get("meep_peak", {}).get("status") == report.get("theory_peak", {}).get("status") == "resolved_sampled_peak"
        if kind == "single" and resolved:
            baseline[orientation] = (meep_peak, theory_peak)
        reference = baseline.get(orientation)
        row = {"simulation": directory.name, "kind": kind, "orientation": orientation,
               "case_status": "completed_comparison" if report else "not_run",
               "failure_reasons": report.get("failure_reasons", []),
               "agreement_checks": report.get("agreement_checks", {}),
               "applicability_checks": report.get("applicability_checks", {}),
               "applicability_failure_reasons": report.get("applicability_failure_reasons", []),
               "relative_peak_wavelength_error": report.get("relative_peak_wavelength_error"),
               "complex_field_relative_l2_error": report.get("complex_scattered_field_relative_l2_error"),
               "surface_gap_nm": gap, "model_agreement": report.get("agreement_status", report.get("status", "not_run")),
               "applicability": report.get("applicability_status", report.get("reason", "not_run")),
               "meep_peak_nm": meep_peak, "theory_peak_nm": theory_peak,
               "meep_shift_from_single_nm": meep_peak-reference[0] if kind == "pair" and reference and resolved else None,
               "theory_shift_from_single_nm": theory_peak-reference[1] if kind == "pair" and reference and resolved else None,
               "numerical_checks": {key: convergence.get(key, {}).get("convergence_status", "not_run") for key in ("resolution", "domain", "time")}}
        row["validated"] = (row["model_agreement"] == "passed" and row["applicability"] == "passed"
                            and all(s == "passed" for s in row["numerical_checks"].values()))
        reports.append(row)
        lines += [f"## {directory.name}"]
        if not report:
            lines += ["- Status: NOT RUN — no theoretical comparison is available for this case.",
                      "- Quantitative validation: not assessed.", ""]
            continue
        lines += [f"- Model agreement: {row['model_agreement']}",
                  f"- Applicability: {row['applicability']}",
                  f"- Quantitatively validated: {row['validated']}",
                  f"- Peak wavelength, MEEP / notes: {meep_peak} / {theory_peak} nm"]
        for label, value in (("Peak wavelength error", row["relative_peak_wavelength_error"]),
                             ("Complex scattered-field relative L2 error", row["complex_field_relative_l2_error"])):
            lines.append(f"- {label}: {100*value:.3f}%" if value is not None else f"- {label}: not available in this report")
        for key, check in row["applicability_checks"].items():
            lines.append(f"- {key}: {check['status']} (value {check['value']:.6g}; limit {check['limit']:.6g})")
        for key, status in row["numerical_checks"].items():
            lines.append(f"- Numerical {key} convergence: {status}")
        if row["failure_reasons"]:
            lines.append("- Failed model checks: " + ", ".join(row["failure_reasons"]))
        if row["applicability_failure_reasons"]:
            lines.append("- Failed applicability checks: " + ", ".join(row["applicability_failure_reasons"]))
        lines.append("")
    singles = [r for r in reports if r["kind"] == "single"]
    polarization = {"status": "unavailable"}
    if len(singles) == 2 and all(r["meep_peak_nm"] for r in singles):
        polarization = {"status": "compared", "absolute_peak_difference_nm": abs(singles[0]["meep_peak_nm"]-singles[1]["meep_peak_nm"]),
                        "note": "An isolated isotropic sphere has polarization-independent polarizability. Ex and Ez sample different axial/transverse field factors on the same x-line."}
    if len(singles) == 2:
        paths = [root/r["simulation"] / "validation" / comparison_name / "notes_fields.csv" for r in singles]
        if all(p.is_file() for p in paths):
            tables = []
            for path in paths:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    tables.append({tuple(round(float(row[k]), 10) for k in ("wavelength_nm", "x_um", "y_um", "z_um")): row for row in csv.DictReader(handle)})
            common = sorted(set(tables[0]) & set(tables[1]))
            if common:
                axial = np.asarray([complex(float(tables[0][k]["meep_Ex_real"]), float(tables[0][k]["meep_Ex_imag"])) for k in common])
                transverse = np.asarray([complex(float(tables[1][k]["meep_Ez_real"]), float(tables[1][k]["meep_Ez_imag"])) for k in common])
                error = float(np.linalg.norm(axial+2*transverse)/max(np.linalg.norm(axial), 1e-30))
                polarization.update(axial_equals_minus_twice_transverse_relative_l2_error=error,
                                    field_factor_agreement="passed" if error <= .05 else "failed", common_sample_count=len(common))
    folder = root / f"{safe_simulation_name(prefix)}_sequence"
    folder.mkdir(parents=True, exist_ok=True)
    summary_stem = safe_simulation_name(summary_name)
    (folder / f"{summary_stem}.json").write_text(json.dumps({"cases": reports, "single_polarization": polarization}, indent=2)+"\n")
    lines += ["Both symmetric and antisymmetric pair modes are exported as theory predictions. The plane-wave cases test the symmetric mode only.",
              "Dipole amplitude/phase estimates come from inverse fitting of exterior fields and are conditional on the electrostatic approximation."]
    (folder / f"{summary_stem}.md").write_text("\n".join(lines)+"\n")
    print(f"Sequence summary: {folder}")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/notes_single_sphere.json")
    parser.add_argument("--name", default="electrostatic_notes_v2", help="Prefix for all cases; reuse for an identical restart.")
    parser.add_argument("--stage", choices=("single", "pairs", "convergence", "summary", "all"), default="single")
    parser.add_argument("--convergence-scope", choices=("single", "pairs", "all"), default="single")
    parser.add_argument("--comparison-name", default="notes_comparison", help="Named theory-comparison folder to write or summarize.")
    parser.add_argument("--summary-name", default="sequence_summary", help="Output stem for the sequence Markdown and JSON summaries.")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-reference-cache", action="store_true")
    args = parser.parse_args()
    base = load_project_config(args.config)
    if base["geometry"]["particle_shape"] != "sphere" or base["geometry"]["particle_count"] != 1 or base["geometry"]["defects"]:
        parser.error("The sequence base must be one unshifted sphere.")
    entries = cases(base, args.name)
    root = args.results_root.resolve() if args.results_root else find_project_root()/"data/results"
    if args.dry_run:
        for kind, orientation, gap, config in entries:
            resolution = int(config["numerics"]["broadband_resolution"])
            ladder = [resolution, resolution+40, resolution+80]
            print(f"{kind}: {config['simulation_name']}; {orientation}; gap={gap} nm; resolution={resolution}; nfreq={config['numerics']['nfreq']}; adaptive ladder={ladder}")
        print("Stages: single -> convergence (single) -> pairs -> convergence (pairs) -> summary. No simulation started.")
        return
    # Complete the isolated-sphere numerical checks before progressing to pairs.
    for group in ("single", "pair"):
        if args.stage == "all" or args.stage.replace("pairs", "pair") == group:
            for kind, _, _, config in entries:
                if kind != group:
                    continue
                directory = ensure_simulation(config, root, not args.no_reference_cache)
                run_validation(directory, validation_name=args.comparison_name, checks=("configuration", "theory"))
        scope = "all" if args.stage == "all" else args.convergence_scope.replace("pairs", "pair")
        if args.stage in ("convergence", "all") and scope in ("all", group):
            for kind, _, _, config in entries:
                if kind != group:
                    continue
                directory = root / safe_simulation_name(config["simulation_name"])
                if not (directory / "config_resolved.json").is_file():
                    raise ValidationError(f"Run the {group} stage first. Missing {directory.name}.")
                run_validation(directory, validation_name="notes_convergence", checks=("resolution", "domain", "time"), cache_enabled=not args.no_reference_cache)
    write_summary(root, entries, args.name, args.comparison_name, args.summary_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Sequence interrupted. Repeat the same command to reuse completed results.")
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Sequence failed: {exc}") from exc
