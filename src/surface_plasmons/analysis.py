"""Numerical analysis tools for surface-plasmon simulation results.

This module deliberately does not import MEEP.  It can therefore be used on a
normal Python installation to analyse saved spectra and NumPy field arrays.
All wavelengths and distances use micrometres unless a function says otherwise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import json
import math

import numpy as np

from .database import SimulationDatabase


class AnalysisError(ValueError):
    """Raised when result data cannot be analysed safely."""


@dataclass(frozen=True, slots=True)
class SpectralPeak:
    """One maximum or minimum detected in a spectrum."""

    index: int
    wavelength_um: float
    frequency: float
    value: float
    prominence: float
    width_um: float | None
    quality_factor: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpectrumAnalysis:
    """Summary statistics and resonances for one spectral monitor."""

    monitor_name: str
    peak_mode: str
    sample_count: int
    wavelength_min_um: float
    wavelength_max_um: float
    value_min: float
    value_max: float
    value_mean: float
    integrated_area: float
    centroid_wavelength_um: float | None
    primary_peak: SpectralPeak | None
    peaks: tuple[SpectralPeak, ...]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class FieldAnalysis:
    """Amplitude, intensity, and enhancement statistics for a field array."""

    component: str
    shape: tuple[int, ...]
    finite_samples: int
    amplitude_min: float
    amplitude_max: float
    amplitude_mean: float
    intensity_max: float
    intensity_mean: float
    intensity_percentile_95: float
    maximum_enhancement: float
    mean_enhancement: float

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class EnergyBalance:
    """Checks the identity R + T + A = 1 for normalized spectra."""

    sample_count: int
    residual_mean: float
    residual_rms: float
    residual_abs_max: float
    within_tolerance_fraction: float
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_spectrum(
    wavelengths_um: Sequence[float],
    values: Sequence[complex | float],
    *,
    component: str = "real",
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite, wavelength-sorted data with duplicate wavelengths averaged.

    ``component`` may be ``real``, ``imag``, ``magnitude``, ``phase``, or
    ``intensity``.  The returned arrays are always one-dimensional floats.
    """

    wavelengths = np.asarray(wavelengths_um, dtype=float).reshape(-1)
    raw = np.asarray(values).reshape(-1)
    if wavelengths.size != raw.size:
        raise AnalysisError("wavelengths and values must have the same length.")
    if wavelengths.size == 0:
        raise AnalysisError("A spectrum must contain at least one sample.")
    if np.any(wavelengths <= 0):
        raise AnalysisError("Wavelengths must be positive.")

    component_key = component.lower().replace("abs", "magnitude")
    if component_key == "real":
        signal = np.real(raw)
    elif component_key == "imag":
        signal = np.imag(raw)
    elif component_key == "magnitude":
        signal = np.abs(raw)
    elif component_key == "phase":
        signal = np.angle(raw)
    elif component_key == "intensity":
        signal = np.abs(raw) ** 2
    else:
        raise AnalysisError(
            "component must be real, imag, magnitude, phase, or intensity."
        )
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(wavelengths) & np.isfinite(signal)
    wavelengths = wavelengths[finite]
    signal = signal[finite]
    if wavelengths.size == 0:
        raise AnalysisError("The spectrum contains no finite samples.")

    order = np.argsort(wavelengths, kind="stable")
    wavelengths = wavelengths[order]
    signal = signal[order]
    unique, inverse = np.unique(wavelengths, return_inverse=True)
    if unique.size != wavelengths.size:
        sums = np.zeros(unique.size, dtype=float)
        counts = np.zeros(unique.size, dtype=float)
        np.add.at(sums, inverse, signal)
        np.add.at(counts, inverse, 1.0)
        wavelengths = unique
        signal = sums / counts
    return wavelengths, signal


def find_spectral_peaks(
    wavelengths_um: Sequence[float],
    values: Sequence[complex | float],
    *,
    mode: str = "max",
    component: str = "real",
    prominence: float | None = None,
    minimum_distance_um: float | None = None,
    max_peaks: int | None = None,
) -> tuple[SpectralPeak, ...]:
    """Find and rank spectral resonances.

    SciPy is used when installed.  A deterministic local-extrema fallback is
    used otherwise.  Peaks are ranked by prominence, not wavelength order.
    """

    wavelengths, signal = prepare_spectrum(
        wavelengths_um, values, component=component
    )
    mode_key = mode.lower()
    if mode_key not in {"max", "min", "absolute"}:
        raise AnalysisError("mode must be max, min, or absolute.")
    if prominence is not None and prominence < 0:
        raise AnalysisError("prominence cannot be negative.")
    if minimum_distance_um is not None and minimum_distance_um <= 0:
        raise AnalysisError("minimum_distance_um must be positive.")
    if max_peaks is not None and max_peaks < 1:
        raise AnalysisError("max_peaks must be positive or None.")

    detection = _detection_signal(signal, mode_key)
    peak_indices, prominences = _peak_indices(
        wavelengths,
        detection,
        prominence=prominence,
        minimum_distance_um=minimum_distance_um,
    )
    ranked = sorted(
        zip(peak_indices, prominences),
        key=lambda item: (float(item[1]), float(detection[item[0]])),
        reverse=True,
    )
    if max_peaks is not None:
        ranked = ranked[:max_peaks]
    peaks = []
    for index, peak_prominence in ranked:
        width = _half_height_width(wavelengths, detection, int(index))
        wavelength = float(wavelengths[index])
        peaks.append(
            SpectralPeak(
                index=int(index),
                wavelength_um=wavelength,
                frequency=1.0 / wavelength,
                value=float(signal[index]),
                prominence=float(peak_prominence),
                width_um=width,
                quality_factor=(wavelength / width if width and width > 0 else None),
            )
        )
    return tuple(peaks)


def analyze_spectrum(
    wavelengths_um: Sequence[float],
    values: Sequence[complex | float],
    *,
    monitor_name: str = "spectrum",
    peak_mode: str = "max",
    component: str = "real",
    prominence: float | None = None,
    minimum_distance_um: float | None = None,
    max_peaks: int = 10,
) -> SpectrumAnalysis:
    """Calculate descriptive statistics, resonances, FWHM, and Q factors."""

    wavelengths, signal = prepare_spectrum(
        wavelengths_um, values, component=component
    )
    peaks = find_spectral_peaks(
        wavelengths,
        signal,
        mode=peak_mode,
        prominence=prominence,
        minimum_distance_um=minimum_distance_um,
        max_peaks=max_peaks,
    )
    # Calculate the trapezoidal integral directly.  This avoids relying on
    # ``np.trapz`` (removed in newer NumPy) or ``np.trapezoid`` (unavailable in
    # some older NumPy releases).
    area = float(
        np.sum(0.5 * (signal[:-1] + signal[1:]) * np.diff(wavelengths))
    )
    weights = _detection_signal(signal, peak_mode.lower())
    weights = weights - np.min(weights)
    weight_sum = float(np.sum(weights))
    centroid = (
        float(np.sum(wavelengths * weights) / weight_sum)
        if weight_sum > 0
        else None
    )
    return SpectrumAnalysis(
        monitor_name=str(monitor_name),
        peak_mode=peak_mode.lower(),
        sample_count=int(wavelengths.size),
        wavelength_min_um=float(wavelengths[0]),
        wavelength_max_um=float(wavelengths[-1]),
        value_min=float(np.min(signal)),
        value_max=float(np.max(signal)),
        value_mean=float(np.mean(signal)),
        integrated_area=area,
        centroid_wavelength_um=centroid,
        primary_peak=peaks[0] if peaks else None,
        peaks=peaks,
    )


def analyze_field(
    field: np.ndarray | Sequence[Any] | str | Path,
    *,
    component: str = "field",
    incident_amplitude: float = 1.0,
) -> FieldAnalysis:
    """Analyse a complex field array and calculate |E|/|E0| enhancement."""

    if not isinstance(incident_amplitude, (int, float)) or incident_amplitude <= 0:
        raise AnalysisError("incident_amplitude must be positive.")
    array = _load_array(field)
    amplitude = np.abs(array)
    finite = amplitude[np.isfinite(amplitude)]
    if finite.size == 0:
        raise AnalysisError("The field contains no finite values.")
    intensity = finite**2
    return FieldAnalysis(
        component=component,
        shape=tuple(int(value) for value in array.shape),
        finite_samples=int(finite.size),
        amplitude_min=float(np.min(finite)),
        amplitude_max=float(np.max(finite)),
        amplitude_mean=float(np.mean(finite)),
        intensity_max=float(np.max(intensity)),
        intensity_mean=float(np.mean(intensity)),
        intensity_percentile_95=float(np.percentile(intensity, 95)),
        maximum_enhancement=float(np.max(finite) / float(incident_amplitude)),
        mean_enhancement=float(np.mean(finite) / float(incident_amplitude)),
    )


def check_energy_balance(
    reflectance: Sequence[float],
    transmittance: Sequence[float],
    absorption: Sequence[float],
    *,
    tolerance: float = 1e-3,
) -> EnergyBalance:
    """Measure numerical error in ``reflectance + transmittance + absorption``."""

    if tolerance <= 0:
        raise AnalysisError("tolerance must be positive.")
    arrays = [np.asarray(item, dtype=float).reshape(-1) for item in (
        reflectance, transmittance, absorption
    )]
    if len({array.size for array in arrays}) != 1:
        raise AnalysisError("R, T, and A arrays must have the same length.")
    finite = np.isfinite(arrays[0]) & np.isfinite(arrays[1]) & np.isfinite(arrays[2])
    residual = arrays[0][finite] + arrays[1][finite] + arrays[2][finite] - 1.0
    if residual.size == 0:
        raise AnalysisError("R, T, and A contain no common finite samples.")
    return EnergyBalance(
        sample_count=int(residual.size),
        residual_mean=float(np.mean(residual)),
        residual_rms=float(np.sqrt(np.mean(residual**2))),
        residual_abs_max=float(np.max(np.abs(residual))),
        within_tolerance_fraction=float(np.mean(np.abs(residual) <= tolerance)),
        tolerance=float(tolerance),
    )


def load_run(
    run: str | Path | Mapping[str, Any] | Any,
    *,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a result object, JSON summary, or database run identifier."""

    if is_dataclass(run):
        return _jsonable(asdict(run))
    if isinstance(run, Mapping):
        return dict(run)
    path = Path(run).expanduser() if isinstance(run, (str, Path)) else None
    if path is not None and path.is_file():
        if path.suffix.lower() != ".json":
            raise AnalysisError("Run files must be JSON summaries.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"Could not read run summary {path}: {exc}") from exc
    if isinstance(run, str):
        return SimulationDatabase(database_path).export_run(run)
    raise AnalysisError("run must be a result, mapping, JSON path, or run ID.")


def extract_spectrum(
    run: str | Path | Mapping[str, Any] | Any,
    monitor_name: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract one monitor from an in-memory or database-backed run."""

    payload = load_run(run, database_path=database_path)
    spectra = payload.get("spectra", {})
    if monitor_name not in spectra:
        available = ", ".join(sorted(map(str, spectra))) or "none"
        raise AnalysisError(
            f"Monitor {monitor_name!r} is unavailable. Available monitors: {available}."
        )
    monitor = spectra[monitor_name]
    if isinstance(monitor, Mapping):
        wavelengths = monitor.get("wavelengths_um", monitor.get("wavelength_um"))
        values = monitor.get("values", monitor.get("value"))
    elif monitor and isinstance(monitor[0], Mapping):
        wavelengths = [sample["wavelength_um"] for sample in monitor]
        values = [sample.get("value", sample.get("value_real", 0.0)) for sample in monitor]
    else:
        wavelengths = payload.get("wavelengths_um")
        values = monitor
    if wavelengths is None or values is None:
        raise AnalysisError(f"Monitor {monitor_name!r} has incomplete spectral data.")
    return prepare_spectrum(wavelengths, values)


def analyze_run(
    run: str | Path | Mapping[str, Any] | Any,
    *,
    database_path: str | Path | None = None,
    monitors: Sequence[str] | None = None,
    prominence: float | None = None,
    include_fields: bool = True,
) -> dict[str, Any]:
    """Produce a JSON-ready analysis report for an entire simulation run."""

    payload = load_run(run, database_path=database_path)
    spectra = payload.get("spectra", {})
    selected = list(monitors) if monitors is not None else list(spectra)
    spectral_reports: dict[str, Any] = {}
    for name in selected:
        wavelengths, values = extract_spectrum(payload, name)
        spectral_reports[name] = analyze_spectrum(
            wavelengths,
            values,
            monitor_name=name,
            peak_mode="max",
            prominence=prominence,
        ).to_dict()

    report: dict[str, Any] = {
        "run_id": _run_identifier(payload),
        "geometry_name": _geometry_name(payload),
        "spectra": spectral_reports,
    }
    if all(name in spectra for name in ("reflectance", "transmittance", "absorption")):
        _, reflectance = extract_spectrum(payload, "reflectance")
        _, transmittance = extract_spectrum(payload, "transmittance")
        _, absorption = extract_spectrum(payload, "absorption")
        report["energy_balance"] = check_energy_balance(
            reflectance, transmittance, absorption
        ).to_dict()

    field_reports: dict[str, Any] = {}
    if include_fields:
        for kind, artifact_path in _artifact_paths(payload).items():
            if not kind.startswith("field_") or not kind.endswith("_array"):
                continue
            path = Path(artifact_path)
            if path.is_file() and path.suffix.lower() == ".npy":
                component = kind[len("field_") : -len("_array")]
                field_reports[component] = analyze_field(
                    path, component=component
                ).to_dict()
    if field_reports:
        report["fields"] = field_reports
    return report


def export_analysis(
    analysis: Mapping[str, Any] | SpectrumAnalysis | FieldAnalysis | EnergyBalance,
    path: str | Path,
) -> Path:
    """Write an analysis object to a readable JSON file."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = analysis.to_dict() if hasattr(analysis, "to_dict") else analysis
    destination.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return destination


def calculate_fwhm(
    wavelengths_um: Sequence[float],
    values: Sequence[complex | float],
    *,
    mode: str = "max",
    component: str = "real",
) -> float | None:
    """Return the primary resonance's full width at half maximum in um."""

    peaks = find_spectral_peaks(
        wavelengths_um,
        values,
        mode=mode,
        component=component,
        max_peaks=1,
    )
    return peaks[0].width_um if peaks else None


def calculate_quality_factor(
    resonance_wavelength_um: float,
    fwhm_um: float,
) -> float:
    """Return the dimensionless resonance quality factor λ_res/FWHM."""

    if resonance_wavelength_um <= 0 or fwhm_um <= 0:
        raise AnalysisError("Resonance wavelength and FWHM must be positive.")
    return float(resonance_wavelength_um) / float(fwhm_um)


def spp_effective_index(
    epsilon_metal: complex | Sequence[complex],
    epsilon_dielectric: complex | Sequence[complex],
) -> complex | np.ndarray:
    r"""Return :math:`\sqrt{\epsilon_m\epsilon_d/(\epsilon_m+\epsilon_d)}`."""

    metal = np.asarray(epsilon_metal, dtype=complex)
    dielectric = np.asarray(epsilon_dielectric, dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sqrt(metal * dielectric / (metal + dielectric))
    result = _positive_imaginary_branch(result)
    return _scalar_or_array(result)


def spp_wavevector(
    wavelengths_um: float | Sequence[float],
    epsilon_metal: complex | Sequence[complex],
    epsilon_dielectric: complex | Sequence[complex],
) -> complex | np.ndarray:
    """Return the complex SPP propagation constant in rad/um."""

    wavelengths = np.asarray(wavelengths_um, dtype=float)
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise AnalysisError("wavelengths_um must contain finite positive values.")
    result = (2 * np.pi / wavelengths) * np.asarray(
        spp_effective_index(epsilon_metal, epsilon_dielectric)
    )
    return _scalar_or_array(result)


def spp_propagation_length(
    wavelengths_um: float | Sequence[float],
    epsilon_metal: complex | Sequence[complex],
    epsilon_dielectric: complex | Sequence[complex],
) -> float | np.ndarray:
    """Return 1/e intensity propagation length in micrometres."""

    wavevector = np.asarray(
        spp_wavevector(wavelengths_um, epsilon_metal, epsilon_dielectric)
    )
    attenuation = 2 * np.abs(np.imag(wavevector))
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(attenuation > 0, 1.0 / attenuation, np.inf)
    return _scalar_or_array(result)


def spp_penetration_depths(
    wavelengths_um: float | Sequence[float],
    epsilon_metal: complex | Sequence[complex],
    epsilon_dielectric: complex | Sequence[complex],
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return approximate field penetration depths (metal, dielectric) in um."""

    wavelengths = np.asarray(wavelengths_um, dtype=float)
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise AnalysisError("wavelengths_um must contain finite positive values.")
    metal = np.asarray(epsilon_metal, dtype=complex)
    dielectric = np.asarray(epsilon_dielectric, dtype=complex)
    k0 = 2 * np.pi / wavelengths
    kspp = np.asarray(spp_wavevector(wavelengths, metal, dielectric))
    decay_metal = _positive_real_branch(np.sqrt(kspp**2 - metal * k0**2))
    decay_dielectric = _positive_real_branch(np.sqrt(kspp**2 - dielectric * k0**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        metal_depth = np.where(np.real(decay_metal) > 0, 1 / np.real(decay_metal), np.inf)
        dielectric_depth = np.where(
            np.real(decay_dielectric) > 0, 1 / np.real(decay_dielectric), np.inf
        )
    return _scalar_or_array(metal_depth), _scalar_or_array(dielectric_depth)


def _detection_signal(signal: np.ndarray, mode: str) -> np.ndarray:
    if mode == "max":
        return np.asarray(signal, dtype=float)
    if mode == "min":
        return -np.asarray(signal, dtype=float)
    if mode == "absolute":
        return np.abs(np.asarray(signal, dtype=float))
    raise AnalysisError("peak_mode must be max, min, or absolute.")


def _peak_indices(
    wavelengths: np.ndarray,
    detection: np.ndarray,
    *,
    prominence: float | None,
    minimum_distance_um: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if detection.size == 1:
        return np.array([0]), np.array([0.0])
    distance_samples = None
    if minimum_distance_um is not None and wavelengths.size > 1:
        spacing = float(np.median(np.diff(wavelengths)))
        if spacing > 0:
            distance_samples = max(1, int(math.ceil(minimum_distance_um / spacing)))
    try:
        from scipy.signal import find_peaks

        kwargs: dict[str, Any] = {}
        if prominence is not None:
            kwargs["prominence"] = float(prominence)
        if distance_samples is not None:
            kwargs["distance"] = distance_samples
        indices, properties = find_peaks(detection, **kwargs)
        values = properties.get("prominences")
        if values is None:
            values = np.asarray([
                _simple_prominence(detection, int(index)) for index in indices
            ])
    except ImportError:
        candidates = [
            index
            for index in range(1, detection.size - 1)
            if detection[index] >= detection[index - 1]
            and detection[index] > detection[index + 1]
        ]
        if not candidates:
            candidates = [int(np.argmax(detection))]
        values = np.asarray([_simple_prominence(detection, index) for index in candidates])
        indices = np.asarray(candidates, dtype=int)
        keep = np.ones(indices.size, dtype=bool)
        if prominence is not None:
            keep &= values >= prominence
        indices, values = indices[keep], values[keep]
        if distance_samples is not None:
            indices, values = _enforce_peak_distance(
                indices, values, detection, distance_samples
            )
    if indices.size == 0:
        global_index = int(np.argmax(detection))
        global_prominence = _simple_prominence(detection, global_index)
        if prominence is None or global_prominence >= prominence:
            return np.array([global_index]), np.array([global_prominence])
    return np.asarray(indices, dtype=int), np.asarray(values, dtype=float)


def _simple_prominence(signal: np.ndarray, index: int) -> float:
    left_min = float(np.min(signal[: index + 1]))
    right_min = float(np.min(signal[index:]))
    return max(0.0, float(signal[index]) - max(left_min, right_min))


def _enforce_peak_distance(
    indices: np.ndarray,
    prominences: np.ndarray,
    signal: np.ndarray,
    distance: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(indices.size),
        key=lambda item: (prominences[item], signal[indices[item]]),
        reverse=True,
    )
    chosen: list[int] = []
    for candidate in order:
        index = int(indices[candidate])
        if all(abs(index - other) >= distance for other in chosen):
            chosen.append(index)
    chosen.sort()
    lookup = {int(index): float(value) for index, value in zip(indices, prominences)}
    return np.asarray(chosen), np.asarray([lookup[index] for index in chosen])


def _half_height_width(
    wavelengths: np.ndarray, detection: np.ndarray, index: int
) -> float | None:
    if detection.size < 3 or index <= 0 or index >= detection.size - 1:
        return None
    baseline = max(
        float(np.min(detection[: index + 1])),
        float(np.min(detection[index:])),
    )
    half = baseline + 0.5 * (float(detection[index]) - baseline)
    left = None
    for position in range(index - 1, -1, -1):
        if detection[position] <= half:
            left = _linear_crossing(
                wavelengths[position], detection[position],
                wavelengths[position + 1], detection[position + 1], half,
            )
            break
    right = None
    for position in range(index, detection.size - 1):
        if detection[position + 1] <= half:
            right = _linear_crossing(
                wavelengths[position], detection[position],
                wavelengths[position + 1], detection[position + 1], half,
            )
            break
    if left is None or right is None or right <= left:
        return None
    return float(right - left)


def _linear_crossing(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return float(0.5 * (x0 + x1))
    fraction = (target - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def _load_array(value: np.ndarray | Sequence[Any] | str | Path) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise AnalysisError(f"Array file does not exist: {path}")
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path, allow_pickle=False))
        if path.suffix.lower() == ".npz":
            archive = np.load(path, allow_pickle=False)
            if len(archive.files) != 1:
                raise AnalysisError("NPZ field files must contain exactly one array.")
            return np.asarray(archive[archive.files[0]])
        raise AnalysisError("Field arrays must use .npy or single-array .npz files.")
    return np.asarray(value)


def _artifact_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        return {str(key): str(value) for key, value in artifacts.items()}
    result = {}
    for artifact in artifacts or []:
        if isinstance(artifact, Mapping) and "kind" in artifact and "path" in artifact:
            result[str(artifact["kind"])] = str(artifact["path"])
    return result


def _run_identifier(payload: Mapping[str, Any]) -> str | None:
    if "run_id" in payload:
        return str(payload["run_id"])
    if isinstance(payload.get("run"), Mapping):
        value = payload["run"].get("run_id")
        return str(value) if value is not None else None
    return None


def _geometry_name(payload: Mapping[str, Any]) -> str | None:
    if "geometry_name" in payload:
        return str(payload["geometry_name"])
    if isinstance(payload.get("run"), Mapping):
        value = payload["run"].get("geometry_name")
        return str(value) if value is not None else None
    return None


def _positive_imaginary_branch(value: np.ndarray) -> np.ndarray:
    return np.where(np.imag(value) < 0, -value, value)


def _positive_real_branch(value: np.ndarray) -> np.ndarray:
    return np.where(np.real(value) < 0, -value, value)


def _scalar_or_array(value: np.ndarray) -> Any:
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


__all__ = [
    "AnalysisError",
    "EnergyBalance",
    "FieldAnalysis",
    "SpectralPeak",
    "SpectrumAnalysis",
    "analyze_field",
    "analyze_run",
    "analyze_spectrum",
    "calculate_fwhm",
    "calculate_quality_factor",
    "check_energy_balance",
    "export_analysis",
    "extract_spectrum",
    "find_spectral_peaks",
    "load_run",
    "prepare_spectrum",
    "spp_effective_index",
    "spp_penetration_depths",
    "spp_propagation_length",
    "spp_wavevector",
]


# Readable compatibility aliases for earlier scripts.
analyze_simulation = analyze_run
field_enhancement = analyze_field
find_resonances = find_spectral_peaks
load_results = load_run
__all__.extend([
    "analyze_simulation",
    "field_enhancement",
    "find_resonances",
    "load_results",
])
