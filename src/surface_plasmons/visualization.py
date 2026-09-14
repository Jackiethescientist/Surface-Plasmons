"""
Matplotlib is imported lazily, so importing :mod:`surface_plasmons` does not
open a graphical backend.  Every plotting function returns ``(figure, axes)``
and can optionally save a PNG, PDF, or SVG using ``save_path``.
"""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np

from .analysis import (
    extract_spectrum,
    load_run,
    prepare_spectrum,
    spp_wavevector,
)


class VisualizationError(ValueError):
    """Raised when data cannot be represented by the requested plot."""


@dataclass(frozen=True, slots=True)
class PlotStyle:
    """Consistent defaults for reports and research-paper figures."""

    figsize: tuple[float, float] = (7.2, 5.0)
    dpi: int = 180
    font_size: float = 11.0
    line_width: float = 2.0
    grid_alpha: float = 0.25
    color_cycle: tuple[str, ...] = (
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#000000",
    )

    def validate(self) -> None:
        if len(self.figsize) != 2 or any(value <= 0 for value in self.figsize):
            raise VisualizationError("figsize must contain two positive values.")
        if self.dpi < 1 or self.font_size <= 0 or self.line_width <= 0:
            raise VisualizationError("dpi, font_size, and line_width must be positive.")


DEFAULT_STYLE = PlotStyle()


def plot_spectrum(
    wavelengths_um: Sequence[float],
    values: Sequence[complex | float],
    *,
    label: str | None = None,
    component: str = "real",
    wavelength_unit: str = "nm",
    title: str | None = None,
    ylabel: str = "Response",
    normalize: bool = False,
    mark_peak: bool = False,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot one spectrum against wavelength."""

    wavelengths, signal = prepare_spectrum(
        wavelengths_um, values, component=component
    )
    if normalize:
        scale = float(np.max(np.abs(signal)))
        if scale > 0:
            signal = signal / scale
        ylabel = f"Normalized {ylabel.lower()}"
    x, xlabel = _wavelength_axis(wavelengths, wavelength_unit)
    figure, axis = _get_figure_axis(ax, style)
    _apply_style(axis, style)
    axis.plot(x, signal, label=label, linewidth=style.line_width)
    if mark_peak:
        index = int(np.argmax(signal))
        axis.scatter([x[index]], [signal[index]], color="#D55E00", zorder=4)
        axis.annotate(
            f"{x[index]:.4g} {wavelength_unit}",
            (x[index], signal[index]),
            xytext=(8, 8),
            textcoords="offset points",
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title)
    if label:
        axis.legend(frameon=False)
    _finish(figure, style, save_path, show)
    return figure, axis


def compare_spectra(
    spectra: Mapping[str, tuple[Sequence[float], Sequence[complex | float]]],
    *,
    component: str = "real",
    wavelength_unit: str = "nm",
    title: str = "Spectral comparison",
    ylabel: str = "Response",
    normalize: bool = False,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Overlay labelled spectra on common wavelength axes."""

    if not spectra:
        raise VisualizationError("At least one spectrum is required.")
    figure, axis = _get_figure_axis(ax, style)
    _apply_style(axis, style)
    for index, (label, (wavelengths, values)) in enumerate(spectra.items()):
        wavelength_array, signal = prepare_spectrum(
            wavelengths, values, component=component
        )
        if normalize:
            scale = float(np.max(np.abs(signal)))
            if scale > 0:
                signal = signal / scale
        x, xlabel = _wavelength_axis(wavelength_array, wavelength_unit)
        axis.plot(
            x,
            signal,
            label=label,
            linewidth=style.line_width,
            color=style.color_cycle[index % len(style.color_cycle)],
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(f"Normalized {ylabel.lower()}" if normalize else ylabel)
    axis.set_title(title)
    axis.legend(frameon=False)
    _finish(figure, style, save_path, show)
    return figure, axis


def plot_run_spectra(
    run: str | Path | Mapping[str, Any] | Any,
    *,
    monitors: Sequence[str] | None = None,
    database_path: str | Path | None = None,
    wavelength_unit: str = "nm",
    title: str | None = None,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot selected monitors from a result, JSON summary, or database run ID."""

    payload = load_run(run, database_path=database_path)
    available = list(payload.get("spectra", {}))
    selected = list(monitors) if monitors is not None else _preferred_monitors(available)
    if not selected:
        raise VisualizationError("The run contains no spectra.")
    data = {
        name: extract_spectrum(payload, name)
        for name in selected
    }
    if title is None:
        run_id = _run_id(payload)
        title = f"Simulation spectra — {run_id}" if run_id else "Simulation spectra"
    return compare_spectra(
        data,
        wavelength_unit=wavelength_unit,
        title=title,
        ylabel="Flux / normalized response",
        ax=ax,
        style=style,
        save_path=save_path,
        show=show,
    )


def plot_energy_balance(
    wavelengths_um: Sequence[float],
    reflectance: Sequence[float],
    transmittance: Sequence[float],
    absorption: Sequence[float],
    *,
    wavelength_unit: str = "nm",
    include_sum: bool = True,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot R, T, A and optionally their numerical sum."""

    wavelengths = np.asarray(wavelengths_um, dtype=float).reshape(-1)
    arrays = {
        "Reflectance": np.asarray(reflectance, dtype=float).reshape(-1),
        "Transmittance": np.asarray(transmittance, dtype=float).reshape(-1),
        "Absorption": np.asarray(absorption, dtype=float).reshape(-1),
    }
    if any(values.size != wavelengths.size for values in arrays.values()):
        raise VisualizationError("Wavelength, R, T, and A arrays must have equal lengths.")
    x, xlabel = _wavelength_axis(wavelengths, wavelength_unit)
    order = np.argsort(x)
    figure, axis = _get_figure_axis(ax, style)
    _apply_style(axis, style)
    for index, (name, values) in enumerate(arrays.items()):
        axis.plot(
            x[order],
            values[order],
            label=name,
            linewidth=style.line_width,
            color=style.color_cycle[index],
        )
    if include_sum:
        total = sum(arrays.values())
        axis.plot(
            x[order], total[order], label="R + T + A", color="#666666",
            linewidth=1.4, linestyle="--",
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Fraction of incident power")
    axis.set_title("Energy balance")
    axis.legend(frameon=False, ncol=2)
    _finish(figure, style, save_path, show)
    return figure, axis


def plot_field(
    field: np.ndarray | Sequence[Any] | str | Path,
    *,
    component: str = "E",
    quantity: str = "magnitude",
    plane_index: int | None = None,
    extent_um: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    cmap: str | None = None,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot a 2-D slice of a real or complex field array."""

    array = _load_array(field)
    data = _slice_2d(array, plane_index)
    quantity_key = quantity.lower()
    if quantity_key in {"magnitude", "abs", "amplitude"}:
        displayed = np.abs(data)
        colorbar_label = f"|{component}|"
        cmap = cmap or "magma"
    elif quantity_key == "intensity":
        displayed = np.abs(data) ** 2
        colorbar_label = f"|{component}|²"
        cmap = cmap or "inferno"
    elif quantity_key == "real":
        displayed = np.real(data)
        colorbar_label = f"Re({component})"
        cmap = cmap or "RdBu_r"
    elif quantity_key == "imag":
        displayed = np.imag(data)
        colorbar_label = f"Im({component})"
        cmap = cmap or "RdBu_r"
    elif quantity_key == "phase":
        displayed = np.angle(data)
        colorbar_label = f"phase({component}) [rad]"
        cmap = cmap or "twilight"
    else:
        raise VisualizationError(
            "quantity must be magnitude, intensity, real, imag, or phase."
        )
    figure, axis = _get_figure_axis(ax, style)
    kwargs: dict[str, Any] = {
        "origin": "lower",
        "aspect": "equal",
        "cmap": cmap,
    }
    if extent_um is not None:
        if len(extent_um) != 4:
            raise VisualizationError("extent_um must be (xmin, xmax, ymin, ymax).")
        kwargs["extent"] = extent_um
    if quantity_key in {"real", "imag"}:
        limit = float(np.nanmax(np.abs(displayed)))
        if math.isfinite(limit) and limit > 0:
            kwargs.update(vmin=-limit, vmax=limit)
    image = axis.imshow(displayed.T, **kwargs)
    axis.set_xlabel("x (µm)" if extent_um else "x grid index")
    axis.set_ylabel("y (µm)" if extent_um else "y grid index")
    axis.set_title(title or f"{component} {quantity_key}")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    _finish(figure, style, save_path, show)
    return figure, axis


def plot_epsilon(
    epsilon: np.ndarray | Sequence[Any] | str | Path,
    *,
    component: str = "real",
    plane_index: int | None = None,
    extent_um: tuple[float, float, float, float] | None = None,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot relative permittivity using real, imaginary, or magnitude values."""

    component_key = component.lower()
    if component_key not in {"real", "imag", "magnitude", "abs"}:
        raise VisualizationError("epsilon component must be real, imag, or magnitude.")
    quantity = "magnitude" if component_key == "abs" else component_key
    return plot_field(
        epsilon,
        component="εᵣ",
        quantity=quantity,
        plane_index=plane_index,
        extent_um=extent_um,
        title=f"Relative permittivity — {quantity}",
        cmap="viridis" if quantity in {"real", "magnitude"} else "RdBu_r",
        ax=ax,
        style=style,
        save_path=save_path,
        show=show,
    )


def plot_sweep_1d(
    sweep: str | Path | Mapping[str, Any] | Any,
    parameter: str,
    metric: str,
    *,
    aggregate: str = "mean",
    parameter_label: str | None = None,
    metric_label: str | None = None,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot one output metric against one swept parameter."""

    payload = _load_sweep_payload(sweep)
    grouped: dict[float, list[float]] = {}
    for run in payload.get("runs", []):
        if run.get("status") not in {"completed", "skipped"}:
            continue
        parameters = run.get("parameters", {})
        metrics = run.get("metrics", {})
        if parameter not in parameters or metric not in metrics:
            continue
        try:
            x_value = float(parameters[parameter])
            y_value = float(metrics[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(x_value) and math.isfinite(y_value):
            grouped.setdefault(x_value, []).append(y_value)
    if not grouped:
        raise VisualizationError(
            f"No completed rows contain numeric {parameter!r} and {metric!r}."
        )
    x = np.asarray(sorted(grouped), dtype=float)
    y = np.asarray([_aggregate(grouped[value], aggregate) for value in x])
    spread = np.asarray([np.std(grouped[value]) for value in x])
    figure, axis = _get_figure_axis(ax, style)
    _apply_style(axis, style)
    axis.plot(x, y, marker="o", linewidth=style.line_width)
    if any(len(grouped[value]) > 1 for value in x):
        axis.fill_between(x, y - spread, y + spread, alpha=0.18, label="±1 SD")
        axis.legend(frameon=False)
    axis.set_xlabel(parameter_label or parameter)
    axis.set_ylabel(metric_label or metric)
    axis.set_title(f"{metric_label or metric} vs {parameter_label or parameter}")
    _finish(figure, style, save_path, show)
    return figure, axis


def plot_sweep_heatmap(
    sweep: str | Path | Mapping[str, Any] | Any,
    x_parameter: str,
    y_parameter: str,
    metric: str,
    *,
    aggregate: str = "mean",
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot a two-parameter sweep as an aggregated metric heatmap."""

    payload = _load_sweep_payload(sweep)
    grouped: dict[tuple[Any, Any], list[float]] = {}
    for run in payload.get("runs", []):
        if run.get("status") not in {"completed", "skipped"}:
            continue
        parameters = run.get("parameters", {})
        metrics = run.get("metrics", {})
        if x_parameter not in parameters or y_parameter not in parameters or metric not in metrics:
            continue
        try:
            value = float(metrics[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            grouped.setdefault(
                (parameters[x_parameter], parameters[y_parameter]), []
            ).append(value)
    if not grouped:
        raise VisualizationError("No completed rows contain the requested heatmap values.")
    x_values = _sorted_unique(key[0] for key in grouped)
    y_values = _sorted_unique(key[1] for key in grouped)
    matrix = np.full((len(y_values), len(x_values)), np.nan)
    for y_index, y_value in enumerate(y_values):
        for x_index, x_value in enumerate(x_values):
            samples = grouped.get((x_value, y_value), [])
            if samples:
                matrix[y_index, x_index] = _aggregate(samples, aggregate)
    figure, axis = _get_figure_axis(ax, style)
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(x_values)), [_format_tick(value) for value in x_values])
    axis.set_yticks(range(len(y_values)), [_format_tick(value) for value in y_values])
    axis.set_xlabel(x_parameter)
    axis.set_ylabel(y_parameter)
    axis.set_title(metric)
    figure.colorbar(image, ax=axis, label=metric)
    _finish(figure, style, save_path, show)
    return figure, axis


def plot_spp_dispersion(
    wavelengths_um: Sequence[float],
    epsilon_metal: Sequence[complex] | complex,
    epsilon_dielectric: Sequence[complex] | complex,
    *,
    wavelength_unit: str = "nm",
    normalized: bool = True,
    ax: Any | None = None,
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, Any]:
    """Plot the real and imaginary SPP propagation constant."""

    wavelengths = np.asarray(wavelengths_um, dtype=float)
    kspp = np.asarray(spp_wavevector(wavelengths, epsilon_metal, epsilon_dielectric))
    if kspp.ndim == 0:
        kspp = np.full(wavelengths.shape, kspp.item(), dtype=complex)
    if kspp.shape != wavelengths.shape:
        try:
            kspp = np.broadcast_to(kspp, wavelengths.shape)
        except ValueError as exc:
            raise VisualizationError("Permittivity data cannot be broadcast to wavelengths.") from exc
    if normalized:
        k0 = 2 * np.pi / wavelengths
        kspp = kspp / k0
        ylabel = "Normalized wavevector kₛₚₚ/k₀"
    else:
        ylabel = "Wavevector (rad/µm)"
    x, xlabel = _wavelength_axis(wavelengths, wavelength_unit)
    order = np.argsort(x)
    figure, axis = _get_figure_axis(ax, style)
    _apply_style(axis, style)
    axis.plot(
        x[order], np.real(kspp)[order], label="Real part",
        linewidth=style.line_width, color=style.color_cycle[0],
    )
    axis.plot(
        x[order], np.imag(kspp)[order], label="Imaginary part",
        linewidth=style.line_width, color=style.color_cycle[1],
    )
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title("Surface-plasmon-polariton dispersion")
    axis.legend(frameon=False)
    _finish(figure, style, save_path, show)
    return figure, axis


def create_run_summary_figure(
    run: str | Path | Mapping[str, Any] | Any,
    *,
    database_path: str | Path | None = None,
    wavelength_unit: str = "nm",
    style: PlotStyle = DEFAULT_STYLE,
    save_path: str | Path | None = None,
    show: bool = False,
) -> tuple[Any, np.ndarray]:
    """Create a four-panel spectrum, epsilon, field, and metadata summary."""

    payload = load_run(run, database_path=database_path)
    plt = _pyplot()
    style.validate()
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    available = list(payload.get("spectra", {}))
    selected = _preferred_monitors(available)
    if selected:
        plot_run_spectra(
            payload,
            monitors=selected,
            wavelength_unit=wavelength_unit,
            ax=axes[0, 0],
            style=style,
        )
    else:
        _empty_panel(axes[0, 0], "No spectra saved")

    artifacts = _artifact_paths(payload)
    epsilon_path = artifacts.get("epsilon_array")
    if epsilon_path and Path(epsilon_path).is_file():
        plot_epsilon(epsilon_path, ax=axes[0, 1], style=style)
    else:
        _empty_panel(axes[0, 1], "No epsilon array saved")

    field_items = [
        (kind, path)
        for kind, path in artifacts.items()
        if kind.startswith("field_") and kind.endswith("_array") and Path(path).is_file()
    ]
    if field_items:
        kind, field_path = field_items[0]
        component = kind[len("field_") : -len("_array")]
        plot_field(field_path, component=component, ax=axes[1, 0], style=style)
    else:
        _empty_panel(axes[1, 0], "No field array saved")

    axes[1, 1].axis("off")
    metadata = _summary_metadata(payload)
    axes[1, 1].text(
        0.02,
        0.98,
        "\n".join(f"{key}: {value}" for key, value in metadata.items()),
        va="top",
        ha="left",
        fontsize=style.font_size,
        family="monospace",
    )
    figure.suptitle(f"Simulation summary — {_run_id(payload) or 'run'}", fontsize=14)
    _finish(figure, style, save_path, show)
    return figure, axes


def save_figure(
    figure: Any,
    path: str | Path,
    *,
    dpi: int = 180,
    transparent: bool = False,
    close: bool = False,
) -> Path:
    """Save a Matplotlib figure and return its absolute path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", transparent=transparent)
    if close:
        _pyplot().close(figure)
    return destination


def _get_figure_axis(ax: Any | None, style: PlotStyle) -> tuple[Any, Any]:
    style.validate()
    if ax is not None:
        return ax.figure, ax
    figure, axis = _pyplot().subplots(figsize=style.figsize, constrained_layout=True)
    return figure, axis


def _apply_style(axis: Any, style: PlotStyle) -> None:
    axis.grid(True, alpha=style.grid_alpha)
    axis.tick_params(labelsize=style.font_size)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _finish(
    figure: Any,
    style: PlotStyle,
    save_path: str | Path | None,
    show: bool,
) -> None:
    if save_path is not None:
        save_figure(figure, save_path, dpi=style.dpi)
    if show:
        _pyplot().show()


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for visualization. Install it in the plasmon environment."
        ) from exc
    return plt


def _wavelength_axis(
    wavelengths_um: np.ndarray, unit: str
) -> tuple[np.ndarray, str]:
    key = unit.lower().replace("μ", "u").replace("µ", "u")
    if key in {"um", "micrometre", "micrometer", "micrometres", "micrometers"}:
        return np.asarray(wavelengths_um, dtype=float), "Wavelength (µm)"
    if key in {"nm", "nanometre", "nanometer", "nanometres", "nanometers"}:
        return np.asarray(wavelengths_um, dtype=float) * 1000.0, "Wavelength (nm)"
    raise VisualizationError("wavelength_unit must be 'um' or 'nm'.")


def _load_array(value: np.ndarray | Sequence[Any] | str | Path) -> np.ndarray:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise VisualizationError(f"Array file does not exist: {path}")
        if path.suffix.lower() == ".npy":
            return np.asarray(np.load(path, allow_pickle=False))
        if path.suffix.lower() == ".npz":
            archive = np.load(path, allow_pickle=False)
            if len(archive.files) != 1:
                raise VisualizationError("NPZ plots require exactly one array.")
            return np.asarray(archive[archive.files[0]])
        raise VisualizationError("Arrays must use .npy or single-array .npz files.")
    return np.asarray(value)


def _slice_2d(array: np.ndarray, plane_index: int | None) -> np.ndarray:
    value = np.squeeze(np.asarray(array))
    if value.ndim == 0:
        return value.reshape(1, 1)
    if value.ndim == 1:
        return value.reshape(value.shape[0], 1)
    while value.ndim > 2:
        index = value.shape[-1] // 2 if plane_index is None else int(plane_index)
        if not -value.shape[-1] <= index < value.shape[-1]:
            raise VisualizationError("plane_index is outside the array.")
        value = np.take(value, index, axis=-1)
    return value


def _preferred_monitors(available: Sequence[str]) -> list[str]:
    preferred = ["reflectance", "transmittance", "absorption"]
    selected = [name for name in preferred if name in available]
    return selected or list(available)


def _load_sweep_payload(value: str | Path | Mapping[str, Any] | Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    from .sweep import load_sweep

    return load_sweep(value).to_dict()


def _aggregate(values: Sequence[float], method: str) -> float:
    array = np.asarray(values, dtype=float)
    key = method.lower()
    if key == "mean":
        return float(np.mean(array))
    if key == "median":
        return float(np.median(array))
    if key == "min":
        return float(np.min(array))
    if key == "max":
        return float(np.max(array))
    raise VisualizationError("aggregate must be mean, median, min, or max.")


def _sorted_unique(values: Sequence[Any] | Any) -> list[Any]:
    unique = list(dict.fromkeys(values))
    try:
        return sorted(unique, key=float)
    except (TypeError, ValueError):
        return sorted(unique, key=str)


def _format_tick(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return str(value)


def _artifact_paths(payload: Mapping[str, Any]) -> dict[str, str]:
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, Mapping):
        return {str(key): str(value) for key, value in artifacts.items()}
    return {
        str(item["kind"]): str(item["path"])
        for item in artifacts or []
        if isinstance(item, Mapping) and "kind" in item and "path" in item
    }


def _run_id(payload: Mapping[str, Any]) -> str | None:
    if payload.get("run_id") is not None:
        return str(payload["run_id"])
    run = payload.get("run")
    if isinstance(run, Mapping) and run.get("run_id") is not None:
        return str(run["run_id"])
    return None


def _summary_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload.get("run") if isinstance(payload.get("run"), Mapping) else payload
    metrics = payload.get("metrics", {})
    if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        metrics = {
            item.get("name"): item.get("value")
            for item in metrics
            if isinstance(item, Mapping)
        }
    return {
        "run_id": _run_id(payload) or "unknown",
        "status": run.get("status", "unknown"),
        "geometry": run.get("geometry_name", "unknown"),
        "elapsed_s": _format_tick(
            run.get("elapsed_seconds", metrics.get("elapsed_seconds", "unknown"))
        ),
        "meep_time": _format_tick(metrics.get("meep_time", "unknown")),
        "output": run.get("output_directory", payload.get("output_directory", "unknown")),
    }


def _empty_panel(axis: Any, message: str) -> None:
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", color="#666666")


__all__ = [
    "DEFAULT_STYLE",
    "PlotStyle",
    "VisualizationError",
    "compare_spectra",
    "create_run_summary_figure",
    "plot_energy_balance",
    "plot_epsilon",
    "plot_field",
    "plot_run_spectra",
    "plot_spectrum",
    "plot_spp_dispersion",
    "plot_sweep_1d",
    "plot_sweep_heatmap",
    "save_figure",
]


# Compatibility names used by earlier plotting scripts.
plot_field_map = plot_field
plot_results = plot_run_spectra
plot_sweep = plot_sweep_1d
__all__.extend(["plot_field_map", "plot_results", "plot_sweep"])
