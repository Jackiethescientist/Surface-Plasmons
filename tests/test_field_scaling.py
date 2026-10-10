from pathlib import Path

import numpy as np
import pytest

from surface_plasmons.project import normalise_project_config
from surface_plasmons.workflow import (
    StageFields,
    _normalised_fields,
    _save_broadband_plots,
)


def _stage(ex, ey=None, ez=None):
    ex = np.asarray(ex, dtype=complex)
    zeros = np.zeros_like(ex)
    return StageFields(
        frequencies=np.asarray([1.5]),
        fields={"Ex": ex, "Ey": zeros if ey is None else np.asarray(ey), "Ez": zeros if ez is None else np.asarray(ez)},
        coordinates={"x": np.asarray([0.0, 0.1]), "y": np.asarray([0.1]), "z": np.asarray([0.0])},
        output_directory=Path("."),
    )


def test_amplitude_and_intensity_scales_are_distinct_and_label_ready() -> None:
    actual = _stage([[[[2.0]], [[4.0]]]])
    reference = _stage([[[[1.0]], [[1.0]]]])

    result = _normalised_fields(
        actual,
        reference,
        reference_component="Ex",
        mode="global_complex_reference",
    )

    assert result["amplitude_enhancement"].reshape(-1) == pytest.approx([2.0, 4.0])
    assert result["intensity_enhancement"].reshape(-1) == pytest.approx([4.0, 16.0])
    assert result["normalised_Ex"].reshape(-1) == pytest.approx([2.0, 4.0])


def test_global_complex_reference_does_not_divide_by_each_point() -> None:
    actual = _stage([[[[2.0]], [[2.0]]]])
    reference = _stage([[[[1.0]], [[1e-9]]]])

    result = _normalised_fields(
        actual,
        reference,
        reference_component="Ex",
        mode="global_complex_reference",
    )

    # The median reference is global, so one tiny point cannot create a 1e18 spike.
    assert float(np.nanmax(result["intensity_enhancement"])) < 100.0


def test_canonical_heatmap_is_amplitude_not_squared_intensity(
    tmp_path, monkeypatch
) -> None:
    import matplotlib.axes

    captured_images = []
    original_imshow = matplotlib.axes.Axes.imshow

    def capture_imshow(axis, values, *args, **kwargs):
        captured_images.append(np.asarray(values).copy())
        return original_imshow(axis, values, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", capture_imshow)

    class Workspace:
        directory = tmp_path

        @staticmethod
        def record_artifact(_name, _path):
            return None

    (tmp_path / "frequency_scan").mkdir()
    fields = StageFields(
        frequencies=np.asarray([2.0, 1.0]),
        fields={
            "Ex": np.ones((2, 2, 1, 1), dtype=complex),
            "Ey": np.zeros((2, 2, 1, 1), dtype=complex),
            "Ez": np.zeros((2, 2, 1, 1), dtype=complex),
        },
        coordinates={
            "x": np.asarray([-0.1, 0.1]),
            "y": np.asarray([0.1]),
            "z": np.asarray([0.0]),
        },
        output_directory=tmp_path,
    )
    amplitude = np.asarray([[2.0, 3.0], [4.0, 5.0]])
    intensity = amplitude**2
    config = normalise_project_config(
        {"simulation_name": "plot-scale-test", "geometry": {"particle_count": 1}}
    )

    _save_broadband_plots(
        config,
        fields,
        amplitude,
        intensity,
        fields.coordinates["x"],
        Workspace(),
    )

    wavelength_order = np.argsort(1000.0 / fields.frequencies)
    assert captured_images[0] == pytest.approx(amplitude[wavelength_order])
    assert captured_images[1] == pytest.approx(intensity[wavelength_order])
    assert (tmp_path / "frequency_scan" / "field_position_frequency_heatmap.png").is_file()
    assert (
        tmp_path / "frequency_scan" / "field_intensity_position_frequency_heatmap.png"
    ).is_file()
