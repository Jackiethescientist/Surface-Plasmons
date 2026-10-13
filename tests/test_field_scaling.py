from pathlib import Path

import numpy as np
import pytest

from surface_plasmons.project import normalise_project_config
from surface_plasmons.workflow import (
    StageFields,
    _normalised_fields,
    _save_broadband_plots,
    _save_field_map,
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


def test_all_enhancement_outputs_use_field_amplitude_ratio() -> None:
    actual = _stage([[[[2.0]], [[4.0]]]])
    reference = _stage([[[[1.0]], [[1.0]]]])

    result = _normalised_fields(
        actual,
        reference,
        reference_component="Ex",
        mode="global_complex_reference",
    )

    assert result["amplitude_enhancement"].reshape(-1) == pytest.approx([2.0, 4.0])
    assert result["enhancement"].reshape(-1) == pytest.approx([2.0, 4.0])
    assert result["Ex_amplitude_ratio"].reshape(-1) == pytest.approx([2.0, 4.0])
    assert result["normalised_Ex"].reshape(-1) == pytest.approx([2.0, 4.0])
    assert result["normalised_reference_Ex"].reshape(-1) == pytest.approx(
        [1.0, 1.0]
    )
    assert result["normalised_scattered_Ex"].reshape(-1) == pytest.approx(
        [1.0, 3.0]
    )
    assert result["scattered_amplitude_enhancement"].reshape(-1) == pytest.approx(
        [1.0, 3.0]
    )
    assert "intensity_enhancement" not in result


def test_global_complex_reference_does_not_divide_by_each_point() -> None:
    actual = _stage([[[[2.0]], [[2.0]]]])
    reference = _stage([[[[1.0]], [[1e-9]]]])

    result = _normalised_fields(
        actual,
        reference,
        reference_component="Ex",
        mode="global_complex_reference",
    )

    # The median reference is global, so one tiny point cannot create a 1e9 spike.
    assert float(np.nanmax(result["amplitude_enhancement"])) < 10.0


def test_component_maps_use_unsquared_amplitude_ratios() -> None:
    actual = _stage(
        [[[[3.0]], [[3.0]]]],
        ey=[[[[4.0]], [[4.0]]]],
    )
    reference = _stage([[[[1.0]], [[1.0]]]])

    result = _normalised_fields(
        actual,
        reference,
        reference_component="Ex",
        mode="pointwise_amplitude_reference",
    )

    assert result["Ex_amplitude_ratio"].reshape(-1) == pytest.approx([3.0, 3.0])
    assert result["Ey_amplitude_ratio"].reshape(-1) == pytest.approx([4.0, 4.0])
    assert result["amplitude_enhancement"].reshape(-1) == pytest.approx([5.0, 5.0])


def test_saved_field_component_maps_use_e_over_e_reference(tmp_path) -> None:
    shape = (1, 2, 2, 2)
    zeros = np.zeros(shape, dtype=complex)
    actual = StageFields(
        frequencies=np.asarray([1.5]),
        fields={"Ex": zeros + 3.0, "Ey": zeros + 4.0, "Ez": zeros},
        coordinates={
            "x": np.asarray([-0.1, 0.1]),
            "y": np.asarray([0.05, 0.15]),
            "z": np.asarray([-0.1, 0.1]),
        },
        output_directory=tmp_path,
    )
    reference = StageFields(
        frequencies=np.asarray([1.5]),
        fields={"Ex": zeros + 1.0, "Ey": zeros, "Ez": zeros},
        coordinates=actual.coordinates,
        output_directory=tmp_path,
    )

    class Workspace:
        directory = tmp_path

        @staticmethod
        def record_artifact(_name, _path):
            return None

    config = normalise_project_config(
        {"simulation_name": "component-map-test", "geometry": {"particle_count": 1}}
    )
    report = _save_field_map(600.0, actual, reference, Workspace(), config)
    folder = tmp_path / "field_maps" / "600.000nm"

    assert np.load(folder / "field_total_enhancement.npy") == pytest.approx(5.0)
    assert np.load(folder / "field_amplitude_enhancement.npy") == pytest.approx(5.0)
    assert not (folder / "field_intensity_enhancement.npy").exists()
    assert report["scale_definition"] == "|E| / |E_reference|"
    assert (folder / "field_components_xy.png").is_file()


def test_only_amplitude_heatmap_is_created(
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
    config = normalise_project_config(
        {"simulation_name": "plot-scale-test", "geometry": {"particle_count": 1}}
    )

    _save_broadband_plots(
        config,
        fields,
        amplitude,
        fields.coordinates["x"],
        Workspace(),
    )

    wavelength_order = np.argsort(1000.0 / fields.frequencies)
    assert captured_images[0] == pytest.approx(amplitude[wavelength_order])
    assert len(captured_images) == 1
    assert (tmp_path / "frequency_scan" / "field_position_frequency_heatmap.png").is_file()
    assert not (
        tmp_path / "frequency_scan" / "field_intensity_position_frequency_heatmap.png"
    ).exists()
