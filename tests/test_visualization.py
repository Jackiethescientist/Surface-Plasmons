import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from surface_plasmons.visualization import plot_field, plot_spectrum


def test_spectrum_plot_can_be_saved_headlessly(tmp_path) -> None:
    wavelengths = np.linspace(0.45, 0.80, 51)
    values = np.sin(np.linspace(0.0, np.pi, wavelengths.size)) ** 2
    output = tmp_path / "spectrum.png"

    figure, axis = plot_spectrum(
        wavelengths,
        values,
        label="response",
        mark_peak=True,
        save_path=output,
    )

    assert output.is_file()
    assert axis.get_xlabel() == "Wavelength (nm)"
    plt.close(figure)


def test_complex_field_plot_can_be_saved_headlessly(tmp_path) -> None:
    x = np.linspace(-1.0, 1.0, 20)
    field = np.outer(np.exp(1j * x), np.exp(-x**2))
    output = tmp_path / "field.png"

    figure, _ = plot_field(field, quantity="intensity", save_path=output)

    assert output.is_file()
    plt.close(figure)
