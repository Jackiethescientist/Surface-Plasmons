import numpy as np
import pytest

from surface_plasmons.analysis import (
    analyze_field,
    analyze_spectrum,
    calculate_fwhm,
    check_energy_balance,
    spp_effective_index,
    spp_wavevector,
)


def test_gaussian_spectrum_peak_and_width() -> None:
    wavelengths = np.linspace(0.45, 0.75, 1201)
    sigma = 0.020
    values = np.exp(-0.5 * ((wavelengths - 0.600) / sigma) ** 2)

    report = analyze_spectrum(
        wavelengths,
        values,
        monitor_name="extinction",
        prominence=0.05,
    )

    assert report.sample_count == wavelengths.size
    assert report.primary_peak is not None
    assert report.primary_peak.wavelength_um == pytest.approx(0.600, abs=5e-4)
    assert report.integrated_area > 0

    expected_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma
    assert calculate_fwhm(wavelengths, values) == pytest.approx(
        expected_fwhm,
        rel=0.02,
    )


def test_field_and_energy_balance_statistics() -> None:
    field = np.array([[1.0 + 0.0j, 2.0j], [0.5, -1.0j]])
    report = analyze_field(field, incident_amplitude=1.0)

    assert report.shape == (2, 2)
    assert report.maximum_enhancement == pytest.approx(2.0)

    balance = check_energy_balance(
        reflectance=[0.2, 0.3],
        transmittance=[0.7, 0.5],
        absorption=[0.1, 0.2],
    )
    assert balance.residual_abs_max == pytest.approx(0.0, abs=1e-12)
    assert balance.within_tolerance_fraction == 1.0


def test_analytical_spp_helpers_return_physical_branch() -> None:
    effective_index = spp_effective_index(-10.0 + 1.0j, 1.0)
    wavevector = spp_wavevector(0.633, -10.0 + 1.0j, 1.0)

    assert np.real(effective_index) > 1.0
    assert np.imag(effective_index) >= 0.0
    assert np.real(wavevector) > 0.0
    assert np.imag(wavevector) >= 0.0
