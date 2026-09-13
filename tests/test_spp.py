import numpy as np
from scipy.constants import c

from surface_plasmon_model import spp_wavevector


def test_spp_matches_dispersion_equation():
    omega = 3.0e15

    epsilon_m = -10.0 + 1.0j
    epsilon_d = 1.0

    calculated = spp_wavevector(
        omega,
        epsilon_m,
        epsilon_d,
    )

    k0 = omega / c

    expected = k0 * np.sqrt(
        (epsilon_m * epsilon_d)
        / (epsilon_m + epsilon_d)
    )

    np.testing.assert_allclose(
        calculated,
        expected,
        rtol=1e-12,
    )


def test_spp_wavevector_is_complex_for_lossy_metal():
    omega = 3.0e15

    k_spp = spp_wavevector(
        omega,
        epsilon_m=-10.0 + 1.0j,
        epsilon_d=1.0,
    )

    assert np.iscomplexobj(k_spp)


def test_spp_real_part_exceeds_free_space_wavevector():
    omega = 3.0e15

    k0 = omega / c

    k_spp = spp_wavevector(
        omega,
        epsilon_m=-10.0 + 0.1j,
        epsilon_d=1.0,
    )

    assert np.real(k_spp) > k0


def test_spp_supports_numpy_arrays():
    omega = np.array([
        2.0e15,
        3.0e15,
        4.0e15,
    ])

    epsilon_m = np.array([
        -20.0 + 0.5j,
        -10.0 + 0.8j,
        -5.0 + 1.0j,
    ])

    k_spp = spp_wavevector(
        omega,
        epsilon_m,
        epsilon_d=1.0,
    )

    assert k_spp.shape == omega.shape


from surface_plasmon_model import drude_permittivity


def test_drude_to_spp_pipeline():
    wavelength = 632.8e-9

    omega = 2 * np.pi * c / wavelength

    epsilon_m = drude_permittivity(
        omega,
        epsilon_inf=9.5,
        omega_p=1.37e16,
        gamma=1.05e14,
    )

    k_spp = spp_wavevector(
        omega,
        epsilon_m,
        epsilon_d=1.0,
    )

    assert np.isfinite(np.real(k_spp))
    assert np.isfinite(np.imag(k_spp))

    assert np.real(k_spp) > 0