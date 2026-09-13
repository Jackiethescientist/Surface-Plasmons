import numpy as np

from surface_plasmon_model import drude_permittivity


def test_drude_returns_complex_value():
    omega = 3.0e15

    epsilon = drude_permittivity(
        omega,
        epsilon_inf=9.5,
        omega_p=1.37e16,
        gamma=1.05e14,
    )

    assert np.iscomplexobj(epsilon)


def test_drude_matches_equation():
    omega = 3.0e15
    epsilon_inf = 9.5
    omega_p = 1.37e16
    gamma = 1.05e14

    calculated = drude_permittivity(
        omega,
        epsilon_inf,
        omega_p,
        gamma,
    )

    expected = (
        epsilon_inf
        - omega_p**2
        / (omega**2 + 1j * gamma * omega)
    )

    np.testing.assert_allclose(
        calculated,
        expected,
        rtol=1e-12,
    )


def test_drude_without_damping_is_real():
    omega = 3.0e15

    epsilon = drude_permittivity(
        omega,
        epsilon_inf=9.5,
        omega_p=1.37e16,
        gamma=0.0,
    )

    assert np.isclose(np.imag(epsilon), 0.0)


def test_drude_supports_numpy_arrays():
    omega = np.array([
        2.0e15,
        3.0e15,
        4.0e15,
    ])

    epsilon = drude_permittivity(
        omega,
        epsilon_inf=9.5,
        omega_p=1.37e16,
        gamma=1.05e14,
    )

    assert epsilon.shape == omega.shape