import numpy as np


def drude_permittivity(omega, epsilon_inf, omega_p, gamma):
    """
    Calculate the complex relative permittivity
    of a metal using the Drude model.

    Parameters
    ----------
    omega : float or ndarray
        Angular frequency in rad/s.

    epsilon_inf : float
        High-frequency permittivity.

    omega_p : float
        Plasma frequency in rad/s.

    gamma : float
        Damping frequency in rad/s.

    Returns
    -------
    complex or ndarray
        Complex relative permittivity.
    """

    epsilon = (
        epsilon_inf
        - omega_p**2 /
        (omega**2 + 1j * gamma * omega)
    )

    return epsilon