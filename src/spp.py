import numpy as np

from scipy.constants import c


def spp_wavevector(omega, epsilon_m, epsilon_d):
    """
    Calculate the complex wavevector
    of a surface plasmon polariton.
    """

    k0 = omega / c

    k_spp = k0 * np.sqrt(
        (epsilon_m * epsilon_d)
        /
        (epsilon_m + epsilon_d)
    )

    return k_spp