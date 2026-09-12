import numpy as np

from scipy.constants import c

from src.drude_model import drude_permittivity
from src.spp import spp_wavevector


wavelength = 632.8e-9

omega = 2 * np.pi * c / wavelength


epsilon_inf = 9.5
omega_p = 1.37e16
gamma = 1.05e14


epsilon_m = drude_permittivity(
    omega,
    epsilon_inf,
    omega_p,
    gamma
)


epsilon_d = 1.0


k_spp = spp_wavevector(
    omega,
    epsilon_m,
    epsilon_d
)


print("Metal permittivity:")
print(epsilon_m)

print()

print("SPP wavevector:")
print(k_spp)

print()

print("Real part:")
print(np.real(k_spp))

print("Imaginary part:")
print(np.imag(k_spp))

L_spp = 1 / (2 * np.imag(k_spp))

print()

print("Propagation length:")
print(L_spp, "m")

print("Propagation length:")
print(L_spp * 1e6, "μm")

lambda_spp = 2 * np.pi / np.real(k_spp)

print()

print("SPP wavelength:")
print(lambda_spp * 1e9, "nm")