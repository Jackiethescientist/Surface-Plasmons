"""Only the electrostatic formulas in the October 2026 validation notes.

Lengths: um. Reduced alpha = alpha_SI / epsilon_0 / 1e-18 (um^3).
Reduced moment q = p_SI / (epsilon_0 * E_ref * 1e-18), so G q = E/E_ref.
The transverse formula in the notes gives a magnitude; its vector sign is -.
"""
from __future__ import annotations

import numpy as np

FORMULAS = {
    "polarizability": "alpha0 = 4*pi*epsilon_m*a^3*(epsilon_p-epsilon_m)/(epsilon_p+2*epsilon_m)",
    "axial_field": "E_parallel = 2*q_parallel/(4*pi*epsilon_m*d^3)",
    "transverse_field": "E_perp_vector = -q_perp/(4*pi*epsilon_m*d^3)",
    "coupled_dipoles": "q_i = alpha0*(E_inc,i + sum_(j!=i) G0_ij*q_j)",
    "electrostatic_tensor": "G0(r) = (3*rhat*rhat^T-I)/(4*pi*epsilon_m*|r|^3)",
    "pair_modes": "alpha0^-1 = +C (symmetric), -C (antisymmetric)",
}


def polarizability(radius_um: float, epsilon_particle: complex, epsilon_host: complex) -> complex:
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("Sphere radius must be finite and positive.")
    ep, em = complex(epsilon_particle), complex(epsilon_host)
    if not np.isfinite(ep) or not np.isfinite(em) or em == 0:
        raise ValueError("Permittivities must be finite and the host nonzero.")
    if ep + 2 * em == 0:
        raise ValueError("Lossless polarizability pole lies exactly on this sample.")
    return 4 * np.pi * em * radius_um**3 * (ep - em) / (ep + 2 * em)


def green(displacement_um, epsilon_host: complex = 1.0) -> np.ndarray:
    r = np.asarray(displacement_um, dtype=float)
    d = np.linalg.norm(r)
    if r.shape != (3,) or not np.all(np.isfinite(r)) or d <= 0:
        raise ValueError("Dipole displacement must be a finite nonzero 3-vector.")
    n = r / d
    return (3 * np.outer(n, n) - np.eye(3)) / (4 * np.pi * epsilon_host * d**3)


def interaction_matrix(positions_um, epsilon_host: complex = 1.0) -> np.ndarray:
    positions = np.asarray(positions_um, dtype=float)
    count = len(positions)
    matrix = np.zeros((3 * count, 3 * count), dtype=complex)
    for i in range(count):
        for j in range(count):
            if i != j:
                matrix[3*i:3*i+3, 3*j:3*j+3] = green(positions[i] - positions[j], epsilon_host)
    return matrix


def solve(positions_um, alpha: complex, incident_fields, epsilon_host: complex = 1.0):
    incident = np.asarray(incident_fields, dtype=complex)
    matrix = interaction_matrix(positions_um, epsilon_host)
    system = np.eye(matrix.shape[0]) - alpha * matrix
    moments = np.linalg.solve(system, alpha * incident.reshape(-1)).reshape(-1, 3)
    residual = moments.reshape(-1) - alpha * (incident.reshape(-1) + matrix @ moments.reshape(-1))
    relative_residual = np.linalg.norm(residual) / max(np.linalg.norm(moments), 1e-30)
    return moments, float(relative_residual)


def field_matrix(points_um, positions_um, epsilon_host: complex = 1.0) -> np.ndarray:
    points, positions = np.asarray(points_um), np.asarray(positions_um)
    matrix = np.zeros((3 * len(points), 3 * len(positions)), dtype=complex)
    for i, point in enumerate(points):
        for j, position in enumerate(positions):
            matrix[3*i:3*i+3, 3*j:3*j+3] = green(point - position, epsilon_host)
    return matrix


def pair_couplings(distance_um: float, epsilon_host: complex = 1.0):
    scale = 1 / (4 * np.pi * epsilon_host * distance_um**3)
    return {"longitudinal": 2 * scale, "transverse": -scale}
