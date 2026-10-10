"""Equation-2 coupled-dipole model for finite particle lines.

The implementation uses the paper's convention

    E_loc,i = E0 exp(i k.r_i) - sum_(j != i) A_ij . P_j,
    P_i = alpha_i E_loc,i.

Lengths are micrometres. Dipole moments are stored as ``P / epsilon_0`` and
therefore have units of electric field times um^3. With this convention the
sphere polarizability has units of um^3 and all epsilon-zero factors cancel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np


class TheoryError(ValueError):
    """Raised for invalid coupled-dipole inputs."""


@dataclass(frozen=True, slots=True)
class ProfileComparison:
    """Shape-only comparison retained for backwards compatibility."""

    sample_count: int
    normalised_rmse: float
    correlation: float | None
    maximum_index_simulation: int
    maximum_index_theory: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "sample_count": self.sample_count,
            "normalised_rmse": self.normalised_rmse,
            "correlation": self.correlation,
            "maximum_index_simulation": self.maximum_index_simulation,
            "maximum_index_theory": self.maximum_index_theory,
        }


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """Same-scale comparison of complex vector fields."""

    sample_count: int
    complex_nrmse: float
    amplitude_nrmse: float
    intensity_nrmse: float
    amplitude_correlation: float | None
    mean_phase_error_rad: float | None
    maximum_index_simulation: int
    maximum_index_theory: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "sample_count": self.sample_count,
            "complex_nrmse": self.complex_nrmse,
            "amplitude_nrmse": self.amplitude_nrmse,
            "intensity_nrmse": self.intensity_nrmse,
            "amplitude_correlation": self.amplitude_correlation,
            "mean_phase_error_rad": self.mean_phase_error_rad,
            "maximum_index_simulation": self.maximum_index_simulation,
            "maximum_index_theory": self.maximum_index_theory,
        }


@dataclass(frozen=True, slots=True)
class CoupledDipoleResult:
    """Solution of Equation 2 at all particle sites."""

    positions_um: np.ndarray
    incident_fields: np.ndarray
    local_fields: np.ndarray
    dipole_moments: np.ndarray
    interaction_matrix_A: np.ndarray
    system_matrix: np.ndarray
    wavelength_um: float
    equation: str = "E_loc,i = E0 exp(i k.r_i) - sum_(j!=i) A_ij.P_j"


def _coordinates(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    coordinates = np.asarray(value, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.shape[0] < 1:
        raise TheoryError(f"{name} must have shape (count, 3).")
    if not np.all(np.isfinite(coordinates)):
        raise TheoryError(f"{name} contains non-finite values.")
    return coordinates


def _unit_vector(value: Sequence[complex], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=complex).reshape(-1)
    if vector.size != 3:
        raise TheoryError(f"{name} must contain three components.")
    length = float(np.sqrt(np.sum(np.abs(vector) ** 2)))
    if not math.isfinite(length) or length <= 0:
        raise TheoryError(f"{name} must be non-zero and finite.")
    return vector / length


def component_vector(component: str) -> np.ndarray:
    """Return the Cartesian unit vector for Ex, Ey, or Ez."""

    lookup = {
        "ex": (1.0, 0.0, 0.0),
        "ey": (0.0, 1.0, 0.0),
        "ez": (0.0, 0.0, 1.0),
        "x": (1.0, 0.0, 0.0),
        "y": (0.0, 1.0, 0.0),
        "z": (0.0, 0.0, 1.0),
    }
    try:
        return np.asarray(lookup[str(component).lower()], dtype=complex)
    except KeyError as exc:
        raise TheoryError("component must be Ex, Ey, or Ez.") from exc


def sphere_polarizability(
    radius_um: float,
    epsilon_particle: complex,
    epsilon_background: complex,
    *,
    wavelength_um: float | None = None,
    radiation_correction: bool = True,
) -> complex:
    """Return ``P/epsilon_0 = alpha E_loc`` for a small sphere.

    The background-permittivity factor is included consistently with the
    Green tensor. A radiation-reaction correction is optional; the interaction
    itself remains the quasistatic Equation-2 baseline.
    """

    if radius_um <= 0:
        raise TheoryError("radius_um must be positive.")
    eps_p = complex(epsilon_particle)
    eps_b = complex(epsilon_background)
    denominator = eps_p + 2.0 * eps_b
    if abs(denominator) <= 1e-30:
        raise TheoryError("The quasistatic polarizability denominator is zero.")
    alpha0 = 4.0 * math.pi * eps_b * radius_um**3 * (
        (eps_p - eps_b) / denominator
    )
    if wavelength_um is None or not radiation_correction:
        return complex(alpha0)
    if wavelength_um <= 0:
        raise TheoryError("wavelength_um must be positive.")
    wavevector = 2.0 * math.pi * np.sqrt(eps_b) / wavelength_um
    return complex(
        alpha0 / (1.0 - 1j * wavevector**3 * alpha0 / (6.0 * math.pi * eps_b))
    )


def quasistatic_green_tensor(
    displacement_um: Sequence[float],
    epsilon_background: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Map reduced dipole moment ``P/epsilon_0`` to electric field."""

    displacement = np.asarray(displacement_um, dtype=float)
    distance = float(np.linalg.norm(displacement))
    if distance <= 0:
        raise TheoryError("Dipole separation must be positive.")
    direction = displacement / distance
    return (
        3.0 * np.outer(direction, direction) - np.eye(3)
    ) / (4.0 * math.pi * complex(epsilon_background) * distance**3)


def equation2_interaction_tensor_A(
    displacement_um: Sequence[float],
    epsilon_background: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Return the paper's ``A_ij`` tensor for the quasistatic baseline.

    The physical dipole field is ``G.P``. Because Equation 2 writes the
    interaction as ``-A.P``, this convention requires ``A = -G``.
    """

    return -quasistatic_green_tensor(displacement_um, epsilon_background)


def _image_transform(
    epsilon_background: complex,
    epsilon_substrate: complex,
) -> np.ndarray:
    beta = (complex(epsilon_background) - complex(epsilon_substrate)) / (
        complex(epsilon_background) + complex(epsilon_substrate)
    )
    return np.diag([beta, -beta, beta]).astype(complex)


def solve_coupled_dipoles(
    positions_um: Sequence[Sequence[float]],
    alpha: complex | Sequence[complex],
    *,
    wavelength_um: float,
    epsilon_background: complex = 1.0 + 0.0j,
    incident_direction: Sequence[float] = (0.0, -1.0, 0.0),
    incident_polarisation: Sequence[complex] = (1.0, 0.0, 0.0),
    incident_amplitude: complex = 1.0 + 0.0j,
    epsilon_substrate: complex | None = None,
    surface_y_um: float = 0.0,
) -> CoupledDipoleResult:
    """Solve the complex vector form of Equation 2.

    Direct particle interactions use exactly ``j != i``. When a substrate is
    supplied, electrostatic image-dipole blocks are added as an explicitly
    labelled approximation; the homogeneous direct-particle part still obeys
    Equation 2 exactly.
    """

    if wavelength_um <= 0:
        raise TheoryError("wavelength_um must be positive.")
    positions = _coordinates(positions_um, "positions_um")
    count = len(positions)
    alphas = np.asarray(alpha, dtype=complex)
    if alphas.ndim == 0:
        alphas = np.full(count, complex(alphas), dtype=complex)
    alphas = alphas.reshape(-1)
    if alphas.size != count or np.any(np.abs(alphas) <= 1e-30):
        raise TheoryError("alpha must be non-zero and contain one value per particle.")

    direction = _unit_vector(incident_direction, "incident_direction")
    if np.max(np.abs(direction.imag)) > 1e-14:
        raise TheoryError("incident_direction must be real.")
    direction_real = direction.real
    polarisation = _unit_vector(incident_polarisation, "incident_polarisation")
    if abs(np.vdot(direction, polarisation)) > 1e-10:
        raise TheoryError("Incident polarisation must be transverse to propagation.")
    wavevector = 2.0 * math.pi * np.sqrt(complex(epsilon_background)) / wavelength_um
    phase = np.exp(1j * wavevector * (positions @ direction_real))
    incident = complex(incident_amplitude) * phase[:, np.newaxis] * polarisation

    interaction_A = np.zeros((3 * count, 3 * count), dtype=complex)
    image_transform = (
        _image_transform(epsilon_background, epsilon_substrate)
        if epsilon_substrate is not None else None
    )
    for i in range(count):
        row = slice(3 * i, 3 * i + 3)
        for j in range(count):
            column = slice(3 * j, 3 * j + 3)
            if i != j:
                interaction_A[row, column] += equation2_interaction_tensor_A(
                    positions[i] - positions[j], epsilon_background
                )
            if image_transform is not None:
                image_position = positions[j].copy()
                image_position[1] = 2.0 * surface_y_um - image_position[1]
                displacement = positions[i] - image_position
                if np.linalg.norm(displacement) > 1e-15:
                    interaction_A[row, column] += (
                        equation2_interaction_tensor_A(displacement, epsilon_background)
                        @ image_transform
                    )

    alpha_matrix = np.zeros_like(interaction_A)
    for index, value in enumerate(alphas):
        block = slice(3 * index, 3 * index + 3)
        alpha_matrix[block, block] = value * np.eye(3)

    # P_i = alpha_i(E_inc,i - sum_j A_ij.P_j)
    # therefore (I + alpha A)P = alpha E_inc.
    system = np.eye(3 * count, dtype=complex) + alpha_matrix @ interaction_A
    driving = alpha_matrix @ incident.reshape(-1)
    try:
        dipoles = np.linalg.solve(system, driving).reshape(count, 3)
    except np.linalg.LinAlgError as exc:
        raise TheoryError("The coupled-dipole matrix is singular.") from exc
    local = dipoles / alphas[:, np.newaxis]
    return CoupledDipoleResult(
        positions_um=positions,
        incident_fields=incident,
        local_fields=local,
        dipole_moments=dipoles,
        interaction_matrix_A=interaction_A,
        system_matrix=system,
        wavelength_um=float(wavelength_um),
    )


def electric_field_from_dipoles(
    observation_points_um: Sequence[Sequence[float]],
    solution: CoupledDipoleResult,
    *,
    epsilon_background: complex = 1.0 + 0.0j,
    incident_direction: Sequence[float] = (0.0, -1.0, 0.0),
    incident_polarisation: Sequence[complex] = (1.0, 0.0, 0.0),
    incident_amplitude: complex = 1.0 + 0.0j,
    epsilon_substrate: complex | None = None,
    surface_y_um: float = 0.0,
) -> np.ndarray:
    """Evaluate Equation 2's solved dipoles at external observation points."""

    points = _coordinates(observation_points_um, "observation_points_um")
    direction = _unit_vector(incident_direction, "incident_direction")
    polarisation = _unit_vector(incident_polarisation, "incident_polarisation")
    wavevector = 2.0 * math.pi * np.sqrt(complex(epsilon_background)) / solution.wavelength_um
    phase = np.exp(1j * wavevector * (points @ direction.real))
    fields = complex(incident_amplitude) * phase[:, np.newaxis] * polarisation
    image_transform = (
        _image_transform(epsilon_background, epsilon_substrate)
        if epsilon_substrate is not None else None
    )
    for point_index, point in enumerate(points):
        for particle_index, position in enumerate(solution.positions_um):
            displacement = point - position
            if np.linalg.norm(displacement) <= 1e-15:
                raise TheoryError("Observation points cannot coincide with point dipoles.")
            tensor_A = equation2_interaction_tensor_A(displacement, epsilon_background)
            fields[point_index] -= tensor_A @ solution.dipole_moments[particle_index]
            if image_transform is not None:
                image_position = position.copy()
                image_position[1] = 2.0 * surface_y_um - image_position[1]
                image_A = equation2_interaction_tensor_A(
                    point - image_position, epsilon_background
                ) @ image_transform
                fields[point_index] -= image_A @ solution.dipole_moments[particle_index]
    return fields


def coupled_dipole_profile(
    positions_um: Sequence[Sequence[float]],
    alpha: complex,
    *,
    polarisation_axis: str = "x",
    incident_field: complex = 1.0 + 0.0j,
    epsilon_background: complex = 1.0 + 0.0j,
    epsilon_substrate: complex | None = None,
    surface_y_um: float = 0.0,
    wavelength_um: float = 1.0,
) -> np.ndarray:
    """Return one component of the vector Equation-2 local-field solution."""

    polarisation = component_vector(polarisation_axis)
    direction = np.asarray([0.0, -1.0, 0.0])
    if abs(np.vdot(direction, polarisation)) > 1e-12:
        direction = np.asarray([0.0, 0.0, 1.0])
    solution = solve_coupled_dipoles(
        positions_um,
        alpha,
        wavelength_um=wavelength_um,
        epsilon_background=epsilon_background,
        incident_direction=direction,
        incident_polarisation=polarisation,
        incident_amplitude=incident_field,
        epsilon_substrate=epsilon_substrate,
        surface_y_um=surface_y_um,
    )
    index = int(np.argmax(np.abs(polarisation)))
    return solution.local_fields[:, index]


def compare_profiles(
    simulation: Sequence[float], theory: Sequence[float]
) -> ProfileComparison:
    """Compare profile shapes after independent maximum normalization."""

    measured = np.asarray(simulation, dtype=float).reshape(-1)
    predicted = np.asarray(theory, dtype=float).reshape(-1)
    if measured.size != predicted.size or measured.size == 0:
        raise TheoryError("Profiles must be non-empty and have the same length.")
    finite = np.isfinite(measured) & np.isfinite(predicted)
    measured = measured[finite]
    predicted = predicted[finite]
    if measured.size == 0:
        raise TheoryError("Profiles contain no common finite samples.")
    measured_scale = float(np.max(np.abs(measured))) or 1.0
    predicted_scale = float(np.max(np.abs(predicted))) or 1.0
    measured_normalised = measured / measured_scale
    predicted_normalised = predicted / predicted_scale
    rmse = float(np.sqrt(np.mean((measured_normalised - predicted_normalised) ** 2)))
    if measured.size > 1 and np.std(measured_normalised) > 0 and np.std(predicted_normalised) > 0:
        correlation = float(np.corrcoef(measured_normalised, predicted_normalised)[0, 1])
    else:
        correlation = None
    return ProfileComparison(
        sample_count=int(measured.size),
        normalised_rmse=rmse,
        correlation=correlation,
        maximum_index_simulation=int(np.argmax(measured_normalised)),
        maximum_index_theory=int(np.argmax(predicted_normalised)),
    )


def compare_complex_fields(
    simulation: Sequence[complex] | np.ndarray,
    theory: Sequence[complex] | np.ndarray,
) -> FieldComparison:
    """Compare complex fields on their common physical scale without refitting."""

    measured_original = np.asarray(simulation, dtype=complex)
    predicted_original = np.asarray(theory, dtype=complex)
    if measured_original.shape != predicted_original.shape or measured_original.size == 0:
        raise TheoryError("Complex fields must be non-empty and have the same shape.")
    measured = measured_original.reshape(-1)
    predicted = predicted_original.reshape(-1)
    finite = np.isfinite(measured.real) & np.isfinite(measured.imag)
    finite &= np.isfinite(predicted.real) & np.isfinite(predicted.imag)
    measured = measured[finite]
    predicted = predicted[finite]
    if measured.size == 0:
        raise TheoryError("Complex fields contain no common finite samples.")
    complex_scale = float(np.sqrt(np.mean(np.abs(measured) ** 2))) or 1.0
    complex_nrmse = float(np.sqrt(np.mean(np.abs(measured - predicted) ** 2)) / complex_scale)
    measured_amplitude = np.abs(measured)
    predicted_amplitude = np.abs(predicted)
    amplitude_scale = float(np.sqrt(np.mean(measured_amplitude**2))) or 1.0
    amplitude_nrmse = float(
        np.sqrt(np.mean((measured_amplitude - predicted_amplitude) ** 2)) / amplitude_scale
    )
    measured_intensity = measured_amplitude**2
    predicted_intensity = predicted_amplitude**2
    intensity_scale = float(np.sqrt(np.mean(measured_intensity**2))) or 1.0
    intensity_nrmse = float(
        np.sqrt(np.mean((measured_intensity - predicted_intensity) ** 2)) / intensity_scale
    )
    if measured.size > 1 and np.std(measured_amplitude) > 0 and np.std(predicted_amplitude) > 0:
        correlation = float(np.corrcoef(measured_amplitude, predicted_amplitude)[0, 1])
    else:
        correlation = None
    phase_mask = (measured_amplitude > complex_scale * 1e-9) & (
        predicted_amplitude > complex_scale * 1e-9
    )
    mean_phase_error = (
        float(np.mean(np.abs(np.angle(predicted[phase_mask] * np.conjugate(measured[phase_mask])))))
        if np.any(phase_mask) else None
    )
    if measured_original.ndim >= 2 and measured_original.shape[-1] == 3:
        measured_point_intensity = np.sum(np.abs(measured_original) ** 2, axis=-1).reshape(-1)
        predicted_point_intensity = np.sum(np.abs(predicted_original) ** 2, axis=-1).reshape(-1)
    else:
        measured_point_intensity = measured_intensity
        predicted_point_intensity = predicted_intensity
    return FieldComparison(
        sample_count=int(measured.size),
        complex_nrmse=complex_nrmse,
        amplitude_nrmse=amplitude_nrmse,
        intensity_nrmse=intensity_nrmse,
        amplitude_correlation=correlation,
        mean_phase_error_rad=mean_phase_error,
        maximum_index_simulation=int(np.argmax(measured_point_intensity)),
        maximum_index_theory=int(np.argmax(predicted_point_intensity)),
    )


__all__ = [
    "CoupledDipoleResult",
    "FieldComparison",
    "ProfileComparison",
    "TheoryError",
    "compare_complex_fields",
    "compare_profiles",
    "component_vector",
    "coupled_dipole_profile",
    "electric_field_from_dipoles",
    "equation2_interaction_tensor_A",
    "quasistatic_green_tensor",
    "solve_coupled_dipoles",
    "sphere_polarizability",
]
