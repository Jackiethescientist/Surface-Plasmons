import numpy as np
import pytest

from surface_plasmons.theory import (
    compare_complex_fields,
    compare_profiles,
    coupled_dipole_profile,
    electric_field_from_dipoles,
    equation2_interaction_tensor_A,
    solve_coupled_dipoles,
    sphere_polarizability,
)


def test_coupled_dipole_perfect_line_is_mirror_symmetric() -> None:
    positions = [(-0.3, 0.04, 0.0), (-0.15, 0.04, 0.0), (0.0, 0.04, 0.0), (0.15, 0.04, 0.0), (0.3, 0.04, 0.0)]
    alpha = sphere_polarizability(0.04, -10 + 1j, 1.0, wavelength_um=0.633)
    field = coupled_dipole_profile(positions, alpha, polarisation_axis="x")

    assert np.abs(field) == pytest.approx(np.abs(field[::-1]))


def test_profile_comparison_identical_profiles() -> None:
    report = compare_profiles([1.0, 3.0, 2.0], [2.0, 6.0, 4.0])

    assert report.normalised_rmse == pytest.approx(0.0)
    assert report.correlation == pytest.approx(1.0)
    assert report.maximum_index_simulation == 1


def test_vector_solution_satisfies_equation_2_with_phase() -> None:
    positions = np.asarray([[-0.15, 0.04, 0.0], [0.15, 0.06, 0.0]])
    alpha = sphere_polarizability(0.04, -10 + 1j, 1.0, wavelength_um=0.633)
    solution = solve_coupled_dipoles(
        positions,
        alpha,
        wavelength_um=0.633,
        incident_direction=(0.0, -1.0, 0.0),
        incident_polarisation=(1.0, 0.0, 0.0),
    )
    rhs = solution.incident_fields - (
        solution.interaction_matrix_A @ solution.dipole_moments.reshape(-1)
    ).reshape(solution.local_fields.shape)

    assert solution.local_fields == pytest.approx(rhs, rel=1e-12, abs=1e-12)
    assert np.angle(solution.incident_fields[0, 0]) != pytest.approx(
        np.angle(solution.incident_fields[1, 0])
    )


def test_equation_2_sign_convention_is_A_equals_negative_G() -> None:
    tensor_a = equation2_interaction_tensor_A((0.2, 0.0, 0.0), 1.0)
    dipole = np.asarray([1.0, 0.0, 0.0])

    # A longitudinal dipole produces a positive-x field at a point to its right.
    assert (-tensor_a @ dipole)[0].real > 0


def test_external_field_and_same_scale_comparison() -> None:
    alpha = sphere_polarizability(0.04, -10 + 1j, 1.0, wavelength_um=0.633)
    solution = solve_coupled_dipoles(
        [(0.0, 0.04, 0.0)],
        alpha,
        wavelength_um=0.633,
    )
    field = electric_field_from_dipoles([(0.0, 0.10, 0.0)], solution)
    report = compare_complex_fields(field, field)

    assert field.shape == (1, 3)
    assert np.all(np.isfinite(field))
    assert report.complex_nrmse == pytest.approx(0.0)
    assert report.amplitude_nrmse == pytest.approx(0.0)
