import math

import numpy as np
import pytest

from surface_plasmons.theory import (
    TheoryError,
    compare_complex_fields,
    compare_profiles,
    coupled_dipole_profile,
    dipole_cross_sections,
    electric_field_from_dipoles,
    equation2_interaction_tensor_A,
    mie_dipole_polarizability,
    mie_sphere_cross_sections,
    one_dimensional_dipole_sum,
    one_dimensional_interaction_A,
    paper_equation3_interaction_tensor_A,
    solve_coupled_dipoles,
    solve_one_dimensional_chain,
    sphere_polarizability,
    uniform_chain_dipole_moment,
)


def test_paper_equation_1_vacuum_polarizability() -> None:
    radius = 0.04
    epsilon = -10.0 + 1.0j

    alpha = sphere_polarizability(
        radius,
        epsilon,
        1.0,
        model="quasistatic",
    )
    expected = 4.0 * math.pi * radius**3 * (epsilon - 1.0) / (epsilon + 2.0)

    assert alpha == pytest.approx(expected)


def test_radiation_reaction_is_an_explicit_opt_in_model() -> None:
    baseline = sphere_polarizability(0.04, -10.0 + 1.0j, 1.0)
    corrected = sphere_polarizability(
        0.04,
        -10.0 + 1.0j,
        1.0,
        wavelength_um=0.633,
        model="radiation_reaction",
    )

    assert corrected != pytest.approx(baseline)
    with pytest.raises(TheoryError, match="wavelength_um"):
        sphere_polarizability(
            0.04,
            -10.0 + 1.0j,
            1.0,
            model="radiation_reaction",
        )


def test_paper_equation_3_matches_its_printed_vector_expression() -> None:
    displacement = np.asarray([0.17, -0.08, 0.11])
    dipole = np.asarray([0.4 + 0.2j, -0.3j, 0.7])
    wavelength = 0.633
    distance = float(np.linalg.norm(displacement))
    wavevector = 2.0 * math.pi / wavelength
    phase = np.exp(1j * wavevector * distance)
    expected = phase * (
        wavevector**2
        * np.cross(displacement, np.cross(displacement, dipole))
        / distance**3
        + (1.0 - 1j * wavevector * distance)
        * (
            distance**2 * dipole
            - 3.0 * displacement * np.dot(displacement, dipole)
        )
        / distance**5
    ) / (4.0 * math.pi)

    tensor = paper_equation3_interaction_tensor_A(
        displacement,
        wavelength_um=wavelength,
        epsilon_background=1.0,
    )

    assert tensor @ dipole == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_paper_equation_3_reduces_to_quasistatic_A_at_long_wavelength() -> None:
    displacement = (0.2, -0.1, 0.05)
    retarded = paper_equation3_interaction_tensor_A(
        displacement,
        wavelength_um=1.0e8,
        epsilon_background=1.0,
    )
    quasistatic = equation2_interaction_tensor_A(
        displacement,
        1.0,
        model="quasistatic",
    )

    assert retarded == pytest.approx(quasistatic, rel=1e-8, abs=1e-12)


def test_two_longitudinal_dipoles_match_independent_closed_form() -> None:
    separation = 0.2
    alpha = 1.0e-4 + 2.0e-5j
    solution = solve_coupled_dipoles(
        [(-separation / 2.0, 0.0, 0.0), (separation / 2.0, 0.0, 0.0)],
        alpha,
        wavelength_um=0.633,
        incident_direction=(0.0, -1.0, 0.0),
        incident_polarisation=(1.0, 0.0, 0.0),
        interaction_model="quasistatic",
    )
    longitudinal_A = -2.0 / (4.0 * math.pi * separation**3)
    expected_dipole = alpha / (1.0 + alpha * longitudinal_A)

    assert solution.dipole_moments[:, 0] == pytest.approx(
        [expected_dipole, expected_dipole],
        rel=1e-12,
        abs=1e-12,
    )
    assert solution.dipole_moments[:, 1:] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("interaction_model", ["quasistatic", "paper_retarded"])
def test_single_particle_matches_closed_form_with_incident_phase(
    interaction_model: str,
) -> None:
    wavelength = 0.633
    position = np.asarray([[0.08, 0.13, -0.04]])
    alpha = 1.3e-4 + 4.0e-5j
    incident_amplitude = 1.7 - 0.2j
    wavevector = 2.0 * math.pi / wavelength
    expected_incident_x = incident_amplitude * np.exp(
        -1j * wavevector * position[0, 1]
    )
    expected_dipole_x = alpha * expected_incident_x

    solution = solve_coupled_dipoles(
        position,
        alpha,
        wavelength_um=wavelength,
        incident_direction=(0.0, -1.0, 0.0),
        incident_polarisation=(1.0, 0.0, 0.0),
        incident_amplitude=incident_amplitude,
        interaction_model=interaction_model,
    )

    assert solution.interaction_matrix_A == pytest.approx(0.0, abs=1e-15)
    assert solution.local_fields[0, 0] == pytest.approx(
        expected_incident_x,
        rel=1e-12,
        abs=1e-12,
    )
    assert solution.dipole_moments[0, 0] == pytest.approx(
        expected_dipole_x,
        rel=1e-12,
        abs=1e-12,
    )
    assert solution.dipole_moments[0, 1:] == pytest.approx(0.0, abs=1e-14)


def test_two_retarded_longitudinal_dipoles_match_closed_form() -> None:
    separation = 0.2
    wavelength = 0.633
    alpha = 1.0e-4 + 2.0e-5j
    wavevector = 2.0 * math.pi / wavelength
    phase = np.exp(1j * wavevector * separation)
    # For r and P both along x, the radiation 1/r term in Equation 3 is zero.
    # The printed vector expression then reduces directly to this scalar A_xx.
    longitudinal_A = (
        -2.0
        * phase
        * (1.0 - 1j * wavevector * separation)
        / separation**3
        / (4.0 * math.pi)
    )
    expected_dipole = alpha / (1.0 + alpha * longitudinal_A)

    solution = solve_coupled_dipoles(
        [(-separation / 2.0, 0.0, 0.0), (separation / 2.0, 0.0, 0.0)],
        alpha,
        wavelength_um=wavelength,
        incident_direction=(0.0, -1.0, 0.0),
        incident_polarisation=(1.0, 0.0, 0.0),
        interaction_model="paper_retarded",
    )

    assert solution.dipole_moments[:, 0] == pytest.approx(
        [expected_dipole, expected_dipole],
        rel=1e-12,
        abs=1e-12,
    )
    assert solution.dipole_moments[:, 1:] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("interaction_model", ["quasistatic", "paper_retarded"])
@pytest.mark.parametrize(
    ("orientation", "polarisation", "component_index"),
    [
        ("longitudinal", (1.0, 0.0, 0.0), 0),
        ("transverse", (0.0, 0.0, 1.0), 2),
    ],
)
def test_direct_1d_solver_matches_vector_cda(
    interaction_model: str,
    orientation: str,
    polarisation: tuple[float, float, float],
    component_index: int,
) -> None:
    x = np.asarray([-0.3, -0.15, 0.02, 0.15, 0.3])
    positions = np.column_stack((x, np.full(x.size, 0.04), np.zeros(x.size)))
    alpha = 1.0e-4 + 2.0e-5j
    common_incident_phase = np.exp(-1j * 2.0 * math.pi * 0.04 / 0.633)
    scalar = solve_one_dimensional_chain(
        x,
        alpha,
        wavelength_um=0.633,
        orientation=orientation,
        incident_amplitude=common_incident_phase,
        interaction_model=interaction_model,
    )
    vector = solve_coupled_dipoles(
        positions,
        alpha,
        wavelength_um=0.633,
        incident_direction=(0.0, -1.0, 0.0),
        incident_polarisation=polarisation,
        interaction_model=interaction_model,
    )

    assert scalar.local_fields == pytest.approx(
        vector.local_fields[:, component_index], rel=1e-12, abs=1e-12
    )
    assert scalar.dipole_moments == pytest.approx(
        vector.dipole_moments[:, component_index], rel=1e-12, abs=1e-12
    )


def test_direct_1d_interaction_and_equation4_dipole_sum() -> None:
    spacing = 0.15
    wavelength = 0.633
    neighbour_count = 4
    direct_sum = -2.0 * sum(
        one_dimensional_interaction_A(
            order * spacing,
            wavelength_um=wavelength,
            orientation="longitudinal",
            model="paper_retarded",
        )
        for order in range(1, neighbour_count + 1)
    )
    dipole_sum = one_dimensional_dipole_sum(
        spacing,
        wavelength_um=wavelength,
        neighbour_count=neighbour_count,
        orientation="longitudinal",
        interaction_model="paper_retarded",
    )
    alpha = 1.0e-4 + 2.0e-5j

    assert dipole_sum == pytest.approx(direct_sum, rel=1e-13, abs=1e-13)
    assert uniform_chain_dipole_moment(alpha, dipole_sum) == pytest.approx(
        1.0 / (1.0 / alpha - direct_sum), rel=1e-13, abs=1e-13
    )


def test_mie_dipole_polarizability_reaches_quasistatic_small_sphere_limit() -> None:
    radius = 1.0e-3
    wavelength = 0.633
    epsilon = -10.0 + 1.0j
    exact_a1 = mie_dipole_polarizability(radius, wavelength, epsilon)
    quasistatic = sphere_polarizability(radius, epsilon, 1.0)

    assert exact_a1 == pytest.approx(quasistatic, rel=1e-3)


def test_exact_mie_lossless_sphere_conserves_energy() -> None:
    result = mie_sphere_cross_sections(0.04, 0.633, 2.25 + 0.0j)

    assert result.extinction_cross_section_um2 == pytest.approx(
        result.scattering_cross_section_um2, rel=1e-12, abs=1e-14
    )
    assert result.absorption_cross_section_um2 == pytest.approx(0.0, abs=1e-14)
    assert result.maximum_order >= 1


def test_mie_a1_polarizability_reproduces_electric_dipole_cross_sections() -> None:
    radius = 0.04
    wavelength = 0.633
    epsilon = -10.0 + 1.0j
    mie = mie_sphere_cross_sections(radius, wavelength, epsilon)
    alpha = mie_dipole_polarizability(radius, wavelength, epsilon)
    cross_sections = dipole_cross_sections(alpha, wavelength)
    wavevector = 2.0 * math.pi / wavelength
    expected_electric_dipole_extinction = (
        6.0 * math.pi * mie.electric_coefficients[0].real / wavevector**2
    )

    assert cross_sections["extinction_cross_section_um2"] == pytest.approx(
        expected_electric_dipole_extinction, rel=1e-12, abs=1e-14
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
        interaction_model="paper_retarded",
    )
    rhs = solution.incident_fields - (
        solution.interaction_matrix_A @ solution.dipole_moments.reshape(-1)
    ).reshape(solution.local_fields.shape)

    assert solution.local_fields == pytest.approx(rhs, rel=1e-12, abs=1e-12)
    assert np.angle(solution.incident_fields[0, 0]) != pytest.approx(
        np.angle(solution.incident_fields[1, 0])
    )


def test_equation_2_sign_convention_is_A_equals_negative_G() -> None:
    tensor_a = equation2_interaction_tensor_A(
        (0.2, 0.0, 0.0),
        1.0,
        model="quasistatic",
    )
    dipole = np.asarray([1.0, 0.0, 0.0])

    # A longitudinal dipole produces a positive-x field at a point to its right.
    assert (-tensor_a @ dipole)[0].real > 0


def test_external_field_and_same_scale_comparison() -> None:
    alpha = sphere_polarizability(0.04, -10 + 1j, 1.0, wavelength_um=0.633)
    solution = solve_coupled_dipoles(
        [(0.0, 0.04, 0.0)],
        alpha,
        wavelength_um=0.633,
        interaction_model="paper_retarded",
    )
    field = electric_field_from_dipoles([(0.0, 0.10, 0.0)], solution)
    report = compare_complex_fields(field, field)

    assert field.shape == (1, 3)
    assert np.all(np.isfinite(field))
    assert report.complex_nrmse == pytest.approx(0.0)
    assert report.amplitude_nrmse == pytest.approx(0.0)
    assert report.spatial_sample_count == 1
    assert report.component_sample_count == 3


def test_vector_amplitude_correlation_uses_total_field_per_point() -> None:
    measured = np.asarray([[10.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=complex)
    predicted = np.asarray([[1.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=complex)

    report = compare_complex_fields(measured, predicted)

    assert report.spatial_sample_count == 2
    assert report.component_sample_count == 6
    assert report.amplitude_correlation == pytest.approx(-1.0)
    assert report.maximum_index_simulation == 0
    assert report.maximum_index_theory == 1
    mie_dipole_polarizability,
    mie_sphere_cross_sections,
    one_dimensional_dipole_sum,
    one_dimensional_interaction_A,
