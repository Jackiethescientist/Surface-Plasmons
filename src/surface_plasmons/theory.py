"""Coupled-dipole models for finite particle lines.

The paper's Section 2 convention is

    E_loc,i = E0 exp(i k.r_i) - sum_(j != i) A_ij . P_j,       (2)
    P_i = alpha_i E_loc,i.

Its Equation 3 is a *retarded* dipole interaction.  This module therefore
keeps two interaction models separate:

``quasistatic``
    The strict electrostatic ``1/r^3`` limit.
``paper_retarded``
    The complete ``1/r^3``, ``1/r^2`` and ``1/r`` interaction printed in the
    paper's Equation 3.

The paper writes its vacuum sphere polarizability as
``alpha = 4*pi*a^3*(epsilon-1)/(epsilon+2)``.  Here the stored moment is the SI
moment divided by ``epsilon_0``.  Consequently ``alpha`` has units of um^3 and
the host-medium conversion is included explicitly in both alpha and A.  For a
vacuum background this reduces exactly to the paper's equations.

The paper does not state a phasor time convention explicitly.  Its outgoing
``exp(+i*k*r)`` Green function is consistent with suppressed ``exp(-i*omega*t)``
time dependence; that convention is used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence
import math

import numpy as np


class TheoryError(ValueError):
    """Raised for invalid coupled-dipole inputs."""


InteractionModel = Literal["quasistatic", "paper_retarded"]
PolarizabilityModel = Literal[
    "quasistatic",
    "radiation_reaction",
    "mie_electric_dipole",
]
OneDimensionalOrientation = Literal["longitudinal", "transverse"]

INTERACTION_MODELS = frozenset({"quasistatic", "paper_retarded"})
POLARIZABILITY_MODELS = frozenset(
    {"quasistatic", "radiation_reaction", "mie_electric_dipole"}
)


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
    spatial_sample_count: int
    component_sample_count: int
    complex_nrmse: float
    amplitude_nrmse: float
    amplitude_correlation: float | None
    mean_phase_error_rad: float | None
    maximum_index_simulation: int
    maximum_index_theory: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "sample_count": self.sample_count,
            "spatial_sample_count": self.spatial_sample_count,
            "component_sample_count": self.component_sample_count,
            "complex_nrmse": self.complex_nrmse,
            "amplitude_nrmse": self.amplitude_nrmse,
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
    interaction_model: str = "quasistatic"
    equation: str = "E_loc,i = E0 exp(i k.r_i) - sum_(j!=i) A_ij.P_j"


@dataclass(frozen=True, slots=True)
class OneDimensionalResult:
    """Scalar finite-chain solution obtained directly from the 1-D formulas."""

    positions_um: np.ndarray
    incident_fields: np.ndarray
    local_fields: np.ndarray
    dipole_moments: np.ndarray
    interaction_matrix_A: np.ndarray
    system_matrix: np.ndarray
    wavelength_um: float
    orientation: str
    interaction_model: str


@dataclass(frozen=True, slots=True)
class MieSphereResult:
    """Exact single-sphere Mie coefficients and cross-sections."""

    radius_um: float
    wavelength_um: float
    size_parameter: float
    maximum_order: int
    electric_coefficients: np.ndarray
    magnetic_coefficients: np.ndarray
    extinction_cross_section_um2: float
    scattering_cross_section_um2: float
    absorption_cross_section_um2: float
    extinction_efficiency: float
    scattering_efficiency: float
    absorption_efficiency: float


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
    model: PolarizabilityModel = "quasistatic",
    radiation_correction: bool | None = None,
) -> complex:
    """Return ``P/epsilon_0 = alpha E_loc`` for a sphere dipole.

    ``model="quasistatic"`` is the host-medium form of the paper's Equation 1
    and is the default baseline. ``model="radiation_reaction"`` applies only
    the radiation-reaction correction. ``model="mie_electric_dipole"`` obtains
    the electric-dipole polarizability from the exact Mie coefficient ``a1``;
    it includes finite-size dynamic depolarisation and radiation damping while
    still restricting an array calculation to electric dipoles.

    ``radiation_correction`` remains as a compatibility keyword. New code
    should use the named ``model`` so the approximation level is explicit.
    """

    if radius_um <= 0:
        raise TheoryError("radius_um must be positive.")
    eps_p = complex(epsilon_particle)
    eps_b = complex(epsilon_background)
    if model not in POLARIZABILITY_MODELS:
        raise TheoryError(
            f"Unknown polarizability model {model!r}; choose one of "
            f"{sorted(POLARIZABILITY_MODELS)}."
        )
    if radiation_correction is not None:
        compatibility_model: PolarizabilityModel = (
            "radiation_reaction" if radiation_correction else "quasistatic"
        )
        if model != "quasistatic" and model != compatibility_model:
            raise TheoryError("Conflicting polarizability model options were supplied.")
        model = compatibility_model
    denominator = eps_p + 2.0 * eps_b
    if abs(denominator) <= 1e-30:
        raise TheoryError("The quasistatic polarizability denominator is zero.")
    alpha0 = 4.0 * math.pi * eps_b * radius_um**3 * (
        (eps_p - eps_b) / denominator
    )
    if model == "quasistatic":
        return complex(alpha0)
    if wavelength_um is None or wavelength_um <= 0:
        raise TheoryError(
            f"wavelength_um must be positive for the {model} model."
        )
    if model == "mie_electric_dipole":
        return mie_dipole_polarizability(
            radius_um,
            wavelength_um,
            eps_p,
            eps_b,
        )
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


def paper_equation3_interaction_tensor_A(
    displacement_um: Sequence[float],
    *,
    wavelength_um: float,
    epsilon_background: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Return the retarded ``A_ij`` printed in the paper's Equation 3.

    ``wavelength_um`` is the vacuum wavelength. The paper writes
    ``k = 2*pi/lambda``; for a homogeneous host this implementation uses
    ``k = 2*pi*sqrt(epsilon_background)/wavelength_vacuum``. The prefactor
    converts the paper's vacuum volume-polarizability convention to the
    reduced-SI moment ``P/epsilon_0`` used by this project.

    The tensor is invariant when the displacement direction is reversed, so
    it is compatible with the paper's definition of ``r_ij`` from i to j and
    this solver's source-to-observation displacement.
    """

    if wavelength_um <= 0:
        raise TheoryError("wavelength_um must be positive.")
    displacement = np.asarray(displacement_um, dtype=float)
    distance = float(np.linalg.norm(displacement))
    if distance <= 0:
        raise TheoryError("Dipole separation must be positive.")
    eps_b = complex(epsilon_background)
    direction = displacement / distance
    direction_dyad = np.outer(direction, direction)
    identity = np.eye(3)
    wavevector = 2.0 * math.pi * np.sqrt(eps_b) / wavelength_um
    phase = np.exp(1j * wavevector * distance)
    near_field = (1.0 - 1j * wavevector * distance) * (
        identity - 3.0 * direction_dyad
    ) / distance**3
    radiation_field = wavevector**2 * (direction_dyad - identity) / distance
    return phase * (near_field + radiation_field) / (4.0 * math.pi * eps_b)


def equation2_interaction_tensor_A(
    displacement_um: Sequence[float],
    epsilon_background: complex = 1.0 + 0.0j,
    *,
    wavelength_um: float | None = None,
    model: InteractionModel = "quasistatic",
) -> np.ndarray:
    """Return ``A_ij`` for an explicitly selected Equation-2 model.

    The physical dipole field is ``G.P``. Because Equation 2 writes the
    interaction as ``-A.P``, both supported conventions use ``A = -G``.
    """

    if model == "quasistatic":
        return -quasistatic_green_tensor(displacement_um, epsilon_background)
    if model == "paper_retarded":
        if wavelength_um is None:
            raise TheoryError(
                "wavelength_um is required for the paper_retarded interaction."
            )
        return paper_equation3_interaction_tensor_A(
            displacement_um,
            wavelength_um=wavelength_um,
            epsilon_background=epsilon_background,
        )
    raise TheoryError(
        f"Unknown interaction model {model!r}; choose one of "
        f"{sorted(INTERACTION_MODELS)}."
    )


def one_dimensional_interaction_A(
    separation_um: float,
    *,
    wavelength_um: float,
    epsilon_background: complex = 1.0 + 0.0j,
    orientation: OneDimensionalOrientation = "longitudinal",
    model: InteractionModel = "paper_retarded",
) -> complex:
    """Return the direct scalar Equation-3 interaction for a 1-D chain.

    The chain lies along ``x``. ``longitudinal`` means the dipole is parallel
    to the chain and ``transverse`` means either perpendicular direction. For
    distance ``r`` the formulas are

    ``A_L = -2 exp(ikr) (1-ikr)/(4*pi*eps_b*r^3)`` and

    ``A_T = exp(ikr) [(1-ikr)/r^3-k^2/r]/(4*pi*eps_b)``.

    These are Equation 3 reduced analytically before constructing a matrix.
    """

    distance = abs(float(separation_um))
    if not math.isfinite(distance) or distance <= 0:
        raise TheoryError("separation_um must be finite and non-zero.")
    if wavelength_um <= 0:
        raise TheoryError("wavelength_um must be positive.")
    if orientation not in {"longitudinal", "transverse"}:
        raise TheoryError("orientation must be longitudinal or transverse.")
    if model not in INTERACTION_MODELS:
        raise TheoryError(
            f"Unknown interaction model {model!r}; choose one of "
            f"{sorted(INTERACTION_MODELS)}."
        )
    eps_b = complex(epsilon_background)
    prefactor = 1.0 / (4.0 * math.pi * eps_b)
    if model == "quasistatic":
        coefficient = -2.0 if orientation == "longitudinal" else 1.0
        return complex(prefactor * coefficient / distance**3)
    wavevector = 2.0 * math.pi * np.sqrt(eps_b) / wavelength_um
    phase = np.exp(1j * wavevector * distance)
    if orientation == "longitudinal":
        return complex(
            prefactor
            * phase
            * (-2.0)
            * (1.0 - 1j * wavevector * distance)
            / distance**3
        )
    return complex(
        prefactor
        * phase
        * (
            (1.0 - 1j * wavevector * distance) / distance**3
            - wavevector**2 / distance
        )
    )


def solve_one_dimensional_chain(
    positions_um: Sequence[float],
    alpha: complex | Sequence[complex],
    *,
    wavelength_um: float,
    epsilon_background: complex = 1.0 + 0.0j,
    orientation: OneDimensionalOrientation = "longitudinal",
    incident_direction_cosine: float = 0.0,
    incident_amplitude: complex = 1.0 + 0.0j,
    interaction_model: InteractionModel = "paper_retarded",
) -> OneDimensionalResult:
    """Solve a finite, possibly displaced or vacancy-containing 1-D chain.

    Only the active particle coordinates are supplied, so a missing particle
    is represented by omitting its nominal position. The scalar system is the
    direct 1-D reduction ``(I + alpha*A)P = alpha*E_inc`` of Equation 2.
    """

    positions = np.asarray(positions_um, dtype=float).reshape(-1)
    if positions.size < 1 or not np.all(np.isfinite(positions)):
        raise TheoryError("positions_um must contain finite coordinates.")
    if np.unique(positions).size != positions.size:
        raise TheoryError("positions_um cannot contain duplicate coordinates.")
    if not math.isfinite(float(incident_direction_cosine)) or abs(
        float(incident_direction_cosine)
    ) > 1.0:
        raise TheoryError("incident_direction_cosine must be between -1 and 1.")
    alphas = np.asarray(alpha, dtype=complex)
    if alphas.ndim == 0:
        alphas = np.full(positions.size, complex(alphas), dtype=complex)
    alphas = alphas.reshape(-1)
    if alphas.size != positions.size or np.any(np.abs(alphas) <= 1e-30):
        raise TheoryError("alpha must be non-zero and contain one value per particle.")
    wavevector = 2.0 * math.pi * np.sqrt(complex(epsilon_background)) / wavelength_um
    incident = complex(incident_amplitude) * np.exp(
        1j * wavevector * positions * float(incident_direction_cosine)
    )
    interaction_A = np.zeros((positions.size, positions.size), dtype=complex)
    for i in range(positions.size):
        for j in range(positions.size):
            if i != j:
                interaction_A[i, j] = one_dimensional_interaction_A(
                    positions[i] - positions[j],
                    wavelength_um=wavelength_um,
                    epsilon_background=epsilon_background,
                    orientation=orientation,
                    model=interaction_model,
                )
    alpha_matrix = np.diag(alphas)
    system = np.eye(positions.size, dtype=complex) + alpha_matrix @ interaction_A
    try:
        dipoles = np.linalg.solve(system, alpha_matrix @ incident)
    except np.linalg.LinAlgError as exc:
        raise TheoryError("The one-dimensional coupled-dipole matrix is singular.") from exc
    return OneDimensionalResult(
        positions_um=positions,
        incident_fields=incident,
        local_fields=dipoles / alphas,
        dipole_moments=dipoles,
        interaction_matrix_A=interaction_A,
        system_matrix=system,
        wavelength_um=float(wavelength_um),
        orientation=orientation,
        interaction_model=interaction_model,
    )


def one_dimensional_dipole_sum(
    spacing_um: float,
    *,
    wavelength_um: float,
    neighbour_count: int,
    epsilon_background: complex = 1.0 + 0.0j,
    orientation: OneDimensionalOrientation = "longitudinal",
    interaction_model: InteractionModel = "paper_retarded",
) -> complex:
    """Return the symmetric truncated 1-D dipole sum ``S`` from Equation 5.

    ``neighbour_count`` sites on each side of the reference particle are
    included. Since the paper writes Equation 2 with ``-A.P``, the physical
    dipole sum used in Equation 4 is ``S = -sum(A)``.
    """

    if spacing_um <= 0:
        raise TheoryError("spacing_um must be positive.")
    if isinstance(neighbour_count, bool) or int(neighbour_count) != neighbour_count:
        raise TheoryError("neighbour_count must be a positive integer.")
    neighbour_count = int(neighbour_count)
    if neighbour_count < 1:
        raise TheoryError("neighbour_count must be a positive integer.")
    return complex(
        -2.0
        * sum(
            one_dimensional_interaction_A(
                order * spacing_um,
                wavelength_um=wavelength_um,
                epsilon_background=epsilon_background,
                orientation=orientation,
                model=interaction_model,
            )
            for order in range(1, neighbour_count + 1)
        )
    )


def uniform_chain_dipole_moment(
    alpha: complex,
    dipole_sum: complex,
    *,
    incident_amplitude: complex = 1.0 + 0.0j,
) -> complex:
    """Evaluate the paper's Equation 4, ``P = E0/(1/alpha - S)``."""

    alpha_value = complex(alpha)
    if abs(alpha_value) <= 1e-30:
        raise TheoryError("alpha must be non-zero.")
    denominator = 1.0 / alpha_value - complex(dipole_sum)
    if abs(denominator) <= 1e-30:
        raise TheoryError("The uniform-chain Equation-4 denominator is zero.")
    return complex(incident_amplitude) / denominator


def mie_sphere_cross_sections(
    radius_um: float,
    wavelength_um: float,
    epsilon_particle: complex,
    epsilon_background: complex = 1.0 + 0.0j,
    *,
    maximum_order: int | None = None,
) -> MieSphereResult:
    """Return exact Mie cross-sections for one non-magnetic sphere.

    The host must be lossless because cross-sections in an absorbing host need
    additional conventions. All electric and magnetic multipoles through the
    selected order are retained. The automatic order follows the standard
    ``ceil(x + 4*x^(1/3) + 2)`` truncation estimate.
    """

    if radius_um <= 0 or wavelength_um <= 0:
        raise TheoryError("radius_um and wavelength_um must be positive.")
    eps_b_complex = complex(epsilon_background)
    if abs(eps_b_complex.imag) > 1e-12 or eps_b_complex.real <= 0:
        raise TheoryError("Mie cross-sections require a positive, lossless host permittivity.")
    eps_b = float(eps_b_complex.real)
    refractive_index_background = math.sqrt(eps_b)
    wavevector = 2.0 * math.pi * refractive_index_background / wavelength_um
    size_parameter = wavevector * radius_um
    if maximum_order is None:
        maximum_order = max(
            1,
            int(math.ceil(size_parameter + 4.0 * size_parameter ** (1.0 / 3.0) + 2.0)),
        )
    if isinstance(maximum_order, bool) or int(maximum_order) != maximum_order:
        raise TheoryError("maximum_order must be a positive integer.")
    maximum_order = int(maximum_order)
    if maximum_order < 1:
        raise TheoryError("maximum_order must be a positive integer.")
    try:
        from scipy.special import spherical_jn, spherical_yn
    except (ImportError, ModuleNotFoundError) as exc:
        raise TheoryError("SciPy is required for exact Mie coefficients.") from exc

    orders = np.arange(1, maximum_order + 1)
    relative_index = np.sqrt(complex(epsilon_particle) / eps_b)
    internal_argument = relative_index * size_parameter

    j_external = spherical_jn(orders, size_parameter)
    j_external_prime = spherical_jn(orders, size_parameter, derivative=True)
    y_external = spherical_yn(orders, size_parameter)
    y_external_prime = spherical_yn(orders, size_parameter, derivative=True)
    j_internal = spherical_jn(orders, internal_argument)
    j_internal_prime = spherical_jn(orders, internal_argument, derivative=True)

    psi_external = size_parameter * j_external
    psi_external_prime = j_external + size_parameter * j_external_prime
    psi_internal = internal_argument * j_internal
    psi_internal_prime = j_internal + internal_argument * j_internal_prime
    xi_external = size_parameter * (j_external + 1j * y_external)
    xi_external_prime = (
        j_external
        + 1j * y_external
        + size_parameter * (j_external_prime + 1j * y_external_prime)
    )

    electric = (
        relative_index * psi_internal * psi_external_prime
        - psi_external * psi_internal_prime
    ) / (
        relative_index * psi_internal * xi_external_prime
        - xi_external * psi_internal_prime
    )
    magnetic = (
        psi_internal * psi_external_prime
        - relative_index * psi_external * psi_internal_prime
    ) / (
        psi_internal * xi_external_prime
        - relative_index * xi_external * psi_internal_prime
    )
    weights = 2.0 * orders + 1.0
    common = 2.0 * math.pi / wavevector**2
    extinction = float(common * np.sum(weights * np.real(electric + magnetic)))
    scattering = float(
        common * np.sum(weights * (np.abs(electric) ** 2 + np.abs(magnetic) ** 2))
    )
    absorption = extinction - scattering
    numerical_scale = max(abs(extinction), abs(scattering), 1.0)
    if abs(absorption) <= 1e-13 * numerical_scale:
        absorption = 0.0
    area = math.pi * radius_um**2
    return MieSphereResult(
        radius_um=float(radius_um),
        wavelength_um=float(wavelength_um),
        size_parameter=float(size_parameter),
        maximum_order=maximum_order,
        electric_coefficients=np.asarray(electric, dtype=complex),
        magnetic_coefficients=np.asarray(magnetic, dtype=complex),
        extinction_cross_section_um2=extinction,
        scattering_cross_section_um2=scattering,
        absorption_cross_section_um2=float(absorption),
        extinction_efficiency=extinction / area,
        scattering_efficiency=scattering / area,
        absorption_efficiency=float(absorption) / area,
    )


def mie_dipole_polarizability(
    radius_um: float,
    wavelength_um: float,
    epsilon_particle: complex,
    epsilon_background: complex = 1.0 + 0.0j,
) -> complex:
    """Convert the exact electric Mie coefficient ``a1`` to ``P/epsilon_0E``."""

    result = mie_sphere_cross_sections(
        radius_um,
        wavelength_um,
        epsilon_particle,
        epsilon_background,
    )
    eps_b = float(complex(epsilon_background).real)
    wavevector = 2.0 * math.pi * math.sqrt(eps_b) / wavelength_um
    return complex(6.0 * math.pi * 1j * eps_b * result.electric_coefficients[0] / wavevector**3)


def dipole_cross_sections(
    alpha: complex,
    wavelength_um: float,
    epsilon_background: complex = 1.0 + 0.0j,
) -> dict[str, float]:
    """Return electric-dipole extinction, scattering and absorption in um^2."""

    eps_b_complex = complex(epsilon_background)
    if abs(eps_b_complex.imag) > 1e-12 or eps_b_complex.real <= 0:
        raise TheoryError("Dipole cross-sections require a positive, lossless host permittivity.")
    if wavelength_um <= 0:
        raise TheoryError("wavelength_um must be positive.")
    eps_b = float(eps_b_complex.real)
    wavevector = 2.0 * math.pi * math.sqrt(eps_b) / wavelength_um
    alpha_value = complex(alpha)
    extinction = float(wavevector * alpha_value.imag / eps_b)
    scattering = float(wavevector**4 * abs(alpha_value) ** 2 / (6.0 * math.pi * eps_b**2))
    return {
        "extinction_cross_section_um2": extinction,
        "scattering_cross_section_um2": scattering,
        "absorption_cross_section_um2": extinction - scattering,
    }


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
    interaction_model: InteractionModel = "quasistatic",
    epsilon_substrate: complex | None = None,
    surface_y_um: float = 0.0,
) -> CoupledDipoleResult:
    """Solve the complex vector form of Equation 2.

    Direct particle interactions use exactly ``j != i``. The caller must
    explicitly select the strict quasistatic limit or the paper's retarded
    Equation-3 tensor. When a substrate is supplied, electrostatic image-dipole
    blocks are added as a separately labelled approximation.
    """

    if wavelength_um <= 0:
        raise TheoryError("wavelength_um must be positive.")
    if interaction_model not in INTERACTION_MODELS:
        raise TheoryError(
            f"Unknown interaction model {interaction_model!r}; choose one of "
            f"{sorted(INTERACTION_MODELS)}."
        )
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
                    positions[i] - positions[j],
                    epsilon_background,
                    wavelength_um=wavelength_um,
                    model=interaction_model,
                )
            if image_transform is not None:
                image_position = positions[j].copy()
                image_position[1] = 2.0 * surface_y_um - image_position[1]
                displacement = positions[i] - image_position
                if np.linalg.norm(displacement) > 1e-15:
                    interaction_A[row, column] += (
                        equation2_interaction_tensor_A(
                            displacement,
                            epsilon_background,
                            model="quasistatic",
                        )
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
        interaction_model=interaction_model,
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
            tensor_A = equation2_interaction_tensor_A(
                displacement,
                epsilon_background,
                wavelength_um=solution.wavelength_um,
                model=solution.interaction_model,
            )
            fields[point_index] -= tensor_A @ solution.dipole_moments[particle_index]
            if image_transform is not None:
                image_position = position.copy()
                image_position[1] = 2.0 * surface_y_um - image_position[1]
                image_A = equation2_interaction_tensor_A(
                    point - image_position,
                    epsilon_background,
                    model="quasistatic",
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
    interaction_model: InteractionModel = "quasistatic",
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
        interaction_model=interaction_model,
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
    component_amplitude = np.abs(measured)
    predicted_component_amplitude = np.abs(predicted)
    if measured_original.ndim >= 2 and measured_original.shape[-1] == 3:
        measured_amplitude = np.linalg.norm(measured_original, axis=-1).reshape(-1)
        predicted_amplitude = np.linalg.norm(predicted_original, axis=-1).reshape(-1)
        amplitude_finite = np.isfinite(measured_amplitude) & np.isfinite(
            predicted_amplitude
        )
        measured_amplitude = measured_amplitude[amplitude_finite]
        predicted_amplitude = predicted_amplitude[amplitude_finite]
        spatial_sample_count = int(measured_amplitude.size)
    else:
        measured_amplitude = component_amplitude
        predicted_amplitude = predicted_component_amplitude
        spatial_sample_count = int(measured_amplitude.size)
    amplitude_scale = float(np.sqrt(np.mean(measured_amplitude**2))) or 1.0
    amplitude_nrmse = float(
        np.sqrt(np.mean((measured_amplitude - predicted_amplitude) ** 2)) / amplitude_scale
    )
    if measured.size > 1 and np.std(measured_amplitude) > 0 and np.std(predicted_amplitude) > 0:
        correlation = float(np.corrcoef(measured_amplitude, predicted_amplitude)[0, 1])
    else:
        correlation = None
    phase_mask = (component_amplitude > complex_scale * 1e-9) & (
        predicted_component_amplitude > complex_scale * 1e-9
    )
    mean_phase_error = (
        float(np.mean(np.abs(np.angle(predicted[phase_mask] * np.conjugate(measured[phase_mask])))))
        if np.any(phase_mask) else None
    )
    measured_point_intensity = measured_amplitude**2
    predicted_point_intensity = predicted_amplitude**2
    return FieldComparison(
        sample_count=int(measured.size),
        spatial_sample_count=spatial_sample_count,
        component_sample_count=int(measured.size),
        complex_nrmse=complex_nrmse,
        amplitude_nrmse=amplitude_nrmse,
        amplitude_correlation=correlation,
        mean_phase_error_rad=mean_phase_error,
        maximum_index_simulation=int(np.argmax(measured_point_intensity)),
        maximum_index_theory=int(np.argmax(predicted_point_intensity)),
    )


__all__ = [
    "CoupledDipoleResult",
    "FieldComparison",
    "INTERACTION_MODELS",
    "InteractionModel",
    "MieSphereResult",
    "OneDimensionalOrientation",
    "OneDimensionalResult",
    "POLARIZABILITY_MODELS",
    "PolarizabilityModel",
    "ProfileComparison",
    "TheoryError",
    "compare_complex_fields",
    "compare_profiles",
    "component_vector",
    "coupled_dipole_profile",
    "dipole_cross_sections",
    "electric_field_from_dipoles",
    "equation2_interaction_tensor_A",
    "mie_dipole_polarizability",
    "mie_sphere_cross_sections",
    "one_dimensional_dipole_sum",
    "one_dimensional_interaction_A",
    "paper_equation3_interaction_tensor_A",
    "quasistatic_green_tensor",
    "solve_coupled_dipoles",
    "solve_one_dimensional_chain",
    "sphere_polarizability",
    "uniform_chain_dipole_moment",
]
