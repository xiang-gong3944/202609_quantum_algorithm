"""Error bounds for an open, uniform Heisenberg chain.

The Hamiltonian convention is

    H = J * sum_j (X_j X_{j+1} + Y_j Y_{j+1} + Z_j Z_{j+1}) = A + B,

where ``A`` contains bonds starting at even zero-based sites and ``B`` the
remaining bonds.  ``S_2(h) = exp(-ihA/2) exp(-ihB) exp(-ihA/2)`` and ``S_4``
is Suzuki's five-fold recursive composition.

There are deliberately three distinct error models in this module:

* ``naive_l1_coefficient`` is a finite-r norm bound obtained without using
  commutativity.
* ``commutator_coefficient`` is a finite-r bound based on nested commutators.
  Sparse Pauli coefficient L1 norms upper-bound the required spectral norms.
* ``effective_error_pauli`` constructs the leading BCH error operator used by
  Maxwell et al. (2026).  Its ground-state expectation is an asymptotic,
  state-specific estimate, not a finite-r rigorous bound.
* ``prepare_dephasing_analysis`` isolates the finite-time remainder left by
  replacing the leading BCH operator with its energy-block dephasing.

Every model has the form ``epsilon(r) = coefficient / r**order``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import ceil, factorial
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import csv

import numpy as np
from scipy.linalg import eigh, expm, logm, norm


ORDERS = (2, 4)
METHODS = ("naive_l1", "commutator", "paper_gs")

_I2 = np.eye(2, dtype=complex)
_PAULI = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_PRODUCT = {
    ("I", "I"): (1.0, "I"),
    ("I", "X"): (1.0, "X"),
    ("I", "Y"): (1.0, "Y"),
    ("I", "Z"): (1.0, "Z"),
    ("X", "I"): (1.0, "X"),
    ("X", "X"): (1.0, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("X", "Z"): (-1.0j, "Y"),
    ("Y", "I"): (1.0, "Y"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Y"): (1.0, "I"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "I"): (1.0, "Z"),
    ("Z", "X"): (1.0j, "Y"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "Z"): (1.0, "I"),
}

PauliOperator = dict[str, complex]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by the notebook and programmatic evaluations."""

    n_values: tuple[int, ...] = (4, 6, 8, 10, 12, 16, 20, 24, 32, 40)
    orders: tuple[int, ...] = ORDERS
    target_epsilon: float = 1.0e-3
    coupling: float = 1.0
    exact_diagonalization_max_n: int = 10
    dmrg_bond_dims: tuple[int, ...] = (16, 32, 64)
    dmrg_cutoff: float = 1.0e-9
    dmrg_tolerance: float = 1.0e-7


@dataclass(frozen=True)
class ResultRow:
    """One coefficient and the Trotter count inferred from it."""

    N: int
    order: int
    method: str
    coefficient: float
    epsilon: float
    r_continuous: float
    r: int
    rigorous: bool
    metric_scope: str


@dataclass(frozen=True)
class DephasingAnalysis:
    """Cached eigenspace data for finite-time dephasing calculations."""

    energies: np.ndarray
    eigenvectors: np.ndarray
    error_eigenbasis: np.ndarray
    dephased_eigenbasis: np.ndarray
    inverse_liouvillian_eigenbasis: np.ndarray
    energy_differences: np.ndarray
    same_energy: np.ndarray
    ground_indices: tuple[int, ...]
    gamma_ground: float
    gamma_all: float
    error_norm: float
    ground_diagonal_norm: float
    operator_diagonal_norm: float
    ground_inverse_norm: float
    operator_inverse_norm: float


@dataclass(frozen=True)
class DephasingMetric:
    """Finite-time dephasing residual and two analytic upper bounds."""

    scope: str
    time: float
    gamma: float
    diagonal_norm: float
    residual: float
    relative_residual: float
    inverse_gap_bound: float
    minimum_gap_bound: float


def _validate_n(n_qubits: int) -> int:
    n = int(n_qubits)
    if n < 2:
        raise ValueError("n_qubits must be at least 2")
    return n


def _validate_order(order: int) -> int:
    p = int(order)
    if p not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    return p


def _clean(operator: Mapping[str, complex], atol: float = 1.0e-13) -> PauliOperator:
    return {
        label: complex(coefficient)
        for label, coefficient in operator.items()
        if abs(coefficient) > atol
    }


def pauli_add(
    left: Mapping[str, complex],
    right: Mapping[str, complex],
    scale: complex = 1.0,
) -> PauliOperator:
    """Return ``left + scale * right`` in the sparse Pauli basis."""
    result = dict(left)
    for label, coefficient in right.items():
        result[label] = result.get(label, 0.0j) + scale * coefficient
    return _clean(result)


def pauli_scale(operator: Mapping[str, complex], scale: complex) -> PauliOperator:
    """Scale a sparse Pauli operator."""
    return _clean({label: scale * value for label, value in operator.items()})


def _pauli_product(label_left: str, label_right: str) -> tuple[complex, str]:
    if len(label_left) != len(label_right):
        raise ValueError("Pauli labels must have the same length")
    phase: complex = 1.0
    symbols: list[str] = []
    for left, right in zip(label_left, label_right):
        local_phase, symbol = _PRODUCT[(left, right)]
        phase *= local_phase
        symbols.append(symbol)
    return phase, "".join(symbols)


def _anticommute(label_left: str, label_right: str) -> bool:
    parity = 0
    for left, right in zip(label_left, label_right):
        if left != "I" and right != "I" and left != right:
            parity ^= 1
    return bool(parity)


def pauli_commutator(
    left: Mapping[str, complex], right: Mapping[str, complex]
) -> PauliOperator:
    """Return ``[left, right]`` without constructing dense matrices."""
    result: PauliOperator = {}
    for left_label, left_coefficient in left.items():
        for right_label, right_coefficient in right.items():
            if not _anticommute(left_label, right_label):
                continue
            phase, label = _pauli_product(left_label, right_label)
            value = 2.0 * left_coefficient * right_coefficient * phase
            result[label] = result.get(label, 0.0j) + value
    return _clean(result)


def pauli_l1_norm(operator: Mapping[str, complex]) -> float:
    """Return the coefficient L1 norm, an upper bound on spectral norm."""
    return float(sum(abs(value) for value in operator.values()))


def _bond_label(n_qubits: int, site: int, axis: str) -> str:
    symbols = ["I"] * n_qubits
    symbols[site] = axis
    symbols[site + 1] = axis
    return "".join(symbols)


@lru_cache(maxsize=None)
def _heisenberg_layers_cached(
    n_qubits: int, coupling: float
) -> tuple[tuple[tuple[str, complex], ...], tuple[tuple[str, complex], ...]]:
    n = _validate_n(n_qubits)
    layers: list[PauliOperator] = [{}, {}]
    for site in range(n - 1):
        layer = layers[site % 2]
        for axis in "XYZ":
            layer[_bond_label(n, site, axis)] = complex(coupling)
    return tuple(sorted(layers[0].items())), tuple(sorted(layers[1].items()))


def heisenberg_layers_pauli(
    n_qubits: int, coupling: float = 1.0
) -> tuple[PauliOperator, PauliOperator]:
    """Return the odd-bond layer ``A`` and even-bond layer ``B``."""
    a_items, b_items = _heisenberg_layers_cached(int(n_qubits), float(coupling))
    return dict(a_items), dict(b_items)


def heisenberg_hamiltonian_pauli(
    n_qubits: int, coupling: float = 1.0
) -> PauliOperator:
    """Return the full open-chain Hamiltonian in the Pauli basis."""
    A, B = heisenberg_layers_pauli(n_qubits, coupling)
    return pauli_add(A, B)


def pauli_string_matrix(label: str) -> np.ndarray:
    """Return the dense matrix of one Pauli string."""
    result = np.array([[1.0 + 0.0j]])
    for symbol in label:
        if symbol not in _PAULI:
            raise ValueError("Pauli labels may contain only I, X, Y, and Z")
        result = np.kron(result, _PAULI[symbol])
    return result


def pauli_to_dense(operator: Mapping[str, complex]) -> np.ndarray:
    """Materialize a sparse Pauli operator as a dense matrix."""
    if not operator:
        raise ValueError("cannot infer system size from an empty operator")
    n_qubits = len(next(iter(operator)))
    result = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for label, coefficient in operator.items():
        result += coefficient * pauli_string_matrix(label)
    return result


def heisenberg_dense(
    n_qubits: int, coupling: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dense ``(A, B, H)`` matrices for a small chain."""
    A_pauli, B_pauli = heisenberg_layers_pauli(n_qubits, coupling)
    A = pauli_to_dense(A_pauli)
    B = pauli_to_dense(B_pauli)
    return A, B, A + B


def suzuki_sequence(order: int, scale: float = 1.0) -> list[tuple[str, float]]:
    """Return the ordered ``A``/``B`` exponentials of one Suzuki step."""
    p = _validate_order(order)
    if p == 2:
        return [("A", scale / 2.0), ("B", scale), ("A", scale / 2.0)]
    u = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    child = suzuki_sequence(2, scale * u)
    middle = suzuki_sequence(2, scale * (1.0 - 4.0 * u))
    return child + child + middle + child + child


def suzuki_step_dense(
    A: np.ndarray, B: np.ndarray, step_size: float, order: int
) -> np.ndarray:
    """Construct one dense second- or fourth-order Suzuki step."""
    if A.shape != B.shape or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A and B must be square matrices of the same size")
    factors = {"A": A, "B": B}
    result = np.eye(A.shape[0], dtype=complex)
    for name, scale in suzuki_sequence(order):
        result = result @ expm(-1j * step_size * scale * factors[name])
    return result


def trotter_error_dense(
    A: np.ndarray, B: np.ndarray, time: float, n_steps: int, order: int
) -> float:
    """Return the exact spectral-norm error for a small dense system."""
    r = int(n_steps)
    if r < 1:
        raise ValueError("n_steps must be positive")
    exact = expm(-1j * float(time) * (A + B))
    approximate = np.linalg.matrix_power(
        suzuki_step_dense(A, B, float(time) / r, order), r
    )
    return float(norm(approximate - exact, 2))


def trotter_state_error_dense(
    A: np.ndarray,
    B: np.ndarray,
    time: float,
    n_steps: int,
    order: int,
    state: np.ndarray,
) -> float:
    """Return ``||(S_p^r - exp(-itH)) |state>||_2`` for a small system."""
    r = int(n_steps)
    if r < 1:
        raise ValueError("n_steps must be positive")
    vector = np.asarray(state, dtype=complex)
    if vector.ndim != 1 or vector.shape[0] != A.shape[0]:
        raise ValueError("state must be a vector compatible with A and B")
    if not np.isclose(norm(vector), 1.0, atol=1.0e-10):
        raise ValueError("state must be normalized")
    exact_state = expm(-1j * float(time) * (A + B)) @ vector
    approximate_state = np.linalg.matrix_power(
        suzuki_step_dense(A, B, float(time) / r, order), r
    ) @ vector
    return float(norm(approximate_state - exact_state))


def _layer_norm_sum(n_qubits: int, coupling: float) -> float:
    # Bonds inside either layer have disjoint support.  Each XX+YY+ZZ bond
    # has spectral norm 3|J|, so the layer norms add exactly.
    return 3.0 * (int(n_qubits) - 1) * abs(float(coupling))


def naive_l1_coefficient(
    n_qubits: int, order: int, time: float | None = None, coupling: float = 1.0
) -> float:
    """Return a concrete finite-r L1-bound coefficient.

    This is the anti-Hermitian specialization of the Taylor bound underlying
    Lemma 6 of Childs et al. (2021).  For ``Upsilon`` product-formula stages,
    the global error is at most

        t**(p+1) * ((Upsilon*lambda)**(p+1) + lambda**(p+1))
        / ((p+1)! * r**p).
    """
    n = _validate_n(n_qubits)
    p = _validate_order(order)
    t = float(n if time is None else time)
    if t < 0:
        raise ValueError("time must be nonnegative")
    stage_count = {2: 2, 4: 10}[p]
    lambda_1 = _layer_norm_sum(n, coupling)
    local = ((stage_count * lambda_1) ** (p + 1) + lambda_1 ** (p + 1)) / factorial(p + 1)
    return float(t ** (p + 1) * local)


def _nested(outer: Mapping[str, complex], inner: Mapping[str, complex]) -> PauliOperator:
    return pauli_commutator(outer, inner)


@lru_cache(maxsize=None)
def commutator_local_coefficient(
    n_qubits: int, order: int, coupling: float = 1.0
) -> float:
    """Return the time-independent coefficient in the commutator bound."""
    n = _validate_n(n_qubits)
    p = _validate_order(order)
    A, B = heisenberg_layers_pauli(n, coupling)
    BA = _nested(B, A)

    if p == 2:
        BBA = _nested(B, BA)
        AAB = _nested(A, _nested(A, B))
        return pauli_l1_norm(BBA) / 12.0 + pauli_l1_norm(AAB) / 24.0

    terms = (
        (0.0047, _nested(A, _nested(A, _nested(A, BA)))),
        (0.0057, _nested(A, _nested(A, _nested(B, BA)))),
        (0.0046, _nested(A, _nested(B, _nested(A, BA)))),
        (0.0074, _nested(A, _nested(B, _nested(B, BA)))),
        (0.0097, _nested(B, _nested(A, _nested(A, BA)))),
        (0.0097, _nested(B, _nested(A, _nested(B, BA)))),
        (0.0173, _nested(B, _nested(B, _nested(A, BA)))),
        (0.0284, _nested(B, _nested(B, _nested(B, BA)))),
    )
    return float(sum(weight * pauli_l1_norm(operator) for weight, operator in terms))


def commutator_coefficient(
    n_qubits: int, order: int, time: float | None = None, coupling: float = 1.0
) -> float:
    """Return ``C`` in the rigorous model ``epsilon(r) <= C/r**order``."""
    n = _validate_n(n_qubits)
    p = _validate_order(order)
    t = float(n if time is None else time)
    if t < 0:
        raise ValueError("time must be nonnegative")
    return float(t ** (p + 1) * commutator_local_coefficient(n, p, coupling))


def _symmetric_degree_three(center: PauliOperator, outer: PauliOperator) -> PauliOperator:
    comm = pauli_commutator(center, outer)
    result = pauli_scale(pauli_commutator(comm, center), -1.0 / 12.0)
    return pauli_add(
        result, pauli_commutator(comm, outer), scale=-1.0 / 24.0
    )


def _symmetric_degree_five(
    center1: PauliOperator,
    center3: PauliOperator,
    center5: PauliOperator,
    outer1: PauliOperator,
    outer3: PauliOperator,
    outer5: PauliOperator,
) -> PauliOperator:
    """Degree-five part of ``log(exp(B/2) exp(A) exp(B/2))``."""
    C2 = pauli_commutator(center1, outer1)
    C4 = pauli_add(
        pauli_commutator(center1, outer3),
        pauli_commutator(center3, outer1),
    )
    cubic_center = pauli_add(
        pauli_commutator(C2, center3),
        pauli_commutator(C4, center1),
    )
    cubic_outer = pauli_add(
        pauli_commutator(C2, outer3),
        pauli_commutator(C4, outer1),
    )
    quartic_center = pauli_commutator(pauli_commutator(C2, center1), C2)
    quartic_outer = pauli_commutator(pauli_commutator(C2, outer1), C2)
    cccc = pauli_commutator(
        pauli_commutator(pauli_commutator(C2, center1), center1), center1
    )
    oocc = pauli_commutator(
        pauli_commutator(pauli_commutator(C2, outer1), center1), center1
    )
    oooc = pauli_commutator(
        pauli_commutator(pauli_commutator(C2, outer1), outer1), center1
    )
    oooo = pauli_commutator(
        pauli_commutator(pauli_commutator(C2, outer1), outer1), outer1
    )

    result = pauli_add(center5, outer5)
    for coefficient, operator in (
        (-1.0 / 12.0, cubic_center),
        (-1.0 / 24.0, cubic_outer),
        (-1.0 / 360.0, quartic_center),
        (1.0 / 480.0, quartic_outer),
        (1.0 / 720.0, cccc),
        (1.0 / 180.0, oocc),
        (7.0 / 1440.0, oooc),
        (7.0 / 5760.0, oooo),
    ):
        result = pauli_add(result, operator, scale=coefficient)
    return result


@lru_cache(maxsize=None)
def _effective_error_cached(
    n_qubits: int, order: int, coupling: float
) -> tuple[tuple[str, complex], ...]:
    n = _validate_n(n_qubits)
    p = _validate_order(order)
    H_A, H_B = heisenberg_layers_pauli(n, coupling)
    K_A = pauli_scale(H_A, -1j)
    K_B = pauli_scale(H_B, -1j)

    # S2 = exp(h K_A/2) exp(h K_B) exp(h K_A/2): K_B is the
    # center and K_A the outer generator in the symmetric BCH formula.
    R3 = _symmetric_degree_three(K_B, K_A)
    if p == 2:
        return tuple(sorted(pauli_scale(R3, 1j).items()))

    R5 = _symmetric_degree_five(K_B, {}, {}, K_A, {}, {})
    K = pauli_add(K_A, K_B)
    u = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    v = 1.0 - 4.0 * u
    center = {
        1: pauli_scale(K, v),
        3: pauli_scale(R3, v**3),
        5: pauli_scale(R5, v**5),
    }
    outer = {
        1: pauli_scale(K, 4.0 * u),
        3: pauli_scale(R3, 4.0 * u**3),
        5: pauli_scale(R5, 4.0 * u**5),
    }
    R5_s4 = _symmetric_degree_five(
        center[1], center[3], center[5], outer[1], outer[3], outer[5]
    )
    return tuple(sorted(pauli_scale(R5_s4, 1j).items()))


def effective_error_pauli(
    n_qubits: int, order: int, coupling: float = 1.0
) -> PauliOperator:
    """Return the Hermitian leading BCH error operator ``E_p``.

    The convention is ``log(S_p(h)) = -ihH + h**(p+1) R + ...`` and
    ``E_p = iR``.
    """
    return dict(_effective_error_cached(int(n_qubits), int(order), float(coupling)))


def prepare_dephasing_analysis(
    hamiltonian: np.ndarray,
    error_operator: np.ndarray,
    atol: float = 1.0e-9,
    rtol: float = 1.0e-9,
) -> DephasingAnalysis:
    """Diagonalize ``H`` once and prepare finite-time dephasing data.

    Equal-energy blocks are retained in full, so the construction is
    invariant under basis rotations inside degenerate eigenspaces.
    """
    H = np.asarray(hamiltonian, dtype=complex)
    E = np.asarray(error_operator, dtype=complex)
    if H.ndim != 2 or H.shape[0] != H.shape[1] or E.shape != H.shape:
        raise ValueError("hamiltonian and error_operator must be equal square matrices")
    if not np.allclose(H, H.conj().T, atol=atol, rtol=rtol):
        raise ValueError("hamiltonian must be Hermitian")
    if not np.allclose(E, E.conj().T, atol=atol, rtol=rtol):
        raise ValueError("error_operator must be Hermitian")

    energies, eigenvectors = eigh(H)
    error_eigenbasis = eigenvectors.conj().T @ E @ eigenvectors
    energy_differences = energies[:, None] - energies[None, :]
    energy_scale = np.maximum(
        1.0, np.maximum(np.abs(energies[:, None]), np.abs(energies[None, :]))
    )
    same_energy = np.abs(energy_differences) <= atol + rtol * energy_scale
    dephased_eigenbasis = np.where(same_energy, error_eigenbasis, 0.0j)
    inverse_liouvillian = np.zeros_like(error_eigenbasis)
    inverse_liouvillian[~same_energy] = (
        error_eigenbasis[~same_energy]
        / (1j * energy_differences[~same_energy])
    )

    block_starts = [0]
    for index in range(1, len(energies)):
        if not same_energy[index, index - 1]:
            block_starts.append(index)
    distinct_energies = energies[np.asarray(block_starts, dtype=int)]
    if len(distinct_energies) < 2:
        gamma_ground = gamma_all = float("inf")
    else:
        distinct_gaps = np.diff(distinct_energies)
        gamma_ground = float(distinct_gaps[0])
        gamma_all = float(np.min(distinct_gaps))
    ground_stop = block_starts[1] if len(block_starts) > 1 else len(energies)

    return DephasingAnalysis(
        energies=energies,
        eigenvectors=eigenvectors,
        error_eigenbasis=error_eigenbasis,
        dephased_eigenbasis=dephased_eigenbasis,
        inverse_liouvillian_eigenbasis=inverse_liouvillian,
        energy_differences=energy_differences,
        same_energy=same_energy,
        ground_indices=tuple(range(ground_stop)),
        gamma_ground=gamma_ground,
        gamma_all=gamma_all,
        error_norm=float(norm(error_eigenbasis, 2)),
        ground_diagonal_norm=float(norm(dephased_eigenbasis[:, 0])),
        operator_diagonal_norm=float(norm(dephased_eigenbasis, 2)),
        ground_inverse_norm=float(norm(inverse_liouvillian[:, 0])),
        operator_inverse_norm=float(norm(inverse_liouvillian, 2)),
    )


def _finite_time_residual_eigenbasis(
    analysis: DephasingAnalysis, time: float
) -> np.ndarray:
    t = float(time)
    if t < 0:
        raise ValueError("time must be nonnegative")
    gaps = analysis.energy_differences
    factors = np.exp(0.5j * gaps * t) * np.sinc(gaps * t / (2.0 * np.pi))
    residual = np.where(
        analysis.same_energy, 0.0j, analysis.error_eigenbasis * factors
    )
    return 0.5 * (residual + residual.conj().T)


def finite_time_average_operator(
    analysis: DephasingAnalysis, time: float
) -> np.ndarray:
    """Return ``t^-1 integral_0^t exp(isH) E exp(-isH) ds``."""
    averaged = (
        analysis.dephased_eigenbasis
        + _finite_time_residual_eigenbasis(analysis, time)
    )
    vectors = analysis.eigenvectors
    return vectors @ averaged @ vectors.conj().T


def finite_time_dephasing_residual(
    analysis: DephasingAnalysis, time: float
) -> np.ndarray:
    """Return the finite-time remainder ``Q_t(E)`` in the original basis."""
    vectors = analysis.eigenvectors
    residual = _finite_time_residual_eigenbasis(analysis, time)
    return vectors @ residual @ vectors.conj().T


def dephasing_metric(
    analysis: DephasingAnalysis, time: float, scope: str
) -> DephasingMetric:
    """Evaluate ``Q_t`` and finite-time bounds for ``ground`` or ``operator``."""
    t = float(time)
    if t <= 0:
        raise ValueError("time must be positive when evaluating 1/t bounds")
    residual = _finite_time_residual_eigenbasis(analysis, t)
    if scope == "ground":
        # The even open Heisenberg chains used here have a unique ground state.
        # For a degenerate ground space this reports its first eigenvector and
        # the block dephasing itself remains basis invariant.
        ground_index = analysis.ground_indices[0]
        diagonal_norm = analysis.ground_diagonal_norm
        residual_norm = float(norm(residual[:, ground_index]))
        inverse_norm = analysis.ground_inverse_norm
        gamma = analysis.gamma_ground
    elif scope == "operator":
        diagonal_norm = analysis.operator_diagonal_norm
        residual_norm = float(norm(residual, 2))
        inverse_norm = analysis.operator_inverse_norm
        gamma = analysis.gamma_all
    else:
        raise ValueError("scope must be 'ground' or 'operator'")

    inverse_bound = 2.0 * inverse_norm / t
    minimum_gap_bound = (
        2.0 * analysis.error_norm / (t * gamma) if np.isfinite(gamma) else 0.0
    )
    relative = residual_norm / diagonal_norm if diagonal_norm else float("inf")
    return DephasingMetric(
        scope=scope,
        time=t,
        gamma=float(gamma),
        diagonal_norm=diagonal_norm,
        residual=residual_norm,
        relative_residual=float(relative),
        inverse_gap_bound=float(inverse_bound),
        minimum_gap_bound=float(minimum_gap_bound),
    )


def paper_finite_time_residual_bound(
    analysis: DephasingAnalysis, time: float, eta: float, scope: str
) -> float:
    """Return the Appendix-A finite-time bound for the scaled perturbation.

    The result bounds the dephasing replacement in
    ``exp(i * (t H + eta E))`` and is separate from BCH truncation error.
    """
    t = float(time)
    eta_abs = abs(float(eta))
    if t <= 0:
        raise ValueError("time must be positive")
    if scope == "ground":
        gamma = analysis.gamma_ground
    elif scope == "operator":
        gamma = analysis.gamma_all
    else:
        raise ValueError("scope must be 'ground' or 'operator'")
    if not np.isfinite(gamma):
        return 0.0
    scaled_norm = eta_abs * analysis.error_norm
    return float(4.0 * scaled_norm * np.exp(2.0 * scaled_norm) / (t * gamma))


def nonlinear_dephasing_residual(
    analysis: DephasingAnalysis, time: float, eta: float, scope: str
) -> float:
    """Directly compare full and block-dephased perturbed evolutions."""
    t = float(time)
    if t < 0:
        raise ValueError("time must be nonnegative")
    diagonal_hamiltonian = np.diag(t * analysis.energies)
    full = expm(1j * (diagonal_hamiltonian + float(eta) * analysis.error_eigenbasis))
    dephased = expm(
        1j * (diagonal_hamiltonian + float(eta) * analysis.dephased_eigenbasis)
    )
    difference = full - dephased
    if scope == "ground":
        state = np.zeros(len(analysis.energies), dtype=complex)
        state[analysis.ground_indices[0]] = 1.0
        return float(norm(difference @ state))
    if scope == "operator":
        return float(norm(difference, 2))
    raise ValueError("scope must be 'ground' or 'operator'")


def dephased_error_operator(
    hamiltonian: np.ndarray,
    error_operator: np.ndarray,
    atol: float = 1.0e-9,
    rtol: float = 1.0e-9,
) -> np.ndarray:
    """Return ``sum_lambda P_lambda E P_lambda`` including degeneracies."""
    analysis = prepare_dephasing_analysis(
        hamiltonian, error_operator, atol=atol, rtol=rtol
    )
    vectors = analysis.eigenvectors
    result = vectors @ analysis.dephased_eigenbasis @ vectors.conj().T
    return 0.5 * (result + result.conj().T)


@lru_cache(maxsize=None)
def exact_paper_local_coefficients(
    n_qubits: int, order: int, coupling: float = 1.0
) -> tuple[float, float]:
    """Return ``(ground-state, all-state)`` diagonal BCH coefficients."""
    A, B, H = heisenberg_dense(n_qubits, coupling)
    del A, B
    E = pauli_to_dense(effective_error_pauli(n_qubits, order, coupling))
    values, vectors = eigh(H)
    ground = vectors[:, 0]
    ground_coefficient = abs(np.vdot(ground, E @ ground))
    dephased = dephased_error_operator(H, E)
    all_state_coefficient = norm(dephased, 2)
    return float(ground_coefficient), float(all_state_coefficient)


def _require_quimb():
    try:
        import quimb.tensor as qtn
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "N > exact_diagonalization_max_n requires quimb; "
            "install the repository requirements"
        ) from exc
    return qtn


def _operator_to_mpo(
    operator: Mapping[str, complex], max_bond: int, cutoff: float
):
    """Convert a finite-range Pauli expansion to a compressed quimb MPO."""
    qtn = _require_quimb()
    if not operator:
        raise ValueError("operator must not be empty")
    n_qubits = len(next(iter(operator)))
    grouped: dict[tuple[int, int], PauliOperator] = {}
    for label, coefficient in operator.items():
        support = [site for site, symbol in enumerate(label) if symbol != "I"]
        if not support:
            interval = (0, 1)
            local_label = "I"
        else:
            interval = (min(support), max(support) + 1)
            local_label = label[interval[0] : interval[1]]
        bucket = grouped.setdefault(interval, {})
        bucket[local_label] = bucket.get(local_label, 0.0j) + coefficient

    blocks = []
    for (left, right), local_operator in grouped.items():
        width = right - left
        local_matrix = np.zeros((2**width, 2**width), dtype=complex)
        for label, coefficient in local_operator.items():
            local_matrix += coefficient * pauli_string_matrix(label)
        block = qtn.MatrixProductOperator.from_dense(
            local_matrix,
            dims=2,
            sites=list(range(left, right)),
            L=n_qubits,
            cutoff=cutoff,
        ).fill_empty_sites()
        blocks.append(block)

    result = blocks[0]
    for index, block in enumerate(blocks[1:], start=1):
        result = result + block
        if index % 16 == 0:
            result.compress(max_bond=max_bond, cutoff=cutoff)
    result.compress(max_bond=max_bond, cutoff=cutoff)
    return result


@lru_cache(maxsize=None)
def _dmrg_ground_state_cached(
    n_qubits: int,
    coupling: float,
    bond_dims: tuple[int, ...],
    cutoff: float,
    tolerance: float,
):
    qtn = _require_quimb()
    hamiltonian = qtn.MPO_ham_heis(
        n_qubits, j=4.0 * coupling, S=0.5, cyclic=False
    )
    solver = qtn.DMRG2(hamiltonian, which="SA")
    solver.solve(
        tol=tolerance,
        bond_dims=list(bond_dims),
        cutoffs=cutoff,
        verbosity=0,
    )
    return solver.state.copy(), float(np.real(solver.energy))


def dmrg_ground_state_coefficient(
    n_qubits: int,
    order: int,
    coupling: float = 1.0,
    bond_dims: Sequence[int] = (16, 32, 64),
    cutoff: float = 1.0e-9,
    tolerance: float = 1.0e-7,
) -> float:
    """Estimate ``|<psi_0|E_p|psi_0>|`` with DMRG/MPO."""
    dims = tuple(int(value) for value in bond_dims)
    state, _ = _dmrg_ground_state_cached(
        int(n_qubits), float(coupling), dims, float(cutoff), float(tolerance)
    )
    error_mpo = _operator_to_mpo(
        effective_error_pauli(n_qubits, order, coupling),
        max_bond=max(dims),
        cutoff=cutoff,
    )
    # Applying the MPO first keeps the contraction one-dimensional and avoids
    # invoking cotengra's general hyper-optimizer (and its process pool) for a
    # simple MPS expectation value.
    applied = error_mpo.apply(
        state,
        compress=True,
        max_bond=max(dims),
        cutoff=cutoff,
    )
    expectation = state.overlap(applied)
    return float(abs(complex(expectation)))


def paper_ground_state_coefficient(
    n_qubits: int,
    order: int,
    time: float | None = None,
    coupling: float = 1.0,
    exact_diagonalization_max_n: int = 10,
    bond_dims: Sequence[int] = (16, 32, 64),
    cutoff: float = 1.0e-9,
    tolerance: float = 1.0e-7,
) -> float:
    """Return the full ``t**(p+1)`` GS coefficient for the paper estimate."""
    n = _validate_n(n_qubits)
    p = _validate_order(order)
    t = float(n if time is None else time)
    if n <= exact_diagonalization_max_n:
        local, _ = exact_paper_local_coefficients(n, p, coupling)
    else:
        local = dmrg_ground_state_coefficient(
            n, p, coupling, bond_dims, cutoff, tolerance
        )
    return float(t ** (p + 1) * local)


def paper_all_state_coefficient(
    n_qubits: int,
    order: int,
    time: float | None = None,
    coupling: float = 1.0,
) -> float:
    """Return the small-ED ``t**(p+1) ||D_H(E_p)||`` coefficient."""
    n = _validate_n(n_qubits)
    p = _validate_order(order)
    t = float(n if time is None else time)
    _, local = exact_paper_local_coefficients(n, p, coupling)
    return float(t ** (p + 1) * local)


def required_steps(coefficient: float, epsilon: float, order: int) -> tuple[float, int]:
    """Invert ``coefficient / r**order <= epsilon``."""
    p = _validate_order(order)
    coefficient = float(coefficient)
    epsilon = float(epsilon)
    if coefficient < 0 or epsilon <= 0:
        raise ValueError("coefficient must be nonnegative and epsilon positive")
    continuous = (coefficient / epsilon) ** (1.0 / p) if coefficient else 0.0
    return float(continuous), max(1, int(ceil(continuous)))


def error_model(coefficient: float, r: int | np.ndarray, order: int):
    """Evaluate ``coefficient / r**order`` for scalar or array ``r``."""
    p = _validate_order(order)
    values = np.asarray(r, dtype=float)
    if np.any(values <= 0):
        raise ValueError("all r values must be positive")
    result = float(coefficient) / values**p
    return float(result) if result.ndim == 0 else result


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> list[ResultRow]:
    """Evaluate all configured coefficients and inferred Trotter counts."""
    rows: list[ResultRow] = []
    for n in config.n_values:
        if n % 2:
            raise ValueError("the configured scaling study uses even N only")
        for order in config.orders:
            coefficients = {
                "naive_l1": naive_l1_coefficient(n, order, n, config.coupling),
                "commutator": commutator_coefficient(n, order, n, config.coupling),
                "paper_gs": paper_ground_state_coefficient(
                    n,
                    order,
                    n,
                    config.coupling,
                    config.exact_diagonalization_max_n,
                    config.dmrg_bond_dims,
                    config.dmrg_cutoff,
                    config.dmrg_tolerance,
                ),
            }
            metadata = {
                "naive_l1": (True, "worst_case"),
                "commutator": (True, "worst_case"),
                "paper_gs": (False, "ground_state_asymptotic"),
            }
            for method, coefficient in coefficients.items():
                continuous, integer = required_steps(
                    coefficient, config.target_epsilon, order
                )
                rigorous, scope = metadata[method]
                rows.append(
                    ResultRow(
                        N=n,
                        order=order,
                        method=method,
                        coefficient=coefficient,
                        epsilon=config.target_epsilon,
                        r_continuous=continuous,
                        r=integer,
                        rigorous=rigorous,
                        metric_scope=scope,
                    )
                )

            if n <= config.exact_diagonalization_max_n:
                coefficient = paper_all_state_coefficient(
                    n, order, n, config.coupling
                )
                continuous, integer = required_steps(
                    coefficient, config.target_epsilon, order
                )
                rows.append(
                    ResultRow(
                        N=n,
                        order=order,
                        method="paper_all_state",
                        coefficient=coefficient,
                        epsilon=config.target_epsilon,
                        r_continuous=continuous,
                        r=integer,
                        rigorous=False,
                        metric_scope="all_state_asymptotic",
                    )
                )
    return rows


def save_results(rows: Iterable[ResultRow], path: str | Path) -> Path:
    """Write result rows as a stable CSV table."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in ResultRow.__dataclass_fields__.values()]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return destination


def bch_log_residual(
    n_qubits: int,
    order: int,
    step_size: float,
    coupling: float = 1.0,
) -> float:
    """Return a relative dense check of the leading BCH coefficient."""
    A, B, H = heisenberg_dense(n_qubits, coupling)
    E = pauli_to_dense(effective_error_pauli(n_qubits, order, coupling))
    R = -1j * E
    numerical = (logm(suzuki_step_dense(A, B, step_size, order)) + 1j * step_size * H)
    numerical /= step_size ** (order + 1)
    return float(norm(numerical - R, 2) / max(norm(R, 2), 1.0e-15))


__all__ = [
    "ORDERS",
    "METHODS",
    "ExperimentConfig",
    "ResultRow",
    "DephasingAnalysis",
    "DephasingMetric",
    "heisenberg_layers_pauli",
    "heisenberg_hamiltonian_pauli",
    "heisenberg_dense",
    "pauli_add",
    "pauli_commutator",
    "pauli_l1_norm",
    "pauli_to_dense",
    "suzuki_sequence",
    "suzuki_step_dense",
    "trotter_error_dense",
    "naive_l1_coefficient",
    "commutator_local_coefficient",
    "commutator_coefficient",
    "effective_error_pauli",
    "prepare_dephasing_analysis",
    "finite_time_average_operator",
    "finite_time_dephasing_residual",
    "dephasing_metric",
    "paper_finite_time_residual_bound",
    "nonlinear_dephasing_residual",
    "dephased_error_operator",
    "exact_paper_local_coefficients",
    "dmrg_ground_state_coefficient",
    "paper_ground_state_coefficient",
    "paper_all_state_coefficient",
    "required_steps",
    "error_model",
    "run_experiment",
    "save_results",
    "bch_log_residual",
]
