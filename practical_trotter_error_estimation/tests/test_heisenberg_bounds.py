from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import eigh, norm


STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import heisenberg_bounds as hb


@pytest.mark.parametrize("n_qubits", [4, 6])
def test_even_odd_decomposition_and_layer_commutativity(n_qubits: int) -> None:
    A_pauli, B_pauli = hb.heisenberg_layers_pauli(n_qubits)
    A, B, H = hb.heisenberg_dense(n_qubits)

    assert np.allclose(A + B, H)
    assert np.allclose(H, H.conj().T)

    for layer in (A_pauli, B_pauli):
        labels = list(layer)
        for i, left in enumerate(labels):
            for right in labels[i + 1 :]:
                assert not hb.pauli_commutator({left: 1.0}, {right: 1.0})


@pytest.mark.parametrize("order", hb.ORDERS)
def test_suzuki_convergence_order(order: int) -> None:
    A, B, H = hb.heisenberg_dense(4)
    del H
    rs = np.array([4, 8, 16, 32], dtype=int)
    errors = np.array(
        [hb.trotter_error_dense(A, B, time=1.0, n_steps=int(r), order=order) for r in rs]
    )
    slope = -np.polyfit(np.log(rs), np.log(errors), 1)[0]
    assert abs(slope - order) < 0.2


@pytest.mark.parametrize("order,step_size,tolerance", [(2, 0.02, 0.03), (4, 0.08, 0.08)])
def test_compact_bch_matches_matrix_logarithm(
    order: int, step_size: float, tolerance: float
) -> None:
    assert hb.bch_log_residual(4, order, step_size) < tolerance


@pytest.mark.parametrize("n_qubits", [4, 6])
@pytest.mark.parametrize("order", hb.ORDERS)
@pytest.mark.parametrize("n_steps", [1, 2, 4])
def test_rigorous_bounds_dominate_dense_error(
    n_qubits: int, order: int, n_steps: int
) -> None:
    A, B, _ = hb.heisenberg_dense(n_qubits)
    measured = hb.trotter_error_dense(A, B, n_qubits, n_steps, order)
    naive = hb.error_model(
        hb.naive_l1_coefficient(n_qubits, order), n_steps, order
    )
    commutator = hb.error_model(
        hb.commutator_coefficient(n_qubits, order), n_steps, order
    )
    assert measured <= naive + 1.0e-10
    assert measured <= commutator + 1.0e-10


@pytest.mark.parametrize("order", hb.ORDERS)
def test_ground_state_error_is_bounded_by_spectral_error(order: int) -> None:
    A, B, H = hb.heisenberg_dense(4)
    _, vectors = eigh(H)
    ground_state = vectors[:, 0]
    state_error = hb.trotter_state_error_dense(A, B, 4.0, 5, order, ground_state)
    spectral_error = hb.trotter_error_dense(A, B, 4.0, 5, order)

    assert state_error <= spectral_error + 1.0e-10


def test_dephasing_uses_degenerate_projectors() -> None:
    _, _, H = hb.heisenberg_dense(4)
    E = hb.pauli_to_dense(hb.effective_error_pauli(4, 2))
    D = hb.dephased_error_operator(H, E)

    assert norm(H @ D - D @ H, 2) < 1.0e-9
    assert np.allclose(D, D.conj().T)

    values, vectors = eigh(H)
    rebuilt = np.zeros_like(E)
    rng = np.random.default_rng(1234)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and np.isclose(values[stop], values[start], atol=1e-9):
            stop += 1
        subspace = vectors[:, start:stop]
        random_matrix = rng.normal(size=(stop - start, stop - start)) + 1j * rng.normal(
            size=(stop - start, stop - start)
        )
        rotation, _ = np.linalg.qr(random_matrix)
        rotated = subspace @ rotation
        block = rotated.conj().T @ E @ rotated
        rebuilt += rotated @ block @ rotated.conj().T
        start = stop

    assert np.allclose(D, rebuilt, atol=1.0e-9)
    assert np.isclose(norm(D, 2), norm(rebuilt, 2), atol=1.0e-10)


def test_finite_time_average_matches_direct_quadrature() -> None:
    H = np.diag([0.0, 0.0, 1.7]).astype(complex)
    E = np.array(
        [[1.0, 0.2j, 0.3], [-0.2j, -0.4, 0.1j], [0.3, -0.1j, 0.7]],
        dtype=complex,
    )
    analysis = hb.prepare_dephasing_analysis(H, E)
    time = 2.3
    sample_times = np.linspace(0.0, time, 10001)
    direct = np.zeros_like(E)
    for index, sample_time in enumerate(sample_times):
        phases = np.exp(1j * sample_time * np.diag(H))
        integrand = phases[:, None] * E * phases.conj()[None, :]
        weight = 0.5 if index in (0, len(sample_times) - 1) else 1.0
        direct += weight * integrand
    direct *= (sample_times[1] - sample_times[0]) / time

    averaged = hb.finite_time_average_operator(analysis, time)
    assert np.allclose(averaged, direct, atol=2.0e-8)


def test_dephasing_residual_limit_blocks_and_covariance() -> None:
    H = np.diag([0.0, 0.0, 2.0]).astype(complex)
    E = np.array(
        [[1.0, 0.4, 0.3j], [0.4, -0.2, 0.5], [-0.3j, 0.5, 0.7]],
        dtype=complex,
    )
    analysis = hb.prepare_dephasing_analysis(H, E)
    Q_zero = hb.finite_time_dephasing_residual(analysis, 0.0)
    D = hb.dephased_error_operator(H, E)
    assert np.allclose(Q_zero, E - D, atol=1.0e-10)
    assert np.allclose(
        analysis.error_eigenbasis[analysis.same_energy],
        analysis.dephased_eigenbasis[analysis.same_energy],
    )
    assert np.allclose(
        hb.finite_time_dephasing_residual(analysis, 3.0),
        hb.finite_time_dephasing_residual(analysis, 3.0).conj().T,
    )

    rng = np.random.default_rng(4321)
    random_matrix = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    unitary, _ = np.linalg.qr(random_matrix)
    rotated = hb.prepare_dephasing_analysis(
        unitary @ H @ unitary.conj().T,
        unitary @ E @ unitary.conj().T,
    )
    assert np.allclose(
        hb.finite_time_dephasing_residual(rotated, 3.0),
        unitary @ hb.finite_time_dephasing_residual(analysis, 3.0) @ unitary.conj().T,
        atol=1.0e-9,
    )


@pytest.mark.parametrize("scope", ["ground", "operator"])
@pytest.mark.parametrize("time", [0.7, 4.0, 20.0])
def test_inverse_liouvillian_dephasing_bound(scope: str, time: float) -> None:
    _, _, H = hb.heisenberg_dense(4)
    E = hb.pauli_to_dense(hb.effective_error_pauli(4, 2))
    analysis = hb.prepare_dephasing_analysis(H, E)
    metric = hb.dephasing_metric(analysis, time, scope)

    assert metric.residual <= metric.inverse_gap_bound + 1.0e-10
    assert metric.inverse_gap_bound <= metric.minimum_gap_bound + 1.0e-10


@pytest.mark.parametrize("scope", ["ground", "operator"])
def test_nonlinear_dephasing_residual_has_linear_response(scope: str) -> None:
    _, _, H = hb.heisenberg_dense(4)
    E = hb.pauli_to_dense(hb.effective_error_pauli(4, 2))
    analysis = hb.prepare_dephasing_analysis(H, E)
    time = 4.0
    eta = 1.0e-5
    linear = eta * hb.dephasing_metric(analysis, time, scope).residual
    nonlinear = hb.nonlinear_dephasing_residual(analysis, time, eta, scope)

    assert np.isclose(nonlinear, linear, rtol=2.0e-4, atol=1.0e-11)


@pytest.mark.parametrize("order", hb.ORDERS)
def test_required_steps_inverts_model(order: int) -> None:
    coefficient = 123.45
    epsilon = 1.0e-3
    continuous, integer = hb.required_steps(coefficient, epsilon, order)
    assert np.isclose(hb.error_model(coefficient, continuous, order), epsilon)
    assert hb.error_model(coefficient, integer, order) <= epsilon
    if integer > 1:
        assert hb.error_model(coefficient, integer - 1, order) > epsilon


@pytest.mark.slow
@pytest.mark.parametrize("order", hb.ORDERS)
def test_dmrg_ground_state_coefficient_matches_exact(order: int) -> None:
    pytest.importorskip("quimb")
    exact, _ = hb.exact_paper_local_coefficients(6, order)
    estimated = hb.dmrg_ground_state_coefficient(
        6,
        order,
        bond_dims=(8, 16, 32),
        cutoff=1.0e-10,
        tolerance=1.0e-9,
    )
    assert np.isclose(estimated, exact, rtol=2.0e-5, atol=1.0e-8)
