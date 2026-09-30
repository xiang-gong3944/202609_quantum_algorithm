from __future__ import annotations

from itertools import permutations, product

import numpy as np

from randomized_hamiltonian_simulation import qdrift as qd
from randomized_hamiltonian_simulation import randomized_trotter as rt
from randomized_hamiltonian_simulation.quantum_utils import (
    I2,
    PAULIS,
    Y,
    apply_channel,
    density,
    hermitian_exp,
    trace_norm_error,
)


COEFFICIENTS = np.array([0.9, 0.5, 0.2])
TERMS = COEFFICIENTS[:, None, None] * PAULIS
HAMILTONIAN = np.sum(TERMS, axis=0)
PSI0 = np.array([1.0, 0.0], dtype=complex)
RHO0 = density(PSI0)
FIXED_ORDER = (1, 0, 2)


def assert_density_matrix(rho: np.ndarray, atol: float = 2.0e-12) -> None:
    assert np.allclose(rho, rho.conj().T, atol=atol)
    assert np.isclose(np.trace(rho), 1.0, atol=atol)
    assert np.min(np.linalg.eigvalsh(rho)) >= -atol


def test_step_unitaries_are_unitary() -> None:
    lie = rt.lie_step_unitary(TERMS, 0.17, FIXED_ORDER)
    assert np.allclose(lie.conj().T @ lie, I2)

    for gate in qd.qdrift_step_unitaries(COEFFICIENTS, PAULIS, 1.3, 11):
        assert np.allclose(gate.conj().T @ gate, I2)


def test_randomized_average_matches_explicit_sequence_enumeration() -> None:
    total_time = 0.7
    steps = 2
    candidates = [
        rt.lie_step_unitary(TERMS, total_time / steps, order)
        for order in permutations(range(3))
    ]
    explicit = np.zeros_like(RHO0)
    for first, second in product(candidates, repeat=steps):
        state = second @ first @ PSI0
        explicit += density(state) / len(candidates) ** steps

    calculated = rt.randomized_lie_average_path(
        TERMS, RHO0, total_time, steps
    )[-1]
    assert np.allclose(calculated, explicit)


def test_qdrift_average_matches_explicit_sequence_enumeration() -> None:
    total_time = 0.7
    steps = 2
    _, probabilities, _ = qd.qdrift_distribution(COEFFICIENTS)
    candidates = qd.qdrift_step_unitaries(COEFFICIENTS, PAULIS, total_time, steps)
    explicit = np.zeros_like(RHO0)
    for first_index, second_index in product(range(3), repeat=steps):
        state = candidates[second_index] @ candidates[first_index] @ PSI0
        weight = probabilities[first_index] * probabilities[second_index]
        explicit += weight * density(state)

    calculated = qd.qdrift_average_path(
        COEFFICIENTS, PAULIS, RHO0, total_time, steps
    )[-1]
    assert np.allclose(calculated, explicit)


def test_average_states_and_sampled_paths_are_physical() -> None:
    randomized_average = rt.randomized_lie_average_path(TERMS, RHO0, 1.2, 8)
    qdrift_average = qd.qdrift_average_path(COEFFICIENTS, PAULIS, RHO0, 1.2, 16)
    for rho in np.concatenate((randomized_average, qdrift_average)):
        assert_density_matrix(rho)

    randomized_samples = rt.sample_randomized_lie_paths(
        TERMS, PSI0, 1.2, 8, 12, np.random.default_rng(10)
    )
    qdrift_samples = qd.sample_qdrift_paths(
        COEFFICIENTS, PAULIS, PSI0, 1.2, 16, 12, np.random.default_rng(20)
    )
    assert np.allclose(np.linalg.norm(randomized_samples, axis=2), 1.0)
    assert np.allclose(np.linalg.norm(qdrift_samples, axis=2), 1.0)


def test_qdrift_signed_coefficients_and_probabilities() -> None:
    signed = np.array([0.9, -0.5, 0.2])
    lam, probabilities, signs = qd.qdrift_distribution(signed)
    assert np.isclose(lam, 1.6)
    assert np.allclose(probabilities, np.abs(signed) / lam)
    assert np.array_equal(signs, np.array([1.0, -1.0, 1.0]))

    gates = qd.qdrift_step_unitaries(signed, PAULIS, total_time=0.8, steps=4)
    assert np.allclose(gates[1], hermitian_exp(-Y, lam * 0.8 / 4))


def test_fixed_randomized_and_qdrift_convergence_orders() -> None:
    total_time = 1.6
    step_counts = np.array([16, 24, 32, 48, 64, 96, 128, 192, 256])
    exact_state = hermitian_exp(HAMILTONIAN, total_time) @ PSI0
    exact_rho = density(exact_state)

    fixed_errors = []
    randomized_errors = []
    qdrift_errors = []
    qdrift_bounds = []
    for steps in step_counts:
        fixed_errors.append(trace_norm_error(
            rt.fixed_lie_path(TERMS, RHO0, total_time, int(steps), FIXED_ORDER)[-1],
            exact_rho,
        ))
        randomized_errors.append(trace_norm_error(
            rt.randomized_lie_average_path(TERMS, RHO0, total_time, int(steps))[-1],
            exact_rho,
        ))
        qdrift_errors.append(trace_norm_error(
            qd.qdrift_average_path(
                COEFFICIENTS, PAULIS, RHO0, total_time, int(steps)
            )[-1],
            exact_rho,
        ))
        qdrift_bounds.append(qd.qdrift_channel_bound(
            COEFFICIENTS, total_time, int(steps)
        ))

    tail = slice(-6, None)
    log_steps = np.log(step_counts[tail])
    fixed_slope = np.polyfit(log_steps, np.log(fixed_errors)[tail], 1)[0]
    randomized_slope = np.polyfit(log_steps, np.log(randomized_errors)[tail], 1)[0]
    qdrift_slope = np.polyfit(log_steps, np.log(qdrift_errors)[tail], 1)[0]

    assert -1.2 < fixed_slope < -0.8
    assert -2.2 < randomized_slope < -1.8
    assert -1.2 < qdrift_slope < -0.8
    assert np.all(np.asarray(qdrift_errors) <= np.asarray(qdrift_bounds) + 1.0e-12)


def test_seeded_sampling_is_reproducible() -> None:
    first_randomized = rt.sample_randomized_lie_paths(
        TERMS, PSI0, 1.0, 5, 8, np.random.default_rng(20260925)
    )
    second_randomized = rt.sample_randomized_lie_paths(
        TERMS, PSI0, 1.0, 5, 8, np.random.default_rng(20260925)
    )
    first_qdrift = qd.sample_qdrift_paths(
        COEFFICIENTS, PAULIS, PSI0, 1.0, 5, 8,
        np.random.default_rng(20260925),
    )
    second_qdrift = qd.sample_qdrift_paths(
        COEFFICIENTS, PAULIS, PSI0, 1.0, 5, 8,
        np.random.default_rng(20260925),
    )
    assert np.array_equal(first_randomized, second_randomized)
    assert np.array_equal(first_qdrift, second_qdrift)


def test_apply_channel_rejects_invalid_probabilities() -> None:
    with np.testing.assert_raises(ValueError):
        apply_channel(RHO0, [I2, I2], [0.2, 0.2])
