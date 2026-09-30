"""Numerical simulation of fixed-order and randomized Lie--Trotter formulas."""

from __future__ import annotations

from itertools import permutations
from collections.abc import Sequence

import numpy as np

try:
    from .quantum_utils import apply_channel, compose_in_application_order, hermitian_exp
except ImportError:  # Support importing from a notebook started in this directory.
    from quantum_utils import apply_channel, compose_in_application_order, hermitian_exp


def _validated_terms(terms: np.ndarray) -> np.ndarray:
    terms = np.asarray(terms, dtype=complex)
    if terms.ndim != 3 or terms.shape[1] != terms.shape[2] or len(terms) == 0:
        raise ValueError("terms must have shape (n_terms, dimension, dimension)")
    if not all(np.allclose(term, term.conj().T) for term in terms):
        raise ValueError("all Hamiltonian terms must be Hermitian")
    return terms


def _validated_steps(steps: int) -> int:
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError("steps must be a positive integer")
    return int(steps)


def lie_step_unitary(
    terms: np.ndarray, dt: float, term_order: Sequence[int]
) -> np.ndarray:
    """Return one Lie--Trotter segment in state-application order."""
    terms = _validated_terms(terms)
    order = tuple(int(index) for index in term_order)
    if sorted(order) != list(range(len(terms))):
        raise ValueError("term_order must be a permutation of all term indices")
    gates = [hermitian_exp(terms[index], dt) for index in order]
    return compose_in_application_order(gates)


def _candidate_unitaries(terms: np.ndarray, dt: float) -> tuple[np.ndarray, ...]:
    return tuple(
        lie_step_unitary(terms, dt, order)
        for order in permutations(range(len(terms)))
    )


def fixed_lie_path(
    terms: np.ndarray,
    rho0: np.ndarray,
    total_time: float,
    steps: int,
    term_order: Sequence[int],
) -> np.ndarray:
    """Return density matrices at all fixed-order Lie segment boundaries."""
    terms = _validated_terms(terms)
    steps = _validated_steps(steps)
    rho = np.asarray(rho0, dtype=complex).copy()
    unitary = lie_step_unitary(terms, float(total_time) / steps, term_order)
    path = [rho.copy()]
    for _ in range(steps):
        rho = unitary @ rho @ unitary.conj().T
        path.append(rho.copy())
    return np.asarray(path)


def randomized_lie_average_path(
    terms: np.ndarray, rho0: np.ndarray, total_time: float, steps: int
) -> np.ndarray:
    """Return the exact permutation-averaged density path."""
    terms = _validated_terms(terms)
    steps = _validated_steps(steps)
    rho = np.asarray(rho0, dtype=complex).copy()
    candidates = _candidate_unitaries(terms, float(total_time) / steps)
    path = [rho.copy()]
    for _ in range(steps):
        rho = apply_channel(rho, candidates)
        path.append(rho.copy())
    return np.asarray(path)


def sample_randomized_lie_paths(
    terms: np.ndarray,
    psi0: np.ndarray,
    total_time: float,
    steps: int,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample pure-state paths using an independent permutation per segment."""
    terms = _validated_terms(terms)
    steps = _validated_steps(steps)
    if isinstance(samples, bool) or int(samples) != samples or int(samples) < 1:
        raise ValueError("samples must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    psi0 = np.asarray(psi0, dtype=complex)
    candidates = _candidate_unitaries(terms, float(total_time) / steps)
    paths = np.empty((int(samples), steps + 1, len(psi0)), dtype=complex)
    paths[:, 0] = psi0
    for sample in range(int(samples)):
        state = psi0.copy()
        for segment in range(steps):
            state = candidates[rng.integers(len(candidates))] @ state
            paths[sample, segment + 1] = state
    return paths
