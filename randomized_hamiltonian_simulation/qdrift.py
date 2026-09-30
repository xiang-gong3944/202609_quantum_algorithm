"""Numerical simulation of qDRIFT channels and sampled circuits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from .quantum_utils import apply_channel, hermitian_exp
except ImportError:  # Support importing from a notebook started in this directory.
    from quantum_utils import apply_channel, hermitian_exp


def _validated_steps(steps: int) -> int:
    if isinstance(steps, bool) or int(steps) != steps or int(steps) < 1:
        raise ValueError("steps must be a positive integer")
    return int(steps)


def _validated_inputs(
    coefficients: Sequence[float], operators: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(coefficients, dtype=float)
    operators = np.asarray(operators, dtype=complex)
    if coefficients.ndim != 1 or len(coefficients) == 0:
        raise ValueError("coefficients must be a nonempty vector")
    if operators.ndim != 3 or operators.shape[0] != len(coefficients):
        raise ValueError("operators must have one square matrix per coefficient")
    if operators.shape[1] != operators.shape[2]:
        raise ValueError("operators must be square")
    if not all(np.allclose(operator, operator.conj().T) for operator in operators):
        raise ValueError("operators must be Hermitian")
    return coefficients, operators


def qdrift_distribution(
    coefficients: Sequence[float],
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(lambda, probabilities, signs)`` for signed coefficients."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) == 0:
        raise ValueError("coefficients must be a nonempty vector")
    lam = float(np.sum(np.abs(coefficients)))
    if lam == 0.0:
        raise ValueError("at least one coefficient must be nonzero")
    return lam, np.abs(coefficients) / lam, np.sign(coefficients)


def qdrift_step_unitaries(
    coefficients: Sequence[float],
    operators: np.ndarray,
    total_time: float,
    steps: int,
) -> tuple[np.ndarray, ...]:
    """Return the candidate gates used at each qDRIFT step."""
    coefficients, operators = _validated_inputs(coefficients, operators)
    steps = _validated_steps(steps)
    lam, _, signs = qdrift_distribution(coefficients)
    gate_time = lam * float(total_time) / steps
    return tuple(
        hermitian_exp(sign * operator, gate_time)
        for sign, operator in zip(signs, operators)
    )


def qdrift_average_path(
    coefficients: Sequence[float],
    operators: np.ndarray,
    rho0: np.ndarray,
    total_time: float,
    steps: int,
) -> np.ndarray:
    """Return the exact probability-averaged qDRIFT density path."""
    coefficients, operators = _validated_inputs(coefficients, operators)
    steps = _validated_steps(steps)
    _, probabilities, _ = qdrift_distribution(coefficients)
    candidates = qdrift_step_unitaries(coefficients, operators, total_time, steps)
    rho = np.asarray(rho0, dtype=complex).copy()
    path = [rho.copy()]
    for _ in range(steps):
        rho = apply_channel(rho, candidates, probabilities)
        path.append(rho.copy())
    return np.asarray(path)


def sample_qdrift_paths(
    coefficients: Sequence[float],
    operators: np.ndarray,
    psi0: np.ndarray,
    total_time: float,
    steps: int,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample pure-state paths using independent qDRIFT choices."""
    coefficients, operators = _validated_inputs(coefficients, operators)
    steps = _validated_steps(steps)
    if isinstance(samples, bool) or int(samples) != samples or int(samples) < 1:
        raise ValueError("samples must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    _, probabilities, _ = qdrift_distribution(coefficients)
    candidates = qdrift_step_unitaries(coefficients, operators, total_time, steps)
    psi0 = np.asarray(psi0, dtype=complex)
    paths = np.empty((int(samples), steps + 1, len(psi0)), dtype=complex)
    paths[:, 0] = psi0
    for sample in range(int(samples)):
        state = psi0.copy()
        choices = rng.choice(len(candidates), size=steps, p=probabilities)
        for step, choice in enumerate(choices):
            state = candidates[choice] @ state
            paths[sample, step + 1] = state
    return paths


def qdrift_channel_bound(
    coefficients: Sequence[float], total_time: float, steps: int
) -> float:
    """Return the finite-step channel bound used in the project note."""
    steps = _validated_steps(steps)
    lam, _, _ = qdrift_distribution(coefficients)
    time = float(total_time)
    return float(4 * lam**2 * time**2 / steps * np.exp(2 * lam * abs(time) / steps))

