"""Shared numerical utilities for the one-qubit visualization examples.

This module deliberately contains no plotting code.  The visualization notebook
owns presentation, while the algorithm modules own state evolution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = np.array([X, Y, Z])
PAULI_LABELS = np.array(["X", "Y", "Z"])


def hermitian_exp(matrix: np.ndarray, time: float) -> np.ndarray:
    """Return ``exp(-1j * time * matrix)`` for a Hermitian matrix."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(matrix, matrix.conj().T):
        raise ValueError("matrix must be Hermitian")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.exp(-1j * float(time) * eigenvalues)) @ eigenvectors.conj().T


def pauli_exp(coefficients: Sequence[float], time: float) -> np.ndarray:
    """Return ``exp(-1j * time * coefficients . sigma)`` exactly."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (3,):
        raise ValueError("coefficients must have shape (3,)")
    norm = float(np.linalg.norm(coefficients))
    if norm == 0.0:
        return I2.copy()
    direction_sigma = np.einsum("j,jab->ab", coefficients / norm, PAULIS)
    angle = float(time) * norm
    return np.cos(angle) * I2 - 1j * np.sin(angle) * direction_sigma


def density(state: np.ndarray) -> np.ndarray:
    """Return the density matrix of one state vector."""
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1:
        raise ValueError("state must be a vector")
    return np.outer(state, state.conj())


def bloch(rho: np.ndarray) -> np.ndarray:
    """Return the three-component Bloch vector of a one-qubit state."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2):
        raise ValueError("rho must have shape (2, 2)")
    values = np.array([np.trace(rho @ pauli) for pauli in PAULIS])
    if np.max(np.abs(values.imag)) > 1.0e-10:
        raise ValueError("rho has non-real Pauli expectations")
    return values.real


def trace_norm_error(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Return the trace norm ``||rho - sigma||_1``."""
    rho = np.asarray(rho, dtype=complex)
    sigma = np.asarray(sigma, dtype=complex)
    if rho.shape != sigma.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("rho and sigma must be square matrices of the same shape")
    delta = rho - sigma
    if np.allclose(delta, delta.conj().T):
        return float(np.sum(np.abs(np.linalg.eigvalsh((delta + delta.conj().T) / 2))))
    return float(np.sum(np.linalg.svd(delta, compute_uv=False)))


def apply_channel(
    rho: np.ndarray,
    unitaries: Sequence[np.ndarray],
    probabilities: Sequence[float] | None = None,
) -> np.ndarray:
    """Apply a classical mixture of unitary channels to ``rho``."""
    rho = np.asarray(rho, dtype=complex)
    unitaries = tuple(np.asarray(unitary, dtype=complex) for unitary in unitaries)
    if not unitaries:
        raise ValueError("unitaries must not be empty")
    if probabilities is None:
        weights = np.full(len(unitaries), 1.0 / len(unitaries))
    else:
        weights = np.asarray(probabilities, dtype=float)
        if weights.shape != (len(unitaries),):
            raise ValueError("probabilities must match the number of unitaries")
        if np.any(weights < 0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("probabilities must be nonnegative and sum to one")
    result = np.zeros_like(rho, dtype=complex)
    for probability, unitary in zip(weights, unitaries):
        if unitary.shape != rho.shape:
            raise ValueError("unitary and rho dimensions must match")
        result += probability * (unitary @ rho @ unitary.conj().T)
    return result


def compose_in_application_order(unitaries: Sequence[np.ndarray]) -> np.ndarray:
    """Compose matrices when the first item acts first on the state."""
    unitaries = tuple(np.asarray(unitary, dtype=complex) for unitary in unitaries)
    if not unitaries:
        raise ValueError("unitaries must not be empty")
    dimension = unitaries[0].shape[0]
    result = np.eye(dimension, dtype=complex)
    for unitary in unitaries:
        if unitary.shape != (dimension, dimension):
            raise ValueError("all unitaries must have the same square shape")
        result = unitary @ result
    return result


def exact_density_path(
    hamiltonian: np.ndarray, psi0: np.ndarray, times: Sequence[float]
) -> np.ndarray:
    """Return exact density matrices at the requested times."""
    psi0 = np.asarray(psi0, dtype=complex)
    return np.array([
        density(hermitian_exp(hamiltonian, time) @ psi0) for time in times
    ])


def local_error_coordinates(
    points: np.ndarray,
    exact_point: np.ndarray,
    hamiltonian_axis: np.ndarray,
) -> np.ndarray:
    """Project Bloch-vector errors onto the exact orbit's local tangent plane."""
    points = np.asarray(points, dtype=float)
    exact_point = np.asarray(exact_point, dtype=float)
    axis = np.asarray(hamiltonian_axis, dtype=float)
    if points.shape[-1] != 3 or exact_point.shape != (3,) or axis.shape != (3,):
        raise ValueError("points, exact_point, and hamiltonian_axis must be 3-vectors")
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        raise ValueError("hamiltonian_axis must be nonzero")
    axis = axis / axis_norm
    tangent = np.cross(axis, exact_point)
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm < 1.0e-12:
        reference = np.eye(3)[np.argmin(np.abs(exact_point))]
        tangent = np.cross(reference, exact_point)
        tangent_norm = np.linalg.norm(tangent)
    tangent /= tangent_norm
    transverse = np.cross(exact_point, tangent)
    transverse_norm = np.linalg.norm(transverse)
    if transverse_norm < 1.0e-12:
        raise ValueError("exact_point must be nonzero")
    transverse /= transverse_norm
    differences = points - exact_point
    return np.column_stack((differences @ tangent, differences @ transverse))

