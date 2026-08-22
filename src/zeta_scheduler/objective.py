"""Canonical objective evaluation (single source of truth for ALL methods).

Conventions (audited and preserved):
    bit q of a flat bit-vector corresponds to variable x[j, n] with
    q = j * num_nodes + n.  Measured Qiskit bitstrings are little-endian and
    must be reversed before interpretation.
"""
from __future__ import annotations

import numpy as np

from .problem import SchedulingProblem


# ---- scalar API -------------------------------------------------------------


def decode_assignment(bits, problem: SchedulingProblem) -> np.ndarray:
    """Flat bit sequence (logical order, index = j*num_nodes+n) -> (J, N) matrix."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or bits.size != problem.num_qubits:
        raise ValueError(f"expected {problem.num_qubits} bits, got shape {bits.shape}")
    return bits.reshape(problem.num_jobs, problem.num_nodes).astype(int)


def row_sums(assignment: np.ndarray) -> np.ndarray:
    return assignment.sum(axis=1)


def is_feasible(assignment: np.ndarray) -> bool:
    """True iff every job is assigned to exactly one node."""
    return bool(np.all(row_sums(assignment) == 1))


def node_loads(assignment: np.ndarray, weights) -> np.ndarray:
    return assignment.T @ np.asarray(weights, dtype=float)


def objective_value(assignment: np.ndarray, problem: SchedulingProblem) -> float:
    """F(x) = sum_n (load_n - T_n)^2   (canonical scheduling objective)."""
    loads = node_loads(assignment, problem.weights)
    return float(np.sum((loads - problem.targets) ** 2))


def violation_value(assignment: np.ndarray) -> float:
    """V(x) = sum_j (sum_n x[j,n] - 1)^2  (assignment-constraint violation)."""
    rs = row_sums(assignment)
    return float(np.sum((rs - 1.0) ** 2))


def overload_value(assignment: np.ndarray, problem: SchedulingProblem) -> float:
    """total capacity overload  sum_n max(0, load_n - cap_n); 0 without capacities."""
    if problem.capacities is None:
        return 0.0
    loads = node_loads(assignment, problem.weights)
    return float(np.sum(np.maximum(loads - np.asarray(problem.capacities), 0.0)))


def penalized_objective_value(assignment: np.ndarray, problem: SchedulingProblem, penalty: float) -> float:
    """Soft-QUBO energy in scheduling units: F(x) + P * V(x)."""
    return objective_value(assignment, problem) + penalty * violation_value(assignment)


# ---- vectorized API over flat state indices ---------------------------------
#
# State convention: integer s in [0, 2^Q) encodes a bitstring where bit q of s
# equals x[j, n] with q = j * num_nodes + n (matches Qiskit's little-endian
# probability indexing).


def _bits_matrix(indices: np.ndarray, num_qubits: int) -> np.ndarray:
    """(..., Q) uint8 bits of the given flat indices."""
    idx = np.asarray(indices, dtype=np.int64)[..., None]
    return ((idx >> np.arange(num_qubits, dtype=np.int64)[None, :]) & 1).astype(np.uint8)


class ObjectiveCache:
    """Precomputed per-state arrays over the full 2^Q space.

    For a problem with Q qubits this stores, for every computational basis
    state: assignment-feasibility, canonical objective, violation, and
    (optionally) optimality. Memory: ~Q * 2^Q bytes.
    """

    def __init__(self, problem: SchedulingProblem):
        self.problem = problem
        self.Q = problem.num_qubits
        size = 1 << self.Q
        self.feasible = np.zeros(size, dtype=bool)
        self.objective = np.zeros(size, dtype=np.float64)
        self.violation = np.zeros(size, dtype=np.float64)
        self.overload = np.zeros(size, dtype=np.float64)

        w = np.asarray(problem.weights, dtype=float)
        t = problem.targets
        caps = (
            np.asarray(problem.capacities, dtype=float)
            if problem.capacities is not None
            else None
        )
        J, N = problem.num_jobs, problem.num_nodes
        chunk = 1 << 20
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            idx = np.arange(start, stop, dtype=np.int64)
            b = _bits_matrix(idx, self.Q).reshape(-1, J, N).astype(np.float64)
            rows = b.sum(axis=2)                      # (c, J)
            loads = np.einsum("bjn,j->bn", b, w)      # (c, N)
            self.feasible[start:stop] = np.all(rows == 1, axis=1)
            self.objective[start:stop] = ((loads - t) ** 2).sum(axis=1)
            self.violation[start:stop] = ((rows - 1.0) ** 2).sum(axis=1)
            if caps is not None:
                self.overload[start:stop] = np.maximum(loads - caps, 0.0).sum(axis=1)

        self.f_opt_feasible = float(self.objective[self.feasible].min())
        self.optimal_mask = self.feasible & (
            self.objective <= self.f_opt_feasible + 1e-9
        )

    def energies(self, penalty: float | None = None) -> np.ndarray:
        """Diagonal Hamiltonian energies over all states.

        penalty=None -> canonical objective F(x)
        penalty=P    -> soft-QUBO energies F(x) + P*V(x)
        """
        if penalty is None:
            return self.objective
        return self.objective + penalty * self.violation
