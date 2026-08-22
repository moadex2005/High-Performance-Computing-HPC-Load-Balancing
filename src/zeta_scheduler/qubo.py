"""QUBO construction and penalty calibration.

Canonical QUBO convention (documented and tested):
    E(x) = c + sum_i L_i x_i + sum_{i<j} Qp_ij x_i x_j

Soft-QUBO:
    E_soft(x) = F(x) + P * V(x)
    F(x) = sum_n (load_n - T_n)^2
    V(x) = sum_j (sum_n x[j,n] - 1)^2   (= 0 exactly on feasible assignments)

Penalty sufficiency:
    The penalized ground state is guaranteed feasible iff P > P* where
        P* = max over infeasible x of (F*_feasible - F(x)) / V(x),  clipped at 0.
    `calibrate_penalty` computes P* EXACTLY by full enumeration of the
    2^Q binary space whenever tractable; this is the definitive calibration
    (no heuristics for the executed benchmarks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import ObjectiveCache
from .problem import SchedulingProblem


@dataclass(frozen=True)
class QUBO:
    """E(x) = constant + linear[i] x_i + quadratic[(i,j)] x_i x_j (i < j)."""

    num_qubits: int
    constant: float
    linear: np.ndarray                      # (Q,)
    quadratic: dict[tuple[int, int], float]

    def energy(self, bits) -> float:
        b = np.asarray(bits, dtype=float)
        e = self.constant + float(np.dot(self.linear, b))
        for (i, j), q in self.quadratic.items():
            e += q * b[i] * b[j]
        return e


def build_objective_qubo(problem: SchedulingProblem) -> QUBO:
    """QUBO of the canonical objective F(x) = sum_n (load_n - T_n)^2."""
    J, N = problem.num_jobs, problem.num_nodes
    w = np.asarray(problem.weights, dtype=float)
    t = problem.targets
    L = np.zeros(J * N)
    Qp: dict[tuple[int, int], float] = {}
    c = 0.0
    for n in range(N):
        c += t[n] ** 2
        for j in range(J):
            q = problem.qubit(j, n)
            L[q] += w[j] ** 2 - 2.0 * t[n] * w[j]
        for j1 in range(J):
            for k in range(j1 + 1, J):
                pair = (min(problem.qubit(j1, n), problem.qubit(k, n)),
                        max(problem.qubit(j1, n), problem.qubit(k, n)))
                Qp[pair] = Qp.get(pair, 0.0) + 2.0 * w[j1] * w[k]
    return QUBO(num_qubits=J * N, constant=c, linear=L, quadratic=Qp)


def add_violation_penalty(qubo: QUBO, problem: SchedulingProblem, penalty: float) -> QUBO:
    """Add P * sum_j (sum_n x[j,n] - 1)^2 to a QUBO.

    Expansion on binaries (x^2 = x):
        (sum_n x_jn - 1)^2 = -sum_n x_jn + 2 * sum_{n<m} x_jn x_jm + 1
    """
    J, N = problem.num_jobs, problem.num_nodes
    L = qubo.linear.copy()
    Qp = dict(qubo.quadratic)
    c = qubo.constant + penalty * J
    for j in range(J):
        for n in range(N):
            L[problem.qubit(j, n)] -= penalty
        for n1 in range(N):
            for n2 in range(n1 + 1, N):
                pair = (problem.qubit(j, n1), problem.qubit(j, n2))
                Qp[pair] = Qp.get(pair, 0.0) + 2.0 * penalty
    return QUBO(qubo.num_qubits, c, L, Qp)


@dataclass(frozen=True)
class PenaltyCalibration:
    penalty_star: float          # minimal sufficient penalty (exact)
    recommended: float           # recommended usable value (> P*)
    feasible_optimum: float      # min over feasible states of F
    best_infeasible_energy: dict # diagnostics at the recommended penalty


def calibrate_penalty(problem: SchedulingProblem, safety: float = 1.1,
                      max_states: int = 1 << 24) -> PenaltyCalibration:
    """Exact minimal sufficient penalty via exhaustive enumeration.

    Requires 2^Q enumeration (chunked). For larger Q use an explicit penalty
    and validate it empirically instead.
    """
    if problem.num_qubits > 26:
        raise RuntimeError(
            "exact penalty calibration requires <= 26 qubits; "
            "choose a penalty explicitly and validate it"
        )
    cache = ObjectiveCache(problem)
    f_feas_min = cache.f_opt_feasible
    feas = cache.feasible
    obj = cache.objective
    viol = cache.violation
    infeas_obj = obj[~feas]
    infeas_viol = viol[~feas]
    gap = f_feas_min - infeas_obj
    positive = gap > 0
    p_star = float((gap[positive] / infeas_viol[positive]).max()) if positive.any() else 0.0
    rec = p_star * safety + 1e-6
    energies = cache.energies(rec)
    best_infeas = int(np.argmin(np.where(feas, np.inf, energies)))
    best_any = int(np.argmin(energies))
    return PenaltyCalibration(
        penalty_star=p_star,
        recommended=rec,
        feasible_optimum=f_feas_min,
        best_infeasible_energy={
            "energy_at_recommended": float(energies[best_infeas]),
            "is_feasible": bool(feas[best_infeas]),
            "ground_state_is_feasible": bool(feas[best_any]),
            "ground_state_energy": float(energies[best_any]),
        },
    )
