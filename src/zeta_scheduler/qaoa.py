"""Fair QAOA experiments for Soft-QUBO and XY-QAOA.

Design decisions (fixing audit findings):

* IDENTICAL optimization budget, initialization policy, seeds, stopping rule
  and parameter-count for both methods (same p -> same 2p parameters).
* Optimizer objective: EXACT expectation value <H> of each method's own cost
  Hamiltonian, computed from the full state distribution (no shot noise).
  This fixes the old XY bug of optimizing a sample-min statistic and the old
  Soft bug of under-training (maxiter=8).
* Final solutions are evaluated from the SAME full distribution with the SAME
  selection rule for both methods.
* Canonical energy convention: reported energies are scheduling-domain values
  (Ising energy + analytic offset == F(x) [+ P*V(x)] exactly; unit-tested).

Method definitions:
  xy   : uniform-W init (feasible subspace), cost = Ising(F),        XY-ring mixer
  soft : |+...+> init,                cost = Ising(F + P*V),         X mixer
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit.circuit import Parameter
from scipy.optimize import OptimizeResult, minimize

from .circuits import (
    build_cost_layer,
    build_initial_state_circuit,
    build_plus_state_circuit,
    build_x_mixer_layer,
    build_xy_mixer_layer,
)
from .ising import qubo_to_ising
from .objective import ObjectiveCache
from .problem import SchedulingProblem
from .qubo import add_violation_penalty, build_objective_qubo


@dataclass
class QAOAConfig:
    depth: int = 1
    maxiter: int = 150
    restarts: int = 1
    seed: int = 0
    optimizer: str = "COBYLA"
    rhobeg: float = 0.5
    penalty: float | None = None          # required for method="soft"


@dataclass
class QAOARunResult:
    method: str
    depth: int
    seed: int
    best_params: np.ndarray
    best_expectation: float               # optimized value (scheduling domain)
    final_expectation: float
    num_evaluations: int
    restarts_used: int
    runtime_s: float
    expectation_history: list[float] = field(default_factory=list)


def build_ansatz(problem: SchedulingProblem, method: str, depth: int, penalty=None):
    """Return (circuit, params, ising, energy_array_fn_inputs).

    Both methods get exactly 2*p parameters: gammas then betas.
    """
    if method not in ("xy", "soft"):
        raise ValueError(method)
    qubo_f = build_objective_qubo(problem)
    if method == "soft":
        if penalty is None:
            raise ValueError("method='soft' requires a calibrated penalty")
        ising = qubo_to_ising(add_violation_penalty(qubo_f, problem, penalty))
        init = build_plus_state_circuit(problem.num_qubits)
    else:
        ising = qubo_to_ising(qubo_f)
        init = build_initial_state_circuit(problem)

    gammas = [Parameter(f"g{k}") for k in range(depth)]
    betas = [Parameter(f"b{k}") for k in range(depth)]
    qc = init.copy()
    for k in range(depth):
        qc.compose(build_cost_layer(ising, gammas[k]), inplace=True)
        if method == "xy":
            qc.compose(build_xy_mixer_layer(problem, betas[k]), inplace=True)
        else:
            qc.compose(build_x_mixer_layer(problem.num_qubits, betas[k]), inplace=True)
    return qc, gammas + betas, ising


def run_qaoa(
    problem: SchedulingProblem,
    method: str,
    config: QAOAConfig,
    cache: ObjectiveCache | None = None,
    simulator=None,
    penalty: float | None = None,
) -> QAOARunResult:
    """Optimize QAOA with the EXACT expectation value as optimizer objective.

    `penalty` overrides config.penalty (used internally by the benchmark).
    `simulator` allows injecting a noisy AerSimulator; default noiseless Aer.
    """
    t0 = time.perf_counter()
    cache = cache or ObjectiveCache(problem)
    if method == "soft" and penalty is None:
        penalty = config.penalty
    qc, params_list, _ising = build_ansatz(problem, method, config.depth, penalty)

    # scheduling-domain diagonal energies (== ising energy incl. offset; tested)
    energies = cache.energies(penalty)

    # Fast path: transpile once, evaluate distributions with Aer's C++ engine.
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    sim = simulator if simulator is not None else AerSimulator(method="statevector")
    tqc = transpile(qc, sim, optimization_level=1, seed_transpiler=0)

    def distribution(theta: np.ndarray) -> np.ndarray:
        bound = tqc.assign_parameters(dict(zip(params_list, theta)))
        bound.save_probabilities()
        result = sim.run(bound, shots=1, seed_simulator=config.seed).result()
        return np.asarray(result.data(0)["probabilities"])

    def expectation(theta: np.ndarray) -> float:
        return float(np.dot(distribution(theta), energies))

    # ---- identical initialization policy for both methods -------------------
    rng = np.random.default_rng(config.seed)
    starts = [
        rng.uniform(0.0, 2.0 * np.pi, size=len(params_list))
        for _ in range(config.restarts)
    ]

    history: list[float] = []
    best_theta, best_val = starts[0], np.inf
    nfev_total = 0

    for x0 in starts:
        res: OptimizeResult = minimize(
            expectation, x0,
            method=config.optimizer,
            options={"maxiter": config.maxiter, "rhobeg": config.rhobeg},
            callback=lambda xk: history.append(expectation(xk)),
        )
        val = float(res.fun)
        nfev_total += int(res.nfev)
        if val < best_val:
            best_val, best_theta = val, np.asarray(res.x, dtype=float)

    final_val = expectation(best_theta)
    return QAOARunResult(
        method=method,
        depth=config.depth,
        seed=config.seed,
        best_params=best_theta,
        best_expectation=float(best_val),
        final_expectation=float(final_val),
        num_evaluations=nfev_total,
        restarts_used=config.restarts,
        runtime_s=time.perf_counter() - t0,
        expectation_history=[float(v) for v in history],
    )
