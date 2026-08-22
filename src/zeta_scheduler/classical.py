"""Classical baselines: exact, greedy, uniform-random-feasible, simulated annealing.

All baselines optimize/score the SAME canonical objective F(x).
Random sampling is UNIFORM OVER FEASIBLE ASSIGNMENTS (never raw bitstrings).
"""
from __future__ import annotations

import time

import numpy as np

from .problem import SchedulingProblem


def greedy_solve(problem: SchedulingProblem) -> dict:
    """Sort jobs by descending weight; assign each to the currently least-loaded node."""
    t0 = time.perf_counter()
    J, N = problem.num_jobs, problem.num_nodes
    w = np.asarray(problem.weights)
    loads = np.zeros(N)
    assign = np.full(J, -1, dtype=int)
    for j in np.argsort(-w, kind="stable"):
        n = int(np.argmin(loads))
        assign[j] = n
        loads[n] += w[j]
    from .objective import decode_assignment
    a = decode_assignment(np.eye(N, dtype=int)[assign].reshape(-1), problem)
    return _finish(problem, a, time.perf_counter() - t0)


def random_feasible_sample(problem: SchedulingProblem, seed: int, num_samples: int = 10_000) -> dict:
    """Uniform samples over feasible assignments; returns best and distribution stats."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    J, N = problem.num_jobs, problem.num_nodes
    assigns = rng.integers(0, N, size=(num_samples, J))
    objs = np.empty(num_samples)
    for i in range(num_samples):
        a = np.zeros((J, N), dtype=int)
        a[np.arange(J), assigns[i]] = 1
        objs[i] = _obj_of(a, problem)
    best_i = int(np.argmin(objs))
    a_best = np.zeros((J, N), dtype=int)
    a_best[np.arange(J), assigns[best_i]] = 1
    res = _finish(problem, a_best, time.perf_counter() - t0)
    res.update(
        num_samples=num_samples,
        mean_objective=float(objs.mean()),
        std_objective=float(objs.std()),
        p_optimal_empirical=float((objs <= res["objective"] + 1e-9).mean()),
    )
    return res


def simulated_annealing_solve(
    problem: SchedulingProblem,
    seed: int,
    num_iters: int = 20_000,
    init_temp: float | None = None,
    num_restarts: int | None = None,
) -> dict:
    """Multi-start SA over feasible assignments (single-job reassignment moves).

    The initial temperature is calibrated from the spread of F over random
    feasible assignments (not from an arbitrary constant), which makes the
    schedule instance-adaptive.
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    J, N = problem.num_jobs, problem.num_nodes
    w = np.asarray(problem.weights, dtype=float)
    t = problem.targets

    # temperature scale from random-sampling statistics
    probe = rng.integers(0, N, size=(300, J))
    probe_f = np.empty(len(probe))
    for i, pr in enumerate(probe):
        loads = np.bincount(pr, weights=w, minlength=N)
        probe_f[i] = ((loads - t) ** 2).sum()
    temp_scale = float(np.std(probe_f)) or 1.0

    if num_restarts is None:
        num_restarts = max(1, num_iters // 4000)
    iters_per_restart = max(num_iters // num_restarts, 100)

    def f(loads_vec):
        return float(((loads_vec - t) ** 2).sum())

    best_assign = rng.integers(0, N, size=J).astype(int)
    best_f = f(np.bincount(best_assign, weights=w, minlength=N))
    accepted_total = 0
    tried_total = 0

    eye = np.eye(N)
    for r in range(num_restarts):
        cur = rng.integers(0, N, size=J).astype(int)
        loads = np.bincount(cur, weights=w, minlength=N)
        f_cur = f(loads)
        T0 = temp_scale * (10.0 ** (-2.0 * r / max(num_restarts - 1, 1)))
        temps = T0 * (1e-4) ** (np.arange(iters_per_restart) / max(iters_per_restart - 1, 1))
        for it in range(iters_per_restart):
            j = int(rng.integers(J))
            n_new = int(rng.integers(N))
            old = cur[j]
            if n_new == old:
                continue
            tried_total += 1
            delta_vec = eye[n_new] * w[j] - eye[old] * w[j]
            f_new = f(loads + delta_vec)
            d = f_new - f_cur
            if d <= 0 or rng.random() < np.exp(-d / temps[it]):
                cur[j], loads, f_cur = n_new, loads + delta_vec, f_new
                accepted_total += 1
                if f_cur < best_f:
                    best_f = f_cur
                    best_assign = cur.copy()

    a = np.zeros((J, N), dtype=int)
    a[np.arange(J), best_assign] = 1
    res = _finish(problem, a, time.perf_counter() - t0)
    res.update(
        num_iterations=num_iters,
        acceptance_rate=accepted_total / max(tried_total, 1),
        num_restarts=num_restarts,
    )
    return res


# ---- helpers ----------------------------------------------------------------


def _obj_of(assign_matrix: np.ndarray, problem: SchedulingProblem) -> float:
    from .objective import objective_value
    return objective_value(assign_matrix, problem)


def _finish(problem: SchedulingProblem, assign_matrix: np.ndarray, runtime: float) -> dict:
    from .objective import is_feasible, objective_value, overload_value
    return {
        "assignment": assign_matrix,
        "objective": objective_value(assign_matrix, problem),
        "feasible": is_feasible(assign_matrix),
        "overload": overload_value(assign_matrix, problem),
        "runtime_s": runtime,
    }
