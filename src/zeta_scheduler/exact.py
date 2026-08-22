"""Exact ground-truth solver by exhaustive enumeration over feasible assignments."""
from __future__ import annotations

import time

import numpy as np

from .problem import SchedulingProblem


def solve_exact(problem: SchedulingProblem) -> dict:
    """Enumerate every feasible assignment (each job -> one of N nodes).

    Returns optimal objective, all optimal assignments, runtime.
    Number of candidates: num_nodes ** num_jobs (must be tractable).
    """
    t0 = time.perf_counter()
    J, N = problem.num_jobs, problem.num_nodes
    w = np.asarray(problem.weights, dtype=float)
    t = problem.targets

    n_assign = N ** J
    if n_assign > 50_000_000:
        raise RuntimeError(
            f"exact enumeration needs {n_assign} candidate assignments; too many"
        )

    # vectorized enumeration: rows = job index, columns = assignment id
    assign = np.stack(
        np.unravel_index(np.arange(n_assign), (N,) * J), axis=1
    ).astype(np.int64)  # (n_assign, J)
    loads = np.zeros((n_assign, N), dtype=np.float64)
    for j in range(J):
        np.add.at(loads, (np.arange(n_assign), assign[:, j]), w[j])
    objs = ((loads - t) ** 2).sum(axis=1)

    f_opt = float(objs.min())
    opt_ids = np.flatnonzero(objs <= f_opt + 1e-9)
    opt_assignments = [tuple(int(n) for n in assign[i]) for i in opt_ids]
    overload = (
        float(np.maximum(loads[opt_ids] - np.asarray(problem.capacities), 0).sum())
        if problem.capacities is not None
        else 0.0
    )
    runtime = time.perf_counter() - t0
    return {
        "objective": f_opt,
        "assignments": opt_assignments,
        "num_optimal": len(opt_assignments),
        "overload_at_optimum": overload,
        "runtime_s": runtime,
        "num_candidates": int(n_assign),
    }


def enumerate_feasible_stats(problem: SchedulingProblem, max_assignments: int = 4_194_304) -> dict:
    """Statistics of the objective over ALL feasible assignments.

    Used to prove instances are non-degenerate. Falls back to random sampling
    if the full feasible space is too large.
    """
    J, N = problem.num_jobs, problem.num_nodes
    n_assign = N ** J
    if n_assign <= max_assignments:
        res = solve_exact(problem)
        # recompute full value distribution for uniqueness statistics
        assign = np.stack(
            np.unravel_index(np.arange(n_assign), (N,) * J), axis=1
        ).astype(np.int64)
        loads = np.zeros((n_assign, N), dtype=np.float64)
        w = np.asarray(problem.weights, dtype=float)
        t = problem.targets
        for j in range(J):
            np.add.at(loads, (np.arange(n_assign), assign[:, j]), w[j])
        objs = ((loads - t) ** 2).sum(axis=1)
        uniq = np.unique(np.round(objs, 9))
        return {
            "mode": "exhaustive",
            "num_feasible": int(n_assign),
            "num_unique_values": int(len(uniq)),
            "min": float(objs.min()),
            "max": float(objs.max()),
            "mean": float(objs.mean()),
            "median": float(np.median(objs)),
            "num_optimal": res["num_optimal"],
            "optimal_share": res["num_optimal"] / n_assign,
            "optimal_objective": res["objective"],
            "overload_at_optimum": res["overload_at_optimum"],
        }
    rng = np.random.default_rng(0)
    sample = rng.integers(0, N, size=(100_000, J))
    loads = np.zeros((len(sample), N))
    w = np.asarray(problem.weights, dtype=float)
    for j in range(J):
        np.add.at(loads, (np.arange(len(sample)), sample[:, j]), w[j])
    objs = ((loads - problem.targets) ** 2).sum(axis=1)
    return {
        "mode": "sampled",
        "num_feasible_sampled": int(len(sample)),
        "num_unique_values": int(len(np.unique(np.round(objs, 9)))),
        "min": float(objs.min()),
        "max": float(objs.max()),
        "mean": float(objs.mean()),
    }
