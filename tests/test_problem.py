import numpy as np
import pytest

from zeta_scheduler import SchedulingProblem, default_instance, random_instance
from zeta_scheduler.exact import enumerate_feasible_stats, solve_exact


def test_layout_convention():
    p = SchedulingProblem(weights=(3, 5), capacities=(4, 4))
    assert p.num_qubits == 4
    assert p.qubit(0, 0) == 0 and p.qubit(0, 1) == 1
    assert p.qubit(1, 0) == 2 and p.qubit(1, 1) == 3


def test_targets_uniform_vs_capacity_proportional():
    p1 = SchedulingProblem(weights=(15, 30, 10, 45))
    assert np.allclose(p1.targets, np.full(p1.num_nodes, 100 / p1.num_nodes))
    p2 = default_instance()
    kappa = p2.total_work / sum(p2.capacities)
    assert np.allclose(p2.targets, kappa * np.asarray(p2.capacities))


def test_default_instance_non_degenerate():
    """GATE 1b: headline benchmark must not repeat the old degeneracy."""
    rep = enumerate_feasible_stats(default_instance(), max_assignments=10 ** 9)
    assert rep["mode"] == "exhaustive"
    assert rep["num_feasible"] == 4 ** 10
    # old 5x5 had 35 unique values and 3.84% optimal share
    assert rep["num_unique_values"] > 1000
    assert rep["optimal_share"] < 1e-3
    assert rep["overload_at_optimum"] == 0.0


def test_random_instances_deterministic_and_valid():
    a = random_instance(6, 3, seed=1)
    b = random_instance(6, 3, seed=1)
    assert a.weights == b.weights and a.capacities == b.capacities
    c = random_instance(6, 3, seed=2)
    assert c.weights != a.weights
    for inst in (a, c):
        rep = inst.validate()
        assert rep["optimal_share"] < 1e-2
        assert rep["overload_at_optimum"] <= 1e-9


def test_exact_solver_matches_naive_enumeration():
    p = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(30.0, 40.0))
    res = solve_exact(p)

    best, best_a = np.inf, None
    from itertools import product
    for assign in product(range(2), repeat=3):
        A = np.zeros((3, 2), dtype=int)
        for j, n in enumerate(assign):
            A[j, n] = 1
        loads = A.T @ np.asarray(p.weights)
        o = float(((loads - p.targets) ** 2).sum())
        if o < best - 1e-12:
            best, best_a = o, assign
    assert abs(res["objective"] - best) < 1e-9
    assert tuple(res["assignments"][0]) in {tuple(x) for x in [best_a]}
    assert res["num_candidates"] == 8


def test_exact_solver_known_tiny_answer():
    # weights [10], one job, two nodes -> must go anywhere; F = (10-T_n)^2 sums
    p = SchedulingProblem(weights=(10.0,), capacities=None)
    res = solve_exact(p)
    T = 10.0 / p.num_nodes
    assert abs(res["objective"] - ((10 - T) ** 2 + (p.num_nodes - 1) * T ** 2)) < 1e-9


def test_exact_reports_all_optima_count():
    p = SchedulingProblem(weights=(5.0, 5.0), capacities=None)  # symmetric
    res = solve_exact(p)
    N = p.num_nodes
    # any assignment of the two equal jobs to distinct nodes is optimal
    assert res["num_optimal"] == N * (N - 1)


def test_runtime_recorded():
    res = solve_exact(random_instance(6, 3, seed=3))
    assert res["runtime_s"] >= 0
