import numpy as np
import pytest

from zeta_scheduler import (
    SchedulingProblem,
    default_instance,
    decode_assignment,
    is_feasible,
    objective_value,
    random_instance,
)
from zeta_scheduler.objective import ObjectiveCache, overload_value, violation_value


def _assignment_from_nodes(nodes, N):
    a = np.zeros((len(nodes), N), dtype=int)
    a[np.arange(len(nodes)), nodes] = 1
    return a


def test_hand_computed_objective():
    """Manually verifiable example: 2 jobs [15,30] on 2 nodes."""
    p = SchedulingProblem(weights=(15.0, 30.0))
    # both on node 0: loads (45, 0); T = 22.5 -> F = 22.5^2 + 22.5^2 = 1012.5
    a = _assignment_from_nodes([0, 0], 2)
    assert abs(objective_value(a, p) - 1012.5) < 1e-9
    assert is_feasible(a) and violation_value(a) == 0
    # split: loads (15, 30) -> F = 7.5^2 + 7.5^2 = 112.5
    a = _assignment_from_nodes([0, 1], 2)
    assert abs(objective_value(a, p) - 112.5) < 1e-9


def test_violation_detection():
    p = SchedulingProblem(weights=(15.0, 30.0))
    empty = np.zeros((2, 2), dtype=int)          # job 1 unassigned
    dupl = np.array([[1, 1], [1, 0]])            # job 0 assigned twice
    ok = _assignment_from_nodes([1, 0], 2)
    assert not is_feasible(empty)
    assert violation_value(empty) == pytest.approx(2.0)
    assert not is_feasible(dupl)
    assert violation_value(dupl) == pytest.approx(1.0)
    assert is_feasible(ok) and violation_value(ok) == 0.0


def test_overload_metric():
    p = SchedulingProblem(weights=(60.0,), capacities=(50.0, 50.0))
    a = _assignment_from_nodes([0], 2)
    assert overload_value(a, p) == 10.0
    a = _assignment_from_nodes([1], 2)
    assert overload_value(a, p) == 10.0


def test_decode_roundtrip():
    p = default_instance()
    bits = np.random.default_rng(0).integers(0, 2, size=p.num_qubits)
    A = decode_assignment(bits, p)
    q = 0
    for j in range(p.num_jobs):
        for n in range(p.num_nodes):
            assert A[j, n] == int(bits[q])
            q += 1


def test_cache_matches_scalar_api_exhaustively():
    p = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(25.0, 40.0))
    cache = ObjectiveCache(p)
    for s in range(1 << p.num_qubits):
        bits = [(s >> q) & 1 for q in range(p.num_qubits)]
        A = decode_assignment(bits, p)
        from zeta_scheduler.objective import objective_value as ov, violation_value as vv
        from zeta_scheduler.objective import is_feasible as iff
        assert bool(cache.feasible[s]) == iff(A)
        assert abs(cache.objective[s] - ov(A, p)) < 1e-8
        assert abs(cache.violation[s] - vv(A)) < 1e-8
        if p.capacities is not None:
            assert abs(cache.overload[s] - overload_value(A, p)) < 1e-8


def test_cache_optimum_consistent_with_exact_solver():
    from zeta_scheduler.exact import solve_exact
    p = random_instance(6, 3, seed=11)
    cache = ObjectiveCache(p)
    res = solve_exact(p)
    assert abs(cache.f_opt_feasible - res["objective"]) < 1e-6
