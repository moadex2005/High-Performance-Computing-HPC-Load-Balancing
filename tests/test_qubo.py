import numpy as np
import pytest

from zeta_scheduler import SchedulingProblem, random_instance
from zeta_scheduler.ising import qubo_to_ising
from zeta_scheduler.objective import ObjectiveCache
from zeta_scheduler.qubo import (
    add_violation_penalty,
    build_objective_qubo,
    calibrate_penalty,
)


def _all_bitstrings(Q):
    for s in range(1 << Q):
        yield [(s >> q) & 1 for q in range(Q)]


@pytest.mark.parametrize("seed", [1, 2])
def test_qubo_energy_equals_objective_exhaustively(seed):
    """GATE: implemented QUBO == mathematical objective on the ENTIRE space."""
    p = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(25.0, 40.0))
    qubo = build_objective_qubo(p)
    cache = ObjectiveCache(p)
    for bits in _all_bitstrings(p.num_qubits):
        assert abs(qubo.energy(bits) - cache.objective[sum(b << q for q, b in enumerate(bits))]) < 1e-7


def test_penalized_qubo_energy_exhaustively():
    p = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(25.0, 40.0))
    P = 137.5
    qubo = add_violation_penalty(build_objective_qubo(p), p, P)
    cache = ObjectiveCache(p)
    for bits in _all_bitstrings(p.num_qubits):
        s = sum(b << q for q, b in enumerate(bits))
        expected = cache.objective[s] + P * cache.violation[s]
        assert abs(qubo.energy(bits) - expected) < 1e-7


def test_penalty_calibration_sufficiency():
    """GATE 6: at the recommended penalty, ground state is feasible AND optimal."""
    p = random_instance(6, 3, seed=5)
    cal = calibrate_penalty(p)
    cache = ObjectiveCache(p)
    energies = cache.energies(cal.recommended)
    gs = int(np.argmin(energies))
    assert cache.feasible[gs]
    # and it attains the constrained optimum
    f_opt_feas = cache.objective[cache.feasible].min()
    assert abs(cache.objective[gs] - f_opt_feas) < 1e-9


def test_penalty_star_matches_definition():
    p = SchedulingProblem(weights=(15.0, 30.0), capacities=None)
    cal = calibrate_penalty(p)
    cache = ObjectiveCache(p)
    f_min = cache.f_opt_feasible
    worst_ratio = 0.0
    for s in range(1 << p.num_qubits):
        if cache.feasible[s] or cache.violation[s] <= 0:
            continue
        gap = f_min - cache.objective[s]
        if gap > 0:
            worst_ratio = max(worst_ratio, gap / cache.violation[s])
    assert abs(cal.penalty_star - worst_ratio) < 1e-9


def test_old_project_penalty_would_fail_here_documented():
    """Regression guard: a too-small penalty must produce an infeasible ground
    state (mirrors the audited P=120 failure mode)."""
    p = SchedulingProblem(weights=(15.0, 30.0, 10.0, 45.0), capacities=None)
    cal = calibrate_penalty(p)
    tiny_p = cal.penalty_star * 0.5
    cache = ObjectiveCache(p)
    energies = cache.energies(tiny_p)
    gs = int(np.argmin(energies))
    assert not cache.feasible[gs]


def test_ising_conversion_exhaustively():
    """GATE 3/10: Ising energy (+offset) == QUBO energy on every bitstring."""
    p = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(25.0, 40.0))
    qubo = add_violation_penalty(build_objective_qubo(p), p, 120.0)
    ising = qubo_to_ising(qubo)
    for bits in _all_bitstrings(p.num_qubits):
        assert abs(ising.energy(bits) - qubo.energy(bits)) < 1e-7


def test_ising_offset_canonical_convention():
    """Phase 10: reported energy = circuit-domain <H_Z> + offset == scheduling F."""
    from zeta_scheduler.objective import decode_assignment
    p = random_instance(6, 3, seed=7)
    ising = qubo_to_ising(build_objective_qubo(p))
    rng = np.random.default_rng(0)
    for _ in range(50):
        bits = rng.integers(0, 2, size=p.num_qubits).tolist()
        A = decode_assignment(bits, p)
        from zeta_scheduler.objective import objective_value
        assert abs(ising.energy(bits) - objective_value(A, p)) < 1e-6
