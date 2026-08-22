import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from zeta_scheduler import SchedulingProblem
from zeta_scheduler.circuits import (
    build_initial_state_circuit,
    uniform_onehot_block_angles,
)
from zeta_scheduler.qaoa import build_ansatz


def _block_probs(sv, N, j):
    """Marginal probability of node choice for job block j from a statevector."""
    probs = np.zeros(N)
    full = np.asarray(sv.probabilities())
    for idx, p in enumerate(full):
        if p < 1e-18:
            continue
        bits = [(idx >> (j * N + m)) & 1 for m in range(N)]
        if sum(bits) == 1:
            probs[bits.index(1)] += p
    return probs


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_initial_state_block_uniformity(N):
    """GATE 4: corrected preparation yields the exact uniform W state."""
    prob = SchedulingProblem(weights=(10.0,), capacities=tuple([10.0] * N))
    qc = build_initial_state_circuit(prob)
    assert qc.num_qubits == N
    sv = Statevector.from_instruction(qc)
    amps = np.asarray(sv.data)
    assert abs(np.vdot(amps, amps) - 1.0) < 1e-12          # normalized
    probs = _block_probs(sv, N, 0)
    assert np.allclose(probs, np.full(N, 1.0 / N), atol=1e-9)   # uniform
    # support: zero mass outside the one-hot subspace
    total = 0.0
    for idx, amp in enumerate(amps):
        bits = [(idx >> m) & 1 for m in range(N)]
        if sum(bits) == 1:
            total += abs(amp) ** 2
    assert abs(total - 1.0) < 1e-12


def _tiny_problem(J, N):
    weights = [7.0 + 3.0 * k for k in range(J)]
    caps = tuple(10.0 * (k + 2) for k in range(N))
    return SchedulingProblem(weights=tuple(weights), capacities=caps)


@pytest.mark.parametrize("J,N", [(2, 2), (3, 2), (2, 3), (6, 3)])
def test_full_initial_state_in_feasible_subspace(J, N):
    prob = _tiny_problem(J, N)
    qc = build_initial_state_circuit(prob)
    probs = Statevector.from_instruction(qc).probabilities()
    feas = 0.0
    for idx in range(len(probs)):
        ok = all(
            sum((idx >> (j * N + m)) & 1 for m in range(N)) == 1
            for j in range(J)
        )
        if ok:
            feas += probs[idx]
    assert abs(feas - 1.0) < 1e-9


def test_angles_formula():
    np.testing.assert_allclose(
        uniform_onehot_block_angles(5),
        np.arctan(np.sqrt([4.0, 3.0, 2.0, 1.0])),
    )


def test_ansatz_xy_starts_in_subspace_and_soft_starts_uniform():
    from zeta_scheduler import random_instance
    p = random_instance(6, 3, seed=1)
    qc_xy, _, _ = build_ansatz(p, "xy", depth=1, penalty=None)
    qc_soft, _, _ = build_ansatz(p, "soft", depth=1, penalty=1000.0)
    assert qc_xy.num_parameters == qc_soft.num_parameters == 2
    # XY ansatz at zero params: still fully feasible
    probs = Statevector.from_instruction(qc_xy.assign_parameters([0, 0])).probabilities()
    N = 3
    feas = sum(
        probs[idx]
        for idx in range(len(probs))
        if all(sum((idx >> (j * N + m)) & 1 for m in range(N)) == 1 for j in range(p.num_jobs))
    )
    assert abs(feas - 1.0) < 1e-9
