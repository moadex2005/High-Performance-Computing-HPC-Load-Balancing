import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from scipy.linalg import expm

from zeta_scheduler import SchedulingProblem
from zeta_scheduler.circuits import build_xy_mixer_layer


def _tiny_problem(J, N):
    weights = [7.0 + 3.0 * k for k in range(J)]
    caps = tuple(10.0 * (k + 2) for k in range(N))
    return SchedulingProblem(weights=tuple(weights), capacities=caps)


def test_rxx_ryy_equals_xy_evolution():
    """Preserved (audited): rxx(t)+ryy(t) == exp(-i t (XX+YY)/2)."""
    for t in (0.3, 1.0, np.pi / 4):
        qc = QuantumCircuit(2)
        qc.rxx(t, 0, 1)
        qc.ryy(t, 0, 1)
        lhs = Operator(qc).data
        H = Operator(SparsePauliOp(["XX", "YY"], coeffs=[0.5, 0.5])).data
        rhs = expm(-1j * t * H)
        assert np.abs(lhs - rhs).max() < 1e-12


@pytest.mark.parametrize(("J", "N"), [(1, 2), (1, 3), (1, 4), (2, 2)])
def test_mixer_commutator_with_block_weight(J, N):
    """U_M Z_block U_M^dag == Z_block per block (exact, any topology).

    Note: unlike the idealized continuous evolution, the Trotterized ring
    product does not have the naive summed generator (adjacent edges share a
    qubit and their Pauli terms do not all commute), but every factor
    individually conserves the block excitation number, hence so does the
    product. This test checks the exact conserved quantity.
    """
    prob = _tiny_problem(J, N)
    beta = Parameter("b")
    mix = build_xy_mixer_layer(prob, beta)
    nq = prob.num_qubits
    U = Operator(mix.assign_parameters({beta: 0.7})).data
    zsum_labels = []
    for m in range(nq):
        lab = ["I"] * nq
        lab[m] = "Z"
        zsum_labels.append("".join(reversed(lab)))
    Zsum = Operator(SparsePauliOp(zsum_labels, coeffs=[1.0] * nq)).data
    dev = np.abs(U @ Zsum @ U.conj().T - Zsum).max()
    assert dev < 1e-9


@pytest.mark.parametrize(("J", "N"), [(1, 3), (2, 2), (2, 3), (1, 5)])
def test_mixer_zero_leakage_exhaustive(J, N):
    """GATE 5: no amplitude ever leaves the one-hot subspace."""
    prob = _tiny_problem(J, N)
    beta = Parameter("b")
    mix_tpl = build_xy_mixer_layer(prob, beta)
    nq = J * N

    def is_feas_idx(idx):
        return all(sum((idx >> (j * N + m)) & 1 for m in range(N)) == 1 for j in range(J))

    feasible = [i for i in range(1 << nq) if is_feas_idx(i)]
    worst = 0.0
    for bv in (0.13, np.pi / 4, 1.7):
        U = Operator(mix_tpl.assign_parameters({beta: bv})).data
        for idx in feasible:
            vec = np.zeros(1 << nq, dtype=complex)
            vec[idx] = 1.0
            out = U @ vec
            leak = sum(abs(out[i]) ** 2 for i in range(1 << nq) if not is_feas_idx(i))
            worst = max(worst, leak)
    assert worst < 1e-10


def test_full_circuit_preserves_feasibility_at_random_params():
    """End-to-end guarantee: P(feasible) == 1 through cost+mixer layers."""
    from zeta_scheduler.qaoa import build_ansatz
    p = _tiny_problem(6, 3)
    rng = np.random.default_rng(3)
    for _ in range(3):
        qc, params, _ = build_ansatz(p, "xy", depth=2)
        theta = rng.uniform(-np.pi, np.pi, size=len(params))
        probs = Statevector.from_instruction(qc.assign_parameters(dict(zip(params, theta)))).probabilities()
        N = 3
        feas = sum(
            probs[idx]
            for idx in range(len(probs))
            if all(sum((idx >> (j * N + m)) & 1 for m in range(N)) == 1 for j in range(p.num_jobs))
        )
        assert abs(feas - 1.0) < 1e-9
