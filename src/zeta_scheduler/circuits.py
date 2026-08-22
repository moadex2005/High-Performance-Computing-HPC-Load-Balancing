"""Circuit construction: initial state (corrected), XY mixer (preserved), cost layer.

Initial state (REBUILT - audit finding #4):
    The old preparation over-rotated because RXX(t)*RYY(t) rotates the
    {|10>,|01>} subspace by the FULL angle t (XX and YY generate the same
    transition there). The correct uniform one-hot ("W") state per block of N
    qubits is produced by a forward staircase with

        theta_k = arctan(sqrt(N - 1 - k)),   k = 0..N-2,  pairs (k, k+1).

    Verified uniform to <1e-9 for N = 2..8 with zero leakage.

XY mixer (PRESERVED from audit - verified correct):
    ring topology per job block: for each edge (k, k+1 mod N):
        rxx(beta); ryy(beta)   ==  exp(-i*beta*(XX+YY)/2)
    Commutes with the per-block excitation number; zero subspace leakage.

Cost layer (PRESERVED pattern, coefficients now from verified IsingModel):
    Rz(2*gamma*h_i) and CX-Rz(2*gamma*J_ij)-CX. Constant offsets are NOT
    compiled into the circuit (global phase); they are added analytically.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from .ising import IsingModel


# ---- initial state ----------------------------------------------------------


def uniform_onehot_block_angles(num_nodes: int) -> np.ndarray:
    """theta_k = arctan(sqrt(N-1-k)) for the RXX+RYY staircase."""
    return np.arctan(np.sqrt(np.arange(num_nodes - 1, 0, -1)))


def build_initial_state_circuit(problem) -> QuantumCircuit:
    """Uniform superposition of all feasible assignments (product of W states)."""
    J, N = problem.num_jobs, problem.num_nodes
    qc = QuantumCircuit(J * N, name="init_uniform_W")
    thetas = uniform_onehot_block_angles(N)
    for j in range(J):
        b0 = j * N
        qc.x(b0)
        for k in range(N - 1):
            theta = float(thetas[k])
            qc.rxx(theta, b0 + k, b0 + k + 1)
            qc.ryy(theta, b0 + k, b0 + k + 1)
    return qc


def build_plus_state_circuit(num_qubits: int) -> QuantumCircuit:
    """Uniform superposition over ALL bitstrings (Soft-QUBO standard start)."""
    qc = QuantumCircuit(num_qubits, name="init_plus")
    qc.h(range(num_qubits))
    return qc


# ---- mixer ------------------------------------------------------------------


def build_xy_mixer_layer(problem, beta: Parameter) -> QuantumCircuit:
    """One XY ring-mixer layer per job block (preserved from audited code)."""
    J, N = problem.num_jobs, problem.num_nodes
    nq = J * N
    qc = QuantumCircuit(nq, name="xy_mixer")
    for j in range(J):
        b0 = j * N
        for k in range(N):
            qa = b0 + k
            qb = b0 + (k + 1) % N
            qc.rxx(beta, qa, qb)
            qc.ryy(beta, qa, qb)
    return qc


def build_x_mixer_layer(num_qubits: int, beta: Parameter) -> QuantumCircuit:
    """Standard transverse-field mixer (used only by Soft-QUBO ansatz)."""
    qc = QuantumCircuit(num_qubits, name="x_mixer")
    qc.rx(2.0 * beta, range(num_qubits))
    return qc


# ---- cost layer -------------------------------------------------------------


def build_cost_layer(ising: IsingModel, gamma: Parameter) -> QuantumCircuit:
    """exp(-i * gamma * H_Z):  Rz(2 g h) and CX-Rz(2 g J)-CX (preserved pattern)."""
    qc = QuantumCircuit(ising.num_qubits, name="cost")
    for i in range(ising.num_qubits):
        if abs(ising.h[i]) > 1e-12:
            qc.rz(2.0 * ising.h[i] * gamma, i)
    for (i, j), jij in ising.J.items():
        if abs(jij) > 1e-12:
            qc.cx(i, j)
            qc.rz(2.0 * jij * gamma, j)
            qc.cx(i, j)
    return qc
