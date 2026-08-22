"""Decoding through the REAL pipeline: circuit -> Aer sampling -> assignment."""
import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from zeta_scheduler import SchedulingProblem, decode_assignment


def test_bitstring_roundtrip_through_aer():
    p = SchedulingProblem(weights=(15.0, 30.0), capacities=None)
    qc = QuantumCircuit(p.num_qubits)
    qc.x(p.qubit(0, 1))   # job0 -> node1
    qc.x(p.qubit(1, 0))   # job1 -> node0
    qc.measure_all()
    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend, optimization_level=1)
    counts = backend.run(tqc, seed_simulator=1, shots=64).result().get_counts()
    assert len(counts) == 1
    bs = next(iter(counts))
    decoded = decode_assignment([int(b) for b in reversed(bs)], p)
    assert decoded.tolist() == [[0, 1], [1, 0]]


def test_transpile_does_not_permute_on_aer():
    p = SchedulingProblem(weights=(15.0, 30.0), capacities=None)
    qc = QuantumCircuit(p.num_qubits)
    qc.x(p.qubit(1, 1))   # logical bit q3
    qc.measure_all()
    backend = AerSimulator(method="statevector")
    tqc = transpile(qc, backend, optimization_level=2)
    counts = backend.run(tqc, seed_simulator=1, shots=32).result().get_counts()
    bs = next(iter(counts))
    bits = [int(b) for b in reversed(bs)]
    assert bits == [0, 0, 0, 1]


def test_qubit_index_matches_job_node_mapping():
    p = SchedulingProblem(weights=(7.0, 9.0, 11.0), capacities=(10.0, 10.0, 10.0))
    # place every job on its LAST node: loads = [0, 0, 27], targets = [9, 9, 9]
    # F = 81 + 81 + 324 = 486
    from zeta_scheduler.objective import objective_value
    N = p.num_nodes
    state = sum(1 << p.qubit(j, N - 1) for j in range(p.num_jobs))
    bits = [(state >> q) & 1 for q in range(p.num_qubits)]
    A = decode_assignment(bits, p)
    assert objective_value(A, p) == pytest.approx(486.0)
