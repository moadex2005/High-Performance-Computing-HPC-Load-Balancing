"""Exact QUBO -> Ising conversion with a single canonical offset convention.

Mapping: x_i = (1 - z_i) / 2,  z_i in {-1, +1}.

For E(x) = c + sum_i L_i x_i + sum_{i<j} Qp_ij x_i x_j:

    E = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j

    h_i     = -L_i/2 - (1/4) * sum_{j != i} Qp_{min,max}(i,j)
    J_ij    = Qp_ij / 4
    offset  = c + (1/2) sum_i L_i + (1/4) sum_{i<j} Qp_ij

Canonical reporting convention:
    circuit implements exp(-i * gamma * H_Z) where H_Z = sum h Z + sum J ZZ
    (constants dropped - they are a global phase in the unitary).
    Reported energy := <H_Z> + offset  ==  scheduling-domain energy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .qubo import QUBO


@dataclass(frozen=True)
class IsingModel:
    num_qubits: int
    h: np.ndarray                          # (Q,)
    J: dict[tuple[int, int], float]        # i < j
    offset: float

    def energy(self, bits) -> float:
        """Energy of a computational basis state given as 0/1 bits."""
        b = np.asarray(bits, dtype=float)
        z = 1.0 - 2.0 * b
        e = self.offset + float(np.dot(self.h, z))
        for (i, j), jij in self.J.items():
            e += jij * z[i] * z[j]
        return e


def qubo_to_ising(qubo: QUBO) -> IsingModel:
    Q = qubo.num_qubits
    neighbor = np.zeros(Q)
    h = -qubo.linear / 2.0
    J: dict[tuple[int, int], float] = {}
    quad_sum = 0.0
    for (i, j), q in qubo.quadratic.items():
        assert i < j, "quadratic keys must be upper-triangular"
        J[(i, j)] = q / 4.0
        neighbor[i] += q
        neighbor[j] += q
        quad_sum += q
    h -= neighbor / 4.0
    offset = qubo.constant + qubo.linear.sum() / 2.0 + quad_sum / 4.0
    return IsingModel(num_qubits=Q, h=h, J=J, offset=float(offset))


def ising_to_pauli_labels(ising: IsingModel):
    """Return (labels, coeffs) in Qiskit little-endian string convention.

    Label character k (left to right) corresponds to qubit Q-1-k.
    """
    Q = ising.num_qubits
    labels: list[str] = []
    coeffs: list[float] = []
    rev = lambda idx: "".join("Z" if m == idx else "I" for m in reversed(range(Q)))
    for i in range(Q):
        if abs(ising.h[i]) > 1e-12:
            labels.append(rev(i))
            coeffs.append(float(ising.h[i]))
    for (i, j), v in ising.J.items():
        if abs(v) > 1e-12:
            lab = ["I"] * Q
            lab[i] = "Z"
            lab[j] = "Z"
            labels.append("".join(reversed(lab)))
            coeffs.append(float(v))
    return labels, coeffs
