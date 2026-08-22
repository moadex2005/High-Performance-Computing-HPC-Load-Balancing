"""Generate all README figures from the verified package (no implementation changes).

Figures produced into results/:
    fig_instance.png       default benchmark instance (weights vs capacities/targets)
    fig_init_state.png     broken (audited) vs corrected one-hot state preparation
    fig_prep_dynamics.png  excitation transfer through the staircase (old vs new angles)
    fig_circuit.png        XY-QAOA ansatz circuit diagram (p=1)
    fig_states.png         state-city of a final QAOA state + Bloch view of W-state qubits
    fig_convergence.png    COBYLA convergence, XY vs Soft (p=2)
    fig_quality.png        probability mass over objective values, XY vs Soft
    fig_noise.png          noiseless vs depolarizing-noisy comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector, plot_state_city

from zeta_scheduler import SchedulingProblem, default_instance, random_instance
from zeta_scheduler.circuits import build_initial_state_circuit
from zeta_scheduler.evaluation import distribution_metrics
from zeta_scheduler.exact import solve_exact
from zeta_scheduler.objective import ObjectiveCache
from zeta_scheduler.qaoa import QAOAConfig, run_qaoa
from zeta_scheduler.qubo import calibrate_penalty

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED = "#3b7dd8", "#d8813b", "#2e8b57", "#c0392b"


def _old_notebook_angles(N):
    """The audited (broken) angle schedule from the original XY.ipynb."""
    return [2.0 * np.arccos(np.sqrt((k + 1) / (k + 2))) for k in range(N - 1)]


def _prep_probs(angles, N):
    qc = QuantumCircuit(N)
    qc.x(0)
    for k, t in enumerate(angles):
        qc.rxx(t, k, k + 1)
        qc.ryy(t, k, k + 1)
    sv = Statevector.from_instruction(qc)
    dist = np.zeros(N)
    for idx, p in enumerate(sv.probabilities()):
        bits = [(idx >> m) & 1 for m in range(N)]
        if sum(bits) == 1:
            dist[bits.index(1)] += p
    return dist


def fig_instance():
    p = default_instance()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(p.num_nodes)
    ax.bar(x, np.asarray(p.weights).repeat(0) if False else np.zeros(p.num_nodes), alpha=0)  # spacer
    jobs = np.asarray(p.weights)
    # show job weights as sorted strip + node capacities/targets as bars
    ax2 = ax.twinx()
    ax2.bar(x - 0.2, list(p.capacities), width=0.4, color=BLUE, alpha=0.85, label="capacity $c_n$")
    ax2.bar(x + 0.2, p.targets, width=0.4, color=ORANGE, alpha=0.9, label="target load $T_n$")
    ax2.axhline(p.total_work, color=RED, ls="--", lw=1.2, label=f"total work $W$={p.total_work:.0f}")
    ax.set_xticks(x)
    ax.set_xlabel("node $n$")
    ax2.set_ylabel("load / capacity")
    ax.set_ylabel("job weight (strip below)")
    # job weights strip along the bottom
    for j, w in enumerate(jobs):
        ax.annotate(f"{w:.0f}", (0.995, 0.02 + 0.0 * j), annotation_clip=False)
    handles = [
        plt.Line2D([0], [0], color="gray", marker="s", ls="", label="jobs"),
    ]
    ax2.legend(handles=handles[1:] + ax2.get_legend_handles_labels()[0][:3] +
               [plt.Line2D([], [], marker="s", ls="", color="gray", label=f"job weights {list(map(int,jobs))}")],
               loc="upper left", fontsize=8)
    ax.set_title("Default benchmark instance: 10 jobs → 4 heterogeneous nodes (40 qubits)")
    plt.tight_layout()
    fig.savefig(OUT / "fig_instance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_init_state():
    N = 5
    old = _prep_probs(_old_notebook_angles(N), N)
    new = _prep_probs(list(np.arctan(np.sqrt(np.arange(N - 1, 0, -1)))), N)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, dist, ttl, col in (
        (axes[0], old, "Audited (broken) preparation\n" r"$\theta_k = 2\arccos\sqrt{(k+1)/(k+2)}$", ORANGE),
        (axes[1], new, "Corrected preparation\n" r"$\theta_k = \arctan\sqrt{N-1-k}$", BLUE),
    ):
        ax.bar(range(N), dist, color=col, alpha=0.85)
        ax.axhline(1 / N, color="green", ls="--", lw=1.5, label=f"uniform = {1/N}")
        ax.set_ylim(0, 0.55)
        ax.set_xticks(range(N))
        ax.set_xlabel("node index in job block")
        ax.set_title(ttl)
        ax.legend()
        ax.grid(axis="y", alpha=0.4)
    axes[0].set_ylabel(r"$P(\mathrm{node}=n)$")
    fig.suptitle("One-hot state preparation for one job block (N=5): broken vs fixed",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig_init_state.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # dynamics heatmap: P(excitation at qubit m) after each staircase rung
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, angles, ttl in (
        (axes[0], _old_notebook_angles(N), "broken schedule"),
        (axes[1], list(np.arctan(np.sqrt(np.arange(N - 1, 0, -1)))), "corrected schedule"),
    ):
        M = np.zeros((N, N))  # stage -> per-qubit excitation probability
        qc = QuantumCircuit(N)
        qc.x(0)
        probs_after_x = Statevector.from_instruction(qc).probabilities()
        M[0] = [probs_after_x[i] for i in range(N)]
        for k, t in enumerate(angles):
            qc.rxx(t, k, k + 1)
            qc.ryy(t, k, k + 1)
            pr = Statevector.from_instruction(qc).probabilities()
            M[k + 1] = [pr[i] for i in range(N)]
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(N), ["init"] + [f"after rung {k}" for k in range(N - 1)], fontsize=8)
        ax.set_xticks(range(N), [f"q{m}" for m in range(N)])
        ax.set_title(f"excitation transfer ({ttl})")
    fig.colorbar(im, ax=axes, shrink=0.85, label="P(qubit=1)")
    fig.suptitle("Staircase dynamics: over-rotation pushes amplitude past intermediate nodes",
                 fontweight="bold")
    plt.savefig(OUT / "fig_prep_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_circuit():
    from zeta_scheduler.qaoa import build_ansatz
    tiny = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(25.0, 30.0, 35.0))
    qc_t, _, _ = build_ansatz(tiny, "xy", depth=1)
    figm = qc_t.draw("mpl", style="iqp", fold=140)
    figm.suptitle("XY-QAOA ansatz, p=1 (3 jobs × 3 nodes shown; full suite uses 6×3)", fontsize=10)
    figm.savefig(OUT / "fig_circuit.png", dpi=150, bbox_inches="tight")
    plt.close(figm)


def fig_states():
    """State-city of a final XY-QAOA state (6 qubits = 64 amplitudes) and Bloch view."""
    prob = SchedulingProblem(weights=(15.0, 30.0, 10.0), capacities=(25.0, 30.0, 35.0))
    cache = ObjectiveCache(prob)
    res = run_qaoa(prob, "xy", QAOAConfig(depth=2, maxiter=80, restarts=1, seed=0), cache=cache)
    qc, params, _ = __import__("zeta_scheduler.qaoa", fromlist=["build_ansatz"]).build_ansatz(
        prob, "xy", depth=2)
    bound = qc.assign_parameters(dict(zip(params, res.best_params)))
    sv = Statevector.from_instruction(bound)

    figc = plot_state_city(sv, color=[BLUE, ORANGE], figsize=(11, 4.6))
    figc.suptitle("Final XY-QAOA state (2 jobs × 3 nodes): support confined to feasible subspace",
                  fontsize=10)
    figc.savefig(OUT / "fig_state_city.png", dpi=150, bbox_inches="tight")
    plt.close(figc)

    # Bloch multivector of the uniform W-state block (N=3): each reduced qubit is
    # maximally mixed -> all Bloch vectors at origin (honest caption).
    w3 = QuantumCircuit(3)
    w3.x(0)
    for k, t in enumerate(np.arctan(np.sqrt(np.arange(2, 0, -1)))):
        w3.rxx(t, k, k + 1)
        w3.ryy(t, k, k + 1)
    figb = plot_bloch_multivector(Statevector.from_instruction(w3), figsize=(6.4, 2.6))
    figb.suptitle("Bloch view of each qubit in the prepared W₃ state "
                  "(vectors at origin: entanglement ⇒ uninformative local pictures)", fontsize=9)
    figb.savefig(OUT / "fig_bloch_wstate.png", dpi=150, bbox_inches="tight")
    plt.close(figb)


def fig_convergence():
    prob = random_instance(6, 3, seed=100)
    cache = ObjectiveCache(prob)
    pen = calibrate_penalty(prob).recommended
    out = {}
    for method, penalty in (("xy", None), ("soft", pen)):
        res = run_qaoa(prob, method, QAOAConfig(depth=2, maxiter=150, restarts=1, seed=0),
                       cache=cache, penalty=penalty)
        out[method] = res.expectation_history
    f_star = solve_exact(prob)["objective"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(out["xy"], color=BLUE, marker="o", ms=3, lw=1, label="XY-QAOA ⟨H⟩ trace")
    ax.plot(out["soft"], color=ORANGE, marker="s", ms=3, lw=1, label="Soft-QUBO ⟨H⟩ trace")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer iteration")
    ax.set_ylabel("exact expectation ⟨H⟩ (scheduling domain)")
    ax.set_title(f"COBYLA on instance seed=100, p=2 — exact ⟨H⟩ objective (F* = {f_star:.2f})")
    ax.grid(alpha=0.4)
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT / "fig_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_quality():
    prob = random_instance(6, 3, seed=100)
    cache = ObjectiveCache(prob)
    exact = solve_exact(prob)
    f_opt = exact["objective"]
    pen = calibrate_penalty(prob).recommended
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    from zeta_scheduler.benchmark import _distribution_for_params
    for ax, method, penalty, col in (
        (axes[0], "soft", pen, ORANGE),
        (axes[1], "xy", None, BLUE),
    ):
        cfg = QAOAConfig(depth=2, maxiter=150, restarts=1, seed=0)
        res = run_qaoa(prob, method, cfg, cache=cache, penalty=penalty)
        probs = _distribution_for_params(prob, method, 2, penalty, res.best_params)
        nz = probs > 1e-12
        objs = cache.objective[nz]
        feas = cache.feasible[nz]
        mass = probs[nz]
        edges = np.logspace(np.log10(max(objs.min(), 1e-3)), np.log10(objs.max() * 1.05), 41)
        ids = np.clip(np.digitize(objs, edges) - 1, 0, len(edges) - 2)
        tot_f = np.zeros(len(edges) - 1)
        tot_i = np.zeros(len(edges) - 1)
        for i, f, m in zip(ids, feas, mass):
            (tot_f if f else tot_i)[i] += m
        centers = np.sqrt(edges[:-1] * edges[1:])
        width = np.diff(edges) * 0.9
        ax.bar(centers, tot_i, width=width, color=RED, alpha=0.75,
               label="infeasible states")
        ax.bar(centers, tot_f, width=width, bottom=tot_i, color=GREEN,
               alpha=0.85, label="feasible states")
        ax.axvline(f_opt, color="black", ls="--", lw=1.4, label=f"F* = {f_opt:.2f}")
        ax.set_xscale("log")
        ax.set_xlabel("objective value F(x)")
        ax.set_title(f"{'Soft-QUBO' if method=='soft' else 'XY-QAOA'} final distribution (p=2)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.35)
    axes[0].set_ylabel("probability mass")
    fig.suptitle("Where the quantum distribution sits: feasibility mass vs objective value",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT / "fig_quality.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_noise():
    data = json.loads((OUT / "noise_experiment.json").read_text())["results"]
    keys = ["xy_noiseless", "xy_noisy", "soft_noiseless", "soft_noisy"]
    labels = ["XY\nnoiseless", "XY\nnoisy", "Soft\nnoiseless", "Soft\nnoisy"]
    cols = [BLUE, "#7aa7e0", ORANGE, "#e0b27a"]
    feas = [data[k]["p_feasible"] for k in keys]
    popt = [data[k]["p_optimal"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    axes[0].bar(labels, feas, color=cols)
    axes[0].axhline(1.0, color="gray", ls=":")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("feasibility probability mass")
    axes[0].set_title("Feasibility under depolarizing noise (1e-3 / 1e-2)")
    for i, v in enumerate(feas):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].bar(labels, popt, color=cols)
    axes[1].set_ylabel("P(sample optimal)")
    axes[1].set_title("Optimal-state probability")
    for i, v in enumerate(popt):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.4)
        ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "fig_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("fig_instance");   fig_instance()
    print("fig_init_state"); fig_init_state()
    print("fig_circuit");    fig_circuit()
    print("fig_states");     fig_states()
    print("fig_convergence");fig_convergence()
    print("fig_quality");    fig_quality()
    print("fig_noise");      fig_noise()
    print("done")
