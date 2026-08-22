"""Noise experiment: depolarizing noise, small instance, all four conditions.

noiseless XY / noisy XY / noiseless Soft / noisy Soft at p=1.
Separate from the primary benchmark (kept small on purpose).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from zeta_scheduler import random_instance
from zeta_scheduler.evaluation import distribution_metrics
from zeta_scheduler.exact import solve_exact
from zeta_scheduler.objective import ObjectiveCache
from zeta_scheduler.qaoa import QAOAConfig, build_ansatz, run_qaoa
from zeta_scheduler.qubo import calibrate_penalty


def noisy_simulator(p1: float, p2: float):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=nm)

    def run(qc, params_list, theta, seed):
        tqc = transpile(
            qc, sim, basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=1, seed_transpiler=0,
        )
        bound = tqc.assign_parameters(dict(zip(params_list, theta)))
        bound.save_probabilities()
        res = sim.run(bound, shots=1, seed_simulator=seed).result()
        return np.asarray(res.data(0)["probabilities"])

    return run


def main():
    out_dir = Path(__file__).resolve().parents[1] / "results"
    problem = random_instance(4, 2, seed=3)          # 8 qubits -> density matrix ok
    cache = ObjectiveCache(problem)
    exact = solve_exact(problem)
    f_opt = exact["objective"]
    penalty = calibrate_penalty(problem).recommended
    print(f"instance f*={f_opt:.3f}, soft penalty={penalty:.2f}")

    results = {}
    for method in ("xy", "soft"):
        for label, noise in (("noiseless", None), ("noisy", (1e-3, 1e-2))):
            cfg = QAOAConfig(depth=1, maxiter=60, restarts=1, seed=0)
            if noise is None:
                res = run_qaoa(problem, method, cfg, cache=cache,
                               penalty=penalty if method == "soft" else None)
            else:
                # monkey-run with a custom simulator path
                from zeta_scheduler.qaoa import build_ansatz as ba
                qc, plist, _ = ba(problem, method, 1, penalty if method == "soft" else None)
                energies = cache.energies(penalty if method == "soft" else None)
                runner = noisy_simulator(*noise)

                def expectation(theta, qc=qc, plist=plist, runner=runner, energies=energies):
                    return float(np.dot(runner(qc, plist, theta, 0), energies))

                from scipy.optimize import minimize
                rng = np.random.default_rng(0)
                x0 = rng.uniform(0, 2 * np.pi, size=len(plist))
                opt = minimize(expectation, x0, method="COBYLA",
                               options={"maxiter": 60, "rhobeg": 0.5})
                best_theta = np.asarray(opt.x)

            if noise is not None and 'best_theta' in dir():
                pass
            # final distribution metrics under the SAME noise condition
            if noise is None:
                probs_theta = res.best_params
                from zeta_scheduler.benchmark import _distribution_for_params
                probs = _distribution_for_params(
                    problem, method, 1, penalty if method == "soft" else None, probs_theta)
            else:
                from zeta_scheduler.qaoa import build_ansatz as ba
                qc, plist, _ = ba(problem, method, 1, penalty if method == "soft" else None)
                probs = runner(qc, plist, best_theta, 0)

            m = distribution_metrics(probs, cache, f_opt,
                                     penalty=penalty if method == "soft" else None)
            key = f"{method}_{label}"
            results[key] = {
                "p_feasible": m["p_feasible"],
                "p_optimal": m["p_optimal"],
                "expectation_value": m["expectation_value"],
                "mean_feasible_objective": m["mean_feasible_objective"],
            }
            print(f"{key:>16}: p_feas={m['p_feasible']:.4f} p_opt={m['p_optimal']:.5f} "
                  f"<F>={m['expectation_value']:.2f}")

        # reset for second method iteration cleanliness
        if "best_theta" in dir():
            del best_theta

    out = out_dir / "noise_experiment.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "config": {"jobs": 4, "nodes": 2, "instance_seed": 3, "depth": 1,
                       "depolarizing_1q": 1e-3, "depolarizing_2q": 1e-2,
                       "maxiter": 60, "optimizer": "COBYLA"},
            "f_opt": f_opt,
            "results": results,
        }, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
