"""Benchmark framework: fair experiments, metrics collection, aggregation."""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

from .classical import greedy_solve, random_feasible_sample, simulated_annealing_solve
from .evaluation import distribution_metrics
from .exact import solve_exact
from .objective import ObjectiveCache
from .problem import SchedulingProblem, random_instance
from .qaoa import QAOAConfig, build_ansatz, run_qaoa
from .qubo import calibrate_penalty


def _distribution_for_params(problem, method, depth, penalty, theta):
    qc, params_list, _ = build_ansatz(problem, method, depth, penalty)
    bound = qc.assign_parameters(dict(zip(params_list, theta)))
    from qiskit.quantum_info import Statevector
    return np.asarray(Statevector.from_instruction(bound).probabilities())


def circuit_resources(problem, method: str, depth: int) -> dict:
    """Pre/post-transpilation circuit metrics (Phase 11)."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap  # noqa: F401  (unused, keep simple)
    qc, _, _ = build_ansatz(problem, method, depth, penalty=1.0 if method == "soft" else None)
    ops = qc.count_ops()
    pre = {
        "depth": qc.depth(),
        "size": sum(v for k, v in ops.items() if k != "barrier"),
        "num_qubits": qc.num_qubits,
        "params": qc.num_parameters,
        "two_q_gates": sum(v for k, v in ops.items() if k in ("cx",)),
        "one_q_gates": sum(v for k, v in ops.items() if k not in ("cx", "barrier")),
    }
    backend_basis = ["rz", "sx", "x", "cx"]
    tqc = transpile(qc, basis_gates=backend_basis, optimization_level=1, seed_transpiler=0)
    tops = tqc.count_ops()
    post = {
        "depth": tqc.depth(),
        "size": sum(v for k, v in tops.items() if k != "barrier"),
        "two_q_gates": int(tops.get("cx", 0)),
        "one_q_gates": sum(v for k, v in tops.items() if k not in ("cx", "barrier")),
    }
    return {"pre": pre, "post": post}


def run_single(
    problem: SchedulingProblem,
    method: str,
    depth: int,
    seed: int,
    instance_seed: int | None,
    maxiter: int = 150,
    restarts: int = 1,
    penalty_safety: float = 1.1,
    run_classical: bool = False,
) -> dict:
    """One fair QAOA run (+ optionally attached classical baselines)."""
    t0 = time.perf_counter()
    cache = ObjectiveCache(problem)
    exact = solve_exact(problem)
    f_opt = exact["objective"]

    penalty = None
    if method == "soft":
        penalty = calibrate_penalty(problem, safety=penalty_safety).recommended

    cfg = QAOAConfig(depth=depth, maxiter=maxiter, restarts=restarts, seed=seed)
    res = run_qaoa(problem, method, cfg, cache=cache, penalty=penalty)

    probs = _distribution_for_params(problem, method, depth, penalty, res.best_params)
    metrics = distribution_metrics(probs, cache, f_opt, penalty=penalty)

    row = {
        "instance_seed": instance_seed,
        "method": method,
        "depth": depth,
        "qaoa_seed": seed,
        "jobs": problem.num_jobs,
        "nodes": problem.num_nodes,
        "penalty": penalty,
        "f_opt": f_opt,
        "best_expectation": res.best_expectation,
        "final_expectation": res.final_expectation,
        "num_evaluations": res.num_evaluations,
        "restarts": restarts,
        "maxiter": maxiter,
        "runtime_qaoa_s": res.runtime_s,
        "optimizer": cfg.optimizer,
        **metrics,
        "total_runtime_s": time.perf_counter() - t0,
    }

    if run_classical:
        g = greedy_solve(problem)
        r = random_feasible_sample(problem, seed=seed, num_samples=10_000)
        sa = simulated_annealing_solve(problem, seed=seed, num_iters=20_000)
        row.update(
            greedy_objective=g["objective"],
            greedy_overload=g["overload"],
            random_best_objective=r["objective"],
            random_mean_objective=r["mean_objective"],
            random_p_optimal=r["p_optimal_empirical"],
            sa_objective=sa["objective"],
            sa_runtime_s=sa["runtime_s"],
        )
    return row


def _run_task(task: dict) -> list[dict]:
    """Multiprocessing worker: build instance deterministically, run rows."""
    problem = random_instance(task["jobs"], task["nodes"], seed=task["instance_seed"])
    row = run_single(
        problem,
        task["method"],
        task["depth"],
        task["seed"],
        task["instance_seed"],
        maxiter=task["maxiter"],
        restarts=task["restarts"],
        run_classical=task["run_classical"],
    )
    return [row]


def run_suite(
    jobs: int,
    nodes: int,
    instances: list[int],
    seeds: list[int],
    depths: list[int],
    maxiter: int = 150,
    restarts: int = 1,
    methods: tuple[str, ...] = ("xy", "soft"),
    workers: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """Statistical suite over instances x seeds x depths x methods.

    workers > 1 uses multiprocessing; each worker rebuilds the deterministic
    instance from its seed, keeping results identical to serial execution.
    """
    tasks = []
    for inst_seed in instances:
        for seed in seeds:
            for depth in depths:
                for method in methods:
                    tasks.append(
                        dict(
                            jobs=jobs, nodes=nodes, instance_seed=inst_seed,
                            seed=seed, depth=depth, method=method,
                            maxiter=maxiter, restarts=restarts,
                            run_classical=(depth == depths[0] and seed == seeds[0]),
                        )
                    )
    if workers > 1:
        import multiprocessing as mp

        with mp.Pool(processes=min(workers, len(tasks))) as pool:
            results = pool.map(_run_task, tasks)
        rows = [r for chunk in results for r in chunk]
    else:
        rows = []
        total = len(tasks)
        for i, task in enumerate(tasks):
            rows.extend(_run_task(task))
            if verbose:
                r = rows[-1]
                print(
                    f"[{i+1}/{total}] inst={task['instance_seed']} {task['method']} "
                    f"p={task['depth']} seed={task['seed']}: "
                    f"p_opt={r['p_optimal']:.4f} mean_feas={r['mean_feasible_objective']:.1f} "
                    f"p_feas={r['p_feasible']:.3f} ({r['total_runtime_s']:.1f}s)",
                    flush=True,
                )
    return rows


# ---- aggregation / statistics (Phase 14) ------------------------------------


def aggregate(rows: list[dict]) -> dict:
    """Mean/median/std/95% CI of key metrics grouped by (method, depth)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    keys = ["method", "depth"]
    stats = {}
    for metric in [
        "approx_ratio_selected",
        "p_feasible",
        "p_optimal",
        "best_feasible_objective",
        "mean_feasible_objective",
        "expectation_value",
        "num_evaluations",
        "runtime_qaoa_s",
    ]:
        g = df.groupby(keys)[metric]
        mean = g.mean()
        std = g.std(ddof=1)
        n = g.count()
        ci95 = 1.96 * std / np.sqrt(n)
        stats[metric] = {
            f"{m}|d{d}": {
                "mean": float(mean.loc[(m, d)]),
                "median": float(g.median().loc[(m, d)]),
                "std": float(std.loc[(m, d)]) if n.loc[(m, d)] > 1 else 0.0,
                "ci95_halfwidth": float(ci95.loc[(m, d)]) if n.loc[(m, d)] > 1 else 0.0,
                "n": int(n.loc[(m, d)]),
            }
            for (m, d) in mean.index
        }
    return stats


def save_results(rows: list[dict], out_prefix: str):
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = f"{out_prefix}.csv"
    df.to_csv(csv_path, index=False)
    stats = aggregate(rows)
    with open(f"{out_prefix}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return csv_path, stats
