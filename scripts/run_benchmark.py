"""Reproducible benchmark CLI.

Examples:
    python scripts/run_benchmark.py --smoke
    python scripts/run_benchmark.py --jobs 6 --nodes 3 --instances 10 --seeds 3 \
        --depths 1 2 3 --maxiter 150 --restarts 1 --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from zeta_scheduler import default_instance, random_instance
from zeta_scheduler.benchmark import aggregate, circuit_resources, run_suite, save_results
from zeta_scheduler.classical import greedy_solve, random_feasible_sample, simulated_annealing_solve
from zeta_scheduler.exact import solve_exact
from zeta_scheduler.qubo import calibrate_penalty


def main():
    ap = argparse.ArgumentParser(description="ZETA scheduler benchmark")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--instances", type=int, default=10, help="number of random instances")
    ap.add_argument("--instance-seeds", type=int, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3, help="QAOA seeds per instance")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3])
    ap.add_argument("--maxiter", type=int, default=150)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--methods", type=str, nargs="*", default=["xy", "soft"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--noise", action="store_true", help="(reserved) noisy simulation")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(exist_ok=True)

    if args.smoke:
        jobs, nodes, n_inst, seeds, depths, maxiter = 4, 2, 2, [0], [1], 40
        inst_seeds = [0, 1]
    else:
        jobs, nodes = args.jobs, args.nodes
        n_inst, seeds = args.instances, list(range(args.seeds))
        depths, maxiter = args.depths, args.maxiter
        inst_seeds = args.instance_seeds or list(range(n_inst))

    tag = args.out or ("smoke" if args.smoke else f"bench_j{jobs}n{nodes}_i{n_inst}s{args.seeds}")
    print(f"=== suite: {jobs}x{nodes}, instances={inst_seeds}, seeds={seeds}, "
          f"depths={depths}, maxiter={maxiter}, restarts={args.restarts}, workers={args.workers} ===")

    # ---- headline deterministic instance: classical ground truth report ----
    p0 = default_instance()
    ex = solve_exact(p0)
    print("\n--- default 10x4 instance (classical ground truth) ---")
    g = greedy_solve(p0)
    r = random_feasible_sample(p0, seed=0, num_samples=20_000)
    sa = simulated_annealing_solve(p0, seed=0, num_iters=50_000)
    classical_summary = {
        "exact": {k: ex[k] for k in ("objective", "num_optimal", "num_candidates")},
        "greedy": {"objective": g["objective"], "overload": g["overload"]},
        "random_best": r["objective"],
        "random_mean": r["mean_objective"],
        "sa": {"objective": sa["objective"]},
    }
    print(json.dumps(classical_summary, indent=2))

    # ---- QAOA suite --------------------------------------------------------
    rows = run_suite(
        jobs=jobs,
        nodes=nodes,
        instances=inst_seeds,
        seeds=seeds,
        depths=depths,
        maxiter=maxiter,
        restarts=args.restarts,
        methods=tuple(args.methods),
        workers=args.workers,
    )

    csv_path, stats = save_results(rows, str(out_dir / tag))

    # ---- resource table for the suite geometry ----------------------------
    presc = random_instance(jobs, nodes, seed=inst_seeds[0])
    resources = {
        f"{m}_p{d}": circuit_resources(presc, m, d) for m in args.methods for d in depths
    }

    summary_path = out_dir / f"{tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "config": {
                    "jobs": jobs, "nodes": nodes, "instances": inst_seeds,
                    "qaoa_seeds": seeds, "depths": depths,
                    "maxiter": maxiter, "restarts": args.restarts,
                    "methods": args.methods, "optimizer": "COBYLA",
                },
                "default_instance_classical": classical_summary,
                "penalties_note": "per-instance penalty = 1.1 x exact P* from calibrate_penalty",
                "circuit_resources": resources,
                "statistics": stats,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
