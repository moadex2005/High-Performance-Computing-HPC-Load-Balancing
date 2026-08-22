"""Solution-quality metrics computed from full state distributions.

All metrics are reported in the canonical scheduling domain:
  * feasibility      = one-hot assignment per job
  * objective        = F(x) = sum_n (load_n - T_n)^2
  * optimal          = F(x) <= F* + tol (F* from exact solver)
Capacity overload is reported as a separate diagnostic; it is never hidden.
"""
from __future__ import annotations

import numpy as np

from .objective import ObjectiveCache


def distribution_metrics(
    probs: np.ndarray,
    cache: ObjectiveCache,
    f_opt: float,
    penalty: float | None = None,
    tol: float = 1e-9,
) -> dict:
    """Metrics of a full probability distribution over basis states."""
    feas = cache.feasible
    obj = cache.objective
    viol = cache.violation
    over = cache.overload
    opt_mask = cache.optimal_mask

    p_feas = float(min(1.0, probs[feas].sum()))
    p_opt = float(min(1.0, probs[opt_mask].sum()))

    # best / mean over feasible states present in the distribution
    if p_feas > 0:
        # lowest-objective feasible state with nonzero probability
        idxs = np.flatnonzero(feas & (probs > 0))
        o_nz = obj[idxs]
        best_feasible_obj = float(o_nz.min())
        mean_feasible_obj = float(np.average(o_nz, weights=probs[idxs]))
    else:
        best_feasible_obj = float("nan")
        mean_feasible_obj = float("nan")

    nz_all = probs > 0
    mean_obj_all = float(np.average(obj[nz_all], weights=probs[nz_all])) if nz_all.any() else float("nan")
    std_obj_all = (
        float(np.sqrt(np.average((obj[nz_all] - mean_obj_all) ** 2, weights=probs[nz_all])))
        if nz_all.any()
        else float("nan")
    )

    # most likely measured bitstring
    ml = int(np.argmax(probs))
    # "the solution" under the fixed selection rule: lowest-objective feasible
    # state with nonzero probability; fall back to most likely state overall.
    if p_feas > 0:
        idxs = np.flatnonzero(feas & (probs > 0))
        sel = idxs[int(np.argmin(obj[idxs]))]
        selection_rule = "best-feasible-in-distribution"
    else:
        sel = ml
        selection_rule = "most-likely-state (no feasible state sampled)"
    selected_obj = float(obj[sel])
    selected_overload = float(over[sel])

    approx_best = selected_obj / f_opt if f_opt > 0 else float("nan")
    approx_mean_feasible = (
        f_opt / mean_feasible_obj
        if f_opt > 0 and np.isfinite(mean_feasible_obj) and mean_feasible_obj > 0
        else float("nan")
    )

    return {
        "selected_objective": selected_obj,
        "selection_rule": selection_rule,
        "approx_ratio_selected": float(approx_best),
        "approx_ratio_mean_feasible": float(approx_mean_feasible),
        "p_feasible": p_feas,
        "p_infeasible": 1.0 - p_feas,
        "p_optimal": p_opt,
        "best_feasible_objective": best_feasible_obj,
        "mean_feasible_objective": mean_feasible_obj,
        "mean_objective_all": mean_obj_all,
        "std_objective_all": std_obj_all,
        "most_likely_objective": float(obj[ml]),
        "most_likely_feasible": bool(feas[ml]),
        "selected_overload": selected_overload,
        "expectation_value": float(np.dot(probs, cache.energies(penalty))),
    }


def finite_sample_metrics(
    counts: dict[int, float],
    cache: ObjectiveCache,
    f_opt: float,
    rng: np.random.Generator | None = None,
):
    """Metrics for finite-shot sampling (counts: state index -> shots)."""
    total = sum(counts.values())
    n_feas_shots = sum(c for s, c in counts.items() if cache.feasible[s])
    objs = [cache.objective[s] for s in counts]
    feas_ids = [s for s in counts if cache.feasible[s]]
    best_feas = min((cache.objective[s] for s in feas_ids), default=float("nan"))
    return {
        "shots": int(total),
        "feasible_shot_rate": n_feas_shots / total,
        "best_sampled_feasible_objective": best_feas,
        "mean_sampled_objective": float(np.mean(objs)),
        "p_optimal_empirical": sum(c for s, c in counts.items() if cache.optimal_mask[s]) / total,
        "approx_ratio_best_sampled": (best_feas / f_opt) if f_opt > 0 and feas_ids else float("nan"),
    }
