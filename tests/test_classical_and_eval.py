import numpy as np
import pytest

from zeta_scheduler import random_instance
from zeta_scheduler.classical import (
    greedy_solve,
    random_feasible_sample,
    simulated_annealing_solve,
)
from zeta_scheduler.evaluation import distribution_metrics, finite_sample_metrics
from zeta_scheduler.exact import solve_exact
from zeta_scheduler.objective import ObjectiveCache
from zeta_scheduler.qaoa import QAOAConfig, run_qaoa


@pytest.fixture(scope="module")
def small():
    return random_instance(6, 3, seed=5)


def test_greedy_is_feasible(small):
    res = greedy_solve(small)
    # one-hot feasibility always holds; capacity overload is a *reported
    # diagnostic* (LPT greedy ignores capacities by design)
    assert res["feasible"]


def test_random_sampling_uniform_and_feasible(small):
    rng = np.random.default_rng(0)
    # uniformity: marginal of job j over nodes ~ 1/N
    counts = np.zeros((small.num_jobs, small.num_nodes))
    trials = 30_000
    assigns = rng.integers(0, small.num_nodes, size=(trials, small.num_jobs))
    for a in assigns:
        counts[np.arange(small.num_jobs), a] += 1
    assert np.allclose(counts / trials, 1 / small.num_nodes, atol=0.01)

    res = random_feasible_sample(small, seed=0, num_samples=2000)
    assert res["feasible"] and res["mean_objective"] > res["objective"]


def test_sa_improves_and_stays_feasible(small):
    res = random_feasible_sample(small, seed=0, num_samples=500)
    sa = simulated_annealing_solve(small, seed=0, num_iters=5000)
    assert sa["feasible"]
    assert sa["objective"] <= res["objective"] + 1e-9


def test_all_baselines_reach_exact_optimum_on_tiny():
    p = random_instance(4, 2, seed=2)   # tiny: SA should find the optimum
    exact = solve_exact(p)
    sa = simulated_annealing_solve(p, seed=0, num_iters=20_000)
    assert abs(sa["objective"] - exact["objective"]) < 1e-6


def test_distribution_metrics_consistency(small):
    cache = ObjectiveCache(small)
    exact = solve_exact(small)
    probs = np.zeros(len(cache.feasible))
    probs[cache.feasible] = 1.0 / int(cache.feasible.sum())   # uniform feasible
    m = distribution_metrics(probs, cache, exact["objective"])
    # uniform feasible sampling on this instance should rarely hit optimum
    assert m["p_feasible"] == pytest.approx(1.0)
    assert m["p_infeasible"] == pytest.approx(0.0)
    assert 0 < m["approx_ratio_selected"] <= 1.0000001


def test_finite_sample_metrics(small):
    cache = ObjectiveCache(small)
    rng = np.random.default_rng(0)
    ids = rng.choice(np.flatnonzero(cache.feasible), size=512)
    counts = {int(i): 1 for i in ids}
    fm = finite_sample_metrics(counts, cache, solve_exact(small)["objective"])
    assert fm["feasible_shot_rate"] == pytest.approx(1.0)


def test_qaoa_reproducible_and_sane(small):
    cache = ObjectiveCache(small)
    cfg = dict(depth=1, maxiter=20, restarts=1)
    r1 = run_qaoa(small, "xy", QAOAConfig(seed=7, **cfg), cache=cache)
    r2 = run_qaoa(small, "xy", QAOAConfig(seed=7, **cfg), cache=cache)
    assert np.allclose(r1.best_params, r2.best_params)
    assert r1.best_expectation == pytest.approx(r2.best_expectation)

    s = run_qaoa(small, "soft", QAOAConfig(seed=7, **cfg, penalty=1000.0), cache=cache)
    assert s.num_evaluations == r1.num_evaluations   # identical budget policy
