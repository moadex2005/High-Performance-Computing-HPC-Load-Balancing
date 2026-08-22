# REBUILD_REPORT

**Scope:** complete post-audit rebuild of the ZETA quantum HPC-scheduling project.
**Audit reference:** `AUDIT.md` (source of truth for this rebuild).

---

## 1. What was rebuilt vs. retained

| Component | Decision | Basis |
|---|---|---|
| XY ring mixer (`rxx+ryy`, per-block ring incl. wrap edge) | **RETAINED verbatim** | audit: exact operator identity, zero leakage |
| Ising h/J derivation | **RETAINED (logic)**, re-implemented generically from QUBO | audit: relative energies exact; generic conversion is testable exhaustively |
| Cost layer gates (`rz(2γh)`, CX–rz–CX) | **RETAINED pattern**, offset removed from circuit | audit: gate construction correct; constants handled analytically now |
| Bit ordering / decoding (`q=j·N+n`, little-endian reverse) | **RETAINED** | audit: verified through real transpile path |
| 5×5 benchmark | **REPLACED** with configurable non-degenerate instances (default 10 jobs × 4 nodes) | audit finding #1 (degenerate) |
| Penalty P=120 | **REPLACED** by exact per-instance calibration `P* = max(F*−F)/V over infeasible states` | audit finding #2 |
| Dicke/initial-state preparation | **REBUILT**: `θ_k = arctan(√(N−1−k))` forward staircase | audit finding #4 (over-rotation; node 0 had P=0) |
| Soft-vs-XY comparison harness | **REBUILT**: identical budgets/seeds/init/selection; expectation-value optimizer target | audit findings #3,#5,#10 |
| Result reporting | **REBUILT**: full-distribution metrics (feasibility rate, P(optimal), mean/typical quality), never min-of-samples | audit finding #3 |
| Constant offsets | **FIXED**: single convention — reported energy ≡ scheduling objective, unit-tested exhaustively | audit finding #6 |

## 2. Mathematical formulation (final)

- `x[j,n]∈{0,1}`, qubit `q=j·N+n`; constraint `Σₙ x[j,n]=1 ∀j` (multiple jobs per node allowed)
- `F(x) = Σₙ (load_n − T_n)²`; targets proportional to capacities when given (`T_n = κ·cap_n`)
- Capacity overload `max(0, load_n−cap_n)` = reported diagnostic; every benchmark instance validated so optima have zero overload
- Soft-QUBO energy: `F + P·V`, `V = Σⱼ(Σₙx[j,n]−1)²`, penalty calibrated exactly per instance
- XY cost: `F` only; dynamics confined to the one-hot subspace by construction
- Canonical energy convention: circuit implements the Z-part; reported values add the analytic offset so that **Ising energy ≡ scheduling objective** on every basis state (tested exhaustively)

Empirical formulation finding: exact calibration gives **very small sufficient penalties** on all generated instances — recommended penalties (1.1·`P*`) range from ~10⁻⁶ to 3.10 across the ten 6×3 suite instances (`results/bench_6x3_full.csv`, column `penalty`, i.e. `P*` ≤ ≈2.8), and `P* = 0` on the 4×2 noise instance. Constraint violations are therefore at most marginally attractive under this objective family; Soft-QUBO's poor feasibility is a *sampling-distribution* effect (~99.7 % infeasible mass), not an attractive infeasible optimum. This is instance-dependent, NOT a general theorem: the original 5×5 uniform-target formulation required `P* ≈ 200–400` (audit Probe 1), and its P=120 made the penalized ground state itself infeasible.

## 3. Validation gates — all passed

| Gate | Requirement | Evidence |
|---|---|---|
| G1 | objective correct | hand-computed example; exhaustive cache↔scalar agreement (`test_objective.py`) |
| G2 | exact solver correct | matches naive enumeration; known answers; all-optima counting (`test_problem.py`) |
| G3 | Soft energies correct | exhaustive bitstring equality F+PV ↔ QUBO ↔ Ising incl. offset (`test_qubo.py`) |
| G4 | initial state uniform | uniformity to <1e-9 for N=2..5; normalization; support; full-circuit subspace mass = 1 (`test_initial_state.py`) |
| G5 | mixer preserves feasibility | exhaustive leakage ≤1e-10; exact conserved quantity `U Z_block U† = Z_block`; end-to-end random-params P(feas)=1 (`test_xy_mixer.py`) |
| G6 | penalty yields feasible ground state | calibrated-P ground state feasible & optimal; regression test proves too-small P fails (`test_qubo.py`) |
| G7 | identical underlying objective | both methods built from the same `build_objective_qubo`; offset tests tie energies to `objective_value` |
| G8 | fair experiment runs end-to-end | smoke suite green; identical evaluation counts between methods asserted in tests and observed in CSV |
| G9 | reproducibility | seeded generator + seeded optimizer; repeated runs identical (asserted) |
| G10 | README matches implementation | README written from measured numbers only; commands reproduce artifacts |

Final test status: **52 passed** (`python -m pytest tests -q`).

## 4. Benchmark results

### Default instance (10 jobs × 4 heterogeneous nodes)
- Non-degeneracy proven: 13,365 unique objective values / 1,048,576 feasible assignments; optimal set = 12 assignments (0.0011 %); old instance had 35 unique values with 3.84 % optimal share.
- Exact F\* = 2.00 · SA 6.00 · random-best-20k 14.00 · greedy 422.00.

### Statistical suite (6×3 nodes/jobs, 18 qubits, 180 runs, exact simulation)

One run = one COBYLA optimization of one method on one instance at one depth with one seed (10 instances × 3 seeds × 3 depths × 2 methods).

| Method | p | feasibility (prob. mass) | P(optimal) mean | per-run ratio mean-feasible/F\*, median |
|---|---|---|---|---|
| soft | 1 | 0.31 % | 8×10⁻⁶ | 2261 |
| soft | 2 | 0.31 % | 4×10⁻⁶ | 2074 |
| soft | 3 | 0.32 % | 5×10⁻⁶ | 1981 |
| xy | 1 | **100 %** | 0.23 % | 2212 |
| xy | 2 | **100 %** | 0.28 % | 2219 |
| xy | 3 | **100 %** | 0.30 % | 2213 |

Absolute gaps (mean-feasible − F\*, p=2): soft ≈ 2841, xy ≈ 2954 objective units (F\* averages ≈ 4.7 on these instances). Same objective and same per-instance enumerated optimum underlie every row.

Classical same-instance ratios to optimum: SA **1.18** (exact optimum on 9/10 instances) · random best-of-10k **1.00** (exact optimum on 10/10) · greedy 24.6.

**Conclusions drawn (no spin):**
1. XY-QAOA's genuine advantage is a *structural* feasibility guarantee (100 % vs 0.31 % probability mass, ideal simulation only).
2. Neither QAOA variant competes on solution quality with even trivial classical methods.
3. XY's typical quality is slightly worse than Soft-QUBO's (median ratio 2219 vs 2074 at p=2); XY wins on feasibility mass and P(optimal), not quality.
4. Depth increases P(optimal) only marginally under equal budgets.
5. "Best found" metrics saturate at ratio 1.0 for both methods because exact distributions always give optimal states nonzero probability (verified: mean approx_ratio_selected = 1.000 over all 180 runs) — we report mass/typicality instead of tails.

### Noise experiment (4×2, depolarizing 1e-3 / 1e-2, p=1)
XY feasibility falls 1.000 → **0.621** under noise: the guarantee holds for ideal circuits only. Soft stays ≈0.105. No hardware claims are made anywhere in this repo.

## 5. Artifacts

- `results/bench_6x3_full.csv` — 180 rows × all Phase-13 metrics
- `results/bench_6x3_full_summary.json` — config, classical ground truth, circuit resources (pre/post transpile), aggregated statistics (mean/median/std/95% CI)
- `results/benchmark_plots.png` — distributions across runs (box plots, not single numbers)
- `results/noise_experiment.json`
- `results/smoke.csv`, `results/smoke_summary.json`

## 6. Reproduction commands

```bash
pip install -e .
python -m pytest tests -q
python scripts/run_benchmark.py --smoke
python scripts/run_benchmark.py --jobs 6 --nodes 3 --instances 10 --seeds 3 --depths 1 2 3 --maxiter 150 --restarts 1 --workers 4
python scripts/run_noise.py
python scripts/plot_results.py
```

## 7. Remaining limitations

1. Executed QAOA suites limited to simulation-feasible sizes (≤ ~18–20 qubits statevector). Default 10×4 (40 qubits) fully characterized classically only.
2. All quantum results are simulation-based (noiseless + generic depolarizing); no backend calibration, no hardware execution.
3. Classical SA (ratio 1.18) and even random best-of-10k (ratio 1.00) strongly outperform both QAOA variants on solution quality; no quantum advantage is demonstrated anywhere in this project.
4. Noise destroys part of XY's feasibility guarantee (100 % → 62 % under 1e-3/1e-2 depolarizing); the guarantee holds for ideal circuits only.
5. Capacities handled via proportional targets + diagnostics rather than hard constraints (QUBO cleanliness tradeoff, documented).
6. Exact penalty calibration `P*` requires full enumeration (≤ ~26 qubits); larger instances need analytic bounds — out of scope.
7. COBYLA converges early (~27–55 evals); other optimizers/budgets unexplored (out of scope).
8. Advanced variants (warm start, CVaR, ma-QAOA) deliberately not implemented — correctness first.
