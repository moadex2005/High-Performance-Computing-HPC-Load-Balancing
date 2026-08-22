# Team ZETA — Technical Audit

**Scope:** Full audit of `README.md`, `XY.ipynb`, `Soft.ipynb` (constraint-preserving XY-mixer QAOA vs. penalty-based Soft-QUBO for HPC load balancing).

**Method:** Every suspicion below was verified by independent numerical probes (exact statevector enumeration, exhaustive bitstring energy comparisons, commutator/leakage tests, pipeline reproduction). Probe scripts are preserved in the audit working notes; every claim in this document traces to an executed test.

---

## Executive Summary

The project's **core algorithmic primitives are largely sound** (XY mixer, Ising coefficient derivation, QUBO→Ising conversion, bit ordering), but the **benchmark, the penalty formulation, the initial state, and the experimental methodology are scientifically invalid**. The headline claims do not survive scrutiny:

1. The 5×5 benchmark is **degenerate**: 120 of 3,125 feasible assignments (3.84%) achieve the "optimal" 770.0. The reported XY-QAOA result is obtained with **random circuit parameters** (5/5 trials) and does not measure optimization ability.
2. The Soft-QUBO penalty **P = 120 is mathematically insufficient**: the penalized ground state over all 2²⁵ states has energy **465.0 and is infeasible** (a job duplicated and another dropped). The true penalty threshold is ≈ 400. The reported Soft-QUBO result (3320.0) is the objective of the **single feasible shot out of 1024** (0.1% feasibility) — reproduced exactly.
3. The XY initial state is **not uniform**: node 0 receives exactly **zero** probability; the notebook's *own* uniformity self-check fails.
4. The comparison is **unfair** (different depth, budget, initialization, and solution-selection rules between methods).

**Verdict:** the reported results are artifacts of a degenerate instance plus min-order-statistic sampling. The mixer and Ising machinery are worth keeping; everything experimental must be rebuilt.

---

## 1. Repository / Architecture

| Item | Finding |
|---|---|
| Structure | 3 files only: `README.md`, `XY.ipynb`, `Soft.ipynb`. No package, no tests, no requirements.txt, no CLI. |
| Notebooks | Monolithic; all math inline; `Soft.ipynb` duplicates the QUBO construction that `XY.ipynb` re-implements a third time in `soft_qubo_baseline()`. |
| Dependencies | Pinned loosely (`qiskit>=1.0,<2.0`, Aer, qiskit-optimization 0.7.0, qiskit-algorithms). `pip install` cells inside notebooks. |
| Status | **PARTIALLY CORRECT** (runs, but unverifiable and unreproducible-by-construction; no programmatic entry point). |

## 2. Problem Formulation

Stated problem: assign each of 5 jobs (weights [15, 30, 10, 45, 20]) to one of 5 homogeneous nodes; minimize Σₙ (loadₙ − 24)².

- **What the code actually implements:** one-hot-per-job constraint only. **No capacity constraints exist anywhere** in either notebook, despite README §1 claiming "strict resource and capacity constraints". Nodes are homogeneous with infinite capacity.
- **Multiple jobs per node:** allowed (and required to be allowed — with J = N = 5 the interesting solutions are bijections, but nothing enforces or forbids them).
- **Status:** **PARTIALLY CORRECT** — the implemented problem is a valid (simpler) problem, but it is not the problem the README describes.

## 3. Current QUBO Formulation

`Soft.ipynb` builds: H = Σₙ(Σⱼ w_j x_jn − T)² + P·Σⱼ(Σₙ x_jn − 1)², P = 120, via `QuadraticProgram` + `QuadraticProgramToQubo(penalty=120.0)`.

- Objective expansion (a_q = w² − 2Tw, b_jk = 2ww) — **CORRECT** (verified exhaustively, §14).
- Library conversion QUBO → Ising — **CORRECT** (exact on tiny instance incl. offset).
- **Penalty adequacy — INCORRECT (CRITICAL):** enumerating all 2²⁵ states, the penalized ground state at P = 120 has energy **465.0 < 770.0** and is **infeasible** (row sums [1, 2, 1, 0, 2]: job 1 assigned twice, job 3 dropped). Penalty scan: argmin stays infeasible through P = 200; feasible from P ≈ 400. The README claim "a penalty factor of P = 120.0 is chosen to avoid constraint violations" is **false** — the formulation's global optimum is an invalid schedule, so even a perfect optimizer returns an infeasible answer.

## 4. Current Ising Formulation (XY.ipynb)

Hand-derived H = C + Σ h_q Z_q + Σ J_qq' Z_q Z_q' with h_q = −a_q/2 − Σ b/4, J = b/4, z = 1 − 2x.

- **h and J coefficients: CORRECT.** Exhaustive check over all feasible states of three instance sizes plus 20k–50k random infeasible states: `H_energy − objective` is a **constant** in every case (relative energies exact).
- **Constant offset: INCORRECT.** Code accumulates only Σₙ T² = 2880. True full offset = **11042.5**; implemented energies are shifted by **−8162.5** for the 5×5 instance. The notebook's own spot-check (`H_cost energy = −7392.5` vs `direct obj = 770.0`) would print **MISMATCH**, not the "✓ MATCH" the README implies.
- README §3.3 claims C_offset = 2880.0 "from the full formula" — the formula it prints evaluates to **11042.5**, not 2880. The printed equation is internally inconsistent.
- Impact on optimization: none (constant shifts don't affect variational landscapes or argmins). Impact on reporting: all quoted "energies" are not objective values.

## 5. Current Soft-QUBO QAOA

- p = 1 (`QAOAAnsatz(reps=1)`), x0 = zeros, **COBYLA maxiter = 8** (cell 7). Later "improved" cells: maxiter = 30 (random x0·0.1), p = 2 with maxiter = 15.
- Optimizer target: exact expectation ⟨H_Ising⟩ via `BackendEstimatorV2` — methodologically the *better* choice (ironically better than XY's).
- `apply_layout` wrapped in bare `try/except` — silent-fallback risk if the transpiler ever permutes qubits (on AerSimulator with trivial layout it does not; verified `final_index_layout` is None/identity).
- **Result selection:** best *feasible* sampled bitstring by minimum objective over 1024 shots.
- **Reproduction (executed on this machine):** best feasible = **3320.0 — exactly the README number**, from **1 feasible shot out of 1024 (0.1%)**. Mean sampled objective 10969.5; best sampled overall 475.0 (infeasible, below the feasible optimum — as predicted by §3). The "improved" run: **0.0% feasible shots**.
- **Status:** pipeline math correct; experimental result meaningless (under-trained, infeasible-dominated, single lucky shot).

## 6. Current XY-QAOA

- p = 2, COBYLA, maxiter = 200, 3 restarts (x0 ~ U(0, 2π), seeds 0/17/34).
- **Optimizer target: INCORRECT methodology.** `cost_fn` returns `best_sample(counts_2048)` — the **minimum-order statistic over 2048 finite shots** (preferring feasible), not an expectation value. Optimizing a sample-min is statistically biased and noisy; with 2048 shots on a degenerate landscape it saturates immediately.
- Executed probe: COBYLA's objective is **770.0 at every single evaluation** (7 evals, then termination). The README's "COBYLA convergence" plot is an artifact — the optimizer learns nothing because the reported quantity is already saturated.
- **Random-parameter control:** 5/5 trials with uniformly random parameters produce best-of-4096-shots = **770.0**. The headline result is **independent of optimization**.
- True expectation ⟨H⟩: −4152.7 (θ=0) vs −4239.7 (random params) — i.e. ~3923–4010 in objective space after removing the offset error; **the circuit's actual expected performance never approaches 770**. Only the sampling tail does.
- Final reported value = min over 4096 shots at final params — a tail statistic, not a solution-quality measure.

## 7. Initial State / Dicke Preparation — **INCORRECT (CRITICAL)**

README claims a "uniform superposition of all feasible assignments" with angles θ_k = 2 arccos(√((k+1)/(k+2))).

**Numerical proof of non-uniformity** (exact statevector, single job block):

| N | P(node 0) | P(node 1) | P(node 2) | P(node 3) | P(node 4) |
|---|---|---|---|---|---|
| 3 | 0.000 | 0.111 | 0.889 | — | — |
| 5 | **0.000** | 0.111 | 0.222 | 0.240 | 0.427 |
| 8 | 0.000 | 0.111 | 0.222 | 0.240 | 0.190 … 0.051 |

- Node 0 gets **exactly zero** probability for every N. The state is normalized and fully inside the one-hot subspace (verified: P(feasible) = 1.000000 for the full 25-qubit circuit), but it is **not** the uniform W state.
- Consequence at θ = 0: P(optimal 770-state) = 0 exactly — no bijection is representable because every block avoids node 0.
- The notebook's *own* verification (`should be ~20% each`, 5% tolerance) prints **FAIL** on this distribution. The shipped notebook contradicts its own check.
- **Root cause (mathematical):** XX and YY act *identically* in the {|10⟩,|01⟩} subspace (both map |10⟩↔|01⟩ with + phase). Therefore `RXX(θ)·RYY(θ) = exp(−iθ(XX+YY)/2)` rotates that two-level subspace by the **full angle θ**, not θ/2. The notebook's angle schedule assumes a θ/2-rate rotation, so every rung **over-rotates by 2×**, pushing amplitude past intermediate nodes toward the block's high end.
- **Correct construction (derived & verified uniform to 1e-9 for N = 2…8, zero leakage):** forward staircase, pairs (k, k+1) ascending, with

  θ_k = arctan(√(N−1−k))  (equivalently sin θ_k = 1/√(N−k) with the full-angle convention).

  For N = 5: θ = [1.1071, 1.0472, 0.9553, 0.7854]. The README's listed angles [π/2, 1.2310, π/3, 0.9273] are simply wrong for this gate composition.
- Reversing the application order of the README angles does **not** fix it (all amplitude collapses onto one node).

## 8. Mixer Correctness — **CORRECT (verified)**

- `RXX(β)RYY(β) = exp(−iβ(XX+YY)/2)` — operator equality to 1.1e-16.
- Ring topology includes the wrap-around edge (N−1, 0). ✓
- Commutator ‖[H_mix, Σ_k Z_k]‖ = 0 exactly per block (N = 2, 3, 5). ✓
- **Exhaustive leakage test:** applying U_M(β) to *every* feasible basis state of (1,3), (2,2), (2,3), (1,5) systems at 5 values of β: maximum probability outside the feasible subspace ≤ **9.3e-33** (machine zero). ✓
- Full 25-qubit circuit at θ = 0: P(feasible) = 1.000000. ✓
- The notebook's own mixer test is weak (one input configuration, sampled), but the underlying component is genuinely correct. Infeasibility under the XY circuit is **mathematically impossible**, not merely unlikely — this is the project's one robust positive result.

## 9. Objective Correctness

- f(x) = Σₙ (loadₙ − W/N)² is a legitimate load-balancing objective (N × variance) and is implemented consistently in both notebooks and in the Ising coefficients (relative energies exact).
- **But it is degenerate on the chosen instance** (§16) and identical for both methods — no objective mismatch between Soft and XY was found.
- No capacity term exists despite the README's claims (§2).

## 10. Constraint Correctness

- **Actual enforced constraint (both methods):** for each job j: Σₙ x_jn = 1. Nothing else.
- Soft-QUBO: soft-penalized (insufficiently — §3). XY-QAOA: hard-preserved (§8).
- Capacity constraints: **MISSING** everywhere.
- Status: encoding matches the *simplified* problem, not the advertised one.

## 11. Bit Ordering / Decoding — **CORRECT (verified)**

- Variables created j-major: index q = j·N + n; decode = `reversed(bitstring)` then reshape — consistent with Qiskit little-endian. Round-trip verified through an actual circuit + `transpile(optimization_level=1)` + Aer sampling (no permutation: `final_index_layout` = None/identity on AerSimulator).
- Tiny-instance exhaustive test through the *real library path* (`QuadraticProgram → QuadraticProgramToQubo → to_ising`): direct objective = QUBO polynomial = Ising energy **exactly (0.0 error)** for all 2⁴ bitstrings.
- No silent bit-order bug found.

## 12. Optimization Procedure

| Aspect | Soft-QUBO | XY-QAOA | Fair? |
|---|---|---|---|
| Depth | p = 1 (main) / p = 2 (side cell) | p = 2 | ✗ |
| Initial params | zeros (main) / U(0,0.1) | U(0, 2π), 3 restarts | ✗ |
| Max evals | **8** (main), 30, 15 | 200 × 3 = 600 | ✗ (~75× more for XY) |
| Optimizer objective | exact ⟨H⟩ | sample-min over 2048 shots | ✗ (different quantities) |
| Result selection | best feasible of 1024 shots | best feasible of 4096 shots | ✗ |

- Parameter bounds: none anywhere (fine for QAOA but unbounded drift possible).
- Seeds: `np.random.seed(7)` global, `seed_simulator=7`; restarts use seeds 0/17/34 — deterministic per version but never varied (single-seed science).
- The README table compares numbers produced under these incomparable settings as if they were method capabilities.

## 13. Sampling and Evaluation

- Both notebooks report **min-objective-among-sampled (feasible-preferring)** — an order statistic of a finite sample, not: most-probable bitstring, expectation value, or optimizer output.
- README §6.1/§6.2 presents these tail statistics as method performance ("achieved 770.0", "Soft-QUBO's sub-optimal 3320.0").
- Executed evidence: XY@random-params → 770.0 (5/5); Soft@maxiter=8 → 3320.0 from **1/1024 feasible shots**; Soft "improved" → 0.0% feasible shots.
- Feasibility *rate* is never reported for Soft-QUBO (it is 0.1–0.0% — the actual story); for XY it is trivially 100%.

## 14. Numerical Verification (summary of executed probes)

| Probe | Result |
|---|---|
| P1: enumerate all 3,125 feasible 5×5 assignments | 35 unique objective values; min = max-count value 770.0 ×120 states (3.84%); mean 2920; max 11520 |
| P1: penalized ground state over all 2²⁵, P=120 | **465.0, infeasible** (rows [1,2,1,0,2]); feasible argmin only from P≈400 |
| P2: notebook init state, exact statevector | Non-uniform: P(node0)=0; e.g. N=5 → [0, .111, .222, .240, .427]; notebook's own check FAILS |
| P2: corrected angles θ_k=arctan(√(N−1−k)) | Uniform to <1e-9 for N=2…8, zero leakage |
| P3: mixer operator identity | RXX·RYY = exp(−iθ(XX+YY)/2) to 1.1e-16 |
| P3: commutator & leakage | ‖[H_mix, ΣZ]‖=0; max leakage 9.3e-33 (exhaustive) |
| P4: Ising h/J vs direct objective | Constant offset only: −8162.5 (5×5); relative energies exact everywhere |
| P5: library QUBO/Ising path (tiny) | Exact agreement (0.0) for all bitstrings incl. offset |
| P5: bit-order round trip | Consistent through real transpile+sample path |
| P6: XY with random params | best-of-4096 = 770.0 in 5/5 trials; COBYLA log flat at 770 |
| P6b: P(feasible) full circuit θ=0 | 1.000000; P(optimal)=0 at θ=0 (node-0 blind spot) |
| P7: Soft.ipynb pipeline reproduction | **3320.0 exactly**, from 1/1024 feasible shots (0.1%) |
| P7: README depth claim | Actual p=2 circuit depth **63**, not 189 |

## 15. Reproducibility

- The **3320.0** artifact reproduced exactly on a different Qiskit major version (2.5.2 vs 1.4.5) with seed 7 — the number is stable *because it is a sampling artifact*, not because the experiment is sound.
- XY's 770.0 reproduces trivially (any parameters work).
- No requirements.txt; notebook-embedded installs; results are seed- and version-dependent in general (single seed used everywhere).
- **Status: PARTIALLY CORRECT** — reproducible in the trivial sense, meaningless in the scientific sense.

## 16. Scientific Validity — the 5×5 Degeneracy (proven)

With J = N = 5 and the one-hot-per-job constraint, every assignment that is a bijection yields loads equal to a permutation of the weights, hence objective Σⱼ(wⱼ−24)² = **770 regardless of the permutation**. Enumeration confirms:

- 3,125 feasible assignments → only **35 unique objective values**;
- **120 assignments (3.84%) achieve the minimum 770.0**;
- the greedy baseline reaches 770 *by construction* (any bijection does) — it is not a meaningful baseline here;
- "XY-QAOA converged to the global minimum" (README) is a property of the instance, not of the algorithm.

**The central empirical claim of the project is not supported by the evidence.**

## 17. Problems Found

| # | Component | Severity | Evidence | Recommended fix |
|---|---|---|---|---|
| 1 | 5×5 benchmark instance | **CRITICAL** | P1: 120 optimal states, 3.84%; 35 unique values | Non-degenerate instances (J > N), configurable sizes, exact-solver ground truth |
| 2 | Penalty P = 120 (Soft-QUBO) | **CRITICAL** | P1: penalized ground state 465.0 infeasible; threshold ≈400 | Principled penalty (≥ threshold or auto-scaled); report feasibility rates |
| 3 | Reported metrics are sample-min statistics | **CRITICAL** | P6/P7: 770@random params 5/5; 3320 = 1/1024 feasible shots | Report expectation + distribution metrics (best/mean/P(optimal)/feasibility rate), fixed shot budget, seeds |
| 4 | Initial state non-uniform | **CRITICAL** | P2: P(node0)=0; notebook self-check FAILS | Correct angles θ_k = arctan(√(N−1−k)); add uniformity unit test |
| 5 | Unfair method comparison | **CRITICAL** | §12 table: 8 vs 600 evals, p=1 vs p=2, different targets/selection | Unified harness: same instance, budget, shots, seeds, selection rule |
| 6 | Ising constant offset wrong; README C_offset claim false | HIGH | P4: shift −8162.5; formula evaluates to 11042.5 ≠ 2880 | Compute offset analytically correct; unit-test absolute energies |
| 7 | README depth claim 189 | HIGH | Actual p=2 depth 63 | Regenerate resource table from code |
| 8 | No capacity constraints despite claims | HIGH | Absent from both notebooks | Add optional capacities to problem definition + objective, or remove claim |
| 9 | No exact solver / approximation ratio / statistics | HIGH | Neither notebook computes optimal via enumeration | Exact enumeration solver; ≥5 seeds × ≥10 instances; distributions |
| 10 | XY optimizer target = noisy sample-min | HIGH | Flat 770 COBYLA log; biased statistic | Optimize exact ⟨H⟩ (or CVaR) via statevector/estimator |
| 11 | `try/except` around `apply_layout` | MEDIUM | Silent fallback risk on layout change | Assert layout identity explicitly |
| 12 | Greedy baseline trivially optimal | MEDIUM | Any bijection gives 770 | Keep greedy but only meaningful on non-degenerate instances |
| 13 | DLA/scalability section unverified | LOW | Citation (2026 npj QI 12:61) not checkable offline | Mark as unverified or cite checkable sources |
| 14 | No tests, no package, notebook-installed deps | MEDIUM | Repo structure | Refactor into tested package (Phase 15–16 of rebuild) |

## 18. What Is Actually Correct

Verified correct and worth retaining:

1. **XY ring mixer** — exact operator identity, exact commutation with the constraint, machine-zero leakage (exhaustive). The central architectural idea is sound.
2. **Ising coefficient derivation (h, J)** — relative energies exactly match the true objective on every tested bitstring, feasible or not.
3. **Cost-layer gate construction** (Rz(2γh), CX–Rz–CX for ZZ) — consistent with the z = 1−2x convention.
4. **Soft-QUBO library pipeline** (QuadraticProgram → QUBO → Ising) — exact on tiny instances including offset.
5. **Bit ordering & decoding** — consistent end-to-end through the real transpile path.
6. **Subspace preservation end-to-end** — P(feasible) = 1.000000 for the full 25-qubit circuit.
7. **Worst-case baseline value** (11520) — arithmetic checks out.
8. **Problem expansion algebra** (a_q, b_jk formulas) — correct as written.

## 19. Recommended Rebuild Plan

**CRITICAL FIXES (must precede any new claims)**
1. Non-degenerate, configurable problem generator (default 10 jobs × 4 nodes, heterogeneous capacities; seeded random instances).
2. Principled unified objective + penalty with *verified* penalty adequacy (unit-tested against brute force per instance).
3. Correct uniform W-state preparation (θ_k = arctan(√(N−1−k))) with unit tests.
4. Exact enumeration solver as ground truth; approximation ratio as the primary metric.
5. Fair comparison harness: identical instances, budgets, shots, seeds, selection rules for both methods; expectation-based optimizer objective.
6. Full sampling metrics: feasibility rate, best/mean objective, P(optimal), approximation ratio, evaluations, runtime.

**IMPORTANT IMPROVEMENTS**
7. Classical baselines: exact, greedy, random-feasible, simulated annealing.
8. Multi-seed, multi-instance benchmark suite with statistical summaries and distribution plots.
9. Correct constant offsets; regenerate resource tables from code.
10. Depth sweep p ∈ {1..3(5)}.
11. Noise experiment (depolarizing) on reduced scale, clearly labeled as simulation-only.

**OPTIONAL IMPROVEMENTS**
12. Warm-start QAOA (from greedy solution), CVaR objective, parameter transfer, multi-angle QAOA.
13. MPS-based simulation for larger instances; hardware runs only if actually executed.
14. OR-Tools cross-check of the exact solver.

## 20. Final Verdict

| Question | Answer |
|---|---|
| Is the current Soft-QUBO mathematically correct? | The QUBO/Ising *machinery* is correct, but the **formulation is invalid as used**: P = 120 puts the global optimum at an infeasible state (465 < 770). |
| Is the current XY-QAOA mathematically correct? | Mixer, Ising coefficients, subspace preservation: **yes** (verified). Initial state: **no** (non-uniform, node-0 blind spot). Offset: wrong (constant). Optimizer target: methodologically unsound. |
| Is the current initial state correct? | **No.** Feasible and normalized, but non-uniform with P(node 0) = 0; contradicts README and fails the notebook's own check. |
| Is the current benchmark meaningful? | **No.** Degenerate: 3.84% of the feasible space is optimal; every bijection scores 770. |
| Are the reported results trustworthy? | **No.** 770.0 is reachable with random parameters (5/5); 3320.0 is one lucky feasible shot out of 1024 (0.1% feasibility). Both reproduced as artifacts. |
| Which parts should be retained? | Mixer design, Ising coefficient derivation, cost-layer construction, library QUBO path, decoding conventions, worst-case baseline. |
| Which parts should be rewritten? | Problem instances, penalty scheme, initial-state angles, optimizer objective, result selection, all experimental methodology, README claims, and the project structure (package + tests + CLI). |
