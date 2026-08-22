# Quantum vs. Classical HPC Load Balancing — a verified benchmark

**Constraint-preserving XY-mixer QAOA vs. penalty-based Soft-QUBO for HPC job scheduling — rebuilt from the ground up after a full technical audit of the original hackathon codebase.**

> **Read this first:** the original project (May 2026, Team ZETA) reported that XY-QAOA "achieves 770.0, matching greedy and beating Soft-QUBO's 3320.0". A full audit (`AUDIT.md`) proved those numbers were artifacts: a degenerate benchmark, an insufficient penalty coefficient, a broken initial state, an unfair comparison, and min-of-samples reporting. This repository replaces all of it with a correct, fair, reproducible experiment. The honest headline is now:
>
> **XY-QAOA guarantees feasible schedules by construction (100% in ideal simulation) but does not produce better solutions than trivial classical methods; classical simulated annealing outperforms both QAOA variants on solution quality; hardware noise destroys the feasibility guarantee (62% under depolarizing noise).**

---

## 1. The scheduling problem

Assign each of `J` jobs to one of `N` heterogeneous compute nodes.

- Variables: `x[j,n] ∈ {0,1}` at qubit `q = j·N + n`
- Hard constraint (assignment): `Σₙ x[j,n] = 1 ∀j` — multiple jobs per node allowed
- Node loads: `load_n = Σⱼ wⱼ x[j,n]`
- Target loads: `T_n = κ·cap_n` with `κ = W/Σcap` when capacities are given (proportional balancing), else `T_n = W/N`
- **Objective:** `F(x) = Σₙ (load_n − T_n)²` (load imbalance)
- **Capacity diagnostic:** `overload_n = max(0, load_n − cap_n)` is *reported*, never hidden; every benchmark instance is validated so that its optimal assignments have zero overload

The identical problem object feeds **all six methods**: Soft-QUBO QAOA, XY-QAOA, exact enumeration, greedy, random-feasible sampling, simulated annealing.

Why proportional targets instead of hard capacity constraints? Because `Σₙ(load_n−T_n)²` is exactly quadratic-binary (a clean QUBO), while hinge penalties require slack variables that double the qubit count. The tradeoff is documented rather than hidden.

## 2. Methods

| | Soft-QUBO QAOA | XY-QAOA |
|---|---|---|
| Search space | all 2^(J·N) bitstrings | one-hot subspace only (N^J states) |
| Constraint handling | penalty `P·Σⱼ(Σₙx[j,n]−1)²` added to cost | none needed — ring XY mixer conserves per-block excitation |
| Initial state | `\|+…+⟩` | uniform superposition of feasible assignments (corrected W-state prep) |
| Cost Hamiltonian | Ising(F + P·V) | Ising(F) |
| Mixer | X (`rx(2β)`) | ring `rxx(β)+ryy(β)` per block |

**Penalty calibration (no magic numbers).** For each instance we define

```
P* = sup { (F*_feasible − F(x)) / V(x) : x infeasible }
```

(`V(x) = Σⱼ(Σₙx[j,n]−1)²`). `P` > `P*` guarantees that the penalized ground state is feasible and attains the constrained optimum. `calibrate_penalty` computes `P*` exactly by enumerating the full 2^Q space wherever tractable (≤ 26 qubits).

Measured calibration results (the benchmark stores the *recommended* penalty = 1.1·`P*` per run in the CSV `penalty` column):
- ten 6×3 suite instances: recommended penalties range from ~10⁻⁶ to **3.10**, i.e. `P*` itself never exceeds ≈ **2.8**; one 4×2 instance: `P* = 0`;
- i.e. under this variance-type objective with J > N, constraint violations are at most *marginally* attractive, so a negligibly small penalty already suffices.

**This is an empirical property of these instances, not a general theorem.** Penalties are NOT unnecessary for Soft-QUBO in general: the original project's 5×5 uniform-target formulation had `P* ≈ 200–400` (audit Probe 1) — with P = 120 its penalized ground state was itself infeasible. Whether `P*` is large depends on the objective family and the instance geometry (here: more jobs than nodes plus proportional targets); it must be calibrated or bounded per problem, never assumed.

A practical consequence of the tiny measured `P*`: Soft-QUBO's failure mode on these instances is not an attractive infeasible optimum but its *sampling distribution* — even after expectation-value optimization, ~99.7 % of its output distribution sits on infeasible bitstrings.

## 3. What was wrong before (audit results)

Full details with numerical evidence in [`AUDIT.md`](AUDIT.md). Summary:

1. **Degenerate benchmark.** 5 jobs × 5 nodes ⇒ every bijection yields imbalance 770; 120/3125 feasible states (3.84%) were "optimal". Any parameter setting "solved" it.
2. **P = 120 was invalid.** Brute force over all 2²⁵ states showed the penalized ground state had energy 465 < 770 and was infeasible.
3. **Broken initial state.** `RXX(θ)RYY(θ)` rotates the two-level subspace by the full angle θ, so the published angles over-rotated 2×: node 0 got probability exactly 0. Correct construction: `θ_k = arctan(√(N−1−k))`.
4. **Sampling-tail reporting.** "770.0" appeared with *random* parameters (5/5 trials); "3320.0" was the objective of the single feasible shot among 1024 (0.1% feasibility).
5. **Unfair comparison** (8 vs ~600 optimizer evaluations, p=1 vs p=2, different objectives and selection rules), plus a wrong constant offset (−8162.5) and a false depth claim (189 vs actual 63).

## 4. Verified-correct components retained

The audit proved these correct; they are preserved in `src/zeta_scheduler/circuits.py` / `ising.py`:

- XY ring mixer: `rxx+ryy == exp(-iβ(XX+YY)/2)`, commutes with block weight, machine-zero leakage (exhaustively tested)
- Ising h/J derivation from the expanded objective
- Cost layer gates (`rz(2γh)`, CX–rz–CX)
- Bit ordering `q = j·N+n`, little-endian decoding through real transpile paths

## 5. Benchmark methodology

Everything configurable via CLI; deterministic seeds throughout.

```
python scripts/run_benchmark.py --smoke          # fast end-to-end check
python scripts/run_benchmark.py \
    --jobs 6 --nodes 3 --instances 10 --seeds 3 \
    --depths 1 2 3 --maxiter 150 --restarts 1 --workers 4
```

Fairness guarantees enforced in code:

- same instances (seeded generator, validated non-degenerate + capacity-feasible optimum)
- same depth sweep, parameter count (2p), init policy, seeds, COBYLA budget, restarts
- optimizer objective = **exact expectation value ⟨H⟩** from the full distribution (no shot noise, no sample-min statistic — the original project's central methodological bug)
- final metrics computed from the same full distribution with the same selection rule for both methods
- classical baselines optimize/score the identical objective; random baseline samples **uniformly over feasible assignments only**

Executed headline suite: 6 jobs × 3 nodes (18 qubits), 10 random instances × 3 seeds × p ∈ {1,2,3} × 2 methods = **180 runs**. One *run* = one full optimizer execution (COBYLA from a seeded start) of one method on one instance at one depth with one QAOA seed; its row in the CSV contains the optimizer trace summary plus metrics of the final exact distribution.

**Metric definitions (used consistently in all tables):**

- *Feasibility* = probability mass that the final circuit distribution places on assignment-feasible bitstrings. It is **not** the fraction of feasible bitstrings in the Hilbert space (which would be 3⁶/2¹⁸ ≈ 0.28 % — numerically close to Soft-QUBO's 0.31 % by coincidence).
- *P(optimal)* = probability mass on optimal assignments in the final distribution, averaged over runs.
- *Typical quality* = mean objective over feasible-support states of the distribution, compared to F\* (both per-run ratios and absolute gaps reported below).
- All distributions are exact (full statevector probabilities, noiseless), i.e. the infinite-shot limit — not finite samples.

## 6. Results

### Default 10×4 instance (jobs [15,30,10,45,20,35,25,50,18,32], caps [100,80,130,90])

Non-degenerate by construction: **13,365 unique objective values** across 1,048,576 feasible assignments; optimum F\*=2.0 reached by only 12 assignments (0.0011%); zero overload at optimum.

| Method | Objective | Ratio to optimum |
|---|---|---|
| Exact enumeration | **2.00** | 1.00 |
| Simulated annealing | 6.00 | 3.00 |
| Random best-of-20k | 14.00 | 7.00 |
| Greedy (LPT) | 422.00 | 211.0 |

### Statistical suite (180 runs, exact noiseless simulation)

| Metric (mean over runs unless stated) | Soft-QUBO | XY-QAOA |
|---|---|---|
| Feasibility probability of sampled states | **0.31 %** (range 0.10–0.66 %) | **100 %** (min ≥ 1 − 10⁻¹⁴) |
| P(sample optimal), p=1 | 8×10⁻⁶ | 0.23 % |
| P(sample optimal), p=2 | 4×10⁻⁶ | 0.28 % |
| P(sample optimal), p=3 | 5×10⁻⁶ | 0.30 % |
| Mean sampled feasible objective, p=2 | 2846 | 2959 |
| Typical quality: per-run ratio mean-feasible/F\*, p=2 (median) | 2074 | 2219 |
| Absolute gap mean-feasible − F\* (p=2, avg over runs) | 2841 | 2954 |
| Optimizer evaluations (p=1/2/3) | 26 / 40 / 55 | 27 / 39 / 54 |

Scale note for the quality rows: F\* averages only ≈ 4.7 on these instances while feasible-sample objectives average ≈ 2800–2960, so *relative* ratios look enormous (~2000–2400×); the absolute gaps are ≈ 2720–2960 objective units. Both views are reported; neither is cherry-picked. The same objective `F` and the same exact optimum `F\*` (per-instance enumeration) underlie every row.

Classical baselines on the same 10 suite instances (ratio objective/F\*):

| Baseline | mean ratio | notes |
|---|---|---|
| Simulated annealing | **1.18** | exact optimum on 9/10 instances (worst 2.77×) |
| Random best-of-10k | **1.00** | finds the exact optimum on 10/10 instances |
| Greedy (LPT) | 24.6 | poor on imbalance objectives |

### Honest reading

1. **Feasibility is XY's real advantage** — structural, not empirical: 100 % vs 0.31 % probability mass in ideal simulation. But it is a property of the ideal circuit; see the noise results below.
2. **Neither QAOA variant produces competitive solutions.** A typical sampled feasible assignment sits ~2000–2400× above the optimum (absolute gap ≈ 2700–3000 units), SA sits within 18 % on average, and even a single batch of 10,000 uniform random feasible samples found the exact optimum on every instance. Note also that XY's *typical* quality is slightly **worse** than Soft-QUBO's (per-run median ratio 2219 vs 2074 at p=2) — XY wins decisively on feasibility mass and P(optimal), not on typical solution quality.
3. **Depth helps marginally** (P(optimal): 0.23 % → 0.30 % from p=1→3).
4. **Tail metrics mislead.** With exact distributions, some optimal state almost always has nonzero probability, so "best found" saturates at ratio 1.0 for both methods (verified: mean = 1.000 across all 180 runs). We therefore report probability mass (P(optimal)), typical-sample quality, and feasibility — not min-over-samples.

### Noise (Phase: separate experiment)

Depolarizing noise (p₁=10⁻³ single-qubit, p₂=10⁻² two-qubit), 4×2 instance, p=1 (`results/noise_experiment.json`):

| Condition | Feasibility | P(optimal) |
|---|---|---|
| XY noiseless | 1.000 | 0.106 |
| **XY noisy** | **0.621** | 0.043 |
| Soft noiseless | 0.103 | 0.006 |
| Soft noisy | 0.105 | 0.042 |

**Noise destroys the feasibility guarantee** (depolarizing errors do not respect the constraint manifold). Any hardware claim would need error mitigation; we make none.

## 7. Repository layout

```
src/zeta_scheduler/
    problem.py         # SchedulingProblem, seeded validated instance generation
    objective.py       # canonical F(x), violation, overload; vectorized cache
    exact.py           # exhaustive ground truth + non-degeneracy stats
    classical.py       # greedy / uniform-random-feasible / multi-start SA
    qubo.py            # QUBO builder, exact penalty calibration (P*)
    ising.py           # exact QUBO->Ising with documented offset convention
    circuits.py        # corrected W-state prep, preserved XY mixer, cost layer
    qaoa.py            # fair ansatz construction + expectation optimization
    evaluation.py      # full-distribution metrics
    benchmark.py       # suite runner, aggregation, statistics
scripts/
    run_benchmark.py   # CLI (--smoke, --jobs/--nodes/--instances/--seeds/--depths/--workers)
    run_noise.py       # depolarizing noise experiment
    plot_results.py    # distribution plots
tests/                 # 52 tests incl. all validation gates
notebooks/             # original audited notebooks (historical reference only)
AUDIT.md               # full technical audit with numerical evidence
results/               # CSVs, summary JSONs, plots, noise JSON
```

## 8. Reproduction

```bash
pip install -e .
python -m pytest tests -q                       # validation gates (~1.5 min)
python scripts/run_benchmark.py --smoke         # end-to-end sanity
python scripts/run_benchmark.py                 # statistical suite (as reported)
python scripts/run_noise.py                     # noise experiment
python scripts/plot_results.py                  # figures
```

Environment used for reported numbers: Python 3.11.9, Qiskit 2.5.2, Aer 0.17.2 (`requirements.txt`). All randomness is seeded; reruns reproduce the CSVs bit-for-bit up to floating-point summation order.

## 9. Limitations (stated, not hidden)

- **Small problem sizes.** Exact-statevector QAOA limits executed suites to ~18 qubits (6×3). The default 10×4 headline instance (40 qubits) is fully characterized classically (exact + baselines); QAOA at that size needs MPS/tensor-network or hardware experiments — not run here, hence no claims about it.
- **Simulation-only results.** Every quantum number in this README comes from noiseless or explicitly-depolarizing-noised simulation; no hardware was executed and no device-calibrated noise model was used.
- **Classical methods strongly outperform both QAOA variants on solution quality**: SA reaches 1.18× the optimum on average and random best-of-10k finds it exactly on every suite instance, while typical QAOA samples sit ~2000–2400× above the optimum.
- **Noise destroys part of XY's feasibility guarantee**: under depolarizing noise (1e-3/1e-2), sampled feasibility drops from 100 % to 62 %. The guarantee is a property of ideal circuits only.
- **No quantum advantage is demonstrated or claimed** — not in solution quality, not in runtime (simulation runtimes are irrelevant to any hardware claim), and none should be inferred.
- Capacity handling is soft-by-design (proportional targets + diagnostics), chosen for QUBO cleanliness; instances are pre-validated so optima never overload.
- Exact penalty calibration `P*` requires full 2^Q enumeration; for instances beyond ~26 qubits a penalty must be bounded analytically instead — out of scope here.
- COBYLA converges early (~27–55 evals); other optimizers/budgets unexplored (out of scope).
- Advanced variants (warm start, CVaR, ma-QAOA) deliberately not implemented — correctness first.
- The DLA/trainability discussion in the old README was unverifiable and has been removed.

## 10. Citation / provenance

Built on top of Team ZETA's Mini Alexandria Quantum Hackathon project (May 2026). The audited originals live in `notebooks/`. Every retained component is listed in §4 with its verification test; everything else was rebuilt.
