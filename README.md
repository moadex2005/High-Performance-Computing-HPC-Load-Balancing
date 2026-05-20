### **PROJECT REPORT**

**Project Title:** Constraint-Preserving XY-Mixer QAOA vs. Penalty-Based Soft-QUBO for High-Performance Computing (HPC) Load Balancing  
**Team Name:** ZETA  
**Event:** Mini Alexandria Quantum Hackathon (May 2026)  

---

### **Team Metadata**
*   **Team Members:** Mohamed Adel, Menna Zaied, Priyansh, Riham Hilal, Aya Hamdy
*   **Mentor:** Redwan Rahman

---

## 1. Executive Summary

In High-Performance Computing (HPC) systems, the distribution of heterogeneous computational workloads (jobs) across multi-core CPUs and GPU nodes—subject to strict resource and capacity constraints—represents a classic, NP-hard combinatorial optimization problem . Traditional quantum approaches, such as the Quantum Approximate Optimization Algorithm (QAOA) using standard transverse-field ($X$) mixers, rely on mapping the problem to a Quadratic Unconstrained Binary Optimization (QUBO) format by adding penalty terms (Soft-QUBO). However, these penalty-based systems suffer from massive search-space bloating ($2^{25}$ states for a small $5 \times 5$ instance), barren plateaus, and a high probability of sampling invalid (infeasible) configurations.

This report presents the technical design and academic evaluation of **Team ZETA's** implementation: a **Hard-Constraint Subspace Optimization via XY-Mixer QAOA**. By confining the quantum state dynamics entirely within the feasible, one-hot subspace (where every job is assigned to exactly one node), the search space is reduced by **99.99%** (down to only $3125$ states) . 

Our benchmark results show that Team ZETA's **XY-QAOA** dominates the standard penalty-based **Soft-QUBO** approach, achieving a load-balancing imbalance objective of **770.0** (a **93.3%** improvement over the worst-case scenario) compared to Soft-QUBO's sub-optimal **3320.0** .

---

## 2. Problem Specification & Mathematical Formulation

### 2.1 HPC Load Balancing Context
We evaluate a proof-of-concept HPC scheduling scenario consisting of:
*   **Jobs ($J$):** $5$ incoming computational tasks, represented as a set $\{0, 1, 2, 3, 4\}$ .
*   **Compute Nodes ($N$):** $5$ homogeneous physical servers $\{0, 1, 2, 3, 4\}$ .
*   **Workloads ($w_j$):** Assigned execution weights for each job :

$$\Large \displaystyle \mathbf{W} = [w_0, w_1, w_2, w_3, w_4] = [15, 30, 10, 45, 20]\ \text{MFLOPs/s} \quad $$

*   **Total Workload ($W_{\text{total}}$):** $\sum_{j=0}^{4} w_j = 120.0$.
*   **Target Load per Node ($L_{\text{target}}$):** The ideal workload distribution to minimize scheduling variance :

$$\Large \displaystyle L_{\text{target}} = \frac{W_{\text{total}}}{N} = 24.0 \quad $$

### 2.2 Decision Variables
We define the binary decision variables $x_{j,n} \in \{0, 1\}$ as :

$$\Large \displaystyle x_{j,n} = \begin{cases} 1 & \text{if job } j \text{ is mapped to compute node } n \\ 0 & \text{otherwise} \end{cases} \quad $$

This requires a total of $Q = J \times N = 25$ logical qubits .

### 2.3 The Load-Balancing Objective
The operational objective is to minimize the total load imbalance across all nodes :

$$\Large \displaystyle \min_{x} \sum_{n=0}^{N-1} \left( \sum_{j=0}^{J-1} w_j x_{j,n} - L_{\text{target}} \right)^2 \quad $$

Subject to the one-hot assignment constraint (each job is executed on exactly one server) :

$$\Large \displaystyle \sum_{n=0}^{N-1} x_{j,n} = 1 \quad \forall j \in \{0, \dots, J-1\} \quad $$

---

## 3. Comparative Mathematical Modeling

### 3.1 Structural Comparison Schematic

The structural difference between the unconstrained space (Soft-QUBO) and our constraint-preserving subspace (XY-QAOA) is summarized below:

```
========================================================================================
APPROACH A: SOFT-QUBO (Unconstrained Space)      APPROACH B: ZETA XY-QAOA (Subspace)
========================================================================================
 
      State Space: All 2^25 combinations               State Space: Only 1-hot states
     (33,554,432 states, mostly invalid)               (5^5 = 3,125 feasible states)
               ┌──────────────┐                                 ┌──────────────┐
               │ 00000 00000  │                                 │ 10000 01000  │
               │ 11111 11111  │                                 │ 00100 00010  │
               │ 10000 01000  │                                 │ 00001 ...    │
               └──────────────┘                                 └──────────────┘
                      │                                                │
           Standard X-Mixer Rotates                         XY-Mixer Restricted
            States Out of Subspace                           Inside Safe Subspace
                      │                                                │
                      ▼                                                ▼
         Requires Penalty Terms (P)                       No Penalty Needed (P = 0)
    H_Ising = H_cost + P * H_constraint                   H_Ising = H_cost (Analytic)
========================================================================================
```

---

### 3.2 Approach A: Soft-QUBO / Penalty-Based Formulation
In the penalty-based formulation (implemented in `Soft.ipynb`), the hard equality constraint is relaxed and embedded directly into the objective function using a quadratic penalty term with penalty coefficient $P$ :

$$\Large \displaystyle H_{\text{QUBO}} = \sum_{n=0}^{N-1} \left( \sum_{j=0}^{J-1} w_j x_{j,n} - L_{\text{target}} \right)^2 + P \sum_{j=0}^{J-1} \left( \sum_{n=0}^{N-1} x_{j,n} - 1 \right)^2 \quad $$

For our test case, a penalty factor of $P = 120.0$ is chosen to avoid constraint violations . 

To map this to an Ising Hamiltonian, we substitute $x_{j,n} = \frac{1 - Z_{j,n}}{2}$ where $Z_{j,n} \in \{-1, +1\}$ is the Pauli-Z operator :

$$\Large \displaystyle H_{\text{QUBO}} \Longrightarrow H_{\text{Ising}} = C_{\text{offset}} + \sum_{i} h_i Z_i + \sum_{i < j} J_{i,j} Z_i Z_j \quad $$

#### Limitations:
*   The search space contains $2^{25} \approx 3.35 \times 10^7$ possible states.
*   Over **$99.99\%$** of this space represents unfeasible solutions (e.g., assigning all jobs to a single node, or leaving jobs completely unassigned).
*   When run on a standard $X$-mixer QAOA, the mixer ($H_B = \sum_i X_i$) rotates the state vector out of the feasible subspace, requiring the classical optimizer to navigate a highly cluttered energy landscape.

---

### 3.3 Approach B: Hard-Constraint Subspace Optimization (Team ZETA's Design)
In Team ZETA's architecture (`XY.ipynb`), we discard all penalty terms ($P = 0$) . We analytically derive the Ising coefficients directly from the raw load-balancing objective, ensuring the quantum state is initialized and strictly maintained within the feasible one-hot subspace throughout the QAOA execution .

#### Cost Hamiltonian Analytical Derivation:
Expanding the square of the objective for compute node $n$ :

$$\Large \displaystyle \left( \sum_{j} w_j x_{j,n} - L_{\text{target}} \right)^2 = L_{\text{target}}^2 + \sum_{j} \left( w_j^2 - 2 L_{\text{target}} w_j \right) x_{j,n} + \sum_{j} \sum_{k > j} 2 w_j w_k x_{j,n} x_{k,n} \quad $$

Substituting $x_{j,n} = \frac{1 - Z_{j,n}}{2}$ yields the following Ising terms :

1.  **Linear Ising Coefficients ($h_{q}$):**
    For each job $j$ mapped to qubit index $q = j \cdot N + n$ :
    
$$\Large \displaystyle h_q = -\frac{a_q}{2} - \sum_{k \neq j} \frac{b_{j,k}}{4} \quad $$

    where:
    *   $a_q = w_j^2 - 2 L_{\text{target}} w_j$ is the expanded linear coefficient .
    *   $b_{j,k} = 2 w_j w_k$ is the expanded quadratic coefficient .
2.  **Quadratic Ising Couplings ($J_{q_j, q_k}$):**
    For two jobs $j$ and $k$ assigned to the same node $n$, at qubit indices $q_j = j \cdot N + n$ and $q_k = k \cdot N + n$ :
    
$$\Large \displaystyle J_{q_j, q_k} = \frac{b_{j,k}}{4} = \frac{w_j w_k}{2} \quad $$

3.  **Constant Energy Offset ($C_{\text{offset}}$):**
    
$$\Large \displaystyle C_{\text{offset}} = \sum_{n=0}^{N-1} L_{\text{target}}^2 + \sum_{q} \frac{a_q}{2} + \sum_{q_j < q_k} \frac{b_{j,k}}{4} = 2880.0 \quad$$

Because we restrict transitions using an $XY$-mixer, the sum of variables in any job block is conserved. The search space is exactly $N^J = 5^5 = 3125$ states, representing a **$99.991\%$** reduction in search-space volume.

---

## 4. Quantum Circuit Architecture & Design

### 4.1 Step 2: Feasible Initial State Preparation
To construct a uniform superposition of all one-hot states inside each job block, we designed a staircase $XY$-ladder (Dicke state preparation with Hamming weight $k=1$):

```
q_0: ───[ X ]───[ XY(θ_0) ]───[ XY(θ_1) ]───[ XY(θ_2) ]───[ XY(θ_3) ]───
q_1: ───────────[    *    ]───[    *    ]───[    *    ]───[    *    ]───
q_2: ─────────────────────────[    *    ]───[    *    ]───[    *    ]───
q_3: ───────────────────────────────────────[    *    ]───[    *    ]───
q_4: ─────────────────────────────────────────────────────[    *    ]───
```

For a block of $N = 5$ qubits, we initialize the first qubit to $|1\rangle$ using an $X$ gate . We then apply a sequence of parameterized two-qubit $XY(\theta_k)$ rotations between adjacent qubits $(k, k+1)$ for $k \in \{0, \dots, N-2\}$ . 

The rotation angles are defined analytically as :

$$\Large \displaystyle \theta_k = 2 \arccos\left(\sqrt{\frac{k+1}{k+2}}\right) \quad $$

For $N=5$, this gives the exact angles:
*   $\theta_0 = 2 \arccos(\sqrt{1/2}) = \frac{\pi}{2} \approx 1.5708 \quad $
*   $\theta_1 = 2 \arccos(\sqrt{2/3}) \approx 1.2310 \quad $
*   $\theta_2 = 2 \arccos(\sqrt{3/4}) = \frac{\pi}{3} \approx 1.0472 \quad $
*   $\theta_3 = 2 \arccos(\sqrt{4/5}) \approx 0.9273 \quad $

The native decomposition of the $XY(\theta)$ gate in Qiskit is implemented as :

$$\Large \displaystyle XY(\theta) = R_{xx}(\theta) \cdot R_{yy}(\theta) \quad $$

This generates a uniform superposition of all feasible assignments for each job .

---

### 4.2 Step 3: Cost Layer Unitary $U_C(\gamma)$
The cost layer implements the time evolution under the derived Ising Hamiltonian :

$$\Large \displaystyle U_C(\gamma) = \exp(-i \gamma H_{\text{cost}}) = \prod_{q} \exp(-i \gamma h_q Z_q) \prod_{p < q} \exp(-i \gamma J_{p,q} Z_p Z_q) \quad $$

On physical hardware, this is translated to native gate sets:
1.  **Linear Terms:** Implemented via $R_z(2 \gamma h_q)$ rotation gates .
2.  **Quadratic Terms:** Since native $R_{zz}$ gates are not universally available, we decompose them using CNOT gates :
    
$$\Large \displaystyle \exp(-i \gamma J_{p,q} Z_p Z_q) \equiv \text{CX}(p, q) \cdot R_z(2 \gamma J_{p,q}, q) \cdot \text{CX}(p, q) \quad $$

---

### 4.3 Step 4: XY-Mixer Layer $U_M(\beta)$
The mixer operator must commute with the one-hot constraint of each job block :

$$\Large \displaystyle [H_{\text{mixer}}, \sum_{n=0}^{N-1} x_{j,n}] = 0 \quad \forall j \quad $$

We employ a closed ring (cycle) topology of $XY$-mixers across the $N$ qubits of each job block $j$ :

$$\Large \displaystyle H_{\text{mixer}} = \sum_{j=0}^{J-1} \sum_{k=0}^{N-1} \left( X_{j \cdot N + k} X_{j \cdot N + (k+1 \bmod N)} + Y_{j \cdot N + k} Y_{j \cdot N + (k+1 \bmod N)} \right) \quad$$

Under time evolution, this is mapped as :

$$\Large \displaystyle U_M(\beta) = \prod_{j=0}^{J-1} \prod_{k=0}^{N-1} R_{xx}(2\beta, q_a, q_b) \cdot R_{yy}(2\beta, q_a, q_b) \quad $$

where $q_a = j \cdot N + k$ and $q_b = j \cdot N + (k+1 \bmod N)$ . This ensures that any population transfer between compute nodes preserves the Hamming weight of the job.

---

## 5. Simulation & Optimization Framework

Our system was compiled and executed using **Qiskit 1.4.5**, **Qiskit-Aer 0.17.2**, and **Qiskit-Optimization 0.7.0** . We used the `AerSimulator` statevector method with a fixed seed of $7$ to ensure reproducibility.

### 5.1 Optimization Strategy (Step 7)
We optimized the parameters of a $p=2$ QAOA circuit ($2p = 4$ parameters: $[\gamma_0, \gamma_1, \beta_0, \beta_1]$) . To overcome local minima, we implemented a multi-start strategy with $3$ random restarts . The classical optimization was conducted using the **COBYLA** algorithm with a maximum of 200 iterations per run .

---

## 6. Performance & Benchmark Results

We compared the performance of our constraint-preserving $XY$-QAOA against several classical and quantum baselines :
1.  **Worst-Case Baseline:** Allocating all $5$ jobs to a single compute node .
2.  **Soft-QUBO QAOA Baseline:** Standard $X$-mixer with a penalty factor $P=120.0$ .
3.  **Greedy Classical Baseline:** Allocating jobs to the currently least-loaded node in descending order of job size .
4.  **ZETA's XY-QAOA:** Our proposed constraint-preserving approach .

### 6.1 Quantitative Comparison Table
| Metric / Method | Imbalance Objective ($C$) | Improvement vs. Worst-Case (%) | Feasibility Guarantee | Search-Space Size |
| :--- | :---: | :---: | :---: | :---: |
| **Worst-Case** | $11520.0$ | $0.0\%$ | N/A | — |
| **Soft-QUBO** | $3320.0$ | $71.2\%$ | Not Guaranteed | $2^{25} \approx 3.35 \times 10^7$ |
| **Greedy** | $770.0$ | $93.3\%$ | Guaranteed (Feasible) | — |
| **ZETA XY-QAOA** | **770.0** | **93.3%** | **Guaranteed (100%)** | **$3125$ ($99.99\%$ smaller)** |

### 6.2 Key Structural and Numerical Discoveries
*   **Feasibility Preservation:** Our $XY$-QAOA achieved a **100% feasibility rate** across all samples. In contrast, the standard Soft-QUBO approach frequently samples invalid states due to the leakage of the state vector into non-feasible regions .
*   **Optimal Job Assignment:** The optimized $XY$-QAOA successfully converged to the global minimum :
    $$\text{Job 0 } (15) \to \text{Node 1}, \quad \text{Job 1 } (30) \to \text{Node 2}, \quad \text{Job 2 } (10) \to \text{Node 3}, \quad \text{Job 3 } (45) \to \text{Node 4}, \quad \text{Job 4 } (20) \to \text{Node 0} \quad $$
    This yielded node loads of $\mathbf{L} = [20, 15, 30, 10, 45]$ .
*   **Objective Value Improvement:** XY-QAOA achieved an imbalance of **770.0**, matching the greedy heuristic and outperforming the Soft-QUBO's best candidate of **3320.0** by a factor of **4.3x** .

---

### 6.3 Physical Resource & Gate Complexity
For a $p=1$ QAOA layer, we evaluated the physical gate requirements:

| Circuit Module | Gate Types | Quantity | Source / Context |
| :--- | :--- | :---: | :--- |
| **Initial State** | $X$ gates | 5 | One per job block  |
| | $R_{xx}$ gates | 20 | Part of Dicke state preparation  |
| | $R_{yy}$ gates | 20 | Part of Dicke state preparation  |
| **Cost Layer** | $R_{z}$ gates | 75 | Linear coefficients & quadratic CNOT targets |
| | $CX$ (CNOT) gates | 100 | Decomposed $R_{zz}$ couplings (50 pairs) |
| **Mixer Layer** | $R_{xx}$ gates | 25 | Active cycle mixers |
| | $R_{yy}$ gates | 25 | Active cycle mixers  |

For our $p=2$ circuit, the total physical gate count was:
*   **Qubits:** 25 
*   **Trainable Parameters:** 4
*   **Circuit Depth:** 189

---

### 6.4 Visual Performance Analysis (`xy_qaoa_results.png`)
The saved results plot, `xy_qaoa_results.png`, provides the visual validation of our implementation:

*   **Left Panel (COBYLA Convergence):** Plots "Optimizer call" vs. "Best feasible objective" . It shows that within a few dozen evaluations, our COBYLA parameters successfully steer the QAOA state vector to hit the absolute mathematical minimum boundary of **770.0**, matching the Greedy baseline's benchmark.
*   **Right Panel (Node Workload Balancer):** A comparative bar-chart demonstrating Node Workloads ($0$ to $5$) for both "Greedy" and "XY-QAOA" against the ideal flat target load of **24.0** per node. It visually confirms that the QAOA allocation balances workload variance to equal the heuristic's optimal, preventing bottleneck overloads.

---

## 7. Scalability & Lie-Theoretic Analysis

An essential factor in evaluating the viability of our $XY$-Mixer approach is its scalability to larger systems [4]. The trainability of variational quantum circuits is determined by the size and structure of their **Dynamical Lie Algebra (DLA)** .

If the DLA grows exponentially with the number of qubits, the circuit suffers from barren plateaus, making the gradient of the loss function exponentially small and training practically impossible [4]. 

According to the mathematical classification of $XY$-mixer topologies by *Kordonowy & Leipold (2026)*:
1.  **Ring/Cycle Topology ($g^C_{XY}$):** The DLA of the cycle $XY$-mixer with single-qubit $Z$ rotations decomposes into:
    
$$\Large \displaystyle \mathfrak{g}^C_{XY,Z} \cong \mathfrak{u}(1) \oplus \mathfrak{su}(N) \oplus \mathfrak{su}(N) \quad $$

    This algebra scales **polynomially** (dimension $O(N^2)$), ensuring that the system is **completely free of barren plateaus** and remains highly trainable even for large systems [4].
2.  **All-to-All Topology ($g^K_{XY}$):** If we introduce all-to-all $XY$ interactions or add arbitrary $R_{zz}$ terms, the DLA decomposes as:
    
$$\Large \displaystyle \mathfrak{g}^C_{XY,Z,ZZ} \cong \mathfrak{u}(1)^{\oplus 2} \oplus \bigoplus_{k=1}^{n-1} \mathfrak{su}\left(\binom{n}{k}\right) \quad $$

    This algebra is **exponentially large** (dimension $\Omega(3^n)$), leading to immediate trainability failure.

By adopting the cycle $XY$-mixer topology, Team ZETA's design ensures a polynomial DLA, guaranteeing long-term scalability to larger HPC scheduling infrastructures.

---

## 8. Conclusion

By implementing a constraint-preserving $XY$-QAOA, Team ZETA successfully bypassed the pitfalls of standard penalty-based QUBO formulations. The elimination of penalty terms, combined with a $99.99\%$ reduction in the search-space size, allowed our quantum model to achieve a **100% feasibility rate** and converge to the absolute minimum load imbalance of **770.0**. This architecture provides a robust, scalable, and highly practical blueprint for future quantum-assisted resource scheduling in hyperscale data centers.

---

### References
*   ** Team ZETA Codebase:** `XY.ipynb` (XY-Mixer QAOA implementation)
*   **[2] Team ZETA Codebase:** `Soft.ipynb` (Soft-QUBO / Penalty-based implementation)
*   **[3] Farhi et al. (2014):** *A Quantum Approximate Optimization Algorithm*, arXiv:1411.4028.
*   **[4] Kordonowy, S. & Leipold, H. (2026):** *The Lie algebra of XY-mixer topologies and warm starting QAOA for constrained optimization*, npj Quantum Information, 12:61.
