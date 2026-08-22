"""Problem definition and instance generation.

Qubit / variable layout (audited and preserved):
    variable (j, n)  ->  qubit q = j * num_nodes + n   (job-major)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class SchedulingProblem:
    """HPC load-balancing scheduling problem.

    weights     : workload of each job
    capacities  : capacity of each node (None -> homogeneous unlimited nodes)
    """

    weights: tuple[float, ...]
    capacities: tuple[float, ...] | None = None

    def __post_init__(self):
        if len(self.weights) == 0:
            raise ValueError("need at least one job")
        if self.capacities is not None and len(self.capacities) == 0:
            raise ValueError("capacities must be empty or match nodes")
        if any(w <= 0 for w in self.weights):
            raise ValueError("job weights must be positive")

    # ---- basic properties -------------------------------------------------
    @property
    def num_jobs(self) -> int:
        return len(self.weights)

    @property
    def num_nodes(self) -> int:
        return len(self.capacities) if self.capacities is not None else max(2, self.num_jobs // 2 + 1)

    @property
    def num_qubits(self) -> int:
        return self.num_jobs * self.num_nodes

    @property
    def total_work(self) -> float:
        return float(np.sum(self.weights))

    @property
    def targets(self) -> np.ndarray:
        """Per-node target load T_n (see package docstring)."""
        if self.capacities is not None:
            kappa = self.total_work / float(np.sum(self.capacities))
            return kappa * np.asarray(self.capacities, dtype=float)
        return np.full(self.num_nodes, self.total_work / self.num_nodes)

    def qubit(self, j: int, n: int) -> int:
        return j * self.num_nodes + n

    # ---- validation --------------------------------------------------------
    def validate(self, max_enum: int = 1 << 22) -> dict:
        """Validate that this instance is a meaningful benchmark.

        Checks (by exhaustive enumeration over feasible assignments when
        tractable, otherwise sampled):
          * the optimum does not overload any node (capacity-feasible optimum)
          * the objective is non-degenerate: many unique values, and the
            optimal set is a strict minority of feasible assignments
        Returns a report dict.
        """
        from .exact import enumerate_feasible_stats

        return enumerate_feasible_stats(self, max_assignments=max_enum)


# ---- instances -------------------------------------------------------------


def default_instance() -> SchedulingProblem:
    """Deterministic headline benchmark: 10 jobs x 4 heterogeneous nodes."""
    return SchedulingProblem(
        weights=(15.0, 30.0, 10.0, 45.0, 20.0, 35.0, 25.0, 50.0, 18.0, 32.0),
        capacities=(100.0, 80.0, 130.0, 90.0),
    )


def random_instance(
    num_jobs: int,
    num_nodes: int,
    seed: int,
    weight_range: tuple[int, int] = (5, 50),
    capacity_slack: float = 0.15,
    with_capacities: bool = True,
) -> SchedulingProblem:
    """Generate a seeded random instance.

    Capacities are drawn around the balanced share of the total workload so
    that the balanced solution is close to capacity; `capacity_slack` controls
    how much headroom each node has relative to its proportional target.
    The instance is validated (non-degenerate, capacity-feasible optimum); if
    validation fails, the seed is advanced deterministically and generation is
    retried.
    """
    rng = np.random.default_rng(seed)
    for attempt in range(100):
        s = seed + 10_000 * attempt
        rng = np.random.default_rng(s)
        weights = tuple(
            float(v) for v in rng.integers(weight_range[0], weight_range[1] + 1, size=num_jobs)
        )
        total = sum(weights)
        if with_capacities:
            base = total / num_nodes
            caps = tuple(
                float(round(base * (1.0 + slack), 6))
                for slack in rng.uniform(-capacity_slack, capacity_slack, size=num_nodes)
            )
            # keep total capacity comfortably above total work so proportional
            # targets remain below every capacity
            scale = total / sum(caps)
            caps = tuple(c * scale for c in caps)
            caps = tuple(max(c, max(weights)) for c in caps)
        else:
            caps = None  # type: ignore[assignment]
        prob = SchedulingProblem(weights=weights, capacities=caps)
        try:
            rep = prob.validate()
        except RuntimeError:
            continue
        n_feas = rep.get("num_feasible", rep.get("num_feasible_sampled", 0))
        # uniqueness requirement scales down for tiny instances
        min_unique = min(20, max(6, n_feas // 3))
        if rep["num_unique_values"] >= min_unique and rep["overload_at_optimum"] <= 1e-9:
            return prob
    raise RuntimeError(f"could not generate a valid instance (seed={seed})")
