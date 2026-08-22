"""ZETA scheduler: a verified benchmark for quantum vs classical HPC load balancing.

Mathematical formulation (canonical, used identically by every method):

    Variables        x[j, n] in {0,1}, qubit index q = j * num_nodes + n
    Assignment       each job runs on exactly one node:  sum_n x[j,n] = 1  (forall j)
    Node load        load_n = sum_j w_j x[j,n]
    Target loads     T_n = kappa * capacity_n  if capacities given (kappa = W / sum_c),
                     otherwise T_n = W / num_nodes   (W = total workload)
    Objective        F(x) = sum_n (load_n - T_n)^2            (minimize)
    Capacity metric  overload_n = max(0, load_n - capacity_n) (reported diagnostic;
                     instances are validated so that optimal assignments have zero overload)

The one-hot-per-job constraint is the only hard constraint. It is handled by
penalty (Soft-QUBO) or by subspace-preserving XY evolution (XY-QAOA).
"""

from .problem import SchedulingProblem, default_instance, random_instance
from .objective import decode_assignment, is_feasible, objective_value, violation_value

__version__ = "2.0.0"
__all__ = [
    "SchedulingProblem",
    "default_instance",
    "random_instance",
    "decode_assignment",
    "is_feasible",
    "objective_value",
    "violation_value",
]
