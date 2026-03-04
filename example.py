"""
Example usage of Cost-Aware Bayesian Optimization (CABOP).

This script demonstrates how to set up and run the optimizer on the
Rosenbrock benchmark function with a simple two-group parameter space.
"""

import copy

import numpy as np

from bayesopt import BayesOpt, BOSpace
from utils.benchmarks import rosenbrock


# -----------------------------------------------------------------------------
# Define the parameter space
# -----------------------------------------------------------------------------

PARAMETERS = {
    # Parameter groups - each group has independent cost tracking
    "groups": ["hardware", "software"],

    # Cost model for acquisition function (soft/expected costs)
    "cost": {
        "hardware": {
            "unchanged": 1.0,    # Same as last sample
            "swapped": 10.0,    # Reuse from history
            "acquired": 100.0,  # New fabrication
        },
        "software": {
            "unchanged": 1.0,
            "swapped": 10.0,
            "acquired": 100.0,
        },
    },

    # Actual realized costs (can differ from model costs)
    "actual_cost": {
        "hardware": {
            "unchanged": 1.0,
            "swapped": 10.0,
            "acquired": 100.0,
        },
        "software": {
            "unchanged": 1.0,
            "swapped": 10.0,
            "acquired": 100.0,
        },
    },

    # Individual parameters with bounds, tolerances, and group assignments
    "parameters": {
        "x1": {
            "bound": np.asarray([-2, 2]),
            "tolerance": 0.05,  # Tolerance for matching previous samples
            "group": "hardware",
        },
        "x2": {
            "bound": np.asarray([-1, 3]),
            "tolerance": 0.05,
            "group": "software",
        },
    },
}


# -----------------------------------------------------------------------------
# Optimization loop
# -----------------------------------------------------------------------------

def run_bo(
    objective,
    optimizer: BayesOpt,
    n_init: int = 5,
    n_iter: int = 50,
    max_cost: float = -1.0,
    prefab: dict | None = None,
    update_rule: str = "actual",
):
    """
    Run Bayesian optimization loop.

    Args:
        objective: Objective function to minimize
        optimizer: BayesOpt instance
        n_init: Number of random initialization samples
        n_iter: Maximum number of iterations (if max_cost < 0)
        max_cost: Maximum cumulative cost budget (if > 0, overrides n_iter)
        prefab: Optional dict of prefabricated values to snap to
        update_rule: How to update GP ("actual", "intended", or "both")

    Returns:
        List of ProposeLocationResult objects, one per iteration
    """
    results = []
    cost_sum = 0.0
    iteration = 0

    while True:
        # Check stopping criteria before proposing
        if max_cost < 0 and iteration >= n_iter:
            break
        if max_cost > 0 and cost_sum >= max_cost:
            break

        # Propose next point
        x_candidate, result = optimizer.ask(n_init=n_init)

        # Apply fabrication constraints and get costs
        costs, x_realized = optimizer.select_sample(x_candidate, prefab=prefab)

        # Evaluate objective
        y_next = objective(x_realized, add_noise=0.1, mult_noise=0.1)

        # Report observation to optimizer
        optimizer.tell(x_realized, y_next, x_candidate, update_rule=update_rule)

        # Update result metadata
        result.realized_x = optimizer._numpy_to_design(x_realized)
        result.realized_cost = sum(costs)
        cost_sum += result.realized_cost
        result.iter_count = iteration
        result.random_phase = iteration < n_init
        result.running_cost = cost_sum
        result.current_best_val = optimizer.current_best["y"]
        result.current_best_x = optimizer.current_best["x"]
        result.actual_y = objective(x_realized, add_noise=None, mult_noise=None)
        result.expected_y = y_next
        result.regret = abs(optimizer.current_best["y"])  # Rosenbrock min is 0

        results.append(copy.deepcopy(result))
        iteration += 1

    return results


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Create parameter space and optimizer
    space = BOSpace(parameters=PARAMETERS)
    optimizer = BayesOpt(space, ifCost=True)

    # Optional: define prefabricated values to snap to
    # prefab = {
    #     "x1": [-2, -1.6, -1.2, -0.8, -0.6, -0.2, 0.2, 0.6, 0.8, 1.2, 1.6, 2.0],
    #     "x2": [-0.8, -0.6, -0.2, 0.2, 0.6, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8],
    # }

    # Run optimization
    results = run_bo(
        rosenbrock,
        optimizer,
        n_init=3,
        n_iter=25,
        max_cost=-1,
        prefab=None,
        update_rule="actual",
    )

    # Print final result
    print(f"\nOptimization complete!")
    print(f"Iterations: {len(results)}")
    print(f"Best value found: {optimizer.current_best['y']:.6f}")
    print(f"Best point: {optimizer.current_best['x']}")
    print(f"Total cost: {results[-1].running_cost:.2f}")
