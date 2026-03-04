# CABOP: Cost-Aware Bayesian Optimization

A Python library for Bayesian optimization that incorporates fabrication and acquisition costs into the optimization process. Designed for experimental design scenarios where changing parameters has associated costs.

## Features

- **Cost-aware acquisition function**: Balances expected improvement with fabrication costs
- **Parameter grouping**: Organize parameters into groups with independent cost tracking
- **Fabrication history**: Tracks previous configurations to enable cost savings through reuse
- **Flexible cost model**: Supports three cost tiers per group:
  - `unchanged`: Reuse the most recent configuration (cheapest)
  - `swapped`: Reuse a configuration from history (medium cost)
  - `acquired`: Fabricate a new configuration (most expensive)
- **Prefabricated value support**: Snap proposed values to available prefabricated options
- **Gaussian Process surrogate**: Uses scikit-learn's GP with Matern kernel

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

Install uv if you haven't already:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd CABOP
uv sync
```

This will create a virtual environment and install all required dependencies.

## Quick Start

```python
import numpy as np
from bayesopt import BayesOpt, BOSpace

# Define parameter space with cost structure
parameters = {
    "groups": ["hardware", "software"],
    "cost": {
        "hardware": {"unchanged": 1, "swapped": 10, "acquired": 100},
        "software": {"unchanged": 1, "swapped": 10, "acquired": 100},
    },
    "actual_cost": {
        "hardware": {"unchanged": 1, "swapped": 10, "acquired": 100},
        "software": {"unchanged": 1, "swapped": 10, "acquired": 100},
    },
    "parameters": {
        "x1": {"bound": np.array([-2, 2]), "tolerance": 0.05, "group": "hardware"},
        "x2": {"bound": np.array([-1, 3]), "tolerance": 0.05, "group": "software"},
    },
}

# Create optimizer
space = BOSpace(parameters=parameters)
optimizer = BayesOpt(space, ifCost=True)

# Optimization loop
for i in range(25):
    # Get next point to evaluate
    x_candidate, result = optimizer.ask(n_init=5)

    # Apply fabrication constraints
    costs, x_realized = optimizer.select_sample(x_candidate)

    # Evaluate your objective function
    y = your_objective(x_realized)

    # Report observation
    optimizer.tell(x_realized, y, x_candidate)

# Access best result
print(f"Best value: {optimizer.current_best['y']}")
print(f"Best point: {optimizer.current_best['x']}")
```

## Running the Example

```bash
uv run python example.py
```

## API Reference

### BOSpace

Defines the parameter space for optimization.

```python
@dataclass
class BOSpace:
    parameters: dict  # Contains groups, cost, actual_cost, and parameters
```

The `parameters` dict should contain:
- `groups`: List of group names (e.g., `["hardware", "software"]`)
- `cost`: Expected costs used in the acquisition function
- `actual_cost`: Realized costs for tracking
- `parameters`: Dict mapping parameter names to their config:
  - `bound`: `np.ndarray` of `[lower, upper]` bounds
  - `tolerance`: Float tolerance for matching previous samples
  - `group`: Group name this parameter belongs to

### BayesOpt

The main optimizer class.

```python
optimizer = BayesOpt(
    space: BOSpace,      # Parameter space definition
    ifCost: bool = True, # Use cost-aware acquisition (True) or standard EI (False)
    random_state: int = None  # Random seed for reproducibility
)
```

**Methods:**

- `ask(n_init=5)` - Propose the next point to evaluate. Returns `(x_candidate, result)`.
- `tell(x_realized, y, x_intended=None, update_rule="actual")` - Report an observation.
- `select_sample(x, prefab=None)` - Apply fabrication constraints. Returns `(costs, x_realized)`.

**Update rules for `tell()`:**
- `"actual"`: Fit GP on realized points only
- `"intended"`: Fit GP on intended points only
- `"both"`: Fit GP on both realized and intended points

## Benchmark Functions

The library includes several standard benchmark functions in `utils/benchmarks.py`:

- `rosenbrock` - 2D Rosenbrock function (minimum at (1,1) = 0)
- `rosenbrock_nd` - N-dimensional Rosenbrock
- `goldstein_price` - 2D Goldstein-Price (minimum at (0,-1) = 3)
- `ackley` - 2D Ackley function (minimum at (0,0) = 0)
- `levy` - 2D Levy function (minimum at (1,1) = 0)
- `forrester` - 1D Forrester function

All functions support optional additive and multiplicative noise:

```python
from utils.benchmarks import rosenbrock

# Clean evaluation
y = rosenbrock(x)

# Noisy evaluation
y = rosenbrock(x, add_noise=0.1, mult_noise=0.1)
```

## Dependencies

- numpy
- scipy
- scikit-learn
- loguru
- matplotlib

## License

[Add your license here]
