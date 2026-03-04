"""
Benchmark objective functions for Bayesian optimization testing.

All functions follow the signature:
    f(x, add_noise=None, mult_noise=None) -> float

Where:
    - x: numpy array of input values
    - add_noise: if not None, adds Gaussian noise with this std dev
    - mult_noise: if not None, multiplies by Gaussian noise with this std dev
"""

import numpy as np


def _apply_noise(
    val: float,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """Apply multiplicative and/or additive Gaussian noise to a value."""
    if mult_noise is not None:
        val *= np.random.normal(loc=1.0, scale=mult_noise)
    if add_noise is not None:
        val += np.random.normal(loc=0.0, scale=add_noise)
    return max(0.0, val)


def rosenbrock(
    x: np.ndarray,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """
    Rosenbrock function (2D).

    Global minimum: f(1, 1) = 0
    Typical bounds: x1 in [-2, 2], x2 in [-1, 3]
    """
    x1, x2 = x[0], x[1]
    val = (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2
    return _apply_noise(val, add_noise, mult_noise)


def rosenbrock_nd(
    x: np.ndarray,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """
    Generalized Rosenbrock function for n dimensions.

    Global minimum: f(1, 1, ..., 1) = 0
    Typical bounds: xi in [-5, 10]
    """
    val = float(np.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2))
    return _apply_noise(val, add_noise, mult_noise)


def goldstein_price(
    x: np.ndarray,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """
    Goldstein-Price function (2D).

    Global minimum: f(0, -1) = 3
    Typical bounds: xi in [-2, 2]
    """
    x1, x2 = x[0], x[1]
    term1 = 1 + (x1 + x2 + 1) ** 2 * (
        19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
    )
    term2 = 30 + (2 * x1 - 3 * x2) ** 2 * (
        18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
    )
    val = term1 * term2
    return _apply_noise(val, add_noise, mult_noise)


def ackley(
    x: np.ndarray,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """
    Ackley function (2D).

    Global minimum: f(0, 0) = 0
    Typical bounds: xi in [-5, 5]
    """
    x1, x2 = x[0], x[1]
    a, b, c = 20, 0.2, 2 * np.pi
    sum_sq = x1**2 + x2**2
    cos_sum = np.cos(c * x1) + np.cos(c * x2)
    term1 = -a * np.exp(-b * np.sqrt(0.5 * sum_sq))
    term2 = -np.exp(0.5 * cos_sum)
    val = term1 + term2 + a + np.e
    return _apply_noise(val, add_noise, mult_noise)


def levy(
    x: np.ndarray,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """
    Levy function (2D).

    Global minimum: f(1, 1) = 0
    Typical bounds: xi in [-10, 10]
    """
    x1, x2 = x[0], x[1]
    w1 = 1 + (x1 - 1) / 4
    w2 = 1 + (x2 - 1) / 4
    term1 = np.sin(np.pi * w1) ** 2
    term2 = (w1 - 1) ** 2 * (1 + 10 * np.sin(np.pi * w1 + 1) ** 2)
    term3 = (w2 - 1) ** 2 * (1 + np.sin(2 * np.pi * w2) ** 2)
    val = term1 + term2 + term3
    return _apply_noise(val, add_noise, mult_noise)


def forrester(
    x: np.ndarray,
    add_noise: float | None = None,
    mult_noise: float | None = None,
) -> float:
    """
    Forrester function (1D).

    Global minimum: f(0.757) ≈ -6.02
    Typical bounds: x in [0, 1]
    """
    x1 = x[0]
    val = (6 * x1 - 2) ** 2 * np.sin(12 * x1 - 4)
    return _apply_noise(val, add_noise, mult_noise)


# Registry of available benchmark functions
OBJECTIVE_MAP = {
    "rosenbrock": rosenbrock,
    "rosenbrock_nd": rosenbrock_nd,
    "goldstein_price": goldstein_price,
    "ackley": ackley,
    "levy": levy,
    "forrester": forrester,
}

# Keep backwards compatibility with old name
objective_map = OBJECTIVE_MAP
