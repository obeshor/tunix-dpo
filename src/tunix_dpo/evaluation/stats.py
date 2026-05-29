"""Pure statistical helpers used across the evaluation suite.

Implements:
- Non-parametric bootstrap confidence intervals.
- Cohen's h effect size for two proportions.
- Relative-change calculation.
- Effect-size verbal interpretation.

All functions are pure: numeric in, numeric out.
"""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(
    values: list[float] | np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval for the mean."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = arr.size
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = sample.mean()

    alpha = 1.0 - confidence
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for the difference between two proportions."""
    p1 = max(0.0, min(1.0, float(p1)))
    p2 = max(0.0, min(1.0, float(p2)))
    return 2.0 * math.asin(math.sqrt(p1)) - 2.0 * math.asin(math.sqrt(p2))


def interpret_cohens_h(h: float) -> str:
    """Cohen-style verbal label: 'small', 'medium', or 'large'."""
    h_abs = abs(h)
    if h_abs < 0.2:
        return "small"
    if h_abs < 0.5:
        return "medium"
    return "large"


def relative_change(baseline: float, new: float) -> float:
    """Percent change from ``baseline`` to ``new``. Returns NaN on zero baseline."""
    if baseline == 0:
        return float("nan")
    return 100.0 * (new - baseline) / baseline
