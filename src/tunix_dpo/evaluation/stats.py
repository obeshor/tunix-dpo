"""
Statistical helpers shared by TruthfulQA and ToxiGen evaluators.

All functions are pure — no I/O, no logging.
"""

from __future__ import annotations

import math

import numpy as np


def bootstrap_ci(
    values: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Non-parametric bootstrap CI for the mean.

    Returns
    -------
    (lower, upper) at the requested confidence level.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def cohens_h(p1: float, p2: float) -> float:
    """Effect size for comparing two proportions.

    h = 2·arcsin(√p1) − 2·arcsin(√p2)

    |h| < 0.2  small
    |h| ∈ [0.2, 0.5)  medium
    |h| ≥ 0.5  large
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def relative_change(base: float, target: float) -> float:
    """Signed relative change in percent: (target - base) / base × 100."""
    if base == 0:
        return float("inf")
    return (target - base) / base * 100


def interpret_cohens_h(h: float) -> str:
    """Human-readable effect-size label."""
    ah = abs(h)
    if ah >= 0.5:
        return "large"
    if ah >= 0.2:
        return "medium"
    return "small"
