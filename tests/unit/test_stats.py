"""Unit tests for tunix_dpo.evaluation.stats."""

from __future__ import annotations

import math

import numpy as np

from tunix_dpo.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    interpret_cohens_h,
    relative_change,
)


class TestBootstrapCi:
    def test_constant_distribution_zero_width(self) -> None:
        lo, hi = bootstrap_ci([0.5] * 100, n_boot=200, seed=0)
        assert lo == 0.5
        assert hi == 0.5

    def test_normal_distribution_brackets_mean(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0.6, scale=0.1, size=300).clip(0, 1)
        lo, hi = bootstrap_ci(list(values), n_boot=500, seed=42)
        assert lo < float(np.mean(values)) < hi

    def test_empty_input_returns_nan(self) -> None:
        lo, hi = bootstrap_ci([], n_boot=100)
        assert math.isnan(lo) and math.isnan(hi)


class TestCohensH:
    def test_positive_when_p1_greater(self) -> None:
        assert cohens_h(0.6, 0.4) > 0

    def test_negative_when_p1_smaller(self) -> None:
        assert cohens_h(0.4, 0.6) < 0

    def test_zero_when_equal(self) -> None:
        assert cohens_h(0.5, 0.5) == 0.0

    def test_clamps_extreme_inputs(self) -> None:
        h = cohens_h(1.5, -0.5)
        assert isinstance(h, float)


class TestInterpretCohensH:
    def test_thresholds(self) -> None:
        assert interpret_cohens_h(0.05) == "small"
        assert interpret_cohens_h(0.21) == "medium"
        assert interpret_cohens_h(0.55) == "large"
        assert interpret_cohens_h(-0.55) == "large"


class TestRelativeChange:
    def test_increase(self) -> None:
        assert abs(relative_change(0.5, 0.6) - 20.0) < 1e-6

    def test_decrease(self) -> None:
        assert abs(relative_change(0.5, 0.4) - (-20.0)) < 1e-6

    def test_zero_baseline_nan(self) -> None:
        assert math.isnan(relative_change(0.0, 0.5))
