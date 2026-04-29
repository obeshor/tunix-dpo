"""Unit tests for tunix_dpo.evaluation.stats."""

import math

import pytest

from tunix_dpo.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    interpret_cohens_h,
    relative_change,
)


class TestBootstrapCi:
    def test_output_shape(self) -> None:
        lo, hi = bootstrap_ci([0.5] * 100)
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_lower_less_than_upper(self) -> None:
        values = [0.1, 0.2, 0.3, 0.8, 0.9]
        lo, hi = bootstrap_ci(values, n_boot=1000)
        assert lo < hi

    def test_tight_ci_for_constant(self) -> None:
        lo, hi = bootstrap_ci([0.6] * 200, n_boot=500)
        assert hi - lo < 0.01  # essentially zero variance


class TestCohensH:
    def test_equal_proportions(self) -> None:
        assert cohens_h(0.5, 0.5) == pytest.approx(0.0)

    def test_symmetry(self) -> None:
        assert abs(cohens_h(0.7, 0.4)) == pytest.approx(abs(cohens_h(0.4, 0.7)))

    def test_clamps_out_of_range(self) -> None:
        # Should not raise even with slightly out-of-range inputs
        h = cohens_h(1.1, -0.1)
        assert math.isfinite(h)


class TestInterpretCohensH:
    def test_small(self) -> None:
        assert interpret_cohens_h(0.1) == "small"

    def test_medium(self) -> None:
        assert interpret_cohens_h(0.35) == "medium"

    def test_large(self) -> None:
        assert interpret_cohens_h(0.7) == "large"

    def test_negative_uses_abs(self) -> None:
        assert interpret_cohens_h(-0.6) == "large"


class TestRelativeChange:
    def test_positive_change(self) -> None:
        assert relative_change(0.5, 0.6) == pytest.approx(20.0)

    def test_negative_change(self) -> None:
        assert relative_change(0.4, 0.2) == pytest.approx(-50.0)

    def test_zero_base(self) -> None:
        assert relative_change(0.0, 0.5) == float("inf")
