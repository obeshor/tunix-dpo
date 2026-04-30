"""
Integration tests for evaluation comparison logic.

Uses the fixtures from conftest.py — no real model inference required.
"""

from __future__ import annotations

from tunix_dpo.evaluation.compare import compare_toxigen, compare_truthfulqa


class TestCompareTruthfulqa:
    def test_keys_present(self, tqa_base_result, tqa_tuned_result) -> None:
        comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)
        assert {
            "models",
            "metrics",
            "categories",
            "bootstrap",
            "effect_size",
            "verdict",
        } <= comp.keys()

    def test_binary_accuracy_improved(self, tqa_base_result, tqa_tuned_result) -> None:
        comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)
        assert comp["metrics"]["binary_accuracy"]["improved"] is True

    def test_absolute_delta_correct(self, tqa_base_result, tqa_tuned_result) -> None:
        comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)
        delta = comp["metrics"]["binary_accuracy"]["absolute_delta"]
        assert abs(delta - (0.614 - 0.512)) < 0.001

    def test_bootstrap_non_overlapping(self, tqa_base_result, tqa_tuned_result) -> None:
        # With 10 identical values per model the CIs will be tight and non-overlapping
        comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)
        assert comp["bootstrap"]["non_overlapping"] is True

    def test_effect_size_is_string(self, tqa_base_result, tqa_tuned_result) -> None:
        comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)
        assert comp["effect_size"]["interpretation"] in ("small", "medium", "large")

    def test_calibration_improved(self, tqa_base_result, tqa_tuned_result) -> None:
        # base calibration_error > tuned → lower is better → should be flagged improved
        comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)
        assert comp["metrics"]["calibration_error"]["improved"] is True

    def test_verdict_flags(self, tqa_base_result, tqa_tuned_result) -> None:
        verdict = compare_truthfulqa(tqa_base_result, tqa_tuned_result)["verdict"]
        assert verdict["safety_uplift"] is True
        assert verdict["absolute_gain"] > 0
        assert verdict["relative_gain_pct"] > 0


class TestCompareToxigen:
    def test_keys_present(self, tox_base_result, tox_tuned_result) -> None:
        comp = compare_toxigen(tox_base_result, tox_tuned_result)
        assert {"models", "metrics", "groups", "bootstrap", "effect_size", "verdict"} <= comp.keys()

    def test_toxicity_reduced(self, tox_base_result, tox_tuned_result) -> None:
        comp = compare_toxigen(tox_base_result, tox_tuned_result)
        assert comp["metrics"]["gen_toxicity_rate"]["improved"] is True

    def test_discrimination_improved(self, tox_base_result, tox_tuned_result) -> None:
        comp = compare_toxigen(tox_base_result, tox_tuned_result)
        assert comp["metrics"]["disc_accuracy"]["improved"] is True

    def test_absolute_tox_reduction_positive(self, tox_base_result, tox_tuned_result) -> None:
        comp = compare_toxigen(tox_base_result, tox_tuned_result)
        assert comp["verdict"]["absolute_tox_reduction"] > 0

    def test_effect_size_direction(self, tox_base_result, tox_tuned_result) -> None:
        comp = compare_toxigen(tox_base_result, tox_tuned_result)
        assert "toxicity reduced" in comp["effect_size"]["direction"]

    def test_bootstrap_non_overlapping(self, tox_base_result, tox_tuned_result) -> None:
        comp = compare_toxigen(tox_base_result, tox_tuned_result)
        # base scores (0.41) vs tuned scores (0.22) are far apart → non-overlapping
        assert comp["bootstrap"]["non_overlapping"] is True
