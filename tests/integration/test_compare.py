"""Integration tests for the comparison logic with realistic fixtures."""

from __future__ import annotations

from tunix_dpo.evaluation.compare import compare_toxigen, compare_truthfulqa


def test_compare_truthfulqa_detects_uplift(tqa_base_result, tqa_tuned_result):
    comp = compare_truthfulqa(tqa_base_result, tqa_tuned_result)

    assert comp["verdict"]["safety_uplift"] is True
    assert comp["verdict"]["absolute_gain"] > 0
    assert comp["verdict"]["relative_gain_pct"] > 0
    # Cohen's h should be positive for an improvement
    assert comp["effect_size"]["cohens_h"] > 0
    # Bootstrap CIs should be non-overlapping for this large of an uplift
    assert comp["bootstrap"]["non_overlapping"] is True


def test_compare_toxigen_detects_reduction(tox_base_result, tox_tuned_result):
    comp = compare_toxigen(tox_base_result, tox_tuned_result)

    assert comp["verdict"]["toxicity_reduced"] is True
    assert comp["verdict"]["absolute_tox_reduction"] > 0
    assert comp["verdict"]["discrimination_improved"] is True
    # h > 0 means base toxicity > tuned toxicity, i.e. reduction
    assert comp["effect_size"]["cohens_h"] > 0


def test_no_change_produces_zero_deltas(tqa_base_result):
    comp = compare_truthfulqa(tqa_base_result, tqa_base_result)
    assert comp["verdict"]["absolute_gain"] == 0
    assert comp["effect_size"]["cohens_h"] == 0.0
