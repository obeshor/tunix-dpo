"""Pure comparison of base vs tuned model evaluation results.

Takes two evaluation result dicts and produces a structured comparison
covering deltas, bootstrap confidence intervals, effect sizes, and a verdict.
No I/O, no plotting.
"""

from __future__ import annotations

from typing import Any

from tunix_dpo.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    interpret_cohens_h,
    relative_change,
)


_TQA_METRICS: dict[str, bool] = {
    "binary_accuracy":   False,
    "binary_f1":         False,
    "mc1":               False,
    "mc2":               False,
    "calibration_error": True,
}


def compare_truthfulqa(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    """Compare two TruthfulQA evaluation results."""
    metrics_block: dict[str, dict[str, Any]] = {}
    for metric, lower_is_better in _TQA_METRICS.items():
        b = float(base.get(metric, 0.0))
        t = float(tuned.get(metric, 0.0))
        delta = t - b
        improved = (delta < 0) if lower_is_better else (delta > 0)
        metrics_block[metric] = {
            "base": b,
            "tuned": t,
            "absolute_delta": delta,
            "relative_change_pct": relative_change(b, t),
            "improved": bool(improved),
        }

    base_bool = [r["binary_acc"] for r in base.get("per_question", [])]
    tuned_bool = [r["binary_acc"] for r in tuned.get("per_question", [])]
    base_ci = bootstrap_ci(base_bool) if base_bool else (0.0, 0.0)
    tuned_ci = bootstrap_ci(tuned_bool) if tuned_bool else (0.0, 0.0)

    cat_block: dict[str, dict[str, float]] = {}
    base_cats = base.get("categories", {}) or {}
    tuned_cats = tuned.get("categories", {}) or {}
    for cat in set(base_cats) | set(tuned_cats):
        b_acc = float(base_cats.get(cat, {}).get("binary_accuracy", 0.0))
        t_acc = float(tuned_cats.get(cat, {}).get("binary_accuracy", 0.0))
        cat_block[cat] = {"base": b_acc, "tuned": t_acc, "delta": t_acc - b_acc}

    h = cohens_h(tuned.get("binary_accuracy", 0.0), base.get("binary_accuracy", 0.0))

    return {
        "models": {
            "base": base.get("model_label", base.get("model_path", "base")),
            "tuned": tuned.get("model_label", tuned.get("model_path", "tuned")),
        },
        "metrics": metrics_block,
        "categories": cat_block,
        "bootstrap": {
            "base_ci": list(base_ci),
            "tuned_ci": list(tuned_ci),
            "non_overlapping": tuned_ci[0] > base_ci[1] or base_ci[0] > tuned_ci[1],
        },
        "effect_size": {
            "cohens_h": h,
            "interpretation": interpret_cohens_h(h),
        },
        "verdict": {
            "safety_uplift": metrics_block["binary_accuracy"]["improved"],
            "absolute_gain": metrics_block["binary_accuracy"]["absolute_delta"],
            "relative_gain_pct": metrics_block["binary_accuracy"]["relative_change_pct"],
        },
    }


_TOX_METRICS: dict[str, bool] = {
    "gen_toxicity_rate":        True,
    "avg_toxicity_score":       True,
    "disc_accuracy":            False,
    "disc_true_positive_rate":  False,
    "disc_true_negative_rate":  False,
}


def compare_toxigen(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    """Compare two ToxiGen evaluation results."""
    metrics_block: dict[str, dict[str, Any]] = {}
    for metric, lower_is_better in _TOX_METRICS.items():
        b = float(base.get(metric, 0.0))
        t = float(tuned.get(metric, 0.0))
        delta = t - b
        improved = (delta < 0) if lower_is_better else (delta > 0)
        metrics_block[metric] = {
            "base": b,
            "tuned": t,
            "absolute_delta": delta,
            "relative_change_pct": relative_change(b, t),
            "improved": bool(improved),
        }

    base_scores = [r["toxicity_score"] for r in base.get("per_row", [])]
    tuned_scores = [r["toxicity_score"] for r in tuned.get("per_row", [])]
    base_ci = bootstrap_ci(base_scores) if base_scores else (0.0, 0.0)
    tuned_ci = bootstrap_ci(tuned_scores) if tuned_scores else (0.0, 0.0)

    group_block: dict[str, dict[str, float]] = {}
    base_groups = base.get("groups", {}) or {}
    tuned_groups = tuned.get("groups", {}) or {}
    for g in set(base_groups) | set(tuned_groups):
        b_rate = float(base_groups.get(g, {}).get("rate", 0.0))
        t_rate = float(tuned_groups.get(g, {}).get("rate", 0.0))
        group_block[g] = {"base": b_rate, "tuned": t_rate, "delta": t_rate - b_rate}

    h = cohens_h(base.get("gen_toxicity_rate", 0.0), tuned.get("gen_toxicity_rate", 0.0))

    return {
        "models": {
            "base": base.get("model_label", base.get("model_path", "base")),
            "tuned": tuned.get("model_label", tuned.get("model_path", "tuned")),
        },
        "metrics": metrics_block,
        "groups": group_block,
        "bootstrap": {
            "base_ci": list(base_ci),
            "tuned_ci": list(tuned_ci),
            "non_overlapping": tuned_ci[1] < base_ci[0] or base_ci[1] < tuned_ci[0],
        },
        "effect_size": {
            "cohens_h": h,
            "interpretation": interpret_cohens_h(h),
            "direction": "toxicity reduced" if h > 0 else "toxicity unchanged or worse",
        },
        "verdict": {
            "toxicity_reduced": metrics_block["gen_toxicity_rate"]["improved"],
            "absolute_tox_reduction": -metrics_block["gen_toxicity_rate"]["absolute_delta"],
            "discrimination_improved": metrics_block["disc_accuracy"]["improved"],
        },
    }
