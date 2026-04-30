"""
Pure comparison functions for TruthfulQA and ToxiGen result dicts.

No file I/O.  All functions take the JSON-serialisable dicts produced
by ``runner.py`` and return JSON-serialisable comparison dicts.
"""

from __future__ import annotations

import numpy as np
from tunix_dpo.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    interpret_cohens_h,
    relative_change,
)


def _delta(base: float, tuned: float, lower_better: bool = False) -> dict:
    delta = tuned - base
    improved = (-delta if lower_better else delta) > 0
    return {
        "base": round(base, 4),
        "tuned": round(tuned, 4),
        "absolute_delta": round(delta, 4),
        "relative_pct": round(relative_change(base, tuned), 2),
        "improved": improved,
    }


# ── TruthfulQA ────────────────────────────────────────────────────────────────


def compare_truthfulqa(base: dict, tuned: dict) -> dict:
    """Compare two TruthfulQA result dicts.

    Returns a comparison dict with:
        metrics     — per-metric deltas
        categories  — per-category binary accuracy deltas
        bootstrap   — 95% CI on binary accuracy
        effect_size — Cohen's h
        verdict     — top-level summary flags
    """
    comp: dict = {
        "models": {"base": base["model_label"], "tuned": tuned["model_label"]},
        "n_questions": base["n_questions"],
        "metrics": {},
        "categories": {},
        "bootstrap": {},
        "effect_size": {},
        "verdict": {},
    }

    metric_specs = [
        ("mc1", False),
        ("mc2", False),
        ("binary_accuracy", False),
        ("binary_f1", False),
        ("calibration_error", True),
    ]
    for key, lower_better in metric_specs:
        comp["metrics"][key] = _delta(base[key], tuned[key], lower_better)

    base_bin = [r["binary_acc"] for r in base["per_question"]]
    tuned_bin = [r["binary_acc"] for r in tuned["per_question"]]
    bci, tci = bootstrap_ci(base_bin), bootstrap_ci(tuned_bin)
    comp["bootstrap"] = {
        "metric": "binary_accuracy",
        "base": {
            "mean": round(float(np.mean(base_bin)), 4),
            "ci_95": list(map(lambda x: round(x, 4), bci)),
        },
        "tuned": {
            "mean": round(float(np.mean(tuned_bin)), 4),
            "ci_95": list(map(lambda x: round(x, 4), tci)),
        },
        "non_overlapping": bool(tci[0] > bci[1]),
    }

    h = cohens_h(tuned["binary_accuracy"], base["binary_accuracy"])
    comp["effect_size"] = {
        "cohens_h": round(h, 4),
        "interpretation": interpret_cohens_h(h),
    }

    for cat in sorted(set(list(base.get("categories", {})) + list(tuned.get("categories", {})))):
        bc = base["categories"].get(cat, {})
        tc = tuned["categories"].get(cat, {})
        if bc and tc:
            comp["categories"][cat] = {
                "n": bc["n"],
                "base_binary_acc": round(bc.get("binary_acc", 0.0), 4),
                "tuned_binary_acc": round(tc.get("binary_acc", 0.0), 4),
                "delta": round(tc.get("binary_acc", 0.0) - bc.get("binary_acc", 0.0), 4),
            }

    ba = comp["metrics"]["binary_accuracy"]
    calib = comp["metrics"]["calibration_error"]
    comp["verdict"] = {
        "safety_uplift": ba["improved"],
        "absolute_gain": ba["absolute_delta"],
        "relative_gain_pct": ba["relative_pct"],
        "calibration_improved": calib["improved"],
        "statistically_significant": comp["bootstrap"]["non_overlapping"],
        "effect_size": comp["effect_size"]["interpretation"],
    }
    return comp


# ── ToxiGen ───────────────────────────────────────────────────────────────────


def compare_toxigen(base: dict, tuned: dict) -> dict:
    """Compare two ToxiGen result dicts."""
    comp: dict = {
        "models": {"base": base["model_label"], "tuned": tuned["model_label"]},
        "n_rows": base["n_rows"],
        "metrics": {},
        "groups": {},
        "bootstrap": {},
        "effect_size": {},
        "verdict": {},
    }

    metric_specs = [
        ("gen_toxicity_rate", True),
        ("avg_toxicity_score", True),
        ("disc_accuracy", False),
        ("disc_true_positive_rate", False),
        ("disc_true_negative_rate", False),
    ]
    for key, lower_better in metric_specs:
        comp["metrics"][key] = _delta(base[key], tuned[key], lower_better)

    base_scores = [r["toxicity_score"] for r in base["per_row"]]
    tuned_scores = [r["toxicity_score"] for r in tuned["per_row"]]
    bci, tci = bootstrap_ci(base_scores), bootstrap_ci(tuned_scores)
    comp["bootstrap"] = {
        "metric": "avg_toxicity_score",
        "base": {
            "mean": round(float(np.mean(base_scores)), 4),
            "ci_95": list(map(lambda x: round(x, 4), bci)),
        },
        "tuned": {
            "mean": round(float(np.mean(tuned_scores)), 4),
            "ci_95": list(map(lambda x: round(x, 4), tci)),
        },
        "non_overlapping": bool(tci[1] < bci[0]),
    }

    h = cohens_h(base["gen_toxicity_rate"], tuned["gen_toxicity_rate"])
    comp["effect_size"] = {
        "cohens_h": round(abs(h), 4),
        "interpretation": interpret_cohens_h(h),
        "direction": "toxicity reduced" if h > 0 else "toxicity increased (unexpected)",
    }

    for grp in sorted(set(list(base.get("groups", {})) + list(tuned.get("groups", {})))):
        bg = base["groups"].get(grp, {})
        tg = tuned["groups"].get(grp, {})
        if bg and tg:
            comp["groups"][grp] = {
                "n": bg["n"],
                "base_tox_rate": round(bg.get("gen_tox_rate", 0.0), 4),
                "tuned_tox_rate": round(tg.get("gen_tox_rate", 0.0), 4),
                "tox_delta": round(tg.get("gen_tox_rate", 0.0) - bg.get("gen_tox_rate", 0.0), 4),
                "base_disc_acc": round(bg.get("disc_accuracy", 0.0), 4),
                "tuned_disc_acc": round(tg.get("disc_accuracy", 0.0), 4),
                "disc_delta": round(tg.get("disc_accuracy", 0.0) - bg.get("disc_accuracy", 0.0), 4),
            }

    tox = comp["metrics"]["gen_toxicity_rate"]
    disc = comp["metrics"]["disc_accuracy"]
    comp["verdict"] = {
        "toxicity_reduced": tox["improved"],
        "absolute_tox_reduction": -tox["absolute_delta"],
        "relative_tox_reduction_pct": -tox["relative_pct"],
        "discrimination_improved": disc["improved"],
        "statistically_significant": comp["bootstrap"]["non_overlapping"],
        "effect_size": comp["effect_size"]["interpretation"],
    }
    return comp
