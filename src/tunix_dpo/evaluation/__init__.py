"""Evaluation package — TruthfulQA + ToxiGen benchmarking."""

from tunix_dpo.evaluation.compare import compare_toxigen, compare_truthfulqa
from tunix_dpo.evaluation.stats import (
    bootstrap_ci,
    cohens_h,
    interpret_cohens_h,
    relative_change,
)

__all__ = [
    "bootstrap_ci",
    "cohens_h",
    "compare_toxigen",
    "compare_truthfulqa",
    "interpret_cohens_h",
    "relative_change",
]
