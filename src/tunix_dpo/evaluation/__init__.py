"""Evaluation package — TruthfulQA, ToxiGen, statistical comparison."""

from tunix_dpo.evaluation.stats import bootstrap_ci, cohens_h, relative_change

__all__ = ["bootstrap_ci", "cohens_h", "relative_change"]