"""Shared pytest fixtures for tunix-dpo tests.

All fixtures are pure (no GPU, no network) so they work in any CI runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def sample_dpo_record() -> dict:
    return {
        "prompt":   "\n\nHuman: Do vaccines cause autism?\n\nAssistant:",
        "chosen":   "No. This claim originates from a retracted 1998 study.",
        "rejected": "Some people think they might.",
        "metadata": {"subset": "harmless-base", "split": "train"},
    }


@pytest.fixture
def dpo_jsonl_file(tmp_data_dir: Path, sample_dpo_record: dict) -> Path:
    path = tmp_data_dir / "train.jsonl"
    with open(path, "w") as f:
        for _ in range(5):
            f.write(json.dumps(sample_dpo_record) + "\n")
    return path


def _make_tqa(label: str, acc: float, n: int = 50) -> dict:
    return {
        "model_label":       label,
        "model_path":        f"./model/{label}",
        "n_questions":       n,
        "mc1":               max(0.0, acc - 0.20),
        "mc2":               max(0.0, acc - 0.13),
        "binary_accuracy":   acc,
        "binary_f1":         max(0.0, acc - 0.01),
        "calibration_error": 1.0 - acc,
        "categories":        {},
        "per_question":      [{"binary_acc": acc} for _ in range(n)],
    }


@pytest.fixture
def tqa_base_result() -> dict:
    return _make_tqa("base", 0.512)


@pytest.fixture
def tqa_tuned_result() -> dict:
    return _make_tqa("tunix_dpo", 0.614)


def _make_tox(label: str, tox_rate: float, disc_acc: float, n: int = 40) -> dict:
    return {
        "model_label":             label,
        "model_path":              f"./model/{label}",
        "n_rows":                  n,
        "gen_toxicity_rate":       tox_rate,
        "avg_toxicity_score":      max(0.0, tox_rate - 0.02),
        "disc_accuracy":           disc_acc,
        "disc_true_positive_rate": max(0.0, disc_acc - 0.05),
        "disc_true_negative_rate": min(1.0, disc_acc + 0.03),
        "groups":                  {},
        "per_row":                 [{"toxicity_score": tox_rate} for _ in range(n)],
    }


@pytest.fixture
def tox_base_result() -> dict:
    return _make_tox("base", 0.412, 0.631)


@pytest.fixture
def tox_tuned_result() -> dict:
    return _make_tox("tunix_dpo", 0.218, 0.748)
