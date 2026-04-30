"""
Shared pytest fixtures.

Fixtures defined here are automatically available to every test module
in the ``tests/`` tree without explicit imports.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Temporary directories ─────────────────────────────────────────────────────


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """An empty temporary directory for data files."""
    d = tmp_path / "data"
    d.mkdir()
    return d


# ── Minimal DPO record factories ─────────────────────────────────────────────


@pytest.fixture
def sample_dpo_record() -> dict:
    return {
        "prompt": "\n\nHuman: Do vaccines cause autism?\n\nAssistant:",
        "chosen": "No. This claim originates from a retracted 1998 study.",
        "rejected": "Some people believe they might.",
        "metadata": {"subset": "harmless-base", "split": "train"},
    }


@pytest.fixture
def sample_rm_record() -> dict:
    return {
        "prompt": "\n\nHuman: Is the sky blue?\n\nAssistant:",
        "response": "Yes, due to Rayleigh scattering.",
        "label": 1,
        "metadata": {"subset": "helpful-base", "split": "train"},
    }


@pytest.fixture
def dpo_jsonl_file(tmp_data_dir: Path, sample_dpo_record: dict) -> Path:
    """A JSONL file containing five identical DPO records."""
    path = tmp_data_dir / "train.jsonl"
    with open(path, "w") as f:
        for _ in range(5):
            f.write(json.dumps(sample_dpo_record) + "\n")
    return path


# ── Minimal TruthfulQA result dicts ──────────────────────────────────────────


def _make_tqa_result(label: str, binary_acc: float, n: int = 10) -> dict:
    return {
        "model_label": label,
        "model_path": f"./model/{label}",
        "n_questions": n,
        "mc1": binary_acc - 0.20,
        "mc2": binary_acc - 0.13,
        "binary_accuracy": binary_acc,
        "binary_f1": binary_acc - 0.01,
        "calibration_error": 1.0 - binary_acc,
        "categories": {},
        "per_question": [{"binary_acc": binary_acc} for _ in range(n)],
    }


@pytest.fixture
def tqa_base_result() -> dict:
    return _make_tqa_result("base", 0.512)


@pytest.fixture
def tqa_tuned_result() -> dict:
    return _make_tqa_result("tunix_dpo", 0.614)


# ── Minimal ToxiGen result dicts ──────────────────────────────────────────────


def _make_tox_result(label: str, tox_rate: float, disc_acc: float, n: int = 20) -> dict:
    return {
        "model_label": label,
        "model_path": f"./model/{label}",
        "n_rows": n,
        "gen_toxicity_rate": tox_rate,
        "avg_toxicity_score": tox_rate - 0.02,
        "disc_accuracy": disc_acc,
        "disc_true_positive_rate": disc_acc - 0.05,
        "disc_true_negative_rate": disc_acc + 0.03,
        "groups": {},
        "per_row": [{"toxicity_score": tox_rate} for _ in range(n)],
    }


@pytest.fixture
def tox_base_result() -> dict:
    return _make_tox_result("base", tox_rate=0.412, disc_acc=0.631)


@pytest.fixture
def tox_tuned_result() -> dict:
    return _make_tox_result("tunix_dpo", tox_rate=0.218, disc_acc=0.748)
