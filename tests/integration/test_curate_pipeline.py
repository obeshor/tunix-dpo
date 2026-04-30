"""Integration test for the data curation pipeline (mocks HF)."""

from __future__ import annotations

import json
from unittest.mock import patch

from tunix_dpo.data.curate import process_subset, write_jsonl


@patch("tunix_dpo.data.curate.load_dataset")
def test_process_subset_filters_correctly(mock_load):
    """End-to-end: HF rows → parsed → filtered → DPO records."""
    mock_load.return_value = [
        {
            "chosen":   "\n\nHuman: Q1\n\nAssistant: Good answer here.",
            "rejected": "\n\nHuman: Q1\n\nAssistant: Bad answer here.",
        },
        {
            "chosen":   "\n\nHuman: Q2\n\nAssistant: same response",
            "rejected": "\n\nHuman: Q2\n\nAssistant: same response",
        },
        {
            "chosen":   "\n\nHuman: Q3\n\nAssistant: Long enough answer.",
            "rejected": "\n\nHuman: Q3\n\nAssistant: ok",  # filtered (too short)
        },
    ]
    result = process_subset("helpful-base", "train", "dpo")
    assert "dpo" in result
    assert len(result["dpo"]) == 1  # only the first record passes filters
    assert result["dpo"][0]["chosen"] == "Good answer here."


def test_write_jsonl_roundtrip(tmp_path):
    records = [{"prompt": "p", "chosen": "c", "rejected": "r", "metadata": {}}]
    path = tmp_path / "out.jsonl"
    write_jsonl(records, path)
    with open(path) as f:
        loaded = [json.loads(l) for l in f]
    assert loaded == records
