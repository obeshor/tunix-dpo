"""
Integration test: data curation pipeline end-to-end.

Runs ``process_subset`` against a tiny mock dataset (no network required)
and verifies the output JSONL files are well-formed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tunix_dpo.data.curate import process_subset, write_jsonl
from tunix_dpo.data.parser import is_valid_pair, parse_dialogue

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_raw_row(chosen_resp: str, rejected_resp: str) -> dict:
    base = "\n\nHuman: Is this safe?\n\nAssistant: "
    return {
        "chosen": base + chosen_resp,
        "rejected": base + rejected_resp,
    }


MOCK_ROWS = [
    _make_raw_row("Yes, this is completely safe.", "Not sure, maybe risky."),
    _make_raw_row("No harmful effects have been documented.", "It depends on many factors."),
    _make_raw_row("Identical response.", "Identical response."),  # should be filtered
    _make_raw_row("Ok", "Sure"),  # too short, filtered
]


# ── Tests ──────────────────────────────────────────────────────────────────


class TestWriteJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        path = tmp_path / "out.jsonl"
        write_jsonl(records, path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert loaded == records

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "file.jsonl"
        write_jsonl([{"x": 1}], path)
        assert path.exists()


class TestProcessSubset:
    @patch("tunix_dpo.data.curate.load_dataset")
    def test_dpo_format(self, mock_load: MagicMock) -> None:
        mock_load.return_value = MOCK_ROWS

        result = process_subset("harmless-base", "train", "dpo")

        assert "dpo" in result
        records = result["dpo"]
        # 4 rows, 2 filtered → 2 valid
        assert len(records) == 2
        for rec in records:
            assert "prompt" in rec
            assert "chosen" in rec
            assert "rejected" in rec
            assert rec["chosen"] != rec["rejected"]
            # Prompt must end with "Assistant:" (DPO invariant)
            assert rec["prompt"].endswith("Assistant:")

    @patch("tunix_dpo.data.curate.load_dataset")
    def test_rm_format_doubles_rows(self, mock_load: MagicMock) -> None:
        mock_load.return_value = MOCK_ROWS

        result = process_subset("helpful-base", "train", "rm")

        assert "rm" in result
        # 2 valid pairs → 4 RM rows (chosen label=1 + rejected label=0 each)
        assert len(result["rm"]) == 4
        labels = [r["label"] for r in result["rm"]]
        assert labels.count(1) == 2
        assert labels.count(0) == 2

    @patch("tunix_dpo.data.curate.load_dataset")
    def test_both_format(self, mock_load: MagicMock) -> None:
        mock_load.return_value = MOCK_ROWS

        result = process_subset("helpful-base", "train", "both")

        assert "dpo" in result and "rm" in result
        assert len(result["dpo"]) == 2
        assert len(result["rm"]) == 4

    @patch("tunix_dpo.data.curate.load_dataset")
    def test_metadata_contains_subset_and_split(self, mock_load: MagicMock) -> None:
        mock_load.return_value = [MOCK_ROWS[0]]

        result = process_subset("harmless-base", "test", "dpo")
        meta = result["dpo"][0]["metadata"]
        assert meta["subset"] == "harmless-base"
        assert meta["split"] == "test"


class TestParserIntegration:
    def test_prompt_is_shared_across_chosen_rejected(self) -> None:
        """Core DPO invariant: identical prompt for both completions."""
        row = MOCK_ROWS[0]
        prompt_c, _ = parse_dialogue(row["chosen"])
        prompt_r, _ = parse_dialogue(row["rejected"])
        assert prompt_c == prompt_r

    def test_valid_pair_filters_identical(self) -> None:
        assert is_valid_pair("Identical.", "Identical.") is False

    def test_valid_pair_accepts_distinct(self) -> None:
        assert (
            is_valid_pair("A long enough chosen response here.", "A different rejected response.")
            is True
        )
