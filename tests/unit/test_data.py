"""Unit tests for tunix_dpo.data.parser and tunix_dpo.data.formatter."""

from __future__ import annotations

import pytest

from tunix_dpo.data.formatter import format_dpo, format_rm
from tunix_dpo.data.parser import is_valid_pair, parse_dialogue


class TestParseDialogue:
    def test_single_turn(self) -> None:
        raw = "\n\nHuman: Hello\n\nAssistant: Hi there"
        prompt, response = parse_dialogue(raw)
        assert response == "Hi there"
        assert prompt.endswith("Assistant:")
        assert "Hello" in prompt

    def test_multi_turn(self) -> None:
        raw = (
            "\n\nHuman: Q1\n\nAssistant: A1"
            "\n\nHuman: Q2\n\nAssistant: Final answer"
        )
        prompt, response = parse_dialogue(raw)
        assert response == "Final answer"
        assert prompt.endswith("Assistant:")
        for tok in ("Q1", "A1", "Q2"):
            assert tok in prompt

    def test_dpo_invariant_prompts_match(self) -> None:
        """The DPO invariant: prompt is identical for chosen/rejected."""
        base = "\n\nHuman: Is this safe?\n\nAssistant: "
        pc, _ = parse_dialogue(base + "Yes, completely safe here.")
        pr, _ = parse_dialogue(base + "No, dangerous.")
        assert pc == pr

    def test_empty_input(self) -> None:
        prompt, response = parse_dialogue("")
        assert response == ""
        assert prompt == ""


class TestIsValidPair:
    def test_distinct_long_pair(self) -> None:
        assert is_valid_pair(
            "A long enough chosen response.",
            "A different rejected response.",
        )

    def test_identical_rejected(self) -> None:
        assert not is_valid_pair("same response", "same response")

    def test_too_short(self) -> None:
        assert not is_valid_pair("ok", "yes")

    def test_empty_strings(self) -> None:
        assert not is_valid_pair("", "non-empty")
        assert not is_valid_pair("non-empty", "")


class TestFormatters:
    def test_format_dpo_basic(self) -> None:
        rec = format_dpo("p", "c", "r", {"split": "train"})
        assert rec["prompt"] == "p"
        assert rec["chosen"] == "c"
        assert rec["rejected"] == "r"
        assert rec["metadata"] == {"split": "train"}

    def test_format_dpo_no_metadata(self) -> None:
        rec = format_dpo("p", "c", "r")
        assert rec["metadata"] == {}

    def test_format_rm_chosen(self) -> None:
        rec = format_rm("p", "good", label=1)
        assert rec["label"] == 1

    def test_format_rm_rejected(self) -> None:
        rec = format_rm("p", "bad", label=0)
        assert rec["label"] == 0

    def test_format_rm_invalid_label_raises(self) -> None:
        with pytest.raises(ValueError):
            format_rm("p", "r", label=2)
