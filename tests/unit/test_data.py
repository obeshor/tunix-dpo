"""Unit tests for tunix_dpo.data.parser and tunix_dpo.data.formatter."""

import pytest

from tunix_dpo.data.formatter import format_dpo, format_rm
from tunix_dpo.data.parser import is_valid_pair, parse_dialogue


# ── parse_dialogue ────────────────────────────────────────────────────────────

class TestParseDialogue:
    def test_single_turn(self) -> None:
        raw = "\n\nHuman: Hello!\n\nAssistant: Hi there."
        prompt, response = parse_dialogue(raw)
        assert "Human: Hello!" in prompt
        assert "Assistant:" in prompt
        assert response == "Hi there."

    def test_multi_turn(self) -> None:
        raw = (
            "\n\nHuman: Turn 1"
            "\n\nAssistant: Reply 1"
            "\n\nHuman: Turn 2"
            "\n\nAssistant: Final reply"
        )
        prompt, response = parse_dialogue(raw)
        assert "Turn 1" in prompt
        assert "Reply 1" in prompt
        assert "Turn 2" in prompt
        assert response == "Final reply"
        # The prompt must end with "Assistant:" so both chosen/rejected share it
        assert prompt.endswith("Assistant:")

    def test_prompt_identical_for_chosen_and_rejected(self) -> None:
        """Verifies the key DPO invariant: prompts must match."""
        base = (
            "\n\nHuman: Question"
            "\n\nAssistant: "
        )
        raw_chosen   = base + "Good answer"
        raw_rejected = base + "Bad answer"

        prompt_c, _ = parse_dialogue(raw_chosen)
        prompt_r, _ = parse_dialogue(raw_rejected)
        assert prompt_c == prompt_r

    def test_malformed_returns_fallback(self) -> None:
        raw = "No turn markers here"
        prompt, response = parse_dialogue(raw)
        assert response == ""


# ── is_valid_pair ─────────────────────────────────────────────────────────────

class TestIsValidPair:
    def test_valid_pair(self) -> None:
        assert is_valid_pair("A good response.", "A bad response.") is True

    def test_identical_pair_rejected(self) -> None:
        assert is_valid_pair("Same text", "Same text") is False

    def test_empty_chosen_rejected(self) -> None:
        assert is_valid_pair("", "Something") is False
        assert is_valid_pair("Something", "") is False

    def test_stub_rejected(self) -> None:
        assert is_valid_pair("Ok", "Fine") is False  # both < 10 chars


# ── format_dpo ────────────────────────────────────────────────────────────────

class TestFormatDpo:
    def test_required_keys_present(self) -> None:
        rec = format_dpo("prompt", "chosen", "rejected")
        assert {"prompt", "chosen", "rejected"} <= rec.keys()

    def test_metadata_optional(self) -> None:
        rec = format_dpo("p", "c", "r")
        assert "metadata" not in rec

    def test_metadata_included_when_provided(self) -> None:
        rec = format_dpo("p", "c", "r", metadata={"split": "train"})
        assert rec["metadata"] == {"split": "train"}


# ── format_rm ─────────────────────────────────────────────────────────────────

class TestFormatRm:
    def test_label_1(self) -> None:
        rec = format_rm("p", "response", label=1)
        assert rec["label"] == 1

    def test_label_0(self) -> None:
        rec = format_rm("p", "response", label=0)
        assert rec["label"] == 0

    def test_invalid_label_raises(self) -> None:
        with pytest.raises(ValueError):
            format_rm("p", "r", label=2)
