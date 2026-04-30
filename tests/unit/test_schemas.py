"""Unit tests for tunix_dpo.serving.schemas."""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic")

from tunix_dpo.serving.schemas import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)


class TestChatMessage:
    def test_valid_user_message(self) -> None:
        m = ChatMessage(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(Exception):
            ChatMessage(role="invalid", content="x")


class TestChatCompletionRequest:
    def test_default_model(self) -> None:
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        assert "tunix-dpo" in req.model
        assert "gemma-3-1b-it" in req.model

    def test_temperature_bounds(self) -> None:
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="x")],
            temperature=1.5,
        )
        with pytest.raises(Exception):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="x")],
                temperature=2.5,
            )

    def test_max_tokens_validation(self) -> None:
        with pytest.raises(Exception):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="x")],
                max_tokens=0,
            )


class TestCompletionRequest:
    def test_required_prompt(self) -> None:
        req = CompletionRequest(prompt="Once upon a time")
        assert req.prompt == "Once upon a time"
        assert req.max_tokens == 256

    def test_streaming_flag(self) -> None:
        req = CompletionRequest(prompt="x", stream=True)
        assert req.stream is True
