"""Unit tests for tunix_dpo.serving.schemas."""

import pytest
from pydantic import ValidationError

from tunix_dpo.serving.schemas import (
    BenchmarkRequest,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)


class TestChatMessage:
    def test_valid(self) -> None:
        m = ChatMessage(role="user", content="Hello")
        assert m.role == "user"

    def test_missing_role_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChatMessage(content="Hello")  # type: ignore[call-arg]


class TestChatCompletionRequest:
    def test_defaults(self) -> None:
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert req.temperature == pytest.approx(0.7)
        assert req.stream is False

    def test_temperature_clamped(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                temperature=5.0,  # > 2.0
            )

    def test_max_tokens_clamped(self) -> None:
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=0,  # ge=1
            )


class TestCompletionRequest:
    def test_valid(self) -> None:
        req = CompletionRequest(prompt="Hello")
        assert req.prompt == "Hello"
        assert req.max_tokens == 512


class TestBenchmarkRequest:
    def test_defaults(self) -> None:
        req = BenchmarkRequest()
        assert len(req.prompts) == 1
        assert req.concurrency == 4

    def test_concurrency_clamped(self) -> None:
        with pytest.raises(ValidationError):
            BenchmarkRequest(concurrency=0)  # ge=1
