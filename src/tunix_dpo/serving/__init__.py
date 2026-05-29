"""Serving package — weight export and vLLM inference server."""

try:
    from tunix_dpo.serving.schemas import (
        ChatCompletionRequest,
        ChatMessage,
        CompletionRequest,
    )

    __all__ = ["ChatCompletionRequest", "ChatMessage", "CompletionRequest"]
except ImportError:
    __all__ = []
