"""
OpenAI-compatible Pydantic request/response schemas.

Kept in their own module so the server, tests, and any future
client library can all import schemas without pulling in FastAPI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "tunix-dpo"
    messages: list[ChatMessage]
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = False
    stop: list[str] | None = None
    seed: int | None = None


class CompletionRequest(BaseModel):
    model: str = "tunix-dpo"
    prompt: str
    max_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    stream: bool = False
    stop: list[str] | None = None


class BenchmarkRequest(BaseModel):
    prompts: list[str] = Field(default_factory=lambda: ["Explain why the sky is blue."])
    max_tokens: int = Field(128, ge=1)
    concurrency: int = Field(4, ge=1, le=64)
