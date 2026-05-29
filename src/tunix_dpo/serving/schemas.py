"""OpenAI-compatible request/response schemas (pure Pydantic, no FastAPI)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "tunix-dpo-gemma-3-1b-it"
    messages: list[ChatMessage]
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=200)
    stream: bool = False
    stop: list[str] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage


class CompletionRequest(BaseModel):
    model: str = "tunix-dpo-gemma-3-1b-it"
    prompt: str
    max_tokens: int = Field(256, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=200)
    stream: bool = False
    stop: list[str] | None = None


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: Literal["stop", "length"] = "stop"


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    requests_total: int
    avg_latency_ms: float
    uptime_s: float
