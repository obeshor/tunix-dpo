"""OpenAI-compatible inference server — registered as ``tunix-serve``.

Routes:

- ``GET  /health``               liveness + telemetry snapshot
- ``GET  /metrics``              Prometheus exposition
- ``POST /v1/completions``       plain text completions
- ``POST /v1/chat/completions``  chat completions, with optional SSE streaming
                                 in true OpenAI delta format

The chat template is delegated to ``tokenizer.apply_chat_template`` so we get
Gemma 3's official template (including the required BOS prefix and proper
system-message handling) for free.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager

import click

from tunix_dpo.serving.engine import GenerationParams, VLLMEngine
from tunix_dpo.serving.metrics import Metrics
from tunix_dpo.serving.schemas import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    Usage,
)

logger = logging.getLogger(__name__)


# ── App factory ──────────────────────────────────────────────────────────────

def make_app(engine: VLLMEngine, metrics: Metrics, model_name: str, tokenizer=None):
    """Build the FastAPI app. ``tokenizer`` is used to render Gemma 3 chat
    prompts via its official chat template. If not provided, falls back to a
    minimal template that at least includes the required BOS marker."""
    from fastapi import FastAPI
    from fastapi.responses import PlainTextResponse, StreamingResponse

    @asynccontextmanager
    async def lifespan(app):
        logger.info("Server starting (model=%s)", model_name)
        yield
        logger.info("Server shutting down. Served %d requests.", metrics.requests_total)

    app = FastAPI(title="Tunix DPO Inference Server", lifespan=lifespan)

    # ── Health & metrics ─────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model=model_name,
            requests_total=metrics.requests_total,
            avg_latency_ms=metrics.avg_latency_ms,
            uptime_s=metrics.uptime_s,
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> str:
        return metrics.prometheus_text()

    # ── Plain completions ────────────────────────────────────────────────────

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        rid = metrics.next_id()
        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop=req.stop,
        )
        t0 = time.time()
        try:
            result = await engine.generate(req.prompt, params, rid)
        except Exception as e:  # noqa: BLE001
            metrics.record_error()
            logger.exception("Generation failed: %s", e)
            raise

        latency = (time.time() - t0) * 1000
        metrics.record(result.completion_tokens, latency)

        return CompletionResponse(
            id=rid,
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(index=0, text=result.text, finish_reason=result.finish_reason)
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    # ── Chat completions ─────────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        rid = metrics.next_id()
        prompt = _render_chat_prompt(req.messages, tokenizer)

        params = GenerationParams(
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            stop=req.stop,
        )

        if req.stream:
            return StreamingResponse(
                _openai_chat_stream(engine, prompt, params, rid, model_name, metrics),
                media_type="text/event-stream",
            )

        t0 = time.time()
        try:
            result = await engine.generate(prompt, params, rid)
        except Exception as e:  # noqa: BLE001
            metrics.record_error()
            logger.exception("Chat generation failed: %s", e)
            raise

        latency = (time.time() - t0) * 1000
        metrics.record(result.completion_tokens, latency)

        return ChatCompletionResponse(
            id=rid,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result.text),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    return app


# ── Streaming: produce real OpenAI SSE deltas ────────────────────────────────

async def _openai_chat_stream(engine, prompt: str, params, rid: str,
                              model_name: str, metrics: Metrics):
    """Yield OpenAI-compatible SSE events: each token wrapped in a JSON envelope."""
    t0 = time.time()
    tokens = 0
    created = int(time.time())

    # First chunk announces the assistant role (per OpenAI's protocol)
    first = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"

    try:
        async for delta in engine.stream(prompt, params, rid):
            tokens += 1
            chunk = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
    except Exception as e:  # noqa: BLE001
        metrics.record_error()
        logger.exception("Stream failed: %s", e)

    # Final stop chunk
    stop = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop)}\n\n"
    yield "data: [DONE]\n\n"

    metrics.record(tokens, (time.time() - t0) * 1000)


# ── Chat template ────────────────────────────────────────────────────────────

def _render_chat_prompt(messages: list[ChatMessage], tokenizer=None) -> str:
    """Render messages using Gemma 3's official chat template.

    Prefers ``tokenizer.apply_chat_template`` so we inherit the model's
    canonical template (BOS prefix, proper system-message handling, generation
    prompt). Falls back to a minimal hand-rolled version only if no tokenizer
    is available — and that fallback still emits the required <bos>.
    """
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        as_dicts = [{"role": m.role, "content": m.content} for m in messages]
        try:
            return tokenizer.apply_chat_template(
                as_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("apply_chat_template failed (%s); using fallback.", e)

    # Fallback: hand-rolled Gemma 3 template. Includes the BOS marker which is
    # critical for output quality. System messages are merged into the first
    # user turn following Gemma 3's official handling.
    parts: list[str] = ["<bos>"]
    system_buffer: list[str] = []
    user_buffer: list[str] = []

    def flush_user() -> None:
        if system_buffer or user_buffer:
            content = "\n\n".join([*system_buffer, *user_buffer]).strip()
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            system_buffer.clear()
            user_buffer.clear()

    for m in messages:
        if m.role == "system":
            flush_user()  # system before non-first user → ignore? In practice Gemma 3
                          # collapses systems into the next user turn; we buffer instead.
            system_buffer.append(m.content)
        elif m.role == "user":
            user_buffer.append(m.content)
            flush_user()
        elif m.role == "assistant":
            flush_user()
            parts.append(f"<start_of_turn>model\n{m.content}<end_of_turn>")

    flush_user()
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


# ── CLI entry point ──────────────────────────────────────────────────────────

@click.command()
@click.option("--model", required=True, help="Path or HF ID of the model to serve.")
@click.option("--port", type=int, default=8000)
@click.option("--host", default="0.0.0.0")
@click.option("--dtype", default="bfloat16")
@click.option("--max_model_len", type=int, default=4096)
@click.option("--gpu_memory_utilization", type=float, default=0.90)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--quantization", type=click.Choice(["gptq", "awq", "none"]), default="none")
def main(
    model: str, port: int, host: str, dtype: str,
    max_model_len: int, gpu_memory_utilization: float,
    tensor_parallel_size: int, quantization: str,
) -> None:
    """Start the OpenAI-compatible inference API server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load the tokenizer up-front so we can use its official chat template.
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        logger.info("Tokenizer loaded for chat templating: %s", model)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not load tokenizer (%s); using fallback chat template.", e)

    engine = VLLMEngine(
        model_path=model,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        quantization=None if quantization == "none" else quantization,
    )
    metrics = Metrics()
    app = make_app(engine, metrics, model, tokenizer=tokenizer)

    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
