"""OpenAI-compatible inference server — registered as ``tunix-serve``.

Routes:
- GET  /health                  liveness + telemetry
- GET  /metrics                 Prometheus exposition
- POST /v1/completions          plain text completions
- POST /v1/chat/completions     chat completions (with optional SSE streaming)
"""

from __future__ import annotations

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


def make_app(engine: VLLMEngine, metrics: Metrics, model_name: str):
    """Wire up the FastAPI app."""
    from fastapi import FastAPI, Request
    from fastapi.responses import PlainTextResponse, StreamingResponse

    @asynccontextmanager
    async def lifespan(app):
        logger.info("Server starting (model=%s)", model_name)
        yield
        logger.info("Server shutting down. Served %d requests.", metrics.requests_total)

    app = FastAPI(title="Tunix DPO Inference Server", lifespan=lifespan)

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

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(req: CompletionRequest) -> CompletionResponse:
        rid = metrics.next_id()
        params = GenerationParams(
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, stop=req.stop,
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
            choices=[CompletionChoice(index=0, text=result.text, finish_reason=result.finish_reason)],
            usage=Usage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.prompt_tokens + result.completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        rid = metrics.next_id()
        prompt = _render_chat_prompt(req.messages)
        params = GenerationParams(
            max_tokens=req.max_tokens, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, stop=req.stop,
        )

        if req.stream:
            async def event_stream():
                t0 = time.time()
                tokens = 0
                async for delta in engine.stream(prompt, params, rid):
                    tokens += 1
                    yield f"data: {delta}\n\n"
                yield "data: [DONE]\n\n"
                metrics.record(tokens, (time.time() - t0) * 1000)

            return StreamingResponse(event_stream(), media_type="text/event-stream")

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


def _render_chat_prompt(messages: list[ChatMessage]) -> str:
    """Render messages with the Gemma 3 chat template."""
    parts: list[str] = []
    for m in messages:
        role = "user" if m.role in {"user", "system"} else "model"
        parts.append(f"<start_of_turn>{role}\n{m.content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


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

    engine = VLLMEngine(
        model_path=model,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        quantization=None if quantization == "none" else quantization,
    )
    metrics = Metrics()
    app = make_app(engine, metrics, model)

    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
