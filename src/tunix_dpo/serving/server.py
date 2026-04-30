"""
FastAPI inference server — CLI entry-point.

All domain logic lives in the sibling modules:
    schemas.py  — Pydantic request/response models
    engine.py   — VLLMEngine (async generator)
    metrics.py  — Metrics telemetry

This module only wires them together into routes and starts uvicorn.

Usage
-----
    python -m tunix_dpo.serving.server --model ./exports/tunix_dpo_gemma2b
    tunix-serve --model ./exports/tunix_dpo_gemma2b --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from tunix_dpo.serving.engine import VLLMEngine
from tunix_dpo.serving.metrics import Metrics
from tunix_dpo.serving.schemas import (
    BenchmarkRequest,
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
)

log = logging.getLogger(__name__)


# ── Prompt formatter (Gemma-2 chat template) ─────────────────────────────────


def _gemma_prompt(messages: list[ChatMessage]) -> str:
    parts: list[str] = []
    prefix = ""
    for msg in messages:
        if msg.role == "system":
            prefix = msg.content.strip() + "\n\n"
        elif msg.role == "user":
            parts.append(f"<start_of_turn>user\n{prefix + msg.content}<end_of_turn>")
            prefix = ""
        elif msg.role == "assistant":
            parts.append(f"<start_of_turn>model\n{msg.content}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


# ── App factory ───────────────────────────────────────────────────────────────


def build_app(args: argparse.Namespace) -> FastAPI:
    _metrics: Metrics = Metrics()
    _engine: VLLMEngine | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[misc]
        nonlocal _engine
        _engine = VLLMEngine(
            model_path=args.model,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            quantization=args.quantization,
            max_num_seqs=args.max_num_seqs,
            metrics=_metrics,
        )
        yield
        log.info("Server shutting down.")

    app = FastAPI(
        title="Tunix-DPO Inference API",
        version="1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    def _guard() -> VLLMEngine:
        if _engine is None or not _engine.ready:
            raise HTTPException(503, "Engine not ready")
        return _engine

    @app.get("/health")
    async def health():
        ready = _engine is not None and _engine.ready
        return {
            "status": "ok" if ready else "starting",
            "model": args.model,
            "dtype": args.dtype,
            "metrics": {
                "requests": _metrics.requests_total,
                "tokens": _metrics.tokens_generated,
                "avg_lat_ms": round(_metrics.avg_latency_ms, 1),
                "tok_per_sec": round(_metrics.avg_tokens_per_sec, 1),
            },
        }

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(_metrics.prometheus_text(), media_type="text/plain")

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": "tunix-dpo", "object": "model"}]}

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        engine = _guard()
        rid = _metrics.next_id()
        parts = []
        try:
            async for delta, _ in engine.generate(
                req.prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                50,
                req.repetition_penalty,
                req.stop,
                None,
                rid,
            ):
                parts.append(delta)
        except Exception as exc:
            _metrics.record_error()
            raise HTTPException(500, str(exc)) from exc
        text = "".join(parts)
        return {
            "id": rid,
            "object": "text_completion",
            "model": req.model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }

    @app.post("/v1/chat/completions")
    async def chat(req: ChatCompletionRequest):
        engine = _guard()
        prompt = _gemma_prompt(req.messages)
        rid = _metrics.next_id()

        if req.stream:
            return StreamingResponse(
                _stream(engine, req, prompt, rid), media_type="text/event-stream"
            )

        parts = []
        try:
            async for delta, _ in engine.generate(
                prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.top_k,
                req.repetition_penalty,
                req.stop,
                req.seed,
                rid,
            ):
                parts.append(delta)
        except Exception as exc:
            _metrics.record_error()
            raise HTTPException(500, str(exc)) from exc

        text = "".join(parts)
        return {
            "id": rid,
            "object": "chat.completion",
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    async def _stream(engine: VLLMEngine, req: ChatCompletionRequest, prompt: str, rid: str):
        try:
            async for delta, is_final in engine.generate(
                prompt,
                req.max_tokens,
                req.temperature,
                req.top_p,
                req.top_k,
                req.repetition_penalty,
                req.stop,
                req.seed,
                rid,
            ):
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": "stop" if is_final else None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as exc:
            _metrics.record_error()
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/benchmark")
    async def benchmark(req: BenchmarkRequest):
        engine = _guard()

        async def _one(prompt: str) -> dict:
            rid = _metrics.next_id()
            t0 = time.perf_counter()
            toks: list[str] = []
            async for delta, _ in engine.generate(
                prompt, req.max_tokens, 0.0, 1.0, 1, 1.0, None, None, rid
            ):
                toks.append(delta)
            lat = (time.perf_counter() - t0) * 1000
            n = len("".join(toks).split())
            return {
                "prompt": prompt[:50],
                "latency_ms": round(lat, 1),
                "tokens": n,
                "tok_per_s": round(n / max(lat / 1000, 1e-9), 1),
            }

        prompts = (req.prompts * ((req.concurrency // max(len(req.prompts), 1)) + 1))[
            : req.concurrency
        ]
        results = await asyncio.gather(*[_one(p) for p in prompts])
        lats = sorted(r["latency_ms"] for r in results)
        tps = [r["tok_per_s"] for r in results]
        return {
            "results": results,
            "summary": {
                "n": len(results),
                "avg_ms": round(sum(lats) / len(lats), 1),
                "p50_ms": lats[len(lats) // 2],
                "p95_ms": lats[int(len(lats) * 0.95)],
                "avg_tok_s": round(sum(tps) / len(tps), 1),
            },
        }

    return app


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Tunix-DPO vLLM inference server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"]
    )
    parser.add_argument("--max_model_len", default=4096, type=int)
    parser.add_argument("--gpu_memory_utilization", default=0.90, type=float)
    parser.add_argument("--tensor_parallel_size", default=1, type=int)
    parser.add_argument(
        "--quantization", default=None, choices=[None, "gptq", "awq", "squeezellm", "bitsandbytes"]
    )
    parser.add_argument("--max_num_seqs", default=256, type=int)
    args = parser.parse_args(argv)

    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
