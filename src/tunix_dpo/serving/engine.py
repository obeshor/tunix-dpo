"""Async vLLM engine wrapper.

Falls back to a stub that echoes the prompt when vLLM is not installed
(e.g. on macOS), so the server can still start in development mode.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationParams:
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: list[str] | None = None


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"


class VLLMEngine:
    """Async wrapper around ``vllm.AsyncLLMEngine``."""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        tensor_parallel_size: int = 1,
        quantization: str | None = None,
    ):
        self.model_path = model_path
        self._engine = None
        self._mock = False

        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine
            args = AsyncEngineArgs(
                model=model_path,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                quantization=quantization,
            )
            self._engine = AsyncLLMEngine.from_engine_args(args)
            logger.info("vLLM engine ready: %s", model_path)
        except ImportError:
            logger.warning(
                "vLLM not installed; engine running in MOCK mode "
                "(prompts will be echoed back).",
            )
            self._mock = True

    async def generate(
        self,
        prompt: str,
        params: GenerationParams,
        request_id: str,
    ) -> GenerationResult:
        """Non-streaming generation."""
        if self._mock:
            return GenerationResult(
                text=f"[mock] {prompt[:80]}…",
                prompt_tokens=len(prompt.split()),
                completion_tokens=8,
                finish_reason="stop",
            )

        from vllm import SamplingParams
        sp = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop=params.stop,
        )
        results_gen = self._engine.generate(prompt, sp, request_id)

        final = None
        async for result in results_gen:
            final = result
        assert final is not None

        out = final.outputs[0]
        return GenerationResult(
            text=out.text,
            prompt_tokens=len(final.prompt_token_ids),
            completion_tokens=len(out.token_ids),
            finish_reason="stop" if out.finish_reason == "stop" else "length",
        )

    async def stream(
        self,
        prompt: str,
        params: GenerationParams,
        request_id: str,
    ) -> AsyncIterator[str]:
        """Streaming generation (SSE-friendly token deltas)."""
        if self._mock:
            for word in prompt.split()[:8]:
                yield word + " "
            return

        from vllm import SamplingParams
        sp = SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop=params.stop,
        )

        last_text = ""
        async for result in self._engine.generate(prompt, sp, request_id):
            new_text = result.outputs[0].text
            delta = new_text[len(last_text):]
            if delta:
                yield delta
            last_text = new_text
