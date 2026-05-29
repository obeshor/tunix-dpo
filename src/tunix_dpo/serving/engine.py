"""Async vLLM engine wrapper.

Hides every vLLM-specific call behind a clean async interface. The FastAPI
routes in ``server.py`` import only this class — they never see vLLM types.

VLLM VERSION COMPATIBILITY
--------------------------
vLLM's AsyncLLMEngine signature changed between 0.5.x and 0.7.x:
- 0.5.x: ``AsyncLLMEngine.from_engine_args(args)``
- 0.7.x: ``AsyncLLMEngine.from_engine_args(args)`` (same factory, but
  ``AsyncEngineArgs`` gained/renamed fields)

We feature-detect at construction time and pass only the kwargs the current
version supports. If vLLM is not installed at all, the engine runs in MOCK
mode so the server can boot for development.
"""

from __future__ import annotations

import inspect
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
    """Async wrapper around ``vllm.AsyncLLMEngine`` with version compatibility."""

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
        except ImportError:
            logger.warning(
                "vLLM not installed; engine running in MOCK mode "
                "(prompts will be echoed back).",
            )
            self._mock = True
            return

        # Build the kwargs dict and filter to what AsyncEngineArgs actually accepts
        # — fields have come and gone across vLLM minor versions.
        requested = {
            "model": model_path,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": tensor_parallel_size,
            "quantization": quantization,
        }
        allowed = set(inspect.signature(AsyncEngineArgs.__init__).parameters)
        kwargs = {k: v for k, v in requested.items() if k in allowed and v is not None}
        dropped = sorted(set(requested) - set(kwargs))
        if dropped:
            logger.info("vLLM AsyncEngineArgs ignored unsupported fields: %s", dropped)

        args = AsyncEngineArgs(**kwargs)
        self._engine = AsyncLLMEngine.from_engine_args(args)
        logger.info("vLLM engine ready: %s", model_path)

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

        sp = self._sampling_params(params)
        final = None
        async for result in self._engine.generate(prompt, sp, request_id):
            final = result
        assert final is not None, "vLLM returned no results"

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
        """Streaming generation — yields token deltas as plain strings.

        The server module wraps these in OpenAI SSE envelopes; this layer just
        produces deltas so the engine stays format-agnostic.
        """
        if self._mock:
            for word in prompt.split()[:8]:
                yield word + " "
            return

        sp = self._sampling_params(params)
        last_text = ""
        async for result in self._engine.generate(prompt, sp, request_id):
            new_text = result.outputs[0].text
            delta = new_text[len(last_text):]
            if delta:
                yield delta
            last_text = new_text

    def _sampling_params(self, params: GenerationParams):
        from vllm import SamplingParams
        return SamplingParams(
            max_tokens=params.max_tokens,
            temperature=params.temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            stop=params.stop,
        )
