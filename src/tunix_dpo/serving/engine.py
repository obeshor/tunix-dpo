"""
Async vLLM engine wrapper with mock fallback.

Isolates all vLLM-specific code so the server module stays clean and
tests can run without a GPU by swapping in the mock engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncGenerator, Optional

from tunix_dpo.serving.metrics import Metrics

log = logging.getLogger(__name__)


class VLLMEngine:
    """Async wrapper around ``vllm.AsyncLLMEngine``.

    Falls back to an echo mock if vLLM is not installed, so the server
    can be tested on CPU without a GPU.

    Parameters
    ----------
    model_path:
        HuggingFace model directory or Hub ID.
    dtype:
        ``"bfloat16"`` | ``"float16"`` | ``"auto"``
    max_model_len:
        Maximum sequence length (prompt + generation).
    gpu_memory_utilization:
        Fraction of GPU VRAM reserved for the KV cache.
    tensor_parallel_size:
        Number of GPUs for tensor parallelism.
    quantization:
        ``None`` | ``"gptq"`` | ``"awq"`` | ``"bitsandbytes"``
    max_num_seqs:
        Maximum concurrent sequences in the continuous-batching queue.
    metrics:
        Shared ``Metrics`` instance for telemetry.
    """

    def __init__(
        self,
        model_path:              str,
        dtype:                   str,
        max_model_len:           int,
        gpu_memory_utilization:  float,
        tensor_parallel_size:    int,
        quantization:            Optional[str],
        max_num_seqs:            int,
        metrics:                 Metrics,
    ) -> None:
        self._engine         = None
        self._SamplingParams = None
        self._metrics        = metrics
        self.model_path      = model_path
        self.ready           = False

        try:
            from vllm import AsyncEngineArgs, AsyncLLMEngine       # type: ignore[import]
            from vllm.sampling_params import SamplingParams        # type: ignore[import]

            engine_args = AsyncEngineArgs(
                model=model_path,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                quantization=quantization or None,
                trust_remote_code=True,
                enable_prefix_caching=True,
                disable_log_requests=True,
                max_num_seqs=max_num_seqs,
            )
            self._engine         = AsyncLLMEngine.from_engine_args(engine_args)
            self._SamplingParams = SamplingParams
            self.ready           = True
            log.info("vLLM engine ready: %s", model_path)

        except ImportError:
            log.warning("vLLM not installed — running in echo mock mode.")
            self.ready = True

        except Exception as exc:
            log.error("Engine init failed: %s", exc)
            raise

    async def generate(
        self,
        prompt:             str,
        max_tokens:         int,
        temperature:        float,
        top_p:              float,
        top_k:              int,
        repetition_penalty: float,
        stop:               Optional[list[str]],
        seed:               Optional[int],
        request_id:         str,
    ) -> AsyncGenerator[tuple[str, bool], None]:
        """Yield (text_delta, is_final) pairs.

        The final tuple has is_final=True; text may be empty on the last yield.
        """
        t0 = time.perf_counter()

        if self._engine is None:
            # Mock: stream back the first 80 chars of the prompt
            words = f"[MOCK] {prompt[:80]}".split()
            for i, word in enumerate(words[:max_tokens]):
                await asyncio.sleep(0.005)
                yield word + " ", i == len(words) - 1
            lat = (time.perf_counter() - t0) * 1000
            self._metrics.record(len(words), lat)
            return

        sampling = self._SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop=stop or [],
            seed=seed,
        )

        prev_len     = 0
        final_output = None

        async for req_out in self._engine.generate(prompt, sampling, request_id):
            if req_out.outputs:
                out      = req_out.outputs[0]
                delta    = out.text[prev_len:]
                prev_len = len(out.text)
                is_final = out.finish_reason is not None
                yield delta, is_final
                final_output = out

        lat      = (time.perf_counter() - t0) * 1000
        n_tokens = len(final_output.token_ids) if final_output else 0
        self._metrics.record(n_tokens, lat)
        log.debug("[%s] %d tokens  %.0f ms", request_id, n_tokens, lat)
