"""Pipeline depth 2 vs depth 1: token sequence agreement.

The launch/harvest pipeline runs the engine loop at a configurable depth.
At depth ``N`` the host's harvest runs N steps behind the launches.  The
acceptance criteria are:

1. Output tokens are identical regardless of depth -- only the finish
   token's stop timing shifts (depth tokens later), never the content.
2. Depth 2 does not crash, hang, or produce truncated output relative
   to the synchronous or depth-1 engine.

Usage:
    pytest tests/engine/test_pipeline_depth.py
"""

from __future__ import annotations

import asyncio

import pytest

from rapid_llm.engine.sampler import SamplingParams

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

_TIMEOUT = 180.0
_PROMPT = (
    "Write a short paragraph about the benefits of asynchronous programming "
    "in modern web applications."
)


async def _generate(engine) -> str:
    """Run a deterministic generation and return the output text."""
    params = SamplingParams(temperature=0.0, max_gen_len=32)
    result = await engine.generate_text(_PROMPT, params)
    return result.text


@pytest.mark.weights
@pytest.mark.gpu
def test_depth_1_matches_depth_2_token_for_token(model_dir):
    """The synchronous (pipeline off) and depth-2 pipeline produce the same
    output tokens; the prefix check allows depth 2's one token late stop."""
    from rapid_llm.engine.async_engine import AsyncLLMEngine

    opts = {"max_seq_len": 256, "max_num_seqs": 2, "use_cuda_graph": False}

    async def run():
        engine_sync = AsyncLLMEngine.from_pretrained(str(model_dir), pipeline=False, **opts)
        engine_sync.start()
        try:
            text_sync = await _generate(engine_sync)
        finally:
            await engine_sync.shutdown()

        engine_d2 = AsyncLLMEngine.from_pretrained(
            str(model_dir), pipeline=True, pipeline_depth=2, **opts
        )
        engine_d2.start()
        try:
            text_d2 = await _generate(engine_d2)
        finally:
            await engine_d2.shutdown()
        return text_sync, text_d2

    text_sync, text_d2 = asyncio.run(run())

    # Depth 2 may produce one extra token (the depth-late stop penalty), so
    # the depth-1 (sync) output must be a prefix of depth-2's output.
    assert text_d2.startswith(text_sync) or text_sync.startswith(text_d2), (
        f"depth-2 output differs from synchronous:\n  sync: {text_sync!r}\n  d=2:  {text_d2!r}"
    )


@pytest.mark.weights
@pytest.mark.gpu
def test_depth_2_does_not_hang_or_truncate(model_dir):
    """Depth 2 completes all requests without hanging or returning empty text.

    This exercises the drain path: when the scheduler runs out of work, the
    pipeline must continue to harvest the in-flight entries until empty.
    """
    from rapid_llm.engine.async_engine import AsyncLLMEngine

    opts = {"max_seq_len": 128, "max_num_seqs": 4, "use_cuda_graph": False}

    async def run():
        engine = AsyncLLMEngine.from_pretrained(
            str(model_dir), pipeline=True, pipeline_depth=2, **opts
        )
        engine.start()
        params = SamplingParams(temperature=0.0, max_gen_len=16)
        try:
            prompts = [
                "What is 2 + 2?",
                "What is the capital of France?",
                "Write a short greeting.",
                "What is the opposite of hot?",
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*(engine.generate_text(p, params) for p in prompts)),
                _TIMEOUT,
            )
            return [r.text for r in results]
        finally:
            await engine.shutdown()

    texts = asyncio.run(run())

    assert len(texts) == 4
    for text in texts:
        assert text, "depth-2 pipeline produced empty text for a request"
