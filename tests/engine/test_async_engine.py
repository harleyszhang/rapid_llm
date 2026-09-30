"""Tests for the asyncio front end of the continuous-batching engine.

A ``StubEngine`` stands in for the real one, so the stream protocol is
what gets tested: chunks until finish, the final chunk carrying whole
text, abandoning a stream aborting the request.

Usage:
    pytest tests/engine/test_async_engine.py
"""

from __future__ import annotations

import asyncio
import gc

import pytest
import torch

from rapid_llm.engine.async_engine import AsyncLLMEngine
from rapid_llm.engine.batch_planner import BatchPlanner, Request, SchedulerConfig
from rapid_llm.engine.sampler import SamplingParams

_TIMEOUT = 20.0


class StubEngine:
    """Counts down a fixed number of tokens per request, no model involved.

    Mimics the parts of :class:`Scheduler` the async layer uses:
    a planner to hold requests, ``step()`` returning whoever advanced, and
    per-request ``delta`` / ``text`` / ``finish_reason``.
    """

    def __init__(self, tokens: int = 4, fail_on_step: int | None = None) -> None:
        self.tokenizer = None
        self.planner = BatchPlanner(
            SchedulerConfig(max_seq_len=64, max_num_seqs=4), num_slots=4
        )
        self._tokens = tokens
        self._fail_on_step = fail_on_step
        self.steps = 0
        self.max_concurrent = 0
        self.released = False

    def add_request(
        self, prompt, params=None, request_id=None, prompt_token_ids=None, on_error=None
    ):
        if prompt == "reject me":
            raise ValueError("prompt refused by the stub")
        request = Request(
            request_id=request_id or f"stub-{self.planner.num_waiting}",
            prompt=prompt,
            prompt_token_ids=[1, 2, 3],
            params=params or SamplingParams(),
        )
        self.planner.add_request(request)
        return request

    def abort(self, request_id):
        return self.planner.abort(request_id)

    def has_unfinished_requests(self):
        return self.planner.has_unfinished_requests()

    def step(self):
        self.steps += 1
        if self._fail_on_step is not None and self.steps >= self._fail_on_step:
            raise RuntimeError("stub step exploded")

        scheduled = self.planner.plan()
        batch = scheduled.prefill or scheduled.decode
        self.max_concurrent = max(self.max_concurrent, len(batch))
        for request in batch:
            request.output_token_ids.append(0)
            request.delta = f"t{len(request.output_token_ids)} "
            request.text += request.delta
            if len(request.output_token_ids) >= self._tokens:
                self.planner.finish(request, "length")
        return batch

    def shutdown(self):
        self.released = True


async def collect(engine, prompt, **kwargs):
    return [chunk async for chunk in engine.generate(prompt, **kwargs)]


# --------------------------------------------------------------------------- #
# Streaming and lifecycle (stub engine, CPU)
# --------------------------------------------------------------------------- #
async def test_generate_streams_until_the_request_finishes():
    async with AsyncLLMEngine(StubEngine(tokens=4)) as engine:
        chunks = await asyncio.wait_for(collect(engine, "hi"), _TIMEOUT)

    assert [c.delta for c in chunks] == ["t1 ", "t2 ", "t3 ", "t4 "]
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].is_finished
    assert all(not c.is_finished for c in chunks[:-1])


async def test_the_last_chunk_carries_the_whole_text():
    async with AsyncLLMEngine(StubEngine(tokens=3)) as engine:
        chunks = await asyncio.wait_for(collect(engine, "hi"), _TIMEOUT)

    assert chunks[-1].text == "".join(c.delta for c in chunks)


async def test_generate_text_returns_only_the_final_chunk():
    async with AsyncLLMEngine(StubEngine(tokens=3)) as engine:
        final = await asyncio.wait_for(engine.generate_text("hi"), _TIMEOUT)

    assert final.text == "t1 t2 t3 "
    assert final.finish_reason == "length"


async def test_concurrent_requests_share_one_batch():
    """The whole reason for the worker thread: coroutines batch together."""
    stub = StubEngine(tokens=6)
    async with AsyncLLMEngine(stub) as engine:
        results = await asyncio.wait_for(
            asyncio.gather(*(collect(engine, f"p{i}") for i in range(4))), _TIMEOUT
        )

    assert all(chunks[-1].finish_reason == "length" for chunks in results)
    assert stub.max_concurrent > 1, "requests were served one at a time"


async def test_request_ids_are_reported_back():
    async with AsyncLLMEngine(StubEngine(tokens=2)) as engine:
        chunks = await asyncio.wait_for(collect(engine, "hi", request_id="mine"), _TIMEOUT)

    assert {c.request_id for c in chunks} == {"mine"}


async def test_duplicate_live_request_id_is_rejected_without_stranding_the_first_stream():
    """A second stream used to replace the first one's delivery queue and hang it."""
    async with AsyncLLMEngine(StubEngine(tokens=500)) as engine:
        first = engine.generate("first", request_id="same")
        await asyncio.wait_for(anext(first), _TIMEOUT)

        with pytest.raises(ValueError, match="already active"):
            await anext(engine.generate("second", request_id="same"))

        await first.aclose()


async def test_abandoning_a_stream_aborts_the_request():
    """An abandoned HTTP connection must free its slot, not run to the cap."""
    stub = StubEngine(tokens=500)
    async with AsyncLLMEngine(stub) as engine:
        stream = engine.generate("hi", request_id="dropped")
        async for _ in stream:
            break
        await stream.aclose()

        for _ in range(50):
            await asyncio.sleep(0.02)
            if stub.planner.num_running == 0:
                break

        assert stub.planner.num_running == 0
        assert stub.planner.num_free_slots == stub.planner.num_slots


async def test_an_idle_engine_does_not_spin():
    """With nothing queued the worker blocks on its command queue."""
    stub = StubEngine(tokens=2)
    async with AsyncLLMEngine(stub) as engine:
        engine.start()
        await asyncio.sleep(0.3)
        assert stub.steps == 0

        await asyncio.wait_for(collect(engine, "hi"), _TIMEOUT)
        assert stub.steps > 0


async def test_a_refused_prompt_raises_in_the_caller():
    """Rejection is that one request's problem, and must not kill the worker."""
    stub = StubEngine(tokens=2)
    async with AsyncLLMEngine(stub) as engine:
        with pytest.raises(ValueError):
            await asyncio.wait_for(collect(engine, "reject me"), _TIMEOUT)

        # The engine keeps serving afterwards.
        chunks = await asyncio.wait_for(collect(engine, "fine"), _TIMEOUT)
        assert chunks[-1].finish_reason == "length"


async def test_a_failing_step_surfaces_and_clears_the_queue():
    """A broken step must raise to the caller instead of looping forever."""
    stub = StubEngine(tokens=10, fail_on_step=2)
    async with AsyncLLMEngine(stub) as engine:
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(collect(engine, "hi"), _TIMEOUT)

        assert stub.planner.num_running == 0


async def test_shutdown_is_idempotent_and_stops_the_worker():
    engine = AsyncLLMEngine(StubEngine(tokens=2))
    engine.start()
    await asyncio.wait_for(collect(engine, "hi"), _TIMEOUT)

    await engine.shutdown()
    await engine.shutdown()


async def test_generate_after_shutdown_is_rejected():
    """A stopped worker cannot be restarted because its stop event is permanent."""
    engine = AsyncLLMEngine(StubEngine(tokens=2))
    await engine.shutdown()

    with pytest.raises(RuntimeError, match="has been shut down"):
        await anext(engine.generate("hi"))


async def test_shutdown_releases_the_engine():
    """The worker must hand the engine back, or tensor-parallel ranks never exit.

    Under tensor parallelism the follower processes sit in a broadcast waiting
    for the next plan, and the stop signal only reaches them through the
    engine's own ``shutdown``. A server that closed without it would leave a
    process per extra GPU behind, still holding its weights.
    """
    stub = StubEngine(tokens=2)
    async with AsyncLLMEngine(stub) as engine:
        await asyncio.wait_for(collect(engine, "hi"), _TIMEOUT)

    assert stub.released


async def test_the_engine_serves_a_second_event_loop():
    """A queue awaited on one loop is never woken by a put onto another.

    Regression test: the engine used to bind one loop at ``start()``, so an ASGI
    test client -- which runs the app in its own loop on another thread -- would
    hang forever instead of receiving anything.
    """
    engine = AsyncLLMEngine(StubEngine(tokens=3))
    engine.start()
    try:
        first = await asyncio.wait_for(collect(engine, "loop-one"), _TIMEOUT)
        assert first[-1].finish_reason == "length"

        def other_loop() -> list:
            return asyncio.run(asyncio.wait_for(collect(engine, "loop-two"), _TIMEOUT))

        second = await asyncio.get_running_loop().run_in_executor(None, other_loop)
        assert second[-1].finish_reason == "length"
    finally:
        await engine.shutdown()


# --------------------------------------------------------------------------- #
# Against a real checkpoint
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.weights
async def test_concurrent_coroutines_get_their_own_answers(model_dir):
    """Real model, three coroutines, one batch: nobody may get another's text."""
    from rapid_llm.engine.llm_engine import LLMEngine
    from rapid_llm.engine.scheduler import Scheduler

    engine = AsyncLLMEngine(
        Scheduler(
            LLMEngine(
                str(model_dir), max_seq_len=512, max_gpu_num_blocks=8192, use_cuda_graph=False
            ),
            SchedulerConfig(max_seq_len=512, max_num_seqs=4),
        )
    )
    params = SamplingParams(temperature=0.0, max_gen_len=16, repetition_penalty=1.0)
    prompts = ["The capital of France is", "Two plus two is", "The sky is"]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(collect(engine, p, sampling_params=params) for p in prompts)),
            120.0,
        )
    finally:
        await engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    for prompt, chunks in zip(prompts, results, strict=True):
        text = "".join(c.delta for c in chunks)
        assert text, f"{prompt!r} produced nothing"
        assert text == chunks[-1].text, "deltas must rebuild the final text"
        assert chunks[-1].finish_reason in {"eos", "length"}
