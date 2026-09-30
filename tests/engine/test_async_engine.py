"""Async request-manager tests over a lightweight scheduler process."""

from __future__ import annotations

import asyncio
import queue
import time

import pytest
import torch

from rapid_llm.engine.async_engine import AsyncLLMEngine, SchedulerProcessError
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler_ipc import (
    PROTOCOL_VERSION,
    AbortRequest,
    AddRequest,
    RequestEvent,
    SchedulerEvents,
    SchedulerFailed,
    SchedulerReady,
    ShutdownScheduler,
    UtilityEvent,
    UtilityRequest,
)

_TIMEOUT = 20.0


class StubTokenizer:
    def encode(self, prompt: str, add_special_tokens: bool = True) -> list[int]:
        if prompt == "empty":
            return []
        if prompt == "reject":
            return [999]
        return [1, 2, 3]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(f"t{token} " for token in token_ids)


def _stub_scheduler_process(model, options, commands, events) -> None:
    events.put(SchedulerReady(PROTOCOL_VERSION, "test", 64, 32, 4))
    active: dict[str, tuple[int, int]] = {}
    step = 0
    stopping = False
    while not stopping:
        try:
            command = commands.get(timeout=0.01 if active else 1.0)
        except queue.Empty:
            command = None

        pending = [] if command is None else [command]
        while True:
            try:
                pending.append(commands.get_nowait())
            except queue.Empty:
                break
        for command in pending:
            if isinstance(command, ShutdownScheduler):
                stopping = True
            elif isinstance(command, AbortRequest):
                active.pop(command.request_id, None)
            elif isinstance(command, UtilityRequest):
                events.put(
                    SchedulerEvents(
                        utility_output=UtilityEvent(
                            command.call_id,
                            {"running": len(active), "waiting": 0},
                        )
                    )
                )
            elif isinstance(command, AddRequest):
                if command.prompt_token_ids == (999,):
                    events.put(
                        SchedulerEvents(
                            outputs=(
                                RequestEvent(
                                    command.request_id,
                                    (),
                                    finish_reason="invalid",
                                    error="prompt refused by scheduler",
                                ),
                            )
                        )
                    )
                else:
                    limit = command.sampling_params.max_gen_len or options.get("tokens", 4)
                    active[command.request_id] = (0, limit)

        if stopping or not active:
            continue
        step += 1
        if options.get("fail_after") == step:
            events.put(SchedulerFailed("RuntimeError: stub step exploded"))
            return

        outputs = []
        for request_id, (count, limit) in tuple(active.items()):
            count += 1
            finished = count >= min(limit, options.get("tokens", 4))
            outputs.append(
                RequestEvent(
                    request_id=request_id,
                    new_token_ids=(count,),
                    finish_reason="length" if finished else None,
                    prompt_len=3,
                )
            )
            if finished:
                active.pop(request_id)
            else:
                active[request_id] = (count, limit)
        events.put(SchedulerEvents(outputs=tuple(outputs)))
        time.sleep(0.001)


def _failing_start_process(_model, _options, _commands, events) -> None:
    events.put(SchedulerFailed("RuntimeError: model load failed"))


def _engine(**options) -> AsyncLLMEngine:
    return AsyncLLMEngine(
        "stub",
        StubTokenizer(),
        options,
        process_target=_stub_scheduler_process,
        startup_timeout_s=_TIMEOUT,
    )


async def _collect(engine, prompt, **kwargs):
    return [chunk async for chunk in engine.generate(prompt, **kwargs)]


def test_scheduler_protocol_version_is_pinned():
    assert PROTOCOL_VERSION == 2


async def test_generate_streams_until_the_request_finishes():
    async with _engine(tokens=4) as engine:
        chunks = await asyncio.wait_for(_collect(engine, "hi"), _TIMEOUT)

    assert [chunk.delta for chunk in chunks] == ["t1 ", "t2 ", "t3 ", "t4 "]
    assert chunks[-1].text == "".join(chunk.delta for chunk in chunks)
    assert chunks[-1].finish_reason == "length"
    assert all(not chunk.is_finished for chunk in chunks[:-1])


async def test_generate_text_returns_only_the_final_chunk():
    async with _engine(tokens=3) as engine:
        final = await asyncio.wait_for(engine.generate_text("hi"), _TIMEOUT)

    assert final.text == "t1 t2 t3 "
    assert final.completion_tokens == 3


async def test_concurrent_requests_keep_independent_streams():
    async with _engine(tokens=6) as engine:
        results = await asyncio.wait_for(
            asyncio.gather(*(_collect(engine, f"p{i}") for i in range(4))), _TIMEOUT
        )

    assert all(chunks[-1].text == "t1 t2 t3 t4 t5 t6 " for chunks in results)


async def test_duplicate_live_request_id_is_rejected():
    async with _engine(tokens=500) as engine:
        first = engine.generate("first", request_id="same")
        await asyncio.wait_for(anext(first), _TIMEOUT)
        with pytest.raises(ValueError, match="already active"):
            await anext(engine.generate("second", request_id="same"))
        await first.aclose()


async def test_abandoning_a_stream_sends_abort_and_engine_keeps_serving():
    async with _engine(tokens=500) as engine:
        stream = engine.generate("first", request_id="dropped")
        await asyncio.wait_for(anext(stream), _TIMEOUT)
        await stream.aclose()
        final = await asyncio.wait_for(
            engine.generate_text("next", SamplingParams(max_gen_len=2)), _TIMEOUT
        )

    assert final.is_finished


async def test_parent_and_scheduler_rejections_are_request_local():
    async with _engine(tokens=2) as engine:
        with pytest.raises(ValueError, match="empty"):
            await anext(engine.generate("empty"))
        with pytest.raises(ValueError, match="refused"):
            await anext(engine.generate("reject"))
        assert (await engine.generate_text("fine")).is_finished


async def test_scheduler_failure_reaches_active_and_future_requests():
    engine = _engine(tokens=10, fail_after=2)
    try:
        with pytest.raises(SchedulerProcessError, match="stub step exploded"):
            await asyncio.wait_for(engine.generate_text("hi"), _TIMEOUT)
        with pytest.raises(SchedulerProcessError):
            await anext(engine.generate("later"))
    finally:
        await engine.shutdown()


async def test_hard_process_exit_fails_in_flight_and_future_requests():
    engine = _engine(tokens=500)
    stream = engine.generate("long")
    try:
        await asyncio.wait_for(anext(stream), _TIMEOUT)
        engine._process.kill()
        with pytest.raises(SchedulerProcessError, match="exited unexpectedly"):
            await asyncio.wait_for(anext(stream), _TIMEOUT)
        with pytest.raises(SchedulerProcessError):
            await anext(engine.generate("later"))
    finally:
        await stream.aclose()
        await engine.shutdown()


def test_startup_failure_preserves_the_child_error():
    with pytest.raises(RuntimeError, match="model load failed"):
        AsyncLLMEngine(
            "stub",
            StubTokenizer(),
            {},
            process_target=_failing_start_process,
            startup_timeout_s=_TIMEOUT,
        )


async def test_shutdown_is_idempotent_and_reaps_process():
    engine = _engine(tokens=2)
    process = engine._process
    await asyncio.wait_for(engine.generate_text("hi"), _TIMEOUT)
    await engine.shutdown()
    await engine.shutdown()

    assert not process.is_alive()
    assert process.exitcode == 0


async def test_generate_after_shutdown_is_rejected():
    engine = _engine()
    await engine.shutdown()
    with pytest.raises(RuntimeError, match="closed"):
        await anext(engine.generate("hi"))


async def test_shutdown_ends_open_streams():
    engine = _engine(tokens=500)
    stream = engine.generate("long")
    await asyncio.wait_for(anext(stream), _TIMEOUT)
    await engine.shutdown()
    remaining = [chunk async for chunk in stream]
    assert all(not chunk.is_finished for chunk in remaining)


async def test_request_manager_serves_a_second_event_loop():
    engine = _engine(tokens=3)
    engine.start()
    try:
        first = await asyncio.wait_for(_collect(engine, "loop-one"), _TIMEOUT)

        def other_loop() -> list:
            return asyncio.run(asyncio.wait_for(_collect(engine, "loop-two"), _TIMEOUT))

        second = await asyncio.get_running_loop().run_in_executor(None, other_loop)
    finally:
        await engine.shutdown()

    assert first[-1].text == second[-1].text


@pytest.mark.gpu
@pytest.mark.weights
async def test_concurrent_coroutines_get_their_own_answers(model_dir):
    engine = AsyncLLMEngine.from_pretrained(
        str(model_dir), max_seq_len=512, max_num_seqs=4, use_cuda_graph=False
    )
    params = SamplingParams(
        temperature=0.0,
        max_gen_len=16,
        repetition_penalty=1.0,
        logprobs=3,
        prompt_logprobs=2,
    )
    prompts = ["The capital of France is", "Two plus two is", "The sky is"]
    try:
        info = engine.scheduler_info
        assert info.protocol_version == PROTOCOL_VERSION
        assert info.max_model_len == 512
        assert info.max_num_seqs == 4
        assert info.num_gpu_blocks > 0
        results = await asyncio.wait_for(
            asyncio.gather(
                *(_collect(engine, prompt, sampling_params=params) for prompt in prompts)
            ),
            120.0,
        )
    finally:
        await engine.shutdown()

    for prompt, chunks in zip(prompts, results, strict=True):
        text = "".join(chunk.delta for chunk in chunks)
        assert text, f"{prompt!r} produced nothing"
        assert text == chunks[-1].text
        assert chunks[-1].finish_reason in {"eos", "length"}
        records = [chunk.logprobs for chunk in chunks if chunk.logprobs is not None]
        assert records
        assert all(len(record.top_token_ids) == 3 for record in records)
        assert chunks[-1].prompt_logprobs is not None


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
async def test_tensor_parallel_scheduler_process_serves(model_dir):
    async with AsyncLLMEngine.from_pretrained(
        str(model_dir),
        max_seq_len=256,
        max_num_seqs=2,
        tensor_parallel_size=2,
        use_cuda_graph=False,
    ) as engine:
        final = await asyncio.wait_for(
            engine.generate_text(
                "The capital of France is", SamplingParams(temperature=0.0, max_gen_len=4)
            ),
            _TIMEOUT,
        )
    assert final.is_finished
    assert final.text
