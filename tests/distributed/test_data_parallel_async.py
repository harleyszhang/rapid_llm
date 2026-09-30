"""Async request-manager tests over a data-parallel controller transport."""

from __future__ import annotations

import asyncio
import queue

import pytest

from rapid_llm.engine.async_engine import AsyncLLMEngine, SchedulerProcessError
from rapid_llm.engine.scheduler_ipc import (
    PROTOCOL_VERSION,
    AbortRequest,
    AddRequest,
    RequestEvent,
    SchedulerEvents,
    SchedulerFailed,
    SchedulerReady,
    ShutdownScheduler,
)

_TIMEOUT = 5.0


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(chr(token) for token in token_ids)


class _Controller:
    def __init__(self) -> None:
        self.scheduler_info = SchedulerReady(PROTOCOL_VERSION, "test", 128, 64, 8)
        self.commands = []
        self.events: queue.Queue = queue.Queue()
        self.alive = True
        self.exitcode = (None, None)
        self.shutdown_calls = 0

    def send(self, command) -> None:
        self.commands.append(command)

    def receive(self, timeout=None):
        return self.events.get(timeout=timeout)

    def is_alive(self) -> bool:
        return self.alive

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.alive = False


@pytest.fixture
async def engine():
    controller = _Controller()
    request_manager = AsyncLLMEngine("stub", _Tokenizer(), {}, controller=controller)
    yield request_manager, controller
    await request_manager.shutdown()


async def _wait_for_add(controller: _Controller, request_id: str) -> AddRequest:
    for _ in range(500):
        for command in controller.commands:
            if isinstance(command, AddRequest) and command.request_id == request_id:
                return command
        await asyncio.sleep(0.001)
    raise AssertionError(f"request {request_id} was not submitted")


async def test_streaming_uses_the_controller_typed_transport(engine):
    request_manager, controller = engine

    async def collect():
        return [chunk async for chunk in request_manager.generate("hi", request_id="mine")]

    task = asyncio.create_task(collect())
    command = await _wait_for_add(controller, "mine")
    assert command.prompt_token_ids == (ord("h"), ord("i"))
    controller.events.put(
        SchedulerEvents(
            outputs=(
                RequestEvent("mine", (ord("o"),)),
                RequestEvent("mine", (ord("k"),), finish_reason="eos", prompt_len=2),
            )
        )
    )
    chunks = await asyncio.wait_for(task, _TIMEOUT)
    assert [chunk.text for chunk in chunks] == ["o", "ok"]
    assert chunks[-1].finish_reason == "eos"


async def test_abandoned_stream_sends_typed_abort(engine):
    request_manager, controller = engine
    stream = request_manager.generate("hi", request_id="mine")
    first = asyncio.create_task(anext(stream))
    await _wait_for_add(controller, "mine")
    controller.events.put(SchedulerEvents(outputs=(RequestEvent("mine", (ord("x"),)),)))
    await asyncio.wait_for(first, _TIMEOUT)
    await stream.aclose()
    assert any(
        isinstance(command, AbortRequest) and command.request_id == "mine"
        for command in controller.commands
    )


async def test_controller_failure_fans_out_and_stops_admission(engine):
    request_manager, controller = engine
    task = asyncio.create_task(request_manager.generate_text("hi", request_id="mine"))
    await _wait_for_add(controller, "mine")
    controller.events.put(SchedulerFailed("rank 1 failed"))
    with pytest.raises(SchedulerProcessError, match="rank 1 failed"):
        await asyncio.wait_for(task, _TIMEOUT)
    with pytest.raises(SchedulerProcessError):
        await anext(request_manager.generate("later"))


async def test_shutdown_is_idempotent_and_reclaims_controller(engine):
    request_manager, controller = engine
    await request_manager.shutdown()
    await request_manager.shutdown()
    assert any(isinstance(command, ShutdownScheduler) for command in controller.commands)
    assert controller.shutdown_calls == 1
