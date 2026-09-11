"""Process lifecycle of the engine core client: start, serve, stop, crash.

Each test builds its own client -- the subject is the lifecycle itself, so
nothing may be shared between cases (a killed child cannot be borrowed). The
cases mirror vLLM's ``tests/v1/engine`` lifecycle coverage: graceful shutdown
is a zero exit, a hard kill propagates to every open stream, and a child that
cannot load says why through its death notice.

Usage:
    pytest tests/engine/test_engine_core_client.py
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from rapid_llm.engine.engine_core_client import EngineCoreClient, EngineDeadError
from rapid_llm.engine.sampler import SamplingParams

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

#: Generous: the first request of a test pays for weight loading in the child.
_TIMEOUT = 180.0

#: Small enough that several clients in one session never fight over memory.
_ENGINE_OPTS = {"max_seq_len": 256, "max_num_seqs": 2, "use_cuda_graph": False}


def _load(model_dir) -> EngineCoreClient:
    return EngineCoreClient.from_pretrained(str(model_dir), **_ENGINE_OPTS)


async def test_graceful_shutdown_leaves_no_process_behind(model_dir):
    client = _load(model_dir)
    try:
        client.start()
        final = await asyncio.wait_for(
            client.generate_text("Hello", SamplingParams(temperature=0.0, max_gen_len=4)),
            _TIMEOUT,
        )
        assert final.is_finished
        proc = client._proc
    finally:
        await asyncio.wait_for(client.shutdown(), _TIMEOUT)

    assert not proc.is_alive()
    assert proc.exitcode == 0, "SHUTDOWN must end the child cleanly, not kill it"
    # Idempotent: a second shutdown is a no-op, not an error.
    await client.shutdown()


async def test_requests_after_shutdown_are_refused(model_dir):
    client = _load(model_dir)
    client.start()
    await asyncio.wait_for(client.shutdown(), _TIMEOUT)

    with pytest.raises(RuntimeError):
        await anext(client.generate("hi"))


async def test_a_killed_child_fails_in_flight_and_future_requests(model_dir):
    client = _load(model_dir)
    try:
        client.start()
        params = SamplingParams(temperature=0.0, max_gen_len=256)
        stream = client.generate("Tell me a very long story about the sea", params)
        first = await asyncio.wait_for(anext(stream), _TIMEOUT)
        assert first is not None

        client._proc.kill()  # SIGKILL: no goodbye frame, only the exit itself

        with pytest.raises(EngineDeadError):
            async for _chunk in stream:
                pass
        with pytest.raises(EngineDeadError):
            await asyncio.wait_for(
                client.generate_text("hi", SamplingParams(temperature=0.0, max_gen_len=2)),
                _TIMEOUT,
            )
    finally:
        await asyncio.wait_for(client.shutdown(), _TIMEOUT)


async def test_a_child_that_cannot_load_reports_why(model_dir, tmp_path):
    """The death notice is the point: a load failure must arrive as its reason,
    not as a bare exit code from a liveness poll."""
    broken = tmp_path / "broken-model"
    broken.mkdir()
    for path in Path(model_dir).iterdir():
        if path.is_file() and path.suffix not in (".safetensors", ".bin"):
            shutil.copy(path, broken / path.name)

    with pytest.raises(RuntimeError, match="engine core failed to start"):
        EngineCoreClient.from_pretrained(str(broken), startup_timeout_s=120.0, **_ENGINE_OPTS)


async def test_shutdown_ends_open_streams_instead_of_hanging(model_dir):
    client = _load(model_dir)
    client.start()
    params = SamplingParams(temperature=0.0, max_gen_len=128)
    stream = client.generate("Tell me a long story", params)
    await asyncio.wait_for(anext(stream), _TIMEOUT)

    await asyncio.wait_for(client.shutdown(), _TIMEOUT)

    # The stream ends because there is no more output -- not with an exception,
    # and not by hanging.
    remaining = [chunk async for chunk in stream]
    assert all(not chunk.is_finished for chunk in remaining)
