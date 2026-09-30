"""Tests for O10: background tokenize off the engine's critical path.

A delayed fake tokenizer proves ``add_request`` returns before the encode
lands, that the request joins the scheduler on the first step after it does,
and that rejected or failed encodes finish their request (and fire the
caller's ``on_error``) instead of raising from someone else's thread.

Usage:
    pytest tests/engine/test_async_tokenize.py
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
import torch

from rapid_llm.engine.async_engine import AsyncLLMEngine
from rapid_llm.engine.batch_planner import SchedulerConfig
from rapid_llm.engine.scheduler import Scheduler

_EOS = 2
_WORD = 100


class _DelayedTokenizer:
    """Encodes after a visible delay, so timing assertions are unambiguous."""

    def __init__(self, delay: float = 0.05) -> None:
        self.delay = delay
        self.encodes = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.encodes += 1
        time.sleep(self.delay)
        if text == "explode me":
            raise RuntimeError("tokenizer exploded")
        return [] if text == "empty me" else [10, 11, 12]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(token_ids)


class _ScriptedExecutor:
    """One token row per pass; rows repeat once exhausted."""

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        self._calls = 0
        self.num_slots = 4
        self.num_kv_blocks = 0  # no real cache; the scheduler sizes its own pool

    def execute(self, plan):
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        return torch.tensor((row * width)[:width]), None

    def shutdown(self) -> None:
        pass


def _build_engine(rows, *, delay=0.0) -> Scheduler:
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(spec=SimpleNamespace(is_multimodal=False)),
        device="cpu",
        tokenizer=_DelayedTokenizer(delay),
        stop_token_ids={_EOS},
        max_seq_len=64,
    )
    return Scheduler(
        fake,
        SchedulerConfig(max_seq_len=64, max_num_seqs=4),
        executor=_ScriptedExecutor(rows),
        async_tokenize=True,
    )


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_add_request_returns_before_the_encode_lands():
    """The whole point of O10: the caller is not charged the encode time."""
    engine = _build_engine([[_WORD], [_EOS]], delay=0.05)

    request = engine.add_request("hi")

    assert request.prompt_token_ids == []  # tokens not ready yet
    assert request.request_id in engine._tokenizing
    assert engine.has_unfinished_requests()  # so the loop keeps stepping
    engine.shutdown()


def test_the_request_joins_the_scheduler_on_the_next_step():
    engine = _build_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")

    engine._tokenizing[request.request_id].future.result(timeout=5.0)  # encode landed
    engine.step()  # collects, then admits and prefills in the same step

    assert engine.planner.waiting == []
    assert request.prompt_token_ids == [10, 11, 12]
    assert request.delta  # the prefill pass already ran
    engine.shutdown()


def test_generate_matches_the_synchronous_path():
    """Parallel batch encode must produce the same tokens the loop would."""
    rows = [[_WORD, _WORD], [_EOS, _EOS]]
    async_engine = _build_engine(rows)
    outputs = async_engine.generate(["a", "b"])

    assert len(outputs) == 2
    assert all(output.outputs[0].text for output in outputs)
    assert all(output.outputs[0].finish_reason == "eos" for output in outputs)
    async_engine.shutdown()


def test_explicit_token_ids_bypass_the_pool():
    engine = _build_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi", prompt_token_ids=[7, 8])

    assert engine._tokenizing == {}
    assert engine.planner.waiting == [request]
    assert request.prompt_len == 2
    engine.shutdown()


def test_duplicate_id_while_tokenizing_is_rejected_instead_of_overwriting_the_job():
    engine = _build_engine([[_WORD], [_EOS]], delay=0.05)
    first = engine.add_request("first", request_id="same")

    with pytest.raises(ValueError, match="already active"):
        engine.add_request("second", request_id="same")

    assert engine._tokenizing["same"].request is first
    engine.abort("same")
    engine.shutdown()


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #
def test_an_empty_prompt_finishes_invalid_and_fires_on_error():
    engine = _build_engine([[_WORD], [_EOS]])
    fired: list[tuple[object, BaseException]] = []
    request = engine.add_request("empty me", on_error=lambda r, e: fired.append((r, e)))

    engine._tokenizing[request.request_id].future.result(timeout=5.0)
    engine.step()

    assert request.is_finished
    assert request.finish_reason == "invalid"
    assert isinstance(request.error, ValueError)  # the scheduler's rejection
    assert fired and fired[0][0] is request
    assert request.request_id not in engine._tokenizing
    assert engine.planner.waiting == []  # never admitted
    engine.shutdown()


def test_an_encode_failure_finishes_invalid_and_fires_on_error():
    engine = _build_engine([[_WORD], [_EOS]])
    fired: list[tuple[object, BaseException]] = []
    request = engine.add_request("explode me", on_error=lambda r, e: fired.append((r, e)))

    with pytest.raises(RuntimeError):
        engine._tokenizing[request.request_id].future.result(timeout=5.0)  # encode failed
    engine.step()

    assert request.finish_reason == "invalid"
    assert isinstance(request.error, RuntimeError)
    assert fired and isinstance(fired[0][1], RuntimeError)
    engine.shutdown()


def test_abort_cancels_a_still_tokenizing_request():
    engine = _build_engine([[_WORD], [_EOS]], delay=0.05)
    request = engine.add_request("hi")

    aborted = engine.abort(request.request_id)

    assert aborted is request
    assert request.finish_reason == "abort"
    assert request.is_finished
    assert engine._tokenizing == {}
    assert not engine.has_unfinished_requests()
    engine.shutdown()


async def test_a_rejected_prompt_raises_in_the_caller_through_the_async_front_end():
    """The async front end's ``on_error`` wiring: a prompt the background
    encode rejects must surface as the same ValueError the synchronous path
    raises from ``add_request`` — not a hang, and not a dead worker."""
    engine = _build_engine([[_WORD], [_EOS]])

    async def collect(async_engine, prompt):
        return [chunk async for chunk in async_engine.generate(prompt)]

    async with AsyncLLMEngine(engine) as async_engine:
        with pytest.raises(ValueError):
            await asyncio.wait_for(collect(async_engine, "empty me"), 20.0)
