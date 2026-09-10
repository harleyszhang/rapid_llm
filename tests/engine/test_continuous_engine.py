"""Step-level tests for the continuous-batching engine, without a model.

A scripted executor drives the plan -> execute -> harvest loop on CPU,
covering the harvest layer — stop handling, finish accounting, the
``step()`` return contract: a request stopping this step still appears
in the return with its finish reason, or its stream strands forever.

Usage:
    pytest tests/engine/test_continuous_engine.py
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from rapid_llm.engine.async_engine import AsyncLLMEngine
from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.sampler import PositionLogprobs, SamplingParams
from rapid_llm.engine.scheduler import SchedulerConfig
from rapid_llm.executor.worker import PassLogprobs

_EOS = 2
_WORD = 100  # any token id the stop set does not contain
_TIMEOUT = 20.0


class _FakeTokenizer:
    """The two methods the engine calls: encode for prompts, decode for deltas."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [10, 11, 12]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "x" * len(token_ids)


class _ScriptedExecutor:
    """Returns scripted token rows, one row per ``execute`` call.

    The step loop zips what a pass returns with the requests that pass named,
    in order, so a script row is simply the token each request receives that
    pass. Rows run out and then the last one repeats, keeping a drain loop
    simple to write.
    """

    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows
        self._calls = 0
        self.num_slots = 4
        # 0 means "cannot say": the scheduler then sizes its block pool from the
        # slot geometry, which is what a fake with no real cache wants.
        self.num_kv_blocks = 0

    def execute(self, plan) -> tuple[torch.Tensor, None]:
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        return torch.tensor((row * width)[:width]), None

    def shutdown(self) -> None:
        pass


def _build_engine(rows: list[list[int]]) -> ContinuousBatchingEngine:
    """A real ContinuousBatchingEngine over a fake LLMEngine and scripted passes."""
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec=SimpleNamespace(is_multimodal=False),
            config=SimpleNamespace(kv_cache_torch_dtype=None),
        ),
        device="cpu",
        tokenizer=_FakeTokenizer(),
        stop_token_ids={_EOS},
        max_seq_len=64,
    )
    return ContinuousBatchingEngine(
        fake,
        SchedulerConfig(max_seq_len=64, max_num_seqs=4),
        executor=_ScriptedExecutor(rows),
    )


async def _collect(engine: AsyncLLMEngine, prompt: str) -> list:
    return [chunk async for chunk in engine.generate(prompt)]


def test_a_request_stopping_on_eos_is_returned_by_step():
    """Regression: an eos finish used to be dropped from step()'s return.

    The request finishes inside harvest, which used to skip past it — so a
    caller draining step() never saw the finish reason. It must come back with
    the reason set and an empty delta: the stop token is punctuation, not
    output.
    """
    engine = _build_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")

    first = engine.step()  # prefill: one ordinary token
    assert [r.request_id for r in first] == [request.request_id]
    assert request.delta
    assert request.finish_reason is None

    last = engine.step()  # decode: the stop token
    assert [r.request_id for r in last] == [request.request_id]
    assert request.finish_reason == "eos"
    assert request.delta == ""
    assert request.is_finished
    assert not engine.has_unfinished_requests()
    engine.shutdown()


def test_the_eos_stop_token_is_not_counted_as_output():
    """The request ends with only its prefill token in output_token_ids."""
    engine = _build_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")

    engine.step()
    engine.step()

    assert request.output_token_ids == [_WORD]
    engine.shutdown()


def test_duplicate_live_request_ids_are_rejected_without_losing_state():
    engine = _build_engine([[_WORD], [_EOS]])
    first = engine.add_request("first", request_id="same")

    with pytest.raises(ValueError, match="already active"):
        engine.add_request("second", request_id="same")

    assert engine.scheduler.waiting == [first]
    engine.shutdown()


def test_generated_request_ids_skip_user_supplied_ids():
    engine = _build_engine([[_WORD], [_EOS]])
    explicit = engine.add_request("first", request_id="req-0")
    generated = engine.add_request("second")

    assert explicit.request_id == "req-0"
    assert generated.request_id == "req-1"
    engine.shutdown()


async def test_an_eos_request_does_not_strand_its_stream():
    """Same regression, through the async front end: the stream must hear it.

    The worker publishes only what step() returns, so a finish missing from
    that list leaves the awaiting coroutine blocked on a final chunk that
    never arrives — a hang, not an error, invisible until a request just
    stops responding. The wait_for turns that hang into a test failure.
    """
    engine = _build_engine([[_WORD], [_EOS]])
    async with AsyncLLMEngine(engine) as async_engine:
        chunks = await asyncio.wait_for(_collect(async_engine, "hi"), _TIMEOUT)

    assert chunks[-1].finish_reason == "eos"
    assert chunks[-1].is_finished
    assert all(chunk.finish_reason is None for chunk in chunks[:-1])
    assert chunks[-1].text  # the prefill token still reached the caller


def _record(token_id: int, logprob: float = -0.1) -> PositionLogprobs:
    return PositionLogprobs(
        token_id=token_id,
        logprob=logprob,
        top_token_ids=(token_id,),
        top_logprobs=(logprob,),
    )


class _LogprobsExecutor(_ScriptedExecutor):
    """Scripted tokens plus the logprob records the plan asks for.

    Builds records the way the real worker does: one sampled record per sampled
    row, and per chunk the prompt rows the position contract calls for — every
    row of a partial chunk, all but the sampling row of a final one. Each
    record's ``token_id`` is the row's target, so a test can verify records
    landed on the *right* position, not just the right count.
    """

    def execute(self, plan):
        row = self._rows[min(self._calls, len(self._rows) - 1)]
        self._calls += 1
        width = len(plan.sampling)
        tokens = torch.tensor((row * width)[:width])
        sampled = None
        if any(params.logprobs is not None for params in plan.sampling):
            sampled = tuple(_record(int(token)) for token in tokens)
        prompt: list = [None] * len(plan.slots)
        if plan.prompt_logprobs:
            sampled_set = set(plan.sampled)
            offset = 0
            for index in range(len(plan.slots)):
                chunk = plan.seq_lens[index] - plan.seq_starts[index]
                offset += chunk
                if plan.prompt_logprobs[index] is None:
                    continue
                rows = chunk - 1 if index in sampled_set else chunk
                targets = plan.prompt_targets[offset - chunk : offset - chunk + rows]
                prompt[index] = tuple(_record(target) for target in targets)
        return tokens, PassLogprobs(
            sampled=sampled or (),
            prompt=tuple(prompt),
        )


def _build_logprobs_engine(rows, **config) -> ContinuousBatchingEngine:
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec=SimpleNamespace(is_multimodal=False),
            config=SimpleNamespace(kv_cache_torch_dtype=None),
        ),
        device="cpu",
        tokenizer=_FakeTokenizer(),
        stop_token_ids={_EOS},
        max_seq_len=64,
    )
    return ContinuousBatchingEngine(
        fake,
        SchedulerConfig(max_seq_len=64, max_num_seqs=4, **config),
        executor=_LogprobsExecutor(rows),
    )


def test_logprobs_ride_the_step_to_the_request():
    """A sampled record lands on delta_logprobs for the step and output_logprobs for good."""
    engine = _build_logprobs_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi", SamplingParams(logprobs=1))

    engine.step()  # prefill: one ordinary token
    assert request.delta_logprobs is not None
    assert request.delta_logprobs.token_id == _WORD
    assert len(request.output_logprobs) == 1

    engine.step()  # decode: the eos token, whose record must be dropped
    assert request.finish_reason == "eos"
    assert len(request.output_logprobs) == 1
    assert request.output_logprobs[0].token_id == _WORD
    engine.shutdown()


def test_requests_that_never_ask_carry_no_records():
    engine = _build_logprobs_engine([[_WORD], [_EOS]])
    request = engine.add_request("hi")  # no logprobs requested

    engine.step()
    assert request.delta_logprobs is None
    assert request.output_logprobs is None
    engine.shutdown()


def test_prompt_logprobs_are_attributed_across_chunks():
    """A 3-token prompt with a 2-token chunk budget scores positions 1 and 2.

    The first chunk [0, 2) is partial and scores both its rows (positions 1, 2);
    the final chunk [2, 3) has only its sampling row, so it scores nothing.
    Position 0 has no predictor and stays ``None``. The record's token id is the
    target the fake scored, which pins each record to its position.
    """
    engine = _build_logprobs_engine([[_WORD], [_EOS]], max_num_batched_tokens=2)
    request = engine.add_request("hi", SamplingParams(prompt_logprobs=1))
    assert request.prompt_len == 3  # the fake tokenizer encodes [10, 11, 12]

    engine.step()  # partial chunk [0, 2)
    assert request.prompt_logprobs[0] is None
    assert request.prompt_logprobs[1].token_id == 11
    assert request.prompt_logprobs[2].token_id == 12

    engine.step()  # final chunk + first sampled token; nothing new is attributed
    assert [r is None for r in request.prompt_logprobs] == [True, False, False]
    engine.shutdown()


def test_generate_returns_logprobs_in_the_request_output():
    engine = _build_logprobs_engine([[_WORD], [_EOS]])
    outputs = engine.generate(["hi"], SamplingParams(logprobs=1, prompt_logprobs=1))

    (output,) = outputs
    assert output.prompt_logprobs[0] is None
    assert output.prompt_logprobs[1].token_id == 11
    completion = output.outputs[0]
    assert [r.token_id for r in completion.logprobs] == [_WORD]
    engine.shutdown()


# --------------------------------------------------------------------------- #
# Observability (A7): the engine reports its own numbers
# --------------------------------------------------------------------------- #
def test_a_finished_request_lands_in_the_metrics():
    """One engine run answers for counters, histograms and the gauges."""
    engine = _build_engine([[_WORD], [_EOS]])
    engine.add_request("hi")

    engine.step()
    assert engine.metrics.prompt_tokens_total._values.get((), 0) == 0  # not finished yet
    engine.step()

    text = engine.metrics.render_prometheus()
    assert 'rapid_llm:request_success_total{finish_reason="eos"} 1' in text
    assert "rapid_llm:prompt_tokens_total 3" in text  # the fake prompt is 3 tokens
    assert "rapid_llm:generation_tokens_total 1" in text  # the eos token is not output
    assert "rapid_llm:request_queue_time_seconds_count 1" in text
    assert "rapid_llm:time_to_first_token_seconds_count 1" in text
    assert "rapid_llm:num_requests_running 0" in text  # drained by the end
    engine.shutdown()


def test_an_aborted_request_is_counted_without_finishing():
    engine = _build_engine([[_WORD]])
    request = engine.add_request("hi")

    engine.abort(request.request_id)

    text = engine.metrics.render_prometheus()
    assert 'rapid_llm:request_success_total{finish_reason="abort"} 1' in text
    assert not engine.has_unfinished_requests()
    engine.shutdown()


def test_metrics_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RAPID_LLM_METRICS", "0")
    engine = _build_engine([[_WORD], [_EOS]])
    engine.add_request("hi")
    engine.step()
    engine.step()

    assert not engine.metrics.enabled
    assert engine.metrics.render_prometheus() == "\n"
    engine.shutdown()


# --------------------------------------------------------------------------- #
# O5 speculative decoding: the verify pass anchors on the last generated token
# --------------------------------------------------------------------------- #
class _FakeProposer:
    """Returns scripted draft lists, one per propose() call (last repeats)."""

    def __init__(self, drafts: list[list[int]]) -> None:
        self._drafts = drafts
        self._calls = 0

    def propose(self, token_ids: list[int]) -> list[int]:
        d = self._drafts[min(self._calls, len(self._drafts) - 1)]
        self._calls += 1
        return list(d)


class _SpecVerifyExecutor(_ScriptedExecutor):
    """Adds execute_verify: returns logits whose per-row argmax the test pins.

    ``verify_logits`` is one entry per execute_verify call: a list of token
    ids, one per stretch row (anchor + drafts, concatenated across requests).
    The executor builds a ``[rows, vocab]`` tensor where row ``i`` is a
    one-hot at the scripted id, so ``_speculate_verify``'s per-row argmax reads
    exactly what the test wrote. Sampled tokens are empty: the verify path
    derives its bonus from the argmax, never a sampled row.
    """

    def __init__(self, rows: list[list[int]], verify_logits: list[list[int]]) -> None:
        super().__init__(rows)
        self._verify_logits = verify_logits
        self._verify_calls = 0
        self.verify_called = False

    def execute_verify(self, plan):
        self.verify_called = True
        scripted = self._verify_logits[min(self._verify_calls, len(self._verify_logits) - 1)]
        self._verify_calls += 1
        vocab = max(scripted) + 1 if scripted else 1
        logits = torch.zeros(len(scripted), vocab)
        for i, tid in enumerate(scripted):
            logits[i, tid] = 1.0
        return torch.empty(0, dtype=torch.long), None, logits


def _build_spec_engine(rows, verify_logits, *, drafts, max_seq_len=64, **config):
    fake = SimpleNamespace(
        model_runner=SimpleNamespace(
            spec=SimpleNamespace(is_multimodal=False),
            config=SimpleNamespace(kv_cache_torch_dtype=None),
        ),
        device="cpu",
        tokenizer=_FakeTokenizer(),
        stop_token_ids={_EOS},
        max_seq_len=max_seq_len,
    )
    engine = ContinuousBatchingEngine(
        fake,
        SchedulerConfig(max_seq_len=max_seq_len, max_num_seqs=4, **config),
        executor=_SpecVerifyExecutor(rows, verify_logits),
    )
    engine._speculate = True
    engine._proposer = _FakeProposer(drafts)
    return engine


def test_spec_verify_accepts_all_drafts_and_emits_one_bonus():
    """Anchored verify: logits[j] checks draft[j]; all match → bonus at the end.

    The stretch is [anchor, d0, d1]; row 0's logits verify d0, row 1's verify
    d1, and row 2 (the prediction after the last draft) is the bonus. Accepted
    drafts and the bonus all flow through ``_harvest`` once, so
    ``output_token_ids`` holds exactly [prefill, d0, d1, bonus] with no
    duplication — the double-harvest bug would append the bonus twice.
    """
    engine = _build_spec_engine(
        rows=[[_WORD]],  # prefill: one ordinary token
        verify_logits=[[200, 201, 300]],  # anchor→200, d0→201, d1→300(bonus)
        drafts=[[200, 201]],
    )
    request = engine.add_request("hi")

    engine.step()  # prefill → _WORD
    advanced = engine.step()  # spec verify: accept 200, 201; bonus 300

    assert request.output_token_ids == [_WORD, 200, 201, 300]
    assert request.finish_reason is None
    assert [r.request_id for r in advanced] == [request.request_id]  # one chunk, no dup
    engine.shutdown()


def test_spec_verify_rejects_at_first_mismatch_and_samples_bonus():
    """A mismatch at draft j keeps 0..j-1; the bonus is the model's correction.

    draft1 is 201 but the model's argmax at row 1 is 999 — the prediction after
    draft0. So accepted=1 and the bonus is 999, the token the model wanted
    instead of draft1.
    """
    engine = _build_spec_engine(
        rows=[[_WORD]],
        verify_logits=[[200, 999, 300]],  # d0✓, d1✗(999≠201), bonus=row[1]=999
        drafts=[[200, 201]],
    )
    request = engine.add_request("hi")

    engine.step()  # prefill
    engine.step()  # spec verify

    assert request.output_token_ids == [_WORD, 200, 999]
    engine.shutdown()


def test_spec_verify_checks_the_first_draft_not_just_the_second():
    """Regression: the old verifier compared logits[j] to draft[j+1], skipping
    draft[0] entirely. The anchored stretch puts draft[0] at logits[0], so a
    mismatch there rejects immediately and the bonus is logits[0]'s argmax.
    """
    engine = _build_spec_engine(
        rows=[[_WORD]],
        verify_logits=[[999, 201, 300]],  # d0✗(999≠200), accepted=0, bonus=999
        drafts=[[200, 201]],
    )
    request = engine.add_request("hi")

    engine.step()  # prefill
    engine.step()  # spec verify: reject draft0, bonus=999

    assert request.output_token_ids == [_WORD, 999]
    engine.shutdown()


def test_spec_verify_stops_at_an_eos_draft_mid_chain():
    """An eos draft retires the request; later drafts in the chain are skipped.

    draft1 is the eos token: ``_harvest`` finishes the request there, and the
    ``is_finished`` guard drops draft2 and the bonus — nothing after a stop is
    output, and the eos itself is punctuation, not output.
    """
    engine = _build_spec_engine(
        rows=[[_WORD]],
        verify_logits=[[200, _EOS, 201, 300]],  # d0✓, d1=eos✓, d2✓, bonus=300
        drafts=[[200, _EOS, 201]],
    )
    request = engine.add_request("hi")

    engine.step()  # prefill
    engine.step()  # spec verify

    assert request.output_token_ids == [_WORD, 200]  # eos not counted as output
    assert request.finish_reason == "eos"
    engine.shutdown()


def test_spec_verify_caps_at_max_new_tokens_mid_chain():
    """A length finish mid-chain retires the request; surplus drafts drop.

    ``max_gen_len=2``: prefill's token is one, so one more fits. The first
    accepted draft fills the cap and finishes the request; the remaining draft
    and the bonus are skipped by the ``is_finished`` guard.
    """
    engine = _build_spec_engine(
        rows=[[_WORD]],
        verify_logits=[[200, 201, 300]],  # all would match
        drafts=[[200, 201]],
    )
    request = engine.add_request("hi", SamplingParams(max_gen_len=2))

    engine.step()  # prefill → _WORD (1 output token)
    engine.step()  # spec verify: 200 fills the cap, 201+300 dropped

    assert request.output_token_ids == [_WORD, 200]
    assert request.finish_reason == "length"
    engine.shutdown()


def test_spec_verify_routes_logprob_requests_to_decode():
    """A request asking for output logprobs keeps the decode path.

    The verify pass returns raw logits, not the per-token records the sampler's
    path produces, so a logprob request's records would skip every draft. It
    is routed to ``remaining`` and decodes normally instead.
    """
    engine = _build_spec_engine(
        rows=[[_WORD], [_EOS]],  # prefill, then normal decode (the logprob req)
        verify_logits=[[999]],  # would be called only if spec ran
        drafts=[[200, 201]],
    )
    request = engine.add_request("hi", SamplingParams(logprobs=1))

    engine.step()  # prefill
    engine.step()  # decode (not spec): logprobs request kept out of verify

    assert not engine._executor.verify_called  # execute_verify never ran
    assert request.finish_reason == "eos"
    engine.shutdown()


def test_spec_verify_falls_back_when_no_drafts():
    """No ngram match → no drafts → the request decodes normally."""
    engine = _build_spec_engine(
        rows=[[_WORD], [_EOS]],
        verify_logits=[[999]],
        drafts=[[]],  # proposer finds nothing
    )
    request = engine.add_request("hi")

    engine.step()  # prefill
    engine.step()  # no drafts → decode: _EOS

    assert not engine._executor.verify_called
    assert request.finish_reason == "eos"
    engine.shutdown()
