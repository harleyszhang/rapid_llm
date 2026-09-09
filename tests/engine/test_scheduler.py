"""Tests for the continuous-batching admission policy.

Requests are plain objects driven through the :class:`Scheduler` on
CPU: admission, chunked prefill, decode budget, slot accounting and
preemption — one test class per concern.

Usage:
    pytest tests/engine/test_scheduler.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import (
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)

_MAX_SEQ_LEN = 128


def make_request(request_id: str, prompt_len: int = 4, **params) -> Request:
    """A request whose prompt is ``prompt_len`` throwaway token ids."""
    return Request(
        request_id=request_id,
        prompt="x" * prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        params=SamplingParams(**params),
    )


def make_request_with_tokens(request_id: str, token_ids: list[int], **params) -> Request:
    """A request with explicit token ids (for prefix-cache tests)."""
    return Request(
        request_id=request_id,
        prompt="x" * len(token_ids),
        prompt_token_ids=token_ids,
        params=SamplingParams(**params),
    )


def _decode_once(output: SchedulerOutput) -> None:
    """Give every decoding request one token (chunks advance in schedule)."""
    for r in output.decode:
        r.output_token_ids.append(999)


@pytest.fixture
def scheduler() -> Scheduler:
    """Four slots and a small token budget, so limits are reachable in one step."""
    return Scheduler(
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_num_batched_tokens=64),
        num_slots=4,
    )


# --------------------------------------------------------------------------- #
# 1. Request admission
# --------------------------------------------------------------------------- #
class TestRequestAdmission:
    """Generation cap resolution and unservable prompt refusal."""

    def test_generation_cap_is_resolved_against_the_context_window(self, scheduler):
        """``max_gen_len=None`` means "fill the window", not "unbounded"."""
        request = make_request("a", prompt_len=100)
        scheduler.add_request(request)
        assert request.max_new_tokens == _MAX_SEQ_LEN - 100

    def test_generation_cap_honours_an_explicit_request(self, scheduler):
        request = make_request("a", prompt_len=10, max_gen_len=7)
        scheduler.add_request(request)
        assert request.max_new_tokens == 7

    def test_generation_cap_is_clamped_by_the_window(self, scheduler):
        """A request asking for more than fits gets what fits, not a later overflow."""
        request = make_request("a", prompt_len=120, max_gen_len=1000)
        scheduler.add_request(request)
        assert request.max_new_tokens == _MAX_SEQ_LEN - 120

    @pytest.mark.parametrize(
        ("prompt_len", "why"),
        [(0, "empty prompt"), (_MAX_SEQ_LEN, "no room left to generate")],
    )
    def test_unservable_prompts_are_refused_at_submission(self, scheduler, prompt_len, why):
        """Refusing here is what lets the engine assume every admitted request fits."""
        with pytest.raises(ValueError):
            scheduler.add_request(make_request("a", prompt_len=prompt_len))
        assert scheduler.num_waiting == 0, why

    def test_max_num_seqs_cannot_exceed_the_available_slots(self):
        """Concurrency is bounded by cache slots, whatever the caller asked for."""
        sched = Scheduler(
            SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=64),
            num_slots=3,
        )
        assert sched.max_num_seqs == 3

    def test_max_num_seqs_can_exceed_slots_with_preemption(self):
        """enable_preemption lets max_num_seqs go beyond slot count."""
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=10,
                enable_preemption=True,
            ),
            num_slots=3,
        )
        assert sched.max_num_seqs == 10

    def test_zero_slots_is_rejected(self):
        with pytest.raises(ValueError):
            Scheduler(SchedulerConfig(max_seq_len=_MAX_SEQ_LEN), num_slots=0)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("max_seq_len", 1),
            ("max_num_seqs", 0),
            ("max_num_batched_tokens", 0),
            ("max_chunk_size", -1),
            ("prefix_cache_blocks", 0),
        ],
    )
    def test_invalid_scheduler_limits_are_rejected(self, field, value):
        with pytest.raises(ValueError, match=field):
            SchedulerConfig(**{field: value})

    def test_duplicate_live_request_ids_are_rejected(self, scheduler):
        scheduler.add_request(make_request("same"))
        with pytest.raises(ValueError, match="already active"):
            scheduler.add_request(make_request("same"))

    def test_finished_request_id_can_be_reused(self, scheduler):
        first = make_request("same")
        scheduler.add_request(first)
        scheduler.schedule()
        scheduler.finish(first, "eos")

        second = make_request("same")
        scheduler.add_request(second)
        assert scheduler.waiting == [second]


# --------------------------------------------------------------------------- #
# 2. Prefill scheduling
# --------------------------------------------------------------------------- #
class TestPrefillScheduling:
    """FCFS order, token budget gating, and prefill+decode coexistence."""

    def test_prefill_admits_in_arrival_order(self, scheduler):
        for name in "abc":
            scheduler.add_request(make_request(name))
        assert [r.request_id for r in scheduler.schedule().prefill] == ["a", "b", "c"]

    def test_concurrency_is_capped_and_the_rest_stay_queued(self, scheduler):
        for index in range(6):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        assert len(admitted) == 4  # max_num_seqs
        assert scheduler.num_waiting == 2
        assert all(r.status is RequestStatus.WAITING for r in scheduler.waiting)

    def test_token_budget_bounds_the_padded_prefill_grid(self, scheduler):
        """Prefill pads to the longest prompt, so the budget is measured padded.

        Two 40-token prompts cost 80 padded token-slots against a 64 budget, so the
        second waits even though three slots are still free.
        """
        scheduler.add_request(make_request("long-a", prompt_len=40))
        scheduler.add_request(make_request("long-b", prompt_len=40))
        assert [r.request_id for r in scheduler.schedule().prefill] == ["long-a"]
        assert scheduler.num_waiting == 1

    def test_a_prompt_bigger_than_the_budget_is_chunked_to_fit(self, scheduler):
        """Chunked prefill makes forward progress without violating the budget."""
        scheduler.add_request(make_request("huge", prompt_len=100))
        scheduler.add_request(make_request("small", prompt_len=4))
        output = scheduler.schedule()
        assert [r.request_id for r in output.prefill] == ["huge"]
        assert output.prefill_chunk_lens == [64]
        assert scheduler.num_waiting == 1

    def test_prefill_takes_priority_over_decode(self, scheduler):
        """A queued arrival gets prefilled alongside running decode (chunked prefill)."""
        scheduler.add_request(make_request("a"))
        scheduler.schedule()  # admits and prefills `a`
        scheduler.add_request(make_request("b"))
        output = scheduler.schedule()
        assert [r.request_id for r in output.prefill] == ["b"]
        assert any(r.request_id == "a" for r in output.decode)  # coexists


# --------------------------------------------------------------------------- #
# 3. Chunked prefill
# --------------------------------------------------------------------------- #
class TestChunkedPrefill:
    """Long prompts split into chunks; decode interleaves with prefill chunks."""

    @pytest.fixture
    def chunked_scheduler(self) -> Scheduler:
        return Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                max_chunk_size=32,
            ),
            num_slots=4,
        )

    def test_long_prompt_is_split_into_chunks(self, chunked_scheduler):
        """A 100-token prompt with max_chunk_size=32 takes 4 chunks."""
        sched = chunked_scheduler
        sched.add_request(make_request("long", prompt_len=100))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [32]
        assert sched.num_running == 1  # admitted but still chunking

    def test_chunk_lens_are_correct_per_step(self, chunked_scheduler):
        """Each step processes at most max_chunk_size tokens of the remaining prompt."""
        sched = chunked_scheduler
        sched.add_request(make_request("long", prompt_len=100))
        expected = [32, 32, 32, 4]  # 100 = 32*3 + 4
        actual = []
        for _ in range(4):
            out = sched.schedule()  # commits the chunk on the request itself
            actual += out.prefill_chunk_lens
        assert actual == expected

    def test_final_chunk_moves_request_to_decode(self, chunked_scheduler):
        """After the last chunk commits, the request leaves the prefill group."""
        sched = chunked_scheduler
        sched.add_request(make_request("long", prompt_len=100))
        for _ in range(4):
            sched.schedule()
        # After 4 chunks (32*3+4=100), the request is no longer chunking.
        out = sched.schedule()
        assert not any(r.request_id == "long" for r in out.prefill)
        assert any(r.request_id == "long" for r in out.decode)

    def test_decode_runs_alongside_prefill_chunks(self, chunked_scheduler):
        """Chunked prefill does not block already-decoding requests."""
        sched = chunked_scheduler
        sched.add_request(make_request("short-a", prompt_len=10))
        sched.schedule()  # prefill short-a completely
        # short-a is now decoding; long-b starts chunking.
        sched.add_request(make_request("long-b", prompt_len=100))
        out = sched.schedule()
        assert any(r.request_id == "short-a" for r in out.decode)
        assert any(r.request_id == "long-b" for r in out.prefill)

    def test_max_chunk_size_zero_disables_chunking(self):
        """max_chunk_size=0 means no chunking — the full prompt is one prefill."""
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                max_chunk_size=0,
            ),
            num_slots=4,
        )
        sched.add_request(make_request("long", prompt_len=100))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [100]

    def test_chunked_prefill_is_on_by_default(self):
        """vLLM's default, and ours: a fresh config chunks."""
        config = SchedulerConfig(max_seq_len=_MAX_SEQ_LEN)
        assert config.enable_chunked_prefill
        assert config.max_chunk_size > 0

    def test_disabling_the_flag_prefills_in_one_pass(self):
        """``enable_chunked_prefill=False`` ignores ``max_chunk_size``."""
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                enable_chunked_prefill=False,
                max_chunk_size=32,
            ),
            num_slots=4,
        )
        sched.add_request(make_request("long", prompt_len=100))
        assert sched.schedule().prefill_chunk_lens == [100]

    def test_max_chunk_size_zero_folds_into_the_flag(self):
        """The older spelling of "off" leaves one field to read."""
        config = SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_chunk_size=0)
        assert not config.enable_chunked_prefill

    def test_disabling_needs_a_budget_that_holds_the_longest_prompt(self):
        """Without chunking a whole prompt must fit one pass, as vLLM checks."""
        with pytest.raises(ValueError, match="chunked prefill disabled"):
            SchedulerConfig(
                max_seq_len=4096,
                max_num_batched_tokens=512,
                enable_chunked_prefill=False,
            )


# --------------------------------------------------------------------------- #
# 4. Decode scheduling
# --------------------------------------------------------------------------- #
class TestDecodeScheduling:
    """Full decode coverage and idle no-op."""

    def test_decode_covers_every_running_request(self, scheduler):
        for name in "abc":
            scheduler.add_request(make_request(name))
        scheduler.schedule()  # prefill all three
        output = scheduler.schedule()
        assert output.prefill == []
        assert [r.request_id for r in output.decode] == ["a", "b", "c"]

    def test_a_request_finishing_prefill_joins_decode_the_next_step(self, scheduler):
        """The step that completes a prompt samples it in the prefill pass.

        Decode gets it only from the following step: the engine executes each
        request in exactly one pass per step, so admitting a just-prefilled
        request into the same step's decode batch would run its slot twice.
        """
        scheduler.add_request(make_request("a"))
        scheduler.add_request(make_request("b"))
        first = scheduler.schedule()  # both prompts fit one chunk, both finish
        assert [r.request_id for r in first.prefill] == ["a", "b"]
        assert first.decode == []
        second = scheduler.schedule()
        assert second.prefill == []
        assert [r.request_id for r in second.decode] == ["a", "b"]

    def test_a_mix_of_old_decoders_and_new_prefills_stay_pass_disjoint(self, scheduler):
        """Running decodes and this step's prefills never share a request."""
        scheduler.add_request(make_request("old"))
        scheduler.schedule()  # old is now decoding
        scheduler.add_request(make_request("new"))
        out = scheduler.schedule()  # prefills new while old decodes
        assert [r.request_id for r in out.prefill] == ["new"]
        assert [r.request_id for r in out.decode] == ["old"]
        assert not ({id(r) for r in out.prefill} & {id(r) for r in out.decode})

    def test_an_idle_scheduler_schedules_nothing(self, scheduler):
        assert scheduler.schedule().is_empty
        assert not scheduler.has_unfinished_requests()


# --------------------------------------------------------------------------- #
# 5. Slot accounting
# --------------------------------------------------------------------------- #
class TestSlotAccounting:
    """Slot conservation: no leaks, no double-issues."""

    def test_running_requests_hold_distinct_slots(self, scheduler):
        for index in range(4):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        slots = [r.slot for r in admitted]
        assert len(set(slots)) == 4, "two requests sharing a slot would share KV"
        assert all(0 <= slot < 4 for slot in slots)
        assert scheduler.num_free_slots == 0

    def test_finishing_returns_the_slot_and_retires_the_request(self, scheduler):
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.schedule()
        slot = request.slot
        scheduler.finish(request, "eos")
        assert request.status is RequestStatus.FINISHED
        assert request.finish_reason == "eos"
        assert request.slot is None
        assert scheduler.num_running == 0
        assert scheduler.num_free_slots == 4, f"slot {slot} leaked"

    def test_finishing_twice_does_not_double_free_the_slot(self, scheduler):
        """A double free would hand one slot to two later requests."""
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.schedule()
        scheduler.finish(request, "eos")
        scheduler.finish(request, "length")
        assert scheduler.num_free_slots == 4
        assert request.finish_reason == "eos", "the first reason is the true one"

    def test_a_freed_slot_is_reused_by_a_queued_request(self, scheduler):
        for index in range(5):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        freed = admitted[1].slot
        scheduler.finish(admitted[1], "eos")
        next_group = scheduler.schedule().prefill
        assert [r.request_id for r in next_group] == ["r4"]
        assert next_group[0].slot == freed

    def test_many_finish_and_admit_rounds_conserve_slots(self, scheduler):
        """The leak this catches only surfaces as a stuck engine much later."""
        for round_index in range(20):
            request = make_request(f"r{round_index}")
            scheduler.add_request(request)
            scheduler.schedule()
            scheduler.finish(request, "eos")
        assert scheduler.num_free_slots == 4
        assert scheduler.num_running == 0
        assert not scheduler.has_unfinished_requests()

    def test_running_requests_are_never_preempted_without_enable(self, scheduler):
        """Without enable_preemption, admitted requests keep their slots."""
        for index in range(6):
            scheduler.add_request(make_request(f"r{index}"))
        admitted = scheduler.schedule().prefill
        held = {r.request_id: r.slot for r in admitted}
        for _ in range(5):
            scheduler.schedule()
        assert {r.request_id: r.slot for r in scheduler.running} == held


# --------------------------------------------------------------------------- #
# 6. Preemption (recompute, opt-in oversubscription)
# --------------------------------------------------------------------------- #
def _oversubscribed(num_slots: int = 2, max_num_seqs: int = 3) -> Scheduler:
    return Scheduler(
        SchedulerConfig(
            max_seq_len=_MAX_SEQ_LEN,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=1 << 20,
            max_chunk_size=0,
            enable_preemption=True,
        ),
        num_slots=num_slots,
    )


class TestPreemption:
    """Opt-in oversubscription via recompute preemption."""

    def test_preemption_disabled_by_default(self):
        """Without enable_preemption the batch stays slot-capped and nothing evicts."""
        sched = Scheduler(
            SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=8),
            num_slots=2,
        )
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        for _ in range(4):
            out = sched.schedule()
            _decode_once(out)
        assert sched.num_preemptions == 0

    def test_oversubscription_admits_beyond_slots(self):
        """max_num_seqs may exceed slots; the extra request runs by preempting."""
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        seen: set[str] = set()
        for _ in range(6):
            out = sched.schedule()
            for r in out.decode:
                seen.add(r.request_id)
            _decode_once(out)
        # All three requests get decode turns despite only two slots.
        assert seen == {"r0", "r1", "r2"}
        assert sched.num_preemptions >= 1

    def test_preempted_request_is_reported_in_output(self):
        """SchedulerOutput.preempted names who was evicted this step."""
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        preempted_ids: list[str] = []
        for _ in range(5):
            out = sched.schedule()
            preempted_ids += [r.request_id for r in out.preempted]
            _decode_once(out)
        assert preempted_ids  # at least one preemption was surfaced

    def test_progress_quantum_prevents_livelock(self):
        """A just-recomputed request is protected until it decodes once.

        Without the quantum, two requests could preempt each other every step and
        neither would ever emit a token.
        """
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        total_tokens = 0
        for _ in range(9):
            out = sched.schedule()
            total_tokens += len(out.decode)
            _decode_once(out)
        # Real forward progress: many tokens produced, not a stalled ping-pong.
        assert total_tokens >= 6

    def test_preemption_fairness_all_requests_progress(self):
        """Under sustained oversubscription, no request is permanently starved."""
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        decode_counts: dict[str, int] = {"r0": 0, "r1": 0, "r2": 0}
        for _ in range(12):
            out = sched.schedule()
            for r in out.decode:
                decode_counts[r.request_id] += 1
            _decode_once(out)
        # Every request got at least one decode turn.
        assert all(count > 0 for count in decode_counts.values())

    def test_block_pressure_preempts_the_youngest_unplanned_decode(self):
        """A future decode row is a valid victim until it receives work."""
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=40,
                max_num_seqs=2,
                max_num_batched_tokens=128,
                max_chunk_size=0,
                prefix_cache_blocks=3,
                enable_preemption=True,
            ),
            num_slots=2,
        )
        requests = [make_request("old", prompt_len=15), make_request("young", prompt_len=15)]
        for request in requests:
            sched.add_request(request)
        sched.schedule()
        for request in requests:
            request.output_token_ids.append(999)

        out = sched.schedule()

        assert out.preempted == [requests[1]]
        assert out.decode == [requests[0]]

    def test_preemption_conserves_requests_and_slots(self):
        """Regression: a preempted victim used to leak out of both queues.

        ``_preempt`` re-queues the victim at the *head* of the waiting deque,
        and the admission loop that follows used to pop that head — the victim —
        instead of the candidate it had already peeked. The victim then sat in
        neither queue (never scheduled again) while the candidate stayed queued
        for a second admission holding two slots. After every step each request
        must live in exactly one queue, and slots must be conserved.
        """
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        requests = [make_request(f"r{i}") for i in range(3)]
        for request in requests:
            sched.add_request(request)

        for _ in range(10):
            _decode_once(sched.schedule())

            for request in requests:
                in_waiting = any(r is request for r in sched.waiting)
                in_running = any(r is request for r in sched.running)
                assert in_waiting != in_running, f"{request.request_id} is in both or neither"

            slots = [r.slot for r in sched.running]
            assert all(slot is not None for slot in slots)
            assert all(r.slot is None for r in sched.waiting)
            assert len(set(slots)) == len(slots), "two requests sharing a slot"
            assert len(slots) + sched.num_free_slots == 2, "slot conservation broken"

        assert sched.num_preemptions >= 1, "scenario never reached the bug path"

    def test_preemption_folds_generated_tokens_into_the_prompt(self):
        """Recompute preemption replays the generated tokens, not just the prompt.

        vLLM v1 semantics: the victim's output moves into its prompt, so
        re-admission re-prefills through it and decoding continues the text the
        caller already saw instead of restarting from the bare prompt and
        diverging from it. The generation cap shrinks by the same count so the
        request's *total* output stays bounded by what it asked for.
        """
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))

        victim = None
        for _ in range(8):
            out = sched.schedule()
            _decode_once(out)
            if out.preempted:
                victim = out.preempted[0]
                break
        assert victim is not None, "scenario never reached a preemption"

        moved = len(victim.prompt_token_ids) - 4
        assert moved >= 1, "victim was preempted before generating anything"
        assert victim.prompt_token_ids[:4] == [0, 1, 2, 3], "original prompt must survive"
        assert victim.prompt_token_ids[4:] == [999] * moved, "output folded in verbatim"
        assert victim.output_token_ids == []
        assert victim.max_new_tokens == _MAX_SEQ_LEN - 4 - moved
        assert victim.status is RequestStatus.WAITING

        # Re-admission prefills through the extended prompt: one chunk covers
        # prompt plus folded tokens, so decode resumes where the text broke off.
        for _ in range(4):
            out = sched.schedule()
            if any(r is victim for r in out.prefill):
                break
            _decode_once(out)
        assert victim.status is RequestStatus.RUNNING
        assert victim.num_computed_tokens == victim.prompt_len


# --------------------------------------------------------------------------- #
# 7. SchedulerOutput field integrity
# --------------------------------------------------------------------------- #
class TestSchedulerOutput:
    """prefill_chunk_lens, preempted, and prefill+decode coexistence."""

    def test_chunk_lens_match_remaining_tokens(self):
        sched = Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=1 << 20,
                max_chunk_size=50,
            ),
            num_slots=4,
        )
        sched.add_request(make_request("a", prompt_len=120))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [50]  # first chunk = 50 of 120
        out = sched.schedule()
        assert out.prefill_chunk_lens == [50]  # second chunk
        out = sched.schedule()
        assert out.prefill_chunk_lens == [20]  # last chunk = 20

    def test_is_empty_when_idle(self, scheduler):
        assert scheduler.schedule().is_empty

    def test_prefill_and_decode_coexist(self, scheduler):
        """v0.7: prefill and decode can be non-empty in the same step."""
        scheduler.add_request(make_request("a"))
        scheduler.schedule()  # a is now running (decoding)
        scheduler.add_request(make_request("b"))
        out = scheduler.schedule()
        assert out.prefill and out.decode  # both non-empty

    def test_preempted_field_is_populated_on_eviction(self):
        sched = _oversubscribed(num_slots=2, max_num_seqs=3)
        for i in range(3):
            sched.add_request(make_request(f"r{i}"))
        found_preemption = False
        for _ in range(6):
            out = sched.schedule()
            if out.preempted:
                found_preemption = True
                assert all(r.status is RequestStatus.WAITING for r in out.preempted)
                break
            _decode_once(out)
        assert found_preemption, "expected at least one preemption step"


# --------------------------------------------------------------------------- #
# 8. Abort
# --------------------------------------------------------------------------- #
class TestAbort:
    def test_aborting_a_running_request_frees_its_slot(self, scheduler):
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.schedule()
        assert scheduler.abort("a") is request
        assert request.finish_reason == "abort"
        assert scheduler.num_free_slots == 4

    def test_aborting_a_queued_request_dequeues_it(self, scheduler):
        scheduler.add_request(make_request("a"))
        scheduler.add_request(make_request("b"))
        aborted = scheduler.abort("b")
        assert aborted is not None and aborted.finish_reason == "abort"
        assert scheduler.num_waiting == 1
        assert [r.request_id for r in scheduler.schedule().prefill] == ["a"]

    def test_aborting_an_unknown_id_is_a_no_op(self, scheduler):
        assert scheduler.abort("nope") is None

    def test_aborting_a_queued_request_records_finish_time(self, scheduler):
        scheduler.add_request(make_request("a"))
        request = scheduler.abort("a")
        assert request is not None and request.finish_time is not None

    def test_finish_can_retire_a_waiting_request(self, scheduler):
        request = make_request("a")
        scheduler.add_request(request)
        scheduler.finish(request, "abort")
        assert not scheduler.has_unfinished_requests()
        assert scheduler.abort("a") is None


# --------------------------------------------------------------------------- #
# 9. Request bookkeeping
# --------------------------------------------------------------------------- #
class TestRequestBookkeeping:
    def test_seq_len_counts_prompt_plus_generated(self):
        request = make_request("a", prompt_len=5)
        request.output_token_ids.extend([1, 2, 3])
        assert request.seq_len == 8

    def test_has_room_tracks_the_generation_cap(self, scheduler):
        request = make_request("a", max_gen_len=2)
        scheduler.add_request(request)
        assert request.has_room
        request.output_token_ids.append(1)
        assert request.has_room
        request.output_token_ids.append(2)
        assert not request.has_room

    def test_is_finished_flag(self):
        request = make_request("a")
        assert not request.is_finished
        request.status = RequestStatus.FINISHED
        assert request.is_finished


# --------------------------------------------------------------------------- #
# 10. Prefix cache integration
# --------------------------------------------------------------------------- #
class TestPrefixCacheIntegration:
    """Prefix cache wired into the scheduler's admission/finish/preempt path."""

    def _sched(
        self,
        *,
        enable_prefix_cache: bool = True,
        enable_preemption: bool = False,
        max_chunk_size: int = 0,
        max_num_seqs: int = 8,
        num_slots: int = 8,
    ) -> Scheduler:
        config = SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=65536,
            max_chunk_size=max_chunk_size,
            enable_prefix_cache=enable_prefix_cache,
            enable_preemption=enable_preemption,
        )
        return Scheduler(config, num_slots=num_slots)

    def test_disabled_cache_means_zero_cached_tokens(self):
        sched = self._sched(enable_prefix_cache=False)
        shared = list(range(64))
        sched.add_request(make_request_with_tokens("a", shared))
        sched.schedule()
        sched.add_request(make_request_with_tokens("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_second_request_hits_shared_prefix(self):
        sched = self._sched()
        shared = list(range(64))  # 4 blocks of 16
        sched.add_request(make_request_with_tokens("a", shared))
        out_a = sched.schedule()
        a = next(r for r in out_a.prefill if r.request_id == "a")
        assert a.num_cached_tokens == 0  # first request: cold

        sched.add_request(make_request_with_tokens("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        assert b.num_cached_tokens >= 48  # at least 3 of 4 blocks
        assert b.num_cached_tokens < b.prompt_len  # never skips the whole prompt

    def test_divergent_prompts_get_zero_cached(self):
        sched = self._sched()
        sched.add_request(make_request_with_tokens("a", list(range(64))))
        sched.schedule()
        sched.add_request(make_request_with_tokens("b", list(range(1000, 1064))))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")
        assert b.num_cached_tokens == 0

    def test_hit_rate_grows_with_shared_requests(self):
        sched = self._sched()
        shared = list(range(64))
        rates: list[float] = []
        for name in ("a", "b", "c"):
            sched.add_request(make_request_with_tokens(name, shared))
            sched.schedule()
            rates.append(sched.prefix_cache_hit_rate)
        assert rates[0] == 0.0  # first request: cold
        assert rates[1] > 0.0  # second: hit
        assert rates[2] >= rates[1]  # third: monotonically non-decreasing

    def test_finish_releases_prefix_but_it_stays_cached(self):
        """After the first request finishes, its prefix survives (LRU persistence)."""
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(make_request_with_tokens("a", shared))
        out = sched.schedule()
        a = out.prefill[0]
        sched.finish(a, "eos")
        # req b arrives after a finished — should still hit the shared prefix.
        sched.add_request(make_request_with_tokens("b", shared))
        out_b = sched.schedule()
        b = next(r for r in out_b.prefill if r.request_id == "b")
        assert b.num_cached_tokens >= 48

    def test_preempted_request_resets_cached_tokens(self):
        """Preemption clears num_cached_tokens; recompute starts from scratch."""
        sched = self._sched(
            enable_preemption=True,
            max_num_seqs=3,
            num_slots=2,
        )
        shared = list(range(64))
        for i in range(3):
            sched.add_request(make_request_with_tokens(f"r{i}", shared))
        for _ in range(6):
            out = sched.schedule()
            for r in out.decode:
                r.output_token_ids.append(999)
            if out.preempted:
                for p in out.preempted:
                    assert p.num_cached_tokens == 0
                return
        pytest.fail("expected at least one preemption within 6 steps")

    def test_preemption_releases_prefix_cache_references(self):
        """Regression: preemption used to leak the victim's prefix references.

        Adopting a hit takes one reference per shared block and only ``free``
        returns it, so a preempt cycle that skipped the release inflated every
        shared block's ref_cnt once per cycle — and referenced blocks are never
        eviction candidates, so capacity drained one cycle at a time.

        The invariant is one reference per (running request, block it holds): a
        preempted victim holds nothing, so a reference it failed to return shows
        up here as a surplus.
        """
        sched = self._sched(
            enable_preemption=True,
            max_num_seqs=3,
            num_slots=2,
        )
        shared = list(range(64))  # exactly four 16-token blocks
        for i in range(3):
            sched.add_request(make_request_with_tokens(f"r{i}", shared))

        cache = sched._prefix_cache
        for _ in range(10):
            out = sched.schedule()
            for r in out.decode:
                r.output_token_ids.append(999)
            held = sum(
                len(blocks)
                for r in sched._requests.values()
                if r.slot is not None
                for blocks in cache.block_ids(r.request_id)
            )
            total = sum(block.ref_cnt for block in cache.pool.blocks) - 1  # null block
            assert total == held, f"{total} references for {held} held blocks"
        assert sched.num_preemptions >= 1, "scenario never reached the bug path"

    def test_a_shared_prefix_is_the_same_physical_blocks(self):
        """The baseline that replaced copy segments: reuse is shared rows.

        Two requests over the same prompt must hold *identical* block ids for
        the hit prefix — that is what makes reuse cost a reference count instead
        of a K/V copy.
        """
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(make_request_with_tokens("a", shared))
        sched.schedule()
        sched.add_request(make_request_with_tokens("b", shared))
        out = sched.schedule()
        b = next(r for r in out.prefill if r.request_id == "b")

        cache = sched._prefix_cache
        num_shared = b.num_cached_tokens // PREFIX_CACHE_BLOCK_SIZE
        assert num_shared >= 3
        assert cache.block_ids("b")[0][:num_shared] == cache.block_ids("a")[0][:num_shared]
        # b's own tail is its own: writing it into a's rows would corrupt a.
        assert cache.block_ids("b")[0][num_shared] not in cache.block_ids("a")[0]

    def test_a_hit_costs_only_the_uncached_tail_of_the_pool(self):
        sched = self._sched()
        shared = list(range(64))
        sched.add_request(make_request_with_tokens("a", shared))
        sched.schedule()
        sched.schedule()  # a's blocks committed

        cache = sched._prefix_cache
        before = cache.num_free_blocks
        sched.add_request(make_request_with_tokens("b", shared))
        sched.schedule()
        # Only b's own last block was drawn; three were shared.
        assert before - cache.num_free_blocks == 1


# --------------------------------------------------------------------------- #
# 7. Decode window (O9)
# --------------------------------------------------------------------------- #
class TestDecodeWindow:
    """Fresh prefills wait up to N pure-decode steps before interrupting them."""

    @pytest.fixture
    def windowed(self) -> Scheduler:
        return Scheduler(
            SchedulerConfig(
                max_seq_len=_MAX_SEQ_LEN,
                max_num_seqs=4,
                max_num_batched_tokens=128,
                max_chunk_size=32,
                decode_window_steps=2,
            ),
            num_slots=4,
        )

    def test_fresh_prompt_waits_exactly_the_window(self, windowed):
        """A prompt arriving mid-decode interrupts after 2 pure-decode steps."""
        sched = windowed
        sched.add_request(make_request("a", prompt_len=4))
        assert sched.schedule().prefill  # a's prefill step

        sched.add_request(make_request("b", prompt_len=4))
        for _ in range(2):  # the window: pure-decode steps, b waits
            out = sched.schedule()
            assert out.decode and not out.prefill
            _decode_once(out)
        out = sched.schedule()  # window exhausted: b may interrupt
        assert [r.request_id for r in out.prefill] == ["b"]

    def test_default_admits_immediately(self, scheduler):
        """decode_window_steps=0 keeps the honest baseline: no deferral."""
        sched = scheduler
        sched.add_request(make_request("a", prompt_len=4))
        sched.schedule()
        sched.add_request(make_request("b", prompt_len=4))
        out = sched.schedule()
        assert [r.request_id for r in out.prefill] == ["b"]
        assert out.decode

    def test_no_decode_in_flight_never_waits(self, windowed):
        """With nothing mid-generation, waiting would only delay the first token."""
        sched = windowed
        sched.add_request(make_request("a", prompt_len=4))
        assert sched.schedule().prefill  # admitted in its first step

    def test_resumed_chunk_step_admits_without_waiting(self, windowed):
        """A step already carrying a chunked prefill has paid the interruption."""
        sched = windowed
        sched.add_request(make_request("long", prompt_len=100))
        out = sched.schedule()
        assert out.prefill_chunk_lens == [32]  # capped by the chunk size
        sched.add_request(make_request("b", prompt_len=4))
        out = sched.schedule()  # long's next chunk resumes; b joins immediately
        assert [r.request_id for r in out.prefill] == ["long", "b"]

    def test_negative_window_is_rejected(self):
        with pytest.raises(ValueError):
            SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, decode_window_steps=-1)
