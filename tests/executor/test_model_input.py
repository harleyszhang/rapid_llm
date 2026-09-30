"""The engine -> executor contract: what a step ships, and what it leaves out.

:class:`~rapid_llm.executor.worker.ModelInput` invariants — flattened
rows, cached sampling tensors that detect mutated params, first-chunk
vs resumed-chunk vs decode plans — pinned before any GPU work.

Usage:
    pytest tests/executor/test_model_input.py
"""

from __future__ import annotations

import pickle

import pytest

from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Request, _chunk_work, _decode_work, _prefill_work
from rapid_llm.executor.worker import ModelInput, ModelWorker, PassKind

GREEDY = SamplingParams(temperature=0.0, max_gen_len=8, repetition_penalty=1.0)
PENALISED = SamplingParams(temperature=0.8, repetition_penalty=1.1)


def make_request(
    request_id: str,
    prompt_len: int,
    slot: int,
    *,
    computed: int = 0,
    generated: int = 0,
    params: SamplingParams = GREEDY,
) -> Request:
    """A request already in the state a scheduler would have left it in.

    Token ids are slot-tagged (``slot * 1000 + position``) so a test can tell whose
    tokens ended up in a plan, which is the failure mode that matters: two
    requests' chunks concatenated in the wrong order still have the right *length*.
    """
    request = Request(
        request_id=request_id,
        prompt="x" * prompt_len,
        prompt_token_ids=[slot * 1000 + index for index in range(prompt_len)],
        params=params,
        max_new_tokens=16,
    )
    request.slot = slot
    request.num_computed_tokens = computed or prompt_len
    request.output_token_ids = [900_000 + index for index in range(generated)]
    return request


def a_plan(**overrides) -> ModelInput:
    """A minimal valid one-sequence decode plan, for validation tests to break."""
    fields = {
        "kind": PassKind.DECODE,
        "slots": (0,),
        "seq_starts": (4,),
        "seq_lens": (5,),
        "tokens": (7,),
        "sampling": (GREEDY,),
        "sampled": (0,),
        "gen_counts": (1,),
    }
    return ModelInput(**(fields | overrides))


class TestModelInputInvariants:
    """A plan that cannot describe a pass must not be constructible.

    The worker derives every tensor from these tuples, so an inconsistent plan
    does not fail where it was built — it fails several frames later as a shape
    mismatch inside attention, or worse, not at all.
    """

    def test_a_consistent_plan_is_accepted(self):
        assert a_plan().chunk_lens == (1,)

    def test_chunk_lens_are_derived_not_carried(self):
        plan = a_plan(slots=(0, 1), seq_starts=(0, 10), seq_lens=(6, 14), tokens=tuple(range(10)))
        assert plan.chunk_lens == (6, 4)

    def test_an_empty_plan_is_refused(self):
        with pytest.raises(ValueError, match="at least one sequence"):
            a_plan(
                slots=(),
                seq_starts=(),
                seq_lens=(),
                tokens=(),
                sampling=(),
                sampled=(),
                gen_counts=(),
            )

    def test_sequence_fields_must_agree(self):
        with pytest.raises(ValueError, match="same sequences"):
            a_plan(slots=(0, 1))

    def test_token_count_must_match_the_cache_rows_claimed(self):
        with pytest.raises(ValueError, match="tokens for"):
            a_plan(tokens=(7, 8))

    def test_sampling_fields_must_agree(self):
        with pytest.raises(ValueError, match="same rows"):
            a_plan(gen_counts=(1, 1))

    def test_a_plan_survives_pickle(self):
        """Tensor parallelism broadcasts plans as objects; equality must hold."""
        plan = a_plan()
        assert pickle.loads(pickle.dumps(plan)) == plan


def test_sampling_tensor_cache_detects_mutated_params() -> None:
    """A mutable public SamplingParams object must not leave stale device knobs."""
    worker = object.__new__(ModelWorker)
    worker._device = "cpu"
    worker._sampling_key = None
    worker._sampling = None
    params = SamplingParams(temperature=0.8)

    first = worker._batched_sampling((params,))
    params.temperature = 0.0
    second = worker._batched_sampling((params,))

    assert first is not second
    assert not first.all_greedy
    assert second.all_greedy


class TestFirstChunkPlans:
    """Prompts entering the cache for the first time: the prefill grid."""

    def test_a_whole_prompt_is_planned_from_row_zero(self):
        request = make_request("a", prompt_len=5, slot=2)

        plan, requests, _ = _chunk_work(PassKind.PREFILL, [(request, 5)])

        assert plan.kind is PassKind.PREFILL
        assert plan.slots == (2,)
        assert plan.seq_starts == (0,)
        assert plan.seq_lens == (5,)
        assert plan.tokens == tuple(request.prompt_token_ids)
        assert plan.sampled == (0,)
        assert plan.gen_counts == (0,)
        assert requests == [request]

    def test_tokens_are_concatenated_in_row_order(self):
        first = make_request("a", prompt_len=3, slot=0)
        second = make_request("b", prompt_len=2, slot=1)

        plan, _, _ = _chunk_work(PassKind.PREFILL, [(first, 3), (second, 2)])

        assert plan.tokens == (0, 1, 2, 1000, 1001)
        assert plan.chunk_lens == (3, 2)

    def test_an_unfinished_chunk_asks_for_no_token(self):
        """A capped chunk still runs — its K/V has to land — but samples nothing."""
        request = make_request("a", prompt_len=100, slot=0, computed=64)

        plan, requests, _ = _chunk_work(PassKind.PREFILL, [(request, 64)])

        assert plan.seq_starts == (0,) and plan.seq_lens == (64,)
        assert len(plan.tokens) == 64
        assert plan.sampled == ()
        assert plan.sampling == () and plan.gen_counts == ()
        assert requests == []

    def test_a_mixed_grid_names_the_completed_row(self):
        """The bug this pairing exists for: sampled rows are a subset, in row order.

        A short prompt admitted beside a chunk-capped one shares its grid, and only
        the short one has a token to draw. Naming it by row index rather than by
        position in the completed list is what keeps its logits its own.
        """
        capped = make_request("long", prompt_len=100, slot=0, computed=64)
        short = make_request("short", prompt_len=5, slot=1)

        plan, requests, _ = _chunk_work(PassKind.PREFILL, [(capped, 64), (short, 5)])

        assert plan.sampled == (1,)
        assert plan.sampling == (short.params,)
        assert requests == [short]


class TestResumedChunkPlans:
    """Chunks landing on top of a prefix that is already cached: the extend pass."""

    def test_a_resumed_chunk_starts_where_the_prefix_ended(self):
        request = make_request("a", prompt_len=200, slot=3, computed=128)

        plan, _, _ = _chunk_work(PassKind.EXTEND, [(request, 64)])

        assert plan.kind is PassKind.EXTEND
        assert plan.seq_starts == (64,)
        assert plan.seq_lens == (128,)
        assert plan.tokens == tuple(3000 + index for index in range(64, 128))

    def test_the_route_is_chosen_per_chunk(self):
        """One step, both kernels: a fresh prompt cannot extend, a resumed one must."""
        fresh = make_request("fresh", prompt_len=80, slot=0, computed=64)
        resumed = make_request("resumed", prompt_len=200, slot=1, computed=128)

        work = _prefill_work([fresh, resumed], [64, 64])

        assert [item.plan.kind for item in work] == [PassKind.PREFILL, PassKind.EXTEND]
        assert work[0].plan.slots == (0,) and work[1].plan.slots == (1,)

    def test_a_homogeneous_group_makes_one_plan(self):
        first = make_request("a", prompt_len=8, slot=0)
        second = make_request("b", prompt_len=8, slot=1)

        work = _prefill_work([first, second], [8, 8])

        assert len(work) == 1
        assert work[0].plan.slots == (0, 1)

    def test_resumed_rows_route_by_replay_capacity(self):
        """The resumed group is one pass, routed by its total row count.

        Rows that fit a captured graph batch extend (the pass replays a decode
        graph); anything longer takes the chunked kernel.
        """
        small = make_request("small", prompt_len=20, slot=0, computed=10)
        chunked = make_request("big", prompt_len=300, slot=1, computed=160)

        replayable = _prefill_work([small], [8], chunked_min_rows=33)
        assert replayable[0].plan.kind is PassKind.EXTEND

        longer = _prefill_work([chunked], [140], chunked_min_rows=33)
        assert longer[0].plan.kind is PassKind.PREFILL

        # The whole resumed group routes together, decided by its row total.
        both = _prefill_work([small, chunked], [8, 140], chunked_min_rows=33)
        assert len(both) == 1
        assert both[0].plan.kind is PassKind.PREFILL
        assert both[0].plan.slots == (0, 1)


class TestDecodePlans:
    """One token per running request, planned from host state alone."""

    def test_the_input_token_is_the_last_one_generated(self):
        request = make_request("a", prompt_len=5, slot=0, generated=3)

        plan, requests, _ = _decode_work([request])

        assert plan.kind is PassKind.DECODE
        assert plan.tokens == (request.output_token_ids[-1],)
        assert requests == [request]

    def test_each_row_feeds_one_token_at_its_own_position(self):
        short = make_request("a", prompt_len=5, slot=0, generated=1)
        longer = make_request("b", prompt_len=9, slot=1, generated=4)

        plan, _, _ = _decode_work([short, longer])

        # seq_len counts the token about to be fed, so it writes to seq_len - 1.
        assert plan.seq_lens == (6, 13)
        assert plan.seq_starts == (5, 12)
        assert plan.chunk_lens == (1, 1)

    def test_every_row_is_sampled(self):
        requests = [make_request(str(index), 4, index, generated=1) for index in range(3)]

        plan, _, _ = _decode_work(requests)

        assert plan.sampled == (0, 1, 2)
        assert len(plan.sampling) == 3

    def test_generated_counts_address_the_next_grid_column(self):
        """``gen_counts`` is both the column the new token lands in and the width
        of the history its repetition penalty may see."""
        request = make_request("a", prompt_len=5, slot=0, generated=3, params=PENALISED)

        plan, _, _ = _decode_work([request])

        assert plan.gen_counts == (3,)
        assert plan.sampling == (PENALISED,)
