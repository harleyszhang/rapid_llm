"""The scheduler driving a CPU offloading tier, end to end.

The tier's own suite (``test_offloading_manager.py``) drives the manager in
isolation, case for case against vLLM's ``simple_kv_offload`` tests. This one
wires it into the :class:`Scheduler` and asserts the whole arc: a commit
mirrors blocks down, an admission parks on the copy, the event drain lands it
and re-indexes the GPU blocks, and every unwind path — a request aborted in
flight, a lost seat, a refused pin — releases exactly what it should.

The block arithmetic is exact, which is why every prompt is block-aligned and
chunked prefill is off: one full-attention group, 16-token blocks, and a pool
small enough that a specific block's fate is predictable. The scenario the
suite builds on (``partial_hit_setup``):

* ``a`` (64 tokens, four blocks) runs to completion; its blocks mirror down to
  the CPU tier, which is drained so their keys are loadable;
* ``f`` (an unrelated prompt sized to consume the whole free queue) evicts the
  pool's copy of ``a``'s tail block;
* leaving a request that shares ``a``'s prompt with a 48-token GPU hit and the
  CPU tier holding all 64 tokens — one block left to promote.

Usage:
    pytest tests/engine/kv_offload/test_scheduler_integration.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.kv_offload.cpu import CPUPrimaryTierOffloadingManager
from rapid_llm.engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Request, RequestStatus, Scheduler, SchedulerConfig

BLOCK = PREFIX_CACHE_BLOCK_SIZE
#: ``a``'s prompt: 64 tokens, exactly four blocks.
A_TOKENS = list(range(1000, 1064))


def make_request(request_id: str, token_ids) -> Request:
    """A request over explicit token ids, so block hashes are controllable."""
    token_ids = list(token_ids)
    return Request(
        request_id=request_id,
        prompt="x" * len(token_ids),
        prompt_token_ids=token_ids,
        params=SamplingParams(),
    )


class _FakeEvent:
    """Duck-typed ``torch.cuda.Event``: fires on demand."""

    def __init__(self) -> None:
        self.signaled = False

    def query(self) -> bool:
        return self.signaled


class RecordingCopier:
    """A :class:`BlockCopier` under test control.

    Every batch is held (an unsignaled event, the "in flight" state) until
    :meth:`release_all` — which is what lets a test act between the moment a
    copy is submitted and the moment it lands.
    """

    def __init__(self) -> None:
        self.batches: list[tuple[tuple[object, ...], bool]] = []
        self._held: list[_FakeEvent] = []

    def submit(self, moves, *, is_store):
        self.batches.append((tuple(moves), is_store))
        event = _FakeEvent()
        self._held.append(event)
        return event

    def release_all(self) -> None:
        """Signal every in-flight event, i.e. let the copies land."""
        for event in self._held:
            event.signaled = True
        self._held.clear()

    @property
    def directions(self) -> list[bool]:
        """``True`` per store batch, ``False`` per load batch, in order."""
        return [is_store for _, is_store in self.batches]

    @property
    def last_batch(self) -> tuple[tuple[object, ...], bool]:
        return self.batches[-1]


class _RefusingManager(CPUPrimaryTierOffloadingManager):
    """A manager that refuses every load — the TOCTOU case at its worst."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refusals = 0

    def prepare_load(self, keys, gpu_blocks):
        self.refusals += 1
        return None


def build(
    pool_blocks: int = 10,
    *,
    num_slots: int = 4,
    max_num_seqs: int = 4,
    enable_preemption: bool = False,
    cpu_blocks: int = 32,
    manager_cls=CPUPrimaryTierOffloadingManager,
):
    """A scheduler over a small GPU pool, wired to a tier and a held copier."""
    copier = RecordingCopier()
    manager = manager_cls(num_blocks=cpu_blocks, block_size=BLOCK, copier=copier)
    config = SchedulerConfig(
        max_seq_len=4096,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=65536,
        max_chunk_size=0,  # chunked prefill off: one chunk per prompt, exact blocks
        enable_prefix_cache=True,  # the tier keys blocks by their content hash
        enable_preemption=enable_preemption,
    )
    sched = Scheduler(config, num_slots=num_slots, num_blocks=pool_blocks, offloading=manager)
    return sched, manager, copier


def partial_hit_setup(pool_blocks: int = 10, **kwargs):
    """Split ``a``'s prefix between the GPU pool and the CPU tier.

    ``a`` (64 tokens) runs and finishes, mirroring its four blocks down; the
    drain makes them loadable. ``f`` — sized to consume the whole free queue,
    so it evicts ``a``'s tail block — then runs and finishes too. What is left
    is a pool serving a 48-token prefix of ``a`` with the CPU holding all of
    it, so a request sharing ``a``'s prompt has exactly one block to promote.
    """
    sched, manager, copier = build(pool_blocks=pool_blocks, **kwargs)

    a = make_request("a", A_TOKENS)
    sched.add_request(a)
    sched.schedule()
    sched.finish(a, "eos")  # settles the commit: the store is in flight now
    copier.release_all()
    manager.take_events()  # drained: the CPU holds all four of a's keys

    f = make_request("f", range(2000, 2000 + (pool_blocks - 4) * BLOCK))
    sched.add_request(f)
    sched.schedule()
    sched.finish(f, "eos")  # ate the free queue, evicting a's tail block
    return sched, manager, copier


# --------------------------------------------------------------------------- #
# 1. The happy arc: store on commit, park on admit, land, seat, resume
# --------------------------------------------------------------------------- #
class TestStoreThenPromote:
    """One promotion through its whole lifecycle."""

    def test_gpu_hit_is_extended_from_the_cpu_tier(self):
        sched, manager, copier = partial_hit_setup()
        assert copier.directions == [True, True]  # a's store, f's store

        b = make_request("b", A_TOKENS + list(range(3000, 3032)))
        sched.add_request(b)
        out = sched.schedule()

        # The pool serves 48 tokens of the prompt; the fourth block is being
        # copied, so nothing about b may run this step.
        assert out.is_empty
        assert sched.num_promoting == 1
        assert sched.num_waiting == 0  # parked, not queued
        assert b.status is RequestStatus.WAITING and b.slot is None
        assert b.num_computed_tokens == 0
        assert copier.directions == [True, True, False]
        moves, is_store = copier.last_batch
        assert is_store is False
        assert len(moves) == 1  # exactly the one block the pool was missing

        copier.release_all()
        out = sched.schedule()

        # The copy landed: b is seated and resumes its final 32 tokens.
        assert [r.request_id for r in out.prefill] == ["b"]
        assert b.num_cached_tokens == 64  # GPU head + the promoted block
        assert b.num_computed_tokens == 96
        group_id, start, blocks = b.block_plan[0]
        assert (group_id, start) == (0, 0)
        assert len(blocks) == 6  # the whole prompt maps in one step

        # The resume's commit mirrors the two freshly computed blocks down.
        sched.schedule()
        assert copier.directions == [True, True, False, True]
        moves, is_store = copier.last_batch
        assert is_store is True
        assert len(moves) == 2

        sched.finish(b, "eos")
        assert not sched.has_unfinished_requests()
        assert sched.num_free_blocks == 9

        copier.release_all()
        manager.take_events()
        assert manager.pool.num_used_blocks == 1  # only the CPU null block

    def test_promoted_blocks_become_gpu_hits_for_the_next_request(self):
        sched, _manager, copier = partial_hit_setup()
        b = make_request("b", A_TOKENS + list(range(3000, 3032)))
        sched.add_request(b)
        sched.schedule()  # parks on the copy
        copier.release_all()
        sched.schedule()  # lands, seats, resumes
        assert sched.num_promoting == 0

        c = make_request("c", A_TOKENS + list(range(4000, 4016)))
        sched.add_request(c)
        out = sched.schedule()

        # The block b's copy re-indexed now serves straight from the pool:
        # no second promotion, no second load.
        assert [r.request_id for r in out.prefill] == ["c"]
        assert c.num_cached_tokens == 64
        assert sched.num_promoting == 0
        assert copier.directions.count(False) == 1


# --------------------------------------------------------------------------- #
# 2. Teardown: the request dies while its copy is in flight
# --------------------------------------------------------------------------- #
class TestPromotionTeardown:
    """What a promotion's blocks do when the request is aborted."""

    def test_abort_mid_promotion_defers_the_free_until_the_copy_lands(self):
        sched, _manager, copier = partial_hit_setup()
        b = make_request("b", A_TOKENS + list(range(3000, 3032)))
        sched.add_request(b)
        sched.schedule()
        assert sched.num_promoting == 1

        free_before = sched.num_free_blocks
        assert sched.abort("b") is b
        assert b.is_finished
        # The copy is landing into blocks b holds: freeing them now would let
        # the pool hand out rows the DMA is still writing.
        assert sched.num_free_blocks == free_before
        assert sched.has_unfinished_requests()  # the dying promotion keeps it alive

        copier.release_all()
        sched.schedule()
        assert sched.num_free_blocks == free_before + 4  # the deferred free ran
        assert not sched.has_unfinished_requests()
        assert sched.num_promoting == 0


# --------------------------------------------------------------------------- #
# 3. Seat gating: a landed promotion waits for capacity
# --------------------------------------------------------------------------- #
class TestSeatGating:
    """A promotion that landed still needs a slot and a seat."""

    def test_a_landed_promotion_waits_for_a_seat(self):
        sched, _manager, copier = partial_hit_setup(pool_blocks=11, num_slots=2, max_num_seqs=2)

        # 17 tokens: two blocks with a partial tail, so its decode step never
        # grows it and the block arithmetic stays fixed.
        h1 = make_request("h1", range(5000, 5017))
        sched.add_request(h1)
        sched.schedule()

        b = make_request("b", A_TOKENS + list(range(3000, 3032)))
        h2 = make_request("h2", range(6000, 6017))
        sched.add_request(b)
        sched.add_request(h2)
        out = sched.schedule()

        assert b not in out.prefill and b not in out.decode
        assert b.status is RequestStatus.WAITING
        assert sched.num_promoting == 1

        copier.release_all()
        out = sched.schedule()

        # The copy landed and was indexed, but every seat is taken: b waits
        # in place rather than re-entering the queue (re-admission would
        # allocate a second set of blocks for the same prefix).
        assert b not in out.prefill and b not in out.decode
        assert b.status is RequestStatus.WAITING and b.slot is None
        assert sched.num_promoting == 1
        assert b.num_computed_tokens == 64  # the copy is readable

        sched.finish(h1, "eos")
        out = sched.schedule()
        assert [r.request_id for r in out.prefill] == ["b"]
        assert b.num_computed_tokens == 96


# --------------------------------------------------------------------------- #
# 4. The downgrade: a refused pin falls back to the plain path
# --------------------------------------------------------------------------- #
class TestRefusedPin:
    """A block can vanish between the probe and the pin; nothing may leak."""

    def test_refusal_falls_back_to_the_plain_path_without_losing_the_chain(self):
        sched, manager, copier = partial_hit_setup(manager_cls=_RefusingManager)

        b = make_request("b", A_TOKENS + list(range(3000, 3032)))
        sched.add_request(b)
        out = sched.schedule()

        # The probe found CPU blocks and the GPU allocation succeeded, but the
        # pin was refused: b degrades to the plain path *this* step, with its
        # tracked chain intact (the rollback freed it) and its whole mapping
        # planned.
        assert manager.refusals == 1
        assert sched.num_promoting == 0
        assert [r.request_id for r in out.prefill] == ["b"]
        assert b.num_cached_tokens == 48
        assert b.num_computed_tokens == 96
        group_id, start, blocks = b.block_plan[0]
        assert (group_id, start) == (0, 0)
        assert len(blocks) == 6
        assert copier.directions == [True, True]  # no load batch was submitted


# --------------------------------------------------------------------------- #
# 5. Construction guards
# --------------------------------------------------------------------------- #
class TestConfiguration:
    """Offloading's preconditions are checked at construction."""

    def test_offloading_needs_prefix_caching(self):
        manager = CPUPrimaryTierOffloadingManager(
            num_blocks=8, block_size=BLOCK, copier=RecordingCopier()
        )
        config = SchedulerConfig(
            max_seq_len=4096,
            max_num_seqs=4,
            max_num_batched_tokens=65536,
            max_chunk_size=0,
            enable_prefix_cache=False,
        )
        with pytest.raises(ValueError, match="enable_prefix_cache"):
            Scheduler(config, num_slots=4, num_blocks=10, offloading=manager)
