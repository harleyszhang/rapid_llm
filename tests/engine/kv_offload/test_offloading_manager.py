"""Tests for the CPU primary tier — pure CPU, no GPU or checkpoint.

Case list mirrors vLLM's ``tests/v1/simple_kv_offload/test_scheduler.py``
(case names deliberately close, so the two suites can be read side by side):
roundtrip, duplicate-store skip, LRU order, touch survival, no leaks,
TOCTOU between lookup and prepare, in-flight semantics, event draining and
the tiering pass-through.

Usage:
    pytest tests/engine/kv_offload/test_offloading_manager.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.kv_offload.base import CopyRun, LookupResult
from rapid_llm.engine.kv_offload.cpu import CPUPrimaryTierOffloadingManager
from rapid_llm.engine.kv_offload.tiering import TieringOffloadingManager

BLOCK_SIZE = 4


class _FakeEvent:
    """Duck-typed stand-in for ``torch.cuda.Event``: fires on demand."""

    def __init__(self) -> None:
        self.signaled = False

    def query(self) -> bool:
        return self.signaled


class RecordingCopier:
    """A :class:`~rapid_llm.engine.kv_offload.base.BlockCopier` under test control.

    Records every batch. With ``hold`` set, batches return an unsignaled event
    (the "in flight" state); otherwise they complete synchronously — the same
    two modes the real copier shows under an idle and a loaded stream.
    """

    def __init__(self, hold: bool = False) -> None:
        self.batches: list[tuple[tuple[CopyRun, ...], bool]] = []
        self.hold = hold
        self._held: list[_FakeEvent] = []

    def submit(self, moves, *, is_store):
        self.batches.append((tuple(moves), is_store))
        if not self.hold:
            return None
        event = _FakeEvent()
        self._held.append(event)
        return event

    def release_all(self) -> None:
        for event in self._held:
            event.signaled = True
        self._held.clear()

    @property
    def moves(self) -> list[CopyRun]:
        return [move for batch, _ in self.batches for move in batch]


def make_manager(num_blocks: int = 8, hold: bool = False, bytes_per_block: int = 0):
    """A manager over a small CPU pool, plus its copier."""
    copier = RecordingCopier(hold=hold)
    manager = CPUPrimaryTierOffloadingManager(
        num_blocks=num_blocks,
        block_size=BLOCK_SIZE,
        copier=copier,
        bytes_per_block=bytes_per_block,
    )
    return manager, copier


def store_and_complete(manager, entries):
    """Store ``entries`` and drain the events, i.e. "the copy has landed"."""
    spec = manager.prepare_store(entries)
    manager.take_events()
    return spec


# --------------------------------------------------------------------------- #
# 1. Roundtrip: store -> lookup -> load -> complete
# --------------------------------------------------------------------------- #
class TestStoreLoadRoundtrip:
    """The happy path both directions, and what each side leaves behind."""

    def test_store_then_load_roundtrip(self):
        manager, copier = make_manager()
        store_and_complete(manager, [(10, 101), (11, 102)])

        assert manager.lookup(101) is LookupResult.HIT
        assert copier.batches[-1][1] is True  # the store went down

        spec = manager.prepare_load([101, 102], [20, 21])
        assert spec is not None
        assert spec.keys == (101, 102)
        assert spec.cpu_blocks == (1, 2)
        assert spec.moves == (
            CopyRun(gpu_block=20, cpu_block=1),
            CopyRun(gpu_block=21, cpu_block=2),
        )
        assert copier.batches[-1][1] is False  # the load came back up

        # Pinned while in flight: the pool has two live holders plus the null.
        assert manager.pool.num_used_blocks == 3
        manager.take_events()
        assert manager.pool.num_used_blocks == 1  # unpinned, still cached
        assert manager.lookup(101) is LookupResult.HIT
        assert manager.lookup(102) is LookupResult.HIT

    def test_duplicate_store_skipped(self):
        manager, copier = make_manager()
        store_and_complete(manager, [(10, 101)])
        batches = len(copier.batches)
        spec = store_and_complete(manager, [(12, 101)])
        assert spec.moves == ()
        assert len(copier.batches) == batches  # nothing new was submitted

    def test_missing_key_is_a_miss(self):
        manager, _ = make_manager()
        assert manager.lookup(999) is LookupResult.MISS


# --------------------------------------------------------------------------- #
# 2. In-flight transfers
# --------------------------------------------------------------------------- #
class TestInflightTransfers:
    """What the tier answers while a copy is still moving."""

    def test_store_in_flight_is_not_yet_loadable(self):
        manager, copier = make_manager(hold=True)
        manager.prepare_store([(10, 101)])
        assert manager.has_pending_work()
        assert manager.lookup(101) is LookupResult.MISS  # rows still being written

        copier.release_all()
        events = manager.take_events()
        assert [e.kind for e in events] == ["store"]
        assert events[0].keys == (101,)
        assert not manager.has_pending_work()
        assert manager.lookup(101) is LookupResult.HIT

    def test_duplicate_store_in_flight_skipped(self):
        manager, copier = make_manager(hold=True)
        first = manager.prepare_store([(10, 101)])
        second = manager.prepare_store([(11, 101)])
        assert first.moves != ()
        assert second.moves == ()
        assert len(copier.batches) == 1

    def test_load_blocks_stay_pinned_until_event(self):
        manager, copier = make_manager(hold=True)
        copier.hold = False
        store_and_complete(manager, [(10, 101)])
        copier.hold = True
        manager.prepare_load([101], [20])
        assert manager.pool.num_used_blocks == 2  # pinned by the load

        copier.release_all()
        events = manager.take_events()
        assert events[0].kind == "load"
        assert events[0].gpu_blocks == (20,)
        assert manager.pool.num_used_blocks == 1


# --------------------------------------------------------------------------- #
# 3. Eviction: LRU order, touch, pins
# --------------------------------------------------------------------------- #
class TestEviction:
    """Which block pays when the CPU pool is full."""

    def test_lru_eviction_order(self):
        manager, _ = make_manager(num_blocks=4)  # 3 usable blocks
        store_and_complete(manager, [(10, 1), (11, 2), (12, 3)])

        spec = store_and_complete(manager, [(13, 4)])
        assert spec.evicted == (1,)  # stored first, evicted first
        assert manager.lookup(1) is LookupResult.MISS
        assert manager.lookup(2) is LookupResult.HIT

    def test_touched_blocks_survive_eviction(self):
        manager, _ = make_manager(num_blocks=4)
        store_and_complete(manager, [(10, 1), (11, 2), (12, 3)])

        manager.touch([1])
        store_and_complete(manager, [(13, 4)])
        assert manager.lookup(1) is LookupResult.HIT  # moved to the tail
        assert manager.lookup(2) is LookupResult.MISS

    def test_pinned_blocks_never_evicted(self):
        manager, _ = make_manager(num_blocks=4)
        store_and_complete(manager, [(10, 1), (11, 2), (12, 3)])

        spec = manager.prepare_load([2], [20])  # pins block of key 2
        assert spec is not None
        store_and_complete(manager, [(13, 4)])
        assert manager.lookup(2) is LookupResult.HIT
        assert manager.lookup(1) is LookupResult.MISS

    def test_store_skipped_when_all_blocks_pinned(self):
        manager, copier = make_manager(num_blocks=3)  # 2 usable blocks
        store_and_complete(manager, [(10, 1), (11, 2)])

        manager.prepare_load([1, 2], [20, 21])  # pins both
        spec = manager.prepare_store([(12, 3)])
        assert spec.moves == ()  # nowhere to put it; nothing was copied
        assert len(copier.batches) == 2  # the load batch, and the first store


# --------------------------------------------------------------------------- #
# 4. TOCTOU between the lookup and the prepare
# --------------------------------------------------------------------------- #
class TestLookupToctou:
    """A hit can stop being one before the scheduler gets around to loading it."""

    def test_hit_evicted_before_prepare_load_returns_none(self):
        manager, _ = make_manager(num_blocks=3)  # 2 usable blocks
        store_and_complete(manager, [(10, 1), (11, 2)])
        assert manager.lookup(1) is LookupResult.HIT

        store_and_complete(manager, [(12, 3)])  # evicts key 1
        assert manager.prepare_load([1], [20]) is None

    def test_partial_missing_pins_nothing(self):
        manager, _ = make_manager(num_blocks=3)
        store_and_complete(manager, [(10, 1), (11, 2)])
        store_and_complete(manager, [(12, 3)])  # evicts key 1; key 2 survives

        assert manager.prepare_load([1, 2], [20, 21]) is None
        assert manager.pool.num_used_blocks == 1  # key 2 was *not* pinned either
        assert manager.lookup(2) is LookupResult.HIT


# --------------------------------------------------------------------------- #
# 5. Lifecycle: reset, stats, leaks
# --------------------------------------------------------------------------- #
class TestLifecycle:
    """Shutdown and accounting."""

    def test_no_cpu_block_leak_after_full_cycle(self):
        manager, _ = make_manager()
        for round_trip in range(3):
            keys = [round_trip * 10 + i for i in range(4)]
            store_and_complete(manager, [(100 + i, k) for i, k in enumerate(keys)])
            manager.prepare_load(keys, [200 + i for i in range(4)])
            manager.take_events()
        assert manager.pool.num_used_blocks == 1  # only the null block

    def test_reset_refused_while_pending(self):
        manager, copier = make_manager(hold=True)
        manager.prepare_store([(10, 1)])
        assert manager.reset_cache() is False
        copier.release_all()
        assert manager.reset_cache() is True
        assert manager.lookup(1) is LookupResult.MISS

    def test_reset_refused_while_load_in_flight(self):
        manager, copier = make_manager()
        store_and_complete(manager, [(10, 1)])
        copier.hold = True
        manager.prepare_load([1], [20])  # the load pins the CPU block
        assert manager.pool.num_used_blocks == 2
        assert manager.reset_cache() is False
        copier.release_all()
        manager.take_events()
        assert manager.reset_cache() is True

    def test_stats_counters(self):
        manager, _ = make_manager(bytes_per_block=100)
        assert manager.get_stats().hit_rate == 0.0
        store_and_complete(manager, [(10, 1), (11, 2)])
        manager.lookup(1)  # hit
        manager.lookup(999)  # miss

        manager.prepare_load([1], [20])
        manager.take_events()

        stats = manager.get_stats()
        assert (stats.lookups, stats.hits, stats.misses) == (2, 1, 1)
        assert stats.stores == 2
        assert stats.loads == 1
        assert stats.store_bytes == 200
        assert stats.load_bytes == 100

    def test_pending_store_events_visible(self):
        manager, copier = make_manager(hold=True)
        manager.prepare_store([(10, 1)])
        assert len(manager.pending_store_events()) == 1
        copier.release_all()
        manager.take_events()
        assert manager.pending_store_events() == []


# --------------------------------------------------------------------------- #
# 6. Tiering: the engine-facing handle
# --------------------------------------------------------------------------- #
class TestTieringManager:
    """Pass-through semantics, and the refusal to half-wire secondaries."""

    def test_passthrough(self):
        primary, copier = make_manager()
        tiering = TieringOffloadingManager(primary)
        assert tiering.primary is primary

        store_and_complete(tiering, [(10, 1)])
        assert tiering.lookup(1) is LookupResult.HIT
        spec = tiering.prepare_load([1], [20])
        assert spec is not None
        tiering.take_events()
        assert copier.batches[-1][1] is False
        assert tiering.get_stats().loads == 1
        assert not tiering.has_pending_work()

    def test_secondaries_refused_until_orchestrated(self):
        primary, _ = make_manager()

        class _Stub:
            medium = "STORAGE"
            locality = "LOCAL"

        with pytest.raises(NotImplementedError, match="secondary tiers"):
            TieringOffloadingManager(primary, secondaries=[_Stub()])
