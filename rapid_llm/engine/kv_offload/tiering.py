"""The engine's single handle on the offloading stack.

The engine talks to one :class:`TieringOffloadingManager` and never to a
concrete tier: v0.12.0's stack is the CPU tier alone, so every call passes
straight through, and when a secondary tier (disk, remote) lands this is the
only file that grows an orchestration policy — *how* a primary miss is
answered from below (pull-through? read-through?), how stores spill, and how
two async completion streams merge into one event queue. The engine's call
sites do not change.

Passing a non-empty ``secondaries`` today raises instead of silently ignoring
them: a half-wired tier that answers some lookups from disk and drops the rest
would be a correctness bug that only shows up under cache pressure.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import (
    LoadSpec,
    LookupResult,
    OffloadEvent,
    OffloadingManager,
    OffloadingStats,
    OffloadKey,
    StoreSpec,
)
from .secondary import SecondaryTierManager


class TieringOffloadingManager(OffloadingManager):
    """Drives a primary tier, and (later) any secondary tiers behind it.

    Args:
        primary: The tier that owns the stage block pool — the CPU tier,
            whose blocks the data path copies into and out of.
        secondaries: Tiers behind the primary. Empty in v0.12.0; anything
            else is a wiring error until the orchestration exists.
    """

    def __init__(
        self,
        primary: OffloadingManager,
        secondaries: Sequence[SecondaryTierManager] = (),
    ) -> None:
        if secondaries:
            raise NotImplementedError(
                "secondary tiers are not wired yet (v0.12.0 ships the CPU tier alone); "
                "the orchestration policy lands with the first implementation"
            )
        self._primary = primary

    @property
    def primary(self) -> OffloadingManager:
        """The stage tier every lookup bottoms out in."""
        return self._primary

    # ----------------------------------------------------------------- query #
    def lookup(self, key: OffloadKey) -> LookupResult:
        return self._primary.lookup(key)

    def touch(self, keys: Sequence[OffloadKey]) -> None:
        self._primary.touch(keys)

    # ------------------------------------------------------------------ load #
    def prepare_load(
        self, keys: Sequence[OffloadKey], gpu_blocks: Sequence[int]
    ) -> LoadSpec | None:
        return self._primary.prepare_load(keys, gpu_blocks)

    def complete_load(self, keys: Sequence[OffloadKey]) -> None:
        self._primary.complete_load(keys)

    # ----------------------------------------------------------------- store #
    def prepare_store(self, entries: Sequence[tuple[int, OffloadKey]]) -> StoreSpec:
        return self._primary.prepare_store(entries)

    def complete_store(self, keys: Sequence[OffloadKey]) -> None:
        self._primary.complete_store(keys)

    # ---------------------------------------------------------------- events #
    def take_events(self) -> list[OffloadEvent]:
        return self._primary.take_events()

    def has_pending_work(self) -> bool:
        return self._primary.has_pending_work()

    def pending_store_events(self) -> list[object]:
        return self._primary.pending_store_events()

    # -------------------------------------------------------------- lifecycle #
    def reset_cache(self) -> bool:
        return self._primary.reset_cache()

    def get_stats(self) -> OffloadingStats:
        return self._primary.get_stats()
