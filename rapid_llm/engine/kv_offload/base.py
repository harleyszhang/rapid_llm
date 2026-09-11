"""Offloading vocabulary: tiers, keys, results, specs and the scheduler-side protocol.

An offloading *tier* is a place KV blocks can live outside the GPU pool, named
by two dimensions (the same split vLLM's ``v1/kv_offload`` uses):

* :class:`Medium` — what the bytes sit on (CPU DRAM, disk).
* :class:`Locality` — where that medium is, relative to this engine
  (LOCAL, another GPU).

v0.12.0 ships one tier — the CPU primary tier
(:class:`~rapid_llm.engine.kv_offload.cpu.CPUPrimaryTierOffloadingManager`) —
and the protocols below are what a secondary tier (disk, remote) will plug
into without touching the engine call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Protocol


class Medium(StrEnum):
    """What an offloading tier's bytes sit on."""

    CPU = "CPU"
    STORAGE = "STORAGE"


class Locality(StrEnum):
    """Where a tier's medium is, relative to the engine holding the GPU."""

    LOCAL = "LOCAL"
    REMOTE = "REMOTE"


#: A block's identity in an offloading tier.
#:
#: v0.12.0 offloads exactly one KV group (the full-attention one, the only
#: group ``write_block_tables`` wires through), so the chained block hash is
#: the whole key. A multi-group layout packs ``(group_idx, hash)`` here the
#: way vLLM's ``OffloadKey`` does.
OffloadKey = int


class LookupResult(Enum):
    """What a tier knows about one key at this instant.

    HIT_PENDING and RETRY are part of the protocol but unused by v0.12.0: the
    CPU tier answers synchronously and completely (MISS or HIT), and there is
    no secondary tier whose answers could still be in flight.
    """

    MISS = "MISS"
    HIT = "HIT"
    HIT_PENDING = "HIT_PENDING"
    RETRY = "RETRY"


@dataclass(frozen=True)
class CopyRun:
    """One block's rows, on the two sides of a transfer.

    A run is always a whole block: the CPU tier keeps the same block size as
    the GPU pool, so a transfer is a contiguous row range on both sides and
    never needs a partial copy. The direction is the caller's: a store moves
    ``gpu_block`` -> ``cpu_block``, a load the other way.
    """

    gpu_block: int
    cpu_block: int


@dataclass(frozen=True)
class LoadSpec:
    """What the engine needs to hand to the worker to promote blocks.

    Attributes:
        cpu_blocks: CPU-side block ids, in load order.
        gpu_blocks: The GPU blocks they land in, parallel to ``cpu_blocks``.
        keys: The content keys of the blocks, parallel to the others.
    """

    cpu_blocks: tuple[int, ...]
    gpu_blocks: tuple[int, ...]
    keys: tuple[OffloadKey, ...]

    @property
    def moves(self) -> tuple[CopyRun, ...]:
        """The block moves this load performs."""
        return tuple(
            CopyRun(gpu_block=g, cpu_block=c)
            for g, c in zip(self.gpu_blocks, self.cpu_blocks, strict=True)
        )


@dataclass(frozen=True)
class StoreSpec:
    """What a store would move, and what paying for it cost.

    Attributes:
        moves: The block moves the store performs, already submitted.
        keys: Content keys, parallel to ``moves``.
        evicted: Keys the CPU tier dropped to make room. Their data is gone.
    """

    moves: tuple[CopyRun, ...] = ()
    keys: tuple[OffloadKey, ...] = ()
    evicted: tuple[OffloadKey, ...] = ()


@dataclass(frozen=True)
class OffloadEvent:
    """A transfer that has fully landed, handed back to the engine.

    The engine consumes these once per step: a ``load`` event is what releases
    a request from waiting-for-KV (its blocks are readable now), a ``store``
    event is what makes a key loadable by later requests. Events carry block
    ids rather than request ids — the engine owns the request->blocks mapping,
    and the manager stays request-agnostic.
    """

    kind: str  # "store" | "load"
    keys: tuple[OffloadKey, ...]
    cpu_blocks: tuple[int, ...]
    gpu_blocks: tuple[int, ...] = ()


@dataclass
class OffloadingStats:
    """Cumulative counters for one tier manager."""

    lookups: int = 0
    hits: int = 0
    stores: int = 0
    loads: int = 0
    store_bytes: int = 0
    load_bytes: int = 0
    evictions: int = 0

    @property
    def misses(self) -> int:
        return self.lookups - self.hits

    @property
    def hit_rate(self) -> float:
        """Fraction of lookups served from the tier (0.0 - 1.0)."""
        if self.lookups == 0:
            return 0.0
        return self.hits / self.lookups


class BlockCopier(Protocol):
    """The data path: issues block moves and reports when they land.

    A copier owns the streams, the buffers and the events; the manager owns
    the *decision* of what to move and the CPU-side metadata. On the engine
    side the implementation is
    :class:`~rapid_llm.executor.kv_offload.KVCopyEngine` (CUDA streams); in
    tests it is a synchronous stand-in that returns ``None``.

    The returned event is duck-typed: anything with a ``query() -> bool``
    works, and ``None`` means the move is already done. Keeping it
    unannotated is what lets this package stay torch-free.
    """

    def submit(self, moves: Sequence[CopyRun], *, is_store: bool) -> Any | None:
        """Issue *moves*; return an event that fires at completion, or None."""
        ...


class OffloadingManager(ABC):
    """Scheduler-side protocol over one or more offloading tiers.

    The scheduler talks to this in the same units it talks to
    :class:`~rapid_llm.engine.prefix_cache.PrefixCache`: block keys in, block
    ids out. The call sequence for one request is:

    * :meth:`lookup` per key while walking a prompt's hash chain, stopping at
      the first miss;
    * once the scheduler has found GPU blocks for the hit, :meth:`prepare_load`
      pins the CPU blocks and submits the move;
    * :meth:`complete_load` (or the event drain in :meth:`take_events`) marks
      the load done and lets the blocks be evicted again;
    * after a commit indexes fresh GPU blocks, :meth:`prepare_store` copies
      them down; :meth:`complete_store` makes their keys loadable.

    :meth:`touch` is deliberately separate: a request whose prefix was served
    by the GPU cache never calls prepare_load, but its blocks are just as hot
    and should be kept off the eviction front.
    """

    # ----------------------------------------------------------------- query #
    @abstractmethod
    def lookup(self, key: OffloadKey) -> LookupResult:
        """Whether one block is offloaded and ready to load."""

    @abstractmethod
    def touch(self, keys: Sequence[OffloadKey]) -> None:
        """Mark blocks as recently used, without pinning them."""

    # ------------------------------------------------------------------ load #
    @abstractmethod
    def prepare_load(
        self,
        keys: Sequence[OffloadKey],
        gpu_blocks: Sequence[int],
    ) -> LoadSpec | None:
        """Pin the blocks for *keys* and submit them into *gpu_blocks*.

        Returns None when a key is no longer resident (evicted between the
        lookup and this call); the caller falls back to a full prefill.
        """

    @abstractmethod
    def complete_load(self, keys: Sequence[OffloadKey]) -> None:
        """Mark promoted blocks as done loading and evictable again."""

    # ----------------------------------------------------------------- store #
    @abstractmethod
    def prepare_store(
        self,
        entries: Sequence[tuple[int, OffloadKey]],
    ) -> StoreSpec:
        """Copy fresh GPU blocks ``(gpu_block_id, key)`` into the CPU tier.

        Keys already resident (or already in flight) are skipped. Evicting a
        cached block to make room is allowed here — the CPU pool answers with
        the keys it dropped, and the drain reports them to the engine.
        """

    @abstractmethod
    def complete_store(self, keys: Sequence[OffloadKey]) -> None:
        """Make stored keys loadable by later requests."""

    # ---------------------------------------------------------------- events #
    @abstractmethod
    def take_events(self) -> list[OffloadEvent]:
        """Collect every transfer that has landed since the last call."""

    @abstractmethod
    def has_pending_work(self) -> bool:
        """True while any transfer is in flight."""

    def pending_store_events(self) -> list[Any]:
        """Events a compute stream must wait on before overwriting reused blocks.

        A tier whose stores read the *live* device cache needs this: a block
        the pool just handed out again could be overwritten before the store
        that is still reading it finishes. Tiers that stage their own copies
        return nothing — hence a default, not an abstract method.
        """
        return []

    # -------------------------------------------------------------- lifecycle #
    @abstractmethod
    def reset_cache(self) -> bool:
        """Drop every offloaded block; False when some are still pinned."""

    @abstractmethod
    def get_stats(self) -> OffloadingStats:
        """Cumulative counters for this manager."""


@dataclass
class _Transfer:
    """One submitted batch of moves, and its bookkeeping.

    Internal to the tier managers; in :mod:`rapid_llm.engine.kv_offload.base`
    rather than in one of them so both can type against it.
    """

    kind: str  # "store" | "load"
    keys: tuple[OffloadKey, ...]
    cpu_blocks: tuple[int, ...]
    gpu_blocks: tuple[int, ...]
    event: Any | None
    moves: tuple[CopyRun, ...] = field(default_factory=tuple)
