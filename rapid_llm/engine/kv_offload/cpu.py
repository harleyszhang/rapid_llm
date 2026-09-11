"""The CPU primary tier: GPU blocks mirrored into a host DRAM block pool.

The tier's metadata is a second :class:`~rapid_llm.engine.block_pool.BlockPool`
with host RAM standing behind it instead of HBM — same allocation, reference
counting, hash index and LRU queue, so there is one set of block bookkeeping
in the codebase, not two (vLLM writes a separate CPU block pool; the layout
here makes the reuse possible because the pool was written device-free).

Lifecycle of one block:

* a fresh GPU block is committed (indexed by hash) -> :meth:`prepare_store`
  allocates a CPU block and submits the GPU->CPU copy;
* the copy lands -> :meth:`complete_store` indexes the CPU block and drops
  the allocation reference, so it is cached-but-evictable;
* a later request walks its prompt hash chain -> :meth:`lookup` answers HIT
  while the key is resident;
* the scheduler found GPU blocks for the hit -> :meth:`prepare_load` pins the
  CPU blocks and submits the copy up;
* the copy lands -> :meth:`complete_load` unpins; the GPU side re-indexes the
  promoted blocks under the same hashes.

A key is loadable only between complete_store and the moment the CPU pool
evicts it; eviction is LRU over the CPU pool's free queue and its cost is
paid in accuracy (the data is gone), never correctness.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..block_pool import BlockPool
from .base import (
    BlockCopier,
    CopyRun,
    LoadSpec,
    LookupResult,
    OffloadEvent,
    OffloadingManager,
    OffloadingStats,
    OffloadKey,
    StoreSpec,
    _Transfer,
)


class CPUPrimaryTierOffloadingManager(OffloadingManager):
    """One tier: a CPU block pool behind the GPU pool, plus the moves.

    Args:
        num_blocks: CPU-side capacity in blocks, null block included. Each
            block holds a full block's K/V for every layer, so the host RAM
            this costs is ``num_blocks * block_size * row_bytes``.
        block_size: Tokens per block; must match the GPU pool the tier mirrors.
        copier: The data path. A synchronous stand-in in tests, the CUDA
            copier in the engine.
        bytes_per_block: Size of one block's payload, folded into the byte
            counters. Zero (the default) counts blocks only — the manager has
            no model geometry of its own.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        copier: BlockCopier,
        bytes_per_block: int = 0,
    ) -> None:
        if num_blocks < 2:
            raise ValueError(f"num_blocks must be >= 2 (one is the null block), got {num_blocks}")
        self.pool = BlockPool(num_blocks, block_size)
        self.block_size = block_size
        self.bytes_per_block = bytes_per_block
        self._copier = copier
        self._pending: list[_Transfer] = []
        self._inflight_stores: dict[OffloadKey, _Transfer] = {}
        # Cached blocks currently pinned by an in-flight load. Kept as a count,
        # not a set: the only question ever asked of it is "how many cached
        # blocks are *not* sitting in the free queue", which is what sizes the
        # eviction prediction below.
        self._pinned_cached = 0
        self._stats = OffloadingStats()

    @property
    def medium(self) -> str:
        """The medium this tier's bytes sit on."""
        return "CPU"

    # ----------------------------------------------------------------- query #
    def lookup(self, key: OffloadKey) -> LookupResult:
        """Whether *key* is offloaded and ready to load.

        A key whose store is still in flight is not ready — its rows are
        being written — but it will be momentarily, so this is exactly the
        situation HIT_PENDING exists for. v0.12.0 answers MISS instead (the
        caller recomputes, which is correct, just not optimal) because the
        engine's admission path has no place to park a request on that
        outcome yet.
        """
        self._stats.lookups += 1
        block = self.pool.get_cached_block(key)
        if block is not None:
            self._stats.hits += 1
            return LookupResult.HIT
        return LookupResult.MISS

    def touch(self, keys: Sequence[OffloadKey]) -> None:
        """Bump resident blocks' recency without pinning them.

        A request whose prefix was served by the *GPU* cache still reads the
        blocks it did not reload from here — under a shared workload they are
        hot, and letting the free queue eat them would be a self-inflicted
        miss. Only blocks nobody holds are moved, and moving them is just a
        splice to the queue's tail: the LRU contract is "least recently used
        first", and touch is the "used".
        """
        for key in keys:
            block = self.pool.get_cached_block(key)
            if block is None or block.ref_cnt > 0:
                continue
            self.pool.free_block_queue.remove(block)
            self.pool.free_block_queue.append(block)

    # ------------------------------------------------------------------ load #
    def prepare_load(
        self, keys: Sequence[OffloadKey], gpu_blocks: Sequence[int]
    ) -> LoadSpec | None:
        """Pin *keys*' blocks and submit them into *gpu_blocks*.

        Returns None when any key is no longer resident: the lookup that found
        it and this call are separated by the scheduler's search for GPU
        blocks, and a CPU store from another request can evict it in between.
        Nothing is pinned in that case, so the caller's fallback — recompute
        the whole prompt — needs no unwind of its own.
        """
        if len(keys) != len(gpu_blocks):
            raise ValueError("keys and gpu_blocks must describe the same blocks")
        blocks = []
        for key in keys:
            block = self.pool.get_cached_block(key)
            if block is None:
                # One gone, nothing pinned: the caller recomputes the prompt.
                return None
            blocks.append(block)
        self.pool.touch(blocks)
        self._pinned_cached += len(blocks)

        moves = tuple(
            CopyRun(gpu_block=g, cpu_block=b.block_id)
            for b, g in zip(blocks, gpu_blocks, strict=True)
        )
        event = self._copier.submit(moves, is_store=False)
        self._pending.append(
            _Transfer(
                kind="load",
                keys=tuple(keys),
                cpu_blocks=tuple(m.cpu_block for m in moves),
                gpu_blocks=tuple(gpu_blocks),
                event=event,
                moves=moves,
            )
        )
        return LoadSpec(
            cpu_blocks=tuple(m.cpu_block for m in moves),
            gpu_blocks=tuple(gpu_blocks),
            keys=tuple(keys),
        )

    def complete_load(self, keys: Sequence[OffloadKey]) -> None:
        """Unpin the blocks a load's keys cover, making them evictable again."""
        transfer = self._pop_pending("load", keys)
        if transfer is None:
            return
        self.pool.free_blocks(self.pool.blocks[c] for c in transfer.cpu_blocks)
        self._pinned_cached -= len(transfer.cpu_blocks)
        self._stats.loads += len(transfer.keys)
        self._stats.load_bytes += len(transfer.keys) * self.bytes_per_block

    # ----------------------------------------------------------------- store #
    def prepare_store(self, entries: Sequence[tuple[int, OffloadKey]]) -> StoreSpec:
        """Copy fresh GPU blocks down: ``entries`` is ``(gpu_block, key)``.

        Skips keys already resident or already in flight, so a block that a
        second request commits under the same hash costs one copy, not two.
        A full CPU pool evicts its least-recently-used cached blocks to make
        room and reports them in the spec; pinned blocks are never chosen,
        because the free queue is only reachable through zero references.
        """
        fresh: list[tuple[int, OffloadKey]] = []
        for gpu_block, key in entries:
            if key in self._inflight_stores or self.pool.get_cached_block(key) is not None:
                continue
            fresh.append((gpu_block, key))
        if not fresh:
            return StoreSpec()

        # Predict whether this allocation will have to evict, precisely:
        # blocks are handed out front first, hash-less free blocks sit at the
        # front by construction, and the cached ones not in the free queue are
        # exactly the pinned ones. Snapshotting the index is what lets the
        # spec name the casualties, and it costs O(cached) — so only do it
        # when the count says some will actually be touched.
        cached_free = self.pool.num_cached_blocks - self._pinned_cached
        hash_less_free = self.pool.num_free_blocks - cached_free
        snapshot = (
            set(self.pool.cached_block_hash_to_block) if len(fresh) > hash_less_free else None
        )
        blocks = self.pool.get_new_blocks(len(fresh))
        if blocks is None:
            # Every free block is pinned by an in-flight load. Dropping this
            # store costs a re-store later, never correctness.
            return StoreSpec()
        evicted = (
            () if snapshot is None else tuple(snapshot - set(self.pool.cached_block_hash_to_block))
        )

        moves = tuple(
            CopyRun(gpu_block=gpu_block, cpu_block=block.block_id)
            for (gpu_block, _), block in zip(fresh, blocks, strict=True)
        )
        event = self._copier.submit(moves, is_store=True)
        transfer = _Transfer(
            kind="store",
            keys=tuple(key for _, key in fresh),
            cpu_blocks=tuple(m.cpu_block for m in moves),
            gpu_blocks=tuple(gpu_block for gpu_block, _ in fresh),
            event=event,
            moves=moves,
        )
        self._pending.append(transfer)
        for key in transfer.keys:
            self._inflight_stores[key] = transfer
        return StoreSpec(moves=moves, keys=transfer.keys, evicted=evicted)

    def complete_store(self, keys: Sequence[OffloadKey]) -> None:
        """Index the blocks a store's keys cover, then unpin them.

        Indexing is what makes the keys loadable, so it happens here and not
        at submit time: a block indexed early would be offered to the next
        lookup while its rows are still being written.
        """
        transfer = self._pop_pending("store", keys)
        if transfer is None:
            return
        for key, cpu_block in zip(transfer.keys, transfer.cpu_blocks, strict=True):
            self._inflight_stores.pop(key, None)
            block = self.pool.blocks[cpu_block]
            self.pool.cache_full_blocks([block], [key])
            self.pool.free_blocks([block])
        self._stats.stores += len(transfer.keys)
        self._stats.store_bytes += len(transfer.keys) * self.bytes_per_block

    # ---------------------------------------------------------------- events #
    def take_events(self) -> list[OffloadEvent]:
        """Mark every landed transfer complete and return its event.

        Called once per engine step. A transfer whose copier returned None
        (synchronous stand-ins, and any future inline path) is complete the
        moment it is submitted, which falls out of the same check.
        """
        events: list[OffloadEvent] = []
        pending: list[_Transfer] = []
        for transfer in self._pending:
            if not self._is_done(transfer):
                pending.append(transfer)
                continue
            if transfer.kind == "store":
                self.complete_store(transfer.keys)
            else:
                self.complete_load(transfer.keys)
            events.append(
                OffloadEvent(
                    kind=transfer.kind,
                    keys=transfer.keys,
                    cpu_blocks=transfer.cpu_blocks,
                    gpu_blocks=transfer.gpu_blocks,
                )
            )
        self._pending = pending
        return events

    def pending_store_events(self) -> list[Any]:
        """Events of in-flight stores, for a compute stream to wait on.

        A GPU block evicted from the prefix cache is free to be overwritten by
        the next request the moment the pool hands it out — but a store that
        has not read it yet must read the *old* contents. Waiting on these
        events before issuing compute is what keeps the two ordered; at a few
        microseconds per block they are almost always already signaled.
        """
        return [t.event for t in self._pending if t.kind == "store" and t.event is not None]

    def has_pending_work(self) -> bool:
        """True while any store or load is still in flight."""
        return bool(self._pending)

    # -------------------------------------------------------------- lifecycle #
    def reset_cache(self) -> bool:
        """Drop every offloaded block; refuses while transfers are in flight.

        Landed-but-undrained transfers are drained first, so a caller that has
        waited for its copies and calls this to tear down gets a truthful
        answer about whether anything is still moving.
        """
        self.take_events()
        if self._pending:
            return False
        return self.pool.reset_prefix_cache()

    def get_stats(self) -> OffloadingStats:
        """Cumulative counters, with the CPU pool's live evictions folded in."""
        self._stats.evictions = self.pool.stats.evictions
        return self._stats

    # ---------------------------------------------------------------- helpers #
    def _is_done(self, transfer: _Transfer) -> bool:
        if transfer.event is None:
            return True
        return bool(transfer.event.query())

    def _pop_pending(self, kind: str, keys: Sequence[OffloadKey]) -> _Transfer | None:
        """Remove and return the pending transfer of *kind* covering *keys*."""
        wanted = tuple(keys)
        for index, transfer in enumerate(self._pending):
            if transfer.kind == kind and transfer.keys == wanted:
                return self._pending.pop(index)
        return None
