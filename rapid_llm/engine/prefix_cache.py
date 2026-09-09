"""Prefix caching: reuse the KV of shared prefixes by sharing physical blocks.

:class:`PrefixCache` is the scheduler's whole view of the KV cache: it hashes a
sequence in fixed-size blocks, looks the hashes up in a
:class:`~rapid_llm.engine.block_pool.BlockPool`, and hands back the *physical
blocks* a matching prefix already lives in — reuse is a reference on a shared
block, not a copy (the executor's block table is the indirection letting two
sequences read the same rows). Generated tokens join the hash chain, so a
finished request's whole sequence is reusable by the next prompt that starts
with it — that is what makes multi-turn cheap.

Usage:
    cache = PrefixCache(num_blocks=1024)
    hashes = cache.hash_tokens(prompt); match = cache.lookup(hashes, len(prompt))
    cache.allocate(rid, len(prompt), match); cache.free(rid)
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator, MutableSequence, Sequence
from dataclasses import dataclass, field

from .block_pool import BlockPool, KVCacheBlock
from .kv_cache_spec import KVCacheConfig, KVCacheCoordinator, KVCacheGroup

#: Tokens per prefix-cache block; 16 mirrors vLLM's default page granularity.
#: It lives here rather than on the scheduler because everyone who computes a
#: block hash must agree on it -- the replica's cache and the DP router's
#: affinity index -- and only one of those owns a scheduler.
PREFIX_CACHE_BLOCK_SIZE = 16


def iter_block_hashes(token_ids: Sequence[int], block_size: int, seed: int = 0) -> Iterator[int]:
    """Yield one chained hash per full block of *token_ids*.

    Each block's hash folds in the previous block's hash (and a per-cache
    ``seed``), so identical hashes imply identical prefixes rather than merely
    identical block contents at some offset. A trailing partial block is
    skipped: a half-filled block's KV is not reusable until it is complete.

    The digest is ``blake2b`` rather than the builtin ``hash()`` because these
    values are a *cross-process contract*: the DP router hashes a prompt to
    find which replica already holds its prefix, and the replica hashes it
    again. A router that disagreed with its replicas would not raise — it
    would quietly route every request as a miss.

    Args:
        token_ids: Prompt tokens, each fitting in 32 bits.
        block_size: Tokens per block. Must match every party that hashes.
        seed: Salt folded into the chain; must match every party that hashes.

    Yields:
        One 64-bit hash per complete block, in prefix order.

    Raises:
        struct.error: A token id is negative or does not fit in 32 bits.
    """
    parent = seed & 0xFFFFFFFFFFFFFFFF
    layout = f"<Q{block_size}I"
    num_full = len(token_ids) // block_size
    for b in range(num_full):
        block = token_ids[b * block_size : (b + 1) * block_size]
        digest = hashlib.blake2b(struct.pack(layout, parent, *block), digest_size=8)
        parent = int.from_bytes(digest.digest(), "little")
        yield parent


def extend_block_hashes(
    hashes: MutableSequence[int],
    token_ids: Sequence[int],
    block_size: int,
    seed: int = 0,
) -> None:
    """Append the hashes of blocks *token_ids* has completed since last time.

    The chain is a running digest, so a sequence that has grown by one token
    only needs rehashing when that token completed a block — and then only that
    block. This is what keeps generated tokens cacheable without rehashing the
    whole sequence on every decode step: appending one token costs nothing 15
    steps out of 16, and one blake2b on the sixteenth.

    Args:
        hashes: The chain so far, extended in place.
        token_ids: The whole sequence, prompt and output together.
        block_size: Tokens per block.
        seed: Salt folded into the chain.
    """
    num_full = len(token_ids) // block_size
    parent = hashes[-1] if hashes else (seed & 0xFFFFFFFFFFFFFFFF)
    layout = f"<Q{block_size}I"
    for b in range(len(hashes), num_full):
        block = token_ids[b * block_size : (b + 1) * block_size]
        digest = hashlib.blake2b(struct.pack(layout, parent, *block), digest_size=8)
        parent = int.from_bytes(digest.digest(), "little")
        hashes.append(parent)


@dataclass
class PrefixCacheStats:
    """Cumulative counters for prefix-cache effectiveness (mirrors vLLM's stats).

    Attributes:
        num_requests: How many sequences were looked up.
        queried_tokens: Total tokens looked up.
        hit_tokens: Tokens served from cache.
        evictions: Cached blocks dropped so their rows could be reused.
    """

    num_requests: int = 0
    queried_tokens: int = 0
    hit_tokens: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of queried tokens served from cache (0.0 - 1.0)."""
        if self.queried_tokens == 0:
            return 0.0
        return self.hit_tokens / self.queried_tokens


@dataclass(frozen=True)
class PrefixMatch:
    """The physical blocks a sequence may reuse.

    Attributes:
        num_tokens: Leading tokens covered by the hit, block-aligned in every
            group. Unlike the copy-based scheme this replaces, it is exactly
            what prefill may skip: a hit *is* a physical block, so there is no
            longer a difference between "cached" and "readable".
        blocks: One tuple of blocks per KV cache group, in prefix order. The
            caller passes them straight back to :meth:`PrefixCache.allocate`,
            which is where the references are taken.
    """

    num_tokens: int = 0
    blocks: tuple[tuple[KVCacheBlock, ...], ...] = ()

    def __bool__(self) -> bool:
        return self.num_tokens > 0


@dataclass
class _Sequence:
    """What the cache remembers about one live sequence.

    Attributes:
        block_hashes: Chained hash per completed block of prompt + output,
            extended in place as the sequence grows.
        num_cached_tokens: Tokens whose blocks are already indexed by hash, so
            each step only indexes what is new.
        num_mapped_blocks: Blocks per group the executor has already been told
            to map, so each step emits only the blocks the sequence just grew
            into. A steady decode step therefore emits nothing at all.
    """

    block_hashes: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    num_mapped_blocks: list[int] = field(default_factory=list)


class PrefixCache:
    """Block-level KV allocator with hash-based reuse across sequences.

    Every sequence's KV lives in blocks drawn from one
    :class:`~rapid_llm.engine.block_pool.BlockPool`, and a prefix two sequences
    share is one set of blocks both of them reference. Nothing is copied, and a
    block survives exactly as long as someone holds it — after that it stays
    cached and hittable until its rows are needed for something else.

    The class is host-side and device-free: it hands out block *ids*, and the
    executor turns them into block-table entries. That split is what lets the
    whole thing be tested without a GPU.

    Args:
        num_blocks: Physical blocks in the pool, the null block included. The
            executor derives it from the KV cache it actually allocated.
        block_size: Tokens per block; must match every party that hashes.
        kv_cache_config: The groups to allocate for. Defaults to one
            full-attention group, which is what every homogeneous model needs.
        hash_seed: Salt folded into every block hash, isolating one cache's
            hashes from another's. Under data parallelism every replica — and
            the router's affinity index — must share the seed.
        enable_caching: When False, blocks are still allocated and recycled but
            never indexed, so nothing is reused across sequences. This is
            "prefix caching off", and it is the same code path rather than a
            parallel one.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = PREFIX_CACHE_BLOCK_SIZE,
        kv_cache_config: KVCacheConfig | None = None,
        hash_seed: int = 0,
        enable_caching: bool = True,
    ) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")
        self.block_size = block_size
        self.hash_seed = hash_seed
        self.enable_caching = enable_caching
        self.config = kv_cache_config or KVCacheConfig.homogeneous(block_size)
        self.pool = BlockPool(num_blocks, block_size, enable_caching=enable_caching)
        self.coordinator = KVCacheCoordinator(
            self.pool, self.config.groups, self.config.hash_block_size
        )
        self._stats = PrefixCacheStats()
        # Evictions are the pool's counter, not ours; a reset rebases rather than
        # zeroing it, so the pool stays the single place that counts them.
        self._eviction_base = 0
        self._sequences: dict[str, _Sequence] = {}

    @property
    def stats(self) -> PrefixCacheStats:
        """Cumulative counters, with the pool's live eviction count folded in.

        Read through rather than mirrored on every eviction: an eviction happens
        deep inside an allocation, and a counter copied at some later call would
        read as zero for however long nobody made that call.
        """
        self._stats.evictions = self.pool.stats.evictions - self._eviction_base
        return self._stats

    # ---------------------------------------------------------------- hashing #
    def hash_tokens(self, token_ids: Sequence[int]) -> list[int]:
        """Return the chained hash of every complete block of *token_ids*."""
        return list(iter_block_hashes(token_ids, self.block_size, self.hash_seed))

    def track(self, request_id: str, token_ids: Sequence[int]) -> list[int]:
        """Start (or resume) tracking a sequence, returning its hash chain.

        The chain is owned by the cache rather than recomputed per call, because
        :meth:`observe` extends it one block at a time as the sequence generates.
        """
        state = self._sequences.get(request_id)
        if state is None:
            state = _Sequence(block_hashes=self.hash_tokens(token_ids))
            self._sequences[request_id] = state
        else:
            extend_block_hashes(state.block_hashes, token_ids, self.block_size, self.hash_seed)
        return state.block_hashes

    def observe(self, request_id: str, token_ids: Sequence[int]) -> list[int]:
        """Extend a tracked sequence's chain over tokens it has since produced.

        This is the decode-caching entry point: a generated token that completes
        a block gives that block a hash, and once the block is indexed (see
        :meth:`commit`) the next prompt starting with this whole sequence hits
        it. Untracked sequences are ignored, so a caller need not special-case
        requests admitted before the cache was created.
        """
        state = self._sequences.get(request_id)
        if state is None:
            return []
        extend_block_hashes(state.block_hashes, token_ids, self.block_size, self.hash_seed)
        return state.block_hashes

    def block_hashes(self, request_id: str) -> list[int]:
        """The tracked hash chain of a sequence, empty when it is not tracked."""
        state = self._sequences.get(request_id)
        return state.block_hashes if state is not None else []

    # ----------------------------------------------------------------- lookup #
    def lookup(self, block_hashes: Sequence[int], num_tokens: int) -> PrefixMatch:
        """Find the longest reusable prefix of a sequence with these hashes.

        Args:
            block_hashes: The sequence's chained block hashes.
            num_tokens: Length of the sequence being admitted.

        Returns:
            The blocks every group can serve, and how many tokens that covers.

        The hit is capped at ``num_tokens - 1``: a request whose every token is
        cached still has to run one of them, because the step that produces its
        next token needs logits, and logits come from a forward pass. vLLM caps
        it the same way and for the same reason.
        """
        self._stats.num_requests += 1
        self._stats.queried_tokens += num_tokens
        if not self.enable_caching or num_tokens < 1:
            return PrefixMatch(0, tuple(() for _ in self.config.groups))
        blocks, hit = self.coordinator.find_longest_cache_hit(block_hashes, max(num_tokens - 1, 0))
        self._stats.hit_tokens += hit
        return PrefixMatch(hit, tuple(tuple(group) for group in blocks))

    # ------------------------------------------------------------- allocation #
    def allocate(self, request_id: str, num_tokens: int, match: PrefixMatch | None = None) -> bool:
        """Give *request_id* blocks covering its first *num_tokens* tokens.

        Call it at admission with the :class:`PrefixMatch` the request adopts,
        and again on any step that pushes the sequence past a block boundary
        (with no match). Adopting a match takes a reference on shared blocks; it
        never copies rows.

        Returns:
            False when the pool cannot cover the request, having allocated
            nothing. The caller decides what that means — preempt, or wait.
        """
        adopted = match.blocks if match is not None and match.num_tokens else None
        return self.coordinator.allocate(request_id, num_tokens, adopted)

    def commit(self, request_id: str, num_computed_tokens: int) -> None:
        """Index the request's blocks whose K/V the model has actually written.

        The token count must be *executed*, not merely scheduled. Under a
        committing scheduler those are different moments: ``num_computed_tokens``
        advances when a chunk is planned, one engine step before its K/V exists,
        and a block indexed at planning time would be handed to the next
        admission as readable rows the model had not written yet.
        """
        state = self._sequences.get(request_id)
        if state is None or num_computed_tokens <= state.num_cached_tokens:
            return
        self.coordinator.cache_blocks(request_id, state.block_hashes, num_computed_tokens)
        state.num_cached_tokens = num_computed_tokens

    def trim_window(self, request_id: str, num_computed_tokens: int) -> None:
        """Release blocks that have fallen out of a windowed group's window."""
        self.coordinator.remove_skipped_blocks(request_id, num_computed_tokens)

    def free(self, request_id: str) -> None:
        """Release every block a request holds and stop tracking it."""
        self.coordinator.free(request_id)
        self._sequences.pop(request_id, None)

    def reset(self) -> bool:
        """Drop every cached block and zero the stats, if nothing is in use.

        Returns False and changes nothing while a live request holds a block:
        see :meth:`~rapid_llm.engine.block_pool.BlockPool.reset_prefix_cache`
        for why a forced reset would leak capacity instead of freeing it.
        """
        if not self.pool.reset_prefix_cache():
            return False
        self._sequences.clear()
        self._stats = PrefixCacheStats()
        self._eviction_base = self.pool.stats.evictions
        return True

    # ------------------------------------------------------------------ views #
    def block_ids(self, request_id: str) -> tuple[tuple[int, ...], ...]:
        """Per-group block ids a request holds, in prefix order."""
        return self.coordinator.block_ids(request_id)

    def take_table_writes(self, request_id: str) -> tuple[tuple[int, int, tuple[int, ...]], ...]:
        """Block-table entries the executor has not been given yet, and mark them given.

        This is the whole device-side effect of prefix reuse: the executor points
        the request's table rows at these physical blocks and the reused K/V is
        readable, with nothing copied.

        Draining rather than reporting, because a block only needs mapping once:
        a table entry covers its block's whole row span the moment it is written,
        so a sequence growing into a block it already mapped has nothing to say.
        That makes a steady decode step free and a boundary-crossing one cost one
        block's worth of int32 writes.

        The cursor lives with the rest of the request's state, so it is dropped by
        :meth:`free` along with the blocks -- a preempted request re-mapping from
        scratch is exactly right, since it also re-allocates from scratch.

        Returns:
            ``(group_id, start_block, block_ids)`` per group with anything to
            map, and nothing for groups that are already up to date.
        """
        state = self._sequences.get(request_id)
        if state is None:
            return ()
        groups = self.config.groups
        if not state.num_mapped_blocks:
            state.num_mapped_blocks = [0] * len(groups)
        writes: list[tuple[int, int, tuple[int, ...]]] = []
        mapped = state.num_mapped_blocks
        held = self.block_ids(request_id)
        for index, (group, blocks) in enumerate(zip(groups, held, strict=True)):
            start = mapped[index]
            if len(blocks) > start:
                writes.append((group.group_id, start, blocks[start:]))
                mapped[index] = len(blocks)
        return tuple(writes)

    def num_committed_tokens(self, request_id: str) -> int:
        """Tokens of a sequence whose blocks are already indexed by hash."""
        state = self._sequences.get(request_id)
        return state.num_cached_tokens if state is not None else 0

    @property
    def groups(self) -> tuple[KVCacheGroup, ...]:
        """The KV cache groups this cache allocates for, in group order."""
        return self.config.groups

    @property
    def hit_rate(self) -> float:
        """Cumulative fraction of queried tokens served from cache."""
        return self._stats.hit_rate

    @property
    def num_blocks(self) -> int:
        """Physical blocks in the pool, the null block included."""
        return self.pool.num_blocks

    @property
    def num_free_blocks(self) -> int:
        """Blocks available right now, cached-but-unreferenced ones included."""
        return self.pool.num_free_blocks

    @property
    def num_cached_blocks(self) -> int:
        """Blocks reachable through the hash index."""
        return self.pool.num_cached_blocks

    @property
    def num_referenced_blocks(self) -> int:
        """Blocks a live request holds; the null block is not counted."""
        return sum(1 for b in self.pool.blocks if b.ref_cnt > 0) - 1

    @property
    def num_evictable_blocks(self) -> int:
        """Cached blocks no live request holds — the eviction candidates."""
        return sum(1 for b in self.pool.blocks if b.ref_cnt == 0 and b.block_hash is not None)

    @property
    def utilization(self) -> float:
        """Fraction of the pool a live request holds (0.0 empty, 1.0 full)."""
        return 1.0 - self.num_free_blocks / self.pool.num_blocks
