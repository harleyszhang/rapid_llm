"""KV cache groups: one layout and one cache-hit policy per attention kind.

A *group* is a set of layers that share a KV cache layout and an attention
kind, so they can share one block table. :class:`KVCacheSpec` and its subclasses
describe a group's layout and, crucially, how far back a cached prefix is
reusable for that kind of attention — full attention reuses a prefix from token
0, a sliding window only needs the tail. :class:`KVCacheCoordinator` runs N
groups over one :class:`~rapid_llm.engine.block_pool.BlockPool` and reduces
their answers to the single number the scheduler wants.

Usage:
    config = KVCacheConfig.from_model_config(model_config, tp_size)
    coord = KVCacheCoordinator(pool, config.groups)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .block_pool import BlockPool, KVCacheBlock


def cdiv(a: int, b: int) -> int:
    """Ceiling division, for block counts."""
    return -(-a // b)


@dataclass(frozen=True)
class KVCacheSpec:
    """Layout and hit policy of one KV cache group.

    Attributes:
        block_size: Tokens per block in this group. Must be a multiple of the
            hash block size: a group block is hittable exactly when every hash
            block inside it is, and the chained hash of the last one names all
            of them.
        kv_row: ``(slots, dim)`` of one token's per-layer cache row —
            ``(2 * kv_heads, head_dim)`` for MHA/GQA, ``(1, latent_dim)`` for
            MLA. Two groups with different rows need different buffers, which
            is the layout half of what makes them separate groups. Defaults to
            ``(0, 0)``: only the executor sizes buffers, and the scheduler-side
            allocator works in blocks, so a host-only config need not know the
            model's geometry to be valid.
    """

    block_size: int
    kv_row: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")

    @property
    def kind(self) -> str:
        """Short name of the attention kind, for logs and errors."""
        return type(self).__name__.removesuffix("Spec").lower()

    @property
    def window(self) -> int:
        """Positions this group's attention can reach back; ``0`` means all."""
        return 0

    def page_size_bytes(self, num_layers: int, dtype_size: int) -> int:
        """Bytes one block of this group occupies across all its layers."""
        slots, dim = self.kv_row
        return self.block_size * slots * dim * num_layers * dtype_size

    def num_blocks_for(self, num_tokens: int) -> int:
        """Blocks needed to hold *num_tokens* tokens of one sequence."""
        return cdiv(num_tokens, self.block_size)

    def group_block_hashes(
        self, block_hashes: Sequence[int], hash_block_size: int
    ) -> Sequence[int]:
        """Project hash-granular hashes onto this group's block granularity.

        A group whose ``block_size`` is ``r`` hash blocks wide is hittable at
        block ``j`` only if all ``r`` hash blocks inside it are, and the chained
        hash of the last of them already names every hash block before it — so
        every ``r``-th hash *is* the group block's identity, no rehashing
        needed. This is what lets one prompt-level hash chain serve groups with
        different page sizes.

        Raises:
            ValueError: ``block_size`` is not a multiple of ``hash_block_size``.
        """
        if self.block_size % hash_block_size:
            raise ValueError(
                f"group block_size {self.block_size} must be a multiple of the hash "
                f"block size {hash_block_size}"
            )
        ratio = self.block_size // hash_block_size
        if ratio == 1:
            return block_hashes
        return block_hashes[ratio - 1 :: ratio]

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[int],
        max_length: int,
        pool: BlockPool,
        hash_block_size: int,
    ) -> list[KVCacheBlock]:
        """Return the blocks a prompt with these hashes may reuse.

        Args:
            block_hashes: Chained hash per *hash* block of the prompt, in
                prefix order.
            max_length: Ceiling on the hit in tokens; the caller has already
                reserved the last token for recomputation.
            pool: Where cached blocks are looked up.
            hash_block_size: Tokens per entry of ``block_hashes``.

        Returns:
            One block per hit position, in prefix order. Entries may be the
            pool's null block for a group whose attention cannot reach them
            (see :class:`SlidingWindowSpec`); the *length* of the list times
            :attr:`block_size` is the hit in tokens either way.
        """
        raise NotImplementedError


def _prefix_run_hit(
    spec: KVCacheSpec,
    block_hashes: Sequence[int],
    max_length: int,
    pool: BlockPool,
    hash_block_size: int,
) -> list[KVCacheBlock]:
    """Longest unbroken run of cached blocks from block 0 — full attention's rule.

    Shared by every spec whose queries read the sequence's rows from 0 up, which
    is both MHA/GQA and MLA: they differ in layout, not in what a hit means.
    """
    hashes = spec.group_block_hashes(block_hashes, hash_block_size)
    num_blocks = min(max_length // spec.block_size, len(hashes))
    hit: list[KVCacheBlock] = []
    for index in range(num_blocks):
        block = pool.get_cached_block(hashes[index])
        if block is None:
            break
        hit.append(block)
    # No finer-grained probing inside the first missing block: a block table
    # shares whole blocks, so half a block of matching tokens is not something
    # this layout can express. vLLM probes sub-blocks because its hybrid
    # allocator may hash finer than it pages; here the group's own hashes are
    # already at page granularity.
    return hit


@dataclass(frozen=True)
class FullAttentionSpec(KVCacheSpec):
    """MHA/GQA layers: every query attends over the whole history.

    A hit must be an unbroken run from token 0, because attention reads a
    sequence's rows from 0 up: the first missing block ends the reuse, whatever
    is cached behind it.
    """

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[int],
        max_length: int,
        pool: BlockPool,
        hash_block_size: int,
    ) -> list[KVCacheBlock]:
        return _prefix_run_hit(self, block_hashes, max_length, pool, hash_block_size)


@dataclass(frozen=True)
class MLASpec(KVCacheSpec):
    """DeepSeek-style latent attention: full attention over a latent row.

    The hit policy is full attention's — the layout is what differs. The latent
    row has no head axis to shard, so every tensor-parallel rank holds all of
    it, and a group of MLA layers therefore cannot share a buffer with a group
    of MHA layers even at the same block size.
    """

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[int],
        max_length: int,
        pool: BlockPool,
        hash_block_size: int,
    ) -> list[KVCacheBlock]:
        return _prefix_run_hit(self, block_hashes, max_length, pool, hash_block_size)


@dataclass(frozen=True)
class SlidingWindowSpec(KVCacheSpec):
    """Layers whose query only attends over the last ``sliding_window`` tokens.

    Two consequences, both different from full attention. Blocks that fall out
    of the window are freed while the request is still running, so a group like
    this bounds its own footprint. And a cache hit does *not* have to start at
    token 0: what a resuming query needs is the window's worth of blocks ending
    where it resumes, so the search runs right to left and the positions before
    the run are filled with the pool's null block — they exist in the table but
    are never read.

    Attributes:
        sliding_window: Positions a query can reach back, itself included.
    """

    sliding_window: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.sliding_window < 1:
            raise ValueError(f"sliding_window must be >= 1, got {self.sliding_window}")

    @property
    def window(self) -> int:
        return self.sliding_window

    @property
    def num_window_blocks(self) -> int:
        """Blocks a hit must cover to reconstruct a full window."""
        # A query at position p reads [p - window + 1, p]. The oldest position
        # it needs is window - 1 back, so that many tokens plus the query's own
        # block is what a hit has to hold.
        return cdiv(self.sliding_window - 1, self.block_size) + 1

    def num_blocks_for(self, num_tokens: int) -> int:
        """Blocks addressed by a sequence of this length — *not* live blocks.

        Deliberately uncapped even though only :attr:`num_window_blocks` of them
        hold anything: a block table entry is indexed by absolute position, so
        position ``p`` must keep addressing table slot ``p`` however far the
        window has moved on. What bounds this group's real footprint is
        :meth:`KVCacheCoordinator.remove_skipped_blocks`, which hands the blocks
        below the window back to the pool and leaves null entries behind.
        """
        return cdiv(num_tokens, self.block_size)

    def find_longest_cache_hit(
        self,
        block_hashes: Sequence[int],
        max_length: int,
        pool: BlockPool,
        hash_block_size: int,
    ) -> list[KVCacheBlock]:
        hashes = self.group_block_hashes(block_hashes, hash_block_size)
        num_blocks = min(max_length // self.block_size, len(hashes))
        needed = self.num_window_blocks
        hit: list[KVCacheBlock] = [pool.null_block] * num_blocks
        run = 0
        for index in range(num_blocks - 1, -1, -1):
            block = pool.get_cached_block(hashes[index])
            if block is None:
                run = 0
                continue
            hit[index] = block
            run += 1
            if run >= needed:
                # A full window ending at ``index + run - 1``: everything past
                # it is a miss, so trim there and keep the nulls in front.
                del hit[index + run :]
                return hit
        # Never assembled a whole window. The tail run is still reusable — it is
        # what a query at its end can see — so keep exactly that.
        del hit[run:]
        return hit


@dataclass(frozen=True)
class KVCacheGroup:
    """One group: its spec, the layers in it, and its position in the config.

    Attributes:
        group_id: Index of this group; also the index of its block table.
        spec: Layout and hit policy.
        layer_ids: Decoder layer indices belonging to this group.
    """

    group_id: int
    spec: KVCacheSpec
    layer_ids: tuple[int, ...]


@dataclass(frozen=True)
class KVCacheConfig:
    """The groups a model's layers fall into.

    Attributes:
        groups: One :class:`KVCacheGroup` per layout/attention kind, in group
            order. Today's rapid_llm models are homogeneous, so this holds one
            group; the machinery does not assume that anywhere.
        hash_block_size: Tokens per entry of a prompt's hash chain. Every
            group's ``block_size`` is a multiple of it.
    """

    groups: tuple[KVCacheGroup, ...]
    hash_block_size: int

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("a KV cache config needs at least one group")
        for group in self.groups:
            group.spec.group_block_hashes((), self.hash_block_size)  # validates divisibility

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def is_homogeneous(self) -> bool:
        """Whether one group covers every layer — the fast path everywhere."""
        return len(self.groups) == 1

    def group_of_layer(self, layer_index: int) -> int:
        """Group id owning *layer_index*.

        Raises:
            KeyError: The layer belongs to no group.
        """
        for group in self.groups:
            if layer_index in group.layer_ids:
                return group.group_id
        raise KeyError(f"layer {layer_index} belongs to no KV cache group")

    @classmethod
    def homogeneous(
        cls,
        block_size: int,
        num_layers: int = 1,
        kv_row: tuple[int, int] = (0, 0),
        hash_block_size: int | None = None,
    ) -> KVCacheConfig:
        """One full-attention group over every layer.

        The default for a caller that allocates blocks without knowing the
        model's geometry -- the scheduler, and every host-side test. It is not a
        degenerate case: every non-MLA rapid_llm model in tree produces exactly
        this config.
        """
        spec = FullAttentionSpec(block_size=block_size, kv_row=kv_row)
        group = KVCacheGroup(0, spec, tuple(range(num_layers)))
        return cls((group,), hash_block_size or block_size)

    @classmethod
    def from_model_config(
        cls,
        config,
        tp_size: int = 1,
        block_size: int = 16,
        hash_block_size: int | None = None,
    ) -> KVCacheConfig:
        """Derive the groups from a :class:`~rapid_llm.models.config.ModelConfig`.

        MLA checkpoints get one latent group, everything else one full-attention
        group; a config that declares ``sliding_window`` together with a set of
        windowed layer indices (``sliding_window_layers``) splits into two. The
        split is driven by the checkpoint rather than by a flag because it is a
        property of the model, not a deployment choice.

        Args:
            config: The model config, already normalised across HF field names.
            tp_size: Tensor-parallel world size; MHA/GQA rows are sharded by it.
            block_size: Tokens per block for every group.
            hash_block_size: Tokens per hash-chain entry; defaults to
                ``block_size``.
        """
        hash_block_size = hash_block_size or block_size
        num_layers = config.num_layers
        all_layers = tuple(range(num_layers))

        if getattr(config, "is_mla", False):
            spec: KVCacheSpec = MLASpec(
                block_size=block_size,
                kv_row=(1, config.kv_lora_rank + config.qk_rope_head_dim),
            )
            return cls((KVCacheGroup(0, spec, all_layers),), hash_block_size)

        kv_heads, remainder = divmod(config.num_kv_heads, tp_size)
        if remainder or kv_heads == 0:
            raise ValueError(
                f"{config.num_kv_heads} key/value heads do not divide across {tp_size} ranks"
            )
        full_row = (2 * kv_heads, config.head_dim)
        window = getattr(config, "sliding_window", None) or 0
        windowed = tuple(getattr(config, "sliding_window_layers", ()) or ())
        if not window or not windowed:
            full = FullAttentionSpec(block_size=block_size, kv_row=full_row)
            return cls((KVCacheGroup(0, full, all_layers),), hash_block_size)

        dense = tuple(i for i in all_layers if i not in set(windowed))
        groups = [
            KVCacheGroup(
                0, SlidingWindowSpec(block_size, full_row, sliding_window=window), tuple(windowed)
            )
        ]
        if dense:
            groups.append(KVCacheGroup(1, FullAttentionSpec(block_size, full_row), dense))
        return cls(tuple(groups), hash_block_size)


@dataclass
class _RequestBlocks:
    """A request's blocks, per group, in prefix order.

    Attributes:
        per_group: ``per_group[g][j]`` is the block holding group ``g``'s block
            ``j`` of this sequence. Entries may be the null block for a
            windowed group whose older blocks have been released.
        num_cached: Blocks already indexed by hash per group, so
            :meth:`KVCacheCoordinator.cache_blocks` can resume where it left
            off instead of rescanning the prompt every step.
    """

    per_group: list[list[KVCacheBlock]] = field(default_factory=list)
    num_cached: list[int] = field(default_factory=list)


class KVCacheCoordinator:
    """Runs every KV cache group over one block pool.

    The scheduler asks about *tokens* and the pool answers in *blocks*; this is
    where the two meet, and where N groups are reduced to one answer: a prompt's
    reusable prefix is the shortest hit across groups, because a step computes
    all layers or none.

    Args:
        pool: The physical block pool the groups draw from.
        groups: Groups in group-id order.
        hash_block_size: Tokens per entry of a prompt's hash chain.
    """

    def __init__(
        self,
        pool: BlockPool,
        groups: Sequence[KVCacheGroup],
        hash_block_size: int,
    ) -> None:
        self.pool = pool
        self.groups = tuple(groups)
        self.hash_block_size = hash_block_size
        self._requests: dict[str, _RequestBlocks] = {}

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    # ----------------------------------------------------------------- lookup #
    def find_longest_cache_hit(
        self, block_hashes: Sequence[int], max_length: int
    ) -> tuple[list[list[KVCacheBlock]], int]:
        """Longest prefix every group can serve, and its length in tokens.

        The result is aligned down to the coarsest group block size, so the
        length is a whole number of blocks in *every* group — a partially
        covered block in one group would have no table entry to point at.

        Returns:
            ``(per-group hit blocks, num_computed_tokens)``. The token count is
            zero when nothing is reusable, in which case the block lists are
            empty too.
        """
        hits = [
            group.spec.find_longest_cache_hit(
                block_hashes, max_length, self.pool, self.hash_block_size
            )
            for group in self.groups
        ]
        length = min(
            len(hit) * group.spec.block_size for hit, group in zip(hits, self.groups, strict=True)
        )
        if len(self.groups) > 1:
            stride = max(group.spec.block_size for group in self.groups)
            length -= length % stride
        if length == 0:
            return [[] for _ in self.groups], 0
        for hit, group in zip(hits, self.groups, strict=True):
            del hit[length // group.spec.block_size :]
        return hits, length

    # ------------------------------------------------------------- allocation #
    def allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[Sequence[KVCacheBlock]] | None = None,
    ) -> bool:
        """Make sure *request_id* owns blocks covering its first *num_tokens*.

        Called both at admission (with the hit blocks this request adopts) and
        on every step that pushes a sequence past a block boundary. Reuse is
        recorded by taking a reference on the shared block, not by copying its
        rows — the block table is the indirection that makes that legal.

        Returns:
            True when the request's blocks now cover ``num_tokens``. False when
            the pool is short, in which case *nothing* was allocated: a partial
            allocation would leave the caller having to unwind it, and every
            caller's answer to "no room" is to preempt or wait.
        """
        state = self._requests.setdefault(request_id, _RequestBlocks())
        if not state.per_group:
            state.per_group = [[] for _ in self.groups]
            state.num_cached = [0 for _ in self.groups]

        adopted = list(new_computed_blocks) if new_computed_blocks else []
        if adopted:
            if len(adopted) != len(self.groups):
                raise ValueError("new_computed_blocks must have one entry per group")
            for group_id, blocks in enumerate(adopted):
                if state.per_group[group_id]:
                    raise ValueError(
                        f"request {request_id} already holds blocks in group {group_id}"
                    )
                # Referenced before the fresh blocks are drawn, not after: an
                # adopted block whose ref_cnt is still zero sits in the free
                # queue, and get_new_blocks would happily hand out the very rows
                # this request is adopting.
                self.pool.touch(blocks)
                state.per_group[group_id] = list(blocks)
                # Adopted blocks are cached by definition, so caching may resume
                # after them rather than walking them again.
                state.num_cached[group_id] = len(blocks)

        # Size the whole request first: a group that cannot be served must not
        # leave the other groups half-extended.
        needed = [
            max(group.spec.num_blocks_for(num_tokens) - len(blocks), 0)
            for group, blocks in zip(self.groups, state.per_group, strict=True)
        ]
        total = sum(needed)
        if total == 0:
            return True
        fresh = self.pool.get_new_blocks(total)
        if fresh is None:
            if adopted:
                # Give the references back, so a rejected admission leaves the
                # pool exactly as it found it. The blocks re-enter the free queue
                # at its tail rather than where they sat, which costs a little
                # LRU precision and nothing else -- they are still cached, and
                # still hittable.
                self.free(request_id)
            return False
        cursor = 0
        for blocks, count in zip(state.per_group, needed, strict=True):
            blocks += fresh[cursor : cursor + count]
            cursor += count
        return True

    def cache_blocks(
        self, request_id: str, block_hashes: Sequence[int], num_computed_tokens: int
    ) -> list[tuple[int, int, int]]:
        """Index the request's blocks that are now fully computed.

        Only whole blocks below ``num_computed_tokens`` are indexed, and only
        once: a block that already carries a hash is skipped, so calling this
        every step costs work proportional to the *new* blocks.

        Callers must pass a token count the model has actually executed. A block
        indexed before its K/V is written would be offered to the next
        admission as readable rows, and the reader would attend over whatever
        was in the cache before.

        Returns:
            The blocks that actually gained their hash, as
            ``(group_id, block_id, block_hash)`` triples in indexing order —
            what an offloading tier mirrors down. Empty on the common decode
            step, where nothing new became indexable.
        """
        state = self._requests.get(request_id)
        if state is None or not state.per_group:
            return []
        fresh: list[tuple[int, int, int]] = []
        for index, group in enumerate(self.groups):
            spec = group.spec
            hashes = spec.group_block_hashes(block_hashes, self.hash_block_size)
            full = min(num_computed_tokens // spec.block_size, len(hashes))
            start = state.num_cached[index]
            if full <= start:
                continue
            blocks = state.per_group[index][start:full]
            if len(blocks) < full - start:
                continue  # blocks not allocated yet; a later step will index them
            gained = [
                (block.block_id, block_hash)
                for block, block_hash in zip(blocks, hashes[start:full], strict=True)
                if block.block_hash is None
            ]
            self.pool.cache_full_blocks(blocks, hashes[start:full])
            fresh.extend((index, block_id, block_hash) for block_id, block_hash in gained)
            state.num_cached[index] = full
        return fresh

    def free(self, request_id: str) -> None:
        """Release everything *request_id* holds, tail blocks first.

        Tail first because the head of a prompt is the part another request is
        likely to share: freeing in reverse puts the tail nearer the front of
        the free queue, so pressure eats the least reusable blocks first.
        """
        state = self._requests.pop(request_id, None)
        if state is None:
            return
        for blocks in state.per_group:
            self.pool.free_blocks(reversed(blocks))

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        """Release windowed groups' blocks that have fallen out of the window.

        A sliding-window group only ever reads the last ``window`` positions, so
        blocks entirely below that are dead the moment the sequence passes them.
        Their table entries are repointed at the null block by the caller, which
        is why they can be released while the request runs.

        No-op for full-attention and MLA groups, whose every block stays live.
        """
        state = self._requests.get(request_id)
        if state is None:
            return
        for group, blocks in zip(self.groups, state.per_group, strict=True):
            window = group.spec.window
            if not window:
                continue
            block_size = group.spec.block_size
            # The oldest position still readable, floored to its block: that
            # block is live, everything strictly before it is not.
            keep_from = min(max(num_computed_tokens - window, 0) // block_size, len(blocks))
            dead = [b for b in blocks[:keep_from] if b is not self.pool.null_block]
            if not dead:
                continue
            for index in range(keep_from):
                blocks[index] = self.pool.null_block
            self.pool.free_blocks(reversed(dead))

    # ------------------------------------------------------------------ views #
    def block_ids(self, request_id: str) -> tuple[tuple[int, ...], ...]:
        """Per-group block ids of a request, in prefix order."""
        state = self._requests.get(request_id)
        if state is None:
            return tuple(() for _ in self.groups)
        return tuple(tuple(b.block_id for b in blocks) for blocks in state.per_group)

    def num_tracked_requests(self) -> int:
        """Requests currently holding blocks — a leak check for tests."""
        return len(self._requests)
