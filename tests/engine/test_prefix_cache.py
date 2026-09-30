"""Tests for prefix caching — pure CPU, no GPU or checkpoint required.

The baseline every assertion here rests on is *physical sharing*: a hit hands
back the very blocks the prefix already lives in, so two sequences sharing a
prefix hold identical block ids and nothing is copied. One class per behaviour:
hashing, lookup, allocation and reference counting, the commit contract, LRU
eviction, decode-token caching, multiple KV groups, table writes, stats, and the
scheduler integration.

Usage:
    pytest tests/engine/test_prefix_cache.py
"""

from __future__ import annotations

import pytest

from rapid_llm.engine.batch_planner import BatchPlanner, Request, SchedulerConfig
from rapid_llm.engine.kv_cache_spec import FullAttentionSpec, KVCacheConfig, KVCacheGroup
from rapid_llm.engine.prefix_cache import (
    PREFIX_CACHE_BLOCK_SIZE,
    PrefixCache,
    PrefixCacheStats,
    extend_block_hashes,
    iter_block_hashes,
)
from rapid_llm.engine.sampler import SamplingParams

#: Blocks in a test pool. Big enough that nothing evicts unless a test asks for
#: pressure, small enough that exhaustion is reachable when one does.
_POOL = 64


def _cache(**kwargs) -> PrefixCache:
    kwargs.setdefault("num_blocks", _POOL)
    return PrefixCache(**kwargs)


def _req(request_id: str, token_ids: list[int]) -> Request:
    return Request(
        request_id=request_id,
        prompt="x" * len(token_ids),
        prompt_token_ids=token_ids,
        params=SamplingParams(temperature=0.0, max_gen_len=32),
    )


def _serve(cache: PrefixCache, request_id: str, tokens: list[int]) -> int:
    """Admit a sequence the way the scheduler does, and return its hit length.

    Track, look up, adopt the hit, then commit everything — the commit standing
    in for "the model ran and its K/V landed", which is the only state in which
    a block may be offered to someone else.
    """
    hashes = cache.track(request_id, tokens)
    match = cache.lookup(hashes, len(tokens))
    assert cache.allocate(request_id, len(tokens), match)
    cache.commit(request_id, len(tokens))
    return match.num_tokens


# --------------------------------------------------------------------------- #
# 1. Block hashing
# --------------------------------------------------------------------------- #
class TestHashing:
    """Full blocks only, chained, and salted by the seed."""

    @pytest.mark.parametrize(
        ("num_tokens", "block_size", "expected"),
        [(16, 4, 4), (17, 4, 4), (15, 4, 3), (32, 8, 4), (33, 8, 4), (3, 16, 0)],
    )
    def test_only_complete_blocks_are_hashed(self, num_tokens, block_size, expected):
        assert len(list(iter_block_hashes(list(range(num_tokens)), block_size))) == expected

    def test_the_chain_makes_position_part_of_a_blocks_identity(self):
        """Same 4 tokens, different prefix before them: different hash.

        Without chaining, block ``[4,5,6,7]`` would hash the same wherever it
        appeared, and a "hit" could hand a sequence a block whose K/V was
        computed under a completely different history.
        """
        a = list(iter_block_hashes([0, 1, 2, 3, 4, 5, 6, 7], 4))
        b = list(iter_block_hashes([9, 9, 9, 9, 4, 5, 6, 7], 4))
        assert a[0] != b[0]
        assert a[1] != b[1]  # the shared second block still differs

    def test_a_shared_prefix_shares_its_leading_hashes(self):
        a = list(iter_block_hashes(list(range(32)), 4))
        b = list(iter_block_hashes(list(range(16)) + [99] * 16, 4))
        assert a[:4] == b[:4]
        assert a[4] != b[4]

    def test_the_seed_isolates_one_caches_hashes_from_anothers(self):
        tokens = list(range(32))
        assert list(iter_block_hashes(tokens, 16, seed=0)) != list(
            iter_block_hashes(tokens, 16, seed=1)
        )

    def test_extend_matches_a_full_rehash(self):
        """The incremental chain is the same chain, or decode caching is wrong."""
        tokens = list(range(80))
        chain: list[int] = []
        for length in range(len(tokens) + 1):
            extend_block_hashes(chain, tokens[:length], 16)
        assert chain == list(iter_block_hashes(tokens, 16))

    def test_extend_only_appends_when_a_block_completes(self):
        chain: list[int] = []
        extend_block_hashes(chain, list(range(16)), 16)
        assert len(chain) == 1
        extend_block_hashes(chain, list(range(31)), 16)
        assert len(chain) == 1  # 15 more tokens, still one block short
        extend_block_hashes(chain, list(range(32)), 16)
        assert len(chain) == 2

    def test_a_token_id_wider_than_32_bits_is_rejected(self):
        import struct

        with pytest.raises(struct.error):
            list(iter_block_hashes([1 << 40] * 16, 16))


# --------------------------------------------------------------------------- #
# 2. Lookup
# --------------------------------------------------------------------------- #
class TestLookup:
    """What a sequence may skip, and the one token it may never skip."""

    def test_an_empty_cache_never_hits(self):
        cache = _cache()
        tokens = list(range(64))
        assert cache.lookup(cache.hash_tokens(tokens), len(tokens)).num_tokens == 0

    def test_a_repeated_prompt_hits_all_but_its_last_block(self):
        """The cap is ``num_tokens - 1``, rounded down to a block boundary.

        A request whose every token is cached still has to run one, because its
        first sampled token comes from logits and logits come from a forward
        pass. Sharing is by whole blocks, so a cap that lands mid-block costs
        the whole block: 64 tokens cap at 63, which is 3 blocks, not 4.
        """
        cache = _cache()
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        assert _serve(cache, "b", tokens) == 48

    def test_a_hit_stops_where_the_prompts_diverge(self):
        cache = _cache()
        _serve(cache, "a", list(range(64)))
        assert _serve(cache, "b", list(range(32)) + [99] * 32) == 32

    def test_a_longer_prompt_reuses_the_shorter_one_whole(self):
        """The cap only bites the *querying* sequence's own last block."""
        cache = _cache()
        _serve(cache, "a", list(range(32)))
        assert _serve(cache, "b", list(range(64))) == 32

    def test_a_partial_trailing_block_is_not_reusable(self):
        cache = _cache()
        _serve(cache, "a", list(range(20)))  # one full block plus 4 tokens
        assert _serve(cache, "b", list(range(20))) == 16

    def test_a_prompt_shorter_than_a_block_can_never_hit(self):
        cache = _cache()
        _serve(cache, "a", [1, 2, 3])
        assert _serve(cache, "b", [1, 2, 3]) == 0

    def test_caching_off_allocates_but_never_reuses(self):
        """Caching off is the same path with the index switched off, not a fork."""
        cache = _cache(enable_caching=False)
        tokens = list(range(64))
        assert _serve(cache, "a", tokens) == 0
        assert _serve(cache, "b", tokens) == 0
        assert cache.block_ids("a")[0] != cache.block_ids("b")[0]


# --------------------------------------------------------------------------- #
# 3. Physical sharing
# --------------------------------------------------------------------------- #
class TestPhysicalSharing:
    """A hit is the same rows, not a copy of them."""

    def test_two_sharers_hold_the_same_block_ids(self):
        cache = _cache()
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        hit = _serve(cache, "b", tokens)

        shared = hit // PREFIX_CACHE_BLOCK_SIZE
        a_blocks, b_blocks = cache.block_ids("a")[0], cache.block_ids("b")[0]
        assert a_blocks[:shared] == b_blocks[:shared]
        # The tail is the sequence's own: b re-runs its last block, and writing
        # it into a's rows would corrupt a.
        assert a_blocks[shared:] != b_blocks[shared:]

    def test_sharing_costs_only_the_uncached_tail(self):
        cache = _cache()
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        before = cache.num_free_blocks
        _serve(cache, "b", tokens)
        assert cache.num_free_blocks == before - 1  # only b's own last block

    def test_the_null_block_is_never_handed_out(self):
        cache = _cache()
        _serve(cache, "a", list(range(64)))
        assert 0 not in cache.block_ids("a")[0]

    def test_a_shared_block_is_held_by_every_sharer(self):
        cache = _cache()
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        _serve(cache, "b", tokens)
        shared = cache.block_ids("b")[0][0]
        assert cache.pool.blocks[shared].ref_cnt == 2

        cache.free("a")
        assert cache.pool.blocks[shared].ref_cnt == 1  # b still reads it


# --------------------------------------------------------------------------- #
# 4. Allocation, reference counting and release
# --------------------------------------------------------------------------- #
class TestAllocation:
    """Blocks come from one pool, and every one of them comes back."""

    def test_allocate_covers_the_partial_tail_block(self):
        cache = _cache()
        assert cache.allocate("a", 17)
        assert len(cache.block_ids("a")[0]) == 2  # 16 + 1 token needs two blocks

    def test_growing_a_sequence_only_adds_what_it_grew_into(self):
        cache = _cache()
        cache.allocate("a", 16)
        first = cache.block_ids("a")[0]
        cache.allocate("a", 32)
        grown = cache.block_ids("a")[0]
        assert grown[: len(first)] == first  # earlier blocks stay put
        assert len(grown) == 2

    def test_free_returns_every_block(self):
        cache = _cache()
        before = cache.num_free_blocks
        cache.allocate("a", 64)
        assert cache.num_free_blocks < before
        cache.free("a")
        assert cache.num_free_blocks == before

    def test_a_failed_allocation_allocates_nothing(self):
        """No partial grants: every caller's answer to "no room" is to wait."""
        cache = _cache(num_blocks=4)  # 3 usable
        before = cache.num_free_blocks
        assert not cache.allocate("a", 16 * 10)
        assert cache.num_free_blocks == before
        assert cache.block_ids("a")[0] == ()

    def test_a_rejected_admission_returns_the_hit_blocks_it_touched(self):
        """Adopting a hit takes references; a rejection has to give them back.

        The order matters and is easy to get wrong: an adopted block whose
        ref_cnt is still zero sits in the free queue, so the references have to
        be taken *before* the fresh blocks are drawn. That is also what makes
        this unwind necessary when the draw then fails.
        """
        cache = _cache(num_blocks=5)  # 4 usable
        tokens = list(range(64))
        _serve(cache, "a", tokens)  # holds all 4
        assert cache.num_free_blocks == 0

        hashes = cache.track("b", tokens)
        match = cache.lookup(hashes, len(tokens))
        assert match.num_tokens == 48
        # b adopts 3 blocks and needs a 4th of its own, and there is none.
        assert not cache.allocate("b", 64, match)
        for block in match.blocks[0]:
            assert block.ref_cnt == 1  # a's reference, and only a's

    def test_free_is_safe_for_a_request_that_never_allocated(self):
        cache = _cache()
        cache.free("never-seen")  # must not raise

    def test_freeing_twice_does_not_double_credit_the_pool(self):
        cache = _cache()
        before = cache.num_free_blocks
        cache.allocate("a", 64)
        cache.free("a")
        cache.free("a")
        assert cache.num_free_blocks == before

    def test_every_request_releasing_leaves_nothing_tracked(self):
        cache = _cache()
        for i in range(4):
            _serve(cache, f"r{i}", list(range(i * 64, i * 64 + 64)))
        for i in range(4):
            cache.free(f"r{i}")
        assert cache.coordinator.num_tracked_requests() == 0
        assert cache.num_referenced_blocks == 0


# --------------------------------------------------------------------------- #
# 5. The commit contract
# --------------------------------------------------------------------------- #
class TestCommitContract:
    """A block becomes reusable when its K/V exists, not when it is planned."""

    def test_an_allocated_but_uncommitted_prefix_is_not_reusable(self):
        """The read-before-write window this contract exists to close.

        ``num_computed_tokens`` advances when a chunk is *planned*, one engine
        step before the model writes it. A block offered then would be handed to
        the next admission as readable rows nobody had written.
        """
        cache = _cache()
        tokens = list(range(64))
        cache.track("a", tokens)
        cache.allocate("a", 64)
        assert cache.lookup(cache.hash_tokens(tokens), 64).num_tokens == 0

        cache.commit("a", 64)
        assert cache.lookup(cache.hash_tokens(tokens), 64).num_tokens == 48

    def test_commit_indexes_only_whole_committed_blocks(self):
        cache = _cache()
        tokens = list(range(64))
        cache.track("a", tokens)
        cache.allocate("a", 64)
        cache.commit("a", 40)  # 2 whole blocks plus 8 tokens
        assert cache.num_cached_blocks == 2

    def test_commit_is_idempotent(self):
        cache = _cache()
        tokens = list(range(64))
        cache.track("a", tokens)
        cache.allocate("a", 64)
        cache.commit("a", 64)
        cache.commit("a", 64)
        assert cache.num_cached_blocks == 4
        assert cache.num_committed_tokens("a") == 64

    def test_commit_ignores_an_untracked_request(self):
        cache = _cache()
        cache.commit("never-seen", 64)  # must not raise
        assert cache.num_cached_blocks == 0

    def test_a_committed_block_outlives_its_request(self):
        cache = _cache()
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        cache.free("a")
        assert cache.num_referenced_blocks == 0
        # Freed is not forgotten: the rows are reclaimable but still indexed.
        assert _serve(cache, "b", tokens) == 48


# --------------------------------------------------------------------------- #
# 6. LRU eviction
# --------------------------------------------------------------------------- #
class TestEviction:
    """Pressure eats the least recently useful blocks, and only those."""

    def test_a_referenced_block_is_never_evicted(self):
        cache = _cache(num_blocks=6)
        held = list(range(64))
        _serve(cache, "held", held)  # 4 blocks, still referenced

        # Churn far more sequences than the pool can hold. Each fails or takes
        # only free rows; none may take the live request's.
        for i in range(20):
            cache.allocate(f"churn{i}", 32)
            cache.free(f"churn{i}")

        assert cache.block_ids("held")[0] != ()
        assert cache.lookup(cache.hash_tokens(held), 64).num_tokens == 48

    def test_pressure_evicts_an_unreferenced_cached_block(self):
        cache = _cache(num_blocks=4)  # 3 usable
        first = list(range(32))
        _serve(cache, "a", first)  # 2 blocks, cached
        cache.free("a")
        assert cache.num_evictable_blocks == 2

        # 3 blocks' worth from a different prompt: the only way to satisfy it is
        # to drop a's cached blocks.
        assert cache.allocate("b", 48)
        assert cache.lookup(cache.hash_tokens(first), 32).num_tokens == 0
        assert cache.stats.evictions > 0

    def test_eviction_takes_the_oldest_release_first(self):
        """Cached blocks queue FIFO by release, so the LRU sequence goes first."""
        cache = _cache(num_blocks=5)  # 4 usable
        old, new = list(range(32)), list(range(100, 132))
        _serve(cache, "old", old)
        cache.free("old")
        _serve(cache, "new", new)
        cache.free("new")
        assert cache.num_evictable_blocks == 4

        assert cache.allocate("pressure", 32)  # needs 2 of the 4
        assert cache.lookup(cache.hash_tokens(old), 32).num_tokens == 0
        assert cache.lookup(cache.hash_tokens(new), 32).num_tokens == 16

    def test_a_hit_rescues_a_block_from_the_eviction_queue(self):
        """Adopting a cached block references it, which takes it out of reach."""
        cache = _cache(num_blocks=5)
        old, new = list(range(32)), list(range(100, 132))
        _serve(cache, "old", old)
        cache.free("old")
        _serve(cache, "new", new)
        cache.free("new")

        _serve(cache, "revived", old)  # references old's first block again
        assert cache.allocate("pressure", 32)
        # old's blocks are held now, so pressure had to eat new's instead.
        assert cache.lookup(cache.hash_tokens(new), 32).num_tokens == 0

    def test_the_eviction_counter_reaches_the_stats(self):
        """Read through from the pool, so it is never stale."""
        cache = _cache(num_blocks=4)
        _serve(cache, "a", list(range(32)))
        cache.free("a")
        assert cache.stats.evictions == 0
        cache.allocate("b", 48)
        assert cache.stats.evictions > 0


# --------------------------------------------------------------------------- #
# 7. Decode-token caching
# --------------------------------------------------------------------------- #
class TestDecodeCaching:
    """Generated tokens join the chain, which is what makes a turn-2 prompt cheap."""

    def test_a_generated_block_becomes_reusable(self):
        cache = _cache()
        prompt, output = list(range(32)), list(range(100, 132))

        cache.track("a", prompt)
        cache.allocate("a", 32)
        cache.commit("a", 32)
        # Decode: the sequence grows a token at a time; a completed block gets a
        # hash, and the next step's commit indexes it.
        for step in range(1, len(output) + 1):
            whole = prompt + output[:step]
            cache.observe("a", whole)
            assert cache.allocate("a", len(whole))
            cache.commit("a", len(whole))

        # Turn two: the previous turn's prompt *and* answer are a cached prefix.
        turn_two = prompt + output + [7] * 16
        assert _serve(cache, "b", turn_two) == 64

    def test_the_generated_prefix_is_shared_physically_too(self):
        cache = _cache()
        prompt, output = list(range(32)), list(range(100, 132))
        cache.track("a", prompt)
        cache.allocate("a", 32)
        cache.commit("a", 32)
        whole = prompt + output
        cache.observe("a", whole)
        cache.allocate("a", len(whole))
        cache.commit("a", len(whole))

        _serve(cache, "b", whole + [7] * 16)
        assert cache.block_ids("b")[0][:4] == cache.block_ids("a")[0][:4]

    def test_observe_ignores_an_untracked_request(self):
        cache = _cache()
        assert cache.observe("never-seen", list(range(32))) == []

    def test_only_complete_generated_blocks_are_indexed(self):
        cache = _cache()
        prompt = list(range(32))
        cache.track("a", prompt)
        cache.allocate("a", 40)
        cache.observe("a", prompt + list(range(100, 108)))  # 8 tokens: no block
        cache.commit("a", 40)
        assert cache.num_cached_blocks == 2


# --------------------------------------------------------------------------- #
# 8. Multiple KV cache groups
# --------------------------------------------------------------------------- #
def _two_group_config(sizes: tuple[int, int]) -> KVCacheConfig:
    """Two full-attention groups paging at different block sizes."""
    return KVCacheConfig(
        groups=tuple(
            KVCacheGroup(
                group_id=index,
                spec=FullAttentionSpec(block_size=size, kv_row=(2, 8)),
                layer_ids=(index,),
            )
            for index, size in enumerate(sizes)
        ),
        hash_block_size=min(sizes),
    )


class TestMultipleGroups:
    """Every group gets its own blocks, and a hit is what *all* of them can serve."""

    def test_each_group_is_allocated_separately(self):
        cache = _cache(kv_cache_config=_two_group_config((16, 32)))
        assert cache.allocate("a", 64)
        coarse, fine = cache.block_ids("a")
        assert len(coarse) == 4  # 64 / 16
        assert len(fine) == 2  # 64 / 32
        assert set(coarse).isdisjoint(fine)  # one pool, no aliasing

    def test_a_hit_is_the_shortest_prefix_every_group_can_serve(self):
        cache = _cache(kv_cache_config=_two_group_config((16, 32)))
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        hit = _serve(cache, "b", tokens)
        # Capped at 63; the coarse group can serve 3 of its blocks (48 tokens),
        # the 32-token group only 1 (32) -- and a hit has to be whole blocks in
        # both, so 32 wins.
        assert hit == 32

    def test_a_hit_is_block_aligned_in_every_group(self):
        cache = _cache(kv_cache_config=_two_group_config((16, 48)))
        tokens = list(range(96))
        _serve(cache, "a", tokens)
        assert _serve(cache, "b", tokens) % 48 == 0

    def test_freeing_returns_every_groups_blocks(self):
        cache = _cache(kv_cache_config=_two_group_config((16, 32)))
        before = cache.num_free_blocks
        cache.allocate("a", 64)
        cache.free("a")
        assert cache.num_free_blocks == before

    def test_one_short_group_blocks_the_whole_allocation(self):
        """A group that cannot be served must not leave the others half-extended."""
        cache = _cache(num_blocks=5, kv_cache_config=_two_group_config((16, 32)))
        assert not cache.allocate("a", 64)  # needs 4 + 2, only 4 usable
        assert cache.block_ids("a") == ((), ())


# --------------------------------------------------------------------------- #
# 9. Table writes
# --------------------------------------------------------------------------- #
class TestTableWrites:
    """What the executor is told to point its block table at, and only once."""

    def test_a_fresh_request_emits_every_block_from_zero(self):
        cache = _cache()
        cache.track("a", list(range(64)))
        cache.allocate("a", 64)
        writes = cache.take_table_writes("a")
        assert len(writes) == 1
        group_id, start_block, block_ids = writes[0]
        assert (group_id, start_block) == (0, 0)
        assert block_ids == cache.block_ids("a")[0]

    def test_growing_emits_only_the_new_blocks(self):
        cache = _cache()
        cache.track("a", list(range(32)))
        cache.allocate("a", 32)
        cache.take_table_writes("a")
        cache.allocate("a", 48)
        _, start_block, block_ids = cache.take_table_writes("a")[0]
        assert start_block == 2
        assert block_ids == cache.block_ids("a")[0][2:]

    def test_a_second_call_with_nothing_new_emits_nothing(self):
        """A steady decode step costs no device writes at all.

        A table entry covers its block's whole row span the moment it is
        written, so a sequence advancing inside a block it already mapped has
        nothing left to say.
        """
        cache = _cache()
        cache.track("a", list(range(32)))
        cache.allocate("a", 32)
        assert cache.take_table_writes("a") != ()
        assert cache.take_table_writes("a") == ()

    def test_a_partial_tail_block_is_emitted_once_and_not_again(self):
        cache = _cache()
        cache.track("a", list(range(20)))
        cache.allocate("a", 20)  # 1 full block plus 4 tokens in the next
        assert cache.take_table_writes("a")[0][2] == cache.block_ids("a")[0]
        cache.allocate("a", 32)  # still the same two blocks
        assert cache.take_table_writes("a") == ()

    def test_each_group_gets_its_own_start_block(self):
        cache = _cache(kv_cache_config=_two_group_config((16, 32)))
        cache.track("a", list(range(32)))
        cache.allocate("a", 32)
        cache.take_table_writes("a")
        cache.allocate("a", 64)
        writes = cache.take_table_writes("a")
        assert {(gid, start) for gid, start, _ in writes} == {(0, 2), (1, 1)}

    def test_freeing_resets_the_cursor(self):
        """A preempted request re-maps from scratch, as it re-allocates from scratch."""
        cache = _cache()
        cache.track("a", list(range(32)))
        cache.allocate("a", 32)
        cache.take_table_writes("a")
        cache.free("a")

        cache.track("a", list(range(32)))
        cache.allocate("a", 32)
        assert cache.take_table_writes("a")[0][1] == 0

    def test_an_unknown_request_emits_nothing(self):
        assert _cache().take_table_writes("never-seen") == ()


# --------------------------------------------------------------------------- #
# 10. Reset, stats and configuration
# --------------------------------------------------------------------------- #
class TestResetAndStats:
    def test_reset_is_refused_while_a_block_is_held(self):
        """Not a silent no-op: clearing the index under a live reader would leak.

        The block would stay referenced (so never reallocated) and unindexed (so
        never hit) — capacity lost for the request's lifetime.
        """
        cache = _cache()
        _serve(cache, "a", list(range(64)))
        assert cache.reset() is False
        assert cache.num_cached_blocks == 4

    def test_reset_clears_the_index_and_the_stats_once_idle(self):
        cache = _cache()
        _serve(cache, "a", list(range(64)))
        cache.free("a")
        assert cache.reset() is True
        assert cache.num_cached_blocks == 0
        assert cache.stats.num_requests == 0
        assert cache.num_free_blocks == cache.num_blocks - 1  # the null block

    def test_reset_preserves_the_configuration(self):
        cache = _cache(block_size=8, hash_seed=7)
        assert cache.reset() is True
        assert (cache.block_size, cache.hash_seed) == (8, 7)
        assert cache.num_blocks == _POOL

    def test_stats_count_every_lookup_and_its_hits(self):
        cache = _cache()
        tokens = list(range(64))
        _serve(cache, "a", tokens)
        _serve(cache, "b", tokens)
        assert cache.stats.num_requests == 2
        assert cache.stats.queried_tokens == 128
        assert cache.stats.hit_tokens == 48
        assert cache.hit_rate == pytest.approx(48 / 128)

    def test_hit_rate_is_zero_before_any_lookup(self):
        assert _cache().hit_rate == 0.0
        assert PrefixCacheStats().hit_rate == 0.0

    def test_utilization_tracks_what_is_held(self):
        cache = _cache()
        assert cache.utilization == pytest.approx(1 / _POOL)  # the null block
        cache.allocate("a", 16 * 8)
        assert cache.utilization == pytest.approx(9 / _POOL)

    @pytest.mark.parametrize("block_size", [0, -1])
    def test_an_invalid_block_size_is_rejected(self, block_size):
        with pytest.raises(ValueError, match="block_size"):
            _cache(block_size=block_size)

    @pytest.mark.parametrize("num_blocks", [0, 1])
    def test_a_pool_with_no_usable_block_is_rejected(self, num_blocks):
        with pytest.raises(ValueError, match="at least 2 blocks"):
            PrefixCache(num_blocks=num_blocks)

    def test_two_seeds_do_not_share_hits(self):
        tokens = list(range(64))
        a, b = _cache(hash_seed=1), _cache(hash_seed=2)
        _serve(a, "x", tokens)
        assert a.hash_tokens(tokens) != b.hash_tokens(tokens)


# --------------------------------------------------------------------------- #
# 11. BatchPlanner integration
# --------------------------------------------------------------------------- #
_MAX_SEQ_LEN = 512


class TestBatchPlannerIntegration:
    """The planner's side: admission adopts blocks, and every exit returns them."""

    @staticmethod
    def _sched(**kwargs) -> BatchPlanner:
        num_slots = kwargs.pop("num_slots", 4)
        num_blocks = kwargs.pop("num_blocks", 128)
        kwargs.setdefault("max_seq_len", _MAX_SEQ_LEN)
        kwargs.setdefault("enable_prefix_cache", True)
        return BatchPlanner(SchedulerConfig(**kwargs), num_slots=num_slots, num_blocks=num_blocks)

    @staticmethod
    def _run(sched: BatchPlanner, request: Request, steps: int = 1) -> None:
        """Drive *steps* schedule/execute cycles, faking the sampled tokens."""
        for _ in range(steps):
            out = sched.plan()
            for r in out.decode:
                r.output_token_ids.append(999)

    def test_a_second_admission_reuses_the_first_prompts_blocks(self):
        sched = self._sched()
        tokens = list(range(64))
        first, second = _req("a", tokens), _req("b", tokens)

        sched.add_request(first)
        sched.plan()  # a admitted, prefill planned
        sched.add_request(second)
        sched.plan()  # a's blocks committed; b admitted against them

        assert second.num_cached_tokens == 48
        shared = 48 // PREFIX_CACHE_BLOCK_SIZE
        a_blocks = sched._prefix_cache.block_ids("a")[0]
        b_blocks = sched._prefix_cache.block_ids("b")[0]
        assert b_blocks[:shared] == a_blocks[:shared]

    def test_the_hit_shows_up_as_the_block_plan_the_executor_applies(self):
        sched = self._sched()
        tokens = list(range(64))
        sched.add_request(_req("a", tokens))
        sched.plan()
        second = _req("b", tokens)
        sched.add_request(second)
        sched.plan()

        assert len(second.block_plan) == 1
        group_id, start_block, block_ids = second.block_plan[0]
        assert (group_id, start_block) == (0, 0)
        assert block_ids == sched._prefix_cache.block_ids("b")[0]

    def test_two_requests_admitted_in_one_step_cannot_share(self):
        """Neither has executed yet, so neither's blocks hold any K/V.

        This is the read-before-write window: the first request's blocks are
        allocated but empty until the step *after* the one that planned them.
        """
        sched = self._sched()
        tokens = list(range(64))
        first, second = _req("a", tokens), _req("b", tokens)
        sched.add_request(first)
        sched.add_request(second)
        sched.plan()
        assert (first.num_cached_tokens, second.num_cached_tokens) == (0, 0)
        assert set(sched._prefix_cache.block_ids("a")[0]).isdisjoint(
            sched._prefix_cache.block_ids("b")[0]
        )

    def test_caching_off_admits_without_ever_hitting(self):
        sched = self._sched(enable_prefix_cache=False)
        tokens = list(range(64))
        for name in ("a", "b"):
            sched.add_request(_req(name, tokens))
            sched.plan()
        assert sched._requests["b"].num_cached_tokens == 0

    def test_a_chunked_prefill_still_reuses_what_it_can(self):
        sched = self._sched(max_chunk_size=16)
        tokens = list(range(64))
        first = _req("a", tokens)
        sched.add_request(first)
        for _ in range(6):  # 64 tokens in 16-token chunks, plus slack
            sched.plan()
        assert first.prefill_done

        second = _req("b", tokens)
        sched.add_request(second)
        sched.plan()
        # The hit covers 3 blocks; the remaining chunk is capped at 16.
        assert second.num_cached_tokens == 48
        assert second.num_computed_tokens == 64

    def test_a_finished_request_leaves_its_blocks_cached_but_unheld(self):
        sched = self._sched()
        tokens = list(range(64))
        request = _req("a", tokens)
        sched.add_request(request)
        sched.plan()
        sched.plan()  # commits a's blocks
        sched.finish(request, "stop")

        cache = sched._prefix_cache
        assert cache.num_referenced_blocks == 0
        assert cache.num_cached_blocks >= 3
        assert cache.block_ids("a") == ((),)

    def test_aborting_a_waiting_request_releases_nothing_and_leaks_nothing(self):
        sched = self._sched()
        cache = sched._prefix_cache
        before = cache.num_free_blocks
        sched.add_request(_req("a", list(range(64))))
        sched.abort("a")
        assert cache.num_free_blocks == before
        assert cache.coordinator.num_tracked_requests() == 0

    def test_preemption_returns_the_victims_blocks(self):
        """Regression: a preempt cycle used to leak one reference per cycle.

        A leaked reference is invisible until the pool is tight and then fatal:
        referenced blocks are never eviction candidates, so capacity drains one
        cycle at a time. All three requests share the same four blocks, so the
        total reference count must stay at one per *running* request.
        """
        sched = self._sched(enable_preemption=True, max_num_seqs=3, num_slots=2)
        shared = list(range(64))
        for i in range(3):
            sched.add_request(_req(f"r{i}", shared))

        cache = sched._prefix_cache
        for _ in range(10):
            out = sched.plan()
            for r in out.decode:
                r.output_token_ids.append(999)
            running = sum(1 for r in sched._requests.values() if r.slot is not None)
            total = sum(block.ref_cnt for block in cache.pool.blocks) - 1  # null block
            assert total <= running * 5, f"{total} references for {running} running requests"

    def test_a_full_pool_stalls_admission_instead_of_overcommitting(self):
        sched = self._sched(num_blocks=8, num_slots=4, max_num_seqs=4)
        for i in range(4):
            sched.add_request(_req(f"r{i}", list(range(i * 64, i * 64 + 64))))
        for _ in range(4):
            sched.plan()

        cache = sched._prefix_cache
        assert cache.num_free_blocks >= 0
        admitted = sum(1 for r in sched._requests.values() if r.slot is not None)
        assert 0 < admitted < 4  # 7 usable blocks cannot hold 4 x 4 blocks

    def test_a_decode_step_grows_the_allocation_across_a_block_boundary(self):
        sched = self._sched()
        tokens = list(range(16))  # ends exactly on a block boundary
        request = _req("a", tokens)
        sched.add_request(request)
        cache = sched._prefix_cache

        sched.plan()  # prefill: 16 tokens, 1 block
        assert len(cache.block_ids("a")[0]) == 1
        self._run(sched, request, steps=1)  # first decode token: position 16
        assert len(cache.block_ids("a")[0]) == 2
        # The new block is exactly what the executor is told to map.
        assert request.block_plan == ((0, 1, cache.block_ids("a")[0][1:]),)

    def test_a_steady_decode_step_maps_nothing(self):
        sched = self._sched()
        request = _req("a", list(range(20)))
        sched.add_request(request)
        sched.plan()
        self._run(sched, request, steps=2)
        assert request.block_plan == ()  # still inside block 1

    def test_generated_tokens_become_a_hit_for_the_next_prompt(self):
        """End to end decode caching: turn two skips turn one's whole answer."""
        sched = self._sched()
        prompt = list(range(32))
        first = _req("a", prompt)
        sched.add_request(first)
        sched.plan()
        for i in range(34):  # generate well past two block boundaries
            self._run(sched, first, steps=1)
            assert first.output_token_ids, f"no token generated by step {i}"

        turn_two = prompt + first.output_token_ids[:32] + [5] * 16
        second = _req("b", turn_two)
        sched.add_request(second)
        sched.plan()
        assert second.num_cached_tokens >= 48  # past the prompt, into the answer
