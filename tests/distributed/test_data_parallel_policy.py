"""Unit tests for policies embedded in the data-parallel controller."""

from __future__ import annotations

import pytest

from rapid_llm.engine.data_parallel import (
    LOAD_BALANCE_POLICIES,
    CacheAwarePolicy,
    RequestCountPolicy,
    RoundRobinPolicy,
    TokenCountPolicy,
    create_load_policy,
)
from rapid_llm.engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE

_ALL = [RoundRobinPolicy, RequestCountPolicy, TokenCountPolicy, CacheAwarePolicy]


def _prompt(tag: int, blocks: int) -> list[int]:
    return [tag * 1000 + index for index in range(blocks * PREFIX_CACHE_BLOCK_SIZE)]


def _shared(prefix: list[int], tag: int, blocks: int) -> list[int]:
    return prefix + _prompt(tag, blocks)


def test_factory_builds_every_policy():
    assert LOAD_BALANCE_POLICIES == (
        "round_robin",
        "total_requests",
        "total_tokens",
        "cache_aware",
    )
    for name in LOAD_BALANCE_POLICIES:
        assert create_load_policy(name, 2).dp_size == 2


def test_factory_rejects_unknown_policy():
    with pytest.raises(ValueError, match="unknown load_balancer"):
        create_load_policy("magic", 2)


@pytest.mark.parametrize("policy", _ALL)
def test_non_positive_replica_count_is_rejected(policy):
    with pytest.raises(ValueError, match="dp_size must be >= 1"):
        policy(0)


def test_round_robin_cycles_independently_of_release():
    policy = RoundRobinPolicy(2)
    assert [policy.select(request_id=f"r{i}") for i in range(4)] == [0, 1, 0, 1]
    policy.release("r0")
    assert policy.select(request_id="r4") == 0


def test_request_count_prefers_a_replica_after_exact_release():
    policy = RequestCountPolicy(2)
    assert policy.select(request_id="a") == 0
    assert policy.select(request_id="b") == 1
    assert policy.select(request_id="c") == 0
    policy.release("c")
    policy.release("a")
    assert policy.load == (0, 1)
    assert policy.select(request_id="d") == 0


def test_token_count_routes_skewed_prompts_and_releases_out_of_order():
    policy = TokenCountPolicy(2)
    assert policy.select(1000, request_id="long") == 0
    assert policy.select(10, request_id="short-1") == 1
    assert policy.select(10, request_id="short-2") == 1
    assert policy.load == (1000, 20)
    policy.release("short-2")
    assert policy.load == (1000, 10)
    policy.release("long")
    assert policy.load == (0, 10)


def test_zero_token_requests_still_fill_the_pool():
    policy = TokenCountPolicy(2)
    assert [policy.select(0, request_id=f"r{i}") for i in range(4)] == [0, 1, 0, 1]


def test_cache_aware_reuses_a_prefix():
    prefix = _prompt(1, 8)
    policy = CacheAwarePolicy(2)
    first = policy.select(token_ids=_shared(prefix, 2, 1), request_id="first")
    policy.release("first")
    second = policy.select(token_ids=_shared(prefix, 3, 1), request_id="second")
    assert first == second == 0


def test_cache_aware_balances_unrelated_prompts():
    policy = CacheAwarePolicy(2)
    picks = [policy.select(token_ids=_prompt(tag, 4), request_id=f"r{tag}") for tag in range(4)]
    assert picks == [0, 1, 0, 1]


def test_cache_aware_releases_exact_charge_out_of_order():
    policy = CacheAwarePolicy(2)
    first = _prompt(1, 4)
    second = _prompt(2, 1)
    assert policy.select(token_ids=first, request_id="long") == 0
    assert policy.select(token_ids=second, request_id="short") == 1
    before = policy.load
    policy.release("short")
    assert policy.load == (before[0], 0)
    policy.release("long")
    assert policy.load == (0, 0)


def test_cache_index_survives_release_and_is_bounded():
    policy = CacheAwarePolicy(1, index_capacity=4)
    old = _prompt(1, 4)
    fresh = _prompt(2, 4)
    policy.select(token_ids=old, request_id="old")
    policy.release("old")
    assert policy.cached_tokens(old, 0) == len(old)
    policy.select(token_ids=fresh, request_id="fresh")
    assert policy.resident_blocks(0) == 4
    assert policy.cached_tokens(fresh, 0) == len(fresh)
    assert policy.cached_tokens(old, 0) == 0


def test_cache_aware_credits_only_complete_leading_blocks():
    policy = CacheAwarePolicy(1)
    prompt = [*_prompt(1, 2), 777]
    policy.select(token_ids=prompt, request_id="one")
    assert policy.cached_tokens(prompt, 0) == 2 * PREFIX_CACHE_BLOCK_SIZE
    diverged = _prompt(9, 1) + prompt
    assert policy.cached_tokens(diverged, 0) == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"block_size": 0}, "block_size must be >= 1"),
        ({"index_capacity": 0}, "index_capacity must be >= 1"),
    ],
)
def test_cache_aware_rejects_invalid_index(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CacheAwarePolicy(2, **kwargs)
