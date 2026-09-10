"""C-axis compute-communication overlap primitives: policy, pool, deferral.

CPU tests pin the dispatch decision, the chunk arithmetic and the world-of-one
pass-through; the two-rank NCCL payloads pin that an all-reduce posted on the
comm stream carries exactly the blocking one's values, that the L3 chunked
path and the deferred (TBO) path agree with it, that :class:`CollectiveStats`
still sees the traffic, and that the timeline records the reduce region.

Usage:
    pytest tests/executor/test_comm_overlap.py
"""

from __future__ import annotations

import os

import pytest
import torch

import rapid_llm.batch_overlap.comm_overlap as comm_overlap
from rapid_llm.batch_overlap.comm_overlap import (
    COMM_OVERLAP_ENV,
    L3_CHUNKS_ENV,
    L3_MIN_ROWS_ENV,
    CommOverlapPolicy,
    CommStreamPool,
    _chunk_bounds,
    _dispatch_mode,
    comm_overlap_policy,
    deferred_all_reduce,
    reset_comm_overlap_policy,
)
from rapid_llm.distributed import parallel_state as ps
from rapid_llm.modules import RowParallelLinear
from rapid_llm.tools.observability import Collective, CollectiveStats
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

IN, OUT = 64, 32


@pytest.fixture(autouse=True)
def _fresh_comm_policy():
    """Every test reads the policy from the env it set, never the last test's cache."""
    reset_comm_overlap_policy()
    yield
    reset_comm_overlap_policy()


@pytest.fixture
def grid(monkeypatch):
    """Become one rank of a ``tp_size``-wide grid, with no process group behind it."""

    def enter(rank: int, tp_size: int) -> None:
        monkeypatch.setattr(ps, "_TP_RANK", rank)
        monkeypatch.setattr(ps, "_TP_WORLD_SIZE", tp_size)

    return enter


# --------------------------------------------------------------------------- #
# Policy parsing and chunk arithmetic (CPU)
# --------------------------------------------------------------------------- #
def test_l3_policy_is_off_by_default(monkeypatch):
    monkeypatch.delenv(COMM_OVERLAP_ENV, raising=False)
    assert not CommOverlapPolicy.from_env().enabled
    for raw in ("0", "false", "off", "OFF"):
        monkeypatch.setenv(COMM_OVERLAP_ENV, raw)
        assert not CommOverlapPolicy.from_env().enabled


def test_l3_policy_accepts_the_on_spellings_and_parameters(monkeypatch):
    for raw in ("1", "chunked", "l3", "on"):
        monkeypatch.setenv(COMM_OVERLAP_ENV, raw)
        assert CommOverlapPolicy.from_env().enabled
    monkeypatch.setenv(COMM_OVERLAP_ENV, "1")
    monkeypatch.setenv(L3_MIN_ROWS_ENV, "4096")
    monkeypatch.setenv(L3_CHUNKS_ENV, "4")
    policy = CommOverlapPolicy.from_env()
    assert policy.min_rows == 4096
    assert policy.chunks == 4


def test_policy_cache_is_read_once_per_process(monkeypatch):
    monkeypatch.setenv(COMM_OVERLAP_ENV, "1")
    assert comm_overlap_policy().enabled
    monkeypatch.setenv(COMM_OVERLAP_ENV, "0")
    assert comm_overlap_policy().enabled, "a cached policy must not re-read the env"
    reset_comm_overlap_policy()
    assert not comm_overlap_policy().enabled


def test_chunk_count_floors_small_gemms():
    policy = CommOverlapPolicy(enabled=True, chunks=4)
    assert policy.chunk_count(0) == 1
    assert policy.chunk_count(100) == 1, "100 rows cannot buy four 256-row chunks"
    assert policy.chunk_count(600) == 2
    assert policy.chunk_count(2_000) == 4
    assert CommOverlapPolicy(enabled=False, chunks=4).chunk_count(2_000) == 1


def test_chunk_bounds_cover_every_row_exactly_once():
    assert _chunk_bounds(7, 3) == [(0, 3), (3, 5), (5, 7)]
    assert _chunk_bounds(8, 2) == [(0, 4), (4, 8)]
    assert _chunk_bounds(3, 4) == [(0, 1), (1, 2), (2, 3)], "short chunks collapse away"
    for rows, count in ((97, 4), (256, 2), (1_023, 3)):
        covered = [row for start, stop in _chunk_bounds(rows, count) for row in range(start, stop)]
        assert covered == list(range(rows))


def test_dispatch_mode_priority():
    """World of one beats everything; deferred (TBO) beats chunked (L3)."""
    on = CommOverlapPolicy(enabled=True, min_rows=8)
    off = CommOverlapPolicy(enabled=False)
    assert _dispatch_mode(1, False, on, 10_000) == "passthrough"
    assert _dispatch_mode(1, True, on, 10_000) == "passthrough"
    assert _dispatch_mode(2, True, on, 10_000) == "deferred"
    assert _dispatch_mode(2, False, on, 10_000) == "chunked"
    assert _dispatch_mode(2, False, on, 7) == "blocking", "below min_rows"
    assert _dispatch_mode(2, False, off, 10_000) == "blocking"


# --------------------------------------------------------------------------- #
# World-of-one and dispatch on CPU (no process group)
# --------------------------------------------------------------------------- #
def _cpu_layer(tp_size: int) -> RowParallelLinear:
    """A deterministically weighted layer for the ambient grid (CPU, fp32)."""
    layer = RowParallelLinear(IN, OUT, params_dtype=torch.float32)
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(OUT, IN // tp_size, generator=gen))
    return layer


def test_world_of_one_passes_the_gemm_straight_through(grid):
    grid(0, 1)
    layer = _cpu_layer(1)
    x = torch.randn(2, 3, IN)
    assert torch.equal(layer.forward(x), layer.apply_linear(x))


def test_reduce_results_false_hands_back_the_partial_sum(grid):
    """``reduce_results=False`` means the caller reduces; the dispatcher must not.

    The flag is the layer's documented escape hatch for a caller that wants to
    time, batch or skip the collective, and this dispatcher is the only place it
    can still be honoured — reducing anyway would silently double-count the
    partial for whoever reduces downstream.
    """
    grid(0, 2)
    layer = RowParallelLinear(IN, OUT, params_dtype=torch.float32, reduce_results=False)
    gen = torch.Generator().manual_seed(3)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(OUT, IN // 2, generator=gen))
    x = torch.randn(2, 3, IN // 2)
    assert torch.equal(layer.forward(x), layer.apply_linear(x))


def test_deferred_context_on_a_world_of_one_is_value_preserving(grid):
    """Defer with no peer to reduce with returns the tensor untouched."""
    grid(0, 1)
    layer = _cpu_layer(1)
    x = torch.randn(2, 3, IN)
    with deferred_all_reduce("cpu") as ctx:
        out = layer.forward(x)
        assert torch.equal(out, layer.apply_linear(x))
        ctx.drain()  # no events pending: a no-op


def test_deferred_dispatch_uses_the_pool_of_the_calling_device(grid):
    """A tp world without a group (CPU tests) defers into a no-op all-reduce."""
    grid(0, 2)
    layer = _cpu_layer(2)
    x = torch.randn(2, 3, IN // 2)  # the rank's slice of the contracted dim
    with deferred_all_reduce("cpu") as ctx:
        out = layer.forward(x)
        assert torch.equal(out, layer.apply_linear(x))
        assert ctx is comm_overlap.current_deferred_ar()


def test_below_threshold_falls_back_to_the_blocking_reduce(grid, monkeypatch):
    grid(0, 2)
    monkeypatch.setenv(COMM_OVERLAP_ENV, "1")
    monkeypatch.setenv(L3_MIN_ROWS_ENV, "4096")
    calls: list[tuple] = []

    def spy(tensor, op=None, *, group=None):
        calls.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(comm_overlap, "tensor_model_parallel_all_reduce", spy)
    layer = _cpu_layer(2)
    layer.forward(torch.randn(2, 4, IN // 2))  # 8 tokens: far below 4096
    assert calls == [(2, 4, OUT)]


# --------------------------------------------------------------------------- #
# Two-rank NCCL payloads
# --------------------------------------------------------------------------- #
def _shard_weight(rank: int, world: int) -> torch.Tensor:
    """This rank's column slice of a deterministic full weight."""
    gen = torch.Generator().manual_seed(7)
    full = torch.randn(OUT, IN, generator=gen, dtype=torch.float32)
    width = IN // world
    return full[:, rank * width : (rank + 1) * width].contiguous()


def _layer_on(rank: int, device: torch.device) -> RowParallelLinear:
    layer = RowParallelLinear(IN, OUT, params_dtype=torch.float32).to(device)
    with torch.no_grad():
        layer.weight.copy_(_shard_weight(rank, 2).to(device))
    return layer


def _payload_async_matches_blocking(rank: int) -> str:
    device = torch.device("cuda", rank)
    gen = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(128, 64, generator=gen).to(device)
    pool = CommStreamPool.for_device(device)
    async_view = x.clone()
    event = pool.all_reduce_async(async_view, label="ar.async_test")
    assert event is not None, "a two-rank NCCL group must return a fence event"
    event.synchronize()  # host-side fence: the value is final now
    blocking = x.clone()
    ps.tensor_model_parallel_all_reduce(blocking)
    assert torch.equal(async_view, blocking)
    return "ok"


def _payload_chunked_matches_blocking(rank: int) -> str:
    os.environ[COMM_OVERLAP_ENV] = "1"
    os.environ[L3_MIN_ROWS_ENV] = "16"
    os.environ[L3_CHUNKS_ENV] = "4"
    reset_comm_overlap_policy()
    device = torch.device("cuda", rank)
    layer = _layer_on(rank, device)
    gen = torch.Generator().manual_seed(200 + rank)
    x = torch.randn(32, 32, IN // 2, generator=gen).to(device)  # 1024 tokens
    ref = ps.tensor_model_parallel_all_reduce(layer.apply_linear(x.reshape(-1, IN // 2))).view(
        32, 32, OUT
    )
    out = layer.forward(x)
    assert out.shape == (32, 32, OUT), "the leading dims come back as they went in"
    # Row-chunked GEMMs pick different cuBLAS kernels than one wide GEMM, so
    # parity is mathematical, not bitwise (a two-term all-reduce is exact).
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)
    return "ok"


def _payload_deferred_matches_blocking(rank: int) -> str:
    device = torch.device("cuda", rank)
    layer = _layer_on(rank, device)
    gen = torch.Generator().manual_seed(300 + rank)
    x = torch.randn(2, 8, IN // 2, generator=gen).to(device)
    ref = ps.tensor_model_parallel_all_reduce(layer.apply_linear(x))
    with deferred_all_reduce(device) as ctx:
        partial = layer.forward(x)  # deferred: the value is not final yet
        ctx.drain()  # the consume point
        fenced = partial.clone()
    assert torch.equal(fenced, ref)
    return "ok"


def _payload_collecting_routes_events_per_batch(rank: int) -> str:
    device = torch.device("cuda", rank)
    layer = _layer_on(rank, device)
    x = torch.randn(2, 4, IN // 2, device=device)
    with deferred_all_reduce(device) as ctx:
        events: list = []
        with ctx.collecting(events):
            partial = layer.forward(x)
        assert len(events) == 1, "one row-parallel forward defers exactly one reduce"
        ctx.fence(events)  # empties the collector
        assert events == []
        assert ctx._events == [], "a consumed event must not be fenced again at context exit"
        assert torch.equal(partial, ps.tensor_model_parallel_all_reduce(layer.apply_linear(x)))
    return "ok"


def _payload_stats_counts_async_traffic(rank: int) -> str:
    device = torch.device("cuda", rank)
    pool = CommStreamPool.for_device(device)
    x = torch.ones(128, 64, device=device)
    with CollectiveStats.collect() as stats:
        event = pool.all_reduce_async(x)
        assert event is not None
        event.synchronize()
    tally = stats.tally(Collective.ALL_REDUCE)
    assert tally.calls == 1
    assert tally.nbytes == 128 * 64 * 4, "the async path reports the same payload"
    return "ok"


def _payload_timeline_records_the_reduce_region(rank: int) -> str:
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "1"
    CommStreamPool.reset()  # rebuild with a timeline enabled pool
    device = torch.device("cuda", rank)
    pool = CommStreamPool.for_device(device)
    x = torch.ones(64, 32, device=device)
    event = pool.all_reduce_async(x, label="ar.timeline")
    event.synchronize()
    records = pool.timeline.collect()
    ar = next(r for r in records if r.name == "ar.timeline")
    assert ar.stream == "comm"
    assert ar.duration_ms >= 0.0
    return "ok"


@needs_gpus(2)
def test_async_all_reduce_matches_blocking_on_two_ranks():
    assert run_on_tp_ranks(_payload_async_matches_blocking, tp_size=2) == ["ok", "ok"]


@needs_gpus(2)
def test_chunked_row_parallel_matches_blocking_on_two_ranks():
    assert run_on_tp_ranks(_payload_chunked_matches_blocking, tp_size=2) == ["ok", "ok"]


@needs_gpus(2)
def test_deferred_all_reduce_matches_blocking_on_two_ranks():
    assert run_on_tp_ranks(_payload_deferred_matches_blocking, tp_size=2) == ["ok", "ok"]


@needs_gpus(2)
def test_collecting_routes_events_per_batch_on_two_ranks():
    assert run_on_tp_ranks(_payload_collecting_routes_events_per_batch, tp_size=2) == ["ok", "ok"]


@needs_gpus(2)
def test_stats_counts_async_traffic_on_two_ranks():
    assert run_on_tp_ranks(_payload_stats_counts_async_traffic, tp_size=2) == ["ok", "ok"]


@needs_gpus(2)
def test_timeline_records_the_reduce_region_on_two_ranks():
    assert run_on_tp_ranks(_payload_timeline_records_the_reduce_region, tp_size=2) == ["ok", "ok"]
