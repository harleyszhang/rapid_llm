"""Tests for the DP x TP rank grid in :mod:`rapid_llm.distributed.parallel_state`.

Pure CPU: initialise a grid and assert the coordinate maths — distinct
cells, contiguous TP groups — plus rejection of out-of-range ranks. The
collectives run over a real two-rank gloo group, which needs no device.

Usage:
    pytest tests/distributed/test_parallel_state.py
"""

from __future__ import annotations

import pytest

from rapid_llm.distributed import parallel_state as ps
from tests.distributed.tp_harness import run_on_tp_ranks


@pytest.fixture(autouse=True)
def _reset_grid():
    """Restore the world of one after each test.

    The grid is module-level state and model layers read it while being built, so a
    test that left ``dp_size=4`` behind would change how unrelated tests place ranks.
    """
    yield
    ps.destroy_parallel()


def test_defaults_to_a_world_of_one():
    assert ps.get_tensor_model_parallel_rank() == 0
    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_data_parallel_rank() == 0
    assert ps.get_data_parallel_world_size() == 1
    assert ps.get_world_size() == 1


@pytest.mark.parametrize(
    ("global_rank", "tp_size", "dp_size", "dp_rank", "tp_rank"),
    [
        # Pure DP: every rank is its own replica.
        (0, 1, 4, 0, 0),
        (3, 1, 4, 3, 0),
        # Pure TP: one replica, ranks are TP ranks.
        (0, 4, 1, 0, 0),
        (3, 4, 1, 0, 3),
        # 2x2 grid: replica 1 owns global ranks 2 and 3.
        (1, 2, 2, 0, 1),
        (2, 2, 2, 1, 0),
        (3, 2, 2, 1, 1),
    ],
)
def test_grid_coordinates(global_rank, tp_size, dp_size, dp_rank, tp_rank):
    """``global_rank = dp_rank * tp_size + tp_rank``, decomposed the other way."""
    assert ps.grid_coordinates(global_rank, tp_size, dp_size) == (dp_rank, tp_rank)


def test_every_global_rank_maps_to_a_distinct_cell():
    """No two ranks of a 2x3 grid may share a ``(dp_rank, tp_rank)`` cell."""
    tp_size, dp_size = 3, 2
    cells = [
        ps.grid_coordinates(global_rank, tp_size, dp_size)
        for global_rank in range(tp_size * dp_size)
    ]

    assert len(set(cells)) == tp_size * dp_size


def test_a_replicas_tp_ranks_are_contiguous():
    """Replica ``d`` must own exactly the global ranks ``[d*tp, (d+1)*tp)``.

    The TP process group is built from that range, so a layout where a replica's ranks
    were strided would all-reduce across replicas instead of within one.
    """
    tp_size, dp_size = 2, 3
    for dp_rank in range(dp_size):
        owned = [
            global_rank
            for global_rank in range(tp_size * dp_size)
            if ps.grid_coordinates(global_rank, tp_size, dp_size)[0] == dp_rank
        ]
        assert owned == list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))


def test_rank_outside_the_grid_is_rejected():
    with pytest.raises(ValueError, match="outside a 2x2 grid"):
        ps.grid_coordinates(4, tp_size=2, dp_size=2)


@pytest.mark.parametrize(("tp_size", "dp_size"), [(0, 1), (1, 0), (-1, 2)])
def test_non_positive_sizes_are_rejected(tp_size, dp_size):
    with pytest.raises(ValueError, match="must be >= 1"):
        ps.grid_coordinates(0, tp_size=tp_size, dp_size=dp_size)


def test_init_parallel_records_pure_dp_without_a_process_group():
    """``tp_size=1`` must place the rank and return, touching no collective library."""
    import torch.distributed as dist

    ps.init_parallel(global_rank=1, tp_size=1, dp_size=2)

    assert (ps.get_data_parallel_rank(), ps.get_tensor_model_parallel_rank()) == (1, 0)
    assert ps.get_data_parallel_world_size() == 2
    assert ps.get_world_size() == 2
    assert not dist.is_initialized()


def test_init_parallel_validates_before_mutating_state():
    """A bad rank must leave the previous grid intact, not half-applied."""
    ps.init_parallel(global_rank=1, tp_size=1, dp_size=2)

    with pytest.raises(ValueError):
        ps.init_parallel(global_rank=9, tp_size=1, dp_size=2)

    assert ps.get_data_parallel_rank() == 1
    assert ps.get_data_parallel_world_size() == 2


def test_init_tensor_parallel_is_the_dp_1_case():
    """The TP entry point must place the process in a single-replica grid."""
    ps.init_tensor_parallel(rank=0, world_size=1)

    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_data_parallel_world_size() == 1


def test_destroy_parallel_restores_the_world_of_one():
    ps.init_parallel(global_rank=2, tp_size=1, dp_size=4)
    assert ps.get_data_parallel_rank() == 2

    ps.destroy_parallel()

    assert ps.get_data_parallel_rank() == 0
    assert ps.get_data_parallel_world_size() == 1


def test_collectives_are_no_ops_without_tensor_parallelism():
    """Every collective must return its input untouched in a world of one.

    They are called unconditionally by ``RowParallelLinear`` and the KV-cache
    profiler, so on a single GPU they have to be free *and* transparent.
    """
    import torch

    ps.init_parallel(global_rank=1, tp_size=1, dp_size=2)
    tensor = torch.ones(4)

    assert ps.tensor_model_parallel_all_reduce(tensor) is tensor
    assert ps.tensor_model_parallel_all_gather(tensor) is tensor
    assert ps.reduce_scatter(tensor) is tensor
    assert ps.all_to_all(tensor) is tensor
    assert ps.tensor_model_parallel_broadcast(tensor) is tensor
    ps.send(tensor, dst=0)  # no peer: must do nothing, not deadlock
    assert ps.recv(tensor, src=0) is tensor
    assert ps.tensor_model_parallel_all_reduce_min(7) == 7


def test_divide_names_the_dimension_that_does_not_fit():
    assert ps.divide(8, 4, "attention heads") == 2
    with pytest.raises(ValueError, match="attention heads 6 does not divide across 4"):
        ps.divide(6, 4, "attention heads")


def test_p2p_all_reduce_posts_send_and_receive_together(monkeypatch):
    """Both peers must post receive work before waiting, or blocking sends deadlock."""
    import torch

    posted = []

    class Work:
        def wait(self):
            return True

    def p2p_op(op, tensor, *, group_peer, group):
        entry = (op, tensor, group_peer, group)
        posted.append(entry)
        return entry

    def batch(ops):
        assert len(ops) == 2
        ops[1][1].fill_(2)
        return [Work(), Work()]

    group = object()
    monkeypatch.setattr(ps, "_TP_WORLD_SIZE", 2)
    monkeypatch.setattr(ps, "_TP_RANK", 0)
    monkeypatch.setattr(ps, "_TP_GROUP", group)
    monkeypatch.setattr(ps.dist, "get_backend", lambda _group: "nccl")
    monkeypatch.setattr(ps.dist, "P2POp", p2p_op)
    monkeypatch.setattr(ps.dist, "batch_isend_irecv", batch)

    tensor = torch.ones(4)
    assert torch.equal(ps.p2p_all_reduce(tensor), torch.full((4,), 3.0))
    assert [entry[2] for entry in posted] == [1, 1]
    ps.abandon_parallel()


def test_small_all_reduce_does_not_switch_collectives_after_p2p_error(monkeypatch):
    """A partial P2P failure must propagate; fallback could deadlock its peer."""
    import torch

    monkeypatch.setattr(ps, "_TP_WORLD_SIZE", 2)
    monkeypatch.setattr(ps, "_TP_GROUP", object())
    monkeypatch.setattr(ps.dist, "get_backend", lambda _group: "nccl")
    def fail_p2p(_tensor):
        raise RuntimeError("p2p")

    monkeypatch.setattr(ps, "p2p_all_reduce", fail_p2p)
    monkeypatch.setattr(
        ps.dist,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("must not enter a mismatched collective"),
    )

    with pytest.raises(RuntimeError, match="p2p"):
        ps.tensor_model_parallel_all_reduce(torch.ones(4))
    ps.abandon_parallel()


def _warmup_probe_worlds(rank: int) -> list[int]:
    """The world size of every group ``warmup_collectives`` issues a probe on."""
    import torch.distributed as dist

    worlds: list[int] = []
    real_all_reduce = dist.all_reduce

    def spy(tensor, *, op=None, group=None):
        real_all_reduce(tensor, op=op, group=group)
        worlds.append(dist.get_world_size(group))

    dist.all_reduce = spy
    try:
        ps.warmup_collectives()
    finally:
        dist.all_reduce = real_all_reduce
    return worlds


def test_warmup_collectives_probes_every_group_the_grid_built():
    """TP, the DP-attention group and the grid-wide EP group each get one probe.

    Under DP-attention the module owns three NCCL communicators, and a CUDA-graph
    capture may hold the *first* collective of any of them — the failure
    :func:`warmup_collectives` exists to prevent. A TP-only world must keep
    probing exactly one group.
    """
    tp_only = run_on_tp_ranks(_warmup_probe_worlds, tp_size=2, backend="gloo")
    assert tp_only == [[2], [2]]
    both = run_on_tp_ranks(
        _warmup_probe_worlds,
        tp_size=2,
        dp_size=2,
        backend="gloo",
        enable_expert_parallel=True,
        enable_dp_attention=True,
    )
    assert both == [[2, 2, 4]] * 4


# --------------------------------------------------------------------------- #
# Two-rank gloo collectives: the tensor primitives over a real process group.
# --------------------------------------------------------------------------- #
def _reduce_scatter_is_all_reduce_then_slice(rank: int) -> bool:
    """reduce_scatter(dim=-1) must equal an all-reduce followed by this rank's slice."""
    import torch

    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4) * (rank + 1)
    shard = ps.reduce_scatter(tensor.clone(), dim=-1)
    reduced = ps.tensor_model_parallel_all_reduce(tensor.clone())
    return bool(torch.equal(shard, reduced[:, rank * 2 : (rank + 1) * 2]))


def _all_to_all_transposes_rank_slices(rank: int) -> bool:
    """Slice ``j`` of rank ``i`` must land on rank ``j``: rank ``r`` ends holding row ``r`` of every rank."""
    import torch

    tensor = torch.tensor([[rank * 10], [rank * 10 + 1]], dtype=torch.int64)
    received = ps.all_to_all(tensor)
    expected = torch.tensor([[rank], [10 + rank]], dtype=torch.int64)
    return bool(torch.equal(received, expected))


def _send_recv_pair(rank: int) -> bool:
    """Rank 0's send must arrive verbatim in rank 1's receive buffer."""
    import torch

    if rank == 0:
        ps.send(torch.full((3,), 7.0), dst=1)
        return True
    buffer = torch.zeros(3)
    ps.recv(buffer, src=0)
    return bool(torch.equal(buffer, torch.full((3,), 7.0)))


def _default_group_is_the_tp_group(rank: int) -> bool:
    """``group=None`` must move exactly the bytes an explicit TP membership group moves."""
    import torch
    import torch.distributed as dist

    explicit = dist.new_group([0, 1], backend="gloo")
    tensor = torch.arange(4, dtype=torch.float32) * (rank + 1)
    implicit = ps.reduce_scatter(tensor.clone(), dim=0)
    over_explicit = ps.reduce_scatter(tensor.clone(), dim=0, group=explicit)
    return bool(torch.equal(implicit, over_explicit))


class TestGlooCollectives:
    """The data-plane primitives over a real two-rank gloo group, no device needed."""

    def test_reduce_scatter_is_all_reduce_then_slice(self):
        both = run_on_tp_ranks(_reduce_scatter_is_all_reduce_then_slice, tp_size=2, backend="gloo")
        assert both == [True, True]

    def test_all_to_all_transposes_rank_slices(self):
        both = run_on_tp_ranks(_all_to_all_transposes_rank_slices, tp_size=2, backend="gloo")
        assert both == [True, True]

    def test_send_recv_pair(self):
        both = run_on_tp_ranks(_send_recv_pair, tp_size=2, backend="gloo")
        assert both == [True, True]

    def test_default_group_is_the_tp_group(self):
        both = run_on_tp_ranks(_default_group_is_the_tp_group, tp_size=2, backend="gloo")
        assert both == [True, True]
