"""CPU tests for the decode-context-parallel KV layout and LSE reduction."""

from __future__ import annotations

import pytest
import torch

from rapid_llm.distributed.dcp import (
    dcp_block_rows,
    dcp_kv_head_range,
    dcp_local_seq_len,
    merge_attention_partials,
    merge_dcp_attention,
)
from tests.distributed.tp_harness import run_on_tp_ranks


def test_merge_attention_partials_matches_full_softmax():
    scores = torch.tensor(
        [
            [[1000.0, 999.0, -1000.0, 998.0], [0.0, 1.0, 2.0, 3.0]],
            [[-4.0, -3.0, -2.0, -1.0], [40.0, -40.0, 39.0, -39.0]],
        ]
    )
    values = torch.arange(2 * 2 * 4 * 3, dtype=torch.float32).reshape(2, 2, 4, 3)

    local_outputs = []
    local_lses = []
    for shard in scores.chunk(2, dim=-1):
        shard_index = len(local_outputs)
        shard_values = values[:, :, shard_index * 2 : (shard_index + 1) * 2]
        local_lses.append(torch.logsumexp(shard, dim=-1))
        local_outputs.append((torch.softmax(shard, dim=-1).unsqueeze(-1) * shard_values).sum(dim=2))

    output, lse = merge_attention_partials(torch.stack(local_outputs), torch.stack(local_lses))
    expected_lse = torch.logsumexp(scores, dim=-1)
    expected_output = (torch.softmax(scores, dim=-1).unsqueeze(-1) * values).sum(dim=2)

    torch.testing.assert_close(lse, expected_lse)
    torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-4)


def test_merge_attention_partials_preserves_empty_history_semantics():
    outputs = torch.randn(2, 3, 4, 5)
    lses = torch.full((2, 3, 4), -float("inf"))

    output, lse = merge_attention_partials(outputs, lses)

    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isneginf(lse).all()


def test_merge_attention_partials_validates_shapes():
    with pytest.raises(ValueError, match="require outputs"):
        merge_attention_partials(torch.zeros(2, 3, 4), torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="do not align"):
        merge_attention_partials(torch.zeros(2, 3, 4, 5), torch.zeros(2, 3, 5))


def _distributed_merge(rank: int) -> dict[str, bool]:
    heads, head_dim = 4, 3
    local_output = torch.arange(2 * heads * head_dim, dtype=torch.float32).reshape(
        2, heads, head_dim
    )
    local_output = local_output + rank * 100.0
    local_lse = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [-float("inf"), -float("inf"), -float("inf"), -float("inf")]]
    )
    if rank == 1:
        local_lse[0] = torch.tensor([4.0, 3.0, 2.0, 1.0])

    output, lse = merge_dcp_attention(local_output, local_lse)

    all_outputs = torch.stack(
        [
            torch.arange(2 * heads * head_dim, dtype=torch.float32).reshape(2, heads, head_dim),
            torch.arange(2 * heads * head_dim, dtype=torch.float32).reshape(2, heads, head_dim)
            + 100.0,
        ]
    )
    all_lses = torch.stack(
        [
            torch.tensor(
                [
                    [0.0, 1.0, 2.0, 3.0],
                    [-float("inf"), -float("inf"), -float("inf"), -float("inf")],
                ]
            ),
            torch.tensor(
                [
                    [4.0, 3.0, 2.0, 1.0],
                    [-float("inf"), -float("inf"), -float("inf"), -float("inf")],
                ]
            ),
        ]
    )
    global_lse = torch.logsumexp(all_lses, dim=0)
    weights = torch.where(
        torch.isfinite(global_lse).unsqueeze(0),
        torch.exp(all_lses - global_lse.unsqueeze(0)),
        torch.zeros_like(all_lses),
    )
    expected_output = (all_outputs * weights.unsqueeze(-1)).sum(dim=0)
    start = rank * (heads // 2)
    expected_output = expected_output[:, start : start + heads // 2]
    expected_lse = global_lse[:, start : start + heads // 2]
    return {
        "output": torch.allclose(output, expected_output),
        "lse": torch.allclose(lse[0], expected_lse[0]) and torch.isneginf(lse[1]).all().item(),
        "empty_output": torch.equal(output[1], torch.zeros_like(output[1])),
    }


@pytest.mark.parametrize("backend", ["ag_rs", "a2a"])
def test_distributed_dcp_backends_match_reference(backend):
    seen = run_on_tp_ranks(
        _distributed_merge,
        tp_size=2,
        backend="gloo",
        decode_context_parallel_size=2,
        dcp_comm_backend=backend,
    )
    assert seen == [
        {"output": True, "lse": True, "empty_output": True},
        {"output": True, "lse": True, "empty_output": True},
    ]


def test_dcp_world_of_one_is_an_identity():
    output = torch.randn(2, 3, 4)
    lse = torch.randn(2, 3)

    merged_output, merged_lse = merge_dcp_attention(output, lse)

    assert merged_output is output
    assert merged_lse is lse


@pytest.mark.parametrize("dcp", [1, 2, 4, 8])
def test_local_lengths_partition_every_history(dcp):
    """The shares of a length-``L`` history are disjoint and cover it: summed
    over ranks the local lengths give ``L``, for ints and tensors alike, and a
    rank past the end of a short history gets 0 rather than a negative length.
    """
    lengths = torch.arange(0, 70, dtype=torch.int32)
    shares = torch.stack([dcp_local_seq_len(lengths.clone(), r, dcp) for r in range(dcp)])
    assert torch.equal(shares.sum(dim=0), lengths)
    assert (shares >= 0).all()
    for length in (0, 1, dcp - 1, 37):
        local = [dcp_local_seq_len(length, r, dcp) for r in range(dcp)]
        assert local == [len(range(r, length, dcp)) for r in range(dcp)]


def test_block_rows_agree_with_the_local_lengths():
    """Row ``offset // dcp`` of a block's share is exactly the owned position's
    index in the rank's local history -- the identity the strided table view
    and ``dcp_local_seq_len`` both rely on.
    """
    dcp, block_size = 4, 16
    for rank in range(dcp):
        rows = dcp_block_rows(5, block_size, rank, dcp)
        owned = [rows[p] - 5 * block_size // dcp for p in range(rank, block_size, dcp)]
        assert owned == list(range(dcp_local_seq_len(block_size, rank, dcp)))
        assert all(rows[p] == 0 for p in range(block_size) if p % dcp != rank)


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,dcp,expected",
    [
        pytest.param(7, 2, 2, [(0, 1), (1, 2)], id="qwen2.5-tp2-dcp2"),
        pytest.param(8, 4, 2, [(0, 2), (2, 4)], id="qwen3-tp2-dcp2"),
        pytest.param(4, 8, 4, [(0, 2), (2, 4), (4, 6), (6, 8)], id="qwen3-tp4-dcp4"),
        pytest.param(2, 1, 2, [(0, 1), (0, 1)], id="mqa"),
    ],
)
def test_kv_head_range_follows_the_gqa_pairing(num_heads, num_kv_heads, dcp, expected):
    assert [dcp_kv_head_range(num_heads, num_kv_heads, r, dcp) for r in range(dcp)] == expected


def test_kv_head_range_refuses_query_heads_straddling_kv_heads():
    # 3 local query heads at GQA ratio 2: rank 1 owns heads 3..5, half of KV head 1.
    with pytest.raises(ValueError, match="whole KV heads"):
        dcp_kv_head_range(3, 3, 1, 2)


# --------------------------------------------------------------------------- #
# What the engine refuses to combine with context parallelism
# --------------------------------------------------------------------------- #
def _build_engine_on(monkeypatch, **widths):
    """``LLMEngine`` as rank 0 of a patched TP=2 grid; refusals fire before any I/O."""
    from rapid_llm.distributed.topology import ParallelTopology
    from rapid_llm.engine import llm_engine

    topology = ParallelTopology(global_rank=0, tp_size=2, dp_size=1, **widths)
    monkeypatch.setattr(llm_engine, "get_parallel_topology", lambda: topology)
    return llm_engine.LLMEngine(
        checkpoints_dir="/nonexistent",
        tensor_parallel_size=2,
        prefill_context_parallel_size=topology.pcp_size,
        decode_context_parallel_size=topology.dcp_size,
        device="cpu",
    )


def test_engine_refuses_prefill_context_parallelism(monkeypatch):
    """The PCP groups form, but no layer shards over them yet; serving must not pretend."""
    with pytest.raises(NotImplementedError, match="prefill context parallelism"):
        _build_engine_on(monkeypatch, pcp_size=2)


def test_engine_refuses_two_batch_overlap_under_dcp(monkeypatch):
    """TBO's halves would issue the DCP merge collectives out of step across ranks."""
    from rapid_llm.batch_overlap.two_batch_overlap import TBO_ENV, reset_tbo_policy

    monkeypatch.setenv(TBO_ENV, "1")
    reset_tbo_policy()
    try:
        with pytest.raises(ValueError, match="two-batch overlap"):
            _build_engine_on(monkeypatch, dcp_size=2)
    finally:
        monkeypatch.delenv(TBO_ENV)
        reset_tbo_policy()
