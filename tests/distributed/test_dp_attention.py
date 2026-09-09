"""DP-attention over a real four-rank grid: attention per replica, experts pooled.

Four tiers of claim. The first three are on gloo/CPU because what DP-attention
adds is *topology and data movement*, not numerics:

* **the grid** — a ``dp2 x tp2`` world builds a DP group per TP lane and one EP
  group spanning all four ranks, so eight experts land two per rank rather than
  four per rank twice over. This is the difference between vLLM's DP-attention
  topology and this repo's replica-level DP, and it is the whole reason
  ``get_ep_world_size()`` had to stop being a synonym for the TP world size.
* **the pooling** — :func:`dp_gather` / :func:`dp_scatter` round-trip a ragged
  batch (each replica brings a different number of tokens) through the padded
  rectangle the exchange needs.
* **the numerics** — a :class:`SparseMoeBlock` forward under ``dp2 x tp2`` with
  EP equals the same block with every expert local, fed the concatenation of
  both replicas' tokens. That is the property that makes DP-attention an
  optimisation rather than a different model: pooling the tokens changes which
  rank computes an expert, never what the expert computes.

The fourth is the same numerics claim on **nccl / 2 GPUs**, which is not
redundant with the gloo tier: the pooled all-gather and the expert exchange run
on a different transport, and the routed stage lands on the fused grouped GEMM
in bfloat16 rather than a reference CPU path.

Usage:
    pytest tests/distributed/test_dp_attention.py
"""

from __future__ import annotations

import json
import os
import tempfile

import torch

from rapid_llm.distributed import parallel_state as ps
from tests.distributed.tp_harness import needs_gpus, run_on_tp_ranks

_NUM_EXPERTS = 8
_HIDDEN = 64
_INTER = 32
_TOP_K = 2

#: Tokens each replica brings to the step. Deliberately unequal: the padded
#: rectangle :attr:`DPMetadata.padded_tokens` builds is the only reason a
#: ragged batch can ride one equal-split all-gather, so a test where every
#: replica brought the same count would never exercise it.
_ROWS_PER_REPLICA = (3, 5)

#: Tolerance for the fp32 tiers. Not epsilon, and not a fudge: the pooled path
#: and the all-local reference sum the same products in a different order (the
#: exchange groups rows by owning expert, and EP applies the routing weight on
#: the sender), so the two disagree by fp32 rounding on a reduction of
#: ``_HIDDEN`` + ``_INTER`` terms. What the test is licensed to claim is that
#: the difference stays at that scale; a wrong expert or a mis-ordered gather
#: moves whole rows by O(1), which this still catches by three orders of
#: magnitude.
_FP32_TOL = {"rtol": 1e-3, "atol": 2e-4}

#: Tolerance for the device tier, where the routed stage is a bfloat16 grouped
#: GEMM. Matches the EP tests' tolerance for the same kernel.
_BF16_TOL = {"rtol": 2e-2, "atol": 2e-2}

_CONFIG_BODY = {
    "model_type": "deepseek_v2",
    "torch_dtype": "float32",
    "hidden_size": _HIDDEN,
    "intermediate_size": 128,
    "moe_intermediate_size": _INTER,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "n_shared_experts": 0,
    "n_routed_experts": _NUM_EXPERTS,
    "num_experts_per_tok": _TOP_K,
    "routed_scaling_factor": 1.0,
    "first_k_dense_replace": 0,
    "norm_topk_prob": True,
    "kv_lora_rank": 16,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 32,
    "v_head_dim": 32,
    "vocab_size": 128,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _make_config(torch_dtype: str = "float32"):
    """A ``ModelConfig`` for the MoE body above, from a throwaway config.json.

    ``float32`` for the gloo tiers, where the point is exact data movement and a
    tight tolerance keeps the assertion honest; ``bfloat16`` for the device tier,
    which has to run the dtype the fused grouped GEMM is written for.
    """
    from rapid_llm.models.config import ModelConfig

    with tempfile.TemporaryDirectory() as directory:
        with open(os.path.join(directory, "config.json"), "w") as handle:
            json.dump({**_CONFIG_BODY, "torch_dtype": torch_dtype}, handle)
        return ModelConfig.from_pretrained(directory, max_seq_len=128)


def _global_weights(dtype: torch.dtype):
    """Router + stacked experts, identical on every rank: they *are* the model."""
    gen = torch.Generator().manual_seed(1234)
    gate_up = torch.randn(_NUM_EXPERTS, 2 * _INTER, _HIDDEN, generator=gen) * 0.05
    down = torch.randn(_NUM_EXPERTS, _HIDDEN, _INTER, generator=gen) * 0.05
    router = torch.randn(_NUM_EXPERTS, _HIDDEN, generator=gen) * 0.1
    return gate_up.to(dtype), down.to(dtype), router.to(dtype)


def _replica_tokens(replica: int, dtype: torch.dtype) -> torch.Tensor:
    """The batch replica ``replica`` is serving.

    Keyed on the *replica*, not the global rank: TP peers inside a replica hold
    the same requests and only differ in which slice of the weights they own, so
    a per-rank batch would be a state DP-attention cannot be in.
    """
    gen = torch.Generator().manual_seed(100 + replica)
    return (torch.randn(_ROWS_PER_REPLICA[replica], _HIDDEN, generator=gen) * 0.5).to(dtype)


def _loaded_block(config, *, dtype: torch.dtype):
    """A :class:`SparseMoeBlock` holding this rank's share of the global weights."""
    from rapid_llm.modules.moe import SparseMoeBlock

    block = SparseMoeBlock(config)
    gate_up, down, router = _global_weights(dtype)
    with torch.no_grad():
        block.gate_weight.copy_(router)
    stacked_gate_up = block.experts["gate_up_proj"]
    stacked_down = block.experts["down_proj"]
    gate_half, up_half = gate_up.chunk(2, dim=1)
    for expert in range(_NUM_EXPERTS):
        block._expert_loader(stacked_gate_up, gate_half[expert], (expert, 0))
        block._expert_loader(stacked_gate_up, up_half[expert], (expert, 1))
        block._expert_loader(stacked_down, down[expert], (expert, 2))
    return block


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def _topology(rank: int) -> dict:
    """What groups this rank sees, and how the experts were placed in them."""
    import torch.distributed as dist

    from rapid_llm.modules.moe import SparseMoeBlock

    block = SparseMoeBlock(_make_config())
    dp_group = ps.get_data_parallel_group()
    ep_group = ps.get_ep_group()
    return {
        "rank": rank,
        "dp_attention": ps.dp_attention_enabled(),
        "dp_ranks": dist.get_process_group_ranks(dp_group),
        "ep_ranks": dist.get_process_group_ranks(ep_group),
        "ep_rank": ps.get_ep_rank(),
        "ep_world": ps.get_ep_world_size(),
        "num_local_experts": block.num_local_experts,
        "expert_offset": block.expert_offset,
        # Under EP the intermediate dimension is *not* TP-split: the two expert
        # splits are mutually exclusive, and a widened EP group must not change
        # that.
        "moe_intermediate": block.moe_intermediate_size,
    }


def test_grid_builds_dp_groups_and_a_grid_wide_ep_group():
    """``dp2 x tp2``: a DP group per TP lane, one EP group over all four ranks."""
    seen = run_on_tp_ranks(
        _topology,
        tp_size=2,
        dp_size=2,
        backend="gloo",
        enable_expert_parallel=True,
        enable_dp_attention=True,
    )
    assert all(entry["dp_attention"] for entry in seen)
    # Lane 0 of both replicas, then lane 1 -- vLLM's transpose of the (DP, TP)
    # plane, not the contiguous slices the TP groups take.
    assert [entry["dp_ranks"] for entry in seen] == [[0, 2], [1, 3], [0, 2], [1, 3]]
    assert all(entry["ep_ranks"] == [0, 1, 2, 3] for entry in seen)
    assert [entry["ep_rank"] for entry in seen] == [0, 1, 2, 3]
    assert all(entry["ep_world"] == 4 for entry in seen)
    # Eight experts over the grid: two per rank, not four per rank twice over.
    assert all(entry["num_local_experts"] == 2 for entry in seen)
    assert [entry["expert_offset"] for entry in seen] == [0, 2, 4, 6]
    assert all(entry["moe_intermediate"] == _INTER for entry in seen)


def _replica_dp_stays_replica_dp(rank: int) -> dict:
    """The same ``dp2 x tp2`` world *without* DP-attention: nothing widens."""
    import torch.distributed as dist

    return {
        "dp_attention": ps.dp_attention_enabled(),
        "dp_group": ps.get_data_parallel_group() is None,
        "ep_ranks": dist.get_process_group_ranks(ps.get_ep_group()),
        "ep_world": ps.get_ep_world_size(),
    }


def test_replica_dp_keeps_ep_inside_one_replica():
    """Without DP-attention the EP group is still the replica's TP group.

    The regression guard for widening it: replica-level DP means two independent
    models, so pooling tokens across the axis would mix two engines' batches.
    """
    seen = run_on_tp_ranks(
        _replica_dp_stays_replica_dp,
        tp_size=2,
        dp_size=2,
        backend="gloo",
        enable_expert_parallel=True,
    )
    assert not any(entry["dp_attention"] for entry in seen)
    assert all(entry["dp_group"] for entry in seen)
    assert [entry["ep_ranks"] for entry in seen] == [[0, 1], [0, 1], [2, 3], [2, 3]]
    assert all(entry["ep_world"] == 2 for entry in seen)


# --------------------------------------------------------------------------- #
# the pooling
# --------------------------------------------------------------------------- #
def _gather_scatter_round_trip(rank: int) -> dict:
    """A ragged batch through the padded rectangle and back."""
    from rapid_llm.distributed.dp_attention import dp_attention_region, dp_gather, dp_scatter

    replica = ps.get_data_parallel_rank()
    rows = _ROWS_PER_REPLICA[replica]
    # Row content identifies its owner: replica r's row i is filled with r*10+i,
    # so a mis-ordered gather is visible rather than merely wrong by epsilon.
    mine = torch.arange(rows, dtype=torch.float32).add(10 * replica).unsqueeze(-1).repeat(1, 4)
    with dp_attention_region(rows) as metadata:
        pooled = dp_gather(mine, metadata)
        back = dp_scatter(pooled, metadata)
        return {
            "counts": list(metadata.num_tokens_across_dp),
            "padded": metadata.padded_tokens,
            "offset": metadata.local_offset,
            "pooled": pooled[:, 0].tolist(),
            "round_trip": torch.equal(back, mine),
        }


def test_gather_pads_to_the_widest_replica_and_scatter_inverts_it():
    """Every rank sees both replicas' rows, padded to the wider one, in rank order."""
    seen = run_on_tp_ranks(
        _gather_scatter_round_trip,
        tp_size=2,
        dp_size=2,
        backend="gloo",
        enable_expert_parallel=True,
        enable_dp_attention=True,
    )
    assert all(entry["counts"] == list(_ROWS_PER_REPLICA) for entry in seen)
    assert all(entry["padded"] == max(_ROWS_PER_REPLICA) for entry in seen)
    assert all(entry["round_trip"] for entry in seen)
    # Replica 0's three rows, two zero pad rows, then replica 1's five.
    expected = [0.0, 1.0, 2.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    assert all(entry["pooled"] == expected for entry in seen)
    # Lane 0 of a replica and lane 1 of the same replica pool the same batch.
    assert [entry["offset"] for entry in seen] == [0, 0, 5, 5]


# --------------------------------------------------------------------------- #
# the numerics
# --------------------------------------------------------------------------- #
def _moe_under_dp_attention(rank: int) -> list:
    """This rank's MoE output for its replica's batch, pooled across the DP axis."""
    from rapid_llm.distributed.dp_attention import dp_attention_region

    config = _make_config()
    block = _loaded_block(config, dtype=config.dtype)
    tokens = _replica_tokens(ps.get_data_parallel_rank(), config.dtype)
    with torch.no_grad(), dp_attention_region(tokens.shape[0]):
        return block(tokens).float().tolist()


def _moe_all_experts_local(rank: int) -> list:
    """The reference: one process, every expert local, both batches concatenated."""
    config = _make_config()
    block = _loaded_block(config, dtype=config.dtype)
    tokens = torch.cat([_replica_tokens(r, config.dtype) for r in range(len(_ROWS_PER_REPLICA))])
    with torch.no_grad():
        return block(tokens).float().tolist()


def test_dp_attention_moe_matches_a_single_process_forward():
    """``dp2 x tp2`` + EP == one process holding every expert, fed both batches.

    The per-rank outputs concatenate back into the reference in replica order,
    and the two TP lanes of a replica agree exactly -- they hold the same
    requests, so a disagreement would mean the pooling depended on which slice
    of the attention weights a rank owns.
    """
    sharded = run_on_tp_ranks(
        _moe_under_dp_attention,
        tp_size=2,
        dp_size=2,
        backend="gloo",
        enable_expert_parallel=True,
        enable_dp_attention=True,
    )
    (reference,) = run_on_tp_ranks(_moe_all_experts_local, tp_size=1, backend="gloo")

    assert sharded[0] == sharded[1], "TP lanes of replica 0 disagree"
    assert sharded[2] == sharded[3], "TP lanes of replica 1 disagree"
    pooled = torch.tensor(sharded[0] + sharded[2])
    expected = torch.tensor(reference)
    assert pooled.shape == expected.shape
    torch.testing.assert_close(pooled, expected, **_FP32_TOL)


def _moe_dp_only(rank: int) -> list:
    """``dp2 x tp1`` + EP: the minimal DeepSeek topology, two ranks, four experts each."""
    from rapid_llm.distributed.dp_attention import dp_attention_region

    config = _make_config()
    block = _loaded_block(config, dtype=config.dtype)
    assert block.num_local_experts == _NUM_EXPERTS // 2
    tokens = _replica_tokens(ps.get_data_parallel_rank(), config.dtype)
    with torch.no_grad(), dp_attention_region(tokens.shape[0]):
        return block(tokens).float().tolist()


def test_dp_attention_without_tensor_parallelism():
    """``dp2 x tp1`` still builds a DP group: pure DP-attention needs no TP.

    The case the old ``tp_size <= 1`` early return in ``init_parallel`` made
    unreachable, and the smallest topology that is recognisably sglang's
    DeepSeek deployment -- attention data-parallel, experts split across the
    same ranks.
    """
    sharded = run_on_tp_ranks(
        _moe_dp_only,
        tp_size=1,
        dp_size=2,
        backend="gloo",
        enable_expert_parallel=True,
        enable_dp_attention=True,
    )
    (reference,) = run_on_tp_ranks(_moe_all_experts_local, tp_size=1, backend="gloo")
    pooled = torch.tensor(sharded[0] + sharded[1])
    torch.testing.assert_close(pooled, torch.tensor(reference), **_FP32_TOL)


# --------------------------------------------------------------------------- #
# nccl / 2 GPUs: the same pooling on the real transport and the fused GEMM
# --------------------------------------------------------------------------- #
def _moe_dp_only_cuda(rank: int) -> list:
    """``dp2 x tp1`` + EP on device: pooled all-gather over nccl, experts fused."""
    from rapid_llm.distributed.dp_attention import dp_attention_region

    device = f"cuda:{rank}"
    config = _make_config("bfloat16")
    # Weights are loaded on the host and then moved: ``_expert_loader`` writes
    # host slices, and doing that into device parameters would copy per expert.
    block = _loaded_block(config, dtype=config.dtype).to(device).eval()
    assert block.num_local_experts == _NUM_EXPERTS // 2
    tokens = _replica_tokens(ps.get_data_parallel_rank(), config.dtype).to(device)
    with torch.no_grad(), dp_attention_region(tokens.shape[0]):
        return block(tokens).float().cpu().tolist()


def _moe_all_experts_local_cuda(rank: int) -> list:
    """The device reference: one rank, every expert local, both batches concatenated."""
    device = f"cuda:{rank}"
    config = _make_config("bfloat16")
    block = _loaded_block(config, dtype=config.dtype).to(device).eval()
    assert block.num_local_experts == _NUM_EXPERTS
    tokens = torch.cat(
        [_replica_tokens(r, config.dtype) for r in range(len(_ROWS_PER_REPLICA))]
    ).to(device)
    with torch.no_grad():
        return block(tokens).float().cpu().tolist()


@needs_gpus(2)
def test_dp_attention_moe_on_device_matches_all_experts_local():
    """``dp2 x tp1`` + EP over nccl == one device holding every expert.

    Tolerance is the bfloat16 grouped-GEMM tolerance used elsewhere in the EP
    tests, not the gloo tier's: the two sides reduce the same products in a
    different order, so agreeing to fp32 epsilon was never the claim.
    """
    sharded = run_on_tp_ranks(
        _moe_dp_only_cuda,
        tp_size=1,
        dp_size=2,
        backend="nccl",
        enable_expert_parallel=True,
        enable_dp_attention=True,
    )
    (reference,) = run_on_tp_ranks(_moe_all_experts_local_cuda, tp_size=1, backend="nccl")
    pooled = torch.tensor(sharded[0] + sharded[1])
    expected = torch.tensor(reference)
    assert pooled.shape == expected.shape
    torch.testing.assert_close(pooled, expected, **_BF16_TOL)
