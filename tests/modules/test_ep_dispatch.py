"""AllToAllDispatcher's permutation math, single process (no ``dist``).

With no process group the EP world is one, so ``all_to_all_async`` degenerates
to a local copy and the whole dispatch→experts→combine round trip runs in one
process — exactly the surface this file pins: that the sort/segment permutation
routes each routing slot to the right buffer row, that the return trip un-does
it, and that the sender-side weights land as ``sum_k w_k * expert_k(x)``. The
multi-rank exchange and padding live in ``tests/distributed/test_ep_moe.py``.

Usage:
    pytest tests/modules/test_ep_dispatch.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.modules.moe import AllToAllDispatcher
from tests.registry import import_needs_cuda

import_needs_cuda()


def _grouped_expert(recv_x: torch.Tensor, local_ids: torch.Tensor, weight: torch.Tensor):
    """A pure-torch stand-in for the fused grouped GEMM: ``out[r] = recv_x[r] @ W[id_r]``."""
    return torch.einsum("nh,nhe->ne", recv_x, weight[local_ids.reshape(-1)])


def _reference(x, ids, weights, weight):
    """``sum_k w_k * (x @ W[id_k])`` per token, computed the obvious way."""
    rows, k = ids.shape
    out = torch.zeros(rows, weight.shape[2], dtype=x.dtype)
    for t in range(rows):
        for j in range(k):
            out[t] += weights[t, j] * (x[t] @ weight[ids[t, j]])
    return out


def test_world_of_one_roundtrip_matches_reference():
    """EP=1: dispatch→grouped-expert→combine equals the per-token weighted sum."""
    torch.manual_seed(0)
    rows, hidden, k, num_experts, dout = 6, 8, 2, 6, 4
    x = torch.randn(rows, hidden)
    ids = torch.randint(0, num_experts, (rows, k))
    weights = torch.rand(rows, k)
    weight = torch.randn(num_experts, hidden, dout)

    dispatcher = AllToAllDispatcher(num_experts, num_experts, 0)
    handle, local_x, local_ids, _ = dispatcher.dispatch(x, ids, weights)
    local_out = _grouped_expert(local_x, local_ids, weight)
    out = dispatcher.combine(handle, local_out)

    torch.testing.assert_close(out, _reference(x, ids, weights, weight), atol=1e-5, rtol=1e-5)


def test_dispatched_weights_are_unit_and_applied_on_combine():
    """The expert side sees unit weights; the sender's weights scale on the way back."""
    torch.manual_seed(1)
    rows, hidden, k, num_experts = 4, 6, 2, 4
    x = torch.randn(rows, hidden)
    ids = torch.randint(0, num_experts, (rows, k))
    weights = torch.rand(rows, k)

    dispatcher = AllToAllDispatcher(num_experts, num_experts, 0)
    handle, _local_x, _local_ids, local_weights = dispatcher.dispatch(x, ids, weights)
    # dispatch_b hands the experts unit weights — routing weights stay on the sender.
    assert torch.all(local_weights == 1.0)
    # ...and combine_b applies them: identity expert ⇒ out = sum_k w_k * x.
    out = dispatcher.combine(handle, _local_x)
    expected = weights.sum(dim=1, keepdim=True) * x
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_two_phase_api_matches_synchronous():
    """``dispatch_a``/``dispatch_b`` then ``combine_a``/``combine_b`` == the paired calls."""
    torch.manual_seed(2)
    rows, hidden, k, num_experts, dout = 5, 8, 2, 6, 3
    x = torch.randn(rows, hidden)
    ids = torch.randint(0, num_experts, (rows, k))
    weights = torch.rand(rows, k)
    weight = torch.randn(num_experts, hidden, dout)

    dispatcher = AllToAllDispatcher(num_experts, num_experts, 0)

    handle = dispatcher.dispatch_a(x, ids, weights)
    local_x, local_ids, _ = dispatcher.dispatch_b(handle)
    local_out = _grouped_expert(local_x, local_ids, weight)
    handle = dispatcher.combine_a(handle, local_out)
    split_out = dispatcher.combine_b(handle)

    sync_handle, sx, sid, _ = dispatcher.dispatch(x, ids, weights)
    sync_out = dispatcher.combine(sync_handle, _grouped_expert(sx, sid, weight))

    torch.testing.assert_close(split_out, sync_out, atol=1e-6, rtol=1e-6)


def test_placement_that_does_not_tile_the_group_is_rejected():
    """A world of one owns every expert; ``num_local < num_experts`` with no group
    would index the send buffer out of bounds, so ``dispatch_a`` refuses it.

    (Offset rebasing across a real split is exercised over gloo in
    ``tests/distributed/test_ep_moe.py``, where ``ep_size`` matches.)"""
    torch.manual_seed(3)
    x = torch.randn(4, 6)
    ids = torch.randint(0, 6, (4, 2))
    weights = torch.rand(4, 2)

    dispatcher = AllToAllDispatcher(6, 3, 3)  # second-of-two placement, but no group
    with pytest.raises(ValueError, match="cannot host"):
        dispatcher.dispatch(x, ids, weights)


def test_num_experts_not_divisible_is_rejected():
    with pytest.raises(ValueError, match="do not split"):
        AllToAllDispatcher(7, 3, 0)


@pytest.mark.parametrize(
    "x,ids,weights,message",
    [
        (torch.randn(2, 3), torch.tensor([[0], [4]]), torch.ones(2, 1), "expert ids"),
        (torch.randn(2, 3), torch.zeros(3, 1, dtype=torch.long), torch.ones(3, 1), "rows"),
        (torch.randn(2, 3), torch.zeros(2, 1, dtype=torch.long), torch.ones(2), "2-D routing"),
    ],
)
def test_invalid_routing_is_rejected(x, ids, weights, message):
    with pytest.raises(ValueError, match=message):
        AllToAllDispatcher(4, 4, 0).dispatch_a(x, ids, weights)


def test_custom_unquantized_moe_method_is_not_bypassed():
    """Model-specific epilogues must survive the EP local-expert path."""
    from rapid_llm.modules.moe import SparseMoeBlock
    from rapid_llm.modules.quantization import UnquantizedFusedMoEMethod

    class CustomMethod(UnquantizedFusedMoEMethod):
        def apply(self, block, x, topk_weights, topk_ids):
            return x + 7

    block = object.__new__(SparseMoeBlock)
    block.quant_method = CustomMethod()
    x = torch.randn(2, 4)
    out = SparseMoeBlock._run_experts(
        block,
        x,
        torch.zeros(2, 1, dtype=torch.long),
        torch.ones(2, 1),
    )
    torch.testing.assert_close(out, x + 7)
