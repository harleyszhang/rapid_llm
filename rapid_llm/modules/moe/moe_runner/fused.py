"""Triton fused-MoE runner core (stage 3-4 compute) + its identity permutes.

Wraps the block's quantization method (``quant_method.apply``) — the same call
:meth:`SparseMoeBlock._run_experts` made before the refactor, including the
``down_overlap_args`` fast path that splits the down projection into row chunks
for the unquantized kernel. The pre/post permutes registered here for both a2a
backends are identity: rapid_llm's dispatch already delivers the layout the
grouped GEMM wants, so the hooks exist only as the seam a reshuffling backend
would fill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...quantization import UnquantizedFusedMoEMethod
from ..utils import MoeA2ABackend, MoeRunnerBackend
from .base import (
    MoeRunnerCore,
    register_post_permute,
    register_pre_permute,
)
from .runner import register_moe_runner_core

if TYPE_CHECKING:
    import torch


@register_moe_runner_core(MoeRunnerBackend.TRITON)
class FusedMoERunnerCore(MoeRunnerCore):
    """Grouped GEMM over the received batch via the block's quant method."""

    def run(
        self,
        block: torch.nn.Module,
        local_x: torch.Tensor,
        local_ids: torch.Tensor,
        local_weights: torch.Tensor,
        down_overlap_args=None,
    ) -> torch.Tensor:
        """Run the local experts with the block's quantization method.

        ``down_overlap_args`` splits the down projection into row chunks and
        publishes an event per chunk for the unquantized kernel. Quantized
        kernels use their normal fused down projection.
        """
        if (
            down_overlap_args is None
            or type(block.quant_method) is not UnquantizedFusedMoEMethod
        ):
            return block.quant_method.apply(block, local_x, local_weights, local_ids)

        from ....kernels import fused_moe

        return fused_moe(
            local_x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            local_weights,
            local_ids,
            down_overlap_args=down_overlap_args,
        )


def _identity_pre(local_x, local_ids, local_weights):
    """No layout change: rapid_llm's dispatch already produces the GEMM layout."""
    return local_x, local_ids, local_weights


def _identity_post(out):
    """No layout change on the way out (combine un-permutes, not the runner)."""
    return out


# Both dispatch backends feed the Triton core the layout it wants, so the hooks
# are identity — registered to keep the (a2a, runner) seam populated and honest.
for _a2a in (MoeA2ABackend.ALL_TO_ALL, MoeA2ABackend.NONE):
    register_pre_permute(_a2a, MoeRunnerBackend.TRITON)(_identity_pre)
    register_post_permute(_a2a, MoeRunnerBackend.TRITON)(_identity_post)
