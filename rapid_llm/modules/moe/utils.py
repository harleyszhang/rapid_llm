"""MoE backend enums + resolver (mirrors sglang ``srt/layers/moe/utils.py``).

sglang keys its dispatcher/runner registries on ``(MoeA2ABackend, MoeRunnerBackend)``.
rapid_llm now has two EP token-exchange backends -- the all-to-all dispatcher and
the AgRs v-collectives (see ``distributed.dp_attention``) -- and one Triton
runner, so these enums are deliberately short.
"""

from __future__ import annotations

import os
from enum import Enum

from ...distributed.parallel_state import (
    dp_attention_enabled,
    expert_parallel_enabled,
    get_ep_world_size,
)


class MoeA2ABackend(Enum):
    """The token-dispatch (all-to-all) backend, keyed by the comm path.

    ``NONE`` is the TP / non-EP passthrough (every rank already holds all
    tokens); ``ALL_TO_ALL`` is the EP path that shuffles tokens to the ranks
    owning their experts. ``ALLGATHER_REDUCESCATTER`` is vLLM's default
    backend of the same name: the batch pools with a ragged all-gather (the
    routing results travel with the hidden states) and combines with a
    reduce-scatter. Corresponds to sglang's ``MoeA2ABackend`` (whose ``DEEPEP``
    member is the same role rapid_llm's ``ALL_TO_ALL`` fills).
    """

    NONE = "none"
    ALL_TO_ALL = "all_to_all"
    ALLGATHER_REDUCESCATTER = "allgather_reducescatter"


class MoeRunnerBackend(Enum):
    """The expert-compute backend. rapid_llm's fused grouped GEMM is Triton."""

    TRITON = "triton"


def get_moe_a2a_backend() -> MoeA2ABackend:
    """Resolve the active dispatch backend from the parallel state.

    Under DP-attention the EP group spans the replicas, which makes
    ``ALLGATHER_REDUCESCATTER`` the default: the route runs locally and pools
    with the hidden states, the pool is ragged (no pad rows burn expert FLOPs),
    and no dispatcher is needed -- the exchange lives in
    ``distributed.dp_attention``. ``RAPID_MOE_A2A_BACKEND=all_to_all`` falls
    back to the a2a pooling for A/B. Without DP-attention the EP group is one
    replica's TP group and ``ALL_TO_ALL`` is the established path; ``NONE``
    otherwise (dense TP, or EP degree 1).
    """
    if not (expert_parallel_enabled() and get_ep_world_size() > 1):
        return MoeA2ABackend.NONE
    if not dp_attention_enabled():
        return MoeA2ABackend.ALL_TO_ALL
    if os.environ.get("RAPID_MOE_A2A_BACKEND") == "all_to_all":
        return MoeA2ABackend.ALL_TO_ALL
    return MoeA2ABackend.ALLGATHER_REDUCESCATTER
