"""MoE backend enums + resolver (mirrors sglang ``srt/layers/moe/utils.py``).

sglang keys its dispatcher/runner registries on ``(MoeA2ABackend, MoeRunnerBackend)``.
rapid_llm has one all-to-all dispatch backend and one Triton runner, so these
enums are deliberately short — the extra members are the extension slots a
second backend would register into, not machinery in use today.
"""

from __future__ import annotations

from enum import Enum

from ...distributed.parallel_state import expert_parallel_enabled, get_ep_world_size


class MoeA2ABackend(Enum):
    """The token-dispatch (all-to-all) backend, keyed by the comm path.

    ``NONE`` is the TP / non-EP passthrough (every rank already holds all
    tokens); ``ALL_TO_ALL`` is the EP path that shuffles tokens to the ranks
    owning their experts. Corresponds to sglang's ``MoeA2ABackend`` (whose
    ``DEEPEP`` member is the same role rapid_llm's ``ALL_TO_ALL`` fills).
    """

    NONE = "none"
    ALL_TO_ALL = "all_to_all"


class MoeRunnerBackend(Enum):
    """The expert-compute backend. rapid_llm's fused grouped GEMM is Triton."""

    TRITON = "triton"


def get_moe_a2a_backend() -> MoeA2ABackend:
    """Resolve the active dispatch backend from the parallel state.

    ``ALL_TO_ALL`` when expert parallelism is enabled over a group wider than
    one; ``NONE`` otherwise (dense TP, or EP degree 1).
    """
    if expert_parallel_enabled() and get_ep_world_size() > 1:
        return MoeA2ABackend.ALL_TO_ALL
    return MoeA2ABackend.NONE
