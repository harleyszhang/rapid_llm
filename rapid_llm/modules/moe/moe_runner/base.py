"""Expert-compute seam (stages 3-4): runner config, core ABC + permute pools.

Mirrors sglang's ``moe_runner/base.py``: a :class:`MoeRunnerConfig` (the shapes
the core needs), a :class:`MoeRunnerCore` ABC (the grouped-GEMM backend), and the
decorator-registry pools that key a fully-fused func / pre-permute / post-permute
on ``(MoeA2ABackend, MoeRunnerBackend)``. rapid_llm's dispatch already hands the
runner the layout it wants, so the registered permutes are identity — the pools
are the extension slots a layout-shuffling backend (DeepEP low-latency, pplx)
would fill, not machinery in use today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..utils import MoeA2ABackend, MoeRunnerBackend

if TYPE_CHECKING:
    import torch


@dataclass
class MoeRunnerConfig:
    """The shapes/placement a runner core needs, read off the block's config."""

    num_experts: int
    num_local_experts: int
    expert_offset: int
    hidden_size: int
    top_k: int
    moe_intermediate_size: int
    runner_backend: MoeRunnerBackend = MoeRunnerBackend.TRITON


class MoeRunnerCore(ABC):
    """The expert-compute backend: grouped GEMM over the received batch."""

    def __init__(self, config: MoeRunnerConfig) -> None:
        self.config = config

    @abstractmethod
    def run(
        self,
        block: torch.nn.Module,
        local_x: torch.Tensor,
        local_ids: torch.Tensor,
        local_weights: torch.Tensor,
        down_overlap_args=None,
    ) -> torch.Tensor:
        """Run the local experts over ``local_x`` (pad rows carry -1 ids)."""


# --------------------------------------------------------------------------- #
# Registries: (a2a_backend, runner_backend) -> fully-fused func / permute hooks.
# --------------------------------------------------------------------------- #
#: The fully-fused fast path (dispatch+GEMM+combine in one). rapid_llm registers
#: none, so the runner always takes the pre_permute -> core -> post_permute path.
FusedOpPool: dict[tuple[MoeA2ABackend, MoeRunnerBackend], Callable] = {}
#: Layout adapters between the dispatch output and the core's expected input/output.
PermuteMethodPool: dict[str, dict[tuple[MoeA2ABackend, MoeRunnerBackend], Callable]] = {
    "pre": {},
    "post": {},
}


def register_fused_func(a2a_backend: MoeA2ABackend, runner_backend: MoeRunnerBackend):
    """Register a fully-fused ``(dispatch_output, ...) -> combine_input`` func."""

    def decorator(func: Callable) -> Callable:
        FusedOpPool[(a2a_backend, runner_backend)] = func
        return func

    return decorator


def register_pre_permute(a2a_backend: MoeA2ABackend, runner_backend: MoeRunnerBackend):
    """Register the pre-GEMM layout adapter for ``(a2a, runner)``."""

    def decorator(func: Callable) -> Callable:
        PermuteMethodPool["pre"][(a2a_backend, runner_backend)] = func
        return func

    return decorator


def register_post_permute(a2a_backend: MoeA2ABackend, runner_backend: MoeRunnerBackend):
    """Register the post-GEMM layout adapter for ``(a2a, runner)``."""

    def decorator(func: Callable) -> Callable:
        PermuteMethodPool["post"][(a2a_backend, runner_backend)] = func
        return func

    return decorator
