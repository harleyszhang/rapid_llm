"""Runner orchestrator (stages 3-4): fused-func fast path, else pre → core → post.

Mirrors sglang's ``moe_runner/runner.py``. :class:`MoeRunner` keys the pools on
``(a2a_backend, runner_backend)``: a registered fully-fused func short-circuits
the whole expert stage; otherwise it runs the registered pre-permute, the core's
grouped GEMM, then the post-permute. rapid_llm registers a Triton core and
identity permutes, so ``run`` is the grouped GEMM with two no-op hook points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils import MoeA2ABackend, MoeRunnerBackend
from .base import FusedOpPool, MoeRunnerConfig, MoeRunnerCore, PermuteMethodPool

if TYPE_CHECKING:
    import torch

#: runner backend -> core class, populated by :func:`register_moe_runner_core`.
_CORE_REGISTRY: dict[MoeRunnerBackend, type[MoeRunnerCore]] = {}


def register_moe_runner_core(backend: MoeRunnerBackend):
    """Register a :class:`MoeRunnerCore` under ``backend`` (sglang decorator style)."""

    def decorator(cls: type[MoeRunnerCore]) -> type[MoeRunnerCore]:
        _CORE_REGISTRY[backend] = cls
        return cls

    return decorator


class MoeRunner:
    """Expert-compute orchestrator: resolves the core + permutes for a config."""

    def __init__(self, config: MoeRunnerConfig, a2a_backend: MoeA2ABackend) -> None:
        self.config = config
        self.a2a_backend = a2a_backend
        self.runner_backend = config.runner_backend
        try:
            core_cls = _CORE_REGISTRY[self.runner_backend]
        except KeyError as exc:
            raise KeyError(f"no MoE runner core registered for {self.runner_backend}") from exc
        self.core = core_cls(config)

    def run(
        self,
        block: torch.nn.Module,
        local_x: torch.Tensor,
        local_ids: torch.Tensor,
        local_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped GEMM over ``local_x``; a fused func or permutes fire if registered."""
        key = (self.a2a_backend, self.runner_backend)
        fused = FusedOpPool.get(key)
        if fused is not None:
            return fused(block, local_x, local_ids, local_weights)
        pre = PermuteMethodPool["pre"].get(key)
        if pre is not None:
            local_x, local_ids, local_weights = pre(local_x, local_ids, local_weights)
        out = self.core.run(block, local_x, local_ids, local_weights)
        post = PermuteMethodPool["post"].get(key)
        if post is not None:
            out = post(out)
        return out
