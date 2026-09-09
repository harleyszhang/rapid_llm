"""Expert-compute runner: config, core ABC, orchestrator + registries.

Importing this package registers the Triton core and the identity permutes (the
decorators in :mod:`.fused` run on import), so :class:`MoeRunner` can resolve the
core for :class:`~rapid_llm.modules.moe.utils.MoeRunnerBackend.TRITON`.
"""

from __future__ import annotations

from .base import (
    FusedOpPool,
    MoeRunnerConfig,
    MoeRunnerCore,
    PermuteMethodPool,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from .fused import FusedMoERunnerCore
from .runner import MoeRunner, register_moe_runner_core

__all__ = [
    "FusedMoERunnerCore",
    "FusedOpPool",
    "MoeRunner",
    "MoeRunnerConfig",
    "MoeRunnerCore",
    "PermuteMethodPool",
    "register_fused_func",
    "register_moe_runner_core",
    "register_post_permute",
    "register_pre_permute",
]
