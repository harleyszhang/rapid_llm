"""Mixture-of-experts package: a five-stage routed sparse FFN, mirroring sglang.

The single ``moe.py`` module was split into the sglang-style layered layout:

* :mod:`.router` -- stage 1 (router GEMM + top-k) -> :class:`TopKRouter`/:class:`TopKOutput`.
* :mod:`.token_dispatcher` -- stages 2 & 5 (dispatch/combine) -> :class:`StandardDispatcher`
  (non-EP passthrough) and :class:`AllToAllDispatcher` (the a2a EP path).
* :mod:`.moe_runner` -- stages 3-4 (expert grouped GEMM) -> :class:`MoeRunner`.
* :mod:`.layer` -- the orchestrator -> :class:`SparseMoeBlock` + :class:`MoEOpContext`.

This ``__init__`` re-exports the names the rest of the codebase and the tests
import from ``rapid_llm.modules.moe`` (``SparseMoeBlock``, ``AllToAllDispatcher``,
``DispatchHandle``, ``MoEOpContext``, ``grouped_topk``, ...), so the split is
invisible to every existing importer. The router GEMM stays patchable at
``rapid_llm.modules.moe.router._router_gemm``.
"""

from __future__ import annotations

from ...kernels import grouped_topk
from .layer import MoEOpContext, SparseMoeBlock
from .router import TopKOutput, TopKRouter
from .token_dispatcher import (
    AllToAllCombineInput,
    AllToAllDispatcher,
    AllToAllDispatchOutput,
    DispatchHandle,
    StandardDispatcher,
)
from .utils import MoeA2ABackend, MoeRunnerBackend, get_moe_a2a_backend

__all__ = [
    "AllToAllCombineInput",
    "AllToAllDispatchOutput",
    "AllToAllDispatcher",
    "DispatchHandle",
    "MoEOpContext",
    "MoeA2ABackend",
    "MoeRunnerBackend",
    "SparseMoeBlock",
    "StandardDispatcher",
    "TopKOutput",
    "TopKRouter",
    "get_moe_a2a_backend",
    "grouped_topk",
]
