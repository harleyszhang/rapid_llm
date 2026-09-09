"""Token dispatchers: the dispatch/combine seam (stages 2 & 5) + its registry.

``BaseDispatcher`` owns the comm path so a routed MoE layer only sees
``dispatch`` (tokens out → this rank's expert batch) and ``combine`` (expert
results → per-token weighted sum). Two backends register on import, keyed by
:class:`~rapid_llm.modules.moe.utils.MoeA2ABackend`: ``standard`` is the
non-EP passthrough — every rank already holds all tokens — and ``all_to_all``
is the EP path over NCCL ``all_to_all_single``. Both ``DispatchOutput`` /
``CombineInput`` carry a format tag so the runner can branch on layout
without isinstance checks.

Usage:
    dispatcher = get_dispatcher(backend)   # registered by importing this package
"""

from __future__ import annotations

from .all_to_all import (
    AllToAllCombineInput,
    AllToAllDispatcher,
    AllToAllDispatchOutput,
    DispatchHandle,
)
from .base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
    get_dispatcher,
    get_dispatcher_class,
    register_dispatcher,
)
from .standard import StandardCombineInput, StandardDispatcher, StandardDispatchOutput

__all__ = [
    "AllToAllCombineInput",
    "AllToAllDispatchOutput",
    "AllToAllDispatcher",
    "BaseDispatcher",
    "CombineInput",
    "CombineInputFormat",
    "DispatchHandle",
    "DispatchOutput",
    "DispatchOutputFormat",
    "StandardCombineInput",
    "StandardDispatchOutput",
    "StandardDispatcher",
    "get_dispatcher",
    "get_dispatcher_class",
    "register_dispatcher",
]
