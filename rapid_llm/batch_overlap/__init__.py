"""Batch overlap: both overlap axes, in sglang's ``batch_overlap`` layout.

Five pieces: ``overlap`` — the host↔device axis (L1: :class:`StreamPool`
async uploads/readbacks, :class:`Timeline` evidence); ``operations`` — the
stage/yield primitives (:class:`YieldOperation`, :class:`StateDict`);
``comm_overlap`` — the comm-stream plumbing both overlap policies ride (L2
deferred all-reduces, L3 chunked all-reduces, EP all-to-all);
``two_batch_overlap`` — the L2 decode ping-pong executor;
``operations_strategy`` — per-layer op streams, the layers' own bound
methods (sglang's decode strategy).

Usage:
    from rapid_llm.batch_overlap import deferred_all_reduce, StateDict
    from rapid_llm.batch_overlap.two_batch_overlap import tbo_policy

The root re-exports only the kernel-free modules: ``modules/`` imports here,
and the eager import boundary (:mod:`tests.test_imports`) forbids pulling the
Triton kernels — call sites import ``two_batch_overlap`` directly.
"""

from .comm_overlap import (
    CommOverlapPolicy,
    CommStreamPool,
    DeferredArContext,
    comm_overlap_policy,
    current_deferred_ar,
    deferred_all_reduce,
    reset_comm_overlap_policy,
    row_parallel_forward,
)
from .operations import (
    StateDict,
    YieldOperation,
    execute_operations,
    execute_overlapped_operations,
)
from .overlap import OverlapPolicy, RegionRecord, StreamPool, Timeline

__all__ = [
    "CommOverlapPolicy",
    "CommStreamPool",
    "DeferredArContext",
    "OverlapPolicy",
    "RegionRecord",
    "StateDict",
    "StreamPool",
    "Timeline",
    "YieldOperation",
    "comm_overlap_policy",
    "current_deferred_ar",
    "deferred_all_reduce",
    "execute_operations",
    "execute_overlapped_operations",
    "reset_comm_overlap_policy",
    "row_parallel_forward",
]
