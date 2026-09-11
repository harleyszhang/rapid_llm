"""KV offloading: a CPU tier behind the GPU prefix cache, and the seam for more.

Layout mirrors vLLM's ``vllm/v1/kv_offload`` on purpose, so the concepts map
one-to-one when reading either codebase:

* :mod:`base` — the vocabulary: tiers (:class:`Medium` x :class:`Locality`),
  keys, lookup results, load/store specs, and the scheduler-side
  :class:`OffloadingManager` protocol.
* :mod:`cpu` — the CPU primary tier, a host block pool fed by async copies.
* :mod:`secondary` — what a disk or remote tier will implement.
* :mod:`tiering` — the engine's single handle; v0.12.0 passes through to the
  CPU tier, and grows the multi-tier orchestration when a second tier lands.

The data path — CUDA streams and the pinned host buffers — lives on the
executor side in :mod:`rapid_llm.executor.kv_offload`, because this package
is deliberately device-free: it can be (and is) unit-tested with no GPU.
"""

from .base import (
    BlockCopier,
    CopyRun,
    LoadSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadEvent,
    OffloadingManager,
    OffloadingStats,
    OffloadKey,
    StoreSpec,
)
from .cpu import CPUPrimaryTierOffloadingManager
from .secondary import JobResult, SecondaryTierManager, TransferJob
from .tiering import TieringOffloadingManager

__all__ = [
    "BlockCopier",
    "CPUPrimaryTierOffloadingManager",
    "CopyRun",
    "JobResult",
    "LoadSpec",
    "Locality",
    "LookupResult",
    "Medium",
    "OffloadEvent",
    "OffloadKey",
    "OffloadingManager",
    "OffloadingStats",
    "SecondaryTierManager",
    "StoreSpec",
    "TieringOffloadingManager",
    "TransferJob",
]
