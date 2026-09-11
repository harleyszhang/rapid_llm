"""Secondary-tier protocol: what a disk or remote medium must implement.

v0.12.0 defines the seam but ships no implementation — the CPU tier answers
every lookup synchronously, so nothing exercises these paths yet. The
definitions live here (rather than being deferred) because a secondary tier's
shape constrains the primary's: knowing that a disk tier answers lookups
*asynchronously* is what keeps the CPU tier's HIT/MISS answer in the protocol
as a complete one, and knowing that stores outlive the process is why keys
are content hashes rather than block ids.

A secondary tier is deliberately *dumber* than the primary: it is handed
transfer jobs whose block ids are already chosen (the primary owns the block
pool that receives them) and answers with finished jobs in batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .base import Locality, LookupResult, Medium, OffloadKey


@dataclass(frozen=True)
class TransferJob:
    """What one secondary-tier transfer moves.

    Attributes:
        job_id: Unique id within the tier; echoed in the result.
        keys: Content keys of the blocks being moved.
        is_store: True to push primary -> secondary, False to pull back.
        stage_blocks: Primary-side (CPU-pool) block ids, parallel to ``keys``:
            the source of a store and the destination of a load.
    """

    job_id: int
    keys: tuple[OffloadKey, ...]
    is_store: bool
    stage_blocks: tuple[int, ...]


@dataclass(frozen=True)
class JobResult:
    """A finished transfer, as the tier reports it.

    Attributes:
        job_id: The id of the job that finished.
        success: False when the medium failed; partial data is never staged.
        keys: The keys moved, parallel to the originating job.
        transfer_bytes: Payload size, for the bandwidth counters.
        elapsed: Wall time in seconds, for the latency counters.
    """

    job_id: int
    success: bool
    keys: tuple[OffloadKey, ...] = ()
    transfer_bytes: int = 0
    elapsed: float = 0.0


class SecondaryTierManager(ABC):
    """One secondary tier (a medium at a locality), driven by the primary.

    Implementations own their own concurrency: :meth:`submit_store` and
    :meth:`submit_load` must return promptly, and finished work is collected
    by polling :meth:`get_finished_jobs`. Nothing here blocks the engine.
    """

    #: What this tier's bytes sit on.
    medium: Medium
    #: Where that medium lives, relative to this engine.
    locality: Locality

    @abstractmethod
    def lookup(self, key: OffloadKey) -> LookupResult:
        """Whether one key is resident on this tier.

        Unlike the primary's synchronous answer, a tier whose index lives
        off-process answers HIT_PENDING for "asked, answer on the way" — the
        caller must treat it as a miss that may become a hit.
        """

    @abstractmethod
    def submit_store(self, job: TransferJob) -> None:
        """Queue a primary -> secondary transfer; returns immediately."""

    @abstractmethod
    def submit_load(self, job: TransferJob) -> None:
        """Queue a secondary -> primary transfer; returns immediately."""

    @abstractmethod
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Drain and return every job finished since the last call."""

    @abstractmethod
    def touch(self, keys: Sequence[OffloadKey]) -> None:
        """Refresh recency on this tier, without moving data."""

    @abstractmethod
    def drain_jobs(self) -> None:
        """Wait for every queued job to finish, for shutdown."""
