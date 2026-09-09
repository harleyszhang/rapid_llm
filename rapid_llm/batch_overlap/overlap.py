"""Cross-stream overlap: the L1 policy, its stream pool and its timeline.

The package's host↔device axis. :class:`OverlapPolicy` (``RAPID_LLM_OVERLAP``)
decides whether input uploads run on a copy stream; :class:`StreamPool`
stages the async uploads — and, in the other direction, pinned readbacks —
while :class:`Timeline` records regions as evidence. The siblings
(``comm_overlap``, ``two_batch_overlap``) own the compute↔communication axis
and share this :class:`Timeline`, so copy and comm regions compare on one
device clock.

Usage:
    policy = OverlapPolicy.from_env()
    timeline = Timeline.from_env("cuda"); print(timeline.summary())
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

from ..utils.logger import get_logger

_log = get_logger(__name__)

#: Environment variable switching every overlap site at once (``0`` disables).
OVERLAP_ENV = "RAPID_LLM_OVERLAP"

#: Environment variable turning on per-region CUDA-event recording.
TIMELINE_ENV = "RAPID_LLM_OVERLAP_TIMELINE"


@dataclass(frozen=True)
class OverlapPolicy:
    """One flag for every cross-stream overlap site.

    ``enabled=False`` collapses every path to inline single-stream behaviour (also
    the answer with no CUDA device — the pool is never built).
    """

    enabled: bool = True

    @classmethod
    def from_env(cls) -> OverlapPolicy:
        """Read ``RAPID_LLM_OVERLAP``; anything but ``0``/``false``/``off`` means on."""
        raw = os.environ.get(OVERLAP_ENV, "1").strip().lower()
        return cls(enabled=raw not in ("0", "false", "off"))


class StreamPool:
    """The copy stream and the pinned staging rings overlap traffic rides on.

    Staging buffers are pinned host memory reused through freelists: a buffer
    re-enters rotation only once the event after its last copy completes, else a
    fresh one is allocated (the ring grows to the high-water mark of passes in
    flight). A busy buffer is never force-reused — overwriting bytes a copy engine
    is reading is the classic pinned-memory race. Two rings share one copy stream,
    one per direction:

    * **Upload** (:meth:`upload_async`): host values go out on H2D; the compute
      stream waits on the event before its kernels read them.
    * **Readback** (:meth:`readback_async`): device results come back on D2H, ordered
      after the compute queue at issue time; the *host* waits on the event one step
      later. Its buffer is held out of the ring until :meth:`release_readback`,
      because the copy event says only that the copy engine finished -- not that the
      host has read the view handed out alongside it.

    Args:
        device: Device string the streams belong to.
        policy: Shared switch; disabled makes both methods plain blocking copies
            returning no event, so call sites read the same either way.
        timeline: Where copy regions are recorded; disabled is a no-op.
    """

    def __init__(
        self, device: str, policy: OverlapPolicy, timeline: Timeline | None = None
    ) -> None:
        self._device = torch.device(device)
        self._policy = policy
        self._timeline = timeline if timeline is not None else Timeline()
        self._copy_stream: torch.cuda.Stream | None = None
        # (flat pinned buffer, event for its copy's completion); event is None only
        # for a buffer that never carried a copy (hence free). One deque per
        # direction so uploads and readbacks do not contend at crossed lifetimes.
        self._staging: deque[tuple[torch.Tensor, torch.cuda.Event | None]] = deque()
        self._spill: deque[tuple[torch.Tensor, torch.cuda.Event | None]] = deque()
        # Readback buffers handed out and not yet released, keyed on the storage's
        # address so a returned view can be matched back to its buffer. They sit
        # outside ``_spill`` on purpose: see ``release_readback``.
        self._in_use: dict[int, tuple[torch.Tensor, torch.cuda.Event | None]] = {}

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """The side stream copies are issued on, created on first use."""
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=self._device)
        return self._copy_stream

    def upload_async(
        self, values: Any, *, dtype: torch.dtype, label: str = "upload"
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Upload host values and return ``(device tensor, completion event)``.

        Policy on: values land in a pinned staging buffer and the H2D copy is issued
        on the copy stream (this returns without waiting). Policy off: the legacy
        inline upload, event ``None`` (a :meth:`consume` no-op). Either way the tensor
        keeps ``values``' shape. The device tensor is allocated on the copy stream, so
        a recycled block comes only from an already-finished copy; the compute-stream
        race is covered by :meth:`consume`'s ``record_stream``. ``label`` names the
        timeline region.
        """
        host = torch.as_tensor(values, dtype=dtype)
        flat = host.reshape(-1)
        if not self._policy.enabled:
            return host.to(self._device), None

        staging = self._acquire(self._staging, dtype, flat.numel())
        staging[: flat.numel()].copy_(flat)
        with torch.cuda.stream(self.copy_stream), self._timeline.region(label, "copy"):
            device_tensor = staging[: flat.numel()].to(self._device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
        device_tensor = device_tensor.view(host.shape)
        # The fresh event replaces the buffer's old marker: busy until this copy lands.
        self._staging.append((staging, event))
        return device_tensor, event

    def readback_async(
        self, device_tensor: torch.Tensor, *, label: str = "readback"
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Copy device results into pinned host memory; ``(host view, event)``.

        The D2H copy rides the copy stream, ordered after the compute queue at issue
        time (exactly the kernels that produced ``device_tensor``). The host does not
        wait: it reads the view later, after ``event.synchronize()``, by which point
        the copy landed under the next pass. Policy off: a plain blocking ``.cpu()``,
        event ``None``.

        The returned view aliases a ring buffer, and the holder owes a
        :meth:`release_readback` once it has read it. Recycling on the copy event
        alone would be wrong: the launch/harvest loop issues the *next* pass's copy
        before it reads the previous pass's tokens, so an event-freed buffer would
        already hold the wrong step's values by the time the harvest looked at it.
        """
        if not self._policy.enabled:
            return device_tensor.cpu(), None
        flat = device_tensor.reshape(-1)
        pinned = self._acquire(self._spill, flat.dtype, flat.numel())
        # Grabbed before the copy-stream context: inside it the "current" stream is
        # the copy stream, and waiting on ourselves would order nothing.
        compute = torch.cuda.current_stream(self._device)
        with torch.cuda.stream(self.copy_stream), self._timeline.region(label, "copy"):
            self.copy_stream.wait_stream(compute)
            pinned[: flat.numel()].copy_(flat, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
            # The source is read on the copy stream while the caller may have dropped
            # its reference; without this the allocator could recycle the block and
            # the copy reads garbage — the mirror of consume()'s record_stream.
            flat.record_stream(self.copy_stream)
        self._in_use[pinned.data_ptr()] = (pinned, event)
        return pinned[: flat.numel()].view(device_tensor.shape), event

    def release_readback(self, host_view: torch.Tensor) -> None:
        """Return a readback buffer to the ring once its holder has read it.

        A view whose buffer was never handed out (overlap off, or an already
        released one) is ignored, so the call is safe to make unconditionally.
        """
        entry = self._in_use.pop(host_view.data_ptr(), None)
        if entry is not None:
            self._spill.append(entry)

    def consume(self, event: torch.cuda.Event | None, *tensors: torch.Tensor | None) -> None:
        """Make the current stream wait for an :meth:`upload_async` event.

        Stream-ordered, not host-side: the CPU keeps running and kernels launched
        after this cannot start before the copy lands. ``None`` (overlap off) is a
        no-op. Pass the uploaded tensors along: they are ``record_stream``-ed here,
        since their blocks belong to the copy stream's pool — without the mark a
        tensor freed after its pass could be recycled while this stream reads it.
        """
        if event is None:
            return
        stream = torch.cuda.current_stream(self._device)
        stream.wait_event(event)
        for tensor in tensors:
            if tensor is not None:
                tensor.record_stream(stream)

    def _acquire(
        self,
        ring: deque[tuple[torch.Tensor, torch.cuda.Event | None]],
        dtype: torch.dtype,
        numel: int,
    ) -> torch.Tensor:
        """Find a free buffer of ``ring`` holding ``numel`` elements of ``dtype``.

        "Free" means the event after its last copy completed. Busy buffers stay in the
        ring; a fresh allocation is cheaper than a host sync on a copy that has likely
        finished. Wrong-dtype or too-small buffers are retired.
        """
        for _ in range(len(ring)):
            buffer, event = ring.popleft()
            if event is not None and not event.query():
                ring.append((buffer, event))  # still in flight
                continue
            if buffer.dtype == dtype and buffer.numel() >= numel:
                return buffer
            # Completed but unusable for this request: drop it.
        return torch.empty(numel, dtype=dtype, pin_memory=True)

    def pending(self) -> int:
        """How many copies are in flight right now, either direction (test hook)."""
        return sum(
            1
            for ring in (self._staging, self._spill, self._in_use.values())
            for _, event in ring
            if event is not None and not event.query()
        )


@dataclass
class RegionRecord:
    """One resolved timeline region: a named span of work on one stream.

    ``start_ms``/``end_ms`` are on the shared device clock, so copy and compute
    regions compare directly (overlap is ``a.start < b.end and b.start < a.end``).
    """

    name: str
    stream: str
    start_ms: float
    end_ms: float

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Timeline:
    """CUDA-event recorder for overlap evidence; costs one flag check when off.

    Events, not host timestamps, because "did the copy and compute regions overlap
    on the device?" is a device-side fact (host clocks only say when work launched).
    All events share one clock via an epoch event recorded when the first region
    opens, so cross-stream comparison is valid.

    Args:
        enabled: From ``RAPID_LLM_OVERLAP_TIMELINE``; ``False`` makes :meth:`region`
            a near-free context manager.
        device: Device the events are recorded against.
    """

    enabled: bool = False
    device: str = "cuda"
    _epoch: torch.cuda.Event | None = field(default=None, repr=False)
    _pending: list[tuple[str, str, torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list, repr=False
    )

    @classmethod
    def from_env(cls, device: str = "cuda") -> Timeline:
        raw = os.environ.get(TIMELINE_ENV, "0").strip().lower()
        return cls(enabled=raw in ("1", "true", "on"), device=device)

    @contextmanager
    def region(self, name: str, stream: str = "compute") -> Iterator[None]:
        """Record a named region on the current stream; near-free when disabled."""
        if not self.enabled:
            yield
            return
        if self._epoch is None:
            self._epoch = torch.cuda.Event(enable_timing=True)
            self._epoch.record(torch.cuda.current_stream(self.device))
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream(self.device))
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream(self.device))
            self._pending.append((name, stream, start, end))

    def collect(self) -> list[RegionRecord]:
        """Synchronise and resolve every recorded region onto the shared clock."""
        if self._epoch is None:
            return []
        self._epoch.synchronize()
        records = []
        for name, stream, start, end in self._pending:
            end.synchronize()
            records.append(
                RegionRecord(
                    name,
                    stream,
                    self._epoch.elapsed_time(start),
                    self._epoch.elapsed_time(end),
                )
            )
        return records

    def summary(self) -> str:
        """Human-readable region table, for benchmark logs and test output."""
        return "\n".join(
            f"{r.stream:>8}  {r.name:<24} [{r.start_ms:9.3f}, {r.end_ms:9.3f}] ms"
            for r in self.collect()
        )
