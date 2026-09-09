"""Compute-communication overlap: the C-axis plumbing shared by L2 and L3.

:mod:`rapid_llm.batch_overlap.overlap` owns the host↔device axis; this module
owns the compute↔communication one. Both policies stand on one primitive — an
all-reduce on a dedicated comm stream, fenced with events:

* **L3** — :class:`CommOverlapPolicy` chunks a row-parallel GEMM's tokens, so
  chunk ``k+1``'s GEMM computes while chunk ``k``'s all-reduce is on the wire.
* **L2 (TBO)** — :func:`deferred_all_reduce` switches every
  :class:`~rapid_llm.modules.linear.RowParallelLinear` in the context to
  deferred mode: the all-reduce posts async, the caller fences where the data
  is consumed — what lets two micro-batches ping-pong.

Usage:
    with deferred_all_reduce():          # TBO: RowParallelLinear forwards defer
        ...                              # exit fences outstanding events
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.distributed as dist

from ..distributed.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from ..distributed.sequence_parallel import is_g_bar_boundary, row_parallel_g_bar
from ..tools.observability import Collective, CollectiveStats
from ..utils.env_compat import env_flag, getenv
from ..utils.logger import get_logger
from .overlap import Timeline

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..modules.linear import LinearBase

_log = get_logger(__name__)

#: Environment variable switching the L3 chunked all-reduce on (``0`` disables).
COMM_OVERLAP_ENV = "RAPID_LLM_COMM_OVERLAP"

#: Token count a row-parallel GEMM must reach before L3 splits it at all.
L3_MIN_ROWS_ENV = "RAPID_LLM_L3_MIN_ROWS"

#: How many row chunks L3 splits into, at most.
L3_CHUNKS_ENV = "RAPID_LLM_L3_CHUNKS"

#: Floor on rows per chunk: below this an extra all-reduce buys latency rather
#: than overlap — PCIe small-message reductions sit at a fixed cost, and
#: splitting thinner than that pays it more times for the same bytes.
L3_MIN_CHUNK_ROWS = 256


@dataclass(frozen=True)
class CommOverlapPolicy:
    """The L3 switch: chunked all-reduce inside one row-parallel GEMM.

    Off by default — the blocking all-reduce is the semantics every test was
    written against, and a policy that changes numerics order without an opt-in
    would fail them quietly. ``min_rows`` keeps small GEMMs (a decode step of
    four requests) on the blocking path where one collective beats two.

    Args:
        enabled: Whether eligible GEMMs split their all-reduce.
        min_rows: Token count a GEMM must reach before it is eligible.
        chunks: Maximum chunk count; :meth:`chunk_count` may shrink it.
    """

    enabled: bool = False
    min_rows: int = 512
    chunks: int = 2

    @classmethod
    def from_env(cls) -> CommOverlapPolicy:
        """Read ``RAPID_LLM_COMM_OVERLAP``; anything but ``0``/``false``/``off`` means on."""
        return cls(
            enabled=env_flag(COMM_OVERLAP_ENV),
            min_rows=int(getenv(L3_MIN_ROWS_ENV, "512")),
            chunks=max(2, int(getenv(L3_CHUNKS_ENV, "2"))),
        )

    def chunk_count(self, rows: int) -> int:
        """Chunks a ``rows``-token GEMM actually splits into.

        Capped twice: by ``chunks``, and by the floor of
        :data:`L3_MIN_CHUNK_ROWS` rows per chunk — a 300-row GEMM asked for
        four chunks gets one instead, because four ~75-row all-reduces cost
        more than the single reduction they replace.
        """
        if not self.enabled or rows <= 0:
            return 1
        by_floor = max(1, rows // L3_MIN_CHUNK_ROWS)
        return max(1, min(self.chunks, by_floor))


_policy_cache: CommOverlapPolicy | None = None


def comm_overlap_policy() -> CommOverlapPolicy:
    """The L3 policy, read once per process.

    An environment lookup per GEMM would be measurable on the decode path
    (hundreds of string parses per step); the process is the natural lifetime
    because benchmark arms run as separate processes.
    """
    global _policy_cache
    if _policy_cache is None:
        _policy_cache = CommOverlapPolicy.from_env()
    return _policy_cache


def reset_comm_overlap_policy() -> None:
    """Forget the cached policy — test hook after monkeypatching the env."""
    global _policy_cache
    _policy_cache = None


def _chunk_bounds(rows: int, count: int) -> list[tuple[int, int]]:
    """Split ``[0, rows)`` into ``count`` near-equal ``[start, stop)`` spans.

    The remainder spreads one row at a time over the leading chunks, so the
    bounds are deterministic for any ``(rows, count)`` and cover every row
    exactly once.
    """
    base, extra = divmod(rows, count)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(count):
        stop = start + base + (1 if index < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


class CommStreamPool:
    """The stream NCCL all-reduces are issued on — one per device.

    NCCL collectives are stream-ordered: issued inside a
    ``torch.cuda.stream(comm)`` context they execute on that stream, which is
    what lets an all-reduce run *beside* the compute stream's kernels instead
    of *between* them. The pool owns that stream plus the two pieces of
    discipline every async reduce on it needs:

    * **Ordering**: ``comm.wait_stream(compute)`` before the reduce, so it
      observes the kernels that produced the tensor (the caller issues the
      reduce right after the GEMM, so the compute stream's tail *is* that
      GEMM).
    * **Lifetime**: ``tensor.record_stream(comm)`` after it, so the caching
      allocator does not hand the partial's block to a new allocation while
      the reduction is still reading and writing it.

    Args:
        device: Device the stream belongs to; a bare ``"cuda"`` resolves to
            the current device so the pool key matches ``x.device`` spellings.
        timeline: Where reduce regions are recorded; ``None`` creates a
            timeline from the environment (disabled unless
            ``RAPID_LLM_OVERLAP_TIMELINE`` is set), the same switch the L1
            copy regions use.
    """

    _by_device: ClassVar[dict[str, CommStreamPool]] = {}

    def __init__(self, device: str | torch.device, timeline: Timeline | None = None) -> None:
        self._device = torch.device(device)
        self._timeline = timeline
        self._stream: torch.cuda.Stream | None = None

    @classmethod
    def for_device(
        cls, device: str | torch.device = "cuda", *, timeline: Timeline | None = None
    ) -> CommStreamPool:
        """The shared pool for ``device``, created on first use."""
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None and torch.cuda.is_available():
            resolved = torch.device("cuda", torch.cuda.current_device())
        key = str(resolved)
        pool = cls._by_device.get(key)
        if pool is None:
            pool = cls(resolved, timeline if timeline is not None else Timeline.from_env(key))
            cls._by_device[key] = pool
            _log.debug("comm stream pool created for %s", key)
        return pool

    @classmethod
    def reset(cls) -> None:
        """Drop the per-device singletons — test hook (streams are recreated lazily)."""
        cls._by_device.clear()

    @property
    def device(self) -> torch.device:
        """The device this pool's stream and timeline events belong to."""
        return self._device

    @property
    def timeline(self) -> Timeline:
        """Where the pool's reduce regions are recorded."""
        if self._timeline is None:
            self._timeline = Timeline.from_env(str(self._device))
        return self._timeline

    @property
    def stream(self) -> torch.cuda.Stream:
        """The communication stream, created on first use."""
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self._device)
        return self._stream

    def all_reduce_async(
        self,
        tensor: torch.Tensor,
        *,
        group: dist.ProcessGroup | None = None,
        label: str = "all_reduce",
    ) -> torch.cuda.Event | None:
        """Post ``tensor``'s all-reduce on the comm stream; return its event.

        The call returns as soon as the reduction is *enqueued* — the tensor's
        value is final only after the returned event is fenced (waited on by
        the consuming stream, or synchronised by the host). ``None`` means no
        fence is needed, and covers three cases: a group with no peer (the
        reduction is a no-op), a non-NCCL backend (``dist.all_reduce`` on gloo
        blocks until the tensor is final), and a CPU pool (no stream exists).

        Args:
            tensor: The partial sum to reduce, in place across the group.
            group: ``None`` means the module-state TP group.
            label: Timeline region name the reduce is recorded under.

        Returns:
            The reduction's completion event on the comm stream, or ``None``
            when the tensor is already final.
        """
        group = get_tensor_model_parallel_group() if group is None else group
        if group is None or dist.get_world_size(group) <= 1:
            return None
        payload = tensor.numel() * tensor.element_size()
        if dist.get_backend(group) != "nccl":
            dist.all_reduce(tensor, group=group)
            CollectiveStats.record(Collective.ALL_REDUCE, payload)
            return None
        if torch.cuda.is_current_stream_capturing():
            # Keep the fork/join shape during capture: PyTorch records kernels
            # from every stream while a capture is open, so the comm stream's
            # wait becomes a graph branch and the consumer's wait on the event
            # the join edge. Posting on the capture stream itself would flatten
            # those edges into strict issue order -- the overlap a TBO step
            # exists to create. The fork is mandatory (an unordered side stream
            # would race the capture stream) and the event is real: a capture
            # cannot lean on the host to synchronise, only on graph edges.
            capture = torch.cuda.current_stream(self._device)
            comm = self.stream
            comm.wait_stream(capture)
            with torch.cuda.stream(comm):
                dist.all_reduce(tensor, group=group)
            event = torch.cuda.Event()
            event.record(comm)
            tensor.record_stream(comm)
            CollectiveStats.record(Collective.ALL_REDUCE, payload)
            return event
        compute = torch.cuda.current_stream(self._device)
        comm = self.stream
        comm.wait_stream(compute)
        with torch.cuda.stream(comm), self.timeline.region(label, "comm"):
            dist.all_reduce(tensor, group=group)
        event = torch.cuda.Event()
        event.record(comm)
        tensor.record_stream(comm)
        CollectiveStats.record(Collective.ALL_REDUCE, payload)
        return event

    def all_to_all_async(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        *,
        group: dist.ProcessGroup | None = None,
        label: str = "all_to_all",
    ) -> torch.cuda.Event | None:
        """Post an equal-split all-to-all on the comm stream; return its event.

        The EP dispatch/combine exchange rides the same discipline as
        :meth:`all_reduce_async`: ordered after the compute stream's queue at
        issue (the permuted send buffer is its tail), executed beside it, and
        fenced by the consumer through the returned event. ``None`` again
        means no fence is needed — a group with no peer (the exchange is an
        identity copy) or a non-NCCL backend (the collective blocks until
        ``output`` is final).

        Args:
            output: Receive buffer, same shape as ``input`` (equal split).
            input: Send buffer; slice ``j`` along dim 0 lands on rank ``j``.
            group: ``None`` means the module-state TP group (the EP group is
                the TP group, see :func:`~rapid_llm.distributed.parallel_state.get_ep_group`).
            label: Timeline region name the exchange is recorded under.

        Returns:
            The exchange's completion event on the comm stream, or ``None``
            when ``output`` is already final.
        """
        group = get_tensor_model_parallel_group() if group is None else group
        if group is None or dist.get_world_size(group) <= 1:
            output.copy_(input)
            return None
        payload = input.numel() * input.element_size()
        if dist.get_backend(group) != "nccl":
            dist.all_to_all_single(output, input, group=group)
            CollectiveStats.record(Collective.ALL_TO_ALL, payload)
            return None
        if torch.cuda.is_current_stream_capturing():
            # Same capture rule as :meth:`all_reduce_async`: keep the
            # fork/join edges the eager path builds. Flattening the exchange
            # onto the capture stream would replay it between — not beside —
            # the compute kernels, and a TBO or SBO graph would lose exactly
            # the overlap it was captured for.
            capture = torch.cuda.current_stream(self._device)
            comm = self.stream
            comm.wait_stream(capture)
            with torch.cuda.stream(comm):
                dist.all_to_all_single(output, input, group=group)
            event = torch.cuda.Event()
            event.record(comm)
            # Both buffers belong to the exchange while it is in flight —
            # under capture too: the caller may drop either reference the
            # moment this returns, and a capture-time free would let the
            # graph's pool reissue the block at the address the exchange is
            # still reading. Recording defers that reuse past the branch.
            input.record_stream(comm)
            output.record_stream(comm)
            CollectiveStats.record(Collective.ALL_TO_ALL, payload)
            return event
        compute = torch.cuda.current_stream(self._device)
        comm = self.stream
        comm.wait_stream(compute)
        with torch.cuda.stream(comm), self.timeline.region(label, "comm"):
            dist.all_to_all_single(output, input, group=group)
        event = torch.cuda.Event()
        event.record(comm)
        # Both buffers belong to the exchange while it is in flight: the comm
        # stream reads the send buffer and writes the receive buffer, and the
        # caller may drop either reference the moment this call returns.
        input.record_stream(comm)
        output.record_stream(comm)
        CollectiveStats.record(Collective.ALL_TO_ALL, payload)
        return event


class DeferredArContext:
    """TBO's communication mode: reduces are posted async, fenced by consumers.

    Inside :func:`deferred_all_reduce`, every
    :class:`~rapid_llm.modules.linear.RowParallelLinear` all-reduce becomes a
    :meth:`defer`: the reduction is enqueued on the comm stream and its
    completion event is handed to whoever asked for it. The tensor the layer
    returns is a *promise* — its value is final only after the caller fences
    (the two-batch executor waits exactly at the point the next stage reads
    the tensor, which is what leaves room for the other micro-batch's compute
    to overlap the reduction in between).

    Args:
        pool: The comm stream reductions are posted on.
    """

    def __init__(self, pool: CommStreamPool) -> None:
        self._pool = pool
        self._events: list[torch.cuda.Event] = []
        self._collector: list[torch.cuda.Event] | None = None

    @property
    def pool(self) -> CommStreamPool:
        """The comm stream this context posts on."""
        return self._pool

    @contextmanager
    def collecting(self, events: list[torch.cuda.Event]) -> Iterator[None]:
        """Route the completes of reduces deferred inside this block to ``events``.

        The TBO executor opens one collector per micro-batch stage, so the
        stage that consumes a tensor fences on *its own* events — never on the
        other micro-batch's, which would serialise the ping-pong it exists to
        create. Collectors nest; the innermost wins.
        """
        previous, self._collector = self._collector, events
        try:
            yield
        finally:
            self._collector = previous

    def defer(
        self, tensor: torch.Tensor, *, group: dist.ProcessGroup | None = None
    ) -> torch.Tensor:
        """Post ``tensor``'s all-reduce async and return the tensor unfenced.

        The value is final once :meth:`fence` has been called on the event(s)
        this deferral produced. A world of one (or a non-NCCL backend) leaves
        the tensor untouched and deferral is a no-op — the same values, just
        no event to wait for.
        """
        event = self._pool.all_reduce_async(tensor, group=group)
        if event is None:
            return tensor
        self._events.append(event)
        if self._collector is not None:
            self._collector.append(event)
        return tensor

    def fence(self, events: list[torch.cuda.Event]) -> None:
        """Make the current compute stream wait for ``events`` — the consume point.

        Fencing is stream-ordered, not host-blocking: kernels launched after
        this call on the current stream simply cannot start before the
        reductions complete. The list is emptied, so a stage's collector can
        be reused for the next layer.
        """
        if not events:
            return
        pending = tuple(events)
        stream = torch.cuda.current_stream(self._pool.device)
        for event in pending:
            stream.wait_event(event)
        events.clear()
        if events is not self._events:
            completed = {id(event) for event in pending}
            self._events[:] = [event for event in self._events if id(event) not in completed]

    def fence_pending_reads(self) -> None:
        """Fence what the innermost collector gathered — an in-stage read point.

        A stage that reads its own deferred output in place — the MoE block
        sums the shared expert's reduction into the routed output inside the
        very stage that deferred it — fences here: the promise a defer
        returns is only final after the fence. With no collector open (a
        bare deferred context), every outstanding event waits instead, which
        keeps the read correct without TBO's per-stage granularity.
        """
        self.fence(self._collector if self._collector is not None else self._events)

    def drain(self) -> None:
        """Fence events not already consumed by a stage collector."""
        self.fence(self._events)


_deferred: ContextVar[DeferredArContext | None] = ContextVar(
    "rapid_llm_deferred_all_reduce", default=None
)


def current_deferred_ar() -> DeferredArContext | None:
    """The deferred-AR context this call site runs in, if any."""
    return _deferred.get()


@contextmanager
def deferred_all_reduce(
    device: str | torch.device = "cuda",
) -> Iterator[DeferredArContext]:
    """Switch row-parallel all-reduces inside this block to deferred mode.

    The context owns the discipline: every deferral's event is fenced before
    the block exits, so no deferred value can escape unfenced even if the
    consumer inside forgot. The contextvar (not a module global) carries it,
    so the DP replicas' worker threads each hold their own.

    Args:
        device: Device whose comm stream pool reductions are posted on.
    """
    ctx = DeferredArContext(CommStreamPool.for_device(device))
    token = _deferred.set(ctx)
    try:
        yield ctx
    finally:
        ctx.drain()
        _deferred.reset(token)


def _dispatch_mode(world_size: int, deferred: bool, policy: CommOverlapPolicy, rows: int) -> str:
    """Pure dispatch decision for one row-parallel forward.

    Order matters and is the priority claim: a deferred context (TBO) beats
    the L3 policy — both target the same all-reduce, and splitting a reduction
    the ping-pong is already overlapping with other compute would chop the
    comm stream into messages too small to hide anything.

    Returns:
        One of ``"passthrough"``, ``"deferred"``, ``"chunked"``, ``"blocking"``.
    """
    if world_size <= 1:
        return "passthrough"
    if deferred:
        return "deferred"
    if policy.enabled and rows >= policy.min_rows:
        return "chunked"
    return "blocking"


def row_parallel_forward(layer: LinearBase, x: torch.Tensor) -> torch.Tensor:
    """One row-parallel GEMM plus its all-reduce, dispatched by the active mode.

    Args:
        layer: The row-parallel layer; only :meth:`apply_linear
            <rapid_llm.modules.linear.LinearBase.apply_linear>` is called, plus
            its ``reduce_results`` flag.
        x: ``[..., input_size]`` activations — the leading dims are tokens.
    """
    world_size = get_tensor_model_parallel_world_size()
    rows = x.shape[:-1].numel()
    # A layer built with ``reduce_results=False`` promised its caller the raw
    # partial sum (it reduces elsewhere, or not at all); honour that here, since
    # this dispatcher is the only place the flag can still be read.
    if not getattr(layer, "reduce_results", True):
        return layer.apply_linear(x)
    # Inside a sequence-parallel region a marked layer is the *g-bar* boundary:
    # its partial is reduce-scattered to this rank's token shard, which replaces
    # the all-reduce outright instead of rescheduling it. Ahead of the overlap
    # modes because there is then nothing left for them to defer or chunk — and
    # the region declined to open next to either of them in the first place.
    if is_g_bar_boundary(layer):
        return row_parallel_g_bar(layer, x)
    if x.device.type == "cpu" and _deferred.get() is None:
        return tensor_model_parallel_all_reduce(layer.apply_linear(x))
    mode = _dispatch_mode(world_size, _deferred.get() is not None, comm_overlap_policy(), rows)
    if mode == "passthrough":
        return layer.apply_linear(x)
    if mode == "deferred":
        return _deferred.get().defer(layer.apply_linear(x))
    if mode == "chunked":
        return _chunked_row_parallel(layer, x, rows)
    return tensor_model_parallel_all_reduce(layer.apply_linear(x))


def _chunked_row_parallel(layer: LinearBase, x: torch.Tensor, rows: int) -> torch.Tensor:
    """L3: split the GEMM's rows, overlap each chunk's reduce with the next GEMM.

    The token grid flattens before splitting (a ``[batch, seq, hidden]``
    activation becomes ``[batch * seq, hidden]``), so chunk boundaries are in
    tokens and the output is viewed back to the input's leading dims. Row
    independence is what makes this legal: no GEMM output element reads
    another row of the input, so the sum a chunk's all-reduce computes is the
    sum the unsplit GEMM would have produced for those rows.
    """
    policy = comm_overlap_policy()
    count = policy.chunk_count(rows)
    if count <= 1:
        return tensor_model_parallel_all_reduce(layer.apply_linear(x))
    pool = CommStreamPool.for_device(x.device)
    flat = x.reshape(rows, x.shape[-1])
    partials: list[torch.Tensor] = []
    events: list[torch.cuda.Event] = []
    for index, (start, stop) in enumerate(_chunk_bounds(rows, count)):
        with pool.timeline.region(f"l3.gemm.{index}", "compute"):
            partial = layer.apply_linear(flat[start:stop])
        event = pool.all_reduce_async(partial, label=f"l3.all_reduce.{index}")
        partials.append(partial)
        if event is not None:
            events.append(event)
    if events:
        compute = torch.cuda.current_stream(x.device)
        for event in events:
            compute.wait_event(event)
    out = torch.cat(partials, dim=0)
    return out.view(*x.shape[:-1], out.shape[-1])
