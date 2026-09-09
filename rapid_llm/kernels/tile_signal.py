"""L4 tile-signaling: a producer/consumer pair of persistent Triton kernels.

Lets a dependent kernel start consuming a GEMM's output tiles before the GEMM
finishes: the producer publishes each tile with a release-semantics flag
write, the consumer acquires the flag (a bounded spin) before reading. One
intra-kernel pipeline: ``x @ W_gate, x @ W_up -> silu(gate) * up`` — the MLP
epilogue streams over tiles the GEMM is still producing, on two CUDA streams
of the same GPU. Eager runs use a monotonic epoch, so stale flags never look
ready; CUDA Graph capture records one flag clear because replay cannot advance
a Python epoch.

Deadlock freedom is a sizing property, not a hope: both kernels are persistent
(fixed grid, work pulled through an atomic counter) and the launch splits the
SM budget so producer + consumer blocks always fit resident at once. The
producer depends on nothing, so it always drains; the consumer's spin is
bounded (``MAX_SPIN``) and a dropped tile is counted in a device counter the
host reads — a loud failure, never silently-wrong data.

Usage:
    from rapid_llm.kernels.tile_signal import TileSignalBuffer, pipelined_gemm_swiglu
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .ops.activation.activations import silu

#: Default spin bound: ~1e6 acquire polls before a consumer block gives up.
#: Generous on purpose — the producer cannot stall (no dependencies), so any
#: spin near this bound already means a bug, and the count surfaces it.
DEFAULT_MAX_SPIN = 1 << 20

_producer_streams: dict[int, torch.cuda.Stream] = {}


def _producer_stream(device: torch.device) -> torch.cuda.Stream:
    """Reuse one tile producer stream per device."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _producer_streams.get(index)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _producer_streams[index] = stream
    return stream


@triton.jit
def _tile_signal_gemm_kernel(
    a_ptr,
    gate_w_ptr,
    up_w_ptr,
    gate_ptr,
    up_ptr,
    flags_ptr,
    work_ptr,
    epoch,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wn,
    stride_uwk,
    stride_uwn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent producer: pull a tile, compute its gate and up GEMM, publish.

    One work item is one output tile (``BLOCK_M x BLOCK_N`` of the activation
    row space): the same A rows are dotted against both weight matrices, so
    the pair shares one load of ``a``. Publication is
    ``tl.atomic_xchg(flag, epoch, sem="release")`` — every store of the tile
    happens before the flag, and the consumer's acquire pairs with it.
    """
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    tile = tl.atomic_add(work_ptr, 1)
    while tile < num_tiles:
        pid_m = tile % num_pid_m
        pid_n = tile // num_pid_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        gw_ptrs = gate_w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        uw_ptrs = up_w_ptr + offs_k[:, None] * stride_uwk + offs_n[None, :] * stride_uwn

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < k_rem), other=0.0)
            gate_w = tl.load(
                gw_ptrs, mask=(offs_k[:, None] < k_rem) & n_mask[None, :], other=0.0
            )
            up_w = tl.load(
                uw_ptrs, mask=(offs_k[:, None] < k_rem) & n_mask[None, :], other=0.0
            )
            acc_gate = tl.dot(a, gate_w, acc_gate)
            acc_up = tl.dot(a, up_w, acc_up)
            a_ptrs += BLOCK_K * stride_ak
            gw_ptrs += BLOCK_K * stride_wk
            uw_ptrs += BLOCK_K * stride_uwk

        g_ptrs = gate_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(
            g_ptrs,
            acc_gate.to(g_ptrs.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )
        u_ptrs = up_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(
            u_ptrs,
            acc_up.to(u_ptrs.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )

        # Publish only after both stores: release orders them before the
        # consumer's matching acquire read.
        tl.atomic_xchg(flags_ptr + tile, epoch, sem="release")
        tile = tl.atomic_add(work_ptr, 1)


@triton.jit
def _tile_signal_consume_kernel(
    gate_ptr,
    up_ptr,
    out_ptr,
    flags_ptr,
    work_ptr,
    fail_ptr,
    epoch,
    M,
    N,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_SPIN: tl.constexpr,
):
    """Persistent consumer: acquire a tile's flag, then run its epilogue.

    The spin reads the flag with ``sem="acquire"`` so the producer's tile
    stores are visible before the loads below. ``MAX_SPIN`` bounds the wait —
    the sizing rule (grids sum to at most the SM count) makes reaching it
    impossible in a correct launch, and a tile that gives up is counted in
    ``fail_ptr`` instead of writing garbage.
    """
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n

    tile = tl.atomic_add(work_ptr, 1)
    while tile < num_tiles:
        ready = tl.atomic_add(flags_ptr + tile, 0, sem="acquire")
        spins = 0
        while (ready < epoch) & (spins < MAX_SPIN):
            ready = tl.atomic_add(flags_ptr + tile, 0, sem="acquire")
            spins += 1

        if ready >= epoch:
            pid_m = tile % num_pid_m
            pid_n = tile // num_pid_m
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            offs = offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn

            gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(up_ptr + offs, mask=mask, other=0.0)
            out = silu(gate) * up
            out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
            tl.store(out_ptrs, out.to(out_ptrs.dtype.element_ty), mask=mask)
        else:
            # Bounded wait exhausted: the producer cannot be alive (correct
            # sizing rules this out), so count and move on — never write.
            tl.atomic_add(fail_ptr, 1)
        tile = tl.atomic_add(work_ptr, 1)


class TileSignalBuffer:
    """Device-side state shared by a tile-signaling producer/consumer pair.

    Holds the tile flags (one ``int32`` per tile, monotonically raised to the
    current epoch by the producer), the two work counters, and the consumer's
    dropped-tile counter. Entry points reset the counters on their launch
    streams; graph capture also records a flag clear for replay safety.

    The buffer is sized for one problem shape; reusing it for another shape
    with fewer tiles is fine (extra flags stay stale, below every future
    epoch), but a larger shape needs a new buffer.
    """

    def __init__(self, num_tiles: int, device: str | torch.device = "cuda") -> None:
        self.num_tiles = int(num_tiles)
        self.device = torch.device(device)
        self.flags = torch.zeros(self.num_tiles, dtype=torch.int32, device=self.device)
        self.producer_work = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.consumer_work = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.fail_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        self._epoch = 0

    @classmethod
    def for_problem(
        cls,
        m: int,
        n: int,
        block_m: int,
        block_n: int,
        device: str | torch.device = "cuda",
    ) -> TileSignalBuffer:
        """Size a buffer for ``[m, n]`` GEMM output tiled by ``block_m x block_n``."""
        num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
        return cls(num_tiles, device)

    def advance_epoch(self) -> int:
        """Enter the next run; returns the epoch its flags will carry.

        Epochs are monotonic ``int32``. After ~2^31 runs the counter would wrap
        onto stale flags, so it resets — the buffer must not have a run in
        flight when that happens (concurrent runs over one buffer are not a
        supported shape anyway).
        """
        if self._epoch >= torch.iinfo(torch.int32).max - 1:
            self.flags.zero_()
            self._epoch = 0
        self._epoch += 1
        return self._epoch

    @property
    def epoch(self) -> int:
        """Epoch of the most recent run started through :meth:`advance_epoch`."""
        return self._epoch

    def dropped_tiles(self) -> int:
        """How many tiles the consumer gave up on; ``item()`` synchronises.

        Call after the run's kernels have completed (any sync point makes the
        read exact); a non-zero result means the launch's SM sizing or the
        producer itself was broken.
        """
        return int(self.fail_count.item())


def default_block_split(
    num_tiles: int,
    sm_count: int,
    producer_blocks: int | None = None,
    consumer_blocks: int | None = None,
) -> tuple[int, int]:
    """Grid sizes whose sum never exceeds the resident-SM budget.

    The default gives the producer two thirds of the SMs (the GEMM is the
    expensive side) and clamps both grids to the tile count — running more
    blocks than tiles only adds idle programs. Explicit values are honoured as
    long as they still fit the budget; the entry points use this to keep the
    producer/consumer pair co-resident, which is what makes the consumer's
    bounded wait sound.
    """
    if sm_count < 2:
        sm_count = 2
    default_producer = max(1, sm_count * 2 // 3)
    producer = default_producer if producer_blocks is None else max(1, producer_blocks)
    consumer = max(1, sm_count - producer) if consumer_blocks is None else max(1, consumer_blocks)
    if producer + consumer > sm_count:
        raise ValueError(
            f"grid sum {producer}+{consumer} exceeds the SM budget {sm_count}; "
            "the tile-signal pair must stay co-resident"
        )
    producer = min(producer, max(1, num_tiles))
    consumer = min(consumer, max(1, num_tiles))
    return producer, consumer


def _launch_common(
    a: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    buffer: TileSignalBuffer,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    max_spin: int,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Validate shapes, allocate intermediates, return launch parameters."""
    if a.ndim != 2 or gate_w.ndim != 2 or up_w.ndim != 2:
        raise ValueError("activations and weights must be 2-D tensors")
    if not a.is_cuda or gate_w.device != a.device or up_w.device != a.device:
        raise ValueError("activations, weights, and signal buffer must share one CUDA device")
    if buffer.flags.device != a.device:
        raise ValueError("activations, weights, and signal buffer must share one CUDA device")
    if gate_w.dtype != a.dtype or up_w.dtype != a.dtype:
        raise ValueError("activations and weights must have the same dtype")
    if any(value < 1 or value & (value - 1) for value in (block_m, block_n, block_k)):
        raise ValueError("block sizes must be positive powers of two")
    if num_warps < 1 or num_stages < 1 or max_spin < 1:
        raise ValueError("num_warps, num_stages, and max_spin must be positive")
    m, k = a.shape
    n = gate_w.shape[1]
    if up_w.shape != gate_w.shape:
        raise ValueError(f"gate/up weight shapes differ: {tuple(gate_w.shape)} vs {tuple(up_w.shape)}")
    if gate_w.shape[0] != k:
        raise ValueError(f"weight K {gate_w.shape[0]} does not match activation K {k}")
    num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    if num_tiles > buffer.num_tiles:
        raise ValueError(
            f"problem needs {num_tiles} tiles but the buffer holds {buffer.num_tiles}"
        )
    gate = torch.empty((m, n), dtype=a.dtype, device=a.device)
    up = torch.empty_like(gate)
    out = torch.empty_like(gate)
    sm_count = torch.cuda.get_device_properties(a.device).multi_processor_count
    producer, consumer = default_block_split(num_tiles, sm_count)
    return num_tiles, sm_count, gate, up, out, producer, consumer, n


def pipelined_gemm_swiglu(
    a: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    buffer: TileSignalBuffer,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    max_spin: int = DEFAULT_MAX_SPIN,
    producer_blocks: int | None = None,
    consumer_blocks: int | None = None,
    timeline=None,
) -> torch.Tensor:
    """Overlapped ``silu(a @ gate_w) * (a @ up_w)`` on two CUDA streams.

    The producer runs on a private stream, the consumer on the caller's
    current stream; tiles flow between them through the buffer's release /
    acquire flags, so the epilogue of early tiles overlaps the GEMM of later
    ones. Only the epilogue waits — the GEMM side never blocks.

    Args:
        a: ``[M, K]`` activations.
        gate_w: ``[K, N]`` gate projection weights (already transposed from
            ``nn.Linear``'s ``[N, K]``).
        up_w: ``[K, N]`` up projection weights, same layout as ``gate_w``.
        buffer: Signalling state for this problem's tile count.
        timeline: Optional :class:`~rapid_llm.batch_overlap.overlap.Timeline`; the
            producer and consumer regions are recorded on their own streams,
            giving direct overlap evidence.

    Returns:
        ``[M, N]`` product, in the inputs' dtype.
    """
    num_tiles, sm_count, gate, up, out, producer, consumer, n = _launch_common(
        a, gate_w, up_w, buffer, block_m, block_n, block_k, num_warps, num_stages, max_spin
    )
    if producer_blocks is not None or consumer_blocks is not None:
        producer, consumer = default_block_split(
            num_tiles, sm_count, producer_blocks, consumer_blocks
        )
    m, k = a.shape
    epoch = buffer.advance_epoch()

    producer_stream = _producer_stream(a.device)
    current = torch.cuda.current_stream(a.device)
    buffer.fail_count.zero_()
    if torch.cuda.is_current_stream_capturing():
        # Replay does not execute Python's epoch increment. Capturing this clear
        # prevents a previous replay's flags from satisfying the fixed epoch.
        buffer.flags.zero_()

    # Producer side: counter reset and kernel both on the producer stream;
    # the stream first waits for the caller's prior work on `a` and weights.
    with torch.cuda.stream(producer_stream):
        producer_stream.wait_stream(current)
        buffer.producer_work.zero_()
        if timeline is not None:
            with timeline.region("l4.gemm", "producer"):
                _tile_signal_gemm_kernel[(producer,)](
                    a, gate_w, up_w, gate, up,
                    buffer.flags, buffer.producer_work, epoch,
                    m, n, k,
                    a.stride(0), a.stride(1),
                    gate_w.stride(0), gate_w.stride(1),
                    up_w.stride(0), up_w.stride(1),
                    gate.stride(0), gate.stride(1),
                    BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                    num_warps=num_warps, num_stages=num_stages,
                )
        else:
            _tile_signal_gemm_kernel[(producer,)](
                a, gate_w, up_w, gate, up,
                buffer.flags, buffer.producer_work, epoch,
                m, n, k,
                a.stride(0), a.stride(1),
                gate_w.stride(0), gate_w.stride(1),
                up_w.stride(0), up_w.stride(1),
                gate.stride(0), gate.stride(1),
                BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
                num_warps=num_warps, num_stages=num_stages,
            )
    # The producer stream wrote tensors allocated on the caller's stream.
    a.record_stream(producer_stream)
    gate_w.record_stream(producer_stream)
    up_w.record_stream(producer_stream)
    gate.record_stream(producer_stream)
    up.record_stream(producer_stream)

    # Consumer side: counter reset and kernel on the caller's stream, which
    # only ever waits through the tile flags (never a stream-wide barrier).
    buffer.consumer_work.zero_()
    if timeline is not None:
        with timeline.region("l4.epilogue", "consumer"):
            _tile_signal_consume_kernel[(consumer,)](
                gate, up, out,
                buffer.flags, buffer.consumer_work, buffer.fail_count, epoch,
                m, n,
                gate.stride(0), gate.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=block_m, BLOCK_N=block_n, MAX_SPIN=max_spin,
                num_warps=num_warps, num_stages=num_stages,
            )
    else:
        _tile_signal_consume_kernel[(consumer,)](
            gate, up, out,
            buffer.flags, buffer.consumer_work, buffer.fail_count, epoch,
            m, n,
            gate.stride(0), gate.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, MAX_SPIN=max_spin,
            num_warps=num_warps, num_stages=num_stages,
        )

    # The caller's stream rejoins the producer's for the epilogue's inputs.
    current.wait_stream(producer_stream)
    return out


def serial_gemm_swiglu(
    a: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    buffer: TileSignalBuffer,
    *,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 2,
    max_spin: int = DEFAULT_MAX_SPIN,
) -> torch.Tensor:
    """Same kernels, same grids, one stream, in order — the A/B control arm.

    The producer drains every tile (flags included) before the consumer
    starts, so the acquire polls return immediately and the only difference
    versus :func:`pipelined_gemm_swiglu` is the execution strategy — which is
    exactly what the L4 benchmark needs to isolate.
    """
    _num_tiles, _sm_count, gate, up, out, producer, consumer, n = _launch_common(
        a, gate_w, up_w, buffer, block_m, block_n, block_k, num_warps, num_stages, max_spin
    )
    m, k = a.shape
    epoch = buffer.advance_epoch()

    buffer.fail_count.zero_()
    buffer.producer_work.zero_()
    _tile_signal_gemm_kernel[(producer,)](
        a, gate_w, up_w, gate, up,
        buffer.flags, buffer.producer_work, epoch,
        m, n, k,
        a.stride(0), a.stride(1),
        gate_w.stride(0), gate_w.stride(1),
        up_w.stride(0), up_w.stride(1),
        gate.stride(0), gate.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )

    buffer.consumer_work.zero_()
    _tile_signal_consume_kernel[(consumer,)](
        gate, up, out,
        buffer.flags, buffer.consumer_work, buffer.fail_count, epoch,
        m, n,
        gate.stride(0), gate.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=block_m, BLOCK_N=block_n, MAX_SPIN=max_spin,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
