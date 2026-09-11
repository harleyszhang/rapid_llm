"""The offload data path: block copies between the device cache and host RAM.

:class:`KVCopyEngine` is :class:`~rapid_llm.engine.kv_offload.base.BlockCopier`
on real hardware — two CUDA streams (one per direction), pinned host buffers
mirroring the GPU cache layer by layer, and one event per submitted batch so
the scheduler side can poll completion without blocking.

Stream discipline, which is the whole correctness argument:

* a **store** reads live GPU blocks, so its stream first waits on the caller's
  stream — the copy sees everything already written, i.e. the block's K/V;
* the *engine* keeps the other half of that contract: a GPU block whose store
  is still in flight is about to be overwritten by the next request, and
  compute must wait on :meth:`CPUPrimaryTierOffloadingManager.pending_store_events`
  before writing it;
* a **load** writes a freshly allocated GPU block that nobody is reading, so
  it needs no wait going in — the engine's guarantee that the request is not
  scheduled until the load's event fires provides the wait going out.

One batch, one stream, one event: moves handed to one :meth:`submit` call run
in order on that stream, and the single event fires when the last has landed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..engine.kv_offload.base import CopyRun


def alloc_cpu_kv_buffers(
    num_blocks: int,
    block_size: int,
    kv_row: tuple[int, int],
    num_layers: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Pinned host mirrors of the GPU cache, one tensor per layer.

    Shape is ``[num_blocks * block_size, *kv_row]`` — the same row layout as
    the device buffer, only longer, so a block copy is one contiguous slice on
    both sides. Pinning matters for two reasons: the copies stay asynchronous
    (pageable memory would serialize them through a staging buffer), and the
    NIC can DMA straight out of host RAM when a remote tier lands.
    """
    rows = num_blocks * block_size
    return [
        torch.empty((rows, *kv_row), dtype=dtype, pin_memory=True) for _ in range(num_layers)
    ]


class KVCopyEngine:
    """Issues block moves over dedicated streams; returns an event per batch.

    Args:
        gpu_buffers: The device cache, one tensor per layer, rows indexed
            ``block_id * block_size + offset``.
        cpu_buffers: The host mirrors built by :func:`alloc_cpu_kv_buffers`,
            same layout, ``num_cpu_blocks * block_size`` rows.
        block_size: Tokens per block; the row count of one move on each side.
    """

    def __init__(
        self,
        gpu_buffers: Sequence[torch.Tensor],
        cpu_buffers: Sequence[torch.Tensor],
        block_size: int,
    ) -> None:
        if len(gpu_buffers) != len(cpu_buffers):
            raise ValueError(
                f"layer count mismatch: {len(gpu_buffers)} GPU vs {len(cpu_buffers)} CPU"
            )
        self._gpu = list(gpu_buffers)
        self._cpu = list(cpu_buffers)
        self._block_size = block_size
        self._store_stream = torch.cuda.Stream()
        self._load_stream = torch.cuda.Stream()

    def submit(self, moves: Sequence[CopyRun], *, is_store: bool) -> torch.cuda.Event:
        """Run *moves* on the store or load stream; return their completion event.

        The event is recorded on the same stream as the copies, so it fires
        exactly when the last layered slice of the last move has landed —
        there is no host-side synchronization anywhere in this method.
        """
        stream = self._store_stream if is_store else self._load_stream
        block_size = self._block_size
        # A store must not outrun compute: wait on the caller's stream so every
        # already-issued kernel that wrote these blocks is visible to the copy.
        if is_store:
            stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for move in moves:
                gpu_start = move.gpu_block * block_size
                cpu_start = move.cpu_block * block_size
                for gpu_layer, cpu_layer in zip(self._gpu, self._cpu, strict=True):
                    if is_store:
                        cpu_layer[cpu_start : cpu_start + block_size].copy_(
                            gpu_layer[gpu_start : gpu_start + block_size], non_blocking=True
                        )
                    else:
                        gpu_layer[gpu_start : gpu_start + block_size].copy_(
                            cpu_layer[cpu_start : cpu_start + block_size], non_blocking=True
                        )
        event = torch.cuda.Event()
        event.record(stream)
        return event
