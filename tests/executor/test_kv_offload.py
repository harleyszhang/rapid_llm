"""Tests for the offload data path — real CUDA streams, small tensors.

Case list mirrors vLLM's ``tests/v1/simple_kv_offload/test_worker.py``:
correct rows move in both directions, a store is ordered after the compute
that wrote the block, and the completion event reports landing.

Runs on a CUDA device; auto-skipped otherwise (the ``gpu`` marker).

Usage:
    pytest tests/executor/test_kv_offload.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.engine.kv_offload.base import CopyRun
from rapid_llm.executor.kv_offload import KVCopyEngine, alloc_cpu_kv_buffers

pytestmark = pytest.mark.gpu

BLOCK_SIZE = 4
NUM_LAYERS = 2
NUM_GPU_BLOCKS = 4
NUM_CPU_BLOCKS = 8


@pytest.fixture
def copier() -> KVCopyEngine:
    """A two-layer engine over small float32 buffers."""
    kv_row = (2, 4)
    gpu = [
        torch.zeros((NUM_GPU_BLOCKS * BLOCK_SIZE, *kv_row), dtype=torch.float32, device="cuda")
        for _ in range(NUM_LAYERS)
    ]
    cpu = alloc_cpu_kv_buffers(NUM_CPU_BLOCKS, BLOCK_SIZE, kv_row, NUM_LAYERS, torch.float32)
    return KVCopyEngine(gpu, cpu, BLOCK_SIZE)


def test_store_then_load_roundtrip(copier: KVCopyEngine):
    """Fill a GPU block, store it, overwrite, load it back."""
    for layer in copier._gpu:
        layer[0:BLOCK_SIZE].fill_(1.0)

    copier.submit([CopyRun(gpu_block=0, cpu_block=1)], is_store=True).synchronize()
    for layer in copier._cpu:
        assert torch.all(layer[BLOCK_SIZE : 2 * BLOCK_SIZE] == 1.0)

    for layer in copier._cpu:
        layer[BLOCK_SIZE : 2 * BLOCK_SIZE].fill_(2.0)
    for layer in copier._gpu:
        layer[0:BLOCK_SIZE].fill_(-1.0)

    copier.submit([CopyRun(gpu_block=0, cpu_block=1)], is_store=False).synchronize()
    for layer in copier._gpu:
        assert torch.all(layer[0:BLOCK_SIZE] == 2.0)
    # The CPU copy is a copy: it survives the load.
    for layer in copier._cpu:
        assert torch.all(layer[BLOCK_SIZE : 2 * BLOCK_SIZE] == 2.0)


def test_moves_whole_blocks_and_nothing_else(copier: KVCopyEngine):
    """A move touches its block's rows, and only those."""
    for layer in copier._gpu:
        layer.fill_(-1.0)
        layer[BLOCK_SIZE : 2 * BLOCK_SIZE].fill_(7.0)  # gpu block 1
    for layer in copier._cpu:
        layer.zero_()  # the host buffer starts uninitialized (torch.empty)

    copier.submit([CopyRun(gpu_block=1, cpu_block=2)], is_store=True).synchronize()
    for layer in copier._cpu:
        assert torch.all(layer[2 * BLOCK_SIZE : 3 * BLOCK_SIZE] == 7.0)
        untouched = torch.cat([layer[: 2 * BLOCK_SIZE], layer[3 * BLOCK_SIZE :]])
        assert torch.all(untouched == 0.0)  # the rest of the host buffer is untouched


def test_store_orders_after_compute_write(copier: KVCopyEngine):
    """A store must see work already queued on the caller's stream.

    The slow kernel holds the compute stream busy, so a store that failed to
    wait on it would read the pre-kernel zeros and land them in host memory.
    """
    block = copier._gpu[0]
    torch.cuda._sleep(50_000_000)  # ~30ms of stream time; keeps the race honest
    block[0:BLOCK_SIZE].fill_(3.0)

    copier.submit([CopyRun(gpu_block=0, cpu_block=1)], is_store=True).synchronize()
    assert torch.all(copier._cpu[0][BLOCK_SIZE : 2 * BLOCK_SIZE] == 3.0)


def test_batched_moves_share_one_event(copier: KVCopyEngine):
    """One submit, several moves, one event that fires when the last lands."""
    for layer in copier._gpu:
        layer[0:BLOCK_SIZE].fill_(1.0)
        layer[BLOCK_SIZE : 2 * BLOCK_SIZE].fill_(2.0)

    event = copier.submit(
        [CopyRun(gpu_block=0, cpu_block=1), CopyRun(gpu_block=1, cpu_block=2)],
        is_store=True,
    )
    event.synchronize()
    assert event.query()
    for layer in copier._cpu:
        assert torch.all(layer[BLOCK_SIZE : 2 * BLOCK_SIZE] == 1.0)
        assert torch.all(layer[2 * BLOCK_SIZE : 3 * BLOCK_SIZE] == 2.0)


def test_layer_count_mismatch_rejected():
    """A buffer pair describing different models is a wiring bug, not a crash."""
    gpu = [torch.zeros((4, 2), device="cuda"), torch.zeros((4, 2), device="cuda")]
    cpu = [torch.zeros((4, 2), pin_memory=True)]
    with pytest.raises(ValueError, match="layer count mismatch"):
        KVCopyEngine(gpu, cpu, BLOCK_SIZE)
