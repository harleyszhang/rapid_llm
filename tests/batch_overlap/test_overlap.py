"""L1 cross-stream overlap: the policy, the stream pool, the timeline.

Stub KV manager and runner drive the row arithmetic on CPU; the env
spellings (``1``/``0``, ``on``/``off``) and the extend-row planning are
checked without a GPU.

Usage:
    pytest tests/batch_overlap/test_overlap.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rapid_llm.batch_overlap.overlap import (
    OVERLAP_ENV,
    OverlapPolicy,
    StreamPool,
    Timeline,
)
from rapid_llm.executor.slot_batch import SlotBatch, flatten_extend_rows


# --------------------------------------------------------------------------- #
# Policy parsing (CPU)
# --------------------------------------------------------------------------- #
def test_overlap_is_on_unless_explicitly_disabled(monkeypatch):
    monkeypatch.delenv(OVERLAP_ENV, raising=False)
    assert OverlapPolicy.from_env().enabled
    for raw in ("1", "true", "ON", "yes"):
        monkeypatch.setenv(OVERLAP_ENV, raw)
        assert OverlapPolicy.from_env().enabled


def test_overlap_honours_the_off_spellings(monkeypatch):
    for raw in ("0", "false", "off", "OFF"):
        monkeypatch.setenv(OVERLAP_ENV, raw)
        assert not OverlapPolicy.from_env().enabled


# --------------------------------------------------------------------------- #
# Host helpers shared by the inline and prepared paths (CPU)
# --------------------------------------------------------------------------- #
def test_flatten_extend_rows_expands_one_row_per_token():
    rows_slot, rows_len = flatten_extend_rows([2, 5], [3, 10], [4, 2])
    assert rows_slot == [2, 2, 2, 2, 5, 5]
    # Cache length once the row's own K/V lands: absolute position plus one.
    assert rows_len == [4, 5, 6, 7, 11, 12]


def test_flatten_extend_rows_rejects_ragged_inputs():
    with pytest.raises(ValueError):
        flatten_extend_rows([0], [0, 1], [1])


class _StubKVCacheManager:
    def __init__(self, rows: int) -> None:
        self.gpu_kv_buffer = [torch.zeros(rows, 1, 1)]
        self.claimed: int | None = None

    def claim(self, rows: int) -> None:
        self.claimed = rows


class _StubRunner:
    """What :class:`SlotBatch` touches, plus a graph width to pad towards."""

    def __init__(self, num_slots: int, max_seq_len: int, graph_width: int = 0) -> None:
        self.device = torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.b_req_tokens_table = torch.zeros(num_slots, max_seq_len, dtype=torch.int32)
        # begin_* read the table and overwrite everything else on the metadata.
        self.atten_info = SimpleNamespace(b_req_tokens_table=self.b_req_tokens_table)
        self.kv_cache_manager = _StubKVCacheManager(num_slots * max_seq_len)
        self._graph_width = graph_width

    def graph_batch_size(self, batch_size: int) -> int:
        return max(batch_size, self._graph_width) if self._graph_width else batch_size


def _batch(num_slots: int = 4, max_seq_len: int = 16, graph_width: int = 0) -> SlotBatch:
    return SlotBatch(_StubRunner(num_slots, max_seq_len, graph_width))


def test_plan_extend_rows_matches_begin_extends_metadata():
    """The prepared upload and the metadata setter must describe the same rows."""
    batch = _batch()
    slots, starts, ends = [0, 1], [3, 0], [6, 5]
    planned = batch.plan_extend_rows(slots, starts, ends)
    batch.begin_extend(slots, starts, ends)
    assert planned[0] == batch._b_req_idx.tolist()
    assert planned[1] == batch._b_seq_len.tolist()


def test_plan_extend_rows_pads_onto_the_graph_width_with_the_filler_slot():
    batch = _batch(graph_width=8)
    rows_slot, rows_len = batch.plan_extend_rows([0], [0], [3])
    # Three real rows plus five fillers pointing at the reserved last slot.
    assert rows_slot == [0, 0, 0] + [3] * 5
    assert rows_len[:3] == [1, 2, 3]
    assert rows_len[3:] == [3] * 5  # filler length tracks the longest real row


def test_pad_decode_rows_grows_to_the_graph_width():
    batch = _batch(graph_width=4)
    slots, lens = batch.pad_decode_rows([0, 2], [7, 9])
    assert slots == [0, 2, 3, 3]
    assert lens == [7, 9, 9, 9]


def test_pad_decode_rows_with_a_single_slot_never_pads():
    batch = _batch(num_slots=1, graph_width=4)
    assert batch.pad_decode_rows([0], [7]) == ([0], [7])


# --------------------------------------------------------------------------- #
# Timeline, disabled (CPU)
# --------------------------------------------------------------------------- #
def test_a_disabled_timeline_records_nothing():
    timeline = Timeline(enabled=False)
    with timeline.region("anything"):
        pass
    assert timeline.collect() == []
    assert timeline.summary() == ""


# --------------------------------------------------------------------------- #
# StreamPool (GPU)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_upload_lands_the_values_and_the_shape():
    pool = StreamPool("cuda", OverlapPolicy(enabled=True))
    tensor, event = pool.upload_async([[1, 2], [3, 4]], dtype=torch.long)
    assert tensor.shape == (2, 2)
    pool.consume(event, tensor)
    assert tensor.tolist() == [[1, 2], [3, 4]]


@pytest.mark.gpu
def test_uploads_in_flight_never_overwrite_each_other():
    """The staging ring must grow rather than recycle a buffer a copy is reading.

    Eight uploads are issued back to back with no consume in between; if a busy
    staging buffer were force-reused, some device tensor would read a later
    upload's bytes.
    """
    pool = StreamPool("cuda", OverlapPolicy(enabled=True))
    in_flight = [pool.upload_async([i] * 64, dtype=torch.long) for i in range(8)]
    for i, (tensor, event) in enumerate(in_flight):
        pool.consume(event, tensor)
        assert tensor.tolist() == [i] * 64


@pytest.mark.gpu
def test_a_disabled_pool_falls_back_to_a_blocking_upload():
    pool = StreamPool("cuda", OverlapPolicy(enabled=False))
    tensor, event = pool.upload_async([1, 2, 3], dtype=torch.long)
    assert event is None
    pool.consume(None)  # no-op, and no error
    assert tensor.tolist() == [1, 2, 3]


@pytest.mark.gpu
def test_the_timeline_sees_the_copy_region():
    timeline = Timeline(enabled=True, device="cuda")
    pool = StreamPool("cuda", OverlapPolicy(enabled=True), timeline)
    tensor, event = pool.upload_async([1, 2, 3], dtype=torch.long, label="upload.test")
    pool.consume(event, tensor)
    torch.cuda.synchronize()

    records = timeline.collect()
    assert [r.name for r in records] == ["upload.test"]
    assert records[0].stream == "copy"
    assert records[0].duration_ms >= 0.0


@pytest.mark.gpu
def test_readback_lands_the_values_and_the_shape():
    pool = StreamPool("cuda", OverlapPolicy(enabled=True))
    device = torch.arange(6, device="cuda", dtype=torch.long).view(2, 3)
    host, event = pool.readback_async(device, label="readback.test")
    assert host.shape == (2, 3)
    event.synchronize()
    assert host.tolist() == [[0, 1, 2], [3, 4, 5]]


@pytest.mark.gpu
def test_readbacks_in_flight_never_overwrite_each_other():
    """The spill ring must grow rather than recycle a buffer a copy needs.

    A long-running kernel queue sits ahead of every readback's copy (the copy
    is ordered after the current compute stream), so each buffer is provably
    still busy — its event incomplete — when the next readback acquires. A
    forced reuse would overwrite bytes the caller still holds.
    """
    pool = StreamPool("cuda", OverlapPolicy(enabled=True))
    # Warm up every first-call cost first (the _sleep module load, the pool's
    # pinned allocation): the stall below must still be running when the later
    # readbacks acquire, or a finished event lets the ring legally recycle.
    torch.cuda._sleep(1)
    _warm, warm_event = pool.readback_async(torch.zeros(8, device="cuda", dtype=torch.long))
    warm_event.synchronize()
    torch.cuda.synchronize()

    # A spin kernel the readbacks' copies queue behind: it outlives the host
    # issuing all four readbacks, so every buffer is still busy when the next
    # one acquires.
    torch.cuda._sleep(20_000_000)
    views = [
        pool.readback_async(torch.full((1024,), i, device="cuda", dtype=torch.long))
        for i in range(4)
    ]
    torch.cuda.synchronize()
    for i, (host, event) in enumerate(views):
        event.synchronize()
        assert host.tolist() == [i] * 1024


@pytest.mark.gpu
def test_a_readback_buffer_is_not_recycled_until_released():
    """The copy event alone must not free a buffer its holder is still reading.

    The launch/harvest loop issues the *next* pass's copy before it reads the
    previous pass's tokens, so a buffer freed on its event would already carry
    the wrong step's values by the time the harvest looked at it. Holding
    handed-out buffers out of the ring is what makes the split safe.
    """
    pool = StreamPool("cuda", OverlapPolicy(enabled=True))
    first, first_event = pool.readback_async(torch.full((4,), 1, device="cuda", dtype=torch.long))
    first_event.synchronize()

    # The copy has landed, so an event-only ring would call this buffer free.
    second, second_event = pool.readback_async(torch.full((4,), 2, device="cuda", dtype=torch.long))
    second_event.synchronize()
    assert first.tolist() == [1] * 4, "the next readback overwrote a buffer still held"
    assert second.tolist() == [2] * 4

    # Releasing hands it back, and the next readback reuses that same storage --
    # so the ring recycles instead of growing one buffer per step forever.
    pool.release_readback(first)
    third, third_event = pool.readback_async(torch.full((4,), 3, device="cuda", dtype=torch.long))
    third_event.synchronize()
    assert third.data_ptr() == first.data_ptr(), "a released buffer was not recycled"
    assert third.tolist() == [3] * 4


@pytest.mark.gpu
def test_releasing_an_unstaged_view_is_a_no_op():
    """Overlap off hands back a plain ``.cpu()`` tensor, which owns its own storage."""
    pool = StreamPool("cuda", OverlapPolicy(enabled=False))
    host, event = pool.readback_async(torch.arange(3, device="cuda", dtype=torch.long))
    assert event is None
    pool.release_readback(host)  # must not raise
    pool.release_readback(torch.arange(3))  # nor a view it never handed out


@pytest.mark.gpu
def test_a_disabled_pool_falls_back_to_a_blocking_readback():
    pool = StreamPool("cuda", OverlapPolicy(enabled=False))
    device = torch.tensor([7, 8, 9], device="cuda", dtype=torch.long)
    host, event = pool.readback_async(device)
    assert event is None
    assert host.tolist() == [7, 8, 9]
