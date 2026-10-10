"""Block-table arithmetic of :meth:`SlotBatch.write_block_tables` — CPU, no model.

A stub runner and KV manager stand in for the device: the assertions are about
which physical rows a slot's table entries name after a write. That is the whole
mechanism behind prefix reuse — two slots whose entries name the same rows read
the same K/V, and no K/V is moved.

Usage:
    pytest tests/executor/test_slot_batch_blocks.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.executor.slot_batch import SlotBatch

LAYERS = 2
HEADS = 2
HEAD_DIM = 3
#: Table rows the stub runner offers. The last is the filler slot, so three are
#: addressable by requests.
SLOTS = 4


class _StubKVCacheManager:
    """Just the per-layer K/V buffer the batch zeroes its scratch block in."""

    def __init__(self, rows: int) -> None:
        self.gpu_kv_buffer = [torch.full((rows, HEADS, HEAD_DIM), 7.0) for _ in range(LAYERS)]
        self.gpu_num_blocks = rows


class _StubRunner:
    """Minimal stand-in for :class:`~rapid_llm.executor.model_runner.ModelRunner`."""

    def __init__(self, num_slots: int, max_seq_len: int, rows: int | None = None) -> None:
        self.device = torch.device("cpu")
        self.max_seq_len = max_seq_len
        self.b_req_tokens_table = torch.empty(num_slots, max_seq_len, dtype=torch.int32)
        # Deliberately not zero: the batch must actively point unmapped entries
        # at the null block rather than inheriting whatever was there.
        self.b_req_tokens_table.fill_(-1)
        self.atten_info = type("_Atten", (), {"b_req_tokens_table": self.b_req_tokens_table})()
        self.kv_cache_manager = _StubKVCacheManager(rows if rows else num_slots * max_seq_len)


def _batch(num_slots: int = SLOTS, max_seq_len: int = 64) -> SlotBatch:
    return SlotBatch(_StubRunner(num_slots, max_seq_len))


def _table(batch: SlotBatch) -> torch.Tensor:
    return batch._atten.b_req_tokens_table


def _rows_of(block_id: int, block_size: int) -> list[int]:
    """The physical rows block *block_id* owns."""
    return list(range(block_id * block_size, (block_id + 1) * block_size))


# --------------------------------------------------------------------------- #
# 1. The initial table
# --------------------------------------------------------------------------- #
class TestInitialState:
    """Nothing is mapped until the scheduler says so."""

    def test_unmapped_entries_point_at_the_null_block(self):
        """Block 0 is reserved, so an unmapped entry names rows nobody reads.

        The alternative — leaving the entry at whatever it was — would have a
        stray row of a live sequence read as if it were position 0 of somebody
        else's prompt.
        """
        batch = _batch()
        table = _table(batch)
        for slot in range(batch.num_slots):
            assert torch.equal(table[slot], torch.zeros_like(table[slot]))

    def test_the_last_slot_is_the_filler_and_is_not_handed_out(self):
        batch = _batch(num_slots=SLOTS)
        assert batch.num_slots == SLOTS - 1
        assert batch._filler_slot == SLOTS - 1

    def test_the_filler_row_tiles_the_null_block(self):
        """Filler rows cost one block, not ``max_seq_len`` rows.

        Padding a decode batch up to a captured graph size means its filler rows
        may be asked to attend at any position, so the whole row is tiled with
        the scratch block's rows — every position lands inside it.
        """
        batch = _batch()
        row = _table(batch)[batch._filler_slot]
        expected = torch.arange(row.numel(), dtype=row.dtype) % batch.block_size
        assert torch.equal(row, expected)
        assert int(row.max()) < batch.block_size

    def test_the_scratch_block_is_zeroed(self):
        """Uninitialised fp16 can hold NaN, and a cache of NaNs lies to a debugger."""
        batch = _batch()
        buffer = batch._runner.kv_cache_manager.gpu_kv_buffer[0]
        assert torch.count_nonzero(buffer[: batch.block_size]) == 0
        assert torch.count_nonzero(buffer[batch.block_size :]) > 0  # nothing else touched

    def test_a_single_slot_runner_switches_padding_off(self):
        batch = _batch(num_slots=1)
        assert batch._filler_slot is None
        assert batch.num_slots == 1


# --------------------------------------------------------------------------- #
# 2. Writing block tables
# --------------------------------------------------------------------------- #
class TestWriteBlockTables:
    """One write per block, expanded into that block's rows."""

    def test_a_block_expands_into_its_own_physical_rows(self):
        batch = _batch()
        size = batch.block_size
        batch.write_block_tables([(1, 0, 0, (5,))])

        row = _table(batch)[1]
        assert row[:size].tolist() == _rows_of(5, size)
        assert torch.count_nonzero(row[size:]) == 0  # nothing beyond it moved

    def test_several_blocks_land_consecutively(self):
        batch = _batch()
        size = batch.block_size
        batch.write_block_tables([(0, 0, 0, (9, 3, 7))])

        row = _table(batch)[0]
        expected = _rows_of(9, size) + _rows_of(3, size) + _rows_of(7, size)
        assert row[: 3 * size].tolist() == expected

    def test_start_block_offsets_the_write(self):
        """A grown sequence maps its new block without rewriting the old ones."""
        batch = _batch()
        size = batch.block_size
        batch.write_block_tables([(0, 0, 0, (4,))])
        batch.write_block_tables([(0, 0, 1, (6,))])

        row = _table(batch)[0]
        assert row[:size].tolist() == _rows_of(4, size)
        assert row[size : 2 * size].tolist() == _rows_of(6, size)

    def test_two_slots_sharing_a_block_name_identical_rows(self):
        """The whole point: reuse is the same rows, not a copy of them.

        Where the fixed-slot layout had to move a prefix's K/V into the new
        occupant's rows, this is two table rows agreeing — so a 2 000-token
        shared prefix costs a few hundred int32 writes and no K/V traffic.
        """
        batch = _batch()
        size = batch.block_size
        shared = (11, 12, 13)
        batch.write_block_tables([(0, 0, 0, (*shared, 20))])
        batch.write_block_tables([(1, 0, 0, (*shared, 21))])

        table = _table(batch)
        span = len(shared) * size
        assert torch.equal(table[0, :span], table[1, :span])
        # Each keeps its own tail: writing one into the other's rows would
        # corrupt a sequence that is still reading them.
        assert not torch.equal(table[0, span : span + size], table[1, span : span + size])

    def test_writes_in_one_call_are_independent(self):
        batch = _batch()
        size = batch.block_size
        batch.write_block_tables([(0, 0, 0, (2,)), (1, 0, 1, (8,)), (2, 0, 0, (2,))])

        table = _table(batch)
        assert table[0, :size].tolist() == _rows_of(2, size)
        assert table[1, :size].tolist() == [0] * size  # slot 1 mapped only block 1
        assert table[1, size : 2 * size].tolist() == _rows_of(8, size)
        assert table[2, :size].tolist() == _rows_of(2, size)  # aliases slot 0, legally

    def test_no_writes_touches_nothing(self):
        """The steady-state decode step, which must cost no kernel launch."""
        batch = _batch()
        before = _table(batch).clone()
        batch.write_block_tables(())
        assert torch.equal(_table(batch), before)

    def test_an_empty_block_list_is_skipped(self):
        batch = _batch()
        before = _table(batch).clone()
        batch.write_block_tables([(0, 0, 0, ())])
        assert torch.equal(_table(batch), before)

    def test_a_write_is_idempotent(self):
        batch = _batch()
        batch.write_block_tables([(0, 0, 0, (3, 4))])
        once = _table(batch).clone()
        batch.write_block_tables([(0, 0, 0, (3, 4))])
        assert torch.equal(_table(batch), once)


# --------------------------------------------------------------------------- #
# 3. Boundaries
# --------------------------------------------------------------------------- #
class TestBoundaries:
    """The context limit, and groups the device does not have a table for."""

    def test_a_block_overhanging_the_context_limit_is_truncated(self):
        """A sequence at the limit has a whole last block whose tail it never reaches."""
        batch = _batch(max_seq_len=20)  # 1 block plus 4 columns
        size = batch.block_size
        batch.write_block_tables([(0, 0, 1, (5,))])

        row = _table(batch)[0]
        assert row[size:].tolist() == _rows_of(5, size)[: 20 - size]

    def test_a_write_starting_past_the_limit_is_dropped(self):
        batch = _batch(max_seq_len=20)
        before = _table(batch).clone()
        batch.write_block_tables([(0, 0, 2, (5,))])  # column 32, past 20
        assert torch.equal(_table(batch), before)

    def test_a_second_kv_group_is_refused_loudly(self):
        """Group 0 is all the executor wires today, and a silent drop would
        leave a group's rows pointing at the null block — wrong logits, no error."""
        batch = _batch()
        with pytest.raises(NotImplementedError, match="group 1"):
            batch.write_block_tables([(0, 1, 0, (3,))])

    def test_the_table_dtype_is_preserved(self):
        """The kernels index with int32; a widened table would read garbage."""
        batch = _batch()
        batch.write_block_tables([(0, 0, 0, (3,))])
        assert _table(batch).dtype is torch.int32


# --------------------------------------------------------------------------- #
# 4. Decode context parallelism
# --------------------------------------------------------------------------- #
def _dcp_batch(monkeypatch, rank: int, dcp: int) -> SlotBatch:
    from rapid_llm.distributed import parallel_state as ps

    monkeypatch.setattr(ps, "_DCP_RANK", rank)
    monkeypatch.setattr(ps, "_DCP_WORLD_SIZE", dcp)
    return _batch()


class TestDecodeContextParallel:
    """Global positions in the table, this rank's packed rows behind them."""

    @pytest.mark.parametrize("dcp", [2, 4])
    def test_every_position_is_owned_once_and_the_strided_view_is_packed(self, monkeypatch, dcp):
        """Position ``p`` belongs to rank ``p % dcp``: its entry names a row of
        the block's ``block_size / dcp`` share there, and row 0 (discard)
        everywhere else. The attention read ``table[slot, rank::dcp]`` must then
        be the share's rows in order -- a gap or a repeat there is a position
        attended twice or never, with nothing downstream to notice.
        """
        blocks = (9, 3)
        owners = []
        for rank in range(dcp):
            batch = _dcp_batch(monkeypatch, rank, dcp)
            per_rank = batch.rows_per_block
            assert per_rank == batch.block_size // dcp
            batch.write_block_tables([(0, 0, 0, blocks)])

            row = _table(batch)[0, : len(blocks) * batch.block_size]
            owners.append(row != 0)
            expected = [b * per_rank + i for b in blocks for i in range(per_rank)]
            assert row[rank::dcp].tolist() == expected
        owned = torch.stack(owners).int()
        assert torch.equal(owned.sum(dim=0), torch.ones_like(owned[0]))

    def test_the_filler_row_and_scratch_stay_inside_the_null_block_share(self, monkeypatch):
        batch = _dcp_batch(monkeypatch, 1, 2)
        row = _table(batch)[batch._filler_slot]
        assert int(row.max()) < batch.rows_per_block
        buffer = batch._runner.kv_cache_manager.gpu_kv_buffer[0]
        assert torch.count_nonzero(buffer[: batch.rows_per_block]) == 0
        assert torch.count_nonzero(buffer[batch.rows_per_block :]) > 0

    def test_a_block_size_dcp_does_not_divide_is_refused(self, monkeypatch):
        with pytest.raises(ValueError):
            _dcp_batch(monkeypatch, 0, 3)
