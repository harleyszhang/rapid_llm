"""Slot-based KV layout and per-step attention metadata for continuous batching.

:class:`SlotBatch` gives every active sequence one block table row ("slot")
and builds the flat row indices each phase needs: prefill rows, extend rows
via :func:`flatten_extend_rows`, or padded decode rows for graph replay.

Usage:
    batch = SlotBatch(runner); batch.begin_decode(slots, seq_lens)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ..engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from .model_runner import ModelRunner


def flatten_extend_rows(
    slots: Sequence[int],
    starts: Sequence[int],
    chunk_lens: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Expand ``(slot, start, chunk_len)`` triples into one row per token, on the host.

    Host twin of :meth:`SlotBatch._flatten_rows` (plain lists, so a pass is laid
    out before any device work). Row ``r`` of request ``i`` has
    ``rows_slot[r] == slots[i]`` and ``rows_len[r]`` = its absolute position + 1.

    Raises:
        ValueError: The three sequences do not describe the same chunks.
    """
    if not len(slots) == len(starts) == len(chunk_lens):
        raise ValueError("slots, starts and chunk_lens must describe the same chunks")
    rows_slot: list[int] = []
    rows_len: list[int] = []
    for slot, start, length in zip(slots, starts, chunk_lens, strict=True):
        rows_slot += [slot] * length
        rows_len += list(range(start + 1, start + length + 1))
    return rows_slot, rows_len


class SlotBatch:
    """Continuous-batching view of the KV cache: paged blocks, stable metadata.

    Two decisions keep a steady decode step free of host-device traffic:

    - **Block table per slot.** Slot ``s`` owns row ``s`` of ``b_req_tokens_table``;
      entry ``p`` is the cache row token ``p`` lives in. The mapping need not be
      contiguous, so sequences sharing a prefix point at the same physical rows —
      reuse costs a refcount, no K/V movement. Concurrency is bounded by the block
      pool, not the slot count.
    - **Composition-stable metadata.** ``b_req_idx``/``b_seq_len`` are rebuilt from
      the host only when a request joins or leaves; a steady step just increments
      lengths on-device, so decode metadata costs two kernels and no transfers.

    Args:
        runner: Executor whose cache, block table and metadata this drives.
    """

    def __init__(self, runner: ModelRunner) -> None:
        self._runner = runner
        self._atten = runner.atten_info
        self.device = runner.device
        self.max_seq_len = runner.max_seq_len
        self.block_size = PREFIX_CACHE_BLOCK_SIZE

        table = runner.b_req_tokens_table
        total_slots, row_len = table.shape

        # The last slot backs filler rows that pad a decode batch to a captured
        # graph size; with one slot there is nothing to spare, so padding is off.
        self._filler_slot: int | None = total_slots - 1 if total_slots > 1 else None
        self.num_slots = total_slots - 1 if total_slots > 1 else 1

        # Entries start at block 0, the reserved null block the allocator never
        # hands out, so an unmapped entry names rows no live sequence reads.
        table.zero_()
        if self._filler_slot is not None:
            # Filler rows attend over the null block, tiled so any padded position
            # lands inside it; zero it so debug sessions do not read NaN.
            table[self._filler_slot] = (
                torch.arange(row_len, dtype=table.dtype, device=self.device) % self.block_size
            )
            for layer in runner.kv_cache_manager.gpu_kv_buffer:
                layer[: self.block_size].zero_()

        # Per-sequence offsets in a flattened prefill grid, scaled per call.
        self._row_offsets = torch.arange(total_slots, dtype=torch.int32, device=self.device)

        # Device metadata plus the host mirror used to decide whether the next
        # step can reuse it.
        self._b_req_idx: torch.Tensor | None = None
        self._b_seq_len: torch.Tensor | None = None
        self._host_slots: list[int] = []
        self._host_lens: list[int] = []

    # ------------------------------------------------------------------ steps #
    def write_block_tables(self, writes: Sequence[tuple[int, int, int, tuple[int, ...]]]) -> None:
        """Point slots' table entries at the physical blocks they were given.

        The entire device-side cost of prefix reuse: a paged table only names the
        rows a prefix already lives in, so shared tokens cost a few int32 writes,
        no K/V movement. Entries past ``max_seq_len`` are dropped (the last block
        of a sequence at the limit names positions it never reaches).

        Args:
            writes: ``(slot, group_id, start_block, block_ids)`` per grant.

        Raises:
            NotImplementedError: A plan named a KV cache group other than 0
                (only homogeneous models are wired through today).
        """
        if not writes:
            return
        table = self._atten.b_req_tokens_table
        width = table.shape[1]
        size = self.block_size
        slots: list[int] = []
        positions: list[int] = []
        rows: list[int] = []
        for slot, group_id, start_block, block_ids in writes:
            if group_id != 0:
                raise NotImplementedError(
                    f"KV cache group {group_id} has no block table on the device yet"
                )
            start = start_block * size
            if not block_ids or start >= width:
                continue
            count = min(len(block_ids) * size, width - start)
            # Build one contiguous host batch for all grants in the plan.  The
            # old loop made a fresh H2D allocation and assignment per grant;
            # prefix-cache admissions can carry hundreds of grants at once.
            slots.extend([slot] * count)
            positions.extend(range(start, start + count))
            mapped_rows = [
                block_id * size + offset for block_id in block_ids for offset in range(size)
            ]
            rows.extend(mapped_rows[:count])
        if slots:
            table[
                torch.tensor(slots, dtype=torch.long, device=self.device),
                torch.tensor(positions, dtype=torch.long, device=self.device),
            ] = torch.tensor(rows, dtype=table.dtype, device=self.device)

    def begin_prefill(
        self,
        slots: Sequence[int],
        seq_starts: Sequence[int],
        seq_lens: Sequence[int],
    ) -> None:
        """Point the attention metadata at a padded prefill grid for ``slots``.

        The model flattens its ``[n, width]`` grid row-major, so sequence ``i``'s
        column ``j`` lands in slot ``slots[i]``'s cache row ``seq_starts[i] + j``
        (chunked prefill resumes mid-prompt, so rows start at their own position).
        ``seq_lens[i]`` is the total cached length after this chunk and bounds
        attention; padding positions write junk K/V but are never read. A pass
        with any ``seq_starts[i] > 0`` arms the chunked metadata
        (``b_prefix_len``/``b_kv_base``); a first-chunk pass clears it. The cache
        dtype does not enter into it: the chunked kernel dequantises fp8 rows.

        Args:
            slots: Slot id per sequence.
            seq_starts: First cache row each chunk writes.
            seq_lens: Total cached length per sequence once the chunk lands.

        Raises:
            ValueError: A chunk is wider than ``max_seq_len``.
        """
        # Grid width is the widest chunk (rows start at their own positions, so
        # the grid must match the widest chunk for cur_select_index to align).
        max_prompt_len = max(end - start for start, end in zip(seq_starts, seq_lens, strict=True))
        if max_prompt_len > self.max_seq_len:
            raise ValueError(
                f"prompt length {max_prompt_len} exceeds max_seq_len {self.max_seq_len}"
            )

        n = len(slots)
        b_req_idx = self._to_device(slots)
        starts = self._to_device(seq_starts)
        table = self._atten.b_req_tokens_table

        self._atten.b_req_idx = b_req_idx
        self._atten.b_seq_len = self._to_device(seq_lens)
        self._atten.max_actual_seq_len = max(seq_lens)
        self._atten.is_prefill = True
        # Column j of row i maps to cache row starts[i] + j (an outer sum).
        cols = starts.unsqueeze(1) + torch.arange(max_prompt_len, device=self.device).unsqueeze(0)
        self._atten.cur_select_index = table[b_req_idx.unsqueeze(1), cols].reshape(-1)
        self._atten.b_start_loc = self._row_offsets[:n] * max_prompt_len

        if any(start > 0 for start in seq_starts):
            # KV row 0 of each slot: the base its history hangs off.
            self._atten.b_prefix_len = starts
            self._atten.b_kv_base = table[b_req_idx, 0]
            self._atten.max_chunk_len = max_prompt_len
        else:
            self._clear_chunked_metadata()

        # A prefill changes the running set, so the next decode rebuilds metadata.
        self._host_slots, self._host_lens = [], []

    def begin_extend(
        self,
        slots: Sequence[int],
        seq_starts: Sequence[int],
        seq_lens: Sequence[int],
    ) -> int:
        """Point the attention metadata at one decode row per *token* of the chunks.

        The prefill kernel is pure self-attention over the current grid and cannot
        see K/V earlier chunks wrote, so a chunk resuming mid-prompt runs through
        the decode kernel: each (request, position) becomes one row attending over
        the slot's rows ``[0, position + 1)`` — causal extend at one row per token.
        Rows are one token wide, so padding keeps the pass on the decode graphs.

        Args:
            slots: Slot id per chunk.
            seq_starts: First cache row each chunk writes.
            seq_lens: Total cached length per sequence once the chunk lands.

        Returns:
            Row count submitted, filler included (caller discards trailing rows).
        """
        starts, ends = list(seq_starts), list(seq_lens)
        chunk_lens = [end - start for start, end in zip(starts, ends, strict=True)]
        if max(ends) > self.max_seq_len:
            raise ValueError(f"sequence length {max(ends)} exceeds max_seq_len {self.max_seq_len}")

        rows_slot, rows_len = self._flatten_rows(slots, starts, chunk_lens)

        # Filler rows pad onto a decode graph; their fake length tracks the
        # longest real cache so the bucket matches, junk K/V stays in the slot.
        filler_len = min(max(ends), self.max_seq_len)
        if self._filler_slot is not None:
            pad = self._runner.graph_batch_size(len(rows_slot)) - len(rows_slot)
            if pad > 0:
                rows_slot = torch.cat(
                    [
                        rows_slot,
                        torch.full(
                            (pad,), self._filler_slot, dtype=rows_slot.dtype, device=self.device
                        ),
                    ]
                )
                rows_len = torch.cat(
                    [
                        rows_len,
                        torch.full((pad,), filler_len, dtype=rows_len.dtype, device=self.device),
                    ]
                )

        table = self._atten.b_req_tokens_table
        self._b_req_idx = rows_slot
        self._b_seq_len = rows_len
        self._atten.b_req_idx = rows_slot
        self._atten.b_seq_len = rows_len
        self._atten.max_actual_seq_len = max(ends)
        self._atten.is_prefill = False
        # Host mirror of the row lens (filler included): the runner's per-step
        # prepare hook reads it instead of syncing the device lengths. The
        # host plan is the same flatten+pad the device rows just installed.
        self._atten.b_seq_len_cpu = torch.tensor(
            self.plan_extend_rows(slots, seq_starts, seq_lens)[1], dtype=torch.long
        )
        # Row `seq_len - 1` is where this row's fresh K/V lands.
        self._atten.cur_select_index = table[rows_slot, rows_len - 1]
        self._atten.b_start_loc = None
        self._clear_chunked_metadata()

        # A prefill changes the running set, so the next decode rebuilds metadata.
        self._host_slots, self._host_lens = [], []
        return len(rows_slot)

    def begin_decode(self, slots: Sequence[int], seq_lens: Sequence[int]) -> int:
        """Point the attention metadata at one decode step and report the batch size.

        Args:
            slots: Slot id per running request.
            seq_lens: Length each sequence has *after* this step's token (the new
                K/V is written to ``seq_lens[i] - 1``).

        Returns:
            Batch size submitted; exceeds ``len(slots)`` when padded to a captured
            graph size (caller discards trailing rows).
        """
        padded_slots, padded_lens = self.pad_decode_rows(slots, seq_lens)

        if padded_slots == self._host_slots and padded_lens == [
            length + 1 for length in self._host_lens
        ]:
            # Same requests, one token further: advance the device lengths in
            # place (steady state, nothing crosses the PCIe bus).
            self._b_seq_len += 1
        else:
            self._b_req_idx = self._to_device(padded_slots)
            self._b_seq_len = self._to_device(padded_lens)
        self._host_slots, self._host_lens = padded_slots, padded_lens

        table = self._atten.b_req_tokens_table
        self._atten.b_req_idx = self._b_req_idx
        self._atten.b_seq_len = self._b_seq_len
        self._atten.max_actual_seq_len = max(seq_lens)
        self._atten.is_prefill = False
        # Host mirror of the padded lens: same numbers the device lengths
        # hold (steady steps grew both by one), so the runner's per-step
        # prepare hook plans without a device sync.
        self._atten.b_seq_len_cpu = torch.tensor(padded_lens, dtype=torch.long)
        # Row `seq_len - 1` of each slot: where this step's K/V goes.
        self._atten.cur_select_index = table[self._b_req_idx, self._b_seq_len - 1]
        self._atten.b_start_loc = None
        self._clear_chunked_metadata()
        return len(padded_slots)

    def plan_extend_rows(
        self,
        slots: Sequence[int],
        seq_starts: Sequence[int],
        seq_lens: Sequence[int],
    ) -> tuple[list[int], list[int]]:
        """Lay out the rows :meth:`begin_extend` would submit, without touching the device.

        The upload path builds ``input_ids`` before the metadata setter runs, so it
        needs the row plan (flattening + graph padding) on the host. Returns exactly
        what :meth:`begin_extend` with the same args installs (pinned by tests).

        Args:
            slots: Slot id per chunk.
            seq_starts: First cache row each chunk writes.
            seq_lens: Total cached length per sequence once the chunk lands.

        Returns:
            ``(rows_slot, rows_len)`` host lists, filler included.
        """
        starts, ends = list(seq_starts), list(seq_lens)
        chunk_lens = [end - start for start, end in zip(starts, ends, strict=True)]
        rows_slot, rows_len = flatten_extend_rows(slots, starts, chunk_lens)
        if self._filler_slot is not None:
            pad = self._runner.graph_batch_size(len(rows_slot)) - len(rows_slot)
            if pad > 0:
                filler_len = min(max(ends), self.max_seq_len)
                rows_slot += [self._filler_slot] * pad
                rows_len += [filler_len] * pad
        return rows_slot, rows_len

    def pad_decode_rows(
        self, slots: Sequence[int], seq_lens: Sequence[int]
    ) -> tuple[list[int], list[int]]:
        """Grow the batch to the next captured CUDA-graph size, if there is one.

        Graphs are captured for a fixed grid, so an unpadded batch would fall back
        to eager and lose the graph's win; filler-slot rows carry the batch's max
        length. The upload path calls this too, to know the padded width before it
        builds the inputs. With one slot there is nothing to spare, so it passes
        through.
        """
        slots, seq_lens = list(slots), list(seq_lens)
        if self._filler_slot is None:
            return slots, seq_lens

        target = self._runner.graph_batch_size(len(slots))
        pad = target - len(slots)
        if pad <= 0:
            return slots, seq_lens

        filler_len = min(max(seq_lens), self.max_seq_len)
        return slots + [self._filler_slot] * pad, seq_lens + [filler_len] * pad

    def reset(self) -> None:
        """Forget the running set so the next decode rebuilds its metadata."""
        self._host_slots, self._host_lens = [], []

    @property
    def seq_lens(self) -> torch.Tensor:
        """Cache length per row of the batch last submitted (filler included, so
        shaped for the model call). A token at cache row ``seq_len - 1`` sits at
        that absolute position, serving both decode and extend.
        """
        if self._b_seq_len is None:
            raise RuntimeError("begin_decode() must run before seq_lens is read")
        return self._b_seq_len

    # -------------------------------------------------------------- internals #
    def _clear_chunked_metadata(self) -> None:
        """Drop the chunked-prefill fields for a non-chunked pass.

        The metadata is one reused instance, so a grid pass that armed them would
        otherwise leak them into the next decode and reroute its attention.
        """
        self._atten.b_prefix_len = None
        self._atten.b_kv_base = None
        self._atten.max_chunk_len = 0

    def _flatten_rows(
        self,
        slots: Sequence[int],
        starts: Sequence[int],
        chunk_lens: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand ``(slot, start, len)`` triples into one row per token.

        Returns ``(row_slots, row_lens)``; row ``r`` of request ``i`` has
        ``row_slots[r] == slots[i]`` and ``row_lens[r]`` = its absolute position + 1.
        Built with gathers, not a per-token loop, so host cost stays flat.
        """
        lens = self._to_device(chunk_lens)
        # Host sum of a list the caller held: no .item() round-trip.
        total = sum(chunk_lens)
        # Which request each row belongs to, then its offset within that stretch.
        row_req = torch.repeat_interleave(torch.arange(len(slots), device=self.device), lens)
        stretch_starts = torch.cumsum(lens, 0) - lens
        within = torch.arange(total, device=self.device) - stretch_starts[row_req]
        row_lens = self._to_device(starts)[row_req] + within + 1
        row_slots = self._to_device(slots)[row_req]
        return row_slots, row_lens

    def _to_device(self, values: Sequence[int]) -> torch.Tensor:
        """Upload a host list as a fresh int64 tensor.

        A new allocation, not a reused buffer: the previous step's tensor may still
        be queued, and overwriting it would race with pending kernels. Only happens
        when the running set changes, so off the steady-state path.
        """
        return torch.tensor(list(values), dtype=torch.long, device=self.device)
