"""Paged KV cache: block allocation with reference counting, plus sizing.

:class:`MemoryProfiler` measures free device memory to decide how many cache blocks
fit; :class:`KVCacheManager` hands out and ref-counts block indices so prefill and
decode reserve rows and release them when a sequence ends.

Usage:
    idx = kv.alloc_kvcache_index(need_size); kv.release_ref(idx)
"""

import gc

import torch

from ..distributed.dp_attention import dp_attention_region
from ..utils.logger import get_logger
from .attention_metadata import AttentionMetadata

logger = get_logger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """Return the size of one element of ``dtype`` in bytes."""
    return torch.tensor([], dtype=dtype).element_size()


class MemoryProfiler:
    """Estimates how many KV-cache tokens fit in the remaining GPU memory.

    A short dummy forward measures peak activation memory; the leftover budget
    (``total * utilization - peak``) is divided by the per-token KV size. The
    constructed model supplies every dimension, so no dummy-input config is needed.

    Args:
        num_layers: Decoder layer count.
        kv_row: ``(slots, dim)`` of one token's per-layer cache row —
            ``(2 * kv_heads, head_dim)`` for MHA/GQA, ``(1, kv_lora_rank +
            qk_rope_head_dim)`` for MLA (latent row, replicated across ranks).
        gpu_memory_utilization: Fraction of GPU memory the cache may occupy.
        dtype: KV-cache dtype.
        device: Torch device string.
        reserved_bytes: Extra budget to withhold (e.g. CUDA graph workspace).
    """

    def __init__(
        self,
        num_layers: int,
        kv_row: tuple[int, int],
        gpu_memory_utilization: float = 0.9,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        reserved_bytes: int = 0,
    ) -> None:
        self.num_layers = num_layers
        self.kv_row = kv_row
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.device = device
        self.reserved_bytes = reserved_bytes

    def _kv_bytes_per_token(self) -> int:
        slots, dim = self.kv_row
        return slots * dim * self.num_layers * get_dtype_size(self.dtype)

    def _run_dummy_forward(self, model, vocab_size: int, seq_len: int = 32) -> None:
        """Drive one prefill pass so peak activation memory is recorded."""
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=self.device)
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        dummy = AttentionMetadata()
        dummy.kv_buffer = [
            torch.empty(
                (seq_len, *self.kv_row),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        dummy.cur_select_index = torch.arange(seq_len, dtype=torch.int32, device=self.device)
        dummy.b_req_tokens_table = torch.arange(
            seq_len, dtype=torch.int32, device=self.device
        ).view(1, seq_len)
        dummy.b_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy.b_req_idx = torch.tensor([0], dtype=torch.int32, device=self.device)
        dummy.b_seq_len = torch.tensor([seq_len], dtype=torch.int32, device=self.device)
        dummy.max_actual_seq_len = seq_len

        with torch.no_grad(), dp_attention_region(input_ids.numel()):
            model(input_ids, position_ids, dummy)

    def available_kv_blocks(self, model, vocab_size: int) -> int:
        """Return the number of KV-cache tokens that fit in free GPU memory.

        Falls back to a small fixed budget on CPU (profiling APIs do not apply),
        keeping unit tests runnable without a GPU.
        """
        if torch.device(self.device).type == "cpu":
            logger.warning("CUDA unavailable; using a minimal KV cache for CPU execution")
            return 4096

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        _, total_gpu_memory = torch.cuda.mem_get_info()

        self._run_dummy_forward(model, vocab_size)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        torch.cuda.empty_cache()
        # Account for memory allocated outside the torch caching allocator.
        torch_current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        free_now, _ = torch.cuda.mem_get_info()
        non_torch = (total_gpu_memory - free_now) - torch_current
        if non_torch > 0:
            peak_memory += non_torch

        budget = total_gpu_memory * self.gpu_memory_utilization - peak_memory - self.reserved_bytes
        num_blocks = max(int(budget // self._kv_bytes_per_token()), 0)

        logger.info(
            "KV-cache profiling: total=%.2f GB peak=%.2f GB -> %d cache tokens (util=%.2f)",
            total_gpu_memory / 1024**3,
            peak_memory / 1024**3,
            num_blocks,
            self.gpu_memory_utilization,
        )

        gc.collect()
        torch.cuda.empty_cache()
        return num_blocks


class KVCacheManager:
    """Owns the paged KV buffers and hands out cache rows by reference count.

    One row per token (``block_size=1``). ``kv_mem_use_state[i]`` is row ``i``'s
    refcount (free at zero); :attr:`can_use_mem_size` counts free rows.
    :meth:`alloc_kvcache_index` picks the cheapest strategy that fits: bump cursor,
    contiguous-run search, then scattered rows.

    Args:
        num_layers: Decoder layer count.
        kv_row: ``(slots, dim)`` of one token's per-layer cache row (as in
            :class:`MemoryProfiler`).
        gpu_num_blocks: Cache capacity in blocks (profiled or caller-set).
        block_size: Tokens per block; only 1 is implemented.
        dtype: KV-cache dtype.
        device: Torch device string.
    """

    def __init__(
        self,
        num_layers,
        kv_row,
        gpu_num_blocks,
        block_size=1,
        dtype=torch.bfloat16,
        device="cuda",
        watermark: float = 0.1,
        hysteresis: float = 0.05,
    ):
        self.num_layers = num_layers
        self.kv_row = tuple(kv_row)
        self.gpu_num_blocks = gpu_num_blocks
        self.block_size = block_size
        self.max_num_tokens = gpu_num_blocks * block_size

        self.dtype = dtype
        self.device = device
        self.can_use_mem_size = gpu_num_blocks  # rows currently free

        # Watermark: refuse new requests when free blocks drop below this fraction.
        # Hysteresis: after a dip under the watermark, admission resumes only above
        # watermark + recovery band, so a level oscillating around one threshold
        # cannot flap admit/preempt (O9).
        self._watermark_blocks = int(gpu_num_blocks * watermark)
        self._recover_blocks = int(gpu_num_blocks * hysteresis)
        self._under_pressure = False
        logger.info(
            "KV cache: %d blocks, watermark=%d blocks (%.0f%%), recovery band=%d blocks",
            gpu_num_blocks,
            self._watermark_blocks,
            watermark * 100,
            self._recover_blocks,
        )

        # Row indices to hand out, and the per-row reference count.
        self.kv_mem_pos_indices = torch.arange(
            0, self.max_num_tokens, dtype=torch.long, device=self.device
        )
        self.kv_mem_use_state = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, device=self.device
        )
        # Pre-cast int32 copy so the bump fast path returns a view, not a per-step cast.
        self.kv_mem_pos_indices_int32 = self.kv_mem_pos_indices.to(torch.int32)
        # Append-only cursor, and whether it is still exact (invalidated by a partial
        # free, restored by ``free_all``).
        self._bump_cursor = 0
        self._bump_is_exact = True

        # Initialize the gpu_kv_buffer
        self.init_kv_buffers(self.max_num_tokens, self.kv_row, num_layers, dtype, device)

    def can_admit(self, need_blocks: int) -> bool:
        """Check if a new request requiring *need_blocks* can be admitted.

        False when free capacity after allocation would drop below the watermark
        (preventing eviction cascades). A pure read, but it remembers which side of
        the watermark the level last sat on: after a dip the bar rises by the recovery
        band until the level climbs back, so an oscillating level cannot flap admission
        (O9 hysteresis).
        """
        if self._under_pressure:
            if self.can_use_mem_size >= self._watermark_blocks + self._recover_blocks:
                self._under_pressure = False
        elif self.can_use_mem_size < self._watermark_blocks:
            self._under_pressure = True
        floor = (
            self._watermark_blocks + self._recover_blocks
            if self._under_pressure
            else self._watermark_blocks
        )
        return self.can_use_mem_size - need_blocks >= floor

    @property
    def utilization(self) -> float:
        """Fraction of KV cache currently in use (0.0 empty, 1.0 full)."""
        return 1.0 - (self.can_use_mem_size / self.max_num_tokens)

    def init_kv_buffers(
        self,
        max_num_tokens,
        kv_row,
        num_layers,
        dtype,
        device: str = "cuda",
    ) -> None:
        """Pre-allocate one KV tensor per layer, ``[max_num_tokens, *kv_row]``.

        MHA/GQA row is ``(2 * kv_heads, head_dim)`` (K heads then V, so a decode step
        writes both in one launch); MLA is ``(1, latent_dim)`` (K and V share one
        latent vector, written whole).
        """
        # TODO: reshape into [blocks, block_size, ...] to support PagedAttention.
        self.gpu_kv_buffer = [
            torch.empty((max_num_tokens, *kv_row), dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        logger.debug(f"gpu_kv_buffer per layer shape: {self.gpu_kv_buffer[0].shape}")

    @torch.no_grad()
    def alloc_kvcache(self, need_size):
        """Reserve ``need_size`` free rows, wherever they are. Returns ``None`` if short."""
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        select_index = can_use_pos_index[0:need_size]
        self.add_ref(select_index)

        return select_index

    @torch.no_grad()
    def alloc_contiguous_kvcache(self, need_size):
        """Reserve ``need_size`` *consecutive* free rows, or ``None`` if no such run.

        Returns:
            ``(select_index, start_index, end_index)``, or ``None``.
        """
        if need_size > self.can_use_mem_size:
            logger.warning(
                f"warn no enough contiguous cache need_size {need_size} left_size {self.can_use_mem_size}"
            )
            return None

        can_use_pos_index = torch.nonzero(self.kv_mem_use_state == 0).view(-1)
        N = can_use_pos_index.numel()
        if need_size <= N:
            # Two views of the free list offset by need_size - 1: start_indices[j] and
            # end_indices[j] are a candidate window's ends (last valid start is at
            # N - need_size; slicing excludes the stop, hence + 1).
            start_indices = can_use_pos_index[: N - need_size + 1]
            end_indices = can_use_pos_index[need_size - 1 :]
            # A window is consecutive exactly when its ends differ by need_size - 1;
            # larger means a used row sits between.
            contiguous_blocks = (end_indices - start_indices == need_size - 1).nonzero(
                as_tuple=True
            )[0]

            if contiguous_blocks.numel() > 0:
                start_index = start_indices[contiguous_blocks[0]].item()  # first run wins
                end_index = start_index + need_size
                select_index = self.kv_mem_pos_indices[start_index:end_index]
                self.add_ref(select_index)
                return select_index, start_index, end_index

        return None

    @torch.no_grad()
    def alloc_kvcache_index(self, need_size):
        """Reserve ``need_size`` cache rows, preferring a contiguous run.

        On the decode hot path. The general search costs a ``nonzero`` plus two
        ``.item()`` reads (three device syncs that stall the pipeline). While the cache
        is append-only (every ``generate()`` starts here, via :meth:`free_all`) the
        answer is the next ``need_size`` rows, so the bump cursor returns them with no
        device reads; a partial free falls back to the search.
        """
        if self._bump_is_exact and self._bump_cursor + need_size <= self.max_num_tokens:
            start = self._bump_cursor
            select_index = self.kv_mem_pos_indices_int32[start : start + need_size]
            self.kv_mem_use_state[start : start + need_size] += 1
            self._bump_cursor += need_size
            self.can_use_mem_size -= need_size
            return select_index

        alloc_mem = self.alloc_contiguous_kvcache(need_size)
        if alloc_mem is not None:
            select_index, _start_index, _end_index = alloc_mem
        else:
            select_index = self.alloc_kvcache(need_size)
        return select_index.to(torch.int32)

    @torch.no_grad()
    def add_ref(self, token_index: torch.Tensor):
        """Increment the reference count of the given rows.

        Only previously-free rows reduce :attr:`can_use_mem_size`; a second reference
        on an already-held row costs no capacity.
        """
        state = self.kv_mem_use_state[token_index]
        has_used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size -= all_tokens - has_used_tokens

        self.kv_mem_use_state[token_index] += 1
        return

    @torch.no_grad()
    def release_ref(self, token_index: torch.Tensor):
        """Decrement the reference count of the given rows, freeing those that reach zero.

        ``token_index`` may name a row more than once (the engine releases prefill and
        every decode step in one tensor), so counts are collapsed with ``unique`` first.
        """
        # Freeing leaves holes, so the append-only cursor no longer describes the free
        # list; fall back to searching until ``free_all``.
        self._bump_is_exact = False
        token_index, counts = token_index.unique(return_counts=True)
        self.kv_mem_use_state[token_index] -= counts
        state = self.kv_mem_use_state[token_index]
        used_tokens = torch.count_nonzero(state).item()
        all_tokens = len(state)
        self.can_use_mem_size += all_tokens - used_tokens
        return

    @torch.no_grad()
    def claim(self, num_rows: int) -> None:
        """Hand the first ``num_rows`` rows to an external owner, permanently.

        Continuous batching maps each slot onto a fixed contiguous region (see
        :class:`~rapid_llm.executor.slot_batch.SlotBatch`); marking those rows used
        keeps :attr:`can_use_mem_size` honest, and the bump cursor resumes past the
        claimed region so both schemes share one pool.
        """
        if num_rows > self.can_use_mem_size:
            raise ValueError(f"cannot claim {num_rows} rows: only {self.can_use_mem_size} are free")
        self.kv_mem_use_state[:num_rows] += 1
        self.can_use_mem_size -= num_rows
        self._bump_cursor = max(self._bump_cursor, num_rows)

    @torch.no_grad()
    def free(self, free_index):
        """Release rows by index, logging when that empties the cache."""
        free_index = free_index.long()
        self.release_ref(free_index)
        if self.can_use_mem_size == len(self.kv_mem_use_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    @torch.no_grad()
    def free_all(
        self,
    ):
        """Drop every reference at once, as each ``generate()`` call starts by doing."""
        self.can_use_mem_size = len(self.kv_mem_use_state)
        self.kv_mem_use_state[:] = 0
        # The cache is empty again, so appending from row 0 is exact once more.
        self._bump_cursor = 0
        self._bump_is_exact = True
