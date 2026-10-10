"""Native MLA attention: latent-cache decode and chunked-upsample prefill.

MLA caches one latent row per token — ``c_kv`` (``kv_lora_rank``) plus the
rope segment ``k_pe`` (``qk_rope_head_dim``) shared by every head — instead of
per-head K/V. Decode attends ``q = [q_nope_absorbed, q_pe]`` against the latent
directly (MQA over the cache, V = the ``c_kv`` half), so no per-head K/V is
ever materialised; prefill cannot absorb (every new token queries every other
new token), so it up-projects the latent to per-head K/V in chunks and hands
the packed batch to the existing no-pad prefill kernel.

Usage:
    out = mla_decode(q, latent_cache, block_table, cache_seqlens,
                     max_seq_len=max_len, sm_scale=scale)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .flashattention2_nopad import flash_attention2_no_pad
from .flashdecoding import flash_decode_stage2

#: Width of the rope segment inside a latent row (and of ``q_pe``). The whole
#: MLA ecosystem specialises on the DeepSeek geometry — FlashMLA's CUDA kernel
#: hard-codes head_dim_ckv=576 / head_dim_v=512 — and the ``MlaDecodeOp``
#: contract carries no dimension argument, so the split is fixed here: the
#: lora half is whatever remains of the cache row. A model with a different
#: rope width fails the power-of-two check loudly instead of mis-splitting.
QK_ROPE_HEAD_DIM = 64

#: History length per stage-1 program; matches flash_decoding's partition so
#: the shared stage 2 sees the same geometry.
_PARTITION_SIZE = 128
_BLOCK_N = 16

#: Token budget per up-projection chunk in :func:`mla_prefill`. The chunked
#: loop caps the transient GEMM workspace on long prompts; the K/V buffers
#: themselves are sized by the step's token count either way. Module-level so
#: tests can shrink it and force chunk crossings.
_PREFILL_UPSAMPLE_CHUNK = 8192


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _latent_split(q: torch.Tensor, kv_cache: torch.Tensor) -> tuple[int, int]:
    """Validate the decode geometry and return ``(kv_lora_rank, rope_dim)``.

    ``q`` carries the absorbed nope part followed by the rope part, so its
    width must equal the cache row's; the value width is the lora half.
    """
    if q.dim() != 3:
        raise ValueError(f"mla_decode wants q [batch, heads, qk_dim], got {tuple(q.shape)}")
    if kv_cache.dim() != 3:
        raise ValueError(
            f"mla_decode wants kv_cache [pages, page_size, latent_dim], got {tuple(kv_cache.shape)}"
        )
    qk_dim = q.shape[-1]
    latent_dim = kv_cache.shape[-1]
    if qk_dim != latent_dim:
        raise ValueError(f"q width {qk_dim} != latent row width {latent_dim}")
    lora_rank = latent_dim - QK_ROPE_HEAD_DIM
    if not _is_pow2(lora_rank):
        raise ValueError(
            f"kv_lora_rank must be a power of two for the Triton loads, got {lora_rank} "
            f"(latent row {latent_dim} - rope {QK_ROPE_HEAD_DIM})"
        )
    return lora_rank, QK_ROPE_HEAD_DIM


@triton.jit
def _mla_decode_stage1_kernel(
    Q,
    Latent,  # flat [num_pages * page_size, latent_dim] rows
    Block_Table,
    Cache_Seqlens,
    sm_scale,
    Mid_O,
    Mid_O_LogExpSum,
    stride_q_b,
    stride_q_h,
    stride_q_d,
    stride_lat_row,
    stride_lat_d,
    stride_bt_b,
    stride_bt_p,
    mido_batch_stride,
    mido_heads_stride,
    mido_partitions_stride,
    mido_dim_stride,
    miles_batch_stride,
    miles_heads_stride,
    miles_partitions_stride,
    page_size,
    LORA_RANK: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Partial latent attention for one (batch row, head, history partition).

    The latent row is loaded as two power-of-two segments — ``c_kv`` over
    ``arange(0, LORA_RANK)`` and ``k_pe`` over ``arange(0, ROPE_DIM)`` — because
    576 is not a power of two and a masked 1024-wide load would waste 44% of
    the width. The ``c_kv`` load serves both the score and the value: V *is*
    the lora half, which is the whole point of the absorbed decode form.
    """
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    seq_block_pid = tl.program_id(2)

    cur_batch_seq_len = tl.load(Cache_Seqlens + batch_pid)
    cur_batch_partition_start_index = seq_block_pid * BLOCK_SEQ
    cur_batch_partition_end_index = tl.minimum(
        cur_batch_seq_len, cur_batch_partition_start_index + BLOCK_SEQ
    )

    num_blocks = tl.where(
        cur_batch_partition_end_index - cur_batch_partition_start_index <= 0,
        0,
        (cur_batch_partition_end_index - cur_batch_partition_start_index + BLOCK_N - 1) // BLOCK_N,
    )

    offs_n = cur_batch_partition_start_index + tl.arange(0, BLOCK_N)
    offs_lora = tl.arange(0, LORA_RANK)
    offs_rope = tl.arange(0, ROPE_DIM)

    q_base = Q + batch_pid * stride_q_b + head_pid * stride_q_h
    q_nope = tl.load(q_base + offs_lora * stride_q_d)  # [LORA_RANK]
    q_pe = tl.load(q_base + (LORA_RANK + offs_rope) * stride_q_d)  # [ROPE_DIM]

    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([LORA_RANK], dtype=tl.float32)

    for start_n in range(0, num_blocks, 1):
        pos = offs_n + start_n * BLOCK_N  # [BLOCK_N] history positions
        pos_mask = pos < cur_batch_partition_end_index

        # Position -> physical row: page id from the table, offset inside it.
        # Per-position div/mod, because a BLOCK_N window may straddle pages.
        page_id = tl.load(
            Block_Table + batch_pid * stride_bt_b + (pos // page_size) * stride_bt_p,
            mask=pos_mask,
            other=0,
        )
        rows = page_id * page_size + pos % page_size
        lat_base = Latent + rows[:, None] * stride_lat_row

        c_kv = tl.load(
            lat_base + offs_lora[None, :] * stride_lat_d, mask=pos_mask[:, None], other=0.0
        )  # [BLOCK_N, LORA_RANK]
        k_pe = tl.load(
            lat_base + (LORA_RANK + offs_rope)[None, :] * stride_lat_d,
            mask=pos_mask[:, None],
            other=0.0,
        )  # [BLOCK_N, ROPE_DIM]

        # fp32 promotion of the dot — the same rounding ladder flash_decoding
        # documents: a 576-term bf16 accumulation would cost ~1e-2 per layer.
        scores = tl.sum(q_nope[None, :].to(tl.float32) * c_kv.to(tl.float32), axis=1)
        scores += tl.sum(q_pe[None, :].to(tl.float32) * k_pe.to(tl.float32), axis=1)
        scores *= sm_scale
        scores = tl.where(pos_mask, scores, float("-inf"))  # [BLOCK_N]

        current_max = tl.max(scores)
        m_ij = tl.maximum(m_i, current_max)
        p = tl.exp(scores - m_ij)
        alpha = tl.exp(m_i - m_ij)
        d_i = alpha * d_i + tl.sum(p, axis=0)
        acc = alpha * acc + tl.sum(p[:, None] * c_kv.to(tl.float32), axis=0)
        m_i = m_ij

    # Same contract as flash_decoding's stage 1: normalised partial plus LSE,
    # and no store at all when the partition lies beyond this row's history —
    # stage 2 reduces exactly ``cdiv(seq_len, BLOCK_SEQ)`` written partitions.
    need_store = tl.where(num_blocks == 0, 0, 1)
    for _ in range(0, need_store, 1):
        off_mid_o = (
            batch_pid * mido_batch_stride
            + head_pid * mido_heads_stride
            + seq_block_pid * mido_partitions_stride
            + offs_lora * mido_dim_stride
        )
        tl.store(Mid_O + off_mid_o, acc / d_i)
        off_mid_lse = (
            batch_pid * miles_batch_stride
            + head_pid * miles_heads_stride
            + seq_block_pid * miles_partitions_stride
        )
        tl.store(Mid_O_LogExpSum + off_mid_lse, m_i + tl.log(d_i))


@torch.no_grad()
def mla_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    max_seq_len: int,
    sm_scale: float = 1.0,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decode attention over the MLA latent KV cache (native Triton row).

    Args:
        q: ``[batch, num_heads, qk_head_dim]`` absorbed query: the nope part
            already multiplied through ``w_uk`` (width ``kv_lora_rank``),
            followed by the rope part (width 64).
        kv_cache: ``[num_pages, page_size, kv_lora_dim]`` latent cache, rows
            laid out ``[c_kv, k_pe]``. The mainline token-slot pool
            ``[max_tokens, 1, latent_dim]`` is the ``page_size == 1`` case.
        block_table: ``[batch, max_pages]`` page ids per sequence.
        cache_seqlens: ``[batch]`` cached length per sequence, this step's
            token included.
        max_seq_len: Longest row, sizing the kernel grid.
        sm_scale: Softmax scale.
        return_lse: Also return the per-row, per-head logsumexp. Empty histories
            produce a zero output and ``-inf`` LSE.

    Returns:
        ``[batch, num_heads, kv_lora_rank]`` — the per-head mixture of the
        cached ``c_kv`` rows, ready for the ``w_uv`` up-projection; when
        ``return_lse`` is true, paired with ``[batch, num_heads]`` fp32 LSE.
    """
    lora_rank, rope_dim = _latent_split(q, kv_cache)
    batch, num_heads, _ = q.shape
    num_pages, page_size, latent_dim = kv_cache.shape
    if max_seq_len <= 0:
        output = torch.zeros((batch, num_heads, lora_rank), dtype=q.dtype, device=q.device)
        if return_lse:
            lse = torch.full(
                (batch, num_heads), -float("inf"), dtype=torch.float32, device=q.device
            )
            return output, lse
        return output

    # The kernel walks flat rows; a paged cache is allocated contiguously, and
    # silently paying a copy here would hide a layout bug in the caller.
    if not kv_cache.is_contiguous():
        raise ValueError("mla_decode wants a contiguous latent cache")
    flat = kv_cache.view(num_pages * page_size, latent_dim)

    max_num_partitions = (max_seq_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE
    mid_o = torch.empty(
        (batch, num_heads, max_num_partitions, lora_rank), dtype=torch.float32, device=q.device
    )
    mid_o_logexpsum = torch.empty(
        (batch, num_heads, max_num_partitions), dtype=torch.float32, device=q.device
    )

    grid = (batch, num_heads, max_num_partitions)
    _mla_decode_stage1_kernel[grid](
        q,
        flat,
        block_table,
        cache_seqlens,
        sm_scale,
        mid_o,
        mid_o_logexpsum,
        *q.stride(),
        *flat.stride(),
        *block_table.stride(),
        *mid_o.stride(),
        *mid_o_logexpsum.stride(),
        page_size,
        LORA_RANK=lora_rank,
        ROPE_DIM=rope_dim,
        BLOCK_SEQ=_PARTITION_SIZE,
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=2,
    )

    # Cross-partition LSE reduction is geometry-agnostic (any power-of-two
    # head dim), so the MLA row reuses flash_decoding's stage 2 verbatim.
    atten_output = torch.empty((batch, num_heads, lora_rank), dtype=q.dtype, device=q.device)
    lse_output = (
        torch.empty((batch, num_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    flash_decode_stage2(
        mid_o,
        mid_o_logexpsum,
        atten_output,
        cache_seqlens,
        _PARTITION_SIZE,
        lse_output=lse_output,
    )
    return (atten_output, lse_output) if return_lse else atten_output


@torch.no_grad()
def mla_decode_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    max_seq_len: int,
    sm_scale: float = 1.0,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for MLA decode over paged latent cache."""

    batch, num_heads, _ = q.shape
    _num_pages, page_size, latent_dim = kv_cache.shape
    lora_rank = latent_dim - qk_rope_head_dim
    out = torch.zeros((batch, num_heads, lora_rank), dtype=q.dtype, device=q.device)
    lse = torch.full((batch, num_heads), -float("inf"), dtype=torch.float32, device=q.device)

    for b in range(batch):
        length = int(cache_seqlens[b])
        if length == 0:
            continue
        num_pages = (length + page_size - 1) // page_size
        pages = block_table[b, :num_pages].tolist()
        latent = torch.cat([kv_cache[p] for p in pages], dim=0)[:length].float()
        c_kv, k_pe = latent[:, :lora_rank], latent[:, lora_rank:]
        qf = q[b].float()
        scores = (qf[:, :lora_rank] @ c_kv.T + qf[:, lora_rank:] @ k_pe.T) * sm_scale
        lse[b] = torch.logsumexp(scores, dim=-1)
        out[b] = (scores.softmax(dim=-1) @ c_kv).to(q.dtype)
    return (out, lse) if return_lse else out


@torch.no_grad()
def mla_prefill(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    c_kv: torch.Tensor,
    k_pe: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    sm_scale: float,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Prefill attention via chunked up-projection of the fresh latent.

    Args:
        q_nope: ``[tokens, num_heads, qk_nope_head_dim]`` query, un-absorbed.
        q_pe: ``[tokens, num_heads, qk_rope_head_dim]`` rope segment of q.
        c_kv: ``[tokens, kv_lora_rank]`` fresh latent rows.
        k_pe: ``[tokens, qk_rope_head_dim]`` fresh rope keys, one per token,
            shared by all heads.
        w_uk: ``[num_heads, kv_lora_rank, qk_nope_head_dim]`` per-head K
            up-projection (a transposed view of the ``kv_b_proj`` K half).
        w_uv: ``[num_heads, kv_lora_rank, v_head_dim]`` per-head V
            up-projection, same source.
        sm_scale: Softmax scale, plain ``1 / sqrt(qk_nope + qk_rope)`` times
            whatever mscale the rope config applies.
        b_start_loc: ``[batch]`` first packed row of each sequence.
        b_seq_len: ``[batch]`` true length of each sequence.
        max_seq_len: Longest sequence, sizing the flash tiles.

    Returns:
        ``[tokens, num_heads, v_head_dim]`` attention output.
    """
    tokens, num_heads, nope_dim = q_nope.shape
    rope_dim = q_pe.shape[-1]
    v_dim = w_uv.shape[-1]
    qk_dim = nope_dim + rope_dim
    pad_dim = triton.next_power_of_2(max(qk_dim, v_dim))

    q_pad = q_nope.new_zeros((tokens, num_heads, pad_dim))
    q_pad[..., :nope_dim] = q_nope
    q_pad[..., nope_dim:qk_dim] = q_pe
    k_pad = q_nope.new_zeros((tokens, num_heads, pad_dim))
    v_pad = q_nope.new_zeros((tokens, num_heads, pad_dim))

    # Chunked up-projection to avoid OOM on long sequences
    for start in range(0, tokens, _PREFILL_UPSAMPLE_CHUNK):
        end = min(start + _PREFILL_UPSAMPLE_CHUNK, tokens)
        latent = c_kv[start:end]  # [t, kv_lora_rank]
        k_pad[start:end, :, :nope_dim] = torch.einsum("tl,hld->thd", latent, w_uk)
        k_pad[start:end, :, nope_dim:qk_dim] = k_pe[start:end, None, :]
        v_pad[start:end, :, :v_dim] = torch.einsum("tl,hld->thd", latent, w_uv)

    out = flash_attention2_no_pad(
        q_pad, k_pad, v_pad, sm_scale, b_start_loc, b_seq_len, max_seq_len
    )

    return out[..., :v_dim]
