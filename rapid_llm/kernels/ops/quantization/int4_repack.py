"""One-time int4 weight preprocessing: int32 words to byte pairs.

The fused MoE kernel's int4 mode consumes two nibbles per ``uint8`` byte along
K (vLLM's layout): a byte tile loads with each byte repeated in its two nibble
rows (a 2x repeat L1 absorbs) and the in-loop unpack is one shift-and-mask per
element, no 3-D expand, no ``tl.reshape``. Checkpoints (and
``quantize_int4_groupwise``) ship eight nibbles per int32 word, so stacked
expert weights cross this bridge once at load — ``awq_marlin_repack``'s role
in vLLM.

Staying on the word format measured ~10x slower on an H100 (Qwen3-30B-A3B,
t4096: 18109 us against 1916) — 64 KB per pipeline stage where the byte tile
needs 16 KB, every word fetched 8x instead of 2x. The failure is the format,
not the idiom.

Usage:
    kernel_w = repack_int4_experts(checkpoint_w)  # [E, N, K//8] -> [E, N, K//2]
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _int4_repack_kernel(
    src_ptr,
    dst_ptr,
    num_words,
    BLOCK_W: tl.constexpr,
):
    """One program repacks ``BLOCK_W`` int32 words into ``4 * BLOCK_W`` bytes.

    Word ``i`` holds the nibbles of K indices ``[8i, 8i+7]`` (nibble ``j`` at
    bits ``4j``, the packing ``quantize_int4_groupwise`` and GPTQ/canonical AWQ
    checkpoints share). Byte ``4i + m`` takes nibble ``2m`` as its low half and
    nibble ``2m + 1`` as its high half, so byte ``b`` covers K ``[2b, 2b+1]`` —
    the order the GEMM kernel's ``(offs_k % 2) * 4`` shifter expects.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_W + tl.arange(0, BLOCK_W)
    mask = offs < num_words
    w = tl.load(src_ptr + offs, mask=mask, other=0)  # [BLOCK_W] int32

    # Nibbles at even word positions (bits 0, 8, 16, 24) land in the low half
    # of the four output bytes; the odd positions four bits up land in the high.
    shifts = (tl.arange(0, 4) * 8).to(tl.int32)
    lo = (w[:, None] >> shifts[None, :]) & 0xF  # [BLOCK_W, 4]
    hi = (w[:, None] >> (shifts[None, :] + 4)) & 0xF
    out = lo | (hi << 4)  # each lane 0..255

    offs_byte = offs[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(dst_ptr + offs_byte, out.to(tl.uint8), mask=mask[:, None])


def repack_int4_experts(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K//8]`` int32 (8 nibbles per word) to ``[..., K//2]`` uint8.

    A pure layout change: every nibble keeps its value and its K index, so the
    dequantisation arithmetic — group scales, zero points — is untouched and
    kernel outputs stay bit-identical. Leading dims are preserved, so the same
    op serves stacked MoE experts ``[E, N, K//8]`` and a dense ``[N, K//8]``
    weight.

    Args:
        packed: int32 tensor whose last dim packs 8 int4 values per word
            (the ``quantize_int4_groupwise`` / GPTQ / canonical-AWQ layout).

    Returns:
        uint8 tensor with identical leading dims and a last dim 4x longer.
    """
    if packed.dtype != torch.int32:
        raise ValueError(f"packed int4 weights must be int32, got {packed.dtype}")
    if not packed.is_contiguous():
        packed = packed.contiguous()
    words = packed.reshape(-1)
    num_words = words.numel()
    out_shape = (*packed.shape[:-1], packed.shape[-1] * 4)
    if num_words == 0:
        return torch.empty(out_shape, dtype=torch.uint8, device=packed.device)
    out = torch.empty(num_words * 4, dtype=torch.uint8, device=packed.device)
    # 1024 words = 4 KB read and 4 KB written per program; the op is one-time
    # and bandwidth-bound, so the block size is not tuned per shape.
    BLOCK_W = 1024
    grid = (triton.cdiv(num_words, BLOCK_W),)
    _int4_repack_kernel[grid](words, out, num_words, BLOCK_W=BLOCK_W, num_warps=4)
    return out.reshape(out_shape)
