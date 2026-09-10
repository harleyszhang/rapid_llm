"""Fused MoE: top-k routed experts as a Triton grouped GEMM.

Pipeline: moe_align_block_size -> GEMM1 (gate_up) -> silu_and_mul -> GEMM2
(down, router weight folded in) -> moe_sum. fp16, fp8, int8, int4 and mxfp4
packed expert weights with group-wise scales. The activation may be fp16 or
bf16 and every quantised weight tile widens to whichever it is — ``tl.dot``
needs both operands in one type, so the compute dtype is the activation's to
choose. The exceptions are :func:`fused_moe_w8a8_fp8` and
:func:`fused_moe_w8a8_int8`, where the activation is quantised too and both
operands stay 8-bit through the dot. Whether an fp8 dot lands on the fp8
tensor cores is Triton's call: ``wgmma`` only from ``BLOCK_M >= 64``, fp16
``mma.sync`` below — either way the inner loop skips the bit-trick
dequantisation the weight-only modes pay, which is where their speed comes
from (see ``_FP8_A8_PROMOTE_EVERY``).

Usage:
    out = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)
    out = fused_moe(x, qw1, qw2, tw, ids, w1_scale=s1, w2_scale=s2,
                    group_n=128, group_k=128)
    out = fused_moe_w8a8_fp8(x, qw1, qw2, tw, ids, w1_scale=s1, w2_scale=s2,
                             group_n=1, group_k=hidden)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..activation.activations import silu
from ..quantization.fp8 import FP8_E4M3_MAX, fp8_quantize_per_token
from ..quantization.w8a8 import int8_quantize_per_token
from ..quantization.w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3
from ..tile_policy import TileTier, has_native_fp8, resolve_tiles, tile_tier
from ..utils import torch_to_triton_dtype

#: ``QUANT_MODE`` values shared by the kernel and its launcher. Modes 1-4 are
#: weight-only — the activation stays fp16/bf16 — while modes 5 and 6 are true
#: W8A8: both operands enter the dot as 8-bit (e4m3 / int8). Neither can be
#: inferred from ``w1.dtype`` the way the others are, because W8A8 fp8 stores
#: the same ``uint8`` experts as weight-only fp8 and W8A8 int8 the same
#: ``int8`` experts as weight-only int8; the caller selects the mode through
#: the entry point it picks. These are plain Python ints, invisible to the
#: ``@triton.jit`` body -- Triton only resolves globals that are ``tl.constexpr``
#: instances -- so inside the kernel the modes are spelled as literals
#: (``if QUANT_MODE == 3:``) and only launcher-side code uses the names.
_QUANT_NONE = 0
_QUANT_FP8 = 1
_QUANT_INT8 = 2
_QUANT_INT4 = 3
_QUANT_MXFP4 = 4
_QUANT_FP8_A8 = 5
_QUANT_INT8_A8 = 6

#: k-tile of the quantised path: one byte per weight element, k-contiguous, so a
#: 128-wide tile is one full memory transaction per output channel.
_QUANT_BLOCK_K = 128

#: Exponent correction for the pre-sm89 path of mode 5, where *both* operands
#: are widened by the e4m3 -> fp16 bit trick and each is short a factor of 256.
#: Derived from the one imported constant rather than restated, so it cannot
#: drift from ``quantization.fp8``'s copy of the same reasoning.
_FP8_BIT_TRICK_SCALE_SQ = FP8_E4M3_BIT_TRICK_SCALE * FP8_E4M3_BIT_TRICK_SCALE

#: How many k elements mode 5 may accumulate inside Hopper's fp8 ``wgmma`` before
#: the partial sum is promoted into a real fp32 accumulator. The instruction's
#: internal accumulation is *not* full fp32, and Triton's sm90 default
#: (``max_num_imprecise_acc_default = 2**30``) never promotes at all.
#:
#: Measured on H100 (sm90), K=512 dot with both operands filling the e4m3 range,
#: against an fp32 reference over the same bytes -- p99.9 relative error:
#: unbounded 7.0e-2, promote every 128 3.3e-2, promote every 0 4.5e-6. The last
#: is not a more accurate fp8 dot: at 0 Triton drops ``wgmma`` and widens both
#: operands to fp16 ``mma.sync`` instead, so there is no precise fp8 MMA to pick.
#:
#: At the Qwen3-30B-A3B MoE geometry all three are indistinguishable (RMS relative
#: error 6.5e-3 / 6.6e-3 / 6.8e-3): the two e4m3 roundings swamp the accumulator,
#: and the grouped GEMM is bound by re-reading expert weights once per row-block,
#: not by the MMA. 128 — one promotion per BLOCK_K=128 k-tile — is what the dot
#: runs at: against 32 it bought 10-11% on the 4096-token prefill (869 vs 972 us
#: on the same tile) and nothing anywhere smaller, because below BLOCK_M=64 there
#: is no wgmma to promote.
_FP8_A8_PROMOTE_EVERY = 128

#: Logical K elements per stored element of B, for the two packed modes. int4
#: keeps two nibbles in a byte, even K in the low one: the kernel loads the dense
#: ``[BLOCK_K // 2, BLOCK_N]`` byte tile and separates the two nibble planes in
#: registers, so the unpack is one shift-and-mask with no 3-D expand and no
#: reshape. Checkpoints ship 8 nibbles per int32 word instead, so
#: :func:`repack_int4_experts` bridges the two layouts once at load.
_INT4_PACK_FACTOR = 2

#: Number of MXFP4 (e2m1) values packed per output element of the B tensor. The
#: DeepSeek-V4 checkpoint layout ships eight nibbles per ``int32`` word, so B is
#: ``[E, N, K//8]`` and the logical K is the packed last dim times 8 -- four
#: times the int4 factor, which is why the two packed modes cannot share one
#: constant when the launcher recovers ``k_logical`` from the tensor shape.
_MXFP4_PACK_FACTOR = 8

#: Largest magnitude symmetric int8 stores; the int8 A-quantising path scales
#: by this, mirroring ``FP8_E4M3_MAX`` for the e4m3 modes.
_INT8_MAX = 127.0

#: A-row count at and below which the W8A8 modes quantise the activation *inside*
#: the GEMM kernel instead of in the separate quantiser. The separate path costs
#: a launch plus its host time (~35 us per call at 1 token on an H100, against
#: ~15 us of device work in both GEMMs combined) -- on a launch-bound decode
#: step that host time is the whole fp8 W8A8 regression. Inline removes it, but
#: every program re-derives the amax over the full K of its gathered A rows, so
#: the re-read grows with the grid -- once the GEMM is the cost (not the launch)
#: the repetition is pure loss. Measured on H100 at the Qwen3-30B-A3B geometry
#: with the threshold swept (both modes, us): at 1/8 tokens inline wins ~34/19 us
#: either side of every value from 8 to 512; at 64 tokens (GEMM1's A is 64 rows)
#: inline loses ~17 us and at 512 (GEMM1's A is 512 rows) ~62 us, because the
#: amax pass over a [rows, K] tile runs in every one of the grid's
#: m-blocks x n-blocks programs. 32 sits between: it keeps every decode-shape
#: A (1 row, and top_k=8 rows into GEMM2) inline while anything whose GEMM has
#: left the launch-bound regime goes to the quantiser. Both GEMMs are judged on
#: their own A row count: GEMM1's is ``num_tokens``, GEMM2's the expanded
#: ``num_tokens * top_k``.
_INLINE_A_QUANT_MAX_ROWS = 32


# --------------------------------------------------------------------------- #
# Token alignment (2 Triton launches; every output shape is static -> no host sync)
# --------------------------------------------------------------------------- #
#: Slots per tile of the alignment kernels. Caps how much of ``topk_ids`` one
#: program holds; the scatter walks the rest in a loop.
_ALIGN_BLOCK_S = 1024

#: Whether the scatter kernel counts the experts itself, skipping both the
#: histogram launch and the ``torch.zeros`` that feeds it, is decided by one
#: tile: a program can only histogram the slots it holds. Within that limit the
#: fused form won at every size measured on H100 (slots 8 / 64 / 256 / 1024:
#: 5.7 / 7.6 / 7.3 / 11.5 us against 10.4 / 11.4 / 12.2 / 14.3, and 45 us of host
#: time against 77 flat), so there is no second threshold to tune -- the
#: [BLOCK_S, BLOCK_E] compare it adds stays cheaper than two launches even at the
#: full tile. Above one tile it is not a choice, it is unimplementable.


@triton.jit
def _moe_align_count_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_slots,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Histogram of expert ids: ``counts[e] = #{slots routed to e}``.

    Duplicate lanes in one tile hit the same address; the hardware serialises
    them, which is what makes a vector ``atomic_add`` a histogram. An atomic
    also keeps the count on the device, unlike ``bincount``, whose host read of
    the max element both stalls the launch queue and makes the layer
    uncapturable as a CUDA graph.

    A slot whose id is outside ``[0, NUM_EXPERTS)`` routes to no local expert
    and is dropped from the histogram. That is what lets a caller hand this
    kernel a buffer with holes -- the EP dispatcher marks its all-to-all
    padding rows with ``-1`` so the grouped GEMM never runs them, the same
    ``off_experts == -1`` contract sglang's expert map uses. The scatter kernel
    only ever claims ids in ``[0, NUM_EXPERTS)``, so guarding the count here is
    all it takes for the two halves to agree; without it the sentinel would
    write ``counts`` out of bounds.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs < num_slots
    experts = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    in_range = mask & (experts >= 0) & (experts < NUM_EXPERTS)
    tl.atomic_add(counts_ptr + experts, 1, mask=in_range)


@triton.jit
def _moe_align_scatter_kernel(
    topk_ids_ptr,
    counts_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_post_ptr,
    num_slots,
    BLOCK_SIZE: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    BLOCK_S: tl.constexpr,
    FUSED_COUNT: tl.constexpr,
):
    """One program per expert: writes that expert's whole padded run.

    Program ``e`` re-derives the block offsets from the full ``counts`` vector
    rather than reading a prefix-sum some earlier kernel left behind. ``E``
    programs each summing ``E`` values is the same order of work as the scan
    would be, and it removes a launch plus a global round-trip from a path whose
    cost is launches, not arithmetic.

    Order inside a run is the flat slot order, as a stable sort would give:
    ``tl.cumsum`` ranks the hits within a tile and ``written`` carries the count
    across tiles. Nothing downstream can observe the order -- each slot's output
    row is computed independently and ``_moe_sum_kernel`` indexes slots
    directly -- but an atomic cursor would make the buffer differ run to run,
    which turns any future comparison of two runs into a false positive.
    """
    e = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    if FUSED_COUNT:
        # Every program already reads every slot to find its own, so when the
        # slots fit one tile it can histogram all E experts from the same tile
        # instead of waiting on a counting kernel -- trading a [BLOCK_S, BLOCK_E]
        # compare per program for two launches (the histogram and the zeroing of
        # its output), which measured cheaper at every size that fits -- see the
        # note above ``_ALIGN_BLOCK_S``. The launcher only sets it when one tile
        # covers the slots; with it set and more slots than that, the counts
        # would silently be a prefix of the real ones.
        offs_s = tl.arange(0, BLOCK_S)
        ids_all = tl.load(topk_ids_ptr + offs_s, mask=offs_s < num_slots, other=-1)
        counts = tl.sum((ids_all[:, None] == offs_e[None, :]).to(tl.int32), axis=0)
    else:
        counts = tl.load(counts_ptr + offs_e, mask=offs_e < NUM_EXPERTS, other=0)
    padded = ((counts + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    # Exclusive prefix over experts, and the run's own extent.
    start = tl.sum(tl.where(offs_e < e, padded, 0))
    my_count = tl.sum(tl.where(offs_e == e, counts, 0))
    my_padded = ((my_count + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if e == 0:
        tl.store(num_post_ptr, tl.sum(padded))

    written = 0
    for s in range(0, num_slots, BLOCK_S):
        offs = s + tl.arange(0, BLOCK_S)
        in_range = offs < num_slots
        ids = tl.load(topk_ids_ptr + offs, mask=in_range, other=-1)
        hit = ids == e
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        tl.store(sorted_ids_ptr + start + written + rank, offs.to(tl.int32), mask=hit)
        written += tl.sum(hit.to(tl.int32))

    # Sentinel (== num_slots) on the tail this run padded, so the GEMM masks it.
    # At most BLOCK_SIZE - 1 slots, and exactly 0 when the expert drew nothing.
    offs_pad = tl.arange(0, BLOCK_PAD)
    tl.store(
        sorted_ids_ptr + start + my_count + offs_pad,
        num_slots,
        mask=offs_pad < my_padded - my_count,
    )

    # expert_ids[b] = expert owning row-block b, for this run's blocks only;
    # the runs partition [0, num_tokens_post_padded), so every readable block
    # gets written exactly once.
    num_blocks = my_padded // BLOCK_SIZE
    first_block = start // BLOCK_SIZE
    for b in range(0, num_blocks, BLOCK_PAD):
        offs_b = b + tl.arange(0, BLOCK_PAD)
        tl.store(expert_ids_ptr + first_block + offs_b, e, mask=offs_b < num_blocks)


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort token slots by expert and pad each expert's run to ``block_size``.

    Args:
        topk_ids: ``[num_tokens, top_k]`` expert indices.
        block_size: GEMM row-block the kernel iterates over.
        num_experts: Total expert count ``E``.

    Returns:
        ``(sorted_token_ids, expert_ids, num_tokens_post_padded)``; see the
        module docstring for the exact protocol.
    """
    device = topk_ids.device
    flat_experts = topk_ids.reshape(-1)
    if flat_experts.dtype != torch.int32:
        flat_experts = flat_experts.to(torch.int32)
    flat_experts = flat_experts.contiguous()
    num_slots = flat_experts.numel()

    # Only experts that actually appear can waste padding, and at most
    # min(E, num_slots) of them do -- one slot cannot open two runs. The loose
    # `num_slots + E * (block_size - 1)` bound this used to carry is the same
    # number for prefill but wildly wrong at decode: 8 slots over 128 experts
    # sized the buffer at 1928 slots instead of 128, and `_invoke_moe_gemm`
    # takes its grid from that length, so 121 of every 129 row-blocks existed
    # only to load `num_tokens_post_padded` and return.
    max_active = min(num_experts, num_slots)
    max_padded = num_slots + max_active * (block_size - 1)
    max_num_blocks = (max_padded + block_size - 1) // block_size

    # Untouched past num_tokens_post_padded, which is fine: the GEMM tests that
    # bound before it reads either buffer, so the tail is unreachable rather
    # than merely unused. `torch.full` would cost a launch to prove the same.
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(max_num_blocks, dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)

    block_s = min(triton.next_power_of_2(num_slots), _ALIGN_BLOCK_S)
    fused_count = num_slots <= block_s
    counts = None
    if not fused_count:
        counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        _moe_align_count_kernel[(triton.cdiv(num_slots, block_s),)](
            flat_experts, counts, num_slots, NUM_EXPERTS=num_experts, BLOCK_S=block_s, num_warps=4
        )
    _moe_align_scatter_kernel[(num_experts,)](
        flat_experts,
        counts,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_slots,
        BLOCK_SIZE=block_size,
        NUM_EXPERTS=num_experts,
        BLOCK_E=triton.next_power_of_2(num_experts),
        BLOCK_PAD=triton.next_power_of_2(block_size),
        BLOCK_S=block_s,
        FUSED_COUNT=fused_count,
        num_warps=4,
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


# --------------------------------------------------------------------------- #
# Grouped GEMM
# --------------------------------------------------------------------------- #
@triton.jit
def _fused_moe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    b_zeros_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_slots,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsn,
    stride_bsk,
    GROUP_N: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    top_k: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    NATIVE_FP8: tl.constexpr,
    FP8_CVT: tl.constexpr,
    K_PROMOTE: tl.constexpr,
    DEQUANT_SCALE: tl.constexpr,
    HAS_ZEROS: tl.constexpr,
    SCALE_HOISTED: tl.constexpr,
    EVEN_K: tl.constexpr,
    compute_type: tl.constexpr,
    A_QUANT: tl.constexpr,
    A_QMAX: tl.constexpr,
):
    """One C row-block of ``A @ B[expert]`` where rows of A are gathered tokens.

    A: ``[num_tokens, K]`` activations. B: ``[E, N, K]`` stacked expert weights,
    fp16 or 8-bit or int4 packed. C: ``[num_tokens * top_k, N]`` (each token's per-slot output row).
    When ``QUANT_MODE`` is non-zero, ``b_scale_ptr`` holds dequantisation scales.
    When ``QUANT_MODE`` is 3 (INT4), B is ``[E, N, K//2]`` uint8 (2 nibbles per
    byte, low nibble = lower K, the layout :func:`repack_int4_experts` leaves),
    ``b_scale_ptr`` is ``[E, N, K//group_k]``, and ``b_zeros_ptr`` optionally
    holds zero points. When ``QUANT_MODE`` is 4 (MXFP4), B is ``[E, N, K//8]``
    int32 (8 e2m1 nibbles per word) and ``b_scale_ptr`` is the per-32 e8m0 scale
    ``[E, N, K//32]``; MXFP4 carries no zero point.
    When ``QUANT_MODE`` is 5 or 6 (fp8 / int8 W8A8), A is 8-bit too and
    ``a_scale_ptr`` holds one fp32 scale per A row; ``NATIVE_FP8`` then picks
    between keeping both operands 8-bit for the sm89+ fp8 MMA and the pre-sm89
    widening. Both are read only in those modes, and unused elsewhere.

    ``A_QUANT`` moves the mode-5/6 activation quantisation *inside* this kernel:
    A arrives at full precision (fp16/bf16), one extra pass over the gathered rows
    derives each row's scale, and the k-loop quantises on the fly instead of
    reading pre-quantised bytes. ``a_scale_ptr`` is then unread. This exists for
    the launch-bound decode shapes, where the separate quantiser kernel's host
    time exceeds the whole GEMM's device time (measured on H100 at 1 token:
    ~35 us of host per call against ~15 us of device work in both GEMMs).
    """
    # Grouped pid ordering for L2 reuse (same scheme as vLLM/triton matmul).
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_slots
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak

    if QUANT_MODE == 3:
        # INT4: B is [E, N, K//2] uint8, two nibbles per byte (low nibble = lower
        # K), the layout ``repack_int4_experts`` leaves. Replicated addressing --
        # each byte is loaded for both its nibble rows (``offs_k // 2``) -- makes
        # the k-tile [BLOCK_K, BLOCK_N] like the 8-bit paths, so the in-loop unpack
        # is one shift-and-mask with no reshape; the 2x load is an L1 hit (see
        # ``_INT4_PACK_FACTOR``).
        offs_kb = offs_k // 2
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[None, :] * stride_bn
            + offs_kb[:, None] * stride_bk
        )
    elif QUANT_MODE == 4:
        # MXFP4: B is [E, N, K//8] int32, eight nibbles per word. stride_bk is
        # per-word; each k-tile of BLOCK_K logical elements loads BLOCK_K//8 words.
        offs_k_words = tl.arange(0, BLOCK_K // 8)
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_bn[None, :] * stride_bn
            + offs_k_words[:, None] * stride_bk
        )
    else:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
    if QUANT_MODE != 0:
        # Scale row is fixed for this tile; only the k-block index advances.
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + (offs_bn // GROUP_N) * stride_bsn
        if SCALE_HOISTED:
            # One group covers K, so this is the whole tile's scale: read it here
            # and leave the k-loop free to accumulate in the dot.
            b_scale = tl.load(b_scale_ptrs)

    if A_QUANT:
        # Row amax over the gathered A rows, the same rows the k-loop reads.
        # Grid-wide this pass re-reads A once per n-block, which is why the
        # launcher only sets A_QUANT on small token counts: at 1 token the
        # whole activation is a few KB, at 4096 it is 16 MB the GEMM would
        # otherwise touch once.
        amax = tl.zeros((BLOCK_M,), dtype=tl.float32)
        a_qrow_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            offs_kq = k * BLOCK_K + offs_k
            a_row = tl.load(
                a_qrow_ptrs + offs_kq[None, :],
                mask=token_mask[:, None] & (offs_kq[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(a_row), axis=1))
        # Same scale convention as the separate quantisers: amax / QMAX, and
        # 1.0 on an all-zero row so it stays zero instead of dividing by it.
        a_scale = tl.where(amax > 0.0, amax / A_QMAX, 1.0)

    # fp32 accumulation keeps the K-loop noise below the fp16 storage floor.
    # Mode 6 hoisted is the one int8 exception: the integer tensor cores
    # accumulate in int32, which is exact, and converting every k-tile to fp32
    # before the epilogue would lose that.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_int = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if A_QUANT:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
                other=0.0,
            ).to(tl.float32)
            q = a / a_scale[:, None]
            if QUANT_MODE == 5:
                a = tl.minimum(tl.maximum(q, -A_QMAX), A_QMAX).to(tl.float8e4nv)
            else:
                # rint, not a plain .to(int8): the torch reference rounds to
                # nearest even, and .to truncates toward zero -- a different
                # byte for every value whose fraction is above one half.
                r = tl.extra.cuda.libdevice.rint(q)
                a = tl.minimum(tl.maximum(r, -A_QMAX), A_QMAX).to(tl.int8)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K),
                # An int literal, not 0.0: in modes 5/6 this pointer is ``uint8``
                # or ``int8``, and the e4m3 / int8 bit pattern 0 is +0 anyway,
                # so one spelling serves every dtype the pointer can carry.
                other=0,
            )
        if QUANT_MODE == 3:
            # INT4 path: one dense [BLOCK_K // 2, BLOCK_N] byte tile; the two
            # nibble planes come out in registers (see the addressing note
            # above the loop). The masked form predicates per element along k,
            # which decomposes the load into scalar bytes -- the same reason
            # replicated addressing was slow -- so it is compiled out whenever
            # K is tile-aligned, which both Qwen3-30B-A3B GEMMs are.
            if EVEN_K:
                b_byte = tl.load(b_ptrs)
            else:
                rem = K - k * BLOCK_K
                b_byte = tl.load(
                    b_ptrs,
                    # ceil: the low plane covers k = 2i and the high one
                    # k = 2i + 1, so an odd remainder leaves one dead high
                    # nibble -- whose A column the a-load's own mask has
                    # already zeroed.
                    mask=offs_kb[:, None] < (rem + 1) // 2,
                    other=0,
                )
            # Straight to compute_type: both planes are small integers and the
            # zero point is an integer in [0, 15], so the difference is exact
            # in the 16-bit formats -- no fp32 detour, and the epilogue's fp32
            # scale multiply is where the precision budget belongs anyway.
            b_lo = (b_byte & 0xF).to(compute_type)
            b_hi = ((b_byte >> 4) & 0xF).to(compute_type)
            # Load scale and optionally zero point
            if not SCALE_HOISTED:
                b_scale = tl.load(b_scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_bsk)
            if HAS_ZEROS:
                # The zero point joins its plane in compute_type: an fp32 zero
                # would promote the subtraction and drag the operand back out
                # of the dtype the dot needs.
                b_zero = tl.load(
                    b_zeros_ptr
                    + off_experts * stride_bse
                    + (offs_bn // GROUP_N) * stride_bsn
                    + ((k * BLOCK_K) // GROUP_K) * stride_bsk
                ).to(compute_type)
                b_lo = b_lo - b_zero[None, :]
                b_hi = b_hi - b_zero[None, :]
            # (m, k) with k = 2i + j reshapes row-major to
            # [BLOCK_M, BLOCK_K // 2, 2], so tl.split hands out the even and
            # odd k columns with no memory round trip; each half multiplies
            # its own nibble plane and the pair sums to the old full-K dot.
            a_even, a_odd = tl.split(tl.reshape(a, (BLOCK_M, BLOCK_K // 2, 2)))
            if SCALE_HOISTED:
                accumulator = tl.dot(a_even, b_lo, acc=accumulator)
                accumulator = tl.dot(a_odd, b_hi, acc=accumulator)
            else:
                # Both planes share the k-block's scale (GROUP_K is a multiple
                # of BLOCK_K -- the group_k check in _fused_moe guarantees it),
                # so one multiply covers the two half-K dots together.
                accumulator += (tl.dot(a_even, b_lo) + tl.dot(a_odd, b_hi)) * b_scale[None, :]
            # ``_INT4_PACK_FACTOR`` spelled as its literal: Triton kernels only
            # resolve globals that are tl.constexpr instances (see the modes'
            # note above), so the kernel body cannot name the launcher's constant.
            b_ptrs += (BLOCK_K // 2) * stride_bk
        elif QUANT_MODE == 4:
            # MXFP4 path: load int32 words [BLOCK_K//8, BLOCK_N], unpack to
            # [BLOCK_K, BLOCK_N] e2m1 code points (low nibble = lower K position).
            b_packed = tl.load(
                b_ptrs,
                mask=offs_k_words[:, None] < (K // 8) - k * (BLOCK_K // 8),
                other=0,
            )  # [BLOCK_K//8, BLOCK_N]
            # Unpack 8 nibbles per word: shift by [0,4,8,...,28] and mask 0xF
            shifts = (tl.arange(0, 8) * 4).to(tl.int32)  # [8]
            # Reshape for broadcast: [BLOCK_K//8, 1, BLOCK_N] x [1, 8, 1] -> [BLOCK_K//8, 8, BLOCK_N]
            b_expanded = (b_packed[:, None, :] >> shifts[None, :, None]) & 0xF
            # Flatten to [BLOCK_K, BLOCK_N]
            nibbles = tl.reshape(b_expanded, (BLOCK_K, BLOCK_N))
            # e2m1: bit3 sign, bits[2:1] exponent, bit0 mantissa.
            # Normal: (1 + m/2) * 2^(e-1); e==0: m*0.5 subnormal, 0 stays 0.
            # The scale/decode arithmetic stays fp32; only the product the
            # tensor cores consume comes back to the activation's dtype.
            sign = ((nibbles >> 3) & 1).to(tl.float32)
            e = (nibbles >> 1) & 3
            m = (nibbles & 1).to(tl.float32)
            mag = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
            val = tl.where(sign == 1, -mag, mag)
            # GROUP_K (32) is narrower than BLOCK_K, so one k-tile spans
            # BLOCK_K // GROUP_K scale groups: load one scale per group
            # and broadcast it over the group's k rows, instead of the
            # per-tile scalar the wider-group formats use.
            scale_row = k * (BLOCK_K // GROUP_K) + tl.arange(0, BLOCK_K // GROUP_K)
            s = tl.load(
                b_scale_ptrs[None, :] + scale_row[:, None] * stride_bsk,
                mask=(scale_row < K // GROUP_K)[:, None],
                other=0.0,
            )  # [BLOCK_K // GROUP_K, BLOCK_N]
            val_g = tl.reshape(val, (BLOCK_K // GROUP_K, GROUP_K, BLOCK_N))
            b = tl.reshape(val_g * s[:, None, :], (BLOCK_K, BLOCK_N)).to(a.dtype)
            accumulator += tl.dot(a, b)
            b_ptrs += (BLOCK_K // 2 if QUANT_MODE == 3 else BLOCK_K // 8) * stride_bk
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
            if QUANT_MODE == 0:
                accumulator = tl.dot(a, b, acc=accumulator)
            else:
                # Widened to the activation's dtype, not to a fixed fp16:
                # ``tl.dot`` rejects mixed operand types, and a bf16 model would
                # otherwise fail to compile every quantised expert GEMM. The
                # e4m3 bit trick lands in fp16 by construction and its values
                # carry 4 significant bits, so the extra hop to bf16 is exact.
                if QUANT_MODE == 5:
                    # W8A8 fp8: nothing is widened to compute_type at all. Both
                    # operands stay 8-bit into the dot, and Triton decides from
                    # BLOCK_M whether that becomes an fp8 wgmma or an fp16
                    # mma.sync. B is already loaded k-contiguous ([E, N, K] means
                    # stride_bk == 1), the layout the fp8 MMA wants, so no
                    # transpose is needed here -- unlike the dense
                    # ``fp8_matmul``, whose B arrives as [N, K].
                    if NATIVE_FP8:
                        a = a.to(tl.float8e4nv, bitcast=True)
                        b = b.to(tl.float8e4nv, bitcast=True)
                    else:
                        a = dequant_fp8e4m3(a)
                        b = dequant_fp8e4m3(b)
                elif QUANT_MODE == 6:
                    # W8A8 int8: both operands stay int8 into the dot and the
                    # integer tensor cores accumulate in int32, which is exact.
                    # No capability split: the imma path exists from Turing on,
                    # and pre-Turing devices have no tensor cores for the other
                    # modes either.
                    pass
                elif QUANT_MODE == 1:
                    # sm89+ widens e4m3 with one hardware cvt per element; the bit
                    # trick below is the five-integer-op software path for anything
                    # older. Both land in ``compute_type`` with the same rounding
                    # (e4m3 is exact in fp16, so the trick's 256x must be folded
                    # into DEQUANT_SCALE by the launcher -- the cvt needs no
                    # correction), which is what keeps the golden bytes stable
                    # across the capability split.
                    if FP8_CVT:
                        b = b.to(tl.float8e4nv, bitcast=True).to(compute_type)
                    else:
                        b = dequant_fp8e4m3(b).to(compute_type)
                else:
                    b = b.to(compute_type)
                    if HAS_ZEROS:
                        # Asymmetric int8 (GPTQ bits=8): the integer zero point
                        # shifts the int8 codes, so it is subtracted before the
                        # dot, mirroring the int4 path. Both the int8 byte and
                        # the zero point are integers exact in compute_type, so
                        # the difference is too. Loaded per k-tile like the int4
                        # zero point: under SCALE_HOISTED (one group spanning K)
                        # it is the same value every iteration, which is correct.
                        b_zero = tl.load(
                            b_zeros_ptr
                            + off_experts * stride_bse
                            + (offs_bn // GROUP_N) * stride_bsn
                            + ((k * BLOCK_K) // GROUP_K) * stride_bsk
                        ).to(compute_type)
                        b = b - b_zero[None, :]
                # Hoisted out of the dot, so the tensor cores see two 16-bit
                # operands instead of the fp32 scale's type. With one scale group
                # over K it leaves the loop entirely: the dot then accumulates in
                # place (``acc=``) instead of producing a tile that a separate
                # fp32 multiply-add has to fold in once per k-tile.
                if not SCALE_HOISTED:
                    b_scale = tl.load(b_scale_ptrs + ((k * BLOCK_K) // GROUP_K) * stride_bsk)
                if QUANT_MODE == 5:
                    # Hopper's fp8 wgmma accumulates at reduced precision inside
                    # the instruction, and Triton's sm90 default lets that run
                    # unbounded. ``K_PROMOTE`` caps how many k elements may
                    # accumulate before the result is promoted into a real fp32
                    # accumulator; 0 makes Triton drop wgmma entirely and widen
                    # both operands to fp16 instead. See ``_FP8_A8_PROMOTE_EVERY``.
                    if SCALE_HOISTED:
                        accumulator = tl.dot(a, b, acc=accumulator, max_num_imprecise_acc=K_PROMOTE)
                    else:
                        accumulator += (
                            tl.dot(a, b, max_num_imprecise_acc=K_PROMOTE) * b_scale[None, :]
                        )
                elif QUANT_MODE == 6:
                    if SCALE_HOISTED:
                        # out_dtype is spelled, not defaulted: the default is
                        # fp32, and tl.dot asserts the accumulator's dtype equals
                        # it -- which an int32 accumulator never will.
                        acc_int = tl.dot(a, b, acc=acc_int, out_dtype=tl.int32)
                    else:
                        accumulator += tl.dot(a, b).to(tl.float32) * b_scale[None, :]
                elif SCALE_HOISTED:
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b) * b_scale[None, :]
            b_ptrs += BLOCK_K * stride_bk
        a_ptrs += BLOCK_K * stride_ak

    if QUANT_MODE == 6 and SCALE_HOISTED:
        accumulator = acc_int.to(tl.float32)

    if QUANT_MODE != 0 and SCALE_HOISTED:
        # The whole k-loop shared this scale, so one multiply here replaces the
        # cdiv(K, BLOCK_K) the in-loop form would have done. The ``QUANT_MODE``
        # half of the guard mirrors where ``b_scale`` is defined: mode 0 has no
        # scale tensor at all, so hoisting is meaningless rather than free there.
        accumulator *= b_scale[None, :]
    accumulator *= DEQUANT_SCALE
    if QUANT_MODE == 5 or QUANT_MODE == 6:
        if A_QUANT:
            # The scale is a register this kernel derived; multiplying here is
            # one instruction and no memory traffic. ``a_scale_ptr`` is unused.
            accumulator *= a_scale[:, None]
        else:
            # One scale per activation row, so it leaves the k-loop untouched. The
            # mask matters: a padded slot's ``offs_token`` is the sentinel
            # ``num_valid_slots``, one past the last real row of a_scale.
            a_scale = tl.load(a_scale_ptr + offs_token // top_k, mask=token_mask, other=0.0)
            accumulator *= a_scale[:, None]
    # Router weights multiply in fp32 before the final downcast (gemm2 only).
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


#: Tile per (quant mode, rows-per-expert tier): ``(BLOCK_M, BLOCK_N, num_warps)``.
#: ``BLOCK_K`` is 128 for every mode and ``GROUP_M``/``num_stages`` are 8/3, so
#: those live in :func:`_launch_config` and only the axes that actually move are
#: tabulated. See that function for the sweep behind every entry.
_TILE_TABLE: dict[int, tuple[tuple[int, int, int], ...]] = {
    _QUANT_NONE: ((16, 64, 4), (64, 64, 4), (128, 128, 8), (128, 128, 8)),
    _QUANT_FP8: ((16, 64, 4), (64, 128, 4), (64, 128, 4), (128, 256, 8)),
    _QUANT_INT8: ((16, 64, 4), (32, 128, 4), (64, 128, 4), (128, 256, 8)),
    _QUANT_INT4: ((16, 128, 4), (64, 128, 4), (64, 128, 4), (64, 128, 4)),
    # MXFP4 inherits int4's tiles: same 4-bit packed B bytes, same nibble-plane
    # epilogue, and its 32-element scale groups are read per k-tile either way.
    _QUANT_MXFP4: ((16, 128, 4), (64, 128, 4), (64, 128, 4), (64, 128, 4)),
    _QUANT_FP8_A8: ((16, 64, 4), (32, 128, 4), (64, 128, 4), (64, 128, 4)),
    _QUANT_INT8_A8: ((16, 64, 4), (32, 128, 4), (64, 128, 4), (128, 256, 8)),
}

#: Pre-Hopper fallback, shared by every format. NOT measured — the sm90 table
#: above is sized against H100's 228 KB shared memory (its unquantised 128x128
#: tier needs 288 KB over three stages and only fits because Hopper carries
#: that budget); sm86-class parts have 100 KB, so the wide tiers would spill or
#: fail to compile outright. These tiles keep two stages of (64, 128) bf16
#: operands inside ~64 KB. The intended path for any hardware the sweep never
#: ran on is the autotune store, whose per-GPU entry takes precedence.
_TILE_TABLE_PRE_HOPPER: tuple[tuple[int, int, int], ...] = (
    (16, 64, 4),
    (32, 64, 4),
    (64, 64, 4),
    (64, 64, 4),
)


def _launch_config(
    num_tokens: int, quant_mode: int, rows_per_expert: float, device_index: int | None
) -> dict:
    """Heuristic tile config, used whenever the autotune store has no entry.

    ``BLOCK_M`` must be identical for both GEMMs because they share one alignment.

    ``BLOCK_M`` is chosen from ``rows_per_expert`` (``num_tokens * top_k / E``),
    not from ``num_tokens``, because that is the quantity the tile trades against:
    an expert holding fewer rows than ``BLOCK_M`` pads the rest away and does the
    MMA on them anyway, while an expert holding more spills into a second
    row-block that re-reads its whole weight tile. Token count alone cannot see
    either -- 64 tokens is 4 rows per expert at Qwen3-30B-A3B's 128 experts and
    16 at Mixtral's 8, and the old tiers read both as "32".

    Tiles are per format *and* per tier, from a sweep of BLOCK_M x BLOCK_N over
    the Qwen3-30B-A3B expert geometry on an H100 (``RAPID_LLM_AUTOTUNE=0``,
    BLOCK_K=128 throughout). Entry is ``(BLOCK_M, BLOCK_N, num_warps)``;
    ``rows`` tiers are ``<16``, ``16..32``, ``32..64`` and ``>64`` rows per
    expert:

    ============  ==========  ===========  ===========  ===========
    format        <16         16..32       32..64       >64
    ============  ==========  ===========  ===========  ===========
    unquantised   16x64  w4   64x64  w4    128x128 w8   128x128 w8
    fp8 W8A16     16x64  w4   64x128 w4    64x128  w4   128x256 w8
    int8 W8A16    16x64  w4   32x128 w4    64x128  w4   128x256 w8
    int4          16x128 w4   64x128 w4    64x128  w4   64x128  w4
    fp8 W8A8      16x64  w4   32x128 w4    64x128  w4   64x128  w4
    int8 W8A8     16x64  w4   32x128 w4    64x128  w4   128x256 w8
    ============  ==========  ===========  ===========  ===========

    At the 4096-token prefill those tiles bought 19-28% over the old two-tier
    heuristic (all in us): unquantised 1430 -> 1058, fp8 W8A8 1073 -> 869 (with
    ``_FP8_A8_PROMOTE_EVERY=128``, without which the 64x128 tile loses ~12%),
    fp8 W8A16 1949 -> 1418, int8 W8A16 1585 -> 1148. The int4 row was reswept
    after the dual-dot kernel replaced the int32-format unpacking (that
    sweep's 2366 -> 1834 is void): the tiles survived, and BLOCK_K=256 (four
    half-K dots per k-iteration) and BLOCK_N=256 both measured 1.6-2x slower
    at t4096. The int8 W8A8 row was written after the sweep and inherits int8
    W8A16's tiles: its B bytes are identical and the int8 tensor cores only
    ever compute faster than the widened dot, so the memory-bound tile
    preference transfers.

    ``BLOCK_N`` stops at 128 for the unquantised mode by shared memory, not by
    preference: a bf16 tile of (128, 128) + (128, 256) operands needs 288 KB over
    three stages against H100's 228 KB. The 8-bit modes store B as one byte per
    element and fit the wider tile; int4 and fp8 W8A8 measured *slower* on it --
    int4's nibble planes are (BLOCK_K, BLOCK_N) compute_type tiles held in
    registers, and mode 5's epilogue wants the narrower tile -- so only the two
    weight-only 8-bit modes take it.

    ``num_tokens`` and ``quant_mode`` stay in the signature because callers pass
    them and a future divergence on either belongs here rather than at the call
    sites.
    """
    if rows_per_expert < 16:
        tier = 0
    elif rows_per_expert <= 32:
        tier = 1
    elif rows_per_expert <= 64:
        tier = 2
    else:
        tier = 3
    if tile_tier(device_index) is TileTier.PRE_HOPPER:
        # Conservative pre-Hopper tiles (unmeasured; autotune overrides).
        block_m, block_n, num_warps = _TILE_TABLE_PRE_HOPPER[tier]
        return {
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": _QUANT_BLOCK_K,
            "GROUP_M": 8,
            "num_warps": num_warps,
            "num_stages": 2,
        }
    block_m, block_n, num_warps = _TILE_TABLE[quant_mode][tier]
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": _QUANT_BLOCK_K,
        "GROUP_M": 8,
        "num_warps": num_warps,
        "num_stages": 3,
    }


def _invoke_moe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scale: torch.Tensor | None,
    b_scale: torch.Tensor | None,
    b_zeros: torch.Tensor | None,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
    quant_mode: int,
    group_n: int,
    group_k: int,
    config: dict,
    a_quant: str | None = None,
) -> None:
    """Launch one grouped GEMM: ``C[slot] = A[slot // top_k] @ B[expert].T``.

    ``a_quant`` (``"fp8"`` / ``"int8"``) moves the per-row activation
    quantisation inside the kernel — ``a`` then arrives at full precision and
    ``a_scale`` is ignored — for the launch-bound shapes; see
    ``_INLINE_A_QUANT_MAX_ROWS``.
    """
    assert a.stride(-1) == 1 and b.stride(-1) == 1, "last dims must be contiguous"
    em = sorted_token_ids.numel()
    num_slots = c.shape[0]
    n, k = b.shape[1], b.shape[2]
    # Only the A8 mode puts A through the tensor cores as fp8, so only it cares
    # which of the two widenings the kernel compiles; the weight-only modes
    # always take the bit trick and always pay its single correction factor.
    native_fp8 = quant_mode == _QUANT_FP8_A8 and has_native_fp8(a.device.index)
    # Mode 1's widening has its own capability split: sm89+ converts e4m3 with a
    # hardware cvt (one instruction, no 256x correction), older devices keep the
    # bit trick and its DEQUANT_SCALE.
    fp8_cvt = quant_mode == _QUANT_FP8 and has_native_fp8(a.device.index)
    if quant_mode == _QUANT_FP8_A8:
        dequant_scale = 1.0 if native_fp8 else _FP8_BIT_TRICK_SCALE_SQ
    elif quant_mode == _QUANT_FP8:
        dequant_scale = 1.0 if fp8_cvt else FP8_E4M3_BIT_TRICK_SCALE
    else:
        dequant_scale = 1.0
    # INT4 packs 2 nibbles per uint8 byte ([E, N, K//2]); MXFP4 packs 8 e2m1
    # nibbles per int32 word ([E, N, K//8]). Both store the packed last dim, so
    # the logical K the k-loop reduces over is that dim times the mode's own
    # pack factor -- sharing one factor here under-counted MXFP4's K by 4x and
    # left three quarters of the reduction unaccumulated.
    if quant_mode == _QUANT_INT4:
        k_logical = k * _INT4_PACK_FACTOR
    elif quant_mode == _QUANT_MXFP4:
        k_logical = k * _MXFP4_PACK_FACTOR
    else:
        k_logical = k
    grid = (triton.cdiv(em, config["BLOCK_M"]) * triton.cdiv(n, config["BLOCK_N"]),)
    group_k_eff = min(group_k, k_logical) if group_k else 1
    _fused_moe_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        b_zeros,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k_logical,
        em,
        num_slots,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        *(b_scale.stride() if b_scale is not None else (0, 0, 0)),
        GROUP_N=group_n or 1,
        GROUP_K=group_k_eff,
        BLOCK_M=config["BLOCK_M"],
        BLOCK_N=config["BLOCK_N"],
        BLOCK_K=config["BLOCK_K"],
        GROUP_M=config["GROUP_M"],
        top_k=top_k,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        QUANT_MODE=quant_mode,
        NATIVE_FP8=native_fp8,
        FP8_CVT=fp8_cvt,
        K_PROMOTE=min(_FP8_A8_PROMOTE_EVERY, config["BLOCK_K"]),
        DEQUANT_SCALE=dequant_scale,
        HAS_ZEROS=b_zeros is not None,
        EVEN_K=k_logical % config["BLOCK_K"] == 0,
        # Per-output-channel scales (one group spanning K) are the common case for
        # every 8-bit expert format here, and they let the k-loop accumulate
        # inside ``tl.dot``. The int4 and MXFP4 modes are excluded: their branch
        # folds the group scale (and zero point) into ``b`` *inside* the loop
        # unconditionally, so hoisting would multiply it a second time in the
        # epilogue -- which collapsed the down GEMM (group_k covers intermediate,
        # so it hoisted) to ~scale^2 of the right answer. Block-wise fp8 keeps the
        # in-loop form too because its scale genuinely changes with k.
        SCALE_HOISTED=(
            quant_mode not in (_QUANT_NONE, _QUANT_INT4, _QUANT_MXFP4) and group_k_eff >= k_logical
        ),
        compute_type=torch_to_triton_dtype[c.dtype],
        # A_QUANT quantises the activation inside the kernel (see
        # ``_INLINE_A_QUANT_MAX_ROWS``); A_QMAX is the target format's range and
        # only read when A_QUANT is set.
        A_QUANT=a_quant is not None,
        A_QMAX={"fp8": FP8_E4M3_MAX, "int8": _INT8_MAX}.get(a_quant, 0.0),
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


# --------------------------------------------------------------------------- #
# Activation and top-k reduction
# --------------------------------------------------------------------------- #
@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,
    stride_xm,
    N,
    LIMIT: tl.constexpr,
    QUANT_OUT: tl.constexpr,
    QMAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``out = silu(x[:, :N]) * x[:, N:]`` on a contiguous ``[tokens, 2N]`` input.

    ``LIMIT`` clamps the gate at ``+LIMIT`` and up at ``+-LIMIT`` before the
    activation — DeepSeek-V4's ``swiglu_limit`` (gpt-oss semantics). ``inf``
    (the default every other family passes) makes both clamps no-ops.
    """
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    # silu evaluates its sigmoid in fp32, matching the dense swiglu kernel.
    gate = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + pid_m * stride_xm + N + offs, mask=mask, other=0.0).to(tl.float32)
    if float("inf") > LIMIT:
        gate = tl.minimum(gate, LIMIT)
        up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
    out = silu(gate) * up
    if QUANT_OUT:
        # Masked lanes carry 0 from the loads above, so they cannot raise the
        # amax. Same scale convention as the per-token quantisers: exactly
        # ``amax / QMAX``, and 1.0 on an all-zero row so it stays zero.
        amax = tl.max(tl.abs(out))
        scale = tl.where(amax > 0.0, amax / QMAX, 1.0)
        tl.store(scale_ptr + pid_m, scale)
        q = tl.minimum(tl.maximum(out / scale, -QMAX), QMAX)
        if QUANT_OUT == 1:
            tl.store(
                out_ptr + pid_m * N + offs,
                q.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
                mask=mask,
            )
        else:
            # rint, not a plain .to(int8): torch's .round() is round-to-nearest
            # even and .to truncates toward zero -- a different byte for every
            # value whose fraction is above one half.
            r = tl.extra.cuda.libdevice.rint(q)
            tl.store(out_ptr + pid_m * N + offs, r.to(tl.int8), mask=mask)
    else:
        tl.store(out_ptr + pid_m * N + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _moe_sum_kernel(
    input_ptr,
    output_ptr,
    N,
    top_k: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """``output[m] = sum_k input[m * top_k + k]`` over the expanded slot dim."""
    pid_m = tl.program_id(0).to(tl.int64)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    base = input_ptr + pid_m * top_k * N
    for i in tl.static_range(top_k):
        acc += tl.load(base + i * N + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + pid_m * N + offs, acc.to(output_ptr.dtype.element_ty), mask=mask)


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #
def _quant_mode(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    zeros: torch.Tensor | None = None,
    mxfp4: bool = False,
) -> int:
    """Classify an expert weight tensor into one of the ``QUANT_MODE`` values.

    Never returns the W8A8 modes: the dtype cannot tell weight-only fp8 from W8A8
    fp8 (both store ``uint8`` e4m3 experts) nor weight-only int8 from W8A8 int8
    (both ``int8``). Only the entry point the caller chose says which, so
    :func:`fused_moe_w8a8_fp8` and :func:`fused_moe_w8a8_int8` promote the mode
    after this classification.

    ``zeros`` disambiguates the other dtype collision: byte-packed int4 experts
    are ``uint8`` just like fp8, and both int4 and asymmetric int8 (GPTQ
    ``bits=8``) carry zero points, so ``zeros`` plus the dtype picks between
    them. ``mxfp4`` is checked first because its nibbles ride the int32 word
    packing that linear int4 is explicitly *not* allowed to reach the kernel
    with. The int32 word packing that int4 and GPTQ-8 checkpoints ship never
    reaches the kernel -- converting it is a one-time load step
    (:func:`repack_int4_experts` / :func:`unpack_int8_experts`), not something a
    per-call path should pay.
    """
    if scale is None:
        return _QUANT_NONE
    if mxfp4:
        if weight.dtype != torch.int32:
            raise ValueError(f"mxfp4 expert weights must be int32 packed, got {weight.dtype}")
        return _QUANT_MXFP4
    if zeros is not None:
        if weight.dtype == torch.uint8:
            return _QUANT_INT4
        if weight.dtype == torch.int8:
            # Asymmetric int8 (GPTQ bits=8): one int8 byte per element plus an
            # integer zero point, dequantised as (byte - zero) * scale. The
            # kernel's HAS_ZEROS branch subtracts the zero point; symmetric int8
            # (zeros is None) takes the same mode without it.
            return _QUANT_INT8
        raise ValueError(
            "int4 expert weights must be byte-packed uint8 [E, N, K//2], got "
            f"{weight.dtype}; convert the [E, N, K//8] int32 checkpoint layout "
            "once at load with repack_int4_experts"
        )
    if weight.dtype == torch.uint8:
        return _QUANT_FP8
    if weight.dtype == torch.int8:
        return _QUANT_INT8
    if weight.dtype == torch.int32:
        raise ValueError(
            "int4 expert weights must be byte-packed uint8 [E, N, K//2]; the "
            "[E, N, K//8] int32 checkpoint layout needs repack_int4_experts "
            "(once, at load)"
        )
    raise ValueError(f"quantised expert weights must be uint8 or int8, got {weight.dtype}")


def _fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
    act_quant: str | None = None,
) -> torch.Tensor:
    """Body of the three public entry points; ``act_quant`` is their only
    difference: ``None`` leaves the activation at full precision, ``"fp8"`` and
    ``"int8"`` quantise it per token.

    Private, and the thin wrappers below are what callers and the registry see,
    because ``tests/ops/test_native_specs.py`` pins every registered target's
    parameter *names* to the :class:`MoeOp` ABC. A public ``act_quant`` flag
    would either break that pin or have to be added to the ABC, where it would
    become a promise the DeepGEMM row cannot keep. Three names for three rows
    costs less than one signature that lies.
    """
    num_tokens, hidden = hidden_states.shape
    num_experts, two_inter, _ = w1.shape
    intermediate = two_inter // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    quant_mode = _quant_mode(w1, w1_scale, w1_zeros)
    if quant_mode != _quant_mode(w2, w2_scale, w2_zeros):
        raise ValueError("w1 and w2 must use the same quantisation format")
    if quant_mode and group_k % 128 != 0 and group_k < min(hidden, intermediate):
        raise ValueError(f"group_k ({group_k}) must be a multiple of 128 unless it covers K")
    if act_quant == "fp8":
        if quant_mode != _QUANT_FP8:
            raise ValueError(
                "fp8 W8A8 experts must be uint8 e4m3 bytes with scales, got "
                f"{w1.dtype} (mode {quant_mode})"
            )
        quant_mode = _QUANT_FP8_A8
    elif act_quant == "int8":
        if quant_mode != _QUANT_INT8:
            raise ValueError(
                "int8 W8A8 experts must be int8 bytes with scales, got "
                f"{w1.dtype} (mode {quant_mode})"
            )
        if w1_zeros is not None:
            # W8A8 int8 is symmetric per-channel; the asymmetric zero points
            # belong to the weight-only int8 path (GPTQ bits=8), not here.
            raise ValueError(
                "int8 W8A8 experts are symmetric and carry no zero points; zero "
                "points belong to weight-only int8 (GPTQ bits=8) or int4"
            )
        quant_mode = _QUANT_INT8_A8
    elif act_quant is not None:
        raise ValueError(f"act_quant must be None, 'fp8' or 'int8', got {act_quant!r}")

    topk_ids = topk_ids.to(torch.int32)
    flat_weights = topk_weights.reshape(-1).to(dtype).contiguous()

    # Autotune lookup: use persisted best config if available, else heuristic.
    # The unquantised key follows the activation dtype — bf16 and fp16 compile
    # the same inner loop but are tuned as separate entries, and a bf16
    # checkpoint must not silently read fp16's tile table. The quantised keys
    # name the weight format, which is what pins the kernel's tiles.
    act_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"
    dtype_label = {
        _QUANT_NONE: act_dtype,
        _QUANT_FP8: "fp8",
        _QUANT_INT8: "int8",
        _QUANT_INT4: "int4",
        # Their own keys, not the weight-only names: the W8A8 modes compile
        # different inner loops and want different tiles, so sharing a TuneKey
        # would let one mode's search install a config the other never measured.
        _QUANT_FP8_A8: "fp8_a8",
        _QUANT_INT8_A8: "int8_a8",
    }.get(quant_mode, act_dtype)
    config = resolve_tiles(
        "fused_moe",
        m=num_tokens,
        n=two_inter,
        k=hidden,
        dtype_label=dtype_label,
        heuristic=lambda dev: _launch_config(
            num_tokens, quant_mode, num_tokens * top_k / num_experts, dev
        ),
        device_index=device.index,
    )
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, config["BLOCK_M"], num_experts
    )

    # GEMM1: [M, hidden] x [E, 2I, hidden] -> [M * top_k, 2I]. In W8A8 the
    # activation is quantised once per token, before the gather: a slot reads a
    # row, so quantising the [M, hidden] input costs top_k times less than
    # quantising the expanded slots would, and every slot of a token then shares
    # one scale. Below ``_INLINE_A_QUANT_MAX_ROWS`` tokens the quantisation moves
    # inside the GEMM kernel (``a1_quant``) and this stays a full-precision read;
    # either way there is no host synchronisation, so the layer stays
    # CUDA-graph capturable.
    a1, a1_scale = hidden_states, None
    a1_quant = None
    if quant_mode == _QUANT_FP8_A8:
        if num_tokens <= _INLINE_A_QUANT_MAX_ROWS:
            a1_quant = "fp8"
        else:
            a1, a1_scale = fp8_quantize_per_token(hidden_states)
            a1_scale = a1_scale.reshape(-1)
    elif quant_mode == _QUANT_INT8_A8:
        if num_tokens <= _INLINE_A_QUANT_MAX_ROWS:
            a1_quant = "int8"
        else:
            a1, a1_scale = int8_quantize_per_token(hidden_states)
    gate_up = torch.empty((num_tokens * top_k, two_inter), device=device, dtype=dtype)
    _invoke_moe_gemm(
        a1,
        w1,
        gate_up,
        a1_scale,
        w1_scale,
        w1_zeros,
        flat_weights,
        sorted_ids,
        expert_ids,
        num_post,
        top_k,
        mul_routed_weight=False,
        quant_mode=quant_mode,
        group_n=group_n,
        group_k=group_k,
        config=config,
        a_quant=a1_quant,
    )

    # silu(gate) * up -> [M * top_k, I], quantised on the way out when W8A8 can
    # fuse it (see ``_silu_and_mul_kernel``). Otherwise the wide-FFN case falls
    # back to inline quantisation in GEMM2 below, and prefill shapes to the
    # separate quantiser — a wide MoE keeps working either way, just with one
    # more launch.
    act_quant_out, act_qmax = 0, 0.0
    if quant_mode == _QUANT_FP8_A8:
        act_quant_out, act_qmax = 1, FP8_E4M3_MAX
    elif quant_mode == _QUANT_INT8_A8:
        act_quant_out, act_qmax = 2, _INT8_MAX
    block_n = min(triton.next_power_of_2(intermediate), 1024)
    fuse_act_quant = act_quant_out != 0 and block_n >= intermediate
    if fuse_act_quant:
        act_dtype = torch.uint8 if act_quant_out == 1 else torch.int8
    else:
        act_dtype = dtype
    act = torch.empty((num_tokens * top_k, intermediate), device=device, dtype=act_dtype)
    act_scale = (
        torch.empty(num_tokens * top_k, device=device, dtype=torch.float32)
        if fuse_act_quant
        else None
    )
    _silu_and_mul_kernel[(num_tokens * top_k, triton.cdiv(intermediate, block_n))](
        gate_up,
        act,
        act_scale,
        gate_up.stride(0),
        intermediate,
        LIMIT=float("inf"),
        QMAX=act_qmax,
        QUANT_OUT=act_quant_out if fuse_act_quant else 0,
        BLOCK_N=block_n,
        num_warps=4,
    )

    # GEMM2 with the routing weight folded in: [M * top_k, I] x [E, hidden, I].
    # ``act`` is already expanded per slot, so the kernel's A-row gather must use
    # top_k=1 (vLLM does the same: the second invocation passes ``1``), turning
    # ``offs_token // top_k`` into the identity on slot indices -- which is also
    # what makes one a_scale row per slot the right shape here.
    a2, a2_scale = act, act_scale
    a2_quant = None
    if act_quant_out and not fuse_act_quant:
        # GEMM2's A is the per-slot silu output, already expanded, so its row
        # count is ``num_tokens * top_k`` — that is what the inline threshold is
        # judged on here, not ``num_tokens``.
        if num_tokens * top_k <= _INLINE_A_QUANT_MAX_ROWS:
            a2_quant = "fp8" if act_quant_out == 1 else "int8"
        elif act_quant_out == 1:
            a2, a2_scale = fp8_quantize_per_token(act)
            a2_scale = a2_scale.reshape(-1)
        else:
            a2, a2_scale = int8_quantize_per_token(act)
    expanded = torch.empty((num_tokens * top_k, hidden), device=device, dtype=dtype)
    _invoke_moe_gemm(
        a2,
        w2,
        expanded,
        a2_scale,
        w2_scale,
        w2_zeros,
        flat_weights,
        sorted_ids,
        expert_ids,
        num_post,
        1,
        mul_routed_weight=True,
        quant_mode=quant_mode,
        group_n=group_n,
        group_k=group_k,
        config=config,
        a_quant=a2_quant,
    )

    # Reduce over the top_k slot dim -> [M, hidden]
    out = torch.empty((num_tokens, hidden), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(hidden), 1024)
    _moe_sum_kernel[(num_tokens, triton.cdiv(hidden, block_n))](
        expanded, out, hidden, top_k=top_k, BLOCK_N=block_n, num_warps=4
    )
    return out


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
    swiglu_limit: float = float("inf"),
    mxfp4: bool = False,
) -> torch.Tensor:
    """Run the routed-expert FFN: ``sum_k w_k * (silu(x @ W1g.T) * (x @ W1u.T)) @ W2.T``.

    Weight-only for every quantised format: the expert tiles are widened to the
    activation's dtype inside the loop and the activation itself is never
    touched. :func:`fused_moe_w8a8_fp8` and :func:`fused_moe_w8a8_int8` are the
    entry points that quantise it.

    Args:
        swiglu_limit: Clamp gate at ``+limit`` and up at ``+-limit`` before the
            activation (DeepSeek-V4's gpt-oss-style bounded SwiGLU). ``inf``
            keeps the plain silu every other family uses.
        hidden_states: ``[num_tokens, hidden]`` activations (fp16, contiguous rows).
        w1: ``[E, 2 * moe_intermediate, hidden]`` fused gate/up projections, fp16
            or 8-bit (``uint8`` fp8-e4m3 / ``int8``) or int4 byte-packed (``uint8``,
            two nibbles per byte -- what ``repack_int4_experts`` leaves).
        w2: ``[E, hidden, moe_intermediate]`` down projections, same dtype as ``w1``.
        topk_weights: ``[num_tokens, top_k]`` routing weights.
        topk_ids: ``[num_tokens, top_k]`` expert indices.
        w1_scale: Dequantisation scales; ``None`` selects the fp16 path.
        w2_scale: Scales for ``w2``.
        w1_zeros: Zero points for int4 (AWQ/GPTQ); ``None`` for symmetric.
        w2_zeros: Zero points for ``w2``; ``None`` for symmetric.
        group_n: Rows of one scale block (``1`` = per output channel).
        group_k: Columns of one scale block.
        mxfp4: ``True`` interprets the int32-packed nibbles as MXFP4 e2m1
            code points with per-``group_k``-element e8m0 scales (DeepSeek-V4
            routed experts; ``group_k`` is 32 there) instead of linear int4.

    Returns:
        ``[num_tokens, hidden]`` combined expert output, in ``hidden_states``' dtype.
    """
    num_tokens, hidden = hidden_states.shape
    num_experts, two_inter, _ = w1.shape
    intermediate = two_inter // 2
    top_k = topk_ids.shape[1]
    device = hidden_states.device
    dtype = hidden_states.dtype

    quant_mode = _quant_mode(w1, w1_scale, w1_zeros, mxfp4)
    if quant_mode != _quant_mode(w2, w2_scale, w2_zeros, mxfp4):
        raise ValueError("w1 and w2 must use the same quantisation format")
    if (
        quant_mode
        and quant_mode != _QUANT_MXFP4
        and group_k % 128 != 0
        and (group_k < min(hidden, intermediate))
    ):
        raise ValueError(f"group_k ({group_k}) must be a multiple of 128 unless it covers K")
    if quant_mode == _QUANT_MXFP4 and group_k != 32:
        raise ValueError(f"mxfp4 scale groups are 32 elements wide, got group_k={group_k}")

    topk_ids = topk_ids.to(torch.int32)
    flat_weights = topk_weights.reshape(-1).to(dtype).contiguous()

    # Autotune lookup: use persisted best config if available, else heuristic.
    from ...dispatcher.autotune import get_best_config

    # The unquantised key follows the activation dtype — bf16 and fp16 compile
    # the same inner loop but are tuned as separate entries, and a bf16
    # checkpoint must not silently read fp16's tile table.
    act_dtype = "bf16" if dtype == torch.bfloat16 else "fp16"
    dtype_label = {
        _QUANT_NONE: act_dtype,
        _QUANT_FP8: "fp8",
        _QUANT_INT8: "int8",
        _QUANT_INT4: "int4",
        _QUANT_MXFP4: "mxfp4",
    }.get(quant_mode, act_dtype)
    config = get_best_config("fused_moe", m=num_tokens, n=two_inter, k=hidden, dtype=dtype_label)
    if config is None:
        config = _launch_config(
            num_tokens, quant_mode, num_tokens * top_k / num_experts, device.index
        )
    sorted_ids, expert_ids, num_post = moe_align_block_size(
        topk_ids, config["BLOCK_M"], num_experts
    )

    # GEMM1: [M, hidden] x [E, 2I, hidden] -> [M * top_k, 2I]
    gate_up = torch.empty((num_tokens * top_k, two_inter), device=device, dtype=dtype)
    _invoke_moe_gemm(
        hidden_states,
        w1,
        gate_up,
        None,  # a_scale: the weight-only path keeps activations at full precision
        w1_scale,
        w1_zeros,
        flat_weights,
        sorted_ids,
        expert_ids,
        num_post,
        top_k,
        mul_routed_weight=False,
        quant_mode=quant_mode,
        group_n=group_n,
        group_k=group_k,
        config=config,
    )

    # silu(gate) * up -> [M * top_k, I]
    act = torch.empty((num_tokens * top_k, intermediate), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(intermediate), 1024)
    _silu_and_mul_kernel[(num_tokens * top_k, triton.cdiv(intermediate, block_n))](
        gate_up,
        act,
        None,  # no activation scale on the weight-only path (QUANT_OUT=0)
        gate_up.stride(0),
        intermediate,
        LIMIT=swiglu_limit,
        QUANT_OUT=0,
        QMAX=0.0,
        BLOCK_N=block_n,
        num_warps=4,
    )

    # GEMM2 with the routing weight folded in: [M * top_k, I] x [E, hidden, I].
    # ``act`` is already expanded per slot, so the kernel's A-row gather must use
    # top_k=1 (vLLM does the same: the second invocation passes ``1``), turning
    # ``offs_token // top_k`` into the identity on slot indices.
    expanded = torch.empty((num_tokens * top_k, hidden), device=device, dtype=dtype)
    _invoke_moe_gemm(
        act,
        w2,
        expanded,
        None,
        w2_scale,
        w2_zeros,
        flat_weights,
        sorted_ids,
        expert_ids,
        num_post,
        1,
        mul_routed_weight=True,
        quant_mode=quant_mode,
        group_n=group_n,
        group_k=group_k,
        config=config,
    )

    # Reduce over the top_k slot dim -> [M, hidden]
    out = torch.empty((num_tokens, hidden), device=device, dtype=dtype)
    block_n = min(triton.next_power_of_2(hidden), 1024)
    _moe_sum_kernel[(num_tokens, triton.cdiv(hidden, block_n))](
        expanded, out, hidden, top_k=top_k, BLOCK_N=block_n, num_warps=4
    )
    return out


def fused_moe_w8a8_fp8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """The same FFN with the activations in fp8 as well: true W8A8 experts.

    Both grouped GEMMs run on the fp8 tensor cores (sm89+) instead of widening
    the expert tile to bf16 first, which is the difference between quantising for
    memory and quantising for arithmetic. The activation is quantised per token
    before GEMM1 and per slot-row before GEMM2 — inline in the GEMM kernels on
    launch-bound decode shapes, else in
    :func:`~rapid_llm.kernels.ops.quantization.fp8.fp8_quantize_per_token` —
    with no host synchronisation either way; the layer must stay CUDA-graph
    capturable, which is also why ``moe_align_block_size`` avoids ``bincount``.

    ``w1_zeros``/``w2_zeros`` exist only to keep the :class:`MoeOp` contract; fp8
    is symmetric, so a non-``None`` value is a caller error.

    Args:
        See :func:`fused_moe`. ``w1``/``w2`` must be ``uint8`` e4m3 bytes with
        scales — anything else raises, rather than silently degrading to the
        weight-only path.

    Returns:
        ``[num_tokens, hidden]`` combined expert output, in ``hidden_states``' dtype.
    """
    if w1_zeros is not None or w2_zeros is not None:
        raise ValueError("fp8 is symmetric; zero points belong to the int4 path")
    return _fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        group_n=group_n,
        group_k=group_k,
        act_quant="fp8",
    )


def fused_moe_w8a8_int8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    group_n: int = 0,
    group_k: int = 0,
) -> torch.Tensor:
    """The routed-expert FFN with int8 activations as well: SmoothQuant experts.

    Same pipeline as :func:`fused_moe_w8a8_fp8` over int8 weights: both operands
    enter the dot as int8 and the integer tensor cores (``imma``, from Turing on)
    accumulate in int32, which is exact — the only fp8-vs-int8 asymmetry is that
    there is no capability split, so ``NATIVE_FP8``'s sm89 widening does not
    exist here.

    ``w1_zeros``/``w2_zeros`` exist only to keep the :class:`MoeOp` contract;
    symmetric int8 has no zero points, so a non-``None`` value is a caller error.

    Args:
        See :func:`fused_moe`. ``w1``/``w2`` must be ``int8`` bytes with scales
        — anything else raises, rather than silently degrading to the weight-only
        path.

    Returns:
        ``[num_tokens, hidden]`` combined expert output, in ``hidden_states``' dtype.
    """
    if w1_zeros is not None or w2_zeros is not None:
        raise ValueError("symmetric int8 has no zero points; those belong to the int4 path")
    return _fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        group_n=group_n,
        group_k=group_k,
        act_quant="int8",
    )
