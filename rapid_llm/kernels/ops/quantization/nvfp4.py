"""NVFP4 GEMM: e2m1 weights with 16-element e4m3 block scales, 16-bit activations.

NVFP4 (NVIDIA ModelOpt / TensorRT-LLM) is a *two-level* format, which is the
whole reason it beats plain int4 on accuracy at the same bit width:

* each element is fp4-e2m1 — representable magnitudes ``{0, .5, 1, 1.5, 2, 3,
  4, 6}``, two to a byte;
* every **16** consecutive k elements share one fp8-e4m3 block scale (a
  ``uint8`` bit pattern) — 8x finer than AWQ's usual 128, which is what lets
  4 bits of near-mantissa-free weight stay usable;
* one fp32 global scale for the whole tensor, whose only job is to bring the
  block scales into e4m3's range.

Reconstruction is ``w = e2m1(nibble) * e4m3(block_scale) * global_scale``.

Usage:
    y = nvfp4_matmul(x, qweight, block_scale, global_scale)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..tile_policy import TileTier, resolve_tiles, tile_tier
from .w8a16 import FP8_E4M3_BIT_TRICK_SCALE, dequant_fp8e4m3

#: Weight elements sharing one block scale. Fixed by the format, not tunable.
NVFP4_BLOCK = 16

#: Largest finite e2m1 magnitude, i.e. what a block's amax is mapped onto.
E2M1_MAX = 6.0

#: Largest finite e4m3 magnitude, i.e. what the *block scales* are mapped onto.
FP8_E4M3_MAX = 448.0

#: e2m1 values are two to a byte.
_PACK_FACTOR = 2


# --------------------------------------------------------------------------- #
# e2m1 -> fp32
# --------------------------------------------------------------------------- #
@triton.jit
def dequant_e2m1(nibble):
    """Widen e2m1 nibbles to exact fp32 values.

    The 4 bits are ``s.ee.m``. For ``e >= 1`` the value is
    ``2**(e-1) * (1 + m/2)``, which is an fp32 bit pattern assembled directly:
    exponent field ``e - 1 + 127``, mantissa top bit ``m``. ``e == 0`` is the
    subnormal row and does not follow that rule — it encodes ``m * 0.5``, i.e.
    ``{0, 0.5}`` — so it is patched in afterwards rather than computed.
    """
    exponent = (nibble >> 1) & 0x3
    mantissa = nibble & 0x1
    # Exponent field e-1+127 == e+126; mantissa's single bit is fp32's bit 22.
    bits = ((exponent + 126) << 23) | (mantissa << 22)
    magnitude = bits.to(tl.float32, bitcast=True)
    magnitude = tl.where(exponent == 0, mantissa.to(tl.float32) * 0.5, magnitude)
    return tl.where((nibble >> 3) & 0x1 == 1, -magnitude, magnitude)


# --------------------------------------------------------------------------- #
# GEMM kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _nvfp4_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_ptr,
    gscale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    stride_sn,
    stride_sk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    DEQUANT_SCALE: tl.constexpr,
):
    """One ``[BLOCK_M, BLOCK_N]`` tile of ``C = A @ dequant(B).T``.

    A is ``[M, K]`` 16-bit, B is ``[N, K//2]`` packed e2m1, the block scales are
    ``[N, K//16]`` e4m3 bytes. ``BLOCK_K`` is a multiple of 16, so the
    ``BLOCK_K // 16`` scales loaded per iteration cover the k-tile exactly.
    """
    NIBBLE_COLS: tl.constexpr = BLOCK_K // 2
    SCALE_COLS: tl.constexpr = BLOCK_K // 16

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Grouped pid ordering: consecutive programs walk down a column strip so the
    # A tiles they share stay resident in L2.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_byte = tl.arange(0, NIBBLE_COLS)
    offs_blk = tl.arange(0, SCALE_COLS)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak

    # B and its scales are indexed [n, k]: the nibble pair has to end up adjacent
    # in the last axis for the reshape below to land in k order, so the tile is
    # built row-major over N and transposed for the dot.
    b_ptrs = b_ptr + offs_bn[:, None] * stride_bn + offs_byte[None, :] * stride_bk
    s_ptrs = scale_ptr + offs_bn[:, None] * stride_sn + offs_blk[None, :] * stride_sk

    # Low nibble is the even k index, high nibble the odd one.
    shifts = tl.arange(0, 2) * 4

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_rem, other=0.0)
        # A masked-off byte reads as 0 -> two e2m1 zeros, and a masked-off scale
        # as e4m3 zero, so the tail of a ragged K contributes nothing without a
        # separate epilogue.
        packed = tl.load(b_ptrs, mask=offs_byte[None, :] * 2 < k_rem, other=0)
        scale_bytes = tl.load(s_ptrs, mask=offs_blk[None, :] * 16 < k_rem, other=0)

        nibbles = (packed.to(tl.int32)[:, :, None] >> shifts[None, None, :]) & 0xF
        b = dequant_e2m1(tl.reshape(nibbles, (BLOCK_N, BLOCK_K)))

        # One scale per 16 k elements, broadcast across the run it covers. The
        # multiply cannot move outside the dot the way w8a16's can: the scale
        # varies *within* the k-tile.
        scales = dequant_fp8e4m3(scale_bytes).to(tl.float32)
        scales = tl.reshape(
            tl.broadcast_to(scales[:, :, None], (BLOCK_N, SCALE_COLS, 16)),
            (BLOCK_N, BLOCK_K),
        )
        # Narrowed to the activation's dtype for tl.dot, which needs both
        # operands in one type. Exact: 2 significant bits from e2m1 times 4 from
        # e4m3 needs 6, and the narrowest activation here (bf16) carries 8.
        b = (b * scales).to(a.dtype)
        accumulator += tl.dot(a, tl.trans(b))

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += NIBBLE_COLS * stride_bk
        s_ptrs += SCALE_COLS * stride_sk

    # Both remaining factors are k-invariant, so they ride out here: the fp32
    # global scale, and the 2**8 the e4m3 bit trick left on the block scales.
    accumulator *= tl.load(gscale_ptr) * DEQUANT_SCALE
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# --------------------------------------------------------------------------- #
# Launch configuration
# --------------------------------------------------------------------------- #
def _launch_config(num_tokens: int, device_index: int | None) -> dict:
    """Tile shape for ``num_tokens`` rows on this device.

    Hopper and above: measured, not inherited. A sweep of ``BLOCK_M x BLOCK_N
    x BLOCK_K x GROUP_M x warps x stages`` over the four Qwen3-4B projection
    shapes, scored on the sum of the four latencies at each token count,
    produced the sm90 table below on an H100.

    Pre-Hopper: **not measured** — nvfp4 has only been swept on sm90, and the
    sm90 table does not even fit sm86-class parts: ``BLOCK_K=256`` with 3
    stages needs ~110 KB of shared memory per program against A10's 100 KB
    budget, so Triton would spill or refuse the config outright. The table
    below is a conservative default for those devices (halved k-tile, two
    stages, ~25 KB per program) rather than a measured one — the intended
    path for any hardware this file was never swept on is the autotune
    store, whose per-GPU entry :func:`~rapid_llm.kernels.ops.tile_policy.
    resolve_tiles` consults before either table.
    """
    if tile_tier(device_index) is TileTier.PRE_HOPPER:
        # Conservative pre-Hopper default (unmeasured; autotune overrides).
        if num_tokens <= 32:
            return {
                "BLOCK_M": 16,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
                "num_warps": 4,
                "num_stages": 2,
            }
        if num_tokens <= 512:
            return {
                "BLOCK_M": 32,
                "BLOCK_N": 64,
                "BLOCK_K": 128,
                "GROUP_M": 8,
                "num_warps": 4,
                "num_stages": 2,
            }
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 128,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        }
    if num_tokens <= 16:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 32,
            "BLOCK_K": 256,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    if num_tokens <= 32:
        return {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "BLOCK_K": 256,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    if num_tokens <= 512:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_K": 256,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        }
    # Past ~1k tokens the dot finally has enough rows to pay for a wider n-tile,
    # and the deep pipeline stops helping: 2 stages beat 3 by 7%.
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 256,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 2,
    }


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def nvfp4_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(qweight).T (+ bias)`` for NVFP4 weights.

    Args:
        x: ``[..., K]`` fp16 or bf16 activations; leading dims are flattened.
        qweight: ``[N, K // 2]`` ``uint8``, two e2m1 nibbles per byte, low nibble
            first. Last dim must be contiguous.
        block_scale: ``[N, K // 16]`` ``uint8`` e4m3 bit patterns, one per 16
            consecutive k elements.
        global_scale: fp32 scalar tensor (any shape with one element). Stays on
            the device — reading it here would put a sync on the decode path.
        bias: Optional ``[N]`` bias, added in fp32 before the output cast.

    Returns:
        ``[..., N]`` in ``x``'s dtype.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"nvfp4 activations must be fp16 or bf16, got {x.dtype}")
    if qweight.dtype != torch.uint8:
        raise ValueError(f"qweight must be uint8 (packed e2m1), got {qweight.dtype}")
    if block_scale.dtype != torch.uint8:
        raise ValueError(f"block_scale must be uint8 (e4m3 bits), got {block_scale.dtype}")
    if global_scale.numel() != 1:
        raise ValueError(f"global_scale must hold one element, got {global_scale.numel()}")
    if qweight.stride(-1) != 1:
        raise ValueError("qweight last dimension must be contiguous")

    n, k_packed = qweight.shape
    k = k_packed * _PACK_FACTOR
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} cols but weight expects {k}")
    # A ragged block would need the tail scale to cover fewer than 16 elements,
    # which the format has no way to express.
    if k % NVFP4_BLOCK != 0:
        raise ValueError(f"K ({k}) must be a multiple of {NVFP4_BLOCK}")
    if tuple(block_scale.shape) != (n, k // NVFP4_BLOCK):
        raise ValueError(
            f"block_scale must be {(n, k // NVFP4_BLOCK)}, got {tuple(block_scale.shape)}"
        )

    leading = x.shape[:-1]
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    # Autotune lookup (per-GPU entry) or the device-tiered heuristic, both
    # behind ``resolve_tiles``. The kernel masks a ragged k-tail, so the only
    # invariant a tuned BLOCK_K has to keep is the 16-element scale group —
    # ``block_k_multiple`` converges it.
    cfg = resolve_tiles(
        "nvfp4_matmul",
        m=m,
        n=n,
        k=k,
        dtype_label="nvfp4",
        heuristic=lambda dev: _launch_config(m, dev),
        device_index=x.device.index,
        block_k_multiple=NVFP4_BLOCK,
    )
    grid = (triton.cdiv(m, cfg["BLOCK_M"]) * triton.cdiv(n, cfg["BLOCK_N"]),)
    _nvfp4_matmul_kernel[grid](
        a,
        qweight,
        out,
        block_scale,
        global_scale,
        bias,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        out.stride(0),
        out.stride(1),
        block_scale.stride(0),
        block_scale.stride(1),
        HAS_BIAS=bias is not None,
        DEQUANT_SCALE=FP8_E4M3_BIT_TRICK_SCALE,
        **cfg,
    )
    return out.reshape(*leading, n)


# --------------------------------------------------------------------------- #
# Quantisation
# --------------------------------------------------------------------------- #
#: Midpoints between consecutive e2m1 magnitudes ``{0,.5,1,1.5,2,3,4,6}``. Seven
#: boundaries put a magnitude into one of eight buckets, which *is* the 3-bit
#: magnitude field, so no value table or search is needed to encode.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def quantize_nvfp4_blockwise(
    weight: torch.Tensor, block: int = NVFP4_BLOCK
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise a ``[N, K]`` float weight to NVFP4.

    The two levels are fitted outwards. The global scale maps the tensor amax
    onto ``6 * 448``, the largest product the format can name, so that every
    block scale lands inside e4m3's range; each block scale then maps its own
    amax onto 6.

    One subtlety worth the extra line: the elements are divided by the block
    scale *after* it has been rounded to e4m3, not by the ideal scale. Dividing
    by the ideal one would leave the e4m3 rounding error to compound with the
    e2m1 rounding error instead of being absorbed by it.

    Args:
        weight: ``[N, K]`` float tensor. ``K`` must be a multiple of ``block``.
        block: Elements per block scale. Only :data:`NVFP4_BLOCK` is a valid
            NVFP4 checkpoint; the parameter exists for tests.

    Returns:
        ``(packed, block_scale, global_scale)`` — ``[N, K // 2]`` uint8 nibble
        pairs, ``[N, K // block]`` uint8 e4m3 bit patterns, and a one-element
        fp32 tensor. All on ``weight``'s device.
    """
    if weight.ndim != 2:
        raise ValueError(f"nvfp4 quantisation expects a 2-D weight, got {tuple(weight.shape)}")
    n, k = weight.shape
    if k % block != 0:
        raise ValueError(f"K ({k}) must be a multiple of block ({block})")
    if block % _PACK_FACTOR != 0:
        raise ValueError(f"block ({block}) must be even to pack two values per byte")

    w = weight.detach().float()
    amax = w.abs().amax()
    # An all-zero weight has no scale to derive; 1.0 keeps the reconstruction at
    # zero instead of propagating a NaN.
    global_scale = torch.where(amax > 0, amax / (E2M1_MAX * FP8_E4M3_MAX), torch.ones_like(amax))

    blocks = w.unflatten(-1, (k // block, block))
    raw_scale = (blocks.abs().amax(-1) / (E2M1_MAX * global_scale)).clamp(max=FP8_E4M3_MAX)
    scale_q = raw_scale.to(torch.float8_e4m3fn)

    effective = scale_q.float() * global_scale
    divisor = torch.where(effective > 0, effective, torch.ones_like(effective))
    scaled = blocks / divisor.unsqueeze(-1)

    boundaries = torch.tensor(_E2M1_MIDPOINTS, dtype=torch.float32, device=w.device)
    # bucketize rather than an argmin over a value table: the table form
    # materialises ``[N, K, 8]``, which for a 2560x9728 projection is 800 MB.
    # right=False breaks an exact midpoint towards the smaller magnitude.
    magnitude = torch.bucketize(scaled.abs(), boundaries).to(torch.uint8)
    codes = magnitude | ((scaled < 0).to(torch.uint8) << 3)

    pairs = codes.flatten(-2).reshape(n, k // _PACK_FACTOR, _PACK_FACTOR)
    packed = pairs[..., 0] | (pairs[..., 1] << 4)
    return packed.contiguous(), scale_q.view(torch.uint8).contiguous(), global_scale.reshape(1)
