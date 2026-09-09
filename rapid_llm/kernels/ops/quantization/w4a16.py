"""W4A16 GEMM: 4-bit weights (AWQ/GPTQ), fp16/bf16 activations, fp32 accumulation.

Checkpoints pack 8 int4 values per int32 word along K with one fp32 scale (and
zero point) per ``group_size`` channels. The kernel unpacks nibbles, applies
the group-wise dequantisation and multiplies by the fp16 activation inside the
GEMM loop, so the weight never exists at fp16 in HBM.

``BLOCK_K`` is decoupled from ``group_size``; the unpack follows the
:mod:`.nvfp4` idiom — coalesced word load, 3-D shift/reshape to nibbles in
registers, fp32 dequant, ``tl.trans``, ``tl.dot``. Measured on an H100
(N=K=4096, fp16): ``BLOCK_K=256`` fills a 128-byte transaction per output
channel where a per-group loop fetches half of one — 1.4-1.6x at decode
widths, parity at prefill; 512 loses to register pressure, so the tune space
stops at 256. An fp16 magic-number dequant measured 9% faster at m=64 but 12%
slower at m=1, so there is one code path.

Packing order (AWQ/GPTQ standard): int32 word ``w`` holds K indices
``[8*i, 8*i+7]`` as ``nibble_j = (w >> (4*j)) & 0xF``, j = 0..7; the
dequantised value is ``(nibble - zero) * scale``.

Usage:
    y = w4a16_matmul(x, qweight, scales, zeros)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..tile_policy import TileTier, resolve_tiles, tile_tier

_PACK_FACTOR = 8


# --------------------------------------------------------------------------- #
# GEMM kernel — coalesced word loads, in-register unpack, tl.dot
# --------------------------------------------------------------------------- #
@triton.jit
def _w4a16_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    scale_ptr,
    zero_ptr,
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
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """One [BLOCK_M, BLOCK_N] tile of C = A @ dequant(B).T.

    ``BLOCK_K`` is a multiple of ``GROUP_SIZE`` (clamped by the launcher), so
    one iteration covers ``BLOCK_K // GROUP_SIZE`` quantisation groups: the
    packed tile loads as [BLOCK_N, BLOCK_K//8] int32 words — a coalesced
    row-major read — and the scales as [BLOCK_N, BLOCK_K//GROUP_SIZE]. Both
    are indexed [n, k], so the operand is transposed in registers for the dot.
    K divides ``BLOCK_K`` evenly (the launcher guarantees it), so the loop
    needs no k mask.
    """
    WORDS: tl.constexpr = BLOCK_K // 8
    SCALES: tl.constexpr = BLOCK_K // GROUP_SIZE

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_word = tl.arange(0, WORDS)
    offs_scale = tl.arange(0, SCALES)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_word[None, :] * stride_bk
    s_ptrs = scale_ptr + offs_n[:, None] * stride_sn + offs_scale[None, :] * stride_sk
    z_ptrs = zero_ptr + offs_n[:, None] * stride_sn + offs_scale[None, :] * stride_sk

    # Shift constants for unpacking 8 nibbles from one int32
    shifts = (tl.arange(0, 8) * 4).to(tl.int32)  # [8]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BLOCK_K)):
        a_tile = tl.load(a_ptrs)
        b_packed = tl.load(b_ptrs)  # [BLOCK_N, WORDS] int32
        scale = tl.load(s_ptrs)  # [BLOCK_N, SCALES] fp32
        zero = tl.load(z_ptrs)  # [BLOCK_N, SCALES] fp32

        # [BLOCK_N, WORDS, 1] >> [1, 1, 8] -> [BLOCK_N, WORDS, 8]; the reshape
        # lands in k order because nibble j of word w covers k = w*8 + j.
        b_expanded = (b_packed[:, :, None] >> shifts[None, None, :]) & 0xF
        b_flat = tl.reshape(b_expanded, (BLOCK_N, BLOCK_K)).to(tl.float32)
        scale_b = tl.reshape(
            tl.broadcast_to(scale[:, :, None], (BLOCK_N, SCALES, GROUP_SIZE)),
            (BLOCK_N, BLOCK_K),
        )
        zero_b = tl.reshape(
            tl.broadcast_to(zero[:, :, None], (BLOCK_N, SCALES, GROUP_SIZE)),
            (BLOCK_N, BLOCK_K),
        )

        # Dequant in fp32, narrowed to the activation's dtype for the dot. The
        # (nibble - zero) factor is exact — both are integers in [0, 15] — so
        # only the scale's low mantissa bits round, as they did before.
        b_dequant = (b_flat - zero_b) * scale_b
        b_tile = tl.trans(b_dequant).to(a_tile.dtype)  # [BLOCK_K, BLOCK_N]

        accumulator += tl.dot(a_tile, b_tile)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += WORDS * stride_bk
        s_ptrs += SCALES * stride_sk
        z_ptrs += SCALES * stride_sk

    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]

    # Store output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def launch_config(m: int, device_index: int | None = None) -> dict[str, int]:
    """Tile config for ``m`` rows when the autotune store has no entry.

    Hopper and above, measured on an H100: ``BLOCK_N=32`` is what the unpacked
    ``[BLOCK_N, BLOCK_K]`` operand's register budget wants, and ``BLOCK_K=256``
    fills a 128-byte transaction per output channel where ``group_size=128``
    fills half of one. The switches at 32 and 128 rows are both ``bucket_m``
    boundaries, so one store entry never stands in for two heuristic choices.

    Pre-Hopper: **not measured**. The sm90 table's ``BLOCK_K=256`` at four
    stages wants ~190 KB of shared memory per program; sm86-class parts carry
    100 KB, so those configs would spill or fail to compile. The branch below
    halves the k-tile and the pipeline depth (~64 KB) as a conservative
    default — the intended path for hardware this file was never swept on is
    the autotune store, whose per-GPU entry takes precedence.

    Public because ``benchmarks/kernels/bench_quant_gemm.py --tune`` needs the
    config it is trying to beat.
    """
    if tile_tier(device_index) is TileTier.PRE_HOPPER:
        # Conservative pre-Hopper default (unmeasured; autotune overrides).
        if m <= 32:
            return {
                "BLOCK_M": 16,
                "BLOCK_N": 32,
                "BLOCK_K": 128,
                "GROUP_M": 8,
                "num_warps": 4,
                "num_stages": 2,
            }
        if m <= 128:
            return {
                "BLOCK_M": 32,
                "BLOCK_N": 32,
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
    if m <= 32:
        return {
            "BLOCK_M": 16,
            "BLOCK_N": 32,
            "BLOCK_K": 256,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    if m <= 128:
        return {
            "BLOCK_M": 64,
            "BLOCK_N": 32,
            "BLOCK_K": 256,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_K": 256,
        "GROUP_M": 8,
        "num_warps": 4,
        "num_stages": 3,
    }


def _launch(
    a: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    config: dict,
    group_size: int,
) -> None:
    """One kernel launch under an explicit tile config.

    Shared by :func:`w4a16_matmul` and the autotune collector, so a searched
    config is measured on exactly the launch the runtime performs.
    """
    block_m, block_n = config["BLOCK_M"], config["BLOCK_N"]

    block_k = config.get("BLOCK_K") or group_size
    if k % block_k or block_k % group_size:
        block_k = group_size
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    _w4a16_matmul_kernel[grid](
        a,
        qweight,
        out,
        scales,
        zeros,
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
        scales.stride(0),
        scales.stride(1),
        GROUP_SIZE=group_size,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=config["GROUP_M"],
        HAS_BIAS=bias is not None,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


def w4a16_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """``x @ dequant(qweight).T (+ bias)`` with int4 weights unpacked in-kernel.

    The k-tile (``BLOCK_K``) is a multiple of ``group_size`` chosen per M
    bucket — from the autotune store when an entry exists, else the measured
    heuristic below.

    Args:
        x: ``[..., K]`` fp16 activations. Leading dims are flattened.
        qweight: ``[N, K//8]`` packed int32 weights (8 int4 values per word).
        scales: ``[N, K//group_size]`` fp32 dequantisation scales.
        zeros: ``[N, K//group_size]`` fp32 zero points.
        group_size: Number of input channels per quantisation group.
        bias: Optional ``[N]`` bias, added in fp32 before the output cast.

    Returns:
        ``[..., N]`` in ``x``'s dtype.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"w4a16 activations must be fp16 or bf16, got {x.dtype}")
    if qweight.dtype != torch.int32:
        raise ValueError(f"qweight must be int32 (packed int4), got {qweight.dtype}")

    n, k_packed = qweight.shape
    k = k_packed * _PACK_FACTOR
    if x.shape[-1] != k:
        raise ValueError(f"x has {x.shape[-1]} cols but weight expects {k}")
    if k % group_size != 0:
        raise ValueError(f"K ({k}) must be a multiple of group_size ({group_size})")
    if group_size & (group_size - 1) != 0 or group_size < 16:
        raise ValueError(
            f"group_size must be a power of two >= 16 (tl.arange / tl.dot), got {group_size}"
        )

    leading = x.shape[:-1]
    a = x.reshape(-1, k)
    if a.stride(-1) != 1:
        a = a.contiguous()
    m = a.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)

    # Autotune lookup (per-GPU entry) or the device-tiered heuristic.
    config = resolve_tiles(
        "w4a16_matmul",
        m=m,
        n=n,
        k=k,
        dtype_label="int4",
        heuristic=lambda dev: launch_config(m, dev),
        device_index=x.device.index,
    )

    _launch(a, qweight, scales, zeros, bias, out, m, n, k, config, group_size)
    return out.reshape(*leading, n)
