"""Standalone dynamic activation quantisation (per-token-group).

The W8A8 GEMM files quantise per *token* (what their epilogues consume); this
is the finer op block-wise schemes need, mirroring sglang's
``per_token_group_quant`` — one scale per ``group_size``-element slice of
every token, the activation side of a ``block_shape=[128, 128]`` W8A8
checkpoint.

One launch, one pass: a *group* fits one tile, so the amax reduction and the
scaling are register work and the row is read once (a per-token scale needs
the whole row's amax first, hence the two-pass walk). Each program carries
``_QUANT_TILE`` elements' worth of groups (eight at 128) where sglang's flat
kernel carries one.

Usage:
    qx, scales = per_token_group_quant(x, 128, out_dtype=torch.uint8)

Scale layout is decided at allocation, not consumption: pass ``layout=`` and
the grid is born in the consumer's storage order (``scale_layout.py``), or
hand in caller-owned buffers whose strides declare the layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..activation.activations import silu
from .fp8 import FP8_E4M3_MAX, has_native_fp8
from .scale_layout import (
    COLUMN_MAJOR,
    ROW_MAJOR,
    ScaleLayout,
    create_scale_output,
    infer_scale_layout,
)
from .w8a8 import _INT8_MAX

_QUANT_TILE = 1024


# --------------------------------------------------------------------------- #
# Kernel
# --------------------------------------------------------------------------- #
@triton.jit
def _per_token_group_quant_kernel(
    x_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_qm,
    stride_sm,
    stride_sk,
    G,
    H,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_PROG: tl.constexpr,
    QMAX: tl.constexpr,
    OUT_FP8: tl.constexpr,
    FUSE_SILU: tl.constexpr,
    EPS: tl.constexpr,
):
    """Quantise ``[GROUPS_PER_PROG, GROUP_SIZE]`` groups of one token row.

    A group's amax and the elements it scales live in the same tile, so the
    reduction (``axis=1``) and the scaling are register work — no second pass
    through HBM, which is the tax the per-token quantisers must pay because a
    whole row does not fit alongside its own amax.
    """
    row = tl.program_id(0).to(tl.int64)
    pid_g = tl.program_id(1)

    g = pid_g * GROUPS_PER_PROG + tl.arange(0, GROUPS_PER_PROG)
    k = tl.arange(0, GROUP_SIZE)
    offs = g[:, None] * GROUP_SIZE + k[None, :]
    mask = offs < H

    gate = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    if FUSE_SILU:
        # Gate/up halves share the row's base pointer; ``offs`` stays in
        # ``[0, H)`` so the up load at ``H + offs`` cannot leave the row.
        up = tl.load(x_ptr + row * stride_xm + H + offs, mask=mask, other=0.0).to(tl.float32)
        val = silu(gate) * up
    else:
        val = gate

    amax = tl.max(tl.abs(val), axis=1)
    # sglang's eps floor, not the per-token quantisers' 1.0-on-zero: a scale of
    # eps/QMAX keeps an all-zero group's bytes exactly zero without a branch,
    # and matches the reference the block-wise schemes were calibrated on.
    scale = tl.maximum(amax, EPS) / QMAX
    q = val / scale[:, None]
    # amax/QMAX makes the clamp a no-op by construction; it stays as the guard
    # against a non-finite input turning into a NaN byte pattern.
    q = tl.minimum(tl.maximum(q, -QMAX), QMAX)

    tl.store(s_ptr + row * stride_sm + g * stride_sk, scale, mask=g < G)
    if OUT_FP8:
        # Same cast story as fp8.py's per-token kernel: the hardware cvt is
        # round-to-nearest-even except on an exact e4m3 tie, where it may pick
        # the other neighbour from torch's software cast — one code, about one
        # element in 30k. The byte-difference tests gate that bound.
        q = q.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    else:
        # rint, not a plain .to(int8): round-to-nearest-even matches torch's
        # .round() and the per-token int8 quantiser, where .to truncates toward
        # zero. Like the e4m3 cast above it agrees with the torch chain except
        # on a quotient landing exactly on a .5 boundary, where a 1 ULP
        # difference between this kernel's and torch's fp32 division flips the
        # tie; the byte-difference tests gate that bound.
        q = tl.extra.cuda.libdevice.rint(q).to(tl.int8)
    tl.store(q_ptr + row * stride_qm + offs, q, mask=mask)


def _per_token_group_quant_torch(
    flat: torch.Tensor,
    h: int,
    group_size: int,
    fuse_silu_and_mul: bool,
    qmax: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager spelling of the kernel, for devices without the fp8 cast.

    Same role as ``fp8_quantize_per_token``'s torch fallback: pre-sm89 Triton
    cannot emit the e4m3 conversion, and block-wise W8A8 has no native MMA
    there either. int8 never routes here — its conversion is plain arithmetic.
    """
    val = flat.float()
    if fuse_silu_and_mul:
        gate, up = val[:, :h], val[:, h:]
        val = gate * torch.sigmoid(gate) * up
    t, g = flat.shape[0], h // group_size
    grouped = val.reshape(t, g, group_size)
    amax = grouped.abs().amax(dim=-1)
    scale = amax.clamp_min(eps) / qmax
    q = (grouped / scale[:, :, None]).clamp(-qmax, qmax).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).reshape(t, h), scale


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def per_token_group_quant(
    x: torch.Tensor,
    group_size: int = 128,
    *,
    out_dtype: torch.dtype = torch.int8,
    column_major_scales: bool = False,
    fuse_silu_and_mul: bool = False,
    eps: float = 1e-10,
    output_q: torch.Tensor | None = None,
    output_s: torch.Tensor | None = None,
    layout: ScaleLayout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise activations with one scale per token per ``group_size`` elements.

    Where the per-token quantisers give every row one scale, this gives every
    ``group_size``-wide slice of every row its own — the granularity a
    block-quantised *weight* can exploit. No host synchronisation, so a layer
    holding this on its critical path stays CUDA-graph capturable.

    The scale layout is decided where the buffer is born, not where it is
    read: pass ``layout=`` and the grid comes out of ``torch.empty`` already
    in the consumer's storage order, or hand in caller-owned buffers and
    their strides declare the layout. Either way no post-hoc transpose rides
    the critical path.

    Args:
        x: ``[..., K]`` float activations. Leading dims are flattened to rows.
            With ``fuse_silu_and_mul`` the row holds ``[gate | up]`` halves
            (``K`` even) and the quantised value is ``silu(gate) * up``.
        group_size: Elements per scale — a power of two that divides the
            (post-halving) row width; 128 matches the blockwise checkpoints.
        out_dtype: ``torch.int8``, or ``torch.uint8`` for fp8-e4m3 bytes (the
            bit-pattern convention every fp8 op in this package uses).
        column_major_scales: Legacy boolean spelling of
            ``layout=COLUMN_MAJOR``, 2D inputs only. Superseded by ``layout``,
            which also covers TMA-padded grids.
        fuse_silu_and_mul: Quantise ``silu(x[:, :K//2]) * x[:, K//2:]`` straight
            from the gate/up buffer, saving the fp16 round trip through HBM a
            separate activation kernel would take before the quantiser.
        eps: Amax floor, so an all-zero group divides by ``eps/QMAX`` rather
            than zero (sglang bakes in ``1e-10``).
        output_q: Caller-owned buffer ``[..., H]`` in ``out_dtype``; filled in
            place instead of allocating — e.g. a CUDA-graph-captured slab.
        output_s: Caller-owned fp32 scale grid ``[..., K/group_size]``; its
            strides declare the layout and are honoured as-is, with a loud
            error on contradicting ``layout``/``column_major_scales``.
        layout: The ``ScaleLayout`` to allocate the scale grid in when
            ``output_s`` is not given, e.g. ``COLUMN_MAJOR_TMA`` for a TMA-fed
            operand. ``None`` keeps the row-major default.

    Returns:
        ``(q, scales)``: ``q`` shaped like ``x`` (or ``[..., K/2]`` when
        fused) in ``out_dtype``, ``scales`` fp32 shaped ``[..., K/group_size]``
        in the layout the allocation decided.
    """
    if out_dtype not in (torch.int8, torch.uint8):
        raise ValueError(f"out_dtype must be int8 or uint8 (e4m3 bytes), got {out_dtype}")
    if group_size <= 0 or group_size & (group_size - 1):
        raise ValueError(f"group_size must be a power of two, got {group_size}")
    if not x.is_floating_point():
        raise ValueError(f"x must be a float tensor, got {x.dtype}")

    k = x.shape[-1]
    if fuse_silu_and_mul:
        if k % 2:
            raise ValueError(f"fuse_silu_and_mul needs an even row width, got {k}")
        h = k // 2
    else:
        h = k
    if h % group_size:
        raise ValueError(f"row width {h} is not a multiple of group_size {group_size}")
    if layout is not None and column_major_scales and layout.column_major != column_major_scales:
        raise ValueError(
            f"layout {layout} contradicts column_major_scales={column_major_scales}; pass one"
        )

    qmax = FP8_E4M3_MAX if out_dtype is torch.uint8 else _INT8_MAX
    flat = x.reshape(-1, k)
    if flat.stride(-1) != 1:
        flat = flat.contiguous()
    t = flat.shape[0]
    g = h // group_size

    if output_q is not None:
        if output_q.dtype != out_dtype:
            raise ValueError(f"output_q must have dtype {out_dtype}, got {output_q.dtype}")
        if tuple(output_q.shape) != (*x.shape[:-1], h):
            raise ValueError(
                f"output_q must have shape {(*x.shape[:-1], h)}, got {tuple(output_q.shape)}"
            )
    if output_s is not None:
        if tuple(output_s.shape) != (*x.shape[:-1], g):
            raise ValueError(
                f"output_s must have shape {(*x.shape[:-1], g)}, got {tuple(output_s.shape)}"
            )
        # A caller-owned buffer carries its own layout decision: read it back
        # off the strides and fail loudly if it contradicts what was asked.
        resolved = infer_scale_layout(output_s)
        if layout is not None and resolved != layout:
            raise ValueError(f"output_s strides describe {resolved}, but layout declares {layout}")
        if column_major_scales and not resolved.column_major:
            raise ValueError(
                "output_s strides describe row-major scales, "
                "but column_major_scales=True was requested"
            )
    elif layout is not None:
        resolved = layout
    else:
        resolved = COLUMN_MAJOR if column_major_scales else ROW_MAJOR
    if resolved.column_major and x.dim() != 2:
        raise ValueError("column-major scales support 2D inputs only")

    if output_q is not None:
        q = output_q
    else:
        q = torch.empty((t, h), device=x.device, dtype=out_dtype)
    if output_s is not None:
        scales = output_s
    else:
        # create_scale_output hands the grid back in *resolved*'s storage
        # order -- column-major is an allocated-[G, T] view, so the buffer
        # stays what the kernel wrote while the caller sees [T, G] with token
        # stride 1. The grid belongs to the *logical* output
        # (fuse_silu_and_mul halves the row), so allocate to the output shape.
        scales = create_scale_output((*x.shape[:-1], h), x.device, group_size, resolved)

    if t == 0 or g == 0:
        return q.reshape(*x.shape[:-1], h), scales

    if out_dtype is torch.uint8 and not has_native_fp8(x.device.index):
        eq, es = _per_token_group_quant_torch(flat, h, group_size, fuse_silu_and_mul, qmax, eps)
        # The eager spelling mints fresh contiguous buffers; fold them into
        # whatever the caller owns (or the layout demanded) instead of
        # letting the layout slip back to "however eager happened to write
        # it". copy_ carries the values across any stride arrangement.
        if output_q is not None:
            q.view(t, h).copy_(eq)
        else:
            q = eq
        scales.copy_(es.reshape(scales.shape))
        return q.reshape(*x.shape[:-1], h), scales

    # Tile size: _QUANT_TILE elements split as [groups, group_size], capped by
    # however few groups a narrow row actually has. num_warps follows the
    # element count — 256 per warp, the same ratio the per-token quantisers'
    # 1024/4 uses.
    groups_per_prog = max(1, min(g, _QUANT_TILE // group_size))
    # 2D views for the kernel's stride pair: row-major grids flatten their
    # leading dims (contiguous), column-major ones are already 2D.
    q_2d = q.view(t, h)
    scales_2d = scales.view(t, g)
    _per_token_group_quant_kernel[(t, triton.cdiv(g, groups_per_prog))](
        flat,
        q_2d,
        scales_2d,
        flat.stride(0),
        q_2d.stride(0),
        scales_2d.stride(0),
        scales_2d.stride(1),
        g,
        h,
        GROUP_SIZE=group_size,
        GROUPS_PER_PROG=groups_per_prog,
        QMAX=qmax,
        OUT_FP8=out_dtype is torch.uint8,
        FUSE_SILU=fuse_silu_and_mul,
        EPS=eps,
        num_warps=max(1, (groups_per_prog * group_size) // 256),
    )
    return q.reshape(*x.shape[:-1], h), scales
