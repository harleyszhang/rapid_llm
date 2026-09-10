"""Fused EP dispatch/combine kernels for the all-to-all token exchange.

The reference path in ``AllToAllDispatcher`` spells placement as an aten op
chain — sort by destination, searchsorted for the segment starts, arange/where
for the in-segment position, then indexed scatters for the rows and ids — and
a cat/gather/scatter/mul/sum chain on the way back. Per MoE layer that is
~30 tiny kernels (~85 us on H100 at decode shapes), a third of an ep2 decode
step, none of it bandwidth. This module does the same work in one launch per
phase:

* ``ep_dispatch_place`` — one program per routing slot: an atomic ticket per
    destination rank gives the buffer row; the program copies the token row
    into the payload and stamps the global expert id after it. Pad rows get
    their id column pre-filled to -1 (``_ep_prep_kernel``), which is all the
    receiver needs — pad row *data* never reaches a consumer, so the payload
    is not zeroed.
* ``ep_local_ids`` — rebase the received ids to this rank's expert window,
    pads (and anything outside the window) to -1 for the grouped GEMM.
* ``ep_combine`` — per token, ``sum_k w_k * recv[send_pos[t*k + j]]`` with the
    product rounded to the activation dtype before the fp32 accumulate, the
    exact arithmetic of the aten mul-then-reduce chain it replaces.

Placement takes atomic tickets instead of the stable sort, so the buffer order
within a destination segment is arbitrary. The output is placement-invariant
bit-for-bit: the grouped GEMM's rows are independent, and combine gathers each
slot by its own ``send_pos`` and sums the top-k axis in routing order. What is
NOT order-invariant is capacity-bounded dropping (which slots overflow depends
on arrival order), so the capacity path keeps the deterministic sort — the
dispatcher picks per call, and this module is only engaged at ``cap == n``.

Everything here is CUDA-graph safe: shapes and pointer arithmetic are pure
functions of the launch arguments and there are no host reads.

Usage:
    from rapid_llm.kernels.ops.moe.ep_dispatch import (
        ep_combine, ep_dispatch_place, ep_local_ids,
    )
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Largest EP group the prep kernel's single-block counter clear supports.
_MAX_EP_SIZE = 32


@triton.jit
def _ep_prep_kernel(
    counters_ptr,  # [ep_size] int32, zeroed here
    ids_ptr,  # payload id column, strided [buf] int32 view
    ids_row_stride,  # int32 elements between payload rows
    buf,
    ep_size,
    EP_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Zero the placement counters; stamp every payload row's id as pad (-1)."""
    pid = tl.program_id(0)
    if pid == 0:
        offs = tl.arange(0, EP_MAX)
        tl.store(counters_ptr + offs, tl.zeros([EP_MAX], tl.int32), mask=offs < ep_size)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        ids_ptr + rows * ids_row_stride,
        tl.full([BLOCK], -1, tl.int32),
        mask=rows < buf,
    )


@triton.jit
def _ep_place_scatter_kernel(
    ids_ptr,  # [n] global expert ids, flattened topk_ids (int32/int64)
    x_ptr,  # [rows, hidden] token embeddings
    payload_x_ptr,  # payload flat, viewed as x's dtype
    payload_ids_ptr,  # payload id column, strided [buf] int32 view
    counters_ptr,  # [ep_size] int32 tickets, zeroed by the prep kernel
    send_pos_ptr,  # [n] int64 out: payload row each slot occupies
    hidden,
    top_k,
    num_local,
    cap,
    x_row_stride,  # elements between token rows in x
    payload_row_elems,  # payload row stride in x-dtype elements
    payload_ids_row_stride,  # payload row stride in int32 elements
    BLOCK_H: tl.constexpr,
):
    """Place one routing slot: ticket a row, copy the token, stamp the id.

    ``slot`` indexes the flattened ``[rows, top_k]`` routing; its token row is
    ``slot // top_k``. The atomic ticket orders slots within a destination
    segment arbitrarily — see the module docstring for why that is safe.
    """
    slot = tl.program_id(0)
    eid = tl.load(ids_ptr + slot)
    dest = (eid // num_local).to(tl.int32)
    pos = tl.atomic_add(counters_ptr + dest, 1)
    sp = (dest * cap + pos).to(tl.int64)
    tl.store(send_pos_ptr + slot, sp)
    tl.store(payload_ids_ptr + sp * payload_ids_row_stride, eid.to(tl.int32))
    src = x_ptr + (slot // top_k).to(tl.int64) * x_row_stride
    dst = payload_x_ptr + sp * payload_row_elems
    for h0 in range(0, hidden, BLOCK_H):
        offs = h0 + tl.arange(0, BLOCK_H)
        m = offs < hidden
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


@triton.jit
def _ep_local_ids_kernel(
    ids_ptr,  # received payload id column, strided [buf] int32 view
    out_ptr,  # [buf] int32 out
    ids_row_stride,
    buf,
    expert_offset,
    num_local,
    BLOCK: tl.constexpr,
):
    """Rebase received global ids to local; anything outside the window -> -1."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = rows < buf
    eid = tl.load(ids_ptr + rows * ids_row_stride, mask=m, other=-1)
    local = eid - expert_offset
    tl.store(out_ptr + rows, tl.where((local >= 0) & (local < num_local), local, -1), mask=m)


@triton.jit
def _ep_combine_kernel(
    recv_ptr,  # [buf, hidden] expert outputs back on this rank
    send_pos_ptr,  # [n] int64 payload rows of each slot
    weights_ptr,  # [n] routing weights, activation dtype
    out_ptr,  # [rows, hidden] out
    hidden,
    recv_row_stride,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``out[t] = sum_j w[t,j] * recv[send_pos[t*K + j]]`` in routing order.

    The product is rounded to the activation dtype before the fp32 accumulate
    — the aten chain this replaces materialised ``results * weights`` in the
    activation dtype and then reduced, so the rounding keeps this kernel
    bit-identical to it.
    """
    t = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < hidden
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in tl.static_range(TOP_K):
        sp = tl.load(send_pos_ptr + t * TOP_K + j)
        w = tl.load(weights_ptr + t * TOP_K + j).to(tl.float32)
        v = tl.load(recv_ptr + sp * recv_row_stride + offs, mask=m, other=0.0).to(tl.float32)
        acc += (v * w).to(recv_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + t * hidden + offs, acc.to(out_ptr.dtype.element_ty), mask=m)


# --------------------------------------------------------------------------- #
# Python wrappers
# --------------------------------------------------------------------------- #
def ep_dispatch_place(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    payload_x: torch.Tensor,
    payload_ids: torch.Tensor,
    *,
    num_local: int,
    cap: int,
    ep_size: int,
) -> torch.Tensor:
    """Fill the dispatch payload; return ``send_pos`` (``[rows*top_k]`` int64).

    ``payload_x`` is the send buffer flattened and viewed as ``x``'s dtype
    (``ep_size * cap`` rows of ``payload_ids.stride(0) * 4`` bytes);
    ``payload_ids`` is its per-row id field as a strided ``[buf]`` int32 view.
    Pad rows keep the -1 id the prep kernel stamps; their data is never read.
    """
    rows, hidden = x.shape
    top_k = topk_ids.shape[1]
    n = rows * top_k
    buf = ep_size * cap
    if ep_size > _MAX_EP_SIZE:
        raise ValueError(f"fused EP dispatch supports ep_size <= {_MAX_EP_SIZE}")
    counters = torch.empty(ep_size, dtype=torch.int32, device=x.device)
    send_pos = torch.empty(n, dtype=torch.int64, device=x.device)
    _ep_prep_kernel[(triton.cdiv(buf, 512),)](
        counters,
        payload_ids,
        payload_ids.stride(0),
        buf,
        ep_size,
        EP_MAX=_MAX_EP_SIZE,
        BLOCK=512,
    )
    _ep_place_scatter_kernel[(n,)](
        topk_ids.reshape(-1),
        x,
        payload_x,
        payload_ids,
        counters,
        send_pos,
        hidden,
        top_k,
        num_local,
        cap,
        x.stride(0),
        payload_x.numel() // buf,
        payload_ids.stride(0),
        BLOCK_H=min(triton.next_power_of_2(hidden), 2048),
        num_warps=8,
    )
    return send_pos


def ep_local_ids(
    payload_ids: torch.Tensor, *, expert_offset: int, num_local: int
) -> torch.Tensor:
    """Local-window ids for the received payload (``[buf]`` int32, pads -1)."""
    buf = payload_ids.shape[0]
    out = torch.empty(buf, dtype=torch.int32, device=payload_ids.device)
    _ep_local_ids_kernel[(triton.cdiv(buf, 512),)](
        payload_ids,
        out,
        payload_ids.stride(0),
        buf,
        expert_offset,
        num_local,
        BLOCK=512,
    )
    return out


def ep_combine(
    recv_out: torch.Tensor,
    send_pos: torch.Tensor,
    flat_weights: torch.Tensor,
    rows: int,
    top_k: int,
) -> torch.Tensor:
    """Weighted per-token reduce of the returned exchange, ``[rows, hidden]``."""
    hidden = recv_out.shape[1]
    out = torch.empty((rows, hidden), dtype=recv_out.dtype, device=recv_out.device)
    block = min(triton.next_power_of_2(hidden), 1024)
    _ep_combine_kernel[(rows, triton.cdiv(hidden, block))](
        recv_out,
        send_pos,
        flat_weights,
        out,
        hidden,
        recv_out.stride(0),
        TOP_K=top_k,
        BLOCK=block,
        num_warps=4,
    )
    return out
