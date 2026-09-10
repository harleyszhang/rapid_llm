"""Fused greedy router: softmax + top-k + renormalisation in one launch.

The GREEDY routing family (Qwen3-MoE, DeepSeek-V2-Lite) runs an fp32 softmax
over the expert axis, a ``torch.topk``, then ``w / sum(w) * scale`` and a cast
— six+ kernels per MoE layer (~17 us of a decode step on H100 at bs16), all of
them launch-bound at a few hundred elements of work. One program per token
does the whole thing in registers here.

Semantics are the aten chain's: fp32 softmax with the max subtracted, top-k in
descending weight order, optional renormalisation, then the routed scaling
factor, cast to the output dtype on store. One deliberate divergence: exact
fp32 ties break to the lower expert id (``tl.argmax`` is left-stable), while
``torch.topk``'s bitonic sort is not stable — tie order is unspecified there.
Ties between distinct experts' fp32 logits are measure-zero off synthetic
inputs; the CUDA-graph parity gate compares this kernel against itself, so it
is unaffected.

CUDA-graph safe: static shapes, no host reads. The grouped/biased routing
families stay in ``grouped_topk.py``.

Usage:
    weights, ids = fused_topk_softmax(router_logits, top_k=8, renormalize=True,
                                      routed_scaling_factor=1.0, out_dtype=torch.bfloat16)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_topk_softmax_kernel(
    logits_ptr,  # [rows, E] fp32 router logits
    out_w_ptr,  # [rows, TOP_K] out dtype
    out_ids_ptr,  # [rows, TOP_K] int64
    E,
    stride_l,
    scale,
    TOP_K: tl.constexpr,
    TOP_K_PAD: tl.constexpr,
    BLOCK_E: tl.constexpr,
    RENORM: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    m = offs < E
    logits = tl.load(logits_ptr + row * stride_l + offs, mask=m, other=float("-inf"))
    # fp32 softmax, max-subtracted — the aten chain's arithmetic.
    p = tl.exp(logits - tl.max(logits, 0))
    p = tl.where(m, p, 0.0)
    p = p / tl.sum(p, 0)

    # Extract the top-k in descending order, clearing each pick from p.
    kk = tl.arange(0, TOP_K_PAD)
    sel_w = tl.zeros([TOP_K_PAD], tl.float32)
    sel_i = tl.zeros([TOP_K_PAD], tl.int64)
    for j in range(TOP_K):
        w = tl.max(p, 0)
        idx = tl.argmax(p, 0)
        sel_w = tl.where(kk == j, w, sel_w)
        sel_i = tl.where(kk == j, idx.to(tl.int64), sel_i)
        p = tl.where(offs == idx, float("-inf"), p)
    if RENORM:
        sel_w = sel_w / tl.sum(sel_w, 0)
    sel_w = sel_w * scale
    tl.store(out_w_ptr + row * TOP_K + kk, sel_w.to(out_w_ptr.dtype.element_ty), mask=kk < TOP_K)
    tl.store(out_ids_ptr + row * TOP_K + kk, sel_i, mask=kk < TOP_K)


def fused_topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
    *,
    renormalize: bool,
    routed_scaling_factor: float,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``softmax -> topk -> renorm -> scale`` over fp32 logits, one launch.

    Args:
        router_logits: ``[rows, num_experts]`` fp32.
        top_k: Experts per token.
        renormalize: Divide the selected weights by their sum (HF
            ``norm_topk_prob``).
        routed_scaling_factor: Multiplier applied after any renorm.
        out_dtype: Weight dtype (the model dtype); ids come back int64 like
            ``torch.topk``.

    Returns:
        ``(topk_weights, topk_ids)`` as ``[rows, top_k]``.
    """
    rows, e = router_logits.shape
    if router_logits.dtype != torch.float32:
        raise ValueError(f"fused_topk_softmax wants fp32 logits, got {router_logits.dtype}")
    weights = torch.empty((rows, top_k), dtype=out_dtype, device=router_logits.device)
    ids = torch.empty((rows, top_k), dtype=torch.int64, device=router_logits.device)
    _fused_topk_softmax_kernel[(rows,)](
        router_logits,
        weights,
        ids,
        e,
        router_logits.stride(0),
        float(routed_scaling_factor),
        TOP_K=top_k,
        TOP_K_PAD=triton.next_power_of_2(top_k),
        BLOCK_E=triton.next_power_of_2(e),
        RENORM=renormalize,
        num_warps=4,
    )
    return weights, ids
