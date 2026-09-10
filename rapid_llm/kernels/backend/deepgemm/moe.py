"""DeepGEMM grouped fp8 MoE — the ``deepgemm/grouped_fp8_moe`` row's wrapper.

:func:`grouped_moe` repacks per-expert weights into contiguous grouped
tensors (:func:`_nt_experts`) and runs the grouped GEMM for the tokens
each expert actually holds.

Usage:
    y = grouped_moe(hidden_states, w1, w2, topk_weights, topk_ids)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .quant import nt_block_fp8_from_checkpoint, per_token_group_quant_fp8

#: The contiguous grouped kernel wants each expert segment aligned to this.
ALIGNMENT = 128

# data_ptr -> (w_fp8, w_scales, source_weight). Holding the source tensor is
# what makes the data_ptr key sound: while the reference lives, the caching
# allocator cannot hand that address to another tensor, so a hit is always
# this weight's own repack. A shape check alone could not promise that -- a
# freed-and-reused allocation of the same shape would slip through. The same
# contract as ``linear._NT_CACHE``.
_EXPERT_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _nt_experts(
    w: torch.Tensor,
    w_scale: torch.Tensor | None,
    group_n: int,
    group_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached stacked NT fp8 experts for ``w`` ``[E, n, k]``."""
    key = w.data_ptr()
    hit = _EXPERT_CACHE.get(key)
    if hit is not None and hit[2] is w:
        return hit[0], hit[1]
    w_fp8, w_scales = nt_block_fp8_from_checkpoint(w, w_scale, group_n, group_k)
    _EXPERT_CACHE[key] = (w_fp8, w_scales, w)
    return w_fp8, w_scales


def grouped_moe(
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
    """Run all dispatched experts via DeepGEMM's grouped fp8 GEMMs.

    Args:
        hidden_states: ``[tokens, hidden]`` activations.
        w1: ``[E, 2 * intermediate, hidden]`` fused gate/up projections.
        w2: ``[E, hidden, intermediate]`` down projections.
        topk_weights: ``[tokens, topk]`` router weights.
        topk_ids: ``[tokens, topk]`` expert id per (token, slot).
        w1_scale / w2_scale: Native block scales for the expert weights.
        w1_zeros / w2_zeros: Ignored — fp8 is symmetric.
        group_n / group_k: Native scale-block geometry of ``w1``/``w2``.
        swiglu_limit: Clamp gate at ``+limit`` and up at ``+-limit`` before the
            activation — the same gpt-oss semantics the native ``fused_moe``
            applies (DeepSeek-V4's ``7.0``); ``inf`` keeps plain SwiGLU.
        mxfp4: Unsupported here (DeepGEMM groups fp8 experts only); present so
            the signature satisfies the shared ``moe`` contract. The native
            Triton row carries the MXFP4 e2m1 decode.

    Returns:
        ``[tokens, hidden]`` in ``hidden_states.dtype``.
    """
    if mxfp4:
        raise ValueError(
            "the deepgemm backend does not decode MXFP4 experts; "
            "route the block at the native Triton row instead"
        )
    import deep_gemm  # the JIT kernels live with the library; import at call time

    tokens, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    num_experts = w1.shape[0]
    device = hidden_states.device

    # --- group tokens per expert, segments aligned to ALIGNMENT ------------ #
    flat_ids = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat_ids, stable=True)  # flat (token, slot) indices
    sorted_ids = flat_ids[order]
    counts = torch.bincount(flat_ids, minlength=num_experts)
    padded_counts = ((counts + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
    group_starts = torch.cumsum(padded_counts, 0) - padded_counts
    m_padded = int(padded_counts.sum())

    # Scatter real rows to their aligned slot; padding rows stay zero.
    ranks = torch.arange(flat_ids.numel(), device=device) - group_starts[sorted_ids]
    dest = group_starts[sorted_ids] + ranks
    a = torch.zeros(m_padded, hidden, dtype=hidden_states.dtype, device=device)
    a[dest] = hidden_states[order // topk]
    m_indices = torch.repeat_interleave(torch.arange(num_experts, device=device), padded_counts).to(
        torch.int32
    )

    # --- gate/up grouped GEMM, SwiGLU, down grouped GEMM ------------------- #
    w1_fp8, w1_scales = _nt_experts(w1, w1_scale, group_n, group_k)
    qa, qa_scales = per_token_group_quant_fp8(a)
    h = deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        qa, qa_scales, w1_fp8, w1_scales, m_indices
    )  # [m_padded, 2 * intermediate]
    gate, up = h.chunk(2, dim=-1)
    # The contract's swiglu_limit: gate at +limit, up at +-limit, then silu*up
    # — matching the native kernel's LIMIT branch bit for bit.
    if swiglu_limit < float("inf"):
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    h = F.silu(gate) * up

    w2_fp8, w2_scales = _nt_experts(w2, w2_scale, group_n, group_k)
    qh, qh_scales = per_token_group_quant_fp8(h)
    y = deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        qh, qh_scales, w2_fp8, w2_scales, m_indices
    )  # [m_padded, hidden]

    # --- gather real rows back and reduce with the router weights ---------- #
    y_tok = y[dest].view(tokens, topk, hidden)
    final = (y_tok * topk_weights.to(y.dtype).unsqueeze(-1)).sum(dim=1)
    return final.to(hidden_states.dtype)
