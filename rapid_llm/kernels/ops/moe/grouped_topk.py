"""Grouped top-k MoE routing: DeepSeek's grouped/biased router as one kernel.

``grouped_topk`` covers DeepSeek's two grouped families — V2's
``group_limited_greedy`` and V2.5+/V3's ``noaux_tc``. The torch reference
(:func:`grouped_topk_torch`) pins the tests and serves CPU inputs; on CUDA
one program per token keeps all intermediates in registers where the
reference launches ~10 little kernels, which dominates at decode batch sizes.

Semantics (vLLM-aligned): score in fp32; biased scores *choose*, original ones
*weight*; keep ``topk_group`` groups (sum of two best when biased, else max)
and mask the rest; top-k, renormalise, routed scale. Exact fp32 ties may break
differently than ``torch.topk`` (measure-zero); every tie-break-independent
rule is pinned by the tests.

Usage:
    from rapid_llm.kernels import grouped_topk
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is optional on CPU-only and macOS installs.
    triton = None

    class _TritonLanguageStub:
        constexpr = object()

    tl = _TritonLanguageStub()


def _validate_geometry(
    n_experts: int, top_k: int, num_expert_group: int, topk_group: int
) -> int:
    """Validate routing geometry and return experts per group."""
    if n_experts < 1:
        raise ValueError("num_experts must be positive")
    if num_expert_group < 1 or n_experts % num_expert_group:
        raise ValueError(
            f"num_experts {n_experts} must divide into num_expert_group {num_expert_group}"
        )
    if not 1 <= topk_group <= num_expert_group:
        raise ValueError(f"topk_group must be in [1, {num_expert_group}], got {topk_group}")
    per_group = n_experts // num_expert_group
    if not 1 <= top_k <= topk_group * per_group:
        raise ValueError(
            f"top_k must be in [1, {topk_group * per_group}] for the selected groups, got {top_k}"
        )
    return per_group

_jit = triton.jit if triton is not None else lambda function: function


def grouped_topk_torch(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k routing: DeepSeek-V2's ``group_limited_greedy`` and
    V2.5+/V3's ``noaux_tc``, semantics aligned with vLLM's ``grouped_topk``.

    Pipeline: score every expert in fp32 (softmax or sigmoid); when a
    correction bias exists, biased scores *choose* the experts while original
    scores *weight* them — which is why the bias can shift selection without
    shifting the output magnitude. Group scores are the sum of each group's
    two best experts with a bias (one outlier alone cannot win a group), the
    plain max otherwise; only the ``topk_group`` strongest groups survive.
    Top-k over the survivors, then renormalise and apply the routed scale.

    Args:
        router_logits: ``[tokens, num_experts]`` raw gating output, fp32.
        top_k: Experts each token activates.
        renormalize: Whether to renormalise the selected weights.
        num_expert_group: Expert groups; must divide the expert count.
        topk_group: Groups each token may draw experts from.
        scoring_func: ``"softmax"`` (V2 family) or ``"sigmoid"`` (V2.5+/V3).
        routed_scaling_factor: The DeepSeek routed-output scale.
        e_score_correction_bias: ``[num_experts]`` fp32 correction bias a
            ``noaux_tc`` checkpoint ships; ``None`` selects and weights with
            the same score.

    Returns:
        ``(weights, ids)``, each ``[tokens, top_k]``; weights fp32.

    Raises:
        ValueError: If ``scoring_func`` is unsupported, or the expert count
            does not split into ``num_expert_group`` equal groups.
    """
    n_experts = router_logits.shape[-1]
    per_group = _validate_geometry(n_experts, top_k, num_expert_group, topk_group)
    if scoring_func == "softmax":
        scores = torch.softmax(router_logits, dim=-1)
    elif scoring_func == "sigmoid":
        scores = torch.sigmoid(router_logits)
    else:
        raise ValueError(f"unsupported MoE scoring_func {scoring_func!r}")
    num_tokens = scores.shape[0]

    # Biased scores only choose which experts run; the originals decide how
    # much each counts.
    original_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias.unsqueeze(0)
        if per_group >= 2:
            group_scores = (
                scores.view(num_tokens, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )
        else:
            # One-expert groups (trimmed checkpoints that keep the full
            # ``n_group``) have no top-2 to sum; the group score degenerates
            # to its single biased expert. The math is the limit of the top-2
            # sum, and the reference implementations (transformers, vLLM)
            # simply crash on this geometry instead.
            group_scores = scores.view(num_tokens, num_expert_group, -1).max(dim=-1).values
    else:
        group_scores = scores.view(num_tokens, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_expert_group, n_experts // num_expert_group)
        .reshape(num_tokens, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    topk_ids = torch.topk(tmp_scores, k=top_k, dim=-1, sorted=False)[1]
    # Gather the originals: the -inf cells never survive selection, so this
    # matches vLLM's "topk values when no bias, gather originals when there is
    # one" with one code path.
    topk_weights = original_scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids


# --------------------------------------------------------------------------- #
# The kernel: one program per token, everything in registers.
# --------------------------------------------------------------------------- #
@_jit
def _grouped_topk_kernel(
    logits_ptr,
    bias_ptr,
    weights_ptr,
    ids_ptr,
    num_experts,
    scale,
    n_group,
    per_group,
    topk_group,
    top_k,
    HAS_BIAS: tl.constexpr,
    SIGMOID: tl.constexpr,
    RENORM: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    token = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < num_experts
    # Padding lanes load as -inf: they cannot win a softmax (exp(-inf) = 0)
    # and sigmoid(-inf) = 0 keeps them out of every score — the expert count
    # need not be a power of two (V2 routes 160 experts through BLOCK_E=256).
    logits = tl.load(logits_ptr + token * num_experts + offs_e, mask=valid, other=-float("inf")).to(
        tl.float32
    )

    if SIGMOID:
        scores = tl.sigmoid(logits)
    else:
        centered = tl.exp(logits - tl.max(logits, 0))
        scores = centered / tl.sum(centered, 0)

    biased = scores
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_e, mask=valid, other=0.0)
        biased = scores + bias

    # Group scores: the sum of the group's two best biased scores when a bias
    # exists, the plain max otherwise. Exactly one occurrence of the max is
    # removed for the second — tl.argmax names one lane, so a duplicated max
    # still counts twice, matching torch.topk(2) on tied values.
    offs_g = tl.arange(0, BLOCK_G)
    group_of = offs_e // per_group
    group_scores = tl.full((BLOCK_G,), -float("inf"), tl.float32)
    for g in range(n_group):
        in_group = (group_of == g) & valid
        v = tl.where(in_group, biased, -float("inf"))
        top1 = tl.max(v, 0)
        if HAS_BIAS:
            without_top1 = tl.where(offs_e == tl.argmax(v, 0), -float("inf"), v)
            top2 = tl.max(without_top1, 0)
            group_scores = tl.where(offs_g == g, top1 + top2, group_scores)
        else:
            group_scores = tl.where(offs_g == g, top1, group_scores)

    # Keep the topk_group strongest groups; every other group's experts are
    # out of the selection entirely, however large their biased scores.
    group_alive = tl.zeros((BLOCK_G,), tl.int1)
    for _ in range(topk_group):
        best = tl.argmax(group_scores, 0)
        group_alive = group_alive | (offs_g == best)
        group_scores = tl.where(offs_g == best, -float("inf"), group_scores)
    in_alive_group = (
        tl.sum(((offs_g[:, None] == group_of[None, :]) & group_alive[:, None]).to(tl.int32), 0) > 0
    )

    # Expert selection over the survivors by BIASED score, descending; the
    # weights gathered afterwards are the ORIGINAL scores — the bias chooses,
    # it never weighs. Each round removes exactly the winner, so the top_k
    # survivors are distinct lanes and each writes its own output column.
    candidates = tl.where(in_alive_group & valid, biased, -float("inf"))
    selected = tl.zeros((BLOCK_E,), tl.int1)
    rank = tl.full((BLOCK_E,), -1, tl.int32)
    for k in range(top_k):
        expert = tl.argmax(candidates, 0)
        selected = selected | (offs_e == expert)
        rank = tl.where(offs_e == expert, k, rank)
        candidates = tl.where(offs_e == expert, -float("inf"), candidates)

    selected_scores = tl.where(selected, scores, 0.0)
    if RENORM:
        weights = selected_scores / tl.sum(selected_scores, 0) * scale
    else:
        weights = selected_scores * scale

    out = token * top_k + rank
    tl.store(weights_ptr + out, weights, mask=selected)
    tl.store(ids_ptr + out, offs_e.to(ids_ptr.dtype.element_ty), mask=selected)


def grouped_topk(
    router_logits: torch.Tensor,
    *,
    top_k: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grouped top-k routing with the fused kernel on CUDA (see module docstring).

    Signature, semantics and error contract are :func:`grouped_topk_torch`'s;
    on CUDA with fp32 logits and a geometry the kernel serves, the work runs
    in one Triton program per token. Outside that contract — CPU tensors,
    non-fp32 logits or a bias with one-expert groups — the call falls back to
    the torch reference, so the two
    paths never disagree on what runs.

    Returns:
        ``(weights, ids)``, each ``[tokens, top_k]``; weights fp32, ids int64
        (the dtypes the torch reference returns).
    """
    n_experts = router_logits.shape[-1]
    per_group = _validate_geometry(n_experts, top_k, num_expert_group, topk_group)
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError(f"unsupported MoE scoring_func {scoring_func!r}")
    kernelable = (
        triton is not None
        and router_logits.is_cuda
        and router_logits.dtype == torch.float32
        # the biased group score is a top-2 sum — a one-expert group has none.
        and (e_score_correction_bias is None or per_group >= 2)
    )
    if not kernelable:
        return grouped_topk_torch(
            router_logits,
            top_k=top_k,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

    logits = router_logits.contiguous()
    num_tokens = logits.shape[0]
    weights = torch.empty((num_tokens, top_k), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_tokens, top_k), dtype=torch.int64, device=logits.device)
    if num_tokens == 0:
        return weights, ids
    bias = e_score_correction_bias
    block_e = triton.next_power_of_2(n_experts)
    _grouped_topk_kernel[(num_tokens,)](
        logits,
        # a dead pointer the HAS_BIAS=False specialisation never loads.
        bias if bias is not None else logits,
        weights,
        ids,
        n_experts,
        float(routed_scaling_factor),
        num_expert_group,
        per_group,
        topk_group,
        top_k,
        HAS_BIAS=bias is not None,
        SIGMOID=scoring_func == "sigmoid",
        RENORM=bool(renormalize),
        BLOCK_E=block_e,
        BLOCK_G=triton.next_power_of_2(num_expert_group),
        # the whole tile is a few hundred lanes; reductions are cheapest on
        # one warp until the padded expert count outgrows it.
        num_warps=max(1, min(4, block_e // 512)),
    )
    return weights, ids
