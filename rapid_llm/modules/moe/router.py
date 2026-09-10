"""Routing layer (stage 1): router GEMM + top-k selection.

Mirrors sglang's ``TopK``/``TopKOutput`` seam. :class:`TopKRouter` turns token
embeddings into ``(weights, ids)`` through the tiered router GEMM and one of two
selection families (greedy top-k, or the grouped ``noaux_tc``/``group_limited_greedy``
selection re-exported from :mod:`rapid_llm.kernels.ops.moe.grouped_topk`). The
``gate_weight`` parameter itself stays on :class:`~rapid_llm.modules.moe.layer.SparseMoeBlock`
(checkpoint loading and the weight scan reach for it there); the router only reads it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from enum import Enum
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ...kernels import grouped_topk
from ...kernels.ops.moe.topk_softmax import fused_topk_softmax

# --------------------------------------------------------------------------- #
# Router GEMM: a vllm-style tiered dispatch (mirrors GateLinear's 5 tiers).
# --------------------------------------------------------------------------- #
# Each tier is gated on its deps. Only tiers 4-5 (pure torch) run here; tiers 1-3 are
# slots for vllm's CuteDSL kernels and compiled fp32 op (they need the cutlass Python
# DSL + quack, or vllm's _C extension), activated at runtime if ported in or present.

#: tier 1 / tier 3 CuteDSL kernels are injectable hooks. rapid_llm has not
#: ported vllm's CuteDSL router kernels (``ll_bf16_gemm``, ``bf16x3``); assign
#: a callable here once ported and the corresponding tier activates.
_LL_BF16_GEMM: Callable | None = None  # tier 1: (x, w, out_dtype) -> fp32
_BF16X3_GEMM: Callable | None = None  # tier 3: (x, w) -> fp32

#: vllm's tier-2 fp32 kernel is instantiated only for these (hidden, experts).
_FP32_ROUTER_SHAPES = frozenset({(3072, 256), (6144, 128)})


@functools.lru_cache(maxsize=1)
def _fp32_router_op_available() -> bool:
    """Whether vllm's compiled ``fp32_router_gemm`` op is present (tier 2)."""
    return hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "fp32_router_gemm")


def _router_gemm(x: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """Router-logits GEMM (fp32 out) via a vllm-style 5-tier dispatch.

    Tiers, fastest first, each gated on its deps (only the runnable ones fire):

    1. CuteDSL ``ll_bf16_gemm``  — SM90+, M<=16, bf16, K%8==0    (hook; unported)
    2. vllm ``fp32_router_gemm`` — fp32 weight, tuned shapes, M<=32 (opportunistic)
    3. CuteDSL ``bf16x3``        — SM100                          (hook; unported)
    4. cuBLAS bf16->fp32         — ``torch.mm(out_dtype=fp32)``   (active)
    5. ``F.linear`` fp32         — CPU / non-bf16 fallback        (active)

    Every tier emits fp32 logits, so the downstream topk is identical whichever fires.
    """
    on_cuda = gate_weight.is_cuda
    low_prec = gate_weight.dtype in (torch.bfloat16, torch.float16)
    m = x.shape[0]
    k, n = gate_weight.shape[1], gate_weight.shape[0]

    # tier 1: CuteDSL low-latency bf16 GEMM (small-M decode).
    if _LL_BF16_GEMM is not None and on_cuda and low_prec and m <= 16 and k % 8 == 0:
        return _LL_BF16_GEMM(x, gate_weight, torch.float32)

    # tier 2: vllm's compiled fp32 router kernel — opportunistic; needs its op
    # present, an fp32 weight, and one of the shapes it was instantiated for.
    if (
        _fp32_router_op_available()
        and on_cuda
        and gate_weight.dtype == torch.float32
        and m <= 32
        and (k, n) in _FP32_ROUTER_SHAPES
    ):
        out = torch.empty(m, n, device=x.device, dtype=torch.float32)
        torch.ops._C.fp32_router_gemm(out, x, gate_weight)
        return out

    # tier 3: CuteDSL bf16x3 (SM100).
    if _BF16X3_GEMM is not None and on_cuda and low_prec:
        return _BF16X3_GEMM(x, gate_weight)

    # tier 4: cuBLAS bf16 x bf16 -> fp32 (one tensor-core GEMM, fp32 epilogue).
    if on_cuda and low_prec:
        x_gemm = x if x.dtype == gate_weight.dtype else x.to(gate_weight.dtype)
        return torch.mm(x_gemm, gate_weight.t(), out_dtype=torch.float32)

    # tier 5: fp32 fallback (CPU, or a non-bf16/fp16 weight).
    return F.linear(x.float(), gate_weight.float())


class RoutingMethodType(Enum):
    """The routing families ``topk_method`` selects, sglang-style.

    ``GREEDY`` is plain post-softmax top-k (Qwen3-MoE, DeepSeek-V2-Lite);
    ``GROUPED_TOPK`` and ``NOAUX_TC`` both go through :func:`grouped_topk` (the
    group-limited selection and the biased ``noaux_tc`` routing DeepSeek ships).
    """

    GREEDY = "greedy"
    GROUPED_TOPK = "group_limited_greedy"
    NOAUX_TC = "noaux_tc"

    @classmethod
    def from_topk_method(cls, topk_method: str) -> RoutingMethodType:
        try:
            return cls(topk_method)
        except ValueError:
            return cls.GREEDY


class TopKOutput(NamedTuple):
    """The router's product (mirrors sglang's ``StandardTopKOutput``).

    Attributes:
        topk_weights: ``[tokens, top_k]`` routing weights, in the model dtype.
        topk_ids: ``[tokens, top_k]`` global expert ids.
        router_logits: ``[tokens, num_experts]`` fp32 logits (kept for probes).
    """

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    router_logits: torch.Tensor | None = None


class TopKRouter:
    """Stage 1: router GEMM + top-k selection into a :class:`TopKOutput`.

    Stateless apart from the routing hyper-parameters read off the config; the
    ``gate_weight`` (and the optional ``noaux_tc`` correction bias) live on the
    owning block and are passed in per call, so checkpoint loading and the
    ``isinstance(module, SparseMoeBlock)`` weight scan are unaffected.
    """

    def __init__(
        self,
        top_k: int,
        norm_topk_prob: bool,
        routed_scaling_factor: float,
        scoring_func: str,
        topk_method: str,
        n_group: int,
        topk_group: int,
    ) -> None:
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.routing_method = RoutingMethodType.from_topk_method(topk_method)
        self.n_group = n_group
        self.topk_group = topk_group
        # One-launch softmax+topk+renorm for the GREEDY family (six aten
        # kernels collapse into one); ``=0`` restores the reference chain.
        self._fused_topk = os.environ.get("RAPID_LLM_FUSED_ROUTER", "1") != "0"

    def forward(
        self,
        x: torch.Tensor,
        gate_weight: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
    ) -> TopKOutput:
        """Compute per-token expert ids and weights (HF-compatible ordering).

        Returns a :class:`TopKOutput`; ``weights`` are in ``x.dtype``.
        """
        # fp32 logits, as DeepSeek's router and qwen3's reference semantics require: a
        # bf16/fp16 output can flip a topk pick on near-ties, and a wrong expert costs far
        # more than the precision. The GEMM runs through the tiered dispatch above.
        router_logits = _router_gemm(x, gate_weight)
        if self.routing_method in (RoutingMethodType.NOAUX_TC, RoutingMethodType.GROUPED_TOPK):
            weights, ids = grouped_topk(
                router_logits,
                top_k=self.top_k,
                renormalize=self.norm_topk_prob,
                num_expert_group=self.n_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=correction_bias,
            )
            return TopKOutput(weights.to(x.dtype), ids, router_logits)
        if self._fused_topk and router_logits.is_cuda and router_logits.dtype == torch.float32:
            weights, ids = fused_topk_softmax(
                router_logits,
                self.top_k,
                renormalize=self.norm_topk_prob,
                routed_scaling_factor=self.routed_scaling_factor,
                out_dtype=x.dtype,
            )
            return TopKOutput(weights, ids, router_logits)
        # fp32 softmax over the full expert set — topk must come after softmax.
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # The scale widens the routed half only (the shared expert is added unscaled in
        # ``forward``); qwen3_moe leaves the factor at 1.0.
        routing_weights = routing_weights * self.routed_scaling_factor
        return TopKOutput(routing_weights.to(x.dtype), selected_experts, router_logits)
