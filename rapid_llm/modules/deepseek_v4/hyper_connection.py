"""Manifold-constrained hyper-connections (mHC) for DeepSeek-V4.

The mHC residual turns each block into a stream mixer over ``hc_mult``
parallel residual streams rather than a single one: a learned projection
produces collapse / placement / mixing weights, and the mixing matrix is
projected onto the doubly-stochastic manifold by Sinkhorn iteration. Both
modules here run their projection in strict fp32 to stay on the reference
trajectory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.config import ModelConfig
from ..quantization import RawParameter
from .norm import DeepseekV4UnweightedRMSNorm


class DeepseekV4HyperConnection(nn.Module):
    """Learned (fn, base, scale) turning ``hc_mult`` residual streams into
    collapse / placement / mixing weights.

    ``fn``/``base``/``scale`` stay fp32 (:class:`RawParameter`): the Sinkhorn
    projection runs in float and transformers keeps the whole module in its
    strict-fp32 list, so a bf16 cast would silently move the doubly-stochastic
    projection off the reference trajectory.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hc = config.hc_mult
        self.hc_mult = hc
        self.hc_sinkhorn_iters = int(getattr(config, "hc_sinkhorn_iters", 20))
        self.hc_eps = float(getattr(config, "hc_eps", 1e-6))
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + hc) * hc
        self.fn = RawParameter(torch.empty(mix, hc * config.hidden_size, dtype=torch.float32))
        self.base = RawParameter(torch.empty(mix, dtype=torch.float32))
        self.scale = RawParameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        mix = F.linear(flat, self.fn.float())
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        hc = self.hc_mult
        # pre/post/comb follow the reference's exact forms: pre in (0,1)+eps,
        # post in (0,2) with no eps (the reference's ``2 * sigmoid``), comb
        # softmaxed row-wise then Sinkhorn-projected — the softmax start plus
        # a leading column normalisation, then ``iters - 1`` row/column pairs.
        pre = torch.sigmoid(mix[..., :hc] * pre_scale + self.base[:hc]) + self.hc_eps
        post = 2 * torch.sigmoid(mix[..., hc : 2 * hc] * post_scale + self.base[hc : 2 * hc])
        comb_logits = mix[..., 2 * hc :].view(*mix.shape[:-1], hc, hc) * comb_scale + self.base[
            2 * hc :
        ].view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    """Final stream collapse before the shared RMSNorm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hc = config.hc_mult
        self.hc_mult = hc
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = float(getattr(config, "hc_eps", 1e-6))
        self.hc_fn = RawParameter(torch.empty(hc, hc * config.hidden_size, dtype=torch.float32))
        self.hc_base = RawParameter(torch.empty(hc, dtype=torch.float32))
        self.hc_scale = RawParameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(x.flatten(2).float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


__all__ = [
    "DeepseekV4HyperConnection",
    "DeepseekV4HyperHead",
]
