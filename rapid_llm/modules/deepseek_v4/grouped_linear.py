"""Grouped low-rank output projection for DeepSeek-V4.

``o_a_proj`` is block-diagonal: the attention's stacked output splits into
``o_groups`` contiguous groups, each projected independently to a
``o_lora_rank``-wide slice, and ``o_b_proj`` then mixes the concatenation.
Tensor parallelism keeps groups whole — each rank owns ``o_groups // world``
complete groups, which requires ``o_groups % world == 0`` (checked by the
attention). The head split and the group split then provably coincide: rank
``r``'s local heads are exactly its groups' inputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from ..quantization import QuantizationConfig, RawParameter


class DeepseekV4GroupedLinear(nn.Module):
    """Block-diagonal grouped linear for ``o_a_proj``.

    The attention's stacked output (``num_heads * head_dim`` wide) splits into
    ``o_groups`` contiguous groups, each projected independently to a
    ``o_lora_rank``-wide slice; ``o_b_proj`` then mixes the concatenation.
    Tensor parallelism keeps groups whole — each rank owns
    ``o_groups // world`` complete groups, which requires
    ``o_groups % world == 0`` (checked by the attention). The head split and
    the group split then provably coincide: rank ``r``'s local heads are
    exactly its groups' inputs.

    With ``quant`` the weight is stored blockwise-fp8 (``[out, in]`` uint8 plus
    a scale grid); each group runs its own w8a16 GEMM over the rows it owns.
    Group boundaries are ``o_lora_rank``-wide — a multiple of the 128 block —
    so every group's weight rows and scale rows are self-contained.
    """

    def __init__(
        self,
        in_per_group: int,
        out_features: int,
        n_groups: int,
        dtype: torch.dtype,
        quant: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.in_per_group = in_per_group
        self.quant = quant
        if quant is None:
            self.weight = nn.Parameter(torch.empty(out_features, in_per_group, dtype=dtype))
        else:
            # Blockwise fp8 storage mirrors Fp8LinearMethod's layout; the
            # grouped apply below is what makes this module instead of a plain
            # linear.
            self.weight = RawParameter(
                torch.empty(out_features, in_per_group, dtype=quant.storage_dtype)
            )
            self.weight_scale_inv = RawParameter(
                torch.empty(*quant.scale_shape(out_features, in_per_group), dtype=torch.float32)
            )
            self.weight_scale_inv.weight_loader = self._loader
        self.weight.weight_loader = self._loader

    @staticmethod
    def _loader(param, loaded, shard_id) -> torch.Tensor:
        """Fill the local groups' rows from the full grouped weight."""
        world = get_tensor_model_parallel_world_size()
        if world == 1:
            if param.shape != loaded.shape:
                raise ValueError(
                    f"grouped weight of shape {tuple(loaded.shape)} does not fit "
                    f"parameter of shape {tuple(param.shape)}"
                )
            param.data.copy_(loaded)
            return param.data
        rows = param.shape[0]
        rank = get_tensor_model_parallel_rank()
        loaded = loaded.narrow(0, rank * rows, rows)
        if param.shape != loaded.shape:
            raise ValueError(
                f"grouped weight of shape {tuple(loaded.shape)} does not fit "
                f"parameter of shape {tuple(param.shape)}"
            )
        param.data.copy_(loaded)
        return param.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[..., n_groups, in_per_group] -> [..., n_groups, out_per_group]``."""
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        if hidden_dim != self.in_per_group:
            raise ValueError(f"grouped linear input width {hidden_dim} != {self.in_per_group}")
        if self.quant is None:
            w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
            x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
            y = torch.bmm(x, w).transpose(0, 1)
            return y.reshape(*input_shape, self.n_groups, -1)
        # fp8: per-group w8a16 GEMMs over each group's own weight rows. The
        # dispatcher caches per (scheme, shape), so the per-group calls share
        # one kernel selection.
        from ..quantization.base_config import run_quant_linear

        out_per_group = self.weight.shape[0] // self.n_groups
        scale_rows = self.weight_scale_inv.shape[0] // self.n_groups
        group_k = min(self.quant.group_k, hidden_dim)
        x = x.reshape(-1, self.n_groups, hidden_dim)
        outs = []
        for g in range(self.n_groups):
            w_g = self.weight.narrow(0, g * out_per_group, out_per_group)
            s_g = self.weight_scale_inv.narrow(0, g * scale_rows, scale_rows)
            outs.append(
                run_quant_linear(
                    "fp8",
                    x[:, g],
                    w_g,
                    weight_scale=s_g,
                    group_n=self.quant.group_n,
                    group_k=group_k,
                )
            )
        y = torch.stack(outs, dim=1)
        return y.reshape(*input_shape, self.n_groups, out_per_group)


__all__ = ["DeepseekV4GroupedLinear"]
