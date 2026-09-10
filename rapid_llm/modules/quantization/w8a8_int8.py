"""W8A8 int8: SmoothQuant — int8 weights + dynamic per-token int8 activations.

:class:`W8A8Int8Config` keeps weights per-channel int8; the methods
quantise activations per token and call the smoothquant GEMM with both
scales applied in the epilogue.

Usage:
    quant = W8A8Int8Config(ignored)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
    allocate_expert_weights,
    allocate_linear_weights,
    run_quant_linear,
)
from .parameter import RawParameter
from .utils import quantize_int8_per_channel

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class W8A8Int8Config(QuantizationConfig):
    """SmoothQuant W8A8: per-channel int8 weights + dynamic per-token int8 activations.

    Used by ``--quantization smoothquant``.
    """

    is_dynamic: bool = True

    def __init__(self, ignored: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.group_n = 1
        self.group_k = 1 << 30
        self.ignored = ignored

    def get_name(self) -> str:
        return "w8a8_int8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75  # int8 tensor cores from Turing

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W8A8Int8Config:
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(ignored=ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, W8A8Int8LinearMethod, W8A8Int8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.int8


class W8A8Int8LinearMethod(LinearMethodBase):
    """SmoothQuant: int8 weights (per-channel) + int8 activations (per-token dynamic)."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        allocate_linear_weights(layer, input_size, output_size)

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return run_quant_linear(
            "w8a8_int8",
            x,
            layer.weight,
            weight_scale=layer.weight_scale_inv,
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        qweight, scale = quantize_int8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)


class W8A8Int8MoEMethod(FusedMoEMethodBase):
    """SmoothQuant stacked experts: int8 weights + per-token int8 activations through grouped GEMM."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        return allocate_expert_weights(block)

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe_w8a8_int8

        # The W8A8 entry point, not the weight-only ``fused_moe``: both store
        # int8 experts with per-channel scales, so the dtype cannot tell the
        # modes apart — only this one quantises the activation.
        return fused_moe_w8a8_int8(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale_inv"],
            w2_scale=block.experts["down_proj_scale_inv"],
            group_n=1,
            group_k=max(block.hidden_size, block.moe_intermediate_size),
            # V4's bounded SwiGLU rides inside the activation epilogue; every
            # other family's blocks have no attribute and keep plain silu.
            swiglu_limit=float(getattr(block, "swiglu_limit", float("inf"))),
        )

    def quantize_from_fp16(self, block: nn.Module, config: QuantizationConfig) -> None:
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = quantize_int8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
