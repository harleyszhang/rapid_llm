"""W8A8 fp8: fp8-e4m3 weights + dynamic per-token fp8 activations.

:class:`W8A8Fp8Config` pairs block-quantised weights with dynamic
per-token activation quant; the linear/MoE methods call the true W8A8
fp8 GEMM — both operands really are fp8 at run time.

Usage:
    quant = W8A8Fp8Config(group_n, group_k, ignored)
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
from .utils import quantize_fp8_per_channel

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class W8A8Fp8Config(QuantizationConfig):
    """True W8A8 with fp8-e4m3: weights per-channel, activations per-token.

    Used by ``--quantization fp8`` (runtime path: converts fp16 checkpoint to
    fp8 per-channel weights on the fly).
    """

    is_dynamic: bool = True

    def __init__(
        self, group_n: int = 1, group_k: int = 1 << 30, ignored: tuple[str, ...] = ()
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k
        self.ignored = ignored

    def get_name(self) -> str:
        return "w8a8_fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> W8A8Fp8Config:
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(ignored=ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, W8A8Fp8LinearMethod, W8A8Fp8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8


class W8A8Fp8LinearMethod(LinearMethodBase):
    """fp8-e4m3 weights + per-token fp8-e4m3 activations (no calibration)."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        allocate_linear_weights(layer, input_size, output_size)

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        config: W8A8Fp8Config = layer.quant  # type: ignore[assignment]
        # per-token activation quantisation happens inside the selected impl
        return run_quant_linear(
            "w8a8_fp8",
            x,
            layer.weight,
            weight_scale=layer.weight_scale_inv,
            group_n=config.group_n,
            group_k=min(config.group_k, layer.input_size),
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        qweight, scale = quantize_fp8_per_channel(layer.weight.data)
        layer.weight = RawParameter(qweight)
        layer.weight_scale_inv = RawParameter(scale)


class W8A8Fp8MoEMethod(FusedMoEMethodBase):
    """W8A8 fp8 stacked experts: fp8 weights + per-token fp8 activations.

    Weights are byte-identical to :class:`~.fp8.Fp8MoEMethod`'s; only the entry
    point differs — ``fused_moe_w8a8_fp8`` quantises the activation and runs the
    fp8 tensor cores, while ``fused_moe`` widens the expert tile to bf16 and
    leaves the activation alone.
    """

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        return allocate_expert_weights(block)

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe_w8a8_fp8

        return fused_moe_w8a8_fp8(
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
            qweight, scale = quantize_fp8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
