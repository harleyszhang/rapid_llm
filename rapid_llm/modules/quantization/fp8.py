"""Fp8 weight-only config (fp8-e4m3 weights, fp16 activations).

:class:`Fp8Config` marks checkpoint-stored fp8 weights;
:class:`Fp8LinearMethod` dequantises per block inside the w8a16 kernel,
so no fp8 arithmetic happens at run time.

Usage:
    quant = Fp8Config(group_n, group_k, ignored)
"""

from __future__ import annotations

import os
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

#: Block size of the fine-grained FP8 format (Qwen/DeepSeek checkpoints).
FP8_BLOCK = 128

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class Fp8Config(QuantizationConfig):
    """fp8-e4m3 weight-only (W8A16) with block-wise or per-channel scales."""

    def __init__(
        self,
        group_n: int = FP8_BLOCK,
        group_k: int = FP8_BLOCK,
        ignored: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k
        self.ignored = ignored
        self.method = ""

    def get_name(self) -> str:
        return "fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89  # Ada / Hopper for native fp8; Ampere via software

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8Config:
        fmt = str(config.get("fmt", "e4m3")).lower()
        if fmt != "e4m3":
            raise ValueError(f"unsupported fp8 format {fmt!r}; only e4m3 is implemented")
        block = config.get("weight_block_size") or [FP8_BLOCK, FP8_BLOCK]
        gn, gk = int(block[0]), int(block[1])
        if gk % FP8_BLOCK != 0 or gn % FP8_BLOCK != 0:
            raise ValueError(
                f"weight_block_size {block} is not a multiple of {FP8_BLOCK}; "
                "the w8a16 kernel tiles k in 128-wide steps"
            )
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(gn, gk, ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        return self._dispatch(layer, prefix, Fp8LinearMethod, Fp8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8

    @property
    def is_fp8(self) -> bool:
        return True


class Fp8LinearMethod(LinearMethodBase):
    """fp8-e4m3 weight + fp16 activation; per-channel or block-wise scale grid.

    Post-load the weight is dequantised back to the activation dtype and the
    layer becomes a plain cuBLAS GEMM: at decode/prefill shapes the w8a16
    Triton kernel that widens fp8 tiles in-register runs ~3x off the bf16
    bandwidth floor (H100: 19.1us vs 6.0us at M=16, 69.2us vs 19.4us at
    M=1024), while the dense projections are ~3% of an MoE checkpoint, so the
    widened copy costs little memory. ``RAPID_FP8_LINEAR_DEQUANT=0`` keeps the
    fp8 storage + w8a16 kernel (e.g. when a dense fp8 model must fit).
    """

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        allocate_linear_weights(layer, input_size, output_size)

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if getattr(layer, "weight_scale_inv", None) is None:
            # Widened at load: plain GEMM in the activation dtype.
            return torch.nn.functional.linear(x, layer.weight, bias)
        config: Fp8Config = layer.quant  # type: ignore[assignment]
        return run_quant_linear(
            "fp8",
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
        # The ``--quantization fp8`` path exists to *save* memory; undoing it
        # in the post-load hook would defeat the request.
        layer._fp8_keep_quantized = True

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Swap the fp8 checkpoint weight for its widened form, once.

        e4m3 widens to the 16-bit activation dtype exactly (3 mantissa bits
        fit in bf16's 8 / fp16's 10), so the only rounding is the fp32 scale
        multiply -- the same rounding the w8a16 kernel applies per tile. Runs
        while parameters sit on the load device.
        """
        if os.environ.get("RAPID_FP8_LINEAR_DEQUANT", "1") == "0":
            return
        if getattr(layer, "_fp8_keep_quantized", False):
            return
        w = layer.weight
        if w.dtype not in (torch.uint8, torch.float8_e4m3fn):
            return
        config: Fp8Config = layer.quant  # type: ignore[assignment]
        scale = layer.weight_scale_inv
        if w.dtype == torch.uint8:
            wf = w.view(torch.float8_e4m3fn).float()
        else:
            wf = w.float()
        n, k = wf.shape
        if scale.numel() == n:
            wf = wf * scale.reshape(n, 1)
        else:
            full = (
                scale.reshape(-1, scale.shape[-1])
                .repeat_interleave(config.group_n, 0)
                .repeat_interleave(config.group_k, 1)
            )
            wf = wf * full[:n, :k]
        layer.weight = RawParameter(wf.to(layer.dtype))
        layer.weight_scale_inv = None


class Fp8MoEMethod(FusedMoEMethodBase):
    """fp8 stacked experts (checkpoint), fp16 activations through grouped GEMM."""

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        return allocate_expert_weights(block)

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe

        config: Fp8Config = block.quant  # type: ignore[assignment]
        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale_inv"],
            w2_scale=block.experts["down_proj_scale_inv"],
            group_n=config.group_n,
            group_k=min(config.group_k, max(block.hidden_size, block.moe_intermediate_size)),
        )

    def quantize_from_fp16(self, block: nn.Module, config: QuantizationConfig) -> None:
        for name in ("gate_up_proj", "down_proj"):
            qweight, scale = quantize_fp8_per_channel(block.experts[name].data)
            block.experts[name] = RawParameter(qweight)
            block.experts[f"{name}_scale_inv"] = RawParameter(scale)
