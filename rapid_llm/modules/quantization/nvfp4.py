"""NVFP4 config and method: 4-bit e2m1 weights with two levels of scale.

NVIDIA ModelOpt's ``modelopt_fp4`` / TensorRT-LLM's NVFP4:

* ``weight``       ``[N, K // 2]`` uint8 — two e2m1 nibbles per byte, low nibble
  at the even k index;
* ``weight_scale`` ``[N, K // 16]`` uint8 — one fp8-e4m3 bit pattern per 16
  consecutive k elements;
* ``weight_global_scale`` one fp32 element — brings the block scales into e4m3
  range.

Weight-only, permanently so on this hardware: sm90 has no fp4 MMA, so
activations stay 16-bit and the win is bytes, not FLOPs — lower decode latency
and a smaller resident model (:mod:`rapid_llm.kernels.ops.quantization.nvfp4`
has the format story). MoE experts are not implemented — the fused grouped
GEMM would need its own two-level unpacking kernel — so
:meth:`NVFP4Config.get_quant_method` raises rather than handing a MoE block a
linear method it cannot use.
"""

from __future__ import annotations

from math import lcm
from typing import Any

import torch
import torch.nn as nn

from .base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
    run_quant_linear,
)
from .parameter import RawParameter

#: Weight elements per byte.
_PACK_FACTOR = 2

#: NVFP4's fixed block-scale width. This is format metadata, not a kernel
#: implementation detail: configuration must be readable on CPU-only installs
#: where Triton is intentionally absent.
_NVFP4_BLOCK = 16

#: Smallest k-shard splitting neither a byte nor a block scale.
_SHARD_GRANULARITY = lcm(_PACK_FACTOR, _NVFP4_BLOCK)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class NVFP4Config(QuantizationConfig):
    """NVFP4 checkpoint/runtime config: e2m1 weights, e4m3 block scales.

    ``group_k`` is 16 and not configurable — the block length is part of the
    format, not a checkpoint choice, so ``from_config`` has nothing to read.
    """

    def __init__(self, ignored: tuple[str, ...] = ()) -> None:
        super().__init__()
        self.group_n = 1
        self.group_k = _NVFP4_BLOCK
        self.ignored = ignored
        self.method = "nvfp4"

    def get_name(self) -> str:
        return "nvfp4"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # Weight-only: the activation keeps its own dtype, so both 16-bit types
        # run through the one kernel.
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Nothing here uses an fp4/fp8 tensor-core instruction, so 89 is not a
        # hard requirement; it is the floor the e4m3 block-scale decode inherits
        # from the shared w8a16 bit trick, kept at 89 to avoid claiming an
        # untested capability.
        return 89

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> NVFP4Config:
        bits = int(config.get("bits", 4))
        if bits != 4:
            raise ValueError(f"only 4-bit NVFP4 is supported, got {bits}")
        group_size = int(config.get("group_size", _NVFP4_BLOCK))
        if group_size != _NVFP4_BLOCK:
            raise ValueError(
                f"NVFP4 block size is fixed at {_NVFP4_BLOCK} by the format, "
                f"checkpoint declares {group_size}"
            )
        ignored = tuple(config.get("modules_to_not_convert") or ())
        return cls(ignored)

    def get_quant_method(self, layer: nn.Module, prefix: str = "") -> QuantizeMethodBase | None:
        from ..moe import SparseMoeBlock
        from .unquant import UnquantizedFusedMoEMethod

        if isinstance(layer, SparseMoeBlock) and self.quantizes(prefix):
            raise NotImplementedError(
                "NVFP4 MoE experts are not implemented; the fused grouped GEMM "
                "has no two-level unpacking path. Use --quantization fp8 or int4 "
                f"for MoE models, or add {prefix!r} to modules_to_not_convert."
            )
        # UnquantizedFusedMoEMethod is reachable only for an ignored MoE prefix,
        # which the branch above deliberately lets through.
        return self._dispatch(layer, prefix, NVFP4LinearMethod, UnquantizedFusedMoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8

    def shard_is_aligned(self, size: int) -> bool:
        """Whether a TP shard of ``size`` channels keeps whole bytes and blocks.

        The k axis carries both the packing (2 elements per byte) and the block
        scales (16), so the granularity is their lcm — 16, not 32; 32 would
        reject k=4864, the ``down_proj`` shard Qwen3-4B gets under TP2. The n
        axis is unconstrained: nothing is packed along it.
        """
        return size % _SHARD_GRANULARITY == 0


# --------------------------------------------------------------------------- #
# Linear method
# --------------------------------------------------------------------------- #
class NVFP4LinearMethod(LinearMethodBase):
    """NVFP4 weight-only linear; runs ``native/linear_nvfp4``."""

    def create_weights(self, layer: nn.Module, input_size: int, output_size: int, **kw) -> None:
        block = _NVFP4_BLOCK
        if input_size % block != 0:
            raise ValueError(f"NVFP4 needs in_features divisible by {block}, got {input_size}")
        config: NVFP4Config = layer.quant  # type: ignore[assignment]
        layer.weight = RawParameter(
            torch.empty(output_size, input_size // _PACK_FACTOR, dtype=torch.uint8)
        )
        # uint8, not float8_e4m3fn: the kernel bit-shifts these bytes, and
        # RawParameter keeps the loader from casting them to the activation
        # dtype on the way in.
        layer.weight_scale = RawParameter(
            torch.empty(*config.scale_shape(output_size, input_size), dtype=torch.uint8)
        )
        layer.weight_global_scale = RawParameter(torch.empty(1, dtype=torch.float32))

    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return run_quant_linear(
            "nvfp4",
            x,
            layer.weight,
            weight_scale=layer.weight_scale,
            weight_global_scale=layer.weight_global_scale,
            bias=bias,
        )

    def quantize_from_fp16(self, layer: nn.Module, config: QuantizationConfig) -> None:
        if layer.weight.device.type == "cpu":
            raise NotImplementedError("NVFP4 runtime quantization requires CUDA; use int4 on CPU")
        from ...kernels.ops.quantization import quantize_nvfp4_blockwise

        packed, block_scale, global_scale = quantize_nvfp4_blockwise(layer.weight.data)
        layer.weight = RawParameter(packed)
        layer.weight_scale = RawParameter(block_scale)
        layer.weight_global_scale = RawParameter(global_scale)
