"""MXFP4 routed experts and the DeepSeek-V4 fp8+mxfp4 checkpoint config.

DeepSeek-V4 checkpoints store routed experts as e2m1 nibbles packed two per
byte (even K position in the low nibble) with per-32-element e8m0 scales,
while every linear projection — attention, indexer, shared experts — is
blockwise fp8-e4m3 with e8m0 scales. :class:`Mxfp4MoEMethod` runs the routed
half through the mxfp4 branch of the fused grouped-GEMM kernel;
:class:`DeepseekV4Fp8Config` is the one config both halves dispatch through.

Usage:
    quant = DeepseekV4Fp8Config.from_config(checkpoint["quantization_config"])
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ...utils.torch_compat import is_float8_e8m0fnu
from .base_config import (
    FusedMoEMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
    expert_scale_parameter,
)
from .fp8 import FP8_BLOCK, Fp8LinearMethod, Fp8MoEMethod
from .parameter import RawParameter

#: e2m1 code points are 32 elements wide (OCP MX); the fused kernel's mxfp4
#: branch is hard-wired to this group size.
MXFP4_GROUP = 32


def repack_mxfp4_pairs(packed: torch.Tensor) -> torch.Tensor:
    """``[..., K//2]`` byte-packed e2m1 pairs -> ``[..., K//8]`` int32 words.

    The checkpoint stores two nibbles per byte — even K position in the low
    nibble, odd in the high (the packing order vLLM's MXFP4 quantiser emits) —
    while the fused kernel consumes 8 nibbles per int32 word with K ascending
    across the shifts. Four little-endian bytes per word makes the two orders
    coincide: word ``j`` covers K ``[8j, 8j+8)``, shift ``4*i`` reading K
    ``8j + i``.
    """
    b = packed.view(torch.uint8)
    if b.shape[-1] % 4 != 0:
        raise ValueError(
            f"mxfp4 packed weight's last dim ({b.shape[-1]}) is not a multiple "
            "of 4 bytes (32 nibbles); the repack cannot form whole int32 words"
        )
    return (
        b[..., 0::4].to(torch.int64)
        | (b[..., 1::4].to(torch.int64) << 8)
        | (b[..., 2::4].to(torch.int64) << 16)
        | (b[..., 3::4].to(torch.int64) << 24)
    ).to(torch.int32)


def e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """e8m0 scale table -> fp32 (each byte is ``2 ** (x - 127)``)."""
    if is_float8_e8m0fnu(scale.dtype):
        return scale.to(torch.float32)
    if scale.dtype == torch.uint8:
        return torch.exp2(scale.to(torch.int32) - 127)
    if scale.is_floating_point():
        return scale.to(torch.float32)
    raise ValueError(f"e8m0 scale table has unexpected dtype {scale.dtype}")


#: e2m1 code points indexed by nibble (bit 3 sign, bits [2:1] exponent, bit 0
#: mantissa) — the table the fused kernel decodes on device.
_E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``[..., K//2]`` byte-packed e2m1 + ``[..., K//32]`` e8m0 scales -> fp32.

    The host-side inverse of the checkpoint packing: even K position in the
    low nibble, odd in the high — the order :func:`repack_mxfp4_pairs`
    documents. Shapes are whole 32-element groups (:data:`MXFP4_GROUP`).
    """
    b = packed.view(torch.uint8).to(torch.long)
    vals = torch.empty(*b.shape[:-1], b.shape[-1] * 2, dtype=torch.float32)
    vals[..., 0::2] = _E2M1_VALUES[b & 0xF]
    vals[..., 1::2] = _E2M1_VALUES[b >> 4]
    return vals * e8m0_to_fp32(scale).repeat_interleave(MXFP4_GROUP, dim=-1)


class Mxfp4MoEMethod(FusedMoEMethodBase):
    """Routed experts as int32-packed e2m1 with per-32 fp32 scales.

    Parameters follow the fp8 expert naming (``gate_up_proj`` /
    ``down_proj`` plus ``_scale_inv`` twins) so the checkpoint-key translation
    and the per-expert stacking loader treat both formats identically; only
    the storage layout differs — ``[E, N, K//8]`` int32 weights and
    ``[E, N, K//32]`` fp32 scales.
    """

    def create_weights(self, block: nn.Module) -> dict[str, nn.Parameter]:
        num_experts = getattr(block, "num_local_experts", block.num_experts)
        gate_up_n, gate_up_k = 2 * block.moe_intermediate_size, block.hidden_size
        down_n, down_k = block.hidden_size, block.moe_intermediate_size
        return {
            "gate_up_proj": RawParameter(
                torch.empty(num_experts, gate_up_n, gate_up_k // 8, dtype=torch.int32)
            ),
            "gate_up_proj_scale_inv": expert_scale_parameter(
                num_experts, (gate_up_n, gate_up_k // MXFP4_GROUP)
            ),
            "down_proj": RawParameter(
                torch.empty(num_experts, down_n, down_k // 8, dtype=torch.int32)
            ),
            "down_proj_scale_inv": expert_scale_parameter(
                num_experts, (down_n, down_k // MXFP4_GROUP)
            ),
        }

    def apply(self, block, x, topk_weights, topk_ids) -> torch.Tensor:
        from ...kernels import fused_moe

        return fused_moe(
            x,
            block.experts["gate_up_proj"],
            block.experts["down_proj"],
            topk_weights,
            topk_ids,
            w1_scale=block.experts["gate_up_proj_scale_inv"],
            w2_scale=block.experts["down_proj_scale_inv"],
            group_n=1,
            group_k=MXFP4_GROUP,
            # V4's bounded SwiGLU rides inside the activation epilogue; every
            # other family's blocks have no attribute and keep plain silu.
            swiglu_limit=float(getattr(block, "swiglu_limit", float("inf"))),
            mxfp4=True,
        )


class DeepseekV4Fp8Config(QuantizationConfig):
    """DeepSeek-V4 fp8 checkpoint: blockwise-fp8 linears, fp4 (or fp8) experts.

    Attributes:
        expert_dtype: ``"fp4"`` routes the MoE through
            :class:`Mxfp4MoEMethod`; ``"fp8"`` (the Flash-Base layout) is not
            implemented — its experts would need an fp8 grouped-GEMM variant
            carrying V4's bounded-SwiGLU epilogue.
    """

    def __init__(
        self,
        group_n: int = FP8_BLOCK,
        group_k: int = FP8_BLOCK,
        ignored: tuple[str, ...] = (),
        expert_dtype: str = "fp4",
    ) -> None:
        super().__init__()
        self.group_n = group_n
        self.group_k = group_k
        self.ignored = ignored
        self.expert_dtype = expert_dtype
        self.method = ""

    def get_name(self) -> str:
        return "deepseek_v4_fp8"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # The mxfp4 branch decodes e2m1 in the kernel and fp8 runs w8a16 —
        # both are integer/byte storage with 16-bit arithmetic, so Ampere works.
        return 80

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DeepseekV4Fp8Config:
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
        from ..moe import SparseMoeBlock

        if isinstance(layer, SparseMoeBlock):
            if self.expert_dtype == "fp4":
                return Mxfp4MoEMethod()
            raise NotImplementedError(
                "DeepSeek-V4 fp8-expert checkpoints (expert_dtype=fp8, the "
                "Flash-Base layout) are not supported; the fp4-expert Flash "
                "checkpoints are"
            )
        # The SparseMoeBlock case is handled above, so the routed half of the
        # dispatch pair is unreachable — Fp8MoEMethod stands in for completeness.
        return self._dispatch(layer, prefix, Fp8LinearMethod, Fp8MoEMethod)

    @property
    def storage_dtype(self) -> torch.dtype:
        return torch.uint8

    @property
    def is_fp8(self) -> bool:
        return True
