"""Tests for the quant-method strategies and the runtime quantisation schemes.

Registry lookups, checkpoint-method dispatch (int4 variants), weight
creation shapes for every scheme, and rejection paths — the strategy
layer without kernels.

Usage:
    pytest tests/models/test_quant_methods.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rapid_llm.modules.linear import ReplicatedLinear
from rapid_llm.modules.moe import SparseMoeBlock
from rapid_llm.modules.quantization import (
    QuantizationConfig,
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from rapid_llm.modules.quantization.awq import AWQLinearMethod
from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8LinearMethod, BlockInt8MoEMethod
from rapid_llm.modules.quantization.fp8 import Fp8LinearMethod, Fp8MoEMethod
from rapid_llm.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_int8_groupwise,
)
from rapid_llm.modules.quantization.w8a8_fp8 import W8A8Fp8LinearMethod
from rapid_llm.modules.quantization.w8a8_int8 import W8A8Int8LinearMethod
from tests.conftest import needs_capability
from tests.registry import import_needs_cuda

import_needs_cuda()


class _StubMoeBlock(SparseMoeBlock):
    """Minimum surface a MoeQuantMethod touches: sizes, ``quant``, ``experts``."""

    def __init__(
        self,
        num_experts: int = 4,
        hidden_size: int = 256,
        moe_intermediate_size: int = 128,
        quant: QuantizationConfig | None = None,
    ) -> None:
        # Bypass SparseMoeBlock.__init__ which requires a full ModelConfig
        nn.Module.__init__(self)
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.quant = quant


def _fill_fp16(layer: ReplicatedLinear, scale: float = 0.05) -> torch.Tensor:
    """Give an fp16 layer a known weight and return a float copy of it."""
    with torch.no_grad():
        layer.weight.copy_(torch.randn_like(layer.weight, dtype=torch.float32) * scale)
    return layer.weight.data.float().clone()


# --------------------------------------------------------------------------- #
# Registry: config.get_quant_method() dispatch
# --------------------------------------------------------------------------- #
def test_linear_method_registry():
    from rapid_llm.modules.quantization.awq import AWQConfig
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config
    from rapid_llm.modules.quantization.fp8 import Fp8Config
    from rapid_llm.modules.quantization.w8a8_int8 import W8A8Int8Config

    layer = ReplicatedLinear(64, 128)
    assert isinstance(Fp8Config(128, 128).get_quant_method(layer), Fp8LinearMethod)
    assert isinstance(BlockInt8Config.per_channel().get_quant_method(layer), BlockInt8LinearMethod)
    assert isinstance(W8A8Int8Config().get_quant_method(layer), W8A8Int8LinearMethod)
    assert isinstance(AWQConfig().get_quant_method(layer), AWQLinearMethod)


def test_int4_method_dispatches_on_checkpoint_method():
    """AWQ/GPTQ checkpoints get their named method classes."""
    from rapid_llm.modules.quantization.awq import AWQConfig
    from rapid_llm.modules.quantization.gptq import GPTQConfig, GPTQLinearMethod

    layer = ReplicatedLinear(64, 128)
    assert isinstance(AWQConfig().get_quant_method(layer), AWQLinearMethod)
    assert isinstance(GPTQConfig().get_quant_method(layer), GPTQLinearMethod)


def test_moe_method_registry():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config
    from rapid_llm.modules.quantization.fp8 import Fp8Config

    # Use a stub that inherits from SparseMoeBlock for isinstance check
    block = _StubMoeBlock()
    assert isinstance(Fp8Config(128, 128).get_quant_method(block), Fp8MoEMethod)
    assert isinstance(BlockInt8Config.per_channel().get_quant_method(block), BlockInt8MoEMethod)


def test_moe_method_rejects_int4():
    """AWQ has no MoE support; get_quant_method returns UnquantizedFusedMoEMethod for ignored."""
    # AWQ doesn't have a MoE method at all — it only returns linear methods.
    # The config simply doesn't support MoE layers.
    pass


# --------------------------------------------------------------------------- #
# create_weights: parameter layout per scheme
# --------------------------------------------------------------------------- #
def test_create_weights_unquantized():
    # No dtype is prescribed: a bare instantiation follows
    # ``torch.get_default_dtype()`` (vLLM's auto convention — the model layer
    # passes ``config.dtype`` down in production), and an explicit
    # ``params_dtype`` is honoured — that is the wire the model layer uses to
    # pass ``config.dtype`` down.
    layer = ReplicatedLinear(64, 128)
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.get_default_dtype()
    assert not hasattr(layer, "weight_scale_inv")
    layer_fp16 = ReplicatedLinear(64, 128, params_dtype=torch.float16)
    assert layer_fp16.weight.dtype == torch.float16


def test_create_weights_int8_per_channel():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    layer = ReplicatedLinear(64, 128, quant=BlockInt8Config.per_channel())
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 1)
    assert layer.weight_scale_inv.dtype == torch.float32


def test_create_weights_int8_blockwise():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    layer = ReplicatedLinear(64, 128, quant=BlockInt8Config.groupwise(group_size=32))
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 2)  # 64 / 32


def test_create_weights_fp8_per_channel():
    from rapid_llm.modules.quantization.fp8 import Fp8Config

    layer = ReplicatedLinear(64, 128, quant=Fp8Config(group_n=1, group_k=1 << 30))
    assert layer.weight.shape == (128, 64)
    assert layer.weight.dtype == torch.uint8  # e4m3 bit pattern container
    assert layer.weight_scale_inv.shape == (128, 1)


def test_create_weights_int4():
    from rapid_llm.modules.quantization.awq import AWQConfig

    layer = ReplicatedLinear(256, 128, quant=AWQConfig(group_size=128))
    assert layer.weight.shape == (128, 32)  # 8 int4 values per int32 word
    assert layer.weight.dtype == torch.int32
    assert layer.weight_scale.shape == (128, 2)
    assert layer.weight_zeros.shape == (128, 2)


def test_create_weights_smoothquant():
    from rapid_llm.modules.quantization.w8a8_int8 import W8A8Int8Config

    layer = ReplicatedLinear(64, 128, quant=W8A8Int8Config())
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 1)


def test_create_weights_nvfp4():
    from rapid_llm.modules.quantization.nvfp4 import NVFP4Config

    layer = ReplicatedLinear(256, 128, quant=NVFP4Config())
    assert layer.weight.shape == (128, 128)  # two e2m1 nibbles per byte
    assert layer.weight.dtype == torch.uint8
    assert layer.weight_scale.shape == (128, 16)  # one e4m3 scale per 16 k
    assert layer.weight_scale.dtype == torch.uint8
    assert layer.weight_global_scale.shape == (1,)
    assert layer.weight_global_scale.dtype == torch.float32


def test_nvfp4_rejects_in_features_that_split_a_block():
    from rapid_llm.modules.quantization.nvfp4 import NVFP4Config

    with pytest.raises(ValueError, match="divisible by 16"):
        ReplicatedLinear(72, 128, quant=NVFP4Config())


def test_nvfp4_shard_granularity_is_the_lcm_not_the_product():
    """16, not 32: a k-shard must hold whole bytes (2) and whole blocks (16).

    4864 is the ``down_proj`` k-shard Qwen3-4B gets under TP2 and is divisible
    by 16 but not 32, so the over-strict rule would reject a shard the format
    handles perfectly.
    """
    from rapid_llm.modules.quantization.nvfp4 import NVFP4Config

    config = NVFP4Config()
    assert config.shard_is_aligned(16)
    assert config.shard_is_aligned(4864)
    assert not config.shard_is_aligned(8)
    assert not config.shard_is_aligned(24)


def test_nvfp4_refuses_moe_experts_rather_than_misserving_them():
    from rapid_llm.modules.quantization.nvfp4 import NVFP4Config

    config = NVFP4Config()
    block = _StubMoeBlock(quant=config)
    with pytest.raises(NotImplementedError, match="NVFP4 MoE experts"):
        config.get_quant_method(block, "mlp")

    # An *ignored* MoE prefix is the one case that must still pass through, so
    # a checkpoint that left its experts in bf16 still loads.
    ignored = NVFP4Config(ignored=("mlp",))
    assert isinstance(ignored.get_quant_method(block, "mlp"), UnquantizedFusedMoEMethod)


@needs_capability((10, 0), "nvfp4 blockwise quantization")
def test_nvfp4_quantize_from_fp16_roundtrip():
    """The runtime ``--quantization nvfp4`` path, end to end on one layer.

    On the card, not on CPU staging: ``quantize_from_fp16`` refuses a CPU
    weight (there is no CPU implementation of the blockwise kernel), and the
    claim is about what the kernel writes — so the layer lives on the device
    the capability gate allows, and the reference decoder reads it there.
    """
    from rapid_llm.modules.quantization.nvfp4 import NVFP4Config, NVFP4LinearMethod
    from tests.reference import nvfp4_dequant

    torch.manual_seed(0)
    config = NVFP4Config()
    layer = ReplicatedLinear(256, 128).to("cuda")
    original = _fill_fp16(layer)

    NVFP4LinearMethod().quantize_from_fp16(layer, config)
    assert layer.weight.shape == (128, 128)
    assert layer.weight_scale.shape == (128, 16)

    restored = nvfp4_dequant(layer.weight.data, layer.weight_scale.data, layer.weight_global_scale)
    # 4 bits with 16-element blocks: ~10% relative on a Gaussian tensor. The
    # bound is what the format costs, not a target the kernel can improve.
    rel = (restored - original).norm() / original.norm()
    assert rel < 0.15, f"relative error {rel:.4f} is worse than NVFP4 should be"


def test_moe_create_weights_int8():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    quant = BlockInt8Config.per_channel()
    block = _StubMoeBlock(quant=quant)
    method = BlockInt8MoEMethod()
    params = method.create_weights(block)
    assert params["gate_up_proj"].shape == (4, 256, 256)  # [E, 2I, H]
    assert params["down_proj"].shape == (4, 256, 128)  # [E, H, I]
    assert params["gate_up_proj"].dtype == torch.int8
    assert params["gate_up_proj_scale_inv"].shape == (4, 256, 1)
    assert params["down_proj_scale_inv"].shape == (4, 256, 1)


# --------------------------------------------------------------------------- #
# quantize_: fp16 -> low-bit conversion per runtime scheme
# --------------------------------------------------------------------------- #
def test_quantize_int8_per_channel_roundtrip():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.per_channel())
    assert isinstance(layer.quant_method, BlockInt8LinearMethod)
    assert layer.weight.dtype == torch.int8
    deq = layer.weight.float() * layer.weight_scale_inv
    torch.testing.assert_close(deq, w, rtol=2e-2, atol=2e-3)


def test_quantize_int8_blockwise_roundtrip():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.groupwise(group_size=128))
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale_inv.shape == (128, 2)
    s = layer.weight_scale_inv.repeat_interleave(128, dim=1)
    torch.testing.assert_close(layer.weight.float() * s, w, rtol=2e-2, atol=2e-3)


def test_quantize_fp8_per_channel_roundtrip():
    from rapid_llm.modules.quantization.fp8 import Fp8Config

    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(Fp8Config(group_n=1, group_k=1 << 30))
    assert isinstance(layer.quant_method, (Fp8LinearMethod, W8A8Fp8LinearMethod))
    assert layer.weight.dtype == torch.uint8
    deq = layer.weight.view(torch.float8_e4m3fn).float() * layer.weight_scale_inv
    # e4m3 has 3 mantissa bits, so the tolerance is wider than int8's.
    torch.testing.assert_close(deq, w, rtol=1e-1, atol=1e-2)


def test_quantize_int4_roundtrip():
    from rapid_llm.modules.quantization.awq import AWQConfig

    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(AWQConfig(group_size=128))
    assert isinstance(layer.quant_method, AWQLinearMethod)
    assert layer.weight.dtype == torch.int32
    deq = _dequant_int4(layer.weight, layer.weight_scale, layer.weight_zeros, 128, 128, 256)
    # int4's quantisation step is ~amax/7 per group, so the honest bound is the
    # same one the kernel test uses.
    torch.testing.assert_close(deq, w, rtol=5e-2, atol=5e-2)


def test_quantize_smoothquant_roundtrip():
    from rapid_llm.modules.quantization.w8a8_int8 import W8A8Int8Config

    layer = ReplicatedLinear(256, 128)
    w = _fill_fp16(layer)
    layer.quantize_(W8A8Int8Config())
    assert isinstance(layer.quant_method, W8A8Int8LinearMethod)
    assert layer.weight.dtype == torch.int8
    deq = layer.weight.float() * layer.weight_scale_inv
    torch.testing.assert_close(deq, w, rtol=2e-2, atol=2e-3)


def test_quantize_is_idempotent_guard():
    """A layer that already carries quantised weights is left alone."""
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config
    from rapid_llm.modules.quantization.fp8 import Fp8Config

    layer = ReplicatedLinear(256, 128)
    _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.per_channel())
    weight_after_first = layer.weight
    layer.quantize_(Fp8Config(group_n=1, group_k=1 << 30))  # must be a no-op
    assert layer.weight is weight_after_first
    assert layer.quant.get_name() == "blockwise_int8"


def test_unquantized_method_cannot_convert():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    layer = ReplicatedLinear(64, 128)
    with pytest.raises(NotImplementedError, match="cannot be computed from fp16"):
        UnquantizedLinearMethod().quantize_from_fp16(layer, BlockInt8Config.per_channel())


def test_moe_convert_from_fp16_int8():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    block = _StubMoeBlock()
    block.experts = nn.ParameterDict(UnquantizedFusedMoEMethod().create_weights(block))
    with torch.no_grad():
        for p in block.experts.values():
            p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.05)
    ref = block.experts["gate_up_proj"].data.float().clone()

    quant = BlockInt8Config.per_channel()
    BlockInt8MoEMethod().quantize_from_fp16(block, quant)

    assert block.experts["gate_up_proj"].dtype == torch.int8
    assert block.experts["down_proj"].dtype == torch.int8
    deq = block.experts["gate_up_proj"].float() * block.experts["gate_up_proj_scale_inv"]
    torch.testing.assert_close(deq, ref, rtol=2e-2, atol=2e-3)


def test_moe_convert_from_fp16_fp8():
    from rapid_llm.modules.quantization.fp8 import Fp8Config

    block = _StubMoeBlock()
    block.experts = nn.ParameterDict(UnquantizedFusedMoEMethod().create_weights(block))
    Fp8MoEMethod().quantize_from_fp16(block, Fp8Config(group_n=1, group_k=1 << 30))
    assert block.experts["gate_up_proj"].dtype == torch.uint8
    assert block.experts["down_proj_scale_inv"].shape == (4, 256, 1)


def _dequant_int4(qw, scales, zeros, group_size, n, k):
    """Unpack ``[N, K//8]`` int32 nibbles and apply ``(q - zero) * scale``."""
    words = qw.to(torch.int64)
    shifts = 4 * torch.arange(8)
    w = ((words.unsqueeze(-1) >> shifts) & 0xF).flatten(-2).float()[:, :k]
    w = w.unflatten(-1, (k // group_size, group_size))
    return ((w - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)).flatten(-2)


# --------------------------------------------------------------------------- #
# Parameter factories
# --------------------------------------------------------------------------- #
def test_quantize_int8_groupwise_shapes_and_accuracy():
    w = torch.randn(64, 256) * 0.05
    qw, scale = quantize_int8_groupwise(w, group_size=128)
    assert qw.dtype == torch.int8
    assert qw.shape == (64, 256)
    assert scale.shape == (64, 2)
    deq = qw.float() * scale.repeat_interleave(128, dim=1)
    torch.testing.assert_close(deq, w, rtol=1e-2, atol=1e-3)


def test_quantize_int8_groupwise_rejects_uneven_k():
    with pytest.raises(ValueError, match="multiple of group_size"):
        quantize_int8_groupwise(torch.randn(8, 100), group_size=128)


def test_quantize_fp8_per_channel():
    w = torch.randn(64, 128) * 0.05
    qw, scale = quantize_fp8_per_channel(w)
    assert qw.dtype == torch.uint8
    assert scale.shape == (64, 1)
    deq = qw.view(torch.float8_e4m3fn).float() * scale
    torch.testing.assert_close(deq, w, rtol=1e-1, atol=1e-2)


def test_quantize_fp8_per_channel_zero_row():
    """An all-zero channel must not produce a zero scale (division guard)."""
    w = torch.randn(4, 32)
    w[1].zero_()
    qw, scale = quantize_fp8_per_channel(w)
    assert torch.isfinite(scale).all()
    assert (scale > 0).all()
    assert qw[1].view(torch.float8_e4m3fn).float().abs().max() == 0


# --------------------------------------------------------------------------- #
# Runtime schemes and shard alignment
# --------------------------------------------------------------------------- #
def test_for_runtime_scheme_covers_every_registered_name():
    from rapid_llm.modules.quantization import RUNTIME_SCHEMES, for_runtime_scheme

    for name in RUNTIME_SCHEMES:
        quant = for_runtime_scheme(name)
        assert quant is not None


def test_for_runtime_scheme_rejects_unknown():
    from rapid_llm.modules.quantization import for_runtime_scheme

    with pytest.raises(ValueError, match="unknown runtime quantisation"):
        for_runtime_scheme("int2")


def test_shard_is_aligned_per_channel_always():
    """One scale per output row: no block for a TP shard to cut."""
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    quant = BlockInt8Config.per_channel()
    assert quant.shard_is_aligned(96)
    assert quant.shard_is_aligned(1)


def test_shard_is_aligned_blockwise():
    from rapid_llm.modules.quantization.awq import AWQConfig
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config
    from rapid_llm.modules.quantization.fp8 import Fp8Config

    for quant in (
        BlockInt8Config.groupwise(group_size=128),
        AWQConfig(group_size=128),
        Fp8Config(128, 128),
    ):
        assert quant.shard_is_aligned(256)
        assert not quant.shard_is_aligned(96)


# --------------------------------------------------------------------------- #
# End-to-end: a quantised layer's forward (needs the Triton kernels)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_replicated_linear_int8_forward_matches_reference():
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config

    layer = ReplicatedLinear(256, 128).cuda()
    w = _fill_fp16(layer)
    layer.quantize_(BlockInt8Config.per_channel())
    x = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.5

    out = layer(x)

    ref = x.float() @ (layer.weight.float() * layer.weight_scale_inv).T
    torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
    # The reference dequantises the stored int8, so the layer must sit within
    # int8's rounding noise of the original fp16 product.
    fp16_ref = x.float() @ w.cuda().T
    torch.testing.assert_close(out.float(), fp16_ref, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# GPTQ bits=8: config, dense layer end-to-end, MoE create + repack
# --------------------------------------------------------------------------- #
def test_gptq_config_accepts_bits_8():
    """``bits`` selects the pack factor and the kernel; 4/8 share the container."""
    from rapid_llm.modules.quantization.gptq import GPTQConfig

    qc = GPTQConfig.from_config({"bits": 8, "group_size": 128})
    assert not qc.is_int4
    assert qc.is_packed
    assert qc.pack_factor == 4
    assert qc.storage_dtype == torch.int32
    assert qc.scale_shape(256, 512) == (256, 4)

    qc4 = GPTQConfig.from_config({"bits": 4, "group_size": 128})
    assert qc4.is_int4 and qc4.is_packed and qc4.pack_factor == 8

    with pytest.raises(ValueError, match="4- and 8-bit"):
        GPTQConfig.from_config({"bits": 2})


def test_packed_flag_separates_packed_from_per_element_formats():
    """``is_packed`` owns the checkpoint-adapter trigger; ``is_int4`` is a query."""
    from rapid_llm.modules.quantization.awq import AWQConfig
    from rapid_llm.modules.quantization.blockwise_int8 import BlockInt8Config
    from rapid_llm.modules.quantization.fp8 import Fp8Config
    from rapid_llm.modules.quantization.gptq import GPTQConfig

    for qc in (AWQConfig(), GPTQConfig(), GPTQConfig(bits=8)):
        assert qc.is_packed
    for qc in (Fp8Config(128, 128), BlockInt8Config.per_channel()):
        assert not qc.is_packed


@pytest.mark.gpu
def test_gptq_int8_layer_quantize_and_forward():
    """Runtime conversion for ``bits=8``, end to end on one layer.

    ``quantize_from_fp16`` packs ``[N, K//4]`` int32 (the checkpoint container);
    ``process_weights_after_loading`` unpacks to ``[N, K]`` int8 (the kernel's),
    so the same tensors flow whether the weights came from a checkpoint or were
    computed here — the point of the packed intermediate.
    """
    from rapid_llm.modules.quantization.gptq import GPTQConfig

    torch.manual_seed(0)
    layer = ReplicatedLinear(256, 128).cuda()
    w = _fill_fp16(layer)
    layer.quantize_(GPTQConfig(group_size=128, bits=8))

    assert layer.weight.dtype == torch.int8
    assert layer.weight.shape == (128, 256)
    assert layer.weight_scale.shape == (128, 2)
    assert layer.weight_zeros.shape == (128, 2)

    x = torch.randn(8, 256, device="cuda", dtype=torch.float16) * 0.5
    out = layer(x)

    deq = (
        (layer.weight.float().reshape(128, 2, 128) - layer.weight_zeros.unsqueeze(-1))
        * layer.weight_scale.unsqueeze(-1)
    ).reshape(128, 256)
    torch.testing.assert_close(out.float(), x.float() @ deq.T, rtol=1e-2, atol=1e-2)
    # And within int8's rounding of the original fp16 product.
    torch.testing.assert_close(out.float(), x.float() @ w.cuda().T, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_gptq_int8_moe_create_and_repack():
    """The MoE hook swaps checkpoint words for the fused kernel's byte layout.

    ``create_weights`` allocates ``[E, N, K//4]`` int32 so the expert loader
    fills it directly; ``process_weights_after_loading`` expands to ``[E, N, K]``
    int8 — the asymmetric container the kernel's zeros branch dequantises.
    """
    from rapid_llm.modules.quantization.gptq import GPTQConfig, GPTQMoEMethod
    from rapid_llm.modules.quantization.parameter import RawParameter

    torch.manual_seed(0)
    block = _StubMoeBlock(quant=GPTQConfig(group_size=128, bits=8))
    method = GPTQMoEMethod()
    params = method.create_weights(block)
    assert params["gate_up_proj"].shape == (4, 256, 64)  # [E, 2I, H//4]
    assert params["gate_up_proj"].dtype == torch.int32
    assert params["down_proj"].shape == (4, 256, 32)  # [E, H, I//4]
    assert params["gate_up_proj_zeros"].shape == (4, 256, 2)

    # Fill with checkpoint-like words on the device the hook runs on, then run
    # the swap and check it against an independent torch unpack.
    block.experts = nn.ParameterDict(
        {name: RawParameter(p.data.cuda()) for name, p in params.items()}
    )
    for name in ("gate_up_proj", "down_proj"):
        block.experts[name].data.copy_(
            torch.randint(
                -(2**31),
                2**31 - 1,
                block.experts[name].shape,
                device="cuda",
                dtype=torch.int64,
            ).to(torch.int32)
        )
    packed = {name: block.experts[name].data.clone() for name in ("gate_up_proj", "down_proj")}

    method.process_weights_after_loading(block)

    shifts = torch.arange(0, 32, 8, device="cuda", dtype=torch.int32)
    for name in ("gate_up_proj", "down_proj"):
        ref = (
            ((packed[name].unsqueeze(-1) >> shifts) & 0xFF)
            .flatten(-2)
            .to(torch.uint8)
            .view(torch.int8)
        )
        assert block.experts[name].shape == ref.shape
        assert block.experts[name].dtype == torch.int8
        assert torch.equal(block.experts[name].data, ref)
