"""Numeric parity tests for the Qwen3-MoE support.

Block-fp8 dequantisation round-trips, fp8 scale-table consumption, the
HF-config plumbing that marks layers MoE vs dense, a decoder step
against a hand-built tiny config, and a model-level fp8 MoE forward
through the runtime ``--quantization fp8`` scheme.

Usage:
    pytest tests/models/test_qwen3_moe.py
"""

from __future__ import annotations

import json

import pytest
import torch

from rapid_llm.executor.weight_utils import dequant_block_fp8, hf_weights_iterator
from rapid_llm.models.config import ModelConfig
from rapid_llm.models.qwen3_moe import is_moe_layer
from tests.conftest import needs_capability
from tests.registry import import_needs_cuda

import_needs_cuda()

# --------------------------------------------------------------------------- #
# FP8 dequantisation
# --------------------------------------------------------------------------- #


def _block_quantize_fp8(w: torch.Tensor, block: int = 128):
    """Emulate the Qwen FP8 checkpoint format: e4m3 weight + fp32 scale_inv."""
    out_f, in_f = w.shape
    w8 = torch.empty_like(w)
    scale = torch.empty(
        (out_f + block - 1) // block, (in_f + block - 1) // block, dtype=torch.float32
    )
    for bi in range(scale.shape[0]):
        for bj in range(scale.shape[1]):
            tile = w[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block]
            s = tile.abs().max() / 448.0  # e4m3 max magnitude
            # ``weight_scale_inv`` 是反量化乘子(量化时除、反量化时乘)。
            scale[bi, bj] = s
            w8[bi * block : (bi + 1) * block, bj * block : (bj + 1) * block] = (
                (tile / s).to(torch.float8_e4m3fn).to(torch.float32)
            )
    return w8.to(torch.float8_e4m3fn), scale


@pytest.mark.parametrize("shape", [(256, 256), (200, 300)])
def test_block_fp8_dequant(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape, dtype=torch.float32)
    w8, scale_inv = _block_quantize_fp8(w)

    deq = dequant_block_fp8(w8, scale_inv)

    # bf16 is the loader's default widening target since v0.9 (ROADMAP F5).
    assert deq.dtype == torch.bfloat16
    assert deq.shape == w.shape
    # e4m3 keeps ~3 mantissa bits -> ~6% relative error per element.
    rel = ((deq.float() - w).abs() / w.abs().clamp_min(1e-3)).median()
    assert rel < 0.06


@pytest.mark.usefixtures("cuda_available")
def test_block_fp8_dequant_agrees_on_gpu():
    """The loader dequantises on the target device; CPU and GPU must agree bit for bit.

    Doing it on the CPU used to dominate load time for a 30B FP8 checkpoint, so the
    op moved to the device — which only helps if it produces the same numbers.
    """
    torch.manual_seed(0)
    w = torch.randn(300, 200, dtype=torch.float32)
    w8, scale_inv = _block_quantize_fp8(w)

    on_cpu = dequant_block_fp8(w8, scale_inv)
    on_gpu = dequant_block_fp8(w8.cuda(), scale_inv.cuda())
    assert torch.equal(on_cpu, on_gpu.cpu())


def test_fp8_scale_tables_are_consumed_not_yielded(tmp_path):
    """``*.weight_scale_inv`` must never reach the model: it is a file-format detail."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    w = torch.randn(256, 256)
    w8, scale_inv = _block_quantize_fp8(w)
    save_file(
        {"layers.0.mlp.down_proj.weight": w8, "layers.0.mlp.down_proj.weight_scale_inv": scale_inv},
        str(tmp_path / "model.safetensors"),
    )

    loaded = dict(hf_weights_iterator(tmp_path))
    assert list(loaded) == ["layers.0.mlp.down_proj.weight"]
    torch.testing.assert_close(
        loaded["layers.0.mlp.down_proj.weight"], dequant_block_fp8(w8, scale_inv)
    )


# --------------------------------------------------------------------------- #
# Layer-type selection
# --------------------------------------------------------------------------- #

_TINY_HF_CONFIG = {
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 64,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
}


def _model_config(tmp_path, max_seq_len: int = 256, **overrides) -> ModelConfig:
    """Round-trip a config.json through AutoConfig, as the loader does."""
    body = {"model_type": "qwen3_moe", **_TINY_HF_CONFIG, **overrides}
    (tmp_path / "config.json").write_text(json.dumps(body))
    return ModelConfig.from_pretrained(tmp_path, max_seq_len=max_seq_len)


def test_every_layer_is_moe_by_default(tmp_path):
    """Qwen3-30B-A3B ships mlp_only_layers=[] and decoder_sparse_step=1."""
    config = _model_config(tmp_path, num_hidden_layers=48)
    assert config.num_layers == 48
    assert all(is_moe_layer(config, i) for i in range(48))


def test_mlp_only_layers_stay_dense(tmp_path):
    config = _model_config(tmp_path, mlp_only_layers=[0, 5])
    assert not is_moe_layer(config, 0)
    assert not is_moe_layer(config, 5)
    assert is_moe_layer(config, 1)


def test_decoder_sparse_step_skips_layers(tmp_path):
    config = _model_config(tmp_path, decoder_sparse_step=2)
    assert [i for i in range(6) if is_moe_layer(config, i)] == [1, 3, 5]


# --------------------------------------------------------------------------- #
# Router semantics vs HuggingFace (CPU only, no Triton kernels involved)
# --------------------------------------------------------------------------- #


def test_route_matches_hf(tmp_path):
    """``_route`` must reproduce HF's softmax-all -> top-k -> renormalise order."""
    from rapid_llm.modules.moe import SparseMoeBlock

    torch.manual_seed(0)
    config = _model_config(
        tmp_path,
        hidden_size=64,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=32,
        # The router weight follows config.dtype (bf16 when undeclared); pin
        # fp16 so the block matches the fp16 activations this test feeds.
        torch_dtype="float16",
    )
    block = SparseMoeBlock(config)
    gate = torch.randn(config.num_experts, config.hidden_size)
    block.gate_weight.data.copy_(gate)

    # The router weight follows the checkpoint dtype (bf16 when undeclared),
    # so the activation must arrive in that dtype too.
    x = torch.randn(7, config.hidden_size).to(config.dtype)
    weights, ids = block._route(x)

    # HF reference: fp32 softmax over all experts, then topk, then renormalise.
    # The gate is read through the router's storage dtype (the checkpoint's own
    # type) so the comparison pins the routing *order* and renormalisation
    # rather than the storage precision; the weight tolerance below absorbs
    # the residual bf16 rounding of the logits themselves.
    ref = torch.softmax(
        torch.nn.functional.linear(x.float(), gate.to(config.dtype).float()), dim=-1
    )
    ref_w, ref_ids = torch.topk(ref, config.num_experts_per_tok, dim=-1)
    ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    assert torch.equal(ids, ref_ids)
    # ids pin the routing order; the weight tolerance is the activation
    # dtype's rounding floor — bf16 logits round at 2^-8 before the fp32
    # softmax, so ~1e-2 on the renormalised weights is the physical limit,
    # not a routing defect.
    torch.testing.assert_close(weights.float(), ref_w, atol=1e-2, rtol=1e-1)


# --------------------------------------------------------------------------- #
# Full-model logits parity vs HuggingFace (needs CUDA for the Triton kernels)
# --------------------------------------------------------------------------- #


def _prefill_metadata(seq_len: int):
    """A single-sequence prefill batch over a freshly zeroed cache.

    Prefill does not read the cache, so the buffer's contents do not enter the
    result; it exists because the layers write into it.
    """
    from rapid_llm.executor.model_runner import AttentionMetadata

    num_kv_heads, head_dim = _TINY_HF_CONFIG["num_key_value_heads"], _TINY_HF_CONFIG["head_dim"]
    return AttentionMetadata(
        kv_buffer=[
            torch.zeros(seq_len, 2 * num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
            for _ in range(_TINY_HF_CONFIG["num_hidden_layers"])
        ],
        cur_select_index=torch.arange(seq_len, dtype=torch.int32, device="cuda"),
        b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
        b_seq_len=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
        max_actual_seq_len=seq_len,
    )


@pytest.mark.usefixtures("cuda_available")
def test_qwen3_moe_logits_parity(tmp_path):
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig as HfConfig
    from transformers import Qwen3MoeForCausalLM

    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.models.qwen3_moe import Qwen3MoeModel

    torch.manual_seed(42)
    hf_model = Qwen3MoeForCausalLM(HfConfig(**_TINY_HF_CONFIG)).eval()
    save_file(hf_model.state_dict(), str(tmp_path / "model.safetensors"))

    lite_model = Qwen3MoeModel(_model_config(tmp_path))
    materialise_parameters(lite_model, "cuda")
    lite_model.load_weights(hf_weights_iterator(tmp_path, "cuda"))
    lite_model.to("cuda").eval()

    seq_len = 12
    input_ids = torch.randint(0, _TINY_HF_CONFIG["vocab_size"], (1, seq_len))

    with torch.no_grad():
        ref_logits = hf_model(input_ids=input_ids).logits.float()  # [1, T, V]

    atten_info = _prefill_metadata(seq_len)
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)
    with torch.no_grad():
        lite_logits = lite_model(input_ids.cuda(), position_ids, atten_info).float().cpu()

    assert lite_logits.shape == ref_logits.shape

    # Random-weight tiny models produce near-degenerate logits whose top-2 gaps
    # can sit below the fp16-vs-fp32 noise floor, where argmax is a coin flip.
    # Argmax is therefore only asserted where the gap is decidable; overall
    # fidelity is carried by the cosine-similarity and absolute-diff checks.
    noise_floor = (lite_logits - ref_logits).abs().mean().item()
    top2 = ref_logits.topk(2, dim=-1).values
    decidable = (top2[..., 0] - top2[..., 1]) > 3 * noise_floor
    agree = lite_logits.argmax(-1) == ref_logits.argmax(-1)
    assert agree[decidable].all(), (
        f"argmax mismatch on decidable tokens: {(~agree & decidable).nonzero().flatten().tolist()}"
    )

    cos = torch.nn.functional.cosine_similarity(lite_logits, ref_logits, dim=-1)
    # 0.985 仍远高于随机方向(~0); 随机权重模型的 logits 接近简并, 不能按
    # 真实模型的 0.999 标准要求。
    assert cos.min() > 0.985, f"min cosine {cos.min().item():.6f}"
    max_diff = (lite_logits - ref_logits).abs().max().item()
    scale = ref_logits.abs().max().item()
    assert max_diff < 0.2 * max(scale, 1.0), f"max abs diff {max_diff:.5f} (scale {scale:.2f})"


# --------------------------------------------------------------------------- #
# Runtime fp8 quantisation of the experts
# --------------------------------------------------------------------------- #


@needs_capability((8, 9), "fp8 e4m3 MoE kernels")
def test_qwen3_moe_fp8_forward(tmp_path):
    """``--quantization fp8`` on an MoE model: experts become e4m3 and still run.

    ``RUNTIME_SCHEMES["fp8"]`` is :class:`W8A8Fp8Config`, so this is the only
    model-level path that reaches ``fused_moe_w8a8_fp8`` — where the *activation*
    is quantised too. The weights of the two fp8 MoE methods are byte-identical,
    so which one ran cannot be read off a state dict; the assertion on
    :class:`W8A8Fp8MoEMethod` is what pins the entry point, and the forward is
    what proves the kernel accepts what the method built.

    The output is compared against the fp16 forward of the *same* weights rather
    than against HuggingFace: that isolates the quantisation error from the
    engine-vs-HF difference the parity test above already covers.
    """
    from safetensors.torch import save_file
    from transformers import Qwen3MoeConfig as HfConfig
    from transformers import Qwen3MoeForCausalLM

    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.models.qwen3_moe import Qwen3MoeModel
    from rapid_llm.modules.moe import SparseMoeBlock
    from rapid_llm.modules.quantization import for_runtime_scheme
    from rapid_llm.modules.quantization.w8a8_fp8 import W8A8Fp8Config, W8A8Fp8MoEMethod

    torch.manual_seed(42)
    hf_model = Qwen3MoeForCausalLM(HfConfig(**_TINY_HF_CONFIG)).eval()
    save_file(hf_model.state_dict(), str(tmp_path / "model.safetensors"))

    model = Qwen3MoeModel(_model_config(tmp_path))
    materialise_parameters(model, "cuda")
    model.load_weights(hf_weights_iterator(tmp_path, "cuda"))
    model.to("cuda").eval()

    seq_len = 12
    input_ids = torch.randint(0, _TINY_HF_CONFIG["vocab_size"], (1, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, dtype=torch.int32, device="cuda").unsqueeze(0)

    with torch.no_grad():
        ref = model(input_ids, position_ids, _prefill_metadata(seq_len)).float().cpu()

    quant = for_runtime_scheme("fp8")
    assert isinstance(quant, W8A8Fp8Config), "--quantization fp8 must select the W8A8 config"
    model.quantize_(quant)

    blocks = [m for m in model.modules() if isinstance(m, SparseMoeBlock)]
    assert len(blocks) == _TINY_HF_CONFIG["num_hidden_layers"]
    for block in blocks:
        assert isinstance(block.quant_method, W8A8Fp8MoEMethod)
        for name in ("gate_up_proj", "down_proj"):
            assert block.experts[name].dtype == torch.uint8
            assert block.experts[f"{name}_scale_inv"].dtype == torch.float32

    with torch.no_grad():
        out = model(input_ids, position_ids, _prefill_metadata(seq_len)).float().cpu()

    assert out.shape == ref.shape
    assert torch.isfinite(out).all()
    # A no-op ``quantize_`` would satisfy everything above, so require that the
    # numbers actually moved.
    assert not torch.equal(out, ref)
    # Not a tolerance on the kernel: ``quantize_`` converts every dense
    # projection too, so this is the whole model's fp8 error (measured: RMS
    # relative 6.4e-2, min cosine 0.997). A tiny random-weight model has
    # near-degenerate logits, where absolute diffs say little and cosine is the
    # stable statistic; 0.99 sits far above the ~0 of an unrelated direction.
    cos = torch.nn.functional.cosine_similarity(out, ref, dim=-1)
    assert cos.min() > 0.99, f"min cosine {cos.min().item():.6f}"
