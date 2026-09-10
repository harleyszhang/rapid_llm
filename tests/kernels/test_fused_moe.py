"""Tests for the fused MoE grouped-GEMM kernels against a pure-torch reference.

``moe_align_block_size`` sorting and padding is checked exactly, then
the fused kernel is diffed against ``fused_moe_reference`` across
token, expert and top-k counts, including every quantised expert
format and the fp8 W8A8 mode.

Usage:
    pytest tests/kernels/test_fused_moe.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import rapid_llm.kernels.ops.moe.fused_moe as _fused_moe_mod
from rapid_llm.kernels.ops.moe.fused_moe import (
    fused_moe,
    fused_moe_w8a8_fp8,
    fused_moe_w8a8_int8,
    moe_align_block_size,
)
from rapid_llm.kernels.ops.quantization import repack_int4_experts, unpack_int8_experts
from rapid_llm.modules.quantization.utils import (
    quantize_fp8_per_channel,
    quantize_fp8_per_token,
    quantize_int4_groupwise,
    quantize_int8_groupwise_asym,
    quantize_int8_per_channel,
)
from tests.reference import fused_moe_reference

# Redundant with the automatic `gpu` mark applied to tests/kernels/ by
# tests/conftest.py, but harmless and keeps the file self-describing.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# --------------------------------------------------------------------------- #
# moe_align_block_size
# --------------------------------------------------------------------------- #
def test_align_sorts_by_expert_and_pads():
    topk_ids = torch.tensor([[2, 0], [1, 2]], device="cuda", dtype=torch.int32)
    block_size = 4
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_size, 3)
    num_post = int(num_post.item())

    # Expert 0: 1 slot (id 1), expert 1: 1 slot (id 2), expert 2: 2 slots (ids 0, 3);
    # each run padded up to a multiple of 4.
    assert num_post == 3 * block_size
    valid = sorted_ids[:num_post].tolist()
    assert [v for v in valid if v != topk_ids.numel()] == [1, 2, 0, 3]
    # One block per expert here; ids are ordered by expert.
    assert expert_ids[: num_post // block_size].tolist() == [0, 1, 2]
    # Padding slots carry the sentinel (== num_slots) so the kernel masks them.
    assert (sorted_ids[1:4] == topk_ids.numel()).all()


def test_align_no_padding_waste_when_full():
    # 8 slots all routed to one expert, block 4 -> exactly 2 blocks, no sentinel.
    topk_ids = torch.full((4, 2), 5, device="cuda", dtype=torch.int32)
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 4, 8)
    assert int(num_post.item()) == 8
    assert sorted_ids[:8].max() < 8  # every slot is real
    assert expert_ids[:2].tolist() == [5, 5]


# --------------------------------------------------------------------------- #
# Launch-config fallback (the path an empty autotune store takes)
# --------------------------------------------------------------------------- #
def test_launch_config_fallback_runs_on_every_tier(monkeypatch):
    """``_launch_config`` must evaluate on this device, store or no store.

    Regression test. The PRE_HOPPER branch reads ``tile_tier``/``TileTier``; when
    those names were missing from the module import the heuristic raised
    ``NameError`` on every device, yet was only *reached* on GPUs whose autotune
    store had no entry for the shape -- a populated H100 store never touched the
    line, so the defect hid there and a fresh A10 crashed inside ``fused_moe``.
    Forcing the env off guarantees the fallback runs here whatever the store
    holds, across one rows-per-expert value per tier boundary and every mode.
    """
    monkeypatch.setenv("LITE_LLAMA_AUTOTUNE", "0")
    from rapid_llm.kernels.ops.moe.fused_moe import _launch_config

    device = torch.cuda.current_device()
    for mode in range(7):
        for rows in (1.0, 16.0, 33.0, 65.0):
            cfg = _launch_config(64, mode, rows, device)
            assert cfg["BLOCK_M"] > 0 and cfg["BLOCK_N"] > 0
            assert cfg["BLOCK_K"] == 128
            assert cfg["num_warps"] in (1, 2, 4, 8, 16)
            assert cfg["num_stages"] in (2, 3)


def test_fused_moe_runs_with_autotune_disabled(monkeypatch):
    """The heuristic-only path (a new GPU's empty store) must produce output.

    The companion to the test above: it pins the *names* the fallback needs, this
    one pins that a full forward through both grouped GEMMs survives on it.
    """
    monkeypatch.setenv("LITE_LLAMA_AUTOTUNE", "0")
    x = torch.randn(4, 64, device="cuda", dtype=torch.float16) / 8
    w1 = torch.randn(4, 32, 64, device="cuda", dtype=torch.float16) / 8
    w2 = torch.randn(4, 64, 16, device="cuda", dtype=torch.float16) / 4
    ids = torch.randint(0, 4, (4, 2), device="cuda", dtype=torch.int32)
    weights = torch.ones(4, 2, device="cuda", dtype=torch.float16)
    out = fused_moe(x, w1, w2, weights, ids)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# fused_moe vs reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_tokens", [1, 3, 37, 128])
@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (128, 8)])
def test_fused_moe_matches_reference(num_tokens, num_experts, top_k):
    hidden, inter = 256, 128
    dtype = torch.float16
    hidden_states = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / hidden**0.5
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=dtype) / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / inter**0.5
    topk_ids = torch.randint(0, num_experts, (num_tokens, top_k), device="cuda")
    topk_weights = torch.softmax(
        torch.randn(num_tokens, top_k, device="cuda", dtype=torch.float32), dim=-1
    ).to(dtype)

    out = fused_moe(hidden_states, w1, w2, topk_weights, topk_ids)
    ref = fused_moe_reference(hidden_states, w1, w2, topk_weights, topk_ids)

    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_fused_moe_routing_weight_folded():
    """Zeroing a routing weight must zero that slot's contribution."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    dtype = torch.float16
    x = torch.randn(5, hidden, device="cuda", dtype=dtype)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=dtype) / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / inter**0.5
    ids = torch.randint(0, num_experts, (5, top_k), device="cuda")

    weights = torch.rand(5, top_k, device="cuda", dtype=dtype)
    weights[:, 1] = 0  # second slot contributes nothing
    out = fused_moe(x, w1, w2, weights, ids)

    ref = fused_moe_reference(x, w1, w2, weights, ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------- #
# Quantised expert weights
# --------------------------------------------------------------------------- #
#: int4 group size, matching what AWQ/GPTQ checkpoints ship and what
#: ``AWQConfig.group_k`` defaults to.
_INT4_GROUP = 128


def _fp8_experts(w: torch.Tensor):
    q, s = quantize_fp8_per_channel(w)
    # Widened from the *same bytes* by torch, which separates the kernel's
    # bit-trick dequant from fp8 rounding: both sides see identical values.
    return (q, s, None), q.view(torch.float8_e4m3fn).float() * s


def _int8_experts(w: torch.Tensor):
    q, s = quantize_int8_per_channel(w)
    return (q, s, None), q.float() * s


def _int4_experts(w: torch.Tensor):
    # quantize_int4_groupwise packs with a 2D reshape, so it takes [N, K] only.
    parts = [quantize_int4_groupwise(w[e], _INT4_GROUP) for e in range(w.shape[0])]
    q, s, z = (torch.stack(t) for t in zip(*parts, strict=True))
    e, n, _ = q.shape
    k = w.shape[-1]
    # `& 0xF` after the shift is load-bearing: a top nibble can set the sign bit
    # and torch's int32 `>>` is arithmetic, so it sign-extends.
    shifts = torch.arange(8, device=w.device, dtype=torch.int32) * 4
    nibbles = ((q.unsqueeze(-1) >> shifts) & 0xF).reshape(e, n, k).float()
    groups = nibbles.reshape(e, n, k // _INT4_GROUP, _INT4_GROUP)
    deq = ((groups - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape(e, n, k)
    # The kernel eats the byte layout (two nibbles per uint8); cross the same
    # bridge process_weights_after_loading does at load, after the reference
    # has unpacked the words it was derived from.
    return (repack_int4_experts(q), s, z), deq


def _int8_asym_experts(w: torch.Tensor):
    # Like the int4 helper: the quantiser is 2D-only, and the reference
    # dequantises from the *packed* words so both sides share one bit stream.
    parts = [quantize_int8_groupwise_asym(w[e], _INT4_GROUP) for e in range(w.shape[0])]
    q, s, z = (torch.stack(t) for t in zip(*parts, strict=True))
    e, n, _ = q.shape
    k = w.shape[-1]
    # The kernel eats one int8 byte per element — the layout
    # process_weights_after_loading leaves after unpacking the words.
    un = unpack_int8_experts(q)
    groups = un.reshape(e, n, k // _INT4_GROUP, _INT4_GROUP)
    deq = ((groups - z.unsqueeze(-1)) * s.unsqueeze(-1)).reshape(e, n, k)
    return (un, s, z), deq


_QUANT_FORMATS = {
    "fp8": _fp8_experts,
    "int8": _int8_experts,
    "int8_asym": _int8_asym_experts,
    "int4": _int4_experts,
}


@pytest.mark.parametrize("fmt", sorted(_QUANT_FORMATS))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_quantised_matches_reference(fmt, dtype):
    """Every 8/4-bit expert format, at both activation dtypes.

    Each format runs a different branch of the kernel's inner loop -- an e4m3 bit
    trick, a symmetric int8 convert, an int8 convert with a zero point (the
    GPTQ ``bits=8`` asymmetric mode), a nibble unpack with a zero point -- so a
    format verified through a sibling is unverified. The reference multiplies
    the *dequantised* weights, which isolates the kernel's arithmetic from the
    format's own error: both sides see the same numbers.

    Both dtypes, because neither axis used to work. The quantised branches widened
    the weight tile to a hard-coded fp16, so ``tl.dot`` got two operand types on
    any bf16 model -- which is every Qwen3-MoE checkpoint -- and the layer failed
    to compile rather than to compute. int4 was worse: it folded its fp32 scale
    into the operand before the dot, so that branch could not compile at *any*
    activation dtype, and no test reached it.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    # 33 tokens is not a multiple of any BLOCK_M the launcher picks, so the padded
    # slots and their sentinel mask are exercised rather than avoided.
    tokens = 33
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    (q1, s1, z1), ref1 = _QUANT_FORMATS[fmt](w1)
    (q2, s2, z2), ref2 = _QUANT_FORMATS[fmt](w2)

    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    ids = torch.rand(tokens, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1).to(dtype)

    # Per-channel scales are one group spanning all of K, and the two GEMMs have
    # different K (hidden for gate_up, inter for down) but share one group_k, which
    # the launcher clamps with min(group_k, K) -- so the larger K covers both.
    # int8_asym shares the int4 group size: both are the group-wise GPTQ layout.
    grouped = fmt in ("int4", "int8_asym")
    group_k = _INT4_GROUP if grouped else max(hidden, inter)
    out = fused_moe(
        x,
        q1,
        q2,
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        w1_zeros=z1,
        w2_zeros=z2,
        group_n=1,
        group_k=group_k,
    )
    ref = fused_moe_reference(x, ref1, ref2, weights, ids)

    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_fused_moe_rejects_mixed_quant_formats():
    """A scale on one weight and not the other is a caller bug, not a fp16 path."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, num_experts, (3, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(3, top_k, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="same quantisation format"):
        fused_moe(x, q1, w2.half(), weights, ids, w1_scale=s1, group_n=1, group_k=hidden)


# --------------------------------------------------------------------------- #
# fp8 W8A8 experts (activation quantised too)
# --------------------------------------------------------------------------- #
#: RMS error relative to the reference's RMS, per activation dtype. This is the
#: gate rather than a max-element ``atol`` because the max grows with the number
#: of output elements (a max over more samples of the same distribution), while
#: the RMS is stable across shapes: measured 1.1e-3 / 4.6e-3 / 5.7e-3 for fp16 at
#: 1 / 33 / 129 tokens and 2.8e-2 / 1.3e-2 / 1.4e-2 for bf16.
#:
#: bf16 is ~2.5x looser because the mode stores three intermediates in the
#: activation dtype -- the silu output, each slot's GEMM2 row, and the sum -- and
#: bf16 keeps 8 mantissa bits where fp16 keeps 11.
_A8_RMS_REL = {torch.float16: 1.5e-2, torch.bfloat16: 4.0e-2}

#: Largest single element error, as a fraction of the reference's peak. Measured
#: 1.4e-2 (fp16) and 2.7e-2 (bf16); this bounds it rather than tracking it.
_A8_MAX_OVER_PEAK = 5.0e-2

#: How far quantising the activation moves the answer, RMS relative, versus the
#: same weights with a full-precision activation. Measured 4.4e-2 -- an order of
#: magnitude above ``_A8_RMS_REL`` because it is a different quantity: that one is
#: the kernel's error against its own inputs, this one is the *cost of the mode*.
#: e4m3 carries 3 mantissa bits, so each element is good to about 3%, and a dot of
#: K random-sign terms keeps that 3% rather than averaging it down (the error grows
#: like sqrt(K), and so does the sum's own magnitude). Two such GEMMs in series
#: give sqrt(2) x 3% ~ 4e-2, which is what shows up.
_A8_VS_A16_RMS_REL = 6.0e-2


def _fp8_round_trip(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-token e4m3 quantise-dequantise, in torch, at the kernel's precision.

    Downcast to ``dtype`` first because the kernel quantises what it stored: the
    activation as it arrived, and the silu output after it was written back in the
    activation dtype. Quantising an fp32 intermediate would compare the kernel
    against a pipeline it does not run.

    Uses the torch quantiser (``modules.quantization.utils``), not the Triton one
    the kernel calls;
    ``test_fp8_quantize_per_token_matches_torch_helper`` gates that the two agree.
    """
    q, scale = quantize_fp8_per_token(t.to(dtype))
    return q.view(torch.float8_e4m3fn).float() * scale


def _int8_round_trip(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Per-token symmetric int8 quantise-dequantise, in torch, at the kernel's
    precision.

    Mirrors :func:`_fp8_round_trip`: the kernel quantises what it stored, so the
    reference downcasts to ``dtype`` first. ``round`` — round to nearest even —
    matches both A-quantising routes under test, the inline one in the GEMM and
    ``int8_quantize_per_token``; a truncating spelling would compare the kernel
    against a pipeline it does not run.
    """
    flat = t.to(dtype).reshape(-1, t.shape[-1]).float()
    scale = flat.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = (flat / scale).round().clamp(-127, 127)
    return (q * scale).reshape(t.shape)


@pytest.mark.parametrize("tokens", [1, 33, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8_a8_matches_reference(tokens, dtype):
    """``fused_moe_w8a8_fp8`` against a torch reference that quantises both sides.

    The reference applies the same per-token e4m3 round trip to the input and to
    the silu output, so what is being checked is the kernel's arithmetic, not
    e4m3's dynamic range. The weight-only rows above cannot stand in for this one:
    mode 4 compiles a different inner loop, where neither operand is widened and
    the activation carries its own row scale.

    The token counts pick the launcher's three ``BLOCK_M`` tiers (16 / 32 / 64),
    which matters more here than for any other format: Triton emits Hopper's fp8
    ``wgmma`` only from ``BLOCK_M >= 64`` and widens both e4m3 operands to an fp16
    ``mma.sync`` below it, so 129 tokens is the only case that reaches the fp8
    tensor cores at all. None of 1, 33 or 129 is a multiple of its tile, so the
    padded slots and their sentinel mask are exercised rather than avoided.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    # Widened from the same bytes, so the reference's weights are the kernel's.
    d1 = q1.view(torch.float8_e4m3fn).float() * s1
    d2 = q2.view(torch.float8_e4m3fn).float() * s2

    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    ids = torch.rand(tokens, num_experts, device="cuda").topk(top_k, dim=-1).indices
    ids = ids.to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1).to(dtype)

    out = fused_moe_w8a8_fp8(
        x,
        q1,
        q2,
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        group_n=1,
        group_k=max(hidden, inter),
    )
    ref = fused_moe_reference(
        x, d1, d2, weights, ids, act_quant=lambda t: _fp8_round_trip(t, dtype)
    )

    assert out.dtype == dtype
    err = (out.float() - ref).abs()
    rms_rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rms_rel < _A8_RMS_REL[dtype], f"rms relative error {rms_rel:.3e}"
    peak_rel = (err.max() / ref.abs().max()).item()
    assert peak_rel < _A8_MAX_OVER_PEAK, f"worst element {peak_rel:.3e} of peak"


def test_fused_moe_fp8_a8_differs_from_weight_only():
    """The two fp8 entry points are not aliases of one another.

    They were, before this mode existed: ``W8A8Fp8MoEMethod.apply`` and
    ``Fp8MoEMethod.apply`` ran the same weight-only kernel, so the W8A8 scheme was
    a name with no behaviour behind it. Quantising the activation has to move the
    numbers, or the mode is decoration.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    x = torch.randn(64, hidden, device="cuda", dtype=torch.float16)
    ids = torch.rand(64, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(64, top_k, device="cuda"), dim=-1).to(torch.float16)

    kw = {"w1_scale": s1, "w2_scale": s2, "group_n": 1, "group_k": max(hidden, inter)}
    a16 = fused_moe(x, q1, q2, weights, ids, **kw)
    a8 = fused_moe_w8a8_fp8(x, q1, q2, weights, ids, **kw)

    assert not torch.equal(a16, a8)
    # Same operation either way, so the gap must stay at the size of one extra
    # e4m3 rounding per operand and not grow into a different answer.
    rel = (
        (a8.float() - a16.float()).pow(2).mean().sqrt() / a16.float().pow(2).mean().sqrt()
    ).item()
    assert rel < _A8_VS_A16_RMS_REL, f"the two fp8 paths diverge by {rel:.3e}"


def test_fused_moe_fp8_a8_rejects_non_fp8_experts():
    """Mode 4 cannot be inferred from a dtype, so a wrong caller must be told."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, num_experts, (3, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(3, top_k, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="uint8 e4m3"):
        fused_moe_w8a8_fp8(
            x, q1, q2, weights, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=hidden
        )


# --------------------------------------------------------------------------- #
# int8 W8A8 experts (activation quantised too)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tokens", [1, 33, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_int8_a8_matches_reference(tokens, dtype):
    """``fused_moe_w8a8_int8`` against a torch reference that quantises both
    sides, symmetric to the fp8 A8 test above.

    The tolerance is shared with fp8 deliberately: int8's own error is smaller
    (7 significant bits against e4m3's 3-mantissa-bit values, ~0.4% per element
    against ~3%), so what the gate measures here is the fp16/bf16 intermediate
    storage — the silu output and each slot's GEMM2 row — which both modes pay
    equally.

    The token counts pick the same BLOCK_M tiers as the fp8 row. Unlike fp8,
    whose wgmma only exists from BLOCK_M=64, int8's ``imma`` runs at every tier
    from Turing on, and its int32 accumulation is exact — the reason mode 5 has
    no ``K_PROMOTE`` analogue to tune.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    # Dequantised from the same bytes, so the reference's weights are the kernel's.
    d1 = q1.float() * s1
    d2 = q2.float() * s2

    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype)
    ids = torch.rand(tokens, num_experts, device="cuda").topk(top_k, dim=-1).indices
    ids = ids.to(torch.int32)
    weights = torch.softmax(torch.randn(tokens, top_k, device="cuda"), dim=-1).to(dtype)

    out = fused_moe_w8a8_int8(
        x,
        q1,
        q2,
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        group_n=1,
        group_k=max(hidden, inter),
    )
    ref = fused_moe_reference(
        x, d1, d2, weights, ids, act_quant=lambda t: _int8_round_trip(t, dtype)
    )

    assert out.dtype == dtype
    err = (out.float() - ref).abs()
    rms_rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    assert rms_rel < _A8_RMS_REL[dtype], f"rms relative error {rms_rel:.3e}"
    peak_rel = (err.max() / ref.abs().max()).item()
    assert peak_rel < _A8_MAX_OVER_PEAK, f"worst element {peak_rel:.3e} of peak"


def test_fused_moe_int8_a8_differs_from_weight_only():
    """The two int8 entry points are not aliases of one another.

    ``W8A8Int8MoEMethod.apply`` used to call the weight-only ``fused_moe`` — a
    scheme named W8A8 that never quantised an activation — so the divergence
    this pins is not hypothetical decoration.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    x = torch.randn(64, hidden, device="cuda", dtype=torch.float16)
    ids = torch.rand(64, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(64, top_k, device="cuda"), dim=-1).to(torch.float16)

    kw = {"w1_scale": s1, "w2_scale": s2, "group_n": 1, "group_k": max(hidden, inter)}
    a16 = fused_moe(x, q1, q2, weights, ids, **kw)
    a8 = fused_moe_w8a8_int8(x, q1, q2, weights, ids, **kw)

    assert not torch.equal(a16, a8)
    # int8's quantisation cost sits well under e4m3's, so the fp8-vs-A16 bound
    # (6.0e-2) holds with room to spare: it stays a bound on "same operation,
    # one extra rounding per operand", not a tighter int8-specific one.
    rel = (
        (a8.float() - a16.float()).pow(2).mean().sqrt() / a16.float().pow(2).mean().sqrt()
    ).item()
    assert rel < _A8_VS_A16_RMS_REL, f"the two int8 paths diverge by {rel:.3e}"


def test_fused_moe_int8_a8_rejects_non_int8_experts():
    """Mode 5 cannot be inferred from a dtype, so a wrong caller must be told."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_fp8_per_channel(w1)
    q2, s2 = quantize_fp8_per_channel(w2)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, num_experts, (3, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(3, top_k, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="int8 bytes with scales"):
        fused_moe_w8a8_int8(
            x, q1, q2, weights, ids, w1_scale=s1, w2_scale=s2, group_n=1, group_k=hidden
        )


def test_fused_moe_int8_a8_rejects_zero_points():
    """Symmetric int8 has no zero points; the int4-style arguments are a bug."""
    hidden, inter, num_experts, top_k = 128, 64, 4, 2
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    q1, s1 = quantize_int8_per_channel(w1)
    q2, s2 = quantize_int8_per_channel(w2)
    x = torch.randn(3, hidden, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, num_experts, (3, top_k), device="cuda", dtype=torch.int32)
    weights = torch.rand(3, top_k, device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="no zero points"):
        fused_moe_w8a8_int8(
            x,
            q1,
            q2,
            weights,
            ids,
            w1_scale=s1,
            w2_scale=s2,
            w1_zeros=torch.zeros_like(s1),
            group_n=1,
            group_k=hidden,
        )


@pytest.mark.parametrize("mode", ["fp8", "int8"])
def test_fused_moe_a8_inline_quant_matches_separate(mode, monkeypatch):
    """The two activation-quantising routes produce identical output, not close
    output.

    Below ``_INLINE_A_QUANT_MAX_ROWS`` rows the quantisation happens inside the
    GEMM kernel (an amax pass over the gathered rows, then scaling on the fly);
    above it a separate quantiser kernel runs before the GEMM. Both compute
    ``scale = amax / QMAX`` in fp32 and round with the same instruction (the
    e4m3 cvt, or libdevice rint for int8), so the quantised operand — and with
    it the whole layer — must agree bitwise. Any relaxation here would be a
    silent divergence between the decode and the prefill answer.

    600 tokens sends both GEMMs over the default threshold at once (600 rows
    into GEMM1, ``600 * top_k`` slot rows into GEMM2), and inter=1536 — over
    the silu kernel's 1024-wide block cap and not a power of two — keeps
    GEMM2's quantisation on the route under test instead of fused into the
    silu output. Raising the threshold then flips every branch at once.
    """
    hidden, inter, num_experts, top_k = 256, 1536, 8, 2
    torch.manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    if mode == "fp8":
        q1, s1 = quantize_fp8_per_channel(w1)
        q2, s2 = quantize_fp8_per_channel(w2)
        call = fused_moe_w8a8_fp8
    else:
        q1, s1 = quantize_int8_per_channel(w1)
        q2, s2 = quantize_int8_per_channel(w2)
        call = fused_moe_w8a8_int8
    x = torch.randn(600, hidden, device="cuda", dtype=torch.float16)
    ids = torch.rand(600, num_experts, device="cuda").topk(top_k, dim=-1).indices
    ids = ids.to(torch.int32)
    weights = torch.softmax(torch.randn(600, top_k, device="cuda"), dim=-1).to(torch.float16)
    kw = {"w1_scale": s1, "w2_scale": s2, "group_n": 1, "group_k": max(hidden, inter)}

    separate = call(x, q1, q2, weights, ids, **kw)
    monkeypatch.setattr(_fused_moe_mod, "_INLINE_A_QUANT_MAX_ROWS", 1 << 30)
    inline = call(x, q1, q2, weights, ids, **kw)

    torch.testing.assert_close(inline, separate, rtol=0, atol=0)


@pytest.mark.parametrize("entry", ["weight_only", "w8a8_fp8", "w8a8_int8"])
def test_fused_moe_swiglu_limit_reaches_every_entry_point(entry):
    """``swiglu_limit`` clamps the activation on all three MoE entry points.

    Regression test. The W8A8 rows had no ``swiglu_limit`` parameter and their
    shared body hardcoded ``LIMIT=inf``, so a DeepSeek-V4 quantised to W8A8
    ran plain-silu routed experts and quietly disagreed with its own fp16
    self. The spec test pins that the parameter *exists*; this one pins that
    it *acts*. ``limit=0.75`` sits inside the standard-normal gate/up range,
    so a large share of slots clamp and a dropped bound cannot hide in
    tolerance noise.
    """
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    limit = 0.75
    dtype = torch.float16
    torch.manual_seed(0)
    x = torch.randn(33, hidden, device="cuda", dtype=dtype)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda") / hidden**0.5
    w2 = torch.randn(num_experts, hidden, inter, device="cuda") / inter**0.5
    ids = torch.rand(33, num_experts, device="cuda").topk(top_k, dim=-1).indices.to(torch.int32)
    weights = torch.softmax(torch.randn(33, top_k, device="cuda"), dim=-1).to(dtype)

    if entry == "weight_only":
        call, kw = fused_moe, {}
        # The weight-only path wants the experts in the activation dtype; the
        # quantised branches below quantise the fp32 originals, like the a8 tests.
        w1, w2 = w1.to(dtype), w2.to(dtype)
        ref = fused_moe_reference(x, w1, w2, weights, ids, swiglu_limit=limit)
    else:
        if entry == "w8a8_fp8":
            call = fused_moe_w8a8_fp8
            (q1, s1, _), d1 = _fp8_experts(w1)
            (q2, s2, _), d2 = _fp8_experts(w2)
            round_trip = _fp8_round_trip
        else:
            call = fused_moe_w8a8_int8
            (q1, s1, _), d1 = _int8_experts(w1)
            (q2, s2, _), d2 = _int8_experts(w2)
            round_trip = _int8_round_trip
        kw = {"w1_scale": s1, "w2_scale": s2, "group_n": 1, "group_k": max(hidden, inter)}
        ref = fused_moe_reference(
            x,
            d1,
            d2,
            weights,
            ids,
            act_quant=lambda t: round_trip(t, dtype),
            swiglu_limit=limit,
        )
        w1, w2 = q1, q2

    bounded = call(x, w1, w2, weights, ids, swiglu_limit=limit, **kw)
    plain = call(x, w1, w2, weights, ids, **kw)
    # The clamp has to bite: 0.75 is inside the gate/up range, so the two calls
    # disagree far beyond either tolerance — a dropped bound shows up here
    # rather than hiding as a rounding difference.
    assert not torch.equal(bounded, plain)

    if entry == "weight_only":
        torch.testing.assert_close(bounded.float(), ref, rtol=2e-2, atol=2e-2)
    else:
        err = (bounded.float() - ref).abs()
        rms_rel = (err.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
        assert rms_rel < _A8_RMS_REL[dtype], f"rms relative error {rms_rel:.3e}"
        peak_rel = (err.max() / ref.abs().max()).item()
        assert peak_rel < _A8_MAX_OVER_PEAK, f"worst element {peak_rel:.3e} of peak"


@pytest.mark.parametrize("act_dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_fp8_blockwise_matches_reference(act_dtype):
    """fp8-e4m3 expert weights with 128x128 block scales, in either activation
    dtype.

    Regression: the dequantised operand used to be hardcoded fp16, so bf16
    activations (Qwen3-30B-A3B-Instruct-2507-FP8) failed kernel compilation
    with "Both operands must be same dtype. Got bf16 and fp16".
    """
    torch.manual_seed(0)
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    gn = gk = 128
    x = torch.randn(7, hidden, device="cuda", dtype=act_dtype) / hidden**0.5
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=torch.float32) * 0.05
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=torch.float32) * 0.05
    qw1 = w1.to(torch.float8_e4m3fn).view(torch.uint8)
    qw2 = w2.to(torch.float8_e4m3fn).view(torch.uint8)
    s1 = (
        torch.rand(
            (num_experts, (2 * inter + gn - 1) // gn, (hidden + gk - 1) // gk), device="cuda"
        )
        + 0.5
    )
    s2 = (
        torch.rand((num_experts, (hidden + gn - 1) // gn, (inter + gk - 1) // gk), device="cuda")
        + 0.5
    )
    ids = torch.randint(0, num_experts, (7, top_k), device="cuda")
    weights = torch.softmax(torch.randn(7, top_k, device="cuda", dtype=torch.float32), dim=-1)

    out = fused_moe(x, qw1, qw2, weights, ids, w1_scale=s1, w2_scale=s2, group_n=gn, group_k=gk)

    # Reference: dequantise (the e4m3 values are exact in fp32) and matmul in fp32.
    deq1 = qw1.view(torch.float8_e4m3fn).float() * s1.repeat_interleave(gn, 1).repeat_interleave(
        gk, 2
    )
    deq2 = qw2.view(torch.float8_e4m3fn).float() * s2.repeat_interleave(gn, 1).repeat_interleave(
        gk, 2
    )
    ref = fused_moe_reference(x, deq1, deq2, weights.to(act_dtype), ids)
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


# e2m1 code points indexed by nibble (bit3 sign, bits[2:1] exponent, bit0 mantissa).
_E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def _pack_mxfp4(nibbles: torch.Tensor) -> torch.Tensor:
    """``[E, N, K]`` uint8 nibbles -> ``[E, N, K//8]`` int32, low nibble = lower K."""
    b = (nibbles[..., 0::2] & 0xF) | ((nibbles[..., 1::2] & 0xF) << 4)
    return (
        b[..., 0::4].to(torch.int64)
        | (b[..., 1::4].to(torch.int64) << 8)
        | (b[..., 2::4].to(torch.int64) << 16)
        | (b[..., 3::4].to(torch.int64) << 24)
    ).to(torch.int32)


@pytest.mark.parametrize("act_dtype", [torch.float16, torch.bfloat16])
def test_fused_moe_mxfp4_matches_reference(act_dtype):
    """MXFP4 e2m1 expert weights with per-32 e8m0 scales (DeepSeek-V4 routed
    experts).

    GROUP_K (32) is narrower than BLOCK_K, so one k-tile spans four scale
    groups; the kernel must switch scales inside the tile. swiglu_limit
    exercises V4's bounded SwiGLU on the same launch.
    """
    torch.manual_seed(0)
    hidden, inter, num_experts, top_k = 256, 128, 8, 2
    limit = 7.0
    x = torch.randn(7, hidden, device="cuda", dtype=act_dtype) / hidden**0.5
    n1 = torch.randint(0, 16, (num_experts, 2 * inter, hidden), device="cuda", dtype=torch.uint8)
    n2 = torch.randint(0, 16, (num_experts, hidden, inter), device="cuda", dtype=torch.uint8)
    # e8m0 scales are powers of two.
    s1 = torch.exp2(
        torch.randint(-6, -1, (num_experts, 2 * inter, hidden // 32), device="cuda").float()
    )
    s2 = torch.exp2(
        torch.randint(-6, -1, (num_experts, hidden, inter // 32), device="cuda").float()
    )
    ids = torch.randint(0, num_experts, (7, top_k), device="cuda")
    weights = torch.softmax(torch.randn(7, top_k, device="cuda", dtype=torch.float32), dim=-1)

    out = fused_moe(
        x,
        _pack_mxfp4(n1),
        _pack_mxfp4(n2),
        weights,
        ids,
        w1_scale=s1,
        w2_scale=s2,
        group_n=1,
        group_k=32,
        swiglu_limit=limit,
        mxfp4=True,
    )

    # Reference: dequantise via the LUT and matmul in fp32, with V4's clamps.
    lut = torch.tensor(_E2M1_LUT, device="cuda")
    deq1 = lut[n1.long()] * s1.repeat_interleave(32, 2)
    deq2 = lut[n2.long()] * s2.repeat_interleave(32, 2)
    xf = x.float()
    ref = torch.zeros_like(xf)
    flat_ids = ids.reshape(-1)
    flat_weights = weights.reshape(-1)
    token_of_slot = torch.arange(x.shape[0], device="cuda").repeat_interleave(top_k)
    for e in flat_ids.unique():
        sel = flat_ids == e
        rows = token_of_slot[sel]
        gate_up = xf[rows] @ deq1[e].T
        gate = gate_up[:, :inter].clamp(max=limit)
        up = gate_up[:, inter:].clamp(min=-limit, max=limit)
        h = F.silu(gate) * up
        ref.index_add_(0, rows, (h @ deq2[e].T) * flat_weights[sel, None])

    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_fused_moe_mxfp4_rejects_non_32_group_k():
    x = torch.randn(2, 64, device="cuda", dtype=torch.float16)
    w1 = torch.zeros(4, 32, 64, device="cuda", dtype=torch.int32)
    w2 = torch.zeros(4, 64, 16, device="cuda", dtype=torch.int32)
    ids = torch.zeros(2, 2, device="cuda", dtype=torch.int32)
    weights = torch.ones(2, 2, device="cuda", dtype=torch.float16)
    scale = torch.ones(4, 32, 1, device="cuda")
    with pytest.raises(ValueError, match="group_k"):
        fused_moe(
            x,
            w1,
            w2,
            weights,
            ids,
            w1_scale=scale,
            w2_scale=scale,
            group_n=1,
            group_k=64,
            mxfp4=True,
        )
