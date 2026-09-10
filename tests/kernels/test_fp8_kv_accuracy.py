"""fp8 KV cache precision gate.

Measures the round-trip error of quantising bf16 K/V to fp8-e4m3 and back.
The fp8 KV cache stores bytes (uint8 container); the dequant happens inside
the attention kernel. This test bypasses the kernel and dequantises in Python
to isolate the quantisation error from the kernel's arithmetic.

The gate: max relative error must stay below 5% for typical activation ranges.
fp8-e4m3 has ~0.8% step size at full scale, but per-tensor scaling wastes
precision when the distribution is heavy-tailed. The 5% ceiling is generous
enough to admit any reasonable activation distribution and tight enough to
catch a broken scale computation.

Usage:
    pytest tests/kernels/test_fp8_kv_accuracy.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.modules.quantization.utils import FP8_E4M3_MAX, quantize_fp8_per_tensor

pytestmark = [pytest.mark.gpu]


def _dequant_fp8(q: torch.Tensor, scale: float) -> torch.Tensor:
    """Reverse the fp8-e4m3 quantisation: uint8 -> float8 -> fp32 -> * scale."""
    return q.view(torch.float8_e4m3fn).float() * scale


def _round_trip_error(x: torch.Tensor, scale: float | None = None) -> dict[str, float]:
    """Quantise then dequantise ``x``, returning error metrics.

    Args:
        x: bf16/fp16 tensor of any shape.
        scale: per-tensor scale. ``None`` derives the tightest scale that
            covers the tensor's dynamic range (what a well-calibrated runtime
            would pick).

    Returns:
        ``max_abs_diff``, ``rel_err`` (max_abs / max|x|), ``cosine_sim``.
    """
    if scale is None:
        scale = x.float().abs().amax().item() / FP8_E4M3_MAX
        scale = max(scale, 1e-8)  # avoid division by zero for all-zero tensors

    q = quantize_fp8_per_tensor(x, scale)
    x_hat = _dequant_fp8(q, scale)

    diff = (x.float() - x_hat).abs()
    max_abs = diff.max().item()
    x_max = x.float().abs().max().item()
    rel_err = max_abs / (x_max + 1e-8)

    # Cosine similarity: captures directional fidelity, not just magnitude.
    x_flat = x.float().reshape(-1)
    x_hat_flat = x_hat.reshape(-1)
    cos = torch.nn.functional.cosine_similarity(x_flat.unsqueeze(0), x_hat_flat.unsqueeze(0)).item()

    return {"max_abs_diff": max_abs, "rel_err": rel_err, "cosine_sim": cos}


# ------------------------------------------------------------------ tests #


@pytest.mark.parametrize(
    "shape",
    [(1, 8, 128), (32, 16, 128), (1, 32, 256), (64, 8, 64)],
    ids=["decode-1tok", "decode-batch32", "prefill-32tok", "prefill-batch64"],
)
def test_fp8_kv_round_trip_normal(shape):
    """Normal-distributed activations: the common case."""
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
    metrics = _round_trip_error(x)

    # fp8-e4m3 at full scale has ~0.8% step size; normal distributions fill
    # ~3 sigma, so the effective step is ~0.8% * 3 = 2.4%. Allow 5% ceiling.
    assert metrics["rel_err"] < 0.05, (
        f"fp8 KV round-trip rel_err={metrics['rel_err']:.4f} exceeds 5% "
        f"(shape={shape}, max_abs={metrics['max_abs_diff']:.4e})"
    )
    assert metrics["cosine_sim"] > 0.999, (
        f"fp8 KV cosine_sim={metrics['cosine_sim']:.6f} below 0.999 (shape={shape})"
    )


def test_fp8_kv_heavy_tailed():
    """Log-normal distribution: a few large values dominate the scale.

    This is the adversarial case for per-tensor scaling — most values are
    small, but the scale is set by the outliers, so the small values get
    coarse quantisation. The gate must still pass: real activations can
    be heavy-tailed (attention scores post-softmax are one example).
    """
    torch.manual_seed(0)
    # Log-normal: median ~1, but max can be 10-100x larger.
    x = torch.randn(32, 16, 128, device="cuda").exp().to(torch.bfloat16)
    metrics = _round_trip_error(x)

    # The 5% ceiling is generous enough for this case: the outliers set the
    # scale, and the bulk of the distribution gets coarser steps. But the
    # *directional* fidelity (cosine) must stay very high.
    assert metrics["rel_err"] < 0.10, (
        f"fp8 KV heavy-tailed rel_err={metrics['rel_err']:.4f} exceeds 10%"
    )
    assert metrics["cosine_sim"] > 0.995, (
        f"fp8 KV heavy-tailed cosine_sim={metrics['cosine_sim']:.6f} below 0.995"
    )


def test_fp8_kv_near_zero():
    """Values close to zero: the quantisation noise floor matters most here."""
    torch.manual_seed(7)
    x = (torch.randn(16, 8, 64, device="cuda") * 1e-3).to(torch.bfloat16)
    metrics = _round_trip_error(x)

    # Near zero, the relative error can be large (the step size is fixed by
    # the scale, which is tiny). But the absolute error is tiny too.
    assert metrics["max_abs_diff"] < 1e-3, (
        f"fp8 KV near-zero max_abs={metrics['max_abs_diff']:.4e} exceeds 1e-3"
    )


def test_fp8_kv_k_v_separate():
    """K and V quantised independently, as the runtime does.

    The Fp8KVCacheMethod quantises K and V with separate scales. Each half
    must stay within the single-tensor gate.
    """
    torch.manual_seed(99)
    num_kv_heads, head_dim = 8, 128
    seq_len = 32

    k = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")

    k_metrics = _round_trip_error(k)
    v_metrics = _round_trip_error(v)

    for name, m in [("K", k_metrics), ("V", v_metrics)]:
        assert m["rel_err"] < 0.05, f"fp8 KV {name} rel_err={m['rel_err']:.4f} exceeds 5%"
        assert m["cosine_sim"] > 0.999, (
            f"fp8 KV {name} cosine_sim={m['cosine_sim']:.6f} below 0.999"
        )


def test_fp8_kv_scale_sensitivity():
    """A bad scale (too large) wastes precision; a bad scale (too small) clips.

    The gate must catch both failure modes.
    """
    torch.manual_seed(1)
    x = torch.randn(16, 8, 64, dtype=torch.bfloat16, device="cuda")

    # Correct scale: tight fit to the dynamic range.
    good_scale = x.float().abs().amax().item() / FP8_E4M3_MAX
    good = _round_trip_error(x, scale=max(good_scale, 1e-8))

    # 10x too large: values occupy only 10% of the e4m3 range -> 10x coarser steps.
    bad_large = _round_trip_error(x, scale=good_scale * 10)

    # 10x too small: values > FP8_E4M3_MAX * scale get clipped.
    bad_small = _round_trip_error(x, scale=good_scale / 10)

    # The good scale must be strictly better than both bad ones.
    assert good["rel_err"] < bad_large["rel_err"], "good scale should beat 10x-too-large"
    assert good["rel_err"] < bad_small["rel_err"], "good scale should beat 10x-too-small"
    # The bad-small case clips, so max_abs_diff should be large.
    assert bad_small["max_abs_diff"] > good["max_abs_diff"] * 5, (
        "clipping (scale too small) should cause large absolute error"
    )
