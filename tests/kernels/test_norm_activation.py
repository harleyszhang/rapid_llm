"""Tests for the fused norm and activation kernels.

``skip_rmsnorm`` is checked for value, post-add residual and unit-RMS
invariants, ``qk_rmsnorm`` for bit-identity against two ``skip_rmsnorm``
calls; the SwiGLU pair (split and fused) against eager references including
the zero-gate edge.

Usage:
    pytest tests/kernels/test_norm_activation.py
"""

from __future__ import annotations

import pytest
import torch

from rapid_llm.kernels import (
    qk_rmsnorm,
    skip_rmsnorm,
    swiglu_forward,
    swiglu_forward_fused,
    swiglu_forward_fused_bounded,
)
from tests import reference

_RTOL, _ATOL = 2e-2, 2e-2


# --------------------------------------------------------------------------- #
# skip_rmsnorm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 512), id="single-token"),
        pytest.param((2, 16, 1024), id="typical-prefill"),
        pytest.param((1, 7, 896), id="non-power-of-two-hidden"),
        pytest.param((4, 1, 2048), id="decode-batch"),
    ],
)
def test_skip_rmsnorm_matches_reference(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    residual = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)

    # The kernel overwrites ``residual`` in place, so the reference has to run
    # against pristine copies (see test_skip_rmsnorm_writes_residual_in_place).
    ref_out, ref_residual = reference.skip_rmsnorm(x.clone(), residual.clone(), weight)
    out, new_residual = skip_rmsnorm(x, residual, weight)

    torch.testing.assert_close(out.float(), ref_out.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(new_residual.float(), ref_residual.float(), rtol=_RTOL, atol=_ATOL)


def test_skip_rmsnorm_returns_the_post_add_residual():
    """The second return value must be ``x + residual``, not ``residual``.

    Returning the un-added residual is the plausible-but-wrong variant: it keeps
    shapes and dtypes intact, so only a value assertion catches it.
    """
    shape = (1, 4, 256)
    x = torch.full(shape, 2.0, device="cuda", dtype=torch.float16)
    residual = torch.full(shape, 3.0, device="cuda", dtype=torch.float16)
    weight = torch.ones(shape[-1], device="cuda", dtype=torch.float16)

    _, new_residual = skip_rmsnorm(x, residual, weight)
    assert torch.allclose(
        new_residual.float(), torch.full(shape, 5.0, device="cuda"), rtol=_RTOL, atol=_ATOL
    )


def test_skip_rmsnorm_without_residual_is_plain_rmsnorm():
    """``residual=None`` degenerates to RMSNorm and echoes ``x`` back."""
    shape = (2, 8, 512)
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)

    out, passthrough = skip_rmsnorm(x, None, weight)
    ref_out, _ = reference.skip_rmsnorm(x, None, weight)

    torch.testing.assert_close(out.float(), ref_out.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(passthrough.float(), x.float(), rtol=_RTOL, atol=_ATOL)


def test_skip_rmsnorm_normalises_to_unit_rms():
    """With unit weight the output RMS must be ~1; that is the kernel's contract.

    Checked independently of the reference so a shared misunderstanding of the
    formula cannot make both sides agree on the wrong answer.
    """
    x = torch.randn(1, 32, 1024, device="cuda", dtype=torch.float16) * 5.0
    weight = torch.ones(1024, device="cuda", dtype=torch.float16)
    out, _ = skip_rmsnorm(x, None, weight)
    rms = out.float().pow(2).mean(dim=-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=5e-2, atol=5e-2)


# --------------------------------------------------------------------------- #
# qk_rmsnorm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "geometry",
    [
        pytest.param((1, 32, 8, 128), id="decode-gqa"),
        pytest.param((64, 32, 8, 128), id="prefill-gqa"),
        pytest.param((16, 16, 16, 128), id="mha"),
        pytest.param((3, 5, 3, 96), id="non-power-of-two"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qk_rmsnorm_is_bit_identical_to_two_skip_rmsnorm(geometry, dtype):
    """One fused launch must reproduce two ``skip_rmsnorm`` calls byte for byte.

    The fusion exists to remove a launch, not to change arithmetic: it keeps
    ``rms_norm_kernel``'s 1-D tile shape and warp count, so the fp32 variance
    reduction is bit-identical and no generated token moves. A tolerance-based
    comparison would pass while silently invalidating every golden baseline, so
    the assertion is exact equality.
    """
    tokens, n_qh, n_kh, head_dim = geometry
    torch.manual_seed(0)
    eps = 1e-5
    q = torch.randn(tokens, n_qh, head_dim, device="cuda", dtype=dtype) * 0.5
    k = torch.randn(tokens, n_kh, head_dim, device="cuda", dtype=dtype) * 0.5
    q_weight = torch.randn(head_dim, device="cuda", dtype=dtype) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, device="cuda", dtype=dtype) * 0.1 + 1.0

    ref_q, _ = skip_rmsnorm(q, None, q_weight, eps)
    ref_k, _ = skip_rmsnorm(k, None, k_weight, eps)
    out_q, out_k = qk_rmsnorm(q, k, q_weight, k_weight, eps)

    assert torch.equal(ref_q, out_q)
    assert torch.equal(ref_k, out_k)


def test_qk_rmsnorm_matches_eager_reference():
    """Also checked against the eager formula, not only against the kernel.

    Comparing two kernels that share a bug would pass; the reference is the
    independent spelling of RMSNorm.
    """
    tokens, n_qh, n_kh, head_dim = 2, 8, 2, 128
    q = torch.randn(tokens, n_qh, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(tokens, n_kh, head_dim, device="cuda", dtype=torch.float16)
    q_weight = torch.randn(head_dim, device="cuda", dtype=torch.float16)
    k_weight = torch.randn(head_dim, device="cuda", dtype=torch.float16)

    out_q, out_k = qk_rmsnorm(q, k, q_weight, k_weight)
    ref_q, _ = reference.skip_rmsnorm(q.reshape(-1, head_dim), None, q_weight)
    ref_k, _ = reference.skip_rmsnorm(k.reshape(-1, head_dim), None, k_weight)

    torch.testing.assert_close(
        out_q.reshape(-1, head_dim).float(), ref_q.float(), rtol=_RTOL, atol=_ATOL
    )
    torch.testing.assert_close(
        out_k.reshape(-1, head_dim).float(), ref_k.float(), rtol=_RTOL, atol=_ATOL
    )


def test_qk_rmsnorm_rejects_mismatched_head_dim():
    """q and k share one BLOCK_SIZE, so differing head_dim must fail loudly."""
    q = torch.randn(1, 4, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        qk_rmsnorm(q, k, torch.ones(128, device="cuda"), torch.ones(64, device="cuda"))


# --------------------------------------------------------------------------- #
# swiglu_forward
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 256), id="single-token"),
        pytest.param((2, 16, 4864), id="qwen2-intermediate"),
        pytest.param((1, 5, 300), id="non-power-of-two"),
    ],
)
def test_swiglu_matches_reference(shape):
    gate = torch.randn(shape, device="cuda", dtype=torch.float16)
    up = torch.randn(shape, device="cuda", dtype=torch.float16)

    out = swiglu_forward(gate, up)
    ref = reference.swiglu(gate, up)

    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)
    assert out.shape == gate.shape


def test_swiglu_zero_gate_gives_zero():
    """silu(0) == 0, so a zero gate must annihilate the up projection."""
    gate = torch.zeros(1, 4, 128, device="cuda", dtype=torch.float16)
    up = torch.randn(1, 4, 128, device="cuda", dtype=torch.float16)
    out = swiglu_forward(gate, up)
    assert torch.count_nonzero(out) == 0


# --------------------------------------------------------------------------- #
# swiglu_forward_fused
#
# The merged gate/up GEMM emits one [..., 2 * inter] tensor; this variant must
# read the halves out of it without the caller splitting them first.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 256), id="single-token"),
        pytest.param((2, 16, 4864), id="qwen2-intermediate"),
        pytest.param((1, 5, 300), id="non-power-of-two"),
    ],
)
def test_swiglu_fused_matches_split_reference(shape):
    """The fused layout must agree with activating the halves separately."""
    gate = torch.randn(shape, device="cuda", dtype=torch.float16)
    up = torch.randn(shape, device="cuda", dtype=torch.float16)
    fused = torch.cat([gate, up], dim=-1)

    out = swiglu_forward_fused(fused)
    ref = swiglu_forward(gate, up)

    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)
    assert out.shape == gate.shape
    assert fused.shape[-1] == 2 * gate.shape[-1]  # input consumed in place, no split


def test_swiglu_fused_zero_gate_gives_zero():
    """A zero gate half must annihilate the up half even in the fused layout."""
    inter = 128
    fused = torch.cat([torch.zeros(1, 4, inter), torch.randn(1, 4, inter)], dim=-1).to(
        device="cuda", dtype=torch.float16
    )
    out = swiglu_forward_fused(fused)
    assert torch.count_nonzero(out) == 0


def test_swiglu_fused_applies_bound_with_bfloat16_rounding():
    """The bounded path follows clamp, bf16 SiLU, then multiply ordering."""
    torch.manual_seed(9)
    fused = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    gate, up = fused.chunk(2, dim=-1)
    limit = 0.75

    actual = swiglu_forward_fused_bounded(fused, limit)
    expected = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(
        min=-limit, max=limit
    )

    assert torch.equal(actual, expected)


def test_skip_rmsnorm_writes_residual_in_place():
    """The kernel stores ``x + residual`` back into the caller's tensor.

    This aliasing is load-bearing: the decoder layer relies on it to thread the
    running residual forward without an extra allocation. It also means a caller
    that needs the original residual afterwards must clone it first, so the
    behaviour is pinned here rather than left as folklore.
    """
    shape = (1, 4, 256)
    x = torch.full(shape, 2.0, device="cuda", dtype=torch.float16)
    residual = torch.full(shape, 3.0, device="cuda", dtype=torch.float16)
    weight = torch.ones(shape[-1], device="cuda", dtype=torch.float16)

    _, returned = skip_rmsnorm(x, residual, weight)

    expected = torch.full(shape, 5.0, device="cuda")
    torch.testing.assert_close(residual.float(), expected, rtol=_RTOL, atol=_ATOL)
    assert returned.data_ptr() == residual.data_ptr()
