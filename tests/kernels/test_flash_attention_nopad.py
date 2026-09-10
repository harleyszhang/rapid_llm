"""Numerical tests for the prefill attention kernel.

Ragged batches, GQA ratios and head dims are parametrised against the
pure-torch reference; isolation cases prove the first token attends
only to itself and sequences do not leak.

Usage:
    pytest tests/kernels/test_flash_attention_nopad.py
"""

from __future__ import annotations

import math

import pytest
import torch

from rapid_llm.kernels import flash_attention2_no_pad
from tests.reference import varlen_causal_attention

# The kernel keeps an fp32 accumulator, but inputs and the PV product are fp16,
# so error scales with sequence length and head_dim.
_RTOL, _ATOL = 2e-2, 2e-2


def _packed_qkv(seq_lens, num_q_heads, num_kv_heads, head_dim):
    """Build a packed varlen batch plus its offset/length metadata."""
    total = sum(seq_lens)
    # Scaled down so fp16 qk products stay well inside range.
    q = torch.randn(total, num_q_heads, head_dim, device="cuda", dtype=torch.float16) * 0.3
    k = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.3
    v = torch.randn(total, num_kv_heads, head_dim, device="cuda", dtype=torch.float16) * 0.3

    starts, offset = [], 0
    for n in seq_lens:
        starts.append(offset)
        offset += n

    b_start_loc = torch.tensor(starts, dtype=torch.int32, device="cuda")
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return q, k, v, b_start_loc, b_seq_len


def _run(q, k, v, b_start_loc, b_seq_len, head_dim):
    """Invoke kernel and reference with the same plain softmax scale."""
    plain_scale = 1.0 / math.sqrt(head_dim)
    out = flash_attention2_no_pad(
        q, k, v, plain_scale, b_start_loc, b_seq_len, int(b_seq_len.max())
    )
    ref = varlen_causal_attention(q, k, v, b_start_loc, b_seq_len, plain_scale)
    return out, ref


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([16], id="single-short"),
        pytest.param([64], id="single-exact-block"),
        pytest.param([100], id="single-ragged"),
        pytest.param([32, 32], id="two-equal"),
        pytest.param([7, 100, 33], id="three-ragged"),
        pytest.param([1, 64], id="len1-plus-block"),
    ],
)
def test_matches_reference_across_lengths(seq_lens):
    """Ragged lengths are the point: BLOCK_M is 64, so 7/33/100 all straddle it.

    A length-1 sequence next to a full block is what a packed-vs-padded offset
    bug corrupts first.
    """
    head_dim = 64
    out, ref = _run(*_packed_qkv(seq_lens, 4, 4, head_dim), head_dim)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads",
    [
        pytest.param(4, 4, id="mha"),
        pytest.param(8, 2, id="gqa-4x"),
        pytest.param(14, 2, id="gqa-7x"),
        pytest.param(8, 1, id="mqa"),
    ],
)
def test_matches_reference_across_gqa_ratios(num_q_heads, num_kv_heads):
    """Query head ``h`` must read KV head ``h // groups``.

    Getting this backwards (tiling instead of interleaving) still produces
    finite, plausible output, so only a reference comparison catches it. The 7x
    ratio is deliberately not a power of two.
    """
    head_dim = 64
    out, ref = _run(*_packed_qkv([48, 16], num_q_heads, num_kv_heads, head_dim), head_dim)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize("head_dim", [32, 64, 128])
def test_matches_reference_across_head_dims(head_dim):
    """``num_warps`` switches at head_dim 64, so both sides need covering."""
    out, ref = _run(*_packed_qkv([40, 24], 4, 2, head_dim), head_dim)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_first_token_attends_only_itself():
    """Position 0 may only attend itself, so its output must equal ``v[0]``.

    This pins causality without relying on the reference: if the mask leaked,
    row 0 would blend in later values and drift away from ``v[0]``.
    """
    head_dim, seq_len = 64, 32
    q, k, v, b_start_loc, b_seq_len = _packed_qkv([seq_len], 4, 4, head_dim)
    out = flash_attention2_no_pad(
        q, k, v, 1.0 / math.sqrt(head_dim), b_start_loc, b_seq_len, seq_len
    )
    torch.testing.assert_close(out[0].float(), v[0].float(), rtol=_RTOL, atol=_ATOL)


def test_sequences_do_not_leak_into_each_other():
    """Perturbing sequence 1's values must leave sequence 0's output untouched.

    The packed layout puts both sequences in one tensor, so a wrong
    ``b_start_loc`` stride shows up here as a changed first sequence.
    """
    head_dim = 64
    seq_lens = [24, 40]
    q, k, v, b_start_loc, b_seq_len = _packed_qkv(seq_lens, 4, 4, head_dim)
    scale = 1.0 / math.sqrt(head_dim)
    max_len = max(seq_lens)
    split = seq_lens[0]

    before = flash_attention2_no_pad(q, k, v, scale, b_start_loc, b_seq_len, max_len)

    v_perturbed = v.clone()
    v_perturbed[split:] += 1.0
    after = flash_attention2_no_pad(q, k, v_perturbed, scale, b_start_loc, b_seq_len, max_len)

    torch.testing.assert_close(
        before[:split].float(), after[:split].float(), rtol=_RTOL, atol=_ATOL
    )
    # Guard against a vacuous pass: the second sequence really did change.
    assert not torch.allclose(before[split:].float(), after[split:].float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.skipif(
    not __import__("rapid_llm.kernels.backend.flashinfer", fromlist=["available"]).available(),
    reason="flashinfer not installed",
)
@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([13, 15, 12], id="ragged"),
        pytest.param([3, 5], id="uneven-pair"),
        pytest.param([32, 32], id="equal-lengths"),
    ],
)
def test_flashinfer_prefill_survives_the_padded_grid(seq_lens):
    """The engine pads every prefill chunk to the widest (slot_batch).

    A ragged wrapper wants compact rows, so unequal lengths cannot reach it:
    the flashinfer backend must hand the pass to this kernel — and its answer
    on the real rows must still match the reference. Equal lengths used to be
    the only shapes that worked; they still do.
    """
    from rapid_llm.kernels.backend.flashinfer.attention import prefill_attention

    width = max(seq_lens)
    rows = len(seq_lens) * width
    head_dim = 64
    # Junk in the padding rows: the kernels must never let it reach the output
    # of a real row (the engine overwrites those rows on later steps).
    q = torch.randn(rows, 4, head_dim, device="cuda", dtype=torch.float16) * 0.3
    k = torch.randn(rows, 2, head_dim, device="cuda", dtype=torch.float16) * 0.3
    v = torch.randn(rows, 2, head_dim, device="cuda", dtype=torch.float16) * 0.3

    b_start_loc = torch.tensor(
        [i * width for i in range(len(seq_lens))], dtype=torch.int32, device="cuda"
    )
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    plain_scale = 1.0 / math.sqrt(head_dim)

    out = prefill_attention(q, k, v, plain_scale, b_start_loc, b_seq_len, width)
    ref = varlen_causal_attention(q, k, v, b_start_loc, b_seq_len, plain_scale)

    for i, n in enumerate(seq_lens):  # compare only the real rows
        sl = slice(b_start_loc[i].item(), b_start_loc[i].item() + n)
        torch.testing.assert_close(out[sl].float(), ref[sl].float(), rtol=_RTOL, atol=_ATOL)
