"""Numerical tests for the native MLA decode/prefill kernels.

Decode is diffed against the in-file PyTorch reference (the semantic definer)
across page sizes, history lengths and scattered block tables; prefill against
an fp32 up-projection + causal-attention oracle, with the upsample chunk
shrunk to force chunk crossings.

Usage:
    pytest tests/kernels/test_mla.py
"""

from __future__ import annotations

import math

import pytest
import torch

from rapid_llm.kernels.ops.attention import mla
from rapid_llm.kernels.ops.attention.mla import (
    QK_ROPE_HEAD_DIM,
    mla_decode,
    mla_decode_reference,
    mla_prefill,
)

_LORA = 512
_ROPE = QK_ROPE_HEAD_DIM
_QK_DIM = _LORA + _ROPE
# Both sides accumulate in fp32; only the reduction order differs.
_RTOL, _ATOL = 1e-2, 1e-2


def _latent_cache(seq_lens, page_size, lora=_LORA, *, scattered=False):
    """Allocate a paged latent pool and map each sequence's tokens into it.

    Returns:
        ``(kv_cache, block_table, cache_seqlens)`` with the cache
        ``[num_pages, page_size, lora + rope]``.
    """
    pages_per_seq = [(n + page_size - 1) // page_size for n in seq_lens]
    num_pages = sum(pages_per_seq)
    kv_cache = torch.randn(num_pages, page_size, lora + _ROPE, device="cuda", dtype=torch.bfloat16)
    max_pages = max(pages_per_seq)
    block_table = torch.zeros(len(seq_lens), max_pages, dtype=torch.int32, device="cuda")
    if scattered:
        # One shared permutation carved into per-sequence slices: disjoint but
        # non-monotonic page ids, the layout a real pool reaches after churn.
        perm = torch.randperm(num_pages, device="cuda").to(torch.int32)
        offset = 0
        for i, count in enumerate(pages_per_seq):
            block_table[i, :count] = perm[offset : offset + count]
            offset += count
    else:
        offset = 0
        for i, count in enumerate(pages_per_seq):
            block_table[i, :count] = torch.arange(
                offset, offset + count, dtype=torch.int32, device="cuda"
            )
            offset += count
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    return kv_cache, block_table, cache_seqlens


def _run_decode(seq_lens, num_heads, page_size, lora=_LORA, *, scattered=False, q=None):
    """Run kernel and reference over the same random latent cache."""
    kv_cache, block_table, cache_seqlens = _latent_cache(
        seq_lens, page_size, lora, scattered=scattered
    )
    if q is None:
        q = (
            torch.randn(len(seq_lens), num_heads, lora + _ROPE, device="cuda", dtype=torch.bfloat16)
            * 0.3
        )
    scale = 1.0 / math.sqrt(lora + _ROPE)
    max_len = max(seq_lens)
    out = mla_decode(q, kv_cache, block_table, cache_seqlens, max_seq_len=max_len, sm_scale=scale)
    ref = mla_decode_reference(
        q, kv_cache, block_table, cache_seqlens, max_seq_len=max_len, sm_scale=scale
    )
    return out, ref


# --------------------------------------------------------------------------- #
# decode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "seq_lens",
    [
        pytest.param([1], id="len1"),
        pytest.param([15], id="below-block-n"),
        pytest.param([16], id="exact-block-n"),
        pytest.param([17], id="above-block-n"),
        pytest.param([128], id="exact-partition"),
        pytest.param([129], id="above-partition"),
        pytest.param([200], id="mid-second-partition"),
        pytest.param([256], id="two-full-partitions"),
    ],
)
def test_decode_matches_reference_across_history_lengths(seq_lens):
    """Lengths straddle BLOCK_N (16) and the 128-row partition boundary."""
    out, ref = _run_decode(seq_lens, 16, page_size=16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


@pytest.mark.parametrize(
    "page_size",
    [
        pytest.param(1, id="page1-mainline-pool"),
        pytest.param(4, id="tiny-pages-straddle-block-n"),
        pytest.param(16, id="page16"),
        pytest.param(128, id="page-equals-partition"),
    ],
)
def test_decode_matches_reference_across_page_sizes(page_size):
    """The page walk must hold when one BLOCK_N window spans several pages."""
    out, ref = _run_decode([64, 130], 8, page_size=page_size)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_matches_reference_with_scattered_pages():
    """Page ids need not be contiguous or ordered — the table is the truth."""
    out, ref = _run_decode([33, 130], 8, page_size=16, scattered=True)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_matches_reference_with_ragged_batch():
    out, ref = _run_decode([17, 129, 64, 3], 16, page_size=16)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_matches_reference_with_smaller_lora():
    """The kernel is generic over the (power-of-two) lora rank."""
    out, ref = _run_decode([40, 150], 8, page_size=16, lora=256)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_return_lse_matches_reference_and_handles_empty_history():
    seq_lens = [0, 129]
    kv_cache, block_table, cache_seqlens = _latent_cache(seq_lens, 16)
    q = torch.randn(2, 8, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    scale = 1.0 / math.sqrt(_QK_DIM)

    output, lse = mla_decode(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        max_seq_len=max(seq_lens),
        sm_scale=scale,
        return_lse=True,
    )
    expected_output, expected_lse = mla_decode_reference(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        max_seq_len=max(seq_lens),
        sm_scale=scale,
        return_lse=True,
    )

    torch.testing.assert_close(output.float(), expected_output.float(), rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(lse[1], expected_lse[1], rtol=_RTOL, atol=_ATOL)
    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.isneginf(lse[0]).all()


def test_decode_empty_batch_max_returns_without_launching_kernel():
    kv_cache, block_table, cache_seqlens = _latent_cache([0, 0], 16)
    q = torch.randn(2, 8, _QK_DIM, device="cuda", dtype=torch.bfloat16)

    output, lse = mla_decode(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        max_seq_len=0,
        return_lse=True,
    )

    assert torch.equal(output, torch.zeros_like(output))
    assert torch.isneginf(lse).all()


def test_decode_strided_table_shares_merge_to_the_full_attention():
    """DCP's latent decode on one GPU: each rank attends ``table[:, r::dcp]``
    (page size 1, the engine's pool) with its local length; the LSE merge of
    the shares must equal attention over the whole history.
    """
    from rapid_llm.distributed.dcp import dcp_local_seq_len, merge_attention_partials

    dcp = 2
    seq_lens = [1, 64, 131]
    kv_cache, block_table, cache_seqlens = _latent_cache(seq_lens, 1, scattered=True)
    q = torch.randn(len(seq_lens), 8, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    scale = 1.0 / math.sqrt(_QK_DIM)

    outputs, lses = [], []
    for rank in range(dcp):
        output, lse = mla_decode(
            q,
            kv_cache,
            block_table[:, rank::dcp],
            dcp_local_seq_len(cache_seqlens, rank, dcp),
            max_seq_len=dcp_local_seq_len(max(seq_lens), 0, dcp),
            sm_scale=scale,
            return_lse=True,
        )
        outputs.append(output.float())
        lses.append(lse)
    merged, _ = merge_attention_partials(torch.stack(outputs), torch.stack(lses))

    ref = mla_decode_reference(
        q, kv_cache, block_table, cache_seqlens, max_seq_len=max(seq_lens), sm_scale=scale
    )
    torch.testing.assert_close(merged, ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_tolerates_oversized_max_seq_len():
    """A grid sized beyond every row's history must not write or read there.

    The no-store path in stage 1 is the only thing standing between a padded
    grid and uninitialised ``mid_o`` rows leaking into the stage-2 reduction.
    """
    seq_lens = [5, 40]
    kv_cache, block_table, cache_seqlens = _latent_cache(seq_lens, 16)
    q = torch.randn(2, 8, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    scale = 1.0 / math.sqrt(_QK_DIM)
    out = mla_decode(q, kv_cache, block_table, cache_seqlens, max_seq_len=384, sm_scale=scale)
    ref = mla_decode_reference(
        q, kv_cache, block_table, cache_seqlens, max_seq_len=384, sm_scale=scale
    )
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_rope_segment_alone_drives_scores():
    """Zero the nope part: scores must come from ``k_pe`` alone.

    If the kernel dropped the rope segment, every position would score 0 and
    the output would be the uniform average — which the reference only agrees
    with when the k_pe column happens to be constant.
    """
    seq_lens = [50, 90]
    kv_cache, block_table, cache_seqlens = _latent_cache(seq_lens, 16)
    q = torch.randn(2, 8, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    q[..., :_LORA] = 0.0
    scale = 1.0 / math.sqrt(_QK_DIM)
    out = mla_decode(q, kv_cache, block_table, cache_seqlens, max_seq_len=90, sm_scale=scale)
    ref = mla_decode_reference(
        q, kv_cache, block_table, cache_seqlens, max_seq_len=90, sm_scale=scale
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=_RTOL, atol=_ATOL)


def test_decode_single_token_history_returns_its_latent_row():
    """Attention over one row is the identity on its ``c_kv`` half.

    Exact, not approximate: exp(0) == 1 through both stages, so the bf16 row
    round-trips through fp32 unchanged.
    """
    kv_cache, block_table, cache_seqlens = _latent_cache([1, 1], 16)
    q = torch.randn(2, 16, _QK_DIM, device="cuda", dtype=torch.bfloat16)
    out = mla_decode(
        q, kv_cache, block_table, cache_seqlens, max_seq_len=1, sm_scale=1.0 / math.sqrt(_QK_DIM)
    )
    expected = torch.stack([kv_cache[int(block_table[b, 0]), 0, :_LORA] for b in range(2)])
    expected = expected.unsqueeze(1).expand(-1, 16, -1)
    assert torch.equal(out, expected)


def test_decode_uniform_latent_reproduces_the_value():
    """Identical latent rows pin the softmax normalisation, reference-free."""
    seq_len, num_heads = 150, 8
    page_size = 16
    num_pages = (seq_len + page_size - 1) // page_size
    latent_row = torch.randn(_QK_DIM, device="cuda", dtype=torch.bfloat16)
    kv_cache = latent_row.expand(num_pages, page_size, _QK_DIM).contiguous()
    block_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    cache_seqlens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    q = torch.randn(1, num_heads, _QK_DIM, device="cuda", dtype=torch.bfloat16) * 0.3
    out = mla_decode(
        q,
        kv_cache,
        block_table,
        cache_seqlens,
        max_seq_len=seq_len,
        sm_scale=1.0 / math.sqrt(_QK_DIM),
    )
    expected = latent_row[:_LORA].expand(1, num_heads, _LORA)
    torch.testing.assert_close(out.float(), expected.float(), rtol=_RTOL, atol=_ATOL)


# --------------------------------------------------------------------------- #
# prefill
# --------------------------------------------------------------------------- #

_NOPE = 128
_V_DIM = 128


def _prefill_inputs(seq_lens, num_heads, lora=_LORA):
    """Fresh latent + projection weights at realistic magnitudes.

    The up-projections are scaled by ``1/sqrt(lora)`` the way trained weights
    keep activations unit-scale, so the attention operates in a sane softmax
    regime.
    """
    total = sum(seq_lens)
    c_kv = torch.randn(total, lora, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.randn(total, _ROPE, device="cuda", dtype=torch.bfloat16)
    q_nope = torch.randn(total, num_heads, _NOPE, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn(total, num_heads, _ROPE, device="cuda", dtype=torch.bfloat16)
    w_uk = torch.randn(num_heads, lora, _NOPE, device="cuda", dtype=torch.bfloat16) / math.sqrt(
        lora
    )
    w_uv = torch.randn(num_heads, lora, _V_DIM, device="cuda", dtype=torch.bfloat16) / math.sqrt(
        lora
    )
    b_seq_len = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    starts = [0]
    for n in seq_lens[:-1]:
        starts.append(starts[-1] + n)
    b_start_loc = torch.tensor(starts, dtype=torch.int32, device="cuda")
    return q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, b_start_loc, b_seq_len


def _prefill_fp32_reference(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len):
    """fp32 ground truth: full up-projection, then per-sequence causal softmax."""
    tokens, num_heads, _ = q_nope.shape
    k_nope = torch.einsum("tl,hld->thd", c_kv.float(), w_uk.float())
    v = torch.einsum("tl,hld->thd", c_kv.float(), w_uv.float())
    k = torch.cat([k_nope, k_pe[:, None, :].float().expand(tokens, num_heads, _ROPE)], dim=-1)
    q = torch.cat([q_nope, q_pe], dim=-1).float()
    out = torch.empty((tokens, num_heads, _V_DIM), dtype=torch.float32, device=q.device)
    for i in range(b_seq_len.shape[0]):
        start, length = int(b_start_loc[i]), int(b_seq_len[i])
        sl = slice(start, start + length)
        scores = (
            q[sl].transpose(0, 1) @ k[sl].transpose(0, 1).transpose(-1, -2)
        ) * sm_scale  # [H, n, n]
        causal = torch.ones(length, length, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        out[sl] = (scores.softmax(dim=-1) @ v[sl].transpose(0, 1)).transpose(0, 1)
    return out


@pytest.mark.parametrize(
    "seq_lens,num_heads",
    [
        pytest.param([5], 16, id="short-single"),
        pytest.param([130], 16, id="crosses-block-m"),
        pytest.param([17, 129, 64, 3], 8, id="ragged-batch"),
    ],
)
def test_prefill_matches_fp32_reference(seq_lens, num_heads):
    """Chunked-upsample kernel path vs the fp32 full-upsample oracle."""
    inputs = _prefill_inputs(seq_lens, num_heads)
    q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, b_start_loc, b_seq_len = inputs
    sm_scale = 1.0 / math.sqrt(_NOPE + _ROPE)
    out = mla_prefill(
        q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len, max(seq_lens)
    )
    ref = _prefill_fp32_reference(
        q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len
    )
    assert out.shape == (sum(seq_lens), num_heads, _V_DIM)
    torch.testing.assert_close(out.float(), ref, rtol=_RTOL, atol=_ATOL)


def test_prefill_chunk_boundary_is_invisible(monkeypatch):
    """Shrinking the upsample chunk below the token count must not move the output.

    Chunking is a workspace bound, not a numerics feature: the same einsum
    runs per slice, so any drift beyond bf16 noise means a slice indexing bug.
    """
    inputs = _prefill_inputs([100, 60], 8)
    q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, b_start_loc, b_seq_len = inputs
    sm_scale = 1.0 / math.sqrt(_NOPE + _ROPE)
    ref = _prefill_fp32_reference(
        q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len
    )
    monkeypatch.setattr(mla, "_PREFILL_UPSAMPLE_CHUNK", 37)
    out = mla_prefill(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len, 100)
    torch.testing.assert_close(out.float(), ref, rtol=_RTOL, atol=_ATOL)


def test_prefill_single_token_sequences_return_their_value_rows():
    """One-token sequences make attention the identity on the upsampled V."""
    inputs = _prefill_inputs([1, 1, 1], 8)
    q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, b_start_loc, b_seq_len = inputs
    sm_scale = 1.0 / math.sqrt(_NOPE + _ROPE)
    out = mla_prefill(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, sm_scale, b_start_loc, b_seq_len, 1)
    # fp32 einsum rounded once — the kernel's own bf16 einsum may differ by a
    # rounding, so this stays within tolerance rather than asserting equality.
    expected = torch.einsum("tl,hld->thd", c_kv.float(), w_uv.float())
    torch.testing.assert_close(out.float(), expected, rtol=_RTOL, atol=_ATOL)
