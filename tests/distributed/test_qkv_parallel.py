"""Tests for the fused, head-aware QKV projection under tensor parallelism.

A one-layer checkpoint is loaded per rank slice through ``tp_harness``;
each rank must hold exactly its heads, and the fused output must equal
the unsharded reference.

Usage:
    pytest tests/distributed/test_qkv_parallel.py
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rapid_llm.distributed import parallel_state as ps
from rapid_llm.models import weights
from rapid_llm.models.base import CausalLM
from rapid_llm.modules import QKVParallelLinear
from tests.registry import import_needs_cuda

import_needs_cuda()

HIDDEN = 32
HEAD_DIM = 8
NUM_HEADS = 8
NUM_KV_HEADS = 4  # GQA: two query heads per key/value head
ATTN = "layers.0.self_attn"


@pytest.fixture
def grid(monkeypatch):
    """Become one rank of a ``tp_size``-wide grid, with no process group behind it."""

    def enter(rank: int, tp_size: int) -> None:
        monkeypatch.setattr(ps, "_TP_RANK", rank)
        monkeypatch.setattr(ps, "_TP_WORLD_SIZE", tp_size)

    return enter


class _OneLayerDecoder(nn.Module):
    """Just enough of a decoder for the real translator and sharder to run on it.

    The bias is present because Qwen2 ships one, and a 1-D tensor is where an offset
    computed from the wrong axis would still produce the right shape.
    """

    def __init__(self) -> None:
        super().__init__()
        attn = nn.Module()
        # fp16, not the auto default: ``_numbered``'s labels below are only
        # exact in an 11-bit mantissa, and this mirrors what an fp16
        # checkpoint's config.dtype passes down the real load path.
        attn.qkv_proj = QKVParallelLinear(
            HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, bias=True, params_dtype=torch.float16
        )
        layer = nn.Module()
        layer.self_attn = attn
        self.layers = nn.ModuleList([layer])


def _numbered(rows: int, base: int, *, matrix: bool) -> torch.Tensor:
    """Rows labelled ``base + i``, so a misplaced block shows up in the value itself.

    Every label stays below 1024, where fp16 still represents integers exactly, which
    is what lets the reassembly assertion run at ``atol=0``.
    """
    labels = base + torch.arange(rows, dtype=torch.float32)
    return labels.unsqueeze(1).expand(rows, HIDDEN).contiguous() if matrix else labels


def _checkpoint() -> dict[str, torch.Tensor]:
    """An HF-shaped checkpoint: three separate projections, disjoint label ranges."""
    q, kv = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
    return {
        f"{ATTN}.q_proj.weight": _numbered(q, 0, matrix=True),
        f"{ATTN}.k_proj.weight": _numbered(kv, 100, matrix=True),
        f"{ATTN}.v_proj.weight": _numbered(kv, 200, matrix=True),
        f"{ATTN}.q_proj.bias": _numbered(q, 300, matrix=False),
        f"{ATTN}.k_proj.bias": _numbered(kv, 400, matrix=False),
        f"{ATTN}.v_proj.bias": _numbered(kv, 500, matrix=False),
    }


def _load_this_rank() -> QKVParallelLinear:
    """Build and fill the layer exactly as the loader does, for the ambient grid."""
    model = _OneLayerDecoder()
    weights.load_weights(
        model,
        _checkpoint().items(),
        lambda key: weights.translate_text_key(key, CausalLM.packed_modules_mapping),
    )
    return model.layers[0].self_attn.qkv_proj


def _blocks(proj: QKVParallelLinear, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The ``[q | k | v]`` blocks of a parameter, which stacks along dim 0."""
    q, kv = proj.q_size, proj.kv_size
    return tensor[:q], tensor[q : q + kv], tensor[q + kv :]


# --------------------------------------------------------------------------- #
# Layout arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_each_block_is_its_own_head_count_divided(grid, tp_size: int):
    """The definition of the layer: two divisions, not one."""
    grid(0, tp_size)
    proj = QKVParallelLinear(HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)

    assert (proj.num_heads, proj.num_kv_heads) == (NUM_HEADS // tp_size, NUM_KV_HEADS // tp_size)
    assert proj.q_size == proj.num_heads * HEAD_DIM
    assert proj.kv_size == proj.num_kv_heads * HEAD_DIM
    assert proj.weight.shape == (proj.q_size + 2 * proj.kv_size, HIDDEN)
    assert proj.full_output_size == (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM


def test_one_division_of_the_fused_width_would_leave_a_rank_without_key_value_heads(grid):
    """Why this is a class and not ``ColumnParallelLinear(hidden, q + 2 * kv)``.

    A single cut is blind to where q ends. With 8 query heads, 4 key/value heads and 4
    ranks it hands rank 0 four query heads and no key or value heads at all, while the
    per-block split gives every rank the 2/1/1 that attention actually needs. Both give
    the same local width, which is why nothing downstream would object.
    """
    grid(0, 4)
    proj = QKVParallelLinear(HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
    fused_width = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM

    assert proj.output_size == fused_width // 4  # indistinguishable by shape
    assert (proj.q_size, proj.kv_size) == (2 * HEAD_DIM, HEAD_DIM)
    assert fused_width // 4 == 4 * HEAD_DIM  # what one cut would give: query only


def test_the_head_count_that_runs_out_first_is_the_one_named(grid):
    """``num_kv_heads`` is the smaller count, so it caps the usable tensor-parallel size."""
    grid(0, 8)
    with pytest.raises(ValueError, match="key/value heads 4 does not divide across 8"):
        QKVParallelLinear(HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)


# --------------------------------------------------------------------------- #
# Loading: where each checkpoint tensor lands
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("leaf", ["weight", "bias"])
def test_the_ranks_blocks_reassemble_the_checkpoint(grid, tp_size: int, leaf: str):
    """Rank-order concatenation of each block must rebuild the tensor it came from.

    This is the whole contract in one assertion. It fails if a block is placed at the
    wrong offset, if the ranks' shares are interleaved the other way round (block-major
    instead of rank-major), or if a rank's slice of the incoming tensor is taken from
    the wrong position. Coverage — that no row of the fused parameter is left unwritten —
    is checked by :func:`~rapid_llm.models.weights.load_weights` itself.
    """
    checkpoint = _checkpoint()
    gathered: list[list[torch.Tensor]] = [[], [], []]

    for rank in range(tp_size):
        grid(rank, tp_size)
        proj = _load_this_rank()
        param = proj.weight if leaf == "weight" else proj.bias
        for block, collected in zip(_blocks(proj, param.data), gathered, strict=True):
            collected.append(block.float())

    for name, collected in zip("qkv", gathered, strict=True):
        expected = checkpoint[f"{ATTN}.{name}_proj.{leaf}"]
        torch.testing.assert_close(torch.cat(collected), expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [2, 4])
def test_a_rank_holds_the_heads_its_index_selects(grid, tp_size: int):
    """Rank ``r`` takes head range ``r * local`` — the same rule for q and for k/v.

    Reassembly alone would also pass if the ranks agreed on a permutation of the heads;
    this pins the assignment down, because RoPE and the KV cache index heads by position
    within the rank and would silently pair query head ``h`` with the wrong cache page.
    """
    for rank in range(tp_size):
        grid(rank, tp_size)
        proj = _load_this_rank()
        q, k, v = (block.float()[:, 0] for block in _blocks(proj, proj.weight.data))

        first_q_head = rank * proj.num_heads * HEAD_DIM
        first_kv_head = rank * proj.num_kv_heads * HEAD_DIM
        assert q[0] == first_q_head
        assert k[0] == 100 + first_kv_head
        assert v[0] == 200 + first_kv_head


def test_split_hands_out_views_in_qkv_order_with_heads_still_adjacent(grid):
    """The fusion is only free if nothing downstream copies the blocks out again.

    ``rope_emb_forward`` rotates q and k in place exactly when a block reshaped to
    ``[tokens, heads, head_dim]`` has stride ``(fused_width, head_dim, 1)``; the data
    pointers additionally pin the ``[q | k | v]`` order the KV-cache write assumes.
    """
    grid(1, 2)
    proj = QKVParallelLinear(HIDDEN, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM)
    tokens = 3
    qkv = torch.zeros(tokens, proj.output_size)
    q, k, v = proj.split(qkv)
    stride = qkv.element_size()

    assert q.data_ptr() == qkv.data_ptr()
    assert k.data_ptr() == q.data_ptr() + proj.q_size * stride
    assert v.data_ptr() == k.data_ptr() + proj.kv_size * stride
    for block, heads in ((q, proj.num_heads), (k, proj.num_kv_heads), (v, proj.num_kv_heads)):
        view = block.view(tokens, heads, proj.head_dim)
        assert view.stride() == (proj.output_size, proj.head_dim, 1)


# --------------------------------------------------------------------------- #
# What the fusion costs numerically
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("tokens", [1, 8, 512])
def test_the_fused_gemm_answers_what_three_separate_ones_answer(tokens: int):
    """Widening N may not change an output element by more than one bf16 rounding step.

    Each element's reduction is over ``hidden`` either way, so on the batched path the
    two agree bit for bit. At ``tokens == 1`` cuBLAS splits the reduction differently for
    the wider gemv and a single ulp can move — that is the entire numerical cost of the
    fusion, and it is why the ``tests/golden`` baseline was re-recorded when it landed.
    The geometry is Qwen2.5-0.5B's, the checkpoint that baseline is recorded from; the
    activation follows the weight's element type because cuBLAS takes no implicit
    promotion between the two 16-bit formats.
    """
    hidden, heads, kv_heads, head_dim = 896, 14, 2, 64
    torch.manual_seed(0)
    # bf16 explicitly: the assertion's tolerance is one bf16 ulp, so the test
    # names the precision instead of riding the auto default.
    proj = QKVParallelLinear(
        hidden, heads, kv_heads, head_dim, bias=True, params_dtype=torch.bfloat16
    ).cuda()
    proj.weight.data.normal_(0, 0.05)
    proj.bias.data.normal_(0, 0.1)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    separate = [
        F.linear(x, weight, bias)
        for weight, bias in zip(
            _blocks(proj, proj.weight.data), _blocks(proj, proj.bias.data), strict=True
        )
    ]
    for got, want in zip(proj.project(x), separate, strict=True):
        torch.testing.assert_close(got, want, rtol=torch.finfo(torch.bfloat16).eps, atol=0)
