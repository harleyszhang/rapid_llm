"""Decode-context-parallel KV layout and softmax reduction.

Positions are dealt round-robin: position ``p`` of every sequence is cached only
on DCP rank ``p % dcp``.  Each rank attends over its own share, producing one
normalized partial output plus its logsumexp (LSE), and this module combines
those sufficient statistics without gathering the KV cache itself.

The scheduler keeps its global ``block_size``-token blocks (so prefix hashing
and admission are unchanged); one block occupies ``block_size // dcp`` cache
rows on every rank.  A position a rank does not own maps to row 0 of the
reserved null block, which turns its K/V write into a discard without any
caller having to know which positions are local.
"""

from __future__ import annotations

import torch

from .parallel_state import (
    all_to_all,
    decode_context_parallel_all_gather,
    get_dcp_comm_backend,
    get_decode_context_parallel_group,
    get_decode_context_parallel_rank,
    get_decode_context_parallel_world_size,
    reduce_scatter,
)


def dcp_rows_per_block(block_size: int, dcp_size: int) -> int:
    """Cache rows one scheduler block occupies on each DCP rank.

    Raises:
        ValueError: ``dcp_size`` does not divide ``block_size`` -- the round-robin
            deal would give ranks unequal shares of a block.
    """
    if block_size % dcp_size:
        raise ValueError(
            f"decode_context_parallel_size={dcp_size} must divide the KV block size {block_size}"
        )
    return block_size // dcp_size


def dcp_block_rows(block_id: int, block_size: int, dcp_rank: int, dcp_size: int) -> list[int]:
    """Cache row of every position in one scheduler block, as seen by ``dcp_rank``.

    Owned offsets (``offset % dcp_size == dcp_rank``) are packed densely into the
    block's ``block_size // dcp_size`` rows; every other offset is the discard
    row 0. ``dcp_size == 1`` is the plain contiguous block.
    """
    per_rank = dcp_rows_per_block(block_size, dcp_size)
    base = block_id * per_rank
    return [
        base + offset // dcp_size if offset % dcp_size == dcp_rank else 0
        for offset in range(block_size)
    ]


def dcp_local_seq_len(seq_len, dcp_rank: int, dcp_size: int):
    """How many of the first ``seq_len`` positions ``dcp_rank`` owns.

    Works on ints and integer tensors alike (``seq_len >= 0``): it is
    ``ceil((seq_len - rank) / size)`` clamped at zero, written so the
    numerator never goes negative for ``seq_len >= 1``.
    """
    local = (seq_len - dcp_rank + dcp_size - 1) // dcp_size
    if isinstance(local, torch.Tensor):
        return local.clamp_min_(0)
    return max(local, 0)


def dcp_kv_head_range(
    num_heads: int, num_kv_heads: int, dcp_rank: int, dcp_size: int
) -> tuple[int, int]:
    """The local KV heads one rank's own query heads read during prefill.

    Under DCP the KV heads are sharded ``tp / dcp`` ways, so a rank holds the KV
    heads of its whole DCP group while its query heads are still its own
    ``1 / tp`` share. Prefill attends fresh K/V on the local query heads only, so
    it needs the KV sub-range those heads map to under GQA.

    Args:
        num_heads: This rank's query heads.
        num_kv_heads: This rank's (DCP-group-wide) KV heads.

    Raises:
        ValueError: The rank's query heads straddle KV heads in a way one GQA
            ratio cannot express.
    """
    group_heads = num_heads * dcp_size
    if group_heads % num_kv_heads:
        raise ValueError(
            f"{group_heads} DCP-group query heads do not share {num_kv_heads} KV heads evenly"
        )
    ratio = group_heads // num_kv_heads
    first = dcp_rank * num_heads
    start = first // ratio
    end = (first + num_heads - 1) // ratio + 1
    span = end - start
    if num_heads % span or (num_heads // span != ratio and span != 1):
        raise ValueError(
            f"DCP rank {dcp_rank}'s {num_heads} query heads do not map onto whole KV heads "
            f"(GQA ratio {ratio})"
        )
    return start, end


def merge_attention_partials(
    outputs: torch.Tensor, lses: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge normalized attention partials along their leading rank axis.

    Args:
        outputs: ``[ranks, batch, heads, head_dim]`` normalized local outputs.
        lses: ``[ranks, batch, heads]`` fp32-compatible local logsumexp values.

    Returns:
        The globally normalized output and LSE. If every rank has an empty
        history for a row/head, its output is zero and its LSE remains ``-inf``.
    """
    if outputs.dim() != 4 or lses.dim() != 3:
        raise ValueError(
            "DCP partials require outputs [ranks, batch, heads, dim] and "
            f"LSE [ranks, batch, heads], got {tuple(outputs.shape)} and {tuple(lses.shape)}"
        )
    if outputs.shape[:3] != lses.shape:
        raise ValueError(
            f"DCP output/LSE shapes do not align: {tuple(outputs.shape)} vs {tuple(lses.shape)}"
        )

    local_lses = lses.float()
    global_lse = torch.logsumexp(local_lses, dim=0)
    finite = torch.isfinite(global_lse).unsqueeze(0)
    weights = torch.where(
        finite,
        torch.exp(local_lses - global_lse.unsqueeze(0)),
        torch.zeros_like(local_lses),
    )
    merged = (outputs.float() * weights.unsqueeze(-1)).sum(dim=0)
    return merged.to(outputs.dtype), global_lse


def _validate_local_partials(
    local_output: torch.Tensor, local_lse: torch.Tensor, world_size: int
) -> int:
    if local_output.dim() != 3 or local_lse.dim() != 2:
        raise ValueError(
            "DCP local attention requires output [batch, heads, dim] and "
            f"LSE [batch, heads], got {tuple(local_output.shape)} and {tuple(local_lse.shape)}"
        )
    if local_output.shape[:2] != local_lse.shape:
        raise ValueError(
            f"DCP output/LSE shapes do not align: {tuple(local_output.shape)} vs "
            f"{tuple(local_lse.shape)}"
        )
    heads = local_output.shape[1]
    if heads % world_size:
        raise ValueError(
            f"DCP query heads {heads} must be divisible by decode_context_parallel_size={world_size}"
        )
    return heads // world_size


@torch.no_grad()
def merge_dcp_attention(
    local_output: torch.Tensor,
    local_lse: torch.Tensor,
    *,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine rank-local decode attention and return this rank's Q-head shard.

    ``local_output`` contains all Q heads in this DCP group evaluated against
    this rank's KV shard. ``ag_rs`` gathers only LSE, weights the local output,
    then reduce-scatters heads. ``a2a`` exchanges head chunks of output and LSE,
    then performs the stable merge locally.
    """
    world_size = get_decode_context_parallel_world_size()
    if world_size <= 1:
        return local_output, local_lse

    heads_per_rank = _validate_local_partials(local_output, local_lse, world_size)
    backend = get_dcp_comm_backend() if backend is None else backend
    group = get_decode_context_parallel_group()

    if backend == "ag_rs":
        gathered_lse = decode_context_parallel_all_gather(local_lse.unsqueeze(0), dim=0)
        global_lse = torch.logsumexp(gathered_lse.float(), dim=0)
        finite = torch.isfinite(global_lse)
        weights = torch.where(
            finite,
            torch.exp(local_lse.float() - global_lse),
            torch.zeros_like(global_lse),
        )
        weighted_output = local_output.float() * weights.unsqueeze(-1)
        output = reduce_scatter(weighted_output, dim=1, group=group).to(local_output.dtype)
        start = get_decode_context_parallel_rank() * heads_per_rank
        return output, global_lse.narrow(1, start, heads_per_rank)

    if backend == "a2a":
        exchanged_output = all_to_all(local_output.movedim(1, 0), group=group)
        exchanged_lse = all_to_all(local_lse.movedim(1, 0), group=group)
        batch, head_dim = local_output.shape[0], local_output.shape[2]
        output_partials = exchanged_output.reshape(
            world_size, heads_per_rank, batch, head_dim
        ).permute(0, 2, 1, 3)
        lse_partials = exchanged_lse.reshape(world_size, heads_per_rank, batch).permute(0, 2, 1)
        return merge_attention_partials(output_partials, lse_partials)

    raise ValueError(f"unknown DCP communication backend {backend!r}; choose from ag_rs, a2a")
