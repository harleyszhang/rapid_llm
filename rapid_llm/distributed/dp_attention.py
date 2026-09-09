"""Data-parallel attention: attention per DP rank, experts pooled across them.

The parallelism MoE inference wants — the one sglang and vLLM both ship for
DeepSeek-V3. The two stages of a decoder layer scale oppositely:

* **Attention** is per request: with MLA the KV cache is one compressed vector
  per layer, so TP duplicates the cache and every rank pays the whole read;
  replicate the small weights instead — ``dp`` x KV capacity, ``dp`` x batch,
  no collective at all.
* **The experts** are per token, hundreds of them: they want the widest token
  pool, so the routed stage runs over the *union* of every DP rank's tokens,
  experts split whole-expert across the ``dp x tp`` grid.

The seam between the two: :func:`dp_attention_region` is the once-per-step
handshake (peer token counts); :func:`dp_gather` / :func:`dp_scatter` pool the
tokens for the routed stage and hand back each rank's rows. The MoE block calls
the pair — where vLLM puts it too, inside ``FusedMoE``'s prepare/finalize — so
nothing between the two stages has to know the batch is split.

Usage:
    with dp_attention_region(num_tokens=input_ids.numel()):
        logits = model(input_ids, position_ids, atten_info)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from .parallel_state import (
    data_parallel_all_gather,
    dp_attention_enabled,
    get_data_parallel_cpu_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
)

__all__ = [
    "DPMetadata",
    "coordinate_tokens_across_dp",
    "current_dp_metadata",
    "dp_attention_region",
    "dp_gather",
    "dp_scatter",
]


@dataclass(frozen=True)
class DPMetadata:
    """How many tokens each DP rank brought to this step.

    vLLM's ``DPMetadata`` (``vllm/forward_context.py``) under a different
    spelling: the same ``num_tokens_across_dp`` vector, agreed once per forward
    and read by every MoE layer.

    Args:
        num_tokens_across_dp: Rows per DP rank, in rank order.
        rank: This process's rank within the DP group.
    """

    num_tokens_across_dp: tuple[int, ...]
    rank: int

    @property
    def world_size(self) -> int:
        """Ranks sharing the step."""
        return len(self.num_tokens_across_dp)

    @property
    def local_tokens(self) -> int:
        """Rows this rank brought -- what :func:`dp_scatter` hands back."""
        return self.num_tokens_across_dp[self.rank]

    @property
    def padded_tokens(self) -> int:
        """Rows every rank *contributes*, the largest local count.

        The pooled batch is a rectangle rather than a ragged concatenation. A
        ragged one would need a variable-split ``all_gatherv`` whose splits are
        host values, which costs a device-to-host sync per layer and cannot be
        captured into a CUDA graph; padding to the maximum makes every rank's
        contribution the same shape, so one plain all-gather does it and the
        shape is a function of the step's buckets, not its data. It is also what
        vLLM pads to, for the same reason.

        The padded rows are zeros. They route (to whichever experts a zero
        hidden state scores highest), run through those experts, and are dropped
        by :func:`dp_scatter` -- so they cost expert FLOPs and change nothing.
        """
        return max(self.num_tokens_across_dp)

    @property
    def total_tokens(self) -> int:
        """Rows in the pooled batch the routed stage runs over."""
        return self.padded_tokens * self.world_size

    @property
    def local_offset(self) -> int:
        """Where this rank's rows start in the pooled batch."""
        return self.rank * self.padded_tokens


_metadata: ContextVar[DPMetadata | None] = ContextVar("rapid_llm_dp_metadata", default=None)


def current_dp_metadata() -> DPMetadata | None:
    """The step geometry this call site runs inside, or ``None``.

    ``None`` means "no pooling to do": either DP-attention is off, or the DP axis
    holds one rank. Both leave the MoE block on its existing path.
    """
    return _metadata.get()


@contextmanager
def dp_attention_region(num_tokens: int) -> Iterator[DPMetadata | None]:
    """Agree the step's token geometry with the DP peers, once per forward.

    The handshake is a collective, so it is not optional and it is not
    per-layer: every rank in the DP group must reach it exactly once per step,
    including a rank whose batch is empty (which contributes 0 and still has to
    show up, or its peers wait in the gather for a rank that never arrives).
    That is the lockstep vLLM's ``coordinate_batch_across_dp`` enforces, and the
    reason an idle DP rank runs a dummy step rather than skipping one.

    Args:
        num_tokens: Rows this rank's batch contributes -- ``batch * seq_len``,
            the same count the MoE block will see after its flatten.

    Yields:
        The step's :class:`DPMetadata`, or ``None`` when there is nothing to
        pool, in which case no collective was issued and none is expected.

    Raises:
        RuntimeError: If sequence parallelism is also on. Both re-partition the
            token axis before the MoE sees it, so the row count the handshake
            agreed would no longer be the row count the block is holding; the
            combination needs the SP shard to be folded into this geometry and
            is not supported yet.
    """
    if not dp_attention_enabled():
        yield None
        return
    from .sequence_parallel import sequence_parallel_enabled

    if sequence_parallel_enabled():
        raise RuntimeError(
            "DP-attention and sequence parallelism both re-partition the token axis "
            "before the MoE stage; run with one of them"
        )
    metadata = coordinate_tokens_across_dp(num_tokens)
    token = _metadata.set(metadata)
    try:
        yield metadata
    finally:
        _metadata.reset(token)


def coordinate_tokens_across_dp(num_tokens: int) -> DPMetadata:
    """Exchange local token counts across the DP group.

    One int per rank over the gloo twin of the DP group: the counts are host
    values (they size the gather buffers), so sending them over NCCL would only
    add a device round trip to a decision the host has to read anyway.
    """
    counts = torch.tensor([num_tokens], dtype=torch.int64)
    gathered = data_parallel_all_gather(counts, dim=0, group=get_data_parallel_cpu_group())
    across = tuple(int(n) for n in gathered.tolist())
    if len(across) != get_data_parallel_world_size():
        # A DP group that does not span the replicas would silently pool a
        # subset of the batch, which reads as a numerical bug much later.
        raise RuntimeError(
            f"DP handshake returned {len(across)} counts for "
            f"{get_data_parallel_world_size()} replicas"
        )
    return DPMetadata(num_tokens_across_dp=across, rank=get_data_parallel_rank())


def dp_gather(x: torch.Tensor, metadata: DPMetadata) -> torch.Tensor:
    """Pool ``[local_tokens, hidden]`` into the step's ``[total_tokens, hidden]``.

    Pads to :attr:`DPMetadata.padded_tokens` first, so every rank contributes
    the same shape and one all-gather suffices. Inverse of :func:`dp_scatter`.

    Raises:
        ValueError: If ``x`` does not hold the row count the handshake agreed --
            something between :func:`dp_attention_region` and here re-partitioned
            the token axis.
    """
    if x.shape[0] != metadata.local_tokens:
        raise ValueError(
            f"DP gather expected {metadata.local_tokens} rows (the count this rank "
            f"declared for the step) but holds {x.shape[0]}"
        )
    padded = metadata.padded_tokens
    if x.shape[0] != padded:
        x = torch.nn.functional.pad(x, (0, 0, 0, padded - x.shape[0]))
    return data_parallel_all_gather(x, dim=0)


def dp_scatter(x: torch.Tensor, metadata: DPMetadata) -> torch.Tensor:
    """Take this rank's rows out of a pooled ``[total_tokens, hidden]`` result.

    A slice, not a collective: the routed combine already landed every pooled
    token's full expert sum on every rank, so each rank only has to find its
    own. Inverse of :func:`dp_gather`.
    """
    return x[metadata.local_offset : metadata.local_offset + metadata.local_tokens]
