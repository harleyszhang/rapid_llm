"""DeepSeek-V3-style DP attention with DPxTP expert parallelism:

- Attention is replicated across DP ranks: each rank owns its requests and
  KV cache, avoiding the redundant KV-cache reads that TP would impose.
- Expert layers pool tokens from all DP ranks. Once per step, a handshake
  exchanges local token counts; the routed stage then runs over the whole
  pool, and the combine hands each rank its own rows back.

Two pooling contracts share that geometry:

* **AgRs** (the default; vLLM's ``allgather_reducescatter`` backend): every
  rank routes its own rows first and :func:`dp_dispatch` carries the hidden
  states, weights and ids together into a ragged pool -- ``sum(counts)``
  rows, no padding. :func:`dp_combine` reduce-scatters the pool's expert
  output back to each rank's chunk. One exchange in, one exchange out, and
  no pad rows waste expert FLOPs.
* **a2a** (kept for A/B): :func:`dp_gather` pads every rank to the largest
  count and builds a rectangle, the MoE block routes the pooled rows, and an
  all-to-all dispatcher exchanges them by expert; :func:`dp_scatter` slices
  the result back out.

The transport has no ragged collective to call -- torch binds no
``all_gatherv`` and rapid_llm ships no pynccl -- so both v-collectives run
equal-length on the wire and compact on the receiver (gather) or zero-pad on
the sender (reduce); see :mod:`rapid_llm.distributed.parallel_state`. That is
the same degradation vLLM takes under CUDA graph capture, where a host-side
ragged split is illegal.

The handshake runs over the CPU Gloo group, so the counts stay host values
and the shapes they fix derive from the step's buckets, not its data -- the
compaction and scatter plans follow from them, which is what keeps the
forward capturable.

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
    all_gatherv,
    data_parallel_all_gather,
    dp_attention_enabled,
    get_data_parallel_cpu_group,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_world_size,
    reduce_scatterv,
    tensor_model_parallel_all_reduce,
)

__all__ = [
    "DPMetadata",
    "coordinate_tokens_across_dp",
    "current_dp_metadata",
    "dp_attention_region",
    "dp_combine",
    "dp_dispatch",
    "dp_gather",
    "dp_scatter",
]


@dataclass(frozen=True)
class DPMetadata:
    """How many tokens each DP rank brought to this step.

    vLLM's ``DPMetadata`` (``vllm/forward_context.py``) under a different
    spelling: the same ``num_tokens_across_dp`` vector, agreed once per forward
    and read by every MoE layer. Both pooling contracts derive their geometry
    from it -- the ragged pool is the concatenation of these counts, the a2a
    rectangle is every rank padded to the largest of them.

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
        """Rows this rank brought -- what :func:`dp_scatter` and
        :func:`dp_combine` hand back."""
        return self.num_tokens_across_dp[self.rank]

    @property
    def uniform(self) -> bool:
        """Whether every rank brought the same count.

        The AgRs fast case: equal shards let both v-collectives take their
        plain, plan-free route (see :mod:`rapid_llm.distributed.parallel_state`).
        """
        return len(set(self.num_tokens_across_dp)) == 1

    @property
    def padded_tokens(self) -> int:
        """Rows every rank contributes to the *a2a* pool, the largest count.

        Only the a2a pooling (:func:`dp_gather`) pads to the maximum: it is one
        plain all-gather, which takes equal contributions. The padded rows are
        zeros; they route (to whichever experts a zero hidden state scores
        highest), run through those experts, and are dropped by
        :func:`dp_scatter` -- so they cost expert FLOPs and change nothing.
        That waste is what the AgRs pool (:func:`dp_dispatch`) does not pay:
        it gathers ragged. The maximum itself is a function of the step's
        buckets, not its data, so the shape stays CUDA-graph friendly.
        """
        return max(self.num_tokens_across_dp)

    @property
    def total_tokens(self) -> int:
        """Rows in the ragged pool the routed stage runs over."""
        return sum(self.num_tokens_across_dp)

    @property
    def local_offset(self) -> int:
        """Where this rank's rows start in the *ragged* pool.

        The cumulative count of every earlier rank -- the ragged
        concatenation's layout, which is what :func:`dp_dispatch` and
        :func:`dp_combine` run on.
        """
        return sum(self.num_tokens_across_dp[: self.rank])

    @property
    def padded_offset(self) -> int:
        """Where this rank's rows start in the *a2a rectangle*.

        ``rank * padded_tokens`` -- the rectangle's layout, which
        :func:`dp_scatter` slices with. The two offsets differ whenever the
        counts are not uniform.
        """
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
    """Pool ``[local_tokens, hidden]`` into the step's padded a2a rectangle.

    Pads to :attr:`DPMetadata.padded_tokens` first, so every rank contributes
    the same shape and one plain all-gather suffices -- the a2a path's pooling,
    kept alongside :func:`dp_dispatch` for A/B. Inverse of :func:`dp_scatter`.

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
    """Take this rank's rows out of the a2a rectangle's routed result.

    A slice, not a collective: the routed combine already landed every pooled
    token's full expert sum on every rank, so each rank only has to find its
    own. Inverse of :func:`dp_gather`.
    """
    return x[metadata.padded_offset : metadata.padded_offset + metadata.local_tokens]


def dp_dispatch(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    metadata: DPMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool the hidden states *and* this rank's routing into the ragged pool.

    vLLM's AgRs dispatch (``AgRsAll2AllManager.dispatch``): the route runs
    before the exchange, on this rank's own rows, and its results travel with
    the hidden states -- one ragged all-gather and every rank holds the whole
    pool together with the whole routing. The pool is ragged:
    ``sum(num_tokens_across_dp)`` rows where the a2a rectangle :func:`dp_gather`
    builds would hold ``padded * world``. The difference is the pad rows --
    zeros that route, run through their experts, and are thrown away again.

    Args:
        x: This rank's hidden rows, ``[local_tokens, hidden]``.
        topk_weights: This rank's routing weights, ``[local_tokens, top_k]``.
        topk_ids: This rank's routing ids, ``[local_tokens, top_k]``, in the
            global id space; the caller rebases them to the local expert window
            (``-1`` outside) after the gather.
        metadata: The step geometry.

    Returns:
        ``(pool_x, pool_weights, pool_ids)``, one concatenation each, in group
        rank order.

    Raises:
        ValueError: If ``x`` does not hold the row count the handshake agreed --
            something between :func:`dp_attention_region` and here re-partitioned
            the token axis.
    """
    if x.shape[0] != metadata.local_tokens:
        raise ValueError(
            f"DP dispatch expected {metadata.local_tokens} rows (the count this rank "
            f"declared for the step) but holds {x.shape[0]}"
        )
    pool_x, pool_weights, pool_ids = all_gatherv(
        [x, topk_weights, topk_ids],
        dim=0,
        sizes=metadata.num_tokens_across_dp,
        group=get_data_parallel_group(),
    )
    return pool_x, pool_weights, pool_ids


def dp_combine(pooled_out: torch.Tensor, metadata: DPMetadata) -> torch.Tensor:
    """Sum the ragged pool's expert output and keep this rank's rows.

    vLLM's AgRs combine (``AgRsAll2AllManager.combine``): one reduce-scatter
    over the pool returns each rank its own chunk, already summed. Which
    peers' partials meet in that sum is the topology's business:

    * ``tp == 1``: the DP group spans the replicas, so the reduce covers every
      rank holding the pool -- the complete expert sum.
    * ``tp > 1``: a DP group is one TP lane, and a lane's ranks own a stride
      of the expert windows, so the reduce covers only that stride's experts.
      The trailing TP all-reduce sums the replica's lanes -- whose windows
      partition the expert set -- and the full sum is back. vLLM does not
      need that step because its MoE runs sequence-parallel under TP>1: the
      ranks meeting in its reduce own disjoint experts *and* disjoint chunks.
      Here the lanes hold identical tokens, so the chunks overlap and the
      missing experts are added back explicitly instead.

    Args:
        pooled_out: The pool's routed output, ``[total_tokens, hidden]``.
        metadata: The step geometry.

    Returns:
        This rank's ``[local_tokens, hidden]`` expert sum.
    """
    out = reduce_scatterv(
        pooled_out,
        sizes=metadata.num_tokens_across_dp,
        dim=0,
        group=get_data_parallel_group(),
    )
    if get_tensor_model_parallel_world_size() > 1:
        out = tensor_model_parallel_all_reduce(out)
    return out
