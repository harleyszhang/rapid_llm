"""The DP x TP rank grid: one global rank space both parallel axes agree on.

``init_parallel`` splits the world into contiguous TP groups — one per DP
replica — and builds the TP process group; asking for DP-attention additionally
builds one group per TP lane (the DP axis) and can widen the EP group to the
whole grid. The ``get_*`` accessors then answer rank queries from module state
anywhere in the codebase. The collective
names follow vLLM's ``vllm.distributed.parallel_state`` spelling
(``tensor_model_parallel_all_reduce`` and friends) so both codebases read the
same at the call site.

Usage:
    init_parallel(global_rank, tp_size, dp_size, master_port)
    assert get_tensor_model_parallel_rank() < get_tensor_model_parallel_world_size()
"""

from __future__ import annotations

import os
import pickle
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from ..tools.observability import Collective, CollectiveStats
from ..utils.logger import get_logger

_log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Module-level state
# --------------------------------------------------------------------------- #
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1
_TP_GROUP: dist.ProcessGroup | None = None
_TP_CPU_GROUP: dist.ProcessGroup | None = None
_DP_RANK: int = 0
_DP_WORLD_SIZE: int = 1
_DP_GROUP: dist.ProcessGroup | None = None
_DP_CPU_GROUP: dist.ProcessGroup | None = None
_DP_ATTENTION: bool = False
_EP_ENABLED: bool = False
_EP_GROUP: dist.ProcessGroup | None = None
_EP_RANK: int = 0
_EP_WORLD_SIZE: int = 1


def grid_coordinates(global_rank: int, tp_size: int, dp_size: int) -> tuple[int, int]:
    """Decompose a global rank into ``(dp_rank, tp_rank)``.

    The inverse of the layout ``global_rank = dp_rank * tp_size + tp_rank``. Kept
    separate from :func:`init_parallel` because it is the whole of the grid's logic and
    none of its side effects: a caller — or a test — can ask where rank 5 of a 3x2 grid
    sits without a CUDA device or an NCCL rendezvous.

    Args:
        global_rank: This process's rank in ``[0, tp_size * dp_size)``.
        tp_size: Ranks per replica.
        dp_size: Number of replicas.

    Returns:
        ``(dp_rank, tp_rank)`` — which replica, and which rank inside it.

    Raises:
        ValueError: If the sizes are not positive, or ``global_rank`` falls outside
            the grid.
    """
    if tp_size < 1 or dp_size < 1:
        raise ValueError(f"tp_size and dp_size must be >= 1, got {tp_size} and {dp_size}")
    world_size = tp_size * dp_size
    if not 0 <= global_rank < world_size:
        raise ValueError(
            f"global_rank {global_rank} is outside a {dp_size}x{tp_size} grid of {world_size} ranks"
        )
    return global_rank // tp_size, global_rank % tp_size


def init_parallel(
    global_rank: int = 0,
    tp_size: int = 1,
    dp_size: int = 1,
    master_port: int = 29500,
    backend: str | None = None,
    enable_expert_parallel: bool = False,
    enable_dp_attention: bool = False,
) -> None:
    """Place this process in the ``dp_size x tp_size`` rank grid.

    The coordinates come from :func:`grid_coordinates`, so a caller only has to hand
    out consecutive global ranks. Which groups get built depends on what the DP axis
    *means*:

    * **Replica DP** (the default): each replica is a whole, independent model and
      the replicas share no tensors, so there is nothing to rendezvous about across
      the DP axis and NCCL is never touched — with ``tp_size == 1`` this call does
      not touch ``torch.distributed`` at all.
    * **DP-attention** (``enable_dp_attention``, vLLM's ``data_parallel_size`` with
      ``enable_expert_parallel``): the replicas hold *one* model between them. Each
      DP rank keeps a full copy of the attention weights and its own KV cache, so
      attention stays local and needs no collective; the MoE stage then pools every
      rank's tokens, which means the DP axis needs a group of its own *and* the EP
      group has to span the whole ``dp_size x tp_size`` grid rather than one
      replica's TP group. That is vLLM's topology, where the TP group is
      ``all_ranks.view(-1, tp)`` and the EP group is the DP/TP transpose flattened.

    Note that once any group is built this call *blocks* until all
    ``tp_size * dp_size`` ranks have joined: that is what ``init_process_group`` means.

    Args:
        global_rank: This process's rank in ``[0, tp_size * dp_size)``.
        tp_size: Ranks per replica (weights split across them).
        dp_size: Number of replicas (requests split across them).
        master_port: TCP port for the TP rendezvous (rank 0 listens).
        backend: ``torch.distributed`` backend; ``None`` picks ``nccl`` on a machine
            with GPUs and ``gloo`` without, which is what lets the sharded layers and
            the vocabulary-parallel sampler be verified on CPU.
        enable_expert_parallel: Whether MoE experts split whole-expert across the EP
            group instead of being sliced along their intermediate dimension.
        enable_dp_attention: Whether the DP axis is a shard of one model (see above)
            rather than a set of independent replicas. Ignored when ``dp_size == 1``,
            where the two are the same topology.

    Raises:
        ValueError: If the sizes are not positive, or ``global_rank`` falls outside
            the grid.
    """
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP, _TP_CPU_GROUP, _DP_RANK, _DP_WORLD_SIZE
    global _DP_GROUP, _DP_CPU_GROUP, _DP_ATTENTION
    global _EP_ENABLED, _EP_GROUP, _EP_RANK, _EP_WORLD_SIZE

    _DP_RANK, _TP_RANK = grid_coordinates(global_rank, tp_size, dp_size)
    _DP_WORLD_SIZE = dp_size
    _TP_WORLD_SIZE = tp_size
    _EP_ENABLED = enable_expert_parallel
    _DP_ATTENTION = enable_dp_attention and dp_size > 1
    world_size = tp_size * dp_size
    # How wide the expert split is: over one replica's TP group by default, over
    # the whole grid under DP-attention — which is the point of pairing the two.
    # The flag alone never implies a distributed path; a world of one answers 1.
    if not _EP_ENABLED:
        _EP_RANK, _EP_WORLD_SIZE = 0, 1
    elif _DP_ATTENTION:
        _EP_RANK, _EP_WORLD_SIZE = global_rank, world_size
    else:
        _EP_RANK, _EP_WORLD_SIZE = _TP_RANK, _TP_WORLD_SIZE
    if tp_size <= 1 and not _DP_ATTENTION:
        _EP_RANK, _EP_WORLD_SIZE = 0, 1
        return

    backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "1")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=global_rank, world_size=world_size)

    # ``new_group`` is itself collective: every rank must build every group, in
    # the same order, even the ones it is not a member of. Hence the loops.
    if tp_size > 1:
        for replica in range(dp_size):
            members = list(range(replica * tp_size, (replica + 1) * tp_size))
            group = dist.new_group(members, backend=backend)
            # A second, CPU-backed group over the same ranks carries the *control*
            # plane: :func:`tensor_model_parallel_broadcast_object_list` ships pickled
            # plans, and nccl can only move device memory, so it would have to stage
            # every plan through the GPU. gloo sends the bytes straight from host memory.
            cpu_group = group if backend == "gloo" else dist.new_group(members, backend="gloo")
            if replica == _DP_RANK:
                _TP_GROUP = group
                _TP_CPU_GROUP = cpu_group

    if _DP_ATTENTION:
        # The DP axis is the *stride* through the grid: rank ``lane`` of every
        # replica forms one group, which is vLLM's transpose of the (DP, TP)
        # plane. Peers in a DP group hold identical shards of the weights and
        # differ only in which requests they are serving.
        for lane in range(tp_size):
            members = list(range(lane, world_size, tp_size))
            group = dist.new_group(members, backend=backend)
            # The lockstep handshake (how many tokens each rank brings to this
            # step) is host-side control, so it wants gloo for the same reason
            # the plan broadcast does.
            cpu_group = group if backend == "gloo" else dist.new_group(members, backend="gloo")
            if lane == _TP_RANK:
                _DP_GROUP = group
                _DP_CPU_GROUP = cpu_group

    if _EP_ENABLED:
        # Without DP-attention EP is a mode over the TP grid, not a new
        # rendezvous: the same ranks that all-reduce attention partials exchange
        # MoE tokens, so the group object is literally shared. With it, the
        # experts spread over every rank in the grid and need their own group.
        _EP_GROUP = (
            dist.new_group(list(range(world_size)), backend=backend) if _DP_ATTENTION else _TP_GROUP
        )

    _log.info(
        "parallel state: global rank %d/%d (dp %d/%d, tp %d/%d)%s%s",
        global_rank,
        world_size,
        _DP_RANK,
        dp_size,
        _TP_RANK,
        tp_size,
        ", dp-attention" if _DP_ATTENTION else "",
        f", ep {_EP_RANK}/{_EP_WORLD_SIZE}" if _EP_ENABLED else "",
    )


def init_tensor_parallel(
    rank: int = 0,
    world_size: int = 1,
    master_port: int = 29500,
    backend: str | None = None,
    enable_expert_parallel: bool = False,
) -> None:
    """Initialise a TP-only world: one replica whose ranks are ``[0, world_size)``.

    Kept as the TP entry point (the CLI, benchmarks and tests all call it); it is
    :func:`init_parallel` with ``dp_size=1``.

    Args:
        rank: This process's rank within the TP group.
        world_size: Number of TP ranks.
        master_port: TCP port for the rendezvous (rank 0 listens).
        backend: See :func:`init_parallel`.
        enable_expert_parallel: See :func:`init_parallel`.
    """
    init_parallel(
        global_rank=rank,
        tp_size=world_size,
        dp_size=1,
        master_port=master_port,
        backend=backend,
        enable_expert_parallel=enable_expert_parallel,
    )


def destroy_parallel() -> None:
    """Tear down the process group and reset the grid to a world of one."""
    if _TP_GROUP is not None or _DP_GROUP is not None:
        dist.destroy_process_group()
    _reset_grid()


def abandon_parallel() -> None:
    """Reset the grid to a world of one without waiting on the process groups.

    For a teardown whose NCCL abort refuses to complete: a communicator whose
    collectives were captured into a CUDA graph can park ``destroy_process_group``
    in a futex no rank can wake — a PyTorch/NCCL interaction, not something this
    module can unstick from inside the call.
    """
    _reset_grid()


def _reset_grid() -> None:
    """Return every module-level coordinate to its single-process default."""
    global _TP_RANK, _TP_WORLD_SIZE, _TP_GROUP, _TP_CPU_GROUP, _DP_RANK, _DP_WORLD_SIZE
    global _DP_GROUP, _DP_CPU_GROUP, _DP_ATTENTION
    global _EP_ENABLED, _EP_GROUP, _EP_RANK, _EP_WORLD_SIZE
    _TP_RANK = 0
    _TP_WORLD_SIZE = 1
    _TP_GROUP = None
    _TP_CPU_GROUP = None
    _DP_RANK = 0
    _DP_WORLD_SIZE = 1
    _DP_GROUP = None
    _DP_CPU_GROUP = None
    _DP_ATTENTION = False
    _EP_ENABLED = False
    _EP_GROUP = None
    _EP_RANK = 0
    _EP_WORLD_SIZE = 1


def destroy_tensor_parallel() -> None:
    """Alias of :func:`destroy_parallel`, kept for the existing TP call sites."""
    destroy_parallel()


# --------------------------------------------------------------------------- #
# Accessors
# --------------------------------------------------------------------------- #
def get_tensor_model_parallel_rank() -> int:
    """This process's rank within its replica's TP group (0 when TP is disabled)."""
    return _TP_RANK


def get_tensor_model_parallel_world_size() -> int:
    """Number of TP ranks per replica (1 when TP is disabled)."""
    return _TP_WORLD_SIZE


def get_data_parallel_rank() -> int:
    """Which replica this process belongs to (0 when DP is disabled)."""
    return _DP_RANK


def get_data_parallel_world_size() -> int:
    """Number of model replicas (1 when DP is disabled)."""
    return _DP_WORLD_SIZE


def get_world_size() -> int:
    """Total ranks across the grid, ``dp_size * tp_size``."""
    return _DP_WORLD_SIZE * _TP_WORLD_SIZE


def get_tensor_model_parallel_group() -> dist.ProcessGroup | None:
    """The replica's TP process group, or ``None`` when TP is disabled.

    Callers that issue their own collectives on a side stream — the overlap
    package's deferred all-reduce, for instance — need the group object rather
    than the wrapped helpers, because they post the reduction themselves and
    fence it with their own events.
    """
    return _TP_GROUP


def expert_parallel_enabled() -> bool:
    """Whether MoE experts split whole-expert across the EP group.

    The flag alone does not imply a distributed EP path: a world of one (or a
    TP-disabled process) answers ``True`` here but :func:`get_ep_world_size`
    stays 1, and callers treat that as the no-op it is.
    """
    return _EP_ENABLED


def dp_attention_enabled() -> bool:
    """Whether the DP axis shards one model rather than holding whole replicas.

    ``True`` only when ``init_parallel`` was asked for DP-attention *and* there is
    more than one replica — with ``dp_size == 1`` the two topologies coincide, so
    every DP-attention branch in the codebase can stay off.
    """
    return _DP_ATTENTION


def get_data_parallel_group() -> dist.ProcessGroup | None:
    """The DP process group under DP-attention, else ``None``.

    Its members are the ranks holding the *same* weight shard for different
    requests — rank ``tp_rank`` of every replica. ``None`` for replica-level DP,
    where by construction there is nothing to say across the axis.
    """
    return _DP_GROUP


def get_data_parallel_cpu_group() -> dist.ProcessGroup | None:
    """The gloo twin of :func:`get_data_parallel_group`, for the token handshake."""
    return _DP_CPU_GROUP


def get_ep_group() -> dist.ProcessGroup | None:
    """The EP process group, or ``None`` when EP is off.

    One replica's TP group for plain EP; the whole ``dp_size x tp_size`` grid
    under DP-attention, which is what lets ``dp2 x tp2`` host a 4-way expert
    split instead of two independent 2-way ones.
    """
    return _EP_GROUP if _EP_ENABLED else None


def get_ep_rank() -> int:
    """This process's rank within the EP group (0 when EP is off)."""
    return _EP_RANK


def get_ep_world_size() -> int:
    """Number of ranks experts split across (1 when EP is off)."""
    return _EP_WORLD_SIZE


def divide(a: int, b: int, what: str = "") -> int:
    """``a // b`` with a clear error when it does not divide evenly.

    Tensor-parallel sharding divides dimensions across ranks; a non-divisible
    width means the requested TP size is too large for this model.
    """
    if a % b != 0:
        raise ValueError(f"{what or 'value'} {a} does not divide across {b} tensor-parallel ranks")
    return a // b


# --------------------------------------------------------------------------- #
# Collectives
# --------------------------------------------------------------------------- #
def _payload(tensor: torch.Tensor) -> int:
    """Bytes one rank contributes to a collective over ``tensor``."""
    return tensor.numel() * tensor.element_size()


def _resolve(group: dist.ProcessGroup | None) -> dist.ProcessGroup | None:
    """``None`` means the module-state TP group; an explicit group is used as given."""
    return _TP_GROUP if group is None else group


def _world_of_one(group: dist.ProcessGroup | None) -> bool:
    """Whether a collective over ``group`` has no peer to talk to."""
    return _TP_WORLD_SIZE <= 1 if group is None else dist.get_world_size(group) <= 1


def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """Sum ``tensor`` across all TP ranks. No-op when ``world_size == 1``.

    For TP=2 with NCCL backend and small payloads (≤ 64 KiB), the call routes
    through :func:`p2p_all_reduce` — a pair of send/recv that bypasses NCCL's
    ring setup and lands in 5–8 µs instead of 15–25 µs on PCIe interconnects.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    if (
        _TP_WORLD_SIZE == 2
        and _TP_GROUP is not None
        and dist.get_backend(_TP_GROUP) == "nccl"
        and tensor.numel() * tensor.element_size() <= 65536
    ):
        # Do not fall back after a P2P error: the peer may already have posted
        # its half, so entering a different collective can deadlock.
        return p2p_all_reduce(tensor)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
    CollectiveStats.record(Collective.ALL_REDUCE, _payload(tensor))
    return tensor


def tensor_model_parallel_all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """Element-wise maximum of ``tensor`` across this replica's TP ranks, in place.

    The other half of a vocabulary-parallel ``log_softmax``: each rank holds a slice of
    the vocabulary, so the row maximum that keeps ``exp`` from overflowing has to be the
    maximum over *all* slices. One float per row crosses the wire, not one per token id.

    No-op when ``tp_world_size == 1``.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=_TP_GROUP)
    CollectiveStats.record(Collective.ALL_REDUCE_MAX, _payload(tensor))
    return tensor


def tensor_model_parallel_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Concatenate every TP rank's ``tensor`` along ``dim``, rank order.

    Used for the small candidate tensors of vocabulary-parallel sampling — the ``k``
    best ``(logit, id)`` pairs per rank — where the transfer is ``O(k * tp)`` and
    independent of the vocabulary size. Every rank must pass the same shape.

    Returns ``tensor`` itself when ``tp_world_size == 1``.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    tensor = tensor.contiguous()
    parts = [torch.empty_like(tensor) for _ in range(_TP_WORLD_SIZE)]
    dist.all_gather(parts, tensor, group=_TP_GROUP)
    CollectiveStats.record(Collective.ALL_GATHER, _payload(tensor))
    return torch.cat(parts, dim=dim)


def data_parallel_all_gather(
    tensor: torch.Tensor, dim: int = 0, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Concatenate every DP peer's ``tensor`` along ``dim``, rank order.

    The DP-attention pooling primitive. Attention ran on this rank's slice of the
    batch, so the MoE stage only holds ``1/dp`` of the step's tokens; gathering
    them is what lets one expert exchange carry the whole global batch, and it is
    also the only reason a DP group exists. Every rank must pass the same shape,
    which is what the lockstep padding in
    :mod:`rapid_llm.distributed.dp_attention` is for.

    Args:
        tensor: This rank's contribution.
        dim: Axis to concatenate along (0 — the token axis — for hidden states).
        group: Defaults to the DP group; pass the CPU twin for control traffic.

    Returns:
        ``tensor`` itself when the DP axis holds one rank.
    """
    group = _DP_GROUP if group is None else group
    if group is None or dist.get_world_size(group) <= 1:
        return tensor
    tensor = tensor.contiguous()
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, tensor, group=group)
    CollectiveStats.record(Collective.ALL_GATHER, _payload(tensor))
    return torch.cat(parts, dim=dim)


def data_parallel_all_reduce(
    tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM
) -> torch.Tensor:
    """Reduce ``tensor`` across the DP group in place. No-op without DP-attention.

    Carries the *decisions* the DP ranks have to take together — how many tokens
    this step pads to, whether any rank still has unfinished work — rather than
    activations, so it is a handful of ints per step.
    """
    if _DP_GROUP is None or dist.get_world_size(_DP_GROUP) <= 1:
        return tensor
    dist.all_reduce(tensor, op=op, group=_DP_GROUP)
    CollectiveStats.record(Collective.ALL_REDUCE, _payload(tensor))
    return tensor


def reduce_scatter(
    tensor: torch.Tensor, dim: int = -1, *, group: dist.ProcessGroup | None = None
) -> torch.Tensor:
    """Sum ``tensor`` across ranks, then keep only this rank's shard along ``dim``.

    An all-reduce is a reduce-scatter followed by an all-gather; when the caller
    only wants its shard (sequence parallelism, EP output combining), the gather
    half of that traffic is pure waste, which is what this primitive skips.

    Raises:
        ValueError: If ``dim`` does not divide across the group's ranks.
    """
    if _world_of_one(group):
        return tensor
    group = _resolve(group)
    world = dist.get_world_size(group)
    if tensor.shape[dim] % world != 0:
        raise ValueError(
            f"reduce_scatter dim {dim} of shape {tuple(tensor.shape)} does not divide "
            f"across {world} ranks"
        )
    # reduce_scatter_tensor splits along dim 0, so the shard dim moves there first.
    moved = tensor.movedim(dim, 0).contiguous()
    shard = torch.empty(
        moved.shape[0] // world, *moved.shape[1:], dtype=moved.dtype, device=moved.device
    )
    dist.reduce_scatter_tensor(shard, moved, op=dist.ReduceOp.SUM, group=group)
    CollectiveStats.record(Collective.REDUCE_SCATTER, _payload(moved))
    return shard.movedim(0, dim)


#: ``(sizes, padded, device)`` -> the pool <-> rectangle position plan the ragged
#: v-collectives share. Host-computed from sizes the step handshake already agreed
#: on, and cached like the dispatcher's unit weights: warm from the pre-capture
#: eager steps, so a CUDA graph only replays the copy.
_RAGGED_PLANS: dict[tuple[tuple[int, ...], int, torch.device], torch.Tensor] = {}


def _ragged_plan(sizes: tuple[int, ...], padded: int, device: torch.device) -> torch.Tensor:
    """Positions of the ragged pool's rows in the ``world x padded`` rectangle.

    One plan serves both directions: ``gathered.index_select(0, plan)`` compacts
    the all-gather rectangle back to ``[sum(sizes)]``, and
    ``buf.index_copy_(0, plan, pool)`` scatters the pool into the zero-padded
    reduce-scatter buffer. Chunk ``r`` sits at rows
    ``[r * padded, r * padded + sizes[r])`` -- the layout
    ``dist.all_gather`` / ``dist.reduce_scatter_tensor`` build when every rank
    contributes ``padded`` rows.
    """
    key = (sizes, padded, device)
    plan = _RAGGED_PLANS.get(key)
    if plan is None:
        plan = torch.cat(
            [
                torch.arange(r * padded, r * padded + rows, dtype=torch.int64, device=device)
                for r, rows in enumerate(sizes)
            ]
        )
        _RAGGED_PLANS[key] = plan
    return plan


def all_gatherv(
    tensors: Sequence[torch.Tensor],
    dim: int = 0,
    sizes: Sequence[int] | None = None,
    *,
    group: dist.ProcessGroup | None = None,
) -> list[torch.Tensor]:
    """Gather uneven shards into one ragged concatenation per tensor.

    vLLM's ``all_gatherv`` role (``AgRsAll2AllManager.dispatch``): the DP ranks
    serve different batch sizes, so the pooled batch is the concatenation of
    unequal shards, which a plain all-gather -- every rank contributing the same
    shape -- cannot express. This transport has no ragged collective to call:
    torch exposes no ``all_gatherv`` binding and rapid_llm ships no pynccl, so
    the wire stays equal-length. Each rank pads its shard to ``max(sizes)``, one
    plain all-gather runs, and the receiver compacts the ``world x padded``
    rectangle back with a cached index plan (:func:`_ragged_plan`) -- the same
    degradation vLLM takes under CUDA graph capture, where a host-side ragged
    split is illegal. Equal sizes take the plain gather directly, with no pad
    and no compaction.

    Args:
        tensors: This rank's shards. The MoE exchange passes the hidden states
            and the routing results (weights, ids) together, as vLLM's AgRs
            dispatch does; each is gathered under the same plan.
        dim: Shard axis -- 0, the token axis, for the MoE exchange.
        sizes: Shard length per rank, in group order. ``None`` means every rank
            brings the same length.
        group: Defaults to the DP group, the axis this was added for.

    Returns:
        One gathered tensor per input, concatenated in group rank order.

    Raises:
        ValueError: If ``sizes`` is not one entry per rank, or a tensor's shard
            axis does not hold the length this rank declared.
    """
    group = _DP_GROUP if group is None else group
    if group is None or dist.get_world_size(group) <= 1:
        return list(tensors)
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    first = tensors[0]
    dim = dim % first.dim()
    if sizes is None:
        sizes = (first.shape[dim],) * world
    sizes = tuple(int(count) for count in sizes)
    if len(sizes) != world:
        raise ValueError(f"all_gatherv got {len(sizes)} sizes for {world} ranks")
    for tensor in tensors:
        if tensor.shape[dim] != sizes[rank]:
            raise ValueError(
                f"all_gatherv shard with {tensor.shape[dim]} rows along dim {dim} does "
                f"not hold the {sizes[rank]} rows this rank declared"
            )
    if len(set(sizes)) == 1:
        return [data_parallel_all_gather(tensor, dim, group=group) for tensor in tensors]
    padded = max(sizes)
    plan = _ragged_plan(sizes, padded, first.device)
    gathered_tensors = []
    for tensor in tensors:
        # Pad only the shard axis; F.pad's tuple runs from the last dim inwards.
        pads = [0, 0] * tensor.dim()
        pads[2 * (tensor.dim() - 1 - dim) + 1] = padded - tensor.shape[dim]
        padded_in = torch.nn.functional.pad(tensor, pads) if any(pads) else tensor
        gathered = data_parallel_all_gather(padded_in, dim, group=group)
        gathered_tensors.append(torch.index_select(gathered, dim, plan))
    return gathered_tensors


def reduce_scatterv(
    tensor: torch.Tensor,
    sizes: Sequence[int],
    dim: int = 0,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Sum the ragged pool across ranks, keeping this rank's shard.

    vLLM's ``reduce_scatterv`` role (``AgRsAll2AllManager.combine``, the AgRs
    combine half): every rank holds the pooled expert output -- an uneven
    concatenation, one chunk per rank -- and only wants the group-sum of its
    own chunk back. ``reduce_scatter_tensor`` splits evenly, so ragged sizes go
    through the equal-length adaptation: the pool is scattered into a
    zero-padded ``world x padded`` buffer (each chunk followed by the zeros
    standing in for its missing rows), one plain reduce-scatter runs, and the
    receiver narrows its ``padded``-row shard to ``sizes[rank]``. Equal sizes
    reduce-scatter directly and skip the padding; see :func:`all_gatherv` for
    why the wire is equal-length either way -- there is no
    ``reduce_scatterv`` binding to call.

    Args:
        tensor: The full pool -- ``sum(sizes)`` rows along ``dim``, chunk ``r``
            at the pool offset of rank ``r``, in group rank order.
        sizes: Rows per rank's chunk, in group order.
        dim: Shard axis -- 0, the token axis, for the MoE combine.
        group: Defaults to the DP group, the axis this was added for.

    Returns:
        ``sizes[this rank]`` rows, summed across the group.

    Raises:
        ValueError: If ``sizes`` is not one entry per rank, or the pool does not
            hold ``sum(sizes)`` rows along ``dim``.
    """
    group = _DP_GROUP if group is None else group
    if group is None or dist.get_world_size(group) <= 1:
        return tensor
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    sizes = tuple(int(count) for count in sizes)
    if len(sizes) != world:
        raise ValueError(f"reduce_scatterv got {len(sizes)} sizes for {world} ranks")
    dim = dim % tensor.dim()
    if tensor.shape[dim] != sum(sizes):
        raise ValueError(
            f"reduce_scatterv pool with {tensor.shape[dim]} rows along dim {dim} does not "
            f"hold the declared {sum(sizes)}"
        )
    if len(set(sizes)) == 1:
        return reduce_scatter(tensor, dim, group=group)
    moved = tensor.movedim(dim, 0).contiguous()
    padded = max(sizes)
    plan = _ragged_plan(sizes, padded, tensor.device)
    buf = torch.zeros(padded * world, *moved.shape[1:], dtype=moved.dtype, device=moved.device)
    buf.index_copy_(0, plan, moved)
    shard = torch.empty(padded, *moved.shape[1:], dtype=moved.dtype, device=moved.device)
    dist.reduce_scatter_tensor(shard, buf, op=dist.ReduceOp.SUM, group=group)
    CollectiveStats.record(Collective.REDUCE_SCATTER, _payload(buf))
    return shard.narrow(0, 0, sizes[rank]).movedim(0, dim)


def all_to_all(tensor: torch.Tensor, *, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Equal-split exchange: slice ``j`` of rank ``i``'s tensor lands on rank ``j``.

    The EP token-exchange primitive: with tokens sorted by expert id, one
    all_to_all routes every token to the rank owning its expert. Splits are along
    dim 0, whose size must divide by the world size.

    Raises:
        ValueError: If dim 0 does not divide across the group's ranks.
    """
    if _world_of_one(group):
        return tensor
    group = _resolve(group)
    world = dist.get_world_size(group)
    if tensor.shape[0] % world != 0:
        raise ValueError(
            f"all_to_all dim 0 of shape {tuple(tensor.shape)} does not divide across {world} ranks"
        )
    tensor = tensor.contiguous()
    output = torch.empty_like(tensor)
    dist.all_to_all_single(output, tensor, group=group)
    CollectiveStats.record(Collective.ALL_TO_ALL, _payload(tensor))
    return output


def send(tensor: torch.Tensor, dst: int, *, group: dist.ProcessGroup | None = None) -> None:
    """Point-to-point handoff to rank ``dst`` *within the group* (CP/PD plumbing).

    Pairs with :func:`recv`; a world of one has no peer, so both are no-ops there.
    """
    if _world_of_one(group):
        return
    dist.send(tensor.contiguous(), group_dst=dst, group=_resolve(group))
    CollectiveStats.record(Collective.SEND, _payload(tensor))


def recv(tensor: torch.Tensor, src: int, *, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    """Fill ``tensor`` from rank ``src`` *within the group*; pairs with :func:`send`."""
    if _world_of_one(group):
        return tensor
    dist.recv(tensor, group_src=src, group=_resolve(group))
    CollectiveStats.record(Collective.RECV, _payload(tensor))
    return tensor


def tensor_model_parallel_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast ``tensor`` from the TP-local ``src`` rank to all TP ranks.

    Used after sampling: rank 0 draws the next token and broadcasts it so every
    rank feeds the same input_ids on the next decode step. Without this, each
    rank would run ``torch.multinomial`` with an independent RNG state and
    diverge on the first non-greedy sample.

    No-op when ``tp_world_size == 1``.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    # ``group_src`` keeps ``src`` a group-local rank for every group, which the old
    # hand-rolled ``dp_rank * tp_size + src`` arithmetic only managed for the TP one.
    dist.broadcast(tensor, group_src=src, group=_TP_GROUP)
    CollectiveStats.record(Collective.BROADCAST, _payload(tensor))
    return tensor


def tensor_model_parallel_barrier() -> None:
    """Hold every TP rank here until all of them arrive, over the CPU group.

    Exists for teardown: ``ncclCommAbort`` was a collective call in some NCCL
    versions, so a rank that destroys its communicator while a peer is still
    working parks that peer inside ``destroy_process_group`` forever. A gloo
    barrier right before the destroy lines the ranks up so their aborts run
    back to back — the rendezvous is host-side, so it costs nothing on the
    device and cannot itself be blocked by whatever the NCCL streams hold.
    """
    if _TP_WORLD_SIZE <= 1:
        return
    dist.barrier(group=_TP_CPU_GROUP)


def tensor_model_parallel_broadcast_object_list(obj: Any = None, src: int = 0) -> Any:
    """Broadcast any picklable object from TP-local ``src`` to every TP rank.

    This is the *control* plane. A tensor-parallel step is decided once — on the
    rank that owns the scheduler — and then run everywhere, so the decision has
    to travel: which slots, which token ids, which sampling parameters. Sending
    it as an object rather than as a pile of padded tensors means the receiving
    ranks reconstruct exactly what the sender planned, with no encoding to keep
    in sync on both sides; and it goes over the gloo group, so a few hundred
    bytes of control never touch device memory or serialise behind the NCCL
    stream that the data plane is using.

    Non-root ranks ignore ``obj`` and return what the root sent. Returns ``obj``
    unchanged when ``tp_world_size == 1``, which is what lets the single-process
    path call this without a branch.
    """
    if _TP_WORLD_SIZE <= 1:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, group_src=src, group=_TP_CPU_GROUP)
    # Sizing a plan means pickling it a second time, so it happens only while a window
    # is open: observability may cost something when asked for, never when not. Sized
    # after the broadcast so a follower reports the same bytes the driver sent.
    if CollectiveStats.collecting():
        CollectiveStats.record(Collective.BROADCAST_OBJECT, len(pickle.dumps(payload[0])))
    return payload[0]


def tensor_model_parallel_all_reduce_min(value: int) -> int:
    """Smallest ``value`` across this replica's TP group.

    Used to agree on a KV-cache size: profiling runs per rank and two cards with
    different amounts of free memory would otherwise size their caches
    differently, which desynchronises every subsequent allocation index. DP
    replicas are deliberately excluded — they allocate independently, so one
    replica on a busier card is free to hold a smaller cache.
    """
    if _TP_WORLD_SIZE <= 1:
        return value
    # The tensor must land on *this process's* device, which is not ``_TP_RANK``:
    # under DP x TP the process owns device ``dp_rank * tp_size + tp_rank``, and under
    # ``CUDA_VISIBLE_DEVICES`` the ordinal is remapped again. Asking torch which device
    # the process already selected is the only spelling that is right in every case;
    # the old ``cuda:{_TP_RANK}`` made replica 1 all-reduce from replica 0's card.
    on_gpu = dist.get_backend(_TP_GROUP) == "nccl"
    device = torch.device("cuda", torch.cuda.current_device()) if on_gpu else None
    tensor = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=_TP_GROUP)
    CollectiveStats.record(Collective.ALL_REDUCE_MIN, _payload(tensor))
    return int(tensor.item())


def tensor_model_parallel_ranks_agree(value: int) -> bool:
    """Whether every TP rank passed the same ``value``. ``True`` when TP is off.

    A consensus primitive rather than a reduction, for decisions that must come
    out the same on every rank *or not be taken at all*. Whether to keep a set of
    captured CUDA graphs is one: the graphs contain collectives, so ranks that
    disagree about which graph to replay do not produce different answers — they
    stop, one of them waiting in an all-reduce its peer never issues. A caller can
    therefore branch on this result and know its peers branch with it.
    """
    if _TP_WORLD_SIZE <= 1:
        return True
    on_gpu = dist.get_backend(_TP_GROUP) == "nccl"
    device = torch.device("cuda", torch.cuda.current_device()) if on_gpu else None
    tensor = torch.tensor([value, -value], dtype=torch.int64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=_TP_GROUP)
    CollectiveStats.record(Collective.ALL_REDUCE_MIN, _payload(tensor))
    low, negated_high = tensor.tolist()
    return low == -negated_high


def p2p_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce via point-to-point send/recv for TP=2 (O3.1).

    For two ranks, an all-reduce is just ``result = a + b``. Each rank sends
    its partial to the peer and receives the peer's partial, then adds locally.
    On PCIe interconnects with small messages (decode-step hidden states, a
    few KiB), a pair of send/recv completes in 5–8 µs versus 15–25 µs for the
    NCCL ring — the ring's fixed setup cost dominates at these sizes.

    The paired nonblocking send and receive are posted together, so neither
    peer waits before both receive operations exist.

    Falls back to ``dist.all_reduce`` when the group has more than two ranks or
    uses a non-NCCL backend.
    """
    if _TP_WORLD_SIZE <= 1:
        return tensor
    if _TP_WORLD_SIZE != 2 or _TP_GROUP is None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
        CollectiveStats.record(Collective.ALL_REDUCE, _payload(tensor))
        return tensor
    backend = dist.get_backend(_TP_GROUP)
    if backend != "nccl":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=_TP_GROUP)
        CollectiveStats.record(Collective.ALL_REDUCE, _payload(tensor))
        return tensor
    peer = 1 - _TP_RANK
    send_buffer = tensor.contiguous()
    recv_buffer = torch.empty_like(tensor)
    requests = dist.batch_isend_irecv(
        [
            dist.P2POp(dist.isend, send_buffer, group_peer=peer, group=_TP_GROUP),
            dist.P2POp(dist.irecv, recv_buffer, group_peer=peer, group=_TP_GROUP),
        ]
    )
    for request in requests:
        request.wait()
    tensor.add_(recv_buffer)
    CollectiveStats.record(Collective.ALL_REDUCE, _payload(tensor))
    return tensor


def warmup_collectives() -> None:
    """Force this rank's communicators into existence. No-op when there are none.

    NCCL builds a communicator's device resources on its first collective, and a
    CUDA graph capture cannot allocate — so the first all-reduce inside a capture
    region is the one that fails, or worse, hangs while one rank allocates and the
    others wait. Issuing one throwaway all-reduce per group beforehand moves that
    initialisation outside every capture. The groups are the module's own — TP,
    the DP-attention group, the grid-wide EP group — deduplicated, since plain EP
    shares the TP group's object.

    The value is checked rather than discarded: a reduction over ones must come
    back as the group's rank count. That makes this a cheap assertion that each
    group about to be baked into a graph is the group this rank thinks it is — a
    mismatch found here raises, whereas the same mismatch found during capture
    surfaces as a hang with no message.

    Not reported to :class:`CollectiveStats`: every other collective here is
    traffic a *step* pays, and folding a one-off initialisation into that total
    would overstate the per-step cost by a constant.
    """
    groups: list[tuple[dist.ProcessGroup, int]] = []
    if _TP_GROUP is not None and _TP_WORLD_SIZE > 1:
        groups.append((_TP_GROUP, _TP_WORLD_SIZE))
    if _DP_GROUP is not None:
        groups.append((_DP_GROUP, _DP_WORLD_SIZE))
    if _EP_GROUP is not None and _EP_GROUP is not _TP_GROUP and _EP_GROUP is not _DP_GROUP:
        groups.append((_EP_GROUP, _EP_WORLD_SIZE))
    for group, expected in groups:
        on_gpu = dist.get_backend(group) == "nccl"
        device = torch.device("cuda", torch.cuda.current_device()) if on_gpu else None
        probe = torch.ones(1, dtype=torch.float32, device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=group)
        total = int(probe.item())
        if total != expected:
            raise RuntimeError(
                f"collective warmup summed {total} over a group of {expected} ranks; "
                "the process group does not span the ranks this process believes it does"
            )
