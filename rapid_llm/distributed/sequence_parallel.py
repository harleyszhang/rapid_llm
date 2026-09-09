"""Optional sequence parallelism: a token-sharded region over the decoder stack.

Megatron's shape on the module tree, not a compiler pass: between embedding and
final norm the residual stream carries only this rank's token shard, and the
boundary collectives live inside the linears that already own the TP
communication — **g-bar**: row-parallel (``o_proj``, ``down_proj``) ends its
partial sum with *reduce-scatter*, returning the shard already reduced;
**g**: column-parallel (``qkv_proj``, ``gate_up_proj``) *all-gathers* the
shard before its GEMM, because attention reads across tokens. Everything
between is per-row (RMSNorm, residual add, SwiGLU) — hand a row-wise kernel a
shard and it computes on the shard — so no sequence-parallel norm operator
exists and both block seams are covered without the region naming either.

It does not save bytes: a ring all-reduce *is* a reduce-scatter plus an
all-gather, so the pair moves what the collective it replaced moved — vLLM's
``SequenceParallelismPass`` ("lays the groundwork for subsequent fusion
passes") and SGLang's ``layernorm_sp`` ("no extra communication volume") say
as much. What the region buys: norms, residual adds and a long-context
prefill's transient activations at ``1/world_size``, plus the shape a later
``fused_matmul_reduce_scatter`` / ``fused_all_gather_matmul`` needs to fuse
the collective into the GEMM. Off by default until a deployment measures it.

Usage:
    SequenceParallelPass().apply(model)        # after the model is built
    with sequence_parallel_region(num_tokens, marked=True) as region:
        ...                                     # marked linears take the boundaries
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..utils.env_compat import env_flag, getenv
from ..utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from collections.abc import Iterator

    import torch.nn as nn

    from ..modules.linear import LinearBase

_log = get_logger(__name__)

#: Environment variable switching sequence parallelism on.
SP_ENV = "RAPID_LLM_SEQUENCE_PARALLEL"

#: Token count a step must reach before a region opens at all.
SP_MIN_TOKENS_ENV = "RAPID_LLM_SP_MIN_TOKENS"

#: Default for :data:`SP_MIN_TOKENS_ENV`. The region's win scales with the rows
#: it takes off each rank's norms and activations, while its cost — two
#: collectives per block where the all-reduce path issues one — is per call. A
#: decode step of a few hundred rows is all cost, so the floor sits above one.
DEFAULT_MIN_TOKENS = 1024

#: Attribute marking a row-parallel projection as a *g-bar* boundary.
G_BAR_ATTR = "_sp_g_bar"

#: Attribute marking a column-parallel projection as a *g* boundary.
G_ATTR = "_sp_g"

_SP_ATTRS = (G_BAR_ATTR, G_ATTR)

#: ``block attribute -> (projection path, accepted classes, boundary attribute)``.
#: Every decoder block must contribute all four, or the pass marks nothing (see
#: :meth:`SequenceParallelPass.apply`).
_ROLES = (
    ("self_attn.qkv_proj", ("ColumnParallelLinear", "QKVParallelLinear"), G_ATTR),
    ("self_attn.o_proj", ("RowParallelLinear",), G_BAR_ATTR),
    ("mlp.gate_up_proj", ("ColumnParallelLinear",), G_ATTR),
    ("mlp.down_proj", ("RowParallelLinear",), G_BAR_ATTR),
)


def sequence_parallel_enabled() -> bool:
    """Whether sequence parallelism is switched on for this process.

    Disabled by default: the region trades communication volume for activation
    memory rather than reducing it, so it wants a measured workload behind it.
    """
    return env_flag(SP_ENV)


def sequence_parallel_min_tokens() -> int:
    """Token floor a step must clear for a region to open."""
    return int(getenv(SP_MIN_TOKENS_ENV, str(DEFAULT_MIN_TOKENS)))


@dataclass(frozen=True)
class SequenceParallelRegion:
    """The open region's token geometry; every boundary reads it from here.

    Args:
        num_tokens: Rows in the full grid — what a *g* gathers back to, and what
            the position embeddings and attention metadata were built for.
        world_size: Ranks the tokens are shared across.
        rank: This process's rank within them.
    """

    num_tokens: int
    world_size: int
    rank: int

    @property
    def padded_tokens(self) -> int:
        """``num_tokens`` rounded up to a whole number of shards.

        A step's token count is whatever the batch happened to be, so the grid
        is padded rather than the region declining to open: the padded rows are
        zeros, they are only ever touched by per-row arithmetic, and
        :func:`exit_gather` drops them.
        """
        return -(-self.num_tokens // self.world_size) * self.world_size

    @property
    def local_tokens(self) -> int:
        """Rows this rank holds while the region is open."""
        return self.padded_tokens // self.world_size


_region: ContextVar[SequenceParallelRegion | None] = ContextVar(
    "rapid_llm_sequence_parallel_region", default=None
)


def current_region() -> SequenceParallelRegion | None:
    """The region this call site runs inside, or ``None``."""
    return _region.get()


def sp_active() -> bool:
    """Whether a sequence-parallel region is open around this call."""
    return current_region() is not None


def is_g_boundary(layer: nn.Module) -> bool:
    """Whether ``layer`` gathers the token shard back to the full grid here."""
    return current_region() is not None and getattr(layer, G_ATTR, False)


def is_g_bar_boundary(layer: nn.Module) -> bool:
    """Whether ``layer`` reduce-scatters its partial to a token shard here."""
    return current_region() is not None and getattr(layer, G_BAR_ATTR, False)


@contextmanager
def sequence_parallel_region(
    num_tokens: int, *, marked: bool
) -> Iterator[SequenceParallelRegion | None]:
    """Open a token-sharded region for this forward, if this step is eligible.

    Every gate is checked here, once per forward, and nothing downstream
    re-derives the answer: the boundaries are only correct as a set, and a
    ``g-bar`` that scattered without the matching ``g`` gathering would feed
    attention one rank's tokens.

    Two of the gates are other people's claims on the same collective. A
    deferred (TBO) context and an active L3 policy both overlap the row-parallel
    all-reduce that ``g-bar`` would remove outright, and they were asked for
    explicitly, so the region yields to them.

    Safe to capture: every decision this makes is *inside* the region, and what
    a capture records is the collectives, whose shapes are the bucket's. Nothing
    outside reads a region flag, which is the trap that would matter -- a Python
    assignment made during capture is not re-executed on replay, so a caller
    that cached "this step was sequence-parallel" would replay a stale answer.

    Args:
        num_tokens: Rows in this step's grid.
        marked: Whether :class:`SequenceParallelPass` marked this model's
            boundaries. A model it refused leaves this ``False``.

    Yields:
        The open region, or ``None`` when this step stays on the all-reduce path
        — in which case the caller must not scatter its input either.
    """
    from .parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    world_size = get_tensor_model_parallel_world_size()
    if not _eligible(num_tokens, marked=marked, world_size=world_size):
        yield None
        return
    region = SequenceParallelRegion(
        num_tokens=num_tokens, world_size=world_size, rank=get_tensor_model_parallel_rank()
    )
    token = _region.set(region)
    try:
        yield region
    finally:
        _region.reset(token)


def _eligible(num_tokens: int, *, marked: bool, world_size: int) -> bool:
    """Whether a region may open for a step of ``num_tokens`` rows."""
    if not marked or world_size <= 1 or num_tokens < sequence_parallel_min_tokens():
        return False
    # Imported here, not at module scope: the comm-overlap module reads this one
    # for its ``g-bar`` dispatch, so the dependency only runs one way statically.
    from ..batch_overlap.comm_overlap import comm_overlap_policy, current_deferred_ar

    if current_deferred_ar() is not None:
        return False
    policy = comm_overlap_policy()
    return not (policy.enabled and num_tokens >= policy.min_rows)


# --------------------------------------------------------------------------- #
# Region entry and exit
# --------------------------------------------------------------------------- #
def entry_scatter(hidden_states: torch.Tensor, region: SequenceParallelRegion) -> torch.Tensor:
    """Take this rank's token shard of the region's input.

    A local slice, not a collective: every rank already holds the whole
    embedding output, so the shard is free. The grid flattens to one row per
    token and comes back as ``[1, local_tokens, hidden]`` — the residual stream
    keeps three dimensions through the region so the blocks need no reshaping,
    but its rows are no longer a ``[batch, seq]`` grid.
    """
    hidden = hidden_states.shape[-1]
    flat = _pad_rows(hidden_states.reshape(-1, hidden), region.padded_tokens)
    start = region.rank * region.local_tokens
    return flat[start : start + region.local_tokens].unsqueeze(0)


def exit_gather(
    hidden_states: torch.Tensor, region: SequenceParallelRegion, shape: torch.Size
) -> torch.Tensor:
    """Rebuild the full token grid from every rank's shard, in ``shape``.

    The last collective of the region, and the one place the padding is
    discarded. Callers that index by global row — the logits gather — have to
    run after this.
    """
    from .parallel_state import tensor_model_parallel_all_gather

    flat = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    full = tensor_model_parallel_all_gather(flat, dim=0)
    return full[: region.num_tokens].view(shape)


# --------------------------------------------------------------------------- #
# The two boundaries
# --------------------------------------------------------------------------- #
def row_parallel_g_bar(layer: LinearBase, x: torch.Tensor) -> torch.Tensor:
    """*g-bar*: the row-parallel GEMM's partial, reduce-scattered to a shard.

    Replaces the layer's all-reduce rather than joining it — the caller reads
    only this rank's rows, so the gather half of that reduction would be traffic
    nobody consumes.

    Args:
        layer: The marked row-parallel projection; only ``apply_linear`` is used.
        x: ``[..., input_size]`` activations over the *full* grid (a ``g``
            gathered them), whose trailing rows may exceed the real token count
            only after the padding below.
    """
    from .parallel_state import reduce_scatter

    region = current_region()
    partial = layer.apply_linear(x)
    flat = _pad_rows(partial.reshape(-1, partial.shape[-1]), region.padded_tokens)
    return _as_input_rank(reduce_scatter(flat, dim=0), x)


def column_parallel_g(layer: LinearBase, x: torch.Tensor) -> torch.Tensor:
    """*g*: this rank's token shard all-gathered to the full grid, then the GEMM.

    The GEMM runs on the full grid because what consumes it reads across tokens
    — attention over the sequence, and the RoPE tables and attention metadata
    were built for the whole step. The padding is dropped here rather than
    carried into those: a padded row has no position.

    Args:
        layer: The marked column-parallel projection.
        x: ``[..., input_size]`` this rank's shard of the residual stream.
    """
    from .parallel_state import tensor_model_parallel_all_gather

    region = current_region()
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    full = tensor_model_parallel_all_gather(flat, dim=0)[: region.num_tokens]
    return _as_input_rank(layer.apply_linear(full), x)


def _pad_rows(flat: torch.Tensor, rows: int) -> torch.Tensor:
    """Extend ``flat`` to ``rows`` rows with zeros (a no-op when it already is)."""
    missing = rows - flat.shape[0]
    if missing <= 0:
        return flat
    return torch.cat((flat, flat.new_zeros(missing, flat.shape[1])), dim=0)


def _as_input_rank(out: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Give ``out`` (``[rows, features]``) the dimensionality ``like`` had.

    Row counts change at every boundary while the residual stream keeps one
    shape, so the leading dimension comes from the input: a two-dimensional
    caller (the fused QKV projection, which flattens before projecting) gets two
    dimensions back, and the ``[1, rows, hidden]`` stream stays three.
    """
    if like.dim() <= 2:
        return out
    return out.view(*like.shape[:-2], *out.shape)


class SequenceParallelPass:
    """Marks the linear layers that bound a sequence-parallel region.

    A region is a property of the whole stack, not of one block: the rows the
    embedding scattered are only reassembled at the final gather, so every block
    in between must hand its shard on. The pass therefore marks all four
    boundaries of every block or none at all — a stack with one block it cannot
    place is left entirely on the all-reduce path, and :attr:`refusal` says which
    one and why.

    That rule is what excludes the constructions not yet covered: MLA's
    projections are not a fused QKV, a ``SparseMoeBlock`` is not a gate/up plus
    down pair, and DeepSeek-V4's block runs its own forward with a
    hyper-connection residual.

    Args:
        enabled: Override :func:`sequence_parallel_enabled`; ``None`` reads the env.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = sequence_parallel_enabled() if enabled is None else enabled
        #: Names of the boundaries this pass marked, for observability.
        self.matched: list[str] = []
        #: Why the last :meth:`apply` marked nothing, or ``None``.
        self.refusal: str | None = None

    def apply(self, model: nn.Module) -> int:
        """Mark ``model``'s region boundaries.

        Idempotent: marks from a previous run are cleared first, so a re-run
        under a changed grid (or with the pass switched off) leaves no boundary
        marked that :attr:`matched` does not report.

        Args:
            model: The built model (before CUDA-graph capture).

        Returns:
            The number of boundaries marked; ``0`` when the pass is disabled, TP
            is off, or the model is not one this pass can shard.
        """
        self.matched.clear()
        self.refusal = None
        # Cleared before the early returns: a disabled re-run must not leave a
        # previous run's marks behind, or ``matched`` and the module tree
        # disagree about which boundaries are live.
        for module in model.modules():
            for attr in _SP_ATTRS:
                if hasattr(module, attr):
                    delattr(module, attr)
        if not self.enabled:
            return 0
        from .parallel_state import get_tensor_model_parallel_world_size

        if get_tensor_model_parallel_world_size() <= 1:
            # No peers to shard tokens across. Logged because the usual cause is
            # order, not intent: the pass runs from ``CausalLM.__init__``, and a
            # caller that builds the model before ``init_parallel`` sees the grid
            # it has not joined yet.
            self.refusal = "tensor-parallel world size is 1"
            _log.info(
                "SequenceParallelPass requested but %s; nothing marked "
                "(is the model built before init_parallel?)",
                self.refusal,
            )
            return 0

        boundaries = self._boundaries(model)
        if boundaries is None:
            _log.info("SequenceParallelPass marked nothing: %s", self.refusal)
            return 0
        for name, module, attr in boundaries:
            setattr(module, attr, True)
            self.matched.append(name)
        _log.info(
            "SequenceParallelPass marked %d region boundaries (%d blocks)",
            len(self.matched),
            len(self.matched) // len(_ROLES),
        )
        return len(self.matched)

    def _boundaries(self, model: nn.Module) -> list[tuple[str, nn.Module, str]] | None:
        """Every block's four boundaries, or ``None`` with :attr:`refusal` set.

        Resolved by attribute path from each block rather than by walking the
        tree for layer types: an MoE block's shared expert owns a
        ``down_proj`` of its own that is *not* a boundary (its output is summed
        with the routed one before the seam), and a tree walk would mark it.
        """
        blocks = [(name, module) for name, module in model.named_modules() if _is_block(module)]
        if not blocks:
            self.refusal = "the model exposes no two-stage decoder block"
            return None
        found: list[tuple[str, nn.Module, str]] = []
        for name, block in blocks:
            for path, accepted, attr in _ROLES:
                module = _resolve(block, path)
                if module is None:
                    self.refusal = f"{name}.{path} is missing"
                    return None
                if type(module).__name__ not in accepted:
                    self.refusal = (
                        f"{name}.{path} is {type(module).__name__}, not one of {'/'.join(accepted)}"
                    )
                    return None
                if attr == G_BAR_ATTR and not getattr(module, "reduce_results", True):
                    # It promised its caller the raw partial, so it can never
                    # complete the region's reduction.
                    self.refusal = f"{name}.{path} sets reduce_results=False"
                    return None
                found.append((f"{name}.{path}", module, attr))
        return found


def _is_block(module: nn.Module) -> bool:
    """Whether ``module`` is a decoder block of the shape the region shards.

    Recognised structurally rather than by class, so the pass never imports the
    model layer (which would cycle): a block is what owns the two-stage split.
    """
    return callable(getattr(module, "forward_attn_stage", None)) and callable(
        getattr(module, "forward_mlp_stage", None)
    )


def _resolve(root: nn.Module, path: str) -> object:
    """Follow a dotted attribute ``path`` from ``root``; ``None`` if it breaks."""
    current: object = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current
