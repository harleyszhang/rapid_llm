"""L2 two-batch overlap: the decode ping-pong over a deferred all-reduce.

One tensor-parallel decode step splits its batch into two halves that
ping-pong at layer-segment granularity — half A's o_proj all-reduce is on
the wire while half B's attention GEMMs are on the SMs::

    compute:  A.attn0  B.attn0  A.mlp0  B.mlp0  A.attn1  B.attn1  ...
    comm:       [AR Ao0] [AR Bo0] [AR Ad0] [AR Bd0]  ...

The split is sglang's: both halves carry the same padded row count
(:class:`TboSplitter`), so the EP all-to-all is an equal split and captured
graph shapes never drift with the batch's parity. The op stream is the
layers' own bound methods (OperationsStrategy); the caller answers the
*policy* question, this module the *execution* one — both arms run the same
stream, so the serial run is the interleaved run's reference.

Usage:
    model_forward_maybe_tbo(model, enable_tbo=True, input_ids=ids, position_ids=pos)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from ..executor.attention_metadata import AttentionMetadata
from ..kernels import skip_rmsnorm
from .comm_overlap import CommStreamPool, DeferredArContext, deferred_all_reduce
from .operations import (
    StateDict,
    execute_operations,
    execute_overlapped_operations,
)
from .operations_strategy import OperationsStrategy
from .overlap import Timeline

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..models.base import CausalLM

#: Environment variable switching the L2 two-batch overlap on (``0`` disables).
TBO_ENV = "RAPID_LLM_TBO"
TBO_MIN_ROWS_ENV = "RAPID_LLM_TBO_MIN_ROWS"

#: Per-architecture roofline estimates — peak bf16 tensor FLOPS and HBM
#: bandwidth in bytes/s — used to place the GEMM *ridge point*: the decode batch
#: above which a GEMM is compute-bound rather than weight-read-bound. These are
#: documented estimates, not measured specs; the gate only needs their ratio.
_ROOFLINE: dict[tuple[int, int], tuple[float, float]] = {
    (8, 6): (125e12, 600e9),  # GA10x: A10 / A16 / RTX 30
    (8, 0): (312e12, 2.039e12),  # GA100: A100
    (9, 0): (990e12, 3.35e12),  # GH100: H100
}
_ROOFLINE_FALLBACK: tuple[float, float] = (200e12, 1.0e12)

#: A real GEMM reaches only a fraction of peak FLOPS at small ``M``, so the batch
#: where halving stops doubling the weight read sits *above* the theoretical
#: ridge ``peak_flops / mem_bw``. This factor covers that gap, calibrated against
#: ``benchmarks/kernels/bench_tbo_cost_model.py``: on an A10 (theoretical ridge ~208) the
#: measured split penalty is still 1.98x at batch 16 and 1.26x even at 512, so
#: the profitable regime starts well past the theoretical ridge.
_RIDGE_SAFETY: float = 2.5

_ridge_cache: int | None = None


def _ridge_rows() -> int:
    """Decode batch above which a row-parallel GEMM is compute-bound.

    Below this batch a decode GEMM is weight-read-bound: its time is flat in the
    batch because reading the weight shard dominates. Splitting the batch into
    two halves then makes each half re-read the same shard, so the pair costs
    ~2x the single full-batch GEMM — measured at 1.98x for Qwen2.5-1.5B TP2 on an
    A10 (``benchmarks/kernels/bench_tbo_cost_model.py``). That doubled weight read is
    exactly what swallows the all-reduce TBO hides, which is why TBO is a net
    loss across the whole memory-bound decode range. Above the ridge the GEMM is
    compute-bound, halving keeps total FLOPS constant, and the split is ~free —
    the only regime where hiding the all-reduce is a genuine gain.
    """
    global _ridge_cache
    if _ridge_cache is None:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            peak_flops, mem_bw = _ROOFLINE.get((props.major, props.minor), _ROOFLINE_FALLBACK)
        else:
            peak_flops, mem_bw = _ROOFLINE_FALLBACK
        _ridge_cache = int(peak_flops / mem_bw * _RIDGE_SAFETY)
    return _ridge_cache


#: State keys that outlive the whole layer stack. Everything else is an
#: intermediate, and :func:`_head` asserts none survived.
_PERSISTENT_KEYS = (
    "ar",
    "ar_events",
    "atten_info",
    "position_embeddings",
    "tag",
    "timeline",
)


@dataclass(frozen=True)
class TboPolicy:
    """Policy for enabling two-batch overlap in TP decode steps.

    Args:
        enabled: Whether eligible decode steps run two-batch overlapped.
        min_rows: Decode rows a step must reach before TBO activates. By
            default this is the roofline ridge point (:func:`_ridge_rows`), so
            TBO only fires in the compute-bound regime where its split is ~free
            and never in the memory-bound range where the split doubles the
            weight read and the overlap is a net loss. An explicit
            ``RAPID_LLM_TBO_MIN_ROWS`` overrides it, which is how the parity
            tests and benchmarks force the interleave on at a small batch.
    """

    enabled: bool = False
    min_rows: int = 8

    @classmethod
    def from_env(cls) -> TboPolicy:
        raw = os.environ.get(TBO_ENV, "0").strip().lower()
        explicit = os.environ.get(TBO_MIN_ROWS_ENV)
        # An explicit floor is a deliberate override (a parity test forcing the
        # interleave at batch 2); with none set, the cost-model ridge gates TBO
        # to the compute-bound regime so it can never regress a decode step.
        min_rows = max(2, int(explicit)) if explicit is not None else _ridge_rows()
        return cls(
            enabled=raw not in ("", "0", "false", "off"),
            min_rows=min_rows,
        )

    def active(
        self, *, world_size: int, rows: int, graph_active: bool, expert_parallel: bool = False
    ) -> bool:
        """Whether a decode step of ``rows`` requests should run overlapped.

        ``graph_active`` excludes the graph path: a captured graph replays
        one fixed kernel sequence, so an eager TBO must stand down whenever
        the step might be served by a graph instead. Graphs *can* carry the
        interleave now (see :meth:`capture_eligible`) — that decision is
        made once, at capture time, not per step.

        ``expert_parallel`` exempts the EP routed path from the ridge floor.
        The ridge is a dense-TP cost model: what TBO hides is an all-reduce's
        wire time and what the split costs is a *doubled read of the same
        weight shard*. An EP step hides an all-to-all instead, and its split
        duplicates only routed activations — every rank still reads its own
        ``num_local`` expert shards once no matter how many rows arrive, so
        the memory-bound split cost is a fraction of the dense one. sglang
        and DeepSeek show the profitable EP regime starts well below this
        dense ridge, so gating EP rows on it would turn the feature off
        everywhere it matters.
        """
        if expert_parallel:
            return self.enabled and world_size > 1 and not graph_active and rows >= 2
        return self.enabled and world_size > 1 and not graph_active and rows >= self.min_rows

    def capture_eligible(
        self, *, world_size: int, batch: int, expert_parallel: bool = False
    ) -> bool:
        """Whether a graph of ``batch`` rows should record the TBO interleave.

        Same conditions as :meth:`active`, judged on the captured batch
        size at capture time instead of the live row count per step: a replay
        fixes the kernel sequence, so the shape is picked once. Batches below
        ``min_rows`` capture the plain forward and keep their eager floor.
        An EP grid ignores ``min_rows`` for the same reason :meth:`active`
        does — the dense ridge does not model the a2a trade-off.
        """
        if expert_parallel:
            return self.enabled and world_size > 1 and batch >= 2
        return self.enabled and world_size > 1 and batch >= self.min_rows


_policy_cache: TboPolicy | None = None


def tbo_policy() -> TboPolicy:
    global _policy_cache
    if _policy_cache is None:
        _policy_cache = TboPolicy.from_env()
    return _policy_cache


def reset_tbo_policy() -> None:
    global _policy_cache
    _policy_cache = None


@dataclass
class TboHalf:
    """One half of a decode step, padded to the shared micro-batch length.

    Attributes:
        input_ids: ``[padded_len, 1]`` one-token ids; rows past ``num_rows``
            repeat the half's last real row.
        positions: ``[padded_len, 1]`` absolute positions, padded the same way.
        atten_info: Metadata for this half's rows, padded to match.
        num_rows: Real rows before padding — the head keeps exactly this many
            logits, so a padded row never reaches the sampler.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    atten_info: AttentionMetadata
    num_rows: int


class TboSplitter:
    """Split a decode step into two equal-length halves.

    sglang's discipline (``_split_array_by_balanced_sum`` plus
    ``tbo_padded_len``): both micro-batches carry the same row count, so the EP
    all-to-all is an equal split with no internal padding, and a captured
    graph's shapes do not drift with the batch's parity. A decode row is one
    token, so a balanced split is an equal-count one; an odd batch pads the
    short half by repeating its last real row — the duplicate attends to the
    same request's KV, writes the same slot with the same value, and its logits
    are dropped by ``num_rows``, which is cheaper than zero-padding and then
    teaching the attention kernels about dead rows.
    """

    def split(
        self, input_ids: torch.Tensor, positions: torch.Tensor, atten_info: AttentionMetadata
    ) -> tuple[TboHalf, TboHalf]:
        rows = input_ids.shape[0]
        if rows < 2:
            raise ValueError(f"two-batch overlap needs >= 2 rows, got {rows}")
        mid = rows // 2
        padded_len = rows - mid  # ceil(rows / 2): the length both halves share
        return (
            self._half(input_ids, positions, atten_info, 0, mid, padded_len),
            self._half(input_ids, positions, atten_info, mid, rows, padded_len),
        )

    def split_prefill(
        self, input_ids: torch.Tensor, positions: torch.Tensor, atten_info: AttentionMetadata
    ) -> tuple[TboHalf, TboHalf]:
        """Split a prefill grid into two halves by *sequence*, balanced by tokens.

        A prefill row is a whole sequence of ``max_prompt_len`` columns, so the
        split unit is the sequence, not the token. sglang balances by token
        count (``_split_array_by_balanced_sum`` over ``extend_lens``) rather
        than by sequence count, because sequences differ wildly in length and
        an equal-count split would leave one half doing most of the work. The
        same padding rule applies: the short half repeats its last row, and
        ``num_rows`` tells the caller which logits to keep.
        """
        num_seqs = input_ids.shape[0]
        if num_seqs < 2:
            raise ValueError(f"prefill two-batch overlap needs >= 2 sequences, got {num_seqs}")
        seq_lens = atten_info.b_seq_len
        mid = _balanced_split_index(seq_lens) if seq_lens is not None else num_seqs // 2
        mid = max(1, min(mid, num_seqs - 1))
        padded_len = max(mid, num_seqs - mid)
        return (
            self._half(input_ids, positions, atten_info, 0, mid, padded_len),
            self._half(input_ids, positions, atten_info, mid, num_seqs, padded_len),
        )

    @staticmethod
    def _half(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        atten_info: AttentionMetadata,
        start: int,
        stop: int,
        padded_len: int,
    ) -> TboHalf:
        return TboHalf(
            input_ids=_pad_rows(input_ids[start:stop], padded_len),
            positions=_pad_rows(positions[start:stop], padded_len),
            atten_info=_narrow_metadata(atten_info, start, stop, padded_len),
            num_rows=stop - start,
        )


def _pad_rows(tensor: torch.Tensor, padded_len: int) -> torch.Tensor:
    """Grow ``[rows, ...]`` to ``[padded_len, ...]`` by repeating its last row."""
    rows = tensor.shape[0]
    if rows == padded_len:
        return tensor
    pad = tensor[-1:].expand(padded_len - rows, *tensor.shape[1:])
    return torch.cat([tensor, pad], dim=0)


def _narrow_metadata(
    meta: AttentionMetadata, start: int, stop: int, padded_len: int
) -> AttentionMetadata:
    """Metadata for rows ``[start, stop)``, padded to the micro-batch length.

    Views, no copy. The padded rows repeat the last real row's entries, so the
    duplicate token attends to the same request's KV and writes the same slot.
    """

    def slice_and_pad(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else _pad_rows(tensor[start:stop], padded_len)

    # Prefill carries b_start_loc: each sequence's offset into the flattened
    # grid. Narrowing shifts the row indices, so the offsets must be rebuilt
    # from the half's own row numbers rather than sliced from the parent's.
    b_start_loc = None
    if meta.b_start_loc is not None:
        stride = int(meta.b_start_loc[1] - meta.b_start_loc[0]) if len(meta.b_start_loc) > 1 else 0
        half_rows = torch.arange(
            padded_len, device=meta.b_start_loc.device, dtype=meta.b_start_loc.dtype
        )
        b_start_loc = half_rows * stride

    return AttentionMetadata(
        kv_buffer=meta.kv_buffer,
        cur_select_index=slice_and_pad(meta.cur_select_index),
        b_req_tokens_table=meta.b_req_tokens_table,
        b_start_loc=b_start_loc,
        b_req_idx=slice_and_pad(meta.b_req_idx),
        b_seq_len=slice_and_pad(meta.b_seq_len),
        max_actual_seq_len=meta.max_actual_seq_len,
        is_prefill=meta.is_prefill,
        b_prefix_len=slice_and_pad(getattr(meta, "b_prefix_len", None)),
        b_kv_base=slice_and_pad(getattr(meta, "b_kv_base", None)),
        max_chunk_len=getattr(meta, "max_chunk_len", None),
    )


def model_forward_maybe_tbo(
    model: CausalLM,
    *,
    enable_tbo: bool,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    atten_info: AttentionMetadata,
    prefill: bool = False,
    timeline: Timeline | None = None,
) -> torch.Tensor:
    """Run one step through the layer stack, overlapped or not.

    sglang's entry of the same name: the caller decides ``enable_tbo`` (the
    scheduler's policy question, :class:`TboPolicy` here) and this function
    owns the execution one — which op stream runs
    (:meth:`OperationsStrategy.init_new_tbo`; ``prefill`` selects the stream)
    and whether one micro-batch threads it alone or two interleave through it.

    sglang returns both arms' ``(hidden_states, residual)`` and leaves the head
    to the model wrapper; here the entry owns the head too, so a caller
    switching arms on the policy sees one output shape either way.

    Returns:
        ``[rows, ..., vocab]`` logits — ``[rows, 1, vocab]`` for a decode step,
        the prefill grid's shape otherwise — rows in batch order.
    """
    if input_ids.device.type == "cpu":
        return model(input_ids, position_ids, atten_info)
    inputs = {"input_ids": input_ids, "position_ids": position_ids, "atten_info": atten_info}
    operations_strategy = OperationsStrategy.init_new_tbo(model.layers, prefill=prefill)
    if enable_tbo:
        return _model_forward_tbo(
            model, inputs, operations_strategy, prefill=prefill, timeline=timeline
        )
    return _model_forward_non_tbo(model, inputs, operations_strategy, timeline=timeline)


def _model_forward_tbo(
    model: CausalLM,
    inputs: dict[str, Any],
    operations_strategy: OperationsStrategy,
    *,
    prefill: bool,
    timeline: Timeline | None,
) -> torch.Tensor:
    """The TBO arm: split, interleave, merge — sglang's three sub-steps.

    ``prefill`` selects the prefill op stream: strict alternation with the
    shared MLP hiding behind the return exchange, rather than the decode
    stream's lead of two stages.
    """
    halves = _model_forward_tbo_split_inputs(inputs, prefill=prefill)
    device = halves[0].input_ids.device
    resolved = timeline or CommStreamPool.for_device(device).timeline
    expert_parallel = _expert_parallel(model)
    with deferred_all_reduce(device) as ar:
        states = [
            _initial_state(
                model,
                half.input_ids,
                half.positions,
                half.atten_info,
                ar,
                resolved,
                tag,
                expert_parallel,
            )
            for half, tag in zip(halves, "ab", strict=True)
        ]
        outputs = execute_overlapped_operations(
            states,
            [operations_strategy.operations, operations_strategy.operations],
            delta_stages=[0, operations_strategy.tbo_delta_stages],
        )
        ar.drain()
    return _model_forward_tbo_merge_outputs(model, outputs, halves, expert_parallel)


def _model_forward_tbo_split_inputs(
    inputs: dict[str, Any], *, prefill: bool
) -> tuple[TboHalf, TboHalf]:
    """Split the step into two halves — decode by row, prefill by sequence."""
    splitter = TboSplitter()
    split = splitter.split_prefill if prefill else splitter.split
    return split(inputs["input_ids"], inputs["position_ids"], inputs["atten_info"])


def _model_forward_tbo_merge_outputs(
    model: CausalLM,
    outputs: Sequence[StateDict],
    halves: Sequence[TboHalf],
    expert_parallel: bool,
) -> torch.Tensor:
    """Head each half, keep its real rows, and concatenate back to batch order.

    sglang's merge stitches ``(hidden_states, residual)`` by the halves'
    parent token ranges; the head plays that role against the rows each half
    kept — the padded rows a split produced are duplicates whose logits must
    not reach the sampler.
    """
    persistent = _PERSISTENT_KEYS + (("moe_ctx",) if expert_parallel else ())
    return torch.cat(
        [
            _head(model, state, half.num_rows, persistent)
            for state, half in zip(outputs, halves, strict=True)
        ],
        dim=0,
    )


def _model_forward_non_tbo(
    model: CausalLM,
    inputs: dict[str, Any],
    operations_strategy: OperationsStrategy,
    *,
    timeline: Timeline | None,
) -> torch.Tensor:
    """The plain arm: one micro-batch through the same op stream, serially.

    sglang's ``_model_forward_non_tbo``: :func:`execute_operations` runs the
    stream the interleave would, with nothing beside it. The deferred
    all-reduce context stays on — the ops fence on it — and with a single
    stream the fences land where a blocking reduction would have, so the
    result matches the plain layer-by-layer forward while keeping *one*
    definition of what a layer does.
    """
    device = inputs["input_ids"].device
    resolved = timeline or CommStreamPool.for_device(device).timeline
    expert_parallel = _expert_parallel(model)
    with deferred_all_reduce(device) as ar:
        state = _initial_state(
            model,
            inputs["input_ids"],
            inputs["position_ids"],
            inputs["atten_info"],
            ar,
            resolved,
            "a",
            expert_parallel,
        )
        state = execute_operations(state, operations_strategy.operations)
        ar.drain()
    persistent = _PERSISTENT_KEYS + (("moe_ctx",) if expert_parallel else ())
    num_rows = inputs["input_ids"].shape[0]
    return _head(model, state, num_rows, persistent)


def _initial_state(
    model: CausalLM,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    atten_info: AttentionMetadata,
    ar: DeferredArContext,
    timeline: Timeline,
    tag: str,
    expert_parallel: bool,
) -> StateDict:
    """One micro-batch's state: the step's inputs plus the overlap plumbing.

    ``moe_ctx`` is added only when a layer runs expert parallel, so dense
    and TP-MoE runs never import the MoE module.
    """
    hidden_states = model.get_input_embeddings(input_ids)
    initial: dict = {
        "hidden_states": hidden_states,
        "residual": None,
        "position_embeddings": model.rotary_emb(hidden_states, positions),
        "atten_info": atten_info,
        "ar": ar,
        "ar_events": [],
        "timeline": timeline,
        "tag": tag,
    }
    if expert_parallel:
        from ..modules.moe import MoEOpContext  # lazy: dense runs stay kernel-light

        initial["moe_ctx"] = MoEOpContext()
    return StateDict(initial)


def _head(
    model: CausalLM,
    state: StateDict,
    num_rows: int,
    persistent: tuple[str, ...],
) -> torch.Tensor:
    """Final norm and vocabulary projection for one micro-batch.

    Keeps the first ``num_rows`` logits only: the padded rows a split
    produced are duplicates, and their predictions must not reach the
    sampler. ``clear`` then proves the stack released every intermediate —
    a leftover is an op that skipped its pop.
    """
    hidden_states = state.pop("hidden_states")
    residual = state.pop("residual")
    state.clear(persistent)
    hidden, _ = skip_rmsnorm(hidden_states, residual, model.norm_weight, model.rms_norm_eps)
    return model.lm_head(hidden)[:num_rows]


def _expert_parallel(model: CausalLM) -> bool:
    """Whether any layer runs MoE over an expert-parallel dispatcher."""
    return any(getattr(layer.mlp, "dispatcher", None) is not None for layer in model.layers)


def _balanced_split_index(seq_lens: torch.Tensor) -> int:
    """The sequence index that splits total tokens most evenly.

    sglang's ``_split_array_by_balanced_sum``: walk the cumulative sum and take
    the index where the two sides are closest. Sequences vary widely in length,
    so splitting by count would hand one half most of the tokens and the
    overlap would degenerate into waiting for the heavier half.
    """
    lens = seq_lens.tolist()
    total = sum(lens)
    best_index, best_diff, running = 1, None, 0
    for i in range(1, len(lens)):
        running += lens[i - 1]
        diff = abs(running - (total - running))
        if best_diff is None or diff <= best_diff:
            best_diff, best_index = diff, i
        else:
            break
    return best_index
