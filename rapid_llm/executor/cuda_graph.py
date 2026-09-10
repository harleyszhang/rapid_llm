"""CUDA Graph capture and replay for the decode phase.

:class:`CUDAGraphManager` captures one graph per (batch, seq-bucket) shape via
:class:`CUDAGraphRunner`; ``try_replay`` launches the matching graph when a decode
step fits one, else returns None so eager runs. Lazy mode (O13) captures only a
seed pair at startup and the rest on first use, so cold start stays seconds-scale;
a shape whose on-demand capture OOMs is blacklisted and runs eager.

Under TP a captured region contains collectives, so every rank must choose the
same graph. Startup grids are fingerprinted, and lazy captures reach consensus
before a newly captured shape can replay.

Usage:
    mgr = CUDAGraphManager(model)
    logits = mgr.try_replay(input_ids, position_ids, attn_info)
"""

from __future__ import annotations

import logging
import os
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..distributed.parallel_state import (
    expert_parallel_enabled,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce_min,
    tensor_model_parallel_ranks_agree,
    warmup_collectives,
)
from ..engine.prefix_cache import PREFIX_CACHE_BLOCK_SIZE
from .attention_metadata import AttentionMetadata

logger = logging.getLogger(__name__)

# One graph per (batch_size, seq_len_bucket): a graph fixes the input shapes and
# ``max_actual_seq_len``, so both axes are enumerated. Past the largest bucket a
# step falls back to eager.
DEFAULT_BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_SEQ_LEN_BUCKETS: tuple[int, ...] = (256, 512, 1024, 2048, 4096)

# ~38 MB per graph on a 0.5B fp16 model, rounded up; the OOM fallback in
# ``ModelRunner.enable_cuda_graph`` covers models that exceed it.
WORKSPACE_BYTES_PER_GRAPH: int = 64 * 1024**2

# Lazy mode seeds these two shapes (O13): batch 1 on the smallest bucket, largest
# batch on the largest. In-between shapes pay ~0.5-1 s once, on first use.
LAZY_SEED_SHAPES: int = 2

# Largest graph-vs-eager logit difference tolerated when keeping graphs under TP.
# Not zero: a replayed all-reduce may sum in a different order (~1e-3 in bf16).
TP_GRAPH_PARITY_ATOL: float = 1e-2

# Seed for the parity check's synthetic step, drawn on the host so every rank
# generates identical ids without a broadcast.
_PARITY_SEED: int = 0x5EED

# Opt-in per-step check that every rank picked the same graph; off by default (it
# adds a collective to the decode path).
_LOCKSTEP_ENV: str = "RAPID_LLM_TP_GRAPH_CHECK"


def estimate_capture_workspace(max_seq_len: int, *, lazy: bool = False) -> int:
    """Upper bound on the bytes the capture plan will pin.

    An upper bound by necessity: the KV profiler runs before ``max_request_num``
    exists, so every default batch size is assumed to survive clamping. Lazy mode
    reserves only the seed pair; on-demand graphs take workspace from what is free,
    and a shape that OOMs stays eager.
    """
    n_buckets = sum(1 for b in DEFAULT_SEQ_LEN_BUCKETS if b <= max_seq_len)
    graphs = LAZY_SEED_SHAPES if lazy else len(DEFAULT_BATCH_SIZES) * max(n_buckets, 1)
    return graphs * WORKSPACE_BYTES_PER_GRAPH


@dataclass(frozen=True)
class _GraphKey:
    """Identifies one captured graph by its persistent shape."""

    batch_size: int
    seq_len_bucket: int


class CUDAGraphRunner:
    """One captured decode step for a fixed ``(batch_size, seq_len_bucket)`` pair.

    All tensors the graph reads or writes live inside the runner; callers push new
    values via :meth:`replay` and get the output logits back.

    Args:
        model: Eager module to capture.
        batch_size: Sequences the graph serves.
        seq_len_bucket: ``max_actual_seq_len`` at capture; replay is valid only
            when the current context length is ``<=`` this.
        kv_buffer: Layer-wise paged KV tensors (shared with the executor).
        b_req_tokens_table: Request-to-cache-row map (shared with the executor).
        device: Torch device string.
        step: The callable to record — ``(input_ids, position_ids, atten_info) ->
            logits``. ``None`` records the model's plain forward; a graph
            captured in the two-batch-overlap shape records the interleaved
            halves instead (their deferred all-reduces ride the comm stream,
            whose fork/join event edges record into the graph like any other
            dependency, so replay keeps the compute/comm overlap).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        batch_size: int,
        seq_len_bucket: int,
        kv_buffer: list[torch.Tensor],
        b_req_tokens_table: torch.Tensor,
        device: str = "cuda",
        step: Callable[[torch.Tensor, torch.Tensor, AttentionMetadata], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.seq_len_bucket = seq_len_bucket
        self.device = device
        self._step = step if step is not None else model

        # Persistent input surface — everything the graph reads goes through these.
        self.input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        self.atten_info = AttentionMetadata()
        self.atten_info.kv_buffer = kv_buffer  # shared list; storage is persistent
        self.atten_info.b_req_tokens_table = b_req_tokens_table
        self.atten_info.cur_select_index = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.atten_info.b_seq_len = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.atten_info.b_req_idx = torch.arange(batch_size, dtype=torch.long, device=device)

        # A Python int, so it is baked in: it fixes flash_decoding's mid_o shape and grid.
        self.atten_info.max_actual_seq_len = seq_len_bucket
        self.atten_info.is_prefill = False

        self._graph: torch.cuda.CUDAGraph | None = None
        self._output: torch.Tensor | None = None

    def capture(self, warmup_metadata: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        """Warm up on a side stream, then record the graph on the current stream.

        ``warmup_metadata`` is ``(b_req_idx, cur_select_index)`` from the live step
        that triggered a lazy capture: the warmup writes throwaway K/V where that
        step's real pass is about to write, so the replay overwrites it. ``None``
        (startup capture) keeps the zero buffers (cache empty, rows 0..batch safe).
        """
        if warmup_metadata is not None:
            b_req_idx, cur_select_index = warmup_metadata
            # Warm up on real work: at zero length stage 1 visits no K/V rows, so a
            # fault would surface inside the capture, not the warmup.
            self.atten_info.b_req_idx.copy_(b_req_idx)
            self.atten_info.cur_select_index.copy_(cur_select_index)
            # The longest legal length walks every split branch the kernel has,
            # and writes at a position the real request has not reached yet.
            self.atten_info.b_seq_len.fill_(
                min(self.seq_len_bucket, self.atten_info.b_req_tokens_table.shape[1] - 1)
            )
        else:
            # Warm up on real work (see above).
            self.atten_info.b_seq_len.fill_(min(self.seq_len_bucket, 32))

        # Distinct token per row instead of the all-zero buffer: identical tokens
        # embed identically and route every slot to the same experts, which under
        # EP piles ~all the load on one rank -- a degenerate imbalance that a
        # capacity-bounded MoE dispatch would reject during this eager warmup even
        # though real (varied) traffic never looks like that. arange stays well
        # inside the vocab (batch << vocab) and only feeds the embedding lookup;
        # replay overwrites input_ids with the live batch, so the values here do
        # not affect the recorded graph's correctness.
        self.input_ids.copy_(
            torch.arange(self.batch_size, device=self.device, dtype=self.input_ids.dtype).unsqueeze(1)
        )

        # The capture stream must be idle, so warmup runs on its own stream, fenced
        # both ways. Those passes force Triton JIT, cuBLAS workspaces and allocator
        # blocks to happen *before* the capture, which cannot allocate — and under
        # the TBO shape they also settle the NCCL communicator's channels, which
        # must exist before its kernels can be recorded.
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                _ = self._step(self.input_ids, self.position_ids, self.atten_info)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        # Under TP the capture region contains collectives; assert NCCL is already
        # initialised (the failure it prevents presents as a hang, not an error).
        warmup_collectives()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._output = self._step(self.input_ids, self.position_ids, self.atten_info)

    def _fill_probe_inputs(self, vocab_size: int) -> None:
        """Load a synthetic but valid decode step into the persistent buffers.

        Seeded on the host, so two TP ranks produce identical ids with no broadcast.
        KV rows are ``arange``, not all-zero: pointed at row 0 the batch's writes
        would race and the step would stop being reproducible.
        """
        generator = torch.Generator().manual_seed(
            _PARITY_SEED + self.batch_size * 100_003 + self.seq_len_bucket
        )
        ids = torch.randint(0, vocab_size, (self.batch_size, 1), generator=generator)
        length = min(self.seq_len_bucket, 32)

        self.input_ids.copy_(ids)
        self.position_ids.fill_(length - 1)
        self.atten_info.b_seq_len.fill_(length)
        self.atten_info.cur_select_index.copy_(
            torch.arange(self.batch_size, dtype=self.atten_info.cur_select_index.dtype)
        )
        self.atten_info.b_req_idx.copy_(
            torch.arange(self.batch_size, dtype=self.atten_info.b_req_idx.dtype)
        )

    def parity_error(self, vocab_size: int) -> float:
        """Largest absolute logit difference between this graph and its eager step.

        Both halves run on the same synthetic inputs through the *same callable the
        capture recorded* (:attr:`_step`, not the plain forward), so anything beyond
        floating point reassociation means the graph is not computing what its step
        computes — a stale pointer, an uncaptured buffer, a collective on the wrong
        stream. Under TP that is worth checking before graphs go live (the same bug
        with a collective desynchronises the group and hangs). Safe here only because
        no request exists yet: both forwards scribble low cache rows, as the capture
        warmup did, and every row is rewritten before it is read for real.

        The step shape matters: a two-batch-overlap graph splits the batch and runs
        half-width GEMMs over half-size expert dispatch buffers, so against the
        plain forward its logits drift by reassociations that a deep stack amplifies
        past any tolerance. Comparing like with like keeps the gate's signal — a
        broken capture still fails hard — without punishing the overlap for running
        a different, equally valid decomposition.
        """
        if self._graph is None or self._output is None:
            raise RuntimeError("capture() must be called before parity_error()")

        self._fill_probe_inputs(vocab_size)
        eager = self._step(self.input_ids, self.position_ids, self.atten_info).float().clone()
        # Replayed directly (not via :meth:`replay`): the buffers already hold the
        # probe values, so a copy onto themselves would add nothing.
        self._graph.replay()
        return (eager - self._output.float()).abs().max().item()

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cur_select_index: torch.Tensor,
        b_seq_len: torch.Tensor,
        b_req_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Push new inputs into the persistent buffers and replay the graph.

        The caller keeps ``max_actual_seq_len`` ``<=`` this runner's
        :attr:`seq_len_bucket` (the executor enforces it via bucket selection).
        """
        if self._graph is None or self._output is None:
            raise RuntimeError("capture() must be called before replay()")

        # ``.copy_()`` writes into the SAME storage the graph captured, so its
        # baked pointers stay valid.
        self.input_ids.copy_(input_ids.view(self.batch_size, 1))
        self.position_ids.copy_(position_ids.view(self.batch_size, 1))
        self.atten_info.cur_select_index.copy_(cur_select_index)
        self.atten_info.b_seq_len.copy_(b_seq_len)
        self.atten_info.b_req_idx.copy_(b_req_idx.to(self.atten_info.b_req_idx.dtype))

        self._graph.replay()
        return self._output


class CUDAGraphManager:
    """Holds one :class:`CUDAGraphRunner` per ``(batch_size, seq_len_bucket)``.

    Args:
        model: Eager model to capture.
        kv_buffer: Layer-wise KV tensors owned by the executor.
        b_req_tokens_table: Request-to-cache-row map owned by the executor.
        batch_sizes: Batch sizes to capture; smaller batches fall back to eager.
        seq_len_buckets: Ascending ``max_actual_seq_len`` ceilings to capture.
        device: Torch device string.
        step_factory: ``batch_size -> step`` (or ``None`` for the plain forward);
            the runner records whatever it is handed. The factory decides the
            two-batch-overlap shape per batch — typically by asking the TBO
            policy whether that batch size reaches its activation floor.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        kv_buffer: list[torch.Tensor],
        b_req_tokens_table: torch.Tensor,
        batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        seq_len_buckets: tuple[int, ...] = DEFAULT_SEQ_LEN_BUCKETS,
        device: str = "cuda",
        lazy: bool = False,
        step_factory: Callable[[int], Callable | None] | None = None,
    ) -> None:
        self.model = model
        self.kv_buffer = kv_buffer
        self.b_req_tokens_table = b_req_tokens_table
        self.batch_sizes = tuple(sorted(set(batch_sizes)))
        self.seq_len_buckets = tuple(sorted(set(seq_len_buckets)))
        if not self.batch_sizes or any(size < 1 for size in self.batch_sizes):
            raise ValueError(f"CUDA graph batch sizes must be positive: {self.batch_sizes}")
        if not self.seq_len_buckets or any(size < 1 for size in self.seq_len_buckets):
            raise ValueError(
                f"CUDA graph sequence-length buckets must be positive: {self.seq_len_buckets}"
            )
        self.device = device
        self._lazy = lazy
        self._step_factory = step_factory
        self._runners: dict[_GraphKey, CUDAGraphRunner] = {}
        # Shapes whose on-demand capture failed (usually OOM): never retried.
        self._failed: set[_GraphKey] = set()
        # Steps served by a replay; capturing and *using* graphs are separate facts
        # (a never-matching bucket leaves decode eager while graphs sit in memory).
        self.replays: int = 0
        # Worst graph-vs-eager logit difference, set by whichever gate ran the
        # comparison. Recorded, not re-measured on demand: the comparison replays
        # graphs, and under TP a replayed all-reduce deadlocks once followers are
        # in their serve loop.
        self.parity_error: float | None = None
        # Per-graph parity at the same gate run, keyed "b<batch>@<bucket>" —
        # the diagnostics a rejected grid needs (which shape, how badly).
        self.parity_errors: dict[str, float] = {}
        # Read once: consulted on every decode step.
        self._check_lockstep = os.environ.get(_LOCKSTEP_ENV) == "1" and get_tensor_model_parallel_world_size() > 1

    def _new_runner(self, key: _GraphKey) -> CUDAGraphRunner:
        """Build the runner for ``key``, asking the factory for its step shape."""
        step = self._step_factory(key.batch_size) if self._step_factory is not None else None
        return CUDAGraphRunner(
            self.model,
            batch_size=key.batch_size,
            seq_len_bucket=key.seq_len_bucket,
            kv_buffer=self.kv_buffer,
            b_req_tokens_table=self.b_req_tokens_table,
            device=self.device,
            step=step,
        )

    def capture_all(self) -> None:
        """Capture a graph for every ``(batch_size, seq_len_bucket)`` pair."""
        for bs in self.batch_sizes:
            for bucket in self.seq_len_buckets:
                key = _GraphKey(bs, bucket)
                runner = self._new_runner(key)
                runner.capture()
                self._runners[key] = runner

    def capture_seed(self) -> None:
        """Capture only the seed pair; the rest wait for their first use.

        Seeds: batch 1 on the smallest bucket (a fresh request starts on the graph
        path) and the largest batch on the largest bucket. In-between shapes are
        captured on demand inside :meth:`try_replay`.
        """
        if not self.batch_sizes or not self.seq_len_buckets:
            return
        seeds = (
            _GraphKey(self.batch_sizes[0], self.seq_len_buckets[0]),
            _GraphKey(self.batch_sizes[-1], self.seq_len_buckets[-1]),
        )
        for key in dict.fromkeys(seeds):  # de-duplicated, order kept
            runner = self._new_runner(key)
            runner.capture()
            self._runners[key] = runner

    def _on_grid(self, key: _GraphKey) -> bool:
        """Whether the key belongs to the configured capture grid."""
        return key.batch_size in self.batch_sizes and key.seq_len_bucket in self.seq_len_buckets

    def _capture_on_miss(
        self, key: _GraphKey, atten_info: AttentionMetadata
    ) -> CUDAGraphRunner | None:
        """Capture a missing shape right where a step asked for it (O13).

        The warmup borrows the live step's ``b_req_idx``/``cur_select_index`` so its
        throwaway writes land where the real pass (replaying right after) writes. A
        failure (typically OOM) blacklists the shape and this step runs eager.
        """
        if key in self._failed or not self._on_grid(key):
            return None
        runner: CUDAGraphRunner | None = None
        try:
            # The capture stream must be the only work in flight: pending readback
            # copies and kernels must land first.
            torch.cuda.synchronize()
            runner = self._new_runner(key)
            runner.capture(warmup_metadata=(atten_info.b_req_idx, atten_info.cur_select_index))
            self._runners[key] = runner
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(
                exc
            ).lower():
                raise
            runner = None
            self._failed.add(key)
            logger.warning(
                "Lazy capture of decode graph batch=%d bucket=%d ran out of "
                "memory; that shape stays eager",
                key.batch_size,
                key.seq_len_bucket,
            )
        if get_tensor_model_parallel_world_size() > 1 and not tensor_model_parallel_ranks_agree(
            int(runner is not None)
        ):
            # A peer OOM'd. Retire this rank's successful capture too: replaying
            # it while the peer runs eager would deadlock at the first collective.
            self._runners.pop(key, None)
            self._failed.add(key)
            runner = None
            torch.cuda.empty_cache()
            logger.warning(
                "Lazy capture of decode graph batch=%d bucket=%d differed across "
                "tensor-parallel ranks; that shape stays eager everywhere",
                key.batch_size,
                key.seq_len_bucket,
            )
        if runner is not None:
            logger.info(
                "Lazy-captured decode graph batch=%d bucket=%d on first use "
                "(%d/%d shapes captured)",
                key.batch_size,
                key.seq_len_bucket,
                len(self._runners),
                len(self.batch_sizes) * len(self.seq_len_buckets),
            )
        return runner

    def grid_fingerprint(self) -> int:
        """A number equal on two ranks exactly when their grids are.

        Compared across ranks before graphs serve traffic. Grid agreement matters:
        a rank one bucket short runs eager while its peer replays, and the peer's
        captured all-reduce waits on a collective never issued — a hang, not a wrong
        answer, so it is excluded up front. ``crc32`` (not :func:`hash`, unstable
        across processes), offset by one so a real grid never collides with the ``0``
        meaning "captured nothing".
        """
        grid = repr(
            (
                self.batch_sizes,
                self.seq_len_buckets,
                sorted((key.batch_size, key.seq_len_bucket) for key in self._runners),
            )
        )
        return 1 + zlib.crc32(grid.encode())

    def max_parity_error(self, vocab_size: int) -> float:
        """Worst :meth:`CUDAGraphRunner.parity_error` over every captured graph.

        Every graph is checked, not a sample: they differ in batch size and
        ``max_actual_seq_len`` (which selects the FlashDecoding grid), so a fault can
        live in one bucket alone. Only safe while every rank calls it (it replays
        graphs, and a captured all-reduce needs its peers); the result is left in
        :attr:`parity_error`.
        """
        errors = {
            f"b{runner.batch_size}@{runner.seq_len_bucket}": runner.parity_error(vocab_size)
            for runner in self._runners.values()
        }
        self.parity_errors = {key: round(value, 6) for key, value in errors.items()}
        self.parity_error = max(errors.values(), default=0.0)
        return self.parity_error

    def discard(self) -> None:
        """Drop every captured graph and release its memory.

        Called when a gate rejects the captured set. Graphs hold a private pool
        returned only when the last reference goes, and the KV profiler already
        budgeted that memory, so leaving it pinned would shrink the cache for an
        unused feature.
        """
        self._runners.clear()
        torch.cuda.empty_cache()

    def __len__(self) -> int:
        """Number of captured graphs."""
        return len(self._runners)

    def _pick_bucket(self, current_max_seq_len: int) -> int | None:
        """Smallest bucket ceiling that fits; ``None`` if the request is too long."""
        for bucket in self.seq_len_buckets:
            if bucket >= current_max_seq_len:
                return bucket
        return None

    def pad_to(self, batch_size: int) -> int | None:
        """Smallest captured batch size that can absorb ``batch_size``.

        Continuous batching rarely lands on the captured grid; padding to a captured
        size and discarding filler rows keeps steps on the graph path (the extra rows
        cost far less than the ~300 launches an eager step pays). ``None`` when the
        batch exceeds anything captured.
        """
        for bs in self.batch_sizes:
            if bs >= batch_size:
                return bs
        return None

    def try_replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        atten_info: AttentionMetadata,
    ) -> torch.Tensor | None:
        """Run the matching captured graph if one exists, else return ``None``.

        Only decode steps (``seq_len == 1``) with a supported batch size and a
        max-seq length fitting the largest bucket are eligible; the caller falls back
        to eager on ``None``. Under TP the decision is identical on every rank
        structurally: it reads only ``input_ids.shape`` and ``max_actual_seq_len``,
        both derived from the broadcast ``ModelInput``. ``RAPID_LLM_TP_GRAPH_CHECK=1``
        verifies that per step (costs a collective, so off by default).
        """
        runner = self._select(input_ids, atten_info)
        if self._check_lockstep:
            self._assert_lockstep(runner)
        if runner is None:
            return None
        self.replays += 1
        return runner.replay(
            input_ids,
            position_ids,
            atten_info.cur_select_index,
            atten_info.b_seq_len,
            atten_info.b_req_idx,
        )

    def _select(
        self, input_ids: torch.Tensor, atten_info: AttentionMetadata
    ) -> CUDAGraphRunner | None:
        """The graph this step is eligible for, or ``None`` for the eager path.

        Lazy mode captures a missing shape here (not skips it), keeping the on-demand
        capture inside the decision every rank computes identically (the shape is a
        function of the broadcast ``ModelInput``). A one-rank capture failure is
        resolved by the miss path before either rank can replay.
        """
        batch_size, seq_len = input_ids.shape
        if seq_len != 1:
            return None
        bucket = self._pick_bucket(atten_info.max_actual_seq_len)
        if bucket is None:
            return None
        key = _GraphKey(batch_size, bucket)
        runner = self._runners.get(key)
        if runner is None and self._lazy:
            runner = self._capture_on_miss(key, atten_info)
        return runner

    def _assert_lockstep(self, runner: CUDAGraphRunner | None) -> None:
        """Fail loudly when the ranks of a TP group chose differently.

        The eager fallback counts as a choice (one rank replaying while another runs
        eager is the mismatch that hangs), so ``None`` gets its own fingerprint.
        Raising is the point: convert a wedged group into a stack trace; the collective
        is issued unconditionally so it cannot itself desynchronise.
        """
        choice = (
            0
            if runner is None
            else 1 + zlib.crc32(f"{runner.batch_size}:{runner.seq_len_bucket}".encode())
        )
        if not tensor_model_parallel_ranks_agree(choice):
            raise RuntimeError(
                "tensor-parallel ranks disagree on which decode CUDA graph to replay "
                f"(this rank chose {choice}); the group would deadlock in the graph's "
                "all-reduce. Run with RAPID_LLM_TP_CUDA_GRAPH=0 to fall back to eager."
            )


# --------------------------------------------------------------------------- #
# Prefill graphs: the same capture discipline applied to [n_seqs, width] grids.
# --------------------------------------------------------------------------- #
#: Grid widths a prefill pass may be captured at. The width must match exactly
#: (no column padding); the request count pads up to a batch bucket with rows
#: whose K/V writes land in the reserved null block. Dense where live traffic
#: is — short prompts and one-token resume chunks — coarse past it. The 512
#: ceiling mirrors the scheduler's ``DEFAULT_MAX_CHUNK_SIZE``: chunking is on
#: by default, so no grid is ever wider, and wide grids (where the host launch
#: cost is a small fraction of the GPU time) are not worth a pool anyway.
PREFILL_WIDTH_BUCKETS: tuple[int, ...] = (
    *range(1, 33),
    40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256,
    320, 384, 448, 512,
)

#: Token ceiling for a capturable grid (``n_bucket * width``). The private
#: pool a capture pins grows with the token count; past this the graph saves
#: less wall time than the pool is worth.
PREFILL_GRAPH_MAX_TOKENS = 8192

#: Same ceiling under EP, far lower: the dispatch workspace pinned per capture
#: is ``ep_size * tokens * top_k`` rows of the exchange layout, ~10 MiB per
#: token per layer, so a 1k-token grid already pins ~7 GiB across 48 layers.
#: Past this the pool costs more than the launch savings buy.
PREFILL_GRAPH_MAX_TOKENS_EP = 1024

#: Budget withheld from the KV pool for prefill-graph pools when the feature
#: is on. Lazy capture cannot know its shapes at profiling time, so this is a
#: flat allowance; shapes that would exceed what remains stay eager (the
#: free-memory gate in ``_capture_and_run`` decides). The EP figure matches
#: one ~1k-token grid's dispatch workspace plus a second small grid.
PREFILL_GRAPH_RESERVED_BYTES = 2 << 30  # 2 GiB
PREFILL_GRAPH_RESERVED_BYTES_EP = 9 << 30  # 9 GiB

#: Opt-out for the prefill grid graphs (``=0`` falls back to eager prefill).
_PREFILL_GRAPH_ENV = "RAPID_LLM_PREFILL_GRAPH"

#: Free VRAM a prefill capture must leave untouched. Capture pins its workspace
#: in a private graph pool for the process's lifetime, and the KV profiler's
#: 10% margin is already mostly spoken for by the decode grid (EP's a2a buffers
#: make it larger still); a capture that succeeds but leaves the card full
#: turns the *next* routine allocation (sampler logits, uploads) into an OOM
#: far from the cause. Shapes are refused while free memory sits below this.
_PREFILL_GRAPH_MIN_FREE = 4 << 30  # 4 GiB


class PrefillGraphRunner:
    """One captured prefill grid step for a fixed ``(n_seqs, width)`` pair.

    The eager prefill pays its ~50 ms of host-side launch work on every first
    token; this runner pays it once at capture. Everything the recorded op
    stream reads lives in persistent buffers; :meth:`replay` copies the live
    step's values in and replays.

    The chunked attention metadata is always armed: at ``prefix_len == 0`` the
    chunked kernel is the no-pad kernel with one extra indirection (it reads
    the cache rows this same pass just wrote), so one recorded stream serves
    first chunks and prefix-cache resumes alike. Grid width fixes every host
    side tiling decision (``max_actual_seq_len``/``max_chunk_len``), and the
    kernels mask by the device-side lengths, so any per-row length ``<= width``
    replays correctly.

    Rows past the live request count are padding: they attend the reserved
    null block (slot-table entries are zero-initialised, and the allocator
    never hands block 0 out), so their K/V writes land where no live sequence
    reads, and the caller discards their logits.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        n_seqs: int,
        width: int,
        kv_buffer: list[torch.Tensor],
        b_req_tokens_table: torch.Tensor,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.n_seqs = n_seqs
        self.width = width
        self.device = device

        self.input_ids = torch.zeros(n_seqs, width, dtype=torch.long, device=device)
        self.position_ids = torch.zeros(n_seqs, width, dtype=torch.long, device=device)
        self.logits_positions = torch.zeros(n_seqs, dtype=torch.long, device=device)

        info = AttentionMetadata()
        info.kv_buffer = kv_buffer
        info.b_req_tokens_table = b_req_tokens_table
        info.b_req_idx = torch.zeros(n_seqs, dtype=torch.long, device=device)
        info.b_seq_len = torch.zeros(n_seqs, dtype=torch.long, device=device)
        info.cur_select_index = torch.zeros(
            n_seqs * width, dtype=torch.int32, device=device
        )
        info.b_start_loc = (
            torch.arange(n_seqs, dtype=torch.int32, device=device) * width
        )
        info.is_prefill = True
        info.max_actual_seq_len = width
        info.b_prefix_len = torch.zeros(n_seqs, dtype=torch.long, device=device)
        info.b_kv_base = torch.zeros(
            n_seqs, dtype=b_req_tokens_table.dtype, device=device
        )
        info.max_chunk_len = width
        self.atten_info = info
        # Null-block rows a padding row may scribble on (tiled past its size).
        self._null_rows = (
            torch.arange(width, dtype=torch.int32, device=device)
            % PREFIX_CACHE_BLOCK_SIZE
        )

        self._graph: torch.cuda.CUDAGraph | None = None
        self._output: torch.Tensor | None = None

    def _load(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        logits_positions: torch.Tensor,
        src: AttentionMetadata,
    ) -> None:
        """Copy a live step's inputs into the persistent buffers, padding rows."""
        n_real = input_ids.shape[0]
        n, width = self.n_seqs, self.width
        info = self.atten_info
        if n_real == n:
            self.input_ids.copy_(input_ids)
            self.position_ids.copy_(position_ids)
            self.logits_positions.copy_(logits_positions)
            info.b_req_idx.copy_(src.b_req_idx)
            info.b_seq_len.copy_(src.b_seq_len)
            info.cur_select_index.copy_(src.cur_select_index)
            if src.b_prefix_len is not None:
                info.b_prefix_len.copy_(src.b_prefix_len)
            else:
                info.b_prefix_len.zero_()
        else:
            self.input_ids.zero_()
            self.input_ids[:n_real].copy_(input_ids)
            self.position_ids.zero_()
            self.position_ids[:n_real].copy_(position_ids)
            self.logits_positions.zero_()
            self.logits_positions[:n_real].copy_(logits_positions)
            info.b_req_idx.zero_()
            info.b_req_idx[:n_real].copy_(src.b_req_idx)
            # Padding rows attend a single null-block row and are discarded.
            info.b_seq_len.fill_(1)
            info.b_seq_len[:n_real].copy_(src.b_seq_len)
            info.b_prefix_len.zero_()
            if src.b_prefix_len is not None:
                info.b_prefix_len[:n_real].copy_(src.b_prefix_len)
            cur2d = info.cur_select_index.view(n, width)
            cur2d[:n_real].copy_(src.cur_select_index.view(n_real, width))
            cur2d[n_real:].copy_(self._null_rows.expand(n - n_real, width))
        # ``b_kv_base`` only feeds the chunked kernel; padding rows may read the
        # null block. Slot 0's row 0 belongs to a live request, so it is never
        # a safe base — but nothing written by these rows is ever read back.
        info.b_kv_base.copy_(info.b_req_tokens_table[info.b_req_idx, 0])

    def _step(self) -> torch.Tensor:
        return self.model(
            self.input_ids,
            self.position_ids,
            self.atten_info,
            logits_positions=self.logits_positions,
        )

    def capture(self) -> None:
        """Warm up on a side stream, then record. Inputs are already loaded."""
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        warmup_collectives()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._output = self._step()

    def replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        logits_positions: torch.Tensor,
        src: AttentionMetadata,
    ) -> torch.Tensor:
        """Load the live step's values and replay; returns ``[n_seqs, vocab]``."""
        if self._graph is None or self._output is None:
            raise RuntimeError("capture() must be called before replay()")
        self._load(input_ids, position_ids, logits_positions, src)
        self._graph.replay()
        return self._output


class PrefillGraphManager:
    """Holds one :class:`PrefillGraphRunner` per ``(n_bucket, width)`` grid.

    Lazily captured: a grid whose width has no bucket (or whose capture
    failed) runs eager, so the feature cannot regress a shape it does not
    recognise. Every capture is gated on a graph-vs-eager logit comparison
    against the live step that triggered it (both run; the eager result is
    served when they disagree), and under TP the capture/keep decision is
    reconciled across ranks before the graph may serve — a rank replaying
    while its peer runs eager wedges the group at the first collective.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        kv_buffer: list[torch.Tensor],
        b_req_tokens_table: torch.Tensor,
        n_buckets: tuple[int, ...] = DEFAULT_BATCH_SIZES,
        width_buckets: tuple[int, ...] = PREFILL_WIDTH_BUCKETS,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.kv_buffer = kv_buffer
        self.b_req_tokens_table = b_req_tokens_table
        self.n_buckets = tuple(sorted(set(n_buckets)))
        self.width_buckets = frozenset(width_buckets)
        self.device = device
        self._runners: dict[tuple[int, int], PrefillGraphRunner] = {}
        self._failed: set[tuple[int, int]] = set()
        self.replays: int = 0

    def __len__(self) -> int:
        return len(self._runners)

    def discard(self) -> None:
        self._runners.clear()
        torch.cuda.empty_cache()

    def _pad_n(self, n: int) -> int | None:
        for bucket in self.n_buckets:
            if bucket >= n:
                return bucket
        return None

    def try_replay(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        logits_positions: torch.Tensor,
        atten_info: AttentionMetadata,
    ) -> torch.Tensor | None:
        """Replay the matching prefill graph, or ``None`` to stay eager.

        On a first-seen shape this captures and validates the graph inline
        (one-off cost), serving the step from the eager reference pass that
        validated it. The decision reads only the broadcast plan's shapes, so
        every TP rank takes the same branch structurally.
        """
        n_real, width = input_ids.shape
        n_bucket = self._pad_n(n_real)
        if n_bucket is None or width not in self.width_buckets:
            return None
        max_tokens = (
            PREFILL_GRAPH_MAX_TOKENS_EP if expert_parallel_enabled() else PREFILL_GRAPH_MAX_TOKENS
        )
        if n_bucket * width > max_tokens:
            return None
        key = (n_bucket, width)
        if key in self._failed:
            return None
        runner = self._runners.get(key)
        if runner is None:
            return self._capture_and_run(
                key, input_ids, position_ids, logits_positions, atten_info
            )
        self.replays += 1
        out = runner.replay(input_ids, position_ids, logits_positions, atten_info)
        # The grid may have been padded up to ``n_bucket``; the pad rows'
        # logits are garbage (their K/V went to the null block), so drop them
        # before the caller's row-indexed sampling sees them.
        return out[:n_real] if runner.n_seqs != n_real else out

    def _capture_and_run(
        self,
        key: tuple[int, int],
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        logits_positions: torch.Tensor,
        atten_info: AttentionMetadata,
    ) -> torch.Tensor | None:
        """Capture ``key`` against the live step, gate on parity, serve the step.

        The warmup, the capture and the parity replay all write the very
        cache rows this pass is about to write, with identical values, and the
        eager reference pass — the same padded grid, run unrecorded — writes
        them a final time, which is what makes the comparison mean anything.
        The step is served from that reference either way; parity decides only
        whether the graph may serve later steps (a failure blacklists the
        shape). ``None`` means capture itself failed (the caller runs its own
        eager pass on the live metadata).
        """
        n_bucket, width = key
        free, _ = torch.cuda.mem_get_info()
        if free < _PREFILL_GRAPH_MIN_FREE:
            # Never capture into a full card: the graph pool would crowd out
            # the headroom routine step allocations need. Blacklist the shape
            # (memory pressure only grows from here) and stay eager.
            logger.info(
                "Prefill graph n=%d width=%d refused: only %.2f GiB free (need %.2f)",
                n_bucket,
                width,
                free / 2**30,
                _PREFILL_GRAPH_MIN_FREE / 2**30,
            )
            self._failed.add(key)
            return None
        runner: PrefillGraphRunner | None = None
        try:
            # Capture must be the only work in flight: the pending uploads and
            # the previous step's tail have to land first.
            torch.cuda.synchronize()
            runner = PrefillGraphRunner(
                self.model,
                n_seqs=n_bucket,
                width=width,
                kv_buffer=self.kv_buffer,
                b_req_tokens_table=self.b_req_tokens_table,
                device=self.device,
            )
            runner._load(input_ids, position_ids, logits_positions, atten_info)
            runner.capture()
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError) and "out of memory" not in str(
                exc
            ).lower():
                raise
            logger.warning(
                "Lazy capture of prefill graph n=%d width=%d ran out of memory; "
                "that shape stays eager",
                n_bucket,
                width,
            )
            runner = None

        if get_tensor_model_parallel_world_size() > 1 and not tensor_model_parallel_ranks_agree(
            int(runner is not None)
        ):
            # A peer failed to capture: retire this rank's graph too, or the
            # first replayed collective waits on a peer that never issues it.
            runner = None
            torch.cuda.empty_cache()

        if runner is None:
            self._failed.add(key)
            return None

        replayed = runner.replay(input_ids, position_ids, logits_positions, atten_info)
        # Parity reference: the very grid the graph recorded, run eagerly. A
        # live n_real-row pass is the wrong yardstick — its narrower GEMMs take
        # different reduction orders and differ at bf16 rounding level, which
        # says nothing about whether the graph replays its own stream faithfully.
        reference = runner._step()
        n_real = input_ids.shape[0]
        error = (replayed.float() - reference.float()).abs().max().item()
        keep = error <= TP_GRAPH_PARITY_ATOL
        if get_tensor_model_parallel_world_size() > 1:
            keep = tensor_model_parallel_all_reduce_min(int(keep)) == 1
        # The eager reference pass ran last and wrote the same cache rows the
        # replay did, and its storage aliases nothing the graph holds — serve it
        # either way; ``keep`` only decides whether the graph may serve later.
        served = reference if runner.n_seqs == n_real else reference[:n_real]
        if not keep:
            logger.warning(
                "Prefill graph n=%d width=%d disagrees with its eager grid by %.3e "
                "(tolerance %.1e); that shape stays eager",
                n_bucket,
                width,
                error,
                TP_GRAPH_PARITY_ATOL,
            )
            self._failed.add(key)
            return served
        self._runners[key] = runner
        self.replays += 1
        logger.info(
            "Lazy-captured prefill graph n=%d width=%d (parity %.2e; %d shapes held)",
            n_bucket,
            width,
            error,
            len(self._runners),
        )
        return served
