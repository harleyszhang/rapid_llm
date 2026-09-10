"""Model worker: runs one :class:`ModelInput` — forward *and* sampling — per rank.

:class:`ModelWorker` turns a plan into launch-ready rows (prefill grid,
extend or padded decode), runs the forward pass, samples the new tokens,
and applies the TP collectives so every rank agrees on the result.

Usage:
    worker = ModelWorker(engine, max_num_seqs, max_seq_len)
    tokens, logprobs = worker.execute(model_input)
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import torch

from ..batch_overlap.overlap import OverlapPolicy, StreamPool, Timeline
from ..batch_overlap.two_batch_overlap import tbo_policy
from ..distributed.parallel_state import (
    expert_parallel_enabled,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_broadcast,
)
from ..engine.sampler import (
    BatchedSamplingParams,
    GeneratedSpan,
    PositionLogprobs,
    SamplingParams,
    rows_logprobs,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from ..engine.llm_engine import LLMEngine

#: Environment variable enabling the launch/harvest engine pipeline (O2): the
#: engine plans and launches step N+1 while step N is still on the GPU, and
#: harvests N's tokens one step late. Off by default — it deliberately delays
#: stop handling by one token, so it is a deployment choice, not a default.
PIPELINE_ENV = "RAPID_LLM_PIPELINE"


def pipeline_enabled() -> bool:
    """Read ``RAPID_LLM_PIPELINE``; only ``1``/``true``/``on`` means on."""
    raw = os.environ.get(PIPELINE_ENV, "0").strip().lower()
    return raw in ("1", "true", "on")


class PassKind(StrEnum):
    """Which kernel path a plan takes through attention.

    PREFILL: padded ``[sequences, width]`` grid through a prefill kernel.
    EXTEND: one decode-style row per token over the slot's cached history.
    DECODE: one row per sequence, one token each.
    """

    PREFILL = "prefill"
    EXTEND = "extend"
    DECODE = "decode"


@dataclass(frozen=True)
class ModelInput:
    """One forward pass's plan. Fields are tuples/immutable, so picklable and
    broadcastable to TP workers. Sequence ``i`` covers cache span
    ``[seq_starts[i], seq_lens[i])``; ``tokens`` is those spans concatenated.

    Attributes:
        kind: Kernel path (see :class:`PassKind`).
        slots / seq_starts / seq_lens: Per-sequence cache slot, first written
            row, and total cached length after this pass.
        tokens: Input ids, all chunks concatenated (padding added by the worker).
        sampling / sampled / gen_counts: Parallel to the sampled subset;
            ``gen_counts`` doubles as the write column and penalty-window width.
        block_writes: ``(slot, group_id, start_block, block_ids)`` block-table
            entries installed before the pass; travel with the plan so every TP
            rank applies them to its own table.
        prompt_logprobs: Per-sequence prompt-scoring top-k, parallel to
            ``slots``; ``None`` where not asked.
        prompt_targets: Token id each input row is scored against, parallel to
            ``tokens`` (the sampled row's entry is unused).
    """

    kind: PassKind
    slots: tuple[int, ...]
    seq_starts: tuple[int, ...]
    seq_lens: tuple[int, ...]
    tokens: tuple[int, ...]
    sampling: tuple[SamplingParams, ...]
    sampled: tuple[int, ...]
    gen_counts: tuple[int, ...]
    block_writes: tuple[tuple[int, int, int, tuple[int, ...]], ...] = ()
    prompt_logprobs: tuple[int | None, ...] = ()
    prompt_targets: tuple[int, ...] = ()
    # O5 speculative decoding: when True, execute() returns the full logits
    # tensor alongside sampled tokens. The engine uses these for per-position
    # draft verification. Default False keeps the normal path lean.
    return_logits: bool = False

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("a ModelInput must cover at least one sequence")
        if not len(self.slots) == len(self.seq_starts) == len(self.seq_lens):
            raise ValueError("slots, seq_starts and seq_lens must describe the same sequences")
        if len(self.tokens) != sum(self.chunk_lens):
            raise ValueError(f"got {len(self.tokens)} tokens for {sum(self.chunk_lens)} cache rows")
        if not len(self.sampled) == len(self.sampling) == len(self.gen_counts):
            raise ValueError("sampled, sampling and gen_counts must describe the same rows")
        if self.prompt_logprobs and len(self.prompt_logprobs) != len(self.slots):
            raise ValueError("prompt_logprobs must be empty or parallel to slots")
        if self.prompt_targets and len(self.prompt_targets) != len(self.tokens):
            raise ValueError("prompt_targets must be empty or parallel to tokens")
        if any(k is not None for k in self.prompt_logprobs) and not self.prompt_targets:
            raise ValueError("prompt_logprobs needs prompt_targets to score against")

    @property
    def chunk_lens(self) -> tuple[int, ...]:
        """Tokens this pass feeds per sequence."""
        return tuple(end - start for start, end in zip(self.seq_starts, self.seq_lens, strict=True))


@dataclass(frozen=True)
class PassLogprobs:
    """Logprob records one pass produced, beside the tokens it drew.

    Attributes:
        sampled: Per ``ModelInput.sampled`` row — the distribution its token was
            drawn from, ``None`` where not asked.
        prompt: Per ``ModelInput.slots`` sequence — chunk positions scored
            against the prompt's tokens, ``None`` where not asked. The sampled
            row appears in ``sampled``, not here.
    """

    sampled: tuple[PositionLogprobs | None, ...] = ()
    prompt: tuple[tuple[PositionLogprobs, ...] | None, ...] = ()


@dataclass
class _PreparedPass:
    """One pass's device-bound inputs, built on the host and uploaded.

    With the overlap policy on the upload rides the copy stream and ``event``
    marks its completion; with it off the upload is inline and ``event`` is
    ``None``. The tensors fed to the model are identical either way.

    Attributes:
        input_ids: ``[n, width]`` prefill grid, or ``[padded, 1]`` extend/decode
            rows (filler included).
        positions: Absolute positions, prefill only (extend/decode derive from
            slot metadata).
        logits_positions: Per-row next-token logits gather index, prefill only.
        event: Completion of this pass's copies; ``None`` when inline.
        padded: Rows submitted, extend/decode only (trailing filler discarded).
    """

    input_ids: torch.Tensor
    event: torch.cuda.Event | None
    positions: torch.Tensor | None = None
    logits_positions: torch.Tensor | None = None
    padded: int = 0


class ModelWorker:
    """Executes plans against this rank's model shard and KV cache.

    Holds the mutable execution state — the fixed-slot KV view and the
    ``[num_slots, max_seq_len]`` generated-token grid the repetition penalty
    reads — both indexed by slot, so nothing is invalidated when the running set
    changes. With the pipeline on it also keeps each slot's next decode input on
    the device (``_next_tokens``), so tokens never round-trip through the host.

    Args:
        engine: Built :class:`~rapid_llm.engine.llm_engine.LLMEngine`; the
            worker takes its KV cache via the slot view.
        max_num_seqs: Concurrency ceiling; caps slots and the grid height.
        max_seq_len: Context bound; the grid width.
        pipeline: Decode inputs from the device-side next-token grid. ``None``
            reads :data:`PIPELINE_ENV` (how a TP follower learns the driver's
            choice).
    """

    def __init__(
        self,
        engine: LLMEngine,
        max_num_seqs: int,
        max_seq_len: int,
        *,
        pipeline: bool | None = None,
    ) -> None:
        self._runner = engine.model_runner
        self._sampler = engine.sampler
        self._device = engine.device
        self._pad_id = engine.pad_id
        self._slot_batch = self._runner.enable_slot_kv_cache()
        self._pipeline = pipeline_enabled() if pipeline is None else pipeline

        # Slot ids cap at max_num_seqs, keeping the grid proportional to the
        # requested concurrency, not to however many slots fit the cache.
        self.num_slots = min(self._slot_batch.num_slots, max_num_seqs)
        # Cache rows in scheduler block units: the block pool admits requests,
        # so it is sized by real memory, not the table's geometry.
        self.num_kv_blocks = (
            self._runner.kv_cache_manager.gpu_num_blocks // self._slot_batch.block_size
        )
        self._gen_grid = torch.zeros(
            (self.num_slots, max_seq_len), dtype=torch.long, device=self._device
        )
        self._columns = torch.arange(max_seq_len, device=self._device)
        self._no_tokens = torch.empty(0, dtype=torch.long, device=self._device)
        # Pipeline feedback lane: each slot's last sampled token, on-device so
        # the next decode pass reads it without a host round-trip.
        self._next_tokens = torch.full(
            (self.num_slots,), self._pad_id, dtype=torch.long, device=self._device
        )

        # Sampling knobs cost four uploads to rebuild; a steady batch reuses
        # them, keyed on the plan's own rows.
        self._sampling_key: tuple[tuple[float, float, float, bool, int | None], ...] | None = None
        self._sampling: BatchedSamplingParams | None = None

        # L1 cross-stream overlap: uploads ride a copy stream so the host never
        # stalls on a pageable H2D; off on non-CUDA (inline blocking upload).
        on_cuda = torch.device(self._device).type == "cuda"
        self._policy = OverlapPolicy.from_env() if on_cuda else OverlapPolicy(enabled=False)
        self.timeline = Timeline.from_env(str(self._device)) if on_cuda else Timeline(enabled=False)
        self._pool = StreamPool(str(self._device), self._policy, self.timeline)

    @torch.inference_mode()
    def execute(self, model_input: ModelInput) -> tuple[torch.Tensor, PassLogprobs | None]:
        """Run one pass; return sampled tokens and any logprob records.

        Returns:
            ``(tokens, records)``: token ids identical across TP ranks, and the
            :class:`PassLogprobs` (``None`` when nobody asked). A pass whose
            sequences all still owe tokens runs the model for its K/V and
            returns an empty tensor.
        """
        tokens, records, _ = self._execute_inner(model_input)
        return tokens, records

    def execute_verify(
        self, model_input: ModelInput
    ) -> tuple[torch.Tensor, PassLogprobs | None, torch.Tensor | None]:
        """O5 speculative decoding: run a verify pass, return tokens + logits.

        Like :meth:`execute` but also returns the full ``[rows, vocab]`` logits
        tensor (one row per input token, in sequence order). The engine uses
        these for per-position draft verification. Returns ``None`` for the
        logits when the model produced no output (all sequences owe tokens).
        """
        return self._execute_inner(model_input)

    def _execute_inner(
        self, model_input: ModelInput
    ) -> tuple[torch.Tensor, PassLogprobs | None, torch.Tensor | None]:
        prepared = self.prepare(model_input)
        # Install block tables before the forward (so rows have pages) but after
        # prepare (which only gathered entries for rows the plan names).
        self._slot_batch.write_block_tables(model_input.block_writes)
        logits, prompt = self._forward(model_input, prepared)
        all_logits: torch.Tensor | None = None
        if model_input.return_logits and logits is not None:
            # logits here is the sampled-row slice; for verify we need the
            # full [rows, vocab] tensor. _forward already computed it; the
            # caller set return_logits so _forward returned the full thing.
            all_logits = logits
        sampled_records = None
        if logits is None:
            tokens = self._no_tokens
        else:
            # When return_logits is set, _forward returns the full [rows, vocab]
            # tensor instead of the sampled-row slice. Sample from the last row
            # per sequence (the normal path for EXTEND).
            if model_input.return_logits and all_logits is not None:
                # Re-derive the sampled-row logits for _sample.
                # all_logits is [n_tokens, vocab] (2D); index with a list
                # (not a tuple, which would be multi-dim indexing).
                import itertools as _it
                ends = list(_it.accumulate(model_input.chunk_lens))
                row_list = [ends[index] - 1 for index in model_input.sampled]
                sample_logits = all_logits[row_list] if row_list else all_logits[:0]
                tokens, sampled_records = self._sample(model_input, sample_logits)
            else:
                tokens, sampled_records = self._sample(model_input, logits)
        if not any(prompt) and sampled_records is None:
            return tokens, None, all_logits
        return tokens, PassLogprobs(
            sampled=tuple(sampled_records) if sampled_records is not None else (),
            prompt=prompt,
        ), all_logits

    # -------------------------------------------------------------- preparing #
    def prepare(self, plan: ModelInput) -> _PreparedPass:
        """Build one pass's inputs on the host and start their upload.

        Layout arithmetic (row expansion, graph padding) runs on the host; the
        upload leaves on the copy stream. The returned event is what the forward
        consumes before launching kernels (``None`` when the policy is off).
        """
        match plan.kind:
            case PassKind.PREFILL:
                return self._prepare_grid(plan)
            case PassKind.EXTEND:
                return self._prepare_rows(plan, PassKind.EXTEND)
            case PassKind.DECODE:
                return self._prepare_rows(plan, PassKind.DECODE)

    def _prepare_grid(self, plan: ModelInput) -> _PreparedPass:
        """Pad the prompt chunks into a grid and upload it with its positions."""
        chunk_lens = plan.chunk_lens
        width = max(chunk_lens)
        grid, offset = [], 0
        for chunk in chunk_lens:
            grid.append(plan.tokens[offset : offset + chunk] + (self._pad_id,) * (width - chunk))
            offset += chunk
        # Padded columns run past a row's real position, but attention stops at
        # b_seq_len, so the junk positions are inert.
        positions = [list(range(start, start + width)) for start in plan.seq_starts]

        input_ids, _ = self._pool.upload_async(
            grid, dtype=torch.long, label="upload.prefill.tokens"
        )
        positions_t, _ = self._pool.upload_async(
            positions, dtype=torch.long, label="upload.prefill.positions"
        )
        # A row's next-token logits sit at its last real column.
        logits_pos, event = self._pool.upload_async(
            [chunk - 1 for chunk in chunk_lens], dtype=torch.long, label="upload.prefill.logits"
        )
        return _PreparedPass(
            input_ids=input_ids,
            event=event,
            positions=positions_t,
            logits_positions=logits_pos,
        )

    def _prepare_rows(self, plan: ModelInput, kind: PassKind) -> _PreparedPass:
        """Upload one-token rows padded onto the captured graph width.

        Extend and decode share the shape (one row per token, filler to the graph
        width); they differ only in the planning helper. Filler rows exist to
        reach a captured batch size and are discarded with their logits.
        """
        if kind is PassKind.EXTEND:
            rows_slot, _ = self._slot_batch.plan_extend_rows(
                plan.slots, plan.seq_starts, plan.seq_lens
            )
            padded = len(rows_slot)
        else:
            padded_slots, _ = self._slot_batch.pad_decode_rows(plan.slots, plan.seq_lens)
            padded = len(padded_slots)

        if kind is PassKind.DECODE and self._pipeline:
            # The launch/harvest pipeline: the plan's token entries are
            # placeholders (``-1``), because the engine has not harvested the
            # tokens it would name. Gather the real inputs off the device grid
            # the last pass's sampler wrote, pad with inert ids up to the graph
            # width, and skip the upload entirely — there is nothing to upload.
            #
            # The gather index rides the pool's pinned async upload: a pageable
            # ``torch.tensor(..., device=...)`` would synchronise on the copy,
            # and the copy sits behind the previous step's still-running replay
            # — the host would wait a whole forward out per step, exactly the
            # serialisation the pipeline exists to remove.
            real = len(plan.slots)
            slots_idx, slot_event = self._pool.upload_async(
                plan.slots, dtype=torch.long, label="upload.decode.slots"
            )
            self._pool.consume(slot_event, slots_idx)
            gathered = self._next_tokens[slots_idx]
            pad = padded - real
            if pad > 0:
                gathered = torch.cat(
                    [
                        gathered,
                        torch.full((pad,), self._pad_id, dtype=torch.long, device=self._device),
                    ]
                )
            return _PreparedPass(input_ids=gathered.view(padded, 1), event=None, padded=padded)

        rows = plan.tokens + (self._pad_id,) * (padded - len(plan.tokens))
        input_ids, event = self._pool.upload_async(
            rows, dtype=torch.long, label=f"upload.{kind}.tokens"
        )
        return _PreparedPass(input_ids=input_ids.view(padded, 1), event=event, padded=padded)

    # ---------------------------------------------------------------- forwards #
    def _forward(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """The pass's sampled-row logits and its per-sequence prompt records.

        The second element is parallel to ``plan.slots`` and all-``None`` for a
        pass nobody asked prompt logprobs of — decode passes always, chunk
        passes by request.
        """
        match plan.kind:
            case PassKind.PREFILL:
                return self._forward_grid(plan, prepared)
            case PassKind.EXTEND:
                return self._forward_extend(plan, prepared)
            case PassKind.DECODE:
                return self._forward_decode(plan, prepared)

    def _forward_grid(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """A padded token grid through the prefill kernel."""
        self._slot_batch.begin_prefill(plan.slots, plan.seq_starts, plan.seq_lens)
        # Grid, positions and gather index all left on the copy stream; one
        # stream-ordered wait covers them.
        self._pool.consume(
            prepared.event, prepared.input_ids, prepared.positions, prepared.logits_positions
        )
        wants_prompt = any(k is not None for k in plan.prompt_logprobs)
        if not wants_prompt:
            # A captured grid replays the whole prefill for a few buffer
            # copies; the eager pass below pays its ~50 ms of host launches on
            # every first token. Prompt-logprob passes need the full-grid
            # logits a capture does not produce, so they stay eager.
            replayed = self._runner.try_replay_prefill(
                prepared.input_ids, prepared.positions, prepared.logits_positions
            )
            if replayed is not None:
                return self._pick(replayed, plan.sampled, len(plan.slots)), (None,) * len(
                    plan.slots
                )
        with self.timeline.region("forward.prefill", "compute"):
            # Prompt scoring needs every column's logits, so the gather is
            # skipped and lm_head projects the whole grid (paid in GEMM width).
            logits = self._runner.forward(
                prepared.input_ids,
                prepared.positions,
                None,
                logits_positions=None if wants_prompt else prepared.logits_positions,
            )
        if not wants_prompt:
            return self._pick(logits, plan.sampled, len(plan.slots)), (None,) * len(plan.slots)
        prompt = self._prompt_records(plan, logits, grid=True)
        rows = torch.arange(len(plan.slots), device=logits.device)
        gathered = logits[rows, prepared.logits_positions]
        return self._pick(gathered, plan.sampled, len(plan.slots)), prompt

    def _forward_extend(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """Chunks resuming on a cached prefix: one decode-style row per token."""
        padded = self._slot_batch.begin_extend(plan.slots, plan.seq_starts, plan.seq_lens)
        self._pool.consume(prepared.event, prepared.input_ids)
        # begin_extend set b_seq_len to each row's position + 1; the fed token's
        # position is that minus one.
        positions = (self._slot_batch.seq_lens - 1).view(-1, 1)

        with self.timeline.region("forward.extend", "compute"):
            # No logits_positions: every row is projected anyway, so prompt
            # scoring is free — a sequence's rows are its prompt distributions.
            logits = self._runner.forward(prepared.input_ids, positions, None)
        # A sequence's next-token logits sit on the last row of its stretch.
        flat = logits[:, -1, :]
        prompt = (None,) * len(plan.slots)
        if any(k is not None for k in plan.prompt_logprobs):
            prompt = self._prompt_records(plan, flat, grid=False)
        # O5 speculative decoding: return the full [num_tokens, vocab] logits
        # for per-position verification instead of the sampled-row slice.
        if plan.return_logits:
            n_real = len(plan.tokens)
            return flat[:n_real], prompt
        ends = list(itertools.accumulate(plan.chunk_lens))
        rows = tuple(ends[index] - 1 for index in plan.sampled)
        return self._pick(flat, rows, padded), prompt

    def _forward_decode(
        self, plan: ModelInput, prepared: _PreparedPass
    ) -> tuple[torch.Tensor | None, tuple[tuple[PositionLogprobs, ...] | None, ...]]:
        """One token for every sequence in the plan.

        With the L2 policy active (tensor parallelism, enough rows), the step
        runs two-batch overlapped through the batch_overlap entry:
        ``begin_decode`` still installs the metadata for the *whole* step
        first, then the overlapped arm splits it into halves whose deferred
        all-reduces ping-pong with each other's compute. Same logits shape,
        same row order, same downstream sampling -- the split is invisible
        past this method.

        A matching CUDA graph replays its captured policy. A graph miss keeps
        this eager policy, so sparse capture grids do not disable overlap.
        """
        rows = len(plan.slots)
        self._slot_batch.begin_decode(plan.slots, plan.seq_lens)
        self._pool.consume(prepared.event, prepared.input_ids)
        # The fed token sits at its cache row: length minus one.
        positions = self._slot_batch.seq_lens.view(-1, 1) - 1

        with self.timeline.region("forward.decode", "compute"):
            # Replay wins when this shape has a graph. A graph miss stays on
            # the TBO-capable eager path instead of disabling overlap for every
            # shape merely because some other graph exists.
            logits = self._runner.forward_maybe_tbo(
                prepared.input_ids,
                positions,
                enable_tbo=tbo_policy().active(
                    world_size=get_tensor_model_parallel_world_size(),
                    rows=rows,
                    graph_active=False,
                    expert_parallel=expert_parallel_enabled(),
                ),
            )
        # Decode never has prompt positions to score: they were all covered
        # during prefill, so the second element is uniformly empty.
        return self._pick(logits[:rows, -1, :], plan.sampled, rows), (None,) * len(plan.slots)

    # ---------------------------------------------------------------- sampling #
    def _sample(
        self, plan: ModelInput, logits: torch.Tensor
    ) -> tuple[torch.Tensor, list[PositionLogprobs | None] | None]:
        """Draw one token per sampled row and record it in the generated grid.

        The second return is the per-row logprob records, ``None`` when nobody
        asked.
        """
        if not plan.sampled:
            return torch.empty(0, dtype=torch.long, device=logits.device), None

        sampling = self._batched_sampling(plan.sampling)
        # Both index uploads ride the pinned async lane for the same reason as
        # the pipeline's gather index: a pageable H2D would synchronise behind
        # whatever the queue still holds, serialising the host against the GPU
        # mid-pass. They are tiny, but tiny and blocking still blocks.
        slots, slots_event = self._pool.upload_async(
            [plan.slots[index] for index in plan.sampled], dtype=torch.long,
            label="upload.sample.slots",
        )
        # Where each row's new token goes, which is also how much history its
        # repetition penalty may look at.
        columns, columns_event = self._pool.upload_async(
            plan.gen_counts, dtype=torch.long, label="upload.sample.columns"
        )
        self._pool.consume(slots_event, slots)
        self._pool.consume(columns_event, columns)

        generated = None
        width = max(plan.gen_counts)
        if sampling.any_penalty and width:
            span = self._columns[:width].unsqueeze(0)
            generated = GeneratedSpan(
                self._gen_grid[slots.unsqueeze(1), span], span < columns.unsqueeze(1)
            )

        ids, records = self._sampler.sample_batched_with_logprobs(logits, sampling, generated)
        # Every TP rank must hold the same ids: non-greedy sampling draws from a
        # per-rank RNG, so the broadcast keeps ranks from diverging.
        tokens = tensor_model_parallel_broadcast(ids.reshape(-1))
        self._gen_grid[slots, columns] = tokens
        if self._pipeline:
            # Device-side feedback: the next pass reads this slot's input here.
            self._next_tokens[slots] = tokens
        return tokens, records

    def readback(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """Stage a pass's sampled tokens for the host, without waiting.

        The D2H copy rides the copy stream; the caller syncs on the returned
        event one step later. With the policy off (or on CPU) it is a plain
        ``.cpu()`` and the event is ``None``.
        """
        return self._pool.readback_async(tokens, label="readback.tokens")

    def release_readback(self, host: torch.Tensor) -> None:
        """Hand a staged token buffer back once the host has read it.

        See :meth:`~rapid_llm.batch_overlap.overlap.StreamPool.release_readback`: the
        buffer cannot rejoin the ring on its copy event alone, because the next
        pass's copy is issued before this pass's tokens are harvested.
        """
        self._pool.release_readback(host)

    def _batched_sampling(self, params: tuple[SamplingParams, ...]) -> BatchedSamplingParams:
        """Device-side sampling knobs, rebuilt only when the sampled rows change.

        A steady decode batch reuses them; the key snapshots the five values
        feeding the tensors, so an in-place parameter change cannot leave stale
        device values.
        """
        # SamplingParams is mutable; cache by value snapshot, not object equality
        # (an in-place mutation would otherwise compare equal and stay stale).
        key = tuple(
            (p.temperature, p.top_p, p.repetition_penalty, p.is_greedy, p.logprobs) for p in params
        )
        if key != self._sampling_key:
            self._sampling = BatchedSamplingParams.build(params, self._device)
            self._sampling_key = key
        return self._sampling  # type: ignore[return-value]

    # --------------------------------------------------------------- internals #
    def _prompt_records(
        self, plan: ModelInput, logits: torch.Tensor, *, grid: bool
    ) -> tuple[tuple[PositionLogprobs, ...] | None, ...]:
        """Score each asking sequence's chunk against the prompt's own tokens.

        ``grid`` selects the logits layout (prefill ``[n, width, V]`` vs extend
        ``[total_tokens, V]``). Row ``j`` of sequence ``i`` predicted position
        ``seq_starts[i] + j + 1``, scored against ``prompt_targets`` at the same
        offset; the sampled row is excluded (the sampler records it).
        """
        records: list[tuple[PositionLogprobs, ...] | None] = []
        sampled = set(plan.sampled)
        offset = 0
        for index, k in enumerate(plan.prompt_logprobs):
            chunk = plan.chunk_lens[index]
            if k is None:
                records.append(None)
            else:
                rows = chunk - 1 if index in sampled else chunk
                if rows <= 0:
                    # A one-token final chunk covers no prompt position.
                    records.append(())
                else:
                    rows_t = logits[index, :rows] if grid else logits[offset : offset + rows]
                    targets = self._to_device(plan.prompt_targets[offset : offset + rows])
                    records.append(tuple(rows_logprobs(rows_t, targets, k)))
            offset += chunk
        return tuple(records)

    def _pick(self, logits: torch.Tensor, rows: tuple[int, ...], total: int) -> torch.Tensor | None:
        """Narrow logits to the sampled rows; ``None`` when none (the pass ran
        for its K/V alone).
        """
        if not rows:
            return None
        if len(rows) == total:
            return logits
        return logits[self._to_device(rows)]

    def _to_device(self, values: Sequence[int]) -> torch.Tensor:
        """Upload a host index list as a fresh int64 tensor.

        A new allocation, not a reused staging buffer: the previous step's tensor
        may still be queued, and overwriting it would race with pending kernels.
        """
        return torch.tensor(values, dtype=torch.long, device=self._device)
