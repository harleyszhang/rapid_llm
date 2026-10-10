"""Unified generation engine: one-shot batches over the executor.

:class:`LLMEngine` owns the tokenizer, model runner and executor end to
end; ``generate`` tokenises prompts, runs whole-batch passes until every
request finishes, then detokenises the results.

Usage:
    engine = LLMEngine(checkpoints_dir, tokenizer_path)
    outputs = engine.generate(prompt_token_ids, params)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from transformers import AutoTokenizer

from ..distributed.parallel_state import (
    get_parallel_topology,
    tensor_model_parallel_broadcast,
)
from ..executor.model_runner import ModelRunner
from ..utils.logger import get_logger
from ..utils.path_utils import get_model_name_from_path
from .detokenizer import IncrementalDetokenizer
from .sampler import GeneratedSpan, PositionLogprobs, Sampler, SamplingParams, rows_logprobs
from .stop_criteria import (
    POLL_INTERVAL,
    StopCriteria,
    detect_repetition,
    load_stop_token_ids,
)

_log = get_logger(__name__)


class _DecodeSession:
    """One ``generate()`` call: owns the token grid, KV reservations and stop state.

    Splitting this out of :class:`LLMEngine` keeps the ~15 pieces of per-request
    state (token grid, prompt masks, mrope deltas, stop flags, detokeniser) out of
    the engine, which is otherwise a long-lived, request-independent object.

    Args:
        engine: Owning engine, for the executor/tokenizer/sampler.
        prompt_token_ids: One token-id list per sequence.
        params: Sampling configuration.
        position_ids: Optional explicit positions (mrope path).
        multi_modal_inputs: Processor outputs, consumed during prefill only.
    """

    def __init__(
        self,
        engine: LLMEngine,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
        position_ids: torch.Tensor | None,
        multi_modal_inputs: dict[str, Any] | None,
    ) -> None:
        self._engine = engine
        self._params = params
        self._position_ids = position_ids
        self._multi_modal_inputs = multi_modal_inputs
        device = engine.device

        # Each generate() call is an independent batch: reset the paged-KV
        # allocator so a long-lived engine does not leak cache rows.
        engine.model_runner.kv_cache_manager.free_all()

        self.batch_size = len(prompt_token_ids)
        self._prompt_lens = [len(ids) for ids in prompt_token_ids]
        self._max_prompt_len = max(self._prompt_lens)
        if self._max_prompt_len > engine.max_seq_len:
            raise ValueError(
                f"prompt length {self._max_prompt_len} exceeds max_seq_len {engine.max_seq_len}"
            )

        max_gen_len = params.max_gen_len or (engine.max_seq_len - self._max_prompt_len)
        self._total_len = min(engine.max_seq_len, self._max_prompt_len + max_gen_len)

        pad_id = engine.pad_id
        self._tokens = torch.full(
            (self.batch_size, self._total_len), pad_id, dtype=torch.long, device=device
        )
        # One host-to-device copy for the whole batch instead of one per sequence.
        for i, ids in enumerate(prompt_token_ids):
            self._tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        # True where the position holds a real prompt token (never overwritten).
        self._prompt_mask = self._tokens != pad_id

        prompt_len_tensor = torch.tensor(self._prompt_lens, dtype=torch.long, device=device)
        # Deliberately a separate tensor from the one handed to the executor:
        # ``prefill_alloc_kv_cache`` keeps that one as ``atten_info.b_seq_len`` and
        # ``decode_alloc_kv_cache`` grows it in place every step. Aliasing the two
        # would silently turn the prompt lengths into running sequence lengths.
        self._prompt_lens_gpu = prompt_len_tensor.clone().unsqueeze(-1)  # [batch, 1]
        # Host mirror of ``atten_info.b_seq_len`` for the decode steps: prefill
        # leaves the GPU lengths at the prompt lengths and every
        # ``decode_alloc_kv_cache`` grows them in place by one, so this ledger
        # grows in lockstep on the host. It is what the runner's per-step
        # prepare hook (``KernelSpec.step_prepare``) reads to build attention
        # plans without a device sync.
        self._host_lens_t = torch.tensor(self._prompt_lens, dtype=torch.long)
        # Positions past each sequence's own prompt, i.e. the span the repetition
        # penalty may look at. Precomputed once so the hot loop only slices it.
        self._is_generated = (
            torch.arange(self._total_len, device=device).unsqueeze(0) >= self._prompt_lens_gpu
        )
        # Tokens the caller will actually see: written when a position is not part
        # of the prompt and the sequence had not already stopped.
        self._accepted = torch.zeros(
            (self.batch_size, self._total_len), dtype=torch.bool, device=device
        )

        self._stop = StopCriteria(
            self.batch_size,
            engine.stop_token_ids,
            engine.model_runner.vocab_size,
            device=device,
        )
        self._detokenizer = IncrementalDetokenizer(engine.tokenizer, self.batch_size)
        self.completions = [""] * self.batch_size
        # F6: per-sequence logprob records, collected only when asked for. The
        # records live on the host (sampler syncs them), so with logprobs on
        # every step synchronises — that is the feature's inherent cost.
        self.output_logprobs: list[list[PositionLogprobs]] | None = (
            [[] for _ in range(self.batch_size)] if params.logprobs is not None else None
        )
        self.prompt_logprobs: list[list[PositionLogprobs | None]] | None = None

        self._allocated = [
            engine.model_runner.prefill_alloc_kv_cache(
                self._max_prompt_len,
                prompt_len_tensor,
                torch.arange(self.batch_size, device=device),
            )
        ]
        self._position_deltas = self._mrope_position_deltas()

    def _mrope_position_deltas(self) -> torch.Tensor | None:
        """Per-sequence offset from token count to last mrope position.

        With mrope (Qwen3-VL), a vision block advances the position by only
        ``max(h, w)`` while contributing ``h*w`` tokens, so the last prompt
        position is *below* the token count. Decode must continue from that last
        mrope position, so the gap is computed once here and added to every decode
        step (mirrors HF's ``rope_deltas``).
        """
        position_ids = self._position_ids
        if position_ids is None or position_ids.ndim != 3:
            return None
        device = self._engine.device
        last_index = self._prompt_lens_gpu.view(-1) - 1
        # [3, batch] — position components of each sequence's last prompt token.
        last_positions = position_ids[:, torch.arange(self.batch_size, device=device), last_index]
        return last_positions.max(dim=0).values + 1 - self._prompt_lens_gpu.view(-1)

    def run(self, stream: bool) -> Iterator[list[str]]:
        """Drive prefill + decode.

        Args:
            stream: When ``True``, yield each step's text delta and therefore read
                the sampled tokens back every step. When ``False`` nothing is
                yielded and the host reads tokens back once per
                :data:`~rapid_llm.engine.stop_criteria.POLL_INTERVAL` steps,
                which keeps it off the critical path.

        Yields:
            One list of per-sequence text deltas per step (streaming only).
        """
        engine = self._engine
        params = self._params
        prev_pos = 0
        pending_from = self._max_prompt_len  # first column not yet read back

        for step, cur_pos in enumerate(range(self._max_prompt_len, self._total_len), start=1):
            is_prefill = prev_pos == 0
            if not is_prefill:
                # Publish the host ledger for this step's decode kernels: the
                # runner's prepare hook reads it instead of syncing the GPU
                # lengths. The tensor is mutated in place below, so the
                # reference stays valid for every later step.
                engine.model_runner.atten_info.b_seq_len_cpu = self._host_lens_t
            input_ids = self._tokens[:, prev_pos:cur_pos]
            step_positions = self._step_positions(input_ids, prev_pos, is_prefill)

            wants_prompt = is_prefill and params.prompt_logprobs is not None
            logits = engine.model_runner.forward(
                input_ids,
                step_positions,
                self._multi_modal_inputs if is_prefill else None,
                # Each sequence's next-token logits sit at its own last real
                # prompt position; asking the model to gather them before the
                # lm_head GEMM keeps the projection to one row per sequence.
                # Prompt-logprob requests score every prompt position instead:
                # skip the gather and take the full grid from the same pass.
                logits_positions=(
                    None
                    if wants_prompt
                    else (self._prompt_lens_gpu.view(-1) - 1 if is_prefill else None)
                ),
            )
            if wants_prompt:
                self._collect_prompt_logprobs(logits)
                rows = torch.arange(self.batch_size, device=logits.device)
                logits = logits[rows, self._prompt_lens_gpu.view(-1) - 1]
            self._allocated.append(engine.model_runner.decode_alloc_kv_cache(self.batch_size))
            # Mirror the in-place growth of atten_info.b_seq_len on the host,
            # keeping the ledger exactly one step ahead of the next forward.
            self._host_lens_t += 1

            generated = None
            if params.repetition_penalty != 1.0:
                generated = GeneratedSpan(
                    self._tokens[:, :cur_pos], self._is_generated[:, :cur_pos]
                )
            step_records = None
            if params.logprobs is not None:
                next_token, step_records = engine.sampler.sample_with_logprobs(
                    logits, params, generated
                )
                next_token = next_token.reshape(-1)
            else:
                next_token = engine.sampler.sample(logits, params, generated).reshape(-1)
            # TP synchronisation: rank 0 samples, then broadcasts the token ids
            # to all other TP ranks. Without this, non-greedy sampling would
            # diverge across ranks (each has an independent RNG state). The
            # sampler already broadcasts before computing records, so this is
            # a no-op for them — and in a world of one the collective's own
            # early return makes issuing the call unconditionally safe.
            next_token = tensor_model_parallel_broadcast(next_token)

            # Only fill positions that are not part of the original prompt.
            writable = ~self._prompt_mask[:, cur_pos]
            self._tokens[:, cur_pos] = torch.where(writable, next_token, self._tokens[:, cur_pos])
            # Device-side only: no synchronisation happens here.
            self._stop.update(next_token, writable)
            self._accepted[:, cur_pos] = writable & ~self._stop.finished
            if step_records is not None:
                # A record accompanies every accepted token; the eos token (and
                # anything past a stop) is dropped, matching the text path.
                for i, keep in enumerate(self._accepted[:, cur_pos].tolist()):
                    if keep and step_records[i] is not None:
                        self.output_logprobs[i].append(step_records[i])

            prev_pos = cur_pos
            is_poll = step % POLL_INTERVAL == 0

            if stream:
                deltas = self._flush(pending_from, cur_pos + 1, check_repeat=is_poll)
                pending_from = cur_pos + 1
                yield deltas
            elif is_poll:
                self._flush(pending_from, cur_pos + 1, check_repeat=True)
                pending_from = cur_pos + 1

            if is_poll and self._stop.all_finished():
                break

        # Trailing columns the poll interval never covered.
        if pending_from < self._total_len:
            deltas = self._flush(pending_from, self._total_len, check_repeat=False)
            if stream and any(deltas):
                yield deltas

        # Release every cache row reserved across prefill + decode.
        engine.model_runner.kv_cache_manager.release_ref(torch.cat(self._allocated).long())
        engine.last_stop_reasons = self._stop.reasons()
        engine.last_output_logprobs = self.output_logprobs
        engine.last_prompt_logprobs = self.prompt_logprobs

    def _collect_prompt_logprobs(self, logits: torch.Tensor) -> None:
        """Score every prompt position from the full-grid prefill logits.

        Row ``j`` of sequence ``i`` predicts position ``j + 1``, so position 0
        has no predictor and is reported as ``None`` — the same contract the
        continuous engine uses for chunked prefill.
        """
        k = self._params.prompt_logprobs
        assert k is not None
        collected: list[list[PositionLogprobs | None]] = []
        for i in range(self.batch_size):
            plen = self._prompt_lens[i]
            records: list[PositionLogprobs | None] = [None]
            if plen > 1:
                targets = self._tokens[i, 1:plen]
                records.extend(rows_logprobs(logits[i, : plen - 1], targets, k))
            collected.append(records)
        self.prompt_logprobs = collected

    def _flush(self, start: int, end: int, check_repeat: bool) -> list[str]:
        """Read back columns ``[start, end)`` and turn them into text deltas.

        This is the only place the host reads sampled tokens, so it is also the
        only synchronisation point in the loop. Columns the sequence did not
        accept (prompt positions, or anything after it stopped) arrive as ``-1``
        and produce no text.
        """
        if start >= end:
            return [""] * self.batch_size
        columns = torch.where(self._accepted[:, start:end], self._tokens[:, start:end], -1).tolist()

        deltas = [""] * self.batch_size
        check = check_repeat and self._params.stop_on_repeat
        for i, row in enumerate(columns):
            delta = ""
            for token_id in row:
                if token_id >= 0:
                    delta += self._detokenizer.append(i, token_id)
            # The candidate text includes this window's tokens, matching the
            # historical behaviour where the token that completes a loop is
            # detected but not emitted.
            if (
                check
                and delta
                and self._is_running(i)
                and detect_repetition(self._detokenizer.text(i))
            ):
                self._stop.mark_repeat(i)
                delta = ""
            deltas[i] = delta
            self.completions[i] += delta
        return deltas

    def _is_running(self, index: int) -> bool:
        """Whether sequence ``index`` is still generating.

        Reads one device flag, but only on the poll cadence and only when the
        repetition breaker is armed.
        """
        return not bool(self._stop.finished[index])

    def _step_positions(
        self, input_ids: torch.Tensor, prev_pos: int, is_prefill: bool
    ) -> torch.Tensor:
        """Return position ids for this step.

        For multimodal prefill the caller supplies mrope positions directly. For
        text prefill a single ``arange`` is fine because padded tokens are masked
        out by ``b_seq_len`` inside the attention kernel. Decode is the delicate
        case: each sequence has its own current position ``prompt_lens[i] + k``,
        so the offset must be applied per row rather than shared across the batch.
        For mrope, ``position_deltas[i]`` shifts each row to continue from its last
        multimodal position instead of its token count.
        """
        if is_prefill:
            if self._position_ids is not None:
                return self._position_ids
            seq_len = input_ids.shape[1]
            return (
                torch.arange(prev_pos, prev_pos + seq_len, device=self._engine.device)
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )

        # Decode: seq_len is 1 and each sequence is at its own step count relative
        # to its own prompt length. ``max_prompt_len`` is the loop's starting
        # column, so the offset is known on the host — reading it off the device
        # would cost a synchronisation every step.
        positions = self._prompt_lens_gpu + (prev_pos - self._max_prompt_len)
        if self._position_deltas is not None:
            positions = positions + self._position_deltas.view(-1, 1)
        return positions


class LLMEngine:
    """Loads a model and generates text from tokenised prompts.

    Args:
        checkpoints_dir: HuggingFace checkpoint directory (``config.json`` plus
            ``*.safetensors``).
        tokenizer_path: Tokenizer location; defaults to ``checkpoints_dir``.
        max_seq_len: Context bound; also caps the KV cache.
        max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
        device: Torch device string.
        use_cuda_graph: Capture decode CUDA graphs. Text models default to ``True``
            because replaying a graph removes the ~300 kernel launches an eager
            decode step costs; multimodal models ignore it.
        cuda_graph_lazy: O13 lazy capture — capture only a seed pair at startup
            and the remaining ``(batch, bucket)`` shapes the first time a step
            needs them. Cold start stops paying for shapes the workload may
            never produce; the first step on a new shape pays ~0.5–1 s once.
    """

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str | None = None,
        max_seq_len: int = 2048,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = True,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        prefill_context_parallel_size: int = 1,
        prefill_context_parallel_strategy: str = "zigzag",
        decode_context_parallel_size: int = 1,
        dcp_comm_backend: str = "ag_rs",
        kv_cache_dtype: str = "auto",
        cuda_graph_lazy: bool = False,
        hf_overrides: dict[str, object] | None = None,
    ) -> None:
        self.device = device
        self.model_path = checkpoints_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.enable_expert_parallel = enable_expert_parallel
        topology = get_parallel_topology()
        requested = (
            tensor_parallel_size,
            prefill_context_parallel_size,
            prefill_context_parallel_strategy,
            decode_context_parallel_size,
            dcp_comm_backend,
        )
        joined = (
            topology.tp_size,
            topology.pcp_size,
            topology.pcp_strategy,
            topology.dcp_size,
            topology.dcp_comm_backend,
        )
        if requested != joined:
            raise ValueError(
                "engine parallel configuration does not match the process groups: "
                f"requested={requested}, joined={joined}"
            )
        self.prefill_context_parallel_size = prefill_context_parallel_size
        self.prefill_context_parallel_strategy = prefill_context_parallel_strategy
        self.decode_context_parallel_size = decode_context_parallel_size
        self.dcp_comm_backend = dcp_comm_backend
        if prefill_context_parallel_size > 1:
            # The groups form (attn-TP = TP / PCP), but every model layer still
            # shards over the whole TP group; serving PCP needs the attention
            # and MLP shardings split first.
            raise NotImplementedError(
                "prefill context parallelism is not wired into the model data path yet"
            )
        if decode_context_parallel_size > 1:
            from ..batch_overlap.two_batch_overlap import tbo_policy

            if tbo_policy().enabled:
                raise ValueError(
                    "two-batch overlap splits a decode step around the TP all-reduce; "
                    "its halves do not schedule the DCP gather/merge collectives"
                )

        # EP keeps its graphs. Its decode path posts the dispatch/combine
        # all-to-all exchanges on the shared comm stream with the same
        # wait_stream -> NCCL -> record -> fence discipline the deferred
        # all-reduce uses, and TBO already captures that discipline into a
        # graph (the fork/join event edges record like any other dependency).
        # The exchange buffers are equal-split, so their shapes are fixed by the
        # captured batch and the routing kernels recompute from the live ids at
        # replay — a captured a2a replays correctly in lockstep across ranks,
        # exactly like the captured TP all-reduce. Keeping graphs on is what
        # lets the EP decode step escape the Python launch floor, which is the
        # only thing that ever swallowed the SBO overlap's gain.
        if enable_expert_parallel and use_cuda_graph:
            # EP's a2a buffers (ep_size * rows * top_k * hidden) make each
            # captured graph far larger than a dense TP one, so the full grid
            # does not fit beside a profiled KV pool. Lazy capture seeds a pair
            # and captures the rest on demand — the mode that keeps EP+graph
            # within memory; the OOM fallback in enable_cuda_graph covers any
            # shape that still does not fit.
            cuda_graph_lazy = True

        self.model_runner = ModelRunner.build(
            checkpoints_dir=checkpoints_dir,
            max_seq_len=max_seq_len,
            max_gpu_num_blocks=max_gpu_num_blocks,
            device=device,
            use_cuda_graph=use_cuda_graph,
            quantization=quantization,
            kv_cache_dtype=kv_cache_dtype,
            cuda_graph_lazy=cuda_graph_lazy,
            hf_overrides=hf_overrides,
        )

        if use_cuda_graph:
            from ..batch_overlap.two_batch_overlap import tbo_policy

            self.model_runner.enable_cuda_graph(lazy=cuda_graph_lazy, tbo=tbo_policy().enabled)
        self.tokenizer = self._load_tokenizer(tokenizer_path or checkpoints_dir)
        self.sampler = Sampler()
        self.max_seq_len = self.model_runner.max_seq_len

        self.stop_token_ids = load_stop_token_ids(checkpoints_dir, self.tokenizer)
        self.last_stop_reasons: list[str] | None = None
        self.last_output_logprobs: list[list[PositionLogprobs]] | None = None
        self.last_prompt_logprobs: list[list[PositionLogprobs | None]] | None = None

    @staticmethod
    def _load_tokenizer(path: str) -> AutoTokenizer:
        """LLaVA ships a slow tokenizer whose fast variant changes special-token
        handling, so it must load with use_fast=False.
        """
        use_fast = "llava" not in get_model_name_from_path(path).lower()
        return AutoTokenizer.from_pretrained(path, use_fast=use_fast, trust_remote_code=True)

    @property
    def pad_id(self) -> int:
        """Fill id for non-prompt columns; falls back to EOS when unset."""
        tok = self.tokenizer
        return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
        position_ids: torch.Tensor | None = None,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> Iterator[list[str]]:
        """Run prefill + incremental decode, yielding new text per batch each step.

        Args:
            prompt_token_ids: One token-id list per sequence in the batch.
            params: Sampling configuration.
            position_ids: Optional explicit positions (used by the mrope path for
                multimodal prompts). Defaults to a plain arange per step.
            multi_modal_inputs: Processor outputs consumed during prefill only.

        Yields:
            A list (one entry per sequence) of the text produced *this* step.
        """
        session = _DecodeSession(self, prompt_token_ids, params, position_ids, multi_modal_inputs)
        yield from session.run(stream=True)

    @torch.inference_mode()
    def generate_text(
        self,
        prompt_token_ids: list[list[int]],
        params: SamplingParams,
        position_ids: torch.Tensor | None = None,
        multi_modal_inputs: dict[str, Any] | None = None,
    ) -> list[str]:
        """Return full completions, without streaming.

        Takes the non-streaming path through :class:`_DecodeSession`, which reads
        sampled tokens back on a coarse interval rather than every step.
        """
        session = _DecodeSession(self, prompt_token_ids, params, position_ids, multi_modal_inputs)
        for _ in session.run(stream=False):
            pass
        return session.completions
