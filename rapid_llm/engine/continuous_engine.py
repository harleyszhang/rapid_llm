"""Continuous batching: a step-driven engine where requests join and leave freely.

Each ``step()`` asks the :class:`~rapid_llm.engine.scheduler.Scheduler` for a
plan, runs it through the executor, harvests sampled tokens and updates request
state — chunked prefills and running decodes share one pass. The
``PIPELINE_ENV`` mode reshapes that order into launch/harvest pipelining:
schedule and launch step N while N-1 is still on the GPU, then harvest N-1's
tokens — their readback has landed under N's forward, so the host's
detokenise/stop work overlaps compute instead of serialising against it.
Decode inputs never cross to the host in that mode: the worker feeds them back
on the device (see :meth:`~rapid_llm.executor.worker.ModelWorker`).

Usage:
    engine.add_request(prompt, params)
    finished = engine.step()
"""

from __future__ import annotations

import itertools
import math
import os
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING, NamedTuple

import torch

from ..distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from ..executor.cuda_graph import DEFAULT_BATCH_SIZES
from ..executor.executor import (
    Executor,
    MultiprocExecutor,
    UniProcExecutor,
    launch_tensor_parallel,
    reclaim_tensor_parallel_followers,
)
from ..executor.worker import ModelInput, PassKind, PassLogprobs, pipeline_enabled
from ..models.config import read_model_type
from ..models.registry import ModelRegistry
from ..tools.observability import EngineMetrics, Tracer
from ..utils.env_compat import getenv
from ..utils.logger import get_logger
from .detokenizer import IncrementalDetokenizer
from .kv_offload import OffloadingManager
from .ngram_proposer import NgramProposer
from .outputs import CompletionOutput, RequestOutput
from .sampler import PositionLogprobs, SamplingParams
from .scheduler import (
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
)
from .stop_criteria import POLL_INTERVAL, detect_repetition

if TYPE_CHECKING:
    from .llm_engine import LLMEngine

#: Set to ``0`` to keep the pre-fused behaviour: resumed chunks (and prefix-cache
#: hit remainders) extend one decode-style row per token instead of running as
#: a grid pass through the chunked prefill kernel. A kill-switch rather than a
#: config field because the engine decides once, from the cache dtype.
_FUSED_CHUNK_ENV = "RAPID_LLM_FUSED_CHUNK_PREFILL"

logger = get_logger(__name__)

#: Ngram speculative decoding (O5): ``1``/``true``/``on`` enables ngram proposal
#: for decode requests. The proposer scans the prompt + generated tokens for
#: repeated n-grams and proposes draft continuations; a verify pass checks them
#: in one forward, accepting matches and sampling at the first mismatch.
_SPECULATE_ENV = "LITE_LLAMA_SPECULATE"


class _Work(NamedTuple):
    """A plan plus the requests whose tokens it will produce, in that order.

    The plan names slots, not requests, so the step keeps request objects
    alongside it. ``requests`` is parallel to ``plan.sampled``;
    ``chunk_requests`` is parallel to ``plan.slots`` (chunk passes only).
    """

    plan: ModelInput
    requests: list[Request]
    chunk_requests: tuple[Request, ...] = ()


# Background tokenize workers (O10). Tokenizer.encode releases the GIL, so a
# few workers give a burst of arrivals genuinely parallel encoding; idle
# threads cost nothing.
_TOKENIZE_WORKERS = 4


class _TokenizeJob(NamedTuple):
    """One background encode (O10): a request awaiting its prompt tokens."""

    request: Request
    future: Future[list[int]]
    on_error: Callable[[Request, BaseException], None] | None


def _chunk_work(kind: PassKind, chunks: list[tuple[Request, int]]) -> _Work:
    """Plan one prompt-chunk pass; the two routes differ only in ``kind``.

    Chunk ``i`` writes cache rows ``[num_computed_tokens - chunk,
    num_computed_tokens)`` of its slot — the scheduler already advanced the
    counter. A chunk resuming on a prefix-cache hit also carries the copies.
    """
    slots, starts, lens, tokens = [], [], [], []
    sampled, requests = [], []
    prompt_logprobs, prompt_targets = [], []
    writes: list[tuple[int, int, int, tuple[int, ...]]] = []
    for row, (request, chunk) in enumerate(chunks):
        start = request.num_computed_tokens - chunk
        end = request.num_computed_tokens
        slots.append(request.slot)
        starts.append(start)
        lens.append(end)
        tokens.extend(request.prompt_token_ids[start:end])
        prompt_logprobs.append(request.params.prompt_logprobs)
        # Row j is scored against the token at start+j+1. A partial chunk's
        # last row targets the *next* chunk's first token; a final chunk's
        # last row is sampled and has no target. Both tails pad with 0.
        targets = request.prompt_token_ids[start + 1 : end + 1]
        prompt_targets.extend(targets + [0] * (chunk - len(targets)))
        writes += [
            (request.slot, group_id, start_block, block_ids)
            for group_id, start_block, block_ids in request.block_plan
        ]
        if request.num_computed_tokens == request.prompt_len:
            # Only a finished prompt has a next token to sample; the pass
            # mixes both, so sampled rows are a subset named by row index.
            sampled.append(row)
            requests.append(request)

    wants_prompt = any(k is not None for k in prompt_logprobs)
    return _Work(
        ModelInput(
            kind=kind,
            slots=tuple(slots),
            seq_starts=tuple(starts),
            seq_lens=tuple(lens),
            tokens=tuple(tokens),
            sampling=tuple(request.params for request in requests),
            sampled=tuple(sampled),
            # A first token has no repetition-penalty history yet.
            gen_counts=(0,) * len(requests),
            block_writes=tuple(writes),
            prompt_logprobs=tuple(prompt_logprobs) if wants_prompt else (),
            prompt_targets=tuple(prompt_targets) if wants_prompt else (),
        ),
        requests,
        tuple(request for request, _ in chunks),
    )


def _prefill_work(
    group: list[Request], chunk_lens: list[int], chunked_min_rows: float = math.inf
) -> list[_Work]:
    """Split the step's prompt chunks by the kernel each may legally use.

    A *first* chunk (``num_computed_tokens == chunk``) runs as a padded grid
    through the prefill kernel — nothing of the prompt is cached yet. A
    *resumed* chunk lands on cached rows, and its pass is routed by row count:

    * a short remainder whose rows fit inside a captured CUDA-graph batch
      extends instead — decode-style rows replay a decode graph, the cheapest
      pass of all;
    * a longer one (``rows >= chunked_min_rows``) runs as a grid pass through
      the chunked prefill kernel — queries are the chunk's rows, keys/values
      its slot's own cache rows, tensor-core tiled. An extend pass of that
      width cannot replay (it exceeds every captured batch) and pays one
      decode-style row per token, roughly an order of magnitude more per token
      than the grid.

    ``chunked_min_rows`` is ``math.inf`` when the chunked kernel may not run
    at all (an fp8 cache the kernel cannot decode, or the
    ``RAPID_LLM_FUSED_CHUNK_PREFILL=0`` kill-switch), and the smallest value
    that cannot replay a graph (one past the largest captured batch size, so
    ``1`` with graphs off) otherwise.
    """
    pairs = list(zip(group, chunk_lens, strict=True))
    fresh = [pair for pair in pairs if pair[0].num_computed_tokens == pair[1]]
    resumed = [pair for pair in pairs if pair[0].num_computed_tokens > pair[1]]
    works = [_chunk_work(PassKind.PREFILL, fresh)] if fresh else []
    if resumed:
        rows = sum(chunk for _, chunk in resumed)
        kind = PassKind.PREFILL if rows >= chunked_min_rows else PassKind.EXTEND
        works.append(_chunk_work(kind, resumed))
    return works


def _decode_work(running: list[Request], *, from_device: bool = False) -> _Work:
    """Plan one decode token for every fully prefilled request.

    The input token is the last one each request generated — already back on
    the host from the previous step's synchronisation. With ``from_device``
    (the launch/harvest pipeline) it is instead whatever the device sampled
    last, which the host has *not* harvested yet: the plan carries a
    placeholder id only the worker's device-side gather can replace, and every
    length is the optimistic one — the request's bookkeeping plus its
    launched-but-unharvested token.
    """

    def ahead(request: Request) -> int:
        """How far the device is past the host's ledger; 0 outside the pipeline."""
        return request.pending_tokens if from_device else 0

    # The request already counts the token it is about to feed, plus the one
    # the pipeline fed it before the host ever saw it.
    seq_lens = tuple(request.seq_len + ahead(request) for request in running)
    return _Work(
        ModelInput(
            kind=PassKind.DECODE,
            slots=tuple(request.slot for request in running),
            # One token per row, landing at the row its cache length points at.
            seq_starts=tuple(length - 1 for length in seq_lens),
            seq_lens=seq_lens,
            # ``-1`` is deliberately not a token id: if a placeholder ever
            # reaches an embedding, the pipeline's device-side gather was
            # skipped and the failure is loud, never a silently wrong token.
            tokens=(
                tuple(-1 for _ in running)
                if from_device
                else tuple(request.output_token_ids[-1] for request in running)
            ),
            sampling=tuple(request.params for request in running),
            sampled=tuple(range(len(running))),
            gen_counts=tuple(len(request.output_token_ids) + ahead(request) for request in running),
            # A decode step that crossed a block boundary was given a fresh page
            # by the scheduler; its table entry has to land before the gather.
            block_writes=tuple(
                (request.slot, group_id, start_block, block_ids)
                for request in running
                for group_id, start_block, block_ids in request.block_plan
            ),
        ),
        running,
    )


class ContinuousBatchingEngine:
    """Serves independently arriving requests as one continuously reshaped batch.

    Drive it by calling :meth:`step` in a loop: each call runs the scheduler's
    plan (prompt chunks, then a decode token for everything running) and returns
    the requests that produced a token. One host-device synchronisation per
    step is deliberate — it reads sampled tokens back to detokenise and to
    decide who stops, so a finished request leaves the batch on the next step.

    Args:
        engine: A built :class:`~rapid_llm.engine.llm_engine.LLMEngine`; takes
            over its KV cache and must be the only user of it.
        config: Admission limits. Defaults derive ``max_seq_len`` from the engine.
        executor: Where passes run. Defaults to a
            :class:`~rapid_llm.executor.executor.UniProcExecutor` (single GPU);
            injecting a fake is how a test drives the step loop without a model.
        async_tokenize: O10 — encode prompts on a background thread pool
            instead of the caller's thread. ``add_request`` returns a request
            whose tokens fill in on the first :meth:`step` after the encode
            lands, so a large prompt no longer stalls the engine loop (or the
            step cadence) for its tens of milliseconds of encoding.
        offloading: A CPU tier whose blocks promote into the GPU pool, built by
            the caller around the executor's cache (see
            :class:`~rapid_llm.engine.kv_offload.base.OffloadingManager`).
            ``None`` (the default) keeps the engine's KV GPU-only.

    Raises:
        NotImplementedError: The checkpoint is multimodal — vision prefill
            needs per-request processor outputs the batched grid has no place for.
    """

    def __init__(
        self,
        engine: LLMEngine,
        config: SchedulerConfig | None = None,
        executor: Executor | None = None,
        *,
        pipeline: bool | None = None,
        async_tokenize: bool = False,
        offloading: OffloadingManager | None = None,
    ) -> None:
        if engine.model_runner.spec.is_multimodal:
            raise NotImplementedError(
                "continuous batching supports text-only checkpoints; "
                "use LLM.generate() for vision-language models"
            )

        self.engine = engine
        self.device = engine.device
        self.tokenizer = engine.tokenizer
        self.stop_token_ids = engine.stop_token_ids

        config = config or SchedulerConfig(max_seq_len=engine.max_seq_len)
        if config.max_seq_len > engine.max_seq_len:
            raise ValueError(
                f"scheduler max_seq_len {config.max_seq_len} exceeds the engine's "
                f"{engine.max_seq_len}"
            )
        self.config = config

        self._pipeline = pipeline_enabled() if pipeline is None else pipeline
        if self._pipeline and config.enable_preemption:
            raise ValueError(
                "the launch/harvest pipeline cannot plan one token ahead for a "
                "request preemption is about to recompute from scratch; drop "
                "enable_preemption or leave the pipeline off"
            )

        # Resumed chunks route by row count (see ``_prefill_work``): a short
        # remainder extends as rows that replay a decode graph, a longer one
        # runs through the chunked prefill kernel, whose tensor-core tiling
        # beats paying one decode-style row per token — the per-token cost
        # that kept prefix-cache hits from lowering TTFT. An fp8 cache stores
        # e4m3 bytes the chunk kernel cannot decode, so those keep extending.
        # Read through ``getattr`` so a test double whose model_runner is a
        # bare namespace (no model behind it, so no cache dtype either) keeps
        # constructing the engine instead of mimicking runner internals.
        runner_config = getattr(engine.model_runner, "config", None)
        kv_fp8 = runner_config is not None and runner_config.kv_cache_torch_dtype == torch.uint8
        fused = getenv(_FUSED_CHUNK_ENV, "1") != "0" and not kv_fp8
        if kv_fp8 and config.enable_chunked_prefill:
            # Say it once, the way vLLM announces a config it downgraded for
            # you: chunked prefill still runs, only its resumed chunks take the
            # slower route, so a TTFT that regressed against a bf16 cache has a
            # line in the log to point at.
            logger.info(
                "fp8 KV cache: chunked prefill stays on, but resumed chunks run "
                "through the extend pass -- the chunk kernel cannot read e4m3 bytes"
            )

        self._executor: Executor = executor or UniProcExecutor(
            engine, config.max_num_seqs, config.max_seq_len, pipeline=self._pipeline
        )
        # The replay cap reads the runner's live graph manager, so it is
        # settled after the executor; ``math.inf`` disables the chunked route.
        # With graphs off the cap falls back to the *configured* batch sizes:
        # EXTEND (fp32 vector sums) and the chunked kernel (``tl.dot``) are not
        # numerically equivalent, so a threshold that flipped with the graph
        # switch made eager and graph diverge from the first token. Replay is
        # bit-identical (verified at capture), so a switch-invariant threshold
        # suffices -- measured: 9 output mismatches removed at no TTFT/TPOT cost.
        manager = self._graph_manager()
        cap = max(manager.batch_sizes, default=0) if manager else max(DEFAULT_BATCH_SIZES)
        self._chunked_min_rows = cap + 1 if fused else math.inf
        # The executor owns the cache, so it decides how many requests can be in
        # flight; the scheduler hands out exactly those slots, and pages out of a
        # pool sized by the cache the executor actually profiled.
        self.scheduler = Scheduler(
            config, self._executor.num_slots, self._executor.num_kv_blocks or None, offloading
        )
        self._offloading = offloading

        self._detokenizers: dict[str, IncrementalDetokenizer] = {}
        self._request_ids = itertools.count()
        self._step_count = 0
        # O5 ngram speculative decoding: proposer scans prompt + generated
        # tokens for repeated n-grams and proposes draft continuations.
        # Enabled via LITE_LLAMA_SPECULATE=1; off by default because the
        # verify pass adds a forward per step and only pays off when the
        # workload has repetitive structure (code, repeated templates).
        self._speculate = os.environ.get(_SPECULATE_ENV, "0").strip().lower() in ("1", "true", "on")
        self._proposer = NgramProposer(max_ngram_size=5, max_draft=6) if self._speculate else None
        # O10 background tokenize: pool created on first use, jobs collected at
        # the top of every step (same thread that calls add_request, so the
        # dict needs no lock).
        self._async_tokenize = async_tokenize
        self._tokenize_pool: ThreadPoolExecutor | None = None
        self._tokenizing: dict[str, _TokenizeJob] = {}
        # One step's launched-but-unharvested passes, when pipelined: each entry
        # is the launched list of (work, host tokens, event, records). Depth is
        # one by construction — every step harvests before it schedules.
        self._inflight: deque[
            list[tuple[_Work, torch.Tensor, torch.cuda.Event | None, PassLogprobs | None]]
        ] = deque()

        # Observability: metrics and tracing are cheap no-ops when disabled,
        # so the hot loop never branches on them.
        self.metrics = EngineMetrics.from_env()
        self.tracer = Tracer.from_env()
        self._spans: dict[str, object] = {}

    # ------------------------------------------------------------------ build #
    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        tokenizer: str | None = None,
        max_seq_len: int = 2048,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill: bool = True,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        max_gpu_num_blocks: int | None = None,
        device: str = "cuda",
        use_cuda_graph: bool = True,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        enable_expert_parallel: bool = False,
        kv_cache_dtype: str = "auto",
        enable_prefix_cache: bool = False,
        prefix_cache_blocks: int | None = None,
        enable_preemption: bool = False,
        decode_window_steps: int = 0,
        cuda_graph_lazy: bool = False,
        async_tokenize: bool = False,
        pipeline: bool | None = None,
        hf_overrides: dict[str, object] | None = None,
    ) -> ContinuousBatchingEngine:
        """Load a checkpoint and wrap it in a continuous-batching engine.

        Args:
            model: HuggingFace checkpoint directory.
            tokenizer: Tokenizer location; defaults to ``model``.
            max_seq_len: Context window, and the per-slot cache size.
            max_num_seqs: Concurrency ceiling.
            max_num_batched_tokens: Padded token budget for one prefill group.
            enable_chunked_prefill: Whether a prompt may prefill across steps
                (vLLM's default: on). Off prefills each prompt in one pass and
                needs ``max_num_batched_tokens >= max_seq_len``.
            max_chunk_size: Maximum tokens a request may prefill in one step;
                ``0`` disables chunking.
            max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
            device: Torch device string.
            use_cuda_graph: Capture decode graphs. Worth keeping on: continuous
                batching pads odd batch sizes onto the captured grid, so most
                steps stay on the graph path. Also honoured above
                ``tensor_parallel_size`` 1, where the capture takes the sharded
                layers' all-reduce with it — see
                :meth:`~rapid_llm.executor.model_runner.ModelRunner.enable_cuda_graph`
                for the checks that decide whether those graphs are kept.
            quantization: Runtime weight quantisation, forwarded to the engine.
                Orthogonal to batching -- it changes the linear layers, not the
                KV cache or the schedule.
            tensor_parallel_size: GPUs this replica's weights are split over.
                Above 1, ranks 1.. spawn as followers and every step's plan is
                broadcast to them; this process stays rank 0. If this process
                already sits in a TP group (the CLI, a DP controller), that group
                is reused and the value only has to agree with it.
            enable_expert_parallel: Split MoE experts whole-across-ranks over
                the TP group instead of TP-splitting each expert's intermediate
                dim (vLLM semantics). Attention and shared experts stay TP;
                routed tokens travel by all-to-all. Decode keeps its CUDA
                graphs — the a2a exchanges capture with the same comm-stream
                discipline the deferred all-reduce uses, and EP defaults to
                lazy capture so the larger exchange buffers fit.
            kv_cache_dtype: KV-cache element type, forwarded to the engine
                (``"auto"`` for fp16, or an fp8 spelling to halve the cache).
            enable_prefix_cache: Reuse the K/V of prompt prefixes already
                resident in the cache. Off by default: it only pays when prompts
                share a prefix, and otherwise costs a hash per block. See
                :mod:`rapid_llm.engine.prefix_cache`.
            prefix_cache_blocks: Optional physical prefix-cache capacity in
                16-token blocks.  ``None`` uses the profiled KV capacity.
            enable_preemption: Allow the scheduler to recompute and evict a
                young decode request when logical concurrency exceeds slots.
            decode_window_steps: O9 decode window — how many pure-decode steps
                a fresh prompt waits at most before it may interrupt them.
                ``0`` (default) admits immediately; ``N > 0`` trades a little
                TTFT for TPOT smoothness under bursty arrivals.
            cuda_graph_lazy: O13 lazy graph capture — seed pair at startup,
                remaining shapes captured on first use. Pairs with
                ``use_cuda_graph``; ignored when graphs are off or TP > 1.
            async_tokenize: O10 — encode prompts on a background thread pool
                so a large prompt's tens of milliseconds of tokenization stop
                serialising against the engine loop and every step it would
                have delayed. ``add_request`` returns immediately; the request
                joins the scheduler once its tokens are ready.
            pipeline: Run the launch/harvest engine loop (O2): launches run one
                step ahead of harvests so the host's bookkeeping overlaps
                compute, and decode inputs stay on the device. ``None`` reads
                :data:`~rapid_llm.executor.worker.PIPELINE_ENV`. Stop handling
                runs one token late, and a request asking for logprobs pays
                its synchronisation inside the pass as usual.
            hf_overrides: Fields applied over the checkpoint's ``config.json``
                (vLLM ``--hf-overrides`` semantics), e.g.
                ``{"num_hidden_layers": 1}`` to run a trimmed stack — the
                supported way to exercise one family's layer arithmetic
                without paying for the whole model. Passed to every rank.

        Raises:
            NotImplementedError: The checkpoint is multimodal.
            ValueError: ``tensor_parallel_size`` contradicts a group this process
                is already a member of.
        """
        # Keep CPU-only planning and fake-executor tests importable without
        # Triton. A real model is the only path that needs the GPU engine.
        from .llm_engine import LLMEngine

        spec = ModelRegistry.resolve(read_model_type(model))
        if spec.is_multimodal:
            raise NotImplementedError(
                "continuous batching supports text-only checkpoints; "
                "use LLM.generate() for vision-language models"
            )

        engine_kwargs = {
            "checkpoints_dir": model,
            "tokenizer_path": tokenizer,
            "max_seq_len": max_seq_len,
            "max_gpu_num_blocks": max_gpu_num_blocks,
            # Above tensor_parallel_size 1 the captured all-reduce is only
            # replayed once the startup gates in ``ModelRunner`` accept it.
            "use_cuda_graph": use_cuda_graph,
            "quantization": quantization,
            "kv_cache_dtype": kv_cache_dtype,
            "cuda_graph_lazy": cuda_graph_lazy,
            "hf_overrides": hf_overrides,
        }

        resolved_pipeline = pipeline_enabled() if pipeline is None else pipeline

        # Followers must exist before this rank builds its engine: sharded
        # layers read their width from the process group.
        followers: tuple[BaseProcess, ...] = ()
        joined = get_tensor_model_parallel_world_size()
        if joined > 1 and joined != tensor_parallel_size:
            raise ValueError(
                f"this process is already rank {get_tensor_model_parallel_rank()} of a {joined}-way "
                f"tensor-parallel group, but tensor_parallel_size={tensor_parallel_size}"
            )
        if joined == 1 and tensor_parallel_size > 1:
            followers = launch_tensor_parallel(
                tensor_parallel_size,
                engine_kwargs,
                max_num_seqs,
                enable_expert_parallel=enable_expert_parallel,
                device=device,
                pipeline=resolved_pipeline,
            )

        # From here on this process is rank 0 of a group it owns (when it
        # launched followers). Any failure in the build — an OOM loading the
        # weight shard is the common one — must hand that group back before
        # the exception escapes: a leftover half-dead parallel state re-shards
        # the next engine this process builds and turns every later test in
        # it into an all_reduce against a process group that no longer exists.
        try:
            engine = LLMEngine(
                device=device,
                tensor_parallel_size=tensor_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                **engine_kwargs,
            )
            config = SchedulerConfig(
                max_seq_len=engine.max_seq_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=enable_chunked_prefill,
                max_chunk_size=max_chunk_size,
                enable_prefix_cache=enable_prefix_cache,
                prefix_cache_blocks=prefix_cache_blocks,
                enable_preemption=enable_preemption,
                decode_window_steps=decode_window_steps,
            )
            executor: Executor | None = None
            if get_tensor_model_parallel_world_size() > 1:
                executor = MultiprocExecutor(
                    engine,
                    config.max_num_seqs,
                    config.max_seq_len,
                    followers,
                    pipeline=resolved_pipeline,
                )
            return cls(
                engine, config, executor, pipeline=resolved_pipeline, async_tokenize=async_tokenize
            )
        except BaseException:
            if followers:
                reclaim_tensor_parallel_followers(followers)
            raise

    # ------------------------------------------------------------- public API #
    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        prompt_token_ids: list[int] | None = None,
        on_error: Callable[[Request, BaseException], None] | None = None,
    ) -> Request:
        """Queue a request and return the handle that tracks it.

        The handle is updated in place by :meth:`step`, so a caller can read
        ``delta``, ``text`` and ``finish_reason`` as generation proceeds.

        Args:
            prompt: Prompt text, already chat-templated if the model wants that.
            sampling_params: Per-request knobs; defaults to :class:`SamplingParams`.
            request_id: Caller-supplied id; generated when omitted.
            prompt_token_ids: Pre-tokenised prompt, to skip re-encoding.
            on_error: O10 — fired (on the engine thread, from :meth:`step`)
                when a background encode fails or the tokenised prompt is
                rejected. The synchronous path raises from here instead.
        """
        if request_id is None:
            request_id = f"req-{next(self._request_ids)}"
            while self._request_id_in_use(request_id):
                request_id = f"req-{next(self._request_ids)}"
        elif self._request_id_in_use(request_id):
            raise ValueError(f"request id {request_id!r} is already active")

        if prompt_token_ids is None and self._async_tokenize:
            # O10: encode off this thread. The request joins the scheduler (or
            # fails, firing ``on_error``) at the top of the next step, so a
            # large prompt never stalls the loop that drives every other
            # request's steps.
            request = Request(
                request_id=request_id,
                prompt=prompt,
                prompt_token_ids=[],
                params=sampling_params or SamplingParams(),
            )
            future = self._ensure_tokenize_pool().submit(
                self.tokenizer.encode, prompt, add_special_tokens=True
            )
            self._tokenizing[request_id] = _TokenizeJob(request, future, on_error)
            return request

        request = Request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=(
                prompt_token_ids
                if prompt_token_ids is not None
                else self.tokenizer.encode(prompt, add_special_tokens=True)
            ),
            params=sampling_params or SamplingParams(),
        )
        self._register_request(request)
        return request

    def _request_id_in_use(self, request_id: str) -> bool:
        """Check every live phase of the engine's request-id namespace.

        A tokenisation job is not yet visible to :class:`Scheduler`, so the
        scheduler's duplicate check alone used to let a second explicit id
        overwrite the first job in ``_tokenizing``.  Keep this check at the
        public admission boundary so all backends observe one namespace.
        """
        return (
            request_id in self._tokenizing
            or request_id in self._detokenizers
            or self.scheduler.has_request_id(request_id)
        )

    def _register_request(self, request: Request) -> None:
        """Hand a fully tokenised request to the scheduler and open its buffers."""
        self.scheduler.add_request(request)
        self._detokenizers[request.request_id] = IncrementalDetokenizer(self.tokenizer, 1)
        self._spans[request.request_id] = self.tracer.start_span(
            "request", request_id=request.request_id, prompt_tokens=request.prompt_len
        )

    def _ensure_tokenize_pool(self) -> ThreadPoolExecutor:
        """The shared encode pool, created on first background tokenize."""
        if self._tokenize_pool is None:
            self._tokenize_pool = ThreadPoolExecutor(
                max_workers=_TOKENIZE_WORKERS, thread_name_prefix="rapid-llm-tokenize"
            )
        return self._tokenize_pool

    def _collect_tokenized(self) -> None:
        """Fold finished background encodes into the queue (O10).

        Runs at the top of every step, on the engine thread — the same one
        that called ``add_request``, so the dict needs no lock. A finished
        job's tokens join the scheduler; a failed one (encode error, empty or
        over-long prompt) finishes its request with ``finish_reason="invalid"``
        and the exception on ``request.error``, and fires the caller's
        ``on_error`` — exactly what the synchronous path would have raised.
        """
        if not self._tokenizing:
            return
        for request_id, job in [*self._tokenizing.items()]:
            if not job.future.done():
                continue
            del self._tokenizing[request_id]
            request = job.request
            try:
                request.prompt_token_ids = job.future.result()
                self._register_request(request)
            except Exception as exc:
                request.error = exc
                request.status = RequestStatus.FINISHED
                request.finish_reason = "invalid"
                self.metrics.finished.inc(finish_reason="invalid")
                if job.on_error is not None:
                    job.on_error(request, exc)

    def abort(self, request_id: str) -> Request | None:
        """Cancel a request; its slot is free for the next step."""
        job = self._tokenizing.pop(request_id, None)
        if job is not None:
            # Never reached the scheduler; its encode result is garbage now.
            request = job.request
            request.status = RequestStatus.FINISHED
            request.finish_reason = "abort"
            self.metrics.finished.inc(finish_reason="abort")
            return request
        request = self.scheduler.abort(request_id)
        if request is not None:
            self.metrics.finished.inc(finish_reason="abort")
            self.tracer.end_span(self._spans.pop(request.request_id, None), finish_reason="abort")
            self._retire(request)
        return request

    def has_unfinished_requests(self) -> bool:
        """Whether anything is queued, in flight, or awaiting its harvest."""
        return (
            self.scheduler.has_unfinished_requests()
            or bool(self._inflight)
            or bool(self._tokenizing)
        )

    def _await_offloaded_stores(self) -> None:
        """Order this step's writes behind any store still reading the cache.

        A store reads GPU blocks in place, so a block the pool has just reused
        must not be overwritten before the copy lands. The wait goes on the
        current stream — every launch of this step queues behind it — and costs
        nothing when offloading is off or no store is in flight.
        """
        if self._offloading is None:
            return
        for event in self._offloading.pending_store_events():
            torch.cuda.current_stream().wait_event(event)

    @torch.inference_mode()
    def step(self) -> list[Request]:
        """Run one engine step and return the requests it advanced.

        Returns:
            The requests that produced a token this step, in pass order. A
            request that stopped on a stop token is included with an empty
            ``delta`` — the async front end learns a request ended only from
            this list, so leaving it out would strand its stream. Under the
            pipeline the requests reported are those whose *previous* step's
            tokens this step harvested: the return value is one step behind
            the launches, which is the latency the mode trades for overlap.
        """
        self._collect_tokenized()
        self._await_offloaded_stores()
        if self._pipeline:
            return self._step_pipelined()
        return self._step_synchronous()

    @torch.inference_mode()
    def _step_pipelined(self) -> list[Request]:
        """Launch step N, then harvest step N-1 — the O2 engine loop.

        The order is the whole point. Scheduling and launching happen while
        the previous step's forward is still on the GPU, and the host stops
        only to harvest the *previous* step's tokens — by then their readback
        has landed under the current forward, so the wait is zero and the
        detokenise/stop work overlaps compute instead of serialising against
        it.

        What that costs, explicitly:

        * Stop handling runs one token late: a request that samples eos is
          retired one step after the synchronous engine would retire it, and
          the extra pass it rides is wasted compute whose token is discarded
          here. Late, not wrong — the stream hears the same finish reason,
          one step later.
        * The host's request ledger is optimistic: between launch and
          harvest, ``pending_tokens`` says the device is one token ahead, and
          the next decode plan adds exactly that back to write the right
          cache row.
        * A pass whose requests asked for logprob records already paid a host
          synchronisation inside execute (records are host objects), so it
          simply rides the same one-step-late harvest without extra cost.
        """
        scheduled = self.scheduler.schedule()
        if scheduled.is_empty and not self._inflight:
            return []

        self._step_count += 1
        # Freshly admitted requests owe a queue-time observation (num_computed
        # equals their first chunk); resumed chunks and preemption re-admissions
        # are skipped or re-counted by the same test.
        for request, chunk in zip(scheduled.prefill, scheduled.prefill_chunk_lens, strict=True):
            if request.num_computed_tokens == chunk:
                self.metrics.observe_queue_time(request)
        work: list[_Work] = []
        if scheduled.prefill:
            work += _prefill_work(scheduled.prefill, scheduled.prefill_chunk_lens)
        if scheduled.decode:
            work.append(_decode_work(scheduled.decode, from_device=True))

        # Detach the previous step's passes before this step's launches join
        # the queue: taking the harvest set first is what pins the depth at
        # one — a step never harvests what it just launched, so the tokens it
        # reads are always one forward old.
        previous = self._inflight.popleft() if self._inflight else None

        # Launch only: no token is read back here. The readback rides the
        # executor's copy stream behind the pass that produced it, and its
        # event is honoured one step later.
        staged: list[tuple[_Work, torch.Tensor, PassLogprobs | None]] = []
        for work_item in work:
            tokens, logprobs = self._executor.execute(work_item.plan)
            staged.append((work_item, tokens, logprobs))

        advanced: list[Request] = []
        # Harvest the previous step's tokens *before* this step's readbacks
        # are issued. A readback recycles whichever pinned buffer's copy
        # event has completed, and the previous step's completed long ago —
        # read after the new readback, its view would already hold this
        # step's tokens (the pool's "harvest N-1 strictly before launch N's
        # readback" contract). The executes above queued this step's
        # kernels, so the host still harvests while the GPU runs.
        if previous is not None:
            for work_item, host, event, logprobs in previous:
                if logprobs is not None and any(logprobs.prompt):
                    self._attribute_prompt_logprobs(work_item, logprobs.prompt)
                records = (
                    logprobs.sampled
                    if logprobs is not None and logprobs.sampled
                    else (None,) * len(work_item.requests)
                )
                if event is not None:
                    # Zero wait in the steady state: this copy landed one
                    # forward ago. Only a drained queue's final harvest pays.
                    event.synchronize()
                values = host.tolist()
                # The buffer goes back only now. This step's launches already
                # ran above, so releasing earlier would have let their copies
                # overwrite the tokens being read here.
                self._executor.release_readback(host)
                emitted: list[tuple[Request, int, PositionLogprobs | None]] = []
                for request, token, record in zip(work_item.requests, values, records, strict=True):
                    # The token is spent either way — the ledger closes for
                    # retired requests too, so it drains to exactly zero.
                    request.pending_tokens -= 1
                    if request.is_finished:
                        # The request stopped at an earlier harvest; this pass
                        # is the one extra token the late stop costs, and its
                        # output is discarded, not appended.
                        continue
                    emitted.append((request, token, record))
                advanced += self._harvest(emitted)

        if staged:
            launched: list[
                tuple[_Work, torch.Tensor, torch.cuda.Event | None, PassLogprobs | None]
            ] = []
            for work_item, tokens, logprobs in staged:
                host, event = self._executor.readback_async(tokens)
                launched.append((work_item, host, event, logprobs))
            # Optimistic advance: every request this step samples for owes one
            # token the host has not harvested; the next plan adds it back.
            for work_item, *_ in launched:
                for request in work_item.requests:
                    request.pending_tokens += 1
            self._inflight.append(launched)
        # Counter properties, not len(running): those copy the lists.
        self.metrics.observe_load(self.scheduler.num_running, self.scheduler.num_waiting)
        return advanced

    @torch.inference_mode()
    def _step_synchronous(self) -> list[Request]:
        """Plan, execute, and harvest in one step — the synchronous loop."""
        scheduled = self.scheduler.schedule()
        if scheduled.is_empty:
            return []

        self._step_count += 1
        # Freshly admitted requests owe a queue-time observation (num_computed
        # equals their first chunk); resumed chunks and preemption re-admissions
        # are skipped or re-counted by the same test.
        for request, chunk in zip(scheduled.prefill, scheduled.prefill_chunk_lens, strict=True):
            if request.num_computed_tokens == chunk:
                self.metrics.observe_queue_time(request)
        work: list[_Work] = []
        if scheduled.prefill:
            work += _prefill_work(
                scheduled.prefill, scheduled.prefill_chunk_lens, self._chunked_min_rows
            )

        # O5 speculative decoding: before the normal decode pass, propose draft
        # tokens for decode requests and verify them in one EXTEND forward.
        # Requests the verify pass could not serve fall through to the normal
        # decode pass below; that pass harvested its own tokens, so its
        # requests join the return value directly instead of re-entering this
        # one's harvest.
        spec_advanced: list[Request] = []
        remaining_decode = list(scheduled.decode)
        if self._speculate and self._proposer and remaining_decode:
            spec_advanced, remaining_decode = self._speculate_verify(remaining_decode)

        if remaining_decode:
            work.append(_decode_work(remaining_decode))

        # Execute every pass before reading any tokens back: a step's passes
        # are slot-disjoint, so one synchronisation per step suffices, and a
        # later pass's input prep rides the copy stream while an earlier pass's
        # forward is still on the GPU (the L1 overlap site).
        pending: list[tuple[_Work, torch.Tensor, PassLogprobs | None]] = []
        for work_item in work:
            tokens, logprobs = self._executor.execute(work_item.plan)
            pending.append((work_item, tokens, logprobs))

        emitted: list[tuple[Request, int, PositionLogprobs | None]] = []
        for work_item, tokens, logprobs in pending:
            # ``prompt`` is uniformly None for a decode pass.
            if logprobs is not None and any(logprobs.prompt):
                self._attribute_prompt_logprobs(work_item, logprobs.prompt)
            records = (
                logprobs.sampled
                if logprobs is not None and logprobs.sampled
                else (None,) * len(work_item.requests)
            )
            emitted += [
                (request, token, record)
                for (request, token), record in zip(
                    zip(work_item.requests, tokens.tolist(), strict=True), records, strict=True
                )
            ]
        # Counter properties, not len(running): those copy the lists.
        advanced = self._harvest(emitted) + spec_advanced
        self.metrics.observe_load(self.scheduler.num_running, self.scheduler.num_waiting)
        return advanced

    def _speculate_verify(
        self, decode_requests: list[Request]
    ) -> tuple[list[Request], list[Request]]:
        """O5 ngram speculative decoding: propose, verify, accept.

        For each decode request, proposes draft tokens from ngram lookup in
        the prompt + generated text, then verifies them in one EXTEND pass
        whose stretch is anchored on the last generated token: the anchor
        writes the KV row the decode pass it replaces would have written,
        and draft ``j`` lands at row ``seq_len + j``. Row ``j`` of the
        returned logits is the prediction after stretch row ``j``, so
        ``logits[j]`` is exactly where draft ``j`` is checked; a mismatch at
        ``j`` keeps drafts ``0..j-1`` and the argmax at ``j`` is the bonus
        token. Accepted drafts and the bonus all flow through
        :meth:`_harvest`, which owns detokenising, stop handling and the
        length cap — an eos draft retires the request, and tokens past a
        stop belong to nobody.

        Returns:
            ``(advanced, remaining)``: the requests the pass emitted tokens
            for, and requests that need normal decode (no drafts found, no
            block rows for the stretch, a logprob request the pass cannot
            record for, or the executor returned no logits).
        """
        assert self._proposer is not None
        # Phase 1: propose drafts, keeping only what the pool has rows for.
        # logprob requests keep the decode path: the verify pass returns raw
        # logits, not the per-token records the sampler's path produces, so
        # their records would skip every draft.
        drafts: list[tuple[Request, list[int]]] = []
        remaining: list[Request] = []
        for req in decode_requests:
            if req.params.logprobs is not None:
                remaining.append(req)
                continue
            all_ids = list(req.prompt_token_ids) + list(req.output_token_ids)
            proposed = self._proposer.propose(all_ids)
            fit = self.scheduler.reserve_speculative(req, len(proposed)) if proposed else 0
            if fit:
                drafts.append((req, proposed[:fit]))
            else:
                remaining.append(req)

        if not drafts:
            return [], remaining

        # Phase 2: build a verify EXTEND ModelInput. Each speculative request
        # contributes one stretch: the anchor (its last generated token) plus
        # its drafts, one contiguous span of cache rows the extend pass writes.
        slots, seq_starts, seq_lens, tokens = [], [], [], []
        block_writes: list[tuple[int, int, int, tuple[int, ...]]] = []
        for req, draft_ids in drafts:
            cur_seq_len = req.seq_len
            slots.append(req.slot)
            # The anchor's row, then one row per draft: rows seq_len-1 ..
            # seq_len+n-1, all of them this pass's to write.
            seq_starts.append(cur_seq_len - 1)
            seq_lens.append(cur_seq_len + len(draft_ids))
            tokens.append(req.output_token_ids[-1])
            tokens.extend(draft_ids)
            block_writes += [
                (req.slot, group_id, start_block, block_ids)
                for group_id, start_block, block_ids in req.block_plan
            ]

        verify_plan = ModelInput(
            kind=PassKind.EXTEND,
            slots=tuple(slots),
            seq_starts=tuple(seq_starts),
            seq_lens=tuple(seq_lens),
            tokens=tuple(tokens),
            # No sampled rows: the bonus comes from the argmax below, and a
            # sampled row here would drop a token nobody asked for into the
            # generation grid the repetition penalty reads.
            sampling=(),
            sampled=(),
            gen_counts=(),
            block_writes=tuple(block_writes),
            return_logits=True,
        )

        # Phase 3: execute the verify pass.
        _tokens, _records, all_logits = self._executor.execute_verify(verify_plan)

        if all_logits is None:
            # Executor produced no logits; fall back to normal decode.
            return [], decode_requests

        # Phase 4: check each draft against the model's argmax, then hand the
        # accepted span plus the bonus to _harvest — the same stop, length and
        # detokenise treatment a decode token gets.
        emitted: list[tuple[Request, int, PositionLogprobs | None]] = []
        logits_offset = 0
        for req, draft_ids in drafts:
            n_draft = len(draft_ids)
            # The anchor's row is logits[0]; draft j's row is logits[j + 1],
            # so the whole span needs n_draft + 1 rows of predictions.
            req_logits = all_logits[logits_offset : logits_offset + n_draft + 1]
            logits_offset += n_draft + 1

            accepted = 0
            for j in range(n_draft):
                if req_logits[j].argmax().item() != draft_ids[j]:
                    break
                accepted += 1
            bonus = req_logits[accepted].argmax().item()
            emitted.extend((req, token, None) for token in (*draft_ids[:accepted], bonus))

        # One entry per request: the stream layer publishes a chunk per
        # returned request, and this pass produces each request's whole step
        # in one delta. Request is not hashable (a mutable dataclass), so
        # dedup by identity — the same object appears once per token it
        # emitted.
        advanced = self._harvest(emitted)
        seen: set[int] = set()
        unique: list[Request] = []
        for request in advanced:
            if id(request) not in seen:
                seen.add(id(request))
                unique.append(request)
        return unique, remaining

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Run a whole prompt set through the scheduler and return the completions.

        Offline convenience wrapper: submits every prompt at once and drives
        :meth:`step` to exhaustion. A set exceeding ``max_num_seqs`` is admitted
        in waves, and short answers free their slots early.

        Returns:
            One :class:`~rapid_llm.engine.outputs.RequestOutput` per prompt, in
            submission order.
        """
        params = sampling_params or SamplingParams()
        if self._async_tokenize and len(prompts) > 1:
            # O10: encode releases the GIL, so the batch tokenises in parallel
            # — sum(encode_i) collapses to max(encode_i) before the first step.
            pool = self._ensure_tokenize_pool()
            futures = [
                pool.submit(self.tokenizer.encode, p, add_special_tokens=True) for p in prompts
            ]
            requests = [
                self.add_request(prompt, params, prompt_token_ids=future.result())
                for prompt, future in zip(prompts, futures, strict=True)
            ]
        else:
            requests = [self.add_request(prompt, params) for prompt in prompts]
        while self.has_unfinished_requests():
            self.step()
        return [
            RequestOutput(
                prompt=request.prompt,
                outputs=[
                    CompletionOutput(
                        0, request.text, request.finish_reason, logprobs=request.output_logprobs
                    )
                ],
                prompt_logprobs=request.prompt_logprobs,
            )
            for request in requests
        ]

    def shutdown(self) -> None:
        """Release the executor. The engine cannot serve any more steps after this."""
        # Unharvested passes are gone with the executor; their pinned buffers
        # belong to it, and their requests are nobody's to advance any more.
        self._inflight.clear()
        self._tokenizing.clear()
        if self._tokenize_pool is not None:
            self._tokenize_pool.shutdown(wait=False)
        self._executor.shutdown()
        # Break every reference this shell still holds into the engine's object
        # graph (executor -> worker -> model runner -> weights and KV cache).
        # The caller usually drops its last reference right after shutdown, but
        # a reference cycle would then wait for a gc pass: a second engine
        # built in this process would profile a KV budget of zero cache tokens
        # and crash on its first step. Nulling the links frees the weights and
        # the cache by refcount alone; empty_cache hands them back to the
        # driver so the profiler sees them as free.
        import gc

        self._executor = None
        self.engine = None
        gc.collect()
        if torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()

    def timeline_summary(self) -> str:
        """Stream region table of the steps run so far; empty unless tracing is on."""
        if self._executor is None:
            return ""
        return self._executor.timeline_summary()

    def _graph_manager(self):
        """The driver-side CUDA-graph manager, or ``None`` when absent.

        Test doubles stand in for the executor without a worker, so every hop
        of the chain is optional.
        """
        worker = getattr(self._executor, "_worker", None)
        runner = getattr(worker, "_runner", None)
        return getattr(runner, "_graph_manager", None)

    # ---------------------------------------------------------------- harvest #
    def _attribute_prompt_logprobs(
        self, work: _Work, prompt: tuple[tuple[PositionLogprobs, ...] | None, ...]
    ) -> None:
        """Place a chunk pass's prompt records on their requests, by position.

        Entry ``j`` of sequence ``i`` covers position ``seq_starts[i] + j + 1``.
        Position 0 and prefix-cache hits stay ``None``; the list is allocated
        at full prompt length on first contact, so chunks may land in any order.
        """
        for request, start, records in zip(
            work.chunk_requests, work.plan.seq_starts, prompt, strict=True
        ):
            if records is None:
                continue
            if request.prompt_logprobs is None:
                request.prompt_logprobs = [None] * request.prompt_len
            for j, record in enumerate(records):
                request.prompt_logprobs[start + j + 1] = record

    def _harvest(
        self, emitted: list[tuple[Request, int, PositionLogprobs | None]]
    ) -> list[Request]:
        """Read the step's tokens back, detokenise them, and retire whoever stopped.

        The only host-device synchronisation in the loop. It makes stop handling
        exact — a stop token retires the request on the next step, and the freed
        slot goes straight to a queued request.

        A step may hand one request several tokens (speculative decoding), so
        the loop accumulates: ``delta`` resets once per request and gathers
        every token's text, and a stop or length finish mid-chain retires the
        request right there — tokens after a stop belong to nobody, and are
        skipped rather than emitted.
        """
        now = time.monotonic()
        check_repeat = self._step_count % POLL_INTERVAL == 0
        advanced: list[Request] = []
        opened: set[str] = set()

        for request, token_id, record in emitted:
            if request.is_finished:
                # A token past this request's stop: the chain already retired
                # it, and nothing after a stop is output.
                continue
            if request.first_token_time is None:
                request.first_token_time = now
            if request.request_id not in opened:
                opened.add(request.request_id)
                request.delta = ""
                request.delta_logprobs = None

            if token_id in self.stop_token_ids:
                # The stop token is model punctuation, not output; the request
                # still belongs in this step's return — its stream has to hear
                # the finish reason (see step()).
                self._finish(request, "eos")
                advanced.append(request)
                continue

            request.output_token_ids.append(token_id)
            if record is not None:
                # Parallel to output_token_ids: one record per accepted token.
                if request.output_logprobs is None:
                    request.output_logprobs = []
                request.output_logprobs.append(record)
                request.delta_logprobs = record
            piece = self._detokenizers[request.request_id].append(0, token_id)
            request.delta += piece
            request.text += piece
            advanced.append(request)

            if not request.has_room or request.seq_len >= self.config.max_seq_len:
                self._finish(request, "length")
            elif check_repeat and request.params.stop_on_repeat and detect_repetition(request.text):
                self._finish(request, "repeat")

        return advanced

    def _finish(self, request: Request, reason: str) -> None:
        self.scheduler.finish(request, reason)
        self.metrics.observe_finish(request)
        self.tracer.end_span(
            self._spans.pop(request.request_id, None),
            finish_reason=reason,
            output_tokens=len(request.output_token_ids),
        )
        self._retire(request)

    def _retire(self, request: Request) -> None:
        """Drop the per-request state the engine owns; the caller keeps the handle."""
        self._detokenizers.pop(request.request_id, None)
