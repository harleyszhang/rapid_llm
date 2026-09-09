"""Request scheduling with chunked prefill and prefix caching.

Per step the :class:`Scheduler` decides which requests prefill (in chunks),
which decode, and which slot each holds, so a long prompt cannot stall
running decodes. Scheduling commits: once planned, a chunk is assumed to
have run, and the request objects are the single source of truth.

Usage:
    scheduler = Scheduler(SchedulerConfig(...), num_slots)
    scheduler.add_request(request); plan = scheduler.schedule()
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import StrEnum

from .prefix_cache import PREFIX_CACHE_BLOCK_SIZE, PrefixCache
from .sampler import PositionLogprobs, SamplingParams


class RequestStatus(StrEnum):
    """Where a request sits in its lifecycle.

    ``WAITING`` requests hold no cache slot; ``RUNNING`` ones own exactly one and
    are part of every step until they finish.
    """

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass(slots=True)
class Request:
    """One generation request, from arrival to completion.

    Carries both the inputs and everything the engine accumulates about it, so a
    caller holding a :class:`Request` can report progress without consulting the
    engine. ``delta`` is deliberately per-step scratch: the engine overwrites it
    each step and streaming callers drain it, while ``text`` keeps the whole
    completion for callers that only want the final answer.

    Attributes:
        request_id: Caller-visible identifier, unique among live requests.
        prompt: The prompt as submitted, kept for echoing back in the response.
        prompt_token_ids: Tokenised prompt.
        params: Per-request sampling configuration.
        max_new_tokens: Generation cap, resolved against the context window at
            admission so the scheduler never has to consult the engine.
        num_computed_tokens: Prompt tokens whose KV is already in the cache —
            the cached prefix plus every chunk scheduled so far. The next chunk
            of this request starts at exactly this offset.
        arrival_time: ``time.monotonic()`` when the request entered the queue.
        scheduled_time: When the request last left the queue for a slot — the
            queue wait is ``scheduled_time - arrival_time`` (re-admission after
            a preemption restarts it, since the request was waiting again).
        status: Lifecycle position.
        slot: Cache slot while running, ``None`` otherwise.
        output_token_ids: Tokens generated so far.
        text: Detokenised completion so far.
        delta: Text produced by the most recent step only.
        finish_reason: ``"eos"``, ``"length"``, ``"repeat"``, ``"abort"`` or
            ``"invalid"`` (background tokenize rejected the prompt, O10).
        first_token_time: When the first token became visible (for TTFT).
        finish_time: When the request finished.
        num_cached_tokens: Leading prompt tokens this request reused from the
            prefix cache instead of prefilling. Their K/V sits in physical
            blocks this request now *shares* with whoever computed them, so
            ``num_computed_tokens`` starts here rather than at 0 and nothing
            was copied to get there.
        block_plan: Per-step scratch: the block-table entries the executor must
            write before this step's pass reads them, as ``(group_id,
            start_block, block_ids)`` per KV cache group. Set on the step a
            request is admitted (its whole mapping, reused blocks included) and
            on any later step that grew it past a block boundary; empty on
            steps that need no new mapping.
        prompt_logprobs: Per-position records for the prompt, ``prompt_len``
            long once prefill completes; position 0 and prefix-cache hits stay
            ``None`` (their predictor never ran). Built chunk by chunk; ``None``
            until the first chunk of a request that asked arrives, ``None``
            forever when it did not.
        output_logprobs: Per-token records for the generated span, parallel to
            ``output_token_ids``; ``None`` when the request did not ask.
        delta_logprobs: Per-step scratch like ``delta``: the record of the token
            this step produced, drained by the streaming layer. ``None`` on
            steps with no new token, and for requests that did not ask.
        pending_tokens: Tokens whose pass has launched but whose harvest has
            not run — the optimistic half of the launch/harvest pipeline. Zero
            forever in the synchronous engine; one while the pipeline is full,
            which is exactly the gap between what the device has and what the
            host has harvested, and what the next decode plan adds back to the
            request's bookkeeping to keep writing the right cache rows.
        error: The exception that rejected this request before it reached the
            scheduler — a background-tokenize failure or an empty/over-long
            prompt (O10). ``None`` on every request that actually ran; the
            synchronous path raises from ``add_request`` instead.
    """

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    params: SamplingParams
    max_new_tokens: int = 0
    num_computed_tokens: int = 0
    arrival_time: float = field(default_factory=time.monotonic)
    scheduled_time: float | None = None
    status: RequestStatus = RequestStatus.WAITING
    slot: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    text: str = ""
    delta: str = ""
    finish_reason: str | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    num_cached_tokens: int = 0
    block_plan: tuple[tuple[int, int, tuple[int, ...]], ...] = ()
    prompt_logprobs: list[PositionLogprobs | None] | None = None
    output_logprobs: list[PositionLogprobs] | None = None
    delta_logprobs: PositionLogprobs | None = None
    pending_tokens: int = 0
    error: BaseException | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def seq_len(self) -> int:
        """Tokens currently in this request's KV cache."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status is RequestStatus.FINISHED

    @property
    def has_room(self) -> bool:
        """Whether the generation cap still allows another token."""
        return len(self.output_token_ids) < self.max_new_tokens

    @property
    def prefill_done(self) -> bool:
        """Whether the whole prompt is in the cache, i.e. decode may proceed."""
        return self.num_computed_tokens >= self.prompt_len


DEFAULT_MAX_NUM_SEQS = 32
DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MAX_CHUNK_SIZE = 512


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Limits the admission policy enforces.

    Attributes:
        max_seq_len: Context window; also the per-slot cache capacity.
        max_num_seqs: Ceiling on concurrently running requests.
        max_num_batched_tokens: Ceiling on the padded token count of one
            prefill group. Measured on *chunks*, not whole prompts: with
            chunked prefill a step's grid is as wide as its longest chunk.
        enable_chunked_prefill: Whether one prompt may prefill across several
            steps. On by default, as in vLLM: a long prompt is split into
            ``max_chunk_size`` pieces that interleave with decode, so a single
            32k prefill cannot stall every running request. Off means each
            prompt prefills in one pass, which needs a token budget wide
            enough to hold the longest context — ``max_num_batched_tokens >=
            max_seq_len``, validated here the way vLLM validates it. Passing
            ``max_chunk_size=0`` is the older spelling of off and still turns
            this flag off, so there is one source of truth to read.
        max_chunk_size: Maximum tokens per prefill chunk, once chunking is on.
            A prompt longer than this is split into chunks interleaved with
            decode steps. Ignored when ``enable_chunked_prefill`` is False.
        enable_prefix_cache: Whether to reuse blocks across sequences by block
            hash. Blocks are paged and reference-counted either way; this only
            decides whether a completed block is *indexed* so another sequence
            can share it.
        prefix_cache_blocks: Override for the physical block-pool size, in
            blocks of :data:`~rapid_llm.engine.prefix_cache.PREFIX_CACHE_BLOCK_SIZE`
            tokens. ``None`` takes the executor's real cache size, falling back
            to a bound derived from the slot geometry (see
            :meth:`Scheduler._default_num_blocks`).
        enable_preemption: When True, ``max_num_seqs`` is honoured as a desired
            concurrency even beyond the slot count: an oversubscribed batch
            time-shares slots by preempting (recompute) the youngest running
            request to admit an older waiting one. When False (default) the
            batch is capped at the slot count and nothing is ever evicted.
        decode_window_steps: How many pure-decode steps a fresh prompt waits
            at most before it may interrupt them (O9 decode window). ``0`` (the
            default) admits immediately — the honest baseline. ``N > 0`` trades
            a little TTFT for decode smoothness: while decodes are running and
            no chunked prefill is already in flight, admission is deferred up
            to ``N`` steps, so a burst of prompts cannot stretch every running
            request's step back-to-back. Steps carrying resumed chunks never
            defer — the interruption is already paid.
    """

    max_seq_len: int = 2048
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS
    enable_chunked_prefill: bool = True
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
    enable_prefix_cache: bool = False
    prefix_cache_blocks: int | None = None
    enable_preemption: bool = False
    decode_window_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_seq_len < 2:
            raise ValueError(f"max_seq_len must be >= 2, got {self.max_seq_len}")
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be >= 1, got {self.max_num_seqs}")
        if self.max_num_batched_tokens < 1:
            raise ValueError(
                f"max_num_batched_tokens must be >= 1, got {self.max_num_batched_tokens}"
            )
        if self.max_chunk_size < 0:
            raise ValueError(f"max_chunk_size must be >= 0, got {self.max_chunk_size}")
        # ``max_chunk_size=0`` predates the flag; fold it in so the scheduler
        # only ever has to consult ``enable_chunked_prefill``.
        if self.max_chunk_size == 0:
            object.__setattr__(self, "enable_chunked_prefill", False)
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_seq_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is smaller than "
                f"max_seq_len ({self.max_seq_len}) with chunked prefill disabled: a whole "
                "prompt must fit one prefill pass. Raise max_num_batched_tokens, lower "
                "max_seq_len, or leave chunked prefill on."
            )
        # Two is the floor rather than one: block 0 is the reserved null block.
        if self.prefix_cache_blocks is not None and self.prefix_cache_blocks < 2:
            raise ValueError(
                f"prefix_cache_blocks must be >= 2 or None, got {self.prefix_cache_blocks}"
            )
        if self.decode_window_steps < 0:
            raise ValueError(f"decode_window_steps must be >= 0, got {self.decode_window_steps}")


@dataclass(frozen=True, slots=True)
class SchedulerOutput:
    """What one engine step should run.

    Prefill and decode coexist in one step: the engine executes the prefill
    pass, then the decode pass, so chunked prefill never stalls decode.

    Attributes:
        prefill: Requests receiving prefill work this step, in batch order.
        decode: Requests receiving one decode token this step.
        prefill_chunk_lens: Tokens to process per prefill request. A request
            whose chunk completes its prompt produces its first sampled token
            this step; a partial chunk produces none.
        preempted: Requests evicted this step (for logging/metrics).
    """

    prefill: list[Request] = field(default_factory=list)
    decode: list[Request] = field(default_factory=list)
    prefill_chunk_lens: list[int] = field(default_factory=list)
    preempted: list[Request] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode


class Scheduler:
    """FCFS admission with chunked prefill and preemption support.

    The scheduling step is a three-stage template (mirrors vLLM v1's
    ``schedule``): resume in-flight partial prefills, admit queued requests
    into whatever budget and slots remain, then collect the decode batch.
    Every stage advances ``num_computed_tokens`` on the request itself as it
    commits work, which is why no external ``advance`` protocol exists.

    The scheduler is also the KV cache's allocator: every stage that plans work
    first reserves the physical blocks that work will write, through the
    :class:`~rapid_llm.engine.prefix_cache.PrefixCache`. That is what makes
    reuse free — a shared prefix is a shared block, and the ``block_plan`` each
    request carries is the executor's instruction to point its table rows at it.

    Args:
        config: Admission and chunking limits.
        num_slots: Cache slots available.
        num_blocks: Physical KV blocks the executor allocated, the null block
            included. ``None`` derives a bound from the slot geometry (see
            :meth:`_default_num_blocks`), which is what a scheduler driven
            without an executor gets.
    """

    def __init__(
        self, config: SchedulerConfig, num_slots: int, num_blocks: int | None = None
    ) -> None:
        if num_slots < 1:
            raise ValueError(f"need at least one cache slot, got {num_slots}")
        self.config = config
        self.num_slots = num_slots
        # Preemption lets the running set exceed the slot count (slots are
        # time-shared via recompute); otherwise concurrency is slot-capped.
        self.max_num_seqs = (
            config.max_num_seqs if config.enable_preemption else min(config.max_num_seqs, num_slots)
        )

        # Ordered map = FCFS iteration plus constant-time cancellation. Serving
        # queues can be much deeper than the running batch, so a deque scan on
        # every disconnected client becomes measurable under overload.
        self._waiting: OrderedDict[str, Request] = OrderedDict()
        self._running: list[Request] = []

        self._requests: dict[str, Request] = {}
        self._free_slots: list[int] = list(reversed(range(num_slots)))

        # One allocator whether or not reuse is on: paging is how a request gets
        # its rows at all now, so "prefix caching off" only switches off the hash
        # index. That keeps admission on a single code path instead of a
        # branch-per-stage null object.
        self._prefix_cache = PrefixCache(
            num_blocks=(
                config.prefix_cache_blocks or num_blocks or self._default_num_blocks(num_slots)
            ),
            block_size=PREFIX_CACHE_BLOCK_SIZE,
            enable_caching=config.enable_prefix_cache,
        )

        # Blocks whose K/V a planned pass will write, awaiting the step that
        # proves it ran; see :meth:`_commit_pending_blocks`.
        self._pending_blocks: list[tuple[Request, int]] = []
        self.num_preemptions: int = 0
        # Pure-decode steps deferred since the first request started waiting
        # (O9 decode window): the wait is bounded by ``decode_window_steps``.
        self._deferred_steps: int = 0

    def _default_num_blocks(self, num_slots: int) -> int:
        """Pool size to use when the executor did not report its cache size.

        Enough blocks for every slot to hold a full context — the capacity the
        fixed-slot layout used to reserve outright. Paging pays off precisely
        when the pool is *smaller* than this, but a scheduler built without an
        executor (every unit test) should not be where that is discovered.
        """
        return max(2, num_slots * self.config.max_seq_len // PREFIX_CACHE_BLOCK_SIZE + 1)

    # ------------------------------------------------------------------ queue #
    def add_request(self, request: Request) -> None:
        """Queue a request, resolving its generation cap against the context window.

        Raises:
            ValueError: The prompt is empty, or leaves no room to generate.
        """
        if request.request_id in self._requests:
            raise ValueError(f"request id {request.request_id!r} is already active")
        if request.status is not RequestStatus.WAITING or request.slot is not None:
            raise ValueError("only a fresh waiting request can be submitted")

        limit = self.config.max_seq_len
        if request.prompt_len == 0:
            raise ValueError(f"request {request.request_id} has an empty prompt")
        if request.prompt_len >= limit:
            raise ValueError(
                f"request {request.request_id} prompt length {request.prompt_len} "
                f"leaves no room under max_seq_len {limit}"
            )

        room = limit - request.prompt_len
        requested = request.params.max_gen_len
        request.max_new_tokens = min(requested, room) if requested else room
        self._requests[request.request_id] = request
        self._waiting[request.request_id] = request

    def has_request_id(self, request_id: str) -> bool:
        """Whether ``request_id`` belongs to a live scheduler request.

        The engine owns a few request states that have not reached the
        scheduler yet (notably background tokenisation), but the scheduler is
        authoritative once admission succeeds.  Exposing this narrow query
        keeps callers from reaching into ``_requests`` merely to protect the
        public request-id namespace.
        """
        return request_id in self._requests

    def abort(self, request_id: str) -> Request | None:
        """Drop a request wherever it is; returns it, or ``None`` if unknown.

        A running request releases its slot immediately, so an abandoned HTTP
        connection frees capacity on the next step rather than at its length cap.
        """
        request = self._requests.get(request_id)
        if request is None:
            return None
        if request.status is RequestStatus.WAITING:
            self._waiting.pop(request_id, None)
            request.status = RequestStatus.FINISHED
            request.finish_reason = "abort"
            request.finish_time = time.monotonic()
            self._prefix_cache.free(request_id)
            self._requests.pop(request_id, None)
        else:
            self.finish(request, "abort")
        return request

    # -------------------------------------------------------------- scheduling #
    def schedule(self) -> SchedulerOutput:
        """Decide this step's work: chunked prefill + decode batch.

        Prefill and decode coexist: partial prefill chunks for some requests
        and decode tokens for the rest run in the same step, so a long prompt
        stretches over several steps without freezing anyone's decode. Chunk
        progress is committed here — the returned output describes work already
        accounted for in each request's ``num_computed_tokens``.
        """
        prefill: list[Request] = []
        chunk_lens: list[int] = []
        preempted: list[Request] = []

        # Stage 0: last step's passes have executed by now, so the blocks they
        # filled really hold their K/V and may be offered to other requests.
        self._commit_pending_blocks()

        # Stage 1: resume requests whose prefill is still in flight. Their
        # slot and blocks are held, so they come before new admissions — FCFS by
        # arrival, and re-admitting a new request first would starve them.
        for request in self._running:
            if request.prefill_done:
                continue
            # A mapping belongs to the step that created it; a resumed chunk
            # emits only whatever blocks it just grew into.
            request.block_plan = ()
            chunk = self._chunk_of(request)
            reach = request.num_computed_tokens + chunk
            if not self._reserve(request, reach, prefill, preempted):
                # No blocks for the next chunk and nobody left to evict: the
                # request keeps its slot and its progress, and retries next step.
                continue
            prefill.append(request)
            chunk_lens.append(chunk)
            request.num_computed_tokens = reach
            self._map_blocks(request)
            self._track_pending(request, reach)

        # Stage 2: admit queued requests into the remaining budget and slots.
        self._admit(prefill, chunk_lens, preempted)

        # Stage 3: decode every request that finished prefill before this
        # step. A prompt completed *this* step is sampled in its prefill pass
        # and joins decode on the next one; partial prefills never pass the
        # ``prefill_done`` gate.
        just_prefilled = {id(request) for request in prefill if request.prefill_done}
        decode = [
            request
            for request in self._running
            if request.prefill_done and id(request) not in just_prefilled
        ]
        self._grow_decode(decode, prefill, preempted)

        return SchedulerOutput(
            prefill=prefill, decode=decode, prefill_chunk_lens=chunk_lens, preempted=preempted
        )

    def _grow_decode(
        self, decode: list[Request], prefill: list[Request], preempted: list[Request]
    ) -> None:
        """Reserve the block each decode row's token will be written into.

        A decode step writes one row per request, and every sixteenth of them
        starts a new block — so this is where a running sequence grows its
        allocation, and where a full pool is felt. Requests that cannot be grown
        are preempted and dropped from the batch (``decode`` is filtered in
        place), because a row with nowhere to write is not a step it can take.
        """
        limit = self.config.max_seq_len
        granted = list(prefill)
        for request in list(decode):
            if request.status is not RequestStatus.RUNNING:
                continue
            request.block_plan = ()
            # The token this step produces lands at position ``seq_len``, plus
            # whatever the pipeline launched and the host has not harvested.
            # Clamped: a request at the context limit writes no further row.
            reach = min(request.seq_len + request.pending_tokens + 1, limit)
            if self._reserve(request, reach, granted, preempted):
                self._map_blocks(request)
                self._track_pending(request, min(request.seq_len, limit))
                granted.append(request)
                continue
            self._preempt(request)
            preempted.append(request)
        if preempted:
            decode[:] = [r for r in decode if r.status is RequestStatus.RUNNING]

    def _admit(
        self, prefill: list[Request], chunk_lens: list[int], preempted: list[Request]
    ) -> None:
        """Admit waiting requests, committing their first chunk (or whole prompt).

        A newly admitted request's first chunk is its uncached remainder capped
        at ``max_chunk_size``; short prompts therefore finish prefill in this
        very step, while long ones re-enter through Stage 1 on later steps.

        Preempted victims are appended to *preempted*, for reporting.
        """
        longest = max(chunk_lens, default=0)

        if self._defer_admission(prefill):
            self._deferred_steps += 1
            return
        self._deferred_steps = 0

        while self._waiting:
            if len(self._running) >= self.max_num_seqs:
                break
            candidate = next(iter(self._waiting.values()))

            chunk = self._chunk_of(candidate, computed=0)
            padded = max(longest, chunk) * (len(prefill) + 1)
            if prefill and padded > self.config.max_num_batched_tokens:
                break

            if not self._free_slots:
                victim = self._maybe_preempt(prefill)
                if victim is None:
                    break
                preempted.append(victim)

            # Prefix cache: find the longest prefix already sitting in physical
            # blocks, then take a *reference* on those very blocks — reuse is a
            # shared block, not a copy of its rows. Never the whole prompt: at
            # least one token must run to produce the first logits, exactly as
            # vLLM keeps the last block uncached.
            rid = candidate.request_id
            hashes = self._prefix_cache.track(rid, candidate.prompt_token_ids)
            match = self._prefix_cache.lookup(hashes, candidate.prompt_len)
            chunk = self._chunk_of(candidate, computed=match.num_tokens)
            if not self._prefix_cache.allocate(rid, match.num_tokens + chunk, match):
                # Nothing was allocated, so admission simply waits for the pool
                # to drain — having first offered a victim towards that.
                self._prefix_cache.free(rid)
                victim = self._maybe_preempt(prefill)
                if victim is not None:
                    preempted.append(victim)
                break

            self._waiting.pop(rid)
            candidate.slot = self._free_slots.pop()
            candidate.status = RequestStatus.RUNNING
            candidate.scheduled_time = time.monotonic()
            self._running.append(candidate)

            candidate.num_cached_tokens = match.num_tokens
            candidate.num_computed_tokens = match.num_tokens + chunk
            self._map_blocks(candidate)

            prefill.append(candidate)
            chunk_lens.append(chunk)
            self._track_pending(candidate, candidate.num_computed_tokens)
            longest = max(longest, chunk)

            if preempted:
                break

    def _defer_admission(self, resume_prefill: list[Request]) -> bool:
        """Whether this step stays pure-decode instead of admitting new prefills.

        Only a step that would otherwise be pure decode is worth protecting:
        someone is mid-generation and a fresh prefill would stretch their step.
        With no decode in flight, or a chunk already resuming this step, waiting
        would only delay the first token with nothing to smooth.
        """
        window = self.config.decode_window_steps
        if window <= 0 or not self._waiting or resume_prefill:
            return False
        if self._deferred_steps >= window:
            return False
        return any(request.prefill_done for request in self._running)

    def _chunk_of(self, request: Request, computed: int | None = None) -> int:
        """Next chunk to run, chunk- and budget-capped.

        ``computed=0`` prices a fresh admission from progress zero — an upper
        bound, since how much the prefix cache will cover is only known once
        the request holds a slot. Under-pricing would admit a group that
        overflows ``max_num_batched_tokens``.
        """
        progress = request.num_computed_tokens if computed is None else computed
        remaining = request.prompt_len - progress
        if not self.config.enable_chunked_prefill:
            return remaining
        size = self.config.max_chunk_size
        # vLLM's chunked prefill consumes at most the iteration token budget.
        # Without this cap, one long request violates the advertised ceiling.
        return min(remaining, size, self.config.max_num_batched_tokens)

    # ---------------------------------------------------------------- blocks #
    def _reserve(
        self,
        request: Request,
        num_tokens: int,
        protect: list[Request],
        victims: list[Request],
    ) -> bool:
        """Grow *request*'s block coverage to *num_tokens*, evicting if it must.

        Preemption here is a mechanism rather than the ``enable_preemption``
        policy: a request that already holds a slot has nowhere else to put the
        rows its next tokens need, so evicting somebody is the only alternative
        to deadlock. It stays unreachable while the pool is sized for every slot
        to hold a full context, which is the default.

        Args:
            request: The request to grow. Never itself a victim.
            num_tokens: Tokens its blocks must cover after this call.
            protect: Requests already granted work this step; evicting one would
                pull the blocks out from under a pass that is about to run.
            victims: Extended with whoever was evicted, for reporting.

        Returns:
            Whether the request's blocks now cover ``num_tokens``.
        """
        protected = {id(candidate) for candidate in protect}
        protected.add(id(request))
        while not self._prefix_cache.allocate(request.request_id, num_tokens):
            victim = self._preempt_victim(protected)
            if victim is None:
                return False
            victims.append(victim)
        return True

    def _map_blocks(self, request: Request) -> None:
        """Record the block-table entries the executor owes this request.

        This is the whole device-side cost of prefix reuse: pointing table rows
        at blocks somebody else filled, with no K/V moved. The cache emits each
        block once, so a steady decode carries an empty plan fifteen steps out of
        sixteen and a boundary-crossing one carries a single block.
        """
        request.block_plan = self._prefix_cache.take_table_writes(request.request_id)

    def _commit_pending_blocks(self) -> None:
        """Index the blocks last step's passes actually filled.

        Registration deliberately lags one step behind planning:
        ``num_computed_tokens`` advances when a chunk is *planned*, and a block
        offered to the next admission before the model wrote its K/V would be
        read as though it held the prefix. Generated tokens ride the same path,
        which is what makes a finished answer reusable by its own follow-up turn.
        """
        for request, upto in self._pending_blocks:
            if request.status is RequestStatus.RUNNING:
                self._commit(request, upto)
        self._pending_blocks.clear()

    def _commit(self, request: Request, upto: int) -> None:
        """Hash and index whatever full blocks of this request are now computed."""
        rid = request.request_id
        if upto // PREFIX_CACHE_BLOCK_SIZE > len(self._prefix_cache.block_hashes(rid)):
            # Generated tokens completed a block: extend the chain over them.
            # Skipped fifteen steps out of sixteen, which is what keeps decode
            # caching close to free.
            self._prefix_cache.observe(rid, request.prompt_token_ids + request.output_token_ids)
        self._prefix_cache.commit(rid, upto)

    def _track_pending(self, request: Request, upto: int) -> None:
        """Queue a registration for the rows the step just planned will write.

        Per chunk, not per completed prompt: a shareable prefix is usually long
        enough to be split, and waiting for the whole prompt would leave its
        blocks unindexed exactly while the requests that would reuse them arrive.
        """
        self._pending_blocks.append((request, upto))

    def _settle_pending_blocks(self, request: Request) -> None:
        """Cash in a retiring request's registration instead of discarding it.

        Retirement runs after the step executed, so its last blocks really do
        hold their K/V — and the request that first pays for a shared prompt
        typically finishes before the ones that would inherit it arrive.
        """
        kept: list[tuple[Request, int]] = []
        for candidate, upto in self._pending_blocks:
            if candidate is request:
                self._commit(request, upto)
            else:
                kept.append((candidate, upto))
        self._pending_blocks = kept

    def _drop_pending_blocks(self, request: Request) -> None:
        """Cancel a queued registration, by identity.

        Mandatory for preemption: it moves generated tokens into
        ``prompt_token_ids``, so a replayed registration would hash the *new*
        prompt onto blocks holding the old sequence's K/V.
        """
        self._pending_blocks = [entry for entry in self._pending_blocks if entry[0] is not request]

    def _maybe_preempt(self, prefill: list[Request]) -> Request | None:
        """Evict the youngest eligible running request, if the policy allows it.

        This is the *policy* gate, used where preemption buys concurrency rather
        than correctness: a deployment that left ``enable_preemption`` off gets
        no eviction, and admission simply waits.
        """
        if not self.config.enable_preemption:
            return None
        return self._preempt_victim({id(request) for request in prefill})

    def _preempt_victim(self, protect: set[int]) -> Request | None:
        """Evict the youngest eligible running request, or ``None``.

        Eligible = past prefill, not already granted work this step, and past
        the progress quantum (>=1 output token) — without the quantum a
        just-recomputed request could be evicted again before making progress.
        """
        for request in reversed(self._running):
            if not request.prefill_done or id(request) in protect:
                continue
            if request.output_token_ids:
                self._preempt(request)
                return request
        return None

    def _preempt(self, request: Request) -> None:
        """Evict a running request back to the waiting queue (recompute strategy).

        The KV built so far is dropped. Generated tokens move into the prompt
        — vLLM v1's recompute semantics — so re-admission replays them and
        decoding continues the text the caller already saw. The generation
        cap shrinks by the moved tokens.
        """
        if request.slot is not None:
            self._free_slots.append(request.slot)
            request.slot = None
        request.status = RequestStatus.WAITING
        request.num_computed_tokens = 0
        request.num_cached_tokens = 0
        request.block_plan = ()
        # The optimistic ledger follows the real one: re-admission replays the
        # prompt through prefill, so nothing is launched-and-unharvested any
        # more. (The engine refuses pipeline + preemption together; this line
        # is the belt to that braces.)
        request.pending_tokens = 0
        # The prompt changes underneath the logprob records (generated tokens
        # move into it), so both spans restart empty for re-admission.
        request.prompt_logprobs = None
        request.output_logprobs = None
        request.delta_logprobs = None
        self._drop_pending_blocks(request)
        # Give the blocks back. Skipping this would pin one set of rows per
        # preempt cycle, and a referenced block never evicts.
        self._prefix_cache.free(request.request_id)
        moved = len(request.output_token_ids)
        request.prompt_token_ids.extend(request.output_token_ids)
        request.output_token_ids.clear()
        request.max_new_tokens -= moved
        self._discard_running(request)
        self._waiting[request.request_id] = request
        self._waiting.move_to_end(request.request_id, last=False)
        self.num_preemptions += 1

    def finish(self, request: Request, reason: str) -> None:
        """Retire a request and release whatever resources it holds."""
        if request.status is RequestStatus.FINISHED:
            return
        was_running = request.status is RequestStatus.RUNNING
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        request.finish_time = time.monotonic()
        if request.slot is not None:
            self._free_slots.append(request.slot)
            request.slot = None
        if was_running:
            self._discard_running(request)
            self._settle_pending_blocks(request)
        else:
            self._waiting.pop(request.request_id, None)
        # A shared prefix survives as long as another live request references it;
        # a queued request may still hold the hash chain of an admission the pool
        # refused, and freeing by id leaves nothing behind either way.
        self._prefix_cache.free(request.request_id)
        if self._requests.get(request.request_id) is request:
            self._requests.pop(request.request_id)

    def _discard_running(self, request: Request) -> None:
        """Remove ``request`` from the running list by identity.

        :class:`Request` is value-equal, so ``list.remove`` could drop a
        *different* request with matching fields.
        """
        for index, candidate in enumerate(self._running):
            if candidate is request:
                del self._running[index]
                return

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Fraction of queried prompt tokens served from the prefix cache."""
        return self._prefix_cache.hit_rate

    @property
    def num_free_blocks(self) -> int:
        """Physical KV blocks available right now."""
        return self._prefix_cache.num_free_blocks

    @property
    def kv_cache_utilization(self) -> float:
        """Fraction of the block pool live requests hold (0.0 empty, 1.0 full)."""
        return self._prefix_cache.utilization

    @property
    def running(self) -> list[Request]:
        """Requests in the decode batch, in admission order."""
        return list(self._running)

    @property
    def waiting(self) -> list[Request]:
        """Queued requests, in arrival order."""
        return list(self._waiting.values())

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def has_unfinished_requests(self) -> bool:
        """Whether anything is queued or in flight."""
        return bool(self._waiting or self._running)
