"""Data parallelism: N whole-model replicas, one process each, routed between.

:class:`DataParallelEngine` is the parent half: it spawns one ``_dp_worker``
process per replica, routes every request through a
:class:`~rapid_llm.engine.dp_load_balancer.LoadBalancer`, and merges the
replicas' results into one ordered output stream.

Usage:
    engine = DataParallelEngine(model, data_parallel_size=2)
    outs = engine.generate(prompts, sampling_params)
"""

from __future__ import annotations

import contextlib
import queue
import time
import traceback
from typing import TYPE_CHECKING, Any

import torch.multiprocessing as mp

from ..utils.logger import get_logger
from .dp_load_balancer import LOAD_BALANCERS, make_load_balancer
from .outputs import CompletionOutput, RequestOutput
from .sampler import SamplingParams
from .scheduler import DEFAULT_MAX_CHUNK_SIZE, DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS

if TYPE_CHECKING:  # pragma: no cover - the worker imports these in its own process
    from .continuous_engine import ContinuousBatchingEngine

_log = get_logger(__name__)

#: Seconds a replica is given to import torch, load the checkpoint and profile its
#: KV cache. Loading a 30B FP8 checkpoint from a cold page cache is minutes, not
#: seconds, so this is deliberately generous — it exists to turn a wedged worker
#: into an error rather than a hang.
STARTUP_TIMEOUT_S = 900.0

#: How often a blocked wait looks up from the queue to check the workers are alive.
#: Generation has no timeout — a long prompt legitimately takes as long as it takes —
#: so liveness, not the clock, is what distinguishes "still working" from "died".
_LIVENESS_POLL_S = 5.0

#: Sentinel that tells a replica to leave its engine loop and exit.
_SHUTDOWN = None


class _ReplicaLoop:
    """A replica's resident engine loop: commands in, progress out.

    This is what separates a replica from a call to ``generate()``. The engine
    lives across dispatches, so a batch is no longer a barrier: the loop admits
    whatever has arrived between steps, and a sequence that stops frees its slot to
    the next request on the following step instead of padding the batch out to the
    longest answer in it. Idle, it blocks on the queue rather than spinning.

    One loop serves two front ends, told apart by the inbound message's leading
    tag rather than a constructor flag — a replica must not care which kind of
    coordinator owns it:

    * **batch** (``("batch", batch_id, indices, prompts, params)``) is
      :meth:`DataParallelEngine.generate`'s unit of work, and is answered as a
      whole — exactly one ``("done" | "error", batch_id, payload)`` message —
      because that is what the dispatching side counts.
    * **streaming** (``("add", request_id, prompt, params)``, cancelled by
      ``("abort", request_id)``) forwards one online request at a time and is
      answered per request: a ``("delta", request_id, ...)`` per step and a final
      ``("finished" | "failed", request_id, ...)``. Same bookkeeping as the batch
      path with a different grouping, which is why they share this loop.

    Args:
        engine: This replica's :class:`ContinuousBatchingEngine`. Under TP it is the
            leader's, and its executor broadcasts every plan to the follower ranks.
        requests: Inbound command tuples (see above), or :data:`_SHUTDOWN`.
        results: Outbound ``("done" | "error", batch_id, payload)`` and
            ``("delta" | "finished" | "failed", request_id, ...)`` messages.
    """

    def __init__(
        self,
        engine: ContinuousBatchingEngine,
        requests: mp.Queue,
        results: mp.Queue,
    ) -> None:
        self._engine = engine
        self._requests = requests
        self._results = results
        self._batch_of: dict[str, tuple[int, int]] = {}
        self._solo: set[str] = set()
        self._live: dict[str, Any] = {}
        self._left: dict[int, int] = {}
        self._answers: dict[int, list[tuple[int, str, str | None]]] = {}

    def run(self) -> None:
        """Serve until the coordinator sends :data:`_SHUTDOWN` and the work drains.

        A stop signal that arrives mid-batch stops *admission*, not the batch: the
        requests already in flight are worth the steps they have already cost, and
        the coordinator is waiting for their answers.
        """
        stopping = False
        while True:
            idle = not self._engine.has_unfinished_requests()
            if stopping and idle:
                return
            if not stopping:
                stopping = self._take_arrivals(block=idle)
            if self._engine.has_unfinished_requests():
                self._step()

    def _take_arrivals(self, block: bool) -> bool:
        """Admit everything waiting on the queue; ``True`` if asked to stop.

        Blocking is conditional on having nothing to run: an idle replica must not
        spin a core, and a busy one must not stall its decode waiting for work that
        may never come. The first message may block, the rest never do.
        """
        while True:
            try:
                message = self._requests.get(block=block)
            except queue.Empty:
                return False
            if message is _SHUTDOWN:
                return True
            self._handle(message)
            block = False

    def _handle(self, message: tuple) -> None:
        """Dispatch one inbound command by its leading tag."""
        kind = message[0]
        if kind == "batch":
            self._admit_batch(message[1], message[2], message[3], message[4])
        elif kind == "add":
            self._admit_one(message[1], message[2], message[3])
        elif kind == "abort":
            self._abort_one(message[1])

    def _admit_batch(
        self, batch_id: int, indices: list[int], prompts: list[str], params: SamplingParams
    ) -> None:
        """Turn one dispatched batch into engine requests, remembering who is who."""
        self._left[batch_id] = len(prompts)
        self._answers[batch_id] = []
        for index, prompt in zip(indices, prompts, strict=True):
            try:
                request = self._engine.add_request(prompt, params)
            except ValueError:
                # An unservable prompt (empty, or longer than the context window)
                # fails its batch rather than being silently answered with "":
                # the caller asked for a completion and there is none.
                self._fail(batch_id, traceback.format_exc())
                return
            self._batch_of[request.request_id] = (batch_id, index)
            self._live[request.request_id] = request

    def _admit_one(self, request_id: str, prompt: str, params: SamplingParams) -> None:
        """Admit one streamed request under the caller's own id.

        A refused prompt fails that request alone — the streaming front end has a
        caller waiting on exactly this id, so the failure unit is the request, not
        some batch it happens to travel with.
        """
        try:
            request = self._engine.add_request(prompt, params, request_id=request_id)
        except ValueError:
            self._results.put(("failed", request_id, traceback.format_exc()))
            return
        self._solo.add(request_id)
        self._live[request_id] = request

    def _abort_one(self, request_id: str) -> None:
        """Cancel one streamed request; its slot is free for the next step.

        No acknowledgement is sent: the consumer that asked for the abort has
        already stopped reading, and any ``finished`` message racing this command
        lands on a stream nobody holds — the coordinator drops it by id.
        """
        self._engine.abort(request_id)
        self._forget(request_id)

    def _step(self) -> None:
        """Run one engine step, then report whoever finished on it.

        ``step`` returns the requests it *advanced*, which excludes the ones that
        just stopped — a sequence's stop token is not output. So completion is read
        off the handles this loop holds, which the engine updates in place.
        Streamed requests additionally report every delta the step produced: their
        consumer is per-request, so waiting for a batch boundary would be waiting
        for a grouping that does not exist.
        """
        try:
            advanced = self._engine.step()
        except Exception:
            detail = traceback.format_exc()
            for batch_id in list(self._left):
                self._fail(batch_id, detail)
            for request_id in list(self._solo):
                self._fail_one(request_id, detail)
            return

        for request in advanced:
            if request.request_id in self._solo:
                self._results.put(
                    (
                        "delta",
                        request.request_id,
                        request.delta,
                        request.text,
                        request.prompt_len,
                        len(request.output_token_ids),
                    )
                )

        for request_id, request in list(self._live.items()):
            if not request.is_finished:
                continue
            if request_id in self._solo:
                self._results.put(
                    (
                        "finished",
                        request_id,
                        request.finish_reason,
                        request.text,
                        request.prompt_len,
                        len(request.output_token_ids),
                    )
                )
                self._forget(request_id)
                continue
            batch_id, index = self._batch_of[request_id]
            self._answers[batch_id].append((index, request.text, request.finish_reason))
            self._forget(request_id)
            self._left[batch_id] -= 1
            if self._left[batch_id] == 0:
                del self._left[batch_id]
                self._results.put(("done", batch_id, self._answers.pop(batch_id)))

    def _fail(self, batch_id: int, detail: str) -> None:
        """Abandon a batch: abort what is still running and report it once.

        Aborting matters as much as reporting — a request left in the scheduler
        would hold its cache slot forever, and this replica has to keep serving the
        batches that did not fail.
        """
        for request_id, (batch, _index) in list(self._batch_of.items()):
            if batch == batch_id:
                self._engine.abort(request_id)
                self._forget(request_id)
        self._left.pop(batch_id, None)
        self._answers.pop(batch_id, None)
        self._results.put(("error", batch_id, detail))

    def _fail_one(self, request_id: str, detail: str) -> None:
        """Abandon one streamed request: abort it, forget it, report it once."""
        self._engine.abort(request_id)
        self._forget(request_id)
        self._results.put(("failed", request_id, detail))

    def _forget(self, request_id: str) -> None:
        self._batch_of.pop(request_id, None)
        self._solo.discard(request_id)
        self._live.pop(request_id, None)


def _dp_worker(
    global_rank: int,
    dp_rank: int,
    tp_rank: int,
    dp_size: int,
    tp_size: int,
    engine_kwargs: dict[str, Any],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
    max_chunk_size: int,
    enable_prefix_cache: bool,
    prefix_cache_blocks: int | None,
    enable_preemption: bool,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """One cell of the ``dp_size x tp_size`` grid, in its own process on its own GPU.

    There is one process per cell, not per replica: with ``tp_size > 1``
    ``init_parallel`` rendezvouses a world of ``dp_size * tp_size`` ranks, so
    spawning only ``dp_size`` of them hangs forever waiting for ranks nobody
    started. Each cell occupies device ``global_rank``.

    The two roles differ by design:

    * the **leader** (``tp_rank == 0``) owns the replica's scheduler and runs
      :class:`_ReplicaLoop`. It is the only rank that reads the request queue, and
      the only one that answers.
    * a **follower** never sees a request at all. It receives each step's plan from
      its leader over the control plane and runs it, so the ranks agree by
      construction instead of by both deriving the same batch from a broadcast
      prompt.

    Runs in a spawned process, so it takes only picklable arguments and imports
    torch itself. Every startup outcome is reported through ``result_queue`` —
    including a failed build, which arrives as an ``"error"`` message so the
    coordinator raises instead of blocking on a rank that will never answer.

    Args:
        global_rank: This process's rank in ``[0, dp_size * tp_size)``, and its device.
        dp_rank: Which replica this process belongs to.
        tp_rank: Which rank inside that replica; ``0`` is the leader.
        dp_size: Total replicas (for the parallel-state grid).
        tp_size: TP ranks per replica (1 for pure DP).
        engine_kwargs: Checkpoint and model options, as :class:`LLM` takes them,
            minus ``device``.
        max_num_seqs: Requests this replica may keep in flight.
        max_num_batched_tokens: Padded token budget for one prefill group.
        enable_prefix_cache: Leader-only, and separate from ``engine_kwargs`` for
            that reason: the prefix cache is a scheduler concern, and a follower
            has no scheduler to give it to.
        request_queue: Leader-only inbound queue; followers are handed a dummy.
        result_queue: Where ``("ready" | "done" | "error" | "delta" | ...
            | "failed", ...)`` messages go back.
    """
    import torch

    from ..distributed.parallel_state import init_parallel
    from ..executor.executor import serve_plans
    from .continuous_engine import ContinuousBatchingEngine
    from .llm import LLM

    engine: ContinuousBatchingEngine | None = None
    try:
        engine_kwargs = dict(engine_kwargs)
        on_cpu = engine_kwargs.pop("device", "cuda") == "cpu"
        if not on_cpu:
            torch.cuda.set_device(global_rank)
        # The EP flag rides in engine_kwargs (it configures the replica's
        # engine too) but must reach the group state *before* any model is
        # built: expert placement is read at construction, not at forward.
        ep_enabled = engine_kwargs.get("enable_expert_parallel", False)
        init_parallel(
            global_rank=global_rank,
            tp_size=tp_size,
            dp_size=dp_size,
            enable_expert_parallel=ep_enabled,
            backend="gloo" if on_cpu else "nccl",
        )
        device = "cpu" if on_cpu else f"cuda:{global_rank}"
        if tp_rank == 0:
            # ``from_pretrained`` finds this process already in a TP group and
            # therefore spawns nothing: the grid is the coordinator's to own.
            engine = ContinuousBatchingEngine.from_pretrained(
                device=device,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=enable_chunked_prefill,
                max_chunk_size=max_chunk_size,
                enable_prefix_cache=enable_prefix_cache,
                prefix_cache_blocks=prefix_cache_blocks,
                enable_preemption=enable_preemption,
                enable_expert_parallel=ep_enabled,
                **{k: v for k, v in engine_kwargs.items() if k != "enable_expert_parallel"},
            )
        else:
            follower = LLM(
                device=device, **{k: v for k, v in engine_kwargs.items() if k != "pipeline"}
            )
        result_queue.put(("ready", global_rank, None))
    except Exception:
        result_queue.put(("error", global_rank, traceback.format_exc()))
        return

    if engine is None:
        # Exits when the leader broadcasts its stop signal, which is what makes
        # shutting a replica down a single message to its leader.
        serve_plans(follower, max_num_seqs, pipeline=engine_kwargs.get("pipeline"))
        return

    try:
        _ReplicaLoop(engine, request_queue, result_queue).run()
    except Exception:
        _log.exception("replica %d engine loop failed", dp_rank)
    finally:
        engine.shutdown()


class DataParallelEngine:
    """Route generation across ``data_parallel_size`` model replicas.

    Mirrors :meth:`~rapid_llm.engine.llm.LLM.generate` so a caller can swap one for
    the other; what differs is that nothing is loaded in *this* process — the model
    lives in the workers, and this object only routes. Deliberately not an ``LLM``
    subclass for that reason: it owns no model, no KV cache and no sampler, only the
    worker processes and a :class:`~rapid_llm.engine.dp_load_balancer.LoadBalancer`.

    Replica ``i`` spans devices ``[i * tp, (i+1) * tp)`` and is served by that many
    processes — one per cell of the rank grid in
    :mod:`rapid_llm.distributed.parallel_state`. The replica's leader is the process on
    its first device; it holds the scheduler, reads the queue and answers.

    Args:
        model: HuggingFace checkpoint directory, as for :class:`LLM`.
        data_parallel_size: Number of replicas; must not exceed the visible GPUs.
        tensor_parallel_size: TP ranks *within* each replica (1 = pure DP).
        load_balancer: Routing policy name, one of
            :data:`~rapid_llm.engine.dp_load_balancer.LOAD_BALANCERS`.
        max_num_seqs: Requests each replica keeps in flight. Per replica, not in
            total: DP multiplies concurrency along with throughput.
        max_num_batched_tokens: Padded token budget for one prefill group.
        enable_chunked_prefill: Let a prompt prefill across steps, as in vLLM.
            On by default; off makes every replica prefill each prompt in one
            pass and so needs ``max_num_batched_tokens >= max_seq_len``.
        max_chunk_size: Maximum prompt tokens one request prefills per step;
            ``0`` disables chunking.
        enable_prefix_cache: Give every replica a prefix cache. Each one is *local*:
            replicas share no KV, so a prompt whose prefix was prefilled on replica 0
            hits nothing on replica 1. That is what ``load_balancer="cache_aware"``
            exists to fix, and the two are meant to be turned on together.
        prefix_cache_blocks: Optional per-replica prefix-cache capacity in
            16-token blocks; ``None`` uses the profiled cache capacity.
        enable_preemption: Allow a replica's scheduler to evict and recompute
            decode work when its logical concurrency exceeds cache slots.
        **engine_kwargs: Forwarded verbatim to each replica's :class:`LLM`
            (``max_seq_len``, ``quantization``, ``use_cuda_graph``, ...). ``device``
            is not accepted: it is derived from the replica's position in the grid.

    Raises:
        ValueError: If a parallel size is below 1, the grid exceeds visible GPUs,
            ``device`` was passed, or the policy name is unknown.
        RuntimeError: If a replica fails to build.
    """

    def __init__(
        self,
        model: str,
        data_parallel_size: int = 1,
        tensor_parallel_size: int = 1,
        load_balancer: str = "round_robin",
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens: int = DEFAULT_MAX_NUM_BATCHED_TOKENS,
        enable_chunked_prefill: bool = True,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        enable_prefix_cache: bool = False,
        prefix_cache_blocks: int | None = None,
        enable_preemption: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        import torch

        if data_parallel_size < 1:
            raise ValueError(f"data_parallel_size must be >= 1, got {data_parallel_size}")
        if tensor_parallel_size < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}")
        if "device" in engine_kwargs and engine_kwargs["device"] != "cpu":
            raise ValueError(
                "CUDA device is derived from the replica's rank; use device='cpu' for CPU workers"
            )
        if load_balancer not in LOAD_BALANCERS:
            raise ValueError(
                f"unknown load_balancer {load_balancer!r}; choose from {LOAD_BALANCERS}"
            )
        needed = data_parallel_size * tensor_parallel_size
        visible = torch.cuda.device_count()
        if engine_kwargs.get("device") != "cpu" and needed > visible:
            raise ValueError(
                f"data_parallel_size={data_parallel_size} x "
                f"tensor_parallel_size={tensor_parallel_size} needs {needed} GPUs, "
                f"but only {visible} are visible"
            )

        self.model = model
        self.data_parallel_size = data_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.world_size = needed
        self._balancer = make_load_balancer(load_balancer, data_parallel_size)
        self._engine_kwargs = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            **engine_kwargs,
        }
        if self._engine_kwargs.get("use_cuda_graph") is None:
            # ``LLM`` reads None as "decide from the architecture"; a replica's
            # engine takes a bool, and the text-only default is to capture. Each
            # replica captures its own graphs, and DP replicas share no
            # collectives, so a replica's grid is nobody else's business -- the
            # cross-rank agreement a TP group needs is confined to its own group.
            self._engine_kwargs.pop("use_cuda_graph", None)
        self._tokenizer = None
        self._next_batch_id = 0
        self._closed = False

        # "spawn" is required: a forked child inherits a CUDA context that cannot be
        # re-initialised on another device.
        ctx = mp.get_context("spawn")
        # One queue per *replica*, not per cell: a request goes to the leader, which
        # broadcasts the resulting plans to its followers over the control plane.
        self._request_queues = [ctx.Queue() for _ in range(data_parallel_size)]
        self._result_queue: mp.Queue = ctx.Queue()
        self._workers = [
            ctx.Process(
                target=_dp_worker,
                args=(
                    global_rank,
                    global_rank // tensor_parallel_size,
                    global_rank % tensor_parallel_size,
                    data_parallel_size,
                    tensor_parallel_size,
                    self._engine_kwargs,
                    max_num_seqs,
                    max_num_batched_tokens,
                    enable_chunked_prefill,
                    max_chunk_size,
                    enable_prefix_cache,
                    prefix_cache_blocks,
                    enable_preemption,
                    self._request_queues[global_rank // tensor_parallel_size],
                    self._result_queue,
                ),
                daemon=True,
            )
            for global_rank in range(needed)
        ]
        for worker in self._workers:
            worker.start()
        self._await_ready()

    def _await_message(self, timeout_s: float | None = None) -> tuple:
        """Take one message off the result queue, or raise if a replica died.

        A plain blocking ``get`` would wait forever on a worker killed by the OOM
        killer or a segfaulting kernel — there is no message coming, and the caller
        cannot tell that from a slow generation. Polling and checking ``is_alive``
        turns that silence into an error.

        Args:
            timeout_s: Overall deadline, or ``None`` to wait as long as the workers
                stay alive.

        Raises:
            RuntimeError: If a replica exited without reporting, or the deadline passed.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            try:
                return self._result_queue.get(timeout=_LIVENESS_POLL_S)
            except queue.Empty:
                dead = [w for w in self._workers if not w.is_alive()]
                if dead:
                    raise RuntimeError(
                        "data-parallel replica(s) "
                        f"{[w.pid for w in dead]} exited without reporting a result "
                        f"(exit codes {[w.exitcode for w in dead]}); "
                        "the usual cause is the host or GPU running out of memory"
                    ) from None
                if deadline is not None and time.monotonic() > deadline:
                    raise RuntimeError(
                        f"data-parallel replicas did not respond within {timeout_s:.0f}s"
                    ) from None

    def _await_ready(self) -> None:
        """Block until every rank of the grid reports a built engine, or raise."""
        pending = set(range(self.world_size))
        while pending:
            kind, global_rank, detail = self._await_message(STARTUP_TIMEOUT_S)
            if kind == "error":
                self.shutdown()
                raise RuntimeError(f"data-parallel rank {global_rank} failed to start:\n{detail}")
            pending.discard(global_rank)
        _log.info(
            "data parallel ready: %d replicas x TP %d = %d ranks (%s)",
            self.data_parallel_size,
            self.tensor_parallel_size,
            self.world_size,
            type(self._balancer).__name__,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    @property
    def tokenizer(self):
        """Tokenizer of the served checkpoint, loaded on first use.

        The replicas own the ones that matter; this is a convenience for callers that
        count tokens (benchmarks, evals) and would otherwise reach into a worker.
        """
        if self._tokenizer is None:
            from .llm_engine import LLMEngine

            self._tokenizer = LLMEngine._load_tokenizer(self.model)
        return self._tokenizer

    def _tokenize_for_routing(self, prompts: list[str]) -> list[list[int]] | None:
        """Prompt token ids for the balancer, or ``None`` if it reads neither ids nor lengths.

        Prompt *characters* are not prompt tokens: the ratio swings from ~1 for CJK to
        ~6 for whitespace-heavy code, so routing on ``len(prompt)`` silently mis-sizes
        every non-English batch. The tokenizer is the only honest answer, and it is
        called once for the whole batch.

        One pass serves both kinds of token-aware policy — a length is ``len(ids)`` — so
        there is never a reason to tokenise twice. ``round_robin`` reads neither, and for
        it this returns ``None`` without ever touching the tokenizer, which is also what
        keeps a router that never needs one from loading it at all.

        The ids must agree with what the replica itself would produce, because
        ``cache_aware`` hashes them to guess what that replica already has cached; this
        uses the same tokenizer and the same ``add_special_tokens=True`` as
        :class:`~rapid_llm.engine.continuous_engine.ContinuousBatchingEngine` does on
        admission.
        """
        balancer = self._balancer
        if not (balancer.needs_token_estimate or balancer.needs_token_ids):
            return None
        encoded = self.tokenizer(prompts, add_special_tokens=True)["input_ids"]
        return [list(ids) for ids in encoded]

    def _select(self, token_ids: list[int] | None) -> int:
        """One routing decision — the only place a balancer's arguments are assembled.

        Both entry points (batched :meth:`generate` and the async engine's per-request
        stream) route through here so a new policy cannot be wired correctly in one and
        starved in the other.

        Args:
            token_ids: The prompt's tokens, or ``None`` when the policy asked for
                neither ids nor lengths.

        Raises:
            ValueError: The policy reads ids but none were supplied. Routing on ``None``
                would silently degrade ``cache_aware`` to ``total_tokens`` — every
                request a cache miss, with nothing in the output to say so.
        """
        if token_ids is None and self._balancer.needs_token_ids:
            raise ValueError(
                f"{type(self._balancer).__name__} routes on prompt tokens, but none were "
                "supplied; _tokenize_for_routing must run first"
            )
        return self._balancer.select(
            estimated_tokens=0 if token_ids is None else len(token_ids), token_ids=token_ids
        )

    def _route(self, num_prompts: int, token_ids: list[list[int]] | None = None) -> list[list[int]]:
        """Ask the balancer which replica each prompt goes to.

        Returns one index list per replica: replica ``r`` should generate
        ``[prompts[i] for i in result[r]]``. Grouping the per-request decisions back
        into one sub-batch per replica keeps each replica doing a single efficient
        batched forward, while the *choice* stays a per-request policy — so swapping in
        a load-aware balancer changes the split without touching this code.

        Decisions are made in order and each one is told to the balancer immediately, so
        a load-aware policy sees the prompts it has already placed in this same batch
        rather than routing all of them against an idle pool.

        Args:
            num_prompts: How many prompts to place.
            token_ids: Per-prompt tokens from :meth:`_tokenize_for_routing`.

        Raises:
            ValueError: ``token_ids`` was given but does not have ``num_prompts``
                entries, which would misalign every decision after the first gap.
        """
        if token_ids is not None and len(token_ids) != num_prompts:
            raise ValueError(f"expected {num_prompts} token id lists, got {len(token_ids)}")
        buckets: list[list[int]] = [[] for _ in range(self.data_parallel_size)]
        for index in range(num_prompts):
            buckets[self._select(None if token_ids is None else token_ids[index])].append(index)
        return buckets

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        """Generate one completion per prompt, in the caller's order.

        The call blocks until every replica has finished its share, so the batch
        latency is the slowest replica's — with one prompt and four replicas, three of
        them sit idle. DP buys throughput on many requests, not latency on few.

        Args:
            prompts: A single prompt or a batch.
            sampling_params: Defaults to :class:`SamplingParams` defaults.

        Returns:
            One :class:`RequestOutput` per prompt, ordered as ``prompts`` was.

        Raises:
            RuntimeError: If the engine is closed, a replica raised while generating,
                or a replica died without answering.
        """
        if self._closed:
            raise RuntimeError("this DataParallelEngine has been shut down")

        params = sampling_params or SamplingParams()
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if not prompts:
            return []

        token_ids = self._tokenize_for_routing(prompts)
        buckets = self._route(len(prompts), token_ids)
        dispatched = 0
        for replica, indices in enumerate(buckets):
            if not indices:
                continue
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            message = ("batch", batch_id, indices, [prompts[i] for i in indices], params)
            self._request_queues[replica].put(message)
            dispatched += 1

        texts: list[str | None] = [None] * len(prompts)
        reasons: list[str | None] = [None] * len(prompts)
        failure: RuntimeError | None = None
        for _ in range(dispatched):
            kind, batch_id, payload = self._await_message()
            if kind == "error":
                # Drain the rest of this round before raising: the sibling replicas
                # have already (or are about to) put their "done" messages into the
                # shared queue, and anything left there would be misread as a result
                # by the next generate() call.
                if failure is None:
                    failure = RuntimeError(f"data-parallel replica failed on request:\n{payload}")
                continue
            if failure is not None:
                continue  # a sibling's results; this call is failing anyway
            for index, text, reason in payload:
                texts[index] = text
                reasons[index] = reason

        # Every request on a replica is done, so let a load-aware balancer forget
        # them — subtracting the same estimate that was added. Runs on the error path
        # too: skipping it would leak in-flight load forever.
        for replica, indices in enumerate(buckets):
            for index in indices:
                estimate = 0 if token_ids is None else len(token_ids[index])
                self._balancer.release(replica, estimated_tokens=estimate)

        if failure is not None:
            raise failure

        return [
            RequestOutput(prompt=prompt, outputs=[CompletionOutput(0, text or "", reason)])
            for prompt, text, reason in zip(prompts, texts, reasons, strict=True)
        ]

    def shutdown(self) -> None:
        """Stop every replica and release its GPU memory. Idempotent.

        One message per replica is enough: its leader leaves the engine loop, and
        releasing the engine on the way out broadcasts the stop signal that ends its
        followers. Nothing is killed that has not been asked first — a rank in the
        middle of a forward would otherwise leave its CUDA context for the driver to
        clean up.
        """
        if self._closed:
            return
        self._closed = True
        for request_queue in self._request_queues:
            # The queue is already closed if its worker died with it.
            with contextlib.suppress(ValueError, OSError):
                request_queue.put(_SHUTDOWN)
        for worker in self._workers:
            worker.join(timeout=30)
            if worker.is_alive():
                _log.warning("data-parallel replica %s did not exit; terminating", worker.pid)
                worker.terminate()
                worker.join(timeout=5)

    def __enter__(self) -> DataParallelEngine:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    def __del__(self) -> None:
        # Best effort: interpreter teardown may already have dropped what shutdown
        # needs, and a failure here would only mask the real exit path.
        with contextlib.suppress(Exception):
            self.shutdown()
