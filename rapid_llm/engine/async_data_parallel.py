"""Asyncio front end for the data-parallel coordinator.

:class:`AsyncDataParallelEngine` adds a pump thread to the process-based
:class:`~rapid_llm.engine.data_parallel.DataParallelEngine`: the thread
shuttles requests and results between worker queues and asyncio streams.

Usage:
    engine = AsyncDataParallelEngine(model, data_parallel_size=2)
    stream = await engine.generate(prompt)
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
from collections.abc import AsyncIterator
from typing import Any

from ..utils.logger import get_logger
from .async_engine import StreamedOutput, _RequestStream
from .data_parallel import DataParallelEngine
from .sampler import SamplingParams
from .scheduler import DEFAULT_MAX_CHUNK_SIZE, DEFAULT_MAX_NUM_BATCHED_TOKENS, DEFAULT_MAX_NUM_SEQS

logger = get_logger(__name__)

#: Message tag that ends the pump thread. A tag rather than the ``None``
#: sentinel (which stops the *replicas* on the request side) so the two queue
#: directions cannot be confused by a reader of either.
_PUMP_STOP = "pump-stop"

#: How long shutdown waits for the pump to notice the stop tag before giving up
#: on a clean join. The pump is mid-``get`` at worst, so this is generous.
_PUMP_JOIN_TIMEOUT_S = 30.0


class AsyncDataParallelEngine(DataParallelEngine):
    """Serves many concurrent coroutines from ``data_parallel_size`` replicas.

    Construction starts the replica processes exactly as the synchronous engine
    does (and raises the same errors); ``start()`` then adds the pump thread,
    and every ``generate()`` coroutine routes through the load balancer to one
    replica's queue, streaming that request's chunks back as the replica
    reports them. A consumer that abandons its stream aborts the request on the
    replica, so an HTTP connection that drops frees its KV slot instead of
    decoding to the length cap.

    Args:
        model: HuggingFace checkpoint directory, as for :class:`LLM`.
        data_parallel_size: Number of replicas; must not exceed the visible GPUs.
        tensor_parallel_size: TP ranks *within* each replica (1 = pure DP).
        load_balancer: Routing policy, one of
            :data:`~rapid_llm.engine.dp_load_balancer.LOAD_BALANCERS`. The
            load-aware names finally have someone to serve here: with the batch
            API every prompt arrived at once, so "least in-flight" had nothing
            to measure.
        max_num_seqs: Requests each replica keeps in flight. Per replica: DP
            multiplies concurrency along with throughput.
        max_num_batched_tokens: Padded token budget for one prefill group.
        enable_prefix_cache: Give every replica a prefix cache. Pairs with
            ``load_balancer="cache_aware"``, which is what stops the caches from
            being ``data_parallel_size`` unrelated ones — and where the online path
            has the advantage over the batch API, since requests arrive over time
            and a prefix populated by one can still be hot for the next.
        **engine_kwargs: Forwarded verbatim to each replica's engine, as for
            :class:`~rapid_llm.engine.data_parallel.DataParallelEngine`.
            ``device`` is not accepted; it is derived from the grid position.
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
        super().__init__(
            model=model,
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            load_balancer=load_balancer,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
            max_chunk_size=max_chunk_size,
            enable_prefix_cache=enable_prefix_cache,
            prefix_cache_blocks=prefix_cache_blocks,
            enable_preemption=enable_preemption,
            **engine_kwargs,
        )
        self._streams: dict[str, _RequestStream] = {}
        self._stream_snapshot: dict[str, _RequestStream] = {}
        self._request_ids = itertools.count()
        self._pump: threading.Thread | None = None
        self._shutting_down = False
        self._failure: RuntimeError | None = None
        # Serializes starting, admitting a request, and shutdown. A request is
        # registered and sent to its replica under this lock, so shutdown cannot
        # stop the pump in the gap and leave a stream with no possible result.
        self._lifecycle_lock = threading.Lock()
        # Guards mutations to ``_streams`` and its copy-on-write snapshot. The
        # pump reads only the snapshot, avoiding lock contention for every
        # streamed token while coroutines register and drop entries.
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> AsyncDataParallelEngine:
        """Load a checkpoint and wrap it for async serving.

        The same spelling :meth:`~rapid_llm.engine.async_engine.AsyncLLMEngine.from_pretrained`
        uses, so an entrypoint that builds one engine kind can build the other
        without changing the shape of its call.
        """
        return cls(model=model, **kwargs)

    # ------------------------------------------------------------- lifecycle #
    def start(self) -> None:
        """Start the result pump. Idempotent, and safe to call from any loop."""
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        """Start the result pump while ``_lifecycle_lock`` is held."""
        if self._closed or self._shutting_down:
            raise RuntimeError("this AsyncDataParallelEngine has been shut down")
        if self._failure is not None:
            raise RuntimeError("this AsyncDataParallelEngine has failed") from self._failure
        if self._pump is not None:
            return
        self._pump = threading.Thread(
            target=self._pump_results, name="rapid-llm-dp-pump", daemon=True
        )
        self._pump.start()

    async def shutdown(self) -> None:
        """Stop the pump, then the replicas. Idempotent.

        The pump stops *first*: once it has joined, no message it would read can
        matter, so the replicas may be told to stop without anybody listening.
        Both stages block (a replica drains its in-flight batch before exiting),
        so both run off the event loop.
        """
        with self._lifecycle_lock:
            if self._closed or self._shutting_down:
                return
            self._shutting_down = True
            pump = self._pump

        if pump is not None:
            with contextlib.suppress(ValueError, OSError):
                self._result_queue.put((_PUMP_STOP, None, None))
            await asyncio.get_running_loop().run_in_executor(None, pump.join, _PUMP_JOIN_TIMEOUT_S)
            with self._lifecycle_lock:
                if self._pump is pump:
                    self._pump = None
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._stream_snapshot = {}
        for stream in streams:
            stream.push(None)
        await asyncio.get_running_loop().run_in_executor(None, DataParallelEngine.shutdown, self)

    async def __aenter__(self) -> AsyncDataParallelEngine:
        self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.shutdown()

    def __del__(self) -> None:
        # Not the async shutdown(): interpreter teardown cannot await a
        # coroutine, and the parent's synchronous shutdown is what actually
        # reaps the replica processes. The pump is a daemon thread and dies
        # with the process.
        with contextlib.suppress(Exception):
            DataParallelEngine.shutdown(self)

    # ------------------------------------------------------------ public API #
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[StreamedOutput]:
        """Stream one request's completion from whichever replica the balancer picks.

        The request is submitted on first iteration and aborted if the consumer
        stops early — an abandoned HTTP connection frees its replica's cache slot
        on the next step rather than decoding to its length cap.

        Args:
            prompt: Prompt text, already chat-templated if the model expects that.
            sampling_params: Per-request knobs.
            request_id: Caller-supplied id; generated when omitted.

        Yields:
            :class:`~rapid_llm.engine.async_engine.StreamedOutput` chunks, the
            last one carrying a finish reason.

        Raises:
            RuntimeError: If the engine is shut down, or the replica failed or
                died while serving this request.
        """
        token_ids = self._tokenize_for_routing([prompt])
        prompt_ids = None if token_ids is None else token_ids[0]
        estimate = 0 if prompt_ids is None else len(prompt_ids)
        with self._lifecycle_lock:
            self._start_locked()
            request_id = request_id or f"dp-{next(self._request_ids)}"
            stream = _RequestStream(request_id, asyncio.get_running_loop())
            with self._lock:
                if request_id in self._streams:
                    raise ValueError(f"request id {request_id!r} is already active")
                self._streams[request_id] = stream
                self._stream_snapshot = self._streams.copy()
            replica = self._select(prompt_ids)
            try:
                self._request_queues[replica].put(("add", request_id, prompt, sampling_params))
            except BaseException:
                with self._lock:
                    if self._streams.get(request_id) is stream:
                        self._streams.pop(request_id)
                        self._stream_snapshot = self._streams.copy()
                self._balancer.release(replica, estimated_tokens=estimate)
                raise
        try:
            while True:
                chunk = await stream.get()
                if chunk is None:
                    return
                yield chunk
                if chunk.is_finished:
                    return
        finally:
            with self._lock:
                if self._streams.get(request_id) is stream:
                    self._streams.pop(request_id)
                    self._stream_snapshot = self._streams.copy()
            if not stream.finished and not self._closed and not self._shutting_down:
                # A dead replica's queue is not ours to notice: the put is
                # best-effort the same way the parent's shutdown puts are.
                with contextlib.suppress(ValueError, OSError):
                    self._request_queues[replica].put(("abort", request_id))
            # Runs on every exit, error path included: a load-aware balancer
            # that only ever heard ``select`` would count this request forever.
            self._balancer.release(replica, estimated_tokens=estimate)

    async def generate_text(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> StreamedOutput:
        """Await a whole completion, discarding the intermediate chunks."""
        last: StreamedOutput | None = None
        async for chunk in self.generate(prompt, sampling_params, request_id):
            last = chunk
        if last is None:
            raise RuntimeError(f"request {request_id} produced no output")
        return last

    # -------------------------------------------------------------- the pump #
    def _pump_results(self) -> None:
        """Drain the result queue onto streams. The only reader of that queue.

        Waits as long as the replicas stay alive — generation has no deadline —
        and relies on :meth:`DataParallelEngine._await_message` to turn a dead
        replica into an error rather than a hang, which is why the pump does
        not reimplement that logic. On such a failure every stream still open
        hears the error: from a coroutine's side, a coordinator whose workers
        are gone is indistinguishable from one that never answers.
        """
        while True:
            try:
                message = self._await_message()
            except RuntimeError as exc:
                failure = RuntimeError(f"data-parallel engine failed: {exc}")
                with self._lifecycle_lock:
                    self._failure = failure
                with self._lock:
                    streams = list(self._streams.values())
                    self._streams.clear()
                    self._stream_snapshot = {}
                for stream in streams:
                    stream.finished = True
                    stream.push(failure)
                return
            if message[0] == _PUMP_STOP:
                return
            self._deliver(message)

    def _deliver(self, message: tuple) -> None:
        """Move one replica message onto its stream, if anyone still holds it."""
        kind, request_id = message[0], message[1]
        stream = self._get_stream(request_id)
        if stream is None:
            # The consumer went away; the abort command it queued on its way
            # out will reclaim the replica's slot, so there is nothing to do.
            return
        if kind == "delta":
            _, _, delta, text, prompt_tokens, completion_tokens = message
            stream.push(
                StreamedOutput(
                    request_id=request_id,
                    delta=delta,
                    text=text,
                    finish_reason=None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        elif kind == "finished":
            _, _, reason, text, prompt_tokens, completion_tokens = message
            stream.finished = True
            stream.push(
                StreamedOutput(
                    request_id=request_id,
                    delta="",
                    text=text,
                    finish_reason=reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        elif kind == "failed":
            stream.finished = True
            stream.push(RuntimeError(f"replica failed on request {request_id}:\n{message[2]}"))

    def _get_stream(self, request_id: str) -> _RequestStream | None:
        """Look up a stream without locking the result-pump hot path."""
        return self._stream_snapshot.get(request_id)
