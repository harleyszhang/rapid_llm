"""Asyncio front end for the continuous-batching engine.

:class:`AsyncLLMEngine` runs one pump thread that drives the synchronous
engine's ``step()`` loop and fans results out to per-request asyncio
streams, so many callers share a single batching loop.

Usage:
    engine = AsyncLLMEngine(sync_engine)
    async for chunk in await engine.generate(prompt): ...
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import queue
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from ..utils.logger import get_logger
from .sampler import PositionLogprobs, SamplingParams
from .scheduler import Scheduler

logger = get_logger(__name__)


@dataclass(frozen=True)
class StreamedOutput:
    """One increment of a request's completion.

    Attributes:
        request_id: Which request this belongs to.
        delta: Text produced since the previous chunk; may be empty when a token
            did not complete a character.
        text: The completion so far, including ``delta``.
        finish_reason: ``None`` while generating, else why it stopped.
        prompt_tokens: Prompt size as the engine tokenised it.
        completion_tokens: Tokens sampled so far, this chunk included.
        logprobs: The record for the token this chunk carries; ``None`` unless
            the request asked for them.
        prompt_logprobs: The request's whole prompt records, attached to the
            final chunk only; ``None`` unless the request asked for them.
    """

    request_id: str
    delta: str
    text: str
    finish_reason: str | None
    # Defaults keep hand-built chunks (tests, fakes) valid; the worker always
    # fills these from the engine's own bookkeeping so a caller reporting
    # usage never has to re-encode the text — which would be slow and lossy at
    # token boundaries.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: PositionLogprobs | None = None
    prompt_logprobs: tuple[PositionLogprobs | None, ...] | None = None

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


class _RequestStream:
    """Delivery queue for one request, written by the worker, read by a coroutine.

    Holds the event loop of the coroutine that created it, not a loop the engine
    picked once at startup. Requests from different loops therefore coexist —
    which matters because an ASGI test client, a notebook and ``uvicorn`` each run
    their own, and a queue awaited on one loop is never woken by a put scheduled
    onto another.
    """

    def __init__(self, request_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.request_id = request_id
        self._loop = loop
        self._queue: asyncio.Queue[StreamedOutput | BaseException | None] = asyncio.Queue()
        self.finished = False

    def push(self, item: StreamedOutput | BaseException | None) -> None:
        """Hand an item to the consuming coroutine. Called from the worker thread.

        ``call_soon_threadsafe`` is the whole point: ``asyncio.Queue`` is not
        thread-safe, so the put has to be scheduled onto the loop rather than
        performed on the worker. A loop that is already closing rejects the
        callback, which only happens during shutdown and costs a dropped chunk of
        a request nobody is reading any more.
        """
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self) -> StreamedOutput | None:
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item


class AsyncLLMEngine:
    """Serves many concurrent coroutines from one continuously batched engine.

    Args:
        engine: The engine to drive. This object owns it from here on; calling
            ``step()`` elsewhere would race the worker thread.
    """

    def __init__(self, engine: Scheduler) -> None:
        self._engine = engine
        self._commands: queue.SimpleQueue[tuple[str, Any] | None] = queue.SimpleQueue()
        self._streams: dict[str, _RequestStream] = {}
        self._stream_snapshot: dict[str, _RequestStream] = {}
        self._request_ids = itertools.count()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()
        self._closed = False
        # Serializes starting, admitting a request, and shutting down. Keeping
        # admission in this critical section prevents shutdown from consuming
        # its sentinel between stream registration and the matching ``add``.
        self._lifecycle_lock = threading.Lock()
        # Guards mutations to ``_streams`` and its copy-on-write snapshot. The
        # worker only reads the snapshot, keeping a mutex out of the per-token
        # publish path while coroutines register and drop entries.
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, model: str, **kwargs: Any) -> AsyncLLMEngine:
        """Load a checkpoint and wrap it for async serving.

        Args:
            model: HuggingFace checkpoint directory.
            **kwargs: Forwarded to
                :meth:`Scheduler.from_pretrained`
        """
        return cls(Scheduler.from_pretrained(model, **kwargs))

    # ------------------------------------------------------------- lifecycle #
    @property
    def tokenizer(self):
        """The engine's tokenizer, for chat templating in an entrypoint layer."""
        return self._engine.tokenizer

    @property
    def metrics(self):
        """The engine's metric registry, for the entrypoint's ``/metrics``."""
        return self._engine.metrics

    def start(self) -> None:
        """Start the worker thread. Idempotent, and safe to call from any loop."""
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        """Start the worker while ``_lifecycle_lock`` is held."""
        if self._closed:
            raise RuntimeError("this AsyncLLMEngine has been shut down")
        if self._worker is not None:
            return
        self._worker = threading.Thread(target=self._run, name="rapid-llm-engine", daemon=True)
        self._worker.start()

    async def shutdown(self) -> None:
        """Stop the worker and fail any stream still being read."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
            if worker is not None:
                self._stopping.set()
                self._commands.put(None)  # wake the worker if it is idle

        if worker is not None:
            await asyncio.get_running_loop().run_in_executor(None, worker.join, 30.0)
            with self._lifecycle_lock:
                if self._worker is worker:
                    self._worker = None
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._stream_snapshot = {}
        for stream in streams:
            stream.push(None)

    async def __aenter__(self) -> AsyncLLMEngine:
        self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.shutdown()

    # ------------------------------------------------------------ public API #
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[StreamedOutput]:
        """Stream one request's completion.

        The request is submitted on first iteration and cancelled if the consumer
        stops early — an abandoned HTTP connection therefore frees its cache slot
        on the next step instead of running to its length cap.

        Args:
            prompt: Prompt text, already chat-templated if the model expects that.
            sampling_params: Per-request knobs.
            request_id: Caller-supplied id; generated when omitted.

        Yields:
            :class:`StreamedOutput` chunks, the last one carrying a finish reason.
        """
        with self._lifecycle_lock:
            self._start_locked()
            request_id = request_id or f"async-{next(self._request_ids)}"
            stream = _RequestStream(request_id, asyncio.get_running_loop())
            with self._lock:
                if request_id in self._streams:
                    raise ValueError(f"request id {request_id!r} is already active")
                self._streams[request_id] = stream
                self._stream_snapshot = self._streams.copy()
            self._commands.put(("add", (request_id, prompt, sampling_params)))
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
                # An older coroutine must never remove a newer stream sharing
                # its id (the admission guard normally prevents this; identity
                # makes cleanup safe even if a future caller bypasses it).
                if self._streams.get(request_id) is stream:
                    self._streams.pop(request_id)
                    self._stream_snapshot = self._streams.copy()
            if not stream.finished:
                self._commands.put(("abort", request_id))

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

    # ----------------------------------------------------------- worker loop #
    def _run(self) -> None:
        """Drive the engine until shutdown. The only thread that touches it.

        Releasing the engine belongs here rather than in :meth:`shutdown`: the
        worker owns it, so the thread that ran every model pass is also the one
        that tells the tensor-parallel followers to stop. Doing it from the event
        loop would issue a collective off the owning thread, and skipping it would
        leave those follower processes waiting for a plan that never comes.
        """
        try:
            while not self._stopping.is_set():
                try:
                    self._drain_commands()
                    if not self._engine.has_unfinished_requests():
                        # Idle: block on the command queue rather than spinning, so a
                        # server with no traffic costs no CPU.
                        self._apply(self._commands.get())
                        continue
                    for request in self._engine.step():
                        self._publish(request)
                except Exception as exc:  # the worker thread must not die silently
                    logger.exception("engine worker step failed")
                    self._fail_all(exc)
                    self._drop_everything()
        finally:
            self._engine.shutdown()

    def _drain_commands(self) -> None:
        while True:
            try:
                self._apply(self._commands.get_nowait())
            except queue.Empty:
                return

    def _apply(self, command: tuple[str, Any] | None) -> None:
        if command is None:  # shutdown sentinel
            return
        kind, payload = command
        if kind == "add":
            request_id, prompt, params = payload
            try:
                self._engine.add_request(
                    prompt, params, request_id=request_id, on_error=self._fail_async
                )
            except ValueError as exc:
                # A rejected prompt (empty, or longer than the context window) is
                # the caller's problem, not a server fault: hand the error to that
                # one stream and keep serving everyone else.
                self._fail(request_id, exc)
        elif kind == "abort":
            self._engine.abort(payload)

    def _fail_async(self, request, exc: BaseException) -> None:
        """Deliver a background-tokenize failure to its stream (O10).

        The engine thread fires this from ``step`` once a request's encode
        failed or its prompt was rejected — the same exception the
        synchronous path above would have raised from ``add_request``.
        """
        self._fail(request.request_id, exc)

    def _publish(self, request) -> None:
        stream = self._get_stream(request.request_id)
        if stream is None:
            # Consumer already went away; the abort command it queued will land
            # on a later iteration, so there is nothing to do here.
            return
        if request.is_finished:
            stream.finished = True
        if request.delta or request.is_finished:
            stream.push(
                StreamedOutput(
                    request_id=request.request_id,
                    delta=request.delta,
                    text=request.text,
                    finish_reason=request.finish_reason,
                    prompt_tokens=request.prompt_len,
                    completion_tokens=len(request.output_token_ids),
                    logprobs=request.delta_logprobs,
                    prompt_logprobs=(
                        tuple(request.prompt_logprobs)
                        if request.is_finished and request.prompt_logprobs is not None
                        else None
                    ),
                )
            )

    def _fail(self, request_id: str, exc: BaseException) -> None:
        stream = self._get_stream(request_id)
        if stream is not None:
            stream.finished = True
            stream.push(exc)

    def _get_stream(self, request_id: str) -> _RequestStream | None:
        """Look up a stream without putting a lock on the publish hot path."""
        return self._stream_snapshot.get(request_id)

    def _fail_all(self, exc: BaseException) -> None:
        streams = list(self._stream_snapshot.values())
        for stream in streams:
            stream.finished = True
            stream.push(exc)

    def _drop_everything(self) -> None:
        """Abort every in-flight request after a step raised.

        Without this the loop would re-enter the same broken step forever, since
        the requests that triggered it are still scheduled. Clearing them returns
        the worker to idle so later requests get a clean engine.
        """
        for request in [*self._engine.planner.running, *self._engine.planner.waiting]:
            try:
                self._engine.abort(request.request_id)
            except Exception:  # already on the failure path
                logger.exception("failed to abort %s during recovery", request.request_id)
