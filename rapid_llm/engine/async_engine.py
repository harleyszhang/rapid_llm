"""Async request manager for a scheduler process.

The parent process owns tokenization, incremental detokenization and request
streams. The child process owns ``Scheduler``, its executor and all device
state. Typed commands and events are the only objects crossing that boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import enum
import itertools
import multiprocessing as mp
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from ..tools.observability.metrics import EngineMetrics
from ..utils.logger import get_logger
from .detokenizer import IncrementalDetokenizer
from .llm_engine import LLMEngine
from .sampler import PositionLogprobs, SamplingParams
from .scheduler import run_scheduler_process
from .scheduler_ipc import (
    PROTOCOL_VERSION,
    AbortRequest,
    AddRequest,
    RequestEvent,
    SchedulerEvents,
    SchedulerFailed,
    SchedulerReady,
    ShutdownScheduler,
    UtilityEvent,
    UtilityRequest,
)

logger = get_logger(__name__)

_STARTUP_TIMEOUT_S = 900.0
_OUTPUT_POLL_S = 0.25
_STATS_INTERVAL_S = 1.0
_JOIN_GRACE_S = 30.0
_TERMINATE_GRACE_S = 10.0


@dataclass(frozen=True)
class StreamedOutput:
    """One increment of a request's completion."""

    request_id: str
    delta: str
    text: str
    finish_reason: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: PositionLogprobs | None = None
    prompt_logprobs: tuple[PositionLogprobs | None, ...] | None = None

    @property
    def is_finished(self) -> bool:
        return self.finish_reason is not None


class SchedulerProcessError(RuntimeError):
    """The scheduler process failed and cannot answer active requests."""


class _Lifecycle(enum.Enum):
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


class _RequestStream:
    """Per-request delivery queue and incremental detokenization state."""

    def __init__(self, request_id: str, loop: asyncio.AbstractEventLoop, tokenizer: Any) -> None:
        self.request_id = request_id
        self._loop = loop
        self._queue: asyncio.Queue[StreamedOutput | BaseException | None] = asyncio.Queue()
        self._detokenizer = IncrementalDetokenizer(tokenizer, 1)
        self._text = ""
        self._completion_tokens = 0
        self.finished = False

    def push(self, item: StreamedOutput | BaseException | None) -> None:
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self) -> StreamedOutput | None:
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def push_event(self, event: RequestEvent) -> None:
        if event.error is not None:
            self.finished = True
            self.push(ValueError(event.error))
            return

        delta = "".join(self._detokenizer.append(0, token) for token in event.new_token_ids)
        if event.new_token_ids:
            self._completion_tokens += len(event.new_token_ids)
            self._text += delta
        if event.finished:
            self.finished = True
        if not delta and not event.finished:
            return
        self.push(
            StreamedOutput(
                request_id=self.request_id,
                delta=delta,
                text=self._text,
                finish_reason=event.finish_reason,
                prompt_tokens=event.prompt_len,
                completion_tokens=self._completion_tokens,
                logprobs=event.delta_logprobs if event.new_token_ids else None,
                prompt_logprobs=event.prompt_logprobs,
            )
        )


class RequestTracker:
    """Own the single request table used by generation, abort and failure paths."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._streams: dict[str, _RequestStream] = {}
        self._snapshot: dict[str, _RequestStream] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str, loop: asyncio.AbstractEventLoop) -> _RequestStream:
        stream = _RequestStream(request_id, loop, self._tokenizer)
        with self._lock:
            if request_id in self._streams:
                raise ValueError(f"request id {request_id!r} is already active")
            self._streams[request_id] = stream
            self._snapshot = self._streams.copy()
        return stream

    def get(self, request_id: str) -> _RequestStream | None:
        return self._snapshot.get(request_id)

    def remove(self, request_id: str, stream: _RequestStream) -> None:
        with self._lock:
            if self._streams.get(request_id) is stream:
                self._streams.pop(request_id)
                self._snapshot = self._streams.copy()

    def fail_all(self, item: BaseException | None) -> None:
        for stream in tuple(self._snapshot.values()):
            stream.finished = True
            stream.push(item)

    def clear(self, item: BaseException | None) -> None:
        with self._lock:
            streams = tuple(self._streams.values())
            self._streams.clear()
            self._snapshot = {}
        for stream in streams:
            stream.finished = True
            stream.push(item)


class AsyncLLMEngine:
    """Serve concurrent coroutines through one isolated scheduler process."""

    def __init__(
        self,
        model: str,
        tokenizer: Any,
        engine_kwargs: dict[str, Any],
        *,
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
        process_target: Callable = run_scheduler_process,
        process_context: Any = None,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._engine_kwargs = dict(engine_kwargs)
        self._metrics = EngineMetrics()
        self._tracker = RequestTracker(tokenizer)
        self._request_ids = itertools.count()
        self._utility_ids = itertools.count()
        self._lifecycle_lock = threading.Lock()
        self._state = _Lifecycle.STARTING
        self._failure: SchedulerProcessError | None = None
        self._receiver: threading.Thread | None = None
        self._stop_receiver = threading.Event()

        ctx = process_context or mp.get_context("spawn")
        self._commands = ctx.Queue()
        self._events = ctx.Queue()
        self._process = ctx.Process(
            target=process_target,
            args=(model, self._engine_kwargs, self._commands, self._events),
            daemon=False,
            name="rapid-llm-scheduler",
        )
        try:
            self._process.start()
            self._ready = self._await_ready(startup_timeout_s)
        except BaseException:
            self._abort_launch()
            raise
        self._state = _Lifecycle.RUNNING

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
        **engine_kwargs: Any,
    ) -> AsyncLLMEngine:
        tokenizer = LLMEngine._load_tokenizer(model)
        return cls(model, tokenizer, engine_kwargs, startup_timeout_s=startup_timeout_s)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def metrics(self) -> EngineMetrics:
        return self._metrics

    @property
    def scheduler_info(self) -> SchedulerReady:
        return self._ready

    def start(self) -> None:
        """Start the single output receiver. The operation is idempotent."""
        with self._lifecycle_lock:
            self._require_running()
            self._start_receiver_locked()

    async def shutdown(self) -> None:
        """Stop admission, stop output delivery, then reclaim the child process."""
        with self._lifecycle_lock:
            if self._state in (_Lifecycle.CLOSING, _Lifecycle.CLOSED):
                return
            self._state = _Lifecycle.CLOSING
            with contextlib.suppress(Exception):
                self._commands.put(ShutdownScheduler())
            self._stop_receiver.set()
            receiver = self._receiver

        loop = asyncio.get_running_loop()
        if receiver is not None:
            await loop.run_in_executor(None, receiver.join, 5.0)
        await loop.run_in_executor(None, self._join_process)
        self._tracker.clear(None)
        self._close_channels()
        with self._lifecycle_lock:
            self._receiver = None
            self._state = _Lifecycle.CLOSED

    async def __aenter__(self) -> AsyncLLMEngine:
        self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.shutdown()

    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[StreamedOutput]:
        """Tokenize, submit and stream one request; abort it on early close."""
        with self._lifecycle_lock:
            self._require_running()
            self._start_receiver_locked()
            request_id = request_id or f"request-{next(self._request_ids)}"
            stream = self._tracker.register(request_id, asyncio.get_running_loop())

        submitted = False
        try:
            token_ids = await asyncio.to_thread(
                self._tokenizer.encode, prompt, add_special_tokens=True
            )
            if not token_ids:
                raise ValueError("the prompt is empty after tokenisation")
            with self._lifecycle_lock:
                self._require_running()
                self._commands.put(
                    AddRequest(
                        request_id=request_id,
                        prompt_token_ids=tuple(token_ids),
                        sampling_params=sampling_params or SamplingParams(),
                        arrival_time=time.monotonic(),
                    )
                )
                submitted = True

            while True:
                chunk = await stream.get()
                if chunk is None:
                    return
                yield chunk
                if chunk.is_finished:
                    return
        finally:
            self._tracker.remove(request_id, stream)
            if submitted and not stream.finished:
                self.abort(request_id)

    async def generate_text(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> StreamedOutput:
        last = None
        async for chunk in self.generate(prompt, sampling_params, request_id):
            last = chunk
        if last is None:
            raise RuntimeError(f"request {request_id} produced no output")
        return last

    def abort(self, request_id: str) -> None:
        with self._lifecycle_lock:
            if self._state is _Lifecycle.RUNNING:
                self._commands.put(AbortRequest(request_id))

    def _await_ready(self, timeout_s: float) -> SchedulerReady:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"scheduler process did not report ready within {timeout_s:.0f}s"
                )
            try:
                event = self._events.get(timeout=min(_OUTPUT_POLL_S, remaining))
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError(
                        f"scheduler process exited with code {self._process.exitcode} during startup"
                    ) from None
                continue
            if isinstance(event, SchedulerFailed):
                raise RuntimeError(f"scheduler process failed to start: {event.message}")
            if not isinstance(event, SchedulerReady):
                logger.warning("scheduler process sent an unexpected startup event; ignoring it")
                continue
            if event.protocol_version != PROTOCOL_VERSION:
                raise RuntimeError(
                    f"scheduler protocol mismatch: child v{event.protocol_version} "
                    f"({event.engine_version}), parent v{PROTOCOL_VERSION}"
                )
            return event

    def _start_receiver_locked(self) -> None:
        if self._receiver is not None:
            return
        self._receiver = threading.Thread(
            target=self._output_loop,
            name="rapid-llm-scheduler-output",
            daemon=True,
        )
        self._receiver.start()

    def _output_loop(self) -> None:
        last_stats = time.monotonic()
        while not self._stop_receiver.is_set():
            try:
                event = self._events.get(timeout=_OUTPUT_POLL_S)
            except queue.Empty:
                if not self._process.is_alive():
                    self._on_process_exit()
                    return
            else:
                self._handle_event(event)
            if self._stop_receiver.is_set():
                return
            now = time.monotonic()
            if now - last_stats >= _STATS_INTERVAL_S:
                last_stats = now
                self._commands.put(UtilityRequest(next(self._utility_ids), "step_stats"))

    def _handle_event(self, event) -> None:
        if isinstance(event, SchedulerFailed):
            self._mark_failed(event.message)
            return
        if not isinstance(event, SchedulerEvents):
            logger.warning("ignoring unexpected scheduler event %s", type(event).__name__)
            return
        if event.utility_output is not None:
            self._apply_utility(event.utility_output)
        for output in event.outputs:
            stream = self._tracker.get(output.request_id)
            if stream is not None:
                stream.push_event(output)

    def _apply_utility(self, output: UtilityEvent) -> None:
        if output.failure_message is not None:
            logger.warning("scheduler utility call failed: %s", output.failure_message)
            return
        result = output.result or {}
        self._metrics.observe_load(int(result.get("running", 0)), int(result.get("waiting", 0)))

    def _on_process_exit(self) -> None:
        with self._lifecycle_lock:
            if self._state is not _Lifecycle.RUNNING:
                return
        self._mark_failed(f"exited unexpectedly with code {self._process.exitcode}")

    def _mark_failed(self, message: str) -> None:
        error = SchedulerProcessError(f"scheduler process failed: {message}")
        with self._lifecycle_lock:
            if self._state is not _Lifecycle.RUNNING:
                return
            self._state = _Lifecycle.FAILED
            self._failure = error
            self._stop_receiver.set()
        logger.error("%s", error)
        self._tracker.fail_all(error)

    def _require_running(self) -> None:
        if self._state is _Lifecycle.FAILED:
            raise self._failure
        if self._state is not _Lifecycle.RUNNING:
            raise RuntimeError(f"AsyncLLMEngine is {self._state.value}")

    def _abort_launch(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=_TERMINATE_GRACE_S)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=_TERMINATE_GRACE_S)
        self._close_channels()

    def _join_process(self) -> None:
        self._process.join(timeout=_JOIN_GRACE_S)
        if self._process.is_alive():
            logger.warning("scheduler process did not stop; terminating it")
            self._process.terminate()
            self._process.join(timeout=_TERMINATE_GRACE_S)
        if self._process.is_alive():
            logger.error("scheduler process ignored termination; killing it")
            self._process.kill()
            self._process.join(timeout=_TERMINATE_GRACE_S)

    def _close_channels(self) -> None:
        for channel in (self._commands, self._events):
            with contextlib.suppress(Exception):
                channel.close()
                channel.join_thread()
