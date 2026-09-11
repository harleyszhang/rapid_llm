"""The engine core client: text in, text out, over the two ZMQ channels.

The parent half of the boundary :mod:`rapid_llm.engine.engine_core` defines.
:class:`EngineCoreClient` mirrors :class:`~rapid_llm.engine.async_engine.AsyncLLMEngine`'s
public surface -- ``tokenizer``, ``metrics``, ``start``, ``generate``,
``generate_text``, ``abort``, ``shutdown`` -- so the API server can be pointed
at either backend (``ServerConfig.engine_backend``) without changing a route.

The split it preserves (vLLM v1's): the tokenizer and the streaming detokeniser
live here, in the parent; the scheduler, executor, KV cache and offloading
tiers live in the spawned :class:`~rapid_llm.engine.engine_core.EngineCoreProc`.
Wire traffic is token ids only, and text is reassembled on this side with the
same :class:`~rapid_llm.engine.detokenizer.IncrementalDetokenizer` the
in-process engine runs -- same token stream, same implementation, so both
backends produce byte-identical deltas.

Threading: coroutines hand commands to the child under ``_send_lock`` (a ZMQ
socket is not thread-safe), and one IO thread polls the output channel, watches
the child for death, and asks for periodic ``step_stats``. Request streams are
read by the IO thread through a copy-on-write snapshot, exactly as the
in-process worker does, so the publish path carries no mutex.

Usage:
    engine = EngineCoreClient.from_pretrained("my_weight/Qwen3-0.6B")
    engine.start()
    async for chunk in await engine.generate("hello"): ...
    await engine.shutdown()
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import multiprocessing as mp
import os
import threading
import time
from collections.abc import AsyncIterator
from typing import Any

import zmq

from ..tools.observability.metrics import EngineMetrics
from ..utils.logger import get_logger
from .async_engine import StreamedOutput
from .detokenizer import IncrementalDetokenizer
from .engine_core import (
    PROTOCOL_VERSION,
    READY_TAG,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReady,
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityOutput,
    UtilityRequest,
    decode_outputs,
    decode_ready,
    encode_abort,
    encode_request,
    encode_utility,
    run_engine_core,
)
from .llm_engine import LLMEngine
from .sampler import SamplingParams

logger = get_logger(__name__)

#: How long the parent waits for the child's READY frame, seconds. Weight
#: loading and any CUDA-graph warmup happen inside this window; 900 matches
#: the data-parallel coordinator's own startup budget.
_STARTUP_TIMEOUT_S = 900.0

#: Output-channel poll period, milliseconds. Bounds how long a dead child goes
#: unnoticed, and how late a finished step's tokens arrive.
_IO_POLL_MS = 250

#: Cadence of ``step_stats`` requests, seconds: the freshness of the
#: ``running``/``waiting`` gauges ``/metrics`` renders.
_STATS_INTERVAL_S = 1.0

#: How long the child gets to exit after SHUTDOWN before it is terminated,
#: then killed, seconds. Teardown releases the GPU and reaps TP followers.
_JOIN_GRACE_S = 30.0
_TERMINATE_GRACE_S = 10.0


class EngineDeadError(RuntimeError):
    """The engine core process died; every in-flight request is unanswerable."""


class _ClientStream:
    """Delivery queue and detokeniser state for one in-flight request.

    The child sends token ids; the text a caller reads is reassembled here, so
    the parent repeats exactly what the in-process engine would have built.
    One instance per request, written by the IO thread, read by one coroutine.
    """

    def __init__(self, request_id: str, loop: asyncio.AbstractEventLoop, tokenizer: Any) -> None:
        self.request_id = request_id
        self._loop = loop
        self._queue: asyncio.Queue[StreamedOutput | BaseException | None] = asyncio.Queue()
        self._detok = IncrementalDetokenizer(tokenizer, 1)
        self._text = ""
        self._completion_tokens = 0
        self.finished = False

    def push(self, item: StreamedOutput | BaseException | None) -> None:
        """Hand an item to the consuming coroutine. Called from the IO thread.

        ``call_soon_threadsafe`` is the whole point: ``asyncio.Queue`` is not
        thread-safe, so the put is scheduled onto the loop rather than performed
        on the IO thread. A loop that is already closing rejects the callback,
        which only happens during shutdown and costs a dropped chunk of a
        request nobody is reading any more.
        """
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def get(self) -> StreamedOutput | None:
        item = await self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def push_output(self, output: EngineCoreOutput) -> None:
        """Fold one engine frame into this stream's text and queue the chunk."""
        if output.error is not None:
            # An admission the engine refused: only the stream that asked
            # hears about it.
            self.finished = True
            self.push(ValueError(output.error))
            return
        delta = ""
        for token_id in output.new_token_ids:
            delta += self._detok.append(0, token_id)
        if output.new_token_ids:
            self._completion_tokens += len(output.new_token_ids)
            self._text += delta
        if output.finished:
            self.finished = True
        if not delta and not output.finished:
            # A token that did not complete a character yet: held back, exactly
            # as the in-process publisher holds it back.
            return
        self.push(
            StreamedOutput(
                request_id=self.request_id,
                delta=delta,
                text=self._text,
                finish_reason=output.finish_reason,
                prompt_tokens=output.prompt_len,
                completion_tokens=self._completion_tokens,
                # ``delta_logprobs`` is per-step scratch; the child drops it on
                # steps that added no token, and the guard mirrors that here.
                logprobs=output.delta_logprobs if output.new_token_ids else None,
                prompt_logprobs=(
                    tuple(output.prompt_logprobs) if output.prompt_logprobs is not None else None
                ),
            )
        )


class EngineCoreClient:
    """Serves many concurrent coroutines from one engine core process.

    Build with :meth:`from_pretrained`; it loads the tokenizer here (the
    parent's half of the split) and blocks until the child reports READY, so an
    instance that exists is one that can serve.
    """

    #: Distinguishes this process's IPC endpoints from any other client's; the
    #: pid distinguishes processes, this distinguishes instances.
    _address_seq = itertools.count()

    def __init__(self, model_dir: str, tokenizer: Any, engine_kwargs: dict[str, Any]) -> None:
        self._model_dir = model_dir
        self._tokenizer = tokenizer
        self._engine_kwargs = dict(engine_kwargs)

        # One context per client, so ``shutdown`` can terminate it without
        # touching anybody else's sockets.
        self._ctx = zmq.Context()
        self._command = self._ctx.socket(zmq.ROUTER)
        self._output = self._ctx.socket(zmq.PULL)
        suffix = f"{os.getpid()}-{next(self._address_seq)}"
        self._command_address = f"ipc:///tmp/rapid-llm-core-{suffix}.cmd"
        self._output_address = f"ipc:///tmp/rapid-llm-core-{suffix}.out"
        # Bind before the child is spawned: connecting to a bound endpoint is
        # instant, and the READY frame has somewhere to land the moment the
        # child sends it.
        self._command.bind(self._command_address)
        self._output.bind(self._output_address)

        self._proc: mp.process.BaseProcess | None = None
        self._child_identity: bytes | None = None
        self._ready: EngineCoreReady | None = None
        self._metrics = EngineMetrics()
        self._streams: dict[str, _ClientStream] = {}
        self._stream_snapshot: dict[str, _ClientStream] = {}
        self._request_ids = itertools.count()
        self._utility_ids = itertools.count()
        self._io_thread: threading.Thread | None = None
        self._stopping = threading.Event()
        self._closed = False
        self._dead: EngineDeadError | None = None
        # Serializes starting, admitting a request, and shutting down --
        # admission in this critical section is what keeps shutdown from
        # racing a stream that is still being registered.
        self._lifecycle_lock = threading.Lock()
        # Guards mutations to ``_streams`` and its copy-on-write snapshot. The
        # IO thread only reads the snapshot, keeping a mutex out of the
        # per-token delivery path.
        self._lock = threading.Lock()
        # A ZMQ socket is not thread-safe; coroutines and the IO thread both
        # send commands (ADD/ABORT from the former, step_stats from the latter).
        self._send_lock = threading.Lock()

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        startup_timeout_s: float = _STARTUP_TIMEOUT_S,
        **engine_kwargs: Any,
    ) -> EngineCoreClient:
        """Spawn an engine core for *model* and wait until it reports ready.

        Args:
            model: HuggingFace checkpoint directory.
            startup_timeout_s: How long the child may take to load and answer.
            **engine_kwargs: Forwarded to
                :meth:`ContinuousBatchingEngine.from_pretrained`.

        Raises:
            RuntimeError: If the child fails to load or speaks another protocol.
            TimeoutError: If it never reports within the deadline.
        """
        tokenizer = LLMEngine._load_tokenizer(model)
        instance = None
        try:
            instance = cls(model, tokenizer, engine_kwargs)
            instance._launch(startup_timeout_s)
        except BaseException:
            if instance is not None:
                instance._abort_launch()
            raise
        return instance

    # ------------------------------------------------------------ lifecycle #
    def _launch(self, startup_timeout_s: float) -> None:
        """Spawn the child and block until its READY frame arrives."""
        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(
            target=run_engine_core,
            args=(self._model_dir,),
            kwargs={
                "command_address": self._command_address,
                "output_address": self._output_address,
                "engine_kwargs": self._engine_kwargs,
            },
            # NOT a daemon: the executor's tensor-parallel followers are daemons,
            # and a daemon process may not have children.
            daemon=False,
            name="rapid-llm-engine-core",
        )
        self._proc.start()
        self._await_ready(startup_timeout_s)
        ready = self._ready
        logger.info(
            "engine core ready (pid %s): %d KV blocks, max %d sequences, %d-token window",
            self._proc.pid,
            ready.num_gpu_blocks,
            ready.max_num_seqs,
            ready.max_model_len,
        )

    def _abort_launch(self) -> None:
        """Best-effort cleanup after a failed :meth:`_launch`."""
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=_TERMINATE_GRACE_S)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=_TERMINATE_GRACE_S)
        self._close_transport()

    def _await_ready(self, timeout_s: float) -> None:
        """Wait for READY on the command channel, watching for earlier failure.

        The output channel is polled alongside: when the child dies during
        construction it says why over there before exiting, and that sentence
        is worth far more than the bare exit code a liveness check would give.
        """
        poller = zmq.Poller()
        poller.register(self._command, zmq.POLLIN)
        poller.register(self._output, zmq.POLLIN)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"engine core did not report ready within {timeout_s:.0f}s")
            events = dict(poller.poll(min(_IO_POLL_MS, remaining * 1000)))
            if self._command in events:
                frames = self._command.recv_multipart()
                if len(frames) == 3 and frames[1] == READY_TAG:
                    self._child_identity = frames[0]
                    self._ready = decode_ready(frames[2])
                    self._check_protocol()
                    return
                logger.warning("engine core sent an unexpected startup frame; ignoring it")
            if self._output in events:
                frame = decode_outputs(self._output.recv())
                if frame.engine_dead is not None:
                    raise RuntimeError(f"engine core failed to start: {frame.engine_dead}")
            if not self._proc.is_alive():
                raise RuntimeError(
                    f"engine core exited with code {self._proc.exitcode} during startup"
                )

    def _check_protocol(self) -> None:
        """Refuse to serve against an engine core that speaks another version."""
        ready = self._ready
        if ready.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(
                f"engine core protocol mismatch: the child speaks "
                f"v{ready.protocol_version} (rapid-llm {ready.engine_version}), "
                f"this client speaks v{PROTOCOL_VERSION}; both halves must be "
                "restarted from the same install"
            )

    def start(self) -> None:
        """Start the IO thread. Idempotent, and safe to call from any loop."""
        with self._lifecycle_lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._closed:
            raise RuntimeError("this EngineCoreClient has been shut down")
        if self._io_thread is not None:
            return
        self._io_thread = threading.Thread(
            target=self._io_loop, name="rapid-llm-core-client", daemon=True
        )
        self._io_thread.start()

    async def shutdown(self) -> None:
        """Stop the child -- graceful, then SIGTERM, then SIGKILL -- and fail leftovers."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            proc = self._proc
            if proc is not None and self._child_identity is not None:
                with contextlib.suppress(EngineDeadError):
                    self._send(EngineCoreRequestType.SHUTDOWN, b"")
            io_thread = self._io_thread
            self._stopping.set()

        loop = asyncio.get_running_loop()
        if proc is not None:
            await loop.run_in_executor(None, self._join_child, proc)
        if io_thread is not None:
            await loop.run_in_executor(None, io_thread.join, 5.0)
            with self._lifecycle_lock:
                if self._io_thread is io_thread:
                    self._io_thread = None

        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._stream_snapshot = {}
        for stream in streams:
            stream.finished = True
            stream.push(None)
        self._close_transport()

    def _join_child(self, proc: mp.process.BaseProcess) -> None:
        """Wait for the child to exit, escalating SIGTERM then SIGKILL."""
        proc.join(timeout=_JOIN_GRACE_S)
        if proc.is_alive():
            logger.warning("engine core did not exit within %.0fs; terminating it", _JOIN_GRACE_S)
            proc.terminate()
            proc.join(timeout=_TERMINATE_GRACE_S)
        if proc.is_alive():
            logger.error("engine core ignored SIGTERM; killing it")
            proc.kill()
            proc.join(timeout=_TERMINATE_GRACE_S)
        if proc.is_alive():  # pragma: no cover - a process that cannot be killed
            logger.error("engine core (pid %s) survived SIGKILL", proc.pid)

    async def __aenter__(self) -> EngineCoreClient:
        self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.shutdown()

    # ------------------------------------------------------------ properties #
    @property
    def tokenizer(self) -> Any:
        """The checkpoint's tokenizer, loaded in this process (the parent's half)."""
        return self._tokenizer

    @property
    def metrics(self) -> EngineMetrics:
        """The parent-side metric registry, for the entrypoint's ``/metrics``.

        The child cannot export its instruments across processes, so the
        occupancy gauges are refreshed from periodic ``step_stats`` replies and
        the rest is filled by the request path -- the numbers ``/metrics``
        promised stay available under either backend.
        """
        return self._metrics

    @property
    def core_info(self) -> EngineCoreReady | None:
        """The child's handshake numbers, or ``None`` before :meth:`start`."""
        return self._ready

    # ------------------------------------------------------------ public API #
    async def generate(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> AsyncIterator[StreamedOutput]:
        """Stream one request's completion.

        Same contract as the in-process engine's: submit on first iteration,
        and abort if the consumer stops early. The prompt is tokenised here --
        the child never sees text -- and the request rides the command channel.

        Args:
            prompt: Prompt text, already chat-templated if the model expects that.
            sampling_params: Per-request knobs.
            request_id: Caller-supplied id; generated when omitted.

        Yields:
            :class:`StreamedOutput` chunks, the last one carrying a finish reason.
        """
        with self._lifecycle_lock:
            self._start_locked()
            if self._dead is not None:
                raise self._dead
            request_id = request_id or f"core-{next(self._request_ids)}"
            stream = _ClientStream(request_id, asyncio.get_running_loop(), self._tokenizer)
            with self._lock:
                if request_id in self._streams:
                    raise ValueError(f"request id {request_id!r} is already active")
                self._streams[request_id] = stream
                self._stream_snapshot = self._streams.copy()
        try:
            self._submit(request_id, prompt, sampling_params)
        except BaseException:
            with self._lock:
                if self._streams.get(request_id) is stream:
                    self._streams.pop(request_id)
                    self._stream_snapshot = self._streams.copy()
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
                # An older coroutine must never remove a newer stream sharing
                # its id (the admission guard normally prevents this; identity
                # makes cleanup safe even if a future caller bypasses it).
                if self._streams.get(request_id) is stream:
                    self._streams.pop(request_id)
                    self._stream_snapshot = self._streams.copy()
            if not stream.finished:
                self.abort(request_id)

    def _submit(self, request_id: str, prompt: str, sampling_params: SamplingParams | None) -> None:
        """Tokenise one request and hand it to the child."""
        prompt_token_ids = self._tokenizer.encode(prompt, add_special_tokens=True)
        if not prompt_token_ids:
            raise ValueError("the prompt is empty after tokenisation")
        self._send(
            EngineCoreRequestType.ADD,
            encode_request(
                EngineCoreRequest(
                    request_id=request_id,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sampling_params or SamplingParams(),
                    arrival_time=time.monotonic(),
                )
            ),
        )

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

    def abort(self, request_id: str) -> None:
        """Cancel a request. Best-effort: a closed channel means the engine core
        is already going away, and the request dies with it.
        """
        if self._child_identity is None:
            return
        with contextlib.suppress(EngineDeadError):
            self._send(EngineCoreRequestType.ABORT, encode_abort(request_id))

    # ----------------------------------------------------------- IO thread #
    def _io_loop(self) -> None:
        """The parent's receive side: outputs in, liveness watched, stats asked."""
        poller = zmq.Poller()
        poller.register(self._output, zmq.POLLIN)
        last_stats = time.monotonic()
        while not self._stopping.is_set():
            if poller.poll(_IO_POLL_MS):
                self._drain_outputs()
            if self._proc is not None and not self._proc.is_alive():
                self._on_child_exit()
                return
            now = time.monotonic()
            if now - last_stats >= _STATS_INTERVAL_S:
                last_stats = now
                self._request_step_stats()

    def _drain_outputs(self) -> None:
        """Take every frame already on the output channel."""
        while True:
            try:
                raw = self._output.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            self._handle_frame(decode_outputs(raw))

    def _handle_frame(self, frame: EngineCoreOutputs) -> None:
        if frame.utility_output is not None:
            self._apply_utility(frame.utility_output)
        for output in frame.outputs:
            stream = self._stream_snapshot.get(output.request_id)
            if stream is None:
                # The consumer is gone; the abort command it queued will land
                # on a later drain, so there is nothing to do here.
                continue
            stream.push_output(output)
        if frame.engine_dead is not None:
            error = EngineDeadError(f"the engine core died: {frame.engine_dead}")
            logger.error("%s", error)
            self._fail_all(error)

    def _on_child_exit(self) -> None:
        """The child is gone: finish every stream accordingly.

        A zero exit or a shutdown in progress is the ordinary ending; anything
        else is a crash, and the streams still open must hear about it instead
        of waiting for tokens that will never come.
        """
        code = self._proc.exitcode
        if self._closed or self._stopping.is_set() or code == 0:
            logger.info("engine core exited with code %s", code)
            self._fail_all(None)
        else:
            error = EngineDeadError(f"the engine core exited unexpectedly with code {code}")
            logger.error("%s", error)
            self._dead = error
            self._fail_all(error)

    def _fail_all(self, item: BaseException | None) -> None:
        """Deliver one terminal item to every open stream (idempotent)."""
        for stream in list(self._stream_snapshot.values()):
            stream.finished = True
            stream.push(item)

    def _apply_utility(self, output: UtilityOutput) -> None:
        if output.failure_message is not None:
            logger.warning("engine core utility call failed: %s", output.failure_message)
            return
        result = output.result or {}
        self._metrics.observe_load(int(result.get("running", 0)), int(result.get("waiting", 0)))

    def _request_step_stats(self) -> None:
        """Ask the child for its queue depth, feeding the ``/metrics`` gauges."""
        with contextlib.suppress(EngineDeadError):
            self._send(
                EngineCoreRequestType.UTILITY,
                encode_utility(
                    UtilityRequest(call_id=next(self._utility_ids), method="step_stats")
                ),
            )

    # ------------------------------------------------------------- transport #
    def _send(self, kind: EngineCoreRequestType, payload: bytes) -> None:
        """Send one command to the child. Serialised by ``_send_lock``."""
        identity = self._child_identity
        if identity is None:
            raise RuntimeError("the engine core connection is not established")
        with self._send_lock:
            try:
                self._command.send_multipart([identity, kind.value, payload])
            except zmq.ZMQError as exc:
                raise EngineDeadError(f"the engine core command channel is closed: {exc}") from exc

    def _close_transport(self) -> None:
        """Close both sockets and terminate the context (idempotent)."""
        with self._send_lock:
            self._command.close(linger=0)
            self._output.close(linger=0)
            self._ctx.term()
