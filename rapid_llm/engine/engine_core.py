"""The engine core process: one ContinuousBatchingEngine behind a ZMQ door.

This is vLLM's EngineCore boundary (``vllm/v1/engine/core.py``) cut to
rapid_llm's size, and it keeps vLLM's split exactly: **token ids in, token ids
out**. The child owns the scheduler, the executor, the KV cache and the
offloading tiers; the parent owns the tokenizer and the streaming detokeniser.
Text never crosses the wire, so the child has no reason to hold detokeniser
state a client could resume from.

Channels (one ZMQ context per side; a socket is never shared between threads):

* **commands**: one connection pair, parent ROUTER -> child DEALER. The child
  sends its READY frame back over this channel -- that is how the parent
  learns its routing identity, the same bootstrap vLLM's handshake socket
  performs. ADD / ABORT / UTILITY / SHUTDOWN ride it afterwards.
* **outputs**: child PUSH -> parent PULL. One frame per engine step carrying
  every request that advanced, plus utility replies and a death notice.

Framing is msgpack, and the READY frame spells the :data:`PROTOCOL_VERSION`,
so a skew between the two sides fails at startup with one clear sentence
instead of at the first undecodable step.

Failure design: a crashed child reports through the output channel
(best-effort) *and* exits non-zero; the parent treats either as fatal to every
in-flight request. A dead parent is noticed by the child through ``getppid()``
on its idle poll, so a killed server never leaves a GPU-holding orphan.

Usage (never directly; :mod:`rapid_llm.engine.engine_core_client` spawns it):
    ctx.Process(target=run_engine_core, args=(...), daemon=False)
"""

from __future__ import annotations

import enum
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgpack
import zmq

from .. import __version__
from ..utils.logger import get_logger
from .continuous_engine import ContinuousBatchingEngine
from .sampler import PositionLogprobs, SamplingParams

if TYPE_CHECKING:
    from .scheduler import Request

logger = get_logger(__name__)

#: Bumped whenever a message's shape changes incompatibly. The parent checks
#: it against its own before serving a single request.
PROTOCOL_VERSION = 1

#: Idle poll period of the child, milliseconds. Also the cadence at which a
#: lost parent is noticed, which bounds how long an orphan can hold the GPU.
_IDLE_POLL_MS = 500

#: First frame of a child->parent message on the command channel. Only READY
#: travels child->parent there; everything else uses the output channel.
READY_TAG = b"\x00"


class EngineCoreRequestType(enum.Enum):
    """The command kinds, as raw bytes so the tag is the frame itself.

    Mirrors ``vllm/v1/engine/__init__.py::EngineCoreRequestType`` minus the
    DP/elastic extensions rapid_llm does not have.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    UTILITY = b"\x02"
    SHUTDOWN = b"\x03"


@dataclass
class EngineCoreRequest:
    """One generation request, as the parent hands it over.

    Attributes:
        request_id: The id both sides key their bookkeeping by.
        prompt_token_ids: Encoded prompt; the child never re-encodes.
        sampling_params: Decoding knobs, serialised field by field.
        arrival_time: Parent-side monotonic clock at admission, carried for
            latency accounting (queue time spans processes).
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float


@dataclass
class UtilityRequest:
    """A control query -- ``step_stats`` today -- answered on the output channel."""

    call_id: int
    method: str


@dataclass
class UtilityOutput:
    """The answer to one :class:`UtilityRequest`. ``failure_message`` when it raised."""

    call_id: int
    result: dict[str, Any] | None = None
    failure_message: str | None = None


@dataclass
class EngineCoreOutput:
    """One request's increment since its previous output.

    ``new_token_ids`` are the tokens appended to the request since the last
    frame (a step may add several under speculative decoding). ``error`` marks
    an admission the engine refused; it rides the request's own channel so
    exactly the stream that asked hears it.
    """

    request_id: str
    new_token_ids: list[int]
    finish_reason: str | None = None
    prompt_len: int = 0
    delta_logprobs: PositionLogprobs | None = None
    prompt_logprobs: list[PositionLogprobs | None] | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass
class EngineCoreOutputs:
    """One output-channel frame: a step's worth of replies, or a death notice.

    ``engine_dead`` is the goodbye note of a crashing child: whatever is in
    flight will never finish, and the parent fails those streams with this
    text. ``timestamp`` is stamped on construction (monotonic clock).
    """

    outputs: list[EngineCoreOutput] = field(default_factory=list)
    utility_output: UtilityOutput | None = None
    engine_dead: str | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


@dataclass
class EngineCoreReady:
    """The handshake frame: the child saying it loaded and is serving.

    Attributes:
        protocol_version: Checked by the parent; a mismatch refuses to run.
        engine_version: ``rapid_llm.__version__``, for the error message the
            check above writes.
        max_model_len: Context window the child actually built with.
        num_gpu_blocks: KV blocks the profile settled on.
        max_num_seqs: Decode concurrency ceiling the child runs under.
    """

    protocol_version: int
    engine_version: str
    max_model_len: int
    num_gpu_blocks: int
    max_num_seqs: int


# --------------------------------------------------------------- serialise #


def _pack(payload: Any) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def _unpack(raw: bytes) -> Any:
    return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def _encode_sampling_params(params: SamplingParams) -> dict[str, Any]:
    """Field-by-field, deliberately: this dict *is* the wire contract, and a
    field added to :class:`SamplingParams` must be added here consciously."""
    return {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "max_gen_len": params.max_gen_len,
        "repetition_penalty": params.repetition_penalty,
        "stop_on_repeat": params.stop_on_repeat,
        "logprobs": params.logprobs,
        "prompt_logprobs": params.prompt_logprobs,
    }


def decode_sampling_params(raw: dict[str, Any]) -> SamplingParams:
    """Rebuild the per-request knobs; ``__post_init__`` re-validates them."""
    return SamplingParams(**raw)


def _encode_record(record: PositionLogprobs | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "token_id": record.token_id,
        "logprob": record.logprob,
        "top_token_ids": list(record.top_token_ids),
        "top_logprobs": list(record.top_logprobs),
    }


def _decode_record(raw: dict[str, Any] | None) -> PositionLogprobs | None:
    if raw is None:
        return None
    return PositionLogprobs(
        token_id=raw["token_id"],
        logprob=raw["logprob"],
        top_token_ids=tuple(raw["top_token_ids"]),
        top_logprobs=tuple(raw["top_logprobs"]),
    )


def encode_request(request: EngineCoreRequest) -> bytes:
    """Serialise an ADD's payload."""
    return _pack(
        {
            "request_id": request.request_id,
            "prompt_token_ids": list(request.prompt_token_ids),
            "sampling_params": _encode_sampling_params(request.sampling_params),
            "arrival_time": request.arrival_time,
        }
    )


def decode_request(raw: bytes) -> EngineCoreRequest:
    """Rebuild an ADD's payload."""
    data = _unpack(raw)
    return EngineCoreRequest(
        request_id=data["request_id"],
        prompt_token_ids=list(data["prompt_token_ids"]),
        sampling_params=decode_sampling_params(data["sampling_params"]),
        arrival_time=data["arrival_time"],
    )


def encode_utility(request: UtilityRequest) -> bytes:
    """Serialise a UTILITY's payload."""
    return _pack({"call_id": request.call_id, "method": request.method})


def decode_utility(raw: bytes) -> UtilityRequest:
    """Rebuild a UTILITY's payload."""
    data = _unpack(raw)
    return UtilityRequest(call_id=data["call_id"], method=data["method"])


def encode_abort(request_id: str) -> bytes:
    """Serialise an ABORT's payload."""
    return _pack({"request_id": request_id})


def decode_abort(raw: bytes) -> str:
    """Rebuild an ABORT's payload: the id to cancel."""
    return _unpack(raw)["request_id"]


def encode_outputs(outputs: EngineCoreOutputs) -> bytes:
    """Serialise one output frame (child side)."""
    return _pack(
        {
            "outputs": [_encode_output(o) for o in outputs.outputs],
            "utility_output": (
                None
                if outputs.utility_output is None
                else {
                    "call_id": outputs.utility_output.call_id,
                    "result": outputs.utility_output.result,
                    "failure_message": outputs.utility_output.failure_message,
                }
            ),
            "engine_dead": outputs.engine_dead,
            "timestamp": outputs.timestamp,
        }
    )


def _encode_output(output: EngineCoreOutput) -> dict[str, Any]:
    return {
        "request_id": output.request_id,
        "new_token_ids": list(output.new_token_ids),
        "finish_reason": output.finish_reason,
        "prompt_len": output.prompt_len,
        "delta_logprobs": _encode_record(output.delta_logprobs),
        "prompt_logprobs": (
            None
            if output.prompt_logprobs is None
            else [_encode_record(record) for record in output.prompt_logprobs]
        ),
        "error": output.error,
    }


def decode_outputs(raw: bytes) -> EngineCoreOutputs:
    """Rebuild an output frame (parent side)."""
    data = _unpack(raw)
    utility = data.get("utility_output")
    return EngineCoreOutputs(
        outputs=[_decode_output(item) for item in data["outputs"]],
        utility_output=(
            None
            if utility is None
            else UtilityOutput(
                call_id=utility["call_id"],
                result=utility["result"],
                failure_message=utility["failure_message"],
            )
        ),
        engine_dead=data.get("engine_dead"),
        timestamp=data.get("timestamp", 0.0),
    )


def _decode_output(data: dict[str, Any]) -> EngineCoreOutput:
    prompt_logprobs = data.get("prompt_logprobs")
    return EngineCoreOutput(
        request_id=data["request_id"],
        new_token_ids=list(data["new_token_ids"]),
        finish_reason=data.get("finish_reason"),
        prompt_len=data.get("prompt_len", 0),
        delta_logprobs=_decode_record(data.get("delta_logprobs")),
        prompt_logprobs=(
            None
            if prompt_logprobs is None
            else [_decode_record(record) for record in prompt_logprobs]
        ),
        error=data.get("error"),
    )


def encode_ready(ready: EngineCoreReady) -> bytes:
    """Serialise the handshake frame."""
    return _pack(
        {
            "protocol_version": ready.protocol_version,
            "engine_version": ready.engine_version,
            "max_model_len": ready.max_model_len,
            "num_gpu_blocks": ready.num_gpu_blocks,
            "max_num_seqs": ready.max_num_seqs,
        }
    )


def decode_ready(raw: bytes) -> EngineCoreReady:
    """Rebuild the handshake frame."""
    data = _unpack(raw)
    return EngineCoreReady(
        protocol_version=data["protocol_version"],
        engine_version=data["engine_version"],
        max_model_len=data["max_model_len"],
        num_gpu_blocks=data["num_gpu_blocks"],
        max_num_seqs=data["max_num_seqs"],
    )


# ------------------------------------------------------------ child side #


class EngineCoreProc:
    """The child half: the engine, two sockets, and a step loop.

    Constructed inside the spawned process (weights load here). The parent is
    assumed to have bound both addresses already -- the sockets connect, which
    blocks nothing and fails loudly if nobody listens.

    Args:
        model_dir: Checkpoint directory.
        command_address: Parent's ROUTER endpoint to connect to.
        output_address: Parent's PULL endpoint to connect to.
        engine_kwargs: Forwarded to
            :meth:`ContinuousBatchingEngine.from_pretrained`.
    """

    def __init__(
        self,
        model_dir: str,
        *,
        command_address: str,
        output_address: str,
        engine_kwargs: dict[str, Any],
    ) -> None:
        self._ctx = zmq.Context()
        self._command = self._ctx.socket(zmq.DEALER)
        self._command.connect(command_address)
        self._output = self._ctx.socket(zmq.PUSH)
        self._output.connect(output_address)

        # Sockets first, engine second: when loading fails, the death notice
        # still reaches the parent, which is waiting on the output channel.
        try:
            self._engine = ContinuousBatchingEngine.from_pretrained(model_dir, **engine_kwargs)
        except BaseException as exc:
            self._signal_dead(f"engine construction failed: {type(exc).__name__}: {exc}")
            raise

        self._poller = zmq.Poller()
        self._poller.register(self._command, zmq.POLLIN)

        #: Per-request count of tokens already reported, so each frame carries
        #: exactly the increment. Entries die with their request.
        self._sent_tokens: dict[str, int] = {}
        self._stopping = False
        self._last_step_ms = 0.0

    # -------------------------------------------------------------- serve #
    def run(self) -> None:
        """Serve until SHUTDOWN, a step failure, or the parent's death."""
        try:
            self._send_ready()
            self._serve_forever()
        except BaseException as exc:  # every failure is the parent's news
            logger.exception("engine core crashed")
            self._signal_dead(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._teardown()

    def _serve_forever(self) -> None:
        parent_pid = os.getppid()
        while not self._stopping:
            busy = self._engine.has_unfinished_requests()
            self._pump_commands(timeout_ms=0 if busy else _IDLE_POLL_MS)
            if self._stopping:
                break
            if self._engine.has_unfinished_requests():
                self._step_and_publish()
            elif os.getppid() != parent_pid:
                logger.info("parent process is gone; engine core is stopping")
                break

    def _teardown(self) -> None:
        try:
            self._engine.shutdown()
        except Exception:  # the engine is already the failure being handled
            logger.exception("engine shutdown failed")
        self._command.close(linger=0)
        self._output.close(linger=0)
        self._ctx.term()

    # ----------------------------------------------------------- commands #
    def _pump_commands(self, timeout_ms: int) -> None:
        """Drain every command waiting on the channel, blocking up to *timeout*."""
        if not self._poller.poll(timeout_ms):
            return
        while True:
            try:
                frames = self._command.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            self._handle_command(frames)

    def _handle_command(self, frames: list[bytes]) -> None:
        if len(frames) != 2:
            logger.error("ignoring malformed command frame with %d parts", len(frames))
            return
        kind = next((item for item in EngineCoreRequestType if item.value == frames[0]), None)
        if kind is None:
            logger.error("ignoring unknown command tag %r", frames[0])
            return
        if kind is EngineCoreRequestType.ADD:
            self._apply_add(decode_request(frames[1]))
        elif kind is EngineCoreRequestType.ABORT:
            self._apply_abort(decode_abort(frames[1]))
        elif kind is EngineCoreRequestType.UTILITY:
            self._apply_utility(decode_utility(frames[1]))
        else:
            self._apply_shutdown()

    def _apply_add(self, request: EngineCoreRequest) -> None:
        try:
            self._engine.add_request(
                "",
                request.sampling_params,
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
            )
        except ValueError as exc:
            # A prompt the engine refuses (empty, over the context window,
            # duplicate id): only the stream that asked should hear about it.
            self._send_outputs(
                EngineCoreOutputs(
                    outputs=[
                        EngineCoreOutput(
                            request_id=request.request_id,
                            new_token_ids=[],
                            finish_reason="invalid",
                            error=str(exc),
                        )
                    ]
                )
            )
            return
        self._sent_tokens[request.request_id] = 0

    def _apply_abort(self, request_id: str) -> None:
        self._engine.abort(request_id)
        self._sent_tokens.pop(request_id, None)

    def _apply_utility(self, request: UtilityRequest) -> None:
        if request.method != "step_stats":
            self._send_outputs(
                EngineCoreOutputs(
                    utility_output=UtilityOutput(
                        call_id=request.call_id,
                        failure_message=f"unknown utility method {request.method!r}",
                    )
                )
            )
            return
        scheduler = self._engine.scheduler
        self._send_outputs(
            EngineCoreOutputs(
                utility_output=UtilityOutput(
                    call_id=request.call_id,
                    result={
                        "running": len(scheduler.running),
                        "waiting": len(scheduler.waiting),
                        "num_promoting": scheduler.num_promoting,
                        "last_step_ms": self._last_step_ms,
                    },
                )
            )
        )

    def _apply_shutdown(self) -> None:
        """Abort what is in flight and end the loop.

        Aborting rather than draining: the parent asked to stop and is not
        reading tokens any more. Their KV and slots free during teardown.
        """
        for request in [*self._engine.scheduler.running, *self._engine.scheduler.waiting]:
            try:
                self._engine.abort(request.request_id)
            except Exception:  # already on the way out
                logger.exception("abort of %s failed during shutdown", request.request_id)
        self._stopping = True

    # -------------------------------------------------------------- step #
    def _step_and_publish(self) -> None:
        started = time.monotonic()
        advanced = self._engine.step()
        self._last_step_ms = (time.monotonic() - started) * 1e3

        outputs = []
        for request in advanced:
            output = self._make_output(request)
            if output.new_token_ids or output.finished:
                outputs.append(output)
        if outputs:
            self._send_outputs(EngineCoreOutputs(outputs=outputs))
        elif self._engine.has_unfinished_requests():
            # Nothing advanced this step -- a scheduling gate or an offload
            # landing in progress. Yield the CPU so the wait does not spin.
            time.sleep(0.001)

    def _make_output(self, request: Request) -> EngineCoreOutput:
        """The request's increment since the last frame."""
        request_id = request.request_id
        sent = self._sent_tokens.get(request_id, 0)
        new_ids = list(request.output_token_ids[sent:])
        if new_ids:
            self._sent_tokens[request_id] = sent + len(new_ids)

        # ``delta_logprobs`` is per-step scratch (it survives a step that added
        # no token), so it is only trusted alongside fresh tokens.
        record = request.delta_logprobs if new_ids else None
        prompt_logprobs = None
        if request.is_finished:
            self._sent_tokens.pop(request_id, None)
            if request.prompt_logprobs is not None:
                prompt_logprobs = list(request.prompt_logprobs)
        return EngineCoreOutput(
            request_id=request_id,
            new_token_ids=new_ids,
            finish_reason=request.finish_reason,
            prompt_len=request.prompt_len,
            delta_logprobs=record,
            prompt_logprobs=prompt_logprobs,
        )

    # ------------------------------------------------------------ frames #
    def _send_ready(self) -> None:
        ready = EngineCoreReady(
            protocol_version=PROTOCOL_VERSION,
            engine_version=__version__,
            max_model_len=self._engine.config.max_seq_len,
            num_gpu_blocks=self._engine.num_kv_blocks,
            max_num_seqs=self._engine.scheduler.max_num_seqs,
        )
        self._command.send_multipart([READY_TAG, encode_ready(ready)])

    def _send_outputs(self, outputs: EngineCoreOutputs) -> None:
        self._output.send(encode_outputs(outputs))

    def _signal_dead(self, message: str) -> None:
        """Best-effort goodbye before the process dies."""
        try:
            self._send_outputs(EngineCoreOutputs(engine_dead=message))
        except Exception:  # the channel is the thing that may already be broken
            logger.exception("failed to report engine death")


def run_engine_core(
    model_dir: str,
    *,
    command_address: str,
    output_address: str,
    engine_kwargs: dict[str, Any],
) -> None:
    """Entry point of the spawned process (module-level so ``spawn`` can pickle it)."""
    EngineCoreProc(
        model_dir,
        command_address=command_address,
        output_address=output_address,
        engine_kwargs=engine_kwargs,
    ).run()
