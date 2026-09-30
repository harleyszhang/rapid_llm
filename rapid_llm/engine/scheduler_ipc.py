"""Typed commands and events shared by scheduler process boundaries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .sampler import PositionLogprobs, SamplingParams

PROTOCOL_VERSION = 2


@dataclass(frozen=True, slots=True)
class AddRequest:
    """Submit one already-tokenized generation request."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    sampling_params: SamplingParams
    arrival_time: float
    stream: bool = True


@dataclass(frozen=True, slots=True)
class AddRequestBatch:
    """Submit requests atomically so one scheduling step sees the whole batch."""

    requests: tuple[AddRequest, ...]


@dataclass(frozen=True, slots=True)
class AbortRequest:
    """Cancel one active or waiting request."""

    request_id: str


@dataclass(frozen=True, slots=True)
class UtilityRequest:
    """Query scheduler process state without entering the model path."""

    call_id: int
    method: str


@dataclass(frozen=True, slots=True)
class WakeScheduler:
    """Wake an idle scheduler so it can join a data-parallel coordination round."""


@dataclass(frozen=True, slots=True)
class ShutdownScheduler:
    """Stop accepting work, drain active requests, and end the event loop."""


type SchedulerCommand = (
    AddRequest | AddRequestBatch | AbortRequest | UtilityRequest | WakeScheduler | ShutdownScheduler
)


@dataclass(frozen=True, slots=True)
class RequestEvent:
    """One request's token increment since its previous event."""

    request_id: str
    new_token_ids: tuple[int, ...]
    finish_reason: str | None = None
    prompt_len: int = 0
    delta_logprobs: PositionLogprobs | None = None
    prompt_logprobs: tuple[PositionLogprobs | None, ...] | None = None
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


@dataclass(frozen=True, slots=True)
class UtilityEvent:
    """Result of one utility query."""

    call_id: int
    result: dict[str, Any] | None = None
    failure_message: str | None = None


@dataclass(frozen=True, slots=True)
class SchedulerReady:
    """Startup handshake emitted after model and cache initialization."""

    protocol_version: int
    engine_version: str
    max_model_len: int
    num_gpu_blocks: int
    max_num_seqs: int


@dataclass(frozen=True, slots=True)
class SchedulerFailed:
    """Fatal scheduler-process failure broadcast to every active request."""

    message: str


@dataclass(frozen=True, slots=True)
class ReplicaReady:
    """Startup handshake from one rank in a data-parallel process grid."""

    global_rank: int
    scheduler: SchedulerReady | None = None


@dataclass(frozen=True, slots=True)
class ReplicaFailed:
    """Fatal startup or runtime failure from one data-parallel rank."""

    global_rank: int
    message: str


@dataclass(frozen=True, slots=True)
class SchedulerEvents:
    """One scheduler step's request and utility events."""

    outputs: tuple[RequestEvent, ...] = ()
    utility_output: UtilityEvent | None = None
    timestamp: float = field(default_factory=time.monotonic)


type SchedulerEvent = (
    SchedulerReady | SchedulerFailed | ReplicaReady | ReplicaFailed | SchedulerEvents
)
