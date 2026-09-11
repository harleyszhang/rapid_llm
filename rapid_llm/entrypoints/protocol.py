"""Request and response schemas for the OpenAI-compatible endpoints.

Pydantic models mirror the OpenAI wire format —
:class:`CompletionRequest`, :class:`ChatCompletionRequest` and the
response objects — so parsing, defaults and validation live once where
FastAPI can see them.

Usage:
    req = CompletionRequest(model="m", prompt="hi")
    params = req.to_sampling_params()
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from ..engine.sampler import SamplingParams

# OpenAI's default is 1.0 for both. rapid_llm's own default (0.6 / 0.9) belongs
# to its CLI, not to a wire protocol that clients expect to behave like OpenAI's.
_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 1.0


def _request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _GenerationOptions(BaseModel):
    """Sampling fields shared by the completion and chat-completion bodies."""

    model: str
    max_tokens: int | None = Field(default=16, ge=1)
    temperature: float = Field(default=_DEFAULT_TEMPERATURE, ge=0.0)
    top_p: float = Field(default=_DEFAULT_TOP_P, gt=0.0, le=1.0)
    stream: bool = False
    # OpenAI expresses this as presence/frequency penalties on a different scale;
    # this is rapid_llm's own multiplicative penalty, passed through by name.
    repetition_penalty: float = Field(default=1.0, gt=0.0)
    n: int = Field(default=1, ge=1)

    @field_validator("n")
    @classmethod
    def _only_one_completion(cls, value: int) -> int:
        if value != 1:
            raise ValueError("n > 1 is not supported; the sampler emits one completion")
        return value

    def to_sampling_params(self) -> SamplingParams:
        """Translate the wire fields into engine sampling parameters."""
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_gen_len=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
        )


class CompletionRequest(_GenerationOptions):
    """Body of ``POST /v1/completions``."""

    prompt: str
    logprobs: int | None = Field(default=None, ge=0, le=20)
    # Not part of OpenAI's schema; vLLM-compatible extension, honoured by name.
    prompt_logprobs: int | None = Field(default=None, ge=0, le=20)

    @field_validator("prompt")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("prompt must not be empty")
        return value

    def to_sampling_params(self) -> SamplingParams:
        params = super().to_sampling_params()
        params.logprobs = self.logprobs
        params.prompt_logprobs = self.prompt_logprobs
        return params


class ChatMessage(BaseModel):
    """One turn of a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(_GenerationOptions):
    """Body of ``POST /v1/chat/completions``."""

    messages: list[ChatMessage] = Field(min_length=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    # Request-scoped parsing switches — rapid_llm's own extension. vLLM and
    # SGLang fix one parser per server process; here each request states what
    # its model emits, so one deployment can serve mixed clients.
    reasoning_parser: Literal["deepseek_r1"] | None = None
    tool_parser: Literal["deepseek", "qwen"] | None = None

    @field_validator("top_logprobs")
    @classmethod
    def _top_logprobs_need_logprobs(cls, value: int | None, info) -> int | None:
        if value is not None and not info.data.get("logprobs", False):
            raise ValueError("top_logprobs requires logprobs to be true")
        return value

    def to_sampling_params(self) -> SamplingParams:
        params = super().to_sampling_params()
        # OpenAI reports the sampled token alone when only the switch is on;
        # k alternatives widen the record. k == 0 means the sampled token only.
        params.logprobs = (self.top_logprobs or 0) if self.logprobs else None
        return params


class UsageInfo(BaseModel):
    """Token accounting for one response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionLogprobs(BaseModel):
    """OpenAI's per-token logprob block: four parallel arrays."""

    tokens: list[str]
    token_logprobs: list[float]
    top_logprobs: list[dict[str, float]]
    text_offset: list[int]


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None
    logprobs: CompletionLogprobs | None = None


class CompletionResponse(BaseModel):
    """Body of a non-streaming ``/v1/completions`` reply."""

    id: str = Field(default_factory=lambda: _request_id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    # vLLM-style extension: per-prompt-position records (position 0 is None).
    prompt_logprobs: list[dict | None] | None = None


class FunctionCall(BaseModel):
    """The callable half of a tool call: a name plus its raw JSON text."""

    name: str
    arguments: str = ""


class MessageToolCall(BaseModel):
    """A finished tool call as it sits inside an assistant message."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str = ""
    # DeepSeek-style extension for thinking models: the chain-of-thought
    # channel, split from content only when the request asks for it.
    reasoning_content: str | None = None
    # OpenAI semantics: present only when the model actually placed calls.
    tool_calls: list[MessageToolCall] | None = None


class ChatTopLogprob(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class ChatTokenLogprob(ChatTopLogprob):
    top_logprobs: list[ChatTopLogprob] = Field(default_factory=list)


class ChatCompletionLogprobs(BaseModel):
    content: list[ChatTokenLogprob]


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionResponse(BaseModel):
    """Body of a non-streaming ``/v1/chat/completions`` reply."""

    id: str = Field(default_factory=lambda: _request_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)


class DeltaFunctionCall(BaseModel):
    """Streaming half of a tool call: each field optional until it arrives."""

    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    """One incremental piece of a tool call, keyed by its call index."""

    index: int
    id: str | None = None
    type: Literal["function"] = "function"
    function: DeltaFunctionCall = Field(default_factory=DeltaFunctionCall)


class ChatCompletionDelta(BaseModel):
    """Incremental content of a streamed chat chunk.

    ``role`` appears on the first chunk only, matching OpenAI, so clients can
    start rendering before any text arrives. ``reasoning_content`` and
    ``tool_calls`` ride the same wire shape OpenAI clients already merge by
    index; they stay ``None`` unless the request turned parsing on.
    """

    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[DeltaToolCall] | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None
    logprobs: ChatCompletionLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    """One ``data:`` frame of a streamed chat completion."""

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionChunk(BaseModel):
    """One ``data:`` frame of a streamed text completion."""

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "rapid_llm"


class ModelList(BaseModel):
    """Body of ``GET /v1/models``."""

    object: Literal["list"] = "list"
    data: list[ModelCard]
