"""OpenAI-compatible HTTP server over :class:`AsyncLLMEngine`.

:class:`OpenAIServer` implements ``/v1/completions``, ``/v1/chat/completions``
(streaming via SSE) and ``/metrics``; ``build_app`` / ``run_server`` wire a
FastAPI app around one engine, with heavy deps imported lazily.

Usage:
    app = build_app(config, engine)
    run_server(config, host, port)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..engine.async_data_parallel import AsyncDataParallelEngine
from ..engine.async_engine import AsyncLLMEngine, StreamedOutput
from ..engine.reasoning import ReasoningSplitter, for_family
from ..engine.sampler import PositionLogprobs, SamplingParams
from ..engine.scheduler import DEFAULT_MAX_CHUNK_SIZE
from ..engine.tool_parser import ToolCall, ToolCallDelta, ToolParser
from ..utils.logger import get_logger
from ..utils.prompt_templates import get_prompter
from .protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionLogprobs,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatTokenLogprob,
    ChatTopLogprob,
    CompletionChoice,
    CompletionChunk,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    DeltaFunctionCall,
    DeltaToolCall,
    FunctionCall,
    MessageToolCall,
    ModelCard,
    ModelList,
    UsageInfo,
    _request_id,
)

if TYPE_CHECKING:
    from ..engine.engine_core_client import EngineCoreClient

    #: What a server run may sit on: the in-process async engine (one replica),
    #: the data-parallel coordinator over several, or a separate engine core
    #: process. All three answer the same generate/metrics/tokenizer calls.
    EngineBackend = AsyncLLMEngine | AsyncDataParallelEngine | EngineCoreClient

logger = get_logger(__name__)

# Terminator every OpenAI-compatible stream ends with; clients watch for it.
_SSE_DONE = "data: [DONE]\n\n"


def _message_tool_call(call: ToolCall) -> MessageToolCall:
    """Lift a parser call onto the wire shape of a chat message."""
    return MessageToolCall(
        id=call.id, function=FunctionCall(name=call.name, arguments=call.arguments)
    )


def _delta_tool_call(delta: ToolCallDelta) -> DeltaToolCall:
    """Lift one stream delta onto the wire shape clients merge by index."""
    return DeltaToolCall(
        index=delta.index,
        id=delta.id,
        function=DeltaFunctionCall(name=delta.name, arguments=delta.arguments),
    )


def _build_parsers(
    body: ChatCompletionRequest,
) -> tuple[ReasoningSplitter | None, ToolParser | None]:
    """Instantiate what the request asked for; ``(None, None)`` passes text through."""
    splitter = for_family(body.reasoning_parser) if body.reasoning_parser else None
    tool_parser = ToolParser.for_model(body.tool_parser) if body.tool_parser else None
    return splitter, tool_parser


def _require_fastapi() -> Any:
    """Import FastAPI, explaining the extra when it is missing.

    Serving is an optional extra, so the dependency is imported here rather than
    at module import: ``import rapid_llm`` must keep working on a machine that
    only ever runs offline generation.
    """
    try:
        import fastapi
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(
            "the API server needs FastAPI and uvicorn; install them with "
            "`pip install 'rapid-llm[serve]'`"
        ) from exc
    return fastapi


@dataclass
class ServerConfig:
    """How to build the engine a server run sits on.

    Attributes:
        model_dir: Checkpoint directory.
        served_model_name: Name reported by ``/v1/models`` and echoed in
            responses; defaults to the checkpoint directory's name.
        max_seq_len: Context window, and the per-slot KV cache size.
        max_num_seqs: Concurrency ceiling — how many requests may decode together.
        max_num_batched_tokens: Padded token budget for one prefill group.
        enable_chunked_prefill: Let a prompt prefill across steps, as in vLLM.
            On by default so one long prompt cannot stall every decode in
            flight; off needs ``max_num_batched_tokens >= max_seq_len``.
        max_chunk_size: Maximum prefill chunk per request; ``0`` disables
            chunking and favours throughput over decode-tail latency.
        max_gpu_num_blocks: Manual KV-cache size in tokens; profiled when ``None``.
        device: Torch device string.
        use_cuda_graph: Capture decode CUDA graphs.
        quantization: Runtime weight quantisation for fp16 checkpoints.
        tensor_parallel_size: GPUs this replica's weights are split over. Above 1
            the engine spawns the follower ranks itself and the server is still
            one process with one scheduler.
        enable_expert_parallel: Split MoE experts whole-across-ranks over the
            TP group instead of TP-splitting each expert (vLLM semantics).
            Decode keeps its CUDA graphs (lazy capture). Per replica, like the
            fields around it.
        kv_cache_dtype: KV-cache element type (``"auto"`` or an fp8 spelling).
        enable_prefix_cache: Reuse block-aligned prompt prefixes in the local
            replica cache.
        prefix_cache_blocks: Optional prefix-cache capacity in 16-token blocks.
        enable_preemption: Allow recompute-based oversubscription of cache slots.
        data_parallel_size: Whole-model replicas serving this one endpoint,
            combined with ``tensor_parallel_size`` into the usual
            ``dp x tp`` GPU grid. Above 1 the lifespan builds an
            :class:`~rapid_llm.engine.async_data_parallel.AsyncDataParallelEngine`
            instead — the replicas multiply concurrent decode, which is what a
            server is for; the fields above apply *per replica*.
        load_balancer: Which replica each request is routed to, one of
            :data:`~rapid_llm.engine.dp_load_balancer.LOAD_BALANCERS`.
        chat_template: ``True`` applies the tokenizer's chat template to
            ``/v1/chat/completions`` messages. Turn it off for base models, which
            have no template and degenerate when given one.
        engine_backend: Where scheduling runs. ``"thread"`` runs the
            continuous-batching engine inside this process; ``"process"``
            spawns a separate engine core process and talks to it over ZMQ
            (needs the ``serve`` extra's ``pyzmq``/``msgpack``), which keeps
            the event loop out of the GPU step path. HTTP behaviour is the
            same either way.
    """

    model_dir: str
    served_model_name: str | None = None
    max_seq_len: int = 2048
    max_num_seqs: int = 32
    max_num_batched_tokens: int = 8192
    enable_chunked_prefill: bool = True
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
    max_gpu_num_blocks: int | None = None
    device: str = "cuda"
    use_cuda_graph: bool = True
    quantization: str | None = None
    tensor_parallel_size: int = 1
    enable_expert_parallel: bool = False
    kv_cache_dtype: str = "auto"
    enable_prefix_cache: bool = False
    prefix_cache_blocks: int | None = None
    enable_preemption: bool = False
    data_parallel_size: int = 1
    load_balancer: str = "round_robin"
    chat_template: bool = True
    engine_backend: str = "thread"

    @property
    def model_name(self) -> str:
        from pathlib import Path

        return self.served_model_name or Path(self.model_dir).name


class _LogprobsCollector:
    """Accumulate streamed :class:`PositionLogprobs` into OpenAI-shaped blocks.

    Owns the running UTF-8 byte offset, which is why it lives for the whole
    request: a token's ``text_offset`` depends on every token before it. Token
    text comes from decoding each id alone — re-encoding text would not
    round-trip at BPE boundaries.
    """

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._entries: list[tuple[PositionLogprobs, str, int]] = []
        self._offset = 0

    def _token_text(self, token_id: int) -> str:
        return self._tokenizer.decode([token_id])

    def add(self, record: PositionLogprobs) -> None:
        text = self._token_text(record.token_id)
        self._entries.append((record, text, self._offset))
        self._offset += len(text.encode())

    def completion_block(self, *, last_only: bool = False) -> CompletionLogprobs:
        entries = self._entries[-1:] if last_only else self._entries
        return CompletionLogprobs(
            tokens=[text for _, text, _ in entries],
            token_logprobs=[record.logprob for record, _, _ in entries],
            top_logprobs=[
                {
                    self._token_text(t): lp
                    for t, lp in zip(record.top_token_ids, record.top_logprobs, strict=True)
                }
                for record, _, _ in entries
            ],
            text_offset=[offset for _, _, offset in entries],
        )

    def chat_block(self, *, last_only: bool = False) -> ChatCompletionLogprobs:
        entries = self._entries[-1:] if last_only else self._entries
        content = []
        for record, text, _ in entries:
            tops = []
            for t, lp in zip(record.top_token_ids, record.top_logprobs, strict=True):
                top_text = self._token_text(t)
                tops.append(
                    ChatTopLogprob(token=top_text, logprob=lp, bytes=list(top_text.encode()))
                )
            content.append(
                ChatTokenLogprob(
                    token=text,
                    logprob=record.logprob,
                    bytes=list(text.encode()),
                    top_logprobs=tops,
                )
            )
        return ChatCompletionLogprobs(content=content)


def _prompt_logprobs_block(tokenizer, records) -> list[dict | None]:
    """Serialise prompt-position records; ``None`` stays ``None`` (position 0,
    prefix-cache hits) so the client sees exactly which positions were scored.
    """
    return [
        None
        if record is None
        else {
            "token": tokenizer.decode([record.token_id]),
            "token_id": record.token_id,
            "logprob": record.logprob,
            "top_logprobs": [
                {"token": tokenizer.decode([t]), "token_id": t, "logprob": lp}
                for t, lp in zip(record.top_token_ids, record.top_logprobs, strict=True)
            ],
        }
        for record in records
    ]


class OpenAIServer:
    """Wire-protocol adapter: OpenAI JSON in, engine calls out.

    Args:
        engine: The async engine to serve from — one replica's
            :class:`~rapid_llm.engine.async_engine.AsyncLLMEngine` or the
            data-parallel front end over several. Either answers with the same
            streamed chunks, which is all this layer reads.
        model_name: Name echoed back in responses.
        chat_template: Whether to apply the tokenizer's chat template.
    """

    def __init__(
        self,
        engine: EngineBackend,
        model_name: str,
        *,
        chat_template: bool = True,
    ) -> None:
        self.engine = engine
        self.model_name = model_name
        self._prompter = get_prompter(engine.tokenizer) if chat_template else None

    # ----------------------------------------------------------------- models #
    def list_models(self) -> ModelList:
        return ModelList(data=[ModelCard(id=self.model_name)])

    def metrics_text(self) -> str:
        """The engine's registry in Prometheus exposition format.

        A front end without a registry of its own (the data-parallel
        coordinator, whose replicas each serve their own) renders empty rather
        than failing the scrape.
        """
        registry = getattr(self.engine, "metrics", None)
        return registry.render_prometheus() if registry is not None else ""

    # ------------------------------------------------------------ completions #
    async def completions(self, body: CompletionRequest):
        params = body.to_sampling_params()
        if body.stream:
            return self._stream_completion(body.prompt, params)
        return await self._full_completion(body.prompt, params)

    async def _full_completion(self, prompt: str, params: SamplingParams):
        collector = (
            _LogprobsCollector(self.engine.tokenizer) if params.logprobs is not None else None
        )
        final: StreamedOutput | None = None
        async for chunk in self.engine.generate(prompt, params):
            final = chunk
            if collector is not None and chunk.logprobs is not None:
                collector.add(chunk.logprobs)
        if final is None:
            raise RuntimeError("request produced no output")
        return CompletionResponse(
            model=self.model_name,
            choices=[
                CompletionChoice(
                    text=final.text,
                    finish_reason=final.finish_reason,
                    logprobs=collector.completion_block() if collector is not None else None,
                )
            ],
            usage=self._usage(final),
            prompt_logprobs=(
                _prompt_logprobs_block(self.engine.tokenizer, final.prompt_logprobs)
                if final.prompt_logprobs is not None
                else None
            ),
        )

    async def _stream_completion(self, prompt: str, params: SamplingParams) -> AsyncIterator[str]:
        response_id = _request_id("cmpl")
        collector = (
            _LogprobsCollector(self.engine.tokenizer) if params.logprobs is not None else None
        )
        async for chunk in self.engine.generate(prompt, params):
            block = None
            if collector is not None and chunk.logprobs is not None:
                collector.add(chunk.logprobs)
                block = collector.completion_block(last_only=True)
            frame = CompletionChunk(
                id=response_id,
                model=self.model_name,
                choices=[
                    CompletionChoice(
                        text=chunk.delta, finish_reason=chunk.finish_reason, logprobs=block
                    )
                ],
            )
            yield f"data: {frame.model_dump_json()}\n\n"
        yield _SSE_DONE

    # ------------------------------------------------------- chat completions #
    async def chat_completions(self, body: ChatCompletionRequest):
        prompt = self._render_chat(body)
        params = body.to_sampling_params()
        if body.stream:
            return self._stream_chat(prompt, params, body)
        return await self._full_chat(prompt, params, body)

    async def _full_chat(self, prompt: str, params: SamplingParams, body: ChatCompletionRequest):
        collector = (
            _LogprobsCollector(self.engine.tokenizer) if params.logprobs is not None else None
        )
        final: StreamedOutput | None = None
        async for chunk in self.engine.generate(prompt, params):
            final = chunk
            if collector is not None and chunk.logprobs is not None:
                collector.add(chunk.logprobs)
        if final is None:
            raise RuntimeError("request produced no output")
        splitter, tool_parser = _build_parsers(body)
        message = ChatCompletionMessage(content=final.text)
        finish = final.finish_reason
        if splitter is not None or tool_parser is not None:
            reasoning, content = "", final.text
            if splitter is not None:
                reasoning, content = splitter.feed(final.text)
                tail_reasoning, tail_content = splitter.finish()
                reasoning, content = reasoning + tail_reasoning, content + tail_content
            calls: list[ToolCall] = []
            if tool_parser is not None:
                content, calls = tool_parser.parse(content)
            message = ChatCompletionMessage(
                content=content,
                reasoning_content=reasoning or None,
                tool_calls=[_message_tool_call(call) for call in calls] or None,
            )
            # A length cut may have sliced the markup mid-call; the cut is the
            # honest finish reason, the calls it produced are still reported.
            if calls and finish != "length":
                finish = "tool_calls"
        return ChatCompletionResponse(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=message,
                    finish_reason=finish,
                    logprobs=collector.chat_block() if collector is not None else None,
                )
            ],
            usage=self._usage(final),
        )

    async def _stream_chat(
        self, prompt: str, params: SamplingParams, body: ChatCompletionRequest
    ) -> AsyncIterator[str]:
        response_id = _request_id("chatcmpl")

        def frame(
            delta: ChatCompletionDelta,
            reason: str | None,
            logprobs: ChatCompletionLogprobs | None = None,
        ) -> str:
            chunk = ChatCompletionChunk(
                id=response_id,
                model=self.model_name,
                choices=[
                    ChatCompletionChunkChoice(delta=delta, finish_reason=reason, logprobs=logprobs)
                ],
            )
            return f"data: {chunk.model_dump_json()}\n\n"

        collector = (
            _LogprobsCollector(self.engine.tokenizer) if params.logprobs is not None else None
        )
        splitter, tool_parser = _build_parsers(body)
        saw_calls = False

        # OpenAI opens with a role-only delta, before any text exists.
        yield frame(ChatCompletionDelta(role="assistant"), None)
        finish: str | None = None
        async for chunk in self.engine.generate(prompt, params):
            block = None
            if collector is not None and chunk.logprobs is not None:
                collector.add(chunk.logprobs)
                block = collector.chat_block(last_only=True)
            reasoning_text, content_text = "", chunk.delta
            if splitter is not None:
                reasoning_text, content_text = splitter.feed(chunk.delta)
            tool_deltas: list[ToolCallDelta] = []
            if tool_parser is not None:
                step = tool_parser.feed(content_text)
                content_text, tool_deltas = step.content, step.calls
            saw_calls = saw_calls or bool(tool_deltas)
            finish = chunk.finish_reason or finish
            # A chunk whose text the suffix window swallowed emits no frame;
            # its logprob block rides whichever frame the chunk does emit.
            first = True
            if reasoning_text:
                yield frame(
                    ChatCompletionDelta(reasoning_content=reasoning_text),
                    None,
                    block if first else None,
                )
                first = False
            if content_text:
                yield frame(
                    ChatCompletionDelta(content=content_text), None, block if first else None
                )
                first = False
            if tool_deltas:
                yield frame(
                    ChatCompletionDelta(
                        tool_calls=[_delta_tool_call(delta) for delta in tool_deltas]
                    ),
                    None,
                    block if first else None,
                )
                first = False
            # A reasoning/tool parser may intentionally withhold this token's
            # text until it has seen more bytes.  Its probability is still a
            # token-level result, so do not lose it merely because there was no
            # visible text delta to attach it to.
            if first and block is not None:
                yield frame(ChatCompletionDelta(), None, block)
        # End-of-stream flush: the suffix windows release what they held and a
        # truncated call surfaces its pieces — all before the terminal frame,
        # because clients stop reading at finish_reason.
        tail_reasoning, tail_content = "", ""
        if splitter is not None:
            tail_reasoning, tail_content = splitter.finish()
        if tail_reasoning:
            yield frame(ChatCompletionDelta(reasoning_content=tail_reasoning), None)
        if tool_parser is not None:
            step = tool_parser.feed(tail_content)
            tail_content, tail_deltas = step.content, step.calls
            step = tool_parser.finish()
            tail_content += step.content
            tail_deltas += step.calls
            saw_calls = saw_calls or bool(tail_deltas)
            if tail_content:
                yield frame(ChatCompletionDelta(content=tail_content), None)
            if tail_deltas:
                yield frame(
                    ChatCompletionDelta(
                        tool_calls=[_delta_tool_call(delta) for delta in tail_deltas]
                    ),
                    None,
                )
        elif tail_content:
            yield frame(ChatCompletionDelta(content=tail_content), None)
        # The finish reason rides its own empty delta — OpenAI's own shape —
        # so no content can arrive after it.
        reason = finish
        if saw_calls and finish != "length":
            reason = "tool_calls"
        yield frame(ChatCompletionDelta(), reason)
        yield _SSE_DONE

    def _render_chat(self, body: ChatCompletionRequest) -> str:
        """Turn chat messages into a single prompt string.

        Without a template (base models) the turns are concatenated verbatim,
        which is the only honest thing to do: inventing ``<|im_start|>`` markers a
        base model never saw makes it echo the role names back.
        """
        if self._prompter is None:
            return "\n".join(message.content for message in body.messages)
        return self._prompter.apply([{"role": m.role, "content": m.content} for m in body.messages])

    def _usage(self, final: StreamedOutput) -> UsageInfo:
        """Report the token counts the engine already keeps.

        Re-encoding the texts would be both slower and subtly wrong: decode does
        not round-trip through encode at token boundaries, so the honest numbers
        are the ones the engine tokenised and sampled.
        """
        return UsageInfo(
            prompt_tokens=final.prompt_tokens,
            completion_tokens=final.completion_tokens,
            total_tokens=final.prompt_tokens + final.completion_tokens,
        )


def build_app(config: ServerConfig, engine: EngineBackend | None = None):
    """Assemble the FastAPI application.

    Args:
        config: Engine and serving options.
        engine: Pre-built engine; when omitted one is loaded on startup — one
            replica's async engine, the data-parallel front end when
            ``config.data_parallel_size`` is above 1, or a separate engine
            core process when ``config.engine_backend == "process"``.
            Injecting a fake here is what lets the protocol layer be tested
            without a GPU or a checkpoint.
    """
    fastapi = _require_fastapi()
    from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

    owns_engine = engine is None
    state: dict[str, EngineBackend | OpenAIServer | None] = {
        "engine": engine,
        "server": None,
    }

    @asynccontextmanager
    async def lifespan(_app):
        if state["engine"] is None:
            if config.data_parallel_size > 1:
                if config.engine_backend == "process":
                    raise RuntimeError(
                        "engine_backend='process' serves one replica; lower "
                        "data_parallel_size to 1 to use it"
                    )
                # ``device`` is deliberately absent: a replica's device is its
                # position in the grid, and the coordinator loads no model.
                logger.info(
                    "loading %s for serving on %d replicas",
                    config.model_dir,
                    config.data_parallel_size,
                )
                state["engine"] = AsyncDataParallelEngine(
                    model=config.model_dir,
                    **({"device": "cpu"} if config.device == "cpu" else {}),
                    data_parallel_size=config.data_parallel_size,
                    load_balancer=config.load_balancer,
                    max_seq_len=config.max_seq_len,
                    max_num_seqs=config.max_num_seqs,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    enable_chunked_prefill=config.enable_chunked_prefill,
                    max_chunk_size=config.max_chunk_size,
                    max_gpu_num_blocks=config.max_gpu_num_blocks,
                    use_cuda_graph=config.use_cuda_graph,
                    quantization=config.quantization,
                    tensor_parallel_size=config.tensor_parallel_size,
                    enable_expert_parallel=config.enable_expert_parallel,
                    kv_cache_dtype=config.kv_cache_dtype,
                    enable_prefix_cache=config.enable_prefix_cache,
                    prefix_cache_blocks=config.prefix_cache_blocks,
                    enable_preemption=config.enable_preemption,
                )
            else:
                engine_kwargs: dict[str, Any] = {
                    "max_seq_len": config.max_seq_len,
                    "max_num_seqs": config.max_num_seqs,
                    "max_num_batched_tokens": config.max_num_batched_tokens,
                    "enable_chunked_prefill": config.enable_chunked_prefill,
                    "max_chunk_size": config.max_chunk_size,
                    "max_gpu_num_blocks": config.max_gpu_num_blocks,
                    "device": config.device,
                    "use_cuda_graph": config.use_cuda_graph,
                    "quantization": config.quantization,
                    "tensor_parallel_size": config.tensor_parallel_size,
                    "enable_expert_parallel": config.enable_expert_parallel,
                    "kv_cache_dtype": config.kv_cache_dtype,
                    "enable_prefix_cache": config.enable_prefix_cache,
                    "prefix_cache_blocks": config.prefix_cache_blocks,
                    "enable_preemption": config.enable_preemption,
                }
                if config.engine_backend == "process":
                    logger.info("loading %s in a separate engine core process", config.model_dir)
                    # Imported here, not at module scope: ZMQ is an optional
                    # dependency the thread backend never needs.
                    from ..engine.engine_core_client import EngineCoreClient

                    state["engine"] = EngineCoreClient.from_pretrained(
                        config.model_dir, **engine_kwargs
                    )
                else:
                    logger.info("loading %s for serving", config.model_dir)
                    state["engine"] = AsyncLLMEngine.from_pretrained(
                        config.model_dir, **engine_kwargs
                    )
        active: EngineBackend = state["engine"]
        active.start()
        state["server"] = OpenAIServer(
            active, config.model_name, chat_template=config.chat_template
        )
        yield
        if owns_engine:
            await active.shutdown()

    app = fastapi.FastAPI(title="rapid_llm", lifespan=lifespan)

    def server() -> OpenAIServer:
        return state["server"]

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics():
        # Prometheus' own content type; the version token is part of the spec.
        return PlainTextResponse(server().metrics_text(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models", response_model=ModelList)
    async def models() -> ModelList:
        return server().list_models()

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        result = await server().completions(body)
        if body.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        result = await server().chat_completions(body)
        if body.stream:
            return StreamingResponse(result, media_type="text/event-stream")
        return result

    @app.exception_handler(ValueError)
    async def value_error_handler(_request, exc: ValueError):
        # Prompts the engine refuses (empty, or past the context window) are
        # client errors; without this they would surface as a 500.
        return JSONResponse(
            status_code=400, content={"error": {"message": str(exc), "type": "invalid_request"}}
        )

    return app


def run_server(config: ServerConfig, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Serve until interrupted."""
    _require_fastapi()
    import uvicorn

    uvicorn.run(build_app(config), host=host, port=port, log_level="info")


def parse_sse(payload: str) -> list[dict[str, Any]]:
    """Decode an SSE body into its JSON frames, dropping the ``[DONE]`` sentinel.

    Lives here rather than in the tests so that the framing rules have exactly one
    definition; a test that re-implemented the split would pass while the server
    emitted subtly malformed frames.
    """
    frames = []
    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        body = line.removeprefix("data: ").strip()
        if body == "[DONE]":
            continue
        frames.append(json.loads(body))
    return frames
