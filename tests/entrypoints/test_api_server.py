"""Tests for the OpenAI-compatible HTTP layer.

A ``FakeEngine`` and FastAPI's test client drive every endpoint —
health, metrics, completions, streaming SSE — so the HTTP contract is
checked without a model or a port.

Usage:
    pytest tests/entrypoints/test_api_server.py
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="needs the `serve` extra")

from fastapi.testclient import TestClient

from rapid_llm.engine.async_engine import StreamedOutput
from rapid_llm.engine.reasoning import _CLOSE as _THINK_CLOSE
from rapid_llm.engine.sampler import PositionLogprobs
from rapid_llm.engine.tool_parser import (
    _DS_ARGS_END,
    _DS_CALLS_BEGIN,
    _DS_CALLS_END,
    _DS_FENCE,
    _DS_HEADER,
)
from rapid_llm.entrypoints.api_server import (
    ServerConfig,
    build_app,
    parse_sse,
)
from rapid_llm.tools.observability.metrics import EngineMetrics

pytestmark = pytest.mark.serving

_REPLY = "Paris is the capital."
_MODEL = "fake-model"


class FakeTokenizer:
    """Just enough tokenizer for templating and token counting.

    ``apply_chat_template`` wraps each turn in visible markers so a test can
    assert the template ran, rather than inferring it from a token count.
    """

    chat_template = "{{ messages }}"

    def __init__(self) -> None:
        self.templated: list[list[dict[str, str]]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.templated.append(list(messages))
        turns = "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return f"{turns}<|assistant|>"

    def encode(self, text, add_special_tokens=True):
        return text.split()

    def decode(self, token_ids) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)


class FakeEngine:
    """Streams a fixed reply word by word and records every request it saw."""

    def __init__(self, reply: str = _REPLY) -> None:
        self.tokenizer = FakeTokenizer()
        self.metrics = EngineMetrics()
        self._reply = reply
        self.seen: list[tuple[str, object]] = []
        self.started = False

    def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.started = False

    async def generate(self, prompt, sampling_params=None, request_id=None):
        self.seen.append((prompt, sampling_params))
        text = ""
        pieces = self._reply.split(" ")
        k = getattr(sampling_params, "logprobs", None)
        prompt_k = getattr(sampling_params, "prompt_logprobs", None)
        for index, piece in enumerate(pieces):
            delta = piece if index == 0 else " " + piece
            text += delta
            last = index == len(pieces) - 1
            record = None
            if k is not None:
                record = PositionLogprobs(
                    token_id=index + 10,
                    logprob=-0.5,
                    top_token_ids=tuple(index + 11 + j for j in range(k)),
                    top_logprobs=tuple(-0.5 - j for j in range(k)),
                )
            prompt_records = None
            if last and prompt_k is not None:
                prompt_len = len(self.tokenizer.encode(prompt))
                prompt_records = (
                    None,
                    *(
                        PositionLogprobs(
                            token_id=position,
                            logprob=-1.0,
                            top_token_ids=(position,),
                            top_logprobs=(-1.0,),
                        )
                        for position in range(1, prompt_len)
                    ),
                )
            yield StreamedOutput(
                request_id=request_id or "fake",
                delta=delta,
                text=text,
                finish_reason="eos" if last else None,
                prompt_tokens=len(self.tokenizer.encode(prompt)),
                completion_tokens=index + 1,
                logprobs=record,
                prompt_logprobs=prompt_records,
            )

    async def generate_text(self, prompt, sampling_params=None, request_id=None):
        final = None
        async for chunk in self.generate(prompt, sampling_params, request_id):
            final = chunk
        return final


@pytest.fixture
def engine() -> FakeEngine:
    return FakeEngine()


def make_client(engine: FakeEngine, **config_kwargs) -> TestClient:
    config = ServerConfig(model_dir="/nonexistent", served_model_name=_MODEL, **config_kwargs)
    return TestClient(build_app(config, engine=engine))


@pytest.fixture
def client(engine) -> TestClient:
    with make_client(engine) as started:
        yield started


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_health_reports_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_metrics_endpoint_serves_prometheus_text(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "rapid_llm:num_requests_running" in response.text


def test_metrics_endpoint_renders_empty_without_a_registry(engine):
    """A front end with no registry of its own (the DP coordinator) must not fail."""
    engine.metrics = None
    with make_client(engine) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert response.text == ""


def test_the_engine_is_started_by_the_app_lifespan(engine):
    with make_client(engine):
        assert engine.started


def test_models_lists_the_served_name(client):
    body = client.get("/v1/models").json()

    assert body["object"] == "list"
    assert [card["id"] for card in body["data"]] == [_MODEL]


# --------------------------------------------------------------------------- #
# /v1/completions
# --------------------------------------------------------------------------- #
def test_completion_returns_the_openai_shape(client):
    body = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "max_tokens": 8}
    ).json()

    assert body["object"] == "text_completion"
    assert body["model"] == _MODEL
    assert body["id"].startswith("cmpl-")
    assert body["choices"] == [
        {"index": 0, "text": _REPLY, "finish_reason": "eos", "logprobs": None}
    ]


def test_completion_reports_token_usage(client):
    body = client.post("/v1/completions", json={"model": _MODEL, "prompt": "one two three"}).json()
    usage = body["usage"]

    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == len(_REPLY.split())
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_usage_counts_come_from_the_engine_not_a_reencode():
    """Decode does not round-trip through encode, so usage must be the engine's
    own counts rather than whatever encoding the texts again happens to give.
    The fixed counts below deliberately disagree with both texts' word counts.
    """

    class FixedCountsEngine(FakeEngine):
        async def generate(self, prompt, sampling_params=None, request_id=None):
            self.seen.append((prompt, sampling_params))
            yield StreamedOutput(
                request_id=request_id or "fake",
                delta=self._reply,
                text=self._reply,
                finish_reason="eos",
                prompt_tokens=7,
                completion_tokens=11,
            )

    with make_client(FixedCountsEngine()) as client:
        usage = client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello"}).json()[
            "usage"
        ]

    assert usage == {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18}


def test_completion_passes_the_prompt_through_untemplated(client, engine):
    client.post("/v1/completions", json={"model": _MODEL, "prompt": "raw prompt"})

    assert engine.seen[0][0] == "raw prompt"
    assert engine.tokenizer.templated == [], "/v1/completions must not apply a chat template"


def test_sampling_fields_reach_the_engine(client, engine):
    client.post(
        "/v1/completions",
        json={
            "model": _MODEL,
            "prompt": "Hello",
            "max_tokens": 11,
            "temperature": 0.25,
            "top_p": 0.5,
            "repetition_penalty": 1.3,
        },
    )
    params = engine.seen[0][1]

    assert params.max_gen_len == 11
    assert params.temperature == pytest.approx(0.25)
    assert params.top_p == pytest.approx(0.5)
    assert params.repetition_penalty == pytest.approx(1.3)


def test_the_protocol_defaults_match_openai_not_the_cli(client, engine):
    """A wire client expects OpenAI's 1.0/1.0, not rapid_llm's own 0.6/0.9."""
    client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello"})
    params = engine.seen[0][1]

    assert params.temperature == pytest.approx(1.0)
    assert params.top_p == pytest.approx(1.0)
    assert params.repetition_penalty == pytest.approx(1.0)


def test_streamed_completion_deltas_rebuild_the_reply(client):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "stream": True}
    )
    frames = parse_sse(response.text)

    assert "".join(f["choices"][0]["text"] for f in frames) == _REPLY
    assert frames[-1]["choices"][0]["finish_reason"] == "eos"
    assert all(f["choices"][0]["finish_reason"] is None for f in frames[:-1])


def test_a_stream_ends_with_the_done_sentinel(client):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "stream": True}
    )

    assert response.text.rstrip().endswith("data: [DONE]")
    assert response.headers["content-type"].startswith("text/event-stream")


def test_every_frame_of_one_stream_shares_an_id(client):
    response = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "stream": True}
    )
    ids = {f["id"] for f in parse_sse(response.text)}

    assert len(ids) == 1, "a client correlates frames by id"


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #
def test_chat_completion_returns_an_assistant_message(client):
    body = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": "Hi"}]},
    ).json()

    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    # The parsing channels serialise as null until a request turns them on.
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": _REPLY,
        "reasoning_content": None,
        "tool_calls": None,
    }
    assert body["choices"][0]["finish_reason"] == "eos"


def test_chat_applies_the_tokenizer_template_to_every_turn(client, engine):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Again"},
    ]
    client.post("/v1/chat/completions", json={"model": _MODEL, "messages": messages})

    assert engine.tokenizer.templated == [messages], "all turns must reach the template"
    assert engine.seen[0][0].endswith("<|assistant|>")


def test_chat_without_a_template_concatenates_the_turns(engine):
    """Base models have no template, and inventing one makes them echo role names."""
    with make_client(engine, chat_template=False) as client:
        client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [
                    {"role": "system", "content": "Be brief."},
                    {"role": "user", "content": "Hi"},
                ],
            },
        )

    assert engine.seen[0][0] == "Be brief.\nHi"
    assert engine.tokenizer.templated == []


def test_streamed_chat_opens_with_a_role_only_delta(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": _MODEL, "messages": [{"role": "user", "content": "Hi"}], "stream": True},
    )
    frames = parse_sse(response.text)

    assert frames[0]["object"] == "chat.completion.chunk"
    assert frames[0]["choices"][0]["delta"] == {
        "role": "assistant",
        "content": None,
        "reasoning_content": None,
        "tool_calls": None,
    }
    content = "".join(f["choices"][0]["delta"].get("content") or "" for f in frames)
    assert content == _REPLY
    # The finish reason rides its own empty-delta frame — OpenAI's own shape —
    # so no text can arrive after it.
    assert frames[-1]["choices"][0]["finish_reason"] == "eos"
    assert frames[-1]["choices"][0]["delta"]["content"] is None
    assert all(f["choices"][0]["finish_reason"] is None for f in frames[:-1])


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_more_than_one_completion_is_refused_rather_than_ignored(client):
    """Silently returning one completion for ``n=4`` is undetectable by a client."""
    response = client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello", "n": 4})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"model": _MODEL, "prompt": ""}, id="empty-prompt"),
        pytest.param({"model": _MODEL, "prompt": "x", "temperature": -1}, id="negative-temp"),
        pytest.param({"model": _MODEL, "prompt": "x", "top_p": 0}, id="zero-top-p"),
        pytest.param({"model": _MODEL, "prompt": "x", "top_p": 1.5}, id="top-p-above-one"),
        pytest.param({"model": _MODEL, "prompt": "x", "max_tokens": 0}, id="zero-max-tokens"),
        pytest.param({"prompt": "x"}, id="missing-model"),
    ],
)
def test_invalid_bodies_are_rejected(client, body):
    assert client.post("/v1/completions", json=body).status_code == 422


def test_chat_requires_at_least_one_message(client):
    response = client.post("/v1/chat/completions", json={"model": _MODEL, "messages": []})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Which engine the lifespan builds
# --------------------------------------------------------------------------- #
def test_two_replicas_build_the_data_parallel_engine(monkeypatch):
    """``data_parallel_size > 1`` must build the DP front end, with every knob.

    The failure mode of a dropped field is a silently wrong engine — decode
    graphs off, a shorter context window — so the whole construction is captured
    rather than a couple of spot checks. And ``device`` must be absent: a
    replica's device is its position in the grid.
    """
    from rapid_llm.entrypoints import api_server

    captured: dict = {}

    class FakeDP(FakeEngine):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)

    monkeypatch.setattr(api_server, "AsyncDataParallelEngine", FakeDP)

    config = ServerConfig(
        model_dir="/nonexistent",
        served_model_name=_MODEL,
        data_parallel_size=2,
        load_balancer="total_tokens",
        max_num_seqs=7,
        max_seq_len=1024,
    )
    with TestClient(build_app(config)) as client:
        body = client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello"}).json()

    assert body["choices"][0]["text"] == _REPLY, "the injected engine really served"
    assert captured["data_parallel_size"] == 2
    assert captured["load_balancer"] == "total_tokens"
    assert captured["tensor_parallel_size"] == 1
    assert captured["max_num_seqs"] == 7
    assert captured["max_seq_len"] == 1024
    assert captured["max_num_batched_tokens"] == 8192
    assert captured["use_cuda_graph"] is True
    assert "device" not in captured, "a replica's device is its grid position"


def test_one_replica_still_builds_the_single_process_engine(monkeypatch):
    """``data_parallel_size == 1`` must keep the original path.

    A data-parallel coordinator of one replica is a whole extra process hop for
    nothing; the default has to stay byte-for-byte the engine it was.
    """
    from rapid_llm.entrypoints import api_server

    def fake_from_pretrained(model, **kwargs):
        assert "data_parallel_size" not in kwargs
        return FakeEngine()

    monkeypatch.setattr(api_server.AsyncLLMEngine, "from_pretrained", fake_from_pretrained)

    config = ServerConfig(model_dir="/nonexistent", served_model_name=_MODEL)
    with TestClient(build_app(config)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        body = client.post("/v1/completions", json={"model": _MODEL, "prompt": "Hello"}).json()

    assert body["choices"][0]["text"] == _REPLY


# --------------------------------------------------------------------------- #
# logprobs (F6)
# --------------------------------------------------------------------------- #
def test_completion_logprobs_follow_the_openai_shape(client, engine):
    body = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "logprobs": 2}
    ).json()

    assert engine.seen[0][1].logprobs == 2, "the k must reach SamplingParams"
    block = body["choices"][0]["logprobs"]
    n_tokens = len(_REPLY.split())
    assert len(block["tokens"]) == n_tokens
    assert len(block["token_logprobs"]) == n_tokens
    assert len(block["top_logprobs"]) == n_tokens
    assert len(block["text_offset"]) == n_tokens
    assert all(len(tops) == 2 for tops in block["top_logprobs"])
    assert block["text_offset"] == sorted(block["text_offset"])


def test_completion_logprobs_zero_reports_only_the_chosen_token(client):
    body = client.post(
        "/v1/completions", json={"model": _MODEL, "prompt": "Hello", "logprobs": 0}
    ).json()

    block = body["choices"][0]["logprobs"]
    assert all(tops == {} for tops in block["top_logprobs"])
    assert all(isinstance(lp, float) for lp in block["token_logprobs"])


def test_completion_prompt_logprobs_mark_position_zero(client, engine):
    body = client.post(
        "/v1/completions",
        json={"model": _MODEL, "prompt": "one two three", "prompt_logprobs": 1},
    ).json()

    assert engine.seen[0][1].prompt_logprobs == 1
    records = body["prompt_logprobs"]
    assert len(records) == 3  # the fake encodes one token per word
    assert records[0] is None, "position 0 has no predictor"
    assert records[1]["token_id"] == 1
    assert len(records[1]["top_logprobs"]) == 1


def test_streamed_completion_chunks_carry_logprobs(client):
    response = client.post(
        "/v1/completions",
        json={"model": _MODEL, "prompt": "Hello", "stream": True, "logprobs": 1},
    )
    frames = parse_sse(response.text)

    blocks = [f["choices"][0]["logprobs"] for f in frames]
    assert all(block is not None for block in blocks)
    assert all(len(block["tokens"]) == 1 for block in blocks), "one token per chunk"
    # The streamed per-chunk offsets still add up over the whole completion.
    assert [block["text_offset"][0] for block in blocks] == sorted(
        block["text_offset"][0] for block in blocks
    )


def test_streamed_chat_preserves_logprobs_when_a_parser_withholds_text():
    """Token metadata is still observable when parser buffering emits no text frame."""

    class EmptyDeltaEngine(FakeEngine):
        async def generate(self, prompt, sampling_params=None, request_id=None):
            record = PositionLogprobs(10, -0.5, (11,), (-0.6,))
            yield StreamedOutput(
                request_id=request_id or "fake",
                delta="",
                text="",
                finish_reason="eos",
                prompt_tokens=1,
                completion_tokens=1,
                logprobs=record,
            )

    with make_client(EmptyDeltaEngine()) as local:
        response = local.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "logprobs": True,
            },
        )

    frames = parse_sse(response.text)
    blocks = [frame["choices"][0].get("logprobs") for frame in frames]
    assert any(block is not None and block["content"][0]["token"] == "<10>" for block in blocks)


def test_chat_logprobs_follow_the_openai_shape(client, engine):
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "logprobs": True,
            "top_logprobs": 1,
        },
    ).json()

    assert engine.seen[0][1].logprobs == 1
    content = body["choices"][0]["logprobs"]["content"]
    assert len(content) == len(_REPLY.split())
    assert all("token" in entry and "logprob" in entry for entry in content)
    assert all(len(entry["top_logprobs"]) == 1 for entry in content)


def test_chat_logprobs_without_top_logprobs_reports_the_sampled_token(client):
    body = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "logprobs": True,
        },
    ).json()

    content = body["choices"][0]["logprobs"]["content"]
    assert all(entry["top_logprobs"] == [] for entry in content)


def test_top_logprobs_without_logprobs_is_rejected(client):
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "top_logprobs": 2,
        },
    )

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Request-scoped reasoning / tool parsing
# --------------------------------------------------------------------------- #
# Marker spellings are assembled from the parser modules, never written out:
# the transport delivering source edits strips anything it parses as markup.
_TOOL_CALL_TEXT = (
    _DS_CALLS_BEGIN
    + _DS_HEADER
    + "get_weather\n"
    + _DS_FENCE
    + '{"city": "Tokyo"}'
    + _DS_ARGS_END
    + _DS_CALLS_END
)
_THINKING_REPLY = "Plan: check the sky." + _THINK_CLOSE + " Calling now. " + _TOOL_CALL_TEXT


class _SingleChunkEngine(FakeEngine):
    """One whole reply in one chunk, with a caller-chosen finish reason."""

    def __init__(self, reply: str, finish: str) -> None:
        super().__init__(reply)
        self._finish = finish

    async def generate(self, prompt, sampling_params=None, request_id=None):
        self.seen.append((prompt, sampling_params))
        yield StreamedOutput(
            request_id=request_id or "fake",
            delta=self._reply,
            text=self._reply,
            finish_reason=self._finish,
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_chat_with_both_parsers_splits_the_message():
    """The two switches compose: reasoning first, tools from what remains."""
    engine = FakeEngine(reply=_THINKING_REPLY)
    with make_client(engine) as client:
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_parser": "deepseek_r1",
                "tool_parser": "deepseek",
            },
        ).json()

    message = body["choices"][0]["message"]
    assert message["reasoning_content"] == "Plan: check the sky."
    assert message["content"] == " Calling now. "
    assert message["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        }
    ]
    assert body["choices"][0]["finish_reason"] == "tool_calls"


def test_reasoning_parser_alone_splits_and_leaves_content_verbatim():
    engine = FakeEngine(reply="Plan: check the sky." + _THINK_CLOSE + " Final answer.")
    with make_client(engine) as client:
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "reasoning_parser": "deepseek_r1",
            },
        ).json()

    message = body["choices"][0]["message"]
    assert message["reasoning_content"] == "Plan: check the sky."
    assert message["content"] == " Final answer."
    assert message["tool_calls"] is None
    assert body["choices"][0]["finish_reason"] == "eos"


def test_streamed_chat_channels_merge_to_the_one_shot_message():
    """The server-level axiom: streamed frames concatenate to the message."""

    def run(stream: bool):
        engine = FakeEngine(reply=_THINKING_REPLY)
        with make_client(engine) as client:
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": _MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": stream,
                    "reasoning_parser": "deepseek_r1",
                    "tool_parser": "deepseek",
                },
            )

    message = run(stream=False).json()["choices"][0]["message"]
    frames = parse_sse(run(stream=True).text)

    # nothing may follow the terminal frame: clients stop reading there
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert all(f["choices"][0]["finish_reason"] is None for f in frames[:-1])
    deltas = [f["choices"][0]["delta"] for f in frames]
    assert "".join(d.get("reasoning_content") or "" for d in deltas) == message["reasoning_content"]
    assert "".join(d.get("content") or "" for d in deltas) == message["content"]
    # tool_calls merge by index, identity first, arguments streaming after
    streamed: dict[int, dict] = {}
    for delta in deltas:
        for piece in delta.get("tool_calls") or []:
            call = streamed.setdefault(piece["index"], {"id": None, "name": "", "arguments": ""})
            call["id"] = call["id"] or piece.get("id")
            call["name"] += piece["function"].get("name") or ""
            call["arguments"] += piece["function"].get("arguments") or ""
    (call,) = message["tool_calls"]
    assert streamed == {
        0: {
            "id": call["id"],
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        }
    }


def test_a_length_cut_keeps_its_reason_even_when_calls_were_extracted():
    """A truncated call still reports its pieces, but the cut is the truth."""
    reply = _DS_CALLS_BEGIN + _DS_HEADER + "get_weather\n" + _DS_FENCE + '{"city": "To'
    engine = _SingleChunkEngine(reply, "length")
    with make_client(engine) as client:
        body = client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "tool_parser": "deepseek",
            },
        ).json()

    assert body["choices"][0]["finish_reason"] == "length"
    (call,) = body["choices"][0]["message"]["tool_calls"]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "To'}


def test_a_length_cut_still_streams_its_truncated_call_before_finishing():
    reply = _DS_CALLS_BEGIN + _DS_HEADER + "get_weather\n" + _DS_FENCE + '{"city": "To'
    engine = _SingleChunkEngine(reply, "length")
    with make_client(engine) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "tool_parser": "deepseek",
            },
        )
    frames = parse_sse(response.text)

    tail = frames[-1]
    assert tail["choices"][0]["finish_reason"] == "length"
    assert tail["choices"][0]["delta"]["content"] is None
    pieces = [
        piece for f in frames[:-1] for piece in f["choices"][0]["delta"].get("tool_calls") or []
    ]
    assert pieces[0]["id"] == "call_0"
    assert "".join(p["function"].get("arguments") or "" for p in pieces) == '{"city": "To'


def test_unknown_parser_names_are_rejected_by_the_schema(client):
    """The switches are a closed set: typos must 422, not silently no-op."""
    for field in ("reasoning_parser", "tool_parser"):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": _MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                field: "bogus",
            },
        )
        assert response.status_code == 422, field
