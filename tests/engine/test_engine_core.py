"""Engine core wire protocol, plus generation through a child process.

Two tiers share the file:

* the serialisation tests are pure CPU -- msgpack frames in, dataclasses out --
  and pin the wire contract field by field, so a payload change without a
  protocol bump fails here first;
* the generation tests need a GPU and a checkpoint, and drive a live
  :class:`~rapid_llm.engine.engine_core_client.EngineCoreClient` so the whole
  parent -> child -> parent loop is what is verified (the scope of vLLM's
  ``tests/v1/engine/test_engine_core.py``).

Usage:
    pytest tests/engine/test_engine_core.py
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from rapid_llm.engine.engine_core import (
    PROTOCOL_VERSION,
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreReady,
    EngineCoreRequest,
    EngineCoreRequestType,
    UtilityOutput,
    UtilityRequest,
    decode_abort,
    decode_outputs,
    decode_ready,
    decode_request,
    decode_utility,
    encode_abort,
    encode_outputs,
    encode_ready,
    encode_request,
    encode_utility,
)
from rapid_llm.engine.sampler import PositionLogprobs, SamplingParams

_TIMEOUT = 180.0


async def collect(engine, prompt, params=None, request_id=None):
    return [chunk async for chunk in engine.generate(prompt, params, request_id)]


# --------------------------------------------------------------------------- #
# Wire protocol (CPU)
# --------------------------------------------------------------------------- #
def test_protocol_version_is_pinned():
    """The handshake compares this value; changing it is a protocol decision."""
    assert PROTOCOL_VERSION == 1


def test_request_type_tags_are_the_agreed_bytes():
    """The tag *is* the frame; reordering these silently misroutes commands."""
    assert EngineCoreRequestType.ADD.value == b"\x00"
    assert EngineCoreRequestType.ABORT.value == b"\x01"
    assert EngineCoreRequestType.UTILITY.value == b"\x02"
    assert EngineCoreRequestType.SHUTDOWN.value == b"\x03"


def test_request_round_trips_field_by_field():
    request = EngineCoreRequest(
        request_id="r-1",
        prompt_token_ids=[11, 22, 33],
        sampling_params=SamplingParams(
            temperature=0.0,
            top_p=0.8,
            max_gen_len=7,
            repetition_penalty=1.2,
            stop_on_repeat=False,
            logprobs=5,
            prompt_logprobs=2,
        ),
        arrival_time=1234.5,
    )
    decoded = decode_request(encode_request(request))
    assert decoded.request_id == request.request_id
    assert decoded.prompt_token_ids == request.prompt_token_ids
    assert decoded.arrival_time == request.arrival_time
    assert decoded.sampling_params == request.sampling_params


def test_received_sampling_params_are_revalidated():
    """``__post_init__`` runs on the far side: a malformed payload cannot be smuggled in."""
    from rapid_llm.engine.engine_core import decode_sampling_params

    with pytest.raises(ValueError):
        decode_sampling_params(
            {
                "temperature": -1.0,
                "top_p": 0.9,
                "max_gen_len": 4,
                "repetition_penalty": 1.1,
                "stop_on_repeat": True,
                "logprobs": None,
                "prompt_logprobs": None,
            }
        )


def test_output_frame_round_trips_tokens_and_logprobs():
    record = PositionLogprobs(
        token_id=7, logprob=-0.25, top_token_ids=(7, 3), top_logprobs=(-0.25, -1.5)
    )
    frame = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="r-1",
                new_token_ids=[7],
                prompt_len=12,
                delta_logprobs=record,
            ),
            EngineCoreOutput(
                request_id="r-2",
                new_token_ids=[],
                finish_reason="length",
                prompt_len=3,
                prompt_logprobs=[None, record],
            ),
        ],
        utility_output=UtilityOutput(call_id=3, result={"running": 1, "waiting": 2}),
        timestamp=1000.0,
    )
    decoded = decode_outputs(encode_outputs(frame))
    assert decoded.timestamp == 1000.0
    assert decoded.engine_dead is None
    first, second = decoded.outputs
    assert first.new_token_ids == [7]
    assert first.delta_logprobs == record
    assert first.prompt_len == 12
    assert not first.finished
    assert second.finished and second.finish_reason == "length"
    assert second.prompt_logprobs == [None, record]
    assert decoded.utility_output.call_id == 3
    assert decoded.utility_output.result == {"running": 1, "waiting": 2}


def test_invalid_admission_frames_round_trip():
    frame = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(
                request_id="bad",
                new_token_ids=[],
                finish_reason="invalid",
                error="prompt is empty after tokenisation",
            )
        ]
    )
    decoded = decode_outputs(encode_outputs(frame))
    assert decoded.outputs[0].error == "prompt is empty after tokenisation"
    assert decoded.outputs[0].finished


def test_utility_and_abort_round_trip():
    utility = decode_utility(encode_utility(UtilityRequest(call_id=9, method="step_stats")))
    assert (utility.call_id, utility.method) == (9, "step_stats")
    assert decode_abort(encode_abort("r-7")) == "r-7"


def test_utility_failures_and_death_notices_round_trip():
    failure = EngineCoreOutputs(
        utility_output=UtilityOutput(call_id=1, failure_message="unknown utility method 'x'")
    )
    decoded = decode_outputs(encode_outputs(failure))
    assert decoded.utility_output.failure_message == "unknown utility method 'x'"
    assert decoded.utility_output.result is None

    death = decode_outputs(encode_outputs(EngineCoreOutputs(engine_dead="RuntimeError: boom")))
    assert death.engine_dead == "RuntimeError: boom"
    assert death.outputs == []


def test_ready_frame_round_trips():
    ready = EngineCoreReady(
        protocol_version=PROTOCOL_VERSION,
        engine_version="0.12.0",
        max_model_len=2048,
        num_gpu_blocks=512,
        max_num_seqs=16,
    )
    assert decode_ready(encode_ready(ready)) == ready


def test_output_frames_are_timestamped_on_construction():
    assert EngineCoreOutputs().timestamp > 0.0


# --------------------------------------------------------------------------- #
# Generation through a live engine core (GPU + checkpoint)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.weights
class TestEngineCoreGeneration:
    """The child runs the same engine, so these re-verify its guarantees at the
    new boundary: stream shape, parameter fidelity, and concurrency."""

    @pytest.fixture(scope="class")
    @classmethod
    def client(cls, model_dir):
        from rapid_llm.engine.engine_core_client import EngineCoreClient

        engine = EngineCoreClient.from_pretrained(
            str(model_dir), max_seq_len=512, max_num_seqs=4, use_cuda_graph=False
        )
        yield engine
        asyncio.run(engine.shutdown())

    async def test_basic_generation_streams_until_finish(self, client):
        params = SamplingParams(temperature=0.0, max_gen_len=8)
        chunks = await asyncio.wait_for(
            collect(client, "The capital of France is", params), _TIMEOUT
        )
        assert chunks, "no chunks were delivered"
        assert not any(chunk.is_finished for chunk in chunks[:-1])
        assert chunks[-1].is_finished
        assert chunks[-1].finish_reason in ("length", "eos", "repeat")
        assert "".join(chunk.delta for chunk in chunks) == chunks[-1].text

    async def test_the_handshake_numbers_are_served(self, client):
        info = client.core_info
        assert info.protocol_version == PROTOCOL_VERSION
        assert info.max_model_len == 512
        assert info.max_num_seqs == 4
        assert info.num_gpu_blocks > 0

    async def test_sampling_params_and_logprobs_cross_the_wire(self, client):
        params = SamplingParams(temperature=0.0, max_gen_len=4, logprobs=3)
        chunks = await asyncio.wait_for(collect(client, "2 + 2 =", params), _TIMEOUT)
        records = [chunk.logprobs for chunk in chunks if chunk.logprobs is not None]
        assert records, "no logprob records crossed the wire"
        assert all(len(record.top_token_ids) == 3 for record in records)
        assert all(record.logprob <= 0.0 for record in records)
        assert chunks[-1].completion_tokens > 0
        assert chunks[-1].prompt_tokens > 0

    async def test_concurrent_requests_keep_their_own_streams(self, client):
        prompts = [f"Question {i}: what is {i} plus {i}? Answer:" for i in range(4)]
        params = SamplingParams(temperature=0.0, max_gen_len=6)
        results = await asyncio.wait_for(
            asyncio.gather(*(collect(client, prompt, params) for prompt in prompts)), _TIMEOUT
        )
        assert len(results) == 4
        for chunks in results:
            assert chunks[-1].is_finished
            assert "".join(chunk.delta for chunk in chunks) == chunks[-1].text

    async def test_streams_carry_the_request_id_they_were_submitted_with(self, client):
        chunks = await asyncio.wait_for(collect(client, "hi", request_id="core-mine"), _TIMEOUT)
        assert all(chunk.request_id == "core-mine" for chunk in chunks)

    async def test_abandoning_a_stream_aborts_the_request(self, client):
        """Closing the generator mid-stream frees the child's slot; the abort
        rides the command channel and a fresh request must still be served."""
        params = SamplingParams(temperature=0.0, max_gen_len=64)
        stream = client.generate("Tell me a very long story about the sea", params)
        first = await asyncio.wait_for(anext(stream), _TIMEOUT)
        assert first is not None
        await stream.aclose()

        final = await asyncio.wait_for(
            client.generate_text("Hello", SamplingParams(temperature=0.0, max_gen_len=2)),
            _TIMEOUT,
        )
        assert final.is_finished

    async def test_duplicate_request_id_is_refused(self, client):
        params = SamplingParams(temperature=0.0, max_gen_len=32)
        stream = client.generate("hi", params, request_id="core-dup")
        await asyncio.wait_for(anext(stream), _TIMEOUT)
        try:
            with pytest.raises(ValueError):
                await anext(client.generate("hi", params, request_id="core-dup"))
        finally:
            await stream.aclose()


@pytest.mark.gpu
@pytest.mark.weights
async def test_process_and_thread_backends_agree_token_for_token(model_dir):
    """The same engine, the same detokeniser -- only the boundary differs. The
    two backends must therefore agree byte for byte. Memory: the engines are
    loaded one after the other, never simultaneously, so OOM is avoided."""
    from rapid_llm.engine.async_engine import AsyncLLMEngine
    from rapid_llm.engine.engine_core_client import EngineCoreClient

    params = SamplingParams(temperature=0.0, max_gen_len=8)
    prompt = "Q: 2 + 2 = ?\nA:"

    # Small KV-cache budget: both engines share the same GPU, and the whole
    # test must fit into the free space the class fixture (*) does not hold.
    # (*) TestEngineCoreGeneration.client, but this function is outside it.
    opts = {"max_seq_len": 128, "max_num_seqs": 1, "use_cuda_graph": False}

    thread = AsyncLLMEngine.from_pretrained(str(model_dir), **opts)
    thread.start()
    try:
        expected = await asyncio.wait_for(thread.generate_text(prompt, params), _TIMEOUT)
    finally:
        await thread.shutdown()

    async with EngineCoreClient.from_pretrained(str(model_dir), **opts) as client:
        got = await asyncio.wait_for(client.generate_text(prompt, params), _TIMEOUT)

    assert got.text == expected.text
    assert got.finish_reason == expected.finish_reason
    assert got.completion_tokens == expected.completion_tokens


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
async def test_tensor_parallel_engine_core_serves(model_dir):
    """The spawn constraint vLLM shares: the executor's TP followers are daemons,
    so the engine core itself must not be -- here it proves it can be a parent."""
    from rapid_llm.engine.engine_core_client import EngineCoreClient

    async with EngineCoreClient.from_pretrained(
        str(model_dir),
        max_seq_len=256,
        max_num_seqs=2,
        tensor_parallel_size=2,
        use_cuda_graph=False,
    ) as client:
        final = await asyncio.wait_for(
            client.generate_text(
                "The capital of France is", SamplingParams(temperature=0.0, max_gen_len=4)
            ),
            _TIMEOUT,
        )
    assert final.is_finished
    assert final.text
