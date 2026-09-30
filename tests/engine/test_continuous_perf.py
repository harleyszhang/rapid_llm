"""Performance regression guards for continuous batching.

Requests arrive on a schedule while the engine steps; the guard asserts
throughput and latency improve over the static baseline by at least the
recorded margins — catching regressions, not chasing speed.

Usage:
    pytest tests/engine/test_continuous_perf.py
"""

from __future__ import annotations

import gc
import time

import pytest
import torch

from rapid_llm.engine.llm_engine import LLMEngine
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Scheduler, SchedulerConfig

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

_MAX_SEQ_LEN = 512
_KV_BLOCKS = 8192
_MAX_GEN = 48
_REQUESTS = 6
_INTERVAL = 0.05  # seconds between arrivals

# Deliberately conservative: the measured gap is several times each of these.
_MIN_THROUGHPUT_GAIN = 2.0
_MIN_LATENCY_GAIN = 2.0

PROMPTS = [
    "Explain what a GPU does.",
    "Write a short poem about rain.",
    "List four prime numbers.",
    "What is the capital of Japan?",
    "Describe the colour blue.",
    "Name three programming languages.",
]
GREEDY = SamplingParams(temperature=0.0, max_gen_len=_MAX_GEN, repetition_penalty=1.0)


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _arrival_schedule() -> list[tuple[float, str]]:
    return [(index * _INTERVAL, PROMPTS[index % len(PROMPTS)]) for index in range(_REQUESTS)]


def _serve_serially(model_dir) -> tuple[float, int, list[float]]:
    """Baseline: the one-shot path can only take the next request once free."""
    engine = LLMEngine(
        str(model_dir),
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_BLOCKS,
        use_cuda_graph=False,
    )
    try:
        encode = engine.tokenizer.encode
        LLMEngine.generate_text(
            engine, [encode(PROMPTS[0])], SamplingParams(temperature=0.0, max_gen_len=4)
        )

        torch.cuda.synchronize()
        started = time.perf_counter()
        tokens, latencies = 0, []
        for offset, prompt in _arrival_schedule():
            arrival = started + offset
            now = time.perf_counter()
            if now < arrival:
                time.sleep(arrival - now)
            text = LLMEngine.generate_text(engine, [encode(prompt)], GREEDY)[0]
            tokens += len(encode(text, add_special_tokens=False))
            latencies.append(time.perf_counter() - arrival)
        torch.cuda.synchronize()
        return time.perf_counter() - started, tokens, latencies
    finally:
        del engine
        _free()


def _serve_continuously(model_dir) -> tuple[float, int, list[float]]:
    """Continuous batching: an arrival is admitted at the next step."""
    engine = Scheduler(
        LLMEngine(
            str(model_dir),
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_BLOCKS,
            use_cuda_graph=False,
        ),
        SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=_REQUESTS),
    )
    try:
        engine.generate(PROMPTS[:1], SamplingParams(temperature=0.0, max_gen_len=4))

        torch.cuda.synchronize()
        started = time.perf_counter()
        pending = _arrival_schedule()
        arrivals: dict[str, float] = {}
        live = []

        while pending or engine.has_unfinished_requests():
            while pending and time.perf_counter() - started >= pending[0][0]:
                offset, prompt = pending.pop(0)
                request = engine.add_request(prompt, GREEDY)
                arrivals[request.request_id] = started + offset
                live.append(request)
            if engine.has_unfinished_requests():
                engine.step()
            elif pending:
                time.sleep(0.001)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        tokens = sum(len(r.output_token_ids) for r in live)
        latencies = [r.finish_time - arrivals[r.request_id] for r in live if r.finish_time]
        return elapsed, tokens, latencies
    finally:
        del engine
        _free()


@pytest.fixture(scope="module")
def measurements(model_dir):
    """Both strategies on the same arrival pattern, serially so neither is starved."""
    serial = _serve_serially(model_dir)
    continuous = _serve_continuously(model_dir)
    return serial, continuous


def test_staggered_arrivals_reach_a_higher_throughput(measurements):
    (serial_s, serial_tokens, _), (cb_s, cb_tokens, _) = measurements
    serial_tps = serial_tokens / serial_s
    cb_tps = cb_tokens / cb_s

    assert cb_tps > serial_tps * _MIN_THROUGHPUT_GAIN, (
        f"continuous {cb_tps:.0f} tok/s vs serial {serial_tps:.0f} tok/s "
        f"({cb_tps / serial_tps:.2f}x, wanted >{_MIN_THROUGHPUT_GAIN}x)"
    )


def test_staggered_arrivals_finish_sooner(measurements):
    (_, _, serial_latencies), (_, _, cb_latencies) = measurements
    serial_mean = sum(serial_latencies) / len(serial_latencies)
    cb_mean = sum(cb_latencies) / len(cb_latencies)

    assert cb_mean * _MIN_LATENCY_GAIN < serial_mean, (
        f"continuous mean latency {cb_mean * 1000:.0f} ms vs serial "
        f"{serial_mean * 1000:.0f} ms ({serial_mean / cb_mean:.2f}x, "
        f"wanted >{_MIN_LATENCY_GAIN}x)"
    )


def test_the_whole_arrival_burst_is_absorbed(measurements):
    """A batch that admits everyone finishes not far behind a single generation.

    Guards the case where the scheduler silently stops admitting -- throughput
    would still look fine while requests quietly serialised.
    """
    (serial_s, _, _), (cb_s, _, _) = measurements
    assert cb_s < serial_s / _MIN_THROUGHPUT_GAIN


def test_every_request_was_actually_served(measurements):
    (_, serial_tokens, serial_latencies), (_, cb_tokens, cb_latencies) = measurements

    assert len(cb_latencies) == _REQUESTS
    assert len(serial_latencies) == _REQUESTS
    # Same prompts and greedy sampling, so the delivered work should be
    # comparable; a large shortfall would mean the comparison is not like-for-like.
    assert cb_tokens > serial_tokens * 0.8
