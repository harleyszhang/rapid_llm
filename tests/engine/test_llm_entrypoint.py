"""Tests for the vLLM-style :class:`~rapid_llm.engine.llm.LLM` entry point.

Return shapes, single-string convenience, greedy determinism, finish
reasons, and the rejected promises (images on a text model, legacy
kwargs) — the public contract of the facade.

Usage:
    pytest tests/engine/test_llm_entrypoint.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rapid_llm import LLM, RequestOutput, SamplingParams, TextGenerator

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

# Small KV reservation so several generators can coexist within one GPU during
# the module's lifetime.
_KV_TOKENS = 2048


@pytest.fixture(scope="module")
def llm(model_dir: Path) -> LLM:
    return LLM(model=str(model_dir), max_seq_len=512, max_gpu_num_blocks=_KV_TOKENS)


def test_generate_batch_returns_request_outputs(llm: LLM):
    prompts = ["The capital of France is", "One plus one equals"]
    params = SamplingParams(temperature=0.0, max_gen_len=8)

    outputs = llm.generate(prompts, params)

    assert len(outputs) == len(prompts)
    for out, prompt in zip(outputs, prompts, strict=True):
        assert isinstance(out, RequestOutput)
        assert out.prompt == prompt
        assert out.text  # non-empty completion
        assert out.outputs[0].finish_reason in ("eos", "length", "repeat")


def test_generate_accepts_a_single_string(llm: LLM):
    """A bare string must not be iterated into one request per character."""
    prompt = "The capital of France is"
    outputs = llm.generate(prompt, SamplingParams(temperature=0.0, max_gen_len=4))
    assert len(outputs) == 1
    assert outputs[0].prompt == prompt


def test_generate_is_deterministic_greedy(llm: LLM):
    params = SamplingParams(temperature=0.0, max_gen_len=12)
    a = llm.generate(["The capital of France is"], params)[0].text
    b = llm.generate(["The capital of France is"], params)[0].text
    assert a == b


def test_finish_reason_is_length_when_capped(llm: LLM):
    """A 4-token cap on a continuation prompt stops on length, not EOS."""
    out = llm.generate(
        ["Count upward: one two three"], SamplingParams(temperature=0.0, max_gen_len=4)
    )[0]
    assert out.outputs[0].finish_reason == "length"


def test_images_rejected_on_text_model(llm: LLM):
    with pytest.raises(ValueError, match="text-only"):
        llm.generate(["hi"], SamplingParams(), images=[object()])  # type: ignore[list-item]


def test_parallel_size_contract(model_dir: Path):
    """Neither TP nor DP fits inside one ``LLM``, and each error names the class that does.

    ``LLM`` is a single replica driven by a lockstep batch loop, so
    ``data_parallel_size>1`` needs
    :class:`~rapid_llm.engine.data_parallel.DataParallelController` and
    ``tensor_parallel_size>1`` needs
    :class:`~rapid_llm.engine.scheduler.Scheduler` — the only
    path whose executor broadcasts each step's plan to follower ranks.

    The TP half is a regression guard. The argument used to be accepted and then
    ignored: no group was started, the run went single-GPU, and the caller's TP
    measurement was really a TP=1 measurement wearing its label.
    """
    with pytest.raises(ValueError, match="Scheduler"):
        LLM(model=str(model_dir), tensor_parallel_size=2, max_seq_len=512)

    with pytest.raises(ValueError, match="DataParallelController"):
        LLM(model=str(model_dir), data_parallel_size=2)


def test_legacy_text_generator_delegates(model_dir: Path):
    """TextGenerator keeps its ``list[str]`` shape on top of the new LLM."""
    gen = TextGenerator(
        checkpoints_dir=str(model_dir),
        max_seq_len=512,
        max_gpu_num_blocks=_KV_TOKENS,
        device="cuda",
    )
    out = gen.generate(["The capital of France is"], SamplingParams(temperature=0.0, max_gen_len=8))
    assert isinstance(out, list)
    assert isinstance(out[0], str)
    assert out[0]
    del gen  # release its KV reservation for later tests
    torch.cuda.empty_cache()
