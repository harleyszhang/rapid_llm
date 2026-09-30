"""Reported log-probabilities must be the ones HuggingFace computes.

The same prompt runs one-shot and chunked; every reported logprob is
compared against HF within tight absolute and mean drift budgets, so
sampling changes cannot hide behind "close enough".

Usage:
    pytest tests/golden/test_logprob_parity.py
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import pytest
import torch

from rapid_llm.engine.batch_planner import SchedulerConfig
from rapid_llm.engine.llm import LLM
from rapid_llm.engine.sampler import PositionLogprobs, SamplingParams
from rapid_llm.engine.scheduler import Scheduler

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]


_PROMPT = (
    "The history of the Roman Empire spans many centuries, and the city of Rome "
    "itself grew from a small settlement beside the Tiber into the capital of a "
    "state that ruled the whole Mediterranean world."
)

_MAX_SEQ_LEN = 512
_KV_BLOCKS = 4096

_CHUNK = 8
_TOP_K = 5
_MAX_GEN = 8
_MAX_DRIFT = 1.0
_MEAN_DRIFT = 0.15


def _params() -> SamplingParams:
    """Greedy, and asking for both halves of the feature in one pass."""
    return SamplingParams(
        temperature=0.0,
        max_gen_len=_MAX_GEN,
        repetition_penalty=1.0,
        stop_on_repeat=False,
        logprobs=_TOP_K,
        prompt_logprobs=_TOP_K,
    )


def _oneshot(model_dir: Path) -> dict[str, Any]:
    """Prompt and step records from the one-shot ``LLM`` path, engine freed after."""
    llm = LLM(
        model=str(model_dir),
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_BLOCKS,
        use_cuda_graph=False,
    )
    try:
        output = llm.generate([_PROMPT], _params())[0]
        return {
            "prompt_ids": llm.tokenizer.encode(_PROMPT, add_special_tokens=True),
            "prompt": list(output.prompt_logprobs or []),
            "steps": list(output.outputs[0].logprobs or []),
            "text": output.outputs[0].text,
        }
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _chunked(model_dir: Path) -> list[PositionLogprobs | None]:
    """The same prompt records from the continuous engine, prefill cut into chunks.

    ``max_chunk_size`` is turned down after the build, the way
    ``tests/engine/test_chunked_prefill`` does it: the scheduler owns the chunking and
    ``from_pretrained`` sizes its token budget for throughput, not for this test.
    """
    engine = Scheduler.from_pretrained(
        str(model_dir),
        max_seq_len=_MAX_SEQ_LEN,
        max_num_seqs=4,
        max_gpu_num_blocks=_KV_BLOCKS,
        use_cuda_graph=False,
    )
    config = SchedulerConfig(max_seq_len=_MAX_SEQ_LEN, max_num_seqs=4, max_chunk_size=_CHUNK)
    engine.config = config
    engine.planner.config = config
    try:
        return list(engine.generate([_PROMPT], _params())[0].prompt_logprobs or [])
    finally:
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def lite(model_dir: Path) -> dict[str, Any]:
    """Both collection paths, run one after another so only one engine is resident."""
    collected = _oneshot(model_dir)
    collected["chunked"] = _chunked(model_dir)
    return collected


@pytest.fixture(scope="module")
def reference(model_dir: Path, lite: dict[str, Any]) -> torch.Tensor:
    """``[positions, vocab]`` log-softmax from transformers over prompt + generation."""
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=torch.bfloat16, attn_implementation="eager"
        )
        .cuda()
        .eval()
    )
    try:
        ids = lite["prompt_ids"] + [record.token_id for record in lite["steps"]]
        with torch.no_grad():
            logits = model(torch.tensor([ids], device="cuda")).logits.float()
        return torch.log_softmax(logits, dim=-1)[0]
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _drift(row: torch.Tensor, token_id: int, reported: float, where: str) -> tuple[float, str]:
    """One comparison: how far a reported value is from the reference's own."""
    theirs = row[token_id].item()
    return abs(reported - theirs), f"{where} (token {token_id}: {reported:+.4f} vs {theirs:+.4f})"


def _agrees(drifts: list[tuple[float, str]]) -> None:
    """Assert a set of deviations is bf16 disagreement rather than a wrong number."""
    assert drifts, "no positions were compared"
    worst, where = max(drifts)
    mean = sum(drift for drift, _ in drifts) / len(drifts)
    assert worst <= _MAX_DRIFT, f"{where} is {worst:.4f} nats from the reference"
    assert mean <= _MEAN_DRIFT, (
        f"{len(drifts)} positions average {mean:.4f} nats from the reference, "
        f"worst {worst:.4f} at {where}: that is a bias, not rounding"
    )


def _scored(records: list[PositionLogprobs | None], reference: torch.Tensor, label: str):
    """Prompt records against the rows that predict them, position 0 excluded."""
    return [
        _drift(reference[i - 1], record.token_id, record.logprob, f"{label} position {i}")
        for i, record in enumerate(records)
        if record is not None
    ]


# --------------------------------------------------------------------------- #
# Contracts the numbers rest on
# --------------------------------------------------------------------------- #
def test_prompt_logprobs_score_the_prompts_own_tokens(lite):
    """Position ``i`` must report the prompt's token ``i``, and position 0 nothing.

    The contract before the numbers. Prompt scoring has no draw in it — the token is
    already known — so the only thing a record can be about is the prompt token at
    that index, and the first position has no predictor to be scored by.
    """
    prompt_ids = lite["prompt_ids"]
    for path, records in (("one-shot", lite["prompt"]), ("chunked", lite["chunked"])):
        assert len(records) == len(prompt_ids), f"{path}: one record per prompt token"
        assert records[0] is None, f"{path}: position 0 cannot be predicted"
        for i, record in enumerate(records[1:], start=1):
            assert record is not None, f"{path}: position {i} was not scored"
            assert record.token_id == prompt_ids[i], (
                f"{path}: position {i} scored token {record.token_id}, "
                f"but the prompt has {prompt_ids[i]} there"
            )


def test_the_run_being_compared_is_not_empty(lite):
    """Guards the comparisons below, which assert nothing over an empty record set."""
    assert lite["text"].strip(), "the checkpoint generated nothing to score"
    assert len(lite["steps"]) == _MAX_GEN
    assert len(lite["prompt_ids"]) > _CHUNK, "the prompt fits in one chunk; chunking untested"


# --------------------------------------------------------------------------- #
# The prompt: scored positions, not sampled ones
# --------------------------------------------------------------------------- #
def test_prompt_logprobs_match_transformers(lite, reference):
    """Every scored prompt position must agree with the reference's log-softmax.

    This is the whole feature in one comparison: the value reported for a prompt
    token is the reference's log-probability of that token given the prefix. A
    missing or partial normaliser shifts every position at once, and a temperature or
    penalty leaking into the prompt path scales them — neither survives an average
    held this close.
    """
    _agrees(_scored(lite["prompt"], reference, "prompt"))


def test_chunked_prefill_scores_the_prompt_the_same_way(lite, reference):
    """Cutting prefill into chunks must not move a prompt logprob."""
    _agrees(_scored(lite["chunked"], reference, "chunked prompt"))


# --------------------------------------------------------------------------- #
# The generation: sampled positions
# --------------------------------------------------------------------------- #
def test_sampled_logprobs_match_transformers(lite, reference):
    """Each decode step must report the reference's score for the token it drew."""
    base = len(lite["prompt_ids"])
    _agrees(
        [
            _drift(reference[base - 1 + step], record.token_id, record.logprob, f"step {step}")
            for step, record in enumerate(lite["steps"])
        ]
    )


def test_greedy_draws_a_token_the_reference_would_also_pick(lite, reference):
    """Greedy decoding must land on the reference's best token, or a tie of it."""
    base = len(lite["prompt_ids"])
    for step, record in enumerate(lite["steps"]):
        row = reference[base - 1 + step]
        best = int(row.argmax())
        assert row[record.token_id].item() >= row[best].item() - _MAX_DRIFT, (
            f"step {step} drew token {record.token_id} at {row[record.token_id].item():+.4f}, "
            f"while the reference would have taken {best} at {row[best].item():+.4f}"
        )


def test_every_reported_alternative_matches_transformers(lite, reference):
    """The top-k is not merely plausible: each entry is the reference's own number."""
    base = len(lite["prompt_ids"])
    rows = [(i - 1, record) for i, record in enumerate(lite["prompt"]) if record is not None]
    rows += [(base - 1 + step, record) for step, record in enumerate(lite["steps"])]
    drifts = []
    for row_index, record in rows:
        assert len(record.top_token_ids) == len(record.top_logprobs) == _TOP_K
        assert len(set(record.top_token_ids)) == _TOP_K, "an alternative was reported twice"
        assert list(record.top_logprobs) == sorted(record.top_logprobs, reverse=True), (
            f"row {row_index}: the top-k came back out of order"
        )
        drifts += [
            _drift(reference[row_index], token_id, value, f"row {row_index} alternative {rank}")
            for rank, (token_id, value) in enumerate(
                zip(record.top_token_ids, record.top_logprobs, strict=True)
            )
        ]
    _agrees(drifts)
