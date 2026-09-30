"""Real-checkpoint end-to-end gate for data-parallel attention.

The reference and DPA arms use the same two GPUs and the same MoE checkpoint:

* ``ep2`` is one TP=2 scheduler with experts split over both ranks;
* ``dpa2`` is two TP=1 schedulers with local attention/KV and one EP group over
  the DP axis.

The reference replays each round-robin bucket separately, matching the local
batch shape seen by each DPA replica. Three workloads cover balanced replicas,
an entirely idle replica, and unequal chunked-prefill pass counts. A timeout
turns a lockstep regression into a test failure rather than a hung test run.

Usage:
    RAPID_LLM_TEST_DPA_DIR=<moe-checkpoint> \
        pytest tests/distributed/test_dp_attention_engine.py -q
"""

from __future__ import annotations

import os
import queue as queue_module
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch.multiprocessing as mp

from rapid_llm import SamplingParams
from tests.conftest import REPO_ROOT, checkpoint_problem
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_DEFAULT_MODEL = "my_weight/DeepSeek-V2-Lite"
_PROBE_TIMEOUT_S = 900.0
_TIE_GAP = 0.5
_MIN_AGREEING_FRACTION = 2 / 3

_GREEDY = SamplingParams(
    temperature=0.0,
    max_gen_len=16,
    repetition_penalty=1.0,
    stop_on_repeat=False,
    logprobs=2,
)

_SHORT_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at",
    "The largest planet in our solar system is",
    "Python is a language that",
    "Machine learning is",
]
_LONG_PROMPT = (
    "Explain distributed inference carefully. " + "Attention and experts cooperate. " * 80
)
_SCENARIOS = {
    "balanced": _SHORT_PROMPTS,
    "idle-replica": [_SHORT_PROMPTS[0]],
    "uneven-prefill": [_SHORT_PROMPTS[1], _LONG_PROMPT],
}


def _record(output) -> dict[str, Any]:
    completion = output.outputs[0]
    records = completion.logprobs or ()
    return {
        "text": completion.text,
        "tokens": [record.token_id for record in records],
        "gaps": [
            float(record.top_logprobs[0] - record.top_logprobs[1])
            if len(record.top_logprobs) >= 2
            else None
            for record in records
        ],
    }


def _reference_scenario(engine, prompts: list[str]) -> list[dict[str, Any]]:
    """Replay the two round-robin buckets at the DPA replicas' local shapes."""
    records: list[dict[str, Any] | None] = [None] * len(prompts)
    for indices in (list(range(0, len(prompts), 2)), list(range(1, len(prompts), 2))):
        if not indices:
            continue
        outputs = engine.generate([prompts[index] for index in indices], _GREEDY)
        for index, output in zip(indices, outputs, strict=True):
            records[index] = _record(output)
    assert all(record is not None for record in records)
    return [record for record in records if record is not None]


def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    try:
        if spec["dpa"]:
            from rapid_llm.engine.data_parallel import DataParallelEngine

            engine = DataParallelEngine(
                model=spec["model"],
                data_parallel_size=2,
                tensor_parallel_size=1,
                enable_dp_attention=True,
                enable_expert_parallel=True,
                use_cuda_graph=False,
                max_seq_len=512,
                max_gpu_num_blocks=4096,
                max_num_seqs=8,
                max_num_batched_tokens=256,
                max_chunk_size=64,
            )
            report = {
                "executor": "DataParallelEngine",
                "world_size": engine.world_size,
                "records": {
                    name: [_record(output) for output in engine.generate(prompts, _GREEDY)]
                    for name, prompts in _SCENARIOS.items()
                },
            }
        else:
            from rapid_llm.engine.scheduler import Scheduler

            engine = Scheduler.from_pretrained(
                model=spec["model"],
                device="cuda:0",
                tensor_parallel_size=2,
                enable_expert_parallel=True,
                use_cuda_graph=False,
                max_seq_len=512,
                max_gpu_num_blocks=4096,
                max_num_seqs=8,
                max_num_batched_tokens=256,
                max_chunk_size=64,
            )
            report = {
                "executor": type(engine._executor).__name__,
                "world_size": 2,
                "records": {
                    name: _reference_scenario(engine, prompts)
                    for name, prompts in _SCENARIOS.items()
                },
            }
        engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _run_probe(model_dir: Path, *, dpa: bool) -> dict[str, Any]:
    context = mp.get_context("spawn")
    results = context.Queue()
    process = context.Process(
        target=_probe,
        args=({"model": str(model_dir), "dpa": dpa}, results),
        daemon=False,
    )
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_PROBE_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(
                f"{'dpa2' if dpa else 'ep2'} probe produced nothing in {_PROBE_TIMEOUT_S:.0f}s"
            )
        if status == "error":
            pytest.fail(f"{'dpa2' if dpa else 'ep2'} probe failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()
            process.join(timeout=30.0)


def _fork(base: list[int], other: list[int]) -> int:
    for step, (want, got) in enumerate(zip(base, other, strict=False)):
        if want != got:
            return step
    return min(len(base), len(other))


def _assert_tie_gap_parity(reference: dict, dpa: dict) -> None:
    for scenario, expected in reference["records"].items():
        actual = dpa["records"][scenario]
        assert len(actual) == len(expected)
        for index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            step = _fork(want["tokens"], got["tokens"])
            if step == len(want["tokens"]) == len(got["tokens"]):
                continue
            margin = want["gaps"][step] if step < len(want["gaps"]) else None
            assert margin is not None and margin <= _TIE_GAP, (
                f"{scenario} prompt {index} diverged at token {step}; the EP2 "
                f"reference margin was {margin}, above the {_TIE_GAP}-nat tie gap"
            )


def _agreement(reference: dict, dpa: dict) -> float:
    same = total = 0
    for scenario, expected in reference["records"].items():
        for want, got in zip(expected, dpa["records"][scenario], strict=True):
            same += sum(
                left == right for left, right in zip(want["tokens"], got["tokens"], strict=False)
            )
            total += len(want["tokens"])
    return same / total if total else 1.0


@pytest.fixture(scope="module")
def model_dir() -> Path:
    path = Path(os.environ.get("RAPID_LLM_TEST_DPA_DIR", _DEFAULT_MODEL))
    if not path.is_absolute():
        path = REPO_ROOT / path
    if problem := checkpoint_problem(path):
        pytest.xfail(f"UNVERIFIED: {problem}")
    return path


@pytest.fixture(scope="module")
def probes(model_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        "ep2": _run_probe(model_dir, dpa=False),
        "dpa2": _run_probe(model_dir, dpa=True),
    }


@needs_gpus(2)
def test_dpa_real_engine_uses_two_replica_ranks(probes):
    assert probes["ep2"]["executor"] == "MultiprocExecutor"
    assert probes["dpa2"]["executor"] == "DataParallelEngine"
    assert probes["dpa2"]["world_size"] == 2


@needs_gpus(2)
def test_dpa_balanced_idle_and_uneven_batches_match_ep(probes):
    _assert_tie_gap_parity(probes["ep2"], probes["dpa2"])


@needs_gpus(2)
def test_dpa_most_generated_tokens_match_ep_byte_for_byte(probes):
    agreement = _agreement(probes["ep2"], probes["dpa2"])
    assert agreement >= _MIN_AGREEING_FRACTION, (
        f"dpa2 agrees with ep2 on only {agreement:.0%} of generated tokens; "
        f"need at least {_MIN_AGREEING_FRACTION:.0%}"
    )
