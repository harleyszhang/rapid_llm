"""End-to-end tensor parallelism: what two ranks answer, one rank must too.

TP-2 groupings generate with real checkpoints and are compared against
single-rank answers; shared-prefix agreement and tie-gap-tolerant greedy
checks keep the comparison deterministic.

Usage:
    pytest tests/distributed/test_tp_engine.py
"""

from __future__ import annotations

import queue as queue_module
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch.multiprocessing as mp

from rapid_llm import SamplingParams
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

#: Both probes plus a follower hold their own weights and KV cache, so this is
#: sized to leave the machine room rather than to stress the scheduler.
_KV_TOKENS = 4096
_MAX_SEQ_LEN = 512
_MAX_NUM_SEQS = 8

#: Loading a checkpoint, profiling a cache and rendezvousing a group. Generous on
#: purpose: it exists to turn a wedged rank into a failure instead of a hang.
_PROBE_TIMEOUT_S = 600.0

#: Greedy, no repetition penalty, no early exit: the only thing that may move a
#: token between the two widths is the arithmetic itself. ``logprobs=2`` is not
#: under test here -- it is the instrument, reporting the runner-up at every step
#: so a tie can be measured instead of inferred.
_GREEDY = SamplingParams(
    temperature=0.0,
    max_gen_len=24,
    repetition_penalty=1.0,
    stop_on_repeat=False,
    logprobs=2,
)

#: Prompt layouts, not a golden baseline (``tests/golden`` owns that). What varies
#: here is the *shape* of the batch a plan describes, because that is what the
#: driver broadcasts and what every rank has to derive identically: one row, rows
#: of unequal length in one prefill, and more rows than one prefill group admits.
_CASES: list[tuple[str, list[str]]] = [
    ("single", ["The capital of France is"]),
    ("mixed", ["Hi", "The history of the Roman Empire spans many centuries, and"]),
    (
        "batch6",
        [
            "One plus one equals",
            "The sun rises in the",
            "Water boils at",
            "Machine learning is",
            "The largest planet is",
            "Python is a language that",
        ],
    ),
]


#: Log-probability margin below which the runner-up is close enough that a
#: differently ordered sum can take the step either way. Half a nat is a few bf16
#: ULPs at this checkpoint's logit scale (logits reach ~16, where one ULP is
#: 0.125): measured here, every step the two widths disagreed about had a margin of
#: 0.125 or less, while a step decided by the weights leads by whole nats. It is an
#: upper bound on the noise, not a fitted constant.
_TIE_GAP = 0.5

#: Characters two answers must share before a divergence counts as arithmetic
#: noise rather than a broken shard. Used where no per-step record is available
#: (the online probe streams text only); a wrong offset or an unmasked row
#: corrupts the first token, so any real prefix at all is evidence.
_MIN_SHARED_PREFIX = 16

#: Fraction of generated tokens the two widths must agree on outright. Every
#: divergence is licensed by a small margin, so without this a checkpoint that had
#: become noise would satisfy the parity test one coin flip at a time.
_MIN_AGREEING_FRACTION = 2 / 3

#: Which prompts the online (async) probe serves concurrently, as ``(case, index)``
#: so its answers can be held against the offline ones.
_ONLINE = [("single", 0), ("batch6", 0)]

#: The two batch shapes every prompt is answered in.
_GROUPINGS = ("batched", "alone")


def _prompt_at(name: str, index: int) -> str:
    return next(prompts for case, prompts in _CASES if case == name)[index]


def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Build one engine of the requested width, answer every case, report the facts.

    Runs in a spawned process, so it takes only picklable arguments and imports
    torch itself. A build failure is reported as a traceback rather than left to
    time out, because a rank that dies during rendezvous takes the group with it.
    """
    try:
        from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_TOKENS,
            max_num_seqs=_MAX_NUM_SEQS,
            # Eager on both sides. Not because a two-rank capture is unsafe --
            # tests/distributed/test_tp_cuda_graph.py asserts that it is not -- but
            # because this file is measuring one variable, the shards, and holding
            # a graph engine against an eager one would fold a second one into
            # every difference.
            use_cuda_graph=False,
            tensor_parallel_size=spec["tp_size"],
        )
        try:
            model = engine.engine.model_runner.model
            report: dict[str, Any] = {
                # Every prompt twice: in its batch, and by itself, so a bug that
                # only bites a multi-row prefill has both shapes to survive.
                "batched": {name: _answer(engine, prompts) for name, prompts in _CASES},
                "alone": {
                    name: [_answer(engine, [prompt])[0] for prompt in prompts]
                    for name, prompts in _CASES
                },
                "executor": type(engine._executor).__name__,
                # Read before shutdown, which reaps them.
                "children": len(mp.active_children()),
                "embed_rows": model.embed_tokens.local_vocab_size,
                "embed_bytes": model.embed_tokens.weight.numel()
                * model.embed_tokens.weight.element_size(),
                "head_rows": model.lm_head.local_vocab_size,
                "vocab_size": model.embed_tokens.vocab_size,
                "tied": model.lm_head.weight is model.embed_tokens.weight,
            }
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _answer(engine, prompts: list[str]) -> list[dict[str, Any]]:
    """Greedy completions for one batch, in submission order, with their margins."""
    return [_record(output.outputs[0]) for output in engine.generate(prompts, _GREEDY)]


def _record(completion) -> dict[str, Any]:
    """One completion as picklable primitives: text, its tokens, and their margins.

    The margins are the point — they say which of these tokens the arithmetic was
    entitled to change its mind about. Nothing but ints, floats and str, because
    this crosses a process boundary on the results queue.
    """
    return {
        "text": completion.text,
        "tokens": [record.token_id for record in completion.logprobs or ()],
        "gaps": [_margin(record) for record in completion.logprobs or ()],
    }


def _margin(record) -> float:
    """How far the sampled token led the runner-up, in log-probability.

    Zero means the two are indistinguishable at this precision, which is exactly
    when a differently ordered sum may pick the other one.
    """
    top = record.top_logprobs
    return float(top[0] - top[1]) if len(top) >= 2 else float("inf")


def _async_probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Serve :data:`_ONLINE` concurrently from a two-rank async engine.

    Online serving puts tensor parallelism somewhere the offline path never does:
    :class:`~rapid_llm.engine.async_engine.AsyncLLMEngine` steps the engine on a
    background thread, so every plan broadcast and every NCCL collective is issued
    off the main thread while coroutines register requests concurrently. That the
    ranks stay in step there is a separate fact from the arithmetic, and it is what
    a deployment actually exercises.
    """
    try:
        import asyncio

        from rapid_llm.engine.async_engine import AsyncLLMEngine

        async def serve() -> list[str]:
            engine = AsyncLLMEngine.from_pretrained(
                spec["model"],
                device="cuda:0",
                max_seq_len=_MAX_SEQ_LEN,
                max_gpu_num_blocks=_KV_TOKENS,
                max_num_seqs=_MAX_NUM_SEQS,
                use_cuda_graph=False,
                tensor_parallel_size=2,
            )
            async with engine:
                streams = [engine.generate_text(prompt, _GREEDY) for prompt in spec["prompts"]]
                return [output.text for output in await asyncio.gather(*streams)]

        answers = asyncio.run(serve())
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", {"answers": answers}))


def _run_probe(
    model_dir: Path,
    tp_size: int,
    target=_probe,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one probe to completion, surfacing its traceback as this test's failure."""
    context = mp.get_context("spawn")
    results = context.Queue()
    spec = {"model": str(model_dir), "tp_size": tp_size, **(extra or {})}
    # Not a daemon: with tp_size > 1 the probe spawns the follower ranks, and a
    # daemonic process is not allowed children.
    process = context.Process(target=target, args=(spec, results), daemon=False)
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_PROBE_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(f"tp={tp_size} probe produced nothing in {_PROBE_TIMEOUT_S:.0f}s")
        if status == "error":
            pytest.fail(f"tp={tp_size} probe failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()


@pytest.fixture(scope="module")
def probes(model_dir: Path) -> dict[int, dict[str, Any]]:
    """One report per width. Module-scoped: two checkpoint loads, many assertions."""
    return {tp_size: _run_probe(model_dir, tp_size) for tp_size in (1, 2)}


def _first_difference(want: str, got: str) -> str:
    """Where two completions diverge, with enough either side to recognise it."""
    for position, (a, b) in enumerate(zip(want, got, strict=False)):
        if a != b:
            return (
                f"diverges at character {position}\n"
                f"  tp=1: ...{want[max(0, position - 30) : position + 30]!r}\n"
                f"  tp=2: ...{got[max(0, position - 30) : position + 30]!r}"
            )
    return f"one is a prefix of the other\n  tp=1: {want!r}\n  tp=2: {got!r}"


def _entries() -> list[tuple[str, int]]:
    """Every ``(case, index)`` pair the probes answer, the unit of comparison."""
    return [(name, index) for name, prompts in _CASES for index in range(len(prompts))]


def _shared_prefix(one: str, two: str) -> int:
    for position, (a, b) in enumerate(zip(one, two, strict=False)):
        if a != b:
            return position
    return min(len(one), len(two))


def _text(probes, width: int, grouping: str, name: str, index: int) -> str:
    return probes[width][grouping][name][index]["text"]


def _first_token_difference(one: list[int], two: list[int]) -> int:
    """Step at which two token sequences part, or the length of the shorter one."""
    for step, (a, b) in enumerate(zip(one, two, strict=False)):
        if a != b:
            return step
    return min(len(one), len(two))


def _fork(probes, grouping: str, name: str, index: int) -> int:
    """Step at which the two widths first said different things about one prompt."""
    return _first_token_difference(
        probes[1][grouping][name][index]["tokens"], probes[2][grouping][name][index]["tokens"]
    )


def _margin_at(probes, grouping: str, name: str, index: int, step: int) -> float:
    """How decisive the single-GPU probe was at one step of one answer.

    A step past the end of the record has no margin to appeal to, and is reported
    as infinitely decisive: two widths that agree on every token but stop at
    different lengths have nothing about the arithmetic to blame.
    """
    gaps = probes[1][grouping][name][index]["gaps"]
    return gaps[step] if step < len(gaps) else float("inf")


def _agreeing(probes) -> set[tuple[str, int]]:
    """Entries the two widths answered byte for byte alike, in both batch shapes."""
    return {
        (name, index)
        for name, index in _entries()
        if all(
            _text(probes, 1, grouping, name, index) == _text(probes, 2, grouping, name, index)
            for grouping in _GROUPINGS
        )
    }


# --------------------------------------------------------------------------- #
# Numerics
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_two_ranks_only_diverge_where_the_arithmetic_had_a_choice(probes):
    """Sharding may only change a token the single GPU could not decide either.

    Sharding is meant to be an arithmetic identity: a row-parallel GEMM plus an
    all-reduce computes the same sum as the whole GEMM, and the sampler's
    two-scalar exchange reconstructs the same log-softmax as the full vocabulary.
    The failures worth fearing produce *plausible* numbers — an off-by-one shard
    offset, a mask that lets another rank's rows into the sum — and they show up
    immediately, on a token the weights had already decided by whole nats.

    So instead of demanding byte equality and excusing the prompts it cannot get,
    this asks each divergence to account for itself: wherever the two widths first
    part company, the single-GPU margin at that very step must be within
    :data:`_TIE_GAP`. A reordered sum can flip a step decided by a few ULPs and
    nothing else, and the check is made in both batch shapes, so a bug that only
    bites a multi-row prefill has nowhere to hide.
    """
    for name, index in _entries():
        for grouping in _GROUPINGS:
            one = probes[1][grouping][name][index]
            two = probes[2][grouping][name][index]
            fork = _fork(probes, grouping, name, index)
            if fork == len(one["tokens"]) == len(two["tokens"]):
                continue
            margin = _margin_at(probes, grouping, name, index, fork)
            assert margin <= _TIE_GAP, (
                f"{grouping} case {name!r} prompt {index} diverges at token {fork}, which one "
                f"GPU decided by {margin:.3f} — too much to be the order of a sum:\n"
                f"{_first_difference(one['text'], two['text'])}"
            )


@needs_gpus(2)
def test_most_of_what_two_ranks_say_is_byte_identical(probes):
    """Guards the test above: a coin flip has to be the exception, not the rule.

    Every divergence there is licensed by a small margin, and a checkpoint that had
    degenerated into noise would have a small margin at every step — it would pass
    while agreeing about almost nothing. So how much the widths agree on outright is
    asserted too, counted in tokens rather than prompts, because one flip fourteen
    tokens in would otherwise write off the thirteen that matched.
    """
    agreed = total = 0
    forks: list[str] = []
    for name, index in _entries():
        for grouping in _GROUPINGS:
            steps = len(probes[1][grouping][name][index]["tokens"])
            fork = _fork(probes, grouping, name, index)
            agreed += fork
            total += steps
            if fork < steps:
                margin = _margin_at(probes, grouping, name, index, fork)
                forks.append(f"{name}/{index} {grouping} at token {fork} (margin {margin:.3f})")

    print(f"\nidentical tokens: {agreed}/{total}")
    for fork in forks:
        print(f"  diverged: {fork}")
    assert agreed >= _MIN_AGREEING_FRACTION * total, (
        f"the two widths agreed on only {agreed}/{total} tokens: {forks}"
    )


@needs_gpus(2)
def test_neither_width_answers_nothing(probes):
    """Guards the comparisons above: two empty answers are also byte-identical.

    The margins are guarded with them, since they are what licenses a divergence —
    a record that reported no margin at all would read as a step nobody can be held
    responsible for.
    """
    for width in (1, 2):
        for grouping in _GROUPINGS:
            for name, prompts in _CASES:
                answers = probes[width][grouping][name]
                assert len(answers) == len(prompts)
                where = f"tp={width}, {grouping} {name}"
                assert all(answer["text"].strip() for answer in answers), where
                assert all(answer["gaps"] for answer in answers), f"{where}: no margins reported"
                assert all(len(answer["gaps"]) == len(answer["tokens"]) for answer in answers), (
                    f"{where}: a margin per token is what licenses a divergence"
                )


# --------------------------------------------------------------------------- #
# Online serving
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_online_serving_over_two_ranks_answers_what_offline_does(probes, model_dir):
    """An async, two-rank engine must serve concurrent requests, and serve them right.

    Two things are asserted at once because they fail together: that the ranks stay
    in step when the plans are broadcast from a worker thread rather than the main
    one (a desynchronised group hangs, which the probe timeout reports), and that
    what comes out is what the offline path produced for the same prompt.

    Byte equality is demanded only for prompts the two offline widths answered
    identically; the others are held to a shared prefix, since an async arrival
    order groups requests into batches the offline path never formed, and the
    streamed output carries text rather than the per-step margins to appeal to.
    """
    answers = _run_probe(
        model_dir,
        tp_size=2,
        target=_async_probe,
        extra={"prompts": [_prompt_at(*entry) for entry in _ONLINE]},
    )["answers"]
    assert len(answers) == len(_ONLINE)

    agreeing = _agreeing(probes)
    for (name, index), online in zip(_ONLINE, answers, strict=True):
        offline = _text(probes, 2, "alone", name, index)
        assert online.strip(), f"case {name!r} prompt {index} came back empty"
        if (name, index) in agreeing:
            assert online == offline, (
                f"case {name!r} prompt {index}: {_first_difference(offline, online)}"
            )
        else:
            assert _shared_prefix(offline, online) >= _MIN_SHARED_PREFIX


# --------------------------------------------------------------------------- #
# What sharding is for: the weights actually get smaller
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_two_ranks_hold_half_the_vocabulary_each(probes):
    """The embedding and the head shrink by exactly the TP width.

    This is the *point* of vocabulary parallelism, and the one property token
    parity cannot see: an implementation that replicated the full table on every
    rank and masked at the end would answer identically while saving nothing.
    """
    vocab = probes[1]["vocab_size"]
    assert vocab % 2 == 0, "an odd vocabulary would make every shard assertion vacuous"
    assert probes[1]["embed_rows"] == vocab
    assert probes[2]["embed_rows"] == vocab // 2
    assert probes[2]["head_rows"] == vocab // 2
    assert probes[2]["embed_bytes"] * 2 == probes[1]["embed_bytes"]


@needs_gpus(2)
def test_tying_survives_sharding(probes):
    """A tied head must stay the *same tensor* as the embedding on every rank.

    Both ends own the same vocabulary slice, so tying stays one allocation rather
    than becoming a special case — and if it silently untied, the head would cost
    a second copy of the largest tensor in a small model.
    """
    assert probes[2]["tied"] == probes[1]["tied"]


# --------------------------------------------------------------------------- #
# Process topology: the cost of parallelism is only paid when asked for
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_one_gpu_stays_in_one_process(probes):
    """The single-GPU default must remain debuggable: no processes, no broadcasts.

    A breakpoint in the engine loop is only a breakpoint in the kernel while the
    forward runs in the calling process, which is why ``UniProcExecutor`` exists
    at all rather than TP=1 being a one-rank group.
    """
    assert probes[1]["executor"] == "UniProcExecutor"
    assert probes[1]["children"] == 0


@needs_gpus(2)
def test_two_gpus_cost_exactly_one_extra_process(probes):
    """TP=2 is two ranks, not a driver plus two workers.

    The driver is rank 0 *and* a worker, so the process count is the rank count.
    """
    assert probes[2]["executor"] == "MultiprocExecutor"
    assert probes[2]["children"] == 1


@needs_gpus(2)
def test_shutdown_returns_the_process_to_a_world_of_one(model_dir: Path):
    """A shut-down TP engine must not re-shard the next one this process builds.

    The executor owns the rank-0 half of the group exactly as it owns the
    follower processes, so both halves have to go in ``shutdown``. Left
    standing, the stale group makes the next engine in this process — the
    benchmark that measures TP=2 then TP=1, the golden test that loads
    transformers after rapid_llm — read a TP size nobody asked for. The
    probes above cannot catch this: each runs in a spawned process whose
    module state dies with it; only this test holds the engine in the test
    process itself.
    """
    from rapid_llm.distributed import parallel_state as ps
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine.from_pretrained(
        model=str(model_dir),
        device="cuda:0",
        max_seq_len=_MAX_SEQ_LEN,
        max_gpu_num_blocks=_KV_TOKENS,
        max_num_seqs=_MAX_NUM_SEQS,
        use_cuda_graph=False,
        tensor_parallel_size=2,
    )
    try:
        assert ps.get_tensor_model_parallel_world_size() == 2
        list(engine.generate([_prompt_at("single", 0)], _GREEDY))
    finally:
        engine.shutdown()
    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_world_size() == 1


@needs_gpus(2)
def test_a_failed_build_returns_the_process_to_a_world_of_one(model_dir: Path, monkeypatch):
    """An engine build that fails after the group rendezvoused must clean up too.

    ``from_pretrained`` launches the followers and joins the group *before* it
    loads the rank-0 shard, so a build that fails there — an OOM loading the
    weights is the common case — used to leave this process holding rank 0 of
    a group whose other half is gone. The stale state then poisoned everything
    later in the process: the next engine read a TP size nobody asked for,
    and a plain CPU model test all-reduced against a group that no longer
    existed. The failure is simulated by patching the engine constructor, so
    the follower side builds a one-layer stack and parks on its first plan.
    """
    from rapid_llm.distributed import parallel_state as ps
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated shard-load failure")

    monkeypatch.setattr("rapid_llm.engine.llm_engine.LLMEngine", _boom)
    with pytest.raises(RuntimeError, match="simulated shard-load failure"):
        ContinuousBatchingEngine.from_pretrained(
            model=str(model_dir),
            device="cuda:0",
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_TOKENS,
            max_num_seqs=_MAX_NUM_SEQS,
            use_cuda_graph=False,
            tensor_parallel_size=2,
            hf_overrides={"num_hidden_layers": 1},
        )
    assert ps.get_tensor_model_parallel_world_size() == 1
    assert ps.get_world_size() == 1
