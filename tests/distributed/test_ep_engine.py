"""Expert parallelism end to end: what two EP ranks answer, TP must too.

This is the integration gate for ``--enable-expert-parallel`` and its
two-batch overlap. It runs the *real* continuous engine on DeepSeek-V2-Lite
across two ranks in three configurations and holds them against each other:

* ``tp2``     — tensor parallelism only, the already-gated baseline
                (:mod:`tests.distributed.test_tp_engine` and
                :mod:`tests.golden.test_deepseek_v2_tp2` own its correctness);
* ``ep2``     — ``enable_expert_parallel=True``, experts split whole across the
                ranks, the ``SparseMoeBlock.forward`` dispatch/combine path;
* ``ep2_tbo`` — the same, with ``RAPID_LLM_TBO`` on, so decode runs the
                :mod:`~rapid_llm.batch_overlap.operations_strategy` op stream
                (dispatch_a / shared / dispatch_b / experts / combine_a /
                combine_b) with the two halves' all-to-all exchanges overlapped;
* ``ep2_graph`` — the same EP forward with decode CUDA graphs on, so the
                dispatch/combine all-to-all exchanges are captured into the
                graph and replayed in lockstep. This is the gate for EP keeping
                its graphs (the a2a rides the same comm-stream discipline TBO's
                deferred all-reduce already captures): a replay that dropped or
                cross-paired an exchange would corrupt tokens decided by whole
                nats, not merely near-ties.

The comparison is *tie-gap tolerant*, not token-exact, and that is a measured
property of this checkpoint rather than a concession — the same discipline
:mod:`tests.distributed.test_tp_engine` applies to TP widths. EP reorders the
MoE reduction (a token's ``k`` expert results are combined by a sender-side
scatter-add instead of a TP all-reduce of intermediate-sliced partials), so it
moves outputs by a few bf16 ULPs. DeepSeek-V2-Lite's router runs some tokens
through near-ties (the golden gate documents ~14% greedy disagreement against a
*faithful* reference), and a ULP is enough to flip those, after which the
autoregressive tail diverges. So each configuration must agree with ``tp2`` on
almost every token outright, and wherever it first parts company, ``tp2``'s own
margin at that step must be within :data:`_TIE_GAP` — a reordered sum may only
change a token the arithmetic could not decide. A real EP bug (a wrong expert
offset, a dropped pad row, a cross-paired all-to-all) forks on a token the
weights decided by whole nats and fails immediately.

Usage:
    pytest tests/distributed/test_ep_engine.py

Needs the DeepSeek-V2-Lite checkpoint under ``my_weight/`` (override with
``RAPID_LLM_TEST_DSV2_DIR``) and two CUDA devices.
"""

from __future__ import annotations

import os
import queue as queue_module
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.multiprocessing as mp

from rapid_llm import SamplingParams
from tests.conftest import REPO_ROOT, checkpoint_problem
from tests.distributed.tp_harness import needs_gpus

# No ``weights`` mark: that mark binds a test to the shared ``model_dir`` fixture
# (the small dense default), and this file gates on its own MoE checkpoint via
# ``dsv2_dir`` below — the same no-silent-skip policy the golden DSV2 gate uses.
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: DeepSeek-V2-Lite (MLA + 64 routed experts); EP only bites a MoE, so the small
#: dense default the rest of the suite runs against would exercise nothing here.
_DSV2 = "my_weight/DeepSeek-V2-Lite"

_KV_TOKENS = 4096
_MAX_SEQ_LEN = 512
_MAX_NUM_SEQS = 8

#: Loading a checkpoint, profiling a cache and rendezvousing a group, three times
#: over. Generous on purpose: it turns a wedged rank into a failure, not a hang.
_PROBE_TIMEOUT_S = 600.0

#: Greedy, no repetition penalty, no early exit — the only thing that may move a
#: token between two configurations is the arithmetic. ``logprobs=2`` is the
#: instrument, not the subject: it reports the runner-up so a fork's margin can
#: be measured rather than inferred.
_GREEDY = SamplingParams(
    temperature=0.0,
    max_gen_len=24,
    repetition_penalty=1.0,
    stop_on_repeat=False,
    logprobs=2,
)

#: Enough prompts that a decode step carries more rows than
#: ``RAPID_LLM_TBO_MIN_ROWS`` (set to 2 below), so the ``ep2_tbo`` probe really
#: runs the overlapped op stream rather than silently falling back to eager.
_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at",
    "The largest planet in our solar system is",
    "Python is a language that",
    "Machine learning is",
]

#: Log-probability margin below which a runner-up is close enough that a
#: reordered sum may take the step either way. DeepSeek-V2-Lite's logits reach
#: ~16, where one bf16 ULP is 0.125; the golden gate pins its own DSV2 tie gap
#: at 0.1 and :mod:`tests.distributed.test_tp_engine` uses 0.5 for TP widths.
#: Half a nat is a few ULPs — an upper bound on the noise, not a fitted value:
#: a step decided by the weights leads by whole nats.
_TIE_GAP = 0.5

#: Fraction of generated tokens a configuration must match ``tp2`` on outright.
#: Every fork is licensed by a small margin, so without this floor a checkpoint
#: that had decayed into noise would pass one coin flip at a time.
_MIN_AGREEING_FRACTION = 2 / 3


def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Build one two-rank engine of the requested flavour, answer, report facts.

    Runs in a spawned process, so it takes only picklable arguments. The TBO env
    is set before the engine is built — :func:`~rapid_llm.batch_overlap.
    two_batch_overlap.tbo_policy` reads and caches it on first use in the worker,
    and the follower ranks this process spawns inherit it. A build failure is
    reported as a traceback rather than left to time out: a rank that dies during
    rendezvous takes the group with it.
    """
    try:
        if spec["tbo"]:
            os.environ["RAPID_LLM_TBO"] = "1"
            os.environ["RAPID_LLM_TBO_MIN_ROWS"] = "2"
        from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_TOKENS,
            max_num_seqs=_MAX_NUM_SEQS,
            # The graph arm captures the EP all-to-all into the decode graph;
            # every other arm stays eager so a difference is never confounded
            # by a second variable. The small KV pool leaves the headroom the
            # (large) EP a2a buffers need at capture time.
            use_cuda_graph=spec["graph"],
            tensor_parallel_size=2,
            enable_expert_parallel=spec["ep"],
        )
        try:
            records = [_record(output.outputs[0]) for output in engine.generate(_PROMPTS, _GREEDY)]
            report = {
                "records": records,
                "executor": type(engine._executor).__name__,
                "children": len(mp.active_children()),
            }
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _record(completion) -> dict[str, Any]:
    """One completion as picklable primitives: text, its tokens, their margins."""
    return {
        "text": completion.text,
        "tokens": [r.token_id for r in completion.logprobs or ()],
        "gaps": [_margin(r) for r in completion.logprobs or ()],
    }


def _margin(record) -> float:
    """How far the sampled token led the runner-up, in log-probability.

    Zero means the two are indistinguishable at this precision — exactly when a
    differently ordered sum may pick the other one.
    """
    top = record.top_logprobs
    return float(top[0] - top[1]) if len(top) >= 2 else float("inf")


def _run_probe(model_dir: Path, *, ep: bool, tbo: bool, graph: bool = False) -> dict[str, Any]:
    """Run one probe to completion, surfacing its traceback as this test's failure."""
    context = mp.get_context("spawn")
    results = context.Queue()
    spec = {"model": str(model_dir), "ep": ep, "tbo": tbo, "graph": graph}
    # Not a daemon: with tp_size > 1 the probe spawns the follower ranks, and a
    # daemonic process is not allowed children.
    process = context.Process(target=_probe, args=(spec, results), daemon=False)
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_PROBE_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(
                f"EP probe (ep={ep}, tbo={tbo}) produced nothing in {_PROBE_TIMEOUT_S:.0f}s"
            )
        if status == "error":
            pytest.fail(f"EP probe (ep={ep}, tbo={tbo}) failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()


def _ep_problem(path: Path) -> str | None:
    """Why this machine cannot run the gate, or ``None`` if it can."""
    problem = checkpoint_problem(path)
    if problem:
        return problem
    if torch.cuda.device_count() < 2:
        return "EP=2 needs two CUDA devices"
    return None


@pytest.fixture(scope="module")
def dsv2_dir() -> Path:
    """The checkpoint under test, under the golden gate's no-silent-skip policy."""
    path = Path(os.environ.get("RAPID_LLM_TEST_DSV2_DIR", _DSV2))
    if not path.is_absolute():
        path = REPO_ROOT / path
    problem = _ep_problem(path)
    if problem:
        pytest.xfail(f"UNVERIFIED: {problem}")
    return path


@pytest.fixture(scope="module")
def probes(dsv2_dir: Path) -> dict[str, dict[str, Any]]:
    """One report per configuration. Module-scoped: three loads, many assertions."""
    return {
        "tp2": _run_probe(dsv2_dir, ep=False, tbo=False),
        "ep2": _run_probe(dsv2_dir, ep=True, tbo=False),
        "ep2_tbo": _run_probe(dsv2_dir, ep=True, tbo=True),
        "ep2_graph": _run_probe(dsv2_dir, ep=True, tbo=False, graph=True),
    }


# --------------------------------------------------------------------------- #
# comparison helpers
# --------------------------------------------------------------------------- #
def _fork(base: list[int], other: list[int]) -> int:
    """Step at which two token sequences first part, or the shorter length."""
    for step, (a, b) in enumerate(zip(base, other, strict=False)):
        if a != b:
            return step
    return min(len(base), len(other))


def _first_difference(want: str, got: str) -> str:
    """Where two completions diverge, with enough either side to recognise it."""
    for position, (a, b) in enumerate(zip(want, got, strict=False)):
        if a != b:
            return (
                f"diverges at character {position}\n"
                f"  tp2: ...{want[max(0, position - 30) : position + 30]!r}\n"
                f"  ep : ...{got[max(0, position - 30) : position + 30]!r}"
            )
    return f"one is a prefix of the other\n  tp2: {want!r}\n  ep: {got!r}"


def _assert_tie_gap_parity(probes: dict[str, Any], name: str) -> None:
    """Every fork between ``name`` and the ``tp2`` baseline sits on a near-tie."""
    base = probes["tp2"]["records"]
    other = probes[name]["records"]
    for index, (want, got) in enumerate(zip(base, other, strict=True)):
        step = _fork(want["tokens"], got["tokens"])
        if step == len(want["tokens"]) == len(got["tokens"]):
            continue
        margin = want["gaps"][step] if step < len(want["gaps"]) else float("inf")
        assert margin <= _TIE_GAP, (
            f"{name} prompt {index} diverges from tp2 at token {step}, which tp2 "
            f"decided by {margin:.3f} — too much to be the order of a sum:\n"
            f"{_first_difference(want['text'], got['text'])}"
        )


def _agreeing_fraction(probes: dict[str, Any], name: str) -> float:
    """Fraction of generated tokens ``name`` matches ``tp2`` on, counted in tokens."""
    base = probes["tp2"]["records"]
    other = probes[name]["records"]
    total = sum(len(want["tokens"]) for want in base)
    same = sum(
        sum(1 for a, b in zip(want["tokens"], got["tokens"], strict=False) if a == b)
        for want, got in zip(base, other, strict=True)
    )
    return same / total if total else 1.0


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_both_ep_probes_really_ran_two_ranks(probes):
    """Guard the rest: the EP probes drove a real two-rank executor, not a fallback.

    A silent single-rank fallback would make every parity check below vacuous —
    it would agree with ``tp2`` by being the same arithmetic. Pin that each probe
    spawned its follower and used the multi-rank executor.
    """
    for name in ("tp2", "ep2", "ep2_tbo", "ep2_graph"):
        assert probes[name]["children"] >= 1, f"{name} spawned no follower rank"
        assert probes[name]["executor"] == "MultiprocExecutor", (
            f"{name} used {probes[name]['executor']}, not the multi-rank executor"
        )


@needs_gpus(2)
def test_ep_forward_only_diverges_where_the_arithmetic_had_a_choice(probes):
    """The EP dispatch/combine forward may only flip a token tp2 could not decide."""
    _assert_tie_gap_parity(probes, "ep2")


@needs_gpus(2)
def test_ep_tbo_only_diverges_where_the_arithmetic_had_a_choice(probes):
    """The EP+TBO op stream may only flip a token tp2 could not decide.

    This is the feature's end-to-end gate: two micro-batches interleave their
    all-to-all exchanges on the shared EP group, and if the collective ordering
    ever cross-paired (half A's dispatch met by half B's combine) the output
    would be corrupt on tokens decided by whole nats, not merely near-ties.
    """
    _assert_tie_gap_parity(probes, "ep2_tbo")


@needs_gpus(2)
def test_ep_graph_replay_only_diverges_where_the_arithmetic_had_a_choice(probes):
    """The captured EP a2a may only flip a token tp2 could not decide.

    The gate for EP keeping its decode graphs: the dispatch/combine exchanges
    are recorded into the graph and replayed in lockstep across ranks. A replay
    that dropped an exchange, replayed it against the wrong peer, or let a
    captured buffer be recycled mid-flight would corrupt tokens the weights
    decided by whole nats — so any fork here must still sit on a near-tie.
    """
    _assert_tie_gap_parity(probes, "ep2_graph")


@needs_gpus(2)
def test_most_of_what_ep_says_is_byte_identical(probes):
    """Guards the two above: a coin flip has to be the exception, not the rule."""
    for name in ("ep2", "ep2_tbo", "ep2_graph"):
        fraction = _agreeing_fraction(probes, name)
        assert fraction >= _MIN_AGREEING_FRACTION, (
            f"{name} agrees with tp2 on only {fraction:.0%} of tokens "
            f"(need >= {_MIN_AGREEING_FRACTION:.0%}) — that is noise, not a "
            f"reordered sum"
        )
