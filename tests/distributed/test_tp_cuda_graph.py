"""Decode CUDA graphs under tensor parallelism.

A captured decode step on one GPU records kernels. On two it records the blocks'
all-reduce as well, and that changes what a bug looks like: a rank that runs eager
while its peer replays does not answer differently, it *stops* — the peer waits in
a collective nobody issues. So the interesting failure here is a hang, and every
assertion below is arranged to come back as a failure instead.

Design: each configuration is measured by a *probe* — a spawned, non-daemonic
process (non-daemonic because it spawns the TP follower itself) that builds one
two-rank engine, reports what the graph machinery did, and answers a fixed prompt
set. The parent never touches CUDA or ``parallel_state``, so a wedged rank cannot
leak a TP grid into the rest of the session, and :data:`_PROBE_TIMEOUT_S` turns a
deadlock into a named failure.

What each probe establishes, and why it needs a probe rather than a unit test:

* **The graphs exist and are used.** Capturing and replaying are separate facts.
  A grid that never matches would leave decode eager with every graph still
  resident, and no output comparison would notice; ``replays`` is the counter that
  does.
* **Installation is itself cross-rank evidence.** ``enable_cuda_graph`` only keeps
  graphs after both ranks agree on a grid fingerprint and both pass a
  graph-versus-eager logit check. Rank 0 reporting graphs therefore means rank 1
  reported a matching grid, because a disagreement retires the graphs on *every*
  rank.
* **The answer does not move.** Graph and eager engines of the same width are
  compared token for token. Unlike the tp=1-versus-tp=2 comparison in
  ``test_tp_engine``, no summation order changes here — a replay issues the same
  kernels in the same order as the capture — so byte equality is the right claim,
  and the measured logit difference is exactly ``0.0`` rather than merely small.

Both quantised crossings run because a quantised kernel picks its launch config
from ``M``, which a capture freezes: a scheme that autotuned per step, or that
synchronised on the host, would fail to capture rather than fail to compute.
``int4`` is absent for a checkpoint reason, not a graph one — AWQ needs
``in_features`` to be a multiple of its 128-wide group, and this checkpoint's 896
becomes 448 when split two ways. int4 x TP x graph is covered on a wider model in
``benchmarks/engine/run.py quant``.

Usage:
    pytest tests/distributed/test_tp_cuda_graph.py    # skips below 2 GPUs
"""

from __future__ import annotations

import os
import queue as queue_module
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch.multiprocessing as mp

from rapid_llm.executor.cuda_graph import TP_GRAPH_PARITY_ATOL
from tests.distributed.tp_harness import needs_gpus

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.slow]

#: Two engines plus a follower hold their own weights and cache; sized to leave the
#: machine room. 512 admits the 256 and 512 capture buckets and nothing larger.
_KV_TOKENS = 4096
_MAX_SEQ_LEN = 512
_MAX_NUM_SEQS = 8

#: A checkpoint load, a KV profile, a rendezvous, and a capture of every
#: (batch, bucket) pair followed by a parity check on each. Generous on purpose:
#: its job is to turn a wedged rank into a failure rather than a hung suite.
_PROBE_TIMEOUT_S = 900.0

#: Long enough that most steps are decodes, which is where the graph lives.
_MAX_GEN = 32

#: Answered twice by every probe: once alone, once as a batch of three. The batch
#: is the case that exercises padding onto the captured grid — three rows land on
#: the batch-size-4 graph — and it is also where a replay that reused a stale row
#: count would corrupt a neighbour.
_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Machine learning is",
]

#: Runtime schemes crossed with graph capture. ``None`` is the bf16 control.
_SCHEMES = [None, "fp8", "nvfp4"]

#: Batch size submitted to check padding: not captured itself, so it must be
#: rounded up to one that is.
_UNCAPTURED_BATCH = 3


def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Build one two-rank engine, report the graph facts, answer every prompt.

    Runs in a spawned process, so it takes only picklable arguments and imports
    torch itself. Environment overrides are applied before the engine is built and
    therefore before the follower is spawned, which inherits them.

    A build failure travels back as a traceback rather than being left to time
    out: a rank that dies during rendezvous takes the group with it, and "the
    probe said nothing" is a much worse diagnostic than the exception.
    """
    try:
        os.environ.update(spec.get("env", {}))

        from rapid_llm import SamplingParams
        from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=_MAX_SEQ_LEN,
            max_gpu_num_blocks=_KV_TOKENS,
            max_num_seqs=_MAX_NUM_SEQS,
            use_cuda_graph=spec["graph"],
            tensor_parallel_size=2,
            quantization=spec["quantization"],
        )
        try:
            runner = engine.engine.model_runner
            manager = runner._graph_manager
            # Read, not re-measured. The comparison replays graphs, and a replayed
            # all-reduce has no counterpart now that rank 1 is in its serve loop --
            # measuring it here hangs the group. The startup gate took it while both
            # ranks were in the same window; this only carries the number out.
            params = SamplingParams(
                temperature=0.0,
                max_gen_len=_MAX_GEN,
                repetition_penalty=1.0,
                stop_on_repeat=False,
            )
            report: dict[str, Any] = {
                "installed": manager is not None,
                "graphs": 0 if manager is None else len(manager),
                "parity": None if manager is None else manager.parity_error,
                "padded_batch": runner.graph_batch_size(_UNCAPTURED_BATCH),
                "alone": [engine.generate([p], params)[0].outputs[0].text for p in _PROMPTS],
                "batched": [o.outputs[0].text for o in engine.generate(list(_PROMPTS), params)],
                # Read after generating: this is the count the decodes produced.
                "replays": 0 if manager is None else manager.replays,
            }
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _run_probe(
    model_dir: Path,
    *,
    graph: bool,
    quantization: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one probe to completion, surfacing its traceback as this test's failure."""
    context = mp.get_context("spawn")
    results = context.Queue()
    spec = {
        "model": str(model_dir),
        "graph": graph,
        "quantization": quantization,
        "env": env or {},
    }
    # Not a daemon: the probe spawns the TP follower, and a daemonic process is
    # not allowed children.
    process = context.Process(target=_probe, args=(spec, results), daemon=False)
    process.start()
    try:
        try:
            status, payload = results.get(timeout=_PROBE_TIMEOUT_S)
        except queue_module.Empty:
            pytest.fail(
                f"probe (graph={graph}, quantization={quantization}, env={env}) produced "
                f"nothing in {_PROBE_TIMEOUT_S:.0f}s; a TP graph mismatch hangs rather "
                f"than raising, so this is the shape that failure takes"
            )
        if status == "error":
            pytest.fail(f"probe (graph={graph}, quantization={quantization}) failed:\n{payload}")
        return payload
    finally:
        process.join(timeout=60.0)
        if process.is_alive():  # pragma: no cover - only on a wedged rank
            process.terminate()


@pytest.fixture(scope="module")
def probes(model_dir: Path) -> dict[tuple[str | None, bool], dict[str, Any]]:
    """One report per ``(scheme, graph)``. Module-scoped: six loads, many assertions."""
    return {
        (scheme, graph): _run_probe(model_dir, graph=graph, quantization=scheme)
        for scheme in _SCHEMES
        for graph in (True, False)
    }


def _first_difference(want: str, got: str) -> str:
    """Where two completions diverge, with enough either side to recognise it."""
    for position, (a, b) in enumerate(zip(want, got, strict=False)):
        if a != b:
            return (
                f"diverges at character {position}\n"
                f"  eager: ...{want[max(0, position - 30) : position + 30]!r}\n"
                f"  graph: ...{got[max(0, position - 30) : position + 30]!r}"
            )
    return f"one is a prefix of the other\n  eager: {want!r}\n  graph: {got!r}"


# --------------------------------------------------------------------------- #
# The feature: graphs are captured, kept, and actually replayed
# --------------------------------------------------------------------------- #
@needs_gpus(2)
@pytest.mark.parametrize("scheme", _SCHEMES, ids=lambda s: s or "bf16")
def test_two_ranks_capture_graphs_and_decode_through_them(probes, scheme):
    """TP=2 must install decode graphs, and the decodes must go through them.

    Three claims in one, because they only mean something together: the graphs
    were captured, both ranks agreed to keep them (installation is gated on a
    cross-rank fingerprint and a cross-rank parity reduction, so rank 0 holding
    graphs implies rank 1 did too), and the scheduler's batches landed on the
    grid instead of falling back. Without the last one this file would pass on an
    implementation that captured graphs and never used one.
    """
    report = probes[(scheme, True)]
    assert report["installed"], "TP=2 captured no decode graphs"
    assert report["graphs"] > 0
    assert report["replays"] > 0, (
        f"{report['graphs']} graphs were captured but no decode step replayed one"
    )
    # Padding is what keeps continuous batching on the grid: an odd batch is
    # rounded up to a captured size and the filler rows are discarded.
    assert report["padded_batch"] > _UNCAPTURED_BATCH
    print(f"\n{scheme or 'bf16'}: {report['graphs']} graphs, {report['replays']} replays")


@needs_gpus(2)
@pytest.mark.parametrize("scheme", _SCHEMES, ids=lambda s: s or "bf16")
def test_replay_agrees_with_eager_logits(probes, scheme):
    """A replayed step must produce the eager step's logits.

    This is the check that separates "the graph ran" from "the graph computed the
    model": a stale pointer, a buffer the capture missed, or a collective recorded
    on the wrong stream all show up here as a logit difference. The tolerance
    allows for the all-reduce summing in a different order than an eager one
    would; in practice a replay issues the identical kernels in the identical
    order and the difference measures exactly zero, which is why the number is
    printed rather than merely compared.
    """
    parity = probes[(scheme, True)]["parity"]
    assert parity is not None, "no graphs were installed, so there is nothing to compare"
    print(f"\n{scheme or 'bf16'}: worst graph-vs-eager logit difference {parity:.3e}")
    assert parity <= TP_GRAPH_PARITY_ATOL


@needs_gpus(2)
@pytest.mark.parametrize("scheme", _SCHEMES, ids=lambda s: s or "bf16")
def test_graph_and_eager_answer_the_same_tokens(probes, scheme):
    """32 greedy steps, byte-identical between a graph engine and an eager one.

    Byte equality is demanded here where ``test_tp_engine`` settles for a shared
    prefix, and the difference is real rather than a change of strictness: that
    file compares one GPU against two, where a row-parallel GEMM plus an
    all-reduce reassociates the sum and a greedy tie can flip. This file compares
    two GPUs against the same two GPUs. Nothing about the arithmetic changes —
    only whether the launches came from Python or from a replay — so a single
    differing byte is a bug, not associativity.

    Both groupings are checked because they stress different things: alone is a
    batch of one, and the three-prompt batch is padded onto a larger graph, which
    is where a replay that reused a stale row count would corrupt a neighbour.
    """
    graphed, eager = probes[(scheme, True)], probes[(scheme, False)]
    assert not eager["installed"], "the eager arm captured graphs; it is not a control"
    for grouping in ("alone", "batched"):
        for index, (want, got) in enumerate(zip(eager[grouping], graphed[grouping], strict=True)):
            assert got.strip(), f"{grouping} prompt {index} came back empty"
            assert want == got, f"{grouping} prompt {index}: {_first_difference(want, got)}"


# --------------------------------------------------------------------------- #
# The escape hatches
# --------------------------------------------------------------------------- #
@needs_gpus(2)
def test_kill_switch_restores_the_eager_path(model_dir):
    """``RAPID_LLM_TP_CUDA_GRAPH=0`` must decode eager and still answer.

    The feature records collectives into a graph, so it needs a way to be turned
    off in the field without a redeploy — and the way has to be verified, since an
    escape hatch is only reached when something is already going wrong.
    """
    report = _run_probe(model_dir, graph=True, env={"RAPID_LLM_TP_CUDA_GRAPH": "0"})
    assert not report["installed"], "the kill-switch did not prevent capture"
    assert report["replays"] == 0
    assert all(answer.strip() for answer in report["alone"])


@needs_gpus(2)
def test_lockstep_check_passes_on_every_decode_step(model_dir):
    """``RAPID_LLM_TP_GRAPH_CHECK=1`` must run clean, not just be wired up.

    The debug gate all-reduces each rank's graph choice on every step, so it turns
    a divergent choice into a raised error instead of a deadlock. Enabling it here
    asserts the property it exists to detect: across a batch of one and a padded
    batch of three, both ranks picked the same graph every time. A probe that
    finishes with an answer is that assertion — a mismatch would raise inside the
    engine loop, and a group that desynchronised anyway would hit the timeout.

    Kept off by default because it puts a collective on the decode path, which is
    the path graphs exist to shorten.
    """
    report = _run_probe(model_dir, graph=True, env={"RAPID_LLM_TP_GRAPH_CHECK": "1"})
    assert report["installed"]
    assert report["replays"] > 0, "the lockstep check never ran: nothing replayed"
    assert all(answer.strip() for answer in report["alone"])
