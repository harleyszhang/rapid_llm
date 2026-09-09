"""The arm loop every overlap benchmark shares.

Each overlap bench answers one question in the same shape: flip a switch,
rebuild the engine, run one greedy workload, compare. This module owns that
loop so a bench only declares what differs — the switches, the engine shape and
the workload — instead of re-implementing the loop, the metric printing and the
timeline counting for the fifth time.

Layering follows vllm's ``benchmarks/``: one shared layer plus one module per
scenario.

* :mod:`benchmarks.overlap.arms` — the arm loop, metrics, timeline evidence
* :mod:`benchmarks.overlap.levels` — the primitives (L1 copy stream, L2 TBO, L3 chunked AR)
* :mod:`benchmarks.overlap.policies` — the policy benches (TBO / SBO / EP / prefill / scaling / matrix)
* :mod:`benchmarks.overlap.nsys` — kernel-level trace evidence
* :mod:`benchmarks.overlap.plot` — the figures, drawn from the json logs

Usage:
    from benchmarks.overlap.arms import Arm, run_arms, compare, tbo_switch
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from rapid_llm.benchmark import (
    PROMPTS,
    BenchResult,
    expand_prompts,
    free_gpu,
    make_backend,
    report_agreement,
)

#: Timeline recording switch; the engine reads it when it is built.
TIMELINE_ENV = "RAPID_LLM_OVERLAP_TIMELINE"

#: Per-GPU throughput is TPS divided by the parallel degree; every arm reports it
#: beside TTFT / TPOT / TPS.
DEFAULT_TP = 2


@dataclass(frozen=True)
class Arm:
    """One measurement arm: the switches, the engine shape and the workload.

    Attributes:
        label: How the arm is named in the table and the json log.
        env: Switches set in the environment before the engine is built — the
            engine reads them at construction, so they must be settled first.
        unset: Environment keys to remove, for switches that must be absent
            rather than zero (``NCCL_MAX_CTAS``).
        reset: Policy-cache reset for the rank-0 process, which outlives the
            arm; follower ranks are rebuilt per arm and pick the env up at spawn.
        engine: ``make_backend`` kwargs — the engine shape this arm runs.
        batch: Requests in the workload.
        prompt_chars: Prompt truncated to this many characters.
        stretch: Stretch each prompt to a few hundred tokens instead of
            truncating it, with the multiplier varying per request so their
            chunk boundaries land at different steps. Prefill-heavy benches
            need this: a short prompt prefills in one step, leaving nothing for
            an overlap to hide under.
        gen_len: Tokens generated per request.
    """

    label: str
    env: Mapping[str, str] = field(default_factory=dict)
    unset: tuple[str, ...] = ()
    reset: Callable[[], None] | None = None
    engine: Mapping[str, object] = field(default_factory=dict)
    batch: int = 16
    prompt_chars: int = 64
    stretch: bool = False
    gen_len: int = 64


def workload(arm: Arm) -> list[str]:
    """The arm's prompt set: stretched for prefill-heavy benches, else truncated."""
    base = expand_prompts(PROMPTS, arm.batch)
    if arm.stretch:
        return [" ".join([prompt] * (18 + 6 * (i % 5))) for i, prompt in enumerate(base)]
    return [prompt[: arm.prompt_chars] for prompt in base]


def run_arm(model_dir: str, arm: Arm) -> tuple[BenchResult, list[str]]:
    """Run one arm: settle the switches, build the engine, measure, tear down."""
    for key, value in arm.env.items():
        os.environ[key] = value
    for key in arm.unset:
        os.environ.pop(key, None)
    if arm.reset is not None:
        arm.reset()

    backend = make_backend(model_dir, **arm.engine)
    try:
        return backend.measure(workload(arm), arm.gen_len, greedy=True), backend.texts()
    finally:
        backend.close()
        free_gpu()


def run_arms(
    model_dir: str, arms: list[Arm], *, repeat: int = 1
) -> dict[str, tuple[BenchResult, list[str]]]:
    """Run every arm, keeping the best of ``repeat`` by wall clock.

    Args:
        model_dir: Checkpoint every arm loads.
        arms: The arms, in the order they should be printed.
        repeat: Runs per arm; the fastest is reported, the rest discarded.
    """
    rows: dict[str, tuple[BenchResult, list[str]]] = {}
    for arm in arms:
        runs = [run_arm(model_dir, arm) for _ in range(repeat)]
        rows[arm.label] = min(runs, key=lambda run: run[0].total_s)
        print_row(arm.label, rows[arm.label][0])
    return rows


def print_row(label: str, result: BenchResult, tp: int = DEFAULT_TP) -> None:
    """The four metrics on one line; TGS is per-GPU throughput."""
    print(
        f"{label:18s} TTFT {result.ttft_ms:8.1f} ms | TPOT {result.tpot_ms:7.2f} ms "
        f"| TPS {result.tps:8.1f} tok/s | TGS/GPU {result.tps / tp:8.1f}",
        flush=True,
    )


def metrics(result: BenchResult, tp: int = DEFAULT_TP) -> dict:
    """The four metrics as a json-ready dict."""
    return {
        "ttft_ms": round(result.ttft_ms, 2),
        "tpot_ms": round(result.tpot_ms, 3),
        "tps": round(result.tps, 1),
        "tgs_per_gpu": round(result.tps / tp, 1),
    }


def compare(
    rows: Mapping[str, tuple[BenchResult, list[str]]],
    base: str,
    others: list[str],
    *,
    metric: str = "tpot_ms",
    tp: int = DEFAULT_TP,
) -> dict:
    """Print each arm against the base arm and check their greedy streams agree.

    A negative percentage means the arm is slower; the sign convention is
    "positive = faster" everywhere in these benches.
    """
    base_result, base_texts = rows[base]
    summary: dict[str, dict] = {base: metrics(base_result, tp)}
    for label in others:
        result, texts = rows[label]
        delta = (
            (getattr(base_result, metric) - getattr(result, metric))
            / getattr(base_result, metric)
            * 100
        )
        print(f"  -> {label:16s} {metric} {delta:+.1f}% (positive = faster)", flush=True)
        report_agreement(base_texts, [(label, texts)])
        summary[label] = metrics(result, tp)
    return summary


def timeline_overlap(
    model_dir: str,
    arm: Arm,
    *,
    left: Callable[[object], bool],
    right: Callable[[object], bool],
) -> str:
    """One eager round with the timeline on: count the intersecting region pairs.

    Runs eager even for a graphed bench: a captured graph bakes the timeline's
    CUDA events into the graph, so ``collect()`` cannot resolve them. Both forms
    ride the same stream fork-join — capture records those edges verbatim — so
    the eager intersections are valid evidence for the replay.

    Args:
        model_dir: Checkpoint to load.
        arm: The arm whose switches select the overlap; its label is reused.
        left: Predicate picking the compute regions (one half's segments, the
            shared MLP, a chunk's GEMM).
        right: Predicate picking the communication regions it should hide under.
    """
    from rapid_llm.batch_overlap.comm_overlap import CommStreamPool

    timeline_arm = Arm(
        label=arm.label,
        env={**arm.env, TIMELINE_ENV: "1"},
        unset=arm.unset,
        reset=arm.reset,
        engine={**arm.engine, "use_cuda_graph": False},
        batch=arm.batch,
        prompt_chars=arm.prompt_chars,
        stretch=arm.stretch,
        gen_len=min(arm.gen_len, 16),
    )
    CommStreamPool.reset()
    for key, value in timeline_arm.env.items():
        os.environ[key] = value
    if timeline_arm.reset is not None:
        timeline_arm.reset()
    backend = make_backend(model_dir, **timeline_arm.engine)
    try:
        backend.measure(workload(timeline_arm), timeline_arm.gen_len, greedy=True)
        records = CommStreamPool.for_device("cuda").timeline.collect()
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)

    compute = [r for r in records if left(r)]
    comm = [r for r in records if right(r)]
    pairs = 0
    overlap_ms = 0.0
    for exchange in comm:
        for region in compute:
            span = min(exchange.end_ms, region.end_ms) - max(exchange.start_ms, region.start_ms)
            if span > 0:
                pairs += 1
                overlap_ms += span
    return (
        f"comm regions {len(comm)}, compute regions {len(compute)}; "
        f"{pairs} overlapping pairs totalling {overlap_ms:.2f} ms of overlap"
    )


# --------------------------------------------------------------------------- #
# Switch helpers: env vars, the policy-cache reset they need, keys to clear
# --------------------------------------------------------------------------- #
#: Every helper returns the same triple so :func:`make_arm` can compose them.
Switch = tuple[dict[str, str], Callable[[], None] | None, tuple[str, ...]]


def l1_switch(on: bool) -> Switch:
    """L1 pinned-copy overlap; read at engine construction, no cached policy."""
    return {"RAPID_LLM_OVERLAP": "1" if on else "0"}, None, ()


def l3_switch(on: bool) -> Switch:
    """L3 chunked all-reduce."""
    from rapid_llm.batch_overlap.comm_overlap import reset_comm_overlap_policy

    return {"RAPID_LLM_COMM_OVERLAP": "1" if on else "0"}, reset_comm_overlap_policy, ()


def tbo_switch(on: bool, *, min_rows: int | None = None) -> Switch:
    """L2 two-batch overlap.

    ``min_rows`` forces the activation floor: the policy gates TBO to the
    compute-bound regime by default, so a bench measuring the interleave at a
    small batch has to override the ridge explicitly, or it compares off
    against off.
    """
    from rapid_llm.batch_overlap.two_batch_overlap import reset_tbo_policy

    env = {"RAPID_LLM_TBO": "1" if on else "0"}
    if min_rows is not None:
        env["RAPID_LLM_TBO_MIN_ROWS"] = str(min_rows)
    return env, reset_tbo_policy, ()


def sbo_switch(on: bool) -> Switch:
    """SBO: the shared MLP on a side stream beside the dispatch exchange."""
    from rapid_llm.batch_overlap.single_batch_overlap import reset_sbo_policy

    return {"RAPID_LLM_SBO": "1" if on else "0"}, reset_sbo_policy, ()


def sm_budget(on: bool) -> Switch:
    """Cap the exchange's CTAs so the GEMM it overlaps keeps the rest.

    vLLM's DBO default is 20 CTAs for the exchange. Off means the keys are
    absent rather than zero, so NCCL reads its own defaults.
    """
    if on:
        return {"NCCL_MAX_CTAS": "20", "NCCL_MIN_CTAS": "20"}, None, ()
    return {}, None, ("NCCL_MAX_CTAS", "NCCL_MIN_CTAS")


def make_arm(
    label: str,
    *switches: Switch,
    engine: Mapping[str, object],
    batch: int = 16,
    prompt_chars: int = 64,
    stretch: bool = False,
    gen_len: int = 64,
) -> Arm:
    """Build an arm from switch helpers plus an engine shape.

    The switches are merged into one env dict; their resets run in order, since
    a bench may flip two policies at once (TBO and SBO).
    """
    env: dict[str, str] = {}
    unset: list[str] = []
    resets: list[Callable[[], None]] = []
    for switch_env, reset, switch_unset in switches:
        env.update(switch_env)
        unset.extend(switch_unset)
        if reset is not None:
            resets.append(reset)

    def reset_all() -> None:
        for reset in resets:
            reset()

    return Arm(
        label=label,
        env=env,
        unset=tuple(unset),
        reset=reset_all if resets else None,
        engine=engine,
        batch=batch,
        prompt_chars=prompt_chars,
        stretch=stretch,
        gen_len=gen_len,
    )
