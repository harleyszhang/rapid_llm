"""Expert parallelism vs tensor parallelism across batches and context lengths.

The first version of this benchmark measured one point (batch 16, short
prompts) against an *eager* TP baseline, which let a 4.5x "EP win" stand that
was mostly CUDA-graph-vs-eager. This version measures the matrix and owes its
reader the whole environment:

* every arm that can run graphs runs graphs — ``tp2_graph`` is the baseline
  EP must beat, and ``tp1`` (single GPU, graphs) is the no-communication
  ceiling every two-rank arm is measured against;
* five scenarios sweep batch size (1 / 16 / 64) and prompt length
  (short / 2k / 32k) — the all-to-all volume scales with routed rows and the
  prefill chunk width, so EP's cost profile depends on both, and one point
  cannot show where the crossover is;
* metrics are per-request (TTFT from each request's own first-token time,
  TPOT from its own span), because a 32k prompt prefills in 512-token
  chunks — a step-interval TTFT would report the first *chunk*'s end;
* the run is offline: every prompt is submitted at once and the engine
  drains it (no serving queue, no arrival process), which the report states
  alongside batch size, measured prompt lengths and generation length.

Usage:
    python benchmarks/parallelism/bench_expert_parallel.py --model <moe-ckpt>
    python benchmarks/parallelism/bench_expert_parallel.py --model <moe-ckpt> \
        --arms tp2_graph,ep2_graph --scenarios bs16-short    # a quick smoke
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.benchmark import (
    PROMPTS,
    describe_footprint,
    expand_prompts,
    peak_mem_gb,
    print_row_table,
    print_run_header,
    require_gpus,
    run_in_spawned_process,
    run_requests,
    sampling_params,
    timestamped_log_path,
    write_json_log,
)
from rapid_llm.tools.observability.collective_stats import CollectiveStats

#: Ranks per engine for the two-rank arms. The tp1 arm is hardcoded to one.
TP_SIZE = 2

#: The five configurations, in run order. ``tp1`` first so the
#: no-communication ceiling is established before anything pays for a wire;
#: each EP flavour is immediately preceded by its TP counterpart, so the
#: eager/graph pairs read as (baseline, feature).
ARMS: list[tuple[str, dict[str, bool | int]]] = [
    ("tp1", {"tp": 1, "ep": False, "graph": True}),
    ("tp2", {"tp": 2, "ep": False, "graph": False}),
    ("tp2_graph", {"tp": 2, "ep": False, "graph": True}),
    ("ep2", {"tp": 2, "ep": True, "graph": False}),
    ("ep2_graph", {"tp": 2, "ep": True, "graph": True}),
]

#: Parity prompts: short factual completions from outside :data:`PROMPTS`, so
#: they never share a prefix with the (same-prompts) warm-up. ``logprobs=2`` is
#: the instrument, not the subject: it reports the runner-up so a fork's margin
#: can be measured, not inferred.
PARITY_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Water boils at",
    "The largest planet in our solar system is",
    "Python is a language that",
    "Machine learning is",
]

#: Log-probability margin below which a runner-up is close enough that a
#: reordered sum may take the step either way — the same value the EP engine
#: gate uses. A step decided by the weights leads by whole nats.
_TIE_GAP = 0.5

#: Fraction of generated tokens an arm must match ``tp2`` on outright. Every
#: fork is licensed by a small margin, so without this floor a checkpoint that
#: had decayed into noise would pass one coin flip at a time.
_MIN_AGREEMENT = 2 / 3

#: Loading a checkpoint, capturing graphs, warming five scenarios and — for
#: the long build — prefilling 4 x 32k prompts, once per (arm, build) probe.
#: Generous on purpose: it turns a wedged rank into a failure, not a hang.
_PROBE_TIMEOUT_S = 2400.0

#: Scenarios sharing a build share one engine (one weight load, one graph
#: capture); the build's shape envelope must cover every scenario in it.
BUILDS: dict[str, dict[str, int]] = {
    # 16 x (2048 + 128) = 34.8k tokens of KV at once; 64-way short batch fits
    # max_num_seqs with room to spare.
    "wide": {"max_seq_len": 4096, "max_num_seqs": 64, "kv_tokens": 40960},
    # 4 x (32768 + 128) = 131.6k tokens of KV; max_position_embeddings is
    # 262k so a 32k prompt plus generation is well inside the window.
    "long": {"max_seq_len": 33280, "max_num_seqs": 8, "kv_tokens": 139264},
}

#: The sweep: batch size x prompt length. ``parity`` rides one scenario (the
#: original batch-16 short one) because the tie-gap question is about decode
#: arithmetic, which every scenario exercises identically.
SCENARIOS: list[dict[str, Any]] = [
    {"label": "bs1-short", "batch": 1, "prompt_kind": "short", "build": "wide"},
    {"label": "bs16-short", "batch": 16, "prompt_kind": "short", "build": "wide", "parity": True},
    {"label": "bs64-short", "batch": 64, "prompt_kind": "short", "build": "wide"},
    {"label": "bs16-2k", "batch": 16, "prompt_kind": 2048, "build": "wide"},
    {"label": "bs4-32k", "batch": 4, "prompt_kind": 32768, "build": "long"},
]

#: Common words for synthetic long prompts. Seeded sampling makes the prompts
#: reproducible; random words route near-uniformly across experts, which is
#: the honest load for measuring exchange volume (no shared prefix, no
#: hotspot expert).
_WORD_POOL = [
    "river", "mountain", "silence", "engine", "pattern", "journey", "amber",
    "circuit", "harvest", "lantern", "meadow", "signal", "thunder", "village",
    "compass", "fabric", "glacier", "horizon", "island", "kernel", "lattice",
    "marble", "nectar", "orbit", "puzzle", "quartz", "ripple", "summit",
    "tunnel", "velvet", "willow", "anchor", "basket", "candle", "dwelling",
    "ember", "forest", "garden", "hammer", "insect", "jungle", "kitchen",
    "ladder", "mirror", "needle", "orchard", "pebble", "quiver", "ribbon",
    "saddle", "timber", "utensil", "vessel", "window", "yarn", "zenith",
    "bridge", "canyon", "domain", "element", "fossil", "gravel", "harbour",
    "ignite", "juncture", "kingdom", "league", "manor", "nation", "object",
    "prairie", "quarry", "radius", "statue", "temple", "unison", "valley",
    "warrant", "yellow", "zephyr", "acorn", "beacon", "cascade", "drizzle",
    "echo", "falcon", "gadget", "hollow", "ivory", "jasmine", "kelp",
    "lagoon", "monsoon", "nomad", "oracle", "pasture", "quaint", "reef",
    "shrine", "tundra", "umbrella", "verdant", "whisper", "yonder", "alloy",
    "brisk", "cobalt", "dune", "elixir", "flint", "gable", "hazel",
    "indent", "jade", "knoll", "lucid", "mural", "nuance", "opal",
    "plume", "quill", "rustic", "serene", "tide", "unity", "vivid",
    "wren", "yarrow", "arbor", "bloom", "cliff", "dawn", "echo",
    "frost", "grove", "haven", "iris", "jewel", "koala", "lotus",
    "maple", "north", "onyx", "pine", "quail", "ridge", "slope",
    "tide", "unity", "vault", "wolf", "yearn", "zonal", "brook",
    "crest", "dale", "elm", "fern", "gale", "heath", "inbox",
]


# --------------------------------------------------------------------------- #
# workload construction (parent process, so every arm gets the same strings)
# --------------------------------------------------------------------------- #
def _synthetic_prompt(tokenizer, target_tokens: int, seed: int) -> str:
    """A ~``target_tokens`` prompt of seeded common words, trimmed at the
    token level so the length is the tokeniser's, not a word count."""
    rng = random.Random(seed)
    words = [rng.choice(_WORD_POOL) for _ in range(int(target_tokens * 1.4) + 64)]
    ids = tokenizer.encode(" ".join(words))[:target_tokens]
    return tokenizer.decode(ids)


def _scenario_prompts(tokenizer, scenario: dict[str, Any]) -> list[str]:
    kind, batch = scenario["prompt_kind"], scenario["batch"]
    if kind == "short":
        return expand_prompts(PROMPTS, batch)
    return [
        _synthetic_prompt(tokenizer, int(kind), seed=1000 + index)
        for index in range(batch)
    ]


def _prompt_stats(tokenizer, prompts: list[str]) -> dict[str, float]:
    """Measured prompt token lengths — the report's ``seq_len`` disclosure."""
    lengths = [len(tokenizer.encode(prompt)) for prompt in prompts]
    return {
        "min": min(lengths),
        "mean": round(statistics.mean(lengths), 1),
        "max": max(lengths),
    }


# --------------------------------------------------------------------------- #
# metrics: per-request, because long prompts prefill in chunks
# --------------------------------------------------------------------------- #
def _metrics(run) -> dict[str, Any]:
    """Fold one :class:`~rapid_llm.benchmark.metrics.RequestRun` per-request.

    TTFT is each request's own first-token time (mean and worst), TPOT its
    own ``(finish - first_token) / (tokens - 1)``. The step-interval basis
    ``BenchResult`` uses would report the first 512-token chunk's end as a
    32k prompt's TTFT, and average the tail steps where most requests have
    already finished.
    """
    ttfts = run.ttfts_ms()
    tpots = run.per_request_tpot_ms()
    generated = run.gen_tokens
    return {
        "ttft_ms": round(statistics.mean(ttfts), 2) if ttfts else 0.0,
        "ttft_ms_max": round(max(ttfts), 2) if ttfts else 0.0,
        "tpot_ms": round(statistics.mean(tpots), 2) if tpots else 0.0,
        "total_s": round(run.total_s, 3),
        "gen_tokens": generated,
        "tps": round(generated / run.total_s, 1) if run.total_s else 0.0,
    }


# --------------------------------------------------------------------------- #
# one (arm, build) pair, one spawned process
# --------------------------------------------------------------------------- #
def _probe(spec: dict[str, Any], results: mp.Queue) -> None:
    """Build one engine of the requested flavour and measure its scenarios.

    Runs in a spawned process: a wedged rank must not take the still-to-run
    probes down, and each build needs a clean GPU. A failure is reported as a
    traceback rather than left to time out.
    """
    try:
        from rapid_llm import SamplingParams
        from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine

        engine = ContinuousBatchingEngine.from_pretrained(
            model=spec["model"],
            device="cuda:0",
            max_seq_len=spec["max_seq_len"],
            max_gpu_num_blocks=spec["kv_tokens"],
            max_num_seqs=spec["max_num_seqs"],
            use_cuda_graph=spec["graph"],
            tensor_parallel_size=spec["tp"],
            enable_expert_parallel=spec["ep"],
        )
        try:
            runner = engine.engine.model_runner
            manager = runner._graph_manager
            weights_gib, kv_tokens = describe_footprint(runner, replicas=spec["tp"])

            scenario_reports = []
            for scenario in spec["scenarios"]:
                prompts, gen_len = scenario["prompts"], scenario["gen_len"]
                # Warm up on the measured workload itself, at the same shape:
                # autotuned tiles are per-shape, so a differently-shaped warm-up
                # would leave the first measured step compiling. The prefix
                # cache is off by default, so re-running these prompts measures
                # fresh prefills, not cache hits.
                engine.generate(prompts, sampling_params(8))

                entry: dict[str, Any] = {"scenario": scenario["label"]}
                if scenario.get("parity"):
                    parity_params = SamplingParams(
                        temperature=0.0,
                        max_gen_len=spec["parity_gen_len"],
                        repetition_penalty=1.0,
                        stop_on_repeat=False,
                        logprobs=2,
                    )
                    entry["parity"] = [
                        _record(output.outputs[0])
                        for output in engine.generate(spec["parity_prompts"], parity_params)
                    ]

                torch.cuda.reset_peak_memory_stats()
                replays_before = manager.replays if manager is not None else None
                with CollectiveStats.collect() as stats:
                    run = run_requests(
                        engine, prompts, sampling_params(gen_len)
                    )
                entry["metrics"] = _metrics(run)
                entry["traffic"] = {
                    op.value: {"calls": tally.calls, "nbytes": tally.nbytes}
                    for op, tally in stats.tallies().items()
                }
                entry["graph_replays"] = (
                    None if replays_before is None else manager.replays - replays_before
                )
                entry["peak_mem_gib"] = round(peak_mem_gb(), 2)
                scenario_reports.append(entry)

            report = {
                "arm": spec["arm"],
                "tp": spec["tp"],
                "ep": spec["ep"],
                "graph": spec["graph"],
                "build": spec["build"],
                "weights_gib": round(weights_gib, 2),
                "kv_tokens": kv_tokens,
                "executor": type(engine._executor).__name__,
                "followers": len(mp.active_children()),
                "scenarios": scenario_reports,
            }
        finally:
            engine.shutdown()
    except Exception:
        results.put(("error", traceback.format_exc()))
    else:
        results.put(("ok", report))


def _run_arm(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one probe to completion, surfacing its traceback as this script's exit."""
    status, payload = run_in_spawned_process(_probe, spec, _PROBE_TIMEOUT_S)
    if status == "timeout":
        raise SystemExit(
            f"arm {spec['arm']}/{spec['build']} produced nothing in "
            f"{_PROBE_TIMEOUT_S:.0f}s — a rank wedged during build or rendezvous"
        )
    if status == "error":
        raise SystemExit(f"arm {spec['arm']}/{spec['build']} failed:\n{payload}")
    return payload


# --------------------------------------------------------------------------- #
# parity against the tp2 baseline
# --------------------------------------------------------------------------- #
def _record(completion) -> dict[str, Any]:
    """One completion as picklable primitives: text, its tokens, their margins.

    A ``None`` margin marks a step with no runner-up logged — a single
    candidate nothing could have tied with.
    """
    gaps: list[float | None] = []
    for record in completion.logprobs or ():
        top = record.top_logprobs
        gaps.append(float(top[0] - top[1]) if len(top) >= 2 else None)
    return {
        "text": completion.text,
        "tokens": [r.token_id for r in completion.logprobs or ()],
        "gaps": gaps,
    }


def _fork(base: list[int], other: list[int]) -> int:
    """Step at which two token sequences first part, or the shorter length."""
    for step, (a, b) in enumerate(zip(base, other, strict=False)):
        if a != b:
            return step
    return min(len(base), len(other))


def _compare_to_baseline(baseline: list[dict], arm: list[dict]) -> dict[str, Any]:
    """Parity of one arm's completions against ``tp2``'s, in tie-gap terms.

    A fork is *licensed* when ``tp2``'s own margin at that step was within
    :data:`_TIE_GAP` — a reordered MoE reduction may only change a token the
    arithmetic could not decide. The agreement fraction guards the other side:
    forks must be the exception, not the rule.
    """
    forks: list[int] = []
    margins: list[float | None] = []
    same = total = 0
    licensed = True
    for want, got in zip(baseline, arm, strict=True):
        step = _fork(want["tokens"], got["tokens"])
        forks.append(step)
        if step == len(want["tokens"]) == len(got["tokens"]):
            margins.append(None)  # identical completions — nothing to license
            same += step
            total += step
            continue
        # A real fork (a wrong token, or one arm stopping early): the baseline's
        # own margin there decides whether the arithmetic could have gone either
        # way. A missing runner-up or a whole-nat lead means an unlicensed fork.
        margin = want["gaps"][step] if step < len(want["gaps"]) else None
        margins.append(margin)
        if margin is None or margin > _TIE_GAP:
            licensed = False
        same += sum(1 for a, b in zip(want["tokens"], got["tokens"], strict=False) if a == b)
        total += len(want["tokens"])
    return {
        "fork_steps": forks,
        "fork_margins": margins,
        "agreement": round(same / total, 4) if total else 1.0,
        "tie_licensed": licensed,
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _entry(reports: list[dict], arm: str, scenario: str) -> dict[str, Any] | None:
    """The (arm, scenario) measurement, from whichever build holds it."""
    for report in reports:
        if report["arm"] != arm:
            continue
        for entry in report["scenarios"]:
            if entry["scenario"] == scenario:
                return entry
    return None


def _traffic_cells(entry: dict[str, Any]) -> tuple[dict, dict, float]:
    """``(a2a, all_reduce, data-plane MiB)`` from one measured run.

    The broadcast of each step's plan rides the control plane (gloo, pickled
    objects); the data plane is everything else, which is the number the
    all-to-all-vs-all-reduce question is about.
    """
    traffic = entry["traffic"]
    a2a = traffic.get("all_to_all", {"calls": 0, "nbytes": 0})
    all_reduce = traffic.get("all_reduce", {"calls": 0, "nbytes": 0})
    control = traffic.get("broadcast_object", {"nbytes": 0})["nbytes"]
    data = (sum(t["nbytes"] for t in traffic.values()) - control) / 2**20
    return a2a, all_reduce, data


def _attach_parity(reports: list[dict]) -> None:
    """Compare every arm's parity records against ``tp2``'s, in place."""
    reference = None
    for report in reports:
        if report["arm"] != "tp2":
            continue
        for entry in report["scenarios"]:
            if "parity" in entry:
                reference = entry["parity"]
    if reference is None:
        return
    for report in reports:
        if report["arm"] == "tp2":
            continue
        for entry in report["scenarios"]:
            if "parity" in entry:
                entry["parity_vs_tp2"] = _compare_to_baseline(reference, entry["parity"])


def _earliest_fork(entry: dict[str, Any]) -> int | None:
    """Earliest step at which this arm really forked, or ``None`` if it never did.

    A fork shorter than the baseline's own completion counts even when the arm
    stopped early: under greedy, identical arithmetic cannot stop at different
    lengths.
    """
    return min(
        (
            fork
            for fork, want in zip(
                entry["parity_vs_tp2"]["fork_steps"], entry["parity"], strict=True
            )
            if fork < len(want["tokens"])
        ),
        default=None,
    )


def _worst_margin(entry: dict[str, Any]) -> str:
    """The largest margin among an arm's real forks — the one a licence must cover."""
    margins = [
        margin
        for margin, fork, want in zip(
            entry["parity_vs_tp2"]["fork_margins"],
            entry["parity_vs_tp2"]["fork_steps"],
            entry["parity"],
            strict=True,
        )
        if fork < len(want["tokens"]) and margin is not None
    ]
    return f"{max(margins):.3f}" if margins else "—"


def _print_tables(reports: list[dict], scenarios: list[dict[str, Any]]) -> None:
    for scenario in scenarios:
        label = scenario["label"]
        tokens = scenario["prompt_tokens"]
        print(
            f"\nscenario {label} — batch {scenario['batch']}, prompt "
            f"{tokens['min']}-{tokens['mean']}-{tokens['max']} tok (min-mean-max), "
            f"gen {scenario['gen_len']} tok, offline"
        )
        tp1 = _entry(reports, "tp1", label)
        tp2 = _entry(reports, "tp2", label)
        tps1 = tp1["metrics"]["tps"] if tp1 else 0.0
        tps2 = tp2["metrics"]["tps"] if tp2 else 0.0
        rows = []
        for arm, _ in ARMS:
            entry = _entry(reports, arm, label)
            if entry is None:
                continue
            metrics = entry["metrics"]
            rows.append(
                [
                    arm,
                    f"{metrics['ttft_ms']:.1f}",
                    f"{metrics['ttft_ms_max']:.1f}",
                    f"{metrics['tpot_ms']:.2f}",
                    f"{metrics['tps']:.1f}",
                    f"{metrics['tps'] / tps1:.2f}x" if tps1 else "—",
                    f"{metrics['tps'] / tps2:.2f}x" if tps2 else "—",
                    f"{entry['peak_mem_gib']:.1f}",
                    "replays " + str(entry["graph_replays"])
                    if entry["graph_replays"] is not None
                    else "eager",
                ]
            )
        print_row_table(
            [
                "config", "TTFT (ms)", "TTFT max", "TPOT (ms)", "TPS",
                "vs tp1", "vs tp2", "peak (GiB)", "graph",
            ],
            [12, 11, 10, 11, 9, 8, 8, 11, 13],
            rows,
        )

        traffic_rows = []
        for arm, _ in ARMS:
            entry = _entry(reports, arm, label)
            if entry is None:
                continue
            a2a, all_reduce, data = _traffic_cells(entry)
            traffic_rows.append(
                [
                    arm,
                    str(a2a["calls"]),
                    f"{a2a['nbytes'] / 2**20:.1f}",
                    str(all_reduce["calls"]),
                    f"{all_reduce['nbytes'] / 2**20:.1f}",
                    f"{data:.1f}",
                ]
            )
        print_row_table(
            ["config", "a2a calls", "a2a (MiB)", "all-red. calls", "all-red. (MiB)", "data plane (MiB)"],
            [12, 11, 11, 15, 15, 17],
            traffic_rows,
        )
        print("  wire bytes are the driver rank's view of the measured run only")
        print(
            "  caveat: a graph replay bypasses the Python accounting, so graph arms' "
            "bytes cover their eager (prefill) passes only — read the eager arms' "
            "rows for per-op wire cost"
        )

    # The parity table rides the bs16-short scenario.
    for scenario in scenarios:
        if not scenario.get("parity"):
            continue
        rows = []
        for arm, _ in ARMS:
            if arm == "tp2":
                rows.append([arm, "—", "—", "—", "reference"])
                continue
            entry = _entry(reports, arm, scenario["label"])
            if entry is None or "parity_vs_tp2" not in entry:
                continue
            verdict = "identical"
            if _earliest_fork(entry) is not None:
                verdict = "tie-licensed" if entry["parity_vs_tp2"]["tie_licensed"] else "SUSPECT"
            if entry["parity_vs_tp2"]["agreement"] < _MIN_AGREEMENT:
                verdict = "SUSPECT (agreement)"
            earliest = _earliest_fork(entry)
            rows.append(
                [
                    arm,
                    f"step {earliest}" if earliest is not None else "none",
                    _worst_margin(entry),
                    f"{entry['parity_vs_tp2']['agreement']:.1%}",
                    verdict,
                ]
            )
        print(f"\nparity vs tp2 ({scenario['label']}, greedy)")
        print_row_table(
            ["config", "first fork", "worst fork margin (nat)", "token agreement", "verdict"],
            [12, 12, 23, 16, 20],
            rows,
        )

    # The cross-scenario matrices: absolute TPS, then the ratios that answer
    # the actual question (does EP beat TP *at the same graph setting*, and
    # what does the second GPU buy over one?).
    matrix_rows = []
    for arm, _ in ARMS:
        row = [arm]
        for scenario in scenarios:
            entry = _entry(reports, arm, scenario["label"])
            row.append(f"{entry['metrics']['tps']:.1f}" if entry else "—")
        matrix_rows.append(row)
    print("\nthroughput (tok/s) by scenario")
    print_row_table(
        ["config"] + [scenario["label"] for scenario in scenarios],
        [12] + [12] * len(scenarios),
        matrix_rows,
    )

    ratio_rows = []
    for scenario in scenarios:
        label = scenario["label"]

        def tps(arm: str, label: str) -> float | None:
            entry = _entry(reports, arm, label)
            return entry["metrics"]["tps"] if entry else None

        pairs = [
            (tps("ep2", label), tps("tp2", label)),
            (tps("ep2_graph", label), tps("tp2_graph", label)),
            (tps("tp2_graph", label), tps("tp1", label)),
            (tps("tp2", label), tps("tp1", label)),
        ]
        ratio_rows.append(
            [label] + [
                f"{num / den:.2f}x" if num and den else "—" for num, den in pairs
            ]
        )
    print("\nthe ratios that answer the question (same graph setting both sides)")
    print_row_table(
        ["scenario", "ep2/tp2", "ep2_graph/tp2_graph", "tp2_graph/tp1", "tp2/tp1"],
        [14, 10, 19, 14, 10],
        ratio_rows,
    )


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expert parallelism vs tensor parallelism across batches "
        "and context lengths on one MoE checkpoint"
    )
    parser.add_argument(
        "--model", required=True, help="MoE checkpoint dir (config.json + safetensors)"
    )
    parser.add_argument(
        "--arms",
        default=",".join(label for label, _ in ARMS),
        help="comma-separated subset of: " + ",".join(label for label, _ in ARMS),
    )
    parser.add_argument(
        "--scenarios",
        default=",".join(s["label"] for s in SCENARIOS),
        help="comma-separated subset of: " + ",".join(s["label"] for s in SCENARIOS),
    )
    parser.add_argument(
        "--gen-len", type=int, default=128, help="Tokens per request in every measured run"
    )
    parser.add_argument(
        "--parity-gen-len", type=int, default=24, help="Tokens per request in the parity run"
    )
    parser.add_argument(
        "--log-dir",
        default="docs/benchmark_logs",
        help="Where the JSON log lands; an empty value skips the log",
    )
    args = parser.parse_args()

    arm_labels = [label.strip() for label in args.arms.split(",") if label.strip()]
    scenario_labels = [label.strip() for label in args.scenarios.split(",") if label.strip()]
    known_arms = {label for label, _ in ARMS}
    known_scenarios = {s["label"] for s in SCENARIOS}
    if unknown := [label for label in arm_labels if label not in known_arms]:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {sorted(known_arms)}")
    if unknown := [label for label in scenario_labels if label not in known_scenarios]:
        raise SystemExit(f"unknown scenario(s) {unknown}; choose from {sorted(known_scenarios)}")
    arms = [(label, dict(flags)) for label, flags in ARMS if label in arm_labels]
    scenarios = [dict(s) for s in SCENARIOS if s["label"] in scenario_labels]

    require_gpus(TP_SIZE)

    # Same prompt strings to every arm, with measured lengths — the report's
    # seq_len disclosure. Built in the parent so no arm can drift.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for scenario in scenarios:
        scenario["prompts"] = _scenario_prompts(tokenizer, scenario)
        scenario["prompt_tokens"] = _prompt_stats(tokenizer, scenario["prompts"])
        scenario["gen_len"] = args.gen_len

    print_run_header(
        args.model,
        {
            "mode": "offline (all prompts submitted at once)",
            "arms": " ".join(arm_labels),
            "gen_len": args.gen_len,
            "parity": f"{len(PARITY_PROMPTS)} x {args.parity_gen_len} tok (bs16-short)",
            "sampling": "greedy",
            "ranks": f"tp1 and tp{TP_SIZE} arms, see table",
        },
    )
    print_row_table(
        ["scenario", "batch", "prompt tok (min-mean-max)", "gen tok", "build (kv tok, max seq)"],
        [14, 7, 26, 8, 30],
        [
            [
                scenario["label"],
                str(scenario["batch"]),
                "{min}-{mean}-{max}".format(**scenario["prompt_tokens"]),
                str(scenario["gen_len"]),
                (
                    f"{scenario['build']} (kv {BUILDS[scenario['build']]['kv_tokens']} tok, "
                    f"seq {BUILDS[scenario['build']]['max_seq_len']})"
                ),
            ]
            for scenario in scenarios
        ],
    )

    # Builds in definition order; every (arm, build) pair is its own process.
    build_names = [name for name in BUILDS if any(s["build"] == name for s in scenarios)]
    reports: list[dict[str, Any]] = []
    for label, flags in arms:
        for build_name in build_names:
            build_scenarios = [
                {k: v for k, v in scenario.items() if k != "build"}
                for scenario in scenarios
                if scenario["build"] == build_name
            ]
            spec = {
                "arm": label,
                "model": args.model,
                "build": build_name,
                **BUILDS[build_name],
                **flags,
                "scenarios": build_scenarios,
                "parity_prompts": PARITY_PROMPTS,
                "parity_gen_len": args.parity_gen_len,
            }
            print(
                f"\n[{label}/{build_name}] tp={flags['tp']} ep={flags['ep']} "
                f"graph={flags['graph']} — building",
                flush=True,
            )
            t0 = time.perf_counter()
            reports.append(_run_arm(spec))
            print(f"[{label}/{build_name}] finished in {time.perf_counter() - t0:.0f}s", flush=True)
            time.sleep(2.0)  # let the driver fully release the previous probe's GPUs

    _attach_parity(reports)
    _print_tables(reports, scenarios)

    if args.log_dir:
        path = timestamped_log_path(args.log_dir, f"expert_parallel_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "mode": "offline (all prompts submitted at once, no serving queue)",
                "arms": {label: flags for label, flags in ARMS if label in arm_labels},
                "scenarios": [
                    {
                        "label": s["label"],
                        "batch": s["batch"],
                        "prompt_kind": s["prompt_kind"],
                        "prompt_tokens": s["prompt_tokens"],
                        "gen_len": s["gen_len"],
                        "build": s["build"],
                        "engine": BUILDS[s["build"]],
                    }
                    for s in scenarios
                ],
                "parity_gen_len": args.parity_gen_len,
                "tie_gap_nats": _TIE_GAP,
                "min_agreement": _MIN_AGREEMENT,
                "metric_basis": "per-request: TTFT from each request's first-token time "
                "(mean and max), TPOT from (finish - first_token)/(tokens-1) — long "
                "prompts prefill in 512-token chunks, so step-interval metrics would "
                "misreport TTFT",
                "traffic_view": "driver rank, measured run only; graph replays bypass the "
                "accounting, so graph arms' bytes are eager passes only",
            },
            reports,
        )


if __name__ == "__main__":
    main()
