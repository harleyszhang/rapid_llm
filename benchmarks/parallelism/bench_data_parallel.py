#!/usr/bin/env python
"""Data-parallel benchmarks: scaling, prefix-cache routing and CUDA graphs.

Three experiments behind one entry point, all driving the same DP coordinator
(``measure_dp`` / ``DataParallelEngine``) and all diffing outputs so speed never
hides wrongness:

Usage:
    python benchmarks/parallelism/bench_data_parallel.py --model <ckpt> --dp 2
    python benchmarks/parallelism/bench_data_parallel.py --mode prefix --model <ckpt> --dp 2
    python benchmarks/parallelism/bench_data_parallel.py --mode graph --model <ckpt> --dp 2
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm import LLM, DataParallelEngine
from rapid_llm.benchmark import (
    PROMPTS,
    expand_prompts,
    free_gpu,
    measure_generate,
    print_row_table,
    print_run_header,
    report_agreement,
    require_gpus,
    timestamped_log_path,
    write_json_log,
)
from rapid_llm.engine.dp_load_balancer import LOAD_BALANCERS, make_load_balancer

# --------------------------------------------------------------------------- #
# DP scaffolding: this script benchmarks the DP feature itself — replica load
# balancing, scaling curves, router policies — so its row shape, args and
# measurement live here. The shared layer's ``DPBackend`` answers a narrower
# question (what does one DP batch throughput), which is what the cross-engine
# runner needs and not what a scaling study does.
# --------------------------------------------------------------------------- #


@dataclass
class TimedRow:
    """One measured configuration: wall time plus the tokens produced in it.

    Every data-parallel row is judged on ``tps``; deriving it here keeps the DP
    tables on one definition instead of copies of the same formula.
    """

    latency_s: float
    gen_tokens: int

    @property
    def tps(self) -> float:
        return self.gen_tokens / self.latency_s if self.latency_s else 0.0

    def as_dict(self) -> dict:
        """The row's fields plus ``tps``, for the JSON log."""
        return {**asdict(self), "tps": round(self.tps, 1)}


def add_dp_args(
    parser,
    *,
    default_model: str = "my_weight/Qwen2.5-0.5B",
    default_gen_len: int = 128,
    default_iters: int = 2,
    default_max_num_seqs: int = 0,
    default_max_seq_len: int = 1024,
    default_max_gpu_num_blocks: int | None = None,
    dp_help: str = "Replica count",
    gen_len_help: str = "Tokens per request",
    blocks_help: str = "KV cache tokens per replica; profiled when omitted",
) -> None:
    """The knobs every data-parallel benchmark shares; keyword arguments move defaults.

    A workload that wants a different ``gen_len`` or a *stated* KV pool overrides
    those defaults rather than re-declaring the other six arguments.
    """
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--dp", type=int, default=2, help=dp_help)
    parser.add_argument("--gen-len", type=int, default=default_gen_len, help=gen_len_help)
    parser.add_argument(
        "--iters", type=int, default=default_iters, help="Timed repeats (median reported)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=default_max_num_seqs,
        help="Replica concurrency ceiling; 0 sizes it to the per-replica batch",
    )
    parser.add_argument("--max-seq-len", type=int, default=default_max_seq_len)
    parser.add_argument(
        "--max-gpu-num-blocks", type=int, default=default_max_gpu_num_blocks, help=blocks_help
    )
    parser.add_argument("--log-dir", default=None, help="Write a JSON log here")


def measure_dp(
    model: str,
    prompts: list[str],
    *,
    dp: int,
    gen_len: int,
    iters: int,
    max_num_seqs: int,
    warmup_prompts: list[str] | None = None,
    **engine_kwargs,
) -> tuple[float, int, list[str], object]:
    """Time one workload through the DP coordinator.

    The engine is built and torn down per row: rows sharing a process would contend
    for KV, which prices the later ones differently from the earlier ones.

    Returns:
        ``(median latency, median output tokens, last round's texts, tokenizer)`` —
        the tokenizer lets a caller replay routing decisions on exactly the ids the
        balancer saw.
    """
    with DataParallelEngine(
        model=model, data_parallel_size=dp, max_num_seqs=max_num_seqs, **engine_kwargs
    ) as engine:
        tokenizer = engine.tokenizer
        latency, tokens, texts = measure_generate(
            engine.generate,
            prompts,
            gen_len=gen_len,
            iters=iters,
            tokenizer=tokenizer,
            warmup_prompts=warmup_prompts,
        )
    free_gpu()
    return latency, tokens, texts, tokenizer


#: Per-mode fallbacks for the knobs ``add_dp_args`` declares without a default, so
#: an explicit ``--gen-len 128`` under ``--mode prefix`` is honoured rather than
#: silently reset to the mode's own number.
_MODE_DEFAULTS = {
    "scaling": {
        "gen_len": 128,
        "iters": 2,
        "max_num_seqs": 0,
        "max_seq_len": 1024,
        "max_gpu_num_blocks": None,
    },
    # gen_len 4 keeps the run prefill-bound. The KV pool is stated rather than
    # profiled: profiling from a one-token forward hands the cache ~90% of the
    # card and leaves nothing for the logits of a wide prefill batch.
    "prefix": {
        "gen_len": 4,
        "iters": 3,
        "max_num_seqs": 16,
        "max_seq_len": 2048,
        "max_gpu_num_blocks": 65536,
    },
    # The CUDA-graph experiment wants capture in the timed build and the same
    # batch per replica, so the graphs' price shows up against the eager cell.
    "graph": {
        "gen_len": 128,
        "iters": 2,
        "max_num_seqs": 0,
        "max_seq_len": 1024,
        "max_gpu_num_blocks": None,
    },
    "skew": {
        "gen_len": 0,
        "iters": 1,
        "max_num_seqs": 0,
        "max_seq_len": 0,
        "max_gpu_num_blocks": None,
    },
}

# --------------------------------------------------------------------------- #
# scaling mode
# --------------------------------------------------------------------------- #


@dataclass
class DPResult(TimedRow):
    """One configuration's measurement. ``tps`` is the number DP is judged on."""

    label: str = ""
    replicas: int = 1
    batch: int = 0

    @property
    def tps_per_gpu(self) -> float:
        return self.tps / self.replicas if self.replicas else 0.0

    def as_dict(self) -> dict:
        return {**super().as_dict(), "tps_per_gpu": round(self.tps_per_gpu, 1)}


def bench_single_process(model, prompts, gen_len, iters, **kw) -> tuple[DPResult, list[str]]:
    """Reference row: one in-process ``LLM``, no coordinator, no IPC."""
    llm = LLM(model=model, **kw)
    try:
        latency, tokens, texts = measure_generate(
            llm.generate, prompts, gen_len=gen_len, iters=iters, tokenizer=llm.tokenizer
        )
    finally:
        del llm
        free_gpu()
    return DPResult(
        latency_s=latency,
        gen_tokens=tokens,
        label="LLM (in-process)",
        replicas=1,
        batch=len(prompts),
    ), texts


def bench_data_parallel(
    model, replicas, prompts, gen_len, iters, load_balancer, max_num_seqs, **kw
) -> tuple[DPResult, list[str]]:
    """One row per replica count, through the DP coordinator.

    ``max_num_seqs`` is the replica's concurrency ceiling, and it has to be stated:
    a replica hosts a *resident* engine, so unlike the one-shot ``LLM`` row it cannot
    size itself to the batch it is handed. Leaving it at the serving default while the
    reference row decodes the whole batch at once would attribute a difference in
    concurrency to a difference in parallelism.
    """
    latency, tokens, texts, _ = measure_dp(
        model,
        prompts,
        dp=replicas,
        gen_len=gen_len,
        iters=iters,
        max_num_seqs=max_num_seqs,
        load_balancer=load_balancer,
        **kw,
    )
    return DPResult(
        latency_s=latency,
        gen_tokens=tokens,
        label=f"DataParallelEngine dp={replicas}",
        replicas=replicas,
        batch=len(prompts),
    ), texts


def print_scaling_table(results: list[DPResult], baseline: DPResult, scaling: str) -> None:
    print_row_table(
        ["config", "replicas", "batch", "latency (s)", "gen tokens", "TPS", "speedup"],
        [28, 9, 7, 12, 12, 11, 12],
        [
            [
                r.label,
                str(r.replicas),
                str(r.batch),
                f"{r.latency_s:.2f}",
                str(r.gen_tokens),
                f"{r.tps:.1f}",
                f"{r.tps / baseline.tps:.2f}x" if baseline.tps else "—",
            ]
            for r in results
        ],
    )
    scaled = [r for r in results if r.replicas > 1]
    if not scaled or not baseline.tps:
        return
    best = max(scaled, key=lambda r: r.tps)
    efficiency = (best.tps / baseline.tps) / best.replicas * 100
    print(
        f"{scaling} scaling at dp={best.replicas}: {best.tps / baseline.tps:.2f}x "
        f"= {efficiency:.0f}% of linear "
        f"({best.tps_per_gpu:.0f} tok/s per GPU vs {baseline.tps:.0f} on one)"
    )
    if scaling == "strong" and efficiency < 70:
        # Not a defect: a batch this small leaves one GPU bandwidth-bound, so the
        # per-replica batch costs nearly the same step time as the whole batch did.
        print(
            "  the fixed batch is too small to saturate one GPU — rerun with a larger "
            "--batch-size, or with --scaling weak to measure serving throughput"
        )


def run_scaling(args) -> None:
    visible = require_gpus(1)
    if args.dp > visible:
        print(
            f"--dp {args.dp} needs {args.dp} GPUs, only {visible} visible; capping at {visible}",
            file=sys.stderr,
        )
        args.dp = visible

    kw = {
        "max_seq_len": args.max_seq_len,
        "max_gpu_num_blocks": args.max_gpu_num_blocks,
        "quantization": args.quantization,
    }

    def prompts_for(replicas: int) -> list[str]:
        total = args.batch_size * replicas if args.scaling == "weak" else args.batch_size
        return expand_prompts(PROMPTS, total)

    print_run_header(
        args.model,
        {
            "mode": "scaling",
            "scaling": args.scaling,
            "batch": f"{args.batch_size}{' per replica' if args.scaling == 'weak' else ' total'}",
            "gen_len": args.gen_len,
            "iters": args.iters,
            "lb": args.load_balancer,
            "max_num_seqs": args.max_num_seqs or "per-replica batch",
            "quant": args.quantization or "fp16",
            "gpus": visible,
        },
    )

    reference, reference_texts = bench_single_process(
        args.model, prompts_for(1), args.gen_len, args.iters, **kw
    )
    results = [reference]
    agreements: list[tuple[str, list[str]]] = []
    for replicas in range(1, args.dp + 1):
        prompts = prompts_for(replicas)
        # Sized to the share one replica receives, so every row decodes as wide a batch
        # as the reference ``LLM`` row does and the comparison is of replica counts.
        per_replica = -(-len(prompts) // replicas)
        result, texts = bench_data_parallel(
            args.model,
            replicas,
            prompts,
            args.gen_len,
            args.iters,
            args.load_balancer,
            args.max_num_seqs or per_replica,
            **kw,
        )
        results.append(result)
        if len(texts) == len(reference_texts):  # comparable only on the same workload
            agreements.append((result.label, texts))

    baseline = next(r for r in results if r.label.endswith("dp=1"))
    print_scaling_table(results, baseline, args.scaling)
    print()
    report_agreement(reference_texts, agreements)

    if args.log_dir:
        path = timestamped_log_path(
            args.log_dir, f"bench_dp_{Path(args.model).name}_b{args.batch_size}"
        )
        write_json_log(
            path,
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "n_gpus": visible,
                "mode": "scaling",
                "scaling": args.scaling,
                "batch_size": args.batch_size,
                "gen_len": args.gen_len,
                "iters": args.iters,
                "load_balancer": args.load_balancer,
                "quantization": args.quantization,
            },
            [r.as_dict() for r in results],
        )


# --------------------------------------------------------------------------- #
# prefix mode
# --------------------------------------------------------------------------- #

#: One sentence of filler, repeated to build a prefix of the requested length. Its
#: content is irrelevant — what matters is that group members share it *exactly*, since
#: the cache matches on whole blocks of token ids.
_FILLER = "Follow every instruction carefully and answer as precisely as you can. "


@dataclass
class PrefixRow(TimedRow):
    """One (cache, policy) configuration's measurement."""

    prefix_cache: bool = False
    policy: str = ""

    @property
    def label(self) -> str:
        return f"cache={'on ' if self.prefix_cache else 'off'} lb={self.policy}"


def build_workload(
    groups: int, per_group: int, prefix_sentences: int, seed: int = 0
) -> tuple[list[str], list[int]]:
    """Requests from ``groups`` prefix groups, arriving in a shuffled order.

    The arrival order is what the experiment turns on, and a *regular* interleave
    invalidates it. Laying the groups out with a fixed stride, for instance, makes the
    group index a function of the arrival index with the same period as round-robin's
    ``i % dp``: with an even ``groups`` and an odd stride, every even-indexed arrival is
    an even-numbered group, so round-robin lands each group on exactly one replica and
    scores as perfectly affinity-aware without knowing anything. A seeded shuffle has no
    period to collide with, which is also the honest model of a server's arrivals.

    Every prompt ends in a unique suffix so that only the shared *prefix* can be reused,
    and no two prompts are the same request.

    Returns:
        The prompts, and each prompt's group index for :func:`describe_routing`.
    """
    prefixes = [
        f"You are assistant number {g}. " + _FILLER * prefix_sentences for g in range(groups)
    ]
    arrivals = [g for g in range(groups) for _ in range(per_group)]
    random.Random(seed).shuffle(arrivals)
    prompts = [
        f"{prefixes[g]}Question {i}: what is {i} plus {i + 1}?" for i, g in enumerate(arrivals)
    ]
    return prompts, arrivals


def describe_routing(prompts: list[str], group_of: list[int], dp: int, tokenizer) -> None:
    """Print, per policy, how many copies of each prefix the pool ends up holding.

    This is the mechanism the latencies are a consequence of, and it is worth showing
    separately because it is exact where a latency is noisy: a ``(group, replica)`` pair
    is one prefix that some replica has to prefill from scratch. Round-robin's count is
    ``dp x groups`` — every group on every replica — and ``groups`` is the floor.

    Replayed on throwaway balancers rather than read out of the engines, because the
    engines are gone by now and the decision is pure: same ids, same policy, same split.
    """
    token_ids = [list(ids) for ids in tokenizer(prompts, add_special_tokens=True)["input_ids"]]
    print(f"routing ({len(prompts)} requests, {len(set(group_of))} prefixes, dp={dp}):")
    for policy in ("round_robin", "cache_aware"):
        balancer = make_load_balancer(policy, dp)
        placed = [balancer.select(estimated_tokens=len(ids), token_ids=ids) for ids in token_ids]
        per_replica = [placed.count(r) for r in range(dp)]
        copies = len(set(zip(group_of, placed, strict=True)))
        print(
            f"  {policy:12} {copies} prefix copies across the pool "
            f"(floor {len(set(group_of))}), requests per replica {per_replica}"
        )


def measure_prefix(
    model: str,
    prompts: list[str],
    *,
    dp: int,
    policy: str,
    prefix_cache: bool,
    gen_len: int,
    iters: int,
    max_num_seqs: int,
    **kw,
) -> tuple[PrefixRow, list[str], object]:
    """Time one configuration end to end through the DP coordinator.

    The engine is rebuilt per row because the prefix cache is created with the
    scheduler (see :func:`measure_dp`).

    Returns:
        The measurement, the completions, and the checkpoint's tokenizer — the last so
        the caller can replay the routing decisions on exactly the ids the router saw.
    """
    latency, gen_tokens, texts, tokenizer = measure_dp(
        model,
        prompts,
        dp=dp,
        gen_len=gen_len,
        iters=iters,
        max_num_seqs=max_num_seqs,
        load_balancer=policy,
        enable_prefix_cache=prefix_cache,
        # Warm up on prompts from *outside* the workload, or the warm-up would leave
        # the measured run's prefixes already cached and every row would report a hit
        # rate it did not earn.
        warmup_prompts=["Warm up the replicas."] * dp,
        **kw,
    )
    row = PrefixRow(
        latency_s=latency, gen_tokens=gen_tokens, prefix_cache=prefix_cache, policy=policy
    )
    return row, texts, tokenizer


def print_prefix_table(rows: list[PrefixRow], baseline: PrefixRow) -> None:
    print_row_table(
        ["config", "latency (s)", "gen tokens", "speedup"],
        [30, 13, 13, 11],
        [
            [
                r.label,
                f"{r.latency_s:.3f}",
                str(r.gen_tokens),
                f"{baseline.latency_s / r.latency_s:.2f}x" if r.latency_s else "—",
            ]
            for r in rows
        ],
    )

    cache_only = next((r for r in rows if r.prefix_cache and r.policy == "round_robin"), None)
    affinity = next((r for r in rows if r.prefix_cache and r.policy == "cache_aware"), None)
    if not (cache_only and affinity):
        return
    print(
        f"prefix cache alone: {baseline.latency_s / cache_only.latency_s:.2f}x   "
        f"+ affinity routing: {baseline.latency_s / affinity.latency_s:.2f}x   "
        f"(affinity adds {cache_only.latency_s / affinity.latency_s:.2f}x over the cache alone)"
    )


def run_prefix(args) -> None:
    require_gpus(args.dp)

    prompts, group_of = build_workload(args.groups, args.per_group, args.prefix_sentences)
    kw = {"max_seq_len": args.max_seq_len, "max_gpu_num_blocks": args.max_gpu_num_blocks}

    print_run_header(
        args.model,
        {
            "mode": "prefix",
            "dp": args.dp,
            "workload": f"{args.groups} prefix groups x {args.per_group} requests",
            "gen_len": args.gen_len,
            "iters": args.iters,
            "max_num_seqs": args.max_num_seqs,
        },
        width=67,
    )

    rows: list[PrefixRow] = []
    texts_by_label: list[tuple[str, list[str]]] = []
    reference: list[str] = []
    for prefix_cache, policy in [
        (False, "round_robin"),
        (True, "round_robin"),
        (True, "cache_aware"),
    ]:
        row, texts, tokenizer = measure_prefix(
            args.model,
            prompts,
            dp=args.dp,
            policy=policy,
            prefix_cache=prefix_cache,
            gen_len=args.gen_len,
            iters=args.iters,
            max_num_seqs=args.max_num_seqs,
            **kw,
        )
        rows.append(row)
        if not reference:
            reference = texts
        else:
            texts_by_label.append((row.label, texts))
        print(f"  {row.label}: {row.latency_s:.3f}s")

    print_prefix_table(rows, rows[0])
    print()
    describe_routing(prompts, group_of, args.dp, tokenizer)
    print()
    report_agreement(reference, texts_by_label)

    if args.log_dir:
        path = timestamped_log_path(args.log_dir, f"bench_dp_prefix_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "gpu": torch.cuda.get_device_name(0),
                "mode": "prefix",
                "dp": args.dp,
                "groups": args.groups,
                "per_group": args.per_group,
                "prefix_sentences": args.prefix_sentences,
                "gen_len": args.gen_len,
                "iters": args.iters,
                "max_num_seqs": args.max_num_seqs,
            },
            [r.as_dict() for r in rows],
        )


# --------------------------------------------------------------------------- #
# graph mode
# --------------------------------------------------------------------------- #


def gpu_used_mb() -> list[int]:
    """Per-GPU used memory in MB, straight from the driver."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(x) for x in out.strip().splitlines()]


def measure_graph_cell(
    model: str, dp: int, graph: bool, batch: int, gen_len: int, iters: int, max_seq_len: int
) -> dict:
    """One ``(dp, graph)`` cell: build time, TPOT, per-GPU memory, texts.

    ``max_num_seqs`` is the replica's concurrency ceiling and has to equal the
    batch: a replica hosts a resident engine, and an oversized ceiling would let
    the eager cell queue differently from the graphed one.
    """
    prompts = expand_prompts(PROMPTS, batch)
    before = gpu_used_mb()
    time.sleep(2.0)  # let the previous cell's workers fully release

    t0 = time.perf_counter()
    with DataParallelEngine(
        model=model,
        data_parallel_size=dp,
        tensor_parallel_size=1,
        load_balancer="round_robin",
        max_num_seqs=batch,
        max_seq_len=max_seq_len,
        use_cuda_graph=graph,
    ) as engine:
        build_s = time.perf_counter() - t0
        median, tokens, texts = measure_generate(
            engine.generate, prompts, gen_len=gen_len, iters=iters, tokenizer=engine.tokenizer
        )
        used = gpu_used_mb()

    return {
        "build_s": round(build_s, 2),
        "tpot_ms": round(median * 1000 / gen_len, 3),
        "tps": round(tokens / median, 1),
        "gpu_used_mb": used[:dp],
        "gpu_delta_mb": [after - base for after, base in zip(used[:dp], before[:dp], strict=True)],
        "texts": texts,
    }


def run_graph(args) -> None:
    visible = require_gpus(1)
    args.dp = min(args.dp, visible)

    print_run_header(
        args.model,
        {
            "mode": "graph",
            "batch_per_replica": args.batch_size,
            "gen_len": args.gen_len,
            "iters": args.iters,
            "dp": args.dp,
        },
    )

    results = {}
    texts_by_cell = {}
    for dp in range(1, args.dp + 1):
        for graph in (False, True):
            label = f"dp{dp}_{'graph' if graph else 'eager'}"
            cell = measure_graph_cell(
                args.model,
                dp,
                graph,
                args.batch_size * dp,
                args.gen_len,
                args.iters,
                args.max_seq_len,
            )
            results[label] = {k: v for k, v in cell.items() if k != "texts"}
            texts_by_cell[label] = cell["texts"]

            print(f"\n{label}  batch={args.batch_size * dp}  gen={args.gen_len}")
            print(f"  build      {cell['build_s']:7.2f} s   (capture included when graph)")
            print(f"  TPOT       {cell['tpot_ms']:7.3f} ms")
            print(f"  throughput {cell['tps']:7.1f} tok/s")
            print(f"  gpu used   {cell['gpu_used_mb']} MB   delta {cell['gpu_delta_mb']} MB")

    print("\n=== deltas ===")
    for dp in range(1, args.dp + 1):
        eager, graph = results[f"dp{dp}_eager"], results[f"dp{dp}_graph"]
        tpot = (eager["tpot_ms"] - graph["tpot_ms"]) / eager["tpot_ms"]
        print(
            f"dp={dp}: graph cuts TPOT by {tpot:.1%} "
            f"({eager['tpot_ms']:.3f} -> {graph['tpot_ms']:.3f} ms), "
            f"capture adds {graph['build_s'] - eager['build_s']:.2f}s build, "
            f"+{sum(graph['gpu_delta_mb']) - sum(eager['gpu_delta_mb'])} MB"
        )
    if args.dp >= 2:
        eager1, graph2 = results["dp1_eager"], results["dp2_graph"]
        print(
            f"no-lock-step: dp2 graph throughput is {graph2['tps'] / eager1['tps']:.2f}x dp1 eager"
        )

    report_agreement(texts_by_cell["dp1_eager"], list(texts_by_cell.items()))

    if args.log_dir:
        path = timestamped_log_path(args.log_dir, f"dp_graph_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "mode": "graph",
                "batch_per_replica": args.batch_size,
                "gen_len": args.gen_len,
                "iters": args.iters,
                "dp": args.dp,
            },
            results,
        )


# --------------------------------------------------------------------------- #
# skew mode
# --------------------------------------------------------------------------- #


def run_skew(args) -> None:
    """DP skew arm: four balancers x skewed lengths -> per-replica imbalance.

    ROADMAP v0.12.0 acceptance: load-aware policies (total_tokens, cache_aware)
    keep per-replica token skew under ``args.skew_threshold``.  No GPU needed:
    routing is replayed on the tokenizer's output.
    """
    from transformers import AutoTokenizer

    dp = args.dp
    n_short, n_long = args.skew_short_count, args.skew_long_count
    # Short prompts cycle through 1-3 sentences so the baseline is uneven
    _SHORT_CYCLE = (1, 2, 3)
    short = [
        f"Quick request {i}: answer concisely. " * _SHORT_CYCLE[i % len(_SHORT_CYCLE)]
        for i in range(n_short)
    ]
    # Vary long prompt sizes so load-aware policies can distinguish them
    _LONG_CYCLE = (80, 50, 30, 70, 100)
    long = [
        "You are an expert. "
        + _FILLER * _LONG_CYCLE[i % len(_LONG_CYCLE)]
        + f" Question {i}: provide a detailed analysis."
        for i in range(n_long)
    ]
    prompts = short + long
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_ids_list = tokenizer(prompts)["input_ids"]

    tok_per_prompt = [len(ids) for ids in token_ids_list]
    total_tok = sum(tok_per_prompt)
    short_toks = tok_per_prompt[:n_short]
    long_toks = tok_per_prompt[n_short:]

    print(f"skew workload: {len(prompts)} requests ({n_short} short, {n_long} long)  dp={dp}")
    print(
        f"  short {min(short_toks)}-{max(short_toks)} tok x {n_short} = {sum(short_toks)}  "
        f"long {min(long_toks)}-{max(long_toks)} tok x {n_long} = {sum(long_toks)}  "
        f"total {total_tok} tok"
    )

    results = []
    for policy in LOAD_BALANCERS:
        balancer = make_load_balancer(policy, dp)
        per_replica = [0] * dp
        for ids in token_ids_list:
            replica = balancer.select(estimated_tokens=len(ids), token_ids=ids)
            per_replica[replica] += len(ids)

        mean = total_tok / dp
        skew = (max(per_replica) - min(per_replica)) / mean if mean else 0.0
        is_load_aware = policy in ("total_tokens", "cache_aware")
        accepted = skew < args.skew_threshold
        msg = "ACCEPTED" if (not is_load_aware or accepted) else "REJECTED"
        line = f"  {policy:18} tokens per replica {per_replica}, skew {skew:6.1%} -> {msg}"
        if is_load_aware:
            line += f"  (threshold {args.skew_threshold:.0%})"
        print(line)
        results.append(
            {
                "policy": policy,
                "per_replica": per_replica,
                "skew": skew,
                "load_aware": is_load_aware,
                "accepted": accepted if is_load_aware else None,
            }
        )

    load_aware_ok = all(r["accepted"] for r in results if r["accepted"] is not None)
    print(f"\n-> load-aware skew criteria: {'ALL ACCEPTED' if load_aware_ok else 'REJECTED'}")

    if args.log_dir:
        from rapid_llm.benchmark import timestamped_log_path, write_json_log

        path = timestamped_log_path(args.log_dir, f"bench_dp_skew_{Path(args.model).name}")
        write_json_log(
            path,
            {
                "model": args.model,
                "mode": "skew",
                "dp": dp,
                "short_count": n_short,
                "short_sentences": args.skew_short_sentences,
                "long_count": n_long,
                "long_sentences": args.skew_long_sentences,
                "threshold": args.skew_threshold,
            },
            results,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--mode",
        default="scaling",
        choices=["scaling", "prefix", "graph", "skew"],
        help="scaling: replica-count throughput; prefix: prefix-cache routing quality; "
        "graph: CUDA graphs under DP; skew: length-imbalance vs policy",
    )
    # The five knobs below default to None so each mode can fill its own fallback
    # without mistaking an explicit value for "not passed".
    add_dp_args(
        parser,
        default_gen_len=None,
        default_iters=None,
        default_max_num_seqs=None,
        default_max_seq_len=None,
        dp_help="scaling: largest replica count (rows for 1..dp); prefix: replicas (fixed)",
        gen_len_help="Tokens per request; short keeps a prefix run prefill-bound",
        blocks_help="KV cache tokens per replica; profiled when omitted",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="[scaling] Prompts: total across replicas (strong) or per replica (weak); "
        "[graph] per replica",
    )
    parser.add_argument(
        "--scaling", default="weak", choices=["strong", "weak"], help="[scaling] workload shape"
    )
    # Taken from the registry rather than spelled out, so a new policy is benchmarkable
    # the moment it is registered.
    parser.add_argument(
        "--load-balancer",
        default="round_robin",
        choices=list(LOAD_BALANCERS),
        help="[scaling] balancer under test",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        choices=["int8", "int4", "smoothquant"],
        help="[scaling] runtime quantisation",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=4,
        help="[prefix] Distinct shared prefixes; > --dp is where affinity pays",
    )
    parser.add_argument(
        "--per-group", type=int, default=16, help="[prefix] Requests sharing each prefix"
    )
    parser.add_argument(
        "--prefix-sentences",
        type=int,
        default=80,
        help="[prefix] Filler sentences in the shared prefix",
    )
    parser.add_argument(
        "--skew-short-count",
        type=int,
        default=10,
        help="[skew] How many short prompts in the workload",
    )
    parser.add_argument(
        "--skew-short-sentences",
        type=int,
        default=2,
        help="[skew] Filler sentences per short prompt",
    )
    parser.add_argument(
        "--skew-long-count",
        type=int,
        default=3,
        help="[skew] How many long prompts in the workload",
    )
    parser.add_argument(
        "--skew-long-sentences",
        type=int,
        default=60,
        help="[skew] Filler sentences per long prompt",
    )
    parser.add_argument(
        "--skew-threshold",
        type=float,
        default=0.10,
        help="[skew] Acceptable skew for load-aware policies",
    )
    args = parser.parse_args()

    for name, value in _MODE_DEFAULTS[args.mode].items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if args.mode == "prefix":
        run_prefix(args)
    elif args.mode == "graph":
        run_graph(args)
    elif args.mode == "skew":
        run_skew(args)
    else:
        run_scaling(args)


if __name__ == "__main__":
    main()
