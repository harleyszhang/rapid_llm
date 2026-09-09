"""Optimization-feature matrix: each feature alone, then in combination.

The continuous-batching engine carries a set of switches -- decode CUDA graph,
lazy capture, prefix cache, chunked prefill, the launch/harvest pipeline,
background tokenize, fp8 KV, ngram speculative decoding and the decode admission
window -- and each is documented with an expected gain. This script measures
them instead of asserting them: one cell per feature against the all-off
baseline, then a set of combinations, so a feature that only pays on top of
another (or one that regresses when combined) shows up as a number rather than
as a claim.

Every cell reports TTFT / TPOT / TPS, plus per-GPU TPS when the run spans more
than one card. ``--verify`` runs greedy and asserts each cell produces the same
text as the baseline, so a speedup that moved the output is a failure, not a win.

The workload has to match the feature being measured: chunked prefill needs
prompts longer than the token budget, and the prefix cache needs prompts that
share one. ``--workload`` picks that; running a short-prompt workload against
``chunked_prefill`` measures a switch that never engaged.

Two features act at engine-build time rather than through a ``from_pretrained``
override: ``overlap_off`` disables the L1 cross-stream overlap
(``RAPID_LLM_OVERLAP=0``; it is on by default, so this cell measures what it
buys) and ``router_fp32_cache`` swaps the MoE router GEMM for the predecessor
the tier-4 refactor replaced. Each is applied per cell and undone after, so it
composes with the kwarg features like any other.

Usage:
    python benchmarks/engine/run.py optimizations --model-dir CKPT --workload short
    python benchmarks/engine/run.py optimizations --model-dir CKPT --workload long \
        --features chunked_prefill pipeline --verify
    python benchmarks/engine/run.py optimizations --model-dir CKPT --workload shared \
        --combos cuda_graph+prefix_cache --json docs/benchmark_logs/engine/optim.json
    python benchmarks/engine/run.py optimizations --model-dir CKPT \
        --features overlap_off router_fp32_cache --greedy --verify
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rapid_llm.benchmark import (
    PROMPTS,
    expand_prompts,
    gpu_tag,
    print_row_table,
    run_requests,
    sampling_params,
    write_json_log,
)

#: Environment variable switching L1 cross-stream overlap (read at engine build).
OVERLAP_ENV = "RAPID_LLM_OVERLAP"

#: Environment variable enabling ngram speculative decoding (read at engine build).
SPECULATE_ENV = "LITE_LLAMA_SPECULATE"

#: fp32 weight copies the ``fp32_cache`` router variant keeps, keyed by ``id()``
#: of the source weight. The dict holds the copy alive, so an id cannot be
#: reused under it.
_FP32_CACHE: dict[int, torch.Tensor] = {}


def _patched_router_gemm(variant: str):
    """A ``_router_gemm`` whose only variant-dependent part is the GEMM itself.

    ``tier4`` is the production path (bf16 operands, fp32 accumulate/output in
    one cuBLAS GEMM); ``fp32_cache`` is the predecessor it replaced -- a cached
    fp32 weight copy plus a per-step ``x.float()`` widen and a simt SGEMM. Both
    emit fp32 logits and pick the same experts, so the A/B isolates the GEMM.
    """

    def _gemm(x: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
        if variant == "tier4":
            low_prec = gate_weight.dtype in (torch.bfloat16, torch.float16)
            if gate_weight.is_cuda and low_prec:
                x_gemm = x if x.dtype == gate_weight.dtype else x.to(gate_weight.dtype)
                return torch.mm(x_gemm, gate_weight.t(), out_dtype=torch.float32)
            return F.linear(x.float(), gate_weight.float())
        if variant == "fp32_cache":
            cached = _FP32_CACHE.get(id(gate_weight))
            if cached is None:
                cached = _FP32_CACHE[id(gate_weight)] = gate_weight.detach().float()
            return F.linear(x.float(), cached)
        raise ValueError(f"unknown router variant {variant!r}")

    return _gemm


#: The all-off cell every feature is compared against. ``max_num_batched_tokens``
#: is left at the engine default, which is far above any prompt here, so no chunk
#: is ever cut and the chunked kernel never runs.
BASELINE: dict[str, Any] = {
    "use_cuda_graph": False,
    "enable_prefix_cache": False,
    "kv_cache_dtype": "auto",
    "cuda_graph_lazy": False,
    "async_tokenize": False,
    "pipeline": False,
    "decode_window_steps": 0,
}

#: One ``from_pretrained`` override per feature. A feature that needs another to
#: exist says so in its own entry (lazy capture is meaningless without graphs).
FEATURES: dict[str, dict[str, Any]] = {
    "cuda_graph": {"use_cuda_graph": True},
    "lazy_graph": {"use_cuda_graph": True, "cuda_graph_lazy": True},
    "prefix_cache": {"enable_prefix_cache": True},
    "chunked_prefill": {"max_num_batched_tokens": 256},
    "pipeline": {"pipeline": True},
    "async_tokenize": {"async_tokenize": True},
    "fp8_kv": {"kv_cache_dtype": "fp8"},
    "decode_window": {"decode_window_steps": 2},
    # Side-effect features (no from_pretrained override): _side_effects applies
    # them at engine-build time. overlap is on by default, so the cell turns it
    # OFF to measure its contribution; router_fp32_cache swaps the router GEMM
    # for the predecessor implementation; speculate turns on ngram speculative
    # decoding (O5) via LITE_LLAMA_SPECULATE -- its payoff is fewer decode steps
    # per token, which the TPOT column reads directly, and greedy verify must
    # still pass because speculation is checked against the target model.
    "overlap_off": {},
    "router_fp32_cache": {},
    "speculate": {},
}

#: Features that may legitimately change the generated text.

#: ``fp8_kv`` stores e4m3 K/V, so it trades precision for capacity.
#: ``async_tokenize`` changes *when* a request is admitted -- an encode that
#: lands later joins a later wave (measured: 3 requests in the first wave
#: against 8 without it) -- and admission sets the prefill batch shape, which
#: this engine's arithmetic is sensitive to. Neither is a regression; reporting
#: them as failures would hide the failures that are real.
#: ``router_fp32_cache`` runs the router GEMM with fp32 operands where tier-4
#: uses bf16 operands with fp32 accumulate, so the logits can differ in the last
#: bits and flip a greedy tie -- it picks the same experts, but the decoded text
#: is not guaranteed bit-identical.
OUTPUT_SHIFTING_FEATURES: frozenset[str] = frozenset(
    {"fp8_kv", "async_tokenize", "router_fp32_cache"}
)

#: Combinations worth measuring: pairs whose mechanisms interact rather than
#: merely add. ``cuda_graph`` is in most of them because it is the one switch
#: whose absence makes every other launch cost visible.
DEFAULT_COMBOS: tuple[tuple[str, ...], ...] = (
    ("cuda_graph", "prefix_cache"),
    ("cuda_graph", "chunked_prefill"),
    ("cuda_graph", "pipeline"),
    ("cuda_graph", "fp8_kv"),
    ("cuda_graph", "prefix_cache", "chunked_prefill"),
    ("cuda_graph", "prefix_cache", "chunked_prefill", "pipeline"),
)

#: Sentences the long and shared workloads are built from.
_SENTENCES = [
    "The history of computing spans mechanical calculators, vacuum tubes and silicon.",
    "A transformer attends over every position, which is why its cost grows quadratically.",
    "Paged memory lets a cache hold more sequences than a contiguous allocation would.",
    "Greedy decoding picks the most likely token, so two engines agree only if their math does.",
    "Bandwidth, not arithmetic, is what bounds a small model's decode step.",
]


@dataclass
class Row:
    """One measured cell.

    ``tps_per_gpu`` is the throughput each card contributed; it equals ``tps`` at
    one card and is what a parallel row is judged on, since aggregate throughput
    grows with the card count by construction.
    """

    label: str
    features: tuple[str, ...]
    ttft_ms: float
    tpot_ms: float
    tps: float
    gen_tokens: int
    latency_s: float
    gpus: int = 1

    @property
    def tps_per_gpu(self) -> float:
        return self.tps / self.gpus

    def as_dict(self) -> dict:
        return {**asdict(self), "tps_per_gpu": round(self.tps_per_gpu, 1)}


def build_prompts(workload: str, batch: int, shared_sentences: int) -> list[str]:
    """The prompt set for one workload shape.

    ``short`` is the shared benchmark prompt list. ``long`` repeats the sentence
    bank until a prompt exceeds any realistic token budget, which is what makes
    ``chunked_prefill`` engage. ``shared`` gives every prompt the same long
    prefix and a distinct tail, which is what makes ``prefix_cache`` hit.
    """
    if workload == "short":
        return expand_prompts(PROMPTS, batch)
    bank = _SENTENCES * 8
    if workload == "long":
        return [
            " ".join(bank) + f"\nQuestion {i}: summarise the passage above." for i in range(batch)
        ]
    if workload == "repetitive":
        # ngram speculation only proposes when the prompt has repeated n-grams:
        # a repeated code snippet gives the proposer something to accept.
        base = "Write a Python function that calculates the sum of two numbers. "
        repeat = "def add(a, b): return a + b. "
        return [base + repeat * 5 for _ in range(batch)]
    prefix = " ".join(bank[:shared_sentences])
    return [f"{prefix}\nQuestion {i}: what does the passage say?" for i in range(batch)]


@contextmanager
def _side_effects(features: tuple[str, ...]):
    """Apply the two build-time side-effect features for one cell, undo on exit.

    ``overlap_off`` sets ``RAPID_LLM_OVERLAP=0`` (the engine reads it when it
    is built); ``router_fp32_cache`` swaps ``moe.router._router_gemm`` for the
    predecessor the tier-4 refactor replaced. Kwarg features are applied by the
    caller through ``FEATURES``, so this wraps only the two that act outside
    ``from_pretrained`` -- and restores whatever they displaced.
    """
    from rapid_llm.modules.moe import router

    saved_overlap = os.environ.get(OVERLAP_ENV)
    saved_speculate = os.environ.get(SPECULATE_ENV)
    saved_router = router._router_gemm
    try:
        if "overlap_off" in features:
            os.environ[OVERLAP_ENV] = "0"
        if "speculate" in features:
            os.environ[SPECULATE_ENV] = "1"
        if "router_fp32_cache" in features:
            router._router_gemm = _patched_router_gemm("fp32_cache")
        yield
    finally:
        for env_name, saved in ((OVERLAP_ENV, saved_overlap), (SPECULATE_ENV, saved_speculate)):
            if saved is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = saved
        router._router_gemm = saved_router


def measure_cell(
    model_dir: str,
    features: tuple[str, ...],
    prompts: list[str],
    args,
) -> tuple[Row, list[str]]:
    """Build one engine with ``features`` applied and measure it.

    Warmed up on a prompt outside the measured set: warming on the set itself
    would leave its blocks in a prefix cache and credit the measured run with a
    hit rate the workload never earned.
    """
    from rapid_llm import ContinuousBatchingEngine, SamplingParams

    kwargs = dict(BASELINE)
    for name in features:
        kwargs.update(FEATURES[name])

    label = "+".join(features) if features else "baseline"
    with _side_effects(features):
        engine = ContinuousBatchingEngine.from_pretrained(
            model=model_dir,
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            tensor_parallel_size=args.tp,
            max_gpu_num_blocks=args.kv_blocks,
            **kwargs,
        )
        try:
            run_requests(
                engine, ["Warm up the kernels."], SamplingParams(max_gen_len=4, temperature=0.0)
            )
            run = run_requests(
                engine, prompts, sampling_params(args.max_gen_len, greedy=args.greedy)
            )
            result = run.result(len(prompts))
            row = Row(
                label=label,
                features=features,
                ttft_ms=result.ttft_ms,
                tpot_ms=result.tpot_ms,
                tps=result.tps,
                gen_tokens=result.gen_tokens,
                latency_s=result.total_s,
                gpus=args.tp,
            )
            return row, run.texts
        finally:
            engine.shutdown()


def parse_cells(args) -> list[tuple[str, ...]]:
    """The cells to run: baseline first, then singles, then the combinations."""
    cells: list[tuple[str, ...]] = [()]
    if args.features:
        cells += [(name,) for name in args.features]
    elif args.mode in ("single", "all"):
        cells += [(name,) for name in FEATURES]
    if args.combos:
        cells += [tuple(spec.split("+")) for spec in args.combos]
    elif args.mode in ("combos", "all"):
        cells += list(DEFAULT_COMBOS)

    unknown = [name for cell in cells for name in cell if name not in FEATURES]
    if unknown:
        raise SystemExit(f"unknown feature(s) {unknown}; choose from {sorted(FEATURES)}")
    # De-duplicated, order kept: the baseline must stay first for the ratios.
    return list(dict.fromkeys(cells))


def print_matrix(rows: list[Row]) -> None:
    baseline = rows[0]
    print_row_table(
        ["cell", "TTFT (ms)", "TPOT (ms)", "TPS", "TPS/GPU", "tokens", "vs baseline"],
        [42, 11, 11, 10, 10, 8, 22],
        [
            [
                row.label,
                f"{row.ttft_ms:.1f}",
                f"{row.tpot_ms:.2f}",
                f"{row.tps:.1f}",
                f"{row.tps_per_gpu:.1f}",
                str(row.gen_tokens),
                "—" if row is baseline else _ratios(row, baseline),
            ]
            for row in rows
        ],
    )


def _ratios(row: Row, baseline: Row) -> str:
    """The cell's three ratios against the baseline, each labelled with its own name.

    Above 1 is faster for throughput and slower for the two latencies, so the
    latency ratios are inverted to read the same way as the throughput one.
    """
    parts = []
    if baseline.ttft_ms:
        parts.append(f"TTFT {baseline.ttft_ms / row.ttft_ms:.2f}x")
    if baseline.tpot_ms:
        parts.append(f"TPOT {baseline.tpot_ms / row.tpot_ms:.2f}x")
    if baseline.tps:
        parts.append(f"TPS {row.tps / baseline.tps:.2f}x")
    return " ".join(parts)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["single", "combos", "all"],
        default="single",
        help="single: baseline plus one feature per cell; combos: baseline plus the pairs; all: both",
    )
    parser.add_argument("--features", nargs="+", choices=sorted(FEATURES), default=None)
    parser.add_argument(
        "--combos",
        nargs="+",
        default=None,
        help="Explicit combinations as 'a+b[+c]'; overrides the default pair list",
    )
    parser.add_argument(
        "--workload",
        choices=["short", "long", "shared", "repetitive"],
        default="short",
        help="short: shared prompt list; long: past any token budget; shared: one long prefix; repetitive: repeated n-grams for speculation",
    )
    parser.add_argument("--batch", type=int, default=8, help="Request count")
    parser.add_argument("--max-gen-len", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--kv-blocks", type=int, default=40960)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--shared-sentences", type=int, default=32)
    parser.add_argument("--greedy", action="store_true", help="temperature=0; required for --verify")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Assert every cell's greedy text matches the baseline's (needs --greedy)",
    )
    parser.add_argument("--json", default=None)


def run(args: argparse.Namespace) -> int:
    if args.verify and not args.greedy:
        raise SystemExit("--verify compares generated text, which needs --greedy")

    prompts = build_prompts(args.workload, args.batch, args.shared_sentences)
    cells = parse_cells(args)

    rows: list[Row] = []
    texts: dict[str, list[str]] = {}
    for cell in cells:
        row, produced = measure_cell(args.model_dir, cell, prompts, args)
        rows.append(row)
        texts[row.label] = produced
        print(row.as_dict(), flush=True)

    print()
    print_matrix(rows)

    rc = 0
    if args.verify:
        baseline_label = rows[0].label

        # A cell carrying an output-shifting feature is allowed to differ; one
        # that does not is a regression, and the two must not read the same.
        def shifts(row: Row) -> bool:
            return bool(OUTPUT_SHIFTING_FEATURES.intersection(row.features))

        lossless = [row for row in rows[1:] if not shifts(row)]
        lossy = [row for row in rows[1:] if shifts(row)]
        for row in lossless:
            if texts[row.label] != texts[baseline_label]:
                print(f"\nERROR: cell {row.label!r} changed the greedy output")
                rc = 1
        if rc == 0:
            print(f"\nverify: all {len(lossless)} exact cells reproduced the baseline's text")
        for row in lossy:
            matched = texts[row.label] == texts[baseline_label]
            print(
                f"note: {row.label!r} may shift output by design; its greedy text "
                f"{'matched' if matched else 'differed from'} the baseline"
            )

    if args.json:
        write_json_log(
            args.json,
            {
                **vars(args),
                "gpu": gpu_tag(),
                "baseline": BASELINE,
                "features": FEATURES,
                "cells": ["+".join(cell) or "baseline" for cell in cells],
            },
            [row.as_dict() for row in rows],
        )
    return rc
