"""Offline throughput benchmark — sglang ``offline_throughput`` mirror, three arms.

Datasets (random-ids / random / custom / sharegpt) drive three engines through
one measurement path — ``rapid_llm`` (per-request TTFT/TPOT percentiles from
the engine's monotonic timestamps), ``transformers`` (HF ``generate``, batch
TTFT) and ``vllm`` (per-request percentiles) — which is what makes their
numbers comparable.

Orchestration only: the engines live once in
:mod:`rapid_llm.benchmark.backends` behind ``build_arm`` / ``measure_rows``;
each arm imports its engine lazily, so ``--engine vllm`` runs under vLLM's own
venv with ``PYTHONPATH=<repo root>``.

Usage:
    .venv/bin/python -m rapid_llm.benchmark.offline_throughput \
        --model my_weight/Qwen2.5-0.5B-Instruct --engine all \
        --random-input-len 4096 --random-output-len 128 --num-prompts 8 \
        --max-seq-len 8192 --iters 2 --log-dir docs/benchmark_logs

Output-length semantics differ by arm (recorded in the JSON): rapid_llm has
no ignore_eos (upper bound); vLLM defaults to ``ignore_eos=True``; HF locks
greedy with ``min_new_tokens`` — all three count tokens actually produced.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .backends import ARMS, build_arm
from .datasets import add_dataset_args, get_dataset
from .utils import get_tokenizer, gpu_tag, kv_args, timestamped_log_path, write_json_log


def measure_arm(name: str, rows, args) -> dict:
    """Build one arm, measure ``rows`` on it, tear it down; return the JSON row.

    The warm-up is narrowed to the first row: with prefix caching on (this
    runner's default, for parity with vLLM's), warming up on the whole set would
    write the measured prompts into the cache and every row would then report a
    hit it did not earn.
    """
    backend = build_arm(name, args)
    try:
        result = backend.measure_rows(
            rows, greedy=args.greedy, iters=args.iters, warmup_rows=rows[:1]
        )
        return {
            **result.as_dict(),
            "input_tokens": sum(row.prompt_len for row in rows),
            **backend.details(),
        }
    finally:
        backend.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="Checkpoint directory or repo id")
    parser.add_argument(
        "--engine",
        default="all",
        choices=["all", *ARMS],
        help="Which arm(s) to measure; 'all' runs every arm in one process",
    )
    parser.add_argument("--iters", type=int, default=2, help="Timed rounds (median reported)")
    parser.add_argument(
        "--greedy", action=argparse.BooleanOptionalAction, default=True,
        help="Greedy decoding (the benchmark default; off = rapid_llm's SAMPLE_KW)",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--data-parallel-size", type=int, default=1,
        help="Rapid_llm arm only: run the batch through N in-process replicas "
        "(vLLM-style inline sharding; no framework-level dp module)",
    )
    parser.add_argument("--max-seq-len", type=int, default=0,
                        help="Context window; 0 = engine default / config")
    parser.add_argument("--max-num-seqs", type=int, default=0,
                        help="Concurrency ceiling; 0 = engine default")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="vLLM gpu_memory_utilization")
    parser.add_argument(
        "--engine-arg", action="append", default=[],
        help="rapid_llm engine kwarg, key=value (repeatable; feature A/B goes here)",
    )
    parser.add_argument(
        "--vllm-arg", action="append", default=[],
        help="vLLM LLM() kwarg, key=value (repeatable)",
    )
    parser.add_argument("--disable-ignore-eos", action="store_true",
                        help="vLLM arm: let requests stop at EOS (default: run to length)")
    parser.add_argument("--log-dir", default=None, help="Write a JSON log here")
    parser.add_argument(
        "--json-out", default=None,
        help="Write the JSON log to this exact path (suite orchestration mode)",
    )
    parser.add_argument("--tag", default="bench_offline", help="JSON filename prefix")
    add_dataset_args(parser)
    return parser


def _fmt_pct(row, p50_key, p99_key, mean_key, width, decimals) -> str:
    """p50/p99 when the arm has per-request timing, else the batch-level mean."""
    if row.get(p50_key):
        return f"{row[p50_key]:>{width}.{decimals}f}/{row[p99_key]:<{width}.{decimals}f}"
    return f"{row.get(mean_key, 0):>{width}.1f}/{'—':>{width}}"


def print_summary(results: dict) -> None:
    """The comparison table: one line per arm, in :data:`ARMS` order."""
    print(f"\n{'─' * 78}")
    print(f"{'engine':<14}{'TTFT p50/p99 ms':>22}{'TPOT p50/p99 ms':>20}"
          f"{'TPS tok/s':>12}{'tokens':>10}")
    print(f"{'─' * 78}")
    for name, row in results.items():
        if "error" in row or "skipped" in row:
            print(f"{name:<14}{row.get('error') or row.get('skipped')}")
            continue
        if "data_parallel_size" in row:  # DP reports throughput, not latency
            print(f"{name:<14}{'DP throughput basis':>22}{'—':>20}"
                  f"{row['tps']:>12.1f}{row['gen_tokens']:>10}")
            continue
        print(f"{name:<14}"
              f"{_fmt_pct(row, 'ttft_p50_ms', 'ttft_p99_ms', 'ttft_ms', 10, 1):>22}"
              f"{_fmt_pct(row, 'tpot_p50_ms', 'tpot_p99_ms', 'tpot_ms', 9, 2):>20}"
              f"{row['tps']:>12.1f}{row['gen_tokens']:>10}")
    if any(row.get("basis") for row in results.values()):
        print("(* = batch-level basis: this row has no per-request latency distribution)")
    print(f"{'─' * 78}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.engine_arg = kv_args(args.engine_arg)
    args.vllm_arg = kv_args(args.vllm_arg)

    tokenizer = get_tokenizer(args.model)
    rows = get_dataset(args, tokenizer)
    if not rows:
        print("dataset produced no rows", file=sys.stderr)
        return 1

    engines = list(ARMS) if args.engine == "all" else [args.engine]
    results: dict = {}
    for name in engines:
        if args.data_parallel_size > 1 and name != "rapid_llm":
            results[name] = {"skipped": "DP arm is rapid_llm-only in this runner"}
            continue
        print(f"\n=== {name} ===", flush=True)
        try:
            results[name] = measure_arm(name, rows, args)
        except Exception as exc:  # one failed arm must not sink the comparison
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            traceback.print_exc()

    print_summary(results)

    config = {
        "model": args.model,
        "engine": args.engine,
        "iters": args.iters,
        "greedy": args.greedy,
        "tensor_parallel_size": args.tensor_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "max_seq_len": args.max_seq_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_mem_util": args.gpu_mem_util,
        "engine_args": args.engine_arg,
        "vllm_args": args.vllm_arg,
        "disable_ignore_eos": args.disable_ignore_eos,
        "num_prompts": args.num_prompts,
        "dataset_name": args.dataset_name,
        "random_input_len": args.random_input_len,
        "random_output_len": args.random_output_len,
        "random_range_ratio": args.random_range_ratio,
        "seed": args.seed,
        "gpu": gpu_tag(),
    }
    if args.json_out:
        write_json_log(Path(args.json_out), config, results)
    if args.log_dir:
        write_json_log(
            timestamped_log_path(args.log_dir, f"{args.tag}_{gpu_tag()}"), config, results
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
