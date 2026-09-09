"""One-batch benchmark — sglang ``one_batch`` mirror.

Correctness first, then a batch-size latency scan, one engine instance per
mode: greedy outputs of the CUDA-graph engine must match the eager engine's
token for token (``--verify`` — replay is bit-identical at capture, so drift
is a bug flag and a low agreement rate voids the speedup); TTFT/TPOT come
from the per-request timestamps — under chunked prefill a step-interval view
misreports the first chunk's end as TTFT.

Driven through the same :class:`~rapid_llm.benchmark.backends.EngineBackend`
the cross-engine runner uses — one measurement implementation for both A/Bs.

Usage:
    .venv/bin/python -m rapid_llm.benchmark.one_batch \
        --model my_weight/Qwen2.5-0.5B-Instruct \
        --batch-sizes 1,2,4,8 --input-len 1024 --output-len 128 --iters 2 \
        --verify --log-dir docs/benchmark_logs
"""

from __future__ import annotations

import argparse
import sys

from .backends import EngineBackend
from .datasets import add_dataset_args, get_dataset
from .utils import (
    get_tokenizer,
    gpu_tag,
    kv_args,
    print_row_table,
    report_agreement,
    timestamped_log_path,
    write_json_log,
)


def parse_batch_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError(f"expected comma-separated sizes >= 1, got {value!r}")
    return sizes


def scan_batch_sizes(backend: EngineBackend, rows, sizes: list[int], iters: int) -> dict[int, dict]:
    """Measure each batch size on one live engine; ``rows[:batch]`` is the batch.

    Warm-up is narrowed to the first row so the prefix cache does not serve the
    measured prompts a hit they did not earn.
    """
    table: dict[int, dict] = {}
    for batch in sizes:
        subset = rows[:batch]
        if len(subset) < batch:
            raise SystemExit(f"--num-prompts must cover the largest batch size (need {batch})")
        result = backend.measure_rows(subset, greedy=True, iters=iters, warmup_rows=rows[:1])
        details = backend.details()
        table[batch] = {
            **result.as_dict(),
            **{k: v for k, v in details.items() if k.startswith("steady_state")},
            "texts": backend.texts(),
        }
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[1, 2, 4, 8])
    parser.add_argument("--input-len", type=int, default=1024,
                        help="Prompt tokens per row (sets the dataset's input length)")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Generated tokens per row (sets the dataset's output length)")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument(
        "--verify", action="store_true",
        help="Require graph-engine outputs to match eager, token for token",
    )
    parser.add_argument("--max-seq-len", type=int, default=0)
    parser.add_argument(
        "--engine-arg", action="append", default=[],
        help="rapid_llm engine kwarg, key=value (repeatable)",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--tag", default="bench_one_batch")
    add_dataset_args(parser)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    engine_kwargs = kv_args(args.engine_arg)
    engine_kwargs.setdefault("enable_prefix_cache", True)
    if args.max_seq_len:
        engine_kwargs.setdefault("max_seq_len", args.max_seq_len)

    # ``--input-len`` / ``--output-len`` are this runner's spelling of the shared
    # dataset lengths: one batch shape per run, so there is nothing to spread.
    args.random_input_len = args.input_len
    args.random_output_len = args.output_len
    args.num_prompts = max(args.batch_sizes)
    rows = get_dataset(args, get_tokenizer(args.model))

    tables: dict[str, dict] = {}
    footprints: dict[str, dict] = {}
    for mode, use_graph in (("graph", True), ("eager", False)):
        print(f"\n=== {mode} ===", flush=True)
        backend = EngineBackend(args.model, use_cuda_graph=use_graph, **engine_kwargs)
        try:
            tables[mode] = scan_batch_sizes(backend, rows, args.batch_sizes, args.iters)
            footprint = backend.details().get("footprint")
            if footprint:
                footprints[mode] = footprint
        finally:
            backend.close()

    print_row_table(
        ["batch", "mode", "TTFT p50/p99 ms", "TPOT p50/p99 ms", "TPS tok/s", "tokens"],
        [7, 7, 20, 20, 12, 8],
        [
            [
                str(batch),
                mode,
                f"{row['ttft_p50_ms']:.1f}/{row['ttft_p99_ms']:.1f}",
                f"{row['tpot_p50_ms']:.2f}/{row['tpot_p99_ms']:.2f}",
                f"{row['tps']:.1f}",
                str(row["gen_tokens"]),
            ]
            for mode in ("graph", "eager")
            for batch, row in tables[mode].items()
        ],
    )

    if args.verify:
        for batch in args.batch_sizes:
            report_agreement(
                tables["eager"][batch]["texts"], [("graph", tables["graph"][batch]["texts"])]
            )

    if args.log_dir:
        results = {
            mode: {str(batch): row for batch, row in table.items()}
            | ({"footprint": footprints[mode]} if mode in footprints else {})
            for mode, table in tables.items()
        }
        config = {
            "model": args.model,
            "batch_sizes": args.batch_sizes,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "iters": args.iters,
            "max_seq_len": args.max_seq_len,
            "engine_args": engine_kwargs,
            "seed": args.seed,
            "gpu": gpu_tag(),
        }
        write_json_log(
            timestamped_log_path(args.log_dir, f"{args.tag}_{gpu_tag()}"), config, results
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
