"""The three engine-level overlap primitives, one CLI.

    python -m benchmarks.overlap.levels --level l1 --timeline
    python -m benchmarks.overlap.levels --level l2 --timeline
    python -m benchmarks.overlap.levels --level l3 --timeline

* **L1** — pinned-copy overlap: the next pass's input upload leaves on a copy
  stream while the current forward runs. On by default, so the A/B here turns
  it *off* for the reference arm.
* **L2** — two-batch overlap: a TP decode step splits into two halves that
  ping-pong, so half A's all-reduce is on the wire while half B computes. Both
  arms run eager, and a graphed reference arm closes the table: the eager TPOTs
  sit on the Python launch floor, which is what makes TBO a net loss at these
  batch sizes, and the reference shows the TPOT the same load gets from replay.
* **L3** — chunked all-reduce: a row-parallel GEMM's output is split by rows and
  each chunk's reduction goes on the wire as soon as its GEMM lands. Prefill is
  where it earns, so the workload stretches the prompts.

L4 (tile-signaling) is a kernel-level primitive with no engine switch; it lives
in ``benchmarks/kernels/bench_tile_signal.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmarks.overlap.arms import (
    TIMELINE_ENV,
    Arm,
    compare,
    l1_switch,
    l3_switch,
    make_arm,
    run_arms,
    tbo_switch,
    timeline_overlap,
    workload,
)
from rapid_llm.benchmark import make_backend, require_gpus, timestamped_log_path, write_json_log

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"

#: L1 runs graphed on one GPU through the continuous engine (the copy-stream
#: overlap and its timeline live in the worker, not in ``TextGenerator``);
#: L2/L3 need TP=2 to have an all-reduce to hide.
ENGINE_L1 = {"continuous": True, "use_cuda_graph": True, "max_seq_len": 2048,
             "max_num_seqs": 16}
ENGINE_TP2 = {"tensor_parallel_size": 2, "use_cuda_graph": False, "max_seq_len": 2048}


def engine_timeline(model_dir: str, arm: Arm) -> str:
    """One round with the engine's own timeline on: the copy vs compute regions."""
    os.environ[TIMELINE_ENV] = "1"
    backend = make_backend(model_dir, **arm.engine)
    try:
        backend.measure(workload(arm), 8, greedy=True)
        return backend.timeline_summary()
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)


def bench_l1(args) -> dict:
    """Copy-stream upload on/off, with the region table as evidence."""
    engine = {**ENGINE_L1, "max_num_batched_tokens": args.max_num_batched_tokens}
    arms = [
        make_arm(
            "overlap_off",
            l1_switch(False),
            engine=engine,
            batch=args.batch,
            stretch=True,
            gen_len=args.gen,
        ),
        make_arm(
            "overlap_on",
            l1_switch(True),
            engine=engine,
            batch=args.batch,
            stretch=True,
            gen_len=args.gen,
        ),
    ]
    print(f"\n=== L1 {CKPT} batch={args.batch} gen={args.gen} (graphed, one GPU) ===")
    rows = run_arms(CKPT, arms, repeat=args.repeat)
    summary = compare(rows, "overlap_off", ["overlap_on"], metric="total_s", tp=1)

    evidence = ""
    if args.timeline:
        print("\n=== timeline: copy-stream vs compute-stream regions ===")
        evidence = engine_timeline(CKPT, arms[1])
        print(evidence)
    return {
        "metrics": summary,
        "timeline": evidence,
        "note": (
            "L1 stages the next pass's input upload on a copy stream so it lands "
            "inside the current forward instead of serialising behind it; the "
            "timeline is the engine's own CUDA-event region table."
        ),
    }


def bench_l2(args) -> dict:
    """Decode two-batch overlap on/off, plus the graphed reference."""
    engine = {**ENGINE_TP2, "max_num_seqs": 64}
    arms = [
        make_arm("eager_off", tbo_switch(False), engine=engine, batch=args.batch, gen_len=args.gen),
        make_arm(
            "eager_on",
            tbo_switch(True, min_rows=args.min_rows),
            engine=engine,
            batch=args.batch,
            gen_len=args.gen,
        ),
        make_arm(
            "graph_reference",
            tbo_switch(False),
            engine={**engine, "use_cuda_graph": True},
            batch=args.batch,
            gen_len=args.gen,
        ),
    ]
    print(f"\n=== L2 {CKPT} TP=2 batch={args.batch} gen={args.gen} ===")
    rows = run_arms(CKPT, arms, repeat=args.repeat)
    summary = compare(rows, "eager_off", ["eager_on", "graph_reference"])

    evidence = ""
    if args.timeline:
        print("\n=== timeline: all-reduce vs the other half's compute ===")
        evidence = timeline_overlap(
            CKPT,
            arms[1],
            left=lambda r: r.stream == "compute" and r.name.endswith(".b"),
            right=lambda r: r.stream == "comm",
        )
        print(evidence)
    return {
        "metrics": summary,
        "timeline": evidence,
        "note": (
            "min_rows is forced to the bench batch so the interleave actually "
            "runs; the policy's default ridge would gate TBO off here and the "
            "table would compare off against off. The eager arms sit on the "
            "Python launch floor, which the graphed reference escapes — read "
            "the TBO delta against that floor, not against the GPU."
        ),
    }


def bench_l3(args) -> dict:
    """Chunked all-reduce on/off on a prefill-heavy workload."""
    engine = {
        **ENGINE_TP2,
        "max_num_seqs": 16,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    arms = [
        make_arm(
            "l3_off",
            l3_switch(False),
            engine=engine,
            batch=args.batch,
            stretch=True,
            gen_len=args.gen,
        ),
        make_arm(
            "l3_on",
            l3_switch(True),
            engine=engine,
            batch=args.batch,
            stretch=True,
            gen_len=args.gen,
        ),
    ]
    print(f"\n=== L3 {CKPT} TP=2 batch={args.batch} gen={args.gen} (prefill-heavy) ===")
    rows = run_arms(CKPT, arms, repeat=args.repeat)
    summary = compare(rows, "l3_off", ["l3_on"], metric="ttft_ms")

    evidence = ""
    if args.timeline:
        print("\n=== timeline: chunk k's reduce vs chunk k+1's GEMM ===")
        evidence = timeline_overlap(
            CKPT,
            arms[1],
            left=lambda r: r.name.startswith("l3.gemm"),
            right=lambda r: r.stream == "comm" and r.name.startswith("l3.all_reduce"),
        )
        print(evidence)
    return {
        "metrics": summary,
        "timeline": evidence,
        "note": (
            "L3 splits a row-parallel GEMM by rows and puts each chunk's "
            "all-reduce on the comm stream as soon as its GEMM lands; the "
            "metric is TTFT because prefill rows are what clear the 256-row "
            "chunk floor."
        ),
    }


BENCHES = {"l1": bench_l1, "l2": bench_l2, "l3": bench_l3}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", choices=sorted(BENCHES), default="l2")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2, help="runs per arm, best kept")
    ap.add_argument("--min-rows", type=int, default=8, help="L2 activation floor override")
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--timeline", action="store_true", help="also collect overlap evidence")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    require_gpus(2 if args.level != "l1" else 1)
    if args.json is None:
        args.json = str(
            timestamped_log_path(
                Path(__file__).resolve().parents[2] / "docs" / "benchmark_logs",
                f"overlap_{args.level}",
            )
        )

    write_json_log(args.json, vars(args), BENCHES[args.level](args))
    print(f"\n-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
