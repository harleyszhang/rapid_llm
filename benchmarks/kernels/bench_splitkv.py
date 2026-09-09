"""O8 split-kv adaptivity: fixed ``PARTITION_SIZE=128`` vs the shape-driven policy.

``native/flash_decoding`` splits the KV history into ``PARTITION_SIZE``-wide
partitions, runs one stage-1 program per ``(batch, head, partition)``, then
combines the partials with an online-softmax stage-2. The split count used to be
a hard-coded 128 regardless of shape; O8 picks it from the decode geometry so the
stage-1 grid fills the SMs without over-splitting:

* **batch=1, long context** — ``batch * heads`` underfills the GPU, so a finer
  split (more partitions) buys parallelism the serial KV walk cannot.
* **large batch** — ``batch * heads`` already saturates the SMs, so every extra
  partition is pure stage-2 combine work and ``mid_o`` traffic; a coarser split
  (down to one partition) wins.

The combine is exact online softmax, so the output is invariant to the partition
size — this benchmark first *proves* that invariance against both the fixed arm
and the paged reference, then times the two arms over a batch x seq-len scaling
sweep. Only speed moves, never numerics.

Switch under test: ``LITE_LLAMA_SPLITKV`` = ``adaptive`` (default) | ``fixed``.

Usage:
    python benchmarks/kernels/bench_splitkv.py \
        --json docs/benchmark_logs/kernels/splitkv_<stamp>.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kv_pool import PagedPool, paged_pool
from microbench import bench, metadata, require_cuda, verify

import rapid_llm.kernels
from rapid_llm.kernels.ops.attention.flashdecoding import (
    _resolve_partition_size,
    flash_decoding,
)
from tests.reference import paged_decode_attention

#: ``(batch, seq_len)`` scaling sweep. batch=1 long context is where a finer
#: split should pay; large batch is where the fixed 128 over-splits and the
#: adaptive policy should collapse toward one partition. head geometry is the
#: Qwen3/Llama-3 decode shape (GQA 4x, 128-dim heads).
HQ, HKV, HDIM = 32, 8, 128
SWEEP: list[tuple[int, int]] = [
    (1, 512),
    (1, 2048),
    (1, 8192),
    (4, 2048),
    (4, 8192),
    (16, 1024),
    (16, 4096),
    (64, 512),
    (64, 2048),
]

_RTOP, _ATOL = 1e-2, 1e-2


def _run(pool: PagedPool, q: torch.Tensor) -> torch.Tensor:
    return flash_decoding(
        q,
        pool.k,
        pool.v,
        1.0 / math.sqrt(HDIM),
        pool.table,
        pool.req_idx,
        pool.seq_lens,
        pool.max_seq_len,
    )


def _partition_for(batch: int, seq_len: int, mode: str) -> int:
    os.environ["LITE_LLAMA_SPLITKV"] = mode
    try:
        return _resolve_partition_size(batch, HQ, seq_len, torch.device("cuda"))
    finally:
        os.environ.pop("LITE_LLAMA_SPLITKV", None)


def check_invariance() -> None:
    """Prove the split count does not move the answer before any timing.

    The adaptive arm and the fixed arm must agree with each other *and* with the
    paged reference: online-softmax combine is exact, so a partition-size change
    that moved the output would be a kernel bug, not a rounding artifact.
    """
    print("Correctness / partition-size invariance:")
    for layout in ("fragmented", "contiguous"):
        pool = paged_pool([37, 512, 2048], num_kv_heads=HKV, head_dim=HDIM, layout=layout)
        q = torch.randn(3, HQ, HDIM, device="cuda", dtype=torch.float16) * 0.3

        os.environ["LITE_LLAMA_SPLITKV"] = "fixed"
        out_fixed = _run(pool, q)
        os.environ["LITE_LLAMA_SPLITKV"] = "adaptive"
        out_adaptive = _run(pool, q)
        os.environ.pop("LITE_LLAMA_SPLITKV", None)

        ref = paged_decode_attention(
            q, pool.k, pool.v, pool.table, pool.seq_lens, 1.0 / math.sqrt(HDIM)
        )
        verify(f"fixed[128] vs ref [{layout}]", out_fixed, ref, rtol=_RTOP, atol=_ATOL)
        verify(f"adaptive vs ref   [{layout}]", out_adaptive, ref, rtol=_RTOP, atol=_ATOL)
        verify("adaptive vs fixed        ", out_adaptive, out_fixed, rtol=_RTOP, atol=_ATOL)


def main() -> int:
    require_cuda()
    torch.set_grad_enabled(False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="evidence JSON path")
    args = ap.parse_args()

    print(metadata())
    print()
    check_invariance()
    print()

    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    header = (
        f"{'batch':>5} {'seq':>6} | {'fixed_us':>9} {'fix_part':>8} {'fix_grid':>9} | "
        f"{'adapt_us':>9} {'adt_part':>8} {'adt_grid':>9} | {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for batch, seq_len in SWEEP:
        lens = [seq_len] * batch
        pool = paged_pool(lens, num_kv_heads=HKV, head_dim=HDIM, layout="fragmented")
        q = torch.randn(batch, HQ, HDIM, device="cuda", dtype=torch.float16) * 0.3

        fix_part = _partition_for(batch, seq_len, "fixed")
        adt_part = _partition_for(batch, seq_len, "adaptive")
        fix_parts = math.ceil(seq_len / fix_part)
        adt_parts = math.ceil(seq_len / adt_part)

        os.environ["LITE_LLAMA_SPLITKV"] = "fixed"
        fix_us = bench(lambda p=pool, q=q: _run(p, q))
        os.environ["LITE_LLAMA_SPLITKV"] = "adaptive"
        adt_us = bench(lambda p=pool, q=q: _run(p, q))
        os.environ.pop("LITE_LLAMA_SPLITKV", None)

        speedup = fix_us / adt_us if adt_us else float("nan")
        print(
            f"{batch:>5} {seq_len:>6} | {fix_us:>9.1f} {fix_part:>8} "
            f"{batch * HQ * fix_parts:>9} | {adt_us:>9.1f} {adt_part:>8} "
            f"{batch * HQ * adt_parts:>9} | {speedup:>7.3f}x"
        )
        rows.append(
            {
                "batch": batch,
                "seq_len": seq_len,
                "num_q_heads": HQ,
                "num_kv_heads": HKV,
                "head_dim": HDIM,
                "fixed_partition_size": fix_part,
                "fixed_stage1_blocks": batch * HQ * fix_parts,
                "fixed_us": round(fix_us, 2),
                "adaptive_partition_size": adt_part,
                "adaptive_num_partitions": adt_parts,
                "adaptive_stage1_blocks": batch * HQ * adt_parts,
                "adaptive_us": round(adt_us, 2),
                "speedup": round(speedup, 4),
            }
        )
        del pool, q
        torch.cuda.empty_cache()

    geo = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
    best = max(rows, key=lambda r: r["speedup"])
    worst = min(rows, key=lambda r: r["speedup"])
    print(
        f"\nnum_sms={num_sms}  geomean speedup={geo:.3f}x  "
        f"best b{best['batch']}_s{best['seq_len']}={best['speedup']:.3f}x  "
        f"worst b{worst['batch']}_s{worst['seq_len']}={worst['speedup']:.3f}x"
    )
    print(
        "Read as: speedup > 1 means the adaptive split is faster than fixed-128.\n"
        "The win is batch=1 short/medium context, where a fixed 128 leaves the\n"
        "stage-1 grid below one wave (num_sms*16 = 1152 blocks on A10) and the\n"
        "policy splits finer to fill it. Every other shape already fills the GPU\n"
        "at 128, so the policy returns the baseline and the cell is exactly 1.0x\n"
        "(coarsening an overfilled grid was measured and dropped: sign flipped\n"
        "with shape, magnitude inside noise). mid_o partials are MB-scale and\n"
        "consumed by the merge immediately (L2-resident), so the win is grid\n"
        "occupancy, not a DRAM round-trip avoided."
    )

    if args.json:
        from rapid_llm.benchmark import write_json_log

        write_json_log(
            args.json,
            {
                "optimization": "O8 adaptive split-kv decode attention",
                "switch": "LITE_LLAMA_SPLITKV=adaptive|fixed",
                "kernel": "native/flash_decoding",
                "inference_mode": "offline kernel microbenchmark (no serving queue)",
                "command": " ".join(sys.argv),
                "num_sms": num_sms,
                "head_geometry": {"num_q_heads": HQ, "num_kv_heads": HKV, "head_dim": HDIM},
                "note": "output invariant to partition size (exact online-softmax combine)",
            },
            {"geomean_speedup": round(geo, 4), "rows": rows},
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
