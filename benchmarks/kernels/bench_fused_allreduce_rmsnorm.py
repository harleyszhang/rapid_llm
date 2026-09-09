"""All-reduce plus fused residual-add/RMSNorm benchmark (TP=2 required).

Compares:
- Baseline: all-reduce + skip_rmsnorm (NCCL collective + Triton kernel)
- Fused:    all-reduce + fused_add_rmsnorm (NCCL collective + fused Triton kernel)

The fused variant saves one HBM read of the residual tensor by combining the
residual-add with the RMSNorm in a single kernel pass.

The benchmark does not claim communication fusion: both arms complete the same
all-reduce, and only the elementwise stage differs.

Usage:
    torchrun --nproc_per_node=2 benchmarks/kernels/bench_fused_allreduce_rmsnorm.py [--json path]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.distributed as dist

from rapid_llm.kernels.ops.layernorm.skip_rmsnorm import fused_add_rmsnorm, skip_rmsnorm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="Output JSON path")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    configs = [
        (4, 2048),
        (16, 4096),
        (32, 4096),
        (64, 8192),
    ]

    if rank == 0:
        print(f"Fused all-reduce RMSNorm benchmark (TP={world_size}, {torch.cuda.get_device_name()})")
        print("=" * 75)
        print("Baseline: all-reduce + skip_rmsnorm")
        print("Fused:    all-reduce + fused_add_rmsnorm")
        print("=" * 75)
        print(f"{'shape':<20} {'baseline(μs)':<15} {'fused(μs)':<15} {'fused/base':<12}")
        print("-" * 90)

    results = []
    for batch_seq, hidden in configs:
        shape = (batch_seq, hidden)
        dtype = torch.float16
        device = f"cuda:{local_rank}"

        partial = torch.randn(shape, dtype=dtype, device=device)
        residual = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(hidden, dtype=dtype, device=device).abs() + 0.5
        eps = 1e-5

        # Warmup
        for _ in range(20):
            x = partial.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            _ = skip_rmsnorm(x, residual.clone(), weight, eps)
            x = partial.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            _ = fused_add_rmsnorm(x, residual.clone(), weight, eps)
        torch.cuda.synchronize()

        n_iters = 500

        # Baseline: all-reduce + skip_rmsnorm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x = partial.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            _, _ = skip_rmsnorm(x, residual.clone(), weight, eps)
        torch.cuda.synchronize()
        t_baseline = (time.perf_counter() - t0) / n_iters * 1e6

        # Fused: all-reduce + fused_add_rmsnorm
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            x = partial.clone()
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            _, _ = fused_add_rmsnorm(x, residual.clone(), weight, eps)
        torch.cuda.synchronize()
        t_fused = (time.perf_counter() - t0) / n_iters * 1e6

        fused_speedup = t_baseline / t_fused

        if rank == 0:
            results.append({
                "shape": f"({batch_seq}, {hidden})",
                "batch_seq": batch_seq,
                "hidden": hidden,
                "baseline_us": round(t_baseline, 3),
                "fused_us": round(t_fused, 3),
                "fused_speedup": round(fused_speedup, 3),
            })
            print(f"{shape!s:<20} {t_baseline:<15.3f} {t_fused:<15.3f} {fused_speedup:<12.3f}")

    dist.destroy_process_group()

    if rank == 0:
        print("-" * 90)
        avg_fused = sum(r["fused_speedup"] for r in results) / len(results)
        print(f"Average fused/base speedup:  {avg_fused:.3f}x")

        output = {
            "benchmark": "fused_allreduce_rmsnorm",
            "description": "all-reduce + fused_add_rmsnorm vs all-reduce + skip_rmsnorm on TP=2",
            "tp_size": world_size,
            "gpu": torch.cuda.get_device_name(),
            "results": results,
            "summary": {"avg_fused_speedup": round(avg_fused, 3)},
            "note": (
                "The fused kernel saves one residual read; on shapes where the "
                "collective dominates the step, that is inside the noise."
            ),
        }

        if args.json:
            with open(args.json, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
