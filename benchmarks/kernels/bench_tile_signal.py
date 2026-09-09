"""L4 tile-signaling: pipelined vs serial producer/consumer kernels, one GPU.

Three arms per shape: the same two persistent kernels run serially on one
stream (the control), the same pair overlapped on two streams through the
tile flags (the treatment), and a plain-torch ``matmul + silu*mul`` baseline
(what an engine without the primitive does today). The first two arms share
kernels, grids and block parameters, so their delta isolates the execution
strategy exactly; the torch arm is an absolute reference and is not expected
to match bit-for-bit (different GEMM code entirely).

PCIe note: L4 is intra-device kernel overlap — it does not touch the
interconnect, so these numbers say nothing about NVLink.

Usage:
    python benchmarks/kernels/bench_tile_signal.py --timeline
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.batch_overlap.overlap import Timeline
from rapid_llm.benchmark import require_gpus, timestamped_log_path, write_json_log
from rapid_llm.kernels.tile_signal import (
    TileSignalBuffer,
    pipelined_gemm_swiglu,
    serial_gemm_swiglu,
)

#: (M rows, N activation width, K hidden) — MLP-shaped problems. The 4480/1536
#: pair is a Qwen2.5-1.5B TP2 MLP; the rest scale M from decode to prefill.
SHAPES = [
    (64, 4480, 1536),
    (256, 4480, 1536),
    (1024, 2048, 1024),
    (2048, 4480, 1536),
    (4096, 4480, 1536),
]

_WARMUP = 3
_RUNS = 10


def _problem(m, n, k):
    """bf16 MLP step inputs, scaled like real activations/weights."""
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
    gate_w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda") * 0.05
    up_w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda") * 0.05
    return a, gate_w, up_w


def _time(fn, a, gate_w, up_w, buffer) -> float:
    """Median device ms of one arm over ``_RUNS`` timed calls."""
    for _ in range(_WARMUP):
        fn(a, gate_w, up_w, buffer)
    torch.cuda.synchronize()

    times = []
    for _ in range(_RUNS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(a, gate_w, up_w, buffer)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), out


def _torch_arm(a, gate_w, up_w, _buffer=None):
    """What engines do without the primitive: cublas GEMMs + epilogue."""
    return torch.nn.functional.silu(a @ gate_w) * (a @ up_w)


def _overlap_evidence(a, gate_w, up_w, buffer) -> dict:
    """One instrumented pipelined round; overlap straight from the timeline."""
    timeline = Timeline(enabled=True, device="cuda")
    for _ in range(_WARMUP):
        pipelined_gemm_swiglu(a, gate_w, up_w, buffer)
    torch.cuda.synchronize()

    for _ in range(3):
        pipelined_gemm_swiglu(a, gate_w, up_w, buffer, timeline=timeline)
    torch.cuda.synchronize()

    records = timeline.collect()
    producers = [r for r in records if r.stream == "producer"]
    consumers = [r for r in records if r.stream == "consumer"]
    overlap_ms = 0.0
    pairs = 0
    for p in producers:
        for c in consumers:
            span = min(p.end_ms, c.end_ms) - max(p.start_ms, c.start_ms)
            if span > 0:
                overlap_ms += span
                pairs += 1
    return {
        "producer_regions": len(producers),
        "consumer_regions": len(consumers),
        "overlapping_pairs": pairs,
        "overlap_ms": round(overlap_ms, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--shapes",
        type=int,
        nargs="+",
        default=None,
        help="indices into the SHAPES table (default: all)",
    )
    parser.add_argument("--timeline", action="store_true", help="collect overlap evidence")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="default: docs/benchmark_logs/overlap/overlap_l4_<stamp>.json",
    )
    args = parser.parse_args()
    if args.json is None:
        args.json = timestamped_log_path(
            Path(__file__).resolve().parents[2] / "docs" / "benchmark_logs" / "overlap", "overlap_l4"
        )

    require_gpus(1)
    shapes = SHAPES if args.shapes is None else [SHAPES[i] for i in args.shapes]

    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"device: {torch.cuda.get_device_name(0)}, {sm} SMs")

    results = {}
    for m, n, k in shapes:
        a, gate_w, up_w = _problem(m, n, k)
        buffer = TileSignalBuffer.for_problem(m, n, 64, 64)

        serial_ms, serial_out = _time(serial_gemm_swiglu, a, gate_w, up_w, buffer)
        piped_ms, piped_out = _time(pipelined_gemm_swiglu, a, gate_w, up_w, buffer)
        torch_ms, _ = _time(_torch_arm, a, gate_w, up_w, buffer)
        assert buffer.dropped_tiles() == 0

        agree = torch.equal(serial_out, piped_out)
        delta = (serial_ms - piped_ms) / serial_ms
        print(f"\nM={m} N={n} K={k}  (tiles {buffer.num_tiles})")
        print(f"  serial    {_B}{serial_ms:8.3f} ms")
        print(f"  pipelined {_B}{piped_ms:8.3f} ms   -> {delta:+.1%}")
        print(f"  torch     {_B}{torch_ms:8.3f} ms   (cublas reference)")
        print(f"  serial==pipelined bitwise: {agree}")

        entry = {
            "serial_ms": round(serial_ms, 3),
            "pipelined_ms": round(piped_ms, 3),
            "torch_ms": round(torch_ms, 3),
            "speedup_pct": round(delta * 100, 2),
            "bitwise_agree": agree,
        }
        if args.timeline:
            entry["overlap"] = _overlap_evidence(a, gate_w, up_w, buffer)
            print(f"  overlap: {entry['overlap']}")
        results[f"{m}x{n}x{k}"] = entry

    if args.json:
        write_json_log(args.json, {"shapes": [list(s) for s in shapes], "sm": sm}, results)
    return 0


_B = ""  # alignment spacer for the table


if __name__ == "__main__":
    sys.exit(main())
