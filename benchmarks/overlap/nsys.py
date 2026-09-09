"""Kernel-level overlap evidence: the nsys payload and its trace analyser.

Two modes in one file:

    # 1. the traced workload, run twice under nsys (overlap off, then on)
    nsys profile --trace=cuda -o off python -m benchmarks.overlap.nsys payload
    nsys profile --trace=cuda -o on  python -m benchmarks.overlap.nsys payload --overlap

    # 2. export the kernel traces and analyse them
    nsys stats -r cuda_gpu_trace --format csv --output off off.nsys-rep
    nsys stats -r cuda_gpu_trace --format csv --output on  on.nsys-rep
    python -m benchmarks.overlap.nsys report \
        --off-csv off_cuda_gpu_trace.csv --on-csv on_cuda_gpu_trace.csv

The question the report answers from kernel-level evidence: did the NCCL
all-reduce kernels run concurrently with compute kernels, and how much of the
reduction time was hidden? The engine's own timeline says the overlap was
scheduled; only a kernel trace says it actually ran that way.

The analysis is interval arithmetic on (start, duration) pairs per stream: a
reduction on the comm stream overlaps compute when its interval intersects any
compute-stream kernel's interval. Two aggregates per trace:

* ``overlap_ms`` — sum over NCCL kernels of the intersection time,
* ``exposed_ms`` — sum of NCCL time with no concurrent compute, the serial
  remainder the overlap could not hide.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REPO = Path(__file__).resolve().parents[2]

MODEL = "my_weight/Qwen2.5-1.5B-Instruct"
BATCH = 16
STEPS = 48


# --------------------------------------------------------------------------- #
# payload: the traced workload
# --------------------------------------------------------------------------- #
def run_payload(overlap: bool) -> int:
    """One fixed TP=2 decode workload; the switches are the only difference.

    Two passes: the first warms allocator blocks, JIT paths and the NCCL
    communicator, so the traced second pass holds steady-state steps.
    """
    from benchmarks.overlap.arms import l1_switch, l3_switch, make_arm, tbo_switch
    from rapid_llm.benchmark import PROMPTS, expand_prompts, make_backend

    arm = make_arm(
        "payload",
        l1_switch(overlap),
        tbo_switch(overlap),
        l3_switch(overlap),
        engine={
            "tensor_parallel_size": 2,
            "use_cuda_graph": False,
            "max_seq_len": 2048,
            "max_num_seqs": 32,
        },
        batch=BATCH,
        gen_len=STEPS,
    )
    import os

    for key, value in arm.env.items():
        os.environ[key] = value
    if arm.reset is not None:
        arm.reset()

    backend = make_backend(MODEL, **arm.engine)
    try:
        prompts = [p[:64] for p in expand_prompts(PROMPTS, BATCH)]
        for _ in range(2):
            backend.measure(prompts, STEPS, greedy=True)
    finally:
        backend.close()
    print(f"payload done (overlap={overlap})")
    return 0


# --------------------------------------------------------------------------- #
# report: interval arithmetic over the exported kernel traces
# --------------------------------------------------------------------------- #
def load_trace(path: Path) -> dict[str, tuple[list, list]]:
    """Split the kernel trace per GPU into (comm, compute) interval lists.

    Returns ``{device: (comm, compute)}`` where each list holds
    ``(start_ns, end_ns, name)`` intervals, sorted by start. NCCL kernels are
    the comm side; the compute side is every kernel that is neither NCCL nor a
    memcpy — copies ride the copy engines and overlap a reduction for free,
    which is not the compute↔communication overlap under test. Both ranks run
    inside one trace, so grouping by the ``Device`` column keeps rank 0's
    (cuda:0) and rank 1's (cuda:1) timelines apart.
    """
    by_device: dict[str, tuple[list, list]] = {}
    with open(path, encoding="utf-8") as handle:
        # nsys prepends metadata lines before the CSV header.
        rows = [line for line in handle if not line.startswith("=")]
        for row in csv.DictReader(rows):
            try:
                start = int(row["Start (ns)"])
                duration = int(row["Duration (ns)"])
            except (KeyError, TypeError, ValueError):
                continue
            name = row.get("Name", "")
            device = row.get("Device", "") or "?"
            comm, compute = by_device.setdefault(device, ([], []))
            interval = (start, start + duration, name)
            lowered = name.lower()
            if "nccl" in lowered:
                comm.append(interval)
            elif "memcpy" not in lowered:
                compute.append(interval)
    for comm, compute in by_device.values():
        comm.sort()
        compute.sort()
    return by_device


def merged(intervals: list[tuple[int, int, str]]) -> list[list[int]]:
    """Merge sorted intervals into disjoint spans (starts/ends stay ordered)."""
    spans: list[list[int]] = []
    for start, end, _ in intervals:
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], end)
        else:
            spans.append([start, end])
    return spans


def overlap_stats(comm: list[tuple[int, int, str]], compute: list[tuple[int, int, str]]) -> dict:
    """Per-trace overlap aggregates.

    Compute kernels merge into disjoint spans first — spans never double-count
    a nanosecond, so the hidden time of one reduction is exactly the sum of its
    intersections with the spans, found by binary search.
    """
    if not comm:
        return {"reductions": 0, "comm_ms": 0.0, "overlap_ms": 0.0, "exposed_ms": 0.0}
    spans = merged(compute)
    ends = [span[1] for span in spans]
    overlap_ns = 0
    total_ns = 0
    for c_start, c_end, _ in comm:
        total_ns += c_end - c_start
        # First span whose end is beyond the reduction's start; every span from
        # here on that begins before the reduction ends intersects it.
        index = bisect.bisect_right(ends, c_start)
        hidden = 0
        while index < len(spans) and spans[index][0] < c_end:
            hidden += min(spans[index][1], c_end) - max(spans[index][0], c_start)
            index += 1
        overlap_ns += min(hidden, c_end - c_start)
    return {
        "reductions": len(comm),
        "comm_ms": total_ns / 1e6,
        "overlap_ms": overlap_ns / 1e6,
        "exposed_ms": (total_ns - overlap_ns) / 1e6,
    }


def write_report(off_csv: Path, on_csv: Path, out: Path) -> int:
    """Aggregate both traces into the markdown table the release report cites."""
    off_devices = load_trace(off_csv)
    on_devices = load_trace(on_csv)
    devices = sorted(set(off_devices) | set(on_devices))

    def pct(part: float, whole: float) -> str:
        return f"{part / whole * 100:.1f}%" if whole else "n/a"

    def trace_rows(label: str, by_device: dict) -> list[str]:
        rows = []
        for device in devices:
            stats = overlap_stats(*by_device.get(device, ([], [])))
            short = device.split("(")[-1].rstrip(")") if "(" in device else device
            rows.append(
                f"| {label} | {short} | {stats['reductions']} | {stats['comm_ms']:.2f} ms | "
                f"{stats['overlap_ms']:.2f} ms ({pct(stats['overlap_ms'], stats['comm_ms'])}) | "
                f"{stats['exposed_ms']:.2f} ms ({pct(stats['exposed_ms'], stats['comm_ms'])}) |"
            )
        return rows

    lines = [
        f"# nsys overlap evidence — TP=2 decode, {Path(MODEL).name}, batch {BATCH}, eager",
        "",
        "Two payloads identical except the overlap switches "
        "(`RAPID_LLM_OVERLAP`/`RAPID_LLM_TBO`/`RAPID_LLM_COMM_OVERLAP`), traced with "
        "`nsys profile --trace=cuda`, kernels exported via "
        "`nsys stats -r cuda_gpu_trace --format csv`, then aggregated per GPU over every "
        "NCCL kernel (both warmup and steady passes — the overlap behaviour is the same in "
        "both). Compute side excludes memcpys: a copy engine overlaps a reduction for free "
        "and is not the claim under test.",
        "",
        "| trace | gpu | NCCL kernels | NCCL total | hidden under compute | exposed (serial) |",
        "| --- | --- | --- | --- | --- | --- |",
        *trace_rows("overlap off", off_devices),
        *trace_rows("overlap on ", on_devices),
        "",
    ]
    for device in devices:
        off_stats = overlap_stats(*off_devices.get(device, ([], [])))
        on_stats = overlap_stats(*on_devices.get(device, ([], [])))
        lines.append(
            f"GPU {device}: reduction time hidden under compute goes "
            f"{pct(off_stats['overlap_ms'], off_stats['comm_ms'])} -> "
            f"{pct(on_stats['overlap_ms'], on_stats['comm_ms'])}."
        )
    lines += [
        "",
        "Interconnect: PCIe (2x A10, no NVLink hardware); the fractions are about this "
        "machine and say nothing about NVLink topologies.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    payload = sub.add_parser("payload", help="the traced workload (run under nsys)")
    payload.add_argument("--overlap", action="store_true", help="turn all three switches on")

    report = sub.add_parser("report", help="analyse two exported kernel traces")
    report.add_argument("--off-csv", required=True, help="cuda_gpu_trace CSV, overlap off")
    report.add_argument("--on-csv", required=True, help="cuda_gpu_trace CSV, overlap on")
    report.add_argument(
        "--out", default=str(REPO / "docs" / "benchmark_logs" / "nsys_overlap_report.md")
    )

    args = parser.parse_args()
    if args.mode == "payload":
        return run_payload(args.overlap)
    return write_report(Path(args.off_csv), Path(args.on_csv), Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
