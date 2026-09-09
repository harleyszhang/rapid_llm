"""Record the v0.11.5 overlap GIFs: L2 ping-pong, L3 chunked AR, L4 tile pipeline.

Three levels, three pictures, one renderer. Every bar is a CUDA-event region the
engine itself recorded (``RAPID_LLM_OVERLAP_TIMELINE=1``) on a real run — L2 and
L3 on a TP=2 Qwen2.5-1.5B decode/prefill, L4 on one GPU's producer/consumer
kernel pair — so what the frame shows is device-side concurrency, not a drawing
of what the code intends to do. The overlap is read back off the same records
and printed in the caption line, which is why the pictures cannot drift from the
benchmark logs.

What to look at in each:

* ``l2`` — three lanes. Half A's all-reduce sits on the comm lane *while* half
  B's segment occupies a compute lane; the red band is their intersection.
* ``l3`` — two lanes. Chunk ``k``'s reduction is on the wire while chunk
  ``k+1``'s GEMM computes; one row-parallel GEMM, split by rows.
* ``l4`` — two lanes, one GPU. The epilogue kernel starts before the GEMM
  kernel finishes, because tiles are published by flag rather than by a
  stream-wide barrier.

Usage:
    python scripts/gen_overlap_gifs.py                      # all three levels
    python scripts/gen_overlap_gifs.py --level l4           # one level
    python scripts/gen_overlap_gifs.py --model-dir my_weight/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.batch_overlap.overlap import RegionRecord

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

W = 1180
TITLE_H, PAD, LINE_H = 36, 18, 25
LANE_H, LANE_GAP = 84, 30
LABEL_W = 132
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
DIM, TEXT_FG, AXIS_FG = (128, 136, 148), (222, 226, 232), (52, 58, 68)
OVERLAP_FG = (245, 99, 72)

CKPT = "my_weight/Qwen2.5-1.5B-Instruct"
TIMELINE_ENV = "RAPID_LLM_OVERLAP_TIMELINE"


@dataclass(frozen=True)
class Lane:
    """One horizontal track: a label, a colour, and which regions belong to it."""

    label: str
    colour: tuple[int, int, int]
    match: Callable[[RegionRecord], bool]


# --------------------------------------------------------------------------- #
# Recorders: one real run per level, regions straight off the engine's timeline
# --------------------------------------------------------------------------- #


def record_l2(model_dir: str) -> list[RegionRecord]:
    """TP=2 decode with TBO on: half-A/B segments plus the deferred reductions."""
    os.environ["RAPID_LLM_TBO"] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from rapid_llm.batch_overlap.comm_overlap import CommStreamPool
    from rapid_llm.batch_overlap.two_batch_overlap import reset_tbo_policy
    from rapid_llm.benchmark import make_backend

    reset_tbo_policy()  # the policy is cached per process; this run opts in
    CommStreamPool.reset()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,  # TBO's policy excludes graphs
        max_seq_len=2048,
        max_num_seqs=32,
    )
    try:
        # Short prompts, long generation: the ping-pong only exists in decode.
        prompts = [
            "Explain the theory of relativity.",
            "Describe the history of the Roman Empire.",
            "Write a tutorial on Python decorators.",
            "Summarise the plot of Hamlet.",
            "List three sorting algorithms.",
            "Describe how a compiler works.",
            "Explain what a cache line is.",
            "Write a limerick about GPUs.",
        ]
        backend.measure(prompts, 8, greedy=True)
        return CommStreamPool.for_device("cuda").timeline.collect()
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)


def record_l3(model_dir: str) -> list[RegionRecord]:
    """TP=2 chunked prefill with L3 on: ``l3.gemm.k`` against ``l3.all_reduce.k``."""
    os.environ["RAPID_LLM_COMM_OVERLAP"] = "1"
    os.environ[TIMELINE_ENV] = "1"
    from rapid_llm.batch_overlap.comm_overlap import CommStreamPool, reset_comm_overlap_policy
    from rapid_llm.benchmark import PROMPTS, expand_prompts, make_backend

    reset_comm_overlap_policy()
    CommStreamPool.reset()
    backend = make_backend(
        model_dir,
        tensor_parallel_size=2,
        use_cuda_graph=False,
        max_seq_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=512,
    )
    try:
        # Prompts long enough that a row-parallel GEMM clears L3's row floor:
        # chunking is what the picture is about, so the GEMM has to be chunked.
        prompts = [" ".join([p] * 22) for p in expand_prompts(PROMPTS, 4)]
        backend.measure(prompts, 8, greedy=True)
        return CommStreamPool.for_device("cuda").timeline.collect()
    finally:
        backend.close()
        os.environ.pop(TIMELINE_ENV, None)


def record_l4() -> list[RegionRecord]:
    """One GPU: the tile-signaling producer/consumer pair, instrumented."""
    import torch

    from rapid_llm.batch_overlap.overlap import Timeline
    from rapid_llm.kernels.tile_signal import TileSignalBuffer, pipelined_gemm_swiglu

    # The Qwen2.5-1.5B TP2 MLP shape at a prefill-ish batch: enough tiles that
    # the consumer has real work to overlap with the producer.
    m, n, k = 2048, 4480, 1536
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda") * 0.1
    gate_w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda") * 0.05
    up_w = torch.randn(k, n, dtype=torch.bfloat16, device="cuda") * 0.05
    buffer = TileSignalBuffer.for_problem(m, n, 64, 64)

    timeline = Timeline(enabled=True, device="cuda")
    for _ in range(3):  # warm the kernels so the recorded spans are steady-state
        pipelined_gemm_swiglu(a, gate_w, up_w, buffer)
    torch.cuda.synchronize()
    for _ in range(3):
        pipelined_gemm_swiglu(a, gate_w, up_w, buffer, timeline=timeline)
    torch.cuda.synchronize()
    assert buffer.dropped_tiles() == 0, "consumer gave up on a tile"
    return timeline.collect()


# --------------------------------------------------------------------------- #
# Lane sets and captions per level
# --------------------------------------------------------------------------- #


def lanes_l2() -> list[Lane]:
    return [
        Lane(
            "compute · half A",
            (94, 193, 117),
            lambda r: r.stream == "compute" and r.name.endswith(".a"),
        ),
        Lane(
            "compute · half B",
            (86, 156, 214),
            lambda r: r.stream == "compute" and r.name.endswith(".b"),
        ),
        Lane("comm stream", (226, 184, 92), lambda r: r.stream == "comm"),
    ]


def lanes_l3() -> list[Lane]:
    return [
        Lane("compute (GEMM chunks)", (94, 193, 117), lambda r: r.stream == "compute"),
        Lane("comm stream", (226, 184, 92), lambda r: r.stream == "comm"),
    ]


def lanes_l4() -> list[Lane]:
    return [
        Lane("producer (GEMM)", (94, 193, 117), lambda r: r.stream == "producer"),
        Lane("consumer (epilogue)", (86, 156, 214), lambda r: r.stream == "consumer"),
    ]


def overlap_stats(records: list[RegionRecord]) -> tuple[int, float]:
    """Count and total the intersections between comm and compute regions."""
    comm = [r for r in records if r.stream == "comm"]
    compute = [r for r in records if r.stream in ("compute", "producer", "consumer")]
    pairs, total = 0, 0.0
    for left in comm or [r for r in records if r.stream == "producer"]:
        for right in compute:
            span = min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms)
            if span > 0:
                pairs += 1
                total += span
    return pairs, total


def caption_l2(records: list[RegionRecord]) -> str:
    pairs, total = overlap_stats(records)
    return (
        f"half A's all-reduce on the comm lane runs while half B computes: "
        f"{pairs} intersecting pairs, {total:.1f} ms of overlap in this window"
    )


def caption_l3(records: list[RegionRecord]) -> str:
    pairs, total = overlap_stats(records)
    return (
        f"chunk k's reduction is on the wire while chunk k+1's GEMM computes: "
        f"{pairs} intersecting pairs, {total:.1f} ms of overlap in this window"
    )


def caption_l4(records: list[RegionRecord]) -> str:
    pairs, total = overlap_stats(records)
    return (
        f"the epilogue kernel starts before the GEMM kernel ends — tiles flow by "
        f"flag, not by a stream barrier: {pairs} pairs, {total:.1f} ms overlapped"
    )


LEVELS: dict[str, dict] = {
    "l2": {
        "record": record_l2,
        "lanes": lanes_l2,
        "caption": caption_l2,
        "title": "rapid-llm  —  L2 two-batch overlap (decode ping-pong over a deferred all-reduce)",
        "out": "docs/images/overlap_l2.gif",
        "regions": 30,
    },
    "l3": {
        "record": record_l3,
        "lanes": lanes_l3,
        "caption": caption_l3,
        "title": "rapid-llm  —  L3 chunked all-reduce (row chunks of one row-parallel GEMM)",
        "out": "docs/images/overlap_l3.gif",
        "regions": 24,
    },
    "l4": {
        "record": record_l4,
        "lanes": lanes_l4,
        "caption": caption_l4,
        "title": "rapid-llm  —  L4 tile-signaling (GEMM producer / epilogue consumer, one GPU)",
        "out": "docs/images/overlap_l4.gif",
        "regions": 6,
    },
}


# --------------------------------------------------------------------------- #
# Windowing and rendering
# --------------------------------------------------------------------------- #


def window(records: list[RegionRecord], budget: int) -> tuple[list[RegionRecord], float, float]:
    """Clip to the first window holding ``budget`` regions, anchored on an overlap.

    The anchor is the first comm (or producer) region that intersects a compute
    region — that intersection is the frame's whole point, so the window starts
    slightly before it. The width adapts to the region density instead of being
    a fixed number of milliseconds: a decode step packs hundreds of tiny
    segments, and a fixed 30 ms window would either be unreadable or empty.
    """
    anchors = [r for r in records if r.stream in ("comm", "producer")]
    compute = [r for r in records if r.stream in ("compute", "consumer")]
    anchor = None
    for candidate in anchors:
        if any(c.start_ms < candidate.end_ms and candidate.start_ms < c.end_ms for c in compute):
            anchor = candidate.start_ms
            break
    if anchor is None and records:
        anchor = records[0].start_ms
    if anchor is None:
        return [], 0.0, 0.0

    lo = anchor - 1.0
    picked = sorted(
        (r for r in records if r.end_ms >= lo and r.start_ms >= lo - 0.5),
        key=lambda r: (r.start_ms, r.end_ms),
    )[:budget]
    if not picked:
        return [], lo, lo
    hi = max(r.end_ms for r in picked) + 0.5
    window_records = [r for r in records if r.end_ms >= lo and r.start_ms <= hi]
    return window_records, lo, hi


def render(
    records: list[RegionRecord],
    lanes: list[Lane],
    title: str,
    note: str,
    t0: float,
    scale: float,
    fonts: tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont],
) -> Image.Image:
    """One frame: every lane, the regions revealed so far, and the overlap bands."""
    bold, small = fonts
    height = TITLE_H + PAD + LINE_H + len(lanes) * (LANE_H + LANE_GAP) + PAD + LINE_H
    canvas = Image.new("RGB", (W, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text((12, 9), title, fill=TITLE_FG, font=small)
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)

    lane_y: dict[int, int] = {}
    for index, lane in enumerate(lanes):
        y = TITLE_H + PAD + LINE_H + index * (LANE_H + LANE_GAP)
        lane_y[index] = y
        draw.text((PAD, y + LANE_H // 2 - 9), lane.label, fill=TEXT_FG, font=bold)
        draw.line([LABEL_W, y + LANE_H, W - PAD, y + LANE_H], fill=AXIS_FG)

    def x_of(ms: float) -> float:
        return LABEL_W + (ms - t0) * scale

    # Overlap bands first, so the bars sit on top of them.
    for left in [r for r in records if r.stream in ("comm", "producer")]:
        for lane_index, lane in enumerate(lanes):
            if lane.match(left):
                continue
            for right in (r for r in records if lane.match(r)):
                start = max(left.start_ms, right.start_ms)
                stop = min(left.end_ms, right.end_ms)
                if stop <= start:
                    continue
                y = lane_y[lane_index] + 12
                draw.rectangle(
                    [x_of(start), y + LANE_H - 30, x_of(stop), y + LANE_H - 24], fill=OVERLAP_FG
                )

    for record in records:
        for lane_index, lane in enumerate(lanes):
            if not lane.match(record):
                continue
            y = lane_y[lane_index] + 12
            x0 = x_of(max(record.start_ms, t0))
            x1 = x_of(min(record.end_ms, t0 + (W - PAD - LABEL_W) / scale))
            draw.rectangle([x0, y, max(x1, x0 + 2), y + LANE_H - 30], fill=lane.colour)
            label = record.name.replace("tbo.", "").replace("l3.", "").replace("l4.", "")
            if x1 - x0 > 74:
                draw.text((x0 + 5, y + 6), label, fill=BG, font=small)
            break

    t_end = t0 + (W - PAD - LABEL_W) / scale
    tick = 0.0
    step = max(1.0, round((t_end - t0) / 8))
    while tick <= t_end - t0:
        x = x_of(t0 + tick)
        draw.line([x, TITLE_H + PAD // 2, x, TITLE_H + PAD // 2 + 6], fill=DIM)
        draw.text((x - 12, TITLE_H - 2), f"{tick:.0f}ms", fill=DIM, font=small)
        tick += step

    draw.text((PAD, height - PAD - LINE_H), note, fill=OVERLAP_FG, font=small)
    return canvas


def build(level: str, model_dir: str, duration: int) -> Path:
    """Record one level, window it, and write its GIF."""
    spec = LEVELS[level]
    records = spec["record"](model_dir) if level != "l4" else spec["record"]()
    if not records:
        raise SystemExit(f"{level}: no timeline regions recorded")
    lanes = spec["lanes"]()
    window_records, lo, hi = window(records, spec["regions"])
    if not window_records:
        raise SystemExit(f"{level}: no overlapping regions to show")
    # Present regions relative to the window, so a bar that started before it
    # visibly continues past the left edge instead of appearing to start at 0.
    window_records = [
        RegionRecord(r.name, r.stream, r.start_ms - lo, r.end_ms - lo) for r in window_records
    ]
    span = hi - lo
    scale = (W - PAD - LABEL_W) / max(span, 1e-3)
    note = spec["caption"](window_records)
    print(
        f"{level}: {len(records)} regions recorded, window shows {len(window_records)} over {span:.1f} ms"
    )

    fonts = (
        ImageFont.truetype(BOLD_PATH, 16),
        ImageFont.truetype(FONT_PATH, 15),
    )
    frames = [
        render(window_records[:count], lanes, spec["title"], note, 0.0, scale, fonts)
        for count in range(1, len(window_records) + 1)
    ]
    frames += [frames[-1]] * 3
    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in frames]

    out = REPO_ROOT / spec["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        out, save_all=True, append_images=palette[1:], duration=duration, loop=0, optimize=True
    )
    print(f"{level}: saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(palette)} frames)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=CKPT)
    ap.add_argument("--level", choices=sorted(LEVELS), nargs="+", default=sorted(LEVELS))
    ap.add_argument("--duration", type=int, default=700, help="ms per frame")
    args = ap.parse_args()
    for level in args.level:
        build(level, args.model_dir, args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
