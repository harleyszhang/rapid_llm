"""Record the L1 cross-stream overlap GIF: input uploads overlap the prior forward.

Drives a real :class:`~rapid_llm.engine.continuous_engine.ContinuousBatchingEngine`
with ``RAPID_LLM_OVERLAP_TIMELINE=1`` over a workload built to contain mixed
prefill/decode steps (long prompts, a small per-step token budget). Every region
rendered is a CUDA-event measurement taken from the engine's own timeline: copy
regions are the pinned-staging H2D uploads issued on the copy stream, compute
regions are the model forwards. The frame to watch is where a ``decode`` upload
falls *inside* a still-running ``prefill`` forward — that intersection is the
overlap, not a rendering trick.

Usage:
    python scripts/gen_overlap_l1_gif.py
    python scripts/gen_overlap_l1_gif.py --model-dir my_weight/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from scripts._viz_lib import (
    BG, TITLE_BG, TITLE_FG, DIM, PROMPT_FG, TEXT_FG,
    GREEN, RED, AMBER, GRID_LINE,
    TITLE_H, PAD, LINE_H,
    FontPack, draw_title_bar, save_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.batch_overlap.overlap import RegionRecord

W, H = 1180, 430
LANE_H, LANE_GAP = 92, 34

# Map _viz_lib aliases for render use
COPY_FG = AMBER
COMPUTE_FG = GREEN
OVERLAP_FG = RED
AXIS_FG = GRID_LINE

#: Left lane label column width; the lanes themselves span the rest.
LABEL_W = 110


def record(model_dir: str):
    """Run a small mixed-step workload and return the timeline regions."""
    os.environ["RAPID_LLM_OVERLAP"] = "1"
    os.environ["RAPID_LLM_OVERLAP_TIMELINE"] = "1"
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.engine.sampler import SamplingParams

    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        use_cuda_graph=True,
    )
    try:
        # Very long prompts, so one pass prefills ~2000 tokens: the forward
        # keeps the GPU busy for ~30 ms while the host (whose eager launch
        # path is ~22 ms) is already preparing the next pass — its upload
        # then lands inside the still-running forward. Shorter prompts make
        # the host the slower side and the intersection never happens.
        base = [
            "Explain the theory of relativity in detail.",
            "Describe the history of the Roman Empire.",
            "Write a tutorial on Python decorators.",
        ]
        prompts = [" ".join([p] * 180) for p in base]
        engine.generate(prompts, SamplingParams(temperature=0.0, max_gen_len=12))
        return engine._executor._worker.timeline.collect()
    finally:
        engine.shutdown()


def window_around_first_overlap(records):
    """Clip to the span around the first copy region that intersects a compute one.

    The instructive moment is a mixed step: the next pass's input upload lands
    on the copy stream while the current forward still occupies the compute
    stream. Returns ``(window, lo, hi, overlapped)`` covering
    ``[anchor - 2ms, anchor + 30ms]``, or the head of the run when no
    intersection exists (overlap disabled, say).
    """
    copies = [r for r in records if r.stream == "copy"]
    computes = [r for r in records if r.stream == "compute"]
    anchor = None
    for copy in copies:
        # A fully interior landing with headroom on both sides reads as the
        # overlap; a copy grazing a forward's last millisecond (the cold first
        # pass, whose host launch is slower than usual) does not.
        if any(
            c.start_ms < copy.start_ms - 10 and copy.end_ms < c.end_ms - 10
            for c in computes
        ):
            anchor = copy.start_ms
            break
    if anchor is None:  # fall back to any intersection, however shallow
        for copy in copies:
            if any(c.start_ms < copy.end_ms and copy.start_ms < c.end_ms for c in computes):
                anchor = copy.start_ms
                break
    if anchor is None:
        lo = min(r.start_ms for r in records[:14])
        hi = max(r.end_ms for r in records[:14])
        return records[:14], lo, hi, False
    lo, hi = anchor - 2.0, anchor + 30.0
    # Keep every region *intersecting* the window: the forward being overlapped
    # started tens of ms before the copy, so filtering on start alone would
    # drop the very region that makes the frame instructive. The renderer
    # clips it to the window.
    window = [r for r in records if r.end_ms >= lo and r.start_ms <= hi]
    return window, lo, hi, True


def render(records, t0: float, scale: float, fonts, note: str) -> Image.Image:
    _body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_title_bar(draw, W, "rapid-llm  —  L1 cross-stream overlap (copy stream vs compute stream)", small=small)

    lanes = {"copy": TITLE_H + PAD + LINE_H, "compute": TITLE_H + PAD + LINE_H + LANE_H + LANE_GAP}
    for name, y in lanes.items():
        draw.text((PAD, y + LANE_H // 2 - 9), f"{name} stream", fill=TEXT_FG, font=bold)
        draw.line([LABEL_W, y + LANE_H, W - PAD, y + LANE_H], fill=AXIS_FG)

    for record in records:
        y = lanes[record.stream] + 12
        # Clip to the window: an overlapped forward runs far past both edges.
        x0 = LABEL_W + (max(record.start_ms, t0) - t0) * scale
        x1 = LABEL_W + (min(record.end_ms, t0 + (W - PAD - LABEL_W) / scale) - t0) * scale
        colour = COPY_FG if record.stream == "copy" else COMPUTE_FG
        draw.rectangle([x0, y, max(x1, x0 + 2), y + LANE_H - 24], fill=colour)
        label = record.name.replace("upload.", "").replace("forward.", "")
        if x1 - x0 > 90:
            draw.text((x0 + 6, y + 8), label, fill=BG, font=small)

    # Time ticks every 5 ms across the window.
    t_end = t0 + (W - PAD - LABEL_W) / scale
    tick = 0.0
    while tick <= t_end - t0:
        x = LABEL_W + tick * scale
        draw.line([x, TITLE_H + PAD // 2, x, TITLE_H + PAD // 2 + 6], fill=DIM)
        draw.text((x - 12, TITLE_H - 2), f"{tick:.0f}ms", fill=DIM, font=small)
        tick += 5.0

    draw.text((PAD, H - PAD - LINE_H), note, fill=OVERLAP_FG if "overlap" in note else DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="docs/images/overlap_l1.gif")
    ap.add_argument("--duration", type=int, default=900, help="ms per frame")
    args = ap.parse_args()

    records = record(args.model_dir)
    window, t0, t1, overlapped = window_around_first_overlap(records)
    if not window:
        print("no timeline regions recorded; is RAPID_LLM_OVERLAP_TIMELINE=1 reachable?")
        return 1
    # Present regions relative to the window, so the overlapping forward's
    # bar visibly continues past both edges instead of starting at the origin.
    window = [
        RegionRecord(r.name, r.stream, r.start_ms - t0, r.end_ms - t0) for r in window
    ]
    span = t1 - t0
    t0, t1 = 0.0, span
    scale = (W - PAD - LABEL_W) / max(span, 1e-3)
    print(f"{len(records)} regions recorded, window shows {len(window)} over {span:.1f} ms")

    fonts = FontPack.mono(17, 17, 15)
    note = (
        "the next pass's input upload lands inside the still-running forward: that is the overlap"
        if overlapped
        else "no copy/compute intersection in this run (overlap disabled or no mixed steps)"
    )
    # Reveal the regions in start order, then hold the full window.
    frames = []
    for count in range(1, len(window) + 1):
        frames.append(render(window[:count], t0, scale, fonts, note))
    frames += [frames[-1]] * 3
    out = Path(args.out)
    save_gif(frames, out, duration=args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
