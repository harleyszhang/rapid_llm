"""Record the preemption GIF: 3 requests time-share 2 slots by recompute.

Drives the real :class:`~rapid_llm.engine.scheduler.Scheduler` with
``enable_preemption=True`` and ``max_num_seqs=3 > num_slots=2``. The row to
watch is PREEMPTED: each step the youngest decoding request is evicted (KV
dropped, re-queued for recompute) so a waiting request gets a slot, giving a
fair round-robin. Every id shown is the real ``SchedulerOutput.preempted`` /
``.decode`` / ``.prefill`` plus ``Scheduler.num_preemptions``.

Usage:
    python scripts/gen_preemption_gif.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Request, Scheduler, SchedulerConfig

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

NUM_SLOTS = 2
MAX_NUM_SEQS = 3

W, H = 1180, 430
TITLE_H, PAD, LINE_H = 36, 18, 25
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
PROMPT_FG, DIM = (118, 214, 118), (128, 136, 148)
PREFILL_FG, DECODE_FG, PREEMPT_FG = (226, 184, 92), (94, 193, 117), (245, 99, 72)


@dataclass
class Frame:
    step: int
    prefill: list[str] = field(default_factory=list)
    decode: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    running: list[str] = field(default_factory=list)
    total_preemptions: int = 0


def record(steps: int = 7) -> list[Frame]:
    config = SchedulerConfig(
        max_seq_len=4096,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=1 << 20,
        max_chunk_size=0,
        enable_preemption=True,
    )
    sched = Scheduler(config, num_slots=NUM_SLOTS)
    for i in range(3):
        sched.add_request(
            Request(
                request_id=f"req-{i}",
                prompt="x" * 20,
                prompt_token_ids=list(range(20)),
                params=SamplingParams(temperature=0.0, max_gen_len=64),
            )
        )

    frames: list[Frame] = []
    for step in range(1, steps + 1):
        out = sched.schedule()
        frames.append(
            Frame(
                step=step,
                prefill=[r.request_id for r in out.prefill],
                decode=[r.request_id for r in out.decode],
                preempted=[r.request_id for r in out.preempted],
                running=[r.request_id for r in sched.running],
                total_preemptions=sched.num_preemptions,
            )
        )
        for r in out.decode:
            r.output_token_ids.append(999)
        sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
    return frames


def _chip(draw, x, y, text, colour, font):
    w = draw.textlength(text, font=font)
    draw.rounded_rectangle([x, y, x + w + 16, y + 22], radius=6, outline=colour, width=2)
    draw.text((x + 8, y + 3), text, fill=colour, font=font)
    return x + w + 16 + 10


def render(frame: Frame, fonts) -> Image.Image:
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 9),
        f"rapid-llm  --  preemption / recompute  ({MAX_NUM_SEQS} reqs share {NUM_SLOTS} slots)",
        fill=TITLE_FG,
        font=small,
    )
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)

    y = TITLE_H + PAD
    draw.text((PAD, y), "$ rapid-llm serve --enable-preemption --max-num-seqs 3", fill=PROMPT_FG, font=body)
    y += LINE_H + 6
    draw.text(
        (PAD, y),
        f"scheduler step {frame.step:<3d}   slots in use {len(frame.running)}/{NUM_SLOTS}   "
        f"total preemptions {frame.total_preemptions}",
        fill=TITLE_FG,
        font=bold,
    )
    y += LINE_H + 10
    draw.line([PAD, y, W - PAD, y], fill=(52, 58, 68))
    y += 14

    draw.text((PAD, y), "PREFILL / recompute", fill=PREFILL_FG, font=bold)
    y += LINE_H
    x = PAD + 16
    if frame.prefill:
        for rid in frame.prefill:
            x = _chip(draw, x, y, rid, PREFILL_FG, body)
    else:
        draw.text((PAD + 16, y + 2), "(none)", fill=DIM, font=body)
    y += LINE_H + 10

    draw.text((PAD, y), "DECODE (+1 token)", fill=DECODE_FG, font=bold)
    y += LINE_H
    x = PAD + 16
    if frame.decode:
        for rid in frame.decode:
            x = _chip(draw, x, y, rid, DECODE_FG, body)
    else:
        draw.text((PAD + 16, y + 2), "(none)", fill=DIM, font=body)
    y += LINE_H + 10

    draw.text((PAD, y), "PREEMPTED (KV dropped -> re-queued for recompute)", fill=PREEMPT_FG, font=bold)
    y += LINE_H
    x = PAD + 16
    if frame.preempted:
        for rid in frame.preempted:
            x = _chip(draw, x, y, rid, PREEMPT_FG, body)
    else:
        draw.text((PAD + 16, y + 2), "(none)", fill=DIM, font=body)

    tail = (
        "youngest decoding request is evicted so a waiting one runs; the progress "
        "quantum keeps every request advancing"
    )
    draw.text((PAD, H - PAD - LINE_H), tail, fill=DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images/preemption.gif")
    ap.add_argument("--duration", type=int, default=900, help="ms per frame")
    args = ap.parse_args()

    frames = record()
    for f in frames:
        print(f"step {f.step}: prefill={f.prefill} decode={f.decode} "
              f"preempted={f.preempted} total={f.total_preemptions}")

    fonts = (
        ImageFont.truetype(FONT_PATH, 17),
        ImageFont.truetype(BOLD_PATH, 17),
        ImageFont.truetype(FONT_PATH, 15),
    )
    images = [render(f, fonts) for f in frames]
    images += [images[-1]] * 2
    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in images]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        out, save_all=True, append_images=palette[1:], duration=args.duration, loop=0, optimize=True
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(palette)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
