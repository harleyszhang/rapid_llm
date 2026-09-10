"""Record the SchedulerOutput GIF: one step's decision object, field by field.

v0.7 widened :class:`~rapid_llm.engine.scheduler.SchedulerOutput` so a single
step can carry prefill AND decode together (v0.6 made them mutually exclusive),
plus per-request ``prefill_chunk_lens`` and a ``preempted`` list. This GIF
renders the actual object returned by ``Scheduler.schedule()`` each step across
a scenario that exercises every field: a long prompt chunk-prefilling while
short requests decode, then an oversubscribed step that also preempts.

Every value shown is read straight off the real SchedulerOutput.

Usage:
    python scripts/gen_scheduler_output_gif.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from scripts._viz_lib import (
    BG, TITLE_BG, TITLE_FG, DIM, PROMPT_FG, TEXT_FG,
    GREEN, RED, YELLOW, AMBER,
    TITLE_H, PAD, LINE_H,
    FontPack, draw_title_bar, save_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Request, Scheduler, SchedulerConfig

W, H = 1180, 430

# Map _viz_lib aliases for render use
KEY_FG = (140, 190, 250)
PREFILL_FG = AMBER
DECODE_FG = GREEN
PREEMPT_FG = RED


@dataclass
class Frame:
    step: int
    prefill: list[str] = field(default_factory=list)
    chunk_lens: list[int] = field(default_factory=list)
    decode: list[str] = field(default_factory=list)
    preempted: list[str] = field(default_factory=list)
    note: str = ""


def record() -> list[Frame]:
    config = SchedulerConfig(
        max_seq_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=1 << 20,
        max_chunk_size=256,
        enable_preemption=True,
    )
    sched = Scheduler(config, num_slots=3)

    def mk(rid: str, prompt_len: int) -> Request:
        return Request(
            request_id=rid,
            prompt="x" * prompt_len,
            prompt_token_ids=list(range(prompt_len)),
            params=SamplingParams(temperature=0.0, max_gen_len=64),
        )

    frames: list[Frame] = []
    notes = {
        1: "two short prompts prefill",
        2: "both decode; a 600-tok prompt arrives and chunk-prefills",
        3: "prefill (chunk) + decode populated in the SAME step",
        4: "chunk continues alongside decode",
        5: "oversubscribed: a preemption fills the preempted list",
    }

    # step 1: two short requests
    sched.add_request(mk("short-a", 20))
    sched.add_request(mk("short-b", 20))
    # a long request that will chunk-prefill
    long_added = False

    for step in range(1, 6):
        if step == 2 and not long_added:
            sched.add_request(mk("long-c", 600))
            long_added = True
        if step == 5:
            sched.add_request(mk("short-d", 20))  # forces oversubscription

        out = sched.schedule()
        frames.append(
            Frame(
                step=step,
                prefill=[r.request_id for r in out.prefill],
                chunk_lens=list(out.prefill_chunk_lens),
                decode=[r.request_id for r in out.decode],
                preempted=[r.request_id for r in out.preempted],
                note=notes.get(step, ""),
            )
        )
        for r in out.decode:
            r.output_token_ids.append(999)
        sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
    return frames


def render(frame: Frame, fonts) -> Image.Image:
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_title_bar(draw, W, "rapid-llm  --  SchedulerOutput per step", small=fonts.small)

    y = TITLE_H + PAD
    draw.text((PAD, y), f"out = scheduler.schedule()   # step {frame.step}", fill=PROMPT_FG, font=body)
    y += LINE_H + 10
    draw.line([PAD, y, W - PAD, y], fill=(52, 58, 68))
    y += 14

    def field_line(key: str, value: str, colour):
        draw.text((PAD, y), f"out.{key:<18s}", fill=KEY_FG, font=bold)
        draw.text((PAD + 250, y), f"= {value}", fill=colour, font=body)

    field_line("prefill", str(frame.prefill) if frame.prefill else "[]", PREFILL_FG)
    y += LINE_H
    field_line("prefill_chunk_lens", str(frame.chunk_lens) if frame.chunk_lens else "[]", PREFILL_FG)
    y += LINE_H
    field_line("decode", str(frame.decode) if frame.decode else "[]", DECODE_FG)
    y += LINE_H
    field_line("preempted", str(frame.preempted) if frame.preempted else "[]", PREEMPT_FG)
    y += LINE_H + 10

    coexist = bool(frame.prefill) and bool(frame.decode)
    draw.text(
        (PAD, y),
        f"prefill+decode coexist this step: {coexist}   (v0.6: mutually exclusive)",
        fill=(94, 193, 117) if coexist else DIM,
        font=bold,
    )

    draw.text((PAD, H - PAD - LINE_H), frame.note, fill=DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images/scheduler_output.gif")
    ap.add_argument("--duration", type=int, default=1200, help="ms per frame")
    args = ap.parse_args()

    frames = record()
    for f in frames:
        print(f"step {f.step}: prefill={f.prefill} chunk_lens={f.chunk_lens} "
              f"decode={f.decode} preempted={f.preempted}")

    fonts = FontPack.mono(17, 17, 15)
    images = [render(f, fonts) for f in frames]
    images += [images[-1]] * 2
    out = Path(args.out)
    save_gif(images, out, duration=args.duration)
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(images)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
