"""Record the chunked-prefill GIF: what a long prompt costs a decoding batch.

Drives the real :class:`Scheduler` with one 2000-token prompt arriving into a
batch of already-decoding requests, one frame per scheduler step. The thing to
look at is the PREFILL token count per step: unchunked, a single step carries
all 2000 tokens and every decode request in that step waits behind them;
chunked, no step carries more than ``max_chunk_size``, so a decode request's
next token is bounded by the chunk size instead of by the prompt length.

Both runs come from the same scheduler code; only ``max_chunk_size`` differs.

Usage:
    python scripts/gen_chunked_prefill_gif.py
    python scripts/gen_chunked_prefill_gif.py --chunk 256 --out docs/images/cp.gif
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from scripts._viz_lib import (
    BG, TITLE_BG, TITLE_FG, DIM, PROMPT_FG, TEXT_FG,
    GREEN, RED, YELLOW, AMBER,
    TITLE_H, PAD, LINE_H,
    FontPack, draw_title_bar, save_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import (
    Request,
    Scheduler,
    SchedulerConfig,
)

#: Four short requests already decoding when the long prompt lands.
SHORT_PROMPT_LEN = 24
NUM_SHORT = 4
#: The prompt that used to block everything.
LONG_PROMPT_LEN = 2000

W, H = 1180, 430

# Map _viz_lib aliases for render use
DECODE_OK = GREEN
PREFILL = AMBER
STALLED = RED


@dataclass
class Frame:
    """One scheduler step, everything the renderer needs."""

    step: int
    chunk_size: int
    prefill_rows: list[tuple[str, int, int, int]] = field(default_factory=list)
    decode_ids: list[str] = field(default_factory=list)
    prefill_done_tokens: int = 0
    #: Prefill tokens carried by *this* step. Every decode request in the same
    #: step waits behind them, so this is the decode-latency knob.
    step_prefill_tokens: int = 0


def _make_request(request_id: str, prompt_len: int) -> Request:
    return Request(
        request_id=request_id,
        prompt="x" * prompt_len,
        prompt_token_ids=list(range(prompt_len)),
        params=SamplingParams(temperature=0.0, max_gen_len=64),
    )


def record(chunk_size: int, max_steps: int = 14) -> list[Frame]:
    """Drive the real scheduler and capture what it decided each step."""
    config = SchedulerConfig(
        max_seq_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=65536,
        max_chunk_size=chunk_size,
    )
    sched = Scheduler(config, num_slots=8)

    # Get the short requests admitted and past their prefill first.
    for i in range(NUM_SHORT):
        sched.add_request(_make_request(f"short-{i}", SHORT_PROMPT_LEN))
    warm = sched.schedule()
    sched.advance_chunks(warm.prefill, warm.prefill_chunk_lens)

    # The long prompt lands into a running batch.
    sched.add_request(_make_request("LONG", LONG_PROMPT_LEN))

    frames: list[Frame] = []
    done_tokens = 0
    for step in range(1, max_steps + 1):
        out = sched.schedule()

        rows: list[tuple[str, int, int, int]] = []
        long_chunk = 0
        for request, chunk_len in zip(out.prefill, out.prefill_chunk_lens, strict=False):
            if request.request_id == "LONG":
                long_chunk = chunk_len
                processed = min(done_tokens + chunk_len, request.prompt_len)
            else:
                processed = chunk_len
            rows.append((request.request_id, chunk_len, processed, request.prompt_len))

        step_tokens = sum(out.prefill_chunk_lens)
        done_tokens = min(done_tokens + long_chunk, LONG_PROMPT_LEN)

        frames.append(
            Frame(
                step=step,
                chunk_size=chunk_size,
                prefill_rows=rows,
                decode_ids=[r.request_id for r in out.decode],
                prefill_done_tokens=done_tokens,
                step_prefill_tokens=step_tokens,
            )
        )

        sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
        if done_tokens >= LONG_PROMPT_LEN and not out.prefill:
            break
    return frames


def _bar(done: int, total: int, width: int = 34) -> str:
    filled = int(width * done / total) if total else 0
    return "#" * filled + "-" * (width - filled)


def render(frame: Frame, fonts, mode_label: str) -> Image.Image:
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_title_bar(draw, W, f"rapid-llm  —  chunked prefill  ({mode_label})", small=fonts.small)

    y = TITLE_H + PAD
    flag = f"--max-chunk-size {frame.chunk_size}" if frame.chunk_size else "--max-chunk-size 0"
    draw.text((PAD, y), f"$ rapid-llm serve {flag}", fill=PROMPT_FG, font=body)
    y += LINE_H + 6

    draw.text(
        (PAD, y),
        f"scheduler step {frame.step:<3d}   "
        f"this step's prefill: {frame.step_prefill_tokens:>4d} tok   "
        f"LONG {frame.prefill_done_tokens:>4d}/{LONG_PROMPT_LEN} "
        f"[{_bar(frame.prefill_done_tokens, LONG_PROMPT_LEN)}]",
        fill=TITLE_FG,
        font=bold,
    )
    y += LINE_H + 10
    draw.line([PAD, y, W - PAD, y], fill=(52, 58, 68))
    y += 12

    draw.text((PAD, y), "PREFILL", fill=PREFILL, font=bold)
    y += LINE_H
    if frame.prefill_rows:
        for request_id, chunk_len, processed, total in frame.prefill_rows:
            label = (
                f"  {request_id:<8s} chunk {chunk_len:>4d} tok"
                f"   {processed:>4d}/{total} of prompt"
            )
            draw.text((PAD, y), label, fill=PREFILL, font=body)
            y += LINE_H
    else:
        draw.text((PAD, y), "  (idle)", fill=DIM, font=body)
        y += LINE_H

    y += 6
    # A step's prefill token count is what its decode requests wait behind.
    heavy = frame.step_prefill_tokens > 1024
    decode_colour = STALLED if heavy else DECODE_OK
    draw.text((PAD, y), "DECODE", fill=decode_colour, font=bold)
    y += LINE_H
    if frame.decode_ids:
        draw.text(
            (PAD, y),
            "  " + "  ".join(f"[{rid} +1 tok]" for rid in frame.decode_ids),
            fill=DECODE_OK,
            font=body,
        )
        y += LINE_H
        wait = (
            f"  ^ these {len(frame.decode_ids)} requests wait behind "
            f"{frame.step_prefill_tokens} prefill tokens this step"
        )
        draw.text((PAD, y), wait, fill=decode_colour, font=small)
    else:
        draw.text((PAD, y), "  (no request decoding yet)", fill=DIM, font=body)
    y += LINE_H

    tail = (
        f"per-step prefill work is capped at {frame.chunk_size} tokens, so decode"
        " latency is bounded by the chunk, not by the prompt"
        if frame.chunk_size
        else "one step carries the whole 2000-token prompt; decode in that step"
        " waits behind all of it"
    )
    draw.text((PAD, H - PAD - LINE_H), tail, fill=DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=512, help="chunk size for the ON run")
    ap.add_argument("--out", default="docs/images/chunked_prefill.gif")
    ap.add_argument("--duration", type=int, default=700, help="ms per frame")
    args = ap.parse_args()

    off = record(chunk_size=0)
    on = record(chunk_size=args.chunk)
    off_peak = max(f.step_prefill_tokens for f in off)
    on_peak = max(f.step_prefill_tokens for f in on)
    print(f"chunk=0   -> {len(off)} steps, peak prefill tokens in one step: {off_peak}")
    print(f"chunk={args.chunk} -> {len(on)} steps, peak prefill tokens in one step: {on_peak}")
    print(f"worst-case per-step prefill work reduced {off_peak / on_peak:.1f}x")

    fonts = FontPack.mono(17, 17, 15)

    # Show the blocking run first, then the chunked one, so the GIF reads as
    # a before/after rather than two unrelated clips.
    sequence = (
        [render(f, fonts, "chunking OFF") for f in off]
        + [render(off[-1], fonts, "chunking OFF")] * 2
        + [render(f, fonts, f"chunking ON, chunk={args.chunk}") for f in on]
        + [render(on[-1], fonts, f"chunking ON, chunk={args.chunk}")] * 3
    )
    images = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in sequence]
    out = Path(args.out)
    save_gif(images, out, duration=args.duration)
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(images)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
