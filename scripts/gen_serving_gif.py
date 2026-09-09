"""Record the README GIF: requests joining and leaving one continuously batched run.

A throughput number in a table does not show what continuous batching *is*. This
renders the batch itself, one frame per decode step, so the thing to look at is
the slot column: a request finishes, its slot goes back to the pool, and a queued
request is decoding in it on the very next frame.

Usage:
    python scripts/gen_serving_gif.py --model-dir my_weight/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.utils.prompt_templates import get_prompter

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Six requests, three slots, and a different length cap each. The staggered caps
# are what puts slot recycling on camera repeatedly instead of once at the end:
# an instruct model left to its own devices answers all six at similar length, so
# they would finish together and the queue would drain in one jump. Per-request
# caps are also something the one-shot path cannot express at all.
PROMPTS = [
    ("Name the capital of Japan.", 18),
    ("List three prime numbers.", 30),
    ("Why is the sky blue? One sentence.", 44),
    ("Write a two-line haiku about rain.", 26),
    ("Say hello in French.", 14),
    ("What is 12 times 12?", 20),
]
MAX_NUM_SEQS = 3

# Terminal palette
W, H = 1180, 400
TITLE_H, PAD, LINE_H = 36, 18, 25
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
PROMPT_FG, DIM, TEXT_FG = (118, 214, 118), (128, 136, 148), (222, 226, 232)
RUNNING, QUEUED, DONE = (118, 214, 118), (226, 184, 92), (110, 160, 226)


@dataclass
class Frame:
    """One decode step's worth of state, everything the renderer needs."""

    step: int
    elapsed: float
    tokens: int
    running: int
    queued: int
    rows: list[tuple[str, str, int, str]] = field(default_factory=list)


def record(model_dir: str, max_gen_len: int) -> list[Frame]:
    """Drive the engine and snapshot the batch after every step."""
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir, max_seq_len=768, max_num_seqs=MAX_NUM_SEQS, max_gpu_num_blocks=24576
    )
    engine.generate([PROMPTS[0][0]], SamplingParams(temperature=0.0, max_gen_len=4))  # warm up

    # Instruct checkpoints need their chat template; fed a bare prompt they drift
    # into completion mode and answer with scraped-looking prose, which would put
    # the model's worst behaviour in the README for no reason. `rapid-llm batch`
    # applies the template too, so this also keeps the GIF honest about the CLI.
    prompter = get_prompter(engine.tokenizer)

    requests = [
        engine.add_request(
            prompter.insert_prompt(prompt) if prompter else prompt,
            SamplingParams(
                temperature=0.0,
                max_gen_len=min(cap, max_gen_len),
                repetition_penalty=1.05,
            ),
            request_id=f"req-{index}",
        )
        for index, (prompt, cap) in enumerate(PROMPTS)
    ]

    frames: list[Frame] = []
    started = time.perf_counter()
    step = 0
    while engine.has_unfinished_requests():
        engine.step()
        step += 1
        rows = []
        for request in requests:
            if request.is_finished:
                status = f"done  {request.finish_reason}"
            elif request.slot is not None:
                status = f"slot {request.slot}"
            else:
                status = "queued"
            rows.append(
                (
                    request.request_id,
                    status,
                    len(request.output_token_ids),
                    " ".join(request.text.split()),
                )
            )
        frames.append(
            Frame(
                step=step,
                elapsed=time.perf_counter() - started,
                tokens=sum(len(r.output_token_ids) for r in requests),
                running=engine.scheduler.num_running,
                queued=engine.scheduler.num_waiting,
                rows=rows,
            )
        )
    return frames


def render(frame: Frame, fonts, model_name: str) -> Image.Image:
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 9), f"rapid-llm  —  continuous batching  ({model_name})", fill=TITLE_FG, font=small
    )
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)

    y = TITLE_H + PAD
    draw.text(
        (PAD, y),
        f"$ rapid-llm batch --max-num-seqs {MAX_NUM_SEQS} --show-stats",
        fill=PROMPT_FG,
        font=body,
    )
    y += LINE_H + 6

    tps = frame.tokens / frame.elapsed if frame.elapsed else 0.0
    draw.text(
        (PAD, y),
        f"step {frame.step:<4d}  running {frame.running}/{MAX_NUM_SEQS}   "
        f"queued {frame.queued}   {frame.tokens:4d} tok   {tps:6.0f} tok/s",
        fill=TITLE_FG,
        font=bold,
    )
    y += LINE_H + 10
    draw.line([PAD, y, W - PAD, y], fill=(52, 58, 68))
    y += 12

    for request_id, status, tokens, text in frame.rows:
        if status.startswith("done"):
            colour, marker = DONE, "*"
        elif status == "queued":
            colour, marker = QUEUED, "."
        else:
            colour, marker = RUNNING, ">"
        draw.text(
            (PAD, y),
            f"{marker} {request_id:<7s} {status:<12s} {tokens:3d} tok",
            fill=colour,
            font=body,
        )
        preview = text[-66:] if len(text) > 66 else text
        draw.text((PAD + 330, y), preview or "-", fill=TEXT_FG if text else DIM, font=body)
        y += LINE_H

    y = H - PAD - LINE_H
    draw.text(
        (PAD, y),
        "a slot freed by a finished request is decoding a queued one on the next step",
        fill=DIM,
        font=small,
    )
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="docs/images/continuous_batching.gif")
    ap.add_argument("--max-gen-len", type=int, default=64, help="ceiling on the per-request caps")
    ap.add_argument("--every", type=int, default=2, help="keep one frame per N steps")
    ap.add_argument("--duration", type=int, default=90, help="ms per frame")
    args = ap.parse_args()

    print("recording ...")
    frames = record(args.model_dir, args.max_gen_len)
    print(f"{len(frames)} steps recorded")

    fonts = (
        ImageFont.truetype(FONT_PATH, 17),
        ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 17),
        ImageFont.truetype(FONT_PATH, 15),
    )
    model_name = Path(args.model_dir).name
    kept = frames[:: args.every]
    kept += [frames[-1]] * 8  # hold the finished state so the loop is readable

    print(f"rendering {len(kept)} frames ...")
    images = [
        render(f, fonts, model_name).convert("P", palette=Image.ADAPTIVE, colors=64) for f in kept
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=args.duration, loop=0, optimize=True
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
