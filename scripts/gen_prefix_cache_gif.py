"""Record the prefix-caching GIF: a shared system prompt is prefilled once.

Drives the real :class:`~rapid_llm.engine.scheduler.Scheduler` with prefix
caching enabled: several requests arrive sharing the same long system prompt,
differing only in the user question. The thing to look at is the CACHED column:
the first request prefills the whole prompt and populates the cache; every later
request reuses the shared blocks and only prefills its own tail.

Every number rendered comes from the scheduler's real admission decision
(`Request.num_cached_tokens`) and `prefix_cache_hit_rate`.

Usage:
    python scripts/gen_prefix_cache_gif.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.engine.scheduler import Request, Scheduler, SchedulerConfig

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

#: A shared 768-token system prompt (48 blocks of 16), then a short unique tail.
SYSTEM_PROMPT_LEN = 768
USER_TAIL_LEN = 32
BLOCK_SIZE = 16

W, H = 1180, 430
TITLE_H, PAD, LINE_H = 36, 18, 25
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
PROMPT_FG, DIM, TEXT_FG = (118, 214, 118), (128, 136, 148), (222, 226, 232)
CACHED_FG, PREFILL_FG, MISS_FG = (94, 193, 117), (226, 184, 92), (245, 99, 72)


@dataclass
class Frame:
    step: int
    request_id: str
    prompt_len: int
    cached: int
    to_prefill: int
    hit_rate: float


def _shared_prompt(tail_seed: int) -> list[int]:
    """A prompt that shares the same system prefix but a unique user tail."""
    system = list(range(SYSTEM_PROMPT_LEN))
    tail = list(range(10_000 + tail_seed * 100, 10_000 + tail_seed * 100 + USER_TAIL_LEN))
    return system + tail


def record() -> list[Frame]:
    """Admit several requests with a shared prefix and capture each decision."""
    config = SchedulerConfig(
        max_seq_len=4096,
        max_num_seqs=8,
        max_num_batched_tokens=1 << 20,
        max_chunk_size=0,
        enable_prefix_cache=True,
    )
    sched = Scheduler(config, num_slots=8)

    frames: list[Frame] = []
    for i in range(4):
        rid = "req-0 (cold)" if i == 0 else f"req-{i} (shares sys prompt)"
        request = Request(
            request_id=f"req-{i}",
            prompt="sys+user",
            prompt_token_ids=_shared_prompt(tail_seed=i),
            params=SamplingParams(temperature=0.0, max_gen_len=32),
        )
        sched.add_request(request)
        out = sched.schedule()
        admitted = next(r for r in out.prefill if r.request_id == f"req-{i}")
        frames.append(
            Frame(
                step=i + 1,
                request_id=rid,
                prompt_len=admitted.prompt_len,
                cached=admitted.num_cached_tokens,
                to_prefill=admitted.prompt_len - admitted.num_cached_tokens,
                hit_rate=sched.prefix_cache_hit_rate,
            )
        )
        sched.advance_chunks(out.prefill, out.prefill_chunk_lens)
    return frames


def _bar(cached: int, total: int, width: int = 40) -> str:
    filled = int(width * cached / total) if total else 0
    return "#" * filled + "." * (width - filled)


def render(frame: Frame, fonts) -> Image.Image:
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text((12, 9), "rapid-llm  —  prefix caching (shared system prompt)", fill=TITLE_FG, font=small)
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)

    y = TITLE_H + PAD
    draw.text((PAD, y), "$ rapid-llm serve --enable-prefix-cache", fill=PROMPT_FG, font=body)
    y += LINE_H + 6
    draw.text(
        (PAD, y),
        f"admit {frame.request_id}   prompt = {SYSTEM_PROMPT_LEN} sys + {USER_TAIL_LEN} user tok",
        fill=TITLE_FG,
        font=bold,
    )
    y += LINE_H + 10
    draw.line([PAD, y, W - PAD, y], fill=(52, 58, 68))
    y += 14

    is_cold = frame.cached == 0
    draw.text((PAD, y), "CACHED (KV reused, prefill skipped)", fill=CACHED_FG, font=bold)
    y += LINE_H
    draw.text(
        (PAD, y),
        f"  {frame.cached:>4d} / {frame.prompt_len} tok   [{_bar(frame.cached, frame.prompt_len)}]",
        fill=MISS_FG if is_cold else CACHED_FG,
        font=body,
    )
    y += LINE_H + 8

    draw.text((PAD, y), "TO PREFILL (actual GEMM/attention work this step)", fill=PREFILL_FG, font=bold)
    y += LINE_H
    draw.text((PAD, y), f"  {frame.to_prefill:>4d} tok", fill=PREFILL_FG, font=body)
    y += LINE_H + 8

    draw.text(
        (PAD, y),
        f"cumulative prefix-cache hit rate: {frame.hit_rate * 100:5.1f}%",
        fill=TITLE_FG,
        font=bold,
    )

    tail = (
        "first request is cold: it prefills the whole prompt and populates the cache"
        if is_cold
        else f"reuses {frame.cached} cached tokens; only the {frame.to_prefill}-token tail is prefilled"
    )
    draw.text((PAD, H - PAD - LINE_H), tail, fill=DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/images/prefix_cache.gif")
    ap.add_argument("--duration", type=int, default=1100, help="ms per frame")
    args = ap.parse_args()

    frames = record()
    for f in frames:
        print(f"step {f.step}: {f.request_id:<28s} cached={f.cached:>4d} "
              f"prefill={f.to_prefill:>4d}  hit_rate={f.hit_rate * 100:.1f}%")

    fonts = (
        ImageFont.truetype(FONT_PATH, 17),
        ImageFont.truetype(BOLD_PATH, 17),
        ImageFont.truetype(FONT_PATH, 15),
    )
    images = [render(f, fonts) for f in frames]
    images += [images[-1]] * 3  # hold the final warm state
    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in images]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        out,
        save_all=True,
        append_images=palette[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(palette)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
