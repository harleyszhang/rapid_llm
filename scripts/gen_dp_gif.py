"""Record the README GIF: one request stream fanned out across data-parallel replicas.

A throughput number in a table does not show what data parallelism *is*. This renders
the router's decision and the replicas running side by side: each prompt is dealt to a
replica by the load balancer, and the two replicas then decode their own batches at the
same time, on their own GPU. The thing to look at is the two lanes advancing together —
that concurrency is the whole speedup.

What is real here and what is staged: the routing (which replica each prompt goes to)
is the actual :class:`RoundRobinBalancer` decision, and each lane's per-step text and
token counts are recorded from a real :class:`ContinuousBatchingEngine` run over that
lane's sub-batch. The two lanes are recorded one after another on a single GPU, then
replayed in lockstep for the animation, because in a real two-GPU run they execute
concurrently — the GIF shows that truth without needing two cards to render it.

Usage:
    python scripts/gen_dp_gif.py --model-dir my_weight/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.dp_load_balancer import RoundRobinBalancer
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.utils.prompt_templates import get_prompter

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"

# Eight short prompts over two replicas: enough that round-robin visibly stripes them
# 0,1,0,1,... into two lanes of four, and short enough that the GIF stays a few seconds.
PROMPTS = [
    "The capital of France is",
    "One plus one equals",
    "Name a primary colour.",
    "Water boils at",
    "The opposite of hot is",
    "A dog says",
    "The sun rises in the",
    "Two plus two equals",
]
DP_SIZE = 2

# Terminal palette (shared with the continuous-batching GIF for a consistent look).
W, H = 1180, 470
TITLE_H, PAD, LINE_H = 36, 18, 25
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
PROMPT_FG, DIM, TEXT_FG = (118, 214, 118), (128, 136, 148), (222, 226, 232)
RUNNING, QUEUED, DONE = (118, 214, 118), (226, 184, 92), (110, 160, 226)
LANE_BG = (22, 25, 31)
LANE_COLOURS = [(120, 190, 240), (200, 150, 240)]  # GPU 0, GPU 1


@dataclass
class LaneStep:
    """One decode step of one replica: everything the renderer needs for its lane."""

    tokens: int
    rows: list[tuple[str, str, int, str]] = field(default_factory=list)


def record_lane(model_dir: str, prompts: list[tuple[int, str]], max_gen_len: int) -> list[LaneStep]:
    """Run one replica's sub-batch through a real engine, snapshotting after each step.

    Args:
        model_dir: Checkpoint to load.
        prompts: ``(original_index, prompt_text)`` pairs routed to this replica.
        max_gen_len: Per-request generation cap.

    Returns:
        One :class:`LaneStep` per decode step this replica took.
    """
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir, max_seq_len=512, max_num_seqs=len(prompts), max_gpu_num_blocks=8192
    )
    prompter = get_prompter(engine.tokenizer)
    requests = [
        engine.add_request(
            prompter.insert_prompt(text) if prompter else text,
            SamplingParams(temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=1.05),
            request_id=f"req-{index}",
        )
        for index, text in prompts
    ]

    steps: list[LaneStep] = []
    while engine.has_unfinished_requests():
        engine.step()
        rows = []
        for request in requests:
            if request.is_finished:
                status = f"done {request.finish_reason}"
            elif request.slot is not None:
                status = "decoding"
            else:
                status = "queued"
            rows.append(
                (
                    request.request_id,
                    status,
                    len(request.output_token_ids),
                    " ".join(request.text.split()),
                ),
            )
        steps.append(LaneStep(tokens=sum(r[2] for r in rows), rows=rows))
    return steps


def _lane_prompts(balancer: RoundRobinBalancer) -> list[list[tuple[int, str]]]:
    """Route the demo prompts into per-replica buckets using the real balancer."""
    buckets: list[list[tuple[int, str]]] = [[] for _ in range(DP_SIZE)]
    for index, prompt in enumerate(PROMPTS):
        buckets[balancer.select()].append((index, prompt))
    return buckets


def render(step: int, lanes: list[LaneStep], fonts, model_name: str) -> Image.Image:
    """Draw one frame: both replica lanes at their state on ``step``."""
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 9), f"rapid-llm  —  data parallelism  ({model_name})", fill=TITLE_FG, font=small
    )
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)

    y = TITLE_H + PAD
    draw.text(
        (PAD, y),
        f"$ rapid-llm batch --data-parallel-size {DP_SIZE}   # round-robin routing",
        fill=PROMPT_FG,
        font=body,
    )
    y += LINE_H + 4
    total = sum(lane.tokens for lane in lanes)
    draw.text(
        (PAD, y),
        f"step {step:<4d}  {DP_SIZE} replicas running concurrently   {total:4d} tok total",
        fill=TITLE_FG,
        font=bold,
    )
    y += LINE_H + 8

    lane_w = (W - 2 * PAD - 16) // DP_SIZE
    for lane_index, lane in enumerate(lanes):
        x0 = PAD + lane_index * (lane_w + 16)
        draw.rectangle([x0, y, x0 + lane_w, H - PAD], fill=LANE_BG)
        draw.text(
            (x0 + 12, y + 8),
            f"GPU {lane_index}  (replica {lane_index})",
            fill=LANE_COLOURS[lane_index],
            font=bold,
        )
        draw.text((x0 + lane_w - 120, y + 8), f"{lane.tokens:3d} tok", fill=DIM, font=body)
        ry = y + 8 + LINE_H + 4
        for request_id, status, tokens, text in lane.rows:
            if status.startswith("done"):
                colour, marker = DONE, "*"
            elif status == "queued":
                colour, marker = QUEUED, "."
            else:
                colour, marker = RUNNING, ">"
            draw.text(
                (x0 + 12, ry),
                f"{marker} {request_id:<6s} {status:<9s} {tokens:2d}",
                fill=colour,
                font=body,
            )
            preview = text[-22:] if len(text) > 22 else text
            draw.text(
                (x0 + 12, ry + LINE_H - 4),
                f"   {preview or '-'}",
                fill=TEXT_FG if text else DIM,
                font=small,
            )
            ry += LINE_H * 2 - 2

    draw.text(
        (PAD, H - PAD + 2),
        "each prompt is dealt to one replica; the replicas decode their own batch at the same time",
        fill=DIM,
        font=small,
    )
    return canvas


def _hold(lanes: list[list[LaneStep]], step: int) -> list[LaneStep]:
    """State of every lane at ``step``, holding a finished lane on its last frame.

    Lanes finish at different steps (they were dealt different prompts), so a lane that
    is already done keeps showing its final frame while the other lane catches up —
    which is exactly what a real replica does: idle, waiting for the batch to drain.
    """
    return [lane[min(step, len(lane) - 1)] for lane in lanes]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B")
    ap.add_argument("--out", default="docs/images/data_parallel.gif")
    ap.add_argument("--max-gen-len", type=int, default=28)
    ap.add_argument("--every", type=int, default=1, help="keep one frame per N steps")
    ap.add_argument("--duration", type=int, default=110, help="ms per frame")
    args = ap.parse_args()

    buckets = _lane_prompts(RoundRobinBalancer(DP_SIZE))
    print("routing: " + "  ".join(f"GPU{i}={[idx for idx, _ in b]}" for i, b in enumerate(buckets)))

    print("recording replicas ...")
    lanes = [record_lane(args.model_dir, bucket, args.max_gen_len) for bucket in buckets]
    n_steps = max(len(lane) for lane in lanes)
    print(f"{n_steps} steps (lane sizes {[len(lane) for lane in lanes]})")

    fonts = (
        ImageFont.truetype(FONT_PATH, 16),
        ImageFont.truetype(FONT_BOLD, 16),
        ImageFont.truetype(FONT_PATH, 14),
    )
    model_name = Path(args.model_dir).name
    frame_steps = list(range(0, n_steps, args.every))
    images = [
        render(step, _hold(lanes, step), fonts, model_name).convert(
            "P", palette=Image.ADAPTIVE, colors=64
        )
        for step in frame_steps
    ]
    images += [images[-1]] * 8  # hold the finished state so the loop is readable

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=args.duration, loop=0, optimize=True
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
