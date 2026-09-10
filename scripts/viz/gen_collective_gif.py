"""Record the README GIF: what a tensor-parallel decode step puts on the wire.

Tensor parallelism is usually shown as a diagram of split matrices, which says nothing
about its cost. The cost is bytes between ranks, and this renders them: a real TP=2 run
of a real checkpoint, with rank 0's collective ledger beside the text it is generating.
Two things are meant to be visible. The layers dominate — every row-parallel projection
all-reduces its activations, twice per layer, and that bar is the whole traffic budget.
And sampling does not: the vocabulary collectives stay flat at a few dozen bytes per
step no matter how large the vocabulary is, because :func:`global_argmax` exchanges two
values per row rather than gathering the logits.

Nothing here is staged. The engine is a real
:class:`~rapid_llm.engine.continuous_engine.ContinuousBatchingEngine` with
``tensor_parallel_size=2``, so this process *is* rank 0 and the follower is a real
process on the second GPU; every number comes from a
:meth:`~rapid_llm.tools.observability.CollectiveStats.collect` window wrapped around
:meth:`step`, nested inside a window over the whole run. The one figure that is
arithmetic rather than measurement is the "if gathered" line, which is what one rank
would contribute to an all-gather of the logits — the alternative implementation, not
this one, and labelled as such.

Usage:
    python scripts/gen_collective_gif.py --model-dir my_weight/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from scripts._viz_lib import (
    BG, TITLE_BG, TITLE_FG, DIM, PROMPT_FG, TEXT_FG,
    GREEN, RED, YELLOW, BLUE, PURPLE, PANEL_BG, BAR_BG,
    DATA_FG, CONTROL_FG, ALERT,
    RUNNING, QUEUED, DONE,
    TITLE_H, PAD, LINE_H,
    FontPack, draw_title_bar, save_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.tools.observability import (
    Collective,
    CollectiveStats,
    Plane,
    Tally,
    human_bytes,
)
from rapid_llm.utils.prompt_templates import get_prompter

TP_SIZE = 2

# Four short prompts: enough that the batch dimension shows up in the byte counts,
# few enough that every request has its own row and the GIF stays a few seconds.
PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The opposite of hot is",
    "The sun rises in the",
]

#: Ops that exist only because the vocabulary is sharded — the sampler's own traffic.
#: ``all_reduce`` is excluded because the layers use it too, and the claim being shown
#: is about sampling alone.
VOCAB_OPS = (Collective.ALL_REDUCE_MAX, Collective.ALL_GATHER)

#: Bytes per logit for the "if the logits were gathered" comparison. The checkpoints
#: here are fp16, and understating the alternative is the right way round: the claim is
#: that a gathering sampler moves *at least* this much.
LOGIT_BYTES = 2

W, H = 1180, 476


@dataclass
class Frame:
    """One decode step, from both sides: the text it produced and the bytes it cost."""

    step: int
    phase: str
    sampled_rows: int = 0
    rows: list[tuple[str, str, int, str]] = field(default_factory=list)
    step_tallies: dict[Collective, Tally] = field(default_factory=dict)
    run_tallies: dict[Collective, Tally] = field(default_factory=dict)
    step_bytes: int = 0
    run_bytes: int = 0
    vocab_bytes: int = 0
    gathered_bytes: int = 0


def _colour(op: Collective) -> tuple[int, int, int]:
    """One colour per plane, so the two budgets read apart at a glance.

    Which plane an op is on comes from the op itself; this only picks the ink.
    """
    return CONTROL_FG if op.plane is Plane.CONTROL else DATA_FG


def _vocab_bytes(tallies: dict[Collective, Tally]) -> int:
    return sum(tallies.get(op, Tally()).nbytes for op in VOCAB_OPS)


def record(model_dir: str, max_gen_len: int) -> tuple[list[Frame], str]:
    """Drive a real TP=2 engine, snapshotting rank 0's ledger after every step.

    Args:
        model_dir: Checkpoint to shard over ``TP_SIZE`` GPUs.
        max_gen_len: Per-request generation cap.

    Returns:
        One :class:`Frame` per step, and the ledger report for the whole run.
    """
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=512,
        max_num_seqs=len(PROMPTS),
        max_gpu_num_blocks=8192,
        tensor_parallel_size=TP_SIZE,
    )
    prompter = get_prompter(engine.tokenizer)
    requests = [
        engine.add_request(
            prompter.insert_prompt(text) if prompter else text,
            SamplingParams(temperature=0.0, max_gen_len=max_gen_len, repetition_penalty=1.05),
            request_id=f"req-{index}",
        )
        for index, text in enumerate(PROMPTS)
    ]
    # What a gathering sampler would move instead: one rank's slice of the logits for
    # every row it samples.
    gathered_per_row = engine.engine.model_runner.vocab_size // TP_SIZE * LOGIT_BYTES

    frames: list[Frame] = []
    try:
        with CollectiveStats.collect() as run:
            while engine.has_unfinished_requests():
                # Exact, not a guess: a request with no output tokens can only be
                # advanced by a prefill pass.
                phase = "prefill" if any(not r.output_token_ids for r in requests) else "decode"
                settled = {r.request_id for r in requests if r.is_finished}
                with CollectiveStats.collect() as step:
                    advanced = engine.step()
                # Rows the sampler saw: those that kept a token, plus those whose token
                # was a stop token — sampled all the same, then dropped as model
                # punctuation rather than output.
                stopped = sum(
                    1 for r in requests if r.request_id not in settled and r.finish_reason == "eos"
                )
                sampled_rows = len(advanced) + stopped
                frames.append(
                    Frame(
                        step=len(frames) + 1,
                        phase=phase,
                        sampled_rows=sampled_rows,
                        rows=[_row(request) for request in requests],
                        step_tallies=step.tallies(),
                        run_tallies=run.tallies(),
                        step_bytes=step.nbytes,
                        run_bytes=run.nbytes,
                        vocab_bytes=_vocab_bytes(step.tallies()),
                        gathered_bytes=sampled_rows * gathered_per_row,
                    )
                )
        report = run.report()
    finally:
        engine.shutdown()
    return frames, report


def _row(request) -> tuple[str, str, int, str]:
    if request.is_finished:
        status = f"done {request.finish_reason}"
    elif request.slot is not None:
        status = "decoding"
    else:
        status = "queued"
    return request.request_id, status, len(request.output_token_ids), " ".join(request.text.split())


def _draw_requests(draw, x0: float, y0: float, width: float, frame: Frame, fonts) -> None:
    """Left panel: the run this traffic belongs to, so the bytes have a subject."""
    body, bold, small = fonts
    draw.rectangle([x0, y0, x0 + width, H - PAD - 22], fill=PANEL_BG)
    draw.text((x0 + 12, y0 + 8), f"replica 0  —  {TP_SIZE} ranks, 1 batch", fill=DATA_FG, font=bold)
    y = y0 + 8 + LINE_H + 2
    for request_id, status, tokens, text in frame.rows:
        if status.startswith("done"):
            colour, marker = DONE, "*"
        elif status == "queued":
            colour, marker = QUEUED, "."
        else:
            colour, marker = RUNNING, ">"
        draw.text(
            (x0 + 12, y),
            f"{marker} {request_id:<6s} {status:<9s} {tokens:2d}",
            fill=colour,
            font=body,
        )
        preview = text[-30:] if len(text) > 30 else text
        draw.text((x0 + 12, y + LINE_H - 5), f"   {preview or '-'}", fill=TEXT_FG, font=small)
        y += LINE_H * 2 - 3


def _draw_ledger(draw, x0: float, y0: float, width: float, frame: Frame, fonts) -> None:
    """Right panel: the ledger, one row per op, bar showing its share of *this* step.

    Scaled within the step rather than across the run, because a prefill pass moves an
    order of magnitude more than the decodes that follow it, and a shared scale would
    flatten every decode frame to a sliver — hiding the comparison the panel is for.
    Cross-step magnitude is in the header instead, where it is a number.
    """
    body, bold, small = fonts
    draw.rectangle([x0, y0, x0 + width, H - PAD - 22], fill=PANEL_BG)
    draw.text((x0 + 12, y0 + 8), "collective ledger  —  rank 0", fill=TITLE_FG, font=bold)
    draw.text((x0 + width - 168, y0 + 8), "this step  /  run", fill=DIM, font=small)

    y = y0 + 8 + LINE_H + 4
    bar_x, bar_w = x0 + 12, width - 24
    scale = max((tally.nbytes for tally in frame.step_tallies.values()), default=0)
    for op in frame.run_tallies:
        colour = _colour(op)
        step_bytes = frame.step_tallies.get(op, Tally()).nbytes
        run_bytes = frame.run_tallies[op].nbytes
        draw.text((bar_x, y), f"{op:<17s}{op.plane:<8s}", fill=colour, font=body)
        draw.text(
            (bar_x + 300, y),
            f"{human_bytes(step_bytes):>9s} /{human_bytes(run_bytes):>9s}",
            fill=TEXT_FG if step_bytes else DIM,
            font=body,
        )
        draw.rectangle([bar_x, y + LINE_H - 4, bar_x + bar_w, y + LINE_H + 2], fill=BAR_BG)
        filled = bar_w * step_bytes / scale if scale else 0
        if filled >= 1:
            draw.rectangle([bar_x, y + LINE_H - 4, bar_x + filled, y + LINE_H + 2], fill=colour)
        y += LINE_H + 12

    y = H - PAD - 22 - 2 * LINE_H - 8
    draw.line([bar_x, y - 8, bar_x + bar_w, y - 8], fill=(52, 58, 68))
    ratio = frame.gathered_bytes / frame.vocab_bytes if frame.vocab_bytes else 0
    draw.text(
        (bar_x, y),
        f"vocabulary collectives  {human_bytes(frame.vocab_bytes):>9s}"
        f"   (2 values x {frame.sampled_rows} row{'' if frame.sampled_rows == 1 else 's'})",
        fill=DATA_FG,
        font=body,
    )
    draw.text(
        (bar_x, y + LINE_H),
        f"if the logits were gathered {human_bytes(frame.gathered_bytes):>9s}"
        + (f"   {ratio:,.0f}x more" if ratio else ""),
        fill=ALERT,
        font=body,
    )


def render(frame: Frame, fonts, model_name: str) -> Image.Image:
    """Draw one frame: the batch on the left, what it cost the wire on the right."""
    body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw_title_bar(draw, W, f"rapid-llm  —  tensor parallelism: what crosses the wire  ({model_name})", small=fonts.small)

    y = TITLE_H + PAD
    draw.text(
        (PAD, y),
        f"$ rapid-llm batch --tensor-parallel-size {TP_SIZE}   # bytes measured on rank 0",
        fill=PROMPT_FG,
        font=body,
    )
    y += LINE_H + 4
    draw.text(
        (PAD, y),
        f"step {frame.step:<4d} {frame.phase:<8s} "
        f"wire {human_bytes(frame.step_bytes)} this step  /  {human_bytes(frame.run_bytes)} so far",
        fill=TITLE_FG,
        font=bold,
    )
    y += LINE_H + 8

    left_w = 420
    _draw_requests(draw, PAD, y, left_w, frame, fonts)
    _draw_ledger(draw, PAD + left_w + 16, y, W - 2 * PAD - left_w - 16, frame, fonts)

    draw.text(
        (PAD, H - PAD - 16),
        "the layers own the traffic; sampling stays flat because the logits never leave their rank",
        fill=DIM,
        font=small,
    )
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="docs/images/tensor_parallel.gif")
    ap.add_argument("--max-gen-len", type=int, default=24)
    ap.add_argument("--every", type=int, default=1, help="keep one frame per N steps")
    ap.add_argument("--duration", type=int, default=130, help="ms per frame")
    args = ap.parse_args()

    print(f"recording a real tp={TP_SIZE} run ...")
    frames, report = record(args.model_dir, args.max_gen_len)
    print(f"{len(frames)} steps\n{report}")

    fonts = FontPack.mono(16, 16, 14)
    model_name = Path(args.model_dir).name
    images = [
        render(frame, fonts, model_name).convert("P", palette=Image.ADAPTIVE, colors=64)
        for frame in frames[:: args.every]
    ]
    images += [images[-1]] * 8  # hold the finished state so the loop is readable
    out = Path(args.out)
    save_gif(images, out, duration=args.duration)
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
