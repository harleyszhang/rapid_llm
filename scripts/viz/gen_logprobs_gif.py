"""Generate the logprobs demo GIF for README.

Shows what F6 actually delivers: one greedy run reports the log-probability of
every *prompt* token and, for each generated token, the top-k alternatives it was
chosen over — both out of the same forward pass, no second scoring run.

Every number in the GIF comes from a real run of the checkpoint (needs a GPU and
``my_weight/Qwen3-0.6B``), the way ``gen_overlap_l1_gif.py`` records its timeline:
a hand-written table would defeat the purpose of showing the feature works.

Usage:
    python scripts/gen_logprobs_gif.py
    python scripts/gen_logprobs_gif.py --model-dir my_weight/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from scripts._viz_lib import (
    BG, TITLE_BG, TITLE_FG, DIM, PROMPT_FG, TEXT_FG,
    CYAN, YELLOW, GREEN, BLUE, RED, BAR_BG,
    TITLE_H, PAD, LINE_H,
    FontPack,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT = REPO_ROOT / "docs" / "images" / "logprobs.gif"

#: Short and factual: the prompt table has to fit on screen, and a prompt the
#: model is confident about makes the one uncertain position stand out.
PROMPT = "The capital of France is"
TOP_K = 5
MAX_GEN = 6

W, H = 1080, 640

# Helper: re-use _viz_lib _fonts but with a 4th 'big' slot
PAD = 16
LINE_H = 24


def _fonts():
    try:
        body = ImageFont.truetype(FONT_PATH, 14)
        bold = ImageFont.truetype(FONT_BOLD, 14)
        small = ImageFont.truetype(FONT_PATH, 12)
        big = ImageFont.truetype(FONT_BOLD, 16)
    except OSError:
        body = bold = small = big = ImageFont.load_default()
    return body, bold, small, big


def collect(model_dir: str) -> dict:
    """One greedy run with both halves of the feature switched on.

    ``prompt_logprobs`` position 0 comes back as ``None`` — nothing predicts the
    first token — and that hole is worth showing rather than hiding: it is the
    contract callers code against.
    """
    from rapid_llm.engine.llm import LLM
    from rapid_llm.engine.sampler import SamplingParams

    llm = LLM(model=model_dir, max_seq_len=256, max_gpu_num_blocks=2048, use_cuda_graph=False)
    output = llm.generate(
        [PROMPT],
        SamplingParams(
            temperature=0.0,
            max_gen_len=MAX_GEN,
            repetition_penalty=1.0,
            stop_on_repeat=False,
            logprobs=TOP_K,
            prompt_logprobs=TOP_K,
        ),
    )[0]
    decode = llm.tokenizer.decode

    records = output.prompt_logprobs or []
    # Position 0 has no record; take its text from the tokenizer so the row still
    # names the token it refers to.
    ids = llm.tokenizer.encode(PROMPT)
    prompt_rows = [
        {"token": decode([ids[i]]) if i < len(ids) else "", "logprob": None}
        if record is None
        else {"token": decode([record.token_id]), "logprob": record.logprob}
        for i, record in enumerate(records)
    ]
    steps = [
        {
            "token": decode([record.token_id]),
            "logprob": record.logprob,
            "alternatives": [
                (decode([token_id]), value)
                for token_id, value in zip(record.top_token_ids, record.top_logprobs, strict=True)
            ],
        }
        for record in output.outputs[0].logprobs or []
    ]
    return {"prompt_rows": prompt_rows, "steps": steps, "text": output.outputs[0].text}


def _title_bar(draw, fonts, model_name: str):
    _, _, small, _ = fonts
    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 9),
        f"rapid_llm  —  logprobs / prompt_logprobs  ({model_name})",
        fill=TITLE_FG,
        font=small,
    )
    for i, colour in enumerate([RED, YELLOW, GREEN]):
        draw.ellipse([W - 78 + i * 18, 11, W - 68 + i * 18, 21], fill=colour)


def _bar(draw, x: int, y: int, logprob: float, colour, width: int = 260):
    """Probability bar: ``exp(logprob)`` is the readable form of a log-probability."""
    draw.rectangle([x, y + 4, x + width, y + LINE_H - 8], fill=BAR_BG)
    filled = int(math.exp(logprob) * width)
    if filled > 0:
        draw.rectangle([x, y + 4, x + filled, y + LINE_H - 8], fill=colour)


def _quoted(token: str) -> str:
    """Tokens carry their leading space; quoting keeps that visible."""
    return f"{token!r}"


def _frame(fonts, model_name: str, header, rows) -> Image.Image:
    """One terminal frame: a few header lines, then already-revealed rows."""
    body, bold, _, _ = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _title_bar(draw, fonts, model_name)

    y = TITLE_H + PAD
    for text, colour, weight in header:
        draw.text((PAD, y), text, fill=colour, font=bold if weight == "bold" else body)
        y += LINE_H
    y += 6
    for row in rows:
        y = row(draw, fonts, y)
    return canvas


def _prompt_row(index: int, entry: dict):
    """A scored prompt position: token, its logprob, and the bar."""

    def draw_row(draw, fonts, y: int) -> int:
        body, bold, _, _ = fonts
        if entry["logprob"] is None:
            draw.text((PAD, y), f"{index:>3d}  {_quoted(entry['token']):<14s}", fill=DIM, font=body)
            draw.text((PAD + 200, y), "    —     nothing predicts position 0", fill=DIM, font=body)
            return y + LINE_H
        draw.text((PAD, y), f"{index:>3d}  {_quoted(entry['token']):<14s}", fill=TEXT_FG, font=body)
        draw.text((PAD + 200, y), f"{entry['logprob']:>8.3f}", fill=CYAN, font=bold)
        _bar(draw, PAD + 300, y, entry["logprob"], CYAN)
        return y + LINE_H

    return draw_row


def _step_rows(step: dict, revealed: bool):
    """The chosen token, then the alternatives it outranked."""

    def draw_row(draw, fonts, y: int) -> int:
        body, bold, _, _ = fonts
        draw.text((PAD, y), f"  {_quoted(step['token']):<14s}", fill=GREEN, font=bold)
        draw.text((PAD + 200, y), f"{step['logprob']:>8.3f}", fill=GREEN, font=bold)
        _bar(draw, PAD + 300, y, step["logprob"], GREEN)
        y += LINE_H
        if not revealed:
            return y
        for token, value in step["alternatives"][1:]:
            draw.text((PAD + 40, y), f"{_quoted(token):<14s}", fill=DIM, font=body)
            draw.text((PAD + 200, y), f"{value:>8.3f}", fill=DIM, font=body)
            _bar(draw, PAD + 300, y, value, BLUE)
            y += LINE_H
        return y

    return draw_row


def _text_row(text: str, colour, weight: str):
    """A plain line placed after the table, for the closing notes."""

    def draw_row(draw, fonts, y: int) -> int:
        body, bold, _, _ = fonts
        draw.text((PAD, y), text, fill=colour, font=bold if weight == "bold" else body)
        return y + LINE_H

    return draw_row


def build_frames(fonts, data: dict, model_name: str) -> tuple[list[Image.Image], list[int]]:
    """Type the call, score the prompt, then generate — one reveal per frame."""
    frames: list[Image.Image] = []
    durations: list[int] = []
    call = (
        'llm.generate(["The capital of France is"], SamplingParams(logprobs=5, prompt_logprobs=5))'
    )

    for cut in range(0, len(call) + 1, 6):
        frames.append(_frame(fonts, model_name, [(f"$ {call[:cut]}", PROMPT_FG, "body")], []))
        durations.append(40)

    prompt_header = [
        (f"$ {call}", PROMPT_FG, "body"),
        ("", TEXT_FG, "body"),
        ("prompt_logprobs — every prompt token scored by the same forward", CYAN, "bold"),
        (f"{'pos':>3s}  {'token':<14s}{'logprob':>10s}   probability", DIM, "body"),
    ]
    rows = data["prompt_rows"]
    for shown in range(1, len(rows) + 1):
        frames.append(
            _frame(
                fonts,
                model_name,
                prompt_header,
                [_prompt_row(i, entry) for i, entry in enumerate(rows[:shown])],
            )
        )
        durations.append(500)
    durations[-1] = 1400

    gen_header = [
        (f"$ {call}", PROMPT_FG, "body"),
        ("", TEXT_FG, "body"),
        ("logprobs — each drawn token, and the alternatives it beat", GREEN, "bold"),
        (f"  {'token':<14s}{'logprob':>10s}   probability", DIM, "body"),
    ]
    steps = data["steps"]
    for shown in range(1, len(steps) + 1):
        drawn = [_step_rows(step, revealed=False) for step in steps[: shown - 1]]
        drawn.append(_step_rows(steps[shown - 1], revealed=True))
        frames.append(_frame(fonts, model_name, gen_header, drawn))
        durations.append(900)

    closing_notes = [
        ("", TEXT_FG, "body"),
        (f"-> {data['text'].strip()!r}", TEXT_FG, "bold"),
        ("", TEXT_FG, "body"),
        (
            "prompt scores and sampled scores come from one forward — no rescoring pass",
            YELLOW,
            "bold",
        ),
        (
            "off by default; logprobs=5 costs ~0.6 ms/step, prompt_logprobs only prefill",
            DIM,
            "body",
        ),
    ]
    frames.append(
        _frame(
            fonts,
            model_name,
            gen_header,
            [_step_rows(step, revealed=False) for step in steps]
            + [_text_row(*note) for note in closing_notes],
        )
    )
    durations.append(4500)
    return frames, durations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen3-0.6B")
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} not found; the GIF renders a real run")
        return 1

    data = collect(str(model_dir))
    frames, durations = build_frames(_fonts(), data, model_dir.name)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(out),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"GIF saved to {out} ({len(frames)} frames, {sum(durations) / 1000:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
