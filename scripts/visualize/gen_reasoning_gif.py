"""Generate the streaming reasoning-parser demo GIF for README.

Shows what F8 actually delivers: one streamed reply arrives as
detokenizer-sized deltas, and ReasoningSplitter routes every delta to its
channel — the think block to ``reasoning_content``, what follows the closing
tag to ``content`` — consuming the tags themselves and holding any partial tag
until it can no longer complete.

The prompt deliberately opens the think tag itself, so a short generation
still drives the full state machine: the splitter is born *inside* a thinking
section (``starts_inside=True``, the deepseek_r1 contract) and has to catch
the closing tag mid-stream. Every character in the GIF comes from a real run
of the checkpoint — a hand-written transcript would defeat the purpose of
showing the parser works.

Usage:
    python scripts/gen_reasoning_gif.py
    python scripts/gen_reasoning_gif.py --model-dir my_weight/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from scripts._viz_lib import (
    AMBER,
    BG,
    BLUE,
    CYAN,
    DIM,
    GREEN,
    PROMPT_FG,
    TEXT_FG,
    YELLOW,
    FontPack,
    draw_title_bar,
    save_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT = REPO_ROOT / "docs" / "images" / "reasoning.gif"

#: Tokens per frame in the streaming section: fast enough to read as a stream,
#: slow enough that the channel switch (think -> content) lands as an event.
TOKENS_PER_FRAME = 2
WRAP_COLS = 122

W, H = 1080, 640
TITLE_H, PAD, LINE_H = 36, 16, 24

#: The user-facing request the GIF types out; the run underneath uses a prompt
#: that opens the think tag so the splitter starts inside a thinking section.
CALL_LINES = [
    "$ curl localhost:8000/v1/chat/completions -d '{",
    '    "stream": true, "reasoning_parser": "deepseek_r1",',
    '    "messages": [{"role": "user", "content": "What is 2+2?"}]}\'',
]


def _fonts():
    fp = FontPack.mono(14, 14, 12)
    return fp.body, fp.bold, fp.small


def collect(model_dir: str) -> dict:
    """One greedy run, replayed through the splitter one detokenized token at a time.

    That replay is the parser's real usage: a server feeds it the same
    increments the detokenizer emits, and the GIF shows exactly what comes
    back per step — including the steps where a delta is swallowed whole
    because it might complete a tag.
    """
    from rapid_llm.engine.llm import LLM
    from rapid_llm.engine.sampler import SamplingParams
    from rapid_llm.serving.reasoning import _OPEN as THINK
    from rapid_llm.serving.reasoning import for_family

    prompt = f"User: What is 2+2?\nAssistant: {THINK}\nThe user asks"
    llm = LLM(model=model_dir, max_seq_len=256, max_gpu_num_blocks=2048, use_cuda_graph=False)
    output = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0.0,
            max_gen_len=64,
            repetition_penalty=1.0,
            stop_on_repeat=False,
            logprobs=1,
        ),
    )[0]
    decode = llm.tokenizer.decode

    splitter = for_family("deepseek_r1")
    steps = []
    for record in output.outputs[0].logprobs or []:
        chunk = decode([record.token_id])
        reasoning, content = splitter.feed(chunk)
        steps.append({"chunk": chunk, "reasoning": reasoning, "content": content})
    tail_reasoning, tail_content = splitter.finish()
    return {
        "steps": steps,
        "tail_reasoning": tail_reasoning,
        "tail_content": tail_content,
        "text": output.outputs[0].text,
    }


def _title_bar(draw, fonts, model_name: str):
    _, _, small = fonts
    draw_title_bar(
        draw, W, f"rapid_llm  —  streaming reasoning parser  ({model_name})", small=small
    )


def _wrapped(draw, fonts, text: str, x: int, y: int, colour) -> int:
    """Draw wrapped text; returns the y after the last line."""
    body, _, _ = fonts
    for line in textwrap.wrap(text, width=WRAP_COLS, break_long_words=False) or [""]:
        draw.text((x, y), line, fill=colour, font=body)
        y += LINE_H
    return y


def _cursor(draw, fonts, x: int, y: int):
    """A block cursor: the frame-level cue that the stream is still open."""
    body, _, _ = fonts
    draw.text((x, y), "▌", fill=CYAN, font=body)


def _stream_frame(
    fonts,
    model_name: str,
    reasoning: str,
    content: str,
    *,
    cursor_on: bool,
    tail: list[tuple[str, object, str]] | None = None,
) -> Image.Image:
    """One frame: the typed call above, the two channels below, optional tail notes."""
    body, bold, _ = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _title_bar(draw, fonts, model_name)

    y = TITLE_H + PAD
    for line in CALL_LINES:
        draw.text((PAD, y), line, fill=PROMPT_FG, font=body)
        y += LINE_H
    y += 10

    draw.text(
        (PAD, y),
        "stream — deltas land in their channel; tags never leak through",
        fill=CYAN,
        font=bold,
    )
    y += LINE_H + 4

    draw.text((PAD, y), "delta.reasoning_content", fill=YELLOW, font=bold)
    y += LINE_H
    lines = textwrap.wrap(reasoning, width=WRAP_COLS, break_long_words=False) or [""]
    for line in lines:
        draw.text((PAD, y), line, fill=AMBER, font=body)
        y += LINE_H
    if cursor_on and not tail:
        _cursor(draw, fonts, PAD + 8, y - LINE_H)
    y += 8

    draw.text((PAD, y), "delta.content", fill=GREEN, font=bold)
    y += LINE_H
    if content:
        for line in textwrap.wrap(content, width=WRAP_COLS, break_long_words=False):
            draw.text((PAD, y), line, fill=GREEN, font=body)
            y += LINE_H
    elif cursor_on and not tail:
        _cursor(draw, fonts, PAD + 8, y)
    y += 6

    for text, colour, weight in tail or []:
        draw.text((PAD, y), text, fill=colour, font=bold if weight == "bold" else body)
        y += LINE_H
    return canvas


def build_frames(fonts, data: dict, model_name: str) -> tuple[list[Image.Image], list[int]]:
    """Type the call, stream the reply, then close with the finish frame and the facts."""
    frames: list[Image.Image] = []
    durations: list[int] = []

    call = "\n".join(CALL_LINES)
    for cut in range(0, len(call) + 1, 7):
        shown = call[:cut].split("\n")
        body, _, _ = fonts
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        _title_bar(draw, fonts, model_name)
        y = TITLE_H + PAD
        for line in shown:
            draw.text((PAD, y), line, fill=PROMPT_FG, font=body)
            y += LINE_H
        if cut < len(call):
            _cursor(draw, fonts, PAD + 8, y - LINE_H)
        frames.append(canvas)
        durations.append(40)
    durations[-1] = 900

    steps = data["steps"]
    reasoning = content = ""
    for start in range(0, len(steps), TOKENS_PER_FRAME):
        for step in steps[start : start + TOKENS_PER_FRAME]:
            reasoning += step["reasoning"]
            content += step["content"]
        frames.append(_stream_frame(fonts, model_name, reasoning, content, cursor_on=True))
        durations.append(320)
    # The closing tag arrives as its own deltas; give the channel switch a beat.
    durations[-1] = 1100

    reasoning += data["tail_reasoning"]
    content += data["tail_content"]
    finish_frame = [
        ("", TEXT_FG, "body"),
        ('data: {"delta": {}, "finish_reason": "stop"}', BLUE, "bold"),
        ("", TEXT_FG, "body"),
        ("the parser is declared per request, not per deployment — one server,", TEXT_FG, "body"),
        ("unchanged, serves R1-style and direct models side by side", TEXT_FG, "body"),
        ("streamed frames concatenated == the one-shot message (tested as an axiom)", DIM, "body"),
        ("cost: ~0.11 µs/token; a delta that might complete a tag is held, not shown", DIM, "body"),
    ]
    frames.append(
        _stream_frame(
            fonts,
            model_name,
            reasoning,
            content,
            cursor_on=False,
            tail=finish_frame,
        )
    )
    durations.append(5200)
    return frames, durations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen3-1.7B")
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        print(f"ERROR: {model_dir} not found; the GIF renders a real run")
        return 1

    data = collect(str(model_dir))
    frames, durations = build_frames(_fonts(), data, model_dir.name)

    out = Path(args.output)
    save_gif(frames, out, duration=durations)
    print(f"GIF saved to {out} ({len(frames)} frames, {sum(durations) / 1000:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
