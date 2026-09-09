"""Generate the quantization demo GIF for README.

Shows: CLI command → loading → benchmark results table with speedup bars.
Uses real benchmark data from `docs/benchmark_logs/quantization/quant_Qwen3-0.6B_20260823.json`.

Usage:
    python scripts/gen_quant_gif.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
OUTPUT = REPO_ROOT / "docs" / "images" / "quantization_benchmark.gif"
RESULTS_PATH = REPO_ROOT / "docs" / "benchmark_logs" / "quantization" / "quant_Qwen3-0.6B_all_20260823.json"

# Terminal palette
W, H = 1080, 640
TITLE_H, PAD, LINE_H = 36, 16, 24
BG = (14, 16, 20)
TITLE_BG = (32, 36, 44)
TITLE_FG = (222, 226, 232)
PROMPT_FG = (118, 214, 118)
DIM = (128, 136, 148)
TEXT_FG = (222, 226, 232)
CYAN = (86, 198, 224)
YELLOW = (253, 188, 64)
GREEN = (94, 193, 117)
BLUE = (110, 160, 226)
RED = (245, 99, 72)
BAR_BG = (38, 42, 52)


def _fonts():
    try:
        body = ImageFont.truetype(FONT_PATH, 14)
        bold = ImageFont.truetype(FONT_BOLD, 14)
        small = ImageFont.truetype(FONT_PATH, 12)
        big = ImageFont.truetype(FONT_BOLD, 16)
    except OSError:
        body = bold = small = big = ImageFont.load_default()
    return body, bold, small, big


def _draw_title_bar(draw, fonts):
    _, _, small, _ = fonts
    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text((12, 9), "rapid_llm  —  Quantization Benchmark (A10 GPU)", fill=TITLE_FG, font=small)
    for i, colour in enumerate([RED, YELLOW, GREEN]):
        draw.ellipse([W - 78 + i * 18, 11, W - 68 + i * 18, 21], fill=colour)


def _make_frame(fonts, lines, cursor_visible=True) -> Image.Image:
    body, bold, _, big = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _draw_title_bar(draw, fonts)

    y = TITLE_H + PAD
    for line in lines:
        if isinstance(line, tuple):
            text, colour, font_choice = line
            f = bold if font_choice == "bold" else (big if font_choice == "big" else body)
            draw.text((PAD, y), text, fill=colour, font=f)
        else:
            draw.text((PAD, y), line, fill=TEXT_FG, font=body)
        y += LINE_H
    if cursor_visible:
        draw.rectangle([PAD, y, PAD + 8, y + LINE_H - 4], fill=PROMPT_FG)
    return canvas


def _make_bar_frame(fonts, results, highlight_idx) -> Image.Image:
    """Draw the benchmark results with speedup bars."""
    body, bold, small, _ = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _draw_title_bar(draw, fonts)

    y = TITLE_H + PAD
    draw.text((PAD, y), "$ python benchmarks/engine/run.py quant --model-dir Qwen3-0.6B", fill=PROMPT_FG, font=body)
    y += LINE_H + 4
    draw.text((PAD, y), "Quantization Speed Benchmark (batch=4, greedy, max_gen=64)", fill=CYAN, font=bold)
    y += LINE_H + 8

    # Header
    draw.text((PAD, y), f"{'Config':<22s} {'TPOT':>7s} {'TPS':>8s} {'Speedup':>8s}  Bar", fill=DIM, font=body)
    y += LINE_H
    draw.line([(PAD, y), (W - PAD, y)], fill=DIM, width=1)
    y += 6

    max_tps = max(r["tps"] for r in results)
    bar_max_w = 320

    colours = [DIM, GREEN, BLUE, (160, 200, 240), YELLOW, (255, 140, 180)]
    labels = ["HF fp16", "lite fp16", "lite int8", "lite int8-blockwise", "lite fp8 W8A8", "lite smoothquant"]

    for i, r in enumerate(results):
        if i > highlight_idx:
            break
        label = labels[i] if i < len(labels) else r["config"]
        colour = colours[i] if i < len(colours) else TEXT_FG
        tpot = f"{r['tpot_ms']:.2f}ms"
        tps = f"{r['tps']:.0f}"
        speedup = f"{r['tps'] / results[0]['tps']:.1f}x" if results[0]["tps"] > 0 else "—"

        draw.text((PAD, y), f"{label:<22s}", fill=colour, font=bold)
        draw.text((PAD + 240, y), f"{tpot:>7s}", fill=TEXT_FG, font=body)
        draw.text((PAD + 310, y), f"{tps:>6s}", fill=TEXT_FG, font=body)
        draw.text((PAD + 380, y), f"{speedup:>6s}", fill=colour, font=bold)

        # Bar
        bar_x = PAD + 450
        bar_w = int((r["tps"] / max_tps) * bar_max_w)
        draw.rectangle([bar_x, y + 2, bar_x + bar_max_w, y + LINE_H - 6], fill=BAR_BG)
        draw.rectangle([bar_x, y + 2, bar_x + bar_w, y + LINE_H - 6], fill=colour)
        y += LINE_H + 6

    # Summary at bottom
    if highlight_idx >= len(results) - 1:
        y += 16
        draw.text(
            (PAD, y),
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            fill=DIM, font=body,
        )
        y += LINE_H
        draw.text((PAD, y), "✓ smoothquant W8A8: 6.7x faster (int8 tensor cores)", fill=(255, 140, 180), font=bold)
        y += LINE_H
        draw.text((PAD, y), "✓ rapid_llm fp16:  6.2x faster than HF transformers", fill=GREEN, font=bold)
        y += LINE_H
        draw.text((PAD, y), "✓ int8 per-channel:  6.1x faster, saves 1 GB memory", fill=BLUE, font=bold)
        y += LINE_H
        draw.text((PAD, y), "✓ fp8 W8A8:   3.0x faster (no weight dequant)", fill=YELLOW, font=bold)
        y += LINE_H + 4
        draw.text(
            (PAD, y),
            "Also supported: AWQ (int4) | GPTQ (int4) | fp8 KV cache",
            fill=DIM, font=small,
        )

    return canvas


def main():
    fonts = _fonts()

    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run benchmarks/engine/run.py quant first.")
        sys.exit(1)

    results = json.loads(RESULTS_PATH.read_text())
    frames: list[Image.Image] = []
    durations: list[int] = []

    # Phase 1: Show the CLI command being typed (5 frames)
    cmd = "$ python benchmarks/engine/run.py quant --model-dir Qwen3-0.6B --schemes fp16 int8 fp8"
    for i in range(0, len(cmd) + 1, 4):
        lines = [(cmd[:i], PROMPT_FG, "body")]
        frames.append(_make_frame(fonts, lines, cursor_visible=True))
        durations.append(40)

    # Phase 2: Loading animation (3 frames)
    for msg in [
        "Loading model Qwen3-0.6B (fp16)...",
        "Loading model Qwen3-0.6B (fp16)... done (0.3s)",
        "Running HF fp16 baseline...",
    ]:
        lines = [
            (cmd, PROMPT_FG, "body"),
            ("", TEXT_FG, "body"),
            (msg, DIM, "body"),
        ]
        frames.append(_make_frame(fonts, lines, cursor_visible=False))
        durations.append(600)

    # Phase 3: Results appearing one by one (4 frames with bars)
    for idx in range(len(results)):
        frames.append(_make_bar_frame(fonts, results, idx))
        durations.append(1200)

    # Phase 4: Final frame with summary (hold longer)
    final = _make_bar_frame(fonts, results, len(results) - 1)
    frames.append(final)
    durations.append(4000)

    # Save GIF
    frames[0].save(
        str(OUTPUT),
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"GIF saved to {OUTPUT} ({len(frames)} frames, {sum(durations)/1000:.1f}s)")


if __name__ == "__main__":
    main()
