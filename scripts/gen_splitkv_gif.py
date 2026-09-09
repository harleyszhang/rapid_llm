"""Generate the O8 adaptive split-kv decode-attention GIF for README / release.

Renders the *measured* A/B sweep from ``docs/benchmark_logs/kernels/splitkv_o8_*.json`` —
no synthetic numbers. The story it tells is occupancy, not arithmetic:

    stage-1 grid = (batch, num_heads, num_partitions), one warp per program.
    One SM wave on A10 = 72 SMs * 16 resident blocks = 1152 block slots.

For batch=1 short/medium context a fixed ``PARTITION_SIZE=128`` launches far
fewer than 1152 stage-1 blocks, so most SMs sit idle. The adaptive policy splits
the KV history finer to fill the wave (batch=1 seq=512: 128 -> 512 blocks,
measured 1.80x). Every shape that already saturates the GPU keeps the baseline,
so those cells are exactly 1.0x — the grids render identically full.

Two scenes: an SM-slot occupancy grid per shape (fixed vs adaptive), then a
diverging speedup sweep around the 1.0x no-regression line.

Usage:
    python scripts/gen_splitkv_gif.py
    python scripts/gen_splitkv_gif.py --json docs/benchmark_logs/kernels/splitkv_o8_<stamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
OUTPUT = REPO_ROOT / "docs" / "images" / "splitkv_o8.gif"

W, H = 1180, 470
TITLE_H, PAD = 36, 20
BG = (14, 16, 20)
TITLE_BG = (32, 36, 44)
TITLE_FG = (222, 226, 232)
DIM = (128, 136, 148)
TEXT_FG = (222, 226, 232)
CYAN = (86, 198, 224)
AMBER = (226, 184, 92)
GREEN = (94, 193, 117)
RED = (245, 99, 72)
CELL_BG = (28, 32, 40)
GRID_LINE = (44, 50, 60)

#: Occupancy grid = one SM wave: 72 SMs x 16 resident blocks (num_warps=1).
GRID_COLS, GRID_ROWS, CELL = 72, 16, 7
GRID_W, GRID_H = GRID_COLS * CELL, GRID_ROWS * CELL
WAVE = GRID_COLS * GRID_ROWS  # 1152 block slots
LEFT_X, RIGHT_X = 24, 636
GRID_Y = 132


def _fonts():
    try:
        return (
            ImageFont.truetype(FONT_PATH, 14),
            ImageFont.truetype(FONT_BOLD, 14),
            ImageFont.truetype(FONT_PATH, 12),
            ImageFont.truetype(FONT_BOLD, 30),
            ImageFont.truetype(FONT_BOLD, 16),
        )
    except OSError:
        d = ImageFont.load_default()
        return d, d, d, d, d


def _title_bar(draw, fonts, subtitle):
    _, _, small, _, _ = fonts
    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text((12, 9), subtitle, fill=TITLE_FG, font=small)
    for i, colour in enumerate([RED, AMBER, GREEN]):
        draw.ellipse([W - 78 + i * 18, 11, W - 68 + i * 18, 21], fill=colour)


def _draw_grid(draw, x, y, filled, colour, fonts):
    """Fill ``filled`` of the WAVE cells row-major; annotate spillover waves."""
    _, _, small, _, _ = fonts
    draw.rectangle([x, y, x + GRID_W, y + GRID_H], fill=CELL_BG)
    shown = min(filled, WAVE)
    for i in range(shown):
        cx = x + (i % GRID_COLS) * CELL
        cy = y + (i // GRID_COLS) * CELL
        draw.rectangle([cx, cy, cx + CELL - 1, cy + CELL - 1], fill=colour)
    for c in range(GRID_COLS + 1):
        gx = x + c * CELL
        draw.line([gx, y, gx, y + GRID_H], fill=GRID_LINE)
    for r in range(GRID_ROWS + 1):
        gy = y + r * CELL
        draw.line([x, gy, x + GRID_W, gy], fill=GRID_LINE)
    pct = 100.0 * filled / WAVE
    if filled >= WAVE:
        waves = filled / WAVE
        tag = f"{filled} blocks  =  {waves:.1f} waves (saturated)"
    else:
        tag = f"{filled} blocks  =  {pct:.0f}% of one wave"
    draw.text((x, y + GRID_H + 8), tag, fill=colour, font=small)


def _shape_frame(row, phase, fonts, geo):
    """phase 0: only the fixed grid lit. phase 1: adaptive lit + speedup badge."""
    _, bold, small, big, mid = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _title_bar(
        draw,
        fonts,
        "lite-llama  —  O8 adaptive split-kv decode attention  (A10, 1 wave = 1152 block slots)",
    )

    b, s = row["batch"], row["seq_len"]
    under = row["fixed_stage1_blocks"] < WAVE
    tagline = "single-request chat decode" if b == 1 and under else (
        "grid already saturates the GPU" if not under else "underfilled grid"
    )
    draw.text((PAD, 50), f"batch={b}   seq_len={s}   ({tagline})", fill=CYAN, font=mid)
    draw.text(
        (PAD, 78),
        f"head geometry: {row['num_q_heads']} q-heads / {row['num_kv_heads']} kv-heads "
        f"/ dim {row['head_dim']}   |   stage-1 blocks = batch * heads * ceil(seq / PARTITION)",
        fill=DIM,
        font=small,
    )

    # Column headers
    draw.text((LEFT_X, GRID_Y - 24), "fixed  PARTITION_SIZE=128", fill=AMBER, font=bold)
    draw.text((RIGHT_X, GRID_Y - 24), f"adaptive  PARTITION_SIZE={row['adaptive_partition_size']}", fill=GREEN, font=bold)

    _draw_grid(draw, LEFT_X, GRID_Y, row["fixed_stage1_blocks"], AMBER, fonts)
    if phase >= 1:
        _draw_grid(draw, RIGHT_X, GRID_Y, row["adaptive_stage1_blocks"], GREEN, fonts)
        # Middle arrow + speedup badge
        mid_x = (LEFT_X + GRID_W + RIGHT_X) // 2
        draw.text((mid_x - 14, GRID_Y + 24), "\u2192", fill=TEXT_FG, font=big)
        sp = row["speedup"]
        changed = row["fixed_stage1_blocks"] != row["adaptive_stage1_blocks"]
        # Only colour a cell when the policy actually moved the grid; an
        # unchanged grid is the baseline by construction, so its delta is
        # run-to-run noise and must read neutral, not as a win or regression.
        col = GREEN if (changed and sp > 1.03) else (RED if (changed and sp < 0.97) else DIM)
        draw.text((mid_x - 34, GRID_Y + 62), f"{sp:.2f}\u00d7", fill=col, font=big)
    else:
        draw.rectangle([RIGHT_X, GRID_Y, RIGHT_X + GRID_W, GRID_Y + GRID_H], fill=CELL_BG)
        for c in range(GRID_COLS + 1):
            draw.line([RIGHT_X + c * CELL, GRID_Y, RIGHT_X + c * CELL, GRID_Y + GRID_H], fill=GRID_LINE)
        for r in range(GRID_ROWS + 1):
            draw.line([RIGHT_X, GRID_Y + r * CELL, RIGHT_X + GRID_W, GRID_Y + r * CELL], fill=GRID_LINE)
        draw.text((RIGHT_X, GRID_Y + GRID_H + 8), "adaptive policy …", fill=DIM, font=small)

    # Timing / verdict strip
    y = GRID_Y + GRID_H + 40
    draw.line([(PAD, y), (W - PAD, y)], fill=GRID_LINE)
    y += 12
    if phase >= 1:
        fus, aus = row["fixed_us"], row["adaptive_us"]
        verdict = (
            f"{fus:.1f}\u00b5s \u2192 {aus:.1f}\u00b5s   ({row['speedup']:.2f}\u00d7)"
        )
        if row["fixed_stage1_blocks"] == row["adaptive_stage1_blocks"]:
            note = "same grid -> policy returns the baseline; cell is 1.0x by construction (deltas are run-to-run noise)"
            vcol = DIM
        else:
            note = "finer split fills idle SMs -> real speedup, output bit-identical (exact online-softmax combine)"
            vcol = GREEN
        draw.text((PAD, y), verdict, fill=vcol, font=bold)
        draw.text((PAD, y + 24), note, fill=DIM, font=small)
    else:
        draw.text((PAD, y), f"fixed-128 launches {row['fixed_stage1_blocks']} stage-1 blocks …", fill=AMBER, font=bold)

    draw.text(
        (PAD, H - 26),
        f"geomean over the 9-shape sweep = {geo:.3f}x   |   switch: LITE_LLAMA_SPLITKV=adaptive|fixed   |   measured, not modelled",
        fill=DIM,
        font=small,
    )
    return canvas


def _sweep_frame(rows, upto, fonts, geo):
    body, bold, small, _, mid = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _title_bar(draw, fonts, "lite-llama  —  O8 split-kv speedup sweep  (adaptive vs fixed-128, A10)")

    draw.text((PAD, 52), "measured speedup per decode shape", fill=CYAN, font=mid)
    draw.text((PAD, 78), "bars right of the 1.00x line = adaptive wins; on it = baseline kept (no regression)", fill=DIM, font=small)

    x0, y0 = 250, 112
    row_h = 32
    scale = 300.0  # px per 1.0x deviation
    for i, r in enumerate(rows):
        if i > upto:
            break
        y = y0 + i * row_h
        label = f"b{r['batch']:<2d}_s{r['seq_len']:<5d}"
        draw.text((PAD, y + 4), label, fill=TEXT_FG, font=body)
        sp = r["speedup"]
        changed = r["fixed_stage1_blocks"] != r["adaptive_stage1_blocks"]
        col = GREEN if (changed and sp > 1.03) else (RED if (changed and sp < 0.97) else DIM)
        dev = sp - 1.0
        bx1 = x0 + dev * scale
        draw.line([x0, y + 2, x0, y + row_h - 8], fill=GRID_LINE)  # 1.0x axis
        draw.rectangle(
            [min(x0, bx1), y + 6, max(x0, bx1), y + row_h - 12],
            fill=col,
        )
        draw.text((max(x0, bx1) + 10, y + 4), f"{sp:.2f}\u00d7", fill=col, font=bold)
        blocks = f"{r['fixed_stage1_blocks']}\u2192{r['adaptive_stage1_blocks']} blocks"
        draw.text((x0 - 150, y + 4), "", fill=DIM, font=small)
        draw.text((W - 250, y + 4), blocks, fill=DIM, font=small)

    if upto >= len(rows) - 1:
        y = y0 + len(rows) * row_h + 8
        draw.line([(PAD, y), (W - PAD, y)], fill=GRID_LINE)
        draw.text((PAD, y + 12), f"geomean = {geo:.3f}x", fill=GREEN, font=bold)
        best = max(rows, key=lambda r: r["speedup"])
        draw.text(
            (PAD + 220, y + 12),
            f"best b{best['batch']}_s{best['seq_len']} = {best['speedup']:.2f}x (128\u2192{best['adaptive_partition_size']} split)",
            fill=CYAN,
            font=bold,
        )
        draw.text(
            (PAD, y + 38),
            "7 of 9 shapes keep PARTITION=128 (already saturate the GPU) -> ~1.0x; their spread is run-to-run noise (same grid both arms).",
            fill=DIM,
            font=small,
        )
    return canvas


def _intro_frame(fonts, geo):
    body, bold, small, _, _ = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    _title_bar(draw, fonts, "lite-llama  —  O8 adaptive split-kv decode attention")
    lines = [
        ("FlashDecoding splits the KV history into PARTITION_SIZE-wide chunks,", TEXT_FG, "body"),
        ("runs one stage-1 program per (batch, head, partition), then combines", TEXT_FG, "body"),
        ("the partials with an exact online-softmax stage-2.", TEXT_FG, "body"),
        ("", TEXT_FG, "body"),
        ("One SM wave on A10 = 72 SMs x 16 resident blocks = 1152 block slots.", CYAN, "bold"),
        ("A fixed PARTITION=128 launches batch*heads*ceil(seq/128) blocks --", TEXT_FG, "body"),
        ("for batch=1 short context that is far below 1152, so SMs sit idle.", TEXT_FG, "body"),
        ("", TEXT_FG, "body"),
        ("O8 picks the split from the decode shape: fill the wave when underfilled,", GREEN, "bold"),
        ("keep the baseline when already saturated. Numerics never move.", GREEN, "bold"),
    ]
    y = TITLE_H + 26
    for text, colour, kind in lines:
        f = bold if kind == "bold" else body
        draw.text((PAD, y), text, fill=colour, font=f)
        y += 26
    # A demo underfilled grid to seed the visual language
    draw.text((PAD, y + 10), "one wave = 1152 slots:", fill=DIM, font=small)
    _draw_grid(draw, LEFT_X, y + 30, 128, AMBER, fonts)
    draw.text((RIGHT_X, y + 30 - 24), "", fill=DIM, font=small)
    _draw_grid(draw, RIGHT_X, y + 30, 512, GREEN, fonts)
    draw.text((LEFT_X, y + 30 - 24), "fixed-128 @ b1_s512", fill=AMBER, font=small)
    draw.text((RIGHT_X, y + 30 - 24), "adaptive-32 @ b1_s512", fill=GREEN, font=small)
    draw.text((PAD, H - 26), f"geomean {geo:.3f}x over a 9-shape sweep  |  measured on A10", fill=DIM, font=small)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="splitkv_o8 benchmark JSON (default: newest)")
    ap.add_argument("--out", default=str(OUTPUT))
    args = ap.parse_args()

    if args.json:
        path = Path(args.json)
    else:
        cands = sorted((REPO_ROOT / "docs" / "benchmark_logs" / "kernels").glob("splitkv_o8_*.json"))
        if not cands:
            print("ERROR: no splitkv_o8_*.json found; run benchmarks/kernels/bench_splitkv.py first.")
            return 1
        path = cands[-1]

    data = json.loads(path.read_text())
    rows = data["results"]["rows"]
    geo = data["results"]["geomean_speedup"]
    print(f"loaded {len(rows)} shapes from {path.name} (geomean {geo:.3f}x)")

    fonts = _fonts()
    frames: list[Image.Image] = []
    durations: list[int] = []

    frames.append(_intro_frame(fonts, geo))
    durations.append(2600)

    for row in rows:
        changed = row["fixed_stage1_blocks"] != row["adaptive_stage1_blocks"]
        frames.append(_shape_frame(row, 0, fonts, geo))
        durations.append(650 if changed else 400)
        frames.append(_shape_frame(row, 1, fonts, geo))
        durations.append(1500 if changed else 750)

    for upto in range(len(rows)):
        frames.append(_sweep_frame(rows, upto, fonts, geo))
        durations.append(380)
    frames.append(_sweep_frame(rows, len(rows) - 1, fonts, geo))
    durations.append(3800)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(
        out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(pal)} frames, {sum(durations)/1000:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
