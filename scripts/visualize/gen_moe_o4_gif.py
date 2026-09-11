"""O4 MoE grouped-GEMM autotune collect: heuristic vs measured tiles + speedup.

Reads the canonical benchmark JSON and produces a dark-theme GIF showing:
  - Per-format (bf16, fp8, int8) × per-token-count (1, 8, 64, 512, 4096) grid
  - Each cell: heuristic tile | tuned tile | speedup bar (green >1, red <1)
  - Final sweep: all 15 speedup bars together

Usage:
    python scripts/gen_moe_o4_gif.py
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Dark theme palette
BG = (18, 18, 24)
FG = (220, 220, 230)
DIM = (80, 80, 100)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
AMBER = (241, 196, 15)
BLUE = (52, 152, 219)
HEADER_BG = (30, 30, 45)

# Layout
W, H = 900, 560
MARGIN = 40
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

try:
    font = ImageFont.truetype(FONT_PATH, 14)
    font_small = ImageFont.truetype(FONT_PATH, 12)
    font_title = ImageFont.truetype(FONT_BOLD_PATH, 20)
    font_label = ImageFont.truetype(FONT_BOLD_PATH, 14)
except OSError:
    font = ImageFont.load_default()
    font_small = font
    font_title = font
    font_label = font

FORMATS = ["bf16", "fp8_w8a16", "int8_w8a16"]
FORMAT_LABELS = {"bf16": "bf16", "fp8_w8a16": "fp8 w8a16", "int8_w8a16": "int8 w8a16"}
TOKENS = [1, 8, 64, 512, 4096]


def load_latest_json() -> dict:
    pattern = str(Path("docs/benchmark_logs/kernels/moe_o4_*.json"))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("No moe_o4_*.json found in docs/benchmark_logs/kernels/")
    with open(files[-1]) as f:
        return json.load(f)


def build_grid(data: dict) -> list[dict]:
    """Flatten JSON rows into a grid of cells."""
    rows = data["results"]["rows"]
    grid = []
    for row in rows:
        grid.append({
            "scheme": row["scheme"],
            "tokens": row["tokens"],
            "heur_tile": row["heuristic_tile"],
            "tuned_tile": row["tuned_tile"],
            "speedup": row["speedup"],
            "tile_changed": row["tile_changed"],
        })
    return grid


def draw_cell(draw: ImageDraw.Draw, x: int, y: int, w: int, h: int, cell: dict, reveal: bool):
    """Draw one grid cell: heuristic tile | tuned tile | speedup bar."""
    # Cell background
    draw.rectangle([x, y, x + w, y + h], fill=HEADER_BG, outline=DIM)

    if not reveal:
        return

    # Heuristic tile (left third)
    left_w = w // 3
    draw.text((x + 4, y + 4), "heur", fill=DIM, font=font_small)
    draw.text((x + 4, y + 20), cell["heur_tile"], fill=FG, font=font)

    # Tuned tile (middle third)
    mid_x = x + left_w
    draw.text((mid_x + 4, y + 4), "tuned", fill=DIM, font=font_small)
    draw.text((mid_x + 4, y + 20), cell["tuned_tile"], fill=FG, font=font)

    # Speedup bar (right third)
    right_x = x + 2 * left_w
    bar_w = w - 2 * left_w - 8
    bar_h = h - 30
    bar_x = right_x + 4
    bar_y = y + 20

    # Background bar (1.0x baseline)
    baseline_x = bar_x + bar_w // 2
    draw.line([baseline_x, bar_y, baseline_x, bar_y + bar_h], fill=DIM, width=1)

    # Speedup bar
    sp = cell["speedup"]
    if sp >= 1.0:
        color = GREEN
        bar_len = int((sp - 1.0) * bar_w * 2)  # 1.5x = full right half
        bar_len = min(bar_len, bar_w // 2)
        draw.rectangle([baseline_x, bar_y + 5, baseline_x + bar_len, bar_y + bar_h - 5], fill=color)
    else:
        color = RED
        bar_len = int((1.0 - sp) * bar_w * 2)
        bar_len = min(bar_len, bar_w // 2)
        draw.rectangle([baseline_x - bar_len, bar_y + 5, baseline_x, bar_y + bar_h - 5], fill=color)

    # Speedup text
    sp_text = f"{sp:.2f}x"
    draw.text((bar_x, y + 4), sp_text, fill=color, font=font_label)


def draw_frame(grid: list[dict], revealed: int, frame_idx: int, total_frames: int) -> Image.Image:
    """Draw one frame with `revealed` cells visible."""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Title
    title = "O4 MoE grouped-GEMM autotune: heuristic vs measured tiles"
    draw.text((MARGIN, 10), title, fill=FG, font=font_title)

    subtitle = f"A10 collect round  |  frame {frame_idx + 1}/{total_frames}"
    draw.text((MARGIN, 35), subtitle, fill=DIM, font=font_small)

    # Grid layout
    grid_x = MARGIN
    grid_y = 70
    cell_w = (W - 2 * MARGIN) // len(TOKENS)
    cell_h = (H - grid_y - 60) // len(FORMATS)

    # Column headers (token counts)
    for j, tokens in enumerate(TOKENS):
        cx = grid_x + j * cell_w + cell_w // 2
        draw.text((cx - 15, grid_y - 18), f"t={tokens}", fill=FG, font=font_label)

    # Row headers (formats) and cells
    for i, fmt in enumerate(FORMATS):
        ry = grid_y + i * cell_h + cell_h // 2 - 8
        draw.text((grid_x - 35, ry), FORMAT_LABELS[fmt], fill=FG, font=font_label)

        for j, tokens in enumerate(TOKENS):
            cx = grid_x + j * cell_w
            cy = grid_y + i * cell_h
            cell_idx = i * len(TOKENS) + j
            reveal = cell_idx < revealed
            cell = next((c for c in grid if c["scheme"] == fmt and c["tokens"] == tokens), None)
            if cell:
                draw_cell(draw, cx, cy, cell_w - 2, cell_h - 2, cell, reveal)

    # Footer
    footer_y = H - 25
    if revealed == len(grid):
        geomean = math.exp(sum(math.log(c["speedup"]) for c in grid) / len(grid))
        footer = f"geomean {geomean:.3f}x  |  best {max(grid, key=lambda c: c['speedup'])['speedup']:.2f}x"
        draw.text((MARGIN, footer_y), footer, fill=GREEN, font=font_label)
    else:
        footer = "revealing cells..."
        draw.text((MARGIN, footer_y), footer, fill=DIM, font=font_small)

    return img


def main():
    data = load_latest_json()
    grid = build_grid(data)

    # Frames: intro + reveal each cell + final hold
    frames = []
    total_cells = len(grid)

    # Intro frame (0 cells revealed)
    frames.append(draw_frame(grid, 0, 0, total_cells + 2))

    # Reveal cells one by one
    for i in range(1, total_cells + 1):
        frames.append(draw_frame(grid, i, i, total_cells + 2))

    # Final hold frame
    frames.append(draw_frame(grid, total_cells, total_cells + 1, total_cells + 2))

    # Save GIF
    out_path = Path("docs/images/moe_o4.gif")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=800,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out_path} ({len(frames)} frames, {out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
