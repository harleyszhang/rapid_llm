"""Generate the rapid_llm vs vLLM comparison image.

Three panels:
  1. the INT4 memory addressing each MoE kernel uses for the same byte-packed
     weight, which is where the vectorisation difference comes from;
  2. the measured INT4 kernel cost, including the four-step evolution and
     vLLM's Triton fallback (docs/benchmark_logs/kernels/fused_moe_h100_20260902_int4byte.json);
  3. the end-to-end rapid_llm vs vLLM comparison and the two orthogonal drivers
     behind it (docs/benchmark_models.md, H100 section).

Usage:
    python scripts/gen_int4_unpack_gif.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts._viz_lib import BG, DIM, GREEN, RED, BLUE, AMBER, PURPLE, PANEL_BG, PANEL_EDGE, draw_panel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT = REPO_ROOT / "docs" / "images" / "rapid_vs_vllm.png"
RESULTS = REPO_ROOT / "docs/benchmark_logs/kernels/fused_moe_h100_20260902_int4byte.json"

W, H = 1440, 1720
FG = (226, 232, 240)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size + (2 if bold else 0))


def panel(draw, box, title):
    draw_panel(draw, box, title)


def byte_cell(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    size: int,
    label: str,
    colour: tuple[int, int, int],
    text_colour: tuple[int, int, int] = BG,
) -> None:
    draw.rounded_rectangle([x, y, x + size, y + size], radius=4, fill=colour, outline=PANEL_EDGE)
    f = font(max(11, size // 3), True)
    bb = draw.textbbox((0, 0), label, font=f)
    draw.text(
        (x + (size - bb[2] + bb[0]) // 2, y + (size - bb[3] + bb[1]) // 2),
        label,
        font=f,
        fill=text_colour,
    )


def build() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((28, 22), "INT4 MoE unpack: rapid_llm vs vLLM", font=font(30, True), fill=FG)
    draw.text(
        (28, 62),
        "same byte-packed weight (2 nibbles per uint8), two ways of addressing it",
        font=font(16),
        fill=DIM,
    )

    # ------------------------------------------------------------------ #
    # Panel 1: addressing
    # ------------------------------------------------------------------ #
    panel(draw, (24, 100, W - 24, 620), "1  memory addressing — where vectorisation is won or lost")

    # --- vLLM side ---
    lx, ly = 48, 152
    draw.text((lx, ly), "vLLM  fused_moe_kernel_gptq_awq", font=font(18, True), fill=RED)
    draw.text(
        (lx, ly + 26),
        "b_ptrs += (offs_k // 2) * stride_bk ;  shifter = (offs_k % 2) * 4",
        font=font(14),
        fill=DIM,
    )

    # storage row
    sy = ly + 62
    draw.text((lx, sy), "storage (bytes):", font=font(14, True), fill=DIM)
    for i in range(8):
        byte_cell(draw, lx + 150 + i * 52, sy - 4, 44, f"b{i}", (74, 84, 98), FG)

    # logical k rows pointing at replicated bytes
    ky = sy + 60
    draw.text((lx, ky), "tile rows (logical k):", font=font(14, True), fill=DIM)
    targets = [0, 0, 1, 1, 2, 2, 3, 3]
    for i in range(8):
        ty = ky + i * 34
        draw.text((lx + 150, ty), f"k={i}", font=font(13), fill=FG)
        tgt = lx + 150 + targets[i] * 52 + 22
        colour = AMBER if i % 2 else BLUE
        draw.line([(lx + 196, ty + 8), (tgt, sy + 18)], fill=colour, width=2)
        byte_cell(draw, lx + 420 + i * 40, ty, 32, f"k{i}", colour, BG)

    draw.text(
        (lx, ky + 8 * 34 + 8),
        "k address sequence: 0,0,1,1,2,2,... -> non-affine in the tile's",
        font=font(14),
        fill=AMBER,
    )
    draw.text(
        (lx, ky + 8 * 34 + 30),
        "lowest dim -> no vector load; 128 scalar ld.global.b8; every byte",
        font=font(14),
        fill=AMBER,
    )
    draw.text(
        (lx, ky + 8 * 34 + 52),
        "loaded twice; per-element variable shift; fp32 dequant in-loop",
        font=font(14),
        fill=AMBER,
    )

    # --- rapid_llm side ---
    rx = W // 2 + 24
    draw.text(
        (rx, ly), "rapid_llm  _fused_moe_kernel (QUANT_MODE=3)", font=font(18, True), fill=GREEN
    )
    draw.text(
        (rx, ly + 26),
        "byte plane [BLOCK_K//2, BLOCK_N] ;  EVEN_K ;  two half-K dots",
        font=font(14),
        fill=DIM,
    )

    draw.text((rx, sy), "storage (bytes):", font=font(14, True), fill=DIM)
    for i in range(8):
        byte_cell(draw, rx + 150 + i * 52, sy - 4, 44, f"b{i}", (74, 84, 98), FG)

    draw.text((rx, ky), "tile rows (half-K planes):", font=font(14, True), fill=DIM)
    for i in range(4):
        ty = ky + i * 34
        draw.text((rx + 150, ty), f"row{i}", font=font(13), fill=FG)
        tgt = rx + 150 + i * 52 + 22
        draw.line([(rx + 206, ty + 8), (tgt, sy + 18)], fill=GREEN, width=2)
        byte_cell(draw, rx + 380, ty, 32, "lo", GREEN, BG)
        byte_cell(draw, rx + 418, ty, 32, "hi", PURPLE, BG)

    draw.text(
        (rx, ky + 4 * 34 + 8),
        "each byte read once, address affine in the lowest dim -> vector",
        font=font(14),
        fill=GREEN,
    )
    draw.text(
        (rx, ky + 4 * 34 + 30),
        "load + cp.async; nibble split is a constant-shift register op;",
        font=font(14),
        fill=GREEN,
    )
    draw.text(
        (rx, ky + 4 * 34 + 52),
        "dequant exact in the integer domain, scale stays in the epilogue",
        font=font(14),
        fill=GREEN,
    )

    # ------------------------------------------------------------------ #
    # Panel 2: measured cost
    # ------------------------------------------------------------------ #
    with open(RESULTS, encoding="utf-8") as fh:
        data = json.load(fh)
    rows = {(r["impl"], r["case"]): r["us"] for r in data["rows"] if r["us"] is not None}

    py = 648
    panel(
        draw, (24, py, W - 24, H - 24), "2  measured cost at t4096 (H100, Qwen3-30B-A3B geometry)"
    )

    steps = [
        ("vLLM Triton fallback (extracted, same geometry)", 15.0, RED, "13-18 ms measured"),
        ("int32 8-nibble packing (before)", 1.92, AMBER, "1.92 ms"),
        ("byte layout + vLLM replicated addressing", 7.35, AMBER, "7.35 ms  (3.8x regression)"),
        ("dense byte load + dual half-K dot", 3.31, BLUE, "3.31 ms"),
        ("EVEN_K + register nibble split (final)", 1.70, GREEN, "1.70 ms"),
        ("reference: int8 same tier", 1.148, DIM, "1.15 ms"),
        ("reference: bf16 same tier", 1.062, DIM, "1.06 ms"),
    ]

    bx, bw_max = 470, 620
    scale = bw_max / 15.0
    by = py + 54
    for i, (label, ms, colour, note) in enumerate(steps):
        yy = by + i * 46
        draw.text((48, yy + 6), label, font=font(14), fill=FG)
        wpx = max(6, int(ms * scale))
        draw.rounded_rectangle([bx, yy, bx + wpx, yy + 26], radius=4, fill=colour)
        draw.text((bx + wpx + 10, yy + 5), note, font=font(13, True), fill=colour)

    fy = by + len(steps) * 46 + 12
    draw.text(
        (48, fy),
        "vLLM's production int4 path is the Marlin CUDA kernel; the Triton kernel above is its",
        font=font(14),
        fill=DIM,
    )
    draw.text(
        (48, fy + 22),
        "sm75-compat fallback (their own note: removable once sm75 support is dropped).",
        font=font(14),
        fill=DIM,
    )
    draw.text(
        (48, fy + 44),
        "At t4096 num_valid_tokens/num_experts = 256 >> 6, so a model not routed to Marlin",
        font=font(14),
        fill=DIM,
    )
    draw.text(
        (48, fy + 66),
        "really does run that Triton kernel -- the comparison is a reachable path, not a strawman.",
        font=font(14),
        fill=DIM,
    )

    # per-token-size table
    ty0 = fy + 104
    draw.text((48, ty0), "all token sizes, us (rapid_llm final):", font=font(15, True), fill=FG)
    hdr = f"{'token':<8}{'bf16':>9}{'int8':>9}{'int4':>9}{'int4 / bf16':>13}"
    draw.text((48, ty0 + 28), hdr, font=font(14, True), fill=DIM)
    for i, tk in enumerate(["t1", "t8", "t64", "t512", "t4096"]):
        case = f"qwen3-30b-a3b {tk}_E128_top8_h2048_i768"
        bf = rows.get(("native/fused_moe [bf16]", case), float("nan"))
        i8 = rows.get(("native/fused_moe [int8]", case), float("nan"))
        i4 = rows.get(("native/fused_moe [int4]", case), float("nan"))
        ratio = i4 / bf
        yy = ty0 + 52 + i * 24
        line = f"{tk:<8}{bf:>9.1f}{i8:>9.1f}{i4:>9.1f}{ratio:>12.2f}x"
        draw.text((48, yy), line, font=font(14), fill=GREEN if ratio < 1.0 else FG)

    # ------------------------------------------------------------------ #
    # Panel 3: end-to-end rapid_llm vs vLLM
    # ------------------------------------------------------------------ #
    ey = H - 560
    panel(
        draw,
        (24, ey, W - 24, H - 24),
        "3  end-to-end: rapid_llm vs vLLM 0.28.0 (H100, ratio = vllm / rapid_llm)",
    )

    e2e = [
        ("Qwen2.5-0.5B  b8  g128", 3.21, 3.08, "dense, step 1.3 ms"),
        ("Qwen2.5-0.5B  b16 g256", 1.23, 1.23, "dense, step 1.5 ms"),
        ("Qwen3-4B      b8  g128", 1.35, 1.34, "dense, step 4.9 ms"),
        ("Qwen3-30B-A3B bf16 b8", 0.86, 0.87, "MoE, step 11.0 ms"),
        ("Qwen3-30B-A3B FP8  b8", 0.66, 0.67, "MoE, W8A16 vs vLLM W8A8"),
    ]
    ex, ew_max = 430, 460
    ey0 = ey + 54
    draw.text((48, ey0), "TPOT ratio (>1 = rapid_llm faster)", font=font(15, True), fill=FG)
    for i, (label, tpot, tps, note) in enumerate(e2e):
        yy = ey0 + 30 + i * 40
        draw.text((48, yy + 8), label, font=font(14), fill=FG)
        colour = GREEN if tpot > 1.0 else RED
        wpx = max(6, int(tpot / 3.4 * ew_max))
        draw.rounded_rectangle([ex, yy, ex + wpx, yy + 26], radius=4, fill=colour)
        draw.text(
            (ex + wpx + 10, yy + 6),
            f"TPOT {tpot:.2f}x   TPS {tps:.2f}x   {note}",
            font=font(13, True),
            fill=colour,
        )
    # unity marker
    ux = ex + int(1.0 / 3.4 * ew_max)
    draw.line([(ux, ey0 + 26), (ux, ey0 + 30 + len(e2e) * 40)], fill=DIM, width=1)
    draw.text((ux - 12, ey0 + 30 + len(e2e) * 40 + 4), "1.0x", font=font(12), fill=DIM)

    dy = ey0 + 30 + len(e2e) * 40 + 30
    draw.text(
        (48, dy), "two orthogonal drivers, not one 'faster engine':", font=font(15, True), fill=FG
    )
    lines = [
        ("1. host cost per decode step  (rapid_llm's edge)", BLUE),
        ("   whole decode step is one CUDA-graph replay; KV bump allocator;", DIM),
        ("   GPU-resident stop criteria polled every 8 steps -> host cost ~0.", DIM),
        ("   worth 3.2x when the step is 1.3 ms, ~0 when the step is 11 ms.", DIM),
        ("2. expert-GEMM kernel maturity  (vLLM's edge)", PURPLE),
        ("   MoE decode is expert-weight bandwidth + grouped-GEMM bound; vLLM's", DIM),
        ("   fused_moe is production-hardened (tile tables, alignment, W8A8 fp8).", DIM),
        ("   FP8 row mixes a numeric-path difference too: W8A16 vs W8A8.", DIM),
    ]
    for i, (txt, colour) in enumerate(lines):
        draw.text((48, dy + 26 + i * 22), txt, font=font(14), fill=colour)

    return img


def main() -> None:
    img = build()
    img.save(OUTPUT)
    print(f"-> {OUTPUT}  ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
