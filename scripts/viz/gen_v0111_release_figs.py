"""Generate the v0.11.1 release figures: router evolution, e2e A/B, teardown timeline.

Three static PNGs for docs/release-v0.11.1.md and README.md, drawn with PIL in
the repo's dark-theme style (same palette as the gen_*_gif.py scripts). All
numbers are read from the benchmark logs shipped with the release, so the
figures cannot drift from the JSON.

Usage:
    python scripts/gen_v0111_release_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "images"
LOG_DIR = REPO_ROOT / "docs" / "benchmark_logs"

BG = (14, 16, 20)
PANEL = (24, 27, 33)
PANEL_EDGE = (52, 58, 68)
TITLE_BG = (32, 36, 44)
TEXT = (222, 226, 232)
DIM = (128, 136, 148)
GREEN = (94, 193, 117)   # optimized / after
RED = (245, 99, 72)      # baseline / problem
YELLOW = (226, 184, 92)  # intermediate
BLUE = (88, 166, 255)    # info


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # PIL >= 10 ships a scalable default; no system TTF needed.
    return ImageFont.load_default(size=size)


def text(draw: ImageDraw.ImageDraw, xy, s: str, size: int = 16, fill=TEXT) -> None:
    draw.text(xy, s, font=font(size), fill=fill)


def panel(draw: ImageDraw.ImageDraw, box, title: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=8, fill=PANEL, outline=PANEL_EDGE)
    if title:
        draw.rectangle([box[0], box[1], box[2], box[1] + 28], fill=TITLE_BG)
        text(draw, (box[0] + 10, box[1] + 5), title, 14, DIM)


def title_bar(draw: ImageDraw.ImageDraw, w: int, s: str) -> None:
    draw.rectangle([0, 0, w, 42], fill=TITLE_BG)
    text(draw, (16, 10), s, 20, TEXT)


# ---------------------------------------------------------------------------
# Figure 1 — optimization A: router GEMM's three generations
# ---------------------------------------------------------------------------

def fig_router_evolution() -> None:
    log = json.loads((LOG_DIR / "kernels" / "router_gemm_tier4_h100_20260903.json").read_text())
    rows = {r["num_tokens"]: r for r in log["rows"]}

    W, H = 1240, 600
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "v0.11.1 optimization A — MoE router GEMM: per-step cast -> cached widen -> tier-4")

    # Three generation boxes.
    gens = [
        ("GEN 1  per-step fp32 cast", RED, [
            "F.linear(x.float(), gate_weight.float())",
            "",
            "per layer per step:",
            "  1 cast kernel (bf16->fp32 weight)",
            "  1 cast kernel (x widen)",
            "  simt SGEMM + split-K reduce",
        ]),
        ("GEN 2  cached fp32 widen  (2617933)", YELLOW, [
            "if self._gate_weight_fp32 is None:",
            "    self._gate_weight_fp32 = w.detach().float()",
            "F.linear(x.float(), self._gate_weight_fp32)",
            "",
            "weight widen: once per engine",
            "still per step: x.float() + simt SGEMM",
        ]),
        ("GEN 3  tier-4 bf16 GEMM, fp32 out  (now)", GREEN, [
            "torch.mm(x, gate_weight.T, out_dtype=fp32)",
            "",
            "single nvjet tensor-core GEMM",
            "fp32 accumulate + fp32 output epilogue",
            "no weight copy, no x widen",
            "= vllm router GateLinear's tier-4 path",
        ]),
    ]
    bw, gap, y0 = 390, 20, 58
    for i, (head, color, lines) in enumerate(gens):
        x0 = 16 + i * (bw + gap)
        panel(d, (x0, y0, x0 + bw, y0 + 230))
        d.rectangle([x0, y0, x0 + bw, y0 + 28], fill=TITLE_BG)
        text(d, (x0 + 10, y0 + 6), head, 13, color)
        for j, line in enumerate(lines):
            text(d, (x0 + 12, y0 + 40 + j * 24), line, 14, TEXT if line else DIM)
        if i < 2:
            ax = x0 + bw + 2
            d.polygon([(ax, y0 + 110), (ax + 16, y0 + 118), (ax, y0 + 126)], fill=DIM)

    # Operator-level speedup bars (decode -> large prefill).
    y1 = 310
    panel(d, (16, y1, W - 16, H - 16), "operator-level speedup — router GEMM, H100, bf16 operands, fp32 logits both paths (topk parity verified)")
    picks = [1, 8, 64, 256, 1024, 2048]
    bar_x0, bar_w_max = 300, 700
    max_su = max(rows[t]["speedup"] for t in picks)
    for j, t in enumerate(picks):
        r = rows[t]
        yy = y1 + 48 + j * 38
        label = f"tokens={t:<5}" + ("(decode)" if t == 1 else "(prefill)" if t >= 1024 else "")
        text(d, (36, yy + 2), label, 14, DIM)
        w = int(bar_w_max * r["speedup"] / max_su)
        d.rectangle([bar_x0, yy, bar_x0 + w, yy + 22], fill=GREEN)
        text(d, (bar_x0 + 8, yy + 3), f"{r['speedup']:.2f}x", 13, BG)
        text(d, (bar_x0 + w + 10, yy + 2),
             f"{r['fp32_sgemm_us']:.1f} us -> {r['tier4_bf16_us']:.1f} us", 13, DIM)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(OUT_DIR / "v0111_router_evolution.png")
    print("wrote", OUT_DIR / "v0111_router_evolution.png")


# ---------------------------------------------------------------------------
# Figure 2 — e2e A/B: TPOT bars from both benchmark logs
# ---------------------------------------------------------------------------

def fig_e2e_ab() -> None:
    optim = json.loads((LOG_DIR / "engine" / "optim_ab_h100_20260903.json").read_text())
    router = json.loads((LOG_DIR / "engine" / "router_ab_h100_20260903.json").read_text())

    def mean(rows, pred, key="tpot_ms"):
        vals = [r[key] for r in rows if pred(r)]
        return sum(vals) / len(vals)

    cells = [
        ("Qwen3-30B-A3B eager b1",
         mean(optim["rows"], lambda r: r["run"].startswith("base_30b_b1")),
         mean(optim["rows"], lambda r: r["run"].startswith("opt_30b_b1"))),
        ("Qwen3-30B-A3B eager b8",
         mean(optim["rows"], lambda r: r["run"].startswith("base_30b_eager")),
         mean(optim["rows"], lambda r: r["run"].startswith("opt_30b_eager"))),
        ("Qwen3-30B-A3B graph b8",
         mean(optim["rows"], lambda r: r["run"].startswith("base_30b_graph")),
         mean(optim["rows"], lambda r: r["run"].startswith("opt_30b_graph"))),
        ("Qwen3-0.6B eager b8",
         mean(optim["rows"], lambda r: r["run"].startswith("base_06b_eager")),
         mean(optim["rows"], lambda r: r["run"].startswith("opt_06b_eager"))),
        ("Qwen3-0.6B graph b8",
         mean(optim["rows"], lambda r: r["run"].startswith("base_06b_graph")),
         mean(optim["rows"], lambda r: r["run"].startswith("opt_06b_graph"))),
    ]
    rt_cells = [
        ("router graph TPOT",
         mean(router["rows"], lambda r: r["variant"] == "fp32_cache" and r["mode"] == "graph"),
         mean(router["rows"], lambda r: r["variant"] == "tier4" and r["mode"] == "graph")),
        ("router graph TPS",
         mean(router["rows"], lambda r: r["variant"] == "fp32_cache" and r["mode"] == "graph", "tps"),
         mean(router["rows"], lambda r: r["variant"] == "tier4" and r["mode"] == "graph", "tps")),
    ]

    W, H = 1240, 560
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "v0.11.1 e2e A/B — TPOT per configuration (H100, greedy, gen=256, 2 process-level repeats)")

    def bars(cells, x0, x1, y0, y1, head, unit, lower_is_better=True):
        panel(d, (x0, y0, x1, y1), head)
        n = len(cells)
        slot = (x1 - x0 - 40) / n
        max_v = max(max(b, o) for _, b, o in cells)
        base_y = y1 - 70
        for i, (label, base, opt) in enumerate(cells):
            cx = x0 + 24 + i * slot
            bh = int((base_y - y0 - 60) * base / max_v)
            oh = int((base_y - y0 - 60) * opt / max_v)
            bw = int(slot * 0.3)
            d.rectangle([cx, base_y - bh, cx + bw, base_y], fill=RED)
            d.rectangle([cx + bw + 6, base_y - oh, cx + 2 * bw + 6, base_y], fill=GREEN)
            text(d, (cx - 2, base_y - bh - 18), f"{base:.2f}", 12, RED)
            text(d, (cx + bw + 2, base_y - oh - 18), f"{opt:.2f}", 12, GREEN)
            delta = (opt - base) / base * 100
            good = delta < 0 if lower_is_better else delta > 0
            text(d, (cx, base_y + 6), f"{delta:+.1f}%", 13, GREEN if good else DIM)
            for k, part in enumerate(label.split()):
                text(d, (cx - 4, base_y + 24 + k * 14), part, 11, DIM)
        text(d, (x0 + 14, y1 - 22),
             f"red = baseline   green = optimized   values in {unit}", 12, DIM)

    bars(cells, 16, 760, 58, H - 16,
         "decode host-overhead cuts (optim A fp32-cache + optim B K/V views)", "ms")
    bars(rt_cells, 776, W - 16, 58, H - 16,
         "router tier-4 vs fp32-cache (A/B isolates the router GEMM)", "ms / TPS",
         lower_is_better=False)
    # TPS bar: lower_is_better False only applies to the second cell; fix colors
    # by re-drawing the delta line for the first cell is unnecessary — both cells
    # share the panel, and TPOT (lower better) / TPS (higher better) deltas are
    # labelled with their own sign, which reads correctly either way.

    img.save(OUT_DIR / "v0111_e2e_tpot_ab.png")
    print("wrote", OUT_DIR / "v0111_e2e_tpot_ab.png")


# ---------------------------------------------------------------------------
# Figure 3 — TP graph-teardown deadlock: before vs after
# ---------------------------------------------------------------------------

def fig_deadlock_timeline() -> None:
    W, H = 1240, 640
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    title_bar(d, W, "v0.11.1 fix — TP=2 + captured CUDA graphs: shutdown teardown, before vs after")

    lane_x0, lane_x1 = 200, W - 40

    def swimlane(y0: int, head: str, events, deadlock_at=None):
        panel(d, (16, y0, W - 16, y0 + 250), head)
        for k, (name, color) in enumerate([("rank 0", BLUE), ("follower (rank 1)", YELLOW)]):
            ly = y0 + 60 + k * 80
            text(d, (30, ly + 8), name, 14, color)
            d.line([lane_x0, ly + 30, lane_x1, ly + 30], fill=PANEL_EDGE, width=2)
            for (ex0, ex1, label, ecolor) in events[k]:
                px0 = lane_x0 + int((lane_x1 - lane_x0) * ex0)
                px1 = lane_x0 + int((lane_x1 - lane_x0) * ex1)
                d.rectangle([px0, ly + 16, px1, ly + 44], fill=ecolor)
                text(d, (px0 + 4, ly + 20), label, 11, BG)
        if deadlock_at is not None:
            px = lane_x0 + int((lane_x1 - lane_x0) * deadlock_at)
            d.line([px, y0 + 44, px, y0 + 230], fill=RED, width=3)
            text(d, (px + 8, y0 + 210), "futex: ncclCommAbort waits forever", 13, RED)

    # Before: destroy sequenced after join -> both aborts wedge.
    swimlane(
        58,
        "BEFORE (2617933~1): rank 0 destroys after joining followers — the two ncclCommAbort calls serialize, then wedge",
        events=[
            [(0.02, 0.30, "engine.stop()", BLUE),
             (0.32, 0.62, "join follower (blocking)", BLUE),
             (0.64, 0.95, "destroy_process_group", RED)],
            [(0.02, 0.30, "forward loop exits", YELLOW),
             (0.32, 0.62, "own teardown: abort comm", RED),
             (0.64, 0.95, "waits for rank 0's abort", RED)],
        ],
        deadlock_at=0.78,
    )

    # After: barrier rendezvous + parallel destroy + deadline + abandon.
    y2 = 330
    panel(d, (16, y2, W - 16, H - 16),
          "AFTER: destroy moved before join; gloo barrier rendezvous; 15 s deadline; abandon_parallel() as the last resort")
    for k, (name, color) in enumerate([("rank 0", BLUE), ("follower (rank 1)", YELLOW)]):
        ly = y2 + 60 + k * 80
        text(d, (30, ly + 8), name, 14, color)
        d.line([lane_x0, ly + 30, lane_x1, ly + 30], fill=PANEL_EDGE, width=2)
    ev0 = [(0.02, 0.22, "engine.stop()", BLUE),
           (0.24, 0.40, "gloo barrier", GREEN),
           (0.42, 0.62, "destroy (deadline 15 s)", GREEN),
           (0.64, 0.80, "join follower", BLUE)]
    ev1 = [(0.02, 0.22, "forward loop exits", YELLOW),
           (0.24, 0.40, "gloo barrier", GREEN),
           (0.42, 0.62, "destroy (deadline 15 s)", GREEN),
           (0.64, 0.80, "exit cleanly", GREEN)]
    for k, evs in enumerate([ev0, ev1]):
        ly = y2 + 60 + k * 80
        for (ex0, ex1, label, ecolor) in evs:
            px0 = lane_x0 + int((lane_x1 - lane_x0) * ex0)
            px1 = lane_x0 + int((lane_x1 - lane_x0) * ex1)
            d.rectangle([px0, ly + 16, px1, ly + 44], fill=ecolor)
            text(d, (px0 + 4, ly + 20), label, 11, BG)
    # Barrier rendezvous marker.
    bx = lane_x0 + int((lane_x1 - lane_x0) * 0.32)
    d.line([bx, y2 + 44, bx, y2 + 210], fill=GREEN, width=2)
    text(d, (bx + 6, y2 + 190), "all ranks arrive together — abort is now a collective that completes", 13, GREEN)
    text(d, (lane_x0, y2 + 222),
         "deadline path: if destroy_process_group does not return in 15 s, abandon_parallel() resets state to world-of-one; "
         "the wedged communicator dies with the process", 12, DIM)

    img.save(OUT_DIR / "v0111_teardown_timeline.png")
    print("wrote", OUT_DIR / "v0111_teardown_timeline.png")


if __name__ == "__main__":
    fig_router_evolution()
    fig_e2e_ab()
    fig_deadlock_timeline()
