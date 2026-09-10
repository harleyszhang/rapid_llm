"""Animate eager vs CUDA-graph decode from real torch.profiler data.

Profiles Qwen2.5-0.5B-Instruct in both modes, isolates ONE decode step per mode
holding the *same* kernels, and renders an animated timeline. Both modes run the
same forward, so the step's kernel count and GPU busy time match (327 kernels,
~1020 µs of GPU work); what differs is how that work is dispatched:

- **Eager**: 327 individual kernel launches, one per kernel. The GPU sits idle
  between them, so the step takes ~16 ms of wall time for ~1 ms of GPU work —
  occupancy ~6 %.
- **Graph**: one ``cudaGraphLaunch`` replays the captured bulk back-to-back, plus
  a few individual launches for the work that stays outside the captured region.
  The same ~1 ms of GPU work finishes in ~1.2 ms of wall time — occupancy ~84 %.

Launch counting covers both CUDA APIs: Triton kernels go through the driver API
(``cuLaunchKernelEx``, category ``cuda_driver``), so counting only
``cuda_runtime`` under-reports eager's launches by ~5x and hides the point.

Usage::

    python scripts/gen_cuda_graph_launch_gif.py
    python scripts/gen_cuda_graph_launch_gif.py --model-dir my_weight/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Rendering constants
# ---------------------------------------------------------------------------


W, H = 1200, 520
TITLE_H = 38
PAD = 18
LINE_H = 22
LANE_H = 88
LANE_GAP = 30
LABEL_W = 130

BG = (14, 16, 20)
TITLE_BG = (32, 36, 44)
TITLE_FG = (222, 226, 232)
TEXT_FG = (222, 226, 232)
DIM = (128, 136, 148)
AXIS_FG = (52, 58, 68)

LAUNCH_FG = (226, 184, 92)  # amber – CPU cudaLaunchKernel
GRAPH_LAUNCH_FG = (94, 193, 117)  # green – cudaGraphLaunch
KERNEL_FG = (70, 150, 220)  # blue – GPU kernel
IDLE_FG = (60, 40, 40)  # dark red tint for idle bands
EAGER_ACCENT = (245, 99, 72)  # red-orange for eager annotations
GRAPH_ACCENT = (94, 193, 117)  # green for graph annotations


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------
#: Eager has no per-step marker, so a step is cut out at the run's largest GPU gaps.
#: The workload below generates 8 decode tokens, hence 8 cuts; the modal-segment
#: filter in :func:`profile_mode` makes the result insensitive to this number.
_N_STEP_CUTS = 8

#: Smallest segment counted as a decode step (this checkpoint runs ~327 kernels per
#: step; prefill and tail fragments are far smaller or far larger).
_MIN_STEP_KERNELS = 100


def _raw_profile(use_cuda_graph: bool, ckpt: str) -> dict:
    """Run profiler, return raw totals (no step isolation)."""
    import torch

    from rapid_llm import SamplingParams, TextGenerator

    gen = TextGenerator(
        checkpoints_dir=ckpt,
        use_cuda_graph=use_cuda_graph,
        max_seq_len=1024,
        max_gpu_num_blocks=8192,
    )
    prompts = ["The capital of France is"] * 4
    list(gen.stream(prompts, SamplingParams(max_gen_len=4, temperature=0.0)))
    torch.cuda.synchronize()

    trace_dir = Path(tempfile.mkdtemp())
    tag = "graph" if use_cuda_graph else "eager"
    trace_path = trace_dir / f"trace_{tag}.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        gen.generate(prompts, SamplingParams(max_gen_len=8, temperature=0.0))
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))

    events = json.loads(trace_path.read_text())["traceEvents"]
    # Both CUDA APIs: torch ops launch through the runtime API (cudaLaunchKernel /
    # cudaLaunchKernelExC) while Triton kernels go through the driver API
    # (cuLaunchKernelEx, category cuda_driver). Counting only cuda_runtime misses
    # the Triton majority — 535 of 2647 launches — and would understate eager's
    # per-step dispatch count by ~5x.
    launches = [
        e
        for e in events
        if e.get("cat") in ("cuda_runtime", "cuda_driver") and "LaunchKernel" in e.get("name", "")
    ]
    graph_launches = [
        e for e in events if e.get("cat") == "cuda_runtime" and "GraphLaunch" in e.get("name", "")
    ]
    kernels = sorted(
        [e for e in events if e.get("cat") == "kernel"],
        key=lambda e: e["ts"],
    )
    launches.sort(key=lambda e: e["ts"])
    graph_launches.sort(key=lambda e: e["ts"])
    return {
        "kernels": kernels,
        "launches": launches,
        "graph_launches": graph_launches,
        "total_gpu_us": sum(k["dur"] for k in kernels),
    }


def profile_mode(use_cuda_graph: bool, ckpt: str, step_kernels_ref: int = 0) -> dict:
    """Profile one mode and isolate ONE decode step for the timeline.

    *step_kernels_ref* is the eager step's kernel count; graph mode takes the same
    number of kernels so both tracks hold one identical step's worth of work.
    """
    raw = _raw_profile(use_cuda_graph, ckpt)
    kernels = raw["kernels"]
    launches = raw["launches"]
    graph_launches = raw["graph_launches"]

    if graph_launches:
        # Graph: a step starts at a cudaGraphLaunch. A launch-to-launch window is not
        # the step, though — part of each step (KV index update, sampling) runs outside
        # the captured region, so that window cuts mid-step (21 of this checkpoint's 24
        # layers). Take the eager step's kernel count from the launch instead: same
        # forward, same kernels, and the whole-run totals agree to 0.3 %.
        n = step_kernels_ref or len(kernels) // max(len(graph_launches), 1)
        i = min(2, len(graph_launches) - 1)
        first = next((j for j, k in enumerate(kernels) if k["ts"] >= graph_launches[i]["ts"]), 0)
        # Slide the start forward until the window holds exactly one cudaGraphLaunch.
        # A step's out-of-capture work runs adjacent to its replay, so a window anchored
        # exactly on the launch reaches into the next one and would show two markers.
        start = first
        for j in range(first, max(first, len(kernels) - n)):
            window = kernels[j : j + n]
            if not window:
                break
            w_lo, w_hi = window[0]["ts"], window[-1]["ts"] + window[-1]["dur"]
            if sum(1 for e in graph_launches if w_lo - 5 <= e["ts"] < w_hi) == 1:
                start = j
                break
        step_kernels = kernels[start : start + n]
        n_steps = len(graph_launches)
    else:
        # Eager: no launch marker, so a step is delimited by the run's largest GPU gaps
        # (the inter-step CPU work: sampling, scheduler, readback). A fixed gap
        # threshold cannot separate them — intra-step gaps reach 100 µs too — so cut at
        # the N largest gaps and keep the modal segment size, which is one full decode
        # step: every per-layer kernel appears once per layer in it.
        gaps = sorted(
            (kernels[i + 1]["ts"] - (kernels[i]["ts"] + kernels[i]["dur"]), i)
            for i in range(len(kernels) - 1)
        )
        cuts = sorted(i for _, i in gaps[-_N_STEP_CUTS:])
        bounds = [0, *(i + 1 for i in cuts), len(kernels)]
        segs = [kernels[a:b] for a, b in itertools.pairwise(bounds) if b - a >= _MIN_STEP_KERNELS]
        sizes = Counter(len(s) for s in segs)
        step_kernels = next(s for s in segs if len(s) == sizes.most_common(1)[0][0])
        n_steps = len(segs)

    lo = step_kernels[0]["ts"]
    hi = step_kernels[-1]["ts"] + step_kernels[-1]["dur"]
    busy = sum(k["dur"] for k in step_kernels)

    return {
        "mode": "graph" if use_cuda_graph else "eager",
        "lo_us": lo,
        "hi_us": hi,
        "step_kernels": step_kernels,
        "step_launches": [e for e in launches if lo - 5 <= e["ts"] < hi],
        "n_kernels": len(step_kernels),
        "n_launches": len([e for e in launches if lo - 5 <= e["ts"] < hi]),
        "n_graph_launches": len([e for e in graph_launches if lo - 5 <= e["ts"] < hi]),
        "span_us": hi - lo,
        "gpu_busy_us": busy,
        "occupancy_pct": busy / (hi - lo) * 100 if hi > lo else 0.0,
        "total_kernels": len(kernels),
        "n_steps": n_steps,
    }


def collect_data(ckpt: str) -> tuple[dict, dict]:
    """Profile both modes and return (eager, graph) dicts.

    Eager goes first: its step kernel count is what the graph window is sized to,
    so the two tracks are guaranteed to hold the same work.
    """
    print("profiling eager mode ...")
    eager = profile_mode(False, ckpt)
    print(
        f"  {eager['total_kernels']} kernels total, {eager['n_steps']} steps; one step = "
        f"{eager['n_kernels']} kernels, {eager['n_launches']} launches, "
        f"{eager['span_us']:.0f} µs wall, {eager['gpu_busy_us']:.0f} µs GPU busy "
        f"({eager['occupancy_pct']:.0f} %)"
    )

    print("profiling graph mode ...")
    graph = profile_mode(True, ckpt, step_kernels_ref=eager["n_kernels"])
    print(
        f"  {graph['total_kernels']} kernels total, {graph['n_steps']} steps; one step = "
        f"{graph['n_kernels']} kernels, {graph['n_graph_launches']} cudaGraphLaunch + "
        f"{graph['n_launches'] - graph['n_graph_launches']} launches outside the capture, "
        f"{graph['span_us']:.0f} µs wall, {graph['gpu_busy_us']:.0f} µs GPU busy "
        f"({graph['occupancy_pct']:.0f} %)"
    )

    if eager["n_kernels"] != graph["n_kernels"]:
        raise AssertionError(
            f"step kernel counts diverged: eager {eager['n_kernels']} vs graph "
            f"{graph['n_kernels']} — the two tracks would not be comparable"
        )
    print(
        f"  same {eager['n_kernels']} kernels both sides; wall time "
        f"{eager['span_us'] / max(graph['span_us'], 1):.1f}x, occupancy "
        f"{eager['occupancy_pct']:.0f} % vs {graph['occupancy_pct']:.0f} %"
    )
    return eager, graph


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _normalise(eager: dict, graph: dict) -> tuple[list, list, float]:
    """Shift both modes so their step starts at t=0, return common scale."""
    e_lo = eager["lo_us"]
    g_lo = graph["lo_us"]
    e_kernels = [
        {"ts": k["ts"] - e_lo, "dur": k["dur"], "name": k.get("name", "")}
        for k in eager["step_kernels"]
    ]
    e_launches = [
        {"ts": l["ts"] - e_lo, "dur": l.get("dur", 2.0), "name": l.get("name", "")}
        for l in eager["step_launches"]
    ]
    g_kernels = [
        {"ts": k["ts"] - g_lo, "dur": k["dur"], "name": k.get("name", "")}
        for k in graph["step_kernels"]
    ]
    g_launches = [
        {"ts": l["ts"] - g_lo, "dur": l.get("dur", 2.0), "name": l.get("name", "")}
        for l in graph["step_launches"]
    ]
    # Use the eager span as the common scale so the graph step visibly
    # finishes earlier within the same window.
    span = max(eager["span_us"], graph["span_us"])

    def _lane(d: dict, kernels: list, launches: list) -> dict:
        return {
            "kernels": kernels,
            "launches": launches,
            "span": d["span_us"],
            "gpu_busy": d["gpu_busy_us"],
            "occupancy": d["occupancy_pct"],
            "n_launches": d["n_launches"],
            "n_graph_launches": d["n_graph_launches"],
            "label": (
                f"eager ({d['n_launches']} launches)"
                if d["mode"] == "eager"
                else (
                    f"graph ({d['n_graph_launches']} graph launch + "
                    f"{d['n_launches'] - d['n_graph_launches']} outside)"
                )
            ),
        }

    return (
        _lane(eager, e_kernels, e_launches),
        _lane(graph, g_kernels, g_launches),
        span,
    )


def _render_frame(
    eager_norm: dict,
    graph_norm: dict,
    common_span: float,
    fonts: tuple,
    n_eager_shown: int,
    n_graph_kernels_shown: int,
    graph_launched: bool,
) -> Image.Image:
    """Render one frame of the animation."""
    _body, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    # Title bar
    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 10),
        "rapid-llm  —  CUDA graph: kernel launch overhead (torch.profiler, Qwen2.5-0.5B, H100)",
        fill=TITLE_FG,
        font=small,
    )

    # 2-lane layout: top = eager, bottom = graph
    lane_y = {
        "eager": TITLE_H + PAD + LINE_H,
        "graph": TITLE_H + PAD + LINE_H + LANE_H + LANE_GAP,
    }
    timeline_w = W - PAD - LABEL_W
    scale = timeline_w / max(common_span, 1.0)

    # Draw lane labels and separators
    for name, y, accent in [
        (eager_norm["label"], lane_y["eager"], EAGER_ACCENT),
        (graph_norm["label"], lane_y["graph"], GRAPH_ACCENT),
    ]:
        draw.text((PAD, y + LANE_H // 2 - 9), name, fill=accent, font=bold)
        draw.line([LABEL_W, y + LANE_H, W - PAD, y + LANE_H], fill=AXIS_FG)

    # --- Eager lane ---
    ey = lane_y["eager"] + 14
    e = eager_norm
    # Draw CPU launches as thin amber bars on the CPU sub-lane
    cpu_h = 18
    for _i, l in enumerate(e["launches"][:n_eager_shown]):
        x0 = LABEL_W + l["ts"] * scale
        x1 = x0 + max(l["dur"] * scale, 3.0)
        draw.rectangle([x0, ey, x1, ey + cpu_h], fill=LAUNCH_FG)

    # Draw GPU kernels as blue bars, staggered after each launch
    for _i, k in enumerate(e["kernels"][:n_eager_shown]):
        x0 = LABEL_W + k["ts"] * scale
        x1 = x0 + max(k["dur"] * scale, 2.0)
        draw.rectangle([x0, ey + cpu_h + 4, x1, ey + cpu_h + 4 + cpu_h], fill=KERNEL_FG)

    # Idle bands between kernels (the gap is the story)
    e_kernels = e["kernels"][:n_eager_shown]
    for i in range(1, len(e_kernels)):
        gap_start = e_kernels[i - 1]["ts"] + e_kernels[i - 1]["dur"]
        gap_end = e_kernels[i]["ts"]
        if gap_end - gap_start > 3.0:
            x0 = LABEL_W + gap_start * scale
            x1 = LABEL_W + gap_end * scale
            draw.rectangle(
                [x0, ey + cpu_h + 4, x1, ey + cpu_h + 4 + cpu_h],
                fill=IDLE_FG,
            )

    # --- Graph lane ---
    gy = lane_y["graph"] + 14
    gh = LANE_H - 28
    g = graph_norm
    if graph_launched:
        # The cudaGraphLaunch in green; the launches for the work that stays outside
        # the captured region in amber — the same colour eager uses, because they are
        # individual dispatches too.
        for l in g["launches"]:
            x0 = LABEL_W + l["ts"] * scale
            x1 = x0 + max(l["dur"] * scale, 3.0)
            is_graph_launch = "GraphLaunch" in l.get("name", "")
            draw.rectangle(
                [x0, gy, x1, gy + gh], fill=GRAPH_LAUNCH_FG if is_graph_launch else LAUNCH_FG
            )
            if is_graph_launch and x1 - x0 > 60:
                draw.text((x0 + 4, gy + 4), "cudaGraphLaunch", fill=BG, font=small)

        # Packed kernels
        for _i, k in enumerate(g["kernels"][:n_graph_kernels_shown]):
            x0 = LABEL_W + k["ts"] * scale
            x1 = x0 + max(k["dur"] * scale, 1.5)
            draw.rectangle([x0, gy + 4, x1, gy + gh - 4], fill=KERNEL_FG)

    # Time axis ticks — pick a step that keeps the axis to ~8-12 ticks whatever the
    # span (an eager step is ~16 ms, a graph step ~1.2 ms).
    tick_step = next(
        (
            s
            for s in (20.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0)
            if common_span / s <= 12
        ),
        5000.0,
    )
    tick = 0.0
    axis_y = lane_y["graph"] + LANE_H
    while tick <= common_span:
        x = LABEL_W + tick * scale
        draw.line([x, axis_y - 4, x, axis_y + 4], fill=DIM)
        draw.text((x - 16, axis_y + 6), f"{tick:.0f}µs", fill=DIM, font=small)
        tick += tick_step

    # Stats footer: same kernels both sides, so the difference is span and occupancy.
    footer_y = H - PAD - LINE_H
    draw.text(
        (PAD, footer_y),
        f"eager: {len(e['kernels'])} kernels, {e['n_launches']} launches, "
        f"{e['span']:.0f}µs wall, GPU busy {e['gpu_busy']:.0f}µs ({e['occupancy']:.0f}%)"
        f"    |    "
        f"graph: {len(g['kernels'])} kernels, {g['n_graph_launches']} cudaGraphLaunch, "
        f"{g['span']:.0f}µs wall, GPU busy {g['gpu_busy']:.0f}µs ({g['occupancy']:.0f}%)",
        fill=DIM,
        font=small,
    )
    return canvas


def build_gif(
    eager: dict,
    graph: dict,
    out_path: str,
    duration_ms: int = 600,
) -> None:
    """Build the animated GIF."""
    e_norm, g_norm, common_span = _normalise(eager, graph)
    fonts = (
        ImageFont.load_default(size=17),
        ImageFont.load_default(size=16),
        ImageFont.load_default(size=14),
    )

    n_eager_total = len(e_norm["kernels"])
    n_graph_total = len(g_norm["kernels"])
    # A step holds ~327 kernels, so reveal in chunks: one kernel per frame would be
    # a 327-frame GIF.
    eager_step = max(1, n_eager_total // 12)
    graph_step = max(1, n_graph_total // 12)

    frames: list[Image.Image] = []

    # Phase 1: eager launches and kernels accumulate, showing the CPU dispatch cost
    for n in range(eager_step, n_eager_total + eager_step, eager_step):
        frames.append(
            _render_frame(
                e_norm,
                g_norm,
                common_span,
                fonts,
                n_eager_shown=min(n, n_eager_total),
                n_graph_kernels_shown=0,
                graph_launched=False,
            )
        )
    # Hold eager complete for 2 frames
    frames += [frames[-1]] * 2

    # Phase 2: graph launches once, then kernels fill in rapidly
    frames.append(
        _render_frame(
            e_norm,
            g_norm,
            common_span,
            fonts,
            n_eager_shown=n_eager_total,
            n_graph_kernels_shown=0,
            graph_launched=True,
        )
    )
    for n in range(graph_step, n_graph_total + graph_step, graph_step):
        frames.append(
            _render_frame(
                e_norm,
                g_norm,
                common_span,
                fonts,
                n_eager_shown=n_eager_total,
                n_graph_kernels_shown=min(n, n_graph_total),
                graph_launched=True,
            )
        )
    # Hold final frame
    frames += [frames[-1]] * 4

    palette = [im.convert("P", palette=Image.ADAPTIVE, colors=64) for im in frames]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette[0].save(
        out,
        save_all=True,
        append_images=palette[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(palette)} frames)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="my_weight/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="docs/images/cuda_graph_launch.gif")
    ap.add_argument("--duration", type=int, default=600, help="ms per frame")
    args = ap.parse_args()

    eager, graph = collect_data(args.model_dir)
    build_gif(eager, graph, args.out, args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())
