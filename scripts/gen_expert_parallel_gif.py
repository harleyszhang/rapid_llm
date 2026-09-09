"""Record the expert-parallelism GIF: where a MoE layer's tokens actually go.

Expert parallelism is usually drawn as a grid of experts with arrows, which
says nothing about what the wires carry or which experts the tokens pick. This
renders both, from a real ``--enable-expert-parallel`` run of a real MoE
checkpoint:

* left panel — the batch being generated (the traffic has a subject);
* right, top — the routing heatmap: all 128 experts as cells, rank 0's block
  and rank 1's block side by side, brightness = how often this step's tokens
  routed there (recorded by wrapping every ``SparseMoeBlock._route``, no
  staging);
* right, bottom — the collective ledger per step: EP's two all-to-alls per
  layer appear, the MoE all-reduce disappears (only attention's remains), and
  a measured TP2 line shows what the same step would have all-reduced.

The two opening frames state the mechanism in terminal text — TP slices every
expert and all-reduces partial sums; EP deals whole experts to ranks and
exchanges only the routed rows.

Usage:
    python scripts/gen_expert_parallel_gif.py --model-dir <moe-ckpt>
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
from rapid_llm.engine.sampler import SamplingParams
from rapid_llm.modules.moe import SparseMoeBlock
from rapid_llm.tools.observability import (
    Collective,
    CollectiveStats,
    Plane,
    Tally,
    human_bytes,
)
from rapid_llm.utils.prompt_templates import get_prompter

try:  # DejaVu ships with matplotlib's wheel; headless hosts often have none
    from matplotlib import get_data_path

    _FONT_DIR = Path(get_data_path()) / "fonts" / "ttf"
except ImportError:  # pragma: no cover - hosts with system fonts installed
    _FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_PATH = str(_FONT_DIR / "DejaVuSansMono.ttf")
FONT_BOLD = str(_FONT_DIR / "DejaVuSansMono-Bold.ttf")

TP_SIZE = 2

# Four short prompts: enough that the batch dimension shows up in the byte counts,
# few enough that every request has its own row and the GIF stays a few seconds.
PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The opposite of hot is",
    "The sun rises in the",
]

# Terminal palette, shared with the other README GIFs.
W, H = 1180, 620
TITLE_H, PAD, LINE_H = 36, 18, 22
BG, TITLE_BG, TITLE_FG = (14, 16, 20), (32, 36, 44), (222, 226, 232)
PROMPT_FG, DIM, TEXT_FG = (118, 214, 118), (128, 136, 148), (222, 226, 232)
RUNNING, QUEUED, DONE = (118, 214, 118), (226, 184, 92), (110, 160, 226)
PANEL_BG, BAR_BG = (22, 25, 31), (34, 38, 46)
DATA_FG, CONTROL_FG = (120, 190, 240), (200, 150, 240)
ALERT = (240, 140, 110)
RANK0_FG, RANK1_FG = (118, 214, 118), (120, 190, 240)


@dataclass
class Frame:
    """One engine step from both sides: the text it produced, the bytes it cost,
    and (EP only) the experts this step's tokens routed to."""

    step: int
    phase: str
    rows: list[tuple[str, str, int, str]] = field(default_factory=list)
    expert_hist: list[int] = field(default_factory=list)
    num_experts: int = 0
    step_tallies: dict[Collective, Tally] = field(default_factory=dict)
    run_tallies: dict[Collective, Tally] = field(default_factory=dict)
    step_bytes: int = 0
    run_bytes: int = 0


class _RoutingRecorder:
    """Tally real top-k expert ids per step by wrapping each block's ``_route``.

    The wrap is per-instance and lives only for this recording: the model's
    own forward keeps calling ``self._route``, which now also hands the ids to
    us. Tensors are collected on the device and folded to a histogram at the
    step boundary — one sync per step, nothing per layer.
    """

    def __init__(self) -> None:
        self._pending: list[torch.Tensor] = []
        self.num_experts = 0

    def install(self, model: torch.nn.Module) -> None:
        """Wrap every :class:`SparseMoeBlock` under ``model``."""
        for module in model.modules():
            if isinstance(module, SparseMoeBlock):
                self.num_experts = module.num_experts
                original = module._route
                module._route = self._wrap(original)

    def _wrap(self, original):
        def wrapped(x: torch.Tensor):
            weights, ids = original(x)
            self._pending.append(ids.detach())
            return weights, ids

        return wrapped

    def begin_step(self) -> None:
        self._pending = []

    def end_step(self) -> list[int]:
        """Fold this step's ids into a per-expert histogram (host side)."""
        hist = [0] * self.num_experts
        if self._pending:
            flat = torch.cat([ids.reshape(-1) for ids in self._pending]).cpu().tolist()
            for expert in flat:
                hist[expert] += 1
        return hist


def _run_engine(model_dir: str, max_gen_len: int, *, ep: bool) -> list[Frame]:
    """Drive one real two-rank engine, snapshotting ledger + routing per step."""
    engine = ContinuousBatchingEngine.from_pretrained(
        model_dir,
        max_seq_len=512,
        max_num_seqs=len(PROMPTS),
        max_gpu_num_blocks=4096,
        tensor_parallel_size=TP_SIZE,
        enable_expert_parallel=ep,
        # The collectives this GIF is about live inside the decode CUDA graph;
        # a replay bypasses both the routing hook and the Python accounting, so
        # every decode frame would show an empty heatmap and 0 B. Eager decode
        # is the honest instrumentation mode; that the a2a also captures and
        # replays inside graphs is bench_expert_parallel.py's ep2_graph arm.
        use_cuda_graph=False,
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
    routing = _RoutingRecorder()
    routing.install(engine.engine.model_runner.model)

    frames: list[Frame] = []
    try:
        with CollectiveStats.collect() as run:
            while engine.has_unfinished_requests():
                # Exact, not a guess: a request with no output tokens can only be
                # advanced by a prefill pass.
                phase = "prefill" if any(not r.output_token_ids for r in requests) else "decode"
                routing.begin_step()
                with CollectiveStats.collect() as step:
                    engine.step()
                frames.append(
                    Frame(
                        step=len(frames) + 1,
                        phase=phase,
                        rows=[_row(request) for request in requests],
                        expert_hist=routing.end_step(),
                        num_experts=routing.num_experts,
                        step_tallies=step.tallies(),
                        run_tallies=run.tallies(),
                        step_bytes=step.nbytes,
                        run_bytes=run.nbytes,
                    )
                )
    finally:
        engine.shutdown()
    return frames


def record(model_dir: str, max_gen_len: int) -> tuple[list[Frame], int]:
    """Record both engines on the same workload: TP2 for the baseline wire cost,
    EP2 for the run the GIF shows. Both run eager — a graph replay would hide
    the very collectives the frames are meant to count (see ``_run_engine``).

    Returns:
        The EP frames, and TP2's mean all-reduce bytes per *decode* step — the
        number the ledger's comparison line shows. Prefill steps are excluded
        from the mean: their all-reduce is prefill-shaped (one wide pass), and
        the claim being shown is about the per-step decode traffic.
    """
    print(f"recording the TP{TP_SIZE} baseline run (same workload) ...")
    tp_frames = _run_engine(model_dir, max_gen_len, ep=False)
    decode_ar = [
        frame.step_tallies.get(Collective.ALL_REDUCE, Tally()).nbytes
        for frame in tp_frames
        if frame.phase == "decode"
    ]
    tp_ar_per_decode = sum(decode_ar) // len(decode_ar) if decode_ar else 0
    print(f"TP2: {len(tp_frames)} steps, all-reduce {human_bytes(tp_ar_per_decode)}/decode step")

    print(f"recording the EP{TP_SIZE} run ...")
    ep_frames = _run_engine(model_dir, max_gen_len, ep=True)
    a2a = sum(
        frame.step_tallies.get(Collective.ALL_TO_ALL, Tally()).nbytes for frame in ep_frames
    )
    print(f"EP2: {len(ep_frames)} steps, all-to-all {human_bytes(a2a)} over the run")
    return ep_frames, tp_ar_per_decode


def _row(request) -> tuple[str, str, int, str]:
    if request.is_finished:
        status = f"done {request.finish_reason}"
    elif request.slot is not None:
        status = "decoding"
    else:
        status = "queued"
    return request.request_id, status, len(request.output_token_ids), " ".join(request.text.split())


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _draw_requests(draw, x0: int, y0: int, width: int, frame: Frame, fonts) -> None:
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


def _draw_heatmap(draw, x0: int, y0: int, width: int, frame: Frame, fonts) -> None:
    """Right panel, top: every expert as a cell, brightness = hits this step.

    The two blocks *are* the two EP ranks — rank 0 owns the left block's
    experts whole, rank 1 the right block's — so a cell lighting up in the
    right block means those rows physically crossed the wire to rank 1 and
    back. Brightness is square-root scaled against this step's peak, so a
    single hit is still visible without letting one hot expert wash out the
    rest.
    """
    _, _, small = fonts
    hist = frame.expert_hist
    n = frame.num_experts or 128
    per_rank = n // TP_SIZE
    cols = 8
    grid_rows = per_rank // cols
    cell, gap = 20, 2
    block_w = cols * (cell + gap) - gap
    mid_gap = 44  # wide enough for the a2a annotation between the blocks
    bx0 = x0 + (width - block_w * 2 - mid_gap) // 2
    peak = max(hist) if hist else 1
    for rank in range(TP_SIZE):
        bx = bx0 + rank * (block_w + mid_gap)
        base = RANK0_FG if rank == 0 else RANK1_FG
        draw.text(
            (bx, y0),
            f"rank {rank} · experts {rank * per_rank}-{(rank + 1) * per_rank - 1}",
            fill=base,
            font=small,
        )
        for local in range(per_rank):
            row, col = divmod(local, cols)
            hits = hist[rank * per_rank + local]
            if hits:
                level = 0.35 + 0.65 * (hits / peak) ** 0.5
                colour = tuple(int(channel * level) for channel in base)
            else:
                colour = BAR_BG
            cx = bx + col * (cell + gap)
            cy = y0 + 18 + row * (cell + gap)
            draw.rectangle([cx, cy, cx + cell - 1, cy + cell - 1], fill=colour)
    # The annotation between the blocks: what crossing blocks means.
    mid_x = bx0 + block_w + mid_gap // 2
    mid_y = y0 + 18 + grid_rows * (cell + gap) // 2
    draw.text((mid_x - 14, mid_y - 22), "a2a", fill=ALERT, font=small)
    draw.text((mid_x - 14, mid_y - 6), "↔", fill=ALERT, font=small)
    note_y = y0 + 18 + grid_rows * (cell + gap) + 4
    draw.text(
        (x0, note_y),
        f"routing this step: {sum(hist)} expert picks, brightness = hits",
        fill=DIM,
        font=small,
    )
    return note_y + LINE_H  # bottom edge, for the ledger to start under


def _colour(op: Collective) -> tuple[int, int, int]:
    """One colour per plane, so the two budgets read apart at a glance."""
    return CONTROL_FG if op.plane is Plane.CONTROL else DATA_FG


def _draw_ledger(
    draw,
    x0: int,
    y0: int,
    width: int,
    frame: Frame,
    fonts,
    tp_ar_per_decode: int,
) -> None:
    """Right panel, bottom: per-op step bytes, plus the measured TP2 comparison.

    Bars are scaled within the step against the same scale the TP2 comparison
    line uses, so the all-to-all bar and the all-reduce-that-would-have-been
    bar are directly comparable lengths.
    """
    body, bold, small = fonts
    draw.rectangle([x0, y0, x0 + width, H - PAD - 22], fill=PANEL_BG)
    draw.text((x0 + 12, y0 + 8), "collective ledger  —  rank 0", fill=TITLE_FG, font=bold)
    draw.text((x0 + width - 190, y0 + 8), "this step  /  run", fill=DIM, font=small)

    y = y0 + 8 + LINE_H + 4
    bar_x, bar_w = x0 + 12, width - 24
    scale = max(
        max((tally.nbytes for tally in frame.step_tallies.values()), default=0),
        tp_ar_per_decode,
        1,
    )
    for op in list(frame.run_tallies)[:5]:
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
        filled = bar_w * step_bytes / scale
        if filled >= 1:
            draw.rectangle([bar_x, y + LINE_H - 4, bar_x + filled, y + LINE_H + 2], fill=colour)
        y += LINE_H + 14

    # The comparison the panel exists for: what TP2 moved on the same workload.
    draw.text((bar_x, y), f"{'tp2 all-reduce':<17s}{'per step':<8s}", fill=ALERT, font=body)
    draw.text(
        (bar_x + 300, y),
        f"{human_bytes(tp_ar_per_decode):>9s} /{'measured':>9s}",
        fill=ALERT,
        font=body,
    )
    draw.rectangle([bar_x, y + LINE_H - 4, bar_x + bar_w, y + LINE_H + 2], fill=BAR_BG)
    filled = bar_w * tp_ar_per_decode / scale
    if filled >= 1:
        draw.rectangle(
            [bar_x, y + LINE_H - 4, bar_x + filled, y + LINE_H + 2], fill=ALERT
        )


def _frame_chrome(draw, fonts, model_name: str, command: str) -> int:
    """Title bar, command line, traffic lights; returns the content top edge."""
    _, _, small = fonts
    draw.rectangle([0, 0, W, TITLE_H], fill=TITLE_BG)
    draw.text(
        (12, 9),
        f"rapid-llm  —  expert parallelism: whole experts, routed rows  ({model_name})",
        fill=TITLE_FG,
        font=small,
    )
    for index, colour in enumerate([(245, 99, 72), (253, 188, 64), (94, 193, 117)]):
        draw.ellipse([W - 78 + index * 18, 11, W - 68 + index * 18, 21], fill=colour)
    y = TITLE_H + PAD
    draw.text((PAD, y), command, fill=PROMPT_FG, font=fonts[0])
    return y + LINE_H + 4


def render(frame: Frame, fonts, model_name: str, tp_ar_per_decode: int) -> Image.Image:
    """Draw one recorded step: batch left, routing + ledger right."""
    _, bold, small = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    y = _frame_chrome(
        draw,
        fonts,
        model_name,
        f"$ rapid-llm batch --tensor-parallel-size {TP_SIZE} --enable-expert-parallel"
        f"   # rank-0 bytes, eager (graphs would hide them)",
    )
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
    right_x, right_w = PAD + left_w + 16, W - 2 * PAD - left_w - 16
    ledger_y = _draw_heatmap(draw, right_x, y, right_w, frame, fonts) + 8
    _draw_ledger(draw, right_x, ledger_y, right_w, frame, fonts, tp_ar_per_decode)

    draw.text(
        (PAD, H - PAD - 16),
        "each token picks 8 of 128 experts; its rows travel to the owning rank and back — "
        "two all-to-alls per layer, and the MoE all-reduce is gone",
        fill=DIM,
        font=small,
    )
    return canvas


def render_concept(kind: str, fonts, model_name: str) -> Image.Image:
    """One of the two opening frames: the mechanism, in terminal text."""
    body, bold, _ = fonts
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    y = _frame_chrome(
        draw, fonts, model_name, "$ rapid-llm batch --enable-expert-parallel   # how it shards"
    )
    draw.text((PAD, y), "how a MoE layer shards across 2 GPUs", fill=TITLE_FG, font=bold)
    y += LINE_H + 10

    columns = {
        "shard": [
            (
                "tensor parallel (baseline)",
                RANK1_FG,
                [
                    "every expert sliced 1/2 + 1/2",
                    "each rank holds a partial of",
                    "every expert's GEMM",
                    "",
                    "   rank 0 partial ──┐",
                    "                    ├── all_reduce",
                    "   rank 1 partial ──┘",
                    "",
                    "one all-reduce per MoE layer:",
                    "the full hidden vector of every",
                    "token crosses, in both halves",
                ],
            ),
            (
                "expert parallel (this run)",
                RANK0_FG,
                [
                    "rank 0 owns experts    0-63",
                    "rank 1 owns experts  64-127",
                    "each expert whole, unsharded",
                    "",
                    "   token ──top-8──▶ experts",
                    "   rows ──dispatch a2a──▶ owner",
                    "   owner ──GEMM──▶ combine a2a",
                    "",
                    "two all-to-alls per MoE layer;",
                    "no MoE all-reduce — the combine",
                    "lands every token's full sum",
                ],
            ),
        ],
        "route": [
            (
                "the exchange, one MoE layer", DATA_FG, [
                    "router: softmax over 128 logits,",
                    "top-8 experts per token (fp32 — a",
                    "wrong pick costs more than a",
                    "rounded weight)",
                    "",
                    "dispatch a2a: each rank receives",
                    "the rows aimed at its experts",
                    "",
                    "combine a2a: weighted sums return",
                    "to the sending rank, which applies",
                    "its own routing weights",
                ]),
            (
                "why it is graph-safe", ALERT, [
                    "capacity-based exchange: buffers",
                    "sized rows x top-k per destination,",
                    "the worst case one rank could own",
                    "",
                    "split sizes are static on the host:",
                    "no GPU->CPU sync per layer, and",
                    "shapes are fixed per (rows, top-k)",
                    "",
                    "so the dispatch/combine all-to-alls",
                    "capture into decode CUDA graphs and",
                    "replay in lockstep across ranks",
                ]),
        ],
    }
    for index, (title, colour, lines) in enumerate(columns[kind]):
        x = PAD + index * (W // 2 - PAD // 2)
        draw.text((x, y), title, fill=colour, font=bold)
        line_y = y + LINE_H + 6
        for line in lines:
            draw.text((x, line_y), line, fill=TEXT_FG, font=body)
            line_y += LINE_H
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/otto-temp/modelzoo_with_full_weights/Qwen3-30B-A3B-Instruct-2507-FP8")
    ap.add_argument("--out", default="docs/images/expert_parallel.gif")
    ap.add_argument("--max-gen-len", type=int, default=24)
    ap.add_argument("--every", type=int, default=1, help="keep one frame per N steps")
    ap.add_argument("--duration", type=int, default=130, help="ms per recorded frame")
    args = ap.parse_args()

    frames, tp_ar_per_decode = record(args.model_dir, args.max_gen_len)
    fonts = (
        ImageFont.truetype(FONT_PATH, 16),
        ImageFont.truetype(FONT_BOLD, 16),
        ImageFont.truetype(FONT_PATH, 14),
    )
    model_name = Path(args.model_dir).name

    images = [
        render_concept("shard", fonts, model_name).convert(
            "P", palette=Image.ADAPTIVE, colors=64
        ),
        render_concept("route", fonts, model_name).convert(
            "P", palette=Image.ADAPTIVE, colors=64
        ),
    ]
    durations = [2800, 2800]
    images += [
        render(frame, fonts, model_name, tp_ar_per_decode).convert(
            "P", palette=Image.ADAPTIVE, colors=64
        )
        for frame in frames[:: args.every]
    ]
    durations += [args.duration] * len(frames[:: args.every])
    images += [images[-1]] * 8  # hold the finished state so the loop is readable
    durations += [args.duration] * 8

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.0f} KB, {len(images)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
