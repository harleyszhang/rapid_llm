"""What the EP exchange's padding costs the expert GEMM.

The EP dispatcher hands ``fused_moe`` a buffer of ``ep_size * rows * top_k``
rows — the worst-case capacity of an equal-split all-to-all, where one rank
could own every routing slot. Only about ``rows * top_k / ep_size`` of those
rows carry a slot this rank actually owns. The rest are padding, and
``dispatch_b`` clamps their expert id into range::

    local_ids = (recv_ids - expert_offset).clamp(0, num_local_experts - 1)

so by the time the GEMM sees them they are indistinguishable from real work.
``num_tokens_post_padded`` cannot save them: that bound only skips the tail
each expert's run pads to ``BLOCK_M``, and these rows sit *inside* expert 0's
run as legitimate members of it.

The question this script answers is how much that is worth, because the answer
is not obviously "a lot". At decode the expert GEMM is weight-bound — it streams
``num_local_experts`` weight tiles whatever the row count — so rows that cost
nothing to add cost nothing to remove. The waste can only show up once there
are enough rows to make the GEMM compute-bound, and where that crossover sits
is the whole design question: it decides whether removing the padding is a
prefill optimisation or an everywhere optimisation.

Three arms, sharing one ``Work``:

* ``ep.padded`` — what the dispatcher does today: ``ep_size * rows * top_k``
  rows, every one bearing a valid local expert id.
* ``ep.exact`` — the floor: only the ``rows * top_k / ep_size`` rows this rank
  owns. Not implementable as-is (the exchange needs static shapes), but it
  bounds what any padding fix can win.
* ``tp.baseline`` — the arm EP has to beat: no exchange, every rank runs all
  ``rows * top_k`` slots against ``num_experts`` experts of half the
  intermediate width.

The shared denominator is the *useful* work, and the two parallelisms make it
identical by construction: EP runs ``E/ep`` experts at full width over
``rows*k/ep`` slots, TP runs ``E`` experts at ``inter/ep`` width over ``rows*k``
slots. Same FLOPs, same weight bytes. So a low TFLOP/s in a row is redundant
work and nothing else, which is exactly the claim under test.

Usage:
    python benchmarks/kernels/bench_ep_padding.py
    python benchmarks/kernels/bench_ep_padding.py --geometry deepseek-v2-lite
    python benchmarks/kernels/bench_ep_padding.py --ep-size 4 --rows 1,64,4096
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.kernels.microbench import Row, Work, bench, metadata, report
from rapid_llm.kernels import fused_moe

#: Tolerance for the correctness gate: bf16 grouped GEMM against a per-expert
#: torch reference, the tolerance the EP tests use for the same kernel.
_RTOL = 2e-2
_ATOL = 2e-2


@dataclass(frozen=True)
class Geometry:
    """A real MoE layer's routed-expert shape."""

    name: str
    num_experts: int
    top_k: int
    hidden: int
    inter: int


#: Both are shapes this repo has benchmarked end to end, so a number here can be
#: checked against an engine-level result rather than standing alone.
_GEOMETRIES = {
    "qwen3-30b-a3b": Geometry("qwen3-30b-a3b", num_experts=128, top_k=8, hidden=2048, inter=768),
    "deepseek-v2-lite": Geometry(
        "deepseek-v2-lite", num_experts=64, top_k=6, hidden=2048, inter=1408
    ),
}


def _weights(num_experts: int, inter: int, hidden: int, dtype, device):
    """Stacked expert tiles in the layout ``fused_moe`` expects."""
    gen = torch.Generator(device=device).manual_seed(0)
    w1 = torch.randn(num_experts, 2 * inter, hidden, generator=gen, device=device, dtype=dtype)
    w2 = torch.randn(num_experts, hidden, inter, generator=gen, device=device, dtype=dtype)
    return w1.mul_(0.02), w2.mul_(0.02)


def _routing(slots: int, top_k: int, num_experts: int, hidden: int, dtype, device):
    """``(x, weights, ids)`` covering ``slots`` routing slots over ``num_experts``.

    ``top_k`` chooses the call form: the EP arms use ``top_k=1`` (one row per
    slot, which is what ``dispatch_b`` hands back), the TP arm ``top_k=k``.
    """
    rows = max(slots // top_k, 1)
    gen = torch.Generator(device=device).manual_seed(1)
    x = torch.randn(rows, hidden, generator=gen, device=device, dtype=dtype).mul_(0.5)
    ids = torch.randint(0, num_experts, (rows, top_k), generator=gen, device=device)
    weights = torch.rand(rows, top_k, generator=gen, device=device, dtype=dtype)
    return x, weights, ids.to(torch.int32)


def _reference(x, w1, w2, weights, ids):
    """Per-expert torch loop — a different construction of the same operation."""
    rows, top_k = ids.shape
    out = torch.zeros(rows, x.shape[1], dtype=torch.float32, device=x.device)
    for slot in range(top_k):
        for expert in range(w1.shape[0]):
            hit = ids[:, slot] == expert
            if not bool(hit.any()):
                continue
            rows_e = x[hit].float()
            gate, up = (rows_e @ w1[expert].float().t()).chunk(2, dim=-1)
            hidden = torch.nn.functional.silu(gate) * up
            out[hit] += (hidden @ w2[expert].float().t()) * weights[hit, slot, None].float()
    return out


def _check(geometry: Geometry, ep_size: int, device) -> None:
    """Correctness before timing, at a size the reference loop can afford."""
    num_local = geometry.num_experts // ep_size
    w1, w2 = _weights(num_local, geometry.inter, geometry.hidden, torch.bfloat16, device)
    x, weights, ids = _routing(64, 1, num_local, geometry.hidden, torch.bfloat16, device)
    with torch.no_grad():
        got = fused_moe(x, w1, w2, weights, ids)
        want = _reference(x, w1, w2, weights, ids)
    torch.testing.assert_close(got.float(), want, rtol=_RTOL, atol=_ATOL)
    print(f"correctness  fused_moe vs per-expert loop  ok  (rtol={_RTOL}, atol={_ATOL})")


def _work(geometry: Geometry, rows: int, ep_size: int, dtype_bytes: int) -> Work:
    """The work the *operation* implies, identical for the EP and TP arms.

    ``useful`` counts the (token, expert) pairs this rank owns; redundant rows
    are excluded by construction, which is what makes a padded arm's TFLOP/s
    read as waste rather than as a different problem size.
    """
    useful = rows * geometry.top_k / ep_size
    # gate_up is [2*inter, hidden] and down is [hidden, inter]: three
    # inter*hidden tiles per expert, 2 FLOP per multiply-accumulate.
    flops = int(6 * useful * geometry.hidden * geometry.inter)
    weight_bytes = (geometry.num_experts // ep_size) * 3 * geometry.inter * geometry.hidden
    act_bytes = useful * geometry.hidden * 2  # read x, write out
    return Work(flops=flops, moved=int((weight_bytes + act_bytes) * dtype_bytes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=sorted(_GEOMETRIES), default="qwen3-30b-a3b")
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--rows", default="1,8,32,128,512,2048")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    geometry = _GEOMETRIES[args.geometry]
    ep = args.ep_size
    if geometry.num_experts % ep or geometry.inter % ep:
        raise SystemExit(f"{geometry.name} does not split {ep} ways")

    print(metadata())
    print(
        f"geometry  {geometry.name}: E={geometry.num_experts} top_k={geometry.top_k} "
        f"hidden={geometry.hidden} inter={geometry.inter}  ep_size={ep}  dtype={args.dtype}"
    )
    _check(geometry, ep, device)

    num_local = geometry.num_experts // ep
    ep_w1, ep_w2 = _weights(num_local, geometry.inter, geometry.hidden, dtype, device)
    tp_w1, tp_w2 = _weights(
        geometry.num_experts, geometry.inter // ep, geometry.hidden, dtype, device
    )

    out: list[Row] = []
    for rows in (int(r) for r in args.rows.split(",")):
        work = _work(geometry, rows, ep, torch.finfo(dtype).bits // 8)
        case = f"rows={rows}"
        for impl, (slots, top_k, experts, w1, w2) in {
            "ep.padded": (ep * rows * geometry.top_k, 1, num_local, ep_w1, ep_w2),
            "ep.exact": (rows * geometry.top_k // ep, 1, num_local, ep_w1, ep_w2),
            "tp.baseline": (
                rows * geometry.top_k,
                geometry.top_k,
                geometry.num_experts,
                tp_w1,
                tp_w2,
            ),
        }.items():
            x, w, ids = _routing(slots, top_k, experts, geometry.hidden, dtype, device)
            us = bench(lambda x=x, w=w, ids=ids, w1=w1, w2=w2: fused_moe(x, w1, w2, w, ids))
            out.append(Row(impl, case, us, work))

    report(out)

    by_case: dict[str, dict[str, float]] = {}
    for row in out:
        by_case.setdefault(row.case, {})[row.impl] = row.us
    print("\nwhat the padding costs, and whether EP wins the GEMM at all:")
    print(f"  {'case':<12}{'padded us':>11}{'exact us':>11}{'waste':>8}{'tp us':>10}{'ep/tp':>8}")
    for case, arms in by_case.items():
        print(
            f"  {case:<12}{arms['ep.padded']:>11.1f}{arms['ep.exact']:>11.1f}"
            f"{arms['ep.padded'] / arms['ep.exact']:>7.2f}x{arms['tp.baseline']:>10.1f}"
            f"{arms['tp.baseline'] / arms['ep.padded']:>7.2f}x"
        )
    print(
        "\n  waste = what removing the padding could win, at best.\n"
        "  ep/tp = ep.padded over the TP arm; below 1.00x means EP loses the\n"
        "          expert GEMM before any exchange cost is counted."
    )


if __name__ == "__main__":
    main()
