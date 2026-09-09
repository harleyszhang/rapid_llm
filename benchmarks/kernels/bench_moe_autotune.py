"""Compare measured MoE grouped-GEMM tiles with the shipped heuristic.

``native/fused_moe`` already dequantises fp8/int8 expert tiles *inside* the GEMM
mainloop (``dequant_fp8e4m3`` on the loaded k-tile), so there is no separate fp16
weight materialisation. On sm86 the missing autotune collect round left an empty store,
``_launch_config`` falls back to ``_TILE_TABLE_PRE_HOPPER``, a conservative table
the sweep never measured (H100-only), so every A10 MoE GEMM ran on a guess.

This A/B times the same grouped GEMMs two ways on the same bytes:

* ``tuned``     -- the autotune store populated by ``bench_fused_moe.py --tune``;
* ``heuristic`` -- ``RAPID_LLM_AUTOTUNE=0``, i.e. the PRE_HOPPER table
                   a machine without a collect round gets.

Speedup > 1 means the collect round bought real time over the shipped guess.
fp8 W8A8 is absent on purpose: Triton cannot emit ``tl.float8e4nv`` below sm89, so
that row does not exist on A10 (see ``active_schemes``).

Switch under test: ``RAPID_LLM_AUTOTUNE`` = ``1`` (store) | ``0`` (heuristic).

Usage:
    python benchmarks/kernels/bench_moe_autotune.py \
        --json docs/benchmark_logs/kernels/moe_autotune_<stamp>.json
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_fused_moe import (
    BUILTIN_GEOMETRIES,
    MoeGeometry,
    _build_bf16,
    _build_fp8,
    _build_int8,
    routing,
)
from microbench import bench, metadata, require_cuda

import rapid_llm.kernels
from rapid_llm.kernels.ops.moe.fused_moe import _launch_config
from rapid_llm.kernels.ops.tile_policy import resolve_tiles

#: The three formats A10 actually serves (no native-fp8 W8A8 row on sm86).
ARMS: tuple[tuple[str, object, str], ...] = (
    ("bf16", _build_bf16, "bf16"),
    ("fp8_w8a16", _build_fp8, "fp8"),
    ("int8_w8a16", _build_int8, "int8"),
)
TOKENS: tuple[int, ...] = (1, 8, 64, 512, 4096)


def _tile_for(tokens: int, geo: MoeGeometry, dtype_label: str, top_k: int) -> dict:
    """The config one arm would launch, resolved exactly as the kernel resolves it."""
    return resolve_tiles(
        "fused_moe",
        m=tokens,
        n=2 * geo.intermediate,
        k=geo.hidden,
        dtype_label=dtype_label,
        heuristic=lambda dev: _launch_config(
            tokens, 0, tokens * top_k / geo.num_experts, dev
        ),
        device_index=torch.cuda.current_device(),
    )


def _set_autotune(on: bool) -> None:
    if on:
        os.environ.pop("RAPID_LLM_AUTOTUNE", None)
    else:
        os.environ["RAPID_LLM_AUTOTUNE"] = "0"


def main() -> int:
    require_cuda()
    torch.set_grad_enabled(False)

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="evidence JSON path")
    args = ap.parse_args()

    print(metadata())
    print()
    geo = BUILTIN_GEOMETRIES[0]
    header = (
        f"{'scheme':<12} {'tokens':>6} | {'heur_us':>9} {'heur_tile':>10} | "
        f"{'tuned_us':>9} {'tuned_tile':>10} | {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, build, dtype_label in ARMS:
        torch.manual_seed(0)
        w1 = (
            torch.randn(
                geo.num_experts, 2 * geo.intermediate, geo.hidden,
                device="cuda", dtype=torch.float32,
            )
            / geo.hidden**0.5
        )
        w2 = (
            torch.randn(geo.num_experts, geo.hidden, geo.intermediate,
                        device="cuda", dtype=torch.float32)
            / geo.intermediate**0.5
        )
        call, _r1, _r2, _sb = build(w1, w2)
        del w1, w2
        torch.cuda.empty_cache()

        for tokens in TOKENS:
            x = torch.randn(tokens, geo.hidden, device="cuda", dtype=torch.bfloat16)
            weights, ids, _ = routing(tokens, geo)

            _set_autotune(False)
            heur_tile = _tile_for(tokens, geo, dtype_label, geo.top_k)
            invoke = partial(call, x, weights, ids)
            heur_us = bench(invoke)
            _set_autotune(True)
            tuned_tile = _tile_for(tokens, geo, dtype_label, geo.top_k)
            tuned_us = bench(invoke)
            _set_autotune(False)

            speedup = heur_us / tuned_us if tuned_us else float("nan")
            ht = "x".join(str(heur_tile[f"BLOCK_{d}"]) for d in "MN")
            tt = "x".join(str(tuned_tile[f"BLOCK_{d}"]) for d in "MN")
            print(
                f"{label:<12} {tokens:>6} | {heur_us:>9.1f} {ht:>10} | "
                f"{tuned_us:>9.1f} {tt:>10} | {speedup:>7.3f}x"
            )
            rows.append(
                {
                    "scheme": label,
                    "tokens": tokens,
                    "heuristic_us": round(heur_us, 2),
                    "heuristic_tile": ht,
                    "tuned_us": round(tuned_us, 2),
                    "tuned_tile": tt,
                    "tile_changed": heur_tile != tuned_tile,
                    "speedup": round(speedup, 4),
                }
            )
            del x, weights, ids
        del call
        torch.cuda.empty_cache()

    geo_mean = math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows))
    changed = [r for r in rows if r["tile_changed"]]
    best = max(rows, key=lambda r: r["speedup"])
    print(
        f"\ngeomean speedup={geo_mean:.3f}x over {len(rows)} cells  "
        f"({len(changed)} cells changed tile)  best {best['scheme']} t{best['tokens']}"
        f"={best['speedup']:.3f}x"
    )
    print(
        "Read as: speedup > 1 means the measured tile beats the\n"
        "unmeasured PRE_HOPPER guess. Cells whose tile did not change are 1.0x by\n"
        "construction (same kernel, same config); their spread is run-to-run noise."
    )

    if args.json:
        from rapid_llm.benchmark import write_json_log

        write_json_log(
            args.json,
            {
                "optimization": "MoE grouped-GEMM autotune (A10 tiles vs PRE_HOPPER heuristic)",
                "switch": "RAPID_LLM_AUTOTUNE=1(store)|0(heuristic)",
                "kernel": "native/fused_moe",
                "inference_mode": "offline kernel microbenchmark (no serving queue)",
                "command": " ".join(sys.argv),
                "geometry": {
                    "label": geo.label,
                    "hidden": geo.hidden,
                    "intermediate": geo.intermediate,
                    "num_experts": geo.num_experts,
                    "top_k": geo.top_k,
                },
                "note": "fp8 dequant is already fused in the GEMM mainloop; this A/B isolates the collect round",
            },
            {"geomean_speedup": round(geo_mean, 4), "rows": rows},
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
