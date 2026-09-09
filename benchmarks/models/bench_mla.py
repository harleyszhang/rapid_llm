"""MLA's first report: KV economics from the config, latency from the GPU.

Two things this script refuses to do. It refuses to compare V2-Lite's 16B
parameters against a "same-size" dense model — no such checkpoint is on the
shelf — so the KV columns come from parsing each model's own config.json, and
the latency columns come from one identical workload on both. And it refuses
to call TP-split what is actually TP-replicated: the latent cache is
single-KV-head, so TP shards the queries but every rank carries the whole
latent — a per-rank pool IS the full pool, and every table that touches KV
capacity says so.

Usage:
    python benchmarks/models/bench_mla.py --batch 8 --max-gen-len 128 \
        --json docs/benchmark_logs/kernels/mla_v0.11.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.benchmark import (
    BenchResult,
    describe_footprint,
    expand_prompts,
    make_backend,
    peak_mem_gb,
    print_table,
    require_gpus,
    reset_peak_mem,
    write_json_log,
)

MLA_CKPT = "my_weight/DeepSeek-V2-Lite"
MHA_CKPT = "my_weight/Qwen3-1.7B"


def kv_geometry(model_dir: str) -> dict:
    """Per-token KV footprint straight from the checkpoint's config.

    Three vocabularies, because the models genuinely differ: MLA stores one
    latent row (``kv_lora_rank + qk_rope_head_dim``), the same architecture
    without compression would store the full per-head K and V, and a GQA model
    stores ``2 * num_key_value_heads * head_dim``. Numbers are elements per
    token per layer; bytes multiply by the KV dtype width, read from the
    checkpoint's own torch_dtype (the cache keeps checkpoint precision unless
    the operator pins a narrower one).
    """
    config = json.loads((Path(model_dir) / "config.json").read_text())
    layers = config["num_hidden_layers"]
    dtype = config.get("torch_dtype", "bfloat16")
    bytes_per_element = 1 if dtype == "float8" else 2
    if "kv_lora_rank" in config:
        latent = config["kv_lora_rank"] + config["qk_rope_head_dim"]
        uncompressed = config["num_key_value_heads"] * (
            config["qk_nope_head_dim"] + config["qk_rope_head_dim"] + config["v_head_dim"]
        )
        return {
            "model_dir": model_dir,
            "kind": "mla",
            "layers": layers,
            "kv_dtype": dtype,
            "latent_elements_per_token_per_layer": latent,
            "uncompressed_elements_per_token_per_layer": uncompressed,
            "latent_bytes_per_token": latent * layers * bytes_per_element,
            "uncompressed_bytes_per_token": uncompressed * layers * bytes_per_element,
        }
    per_layer = 2 * config["num_key_value_heads"] * config["head_dim"]
    return {
        "model_dir": model_dir,
        "kind": "gqa",
        "layers": layers,
        "kv_dtype": dtype,
        "gqa_elements_per_token_per_layer": per_layer,
        "gqa_bytes_per_token": per_layer * layers * bytes_per_element,
    }


def print_geometry(mla: dict, mha: dict) -> None:
    latent = mla["latent_elements_per_token_per_layer"]
    uncompressed = mla["uncompressed_elements_per_token_per_layer"]
    gqa = mha["gqa_elements_per_token_per_layer"]
    print("== KV geometry, parsed from each config.json (elements per token per layer) ==")
    print(f"  DeepSeek-V2-Lite  MLA latent      {latent:5d}   ({mla['layers']} layers)")
    print(
        f"  DeepSeek-V2-Lite  if uncompressed {uncompressed:5d}   -> latent is "
        f"{uncompressed / latent:.1f}x smaller"
    )
    print(f"  Qwen3-1.7B        GQA             {gqa:5d}   ({mha['layers']} layers)")
    print(
        f"  per-token totals: V2-Lite latent {mla['latent_bytes_per_token'] / 1024:.1f} KiB, "
        f"uncompressed {mla['uncompressed_bytes_per_token'] / 1024:.1f} KiB, "
        f"Qwen3-1.7B {mha['gqa_bytes_per_token'] / 1024:.1f} KiB"
    )
    print()


def measure(
    label: str,
    model_dir: str,
    *,
    tensor_parallel_size: int,
    prompts: list[str],
    max_gen_len: int,
    replicas: int,
) -> dict:
    """One backend end to end: latency, footprint, and rank-0 peak memory."""
    print(f"-- {label}: tp={tensor_parallel_size}, building engine ...")
    reset_peak_mem()
    backend = make_backend(
        model_dir, tensor_parallel_size=tensor_parallel_size, use_cuda_graph=False
    )
    result: BenchResult = backend.measure(prompts, max_gen_len, greedy=True)
    weight_gib, kv_tokens = describe_footprint(backend.runner, replicas=replicas)
    peak = peak_mem_gb()
    backend.close()

    # TP>1 and latent KV: the pool is replicated, not sharded — a per-rank
    # capacity of kv_tokens is the whole-model capacity, so replicas is not
    # applied to it. Weight bytes are shards and do multiply.
    row = {
        "model": label,
        **result.as_dict(),
        "weight_gib": round(weight_gib, 2),
        "kv_pool_tokens": kv_tokens,
        "peak_mem_gib_rank0": round(peak, 2),
    }
    print(
        f"{label:20s} {result.row('')}\n"
        f"{'':20s} weights {weight_gib:.2f} GiB | KV pool {kv_tokens} tokens | "
        f"peak {peak:.2f} GiB (rank 0 process)"
    )
    print()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-gen-len", type=int, default=128)
    ap.add_argument("--mla-dir", type=str, default=MLA_CKPT)
    ap.add_argument("--mha-dir", type=str, default=MHA_CKPT)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    require_gpus(2)  # the MLA side of the report is a TP=2 replica

    mla_geo, mha_geo = kv_geometry(args.mla_dir), kv_geometry(args.mha_dir)
    print_geometry(mla_geo, mha_geo)

    prompts = expand_prompts(
        [
            "Explain what a KV cache is and why it dominates serving memory.",
            "Write a short story about a lighthouse keeper who counts ships.",
            "Summarize the trade-offs between MoE and dense transformers.",
            "Describe how latent attention compresses key-value pairs.",
        ],
        args.batch,
    )

    results = {}
    results["mla_tp2"] = measure(
        "DeepSeek-V2-Lite TP=2",
        args.mla_dir,
        tensor_parallel_size=2,
        prompts=prompts,
        max_gen_len=args.max_gen_len,
        replicas=2,
    )
    results["mha_tp1"] = measure(
        "Qwen3-1.7B TP=1",
        args.mha_dir,
        tensor_parallel_size=1,
        prompts=prompts,
        max_gen_len=args.max_gen_len,
        replicas=1,
    )

    print("== Same workload, both engines (decode paths differ: TP=2 runs eager) ==")
    print_table(
        {
            "DeepSeek-V2-Lite TP=2": BenchResult(
                **{
                    k: v
                    for k, v in results["mla_tp2"].items()
                    if k in BenchResult.__dataclass_fields__
                }
            ),
            "Qwen3-1.7B TP=1": BenchResult(
                **{
                    k: v
                    for k, v in results["mha_tp1"].items()
                    if k in BenchResult.__dataclass_fields__
                }
            ),
        }
    )

    if args.json:
        write_json_log(
            args.json,
            {**vars(args), "geometry": {"mla": mla_geo, "mha": mha_geo}},
            results,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
