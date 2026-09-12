"""fp8 KV + chunked prefill 修复的引擎级精度实测。

四种配置, 子进程隔离运行 (extend 开关是进程级 env):

  fp8-chunked:  fp8 KV + chunked 路径 (修复后 fp8 请求的默认路径)
  fp8-extend:   fp8 KV + extend 路径 (修复前 fp8 请求被迫走的路径)
  bf16-chunked: bf16 KV + chunked 路径 (无量化对照)
  bf16-extend:  bf16 KV + extend 路径 (无量化对照)

每条 prompt 单独一批, greedy 生成 32 token, 记录每步 top-16 logprobs。
对比回答两个问题:

  1. fp8-chunked vs fp8-extend — 修复(即路径切换)是否改变数值语义
  2. fp8-chunked vs bf16-chunked — fp8 格式本身的精度代价 (与修复无关)

用法:
    python -m scripts.debug.fp8_kv_chunked_precision \
        --run fp8-chunked --out /tmp/prec_fp8_chunked.json
    python -m scripts.debug.fp8_kv_chunked_precision \
        --run fp8-extend --out /tmp/prec_fp8_extend.json
    python -m scripts.debug.fp8_kv_chunked_precision \
        --run bf16-chunked --out /tmp/prec_bf16_chunked.json
    python -m scripts.debug.fp8_kv_chunked_precision \
        --run bf16-extend --out /tmp/prec_bf16_extend.json
    python -m scripts.debug.fp8_kv_chunked_precision \
        --compare /tmp/prec_fp8_chunked.json /tmp/prec_fp8_extend.json \
                  /tmp/prec_bf16_chunked.json /tmp/prec_bf16_extend.json

结果与解读见 docs/quantization.md "分块预填充下的 fp8 KV cache 数值实测",
原始输出归档于 docs/benchmark_logs/quantization/fp8_kv_chunked_20260911/。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

DEFAULT_MODEL = "my_weight/Qwen2.5-1.5B-Instruct"

CONFIGS = {
    "fp8-chunked": {"kv": "fp8_e4m3", "extend": False},
    "fp8-extend": {"kv": "fp8_e4m3", "extend": True},
    "bf16-chunked": {"kv": "auto", "extend": False},
    "bf16-extend": {"kv": "auto", "extend": True},
}

# 四个长度档: short 单 chunk (两条路径同内核, sanity), medium 2~3 chunk,
# long 14 chunk (长前缀把 chunked/extend 的差异放到最大)。
PROMPTS = [
    ("short", "Explain what a GPU is in one sentence."),
    (
        "medium",
        (
            "Transformers changed natural language processing by replacing recurrence "
            "with attention. The mechanism lets every token attend to every other token "
            "in the context window. "
        )
        * 6,
    ),
    ("long-a", "The history of computing hardware is a long story. " * 350),
    ("long-b", "A compiler translates source code into machine instructions, and " * 300),
]

GEN_LEN = 32
TOP_K = 16


def run(config: str, out_path: str, model: str) -> None:
    cfg = CONFIGS[config]
    if cfg["extend"]:
        os.environ["RAPID_LLM_FUSED_CHUNK_PREFILL"] = "0"
    else:
        os.environ.pop("RAPID_LLM_FUSED_CHUNK_PREFILL", None)

    from rapid_llm.engine import continuous_engine as ce_mod
    from rapid_llm.engine.continuous_engine import ContinuousBatchingEngine
    from rapid_llm.engine.llm_engine import LLMEngine
    from rapid_llm.engine.sampler import SamplingParams
    from rapid_llm.engine.scheduler import SchedulerConfig

    # 记录本配置实际跑过的 pass 种类, 证明配置真的生效。
    pass_counts: dict[str, int] = {}
    _real_prefill_work = ce_mod._prefill_work

    def _traced(group, chunk_lens, chunked_min_rows=float("inf")):
        works = _real_prefill_work(group, chunk_lens, chunked_min_rows)
        for w in works:
            key = w.plan.kind.name
            pass_counts[key] = pass_counts.get(key, 0) + 1
        return works

    ce_mod._prefill_work = _traced

    engine = LLMEngine(model, max_seq_len=8192, kv_cache_dtype=cfg["kv"])
    ce = ContinuousBatchingEngine(
        engine,
        SchedulerConfig(
            max_seq_len=8192,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            max_chunk_size=256,
        ),
    )

    params = SamplingParams(temperature=0.0, max_gen_len=GEN_LEN, logprobs=TOP_K)
    results = []
    for name, text in PROMPTS:
        n_prompt = len(ce.tokenizer.encode(text, add_special_tokens=True))
        t0 = time.perf_counter()
        outs = ce.generate([text], params)
        dt = time.perf_counter() - t0
        gen = outs[0].outputs[0]
        steps = [
            {
                "tok": rec.token_id,
                "lp": rec.logprob,
                "top_ids": list(rec.top_token_ids),
                "top_lps": list(rec.top_logprobs),
            }
            for rec in (gen.logprobs or [])
        ]
        results.append(
            {
                "name": name,
                "n_prompt_tokens": n_prompt,
                "gen_time_s": round(dt, 3),
                "tokens": [s["tok"] for s in steps],
                "steps": steps,
            }
        )
        print(f"  [{config}] {name}: {n_prompt} prompt tok -> {len(steps)} gen tok in {dt:.2f}s")

    ce.shutdown()
    payload = {
        "config": config,
        "kv_dtype": cfg["kv"],
        "extend_forced": cfg["extend"],
        "pass_counts": pass_counts,
        "prompts": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"saved -> {out_path}  passes={pass_counts}")


def _first_diff(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def _step_delta(sa: dict, sb: dict) -> tuple[float, float]:
    """(top-1 logprob 差, 共同 token 内最大 logprob 差) at one position."""
    d_top1 = abs(sa["lp"] - sb["lp"])
    common = set(sa["top_ids"]) & set(sb["top_ids"])
    d_common = max(
        (
            abs(sa["top_lps"][sa["top_ids"].index(t)] - sb["top_lps"][sb["top_ids"].index(t)])
            for t in common
        ),
        default=0.0,
    )
    return d_top1, d_common


def compare(paths: list[str]) -> None:
    runs = []
    for p in paths:
        with open(p) as f:
            runs.append(json.load(f))

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ra, rb = runs[i], runs[j]
            print(
                f"\n=== {ra['config']} (passes={ra['pass_counts']}) "
                f"vs {rb['config']} (passes={rb['pass_counts']}) ==="
            )
            for pa, pb in zip(ra["prompts"], rb["prompts"], strict=True):
                assert pa["name"] == pb["name"]
                na = pa["n_prompt_tokens"]
                diff = _first_diff(pa["tokens"], pb["tokens"])
                if diff is None:
                    tok_note = f"{GEN_LEN}/{GEN_LEN} 一致"
                else:
                    match = diff
                    tok_note = f"{match}/{GEN_LEN} 一致, 首差 @ 位置 {diff}"
                    print(f"    tok[{diff}]: {pa['tokens'][diff]} vs {pb['tokens'][diff]}")

                # 位置级 logprob 差: 只看两配置 token 仍一致的前缀段。
                stop = diff if diff is not None else max(len(pa["steps"]), len(pb["steps"]))
                stop = min(stop, len(pa["steps"]), len(pb["steps"]))
                d_top1 = d_common = 0.0
                for k in range(stop):
                    a, b = _step_delta(pa["steps"][k], pb["steps"][k])
                    d_top1 = max(d_top1, a)
                    d_common = max(d_common, b)

                # 首步 (prefill 后) 的分布差异单独看: 内核差异的直接信号。
                s0a, s0b = pa["steps"][0], pb["steps"][0]
                overlap = len(set(s0a["top_ids"]) & set(s0b["top_ids"])) / TOP_K
                d0_top1, d0_common = _step_delta(s0a, s0b)

                print(
                    f"  {pa['name']:<8} ({na} tok): {tok_note} | "
                    f"一致段 maxΔtop1={d_top1:.4f} maxΔcommon={d_common:.4f} | "
                    f"首步 top16 重叠={overlap:.2f} Δtop1={d0_top1:.4f} Δcommon={d0_common:.4f}"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=sorted(CONFIGS))
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model directory (relative to the repo root by default)",
    )
    args = ap.parse_args()
    if args.run:
        run(args.run, args.out or f"/tmp/prec_{args.run}.json", args.model)
    elif args.compare:
        compare(args.compare)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
