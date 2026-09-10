"""DeepSeek trimmed-checkpoint accuracy: reference implementations vs rapid_llm.

Two cut-down checkpoints serve as the accuracy targets — small enough to run
eagerly, same module structure and quantization paths as the full models:

* **V3-4layers** (bf16, 13 GiB): one A10 holds one implementation at a time,
  so the transformers reference runs first, leaves its logits and greedy
  tokens on the CPU, and frees the GPU before rapid_llm loads. The vLLM
  vote runs from the vLLM source tree's venv (third subcommand); pairwise
  greedy agreement across the three stacks pins residual divergence on bf16
  numerics rather than structure.

* **V4-Flash-6layers** (DSpark keys, fp8 linears, MXFP4 experts): vLLM
  cannot serve it on these A10s — the Lightning Indexer and FlashMLA
  attention require DeepGEMM, SM90+ only
  (``vllm/platforms/cuda.py::support_deep_gemm``) — so the reference is
  transformers 5.15's eager DeepseekV4ForCausalLM on CPU, its weights filled
  from the DSpark files by :mod:`tests.layer.dspark_to_hf`. The
  rapid_llm side consumes the DSpark storage directly (TP-2, fp8/mxfp4)
  over the tests' tp harness, so the two sides meet through their JSONs.

Both targets run 32 greedy steps — through each framework's own incremental
path (transformers' DynamicCache against rapid_llm's paged KV metadata) —
and every step records its top-5 logprobs, the comparison record the
agreement subcommands consume.

    rapid_llm venv, single GPU:
        python -m tests.layer.deepseek v3 parity
    vLLM source tree's venv:
        /path/to/vllm-venv/python -m tests.layer.deepseek v3 vllm
    rapid_llm venv, GPUs (TP-2) / CPU:
        python -m tests.layer.deepseek v4 lite
        python -m tests.layer.deepseek v4 hf
    analyses (either venv):
        python -m tests.layer.deepseek v3 three-way PARITY.json VLLM.json
        python -m tests.layer.deepseek v4 compare LITE.json HF.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GREEDY_STEPS = 32
LOG_DIR = Path(__file__).resolve().parents[2] / "docs" / "benchmark_logs" / "accuracy"

V3_CKPT = "/data/shared/llm_weights/DeepSeek-V3-4layers-MTP-BF16"
V4_CKPT = "/data/shared/llm_weights/DeepSeek-V4-Flash-6layers"

# --------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------- #


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def top5_logprobs(logits_1d: torch.Tensor) -> list[list[float]]:
    """[[id, logprob] * 5] for one vocab row, fp32."""
    logp = torch.log_softmax(logits_1d.float(), dim=-1)
    vals, ids = logp.topk(5)
    return [[int(i), float(v)] for i, v in zip(ids.tolist(), vals.tolist(), strict=True)]


def hf_greedy(model, input_ids: torch.Tensor, n_steps: int = GREEDY_STEPS):
    """Greedy continuation through transformers' DynamicCache.

    Returns ``(tokens, per-step top5)`` — the deterministic record both
    reference sides leave behind for the agreement subcommands.
    """
    with torch.no_grad():
        step = model(input_ids, use_cache=True)
        cache = step.past_key_values
        logits = step.logits[:, -1]
        tokens = [int(logits.argmax(-1))]
        top5s = [top5_logprobs(logits[0])]
        for _ in range(n_steps - 1):
            step = model(logits.argmax(-1, keepdim=True), past_key_values=cache, use_cache=True)
            cache = step.past_key_values
            logits = step.logits[:, -1]
            tokens.append(int(logits.argmax(-1)))
            top5s.append(top5_logprobs(logits[0]))
    return tokens, top5s


def write_arm(prefix: str, config: dict, results: dict) -> Path:
    from rapid_llm.benchmark import timestamped_log_path, write_json_log

    path = timestamped_log_path(LOG_DIR, f"accuracy_{prefix}")
    write_json_log(path, config, results)
    return path


def read_prompts(path: str) -> dict[int, dict]:
    """seq_len -> prompt entry, from either JSON shape (pre/post unification)."""
    data = json.loads(Path(path).read_text())
    prompts = (data.get("results") or data)["prompts"]
    return {p["seq_len"] if "seq_len" in p else p["prefill"]["seq_len"]: p for p in prompts}


def agreement(a: list[int], b: list[int]) -> tuple[float, int]:
    """(fraction of matching steps, first divergent index or -1)."""
    first = -1
    same = 0
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x == y:
            same += 1
        elif first == -1:
            first = i
    return same / len(a), first


# --------------------------------------------------------------------- #
# V3-4layers: transformers / lite / vLLM three-way
# --------------------------------------------------------------------- #

#: The trimmed checkpoint keeps V3's full-size routing fields (8 experts over
#: 8 groups — one expert per group, so the grouped noaux_tc path degenerates).
#: The same regroup the benchmark section of the docs validated as the golden
#: gate restores the grouped semantics; every side must run it.
V3_HF_OVERRIDES = {"n_group": 2, "topk_group": 1, "num_experts_per_tok": 2}

#: Real text at three lengths; the long one is a deterministic repetition of
#: the mid prompt so the parity claims cover a 1024-token prefill too.
V3_PROMPTS = [
    "Explain what a GPU tensor core is and why it matters for deep learning.",
    (
        "The memory hierarchy of a modern accelerator has several levels. "
        "Registers are the fastest but smallest storage, followed by shared "
        "memory, which is programmable and shared across a thread block. "
        "L2 cache sits between the streaming multiprocessors and device "
        "memory, absorbing repeated reads and writes. High-bandwidth memory "
        "attached to the die feeds the compute units with terabytes per "
        "second of bandwidth, while host memory over PCIe is orders of "
        "magnitude slower and should only carry cold data. Kernel design is "
        "therefore a scheduling problem: keep data resident in the lowest "
        "level possible, reuse it as much as possible, and overlap memory "
        "transfers with math so neither resource idles."
    ),
    None,  # filled at runtime as the mid prompt repeated 4x
]


def _v3_prompts() -> list[str]:
    prompts = list(V3_PROMPTS)
    prompts[2] = prompts[2] or prompts[1] * 4
    return prompts


def run_v3_reference(prompt_ids: list[torch.Tensor]) -> dict:
    """transformers side: prefill logits and greedy tokens left on the CPU."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(V3_CKPT)
    for field, value in V3_HF_OVERRIDES.items():
        setattr(config, field, value)
    model = AutoModelForCausalLM.from_pretrained(
        V3_CKPT, config=config, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    out = {"prefill_logits": [], "greedy": []}
    for ids in prompt_ids:
        input_ids = ids.cuda().unsqueeze(0)
        with torch.no_grad():
            logits = model(input_ids).logits[0]  # [seq, vocab] bf16
        out["prefill_logits"].append(logits.float().cpu())
        tokens, top5s = hf_greedy(model, input_ids)
        out["greedy"].append({"tokens": tokens, "top5": top5s})

    del model
    _free()
    return out


def run_v3_lite(prompt_ids: list[torch.Tensor], reference: dict) -> dict:
    """rapid_llm side: loads the same checkpoint, compares on the fly."""
    from rapid_llm.executor.attention_metadata import AttentionMetadata
    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.config import ModelConfig
    from rapid_llm.models.registry import ModelRegistry

    config = ModelConfig.from_pretrained(V3_CKPT, max_seq_len=2048, hf_overrides=V3_HF_OVERRIDES)
    assert config.quant is None, "the V3-4layers checkpoint is plain bf16"

    model = ModelRegistry.resolve("deepseek_v3").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(V3_CKPT, "cuda", dequantize_fp8=False))
    model.to("cuda").eval()

    results = []
    # MLA caches the compressed latent (kv_lora_rank + rope dim) as a single
    # wide head per row; one pre-allocated pool per layer serves prefill and
    # every decode step of this prompt.
    cache_dim = config.kv_lora_rank + config.qk_rope_head_dim
    for ids, ref_logits, ref_greedy in zip(
        prompt_ids, reference["prefill_logits"], reference["greedy"], strict=True
    ):
        seq_len = ids.numel()
        kv = [
            torch.zeros(seq_len + GREEDY_STEPS, 1, cache_dim, dtype=config.dtype, device="cuda")
            for _ in range(config.num_layers)
        ]
        meta = AttentionMetadata(
            kv_buffer=kv,
            cur_select_index=torch.arange(seq_len, dtype=torch.int32, device="cuda"),
            b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
            b_seq_len=torch.tensor([seq_len], dtype=torch.int32, device="cuda"),
            max_actual_seq_len=seq_len,
        )
        pos = torch.arange(seq_len, device="cuda").unsqueeze(0)
        with torch.no_grad():
            logits = model(ids.cuda().unsqueeze(0), pos, meta)[0]  # [seq, vocab]

        ref = ref_logits.cuda()
        diff = (logits.float() - ref).abs()
        lite_top5 = logits.topk(5, dim=-1).indices.cpu()
        ref_top5 = ref.topk(5, dim=-1).indices.cpu()
        prefill = {
            "seq_len": seq_len,
            "logits_std": ref.float().std().item(),
            "max_abs_diff": diff.max().item(),
            "mean_abs_diff": diff.mean().item(),
            "top1_agree": (logits.argmax(-1).cpu() == ref.argmax(-1).cpu()).float().mean().item(),
            "top5_agree": (
                (lite_top5.unsqueeze(-1) == ref_top5.unsqueeze(1)).any(-1).float().mean().item()
            ),
        }

        # Greedy continuation through the paged-cache decode path.
        tokens = []
        with torch.no_grad():
            nxt = logits[-1].argmax().reshape(1, 1).cuda()
            tokens.append(int(nxt))
            for step in range(GREEDY_STEPS - 1):
                # Paged decode over the flat latent pool: page_size is 1, so the
                # page table lists the cache rows this request may read.
                cached = seq_len + step + 1
                meta = AttentionMetadata(
                    kv_buffer=kv,
                    cur_select_index=torch.tensor(
                        [seq_len + step], dtype=torch.int32, device="cuda"
                    ),
                    b_req_tokens_table=torch.arange(cached, dtype=torch.int32, device="cuda").view(
                        1, -1
                    ),
                    b_req_idx=torch.tensor([0], dtype=torch.int32, device="cuda"),
                    b_start_loc=torch.zeros(1, dtype=torch.int32, device="cuda"),
                    b_seq_len=torch.tensor([cached], dtype=torch.int32, device="cuda"),
                    max_actual_seq_len=cached,
                )
                meta.is_prefill = False  # default is True; a decode step must flip it
                pos = torch.full((1, 1), seq_len + step, device="cuda")
                logits = model(nxt, pos, meta)[:, -1]  # [1, vocab]
                nxt = logits.argmax(-1, keepdim=True)
                tokens.append(int(nxt))

        agree, diverge_at = agreement(tokens, ref_greedy["tokens"])
        results.append(
            {
                "prefill": prefill,
                "greedy": {
                    "agreement": agree,
                    "first_divergence": diverge_at,
                    "lite_tokens": tokens,
                    "ref_tokens": ref_greedy["tokens"],
                },
            }
        )

    del model
    _free()
    return {"prompts": results}


def cmd_v3_parity(args) -> int:
    from rapid_llm.benchmark import require_gpus

    require_gpus(1)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(V3_CKPT)
    prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0] for p in _v3_prompts()]
    print("prompt lengths:", [t.numel() for t in prompt_ids])

    t0 = time.time()
    reference = run_v3_reference(prompt_ids)
    print(f"transformers side done in {time.time() - t0:.1f}s")

    t0 = time.time()
    results = run_v3_lite(prompt_ids, reference)
    print(f"rapid_llm side done in {time.time() - t0:.1f}s")

    for p in results["prompts"]:
        pre, greedy = p["prefill"], p["greedy"]
        print(
            f"seq {pre['seq_len']:>5}: max {pre['max_abs_diff']:.4f} "
            f"mean {pre['mean_abs_diff']:.5f} (std {pre['logits_std']:.2f}) "
            f"top1 {pre['top1_agree']:.3f} top5 {pre['top5_agree']:.3f} | "
            f"greedy {greedy['agreement']:.2f} first-div {greedy['first_divergence']}"
        )

    write_arm("v3_parity", {"checkpoint": V3_CKPT, "greedy_steps": GREEDY_STEPS}, results)
    return 0


def cmd_v3_vllm(args) -> int:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(V3_CKPT)
    prompt_ids = [tokenizer(p, return_tensors="pt").input_ids[0].tolist() for p in _v3_prompts()]
    print("prompt lengths:", [len(x) for x in prompt_ids])

    llm = LLM(
        model=V3_CKPT,
        # single card: the bf16 checkpoint is 13 GiB, TP-1 keeps this side
        # comparable with the single-GPU transformers/lite runs
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        hf_overrides=V3_HF_OVERRIDES,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=GREEDY_STEPS, logprobs=5)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids], sp)

    payload = []
    for ids, o in zip(prompt_ids, outs, strict=True):
        gen = o.outputs[0]
        steps = []
        for lp in gen.logprobs:  # list[dict[id -> Logprob]]
            top5 = sorted(lp.items(), key=lambda kv: -kv[1].logprob)[:5]
            steps.append({"top5": [[int(i), float(l.logprob)] for i, l in top5]})
        payload.append(
            {
                "seq_len": len(ids),
                "greedy_tokens": list(gen.token_ids),
                "steps": steps,
            }
        )

    write_arm(
        "v3_vllm", {"checkpoint": V3_CKPT, "hf_overrides": V3_HF_OVERRIDES}, {"prompts": payload}
    )
    for p in payload:
        print(
            f"seq {p['seq_len']:>5}: greedy[:8] {p['greedy_tokens'][:8]} "
            f"| step0 top5 {[i for i, _ in p['steps'][0]['top5']]}"
        )
    return 0


def cmd_v3_three_way(args) -> int:
    lite_ref = read_prompts(args.parity)
    by_seq = read_prompts(args.vllm)

    print(
        f"{'seq':>5} | {'prefill top1':>12} | {'lite~hf':>14} | {'lite~vllm':>14} | {'vllm~hf':>14}"
    )
    print("-" * 72)
    for seq_len in sorted(lite_ref):
        p = lite_ref[seq_len]["greedy"]
        lite, ref = p["lite_tokens"], p["ref_tokens"]
        vv = by_seq[seq_len]["greedy_tokens"]
        rows = [agreement(lite, ref), agreement(lite, vv), agreement(vv, ref)]
        cells = [f"{a:.3f} @{f}" if f >= 0 else f"{a:.3f} all" for a, f in rows]
        print(
            f"{seq_len:>5} | {lite_ref[seq_len]['prefill']['top1_agree']:>12.3f} | "
            f"{cells[0]:>14} | {cells[1]:>14} | {cells[2]:>14}"
        )
    print("\n(cells: greedy agreement over 32 steps, @ = first divergent step)")
    return 0


# --------------------------------------------------------------------- #
# V4-Flash: fp32 CPU oracle vs rapid_llm TP-2
# --------------------------------------------------------------------- #

V4_PREFILL_LENS = [64, 256, 1024]
V4_SEED = 0


def _v4_prompt_ids(length: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(V4_SEED + length)
    return torch.randint(10, 120000, (1, length), generator=g)


def _v4_lite_payload(rank: int) -> dict:
    """Module-level so tp_harness's spawned workers can pickle it."""
    from rapid_llm.distributed.parallel_state import tensor_model_parallel_all_gather
    from rapid_llm.executor.attention_metadata import AttentionMetadata
    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.config import ModelConfig
    from rapid_llm.models.registry import ModelRegistry

    config = ModelConfig.from_pretrained(V4_CKPT, max_seq_len=2048)
    model = ModelRegistry.resolve("deepseek_v4").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(
        hf_weights_iterator(V4_CKPT, "cuda", dequantize_fp8=False, dequant_dtype=config.dtype)
    )
    model.to("cuda").eval()

    def forward(ids: torch.Tensor, pos: torch.Tensor, meta) -> torch.Tensor:
        """Full-vocab logprob row of one step: gather the TP-sharded head."""
        with torch.no_grad():
            logits = model(ids, pos, meta)[:, -1]
        if logits.shape[-1] == config.vocab_size:
            return logits[0]
        return tensor_model_parallel_all_gather(logits[0].contiguous())

    out = []
    for length in V4_PREFILL_LENS:
        ids = _v4_prompt_ids(length).cuda()
        steps, tokens = [], []
        # V4 attention carries its own window/compressor state; a prefill
        # step resets it inside the model's forward.
        meta = AttentionMetadata()
        meta.is_prefill = True
        meta.b_seq_len = torch.full((1,), length, dtype=torch.long)
        full = forward(ids, torch.arange(length, device="cuda").unsqueeze(0), meta)
        tokens.append(int(full.argmax()))
        steps.append({"top5": top5_logprobs(full)})
        for step in range(GREEDY_STEPS - 1):
            meta = AttentionMetadata()
            meta.is_prefill = False
            meta.b_seq_len = torch.full((1,), length + step + 1, dtype=torch.long)
            pos = torch.full((1, 1), length + step, device="cuda")
            full = forward(torch.tensor([[tokens[-1]]], device="cuda"), pos, meta)
            tokens.append(int(full.argmax()))
            steps.append({"top5": top5_logprobs(full)})
        out.append({"seq_len": length, "greedy_tokens": tokens, "steps": steps})
    return {"prompts": out}


def cmd_v4_lite(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "distributed"))
    from tp_harness import needs_gpus, run_on_tp_ranks

    @needs_gpus(2)
    def run() -> None:
        results = run_on_tp_ranks(_v4_lite_payload, tp_size=2)
        write_arm("v4_lite", {"checkpoint": V4_CKPT}, {"prompts": results[0]["prompts"]})

    run()
    return 0


def cmd_v4_hf(args) -> int:
    """Reference side: transformers eager V4 on CPU, DSpark weights converted."""
    import time

    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from rapid_llm.models.config import ModelConfig
    from tests.layer.dspark_to_hf import load_dspark_hf

    config = ModelConfig.from_pretrained(V4_CKPT, max_seq_len=2048).hf_config
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(config)
    t0 = time.time()
    load_dspark_hf(model, V4_CKPT, dtype=torch.float32)
    model.eval()
    print(f"dspark->hf conversion took {time.time() - t0:.0f}s")

    torch.set_num_threads(min(32, torch.get_num_threads()))
    out = []
    for length in V4_PREFILL_LENS:
        t1 = time.time()
        tokens, top5s = hf_greedy(model, _v4_prompt_ids(length))
        print(f"  seq {length}: {GREEDY_STEPS} greedy steps in {time.time() - t1:.0f}s")
        out.append(
            {"seq_len": length, "greedy_tokens": tokens, "steps": [{"top5": t} for t in top5s]}
        )
    write_arm("v4_hf", {"checkpoint": V4_CKPT}, {"prompts": out})
    return 0


def cmd_v4_compare(args) -> int:
    lite, hf = read_prompts(args.lite), read_prompts(args.hf)
    for seq_len in sorted(lite):
        lp, hp = lite[seq_len], hf[seq_len]
        toks_a, toks_b = lp["greedy_tokens"], hp["greedy_tokens"]
        divs = [i for i, (a, b) in enumerate(zip(toks_a, toks_b, strict=True)) if a != b]
        top5_agree, drift = 0.0, []
        for sa, sb in zip(lp["steps"], hp["steps"], strict=True):
            top5_agree += len({i for i, _ in sa["top5"]} & {i for i, _ in sb["top5"]}) / 5
            d, e = dict(sa["top5"]), dict(sb["top5"])
            drift.extend(abs(d[i] - e[i]) for i in d.keys() & e.keys())
        n = len(toks_a)
        print(
            f"seq {seq_len:>5}: greedy {n - len(divs)}/{n} first-div "
            f"{divs[0] if divs else -1} | top5-id-agree {top5_agree / n:.3f} "
            f"| shared-logprob drift max {max(drift, default=0):.4f} "
            f"mean {sum(drift) / len(drift) if drift else 0:.5f}"
        )
        if divs:
            i = divs[0]
            print(f"  lite {toks_a[i : i + 4]} vs hf {toks_b[i : i + 4]}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="family", required=True)

    v3 = sub.add_parser("v3", help="V3-4layers: transformers / lite / vLLM three-way")
    v3_sub = v3.add_subparsers(dest="mode", required=True)
    v3_sub.add_parser("parity", help="transformers vs rapid_llm over the real checkpoint")
    v3_sub.add_parser("vllm", help="the vLLM vote (run from the vLLM venv)")
    three = v3_sub.add_parser("three-way", help="pairwise greedy agreement from two JSONs")
    three.add_argument("parity", help="parity JSON (from the v3 parity subcommand)")
    three.add_argument("vllm", help="vLLM JSON (from the v3 vllm subcommand)")

    v4 = sub.add_parser("v4", help="V4-Flash: fp32 CPU oracle vs lite TP-2")
    v4_sub = v4.add_subparsers(dest="mode", required=True)
    v4_sub.add_parser("lite", help="rapid_llm side, TP-2 over the tp harness")
    v4_sub.add_parser("hf", help="transformers reference side, DSpark weights on CPU")
    compare = v4_sub.add_parser("compare", help="greedy agreement and top-5 drift from two JSONs")
    compare.add_argument("lite", help="lite JSON (from the v4 lite subcommand)")
    compare.add_argument("hf", help="HF JSON (from the v4 hf subcommand)")

    args = parser.parse_args()
    return {
        ("v3", "parity"): cmd_v3_parity,
        ("v3", "vllm"): cmd_v3_vllm,
        ("v3", "three-way"): cmd_v3_three_way,
        ("v4", "lite"): cmd_v4_lite,
        ("v4", "hf"): cmd_v4_hf,
        ("v4", "compare"): cmd_v4_compare,
    }[args.family, args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
