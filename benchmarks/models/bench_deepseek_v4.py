"""DeepSeek V4 trimmed checkpoint: rapid_llm vs transformers, forward speed.

V4 ships no public weights, so both arms load the *same* randomly
initialised trimmed checkpoint built once from transformers 5.8's
``DeepseekV4ForCausalLM``. The reference runs its own eager model;
rapid_llm runs its model-runner API with the V4 caches. The trim keeps
every structural variant (all three attention types, both router
families) at a size one A10 fits in bf16.

Measured as device time (CUDA events, median): prefill latency at three
prompt lengths and decode TPOT at three batch sizes. Two parity blocks
ride along in the JSON so the speed table cannot silently drift from
the numerical alignment the M6 tests pin — greedy token agreement plus
an fp32-vs-bf16 noise floor from the reference itself, because an
untrained checkpoint's flat logits make greedy sensitive to precision
alone.

Usage:
    python benchmarks/models/bench_deepseek_v4.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapid_llm.benchmark import require_gpus, timestamped_log_path, write_json_log

PREFILL_BATCH = 2
PREFILL_SEQS = [256, 1024, 2048]
DECODE_BATCHES = [1, 8, 32]
DECODE_PROMPT = 128
DECODE_STEPS = 16
GREEDY_STEPS = 16

_WARMUP = 3
_RUNS = 10
_DECODE_WARMUP = 1
_DECODE_RUNS = 5

#: Trimmed V4 at bench size — hidden 512 over 12 layers (four passes of the
#: three attention types), 8 routed experts with the first two layers on the
#: hash router. Every V4 mechanism stays in; nothing here is structural dead
#: weight for a speed table.
CONFIG = {
    "model_type": "deepseek_v4",
    "vocab_size": 4096,
    "hidden_size": 512,
    "moe_intermediate_size": 512,
    "num_hidden_layers": 12,
    "layer_types": [
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    * 4,
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 8},
    "mlp_layer_types": ["hash_moe", "hash_moe"] + ["moe"] * 10,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "q_lora_rank": 128,
    "o_groups": 4,
    "o_lora_rank": 64,
    "partial_rotary_factor": 0.5,
    "sliding_window": 128,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 4,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "routed_scaling_factor": 1.5,
    "scoring_func": "sqrtsoftplus",
    "index_n_heads": 2,
    # Stays >= head_dim * partial_rotary_factor (64): the indexer heads take
    # the trailing rope slice, so a smaller table cannot reach its channels.
    "index_head_dim": 128,
    "index_topk": 4,
    "swiglu_limit": 7.0,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}


def _build_pair(workdir: Path):
    """One checkpoint, both runtimes: HF reference init, rapid_llm loads it."""
    from safetensors.torch import save_file
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    from rapid_llm.executor.loader import materialise_parameters
    from rapid_llm.executor.weight_utils import hf_weights_iterator
    from rapid_llm.models.config import ModelConfig
    from rapid_llm.models.registry import ModelRegistry

    (workdir / "config.json").write_text(json.dumps(CONFIG))
    config = ModelConfig.from_pretrained(workdir, max_seq_len=4096)
    torch.manual_seed(0)
    hf_model = DeepseekV4ForCausalLM(config.hf_config).eval()
    state = {key: value.detach().clone() for key, value in hf_model.state_dict().items()}
    save_file(state, str(workdir / "model.safetensors"), metadata={"format": "pt"})
    hf_model = hf_model.to(config.dtype).cuda()

    model = ModelRegistry.resolve("deepseek_v4").load_class()(config)
    materialise_parameters(model, "cuda", dtype=config.dtype)
    model.load_weights(hf_weights_iterator(workdir, dequant_dtype=config.dtype))
    model.cuda()
    return hf_model, model.eval(), config


def _lite_meta(batch: int, seq_len: int, *, prefill: bool):
    """Minimal attention metadata, as the M6 tests drive the model."""
    from rapid_llm.executor.attention_metadata import AttentionMetadata

    meta = AttentionMetadata()
    meta.is_prefill = prefill
    meta.b_seq_len = torch.full((batch,), seq_len, dtype=torch.long)
    return meta


def _median_ms(fn, warmup: int, runs: int) -> float:
    """Median device milliseconds of ``fn`` across ``runs`` timed calls."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _bench_prefill(hf_model, model, config, seq_len: int) -> dict:
    """Both arms re-run the same prompt from cold caches; median latency."""
    from transformers.cache_utils import DynamicCache

    batch = PREFILL_BATCH
    ids = torch.randint(0, CONFIG["vocab_size"], (batch, seq_len), device="cuda")
    pos = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch, -1).contiguous()
    meta = _lite_meta(batch, seq_len, prefill=True)

    def lite():
        model.reset_v4_caches()
        return model(ids, pos, meta)

    def hf():
        cache = DynamicCache(config=config.hf_config)
        return hf_model(input_ids=ids, position_ids=pos, past_key_values=cache).logits

    lite_ms = _median_ms(lite, _WARMUP, _RUNS)
    hf_ms = _median_ms(hf, _WARMUP, _RUNS)

    with torch.no_grad():
        lite_logits, hf_logits = lite().float(), hf().float()
    return {
        "lite_ms": round(lite_ms, 3),
        "hf_ms": round(hf_ms, 3),
        "speedup": round(hf_ms / lite_ms, 3),
        "logits_maxdiff": round((lite_logits - hf_logits).abs().max().item(), 4),
    }


def _bench_decode(hf_model, model, config, batch: int) -> dict:
    """Per-arm TPOT: untimed prefill from cold caches, then timed decode steps.

    The timed span is the whole decode loop on device events, so both arms
    carry their own CPU submission overhead exactly as a serving loop
    would — the reference's Python-side cache plumbing is part of its
    cost, not an artifact to hide.
    """
    from transformers.cache_utils import DynamicCache

    prompt_len = DECODE_PROMPT
    ids = torch.randint(0, CONFIG["vocab_size"], (batch, prompt_len), device="cuda")
    pos_p = (
        torch.arange(prompt_len, device="cuda").unsqueeze(0).expand(batch, -1).contiguous()
    )
    meta_p = _lite_meta(batch, prompt_len, prefill=True)

    def lite_round() -> float:
        model.reset_v4_caches()
        logits = model(ids, pos_p, meta_p)
        tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for step in range(DECODE_STEPS):
            pos_d = torch.full((batch, 1), prompt_len + step, device="cuda")
            logits = model(tok, pos_d, _lite_meta(batch, 1, prefill=False))
            tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / DECODE_STEPS

    def hf_round() -> float:
        cache = DynamicCache(config=config.hf_config)
        out = hf_model(input_ids=ids, position_ids=pos_p, past_key_values=cache)
        tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for step in range(DECODE_STEPS):
            pos_d = torch.full((batch, 1), prompt_len + step, device="cuda")
            out = hf_model(input_ids=tok, position_ids=pos_d, past_key_values=cache)
            tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / DECODE_STEPS

    lite_tpot = _median_ms(lite_round, _DECODE_WARMUP, _DECODE_RUNS)
    hf_tpot = _median_ms(hf_round, _DECODE_WARMUP, _DECODE_RUNS)
    return {
        "lite_tpot_ms": round(lite_tpot, 3),
        "hf_tpot_ms": round(hf_tpot, 3),
        "speedup": round(hf_tpot / lite_tpot, 3),
    }


def _hf_greedy(hf_model, config, ids: torch.Tensor, pos_p: torch.Tensor, steps: int) -> torch.Tensor:
    """Greedy generation with a fresh cache; shared by parity and noise floor."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=config.hf_config)
    out = hf_model(input_ids=ids, position_ids=pos_p, past_key_values=cache)
    toks = [out.logits[:, -1, :].argmax(dim=-1, keepdim=True)]
    for step in range(steps - 1):
        pos_d = torch.full((ids.shape[0], 1), pos_p.shape[1] + step, device="cuda")
        out = hf_model(input_ids=toks[-1], position_ids=pos_d, past_key_values=cache)
        toks.append(out.logits[:, -1, :].argmax(dim=-1, keepdim=True))
    return torch.cat(toks, dim=1)


def _lite_greedy(model, ids: torch.Tensor, pos_p: torch.Tensor, steps: int) -> torch.Tensor:
    """Greedy generation through the model-runner API with V4 caches."""
    batch, prompt_len = ids.shape
    model.reset_v4_caches()
    logits = model(ids, pos_p, _lite_meta(batch, prompt_len, prefill=True))
    toks = [logits[:, -1, :].argmax(dim=-1, keepdim=True)]
    for step in range(steps - 1):
        pos_d = torch.full((batch, 1), prompt_len + step, device="cuda")
        logits = model(toks[-1], pos_d, _lite_meta(batch, 1, prefill=False))
        toks.append(logits[:, -1, :].argmax(dim=-1, keepdim=True))
    return torch.cat(toks, dim=1)


def _parity_suite(hf_model, model, config) -> dict:
    """Greedy agreement from one shared prompt: lite vs HF, and the floor.

    Three runs, one prompt: rapid_llm vs the HF bf16 reference (the parity
    number), and the reference's own fp32 weights against their bf16 cast
    (the noise floor). An untrained checkpoint has flat logit margins, so
    bf16 alone flips a large share of greedy ties — the parity number is
    read against that floor, never against 100%.

    The fp32 copy re-runs the checkpoint's own seed-0 init, so it is the
    same weights the bench loaded, before the bf16 cast; its divergence
    from the reference is pure precision with no implementation in the
    delta.
    """
    from transformers.models.deepseek_v4 import DeepseekV4ForCausalLM

    torch.manual_seed(0)
    fp32_model = DeepseekV4ForCausalLM(config.hf_config).cuda().eval()

    batch, prompt_len = 2, 128
    torch.manual_seed(123)  # fixed prompt, independent of the weight seeds
    ids = torch.randint(0, CONFIG["vocab_size"], (batch, prompt_len), device="cuda")
    pos_p = (
        torch.arange(prompt_len, device="cuda").unsqueeze(0).expand(batch, -1).contiguous()
    )

    with torch.no_grad():
        t32 = _hf_greedy(fp32_model, config, ids, pos_p, GREEDY_STEPS)
        t16 = _hf_greedy(hf_model, config, ids, pos_p, GREEDY_STEPS)
        tl = _lite_greedy(model, ids, pos_p, GREEDY_STEPS)
    del fp32_model

    return {
        "lite_vs_hf_bf16": {
            "greedy_agreement": round((tl == t16).float().mean().item(), 4),
            "steps": GREEDY_STEPS,
        },
        "hf_fp32_vs_hf_bf16": {
            "greedy_agreement": round((t32 == t16).float().mean().item(), 4),
            "steps": GREEDY_STEPS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="default: docs/benchmark_logs/models/deepseek_v4_<stamp>.json",
    )
    args = parser.parse_args()
    if args.json is None:
        args.json = timestamped_log_path(
            Path(__file__).resolve().parents[2] / "docs" / "benchmark_logs" / "models", "deepseek_v4"
        )

    require_gpus(1)
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(
        f"model: trimmed V4 — hidden {CONFIG['hidden_size']}, "
        f"{CONFIG['num_hidden_layers']} layers, {CONFIG['n_routed_experts']} experts, bf16"
    )

    workdir = Path(tempfile.mkdtemp(prefix="v4_bench_"))
    hf_model, model, config = _build_pair(workdir)

    results: dict = {"prefill": {}, "decode": {}}
    with torch.no_grad():
        print(f"\nprefill (batch {PREFILL_BATCH}, cold caches):")
        for seq_len in PREFILL_SEQS:
            entry = _bench_prefill(hf_model, model, config, seq_len)
            results["prefill"][f"s{seq_len}"] = entry
            print(
                f"  seq {seq_len:5d}: lite {entry['lite_ms']:8.2f} ms | "
                f"hf {entry['hf_ms']:8.2f} ms | {entry['speedup']:.2f}x | "
                f"logits maxdiff {entry['logits_maxdiff']}"
            )

        print(f"\ndecode (prompt {DECODE_PROMPT}, {DECODE_STEPS} steps):")
        for batch in DECODE_BATCHES:
            entry = _bench_decode(hf_model, model, config, batch)
            results["decode"][f"b{batch}"] = entry
            print(
                f"  batch {batch:3d}: lite TPOT {entry['lite_tpot_ms']:7.2f} ms | "
                f"hf TPOT {entry['hf_tpot_ms']:7.2f} ms | {entry['speedup']:.2f}x"
            )

        parity = _parity_suite(hf_model, model, config)
        results["greedy_parity"] = parity
        steps = parity["lite_vs_hf_bf16"]["steps"]
        print(f"\ngreedy parity (one shared prompt, {steps} steps):")
        print(
            f"  lite vs hf-bf16:   {parity['lite_vs_hf_bf16']['greedy_agreement']:.1%} token agreement"
        )
        print(
            f"  hf-fp32 vs hf-bf16: {parity['hf_fp32_vs_hf_bf16']['greedy_agreement']:.1%} — "
            "the noise floor: precision alone, no implementation in the delta"
        )

    write_json_log(
        args.json,
        {
            "model": "deepseek_v4 trimmed",
            "config": CONFIG,
            "dtype": str(config.dtype),
            "device": torch.cuda.get_device_name(0),
            "timing": {
                "prefill": {"batch": PREFILL_BATCH, "warmup": _WARMUP, "runs": _RUNS},
                "decode": {"prompt": DECODE_PROMPT, "steps": DECODE_STEPS, "warmup": _DECODE_WARMUP, "runs": _DECODE_RUNS},
            },
            "known_limitations": {
                "greedy_parity": (
                    "lite_vs_hf_bf16 is read against hf_fp32_vs_hf_bf16 in the "
                    "same block, not against 100%: an untrained checkpoint has "
                    "flat logit margins, so bf16 alone flips a large share of "
                    "ties (same weights, no implementation in the delta). The "
                    "module-level parity and the 100% small-config e2e greedy "
                    "in tests/models/test_deepseek_v4.py remain the numerical "
                    "verification; this block is the speed bench's rider"
                ),
                "decode_speed": (
                    "lite decode is CPU-bound: the compressor/indexer walk the "
                    "batch row-by-row in Python, so one batch-32 step issues "
                    "~8.7k kernel launches plus ~0.8k boolean-index syncs for "
                    "~22 ms of GPU work (torch.profiler: 170 ms CPU vs 22 ms "
                    "CUDA). Vectorising the compressor is deferred — this is "
                    "the parity-first implementation"
                ),
            },
        },
        results,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
